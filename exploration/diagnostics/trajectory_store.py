"""Lossless, process-written numeric trajectory chunks and historical JSONL reading.

Each append is one batch control step. Rows include case_id and attempt_id, so
one shared store can serve independently finishing cases. Call flush before
publishing a case summary. Only chunks named in index.json are committed.
"""
import copy
import json
import multiprocessing as mp
from pathlib import Path
import queue
import traceback
import os

import numpy as np

FORMAT = 'numeric_trajectory_chunks_v1'


def _write_json(path, value):
    temporary = path.with_suffix('.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class _Columns:
    def __init__(self, directory):
        self.directory = directory
        self.number = 0
        self.columns = {}

    def save(self, values, dtype):
        name = f'{self.number:05d}.npy'
        self.number += 1
        self.columns[name] = np.asarray(values, dtype=dtype)
        return name

    def publish(self):
        # One durable numeric pool avoids a filesystem transaction per small field.
        size = sum(array.nbytes for array in self.columns.values())
        pool = np.lib.format.open_memmap(self.directory/'columns.npy', mode='w+', dtype=np.uint8, shape=(size,))
        index, offset = {}, 0
        for name, array in self.columns.items():
            flat = np.ascontiguousarray(array).view(np.uint8).reshape(-1)
            pool[offset:offset+array.nbytes] = flat
            index[name] = dict(dtype=array.dtype.str, shape=list(array.shape), offset=offset, bytes=array.nbytes)
            offset += array.nbytes
        pool.flush()
        del pool
        with (self.directory/'columns.npy').open('rb') as stream:
            os.fsync(stream.fileno())
        _write_json(self.directory/'column_index.json', index)
        self.columns.clear()

    def strings(self, values):
        encoded = [v.encode('utf-8', errors='surrogatepass') for v in values]
        offsets = np.cumsum([0]+[len(v) for v in encoded], dtype=np.int64)
        return dict(data=self.save(np.frombuffer(b''.join(encoded), dtype=np.uint8), np.uint8),
                    offsets=self.save(offsets, np.int64))

    @staticmethod
    def numeric_layout(value):
        kind = type(value)
        if kind in (bool, float):
            return (), kind
        if kind is int:
            return ((), int) if -(2**63) <= value < 2**63 else None
        if kind is not list:
            return None
        if not value:
            return (0,), None
        children = [_Columns.numeric_layout(item) for item in value]
        if any(child is None for child in children):
            return None
        shape, leaf = children[0]
        if any(child != (shape, leaf) for child in children[1:]):
            return None
        return (len(value), *shape), leaf

    def dictionary(self, values):
        keys = dict.fromkeys(key for value in values for key in value)
        if any(type(key) is not str for key in keys):
            raise TypeError('Trajectory dictionary keys must be strings')
        fields = {}
        for key in keys:
            positions = [i for i, value in enumerate(values) if key in value]
            field = dict(values=self.encode([values[i][key] for i in positions]))
            if len(positions) != len(values):
                field['positions'] = self.save(positions, np.int64)
            fields[key] = field
        return dict(length=len(values), fields=fields)

    def encode(self, values):
        # Dense fields retain their exact Python scalar type; mixed types use tags.
        types = {type(value) for value in values}
        if not values or types == {type(None)}:
            return dict(null=len(values))
        if types <= {bool, int, float, list} and len(types) == 1:
            layout = self.numeric_layout(values)
            if layout is not None:
                leaf = layout[1]
                dtype = {bool: np.bool_, int: np.int64, float: np.float64, None: np.float64}[leaf]
                return dict(dense=self.save(values, dtype))
        if types == {str}:
            return dict(strings=self.strings(values))
        if types == {dict}:
            return dict(dictionary=self.dictionary(values))
        if types == {list}:
            return dict(sequence=dict(offsets=self.save(np.cumsum([0]+[len(v) for v in values]), np.int64),
                items=self.encode([item for value in values for item in value])))
        # Type tags preserve bool/int/float and heterogeneous event fields exactly.
        kinds = {type(None): 0, bool: 1, int: 2, float: 3, str: 4, list: 5, dict: 6}
        tags = []
        for value in values:
            if type(value) not in kinds:
                raise TypeError(f'Trajectory values must be JSON types, got {type(value).__name__}')
            tags.append(kinds[type(value)])
        node = dict(tags=self.save(tags, np.uint8), groups={})
        for kind in sorted(set(tags)):
            selected = [v for v, tag in zip(values, tags) if tag == kind]
            if kind == 0:
                child = {}
            elif kind in (1, 3):
                child = dict(data=self.save(selected, np.bool_ if kind == 1 else np.float64))
            elif kind == 2:
                if all(-(2**63) <= v < 2**63 for v in selected):
                    child = dict(data=self.save(selected, np.int64))
                else:
                    child = dict(decimal=self.strings([str(v) for v in selected]))
            elif kind == 4:
                child = self.strings(selected)
            elif kind == 5:
                child = dict(offsets=self.save(np.cumsum([0]+[len(v) for v in selected]), np.int64),
                             items=self.encode([item for value in selected for item in value]))
            else:
                child = self.dictionary(selected)
            node['groups'][str(kind)] = child
        return node


def _decode(directory, node, _pool=None):
    if _pool is None and (directory/'column_index.json').is_file():
        _pool = (np.load(directory/'columns.npy', mmap_mode='r', allow_pickle=False),
                 json.loads((directory/'column_index.json').read_text()))

    def array(name):
        if _pool is None:
            return np.load(directory/name, allow_pickle=False)
        pool, index = _pool
        column = index[name]
        data = pool[column['offset']:column['offset']+column['bytes']]
        return data.view(np.dtype(column['dtype'])).reshape(column['shape'])

    def strings(child):
        data, offsets = array(child['data']).tobytes(), array(child['offsets'])
        return [data[a:b].decode('utf-8', errors='surrogatepass') for a, b in zip(offsets[:-1], offsets[1:])]

    if 'dense' in node:
        return array(node['dense']).tolist()
    if 'null' in node:
        return [None]*node['null']
    if 'strings' in node:
        return strings(node['strings'])
    if 'sequence' in node:
        child = node['sequence']
        offsets = array(child['offsets'])
        items = _decode(directory, child['items'], _pool)
        return [items[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
    if 'dictionary' in node:
        child = node['dictionary']
        values = [{} for _ in range(child['length'])]
        for key, field in child['fields'].items():
            positions = array(field['positions']) if 'positions' in field else range(child['length'])
            for i, value in zip(positions, _decode(directory, field['values'], _pool)):
                values[i][key] = value
        return values
    tags = array(node['tags'])
    result = [None]*len(tags)
    for label, child in node['groups'].items():
        kind = int(label)
        positions = np.flatnonzero(tags == kind)
        if kind == 0:
            values = [None]*len(positions)
        elif kind in (1, 2, 3):
            values = ([int(v) for v in strings(child['decimal'])] if 'decimal' in child
                      else array(child['data']).tolist())
        elif kind == 4:
            values = strings(child)
        elif kind == 5:
            offsets = array(child['offsets'])
            items = _decode(directory, child['items'], _pool)
            values = [items[a:b] for a, b in zip(offsets[:-1], offsets[1:])]
        elif kind == 6:
            values = [{} for _ in positions]
            for key, field in child['fields'].items():
                field_positions = array(field['positions']) if 'positions' in field else range(len(positions))
                for i, value in zip(field_positions, _decode(directory, field['values'], _pool)):
                    values[i][key] = value
        else:
            raise ValueError(f'Unknown trajectory type tag: {kind}')
        for i, value in zip(positions, values):
            result[i] = value
    return result


def _load_index(directory):
    path = directory/'index.json'
    index = json.loads(path.read_text()) if path.exists() else dict(format=FORMAT, chunks=[])
    if index.get('format') != FORMAT:
        raise ValueError(f'Unsupported trajectory format: {path}')
    return index


def _writer_main(directory, commands, acknowledgements, chunk_size):
    try:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        index = _load_index(directory)
        existing = [int(p.name.split('_')[1].split('.')[0]) for p in directory.glob('chunk_*')
                    if p.name.split('_')[1].split('.')[0].isdigit()]
        number = max(existing, default=-1)+1
        rows, batches = [], 0

        def publish():
            nonlocal number, rows, batches
            if not rows:
                batches = 0
                return
            name = f'chunk_{number:08d}'
            temporary, final = directory/(name+'.partial'), directory/name
            temporary.mkdir(exist_ok=False)
            columns = _Columns(temporary)
            schema = columns.encode(rows)
            columns.publish()
            _write_json(temporary/'schema.json', schema)
            _sync_directory(temporary)
            os.rename(temporary, final)
            entry = dict(directory=name, rows=len(rows), batches=batches,
                         routes=[list(route) for route in dict.fromkeys(
                             (row['case_id'], row['attempt_id']) for row in rows)])
            index['chunks'].append(entry)
            _write_json(directory/'index.json', index)
            _sync_directory(directory)
            number += 1
            rows, batches = [], 0

        # An empty store is readable even when no steps are collected.
        if not (directory/'index.json').exists():
            _write_json(directory/'index.json', index)
        acknowledgements.put(('ready', None))
        while True:
            kind, payload = commands.get()
            if kind == 'append':
                for row in payload:
                    if type(row) is not dict or type(row.get('case_id')) is not int or type(row.get('attempt_id')) is not int:
                        raise ValueError('Each transition requires integer case_id and attempt_id routing metadata')
                rows.extend(payload)
                batches += 1
                if batches >= chunk_size:
                    publish()
            elif kind in ('flush', 'close'):
                publish()
                acknowledgements.put((kind, payload))
                if kind == 'close':
                    return
            else:
                raise ValueError(f'Unknown trajectory command: {kind}')
    except BaseException:
        acknowledgements.put(('error', traceback.format_exc()))


class TrajectoryWriter:
    """One bounded writer process; append owns a copy and blocks under pressure."""
    def __init__(self, output_root, chunk_size=256, queue_size=4):
        if chunk_size <= 0 or queue_size <= 0:
            raise ValueError('chunk_size and queue_size must be positive')
        context = mp.get_context('spawn')
        self.directory = Path(output_root)/'trajectory_chunks'
        self._commands = context.Queue(maxsize=queue_size)
        self._acks = context.Queue()
        self._closed = False
        self._failure = None
        self._token = 0
        self._process = context.Process(target=_writer_main,
            args=(str(self.directory), self._commands, self._acks, chunk_size), daemon=True)
        self._process.start()
        try:
            self._wait('ready', None)
        except BaseException:
            self._cleanup()
            raise

    def _check(self):
        if self._failure:
            raise RuntimeError(self._failure)
        if not self._process.is_alive():
            # Queue delivery can lag process exit very briefly.
            try:
                kind, message = self._acks.get(timeout=.1)
            except queue.Empty:
                kind, message = 'error', f'Writer exited with code {self._process.exitcode}'
            if kind == 'error':
                self._failure = f'Trajectory writer failed: {message}'
                raise RuntimeError(self._failure)
            raise RuntimeError(f'Trajectory writer exited unexpectedly ({kind})')

    def _send(self, kind, payload):
        if self._closed:
            raise RuntimeError('Trajectory writer is closed')
        while True:
            self._check()
            try:
                self._commands.put((kind, payload), timeout=.1)
                return
            except queue.Full:
                pass

    def _wait(self, expected, token):
        while True:
            try:
                kind, value = self._acks.get(timeout=.1)
            except queue.Empty:
                if self._process.is_alive():
                    continue
                try:
                    kind, value = self._acks.get(timeout=.1)
                except queue.Empty:
                    self._check()
                    continue
            if kind == 'error':
                self._failure = f'Trajectory writer failed: {value}'
                raise RuntimeError(self._failure)
            if (kind, value) != (expected, token):
                raise RuntimeError(f'Unexpected trajectory acknowledgement: {kind}')
            return

    def append(self, rows):
        self._send('append', copy.deepcopy(list(rows)))

    def flush(self):
        self._token += 1
        self._send('flush', self._token)
        self._wait('flush', self._token)

    def _cleanup(self):
        if self._process.is_alive():
            self._process.terminate()
        self._process.join()
        self._commands.cancel_join_thread()
        self._commands.close()
        self._acks.close()
        self._closed = True

    def close(self):
        if self._closed:
            return
        try:
            self._token += 1
            self._send('close', self._token)
            self._wait('close', self._token)
            self._process.join()
        finally:
            self._cleanup()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()


def transition_source(attempt_dir):
    """Return the authoritative source path, retaining legacy JSONL support."""
    attempt_dir = Path(attempt_dir)
    historical = attempt_dir/'transitions.jsonl'
    if historical.is_file():
        return historical
    for parent in (attempt_dir, *attempt_dir.parents):
        path = parent/'trajectory_chunks'/'index.json'
        if path.is_file():
            return path
    raise FileNotFoundError(f'No committed transitions for {attempt_dir}')


def read_transitions(attempt_dir):
    """Reconstruct complete logical rows in recorded order for one attempt."""
    attempt_dir = Path(attempt_dir)
    source = transition_source(attempt_dir)
    if source.name == 'transitions.jsonl':
        return [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    directory = source.parent
    route = None
    if (attempt_dir/'attempt.json').is_file():
        metadata = json.loads((attempt_dir/'attempt.json').read_text())
        route = [metadata['case']['case_id'], metadata['attempt']]
    elif attempt_dir.name.startswith('attempt_') and attempt_dir.parent.name.startswith('case_'):
        route = [int(attempt_dir.parent.name[5:]), int(attempt_dir.name[8:])]
    result = []
    for chunk in _load_index(directory)['chunks']:
        if route is not None and route not in chunk['routes']:
            continue
        path = directory/chunk['directory']
        rows = _decode(path, json.loads((path/'schema.json').read_text()))
        if len(rows) != chunk['rows']:
            raise ValueError(f'Trajectory chunk row count mismatch: {path}')
        result.extend(row for row in rows if route is None or [row['case_id'], row['attempt_id']] == route)
    return result
