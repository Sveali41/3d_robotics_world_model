"""Lossless boundary-frame replay; privileged diagnostics never enter batches."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from domain.robocasa.primitives import RAPS_SPLIT_ACTION_SCHEMA


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(action):
    schema = RAPS_SPLIT_ACTION_SCHEMA
    value = np.asarray(action, dtype=np.float32)
    if value.shape != (schema.action_dim,) or not np.isfinite(value).all():
        raise ValueError("Expected a finite 13D split7 action")
    if (np.abs(value) > 1).any():
        raise ValueError("Split7 action exceeds normalized bounds")
    index = int(value[:schema.num_primitives].argmax())
    result = np.zeros_like(value)
    result[index] = 1
    used = np.asarray(schema.primitive_name_to_action_idx[schema.primitive_names[index]], dtype=int)
    result[schema.num_primitives + used] = value[schema.num_primitives + used]
    return result


class VisualReplay:
    def __init__(self, directory, *, validation_every=5, rows=()):
        self.directory = Path(directory).resolve()
        if validation_every < 2:
            raise ValueError("validation_every must be >=2")
        self.validation_every = int(validation_every)
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "frames").mkdir(exist_ok=True)
        self.rows = list(rows)
        self.frame_count = 0
        self.episode_id = 0
        self.current_frame = None
        self.history_frames = []
        self.history_actions = []
        self.index = self.directory / "visual_replay.jsonl"
        if self.index.exists():
            raise FileExistsError(self.index)
        with self.index.open("w") as stream:
            for row in self.rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")

    def _frame(self, rgb):
        rgb = np.asarray(rgb)
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[-1] != 3:
            raise ValueError("Camera must return uint8 RGB [height,width,3]")
        path = self.directory / "frames" / f"{self.frame_count:09d}.npy"
        with path.open("xb") as stream:
            np.save(stream, rgb, allow_pickle=False)
        self.frame_count += 1
        return dict(path=str(path), sha256=sha256(path))

    @staticmethod
    def read_frame(reference):
        value = np.load(reference["path"], allow_pickle=False)
        if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError("Invalid replay RGB frame")
        return value

    def reset(self, episode_id, rgb):
        self.episode_id = int(episode_id)
        self.current_frame = self._frame(rgb)
        self.history_frames = [self.current_frame]
        self.history_actions = []

    def append(self, skill, rgb):
        if int(skill["episode_id"]) != self.episode_id or self.current_frame is None:
            raise ValueError("Visual replay episode does not match collector")
        following = self._frame(rgb)
        action = canonical(skill["action"])
        duration = int(skill["duration_low_steps"])
        complete = bool(skill["completed"]) and duration > 0
        row = dict(episode_id=self.episode_id, skill_id=int(skill["skill_id"]),
                   split="validation" if self.episode_id % self.validation_every == 0 else "train",
                   requested_action=np.asarray(skill["requested_action"]).tolist(),
                   action=action.tolist(), start=self.current_frame, end=following,
                   duration_low_steps=duration, completed=complete, status=str(skill["status"]),
                   terminated=bool(skill["terminated"]), truncated=bool(skill["truncated"]))
        self.rows.append(row)
        with self.index.open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
        self.current_frame = following
        if complete:
            self.history_actions.append(action)
            self.history_frames.append(following)
        else:
            self.history_frames = [following]
            self.history_actions = []
        return row

    @staticmethod
    def _context(frames, actions, size):
        available = min(size, len(frames))
        pad = size - available
        selected = frames[-available:]
        pixels = [VisualReplay.read_frame(selected[0])] * pad
        pixels += [VisualReplay.read_frame(frame) for frame in selected]
        mask = np.r_[np.zeros(pad, bool), np.ones(available, bool)]
        aligned_actions = np.zeros((size - 1, 13), np.float32)
        if available > 1:
            aligned_actions[-(available - 1):] = np.asarray(actions[-(available - 1):])
        return np.stack(pixels), aligned_actions, mask

    def context(self, history_size=3):
        pixels, actions, mask = self._context(self.history_frames, self.history_actions, history_size)
        return dict(pixels=torch.from_numpy(pixels.transpose(0, 3, 1, 2))[None],
                    history_actions=torch.from_numpy(actions)[None],
                    history_mask=torch.from_numpy(mask)[None])

    def windows(self, split, history_size=3, horizon=3):
        if split not in {"train", "validation"}:
            raise ValueError("Unknown visual replay split")
        groups, chain = [], []
        for row in self.rows:
            if not row["completed"] or row["split"] != split:
                if chain:
                    groups.append(chain)
                chain = []
                continue
            if chain:
                last = chain[-1]
                if (row["episode_id"] != last["episode_id"] or
                        row["skill_id"] != last["skill_id"] + 1 or
                        row["start"] != last["end"] or last["terminated"] or last["truncated"]):
                    groups.append(chain)
                    chain = []
            chain.append(row)
        if chain:
            groups.append(chain)
        return [(group, start, history_size, horizon) for group in groups
                for start in range(len(group) - horizon + 1)]

    def batch(self, windows):
        samples = []
        for group, start, history_size, horizon in windows:
            frames = [group[0]["start"]] + [row["end"] for row in group[:start]]
            past = [row["action"] for row in group[:start]]
            pixels, actions, mask = self._context(frames, past, history_size)
            future = group[start:start + horizon]
            pixels = np.concatenate((pixels, np.stack([self.read_frame(row["end"]) for row in future])))
            samples.append(dict(pixels=pixels.transpose(0, 3, 1, 2), history_actions=actions,
                                history_mask=mask, future_actions=np.asarray([row["action"] for row in future], np.float32),
                                future_mask=np.ones(horizon, bool)))
        if not samples:
            raise ValueError("Cannot create empty visual batch")
        return {key: torch.from_numpy(np.stack([sample[key] for sample in samples])) for key in samples[0]}

    def snapshot(self):
        references = {row[field]["path"]: row[field]["sha256"] for row in self.rows for field in ("start", "end")}
        return dict(rows=self.rows.copy(), frames=references, validation_every=self.validation_every)

    @staticmethod
    def verify_snapshot(snapshot):
        for path, expected in snapshot["frames"].items():
            if sha256(path) != expected:
                raise ValueError(f"Replay frame checksum mismatch: {path}")
