#!/usr/bin/env python3
"""Validate the isolated sparse-apply overlay and load it after Torch, before SAPIEN."""
import argparse
import ctypes
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy
import shutil
import sys
import tempfile
import types

from build_sapien_sparse_patch import VERSION, installed_paths, verify_hash


def launch_environment(runtime, environment=None):
    runtime = Path(runtime).resolve()
    manifest_path = runtime / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != 1 or manifest.get("sapien_version") != VERSION:
        raise RuntimeError(f"Unsupported SAPIEN overlay manifest: {manifest_path}")
    originals = installed_paths()
    for key, path in originals.items():
        recorded = manifest["originals"][key]
        if str(path.resolve()) != recorded["path"]:
            raise RuntimeError(f"Installed {key} path differs from build manifest: {path}")
        verify_hash(path, recorded["sha256"])
    overlay = runtime / "libsapien_sparse_fix.so"
    digest = manifest["overlay"]["sha256"]
    verify_hash(overlay, digest)
    cache = Path(tempfile.gettempdir()) / f"sapien-sparse-{os.getuid()}-{digest}"
    cache.mkdir(mode=0o700, exist_ok=True)
    if cache.is_symlink() or cache.stat().st_uid != os.getuid() or cache.stat().st_mode & 0o077:
        raise RuntimeError(f"Unsafe overlay executable cache: {cache}")
    helper = cache / overlay.name
    if helper.exists():
        verify_hash(helper, digest)
    else:
        descriptor, temporary = tempfile.mkstemp(dir=cache)
        os.close(descriptor)
        try:
            shutil.copyfile(overlay, temporary)
            verify_hash(temporary, digest)
            os.chmod(temporary, 0o700)
            os.replace(temporary, helper)
        finally:
            if Path(temporary).exists():
                Path(temporary).unlink()
    oidn = originals["pysapien"].parent / "oidn_library"
    for stem in ("libOpenImageDenoise", "libOpenImageDenoise_core"):
        target = oidn / f"{stem}.so.2.0.1"
        if not target.is_file():
            raise RuntimeError(f"Required installed OIDN library missing: {target}")
        link = cache / f"{stem}.so.2"
        if link.is_symlink() and link.resolve() == target.resolve():
            continue
        if link.exists() or link.is_symlink():
            raise RuntimeError(f"Unexpected OIDN cache entry: {link}")
        link.symlink_to(target)
    environment = dict(os.environ if environment is None else environment)
    library_dirs = [str(originals["libsapien"].parent), str(oidn), str(cache)]
    if environment.get("LD_LIBRARY_PATH"):
        library_dirs.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_dirs)
    environment["SAPIEN_SPARSE_PATCH_LIBRARY"] = str(helper)
    environment["SAPIEN_SPARSE_PATCH_MANIFEST"] = str(manifest_path)
    return environment


def bootstrap(python_args):
    if python_args[0] in ("-m", "-c"):
        if len(python_args) < 2:
            raise ValueError(f"{python_args[0]} requires an argument")
    elif python_args[0].startswith("-"):
        raise ValueError("Supported Python arguments are -m module, -c code, or a script path")
    # Loading libsapien before Torch causes an observed torch.load RuntimeError.
    importlib.import_module("torch")
    overlay = ctypes.CDLL(os.environ["SAPIEN_SPARSE_PATCH_LIBRARY"], mode=ctypes.RTLD_GLOBAL)
    if python_args[0] == "-m":
        sys.path[0] = os.getcwd()
        sys.argv = python_args[1:]
        runpy.run_module(python_args[1], run_name="__main__", alter_sys=True)
    elif python_args[0] == "-c":
        sys.path[0] = os.getcwd()
        sys.argv = ["-c", *python_args[2:]]
        main_module = types.ModuleType("__main__")
        main_module.__package__ = None
        main_module.__spec__ = None
        sys.modules["__main__"] = main_module
        exec(compile(python_args[1], "<string>", "exec"), main_module.__dict__)
    else:
        sys.path[0] = str(Path(python_args[0]).resolve().parent)
        sys.argv = python_args
        runpy.run_path(python_args[0], run_name="__main__")
    # Keep the CDLL handle alive until the target program has completed.
    return overlay


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(".runtime/sapien_sparse_patch"))
    parser.add_argument("--bootstrap", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("python_args", nargs=argparse.REMAINDER, help="Python arguments following --")
    args = parser.parse_args()
    python_args = args.python_args
    if python_args[:1] == ["--"]:
        python_args = python_args[1:]
    if not python_args:
        parser.error("Provide Python arguments after --, for example -- -m pytest")
    if args.bootstrap:
        bootstrap(python_args)
        return
    if python_args[0].startswith("-") and python_args[0] not in ("-m", "-c"):
        parser.error("Supported Python arguments are -m module, -c code, or a script path")
    if python_args[0] in ("-m", "-c") and len(python_args) < 2:
        parser.error(f"{python_args[0]} requires an argument")
    try:
        environment = launch_environment(args.runtime)
    except (RuntimeError, OSError, ValueError, KeyError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(1, f"SAPIEN overlay launch failed: {error}\n")
    os.execve(sys.executable, [sys.executable, str(Path(__file__).resolve()),
                              "--bootstrap", "--", *python_args], environment)


if __name__ == "__main__":
    main()
