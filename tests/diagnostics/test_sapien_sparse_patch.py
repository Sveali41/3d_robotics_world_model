"""Bounded native-overlay integrity and launcher tests; no physics execution."""
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[2] / "scripts"))
    return (importlib.import_module("build_sapien_sparse_patch"),
            importlib.import_module("run_patched_sapien"))


def test_extracts_only_six_sparse_definitions(modules):
    build, _ = modules
    definitions = [f"void PhysxSystemGpu::{name}(CudaArrayHandle const &indices) {{\n  if (true) {{\n    call();\n  }}\n}}"
                   for name in build.METHODS]
    full = "unrelated_prefix\n" + "\n\n".join(definitions) + "\nunrelated_suffix\n"
    result = build.extract_methods(full)
    assert result == build.PREAMBLE + "\n\n".join(definitions) + "\n}}\n"
    assert result.count("void PhysxSystemGpu::") == 6
    with pytest.raises(RuntimeError, match="exactly one"):
        build.extract_methods(full + "\n" + definitions[0])
    with pytest.raises(RuntimeError, match="exactly one"):
        build.extract_methods(full.replace(definitions[-1], ""))


def test_hash_mismatch_rejected(modules, tmp_path):
    build, _ = modules
    source = tmp_path / "source"
    source.write_bytes(b"unexpected bytes")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        build.verify_hash(source, "0" * 64)


def runtime_fixture(modules, tmp_path, monkeypatch):
    build, launcher = modules
    package = tmp_path / "installed/sapien"
    paths = {"pysapien": package / "pysapien.test.so",
             "libsapien": package.parent / "sapien.libs/libsapien.so",
             "header": package / "include/sapien/physx/physx_system.h"}
    for name, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name)
    oidn = package / "oidn_library"
    oidn.mkdir()
    for name in ("libOpenImageDenoise", "libOpenImageDenoise_core"):
        (oidn / f"{name}.so.2.0.1").write_text(name)
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    overlay = runtime / "libsapien_sparse_fix.so"
    overlay.write_bytes(b"test overlay")
    manifest = {"schema_version": 1, "sapien_version": build.VERSION,
                "originals": {key: build.record(path) for key, path in paths.items()},
                "overlay": build.record(overlay)}
    (runtime / "manifest.json").write_text(json.dumps(manifest))
    executable_tmp = tmp_path / "executable"
    executable_tmp.mkdir()
    monkeypatch.setattr(launcher, "installed_paths", lambda: paths)
    monkeypatch.setattr(launcher.tempfile, "gettempdir", lambda: str(executable_tmp))
    return launcher, runtime, paths, overlay, executable_tmp


def test_launch_preserves_environment_and_copies_to_executable_cache(modules, tmp_path, monkeypatch):
    launcher, runtime, paths, overlay, executable_tmp = runtime_fixture(modules, tmp_path, monkeypatch)
    original = {"PYTHONPATH": "/tmp/numpy-shim:.", "LD_PRELOAD": "/old/preload.so",
                "LD_LIBRARY_PATH": "/old/libs", "CUSTOM_VARIABLE": "preserved"}
    env = launcher.launch_environment(runtime, original)
    helper = Path(env["SAPIEN_SPARSE_PATCH_LIBRARY"])
    assert helper.parent.parent == executable_tmp
    assert helper.read_bytes() == overlay.read_bytes()
    assert env["LD_PRELOAD"] == original["LD_PRELOAD"]
    assert env["PYTHONPATH"] == original["PYTHONPATH"]
    assert env["CUSTOM_VARIABLE"] == "preserved"
    assert env["LD_LIBRARY_PATH"].split(":") == [str(paths["libsapien"].parent),
        str(paths["pysapien"].parent / "oidn_library"), str(helper.parent), "/old/libs"]
    assert env["SAPIEN_SPARSE_PATCH_MANIFEST"] == str(runtime / "manifest.json")
    for name in ("libOpenImageDenoise", "libOpenImageDenoise_core"):
        assert (helper.parent / f"{name}.so.2").resolve() == paths["pysapien"].parent / "oidn_library" / f"{name}.so.2.0.1"
    assert launcher.launch_environment(runtime, original) == env
    assert original["LD_PRELOAD"] == "/old/preload.so"
    helper.write_bytes(b"tampered executable cache")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        launcher.launch_environment(runtime, original)


@pytest.mark.parametrize("target", ["libsapien", "pysapien", "header", "overlay"])
def test_launch_rejects_modified_native_files(modules, tmp_path, monkeypatch, target):
    launcher, runtime, paths, overlay, _ = runtime_fixture(modules, tmp_path, monkeypatch)
    (overlay if target == "overlay" else paths[target]).write_bytes(b"modified")
    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        launcher.launch_environment(runtime)


def test_unsupported_installed_version_is_clear(modules, monkeypatch):
    build, _ = modules
    class Distribution:
        version = "3.0.4"
    monkeypatch.setattr(build.importlib.metadata, "distribution", lambda name: Distribution())
    with pytest.raises(RuntimeError, match="Unsupported SAPIEN 3.0.4"):
        build.installed_paths()


@pytest.mark.parametrize("arguments", [["-m", "example", "arg"], ["-c", "print(1)", "arg"], ["example.py", "arg"]])
def test_bootstrap_loads_torch_before_overlay_and_dispatches(modules, monkeypatch, arguments):
    _, launcher = modules
    calls = []
    monkeypatch.setenv("SAPIEN_SPARSE_PATCH_LIBRARY", "/tmp/helper.so")
    monkeypatch.setattr(launcher.importlib, "import_module", lambda name: calls.append(("import", name)))
    monkeypatch.setattr(launcher.ctypes, "CDLL", lambda path, mode: calls.append(("load", path, mode)))
    monkeypatch.setattr(launcher.runpy, "run_module", lambda *args, **kwargs: calls.append(("module", args, kwargs, sys.argv.copy())))
    monkeypatch.setattr(launcher.runpy, "run_path", lambda *args, **kwargs: calls.append(("script", args, kwargs, sys.argv.copy())))
    monkeypatch.setattr(launcher.sys, "argv", sys.argv.copy())
    monkeypatch.setattr(launcher.sys, "path", sys.path.copy())
    monkeypatch.setitem(sys.modules, "__main__", sys.modules["__main__"])
    launcher.bootstrap(arguments)
    assert calls[:2] == [("import", "torch"), ("load", "/tmp/helper.so", launcher.ctypes.RTLD_GLOBAL)]
    if arguments[0] == "-m":
        assert calls[2] == ("module", ("example",), {"run_name": "__main__", "alter_sys": True}, ["example", "arg"])
    elif arguments[0] == "-c":
        assert sys.argv == ["-c", "arg"]
    else:
        assert calls[2] == ("script", ("example.py",), {"run_name": "__main__"}, arguments)


def test_launcher_torch_checkpoint_round_trip():
    root = Path(__file__).resolve().parents[2]
    runtime = root / ".runtime/sapien_sparse_patch"
    if not (runtime / "manifest.json").exists() or importlib.util.find_spec("sapien") is None:
        pytest.skip("Requires the built overlay and matching SAPIEN container")
    code = """import io, os, torch, sapien
assert not any('libsapien_sparse_fix' in entry for entry in os.environ.get('LD_PRELOAD', '').split(':'))
value = {'tensor': torch.arange(6, dtype=torch.float32).reshape(2, 3)}
stream = io.BytesIO()
torch.save(value, stream)
stream.seek(0)
restored = torch.load(stream, map_location='cpu')
assert torch.equal(value['tensor'], restored['tensor'])
print('torch checkpoint round trip OK')
"""
    result = subprocess.run([sys.executable, str(root / "scripts/run_patched_sapien.py"),
        "--runtime", str(runtime), "--", "-c", code], cwd=root, env=os.environ.copy(),
        text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "torch checkpoint round trip OK" in result.stdout
