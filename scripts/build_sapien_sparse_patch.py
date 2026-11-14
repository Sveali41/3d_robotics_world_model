#!/usr/bin/env python3
"""Build the pinned SAPIEN 3.0.3 sparse-apply overlay inside its container."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request

VERSION = "3.0.3"
COMMIT = "731622eac5b140b320076c8a1b6eb4b553c3ccd4"
SOURCE_SHA256 = "8173ce5960fe79b3657eed60c79a13834a786b409b1eaf56264169be7d65d27b"
HEADER_SHA256 = "95a0653c9dd45702287a59aa9bec902a74b26d812e929340a552308ca8dd5df7"
EIGEN_SHA256 = "8586084f71f9bde545ee7fa6d00288b264a2b7ac3607b974e54d13e7162c1c72"
METHODS = ("gpuApplyArticulationQpos", "gpuApplyArticulationQvel", "gpuApplyArticulationQf",
           "gpuApplyArticulationQTargetPos", "gpuApplyArticulationQTargetVel", "gpuApplyRigidDynamicData")
PREAMBLE = """#include <sapien/physx/physx_system.h>
#include <sapien/profiler.h>
using namespace physx;
namespace sapien { namespace physx {
void body_data_sapien_to_physx(void *, void *, void *, int, CUstream_st *);
void body_data_sapien_to_physx(void *, void *, void *, void *, void *, void *, int, CUstream_st *);
"""


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest() if hasattr(hashlib, "file_digest") else _digest(stream)


def _digest(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def verify_hash(path, expected):
    actual = sha256(path)
    if actual != expected:
        raise RuntimeError(f"SHA256 mismatch for {path}: expected {expected}, found {actual}")
    return actual


def record(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": sha256(path)}


def installed_paths():
    dist = importlib.metadata.distribution("sapien")
    if dist.version != VERSION:
        raise RuntimeError(f"Unsupported SAPIEN {dist.version}; this overlay requires {VERSION}")
    package = Path(dist.locate_file("sapien")).resolve()
    bindings = list(package.glob("pysapien.*.so"))
    if len(bindings) != 1:
        raise RuntimeError(f"Expected one pysapien native binding in {package}, found {bindings}")
    return {"libsapien": package.parent / "sapien.libs" / "libsapien.so",
            "pysapien": bindings[0], "header": package / "include/sapien/physx/physx_system.h"}


def extract_methods(source):
    """Extract only exact sparse overloads; the pinned source has top-level closing braces."""
    definitions = []
    for method in METHODS:
        signature = f"void PhysxSystemGpu::{method}(CudaArrayHandle const &indices) {{"
        if source.count(signature) != 1:
            raise RuntimeError(f"Expected exactly one sparse definition of {method}")
        start = source.index(signature)
        end = source.find("\n}", start)
        if end < 0:
            raise RuntimeError(f"Unterminated definition of {method}")
        definitions.append(source[start:end + 2])
    return PREAMBLE + "\n\n".join(definitions) + "\n}}\n"


def asset(local, destination, url, expected):
    if local is not None:
        path = local.resolve()
    else:
        path = destination
        if not path.exists():
            with urllib.request.urlopen(url, timeout=60) as response, path.open("wb") as output:
                shutil.copyfileobj(response, output)
    verify_hash(path, expected)
    return path


def build(args):
    originals = installed_paths()
    verify_hash(originals["header"], HEADER_SHA256)
    runtime = args.runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    patch = args.patch.resolve()
    with tempfile.TemporaryDirectory(prefix="sapien-sparse-build-") as temporary:
        work = Path(temporary)
        source = asset(args.source, work / "physx_system.cpp",
                       f"https://raw.githubusercontent.com/haosulab/SAPIEN/{COMMIT}/src/physx/physx_system.cpp", SOURCE_SHA256)
        eigen_archive = asset(args.eigen, work / "eigen.tar.gz",
                              "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz", EIGEN_SHA256)
        with tarfile.open(eigen_archive) as archive:
            # The archive is hash pinned; reject paths outside its one expected root.
            for member in archive.getmembers():
                if (member.name != "eigen-3.4.0" and not member.name.startswith("eigen-3.4.0/")) or ".." in Path(member.name).parts or member.issym() or member.islnk():
                    raise RuntimeError(f"Unexpected Eigen archive member: {member.name}")
            archive.extractall(work)
        patched = work / "src/physx/physx_system.cpp"
        patched.parent.mkdir(parents=True)
        shutil.copyfile(source, patched)
        subprocess.run(["patch", "--batch", "--fuzz=0", "-p1", "-i", str(patch)], cwd=work, check=True)
        overlay_source = runtime / "sparse_fix.cpp"
        overlay_source.write_text(extract_methods(patched.read_text()))
        overlay = runtime / "libsapien_sparse_fix.so"
        include = originals["header"].parents[2]
        options = ["-std=c++20", "-O2", "-fPIC", "-shared", "-DSAPIEN_CUDA",
                   f"-I{include}", f"-I{include / 'physx/include'}", f"-I{work / 'eigen-3.4.0'}",
                   f"-L{originals['libsapien'].parent}", "-lsapien", "-Wl,--no-undefined"]
        compiler = shutil.which(args.compiler)
        if compiler is None:
            raise RuntimeError(f"C++ compiler not found: {args.compiler}")
        # Link libraries after the translation unit (required by --as-needed toolchains).
        command = [compiler, str(overlay_source), "-o", str(overlay), *options]
        subprocess.run(command, check=True)
        manifest = {"schema_version": 1, "sapien_version": VERSION, "upstream_commit": COMMIT,
                    "source": record(source), "patch": record(patch), "eigen_archive": record(eigen_archive),
                    "originals": {key: record(path) for key, path in originals.items()},
                    "overlay_source": record(overlay_source), "overlay": record(overlay),
                    "compiler": {"path": compiler, "version": subprocess.check_output([compiler, "--version"], text=True)},
                    "command": command}
        # Temporary input paths are provenance records; their hashes remain authoritative.
        manifest_path = runtime / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(manifest_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", type=Path, default=Path(".runtime/sapien_sparse_patch"))
    parser.add_argument("--patch", type=Path, default=Path(__file__).resolve().parents[1] / "patches/sapien-3.0.3-sparse-joint-indices.patch")
    parser.add_argument("--source", type=Path, help="Cached pinned upstream physx_system.cpp")
    parser.add_argument("--eigen", type=Path, help="Cached pinned Eigen 3.4.0 tar.gz")
    parser.add_argument("--compiler", default="c++")
    try:
        build(parser.parse_args())
    except (RuntimeError, OSError, subprocess.CalledProcessError, importlib.metadata.PackageNotFoundError) as error:
        parser.exit(1, f"SAPIEN overlay build failed: {error}\n")


if __name__ == "__main__":
    main()
