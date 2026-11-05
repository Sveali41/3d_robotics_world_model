"""Save local upstream changes in the main repository; --check detects stale snapshots."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
SOURCES = {
    "pointworld": "exploration/pointworld/official",
    "raps": "exploration/raps/official",
    "maniskill": "domain/ManiSkill",
}


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check snapshots without writing")
    args = parser.parse_args()
    snapshots = {}
    manifest = {}
    for name, relative in SOURCES.items():
        source = ROOT / relative
        commit = git(source, "rev-parse", "HEAD").decode().strip()
        entry = git(ROOT, "ls-files", "--stage", "--", relative).decode().split()
        if len(entry) < 2 or entry[0] != "160000" or entry[1] != commit:
            raise SystemExit(f"{relative}: HEAD must match the main repository's staged submodule commit")
        untracked = git(source, "ls-files", "--others", "--exclude-standard").decode().strip()
        if untracked:
            raise SystemExit(f"{relative}: stage intended new files inside this submodule or ignore artifacts before snapshotting:\n{untracked}")
        patch = git(source, "diff", "--binary", "--full-index", "--no-ext-diff", "HEAD", "--", ".")
        filename = f"patches/{name}.patch"
        snapshots[filename] = patch
        manifest[name] = {"path": relative, "commit": commit, "patch": filename,
                          "sha256": hashlib.sha256(patch).hexdigest()}
    snapshots["patches/upstream.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    for filename, content in snapshots.items():
        target = ROOT / filename
        if args.check:
            if not target.is_file() or target.read_bytes() != content:
                raise SystemExit(f"Stale snapshot: {filename}; run python scripts/snapshot_upstream_patches.py")
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
    print("Upstream patch snapshots match." if args.check else "Upstream patch snapshots saved; no commits created.")


if __name__ == "__main__":
    main()
