#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR="$PWD/.runtime/matplotlib"
revision=53d5bd4c8c96a699e6a615dc390abb670cc9d353
case "${1:-runtime}" in
  runtime)
    if [[ ! -x .runtime/thor/bin/python ]]; then
      uv --cache-dir .runtime/uv-cache venv --python "${THOR_PYTHON:-python3.11}" .runtime/thor
    fi
    if [[ ! -d third_party/procthor/.git ]]; then
      git clone https://github.com/allenai/procthor.git third_party/procthor
      git -C third_party/procthor checkout "$revision"
    fi
    [[ "$(git -C third_party/procthor rev-parse HEAD)" == "$revision" ]] || {
      echo "ProcTHOR revision differs from the pinned source; inspect third_party/procthor." >&2
      exit 1
    }
    uv --cache-dir .runtime/uv-cache pip install --python .runtime/thor/bin/python \
      -r requirements/thor.txt -e third_party/procthor
    ;;
  unity)
    .runtime/thor/bin/python -s - <<'PY'
from pathlib import Path
from ai2thor.build import Build
from ai2thor.platform import Linux64
root = Path('.runtime/thor-unity/releases').resolve()
root.mkdir(parents=True, exist_ok=True)
Build(Linux64, 'f0825767cd50d69f666c7f282e54abfe58f1e917', False,
      releases_dir=str(root)).download()
PY
    ;;
  *) echo "Usage: bash scripts/setup_thor.sh [runtime|unity]" >&2; exit 2 ;;
esac
