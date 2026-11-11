#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONNOUSERSITE=1
training_python="${CEEUS_PYTHON:-/home/siyao/Apps/anaconda3/envs/rlPractice/bin/python}"
"$training_python" -s -c 'import torch, gymnasium, forwardable'
if [[ ! -x .runtime/thor-ceeus/bin/python ]]; then
  uv --cache-dir .runtime/uv-cache venv --system-site-packages \
    --python "$training_python" .runtime/thor-ceeus
fi
uv --cache-dir .runtime/uv-cache pip install --python .runtime/thor-ceeus/bin/python \
  -r requirements/thor_cee_us.txt
.runtime/thor-ceeus/bin/python -s -c \
  'import torch, ai2thor, cv2; from exploration.cee_us.cee_us_backend import CEEUSBackend; print("THOR / CEE-US runtime imports OK")'
