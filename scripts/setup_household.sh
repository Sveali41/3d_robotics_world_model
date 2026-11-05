#!/usr/bin/env bash
# Explicit setup entry point; installs only inside this workspace.
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
partnr_root="$project_root/third_party/partnr-planner"
runtime_root="$project_root/.runtime/partnr"
partnr_revision=ddfff19f4b6c098a31edea4d19e7b75db72433c2
export PYTHONNOUSERSITE=1
export PIP_CACHE_DIR="$project_root/.runtime/pip-cache"

case "${1:-}" in
  runtime)
    if [[ ! -d "$partnr_root/.git" ]]; then
      git clone https://github.com/facebookresearch/partnr-planner.git "$partnr_root"
      git -C "$partnr_root" checkout "$partnr_revision"
    fi
    if [[ "$(git -C "$partnr_root" rev-parse HEAD)" != "$partnr_revision" ]]; then
      echo "PARTNR revision differs from the supported revision: $partnr_revision" >&2
      exit 1
    fi
    git -C "$partnr_root" submodule update --init --recursive
    if [[ ! -x "$runtime_root/bin/python" ]]; then
      conda create --solver libmamba --prefix "$runtime_root" python=3.9.2 -y --quiet
    fi
    if ! "$runtime_root/bin/python" -s -c 'import habitat_sim' >/dev/null 2>&1; then
      conda install --solver libmamba --prefix "$runtime_root" \
        'habitat-sim=0.3.3=py3.9_bullet_linux_*' withbullet 'numpy<2' \
        -c conda-forge -c aihabitat -y --quiet
    fi
    # Keep the desktop simulator; oracle skills need no CUDA Torch or LLM weights.
    "$runtime_root/bin/python" -s -m pip install --index-url https://download.pytorch.org/whl/cpu \
      torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1
    "$runtime_root/bin/python" -s -m pip install \
      -r "$project_root/requirements/partnr_oracle.txt" \
      -e "$partnr_root/third_party/habitat-lab/habitat-lab" \
      -e "$partnr_root/third_party/habitat-lab/habitat-baselines" \
      -e "$partnr_root"
    ;;
  assets)
    "$runtime_root/bin/python" -s \
      "$project_root/scripts/download_household_assets.py" --partnr-root "$partnr_root" --download
    ;;
  *)
    echo "Usage: bash scripts/setup_household.sh runtime|assets" >&2
    exit 2
    ;;
esac
