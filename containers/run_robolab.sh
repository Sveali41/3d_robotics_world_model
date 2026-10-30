#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_dir="$project_root/outputs/robolab-isaac60"
mkdir -p "$runtime_dir"/{home,output,cache,data,logs}

exec apptainer run --nv --cleanenv \
    --home "$runtime_dir/home" \
    --env OMNI_KIT_ACCEPT_EULA=Y \
    --bind "$runtime_dir/output:/opt/robolab/output" \
    --bind "$runtime_dir/cache:/opt/robolab-venv/lib/python3.12/site-packages/isaacsim/kit/cache" \
    --bind "$runtime_dir/data:/opt/robolab-venv/lib/python3.12/site-packages/isaacsim/kit/data" \
    --bind "$runtime_dir/logs:/opt/robolab-venv/lib/python3.12/site-packages/isaacsim/kit/logs" \
    "$project_root/outputs/3d_world_model_robolab_isaac60.sif" "$@"
