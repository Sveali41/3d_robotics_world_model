#!/usr/bin/env bash
# Small official viewer demo; no Torch, PARTNR, full HSSD or robot assets required.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runtime_root="$project_root/.runtime/partnr"
sim_root="$project_root/third_party/habitat-sim"
sim_revision=acbe6f4922e68145e401e55c30f9dfea460a3f24
asset_revision=910c783fb954da8497ea5f811b843a76590ddddc

if [[ ! -d "$sim_root/.git" ]]; then
  git clone --depth 1 --branch v0.3.3 https://github.com/facebookresearch/habitat-sim.git "$sim_root"
fi
if [[ "$(git -C "$sim_root" rev-parse HEAD)" != "$sim_revision" ]]; then
  echo "Expected Habitat-Sim source revision $sim_revision" >&2
  exit 1
fi
if [[ ! -x "$runtime_root/bin/python" ]]; then
  conda create --solver libmamba --prefix "$runtime_root" python=3.9.2 -y --quiet
fi
conda install --solver libmamba --prefix "$runtime_root" \
  'habitat-sim=0.3.3=py3.9_bullet_linux_*' withbullet 'numpy<2' \
  -c conda-forge -c aihabitat -y --quiet
mkdir -p "$project_root/data/habitat_demo"
for asset in van-gogh-room.glb van-gogh-room.navmesh; do
  if [[ ! -f "$project_root/data/habitat_demo/$asset" ]]; then
    curl --fail --location \
      "https://huggingface.co/datasets/ai-habitat/habitat_test_scenes/resolve/$asset_revision/$asset" \
      --output "$project_root/data/habitat_demo/$asset"
  fi
done
echo "Ready to launch from the project root: python main.py room"
