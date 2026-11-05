"""Independent PointWorld members with one frozen DINO at joint inference.

Training uses the upstream single-member trainer. Raw, camera-aggregated DINO
features are shared, never the independently learned visual projections.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch import nn


SCHEMA = "cabinet-pointworld-ensemble-v1"
DINO_PREFIX = "scene_feature_encoder.scene_encoder.dinov3."
NORMALIZERS = ("norm_stats_per_step_mean", "norm_stats_per_step_var",
               "robot_norm_mean", "robot_norm_var", "scene_norm_mean", "scene_norm_var")


def tensor_fingerprint(state, keys):
    digest = hashlib.sha256()
    for key in sorted(keys):
        value = state[key].detach().cpu().contiguous()
        digest.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
    return digest.hexdigest()


class PointWorldEnsemble(nn.Module):
    """Inference-only ensemble; all members query the same points and actions."""

    def __init__(self, members):
        super().__init__()
        if len(members) < 2:
            raise ValueError("Sample-standard-deviation disagreement requires >=2 independent members")
        backbone = members[0].scene_feature_encoder.scene_encoder.dinov3
        shared = {id(p) for p in backbone.parameters()}
        if any(p.requires_grad for p in backbone.parameters()):
            raise ValueError("Shared DINO backbone must be frozen")
        previous = set()
        for member in members:
            if member.scene_feature_encoder.scene_encoder.dinov3 is not backbone:
                raise ValueError("Members must share exactly one DINO backbone")
            private = {id(p) for p in member.parameters()} - shared
            if previous & private:
                raise ValueError("Members share non-DINO parameters; dynamics must be independent")
            previous |= private
        self.members = nn.ModuleList(members)
        self.args = members[0].args
        self.eval()

    def train(self, mode=True):
        if mode:
            raise RuntimeError("Train each member with the official PointWorld trainer; ensemble is inference-only")
        return super().train(False)

    @torch.no_grad()
    def forward(self, batch, training=False, *, point_mask=None):
        if training or self.training:
            raise RuntimeError("Ensemble disagreement requires frozen eval-mode members")
        encoder = self.members[0].scene_feature_encoder.scene_encoder
        features = encoder(batch["scene_flows"][:, 0], batch["scene_exists"][:, 0],
            encoder._extract_camera_data(batch), return_backbone_features=True)
        if not torch.isfinite(features).all():
            raise FloatingPointError("Nonfinite shared DINO features")
        predictions = []
        for member in self.members:
            output = member(batch, training=False, backbone_features=features)
            predicted = output["scene_flows"]
            if output.get("_has_nan_outputs", False) or not torch.isfinite(predicted).all():
                raise FloatingPointError("Nonfinite PointWorld ensemble member output")
            if predicted.shape != batch["scene_flows"].shape:
                raise ValueError("Member prediction does not preserve batch/time/point correspondence")
            predictions.append(predicted)
        predictions = torch.stack(predictions)
        mean = predictions.mean(0)
        std = predictions.std(0, unbiased=True)
        valid = batch["scene_exists"][:, 0].bool()
        if point_mask is not None:
            if point_mask.dtype != torch.bool or point_mask.shape != valid.shape:
                raise ValueError("point_mask must be bool [batch, initial_scene_points]")
            valid = valid & point_mask
        # No normalization by point count, epsilon exploration, or aleatoric variance.
        per_step = std.sum(-1).masked_fill(~valid[:, None, :], 0).sum(-1)
        return dict(scene_flows=mean, member_scene_flows=predictions,
                    disagreement=std, disagreement_per_step=per_step,
                    disagreement_score=per_step.sum(-1), scored_points=valid.sum(-1))

    @classmethod
    def from_checkpoints(cls, paths, *, device="cuda"):
        paths = [Path(p).resolve() for p in paths]
        if len(paths) < 2 or len(set(paths)) != len(paths):
            raise ValueError("Provide >=2 distinct member checkpoint paths")
        root = Path(__file__).resolve().parents[2] / "exploration/pointworld/official"
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from pointworld.base import BaseModel
        from pointworld.checkpoint_contract import read_checkpoint_contract
        members, backbone, reference = [], None, None
        fingerprints = set()
        for path in paths:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            args = deepcopy(payload["args"])
            state = payload["model"]
            contracts = read_checkpoint_contract(payload, context=f"ensemble member {path}")
            signature = (contracts, args.domains, args.robot_features, args.scene_features,
                         getattr(args, "raps_action_dim", 0))
            dino_keys = [k for k in state if k.startswith(DINO_PREFIX)]
            if not dino_keys:
                raise ValueError(f"Checkpoint has no frozen DINO weights: {path}")
            dino_hash = tensor_fingerprint(state, dino_keys)
            norm_hash = tensor_fingerprint(state, NORMALIZERS)
            private_hash = tensor_fingerprint(state, [k for k in state if not k.startswith(DINO_PREFIX)])
            if private_hash in fingerprints:
                raise ValueError("Identical member weights are not an independent ensemble")
            fingerprints.add(private_hash)
            if reference is None:
                reference = signature, dino_hash, norm_hash
            elif reference != (signature, dino_hash, norm_hash):
                raise ValueError("Ensemble members differ in contracts, DINO weights or normalization statistics")
            args.device, args.distributed, args.disable_compile = device, False, True
            member = BaseModel(args, {"scene_features_dim": 31, "robot_features_dim": 16},
                               scene_backbone=backbone)
            if backbone is None:
                member.load_state_dict(state)
                backbone = member.scene_feature_encoder.scene_encoder.dinov3
            else:
                # Avoid reloading shared weights through another member.
                missing, unexpected = member.load_state_dict(
                    {k: v for k, v in state.items() if not k.startswith(DINO_PREFIX)}, strict=False)
                if set(missing) != set(dino_keys) or unexpected:
                    raise ValueError(f"Unexpected ensemble checkpoint keys: {path}")
            member.to(device).eval()
            members.append(member)
            del payload, state
        return cls(members)

    @classmethod
    def from_manifest(cls, path, *, device="cuda"):
        from exploration.pointworld.pointworld_cabinet_offline import sha256
        path = Path(path).resolve()
        manifest = json.loads(path.read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("status") != "complete":
            raise ValueError("Requires a completed PointWorld ensemble manifest")
        paths = []
        for member in manifest["members"]:
            checkpoint = path.parent / member["checkpoint"]
            if sha256(checkpoint) != member["sha256"]:
                raise ValueError(f"Ensemble member hash mismatch: {checkpoint}")
            paths.append(checkpoint)
        ensemble = cls.from_checkpoints(paths, device=device)
        ensemble.manifest = manifest
        return ensemble
