"""Selectable stochastic-JEPA/CEM cabinet experiment; RGB and actions only."""

import importlib.metadata
import json
from copy import deepcopy
from pathlib import Path
import random

import numpy as np
from omegaconf import OmegaConf
import torch

from exploration.cee_us.cee_us_cabinet import ROOT
from .loop import Learner, read_checkpoint, run_loop, write_json
from .replay import VisualReplay, sha256


def resume_config(saved_config, *, stage, total_controls):
    """Return a copied saved config with an optional aligned training budget extension."""
    resolved = deepcopy(saved_config)
    if total_controls is None:
        return resolved
    if isinstance(total_controls, bool) or not isinstance(total_controls, int) or total_controls < 1:
        raise ValueError("--total-controls must be a positive integer")
    saved_controls = saved_config["total_controls"]
    if total_controls == saved_controls:
        return resolved
    if total_controls < saved_controls:
        raise ValueError(f"Resume control budget cannot decrease from {saved_controls} to {total_controls}")
    if stage != "train":
        raise ValueError("Only --stage train may extend a checkpoint control budget")
    remaining = total_controls - saved_config["bootstrap_steps"]
    if remaining < 0 or remaining % saved_config["steps_per_round"] != 0:
        raise ValueError("Extended --total-controls must equal bootstrap_steps plus whole steps_per_round")
    resolved["total_controls"] = total_controls
    resolved["rounds"] = remaining // saved_config["steps_per_round"]
    return resolved


def resolve(config, args):
    config = OmegaConf.to_container(config, resolve=True)
    if config.get("world_model") != "stochastic_jepa" or config.get("planner_type") != "mixed_cem":
        raise ValueError("epic_jepa_cem requires configs/epic_jepa_cabinet_split7.yaml; "
                         "arbitrary world-model/planner combinations are unsupported")
    config["agent"] = "epic_jepa_cem"
    config["collection_policy"] = args.policy
    if args.device:
        config["device"] = args.device
    if args.stage == "smoke":
        config.update(config.pop("smoke"))
    else:
        config.pop("smoke", None)
    if getattr(args, "total_controls", None) is not None:
        config["total_controls"] = args.total_controls
    if args.stage == "train" and args.checkpoint is None and getattr(args, "total_controls", None) is None:
        raise ValueError("Formal JEPA training requires an explicit --total-controls budget")
    for name in ("bootstrap_steps", "rounds", "steps_per_round", "epochs", "batch_size", "episode_steps", "cpu_threads", "total_controls"):
        value = config.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if config["batch_size"] < 2:
        raise ValueError("JEPA BatchNorm training requires batch_size >=2")
    required = config["bootstrap_steps"] + config["rounds"] * config["steps_per_round"]
    if required > config["total_controls"] or (args.stage == "smoke" and config["total_controls"] > 2000):
        raise ValueError("Configured round budgets exceed the explicit control budget or 2000-control smoke cap")
    if (config.get("action_mode") != "ee_delta" or config.get("sim_freq") != 100 or config.get("control_freq") != 20
            or config.get("objective") != "predictive_variance" or config.get("object_heuristic") is not None):
        raise ValueError("JEPA requires direct EE split7, 100/20Hz and pure predictive-variance curiosity")
    settings = config["raps_actor"]
    if settings.get("skill_set") != "split7" or not settings.get("latched_gripper") or settings.get("motion_execution") != "ee_delta":
        raise ValueError("JEPA requires unchanged latched split7 direct-EE primitive settings")
    if args.execution not in (None, "reference") or args.runtime_optimizations:
        raise ValueError("JEPA supports reference execution only; GNN CUDA graph optimizations do not apply")
    if config["device"] == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose --device cpu")
    return config


def run(args, config):
    if args.stage not in {"smoke", "train", "evaluate", "report"}:
        raise ValueError("JEPA supports smoke/train/evaluate/report; run random and curiosity in separate directories")
    if args.stage == "report":
        if args.output is None:
            raise ValueError("report requires --output")
        print((args.output / "result.json").read_text())
        return 0
    if args.stage == "evaluate" and args.checkpoint is None:
        raise ValueError("JEPA evaluate requires --checkpoint")
    resolved = resolve(config, args) if args.checkpoint is None else None
    saved = None
    if args.checkpoint is not None:
        saved_config = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["config"]
        if args.device and args.device != saved_config["device"]:
            raise ValueError("Resume/evaluate must use the checkpoint device configuration")
        if args.policy != saved_config["collection_policy"]:
            raise ValueError("Resume/evaluate collection policy differs from checkpoint")
        saved = read_checkpoint(args.checkpoint, saved_config)
        resolved = resume_config(saved_config, stage=args.stage,
                                 total_controls=getattr(args, "total_controls", None))
    output = Path(args.output or ROOT / "logs" / "epic_jepa_cem").resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    if args.stage == "evaluate":
        if getattr(args, "evaluation_steps", None) is not None or any(args.object_shift_xy):
            raise ValueError("JEPA evaluate uses saved held-out RGB replay; live shifted evaluation is unsupported")
    torch.set_num_threads(resolved["cpu_threads"])
    random.seed(resolved["seed"])
    np.random.seed(resolved["seed"])
    torch.manual_seed(resolved["seed"])
    if resolved["device"] == "cuda":
        torch.cuda.reset_peak_memory_stats()
    from .model import StochasticJEPA
    from .planner import MixedCEM
    model = StochasticJEPA(resolved["model"], device=resolved["device"])
    if model.config["history_size"] != 3 or model.config["horizon"] != 3:
        raise ValueError("Cabinet visual replay currently requires three history and three future frames")
    planner = MixedCEM(model, resolved["planner"], seed=resolved["seed"])
    # Store effective model/planner defaults so checkpoint contracts are complete.
    if saved is None:
        resolved["model"] = model.config
        resolved["planner"] = planner.config
    elif model.config != resolved["model"] or planner.config != resolved["planner"]:
        raise ValueError("Saved model/planner defaults differ from this implementation")
    output.mkdir(parents=True)
    collector = None
    try:
        replay = VisualReplay(output, validation_every=resolved["validation_every"],
                              rows=saved["replay"]["rows"] if saved else ())
        learner = Learner(model, resolved)
        sources = [path for path in (ROOT / "exploration/jepa").rglob("*") if path.suffix in {".py", ".json", ".yaml"}]
        manifest = dict(world_model="stochastic_jepa", planner="mixed_cem", objective="predictive_variance",
                        observation="RGB and canonical primitive action history only", sim_backend="cpu",
                        model_variant="standard conditional flow matching plus SIGReg; not MeanFlow",
                        source_sha256={str(path.relative_to(ROOT)): sha256(path) for path in sources},
                        versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "transformers")},
                        physical_dt_seconds=.05, model_transition="one completed variable-duration primitive",
                        rgb=dict(camera="base_camera", resolution=[320, 180], storage="uint8 lossless NPY"),
                        resume=str(args.checkpoint.resolve()) if args.checkpoint else None,
                        adaptations=["split7 skill-boundary transitions", "pure variance instead of goal-conditioned cost",
                                     "online replay fitting", "mixed categorical/continuous CEM", "masked episode-start history"])
        if saved is not None and resolved["total_controls"] != saved["config"]["total_controls"]:
            manifest["budget_extension"] = dict(
                original_total_controls=saved["config"]["total_controls"],
                new_total_controls=resolved["total_controls"],
                rounds=resolved["rounds"],
            )
        write_json(output / "manifest.json", manifest)
        write_json(output / "config.json", resolved)
        if args.stage == "evaluate":
            model.load_state_dict(saved["model"])
            write_json(output / "evaluation.json", learner.assess(replay))
            return 0
        from .collector import make_collector
        collector = make_collector(resolved, output, replay, planner)
        result = run_loop(resolved, output, learner, planner, replay, collector, resumed=saved)
        print(json.dumps({key: result[key] for key in ("status", "real_controls", "updates", "replay_transitions")}))
    except BaseException as exc:
        write_json(output / "failure.json", dict(type=type(exc).__name__, message=str(exc),
                                                status="stopped_without_random_fallback"))
        raise
    finally:
        if collector is not None:
            collector.env.close()
    return 0
