from omegaconf import OmegaConf


def test_make_env_passes_centered_initial_tcp_default_and_enabled(monkeypatch):
    import exploration.cee_us_cabinet as runner

    source_cfg = OmegaConf.create({
        "sim": {}, "ppo": {}, "max_primitives_per_episode": 1,
        "max_low_level_steps_per_episode": 1,
    })
    monkeypatch.setattr(runner.OmegaConf, "load", lambda _: source_cfg.copy())
    captured = []

    def build_env(cfg):
        captured.append(cfg.centered_initial_tcp)
        return object()

    monkeypatch.setattr("main.build_env", build_env)
    runner.make_env()
    runner.make_env(centered_initial_tcp=True)
    assert captured == [False, True]


def test_primitive_baselines_differ_only_in_disagreement_weight():
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    combined = OmegaConf.to_container(OmegaConf.load(
        root / "configs/raps_gnn_actor_cabinet_400k_episode40s_centered.yaml"))
    heuristic = OmegaConf.to_container(OmegaConf.load(
        root / "configs/raps_gnn_actor_cabinet_400k_heuristic_only.yaml"))
    assert combined["centered_initial_tcp"] is True
    assert heuristic["raps_actor"].pop("disagreement_weight") == 0.0
    assert combined == heuristic


def test_position_shift_is_explicit_and_defaults_preserve_scene(monkeypatch):
    import exploration.cee_us_cabinet as runner
    source_cfg = OmegaConf.create({'sim': {}, 'ppo': {}})
    monkeypatch.setattr(runner.OmegaConf,'load',lambda _:source_cfg.copy())
    captured=[]
    monkeypatch.setattr('main.build_env',lambda cfg:captured.append(cfg))
    runner.make_env()
    runner.make_env(object_shift_xy=(.03,0.))
    assert 'object_shift_xy' not in captured[0]
    assert list(captured[1].object_shift_xy)==[.03,0.]


def test_position_shift_cannot_accidentally_start_training():
    import pytest
    from exploration.cee_us_cabinet import main
    with pytest.raises(SystemExit) as exc:
        main(['train','--object-shift-xy','.03','0'])
    assert exc.value.code == 2
