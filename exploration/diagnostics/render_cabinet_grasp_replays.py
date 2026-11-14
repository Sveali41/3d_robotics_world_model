"""Render bounded, state-verified replays of recorded cabinet controller actions."""
import argparse
import json
import os
from pathlib import Path
import random

import imageio.v2 as imageio
import numpy as np
from omegaconf import OmegaConf
from PIL import Image, ImageDraw
import torch

from exploration.cee_us.cee_us_state import CabinetSkillStateEncoder
from exploration.visualization.scripted_gif import _frame
from main import build_env
from exploration.diagnostics.trajectory_store import read_transitions

FPS = 20
HOLD_FRAMES = 20


def verify_state(actual, expected, label):
    actual, expected = np.asarray(actual, dtype=np.float32), np.asarray(expected, dtype=np.float32)
    if actual.shape != (67,) or expected.shape != (67,) or not np.isfinite(actual).all() or not np.isfinite(expected).all():
        raise ValueError(f'{label}: invalid state67')
    error = float(np.max(np.abs(actual-expected)))
    if not np.array_equal(actual, expected):
        raise ValueError(f'{label}: replay state mismatch (max abs error {error:.9g})')
    return error


def label_lines(case, summary, step, trace, state, initial):
    result = 'success' if summary['success'] else str(summary['report']['failure'] or summary['status'])
    if trace is None:
        feedback = 'reset (no grasp feedback)'
        phase = 'reset'
    else:
        confirmed = bool(trace['target_grasped'] and trace['stable_grasp_ticks'] >= 4)
        feedback = 'confirmed' if confirmed else 'not confirmed'
        phase = trace['phase']
    index = 35 if case['object'] == 'cup' else 47
    height_cm = 100.*(float(state[index])-float(initial[index]))
    return [f"Case {case['case_id']} | {case['object']} | {case['template']}",
            f'Tick {step} | {step/FPS:.2f} s | phase: {phase}',
            f'Grasp: {feedback} | target height change: {height_cm:+.2f} cm',
            f'Recorded final outcome: {result}']


def overlay(frame, lines):
    image = Image.fromarray(_frame(frame))
    draw = ImageDraw.Draw(image, 'RGBA')
    draw.rectangle((0, 0, image.width, 78), fill=(0, 0, 0, 190))
    for i, line in enumerate(lines):
        draw.text((8, 5+i*18), line, fill=(255, 255, 255, 255))
    return np.asarray(image.convert('RGB'))


def read_case(root, case_id):
    directory = root/'cases'/f'case_{case_id:03d}'
    summary = json.loads((directory/'summary.json').read_text())
    case = summary['case']
    if case['case_id'] != case_id:
        raise ValueError(f'Case ID mismatch: {directory}')
    attempt = directory/f"attempt_{summary['attempt']:06d}"
    metadata = json.loads((attempt/'attempt.json').read_text())
    if metadata['case'] != case or metadata['attempt'] != summary['attempt']:
        raise ValueError(f'Attempt metadata mismatch: {attempt}')
    rows = read_transitions(attempt)
    trace = json.loads((attempt/'phase_trace.json').read_text())
    if len(rows) != summary['control_ticks'] or len(trace) != len(rows) or not rows:
        raise ValueError(f'Inconsistent or empty recorded control ticks: {attempt}')
    for i, (row, feedback) in enumerate(zip(rows, trace), 1):
        action = np.asarray(row['full_controller_action'], dtype=np.float32)
        if (row['episode_id'] != 1 or row['step'] != i or feedback['step'] != i or
                action.shape != (7,) or not np.isfinite(action).all()):
            raise ValueError(f'Invalid recorded step {i}: {attempt}')
    return case, summary, metadata, rows, trace


def render_case(root, output, manifest, case_id):
    case, summary, attempt, rows, trace = read_case(root, case_id)
    seed = case['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    config = OmegaConf.create(attempt['environment'])
    if config.sim_backend != 'cpu':
        raise ValueError('GPU trajectory replay is unsupported: preserve the recorded backend; CPU replay cannot verify GPU physics')
    if (str(config.action_mode) != 'ee_delta' or not config.headless or config.sim_backend != 'cpu' or
            config.sim.sim_freq != 100 or config.sim.control_freq != FPS or
            config.render_mode is not None or not config.centered_initial_tcp):
        raise ValueError('Recorded environment does not match the original 100/20 Hz EE setup')
    final_video = output/f'case_{case_id:03d}.mp4'
    partial_video = output/f'.case_{case_id:03d}.partial.mp4'
    poster = output/f'case_{case_id:03d}.png'
    env = writer = None
    initial_frame = None
    try:
        env = build_env(config)
        env.reset(seed=seed)
        encoder = CabinetSkillStateEncoder(env, latched_gripper=True)
        if encoder.contract() != manifest['state_contract']:
            raise ValueError('Replay encoder contract differs from dataset')
        env.reset(seed=seed)
        encoder.gripper_command = 1.0
        initial = encoder.encode()
        max_error = verify_state(initial, rows[0]['state'], 'initial')
        raw = env.unwrapped
        writer = imageio.get_writer(str(partial_video), fps=FPS, macro_block_size=1)
        initial_frame = overlay(raw.render_rgb_array(), label_lines(case, summary, 0, None, initial, initial))
        writer.append_data(initial_frame)
        last_frame = initial_frame
        for i, (row, feedback) in enumerate(zip(rows, trace), 1):
            max_error = max(max_error, verify_state(encoder.encode(), row['state'], f'step {i} before'))
            raw.step(np.asarray(row['full_controller_action'], dtype=np.float32))
            encoder.gripper_command = float(row['next_state'][32])
            actual = encoder.encode()
            max_error = max(max_error, verify_state(actual, row['next_state'], f'step {i} after'))
            last_frame = overlay(raw.render_rgb_array(), label_lines(case, summary, i, feedback, actual, initial))
            writer.append_data(last_frame)
        for _ in range(HOLD_FRAMES):
            writer.append_data(last_frame)
        writer.close()
        writer = None
        Image.fromarray(initial_frame).save(poster)
        os.replace(partial_video, final_video)
        return dict(case_id=case_id, object=case['object'], template=case['template'], seed=seed,
                    attempt=summary['attempt'], video=final_video.name, poster=poster.name,
                    success=summary['success'], failure=summary['report']['failure'],
                    control_ticks=len(rows), fps=FPS, replay_seconds=len(rows)/FPS,
                    hold_seconds=HOLD_FRAMES/FPS, video_seconds=(len(rows)+1+HOLD_FRAMES)/FPS,
                    verified_states=1+2*len(rows), max_abs_state_error=max_error, summary=summary)
    finally:
        if writer is not None:
            writer.close()
        if partial_video.exists():
            partial_video.unlink()
        if env is not None:
            env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', type=int, nargs='+', default=[2, 0, 272, 542, 486, 630])
    args = parser.parse_args(argv)
    if len(set(args.cases)) != len(args.cases) or any(i < 0 for i in args.cases):
        raise ValueError('Case IDs must be distinct and nonnegative')
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads((args.input/'manifest.json').read_text())
    if manifest.get('protocol') != 'paired_axis_order_and_grasp_insertion_v1':
        raise ValueError('Expected paired cabinet grasp sequences dataset')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    result = dict(input=str(args.input.resolve()), frames_per_second=FPS,
                  playback='recorded full controller actions; every state verified exactly', cases=[])
    for case_id in args.cases:
        result['cases'].append(render_case(args.input, args.output, manifest, case_id))
        (args.output/'replay_manifest.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
