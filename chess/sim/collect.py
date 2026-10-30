#!/usr/bin/env python3
"""Expert dataset collection: the scripted IK expert plays a seeded move chain in plain MuJoCo
and writes a marked LeRobotDataset with the same schema, marks, grades and sidecars as the
real teleoperated datasets.

Physics runs in a worker pool (one whole episode per task; a worker ships back only the
qpos trace and the commanded wire actions). The main process replays each trace through
the scene, renders both cameras, and writes frames in episode order. Ground truth
(sim.qpos) is stored per frame for replay; marks come from truth.subtask_bounds and the
grade from truth.outcome — failed episodes are kept and graded (the funnel is data too)
but get no marks when they do not decompose.

  python -m chess.sim.collect king_reach --n 500 --seed 1000 --style reach
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

from arm import mapping as J
from chess import task as T
from chess.sim import scene, truth
from chess.sim.expert import TICK_HZ
from chess.sim.record import Recorder, dataset_root
from chess.real import outcomes
from chess.real.eval_schedule import generate

EPISODE_CAP_S = 90.0     # control-time cap per episode (move + home tail; a stuck plan ends here)

_W = {}


def _init_worker():
    import mujoco
    from chess.sim import expert as X
    from chess.real import phases
    model, meta = scene.build(cam_w=64, cam_h=36)   # physics only; rendering is the main process's
    _W.update(model=model, meta=meta, mujoco=mujoco,
              planner=phases.Planner(scene.BOARD_POSE_JSON,
                                     pose_ok=X.self_clearance(model, meta)),
              substeps=max(1, int(round((1.0 / TICK_HZ) / model.opt.timestep))))


def _run_episode(args):
    """One whole episode's physics in a pool worker: reset, drive the expert (home tail
    included), return the qpos trace + the commanded wire actions. A `style` other than
    "expert" is drawn from (seed, trial): the same chain seed reproduces the demos."""
    tr, piece, style, seed = args
    from chess.sim import expert as X
    st = X.make_style(style, np.random.default_rng([int(seed), int(tr["i"])]))
    mujoco, model, meta = _W["mujoco"], _W["model"], _W["meta"]
    d = mujoco.MjData(model)
    d.qpos[:] = meta["qpos_ready"]
    k = meta["king_qposadr"]
    src, tgt = T.cell_from_name(tr["source"]), T.cell_from_name(tr["target"])
    x, y = scene.cell_world(src, meta["board_pose"], tr["dx_mm"] / 1000.0, tr["dy_mm"] / 1000.0)
    d.qpos[k:k + 7] = [x, y, T.BOARD_TOP + 0.001, 1, 0, 0, 0]
    mujoco.mj_forward(model, d)
    arm_c = meta["arm_ctrladr"]
    qs, acts, end = [], [], "done"
    try:
        drv = X.ExpertDriver(_W["planner"], src, tgt, tr["dx_mm"], tr["dy_mm"], meta,
                             piece, home_tail=True,
                             piece_xy=lambda: (float(d.qpos[k]), float(d.qpos[k + 1])),
                             style=st)
        for _ in range(int(EPISODE_CAP_S * TICK_HZ)):
            wire = drv.next_wire(d.qpos)
            if wire is None:
                break
            d.ctrl[arm_c] = J.wire_to_qpos(wire)
            for _ in range(_W["substeps"]):
                mujoco.mj_step(model, d)
            qs.append(np.asarray(d.qpos, np.float32).copy())
            acts.append(np.asarray(wire, np.float32).copy())
        else:
            end = "timeout"
    except RuntimeError as e:                       # unreachable plan: keep + grade the trace
        end = "error"
        print(f"episode {tr['i']} {tr['source']}->{tr['target']}: {e}", flush=True)
    if not qs:
        qs.append(np.asarray(d.qpos, np.float32).copy())
        acts.append(np.zeros(6, np.float32))
    return {"i": tr["i"], "source": tr["source"], "target": tr["target"],
            "dx_mm": tr["dx_mm"], "dy_mm": tr["dy_mm"], "end": end,
            "qpos": np.stack(qs), "act": np.stack(acts)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", help="dataset name (<namespace>/<name>, or a bare name under local/)")
    ap.add_argument("--n", type=int, default=500, help="episodes to collect")
    ap.add_argument("--seed", type=int, default=1000,
                    help="move-chain + offset seed. NOT 0: that is the evaluation schedules' "
                         "seed (schedules/), and collection must not replay the evaluation's "
                         "exact move/offset sequence")
    ap.add_argument("--centered", action="store_true",
                    help="place the king at exact cell centres (default: uniform offsets "
                         "within OFFSET_MAX_MM)")
    ap.add_argument("--start", default="e4", help="first source square of the move chain")
    ap.add_argument("--cam-w", type=int, default=scene.CAM_W)
    ap.add_argument("--cam-h", type=int, default=scene.CAM_H)
    ap.add_argument("--workers", type=int, default=16, help="physics pool size")
    ap.add_argument("--root", default=None,
                    help="dataset root dir (default $HF_LEROBOT_HOME/<repo id>)")
    ap.add_argument("--style", default="reach", choices=["expert", "reach"],
                    help="reach: the randomized expert of the paper (chess.sim.human, a person "
                         "as a noisy controller); expert: the deterministic scripted move")
    a = ap.parse_args()

    _, root = dataset_root(a.name, a.root)
    if (root / "meta" / "info.json").exists():
        raise SystemExit(f"dataset already exists at {root} — pick a new name")

    sched = generate("easy" if a.centered else "hard", a.n, a.seed, a.start,
                     scene.BOARD_POSE_JSON)
    trials = sched["trials"][: a.n]
    piece = "king"

    fps = int(round(TICK_HZ))
    rec = Recorder(a.name, fps, a.cam_w, a.cam_h,
                   {"args": {"n": a.n, "seed": a.seed, "centered": a.centered, "style": a.style,
                             "cam": [a.cam_w, a.cam_h]}}, root=a.root)

    import multiprocessing as mp
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.monotonic()
    print(f"{a.n} expert episodes -> {root}  ({a.cam_w}x{a.cam_h} @ {fps} fps, "
          f"seed {a.seed}, {'centered' if a.centered else 'random offsets'}, style {a.style})",
          flush=True)
    # spawn, not fork: the main process holds an EGL renderer context by now, and forked
    # children inheriting it die at startup (BrokenProcessPool with no worker traceback)
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker,
                             mp_context=mp.get_context("spawn")) as pool:
        for ti, r in enumerate(pool.map(_run_episode,
                                        ((tr, piece, a.style, a.seed) for tr in trials),
                                        chunksize=1)):
            res = rec.write(r, piece)
            if res is None:
                continue
            if (ti + 1) % 10 == 0 or ti + 1 == len(trials):
                print(f"{ti + 1}/{len(trials)} trials, {rec.ep} episodes ({rec.marked} marked; "
                      f"last: {r['source']}->{r['target']} = {truth.label(res)})",
                      flush=True)
    print("\n" + rec.close(), flush=True)
    info = json.loads((root / "meta" / "info.json").read_text())
    grades = outcomes.load(root).get("outcomes") or {}
    print(f"done in {time.monotonic() - t0:.0f}s: {info['total_episodes']} episodes, "
          f"{info['total_frames']} frames, {rec.marked} marked")
    print(f"milestones {'·'.join(f'{m} {rec.counts[m]}' for m in truth.MILESTONES)}   "
          f"failures {'·'.join(f'{f} {rec.counts[f]}' for f in truth.FAILURES)}   "
          f"good {sum(g == 'good' for g in grades.values())}  "
          f"mid {sum(g == 'mid' for g in grades.values())}  "
          f"bad {sum(g == 'bad' for g in grades.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
