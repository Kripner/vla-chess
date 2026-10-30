#!/usr/bin/env python3
"""Closed-loop evaluation in the simulator: a policy checkpoint (or the scripted expert) runs a
fixed schedule of moves, and every trial is graded from the ground-truth trace with the same
code that grades the collected demonstrations (chess.sim.truth).

Policy runs step E plain-MuJoCo worlds in lockstep in this process; the pi0.5 forward, the
cost, batches across them. The control loop is the deployment loop: chunks of `--nact`
actions, a prefetch with real-time chunking (RTC) `--rtc-delay` ticks before the queue
drains, and the stage switcher scoring {active, next} once per plan. Expert runs use a
process pool, one whole trial per task.

  python -m chess.sim.eval_e2e --ckpt <pretrained_model> --data <train_dataset> \\
      --schedule schedules/hard_v1.json --k 1000 --envs 16 --out evals/my_eval
  python -m chess.sim.eval_e2e --expert --style reach --schedule schedules/hard_v1.json --k 100
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from arm import mapping as J
from chess import task as T
from chess.sim import scene, truth

TRIAL_TIMEOUT_S = 60.0   # whole-episode timeout, control time (no per-stage timeouts).
                         # 60, not 30: near grasp columns (phases.NEAR_TANGENT_R) add a paced
                         # -90 deg wrist spin per end — a both-ends-near move legitimately runs
                         # ~42 s, and a grasp retry adds a few more.
SWITCH_WINS = 2          # consecutive wins for `next` before advancing
HOME_SETTLE_S = 0.5      # arm still this long within HOME_TOL_DEG of READY -> the trial is done
HOME_TOL_DEG = 2.0       # per-joint cal-deg

# The outcome scale and every threshold behind it live in chess.sim.truth — the same
# code that grades collected episodes, so an eval trial and a demo are judged alike.


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return ((c - h) / d, (c + h) / d)


class WorldState:
    """Per-world trial bookkeeping. Grading reads the trace, not incremental flags: the
    verdict is then a pure function of what happened, identical to the collector's."""

    def __init__(self):
        self.trial = None            # schedule entry or None (drained)
        self.ticks = 0
        self.t_start = 0.0
        self.subtask_i = 0
        self.wins = 0
        self.away = False            # plain (no-sub-task) policies: the arm has left READY once
        self.trace = []              # qpos rows, one per control tick
        self.end = None              # done (back at READY after the home switch) | timeout | toppled
        # Deploy-loop state. These MUST live here: reset_world re-runs __init__ when a world
        # takes its next trial, and a prefetch still in flight from the previous trial would
        # otherwise arrive on the old clock (or never), and the world would never plan again.
        self.queue = []              # actions still to execute from the current chunk
        self.n_last = 0              # length of the chunk the queue came from
        self.prev_raw = None         # that chunk, raw, for RTC's leftover
        self.inflight = None         # (arrival tick, raw, actions, queue length at fire)


def _home_reached_fn(smeta, control_hz: float):
    """The five arm joints within HOME_TOL_DEG of READY, in contract cal-deg, for HOME_SETTLE_S.
    The demonstrations end on the same event, so a trial ends where they end: the policy drives
    itself home under the `home` stage after its place->home switch, and the king is graded at
    rest with the jaw long gone."""
    arm_q = list(smeta["arm_qposadr"])
    ready = J.deg_from_qpos(np.asarray(smeta["qpos_ready"], float)[arm_q])[:5]
    n = max(1, int(round(HOME_SETTLE_S * control_hz)))

    def home_reached(w: WorldState) -> bool:
        if len(w.trace) < n:
            return False
        q = np.asarray(w.trace[-n:], float)[:, arm_q]
        return bool(np.all(np.abs(J.deg_from_qpos(q)[:, :5] - ready) <= HOME_TOL_DEG))
    return home_reached


def alloc_eval_dir(out: str | None) -> Path:
    if out:
        p = Path(out)
        p.mkdir(parents=True, exist_ok=True)
        return p
    evals = Path("evals")
    evals.mkdir(parents=True, exist_ok=True)
    n = 1
    while (evals / f"E{n:03d}").exists():
        n += 1
    p = evals / f"E{n:03d}"
    p.mkdir()
    return p


def _write_results(a, sched, trials, rows, out_dir, model, control_hz, t_wall,
                   ckpt=None, envs=None):
    """trials.csv + meta.json + the ready-to-paste registry row (both run paths)."""
    n = len(rows)
    milestones = {m: sum(int(r[m]) for r in rows) for m in truth.MILESTONES}
    fails = {f: sum(int(r[f]) for r in rows) for f in truth.FAILURES}
    strict = sum(int(r["success"]) for r in rows)
    lenient = sum(int(r["lenient"]) for r in rows)
    s_lo, s_hi = wilson(strict, n)
    l_lo, l_hi = wilson(lenient, n)
    ok_s = sorted(float(r["duration_s"]) for r in rows if int(r["success"]))
    success_s = (ok_s[len(ok_s) // 2] if ok_s else None)     # median length of a success
    with (out_dir / "trials.csv").open("w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(sorted(rows, key=lambda r: r["trial"]))
    (out_dir / "meta.json").write_text(json.dumps({
        "eval": out_dir.name, "date": time.strftime("%Y-%m-%d"),
        "ckpt": (ckpt or model), "data": a.data,
        "schedule": sched["id"],
        "schedule_sha256": hashlib.sha256(Path(a.schedule).read_bytes()).hexdigest()[:16],
        "k": len(trials), "envs": (envs if envs is not None else a.workers),
        "control_hz": control_hz, "timeout_s": a.timeout,
        "milestones": milestones, "failures": fails, "strict": [strict, n],
        "lenient": [lenient, n], "success_s": success_s,
        "wall_s": round(time.monotonic() - t_wall, 1)}, indent=1))
    print(f"\n{out_dir.name}: {model} on {sched['id']} k={n}\n"
          f"  success (lenient) {lenient}/{n} ({l_lo:.0%}-{l_hi:.0%})   "
          f"strict {strict}/{n} ({s_lo:.0%}-{s_hi:.0%})\n"
          f"  milestones {' · '.join(f'{m} {milestones[m]}' for m in truth.MILESTONES)}\n"
          f"  failures {' · '.join(f'{f} {fails[f]}' for f in truth.FAILURES)}", flush=True)
    return {"n": n, "strict": strict, "lenient": lenient, "milestones": milestones,
            "timeouts": fails["timed_out"], "toppled": fails["toppled"],
            "success_s": success_s}


def _trial_row(tr, piece, res, duration_s, end) -> dict:
    """One trials.csv row: the schedule entry, how it ended, the milestone ladder, the
    failure modes, the derived grade and where the piece came to rest."""
    off = res["final_off_mm"]
    return {"trial": tr["i"], "source": tr["source"], "target": tr["target"],
            "dx_mm": tr["dx_mm"], "dy_mm": tr["dy_mm"],
            "task": T.move_task(T.cell_from_name(tr["source"]), T.cell_from_name(tr["target"]), piece),
            "duration_s": duration_s, "end": end,
            **{m: int(res[m]) for m in truth.MILESTONES},
            **{f: int(res[f]) for f in truth.FAILURES},
            "success": int(res["success"]), "lenient": int(res["lenient"]),
            "label": truth.label(res), "n_grasps": res["n_grasps"],
            "final_cell": res["final_cell"],
            "final_off_x_mm": round(off[0], 2) if off else "",
            "final_off_y_mm": round(off[1], 2) if off else ""}


def _use_ckpt_stats(pm: Path, pre, post) -> None:
    """Normalize with the stats the checkpoint was TRAINED with. core.build_policy overrides the
    saved normalizer stats with those of `--data`; for a checkpoint trained on another dataset
    that mis-scales states and actions by a few percent of each joint's range. When the two
    agree this is a no-op."""
    from safetensors.torch import load_file
    for pipe, cfg_name in ((pre, "policy_preprocessor.json"), (post, "policy_postprocessor.json")):
        cfg = pm / cfg_name
        step = next((s for s in pipe.steps if hasattr(s, "_tensor_stats")), None)
        if step is None or not cfg.exists():
            continue
        files = [s.get("state_file") for s in json.loads(cfg.read_text())["steps"]
                 if s.get("registry_name") in ("normalizer_processor", "unnormalizer_processor")]
        if not files or not files[0] or not (pm / files[0]).exists():
            continue
        saved, cur = load_file(str(pm / files[0])), step.state_dict()
        diff = max((float((saved[k].float() - cur[k].float().reshape(saved[k].shape)).abs().max())
                    for k in saved if k in cur and cur[k].numel() == saved[k].numel()
                    and "images" not in k and not k.endswith(".count")), default=0.0)
        if diff > 1e-3:
            step._stats_explicitly_provided = False   # lerobot keeps explicit stats otherwise
            step.load_state_dict(saved)
            print(f"{cfg_name}: the checkpoint's normalization stats differ from --data's "
                  f"(max |diff| {diff:.3g}) — using the checkpoint's", flush=True)


def _load_policy(a):
    """Load the stage-conditioned pi0.5 checkpoint + its preprocessors."""
    import os

    from chess.real import outcomes
    from policy import core, pi05_subtask
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    meta = LeRobotDatasetMetadata(outcomes.repo_id(a.data), root=str(outcomes.dataset_root(a.data)))
    pm = Path(os.path.expanduser(a.ckpt))
    if (pm / "checkpoints" / "last").exists():      # a training run dir (finetune_real --out)
        pm = pm / "checkpoints" / "last"
    if (pm / "pretrained_model").exists():
        pm = pm / "pretrained_model"
    policy, pre, post = core.build_policy(meta, device="cpu" if a.offload else a.device,
                                          nact=a.nact, pretrained=str(pm), use_overhead=True,
                                          dtype=a.dtype)
    _use_ckpt_stats(pm, pre, post)
    # A plain policy (another VLA family: pi0, SmolVLA) has no stage sequence: it sees the task
    # text alone, as in its training, and its trial ends when the arm has left READY and settled
    # back there (the demonstrations' end), or times out.
    subtask_on = pi05_subtask.install(policy)
    if not subtask_on:
        print(f"{policy.config.type}: no sub-task sequence — plain task text, trial ends at "
              "READY after leaving it", flush=True)
    if a.steps:
        policy.config.num_inference_steps = int(a.steps)
    if a.offload:
        _offload_to(policy, pre, a.device)
    if getattr(a, "rtc_delay", 0) > 0 and a.rtc_guidance > 0:
        # the deploy setting; --rtc-guidance 0 keeps the latency emulation (prefetch + splice)
        # but samples every chunk unguided
        core.enable_rtc(policy, horizon=a.rtc_horizon or core.RTC_EXEC_HORIZON,
                        guidance=a.rtc_guidance)
    sub_tok = pi05_subtask.find_tokenizer(pre) if subtask_on else None
    from policy import control_rate
    control_hz = control_rate.load(pm) or 15.0
    return policy, pre, post, sub_tok, control_hz, pm


def _offload_to(policy, pre, device: str) -> None:
    """Run a pi0.5 checkpoint on a GPU too small to hold it (a 4 GB laptop card): the
    weights stay in host RAM and accelerate's hooks stream each module's weights to `device`
    for its forward, so peak GPU memory is one module (the 1 GB embedding) plus activations.
    Batches go straight to the GPU, and
    the policy's own device lookup (its first parameter, see PI05Policy._preprocess_images)
    is anchored there so masks and noise are built where the hooked modules compute."""
    import collections

    import torch
    from accelerate import cpu_offload
    from lerobot.processor.device_processor import DeviceProcessorStep

    cpu_offload(policy.model, execution_device=torch.device(device))
    anchor = torch.nn.Parameter(torch.zeros(1, device=device), requires_grad=False)
    policy._parameters = collections.OrderedDict(                     # noqa: SLF001
        [("_offload_anchor", anchor), *policy._parameters.items()])   # noqa: SLF001
    for step in pre.steps:
        if isinstance(step, DeviceProcessorStep):
            step.device = device
            step.__post_init__()
    print(f"offload: weights in RAM, compute on {device}", flush=True)


def _policy_mujoco(a, sched, trials, piece, loaded=None, model_label=None,
                   part=(0, 1)) -> dict:
    """Policy eval: E plain-MuJoCo worlds step in lockstep in this process (physics is cheap;
    the pi0.5 forward is the cost and it batches across worlds), cameras render per plan
    through mujoco.Renderer.
    `part=(i, n)`: this process is one of n running disjoint parts of a split rollout (the
    trainer's DDP ranks): it writes nothing and returns {"rows"} for the caller to merge; only
    part 0 prints progress. (0, 1) is the whole eval, written here."""
    import os

    import torch

    os.environ.setdefault("MUJOCO_GL", "egl")   # mujoco.Renderer needs a GL platform
    dev = torch.device(a.device)
    if dev.type == "cuda" and dev.index is not None:
        # render on this process's own GPU: under DDP every rank rolls out, and mujoco's EGL
        # otherwise opens the first device for all of them (rank 0's card, the fullest one)
        os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", str(dev.index))
    import mujoco

    from chess.real import segments
    from policy import pi05_subtask

    i_part, n_part = part
    verbose, write = i_part == 0, n_part == 1
    policy, pre, post, sub_tok, control_hz, pm = loaded or _load_policy(a)
    nact = policy.config.n_action_steps
    rc = getattr(policy.config, "rtc_config", None)
    if verbose:
        print(f"ckpt {pm}  control {control_hz:g} Hz  chunk {nact}  "
              f"envs {a.envs}{' x %d parts' % n_part if n_part > 1 else ''}  rtc-delay {a.rtc_delay}  "
              f"rtc {'guidance %g horizon %d' % (rc.max_guidance_weight, rc.execution_horizon) if rc else 'off'}  "
              f"steps {getattr(policy.config, 'num_inference_steps', '-')}  "
              f"dtype {getattr(policy.config, 'dtype', '-')}", flush=True)
    model, smeta = scene.build(cam_w=a.cam_w, cam_h=a.cam_h)
    renderer = mujoco.Renderer(model, height=a.cam_h, width=a.cam_w)
    substeps = max(1, int(round((1.0 / control_hz) / model.opt.timestep)))
    arm_q, arm_c = smeta["arm_qposadr"], smeta["arm_ctrladr"]
    king_q = smeta["king_qposadr"]
    E = a.envs
    datas = [mujoco.MjData(model) for _ in range(E)]
    worlds = [WorldState() for _ in range(E)]
    queue = list(trials)
    rows = []
    out_dir = alloc_eval_dir(a.out) if write else None
    seq_len = len(segments.E2E_SEQUENCE)
    max_ticks = int(a.timeout * control_hz)
    home_reached = _home_reached_fn(smeta, control_hz)
    settle = max(1, int(round(HOME_SETTLE_S * control_hz)))   # home_reached's window
    t_wall = time.monotonic()
    def reset_world(i: int):
        w, d = worlds[i], datas[i]
        w.__init__()
        if not queue:
            w.trial = None
            return
        tr = queue.pop(0)
        w.trial = tr
        d.qpos[:] = smeta["qpos_ready"]
        src = T.cell_from_name(tr["source"])
        x, y = scene.cell_world(src, smeta["board_pose"], tr["dx_mm"] / 1000.0,
                                tr["dy_mm"] / 1000.0)
        d.qpos[king_q:king_q + 7] = [x, y, T.BOARD_TOP + 0.001, 1, 0, 0, 0]
        d.qvel[:] = 0.0
        import mujoco as _mj
        _mj.mj_forward(model, d)
        d.ctrl[arm_c] = d.qpos[arm_q]

    def finish(i: int, end: str):
        w = worlds[i]
        tr = w.trial
        sig = truth.Signals(np.asarray(w.trace), smeta, piece)
        res = truth.outcome(sig, T.cell_from_name(tr["source"]),
                            T.cell_from_name(tr["target"]), end)
        rows.append(_trial_row(tr, piece, res, round(w.ticks / control_hz, 2), end))
        if verbose and (len(rows) % 10 == 0 or not queue):
            print(f"{len(rows)}/{len(trials)} trials done "
                  f"(last: {tr['source']}->{tr['target']} = {truth.label(res)})", flush=True)
        reset_world(i)

    def build_batch(active_idx):
        imgs = {"wrist": [], "overhead": []}
        states = []
        for i in active_idx:
            d = datas[i]
            for key, cam in (("wrist", scene.WRIST_CAM), ("overhead", scene.OVERHEAD_CAM)):
                renderer.update_scene(d, camera=cam)
                imgs[key].append(renderer.render())
            # the policy's state is contract cal-deg on all six channels (see arm.mapping)
            states.append(J.deg_from_qpos(np.asarray(d.qpos[arm_q], float)))
        batch = {"observation.images.wrist":
                     torch.as_tensor(np.stack(imgs["wrist"]), dtype=torch.float32)
                     .permute(0, 3, 1, 2) / 255.0,
                 "observation.images.overhead":
                     torch.as_tensor(np.stack(imgs["overhead"]), dtype=torch.float32)
                     .permute(0, 3, 1, 2) / 255.0,
                 "observation.state": torch.as_tensor(np.stack(states), dtype=torch.float32),
                 "task": [T.move_task(T.cell_from_name(worlds[i].trial["source"]),
                                      T.cell_from_name(worlds[i].trial["target"]), piece)
                          for i in active_idx]}
        return pre(batch)

    for i in range(E):
        reset_world(i)

    # The deployment control loop, tick by tick (RTC on): a world executes one queued action
    # per tick. With D = --rtc-delay > 0, a prefetch fires
    # whenever nothing is in flight and anything is queued, from the observation at fire
    # time, with consumed = n_last - len(queue) and the previous raw chunk's leftover for
    # RTC (frozen prefix = min(D, what is left)); it "arrives" D ticks later and replaces the
    # queue minus the actions executed meanwhile (the client's chunk[k_exec:]); a queue that
    # drains first holds the last target until it lands (the client's underrun hold). A world
    # with nothing queued and nothing in flight plans fresh without RTC (trial start). D = 0
    # is plain chunking: fresh plan, execute nact, replan. Stage decisions happen once per
    # plan. A "more than D queued" fire rule instead degenerates at D >= nact/2 into an
    # unguided fresh plan every chunk.
    D = int(a.rtc_delay)

    def decide_subtasks(batch, idx):
        """Score {active, next} for the worlds in `idx` on this batch; advance on SWITCH_WINS."""
        cand_rows, cand_map = [], []
        for j, i in enumerate(idx):
            w = worlds[i]
            if w.subtask_i < seq_len - 1:
                cand_rows += [segments.E2E_SEQUENCE[w.subtask_i],
                              segments.E2E_SEQUENCE[w.subtask_i + 1]]
                cand_map.append((j, i))
        if not cand_rows or sub_tok is None:
            return
        ct, cm = pi05_subtask.label_tokens(sub_tok, cand_rows)
        dev = batch["observation.state"].device
        sel = torch.as_tensor([j for j, _ in cand_map], device=dev)
        sub_batch = {k: (v[sel] if torch.is_tensor(v)
                         else [v[j] for j, _ in cand_map] if isinstance(v, (list, tuple))
                         else v)
                     for k, v in batch.items()}
        lp = policy.score_subtasks(sub_batch, ct.to(dev), cm.to(dev), n_cands=2)
        for r, (j, i) in enumerate(cand_map):
            w = worlds[i]
            if float(lp[r, 1]) > float(lp[r, 0]):
                w.wins += 1
                if w.wins >= SWITCH_WINS:
                    w.subtask_i += 1
                    w.wins = 0
            else:
                w.wins = 0

    def plan(idx, left=None, delay=0):
        """One batched replan for worlds `idx`: observe now, decide the sub-task, sample the
        chunk -> {world: (raw full chunk (1, chunk_size, A), actions (nact, 6))}. `left` = {world: previous raw
        chunk's leftover (1, T, A)} for RTC. A world whose switch has reached `home` keeps
        planning under that label: the policy drives itself back to READY (see home_reached)."""
        batch = build_batch(idx)
        decide_subtasks(batch, idx)
        subs = [segments.E2E_SEQUENCE[worlds[i].subtask_i] for i in idx]
        if sub_tok is not None:
            pi05_subtask.attach_subtask(batch, subs, sub_tok)
        kw = {}
        if left and rc is not None:
            lo = torch.cat([left[i] for i in idx], dim=0)
            kw = {"prev_chunk_left_over": lo, "inference_delay": max(0, min(delay, lo.shape[1]))}
        with torch.no_grad():
            full = policy.predict_action_chunk(batch, **kw)              # (B, chunk_size, A)
        # execute nact; .float(): a bfloat16 family's postprocessor (pi0) hands back bf16,
        # which numpy has no dtype for
        acts = post(full[:, :nact]).float().to("cpu").numpy()
        # the WHOLE chunk is the old plan RTC guides toward: a leftover cut at nact ends at the
        # seam and leaves the post-delay steps unguided
        return {i: (full[b:b + 1].detach().clone(), acts[b]) for b, i in enumerate(idx)}

    while any(w.trial is not None for w in worlds):
        active = [i for i in range(E) if worlds[i].trial is not None]
        # 1. prefetches that arrive this tick: splice, skipping what ran while in flight
        for i in active:
            w = worlds[i]
            if w.inflight is not None and w.ticks >= w.inflight[0]:
                _, raw, acts, qlen = w.inflight
                k_exec = qlen - len(w.queue)
                w.queue, w.n_last, w.prev_raw, w.inflight = list(acts[k_exec:]), nact, raw, None
        # 2. fresh synchronous plans for drained worlds (no RTC — the client's post-drain path)
        fresh = [i for i in active if not worlds[i].queue and worlds[i].inflight is None]
        if fresh:
            for i, (raw, acts) in plan(fresh).items():
                w = worlds[i]
                w.queue, w.n_last, w.prev_raw = list(acts), nact, raw
        active = [i for i in range(E) if worlds[i].trial is not None]
        # 3. prefetches: nothing in flight and anything queued; the frozen prefix is
        #    min(D, queued). RTC needs one leftover length per batch: group by `consumed`
        if D > 0:
            groups = {}
            for i in active:
                w = worlds[i]
                if w.inflight is None and w.queue and w.prev_raw is not None:
                    groups.setdefault(w.n_last - len(w.queue), []).append(i)
            for consumed, idx in groups.items():
                left = {i: worlds[i].prev_raw[:, consumed:, :] for i in idx}
                for i, (raw, acts) in plan(idx, left, D).items():
                    w = worlds[i]
                    w.inflight = (w.ticks + D, raw, acts, len(w.queue))
        active = [i for i in range(E) if worlds[i].trial is not None]
        # 4. one control tick: pop an action per world (or hold the last target while a late
        #    prefetch is in flight), step physics
        for i in active:
            w = worlds[i]
            if w.queue:
                # policy actions are contract cal-deg on ALL six channels (see arm.mapping)
                q5, jaw = J.urdf_from_deg(w.queue.pop(0))
                datas[i].ctrl[arm_c] = np.concatenate([q5, [float(jaw)]])
            for _ in range(substeps):
                mujoco.mj_step(model, datas[i])
            w.trace.append(np.asarray(datas[i].qpos, np.float32).copy())
            w.ticks += 1
            if sub_tok is None and not w.away and w.ticks >= settle and not home_reached(w):
                w.away = True                   # plain policy: the arm has left READY
            if truth.up_cos(datas[i].qpos[king_q + 3:king_q + 7]) < truth.TOPPLED_COS:
                finish(i, "toppled")            # fallen: no higher rung is reachable
            elif (w.subtask_i == seq_len - 1 if sub_tok is not None else w.away) and home_reached(w):
                finish(i, "done")               # back at READY under the home label: the episode's end
            elif w.ticks >= max_ticks:
                finish(i, "timeout")

    if not write:
        return {"rows": rows}
    return _write_results(a, sched, trials, rows, out_dir,
                          model_label or f"{pm.parent.name}@{pm.name}",
                          control_hz, t_wall, ckpt=str(pm), envs=E)


def rollout(policy, pre, post, sub_tok, control_hz, *, schedule, k: int, envs: int, out,
            label: str, data: str, rtc_delay: int = 5, timeout: float = TRIAL_TIMEOUT_S,
            device: str = "cuda", part=(0, 1)) -> dict | None:
    """Closed-loop sim eval of an IN-MEMORY stage-conditioned pi0.5 policy, the trainer's
    periodic and final rollouts: the same worlds, deploy loop, grading and trials.csv/meta.json
    as the CLI. RTC is switched on for the rollout and the policy's config/mode restored after,
    so training and the saved checkpoint are untouched. Returns _write_results' summary.
    `part=(rank, world)` splits the k trials over the DDP ranks (trial i goes to rank
    i % world), each on its own GPU and worlds; the parts are gathered over the process
    group and rank 0 writes the merged eval and returns its summary, the others get None."""
    import types

    import torch

    from policy import core
    sched = json.loads(Path(schedule).read_text())
    i_part, n_part = part
    trials = sched["trials"][:k][i_part::n_part]
    a = types.SimpleNamespace(ckpt=label, data=data, schedule=str(schedule), k=k, envs=envs,
                              out=str(out), device=device, rtc_delay=int(rtc_delay),
                              timeout=timeout, offload=False, workers=None,
                              cam_w=scene.CAM_W, cam_h=scene.CAM_H)
    prev_rtc, was_training = policy.config.rtc_config, policy.training
    if rtc_delay > 0:
        core.enable_rtc(policy)
    policy.eval()
    t_wall = time.monotonic()
    try:
        with torch.no_grad():
            res = _policy_mujoco(a, sched, trials, sched.get("piece", "king"),
                                 loaded=(policy, pre, post, sub_tok, control_hz, Path(label)),
                                 model_label=label, part=part)
    finally:
        policy.config.rtc_config = prev_rtc
        policy.init_rtc_processor()                  # None again when prev_rtc is None
        policy.model.rtc_processor = policy.rtc_processor
        if was_training:
            policy.train()
    if n_part == 1:
        return res
    import torch.distributed as dist
    parts = [None] * n_part
    dist.all_gather_object(parts, res)
    if i_part != 0:
        return None
    rows = [r for p in parts for r in p["rows"]]
    out_dir = alloc_eval_dir(a.out)
    return _write_results(a, sched, sched["trials"][:k], rows, out_dir, label, control_hz,
                          t_wall, ckpt=label, envs=envs)


_W = {}                       # per-worker world (built once by _init_worker)


def _init_worker():
    import mujoco
    from chess.sim import expert as X
    from chess.real import phases
    model, smeta = scene.build(cam_w=64, cam_h=36)
    _W.update(model=model, meta=smeta, mujoco=mujoco,
              planner=phases.Planner(scene.BOARD_POSE_JSON,
                                     pose_ok=X.self_clearance(model, smeta)),
              substeps=max(1, int(round((1.0 / X.TICK_HZ) / model.opt.timestep))))


def _run_trial_mujoco(args):
    """One whole trial in a plain-MuJoCo world: reset, drive, grade. Runs in a pool worker.
    A `style` other than "expert" drives the collection's randomized expert (drawn from
    (seed, trial) as collect.py draws it): the way to score a style over a whole schedule
    without the collection's camera rendering."""
    tr, piece, timeout_s, style, seed, home_tail = args
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
    trace, end = [], "done"
    try:
        drv = X.ExpertDriver(_W["planner"], src, tgt, tr["dx_mm"], tr["dy_mm"], meta, piece,
                             piece_xy=lambda: (float(d.qpos[k]), float(d.qpos[k + 1])),
                             style=st, home_tail=home_tail)
        for _ in range(int(timeout_s * X.TICK_HZ)):
            wire = drv.next_wire(d.qpos)
            if wire is None:
                break
            d.ctrl[arm_c] = J.wire_to_qpos(wire)
            for _ in range(_W["substeps"]):
                mujoco.mj_step(model, d)
            trace.append(d.qpos.copy())
            if truth.up_cos(d.qpos[k + 3:k + 7]) < truth.TOPPLED_COS:
                end = "toppled"
                break
        else:
            end = "timeout"
    except RuntimeError as e:
        end = "error"
        print(f"trial {tr['i']} {tr['source']}->{tr['target']}: {e}", flush=True)
    if not trace:
        trace.append(d.qpos.copy())
    sig = truth.Signals(np.asarray(trace), meta, piece)
    res = truth.outcome(sig, src, tgt, end)
    return _trial_row(tr, piece, res, round(len(trace) / X.TICK_HZ, 2), end)


def _expert_mujoco(a, sched, trials, piece) -> dict:
    """Expert eval: a process pool of plain-MuJoCo worlds, one whole trial per task. Same
    schedule, grading and outputs as the policy path."""
    from concurrent.futures import ProcessPoolExecutor
    from chess.sim.expert import TICK_HZ
    out_dir = alloc_eval_dir(a.out)
    t_wall = time.monotonic()
    print(f"scripted IK expert  {a.workers} workers  control {TICK_HZ:g} Hz  style {a.style}"
          f"{f' (seed {a.style_seed})' if a.style != 'expert' else ''}", flush=True)
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init_worker) as ex:
        for r in ex.map(_run_trial_mujoco,
                        ((tr, piece, a.timeout, a.style, a.style_seed, a.home_tail) for tr in trials),
                        chunksize=4):
            rows.append(r)
            if len(rows) % 25 == 0 or len(rows) == len(trials):
                print(f"{len(rows)}/{len(trials)} trials done "
                      f"(last: {r['source']}->{r['target']} = {r['label']})", flush=True)
    return _write_results(a, sched, trials, rows, out_dir, "expert", TICK_HZ, t_wall)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", default=None, help="pretrained_model dir of the checkpoint")
    ap.add_argument("--data", default=None, help="training dataset (normalization stats)")
    ap.add_argument("--expert", action="store_true",
                    help="evaluate the scripted IK expert (no checkpoint, no rendering)")
    ap.add_argument("--schedule", required=True, help="a schedule (schedules/*.json)")
    ap.add_argument("--k", type=int, default=100, help="trials = the schedule's first k")
    ap.add_argument("--envs", type=int, default=16, help="policy: worlds stepped in lockstep")
    ap.add_argument("--workers", type=int, default=16, help="expert: process pool size")
    ap.add_argument("--timeout", type=float, default=TRIAL_TIMEOUT_S,
                    help="whole-trial timeout, control-time seconds")
    ap.add_argument("--out", default=None, help="eval dir (default: the next evals/E###)")
    ap.add_argument("--style", default="expert", choices=["expert", "reach"],
                    help="--expert only: 'reach' scores the randomized expert of the collection")
    ap.add_argument("--style-seed", type=int, default=0,
                    help="--style reach: the per-trial style draws come from (seed, trial)")
    ap.add_argument("--home-tail", action="store_true",
                    help="--expert only: drive back to READY after the release, as collection "
                         "records it (the grade is the same)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--rtc-delay", type=int, default=5,
                    help="inference latency in control ticks: a prefetch fires this many ticks "
                         "before the queue drains and splices in the RTC-guided chunk minus the "
                         "actions executed meanwhile. 0 = plain chunking (execute nact, replan)")
    ap.add_argument("--nact", type=int, default=10,
                    help="actions executed per chunk before the next plan")
    ap.add_argument("--steps", type=int, default=None,
                    help="flow-matching denoising steps (default: the checkpoint's, 10)")
    ap.add_argument("--dtype", default=None, choices=["bfloat16", "float32"],
                    help="inference dtype (default: the checkpoint's config)")
    ap.add_argument("--rtc-guidance", type=float, default=10.0,
                    help="RTC max guidance weight (core.enable_rtc); 0 = no RTC guidance "
                         "while keeping --rtc-delay's latency emulation")
    ap.add_argument("--rtc-horizon", type=int, default=None,
                    help="RTC execution horizon where the old-plan guidance reaches zero "
                         "(default core.RTC_EXEC_HORIZON)")
    ap.add_argument("--offload", action="store_true",
                    help="weights in host RAM, each module streamed to --device for its forward "
                         "(accelerate cpu_offload): runs pi0.5 on a 4 GB GPU, ~10x slower")
    ap.add_argument("--cam-w", type=int, default=scene.CAM_W)
    ap.add_argument("--cam-h", type=int, default=scene.CAM_H)
    a = ap.parse_args()
    if a.expert == (a.ckpt is not None):
        raise SystemExit("pass either --ckpt <dir> (with --data) or --expert")
    if a.ckpt is not None and a.data is None:
        raise SystemExit("--ckpt needs --data (the training dataset, for normalization)")

    sched = json.loads(Path(a.schedule).read_text())
    trials = sched["trials"][: a.k]
    piece = sched.get("piece", "king")
    if a.expert:
        _expert_mujoco(a, sched, trials, piece)
    else:
        _policy_mujoco(a, sched, trials, piece)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
