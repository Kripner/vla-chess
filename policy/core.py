#!/usr/bin/env python3
"""Training core for the chess VLA trainer (chess.train.finetune_real): policy construction,
optimizer/scheduler presets, the flow-matching step, checkpoint save/resume, in-process
validation and wandb logging.

lerobot is used as a library (SmolVLA/pi0/pi0.5 policies, processors, optimizer and scheduler
presets). Imported, not run.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BASE = "lerobot/pi05_base"


def combine_stats(metas):
    """Normalization stats of the concatenation of `metas`, frame-count weighted: mean/std pool
    exactly, min/max take extremes, count sums, quantiles average the stored per-dataset values
    (lerobot's quantiles are frame-weighted means of per-episode quantiles, so this reproduces
    them); an unhandled stat field raises."""
    import numpy as np
    keys = set.intersection(*[set(m.stats) for m in metas])   # only keys every meta has
    w = np.array([float(m.total_frames) for m in metas])
    out = {}
    for k in sorted(keys):
        per = [m.stats[k] for m in metas]
        if not all(isinstance(s, dict) and {"mean", "std"} <= set(s) for s in per):
            out[k] = per[0]                      # non-numeric stat block: keep the first as-is
            continue
        fields = set.intersection(*[set(s) for s in per])
        mu = np.stack([np.asarray(s["mean"], dtype=np.float64) for s in per])
        sd = np.stack([np.asarray(s["std"], dtype=np.float64) for s in per])
        ww = w.reshape(-1, *([1] * (mu.ndim - 1)))
        mean = (ww * mu).sum(0) / w.sum()
        var = (ww * (sd ** 2 + mu ** 2)).sum(0) / w.sum() - mean ** 2
        agg = {"mean": mean.astype(np.float32),
               "std": np.sqrt(np.maximum(var, 0.0)).astype(np.float32)}
        for edge, fn in (("min", np.minimum), ("max", np.maximum)):
            if edge in fields:
                acc = np.asarray(per[0][edge], dtype=np.float64)
                for s in per[1:]:
                    acc = fn(acc, np.asarray(s[edge], dtype=np.float64))
                agg[edge] = acc.astype(np.float32)
        if "count" in fields:
            agg["count"] = sum(np.asarray(s["count"]) for s in per)
        for q in sorted(f for f in fields if len(f) == 3 and f[0] == "q" and f[1:].isdigit()):
            qq = np.stack([np.asarray(s[q], dtype=np.float64) for s in per])
            wq = w.reshape(-1, *([1] * (qq.ndim - 1)))
            agg[q] = ((wq * qq).sum(0) / w.sum()).astype(np.float32)
        unhandled = fields - set(agg)
        if unhandled:
            raise ValueError(
                f"combine_stats: no rule for stat field(s) {sorted(unhandled)} of {k!r}. "
                "Dropping them silently changes normalization — add an aggregation rule.")
        out[k] = agg
    return out


def rename_map(policy_type: str = "smolvla", use_overhead: bool = False) -> dict:
    """Dataset camera keys -> the policy's image slots. SmolVLA: wrist -> camera2, overhead ->
    camera1 with use_overhead. pi0/pi0.5: wrist -> base_0_rgb, or with use_overhead overhead ->
    base_0_rgb and wrist -> left_wrist_0_rgb. A non-empty map lets lerobot pad unsupplied slots."""
    if policy_type in ("pi05", "pi0"):
        if use_overhead:
            # the fixed external view takes pi0.5's base/scene slot; right_wrist pads out
            return {"observation.images.overhead": "observation.images.base_0_rgb",
                    "observation.images.wrist": "observation.images.left_wrist_0_rgb"}
        return {"observation.images.wrist": "observation.images.base_0_rgb"}
    m = {"observation.images.wrist": "observation.images.camera2"}
    if use_overhead:
        m["observation.images.overhead"] = "observation.images.camera1"
    return m


# Real-dataset columns kept away from the policy: lerobot types every "observation.*" key as a
# state input, so observation.current (per-motor mA) would widen the train state while deploy
# feeds observation.state only. Columns without the prefix (control_source, ...) are skipped anyway.
NON_POLICY_COLS = ("observation.current",)


class _MetaView:
    """Dataset metadata with NON_POLICY_COLS hidden, everything else passed through."""

    def __init__(self, meta, drop=NON_POLICY_COLS):
        self._meta = meta
        self.features = {k: v for k, v in meta.features.items() if k not in drop}
        self.stats = meta.stats

    def __getattr__(self, name):
        return getattr(self._meta, name)


def build_policy(ds_meta, device: str = "cuda", nact: int = 10, pretrained: str = BASE,
                 expert_only: bool = True, dtype: str | None = None,
                 grad_ckpt: bool = False, use_overhead: bool = False,
                 vlm_init: str | None = None):
    """(policy, preprocessor, postprocessor) built like lerobot-train's fresh start: make_policy +
    make_pre_post_processors with the dataset's stats. `pretrained` is a base repo or a checkpoint's
    pretrained_model dir; the policy class follows its config type. expert_only freezes the VLM.
    `vlm_init` (pi0.5 only): NO robot pre-training — the policy takes only its ARCHITECTURE from
    `pretrained`, starts from random weights, and then loads the VLM (SigLIP + Gemma-2B) from
    that HF PaliGemma checkpoint (google/paligemma-3b-pt-224); the action expert, the state and
    action projections and the time embedding stay random: the model without robot pretraining
    of the paper's pretraining ablation."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    ds_meta = _MetaView(ds_meta)                           # analysis-only columns out
    cfg = PreTrainedConfig.from_pretrained(pretrained)
    cfg.pretrained_path = None if vlm_init else pretrained   # make_policy loads weights from here
    cfg.device = device
    cfg.n_action_steps = nact
    if hasattr(cfg, "train_expert_only"):
        cfg.train_expert_only = expert_only  # True freezes the VLM; False trains all params
    if dtype is not None and hasattr(cfg, "dtype"):
        cfg.dtype = dtype                  # pi0.5 loads/forwards in this; bfloat16 halves the footprint
    if grad_ckpt and hasattr(cfg, "gradient_checkpointing"):
        cfg.gradient_checkpointing = True  # recompute activations in backward to fit a bigger batch
    rn = rename_map(cfg.type, use_overhead)
    policy = make_policy(cfg=cfg, ds_meta=ds_meta, rename_map=rn)
    if vlm_init:
        if cfg.type != "pi05":
            raise SystemExit("--vlm-init is a pi0.5 option")
        _load_vlm_init(policy, vlm_init, device)
    if cfg.type in ("pi05", "pi0"):
        # The base's saved pipeline names a step ('relative_actions_processor') this lerobot
        # registers under another name, so from_pretrained KeyErrors: build the pipeline fresh
        # from the factory (its normalizer takes dataset_stats) and inject the rename map by hand.
        pre, post = make_pre_post_processors(policy_cfg=cfg, dataset_stats=ds_meta.stats)
        for step in pre.steps:
            if hasattr(step, "rename_map"):
                step.rename_map = rn
                break
        return policy, pre, post
    feats = {**policy.config.input_features, **policy.config.output_features}
    nm = policy.config.normalization_mapping
    pre, post = make_pre_post_processors(
        policy_cfg=cfg, pretrained_path=pretrained, dataset_stats=ds_meta.stats,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "normalizer_processor": {"stats": ds_meta.stats, "features": feats, "norm_map": nm},
            "rename_observations_processor": {"rename_map": rn}},
        postprocessor_overrides={
            "unnormalizer_processor": {"stats": ds_meta.stats,
                                       "features": policy.config.output_features, "norm_map": nm}})
    return policy, pre, post


def _load_vlm_init(policy, repo: str, device: str) -> None:
    """Copy an HF PaliGemma checkpoint into a randomly initialised pi0.5 policy's VLM
    (policy.model.paligemma_with_expert.paligemma subclasses transformers' PaliGemma, so the
    keys line up). Refuses if any checkpoint tensor finds no home or any VLM tensor stays
    random: a silent partial load would be a different experiment."""
    import torch
    from transformers import PaliGemmaForConditionalGeneration
    vlm = policy.model.paligemma_with_expert.paligemma
    dtype = next(vlm.parameters()).dtype
    hf = PaliGemmaForConditionalGeneration.from_pretrained(repo, dtype=dtype)
    sd = hf.state_dict()
    own_sd = vlm.state_dict()
    for k in ("lm_head.weight", "model.language_model.embed_tokens.weight"):
        # the HF checkpoint pads its vocabulary to 257 216 rows (TPU alignment); pi0.5's
        # config keeps the real 257 152, so drop the unused padding rows
        if k in sd and k in own_sd and sd[k].shape[0] > own_sd[k].shape[0]:
            sd[k] = sd[k][: own_sd[k].shape[0]]
    res = vlm.load_state_dict(sd, strict=False)
    own = set(own_sd.keys())
    missing = [k for k in res.missing_keys]                   # VLM tensors the checkpoint lacks
    if res.unexpected_keys or missing:
        raise SystemExit(f"--vlm-init {repo}: {len(res.unexpected_keys)} checkpoint tensor(s) "
                         f"without a home ({res.unexpected_keys[:5]}), {len(missing)} VLM "
                         f"tensor(s) not covered ({missing[:5]}) — key layout mismatch")
    n = sum(v.numel() for v in sd.values())
    print(f"vlm-init: {len(sd)} tensors ({n / 1e9:.2f}B params) of {repo} -> the VLM "
          f"({len(own)} tensors); expert, projections and time embedding stay random", flush=True)
    del hf, sd
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()


def trainable(policy, expert_only: bool = True):
    """(trainable params, n_trainable, n_total). With expert_only, warns when more than 60% of the
    params carry gradients, meaning the VLM freeze (train_expert_only) did not take."""
    tr = [p for _, p in policy.named_parameters() if p.requires_grad]
    n_tr = sum(p.numel() for p in tr)
    n_all = sum(p.numel() for p in policy.parameters())
    if expert_only and n_tr > 0.6 * n_all:
        print(f"  WARNING: {n_tr/n_all:.0%} of params are trainable — expected the VLM "
              f"frozen (expert-only); is train_expert_only set?", flush=True)
    return tr, n_tr, n_all


# RTC: the old-plan guidance decays to zero at this chunk step. 50 = the whole chunk: every
# chunk stays anchored to the previous plan through most of its length, which turns a jerky,
# over-fast policy into a calm one. In a sweep, horizon 10 and 25 and settings with less
# continuity (no RTC, a hard prefix only, nact 5) all scored well below horizon 50.
RTC_EXEC_HORIZON = 50


def enable_rtc(policy, horizon: int = RTC_EXEC_HORIZON, schedule: str = "linear",
               guidance: float = 10.0) -> bool:
    """Real-Time Chunking on a pi0.5 policy, the DEPLOY setting: the new chunk's first
    `inference_delay` steps are frozen to the previous chunk's leftover, the rest guided
    toward it with weight decaying to zero at `horizon` (see RTC_EXEC_HORIZON). Returns False,
    leaving the policy alone, for a family without RTC: its chunks are then sampled fresh."""
    if not policy.supports_rtc():
        print(f"RTC: {policy.config.type} has no real-time chunking — off", flush=True)
        return False
    # lerobot.configs.types, NOT lerobot.configs: the parent package is a bare namespace
    # package in some builds.
    from lerobot.configs.types import RTCAttentionSchedule
    from lerobot.policies.rtc.configuration_rtc import RTCConfig
    policy.config.rtc_config = RTCConfig(
        enabled=True, execution_horizon=horizon,
        prefix_attention_schedule=RTCAttentionSchedule(schedule.upper()),   # enum is UPPER
        max_guidance_weight=guidance)
    if hasattr(policy, "init_rtc_processor"):
        policy.init_rtc_processor()
    return True


def build_optim(policy, steps: int | None = None, lr: float | None = None,
                grad_clip: float | None = None, scheduler: bool = True,
                slow_params: list | None = None, slow_scale: float = 0.1,
                constant: bool = False, decay_steps: int | None = None, floor: float = 0.1,
                warmup: int | None = None):
    """AdamW (+ cosine-warmup scheduler) from the policy's own lerobot presets; `lr`, `grad_clip`,
    `scheduler` override. Returns (optimizer, scheduler_or_None, grad_clip_norm). The preset's
    num_decay_steps (30000) is stretched to `steps` on longer runs: lerobot only auto-scales it
    down, so a longer run would otherwise sit at decay_lr (2.5e-6) from step 30k on.
    `slow_params` train in their own group at lr*slow_scale (unfrozen pretrained VLM
    layers); the cosine scheduler is a LambdaLR, so the per-group ratio holds through warmup
    and decay. `floor` = the decayed lr as a fraction of the peak (0.1 = lerobot's preset);
    `decay_steps` = warmup-stable-decay (below). `constant` keeps the warmup and then
    holds the peak lr (the cosine's floor raised to its peak): every checkpoint of such a run
    is a finished run of that length, which a decaying schedule's checkpoints are not.
    `warmup` overrides the preset's warmup steps (1000)."""
    slow_ids = {id(p) for p in (slow_params or [])}
    opt_cfg = policy.config.get_optimizer_preset()
    if lr is not None:
        opt_cfg.lr = lr
    if slow_ids:
        tr = [p for p in policy.parameters()           # the expert-only leak check is trainable()
              if p.requires_grad and id(p) not in slow_ids]
        params = [{"params": tr},
                  {"params": [p for p in policy.parameters() if id(p) in slow_ids],
                   "lr": opt_cfg.lr * slow_scale}]
    else:
        # the policy's own grouping, as lerobot-train passes it (pi0/pi0.5/SmolVLA return plain
        # parameters, filtered to the trainable ones)
        params = policy.get_optim_params()
        if not isinstance(params, (dict, list)):
            params = [p for p in params if p.requires_grad]
    opt = opt_cfg.build(params)
    clip = opt_cfg.grad_clip_norm if grad_clip is None else grad_clip
    sched = None
    if scheduler and steps:
        import math
        sched_cfg = policy.config.get_scheduler_preset()
        if warmup is not None:
            sched_cfg.num_warmup_steps = int(warmup)
        if steps > sched_cfg.num_decay_steps:        # shorter runs keep lerobot's auto-scale-down
            sched_cfg.num_decay_steps = steps
        sched_cfg.decay_lr = sched_cfg.peak_lr * (1.0 if constant else floor)
        sched = sched_cfg.build(opt, steps)
        if decay_steps is not None:
            # warmup-stable-decay: lerobot's warmup, flat, then a cosine to `floor` over the LAST
            # `decay_steps` (the cool-down). Resumed from a constant-lr run's checkpoint (the
            # scheduler state is only the step count) this is a decay BRANCH off that trunk: one
            # trunk per dataset size + a short branch per (size, updates) cell instead of a full
            # run per cell.
            warm = int(sched_cfg.num_warmup_steps or 0)
            d0 = steps - decay_steps

            def lam(t):
                if t < warm:
                    return (t + 1) / (warm + 1)
                if t < d0 or not decay_steps:
                    return 1.0
                f = min(1.0, (t - d0) / max(1, decay_steps))
                return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * f))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    return opt, sched, float(clip)


def _prep_images(batch: dict) -> dict:
    """uint8 image tensors -> float [0,1] (lerobot-train's pre-preprocessor step); the normalizer
    leaves images alone. No-op on float batches."""
    for k, v in batch.items():
        if k.startswith("observation.images.") and torch.is_tensor(v) and v.dtype == torch.uint8:
            batch[k] = v.float() / 255.0
    return batch


def flow_loss(policy, preproc, batch: dict):
    """Flow-matching loss on one raw (pre-preprocessor) batch. Calls `policy` via __call__, not
    .forward, so a DistributedDataParallel wrapper's gradient-sync hooks run."""
    loss, _ = policy(preproc(_prep_images(batch)))
    return loss


def optim_step(policy, opt, sched, grad_clip: float) -> float:
    """After backward: clip -> step -> zero_grad -> scheduler step. Returns the pre-clip grad norm."""
    gnorm = torch.nn.utils.clip_grad_norm_(
        [p for p in policy.parameters() if p.requires_grad], grad_clip)
    opt.step()
    # Zero in place, never set_to_none: under DDP (gradient_as_bucket_view) the grads are views
    # into the all-reduce buckets; freed grads get re-allocated as plain tensors by the next
    # backward when it runs under no_sync (--grad-accum), and DDP then copies them into the
    # buckets on the synced micro-batch — two copies of the 5 GB gradient. The memset costs
    # milliseconds.
    opt.zero_grad(set_to_none=False)
    if sched is not None:
        sched.step()
    return float(gnorm)


class _fixed_seed:
    """Run a block under a fixed torch RNG state, then restore it. Validation draws a noise level
    and a noise sample per frame, so checkpoints must be scored on the same draws to compare."""

    def __init__(self, seed: int = 0):
        self.seed = seed

    def __enter__(self):
        self.cpu = torch.random.get_rng_state()
        self.cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(self.seed)
        return self

    def __exit__(self, *exc):
        torch.random.set_rng_state(self.cpu)
        if self.cuda is not None:
            torch.cuda.set_rng_state_all(self.cuda)
        return False


@torch.no_grad()
def val_loss(policy, preproc, loader, batches: int = 32, seed: int = 0) -> float:
    """Mean flow-matching loss over up to `batches` batches of `loader` under a fixed seed.
    Toggles eval() for the pass and restores train(). A subtask policy's per-batch
    last_subtask_ce/acc are averaged onto policy.last_val_subtask_ce/acc for the caller."""
    was_training = policy.training
    policy.eval()
    losses, ces, accs = [], [], []
    with _fixed_seed(seed):
        for k, batch in enumerate(loader):
            if k >= batches:
                break
            losses.append(float(flow_loss(policy, preproc, batch)))
            if getattr(policy, "last_subtask_ce", None) is not None:
                ces.append(policy.last_subtask_ce)
                accs.append(getattr(policy, "last_subtask_acc", 0.0) or 0.0)
    policy.last_val_subtask_ce = sum(ces) / len(ces) if ces else None
    policy.last_val_subtask_acc = sum(accs) / len(accs) if ces else None
    if was_training:
        policy.train()
    return sum(losses) / max(1, len(losses))


def _kin():
    """Lazy FK imports; only the millimetre metrics need arm/ on the path."""
    from arm import mapping as J
    from arm import so101_kin as K
    return J, K


def fingertip_height(state_deg) -> np.ndarray:
    """(N, 6) contract cal-deg -> (N,) moving-fingertip height above the board top (m), by FK."""
    from chess import task as T
    J, K = _kin()
    q_urdf, q_jaw = J.urdf_from_deg(np.atleast_2d(np.asarray(state_deg, float)))
    return np.array([K.moving_tip(q_urdf[i], q_jaw[i])[2] for i in range(len(q_urdf))]) - T.BOARD_TOP


def _tcp_gap(deg):
    """(N, 6) contract cal-deg -> (N, 3) TCP position in the base frame (m) and (N,) jaw gap (m),
    via the deploy FK. The TCP rides the fixed jaw and does not move with the gripper, so the
    gap is returned alongside it."""
    J, K = _kin()
    q_urdf, q_jaw = J.urdf_from_deg(np.asarray(deg, float))
    xyz = np.stack([np.asarray(K.tcp(q)[0], float) for q in q_urdf])
    return xyz, np.array([K.jaw_gap(j) for j in q_jaw])


def val_action_error(policy, preproc, post, loader, batches: int = 32,
                     low_m: float = 0.05, seed: int = 0) -> dict:
    """Mean TCP error (mm) of the postprocessed action chunk against the demonstration, over all
    held-out frames and over frames whose demonstrated fingertip is within `low_m` of the board,
    plus the mean jaw-gap error (mm). Means, not medians: success is conjunctive over time, so
    the tail a median hides is the failure mode. Open-loop, scored from demonstration states."""
    was_training = policy.training
    policy.eval()
    errs, gaps, heights = [], [], []
    with _fixed_seed(seed), torch.no_grad():
        for k, batch in enumerate(loader):
            if k >= batches:
                break
            chunk = policy.predict_action_chunk(preproc(_prep_images(batch)))
            pred = post(chunk).float().cpu().numpy()               # (B, nact, 6) contract deg
            demo = batch["action"][:, :pred.shape[1]].float().cpu().numpy()
            flat = (-1, demo.shape[-1])
            p_xyz, p_gap = _tcp_gap(pred.reshape(flat))
            d_xyz, d_gap = _tcp_gap(demo.reshape(flat))
            errs.append(np.linalg.norm(p_xyz - d_xyz, axis=-1))
            gaps.append(np.abs(p_gap - d_gap))
            heights.append(fingertip_height(demo.reshape(flat)))
    if was_training:
        policy.train()
    e = np.concatenate(errs) * 1000.0                              # m -> mm
    g = np.concatenate(gaps) * 1000.0
    low = np.concatenate(heights) <= low_m
    return {"tcp_mm": float(e.mean()), "gap_mm": float(g.mean()),
            "tcp_mm_low": float(e[low].mean()) if low.any() else float("nan"),
            "n_low": int(low.sum()), "n": int(e.size)}


# Checkpoint layout: checkpoints/<step>/pretrained_model (policy + processors, loadable by
# build_policy) + train_state.pt beside it for resume; checkpoints/last symlink.
def save_ckpt(outdir, step: int, policy, pre, post, opt, sched):
    d = Path(outdir) / "checkpoints" / f"{step:06d}"
    pm = d / "pretrained_model"
    policy.save_pretrained(pm)
    pre.save_pretrained(pm)
    post.save_pretrained(pm)
    torch.save({"step": step, "optimizer": opt.state_dict(),
                "scheduler": sched.state_dict() if sched is not None else None,
                "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state()},
               d / "train_state.pt")
    last = Path(outdir) / "checkpoints" / "last"
    if last.is_symlink() or last.exists():
        last.unlink()
    last.symlink_to(d.name, target_is_directory=True)   # relative -> sibling step dir
    return d


def prune_checkpoints(outdir, keep: int, keep_every: int = 0, keep_at=()) -> list[str]:
    """Remove every numbered checkpoint dir except `keep` (checkpoints/last already points at
    it). A run then holds ONE intermediate checkpoint — the newest, for resume — plus its
    final one: a pi0.5 checkpoint is ~13 GB (model + optimizer state). `keep_every` N also
    spares the multiples of N (the constant-lr runs: their intermediates are evaluated, and a
    decay branch resumes one); `keep_at` spares an explicit list of steps."""
    import shutil
    gone = []
    keep_at = {int(k) for k in keep_at}
    for d in sorted((Path(outdir) / "checkpoints").glob("[0-9]*")):
        if (keep_every and int(d.name) % keep_every == 0) or int(d.name) in keep_at:
            continue
        if d.is_dir() and not d.is_symlink() and int(d.name) != keep:
            shutil.rmtree(d)
            gone.append(d.name)
    return gone


def resume_pretrained(outdir) -> str | None:
    """The last checkpoint's pretrained_model dir to build_policy from, or None."""
    pm = Path(outdir) / "checkpoints" / "last" / "pretrained_model"
    return str(pm) if pm.exists() else None


def load_resume(outdir, opt, sched) -> int:
    """Restore optimizer/scheduler/rng state and return the step to resume at; 0 if fresh.
    Weights are loaded separately by build_policy(pretrained=resume_pretrained(...))."""
    ts = Path(outdir) / "checkpoints" / "last" / "train_state.pt"
    if not ts.exists():
        return 0
    st = torch.load(ts, map_location="cpu", weights_only=False)
    opt.load_state_dict(st["optimizer"])
    if sched is not None and st.get("scheduler"):
        sched.load_state_dict(st["scheduler"])
    torch.set_rng_state(st["torch_rng"])
    np.random.set_state(st["numpy_rng"])
    return int(st["step"])


class Log:
    """wandb wrapper, a no-op when disabled. `step_metric` names the record field used as the chart
    x-axis; without it wandb steps per log() call and interleaved record kinds misalign."""

    def __init__(self, project: str, name: str, config: dict, enabled: bool = True,
                 step_metric: str | None = None):
        self.run = None
        if enabled:
            import wandb
            self.run = wandb.init(project=project, name=name, config=config)
            # run URL on a stable, greppable line
            print(f"[wandb] {self.run.url}", flush=True)
            if step_metric:
                self.run.define_metric(step_metric)
                self.run.define_metric("*", step_metric=step_metric)

    def log(self, d: dict, step: int | None = None):
        if self.run is not None:
            self.run.log(d, step=step)
        return d

    def finish(self):
        if self.run is not None:
            self.run.finish()
