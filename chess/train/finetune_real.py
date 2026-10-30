#!/usr/bin/env python3
"""Fine-tune a VLA on the chess datasets, real or simulated. Views train raw (real fisheye views
are rectified to the sim's pinhole); the gripper channel is remapped 0..100 -> deg; action
targets stride every fps/control_hz frames; the split is fixed (--val-first/--train-first or an
--episodes file). pi0.5 trains with stage prediction (policy/pi05_subtask.py): single-token
stage labels resolved per frame from the dataset's mark rows, CE + flow in one forward
(--train-vlm-top N trains the predictor itself). Other families (pi0, SmolVLA) train on the
task text alone with their stock lerobot recipe. Writes checkpoints plus the control.json and
subtask.json sidecars and provenance.json.

  python -m chess.train.finetune_real <dataset> --pretrained lerobot/pi05_base --steps 20000
  torchrun --nproc_per_node 8 -m chess.train.finetune_real <dataset> ...      # DDP
"""
from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import os
import time
from pathlib import Path

import numpy as np

from chess.real import outcomes as outcomes_sidecar   # meta/outcomes.json: per-episode grade
from chess.real import segments                       # mark rows -> stage names (labels)
from policy import control_rate                       # the train==deploy control-rate contract
from policy import core

# Dataset units: arm joints in lerobot degrees (== contract cal-deg), gripper in RANGE_0_100.
# Training remaps channel 5 (state and action) with mapping.grip_0_100_to_deg, as deploy does;
# the on-disk dataset stays raw.


def _lerobot_augment():
    """lerobot-train's default image augmentation (ImageTransformsConfig(enable=True): 3-of-6 mild
    color/sharpness/small-affine transforms), applied as the dataset's per-sample image_transforms."""
    try:
        from lerobot.datasets.transforms import ImageTransforms, ImageTransformsConfig
    except ModuleNotFoundError:                     # the module's location in other lerobot versions
        from lerobot.transforms.transforms import ImageTransforms, ImageTransformsConfig
    return ImageTransforms(ImageTransformsConfig(enable=True))


ROLLOUT_SCHEDULE = Path(__file__).resolve().parents[2] / "schedules" / "easy_v1.json"
STILL_DEG = 0.25   # per-frame joint delta (deg, arm joints only) below this = parked
VAL_BATCH = 8      # val loader batch (inference only): fixed, so --val-frames means the same frames everywhere


def _still_tail_keep(root, meta) -> dict[int, int]:
    """Frames to keep per episode: length minus the trailing run of motionless frames (max |dq| of
    the 5 arm joints < STILL_DEG per frame, gripper ignored). Without the trim the ~7 parked frames
    per episode make "stay" outnumber "descend" at the home observation and freeze the deployed start.
    Reads the state column as one flat array per file (a per-row python conversion takes over half
    an hour on millions of frames)."""
    import glob
    import pyarrow.parquet as pq
    files = sorted(glob.glob(str(root / "data" / "*" / "*.parquet")))
    # Per file, not pa.concat_tables: some files store observation.state as list<float> where the
    # rest store fixed_size_list<float>[6], and a schema-strict concat rejects the mix.
    ep, fr, st = [], [], []
    for f in files:
        t = pq.read_table(f, columns=["episode_index", "frame_index", "observation.state"])
        ep.append(np.asarray(t.column("episode_index")))
        fr.append(np.asarray(t.column("frame_index")))
        col = t.column("observation.state").combine_chunks()
        try:                                   # fixed_size_list<float>[6]: one flat copy, no python rows
            st.append(np.asarray(col.values.to_numpy(), dtype=np.float32).reshape(len(col), -1))
        except Exception:                      # the list<float> variant: row by row
            st.append(np.stack([np.asarray(x) for x in col.to_pylist()]))
    ep, fr, st = np.concatenate(ep), np.concatenate(fr), np.concatenate(st)
    keep = {}
    for e in np.unique(ep):
        m = ep == e
        s = st[m][np.argsort(fr[m])][:, :5]
        v = np.abs(np.diff(s, axis=0)).max(1)
        moving = v >= STILL_DEG
        trail = int(np.argmax(moving[::-1])) if moving.any() else len(v)
        keep[int(e)] = max(2, len(s) - trail)
    return keep


def _write_provenance(out: Path, args, names, metas, splits, log, extra=None):
    """provenance.json beside the checkpoints: what this run trained on, exactly."""
    import subprocess
    try:
        git = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                             cwd=Path(__file__).parent, timeout=5).stdout.strip()
        git = git if len(git) == 40 else None
    except Exception:
        git = None
    out.mkdir(parents=True, exist_ok=True)
    (out / "provenance.json").write_text(json.dumps({
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "datasets": [{"name": n, "episodes": m.total_episodes, "frames": m.total_frames}
                     for n, m in zip(names, metas)],
        "train_eps": [t for t, _ in splits], "val_eps": [v for _, v in splits],
        "git": git, "wandb": (log.run.url if getattr(log, "run", None) else None),
        **(extra or {})}, indent=1))


class _Resumed:
    """A DistributedSampler minus its first `skip` indices, once: the resumed epoch continues
    where the previous job stopped (the sampler's permutation is a function of (seed, epoch),
    so the consumed prefix is exactly what that job trained on); every later epoch is the
    full shard. Without it a resumed run would replay its epoch from the start and train the
    consumed prefix twice."""

    def __init__(self, base, skip: int):
        self.base, self.skip = base, int(skip)

    def set_epoch(self, ep: int):
        self.base.set_epoch(ep)

    def __iter__(self):
        it = iter(self.base)
        if self.skip:
            it, self.skip = itertools.islice(it, self.skip, None), 0
        return it

    def __len__(self):
        return len(self.base) - self.skip


def fixed_split(meta, val_first: int, train_first: int):
    """The study's fixed split: episodes 0..K-1 are validation, forever; "train on n demos" =
    episodes K..K+n-1 (n = 0 -> all remaining). Membership is by raw collection index;
    drop-graded episodes are excluded without renumbering anyone."""
    n = meta.total_episodes
    drop = outcomes_sidecar.dropped(Path(meta.root)) if getattr(meta, "root", None) else set()
    if drop:
        print(f"outcomes: skipping {len(drop)} dropped episode(s): {sorted(drop)[:10]}", flush=True)
    val = [e for e in range(min(val_first, n)) if e not in drop]
    hi = n if train_first <= 0 else min(n, val_first + train_first)
    train = [e for e in range(val_first, hi) if e not in drop]
    if not train:
        raise SystemExit(f"empty train split: {n} episodes, --val-first {val_first}")
    return train, val


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", nargs="+",
                    help="dataset name(s) (<namespace>/<name>, or a bare name under local/); several "
                         "train as one concatenation with pooled stats")
    ap.add_argument("--pretrained", required=True,
                    help="HF base repo (lerobot/pi05_base, pi0_base, smolvla_base) or a checkpoint's "
                         "pretrained_model dir to fine-tune from (e.g. a policy pretrained in "
                         "simulation); this run's own out dir is resumed first")
    ap.add_argument("--vlm-init", default=None, metavar="HF_REPO",
                    help="NO robot pretraining: architecture from --pretrained, random weights, then "
                         "the VLM from this PaliGemma checkpoint (google/paligemma-3b-pt-224); the "
                         "action expert stays random (policy.core.build_policy)")
    ap.add_argument("--avg-weights", action="store_true",
                    help="keep a uniform running mean of the trainable weights over THIS run's updates "
                         "and save the mean as the final checkpoint's model (SWA)")
    ap.add_argument("--epochs", type=float, default=1.0,
                    help="training length in epochs over the train set (steps = epochs * batches/epoch)")
    ap.add_argument("--steps", type=int, default=None, help="explicit optimizer-step count, overrides --epochs")
    ap.add_argument("--control-hz", type=float, default=15.0,
                    help="deploy control rate (Hz): action targets stride fps/hz frames, recorded in "
                         "control.json (fps must be an integer multiple)")
    ap.add_argument("--batch", type=int, default=8, help="per-GPU batch")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (effective batch = batch * N); steps and "
                         "the LR schedule count optimizer steps")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--nact", type=int, default=10)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--val-first", type=int, default=30,
                    help="episodes 0..K-1 are the fixed validation set")
    ap.add_argument("--train-first", type=int, default=0,
                    help="train on episodes K..K+N-1 (the first N collected demos); 0 = all")
    ap.add_argument("--episodes", default=None,
                    help="JSON split file {\"val\": [...], \"train\": [...]} of raw episode indices "
                         "replacing the first-K rule; single-dataset runs only")
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--rollout-every", type=int, default=5000,
                    help="closed-loop sim rollouts of the live policy every N steps (0 = off): "
                         "the deploy loop on the easy_v1 prefix, success rates -> val/rollout/*, "
                         "files in <out>/rollouts/step_N")
    ap.add_argument("--rollout-k", type=int, default=16, help="trials per periodic rollout")
    ap.add_argument("--final-rollout-k", type=int, default=100,
                    help="trials of the final rollout after the last step (0 = off)")
    ap.add_argument("--val-frames", type=int, default=256,
                    help="held-out frames scored at every --val-every: a fixed-seed sample of the "
                         "val episodes, the SAME frames for every run regardless of --batch")
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--save-every", type=int, default=1000,
                    help="checkpoint cadence; only the NEWEST intermediate is kept (resume) "
                         "plus the final one (core.prune_checkpoints)")
    ap.add_argument("--keep-every", type=int, default=0,
                    help="also keep the checkpoints at multiples of N steps (0 = off)")
    ap.add_argument("--keep-at", default="",
                    help="comma-separated steps to save AND keep, on top of --save-every / --keep-every")
    ap.add_argument("--warmup", type=int, default=None,
                    help="lr warmup steps (default: the policy preset's 1000)")
    ap.add_argument("--lr-schedule", default="cosine", choices=["cosine", "constant"],
                    help="cosine = warmup, then decay to lr * --lr-floor over the run; constant = the "
                         "same warmup, then flat (every checkpoint is then a finished shorter run)")
    ap.add_argument("--decay-steps", type=int, default=None,
                    help="warmup-stable-decay: flat lr, then a cosine cool-down to --lr-floor over the "
                         "LAST N steps. In an out dir whose checkpoints/last links a constant-lr run's "
                         "checkpoint, this is a decay branch off that trunk")
    ap.add_argument("--lr-floor", type=float, default=0.1,
                    help="the decayed lr as a fraction of the peak (cosine and --decay-steps)")
    ap.add_argument("--seed", type=int, default=0,
                    help="training RNG only — the split is fixed (--val-first/--train-first)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--trim-still", action=argparse.BooleanOptionalAction, default=True,
                    help="drop each episode's trailing parked frames from the sampled observations "
                         "(action targets still reach into the tail)")
    ap.add_argument("--augment", action=argparse.BooleanOptionalAction, default=True,
                    help="lerobot-train's mild image augmentation (3-of-6 color/sharpness/affine)")
    ap.add_argument("--use-overhead", action=argparse.BooleanOptionalAction, default=True,
                    help="feed the overhead camera too (pi0/pi0.5: base_0_rgb; SmolVLA: camera1)")
    ap.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                    help="model load/forward dtype (default: the policy's own); bfloat16 for pi0.5")
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="gradient checkpointing (pi0.5)")
    ap.add_argument("--train-full", action="store_true",
                    help="train all params including the VLM backbone (default: action expert only)")
    ap.add_argument("--train-vlm-top", type=int, default=18, metavar="N",
                    help="pi0.5: unfreeze the top N Gemma layers + final norm and attach the stage CE. "
                         "Default 18 = the whole LM; 0 = VLM frozen, CE = monitor only")
    ap.add_argument("--vlm-lr-scale", type=float, default=0.1,
                    help="lr multiplier for the unfrozen VLM group (--train-vlm-top)")
    ap.add_argument("--out", default=None, help="checkpoint dir (default out/ft_<name>)")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--wandb", default="vla-chess", help="wandb project")
    ap.add_argument("--no-wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny: 4 steps, no wandb/ckpt")
    args = ap.parse_args()

    import torch
    import torch.distributed as dist
    # DDP: torchrun sets WORLD_SIZE/RANK/LOCAL_RANK; a plain run has world=1 and every ddp branch no-ops.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    ddp, is_main = world > 1, rank == 0
    if ddp:
        from datetime import timedelta
        dist.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"
        print(f"[ddp] rank {rank}/{world} local_rank {local_rank} on {args.device}", flush=True)
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from arm import mapping as J

    names = [d.split("/")[-1] for d in args.data]
    repo_ids = [outcomes_sidecar.repo_id(d) for d in args.data]
    name = "+".join(names)
    roots = [outcomes_sidecar.dataset_root(d) for d in args.data]
    for root in roots:
        if not (root / "meta" / "info.json").exists():
            raise SystemExit(f"{root} not found — download or collect the dataset first")
    out = Path(args.out or (Path("out") / f"ft_{name}"))
    if args.smoke:
        args.steps, args.val_every, args.save_every, args.log_every = 4, 4, 0, 1

    torch.manual_seed(args.seed)             # same on every rank: identical model init under DDP
    np.random.seed(args.seed + rank)

    metas = [LeRobotDatasetMetadata(r, root=str(rt)) for r, rt in zip(repo_ids, roots)]
    meta = metas[0]
    if len(metas) > 1:
        # Several datasets train as one concatenation: same fps and feature shapes required,
        # normalization = pooled stats of the union (written into meta.stats).
        for m, r in zip(metas[1:], repo_ids[1:]):
            if m.fps != meta.fps:
                raise SystemExit(f"{r} is {m.fps} fps but {repo_ids[0]} is {meta.fps} — one rate per run")
            for k in ("observation.state", "action",
                      "observation.images.wrist", "observation.images.overhead"):
                a, b = meta.features.get(k), m.features.get(k)
                if (a is None) != (b is None) or (a and list(a["shape"]) != list(b["shape"])):
                    raise SystemExit(f"{r} feature {k} does not match {repo_ids[0]} "
                                     f"({b and b['shape']} vs {a and a['shape']})")
        meta.stats = core.combine_stats(metas)
    # Targets stride every ratio-th frame; a non-integer ratio puts delta timestamps between frames.
    ratio = float(meta.fps) / args.control_hz
    if abs(ratio - round(ratio)) > 1e-9 or ratio < 1:
        raise SystemExit(f"dataset fps {meta.fps} is not an integer multiple of "
                         f"--control-hz {args.control_hz:g} — pick a rate that divides the fps")
    ratio = int(round(ratio))
    pretrained = core.resume_pretrained(out) or args.pretrained
    print(f"fine-tuning from {pretrained} on "
          + " + ".join(f"{r} ({m.total_episodes} eps, {m.total_frames} frames)"
                       for r, m in zip(repo_ids, metas))
          + f", fps {meta.fps}"
          + (f" [pooled stats over {len(metas)} datasets]" if len(metas) > 1 else ""), flush=True)
    print(f"control rate: {args.control_hz:g} Hz (dataset {meta.fps} fps -> action-target "
          f"stride {ratio} frame{'s' if ratio > 1 else ''}/tick)", flush=True)
    # Build one rank at a time: building pi0.5 peaks tens of GB of host RAM per rank, and
    # concurrent ranks OOM the cgroup. A flock, not dist.barrier(): no collective may run before
    # the DDP wrap (the NCCL rank->device mapping is not set yet and a barrier here hangs).
    build_lock = Path(os.environ.get("TMPDIR", "/tmp")) / \
        f"ft_build_{os.environ.get('SLURM_JOB_ID', 'local')}.lock"
    with open(build_lock, "a") as _lf:
        if ddp:
            import fcntl
            fcntl.flock(_lf, fcntl.LOCK_EX)
            print(f"[ddp rank {rank}] building policy (exclusive)", flush=True)
        policy, pre, post = core.build_policy(meta, device=args.device, nact=args.nact,
                                              pretrained=str(pretrained), dtype=args.dtype,
                                              grad_ckpt=args.grad_ckpt,
                                              expert_only=not args.train_full,
                                              use_overhead=args.use_overhead,
                                              # a resume loads the out dir's checkpoint, never re-inits
                                              vlm_init=(args.vlm_init if str(pretrained) == args.pretrained else None))
        if ddp:
            fcntl.flock(_lf, fcntl.LOCK_UN)
    _, n_tr, n_all = core.trainable(policy, expert_only=not args.train_full)
    print(f"trainable {n_tr/1e6:.2f}M / {n_all/1e6:.2f}M total "
          f"({'ALL params incl. VLM' if args.train_full else 'VLM frozen'})", flush=True)
    from policy import pi05_subtask
    subtask_on = pi05_subtask.install(policy)
    if subtask_on:
        missing = [n for n, m in zip(names, metas) if "language_persistent" not in m.features]
        if missing:
            raise SystemExit(f"pi0.5 trains with stage prediction, but {missing} carry no stage "
                             "mark rows (language_persistent)")
        print("pi0.5 stage prediction: ON (policy/pi05_subtask.py)", flush=True)
    vlm_params = []
    if args.train_vlm_top and not subtask_on:      # the default is a pi0.5 recipe; other families
        print("--train-vlm-top: ignored — not a pi0.5 stage-prediction policy", flush=True)
    elif args.train_vlm_top:
        if args.train_full:
            raise SystemExit("--train-vlm-top and --train-full are exclusive (full trains all of it)")
        vlm_params = pi05_subtask.unfreeze_vlm_top(policy, args.train_vlm_top)
        print(f"stage predictor: CE attached — top {args.train_vlm_top} VLM layers + final "
              f"norm unfrozen (+{sum(p.numel() for p in vlm_params)/1e6:.1f}M params "
              f"@ lr x{args.vlm_lr_scale:g})", flush=True)
    print(f"views: raw (real fisheye views rectified to the sim's pinhole); "
          f"overhead {'ON' if args.use_overhead else 'OFF (wrist-only)'}", flush=True)

    # One split per dataset, so each contributes its own held-out episodes.
    if args.episodes:
        if len(metas) != 1:
            raise SystemExit("--episodes takes a single dataset")
        sp = json.loads(Path(args.episodes).read_text())
        splits = [([int(e) for e in sp["train"]], [int(e) for e in sp["val"]])]
        print(f"episodes: {args.episodes}: {len(splits[0][0])} train / {len(splits[0][1])} val "
              f"(the file's lists, not the first-K rule)", flush=True)
    else:
        splits = [fixed_split(m, args.val_first, args.train_first) for m in metas]
    # Behavior cloning imitates successes: bad-graded episodes (drops, wrong cells, topples)
    # leave BOTH sides of the split; mid stays (right square, off-centre — a lenient success is
    # still a demonstration). The episode WINDOW stays K..K+N-1, so "n demos" keeps its meaning.
    def _successes(m, eps):
        root = getattr(m, "root", None)
        oc = (outcomes_sidecar.load(Path(root)).get("outcomes") or {}) if root else {}
        cut = [e for e in eps if oc.get(str(e)) == "bad"]
        if cut:
            print(f"BC: excluding {len(cut)} bad-graded episode(s): "
                  f"{cut[:10]}{'...' if len(cut) > 10 else ''}", flush=True)
        return [e for e in eps if e not in set(cut)]
    splits = [(_successes(m, t), _successes(m, v)) for m, (t, v) in zip(metas, splits)]
    # A marked dataset labels every frame from its mark rows (WithSubtask below). An episode
    # the collector left blank — a failure that does not decompose — has no label to give, and
    # one such episode in the mix kills the loader on the first batch that samples it. Out of
    # both splits.
    def _marked(eps, blank, label):
        cut = sorted(set(eps) & blank)
        if cut:
            print(f"marks: excluding {len(cut)} unmarked {label} episode(s): {cut[:10]}", flush=True)
        return [e for e in eps if e not in blank]
    blanks = [segments.unmarked(rt) if "language_persistent" in m.features else set()
              for m, rt in zip(metas, roots)]
    splits = [(_marked(t, b, "train"), _marked(v, b, "val")) for (t, v), b in zip(splits, blanks)]
    train_eps, val_eps = splits[0]
    print("episode split: " + " + ".join(f"{len(t)} train / {len(v)} val" for t, v in splits)
          + (f"  (total {sum(len(t) for t, _ in splits)} / {sum(len(v) for _, v in splits)})"
             if len(splits) > 1 else ""), flush=True)

    # Action targets in tick-seconds (1/control_hz): one action step == one deploy control tick.
    delta = {"action": [i / args.control_hz for i in range(policy.config.chunk_size)]}
    aug = _lerobot_augment() if args.augment else None
    print(f"augment: {'lerobot-standard (3-of-6, mild)' if aug else 'OFF'}", flush=True)

    # Index view over the dataset; the per-row mapping is read off ds.hf_dataset itself.
    class Trimmed(torch.utils.data.Dataset):
        """Frame-level subset: --trim-still's per-episode keep length (needs ds.hf_dataset)."""
        def __init__(self, ds, keep_len):
            ep = np.asarray([int(x) for x in ds.hf_dataset["episode_index"]])
            fr = np.asarray([int(x) for x in ds.hf_dataset["frame_index"]])
            ka = np.zeros(ep.max() + 1, dtype=np.int64)
            for e, k in keep_len.items():
                if e <= ep.max():
                    ka[e] = k
            m = fr < ka[ep]
            self.still_dropped = int((~m).sum())
            self.idx = np.nonzero(m)[0]
            self.dropped = len(ep) - len(self.idx)
            self.ds = ds

        def __len__(self):
            return len(self.idx)

        def __getitem__(self, i):
            return self.ds[int(self.idx[i])]

    class WithSubtask(torch.utils.data.Dataset):
        """Resolve each frame's stage NAME (the single-token conditioning label,
        policy/pi05_subtask.py) from its language_persistent mark rows (lerobot's own
        active_at) and drop the raw rows (they don't collate)."""
        def __init__(self, ds):
            from lerobot.datasets.language_render import active_at
            self.ds, self._at = ds, active_at

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            item = dict(self.ds[i])
            rows = item.pop("language_persistent", None)
            if not rows:                          # the split drops unmarked episodes (segments.unmarked)
                raise RuntimeError(f"episode {int(item['episode_index'])} carries no sub-task mark "
                                   f"rows at t={float(item['timestamp']):.2f} — it cannot train")
            r = self._at(float(item["timestamp"]), persistent=rows, style="subtask")
            if r is None:
                raise RuntimeError(f"no active sub-task at t={float(item['timestamp']):.2f} "
                                   f"(episode {int(item['episode_index'])}) — fix the marks")
            name = segments.name_of(r["content"])
            if name is None:
                raise RuntimeError(f"unrecognized sub-task row {r['content']!r} "
                                   f"(episode {int(item['episode_index'])})")
            item["subtask"] = name
            return item

    class Project(torch.utils.data.Dataset):
        """Restrict items to `keys`: concatenated datasets need one common key set
        (default_collate rejects ragged dicts)."""
        def __init__(self, ds, keys):
            self.ds, self.keys = ds, keys

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            return {k: v for k, v in self.ds[i].items() if k in self.keys}

    class Rectified(torch.utils.data.Dataset):
        """REAL datasets: each raw fisheye view rectified to the sim's pinhole
        (camera.undistort, model.CAM_HFOV), then the augmentation. Recording stays raw; the
        real deployment applies the same rectification. Sim datasets (a `sim.qpos` column)
        render that pinhole already."""
        def __init__(self, ds, aug_):
            from camera.undistort import Undistorter
            self.ds, self.aug = ds, aug_
            self.und = {f"observation.images.{c}": Undistorter(c) for c in ("wrist", "overhead")}
            self.hf_dataset = ds.hf_dataset    # (Trimmed reads it)

        def __len__(self):
            return len(self.ds)

        def __getitem__(self, i):
            item = dict(self.ds[i])
            for k, und in self.und.items():
                if k in item:
                    img = und(item[k].permute(1, 2, 0).contiguous().numpy())   # (C,H,W) float -> HWC
                    item[k] = torch.from_numpy(img).permute(2, 0, 1)
                    if self.aug is not None:
                        item[k] = self.aug(item[k])
            return item

    def build(idx, eps, aug_, label):
        real = "sim.qpos" not in metas[idx].features
        ds = LeRobotDataset(repo_ids[idx], root=str(roots[idx]), video_backend="pyav",
                            delta_timestamps=delta, image_transforms=None if real else aug_,
                            episodes=eps)
        if real:
            ds = Rectified(ds, aug_)
        if args.trim_still:
            n0 = len(ds)
            ds = Trimmed(ds, _still_tail_keep(roots[idx], metas[idx]))
            print(f"trim-still: {names[idx]} {label} drops {ds.still_dropped}/{n0} "
                  f"frames ({ds.still_dropped / max(1, n0):.1%} parked tails)", flush=True)
        if "language_persistent" in metas[idx].features:
            ds = WithSubtask(ds)               # outermost: Trimmed needs ds.hf_dataset
        return ds

    def concat(dss):
        if len(dss) == 1:
            return dss[0]
        keys = set.intersection(*[set(d[0].keys()) for d in dss])
        dropped = sorted(set.union(*[set(d[0].keys()) for d in dss]) - keys)
        if dropped:
            print(f"concat: dropping non-shared column(s) {dropped}", flush=True)
        return torch.utils.data.ConcatDataset([Project(d, keys) for d in dss])

    train_ds = concat([build(i, t, aug, "train") for i, (t, _) in enumerate(splits)])
    # DistributedSampler shards the epoch across ranks (global batch = args.batch * world);
    # steps_per_epoch below reads len(loader) = the per-rank shard.
    train_sampler = (torch.utils.data.distributed.DistributedSampler(
        train_ds, num_replicas=world, rank=rank, shuffle=True, drop_last=True) if ddp else None)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch,
                                         shuffle=(train_sampler is None), sampler=train_sampler,
                                         num_workers=args.num_workers, drop_last=True, pin_memory=True)
    # One val loader per dataset (a pooled number hides which one moved), each a fixed-seed random
    # sample of its held-out frames: val reads a prefix of the loader, and an unshuffled prefix is
    # a few near-duplicate episodes of the first dataset only.
    val_loaders = {}
    if is_main:                                 # held-out val on rank 0 only
        n_val = args.val_frames
        for i, (_, v) in enumerate(splits):
            if not v:
                continue
            ds = build(i, v, None, "val")
            pick = np.random.default_rng(args.seed).permutation(len(ds))[:n_val]
            print(f"val[{names[i]}]: {len(pick)} of {len(ds)} held-out frames "
                  f"({len(v)} episodes)", flush=True)
            val_loaders[names[i]] = torch.utils.data.DataLoader(
                torch.utils.data.Subset(ds, sorted(int(x) for x in pick)),
                batch_size=VAL_BATCH, shuffle=False, num_workers=2, drop_last=False)

    # Resolve --epochs to a step count now that len(loader) is known; --steps (and --smoke) override.
    # args.steps carries the resolved count to the loop, the checkpoints and the wandb config.
    accum = max(1, args.grad_accum)
    steps_per_epoch = max(1, len(loader) // accum)   # a step == one optimizer step (accum micro-batches)
    if args.steps is None:
        args.steps = max(1, round(args.epochs * steps_per_epoch))
    print(f"length: {args.steps} steps = {args.steps / steps_per_epoch:.2f} epochs "
          f"({steps_per_epoch} steps/epoch @ batch {args.batch}"
          f"{f' x{accum} accum = eff {args.batch * accum}' if accum > 1 else ''})", flush=True)
    keep_at = {int(x) for x in args.keep_at.split(",") if x.strip()}
    if keep_at:
        print(f"keeping checkpoints at {sorted(keep_at)}", flush=True)
    opt, sched, clip = core.build_optim(policy, steps=args.steps, lr=args.lr, warmup=args.warmup,
                                        slow_params=vlm_params, slow_scale=args.vlm_lr_scale,
                                        constant=args.lr_schedule == "constant",
                                        decay_steps=args.decay_steps, floor=args.lr_floor)
    n_trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if n_trainable > 2e9:            # --train-full, or most of the VLM via --train-vlm-top
        # Single-tensor Adam: the foreach step materializes a full copy of the second moment
        # (~7.7 GiB at 4.14B trainable), which OOMs full-VLM runs on 40 GB cards.
        for g in opt.param_groups:
            g["foreach"] = False
    start = core.load_resume(out, opt, sched)
    if start:
        skip = (start % steps_per_epoch) * accum * args.batch     # samples per rank, this epoch
        if train_sampler is not None and skip:
            train_sampler = _Resumed(train_sampler, skip)
            loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch, shuffle=False,
                                                 sampler=train_sampler,
                                                 num_workers=args.num_workers, drop_last=True,
                                                 pin_memory=True)
        print(f"resuming at step {start}: epoch {start // steps_per_epoch}, "
              f"skipping its first {skip} samples per rank", flush=True)

    # DDP wraps for the gradient all-reduce only; optimizer/save/val use the underlying policy.
    # pi0.5 has trainable params its flow loss never touches (the expert's LM head), so it needs
    # find_unused_parameters=True (safe with its non-reentrant grad checkpointing). No static_graph.
    train_model = policy
    if ddp:
        from torch.nn.parallel import DistributedDataParallel as DDP
        train_model = DDP(policy, device_ids=[local_rank],
                          find_unused_parameters=(policy.config.type != "smolvla"),
                          gradient_as_bucket_view=True)
        print(f"[ddp rank {rank}] policy wrapped in DDP", flush=True)

    log = core.Log(args.wandb, args.run_name or f"ft_{name}", vars(args),
                   enabled=not (args.no_wandb or args.smoke) and is_main)
    if is_main:
        _write_provenance(out, args, names, metas, splits, log,
                          extra={"subtask_sequence": subtask_on})

    def save_sidecars(pm):
        # train==deploy contracts beside the weights: the control rate and the stage sequence.
        control_rate.save(pm, args.control_hz, meta.fps)
        if subtask_on:                                         # the conditioning contract for serving
            (pm / "subtask.json").write_text(json.dumps(
                {"conditioning": "subtask", "prompt": "whole-move", "labels": "name-token",
                 "sequence": list(segments.E2E_SEQUENCE)}, indent=1))

    def prep(batch):
        # All preprocessing at train time (the on-disk dataset stays raw): gripper 0_100 -> contract
        # deg on state and action (mapping.grip_0_100_to_deg, as at deploy).
        s, a = batch["observation.state"], batch["action"]
        s[:, 5] = torch.as_tensor(J.grip_0_100_to_deg(s[:, 5].cpu().numpy()), dtype=s.dtype)
        a[..., 5] = torch.as_tensor(J.grip_0_100_to_deg(a[..., 5].cpu().numpy()), dtype=a.dtype)
        return batch

    # Stage names ride the batch as "subtask" (WithSubtask); popped before the pipeline
    # (which would drop them) and turned into single-token labels with the pipeline's tokenizer.
    if subtask_on:
        _sub_tok = pi05_subtask.find_tokenizer(pre)
        # subtask_acc = argmax within the five labels (the deployed candidate set)
        policy.model.subtask_label_ids = pi05_subtask.label_tokens(
            _sub_tok, segments.E2E_SEQUENCE)[0].squeeze(1)
        def pre_s(b, _ph=pre, _tok=_sub_tok):
            names = b.pop("subtask")
            return pi05_subtask.attach_subtask(_ph(b), list(names), _tok)
    else:
        def pre_s(b, _ph=pre):
            b.pop("subtask", None)
            return _ph(b)

    def rollouts(step: int, k: int, envs: int, tag: str) -> dict | None:
        """Closed-loop sim rollouts of the live policy on EVERY rank (trials split over the
        ranks, `envs` worlds each, gathered by rank 0): success rates to wandb. Every rank
        must call this; rank 0 returns the summary, the others None."""
        if not subtask_on:
            if is_main:
                print("rollouts: skipped — not a stage-prediction policy (eval_e2e needs one)", flush=True)
            return None
        try:
            from chess.sim import eval_e2e
        except ImportError as e:                     # a training machine without the simulator
            if is_main:
                print(f"rollouts: skipped — sim unavailable here ({e})", flush=True)
            return None
        t_r = time.monotonic()
        res = eval_e2e.rollout(policy, pre, post, _sub_tok, args.control_hz,
                               schedule=ROLLOUT_SCHEDULE, k=k, envs=envs,
                               out=out / "rollouts" / tag, data=names[0],
                               label=f"{out.name}@{step}", device=args.device,
                               part=(rank, world))
        torch.cuda.empty_cache()      # hand the rollout's worlds/caches back before training
        if res is None:               # a non-zero rank's part; rank 0 holds the merged eval
            return None
        n = max(1, res["n"])
        ms = res["milestones"]
        row = {"step": step, "val/rollout/n": res["n"],
               "val/rollout/strict_rate": res["strict"] / n,
               "val/rollout/lenient_rate": res["lenient"] / n,
               **{f"val/rollout/{m}_rate": ms[m] / n for m in ms},
               "val/rollout/timeout_rate": res["timeouts"] / n,
               "val/rollout/toppled_rate": res["toppled"] / n}
        if res["success_s"] is not None:
            row["val/rollout/success_s"] = res["success_s"]     # median length of a success
        log.log(row, step=step)
        print(f"  [rollout] step {step}  {tag}: strict {res['strict']}/{res['n']}  "
              f"lenient {res['lenient']}/{res['n']}  "
              f"milestones {'·'.join(str(ms[m]) for m in ms)}  "
              f"timeouts {res['timeouts']}  toppled {res['toppled']}  "
              f"({time.monotonic() - t_r:.0f} s)", flush=True)
        return res

    policy.train()
    step, t0, ep = start, time.monotonic(), start // steps_per_epoch
    t_log, step_log = t0, start                 # the previous log line: the rate is per window
    done = False
    micro, loss_sum = 0, 0.0                    # accumulate `accum` micro-batches per optimizer step
    avg_params, avg_sum, n_avg = [], [], 0
    if args.avg_weights:
        avg_params = [p for p in policy.parameters() if p.requires_grad]
        avg_sum = [p.detach().float().clone() for p in avg_params]   # seeded with the start weights: the
        n_avg = 1                                                       # mean over the run includes them
        print(f"--avg-weights: uniform mean of {sum(p.numel() for p in avg_params)/1e6:.1f}M trainable "
              f"params over this run's updates (+{sum(a.numel() for a in avg_sum) * 4 / 2**30:.1f} GB fp32)",
              flush=True)
    while not done:
        if train_sampler is not None:
            train_sampler.set_epoch(ep)          # reshuffle each rank's shard in lockstep
        for batch in loader:
            # Under DDP only the LAST micro-batch of a step all-reduces its gradients; the earlier
            # ones accumulate locally (no_sync). Without it --grad-accum N pays N all-reduces of
            # the ~5 GB bf16 gradient per optimizer step.
            last_micro = micro + 1 >= accum
            with (train_model.no_sync() if ddp and not last_micro else contextlib.nullcontext()):
                loss = core.flow_loss(train_model, pre_s, prep(batch)) / accum  # DDP wrapper => grad
                loss.backward()                                                 #   all-reduce on backward
            loss_sum += float(loss)
            micro += 1
            if micro < accum:
                continue                          # keep filling this optimizer step
            micro = 0
            gnorm = core.optim_step(policy, opt, sched, clip)   # clips/steps the effective-batch grad
            step += 1
            if args.avg_weights:                  # uniform mean of the iterates of THIS run (fp32, on the GPU)
                n_avg += 1
                for p, a in zip(avg_params, avg_sum):
                    # lerp_ keeps the update out of place: `.to(a.dtype)` is a no-op on fp32 weights, so an
                    # in-place sub_ on it would subtract the mean from the live weights
                    a.lerp_(p.detach().to(a.dtype), 1.0 / n_avg)
            train_loss, loss_sum = loss_sum, 0.0  # mean loss over the effective batch
            if is_main and step % args.log_every == 0:
                # it/s over THIS log window (a window holding a val pass or a rollout shows the dip)
                now = time.monotonic()
                sps = (step - step_log) / max(1e-9, now - t_log)
                t_log, step_log = now, step
                epoch = step / steps_per_epoch
                mem = torch.cuda.max_memory_reserved() / 2**30     # rank 0's peak: the OOM headroom
                print(f"step {step}/{args.steps} (ep {epoch:.2f})  loss {train_loss:.4f}  gnorm {gnorm:.2f}  "
                      f"{sps:.2f} it/s  (gbatch {args.batch * accum * world})  mem {mem:.1f}G", flush=True)
                row = {"train/loss": train_loss, "train/grad_norm": gnorm,
                       "epoch": epoch, "step": step}
                if subtask_on:
                    row["train/subtask_ce"] = getattr(policy, "last_subtask_ce", None)
                    row["train/subtask_acc"] = getattr(policy, "last_subtask_acc", None)
                log.log(row, step=step)
            did_val = bool(args.val_every) and step % args.val_every == 0
            did_ckpt = (bool(args.save_every) and step % args.save_every == 0) or step in keep_at
            did_roll = (bool(args.rollout_every) and step % args.rollout_every == 0
                        and step < args.steps and not args.smoke)   # the final one runs after
            if did_val and val_loaders:                       # rank 0 owns the held-out val
                row = {"epoch": step / steps_per_epoch, "step": step}
                for dname, vloader in val_loaders.items():
                    # Metric keys: `val/tcp_mm` in the usual single-dataset run, so runs on different
                    # datasets share one wandb panel (the val episodes are the same first-K anyway);
                    # `val/<dataset>/tcp_mm` only when several datasets are validated at once.
                    key = "val/" if len(val_loaders) == 1 else f"val/{dname}/"
                    vl = core.val_loss(policy, pre_s, (prep(b) for b in vloader),
                                       batches=len(vloader))
                    # The same batches scored on the integrated chunk in mm at the fingertip: the flow
                    # loss is a noise-level velocity regression, not a ranking signal.
                    ae = core.val_action_error(policy, pre_s, post, (prep(b) for b in vloader),
                                               batches=len(vloader))
                    sub = ""
                    if getattr(policy, "last_val_subtask_ce", None) is not None:
                        row.update({key + "subtask_ce": policy.last_val_subtask_ce,
                                    key + "subtask_acc": policy.last_val_subtask_acc})
                        sub = (f"  subtask ce {policy.last_val_subtask_ce:.3f} "
                               f"acc {policy.last_val_subtask_acc:.0%}")
                    print(f"  [val] step {step}  {dname}: loss {vl:.4f}  tcp {ae['tcp_mm']:.1f} mm "
                          f"(<=5cm {ae['tcp_mm_low']:.1f}, n={ae['n_low']}/{ae['n']})  "
                          f"gap {ae['gap_mm']:.1f} mm{sub}", flush=True)
                    row.update({key + "loss": vl, key + "tcp_mm": ae["tcp_mm"],
                                key + "tcp_mm_low": ae["tcp_mm_low"],
                                key + "gap_mm": ae["gap_mm"]})
                log.log(row, step=step)
                torch.cuda.empty_cache()   # val's inference blocks back to the driver before training
            if did_ckpt and is_main:                          # rank 0 owns the checkpoint
                d = core.save_ckpt(out, step, policy, pre, post, opt, sched)
                save_sidecars(d / "pretrained_model")
                gone = core.prune_checkpoints(out, step, args.keep_every, keep_at)   # newest intermediate (+ kept)
                print(f"  saved checkpoint {step}" + (f" (pruned {', '.join(gone)})" if gone else ""),
                      flush=True)
            if did_roll:                                      # every rank: its part of the trials
                rollouts(step, args.rollout_k, 16, f"step_{step:06d}")   # 16 worlds per rank, as the final
            if ddp and (did_val or did_ckpt):                 # others wait while rank 0 vals/saves
                dist.barrier()
            if step >= args.steps:
                done = True
                break
        ep += 1
    if not args.smoke and is_main:
        if args.avg_weights:                     # the final model = the mean of the iterates, not the last one
            with torch.no_grad():
                for p, a in zip(avg_params, avg_sum):
                    p.copy_(a.to(p.dtype))
            print(f"--avg-weights: final checkpoint holds the mean over {n_avg} iterates "
                  f"(train_state.pt keeps the optimizer of the last iterate; not for resuming)", flush=True)
        d = core.save_ckpt(out, args.steps, policy, pre, post, opt, sched)
        save_sidecars(d / "pretrained_model")
        core.prune_checkpoints(out, args.steps, args.keep_every, keep_at)   # the final step is the checkpoint
        print(f"saved final checkpoint {args.steps} -> {out}/checkpoints/last", flush=True)
    if ddp:
        dist.barrier()                          # ranks 1+ wait here through rank 0's final save
    if not args.smoke and args.final_rollout_k:
        del opt                                  # free the optimizer state for the worlds
        torch.cuda.empty_cache()
        rollouts(args.steps, args.final_rollout_k, 16, "final")   # every rank, as in training
    if ddp:
        dist.destroy_process_group()
    log.finish()
    if is_main:
        print("FINETUNE DONE", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
