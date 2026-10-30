# VLA Adaptation with Real and Simulated Data: Robotic Chess

Code for the CoRL 2026 workshop paper *VLA Adaptation with Real and Simulated Data: A Case Study on
Robotic Chess* (Matěj Kripner, Milan Straka; Charles University).

An SO-101 arm moves a chess king between two squares given only by their names
(*"Move the king from e4 to a5"*), then returns to its ready pose. This repository contains:

- **the simulator**: a MuJoCo twin of the real setup (arm, board, king and both cameras placed
  from measurement), with ground-truth grading of every trial (`chess/sim/`);
- **the scripted expert**, deterministic and randomized (`--style reach`: a person modelled as a
  noisy controller that aims, reaches, looks and corrects), and the collector that writes its
  demonstrations as LeRobot datasets (`chess/sim/collect.py`);
- **the closed-loop evaluator** with the deployment control loop (action chunks, real-time
  chunking, stage switching) over fixed move schedules (`chess/sim/eval_e2e.py`, `schedules/`);
- **the fine-tuning recipe** for π0.5 with stage prediction, plus π0 and SmolVLA with their stock
  LeRobot recipes (`chess/train/finetune_real.py`, `policy/`).

## Install

Python 3.12, a CUDA GPU for training and policy evaluation, EGL for headless rendering.

```bash
python -m venv venv && . venv/bin/activate
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

All commands run from the repository root. Datasets live in LeRobot's standard place,
`$HF_LEROBOT_HOME/<namespace>/<name>` (default `~/.cache/huggingface/lerobot`); a bare dataset
name means `local/<name>`.

Smoke tests: `python -m arm.mapping`, `python -m arm.so101_kin`, `python -m chess.sim.scene --out /tmp/scene`
(FK parity between the URDF and the MuJoCo model, plus both camera renders).

## Simulated demonstrations

```bash
python -m chess.sim.collect king_reach --n 10300 --seed 7000 --style reach --workers 16
```

The expert plays a seeded chain of moves (each target is the next source) with the king placed
uniformly within ±8 mm of its square centre. Every episode is graded from the simulator state and
marked with its five stages (approach, pick, transport, place, home); failures stay in the dataset
with their grade. The paper's training set is this chain (seed 7000) extended by four 300-trial
chains with seeds 7100, 7200, 7300 and 7400.

Score the expert itself, without rendering:

```bash
python -m chess.sim.eval_e2e --expert --style reach --schedule schedules/hard_v1.json --k 1000
```

## Training

π0.5 as in the paper (8 GPUs, global batch 32, bf16, the whole language model trained for stage
prediction). Splits are files of episode indices; `chess.train.splits` writes them (successes
only, nested prefixes).

```bash
python -m chess.train.splits king_reach --sizes 100,316,1000,3162,10000
torchrun --nproc_per_node 8 -m chess.train.finetune_real king_reach \
    --pretrained lerobot/pi05_base --episodes splits/king_reach_n1000.json \
    --dtype bfloat16 --control-hz 15 --train-vlm-top 18 --vlm-lr-scale 1.0 \
    --batch 2 --grad-accum 2 --warmup 500 --steps 20000 --out out/king_reach_n1000
```

**Learning-rate cool-down.** The paper trains one constant-rate trunk per dataset size and
branches a short cool-down off it for each update budget. The trunk keeps its checkpoints at 90 %
of each budget:

```bash
torchrun ... --lr-schedule constant --steps 164703 --keep-at 900,1866,3869,8022,16631,34483,71494,148233 \
    --out out/trunk
```

A branch is an out dir whose `checkpoints/last` links the trunk checkpoint; the trainer resumes
from it and decays the rate to zero over the remaining 10 %:

```bash
mkdir -p out/branch_79438/checkpoints
ln -s "$PWD/out/trunk/checkpoints/071494" out/branch_79438/checkpoints/071494
ln -s 071494 out/branch_79438/checkpoints/last
torchrun ... --lr-schedule constant --steps 79438 --decay-steps 7944 --lr-floor 0 --out out/branch_79438
```

Other options used in the paper: `--avg-weights` (a uniform average of the weights over the run,
SWA), `--vlm-init google/paligemma-3b-pt-224` (π0.5's architecture without its robot pretraining:
the VLM from PaliGemma, the action expert random), `--pretrained lerobot/pi0_base` or
`lerobot/smolvla_base` (other model families).

**Real data.** The real datasets train with the same command. Their fisheye views are rectified
to the simulator's pinhole camera on the fly (`camera/`), and `splits/` holds the paper's splits
of the real king moves (8, 32, 128 and 256 demonstrations, see Dataset). Pretraining in simulation means
`--pretrained <a simulation checkpoint's pretrained_model dir>`.

## Evaluation

```bash
python -m chess.sim.eval_e2e --ckpt out/king_reach_n1000 --data king_reach \
    --schedule schedules/hard_v1.json --k 1000 --envs 16 --out evals/king_reach_n1000
```

Each trial starts at the ready pose and ends when the policy is back there after switching to its
last stage, or after 60 s. The grade is a ladder of milestones (touched, gripped, lifted,
transported, placed, placed precisely) plus two failure modes (toppled, timed out). Success is
the king placed on the target square; strict success also requires it to be centred. The schedules
are fixed 1,000-move chains: `easy_v1` places the king at the square centre, `hard_v1` within
±8 mm of it. The policy runs in the deployment loop: 10-action chunks at 15 Hz with real-time
chunking (`--rtc-delay 5`). Results go to `trials.csv` and `meta.json` in the eval dir.
`--offload` runs π0.5 on a 4 GB GPU, with the weights streamed from host memory.

## Dataset

The real teleoperated demonstrations of the paper are at
[Kripi/so101-chess-king-moves-real](https://huggingface.co/datasets/Kripi/so101-chess-king-moves-real):
261 king moves in LeRobot v3 format at 30 fps, with both raw camera views, the joint state and
actions in calibrated degrees (gripper 0–100), the stage marks as `language_persistent` rows and
per-episode grades in `meta/outcomes.json`. `splits/` holds the paper's splits of it.

```bash
python -m chess.train.finetune_real Kripi/so101-chess-king-moves-real --pretrained <checkpoint> \
    --episodes splits/so101-chess-king-moves-real_n128.json ...
```

## Layout

```
arm/          SO-101 kinematics (URDF FK/IK) and joint-unit conversions
model.py      physical constants: joint ranges, rest poses, calibrated camera rigs
assets/       SO-101 model (URDF + MJCF with fitted servo dynamics), scanned king
chess/task.py board geometry, square names, task prompts
chess/real/   the scripted planner shared by the real arm and the simulator; dataset sidecars
chess/sim/    MuJoCo scene, scripted expert, grading, collector, evaluator
chess/train/  fine-tuning and splits
policy/       policy construction, training core, π0.5 stage prediction
camera/       real-camera rectification
schedules/    the evaluation schedules
splits/       the paper's splits of the real king-move dataset
```

## License

Apache 2.0 (see `LICENSE`). The SO-101 model in `assets/so101/` comes from
[SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) (Apache 2.0).

## Cite

If you find this work helpful in your research, cite as:

```bibtex
@misc{kripner2026vla,
  title={VLA Adaptation with Real and Simulated Data: A Case Study on Robotic Chess},
  author={Matěj Kripner and Milan Straka},
  howpublished={CoRL 2026 Workshop on Pretrain to Adapt: What Makes a Pretrained Policy Adaptable?},
  year={2026},
  url={https://corl2026-robotcl.github.io/}
}
```
