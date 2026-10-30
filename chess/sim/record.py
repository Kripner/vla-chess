"""The episode writer behind every sim dataset: `collect` (the scripted expert) and
`eval_e2e --record` (a policy's own rollouts) hand it finished episodes — a qpos trace at the
dataset's frame rate plus the wire action held at each frame — and it renders both cameras,
writes the frames, grades the episode (truth.outcome → outcomes.json), logs the move
(moves.json) and marks the stages (truth.subtask_bounds → segments journal). One writer, so a
demonstration and a rollout end up with the same schema, fps, sidecars and grading.

Episode dict `r`: i, source, target, dx_mm, dy_mm, end ("done" | "timeout" | "toppled" | ...),
qpos (T, nq) float32, act (T, 6) float32 in wire units (5 arm cal-deg + gripper 0-100, see
arm.mapping). Episodes shorter than one second are skipped: lerobot cannot represent a
near-empty episode and a frame of stillness is no data.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from arm import mapping as J
from chess import task as T
from chess.sim import scene, truth
from chess.real import movelog, outcomes, segments
import model as M


def git_sha() -> str:
    repo = Path(__file__).resolve().parents[2]
    try:
        out = subprocess.run(["git", "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, cwd=repo)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except Exception:
        pass
    return "unknown"


def dataset_root(name: str, root: str | None = None) -> tuple[str, Path]:
    """(repo id, root dir) of dataset `name`: $HF_LEROBOT_HOME/<repo id> unless `root` is given."""
    return outcomes.repo_id(name), (Path(root) if root else outcomes.dataset_root(name))


class Recorder:
    """Creates the dataset (refusing an existing one: a dataset is never regenerated in place)
    and writes episodes in the order they are given. `scene_parts` = (model, meta, renderer) to
    reuse a caller's scene (the evaluator's); otherwise the writer builds its own at cam_w × cam_h.
    `sim_extra` is merged into meta/sim.json next to scene, nq and git."""

    def __init__(self, name: str, fps: int, cam_w: int, cam_h: int, sim_extra: dict,
                 root: str | None = None, scene_parts=None):
        import mujoco
        from lerobot.datasets.language import LANGUAGE_PERSISTENT, language_feature_info
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self._lang = LANGUAGE_PERSISTENT
        self.repo_id, self.root = dataset_root(name, root)
        if (self.root / "meta" / "info.json").exists():
            raise SystemExit(f"dataset already exists at {self.root} — pick a new name (protocol: "
                             f"a dataset is never regenerated in place)")
        if scene_parts is None:
            model, meta = scene.build(cam_w=cam_w, cam_h=cam_h)
            renderer = mujoco.Renderer(model, height=cam_h, width=cam_w)
        else:
            model, meta, renderer = scene_parts
        self.model, self.meta, self.renderer = model, meta, renderer
        self.data = mujoco.MjData(model)
        self.fps = int(fps)
        pos_names = list(M.MOTOR_NAMES)
        features = {
            "observation.state": {"dtype": "float32", "shape": [6], "names": pos_names},
            "action": {"dtype": "float32", "shape": [6], "names": pos_names},
            "observation.images.overhead": {"dtype": "video", "shape": [cam_h, cam_w, 3],
                                            "names": ["height", "width", "channels"]},
            "observation.images.wrist": {"dtype": "video", "shape": [cam_h, cam_w, 3],
                                         "names": ["height", "width", "channels"]},
            # no observation.current: the real datasets log the servo currents, the sim has none
            "sim.qpos": {"dtype": "float32", "shape": [model.nq], "names": None},
            LANGUAGE_PERSISTENT: language_feature_info()[LANGUAGE_PERSISTENT],
        }
        self.root.parent.mkdir(parents=True, exist_ok=True)
        # encoder_queue_maxsize=0 = an unbounded frame queue: lerobot's streaming encoder drops
        # frames after a 0.1 s wait when its default 30-frame queue is full, silently apart from
        # a log warning, and the video then ends up shorter than the parquet. Memory: a 640x360
        # frame is 0.7 MB, the encoder keeps up on average, so the queue only absorbs bursts.
        self.ds = LeRobotDataset.create(self.repo_id, self.fps, features, root=str(self.root),
                                        robot_type="so_follower", use_videos=True,
                                        streaming_encoding=True, encoder_queue_maxsize=0)
        (self.root / "meta" / "sim.json").write_text(json.dumps(
            {"scene": "chess.sim.scene", "nq": model.nq, "git": git_sha(), **sim_extra},
            indent=2) + "\n")
        self.counts = {k: 0 for k in (*truth.MILESTONES, *truth.FAILURES)}
        self.marked = 0
        self.ep = 0                 # episodes written (skipped traces write none)

    def write(self, r: dict, piece: str) -> dict | None:
        """Render + write one episode; returns truth.outcome's result, or None when skipped."""
        import mujoco
        qpos, act = np.asarray(r["qpos"], np.float32), np.asarray(r["act"], np.float32)
        if len(qpos) < self.fps:
            # A refused plan (rare reach-envelope corner, e.g. e8 at an extreme offset) ships
            # a ~1-frame fallback trace. lerobot cannot represent a near-empty episode (its
            # stats row lacks the image keys and breaks the batched metadata flush), and one
            # frame of stillness is no data anyway. Real failures keep their full traces and
            # stay in: the funnel is data, a refusal is not.
            print(f"skip trial {r['i']} {r['source']}->{r['target']}: "
                  f"{len(qpos)}-frame trace ({r['end']})", flush=True)
            return None
        src, tgt = T.cell_from_name(r["source"]), T.cell_from_name(r["target"])
        task = T.move_task(src, tgt, piece)
        meta, data, renderer = self.meta, self.data, self.renderer
        for t in range(len(qpos)):
            data.qpos[:] = qpos[t]
            mujoco.mj_forward(self.model, data)
            imgs = {}
            for camkey, cam in (("wrist", scene.WRIST_CAM), ("overhead", scene.OVERHEAD_CAM)):
                renderer.update_scene(data, camera=cam)
                imgs[camkey] = renderer.render()
            deg6 = J.deg_from_qpos(np.asarray(qpos[t][meta["arm_qposadr"]], float))
            state = np.concatenate([deg6[:5], [float(J.grip_deg_to_0_100(deg6[5]))]])
            self.ds.add_frame({
                "observation.state": np.asarray(state, np.float32),
                "action": act[t],
                "observation.images.overhead": imgs["overhead"],
                "observation.images.wrist": imgs["wrist"],
                "task": task,
                "sim.qpos": qpos[t],
                self._lang: None,           # filled from the journal (segments.flush)
            })
        self.ds.save_episode()
        sig = truth.Signals(qpos, meta, piece)
        res = truth.outcome(sig, src, tgt, r["end"])
        for k in self.counts:
            self.counts[k] += int(res[k])
        grade = res["grade"]
        odata = outcomes.load(self.root)
        oc_map = dict(odata.get("outcomes") or {})
        oc_map[str(self.ep)] = grade
        odata["outcomes"] = oc_map
        outcomes.save(self.root, odata)
        movelog.record(self.root, self.ep, {
            "task": task, "piece": piece, "source": r["source"], "target": r["target"],
            "phase": None, "offset_mm": [r["dx_mm"], r["dy_mm"]],
            "board": None, "board_source": r.get("board_source", "expert"), "end": r["end"],
            # the milestone ladder + failure modes; outcomes.json keeps only the
            # good/mid/bad grade (the training contract shared with the real datasets)
            **{k: bool(res[k]) for k in (*truth.MILESTONES, *truth.FAILURES)},
            "grade": grade, "label": truth.label(res), "n_grasps": res["n_grasps"],
            "final_off_mm": res["final_off_mm"]})
        bounds = truth.subtask_bounds(sig, tgt) if r["end"] == "done" else None
        rows = ([segments.row(name, src, tgt, f / self.fps, piece) for name, f in bounds.items()]
                if bounds else [])
        if not rows and r.get("mark_partial"):
            # A policy rollout that does not decompose (a timeout, a toppled piece, a piece never
            # released) still gets marks: truth.LivePhase walks the same predicates causally,
            # retracts when a run breaks, and simply stays in the stage the episode ended in;
            # one row per phase change (a mark row sets the active stage from its time on).
            # Every frame then carries the label the trainer's conditioning needs. Demonstrations
            # keep the strict rule (no marks when the episode does not decompose).
            live, phase = truth.LivePhase(meta, tgt, piece), None
            for f in range(len(qpos)):
                name = live.update(qpos[f])
                if name != phase:
                    rows.append(segments.row(name, src, tgt, f / self.fps, piece))
                    phase = name
        if rows:
            segments.journal_append(self.root, self.ep, rows)
            self.marked += 1
        self.ep += 1
        return res

    def close(self) -> str:
        """Finalize the dataset and flush the marks; returns a one-line summary."""
        self.ds.finalize()
        n_marks, _ = segments.flush(self.root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        grades = outcomes.load(self.root).get("outcomes") or {}
        return (f"{info['total_episodes']} episodes, {info['total_frames']} frames, "
                f"{self.marked} marked ({n_marks} journal rows flushed); "
                f"good {sum(g == 'good' for g in grades.values())}  "
                f"mid {sum(g == 'mid' for g in grades.values())}  "
                f"bad {sum(g == 'bad' for g in grades.values())}")
