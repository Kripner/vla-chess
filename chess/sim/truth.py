#!/usr/bin/env python3
"""Ground truth from a simulator trace: the normative implementation of the study's
grade (six milestones + two failure modes) and of the sub-task boundaries.

One module, two consumers — so a collected episode and an eval trial are judged by
exactly the same code:
  * chess/sim/record.py — auto-grades every collected episode and auto-marks its sub-tasks
                       from the `sim.qpos` trace it records (no operator keys in sim).
  * chess/sim/eval_e2e.py — grades each trial from the trace it accumulates.

Everything here is derived from privileged simulator state (the per-frame qpos: the arm
joints and the piece's free joint). Nothing reads images, actions, or the policy's
beliefs, so the verdict cannot be talked into agreeing with the policy.

The trace is `qpos[T, nq]` plus the scene meta from chess.sim.scene.build().
"""
from __future__ import annotations

import numpy as np

from arm import mapping as J
from arm import so101_kin as K
from chess import task as T

# --- thresholds (geometry-derived; each one is a physical statement) ---------------- #
NEAR_M = 0.030           # TCP within this of the piece's grip point = "at the piece"
LIFT_M = 0.015           # piece base this far above the board = airborne (clear of it)
PLACE_EPS = 0.006        # piece base within this of the board when the jaws open = set down
CLOSE_MARGIN = 0.006     # jaw gap under (piece width + this) = closed on the piece
UPRIGHT_COS = 0.966      # ~15 deg tilt
TOPPLED_COS = 0.5        # > 60 deg tilt = fallen. Far past the tipping angle (~22 deg for the king:
                         # atan(base radius / centre-of-mass height)), and no policy of ours rights a
                         # piece, so nothing that happens afterwards can raise the grade: eval_e2e
                         # ends the trial at the first such frame and grades the trace as it stands.
OVER_PIECE_M = 0.040     # TCP within this of the piece's axis (xy) = over it. 40, not 20: a human-style
                         # misaimed grasp (12-25 mm off, corrected part-way down) slides in from outside
                         # 20 mm and closed on the piece with no arrival run at all — 44 of 1769
                         # successful human-style episodes had no pick mark. The
                         # deterministic expert's marks move under 1 frame.
APPROACH_R = 0.090       # ...and within this in 3D = closing on it, not just parked overhead. The hover
                         # sits 86-90 mm from the grip point and the expert's staging pause 25 mm
                         # above it at ~110: this cap is what keeps the staging pause out of the
                         # pick's arrival run (150 mm let it in and the lift never got "back" to that
                         # height — 56/196 episodes lost their marks). A hover a few mm
                         # past the cap only starts the run a few mm into the descent, which the
                         # DESCEND_M mark absorbs.
DESCEND_M = 0.005        # a span that opens with a descent (pick, place) starts once the TCP is this
                         # far below the height it ARRIVED at (over the piece / over the target square).
                         # 5, not 10: a human-style release 15-30 mm above the square descends less
                         # than 10 mm from its arrival over the target (15 of 1769 successful
                         # human-style episodes had no place mark); the expert's place
                         # marks move 6 frames earlier, its pick marks under 1.
LATERAL_M = 0.010        # the carry begins once the held piece is clear of the board and the TCP has
                         # moved this far sideways from where it gripped. Not "the TCP is back within
                         # RETURN_M of the height it ARRIVED over the piece at": true for the expert's
                         # hover, false for a human-style arc that arrives over the crown higher than
                         # it carries (27 short moves never got "back": no transport mark). Against
                         # that older rule the expert's transport marks move 13 frames later (the top
                         # of the lift instead of 20 mm below it).
RETURN_M = 0.020         # home starts once the piece is released and the TCP is back within this of
                         # the height it arrived over the target square at: the retreat is complete.
                         # (READY sits over the middle of the board, so xy alone is not enough)

# The grade: a ladder of six milestones,
# each including the one before, plus two failure modes. success = placed_strict with neither.
MILESTONES = ("touched", "gripped", "lifted", "transported", "placed_rough", "placed_strict")
FAILURES = ("timed_out", "toppled")
JAW_BODIES = ("gripper", "moving_jaw_so101_v1")   # a contact between these and the piece = touching

SEQUENCE = ("approach", "pick", "transport", "place", "home")   # the five sub-tasks, in order


def up_cos(quat):
    """Cosine of the piece's tilt from vertical, from its (w, x, y, z) orientation — the z
    component of its rotated up axis. Works on one quaternion or a (T, 4) trace."""
    q = np.asarray(quat, dtype=float)
    return 1.0 - 2.0 * (q[..., 1] ** 2 + q[..., 2] ** 2)


class Signals:
    """Per-frame ground truth derived from a qpos trace (all arrays are length T)."""

    def __init__(self, qpos, meta, piece: str = "king"):
        q = np.asarray(qpos, dtype=float)
        if q.ndim != 2:
            raise ValueError(f"qpos trace must be (T, nq), got {q.shape}")
        self.n = len(q)
        self.q = q
        arm_q, king_q = meta["arm_qposadr"], meta["king_qposadr"]
        self.board_pose = meta["board_pose"]
        info = T.piece_info()[piece]
        self.piece_r = float(info["base_radius"])
        grip_w = 2.0 * float(info.get("grip_radius", info["base_radius"]))

        self.king_pos = q[:, king_q:king_q + 3]
        quat = q[:, king_q + 3:king_q + 7]
        self.up_cos = up_cos(quat)
        self.upright = self.up_cos >= UPRIGHT_COS
        self.toppled = self.up_cos < TOPPLED_COS       # fallen: the trial can end here

        # piece in board-local coords -> cell + intra-cell offset
        bx, by, yaw = self.board_pose
        r = np.deg2rad(yaw)
        dx, dy = self.king_pos[:, 0] - bx, self.king_pos[:, 1] - by
        self.local_x = dx * np.cos(r) + dy * np.sin(r) + T.BOARD_CENTER[0]
        self.local_y = -dx * np.sin(r) + dy * np.cos(r) + T.BOARD_CENTER[1]

        # arm: jaw opening + the TCP (the point the fingers close on), by forward kinematics
        self.gap = np.array([float(J.gap_for_grip_qpos(v)) for v in q[:, arm_q[5]]])
        self.tcp = np.array([K.tcp(row)[0] for row in q[:, arm_q[:5]]])

        # the piece's grip point: its own axis at the height the jaws meet it
        gz = T.grasp_z(piece)
        self.grip_pt = self.king_pos + np.array([0.0, 0.0, gz])
        self.d_piece = np.linalg.norm(self.tcp - self.grip_pt, axis=1)

        self.near = self.d_piece < NEAR_M
        self.closed = self.gap < (grip_w + CLOSE_MARGIN)
        self.airborne = self.king_pos[:, 2] > (T.BOARD_TOP + LIFT_M)
        self.over_piece = np.linalg.norm(self.tcp[:, :2] - self.king_pos[:, :2], axis=1) < OVER_PIECE_M
        self.in_jaws = self.near & self.closed          # gripped (on the board or not)
        self.carried = self.in_jaws & self.airborne     # gripped AND clear of the board
        self.standing = (self.king_pos[:, 2] <= T.BOARD_TOP + PLACE_EPS) & self.upright
        self._touch = None                              # (T, 2) jaw-piece contacts, replayed on demand

    # -- per-frame predicates the sub-task boundaries are defined by ----------------- #
    def going_for_piece(self) -> np.ndarray:
        """The TCP is over the piece's axis AND within reach of it: reaching for the piece,
        as opposed to merely parked somewhere above the board (READY is over its middle)."""
        return self.over_piece & (self.d_piece < APPROACH_R)


    def carried_ever(self) -> bool:
        return bool(self.carried.any())

    def last_release(self) -> int | None:
        """Frame at which the piece last left the jaws after a carry, or None if it never
        did (still held at the end)."""
        if not self.carried_ever():
            return None
        last_grip = int(np.flatnonzero(self.in_jaws)[-1])
        return None if last_grip >= self.n - 1 else last_grip

    def cell_at(self, i: int):
        return T.cell_of(float(self.local_x[i]), float(self.local_y[i]))

    def off_mm_at(self, i: int):
        cell = self.cell_at(i)
        if cell is None:
            return None
        cx, cy = T.cell_center(*cell)
        return ((float(self.local_x[i]) - cx) * 1000.0, (float(self.local_y[i]) - cy) * 1000.0)


    def _contacts(self) -> np.ndarray:
        """(T, 2): each jaw body (JAW_BODIES order) in contact with the piece, per frame.
        Contacts are a function of the positions alone, so they are replayed from the stored
        qpos through MuJoCo's collision pass (one row per control tick: a graze shorter than a
        tick is invisible). The jaw gap cannot tell a grip from an empty close: fully shut jaws
        read 2 mm, a held king anywhere from 4 to 14 mm (the foam jaws sink in)."""
        if self._touch is None:
            import mujoco
            m = _model()
            d = mujoco.MjData(m)
            body = m.geom_bodyid
            king = m.body(_scene().KING_BODY).id
            jaws = [m.body(n).id for n in JAW_BODIES]
            out = np.zeros((self.n, len(jaws)), dtype=bool)
            for t, row in enumerate(self.q):
                d.qpos[:] = row
                mujoco.mj_kinematics(m, d)
                mujoco.mj_collision(m, d)
                if d.ncon:
                    b1 = body[d.contact.geom1[:d.ncon]]
                    b2 = body[d.contact.geom2[:d.ncon]]
                    for j, jb in enumerate(jaws):
                        out[t, j] = bool(np.any(((b1 == king) & (b2 == jb)) | ((b2 == king) & (b1 == jb))))
            self._touch = out
        return self._touch

    @property
    def touching(self) -> np.ndarray:
        """Either jaw in contact with the piece, per frame."""
        return self._contacts().any(axis=1)

    @property
    def pinched(self) -> np.ndarray:
        """Both jaws in contact with the piece at once: it is between them."""
        return self._contacts().all(axis=1)

    def over_cell(self, cell) -> np.ndarray:
        """The piece's centre inside `cell`, per frame (the square test the grade uses)."""
        cx, cy = T.cell_center(*cell)
        return (np.abs(self.local_x - cx) <= T.CELL / 2) & (np.abs(self.local_y - cy) <= T.CELL / 2)

    def n_grasps(self) -> int:
        """Separate runs of in_jaws: 1 for a clean pick, more when the piece slipped and was re-taken."""
        g = self.in_jaws.astype(int)
        return int(((g[1:] == 1) & (g[:-1] == 0)).sum() + (1 if g[0] else 0))


_MODEL = None


def _scene():
    from chess.sim import scene
    return scene


def _model():
    """The canonical scene's MjModel, built once per process, for contact replay."""
    global _MODEL
    if _MODEL is None:
        _MODEL = _scene().build(cam_w=64, cam_h=36)[0]
    return _MODEL


def outcome(sig: Signals, src_cell, tgt_cell, end: str = "done") -> dict:
    """The grade of one trial or episode.

    Six milestones, a ladder — each includes the one before, so their rates are monotone and
    the funnel is the count at each rung:
      touched        a jaw touched the piece
      gripped        both jaws in contact with the piece at once (it is between them)
      lifted         gripped, piece 15 mm clear of the board
      transported    lifted, piece centre over the target square
      placed_rough   transported, and at the END the piece stands out of the jaws with its
                     centre in the target square
      placed_strict  placed_rough, base entirely inside the square
    Placement is judged on the resting piece at the end, so it needs no separate rules for
    sliding (placed needs transported needs lifted), putting the piece back (its centre is not
    in the target square) or still holding it (it must be out of the jaws).

    Two failure modes: timed_out (the evaluator's `end`) and toppled (tilt past TOPPLED_COS at
    any point). A trial can also fail by finishing wrongly — misplaced, or the piece still in
    hand — which needs no flag of its own: the milestones say how far it got.

    Derived: success = placed_strict with neither failure; lenient = placed_rough with neither;
    grade good/mid/bad (outcomes.json) = success / lenient / the rest.
    `src_cell` is kept for the call sites; the source square plays no role in the grade.
    """
    last = sig.n - 1
    over_tgt = sig.over_cell(tgt_cell)
    r = {}
    r["touched"] = bool(sig.touching.any())
    r["gripped"] = r["touched"] and bool(sig.pinched.any())
    r["lifted"] = r["gripped"] and bool(sig.carried.any())
    r["transported"] = r["lifted"] and bool((sig.carried & over_tgt).any())
    r["placed_rough"] = (r["transported"] and bool(sig.standing[last]) and not bool(sig.in_jaws[last])
                        and bool(over_tgt[last]))
    off = sig.off_mm_at(last)
    inside = (T.CELL / 2 - sig.piece_r) * 1000.0      # base entirely inside the square
    r["placed_strict"] = (r["placed_rough"] and off is not None
                         and max(abs(off[0]), abs(off[1])) <= inside)
    r["timed_out"] = end == "timeout"
    r["toppled"] = bool(sig.toppled.any())
    ok = not (r["timed_out"] or r["toppled"])
    r["success"] = r["placed_strict"] and ok
    r["lenient"] = r["placed_rough"] and ok
    r["grade"] = "good" if r["success"] else "mid" if r["lenient"] else "bad"
    end_cell = sig.cell_at(last)
    r["final_cell"] = T.cell_name(end_cell) if end_cell else ""
    r["final_off_mm"] = off
    r["n_grasps"] = sig.n_grasps()
    return r


def label(res: dict) -> str:
    """One word for logs: success | lenient | the highest milestone reached (+failure modes)."""
    if res["success"]:
        return "success"
    if res["lenient"]:
        return "lenient"
    top = next((m for m in reversed(MILESTONES) if res[m]), "nothing")
    fails = [f for f in FAILURES if res[f]]
    return top + ("+" + "+".join(fails) if fails else "")


class OverTarget:
    """"The piece is above the target square", as a Schmitt trigger.

    It ARRIVES when the piece's centre crosses into the square — the same square test that
    decides the grade (chess.task.cell_of), so a placement that misses the square never
    arrives at all — and it stays "above" until the base has cleared the square entirely.

    The hysteresis is the piece's own base radius, and it is what makes transport -> place
    robust: a carried piece wobbles a few mm in the jaws, so a bare border test decides the
    boundary inside that wobble (measured: 5 mm of wobble against a 15.4 mm border, i.e. 3 mm
    of margin). Widening the square instead would be worse — a padded square reaches into its
    neighbours, and a piece put down on the wrong square would still count as having arrived.

    One definition, two consumers: subtask_bounds runs it over a finished trace, LivePhase
    steps it frame by frame, so the chip and the dataset cannot disagree.
    """

    def __init__(self, cell, piece_r: float):
        self.cx, self.cy = T.cell_center(*cell)
        self.h_in = T.CELL / 2                        # centre inside the square = arrived
        self.h_out = T.CELL / 2 + piece_r             # base clear of the square = left
        self.on = False

    def step(self, local_x: float, local_y: float) -> bool:
        dx, dy = abs(local_x - self.cx), abs(local_y - self.cy)
        self.on = ((self.on and dx <= self.h_out and dy <= self.h_out)
                   or (dx <= self.h_in and dy <= self.h_in))
        return self.on

    def trace(self, sig: Signals) -> np.ndarray:
        """The same trigger run over a whole recorded trace (restarts it)."""
        self.on = False
        return np.array([self.step(x, y) for x, y in zip(sig.local_x, sig.local_y)])


def _final_run_start(mask, end: int, lo: int = 0) -> int | None:
    """Start of the contiguous True run of `mask` that ends at frame `end` (clamped to `lo`),
    or None when `mask` is False at `end`.

    "Final run", not "first crossing": the boundary is the last, committed attempt. The TCP
    often sweeps over the piece on its way in and drifts off again, and the operator fumbles
    a grasp or two; taking the first crossing would open the sub-task at that fly-by. Walking
    back from the event that ENDS the sub-task (the grasp, the release) skips every aborted
    attempt and lands on the one that led somewhere.
    """
    if not bool(mask[end]):
        return None
    i = int(end)
    while i > lo and bool(mask[i - 1]):
        i -= 1
    return int(i)


def _first(lo: int, hi: int, mask) -> int | None:
    """First frame in [lo, hi) where `mask` holds, or None."""
    idx = np.flatnonzero(mask[lo:hi])
    return int(lo + idx[0]) if len(idx) else None


def subtask_bounds(sig: Signals, tgt_cell) -> dict[str, int] | None:
    """Start frame of each sub-task, from the same trace — the automatic replacement for the
    operator's mark key. None when the episode has no clean linear structure to annotate.

    Each boundary is an ARRIVAL, and each is a statement about the ground truth alone (no
    operator key presses, nothing about the arm's commanded goal):

      approach   frame 0. Every episode starts at READY, on its way to the piece.
      pick       the TCP arrived over the piece (the start of the final run of "over the
                 piece's axis and within reach, or already holding it" that ends at the
                 first grasp) and has come DESCEND_M down from the height it arrived at:
                 the descent onto the piece has begun.
      transport  the piece is held and clear of the board and the TCP has moved LATERAL_M
                 sideways from where it gripped: the lift is complete, the carry begins.
      place      the piece arrived over the target square (the start of the final run of
                 `OverTarget(target)`, which see, that ends at the last frame it was held)
                 and the TCP has come DESCEND_M down from the height it arrived at: the
                 descent onto the square has begun.
      home       the piece has left the jaws for the last time and the TCP is back within
                 RETURN_M of the place-arrival height: the retreat is complete, the return
                 to READY begins. Not the release, and not the operator's End press.

    Pick and place are thus "down, act, up", transport and home the legs at height, and both
    learned switches fire with the arm high (with, or without, the piece). Starting transport
    at lift-off (15 mm) and home at the release instead labels the last 1.5 s of every lift
    transport and the 4 s vertical retreat after the release home — both switches then sit on
    the jaw event, the least visible instant of the move (measured on 196 expert episodes).
    A 20 mm pick cylinder, 10 mm descents and a transport that waits for the TCP to return
    within RETURN_M of the pick-arrival height are rules fitted to the deterministic expert's
    hover that the human style breaks three ways (a corrected miss slides in from outside
    20 mm, an arc arrives over the crown higher than a short carry ever gets, a high release
    descends under 10 mm): under them 86 of 1769 successful human-style episodes had no marks,
    75 of which decompose under the current rules.

    Returns None (no marks, and the writer says so) rather than a guess when the episode
    does not decompose: nothing was ever carried, the piece is still held at the end, it was
    not over its target square when released, or a descent / a return never happened.
    """
    if not sig.carried_ever():
        return None
    rel = sig.last_release()                          # last frame the piece was in the jaws
    if rel is None:                                   # never let go: no place, no home
        return None
    grip = np.flatnonzero(sig.in_jaws)
    first_grip, first_carry = int(grip[0]), int(np.flatnonzero(sig.carried)[0])

    # holding it counts as "on it": the TCP wobbles off the piece's axis during a fumbled
    # grasp, and those frames belong to the pick, not to the approach.
    pick_arr = _final_run_start(sig.going_for_piece() | sig.in_jaws, first_grip)
    place_arr = _final_run_start(OverTarget(tgt_cell, sig.piece_r).trace(sig), rel, lo=first_carry)
    if pick_arr is None or place_arr is None:
        return None
    z = sig.tcp[:, 2]
    z_pick, z_place = float(z[pick_arr]), float(z[place_arr])
    pick = _first(pick_arr, first_grip + 1, z <= z_pick - DESCEND_M)
    moved = np.linalg.norm(sig.tcp[:, :2] - sig.tcp[first_grip, :2], axis=1) >= LATERAL_M
    transport = _first(first_carry, place_arr, sig.carried & moved)
    place = _first(place_arr, rel + 1, z <= z_place - DESCEND_M)
    home = _first(rel + 1, sig.n, ~sig.in_jaws & (z >= z_place - RETURN_M))
    if None in (pick, transport, place, home):
        return None

    b = {"approach": 0, "pick": pick, "transport": transport, "place": place, "home": home}
    # strictly increasing: a degenerate span would make a nonsensical annotation — say so
    # instead of guessing.
    fr = [b[k] for k in SEQUENCE]
    if any(y <= x for x, y in zip(fr, fr[1:])) or fr[-1] > sig.n - 1:
        return None
    return b


class LivePhase:
    """The same five boundaries, decided as the episode happens instead of afterwards, so a live
    readout says what the writer is about to annotate (record.py also walks it over a trace to
    mark a policy rollout that subtask_bounds rejects).

    subtask_bounds() takes the start of the FINAL run of each arrival condition — the run that
    ends at the grasp, or at the release — and marks the first frame of it at which the descent
    is under way. Causally that is just: note the TCP height when the run starts, advance once
    the TCP is DESCEND_M below it, and forget the height if the run breaks before the event that
    locks it in (you fall back to the previous sub-task). transport compares against the TCP's
    position at the first grasp, home against the noted place-arrival height. So this advances
    on exactly the predicates subtask_bounds walks back through, and retracts on exactly the
    runs it would skip; the two cannot disagree.
    """

    def __init__(self, meta, tgt_cell, piece: str = "king"):
        self.meta, self.piece = meta, piece
        self.over_tgt = OverTarget(tgt_cell, float(T.piece_info()[piece]["base_radius"]))
        self.i = 0                                    # index into SEQUENCE
        self.z_pick = None                            # TCP height on arriving over the piece (this run)
        self.xy_grip = None                           # TCP xy at the first grasp (subtask_bounds: first_grip)
        self.z_place = None                           # TCP height on arriving over the target (this run)

    def name(self) -> str:
        """The sub-task the episode is in right now (what the live chip shows)."""
        return SEQUENCE[self.i]

    def update(self, qpos_row) -> str:
        """Feed one control tick; returns the current sub-task name."""
        s = Signals(np.asarray(qpos_row)[None, :], self.meta, self.piece)
        going, held = bool(s.going_for_piece()[0]), bool(s.in_jaws[0])
        carried, z = bool(s.carried[0]), float(s.tcp[0, 2])
        over_tgt = self.over_tgt.step(float(s.local_x[0]), float(s.local_y[0]))
        if self.i == 0:
            if going or held:                         # over the piece: note the arrival height once
                if self.z_pick is None:
                    self.z_pick = z
                if z <= self.z_pick - DESCEND_M:
                    self.i = 1                        # descending onto it: the pick has begun
            else:
                self.z_pick = None                    # the run broke: still approaching
        elif self.i == 1:
            if held and self.xy_grip is None:
                self.xy_grip = s.tcp[0, :2].copy()    # where it gripped (noted once, like first_grip)
            if carried and float(np.linalg.norm(s.tcp[0, :2] - self.xy_grip)) >= LATERAL_M:
                self.i = 2                            # lifted and moving sideways: the carry begins
            elif not (going or held):
                self.i, self.z_pick = 0, None         # broke off: still approaching
        elif self.i == 2:
            if over_tgt:                              # over the square: note the arrival height once
                if self.z_place is None:
                    self.z_place = z
                if z <= self.z_place - DESCEND_M:
                    self.i = 3                        # descending onto it: the place has begun
            else:
                self.z_place = None
        elif self.i == 3:
            if not held and z >= self.z_place - RETURN_M:
                self.i = 4                            # released and back up: the return begins
            elif held and not over_tgt:
                self.i, self.z_place = 2, None        # drifted off the square: still carrying
        elif self.i == 4 and held:
            self.i = 3                                # picked it up again to re-place it
        return self.name()
