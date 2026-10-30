#!/usr/bin/env python3
"""Scripted IK expert as a per-world tick driver: the chess/real/phases plans (shared with the
real arm's scripted expert) plus measured-descent and stall-close feedback, expressed as one
goal per control tick so N worlds can step in lockstep.

Feedback comes straight from the world's qpos (the evaluator hands it in every tick):
  * descent exits when the FK'd TCP reaches the grasp band (droop-compensated ladder),
  * the close stops where the measured jaw stalls on the piece (no servo current in sim),
  * the pick aims at the piece's TRUE position (the trial's committed offset — sim resets
    place it exactly).

The driver ends after the release stages (the trial's grading needs release + settle, and
the evaluator teleport-resets the world, so there is nothing to drive home for) — unless
`home_tail=True` (dataset collection): then it appends the recorded drive back to READY,
the tail whose end supervises the learned place->home switch.

Styles: None = the deterministic expert (the evaluation baseline); "reach" =
chess.sim.human.Style, the randomized expert the paper's simulated demonstrations come from.
"""
from __future__ import annotations

import numpy as np

from arm import mapping as J
from arm import so101_kin as K
from chess.real import phases

TICK_HZ = 30.0            # the expert's control rate (the datasets' frame rate)
# Pacing is TCP-DISTANCE based (uniform tip speed, whatever the leg's joint geometry), with
# a joint-rate cap as the servo-limit floor. Joint-space pacing made the demos' speed vary
# wildly per leg (a pan swing raced, the grasp zone crawled) — the tip speed is the visible,
# imitable quantity.
TRAVEL_SPEED = 0.10       # m/s TCP on free legs (approach, transport, return home)
GRASP_SPEED = 0.04        # m/s TCP inside the grasp zone (descents, engage, lifts, place)
MAX_REL = 1.5             # wire deg per tick joint cap while streaming
CARRY_REL = 0.75          # ... while carrying the piece: full joint rate slings the
                          # marginally-pinched king into a tilt it never recovers from
GRASP_REL = 0.5           # ... inside the grasp zone: a full-rate rise with the open fingers
                          # around the crown clips the piece over
NAMES = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper")

# The real expert's close/descent pacing, converted from seconds to ticks.
_CLOSE_TICKS = max(1, int(round(phases.CLOSE_TICK_S * TICK_HZ)))     # per close step
# The close used to walk each 0.8-wire step in 2 ticks and dwell _CLOSE_TICKS — the same
# cadence emitted as one continuous ramp (the step spread over all 6 ticks): the dwells
# read as stop-go lag in the demo, and the stall check only samples once per step anyway.
_CLOSE_RATE = phases.CLOSE_STEP_WIRE / (_CLOSE_TICKS + 2)            # wire per tick
_RUNG_TICKS = 2 * _CLOSE_TICKS                                       # per measured-descent rung
_DWELL_TICKS = max(1, int(round(0.3 * TICK_HZ)))                     # stage settles
_RELEASE_TICKS = max(1, int(round(1.0 * TICK_HZ)))                   # after opening the jaw


STYLES = ("expert", "reach")


def make_style(name: str, rng):
    """The per-episode style object for `--style name` (None = the deterministic expert;
    "reach" = chess.sim.human.Style, a person as a noisy controller)."""
    if name == "expert":
        return None
    if name == "reach":
        from chess.sim import human
        return human.Style(rng)
    raise ValueError(f"unknown style {name!r} (one of {STYLES})")


def self_clearance(model, meta: dict, min_clear: float = 0.003):
    """Planner `pose_ok` for the sim: True when the gripper body keeps `min_clear` from the
    other arm links. The d8/e8 grasp columns (the two cells in front of the base, wrist
    rolled tangential) fold the gripper into the shoulder — the commanded descent poses sit
    2-7 mm inside it — so the descent stalled 3-6 mm above the band on a marginal grasp.
    Vetoing those solves makes the planner take its next approach rung (+10 deg keeps
    6-12 mm there) or refuse the square."""
    import copy
    import mujoco
    m = copy.copy(model)                       # own margins: contacts appear within min_clear
    d = mujoco.MjData(m)
    grip = m.body("gripper").id
    links, b = set(), m.body_parentid[grip]
    while b > 0:
        links.add(b)
        b = m.body_parentid[b]
    arm_geoms = np.flatnonzero(np.isin(m.geom_bodyid, [grip, *links])
                               & ((m.geom_contype > 0) | (m.geom_conaffinity > 0)))
    m.geom_margin[arm_geoms] = min_clear
    calib = J.SO101Calib()
    qpos = np.asarray(meta["qpos_ready"], float).copy()
    body = m.geom_bodyid

    def ok(q) -> bool:
        wire = [calib.rad_to_deg(mn, float(q[i])) for i, mn in enumerate(J._ARM)] + [0.0]
        qpos[meta["arm_qposadr"]] = J.wire_to_qpos(wire)
        d.qpos[:] = qpos
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        b1 = body[d.contact.geom1[:d.ncon]]
        b2 = body[d.contact.geom2[:d.ncon]]
        return not np.any(((b1 == grip) & np.isin(b2, list(links)))
                          | ((b2 == grip) & np.isin(b1, list(links))))
    return ok


class ExpertDriver:
    """One trial's scripted move. next_wire(qpos_row) -> wire goal [6] or None when done.
    Raises RuntimeError from the planner if a pose is unreachable (the evaluator grades the
    trace it has and moves on)."""

    def __init__(self, planner: phases.Planner, src, tgt, dx_mm: float, dy_mm: float,
                 meta: dict, piece: str = "king", home_tail: bool = False, piece_xy=None,
                 style=None):
        self.pl, self.meta, self.piece = planner, meta, piece
        self.src, self.tgt = tuple(src), tuple(tgt)
        self.off = (dx_mm / 1000.0, dy_mm / 1000.0)   # board frame == scene placement frame
        self.home_tail = home_tail
        self.piece_xy = piece_xy       # () -> live base-frame piece (x, y) | None (sim qpos)
        self.style = style             # None: the deterministic expert (evals, the baseline)
        self.qpos: np.ndarray | None = None
        self._gen = self._run() if style is None else style.run(self)   # the style owns its driver

    def next_wire(self, qpos_row: np.ndarray):
        self.qpos = np.asarray(qpos_row, float)
        return next(self._gen, None)

    # -- measured state (from the qpos the evaluator hands in) ------------------------- #
    def _wire_now(self) -> np.ndarray:
        q6 = self.qpos[self.meta["arm_qposadr"]]
        deg6 = J.deg_from_qpos(np.asarray(q6, float))
        return np.concatenate([deg6[:5], [float(J.grip_deg_to_0_100(deg6[5]))]])

    def _tcp_z(self) -> float:
        return float(self.pl.tcp_z(self._wire_now()))

    def _tcp_of(self, wire) -> np.ndarray:
        """Base-frame TCP position of a wire cursor (FK; paces streams by tip distance)."""
        q = np.array([self.pl.calib.deg_to_rad(m, float(wire[i]))
                      for i, m in enumerate(J._ARM)])
        return np.asarray(K.tcp(q[:5])[0], float)

    # -- goal streaming ---------------------------------------------------------------- #
    def _stream(self, goal: dict, settle_ticks: int = _DWELL_TICKS, rel: float = MAX_REL,
                speed: float = TRAVEL_SPEED):
        """Walk the wire cursor toward `goal` at TCP `speed` (tip-distance pacing: the demo
        moves at one visible speed whatever the leg's joint geometry), floored by the `rel`
        joint-rate cap, all joints scaled to arrive TOGETHER,
        then hold it settle_ticks. Per-joint rate clipping instead desynchronizes large
        joint-space seams — on the transport's settle-onto-hover seam it swung the TCP
        up to 60 mm below the carry line and slung the held piece out of the jaw."""
        tgt = np.array([float(goal[m]) for m in NAMES])
        cur = self._cur
        joint_ticks = int(np.ceil(np.abs(tgt - cur).max() / rel))
        tip_ticks = int(np.ceil(np.linalg.norm(self._tcp_of(tgt) - self._tcp_of(cur))
                                / speed * TICK_HZ))
        travel = max(joint_ticks, tip_ticks)
        for i in range(1, travel + 1):
            yield cur + (tgt - cur) * (i / travel)
        self._cur = tgt.copy()
        for _ in range(settle_ticks):
            yield tgt.copy()

    def _chain(self, legs, settle_ticks: int = 0, ease: str = "smooth"):
        """Stream several plan legs as ONE smooth motion. `legs` = [(goals, rel, speed), ...]:
        the goals concatenate into a single polyline (duplicates dropped — dwell repeats are
        meaningless inside a continuous stream), each
        leg paced by tip distance at its own speed with its joint-rate cap as the floor, and
        the WHOLE polyline is time-warped with a smoothstep (`ease="minjerk"`: the gentler
        quintic): one acceleration, one cruise, one deceleration. The arm is still only where
        stillness is functional (the measured close, the descent's measurement rungs, the
        release dwell, the final READY hold) — every other pause or zig-zag between phases is
        lag the demos would teach."""
        pts = [self._cur.copy()]
        cur = self._cur
        for goals, rel, speed in legs:
            for goal in goals:
                tgt = np.array([float(goal[m]) for m in NAMES])
                if np.abs(tgt - cur).max() < 1e-9:
                    continue
                n = max(int(np.ceil(np.abs(tgt - cur).max() / rel)),
                        int(np.ceil(np.linalg.norm(self._tcp_of(tgt) - self._tcp_of(cur))
                                    / speed * TICK_HZ)))
                for i in range(1, n + 1):
                    pts.append(cur + (tgt - cur) * (i / n))
                cur = tgt
        self._cur = cur.copy()
        N = len(pts) - 1
        seq = []
        for k in range(1, N + 1):           # eased resample: same duration, smooth ends
            t = k / N
            x = (t * t * (3.0 - 2.0 * t) if ease == "smooth"
                 else t * t * t * (10.0 + t * (6.0 * t - 15.0))) * N
            i = min(int(x), N - 1)
            a = x - i
            seq.append(pts[i] * (1.0 - a) + pts[i + 1] * a)
        yield from seq
        for _ in range(settle_ticks):
            yield pts[-1].copy()

    def _run(self):
        pl, piece = self.pl, self.piece
        # next_wire() stores qpos before advancing the generator, so the cursor anchors at
        # the world's actual pose (READY after a reset).
        self._cur = self._wire_now()

        # B: hover AT the piece (the committed offset), stage, descend measured, close on stall.
        # One roll for the whole move (move_roll): the wrist never spins while holding.
        hv = pl.hover(self.src, off=self.off, roll=pl.move_roll(self.src, self.tgt))
        approach, descent, lift, z_band = pl.pick_plan(hv, piece=piece, handoff=False,
                                                        touch=self.piece_xy is not None)
        flat = lambda stages: [goal for st in stages for goal in st]
        # Segment 1: READY -> stage -> straight down to the band, ONE motion (the jaw opens
        # during the travel); the measured walk below the band brings its own rung settles.
        yield from self._chain([(approach[0], MAX_REL, TRAVEL_SPEED),
                                (flat(approach[1:]), GRASP_REL, GRASP_SPEED)])
        engage, still, last = None, 0, None
        for z_cmd, goal, follow in descent:         # the measured-descent walk
            yield from self._stream(goal, settle_ticks=_RUNG_TICKS, rel=GRASP_REL,
                                    speed=GRASP_SPEED)
            engage = follow
            z = self._tcp_z()
            if z <= z_band + phases.GRASP_Z_TOL:
                break
            if last is not None and last - z < 0.4 * phases.GRASP_PUSH_STEP:
                still += 1
                if still >= 2:
                    break                           # something solid under the arm
            else:
                still = 0
            last = z
        if engage is not None:                      # fixed pad sideways against the piece
            yield from self._stream(engage, rel=GRASP_REL, speed=GRASP_SPEED)
        # close on stall (position only: sim serves no current)
        goal = dict(engage if engage is not None else descent[-1][1])
        hold = yield from self._close(goal)
        if hold is None:                            # jaw ran shut: nothing grasped
            return
        yield from self._stream(goal, rel=GRASP_REL, speed=GRASP_SPEED)
        yield from self._carry_place(hv, hold, flat(lift(hold)))

    def _carry_place(self, hv, hold: float, lift_goals):
        """Segments 3-5 from a held piece: lift -> carry -> place -> release -> retreat.
        `hv` = the hover the carry rises from."""
        pl, piece = self.pl, self.piece
        g = phases.real_grasp(piece)
        flat = lambda stages: [goal for st in stages for goal in st]
        # Segment 3a: lift -> carry, ONE motion (per-leg speeds inside a single eased
        # profile; the transport's final settle onto the hover pose is dropped — the place
        # approach eases down from the carry line).
        goals_c, hv2 = pl.transport_plan(hv, self.tgt, hold, piece=piece)
        yield from self._chain([(lift_goals, GRASP_REL, GRASP_SPEED),
                                (goals_c[:-1], CARRY_REL, TRAVEL_SPEED)])
        # Measured place aim: shift the whole place column by where the HANGING piece
        # actually is vs the target centre — the blind place lands every mm of in-jaw
        # migration (and the engage bias) in the placement. Board-frame via the pose's
        # rotation; sanity-capped (a dropped piece reads far away -> place blind).
        # The place is BLIND on purpose: runtime aim corrections were measured worse three
        # ways (absolute error embeds the TCP's remaining approach; in-jaw migration embeds
        # the elastic hang offset, which restores itself at touchdown). The mirrored pick/
        # place geometry — touch-adjusted on both sides — is the accurate estimator.
        # Segment 3b: place approach -> down to the place band, ONE motion
        approach_d, descent_d, release, _ = pl.place_plan(hv2, hold, piece=piece,
                                                          handoff=False,
                                                          touch=self.piece_xy is not None)
        yield from self._chain([(flat(approach_d), GRASP_REL, GRASP_SPEED)])
        z_place = g.z + phases.PLACE_DZ
        still, last = 0, None
        for z_cmd, goal, _f in descent_d:
            yield from self._stream(goal, settle_ticks=_RUNG_TICKS, rel=GRASP_REL,
                                    speed=GRASP_SPEED)
            z = self._tcp_z()
            if z <= z_place + phases.GRASP_Z_TOL:
                break
            if last is not None and last - z < 0.4 * phases.GRASP_PUSH_STEP:
                still += 1
                if still >= 2:
                    break
            else:
                still = 0
            last = z
        # Segment 4: release — the one deliberate dwell (the piece must settle free of the jaw)
        stages = release()
        yield from self._chain([(stages[0], GRASP_REL, GRASP_SPEED)],
                               settle_ticks=_RELEASE_TICKS)
        # Segment 5: back off -> rise -> (collection: drive home) — ONE motion. The recorded
        # home tail ends settled at READY; its end supervises the learned place->home switch.
        retreat = [(flat(stages[1:]), GRASP_REL, GRASP_SPEED)]
        if self.home_tail:
            # Plan the return from the retreat's TOP, not from the arm's current (release)
            # pose 60 mm up: planned from down there, z_travel floors at the traverse height,
            # so the tail rose to the hover, sank 10-40 mm onto that line for a second and
            # only then climbed to READY.
            top = stages[-1][-1]
            retreat.append((self.pl.return_plan([float(top[m]) for m in NAMES], rest="ready"),
                            MAX_REL, TRAVEL_SPEED))
        yield from self._chain(retreat, settle_ticks=_RELEASE_TICKS)

    def _close(self, goal: dict):
        """Close on stall at `goal` (position only: sim serves no current), one continuous
        ramp. Returns the hold command, or None when the jaw ran shut on nothing.
        Stalled = no longer closing. |prev - now| < tol misses the king that rocks a few
        degrees under the closing pad and pushes the jaw BACK 0.1-0.2 wire per step: the
        streak would reset every step and the command run to shut with both pads on the
        piece."""
        st = self.style
        rate = _CLOSE_RATE * (st.close_rate if st is not None else 1.0)
        # a style stops closing once the jaw is well past where the piece should be and
        # nothing has stalled it (a person sees the empty pinch); the expert runs to shut
        give_up = (self.pl.grip_wire(phases.contact_gap(self.piece)) - st.CLOSE_MISS_WIRE
                   if st is not None else -1.0)
        cmd = float(goal["gripper"])
        prev, still, hold, free = self._wire_now()[5], 0, None, 0
        now = prev                               # a jaw commanded shut already: the loop never runs
        while cmd > 0.0:
            cmd = max(0.0, cmd - phases.CLOSE_STEP_WIRE)
            goal["gripper"] = cmd
            yield from self._stream(goal, settle_ticks=0, rel=rate, speed=GRASP_SPEED)
            now = self._wire_now()[5]
            still = still + 1 if now > prev - phases.CLOSE_STALL_WIRE else 0
            prev = now
            if still >= phases.CLOSE_STALL_TICKS and now > cmd + phases.CLOSE_STALL_WIRE:
                hold = max(0.0, now - phases.CLOSE_PRELOAD_WIRE)
                break
            free = free + 1 if (now < give_up and still == 0) else 0
            if st is not None and free >= st.CLOSE_MISS_TICKS:
                return None
        if hold is None and now > phases.CLOSE_PRELOAD_WIRE:   # ran to shut, jaw still blocked
            hold = max(0.0, now - phases.CLOSE_PRELOAD_WIRE)
        if hold is not None:
            goal["gripper"] = hold
        return hold
