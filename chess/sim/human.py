"""The 'reach' style: the randomized scripted expert, a person as a noisy controller.

`collect --style reach` / `eval_e2e --expert --style reach`. A scripted path with jittered
parameters keeps the structure a planner has and a hand does not — vertical columns, one
smooth profile per leg.

What a hand does instead: aim at a point with a perception error, launch a ballistic reach
with a bell-shaped speed profile, look, and fire a smaller correction, until the residual is
under a tolerance set by how careful the person is being. That loop is the whole model here
(the optimized-submovement model of motor control), and it produces every human signature at
once: curved, never-vertical approaches; a main movement then one to three shrinking
corrections; undershoot; speed that varies inside a movement; small pauses while looking; and
mistakes that are the same mechanism at a bad draw (a sloppy closing tolerance, a hasty look)
rather than a catalogue of scripted errors. A miss is seen at the jaw (the close meets
nothing) and answered by a reopen and a fresh look, never a seconds-long empty pinch.

The move is six reaches in task space, anchored on the planner's own grasp geometry
(phases.pick_plan / place_plan give the descend-beside, engage and place points, the wrist
roll the move can be solved at, the back-off and rise after a release): READY -> a point
above the crown -> down beside the piece and engage -> close -> up and over to a point above
the target (the lift and the carry overlap, so the path is the swung curve a hand makes)
-> down onto the square -> release -> back off and rise -> READY, snapped exactly at the end.
`Style` holds the person drawn per episode; `run` is the driver generator (expert.ExpertDriver
supplies the tick-level machinery: the wire cursor, FK/IK, the stall-close, the stream).
"""
from __future__ import annotations

import numpy as np

from arm import mapping as J
from arm import so101_kin as K
from chess.real import phases

from chess.sim.expert import (GRASP_REL, GRASP_SPEED, MAX_REL, NAMES, TICK_HZ, TRAVEL_SPEED,
                                 _RELEASE_TICKS)


class Style:
    """The person, drawn once per episode from `rng` (default_rng([seed, trial]))."""

    TEMPO = (0.7, 1.3)               # speed factor on every reach (log-uniform)
    V_TRAVEL = 0.12                  # m/s mean tip speed of a free reach at tempo 1
    V_FINE = 0.05                    # m/s ... of a descent / engage / place reach
    T_MIN = 0.25                     # s, the shortest sub-movement (a small correction)
    T_JITTER = 0.15                  # +- fraction on each sub-movement's duration
    LATENCY = (0.08, 0.25)           # s, the look before a correction starts (log-uniform)
    HESITATE, P_HESITATE = (0.3, 1.0), 0.12   # s, a look that takes longer, rarely
    OVERLAP = (0.3, 0.6)             # the next look happens this fraction before the movement
                                     # ends, so the correction — or the next reach — launches
                                     # while it is still decelerating: one continuous motion
    GAIN = (0.3, 0.5)                # the perception error shrinks by this per correction
    MAX_CORR = 4                     # corrections per reach, then the person moves on
    ERR_COARSE = (0.015, 0.035)      # m, first-look error of a free reach (per person)
    ERR_FINE = (0.004, 0.012)        # m, first-look error of a descent / place (per person)
    ERR_Z = 0.25                     # the vertical error is this fraction of the lateral one
    TOL_COARSE = (0.010, 0.025)      # m, "close enough" for a free reach
    TOL_FINE = (0.0015, 0.006)       # m, "close enough" to close the jaw / let go: the care.
                                     # The jaw admits +-5 mm (OPEN_MARGIN_REAL): the sloppy
                                     # end of this range is where misses and edge grasps come from
    PRE_MARGIN = (0.04, 0.09)        # m above the planner's column top the approach aims at (the
                                     # free reach ends there; the align then lowers onto the top)
    LIFT = (0.02, 0.06)              # m straight up before the carry takes over
    ROLL_JITTER = np.radians(15.0)   # wrist roll around the planner's move roll
    OPEN_MARGIN = (-0.004, 0.008)    # m on OPEN_MARGIN_REAL: how wide the jaw opens
    OPEN_FRAC = (0.4, 1.0)           # the last fraction of the approach the jaw opens over
    RELEASE_EXTRA = (0.0, 8.0)       # wire, the release opened wider than the descent aperture
    RELEASE_DWELL = (0.3, 0.8)       # s, jaw open before the retreat
    CLOSE_RATE = (0.7, 1.5)          # close-ramp factor (ExpertDriver._close)
    CLOSE_MISS_WIRE = 8.0            # ExpertDriver._close: jaw this far past where the piece
    CLOSE_MISS_TICKS = 3             #   should be, still moving freely: an empty pinch, give up
                                     #   (4 wire cuts real grasps short)
    RETRIES = 2                      # empty pinches before the person gives the grasp up

    def __init__(self, rng: np.random.Generator):
        self.rng = rng
        u = rng.uniform
        self.tempo = self._logu(*self.TEMPO)
        self.overlap = float(u(*self.OVERLAP))
        self.gain = float(u(*self.GAIN))
        self.err_coarse = float(u(*self.ERR_COARSE))
        self.err_fine = float(u(*self.ERR_FINE))
        self.tol_coarse = float(u(*self.TOL_COARSE))
        self.tol_fine = float(u(*self.TOL_FINE))
        self.pre_margin = float(u(*self.PRE_MARGIN))
        self.lift = float(u(*self.LIFT))
        self.roll_jitter = float(u(-self.ROLL_JITTER, self.ROLL_JITTER))
        self.open_margin = phases.OPEN_MARGIN_REAL + float(u(*self.OPEN_MARGIN))
        self.open_frac = float(u(*self.OPEN_FRAC))
        self.release_extra = float(u(*self.RELEASE_EXTRA))
        self.release_dwell = int(round(float(u(*self.RELEASE_DWELL)) * TICK_HZ))
        self.close_rate = float(u(*self.CLOSE_RATE))

    def _logu(self, lo: float, hi: float) -> float:
        return float(np.exp(self.rng.uniform(np.log(lo), np.log(hi))))

    def perceive(self, scale: float) -> np.ndarray:
        """A perception error (base frame, m): lateral `scale`, vertical ERR_Z of it."""
        e = self.rng.normal(0.0, 1.0, size=3) * scale
        e[2] *= self.ERR_Z
        return e

    def latency(self) -> int:
        """Ticks of looking before the next sub-movement starts; rarely a hesitation."""
        s = self._logu(*self.LATENCY)
        if self.rng.uniform() < self.P_HESITATE:
            s += float(self.rng.uniform(*self.HESITATE))
        return max(1, int(round(s * TICK_HZ)))

    def duration(self, dist: float, v: float) -> int:
        """Ticks of a sub-movement over `dist` m at mean speed `v` * tempo, jittered."""
        s = max(self.T_MIN, dist / max(v * self.tempo, 1e-6))
        s *= 1.0 + float(self.rng.uniform(-self.T_JITTER, self.T_JITTER))
        return max(2, int(round(s * TICK_HZ)))

    def run(self, drv):
        return run(drv, self)


def _rad(wire) -> np.ndarray:
    """Wire [6] -> the 5 arm joint radians the sim will actually take, unclamped: the planner's
    hover on the tangential-roll cells sits 0.12 rad past the wrist-flex limit, and clamping
    to the calibration range would leave 19 mm of phantom error no correction can remove."""
    w = np.asarray(wire, float)
    return np.asarray(J.urdf_from_deg(np.concatenate([w[:5], [float(J.grip_0_100_to_deg(w[5]))]]))[0], float)


def _fk(q) -> np.ndarray:
    """Base-frame TCP of 5 arm joint radians."""
    return np.asarray(K.tcp(np.asarray(q, float)[:5])[0], float)


def _tcp_of(wire) -> np.ndarray:
    return _fk(_rad(wire))


def _minjerk(u: float) -> float:
    u = min(1.0, max(0.0, u))
    return u * u * u * (10.0 - 15.0 * u + 6.0 * u * u)


class _Motion:
    """The episode's in-flight movements: joint-space displacements superposed on a base
    pose, each a minimum-jerk profile from its start tick. One object lives through the
    whole move, so a reach's last movement blends into the next reach's first instead of
    stopping at every boundary. Gripper commands are events on the same clock."""

    def __init__(self, q0, grip: float):
        self.q0, self.subs, self.grips, self.t = np.asarray(q0, float).copy(), [], [], 0
        self.g_init = float(grip)

    def rebase(self, q, grip: float):
        """After something else moved the arm (the stall-close, a stream): start over here."""
        self.q0, self.subs, self.grips, self.g_init = np.asarray(q, float).copy(), [], [], float(grip)

    def end_q(self) -> np.ndarray:
        return self.q0 + sum((dq for _, _, dq in self.subs), np.zeros(5))

    def q_at(self, t: int) -> np.ndarray:
        return self.q0 + sum((dq * _minjerk((t - t0) / T) for t0, T, dq in self.subs), np.zeros(5))

    def grip_at(self, t: int) -> float:
        g = self.g_init
        for t0, t1, g0, g1 in self.grips:
            if t >= t0:
                g = g0 + (g1 - g0) * min(1.0, (t - t0) / max(t1 - t0, 1))
        return g

    def done(self, t: int) -> bool:
        return (all(t >= t0 + T for t0, T, _ in self.subs)
                and all(t >= t1 for _, t1, _, _ in self.grips))


def _tick(drv, mo: _Motion):
    """Command the motion's pose for this tick; advance the clock."""
    goal = drv.pl.goal(mo.q_at(mo.t), mo.grip_at(mo.t))
    drv._cur = np.array([float(goal[m]) for m in NAMES])
    mo.t += 1
    return drv._cur.copy()


def _settle(drv, mo: _Motion):
    """Run the in-flight movements out (before the jaw closes, before a stream)."""
    while not mo.done(mo.t):
        yield _tick(drv, mo)


def _reach(drv, mo: _Motion, st: Style, goal_fn, roll: float, tol: float, err: float, v: float,
           grip=None, grip_frac: float = 1.0, via=None, precise: bool = False, seed=None):
    """Sub-movements toward goal_fn() (base frame, m) until the motion will end within `tol`
    of it: each is a minimum-jerk joint displacement to a solved pose, launched at the
    person's latency after a look, the next look OVERLAP before it ends, the perception error
    shrinking by GAIN per correction. Returns at the look that finds the residual under
    `tol` — the last movement is still in flight and the next reach's first overlaps it.
    `grip` = (from, to) wire gripper over the last `grip_frac` of the first movement;
    `via` = a displacement launched first (a lift the carry then overlaps); `precise` = the
    planner's full solve (approach ladder, orientation checked — a position-only solve lets
    the wrist pitch drift along a descent until the pads come down on the crown), seeded
    from `seed` (the planner's own nearby pose: from READY's configuration the ladder solve
    lands on a wrist-flex the calibration clips); free reaches use the fast position-only
    solve seeded from the motion's predicted end, so the wrist turns with the movement."""
    pl = drv.pl
    solve = (lambda p, s: pl._solve(p, s, roll)) if precise else (lambda p, s: pl._solve_fast(p, s, roll))
    launched, scale, n_corr = 0, err, 0

    def launch(q_t, q_e, dist: float):
        nonlocal launched
        t0 = mo.t + (st.latency() if (launched or mo.subs) else 0)
        T = st.duration(dist, v)
        mo.subs.append((t0, T, q_t - q_e))
        if launched == 0 and grip is not None:
            mo.grips.append((t0 + int(round((1.0 - grip_frac) * T)), t0 + T, float(grip[0]), float(grip[1])))
        launched += 1
        return t0 + int(T * (1.0 - st.overlap))

    q_e = mo.end_q()
    s0 = q_e if seed is None else np.asarray(seed, float)
    next_look = mo.t
    if via is not None:
        q_t = solve(_fk(q_e) + np.asarray(via, float), s0)
        if q_t is not None:
            next_look = launch(q_t, q_e, float(np.linalg.norm(via)))
            s0 = q_t
    while True:
        if mo.t >= next_look:
            q_e = mo.end_q()
            p_e = _fk(q_e)
            resid = goal_fn() - p_e
            if (float(np.linalg.norm(resid)) <= tol and (launched or mo.subs)) or n_corr > st.MAX_CORR:
                if grip is not None and not launched:
                    # no movement carried the jaw event (nothing solved): the jaw still moves,
                    # else a retry would call _close on a closed jaw
                    mo.grips.append((mo.t, mo.t + st.duration(0.0, v), float(grip[0]), float(grip[1])))
                return
            target = goal_fn() + st.perceive(scale)
            q_t = solve(target, s0)
            n_corr += 1
            scale *= st.gain
            if q_t is None:
                next_look = mo.t + 1                                  # nothing solves there: look again
                continue
            next_look = launch(q_t, q_e, float(np.linalg.norm(target - p_e)))
            s0 = q_t
        yield _tick(drv, mo)


def run(drv, st: Style):
    """The reach-style move as ExpertDriver's generator (see the module docstring)."""
    pl, piece = drv.pl, drv.piece
    drv._cur = drv._wire_now()
    mo = _Motion(_rad(drv._cur), float(drv._cur[5]))
    up = np.asarray(pl.T_base_board[:3, 2], float)                 # board normal, base frame
    tcp = lambda goal: _tcp_of(np.array([float(goal[m]) for m in NAMES]))
    g = phases.real_grasp(piece, st.open_margin)
    open_w = pl.grip_wire(g.open_gap)
    roll0 = pl.move_roll(drv.src, drv.tgt)
    for i, roll in enumerate((roll0 + st.roll_jitter, roll0)):    # a jittered roll the planner
        try:                                                       # cannot solve falls back
            hv = pl.hover(drv.src, off=drv.off, roll=roll)
            lean = pl.place_lean_off(drv.tgt, piece)
            hv2 = pl.hover(drv.tgt, seed=hv.q, roll=roll, off=(float(lean[0]), float(lean[1])))
            approach, descent, _, _ = pl.pick_plan(hv, piece=piece, handoff=False,
                                                   touch=drv.piece_xy is not None,
                                                   open_margin=st.open_margin)
            approach_d, descent_d, release, _ = pl.place_plan(hv2, open_w, piece=piece,
                                                              handoff=False,
                                                              touch=drv.piece_xy is not None,
                                                              open_margin=st.open_margin)
            break
        except RuntimeError:
            if i == 1:
                raise
    # the anchors: the planner's column top and its descend-beside / engage points at the
    # piece, the target's column top and place point, the back-off and rise after a release
    def high(p, q_seed):
        """The approach's end: up to PRE_MARGIN above `p`, as high as the fast IK still solves
        (the far ranks run out of reach). A straight free reach from READY that ended AT the
        column top came in low and sideways over the far ranks and swept the king."""
        m = st.pre_margin
        while m > 0.005:
            if pl._solve_fast(p + up * m, q_seed, roll) is not None:
                return p + up * m
            m *= 0.5
        return p
    p_top = tcp(approach[0][-1])
    p_high = high(p_top, hv.q)
    p_desc, p_eng = tcp(descent[0][1]), tcp(descent[0][2] if descent[0][2] is not None else descent[0][1])
    p_top2 = tcp(approach_d[0][0])
    p_high2 = high(p_top2, hv2.q)
    p_place = tcp(descent_d[0][1])
    stages = release()
    p_back, p_rise = tcp(stages[1][0]), tcp(stages[1][1])
    q_of = lambda goal: _rad(np.array([float(goal[m]) for m in NAMES]))
    q_desc, q_place, q_rise = q_of(descent[0][1]), q_of(descent_d[0][1]), q_of(stages[1][1])
    wide = float(stages[1][0]["gripper"])
    ready = phases.rest_goal("ready")
    p_ready = tcp(ready)
    xy0 = np.asarray(drv.piece_xy(), float) if drv.piece_xy is not None else None

    def live(p):
        """`p` shifted by how far the piece has moved since the start (a nudged piece)."""
        if xy0 is None:
            return p
        d = np.asarray(drv.piece_xy(), float) - xy0
        return p + np.array([d[0], d[1], 0.0])

    err_desc = st.err_fine * st.gain * st.gain     # what is left after aligning from above
    # 1. over the piece, the jaw opening on the way; then align above it — the fine lateral
    # corrections happen at height, where nothing can be knocked over (correcting while
    # descending knocks the king over). From here
    # to the lift the solves are the planner's precise ones: the wrist pitch must hold.
    yield from _reach(drv, mo, st, lambda: p_high, roll, st.tol_coarse, st.err_coarse, st.V_TRAVEL,
                      grip=(float(drv._cur[5]), open_w), grip_frac=st.open_frac)
    yield from _reach(drv, mo, st, lambda: live(p_top), roll, st.tol_fine, st.err_fine, st.V_FINE,
                      precise=True, seed=hv.q)
    # 2. straight down beside it, engage the fixed pad, close; an empty pinch is seen and retried
    hold = None
    for attempt in range(st.RETRIES + 1):
        yield from _reach(drv, mo, st, lambda: live(p_desc), roll, st.tol_fine, err_desc, st.V_FINE,
                          precise=True, seed=q_desc)
        yield from _reach(drv, mo, st, lambda: live(p_eng), roll, st.tol_fine, err_desc, st.V_FINE,
                          precise=True, seed=q_desc)
        yield from _settle(drv, mo)
        goal = {m: float(drv._cur[i]) for i, m in enumerate(NAMES)}
        hold = yield from drv._close(goal)
        mo.rebase(_rad(drv._cur), float(drv._cur[5]))
        if hold is not None:
            break
        if attempt < st.RETRIES:                                   # reopen, come up a little, look again
            yield from _reach(drv, mo, st, lambda: live(p_desc) + up * 0.5 * st.lift, roll,
                              st.tol_fine, st.err_fine, st.V_FINE, grip=(float(drv._cur[5]), open_w),
                              precise=True, seed=hv.q)
    if hold is None:
        return
    # 3-4. lift, and the carry over to the target overlapping it
    yield from _reach(drv, mo, st, lambda: p_high2, roll, st.tol_coarse, st.err_coarse, st.V_TRAVEL,
                      via=up * st.lift)
    # 5. align above the square, straight down onto it, let go, a short dwell
    yield from _reach(drv, mo, st, lambda: p_top2, roll, st.tol_fine, st.err_fine, st.V_FINE,
                      precise=True, seed=hv2.q)
    yield from _reach(drv, mo, st, lambda: p_place, roll, st.tol_fine, err_desc, st.V_FINE,
                      precise=True, seed=q_place)
    yield from _settle(drv, mo)
    open_goal = {m: float(drv._cur[i]) for i, m in enumerate(NAMES)}
    open_goal["gripper"] = open_w + st.release_extra
    yield from drv._stream(open_goal, settle_ticks=st.release_dwell, rel=GRASP_REL,
                           speed=GRASP_SPEED * st.tempo)
    mo.rebase(_rad(drv._cur), float(drv._cur[5]))
    # 6. back the pad off and rise clear, jaw wide; then home, exactly
    yield from _reach(drv, mo, st, lambda: p_rise, roll, st.tol_coarse, st.err_fine, st.V_FINE,
                      grip=(float(drv._cur[5]), wide), via=p_back - _tcp_of(drv._cur), precise=True,
                      seed=q_rise)
    if not drv.home_tail:
        yield from _settle(drv, mo)
        return
    yield from _reach(drv, mo, st, lambda: p_ready, phases.HOME_ROLL, st.tol_coarse, st.err_coarse,
                      st.V_TRAVEL, grip=(float(drv._cur[5]), float(ready["gripper"])))
    yield from _settle(drv, mo)
    yield from drv._stream(ready, settle_ticks=_RELEASE_TICKS, rel=MAX_REL, speed=TRAVEL_SPEED)
