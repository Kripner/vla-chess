"""Scripted IK phases of the decomposed chess move: A (hover over the source), C (carry to
the target hover), E (return to rest), plus scripted B/D (pick_plan/place_plan, the IK-expert
baseline). Square -> base frame goes through the measured board pose (board_pose.json, fitted
on the real arm); joint goals are host-wire dicts {motor: cal-deg, gripper: 0-100}.
Wrist roll is pinned at HOME_ROLL in every solve, except where a -90 deg tangential wrist is
measured necessary (NEAR_TANGENT_R reach core + SLIP_TANGENT_CELLS): there the grasp column
AND its hover pin to -90, and the travel legs blend the roll along the line — the wrist never
spins in place while holding a piece. Reach map: 64/64 squares hover and plan a king grasp
(the near-rank c8-f8 columns only solve rolled).
Run `python -m chess.real.phases [piece]` for the offline reach map over the current pose.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]

import model as M
from arm import so101_kin as K
from chess import task as T
from chess.real.expert import (FINGERTIP, HOME_ROLL, _approach_for, approach_ladder,
                               cap_to_cell, grasp_geometry)
from arm import mapping as J

DEFAULT_POSE = Path(__file__).parent / "board_pose.json"

# Hover ladder and transit margin live in chess.task (shared with the sim's phased expert,
# which cannot import this module); re-exported here for real callers.
_KING_TOP = T.piece_info()["king"]["height"]              # m, the scanned real king
HOVER_NEAR_CM, HOVER_FAR_CM = T.HOVER_NEAR_CM, T.HOVER_FAR_CM
hover_fingertip_z = T.hover_fingertip_z
HOVER_DROP_STEP = 0.005   # m: if the ramped hover height won't IK-solve, drop it in these steps
HOVER_DROP_MAX = 0.020    # m: ... down to this much below the target; first solve wins


def real_grasp(piece: str, margin: float | None = None):
    """Grasp geometry the real arm plans with: the sim expert's, opened by OPEN_MARGIN_REAL
    (`margin` overrides it: the human style's per-episode aperture) and capped at one cell
    (only the knight hits the cap)."""
    return cap_to_cell(grasp_geometry(piece, OPEN_MARGIN_REAL if margin is None else margin))


def contact_gap(piece: str) -> float:
    """Finger gap (m) at which the closing jaw should meet `piece` if it sits at the square centre;
    a close trace that passes this with no current means no piece between the pads."""
    g = real_grasp(piece)
    return g.e_grasp + 0.5 * (g.grip_gap + 0.001)      # + GRIP_SQUEEZE back out of the grip gap


_REST_TICKS = {"ready": M.READY_TICKS, "away": M.AWAY_TICKS, "home": M.HOME_TICKS}


def rest_goal(rest: str = "ready") -> dict:
    """Rest pose ("ready"/"away"/"home") as a host-wire goal, from model's *_TICKS."""
    return goal_from_wire(J.wire_from_ticks(_REST_TICKS[rest]))


def _stage_z(piece: str) -> float:
    """TCP height (m) above the board at which the grasp may move sideways: fingertip clear of
    the piece's top by STAGE_CLEAR."""
    return T.piece_info()[piece]["height"] + FINGERTIP + STAGE_CLEAR


GOAL_CADENCE_S = 0.25    # s between streamed goals, fixed: stretched past ~4x the host's 1 s smoothing
                         #   window the motion becomes a move-and-settle staircase; speed is TIP_SPEED
TIP_SPEED = 0.08         # m/s scripted tip speed on the arm (plans stream undilated; the speed divider
                         #   is the policy's knob); every leg's waypoint spacing derives from it
TRANSPORT_STEP = TIP_SPEED * GOAL_CADENCE_S   # m between waypoints, loaded/short legs (C line)
CARRY_STEP = TIP_SPEED * GOAL_CADENCE_S       # m between waypoints, long level legs (A, E, bowl)
# Piece-safe transit heights. IK reach map: every square solves to >= 12.5 cm fingertip (h8), most 18+.
TRANSIT_MARGIN = T.TRANSIT_MARGIN    # carried-piece clearance over a standing piece (phase C)
A_TRAVERSE_MIN = 0.080 + FINGERTIP   # phase A traverse floor (TCP): fingertip 8 cm, 1.1 cm over a king
SCRIPTED_GRIP = 0.0      # gripper (0-100) held in phases A/C = the skill's hand-off aperture; the
                         #   scripted phases and the learned skill must agree. Closed: an open jaw knocks pieces in transit.
# Collection-only hover jitter (Planner(jitter=True)), uniform per axis in the board frame: demos
# must show the skill correcting an off-centre hand-off. +-8 mm is ~26% of a cell, same square.
JITTER_XY = 0.008
JITTER_Z = 0.005
GOAL_STEP_DEG = 30.0     # max per-arm-joint motion per streamed goal (densify): what the follower covers
                         #   in one cadence tick; a bigger goal makes the servo the pacer, the divider inert
SETTLE_TOL_DEG = 4.5     # per-arm-joint cal-deg for "arrived"; not a precision gate: the P-controlled
                         #   STS3215 holds a steady-state gravity error (>3 deg carrying a piece)

# ---- scripted pick/place (the IK-expert baseline: B and D without a policy); grasp geometry is
# the sim expert's (chess.real.expert.grasp_geometry) through the measured board pose ----
OPEN_MARGIN_REAL = 0.010 # m the jaw opens wider than the piece = +-5 mm placement tolerance; past
                         #   ~12 mm it crosses the 15.5 mm half-cell and knocks neighbours
RETREAT_GAP = 0.045      # m the jaw opens for the rise after a release: ~12 mm clear of the
                         #   released king a side (king only). Crosses the
                         #   half-cell — a crowded board would have to rise at open_gap instead

# On the real arm the close is a feedback loop: step the jaw shut until the piece
# stops it, then hold a bounded preload (a commanded gap is only right for a solid of revolution).
CLOSE_STEP_WIRE = 0.8    # 0-100 gripper units per tick while closing (~1.2 mm of gap)
CLOSE_TICK_S = 0.12      # s per close tick (~10 mm/s): slow enough that a stall means the piece, not lag
CLOSE_STALL_WIRE = 0.15  # jaw movement per tick below this = stalled (free motion moves ~0.8)
CLOSE_STALL_TICKS = 3    # consecutive stalled ticks that count as contact (rejects one bad read)
CLOSE_PRELOAD_WIRE = 2.0 # hold this far past contact (~3 mm): the overdrive is the grip force (bounded
                         #   by the host's Max_Torque_Limit); fully shut would cook the servo over a carry
GRASP_PUSH_M = 0.020     # m below the grasp band the measured-height descent may command: the settle
                         #   tolerance hides ~2 cm of droop; the walk stops when the measurement hits the band
GRASP_PUSH_STEP = 0.002  # m per rung (each rung costs 2 IK solves + a settle)
GRASP_PUSH_BELOW = 0.010 # m below the board surface the fingertip may be commanded while chasing the
                         #   droop; only the command goes there, the measured height and stall stop it
GRASP_Z_TOL = 0.002      # m: measured TCP within this of the band counts as arrived
GRASP_FLOOR = 0.001      # m: the band never takes the fingertip closer than this to the board (pawn band 1.7 mm)
PLACE_DZ = 0.002         # m: release the piece this far above its grasp height (sim expert's drop)
STAGE_CLEAR = 0.008      # m: fingertip clearance over the piece's top for the staging/lift rung
GRASP_DWELL = 4          # goal repeats (x GOAL_CADENCE_S ~ 1 s) at a gripper transition: the settle
                         #   test watches arm joints only, so open/close need their own dwell
LIFT_STEP = 0.008        # m per Cartesian rung between a grasp column's band and its stage
                         #   height, lifting AND lowering: as a single joint-space leap the
                         #   TCP bows up to 19 mm PERPENDICULAR to the jaw at tangential-roll
                         #   columns (the interp arc is radial; at home roll it lies along
                         #   the jaw and the pinch absorbs it) — it sheared the held king
                         #   sideways across the pads (measured g7 drops, the piece walking to
                         #   -5.5 mm perp by the lift top). Rungs keep the path straight.

# Grasp columns — and the hovers above them, so a move never spins the wrist while HOLDING
# the piece (a stationary 90 deg spin at the hover migrates the pad contacts and the king
# walks ~5 mm down the jaw: a measured lift-slip at g7) — run with the wrist rolled -90 deg (jaw
# tangential, wrist camera swung toward the board and away from the robot body; never +90,
# which parks the camera at the base) exactly where measured necessary; HOME_ROLL else:
NEAR_TANGENT_R = 0.11    # m from the pan axis — reach necessity: at home roll the jaw axis
                         #   here points at the robot, the desc offset folds past the reach
                         #   limit (d8/e8 solved no column at all) and the stage-to-band stream
                         #   swung the fixed pad across the crown (c8/f8 closed on air).
SLIP_TANGENT_CELLS = frozenset({"g6", "g7", "h3", "h4", "h5"})
                         # grip retention, measured per cell with the CURRENT jaw pads: the
                         #   home-roll near-radial pinch loses the king in transit at these
                         #   sources (g6/g7 0/31 lenient, h3-h5 drop_move ~3/16); the old
                         #   pads had the twin pocket at c3-c5 (the mirrored jaw face), so
                         #   RE-MEASURE this set whenever the pad collision meshes change.
                         #   Cells that tolerate home keep it — rotating healthy neighbours
                         #   measurably hurt (h7 16/16 -> 11/16).
ROLL_BISECTS = K.ROLL_BISECTS   # halvings between home roll and the free solve's roll (~3 deg resolution)
_IK_W_ORI = 0.02         # soft orientation weight: position solves to mm, approach may tilt a few deg
_IK_TOL_POS = 2e-3       # m: IK position residual accepted for a hover/waypoint
_IK_TOL_ANG = 0.35       # rad: approach-direction residual accepted (soft pose cap ~20 deg)
_IK_AE_STOP = 0.15       # rad: stop ik()'s seed hunt at this approach error — the stretched-cell
                         #   plateau is ~0.12 whatever the seed (hunting more seeds there just
                         #   quadruples the cost), while a >0.15 first-seed solve is a basin miss
                         #   and the hunt must go on (accepting 0.35 outright mis-grasped h3)
_IK_ITERS = 40           # LM iterations per planning solve: same poses as the default 120, ~3x faster
_IK_TOL_FAST = 3e-3      # m: carry/travel via-point position tolerance (the host ramp smooths them)


class Hover:
    """A solved hover: `q` (5 arm rad, roll pinned), `z` (TCP height above the board, m),
    `cell` (row, col), `off` (board-frame jitter (dx, dy, dz) it was drawn with), `roll`
    (the wrist roll it was pinned at — the grasp column below it pins to the same)."""

    def __init__(self, q, z, cell, off=(0.0, 0.0, 0.0), roll=HOME_ROLL):
        self.q, self.z, self.cell = q, float(z), tuple(cell)
        self.off = tuple(float(v) for v in off)
        self.roll = float(roll)


class Planner:
    """Square -> joint-space goals through the measured board pose. Loads board_pose.json and
    the follower calibration once."""

    def __init__(self, board_pose: str | Path | None = None, jitter: bool = False,
                 pose_ok=None):
        pose = json.loads(Path(board_pose or DEFAULT_POSE).read_text())
        self.T_base_board = np.asarray(pose["T_base_board"], float)
        self.calib = J.SO101Calib()
        self._rng = np.random.default_rng() if jitter else None
        # Optional q[5 rad] -> bool veto on every posed solve (_solve_at): the sim passes its
        # self-clearance test (chess.sim.expert.self_clearance); a vetoed solution makes the
        # ladder try its next approach rung, exactly like an IK miss.
        self.pose_ok = pose_ok

    # ---- geometry ----
    def square_xyz(self, cell, h: float, off=(0.0, 0.0)) -> np.ndarray:
        """Base-frame TCP target `h` m above the centre of `cell` (board frame: origin at the
        board centre on the surface, +X = col+, +Y = row+), shifted by board-frame `off` (dx, dy)."""
        r, c = cell
        p = np.array([(c - T.HALF) * T.CELL + off[0], (r - T.HALF) * T.CELL + off[1], h, 1.0])
        return (self.T_base_board @ p)[:3]

    def _offsets(self):
        """Board-frame (dx, dy, dz) hover offsets to try, best first: dead centre, or with jitter
        the drawn offset and then dead centre (a jittered pose must not fail a reachable square)."""
        if self._rng is None:
            return [(0.0, 0.0, 0.0)]
        u = self._rng.uniform(-1.0, 1.0, 3) * np.array([JITTER_XY, JITTER_XY, JITTER_Z])
        return [tuple(u), (0.0, 0.0, 0.0)]

    def _approach_ladder(self, y: float, lean: float = 0.0):
        """Approach directions to try at base-frame `y`, smallest deviation first, against this
        board's centre (chess.real.expert.approach_ladder); `lean` turns the whole ladder."""
        return approach_ladder(y, float(self.T_base_board[1, 3]), lean)

    # Does a pose put the TCP within tol of p? Needed because the TCP sits 7.9 mm off the roll axis.
    _reaches = staticmethod(K.reaches)

    @classmethod
    def _pin_roll(cls, q, p, tol: float, roll: float = HOME_ROLL):
        """Re-solve `q` with the wrist roll at `roll` if that still reaches `p`, else the smallest
        roll away from it that does (so101_kin.pin_roll). Never overwrite q[4] instead."""
        return K.pin_roll(q, p, tol, roll, w_ori=_IK_W_ORI, iters=_IK_ITERS)

    def _solve_at(self, p, appr, seed=None, roll: float = HOME_ROLL):
        """IK at `p` with one given approach direction, roll pinned; None if it misses the
        tolerances. A grasp sequence must hold the approach constant or the fingers sweep sideways."""
        q, pe, ae = K.ik(p, approach=appr, q0=seed, w_ori=_IK_W_ORI, iters=_IK_ITERS,
                         ae_accept=_IK_AE_STOP)
        if pe > _IK_TOL_POS or ae > _IK_TOL_ANG:
            return None
        q = self._pin_roll(q, p, _IK_TOL_POS, roll)
        if q is not None and self.pose_ok is not None and not self.pose_ok(q):
            return None
        return q

    def _reach_core(self, cell) -> bool:
        """True inside NEAR_TANGENT_R of the pan axis, where only the tangential column
        physically reaches (pick AND place alike)."""
        p = self.square_xyz(cell, 0.0)
        return np.hypot(p[0] - K.PAN_ORIGIN[0], p[1] - K.PAN_ORIGIN[1]) < NEAR_TANGENT_R

    def move_roll(self, src_cell, tgt_cell=None) -> float:
        """ONE wrist roll for a whole scripted move: -90 deg from HOME_ROLL (tangential jaw,
        camera board-side) exactly when needed — the SOURCE requires it (reach core, or a
        SLIP_TANGENT_CELLS member: the home pinch loses the piece during lift/carry there) or
        the TARGET is in the reach core (home cannot place there at all; a slip cell as a
        target is fine at home — placing holds nothing long). The roll never changes while
        the piece is held: rotating the pinch around a hanging piece migrates the pad
        contacts and walks it down the jaw whatever the speed (measured at g7); the
        wrist returns home during the empty-jaw phases instead."""
        need = (self._reach_core(src_cell) or T.cell_name(src_cell) in SLIP_TANGENT_CELLS
                or (tgt_cell is not None and self._reach_core(tgt_cell)))
        return HOME_ROLL - np.pi / 2 if need else HOME_ROLL

    def _solve(self, p, seed=None, roll: float = HOME_ROLL):
        """IK over the approach-lean ladder, best lean first; None if every rung fails."""
        for appr in self._approach_ladder(float(p[1])):
            q = self._solve_at(p, appr, seed, roll)
            if q is not None:
                return q
        return None

    def _solve_fast(self, p, seed, roll: float = HOME_ROLL):
        """Position-only IK for a carry/travel via point (single seed, loose tolerance, ~15x faster
        than _solve). Returns q (roll pinned) or None; the caller skips it and the ramp bridges it.
        Travel legs pass a per-point `roll` lerped between their endpoints' rolls, so any wrist
        rotation blends into the motion instead of spinning in place (holding a piece, a
        stationary spin walks it down the jaw)."""
        q, pe, _ = K.ik(p, approach=_approach_for(float(p[1])), q0=seed, w_ori=_IK_W_ORI,
                        iters=_IK_ITERS, tol=_IK_TOL_FAST, pos_only=True)
        if pe > _IK_TOL_FAST:
            return None
        return self._pin_roll(q, p, _IK_TOL_FAST, roll)

    def hover(self, cell, seed=None, off=None, roll: float | None = None) -> Hover:
        """Hover above `cell` at the rank-ramped fingertip height (hover_fingertip_z), dropped in
        HOVER_DROP_STEP steps if the IK can't solve that high, offset by the collection jitter.
        `off` pins the board-frame (dx, dy) instead: the sim expert aims at the piece's TRUE
        position (sim ground truth), not the cell centre — the whole grasp column follows it.
        `roll` pins the wrist (a move-level move_roll); default = this cell's own requirement.
        Raises RuntimeError if nothing in the band solves."""
        if roll is None:
            roll = self.move_roll(cell)
        top = hover_fingertip_z(cell) + FINGERTIP        # TCP height above the surface
        offsets = ([(float(off[0]), float(off[1]), 0.0)] if off is not None
                   else self._offsets())
        for off in offsets:
            for i in range(int(round(HOVER_DROP_MAX / HOVER_DROP_STEP)) + 1):
                h = top + off[2] - i * HOVER_DROP_STEP
                q = self._solve(self.square_xyz(cell, h, off), seed, roll)
                if q is not None:
                    return Hover(q, h, cell, off, roll)
        raise RuntimeError(f"square {T.cell_name(cell)} is out of reach (no hover rung solves)")


    def goal(self, q, grip: float) -> dict:
        """5 arm rad + gripper (0-100) -> the host-wire goal {motor: cal-deg | 0-100}."""
        g = {m: self.calib.rad_to_deg(m, float(q[i])) for i, m in enumerate(J._ARM)}
        g["gripper"] = float(grip)
        return g

    def grip_wire(self, gap: float) -> float:
        """Finger gap (m) -> host-wire gripper command (0-100), via the jaw-angle -> cal-deg chain
        the sim expert's actions use."""
        return float(J.grip_deg_to_0_100(self.calib.rad_to_deg("gripper", K.jaw_for_gap(gap))))

    def place_lean_off(self, cell, piece: str = "king"):
        """Board-frame (dx, dy) aim shift for placing on a far row: the approach leans
        _approach_for's theta toward base +Y there, so the piece held (GRASP_Z + FINGERTIP)
        below the TCP lands its base lever*tan(theta) beyond the TCP aim — measured
        +11.2 mm at rank 1, +5.7 at rank 2 (per-rank placement sweep; 53 mm * tan 12 deg
        = 11.3 predicted). Aim short by exactly that lever. Pick columns stay unshifted:
        the close self-aligns the standing piece (source ranks grade flat)."""
        p = self.square_xyz(cell, 0.0)
        a = _approach_for(float(p[1]))                              # (0, sin t, -cos t)
        lever = T.GRASP_Z.get(piece, max(T.GRASP_Z.values())) + FINGERTIP
        # The wrist leans in its own sagittal plane, so the lever runs RADIALLY from the
        # base, not along +Y — the measured x-fan (a1 +4.4, h1 -3.0 mm) is its x-component.
        r = p[:2] / max(float(np.linalg.norm(p[:2])), 1e-6)
        v = -lever * (a[1] / max(-a[2], 1e-6)) * r                  # base frame
        return tuple(self.T_base_board[:2, :2].T @ v)


    def return_plan(self, state_wire, grip: float | None = None, rest: str = "ready",
                    to: dict | None = None):
        """Phase E, piece-safe: from the measured pose rise to the travel height, run a Cartesian
        line to over the rest pose's TCP, then settle on the rest joint pose (`rest`, or the
        host-wire goal `to`, e.g. the clock-press staging waypoint). `grip` defaults to the
        measured gripper. Falls back to the bare rest goal when the state doesn't FK."""
        tgt = dict(to) if to is not None else rest_goal(rest)
        try:
            s = list(state_wire)
            held = float(s[5]) if grip is None else float(grip)
            s[5] = J.grip_0_100_to_deg(s[5])
            q0, _ = J.urdf_from_deg(np.asarray(s, float))
            p0 = np.asarray(K.tcp(np.asarray(q0, float))[0], float)
            seed = np.asarray(q0, float)[:5]
            r = [float(tgt[m]) for m in J._ARM] + [float(tgt["gripper"])]
            r[5] = J.grip_0_100_to_deg(r[5])
            qr, _ = J.urdf_from_deg(np.asarray(r, float))
            p_ready = np.asarray(K.tcp(np.asarray(qr, float))[0], float)
        except Exception:
            return [tgt]
        # Clear the highest corner of the (tilted) fitted board plane, not a nominal height.
        z_floor = max(float(self.square_xyz(c, A_TRAVERSE_MIN)[2])
                      for c in ((0, 0), (0, T.N - 1), (T.N - 1, 0), (T.N - 1, T.N - 1)))
        z_travel = max(float(p0[2]), z_floor)
        r0, r1 = float(seed[4]), float(qr[4])       # blend any wrist rotation into the travel
        goals = []
        p_hi = np.array([p0[0], p0[1], z_travel])
        q = self._solve_fast(p_hi, seed, r0)
        if q is not None:                     # rise in place first (skip if unsolvable there)
            goals.append(self.goal(q, held))
            seed = q
        p_tgt = np.array([p_ready[0], p_ready[1], z_travel])
        n = max(2, int(np.ceil(np.linalg.norm(p_tgt - p_hi) / CARRY_STEP)) + 1)
        for a in np.linspace(0.0, 1.0, n)[1:]:
            q = self._solve_fast(p_hi + a * (p_tgt - p_hi), seed, r0 + a * (r1 - r0))
            if q is None:
                continue                      # reach dip mid-line: the ramp bridges it
            goals.append(self.goal(q, held))
            seed = q
        goals.append(tgt)                     # settle onto the target pose (off the board)
        return goals

    def transport_plan(self, src: Hover, tgt_cell, grip: float, piece: str = "king",
                       z_extra: float = 0.0, off_extra=(0.0, 0.0)):
        """Phase C, piece-safe: rise above the source, run a Cartesian line at a height that keeps
        the carried piece's bottom TRANSIT_MARGIN over a standing king, descend onto the target
        hover. `grip` is the held gripper command from phase B. `z_extra` lifts the carry line
        above that floor and `off_extra` (board-frame m) shifts the place aim — the sim expert's
        human style varies both. Returns (goals, target hover)."""
        lean = self.place_lean_off(tgt_cell, piece)
        tgt = self.hover(tgt_cell, seed=src.q, roll=src.roll,   # carry at ONE roll: the source
                         off=(lean[0] + float(off_extra[0]),       # hover's (move_roll pinned
                              lean[1] + float(off_extra[1])))
        grasp = T.GRASP_Z.get(piece, max(T.GRASP_Z.values()))       # it); aim short of the lean
        z_carry = max(src.z, tgt.z, grasp + _KING_TOP + TRANSIT_MARGIN + FINGERTIP) + z_extra
        p_hi0 = self.square_xyz(src.cell, z_carry, src.off)
        p_hi1 = self.square_xyz(tgt.cell, z_carry, tgt.off)
        r0, r1 = float(src.q[4]), float(tgt.q[4])   # equal but for pin deviation; lerp anyway
        goals, seed = [], src.q
        q = self._solve_fast(p_hi0, seed, r0)
        if q is not None:                     # rise in place first (skip if unsolvable there)
            goals.append(self.goal(q, grip))
            seed = q
        n = max(2, int(np.ceil(np.linalg.norm(p_hi1 - p_hi0) / TRANSPORT_STEP)) + 1)
        for a in np.linspace(0.0, 1.0, n)[1:]:
            q = self._solve_fast(p_hi0 + a * (p_hi1 - p_hi0), seed, r0 + a * (r1 - r0))
            if q is None:
                continue                      # reach dip mid-line: the ramp bridges it
            goals.append(self.goal(q, grip))
            seed = q
        goals.append(self.goal(tgt.q, grip))  # descend onto the exact hover pose
        return goals, tgt


    # ---- scripted pick/place (the IK-expert baseline: B and D without a policy) ----
    def _grasp_column(self, hv: Hover, z: float, offs, z_up_min: float, lean: float = 0.0):
        """Solve one grasp column over `hv.cell`: TCP at board height `z` offset along the realized
        jaw axis by each of `offs` (m, toward the fixed pad), plus the same offsets on a height ladder
        from the hover down to `z_up_min`, all on one approach direction (else the descent runs
        diagonally; `lean` pitches it, see approach_ladder). Returns (down, up, at), `at(z2)`
        re-solving the offsets at another height."""
        c = self.square_xyz(hv.cell, z, hv.off)
        roll = hv.roll                               # the hover's roll IS the column's roll
        for appr in self._approach_ladder(float(c[1]), lean):
            q_c = self._solve_at(c, appr, hv.q, roll)
            if q_c is None:
                continue
            js = np.asarray(K.tcp(np.asarray(q_c, float))[2], float)
            js = np.array([js[0], js[1], 0.0])
            js /= max(np.linalg.norm(js), 1e-9)          # fixed-pad side, unit horizontal
            down = [self._solve_at(c + e * js, appr, q_c, roll) for e in offs]
            if not all(q is not None for q in down):
                continue
            for i in range(int(max(0.0, hv.z - z_up_min) / HOVER_DROP_STEP) + 1):
                top = self.square_xyz(hv.cell, hv.z - i * HOVER_DROP_STEP, hv.off)
                up = [self._solve_at(top + e * js, appr, hv.q, roll) for e in offs]
                if all(q is not None for q in up):
                    def at(z2, _appr=appr, _js=js, _seed=down, _roll=roll):
                        c2 = self.square_xyz(hv.cell, z2, hv.off)
                        qs = [self._solve_at(c2 + e * _js, _appr, _seed[k], _roll)
                              for k, e in enumerate(offs)]
                        return qs if all(q is not None for q in qs) else None
                    return down, up, at
        raise RuntimeError(f"no IK for the grasp on {T.cell_name(hv.cell)} "
                           f"({z * 100:.1f} cm above the surface)")

    def _stages(self, steps):
        """(q, grip, repeats) triples -> the stage list run_plan streams, one settle each."""
        return [[self.goal(q, grip)] * n for q, grip, n in steps]

    @staticmethod
    def _col_rungs(at, z_lo: float, z_hi: float):
        """Cartesian rungs [(z, qs)] on a grasp column between the band `z_lo` (exclusive)
        and the stage height `z_hi`, top-down in LIFT_STEP steps; unsolvable rungs are
        skipped (the stream bridges them)."""
        out = []
        z = z_hi
        while z > z_lo + 1e-9:
            qs = at(z)
            if qs is not None:
                out.append((z, qs))
            z -= LIFT_STEP
        return out

    def pick_plan(self, hv: Hover, piece: str = "king", z_off: float = 0.0,
                  handoff: bool = True, touch: bool = False, open_margin: float | None = None,
                  lean: float = 0.0):
        """Phase B, scripted: open the jaw, stage, descend beside the piece, engage the fixed
        pad, close on feedback, lift. Returns (approach, descent, lift, z_band): approach
        = stages settled one by one; descent = [(z_cmd, descend goal, engage goal)] from the band down
        to GRASP_PUSH_M below it, walked until the measured TCP reaches the band; lift(hold) = stages.
        `handoff` keeps the phased contract (open AT the hover, lift back ONTO it — learned
        phases hand over there); False streamlines for the pure expert: open at the stage
        pose, end the lift at the column top (the hover round-trips are visible zig-zags).
        `touch` engages the fixed pad to the piece SURFACE (drops PAD_CLEAR) so the close
        pins the piece with zero sideways shove — only sound when the aim is ground truth
        (the sim expert's oracle hover off); the real arm keeps the clearance. `open_margin`
        and `lean` are the human style's per-episode jaw aperture and approach pitch."""
        g = real_grasp(piece, open_margin)
        open_w = self.grip_wire(g.open_gap)
        # `z_off` lowers the whole column: the retry when the jaw met nothing at the planned band.
        z_band = max(FINGERTIP + GRASP_FLOOR, g.z + z_off)
        from chess.real.expert import PAD_CLEAR
        e_g = g.e_grasp - (PAD_CLEAR if touch else 0.0)
        (q_desc, q_grasp), (q_up_desc, q_up_grasp), at = self._grasp_column(
            hv, z_band, (g.e_desc, e_g), _stage_z(piece), lean)
        # Cartesian rungs between the stage and the band keep lift/lower straight (LIFT_STEP);
        # each chain is ONE stage — a continuous stream, settled once at its end (per-rung
        # settles made a stop-go staircase that shook the piece worse than the bow).
        rungs = self._col_rungs(at, z_band, _stage_z(piece))
        approach = self._stages(
            ([(hv.q, open_w, GRASP_DWELL)] if handoff else [])   # open the jaw at the hover
            + [(q_up_desc, open_w, 1 if handoff else GRASP_DWELL)]  # stage over the descent line
        ) + ([[self.goal(qs[0], open_w) for _z, qs in rungs]]    # straight down, one stream
             if rungs else [])
        descent = [(z, gs[0], gs[1])
                   for z, gs in self._push_ladder(z_band, (q_desc, q_grasp), at, open_w)]
        lift = lambda hold: (
            [[self.goal(qs[1], hold) for _z, qs in reversed(rungs)]   # straight up, one stream
             + [self.goal(q_up_grasp, hold)]]
            + (self._stages([(hv.q, hold, 1)]) if handoff else []))  # ... back onto the hover
        return approach, descent, lift, z_band

    def _push_ladder(self, z_band: float, top, at, grip: float):
        """[(z_cmd, [goal per offset])] from `z_band` down in GRASP_PUSH_STEP steps to GRASP_PUSH_M
        below it. Rungs that don't IK-solve are dropped; the band rung always leads."""
        rungs = [(z_band, [self.goal(q, grip) for q in top])]
        for i in range(1, int(GRASP_PUSH_M / GRASP_PUSH_STEP) + 1):
            z = z_band - i * GRASP_PUSH_STEP
            if z < FINGERTIP - GRASP_PUSH_BELOW:
                break          # the command may go below the surface; the measured height and stall stop the walk
            qs = at(z)
            if qs is not None:
                rungs.append((z, [self.goal(q, grip) for q in qs]))
        return rungs


    def place_plan(self, hv: Hover, grip: float, piece: str = "king",
                   handoff: bool = True, touch: bool = False, open_margin: float | None = None,
                   lean: float = 0.0):
        """Phase D, scripted: from the target hover holding the piece (`grip` = the command B
        closed on) lower it onto the square, release, back the fixed pad off, rise clear.
        Returns (approach, descent, release, open_w); `descent` is the measured-height ladder as in
        pick_plan, `release()` gives the stages after the piece is down. `handoff` as in
        pick_plan: True ends the release back on the hover pose (the phased contract), False
        ends at the risen back-off (the expert's home leg continues from there). `touch`
        mirrors pick_plan's: a touch-engaged grip holds the piece PAD_CLEAR closer to the
        TCP than the modelled e_grasp, so the place column drops the clearance too — else
        every placement lands that constant 1.5 mm off along the jaw axis. `open_margin` and
        `lean` as in pick_plan."""
        g = real_grasp(piece, open_margin)
        open_w = self.grip_wire(g.open_gap)
        z_place = g.z + PLACE_DZ
        from chess.real.expert import PAD_CLEAR
        e_g = g.e_grasp - (PAD_CLEAR if touch else 0.0)
        (q_back, q_place), (q_up_back, q_up_place), at = self._grasp_column(
            hv, z_place, (g.e_desc, e_g), _stage_z(piece), lean)
        # Cartesian rungs keep the held piece's lowering straight (LIFT_STEP), one stream
        rungs = self._col_rungs(at, z_place, _stage_z(piece))
        approach = (self._stages([(q_up_place, grip, 1)])   # over the place point, holding
                    + ([[self.goal(qs[1], grip) for _z, qs in rungs]] if rungs else []))
        # descend on the place offset (the piece is pinned against the fixed pad); release there
        descent = [(z, gs[1], None)
                   for z, gs in self._push_ladder(z_place, (q_back, q_place), at, grip)]
        # Retreat with the jaw WIDE: the released king leans on a pad or stands a few mm off
        # the centre line, and a pad 5-9 mm from its stem caught the crown on the way up and
        # shoved the piece 3-8 mm off centre (20 of 300 placements moved > 2 mm,
        # rank-7/8 targets by the fixed pad, rank-1/2 by the moving one). RETREAT_GAP clears
        # it by ~12 mm a side.
        wide = self.grip_wire(RETREAT_GAP)
        release = lambda: (
            self._stages([(q_place, open_w, GRASP_DWELL)])  # release, let the piece settle
            + [[self.goal(q_back, wide),                    # back the fixed pad off, then rise
                self.goal(q_up_back, wide)]]                # clear — one continuous stream
            + (self._stages([(hv.q, open_w, 1)]) if handoff else []))
        return approach, descent, release, open_w

    # ---- measured-state predicates (deployment's segment-completion detector) ----
    def tcp_z(self, state_wire) -> float:
        """Measured TCP height (m) above the board surface, from a host-wire state."""
        s = list(state_wire)
        s[5] = J.grip_0_100_to_deg(s[5])
        q_urdf, _ = J.urdf_from_deg(np.asarray(s, float))
        z_base = K.tcp(np.asarray(q_urdf, float))[0][2]
        return float(z_base - self.T_base_board[2, 3])


def goal_from_wire(state) -> dict:
    """Host-wire state [6] -> goal dict."""
    return dict(zip(J._ARM, (float(v) for v in state[:5])), gripper=float(state[5]))


if __name__ == "__main__":
    # offline reach map + scripted-grasp invariants over the current board pose, every pose FK-checked
    piece = sys.argv[1] if len(sys.argv) > 1 else "king"
    pl = Planner()
    geom = real_grasp(piece)
    cal = pl.calib
    # per-motor calibrated cal-deg limits of this follower
    lim = {m: sorted(((cal.rmin[m] - cal.mid[m]) * 360.0 / J.MAX_RES,
                      (cal.rmax[m] - cal.mid[m]) * 360.0 / J.MAX_RES)) for m in J._ARM}
    hover_ok, grasp_ok, refused, no_grasp, over = [], [], [], [], []
    for row in range(T.N):
        for col in range(T.N):
            cell, name = (row, col), T.cell_name((row, col))
            try:
                hv = pl.hover(cell)
            except RuntimeError:
                refused.append(name)
                continue
            want = pl.square_xyz(cell, hv.z, hv.off)
            hover_ok.append((name, 1e3 * np.linalg.norm(np.asarray(K.tcp(hv.q)[0]) - want)))
            try:
                approach, descent, lift, _z = pl.pick_plan(hv, piece)
                grip = pl.grip_wire(geom.grip_gap)          # a plausible post-close hold
                p_approach, p_descent, p_release, _ = pl.place_plan(hv, grip, piece)
                stages = (approach + [[g] for _, g, f in descent for g in (g, f) if g]
                          + lift(grip) + p_approach
                          + [[g] for _, g, _f in p_descent] + p_release())
            except RuntimeError:
                no_grasp.append(name)
                continue
            # grasp pose error: e_grasp from the square centre, at the grasp band
            (_, q_grasp), _, _ = pl._grasp_column(hv, geom.z, (geom.e_desc, geom.e_grasp),
                                                  _stage_z(piece))
            d = np.asarray(K.tcp(q_grasp)[0]) - pl.square_xyz(cell, geom.z, hv.off)
            grasp_ok.append((name, 1e3 * (np.hypot(d[0], d[1]) - geom.e_grasp), 1e3 * d[2]))
            # every streamed goal must be inside the follower's calibrated range
            for g in [g for s in stages for g in s]:
                for m in J._ARM:
                    if not lim[m][0] <= g[m] <= lim[m][1]:
                        over.append(f"{name}:{m}={g[m]:.0f}")
    print(f"board pose {DEFAULT_POSE.name}, piece {piece}: {len(hover_ok)}/64 squares hover "
          f"(worst FK residual {max(e for _, e in hover_ok):.1f} mm), "
          f"{len(grasp_ok)} also plan a scripted pick+place")
    if refused:
        print(f"  no hover — out of reach with the wrist roll pinned ({len(refused)}): "
              f"{' '.join(refused)}")
    if no_grasp:
        print(f"  hover but no grasp column ({len(no_grasp)}): {' '.join(no_grasp)}")
    worst = max(grasp_ok, key=lambda r: abs(r[1]) + abs(r[2]))
    print(f"  grasp pose: worst lateral {worst[1]:+.1f} mm / height {worst[2]:+.1f} mm ({worst[0]}); "
          f"jaw opens to {pl.grip_wire(geom.open_gap):.2f} (0-100 wire, "
          f"{1e3 * geom.open_gap:.1f} mm) and then closes ON FEEDBACK until the piece stops it "
          f"— the modelled {1e3 * geom.grip_gap:.1f} mm grip gap is only used for the pad offsets")
    # Where the flat pads pinch: GRASP_Z must not put the pads on a slant (widest slice at the
    # pads' bottom, piece still widening below), which wedges the piece out of the jaw.
    prof = T.piece_info()[piece]["radius_by_5mm_slice"]
    lo = max(0, int((geom.z - FINGERTIP) / 0.005))
    hi = max(lo + 1, min(len(prof), int((geom.z + 0.033) / 0.005) + 1))
    r = max(prof[lo:hi])
    at = [5 * k for k in range(lo, hi) if prof[k] == r]
    wedge = at == [5 * lo] and lo > 0 and prof[lo - 1] > r
    print(f"  placement budget: the open jaw straddles the piece on its midline, so the {piece} may "
          f"sit ±{1e3 * (geom.open_gap - geom.grip_gap - 0.001) / 2:.1f} mm off the square centre "
          f"(along the jaw) and still be caught; the open jaw reaches ±{1e3 * geom.open_gap / 2:.1f} mm "
          f"vs the {1e3 * T.CELL / 2:.1f} mm half-cell, and contact is expected at "
          f"{1e3 * contact_gap(piece):.1f} mm = wire {pl.grip_wire(contact_gap(piece)):.1f}")
    print(f"  pads span z {5 * lo}-{5 * (hi - 1)} mm of the {piece} (fingertip "
          f"{1e3 * geom.z - 6.3:.1f} mm above the board) and pinch its widest feature there: "
          f"{2e3 * r:.1f} mm across at z {','.join(map(str, at))} mm"
          + (f"  <-- ON A SLANT: the {piece} is {2e3 * prof[lo - 1]:.1f} mm across just below "
             f"the pads, so the jaw would wedge it out — raise GRASP_Z" if wedge else ""))
    print(f"  goals outside the follower's calibrated range: {' '.join(over) if over else 'none'}")
