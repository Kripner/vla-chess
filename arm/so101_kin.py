#!/usr/bin/env python3
"""SO-101 kinematics (FK, damped-LS IK, jaw gap), numpy + stdlib only, parsed at import from
assets/so101/so101.urdf (the MuJoCo scene's so101.xml is the same chain).
q = 5 arm joint radians, base->tip (shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
wrist_roll), every joint axis +Z in its own frame, zero at the range midpoint (model radians;
historically "MJCF"). arm.mapping converts to/from calibrated degrees.
fk(q) returns T_world<-gripper_link (4x4): the gripper points along local -Z (APPROACH_LOCAL),
the TCP (gripper_frame_link) is TCP_LOCAL, the jaw closes along JAW_AXIS_LOCAL, and
wrist_roll spins about the approach axis. Smoke test: python -m arm.so101_kin
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

import model as M

_URDF = Path(__file__).resolve().parent.parent / "assets" / "so101" / "so101.urdf"
ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


def _rx(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _rpy(r, p, y):
    return _rz(y) @ _ry(p) @ _rx(r)      # URDF fixed-axis roll-pitch-yaw


def _T(R, t):
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, t
    return T


def _parse_urdf(path=_URDF):
    """Parse the SO-101 URDF into {joint name -> dict(parent, child, xyz, R, axis, range)}."""
    root = ET.parse(path).getroot()
    joints = {}
    for j in root.findall("joint"):
        o = j.find("origin")
        xyz = np.array([float(v) for v in (o.get("xyz") or "0 0 0").split()]) if o is not None else np.zeros(3)
        rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()] if o is not None else [0, 0, 0]
        ax = j.find("axis")
        axis = np.array([float(v) for v in ax.get("xyz").split()]) if ax is not None else np.zeros(3)
        lim = j.find("limit")
        rng = (float(lim.get("lower")), float(lim.get("upper"))) if lim is not None else None
        joints[j.get("name")] = dict(parent=j.find("parent").get("link"), child=j.find("child").get("link"),
                                     xyz=xyz, R=_rpy(*rpy), axis=axis, type=j.get("type"), range=rng)
    return joints


_JOINTS = _parse_urdf()
# arm chain, base->tip: each entry is (fixed offset, fixed rotation) applied before Rz(q)
CHAIN = [(_JOINTS[j]["xyz"], _JOINTS[j]["R"]) for j in ARM_JOINTS]
Q_LO = np.array([_JOINTS[j]["range"][0] for j in ARM_JOINTS])
Q_HI = np.array([_JOINTS[j]["range"][1] for j in ARM_JOINTS])
GRIP_LO, GRIP_HI = _JOINTS["gripper"]["range"]

# the arm faces the board (+Y), raised by model.ARM_RAISE (the cam-stand baseplate under the
# real arm); the sim scene and this FK share it. base_link has +X forward at q=0, so a +90 deg
# yaw points the arm at +Y.
BASE = _T(_rz(np.pi / 2), (0.0, 0.0, M.ARM_RAISE))

# TCP (gripper_frame_link) in the gripper_link frame: the grasp point between the fingers.
TCP_LOCAL = _JOINTS["gripper_frame_joint"]["xyz"].copy()   # ~[-0.0079, -0.00022, -0.0981]
APPROACH_LOCAL = np.array([0.0, 0.0, -1.0])   # gripper points along local -Z
# grasp-width axis: the fixed pad sits at X~-8mm and the moving pad swings to +X (closing
# toward -X), so "toward the fixed pad" is -X.
JAW_AXIS_LOCAL = np.array([-1.0, 0.0, 0.0])

# jaw-gap geometry (from the SO-101 finger meshes, see assets/so101/README): the moving pad is
# a point in the moving_jaw frame rotating about the gripper joint; the fixed pad is a plane.
_GJ_T = _T(_JOINTS["gripper"]["R"], _JOINTS["gripper"]["xyz"])   # gripper joint in gripper_link
_P_MOV = np.array([-0.01229, -0.07917, -0.00016])               # moving pad, moving_jaw frame
_FIX_PAD_X = -0.00790                                            # fixed pad plane, gripper_link X

# fixed-pad offset from the TCP along the jaw axis (X), used by the fixed-finger-referenced
# grasp planner (chess.real.expert); ~0 because the TCP sits at the fixed-pad plane.
FJ_PAD_X = float(_FIX_PAD_X - TCP_LOCAL[0])


_CHAIN_T = [_T(R0, off) for (off, R0) in CHAIN]   # fixed per-joint transforms (precomputed)


def fk(q, base: np.ndarray = BASE) -> np.ndarray:
    """5 joint radians -> T_world<-gripper_link (4x4). Each joint is Tf @ Rz(qi); Rz only mixes
    Tf's first two columns, so it is folded in without rebuilding matrices (IK inner loop)."""
    T = base
    for Tf, qi in zip(_CHAIN_T, q):
        c, s = np.cos(qi), np.sin(qi)
        M = Tf.copy()
        M[:, 0] = c * Tf[:, 0] + s * Tf[:, 1]
        M[:, 1] = -s * Tf[:, 0] + c * Tf[:, 1]
        T = T @ M
    return T


def tcp(q, base: np.ndarray = BASE):
    """(tcp position, approach unit vector (world), jaw closing axis (world))."""
    T = fk(q, base)
    return (T[:3, :3] @ TCP_LOCAL + T[:3, 3], T[:3, :3] @ APPROACH_LOCAL,
            T[:3, :3] @ JAW_AXIS_LOCAL)


def moving_tip(q, q_jaw: float, base: np.ndarray = BASE) -> np.ndarray:
    """World position of the moving fingertip pad at arm q and gripper angle q_jaw
    (the point the approach latch in policy/action_history.py tracks)."""
    tip = _GJ_T @ np.append(_rz(q_jaw) @ _P_MOV, 1.0)
    return (fk(q, base) @ tip)[:3]


def jaw_gap(q_jaw: float) -> float:
    """Finger-pad separation (m) at a gripper joint angle (positive = open); ~2 mm at the closed limit."""
    pad = _GJ_T @ np.append(_rz(q_jaw) @ _P_MOV, 1.0)
    return float(pad[0] - _FIX_PAD_X)


def jaw_for_gap(gap: float) -> float:
    """Gripper joint angle whose finger gap is `gap` (bisection over the joint range)."""
    lo, hi = GRIP_LO, GRIP_HI
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        if jaw_gap(mid) < gap:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# pan axis (vertical) location in the world, from FK: the point the shoulder_pan spins about,
# projected to seed IK azimuth (public: planners also gate near-base behavior on the
# horizontal distance to this axis). Computed once from the base placement.
PAN_ORIGIN = (BASE @ np.append(CHAIN[0][0], 1.0))[:3]
W_ORI = 0.1                # meters-per-radian weight of the approach-direction error


def _fk_batch(Q, base):
    """[N,5] joint radians -> [N,4,4] T_world<-gripper_link, batched (the IK Jacobian evaluates
    6 configs per iteration)."""
    N = Q.shape[0]
    T = np.repeat(base[None], N, axis=0)
    for j, Tf in enumerate(_CHAIN_T):
        c, s = np.cos(Q[:, j]), np.sin(Q[:, j])
        M = np.repeat(Tf[None], N, axis=0)
        M[:, :, 0] = c[:, None] * Tf[None, :, 0] + s[:, None] * Tf[None, :, 1]
        M[:, :, 1] = -s[:, None] * Tf[None, :, 0] + c[:, None] * Tf[None, :, 1]
        T = T @ M
    return T


def _err_batch(Q, target_pos, a_star, base, w):
    """[N,5] -> [N,6] error [pos err; w*(a x a*)]. Manual cross product: np.cross's axis
    handling dominates at 100k+ calls in the IK inner loop."""
    T = _fk_batch(Q, base)
    p = T[:, :3, :3] @ TCP_LOCAL + T[:, :3, 3]
    a = T[:, :3, :3] @ APPROACH_LOCAL
    b = a_star
    cr = np.stack([a[:, 1] * b[2] - a[:, 2] * b[1],
                   a[:, 2] * b[0] - a[:, 0] * b[2],
                   a[:, 0] * b[1] - a[:, 1] * b[0]], axis=1)
    return np.concatenate([target_pos - p, w * cr], axis=1)


def _err(q, target_pos, a_star, base, w):
    return _err_batch(np.asarray(q, float)[None], target_pos, a_star, base, w)[0]


def _solve(q, target_pos, a_star, base, iters, tol, w, pos_only=False, free=None):
    """Levenberg-Marquardt on [pos err; w*(a x a*)] with a vectorized numerical Jacobian.
    `pos_only` exits once the position converges (approach residual ignored); `free` is a [5]
    0/1 mask zeroing the step of held joints so the rest solve around them (`lock_roll` in ik)."""
    lam = 1e-3
    e = _err(q, target_pos, a_star, base, w)
    h = 1e-5
    P = h * np.eye(5)
    for _ in range(iters):
        if np.linalg.norm(e[:3]) < tol and (pos_only or np.linalg.norm(e[3:]) / w < 5e-3):
            break
        Ep = _err_batch(q[None] + P, target_pos, a_star, base, w)   # [5,6], one perturb per row
        J = ((Ep - e[None]) / -h).T                                 # [6,5]
        step = J.T @ np.linalg.solve(J @ J.T + lam * np.eye(6), e)
        if free is not None:
            step = step * free
        q_new = np.clip(q + step, Q_LO, Q_HI)
        e_new = _err(q_new, target_pos, a_star, base, w)
        if np.linalg.norm(e_new) < np.linalg.norm(e):
            q, e, lam = q_new, e_new, max(lam / 2, 1e-5)
        else:
            lam = min(lam * 4, 1e2)
            if lam >= 1e1 and np.linalg.norm(e[:3]) > 5e-3:
                break          # LM stalled far from the target: unreachable seed, stop wasting iters
    return q, e


def ik(target_pos, approach=(0.0, 0.0, -1.0), q0=None, base: np.ndarray = BASE,
       iters: int = 120, tol: float = 1e-4, w_ori: float = W_ORI, pos_only: bool = False,
       lock_roll: float | None = None, ae_accept: float = 5e-3):
    """Multi-seed damped-LS IK: TCP at `target_pos`, gripper along `approach`. Returns
    (q, pos_err_m, approach_err_rad); check pos_err before trusting q. `lock_roll` holds
    wrist_roll at that angle and solves the other four joints around it: roll is a null
    direction for the approach but not for position (the TCP sits 7.9 mm off the roll axis).
    `ae_accept` (rad-ish, sin scale) stops the SEED hunt once a solution is good enough for
    the caller: at stretched targets the approach error plateaus a few degrees up whatever
    the seed, and grinding the remaining seeds for the default 0.3 deg quadruples the cost
    of every solve in a rank-1 grasp column (chess/real/phases passes its own acceptance).
    Each seed's LM still polishes to its own convergence — only the hunt stops early."""
    target_pos = np.asarray(target_pos, float)
    a_star = np.asarray(approach, float) / np.linalg.norm(approach)
    # pan is a pure vertical rotation: seed it from the target azimuth (solved pan == azimuth)
    pan = float(np.clip(np.arctan2(target_pos[0] - PAN_ORIGIN[0],
                                   target_pos[1] - PAN_ORIGIN[1]), Q_LO[0], Q_HI[0]))
    seeds = [np.asarray(q0, float)] if q0 is not None else []
    # (lift, elbow, wrist_flex) postures that solve top-down board grasps near->far; seeding
    # close converges in a few iters and the LM stall-exit bails fast on unreachable targets
    for lift, elbow, wf in ((-0.96, 1.07, 1.45), (-0.34, 0.67, 1.24),
                            (0.17, 0.08, 1.32), (0.81, -0.92, 1.66)):
        seeds.append(np.array([pan, lift, elbow, wf, 0.0]))
    free = None
    if lock_roll is not None:
        free = np.array([1.0, 1.0, 1.0, 1.0, 0.0])
        seeds = [np.concatenate([np.asarray(s, float)[:4], [lock_roll]]) for s in seeds]
    best = None
    for s in seeds:
        q, e = _solve(np.clip(s, Q_LO, Q_HI), target_pos, a_star, base, iters, tol, w_ori,
                      pos_only, free)
        pe, ae = np.linalg.norm(e[:3]), np.linalg.norm(e[3:]) / w_ori
        if best is None or pe + 0.05 * ae < best[1] + 0.05 * best[2]:
            best = (q, pe, ae)
        if pe < tol and (pos_only or ae < ae_accept):  # pos_only: first seed that reaches the point wins
            break
    return best[0], float(best[1]), float(np.arcsin(min(1.0, best[2])))


ROLL_BISECTS = 6           # halvings between the home roll and the free-solve roll (~2 deg resolution)


def reaches(q, p, tol: float, base: np.ndarray = BASE) -> bool:
    """True if the TCP at q is within `tol` of `p`. Roll moves the TCP (7.9 mm off the roll
    axis, up to 15.8 mm of travel), so a pose with an edited q[4] must be re-checked."""
    return bool(np.linalg.norm(np.asarray(tcp(q, base)[0], float)
                               - np.asarray(p, float)) <= tol)


def pin_roll(q, p, tol: float, home_roll: float, *, w_ori: float = 0.02, iters: int = 40,
             base: np.ndarray = BASE):
    """Wrist roll at `home_roll` if that still reaches `p`, else the smallest roll away from
    home that does, found by re-solving with the roll locked (never by overwriting q[4]) and
    bisecting between home and the free solve `q`. Returns the pinned pose, or None if nothing
    reaches. Home roll keeps the wrist camera's view (the board crop) as trained."""
    q = np.asarray(q, float)
    home = np.concatenate([q[:4], [home_roll]])
    if reaches(home, p, tol, base):
        return home                          # the free solve was already at home
    cand, pe, _ = ik(p, approach=tcp(q, base)[1], q0=q, w_ori=w_ori, iters=iters,
                     base=base, lock_roll=home_roll)
    if pe <= tol and reaches(cand, p, tol, base):
        return cand
    if not reaches(q, p, tol, base):         # the free solve missed too: nothing to bracket
        return None
    appr = tcp(q, base)[1]
    lo, hi, best = 0.0, float(q[4]) - home_roll, q
    for _ in range(ROLL_BISECTS):
        mid = 0.5 * (lo + hi)
        cand, pe, _ = ik(p, approach=appr, q0=q, w_ori=w_ori, iters=iters, base=base,
                         lock_roll=home_roll + mid)
        if pe <= tol and reaches(cand, p, tol, base):
            best, hi = cand, mid
        else:
            lo = mid
    return best


if __name__ == "__main__":
    # smoke: FK vs the URDF-parse oracle (in the base_link frame), jaw map, IK reachability
    q0 = np.zeros(5)
    Tgl = fk(q0, base=np.eye(4))
    print(f"FK gripper_link @ q=0 (base_link frame): pos {np.round(Tgl[:3,3],4)}")
    print("  (oracle: [0.2932 -0.0002 0.2344])")
    p, a, x = tcp(q0)
    print(f"world TCP={np.round(p,4)} approach={np.round(a,3)} jaw_axis={np.round(x,3)}")
    print(f"jaw gap: closed({GRIP_LO:.3f})={jaw_gap(GRIP_LO)*1000:.1f}mm "
          f"open({GRIP_HI:.2f})={jaw_gap(GRIP_HI)*1000:.1f}mm  q(3cm)={jaw_for_gap(0.03):.3f}rad")
    for z in (0.03, 0.06, 0.12):
        reach = [(y, ik((0.0, y, z))[1] < 2e-3 and ik((0.0, y, z))[2] < 0.03)
                 for y in np.arange(0.10, 0.40, 0.02)]
        ys = [y for y, ok in reach if ok]
        print(f"z={z:.2f}: vertical-reach y in [{min(ys):.2f},{max(ys):.2f}]" if ys else f"z={z:.2f}: none")
    for y in (0.14, 0.22, 0.30):
        for x_ in (-0.08, 0.0, 0.08):
            q, pe, ae = ik((x_, y, 0.05))
            print(f"IK ({x_:+.2f},{y:.2f},0.05): err={pe*1000:.2f}mm tilt={np.degrees(ae):.1f}deg "
                  f"{'ok' if pe < 2e-3 and ae < 0.03 else 'FAR'}")
