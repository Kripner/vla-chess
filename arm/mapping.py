#!/usr/bin/env python3
"""Joint-space conversions for the chess stack (numpy only): contract calibrated degrees
(policy state/actions, lerobot DEGREES) <-> model radians (range-midpoint zero; historically
"MJCF", hence the q_mjcf/MJCF_* names) <-> raw STS3215 ticks <-> so101 agent qpos.
Arm: deg <-> ticks is lerobot's DEGREES (un)normalize against arm/so101_follower_calib.json;
ticks <-> rad is rad = offset + flip*(raw-2048)*2pi/4096 with offset/flip from model.JOINTS;
model radians and sim qpos are identical (so101.urdf and the scene's so101.xml share the chain).
Gripper: calibrated travel maps linearly onto the model jaw range; jaw <-> finger gap via
so101_kin, and the agent gripper qpos is the jaw angle. Smoke test: python -m arm.mapping
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

import model as M
from arm import so101_kin as K

CALIB_PATH = Path(__file__).resolve().parent / "so101_follower_calib.json"

TICKS = 4096          # STS3215 encoder resolution
MAX_RES = TICKS - 1   # lerobot uses model_resolution - 1 = 4095 for DEGREES
TICK_CENTRE = TICKS / 2

MJCF_RANGE = {m: M.JOINTS[m]["range"] for m in M.MOTOR_NAMES}
FLIP = {m: M.JOINTS[m]["flip"] for m in M.MOTOR_NAMES}      # all +1 for the SO-101
RANGE_MODE = {m for m in M.MOTOR_NAMES if M.JOINTS[m].get("range_mapped")}   # {gripper}


class SO101Calib:
    """Per-joint calibrated degrees <-> model radians; the only per-arm input is the calibration JSON."""

    def __init__(self, calib_path: Path = CALIB_PATH):
        raw = json.loads(Path(calib_path).read_text())
        self.mid = {}       # raw-tick centre of each motor's calibrated range
        self.offset = {}    # model angle (rad) at raw=2048
        self.flip = {}
        self.deg_lim = {}   # range-mode joints: (cal_deg at range_min, at range_max)
        self.rmin, self.rmax = {}, {}   # calibrated raw-tick limits (viewer replay)
        for m in M.MOTOR_NAMES:
            c = raw[m]
            self.rmin[m], self.rmax[m] = c["range_min"], c["range_max"]
            self.mid[m] = (c["range_min"] + c["range_max"]) / 2.0
            lo, hi = MJCF_RANGE[m]
            self.offset[m] = (lo + hi) / 2.0
            self.flip[m] = FLIP.get(m, 1)
            if m in RANGE_MODE:
                self.deg_lim[m] = ((c["range_min"] - self.mid[m]) * 360.0 / MAX_RES,
                                   (c["range_max"] - self.mid[m]) * 360.0 / MAX_RES)
        self.lo = {m: MJCF_RANGE[m][0] for m in M.MOTOR_NAMES}
        self.hi = {m: MJCF_RANGE[m][1] for m in M.MOTOR_NAMES}

    def deg_to_rad(self, motor: str, cal_deg: float) -> float:
        """Calibrated degrees -> model joint radians, clamped to the model range."""
        lo, hi = self.lo[motor], self.hi[motor]
        if motor in RANGE_MODE:                      # gripper: linear range map
            d0, d1 = self.deg_lim[motor]
            frac = min(1.0, max(0.0, (cal_deg - d0) / (d1 - d0)))
            return lo + frac * (hi - lo)
        raw = cal_deg * MAX_RES / 360.0 + self.mid[motor]
        rad = self.offset[motor] + self.flip[motor] * (raw - TICK_CENTRE) * (2 * math.pi / TICKS)
        return min(hi, max(lo, rad))                 # clamp to model range

    def rad_to_deg(self, motor: str, rad: float) -> float:
        """Model joint radians -> calibrated degrees."""
        lo, hi = self.lo[motor], self.hi[motor]
        if motor in RANGE_MODE:
            d0, d1 = self.deg_lim[motor]
            frac = (rad - lo) / (hi - lo) if hi != lo else 0.0
            return d0 + frac * (d1 - d0)
        raw = TICK_CENTRE + self.flip[motor] * (rad - self.offset[motor]) * (TICKS / (2 * math.pi))
        return (raw - self.mid[motor]) * 360.0 / MAX_RES


_CAL = SO101Calib()

# model radians <-> so101 sim qpos: so101_kin parses the URDF, and the MuJoCo scene's
# assets/so101/so101.xml is the same chain (scene.py FK-parity smoke), so the remap is identity; kept as named arrays + thin wrappers so callers are unchanged.
ARM_SIGN = np.ones(5)
ARM_OFF = np.zeros(5)

# jaw angle <-> finger gap, tabulated so both directions vectorize (jaw_gap is monotonic over
# the gripper range). The agent gripper qpos is the jaw angle, so gap <-> qpos is this table.
_JAW_T = np.linspace(K.GRIP_LO, K.GRIP_HI, 257)
_GAP_T = np.array([K.jaw_gap(a) for a in _JAW_T])


def arm_mjcf_to_urdf(q_mjcf):
    """[..., 5] model arm radians -> so101 agent arm qpos (identity)."""
    return np.asarray(q_mjcf, float) * ARM_SIGN + ARM_OFF


def arm_urdf_to_mjcf(q_urdf):
    """[..., 5] so101 agent arm qpos -> model arm radians (identity)."""
    return (np.asarray(q_urdf, float) - ARM_OFF) * ARM_SIGN


def grip_qpos_for_gap(gap_m):
    """Finger gap (m) -> so101 gripper joint angle (rad)."""
    return np.interp(np.asarray(gap_m, float), _GAP_T, _JAW_T)


def gap_for_grip_qpos(qpos):
    """so101 gripper joint angle (rad) -> finger gap (m). Inverse of grip_qpos_for_gap."""
    return np.interp(np.asarray(qpos, float), _JAW_T, _GAP_T)


# contract degrees <-> agent qpos, batched. Arm: rad = _D2R_ZERO + _D2R_SCALE * deg, clamped to
# the model range. Gripper: deg -> jaw rad (linear range map) -> gap (table) -> qpos.
_ARM = M.MOTOR_NAMES[:5]
_D2R_SCALE = np.array([_CAL.flip[m] * (MAX_RES / 360.0) * (2 * np.pi / TICKS) for m in _ARM])
_D2R_ZERO = np.array([_CAL.offset[m] + _CAL.flip[m] * (_CAL.mid[m] - TICK_CENTRE)
                      * (2 * np.pi / TICKS) for m in _ARM])
_G_D0, _G_D1 = _CAL.deg_lim["gripper"]
_G_LO, _G_HI = _CAL.lo["gripper"], _CAL.hi["gripper"]


def _grip_deg_to_jaw(deg):
    frac = np.clip((np.asarray(deg, float) - _G_D0) / (_G_D1 - _G_D0), 0.0, 1.0)
    return _G_LO + frac * (_G_HI - _G_LO)


def _grip_jaw_to_deg(jaw):
    frac = (np.asarray(jaw, float) - _G_LO) / (_G_HI - _G_LO)
    return _G_D0 + frac * (_G_D1 - _G_D0)


# Gripper units. The datasets store the gripper as lerobot's RANGE_0_100 ("wire" units), the
# five arm joints as calibrated degrees. The policy sees all six in calibrated degrees: the
# trainer remaps the gripper with grip_0_100_to_deg at load, and the evaluator converts back
# with grip_deg_to_0_100 wherever it writes wire units. Both are linear reparams of the same
# calibrated tick range, so the pair is exact.
def grip_0_100_to_deg(v):
    """RANGE_0_100 gripper (host wire) -> contract cal-deg."""
    return _G_D0 + np.asarray(v, float) / 100.0 * (_G_D1 - _G_D0)


def grip_deg_to_0_100(deg):
    """contract cal-deg gripper -> RANGE_0_100 (host goal). Inverse of grip_0_100_to_deg."""
    return (np.asarray(deg, float) - _G_D0) / (_G_D1 - _G_D0) * 100.0


def urdf_from_deg(deg):
    """[..., 6] contract cal-deg -> ([..., 5] arm radians, [...] jaw joint rad), unclamped:
    kinematics consumers (the approach latch) need the true pose near the limits."""
    deg = np.asarray(deg, float)
    return arm_mjcf_to_urdf(_D2R_ZERO + _D2R_SCALE * deg[..., :5]), _grip_deg_to_jaw(deg[..., 5])


def wire_from_ticks(ticks) -> list:
    """{motor: raw tick} -> wire state [6] (5 arm cal-deg + gripper 0-100)."""
    deg = [(float(ticks[m]) - _CAL.mid[m]) * 360.0 / MAX_RES for m in M.MOTOR_NAMES]
    return [float(d) for d in deg[:5]] + [float(grip_deg_to_0_100(deg[5]))]


# The folded home pose is defined once as raw follower ticks (model.HOME_TICKS); its
# radian/gap form is derived here with the same unclamped tick->rad the FK uses.
def q_mjcf_from_ticks(ticks) -> np.ndarray:
    """Raw ticks -> 5 arm joints in model radians, unclamped (FK needs the physical angle)."""
    return np.array([_CAL.offset[m] + _CAL.flip[m] * (ticks[m] - TICK_CENTRE) * (2 * np.pi / TICKS)
                     for m in _ARM])


def _home_from_ticks():
    t = M.HOME_TICKS
    q = tuple(round(float(v), 4) for v in q_mjcf_from_ticks(t))
    grip_deg = (t["gripper"] - _CAL.mid["gripper"]) * 360.0 / MAX_RES
    gap = round(float(gap_for_grip_qpos(_grip_deg_to_jaw(grip_deg))), 4)
    return q, gap


HOME_Q, HOME_GAP = _home_from_ticks()               # 5 arm model rad ; finger gap (m)


def deg_from_qpos(qpos):
    """[..., 6+] so101 qpos (first 6 dofs) -> [..., 6] contract calibrated degrees (`observation.state`)."""
    qpos = np.asarray(qpos, float)
    q_mjcf = arm_urdf_to_mjcf(qpos[..., :5])
    deg_arm = (q_mjcf - _D2R_ZERO) / _D2R_SCALE
    jaw = np.interp(gap_for_grip_qpos(qpos[..., 5]), _GAP_T, _JAW_T)
    return np.concatenate([deg_arm, _grip_jaw_to_deg(jaw)[..., None]],
                          axis=-1).astype(np.float32)


def wire_to_qpos(wire) -> np.ndarray:
    """Wire state [6] (5 arm cal-deg + gripper 0-100) -> sim arm qpos [6] (5 rad + jaw rad)."""
    wire = np.asarray(wire, float)
    deg6 = np.concatenate([wire[:5], [float(grip_0_100_to_deg(wire[5]))]])
    q5, jaw = urdf_from_deg(deg6)
    return np.concatenate([q5, [float(jaw)]])


def qpos_to_wire(qpos6) -> list[float]:
    """Sim arm qpos [6] -> wire state [6] (5 arm cal-deg + gripper 0-100)."""
    deg6 = deg_from_qpos(np.asarray(qpos6, float))
    return [float(v) for v in deg6[:5]] + [float(grip_deg_to_0_100(deg6[5]))]


if __name__ == "__main__":
    # smoke: round-trips + the home pose in every convention
    calib = SO101Calib()
    print(f"{'motor':14} deg=0 -> model rad   roundtrip max err")
    for m in M.MOTOR_NAMES:
        errs = [abs(d - calib.rad_to_deg(m, calib.deg_to_rad(m, d))) for d in (-30, -10, 0, 10, 30)]
        print(f"{m:14} {calib.deg_to_rad(m, 0.0):9.4f}          {max(errs):.2e}")
    home_deg = np.array([calib.rad_to_deg(m, r) for m, r in zip(_ARM, HOME_Q)]
                        + [calib.rad_to_deg("gripper", K.jaw_for_gap(HOME_GAP))])
    q5, jaw = urdf_from_deg(home_deg)
    assert np.allclose(q5, HOME_Q, atol=1e-6)
    back = deg_from_qpos(np.concatenate([q5, [float(grip_qpos_for_gap(K.jaw_gap(jaw)))]]))
    assert np.max(np.abs(back - home_deg)) < 0.5, back - home_deg
    wire = qpos_to_wire(wire_to_qpos(qpos_to_wire(np.concatenate([q5, [jaw]]))))
    print("HOME deg:", np.round(home_deg, 1), " wire:", np.round(wire, 1))
    print("mapping smoke OK")
