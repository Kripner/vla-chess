#!/usr/bin/env python3
"""Scripted lift-and-place expert for the chess task: IK waypoints on a fixed-duration
schedule (grasp at the grip band, lift, carry above the target cell, lower, release, fold
home), emitted as the 6 absolute calibrated-degree targets of the policy contract.

Planning is pure kinematics (arm.so101_kin) plus the real<->model calibration
(arm.mapping.SO101Calib); the env executes the actions. The wrist roll is pinned at its
home value for the whole episode. `python -m chess.real.expert` prints the planning yield
on real game positions (needs the games file, not shipped) and runs the batched-schedule parity check.
"""
from __future__ import annotations

from typing import NamedTuple

import numpy as np

from chess import task as T
from arm import so101_kin as K
from arm.mapping import HOME_Q

FINGERTIP = 0.0063             # fixed fingertip this far below the TCP (m, from the mesh)
PAD_TOP = 0.033                 # pad top this far above the TCP (pads are 44 mm tall)
LIFT = 0.035                    # carry-height floor: piece base at least this far above the
                                # board; `_carry_lift` raises it over the tallest standing man
OPEN_MARGIN = 0.014             # opening beyond the piece's widest passed feature
GRIP_SQUEEZE = 0.003            # m commanded past pad contact; contact is at e_grasp + grip_radius
                                # (the fixed pad parks PAD_CLEAR outside the piece, so the pads do
                                # not straddle it symmetrically). Sim twin of CLOSE_PRELOAD_WIRE.
PAD_CLEAR = 0.0015              # fixed-pad clearance at the grasp pose
HOME_ROLL = HOME_Q[4]         # the wrist roll stays here for the whole episode

FARROW_TILT_DEG = 12.0          # max forward approach lean at the far rank (strictly vertical
                                # cannot reach row 7); ramps from 0 at row 5
_Y_FLAT = T.cell_center(5, 0)[1]   # at/below this board-Y the approach stays strictly vertical
_Y_FAR = T.cell_center(7, 0)[1]    # the far rank: full FARROW_TILT_DEG lean


def _approach_for(y: float) -> np.ndarray:
    """Unit approach vector at board-Y `y`: straight down up to _Y_FLAT, leaning toward +Y up
    to FARROW_TILT_DEG at _Y_FAR. The TCP still descends vertically at a constant tilt."""
    frac = float(np.clip((y - _Y_FLAT) / (_Y_FAR - _Y_FLAT), 0.0, 1.0))
    th = np.radians(FARROW_TILT_DEG * frac)
    return np.array([0.0, np.sin(th), -np.cos(th)])


def approach_ladder(y: float, board_y: float = T.BOARD_CENTER[1], lean: float = 0.0):
    """Approach directions at base-frame `y`, smallest deviation first: the `_approach_for` lean
    (turned by `lean` rad — the sim expert's human style pitches the grasp a little per
    episode), then +-10..50 deg about it (backward sign first on the near half, where reaching
    down by the base fails forward; forward first on the far half). Shared with
    chess.real.phases, whose frame's board centre is `board_y`."""
    near = y < board_y
    base = _approach_for(y)
    th0 = np.arctan2(base[1], -base[2]) + lean
    for mag in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0):
        for sgn in ((1.0,) if mag == 0.0 else ((-1.0, 1.0) if near else (1.0, -1.0))):
            th = th0 + np.radians(sgn * mag)
            yield np.array([0.0, np.sin(th), -np.cos(th)])


class GraspGeom(NamedTuple):
    """One piece type's grasp geometry, metres; see `grasp_geometry`."""
    z: float           # grasp height of the TCP above the piece's base (= above the board surface)
    open_gap: float    # finger gap the descent passes the piece at
    grip_gap: float    # finger gap that grips it (light preload included)
    e_desc: float      # TCP offset along the jaw axis (fixed-pad side) for the descent
    e_grasp: float     # ... and for the engaged grasp


def grasp_geometry(piece: str, margin: float = OPEN_MARGIN) -> GraspGeom:
    """Where the jaw meets `piece` and how wide it opens/closes, metres: z = TCP height above
    the piece base, open_gap/grip_gap = finger gaps, e_desc/e_grasp = TCP offsets along the jaw
    axis toward the fixed pad for the descent / the engaged grasp. Shared with chess.real.phases."""
    info = T.piece_info()[piece]
    prof = info["radius_by_5mm_slice"]
    gz = T.grasp_z(piece)
    lo = max(0, int((gz - FINGERTIP) / 0.005))
    hi = max(lo + 1, min(len(prof), int((gz + PAD_TOP) / 0.005) + 1))
    r_grip = float(info.get("grip_radius", max(prof[lo:hi])))   # half-width along the closing axis
    r_pass = max(prof[lo:])                    # widest feature the descent passes (max radius, any yaw)
    open_gap = min(0.062, 2 * r_pass + margin)
    # fixed-finger discipline: only one finger closes, so descend with the fixed pad clear of the
    # piece, engage laterally to PAD_CLEAR of the grip band, then close
    e_grasp = r_grip + PAD_CLEAR - K.FJ_PAD_X
    return GraspGeom(z=gz,
                     open_gap=open_gap,
                     grip_gap=max(0.0104, e_grasp + r_grip - GRIP_SQUEEZE),
                     # midline descent: (open_gap - 2*r_pass)/2 of room either side is the grasp's
                     # whole position tolerance (= `margin`; the real arm passes a bigger one)
                     e_desc=open_gap / 2 - K.FJ_PAD_X,
                     e_grasp=e_grasp)


def cap_to_cell(g: GraspGeom) -> GraspGeom:
    """Cap the open gap at one cell (the descend offset follows): wider already sweeps the
    neighbouring square. Only the knight needs it: `radius_by_5mm_slice` is the max radius per
    slice, so its muzzle opens the jaw to about twice its width across the head."""
    if g.open_gap <= T.CELL:
        return g
    return g._replace(open_gap=T.CELL, e_desc=T.CELL / 2 - K.FJ_PAD_X)
