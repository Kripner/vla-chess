#!/usr/bin/env python3
"""Chess-move task logic: board geometry, episode sampling, scoring.

Sim-free (numpy + stdlib); shared by the sim scene (chess.sim), the scripted
expert (chess.real.expert) and the real deployment (chess.real). World frame: arm base
at the origin facing +Y, Z up. The board is the real one (24.75 x 24.75 cm, 8x8 cells of
30.9 mm) flat on the table in front of the arm. Cells are (row, col): row 0 nearest the
arm, col 0 on the arm's left (-X). Smokes: `venv/bin/python -m chess.task`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ASSETS = Path(__file__).resolve().parents[1] / "assets" / "chess"

# board geometry (m); rows span y = 0.110..0.320, every cell graspable with <= 10 deg approach tilt
N = 8
CELL = 0.2475 / N                        # 30.9 mm cells, measured on the real board
BOARD_SIZE = N * CELL                    # 0.2475
# centre: symmetric in X; near slab edge 1.5 cm in front of the cam-stand baseplate (y=0.0756)
BOARD_CENTER = (0.0, 0.238)
BOARD_THICK = 0.0235                     # slab thickness, measured
FLOOR_TOP = 0.0                          # the board lies on the table at world z=0
BOARD_TOP = FLOOR_TOP + BOARD_THICK      # z of the playing surface (0.0235)
# Intra-cell placement bound (mm): the king base (~20.6 mm) overhangs the cell beyond ~8 mm.
# The one definition — collection sampling and the hard-eval schedules both draw within it.
OFFSET_MAX_MM = 8.0


# slab margin around the 8x8: a 3 mm light strip, then a 20.5 mm walnut band with the coordinates
BORDER_LIGHT = 0.003
BORDER_DARK = 0.0205
BORDER_W = BORDER_LIGHT + BORDER_DARK

# Board-frame contract: the robot plays Black. Row 0 (nearest the robot) is rank 8, row 7 is
# rank 1 (White sits on the far side). Files a..h run from White's left, which is +x (the
# robot's right): col 7 -> file a, col 0 -> file h.


def cell_name(cell) -> str:
    """Chess algebraic name of (row, col), e.g. (0, 0) -> 'h8'."""
    r, c = cell
    return f"{chr(ord('a') + (N - 1 - c))}{N - r}"


def cell_from_name(name: str) -> tuple[int, int]:
    """Inverse of cell_name: 'e4' -> (row, col)."""
    return N - int(name[1:]), N - 1 - (ord(name[0].lower()) - ord("a"))


def move_task(source, target, piece: str = "king") -> str:
    """The language instruction for a whole move."""
    return f"Move the {piece} from {cell_name(source)} to {cell_name(target)}"


# Prompts for the learned pick/place skills, used by collection, training and deployment.
# The scripted IK phases already put the arm above the square, so the square is auxiliary
# context; `piece` names what to grasp and is real signal once several pieces stand on the board.
def pick_task(cell, piece: str = "king") -> str:
    return f"Pick up the {piece} from {cell_name(cell)}"


def place_task(cell, piece: str = "king") -> str:
    return f"Put the {piece} down on {cell_name(cell)}"


# The remaining sub-task prompts of an end-to-end move. Defined here — and only here — so the
# writer (mark rows, via chess.real.segments) and everything that reads those rows use
# byte-identical strings.
def approach_task(cell, piece: str = "king") -> str:
    return f"Reach above the {piece} on {cell_name(cell)}"


def transport_task(cell, piece: str = "king") -> str:
    return f"Carry the {piece} above {cell_name(cell)}"


def home_task() -> str:
    return "Return to the ready position"


# Hand-off geometry: where phases A and C deliver the arm and where the learned skills B and
# D start. Owned here because both planners need it (chess.real.phases imports the expert).
# Fingertip hover height ramps linearly with rank: far ranks need margin over standing
# pieces, near ranks cannot always make it. Every rung clears a king's crown (63 mm) by >= 32 mm.
HOVER_NEAR_CM = 9.5      # row 0 = rank 8, nearest the base: the lowest hover
HOVER_FAR_CM = 12.0      # row 7 = rank 1, farthest: the highest
TRANSIT_MARGIN = 0.010   # m: carried-piece bottom clearance over the tallest standing piece in phase C


def hover_fingertip_z(cell) -> float:
    """Fingertip height (m) above the board surface for the hand-off hover at `cell`,
    ramped linearly by row (0 = near = low, 7 = far = high)."""
    frac = cell[0] / (N - 1)
    return (HOVER_NEAR_CM + frac * (HOVER_FAR_CM - HOVER_NEAR_CM)) / 100.0


_INFO = None


def piece_info() -> dict:
    """assets/chess/pieces.json (heights, base radii, width profiles)."""
    global _INFO
    if _INFO is None:
        _INFO = json.loads((ASSETS / "pieces.json").read_text())
    return _INFO


# Grasp (TCP) height above the piece base, m: the one hand-set number of a grasp (gaps and pad
# offsets derive from the piece's radius profile, expert.grasp_geometry). The flat 44 mm pads
# (fingertip 6.3 mm below the TCP to 33 mm above) close on the widest feature in that span.
# king: 0.035 = the measured plateau's cliff-safe edge (a sweep of 200 easy trials
# per height, with the regenerated jaw pads + the touch-engage/eased-motion expert): strict
# 11/120/139/158/158/156/46 at 29/31/33/35/37/39/41 mm. The cliffs are physical — at 41
# the fingertip rises past the 18 mm bulge (z~30) and the pinch loses its widest feature
# (no_grasps), below ~31 the single low contact ring lets the piece pivot and topple. A
# higher pinch also shortens the pendulum below the grip: off_center fell 56 -> 38 per 200
# going 33 -> 35. At k=1000, 37 grew a no_grasp tail (15/1000: droop at stretched poses
# pushes the fingertip past the bulge — the 41 cliff encroaching), while the sweep's 35
# had the same strict with ZERO no_grasps: take the plateau edge AWAY from the cliff.
# NOTE the optimum is CONFIG-DEPENDENT: an earlier sweep (old pads, stop-go motion)
# measured 33 as best — re-sweep after any pad-mesh or motion change.
GRASP_Z ={"king": 0.035, "queen": 0.048, "rook": 0.022,
           "bishop": 0.035, "knight": 0.038, "pawn": 0.024}


def grasp_z(typ: str) -> float:
    """Grasp (TCP) height (m) above the piece base. Only the king's profile is scanned; the
    other five are a scaled CC0 set, so their picks rely on closing on feedback and retrying lower."""
    return GRASP_Z[typ]


# Folded home pose: episodes start and end here; success needs the arm back within HOME_TOL.
# The pose itself is model.HOME_TICKS; its radian form HOME_Q and HOME_GAP live in arm.mapping.
HOME_TOL = 0.25          # rad, per joint


HALF = (N - 1) / 2.0


def cell_center(row: int, col: int) -> tuple[float, float]:
    cx, cy = BOARD_CENTER
    return (cx + (col - HALF) * CELL, cy + (row - HALF) * CELL)


def cell_of(x: float, y: float) -> tuple[int, int] | None:
    """(row, col) containing the point, or None if off the board."""
    cx, cy = BOARD_CENTER
    col = int(np.floor((x - cx) / CELL + HALF + 0.5))
    row = int(np.floor((y - cy) / CELL + HALF + 0.5))
    return (row, col) if 0 <= row < N and 0 <= col < N else None


