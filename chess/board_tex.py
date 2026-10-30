#!/usr/bin/env python3
"""Real-style chessboard texture: the measured board image (checker with a1 dark, a 3 mm light
strip, the walnut margin with the printed coordinates, and the fold seam). Painted on a quad by
the sim scene. Light deps only (numpy + cv2 + chess.task), no sim/torch.

Array convention: robot's view, row 0 = far (+Y), col 0 = left (-X)."""
import numpy as np

from chess import task as T

# Measured colors, camera-sRGB uint8 as the wrist camera sees them in live frames (see
# the sim scene for the sRGB-encoding note on the sim side).
REAL_CELL_DARK = (67, 61, 69)      # (r+c) even: a1 dark, the standard board
REAL_CELL_LIGHT = (143, 145, 148)  # (r+c) odd
REAL_BORDER = (101, 96, 103)       # walnut margin band around the playing area
REAL_KING = (165, 80, 85)          # coral-orange albedo; the wrist sees its shaded sides at ~(133,64,70)
# Piece colour by side. Only the robot's (black) king is measured; the white side is a pale
# cream placeholder that only has to read as the other colour.
REAL_PIECE ={"black": REAL_KING, "white": (206, 200, 188)}
REAL_GLYPH = (164, 165, 169)       # printed coordinate labels (pale gold, blurred)
TEX_PX_MM = 8                      # board-texture resolution (glyphs ~70 px)


def board_half() -> float:
    """Half-extent (m) of the textured quad = playing area + the walnut/light margin."""
    return T.BOARD_SIZE / 2 + T.BORDER_W


def board_texture(cell_dark=None, cell_light=None, border=None, glyph_rgb=None) -> np.ndarray:
    """RGB uint8 texture of the board top (row 0 = far +Y, col 0 = left -X). Colors default to
    the measured REAL_* values; overrides are for scene color DR (env.py jitters them per reconfigure)."""
    import cv2
    cd = REAL_CELL_DARK if cell_dark is None else tuple(cell_dark)
    cl = REAL_CELL_LIGHT if cell_light is None else tuple(cell_light)
    bd = REAL_BORDER if border is None else tuple(border)
    gl = REAL_GLYPH if glyph_rgb is None else tuple(glyph_rgb)
    half = board_half()
    S = int(round(2 * half * 1000 * TEX_PX_MM))
    def pxx(x): return int(round((x + half) * 1000 * TEX_PX_MM))   # x -> col
    def pxy(y): return int(round((half - y) * 1000 * TEX_PX_MM))   # y -> row

    tex = np.zeros((S, S, 3), np.uint8)
    tex[:] = bd
    hb = T.BOARD_SIZE / 2
    a, b = pxx(-hb - T.BORDER_LIGHT), pxx(hb + T.BORDER_LIGHT)
    tex[S - b:S - a, a:b] = cl                      # the 3 mm light frame
    for r in range(T.N):
        for c in range(T.N):
            x0, x1 = pxx(-hb + c * T.CELL), pxx(-hb + (c + 1) * T.CELL)
            y0, y1 = pxy(-hb + (r + 1) * T.CELL), pxy(-hb + r * T.CELL)
            tex[y0:y1, x0:x1] = cd if (r + c) % 2 == 0 else cl

    def glyph(ch, x_m, y_m, rot180):
        f, th, sz = 2.6, 5, 110
        patch = np.zeros((sz, sz, 3), np.uint8)
        (w, h), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, f, th)
        cv2.putText(patch, ch, ((sz - w) // 2, (sz + h) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, f, gl, th, cv2.LINE_AA)
        if rot180:
            patch = cv2.rotate(patch, cv2.ROTATE_180)
        r0, c0 = pxy(y_m) - sz // 2, pxx(x_m) - sz // 2
        roi = tex[r0:r0 + sz, c0:c0 + sz]
        m = patch.max(axis=2) > 0
        roi[m] = patch[m]

    y_lab = hb + T.BORDER_LIGHT + T.BORDER_DARK / 2
    for c in range(T.N):                            # file letters over each col
        x = -hb + (c + 0.5) * T.CELL
        ch = chr(ord("A") + (T.N - 1 - c))          # a-file = col 7
        glyph(ch, x, +y_lab, rot180=True)           # far edge: upright for White
        glyph(ch, x, -y_lab, rot180=False)          # near edge: upright for the robot
    for r in range(T.N):                            # rank digits beside each row
        y = -hb + (r + 0.5) * T.CELL
        glyph(str(T.N - r), -y_lab, y, rot180=True)
        glyph(str(T.N - r), +y_lab, y, rot180=True)
    seam = int(round(0.0006 * 1000 * TEX_PX_MM))    # ~1.2 mm fold seam at y=0
    tex[pxy(0) - seam:pxy(0) + seam, :] = (tex[pxy(0) - seam:pxy(0) + seam, :] * 0.8).astype(np.uint8)
    return tex


def board_png() -> bytes:
    """Board texture as PNG bytes (the scene's material file)."""
    import cv2
    ok, buf = cv2.imencode(".png", board_texture()[..., ::-1])   # RGB -> BGR for cv2
    return buf.tobytes()
