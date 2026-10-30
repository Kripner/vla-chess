#!/usr/bin/env python3
"""MuJoCo scene for the chess task: the SO-101 (assets/so101/so101.xml — same chain as the
URDF, with the fitted STS3215 servo dynamics) + the measured board and king. Everything is
placed from measurement: the arm base at so101_kin.BASE (Rz90, ARM_RAISE), the board at
chess/real/board_pose.json, geometry from chess.task, cameras from model.CAMERAS. Plain
mujoco.MjModel; the collector and the evaluator step one MjData per world.

Smoke (FK parity + camera renders):  python -m chess.sim.scene --out <dir>
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

import model as M
from arm import mapping as J
from arm import so101_kin as K
from chess import task as T

REPO = Path(__file__).resolve().parents[2]
SO101_XML = REPO / "assets" / "so101" / "so101.xml"
KING_MESH = REPO / "assets" / "chess" / "king_lp.obj"   # visual; collision = its coacd pieces
PIECES = json.loads((REPO / "assets" / "chess" / "pieces.json").read_text())
# The SIM board pose is NOMINAL: square to the arm base, at the chess.task geometry. The real
# robot's chess/real/board_pose.json is a touch calibration whose yaw (5.6 deg) and tilt absorb
# the REAL arm's kinematic error — the sim arm is the exact URDF model and has none, so applying
# it here would rotate and tilt the board for no physical reason (and it left the sim's flat
# board up to 7.8 mm off the pose the scripted planner aims at). Pass a measured pose explicitly
# only for a deliberate sim2real experiment.
BOARD_POSE_JSON = Path(__file__).resolve().parent / "board_pose.json"

CAM_W, CAM_H = 640, 360          # default render size (16:9 like the real cams)
TIMESTEP = 0.002                 # physics dt; controllers write ctrl at the capture rate

KING_BODY = "king"
WRIST_CAM, OVERHEAD_CAM = "wrist", "overhead"


def board_pose_xy_yaw(path: Path = BOARD_POSE_JSON) -> tuple[float, float, float]:
    """The scene's board pose as (x, y, yaw_deg); nominal centre when the file is absent."""
    if not Path(path).exists():
        return (float(T.BOARD_CENTER[0]), float(T.BOARD_CENTER[1]), 0.0)
    d = json.loads(Path(path).read_text())
    c = d.get("center_m") or [T.BOARD_CENTER[0], T.BOARD_CENTER[1], 0.0]
    return (float(c[0]), float(c[1]), float(d.get("yaw_deg", 0.0)))


def _quat_wxyz_from_z_yaw(yaw_deg: float) -> list[float]:
    h = np.deg2rad(yaw_deg) / 2.0
    return [float(np.cos(h)), 0.0, 0.0, float(np.sin(h))]


def _camera_quat(t, look_at, up) -> list[float]:
    """MuJoCo camera frame (looks along -Z, +Y up) from an eye/look_at/up rig; wxyz quat."""
    t, look_at, up = (np.asarray(v, float) for v in (t, look_at, up))
    z = t - look_at
    z /= np.linalg.norm(z)                       # camera backward
    x = np.cross(up, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    R = np.stack([x, y, z], axis=1)
    w = np.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w < 1e-9:                                 # not reachable for our rigs
        raise ValueError("degenerate camera quaternion")
    return [float(w), float((R[2, 1] - R[1, 2]) / (4 * w)),
            float((R[0, 2] - R[2, 0]) / (4 * w)), float((R[1, 0] - R[0, 1]) / (4 * w))]


def _fovy_deg(w: int, h: int) -> float:
    """Vertical FOV matching the rectified pinhole HFOV (model.CAM_HFOV) at aspect w:h."""
    return float(np.rad2deg(2.0 * np.arctan(np.tan(np.deg2rad(M.CAM_HFOV) / 2.0) * h / w)))


def _board_texture_png() -> str:
    """The measured board texture (chess.board_tex) as a PNG file for the material."""
    from chess.board_tex import board_png
    f = Path(tempfile.gettempdir()) / "chess_board_tex.png"
    # Written by every process that builds a scene (a collector's or evaluator's 32-64 pool
    # workers at once): write beside and rename, so a compile in another process never
    # reads a truncated file ("empty PNG file" with 96 workers on one node).
    tmp = f.with_name(f"{f.stem}.{os.getpid()}.tmp")
    tmp.write_bytes(board_png())
    os.replace(tmp, f)
    return str(f)


def ready_qpos() -> np.ndarray:
    """Full arm qpos (5 arm rad + jaw rad) at the READY rest (model.READY_TICKS)."""
    q5 = J.q_mjcf_from_ticks(M.READY_TICKS)
    grip_deg = (M.READY_TICKS["gripper"] - J._CAL.mid["gripper"]) * 360.0 / J.MAX_RES
    return np.concatenate([q5, [float(J._grip_deg_to_jaw(grip_deg))]])


def build(cam_w: int = CAM_W, cam_h: int = CAM_H, board_pose: Path = BOARD_POSE_JSON):
    """-> (mujoco.MjModel, meta dict). The scene: floor, board (measured pose + texture),
    the king (free body; mass estimated from wood density, coacd collision of the scan),
    the SO-101 at so101_kin.BASE, wrist + overhead cameras (model.CAMERAS), keyframe
    'ready' (arm at READY, king at the board centre cell)."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(SO101_XML))
    spec.option.timestep = TIMESTEP
    # Grasping-grade contact solve (MuJoCo's own guidance): elliptic cone + high impratio
    # make the friction constraints as hard as the normal ones, killing numerical slip.
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10.0

    # -- foam-like jaw contacts ------------------------------------------------------- #
    # The real jaws have foam glued on, and its observable effect is simply: the piece
    # does not slip. Rather than modelling pad geometry (tried; hard to seat flush on the
    # curved tips), the foam's contact PROPERTIES live directly on the jaws' collision
    # surfaces: high sliding friction, condim 6 (a conforming patch resists twist and
    # roll), soft solref (compliance), priority 1 so these govern jaw-piece contacts.
    for body in ("moving_jaw_so101_v1", "gripper"):
        b = next(bb for bb in spec.bodies if bb.name == body)
        for gm in b.geoms:
            if gm.contype or gm.conaffinity:
                gm.friction = [2.2, 0.08, 0.002]
                gm.condim = 6
                gm.priority = 1
                gm.solref = [0.01, 1.0]

    # The real follower runs with Torque_Limit at 50% (the teleop default) and the
    # STS3215 stalls at ~1.9 Nm, so the jaw tops out near 1.0 Nm — the XML's fitted
    # 2.94 Nm is full stall and crushes the piece millimetres into the jaws.
    grip_act = next(a for a in spec.actuators if a.name == "gripper")
    grip_act.forcerange = [-1.0, 1.0]

    # -- arm joint limits: the URDF's, not the XML's ---------------------------------- #
    # The vendored so101.xml symmetrizes wrist_flex to ±95.0 deg, but the vendored URDF
    # (which the IK planner clips to, arm.so101_kin.Q_LO/Q_HI) allows +102 deg up, and the
    # real arm executes such poses (its calibrated range is ±103 cal-deg). With the XML
    # limit the sim clamps the planner's grasp staging ~7 deg short and scripted picks
    # time out. One model of the arm: align joint range + ctrlrange to the URDF.
    for i, name in enumerate(("shoulder_pan", "shoulder_lift", "elbow_flex",
                              "wrist_flex", "wrist_roll")):
        jnt = next(j for j in spec.joints if j.name == name)
        jnt.range = [float(K.Q_LO[i]), float(K.Q_HI[i])]
        act = next(a for a in spec.actuators if a.name == name)
        act.ctrlrange = [float(K.Q_LO[i]), float(K.Q_HI[i])]

    # -- arm base: the measured world frame (so101_kin.BASE = Rz90 + ARM_RAISE) ------- #
    base = next(b for b in spec.bodies if b.name == "base")
    base.pos = [0.0, 0.0, M.ARM_RAISE]
    base.quat = _quat_wxyz_from_z_yaw(90.0)

    world = spec.worldbody

    # -- floor (the table top, world z=0) --------------------------------------------- #
    spec.add_material(name="mat_floor", rgba=[0.45, 0.42, 0.38, 1.0])
    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[2, 2, 0.1],
                   material="mat_floor")

    # -- board: slab + textured top at the measured pose ------------------------------ #
    bx, by, yaw = board_pose_xy_yaw(board_pose)
    spec.add_texture(name="tex_board", type=mujoco.mjtTexture.mjTEXTURE_2D,
                     file=_board_texture_png())
    spec.add_material(name="mat_board", textures=["", "tex_board"])  # RGB slot
    spec.add_material(name="mat_slab", rgba=[0.42, 0.30, 0.20, 1.0])
    from chess.board_tex import board_half
    he = float(board_half())                        # textured quad half-extent (board + margin)
    bb = world.add_body(name="board", pos=[bx, by, T.FLOOR_TOP],
                        quat=_quat_wxyz_from_z_yaw(yaw))
    bb.add_geom(name="board_slab", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[he, he, T.BOARD_THICK / 2], pos=[0, 0, T.BOARD_THICK / 2],
                material="mat_slab",
                # a stiff anvil: pressing a grasped piece down must not sink it into the
                # wood (measured 2.4 mm with the defaults); priority 1 so these params
                # govern piece-board contacts
                priority=1, solref=[0.005, 1.0], solimp=[0.95, 0.99, 0.001, 0.5, 2.0])
    bb.add_geom(name="board_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[he, he, 0.0005], pos=[0, 0, T.BOARD_THICK + 0.0005],
                material="mat_board", contype=0, conaffinity=0)   # visual skin only

    # -- king: visual mesh + coacd collision from the SAME scan ----------------------- #
    # The scan's radial profile matches the caliper radii (pieces.json) within 0.3 mm,
    # so the convex pieces (assets/chess/collision, coacd threshold 0.02) ARE the
    # measured geometry — smooth faces instead of the old cylinder-slice terraces.
    kd = PIECES["king"]
    spec.add_mesh(name="king_mesh", file=str(KING_MESH))
    spec.add_material(name="mat_king", rgba=[0.85, 0.79, 0.62, 1.0])
    kb = world.add_body(name=KING_BODY, pos=[bx, by, T.BOARD_TOP + 0.001])
    kb.add_freejoint()
    king_parts = sorted((KING_MESH.parent / "collision").glob("king_c*.stl"))
    part_vols = []
    import trimesh
    for f in king_parts:
        part_vols.append(float(trimesh.load(f).volume))
    # pieces.json mass is an ESTIMATE (wood at ~665 kg/m3 for the scan volume), not a
    # weighing — replace it with a scale reading when one exists.
    density = float(kd["mass"]) / sum(part_vols)    # spread the mass over the pieces
    for i, f in enumerate(king_parts):
        spec.add_mesh(name=f"king_c{i}", file=str(f))
        kb.add_geom(name=f"king_c{i}", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"king_c{i}",
                    density=density, rgba=[1, 1, 1, 0], group=3)   # collision, invisible
    kb.add_geom(name="king_visual", type=mujoco.mjtGeom.mjGEOM_MESH, meshname="king_mesh",
                material="mat_king", contype=0, conaffinity=0,
                density=0.0)   # visual only — without this it adds ~10 g of phantom mass

    # -- cameras (measured rigs) ------------------------------------------------------ #
    ov = M.CAMERAS["overhead"]
    world.add_camera(name=OVERHEAD_CAM, pos=list(ov["t"]),
                     quat=_camera_quat(ov["t"], ov["look_at"], ov["up"]),
                     fovy=_fovy_deg(cam_w, cam_h))
    wr = M.CAMERAS["wrist"]
    grip = next(b for b in spec.bodies if b.name == "gripper")   # == URDF gripper_link
    grip.add_camera(name=WRIST_CAM, pos=list(wr["t"]),
                    quat=_camera_quat(wr["t"], wr["look_at"], wr["up"]),
                    fovy=_fovy_deg(cam_w, cam_h))

    # -- light ------------------------------------------------------------------------ #
    world.add_light(name="sun", pos=[0.3, 0.0, 1.2], dir=[-0.2, 0.15, -1.0],
                    diffuse=[0.9, 0.9, 0.9], specular=[0.2, 0.2, 0.2],
                    castshadow=False)

    # -- keyframe: READY, king at the board centre ------------------------------------ #
    q = ready_qpos()
    # qpos layout: resolve joint addresses from the compiled model instead of assuming order.
    mj_model = spec.compile()
    data = mujoco.MjData(mj_model)
    arm_adr = [mj_model.joint(m).qposadr[0] for m in M.MOTOR_NAMES]
    king_adr = mj_model.joint(mj_model.body(KING_BODY).jntadr[0]).qposadr[0]
    qpos0 = np.zeros(mj_model.nq)
    qpos0[king_adr:king_adr + 7] = [bx, by, T.BOARD_TOP + 0.001, 1, 0, 0, 0]
    for a, v in zip(arm_adr, q):
        qpos0[a] = v
    meta = {"arm_qposadr": arm_adr, "king_qposadr": int(king_adr),
            "arm_ctrladr": [mj_model.actuator(m).id for m in M.MOTOR_NAMES],
            "qpos_ready": qpos0, "board_pose": (bx, by, yaw),
            "cam_w": cam_w, "cam_h": cam_h}
    del data
    return mj_model, meta


def cell_world(cell, board_pose_xyyaw, dx_m: float = 0.0, dy_m: float = 0.0):
    """(row, col) + intra-cell offset (m, board frame) -> world (x, y) under the board pose."""
    bx, by, yaw = board_pose_xyyaw
    cx, cy = T.cell_center(*cell)
    r = np.deg2rad(yaw)
    lx, ly = cx - T.BOARD_CENTER[0] + dx_m, cy - T.BOARD_CENTER[1] + dy_m
    return (bx + lx * np.cos(r) - ly * np.sin(r), by + lx * np.sin(r) + ly * np.cos(r))


def main() -> int:
    import argparse

    import mujoco

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=None, help="write wrist/overhead PNGs here")
    ap.add_argument("--steps", type=int, default=500, help="settle steps for the smoke")
    ap.add_argument("--view", action="store_true",
                    help="open the interactive MuJoCo viewer (number keys toggle geom "
                         "groups: 3 = collision shapes, 0/2 = visuals)")
    a = ap.parse_args()

    mj_model, meta = build()
    data = mujoco.MjData(mj_model)
    data.qpos[:] = meta["qpos_ready"]
    mujoco.mj_forward(mj_model, data)

    if a.view:
        import mujoco.viewer
        mujoco.viewer.launch(mj_model, data)
        return 0

    # FK parity: the MJCF chain vs arm/so101_kin at READY (same base transform).
    q5 = np.array([data.qpos[a] for a in meta["arm_qposadr"][:5]])
    fk_pos = K.fk(q5)[:3, 3]
    mj_pos = np.array(data.body("gripper").xpos)
    err_mm = float(np.linalg.norm(fk_pos - mj_pos) * 1000.0)
    print(f"FK parity (gripper_link @ READY): so101_kin {np.round(fk_pos, 4)} "
          f"vs mjcf {np.round(mj_pos, 4)}  |err| = {err_mm:.2f} mm")

    for i, adr in zip(meta["arm_ctrladr"], meta["arm_qposadr"]):   # hold READY
        data.ctrl[i] = data.qpos[adr]
    for _ in range(a.steps):                        # king must rest on the board
        mujoco.mj_step(mj_model, data)
    kz = data.qpos[meta["king_qposadr"] + 2]
    print(f"king rest z after {a.steps} steps: {kz * 1000:.1f} mm (board top {T.BOARD_TOP * 1000:.1f} mm)")

    if a.out:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        try:
            import imageio.v2 as iio
            r = mujoco.Renderer(mj_model, height=meta["cam_h"], width=meta["cam_w"])
            for cam in (WRIST_CAM, OVERHEAD_CAM):
                r.update_scene(data, camera=cam)
                iio.imwrite(out / f"{cam}.png", r.render())
            print(f"renders -> {out}/wrist.png, {out}/overhead.png")
        except Exception as e:                       # headless without EGL etc.
            print(f"render skipped ({type(e).__name__}: {e})")
    ok = err_mm < 5.0
    print("SCENE SMOKE", "OK" if ok else "FK MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
