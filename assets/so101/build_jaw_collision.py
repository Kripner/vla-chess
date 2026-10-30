#!/usr/bin/env python3
"""Rebuild the coacd collision decomposition of the two SO-101 GRASPING parts and rewire
so101.xml to it.

MuJoCo collides a mesh geom by its CONVEX HULL, and both jaws are hooked/concave: a single
hull is 2.33x (moving jaw) / 2.58x (fixed finger) the true volume and fills the gripper mouth,
so the arm grasps the piece through phantom solid. Each part therefore collides as N convex
coacd pieces. Only these two parts are decomposed; every other link keeps its single hull,
which is harmless because nothing else touches the piece.

Writes assets/collision/<part>_c<i>.stl, rewrites the <mesh>/<geom class="collision"> lines for
these two parts in so101.xml (each geom keeps the part's original pos/quat), and re-exports the
multi-object *_col.obj that so101.urdf's <collision> blocks point at, so both descriptions
carry the SAME geometry. Prints the fidelity report; commit the outputs.

NOT REPRODUCIBLE: coacd's MCTS is multithreaded and ignores `seed`, so two runs of this script
give geometrically equivalent but different pieces -- enough to flip a marginal scripted grasp
between success and off_center. The COMMITTED STLs are the artifact of record; re-running this
changes the sim. Do not re-run it to "refresh" the assets, only to change the settings.

  venv/bin/python assets/so101/build_jaw_collision.py
"""
from __future__ import annotations

import re
from pathlib import Path

import coacd
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
A = HERE / "assets"
COL = A / "collision"
XML = HERE / "so101.xml"

PARTS = ("moving_jaw_so101_v1", "wrist_roll_follower_so101_v1")

# coacd settings, chosen by measuring fidelity against piece count (see README). The knob that
# matters is max_convex_hull: threshold and resolution barely move the error (5x threshold and
# 4x resolution bought 1.212x -> 1.188x volume) because coacd's merge pass fuses the fine pieces
# back together regardless. Lifting the cap 24 -> 64 halves the excess volume. merge=False would
# reach 1.011x but ignores the cap entirely and emits ~350 pieces per part, too many to collide.
COACD = dict(threshold=0.01, resolution=4000, preprocess_resolution=100,
             merge=True, max_convex_hull=64, seed=0)


def decompose(name: str):
    mesh = trimesh.load(A / f"{name}.stl", force="mesh")
    parts = coacd.run_coacd(coacd.Mesh(mesh.vertices, mesh.faces), **COACD)
    return mesh, [trimesh.Trimesh(v, f) for v, f in parts]


def fidelity(pieces, mesh, n: int = 30000):
    """(over-fill factor, mm the collider surface protrudes past the true surface).

    Samples the collider's own surface and signs the distance with the true mesh's face normal
    at the closest point -- cheap, and it measures the quantity that matters for a grasp: how
    far the piece is pushed away from where the real jaw would touch it."""
    hulls = [p.convex_hull for p in pieces]
    areas = np.array([h.area for h in hulls])
    counts = np.maximum(1, (n * areas / areas.sum()).astype(int))
    pts = np.vstack([trimesh.sample.sample_surface(h, c)[0] for h, c in zip(hulls, counts)])
    pq = trimesh.proximity.ProximityQuery(mesh)
    out = []
    for i in range(0, len(pts), 5000):
        chunk = pts[i:i + 5000]
        close, dist, fid = pq.on_surface(chunk)
        sign = np.einsum("ij,ij->i", chunk - close, mesh.face_normals[fid])
        out.append(dist[sign > 0])
    return sum(h.volume for h in hulls) / mesh.volume, np.concatenate(out) * 1000.0


def rewrite_xml(counts: dict[str, int]) -> None:
    """Point so101.xml at the new piece counts, keeping each part's pos/quat."""
    src = XML.read_text()

    for name, n in counts.items():
        # <mesh file="collision/<name>_c<i>.stl"/> block
        decl = re.compile(rf'( *)<mesh file="collision/{name}_c\d+\.stl"/>\n+', re.M)
        hits = list(decl.finditer(src))
        indent = hits[0].group(1)
        block = "".join(f'{indent}<mesh file="collision/{name}_c{i}.stl"/>\n' for i in range(n))
        src = src[:hits[0].start()] + block + src[hits[-1].end():]

        # <geom type="mesh" class="collision" pos=... quat=... mesh="<name>_c<i>"/> block
        geom = re.compile(
            rf'( *)<geom type="mesh" class="collision" (pos="[^"]*" quat="[^"]*") '
            rf'mesh="{name}_c\d+"/>\n+', re.M)
        hits = list(geom.finditer(src))
        indent, place = hits[0].group(1), hits[0].group(2)
        block = "".join(
            f'{indent}<geom type="mesh" class="collision" {place} mesh="{name}_c{i}"/>\n'
            for i in range(n))
        src = src[:hits[0].start()] + block + src[hits[-1].end():]

    XML.write_text(src)


def main() -> None:
    COL.mkdir(exist_ok=True)
    counts = {}
    for name in PARTS:
        mesh, pieces = decompose(name)
        for stale in COL.glob(f"{name}_c*.stl"):
            stale.unlink()
        for i, p in enumerate(pieces):
            p.export(COL / f"{name}_c{i}.stl")
        # the URDF's <collision> wants the same geometry as one multi-object OBJ
        sc = trimesh.Scene()
        for i, p in enumerate(pieces):
            sc.add_geometry(p, geom_name=f"{name}_p{i}")
        sc.export(A / f"{name}_col.obj")

        counts[name] = len(pieces)
        vol, out = fidelity(pieces, mesh)
        print(f"{name}: {len(pieces)} pieces, {vol:.3f}x true volume "
              f"(single hull {mesh.convex_hull.volume / mesh.volume:.2f}x)")
        print(f"   protrudes past the true surface: mean {out.mean():.2f} mm, "
              f"p95 {np.percentile(out, 95):.2f} mm, max {out.max():.2f} mm")

    rewrite_xml(counts)
    print(f"rewired {XML} -> " + ", ".join(f"{k} x{v}" for k, v in counts.items()))


if __name__ == "__main__":
    main()
