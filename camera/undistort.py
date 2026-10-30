#!/usr/bin/env python3
"""Undistort real camera frames to the sim's pinhole (square pixels, hfov model.CAM_HFOV, centred
principal point). Applied identically at training and deploy time; sim frames are already that
pinhole and must not pass through here. Per-camera intrinsics: camera/camera_intrinsics_<name>.json.
    python -m camera.undistort <in.png> /tmp/rect.png --camera wrist
"""
import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
from model import CAM_HFOV

_HERE = os.path.dirname(os.path.abspath(__file__))


def intrinsics_path(camera: str) -> str:
    """Runtime intrinsics home for a camera: camera/camera_intrinsics_<camera>.json. The
    wrist & overhead are the SAME InnoMaker model but DIFFERENT physical units — focal length
    and distortion match to ~1%, yet the principal point differs by ~90px, so EACH has its own
    ChArUco calibration. Never share one file."""
    return os.path.join(_HERE, f"camera_intrinsics_{camera}.json")


class Undistorter:
    """Rectify frames to the pinhole. Maps are precomputed per frame resolution (intrinsics scaled
    if it differs from the calibration's); __call__ is one cv2.remap. `camera` picks the
    intrinsics file, `path` overrides it."""

    def __init__(self, camera: str = "wrist", alpha: float = 0.0, path: str | None = None):
        with open(path or intrinsics_path(camera)) as fp:
            c = json.load(fp)
        self.K = np.array([[c["fx"], 0.0, c["cx"]], [0.0, c["fy"], c["cy"]], [0.0, 0.0, 1.0]])
        self.D = np.asarray(c["dist_coeffs"], dtype=float)
        self.calib_w, self.calib_h = c["image_size"]
        self.alpha = alpha
        self._maps = {}        # (w, h) -> (map1, map2, rectified K)

    def _maps_for(self, w: int, h: int):
        key = (w, h)
        if key not in self._maps:
            K = self.K.copy()
            K[0] *= w / self.calib_w        # scale intrinsics to this frame's resolution
            K[1] *= h / self.calib_h
            # target = the sim camera's pinhole: square pixels, CAM_HFOV, centered pp
            f = (w / 2.0) / np.tan(np.radians(CAM_HFOV / 2.0))
            new_k = np.array([[f, 0.0, w / 2.0], [0.0, f, h / 2.0], [0.0, 0.0, 1.0]])
            m1, m2 = cv2.initUndistortRectifyMap(K, self.D, None, new_k, (w, h), cv2.CV_16SC2)
            self._maps[key] = (m1, m2, new_k)
        return self._maps[key]


    def __call__(self, img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        m1, m2, _ = self._maps_for(w, h)
        return cv2.remap(img, m1, m2, interpolation=cv2.INTER_LINEAR)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Undistort an image to the rectified pinhole.")
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--camera", default="wrist", help="wrist|overhead -> camera/camera_intrinsics_<camera>.json")
    ap.add_argument("--calib", default=None, help="explicit intrinsics path (overrides --camera)")
    ap.add_argument("--alpha", type=float, default=0.0)
    a = ap.parse_args()
    u = Undistorter(a.camera, a.alpha, a.calib)
    img = cv2.imread(a.src)
    if img is None:
        raise SystemExit(f"could not read {a.src}")
    cv2.imwrite(a.dst, u(img))
    print(f"{a.src} -> {a.dst}  ({img.shape[1]}x{img.shape[0]}, alpha={a.alpha})")
