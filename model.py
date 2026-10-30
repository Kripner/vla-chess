#!/usr/bin/env python3
"""SO-101 physical description shared by the kinematics, the simulator and the policy.
Plain literals, no imports.
"""

MOTOR_NAMES = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

# Per joint, MOTOR_NAMES order: `range` = URDF joint limits in rad (assets/so101/so101.urdf),
# zero = range midpoint; `flip` = sign from the servo direction to the URDF axis; `range_mapped`
# (gripper only) = its calibrated travel maps linearly onto the jaw range.
JOINTS = {
    "shoulder_pan":  {"range": (-1.91986, 1.91986), "flip":  1},
    "shoulder_lift": {"range": (-1.74533, 1.74533), "flip":  1},
    "elbow_flex":    {"range": (-1.69, 1.69),       "flip":  1},
    "wrist_flex":    {"range": (-1.65806, 1.65806), "flip":  1},
    "wrist_roll":    {"range": (-2.74385, 2.84121), "flip":  1},
    "gripper":       {"range": (-0.17453, 1.74533), "flip":  1, "range_mapped": True},
}

# Rest poses as raw STS3215 ticks (arm.mapping derives the radian forms).
# Home: the folded rest. Ready: the raised stance above the near board where every episode
# starts and ends. Away: parks the arm out of the way.
HOME_TICKS = {"shoulder_pan": 2010, "shoulder_lift": 967, "elbow_flex": 3025,
              "wrist_flex": 1984, "wrist_roll": 2040, "gripper": 2121}
READY_TICKS = {"shoulder_pan": 1980, "shoulder_lift": 1605, "elbow_flex": 1780,
               "wrist_flex": 3129, "wrist_roll": 2046, "gripper": 2128}
AWAY_TICKS = {"shoulder_pan": 1992, "shoulder_lift": 1689, "elbow_flex": 1406,
              "wrist_flex": 2754, "wrist_roll": 2045, "gripper": 2126}

# Cameras: a rigid pose in `frame`, meters. `t` = lens position, `look_at` = a point on the optical
# axis (aim only), `up` = image-up direction (pins the roll). overhead: fixed world pose;
# wrist: in the gripper_link frame. Both are calibrated on the real arm.
# CAM_HFOV (deg): the rectified pinhole FOV the sim renders and the real stream is undistorted to.
CAM_HFOV = 74.9
CAMERAS = {
    "overhead": {"frame": "world",      "t": (0.116, 0.031, 0.392),
                 "look_at": (0.116, 0.220, -0.034), "up": (0.0, 0.0, 1.0)},
    "wrist":    {"frame": "gripper_link", "t": (0.0042, 0.0662, -0.0124), "look_at": (0.0045, 0.0229, -0.1025), "up": (0.9998, -0.0166, 0.0113)},
}

# Arm base height above the table top (m, +Z); world z=0 is the table top, the board lies on it.
ARM_RAISE = 0.0055
