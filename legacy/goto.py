from facedetection import MAX_PITCH_DEG, MAX_YAW_DEG
from reachy_mini import ReachyMini
from reachy_mini.utils import create_head_pose
import numpy as np

with ReachyMini(media_backend="no_media") as mini:
    # Move everything at once
    mini.goto_target(
       head=create_head_pose(z=-30, mm=True),    # Up 10mm
       antennas=np.deg2rad([-180, 180]),           # Antennas out
       #body_yaw=np.deg2rad(0),                 # Turn body
       duration=2.0,
       yaw=offset_x * MAX_YAW_DEG,   # Negative: face left of center -> turn head left
       pitch=offset_y * MAX_PITCH_DEG,  # Take 2 seconds
       method="minjerk"                         # Smooth acceleration
    )