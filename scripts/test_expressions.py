"""Spielt Ausdruecke aus dem Aktionsraum am Roboter/in der Sim ab.

    python scripts/test_expressions.py            # alle
    python scripts/test_expressions.py winken     # nur einer
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reachy_hri.actions import ACTION_NAMES, EXPRESSIONS, Action  # noqa: E402
from reachy_hri.motion import MotionController  # noqa: E402
from reachy_hri.robot import make_backend  # noqa: E402

names = sys.argv[1:] or [n for n in ACTION_NAMES if n in EXPRESSIONS]
robot = make_backend("mock" if "--mock" in names else "reachy")
names = [n for n in names if n != "--mock"]
robot.open()
motion = MotionController(robot)
motion.start()
try:
    for name in names:
        print(name)
        motion.submit(Action(name))
        v = EXPRESSIONS[name][0]
        time.sleep(len(v["keyframes"]) * v["step_s"] + 1.0)
    motion.submit(Action("neutral"))
    time.sleep(1.2)
finally:
    motion.stop()
    robot.close()
