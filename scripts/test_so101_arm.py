#!/usr/bin/env python
"""Smoke test for a single SO-101 follower arm.

Connects to the arm, disables torque so the arm is freely back-drivable, and
streams joint positions to the terminal for ~15 seconds. Move the arm by hand
and watch the values change to confirm the bus, the motor IDs, and the
calibration are all good.

Usage (with the lerobot conda env active):
    python scripts/test_so101_arm.py [--port /dev/ttyACM0] [--id haller_follower] [--seconds 15]
"""

from __future__ import annotations

import argparse
import time

from lerobot.robots.so_follower import SO101Follower, SO101FollowerConfig

JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", default="/dev/ttyACM0")
    parser.add_argument("--id", default="haller_follower")
    parser.add_argument("--seconds", type=float, default=15.0)
    parser.add_argument("--rate-hz", type=float, default=5.0)
    args = parser.parse_args()

    cfg = SO101FollowerConfig(port=args.port, id=args.id, use_degrees=True)
    robot = SO101Follower(cfg)
    robot.connect(calibrate=True)
    robot.bus.disable_torque()

    print(f"Streaming positions for {args.seconds:.0f}s. Move the arm by hand.\n")
    header = "  t(s)  " + "".join(f"{j:>14}" for j in JOINTS)
    print(header)
    print("-" * len(header))

    period = 1.0 / args.rate_hz
    start = time.perf_counter()
    try:
        while (t := time.perf_counter() - start) < args.seconds:
            obs = robot.get_observation()
            row = f"{t:6.2f}  " + "".join(f"{obs[f'{j}.pos']:14.2f}" for j in JOINTS)
            print(row)
            time.sleep(period)
    finally:
        robot.disconnect()
        print("\nDisconnected.")


if __name__ == "__main__":
    main()
