"""The vendored SO-101 MJCF loads cleanly under MuJoCo."""
from __future__ import annotations

import os
from pathlib import Path

import mujoco
import pytest

# Headless render backend — set before any MuJoCo OpenGL is touched.
os.environ.setdefault("MUJOCO_GL", "egl")

REPO_ROOT = Path(__file__).resolve().parents[4]
SO101_DIR = REPO_ROOT / "sim" / "assets" / "so101"


def _find_scene_xml() -> Path:
    # mujoco_menagerie folders ship a "scene.xml" and a robot-named xml; prefer scene.
    for name in ("scene.xml", "so_arm100.xml", "so_arm100_scene.xml"):
        cand = SO101_DIR / name
        if cand.exists():
            return cand
    xmls = list(SO101_DIR.glob("*.xml"))
    if not xmls:
        pytest.skip("no MJCF in sim/assets/so101/")
    return xmls[0]


def test_so101_mjcf_loads():
    xml = _find_scene_xml()
    model = mujoco.MjModel.from_xml_path(str(xml))
    assert model.nq >= 6, f"expected at least 6 joints, got {model.nq}"
    assert model.nu >= 6, f"expected at least 6 actuators, got {model.nu}"
