import json

import pytest

from haller_hmi.presets import PresetStore, PresetNotFound


def test_save_and_load_roundtrip(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("home", "right", {"shoulder_pan": 0.0, "gripper": 0.0})
    store.save("ready", "right", {"shoulder_pan": 30.0})
    out = store.get("home", "right")
    assert out == {"shoulder_pan": 0.0, "gripper": 0.0}
    out2 = store.get("ready", "right")
    assert out2 == {"shoulder_pan": 30.0}


def test_missing_preset_raises(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    with pytest.raises(PresetNotFound):
        store.get("home", "right")


def test_list_returns_all_for_arm(tmp_path):
    store = PresetStore(tmp_path / "presets.json")
    store.save("home", "right", {"shoulder_pan": 0.0})
    store.save("home", "left", {"shoulder_pan": 0.0})
    assert sorted(store.list("right")) == ["home"]
    assert sorted(store.list("left")) == ["home"]


def test_file_persists_on_disk(tmp_path):
    path = tmp_path / "presets.json"
    store1 = PresetStore(path)
    store1.save("home", "right", {"shoulder_pan": 12.5})
    # re-open and read
    store2 = PresetStore(path)
    assert store2.get("home", "right") == {"shoulder_pan": 12.5}
