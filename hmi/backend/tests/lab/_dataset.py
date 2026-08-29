# hmi/backend/tests/lab/_dataset.py
"""Synthetic LeRobot v3.0 dataset trees, written by hand for the Lab tests.

Ported from the kit's `tools/smoke_test_dataui.py::make_dataset`. Every Track B
test builds its dataset here instead of pointing at `~/robot-data/lerobot`: the
real trees are the equivalence anchors, they are gigabytes of mp4, and a test
that marks or prunes one destroys a recording that cost an evening to capture.

The episode CONTENT is the kit's, unchanged, because the ported smoke
assertions grade it and expect `["PASS", "FAIL"]` with "arm never moved" on
episode 1. Generalising the rig, the camera set and the gripper calibration is
additive — every default reproduces the kit's tree.

Nothing here imports `haller_hmi.lab`. A builder that went through the catalog
could not be used to test the catalog, and one bad import would take every
fixture in the suite down with it. json + pandas + numpy, same as the kit.

Layout produced (v3.0 packs MANY episodes into ONE parquet and ONE mp4; an
episode is a slice, which is why the meta rows carry index and timestamp
ranges):

    <root>/meta/info.json
    <root>/meta/tasks.parquet
    <root>/meta/episodes/chunk-000/file-000.parquet
    <root>/data/chunk-000/file-000.parquet
    <root>/videos/<video_key>/chunk-000/file-000.mp4
"""
from __future__ import annotations

import datetime as dt
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# ---- episode content ----

#: A pick-and-place: the arm sweeps and the gripper closes then reopens.
CLEAN = "clean"
#: The clutch was never engaged — the arm sat at rest for the whole episode.
STILL = "still"

# ---- rig column layouts ----

#: The kit's single-arm rig. `.pos`-suffixed, which is how public SO-101
#: recordings spell the six columns.
SOLO_NAMES = (
    "shoulder_pan.pos", "shoulder_lift.pos", "elbow_flex.pos",
    "wrist_flex.pos", "wrist_roll.pos", "gripper.pos",
)

#: Haller's recorder writes the same six joints per side with NO suffix — copied
#: from local/haller_pick_the_red_cube_and_place_it_in_the_box/meta/info.json.
#: A fixture that invented `.pos` here would exercise a spelling that exists
#: nowhere on disk, and index 5 being the LEFT gripper is the whole reason the
#: grader is per-arm.
BIMANUAL_NAMES = (
    "left_shoulder_pan", "left_shoulder_lift", "left_elbow_flex",
    "left_wrist_flex", "left_wrist_roll", "left_gripper",
    "right_shoulder_pan", "right_shoulder_lift", "right_elbow_flex",
    "right_wrist_flex", "right_wrist_roll", "right_gripper",
)

RIG_NAMES = {"solo": SOLO_NAMES, "bimanual": BIMANUAL_NAMES}

#: Sides in column order. "" is the unprefixed solo arm, matching `ArmSpec.side`.
RIG_SIDES = {"solo": ("",), "bimanual": ("left", "right")}

#: Nothing derives the rig from `robot_type` (that comes from the state names),
#: but a bimanual fixture claiming to be an `so_follower` is a lie a reader has
#: to un-learn. Solo keeps the kit's value.
RIG_ROBOT_TYPES = {"solo": "so_follower", "bimanual": "haller_bimanual"}

#: Columns per SO-101 arm; index 0 is shoulder_pan and index 5 the gripper.
ARM_COLUMNS = 6

#: Written for the non-gripper columns of a `gripper_range` calibration block.
#: No rule reads it — only the gripper's min_deg/max_deg is consulted — so it is
#: wide enough to be obviously a placeholder.
JOINT_RANGE_DEG = (-180.0, 180.0)


def state_names(rig: str = "solo") -> tuple[str, ...]:
    """The `observation.state` column names a given rig records."""
    try:
        return RIG_NAMES[rig]
    except KeyError:
        raise ValueError(f"rig must be one of {tuple(RIG_NAMES)}, got {rig!r}") from None


def _is_gripper(name: str) -> bool:
    """The catalog's rule: strip a trailing `.pos`, then match the base."""
    return name.removesuffix(".pos").endswith("gripper")


def _resolve_content(
    arm_content: Mapping[str, str] | None, sides: tuple[str, ...]
) -> dict[str, str]:
    """Per-side overrides, keyed the way `ArmSpec.side` is ("" for a solo rig,
    with "solo" accepted as a readable alias)."""
    out: dict[str, str] = {}
    for key, content in (arm_content or {}).items():
        side = "" if key in ("", "solo") else key
        if side not in sides:
            raise ValueError(f"arm_content side {key!r} is not one of {sides}")
        if content not in (CLEAN, STILL):
            raise ValueError(f"arm_content must be {CLEAN!r} or {STILL!r}, got {content!r}")
        out[side] = content
    return out


def _fill_arm(state: np.ndarray, col0: int, content: str, n: int) -> None:
    """One arm's six columns, in place."""
    state[:, col0 + 5] = 100.0                            # gripper open
    if content == CLEAN:
        state[:, col0] = np.linspace(0, 40, n)            # a real joint sweep
        state[30:60, col0 + 5] = 10.0                     # close, then reopen


def _calibration(
    names: tuple[str, ...],
    gripper_range: tuple[float, float] | Mapping[str, tuple[float, float]],
) -> dict:
    """A `haller_joint_calibration` block shaped like the real bimanual
    dataset's: keyed by RAW column name, most fields null because a declared
    joint range is all a sim rig knows. `RigSpec` reads min_deg/max_deg and
    nothing else, and falls back to (0, 100) when the block is absent — which is
    why `gripper_range=None` writes no block at all."""
    per_column: dict[str, tuple[float, float]] = {}
    if isinstance(gripper_range, Mapping):
        per_column = {str(k): (float(v[0]), float(v[1])) for k, v in gripper_range.items()}
        unknown = set(per_column) - set(names)
        if unknown:
            raise ValueError(f"gripper_range names no such column: {sorted(unknown)}")
    else:
        lo, hi = float(gripper_range[0]), float(gripper_range[1])
        per_column = {n: (lo, hi) for n in names if _is_gripper(n)}

    joints = {}
    for name in names:
        lo, hi = per_column.get(name, JOINT_RANGE_DEG)
        joints[name] = {
            "source": "declared_joint_range",
            "range_min_ticks": None,
            "range_max_ticks": None,
            "homing_offset": None,
            "drive_mode": None,
            "resolution": None,
            "deg_per_tick": None,
            "norm_mode": None,
            "min_deg": float(lo),
            "max_deg": float(hi),
        }
    return {"state_unit": "deg", "joints": joints}


def make_dataset(
    root: Path | str,
    n_episodes: int = 2,
    *,
    rig: str = "solo",
    task: str = "Test task",
    video_keys: Sequence[str] = ("observation.images.top",),
    fps: int = 30,
    gripper_range: tuple[float, float] | Mapping[str, tuple[float, float]] | None = None,
    arm_content: Mapping[str, str] | None = None,
) -> None:
    """Build a readable v3.0 tree at `root`.

    Defaults reproduce the kit's dataset exactly: episode 0 is a clean grasp,
    episode 1 the arm never moving, any further episodes repeat the clean grasp.

    `rig="bimanual"` gives the 12 unprefixed Haller columns with BOTH arms
    carrying that same content. `arm_content={"left": CLEAN, "right": STILL}`
    pins a side for every episode instead, so `n_episodes=1` with that override
    is one episode whose left arm worked and whose right arm never moved — the
    case a solo grader reports as PASS and the per-arm grader must not.

    `gripper_range=(lo, hi)` adds a `haller_joint_calibration` block so the
    grader's 40 %/70 % thresholds land somewhere other than 40 and 70; pass a
    {column: (lo, hi)} mapping to calibrate the two grippers differently.
    """
    root = Path(root)
    names = state_names(rig)
    sides = RIG_SIDES[rig]
    dim = len(names)
    video_keys = tuple(video_keys)
    override = _resolve_content(arm_content, sides)

    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
    for key in video_keys:
        (root / "videos" / key / "chunk-000").mkdir(parents=True, exist_ok=True)

    rows, ep_meta = [], []
    cursor = 0
    for ep in range(n_episodes):
        default = STILL if ep == 1 else CLEAN
        # Distinct lengths, so a renumbering is detectable PER MARK: a prune
        # renumbers the survivors, and the only evidence that the episode at a
        # marked index is a different one is that its length changed. Equal
        # lengths would make the prune invisible and the mark silently wrong.
        n = 90 + ep
        state = np.zeros((n, dim), dtype=np.float32)
        for side, col0 in zip(sides, range(0, dim, ARM_COLUMNS)):
            _fill_arm(state, col0, override.get(side, default), n)
        action = state + 0.1
        for i in range(n):
            rows.append({
                "action": action[i], "observation.state": state[i],
                "timestamp": i / fps, "frame_index": i, "episode_index": ep,
                "index": cursor + i, "task_index": 0,
            })
        meta = {
            "episode_index": ep, "tasks": [task], "length": n,
            "data/chunk_index": 0, "data/file_index": 0,
            "dataset_from_index": cursor, "dataset_to_index": cursor + n,
        }
        for key in video_keys:
            # The whole dataset is one mp4 per key, so an episode's video is the
            # cursor slice of it — that is what lets the player seek instead of
            # transcoding.
            meta[f"videos/{key}/chunk_index"] = 0
            meta[f"videos/{key}/file_index"] = 0
            meta[f"videos/{key}/from_timestamp"] = cursor / fps
            meta[f"videos/{key}/to_timestamp"] = (cursor + n) / fps
        ep_meta.append(meta)
        cursor += n

    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(ep_meta).to_parquet(
        root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    # v3.0 writes the task string as the INDEX, and the catalog reads it back
    # that way.
    pd.DataFrame({"task": [task]}).set_index("task").assign(task_index=0).to_parquet(
        root / "meta" / "tasks.parquet")
    for key in video_keys:
        (root / "videos" / key / "chunk-000" / "file-000.mp4").write_bytes(b"\0" * 64)

    features = {
        # The kit filled the action names with placeholders and its committed
        # verdicts were generated against that; only observation.state names are
        # ever derived from. The bimanual rig mirrors the real dataset instead.
        "action": {
            "dtype": "float32", "shape": [dim],
            "names": ["a"] * dim if rig == "solo" else list(names),
        },
        "observation.state": {"dtype": "float32", "shape": [dim], "names": list(names)},
    }
    for key in video_keys:
        features[key] = {"dtype": "video", "shape": [480, 640, 3]}

    info = {
        "codebase_version": "v3.0", "fps": fps, "robot_type": RIG_ROBOT_TYPES[rig],
        "total_episodes": n_episodes, "total_frames": cursor, "total_tasks": 1,
        # data_path is not in the kit's synthetic info but IS in every real
        # dataset, so a reader that formats paths from it rather than globbing
        # still finds the parquet.
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": features,
    }
    if gripper_range is not None:
        info["haller_joint_calibration"] = _calibration(names, gripper_range)
    (root / "meta" / "info.json").write_text(json.dumps(info))


def write_review(
    root: Path | str,
    marks: Mapping[int, str | Mapping],
    *,
    version: int = 1,
    frames: Mapping[int, int] | None = None,
    fingerprint: Mapping[str, int] | None = None,
) -> Path:
    """Write `review.json` by hand, bypassing `review.py`.

    A test needs to construct files `review.py` will never write again: the real
    46-episode review on `local/so101_pick_cube` is a v1 file with no `tags`, no
    `batches` and no per-mark `frames`, and it must keep reading as 35 keep / 11
    reject. Going through the writer to build that fixture would only prove the
    writer round-trips itself.

    `marks` maps an episode index to a status string, or to a whole entry when a
    note or tags are wanted. `frames` attaches the per-mark lengths that make a
    mark verifiable after a prune — omit it for the v1 case. `fingerprint`
    defaults to the totals in the dataset's own info.json.
    """
    root = Path(root)
    episodes: dict[str, dict] = {}
    for ep, mark in (marks or {}).items():
        entry = dict(mark) if isinstance(mark, Mapping) else {"status": str(mark)}
        if frames and int(ep) in frames:
            entry["frames"] = int(frames[int(ep)])
        episodes[str(int(ep))] = entry

    now = dt.datetime.now(dt.UTC).isoformat(timespec="seconds").replace("+00:00", "Z")
    data: dict = {
        "version": int(version),
        "updated": now,
        "fingerprint": dict(fingerprint) if fingerprint is not None else _fingerprint(root),
        "episodes": episodes,
    }
    if version >= 2:
        data["batches"] = []

    path = root / "review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")
    return path


def _fingerprint(root: Path) -> dict:
    """Totals from the dataset's own info.json. The real v1 review carries them,
    so a fixture that left them empty would not be the file we must keep
    reading."""
    try:
        info = json.loads((root / "meta" / "info.json").read_text())
    except (OSError, ValueError):
        return {}
    return {
        "total_episodes": int(info.get("total_episodes", 0)),
        "total_frames": int(info.get("total_frames", 0)),
    }
