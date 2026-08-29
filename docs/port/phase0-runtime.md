# Phase 0 — runtime split and the RealSense colour path

Bench record for the kit port. Every claim below is a command that was run on
this box on 2026-08-27 and the output it produced. Where a prior measurement
did not reproduce, both numbers are here and the disagreement is called out
rather than averaged away.

Rig: desktop, RTX 4080 SUPER, Python 3.12.3, one Intel RealSense D455
(serial 151422250100, fw 5.13.0.50).

---

## 1. The interpreter split

Two lerobot versions live here on purpose:

| venv | lerobot | role |
| --- | --- | --- |
| `~/venvs/haller-hmi` | 0.5.1 | **serving** — the HMI process itself |
| `~/venvs/haller-lab` | 0.6.1 | **runners** — detached train/rollout jobs |

The serving venv does not move: its recorder carries per-version workarounds
that each encode a dataset-destroying incident, and re-qualifying them costs
more than a second interpreter does. Detached jobs own no robot state and are
the only consumers of what 0.6.1 adds.

Built by `scripts/setup_lab_venv.sh`, which is idempotent (a second run
re-verifies and exits in 2.8 s), takes `--force` to rebuild and `--verify-only`
to skip to the checks. It uses `uv` when present and falls back to
`python -m venv` + `pip`. Cold build with uv 0.11.21: under a minute.

```
$ bash scripts/setup_lab_venv.sh
== reusing /home/odesha/venvs/haller-lab
== lerobot 0.6.1 already installed — skipping install
== verifying
  ok    lerobot version              0.6.1
  ok    lerobot_rollout import       .../lerobot/scripts/lerobot_rollout.py
  ok    pyrealsense2 import          imported
  ok    lerobot-train                /home/odesha/venvs/haller-lab/bin/lerobot-train
  ok    lerobot-rollout              /home/odesha/venvs/haller-lab/bin/lerobot-rollout
  ok    lerobot-record               /home/odesha/venvs/haller-lab/bin/lerobot-record
  ok    lerobot-calibrate            /home/odesha/venvs/haller-lab/bin/lerobot-calibrate
  ok    calibration JSON parses      3 file(s) parsed under .../robots/so_follower
== ready:  source /home/odesha/venvs/haller-lab/bin/activate-haller-lab
```

The script **fails** rather than reporting success if any gate is red. Those
gates are questions 1–4 below, wired in so the answer stays true rather than
being a thing this document once observed.

---

## Q1. Does 0.6.1 read the calibration JSON that 0.5.1 wrote?

**Yes — byte-for-byte identical parse, at both the dataclass and the robot
level.** This is the gate on browser-launched rollout: if the schema had
diverged, a runner would drive the arm with someone else's zero.

First, note the path. `~/.cache/huggingface/lerobot` is a **symlink** to
`/home/odesha/robot-data/lerobot`, so the two spellings in the port plan are
one directory:

```
$ ls -la ~/.cache/huggingface | grep lerobot
lrwxrwxrwx 1 odesha odesha 31 Aug 26 19:58 lerobot -> /home/odesha/robot-data/lerobot
```

Both versions resolve to the same place, via `HF_LEROBOT_HOME`:

```
$ ~/venvs/{haller-hmi,haller-lab}/bin/python -c \
    "from lerobot.utils.constants import HF_LEROBOT_CALIBRATION; print(HF_LEROBOT_CALIBRATION)"
/home/odesha/robot-data/lerobot/calibration        # 0.5.1
/home/odesha/robot-data/lerobot/calibration        # 0.6.1
```

### Dataclass level

`MotorCalibration(**spec)` for every motor in `haller_follower.json`, dumped
canonically (sorted keys) under each interpreter and diffed:

```
$ ~/venvs/haller-hmi/bin/python calib_parse.py $CALIB > calib_051.json   # lerobot 0.5.1
$ ~/venvs/haller-lab/bin/python calib_parse.py $CALIB > calib_061.json   # lerobot 0.6.1
$ diff -u calib_051.json calib_061.json
(no output — identical)
```

The field set is unchanged across the versions:

```
0.5.1 fields: ['drive_mode', 'homing_offset', 'id', 'range_max', 'range_min']
0.6.1 fields: ['drive_mode', 'homing_offset', 'id', 'range_max', 'range_min']
```

### Robot level

The stronger test — construct `SOFollower`, let it resolve its own
`calibration_fpath` and run its own `_load_calibration()`:

```
$ ~/venvs/{haller-hmi,haller-lab}/bin/python calib_robot.py
calibration_fpath = /home/odesha/robot-data/lerobot/calibration/robots/so_follower/haller_follower.json  exists = True
{
  "elbow_flex":    {"drive_mode": 0, "homing_offset":  2024, "id": 3, "range_max": 4063, "range_min": 1847},
  "gripper":       {"drive_mode": 0, "homing_offset": -1797, "id": 6, "range_max": 3492, "range_min": 2045},
  "shoulder_lift": {"drive_mode": 0, "homing_offset":  -481, "id": 2, "range_max": 4095, "range_min":    0},
  "shoulder_pan":  {"drive_mode": 0, "homing_offset": -1790, "id": 1, "range_max": 2673, "range_min":  720},
  "wrist_flex":    {"drive_mode": 0, "homing_offset":  1775, "id": 4, "range_max": 3310, "range_min":  948},
  "wrist_roll":    {"drive_mode": 0, "homing_offset":  -683, "id": 5, "range_max": 4095, "range_min":    0}
}
```

Diffed across the two interpreters with only the version banner stripped:
**identical**. Both versions also resolve the same directory layout
(`<calibration>/robots/so_follower/<id>.json`) and the same
`ROBOTS = "robots"` constant.

**Verdict: GO for browser-launched rollout on the calibration axis.**

### Two things that moved, neither of them the calibration

Recorded because they will surface as `TypeError` the first time a runner is
handed a config the serving venv built:

- `lerobot.constants` does not exist in either version. The module is
  `lerobot.utils.constants`.
- `SOFollowerConfig` gained four fields in 0.6.1:
  `position_p_coefficient`, `position_i_coefficient`, `position_d_coefficient`,
  `num_read_retries`. A config dict built for 0.6.1 and passed to 0.5.1 will be
  rejected on those keys. Cross-version traffic must stay **files**, not
  config objects — which is what this split assumes anyway.

---

## Q2. Is `lerobot.scripts.lerobot_rollout` importable under 0.6.1?

**Yes under 0.6.1, absent under 0.5.1 — as expected. This is the whole reason
the second venv exists.**

```
$ ~/venvs/haller-lab/bin/python -c "import lerobot.scripts.lerobot_rollout as r; print(r.__file__)"
/home/odesha/venvs/haller-lab/lib/python3.12/site-packages/lerobot/scripts/lerobot_rollout.py

$ ~/venvs/haller-hmi/bin/python -c "import lerobot.scripts.lerobot_rollout"
ModuleNotFoundError: No module named 'lerobot.scripts.lerobot_rollout'
```

---

## Q3. Do `lerobot-train` and `lerobot-rollout` exist in the new venv?

**Both exist.** Worth separating, though: `lerobot-train` is *not* new —
0.5.1 already ships it. Exactly two console scripts appear in 0.6.1:

```
$ diff <(ls ~/venvs/haller-hmi/bin | grep ^lerobot-) <(ls ~/venvs/haller-lab/bin | grep ^lerobot-)
0a1
> lerobot-annotate
11a13
> lerobot-rollout
```

`lerobot-rollout` runs and parses args:

```
$ ~/venvs/haller-lab/bin/lerobot-rollout --help | head -3
usage: lerobot-rollout [-h] [--config_path str] [--robot str]
                       [--robot.type {openarm_follower,bi_openarm_follower,rebot_b601_follower,
                        bi_rebot_b601_follower,so100_follower,so101_follower,bi_so_follower,...}]
                       [--teleop str]
```

So the only capability that actually forces the split is `lerobot-rollout` /
`lerobot.scripts.lerobot_rollout`. Training could in principle have stayed in
the serving venv; it moves with rollout because a runner should be one
interpreter, not two.

---

## Q4. Does `pyrealsense2` import there, and does it see the D455?

**Yes to both.**

```
$ ~/venvs/haller-lab/bin/python -c "import pyrealsense2 as rs; ..."
pyrealsense2 2.56.5.9235
  Intel RealSense D455 | serial 151422250100 | fw 5.13.0.50 | usb 2.1
```

It is **absent from the serving venv**, which is the correct state and is what
`haller_hmi/realsense.py`'s soft import is written against:

```
$ ~/venvs/haller-hmi/bin/python -c "import pyrealsense2"
ModuleNotFoundError: No module named 'pyrealsense2'
```

> **Action item, unrelated to the port:** `usb 2.1`. The D455 is negotiating a
> 480 Mbps link (`/sys/bus/usb/devices/1-4/speed` reads `480`), not USB 3's
> 5000. That caps colour resolution and frame rate and will bite during
> recording. Cable or port, not software.

---

## 2. The RealSense colour path

`hmi/backend/haller_hmi/realsense.py` — new, self-contained, not yet wired into
`cameras.py` (see *Not done* below).

### The premise, re-verified — and one correction

The port plan states: *the D455 colour node `/dev/video2` offers only GREY and
UYVY; OpenCV decodes that UYVY wrongly.* Every measurable part of that
reproduced. The **attribution** did not.

`/dev/video2` is not the colour camera. It is the stereo module's infrared
imager. Three independent sources agree:

```
$ for n in 0 1 2 3 4 5; do echo "video$n intf=$(readlink -f /sys/class/video4linux/video$n/device | sed 's|.*/||')"; done
video0 intf=1-4:1.0    video1 intf=1-4:1.0    video2 intf=1-4:1.0    video3 intf=1-4:1.0
video4 intf=1-4:1.3    video5 intf=1-4:1.3

$ ~/venvs/haller-lab/bin/python -c "...physical_port..."
Stereo Module    physical_port = .../1-4:1.0/video4linux/video0
RGB Camera       physical_port = .../1-4:1.3/video4linux/video4
Motion Module    physical_port = .../1-4:1.0/video4linux/video0
```

All six nodes report the same `name` (`Intel(R) RealSense(TM) Depth Ca`), so
nothing in `/dev` distinguishes them. `/dev/video2` gets picked because it is
the first non-depth node that opens.

Format menus, via `VIDIOC_ENUM_FMT` (no `v4l2-ctl` on this box — done with a
20-line `ctypes` ioctl):

```
== /dev/video0    [0] Z16   16-bit Depth
== /dev/video2    [0] GREY  8-bit Greyscale      <- as documented: GREY + UYVY only
                  [1] UYVY  UYVY 4:2:2
                  [2] GREY  8-bit Greyscale
== /dev/video4    [0] YUYV  YUYV 4:2:2           <- the actual RGB camera
```

### The measurements

Same scene, back to back, `colour_health()` on each. Two runs, minutes apart;
figures below are the second (the first agreed to ~1).

**`/dev/video2` — the node the plan names:**

| path | mean | magenta bias | R/B spread | verdict |
| --- | --- | --- | --- | --- |
| V4L2 + OpenCV, implicit | 65.3 | **+28.3** | 4.8 | SUSPECT |
| V4L2 + `set(FOURCC, MJPG)` | 65.2 | **+28.3** | 4.8 | SUSPECT |
| V4L2 raw + explicit `COLOR_YUV2BGR_UYVY` | 65.2 | **+28.3** | 4.8 | SUSPECT |
| librealsense RGB Camera, native `bgr8` | 101.0 | +17.4 | 26.8 | OK |

**`/dev/video4` — the actual RGB camera:**

| path | mean | magenta bias | R/B spread | verdict |
| --- | --- | --- | --- | --- |
| V4L2 + OpenCV, implicit | 100.5 | +17.3 | 23.2 | OK |
| V4L2 + `set(FOURCC, MJPG)` | 100.5 | +17.6 | 23.8 | OK |
| V4L2 raw + explicit `COLOR_YUV2BGR_UYVY` | 113.5 | **−80.7** | 8.2 | (see blind spot) |
| librealsense RGB Camera, native `bgr8` | 101.1 | +18.0 | 27.2 | OK |

What each row settles:

- **MJPG is a no-op, confirmed directly.** The fourcc reads back after the set:
  `/dev/video2 fourcc after set(MJPG): 'UYVY' (wanted 'MJPG')`. You cannot pin
  a format the node does not offer. Numbers identical to the implicit row.
- **Explicit `cvtColor` does not help, confirmed.** Telling OpenCV exactly
  which conversion to run changes nothing to one decimal place (+28.3 either
  way). There is no OpenCV-side fix for that node — because the problem is not
  the conversion, it is that this is not the colour camera.
- **OpenCV decodes the real RGB camera correctly** (100.5 / +17.3 vs
  librealsense's 101.1 / +18.0, within 0.6).

### What this changes: Haller's existing path is already correct

The kit's rule — *never record a RealSense through V4L2* — was derived from
measurements on the wrong node. The decoder was never the problem.

**Haller had already solved this, differently, and its solution measures as
good as librealsense.** `config.yaml` does not name a `/dev/videoN` at all; it
uses a udev symlink, and `scripts/99-haller-devices.rules:56` pins it to the
colour interface explicitly:

```
SUBSYSTEM=="video4linux", ATTRS{idVendor}=="8086", ATTRS{idProduct}=="0b5c",
  ENV{ID_USB_INTERFACE_NUM}=="03", ATTR{index}=="0",
  SYMLINK+="haller_cam_mast", MODE="0666"

$ ls -la /dev/haller_cam_mast
/dev/haller_cam_mast -> video4        # the RGB Camera, interface 1.3
```

Interface `03` + index `0` *is* the RGB camera, by construction — it cannot
resolve onto the IR imager however the kernel renumbers. The config comment
already says why (`the numbering is genuinely unstable: enabling the CSI camera
overlay moved the colour node from /dev/video4 to /dev/video6`).

So there are two independent solutions to one problem, and both work:

| approach | how it addresses the sensor | measured |
| --- | --- | --- |
| udev symlink + OpenCV (Haller, deployed) | USB interface number | 100.5 / +17.3 |
| librealsense (this module) | sensor role, `stream.color` | 101.1 / +18.0 |

**Recommendation: leave the mast camera path alone.** It is correct, it is
already deployed, and it keeps `pyrealsense2` out of the serving venv. This
module earns its place on the things udev cannot do — the emitter (no V4L2
equivalent), a box with no udev rule installed, and the health check.

The correction is worth carrying because it changes what a *diagnosis* looks
like. "Dark and magenta" is not evidence of a bad decoder to go fix; it is
evidence of **the wrong sensor**, and the fix is addressing, not conversion.

### Absolute numbers differ from the kit's; the shape does not

| | kit | here |
| --- | --- | --- |
| bad path | 66.5 / +31.9 | 65.3 / +28.3 |
| explicit UYVY cvtColor | 71.4 / +28.9 | 65.2 / +28.3 |
| good path | 126.1 / +9.7 | 101.0 / +17.4 |

Different room, different light. The bad path reproduces closely; the good path
reads dimmer and warmer here because this room is warmer — see below.

### `colour_health()` and the warm-scene trap

`colour_health(bgr)` returns `mean_brightness`, `magenta_bias =
(R.mean()+B.mean())/2 - G.mean()`, `rb_spread`, `channel_means`, and a verdict.
Documented interpretation: `~0` neutral, `<= +10` normal variation, `>= +25`
the broken decode.

That band is **not as clean as it looks on this rig**. The correct path here
reads **+17.4** — two thirds of the way to the threshold, on a healthy camera —
because the room is warm. The channel means show why the bias alone cannot
tell the two apart, and what can:

```
bad node   B=77.1  G=46.4  R=72.3    bias +28.3   R/B spread  4.8
good path  B=93.5  G=89.5  R=120.2   bias +17.4   R/B spread 26.8
```

Magenta requires R and B to move **together** against G. The broken decode has
them 4.8 apart; the warm room has R alone elevated, 26.8 apart. `rb_spread`
separates the two populations by ~5×, where the bias separates them by 1.6×.

So the module reports `rb_spread` and uses it to sharpen `reason`, but keys
`broken_decode` to the documented `+25` bias threshold alone — a diagnostic
whose rule changes per rig is not a diagnostic, and a reading here has to mean
the same thing as a reading on the next rig.

**Known blind spot, recorded not fixed:** this detects the *magenta* failure,
not every wrong decode. Forcing `COLOR_YUV2BGR_UYVY` onto the RGB camera's YUYV
stream measured bias **−80.7** — violently green — and was reported `ok`. A
negative branch was not added on a guess; the false-positive rate on real
scenes needs characterising first.

### The IR emitter

`disable_emitter()`. All three behavioural claims verified; the *magnitude*
claim did not reproduce.

Persistence across processes, and the write path (which is otherwise never
exercised on an already-clean rig, since it starts and ends at 0):

```
1. baseline, fresh process:              emitter_enabled = 0.0
2. write 1 in process A:                 wrote 1 -> 1.0
3. read back in process B:               emitter_enabled = 1.0     <- persists across processes
4. disable_emitter():                    returned: True            <- write path exercised
5. read back in a fresh process:         emitter_enabled = 0.0     <- it stuck
6. disable_emitter() again:              returned: False           <- idempotent, writes nothing
```

Contamination, measured as Laplacian variance of the frame, emitter on vs off,
alternating runs:

```
colour frame, depth streaming:   emitter=1.0  lapvar 502.8      emitter=0.0  lapvar 504.1
                                 emitter=1.0  lapvar 504.5      emitter=0.0  lapvar 503.8
IR imager,    depth streaming:   emitter=1.0  lapvar 117.2      emitter=0.0  lapvar 113.4
                                 emitter=1.0  lapvar 117.9      emitter=0.0  lapvar 113.7
```

The kit measured **413 → 79** on the colour frame — roughly 5× the apparent
detail, all projected dots. **That did not reproduce here: 502.8 vs 504.1 is
noise.** The projector was verifiably firing — the IR imager saw it, 117.2 vs
113.4, consistently across runs — just faintly.

Two things explain the gap, and both are worth knowing:

1. **Range and ambient light.** Dot contrast falls off with distance and drowns
   in ambient IR. This bench is a lit room at desk distance. A close, dim,
   matte subject — exactly what a wrist camera sees during manipulation — is
   the case where it bites.
2. **Which sensor was measured.** The kit's figure was very likely taken on the
   IR node, the same `/dev/video2` mix-up as above. An IR imager sees the dot
   pattern by design; this D455's RGB module largely does not.

Also learned, and now documented in the module: **the emitter only fires while
a depth stream is running.** A colour-only pipeline is unaffected regardless of
the setting. The first attempt at this measurement enabled colour only and
found nothing, correctly.

`disable_emitter()` stays. It is free, it persists, and it removes a variable
that is scene-dependent rather than absent. But `413 → 79` should not be quoted
as a number this rig reproduces.

### Live report

```
$ PYTHONPATH=hmi/backend ~/venvs/haller-lab/bin/python -m haller_hmi.realsense
== RealSense status
{
  "available": true,
  "pyrealsense2": true,
  "import_error": null,
  "devices": [{"name": "Intel RealSense D455", "serial": "151422250100", "firmware": "5.13.0.50"}]
}

== IR emitter
changed: False

== Colour health (librealsense, bgr8)
shape           (480, 640, 3)
mean brightness 100.8 / 255
magenta bias    +17.6  (>= +25 is the broken UYVY decode)
R/B spread      27.2  (> 15 means warm scene, not magenta)
channel means   B=93.0 G=89.1 R=120.3
verdict         OK — colour path looks correct
```

### Soft dependency, verified

The module must import and report honestly on a box with no `pyrealsense2` —
teleop has to come up on a rig with no depth camera at all. Run under the
**serving** venv, which has no `pyrealsense2`:

```
$ PYTHONPATH=hmi/backend ~/venvs/haller-hmi/bin/python -m haller_hmi.realsense
== RealSense status
{"available": false, "pyrealsense2": false,
 "import_error": "ModuleNotFoundError: No module named 'pyrealsense2'", "devices": []}
No RealSense available — nothing to measure.

$ ... -c "from haller_hmi import realsense as R; ..."
available: False
disable_emitter(): False          # no raise
grab_colour(): None               # no raise
frames yielded: []                # empty stream, not an exception
neutral health: mean 128.0, bias 0.0, ok True   # colour_health works with no camera
```

The guarded import catches `Exception`, not `ImportError`: `pyrealsense2` is a
compiled extension, and a mismatched `libstdc++` or missing `libusb` raises
`OSError` from the loader.

---

## Test baseline

Unchanged. The new module is not yet imported by anything.

```
$ PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio -q
593 passed, 391 warnings in 37.26s
```

---

## Not done — needs a decision outside this change

1. **`pyrealsense2` is not declared anywhere.** Deliberately not added to
   `hmi/backend/pyproject.toml`. It currently exists only in the lab venv,
   pulled in by lerobot's `intelrealsense` extra. If the serving venv is ever
   to render a RealSense preview, it needs its own declaration — as an optional
   extra, so that the serving venv stays installable without it.

2. **`cameras.py` has no `source: "realsense"`.** `CameraConfig.source` is
   `"placeholder" | "opencv" | "csi" | "mjpeg" | "webrtc" | "sim_camera"`, and
   `CameraManager` dispatches on it. Wiring this module in means a new source
   value, a handle mirroring `CameraHandle`'s surface
   (`connect`/`disconnect`/`active`/`latest_rgb`/`latest_jpeg`), and a
   `serial` field on `CameraConfig` to address the camera by role rather than
   by path. `csi_camera.CSICameraHandle` is the precedent for exactly this
   shape.

   **This is not urgent for the mast camera** — §2 shows the existing udev +
   OpenCV path is already correct. It matters for a box without the udev rule.

3. **`disable_emitter()` has no caller.** It is the one thing here with no V4L2
   equivalent, and the emitter setting persists in the camera, so it wants to
   run once at recorder startup rather than on every frame. The kit calls its
   equivalent from `record_so101.py` before building camera configs.

4. **`colour_health()` has no caller either.** The natural home is a one-shot
   check when the recorder opens a camera — log the numbers, refuse or warn on
   `broken_decode`. That is the piece that turns a silently ruined dataset into
   a startup message.
