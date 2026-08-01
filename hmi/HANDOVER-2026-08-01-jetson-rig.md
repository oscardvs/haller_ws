# Handover — Haller rig on the Jetson, 2026-08-01

The rig is **bimanual and live on the Jetson**. Both arms drive, two of three
cameras stream, the Quest teleop path is built and unit-tested but has **never
been driven by a human**. Nothing in this session has been committed — the
working tree carries all of it.

Read this before touching anything, then read
`hmi/HANDOVER-teleop-engagement.md`, whose invariants still hold.

---

## What is running right now

| piece | where | how |
|---|---|---|
| HMI backend | Jetson | `/tmp/start_backend.sh`, log `~/haller-backend.log`, `:8000` |
| Next.js dev | desktop | `:3001`, `NEXT_PUBLIC_BACKEND_URL=https://192.168.0.191:8444/api` |
| Caddy | desktop | `https://192.168.0.191:8444`, config in the session scratchpad |

None of these are services. **They do not survive a reboot** — see task 6.

Hardware: arm `left` = serial `5B14030445` over USB (`/dev/haller_arm_leader`).
Arm `right` = `5B14031413` over the **40-pin UART** (`/dev/haller_arm_uart` →
`ttyTHS1`), bypassing that board's CH343P. Mast = RealSense D455
(`/dev/haller_cam_mast`). Wrist = one IMX219 on CSI slot 1.

---

## Remaining tasks, in the order I would do them

### 1. Recalibrate both arms — the operator asked for this and it is next
Right arm reports `shoulder_pan ±86°` where left reports `±119°`. That gap is far
larger than the arms physically differ, so the right arm's sweep was cut short
last time. The wizard is verified wired end to end: five backend routes in
`server.py` with exact counterparts in `lib/calibration.ts` (NOT `api.ts` —
that is why grepping `api.ts` for "calibration" finds nothing). `start()` calls
`handle.disable_torque()` so the arm is backdrivable; `abort()` re-enables it.
Calibration files live at
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/{haller_leader,haller_follower}.json`
and do **not** travel with a git clone.

### 2. Verify the wrist camera rotation guess
`config.yaml` now sets `flip_method: 3` (CW90) for `wrist_right`. **This is an
unverified guess** from the live view — the operator appeared rotated with "up"
pointing left, which reads as a CCW-rotated source. If 3 is wrong the other
candidate is 1 (CCW90). Restart the backend and look at the Cameras tab.
Correct it in `config.yaml`, never in the browser: the rotation is applied in
the ISP so the live view, the recorded dataset, and anything a policy later sees
are the same pixels.

### 3. Camera streaming is slow — the operator's live complaint, not yet investigated
Everything below is unverified suspicion, not diagnosis. Measure before changing.
- The D455 is on a **USB 2.0 cable**: it enumerates at `480M` even plugged
  directly into a USB 3 port, and Bus 002 has nothing downstream. That caps
  colour+depth bandwidth. A proper USB 3 cable is the cheapest thing to try.
- `cameras.py` serves MJPEG at `STREAM_FPS = 15` with `JPEG_QUALITY = 80`, and
  `latest_jpeg` re-encodes **per subscriber per frame** on the Jetson's CPU. Two
  cameras × several tiles is real CPU. Encoding once and fanning out, or using
  the Tegra hardware JPEG encoder (`nvjpegenc`) in the CSI pipeline, are the
  obvious wins.
- The CSI camera captures at **1280×720@60** and is then re-encoded; nothing
  needs 60 fps for a preview.
- The operator mentioned Ethernet between desktop and Jetson. Note
  `haller-jetson-access` memory: the direct Ethernet cable has a **history of
  dropping IPv6 neighbours across two OS installs** and cost hours. Current
  traffic is over WiFi. If you move to Ethernet, verify it is stable before
  attributing any improvement to it.

### 4. CSI slot 0 — physically broken, decide and move on
The connector's retaining clip is broken, so the ribbon does not seat hard
enough to carry the MIPI lanes. i2c still reads the sensor's full mode table,
which is why it *looks* detected; every capture returns 0 frames with
`INVALID_SETTINGS` + `Argus Correctable Error Status`. Configured as
`source: placeholder` so the HMI starts clean. Either repair the connector or
put the second wrist camera on USB — `docs/power_system.md:121` already budgets
a powered hub for exactly that.

### 5. Drive the Quest teleop with an actual human — it has never been tried
Built and tested but **never operated**. The geometry has 33 passing tests
(`tests/test_vr_input.py`) against the real retargeter, and an end-to-end smoke
test produced sane joint goals, but no human has held the grip.
- Open `https://192.168.0.191:8444/teleop/vr` **in the Quest browser** and accept
  the cert once. `navigator.xr` only exists in a secure context; over plain HTTP
  the property is simply absent and the page will report "unsupported" with no
  prompt explaining why.
- Grip = dead-man, analog trigger = gripper (`1 − trigger`), engagement runs the
  same acquisition countdown as the camera path.
- Expect to tune `BodyModel` limb lengths (exposed in the panel, persisted to
  localStorage). They are not cosmetic: the elbow is synthesized from those
  lengths and the measured shoulder→controller distance, so a model longer than
  the operator's arm means they run out of reach before the robot's elbow
  straightens.
- **Watch the first engagement on a clear bench.** Nothing here has moved a real
  arm from a headset.

### 6. Make the three processes survive a reboot
Backend, Next, and Caddy are all hand-started. systemd units for the backend on
the Jetson, and something equivalent on the desktop, or this has to be
rebuilt by hand every session. Note the repo already has a legacy
`haller-robot.service` on the Orin — see the plan's warning about `Conflicts=`.

### 7. Record the first dataset
The whole point. Blocked on 1, 2 and 5. The Recorder panel is already wired and
warns correctly that with teleop stopped the action column logs last-commanded
targets rather than a demonstration.

### 8. Lower priority, known and understood
- **`arm right` intermittent UART read failures.** 4 occurrences in 5250 log
  lines: `Failed to sync read 'Present_Position' ... [TxRxResult] Incorrect
  status packet!`. Telemetry catches and continues. Suspect Tegra UART timing at
  1 Mbaud. Watch whether it worsens under teleop load — a dropped read during
  recording is worse than during idle telemetry.
- **~1.7% of `/api/base/cmd_vel` 502s** through Caddy (12 of ~700). Transient
  upstream connection-reuse hiccups under the 10 Hz base poll, not a broken
  route. Tune Caddy keepalive if it becomes visible.
- **Quest 502s on `/_next/webpack-hmr`** are Next.js dev hot-reload failing
  through the proxy. Cosmetic. A production `next build` removes it.
- **Hydration warning in the desktop browser** is Grammarly injecting
  `data-gr-ext-installed` into `<body>`. Not a bug in this codebase.
- **DC-DCs are 2 A each** against `docs/power_system.md:119`'s ~5 A per arm (and
  line 194's single 10 A buck). Fine for gentle work. Do not record a dataset
  you intend to train on until this is addressed — brownouts put position
  dropouts into the demonstrations as ground truth.
- **L2 hard E-STOP is still not wired.**

---

## Two failure modes that cost most of a day. Do not relearn them.

**A charge-only USB cable is indistinguishable from an empty port.** Power
reaches the board, servo lights come on, the barrel reads 7.4 V — and the kernel
logs *nothing at all*, because with no D+ pullup there is nothing to detect.
Silence in `dmesg` means no data pair. A failing chip is usually *noisy*.
Corroborating tell: a USB 3 device enumerating at `480M` on a USB 3 port.

**`haller` was not in `dialout`.** `/dev/ttyACM0` only ever worked because the
udev rule forces `MODE="0666"`. Any new serial node is root:dialout 0660 and
fails to open with a `ConnectionError` that reads exactly like absent hardware.
Fixed for `ttyTHS1`, and the user is now in the group, but check this first the
next time a serial device "isn't there".

A methodological note worth inheriting: I concluded the right arm's board was
dead, and it was not. Every USB test had been run with its barrel jack
disconnected. One confirmed cause (the bad cable on the *other* arm) closed the
investigation on a second, separate symptom too early. The operator recovered
the arm by driving it over the UART instead.

---

## Gotchas that will waste your time otherwise

- **Backend tests need ROS's pytest plugins suppressed**, or collection dies:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q -p pytest_asyncio.plugin -p anyio`
  → 266 passed. Frontend: `npx tsc --noEmit` and `pnpm test` (101 passed). There
  is no CI typecheck; run it yourself.
- **The venv is built with `uv`**, not `python3 -m venv` — the Jetson has no
  `python3.12-venv` and installing it needs sudo. Activate with
  `source ~/venvs/haller-hmi/bin/activate-haller-hmi` (sources ROS; `server.py`
  hard-imports `rclpy`).
- **Do not `set -u` in any script that sources ROS.** `setup.bash` dereferences
  `AMENT_TRACE_SETUP_FILES` and dies.
- **Never `pkill -f <pattern>` over ssh when the pattern appears in your own
  command line.** It kills the shell running it. This bit twice.
- **The CSI camera cannot use OpenCV.** IMX219 emits RG10 Bayer and needs the
  Tegra ISP via `nvarguscamerasrc`. lerobot pins `opencv-python-headless
  >=4.9,<4.14`; the only GStreamer-enabled build on the box is the system 4.6.0,
  below that floor. `csi_camera.py` therefore talks to GStreamer directly through
  PyGObject and leaves cv2 alone. Do not "fix" this by swapping OpenCV.
- **`WARMUP_FRAMES = 45`** in `csi_camera.py` is load-bearing: Argus needs time
  for AE/AWB to converge, and a 3-frame grab produces a dark magenta image that
  looks like broken hardware. It is not.
- **The right arm only exists while its jumper selects UART.** Moving it to
  USB—SERVO to test the CH343P makes that arm vanish from the config.
- **Caddy binds wildcard** regardless of the site address — `bind 192.168.0.191`
  is required because Tailscale holds `:8443` and `:8444` on its own interface.
  `skip_install_trust` (it shells to sudo) and `auto_https disable_redirects`
  (port 80 needs root) are also required.
- **`NEXT_PUBLIC_*` is inlined at dev-server start.** Changing the backend URL
  needs a restart, not a reload. This caused a "GET /config failed" that looked
  like a backend outage.

---

## Uncommitted work in the tree

Backend: `vr_input.py` (new, + 33 tests), `csi_camera.py` (new),
`human_teleop.py` (`vr_grip` clutch source), `server.py` (`/ws/teleop/vr/in`),
`config.py` (`sensor_id`, `flip_method`), `cameras.py` (`source: csi` branch),
`config.yaml`.
Frontend: `lib/vrTeleop.ts` (new), `components/VRTeleopPanel.tsx` (new),
`app/teleop/vr/page.tsx` (new), `lib/humanTeleopClient.ts` (generic frame type),
`lib/api.ts`, `components/DeadManIndicator.tsx`,
`components/HumanTeleopPanel.tsx` (clutch source narrowed).
Also `scripts/99-haller-devices.rules`.

All green: 266 backend, 101 frontend, typecheck clean.
