# Weekend 2026-08-08: real data collection from the desktop (no Jetson)

The rig's current state forces a desktop-centric setup: the 7.4 V DC-DC is
burnt (spares on hand, but they are 2 A parts against a ~5 A/arm draw — do
not put them back on the arm rail), and one servo board's USB-C port is
burnt. The Jetson is skipped entirely. Everything below runs on **this
desktop**.

What you have: one Hilti battery, one adjustable bench supply, 2× SO-ARM101
arms on the tower, RealSense D455 (tower), IMX219 gripper cams (CSI —
Jetson-only, unusable this weekend), Meta Quest Pro, one Waveshare servo
board with a working USB-C, one with a burnt USB-C, a generic USB-UART
module, a CANable (wheel base — not needed for arm data collection).

---

## 1. Sim first (zero hardware — works today, verified 2026-08-07)

```bash
scripts/quest-teleop/up.sh --sim
# Quest browser: https://192.168.0.191:8444/teleop/vr  (accept cert once)
```

Draft the task + HF user in the desktop cockpit (`/` → Dataset tab), Enter
Passthrough, **hold A/X ~0.5 s** to start/stop takes from inside the headset
(HUD shows `● REC <frames>`). Datasets: `~/.cache/huggingface/lerobot/
<hf_user>/<task-slug>`, 30 fps h264, state+action+base+wall_clock + both sim
cameras. Full details: `hmi/QUICKSTART-QUEST.md` → "Collecting datasets in
sim". Regression suite: `python scripts/vr_smoke.py --base
http://localhost:8000` (38 checks).

Do this FIRST — it is the exact code path the real run uses, and any problem
found here is cheaper than found with torque on.

## 2. Power: bench supply, not the small bucks

The arm rail wants **7.4 V at ~5 A per arm**. The 2 A bucks brown out under
load and one already died this way — do not use them on the arms, even one
arm at a time (a stall event is exactly a current spike).

1. Bench supply → **7.4 V**, current limit as high as it goes (≥ 10 A if two
   arms, ≥ 5 A for one).
2. Wire it to BOTH servo boards' barrel inputs (DC5521 5.5×2.1 mm). The
   boards pass power through to the servo daisy-chain.
3. Servo power ONLY from the bench supply. The Waveshare boards' USB side is
   logic-level; never feed 7.4 V into a USB port.
4. The Hilti battery stays for the mobile base/wheels later — not needed
   this weekend.

## 3. Bus wiring: two arms, one good USB-C

**Left arm (working board, serial 5B14030445):**
- Jumper on **USB-SERVO**. USB-C straight into the desktop.
- udev symlink `/dev/haller_arm_leader` (install
  `scripts/99-haller-devices.rules` on the desktop if not already:
  `sudo cp scripts/99-haller-devices.rules /etc/udev/rules.d/ &&
   sudo udevadm control --reload-rules && sudo udevadm trigger`).

**Right arm (burnt-USB board, serial 5B14031413):** same trick the Jetson
used — bypass USB entirely, drive the bus off the board's UART header.
- Jumper moves to **UART**.
- USB-UART module wired to the board's UART header: GND→GND, and
  Waveshare's straight-through convention **module TX → header TX,
  module RX → header RX** (proven at 1 Mbaud on this exact board from the
  Jetson's pins 6/8/10).
- Module into a desktop USB port. Then pin it to a stable name:
  1. `lsusb` → note the module's `idVendor:idProduct` (CH340 ≈ `1a86:7523`,
     CP2102 `10c4:ea60`, FTDI `0403:6001`).
  2. Fill them into the commented `haller_arm_uart_usb` block in
     `scripts/99-haller-devices.rules`, install + reload as above.
  3. Replug; `ls -l /dev/haller_arm_uart_usb` must appear.
- Sanity-scan the bus before the HMI: each arm should answer with its six
  STS3215 IDs. (`test.py` / `can_test.py` at repo root are the scratch
  scripts from previous bring-ups; the HMI's startup log also lists found
  servos.)

If a board doesn't enumerate even on its UART header: stop, that arm is out
— go to §5 (hybrid fallback).

## 4. Bring-up: real arms, desktop backend

```bash
scripts/quest-teleop/up.sh --local
```

That's the `up.sh` you know, but the backend runs on this machine against
`hmi/backend/config.desktop-real.yaml` (RealSense on `/dev/haller_cam_mast`
via OpenCV, wrist cams placeholder'd out — no CSI on the desktop — telemetry
at the real-bus 20 Hz, so datasets record at 20 fps). Quest opens the same
URL as sim. **Startup refuses to come up if an arm's bus is missing** —
that's deliberate; check the log at `/tmp/haller-quest/sim-backend.log`.

Then the first-hardware-run checklist in `hmi/QUICKSTART-QUEST.md`, no
shortcuts:
1. **Measure the collision mounts** (base-bolt to base-bolt, ±half into
   `config.desktop-real.yaml` → `collision.mounts`), restart backend.
2. Clearance readout moves when you hand-move the arms toward each other.
3. Direction check one arm at a time (mirror selector if inverted).
4. Trigger range, slow reach, then provoke the guard once gently.
5. Only then record: cockpit Dataset tab draft → A/X hold in the headset.
   Real takes land in the same `~/.cache/huggingface/lerobot/...` tree,
   same schema (wrist channels absent — expected).

## 5. Fallback: only one arm's bus works

`hmi/backend/config.hybrid-real-sim.yaml`: real **left** arm + MuJoCo
**right** arm. VR teleop needs two enabled arms, and this satisfies it —
your left hand drives the real arm, your right hand a sim arm, the collision
guard and recorder see both. Takes record the real side's true state/action
(slice the 6 real dims out of the 12 for single-arm training; the sim half
is padding, not bimanual training data).

```bash
HALLER_HMI_CONFIG=config.hybrid-real-sim.yaml scripts/quest-teleop/up.sh --local
```

## 6. What "done" looks like this weekend

- [ ] Sim: one full task recorded start-to-finish from the headset, replayed
      in sim (`/teleop/sim/start` replay — see QUICKSTART), videos eyeballed.
- [ ] Real: first-run checklist passed with the bench supply.
- [ ] Real: ≥ 20 takes of one task recorded (20 fps, h264, wall_clock on
      every frame — check `skipped` stays 0 in the cockpit).
- [ ] `scripts/vr_smoke.py` still 38/38 after any change you make.

Known gaps carried over (not weekend work): wrist cameras (CSI), wheel base
(CANable + MF5010s), proper 10 A buck for the arm rail (XY6020L on order),
mount measurement permanently into config.yaml.
