# SO-101 arm: motor configuration & calibration

This guide brings up a single SO-101 follower arm. It assumes the [LeRobot environment](./lerobot-environment.md) is already installed.

Source: official LeRobot SO-101 docs — [huggingface.co/docs/lerobot/so101](https://huggingface.co/docs/lerobot/so101).

> **Status note (2026-05-22):** this document is being written alongside the actual bring-up of Haller's first arm. Sections marked **TODO** will be filled in once that step is performed on hardware.

---

## Hardware checklist

Before you start:

- [ ] SO-101 follower arm mechanically assembled per the [TheRobotStudio assembly instructions](https://github.com/TheRobotStudio/SO-ARM100).
- [ ] 6× Feetech STS3215 servos with **1/345 gear ratio** (the follower-arm spec — the leader uses a mix of 1/191, 1/345, 1/147 across joints; do not confuse them).
- [ ] **Check the voltage variant of your servos** — STS3215 ships in multiple SKUs and the label tells you which:
  - **7.4 V variant**: operating range **6.0–7.4 V**. Use a **7.4 V supply**.
  - **12 V variant**: operating range 4–14 V. Use a **12 V supply** (the official SO-101 kit's 12 V / 5 A brick).
  - The two variants look identical but have different voltage tolerance. Wrong supply on the 7.4 V variant = "Input voltage error" + alarm LED at best, dead servos at worst.
- [ ] Feetech bus servo adapter board (Waveshare or equivalent).
  - The two jumpers select the **control path**, not the power source:
    - **`A` channel** = UART control (Pi Zero, ESP32, Arduino, STM32).
    - **`B` channel** = USB control (PCs, Pi 4B, Jetson Orin Nano). This is what you want for a desktop or Jetson host.
  - **USB does not power the servo bus.** The servos are powered exclusively from the barrel jack. You can't skip the supply by "powering from USB" — that only powers the on-board logic.
- [ ] Power supply matching your servo variant (see above), with a center-positive barrel plug that physically fits the board's DC5521 jack.
- [ ] USB cable from the adapter board to the workstation.
- [ ] At least one 3-pin TTL cable for connecting one motor at a time during configuration.

You also need shell access on the workstation with the `lerobot` conda env activated:

```bash
conda activate lerobot
```

## 1. Find the serial port for the bus adapter

Plug the bus adapter into the **barrel-jack power supply** (matched to your servo voltage variant — see checklist) and into the workstation via USB. Jumpers on **B**. Then:

```bash
lerobot-find-port
```

The tool lists all serial devices, asks you to **unplug** the adapter and press Enter, and then reports which port disappeared. On Linux this is usually `/dev/ttyACM0`.

Grant access to the port (one-time, per session — see "Persistent permissions" below for a permanent fix):

```bash
sudo chmod 666 /dev/ttyACM0          # adjust to the port you found
```

> **TODO:** record the actual port string for Haller's follower arm here so future bring-ups can skip the discovery step.

### Persistent permissions (recommended)

Adding your user to the `dialout` group lets you skip `chmod` on every reboot. Log out and back in afterwards.

```bash
sudo usermod -aG dialout "$USER"
```

For predictable device names across reboots, consider a udev rule analogous to Haller's existing `scripts/99-haller-devices.rules`.

## 2. Configure motor IDs and baud rate

Brand-new Feetech motors all ship with ID = `1`. The bus servo protocol requires unique IDs (1–6 for the SO-101 follower), and the controller and all motors must share a baud rate. These values are written to motor EEPROM, so this step is **one-time per motor**.

> **Critical:** only **one motor** may be connected to the bus during this step. The script writes ID `n` to whatever motor it can see — if multiple are on the bus, you'll overwrite IDs you've already set.

Run the LeRobot setup tool for the follower:

```bash
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0       # adjust to your port
```

The script walks you through motors 1 → 6 in order: `shoulder_pan`, `shoulder_lift`, `elbow_flex`, `wrist_flex`, `wrist_roll`, `gripper`. For each one:

1. Disconnect the 3-pin cable from the controller board.
2. Plug the cable into the next motor only (the previous motor can stay attached to its own cable, just unplug from the board).
3. Press Enter. The tool prints e.g. `'gripper' motor id set to 6`.
4. Repeat until done.

If a motor doesn't respond:

- **`Motor 'gripper' (model 'sts3215') was not found`** — the servo's V+ pin has no power. The board's logic is alive (USB-powered) but the servo bus isn't. Confirm the barrel-jack supply is connected and on, and that your supply voltage matches the servo variant (e.g. 7.4 V supply for 7.4 V STS3215). USB alone never powers the servos.
- **`[RxPacketError] Input voltage error!`** — the servo sees voltage outside its safe range and is alarming. You're feeding the wrong supply for this servo's voltage variant. Power off, swap supply (e.g. drop from 12 V to 7.4 V for the 7.4 V variant), retry.
- Check the USB cable between board and computer.
- Check the 3-pin cable is fully seated on both ends.
- On a Waveshare board, confirm both jumpers are on the `B` channel (control path = USB).

### After all six motors are configured

You can now **daisy-chain** the motors with 3-pin cables and connect the chain to the controller board. Plug the chain so that `shoulder_pan` (ID 1) is closest to the controller board and `gripper` (ID 6) is at the far end.

> **TODO:** photo of Haller's wiring + cable-routing convention.

## 3. Calibrate the arm

Calibration aligns the motors' raw encoder positions with the arm's mechanical zero/limits so a model trained on one SO-101 can transfer to another.

```bash
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0       # your port
```

Follow the on-screen prompts; the tool will ask you to physically move the arm to defined poses (typically: home, full extension at each joint, gripper open/closed). Calibration data is written to `~/.cache/huggingface/lerobot/calibration/` (verify the exact path on first run).

> **TODO:** record the calibration output path and copy the calibration file into this repo (e.g. `config/so101/<arm-name>.json`) so it lives next to the rest of the robot config.

## 4. Smoke test

For a follower-only setup (no leader arm yet), run a basic motion check that reads positions and homes the arm:

> **TODO:** insert the verified smoke-test command after the first run. Likely a `lerobot-teleoperate` with a keyboard teleop, or a direct Python snippet using `lerobot.common.robot_devices.robots.so101_follower.SO101Follower` to move to home.

## Adding a second arm

Haller will eventually carry two arms. When you bring up the second:

1. Repeat steps 1–3 using a **different USB port** so each adapter is uniquely identified.
2. Use a different `id` flag (e.g. `--robot.id=left` vs `right`) so calibration data doesn't collide.
3. Capture the serial port mapping in `scripts/99-haller-devices.rules` so the assignment survives reboots.

## References

- LeRobot SO-101 docs: <https://huggingface.co/docs/lerobot/so101>
- LeRobot install: <https://huggingface.co/docs/lerobot/installation>
- SO-ARM100 hardware: <https://github.com/TheRobotStudio/SO-ARM100>
- Feetech STS3215 servo datasheet: search Feetech's site for STS3215
