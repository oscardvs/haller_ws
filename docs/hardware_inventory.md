# Haller Hardware Inventory

Running list of physical parts on the robot, what is known about each, and what
still needs recording.

Status key:

- **✅ Confirmed** — verified in the repo config, on a datasheet, or measured
- **⚠️ Assumed** — inferred but not verified; check before relying on it
- **❓ Unknown** — needs someone to look at the robot and fill it in

Related: [`power_system.md`](power_system.md) covers the power chain in depth.

---

## Compute

| Part | Qty | Details | Status |
|---|---|---|---|
| NVIDIA Jetson Orin Nano Developer Kit | 1 | 9–20 V in, 45 W budget, barrel 5.5×2.5 mm centre-positive. Power modes 15 W / 25 W / MAXN SUPER | ✅ |
| microSD / NVMe boot media | 1 | Capacity and type not recorded | ❓ |

Software baseline: Ubuntu 22.04 / JetPack, ROS 2 Humble.

---

## Sensing

| Part | Qty | Details | Status |
|---|---|---|---|
| Slamtec RPLIDAR A1M8 | 1 | Serial, `/dev/ttyUSB0` @ 115200, scan mode `Sensitivity`, frame `laser_frame`. USB-powered 5 V | ✅ |
| IMX219 camera, module **J8IM228-V3.0** | 1 | CSI, `/dev/video0`, Bayer RGGB 1280×720 @ 30 fps via `gscam`. Wide-angle lens with noted barrel distortion | ✅ |
| IMU | ? | `robot_localization` is a declared dependency, which implies an IMU — none found in config | ❓ |

Camera is not yet intrinsically calibrated (`camera_info_url` is empty in
`haller_vision/config/camera/imx219_hardware.yaml`).

---

## Actuation

Drive layout: **differential drive, 2 driven front wheels + 1 rear caster**
(3 wheels total).

| Part | Qty | Details | Status |
|---|---|---|---|
| LK-TECH / MyActuator **MF5010** BLDC | 2 | One per driven front wheel. 16 V rated winding. Variant **10T vs 35T not recorded** — differs 4× in current, 3.5× in speed | ⚠️ |
| Motor controller **MCF302CB** | ? | Referenced via `docs/LK-demo-MCF302CB-CAN.zip`. Quantity, input voltage range, and whether one board drives both motors all unrecorded | ❓ |
| Gearbox / reduction | ? | See note below | ❓ |
| Driven wheels | 2 | Radius 0.05 m (⌀100 mm), track width 0.34 m | ✅ |
| Rear caster | 1 | Passive | ✅ |

### MF5010 variants (from `docs/MF5010_Specs.pdf`)

| Parameter | 10T | 35T |
|---|---|---|
| Rated current | 5.06 A | 1.35 A |
| Max power | 128 W | 12 W |
| Max speed | 3050 rpm | 870 rpm |
| Speed constant | 150 rpm/V | 27.5 rpm/V |

### Note: is there a reduction stage?

The controller config caps linear velocity at 1.0 m/s. With a 0.05 m wheel
radius that is 20 rad/s ≈ **191 rpm at the wheel**. The MF5010 is rated
2400 rpm (10T). Either there is a gearbox, or the motors run direct-drive at
under 10% of rated speed — which would mean poor torque utilisation and
significant heating at low RPM.

The MF series is generally the gearbox-less line (as opposed to RMD-X). **This
needs confirming physically**, because the whole torque budget depends on it.

---

## Power

Full detail in [`power_system.md`](power_system.md).

| Part | Qty | Details | Status |
|---|---|---|---|
| Hilti **B 22-195** Nuron battery | 1 | 6S3P, 21.6 V nominal, 9.0 Ah, 194.4 Wh, 1.33 kg. Measured 24.3 V at near-full | ✅ |
| Hilti sliding battery connector | 1 | 6 wires: 2× B+, 2× B−, white + blue data (leave unconnected) | ✅ |
| Hilti Nuron charger | ? | C 4-22 / C 6-22 / C 8-22 compatible; which one is on hand not recorded | ❓ |
| **LM2596** buck module | 1 | ~2 A real continuous. **Not suitable for the Jetson** — bench/auxiliary use only | ✅ |
| 12 V ≥ 10 A synchronous buck | 0 | **TO BUY.** Must be 15–40 V input | ❌ Missing |
| 20 A blade fuse + holder | 0 | Main protection at B+ | ❌ Missing |
| 5 A fuse + holder | 0 | DC-DC branch | ❌ Missing |
| TVS diode + bulk capacitor | 0 | Regen spike suppression on the motor rail | ❌ Missing |

---

## Interconnect

| Part | Qty | Details | Status |
|---|---|---|---|
| CAN interface | ? | **Gap — see below** | ❓ |
| USB cable, LiDAR | 1 | To Jetson | ⚠️ |
| CSI ribbon, camera | 1 | To Jetson | ⚠️ |
| Main power wiring | — | Gauge not recorded; needs to carry ~12.5 A peak | ❓ |

### Gap: how do the motors actually connect?

The motor documentation is entirely **CAN** (`LK-TECH motor control protocol
(CAN) V2.35`, `LK-demo-MCF302CB-CAN.zip`, `LK test command-CAN.txt`), but
`haller_hardware_interface` defaults to a **serial** port at 115200 baud.

So either there is a USB-CAN adapter that is not recorded anywhere, or the
hardware interface is still a stub. The interface currently logs
`"Connecting to hardware ... (simulated)"`, which points to the latter.

Whichever it is, the physical CAN adapter (or transceiver wiring to the Jetson's
CAN pins) is an inventory item nobody has written down.

> ⚠️ **Port conflict.** `haller_hardware_interface` defaults to `/dev/ttyUSB0`
> at 115200 — the *same device and baud* the RPLIDAR is configured for in
> `haller_hardware/config/rplidar.yaml`. These cannot coexist. Whichever
> enumerates first wins, and the other fails or misbehaves. Resolve with udev
> symlinks (`/dev/haller_lidar`, `/dev/haller_motors`) before real bring-up.

---

## Open questions

Ordered roughly by how much they block progress.

1. **MF5010 winding: 10T or 35T?** Sets the entire current and fuse budget.
2. **MCF302CB input voltage range?** Determines whether it can sit on the raw
   25.2 V pack rail, or needs its own converter.
3. **Is there a gearbox?** The 191 rpm vs 2400 rpm gap has to be explained.
4. **How do motors physically connect** — USB-CAN adapter, or Jetson CAN pins
   with a transceiver? Record the part.
5. **Resolve the `/dev/ttyUSB0` collision** between LiDAR and motor interface.
6. Is there an IMU? `robot_localization` implies one.
7. Which Hilti charger is available?
8. Main power wiring gauge — must handle ~12.5 A peak.
9. Boot media type and capacity.
10. Camera intrinsic calibration not yet done.

---

## Shopping list

| Item | Priority | Note |
|---|---|---|
| 12 V ≥ 10 A synchronous buck, **15–40 V input** | **Urgent** | Blocks running the Jetson from the pack |
| 20 A blade fuse + inline holder | **Urgent** | Fit before any powered test |
| 5 A fuse + holder | High | DC-DC branch |
| TVS diode + bulk electrolytic cap | Medium | Regen protection on shared rail |
| Second B 22-195 pack | Low | Hot-swap for continuous testing |
