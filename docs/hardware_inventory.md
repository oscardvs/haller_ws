# Haller Hardware Inventory

Per-part status for everything physical on the robot: what is on hand, what is
verified, and what is still an open question.

Status key:

- **✅ Confirmed** — verified in a purchase record, on a datasheet, or measured
- **⚠️ Assumed** — inferred but not verified; check before relying on it
- **❓ Unknown** — needs someone to look at the robot and fill it in
- **❌ Missing** — not on hand, on the buy list

Last reconciled against purchase records **2026-07-27**.

### Which document says what

| Question | Document |
|---|---|
| What is on hand, and what is still unverified? | **this document** |
| What does it cost and what is left to buy? | [BOM](../website/content/docs/hardware/bom.mdx) |
| How does it all connect — fusing, grounding, E-STOP? | [`wiring.md`](wiring.md) |
| Why is the battery chain shaped this way? | [`power_system.md`](power_system.md) |

Where a number appears in more than one of these, `wiring.md` is the authority
for the power tree and fusing.

### Two build phases

The robot is currently **Phase A**: both arms on a static XLeRobot tower, no
mobile base. **Phase B** adds the base. Parts below are Phase A unless marked
**[Phase B]**.

---

## Compute

| Part | Qty | Details | Status |
|---|---|---|---|
| NVIDIA Jetson Orin Nano dev kit, 8 GB | 1 | 9–20 V in, 45 W budget, barrel 5.5 × **2.5** mm centre-positive. Power modes 15 W / 25 W / MAXN SUPER | ✅ |
| NVMe SSD, 512 GB, M.2 | 1 | Boot media | ✅ |
| USB-C supply ≥45 W | 1 | Bench bring-up before the pack is in the loop | ✅ |

Software baseline: Ubuntu 22.04 / JetPack, ROS 2 Humble.

The 8 GB variant is required — the vision pipeline (YOLOv8n + SegFormer-B0,
TensorRT FP16) does not fit in 4 GB.

---

## Arms

The two arms are **mechanically identical**: 6× C001-spec (1/345) STS3215 each,
not the stock leader gearbox mix. Symmetric on purpose — either arm can be
leader or follower, and swapping the end-effector flips the role. Both are
currently driven as **followers**, because teleoperation is the shipped
human-pose webcam path.

| Part | Qty | Details | Status |
|---|---|---|---|
| SO-ARM101 mechanical kit | 2 | FDM-printed parts, fasteners, bearings | ✅ |
| Feetech **STS3215, 7.4 V, 19 kg·cm**, C001 (1/345) | 12 | 6 per arm, 360° magnetic encoder. Operating range **6.0–7.4 V**, €17.83 each | ✅ |
| Waveshare Serial Bus Servo Driver Board | 2 | One per arm. USB ↔ TTL half-duplex, `1a86:55d3`, jumpers on **B**. Barrel jack **DC5521 = 5.5 × 2.1 mm** — *not* the Jetson's 2.5 mm | ✅ |
| 3-pin TTL daisy-chain cables | 10–12 | 5 per arm plus the upstream link | ✅ |
| Soft fin-ray gripper fingers, TPU 95A | 2 sets | Printed from the XLeRobot hardware repo | ✅ |
| XLeRobot tower + arm base plate | 1 | The cart, mecanum base and head gimbal from that BOM are not used | ✅ |
| Spare STS3215, 7.4 V | 0 | `shoulder_lift` carries the most load and is the one that fails | ❌ Missing |

> ⚠️ **These are the 7.4 V / 19 kg·cm servos, not the 12 V / 30 kg·cm variant**
> most SO-101 build logs assume. Torque is about a third lower, so gravity sag
> is more visible and payload is lower — keep anything bolted to the wrist under
> ~40 g. More importantly, **12 V destroys all twelve**. The over-voltage
> protection on the 7.4 V rail is not optional; see [`wiring.md` §5](wiring.md).

---

## Perception

| Part | Qty | Details | Status |
|---|---|---|---|
| Intel RealSense **D455** | 1 | Third-person workspace camera, rigid mount on the tower. 87° × 58° FOV covers the bimanual workspace. Build librealsense with `-DFORCE_RSUSB_BACKEND=ON` on JetPack | ✅ |
| D455 fixed mount | 1 | `Gimbal_Pitch_Holder` from the XLeRobot repo, bolted rigid — no gimbal, deliberately | ✅ |
| Slamtec RPLIDAR A1M8 | 1 | Serial, `/dev/haller_lidar` @ 115200, scan mode `Sensitivity`, frame `laser_frame`. USB-powered 5 V | ✅ |
| IMX219 camera, module **J8IM228-V3.0** | 1 | CSI, `/dev/video0`, Bayer RGGB 1280×720 @ 30 fps via `gscam`. Wide-angle lens with noted barrel distortion | ✅ |
| Jetson CSI FPC, 22-pin 120° | 1 | IMX219 ribbon | ✅ |
| Seeed **XIAO nRF52840 Sense** | 2 | Onboard 6-axis IMU — closes the `robot_localization` IMU gap. Also carries the ADC for the `BatteryState` publisher. **On hand, not yet wired or publishing** | ✅ |
| Wrist / egocentric USB camera, **MJPEG** 640×480@30 | 0 | One per gripper, ~30 × 30 mm, under 40 g. MJPEG support is mandatory — see the USB bandwidth note below | ❌ Missing |

The IMX219 is not yet intrinsically calibrated (`camera_info_url` is empty in
`haller_vision/config/camera/imx219_hardware.yaml`).

---

## Mobile base **[Phase B]**

Drive layout: **differential drive, 2 driven front wheels + 1 rear caster**
(3 wheels total).

| Part | Qty | Details | Status |
|---|---|---|---|
| LK-TECH / MyActuator **MF5010** BLDC | 2 | One per driven front wheel. 16 V rated winding, CAN @ 1 Mbps. Variant **10T vs 35T not recorded** — differs 4× in current, 3.5× in speed | ⚠️ |
| Motor controller | ? | The BOM records the MF5010s as integrated-controller units, but the repo also ships MCF302CB demo material (`docs/LK-demo-MCF302CB-CAN.zip`). **Confirm which is physically present**, and its input voltage range | ❓ |
| **CANable V2** USB-CAN FD adapter | 1 | → `/dev/haller_can`. Closes the long-standing "is there a CAN adapter?" gap | ✅ |
| 22 AWG PTFE twisted pair | 10 m | CAN bus wiring, black/white | ✅ |
| CAN termination resistor, 120 Ω | 1 | Check whether the MF5010s ship with one built in before fitting a second | ❓ |
| Gearbox / reduction | 0 | **None — direct drive.** Motor shaft ↔ wheel 1:1 | ✅ |
| Driven wheels | 2 | Radius 0.05 m (⌀100 mm), track width 0.34 m | ✅ |
| Swivel casters, 3 inch | 4 | Only 1 is needed for the 3-wheel layout; the rest are spares | ✅ |

### MF5010 variants (from `docs/MF5010_Specs.pdf`)

| Parameter | 10T | 35T |
|---|---|---|
| Rated current | 5.06 A | 1.35 A |
| Max power | 128 W | 12 W |
| Max speed | 3050 rpm | 870 rpm |
| Speed constant | 150 rpm/V | 27.5 rpm/V |

### Drive is direct (no reduction)

Confirmed: the motors drive the wheels **1:1, no gearbox**. Consequences:

- **Speed.** The 1.0 m/s velocity cap with a 0.05 m wheel is 20 rad/s ≈
  **191 rpm** at the motor — about 6% of the 10T's 3050 rpm rating. Plenty of
  speed headroom; the robot is nowhere near the motor's top end.
- **Torque = wheel torque, 1:1.** No multiplication. Force per wheel is
  τ / r = 0.26 / 0.05 ≈ **5.2 N rated**, ~8 N at peak torque (0.4 N·m).
- **Current is torque-driven, not speed-driven** (I ≈ τ / Kt, Kt = 0.05 N·m/A
  for the 10T). Running slow does not lower current. Under sustained load
  (heavy robot, ramp) each motor can sit near its **rated 5 A continuously**;
  max torque draws ~8 A. The Phase B fusing in `wiring.md` §3 covers this.

Open sub-item: **total robot mass is not recorded**, so grade/acceleration
capability isn't yet quantified. As a reference point, on a ~15 kg robot two
wheels give ~0.7 m/s² acceleration and ~4° grade at rated torque. Confirm this
is adequate for the intended environment.

---

## Power

Architecture, fusing and grounding live in [`wiring.md`](wiring.md); the battery
envelope derivation lives in [`power_system.md`](power_system.md).

| Part | Qty | Details | Status |
|---|---|---|---|
| Hilti **B 22-195** Nuron battery | 1 | 6S3P, 21.6 V nominal, 9.0 Ah, 194.4 Wh, 1.33 kg. Measured 24.3 V at near-full, no tool handshake needed | ✅ |
| Hilti sliding battery connector | 1 | 6 wires: 2× B+, 2× B−, white + blue data (leave unconnected). **Bond each power pair** | ✅ |
| Hilti Nuron charger | 1 | On hand; exact model (C 4-22 / C 6-22 / C 8-22) not recorded | ⚠️ |
| **6S LiPo 5200 mAh 80C**, XT60 + 6S 40 A BMS | 1 | Alternate pack, 115 Wh. Same 20.4–25.2 V envelope, so the converter design is pack-agnostic. Hilti stays primary | ✅ |
| **LM2596** buck module | 5 | ~2 A real continuous. **Not suitable for the Jetson or the arms** — bench/auxiliary use only, under 1.5 A | ✅ |
| Adjustable buck, **8–40 V in → 7.4 V**, ≥10 A, CC+CV | 0 | The arm rail. CC limit set to 10 A. 300 W class | ❌ Missing |
| Sealed buck, **15–40 V in → 12 V**, ≥10 A, synchronous | 0 | The Jetson rail. Verify the *input range*, not the "24V" in the title | ❌ Missing |
| **TVS diode, 8.0 V standoff** (SMBJ8.0A / P6KE8.2A) | 0 | Crowbar on the 7.4 V rail | ❌ Missing |
| Inline fuse holders + ATO fuses | 0 | Phase A set: 15 A main, 10 A arm branch, 10 A fast-blow arm output, 2× 6 A per-arm, 5 A Jetson | ❌ Missing |
| Main disconnect switch, ≥30 A | 0 | Between pack B+ and the main fuse | ❌ Missing |
| XT60 pigtail for the Hilti harness | 0 | Makes the Hilti pack and the LiPo interchangeable | ❌ Missing |
| **14 AWG silicone wire** | 0 | 2 m. The only gauge not on hand — the assorted hook-up wire is 20–22 AWG, signal only | ❌ Missing |
| TVS 30 V + 2200 µF bulk, motor rail | 0 | BLDC regen suppression. Arm servos do not regen — **[Phase B]** only | ⏸️ Deferred |
| ATO fuses, 25 A main + 20 A motor branch | 0 | **[Phase B]** uprate | ⏸️ Deferred |

### Safety and monitoring

| Part | Qty | Details | Status |
|---|---|---|---|
| **LA36M E-STOP**, 22 mm mushroom, 1NC | 1 | AC/DC 12–24 V, no light. Breaks the **relay coil**, not the arm current — see [`wiring.md` §6](wiring.md) | ✅ |
| 12 V 1-channel relay module, low-level trigger | 3 | The E-STOP contactor. Contacts ~10 A / 30 VDC **[E]** — confirm from the datasheet | ⚠️ |
| Digital battery capacity indicator, 8–100 V DC | 1 | Panel meter, two wires across the pack. Works on either battery | ✅ |
| 2–6S LiPo cell voltage monitor / alarm | 1 | Needs balance taps — **works only with the LiPo**. The Hilti pack exposes no cell taps | ✅ |

---

## Interconnect

| Part | Qty | Details | Status |
|---|---|---|---|
| Wago-style lever connectors, 2/3/5 port | 90 | Star-ground node and rail distribution | ✅ |
| DC barrel pigtails, 18 AWG / 7 A | ~6 | **5.5 × 2.1 confirmed.** A second lot lists 2.1 in the title and 2.5 in the variant — *measure before plugging anything into the Jetson* | ⚠️ |
| Heat-shrink kit, 328 pc | 1 | Harness | ✅ |
| Hook-up wire, assorted | — | 20–22 AWG **[E]** — signal only | ⚠️ |
| Bolt assortment M2–M4, heat-set inserts M2–M6 | — | Mechanical | ✅ |
| 1N4007 diodes | 50 | 1 A / 1000 V — **signal-level only**, not a TVS substitute and not for power rails | ✅ |
| Powered USB 3.0 hub, 4–7 port, **12 V DC input** | 0 | The DC jack lets it run off the 12 V rail instead of a wall wart | ❌ Missing |
| Shielded USB cables ≤1.5 m + ferrite clamps | 0 | 5 clamps | ❌ Missing |

### USB topology

Eight devices eventually, against 4× Type-A on the dev kit (the USB-C port is
occupied by device-mode SSH). Full tree in [`wiring.md` §8](wiring.md).

- **The D455 gets its own root port** — RealSense devices fail intermittently
  behind hubs.
- **Each wrist camera gets its own root port.** Two 640×480@30 streams in raw
  YUYV need ~147 Mbps each against USB 2.0's ~280 Mbps practical ceiling, hence
  the MJPEG requirement.
- Serial devices (both arm boards, CAN, LiDAR) share the powered hub — all are
  kilobit-class.

### Device naming

`scripts/99-haller-devices.rules` assigns stable symlinks by USB serial:
`/dev/haller_can`, `/dev/haller_lidar`, `/dev/haller_arm_follower` (right),
`/dev/haller_arm_leader` (left). Both Waveshare boards enumerate as `1a86:55d3`,
so **serial keying is mandatory**.

> `haller_arm_leader` is the **left** arm. The name records the role it was
> built and first calibrated for (2026-05-22, leader-follower teleop), not what
> it does now — today the HMI drives it as a follower, copying its teleoperator
> calibration into `robots/so_follower/` on startup. Don't rename it: the udev
> rules, the HMI config and the calibration IDs all agree on it, and the arms
> are symmetric enough that the roles can swap again at any time.

> ⚠️ **Stale serial default in the ros2_control description.** The LiDAR has
> moved to `/dev/haller_lidar`, so the old `/dev/ttyUSB0` collision no longer
> exists. But `haller_description/urdf/haller_ros2_control.xacro` still passes
> `serial_port: /dev/ttyUSB0`, and the C++ default in
> `haller_hardware_interface.cpp` matches it. The motors are **CAN**, via
> `/dev/haller_can` — that interface is still a simulated stub, and the serial
> parameter will need to go when it is made real.

---

## Open questions

Ordered roughly by how much they block progress.

1. **Measure the second lot of DC barrel pigtails** — 5.5 × 2.1 or 2.5? The
   Jetson needs 2.5; a 2.1 plug fits loosely and arcs under load.
2. **Does the Waveshare board pass barrel-jack voltage straight through** to the
   servo bus (expected) rather than regulating it? Measure barrel V+ to servo
   connector V+ before trusting the 7.4 V design.
3. **Confirm the relay module's DC contact rating** from its datasheet — 10 A /
   30 VDC is currently assumed.
4. Record the actual insulation colours of the Hilti B+ / B− wires.
5. **[Phase B]** MF5010 winding: **10T or 35T?** Sets the whole current budget.
6. **[Phase B]** Which motor controller is physically fitted, and what is its
   input voltage range?
7. **[Phase B]** Confirm the driver enforces a speed limit at a 25.2 V rail
   against the motor's 16 V winding rating.
8. **Record total robot mass** — needed to quantify direct-drive grade and
   acceleration capability.
9. Which Hilti charger model is on hand?
10. IMX219 intrinsic calibration not yet done.

### Closed by the 2026-07-27 purchase-record audit

- ~~Is there a USB-CAN adapter?~~ CANable V2, on hand.
- ~~What is the boot media?~~ 512 GB NVMe SSD.
- ~~Is there an IMU?~~ XIAO nRF52840 Sense ×2 (not yet wired).
- ~~Main power wiring gauge?~~ 14 AWG for every high-current run; drop budget
  derived in [`wiring.md` §4](wiring.md).
- ~~`/dev/ttyUSB0` collision between LiDAR and motors?~~ The LiDAR is on
  `/dev/haller_lidar`. A stale serial default remains in the xacro — see above.

---

## Shopping list

Prices are estimates in EUR, ex-shipping. The
[BOM](../website/content/docs/hardware/bom.mdx) carries the full costed list and
the sourcing notes; this is the summary.

| Item | ~€ | Priority |
|---|---|---|
| Adjustable buck 8–40 V → 7.4 V, ≥10 A, CC+CV | 18 | **Blocking** — no arm rail without it |
| Sealed buck 15–40 V → 12 V, ≥10 A, synchronous | 18 | **Blocking** — no Jetson on the pack without it |
| TVS 8.0 V (SMBJ8.0A) | 2 | **Blocking** — the cheapest insurance against €215 of servos |
| Fuse holders + ATO fuses (15 / 10 / 6 / 6 / 5 A) | 12 | **Blocking** — fit before any powered test |
| 14 AWG silicone wire, 2 m | 10 | **Blocking** |
| XT60 pigtail + main disconnect ≥30 A | 10 | **Blocking** |
| **Blocking subtotal** | **~70** | |
| Wrist camera modules, MJPEG, ×2 | 55 | Multi-camera datasets |
| Powered USB hub, 12 V input | 25 | Multi-camera datasets |
| Shielded USB cables + ferrites | 20 | Multi-camera datasets |
| Spare STS3215 7.4 V | 18 | Insurance |
| **Total** | **~188** | |

Single-camera dataset recording needs **none** of the €118 below the blocking
line — the D455 on its rigid mount is a complete observation on its own.

**[Phase B]** adds roughly €12 more: the 25 A / 20 A ATO fuses and the motor
rail's TVS + bulk capacitor. Defer until the base is integrated.
