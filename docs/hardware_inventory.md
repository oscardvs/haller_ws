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
| Spare STS3215, 7.4 V, **C001 (1:345)** | 0 | `shoulder_lift` carries the most load and is the one that fails. **€26.15** — [AliExpress 1005012394954023](https://de.aliexpress.com/item/1005012394954023.html), select `Color: Gear Ratio 1-345`. The ratio is a paid option: 1:147 is €26.41 and the cheap €13 listings elsewhere are the 12 V / low-ratio variants, **not** C001 | ❌ Missing |

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
| **ELP 0.3 MP USB camera**, 32×32 mm UVC, MJPEG 640×480@60 | 0 | One per gripper. **€23.51 each** (+€6.61 shipping) — [AliExpress 1005011810535562](https://de.aliexpress.com/item/1005011810535562.html), official ELP store, 5.0★/338 sold. Select lens **L170 (FOV 142°)**. Native VGA, so no downscaling; MJPEG at 60 fps is 2× the required rate. **Confirm the board is the 32×32 and not the 26×26 variant** — the title offers both and the page has no size selector | ❌ Missing |

The IMX219 is not yet intrinsically calibrated (`camera_info_url` is empty in
`haller_vision/config/camera/imx219_hardware.yaml`).

### Wrist camera: the mount fixes the size

The printed adaptor is
[`SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module`](https://github.com/TheRobotStudio/SO-ARM100/tree/main/Optional/SO101_Wrist_Cam_Hex-Nut_Mount_32x32_UVC_Module)
from the SO-ARM100 repo (`SO-ARM101_camera_wrist_mount.stl`, 40 % infill, tree
supports). That pins the camera spec — this is no longer a "~30 × 30 mm"
preference:

| Constraint | Value | Source |
|---|---|---|
| Module footprint | **32 × 32 mm** | upstream README |
| Camera → adaptor | **4 × M2** screws | upstream README |
| Adaptor → wrist | **2 × M3 × 8 mm** + **2 × M3 hex nuts** | upstream README |
| Resolution / rate | 640 × 480 @ 30 fps | upstream README, and §USB topology below |
| Focus | manual — twist the lens to focus | upstream README |
| Mass | < 40 g | 7.4 V servo torque limit |

Upstream recommends an Innomaker module; the part selected here is the
equivalent **ELP 0.3 MP**, bought from ELP's own AliExpress store. Buy branded —
generic "32 × 32" listings routinely omit board dimensions and format tables, and
the mount is already printed to a fixed size, so a module that turns out to be
26 × 26 or 38 × 38 mm is scrap. The M2/M3 hardware is covered by the bolt
assortment already on hand.

Why this one over a higher-resolution module:

- **Native 640 × 480.** The sensor is 0.3 MP, so the required resolution is the
  sensor's own — nothing is downscaled, and there is no temptation to record at
  1080p and blow the USB budget.
- **MJPEG at 60 fps**, double the 30 fps the mount's README and the dataset
  pipeline ask for. The bandwidth trap in §USB topology is closed by
  construction rather than by remembering to force a pixel format.
- **142° lens (option L170).** A wrist camera sits ~10 cm from the gripper; the
  default L36 (60°) would frame little but fingers. Take L21 (72°) instead only
  if barrel distortion turns out to hurt the policy — nothing is calibrated yet
  either way.
- **Micro-USB on the board**, so the wrist lead is thin, flexible and
  replaceable, which matters on a joint calibrated as continuous. Budget two
  micro-USB→A cables in the USB-cable line.

> ⚠️ **Confirm the board size with the seller before ordering.** The listing
> title reads "32x32/26x26mm" but the only variant selector is the lens — there
> is no size option. 26 × 26 does not fit the printed mount.

### Why USB and not another CSI camera

Not a preference — the board can't do it:

- **CSI slot 0 is physically broken.** The connector's retaining clip is snapped,
  so the ribbon won't seat hard enough to carry the MIPI lanes. It probes fine on
  i²c and returns zero frames on every capture. `hmi/backend/config.yaml` has
  `wrist_left` as `source: placeholder` for exactly this reason.
- That leaves **one** working CSI slot for **two** wrist cameras.
- CSI ribbon through the wrist is the wrong cable anyway. `wrist_roll` is
  calibrated as continuous; [`wiring.md` §8](wiring.md) already warns that a USB
  cable routed through it will wind up and snap, and an FPC ribbon tolerates
  twisting far worse than a round cable.

> ⚠️ **These docs disagree with the hardware.** `hmi/HANDOVER-2026-08-01-jetson-rig.md`
> records an **IMX219 currently on the wrist** (CSI slot 1, `wrist_right`,
> 1280×720@60, `flip_method: 3` — itself flagged UNVERIFIED in `config.yaml`).
> The Perception table above still lists the IMX219 as the *base navigation*
> camera on `/dev/video0`. One of the two is stale. Reconcile before ordering,
> because it changes whether the base camera still exists.

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
| **XY6020L** digital buck, 6–70 V in → 0–60 V, 20 A / 1200 W, CC+CV | 0 | The arm rail, set to **7.40 V / 10.0 A CC**. Setpoints are keyed in and stored — **no trimpot**, which retires two of the four [`wiring.md` §5](wiring.md) mitigations. Take the base-plate/cased variant, not the bare board. €24.83 — [AliExpress 1005008144889470](https://de.aliexpress.com/item/1005008144889470.html) | ❌ Missing |
| **RCNUN sealed buck-boost**, 8–40 V in → 12 V, 10 A, IP67 | 0 | The Jetson rail. Select `Color: 10A` / `8-40V` / `12V`. Meets every row of the [`power_system.md` §3](power_system.md) table including the ≥40 V absolute max. €20.79 — [AliExpress 32897068247](https://de.aliexpress.com/item/32897068247.html) | ❌ Missing |
| **TVS diode, 8.0 V standoff** — SMBJ8.0A (SMD) or **P6KE9.1A** (DO-15 axial) | 0 | Crowbar on the 7.4 V rail. ~€2/20 pcs — [P6KE, incl. 9.1A](https://de.aliexpress.com/item/1005006630139887.html) · see the **corrected part number** note below | ❌ Missing |
| Inline fuse holders, 10–18 AWG, ATO/ATC | 6 needed | ~€1.04–2.00 each — [10-pack, 227 sold](https://nl.aliexpress.com/w/wholesale-inline-ATO-blade-fuse-holder-waterproof-14AWG.html) | ❌ Missing |
| ATO blade fuses | 1 set | Phase A: 15 A main, 10 A arm branch, 10 A arm output, 2× **7.5 A** per-arm, 5 A Jetson. **Not 6 A** — see below | ❌ Missing |
| Main disconnect switch, ≥30 A | 0 | Between pack B+ and the main fuse. "Car Battery Cut Off Switch", €3.82, 4.6★/10 000+ sold — [search](https://nl.aliexpress.com/w/wholesale-battery-isolator-switch.html). **Check the form factor**: the cheap ones clamp onto a battery *post* on one side. You need a **two-stud** variant to land ring terminals on 14 AWG | ❌ Missing |
| XT60 pigtail for the Hilti harness | 0 | Makes the Hilti pack and the LiPo interchangeable. "2pcs XT60 Female/Male Plug Battery Connector 14AWG 10cm with Silicone Flexible Wire", **€2.00**, 5.0★/800+ sold — [search](https://nl.aliexpress.com/w/wholesale-xt60-connector-pigtail-silicone-wire.html). Battery side is male by convention, so the harness gets the female | ❌ Missing |
| **14 AWG silicone wire** | 0 | The only gauge not on hand — the assorted hook-up wire is 20–22 AWG, signal only. "2 meter Silicon Wire 8–22 AWG (1 m Red + 1 m Black)", **€4.18**, 4.8★/10 000+ sold — [search](https://nl.aliexpress.com/w/wholesale-14awg-silicone-wire.html). **Select 14 AWG and buy 2** (€7.94): the §4 runs need ~1.5 m red and ~1 m black, so a single 1 m + 1 m pack is short | ❌ Missing |
| TVS 30 V + 2200 µF bulk, motor rail | 0 | BLDC regen suppression. Arm servos do not regen — **[Phase B]** only | ⏸️ Deferred |
| ATO fuses, 25 A main + 20 A motor branch | 0 | **[Phase B]** uprate | ⏸️ Deferred |

### Corrected: the TVS alternate part number was wrong

Earlier revisions of this table and the BOM offered **P6KE8.2A** as a drop-in
alternative to the SMBJ8.0A. **It is not one, and it would have conducted
continuously on a healthy rail.**

The two series number their parts differently:

| Series | The number in the part name is… | Stand-off (V<sub>RWM</sub>) | Breakdown (V<sub>BR</sub>) |
|---|---|---|---|
| SMBJ**8.0**A | the **stand-off** voltage | **8.0 V** | 8.89–9.83 V |
| P6KE**8.2**A | the **breakdown** voltage | **≈7.02 V** ❌ | 7.79–8.61 V |
| P6KE**9.1**A | the **breakdown** voltage | **≈7.78 V** ✅ | 8.65–9.55 V |

A P6KE8.2A's 7.02 V stand-off sits **below** the 7.4 V rail, so it would be
biased into leakage the moment the rail came up — and its 7.79 V minimum
breakdown is only 0.39 V above nominal, close enough to nuisance-clear the
10 A fast-blow fuse under normal ripple. The through-hole part that actually
matches SMBJ8.0A's intent is **P6KE9.1A**.

Prefer the axial DO-15 P6KE9.1A over the SMD SMBJ8.0A for this build — the
harness is wire-and-WAGO, and a DO-214AA surface-mount part has to be soldered
to pigtails or a scrap of protoboard before it can join a 14 AWG rail.

> Confirm V<sub>RWM</sub> against the specific manufacturer's datasheet before
> fitting; the series conventions above are consistent across Vishay and
> Littelfuse, but the exact volts move a little between vendors.

### Corrected: 6 A is not a standard ATO value

[`wiring.md` §3](wiring.md) specifies a **6 A** fuse on each per-arm branch. No
such ATO/ATC blade exists — the standard series runs 1, 2, 3, 4, 5, **7.5**, 10,
15, 20, 25, 30, 35, 40 A.

Use **7.5 A**. 5 A sits exactly at each arm's ~5 A peak and will nuisance-blow
mid-episode.

> ⚠️ **Open decision.** 7.5 A is slightly above the 7 A rating of the 18 AWG
> pigtails that feed the Waveshare boards ([`wiring.md` §4](wiring.md)), so the
> fuse no longer strictly protects the wire. In practice 18 AWG silicone in free
> air carries well over 7 A and the 7 A figure is a connector/vendor rating, not
> a conductor limit — but decide this deliberately rather than by default.

Also worth recording: **"fast blow" needs no special part.** ATO/ATC blade fuses
are fast-acting by construction; only time-delay versions are marked as such. A
standard 10 A blade satisfies the crowbar requirement in
[`wiring.md` §5](wiring.md).

### Converter sourcing notes

Surveyed 2026-08-01. Two traps worth recording, because both were nearly bought:

- **The €3–6 "300 W 20 A 6–40 V → 1.2–36 V CC CV" boards are mislabelled.** The
  most-upvoted review on the 3 000+-sold listing measures the delivered part at
  **Vin 5–30 V, 10 A max output** — not 6–40 V / 20 A. A 30 V ceiling is 4.8 V
  over the pack's full charge, and 10 A against a 10 A load is no margin. This
  is the [`power_system.md` §3](power_system.md) "verify the input range, not
  the title" trap, except the *output* rating lies too.
- **Fixed-output potted modules can't hit 7.4 V.** The common 12/24 V → 3.3–9 V
  sealed converters offer 6 V or 7.5 V and nothing between. 6 V is the servo
  floor; 7.5 V is 0.1 V over the datasheet max with no way to trim it down.

Budget alternatives, if the picks above are unavailable or the €45 is unwelcome:

| Rail | Alternative | € | What you give up |
|---|---|---|---|
| 7.4 V | ZK-SJ20, 7–80 V → 1.4–79 V, 20 A, CC+CV — [1005007979446715](https://de.aliexpress.com/item/1005007979446715.html) | 7.61 | Trimpots return, so §5 mitigations 1–2 apply as written. Only 6 reviews; one reports 2 W idle draw and a soft-start hiccup into a large capacitive load. |
| 12 V | WH 15–35 V → 12 V 10 A, die-cast 74×74×32 mm — [1005006630139887](https://de.aliexpress.com/item/1005006630139887.html) | 10.64 | 35 V ceiling instead of 40 V — a **[Phase B]** regen concern only, bounded by the raw-rail TVS. Pure buck, and it publishes the input window per variant. Only 12 reviews. |

> ⚠️ **Neither 12 V candidate states synchronous rectification.** The RCNUN's
> 90–97 % efficiency claim implies it; the WH publishes no efficiency figure. The
> [`power_system.md` §3](power_system.md) "synchronous" requirement is therefore
> **inferred, not confirmed** on both — measure the case temperature at full load.

> The 9–36 V → 12 V/19 V buck-boost that was in the cart is superseded: at 12 V /
> 10 A it lands at €16–20 with a 36 V ceiling — the WH's shortfall at nearly the
> RCNUN's price, and 117 sold against 1 000+.

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
| Powered USB 3.0 hub, 4–7 port, **12 V DC input** | 0 | The DC jack lets it run off the 12 V rail instead of a wall wart. "4/7/10 Ports USB 3.0 Hub, 12 V Power Adapter, On/Off Switch", **€19.04**, 4.6★/1 000+ sold — [search](https://nl.aliexpress.com/w/wholesale-powered-usb-3.0-hub-12v-dc.html). Take the **7-port**; measure the DC jack (2.1 vs 2.5 mm) and confirm polarity before feeding it from the rail | ❌ Missing |
| Ferrite clamps | 0 | "5 Pcs 3.5/5/7/9/13 mm Toroidal Core Ferrite Bead Clip Choke", **€3.49**, 4.8★/1 000+ sold — [search](https://nl.aliexpress.com/w/wholesale-ferrite-core-clamp-cable.html). The 3.5–13 mm spread covers USB leads *and* the hub's DC cord | ❌ Missing |
| Shielded USB cables ≤1.5 m | 0 | ~€10. Generic — 2× USB-A→C (Waveshare boards), 1× for the CANable, 1× for the LiDAR. Buy to the connectors you actually have rather than to a spec | ❌ Missing |

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
Quoted 2026-08-01 unless marked *est.*

| Item | € | Priority |
|---|---|---|
| [XY6020L digital buck](https://de.aliexpress.com/item/1005008144889470.html), 6–70 V → 7.4 V, 20 A, CC+CV | 24.83 | **Blocking** — no arm rail without it |
| [RCNUN sealed buck-boost](https://de.aliexpress.com/item/32897068247.html), 8–40 V → 12 V, 10 A | 20.79 | **Blocking** — no Jetson on the pack without it |
| TVS **P6KE9.1A**, 20 pcs axial | 2.02 | **Blocking** — the cheapest insurance against €215 of servos |
| Fuse holders ×10 + ATO fuses (15/10/10/7.5/7.5/5 A) | ~12 *est.* | **Blocking** — fit before any powered test |
| 14 AWG silicone wire, 2× (1 m red + 1 m black) | 7.94 | **Blocking** |
| XT60 pigtail pair | 2.00 | **Blocking** |
| Main disconnect, two-stud | 3.82 | **Blocking** |
| **Blocking subtotal** | **~73** | |
| Wrist camera modules, 32 × 32 mm UVC, ×2 | ~55 *est.* | Multi-camera datasets |
| Powered USB hub, 7-port, 12 V input | 19.04 | Multi-camera datasets |
| Ferrite clamps ×5 | 3.49 | Multi-camera datasets |
| Shielded USB cables ≤1.5 m | ~10 *est.* | Multi-camera datasets |
| [Spare STS3215 7.4 V C001](https://de.aliexpress.com/item/1005012394954023.html) | 26.15 | Insurance |
| **Total** | **~187** | |

Single-camera dataset recording needs **none** of the ~€114 below the blocking
line — the D455 on its rigid mount is a complete observation on its own.

The total barely moved from the original ~€188 estimate, but the composition
did: the converters and the spare servo came in **over** estimate, and the wire,
XT60, disconnect and hub came in **under**. Two lines shifted enough to be worth
naming — the **spare servo is €26.15, not €18**, because the C001 (1:345) gear
ratio is a paid option on every listing that offers it; and the **blocking
subtotal fell to €73** even after the converters went up.

**[Phase B]** adds roughly €12 more: the 25 A / 20 A ATO fuses and the motor
rail's TVS + bulk capacitor. Defer until the base is integrated.
