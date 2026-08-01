# Haller Wiring Scheme

Complete point-to-point wiring for the Haller robot: power tree, fusing,
grounding, E-STOP, battery monitoring, and the USB/data tree.

Companion documents:
- [`power_system.md`](power_system.md) — derives the battery envelope and converter requirements
- [`hardware_inventory.md`](hardware_inventory.md) — per-part status and open questions

Each fact is tagged by provenance:

- **[M]** measured on the bench
- **[D]** from a datasheet, product listing, or official page
- **[C]** calculated from [M] or [D]
- **[E]** estimate — not verified, do not design safety margins around it

---

## 0. Two build phases

The wiring differs between where the robot is now and where it is going. Build
Phase A first; it is a strict subset of Phase B.

| | Phase A — arms on the stand | Phase B — full robot |
|---|---|---|
| Loads | 2 arms, Jetson, cameras | + 2 MF5010 BLDC, LiDAR, CAN |
| Rails | 7.4 V, 12 V | + raw pack rail |
| Main fuse | 15 A | 25 A |
| Peak pack current | ~6.7 A [C] | ~19 A [C] |
| Regen risk | none | yes — needs TVS + bulk cap |

Everything below is Phase A unless marked **[Phase B]**.

---

## 1. Confirmed parts on hand

Verified against purchase records. These drive the design — nothing here needs buying.

| Part | Qty | Detail | Use |
|---|---|---|---|
| Hilti B 22-195 Nuron pack | 1 | 6S3P, 21.6 V nom, 9.0 Ah, 194.4 Wh. **24.3 V measured no-load [M]** | Primary pack |
| Hilti sliding connector | 1 | 2× B+, 2× B−, white + blue data | Pack interface |
| **6S LiPo 5200 mAh 80C, XT60** | 1 | 22.2 V nom, 25.2 V full, 115 Wh [D] | Alternate pack — see §2.3 |
| 6S 40 A BMS + balance | 1 | For the bare LiPo | LiPo protection |
| Feetech **STS3215 7.4 V, 19 kg·cm**, magnetic encoder | 12 | 6 per arm. Operating 6.0–7.4 V [D] | Arm joints |
| Waveshare Serial Bus Servo Driver Board | 2 | Barrel jack **DC5521 = 5.5 × 2.1 mm**, USB-C, `1a86:55d3` | One per arm |
| **LA36M E-STOP, 22 mm, 1NC** | 1 | AC/DC 12–24 V, no light | Emergency cut — §6 |
| 12 V 1-channel relay module, low-level trigger | 3 | Contacts ~10 A / 30 VDC [E] | E-STOP contactor — §6 |
| **Digital battery capacity indicator, 8–100 V DC** | 1 | Panel meter | Pack SoC — §7 |
| 2–6S LiPo cell voltage monitor / alarm | 1 | Needs a **balance connector** | LiPo only — §7 |
| **CANable V2 USB-CAN FD** | 1 | Resolves the "is there a CAN adapter?" gap | **[Phase B]** motors |
| 22 AWG PTFE **twisted pair**, 10 m | 1 | Black/white | **[Phase B]** CAN bus |
| Wago-style lever connectors, 2/3/5 port | 90 | | Distribution nodes |
| DC barrel pigtails, 18 AWG, 7 A rated | ~6 | **5.5 × 2.1 confirmed**; a second lot is listed 2.1 in the title and 2.5 in the variant — *measure before use* | Rail → boards |
| LM2596 adjustable buck | 5 | ~2 A real continuous | Aux only, **not** the arms |
| Heat-shrink kit, 328 pc | 1 | | Harness |
| Hook-up wire, assorted colours | — | Appears to be 20–22 AWG [E] | Signal only — see §4 |
| Bolt assortment M2–M4, heat-set inserts M2–M6 | — | | Mechanical |
| XIAO nRF52840 **Sense** | 2 | Onboard 6-axis IMU + ADC | IMU + battery telemetry — §7 |
| NVMe SSD 512 GB | 1 | Resolves "boot media unknown" | Jetson boot |
| Jetson CSI FPC, 22-pin 120° | 1 | | IMX219 |
| RPLIDAR A1M8 | 1 | | **[Phase B]** |
| 1N4007 diodes | 50 | 1 A / 1000 V — **signal-level only** | Not for power rails |

---

## 2. Power tree

### 2.1 Phase A — arms on the stand

```
 HILTI B 22-195  (20.4 – 25.2 V)
 ├── B+ ×2 ──┐
 │           │  BOND THE PAIR                            ┌─────────────────────────────────┐
 │           ├──► XT60 ─► SW ─► 15 A ATO ──┬─► 10 A ATO ►│ XY6020L  6-70 V → 7.40 V  20 A  │
 │           │            ▲     main fuse  │             │ CC 10.0 A · digital setpoint    │
 │           │       disconnect            │             └──────────────┬──────────────────┘
 │           │         ≥30 A               │                            │  7.4 V
 │           │                             │                     10 A fuse (FAST BLOW)
 │           │                             │                            │
 │           │                             │                 TVS 8.0 V ─┤ (P6KE9.1A, to GND)
 │           │                             │                            │
 │           │                             │                ┌───────────▼───────────┐
 │           │                             │                │ RELAY NO  (E-STOP)    │  ◄── §6
 │           │                             │                └───────────┬───────────┘
 │           │                             │                            │
 │           │                             │                 WAGO 3-port node "7V4"
 │           │                             │                   ├── 7.5 A ─► DC5521 ──► Waveshare R ──► arm R bus
 │           │                             │                   └── 7.5 A ─► DC5521 ──► Waveshare L ──► arm L bus
 │           │                             │
 │           │                             └───► 5 A ATO ──► SEALED BUCK 15-40 V → 12 V 10 A
 │           │                                                        │  12 V
 │           │                                               WAGO 3-port node "12V"
 │           │                                                 ├──► DC5525 ──► Jetson Orin Nano
 │           │                                                 ├──► powered USB hub (DC in)
 │           │                                                 └──► relay module VCC (via E-STOP NC) ── §6
 │           │
 │           └──► capacity indicator (+)                                    ── §7
 │
 ├── B− ×2 ──┬──► WAGO 5-port node "GND"  ◄── single star ground point
 │           │      ├── adj buck GND
 │           │      ├── sealed buck GND
 │           │      ├── capacity indicator (−)
 │           │      └── (all downstream returns)
 │
 └── white + blue (Nuron data) ── INDIVIDUALLY INSULATED, NOT CONNECTED
```

Two details in that tree are load-bearing and easy to get backwards:

- **The disconnect switch sits between the XT60 and the main fuse**, so the
  fuse is still the first thing protecting the harness and the switch is the
  first thing you can reach. It is an operational convenience for bench work,
  **not a lockout** — pull the pack before touching the harness.
- **The 10 A fast-blow fuse is upstream of the TVS tap, not downstream.** This
  is the whole crowbar mechanism: over-voltage makes the TVS conduct, and that
  fault current has to flow *through* the fuse to clear it. Put the TVS on the
  buck side of the fuse and it clamps into the converter's current limit
  instead, cooking itself while the servos stay connected to a rail that is too
  high. Order matters more than placement neatness here.

### 2.2 Phase B — additions for the mobile base

```
 15 A main fuse becomes 25 A ATO
    ├── 20 A ──► raw pack rail ──┬── TVS 30 V + 2200 µF/50 V bulk  ◄── regen suppression
    │                            ├── MF5010 motor A
    │                            └── MF5010 motor B
    ├── 10 A ──► 7.4 V arm buck   (unchanged)
    └──  5 A ──► 12 V Jetson buck (unchanged)
```

The bulk capacitor and TVS exist **only** because the BLDC drives push energy
back into the rail on deceleration. The arm servos do not regen; do not fit
these in Phase A.

### 2.3 Which pack

Both packs are 6S with an identical 20.4–25.2 V envelope, so **the converter
design is pack-agnostic** — fit an XT60 to the Hilti harness and the two become
interchangeable.

| | Hilti B 22-195 | 6S LiPo 5200 mAh |
|---|---|---|
| Energy | **194 Wh** | 115 Wh |
| BMS | built in | separate board, must be wired |
| Charger | Hilti Nuron charger | needs a balance charger |
| Burst current | ample | 416 A [D] — vastly more than needed |
| Cell taps exposed | **no** | yes (balance lead) |

**Use the Hilti as the robot pack.** It has 1.7× the energy, an integrated BMS,
a charger you already own, and no fire-risk storage regime. The LiPo's only
advantage is burst current, which nothing here needs.

Keep the LiPo for bench work where you want cell-level visibility — it is the
only one of the two whose balance leads let the 2–6S monitor function at all.

---

## 3. Fusing

| Location | Rating | Type | Protects |
|---|---|---|---|
| B+, immediately at the connector | 15 A (25 A in Phase B) | ATO blade | Everything. **Fit first.** |
| 7.4 V buck input branch | 10 A | ATO blade | Arm converter + its wiring |
| 7.4 V rail, between the buck and the TVS tap | 10 A | **fast blow** | Blows when the TVS conducts — §5 |
| Each arm branch | **7.5 A** | ATO blade | One arm's fault doesn't take out the other |
| 12 V buck input branch | 5 A | ATO blade | Jetson path |
| Raw motor rail **[Phase B]** | 20 A | ATO blade | Motor drives |

Rules:

- The main fuse is the **first** component after B+. Nothing downstream gets
  wired until it is in place.
- Never work on the harness with a pack connected. Physically remove it.
- The 7.4 V output fuse must be **fast blow**. Its job is to clear before the
  TVS overheats, not to protect wiring.

---

## 4. Wire gauge and voltage drop

Voltage drop is the binding constraint on the 7.4 V rail, not ampacity. The
servo floor is 6.0 V, so the entire chain has **1.4 V of headroom** [D]. Budget
under 0.4 V total.

| Run | Current | Gauge | Length | Drop [C] |
|---|---|---|---|---|
| Pack → main fuse → buck inputs | 15 A | **14 AWG** | ≤0.5 m | 0.08 V |
| 7.4 V buck → relay → WAGO node | 10 A | **14 AWG** | ≤0.5 m | 0.08 V |
| 7.4 V node → each Waveshare board | 5 A | **16–18 AWG** (the 18 AWG 7 A pigtails qualify) | ≤0.4 m | 0.17 V |
| 12 V → Jetson | 3.75 A | 18 AWG pigtail | ≤0.5 m | 0.08 V |
| 12 V → USB hub | ~1 A | 20–22 AWG | any | negligible |
| E-STOP coil loop | 70 mA | 22 AWG | any | negligible |
| CAN bus **[Phase B]** | signal | **22 AWG twisted pair** | ≤1 m | — |
| Ground return, star node → pack | 15 A | **14 AWG** | ≤0.5 m | included above |

Worst-case servo terminal voltage: 7.4 − 0.08 − 0.17 = **7.15 V** [C]. Inside
the 6.0–7.4 V window with margin.

> **The assorted hook-up wire is 20–22 AWG and is for signal only.** Do not use
> it for the 7.4 V rail. 14 AWG silicone is the one wire gauge still to buy.

**[Phase B]** the main run carries ~19 A rather than ~6.7 A. 14 AWG still covers
that over ≤0.5 m — chassis-wiring ampacity is ~30 A and the drop is 0.10 V — but
if the run ends up longer than planned, step the pack-to-fuse-to-buck trunk up
to 12 AWG. Nothing downstream of the converters changes.

---

## 5. Over-voltage protection on the 7.4 V rail

The gap between safe (7.4 V) and destructive (12 V) is 4.6 V, set by a trimpot
on a €18 module. Twelve servos at ~€18 each is €215 of downside.

Four independent mitigations. Fit all of them:

1. **Verify into a dummy load.** Set the output with no servos connected, load
   it with a 12 V car bulb or power resistor, and confirm 7.4 V ±0.1 V on a
   multimeter across an input sweep from 25 V down to 19 V.
2. **Lock the trimpot** with nail varnish or thread-lock once verified.
3. **Crowbar.** SMBJ8.0A TVS from the 7.4 V rail to ground, downstream of a
   10 A **fast-blow** fuse. If the output rises, the TVS conducts hard and
   clears the fuse before the servos see the excursion.
4. **CC limit at 10 A** on the converter, so a bus fault current-limits rather
   than running away.

> The 1N4007 diodes on hand are **1 A parts** — they are not a substitute for
> the TVS and must not be placed in any power rail.

### If the converter is an XY6020L

The part selected in
[`hardware_inventory.md`](hardware_inventory.md) is digitally set rather than
trimpot-set. That changes three of the four mitigations above:

- **1 (verify)** becomes continuous — the onboard display reads out V and A while
  the rail is live, instead of a one-off multimeter check into a dummy load. Do
  the dummy-load sweep from 25 V down to 19 V anyway; it proves regulation, not
  just the setpoint.
- **2 (lock the pot)** is retired. There is no pot. This is the whole reason to
  prefer a digital module here — §5 exists because of a trimpot, and this
  deletes it.
- **4 (CC limit)** is keyed in as `10.0 A` and stored, rather than dialled in
  against a load.

**3 does not change.** The crowbar is the only mitigation that survives the
converter itself failing — a shorted high-side FET puts raw pack voltage on the
rail no matter what the display claims. Fit the TVS and the fast-blow fuse
exactly as specified above, in that order.

One new failure mode to close out at bring-up: confirm the module's
**output-on-at-power-up** behaviour. If it powers up with the output disabled,
the arms will not come back after an E-STOP cycle until someone presses a
button on the converter — which is a worse failure than the one it prevents.

---

## 6. E-STOP

Three independent layers, weakest-to-strongest:

| Layer | Mechanism | Effect |
|---|---|---|
| L0 — software | HMI `/estop` route | Torque disabled over serial; arms hold briefly then sag |
| L1 — dead-man | Release the clutch in human-pose teleop — SPACE up, or mouth closed, whichever source is armed for the session | Goals stop flowing within one 16 ms tick; arms freeze in place |
| L2 — hard | LA36M mushroom → relay → 7.4 V rail | Arm power cut; arms go limp |

### L2 wiring — failsafe, using the parts on hand

The E-STOP contact **must not carry the 10 A arm current**. A 22 mm contact
block breaking 10 A DC will pit and eventually weld. Instead the button breaks
the relay's *coil supply* — 70 mA — and the relay contacts do the power switching.

```
  12 V rail ──► [ E-STOP  1NC contact ] ──► relay module  VCC
                                            relay module  GND ──► star ground
                                            relay module  IN  ──► star ground
                                                                  (low-level trigger:
                                                                   permanently commanded ON)

  7.4 V from buck ──► relay  COM
                      relay  NO  ──► WAGO "7V4" node ──► both arms
```

Why this arrangement:

- **Failsafe by construction.** Pressing the button, a broken coil wire, a
  pulled connector, or loss of the 12 V rail all de-energise the coil, which
  opens the **NO** contact and kills the arms. There is no failure that leaves
  the arms live.
- **The contacts switch 7.4 V, not 24 V.** DC arcs sustain far more readily at
  24 V than at 7.4 V, so switching the low-voltage side is the gentler choice
  for the relay.
- **The Jetson stays powered**, so the HMI, the recording process, and your SSH
  session all survive the press.

> **Cutting arm power makes both arms go limp and drop.** With 7.4 V / 19 kg·cm
> servos the drop is gentler than the 12 V variant, but a fully extended arm
> still falls. Keep foam under the workspace. L2 is a last resort — L0 and L1
> are the normal stops.

---

## 7. Battery and IMU telemetry

### Capacity indicator — works on either pack

Two wires across the pack, upstream of the main fuse is fine (it draws µA).
Range is 8–100 V DC [D], pack is 20.4–25.2 V. Mount where you can see it.

### Cell monitor — LiPo only

The 2–6S monitor needs per-cell balance taps. **The Hilti pack does not expose
them** — its connector is B+, B−, and the Nuron data pair only. This part is
usable exclusively with the 6S LiPo.

### `BatteryState` publisher — reuse a XIAO nRF52840 Sense

`power_system.md` §7 has an open item for a ROS `BatteryState` publisher. One of
the XIAO nRF52840 Sense boards covers it, and its onboard 6-axis IMU
simultaneously resolves the "is there an IMU?" inventory gap that
`robot_localization` implies.

```
  Pack B+ ──► 100 kΩ ──┬──► XIAO A0
                       │
                      15 kΩ        100 nF to GND (anti-alias)
                       │
  Pack B− ────────────┴──────────► XIAO GND ──► star ground
```

Divider output at 25.2 V full charge: 25.2 × 15/115 = **3.29 V** [C] — inside
the nRF52840's ADC range with the internal reference. Publish over USB serial.

Thresholds from `power_system.md` §6: warn 21.6 V, critical 20.4 V, hard-stop
19.2 V. **Low-pass the reading** — under load the pack sags several hundred
millivolts and an unfiltered signal trips spuriously during acceleration.

---

## 8. USB and data tree

The Orin Nano dev kit has 4× Type-A. The USB-C port is occupied by device-mode
SSH and is not available for peripherals.

```
 Jetson Type-A #1 ──────────────► RealSense D455        (alone — never behind a hub)
 Jetson Type-A #2 ──────────────► wrist camera RIGHT    [when it arrives]
 Jetson Type-A #3 ──────────────► wrist camera LEFT     [when it arrives]
 Jetson Type-A #4 ──► POWERED HUB (12 V DC in from the 12 V node)
                       ├──► Waveshare board R   → /dev/haller_arm_follower
                       ├──► Waveshare board L   → /dev/haller_arm_leader
                       ├──► CANable V2          → /dev/haller_can     [Phase B]
                       └──► RPLIDAR A1M8        → /dev/haller_lidar   [Phase B]
 Jetson CSI ────────────────────► IMX219 via 22-pin FPC              [Phase B]
```

Rules:

- **The D455 never goes behind a hub.** RealSense devices fail intermittently on
  hubs; this is the single most common RealSense complaint.
- **Cameras get separate root ports.** Two 640×480@30 streams in raw YUYV need
  ~147 Mbps each against USB 2.0's ~280 Mbps practical ceiling [C]. Buy wrist
  cameras with **MJPEG** and force MJPEG in the capture config — roughly 10×
  less bandwidth.
- **Serial devices can share the hub.** All four are kilobit-class.
- **`wrist_roll` is calibrated as continuous.** A USB cable routed through it
  winds up and snaps. Clamp the range in software or leave a service loop with
  strain relief at the wrist.
- Cables ≤1.5 m, shielded, ferrite at the Jetson end.
- Device naming is by udev serial — see `scripts/99-haller-devices.rules`. Both
  Waveshare boards enumerate as `1a86:55d3`, so **serial keying is mandatory**;
  if a board reports no serial, key on the physical port path (`KERNELS=="1-2.3"`).

---

## 9. Grounding and EMC

- **Single star ground.** One WAGO 5-port node bonded to pack B−. Every
  converter, sensor, and the Jetson return lands there. No daisy-chained grounds.
- **Do not let servo return current flow through USB.** The Waveshare board ties
  servo ground to USB ground, so if the board's power return is high-impedance
  the current finds its way home through the USB cable shield — which presents
  as random bus dropouts and dead motors. Give each board a proper 16–18 AWG
  return to the star node.
- **Separate power and data runs.** The servo bus is half-duplex 1 Mbaud sitting
  next to a switching converter. Don't zip-tie USB and 7.4 V power together for
  long parallel stretches; cross at right angles where they must meet.
- **[Phase B]** CAN uses the 22 AWG twisted pair with 120 Ω termination at the
  far end. Check whether the MF5010s ship with a built-in terminator before
  fitting a second one.

---

## 10. Bring-up procedure

Do these in order. Do not skip ahead — each step protects the next.

1. **Harness, no pack.** Build the whole tree with the pack physically removed.
   Fit the disconnect switch and the 15 A main fuse at B+, in that order. Bond
   the B+ pair and the B− pair. Insulate the white and blue Nuron wires
   individually.
2. **Continuity check.** Multimeter across every fuse, and B+ to B− looking for
   a short. Confirm the star ground reaches every intended point.
3. **Polarity check.** Connect the pack. Measure at the XT60, then at each
   converter input. Confirm ~24 V and correct polarity **before** any converter
   is connected.
4. **Set the 7.4 V rail into a dummy load.** No servos, no boards. Load with a
   12 V bulb. Set the pot to 7.4 V. Sweep the input from 25 V down to 19 V and
   confirm the output holds. **Lock the pot.**
5. **Fit the TVS and the fast-blow fuse.** Re-measure. Still 7.4 V.
6. **Test the E-STOP with the dummy load.** Press: output must drop to 0 V.
   Release and reset: output returns. Then pull the relay's coil wire and
   confirm the output drops the same way — that verifies the failsafe.
7. **One arm, one board.** Connect a single Waveshare board and one arm. Measure
   the voltage **at the servo connector**, not at the buck — it must read
   ≥7.0 V under motion.
8. **Second arm.** Repeat. Confirm both arms hold ≥7.0 V with both moving.
9. **Bench the Jetson separately** on its own converter into the 12 V rail
   before joining the two rails to a common pack.
10. **Full system.** Everything on the pack. Run a teleop session and watch for
    voltage sag, bus dropouts, or thermal rise on either converter.
11. **[Phase B]** Bring the motors up last, with the Jetson on a bench supply,
    until the MCF302CB input range is confirmed.

---

## 11. Failure modes

| Symptom | Most likely cause | Check |
|---|---|---|
| `Motor 'x' was not found` | Servo bus has no V+ | 7.4 V at the servo connector; E-STOP not latched; branch fuse |
| `[RxPacketError] Input voltage error!` | Rail out of the 6.0–7.4 V window | Buck output; voltage drop under load; **pot moved** |
| Random arm dropouts under motion | Servo return current through USB ground | Power return gauge and star-ground bonding — §9 |
| Jetson reboots when arms accelerate | Shared rail sag | Confirm the two converters are genuinely independent |
| `VIDIOC_STREAMON: No space left on device` | USB 2.0 bandwidth exhausted | Force MJPEG; move a camera to its own root port — §8 |
| D455 drops out intermittently | Behind a hub, or a marginal cable | Give it a dedicated Type-A port |
| Converter hot, output sagging | Undersized or non-synchronous | The LM2596 is ~2 A real — it is not an arm-rail part |
| Both arms dead, Jetson fine | E-STOP latched, or the relay coil lost 12 V | Working as designed — §6 |

---

## 12. Open items

- [ ] Measure the second lot of DC barrel pigtails — the listing says 5.5 × 2.1
      in the title and 5.5 × 2.5 in the variant. The Jetson needs **2.5**; a 2.1
      plug fits loosely and arcs under load.
- [ ] Confirm the Waveshare board passes barrel-jack voltage **straight through**
      to the servo bus (expected) rather than regulating it. Measure barrel V+
      to servo connector V+ before trusting the 7.4 V design.
- [ ] Record the actual insulation colours of the Hilti B+ / B− wires.
- [ ] Confirm the relay module's DC contact rating from its datasheet — 10 A /
      30 VDC is assumed [E].
- [ ] Buy the two converters — **8–40 V → 7.4 V, ≥10 A, CC+CV** for the arms and
      **15–40 V → 12 V, ≥10 A synchronous** for the Jetson. Nothing in this
      document can be brought up without them.
- [ ] Buy the 8.0 V TVS, the fuse holders and ATO set, the ≥30 A disconnect, the
      XT60 pigtail, and 14 AWG silicone wire — the only gauge not on hand. Full
      costed list in [`hardware_inventory.md`](hardware_inventory.md).
- [ ] **[Phase B]** MF5010 winding variant (10T vs 35T) and MCF302CB input range.
- [ ] **[Phase B]** Resolve the `/dev/ttyUSB0` collision between the LiDAR and
      the motor interface with udev symlinks.
