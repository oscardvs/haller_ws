# Haller Power System

Power architecture for the Haller mobile robot, built around a Hilti B 22-195 Nuron
battery pack.

Each fact below is tagged by provenance:

- **[M]** measured on the bench
- **[D]** from a manufacturer datasheet or official product page
- **[C]** calculated/derived from [M] or [D] values
- **[E]** estimate — not yet verified, do not design safety margins around it

---

## 1. Battery: Hilti B 22-195 Nuron

### Manufacturer specifications [D]

| Parameter | Value |
|---|---|
| Marketed voltage | 22 V |
| Rated capacity | 9.0 Ah |
| Rated energy | 194.4 Wh |
| Weight | 1.33 kg |
| Dimensions (L×W×H) | 156 × 87 × 76 mm |
| Chemistry | Li-ion |

Source: [Hilti B 22-195 product page](https://www.hilti.co.uk/c/CLS_POWER_TOOLS_7125/CLS_BATT_CHARGERS_POWER_STATIONS_7125/r22580127)

### Derived cell configuration [C]

```
194.4 Wh / 9.0 Ah = 21.6 V   → true nominal voltage
21.6 V / 3.6 V per cell      → 6 cells in series (6S)
9.0 Ah / 3.0 Ah per cell     → 3 cells in parallel (3P), assuming 21700 cells
```

**Configuration: 6S3P, 18 cells.** Confirmed by measurement (§1.4).

### Voltage envelope [C]

This is the single most important table in this document. Everything downstream
must tolerate the maximum **and** still regulate at the minimum.

| State | Per cell | Pack |
|---|---|---|
| Full charge (absolute max) | 4.20 V | **25.2 V** |
| Nominal | 3.60 V | 21.6 V |
| Practical operating floor (see §6) | 3.40 V | 20.4 V |
| Cell cutoff (do not reach) | 2.50–3.00 V | 15.0–18.0 V |

**Design window: 20.4 V – 25.2 V** in normal operation, with 25.2 V as the
hard number for component voltage ratings.

### Bench measurements [M]

Measured 2026-07-23, pack near full charge, no load:

| Measurement | Value |
|---|---|
| B+ to B− | **24.3 V** |
| Implied per-cell | 4.05 V |

**Key finding: the pack outputs voltage on B+/B− unconditionally.** No tool
handshake is required to enable the output. This was the main open risk, since
Hilti's Nuron generation introduced a digital tool↔battery handshake and it was
not publicly documented whether that gates the power path. It does not.

Corroborated by lab practice: other users in the lab run this pack directly from
B+/B− and the output holds under load.

### Connector pinout [M]

The pack presents **5 slots**:

| Slot | Function |
|---|---|
| 1 | B− |
| 2 | B− |
| 3 (centre, double-width) | D — data, 2 contacts |
| 4 | B+ |
| 5 | B+ |

The official Hilti sliding connector breaks these out to **6 wires**:

| Wire | Function | Action |
|---|---|---|
| ×2 | B+ | Bond together, fuse, use |
| ×2 | B− | Bond together, use |
| White | Data (Nuron comm bus) | **Leave unconnected** |
| Blue | Data (Nuron comm bus) | **Leave unconnected** |

Notes:

- The doubled B+ and B− contacts exist to share current. **Bond each pair
  together** at the connector — do not run the robot off a single contact, or
  you halve the current rating of the interface.
- The white/blue pair maps to the double-width centre D slot. These carry the
  Nuron data link and are not needed for power. Insulate them individually;
  do not let them short to each other or to either rail.
- TODO: record the actual insulation colours of the B+ and B− wires.

---

## 2. Load budget

| Load | Qty | Voltage | Power | Provenance |
|---|---|---|---|---|
| Jetson Orin Nano devkit | 1 | 9–20 V | 45 W budget | [D] |
| MF5010 BLDC motor | 2 | 16 V rated winding | 128 W max, 5.06 A rated | [D] |
| RPLIDAR A1M8 | 1 | 5 V (USB) | ~2.5 W | [E] |
| IMX219 camera | 1 | from Jetson CSI | negligible | — |

Drive is differential: **2 driven front wheels + rear caster**, so two motors.

**Peak worst case [C]:** 2 × 128 W + 45 W ≈ **301 W**, which is ~12.5 A at 24 V.
This is an absolute stall/max-power figure, not a driving figure — but it sets
the fuse and wire gauge.

**Runtime estimate [E]:** at a realistic ~80 W average draw, 194.4 Wh gives
roughly 2.4 h. Not yet measured.

### Jetson Orin Nano devkit [D]

| Parameter | Value |
|---|---|
| DC jack input range | **9–20 V** |
| Barrel jack | 5.5 mm OD × 2.5 mm ID, centre positive |
| Stock adapter | 19 V @ 2.37 A (45 W) |
| Module power modes | 15 W / 25 W / MAXN SUPER |

> **The pack must never be connected directly to the Jetson.** At 25.2 V full
> charge the pack is 5.2 V above the board's absolute maximum, and even at
> 21.6 V nominal it is over the limit. A step-down converter is mandatory.

Sources: [Carrier board specification](https://developer.nvidia.com/downloads/assets/embedded/secure/jetson/orin_nano/docs/jetson_orin_nano_devkit_carrier_board_specification_sp.pdf),
[NVIDIA developer forums](https://forums.developer.nvidia.com/t/jetson-orin-nano-input-voltage/298625)

### MF5010 motors [D]

From `docs/MF5010_Specs.pdf`. Two winding variants exist and **it is not yet
recorded which one Haller uses** — the difference is large:

| Parameter | 10T | 35T |
|---|---|---|
| Rated voltage | 16 V | 16 V |
| Rated current | 5.06 A | 1.35 A |
| Max power | 128 W | 12 W |
| Max speed | 3050 rpm | 870 rpm |
| Rated torque | 0.26 N·m | 0.24 N·m |
| Speed constant | 150 rpm/V | 27.5 rpm/V |

Recommended drive (DF40): **input 7.4–32 V**, so a 25.2 V pack rail is within
the driver's range.

> **Two open concerns.** (1) The project also references an MCF302CB controller
> (`docs/LK-demo-MCF302CB-CAN.zip`); confirm *that* board's input range rather
> than assuming the DF40's 7.4–32 V. (2) The motor winding is rated 16 V. At
> 25.2 V with a 150 rpm/V constant, the 10T variant's no-load speed exceeds its
> 3050 rpm max rating if the driver ever commands full duty. Verify the driver
> enforces a speed/voltage limit before running on a full pack.

---

## 3. Power architecture

```
                          ┌─── 20 A fuse ──── motor drivers (raw pack, 20.4–25.2 V)
                          │
Pack B+ ──┬───────────────┤
          │               │
          │               └─── 5 A fuse ──── [buck DC-DC] ── 12 V ── Jetson Orin Nano
          │                                                            │
Pack B− ──┴──────────────── common ground ──────────────────┘          └── USB ── RPLIDAR (5 V)
```

**Motors on the raw pack, Jetson behind the converter.** This keeps ~20 A of
motor current out of the DC-DC entirely, and isolates the Jetson from the
voltage sag when four motors start simultaneously.

### Why 12 V and not 19 V

Both are electrically valid. The tradeoff:

| | 12 V | 19 V |
|---|---|---|
| Margin below Jetson's 20 V max | 8 V | **1 V** |
| Current at 45 W | 3.75 A | 2.37 A |
| Regulation floor (pack V needed) | ~14 V | ~21 V |
| Off-the-shelf high-current parts | abundant (automotive) | scarce |

**Decision: 12 V**, on three grounds:

1. **Ceiling margin.** The Jetson's absolute max is 20 V. A 19 V setpoint leaves
   1 V — any converter overshoot on a load transient (four motors cutting out
   at once) reaches the limit, on a board with no input protection behind it.
2. **Parts availability.** 12 V is the automotive rail. A sealed 24 V→12 V 10 A
   synchronous converter is inexpensive and its native 18–32 V input window is
   an exact match for this pack.
3. **Rail reuse.** LiDAR, fans, and any future router/switch all want 12 V.

19 V's genuine advantage is lower current (2.37 A vs 3.75 A), meaning less I²R
loss and an easier output stage. It is a defensible choice; this design takes
the margin instead.

> Note: an earlier draft of this reasoning claimed 19 V would brown out at half
> charge. That was wrong. Losing regulation at 19 V out requires the pack near
> 20 V, which is ~3.33 V/cell — roughly 85% discharged, and below the recharge
> practice in §6. The floor is not the problem; the ceiling is.

### DC-DC converter requirements

| Requirement | Value | Rationale |
|---|---|---|
| Input range | 18–25.2 V minimum | Full pack envelope |
| Absolute max input rating | **≥ 40 V** | Headroom over 25.2 V for regen spikes |
| Output | 12 V | §3 |
| Continuous output current | **≥ 5 A** | 3.75 A for 45 W, plus margin |
| Topology | Synchronous buck | Efficiency and thermals |

#### Selected part class

A **sealed automotive 24 V→12 V 10 A synchronous buck converter**. This class of
part is mass-produced for trucks, RVs and golf carts, so it is inexpensive
(~€12–20) and widely stocked.

**Purchasing criterion — input range must be 15–40 V.** Many modules labelled
"24 V→12 V" specify a *minimum input of 18–20 V* because they target lead-acid
24 V systems. That minimum sits at or above this pack's 20.4 V operating floor,
leaving no margin once the pack sags under motor load. Verify the listed input
range rather than trusting the "24V" in the product title.

Avoid wide-range 20–70 V buck-boost modules: the 20 V minimum is at the floor,
and the added topology buys nothing here.

Requirements to confirm at purchase: 15–40 V in, 12 V ≥ 10 A out, synchronous
rectification, die-cast/potted housing (the housing is the heatsink), and
OV/UV/overload/thermal/short protection.

#### The LM2596 is not suitable [D]

The LM2596 module currently on hand **must not be used to power the Jetson**:

| | LM2596 | Required |
|---|---|---|
| Headline rating | 3 A | 5 A |
| Real continuous, air-cooled | ~2 A | 3.75 A |
| Topology | Non-synchronous (catch diode) | Synchronous |
| Efficiency | 60–90% | — |

At 12 V / 3.75 A it would run at ~190% of its comfortable continuous rating.
The 3 A figure is a thermal limit requiring a heatsink, and the catch diode
dissipates significant power at this step-down ratio. Expected outcome is
thermal shutdown mid-mission, or failure.

Going to 19 V instead only reduces the demand to 2.37 A, which is still above
the ~2 A air-cooled figure. The part is undersized either way.

**Acceptable LM2596 uses:** low-current auxiliary rails (< 1.5 A), and bench
bring-up of the Jetson in 15 W mode with no peripherals attached (~1.3 A at
12 V) to validate the wiring chain before the proper converter arrives.

---

## 4. Protection and fusing

A 194 Wh pack with low internal resistance will deliver hundreds of amps into a
short. That is enough to weld tools and to vent cells.

| Location | Rating | Protects |
|---|---|---|
| B+, at the connector | 20 A blade | Everything — **fit this first** |
| DC-DC input branch | 5 A | Jetson path |

Main fuse sizing: two motors at 5.06 A rated ≈ 10 A continuous, ~12.5 A at peak
power, plus the DC-DC branch. A 20 A fuse clears that with headroom while still
protecting the wiring. (An earlier draft specified 30 A for a four-motor layout;
the robot is two driven wheels plus a caster.)

Rules:

- The main fuse is the **first** component after B+. Nothing gets wired
  downstream until it is in place.
- Never work on the harness with the pack connected. Physically remove it.
- Insulate the white/blue data wires individually.
- Motor regen/back-EMF on deceleration can push the shared rail **above**
  25.2 V. Fit bulk capacitance and a TVS diode across the rail, and ensure the
  DC-DC's absolute max input has real margin over 25.2 V.

---

## 5. Bring-up procedure

1. Fit the 20 A fuse at B+. Bond the two B+ wires and the two B− wires.
2. Insulate white and blue separately.
3. Confirm polarity with a multimeter on DC before connecting any load.
4. Bench-test the DC-DC **into a dummy load** (e.g. a 12 V car bulb) and confirm
   the output holds 12 V across an input sweep from 25 V down to 19 V.
5. Only then connect the Jetson.
6. Bring up motors separately, with the Jetson on a bench supply, until motor
   driver input range is confirmed (§2).

---

## 6. State of charge and deep discharge

The pack's onboard BMS state is not characterised. **Do not rely on it as an
operating limit** — cell protection cutoff is a last-resort safety feature, and
repeatedly reaching it degrades the pack.

### Current lab practice

Recharge when the pack indicator shows **2 of 4 green LEDs**. This is
conservative and good for pack longevity. Keep doing it until the monitor below
is in place.

### Estimated LED-to-voltage mapping [E]

From a generic Li-ion discharge curve — **not measured**, and to be calibrated
by logging actual resting voltage against LED count:

| LEDs | Approx SoC | Per cell | Pack (at rest) |
|---|---|---|---|
| 4/4 | 75–100% | 4.00–4.20 V | 24.0–25.2 V |
| 3/4 | 50–75% | 3.83–4.00 V | 23.0–24.0 V |
| 2/4 | 25–50% | 3.67–3.83 V | 22.0–23.0 V |
| 1/4 | < 25% | < 3.67 V | < 22.0 V |

### Proposed monitor thresholds

To be implemented as a voltage divider + ADC publishing
`sensor_msgs/msg/BatteryState`, wired into `haller_hardware`:

| Level | Per cell | Pack | Behaviour |
|---|---|---|---|
| Warn | 3.60 V | 21.6 V | Log warning, surface in diagnostics |
| Critical | 3.40 V | 20.4 V | Return to dock / stop accepting goals |
| Hard stop | 3.20 V | 19.2 V | Disable motors |

> Measure these **at rest or through a low-pass filter**. Under ~20 A of motor
> load the pack sags several hundred millivolts, and an unfiltered reading will
> trip thresholds spuriously during acceleration.

---

## 7. Open items

- [ ] Record B+/B− wire insulation colours on the Hilti connector
- [ ] Identify MF5010 winding variant (10T vs 35T) — §2
- [ ] Confirm MCF302CB motor controller input voltage range — §2
- [ ] Confirm driver enforces speed limit at 25.2 V rail vs 16 V motor rating
- [ ] Source and fit the ≥ 5 A synchronous 12 V converter
- [ ] Calibrate LED-to-voltage table against measured resting voltages
- [ ] Implement `BatteryState` publisher in `haller_hardware`
- [ ] Measure actual average current draw to replace the runtime estimate
