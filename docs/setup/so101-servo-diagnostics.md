# Diagnosing a "broken" SO-101 servo

Written 2026-08-27, after investigating the desktop arm on the suspicion that
the second or third motor in the line had failed. **It had not.** Every servo on
that arm passed every test. This document records how that was established, the
baselines to compare a future suspect against, and the traps that made the
healthy arm *look* broken along the way.

The single most useful conclusion: on this hardware, **"a joint won't come up"
is far more often the arm wedged against something than a failing servo**, and
the two are easy to tell apart once you know what to measure.

---

## 1. Verdict and evidence

All six STS3215s on the arm: firmware 3.10, model 777, `Status` `0x00`
throughout, 7.3 V rail, 30-38 °C.

| Test | Result |
|---|---|
| Bus scan, 8 baud rates | All 6 IDs at 1 Mbaud only |
| Ping reliability | 200/200 per servo, 0 comm failures, ~0.5 ms flat |
| Latched alarms | `0x00` on all six, including after a hard stall |
| Rail voltage | 7.3 V on 99.5 % of 5194 samples |
| EEPROM config | Identical across 1-5; ID 6 intentionally lower limits |
| Deadband | 3-6 ticks (0.26-0.53°), uniform |
| Endurance, 25 cycles | Zero drift; `elbow_flex` returned to −1 ticks 25/25 |
| Thermal, ~7 min cycling | +1 to +2 °C |
| **Lift against gravity** | **`elbow_flex` raised the whole forearm horizontal → vertical at a 300/1000 torque cap** |

That last row is the decisive one. See §3.

---

## 2. Separating gravity from friction

Load alone cannot tell you whether a joint is stiff or merely loaded. Sweep the
same arc in both directions and decompose the **signed** `Present_Load`:

```
moving +:  L+ = G + F        G = (L+ + L-) / 2     gravity, pose-dependent, benign
moving -:  L- = G - F        F = (L+ - L-) / 2     friction, the gearbox wear metric
```

`F` is pose-independent, so it is the number to compare between joints and
across time. Measured on the healthy arm (torque cap 500):

| joint | friction `F` | deadband |
|---|---|---|
| `shoulder_lift` | 37.6 | 6 ticks |
| `elbow_flex` | 46-48 | 6 ticks |
| `wrist_flex` | 35.0 | 4 ticks |
| `wrist_roll` | 35.0-35.3 | 4 ticks |
| `gripper` | 67 | 3 ticks |

A real gearbox fault reads 2-5× these, not +25 %. `elbow_flex` sits highest
simply because it carries the largest gravity moment in the rest pose. A damaged
tooth shows as a **localized spike at a specific output angle**, repeatable
across passes and directions — not as a uniformly raised floor.

**Do not take `abs()` of the load.** It destroys the sign that separates the two
terms, and it was the reason a first pass mistook a normal joint for a stiff one.

---

## 3. The lift test — the one that actually answers it

Everything above runs at low load and cannot distinguish a healthy servo from
one that fails under real weight. This test can, and it is cheap.

Start with the arm **fully extended upward**. At vertical, gravity torque on a
joint is ~zero; it grows as `sin(angle)` as you rotate away. So:

1. Hold every other joint rigid so the load is well defined.
2. Walk the joint out from vertical toward horizontal in 100-tick steps,
   recording settled load and tracking error at each.
3. From the loaded pose, command it back to vertical at **escalating torque
   caps**, and record the minimum cap that completes the lift.

Result for `elbow_flex`, walking out to 88°:

| offset | angle | tracking err | load |
|---|---|---|---|
| +100 | 8.8° | 0 | 0 |
| +400 | 35.2° | 4 | 40 |
| +700 | 61.5° | 5 | 48 |
| +1000 | 87.9° | 7 | 64 |

Smooth, monotonic, no spikes — a textbook gravity curve. Then the lift back:

| torque cap | result |
|---|---|
| 200/1000 | no lift; load saturated exactly at the cap |
| **300/1000** | **full 987-tick lift, horizontal → vertical** |

**`elbow_flex` does its hardest real job at ~25-30 % of rated torque — better
than 3× headroom.** Use 300 as the reference figure. If a future elbow needs
materially more to make this same lift, that is a genuine fault.

Note that at cap 200 the joint did not *fail* — its load pinned exactly at the
imposed cap. Saturation at the cap is the signature of a torque limit, not of
broken hardware. Always check whether a "no move" is sitting at your own ceiling.

---

## 4. Steady-state droop is not sag

**`I_Coefficient` is 0 on all six servos.** With no integral term, steady-state
error is strictly proportional to load, and the constant is the same on every
joint on this arm:

```
droop (ticks) ~= load / 9
```

Verified end to end: at full gravity load the elbow carried load 64 and drooped
7 ticks; 64/9 = 7.1. At 30 % load a joint would sit ~2.9° below its commanded
angle and stay there.

**This looks exactly like a failing joint and is a tuning property.** Suspect it
before condemning a servo. For this arm's own weight the effect is small (7
ticks, 0.6°); it matters more with a payload.

It also breaks naive tooling: **any arrival check tighter than ~12-15 ticks will
never succeed**, because the servo physically cannot close the last of that
error. A ±4 tick tolerance made a perfectly healthy `shoulder_pan` look stalled.

---

## 5. Traps

- **Goal-lunge.** `Goal_Position` reads 0 on a freshly booted arm. Seed goal =
  present position *before* `Torque_Enable=1`, or every joint slams toward tick
  0. (Also noted in the 2026-08-24 bring-up.)
- **EEPROM read-back.** Reading a register back while `Lock=0` returns garbage —
  `I_Coefficient` read as 250 immediately after a correct write of 8. Read it
  only *after* re-locking. A first attempt aborted on this phantom failure.
  Sequence: `Lock=0` → write → `Lock=1` → *then* verify, with ~0.4 s between
  steps and the port flushed.
- **Sign encodings.** `Present_Load` is sign-magnitude **bit 10**;
  `Homing_Offset` is sign-magnitude **bit 11**. Confirmed against
  `lerobot/motors/feetech/tables.py`.
- **One anomalous sweep means nothing.** A single pass measured `wrist_roll`
  friction at 973 (~full torque). Three repeats gave 35.0-35.3. The bad pass
  also reported nonzero gravity on a *roll* axis, which is physically
  impossible — a good tell that a measurement is corrupt rather than alarming.
  Repeat before believing.
- **`pkill -f <script>` kills your own shell** if the pattern appears in the
  invoking command line. Cost two emergency-stop attempts.

---

## 6. Collisions, and a warning

**Do not run a full-range sweep in a confined workspace**, even at a capped
torque. Doing so wedged this arm against its own base and the table, and later
against itself. A capped torque makes a collision non-damaging — no alarm ever
latched — but it does not make it harmless: the arm ends up tangled, and every
measurement taken while it is jammed is void.

**How to recognise it in the data:** a joint blocked at the *same tick in both
directions* is geometry, not electronics. During the sweep, `shoulder_lift`
reported obstruction at 2716 up and down, and `elbow_flex` at 2708 up and down.
That is what "the third motor won't come up" actually was. Freed, the elbow
lifted its full load at a third of its torque.

Collision points observed this session, **as raw observations, not limits** —
they were recorded with the other joints limp, so they are pose-dependent and
must not be treated as authoritative travel bounds:

| joint | soft limits (EEPROM) | collision seen at |
|---|---|---|
| `shoulder_pan` | 699..3397 | 797 (descending) |
| `shoulder_lift` | 915..3313 | 2716, both directions |
| `elbow_flex` | 810..3037 | 2708, both directions |
| `wrist_flex` | 836..3178 | 1798 (descending) |
| `wrist_roll` | 0..4095 | 1218, both directions |

For reference, `elbow_flex` travel is **810 = arm straight, 3037 = folded**.

---

## 7. What was not tested

- **Real payload.** Everything ran under the arm's own weight only. A fault that
  appears only under an external load is still untested.
- **`I_Coefficient` > 0 on hardware.** The experiment was attempted twice and
  abandoned — first on the read-back trap in §5, then because the arm had fallen
  and was colliding with itself, making every sample an obstruction reading. The
  `load/9` relationship in §4 is measured and solid; the *fix* is unproven.
- **Cross-arm comparison.** The second arm was unreachable from this desktop.

---

## 8. Unrelated issue found in passing

The servos' stored `Homing_Offset` values match **none** of the three files in
`~/.cache/huggingface/lerobot/calibration/robots/so_follower/`. Closest is
`haller_leader.json` (5/6 within ~65 ticks; `wrist_flex` off by 535);
`haller_follower.json` is off by up to **3400 ticks**.

`scripts/test_so101_arm.py` defaults to `--id haller_follower` and calls
`connect(calibrate=True)`, which on a mismatch prompts a recalibration that would
overwrite this arm's zero. If an arm suddenly drives to visibly wrong angles,
check this before suspecting hardware.
