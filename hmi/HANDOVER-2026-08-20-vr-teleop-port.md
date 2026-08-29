# VR teleop, ported onto the `vr-teleop-kit` layering — 2026-08-20

Port of the architecture in
[`Dream-Machines-Robotics/vr-teleop-kit`](https://github.com/Dream-Machines-Robotics/vr-teleop-kit)
(written up at <https://aurelarnold.xyz/blog/vr-teleoperation-stack/>) onto this
rig, plus the two things a single-arm hardware run needed: sessions that accept
one arm, and a collision guard you can switch off from the bench.

Nothing was deleted. The previous wrist-point mode and the original
body-angle mode are both still selectable, because a hardware session that
goes wrong needs somewhere to go that is known to work.

---

## What the reference stack does, and what we took

| their layer | ours | carried over? |
|---|---|---|
| `core/pose_mapping.py` | `haller_hmi/vr_teleop/core/` | **yes, nearly verbatim** — clutch-relative mapping, incremental rotation, absorbing reach limits, rotation pivot, yaw-on-engage |
| `ik/` (DK1, 6-DoF, 3+3) | `haller_hmi/vr_teleop/ik/` | **rewritten** — the SO-101 is 5-DoF and splits 3+2. See below. |
| `relay/` (FastAPI hub + WebXR page) | `haller_hmi/vr_teleop/relay.py` + `web/` | **yes**, mounted inside the existing app rather than run as its own server |
| `lerobot/bi_quest_teleop.py` | `haller_hmi/vr_teleop/teleop.py` | **yes in shape** — per-hand clutch state, gains, filtering, haptics — but it emits joint goals for `HumanTeleopSession` instead of a LeRobot action dict |

Their own porting note says the mapping and the relay carry over to any arm
and the IK does not. That turned out to be exactly right.

### The IK is where the arms genuinely differ

Their solver splits a 6-DoF arm into two 3-DoF problems: joints 1-3 track a
wrist-invariant anchor point, joints 4-6 track orientation, each one damped
least-squares step per call. The reason for splitting is not speed — it is
that *a pure rotation of your hand should keep the robot's wrist still*,
which a coupled solve does not give you.

The SO-101 splits **3 + 2**:

* **Position is the same idea, and easier.** They have to *construct* an
  anchor (a site 10 cm past joint 4, offset to stay off the joint-1 axis).
  Our geometry hands it over: `wrist_flex` pivots exactly at the
  `Wrist_Pitch_Roll` origin, so that origin is invariant to both wrist joints
  and joints 1-3 own position outright, as a square 3×3.
* **Orientation has two axes for a three-dimensional demand.** The wrist
  Jacobian is 3×2; the damped step tracks the reachable part and ignores the
  rest. That is the standing 1-DoF deficit of any 5-DoF arm — with position
  fixed, the gripper's yaw is decided by `shoulder_pan` and no wrist can
  argue. We *report* it (`orient_residual`) and buzz the controller, so the
  operator is told to move their hand rather than twisting harder at
  something that cannot move.
* **Their gimbal machinery does not port.** `wrist_roll`'s axis is Rx(θ₄)·ŷ
  and `wrist_flex`'s is x̂, so the two are perpendicular at every pose — this
  wrist has no internal gimbal lock. Their gimbal-proximity ramp and the
  near-antipodal park gate both exist for a 3-axis wrist that can fold onto
  itself. Shipping them as dead code would advertise a failure mode the arm
  does not have. (`so101_kinematics._self_test` and
  `tests/vr_teleop/test_kinematics.py` pin the orthogonality.)

Three further deviations, each measured rather than assumed:

1. **Conditioning is read from σ_min, not |det J|.** On a 25 cm arm the
   determinant is small everywhere (peak 0.0035, median 0.0011), so it
   conflates "near singular" with "short lever arms" and any threshold that
   catches the first damps most of the workspace. σ_min is in m/rad and says
   the honest thing — how far the tool moves, in its worst direction, per
   radian. Median 0.045, best 0.082, collapsing toward 0 in the singular set;
   the ramp starts at 0.02.
2. **The posture bias is gated, not constant.** The position sub-problem is
   square, so a constant Tikhonov pull has no null space to hide in and shows
   up directly as motion. Measured: **3° of drift per solve at the home pose**
   (which sits near the straight-elbow singularity, so the ramp is open
   there) — i.e. the arm creeping away from the operator during the
   acquisition countdown, which is precisely the invariant the whole
   fast-handover design rests on. It is now scaled by both the singularity
   ramp and the position error, and is exactly zero when the target is where
   the arm already is.
3. **The position target is placed against the REACHABLE orientation.** They
   place their anchor with `R_target`, which is right on a 6-DoF arm. On a
   5-DoF one an unreachable yaw sits in that term forever and drags the
   *tool* off position by up to the 6 cm anchor offset. Measured before the
   fix: a 45° unreachable yaw pulled the tool **52 mm** off target and kept
   it there. The wrist is now pre-solved once, purely to answer "how far over
   can we actually get?", and that answer places the anchor. Same case after:
   **0.00 mm**.

### What we did not port

* **The wrist-pivot calibration ritual.** Their 5-second in-VR
  least-squares solve for the grip→wrist offset is replaced by one adjustable
  number in the client (`wrist pivot (m)`, default 0.09), which moves the
  read-out point off the palm and onto roughly where the wrist turns. Same
  idea, no ritual. Worth revisiting if pure twists still translate.
* **Grasp-force haptics.** They read gripper torque through an optional
  follower method. Our `ArmHandle.read_effort_norm()` already reports a
  normalised per-joint effort, so this is a small addition — but it is
  feedback, not a blocker, and it did not go in for a first run.
* **Their DOM-overlay settings panel.** The Quest Browser has **no
  `dom-overlay` on device** — a finding this repo already paid for once. The
  ported client draws its HUD to a 2D canvas and renders it on world-locked
  WebGL quads instead.

---

## Where the seams are

```
haller_hmi/
├── so101_kinematics.py     FK, full link frames, analytic geometric Jacobians.
│                           ONE chain definition, shared by the collision guard
│                           and the IK (it used to live inside collision.py and
│                           be reached into through a private name).
└── vr_teleop/
    ├── core/               robot-agnostic: quaternions, operator stances,
    │                       ClutchPoseMapper (reach limits, rotation pivot)
    ├── ik/                 SO-101: model + SO101DecoupledIK
    ├── config.py           every live-tunable knob, with hard bounds
    ├── teleop.py           QuestTeleoperator — per-hand state → joint goals
    ├── relay.py            broadcast WS hub + the WebXR page
    └── web/                index.html + client.js
```

`teleop.py` only *produces the target*. It emits the same `KeypointFrame`
shape `vr_pose_mode.py` always did, and `HumanTeleopSession` still owns
everything a bench session needs — per-side authority, the acquisition
countdown and rate ramp, the command filter, the collision guard, the mode
guard, E-STOP, and the recorder's `action` column. That is also what makes
the whole thing sim/real agnostic: the output is joint names and degrees,
which `ArmHandle` and `SimArmHandle` share.

### The one invariant to not break

**While the session has not handed a side over, the commanded pose stays
exactly on the arm.** The acquisition gate hands over when the commanded pose
matches the measured one, and a VR handover is near-instant only because
squeezing the grip anchors the target ON the arm — zero error by
construction. So the mapper re-anchors every frame until the side is DRIVING,
and once more on the first driving frame (the session flips authority inside
its own 60 Hz loop, so without that last one the first driven frame carries
however far the hand travelled in between). Pinned by
`test_gate_error_stays_zero_through_the_countdown` and
`test_handover_starts_from_the_hand_where_it_is_now`.

---

## Single-arm sessions

`HumanTeleopSession.start()` now takes `left_arm=None` or `right_arm=None`.
The absent side never acquires, is never written to, cannot be homed, and
reports `reason: "no_arm"` rather than borrowing the tracking-loss reason
(which would send the operator hunting for a hand that is not missing). The
collision guard sees only the sides that exist, so a single-arm session still
gets the self-collision pairs and the bench floors — which is most of what
bites on one arm anyway.

`POST /teleop/human/start` accepts `null` for either side; at least one is
required.

## The collision guard switch

`POST /teleop/human/collision {"enabled": bool}`, also on the VR page and the
cockpit panel. Two properties make it safe to offer:

* **Off still measures.** `status().collision.slack_m` keeps updating, so an
  operator who turned it off because it bit too early can still watch how
  close they actually are.
* **A guard with no mount geometry can never be switched on.** That is the
  fail-open the module exists to prevent, so `available: false` is one-way
  and enabling returns 409.

What the switch does **not** turn off, deliberately: the teleop's own
workspace floor (the commanded fingertip and wrist can never go below the
bench — this is `vr_teleop.config`, not the guard, precisely so it keeps
working when the guard is off), the per-joint limits, the acquisition ramp,
the rate caps, or the motion envelope.

---

## Verification

* `tests/vr_teleop/` — 73 tests: kinematics vs. central differences, the
  clutch mapping's reach limits, the solver's four load-bearing properties,
  the adapter's contract with the session, the relay's two frame shapes.
* `tests/test_human_teleop_single_arm.py`, `tests/test_collision_toggle.py`,
  `tests/test_routes_vr_teleop.py`.
* Full backend suite: 672 pass.
* Frontend: 140 tests, typecheck clean.
* `scripts/vr_smoke.py`: **50/50 from a cold sim backend**, including a new
  section 9b that drives the ported path end-to-end — single-arm start, the
  egocentric direction mapping, the absorbing reach limit, the guard switch,
  the workspace floor holding with the guard off, and that a frame with no
  `vr_mode` key takes the new path.

Two things the smoke run caught that no unit test would have:

* Sections 10-13 started failing because 9b left an arm parked where the
  body-angle acquisition gate could never match it. Fixed by parking the arm
  through the in-session home — which then failed too, because
  `request_home` correctly skips a DRIVING side and the grip had not been
  released first.
* The browser cached `client.js`, so an edit silently did not take. The relay
  now serves both files `no-store`; a stale client in a headset on someone's
  face looks exactly like a change that did nothing.
