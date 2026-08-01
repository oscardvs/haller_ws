# Motion safety envelope — design

**Date:** 2026-08-01
**Status:** approved, not yet implemented
**Scope:** spec A of three. B is Quest-in-sim bring-up; C is sim-to-real physics
matching (whose validation step is gated on the rig being repaired).

---

## 1. Why

On 2026-08-01 the right arm was recalibrated through the HMI wizard, then Home
was pressed. The arm slewed across its workspace, collided with the bench, and
stalled. Six servos at locked-rotor current cooked the 7.4 V LM2596 DC-DC.

The converter was undersized — `docs/wiring.md:56` marks the LM2596 at "~2 A
real continuous, Aux only, **not** the arms", against `docs/power_system.md`'s
~5 A per arm — so it was destined to fail. But the converter is a symptom.
**The arm should never have made that move**, and with the specified 10 A buck
fitted it would have made the same move into the same bench with a healthy
supply behind it.

### Root cause

`ArmHandle.home()` (`hmi/backend/haller_hmi/arm.py:114`) is:

```python
def home(self) -> dict[str, float]:
    goal = {j: 0.0 for j in self.joint_limits_deg}
    return self.send_goal(goal)
```

and `send_goal` (`arm.py:100`) does two things that combine badly:

```python
if not self.torque_enabled:
    self.enable_torque()          # limp arm silently energized
...
self.robot.send_action(action)    # ONE position write: no interpolation, no speed cap
```

Calibration *redefines where 0° is*. The arm sits wherever the sweep ended,
which is nowhere near the newly-defined zero. So Home takes a limp arm,
energizes it, and commands every joint to slew at full servo speed to a distant
pose. Nothing in the path bounds distance or velocity.

`SimArmHandle` (`haller_hmi/sim/arm.py:83`) duplicates `home()` and `send_goal()`
verbatim, defect included. That duplication is why the sim could not have caught
this: both sides were wrong in the same way.

## 2. Goals

1. A discrete move can never sweep an unplanned path across the workspace.
2. A single corrupted input frame can never produce a large commanded jump.
3. The real and sim paths execute **the same** motion-safety code, so a defect
   cannot exist in one and not the other.
4. E-STOP halts an in-flight move.

## 3. Non-goals

- Collision-aware path planning. Refusing large moves is the mitigation; the arm
  is jogged manually instead.
- Physics fidelity between sim and real — that is spec C.
- Changing the calibration wizard. Its post-save state (arm parked far from the
  new zero) is legitimate; the motion path must cope with it.

## 4. Architecture

### 4.1 Shared primitives — `haller_hmi/safety.py`

Three pure functions beside the existing `clamp_joint_goal` (`safety.py:38`):

```python
def limit_step(
    current: dict[str, float],
    goal: dict[str, float],
    max_step_deg: float,
) -> dict[str, float]:
    """Cap each joint's per-call delta. Streaming path."""

def check_move_size(
    current: dict[str, float],
    goal: dict[str, float],
    threshold_deg: float,
) -> dict[str, float]:
    """Return {joint: delta} for joints exceeding threshold; empty if all within."""

def plan_ramp(
    current: dict[str, float],
    goal: dict[str, float],
    max_speed_deg_s: float,
    hz: float,
) -> list[dict[str, float]]:
    """Interpolated waypoints from current to goal, bounded by max_speed_deg_s."""
```

Pure and synchronous, so they are testable without a world or a serial port.

### 4.2 `MoveExecutor`

New module `haller_hmi/motion.py`. One executor per arm handle. Runs a ramp on a
background thread and re-checks `guard.assert_manual()` before every waypoint.

E-STOP (`server.py:371`) already sets `Mode.STOP` on every guard, so that guard
re-check is the entire cancellation mechanism — no separate channel, and mode
changes and teleop takeover cancel a ramp for free.

A new discrete command cancels any ramp already running on that arm.

### 4.3 Shared `home()`

`home()` is **removed from both `ArmHandle` and `SimArmHandle`** and defined once
against the shared handle interface. Both handles keep their own `send_goal`,
because the transport genuinely differs — `ArmHandle` calls
`robot.send_action({f"{j}.pos": v})`, while `SimArmHandle` translates
snake_case → CamelCase through `LEROBOT_TO_MJCF`, prefixes the arm name, and
calls `world.write_ctrl_deg(...)`. Only the distance/velocity policy is shared,
and it lives in exactly one place.

### 4.4 Why the ramp cannot block

The routes at `server.py:253` (goal), `:284` (home) and `:324` (preset) are
`async def` calling synchronous motion code. A blocking ramp would stall the
event loop for every other client — the same starvation that produced a
"Failed to fetch" banner in the browser during the calibration reload. Hence the
background executor.

## 5. Behaviour

### 5.1 Streaming path

Callers: `teleop.py:157` (60 Hz), `human_teleop.py:927` (VR/MediaPipe),
`sim/teleop.py:109`.

`send_goal` clamps to joint limits as today, then applies
`limit_step(..., max_speed_deg_s / ramp_hz)`. A garbage frame commanding a 100°
jump becomes one bounded step; the next good frame corrects it.

**The implicit torque re-enable is removed.** Every legitimate caller already
enables torque explicitly first — `server.py:275`, `human_teleop.py:498` and
`:558`, `calibration.py:191`. The only path that relied on the side effect was
`home()`, which is precisely the dangerous one.

### 5.2 Discrete path

1. Read current measured joint positions.
2. `check_move_size(current, goal, large_move_deg)`. Non-empty → refuse.
3. Torque disabled → refuse.
4. Otherwise `plan_ramp(...)` and hand the waypoints to the `MoveExecutor`.

## 6. Config

New `motion:` block in `config.yaml`, overridable per arm:

| key | default | rationale |
|---|---|---|
| `max_speed_deg_s` | 60 | The STS3215 reaches ~375°/s at 7.4 V. 60 is ~16% of capability — deliberately conservative for a bench with two arms sharing a workspace. |
| `large_move_deg` | 30 | Above this a discrete move is refused and the operator jogs manually. |
| `ramp_hz` | 50 | Waypoint rate; also sets the streaming per-step cap at `max_speed_deg_s / ramp_hz` = 1.2°. |

## 7. Error handling

Refusals raise `ConflictError`, which the existing route wrappers already map to
HTTP 409 (`server.py:255`, `:286`, `:326`).

| condition | response |
|---|---|
| Any joint delta > `large_move_deg` | 409, listing each offending joint and its delta, and advising manual jog |
| Torque disabled on a discrete move | 409 naming the arm; no silent energize |
| Ramp cancelled by E-STOP or mode change | Not an error. Logged at WARNING; the arm holds position |

## 8. Testing

All tests run without hardware. Sim-backed tests use the existing `MuJoCoWorld`.

1. **The incident, reproduced.** A sim arm parked at a post-calibration pose,
   `home()` → refuses, and the world's joint positions are unchanged. This test
   fails against today's code, which slews.
2. **Ramp bounds.** Every consecutive waypoint delta ≤ `max_speed_deg_s / ramp_hz`.
3. **E-STOP mid-ramp.** Setting `Mode.STOP` during a ramp halts it within one
   waypoint and leaves the arm where it stopped.
4. **Streaming outlier.** A frame commanding +100° on one joint yields a single
   step of `max_speed_deg_s / ramp_hz`, not a jump. This also covers the
   corrupted UART reads suspected of poisoning the right arm's calibration sweep.
5. **No silent energize.** `send_goal` on a torque-disabled handle does not call
   `enable_torque`.
6. **Parity.** A test asserting `ArmHandle` and `SimArmHandle` resolve `home` to
   the same shared implementation. It fails if anyone reintroduces a per-handle
   copy — the structural regression that made this incident possible.

## 9. Rollout

The hardware is down pending a 10 A buck, so this ships and is verified entirely
in sim. When the rig returns, the first real-hardware check is a deliberate
large-move refusal on a clear bench before anything else is driven.

Independently of this spec, the arm rail needs the converter the wiring doc
already specifies — adjustable buck 8–40 V → 7.4 V, 10 A, shared across both arms
through the E-STOP relay — and the 10 A fast-blow fuse at `wiring.md:167`.
Fitting another LM2596 will repeat the failure.
