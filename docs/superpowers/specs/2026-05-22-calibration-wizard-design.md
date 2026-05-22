# HMI Calibration Wizard — Design Spec

**Date:** 2026-05-22
**Status:** Approved (verbal "all good") — ready for implementation plan
**Author:** Oscar Devos
**Builds on:** [`2026-05-22-haller-unified-hmi-design.md`](./2026-05-22-haller-unified-hmi-design.md)

## 1. Problem & motivation

The unified HMI currently relies on the `lerobot-calibrate` CLI for arm calibration. That works, but two friction points pushed this onto the roadmap (see `README.md` and `hmi/README.md`):

1. **First-boot UX.** A fresh deployment has no `robots/so_follower/<id>.json`. The backend's `calibration_bootstrap` falls back to a sibling teleop file when one exists, but if neither exists the server refuses to start and the user has to drop to a terminal, learn the CLI, and re-run the HMI.
2. **Cross-arm 0° mismatch.** Each arm's calibration midpoint `(range_min + range_max) / 2` is the "0°" reference. Calibrating two arms independently lands their midpoints in slightly different physical poses; the leader↔follower teleop launcher then carries a per-joint offset that's most visible on `shoulder_lift`. The current workaround is to re-calibrate one arm by hand while it holds the other arm's neutral pose — a workflow that begs for in-HMI guidance.

The wizard's v1 scope is the **single-arm full calibration flow** — homing offsets + range-of-motion sweep + save — replacing the `lerobot-calibrate` step end-to-end in the browser. The paired-arm co-calibration that would directly solve point (2) is intentionally **deferred to v2**; the v1 wizard makes the manual workaround easier by letting the operator run single-arm calibration on the second arm without leaving the HMI.

## 2. Goals & non-goals

### Goals

- One full single-arm calibration session, browser-driven, no terminal.
- Reuses the existing free-drive (torque-off) UX from `ArmPanel`.
- Live tick / min / max table during sweep, fed by the existing 20 Hz telemetry stream — no second socket.
- Writes the same JSON shape `lerobot-calibrate` produces, so calibration files stay portable between the CLI and the HMI.
- Backup-then-overwrite saves: one rollback file is always available on disk.
- Updates both `robots/so_follower/<id>.json` and any sibling `teleoperators/*/<id>.json` so leader↔follower stays in sync.
- Safe by construction: blocked while any arm is in teleop or AUTO; E-STOP aborts the session with no file written; only one session active across the HMI.
- Surfaces an auto-prompt banner on the dashboard when an arm has no calibration file at backend start.

### Non-goals (v1)

- **Paired leader↔follower co-calibration.** Deferred to v2; the v1 wizard supports single-arm calibration on each arm individually instead.
- **Re-home-only mode.** Set-homing-without-sweep is useful but not enough of a recurring need to justify the extra UI state.
- **View / edit existing calibration.** The on-disk JSON remains the editable artifact for power users.
- **Range expansion beyond the physical sweep.** No "+5° headroom" knob; what the operator wiggles is what gets saved.
- **Editing the proposed JSON in the review step.** Save or cancel only.
- **History / undo beyond `<path>.bak-<ts>`.** One rollback file per save is enough.

## 3. Architecture

### 3.1 State machine

A new module `haller_hmi/calibration.py` owns a per-arm session. At most one session exists across the HMI at any time.

```
IDLE ──start──▶ HOMING ──capture_neutral──▶ SWEEPING ──finish_sweep──▶ REVIEW ──save──▶ DONE
                  │                            │                          │
                  └──abort─────────────────────┴──────────────────────────┴──▶ ABORTED
```

- **IDLE** — no session for this arm.
- **HOMING** — backend disables torque on the target arm; frontend shows live joint ticks; operator hand-poses the arm; clicks Capture.
- **SWEEPING** — backend keeps reading raw ticks at the telemetry rate and accumulates per-joint min/max; frontend renders the live `min | POS | max` table; operator wiggles every joint, clicks Done.
- **REVIEW** — backend holds the full proposed JSON in memory. Frontend shows old vs. new per joint with diff highlighting; operator clicks Save or Cancel.
- **DONE / ABORTED** — terminal states. Saving writes the new JSON, leaves a `.bak-<timestamp>` for the previous one, and reloads the arm so subsequent commands use the new calibration immediately.

E-STOP transitions any active session to ABORTED via the existing `/estop` handler.

### 3.2 Where work happens

```
┌────────────────────────────────────────────────────────────────┐
│ Browser                                                        │
│   CalibrationWizard.tsx        — modal, drives the 6 routes    │
│   CalibrationStatusCard.tsx    — per-arm row on /settings      │
│   useTelemetryStore             — same stream as ArmPanel       │
└──────────────┬──────────────────────────────┬──────────────────┘
   REST (6 routes)                  WS telemetry (existing stream)
               │                              │
               ▼                              ▼
┌────────────────────────────────────────────────────────────────┐
│ FastAPI server.py                                              │
│   /calibration/* routes  ──▶  CalibrationManager (singleton)   │
│                                  └─▶ CalibrationSession         │
│   telemetry tick reads ArmHandle.state_snapshot() and, when a  │
│   session is active for that arm, also reads raw ticks via the │
│   session for the calibration block of the frame.              │
└────────────────────────────────────────────────────────────────┘
                              │
                              ▼
                  SO101Follower / motors_bus
                    sync_read("Present_Position", motors, normalize=False)
                    write("Homing_Offset", motor, value)
                    disable_torque() / enable_torque()
```

The session never opens its own serial connection — it uses the `MotorsBus` already owned by the `ArmHandle`.

### 3.3 Approach: hybrid lerobot I/O + our own state machine

The backend uses lerobot's low-level building blocks for everything that touches the bus (`bus.sync_read`, `bus.write`, `bus.disable_torque`) and our own code for orchestration. The high-level helpers `bus.set_half_turn_homings()` and `bus.record_ranges_of_motion()` are NOT used directly because `record_ranges_of_motion` blocks on `enter_pressed()` and prints to stdout — it expects to own a terminal. We replicate its math (it's a handful of lines) inside `CalibrationSession` so we can drive it asynchronously and cancel cleanly.

## 4. Components

### 4.1 Backend (`hmi/backend/haller_hmi/calibration.py`)

```python
class CalibrationState(str, Enum):
    HOMING = "homing"
    SWEEPING = "sweeping"
    REVIEW = "review"
    DONE = "done"
    ABORTED = "aborted"


@dataclass
class CalibrationSession:
    arm_id: str
    state: CalibrationState
    homing_offsets: dict[str, int]            # written during capture_neutral
    mins: dict[str, int]                      # accumulated during sweep
    maxes: dict[str, int]                     # accumulated during sweep
    proposed: dict[str, dict] | None          # computed by finish_sweep

    def capture_neutral(self, handle: ArmHandle) -> None: ...
    def tick_sweep(self, handle: ArmHandle) -> dict[str, int]: ...   # called by telemetry
    def finish_sweep(self, handle: ArmHandle) -> dict[str, dict]: ...
    def proposed_snapshot(self) -> dict: ...  # for the telemetry calibration block


class CalibrationManager:
    """Singleton; at most one session across all arms."""

    current: CalibrationSession | None

    def start(self, arms: ArmManager, arm_id: str) -> CalibrationSession: ...
    def abort(self) -> None: ...
    def save(self, arms: ArmManager) -> tuple[Path, Path]: ...      # returns (target, backup)
```

**Pre-flight gating in `start`.** Raises `ConflictError` (mapped to HTTP 409) if:
- `self.current is not None` (another session active), or
- any arm in `arms.values()` has `guard.mode is not Mode.MANUAL`.

The check covers every configured arm, not just the target, because the spec calls for "all arms in manual" before calibration begins.

**Capture-neutral math.** For each motor, the session reads the current `Present_Position`, computes the new `homing_offset` using the same formula `MotorsBus._get_half_turn_homings` applies for the Feetech bus (see `lerobot/src/lerobot/motors/feetech/feetech.py`), then `bus.write("Homing_Offset", motor, offset)` so the motor's "0" becomes the captured pose. We do NOT zero `range_min`/`range_max` to a degenerate state on the bus during the session — the proposed range is purely the in-memory `mins`/`maxes` accumulator that gets written to the JSON on save.

**Sweep tick.** Called from the telemetry loop while `state is SWEEPING`. Reads raw ticks via the existing `MotorsBus.sync_read("Present_Position", motors, normalize=False)`, updates `mins` and `maxes`, returns the current tick dict for the telemetry frame.

**Finish-sweep validation.** If any joint has `mins[j] == maxes[j]` the manager raises `UnmovedJointsError` (mapped to HTTP 422) listing the unmoved joints; state stays SWEEPING so the operator can keep wiggling and retry. Otherwise computes the proposed JSON in the lerobot shape (per-joint `id, drive_mode, homing_offset, range_min, range_max`) and transitions to REVIEW.

**Save mechanics.**
1. Resolve target paths: always `robots/so_follower/<calibration_id>.json`, plus any sibling `teleoperators/*/<calibration_id>.json` found via the existing `calibration_bootstrap._find_teleop_calibration()` helper.
2. For each path that already exists, `shutil.move(p, p.with_suffix(f".json.bak-{timestamp}"))` first. Timestamp is UTC ISO-8601 with seconds resolution, `:` replaced with `-`.
3. Write the new JSON to each resolved path with `indent=2, sort_keys=False` to match the existing on-disk format.
4. `drive_mode` carries over from the prior file when one existed; otherwise `0`.
5. Reload the arm: `handle.disconnect()` then `handle.connect()`. Fast (<1s on the existing hardware), uses the code path the server already trusts.
6. Return `(target_path, backup_path | None)` so the route can report the backup name to the UI.

### 4.2 Backend (`hmi/backend/haller_hmi/server.py` additions)

Six new routes. All except `/calibration/status` mutate state and are POST.

| Method | Path | Body | Returns |
|---|---|---|---|
| GET  | `/calibration/status` | — | `{arms: [{id, has_file, path, mtime, in_session: bool}], current_session: {arm_id, state, proposed?: {...}, current?: {...}} \| null}` |
| POST | `/calibration/{arm_id}/start` | — | `{ok, state: "homing"}` — 409 on conflict |
| POST | `/calibration/{arm_id}/capture_neutral` | — | `{ok, state: "sweeping", homing_offsets: {...}}` |
| POST | `/calibration/{arm_id}/finish_sweep` | — | `{ok, state: "review", proposed: {...}, current: {...} \| null}` |
| POST | `/calibration/{arm_id}/save` | — | `{ok, state: "done", path, backup_path}` |
| POST | `/calibration/{arm_id}/abort` | — | `{ok, state: "aborted"}` |

`/abort` is idempotent: aborting a non-existent or already-aborted session returns 200 with `state: "aborted"` so the frontend can call it unconditionally on unmount.

When `current_session.state` is `review`, `/calibration/status` also returns `proposed` and `current` so a reloaded frontend can redraw the review step without re-sweeping.

**Two existing routes are extended:**
- `/estop` calls `calibration_manager.abort()` before disabling torque, so an in-flight session ends cleanly.
- `/arm/<id>/mode` returns 409 `{detail: "arm <id> is being calibrated"}` when `id` is the session's target arm. Other arms are unaffected.

### 4.3 Backend telemetry extension

When a session is active for arm `<id>`, every telemetry frame's `arms[<id>]` block gains a `calibration` field:

```json
"calibration": {
  "state": "sweeping",
  "ticks": {"shoulder_pan": 2050, "shoulder_lift": 1880, "...": "..."},
  "min":   {"shoulder_pan": 790,  "shoulder_lift": 1700, "...": "..."},
  "max":   {"shoulder_pan": 3500, "shoulder_lift": 2400, "...": "..."}
}
```

In `HOMING`, only `ticks` is populated. In `REVIEW` and beyond, no `calibration` block is emitted (the frontend uses the response from `/finish_sweep` instead — REVIEW data is structured, not streaming). The frontend treats absence of the block as "not in a session."

### 4.4 Frontend

**New files:**
- `hmi/frontend/components/CalibrationWizard.tsx` — shadcn `Sheet` from the right, full height; renders one of three step bodies based on session state.
- `hmi/frontend/components/CalibrationStatusCard.tsx` — settings page row: arm id, file status, mtime, Calibrate button.
- `hmi/frontend/lib/calibration.ts` — typed REST client for the six routes.
- `hmi/frontend/__tests__/CalibrationWizard.test.tsx` — vitest + React Testing Library.

**Modified files:**
- `app/settings/page.tsx` — renders `<CalibrationStatusCard>` per configured arm.
- `app/page.tsx` — dashboard banner when any arm has `has_file === false`, with a `Calibrate <arm>` button that opens the wizard.
- `components/ArmPanel.tsx` — when `telemetry.arms[id].calibration` is present, the joint sliders / Home / preset chips are replaced by a thin `CALIBRATING` chip (mirrors the existing teleop-active gating pattern).

**Wizard layout** (one Sheet, body switches by step):

```
┌─────────────────────────────────────────────────────┐
│ Calibrate: right arm                          [✕]   │
├─────────────────────────────────────────────────────┤
│ Step 1 of 3 — Set neutral pose                      │
│                                                     │
│ Move the arm by hand into the pose you want to be   │
│ "0°". Torque is off; the arm is back-drivable.      │
│                                                     │
│ Joint           Ticks                               │
│ shoulder_pan    2048                                │
│ shoulder_lift   2031                                │
│ ...                                                 │
│                                                     │
│ [ Capture neutral ]   [ Cancel ]                    │
└─────────────────────────────────────────────────────┘
```

Step 2 swaps the single-column table for `min | POS | max` and the primary button to `Done sweeping`. Step 3 shows `joint | old → new` for `range_min`, `range_max`, `homing_offset`, with a diff highlight, and `Save` / `Cancel`.

**State source.** Steps 1 and 2 read `telemetry.arms[id].calibration` from the existing `useTelemetryStore` — same stream `ArmPanel` already uses. Step 3 reads the response payload from `/finish_sweep` (structured, not streaming).

**Cancel semantics.** Any Cancel button calls `POST /calibration/{id}/abort` and closes the Sheet. Clicking ✕ or outside the Sheet also aborts; in steps 2 and 3 (where work would be lost), a shadcn `AlertDialog` confirms first.

**Recovery on reload.** On mount, `useEffect` calls `GET /calibration/status`. If `current_session` is non-null, the wizard re-opens at the appropriate step. Backend sessions survive frontend reloads; only `/abort` or `/save` clears them.

**Multi-client.** A second browser that opens the wizard for the same (or any other) arm gets 409 from `/start`. The frontend renders an inline "in progress on another client" notice with a status-polling refresh button.

## 5. Data formats

### 5.1 Calibration JSON (unchanged from lerobot-calibrate)

```json
{
  "shoulder_pan": {
    "id": 1,
    "drive_mode": 0,
    "homing_offset": -1871,
    "range_min": 790,
    "range_max": 3510
  },
  "...": "..."
}
```

Key order is preserved on write (`json.dumps(..., indent=2, sort_keys=False)`) so diffs against pre-existing files stay readable.

### 5.2 Backup filename

`<calibration_id>.json.bak-2026-05-22T14-32-07Z`. UTC, ISO-8601, colons replaced with hyphens so the filename is portable across filesystems. Old backups are never auto-deleted in v1.

## 6. Safety

- **Pre-flight.** `/start` 409s if any configured arm is not in `Mode.MANUAL`. The Calibrate button on the frontend uses the same predicate so the user never sees a spurious failure.
- **Mode lock during the session.** While a session exists, the target arm's `ModeGuard` is held in `MANUAL`; any `/arm/<id>/mode` POST against the target arm 409s with `{detail: "arm <id> is being calibrated"}`. Other arms remain fully functional.
- **Torque.** Torque is disabled on the target arm for the entire session (HOMING and SWEEPING both require back-drivability). Save → reconnect re-enables it. Abort re-enables torque before clearing the session.
- **E-STOP.** `/estop` calls `calibration_manager.abort()` first, then runs the existing torque-disable / mode-STOP path. The wizard auto-closes via the telemetry signal disappearing.
- **Disconnection.** If the bus throws during a sweep tick, the manager sets `session.state = ABORTED` and clears `current`. The next telemetry frame for that arm carries an `alerts: [{level: "error", code: "calibration_bus_error", message, source: "arm:<id>"}]` entry which the frontend can surface.

## 7. Error handling summary

| Situation | Backend | Frontend |
|---|---|---|
| `/start` while another session active | 409 `{detail: "session active for arm <other>"}` | Toast; Calibrate button disabled |
| `/start` while any arm not in `manual` | 409 names the offending arm | Toast; banner names which arm to switch |
| Wrong-state route call (e.g. capture_neutral while SWEEPING) | 409 `{detail: "session is in state <s>"}` | Should not happen (UI gates); toast as failsafe |
| Bus read error during sweep | Session → ABORTED; alert in telemetry | Wizard auto-closes; error toast names the arm |
| Bus disconnect (USB unplugged) | Session → ABORTED; arm marked offline | Same as above |
| `/save` write fails (disk full, perms) | 500 `{detail}`; session stays in REVIEW | Review screen shows the error; Save re-enabled |
| Two clients open the wizard | Second `/start` gets 409 | Inline "in progress on another client" with refresh |
| E-STOP mid-session | Session aborted; existing E-STOP path runs | Wizard auto-closes; E-STOP banner takes over |
| Joint with no motion at `/finish_sweep` | 422 `{detail: "joints with no motion: [...]"}`; state stays SWEEPING | Inline warning lists joints; sweep continues |
| `/abort` on idle/aborted session | 200 `{ok, state: "aborted"}` (no-op) | — |

## 8. Testing

### 8.1 Backend (`hmi/backend/tests/test_calibration.py`, TDD)

Tests mock `SO101Follower` at the same boundary `test_arm.py` already uses, so no hardware is required.

1. `start` from IDLE → state HOMING, torque disabled exactly once on the target arm
2. `start` rejected when another arm isn't in `manual`
3. `start` rejected when a session already exists
4. `capture_neutral` reads ticks once, computes `homing_offset = pos - resolution/2` per joint, transitions to SWEEPING
5. `finish_sweep` from accumulated mins/maxes produces the lerobot-shaped JSON (keys present, types correct)
6. `finish_sweep` rejects when any joint has `min == max` and names the offending joints
7. `save` writes the new file AND moves the old to `<path>.bak-<ts>` (both exist after, contents differ)
8. `save` updates both follower and teleop sibling files when both existed before
9. `save` calls `handle.disconnect()` then `handle.connect()` exactly once each, in that order
10. `abort` from any non-terminal state → ABORTED, torque re-enabled, no file written

### 8.2 Backend routes (`hmi/backend/tests/test_routes.py` additions)

One TestClient case per new route, reusing the existing `app_with_mocks` fixture with a mocked `CalibrationManager`:
- `GET /calibration/status` returns the per-arm list shape
- `POST /calibration/<id>/start` returns 200 from IDLE, 409 on conflict
- `POST /calibration/<id>/capture_neutral` returns 200 from HOMING
- `POST /calibration/<id>/finish_sweep` returns 200 from SWEEPING, 422 on unmoved-joints
- `POST /calibration/<id>/save` returns 200 from REVIEW with `{path, backup_path}`
- `POST /calibration/<id>/abort` returns 200 from any state

### 8.3 Frontend (`hmi/frontend/__tests__/CalibrationWizard.test.tsx`, vitest + RTL)

- Renders step 1 by default; advances to step 2 after `/capture_neutral` mock resolves
- Disables Save in step 3 while `/save` is in flight
- Calls `/abort` exactly once when the Sheet unmounts mid-session
- Shows the AlertDialog confirm when ✕ is clicked in step 2 or 3

### 8.4 Manual smoke (no automated test; documented in `hmi/README.md`)

1. Stop the backend, move the existing `haller_follower.json` aside, restart.
2. Dashboard banner appears: "right arm has no calibration." Click Calibrate.
3. Hand-pose the arm, click Capture neutral.
4. Wiggle every joint; verify the `min | POS | max` table moves; click Done sweeping.
5. Review the diff; click Save.
6. Confirm a `.bak-<ts>` exists next to the new JSON; confirm the live joint sliders work after the auto-reload.
7. Repeat for the leader-as-follower (`haller_leader`) calibration to verify the teleop sibling file also gets written.

## 9. Open questions

None remaining for v1 scope. Paired co-calibration is a known v2 follow-up; the spec for it will land separately once the v1 wizard is in production use.

## 10. References

- Unified HMI spec (parent): [`2026-05-22-haller-unified-hmi-design.md`](./2026-05-22-haller-unified-hmi-design.md)
- Calibration bootstrap helper: `hmi/backend/haller_hmi/calibration_bootstrap.py`
- lerobot calibration primitives: `lerobot/src/lerobot/motors/motors_bus.py` (`set_half_turn_homings`, `record_ranges_of_motion`)
- Existing free-drive UX pattern: `hmi/frontend/components/ArmPanel.tsx`
