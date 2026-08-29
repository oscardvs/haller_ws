# HMI unification — plan of record (2026-08-22)

Orchestrated four-session refactor on branch `refactor/hmi-unify`.
Baseline snapshot: commit `5284be3` (the 2026-08-20 vr-teleop-kit port, exactly as
hardware-tested solo on 2026-08-21). Anything deleted below is recoverable from it —
delete without ceremony.

Orchestrator: the Fable session that authored this file. Work packages:

| WP | Session | Territory |
|---|---|---|
| A | haller-ws-0f | backend core: one input path, legacy deletion, session surgery |
| B | haller-ws-a5 | backend data: cameras runtime control + episode management |
| C | haller-ws-30 | desktop cockpit: prune legacy, new Teleop tab, dataset browser |
| D | haller-ws-1a | in-headset client: Haller HUD wins, absorbs the kit's internals |

## Goal

One teleop system, not three. The Meta Quest drives one or both SO-101 arms through
the vr-teleop-kit internals (clutch mapping + 3+2 decoupled IK) — the only input
path. The **original Haller in-headset UI** (two world-locked quads, grabbable
cluster, opaque two-column menu) is the only VR client — Oscar explicitly prefers it
over the kit's page, which gets deleted. The desktop cockpit is the mission control:
session presets (dual / solo-left / solo-right / sim), a first-class collision-guard
toggle, camera selection (two egocentric + the D455 mast cam), and episode
collection into LeRobot datasets — start/stop/mark from inside the headset without
removing it.

## Decisions of record

1. **Deleted outright** (Oscar's directive — do not keep fallbacks): the MediaPipe
   webcam pipeline, the mouth clutch, the body-angle `"joints"` mode + `BodyModel`
   limb-length system, the wrist-point `"pose"` mode, the pose-reconstruction
   acquisition gate tolerances, the legacy webcam mirror conventions
   (`mirror_mode` / `swap`), and the kit's `/api/vr/` page + `client.js`.
2. **`HumanTeleopSession` survives as the safety core** — per-side authority FSM,
   acquisition ramp, LPF→rate-cap→clamp commit chain, collision-guard integration,
   E-STOP, WS-disconnect grace, tick circuit breaker, `goal_deg` for the recorder.
3. **One teleop socket**: `WS /ws/teleop/vr/in`. It absorbs the relay's extras
   (`config_update`, `request_settings`, `ik_state` push). `/ws/teleop/human/in`
   and the relay's `/vr/ws` + pages die.
4. **Collision guard**: runtime toggle stays exactly as built (off-still-measures,
   `available:false` one-way). It becomes prominent in both UIs with the live
   slack readout. `min_tip_z`/`min_wrist_z` workspace floors stay independent of it.
5. **Servo calibration is NOT legacy.** `calibration.py`, `calibration_bootstrap.py`,
   the wizard, `CalibrateTab`, `/calibration/*` routes: untouched. (The *limb-length*
   calibration dies with `vr_input.py` — do not confuse the two.)
6. **Cameras**: the D455 stays a plain UVC RGB `opencv` source — do NOT add
   `pyrealsense2` or any new dependency (depth is explicitly out of scope; hardware
   budget is €0). New: runtime control of which cameras record, and episode
   management endpoints.
7. **No new frameworks, no redesign of the visual language.** The cockpit's existing
   Haller style (Tailwind v4 tokens `--haller-*`, JetBrains Mono, scanlines,
   label-micro) is the style. New UI must look like it grew there.

## The API contract (the "after" state)

### Surviving REST (unchanged unless noted)

- `GET /health`, `GET /config`, `POST /base/cmd_vel`
- `POST /arm/{id}/goal|mode|home|torque`, `GET/POST/DELETE /arm/{id}/preset*`
- `GET /cameras`, `GET /cameras/{id}/snapshot|stream`
- `POST /estop`
- `GET/POST /teleop` leader-follower trio (untouched)
- `GET /teleop/human` — status; see "status() shape" below
- `POST /teleop/human/start` — **new body: `{left_arm: str|null, right_arm: str|null, hz?: number}`**.
  At least one arm. `swap`, `clutch_source`, and the mouth-calibration gate are gone.
- `POST /teleop/human/stop`, `POST /teleop/human/home`
- `POST /teleop/human/collision {enabled: bool}` — unchanged semantics
- `POST /teleop/sim/*`, `GET /sim/*` (untouched)
- `GET /record/status`, `POST /record/start {repo_id, task}`, `POST /record/stop {save}`
- `GET /calibration/status`, `POST /calibration/{id}/*` (untouched)
- `WS /ws/telemetry` (untouched; its frame keeps the `human_teleop` block)

### Deleted routes

`POST /teleop/human/swap`, `POST /teleop/human/calibrate`,
`POST /teleop/human/mouth/analyze`, `WS /ws/teleop/human/in`,
`GET /vr` + `GET /vr/` + `GET /vr/client.js` + `WS /vr/ws` (the relay router).

### New REST (WP-B implements in a NEW self-contained `APIRouter` module; the
orchestrator mounts it in `server.py` at integration — WP-B does not edit server.py)

- `POST /cameras/{id}/record {record: bool}` → `{id, record}`.
  Runtime toggle of whether this camera lands in recorded episodes. 409 while an
  episode is open (the feature set is frozen at `start_episode`). The recorder
  reads the runtime set, not the config-frozen `record:` flag.
- `GET /record/episodes?repo_id=…` → `{repo_id, episodes: [{index, frames, task,
  length_s}], total_frames, size_bytes}` — read from the dataset meta on disk.
  `repo_id` optional: defaults to the recorder's current/last repo.
- `GET /record/repos` → `{repos: [{repo_id, episodes, size_bytes}]}` — scan of the
  lerobot home (`~/.cache/huggingface/lerobot`).
- `DELETE /record/episodes/last?repo_id=…` → **SHIPPED (WP-B, 2026-08-22)** as an
  in-place pop: `{deleted_index, repo_id, deleted_frames, total_episodes,
  total_frames}`. Never 501. Refuses with 409 when: an episode is open, it is the
  only episode, the metadata row count disagrees with info.json, or the episode's
  video file is shared with an earlier one (foreign dataset). UI degrades on 409,
  not 501. The pop finalizes the recorder's own metadata buffer first (lerobot
  buffers ten episodes in RAM) and recomputes `meta/stats.json` from the
  survivors — `save_episode` folds stats in incrementally, so a popped take
  would otherwise haunt normalisation forever. `meta/tasks.parquet` is
  deliberately untouched (task_index references; an orphan string is free).

  Shipped response-shape extras beyond the original contract (final, per WP-B):
  `GET /record/episodes` adds `root`; `GET /record/repos` adds `root` and
  per-repo `frames`; each `GET /cameras` entry adds `record` (the runtime flag).

### The one teleop socket — `WS /ws/teleop/vr/in`

Inbound (client → server), JSON messages by `type`:
- `"vr_keypoints"` — the existing Haller-client frame shape, MINUS the deleted
  fields (`vr_mode`, `mirror_mode`, `body` limb overrides, mouth/pinch calib).
  No `vr_mode` dispatch remains: every frame feeds `QuestTeleoperator`.
- `"config_update" {…}` — live tuning, clamped by `QuestTeleopConfig`
  BOUNDS/BOOL_KEYS/ENUM_KEYS exactly as the relay did → server replies
  `"config_applied"`.
- `"request_settings"` → server replies `"settings"` (full QuestTeleopConfig).

Outbound (server → client):
- `"ik_state"` at 20 Hz — **same payload shape the relay pushes at baseline**
  (per-side authority/clutch/conditioning/orient_residual etc.). Both WP-A and
  WP-D read the authoritative shape from the baseline:
  `git show 5284be3:hmi/backend/haller_hmi/vr_teleop/relay.py` and
  `git show 5284be3:hmi/backend/haller_hmi/vr_teleop/web/client.js`.
- `"config_applied"`, `"settings"` as above.

Disconnect: the handler calls `notify_ws_disconnected()` only for clients that
actually streamed frames (the relay's `streamed` guard — keep it).

### status() shape (`GET /teleop/human` + telemetry `human_teleop` block)

WP-A finalizes, but MUST keep: `running`, `left_arm`, `right_arm`, `goal_deg`
(the recorder's action column), and per-side `acquire[side].authority` +
`acquire[side].remaining_ms` (read by `QuestTeleoperator.convert` and the HUD
countdown). The `clutch` block shrinks to `{engaged, sides, reason}`. The
`ghost`, `jaw_open`, `t_engage/t_release`, `stale`, `source`, `matched`,
`error_deg`, `tol_deg` keys die.

## Invariants — break any of these and the refactor failed

1. **Zero-error handover**: the mapper re-anchors every frame until a side is
   DRIVING, and once more on the first driving frame. Pinned by
   `test_gate_error_stays_zero_through_the_countdown` and
   `test_handover_starts_from_the_hand_where_it_is_now` — these tests survive.
2. **The acquisition ramp is load-bearing** (`MATCH_DWELL_MS`, `ACQUIRE_RATE_DEG_S`,
   `ACQUIRE_RAMP_MS`, `_ramp_cap`) — the tolerance *gate* dies, the *ramp* does not.
3. `QuestTeleoperator.convert`'s read-back contract (`goal_deg`, `authority`) — §above.
4. Collision toggle semantics: off-still-measures; `available:false` is one-way,
   enabling on it returns 409. Workspace floors (`min_tip_z`/`min_wrist_z`, seeded
   from `server._tip_floor_m()`/`_wrist_floor_m()`) stay ON when the guard is off.
5. E-STOP path untouched: `POST /estop` + B/Y in-headset. The discrete
   `/arm/{id}/home` is REFUSED while a session owns the arms — do not "fix" that.
6. Controller mapping unchanged: per-side grip = dead-man, trigger = gripper
   (1−trigger), B/Y = E-STOP, left-stick hold ≈0.8 s = in-session home, left-stick
   short click on RELEASE = view cycle, A/X hold 500 ms = record toggle.
7. Stance system stays as built (behind/mirror/front, behind = default, det per
   stance is a CHOICE — do not re-litigate; behind stance swaps hand↔arm pairing
   at enterVR, start-time only).
8. Single-arm sessions: absent side never acquires, never written, reports
   `reason:"no_arm"`, cannot be homed.
9. WS-disconnect grace: a disconnect STARTS the window (never re-stamps), a frame
   clears it.
10. Recorder schema and lerobot 0.5.1 pinning: do not migrate lerobot; do not
    change existing feature dtypes/names; one-video-file-per-episode stays.

## Work packages

Rules for every session (non-negotiable):

- Work in `/home/odesha/haller_ws` on branch `refactor/hmi-unify` — verify with
  `git branch --show-current` before the first edit; if it shows anything else,
  STOP and report.
- **No git write operations** (no add/commit/checkout/restore/stash/clean). The
  orchestrator owns git. Read-only git (`log`, `show`, `diff`) is fine.
- **Touch only files in your ownership list.** If a change outside it seems
  necessary, do not make it — record the exact need in your final report.
- Run only your scoped tests during development; the full suites + `vr_smoke.py`
  are the orchestrator's integration pass.
- Backend test incantation (the venv fights you):
  `source ~/venvs/haller-hmi/bin/activate-haller-hmi` (plain `activate` lacks
  rclpy), then
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 MUJOCO_GL=egl python -m pytest -p asyncio <scope> -q`.
  Frontend: `npm test` (vitest) and the repo's typecheck, scoped where possible.
  ~~Known pre-existing flake: 2 `test_recorder.py` failures in full-suite runs~~
  — that diagnosis was wrong. Root cause (WP-B): the venv is
  `--system-site-packages` with no scipy of its own, so the system scipy (built
  for numpy<2) shadowed in against the venv's numpy 2.x and broke every
  real-LeRobotDataset test. Fixed 2026-08-22 by installing scipy into the venv;
  recorder tests now pass everywhere.
- Comments/docstrings that narrate deleted machinery must go with it
  (e.g. `human_teleop.py:1-26,48-69,85-103,582-598`, `server.py:962`).
- Match the house style: terse, reasoned comments that state constraints, not
  history. New UI copy in the cockpit's voice (lowercase labels, label-micro).

### WP-A — backend core (haller-ws-0f)

**Owns:** `hmi/backend/haller_hmi/human_teleop.py`, `server.py`, `safety.py`,
`vr_teleop/**` (incl. deleting `web/` and reworking `relay.py`),
`scripts/vr_smoke.py`, and these tests: `tests/test_human_teleop*.py`,
`tests/test_safety.py`, `tests/test_routes.py`, `tests/test_routes_vr_teleop.py`,
`tests/test_ws_idle_timeout.py`, `tests/vr_teleop/**`, `tests/conftest.py`,
`tests/sim/test_human_teleop_sim.py`.
**Deletes:** `haller_hmi/vr_input.py`, `vr_pose_mode.py`, `retarget.py`,
`vr_teleop/web/`, `tests/test_vr_input.py`, `tests/test_vr_input_sides.py`,
`tests/test_vr_pose_mode.py`, `tests/test_retarget.py`.
**Does not touch:** `recorder.py`, `cameras.py`, `config.py`, `arm.py`,
`motion.py`, `calibration*.py`, `telemetry.py`, `sim/**`, anything frontend.

The work, in dependency order:

1. `safety.py`: delete lines ~211-568 (the entire mouth-clutch half). Keep
   `Mode`/`ModeGuard`/`clamp_joint_goal`/`limit_step`/`step_budget_deg`/
   `check_move_size`/`plan_ramp`. Prune the ~32 mouth tests in `test_safety.py`.
2. `human_teleop.py` surgery per the recon table (the orchestrator's recon lists
   exact line ranges; re-verify against the live file): delete mouth/pinch/mirror/
   ghost/tolerance-gate machinery; collapse `ClutchSource` to the plain
   `dead_man`/`dead_man_sides` booleans; collapse the acquire/frame-age constants
   to the vr_grip values (`ACQUIRE_MS=1000`, `frame_age_ms_loss=700`);
   `_side_goal` collapses to the `joint_goal` dict read; keep everything in
   "Invariants". The session now ingests ONLY converter output (frames carrying
   per-side `joint_goal`).
3. `server.py`: delete the three legacy routes + `/ws/teleop/human/in` + the
   mouth gate in `/teleop/human/start` + the `swap`/`clutch_source` body fields;
   collapse the `/ws/teleop/vr/in` handler to ik-only and absorb the relay
   extras (`config_update`/`request_settings`/`ik_state` @20 Hz push, the
   `streamed` disconnect guard). `relay.py` shrinks to whatever the socket
   handler still needs (the hub/normalizer bits you keep may move into
   `vr_teleop/`; the page-serving routes and `web/` die). Keep
   `_wrist_floor_m`/`_tip_floor_m` and the `QuestTeleopConfig` seeding.
4. Tests: rewrite `_kp_frame` in `tests/test_human_teleop.py` (and the sim copy)
   to build `joint_goal` frames; drop `_fast_acquire`'s four tolerance kwargs;
   delete the ~19 mouth tests + 2 ghost tests + the 3 mirror/tolerance-split
   tests in `test_human_teleop_sides.py` + the 6 mouth-route tests in
   `test_routes.py`; update `conftest.py`'s `app_with_mocks` status stub to the
   new shape; move/rewrite the `/ws/teleop/human/in` case in
   `test_ws_idle_timeout.py` onto the vr socket; extend
   `tests/test_routes_vr_teleop.py` + `tests/vr_teleop/test_relay.py` for the
   unified socket (config_update round-trip, ik_state push, both on
   `/ws/teleop/vr/in`).
5. `scripts/vr_smoke.py`: delete the `"joints"` and `"pose"` sections, renumber,
   keep the ik + recorder + guard-switch sections. Every remaining check passes
   from a cold sim backend (do not run it yourself unless ports are free — note
   in the report if you left it to integration).

**Done when:** backend scope tests green under the incantation; grep finds no
`retarget`, `vr_input`, `vr_pose_mode`, `mouth`, `mirror_mode`, `clutch_source`,
`BodyModel` references in `haller_hmi/` (outside comments that legitimately
describe the past in HANDOVER docs); `server.py` has no `vr_mode` dispatch.

### WP-B — cameras + dataset backend (haller-ws-a5)

**Owns:** `hmi/backend/haller_hmi/cameras.py`, `recorder.py`, `config.py` (camera
dataclass only, if needed), the yaml `cameras:` blocks (only if needed), a NEW
router module `hmi/backend/haller_hmi/routes_data.py`, `tests/test_recorder.py`
(additions), NEW `tests/test_routes_data.py`, `tests/test_config.py` (only if
config changes force it).
**Does not touch:** `server.py` (the orchestrator mounts your router at
integration), `human_teleop.py`, anything frontend.

The work:

1. Runtime record-set: `CameraManager` gains a runtime `record` state per camera
   (initialized from config `record:`), exposed in `list()` so `GET /cameras`
   already reports it, toggled by the new `POST /cameras/{id}/record`. The
   recorder's `start_episode` reads the runtime set instead of the frozen config
   flags. Keep the `dataset_feature_key` collision check against the *runtime*
   set at episode start. 409 the toggle while `_episode_open`.
2. New router (`routes_data.py`, self-contained `APIRouter`, dependencies passed
   in via a `build_router(...)` factory mirroring `relay.py`'s pattern):
   the four endpoints in the contract. Episode listing reads `meta/info.json` +
   `meta/episodes/chunk-*/file-*.parquet` from disk — the v3.0 layout we actually
   write (`episodes.jsonl` is the legacy v2.x path). lerobot starts a NEW
   metadata file on every resume, so any reader must merge all files, not just
   the first (`recorder.read_episode_rows()` does).
3. `DELETE /record/episodes/last`: investigate against lerobot 0.5.1 with our
   one-video-per-episode layout. Implement only with a test proving
   record→delete→record-again leaves a dataset lerobot can still load and resume.
   Otherwise 501 + report.
4. Tests: build your own FastAPI app in-test mounting the router with fakes
   (the existing `test_recorder.py` fakes for telemetry/teleop/cameras are the
   pattern) — do NOT depend on `server.py` mounting.

**Done when:** your test files green; `GET /cameras` carries `record`; the
recorder respects a runtime toggle flipped between episodes; the contract
endpoints answer with the documented shapes.

### WP-C — desktop cockpit (haller-ws-30)

**Owns:** everything under `hmi/frontend/` EXCEPT `components/VRTeleopPanel.tsx`,
`lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`, `app/teleop/vr/**`,
`__tests__/vrTeleop.test.ts` (those are WP-D's).
**Does not touch:** anything backend.

The work:

1. **Delete** (with their tests + knock-on edits — the recon's deletion map):
   `lib/mediapipe.ts`, `HumanTeleopPanel.tsx`, `CameraOverlay.tsx`,
   `PinchCalibrationStep.tsx`, `MouthClutchCalibration.tsx`, `PoseMatchGizmo.tsx`,
   `ScopeBar.tsx`, `cockpit/HumanTab.tsx`, `app/teleop/human/`, the orphans
   (`RecordingPanel.tsx`, `TeleopLauncher.tsx`, `CamerasPanel.tsx`,
   `ui/switch.tsx`, `ui/toggle.tsx`), the pre-cockpit duplicates
   (`ArmPanel.tsx`, `BasePanel.tsx`, `app/arm/`, `app/base/` + their tests;
   `DeepLinkChrome` stays — `/settings` and `/teleop/vr` still use it), and the
   `@mediapipe/tasks-vision` dependency. Knock-ons: `Cockpit.tsx` keep-mounted
   branch, `lib.ts` `"human"` tab + `shouldKeepTeleopMounted`, `CommandBar` hint,
   `TeleopPopover` link, `DeadManIndicator` sources collapse to `vr_grip`,
   `api.ts` legacy calls/types (`humanTeleopCalibrate`, `mouthAnalyze`,
   `humanTeleopSwap`, mouth/pinch/jaw types; `humanTeleopStart` body loses
   `swap`/`clutch_source`).
2. **New "Teleop" tab** replacing the human tab, in the cockpit's own style:
   - Session launcher: preset buttons computed from `/config` arms — Dual,
     Solo left, Solo right (and the sim leader if sim arms exist — stretch);
     stance selector; hz. Calls the new start body.
   - Live session panel: per-side authority/reason chips (reuse
     `DeadManIndicator`), goal vs measured glance, **collision guard as a
     first-class control** — big toggle + live `slack_m` readout +
     `available:false` disabled state with the reason.
   - The headset entry point: show the HTTPS origin URL to open on the Quest
     (`/teleop/vr`), prominent.
   - `SimViewTile` where a sim arm is in play.
3. **Dataset tab grows into the collection workspace**: per-camera "record this"
   toggles wired to `POST /cameras/{id}/record` (disabled while recording);
   episode browser from `GET /record/episodes` (index, frames, task, length) +
   repo picker from `GET /record/repos`; delete-last button that degrades
   gracefully on 501; disk-size readout. Keep `recorderActions` as the one
   start/stop implementation.
4. `api.ts` gains the contract's new calls + types. Where WP-B is not done yet,
   code against the contract shapes in this document.

**Done when:** `npm test` + typecheck green with the legacy files gone; the
cockpit boots with tabs operate/cameras/teleop/dataset/calibrate/settings; no
`@mediapipe` in `package.json`.

### WP-D — the in-headset client (haller-ws-1a)

**Owns:** `components/VRTeleopPanel.tsx`, `lib/vrTeleop.ts`,
`lib/humanTeleopClient.ts`, `app/teleop/vr/page.tsx`, `__tests__/vrTeleop.test.ts`
(+ new test files for your libs), `__tests__/humanTeleopClient.test.ts`.
**Does not touch:** `lib/api.ts` (WP-C owns it — if you need a new REST call,
note it in your report; the socket protocol is yours via `vrTeleop.ts`), anything
backend.

Oscar's directive: **this page's UI won.** The kit's page is being deleted; its
*internals* already drive the backend. Your job is to finish the marriage.

1. **Strip legacy** from the panel + lib: the 3-way `vrMode` select and
   `VRMODE_LS_KEY` (frames no longer send `vr_mode` — every frame takes the ik
   path), the limb-length `<details>` block + `body`/`BodyOverride`/`BODY_LS_KEY`,
   the `mirrorMode` arm-mounting select, the prose branches on
   `vrMode !== "joints"`. `humanTeleopClient.ts` drops its `KeypointFrame`
   default type param (the mediapipe import dies with WP-C's half).
2. **Solo sessions**: kill the ≥2-arms refusal in `app/teleop/vr/page.tsx`; give
   the panel the same preset launcher semantics as the cockpit (dual / solo-left /
   solo-right from the arms actually enabled), including the behind-stance
   pairing swap where both arms exist and its single-arm degenerate case.
3. **Absorb the kit client's genuinely new bits, in Haller HUD style** (read them
   at baseline: `git show 5284be3:hmi/backend/haller_hmi/vr_teleop/web/client.js`):
   - consume the `ik_state` 20 Hz push on the same socket → HUD telemetry
     (per-side conditioning σ_min, `orient_residual` with the haptic buzz on
     orientation deficit — tell the operator to *move*, not twist harder);
   - `config_update`/`request_settings` round-trip → a "tuning" section in the
     HUD menu + desktop panel (wrist pivot (m) default 0.09, gains — whatever
     `QuestTeleopConfig` exposes), clamped server-side;
   - keep A/X hold = record (the kit's precision-modifier idea: only if it finds
     a mapping that doesn't collide; otherwise drop it and say so).
4. **Recording from inside the headset, complete**: the A/X hold toggle exists —
   add the episode counter + frame count on the HUD, a save/discard choice at
   episode end (controller-menu driven, in the existing menu style), and the REC
   state surfaced in haptics/HUD so the operator always knows.
5. Keep every Haller interaction listed in Invariants §6-7 working. The HUD
   stays two world quads + grabbable cluster; camera tile behavior (native-res
   texture, 33 ms throttle, `facing:"operator"` display mirror) unchanged.

**Done when:** your scoped vitest files + typecheck green; the page starts solo
and dual sessions; no `vr_mode`/limb-length/mirror UI remains; ik_state telemetry
renders on the HUD.

## Integration (orchestrator, after all four report)

Mount WP-B's router in `server.py` — its `build_router` takes ZERO-ARG CALLABLES
(`get_cameras=lambda: cameras, get_recorder=lambda: recorder,
lerobot_home=lerobot_home`), load-bearing because server.py mounts routers at
import time but builds `cameras`/`recorder` inside `lifespan` — a router closing
over the values would capture None and 503 forever. Then reconcile any contract
drift; full backend
suite + frontend suite + typecheck; `vr_smoke.py` against a cold sim backend
(`HALLER_HMI_CONFIG=$PWD/config.bimanual-sim.yaml MUJOCO_GL=egl uvicorn
haller_hmi.server:app --port 8077`); update `QUICKSTART-QUEST.md` + memory; commit
per-WP; Oscar does the hardware pass with the Quest.
