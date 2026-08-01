# Handover — apply the v3 cockpit redesign to the Haller HMI

You are picking up a finished design and shipping it into the live app. The
design work is done and signed off. Nothing about it is up for renegotiation
unless you find it contradicts the running code — in which case the code wins
and you flag it.

---

## 1. Where the design lives and how to read it

It is a Claude Design project, not a file in this repo.

- Project id: `2180c554-cd75-47dc-97f4-29703b099b05` ("Haller HMI redesign")
- The file to implement: **`Haller HMI v3.dc.html`**
- Also present: `Haller HMI - Current.dc.html` (a recreation of the *old* HMI —
  useful only for diffing), `Haller HMI v2.dc.html` (superseded, ignore),
  `github.md` (the designer's own screen→file map), `support.js` (generated
  runtime, irrelevant).

Read it with the `DesignSync` tool:

```
DesignSync method=list_files projectId=2180c554-cd75-47dc-97f4-29703b099b05
DesignSync method=get_file  projectId=2180c554-cd75-47dc-97f4-29703b099b05 path="Haller HMI v3.dc.html"
```

`get_file` returns JSON and the file is ~115 KB, so it will land in a persisted
tool-output file rather than your context. Decode it to disk first, then read it
in chunks:

```python
import json
d = json.load(open("<persisted-output-path>"))
open("/tmp/hmi_v3.html", "w").write(d["content"])
```

It is ~1520 lines: markup to about line 775, then a `<script type="text/x-dc">`
block with the logic.

**It is not React.** It is a bespoke template dialect: `<x-dc>` root,
`<sc-if value="{{ flag }}">`, `<sc-for list="{{ xs }}" as="x">`, `{{ }}`
interpolation, inline `style` strings, a `style-hover` attribute, and a
`class Component extends DCLogic` with `state` plus a single `renderVals()` that
returns every interpolated value and handler. Treat `renderVals()` as the spec
for behaviour and the markup as the spec for layout. Translate; do not transpile.

Note the `hint-placeholder-*` attributes are authoring hints for the design tool.
They are not part of the design. Ignore them.

---

## 2. What you are building

A single fixed-viewport cockpit replacing today's scrolling multi-page dashboard.

Shell: `height:100vh`, `overflow:hidden`, CSS grid with four rows —
`44px` header / `26px` telemetry rail / `minmax(0,1fr)` content / `34px` command bar.

Six tabs in the header, all rendered inside the content band:

| Tab | Replaces |
| --- | --- |
| Operate | `app/page.tsx` dashboard + `BasePanel` + both `ArmPanel`s |
| Human teleop | `app/teleop/human/page.tsx` + `HumanTeleopPanel` |
| Calibrate | `CalibrationWizard` + `CalibrationStatusCard` |
| Cameras | `CamerasPanel` |
| Dataset | `RecordingPanel` + a take-composition grid |
| Settings | `app/settings/page.tsx` |

Plus: a bottom command bar with Teleop and Record popovers, an alerts popover
hanging off the rail, and a toast stack.

---

## 3. Decisions already made

**Routes.** Build the cockpit at `/`. Keep `/base`, `/arm/[id]`, `/settings` and
`/teleop/human` on disk and working as unlinked deep links; just remove them from
the header nav. This is reversible and nothing breaks if something still links
there. *(This was Oscar's call to make and he didn't object across two asks — if
he now says delete them, delete them; it's a small change either way.)*

**Colour.** The design already expresses everything as role tokens with the same
names as `app/globals.css` (`--card`, `--muted-foreground`, `--haller-live`, …).
Do not introduce literals. A handful of new ones appear — `--haller-chrome`,
`--haller-inset`, `--haller-thumb` — add those to `globals.css` in both themes.

**Theme switching.** The design uses `[data-theme="dark|light"]`. The repo uses a
`.dark` class (`@custom-variant dark (&:is(.dark *))` in `globals.css`, and
`<html className="dark">` hardcoded in `layout.tsx`). **Keep the repo's class
convention** and adapt the design to it. `next-themes` is already a dependency
and currently unused — wire it up with `attribute="class"`, `defaultTheme="dark"`,
and suppress the hydration warning on `<html>`.

---

## 4. Repo orientation

- Frontend: `hmi/frontend` — Next.js App Router, Tailwind v4, shadcn, zustand,
  sonner, vitest.
- **Read `hmi/frontend/AGENTS.md` first.** This is not the Next.js you know; it
  tells you to consult `node_modules/next/dist/docs/` before writing code. Do that.
- Backend: `hmi/backend` — FastAPI. `config.yaml` is live hardware;
  `config.solo-sim.yaml`, `config.bimanual-sim.yaml`,
  `config.leader-follower-sim.yaml` are MuJoCo-backed and let you exercise the
  whole UI with no hardware attached.
- Scripts: `npm run dev`, `npm run build`, `npm run lint`, `npm test` (vitest).
  There is **no CI typecheck** in this repo — run `npx tsc --noEmit` yourself.
- Existing tests in `hmi/frontend/__tests__/` cover `BasePanel`, `JointSlider`,
  `EStopButton`, `CalibrationWizard`, `HumanTeleopPanel`, `ScopeBar`,
  `DeadManIndicator`, `MouthClutchCalibration`, and the api/telemetry clients.
  **They must still pass.** Where a component is restructured, update its test
  rather than deleting it.

---

## 5. Translation rules

- Reuse components wherever the design's shape still matches — `CameraTile`,
  `JointSlider`, `ModeToggle`, `EStopButton`, `DeadManIndicator`, `ScopeBar`,
  `PinchCalibrationStep`, `MouthClutchCalibration`, `CameraOverlay`,
  `SimViewTile`. Restyle them; don't rewrite them from the mockup.
- All the mockup's data is hardcoded. Every value must come from the real
  sources: `lib/api.ts` (REST), `lib/telemetry.ts` (zustand over websocket),
  `lib/calibration.ts`.
- The mockup's `setInterval(…, 100)` clock, its fake tick/frame counters, and its
  invented `diag`/`calTicks`/`calMin`/`calMax` tables are scaffolding. Real values
  come from telemetry.
- Inline styles → Tailwind classes + the token vars, matching how the existing
  components are written. Keep `label-micro` / `label-tracked` / `scanlines` /
  `corner-frame` utilities.

---

## 6. UX and reactivity — the part that matters

This is a supervisory control surface for a robot. Reactivity is a safety
property here, not polish.

**Telemetry.** `lib/telemetry.ts` is a zustand store fed by a websocket at 20 Hz
(`telemetry.hz` in `config.yaml`). Subscribe with **primitive selectors**
(`useTelemetry(s => s.lastFrame?.base.linear)`), never whole frames in leaf
components — the existing code has comments explaining why React's snapshot cache
needs this. A 20 Hz full-tree re-render will make the joint sliders unusable.

**Link state — three states, not two.** The store today exposes only
`connected: boolean` and silently reconnects after 1 s. The design needs
`Live` (green, pulsing) / `Reconnecting` (amber, no pulse) / `Disconnected` (red,
no pulse), with a detail string shown inline in the rail when down, and the
command-bar hint switching to explain that readouts are frozen and commands will
fail. Extend the store: track `lastFrameAt` and the retry state, and derive.
Everything numeric must fall back to em-dashes rather than showing stale values
as if they were live.

**Boot state.** Before `GET /config` resolves, the content band shows a centred
loading state naming the calls in flight. Don't render an empty cockpit.

**Per-arm silence.** If `lastFrame.arms[id]` is absent, that arm's card shows
"awaiting telemetry…" and **hides the mode segmented control** — matching what
`ArmPanel.tsx` does today. When the link is down the text becomes
"no telemetry — link down".

**Joint sliders must be optimistic.** Telemetry reports measured position at
20 Hz. If the thumb binds directly to telemetry the operator fights the feed
while dragging. Hold commanded position in local state, POST on change, and
reconcile on release. The design deliberately shows *commanded only* — there is
no second marker for measured, because the backend publishes one number per
joint. Don't reintroduce it.

**Keyboard scoping — there is a live bug here, fix it during the port.**
`BasePanel` binds `window` keydown/keyup for WASD with **no check on the event
target**, and `RecordingPanel` has a text input on the same page today. Typing a
task description containing "w", "a", "s" or "d" drives the base. In the cockpit:

- Ignore key events originating from `input`, `textarea`, `select`, or
  `contenteditable`.
- Drive keys are live only on the Operate tab.
- The spacebar dead-man is live only on the Human teleop tab, and must still
  `preventDefault()` so it doesn't scroll or re-trigger a focused button.

**Tab switching must not tear down live sessions.** This is the biggest
structural trap. `HumanTeleopPanel` owns the webcam, three MediaPipe models, and
the ~30 Hz publish loop that *is* the teleop input. If tabs are plain
mount/unmount and the operator clicks Operate mid-session, the robot stops
receiving poses while the backend still thinks a session is running. Keep that
panel mounted and hidden when a session is active (or hoist the runner above the
tab switch). Conversely the base's 10 Hz `cmd_vel` interval *should* stop when
Operate is not visible.

**Popovers.** One at a time — the design models this as a single `pop` slot; keep
that. Escape closes, click-outside closes, focus returns to the trigger.
`:focus-visible` outlines are specified in the design and must survive.

**Toasts.** Use the existing sonner `<Toaster />`, don't rebuild the design's
stack. Every command path already toasts on success and failure — preserve that.
Match the design's placement (bottom-right, above the command bar).

**Motion.** Respect `prefers-reduced-motion` for the ping, E-STOP pulse and
recording blink. The design doesn't mention it; add it.

**Responsive.** Below ~1180 px wide, the Operate tab drops to a single arm with a
picker above the card. Below ~720 px tall, wrist camera tiles collapse to their
label strip so the joint stack and action row keep their space, and the command
bar says so. Content regions own their own `min-h-0` and internal scrolling — the
page itself never scrolls.

---

## 7. Safety semantics that must survive the port

Verify each against the code before you write it. Do not infer these.

- **E-STOP** is always enabled, always full-opacity red, and pulses when
  `alerts.length > 0`. It is never visually disabled.
- **Calibration** cannot start unless every arm is in `manual` and no session is
  in progress. The blocked reason is surfaced on the button.
- **Cancelling calibration** at step 2 or 3 must confirm first, naming what is
  discarded. Step 1 stays instant.
- **Backend-ended calibration** (bus error, arm unplugged) is a distinct state
  from an error string — it says the session block disappeared, that nothing was
  written, and that the fix is the cable.
- **Clutch authority cannot change mid-session** — `disabled={running}` in
  `HumanTeleopPanel`, with a comment explaining the backend forces a disengage on
  a source switch.
- **Mouth clutch** needs talk + open captures ≥ 0.25 apart before it will arm.
- There is a **confidence floor gating teleop**. Read the current value out of the
  code — do not take any number from this document, from the design, or from
  memory. It has been stated wrongly before.

**One new rule the design invented:** it refuses to start recording unless human
teleop is running ("nothing would be logged as action"). Today's `RecordingPanel`
does not enforce this. It is defensible — the recorder logs the teleop's
commanded targets as `action` — but it is a behaviour change. Check whether the
backend already rejects it; if not, ask Oscar whether he wants a hard block or a
warning.

---

## 8. Traps — already verified, don't re-derive

- **`threequarter_sim` already exists** in all three sim configs
  (`config.solo-sim.yaml`, `config.bimanual-sim.yaml`,
  `config.leader-follower-sim.yaml`). The designer's own todo list claims it is
  "only in the frontend's picker" and needs adding to the backend YAML — **that is
  wrong**. Verify, then skip that item.
- **The "Preview states" row in Settings is scaffolding.** The designer added a
  labelled row of buttons to fake link-down / config-loading / arm-silence / bus
  -error so the states could be previewed. Wire those states to real signals and
  then **delete the row entirely**. It must not ship.
- **Sim PiP.** The Human teleop tab pins a sim picture-in-picture bottom-left with
  hide/show. It must use `SimViewTile.pickSimCamera`, which prefers an id
  containing `threequarter` (the overhead camera flattens away `shoulder_lift`
  and `elbow_flex` — exactly the joints you watch while teleoping). It must not
  cover the calibration cards or the dead-man chip; `SimViewTile` already carries
  comments about that collision.
- **Light-theme status colours.** The designer darkened `--haller-live`,
  `--haller-warn` and `--haller-manual` for light mode. `globals.css` currently
  carries an explicit decision that status colours keep the *same* hex in both
  themes "so semantics never shift". These conflict. Pick one, and update the
  comment in `globals.css` to match whichever you pick. Check contrast either way.
- **Joint limits** in the design match `sim/assets/so101/so_arm100.xml` exactly
  (all six, radians → degrees). Camera ids match `config.yaml`, including
  `left_wrist` correctly marked as a reserved slot that only exists as an
  `ArmPanel` fallback. These were checked — trust them.
- **Do not touch calibration data.** The 2026-07-27 calibration has two joints
  recorded across the full tick range. The Calibrate tab makes this more visible;
  that is a hardware task for Oscar, not something to "fix" in the UI.

---

## 9. Running it

The sim configs exercise the entire cockpit with no robot attached — arms,
cameras, telemetry, teleop. Start the backend with one of them and
`npm run dev` in `hmi/frontend`.

For the Human teleop tab specifically: `getUserMedia` needs a secure context, so
drive it from `localhost`, not a LAN IP.

---

## 10. Definition of done

- `npm run build`, `npm run lint`, `npm test`, and `npx tsc --noEmit` all clean.
- Every tab renders against a sim config with real data — no hardcoded values
  left from the mockup.
- Link-down, booting, and arm-silent states reachable and correct **without** the
  Preview-states row, which is gone.
- Theme toggle persists across reload, both themes legible.
- Keyboard: WASD does not leak into text inputs; dead-man works on the teleop tab
  and nowhere else.
- Switching tabs during an active human-teleop session does not interrupt it.
- Existing tests pass or are updated with the component they cover.

---

## 11. Repo conventions

- **Never** add `Co-Authored-By:` trailers. Not in the body, not in the footer.
  This is in `CLAUDE.md` and is absolute.
- Land work directly on `main` as `feat(hmi):` / `fix(hmi):` commits. No PR flow.
- **Never `git add -A` or `git add .`** — there is concurrent untracked work from
  parallel sessions in this tree. Stage explicit paths and verify with
  `git diff --cached --name-only` before committing.
- If an unrelated untracked file rides along anyway, surface it — don't reset or
  amend it away.
- Expect commits from other sessions to interleave with yours.
- Commit in reviewable slices (shell + tokens, then a tab at a time). Don't land
  1500 lines in one commit.
