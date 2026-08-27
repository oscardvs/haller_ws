# Track C — the Lab frontend, handoff

Written 2026-08-27 by haller-ws-fd. Track C is **complete and holding**.

Everything below is what a successor **cannot reconstruct from the code or the
git log**. What the code already says is deliberately not repeated here.

State at handoff: 11 commits, `tsc --noEmit` 0, eslint clean, production build
clean, `/lab/compare` prerenders.

**Track C owns 150 tests and they are 150/150 measured IN ISOLATION** — run
just the six suites below and that number is a fact about this track:

```
npx vitest run __tests__/labClient.test.ts __tests__/labCharts.test.ts   __tests__/labReview.test.tsx __tests__/cockpitTabs.test.tsx   __tests__/api.test.ts __tests__/cockpitLib.test.ts
```

The FULL-SUITE total is deliberately not pinned here, because on this tree it
is not a property of the code. Four sessions share one working tree, so a count
taken while another track is mid-write is a fact about the tree at that instant
and nothing more. Measured across one afternoon at the same HEAD it read 396,
then 395/1, then 396 three times, then 376/20 — that last one was Track D
editing `__tests__/vrTeleopXRLoop.test.tsx`, which was the only file failing
and was dirty in `git status` at the time.

So before chasing any frontend failure: `git status --short -- hmi/frontend/`.
If the failing file is dirty and is not yours, it is someone typing. Confirm
your own scope with the isolated run above rather than re-running the whole
suite and hoping.

---

## 1. The live-backend recipe

**This is the most valuable thing in this document.** It is how every claim
about the Review surface was verified, and it is not written down anywhere
else. It supersedes any mock.

```bash
# 1. the real backend, in sim, with the /lab router mounted
cd hmi/backend
HALLER_HMI_CONFIG=/home/odesha/haller_ws/hmi/backend/config.bimanual-sim.yaml \
MUJOCO_GL=egl \
~/venvs/haller-hmi/bin/python -m uvicorn haller_hmi.server:app \
    --host 127.0.0.1 --port 8021 --log-level warning

# 2. the frontend, BUILT — see the trap below
cd hmi/frontend
NEXT_PUBLIC_BACKEND_URL=http://127.0.0.1:8021 npx next build
npx next start -p 3993
```

Three things that cost hours to learn:

- **`HALLER_HMI_CONFIG`, not `HALLER_CONFIG`.** The wrong name is not an error;
  it silently falls through to the default config and the server tries to open
  `/dev/haller_arm_uart`, fails to connect to real hardware, and exits 3. The
  error message is about a serial port and says nothing about config.
- **`next dev` renders but never hydrates here.** The Turbopack HMR websocket
  fails with `ERR_INVALID_HTTP_RESPONSE`, so no `useEffect` runs, no data is
  fetched, and every panel sits on its loading state. The page *screenshots*
  correctly, which is what makes it expensive — it reads as a broken backend.
  Always `next build` + `next start`. It also type-checks and catches
  Suspense-boundary errors that dev does not.
- **Screenshots must be written under `/home/odesha/haller_ws/.playwright-mcp/`.**
  The Playwright MCP refuses paths outside the repo root; that directory is
  gitignored (`51844b9`).
- **`.next/` is shared exactly like the git index.** One tree, one build
  output. `next build` mints a new `BUILD_ID` and replaces the chunk
  directory, so building while another session is serving from the same tree
  can leave THEIR running server pointing at chunk paths that no longer exist.
  The page then renders and its JavaScript 404s — which looks identical to the
  `next dev` non-hydration trap above, and will send whoever hits it to the
  wrong cause. So: `ss -ltnp | grep next-server` BEFORE building, not just
  before choosing a port. If someone else is serving, coordinate or wait.
  (Observed 2026-08-27: two sessions on :3993, second `next start` refused with
  EADDRINUSE — which is the safe failure. The unsafe one is the build that
  precedes it and succeeds silently.)

The two real datasets under `~/robot-data/lerobot/local/` are the point of
using a real backend: one is 6-channel single-arm with one camera key, the
other 12-channel bimanual with three. **Anything that silently assumes a rig
passes against one and fails against the other.** Render both.

---

## 2. The `RecordStatus` reconcile — half done, half precisely specified

Run against the live backend above, not reasoned about:

```
GET /record/status returns, today:
  recording, repo_id, task, episode_frames, skipped_frames, started_at,
  last_error, auto_scored, success, success_frames

vs lib/api.ts::RecordStatus
  on the wire but NOT in the type:  none
  in the type but not on the wire:  state, episode_index, invalidated_reason,
                                    fps_measured, fps_declared,
                                    record_rate_gate, drops, alerts
```

All eight are Track A's Phase 1 fields and all eight are **optional**, so the
type is correct against today's backend and against A's.

**The remaining half is exactly one question: did Track A ship those eight
with these names and shapes?** Answer it by re-running the same diff against
A's backend. Do not answer it by reading A's code and reasoning — this type
was already wrong twice this port, both times in ways that type-check:

- `state` was nearly `"rolling"`; A caught it. It must be `"recording"` so
  that `recording === (state === "recording")` keeps holding for every call
  site that already reads the boolean.
- `drops` was nearly a flat `Record<string, number>`. It is nested per camera
  and per arm, because the arms are `left`/`right` and nothing stops a camera
  being named for a side — one key, two meanings, and the panel reports a
  confident wrong number.

The diff script is four lines of Python; regenerate it rather than hunting for
it (`curl /record/status`, parse `RecordStatus` out of `lib/api.ts` with a
regex on `^\s{2}(\w+)\??:`, compare the two key sets).

---

## 3. What is NOT verified — do not round this up

**Review is verified against the real backend and the real 707 MB dataset**:
the player, the packed-mp4 seek, the `to − 1/fps` clamp, the trace and gripper
charts, marking, filters, the keyboard triage.

**Train and Compare are verified only against a mock I wrote.** The types match
what Track B is building to and B confirmed no drift — but that is a claim
about types, not about a rendered page. Every bug the real backend found in
Review was a shape the types allowed (`trace.gripper` was a LIST where the
contract line implied a map; it threw on *every* real trace).

When `/lab/runs` lands, do to it exactly what was done to the player: build
against the live backend, open the Train tab, select a real run, and watch the
metric grid, the log tail and the checkpoint list draw. Expect to find
something.

The mock lived in a session scratchpad and is **gone**. It is not worth
rebuilding now that `/lab` is mounted — the real backend is strictly better.

---

## 4. The rate gate — the reading half needs no change

`RECORD_RATE_GATE_FALLBACK = 0.9` in `lib/api.ts` is named FALLBACK so it
cannot be mistaken for the source. `recordRateGate(status)` reads
`status.record_rate_gate` when the backend publishes it and falls back
otherwise; both paths are pinned by tests.

**The source is now real:** `MIN_RATE_FRACTION = 0.9` in
`haller_hmi/safety.py:33`, published by Track A. The number has an owner.

The trap, pinned at that constant's definition site and in
`docs/port/phase2-tick-contract.md`: the gate is a ratio of measured to
**declared** rate, so it only means what it says while `fps_declared` is
exactly the `fps` written into `info.json`. Let those two come apart and the
ratio stays 0.9 while quietly meaning something else — the declared-not-
measured defect re-entering through the machinery built to stop it. When
reconciling the constant, check the *referent*, not just the number.

---

## 5. Invariants that are load-bearing and easy to break silently

Each of these was a decision, not a preference. Changing one is a decision too.

- **The best/final table reads RAW points, never smoothed ones.** A smoothed
  minimum is a value the run never reached, and "best loss 0.071" has to be a
  number that actually happened. The compare page's smoothing slider moves the
  curve, not the figures under it. `components/lab/ComparePane.tsx`.
- **The eval split is never recomputed in the browser.** It comes from
  `GET /lab/datasets/split`. Two implementations of "which episodes does the
  trainer not see" drift, and when they do the `val` badges lie about which
  demonstrations the policy has already learned — the one error on this
  surface that cannot be spotted by looking at it.
- **`sliceFor()` has NO fallback and returns null.** Guessing that an episode
  starts at 0 opens episode 6 at second 0 of an 82-second file and plays five
  other takes under the right label. Silence is the only safe answer to a
  missing slice.
- **The clamp is `to − 1/fps`, not an epsilon.** `to_timestamp` is EXCLUSIVE;
  the last real frame is one frame period before it. Track B bisected this on
  the real files: at 30 fps a 0.01 s epsilon lands *past* the last frame, and
  the last episode of every file — 7 of 46 — shows a blank frame instead of the
  take. `fps` is a REQUIRED prop on `EpisodePlayer` so an assumed 30 cannot
  compile.
- **Gripper guides come from `trace.gripper[].closed_below/open_above`.** Those
  are the exact floats `grade.py` graded with, index-aligned to the column.
  There is deliberately no second path — a `gripperGuides()` helper reading the
  same pair off `arms[]` was deleted for being one. When the backend sends no
  thresholds, draw NONE: an invented threshold looks like a measurement.
  Measured: 40.0/70.0 on the kit dataset, 34.13/67.20 on the degree-calibrated
  bimanual one.
- **`episode_index`, never `index`.** LeRobot's v3.0 parquet carries both and
  `index` is the GLOBAL FRAME index — episode 1's first three frames are
  `episode_index [1,1,1]`, `frame_index [0,1,2]`, `index [855,856,857]`. A
  field called `index` here reads correctly and means something else.
- **Every episode shows `Ep 4` beside a muted `idx 3`.** 1-based in
  conversation, 0-based in storage; both on screen so a disagreement is a
  visible artefact rather than a silent one.
- **Sorting and filtering are server-side.** A 70-seed campaign is thousands of
  episodes and the list is paged; a browser sort orders one page and calls it
  the answer. The shift-range selection follows the VISIBLE order for the same
  reason — a range walking numeric indices behaves correctly on a list sorted
  by index and betrays the operator the first time they sort by duration.
- **`PaneBoundary` is not decoration.** A subordinate widget must not be able
  to unmount the workspace. It caught two unrelated failures an hour apart —
  one a sloppy test fixture, one a genuine contract mismatch — and in both
  cases cost a chart instead of the Data tab. Do not remove it because
  "nothing throws any more".

---

## 6. Testing conventions

- **Mutation-check every test that guards an absence.** A confirm gate is an
  absence assertion in disguise and goes vacuous silently. Break the thing,
  confirm the test goes red, restore. One of my own anchor tests was found
  vacuous this way — it now documents what it *cannot* detect and why
  (`__tests__/labReview.test.tsx`, "keeps ranging from a live anchor").
- `routeFetch()` in `__tests__/cockpitTabs.test.tsx` is the house pattern:
  route by path fragment, collect calls, assert POST bodies verbatim.
- jsdom has no `scrollIntoView`, `ResizeObserver` or `IntersectionObserver`.
  Stub in the test file, not in the component — the guard would be dead code
  in every browser that matters.

---

## 7. Git, on a shared tree

One working tree, one `.git`, four sessions. Anything any session stages sits
in the same index every other session commits from.

```
git add <paths>            # in the same breath as the commit, never before
git commit -F <msgfile> -- <explicit paths>
git show --stat HEAD       # confirm exactly the files you meant
```

**Never `git stash` on this tree.** It empties other tracks' working trees.

---

## 8. Territory

`hmi/frontend/**` **except** the in-headset VR client, which is Track D's:
`components/VRTeleopPanel.tsx`, `lib/vrTeleop.ts`, `lib/humanTeleopClient.ts`,
`app/teleop/vr/**`, and the `vrTeleop*` tests.

`lib/api.ts` is Track C's. Track D requests additions rather than editing it —
that is what stopped `recordStop`'s new `rearm` field from silently becoming
required and breaking two shipped desktop surfaces that call it as `{save}`.

`~/vr-teleop-kit` and `tests/equivalence/**` are read-only to everyone.
