# Quest bimanual teleop — quickstart

Drive both SO-101 arms from a Meta Quest in passthrough AR: you see the real
arms through the headset, each grip is that arm's dead-man, and a capsule
model of both arms runs inside the 60 Hz loop so you cannot command them into
each other or through the bench.

The whole pipeline is exercised end-to-end in sim by `scripts/vr_smoke.py`
(38 checks, run against a live backend). What has **not** happened yet is a
human driving the real arms from the headset — do the *first hardware run*
checklist below before trusting it.

---

## Rehearse on the desktop first (no hardware at all)

```bash
# desktop, repo root
scripts/quest-teleop/up.sh --sim
```

Identical chain — same page, same HTTPS origin, same backend code — except
the backend runs **on this machine against MuJoCo arms**
(`config.bimanual-sim.yaml`). Put on the headset, open the printed URL, and
drive: engagement countdown, per-grip dead-men, the collision guard, E-STOP —
all real, nothing physical that can move. **The MuJoCo workbench camera
floats inside the HUD** (it defaults on whenever the backend's base camera is
a sim render — passthrough shows your room, and the sim arms live only in
that tile; `Cam off` hides it). The desktop cockpit (`/`) shows the same
view. This is the recommended way to learn the controls and check your
limb-length settings before the first hardware run.

## Collecting datasets in sim

The sim chain records the same LeRobot datasets the real rig will — same
recorder, same schema — so headset hours produce training data even with no
arms powered.

Two scenes, picked at launch:

| launch | scene | scored by |
|---|---|---|
| `up.sh --sim` | cubes on a mat — pick, hand over, place | `cube_placed` |
| `up.sh --insertion` | steel bracket + pin — one arm holds, one inserts | `pin_inserted` |

Insertion is the task being collected now, and it is bimanual *structurally*:
the bracket is a free body, so inserting without steadying it just shoves it
across the bench, and the predicate refuses to score a one-handed solve. Read
`docs/setup/insertion-collection.md` before recording it — that is where the
**frozen** instruction string and the 70-seed list live.

1. **Draft the take on the desktop cockpit** (`/` → Dataset tab): the task
   string (one instruction per dataset) and your HF username. The draft
   persists in the browser, so the headset can start takes from it. Paste the
   frozen insertion string *exactly* — LeRobot keys tasks by string, so a
   retyped variant silently splits the dataset into two conditioning groups
   that look like one.
2. **Bring the sim up**: `scripts/quest-teleop/up.sh --insertion` (or
   `--sim`), open the printed URL in the Quest browser, **Enter Passthrough**.
3. **Deal the scene before each take**, one seed per episode off the list —
   thirty layouts teach more than thirty repeats of one:

   ```bash
   curl -sX POST localhost:8000/sim/scene/reset \
     -H 'content-type: application/json' \
     -d '{"seed": 1001, "randomize": true, "home_arms": true}'
   ```

   `home_arms` is refused while a take is open. Reset, *then* record.
4. **Record from inside the headset**: hold **A or X ~0.5 s** to start the
   take — both controllers buzz and the HUD shows `● REC <frames>`. Drive the
   task. Hold A/X again to stop **and save**. Repeat per episode; the draft
   stays put. (Discard stays a cockpit-only action on purpose: a thumb brush
   must never be able to throw a take away.) Fumbled it? Hit **discard take**
   on the cockpit — a failure is not training data. **Takes under 2 frames
   are refused outright**, and `last_error` says so: LeRobot cannot
   compute video stats over a one-frame episode, and the ragged metadata that
   leaves behind takes the *whole dataset* down with it, not just the stray
   take.
5. **Keep `GET /sim/task/status` in view.** On insertion it reports `depth_m`,
   `lateral_m`, `tilt_deg`, `pin_held` and `fixture_held` — so a take that
   looked fine but did not score tells you *which* clause missed instead of
   leaving you guessing.
6. Takes land in `~/.cache/huggingface/lerobot/<hf_user>/<task-slug>` —
   **30 fps, h264**, with `observation.state` + `action` (12-dim, left-then-
   right SO-101 order), `observation.base`, `observation.wall_clock`
   (**seconds since the episode started** — LeRobot's own `timestamp` column
   is synthetic, and a float32 quantises a 2026 epoch to 128-second steps;
   the absolute start is in info.json's `haller_wall_clock` block), and the
   three recorded camera channels (`top` + a wrist close-up per arm — five
   render, three record). Each episode gets **its own video file per
   camera**: LeRobot 0.5.1's packer corrupts the second episode it appends to
   a file, so it is never allowed to pack. The cockpit's `skipped` counter
   next to the frame count tells you a take had gaps; the wall-clock deltas
   tell you where. An **E-STOP mid-take is fine**: the recorder sees the
   session die and saves up to that frame instead of appending a torque-off
   tail.
7. **End with `scripts/quest-teleop/down.sh`, and let it finish.** The dataset
   is unreadable until the backend's shutdown has written the parquet footers
   and `meta/episodes/`; the script waits for exactly that and prints while it
   does. Killing it there destroys the session's takes.
8. **Replay a take onto the sim arms** to eyeball what you actually drove
   (loops until stopped):

   ```bash
   curl -X POST http://localhost:8000/teleop/sim/start \
     -H 'Content-Type: application/json' \
     -d '{"follower": "left", "leader": {"source": "replay",
          "dataset_path": "~/.cache/huggingface/lerobot/<hf_user>/<task-slug>"}}'
   # ...and "follower": "right" for the other arm; /teleop/sim/stop to end.
   ```

`scripts/vr_smoke.py` covers this whole path headlessly — including the
record→save→files-on-disk round trip and the E-STOP-mid-take save — run it
after any backend change that could touch recording.

## Start everything (each session, real arms)

```bash
# desktop, repo root
scripts/quest-teleop/up.sh          # backend on the Jetson
scripts/quest-teleop/up.sh --local  # backend on THIS desktop — arms plugged
                                    # straight in, no Jetson (see
                                    # docs/setup/desktop-real-weekend.md)
```

That checks/starts the backend on the Jetson (over `ssh jetson`), starts the
frontend on :3001 with the right baked-in URL, starts Caddy as the single
HTTPS origin, verifies the chain, and prints the URL:

```
https://192.168.0.191:8444/teleop/vr
```

Open it **in the Quest browser**, accept the self-signed cert once, press
**Enter Passthrough**.

`scripts/quest-teleop/down.sh` stops the desktop half: frontend and Caddy
first, then — only if the backend is a local one `up.sh --sim/--insertion/
--local` started — SIGTERM, and **it waits up to 120 s for that backend to
finalise any recorded dataset**. Let it. It prints `still finalising — do NOT
kill it` while that runs, and a Ctrl+C there is precisely how a session's
takes get destroyed. (The Jetson backend is never stopped from here — it owns
the arms.)

If `up.sh` cannot reach the Jetson: power the rig, then either plug the USB-C
(device-mode ssh) or wait for wifi (192.168.0.124), and on the Jetson run
`~/haller_ws/scripts/quest-teleop/backend-jetson.sh` by hand. To survive
reboots instead, install `scripts/quest-teleop/haller-hmi-backend.service`
(instructions in the file).

### One-time: get this branch onto the Jetson

```bash
ssh jetson 'cd ~/haller_ws && git fetch origin && git checkout feat/quest-bimanual-teleop'
# then restart the backend (it runs from this checkout)
```

## Controls

| input | effect |
|---|---|
| **left grip** (squeeze) | dead-man for the **left arm** only |
| **right grip** | dead-man for the **right arm** only |
| **trigger** (analog) | that arm's gripper — 0 open, 1 closed |
| **B or Y** (either controller) | **E-STOP**: torque off both arms, session stops |
| **hold A or X** (~0.5 s) | start / stop-and-save a dataset take — REC state + frame count show in the HUD |
| **left stick click** | next camera view in the HUD tile |
| **hold left stick** (~0.8 s) | **reset arms to home** (0°, gripper open) — only sides with the grip open; a driving hand always wins |
| **right stick click** | next tile size (S/M/L) |
| **point at the HUD + trigger** (grip open) | **grab and move the HUD** — drag it anywhere, it faces you and the spot persists |
| release a grip | that arm freezes exactly where it is |
| take off headset / open Quest menu | frames force-disengage; arms freeze |

The HUD is two separate floating panels: the **camera tile** (updates at
display rate, native resolution) with the **status/menu panel below it** —
instructions never cover the view. Grab either one to move the pair.

**Position mode (default).** Squeezing a grip *anchors*: your hand's current
position is bound to the arm's current pose, so there is nothing to match —
hold still through the 1 s countdown, feel the buzz, and from then on your
hand's movement drives the gripper tip through IK on the robot's own
kinematics. Release to freeze; move your hand somewhere comfortable and
squeeze again to ratchet across the workspace. Controller pitch/roll steer
the wrist relative to where it was anchored. No limb-length calibration is
involved. (The *hand mapping* selector still offers the legacy body-angle
mode; expect it to fight you — the SO-101's shoulder barely pitches below
horizontal and its elbow folds the opposite way to yours, which is why
position mode exists.) All motion rides the same rate limiter (60 °/s
ceiling at the handle, less during the first 1.5 s ramp).

**Operator stance** (panel selector, shown in the HUD menu) picks how your
hand maps onto the gripper AND which arm each controller drives — in the
egocentric default your right controller drives the arm that appears on the
right of the over-shoulder view (set when the session starts; changing
stance mid-session needs an exit/re-enter). The default is **egocentric** and pairs with the
default `overshoulder` view: goggles on, the tile shows the arms from behind
— push your hand forward and the replica extends INTO the scene, move right
and it goes frame-right. The replica arm moves exactly like your own, which
is why it is the default. *Mirror* is for facing the real arms across the
bench (the arm as your reflection: push away = it extends toward you, hands
together = arms cross) and pairs with the `threequarter` view. *Match the
camera tile* makes motion agree with the front view's screen axes. If the
arm ever drives the opposite way from your hand, the stance is wrong — fix
it here, not with the *arm mounting* selector (that one is for genuinely
mirrored mounts).

**No arm-length calibration exists in position mode, on purpose.** The
squeeze anchors your hand to the arm wherever both are, and only *deltas*
drive it, so limb lengths cancel out — the surveyed VR-teleop stacks
(Open-Teach, MoveIt-Pro Quest, BEAVR, TeleVision) all skip it for the same
reason. Your reach only bounds how far one drag can go before you ratchet.
The *operator limb lengths* inputs on the panel affect the legacy body-angle
mode only.

The HUD floats over passthrough: per-side authority + countdown, grip state,
live collision clearance, a red `● REC <frames>` line while a take rolls, an
E-STOP button you can click with the controller ray, and any backend error.

## The collision guard

`collision.py` sweeps four capsules per arm along the real kinematics and
filters every commanded step: steps that keep ≥ `margin_m` (25 mm) clearance
pass; steps that *improve* a bad pose always pass (escape is never blocked);
steps that would close in get scaled back and stop **at** the margin. Height
floors keep fingertip/wrist/elbow off the bench. The gripper is never frozen
by the guard. When it bites you feel a light buzz and the HUD shows
`◉ COLLISION HOLD`.

Config lives in `hmi/backend/config.yaml` under `collision:`. The mount
positions are **the sim scene's ±0.20 m and have not been measured on the
tower** — do the checklist below before trusting mm-level margins. `enabled:
false` turns the guard off entirely (it will say so in the HUD).

## First hardware run — 10 minutes, in this order

1. **Bench clear, DC-DCs on, second person or desktop cockpit at the E-STOP.**
2. **Mount geometry.** Measure base-plate bolt to bolt between the arms; set
   `collision.mounts` x to ±half that (and `table_z_m` if the bench surface
   isn't the mount plane). Restart the backend after editing.
3. **Clearance readout sanity.** On the VR page (no headset needed — the same
   card is on the 2D page), with the session started but grips open: torque
   off both arms from the cockpit and move them toward each other **by
   hand**. The clearance number must shrink as they approach and go negative
   just before they touch. If it doesn't move, the mounts are wrong.
4. **Direction check, one arm at a time.** In the headset, squeeze ONE grip,
   match the pose, and nudge your hand 5 cm outward. The arm must move
   outward. If it moves **opposite** your hand, flip the *arm mounting*
   selector on the panel (identical ↔ mirrored) — this rig should be
   `identical, side by side`. If the **wrong arm** moves, the left/right
   naming is crossed: swap the two `mounts` entries AND the arm order, or
   physically relabel.
5. **Gripper + speed feel.** Trigger through its range; then drive a slow
   reach. If anything feels fast, lower `motion.max_speed_deg_s` (global or
   per-arm) — the teleop path obeys it at the handle level.
6. **Provoke the guard once, gently.** Drive both hands slowly toward the
   centre; the arms must stop with a buzz before they meet, HUD shows
   `COLLISION HOLD`, and pulling back out must work instantly.
7. Only then: real speed, real manipulation, recording.

## Small manip, then out

Pick a cube, hand it over between arms, set it down. Release grips (arms
freeze), press Exit, take the headset off. The Recorder panel on the desktop
cockpit records datasets while a teleop session is driving.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| "WebXR is not available" | Page not HTTPS, or cert not accepted. `navigator.xr` is simply absent over http — no prompt explains it. |
| Enter does nothing / start refused | Another teleop session (cockpit, sim) holds the arms — stop it first. Needs **2 enabled arms**. |
| Countdown won't finish | HUD `match:` lists the blocking joints; move to the robot's pose. `match: gripper` → squeeze trigger ~90 %. |
| Arm moves opposite the hand | Wrong *operator stance* (egocentric / mirror / camera-tile) on the panel. If only ONE arm is reversed, it's the *arm mounting* parity — see step 4 above. |
| Workspace tile looks left/right-flipped | The active camera faces you (tower mast cam). It should auto-mirror (`facing: operator` in config.yaml); if a camera moved, update its `facing`. |
| An arm "randomly" freezes | Its controller left the Quest's tracking view (HUD: `no tracking`), or its grip slipped below the press threshold. Re-squeeze, re-match. |
| `COLLISION HOLD` where nothing is close | Mounts in `config.yaml` don't match the real plate — measure (step 2/3). |
| E-stopped, want to continue | **Re-arm arms** button on the VR page (sets MANUAL + torque), then Enter Passthrough again. |
| `down.sh` sitting on "still finalising" | It is writing the dataset — parquet footers, videos, `meta/episodes/`. **Wait** (up to 120 s). Ctrl+C there is what leaves a session's takes unreadable, which is why the script never SIGKILLs the backend. |
| Stopped a take, nothing was saved | Under 2 frames, so the recorder refused it — `/record/status` → `last_error` says so. A one-frame episode makes the *whole dataset* unfinalisable, so it is dropped instead. |
| 502s on `/_next/webpack-hmr` in caddy log | Next dev hot-reload through the proxy. Cosmetic. |
| Right arm intermittent `Incorrect status packet` | Known UART flakiness under telemetry; watch whether it worsens under teleop load. |
| Everything up but headset can't load page | Quest must be on the same wifi (`ZTE_DEC155`); check `https://192.168.0.191:8444/api/health` from any browser. |

## What runs where

| piece | host | started by |
|---|---|---|
| FastAPI backend (arms, guard, retarget) | Jetson | `backend-jetson.sh` / systemd unit |
| Next.js frontend :3001 | desktop | `up.sh` |
| Caddy single origin :8444 | desktop | `up.sh` (config: `scripts/quest-teleop/Caddyfile`) |

Deeper background: `hmi/HANDOVER-2026-08-01-jetson-rig.md` (rig state),
`haller_hmi/collision.py` and `vr_input.py` docstrings (how and why).
