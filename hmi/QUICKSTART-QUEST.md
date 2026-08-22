# Quest teleop — quickstart

Drive the SO-101 arms from a Meta Quest in passthrough AR: you see the real
arms through the headset, each grip is that arm's dead-man, and a capsule
model runs inside the 60 Hz loop so you cannot command them into each other
or through the bench.

The whole pipeline is exercised end-to-end in sim by `scripts/vr_smoke.py`
(50 checks, run against a live backend). What has **not** happened yet is a
human driving the real arms from the headset — do the *first hardware run*
checklist below before trusting it.

---

## Just want one arm on the bench tonight?

```bash
# desktop, repo root — ONE real arm, no Jetson
scripts/quest-teleop/up.sh --solo
```

`config.solo-real.yaml`: one arm, one hand, and the collision guard **off**
by default (see [The collision guard](#the-collision-guard) for why, and for
what stays on regardless). Open the printed URL in the Quest browser, pick
**Single arm** and the arm under your right hand, press Start, then Enter VR.

A session no longer needs two arms. Either hand can be left without one: its
controller is simply ignored and nothing is ever written to that side.

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

### Over Tailscale, when the LAN won't carry it

```bash
scripts/quest-teleop/up.sh --tailscale              # composes with the rest:
scripts/quest-teleop/up.sh --insertion --tailscale
```

Same single origin, moved off the LAN onto the tailnet, on :8445 with a real
certificate — no interstitial to accept. It prints the URL, e.g.
`https://odesha.tail4f2a4b.ts.net:8445/teleop/vr`.

Sign the Quest into the same tailnet first (sideload the Tailscale Android
APK); until you do, that name resolves to nothing in the headset browser.
Nothing in `tailscale serve` is touched — Caddy just binds the tailnet address
with a cert from `tailscale cert`.

With real arms on the Jetson, only the headset's hop moves: the desktop still
reaches the Jetson over wifi. If `/api` times out, put the Jetson on the
tailnet too and pass `HALLER_JETSON_IP=<jetson>.<tailnet>.ts.net`.
`--sim`/`--insertion` avoid the hop entirely.

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
| **A or X**, *relay page only* | **precision** — both gains drop while held, for fine work |
| **left stick click** | next camera view in the HUD tile |
| **hold left stick** (~0.8 s) | **reset arms to home** (0°, gripper open) — only sides with the grip open; a driving hand always wins |
| **right stick click** | next tile size (S/M/L) |
| **point at the HUD + trigger** (grip open) | **grab and move the HUD** — drag it anywhere, it faces you and the spot persists |
| release a grip | that arm freezes exactly where it is |
| take off headset / open Quest menu | frames force-disengage; arms freeze |

The HUD is two separate floating panels: the **camera tile** (updates at
display rate, native resolution) with the **status/menu panel below it** —
instructions never cover the view. Grab either one to move the pair.

**Two headset pages exist, and A/X means different things on each.** The
cockpit page (`<origin>/teleop/vr`) is the one above: A/X held is the dataset
take toggle. The relay page (`<origin>/api/vr/`) is the ported one — no
recorder, so A/X is free and is the precision modifier there. Both drive the
same backend through the same converter; pick the cockpit page when you are
recording, the relay page when you are tuning.

**Hand pose mode (default).** Squeezing a grip *anchors*: your hand's current
pose is bound to the arm's current pose, so there is nothing to match — hold
still through the 1 s countdown, feel the buzz, and from then on your hand
drives the gripper. Release to freeze; move your hand somewhere comfortable
and squeeze again to ratchet across the workspace. No limb-length calibration
is involved. All motion rides the same rate limiter (60 °/s ceiling at the
handle, less during the first 1.5 s ramp).

Three things this mode does that are worth knowing at the bench:

* **Position and orientation, not position plus pitch/roll gains.** Your
  hand's full 6-DoF pose is mapped; the arm's three position joints track the
  point and its two wrist joints track as much of the orientation as two axes
  can. When you ask for a twist the wrist physically cannot reach — a yaw,
  usually, since with the tool where it is the gripper's yaw is decided by
  the shoulder — the controller buzzes and the HUD says so. **Move your hand
  instead of twisting harder.**
* **Pushing past the arm's reach feels like a wall, not a wind-up.** The
  target can never run more than 12 cm ahead of where the arm actually is,
  and the excess is *absorbed*. Reversing bites after at most that 12 cm,
  however far past the wall you pushed. The cost is that absorbed travel is
  gone, so hand↔gripper correspondence drifts — re-clutch to realign, which
  is the ratchet you are doing anyway.
* **Hold A or X for precision.** Both gains drop (0.4× by default) while it
  is held, for fine work. It re-anchors on press and release, so it never
  reinterprets motion you already made under a new gain.

The *hand mapping* selector keeps two older modes: **hand position** (the
previous wrist-point mode — 3-DoF position with controller pitch/roll passed
through on fixed gains) as a fallback if the default misbehaves at the bench,
and **body angles** (legacy) which will fight you — the SO-101's shoulder
barely pitches below horizontal and its elbow folds the opposite way to
yours, which is why neither of the other two copies joint angles.

**Tuning it live.** The relay page at `<origin>/api/vr/` has sliders for the
gains, the reach limits, the smoothing and the IK damping, applied on the
next solver tick and clamped server-side. Inside the headset, **click a
thumbstick** to open the same list on the HUD panel and push the stick to walk
it and change values. If the arm feels *sluggish* rather than badly mapped,
the binding limit is usually `motion.max_speed_deg_s` in the config (60 °/s),
not anything on that panel.

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
tower** — do the checklist below before trusting mm-level margins.

### Switching it off

It is a **runtime** switch, not a restart: the *collision guard* selector on
the VR panel and the relay page's Safety card, or

```bash
curl -X POST <origin>/api/teleop/human/collision \
     -H 'content-type: application/json' -d '{"enabled": false}'
```

`config.solo-real.yaml` starts with it off, because with one arm there is no
arm-vs-arm case at all and the mounts are still a guess — and a guard
reasoning about millimetres from geometry nobody has measured is exactly what
clamps an arm that is plainly nowhere near anything.

**Off still measures.** The clearance number keeps updating on the HUD, so
you can drive with the guard off, watch it, and switch the guard on once the
mounts are measured and the number behaves.

**What stays on regardless:** the teleop's own workspace floor (the commanded
fingertip and wrist can never be asked to go below the bench), the per-joint
limits, the acquisition ramp, the per-tick rate caps and the motion envelope.
Turning off the bimanual guard does not remove the bench.

It cannot be switched on at all on a rig with no mounts configured for every
arm — a guard with no geometry for an arm would pass every check for it,
which is the one failure this module exists to prevent. The panel shows
`unavailable` in that case rather than a toggle that does nothing.

## First hardware run — 10 minutes, in this order

**One arm first.** Steps 2, 3 and 6 are about two arms not hitting each
other, so on a single-arm run skip them and run the guard off
(`--solo` already does). Everything else applies unchanged, and step 4 is
the one that matters most.

1. **Bench clear, DC-DCs on, second person or desktop cockpit at the E-STOP.**
2. **Mount geometry.** *(two arms)* Measure base-plate bolt to bolt between the arms; set
   `collision.mounts` x to ±half that (and `table_z_m` if the bench surface
   isn't the mount plane). Restart the backend after editing.
3. **Clearance readout sanity.** *(two arms)* On the VR page (no headset needed — the same
   card is on the 2D page), with the session started but grips open: torque
   off both arms from the cockpit and move them toward each other **by
   hand**. The clearance number must shrink as they approach and go negative
   just before they touch. If it doesn't move, the mounts are wrong.
4. **Direction check, one arm at a time.** In the headset, squeeze ONE grip,
   hold still through the 1 s countdown, then nudge your hand 5 cm outward.
   The arm must move outward. If it moves **opposite** your hand, the
   *operator stance* is wrong for where you are standing — fix it there
   (egocentric if you are behind the arms, mirror if facing them). If the
   **wrong arm** moves, the hand↔arm pairing is crossed: on the relay page
   the selectors are labelled by hand, so just pick the other arm.
   Then check all three axes before trusting any of them: hand up must be
   gripper up, hand forward must be the arm extending.
5. **Gripper + speed feel.** Trigger through its range; then drive a slow
   reach. If anything feels fast, lower `motion.max_speed_deg_s` (global or
   per-arm) — the teleop path obeys it at the handle level.
6. **Provoke the guard once, gently.** *(two arms, guard on)* Drive both hands slowly toward the
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
| Enter does nothing / start refused | Another teleop session (cockpit, sim) holds the arms — stop it first. One arm is enough since 2026-08-20 (`arms` selector on the panel, or `--solo`). |
| Twisting the wrist does nothing and the controller buzzes | The demand is off the arm's reachable orientations — with the tool where it is, the gripper's yaw is set by the shoulder. Two wrist axes cannot cover three; move your hand instead. |
| Arm stops short and pressing harder does nothing | The reach limit is absorbing (that is the wall). Release, reposition your hand, squeeze again — the ratchet is how you cross the workspace. |
| Changed the headset page and nothing happened | Was the cockpit page, not the relay page — or a stale cache. The relay serves `no-store`; the Next.js page needs a real reload. |
| Tracking feels sluggish, not wrong | `motion.max_speed_deg_s` (60 °/s) is the binding limit, below both the IK step cap and the session rate cap. Raise it deliberately, with a hand near the E-STOP. |
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
| Page loads on the desktop, headset can't reach it **at all** | **Check the radio before you blame the router.** `ZTE_DEC155` is ONE SSID on TWO BSSIDs — `10:3C:59:DE:C1:55` (ch 6, 2.4 GHz) and `10:3C:59:E0:C1:55` (ch 44, 5 GHz) — and this router does not bridge clients across them. Split across radios both devices get a 192.168.0.x lease and reach the internet, and cannot see each other at all: `ip neigh` to the Quest goes `FAILED` while the router stays `REACHABLE`, and `tcpdump` sees zero packets from the headset — indistinguishable from AP isolation. Band is re-chosen on every association, so this appears and disappears with no config change. Fix: pin the desktop to the headset's radio, `nmcli con modify ZTE_DEC155 802-11-wireless.bssid <BSSID>` then `nmcli con up ZTE_DEC155` (undo with `bssid ""`), and test with `ping` + `ip neigh` — try both BSSIDs, it is a two-shot experiment. Only if BOTH fail is it really isolation: then `up.sh --tailscale` or a hotspot. The gateway's MAC is the same on both radios, so it does **not** tell you which one you are on — `nmcli -f IN-USE,SSID,BSSID,CHAN,FREQ dev wifi list` does. |
| Desktop silently leaves the hotspot mid-session | Every saved network defaults to autoconnect-priority 0, so NetworkManager can defect back to the house wifi and take the origin with it — Caddy is left holding a socket bound to an address no longer on any interface, and nothing reaches it from anywhere. `nmcli con modify <hotspot> connection.autoconnect-priority 20` before you start recording. |
| Page loads, but nothing ever connects | The dev server has an old `NEXT_PUBLIC_BACKEND_URL` compiled in — Turbopack caches it in `.next/dev` and a restart alone does **not** clear it. `up.sh` detects and fixes this now; if it says the baked URL is UNKNOWN, run `down.sh` then `up.sh`. |

## What runs where

| piece | host | started by |
|---|---|---|
| FastAPI backend (arms, guard, retarget) | Jetson | `backend-jetson.sh` / systemd unit |
| Next.js frontend :3001 | desktop | `up.sh` |
| Caddy single origin :8444 LAN / :8445 tailnet | desktop | `up.sh` (config: `scripts/quest-teleop/Caddyfile`) |

Deeper background: `hmi/HANDOVER-2026-08-01-jetson-rig.md` (rig state),
`haller_hmi/collision.py` and `vr_input.py` docstrings (how and why).
