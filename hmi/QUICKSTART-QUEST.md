# Quest teleop — quickstart

Drive the SO-101 arms from a Meta Quest in passthrough AR: you see the real
arms through the headset, each grip is that arm's dead-man, and a capsule
model runs inside the 60 Hz loop so you cannot command them into each other
or through the bench.

The whole pipeline is exercised end-to-end in sim by `scripts/vr_smoke.py`
(**49 checks, all passing** against a cold backend — including the `ik_state`
push, live tuning, and the record→save→files-on-disk round trip).

One arm *has* been driven on real hardware, on 2026-08-21. **The unified
build has not.** On 2026-08-22 the three input paths collapsed into one, the
kit's second headset page was deleted, and — see the box below — the rule
deciding which hand drives which arm was corrected. Do the
[first hardware run](#first-hardware-run--10-minutes-in-this-order)
checklist before trusting any of it.

> ### ⚠ Which hand drives which arm changed. Check it first.
>
> The pairing used to be read off the **order arms are declared** in the
> config. `config.yaml` declares `[right, left]`; every sim config declares
> `[left, right]` — so the same stance meant opposite things on the two, and
> the rule was right in sim and inverted on the tower. It now reads the arm
> **id**, so `left` is the arm bolted on the left wherever it appears in the
> file.
>
> That is a fix, but it is a fix **nothing has driven yet**: on the real rig
> the behind-stance pairing is now the opposite of what was hardware-tested,
> for the dual session *and* for `--solo`. Step 4 of the hardware checklist
> is exactly this test, and it is one grip and a 5 cm nudge. Do it before
> anything else moves.
>
> `hmi/HARDWARE-PASS-2026-08-22.md` is the full list of what only a live
> headset can settle, this first.

---

## Just want one arm on the bench tonight?

```bash
# desktop, repo root — ONE real arm, no Jetson
scripts/quest-teleop/up.sh --solo
```

`config.solo-real.yaml`: one arm, one hand, and the collision guard **off**
by default (see [The collision guard](#the-collision-guard) for why, and for
what stays on regardless). Open the printed URL in the Quest browser, press
**Enter Passthrough**. With one arm enabled the launcher has already picked
the only session there is (`solo left`); `dual` is drawn beside it, greyed,
with the reason.

In the default egocentric stance that arm answers your **right** hand — it is
named `left` because that is the side of the robot it is bolted to, and
standing behind it you are looking at it from the same side it reaches from.
**That is reversed from the 2026-08-21 hardware run**, so expect it and check
it on the first squeeze.

A session does not need two arms. Either hand can be left without one: its
controller is ignored, nothing is ever written to that side, and that side
reports `no arm this side` rather than sitting silently in `held`.

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
view. This is the recommended way to learn the controls before the first
hardware run — and the only way to rehearse the recording flow without
putting junk episodes in a real dataset.

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
4. **Record from inside the headset, start to finish.** Hold **A or X
   ~0.5 s** to start the take — both controllers buzz and the HUD shows
   `● REC ep <n> · <frames> fr`, where `<n>` is the index this episode will
   actually land at in the dataset, read off the dataset meta. Drive the task.

   Hold A/X again to **end** it, and the menu box turns into the decision:

   ```
   TAKE ENDED · 612 frames
   L stick click = SAVE
   R stick click = DISCARD
   hold A/X = keep rolling
   still rolling until you pick
   ```

   That last line is the honest part and it is worth reading once: the
   backend's `/record/stop` takes the save-or-discard decision **at stop
   time**, so there is no way to close the episode first and choose
   afterwards. The recorder keeps running while you decide, and the tail of a
   saved take is however long you spent holding still. Decide in a second or
   two, or hold A/X again to withdraw the question and carry on recording.
   Both stick clicks belong to the prompt while it is up — no view cycling,
   no tile resizing, no accidental home.

   A failure is not training data: **discard it**. You no longer have to take
   the headset off to do that, and the cockpit's Dataset tab keeps its own
   `discard take` plus a delete-last-episode button for one you only realise
   was bad afterwards. **Takes under 2 frames are refused outright**, and
   `last_error` says so: LeRobot cannot compute video stats over a one-frame
   episode, and the ragged metadata that leaves behind takes the *whole
   dataset* down with it, not just the stray take.
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
ssh jetson 'cd ~/haller_ws && git fetch origin && git checkout main'
# then restart the backend (it runs from this checkout)
```

The unified teleop landed on `refactor/hmi-unify`; check that branch out
instead until it merges. The Jetson must be on the **same** commit as the
desktop — the frontend and the backend agreed on a wire shape that changed
on 2026-08-22, and a headset talking to a pre-unification backend gets a
socket that ignores half of what it sends.

## Picking the session

The panel's **session** row is the launcher: `dual`, and one `solo <arm>`
button per arm the backend has enabled. A preset the rig cannot offer is
drawn greyed with the reason rather than hidden — "there is no second arm" is
a fact about the robot, and a picker that quietly drops the option makes you
doubt your memory. Beside the buttons the panel prints the pairing the
selected one will actually post:

```
session   [ dual ]  [ solo left ]  [ solo right ]
— L hand → right · R hand → left
```

That caption is rendered from the very pairing that goes in the start body,
not recomputed for display, so a button cannot describe one mapping and start
another. It changes when you change stance, which is the point: the stance is
what decides it.

**Which arm each hand gets** comes from the arm's **id**, not from where it
sits in the config file. Standing behind the arms you face the way they
reach, so the arm bolted on the robot's left is the one under your **right**
hand — the pairing crosses. Facing them (mirror / front) it does not. A solo
session puts its arm on the hand the dual session would have given it, so
dropping an arm never changes the shape of the mapping. See the box at the
top of this file: this is the rule that changed on 2026-08-22.

The whole session lives in the headset from here — presets, stance, the
collision guard, tuning and recording all have an in-VR surface. The desktop
cockpit's **Teleop** tab is the same controls on a screen, for when someone
else is watching the bench.

## Controls

| input | effect |
|---|---|
| **left grip** (squeeze) | dead-man for the arm paired to your **left hand** only |
| **right grip** | dead-man for your **right hand's** arm only |
| **trigger** (analog) | that arm's gripper — 0 open, 1 closed |
| **B or Y** (either controller) | **E-STOP**: torque off both arms, session stops |
| **hold A or X** (~0.5 s) | start a take; on an open take, **end it and ask save/discard** |
| **push the left stick away, hold** | **precision** — both gains drop (0.4× by default) while held, for fine work |
| **left stick click** | next camera view in the HUD tile — or **SAVE**, while the end-of-take prompt is up |
| **hold left stick** (~0.8 s) | **reset arms to home** (0°, gripper open) — only sides with the grip open; a driving hand always wins |
| **right stick click** | next tile size (S/M/L) — or **DISCARD**, while the end-of-take prompt is up |
| **hold right stick** (~0.5 s) | open / close the **tuning list**; its stick then walks and adjusts it |
| **point at the HUD + trigger** (grip open) | **grab and move the HUD** — drag it anywhere, it faces you and the spot persists |
| release a grip | that arm freezes exactly where it is |
| take off headset / open Quest menu | frames force-disengage; arms freeze |

Every button was already spoken for by a dead-man, a safety action or the
recorder, which is why precision and the tuning list live on the **sticks**.
A/X stays the record toggle: an accidental take boundary is corrupted data.

The HUD is two separate floating panels: the **camera tile** (updates at
display rate, native resolution) with the **status/menu panel below it** —
instructions never cover the view. Grab either one to move the pair. The menu
box has three faces: the view/bindings list normally, the tuning list while
that is open, and the save/discard decision, which takes the whole box.

**There is one headset page now**, `<origin>/teleop/vr`. The kit's ported page
at `/api/vr/` is gone; its internals are what the surviving page drives, and
everything it could do — live tuning, the precision modifier, the settings
round-trip — has been folded in above.

## How the mapping works

Squeezing a grip *anchors*: your hand's current pose is bound to the arm's
current pose, so there is nothing to match — hold still through the **fixed
~1 s countdown**, feel the buzz, and from then on your hand drives the
gripper. There is no pose-match gate any more; the mapper re-anchors every
frame until authority transfers, so the error it would have measured is zero
by construction and the countdown is the only wait. Release to freeze; move
your hand somewhere comfortable and squeeze again to ratchet across the
workspace. All motion rides the same rate limiter (60 °/s ceiling at the
handle, less during the first 1.5 s ramp).

Three things worth knowing at the bench:

* **Position and orientation, not position plus pitch/roll gains.** Your
  hand's full 6-DoF pose is mapped; the arm's three position joints track the
  point and its two wrist joints track as much of the orientation as two axes
  can. When you ask for a twist the wrist physically cannot reach — a yaw,
  usually, since with the tool where it is the gripper's yaw is decided by
  the shoulder — **the controller hits you with one firm buzz and the HUD
  says `wrist is out of twist — MOVE your hand`.** That buzz fires once, on
  the moment you cross into the deficit, not continuously; it is telling you
  to reposition, and twisting harder at it does nothing at all.
* **Pushing past the arm's reach feels like a wall, not a wind-up.** The
  target can never run more than 12 cm ahead of where the arm actually is,
  and the excess is *absorbed*. Reversing bites after at most that 12 cm,
  however far past the wall you pushed. The cost is that absorbed travel is
  gone, so hand↔gripper correspondence drifts — re-clutch to realign, which
  is the ratchet you are doing anyway.
* **`σ` on a driving side is how much room the arm has to work with.** It is
  the smallest singular value of the position Jacobian, in m/rad: how far the
  tool moves, in its worst direction, per radian of joint travel. Median is
  about 0.045 and the best pose reaches 0.082; **below ~0.02** the arm is
  heading into the singular set, where the solver damps hard and the arm
  feels syrupy in one particular direction (the desktop panel turns the
  number amber there). It is not an error —
  it is the number that tells you *why* a pose feels bad, and the answer is
  usually to bring the elbow off straight.

**Tuning it live.** Hold the **right stick** ~0.5 s to open the tuning list
on the HUD panel; the same stick then walks it (pull back / push away) and
changes the selected value (left / right), one step per deliberate push. Every
value except the wrist pivot goes to the backend as a `config_update` and
comes back **clamped** — a row that snaps to a different number is the robot
telling you what it actually took, not a UI bug. The list carries the eleven
knobs you reach for mid-session:

| knob | what it does |
|---|---|
| translation / rotation gain | hand→tool scale; rotation ships above 1:1 because the wrist_roll span exceeds a human wrist's |
| precision factor | what the two gains are multiplied by while precision is held |
| reach limit (m) / twist limit (rad) | the absorbing wall, above; 0 disables |
| pose smoothing | EMA on the incoming controller pose — lower is smoother and laggier |
| step cap arm / wrist (°) | per-solve joint step, split so fine wrist work is not held back by the arm's cap |
| IK damping, singularity ramp | how hard the solver damps, and where it starts |
| **wrist pivot (m)** | see below |

The **wrist pivot** (default 0.09 m) is the one client-side knob: it slides
the read-out point back along the controller onto roughly where your wrist
actually turns. Without it a pure twist swings the grip point through an arc
that the mapper can only read as translation you never asked for — so if
twisting still drags the gripper sideways, this is the number to move. It
persists per headset.

The desktop panel's **live tuning** section has the same knobs plus the ones
you set once against a bench measurement (posture damping and bias, wrist
damping, and the two workspace floors). It is live only while a session is
running, because the config belongs to the socket: the values reset with it.
The floors are deliberately not in the in-headset list — they bound the
commanded demand and keep working when the collision guard is off, which is
not a thing to nudge with a thumbstick mid-take.

If the arm feels *sluggish* rather than badly mapped, the binding limit is
usually `motion.max_speed_deg_s` in the config (60 °/s), not anything on that
list.

**Operator stance** (panel selector, shown in the HUD menu) picks how your
hand maps onto the gripper AND which arm each controller drives — see
[Picking the session](#picking-the-session). It is set before the session
starts; changing it mid-session needs an exit and re-enter to re-pair the
hands, and the panel says so when you try. Within one browser the cockpit and
the teleop page share the choice — set it in either and the other follows,
including a second tab. **The Quest keeps its own**, which is right: the
stance is a property of where the operator is standing, and the person in the
headset is not standing where the desktop is.

The default is **egocentric** and pairs with the default `overshoulder` view:
goggles on, the tile shows the arms from behind — push your hand forward and
the replica extends INTO the scene, move right and it goes frame-right. The
replica arm moves exactly like your own, which is why it is the default.
*Mirror* is for facing the real arms across the bench (the arm as your
reflection: push away = it extends toward you, hands together = arms cross)
and pairs with the `threequarter` view. *Match the camera tile* makes motion
agree with the front view's screen axes. **If the arm drives the opposite way
from your hand, the stance is wrong for where you are standing** — that is
the only knob for it now; the old *arm mounting* parity selector is gone
along with the webcam-era mirror conventions it belonged to.

**There is no arm-length calibration, on purpose.** The squeeze anchors your
hand to the arm wherever both are, and only *deltas* drive it, so limb
lengths cancel out — the surveyed VR-teleop stacks (Open-Teach, MoveIt-Pro
Quest, BEAVR, TeleVision) all skip it for the same reason. Your reach only
bounds how far one drag can go before you ratchet. The limb-length inputs
that used to sit on the panel belonged to the body-angle mode, and both are
gone.

The HUD floats over passthrough: per-side authority + countdown, grip state,
each driving side's `σ`, live collision clearance, a red
`● REC ep <n> · <frames>` line while a take rolls, `◆ PRECISION` for as long
as the modifier is held, an E-STOP button you can click with the controller
ray, and any backend error. Nothing about the session is hidden behind taking
the headset off.

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
the VR panel, the first-class toggle on the cockpit's **Teleop** tab (with the
live clearance beside it), or

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
4. **Direction check, one arm at a time. This is the step that changed.**
   In the headset, squeeze ONE grip, hold still through the 1 s countdown,
   then nudge your hand 5 cm outward. The arm must move outward.

   * Arm moves **opposite** your hand → the *operator stance* is wrong for
     where you are standing. Egocentric if you are behind the arms, mirror if
     facing them.
   * The **wrong arm** moves → the pairing is crossed. Read the caption under
     the session buttons (`L hand → right · R hand → left`): it is what the
     session was actually started with. On the real rig this pairing is now
     the **reverse** of what was hardware-tested before 2026-08-22, because
     the rule stopped reading config declaration order and started reading the
     arm id — so this is the one thing on the checklist that is genuinely
     unverified. If the caption is right and the arms still cross, the stance
     convention is what needs revisiting, not the ids.

   Then check all three axes before trusting any of them: hand up must be
   gripper up, hand forward must be the arm extending. On `--solo` do this
   too: the same rule decides which hand your one arm answers to.
5. **Gripper + speed feel.** Trigger through its range; then drive a slow
   reach. If anything feels fast, lower `motion.max_speed_deg_s` (global or
   per-arm) — the teleop path obeys it at the handle level.
6. **Provoke the guard once, gently.** *(two arms, guard on)* Drive both hands slowly toward the
   centre; the arms must stop with a buzz before they meet, HUD shows
   `COLLISION HOLD`, and pulling back out must work instantly.
7. Only then: real speed, real manipulation, recording.

Everything above is the safety floor. `hmi/HARDWARE-PASS-2026-08-22.md`
carries the rest of the first live session — the input bindings that have
only ever been exercised in sim (precision on the left stick, the tuning
list, the save/discard prompt) and what to report back about each.

## Small manip, then out

Pick a cube, hand it over between arms, set it down. Release grips (arms
freeze), press Exit, take the headset off.

You do not have to take it off to record: draft the task once on the
cockpit's Dataset tab, then start, end, save and discard takes entirely from
the controllers (see step 4 of
[Collecting datasets in sim](#collecting-datasets-in-sim) — the flow is
identical on real arms). The cockpit's Dataset tab is where you go afterwards,
to browse the episodes, check the frame counts and pop a bad last take.

## Troubleshooting

| symptom | cause / fix |
|---|---|
| "WebXR is not available" | Page not HTTPS, or cert not accepted. `navigator.xr` is simply absent over http — no prompt explains it. |
| Enter does nothing / start refused | Another teleop session (cockpit, sim) holds the arms — stop it first. One arm is enough (pick its `solo <arm>` preset, or `--solo`). |
| One firm buzz, then the twist stops responding | The demand is off the arm's reachable orientations — with the tool where it is, the gripper's yaw is set by the shoulder, and two wrist axes cannot cover three. The HUD says `wrist is out of twist`. **Move your hand**; twisting harder cannot help, and the buzz fires once rather than nagging. |
| Arm stops short and pressing harder does nothing | The reach limit is absorbing (that is the wall). Release, reposition your hand, squeeze again — the ratchet is how you cross the workspace. |
| Changed the headset page and nothing happened | Stale cache. There is one page now and Next.js serves it — a real reload in the headset browser, not a re-entry into VR. |
| Tracking feels sluggish, not wrong | `motion.max_speed_deg_s` (60 °/s) is the binding limit, below both the IK step cap and the session rate cap. Raise it deliberately, with a hand near the E-STOP. |
| Countdown won't finish | It is a fixed ~1 s and there is nothing to match — the mapper re-anchors until authority transfers. A countdown that restarts means the side keeps going untracked (HUD: `no tracking`), or the grip is slipping below the press threshold. |
| Arm moves opposite the hand | Wrong *operator stance* (egocentric / mirror / camera-tile) on the panel — that is the only mapping knob now. |
| The **wrong arm** answers your hand | The pairing. Check the caption under the session buttons; it is what the session was started with. Post-2026-08-22 this is reversed on the real rig from what was last hardware-tested — see the box at the top and checklist step 4. |
| Gains feel halved and the arm lags | The precision modifier is engaged — the left stick is pushed away. The HUD shows `◆ PRECISION` whenever it is. Let the stick centre. |
| A tuning value snaps back to a different number | Working as intended: the backend clamps every knob to its own bounds and echoes what it took. The list shows the robot's value, not your ask. |
| Pure wrist twists drag the gripper sideways | The read-out point is off your actual wrist pivot. Raise or lower **wrist pivot (m)** in the tuning list (default 0.09) until a twist stays put. |
| Ended a take and nothing saved yet | The save/discard prompt is up and the recorder is **still rolling**. Left stick click saves, right discards, A/X withdraws the question. |
| Workspace tile looks left/right-flipped | The active camera faces you (tower mast cam). It should auto-mirror (`facing: operator` in config.yaml); if a camera moved, update its `facing`. |
| An arm "randomly" freezes | Its controller left the Quest's tracking view (HUD: `no tracking`), or its grip slipped below the press threshold. Re-squeeze; the 1 s countdown runs again from wherever the arm now is. |
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
| FastAPI backend (arms, guard, clutch + IK) | Jetson | `backend-jetson.sh` / systemd unit |
| Next.js frontend :3001 | desktop | `up.sh` |
| Caddy single origin :8444 LAN / :8445 tailnet | desktop | `up.sh` (config: `scripts/quest-teleop/Caddyfile`) |

Deeper background: `hmi/HANDOVER-2026-08-01-jetson-rig.md` (rig state),
`hmi/HANDOVER-2026-08-20-vr-teleop-port.md` (why the IK splits 3+2, and what
of the reference stack did *not* port), and the docstrings in
`haller_hmi/collision.py` and `haller_hmi/vr_teleop/` (how and why).
