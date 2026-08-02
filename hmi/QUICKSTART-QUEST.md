# Quest bimanual teleop — quickstart

Drive both SO-101 arms from a Meta Quest in passthrough AR: you see the real
arms through the headset, each grip is that arm's dead-man, and a capsule
model of both arms runs inside the 60 Hz loop so you cannot command them into
each other or through the bench.

The whole pipeline is exercised end-to-end in sim by `scripts/vr_smoke.py`
(15 checks, run against a live backend). What has **not** happened yet is a
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

## Start everything (each session, real arms)

```bash
# desktop, repo root
scripts/quest-teleop/up.sh
```

That checks/starts the backend on the Jetson (over `ssh jetson`), starts the
frontend on :3001 with the right baked-in URL, starts Caddy as the single
HTTPS origin, verifies the chain, and prints the URL:

```
https://192.168.0.191:8444/teleop/vr
```

Open it **in the Quest browser**, accept the self-signed cert once, press
**Enter Passthrough**. `scripts/quest-teleop/down.sh` stops the desktop half.

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
| release a grip | that arm freezes exactly where it is |
| take off headset / open Quest menu | frames force-disengage; arms freeze |

**Position mode (default).** Squeezing a grip *anchors*: your hand's current
position is bound to the arm's current pose, so there is nothing to match —
hold still through the 3 s countdown, feel the buzz, and from then on your
hand's movement drives the gripper tip through IK on the robot's own
kinematics. Release to freeze; move your hand somewhere comfortable and
squeeze again to ratchet across the workspace. Controller pitch/roll steer
the wrist relative to where it was anchored. No limb-length calibration is
involved. (The *hand mapping* selector still offers the legacy body-angle
mode; expect it to fight you — the SO-101's shoulder barely pitches below
horizontal and its elbow folds the opposite way to yours, which is why
position mode exists.) All motion rides the same rate limiter (60 °/s
ceiling at the handle, less during the first 1.5 s ramp).

The HUD floats over passthrough: per-side authority + countdown, grip state,
live collision clearance, an E-STOP button you can click with the controller
ray, and any backend error.

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
| Arm moves opposite the hand | Wrong *arm mounting* parity — see step 4 above. |
| An arm "randomly" freezes | Its controller left the Quest's tracking view (HUD: `no tracking`), or its grip slipped below the press threshold. Re-squeeze, re-match. |
| `COLLISION HOLD` where nothing is close | Mounts in `config.yaml` don't match the real plate — measure (step 2/3). |
| E-stopped, want to continue | **Re-arm arms** button on the VR page (sets MANUAL + torque), then Enter Passthrough again. |
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
