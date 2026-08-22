/**
 * What sessions this rig can actually start, given its arms and the stance.
 *
 * Pure, and separate from the tab that draws it: which hand drives which arm
 * is the one decision in the launcher that is wrong in a way you only find out
 * by moving a real arm, so it is worth pinning in tests rather than reading off
 * a rendered button.
 */
import { pairingFor, type Pairing, type Stance } from "@/lib/stance";

/** The port a sim arm reports. `/config` carries no `source` field, and this
 *  is the value the backend puts there for `source: sim` — a per-arm test,
 *  because a rig can mix one real arm with one sim arm. */
export const SIM_PORT = "(sim)";

export type ConfigArm = { id: string; port: string };

export type SessionPreset = {
  id: string;
  label: string;
  /** Which hand ends up on which arm — the thing the stance actually changes. */
  detail: string;
  pairing: Pairing;
  /** Why this rig cannot offer it, or null. Offered-but-disabled rather than
   *  hidden: "there is no second arm" is a fact about the robot, and a picker
   *  that quietly drops the option makes the operator doubt their memory. */
  unavailable: string | null;
};

export function isSimArm(a: ConfigArm): boolean {
  return a.port === SIM_PORT;
}

export function describePairing(p: Pairing): string {
  const parts: string[] = [];
  if (p.left_arm) parts.push(`L hand → ${p.left_arm}`);
  if (p.right_arm) parts.push(`R hand → ${p.right_arm}`);
  return parts.join(" · ") || "no arm";
}

/** Dual first, then one solo preset per configured arm. A solo session leaves
 *  the other hand's controller ignored and never writes to the absent side. */
export function presetsFor(
  arms: readonly string[],
  stance: Stance,
): SessionPreset[] {
  const dual = pairingFor(stance, arms);
  const presets: SessionPreset[] = [
    {
      id: "dual",
      label: "dual",
      detail: describePairing(dual),
      pairing: dual,
      unavailable:
        arms.length >= 2 ? null : "needs 2 enabled arms in config.yaml",
    },
  ];
  for (const arm of arms) {
    const pairing = pairingFor(stance, arms, arm);
    presets.push({
      id: `solo-${arm}`,
      label: `solo ${arm}`,
      detail: describePairing(pairing),
      pairing,
      unavailable: null,
    });
  }
  return presets;
}

/**
 * The sim leader→follower pairing, or null when this rig cannot do it.
 *
 * A different input path entirely: the operator drags the leader's joints in
 * the native MuJoCo viewer (`MUJOCO_VIEWER=1`) and the follower tracks it, so
 * the leader must be a sim arm. First sim arm leads, first other arm follows —
 * the pairing config.leader-follower-sim.yaml documents.
 */
export function simLeaderFor(
  arms: readonly ConfigArm[],
): { leader: string; follower: string } | null {
  const leader = arms.find(isSimArm);
  if (!leader) return null;
  const follower = arms.find((a) => a.id !== leader.id);
  if (!follower) return null;
  return { leader: leader.id, follower: follower.id };
}
