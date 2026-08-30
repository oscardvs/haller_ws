/**
 * The driver's proof that a running teleop session is its own.
 *
 * WHY IT EXISTS. The backend stops a session 5 s after the pose stream goes
 * quiet, and only a pose FRAME clears that window — deliberately, because
 * clearing it on reconnect once let a stray tab hold a dead session open
 * forever. But a page reload takes the socket down for a second and then
 * cannot send a pose until the operator clicks Enter VR, which takes longer
 * than five seconds. The session died with the operator standing right there.
 *
 * A token separates "the driver came back" from "some tab connected". The
 * backend hands it out only to a connection whose pose frame it actually took,
 * so a page parked on the landing screen never has one.
 *
 * SESSION storage, not local: it survives a reload of THIS tab, which is the
 * whole point, and dies when the tab closes — a closed tab is an operator who
 * has genuinely left, and that session should stop.
 */
const KEY = "haller.teleop.driverToken";

/** Reading storage can throw outright (Safari private mode, a browser set to
 *  block site data), and this runs on the socket's message handler where a
 *  throw is silent. Every accessor is guarded for that reason, and a failure
 *  degrades to exactly the old behaviour: no token, no re-entry window. */
export function rememberDriverToken(token: string): void {
  try {
    window.sessionStorage.setItem(KEY, token);
  } catch { /* no storage: the reload simply costs the session, as before */ }
}

export function driverToken(): string | null {
  try {
    return window.sessionStorage.getItem(KEY) || null;
  } catch { return null; }
}

export function forgetDriverToken(): void {
  try {
    window.sessionStorage.removeItem(KEY);
  } catch { /* nothing to forget */ }
}
