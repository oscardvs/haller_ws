// hmi/frontend/lib/config.ts
export const BACKEND_URL =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "http://localhost:8000";

export const WS_URL =
  BACKEND_URL.replace(/^http/, "ws") + "/ws/telemetry";

/** True for a loopback URL — the one case where the page's own origin is a
 *  better answer than the baked one (a dev bundle). */
const LOOPBACK = /^https?:\/\/(localhost|127\.0\.0\.1)(:|$)/;

export function isLoopback(url: string): boolean {
  return LOOPBACK.test(url);
}

/** The one origin the headset opens.
 *
 *  up.sh serves cockpit and API behind a single HTTPS origin and bakes it
 *  into the bundle as NEXT_PUBLIC_BACKEND_URL with an `/api` suffix — strip
 *  that and the remainder IS the headset's base. `window.location.origin`
 *  would be wrong on a Quest: "localhost" there is the headset itself, so
 *  the page's origin is only the fallback for a loopback (dev) bundle.
 *
 *  Returns null when there is nothing trustworthy to print (loopback bundle
 *  and no page origin — the server render). */
export function headsetOrigin(
  pageOrigin: string | null,
  bakedUrl: string = BACKEND_URL,
): string | null {
  const baked = bakedUrl.replace(/\/api\/?$/, "");
  return LOOPBACK.test(baked) ? pageOrigin : baked;
}
