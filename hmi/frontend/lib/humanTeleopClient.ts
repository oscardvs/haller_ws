/**
 * WebSocket sender for the in-browser pose pipeline.
 *
 *   client.connect();
 *   // Each MediaPipe frame:
 *   client.queueFrame(frame);
 *   // 60 Hz from the render loop:
 *   client.tick();   // sends the latest queued frame if the socket is open
 *
 * The client reconnects after close with a 50 ms backoff (kept short so the
 * operator sees the live feed snap back fast; the backend grace window is 5 s
 * so we have plenty of headroom).
 */
import type { KeypointFrame } from "./mediapipe";

/**
 * `TFrame` defaults to `KeypointFrame` so every existing call site is unchanged.
 * The VR panel instantiates it with `VRFrame` instead: the socket only ever
 * stringifies whatever it is handed, and duplicating the reconnect/coalesce
 * logic for a second frame shape would mean two places to fix the next time the
 * grace window changes.
 */
export class HumanTeleopClient<TFrame = KeypointFrame> {
  private url: string;
  private ws: WebSocket | null = null;
  private latest: TFrame | null = null;
  private shouldReconnect = true;

  constructor(url: string) {
    this.url = url;
  }

  connect() {
    this.shouldReconnect = true;
    this._open();
  }

  close() {
    this.shouldReconnect = false;
    this.ws?.close();
    this.ws = null;
  }

  queueFrame(frame: TFrame) {
    this.latest = frame;
  }

  tick() {
    if (!this.latest) return;
    if (!this.ws || this.ws.readyState !== 1 /* OPEN */) return;
    this.ws.send(JSON.stringify(this.latest));
    this.latest = null;
  }

  private _open() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onclose = () => {
      this.ws = null;
      if (this.shouldReconnect) {
        setTimeout(() => this._open(), 50);
      }
    };
  }
}
