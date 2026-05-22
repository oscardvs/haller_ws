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

export class HumanTeleopClient {
  private url: string;
  private ws: WebSocket | null = null;
  private latest: KeypointFrame | null = null;
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

  queueFrame(frame: KeypointFrame) {
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
