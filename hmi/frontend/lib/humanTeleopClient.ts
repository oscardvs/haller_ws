/**
 * WebSocket sender for the teleop frame stream, plus the socket's small
 * control channel back.
 *
 *   const client = new HumanTeleopClient<VRFrame>(url, {
 *     onOpen: () => client.send({ type: "request_settings" }),
 *     onMessage: (msg) => { ... },
 *   });
 *   client.connect();
 *   // Each sampled XR frame:
 *   client.queueFrame(frame);
 *   // ~30 Hz from the publish loop:
 *   client.tick();   // sends the latest queued frame if the socket is open
 *
 * The client reconnects after close with a 50 ms backoff (kept short so the
 * operator sees the live feed snap back fast; the backend grace window is 5 s
 * so we have plenty of headroom). `onOpen` fires on every open, reconnects
 * included — the server's config lives per connection, so a reconnected
 * client has to ask for the settings again or its sliders describe a
 * teleoperator that no longer exists.
 */

export type HumanTeleopClientOpts = {
  /** Every open, including reconnects. */
  onOpen?: () => void;
  /** One parsed-as-text message from the server. Parsing is the caller's,
   *  so this module stays ignorant of the protocol it carries. */
  onMessage?: (data: string) => void;
};

/**
 * `TFrame` is explicit at the call site: the socket only ever stringifies
 * whatever it is handed, and there is now exactly one frame shape on it.
 */
export class HumanTeleopClient<TFrame> {
  private url: string;
  private ws: WebSocket | null = null;
  private latest: TFrame | null = null;
  private shouldReconnect = true;
  private opts: HumanTeleopClientOpts;

  constructor(url: string, opts: HumanTeleopClientOpts = {}) {
    this.url = url;
    this.opts = opts;
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

  /** Send one control message (`config_update`, `request_settings`). Returns
   *  false when the socket is not open — a tuning nudge that arrives during a
   *  reconnect is dropped, not queued: the operator's next push resends it,
   *  and replaying a stale one over a fresh connection would fight whatever
   *  they set in between. */
  send(message: unknown): boolean {
    if (!this.ws || this.ws.readyState !== 1) return false;
    this.ws.send(JSON.stringify(message));
    return true;
  }

  private _open() {
    const ws = new WebSocket(this.url);
    this.ws = ws;
    ws.onopen = () => this.opts.onOpen?.();
    ws.onmessage = (ev: MessageEvent) => {
      const data = ev?.data;
      if (typeof data === "string") this.opts.onMessage?.(data);
    };
    ws.onclose = () => {
      this.ws = null;
      if (this.shouldReconnect) {
        setTimeout(() => this._open(), 50);
      }
    };
  }
}
