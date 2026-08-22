import { describe, it, expect, beforeEach } from "vitest";

import { HumanTeleopClient } from "../lib/humanTeleopClient";
import { disengagedFrame, type VRFrame } from "../lib/vrTeleop";

/* Minimal fake WebSocket. Each instance pushes itself onto `createdSockets`
   so tests can drive open / close / message events. */
const createdSockets: FakeWS[] = [];

class FakeWS {
  static OPEN = 1;
  url: string;
  readyState = 0;
  onopen?: () => void;
  onclose?: () => void;
  onmessage?: (ev: { data: unknown }) => void;
  sent: string[] = [];

  constructor(url: string) {
    this.url = url;
    createdSockets.push(this);
    queueMicrotask(() => {
      this.readyState = FakeWS.OPEN;
      this.onopen?.();
    });
  }

  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    this.onclose?.();
  }
}

beforeEach(() => {
  createdSockets.length = 0;
  (globalThis as unknown as { WebSocket: unknown }).WebSocket = FakeWS;
});

const settle = () => new Promise((r) => setTimeout(r, 5));

describe("HumanTeleopClient", () => {
  it("does not send when no frame has been queued", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    c.connect();
    await settle();
    c.tick();
    expect(createdSockets[0].sent.length).toBe(0);
  });

  it("sends the latest queued frame on tick()", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    c.connect();
    await settle();
    c.queueFrame(disengagedFrame(1));
    c.tick();
    expect(createdSockets[0].sent.length).toBe(1);
    const sent = JSON.parse(createdSockets[0].sent[0]);
    expect(sent.type).toBe("vr_keypoints");
    expect(sent.dead_man).toBe(false);
    expect(sent.ts_ms).toBe(1);
  });

  it("coalesces: only the newest frame goes out, and only once", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    c.connect();
    await settle();
    c.queueFrame(disengagedFrame(1));
    c.queueFrame(disengagedFrame(2));
    c.tick();
    c.tick();
    expect(createdSockets[0].sent.map((s) => JSON.parse(s).ts_ms)).toEqual([2]);
  });

  it("reconnects after close", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    c.connect();
    await settle();
    createdSockets[0].close();
    // The client should schedule a reconnect.
    await new Promise((r) => setTimeout(r, 60));
    expect(createdSockets.length).toBeGreaterThan(1);
  });

  it("fires onOpen on EVERY open, reconnects included", async () => {
    // The teleoperator's config lives per connection: a client that asked for
    // settings only once would show numbers belonging to a socket that is
    // gone, and every tuning read-out would quietly be a lie.
    let opens = 0;
    const c = new HumanTeleopClient<VRFrame>("ws://x", { onOpen: () => { opens += 1; } });
    c.connect();
    await settle();
    expect(opens).toBe(1);
    createdSockets[0].close();
    await new Promise((r) => setTimeout(r, 60));
    expect(opens).toBe(2);
  });

  it("hands text messages to onMessage and ignores anything else", async () => {
    const seen: string[] = [];
    const c = new HumanTeleopClient<VRFrame>("ws://x", { onMessage: (d) => { seen.push(d); } });
    c.connect();
    await settle();
    createdSockets[0].onmessage?.({ data: '{"type":"ik_state"}' });
    createdSockets[0].onmessage?.({ data: new ArrayBuffer(4) });
    expect(seen).toEqual(['{"type":"ik_state"}']);
  });

  it("send() writes a control message, and reports a closed socket", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    // Before connect there is no socket at all: a tuning nudge is dropped,
    // not queued — replaying a stale one over a fresh connection would fight
    // whatever the operator set in between.
    expect(c.send({ type: "request_settings" })).toBe(false);
    c.connect();
    await settle();
    expect(c.send({ type: "config_update", config: { lam_pos: 0.02 } })).toBe(true);
    expect(JSON.parse(createdSockets[0].sent[0])).toEqual(
      { type: "config_update", config: { lam_pos: 0.02 } });
  });

  it("stops reconnecting after close()", async () => {
    const c = new HumanTeleopClient<VRFrame>("ws://x");
    c.connect();
    await settle();
    c.close();
    await new Promise((r) => setTimeout(r, 60));
    expect(createdSockets.length).toBe(1);
  });
});
