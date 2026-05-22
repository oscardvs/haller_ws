import { describe, it, expect, vi, beforeEach } from "vitest";

import { HumanTeleopClient } from "../lib/humanTeleopClient";

/* Minimal fake WebSocket. Each instance pushes itself onto `createdSockets`
   so tests can drive open / close / message events. */
const createdSockets: FakeWS[] = [];

class FakeWS {
  static OPEN = 1;
  url: string;
  readyState = 0;
  onopen?: () => void;
  onclose?: () => void;
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
  (globalThis as any).WebSocket = FakeWS;
});

describe("HumanTeleopClient", () => {
  it("does not send when no frame has been queued", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    c.tick();
    expect(createdSockets[0].sent.length).toBe(0);
  });

  it("sends the latest queued frame on tick()", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    c.queueFrame({
      type: "keypoints", ts_ms: 1, dead_man: false,
      left: null, right: null,
    });
    c.tick();
    expect(createdSockets[0].sent.length).toBe(1);
    const sent = JSON.parse(createdSockets[0].sent[0]);
    expect(sent.dead_man).toBe(false);
    expect(sent.ts_ms).toBe(1);
  });

  it("reconnects after close", async () => {
    const c = new HumanTeleopClient("ws://x");
    c.connect();
    await new Promise(r => setTimeout(r, 5));
    createdSockets[0].close();
    // The client should schedule a reconnect.
    await new Promise(r => setTimeout(r, 60));
    expect(createdSockets.length).toBeGreaterThan(1);
  });
});
