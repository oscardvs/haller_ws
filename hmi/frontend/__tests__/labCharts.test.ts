// hmi/frontend/__tests__/labCharts.test.ts
//
// The arithmetic that decides whether a chart tells the truth.
//
// Every function here is pure, and every one of them fails silently when it
// is wrong: a decimation that drops the spike, a log axis that swallows a
// zero, a path that draws a straight line across a gap in the data. None of
// those throw. They just draw something plausible that did not happen.
import { describe, it, expect } from "vitest";

import {
  PAD, ema, extent, fmtBytes, fmtDuration, fmtNum, fmtTick, hRule, linePath,
  logTicks, lttb, nearestIndex, niceTicks, padDomain, scale, seriesColor, vRule,
} from "@/components/lab/charts/svg";

const W = 400;
const H = 120;

describe("scale", () => {
  it("maps the domain onto the plot area, inside the padding", () => {
    const sc = scale(W, H, [0, 100], [0, 10]);
    expect(sc.x(0)).toBe(PAD.l);
    expect(sc.x(100)).toBe(W - PAD.r);
    expect(sc.y(10)).toBe(PAD.t);
    expect(sc.y(0)).toBe(H - PAD.b);
  });

  it("inverts x for the hover crosshair", () => {
    const sc = scale(W, H, [0, 100], [0, 1]);
    expect(sc.xInvert(sc.x(37))).toBeCloseTo(37, 6);
  });

  it("survives a zero-span domain rather than dividing by it", () => {
    // A run with one logged step, or a joint that never moved.
    const sc = scale(W, H, [5, 5], [2, 2]);
    expect(Number.isFinite(sc.x(5))).toBe(true);
    expect(Number.isFinite(sc.y(2))).toBe(true);
  });

  it("maps through log10 inside y, so callers pass raw data either way", () => {
    const sc = scale(W, H, [0, 1], [0.01, 10], { log: true });
    // Three decades of domain: 0.1 is one decade above the floor of three.
    expect(sc.y(10)).toBeCloseTo(PAD.t, 6);
    expect(sc.y(0.01)).toBeCloseTo(H - PAD.b, 6);
    expect(sc.y(0.1)).toBeGreaterThan(sc.y(1));
  });

  it("clamps a non-positive value on a log axis instead of dropping it", () => {
    // Dropping it shortens the series, which on a live run looks exactly like
    // the trainer stopping.
    const sc = scale(W, H, [0, 1], [1e-6, 10], { log: true });
    expect(Number.isFinite(sc.y(0))).toBe(true);
    expect(Number.isFinite(sc.y(-3))).toBe(true);
  });
});

describe("extent and padDomain", () => {
  it("ignores non-finite samples", () => {
    expect(extent([3, NaN, 1, Infinity, 9])).toEqual([1, 9]);
  });

  it("returns null when there is nothing finite to measure", () => {
    expect(extent([])).toBeNull();
    expect(extent([NaN, Infinity])).toBeNull();
  });

  it("gives a flat series a plot area to sit in", () => {
    // ±0 would put the line on the frame and the axis at one value.
    expect(padDomain([5, 5])).toEqual([4, 6]);
  });

  it("widens by a fraction of the span", () => {
    expect(padDomain([0, 100], 0.1)).toEqual([-10, 110]);
  });
});

describe("ticks", () => {
  it("picks round numbers inside the domain", () => {
    for (const v of niceTicks(0, 100, 4)) {
      expect(v % 25 === 0 || v % 20 === 0).toBe(true);
    }
    expect(niceTicks(0, 100, 4).length).toBeLessThanOrEqual(6);
  });

  it("never returns a tick outside the domain", () => {
    for (const v of niceTicks(-3.2, 7.8, 4)) {
      expect(v).toBeGreaterThanOrEqual(-3.2);
      expect(v).toBeLessThanOrEqual(7.8);
    }
  });

  it("degrades rather than looping on a degenerate domain", () => {
    expect(niceTicks(5, 5)).toEqual([5]);
    expect(niceTicks(NaN, 10)).toEqual([NaN]);
  });

  it("puts a decade at each power INSIDE a log domain", () => {
    // ACT loss spans 7.25 -> 0.068: two decades and change, and a linear axis
    // flattens everything after about step 50 into the baseline.
    expect(logTicks(0.068, 7.25)).toEqual([0.1, 1]);
  });

  it("never returns a log tick outside the domain", () => {
    // A tick outside the domain lands outside the plot box, and the chart
    // does not clip (the crosshair overhangs) — so it draws on top of
    // whichever chart is above or below it in the metric grid. Every one of
    // the seven charts in a real run did this before the bound was added.
    for (const [lo, hi] of [[1e-6, 1e-4], [0.9, 12], [140, 150]] as const) {
      for (const v of logTicks(lo, hi)) {
        expect(v).toBeGreaterThanOrEqual(lo);
        expect(v).toBeLessThanOrEqual(hi);
      }
    }
  });

  it("labels the endpoints when a domain spans no whole decade", () => {
    // gpu_mem_gb sits at 9.4 all run: no power of ten inside it, and an axis
    // with no labels at all is worse than one labelled with its range.
    expect(logTicks(9.2, 9.6)).toEqual([9.2, 9.6]);
  });
});

describe("linePath", () => {
  it("draws M then L", () => {
    const sc = scale(W, H, [0, 2], [0, 2]);
    expect(linePath([0, 1, 2], [0, 1, 2], sc)).toMatch(/^M[\d. ]+L[\d. ]+L[\d. ]+$/);
  });

  it("BREAKS the path at a gap instead of drawing across it", () => {
    // A straight line across a hole in the data is a claim that the value
    // moved smoothly through it, which is precisely what a gap means it did
    // not do.
    const sc = scale(W, H, [0, 3], [0, 3]);
    const d = linePath([0, 1, 2, 3], [0, NaN, 2, 3], sc);
    expect((d.match(/M/g) ?? []).length).toBe(2);
  });

  it("is empty for an empty series", () => {
    expect(linePath([], [], scale(W, H, [0, 1], [0, 1]))).toBe("");
  });

  it("stops at the shorter of xs and ys", () => {
    const sc = scale(W, H, [0, 3], [0, 3]);
    expect((linePath([0, 1, 2, 3], [0, 1], sc).match(/[ML]/g) ?? []).length).toBe(2);
  });
});

describe("rules", () => {
  it("spans the plot area horizontally and vertically", () => {
    const sc = scale(W, H, [0, 10], [0, 10]);
    expect(hRule(5, sc)).toBe(`M${PAD.l} ${sc.y(5).toFixed(1)}L${W - PAD.r} ${sc.y(5).toFixed(1)}`);
    expect(vRule(5, sc)).toBe(`M${sc.x(5).toFixed(1)} ${PAD.t}L${sc.x(5).toFixed(1)} ${H - PAD.b}`);
  });
});

describe("ema", () => {
  it("returns the series untouched at alpha 0", () => {
    expect(ema([1, 2, 3], 0)).toEqual([1, 2, 3]);
  });

  it("starts at the first sample, so the curve does not ramp out of zero", () => {
    expect(ema([10, 10, 10], 0.9)[0]).toBe(10);
  });

  it("holds a constant series constant", () => {
    for (const v of ema([4, 4, 4, 4], 0.8)) expect(v).toBeCloseTo(4, 9);
  });

  it("lags a step, which is what smoothing IS", () => {
    const out = ema([0, 10, 10, 10], 0.8);
    expect(out[1]).toBeGreaterThan(0);
    expect(out[1]).toBeLessThan(10);
    expect(out[3]).toBeGreaterThan(out[1]);
  });

  it("is forward-only — a later sample never moves an earlier one", () => {
    const short = ema([1, 5, 2], 0.7);
    const long = ema([1, 5, 2, 99, 99], 0.7);
    expect(long.slice(0, 3)).toEqual(short);
  });

  it("passes a gap through rather than poisoning the accumulator", () => {
    const out = ema([1, NaN, 1], 0.5);
    expect(Number.isNaN(out[1])).toBe(true);
    expect(Number.isFinite(out[2])).toBe(true);
  });
});

describe("lttb", () => {
  const N = 5000;
  const xs = Array.from({ length: N }, (_, i) => i);

  it("returns every index when nothing needs dropping", () => {
    expect(lttb([0, 1, 2], [0, 1, 2], 10)).toEqual([0, 1, 2]);
  });

  it("hits the target count and keeps both endpoints", () => {
    const ys = xs.map((i) => Math.sin(i / 50));
    const keep = lttb(xs, ys, 500);
    expect(keep.length).toBe(500);
    expect(keep[0]).toBe(0);
    expect(keep[keep.length - 1]).toBe(N - 1);
  });

  it("returns strictly ascending indices", () => {
    const keep = lttb(xs, xs.map((i) => Math.cos(i / 30)), 300);
    for (let i = 1; i < keep.length; i++) expect(keep[i]).toBeGreaterThan(keep[i - 1]);
  });

  it("KEEPS THE SPIKE that plain striding would drop", () => {
    // This is the whole reason the algorithm is here. A loss curve's spike is
    // the thing you opened the chart to find; a decimation that drops it
    // draws a run that looks healthier than it was.
    const ys = xs.map(() => 1);
    ys[2517] = 40;
    const keep = lttb(xs, ys, 200);
    expect(keep).toContain(2517);
    // Every 25th sample: the stride that a naive decimation would use.
    expect(2517 % Math.floor(N / 200)).not.toBe(0);
  });

  it("keeps a spike near the end too", () => {
    const ys = xs.map(() => 0.5);
    ys[4903] = -12;
    expect(lttb(xs, ys, 150)).toContain(4903);
  });
});

describe("nearestIndex", () => {
  const xs = [0, 10, 20, 30, 40];

  it("finds the closest sample on either side", () => {
    expect(nearestIndex(xs, 21)).toBe(2);
    expect(nearestIndex(xs, 26)).toBe(3);
    expect(nearestIndex(xs, 25)).toBe(2);
  });

  it("clamps past both ends", () => {
    expect(nearestIndex(xs, -99)).toBe(0);
    expect(nearestIndex(xs, 999)).toBe(4);
  });

  it("handles the degenerate sizes", () => {
    expect(nearestIndex([], 3)).toBe(-1);
    expect(nearestIndex([7], 3)).toBe(0);
  });

  it("agrees with a linear scan over a long series", () => {
    const long = Array.from({ length: 4000 }, (_, i) => i * 0.25);
    for (const probe of [0, 3.3, 511.9, 999.75]) {
      let best = 0;
      for (let i = 1; i < long.length; i++) {
        if (Math.abs(long[i] - probe) < Math.abs(long[best] - probe)) best = i;
      }
      expect(nearestIndex(long, probe)).toBe(best);
    }
  });
});

describe("formatters", () => {
  it("keeps a readout inside its column at both extremes", () => {
    expect(fmtNum(0.0680123)).toBe("0.068");
    expect(fmtNum(7.25)).toBe("7.250");
    expect(fmtNum(0.0000034)).toBe("3.4e-6");
    expect(fmtNum(12400000)).toBe("1.2e+7");
    expect(fmtNum(0)).toBe("0.000");
    expect(fmtNum(null)).toBe("—");
    expect(fmtNum(NaN)).toBe("—");
  });

  it("leaves an integer tick bare", () => {
    expect(fmtTick(20000)).toBe("20000");
    expect(fmtTick(0)).toBe("0");
    expect(fmtTick(0.1)).toBe("0.1");
  });

  it("reads durations as durations, not clock times", () => {
    expect(fmtDuration(47)).toBe("47s");
    expect(fmtDuration(64)).toBe("1m 04s");
    expect(fmtDuration(983.334)).toBe("16m 23s");
    expect(fmtDuration(7200)).toBe("2h 00m");
    expect(fmtDuration(null)).toBe("—");
  });

  it("prints bytes the same way the Dataset tab does", () => {
    // Two dataset surfaces printing one directory at two different sizes is
    // the kind of disagreement that makes an operator distrust both.
    expect(fmtBytes(742644287)).toBe("708 MB");   // >=100 drops the decimal
    expect(fmtBytes(14689240)).toBe("14.0 MB");
    expect(fmtBytes(900)).toBe("900 B");
    expect(fmtBytes(undefined)).toBe("—");
  });
});

describe("seriesColor", () => {
  it("is a token, never a hex", () => {
    expect(seriesColor(0)).toBe("var(--chart-1)");
    expect(seriesColor(4)).toBe("var(--chart-5)");
  });

  it("cycles rather than running off the end", () => {
    // A sixth run repeats a hue, which is honest. Inventing a colour is not.
    expect(seriesColor(5)).toBe(seriesColor(0));
    expect(seriesColor(-1)).toBe(seriesColor(4));
  });
});
