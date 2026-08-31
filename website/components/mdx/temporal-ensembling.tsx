'use client';

/* Visualisations for `content/docs/data/temporal-ensembling.mdx`.
 *
 * All three draw with inline SVG rather than a chart library: the shapes here
 * are bespoke (100 staggered bars, a decay curve whose SIGN is the point, a
 * sawtooth against a flat line) and none of them is a chart type a library
 * would give us for free. Colours come from the fumadocs CSS vars, so the
 * diagrams follow the theme without the hex-mirroring `mermaid.tsx` needs —
 * mermaid goes through khroma, which cannot parse oklch(); the browser can.
 *
 * House rule from `app/global.css`: GREEN is identity (the data, the
 * structure), ORANGE is action or event (the executing timestep, the control
 * the reader is dragging). Never both on one element.
 */

import { useId, useMemo, useState } from 'react';

const CHUNK = 100; // Haller's ACT: chunk_size 100
const FPS = 30; // ...at 30 fps, so a chunk is 3.33 s

/* ── shared chrome ───────────────────────────────────────────────────────── */

function Figure({
  label,
  caption,
  children,
}: {
  label: string;
  caption: React.ReactNode;
  children: React.ReactNode;
}) {
  return (
    <figure className="mermaid-frame my-7">
      <div className="h-label mb-3">{label}</div>
      {children}
      <figcaption className="mt-3 text-[0.82rem] leading-[1.5] text-fd-muted-foreground">
        {caption}
      </figcaption>
    </figure>
  );
}

/* ── 1. the overlapping chunks ───────────────────────────────────────────── */

/**
 * The core picture. One hundred chunks, each started one step later than the
 * last, each 100 steps long — so a single timestep in the middle is covered by
 * all of them. Drawn as 100 real rows rather than a representative handful,
 * because "a hundred separate predictions of the same instant" is the whole
 * idea and a stack of twelve undersells it.
 */
export function ChunkOverlap() {
  const W = 720;
  const PAD_L = 66;
  const PAD_R = 18;
  const TOP = 44;
  const ROW = 2.62;
  const BAR = 1.85;
  const SPAN = 2 * CHUNK - 1; // t = 0 … 198
  const px = (W - PAD_L - PAD_R) / SPAN;
  const plotH = CHUNK * ROW;
  const BASE = TOP + plotH;
  const AXIS = BASE + 26; // pushed down: a label now sits under the last bar
  const H = AXIS + 40;

  const execT = CHUNK - 1; // timestep 99: the first one every chunk has seen
  const execX = PAD_L + execT * px;

  const rows = Array.from({ length: CHUNK }, (_, i) => i);

  return (
    <Figure
      label="Fig. 1 — one timestep, a hundred predictions"
      caption={
        <>
          Every row is one chunk of 100 actions, produced from one observation,
          and each starts a step later than the row above. The orange line marks
          a single timestep. <strong>Every one of the 100 rows crosses it</strong> —
          the top row predicted it 99 steps in advance, the bottom row one step
          in advance, and the 98 rows between at every distance in between.
          Temporal ensembling averages that column.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="One hundred staggered action chunks, all overlapping a single executing timestep"
      >
        {/* the chunks */}
        <g>
          {rows.map((i) => (
            <rect
              key={i}
              x={PAD_L + i * px}
              y={TOP + i * ROW}
              width={CHUNK * px}
              height={BAR}
              rx={0.9}
              fill="var(--color-fd-primary)"
              opacity={0.36}
            />
          ))}
        </g>

        {/* the executing column, drawn over everything */}
        <line
          x1={execX}
          y1={TOP - 12}
          x2={execX}
          y2={BASE + 10}
          stroke="var(--haller-action)"
          strokeWidth={1.6}
        />
        {/* the crossing points light up */}
        <g>
          {rows.map((i) => (
            <rect
              key={i}
              x={execX - 1.4}
              y={TOP + i * ROW}
              width={2.8}
              height={BAR}
              fill="var(--haller-action)"
            />
          ))}
        </g>

        {/* row annotations */}
        <text
          x={PAD_L - 8}
          y={TOP + 6}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          obs t=0
        </text>
        <text
          x={PAD_L - 8}
          y={BASE - 1}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          obs t=99
        </text>

        {/* brace-ish rail showing the 100 rows */}
        <line
          x1={PAD_L - 44}
          y1={TOP}
          x2={PAD_L - 44}
          y2={BASE}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
        />
        <text
          x={PAD_L - 49}
          y={TOP + plotH / 2 - 4}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
          transform={`rotate(-90 ${PAD_L - 49} ${TOP + plotH / 2 - 4})`}
        >
          100 chunks
        </text>

        {/* callouts on the extreme rows */}
        <text
          x={PAD_L + CHUNK * px + 8}
          y={TOP + 6}
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          predicted 99 steps ahead
        </text>
        {/* sits BELOW the last bar: the bottom chunk runs to the right edge,
            so there is no clear space beside it */}
        <text
          x={W - PAD_R}
          y={BASE + 11}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          predicted 1 step ahead
        </text>

        {/* label for the column */}
        <text
          x={execX}
          y={TOP - 18}
          textAnchor="middle"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--haller-action)"
        >
          timestep 99 executes here
        </text>

        {/* time axis */}
        <line
          x1={PAD_L}
          y1={AXIS}
          x2={W - PAD_R}
          y2={AXIS}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
        />
        {[0, 50, 99, 150, 198].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L + t * px}
              y1={AXIS}
              x2={PAD_L + t * px}
              y2={AXIS + 5}
              stroke="var(--color-fd-border)"
              strokeWidth={1}
            />
            <text
              x={PAD_L + t * px}
              y={AXIS + 16}
              textAnchor="middle"
              className="h-mono"
              fontSize={8.5}
              fill="var(--color-fd-muted-foreground)"
            >
              {t}
            </text>
          </g>
        ))}
        <text
          x={W - PAD_R}
          y={AXIS + 32}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          timestep →
        </text>
      </svg>
    </Figure>
  );
}

/* ── 2. the weights, and their sign ──────────────────────────────────────── */

type Shares = {
  weights: number[];
  total: number;
  newest1: number;
  newest10: number;
  newest30: number;
  oldest10: number;
  meanAgeSteps: number;
};

/** wᵢ = exp(−coeff·i), i = 0 is the OLDEST prediction. Normalised to shares. */
function shares(coeff: number): Shares {
  const weights = Array.from({ length: CHUNK }, (_, i) => Math.exp(-coeff * i));
  const total = weights.reduce((a, b) => a + b, 0);
  const sum = (from: number, to: number) =>
    weights.slice(from, to).reduce((a, b) => a + b, 0) / total;
  /* prediction i was made (CHUNK−1−i) steps before this one executes */
  const meanAgeSteps =
    weights.reduce((acc, w, i) => acc + w * (CHUNK - 1 - i), 0) / total;
  return {
    weights,
    total,
    newest1: weights[CHUNK - 1] / total,
    newest10: sum(CHUNK - 10, CHUNK),
    newest30: sum(CHUNK - 30, CHUNK),
    oldest10: sum(0, 10),
    meanAgeSteps,
  };
}

const PRESETS = [
  { coeff: 0.01, label: '+0.01', note: 'ACT default · what Haller runs' },
  { coeff: 0, label: '0.00', note: 'uniform' },
  { coeff: -0.01, label: '−0.01', note: 'mirror of the default' },
  { coeff: -0.05, label: '−0.05', note: 'strongly favours fresh' },
];

const pct = (x: number) => `${(x * 100).toFixed(2)}%`;

export function EnsembleWeights() {
  const [coeff, setCoeff] = useState(0.01);
  const gid = useId().replace(/:/g, '_');
  const s = useMemo(() => shares(coeff), [coeff]);

  const W = 720;
  const H = 250;
  const PAD_L = 52;
  const PAD_R = 20;
  const TOP = 18;
  const BASE = H - 46;
  const px = (W - PAD_L - PAD_R) / (CHUNK - 1);

  /* Scale to the largest share at THIS coefficient so the curve always fills
     the box — the reader is comparing the shape and its direction, not
     absolute heights across settings. */
  const maxShare = Math.max(...s.weights) / s.total;
  const y = (share: number) => BASE - (share / maxShare) * (BASE - TOP);

  const pts = s.weights
    .map((w, i) => `${PAD_L + i * px},${y(w / s.total)}`)
    .join(' ');
  const area = `${PAD_L},${BASE} ${pts} ${PAD_L + (CHUNK - 1) * px},${BASE}`;

  const favoursOld = coeff > 0.0005;
  const favoursNew = coeff < -0.0005;

  return (
    <Figure
      label="Fig. 2 — the weighting, and the sign that surprises people"
      caption={
        <>
          Drag the coefficient. At the positive default the curve slopes{' '}
          <em>down to the right</em>: the oldest predictions carry the most
          weight and the freshest observation contributes{' '}
          <strong>{pct(shares(0.01).newest1)}</strong> of the action actually
          sent to the arm. Only a negative coefficient tilts the average toward
          fresh information.
        </>
      }
    >
      {/* control */}
      <div className="mb-4 flex flex-col gap-3">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label
            htmlFor={`coeff-${gid}`}
            className="h-label"
            style={{ letterSpacing: '0.14em' }}
          >
            temporal_ensemble_coeff
          </label>
          <span
            className="h-mono text-[0.95rem] font-semibold tabular-nums"
            style={{ color: 'var(--haller-action)' }}
          >
            {coeff > 0 ? '+' : coeff < 0 ? '−' : ' '}
            {Math.abs(coeff).toFixed(3)}
          </span>
        </div>

        <input
          id={`coeff-${gid}`}
          type="range"
          min={-0.05}
          max={0.05}
          step={0.001}
          value={coeff}
          onChange={(e) => setCoeff(Number(e.target.value))}
          className="w-full accent-[var(--haller-action)]"
          aria-describedby={`coeff-readout-${gid}`}
        />

        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map((p) => {
            const on = Math.abs(coeff - p.coeff) < 0.0005;
            return (
              <button
                key={p.label}
                type="button"
                onClick={() => setCoeff(p.coeff)}
                title={p.note}
                className={`h-mono rounded-md border px-2 py-1 text-[11px] transition-colors ${
                  on
                    ? 'border-[var(--haller-action)] text-[var(--haller-action)] bg-[var(--haller-action-soft)]'
                    : 'border-fd-border text-fd-muted-foreground hover:border-fd-muted-foreground'
                }`}
              >
                {p.label}
              </button>
            );
          })}
        </div>
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`Weight distribution across the 100 predictions at coefficient ${coeff}`}
      >
        <defs>
          <linearGradient id={`fill-${gid}`} x1="0" y1="0" x2="0" y2="1">
            <stop
              offset="0%"
              stopColor="var(--color-fd-primary)"
              stopOpacity={0.34}
            />
            <stop
              offset="100%"
              stopColor="var(--color-fd-primary)"
              stopOpacity={0.03}
            />
          </linearGradient>
        </defs>

        {/* baseline + midline */}
        <line
          x1={PAD_L}
          y1={BASE}
          x2={W - PAD_R}
          y2={BASE}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
        />

        <polygon points={area} fill={`url(#fill-${gid})`} />
        <polyline
          points={pts}
          fill="none"
          stroke="var(--color-fd-primary)"
          strokeWidth={2}
        />

        {/* the two ends, called out by name */}
        <text
          x={PAD_L}
          y={BASE + 17}
          textAnchor="start"
          className="h-mono"
          fontSize={9}
          fill="var(--color-fd-muted-foreground)"
        >
          i = 0
        </text>
        <text
          x={PAD_L}
          y={BASE + 29}
          textAnchor="start"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill={
            favoursOld ? 'var(--haller-action)' : 'var(--color-fd-muted-foreground)'
          }
        >
          OLDEST
        </text>
        <text
          x={PAD_L}
          y={BASE + 40}
          textAnchor="start"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          made 99 steps ago
        </text>

        <text
          x={W - PAD_R}
          y={BASE + 17}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fill="var(--color-fd-muted-foreground)"
        >
          i = 99
        </text>
        <text
          x={W - PAD_R}
          y={BASE + 29}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill={
            favoursNew ? 'var(--haller-action)' : 'var(--color-fd-muted-foreground)'
          }
        >
          NEWEST
        </text>
        <text
          x={W - PAD_R}
          y={BASE + 40}
          textAnchor="end"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          made this step
        </text>

        {/* y label */}
        <text
          x={14}
          y={(TOP + BASE) / 2}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
          transform={`rotate(-90 14 ${(TOP + BASE) / 2})`}
        >
          weight
        </text>

        {/* which way is it leaning */}
        <text
          x={(PAD_L + W - PAD_R) / 2}
          y={TOP + 10}
          textAnchor="middle"
          className="h-mono"
          fontSize={9.5}
          fontWeight={600}
          fill="var(--color-fd-muted-foreground)"
        >
          {favoursOld
            ? 'stale predictions weigh MORE'
            : favoursNew
              ? 'fresh predictions weigh more'
              : 'every prediction weighs the same'}
        </text>
      </svg>

      {/* readout */}
      <div
        id={`coeff-readout-${gid}`}
        className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 sm:grid-cols-4"
      >
        {[
          { k: 'newest 1', v: s.newest1, sub: 'this step' },
          { k: 'newest 10', v: s.newest10, sub: '0.33 s' },
          { k: 'newest 30', v: s.newest30, sub: '1.0 s' },
          { k: 'oldest 10', v: s.oldest10, sub: 'first seen' },
        ].map((c) => (
          <div key={c.k} className="border-t border-fd-border pt-2">
            <div className="h-label" style={{ letterSpacing: '0.12em' }}>
              {c.k}
            </div>
            <div className="h-mono mt-1 text-[1.05rem] tabular-nums text-fd-foreground">
              {pct(c.v)}
            </div>
            <div className="h-mono text-[10px] text-fd-muted-foreground">
              {c.sub}
            </div>
          </div>
        ))}
      </div>

      <p className="mt-3 text-[0.82rem] leading-[1.5] text-fd-muted-foreground">
        Mean age of the information in one executed action:{' '}
        <span className="h-mono tabular-nums text-fd-foreground">
          {s.meanAgeSteps.toFixed(1)} steps
        </span>{' '}
        ={' '}
        <span className="h-mono tabular-nums text-fd-foreground">
          {(s.meanAgeSteps / FPS).toFixed(2)} s
        </span>
        .
      </p>
    </Figure>
  );
}

/* ── 3. blind-and-seamed vs continuous ───────────────────────────────────── */

/* The baseline's information age ramps 0…99 steps across each chunk, so its
   time-average is simply the midpoint. Worth drawing, because it is the
   comparison that stops this page overselling ensembling: at coeff +0.01 the
   ensemble's mean age (57.7 steps) is ABOVE this (49.5). */
const BASELINE_MEAN_AGE = (CHUNK - 1) / 2;

/**
 * The baseline's cost drawn as what it actually is: an information age that
 * ramps from 0 to 3.33 s and snaps back, with a discontinuity in the commanded
 * plan at every snap. Ensembling replaces the sawtooth with a flat line — but
 * that line sits slightly ABOVE the sawtooth's average, not below it. The win
 * is the removed excursion and the removed seam, not fresher information.
 */
export function BlindVsEnsembled() {
  const W = 720;
  const PAD_L = 102; // wide enough for the n_action_steps sub-label
  const PAD_R = 18;
  const STEPS = 300; // 10 s at 30 fps
  const px = (W - PAD_L - PAD_R) / STEPS;

  const rowY = [46, 118]; // the two strategy lanes
  const AGE_TOP = 178;
  const AGE_BASE = 292;
  const H = 336;

  const meanAge = shares(0.01).meanAgeSteps;
  const ageY = (steps: number) =>
    AGE_BASE - (steps / CHUNK) * (AGE_BASE - AGE_TOP);

  /* baseline sawtooth: age climbs 0→99 across each chunk, then resets */
  const saw: string[] = [];
  for (let c = 0; c * CHUNK < STEPS; c++) {
    const x0 = PAD_L + c * CHUNK * px;
    const x1 = PAD_L + Math.min((c + 1) * CHUNK, STEPS) * px;
    saw.push(`${x0},${ageY(0)}`, `${x1},${ageY(Math.min(CHUNK, STEPS - c * CHUNK))}`);
  }

  return (
    <Figure
      label="Fig. 3 — what the baseline pays, and what ensembling pays instead"
      caption={
        <>
          The baseline observes once per chunk, so the information driving the
          arm ages from fresh to 3.33 s stale and snaps back — and the plan
          changes discontinuously at every snap, which is the seam you can see
          in the motion. Ensembling flattens the sawtooth to a constant{' '}
          <span className="h-mono">{(meanAge / FPS).toFixed(2)} s</span>.{' '}
          <strong>
            Note where that sits: above the baseline&rsquo;s average of{' '}
            {(BASELINE_MEAN_AGE / FPS).toFixed(2)} s, not below it.
          </strong>{' '}
          At the positive default, ensembling is on average marginally{' '}
          <em>staler</em> than the baseline. What it removes is the excursion to
          3.33 s and the discontinuity — not the staleness.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Baseline chunked execution compared with temporal ensembling"
      >
        {/* ── lane 1: baseline ── */}
        <text
          x={PAD_L - 10}
          y={rowY[0] - 4}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-foreground)"
        >
          baseline
        </text>
        <text
          x={PAD_L - 10}
          y={rowY[0] + 8}
          textAnchor="end"
          className="h-mono"
          fontSize={7}
          fill="var(--color-fd-muted-foreground)"
        >
          n_action_steps=100
        </text>

        {[0, 1, 2].map((c) => {
          const x0 = PAD_L + c * CHUNK * px;
          const w = Math.min(CHUNK, STEPS - c * CHUNK) * px;
          return (
            <g key={c}>
              <rect
                x={x0}
                y={rowY[0] - 12}
                width={w}
                height={24}
                rx={2}
                fill="var(--color-fd-primary)"
                opacity={0.16}
                stroke="var(--color-fd-primary)"
                strokeOpacity={0.4}
                strokeWidth={1}
              />
              <text
                x={x0 + w / 2}
                y={rowY[0] + 4}
                textAnchor="middle"
                className="h-mono"
                fontSize={8.5}
                fill="var(--color-fd-muted-foreground)"
              >
                blind 3.33 s
              </text>
              {/* the observation that produced this chunk */}
              <circle
                cx={x0}
                cy={rowY[0] - 12}
                r={3.2}
                fill="var(--color-fd-primary)"
              />
              {/* the seam */}
              {c > 0 && (
                <line
                  x1={x0}
                  y1={rowY[0] - 20}
                  x2={x0}
                  y2={rowY[0] + 20}
                  stroke="var(--haller-action)"
                  strokeWidth={1.6}
                  strokeDasharray="3 2"
                />
              )}
            </g>
          );
        })}
        <text
          x={PAD_L + CHUNK * px}
          y={rowY[0] - 26}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--haller-action)"
        >
          seam
        </text>

        {/* ── lane 2: ensembled ── */}
        <text
          x={PAD_L - 10}
          y={rowY[1] - 4}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-foreground)"
        >
          ensembled
        </text>
        <text
          x={PAD_L - 10}
          y={rowY[1] + 8}
          textAnchor="end"
          className="h-mono"
          fontSize={7}
          fill="var(--color-fd-muted-foreground)"
        >
          n_action_steps=1
        </text>

        <rect
          x={PAD_L}
          y={rowY[1] - 12}
          width={STEPS * px}
          height={24}
          rx={2}
          fill="var(--color-fd-primary)"
          opacity={0.16}
          stroke="var(--color-fd-primary)"
          strokeOpacity={0.4}
          strokeWidth={1}
        />
        {/* an observation every single step */}
        <g>
          {Array.from({ length: STEPS / 2 }, (_, k) => (
            <line
              key={k}
              x1={PAD_L + k * 2 * px}
              y1={rowY[1] - 12}
              x2={PAD_L + k * 2 * px}
              y2={rowY[1] - 5}
              stroke="var(--color-fd-primary)"
              strokeWidth={0.8}
              opacity={0.75}
            />
          ))}
        </g>
        <text
          x={PAD_L + (STEPS * px) / 2}
          y={rowY[1] + 5}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          observes every step · no seam
        </text>

        {/* ── age plot ── */}
        <text
          x={PAD_L - 10}
          y={AGE_TOP + 4}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          3.33 s
        </text>
        <text
          x={PAD_L - 10}
          y={AGE_BASE + 3}
          textAnchor="end"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          0 s
        </text>
        <text
          x={20}
          y={(AGE_TOP + AGE_BASE) / 2}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
          transform={`rotate(-90 20 ${(AGE_TOP + AGE_BASE) / 2})`}
        >
          info age
        </text>

        <line
          x1={PAD_L}
          y1={AGE_BASE}
          x2={W - PAD_R}
          y2={AGE_BASE}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
        />
        <line
          x1={PAD_L}
          y1={AGE_TOP}
          x2={W - PAD_R}
          y2={AGE_TOP}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
          strokeDasharray="2 3"
        />

        {/* baseline sawtooth */}
        <polyline
          points={saw.join(' ')}
          fill="none"
          stroke="var(--color-fd-muted-foreground)"
          strokeWidth={1.6}
        />
        <text
          x={PAD_L + 20 * px}
          y={ageY(16)}
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          baseline sawtooth
        </text>

        {/* the baseline's own time-average, so the flat line can be judged
            against something rather than admired on its own */}
        <line
          x1={PAD_L}
          y1={ageY(BASELINE_MEAN_AGE)}
          x2={W - PAD_R}
          y2={ageY(BASELINE_MEAN_AGE)}
          stroke="var(--color-fd-muted-foreground)"
          strokeWidth={1}
          strokeDasharray="4 3"
          opacity={0.8}
        />
        <g>
          <rect
            x={PAD_L + 2}
            y={ageY(BASELINE_MEAN_AGE) + 3}
            width={112}
            height={11}
            rx={2}
            fill="var(--color-fd-card)"
          />
          <text
            x={PAD_L + 6}
            y={ageY(BASELINE_MEAN_AGE) + 11.5}
            className="h-mono"
            fontSize={8}
            fill="var(--color-fd-muted-foreground)"
          >
            baseline mean {(BASELINE_MEAN_AGE / FPS).toFixed(2)} s
          </text>
        </g>

        {/* ensembled flat line */}
        <line
          x1={PAD_L}
          y1={ageY(meanAge)}
          x2={W - PAD_R}
          y2={ageY(meanAge)}
          stroke="var(--color-fd-primary)"
          strokeWidth={2}
        />
        <g>
          <rect
            x={W - PAD_R - 172}
            y={ageY(meanAge) - 15}
            width={170}
            height={12}
            rx={2}
            fill="var(--color-fd-card)"
          />
          <text
            x={W - PAD_R - 4}
            y={ageY(meanAge) - 5.5}
            textAnchor="end"
            className="h-mono"
            fontSize={8.5}
            fontWeight={600}
            fill="var(--color-fd-primary)"
          >
            ensembled · {(meanAge / FPS).toFixed(2)} s, constant
          </text>
        </g>

        {/* time axis */}
        {[0, 100, 200, 300].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L + t * px}
              y1={AGE_BASE}
              x2={PAD_L + t * px}
              y2={AGE_BASE + 5}
              stroke="var(--color-fd-border)"
              strokeWidth={1}
            />
            <text
              x={PAD_L + t * px}
              y={AGE_BASE + 17}
              textAnchor="middle"
              className="h-mono"
              fontSize={8.5}
              fill="var(--color-fd-muted-foreground)"
            >
              {(t / FPS).toFixed(1)}s
            </text>
          </g>
        ))}
      </svg>
    </Figure>
  );
}
