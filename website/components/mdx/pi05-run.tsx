'use client';

/* Visualisations for `content/docs/data/pi05-run.mdx`.
 *
 * Inline SVG rather than a chart library, same as `temporal-ensembling.tsx`:
 * these are bespoke shapes (a dataflow with a cached branch, a discretisation
 * ladder, overlapping confidence intervals, a pipeline with a dashed future)
 * and none is a chart type a library hands you. Colours come from the fumadocs
 * CSS vars so the diagrams follow the theme without hex-mirroring.
 *
 * House rule from `app/global.css`: GREEN is identity (the data, the model,
 * the structure that already exists), ORANGE is action or event (the thing
 * being generated, the control being dragged, the step not yet taken).
 * Never both on one element.
 */

import { useId, useMemo, useState } from 'react';

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

/** A labelled rounded box. `tone` picks which half of the house palette it sits in. */
function Box({
  x,
  y,
  w,
  h,
  title,
  sub,
  tone = 'data',
  dashed = false,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  tone?: 'data' | 'action' | 'muted';
  dashed?: boolean;
}) {
  const stroke =
    tone === 'action'
      ? 'var(--haller-action)'
      : tone === 'muted'
        ? 'var(--color-fd-border)'
        : 'var(--color-fd-primary)';
  const fill =
    tone === 'action'
      ? 'var(--haller-action)'
      : tone === 'muted'
        ? 'var(--color-fd-muted-foreground)'
        : 'var(--color-fd-primary)';
  return (
    <g>
      <rect
        x={x}
        y={y}
        width={w}
        height={h}
        rx={3}
        fill={fill}
        fillOpacity={tone === 'muted' ? 0.05 : 0.14}
        stroke={stroke}
        strokeOpacity={tone === 'muted' ? 0.5 : 0.7}
        strokeWidth={1}
        strokeDasharray={dashed ? '4 3' : undefined}
      />
      <text
        x={x + w / 2}
        y={sub ? y + h / 2 - 2 : y + h / 2 + 3.2}
        textAnchor="middle"
        className="h-mono"
        fontSize={9}
        fontWeight={600}
        fill="var(--color-fd-foreground)"
      >
        {title}
      </text>
      {sub && (
        <text
          x={x + w / 2}
          y={y + h / 2 + 10}
          textAnchor="middle"
          className="h-mono"
          fontSize={7.5}
          fill="var(--color-fd-muted-foreground)"
        >
          {sub}
        </text>
      )}
    </g>
  );
}

function Arrow({
  x1,
  y1,
  x2,
  y2,
  tone = 'data',
  marker,
}: {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  tone?: 'data' | 'action' | 'muted';
  marker: string;
}) {
  const stroke =
    tone === 'action'
      ? 'var(--haller-action)'
      : tone === 'muted'
        ? 'var(--color-fd-border)'
        : 'var(--color-fd-primary)';
  return (
    <line
      x1={x1}
      y1={y1}
      x2={x2}
      y2={y2}
      stroke={stroke}
      strokeWidth={1.3}
      strokeOpacity={0.75}
      markerEnd={`url(#${marker})`}
    />
  );
}

function Defs({ gid }: { gid: string }) {
  return (
    <defs>
      <marker
        id={`ar-data-${gid}`}
        viewBox="0 0 8 8"
        refX={7}
        refY={4}
        markerWidth={5}
        markerHeight={5}
        orient="auto"
      >
        <path d="M0,1 L7,4 L0,7 z" fill="var(--color-fd-primary)" opacity={0.8} />
      </marker>
      <marker
        id={`ar-action-${gid}`}
        viewBox="0 0 8 8"
        refX={7}
        refY={4}
        markerWidth={5}
        markerHeight={5}
        orient="auto"
      >
        <path d="M0,1 L7,4 L0,7 z" fill="var(--haller-action)" opacity={0.9} />
      </marker>
      <marker
        id={`ar-muted-${gid}`}
        viewBox="0 0 8 8"
        refX={7}
        refY={4}
        markerWidth={5}
        markerHeight={5}
        orient="auto"
      >
        <path d="M0,1 L7,4 L0,7 z" fill="var(--color-fd-border)" />
      </marker>
    </defs>
  );
}

/* ── 1. the forward path ─────────────────────────────────────────────────── */

/**
 * π0.5's forward pass, drawn to make three specific things obvious:
 *   1. every camera goes through ONE shared vision tower, with no per-camera weights;
 *   2. the prefix (images + language) is computed ONCE and cached as KV, so the
 *      ten integration steps only re-run the small expert;
 *   3. actions live at 32 dims internally and are cut back to the dataset's
 *      real width on the way out.
 */
export function Pi05Forward() {
  const gid = useId().replace(/:/g, '_');
  const W = 720;
  const H = 400;

  const camY = [38, 74, 110];
  const towerX = 196;
  const prefixX = 320;

  return (
    <Figure
      label="Fig. 2: one forward pass, and the part that runs ten times"
      caption={
        <>
          Every camera passes through the <strong>same</strong> vision tower:{' '}
          <code>embed_prefix</code> loops over the images calling one{' '}
          <code>embed_image</code>, so there is no per-camera parameter anywhere.
          Images and language concatenate into a single attention prefix, which
          the 2B backbone processes <strong>once</strong> and hands on as a KV
          cache. The ten integration steps that actually produce the actions
          re-run only the 300M expert against that cache, which is why ten
          steps costs far less than ten forward passes.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="pi0.5 forward pass: cameras and language into a cached prefix, then ten expert integration steps producing actions"
      >
        <Defs gid={gid} />

        {/* cameras */}
        {camY.map((y, i) => (
          <Box
            key={y}
            x={16}
            y={y}
            w={104}
            h={28}
            title={['top', 'left_wrist', 'right_wrist'][i]}
            sub={i === 2 ? 'absent → −1, mask 0' : '224×224, aspect kept'}
            tone={i === 2 ? 'muted' : 'data'}
            dashed={i === 2}
          />
        ))}
        {camY.map((y, i) => (
          <Arrow
            key={y}
            x1={122}
            y1={y + 14}
            x2={towerX - 2}
            y2={62}
            tone={i === 2 ? 'muted' : 'data'}
            marker={i === 2 ? `ar-muted-${gid}` : `ar-data-${gid}`}
          />
        ))}

        {/* the one shared tower */}
        <Box
          x={towerX}
          y={44}
          w={92}
          h={36}
          title="SigLIP"
          sub="ONE tower, shared"
          tone="data"
        />

        {/* language */}
        <Box
          x={16}
          y={158}
          w={104}
          h={34}
          title="prompt"
          sub="task + state"
        />
        <Arrow
          x1={122}
          y1={175}
          x2={towerX + 44}
          y2={104}
          tone="data"
          marker={`ar-data-${gid}`}
        />
        <text
          x={168}
          y={205}
          className="h-mono"
          fontSize={7.5}
          fill="var(--color-fd-muted-foreground)"
        >
          state is TEXT, see Fig. 3
        </text>

        {/* prefix */}
        <Arrow
          x1={towerX + 92}
          y1={62}
          x2={prefixX - 2}
          y2={62}
          tone="data"
          marker={`ar-data-${gid}`}
        />
        <Box
          x={prefixX}
          y={40}
          w={126}
          h={44}
          title="attention prefix"
          sub="img tokens ⧺ lang tokens"
        />
        <Arrow
          x1={prefixX + 63}
          y1={86}
          x2={prefixX + 63}
          y2={112}
          tone="data"
          marker={`ar-data-${gid}`}
        />
        <Box
          x={prefixX - 4}
          y={114}
          w={134}
          h={38}
          title="PaliGemma 2B"
          sub="runs ONCE per observation"
        />
        <Arrow
          x1={prefixX + 63}
          y1={154}
          x2={prefixX + 63}
          y2={180}
          tone="data"
          marker={`ar-data-${gid}`}
        />
        <Box
          x={prefixX + 6}
          y={182}
          w={114}
          h={32}
          title="KV cache"
          sub="past_key_values"
        />

        {/* the loop */}
        <rect
          x={498}
          y={30}
          width={206}
          height={232}
          rx={5}
          fill="var(--haller-action)"
          fillOpacity={0.05}
          stroke="var(--haller-action)"
          strokeOpacity={0.45}
          strokeWidth={1}
          strokeDasharray="4 3"
        />
        <text
          x={601}
          y={46}
          textAnchor="middle"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--haller-action)"
        >
          × 10 Euler steps
        </text>
        <text
          x={601}
          y={57}
          textAnchor="middle"
          className="h-mono"
          fontSize={7.5}
          fill="var(--color-fd-muted-foreground)"
        >
          t = 1 → 0, dt = −1/10
        </text>

        <Box
          x={524}
          y={68}
          w={154}
          h={32}
          title="noise x₁"
          sub="(chunk, 32)"
          tone="action"
        />
        <Arrow
          x1={601}
          y1={102}
          x2={601}
          y2={126}
          tone="action"
          marker={`ar-action-${gid}`}
        />
        <Box
          x={524}
          y={128}
          w={154}
          h={38}
          title="action expert 300M"
          sub="attends the cached prefix"
          tone="action"
        />
        <Arrow
          x1={601}
          y1={168}
          x2={601}
          y2={192}
          tone="action"
          marker={`ar-action-${gid}`}
        />
        <Box
          x={524}
          y={194}
          w={154}
          h={32}
          title="velocity v_t"
          sub="x_t ← x_t + dt·v_t"
          tone="action"
        />
        {/* feedback loop */}
        <path
          d={`M ${524} ${210} L ${506} ${210} L ${506} ${84} L ${524} ${84}`}
          fill="none"
          stroke="var(--haller-action)"
          strokeWidth={1.3}
          strokeOpacity={0.75}
          markerEnd={`url(#ar-action-${gid})`}
        />

        {/* cache feeds the expert */}
        <path
          d={`M ${prefixX + 120} ${198} L ${470} ${198} L ${470} ${147} L ${522} ${147}`}
          fill="none"
          stroke="var(--color-fd-primary)"
          strokeWidth={1.3}
          strokeOpacity={0.75}
          markerEnd={`url(#ar-data-${gid})`}
        />
        {/* sits clear of the KV-cache box, which ends at x=440 / y=214 */}
        <text
          x={476}
          y={232}
          textAnchor="middle"
          className="h-mono"
          fontSize={7.5}
          fill="var(--color-fd-muted-foreground)"
        >
          reused, not recomputed
        </text>

        {/* output + unpad */}
        <Arrow
          x1={601}
          y1={264}
          x2={601}
          y2={288}
          tone="action"
          marker={`ar-action-${gid}`}
        />
        <Box
          x={524}
          y={290}
          w={154}
          h={32}
          title="actions (chunk, 32)"
          sub="internal width"
          tone="action"
        />
        <Arrow
          x1={524}
          y1={306}
          x2={470}
          y2={306}
          tone="action"
          marker={`ar-action-${gid}`}
        />
        <Box
          x={316}
          y={290}
          w={152}
          h={32}
          title="actions (chunk, 6)"
          sub="unpadded to Haller's arm"
          tone="action"
        />

        {/* the pad/unpad annotation rail */}
        <line
          x1={316}
          y1={338}
          x2={678}
          y2={338}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
          strokeDasharray="2 3"
        />
        <text
          x={497}
          y={354}
          textAnchor="middle"
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          32 dims internally · sliced back to the dataset&rsquo;s real width
        </text>
        <text
          x={497}
          y={368}
          textAnchor="middle"
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          the training loss is truncated the same way, so padding never learns
        </text>
        <text
          x={497}
          y={386}
          textAnchor="middle"
          fontSize={8.5}
          fontStyle="italic"
          fill="var(--color-fd-muted-foreground)"
        >
          a 6-dim arm needs no architectural change at all
        </text>
      </svg>
    </Figure>
  );
}

/* ── 2. the state, as text ───────────────────────────────────────────────── */

const DEMO_STATE = [12.4, -38.9, 61.2, -5.7, 3.1, 88.0];
const DEMO_Q01 = [-90, -95, -10, -70, -95, 0];
const DEMO_Q99 = [90, 20, 130, 70, 95, 100];

/**
 * The single most surprising thing about π0.5 for anyone arriving from ACT:
 * joint angles are not embedded, they are DISCRETISED AND WRITTEN INTO THE
 * PROMPT. This figure walks one state vector down that ladder so the reader
 * sees the actual string the tokenizer receives.
 */
export function StateAsText() {
  const W = 720;
  const H = 268;
  const colW = 88;
  const x0 = 96;

  const norm = DEMO_STATE.map((v, i) =>
    Math.max(-1, Math.min(1, (2 * (v - DEMO_Q01[i])) / (DEMO_Q99[i] - DEMO_Q01[i]) - 1)),
  );
  /* np.digitize(v, linspace(-1,1,257)[:-1]) - 1 */
  const bins = norm.map((v) => {
    const edges = Array.from({ length: 256 }, (_, k) => -1 + (2 * k) / 256);
    let idx = 0;
    for (const e of edges) if (v >= e) idx++;
    return Math.max(0, Math.min(255, idx - 1));
  });

  const rowY = [64, 116, 168];

  return (
    <Figure
      label="Fig. 3: the robot's joint angles become words"
      caption={
        <>
          π0.5 has no separate state encoder. The pre-processor normalises the
          state to [−1, 1], bins it into <strong>256 integers</strong>, and
          splices those integers into the language prompt as literal text. The
          model reads the arm&rsquo;s pose the same way it reads the task. This
          is why quantile normalisation is load-bearing rather than a detail:
          the bin edges are fixed at ±1, so if one outlier stretched the scale,
          every real pose would collapse into a handful of bins.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="A six-dimensional joint state normalised, discretised into 256 bins, and written into a text prompt"
      >
        {/* row labels */}
        {[
          ['raw', 'degrees'],
          ['normalised', 'q01/q99 → ±1'],
          ['discretised', '256 bins'],
        ].map(([a, b], r) => (
          <g key={a}>
            <text
              x={x0 - 14}
              y={rowY[r] + 2}
              textAnchor="end"
              className="h-mono"
              fontSize={9}
              fontWeight={600}
              fill="var(--color-fd-foreground)"
            >
              {a}
            </text>
            <text
              x={x0 - 14}
              y={rowY[r] + 13}
              textAnchor="end"
              className="h-mono"
              fontSize={7}
              fill="var(--color-fd-muted-foreground)"
            >
              {b}
            </text>
          </g>
        ))}

        {/* joint names */}
        {['pan', 'lift', 'elbow', 'w.flex', 'w.roll', 'grip'].map((n, i) => (
          <text
            key={n}
            x={x0 + i * colW + colW / 2}
            y={38}
            textAnchor="middle"
            className="h-mono"
            fontSize={7.5}
            fill="var(--color-fd-muted-foreground)"
          >
            {n}
          </text>
        ))}

        {/* the three rows of values */}
        {[DEMO_STATE.map((v) => v.toFixed(1)), norm.map((v) => v.toFixed(2)), bins.map(String)].map(
          (vals, r) =>
            vals.map((v, i) => (
              <g key={`${r}-${i}`}>
                <rect
                  x={x0 + i * colW + 6}
                  y={rowY[r] - 13}
                  width={colW - 12}
                  height={24}
                  rx={2.5}
                  fill="var(--color-fd-primary)"
                  fillOpacity={r === 2 ? 0.2 : 0.1}
                  stroke="var(--color-fd-primary)"
                  strokeOpacity={0.55}
                  strokeWidth={1}
                />
                <text
                  x={x0 + i * colW + colW / 2}
                  y={rowY[r] + 3}
                  textAnchor="middle"
                  className="h-mono"
                  fontSize={9.5}
                  fill="var(--color-fd-foreground)"
                >
                  {v}
                </text>
              </g>
            )),
        )}

        {/* down arrows between rows */}
        {[0, 1].map((r) =>
          DEMO_STATE.map((_, i) => (
            <line
              key={`a-${r}-${i}`}
              x1={x0 + i * colW + colW / 2}
              y1={rowY[r] + 12}
              x2={x0 + i * colW + colW / 2}
              y2={rowY[r + 1] - 15}
              stroke="var(--color-fd-primary)"
              strokeWidth={1}
              strokeOpacity={0.45}
            />
          )),
        )}

        {/* the resulting prompt */}
        <rect
          x={x0 - 76}
          y={206}
          width={W - (x0 - 76) - 16}
          height={44}
          rx={3}
          fill="var(--haller-action)"
          fillOpacity={0.1}
          stroke="var(--haller-action)"
          strokeOpacity={0.6}
          strokeWidth={1}
        />
        <text
          x={x0 - 66}
          y={222}
          className="h-mono"
          fontSize={8}
          fill="var(--haller-action)"
          fontWeight={600}
        >
          the string the PaliGemma tokenizer actually receives
        </text>
        <text
          x={x0 - 66}
          y={239}
          className="h-mono"
          fontSize={9.5}
          fill="var(--color-fd-foreground)"
        >
          {`Task: pick up the cube…, State: ${bins.join(' ')};  Action:`}
        </text>
      </svg>
    </Figure>
  );
}

/* ── 3. why twenty trials cannot settle it ───────────────────────────────── */

/** Wilson score interval: the one you want for small n and proportions near the edges. */
function wilson(successes: number, n: number, z = 1.96) {
  const p = successes / n;
  const d = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / d;
  const half = (z / d) * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n));
  return { p, lo: Math.max(0, centre - half), hi: Math.min(1, centre + half) };
}

export function TrialResolution() {
  const [n, setN] = useState(20);
  const gid = useId().replace(/:/g, '_');

  /* Hold the observed rate at 45% and 70% and watch the intervals separate. */
  const a = useMemo(() => wilson(Math.round(0.45 * n), n), [n]);
  const b = useMemo(() => wilson(Math.round(0.7 * n), n), [n]);
  const overlap = a.hi > b.lo;

  const W = 720;
  const H = 176;
  const PAD_L = 92;
  const PAD_R = 24;
  const px = (W - PAD_L - PAD_R) / 100;
  const rowY = [64, 116];

  const bar = (
    r: number,
    ci: { p: number; lo: number; hi: number },
    label: string,
    tone: 'data' | 'action',
  ) => {
    const colour = tone === 'action' ? 'var(--haller-action)' : 'var(--color-fd-primary)';
    return (
      <g>
        <text
          x={PAD_L - 12}
          y={rowY[r] + 3}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-foreground)"
        >
          {label}
        </text>
        <line
          x1={PAD_L + ci.lo * 100 * px}
          y1={rowY[r]}
          x2={PAD_L + ci.hi * 100 * px}
          y2={rowY[r]}
          stroke={colour}
          strokeWidth={9}
          strokeOpacity={0.26}
          strokeLinecap="round"
        />
        {[ci.lo, ci.hi].map((e) => (
          <line
            key={e}
            x1={PAD_L + e * 100 * px}
            y1={rowY[r] - 7}
            x2={PAD_L + e * 100 * px}
            y2={rowY[r] + 7}
            stroke={colour}
            strokeWidth={1.4}
          />
        ))}
        <circle cx={PAD_L + ci.p * 100 * px} cy={rowY[r]} r={3.6} fill={colour} />
        <text
          x={PAD_L + ci.hi * 100 * px + 8}
          y={rowY[r] + 3}
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          {(ci.lo * 100).toFixed(0)}–{(ci.hi * 100).toFixed(0)}%
        </text>
      </g>
    );
  };

  return (
    <Figure
      label="Fig. 1: why twenty trials cannot settle a ten-point question"
      caption={
        <>
          Two policies, one truly at 45% and one truly at 70%, each scored{' '}
          <span className="h-mono">{n}</span> times. Drag the trial count: at 20
          the intervals <strong>overlap</strong>, so a good result and a lucky
          one are indistinguishable. Rollout trials, not GPU time, are the
          scarce resource on this project, which is why a change gets screened
          offline before it is allowed to spend any.
        </>
      }
    >
      <div className="mb-4 flex flex-col gap-2">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2">
          <label htmlFor={`n-${gid}`} className="h-label" style={{ letterSpacing: '0.14em' }}>
            trials per policy
          </label>
          <span
            className="h-mono text-[0.95rem] font-semibold tabular-nums"
            style={{ color: 'var(--haller-action)' }}
          >
            {n}
          </span>
          <span
            className="h-mono text-[11px]"
            style={{
              color: overlap ? 'var(--haller-action)' : 'var(--color-fd-muted-foreground)',
            }}
          >
            {overlap ? 'intervals overlap: cannot tell them apart' : 'intervals separate'}
          </span>
        </div>
        <input
          id={`n-${gid}`}
          type="range"
          min={10}
          max={200}
          step={5}
          value={n}
          onChange={(e) => setN(Number(e.target.value))}
          className="w-full accent-[var(--haller-action)]"
        />
      </div>

      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label={`Wilson confidence intervals for a 45% and a 70% policy at ${n} trials each`}
      >
        {/* axis */}
        <line
          x1={PAD_L}
          y1={148}
          x2={W - PAD_R}
          y2={148}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
        />
        {[0, 25, 50, 75, 100].map((t) => (
          <g key={t}>
            <line
              x1={PAD_L + t * px}
              y1={148}
              x2={PAD_L + t * px}
              y2={153}
              stroke="var(--color-fd-border)"
              strokeWidth={1}
            />
            <text
              x={PAD_L + t * px}
              y={165}
              textAnchor="middle"
              className="h-mono"
              fontSize={8.5}
              fill="var(--color-fd-muted-foreground)"
            >
              {t}%
            </text>
          </g>
        ))}
        <text
          x={PAD_L}
          y={26}
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          95% Wilson interval on the true success rate
        </text>

        {/* the overlap band, when there is one */}
        {overlap && (
          <rect
            x={PAD_L + b.lo * 100 * px}
            y={40}
            width={(a.hi - b.lo) * 100 * px}
            height={92}
            fill="var(--haller-action)"
            fillOpacity={0.09}
          />
        )}

        {bar(0, a, 'ACT 45%', 'data')}
        {bar(1, b, 'a 70% policy', 'data')}
      </svg>
    </Figure>
  );
}

/* ── 4. where the data comes from ────────────────────────────────────────── */

/**
 * The pipeline as it stands, with the two things that are not here yet drawn
 * dashed: the wrist channels, and the simulated episodes that can be generated
 * in the right shape before the hardware lands.
 */
export function DataPipeline() {
  const gid = useId().replace(/:/g, '_');
  const W = 720;
  const H = 288;

  return (
    <Figure
      label="Fig. 4: where the episodes come from, and where the next ones will"
      caption={
        <>
          Solid boxes exist today. Dashed ones do not yet: the wrist channels
          arrive with the hardware, and the simulator can produce episodes in
          the three-camera shape before then. The fork in the middle is the
          reason not to record more single-camera demonstrations now, because adding a
          camera changes the dataset schema, and episodes recorded on the old
          one cannot train a policy on the new one.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Data pipeline from teleoperation and simulation into policy training"
      >
        <Defs gid={gid} />

        {/* today's path */}
        <Box x={12} y={44} w={116} h={40} title="teleop" sub="113 episodes" />
        <Arrow x1={130} y1={64} x2={158} y2={64} marker={`ar-data-${gid}`} />
        <Box x={160} y={44} w={116} h={40} title="review" sub="91 keep / 22 reject" />
        <Arrow x1={278} y1={64} x2={306} y2={64} marker={`ar-data-${gid}`} />
        <Box x={308} y={44} w={126} h={40} title="1-camera set" sub="74,382 frames" />
        <Arrow x1={436} y1={64} x2={464} y2={64} marker={`ar-data-${gid}`} />
        <Box x={466} y={44} w={116} h={40} title="re-encode" sub="1.4 GB → 299 MB" />
        <Arrow x1={584} y1={64} x2={612} y2={64} marker={`ar-data-${gid}`} />
        <Box x={614} y={44} w={94} h={40} title="training" sub="in flight" tone="action" />

        {/* the schema fork */}
        <line
          x1={371}
          y1={90}
          x2={371}
          y2={126}
          stroke="var(--haller-action)"
          strokeWidth={1.4}
          strokeDasharray="4 3"
        />
        <text
          x={379}
          y={112}
          className="h-mono"
          fontSize={8.5}
          fontWeight={600}
          fill="var(--haller-action)"
        >
          adding a camera forks the schema here
        </text>

        {/* future: 3-camera */}
        <Box
          x={308}
          y={132}
          w={126}
          h={40}
          title="3-camera set"
          sub="top + 2 wrists"
          dashed
          tone="muted"
        />
        <Box
          x={12}
          y={132}
          w={116}
          h={40}
          title="wrist cams"
          sub="~6 Sept"
          dashed
          tone="muted"
        />
        <Arrow x1={130} y1={152} x2={306} y2={152} tone="muted" marker={`ar-muted-${gid}`} />

        {/* sim branch */}
        <Box
          x={12}
          y={206}
          w={116}
          h={44}
          title="simulator"
          sub="scripted expert"
          dashed
          tone="muted"
        />
        <Box
          x={160}
          y={206}
          w={116}
          h={44}
          title="auto-scored"
          sub="success predicate"
          dashed
          tone="muted"
        />
        <Arrow x1={130} y1={228} x2={158} y2={228} tone="muted" marker={`ar-muted-${gid}`} />
        <path
          d={`M ${278} ${228} L ${371} ${228} L ${371} ${174}`}
          fill="none"
          stroke="var(--color-fd-border)"
          strokeWidth={1.3}
          markerEnd={`url(#ar-muted-${gid})`}
        />

        {/* 3-camera also feeds training */}
        <path
          d={`M ${434} ${152} L ${661} ${152} L ${661} ${86}`}
          fill="none"
          stroke="var(--color-fd-border)"
          strokeWidth={1.3}
          strokeDasharray="4 3"
          markerEnd={`url(#ar-muted-${gid})`}
        />

        <text
          x={12}
          y={276}
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          the simulated route reaches the three-camera shape without waiting for the hardware
        </text>
      </svg>
    </Figure>
  );
}
