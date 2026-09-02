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

/* ── 5. the 2x2 that separated the arm from the checkpoint ───────────────── */

/**
 * The confound, drawn as the square it is. Comparing each arm at its own
 * eval-loss-best checkpoint walks the DIAGONAL of a 2x2, which changes two
 * things at once. Filling in the other two cells recovers the two effects
 * separately, and the two routes around the square must sum to the diagonal,
 * which is a free check that the cells hold the same episodes.
 */
export function ConfoundSquare() {
  const gid = useId().replace(/:/g, '_');
  const W = 740;
  const H = 332;

  const colA = 118;
  const colB = 458;
  const cw = 190;
  const rowT = 84;
  const rowB = 224;
  const ch = 56;
  const cxA = colA + cw / 2;
  const cxB = colB + cw / 2;
  const cyT = rowT + ch / 2;
  const cyB = rowB + ch / 2;

  const edge = (
    x: number,
    y: number,
    est: string,
    t: string,
    dir: string,
    anchor: 'middle' | 'start' | 'end',
  ) => (
    <g>
      <text
        x={x}
        y={y}
        textAnchor={anchor}
        className="h-mono"
        fontSize={9}
        fontWeight={600}
        fill="var(--color-fd-foreground)"
      >
        {est}
      </text>
      <text
        x={x}
        y={y + 12}
        textAnchor={anchor}
        className="h-mono"
        fontSize={8}
        fill="var(--color-fd-muted-foreground)"
      >
        {t}
      </text>
      <text
        x={x}
        y={y + 23}
        textAnchor={anchor}
        className="h-mono"
        fontSize={8}
        fill="var(--color-fd-muted-foreground)"
      >
        {dir}
      </text>
    </g>
  );

  return (
    <Figure
      label="Fig. 5: one comparison, two variables, and the square that pulls them apart"
      caption={
        <>
          Held-out replay MAE, in degrees, over the eight episodes present in
          all four cells and paired per episode, which is why the cell means
          differ a little from the fourteen-episode means in the text. The two{' '}
          <strong>horizontal</strong> edges are the effect under test, the arm.
          The two <strong>vertical</strong> edges are a checkpoint-step effect
          nobody was trying to measure. Comparing each arm at its own
          eval-loss-best checkpoint takes the dashed diagonal, which is the sum
          of one of each and so overstates the arm effect. The square also
          audits itself: both routes from{' '}
          <span className="h-mono">dirty@1500</span> to{' '}
          <span className="h-mono">clean@2000</span> must add to the same
          number, and they agree to a thousandth of a degree.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="A two by two of arm (dirty, clean) against checkpoint step (1500, 2000), with the confounded diagonal drawn across it"
      >
        <Defs gid={gid} />

        {/* column headers */}
        <text
          x={cxA}
          y={30}
          textAnchor="middle"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-muted-foreground)"
        >
          Arm A: dirty, 99 episodes
        </text>
        <text
          x={cxB}
          y={30}
          textAnchor="middle"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-muted-foreground)"
        >
          Arm B: clean, 77 episodes
        </text>

        {/* row labels */}
        <text
          x={colA - 12}
          y={cyT + 3}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-muted-foreground)"
        >
          step 1,500
        </text>
        <text
          x={colA - 12}
          y={cyB + 3}
          textAnchor="end"
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--color-fd-muted-foreground)"
        >
          step 2,000
        </text>

        {/* the diagonal actually compared: cell edge to cell edge, so it
            crosses neither box's text nor either straight edge of the square */}
        <line
          x1={colA + cw - 20}
          y1={rowT + ch}
          x2={colB + 20}
          y2={rowB}
          stroke="var(--haller-action)"
          strokeWidth={1.4}
          strokeDasharray="5 4"
          markerEnd={`url(#ar-action-${gid})`}
        />

        {/* the four cells */}
        <Box
          x={colA}
          y={rowT}
          w={cw}
          h={ch}
          title="dirty @ 1,500"
          sub="12.142 deg, its eval-loss best"
        />
        <Box
          x={colB}
          y={rowT}
          w={cw}
          h={ch}
          title="clean @ 1,500"
          sub="11.471 deg, added by the square"
        />
        <Box
          x={colA}
          y={rowB}
          w={cw}
          h={ch}
          title="dirty @ 2,000"
          sub="11.767 deg, added by the square"
        />
        <Box
          x={colB}
          y={rowB}
          w={cw}
          h={ch}
          title="clean @ 2,000"
          sub="10.435 deg, nearest to its best"
        />

        {/* arm effect, one per row: labelled away from the middle band, which
            belongs to the step effects and the diagonal */}
        <Arrow x1={colA + cw} y1={cyT} x2={colB - 4} y2={cyT} marker={`ar-data-${gid}`} />
        {edge(383, cyT - 30, '+0.670 deg', 't = 1.65', 'clean better, 7/8', 'middle')}
        <Arrow x1={colA + cw} y1={cyB} x2={colB - 4} y2={cyB} marker={`ar-data-${gid}`} />
        {edge(383, cyB + 14, '+1.332 deg', 't = 3.13', 'clean better, 7/8', 'middle')}

        {/* step effect, one per column */}
        <Arrow x1={cxA} y1={rowT + ch} x2={cxA} y2={rowB - 4} marker={`ar-data-${gid}`} />
        {edge(cxA - 10, 172, '+0.375 deg', 't = 0.74', '2,000 better, 6/8', 'end')}
        <Arrow x1={cxB} y1={rowT + ch} x2={cxB} y2={rowB - 4} marker={`ar-data-${gid}`} />
        {edge(cxB + 10, 172, '+1.036 deg', 't = 2.45', '2,000 better, 7/8', 'start')}

        {/* the diagonal is labelled as a legend rather than in the middle of
            the square, where a two-line caption and a descending line collide */}
        <line
          x1={12}
          y1={299}
          x2={44}
          y2={299}
          stroke="var(--haller-action)"
          strokeWidth={1.4}
          strokeDasharray="5 4"
        />
        <text
          x={52}
          y={302}
          className="h-mono"
          fontSize={9}
          fontWeight={600}
          fill="var(--haller-action)"
        >
          the comparison actually made: +1.707 deg, the arm effect and the step effect summed
        </text>
        <text
          x={52}
          y={318}
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          0.375 + 1.332 = 1.707 across the top-then-right route; 0.670 + 1.036 = 1.706 across the
          left-then-bottom one
        </text>
      </svg>
    </Figure>
  );
}

/* ── 6. the ten steps between the two metrics ────────────────────────────── */

/**
 * Why an effect that replay-eval MAE resolves at t = 3.13 can be invisible to
 * eval_loss on the same checkpoint and the same held-out episodes. The two
 * metrics tap different points of the same model: one scores a single denoise
 * against a noised copy of the answer, the other scores the trajectory that
 * comes out of the full integrator.
 */
export function StepsBetweenMetrics() {
  const gid = useId().replace(/:/g, '_');
  const W = 720;
  const H = 246;

  return (
    <Figure
      label="Fig. 6: the ten steps that one metric never runs"
      caption={
        <>
          Same checkpoint, same held-out episodes, two different quantities.{' '}
          <span className="h-mono">eval_loss</span> is the training objective
          with gradients off: it corrupts the known answer to a random degree
          and asks which way to push. Replay-eval MAE starts from pure noise,
          runs the whole integrator, and measures the joint angles that come
          out. A change can move either one while leaving the other flat, which
          is exactly what happened, so only the lower lane belongs in a decision
          about hardware.
        </>
      }
    >
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="w-full"
        role="img"
        aria-label="Two lanes: the eval loss path scoring a single denoising step, and the replay eval path running ten integration steps to an emitted action"
      >
        <Defs gid={gid} />

        {/* eval_loss lane */}
        <text x={12} y={26} className="h-label" fontSize={8.5} fill="var(--color-fd-muted-foreground)">
          eval_loss: a denoising score
        </text>
        <Box x={12} y={38} w={132} h={40} title="true chunk" sub="the answer" tone="muted" />
        <Arrow x1={146} y1={58} x2={172} y2={58} tone="muted" marker={`ar-muted-${gid}`} />
        <Box x={174} y={38} w={148} h={40} title="corrupt at random t" sub="t ~ Beta(1.5, 1.0)" />
        <Arrow x1={324} y1={58} x2={350} y2={58} marker={`ar-data-${gid}`} />
        <Box x={352} y={38} w={140} h={40} title="one denoise call" sub="predict a velocity" />
        <Arrow x1={494} y1={58} x2={520} y2={58} marker={`ar-data-${gid}`} />
        <Box x={522} y={38} w={186} h={40} title="MSE vs true velocity" sub="never emits an action" />

        {/* the shared middle */}
        <line
          x1={12}
          y1={106}
          x2={708}
          y2={106}
          stroke="var(--color-fd-border)"
          strokeWidth={1}
          strokeDasharray="3 4"
        />
        <text
          x={360}
          y={122}
          textAnchor="middle"
          className="h-mono"
          fontSize={8.5}
          fill="var(--color-fd-muted-foreground)"
        >
          one checkpoint, one held-out set, measured twice
        </text>

        {/* MAE lane */}
        <text x={12} y={152} className="h-label" fontSize={8.5} fill="var(--haller-action)">
          replay-eval MAE: a control error
        </text>
        <Box x={12} y={164} w={132} h={40} title="pure noise" sub="no answer given" tone="muted" />
        <Arrow x1={146} y1={184} x2={172} y2={184} tone="muted" marker={`ar-muted-${gid}`} />
        <Box
          x={174}
          y={164}
          w={148}
          h={40}
          title="10 Euler steps"
          sub="t = 1 down to 0"
          tone="action"
        />
        <Arrow x1={324} y1={184} x2={350} y2={184} tone="action" marker={`ar-action-${gid}`} />
        <Box x={352} y={164} w={140} h={40} title="emitted action" sub="6 joints, degrees" tone="action" />
        <Arrow x1={494} y1={184} x2={520} y2={184} tone="action" marker={`ar-action-${gid}`} />
        <Box
          x={522}
          y={164}
          w={186}
          h={40}
          title="MAE vs the human"
          sub="what the arm would be told"
          tone="action"
        />

        <text
          x={12}
          y={230}
          className="h-mono"
          fontSize={8}
          fill="var(--color-fd-muted-foreground)"
        >
          the errors of ten successive velocity predictions compose along the lower lane; the upper
          lane scores one of them, in isolation
        </text>
      </svg>
    </Figure>
  );
}
