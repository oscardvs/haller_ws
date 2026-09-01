import Link from 'next/link';
import { ArrowUpRight } from 'lucide-react';

export default function HomePage() {
  return (
    <main className="relative flex flex-1 flex-col overflow-hidden">
      {/* full-bleed blueprint grid sits behind the hero only */}
      <div className="pointer-events-none absolute inset-x-0 top-0 h-[120vh] [mask-image:linear-gradient(to_bottom,black_0%,black_55%,transparent_100%)]">
        <div className="h-grid h-full w-full" />
      </div>

      <Hero />
      <SpecSheet />
      <RoutesIndex />
      <Footnote />
    </main>
  );
}

/* ────────────────────────────────────────────────────────────────────────── */

function Hero() {
  return (
    <section className="relative mx-auto w-full max-w-6xl px-6 pt-14 pb-24 sm:pt-20 md:pt-28">
      {/* registration marks pinned to the hero block */}
      <span className="h-cross" style={{ top: 24, left: 8 }} />
      <span className="h-cross" style={{ top: 24, right: 8 }} />

      {/* meta bar */}
      <div
        className="h-rise flex flex-wrap items-center gap-x-5 gap-y-2 text-fd-muted-foreground h-mono"
        style={{ animationDelay: '0ms' }}
      >
        <span className="inline-flex items-center gap-2 text-[11px]">
          <span className="h-status-dot" />
          <span>HALLER / V0.1 / OPEN HARDWARE</span>
        </span>
        <span className="hidden h-3 w-px bg-fd-border sm:block" />
        <span className="text-[11px]">N 50°50′ · E 04°21′ · BRUSSELS</span>
        <span className="hidden h-3 w-px bg-fd-border sm:block" />
        <span className="text-[11px]">DOCS · LAST PUSH 2026-05-23</span>
      </div>

      {/* title: serif display, italic accent, mono pip */}
      <h1
        className="h-rise mt-10 text-balance text-[clamp(2.6rem,7.4vw,6.4rem)] leading-[0.96] text-fd-foreground h-display"
        style={{ animationDelay: '80ms' }}
      >
        An open-source{' '}
        <span className="italic text-fd-primary" style={{ fontVariationSettings: "'opsz' 144, 'SOFT' 100" }}>
          bimanual
        </span>{' '}
        mobile manipulator,
        <br className="hidden sm:block" />
        built on ROS 2 and LeRobot.
      </h1>

      {/* deck / lede */}
      <p
        className="h-rise mt-8 max-w-2xl text-pretty text-[1.05rem] leading-[1.55] text-fd-muted-foreground sm:text-[1.12rem]"
        style={{ animationDelay: '160ms' }}
      >
        Haller is a three-wheel differential-drive base (two driven front wheels and a rear caster) carrying two SO-101 arms.
        ROS&nbsp;2 runs the base, LeRobot runs the arms, and one FastAPI&nbsp;+&nbsp;Next.js HMI
        controls both from the browser.{' '}
        <span className="italic text-fd-foreground">Today the arms are the working rig: calibration,
        WebXR teleop from a Meta Quest, and dataset recording all run through the HMI.</span>
      </p>

      {/* CTAs */}
      <div
        className="h-rise mt-10 flex flex-wrap items-center gap-3"
        style={{ animationDelay: '240ms' }}
      >
        <Link
          href="/docs"
          className="group inline-flex h-11 items-center gap-2 rounded-md px-5 text-[13px] font-medium tracking-wide shadow-[0_1px_0_0_rgba(0,0,0,0.08)] transition hover:translate-y-[-1px] hover:shadow-[0_4px_18px_-6px_var(--haller-action)]"
          style={{ background: 'var(--haller-action)', color: 'var(--haller-action-fg)' }}
        >
          <span className="h-mono text-[10px] opacity-80">→</span>
          Read the documentation
        </Link>
        <Link
          href="https://github.com/oscardvs/haller_ws"
          className="group inline-flex h-11 items-center gap-2 rounded-md border border-fd-border bg-fd-card/40 px-5 text-[13px] font-medium tracking-wide text-fd-foreground backdrop-blur transition hover:border-fd-primary/60 hover:bg-fd-card"
        >
          View on GitHub
          <ArrowUpRight className="h-3.5 w-3.5 opacity-70 transition group-hover:translate-x-0.5 group-hover:-translate-y-0.5" />
        </Link>
        <Link
          href="/docs/architecture"
          className="ml-1 inline-flex h-11 items-center text-[13px] text-fd-muted-foreground underline underline-offset-4 decoration-fd-border transition hover:text-fd-primary hover:decoration-fd-primary/60"
        >
          See the architecture diagram
        </Link>
      </div>
    </section>
  );
}

/* ────────────────────────────────────────────────────────────────────────── */

const SPECS = [
  {
    n: '01',
    tag: 'BASE',
    title: 'ROS 2 differential drive',
    body: 'LK-TECH MF5010 BLDCs over CAN at 1 Mbps. RPLIDAR A1M8 publishes 2D scans into Nav2. The whole stack runs on a Jetson Orin Nano with a 0.5 s cmd_vel watchdog as the secondary safety.',
    stat: ['/cmd_vel', '50 Hz odom'],
  },
  {
    n: '02',
    tag: 'ARMS',
    title: 'Two symmetric SO-101 arms',
    body: 'Open hardware from TheRobotStudio: 6× Feetech STS3215 servos each, on half-duplex USB-TTL buses. Either arm can lead or follow. Calibration sidecars are saved and backed up on every write.',
    stat: ['6 DoF × 2', 'in-browser calib'],
  },
  {
    n: '03',
    tag: 'HMI',
    title: 'One browser HMI for arms and base',
    body: 'A Next.js + FastAPI dashboard: joint sliders, pose presets, MJPEG feeds, leader↔follower teleop at 60 Hz, WebXR teleop from a Meta Quest, and an in-process dataset recorder, all behind a single E-STOP.',
    stat: ['60 Hz teleop', '20 Hz telemetry'],
  },
];

function SpecSheet() {
  return (
    <section className="relative mx-auto w-full max-w-6xl px-6 pt-6 pb-24">
      <header className="mb-10 flex items-baseline justify-between">
        <span className="h-label">[ Spec Sheet · 03 ]</span>
        <span className="h-mono text-[10px] text-fd-muted-foreground">
          §01–03 / overview
        </span>
      </header>

      <div className="h-rule mb-8" />

      <ol className="grid gap-px overflow-hidden rounded-md border border-fd-border bg-fd-border md:grid-cols-3">
        {SPECS.map((s) => (
          <li
            key={s.n}
            className="group relative flex flex-col bg-fd-background p-7 transition hover:bg-fd-card"
          >
            <div className="flex items-baseline justify-between">
              <span className="h-mono text-[11px] tracking-widest text-fd-primary">
                [{s.n}]
              </span>
              <span className="h-label">{s.tag}</span>
            </div>

            <h3
              className="mt-6 text-[1.55rem] leading-[1.1] text-fd-foreground h-display"
            >
              {s.title}
            </h3>

            <p className="mt-4 text-[0.95rem] leading-[1.55] text-fd-muted-foreground">
              {s.body}
            </p>

            <ul className="mt-6 flex flex-wrap gap-x-4 gap-y-1 border-t border-fd-border pt-4">
              {s.stat.map((k) => (
                <li
                  key={k}
                  className="h-mono text-[10.5px] uppercase tracking-[0.14em] text-fd-muted-foreground"
                >
                  · {k}
                </li>
              ))}
            </ul>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ────────────────────────────────────────────────────────────────────────── */

const ROUTES: { label: string; href: string; hint: string }[] = [
  { label: 'Architecture', href: '/docs/architecture', hint: 'one-screen mental model' },
  { label: 'Bill of materials', href: '/docs/hardware/bom', hint: 'parts & sourcing' },
  { label: 'Build', href: '/docs/hardware/build', hint: 'mechanical assembly' },
  { label: 'Mobile base (ROS 2)', href: '/docs/base/overview', hint: 'bringup & drive' },
  { label: 'Vision pipeline', href: '/docs/base/vision', hint: 'detection + seg + costmap' },
  { label: 'LeRobot environment', href: '/docs/setup/lerobot-environment', hint: 'python tooling' },
  { label: 'SO-101 arms', href: '/docs/setup/so101-arm', hint: 'calibrate & test' },
  { label: 'HMI overview', href: '/docs/hmi/overview', hint: 'web frontend' },
  { label: 'Calibration wizard', href: '/docs/hmi/calibration-wizard', hint: 'browser flow' },
  { label: 'Leader ↔ follower teleop', href: '/docs/hmi/teleop', hint: '60 Hz mirror' },
  { label: 'Human-pose teleop', href: '/docs/hmi/human-teleop', hint: 'webcam → joints' },
  { label: 'Dataset collection', href: '/docs/data/dataset-collection', hint: 'record & push' },
  { label: 'RunPod inference', href: '/docs/data/runpod-inference', hint: 'cloud GPU' },
  { label: 'Jetson deployment', href: '/docs/deployment/jetson', hint: 'on-robot install' },
  { label: 'Wi-Fi AP fallback', href: '/docs/deployment/wifi-ap', hint: 'HallerRobot SSID' },
  { label: 'Troubleshooting', href: '/docs/troubleshooting', hint: 'symptoms and fixes' },
];

function RoutesIndex() {
  return (
    <section className="relative mx-auto w-full max-w-6xl px-6 pb-28">
      <header className="mb-6 flex items-baseline justify-between">
        <span className="h-label">[ Index ]</span>
        <span className="h-mono text-[10px] text-fd-muted-foreground">
          {ROUTES.length.toString().padStart(2, '0')} entries
        </span>
      </header>
      <div className="h-rule mb-6" />

      <ul className="divide-y divide-fd-border border-y border-fd-border">
        {ROUTES.map((r, i) => (
          <li key={r.href}>
            <Link
              href={r.href}
              className="group flex items-baseline gap-4 py-3.5 transition hover:bg-fd-card hover:pl-2"
            >
              <span className="h-mono w-8 shrink-0 text-[11px] text-fd-muted-foreground">
                {(i + 1).toString().padStart(2, '0')}
              </span>
              <span className="flex-1 text-[1rem] text-fd-foreground transition group-hover:text-fd-primary">
                {r.label}
              </span>
              <span className="hidden sm:inline h-mono text-[10.5px] uppercase tracking-[0.14em] text-fd-muted-foreground">
                {r.hint}
              </span>
              <ArrowUpRight className="h-3.5 w-3.5 -translate-x-1 opacity-0 transition group-hover:translate-x-0 group-hover:opacity-70" />
            </Link>
          </li>
        ))}
      </ul>
    </section>
  );
}

/* ────────────────────────────────────────────────────────────────────────── */

function Footnote() {
  return (
    <section className="mx-auto w-full max-w-6xl px-6 pb-20">
      <div className="h-rule mb-8" />
      <div className="flex flex-wrap items-end justify-between gap-6">
        <p className="max-w-xl text-sm leading-relaxed text-fd-muted-foreground">
          Apache-2.0. The SO-101 mechanical design is licensed under its own
          terms by{' '}
          <Link
            href="https://github.com/TheRobotStudio/SO-ARM100"
            className="text-fd-foreground underline underline-offset-4 decoration-fd-border hover:text-fd-primary hover:decoration-fd-primary/60"
          >
            TheRobotStudio
          </Link>
          .
        </p>
        <p className="h-mono text-[10px] uppercase tracking-[0.18em] text-fd-muted-foreground">
          Set in Fraunces · IBM Plex Sans · JetBrains Mono
        </p>
      </div>
    </section>
  );
}
