import Link from 'next/link';

export default function HomePage() {
  return (
    <main className="flex flex-1 flex-col">
      <section className="flex flex-col items-center justify-center px-6 py-24 text-center md:py-32">
        <p className="mb-4 font-mono text-xs uppercase tracking-[0.18em] text-fd-muted-foreground">
          open hardware + software
        </p>
        <h1 className="max-w-3xl text-4xl font-semibold leading-tight tracking-tight md:text-6xl">
          Haller — a hackable bimanual mobile manipulator.
        </h1>
        <p className="mt-6 max-w-2xl text-base text-fd-muted-foreground md:text-lg">
          A four-wheel differential-drive base carrying two SO-101 arms.
          ROS 2 for the base, LeRobot for the arms, a unified FastAPI + Next.js HMI
          that ties it all together. Build it, calibrate it in the browser, teleoperate it
          across the room.
        </p>
        <div className="mt-10 flex flex-wrap items-center justify-center gap-3">
          <Link
            href="/docs"
            className="inline-flex h-10 items-center rounded-md bg-fd-primary px-5 text-sm font-medium text-fd-primary-foreground transition hover:opacity-90"
          >
            Read the docs
          </Link>
          <Link
            href="https://github.com/oscardvs/haller_ws"
            className="inline-flex h-10 items-center rounded-md border border-fd-border px-5 text-sm font-medium transition hover:bg-fd-accent"
          >
            View on GitHub
          </Link>
        </div>
      </section>

      <section className="mx-auto grid w-full max-w-5xl gap-4 px-6 pb-24 md:grid-cols-3">
        <Feature
          title="ROS 2 base"
          body="Differential drive on LK-TECH MF5010 BLDCs over CAN. RPLIDAR A1M8 for 2D mapping. Runs on Jetson Orin Nano."
        />
        <Feature
          title="Two SO-101 arms"
          body="Open hardware design from TheRobotStudio, 6× Feetech STS3215 servos each. Symmetric build — either arm can lead or follow."
        />
        <Feature
          title="Browser HMI"
          body="One unified web interface for both arms and the base. Joint sliders, teleop launcher, calibration wizard, dataset recording."
        />
      </section>
    </main>
  );
}

function Feature({ title, body }: { title: string; body: string }) {
  return (
    <div className="rounded-lg border border-fd-border bg-fd-card p-5">
      <h3 className="mb-2 font-medium">{title}</h3>
      <p className="text-sm text-fd-muted-foreground">{body}</p>
    </div>
  );
}
