// hmi/frontend/app/layout.tsx
import type { Metadata } from "next";
import Link from "next/link";
import { Geist, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { EStopButton } from "@/components/EStopButton";
import { TelemetryBar } from "@/components/TelemetryBar";

const geistSans = JetBrains_Mono({
  variable: "--font-sans-default",
  subsets: ["latin"],
  display: "swap",
});
const geistSansBody = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
  display: "swap",
});
const jetbrainsMono = JetBrains_Mono({
  variable: "--font-mono-default",
  subsets: ["latin"],
  display: "swap",
});

export const metadata: Metadata = {
  title: "Haller HMI",
  description: "Unified supervisory-control surface for the Haller robot",
};

function Wordmark() {
  return (
    <Link
      href="/"
      className="group inline-flex items-center gap-2 select-none"
      aria-label="Haller HMI — dashboard"
    >
      {/* Status dot — visual identity anchor. Pulses with the live ring. */}
      <span className="relative inline-flex h-2 w-2">
        <span className="absolute inline-flex h-full w-full rounded-full bg-[var(--haller-live)] opacity-70 animate-haller-ping" />
        <span className="relative inline-flex h-2 w-2 rounded-full bg-[var(--haller-live)]" />
      </span>
      <span className="font-mono text-[13px] font-semibold tracking-[0.32em] uppercase text-foreground">
        Haller
      </span>
      <span className="font-mono text-[10px] tracking-[0.32em] uppercase text-muted-foreground border-l border-border pl-2">
        HMI&nbsp;·&nbsp;v0.1
      </span>
    </Link>
  );
}

function NavLink({ href, label }: { href: string; label: string }) {
  return (
    <Link
      href={href}
      className="label-micro text-muted-foreground hover:text-foreground transition-colors px-2 py-1 rounded-sm hover:bg-muted/50"
    >
      {label}
    </Link>
  );
}

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSansBody.variable} ${geistSans.variable} ${jetbrainsMono.variable} antialiased bg-background text-foreground min-h-screen`}
      >
        {/* Top app rail: wordmark + nav + persistent E-STOP.
            Sticky so it's always anchored — operator never loses E-STOP. */}
        <header className="sticky top-0 z-40 border-b border-border bg-background/85 backdrop-blur supports-[backdrop-filter]:bg-background/70">
          <div className="flex items-center justify-between px-3 h-11">
            <div className="flex items-center gap-6">
              <Wordmark />
              <nav className="flex items-center gap-1">
                <NavLink href="/" label="Overview" />
                <NavLink href="/base" label="Base" />
                <NavLink href="/settings" label="Settings" />
              </nav>
            </div>
            <EStopButton />
          </div>
          {/* Persistent telemetry rail under header — visible on every page. */}
          <TelemetryBar />
        </header>
        {children}
        <Toaster richColors closeButton theme="dark" position="bottom-right" />
      </body>
    </html>
  );
}
