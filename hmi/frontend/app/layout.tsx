// hmi/frontend/app/layout.tsx
import type { Metadata } from "next";
import { Geist, JetBrains_Mono } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";
import { ThemeProvider } from "@/components/ThemeProvider";

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

/**
 * The root layout owns the document and nothing else.
 *
 * It used to also render the wordmark, nav and E-STOP for every route. The
 * cockpit at `/` is a fixed-viewport surface that draws its own header, rail
 * and command bar, so a second app-level header would either stack on top of
 * it or push it past 100vh. The remaining deep-link routes (`/base`,
 * `/arm/[id]`, `/settings`, `/teleop/human`) render <DeepLinkChrome /> for
 * themselves instead — same wordmark, same rail, same always-present E-STOP.
 */
export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    // suppressHydrationWarning: next-themes stamps the theme class onto <html>
    // before paint, so the server-rendered markup deliberately disagrees.
    <html lang="en" suppressHydrationWarning>
      <body
        className={`${geistSansBody.variable} ${geistSans.variable} ${jetbrainsMono.variable} antialiased bg-background text-foreground`}
      >
        <ThemeProvider>
          {children}
          {/* offset clears the 34px command bar, so a toast never lands on the
              Teleop/Record controls it is reporting on. */}
          <Toaster richColors closeButton position="bottom-right" offset={44} />
        </ThemeProvider>
      </body>
    </html>
  );
}
