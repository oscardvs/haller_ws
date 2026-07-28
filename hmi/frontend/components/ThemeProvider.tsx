"use client";

/**
 * Theme plumbing for the whole app.
 *
 * `attribute="class"` is not the next-themes default — it is what this repo's
 * Tailwind setup requires: globals.css declares
 * `@custom-variant dark (&:is(.dark *))`, so every `dark:` utility keys off a
 * `.dark` class, not a `data-theme` attribute. The v3 design expressed its
 * themes as `[data-theme="dark|light"]`; that is the design tool's convention,
 * and it is adapted to the class here rather than the other way round.
 *
 * System preference is off on purpose. This is a machine-room surface that
 * ships dark; an operator who wants light asks for it once in Settings and it
 * sticks, rather than flipping under them when the OS crosses sunset.
 */
import { ThemeProvider as NextThemesProvider } from "next-themes";

export function ThemeProvider({ children }: { children: React.ReactNode }) {
  return (
    <NextThemesProvider
      attribute="class"
      defaultTheme="dark"
      enableSystem={false}
      disableTransitionOnChange
    >
      {children}
    </NextThemesProvider>
  );
}
