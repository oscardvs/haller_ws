import { RootProvider } from 'fumadocs-ui/provider/next';
import './global.css';
import { Fraunces, IBM_Plex_Sans, JetBrains_Mono } from 'next/font/google';
import StaticSearchDialog from '@/components/search-dialog';

const fraunces = Fraunces({
  subsets: ['latin'],
  variable: '--font-fraunces',
  axes: ['SOFT', 'opsz'],
  display: 'swap',
});

const plex = IBM_Plex_Sans({
  subsets: ['latin'],
  variable: '--font-plex',
  weight: ['300', '400', '500', '600'],
  display: 'swap',
});

const mono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-mono-jet',
  weight: ['400', '500'],
  display: 'swap',
});

export default function Layout({ children }: LayoutProps<'/'>) {
  return (
    <html
      lang="en"
      className={`${fraunces.variable} ${plex.variable} ${mono.variable}`}
      suppressHydrationWarning
    >
      <body className="flex flex-col min-h-screen">
        <RootProvider search={{ SearchDialog: StaticSearchDialog }}>
          {children}
        </RootProvider>
      </body>
    </html>
  );
}
