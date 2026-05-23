import type { BaseLayoutProps } from 'fumadocs-ui/layouts/shared';
import { gitConfig } from './shared';

function Wordmark() {
  return (
    <span className="h-wordmark">
      <span className="bracket">[</span>
      <span>Haller</span>
      <span className="bracket">]</span>
    </span>
  );
}

export function baseOptions(): BaseLayoutProps {
  return {
    nav: {
      title: <Wordmark />,
      transparentMode: 'top',
    },
    githubUrl: `https://github.com/${gitConfig.user}/${gitConfig.repo}`,
  };
}
