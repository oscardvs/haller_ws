import { getPageImage, getPageMarkdownUrl, source } from '@/lib/source';
import {
  DocsBody,
  DocsDescription,
  DocsPage,
  DocsTitle,
  MarkdownCopyButton,
  ViewOptionsPopover,
} from 'fumadocs-ui/layouts/docs/page';
import { notFound } from 'next/navigation';
import { getMDXComponents } from '@/components/mdx';
import type { Metadata } from 'next';
import { createRelativeLink } from 'fumadocs-ui/mdx';
import { gitConfig } from '@/lib/shared';

export default async function Page(props: PageProps<'/docs/[[...slug]]'>) {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  const MDX = page.data.body;
  const markdownUrl = getPageMarkdownUrl(page).url;

  /* "§ HMI / Calibration" — section pip above the title.
     Skips for the docs index where slugs is empty. */
  const slugs = page.slugs;
  const sectionLabel = formatSectionLabel(slugs);

  return (
    <DocsPage toc={page.data.toc} full={page.data.full}>
      {sectionLabel && (
        <div className="mb-3 flex items-center gap-2">
          <span className="h-mono text-[10.5px] tracking-[0.18em] uppercase text-fd-primary">
            §
          </span>
          <span className="h-label">{sectionLabel}</span>
        </div>
      )}
      <DocsTitle>{page.data.title}</DocsTitle>
      <DocsDescription className="mb-0">{page.data.description}</DocsDescription>
      <div className="flex flex-row gap-2 items-center border-b pb-6">
        <MarkdownCopyButton markdownUrl={markdownUrl} />
        <ViewOptionsPopover
          markdownUrl={markdownUrl}
          githubUrl={`https://github.com/${gitConfig.user}/${gitConfig.repo}/blob/${gitConfig.branch}/content/docs/${page.path}`}
        />
      </div>
      <DocsBody>
        <MDX
          components={getMDXComponents({
            a: createRelativeLink(source, page),
          })}
        />
      </DocsBody>
    </DocsPage>
  );
}

function formatSectionLabel(slugs: string[]): string | null {
  /* Only render the label for nested pages — for top-level entries the H1
     already shows the same word, so a "Architecture / Architecture" pair
     would be noise. */
  if (slugs.length < 2) return null;
  const pretty = (s: string) =>
    s
      .replace(/-/g, ' ')
      .replace(/\b\w/g, (c) => c.toUpperCase())
      .replace(/Hmi/g, 'HMI')
      .replace(/Ros/g, 'ROS')
      .replace(/\bSo101\b/gi, 'SO-101')
      .replace(/\bBom\b/gi, 'BOM');
  return pretty(slugs[0]);
}

export async function generateStaticParams() {
  return source.generateParams();
}

export async function generateMetadata(props: PageProps<'/docs/[[...slug]]'>): Promise<Metadata> {
  const params = await props.params;
  const page = source.getPage(params.slug);
  if (!page) notFound();

  return {
    title: page.data.title,
    description: page.data.description,
    openGraph: {
      images: getPageImage(page).url,
    },
  };
}
