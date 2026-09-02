import defaultMdxComponents from 'fumadocs-ui/mdx';
import { ImageZoom } from 'fumadocs-ui/components/image-zoom';
import type { MDXComponents } from 'mdx/types';
import type { ComponentProps } from 'react';
import { Mermaid } from '@/components/mdx/mermaid';
import {
  ConfoundSquare,
  DataPipeline,
  Pi05Forward,
  StateAsText,
  StepsBetweenMetrics,
  TrialResolution,
} from '@/components/mdx/pi05-run';
import {
  BlindVsEnsembled,
  ChunkOverlap,
  EnsembleWeights,
} from '@/components/mdx/temporal-ensembling';

export function getMDXComponents(components?: MDXComponents) {
  return {
    ...defaultMdxComponents,
    // every image in the docs is click-to-enlarge, same as the diagrams
    img: (props: ComponentProps<'img'>) => (
      <ImageZoom {...(props as unknown as ComponentProps<typeof ImageZoom>)} />
    ),
    Mermaid,
    ChunkOverlap,
    EnsembleWeights,
    BlindVsEnsembled,
    Pi05Forward,
    StateAsText,
    TrialResolution,
    DataPipeline,
    ConfoundSquare,
    StepsBetweenMetrics,
    ...components,
  } satisfies MDXComponents;
}

export const useMDXComponents = getMDXComponents;

declare global {
  type MDXProvidedComponents = ReturnType<typeof getMDXComponents>;
}
