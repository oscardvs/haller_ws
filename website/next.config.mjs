import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

const isProd = process.env.NODE_ENV === 'production';
const repoName = 'haller_ws';
const ghUser = 'oscardvs';

/** @type {import('next').NextConfig} */
const config = {
  reactStrictMode: true,
  output: 'export',
  trailingSlash: true,
  images: { unoptimized: true },
  basePath: isProd ? `/${repoName}` : '',
  assetPrefix: isProd ? `https://${ghUser}.github.io/${repoName}/` : '',
};

export default withMDX(config);
