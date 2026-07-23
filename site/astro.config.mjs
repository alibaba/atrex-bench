import { defineConfig } from 'astro/config';
import { loadEnv } from 'vite';
import tailwind from '@astrojs/tailwind';

const mode = process.argv.includes('--mode')
  ? process.argv[process.argv.indexOf('--mode') + 1]
  : (process.env.NODE_ENV || 'production');
const env = loadEnv(mode, process.cwd(), '');

export default defineConfig({
  site: env.SITE_URL || 'https://alibaba.github.io',
  base: env.SITE_BASE || '/atrex-bench',
  integrations: [tailwind()],
  output: 'static',
});
