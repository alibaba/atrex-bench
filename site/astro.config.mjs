import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://alibaba.github.io',
  base: '/atrex-bench',
  integrations: [tailwind()],
  output: 'static',
});
