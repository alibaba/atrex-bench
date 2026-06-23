import { defineConfig } from 'astro/config';
import tailwind from '@astrojs/tailwind';

export default defineConfig({
  site: 'https://atrex-bench.github.io',
  integrations: [tailwind()],
  output: 'static',
});
