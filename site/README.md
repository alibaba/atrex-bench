# Atrex-Bench Site

Static site for the Atrex-Bench project, built with [Astro](https://astro.build/) + [Tailwind CSS](https://tailwindcss.com/).

## Local Development

```bash
# 1. Extract operator data from the repo (generates src/data/operators.json)
python3 site/scripts/extract-data.py

# 2. Install dependencies
cd site
npm install

# 3. Start dev server
npm run dev
```

## Build

```bash
cd site
npm run build       # output → site/dist/
npm run preview     # preview the build locally
```

## Deploy

`site/dist/` is a self-contained static site — upload it to any static host
(GitHub Pages, Netlify, Vercel, Cloudflare Pages, S3 + CloudFront, …).

```bash
# From repo root
python3 site/scripts/extract-data.py
cd site && npm ci && npm run build
# then publish site/dist/ to your host
```

If deploying under a sub-path, set `base` in `astro.config.mjs` (e.g. `base: '/atrex-bench'`);
for a domain root use `base: '/'`.

> This repository does not ship a CI workflow for deployment. To publish
> automatically, add your own GitHub Actions / GitLab CI pipeline that runs
> `extract-data.py` + `npm run build` and uploads `site/dist/`.

## Project Structure

```
site/
├── astro.config.mjs          # Astro config (base path, integrations)
├── package.json
├── tailwind.config.mjs
├── scripts/
│   └── extract-data.py       # Extracts operator data → src/data/operators.json
├── public/
│   └── favicon.svg
└── src/
    ├── layouts/Layout.astro   # Shared layout (nav, footer, i18n script)
    ├── components/            # Page sections (Hero, Overview, etc.)
    ├── i18n/en.json           # UI text (client-side i18n scaffold)
    ├── pages/
    │   ├── index.astro        # Home (hero + overview)
    │   ├── data.astro         # Data (operators + chart + roofline)
    │   └── doc/
    │       ├── overview.astro  # Evaluation pipeline overview
    │       ├── quickstart.astro
    │       └── format.astro    # Data format spec
    └── styles/global.css      # Tailwind layers + design tokens
```

## Data Pipeline

`scripts/extract-data.py` reads from the repo's `data/` directory and `configs/hardware/`
to generate `src/data/operators.json`. This file includes:

- 30 operator metadata (id, dtype, framework, importance, phase)
- Per-operator roofline summaries (AI, regime, SOL times per hardware)
- Per-shape roofline bounds for XPU-A
- Production kernel performance baselines (XPU-A)
- Hardware specs (P_peak, B_peak)

Run it whenever `data/` or `configs/hardware/` changes.

## Internationalization

The layout ships a small client-side i18n scaffold: `data-i18n` attributes are
swapped from `i18n/<lang>.json` on toggle. Only `en.json` is currently shipped.
To add a language, create `i18n/<lang>.json` and register it in `src/layouts/Layout.astro`.
