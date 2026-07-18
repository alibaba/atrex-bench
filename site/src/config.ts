function json<T>(raw: string | undefined, fallback: T): T {
  if (!raw) return fallback;
  try { return JSON.parse(raw) as T; } catch { return fallback; }
}

export const siteConfig = {
  githubRepo:   import.meta.env.GITHUB_REPO   || 'https://github.com/alibaba/atrex-bench',
  dockerImage:  import.meta.env.DOCKER_IMAGE   || '<YOUR_REGISTRY>/atrex-bench:rocm7.2',
  dataDir:      import.meta.env.DATA_DIR       || 'data',
  defaultHw:    import.meta.env.DEFAULT_HW     || 'XPU-A',
  prodSku:      import.meta.env.PROD_SKU       || 'XPU-A',
  prodHardware: json<string[]>(import.meta.env.PROD_HARDWARE, ['XPU-A', 'H20']),
  hwShort:      json<Record<string, string>>(import.meta.env.HW_SHORT, {
    'XPU-A': 'XPU-A',
    'H20':   'H20',
  }),
};
