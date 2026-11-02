export type LungViewMode = "anatomy" | "scan" | "gradcam";

const clamp01 = (v: number): number => Math.min(1, Math.max(0, v));

/** Normalises a raw scan value the same way the 2D radiograph panes do. */
export function normalizeScanValue(val: number): number {
  return clamp01(val < 0 ? val * 0.5 + 0.5 : val);
}

export function heatColor(h: number, colormap: string): [number, number, number] {
  if (colormap === "Hot") {
    return [
      Math.min(255, h * 3 * 255),
      Math.max(0, Math.min(255, (h - 0.33) * 3 * 255)),
      Math.max(0, Math.min(255, (h - 0.66) * 3 * 255)),
    ];
  }
  return [
    Math.min(255, Math.max(0, 1.5 - Math.abs(h * 4 - 3)) * 255),
    Math.min(255, Math.max(0, 1.5 - Math.abs(h * 4 - 2)) * 255),
    Math.min(255, Math.max(0, 1.5 - Math.abs(h * 4 - 1)) * 255),
  ];
}

/**
 * Composes the texture draped over the 3D lung from the genuine scan and Grad-CAM arrays.
 * Heat is blended per pixel in proportion to saliency, so anatomy stays visible where the
 * model paid no attention. Nothing is synthesised: absent inputs yield null.
 */
export function composeScanPixels(
  raw: number[][],
  heat: number[][] | null,
  mode: LungViewMode,
  opacity: number,
  colormap: string
): { size: number; data: Uint8ClampedArray } | null {
  const size = raw.length;
  if (size === 0 || mode === "anatomy") return null;
  const data = new Uint8ClampedArray(size * size * 4);
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const base = normalizeScanValue(raw[r][c]) * 255;
      let rr = base;
      let gg = base;
      let bb = base;
      if (mode === "gradcam" && heat && heat[r] && heat[r][c] !== undefined) {
        const h = clamp01(heat[r][c]);
        const [hr, hg, hb] = heatColor(h, colormap);
        const a = opacity * Math.min(1, Math.max(0, (h - 0.12) / 0.5));
        rr = (1 - a) * base + a * hr;
        gg = (1 - a) * base + a * hg;
        bb = (1 - a) * base + a * hb;
      }
      const i = (r * size + c) * 4;
      data[i] = rr;
      data[i + 1] = gg;
      data[i + 2] = bb;
      data[i + 3] = 255;
    }
  }
  return { size, data };
}

/** Location (normalised 0..1, origin top-left) of the strongest genuine Grad-CAM activation. */
export function findSaliencyPeak(heat: number[][] | null): { u: number; v: number; value: number } | null {
  if (!heat || heat.length === 0) return null;
  let best = -Infinity;
  let br = 0;
  let bc = 0;
  for (let r = 0; r < heat.length; r++) {
    for (let c = 0; c < heat[r].length; c++) {
      if (heat[r][c] > best) {
        best = heat[r][c];
        br = r;
        bc = c;
      }
    }
  }
  if (!(best > 0)) return null;
  return { u: (bc + 0.5) / heat[0].length, v: (br + 0.5) / heat.length, value: best };
}

