import { describe, it, expect } from "vitest";
import * as THREE from "three";
import {
  BOTH_LUNGS_WINDOW,
  buildBronchialTree,
  buildLungGeometry,
  lungCenterX,
} from "@/lib/lungGeometry";
import { composeScanPixels, findSaliencyPeak, normalizeScanValue } from "@/lib/scanTexture";

describe("lungGeometry", () => {
  it("builds finite, bilaterally placed lung surfaces with UVs inside the texture", () => {
    const right = buildLungGeometry("right", BOTH_LUNGS_WINDOW, 48);
    const left = buildLungGeometry("left", BOTH_LUNGS_WINDOW, 48);

    for (const geo of [right, left]) {
      const pos = geo.attributes.position.array as Float32Array;
      const uv = geo.attributes.uv.array as Float32Array;
      expect(pos.every((v) => Number.isFinite(v))).toBe(true);
      expect(uv.every((v) => v >= 0 && v <= 1)).toBe(true);
    }

    right.computeBoundingBox();
    left.computeBoundingBox();
    expect(right.boundingBox!.getCenter(new THREE.Vector3()).x).toBeLessThan(0);
    expect(left.boundingBox!.getCenter(new THREE.Vector3()).x).toBeGreaterThan(0);
    expect(lungCenterX("right")).toBeLessThan(lungCenterX("left"));
  });

  it("keeps the lungs separated by a mediastinal gap wide enough for the trachea", () => {
    const right = buildLungGeometry("right", BOTH_LUNGS_WINDOW, 48);
    const left = buildLungGeometry("left", BOTH_LUNGS_WINDOW, 48);
    right.computeBoundingBox();
    left.computeBoundingBox();
    expect(left.boundingBox!.min.x - right.boundingBox!.max.x).toBeGreaterThan(0.1);
  });

  it("generates a deterministic tracheobronchial tree rooted at the trachea", () => {
    const a = buildBronchialTree(7);
    const b = buildBronchialTree(7);
    expect(a.length).toBe(b.length);
    expect(a.length).toBeGreaterThan(10);
    expect(a[0].radius).toBeGreaterThan(a[a.length - 1].radius);
  });
});

describe("scanTexture", () => {
  const raw = [
    [0, 0.5],
    [-1, 1],
  ];
  const heat = [
    [0, 0.2],
    [0.9, 0.4],
  ];

  it("returns null in anatomy mode and never fabricates pixels", () => {
    expect(composeScanPixels(raw, heat, "anatomy", 0.7, "Hot")).toBeNull();
    expect(composeScanPixels([], null, "scan", 0.7, "Hot")).toBeNull();
  });

  it("renders the genuine grayscale scan and blends heat only where saliency exists", () => {
    const scan = composeScanPixels(raw, heat, "scan", 0.7, "Hot")!;
    expect(scan.data[0]).toBe(0);
    expect(scan.data[4]).toBe(Math.round(normalizeScanValue(0.5) * 255));

    const cam = composeScanPixels(raw, heat, "gradcam", 1, "Hot")!;
    // zero saliency pixel is untouched, high saliency pixel is shifted toward the heat colour
    expect(cam.data[0]).toBe(scan.data[0]);
    const hotIdx = 2 * 4;
    expect(cam.data[hotIdx]).toBeGreaterThan(scan.data[hotIdx]);
  });

  it("locates the strongest activation and returns null for empty maps", () => {
    const peak = findSaliencyPeak(heat)!;
    expect(peak.value).toBeCloseTo(0.9);
    expect(peak.u).toBeCloseTo(0.25);
    expect(peak.v).toBeCloseTo(0.75);
    expect(findSaliencyPeak(null)).toBeNull();
    expect(findSaliencyPeak([[0, 0]])).toBeNull();
  });
});
