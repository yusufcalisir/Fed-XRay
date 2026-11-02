import * as THREE from "three";

export type LungSide = "right" | "left";

/** Lateral offset of each lung from the mediastinal midline (scene units). */
export const LUNG_OFFSET_X = 0.82;

/** Radiographic convention: patient right is drawn on the viewer's left. */
export function lungCenterX(side: LungSide): number {
  return side === "right" ? -LUNG_OFFSET_X : LUNG_OFFSET_X;
}

const smoothstep = (a: number, b: number, x: number): number => {
  const t = Math.min(1, Math.max(0, (x - a) / (b - a)));
  return t * t * (3 - 2 * t);
};

/** Planar projection window used to map a frontal 2D chest radiograph onto both lungs. */
export interface ProjectionWindow {
  minX: number;
  maxX: number;
  minY: number;
  maxY: number;
}

/**
 * Builds one anatomical lung surface (apex, costal and diaphragmatic faces, concave medial wall
 * with hilum, cardiac notch for the left lung, oblique and horizontal fissures).
 * The shape is an anatomical schematic; it is NOT a patient specific CT reconstruction.
 */
export function buildLungGeometry(
  side: LungSide,
  window: ProjectionWindow,
  segments = 96
): THREE.BufferGeometry {
  const geo = new THREE.SphereGeometry(1, segments, segments);
  const pos = geo.attributes.position as THREE.BufferAttribute;
  const uv = geo.attributes.uv as THREE.BufferAttribute;

  const medial = side === "right" ? 1 : -1; // local +x points toward the mediastinum for the right lung
  const sizeX = side === "left" ? 0.66 : 0.74;
  const sizeY = side === "left" ? 1.36 : 1.42;
  const sizeZ = 0.64;
  const cx = lungCenterX(side);

  for (let i = 0; i < pos.count; i++) {
    let x = pos.getX(i);
    let y = pos.getY(i);
    let z = pos.getZ(i);

    const t = (y + 1) / 2; // 0 at the base, 1 at the apex

    // Apex tapers, lower third is the widest part of the lung.
    const taper = 1 - 0.5 * t * t - 0.08 * (1 - t) * (1 - t);
    x *= taper;
    z *= 0.78 + 0.22 * taper;

    // Diaphragmatic surface is concave: lift the central part of the base.
    const radial = Math.min(1, x * x + z * z);
    y += 0.2 * (1 - radial) * (1 - smoothstep(-1, -0.45, y)) * (y < 0 ? 1 : 0);

    // Flatten the medial wall against the mediastinum.
    const towardMedial = x * medial;
    if (towardMedial > 0) {
      x = x * (1 - 0.42 * smoothstep(0, 0.9, towardMedial));
    }

    // Hilum depression on the medial face.
    const hilum = Math.exp(-((y - 0.12) ** 2 / 0.05 + z * z / 0.1));
    if (towardMedial > 0) x -= medial * 0.16 * hilum;

    // Cardiac notch on the anterior inferomedial border of the left lung.
    if (side === "left") {
      const notch = Math.exp(-((y + 0.22) ** 2 / 0.09)) * Math.max(0, z) * smoothstep(-0.1, 0.6, towardMedial);
      x -= 0.42 * notch * medial;
      z -= 0.18 * notch;
    }

    // Fissures: shallow grooves along the oblique (both lungs) and horizontal (right lung) planes.
    const oblique = Math.abs(0.64 * (y - 0.1) - 0.77 * z * 1.2 + 0.0);
    const groove = Math.exp(-(oblique * oblique) / 0.0016) * 0.045;
    let fissure = groove;
    if (side === "right") {
      fissure += Math.exp(-((y - 0.32) ** 2) / 0.0012) * 0.04 * smoothstep(-0.2, 0.5, z);
    }
    const len = Math.hypot(x, y, z) || 1;
    const k = 1 - fissure / len;
    x *= k;
    y *= k;
    z *= k;

    // Final lung proportions and placement.
    const wx = x * sizeX + cx;
    const wy = y * sizeY;
    const wz = z * sizeZ;
    pos.setXYZ(i, wx, wy, wz);

    uv.setXY(
      i,
      0.1 + 0.8 * ((wx - window.minX) / (window.maxX - window.minX)),
      0.1 + 0.8 * ((wy - window.minY) / (window.maxY - window.minY))
    );
  }

  pos.needsUpdate = true;
  uv.needsUpdate = true;
  geo.computeVertexNormals();
  geo.computeBoundingBox();
  return geo;
}

/** Window spanning both lungs, used for consistent planar texture projection. */
export const BOTH_LUNGS_WINDOW: ProjectionWindow = {
  minX: -LUNG_OFFSET_X - 0.78,
  maxX: LUNG_OFFSET_X + 0.78,
  minY: -1.45,
  maxY: 1.45,
};

function mulberry32(seed: number): () => number {
  let a = seed;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface BronchialSegment {
  start: THREE.Vector3;
  end: THREE.Vector3;
  radius: number;
}

/** Deterministic tracheobronchial tree (trachea, main bronchi, lobar and segmental branches). */
export function buildBronchialTree(seed = 7): BronchialSegment[] {
  const rand = mulberry32(seed);
  const segments: BronchialSegment[] = [];

  segments.push({
    start: new THREE.Vector3(0, 1.42, 0.02),
    end: new THREE.Vector3(0, 0.60, 0),
    radius: 0.092,
  });

  const grow = (origin: THREE.Vector3, dir: THREE.Vector3, length: number, radius: number, depth: number) => {
    const end = origin.clone().addScaledVector(dir, length);
    segments.push({ start: origin.clone(), end, radius });
    if (depth <= 0) return;
    const branches = depth > 2 ? 2 : rand() > 0.35 ? 2 : 1;
    for (let b = 0; b < branches; b++) {
      const spread = new THREE.Vector3(
        (rand() - 0.5) * 0.9,
        -0.15 - rand() * 0.75,
        (rand() - 0.5) * 0.8
      );
      const nd = dir.clone().multiplyScalar(0.8).add(spread).normalize();
      grow(end, nd, length * (0.68 + rand() * 0.08), radius * 0.72, depth - 1);
    }
  };

  for (const sign of [-1, 1]) {
    const carina = new THREE.Vector3(0, 0.62, 0);
    const main = new THREE.Vector3(sign * 0.9, -0.38, -0.05).normalize();
    grow(carina, main, 0.62, 0.062, 4);
  }

  return segments;
}

/** Finds the lung surface point hit by a ray travelling from the viewer into the scene at (x, y). */
export function projectToFrontSurface(
  meshes: THREE.Object3D[],
  x: number,
  y: number
): THREE.Vector3 | null {
  const ray = new THREE.Raycaster(new THREE.Vector3(x, y, 6), new THREE.Vector3(0, 0, -1));
  const hits = ray.intersectObjects(meshes, false);
  return hits.length > 0 ? hits[0].point.clone() : null;
}
