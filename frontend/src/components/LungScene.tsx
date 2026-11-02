"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, Html, Environment } from "@react-three/drei";
import {
  BOTH_LUNGS_WINDOW,
  buildBronchialTree,
  buildLungGeometry,
  lungCenterX,
  projectToFrontSurface,
} from "@/lib/lungGeometry";
import { composeScanPixels, findSaliencyPeak, LungViewMode } from "@/lib/scanTexture";

export interface LungSceneProps {
  containerRef?: React.RefObject<HTMLDivElement>;
  rawImage?: number[][] | null;
  heatmap?: number[][] | null;
  mode: LungViewMode;
  opacity: number;
  colormap: string;
  sway: boolean;
  labels: boolean;
  peakLabel: string;
  rightLabel: string;
  leftLabel: string;
  tracheaLabel: string;
}

function useScanTexture(
  raw: number[][] | null | undefined,
  heat: number[][] | null | undefined,
  mode: LungViewMode,
  opacity: number,
  colormap: string
): THREE.CanvasTexture | null {
  const texture = useMemo(() => {
    if (!raw) return null;
    const composed = composeScanPixels(raw, heat ?? null, mode, opacity, colormap);
    if (!composed) return null;
    const canvas = document.createElement("canvas");
    canvas.width = composed.size;
    canvas.height = composed.size;
    const ctx = canvas.getContext("2d");
    if (!ctx) return null;
    const image = ctx.createImageData(composed.size, composed.size);
    image.data.set(composed.data);
    ctx.putImageData(image, 0, 0);
    const tex = new THREE.CanvasTexture(canvas);
    tex.colorSpace = THREE.SRGBColorSpace;
    tex.anisotropy = 8;
    tex.needsUpdate = true;
    return tex;
  }, [raw, heat, mode, opacity, colormap]);

  useEffect(() => () => texture?.dispose(), [texture]);
  return texture;
}

function Lung({
  side,
  texture,
  meshRef,
}: {
  side: "right" | "left";
  texture: THREE.Texture | null;
  meshRef: (m: THREE.Mesh | null) => void;
}) {
  const geometry = useMemo(() => buildLungGeometry(side, BOTH_LUNGS_WINDOW), [side]);
  useEffect(() => () => geometry.dispose(), [geometry]);
  return (
    <mesh ref={meshRef} geometry={geometry} castShadow receiveShadow>
      {texture ? (
        <meshPhysicalMaterial
          map={texture}
          emissiveMap={texture}
          emissive="#ffffff"
          emissiveIntensity={0.45}
          roughness={0.38}
          metalness={0.0}
          clearcoat={0.6}
          clearcoatRoughness={0.2}
          transmission={0.05}
          thickness={0.4}
        />
      ) : (
        <meshPhysicalMaterial
          color="#d4707e"
          roughness={0.28}
          metalness={0.0}
          sheen={1.2}
          sheenColor={new THREE.Color("#ffb8c2")}
          sheenRoughness={0.3}
          clearcoat={0.8}
          clearcoatRoughness={0.15}
          emissive="#8b1a2e"
          emissiveIntensity={0.18}
          transmission={0.06}
          thickness={0.5}
        />
      )}
    </mesh>
  );
}

/* Anatomical floor reference - glowing diaphragm ring */
function FloorDisc() {
  return (
    <group position={[0, -1.35, 0]}>
      {/* Outer soft glow ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.85, 2.0, 64]} />
        <meshBasicMaterial color="#0d9488" transparent opacity={0.12} side={THREE.DoubleSide} />
      </mesh>
      {/* Inner brighter ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <ringGeometry args={[0.9, 1.25, 64]} />
        <meshBasicMaterial color="#14b8a6" transparent opacity={0.22} side={THREE.DoubleSide} />
      </mesh>
      {/* Subtle filled disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]}>
        <circleGeometry args={[1.1, 64]} />
        <meshBasicMaterial color="#0d9488" transparent opacity={0.10} side={THREE.DoubleSide} />
      </mesh>
    </group>
  );
}

function BronchialTree() {
  const segments = useMemo(() => buildBronchialTree(), []);
  return (
    <group>
      {segments.map((s, i) => {
        const dir = new THREE.Vector3().subVectors(s.end, s.start);
        const length = dir.length();
        const mid = new THREE.Vector3().addVectors(s.start, s.end).multiplyScalar(0.5);
        const quat = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), dir.clone().normalize());
        return (
          <mesh key={i} position={mid} quaternion={quat}>
            <cylinderGeometry args={[s.radius * 0.8, s.radius, length, 14]} />
            <meshStandardMaterial color="#b9e6f2" roughness={0.35} emissive="#2a7a90" emissiveIntensity={0.45} />
          </mesh>
        );
      })}
    </group>
  );
}

function PeakMarker({
  position,
  label,
  containerRef,
}: {
  position: THREE.Vector3;
  label: string;
  containerRef?: React.RefObject<HTMLDivElement>;
}) {
  const ring = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ring.current) return;
    const s = 1 + 0.35 * Math.sin(clock.elapsedTime * 3);
    ring.current.scale.set(s, s, s);
  });
  return (
    <group position={position}>
      <mesh>
        <sphereGeometry args={[0.035, 16, 16]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      <mesh ref={ring}>
        <torusGeometry args={[0.1, 0.008, 12, 48]} />
        <meshBasicMaterial color="#fde047" transparent opacity={0.9} />
      </mesh>
      <Html
        portal={containerRef as unknown as React.MutableRefObject<HTMLElement>}
        zIndexRange={[10, 0]}
        position={[0.18, 0.12, 0.05]}
        style={{ pointerEvents: "none" }}
      >
        <div className="whitespace-nowrap rounded-md border border-amber-300/40 bg-black/70 px-2 py-1 font-mono text-[10px] font-semibold text-amber-200 backdrop-blur">
          {label}
        </div>
      </Html>
    </group>
  );
}

function SceneContents(props: LungSceneProps) {
  const group = useRef<THREE.Group>(null);
  const interacting = useRef(false);
  const meshes = useRef<Record<string, THREE.Mesh | null>>({ right: null, left: null });
  const [peakPos, setPeakPos] = useState<THREE.Vector3 | null>(null);
  const texture = useScanTexture(props.rawImage, props.heatmap, props.mode, props.opacity, props.colormap);

  useFrame(({ clock }) => {
    if (group.current && props.sway && !interacting.current) {
      group.current.rotation.y = Math.sin(clock.elapsedTime * 0.38) * 0.35;
      group.current.rotation.x = Math.sin(clock.elapsedTime * 0.26) * 0.04;
    }
  });

  const peak = useMemo(() => (props.mode === "gradcam" ? findSaliencyPeak(props.heatmap ?? null) : null), [props.mode, props.heatmap]);

  useEffect(() => {
    if (!peak || !group.current) {
      setPeakPos(null);
      return;
    }
    const g = group.current;
    g.updateMatrixWorld(true);
    const inv = new THREE.Matrix4().copy(g.matrixWorld).invert();
    const w = BOTH_LUNGS_WINDOW;
    const vTex = 1 - peak.v;
    const x = w.minX + ((peak.u - 0.1) / 0.8) * (w.maxX - w.minX);
    const y = w.minY + ((vTex - 0.1) / 0.8) * (w.maxY - w.minY);
    const origin = new THREE.Vector3(x, y, 6).applyMatrix4(g.matrixWorld);
    const dir = new THREE.Vector3(0, 0, -1).transformDirection(g.matrixWorld);
    const ray = new THREE.Raycaster(origin, dir);
    const targets = Object.values(meshes.current).filter((m): m is THREE.Mesh => !!m);
    const hits = ray.intersectObjects(targets, false);
    if (hits.length > 0) {
      setPeakPos(hits[0].point.clone().applyMatrix4(inv).add(new THREE.Vector3(0, 0, 0.02)));
    } else {
      const fallback = projectToFrontSurface(targets, x, y);
      setPeakPos(fallback ? fallback.applyMatrix4(inv) : null);
    }
  }, [peak]);

  return (
    <>
      {/* Rich multi-source lighting for convincing anatomy */}
      <ambientLight intensity={0.7} />
      <hemisphereLight args={["#c8e8ff", "#1a0a12", 0.9]} />
      {/* Key light: warm top-front */}
      <directionalLight position={[2.5, 5, 6]} intensity={3.2} color="#fff3e8" castShadow
        shadow-mapSize={[1024, 1024]} shadow-camera-near={0.5} shadow-camera-far={20} />
      {/* Fill light: cool left */}
      <directionalLight position={[-5, 2, 3]} intensity={1.4} color="#a5d8ff" />
      {/* Rim light: back right for depth */}
      <directionalLight position={[4, -1, -4]} intensity={0.8} color="#ff9eaf" />
      {/* Bottom bounce: teal glow from below */}
      <pointLight position={[0, -3, 1.5]} intensity={28} color="#0d9488" distance={6} decay={2} />
      {/* Subtle inner glow */}
      <pointLight position={[0, 0, 2]} intensity={8} color="#fda4af" distance={4} decay={2} />

      {/* Main lung group - centered and proportioned with comfortable margin */}
      <group ref={group} position={[0, -0.06, 0]} scale={0.82}>
        <Lung side="right" texture={texture} meshRef={(m) => (meshes.current.right = m)} />
        <Lung side="left" texture={texture} meshRef={(m) => (meshes.current.left = m)} />
        <BronchialTree />
        {peakPos && <PeakMarker position={peakPos} label={props.peakLabel} containerRef={props.containerRef} />}
        {props.labels && (
          <>
            {/* Labels placed cleanly to the far lateral sides of each lung */}
            <Html
              portal={props.containerRef as unknown as React.MutableRefObject<HTMLElement>}
              zIndexRange={[10, 0]}
              position={[lungCenterX("right") - 0.95, 0.10, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <span className="whitespace-nowrap rounded border border-emerald-400/30 bg-slate-900/90 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-300 backdrop-blur shadow-lg">
                {props.rightLabel}
              </span>
            </Html>
            <Html
              portal={props.containerRef as unknown as React.MutableRefObject<HTMLElement>}
              zIndexRange={[10, 0]}
              position={[lungCenterX("left") + 0.95, 0.10, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <span className="whitespace-nowrap rounded border border-emerald-400/30 bg-slate-900/90 px-2 py-0.5 font-mono text-[10px] font-bold text-emerald-300 backdrop-blur shadow-lg">
                {props.leftLabel}
              </span>
            </Html>
            <Html
              portal={props.containerRef as unknown as React.MutableRefObject<HTMLElement>}
              zIndexRange={[10, 0]}
              position={[0.0, 1.55, 0]}
              center
              style={{ pointerEvents: "none" }}
            >
              <span className="whitespace-nowrap rounded border border-sky-400/30 bg-slate-900/90 px-2 py-0.5 font-mono text-[9px] text-sky-300 backdrop-blur shadow-lg">
                {props.tracheaLabel}
              </span>
            </Html>
          </>
        )}
      </group>
      <FloorDisc />
      <OrbitControls
        enablePan={false}
        enableDamping
        dampingFactor={0.07}
        target={[0, -0.05, 0]}
        minDistance={2.5}
        maxDistance={7.0}
        minPolarAngle={Math.PI * 0.2}
        maxPolarAngle={Math.PI * 0.8}
        onStart={() => { interacting.current = true; }}
        onEnd={() => { interacting.current = false; }}
      />
    </>
  );
}

export default function LungScene(props: LungSceneProps) {
  return (
    <Canvas
      className="w-full h-full"
      dpr={[1, 2]}
      camera={{ position: [0, -0.02, 5.6], fov: 38 }}
      gl={{ antialias: true, alpha: true, preserveDrawingBuffer: false, toneMapping: THREE.ACESFilmicToneMapping }}
      shadows
      onCreated={({ gl, camera, scene }) => {
        gl.toneMappingExposure = 1.2;
        gl.setClearColor(0x000000, 0);
        camera.lookAt(0, -0.05, 0);
        scene.background = null;
        if (gl.domElement) {
          gl.domElement.addEventListener("webglcontextlost", (event: Event) => {
            event.preventDefault();
          }, false);
        }
      }}
      style={{
        width: "100%",
        height: "100%",
        background: "transparent",
      }}
    >
      <SceneContents {...props} />
    </Canvas>
  );
}

