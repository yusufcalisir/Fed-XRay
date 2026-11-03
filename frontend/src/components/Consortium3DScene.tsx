"use client";

import React, { useRef, useMemo, useState, useEffect, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { Html, Line } from "@react-three/drei";

// Detects dark mode via document class mutation observer
function useIsDark(): boolean {
  const [isDark, setIsDark] = useState(false);
  useEffect(() => {
    const check = () =>
      setIsDark(document.documentElement.classList.contains("dark"));
    check();
    const obs = new MutationObserver(check);
    obs.observe(document.documentElement, {
      attributes: true,
      attributeFilter: ["class"],
    });
    return () => obs.disconnect();
  }, []);
  return isDark;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------
interface HospitalNodeData {
  id: string;
  shortName: string;
  location: string;
  modality: string;
  dpEpsilon: number;
  position: [number, number, number];
  themeColor: string;
}

interface Consortium3DSceneProps {
  isSimulatingRound?: boolean;
  selectedNodeId?: string;
  mouseTilt?: { x: number; y: number };
  onSelectNode?: (nodeId: string) => void;
  onSimulationComplete?: () => void;
}

// ---------------------------------------------------------------------------
// Hospital node layout and dimensions
// ---------------------------------------------------------------------------
const NODES: HospitalNodeData[] = [
  {
    id: "node-1",
    shortName: "BCN-20000",
    location: "Barcelona",
    modality: "Dermoscopy",
    dpEpsilon: 1.42,
    position: [-2.8, 0, -1.6],
    themeColor: "#10b981",
  },
  {
    id: "node-3",
    shortName: "ViDIR",
    location: "Vienna",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    position: [2.8, 0, -1.6],
    themeColor: "#06b6d4",
  },
  {
    id: "node-4",
    shortName: "Queensland",
    location: "Brisbane",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    position: [-2.8, 0, 1.6],
    themeColor: "#10b981",
  },
  {
    id: "node-2",
    shortName: "Beth Israel",
    location: "Boston",
    modality: "Chest X-Ray",
    dpEpsilon: 1.38,
    position: [2.8, 0, 1.6],
    themeColor: "#38bdf8",
  },
];

// Monolith body
const MW = 0.85;
const MH = 1.3;
const MD = 0.65;

// Server pedestal and tower
const SBW = 1.1;
const SBH = 0.4;
const SBD = 0.9;
const STW = 0.7;
const STH = 1.8;
const STD = 0.6;
const CRYPTO_Y = 3.0;

// ---------------------------------------------------------------------------
// HospitalMonolith: 3D building with medical cross, beacon, and label
// ---------------------------------------------------------------------------
function HospitalMonolith({
  node,
  isSelected,
  isDark,
  onSelect,
}: {
  node: HospitalNodeData;
  isSelected: boolean;
  isDark: boolean;
  onSelect: (id: string) => void;
}) {
  const floatRef = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    if (!floatRef.current) return;
    const t = clock.elapsedTime;
    const lev = isSelected
      ? Math.sin(t * 3) * 0.12 + 0.35
      : hovered
        ? 0.18
        : Math.sin(t * 2 + node.position[0] * 1.5) * 0.06;
    floatRef.current.position.y = MH / 2 + lev;
  });

  const bodyColor = isDark ? "#1a2538" : "#f1f5f9";
  const emit = isSelected ? 0.28 : hovered ? 0.14 : 0.04;

  return (
    <group position={[node.position[0], 0, node.position[2]]}>
      {/* Ground shadow disc */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <circleGeometry args={[0.7, 32]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.14 : 0.06}
        />
      </mesh>

      {/* Targeting ring */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.012, 0]}>
        <ringGeometry args={[0.6, 0.66, 48]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.55 : 0.18}
        />
      </mesh>

      {/* Floating building group */}
      <group ref={floatRef}>
        {/* Main body */}
        <mesh
          onClick={(e) => {
            e.stopPropagation();
            onSelect(node.id);
          }}
          onPointerOver={(e) => {
            e.stopPropagation();
            setHovered(true);
            document.body.style.cursor = "pointer";
          }}
          onPointerOut={() => {
            setHovered(false);
            document.body.style.cursor = "auto";
          }}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[MW, MH, MD]} />
          <meshPhysicalMaterial
            color={bodyColor}
            metalness={isDark ? 0.35 : 0.05}
            roughness={isDark ? 0.62 : 0.82}
            emissive={node.themeColor}
            emissiveIntensity={emit}
            clearcoat={0.18}
          />
        </mesh>

        {/* Medical cross: vertical bar on front face */}
        <mesh position={[0, 0.08, MD / 2 + 0.006]}>
          <planeGeometry args={[0.055, 0.2]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.75}
            side={THREE.DoubleSide}
          />
        </mesh>
        {/* Medical cross: horizontal bar on front face */}
        <mesh position={[0, 0.08, MD / 2 + 0.006]}>
          <planeGeometry args={[0.2, 0.055]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.75}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Side illumination bands */}
        <mesh
          position={[MW / 2 + 0.004, -0.12, 0]}
          rotation={[0, Math.PI / 2, 0]}
        >
          <planeGeometry args={[MD * 0.72, 0.035]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>
        <mesh
          position={[MW / 2 + 0.004, 0.28, 0]}
          rotation={[0, Math.PI / 2, 0]}
        >
          <planeGeometry args={[MD * 0.72, 0.035]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.4}
            side={THREE.DoubleSide}
          />
        </mesh>

        {/* Beacon tether */}
        <mesh position={[0, MH / 2 + 0.12, 0]}>
          <cylinderGeometry args={[0.004, 0.004, 0.24, 4]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.4}
          />
        </mesh>

        {/* Rooftop beacon sphere */}
        <mesh position={[0, MH / 2 + 0.26, 0]}>
          <sphereGeometry args={[0.06, 12, 12]} />
          <meshBasicMaterial color={node.themeColor} />
        </mesh>
        <pointLight
          position={[0, MH / 2 + 0.26, 0]}
          color={node.themeColor}
          intensity={isSelected ? 4 : 1.2}
          distance={2.5}
          decay={2}
        />
      </group>

      {/* Static label (does not float with the building) */}
      <Html
        position={[0, -0.08, 0.55]}
        center
        zIndexRange={[5, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div
          style={{
            textAlign: "center",
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: "nowrap",
          }}
        >
          <div
            style={{
              fontSize: "10px",
              fontWeight: 700,
              color: isDark ? "#e2e8f0" : "#1e293b",
              letterSpacing: "0.5px",
              textShadow: isDark
                ? "0 1px 10px rgba(0,0,0,0.95)"
                : "0 1px 10px rgba(255,255,255,0.98)",
            }}
          >
            {node.shortName}
          </div>
          <div
            style={{
              fontSize: "7.5px",
              color: isDark ? "#94a3b8" : "#64748b",
              marginTop: "2px",
              fontFamily: "'Inter', sans-serif",
              textShadow: isDark
                ? "0 1px 8px rgba(0,0,0,0.9)"
                : "0 1px 8px rgba(255,255,255,0.95)",
            }}
          >
            {node.location} : {node.modality}
          </div>
          <div
            style={{
              fontSize: "8px",
              color: "#10b981",
              marginTop: "1px",
              fontFamily: "'JetBrains Mono', monospace",
              textShadow: isDark
                ? "0 1px 8px rgba(0,0,0,0.9)"
                : "0 1px 8px rgba(255,255,255,0.95)",
            }}
          >
            ε={node.dpEpsilon}
          </div>
        </div>
      </Html>
    </group>
  );
}

// ---------------------------------------------------------------------------
// LEDStrip: Blinking indicator row on the server tower face
// ---------------------------------------------------------------------------
function LEDStrip({
  position,
  color,
  delay,
}: {
  position: [number, number, number];
  color: string;
  delay: number;
}) {
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const on = Math.sin(clock.elapsedTime * 4 + delay) > 0;
    (ref.current.material as THREE.MeshBasicMaterial).opacity = on
      ? 0.85
      : 0.12;
  });
  return (
    <mesh ref={ref} position={position} rotation={[0, Math.PI / 2, 0]}>
      <planeGeometry args={[STD * 0.55, 0.03]} />
      <meshBasicMaterial
        color={color}
        transparent
        opacity={0.85}
        side={THREE.DoubleSide}
      />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// CentralServer: Aggregator pedestal, tower, LED rows, and floating
// cryptographic core with orbit rings
// ---------------------------------------------------------------------------
function CentralServer({ isDark }: { isDark: boolean }) {
  const baseRef = useRef<THREE.Group>(null);
  const cryptoRef = useRef<THREE.Group>(null);
  const ring1Ref = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;
    if (baseRef.current) {
      baseRef.current.position.y = Math.sin(t * 2.4) * 0.05;
    }
    if (cryptoRef.current) {
      cryptoRef.current.position.y =
        CRYPTO_Y + Math.sin(t * 3.5) * 0.08;
    }
    if (ring1Ref.current) {
      ring1Ref.current.rotation.x = Math.PI / 3.5;
      ring1Ref.current.rotation.y = t * 0.7;
    }
    if (ring2Ref.current) {
      ring2Ref.current.rotation.x = Math.PI / 2.5;
      ring2Ref.current.rotation.z = -t * 1.0;
    }
    if (coreRef.current) {
      const s = 1 + Math.sin(t * 3.5) * 0.15;
      coreRef.current.scale.set(s, s, s);
    }
  });

  const towerColor = isDark ? "#0f172a" : "#e2e8f0";

  return (
    <group>
      {/* Ground energy pool */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.006, 0]}>
        <circleGeometry args={[1.3, 64]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={isDark ? 0.06 : 0.04}
        />
      </mesh>

      {/* Server body (pedestal + tower) with subtle levitation */}
      <group ref={baseRef}>
        {/* Base pedestal */}
        <mesh position={[0, SBH / 2, 0]} castShadow receiveShadow>
          <boxGeometry args={[SBW, SBH, SBD]} />
          <meshPhysicalMaterial
            color={towerColor}
            metalness={0.45}
            roughness={0.55}
            emissive="#10b981"
            emissiveIntensity={0.04}
          />
        </mesh>

        {/* Main tower */}
        <mesh
          position={[0, SBH + STH / 2, 0]}
          castShadow
          receiveShadow
        >
          <boxGeometry args={[STW, STH, STD]} />
          <meshPhysicalMaterial
            color={towerColor}
            metalness={0.5}
            roughness={0.42}
            emissive="#10b981"
            emissiveIntensity={0.06}
            clearcoat={0.25}
          />
        </mesh>

        {/* LED indicator strips on tower face */}
        {[0.35, 0.75, 1.15].map((yOff, i) => (
          <LEDStrip
            key={i}
            position={[STW / 2 + 0.005, SBH + yOff, 0]}
            color={i % 2 === 0 ? "#10b981" : "#06b6d4"}
            delay={i * 1.6}
          />
        ))}
      </group>

      {/* Floating cryptographic core assembly */}
      <group ref={cryptoRef} position={[0, CRYPTO_Y, 0]}>
        {/* Solid core sphere */}
        <mesh ref={coreRef}>
          <sphereGeometry args={[0.2, 24, 24]} />
          <meshBasicMaterial color="#34d399" transparent opacity={0.75} />
        </mesh>

        {/* Outer glow shell */}
        <mesh>
          <sphereGeometry args={[0.35, 16, 16]} />
          <meshBasicMaterial
            color="#10b981"
            transparent
            opacity={0.1}
          />
        </mesh>

        {/* Core point light */}
        <pointLight
          color="#10b981"
          intensity={6}
          distance={5}
          decay={2}
        />

        {/* Orbit ring 1 (emerald) */}
        <mesh ref={ring1Ref}>
          <torusGeometry args={[0.55, 0.018, 8, 64]} />
          <meshBasicMaterial
            color="#10b981"
            transparent
            opacity={0.55}
          />
        </mesh>

        {/* Orbit ring 2 (cyan) */}
        <mesh ref={ring2Ref}>
          <torusGeometry args={[0.4, 0.014, 8, 64]} />
          <meshBasicMaterial
            color="#06b6d4"
            transparent
            opacity={0.45}
          />
        </mesh>
      </group>

      {/* Server identification label */}
      <Html
        position={[0, SBH + STH + 0.15, 0.5]}
        center
        zIndexRange={[5, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div style={{ textAlign: "center", whiteSpace: "nowrap" }}>
          <div
            style={{
              fontSize: "10px",
              fontWeight: 700,
              color: "#e2e8f0",
              fontFamily: "'JetBrains Mono', monospace",
              letterSpacing: "0.8px",
              textShadow: "0 1px 12px rgba(0,0,0,0.95)",
            }}
          >
            CONSORTIUM AGGREGATOR
          </div>
          <div
            style={{
              fontSize: "7.5px",
              color: "#94a3b8",
              fontFamily: "'JetBrains Mono', monospace",
              marginTop: "2px",
              textShadow: "0 1px 8px rgba(0,0,0,0.9)",
            }}
          >
            RLWE CKKS 128-Bit : Zero Plaintext
          </div>
        </div>
      </Html>
    </group>
  );
}

// ---------------------------------------------------------------------------
// PhotonPacket: Animated glowing sphere traveling along a bezier conduit
// ---------------------------------------------------------------------------
function PhotonPacket({
  curve,
  offset,
  color,
  speedMul,
}: {
  curve: THREE.QuadraticBezierCurve3;
  offset: number;
  color: string;
  speedMul: number;
}) {
  const dotRef = useRef<THREE.Mesh>(null);
  const glowRef = useRef<THREE.Mesh>(null);
  const progress = useRef(offset);

  useFrame((_, delta) => {
    progress.current = (progress.current + delta * 0.18 * speedMul) % 1;
    const p = curve.getPoint(progress.current);
    if (dotRef.current) dotRef.current.position.copy(p);
    if (glowRef.current) glowRef.current.position.copy(p);
  });

  return (
    <>
      {/* Bright center dot */}
      <mesh ref={dotRef}>
        <sphereGeometry args={[0.04, 8, 8]} />
        <meshBasicMaterial color="#ffffff" />
      </mesh>
      {/* Colored glow shell */}
      <mesh ref={glowRef}>
        <sphereGeometry args={[0.1, 8, 8]} />
        <meshBasicMaterial color={color} transparent opacity={0.3} />
      </mesh>
    </>
  );
}

// ---------------------------------------------------------------------------
// DataConduit: Bezier curve line with 3 animated photon packets flowing
// from a hospital node to the central cryptographic core
// ---------------------------------------------------------------------------
function DataConduit({
  nodePos,
  color,
  isSimulating,
  isSelected,
  idx,
}: {
  nodePos: [number, number, number];
  color: string;
  isSimulating: boolean;
  isSelected: boolean;
  idx: number;
}) {
  const startY = MH + 0.3;
  const endY = CRYPTO_Y;

  const { curve, linePoints } = useMemo(() => {
    const s = new THREE.Vector3(nodePos[0], startY, nodePos[2]);
    const e = new THREE.Vector3(0, endY, 0);
    const mx = (s.x + e.x) / 2 + (idx % 2 === 0 ? -0.4 : 0.4);
    const my = Math.max(s.y, e.y) + 1.2;
    const mz = (s.z + e.z) / 2;
    const m = new THREE.Vector3(mx, my, mz);
    const c = new THREE.QuadraticBezierCurve3(s, m, e);
    const pts = c
      .getPoints(48)
      .map((p) => [p.x, p.y, p.z] as [number, number, number]);
    return { curve: c, linePoints: pts };
  }, [nodePos, startY, endY, idx]);

  const speedMul = isSimulating ? 3.2 : isSelected ? 1.5 : 1.0;

  return (
    <group>
      {/* Static conduit path */}
      <Line
        points={linePoints}
        color={color}
        lineWidth={1.2}
        transparent
        opacity={0.28}
      />

      {/* Three photon packets per conduit */}
      {[0, 0.33, 0.66].map((off, i) => (
        <PhotonPacket
          key={i}
          curve={curve}
          offset={off}
          color={color}
          speedMul={speedMul}
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// ConsensusShockwave: Expanding dual-ring burst when an encrypted round
// is triggered, using scale-based animation (no geometry recreation)
// ---------------------------------------------------------------------------
function ConsensusShockwave({
  active,
  onComplete,
}: {
  active: boolean;
  onComplete: () => void;
}) {
  const outerRef = useRef<THREE.Mesh>(null);
  const innerRef = useRef<THREE.Mesh>(null);
  const progressRef = useRef(0);
  const wasActiveRef = useRef(false);
  const runningRef = useRef(false);

  useEffect(() => {
    if (active && !wasActiveRef.current) {
      runningRef.current = true;
      progressRef.current = 0;
      if (outerRef.current) outerRef.current.visible = true;
      if (innerRef.current) innerRef.current.visible = true;
    }
    wasActiveRef.current = active;
  }, [active]);

  useFrame((_, delta) => {
    if (!runningRef.current) return;
    progressRef.current += delta * 1.1;

    if (progressRef.current >= 1) {
      runningRef.current = false;
      if (outerRef.current) outerRef.current.visible = false;
      if (innerRef.current) innerRef.current.visible = false;
      onComplete();
      return;
    }

    const p = progressRef.current;
    const scale = 1 + p * 14;
    const alpha = (1 - p) * 0.5;

    if (outerRef.current) {
      outerRef.current.scale.set(scale, scale, 1);
      (outerRef.current.material as THREE.MeshBasicMaterial).opacity =
        alpha;
    }
    if (innerRef.current && p > 0.08) {
      const s2 = 1 + (p - 0.08) * 14;
      innerRef.current.scale.set(s2, s2, 1);
      (innerRef.current.material as THREE.MeshBasicMaterial).opacity =
        alpha * 0.65;
    }
  });

  return (
    <>
      <mesh
        ref={outerRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.25, 0]}
        visible={false}
      >
        <ringGeometry args={[0.35, 0.4, 64]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={0}
          side={THREE.DoubleSide}
        />
      </mesh>
      <mesh
        ref={innerRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.25, 0]}
        visible={false}
      >
        <ringGeometry args={[0.28, 0.33, 64]} />
        <meshBasicMaterial
          color="#06b6d4"
          transparent
          opacity={0}
          side={THREE.DoubleSide}
        />
      </mesh>
    </>
  );
}

// ---------------------------------------------------------------------------
// SceneContents: Composes all 3D elements within the Canvas render tree.
// All R3F hooks (useFrame, useThree) live here inside the Canvas context.
// ---------------------------------------------------------------------------
function SceneContents({
  isSimulatingRound = false,
  selectedNodeId = "node-1",
  mouseTilt = { x: 0, y: 0 },
  onSelectNode,
  onSimulationComplete,
}: Consortium3DSceneProps) {
  const sceneGroupRef = useRef<THREE.Group>(null);
  const isDark = useIsDark();
  const tiltRef = useRef({ x: 0, y: 0 });

  // Smooth damped tilt interpolation driven by parent mouse position
  useFrame(() => {
    tiltRef.current.x +=
      (mouseTilt.x - tiltRef.current.x) * 0.06;
    tiltRef.current.y +=
      (mouseTilt.y - tiltRef.current.y) * 0.06;
    if (sceneGroupRef.current) {
      sceneGroupRef.current.rotation.y = tiltRef.current.x * 0.18;
      sceneGroupRef.current.rotation.x = tiltRef.current.y * 0.1;
    }
  });

  const handleSelect = useCallback(
    (id: string) => onSelectNode?.(id),
    [onSelectNode],
  );

  const handleSimDone = useCallback(
    () => onSimulationComplete?.(),
    [onSimulationComplete],
  );

  return (
    <>
      {/* Lighting rig */}
      <ambientLight intensity={isDark ? 0.25 : 0.55} />
      <directionalLight
        position={[5, 9, 6]}
        intensity={isDark ? 1.8 : 2.8}
        color="#ffffff"
      />
      <directionalLight
        position={[-4, 5, -4]}
        intensity={isDark ? 0.5 : 0.7}
        color="#a5d8ff"
      />
      <pointLight
        position={[0, 5, 0]}
        intensity={isDark ? 3 : 1.5}
        color="#10b981"
        distance={12}
        decay={2}
      />

      {/* Mouse tilt group wraps the entire scene */}
      <group ref={sceneGroupRef}>
        {/* Transparent ground disc for spatial reference */}
        <mesh
          rotation={[-Math.PI / 2, 0, 0]}
          position={[0, -0.005, 0]}
        >
          <circleGeometry args={[6.5, 64]} />
          <meshStandardMaterial
            color={isDark ? "#0c1222" : "#f0f4f8"}
            transparent
            opacity={isDark ? 0.3 : 0.2}
            metalness={0.05}
            roughness={0.95}
          />
        </mesh>

        {/* Concentric depth rings */}
        {[1.5, 3.0, 4.5].map((r, i) => (
          <mesh
            key={`ring-${i}`}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, 0.002, 0]}
          >
            <ringGeometry args={[r - 0.012, r, 64]} />
            <meshBasicMaterial
              color={isDark ? "#1e293b" : "#cbd5e1"}
              transparent
              opacity={0.12}
            />
          </mesh>
        ))}

        {/* Hospital monoliths */}
        {NODES.map((n) => (
          <HospitalMonolith
            key={n.id}
            node={n}
            isSelected={selectedNodeId === n.id}
            isDark={isDark}
            onSelect={handleSelect}
          />
        ))}

        {/* Central aggregator server */}
        <CentralServer isDark={isDark} />

        {/* Encrypted data conduits from each hospital to the server core */}
        {NODES.map((n, i) => (
          <DataConduit
            key={n.id}
            nodePos={n.position}
            color={n.themeColor}
            isSimulating={isSimulatingRound}
            isSelected={selectedNodeId === n.id}
            idx={i}
          />
        ))}

        {/* Consensus shockwave burst */}
        <ConsensusShockwave
          active={isSimulatingRound}
          onComplete={handleSimDone}
        />
      </group>
    </>
  );
}

// ---------------------------------------------------------------------------
// Consortium3DScene: Top-level Canvas wrapper (default export).
// Configures WebGL renderer with transparent alpha, ACES filmic tone
// mapping, and an elevated perspective camera aimed at the network center.
// ---------------------------------------------------------------------------
export default function Consortium3DScene(props: Consortium3DSceneProps) {
  return (
    <Canvas
      className="w-full h-full"
      dpr={[1, 2]}
      camera={{ position: [0, 6, 7.5], fov: 38 }}
      gl={{
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: false,
        toneMapping: THREE.ACESFilmicToneMapping,
      }}
      onCreated={({ gl, camera, scene }) => {
        gl.toneMappingExposure = 1.15;
        gl.setClearColor(0x000000, 0);
        camera.lookAt(0, 0.5, 0);
        scene.background = null;
        gl.domElement.addEventListener(
          "webglcontextlost",
          (e) => e.preventDefault(),
          false,
        );
      }}
      style={{ width: "100%", height: "100%", background: "transparent" }}
    >
      <SceneContents {...props} />
    </Canvas>
  );
}
