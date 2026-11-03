"use client";

import React, { useRef, useMemo, useState, useEffect, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, Edges } from "@react-three/drei";

// Hook to detect dark mode via document observer
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
// Hospital Node Data Types
// ---------------------------------------------------------------------------
interface HospitalNodeData {
  id: string;
  shortName: string;
  location: string;
  modality: string;
  dpEpsilon: number;
  position: [number, number, number];
  themeColor: string;
  accentColor: string;
}

interface Consortium3DSceneProps {
  isSimulatingRound?: boolean;
  selectedNodeId?: string;
  mouseTilt?: { x: number; y: number };
  onSelectNode?: (nodeId: string) => void;
  onSimulationComplete?: () => void;
}

// ---------------------------------------------------------------------------
// Network Node Coordinates (Centered & Balanced Layout)
// ---------------------------------------------------------------------------
const NODES: HospitalNodeData[] = [
  {
    id: "node-1",
    shortName: "BCN-20000",
    location: "Barcelona, ES",
    modality: "Dermoscopy",
    dpEpsilon: 1.42,
    position: [-2.2, 0, -1.3],
    themeColor: "#10b981",
    accentColor: "#34d399",
  },
  {
    id: "node-3",
    shortName: "ViDIR",
    location: "Vienna, AT",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    position: [2.2, 0, -1.3],
    themeColor: "#06b6d4",
    accentColor: "#38bdf8",
  },
  {
    id: "node-4",
    shortName: "Queensland",
    location: "Brisbane, AU",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    position: [-2.2, 0, 1.3],
    themeColor: "#10b981",
    accentColor: "#34d399",
  },
  {
    id: "node-2",
    shortName: "Beth Israel",
    location: "Boston, USA",
    modality: "Chest X-Ray",
    dpEpsilon: 1.38,
    position: [2.2, 0, 1.3],
    themeColor: "#0284c7",
    accentColor: "#38bdf8",
  },
];

// ---------------------------------------------------------------------------
// Sub-component: Hospital Node Computing Tower
// Dark titanium octagonal server chassis with floating holographic crystal
// and sleek floating micro-capsule badge (zero clutter, zero overlap).
// ---------------------------------------------------------------------------
function HospitalNodeTower({
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
  const crystalRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const floatGroup = useRef<THREE.Group>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    // Gentle float oscillation
    if (floatGroup.current) {
      const lev = isSelected
        ? Math.sin(t * 3) * 0.05 + 0.1
        : hovered
          ? 0.06
          : Math.sin(t * 2 + node.position[0]) * 0.02;
      floatGroup.current.position.y = lev;
    }

    // Holographic crystal rotation
    if (crystalRef.current) {
      crystalRef.current.rotation.y = t * 1.4;
      crystalRef.current.rotation.x = Math.sin(t * 0.9) * 0.25;
    }

    // Slow orbiting beacon ring
    if (ringRef.current) {
      ringRef.current.rotation.z = -t * 0.8;
      ringRef.current.rotation.x = Math.PI / 2.3 + Math.sin(t * 0.5) * 0.1;
    }
  });

  return (
    <group position={[node.position[0], 0, node.position[2]]}>
      {/* Ground Concentric Holographic Radar Reticle */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[0.55, 0.58, 48]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.7 : isDark ? 0.3 : 0.4}
        />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <ringGeometry args={[0.38, 0.41, 6]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.8 : isDark ? 0.35 : 0.45}
        />
      </mesh>

      {/* Floating Interactive Computing Assembly */}
      <group ref={floatGroup}>
        {/* Octagonal Polished Titanium / Dark Glass Server Chassis */}
        <mesh
          position={[0, 0.28, 0]}
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
          <cylinderGeometry args={[0.42, 0.46, 0.52, 8]} />
          <meshPhysicalMaterial
            color={isDark ? "#090d16" : "#1e293b"}
            metalness={0.88}
            roughness={0.18}
            clearcoat={0.9}
            clearcoatRoughness={0.15}
            reflectivity={0.95}
          />
          {/* Luminous Precision Edges */}
          <Edges
            scale={1.002}
            threshold={15}
            color={isSelected ? "#ffffff" : node.themeColor}
          />
        </mesh>

        {/* Luminous Chassis Collar Ring */}
        <mesh position={[0, 0.54, 0]}>
          <cylinderGeometry args={[0.32, 0.34, 0.04, 8]} />
          <meshBasicMaterial color={node.themeColor} />
        </mesh>

        {/* Floating Holographic Crystal (Octahedron) */}
        <mesh ref={crystalRef} position={[0, 0.78, 0]}>
          <octahedronGeometry args={[0.14, 0]} />
          <meshPhysicalMaterial
            color={node.accentColor}
            emissive={node.themeColor}
            emissiveIntensity={isSelected ? 1.0 : 0.6}
            roughness={0.05}
            metalness={0.2}
            clearcoat={1}
            transparent
            opacity={0.9}
          />
          <Edges scale={1.02} color="#ffffff" threshold={1} />
        </mesh>

        {/* Orbiting Laser Halo */}
        <mesh ref={ringRef} position={[0, 0.78, 0]}>
          <torusGeometry args={[0.24, 0.008, 8, 32]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.65}
          />
        </mesh>

        {/* Local Soft Glow Light */}
        <pointLight
          position={[0, 0.78, 0]}
          color={node.themeColor}
          intensity={isSelected ? 3.0 : 1.2}
          distance={2.5}
          decay={2}
        />

        {/* Sleek Ultra-Compact Floating Micro-Capsule (Crowned Above Node) */}
        <Html
          position={[0, 1.15, 0]}
          center
          distanceFactor={7}
          zIndexRange={[10, 0]}
          style={{ pointerEvents: "none", userSelect: "none" }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "5px",
              padding: "3px 8px",
              borderRadius: "9999px",
              background: isDark
                ? "rgba(10, 15, 29, 0.88)"
                : "rgba(15, 23, 42, 0.92)",
              border: `1px solid ${isSelected ? "#ffffff" : isDark ? "rgba(71, 85, 105, 0.6)" : "rgba(51, 65, 85, 0.8)"}`,
              boxShadow: isDark
                ? "0 4px 16px rgba(0, 0, 0, 0.7)"
                : "0 4px 14px rgba(15, 23, 42, 0.35)",
              backdropFilter: "blur(8px)",
              fontFamily: "'JetBrains Mono', monospace",
              whiteSpace: "nowrap",
              color: "#ffffff",
              fontSize: "10px",
              lineHeight: 1,
            }}
          >
            <span
              style={{
                width: "6px",
                height: "6px",
                borderRadius: "50%",
                backgroundColor: node.themeColor,
                boxShadow: `0 0 6px ${node.themeColor}`,
                display: "inline-block",
                flexShrink: 0,
              }}
            />
            <span style={{ fontWeight: 700, letterSpacing: "0.3px" }}>
              {node.shortName}
            </span>
            <span style={{ color: "#64748b", fontSize: "9px" }}>·</span>
            <span
              style={{
                color: "#34d399",
                fontWeight: 600,
                fontSize: "9px",
              }}
            >
              ε={node.dpEpsilon}
            </span>
          </div>
        </Html>
      </group>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Central Consortium Aggregator Hyper-Core
// Multi-axis gyroscope rings, pulsating icosahedral encryption core,
// and ground-anchored status capsule (never overlaps nodes).
// ---------------------------------------------------------------------------
function CentralConsortiumCore({ isDark }: { isDark: boolean }) {
  const gyroRing1Ref = useRef<THREE.Mesh>(null);
  const gyroRing2Ref = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const forcefieldRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    // Multi-axis gyroscope rotation
    if (gyroRing1Ref.current) {
      gyroRing1Ref.current.rotation.x = t * 0.75;
      gyroRing1Ref.current.rotation.y = t * 0.45;
    }
    if (gyroRing2Ref.current) {
      gyroRing2Ref.current.rotation.y = -t * 0.9;
      gyroRing2Ref.current.rotation.z = t * 0.55;
    }

    // Cryptographic core pulsating rotation
    if (coreRef.current) {
      coreRef.current.rotation.y = t * 0.6;
      coreRef.current.rotation.x = Math.sin(t * 0.8) * 0.25;
      const s = 1 + Math.sin(t * 3.5) * 0.1;
      coreRef.current.scale.set(s, s, s);
    }

    // Translucent containment forcefield
    if (forcefieldRef.current) {
      const mat = forcefieldRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = 0.07 + Math.sin(t * 2.2) * 0.03;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Ground Concentric Energy Pad */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <circleGeometry args={[1.3, 48]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={isDark ? 0.09 : 0.06}
        />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <ringGeometry args={[0.95, 0.98, 64]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={isDark ? 0.35 : 0.45}
        />
      </mesh>

      {/* Octagonal Cyber-Pedestal */}
      <mesh position={[0, 0.12, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.7, 0.78, 0.24, 8]} />
        <meshPhysicalMaterial
          color={isDark ? "#090d16" : "#0f172a"}
          metalness={0.9}
          roughness={0.2}
          clearcoat={0.8}
        />
        <Edges scale={1.002} threshold={15} color="#10b981" />
      </mesh>

      {/* Secondary Pedestal Column */}
      <mesh position={[0, 0.36, 0]} castShadow receiveShadow>
        <cylinderGeometry args={[0.42, 0.46, 0.24, 8]} />
        <meshPhysicalMaterial
          color={isDark ? "#0f172a" : "#1e293b"}
          metalness={0.85}
          roughness={0.25}
          clearcoat={0.7}
        />
        <Edges scale={1.002} threshold={15} color="#06b6d4" />
      </mesh>

      {/* Forcefield Containment Cylinder */}
      <mesh ref={forcefieldRef} position={[0, 1.05, 0]}>
        <cylinderGeometry args={[0.48, 0.48, 0.9, 32, 1, true]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={0.08}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Suspended Cryptographic Quantum Core Assembly (Y = 1.05) */}
      <group position={[0, 1.05, 0]}>
        {/* Core Icosahedron */}
        <mesh ref={coreRef}>
          <icosahedronGeometry args={[0.2, 0]} />
          <meshPhysicalMaterial
            color="#34d399"
            emissive="#10b981"
            emissiveIntensity={0.85}
            roughness={0.08}
            metalness={0.1}
            clearcoat={1}
            transparent
            opacity={0.9}
          />
          <Edges scale={1.02} color="#ffffff" threshold={1} />
        </mesh>

        {/* Gyroscope Orbit Ring 1 (Emerald) */}
        <mesh ref={gyroRing1Ref}>
          <torusGeometry args={[0.38, 0.01, 12, 64]} />
          <meshPhysicalMaterial
            color="#10b981"
            emissive="#10b981"
            emissiveIntensity={0.7}
            metalness={0.9}
            roughness={0.15}
          />
        </mesh>

        {/* Gyroscope Orbit Ring 2 (Cyan) */}
        <mesh ref={gyroRing2Ref}>
          <torusGeometry args={[0.3, 0.009, 12, 64]} />
          <meshPhysicalMaterial
            color="#06b6d4"
            emissive="#06b6d4"
            emissiveIntensity={0.7}
            metalness={0.9}
            roughness={0.15}
          />
        </mesh>

        {/* Core Point Light */}
        <pointLight
          color="#10b981"
          intensity={4.5}
          distance={4.0}
          decay={2}
        />
      </group>

      {/* Ground-Anchored Minimalist Status Capsule (Positioned in front, zero overlap) */}
      <Html
        position={[0, 0.02, 0.75]}
        center
        distanceFactor={7}
        zIndexRange={[10, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "6px",
            padding: "4px 10px",
            borderRadius: "9999px",
            background: isDark
              ? "rgba(10, 15, 29, 0.9)"
              : "rgba(15, 23, 42, 0.94)",
            border: "1px solid rgba(16, 185, 129, 0.5)",
            boxShadow: isDark
              ? "0 4px 18px rgba(0, 0, 0, 0.75)"
              : "0 4px 16px rgba(15, 23, 42, 0.4)",
            backdropFilter: "blur(10px)",
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: "nowrap",
            color: "#ffffff",
            fontSize: "10.5px",
            lineHeight: 1,
          }}
        >
          <span
            style={{
              width: "7px",
              height: "7px",
              borderRadius: "50%",
              backgroundColor: "#10b981",
              boxShadow: "0 0 8px #10b981",
              display: "inline-block",
              flexShrink: 0,
            }}
          />
          <span
            style={{
              fontWeight: 800,
              letterSpacing: "0.5px",
              color: "#34d399",
            }}
          >
            CONSORTIUM AGGREGATOR
          </span>
          <span style={{ color: "#64748b", fontSize: "9px" }}>·</span>
          <span
            style={{
              color: "#94a3b8",
              fontWeight: 600,
              fontSize: "9px",
            }}
          >
            CKKS 128b
          </span>
        </div>
      </Html>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: High-Speed Micro-Photon Packet
// ---------------------------------------------------------------------------
function HighSpeedPhoton({
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
  const meshRef = useRef<THREE.Mesh>(null);
  const progress = useRef(offset);

  useFrame((_, delta) => {
    progress.current = (progress.current + delta * 0.32 * speedMul) % 1;
    const pt = curve.getPoint(progress.current);
    if (meshRef.current) {
      meshRef.current.position.copy(pt);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.032, 10, 10]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Fiber-Optic Encrypted Conduit
// ---------------------------------------------------------------------------
function FiberConduit({
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
  const { curve, points } = useMemo(() => {
    const start = new THREE.Vector3(nodePos[0], 0.78, nodePos[2]);
    const end = new THREE.Vector3(0, 1.05, 0);
    const midX = (start.x + end.x) / 2 + (idx % 2 === 0 ? -0.2 : 0.2);
    const midY = Math.max(start.y, end.y) + 0.45;
    const midZ = (start.z + end.z) / 2;
    const mid = new THREE.Vector3(midX, midY, midZ);
    const c = new THREE.QuadraticBezierCurve3(start, mid, end);
    const pts = c
      .getPoints(32)
      .map((p) => [p.x, p.y, p.z] as [number, number, number]);
    return { curve: c, points: pts };
  }, [nodePos, idx]);

  const speedMul = isSimulating ? 3.8 : isSelected ? 1.8 : 1.0;

  return (
    <group>
      {/* Sleek Laser Fiber Line */}
      <Line
        points={points}
        color={color}
        lineWidth={1.4}
        transparent
        opacity={isSelected ? 0.7 : 0.3}
      />

      {/* 3 Streamlined Micro-Photons per Conduit */}
      {[0, 0.33, 0.66].map((off, i) => (
        <HighSpeedPhoton
          key={i}
          curve={curve}
          offset={off}
          color="#ffffff"
          speedMul={speedMul}
        />
      ))}
    </group>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Consensus Ripple Shockwave
// ---------------------------------------------------------------------------
function ConsensusShockwaveRipple({
  active,
  onComplete,
}: {
  active: boolean;
  onComplete: () => void;
}) {
  const outerRef = useRef<THREE.Mesh>(null);
  const innerRef = useRef<THREE.Mesh>(null);
  const progressRef = useRef(0);
  const runningRef = useRef(false);
  const prevActive = useRef(false);

  useEffect(() => {
    if (active && !prevActive.current) {
      runningRef.current = true;
      progressRef.current = 0;
      if (outerRef.current) outerRef.current.visible = true;
      if (innerRef.current) innerRef.current.visible = true;
    }
    prevActive.current = active;
  }, [active]);

  useFrame((_, delta) => {
    if (!runningRef.current) return;
    progressRef.current += delta * 0.95;

    if (progressRef.current >= 1) {
      runningRef.current = false;
      if (outerRef.current) outerRef.current.visible = false;
      if (innerRef.current) innerRef.current.visible = false;
      onComplete();
      return;
    }

    const p = progressRef.current;
    const s = 1 + p * 11;
    const alpha = (1 - p) * 0.6;

    if (outerRef.current) {
      outerRef.current.scale.set(s, s, 1);
      (outerRef.current.material as THREE.MeshBasicMaterial).opacity = alpha;
    }
    if (innerRef.current && p > 0.05) {
      const s2 = 1 + (p - 0.05) * 11;
      innerRef.current.scale.set(s2, s2, 1);
      (innerRef.current.material as THREE.MeshBasicMaterial).opacity =
        alpha * 0.7;
    }
  });

  return (
    <>
      <mesh
        ref={outerRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.015, 0]}
        visible={false}
      >
        <ringGeometry args={[0.28, 0.32, 64]} />
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
        position={[0, 0.015, 0]}
        visible={false}
      >
        <ringGeometry args={[0.2, 0.24, 64]} />
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
// Scene Root: Studio Lighting, Parallax, and Geometry Composition
// ---------------------------------------------------------------------------
function SceneRoot({
  isSimulatingRound = false,
  selectedNodeId = "node-1",
  mouseTilt = { x: 0, y: 0 },
  onSelectNode,
  onSimulationComplete,
}: Consortium3DSceneProps) {
  const sceneRef = useRef<THREE.Group>(null);
  const isDark = useIsDark();
  const tilt = useRef({ x: 0, y: 0 });

  useFrame(() => {
    tilt.current.x += (mouseTilt.x - tilt.current.x) * 0.06;
    tilt.current.y += (mouseTilt.y - tilt.current.y) * 0.06;
    if (sceneRef.current) {
      sceneRef.current.rotation.y = tilt.current.x * 0.14;
      sceneRef.current.rotation.x = tilt.current.y * 0.07;
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
      {/* Studio Lighting Rig */}
      <ambientLight intensity={isDark ? 0.45 : 0.9} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={isDark ? 2.2 : 2.5}
        color="#ffffff"
        castShadow
      />
      <directionalLight
        position={[-5, 5, -5]}
        intensity={isDark ? 0.8 : 0.9}
        color="#bae6fd"
      />
      <pointLight
        position={[0, 3.5, 0]}
        intensity={isDark ? 2.5 : 1.2}
        color="#10b981"
        distance={8}
        decay={2}
      />

      {/* Global Parallax Group */}
      <group ref={sceneRef}>
        {/* Subtle Cybernetic Floor Reticle Rings */}
        {[1.6, 2.8, 4.0].map((rad, i) => (
          <mesh
            key={i}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, 0.002, 0]}
          >
            <ringGeometry args={[rad - 0.006, rad, 64]} />
            <meshBasicMaterial
              color={isDark ? "#334155" : "#cbd5e1"}
              transparent
              opacity={isDark ? 0.25 : 0.35}
            />
          </mesh>
        ))}

        {/* 4 Hospital Institutional Computing Towers */}
        {NODES.map((n) => (
          <HospitalNodeTower
            key={n.id}
            node={n}
            isSelected={selectedNodeId === n.id}
            isDark={isDark}
            onSelect={handleSelect}
          />
        ))}

        {/* Central Consortium Aggregator Server Core */}
        <CentralConsortiumCore isDark={isDark} />

        {/* High-Speed Fiber Conduits */}
        {NODES.map((n, i) => (
          <FiberConduit
            key={n.id}
            nodePos={n.position}
            color={n.themeColor}
            isSimulating={isSimulatingRound}
            isSelected={selectedNodeId === n.id}
            idx={i}
          />
        ))}

        {/* Consensus Shockwave Animation */}
        <ConsensusShockwaveRipple
          active={isSimulatingRound}
          onComplete={handleSimDone}
        />
      </group>
    </>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Responsive Camera Controller
// High-angle architectural isometric perspective (~50° tilt) that clearly
// separates the top, center, and bottom nodes without vertical overlap.
// ---------------------------------------------------------------------------
function ResponsiveCameraRig() {
  const { camera, size } = useThree();

  useFrame(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      const aspect = size.width / Math.max(size.height, 1);
      // High-angle isometric camera distance
      const baseDistance = 6.2;
      const targetDistance =
        aspect < 1.35
          ? baseDistance * (1.35 / Math.max(aspect, 0.6))
          : baseDistance;

      // Camera position: Elevated isometric (~50 degree pitch)
      const targetY = targetDistance * 0.85;
      const targetZ = targetDistance * 0.75;

      camera.position.y += (targetY - camera.position.y) * 0.08;
      camera.position.z += (targetZ - camera.position.z) * 0.08;
      camera.lookAt(0, 0.25, 0);
      camera.updateProjectionMatrix();
    }
  });

  return null;
}

// ---------------------------------------------------------------------------
// Consortium3DScene: Exported Canvas Wrapper
// ---------------------------------------------------------------------------
export default function Consortium3DScene(props: Consortium3DSceneProps) {
  return (
    <Canvas
      className="w-full h-full"
      dpr={[1, 2]}
      camera={{ position: [0, 5.8, 5.0], fov: 36 }}
      gl={{
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: false,
        toneMapping: THREE.ACESFilmicToneMapping,
      }}
      onCreated={({ gl, camera, scene }) => {
        gl.toneMappingExposure = 1.15;
        gl.setClearColor(0x000000, 0);
        camera.lookAt(0, 0.25, 0);
        scene.background = null;
        gl.domElement.addEventListener(
          "webglcontextlost",
          (e) => e.preventDefault(),
          false,
        );
      }}
      style={{ width: "100%", height: "100%", background: "transparent" }}
    >
      <ResponsiveCameraRig />
      <SceneRoot {...props} />
    </Canvas>
  );
}
