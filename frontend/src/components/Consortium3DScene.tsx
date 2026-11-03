"use client";

import React, { useRef, useMemo, useState, useEffect, useCallback } from "react";
import * as THREE from "three";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { Html, Line, RoundedBox, Edges } from "@react-three/drei";

// Hook to detect dark mode via document element class observer
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
    position: [-2.15, 0, -1.05],
    themeColor: "#10b981",
    accentColor: "#34d399",
  },
  {
    id: "node-3",
    shortName: "ViDIR",
    location: "Vienna, AT",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    position: [2.15, 0, -1.05],
    themeColor: "#06b6d4",
    accentColor: "#38bdf8",
  },
  {
    id: "node-4",
    shortName: "Queensland",
    location: "Brisbane, AU",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    position: [-2.15, 0, 1.05],
    themeColor: "#10b981",
    accentColor: "#34d399",
  },
  {
    id: "node-2",
    shortName: "Beth Israel",
    location: "Boston, USA",
    modality: "Chest X-Ray",
    dpEpsilon: 1.38,
    position: [2.15, 0, 1.05],
    themeColor: "#0284c7",
    accentColor: "#38bdf8",
  },
];

// ---------------------------------------------------------------------------
// Sub-component: Hospital Node Edge Pod
// Sleek frosted glass / titanium computing block with rotating holographic
// diamond crystal, precision glowing bevel edges, and high-contrast badge.
// ---------------------------------------------------------------------------
function HospitalNodePod({
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
  const groupRef = useRef<THREE.Group>(null);
  const crystalRef = useRef<THREE.Mesh>(null);
  const ringRef = useRef<THREE.Mesh>(null);
  const [hovered, setHovered] = useState(false);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    // Subtle breathing elevation for the pod
    if (groupRef.current) {
      const lev = isSelected
        ? Math.sin(t * 3) * 0.08 + 0.16
        : hovered
          ? 0.1
          : Math.sin(t * 2 + node.position[0]) * 0.03;
      groupRef.current.position.y = lev;
    }

    // Rotating holographic crystal above the pod
    if (crystalRef.current) {
      crystalRef.current.rotation.y = t * 1.2;
      crystalRef.current.rotation.x = Math.sin(t * 0.8) * 0.2;
    }

    // Counter-rotating targeting reticle
    if (ringRef.current) {
      ringRef.current.rotation.z = -t * 0.6;
    }
  });

  return (
    <group position={[node.position[0], 0, node.position[2]]}>
      {/* Ground Concentric Radar Reticle */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <ringGeometry args={[0.65, 0.68, 48]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.6 : isDark ? 0.25 : 0.35}
        />
      </mesh>
      <mesh
        ref={ringRef}
        rotation={[-Math.PI / 2, 0, 0]}
        position={[0, 0.008, 0]}
      >
        <ringGeometry args={[0.48, 0.51, 6]} />
        <meshBasicMaterial
          color={node.themeColor}
          transparent
          opacity={isSelected ? 0.7 : isDark ? 0.3 : 0.4}
        />
      </mesh>

      {/* Floating Interactive Computing Pod Group */}
      <group ref={groupRef}>
        {/* Main Computing Pod Chassis */}
        <mesh
          position={[0, 0.35, 0]}
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
          <RoundedBox args={[0.74, 0.7, 0.74]} radius={0.08} smoothness={4}>
            <meshPhysicalMaterial
              color={isDark ? "#0f172a" : "#ffffff"}
              metalness={isDark ? 0.6 : 0.15}
              roughness={isDark ? 0.25 : 0.35}
              clearcoat={0.7}
              clearcoatRoughness={0.2}
              transmission={isDark ? 0.3 : 0.6}
              opacity={isDark ? 0.95 : 0.92}
              transparent
              reflectivity={0.9}
            />
            {/* Luminous Neon Edges */}
            <Edges
              scale={1.002}
              threshold={15}
              color={isSelected ? "#ffffff" : node.themeColor}
            />
          </RoundedBox>
        </mesh>

        {/* Laser Emitter Beam to Crystal */}
        <mesh position={[0, 0.75, 0]}>
          <cylinderGeometry args={[0.008, 0.008, 0.12, 8]} />
          <meshBasicMaterial
            color={node.themeColor}
            transparent
            opacity={0.8}
          />
        </mesh>

        {/* Holographic Floating Crystal (Octahedron) */}
        <mesh ref={crystalRef} position={[0, 0.92, 0]}>
          <octahedronGeometry args={[0.13, 0]} />
          <meshPhysicalMaterial
            color={node.accentColor}
            emissive={node.themeColor}
            emissiveIntensity={isSelected ? 0.8 : 0.4}
            roughness={0.1}
            metalness={0.2}
            clearcoat={1}
            transparent
            opacity={0.85}
          />
          <Edges scale={1.02} color="#ffffff" threshold={1} />
        </mesh>

        {/* Local Soft Glow Light */}
        <pointLight
          position={[0, 0.92, 0]}
          color={node.themeColor}
          intensity={isSelected ? 3.5 : 1.2}
          distance={2.2}
          decay={2}
        />
      </group>

      {/* High-Contrast Floating Metadata Badge */}
      <Html
        position={[0, 0.02, 0.72]}
        center
        distanceFactor={7.5}
        zIndexRange={[10, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: "nowrap",
            padding: "4px 8px",
            borderRadius: "8px",
            background: isDark
              ? "rgba(15, 23, 42, 0.88)"
              : "rgba(255, 255, 255, 0.92)",
            border: isDark
              ? `1px solid ${isSelected ? node.themeColor : "rgba(51, 65, 85, 0.7)"}`
              : `1px solid ${isSelected ? node.themeColor : "rgba(226, 232, 240, 0.95)"}`,
            boxShadow: isDark
              ? "0 4px 16px rgba(0, 0, 0, 0.55)"
              : "0 4px 12px rgba(15, 23, 42, 0.08)",
            backdropFilter: "blur(8px)",
            textAlign: "center",
            transform: "translateY(0)",
            transition: "all 0.2s ease",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "5px",
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
              }}
            />
            <span
              style={{
                fontSize: "10.5px",
                fontWeight: 700,
                color: isDark ? "#f8fafc" : "#0f172a",
                letterSpacing: "0.4px",
              }}
            >
              {node.shortName}
            </span>
          </div>

          <div
            style={{
              fontSize: "8px",
              color: isDark ? "#94a3b8" : "#475569",
              marginTop: "2px",
              fontFamily: "'Inter', sans-serif",
            }}
          >
            {node.location} · {node.modality}
          </div>

          <div
            style={{
              fontSize: "8.5px",
              fontWeight: 700,
              color: node.themeColor,
              marginTop: "2px",
              letterSpacing: "0.3px",
            }}
          >
            ε = {node.dpEpsilon}
          </div>
        </div>
      </Html>
    </group>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: Central Consortium Aggregator
// Architectural server pedestal with precision gyroscope rings, glowing
// icosahedral RLWE CKKS encryption core, and floating status badge.
// ---------------------------------------------------------------------------
function CentralConsortiumCore({ isDark }: { isDark: boolean }) {
  const gyroRing1Ref = useRef<THREE.Mesh>(null);
  const gyroRing2Ref = useRef<THREE.Mesh>(null);
  const gyroRing3Ref = useRef<THREE.Mesh>(null);
  const coreRef = useRef<THREE.Mesh>(null);
  const columnRef = useRef<THREE.Mesh>(null);

  useFrame(({ clock }) => {
    const t = clock.elapsedTime;

    // Multi-axis gyroscope rotation
    if (gyroRing1Ref.current) {
      gyroRing1Ref.current.rotation.x = t * 0.7;
      gyroRing1Ref.current.rotation.y = t * 0.4;
    }
    if (gyroRing2Ref.current) {
      gyroRing2Ref.current.rotation.y = -t * 0.85;
      gyroRing2Ref.current.rotation.z = t * 0.5;
    }
    if (gyroRing3Ref.current) {
      gyroRing3Ref.current.rotation.x = -t * 0.5;
      gyroRing3Ref.current.rotation.z = -t * 0.75;
    }

    // Cryptographic core pulsating breath
    if (coreRef.current) {
      coreRef.current.rotation.y = t * 0.6;
      coreRef.current.rotation.x = Math.sin(t * 0.8) * 0.3;
      const s = 1 + Math.sin(t * 3.5) * 0.12;
      coreRef.current.scale.set(s, s, s);
    }

    // Vertical containment energy pulse
    if (columnRef.current) {
      const mat = columnRef.current.material as THREE.MeshBasicMaterial;
      mat.opacity = 0.08 + Math.sin(t * 2.5) * 0.04;
    }
  });

  return (
    <group position={[0, 0, 0]}>
      {/* Ground Concentric Energy Pad */}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.005, 0]}>
        <circleGeometry args={[1.5, 48]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={isDark ? 0.08 : 0.05}
        />
      </mesh>
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0.008, 0]}>
        <ringGeometry args={[1.1, 1.13, 64]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={isDark ? 0.3 : 0.4}
        />
      </mesh>

      {/* Cybernetic Server Base Pedestal */}
      <mesh position={[0, 0.12, 0]} castShadow receiveShadow>
        <RoundedBox args={[1.4, 0.24, 1.4]} radius={0.06} smoothness={4}>
          <meshPhysicalMaterial
            color={isDark ? "#090d16" : "#ffffff"}
            metalness={isDark ? 0.7 : 0.2}
            roughness={isDark ? 0.3 : 0.4}
            clearcoat={0.6}
            transparent
            opacity={0.96}
          />
          <Edges scale={1.002} threshold={15} color="#10b981" />
        </RoundedBox>
      </mesh>

      {/* Server Column Core */}
      <mesh position={[0, 0.52, 0]} castShadow receiveShadow>
        <RoundedBox args={[0.76, 0.56, 0.76]} radius={0.05} smoothness={4}>
          <meshPhysicalMaterial
            color={isDark ? "#0f172a" : "#f8fafc"}
            metalness={isDark ? 0.8 : 0.25}
            roughness={0.3}
            clearcoat={0.7}
          />
          <Edges scale={1.002} threshold={15} color="#06b6d4" />
        </RoundedBox>
      </mesh>

      {/* Cylindrical Containment Field */}
      <mesh ref={columnRef} position={[0, 1.35, 0]}>
        <cylinderGeometry args={[0.55, 0.55, 1.1, 32, 1, true]} />
        <meshBasicMaterial
          color="#10b981"
          transparent
          opacity={0.1}
          side={THREE.DoubleSide}
        />
      </mesh>

      {/* Cryptographic Quantum Enclave Core (Y = 1.35) */}
      <group position={[0, 1.35, 0]}>
        {/* Core Icosahedron */}
        <mesh ref={coreRef}>
          <icosahedronGeometry args={[0.22, 0]} />
          <meshPhysicalMaterial
            color="#34d399"
            emissive="#10b981"
            emissiveIntensity={0.9}
            roughness={0.1}
            metalness={0.1}
            clearcoat={1}
            transparent
            opacity={0.9}
          />
          <Edges scale={1.02} color="#ffffff" threshold={1} />
        </mesh>

        {/* Gyroscope Orbit Ring 1 (Emerald) */}
        <mesh ref={gyroRing1Ref}>
          <torusGeometry args={[0.46, 0.012, 12, 64]} />
          <meshPhysicalMaterial
            color="#10b981"
            emissive="#10b981"
            emissiveIntensity={0.6}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>

        {/* Gyroscope Orbit Ring 2 (Cyan) */}
        <mesh ref={gyroRing2Ref}>
          <torusGeometry args={[0.38, 0.01, 12, 64]} />
          <meshPhysicalMaterial
            color="#06b6d4"
            emissive="#06b6d4"
            emissiveIntensity={0.6}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>

        {/* Gyroscope Orbit Ring 3 (Sky) */}
        <mesh ref={gyroRing3Ref}>
          <torusGeometry args={[0.3, 0.008, 12, 64]} />
          <meshPhysicalMaterial
            color="#38bdf8"
            emissive="#38bdf8"
            emissiveIntensity={0.6}
            metalness={0.8}
            roughness={0.2}
          />
        </mesh>

        {/* Central Luminous Point Light */}
        <pointLight
          color="#10b981"
          intensity={5}
          distance={4.5}
          decay={2}
        />
      </group>

      {/* High-Contrast Central Aggregator Badge */}
      <Html
        position={[0, 0.02, 1.05]}
        center
        distanceFactor={7.5}
        zIndexRange={[10, 0]}
        style={{ pointerEvents: "none", userSelect: "none" }}
      >
        <div
          style={{
            fontFamily: "'JetBrains Mono', monospace",
            whiteSpace: "nowrap",
            padding: "5px 12px",
            borderRadius: "10px",
            background: isDark
              ? "rgba(15, 23, 42, 0.92)"
              : "rgba(255, 255, 255, 0.94)",
            border: isDark
              ? "1px solid rgba(16, 185, 129, 0.4)"
              : "1px solid rgba(16, 185, 129, 0.5)",
            boxShadow: isDark
              ? "0 4px 20px rgba(0, 0, 0, 0.65)"
              : "0 4px 14px rgba(15, 23, 42, 0.1)",
            backdropFilter: "blur(10px)",
            textAlign: "center",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              gap: "6px",
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
              }}
            />
            <span
              style={{
                fontSize: "11px",
                fontWeight: 800,
                color: isDark ? "#f8fafc" : "#0f172a",
                letterSpacing: "0.6px",
              }}
            >
              CONSORTIUM AGGREGATOR
            </span>
          </div>

          <div
            style={{
              fontSize: "8.5px",
              fontWeight: 600,
              color: isDark ? "#94a3b8" : "#475569",
              marginTop: "2px",
              letterSpacing: "0.2px",
            }}
          >
            RLWE CKKS 128-Bit · Zero Plaintext
          </div>
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
    progress.current = (progress.current + delta * 0.28 * speedMul) % 1;
    const pt = curve.getPoint(progress.current);
    if (meshRef.current) {
      meshRef.current.position.copy(pt);
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.035, 12, 12]} />
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
    const start = new THREE.Vector3(nodePos[0], 0.92, nodePos[2]);
    const end = new THREE.Vector3(0, 1.35, 0);
    const midX = (start.x + end.x) / 2 + (idx % 2 === 0 ? -0.25 : 0.25);
    const midY = Math.max(start.y, end.y) + 0.6;
    const midZ = (start.z + end.z) / 2;
    const mid = new THREE.Vector3(midX, midY, midZ);
    const c = new THREE.QuadraticBezierCurve3(start, mid, end);
    const pts = c
      .getPoints(36)
      .map((p) => [p.x, p.y, p.z] as [number, number, number]);
    return { curve: c, points: pts };
  }, [nodePos, idx]);

  const speedMul = isSimulating ? 3.8 : isSelected ? 1.8 : 1.0;

  return (
    <group>
      {/* Sleek Fiber Line */}
      <Line
        points={points}
        color={color}
        lineWidth={1.5}
        transparent
        opacity={isSelected ? 0.6 : 0.25}
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
    progressRef.current += delta * 0.9;

    if (progressRef.current >= 1) {
      runningRef.current = false;
      if (outerRef.current) outerRef.current.visible = false;
      if (innerRef.current) innerRef.current.visible = false;
      onComplete();
      return;
    }

    const p = progressRef.current;
    const s = 1 + p * 12;
    const alpha = (1 - p) * 0.6;

    if (outerRef.current) {
      outerRef.current.scale.set(s, s, 1);
      (outerRef.current.material as THREE.MeshBasicMaterial).opacity = alpha;
    }
    if (innerRef.current && p > 0.06) {
      const s2 = 1 + (p - 0.06) * 12;
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
        <ringGeometry args={[0.3, 0.35, 64]} />
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
        <ringGeometry args={[0.22, 0.27, 64]} />
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
// Scene Root: Lighting, Parallax, and Geometry Composition
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
      sceneRef.current.rotation.y = tilt.current.x * 0.16;
      sceneRef.current.rotation.x = tilt.current.y * 0.08;
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
      <ambientLight intensity={isDark ? 0.35 : 0.8} />
      <directionalLight
        position={[5, 8, 5]}
        intensity={isDark ? 2.2 : 2.6}
        color="#ffffff"
        castShadow
      />
      <directionalLight
        position={[-5, 5, -5]}
        intensity={isDark ? 0.8 : 0.9}
        color="#bae6fd"
      />
      <pointLight
        position={[0, 4, 0]}
        intensity={isDark ? 2.5 : 1.2}
        color="#10b981"
        distance={10}
        decay={2}
      />

      {/* Global Parallax Group */}
      <group ref={sceneRef}>
        {/* Subtle Cybernetic Floor Reticle */}
        {[1.8, 3.2, 4.6].map((rad, i) => (
          <mesh
            key={i}
            rotation={[-Math.PI / 2, 0, 0]}
            position={[0, 0.002, 0]}
          >
            <ringGeometry args={[rad - 0.008, rad, 64]} />
            <meshBasicMaterial
              color={isDark ? "#334155" : "#cbd5e1"}
              transparent
              opacity={isDark ? 0.25 : 0.35}
            />
          </mesh>
        ))}

        {/* 4 Hospital Institutional Computing Pods */}
        {NODES.map((n) => (
          <HospitalNodePod
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
// Dynamically adjusts camera distance and elevation based on viewport aspect
// ratio, ensuring full visibility of all nodes and badges on mobile viewports.
// ---------------------------------------------------------------------------
function ResponsiveCameraRig() {
  const { camera, size } = useThree();

  useFrame(() => {
    if (camera instanceof THREE.PerspectiveCamera) {
      const aspect = size.width / Math.max(size.height, 1);
      // When aspect is narrow (e.g. mobile portrait < 1.45), pull camera back
      // proportionally to keep the entire network comfortably contained.
      const baseDistance = 5.8;
      const targetDistance =
        aspect < 1.4
          ? baseDistance * (1.4 / Math.max(aspect, 0.55))
          : baseDistance;

      const targetY = targetDistance * 0.72;
      const targetZ = targetDistance * 0.95;

      camera.position.y += (targetY - camera.position.y) * 0.08;
      camera.position.z += (targetZ - camera.position.z) * 0.08;
      camera.lookAt(0, 0.4, 0);
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
      camera={{ position: [0, 5.2, 6.2], fov: 38 }}
      gl={{
        antialias: true,
        alpha: true,
        preserveDrawingBuffer: false,
        toneMapping: THREE.ACESFilmicToneMapping,
      }}
      onCreated={({ gl, camera, scene }) => {
        gl.toneMappingExposure = 1.1;
        gl.setClearColor(0x000000, 0);
        camera.lookAt(0, 0.45, 0);
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
