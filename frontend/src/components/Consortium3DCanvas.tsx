"use client";

import React, { useEffect, useRef } from "react";

export interface Hospital3DNode {
  id: string;
  name: string;
  shortName: string;
  location: string;
  cityCode: string;
  modality: string;
  dpEpsilon: number;
  scans: number;
  wx: number; // 3D isometric world X (-1 to 1)
  wy: number; // 3D isometric world Y (-1 to 1)
  themeColor: string;
  glowColor: string;
}

interface Consortium3DCanvasProps {
  isSimulatingRound?: boolean;
  selectedNodeId?: string;
  mouseTilt?: { x: number; y: number };
  onSelectNode?: (nodeId: string) => void;
  onSimulationComplete?: () => void;
}

interface PhotonPacket {
  nodeIndex: number;
  progress: number;
  speed: number;
  size: number;
  trail: { x: number; y: number }[];
}

const DEFAULT_NODES: Hospital3DNode[] = [
  {
    id: "node-1",
    name: "BCN-20000 Skin Cancer Hub",
    shortName: "BCN-20000",
    location: "Barcelona, ES",
    cityCode: "BCN",
    modality: "Dermoscopy",
    dpEpsilon: 1.42,
    scans: 40,
    wx: -0.72,
    wy: -0.45,
    themeColor: "#10b981",
    glowColor: "rgba(16, 185, 129, 0.45)",
  },
  {
    id: "node-3",
    name: "ViDIR Dermatopathology Institute",
    shortName: "ViDIR",
    location: "Vienna, AT",
    cityCode: "VIE",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    scans: 40,
    wx: 0.72,
    wy: -0.45,
    themeColor: "#06b6d4",
    glowColor: "rgba(6, 182, 212, 0.45)",
  },
  {
    id: "node-4",
    name: "Queensland Oncology Center",
    shortName: "Queensland",
    location: "Brisbane, AU",
    cityCode: "BNE",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    scans: 40,
    wx: -0.72,
    wy: 0.45,
    themeColor: "#10b981",
    glowColor: "rgba(16, 185, 129, 0.45)",
  },
  {
    id: "node-2",
    name: "Beth Israel Deaconess Medical Center",
    shortName: "Beth Israel",
    location: "Boston, USA",
    cityCode: "BOS",
    modality: "Chest X-Ray",
    dpEpsilon: 1.38,
    scans: 40,
    wx: 0.72,
    wy: 0.45,
    themeColor: "#38bdf8",
    glowColor: "rgba(56, 189, 248, 0.45)",
  },
];

export default function Consortium3DCanvas({
  isSimulatingRound = false,
  selectedNodeId = "node-1",
  mouseTilt = { x: 0, y: 0 },
  onSelectNode,
  onSimulationComplete,
}: Consortium3DCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const hoveredNodeRef = useRef<string | null>(null);
  const currentTiltRef = useRef<{ x: number; y: number }>({ x: 0, y: 0 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    if (
      !ctx ||
      typeof ctx.scale !== "function" ||
      typeof ctx.beginPath !== "function" ||
      typeof ctx.clearRect !== "function"
    ) {
      return;
    }

    let width = 0;
    let height = 0;
    let time = 0;
    let shockwaveRadius = 0;
    let isShockwaveActive = false;

    const handleResize = () => {
      const rect = canvas.parentElement?.getBoundingClientRect();
      if (!rect) return;
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      width = rect.width;
      height = rect.height;
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      ctx.scale(dpr, dpr);
    };

    handleResize();
    window.addEventListener("resize", handleResize);

    // Initialize photon packets for encrypted laser streams
    const packets: PhotonPacket[] = [];
    for (let nodeIdx = 0; nodeIdx < 4; nodeIdx++) {
      for (let p = 0; p < 3; p++) {
        packets.push({
          nodeIndex: nodeIdx,
          progress: (p * 0.33 + nodeIdx * 0.15) % 1,
          speed: 0.004 + (p % 2) * 0.0015,
          size: 3 + (p % 2),
          trail: [],
        });
      }
    }

    if (isSimulatingRound) {
      shockwaveRadius = 10;
      isShockwaveActive = true;
    }

    // -------------------------------------------------------------
    // Isometric 3D Projection Engine
    // -------------------------------------------------------------
    const project = (
      wx: number,
      wy: number,
      wz: number,
      scaleVal: number,
      pitchVal: number,
      yawVal: number
    ) => {
      const cx = width / 2;
      const cy = height / 2 - 6;

      const cosY = Math.cos(yawVal);
      const sinY = Math.sin(yawVal);
      const rx = wx * cosY - wy * sinY;
      const ry = wx * sinY + wy * cosY;

      const cosP = Math.cos(pitchVal);
      const sinP = Math.sin(pitchVal);

      const screenX = cx + rx * scaleVal;
      const screenY = cy + ry * scaleVal * sinP - wz * cosP;
      const depth = ry * cosP + wz * sinP;

      return { x: screenX, y: screenY, depth };
    };

    // Helper: Draw 3D Extruded Polygonal Prism with Depth Shading
    const draw3DBox = (
      bx: number,
      by: number,
      bz: number,
      bw: number,
      bd: number,
      bh: number,
      scaleVal: number,
      pitchVal: number,
      yawVal: number,
      palette: {
        top: string;
        left: string;
        right: string;
        stroke: string;
        lineWidth?: number;
      }
    ) => {
      const p0 = project(bx - bw / 2, by - bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p1 = project(bx + bw / 2, by - bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p2 = project(bx + bw / 2, by + bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p3 = project(bx - bw / 2, by + bd / 2, bz, scaleVal, pitchVal, yawVal);

      const p4 = project(bx - bw / 2, by - bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p5 = project(bx + bw / 2, by - bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p6 = project(bx + bw / 2, by + bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p7 = project(bx - bw / 2, by + bd / 2, bz + bh, scaleVal, pitchVal, yawVal);

      ctx.save();
      ctx.lineWidth = palette.lineWidth || 1;
      ctx.strokeStyle = palette.stroke;

      // Left Shaded Wall Face
      ctx.beginPath();
      ctx.moveTo(p3.x, p3.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p7.x, p7.y);
      ctx.closePath();
      ctx.fillStyle = palette.left;
      ctx.fill();
      ctx.stroke();

      // Right Shaded Wall Face
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p5.x, p5.y);
      ctx.closePath();
      ctx.fillStyle = palette.right;
      ctx.fill();
      ctx.stroke();

      // Top Glass/Metallic Roof Face
      ctx.beginPath();
      ctx.moveTo(p4.x, p4.y);
      ctx.lineTo(p5.x, p5.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p7.x, p7.y);
      ctx.closePath();
      ctx.fillStyle = palette.top;
      ctx.fill();
      ctx.stroke();

      ctx.restore();
      return {
        topCenter: { x: (p4.x + p6.x) / 2, y: (p4.y + p6.y) / 2 },
        groundCenter: { x: (p0.x + p2.x) / 2, y: (p0.y + p2.y) / 2 },
      };
    };

    // -------------------------------------------------------------
    // Main 60 FPS Animation Render Loop
    // -------------------------------------------------------------
    const render = () => {
      time += 0.022;
      ctx.clearRect(0, 0, width, height);

      // Smooth camera tilt damping (lerp)
      currentTiltRef.current.x += (mouseTilt.x - currentTiltRef.current.x) * 0.08;
      currentTiltRef.current.y += (mouseTilt.y - currentTiltRef.current.y) * 0.08;

      const scaleVal = Math.min(width * 0.42, height * 0.54);
      const pitchVal = 0.56 + currentTiltRef.current.y * 0.14;
      const yawVal = currentTiltRef.current.x * 0.22;

      // -------------------------------------------------------------
      // 1. Cybernetic Floor Grid (Subtle Blueprint Depth)
      // -------------------------------------------------------------
      ctx.save();
      ctx.strokeStyle = "rgba(16, 185, 129, 0.05)";
      ctx.lineWidth = 1;

      const gridRange = 1.15;
      const gridStep = 0.23;
      for (let gx = -gridRange; gx <= gridRange + 0.001; gx += gridStep) {
        const start = project(gx, -gridRange, 0, scaleVal, pitchVal, yawVal);
        const end = project(gx, gridRange, 0, scaleVal, pitchVal, yawVal);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
      for (let gy = -gridRange; gy <= gridRange + 0.001; gy += gridStep) {
        const start = project(-gridRange, gy, 0, scaleVal, pitchVal, yawVal);
        const end = project(gridRange, gy, 0, scaleVal, pitchVal, yawVal);
        ctx.beginPath();
        ctx.moveTo(start.x, start.y);
        ctx.lineTo(end.x, end.y);
        ctx.stroke();
      }
      ctx.restore();

      const serverGround = project(0, 0, 0, scaleVal, pitchVal, yawVal);

      // -------------------------------------------------------------
      // 2. Pedestal Ground Rings & Ambient Shadows
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node) => {
        const p = project(node.wx, node.wy, 0, scaleVal, pitchVal, yawVal);
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNodeRef.current;

        ctx.save();
        // Soft contact drop-shadow
        const shadowGrad = ctx.createRadialGradient(p.x, p.y, 4, p.x, p.y, 42);
        shadowGrad.addColorStop(0, "rgba(0, 0, 0, 0.65)");
        shadowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = shadowGrad;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, 44, 22, 0, 0, Math.PI * 2);
        ctx.fill();

        // Enclave perimeter targeting ring
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, 42, 20, time * 0.15, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected
          ? node.themeColor
          : isHovered
          ? "rgba(16, 185, 129, 0.5)"
          : "rgba(16, 185, 129, 0.18)";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash([4, 6]);
        ctx.lineDashOffset = time * 8;
        ctx.stroke();
        ctx.restore();
      });

      // Central server ambient energy pool on floor
      ctx.save();
      const serverPool = ctx.createRadialGradient(serverGround.x, serverGround.y, 6, serverGround.x, serverGround.y, 75);
      serverPool.addColorStop(0, "rgba(16, 185, 129, 0.28)");
      serverPool.addColorStop(0.5, "rgba(6, 182, 212, 0.12)");
      serverPool.addColorStop(1, "rgba(0, 0, 0, 0)");
      ctx.fillStyle = serverPool;
      ctx.beginPath();
      ctx.ellipse(serverGround.x, serverGround.y, 75, 38, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // -------------------------------------------------------------
      // 3. Encrypted Communication Conduits (Curved Laser Lines)
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node, idx) => {
        const isSelected = node.id === selectedNodeId;
        const from = project(node.wx, node.wy, 18, scaleVal, pitchVal, yawVal);
        const to = project(0, 0, 26, scaleVal, pitchVal, yawVal);

        const midX = (from.x + to.x) / 2 + (idx % 2 === 0 ? -14 : 14);
        const midY = (from.y + to.y) / 2 - 16;

        ctx.save();
        // Base conduit track
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.quadraticCurveTo(midX, midY, to.x, to.y);
        ctx.strokeStyle = isSelected || isSimulatingRound
          ? "rgba(16, 185, 129, 0.55)"
          : "rgba(16, 185, 129, 0.2)";
        ctx.lineWidth = isSelected || isSimulatingRound ? 2 : 1.2;
        ctx.stroke();

        // Animated dashed energy pulse
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.quadraticCurveTo(midX, midY, to.x, to.y);
        ctx.strokeStyle = isSelected
          ? "rgba(52, 211, 153, 0.9)"
          : "rgba(16, 185, 129, 0.4)";
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.setLineDash([8, 12]);
        ctx.lineDashOffset = -time * 24;
        ctx.stroke();
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 4. Encrypted Photon Packets (Pure Clean Light Streams)
      // -------------------------------------------------------------
      packets.forEach((p) => {
        const node = DEFAULT_NODES[p.nodeIndex];
        const isSelected = node.id === selectedNodeId;
        const currentSpeed = isSimulatingRound ? p.speed * 3.2 : isSelected ? p.speed * 1.5 : p.speed;
        p.progress = (p.progress + currentSpeed) % 1;

        const u = p.progress;
        const from = project(node.wx, node.wy, 18, scaleVal, pitchVal, yawVal);
        const to = project(0, 0, 26, scaleVal, pitchVal, yawVal);

        const midX = (from.x + to.x) / 2 + (p.nodeIndex % 2 === 0 ? -14 : 14);
        const midY = (from.y + to.y) / 2 - 16;

        const px = (1 - u) * (1 - u) * from.x + 2 * (1 - u) * u * midX + u * u * to.x;
        const py = (1 - u) * (1 - u) * from.y + 2 * (1 - u) * u * midY + u * u * to.y;

        // Maintain speed trail
        p.trail.unshift({ x: px, y: py });
        if (p.trail.length > 5) p.trail.pop();

        ctx.save();
        // Render trail sparks
        p.trail.forEach((pt, ti) => {
          if (ti === 0) return;
          const trailAlpha = (1 - ti / p.trail.length) * 0.45;
          ctx.fillStyle = `rgba(16, 185, 129, ${trailAlpha})`;
          ctx.beginPath();
          ctx.arc(pt.x, pt.y, p.size * (1 - ti / 6), 0, Math.PI * 2);
          ctx.fill();
        });

        // Main luminous photon core
        ctx.shadowColor = node.themeColor;
        ctx.shadowBlur = 12;
        ctx.fillStyle = "#ffffff";
        ctx.beginPath();
        ctx.arc(px, py, p.size * 0.8, 0, Math.PI * 2);
        ctx.fill();

        ctx.fillStyle = node.themeColor;
        ctx.beginPath();
        ctx.arc(px, py, p.size * 1.4, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 5. 3D Hospital Enclave Monoliths (The 4 Institutions)
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node, idx) => {
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNodeRef.current;
        const levitation = isSelected ? Math.sin(time * 3) * 3 + 8 : isHovered ? 4 : Math.sin(time * 2 + idx) * 2;

        const bw = 0.16;
        const bd = 0.14;
        const bh = 34;

        // Architectural Monolith Pedestal
        draw3DBox(
          node.wx,
          node.wy,
          levitation,
          bw,
          bd,
          bh,
          scaleVal,
          pitchVal,
          yawVal,
          {
            top: isSelected ? "rgba(16, 185, 129, 0.35)" : "rgba(30, 41, 59, 0.9)",
            left: isSelected ? "rgba(15, 23, 42, 0.98)" : "rgba(15, 23, 42, 0.92)",
            right: isSelected ? "rgba(8, 12, 20, 0.98)" : "rgba(8, 12, 20, 0.95)",
            stroke: isSelected ? node.themeColor : "rgba(255, 255, 255, 0.18)",
            lineWidth: isSelected ? 2 : 1,
          }
        );

        // Modern horizontal architectural illumination bands
        const band1Left = project(node.wx - bw / 2, node.wy + bd / 2, levitation + 11, scaleVal, pitchVal, yawVal);
        const band1Right = project(node.wx + bw / 2, node.wy + bd / 2, levitation + 11, scaleVal, pitchVal, yawVal);
        const band2Left = project(node.wx - bw / 2, node.wy + bd / 2, levitation + 22, scaleVal, pitchVal, yawVal);
        const band2Right = project(node.wx + bw / 2, node.wy + bd / 2, levitation + 22, scaleVal, pitchVal, yawVal);

        ctx.save();
        ctx.strokeStyle = isSelected ? node.themeColor : "rgba(16, 185, 129, 0.4)";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(band1Left.x, band1Left.y);
        ctx.lineTo(band1Right.x, band1Right.y);
        ctx.moveTo(band2Left.x, band2Left.y);
        ctx.lineTo(band2Right.x, band2Right.y);
        ctx.stroke();

        // Medical Cross Hologram on Monolith Face
        const crossPos = project(node.wx, node.wy + bd / 2 + 0.005, levitation + 16, scaleVal, pitchVal, yawVal);
        ctx.strokeStyle = isSelected ? "#ffffff" : node.themeColor;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(crossPos.x - 4, crossPos.y);
        ctx.lineTo(crossPos.x + 4, crossPos.y);
        ctx.moveTo(crossPos.x, crossPos.y - 4);
        ctx.lineTo(crossPos.x, crossPos.y + 4);
        ctx.stroke();

        // Rooftop Beacon Crystal & Laser Tether
        const roofPeak = project(node.wx, node.wy, levitation + bh + 10, scaleVal, pitchVal, yawVal);
        const roofBase = project(node.wx, node.wy, levitation + bh, scaleVal, pitchVal, yawVal);

        ctx.strokeStyle = node.themeColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(roofBase.x, roofBase.y);
        ctx.lineTo(roofPeak.x, roofPeak.y);
        ctx.stroke();

        // Radiant status flare
        ctx.fillStyle = node.themeColor;
        ctx.shadowColor = node.themeColor;
        ctx.shadowBlur = isSelected ? 16 : 8;
        ctx.beginPath();
        ctx.arc(roofPeak.x, roofPeak.y, isSelected ? 4.5 : 3.2, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Built-in Crisp Institutional Label (Directly below pedestal)
        const groundPos = project(node.wx, node.wy, 0, scaleVal, pitchVal, yawVal);
        ctx.save();
        ctx.textAlign = "center";
        ctx.font = 'bold 11px "JetBrains Mono", monospace';
        ctx.fillStyle = isSelected ? "#ffffff" : "rgba(255, 255, 255, 0.85)";
        ctx.fillText(node.shortName, groundPos.x, groundPos.y + 24);

        ctx.font = '9px "Inter", sans-serif';
        ctx.fillStyle = isSelected ? node.themeColor : "rgba(148, 163, 184, 0.75)";
        ctx.fillText(`${node.location.split(",")[0]} · ${node.modality}`, groundPos.x, groundPos.y + 36);

        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = "rgba(16, 185, 129, 0.85)";
        ctx.fillText(`ε=${node.dpEpsilon}`, groundPos.x, groundPos.y + 48);
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 6. Central Aggregator Server (The Cryptographic Server Core)
      // -------------------------------------------------------------
      const serverLev = Math.sin(time * 2.4) * 2;
      const sw = 0.20;
      const sd = 0.18;

      // Base Foundation Pedestal
      draw3DBox(0, 0, serverLev, sw * 1.35, sd * 1.35, 14, scaleVal, pitchVal, yawVal, {
        top: "#1e293b",
        left: "#0f172a",
        right: "#080c14",
        stroke: "rgba(16, 185, 129, 0.4)",
        lineWidth: 1.5,
      });

      // Mainframe Quantum Server Tower
      draw3DBox(0, 0, serverLev + 14, sw, sd, 48, scaleVal, pitchVal, yawVal, {
        top: "rgba(16, 185, 129, 0.35)",
        left: "#0f172a",
        right: "#06090f",
        stroke: "#10b981",
        lineWidth: 1.8,
      });

      // Animated Server Blade LEDs
      ctx.save();
      for (let rack = 0; rack < 3; rack++) {
        const rz = serverLev + 20 + rack * 12;
        const ledPos = project(0, sd / 2 + 0.005, rz, scaleVal, pitchVal, yawVal);
        const isBlinking = Math.floor(time * 5 + rack) % 2 === 0;

        ctx.fillStyle = isBlinking ? "#10b981" : "#06b6d4";
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 6;
        ctx.beginPath();
        ctx.arc(ledPos.x - 6, ledPos.y, 1.6, 0, Math.PI * 2);
        ctx.arc(ledPos.x, ledPos.y, 1.6, 0, Math.PI * 2);
        ctx.arc(ledPos.x + 6, ledPos.y, 1.6, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Floating Cryptographic Core (CKKS 128-Bit Enclave)
      const coreZ = serverLev + 72;
      const corePos = project(0, 0, coreZ, scaleVal, pitchVal, yawVal);

      ctx.save();
      // Glowing Cryptographic Energy Orb
      const corePulse = 0.8 + Math.sin(time * 3.5) * 0.2;
      const coreGrad = ctx.createRadialGradient(corePos.x, corePos.y, 3, corePos.x, corePos.y, 32 * corePulse);
      coreGrad.addColorStop(0, "rgba(255, 255, 255, 0.98)");
      coreGrad.addColorStop(0.3, "rgba(16, 185, 129, 0.85)");
      coreGrad.addColorStop(0.7, "rgba(6, 182, 212, 0.35)");
      coreGrad.addColorStop(1, "rgba(16, 185, 129, 0)");
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(corePos.x, corePos.y, 32 * corePulse, 0, Math.PI * 2);
      ctx.fill();

      // Rotating Cryptographic Orbit Ring 1 (RLWE CKKS)
      ctx.beginPath();
      ctx.ellipse(corePos.x, corePos.y, 44, 20, time * 0.7, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.6)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 7]);
      ctx.lineDashOffset = time * 16;
      ctx.stroke();

      // Rotating Cryptographic Orbit Ring 2 (FedOpt Momentum)
      ctx.beginPath();
      ctx.ellipse(corePos.x, corePos.y, 32, 15, -time * 1.0, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.7)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 5]);
      ctx.lineDashOffset = -time * 12;
      ctx.stroke();
      ctx.restore();

      // Built-in Crisp Server Core Label (Directly below server pedestal)
      ctx.save();
      ctx.textAlign = "center";
      ctx.font = 'bold 11px "JetBrains Mono", monospace';
      ctx.fillStyle = "#ffffff";
      ctx.fillText("CONSORTIUM AGGREGATOR", serverGround.x, serverGround.y + 24);

      ctx.font = '9px "Inter", sans-serif';
      ctx.fillStyle = "rgba(56, 189, 248, 0.85)";
      ctx.fillText("RLWE CKKS 128-Bit · Zero Plaintext", serverGround.x, serverGround.y + 36);
      ctx.restore();

      // -------------------------------------------------------------
      // 7. Consensus Shockwave Expansion (Triggered Round)
      // -------------------------------------------------------------
      if (isShockwaveActive) {
        shockwaveRadius += 6.5;
        const maxRadius = Math.max(width, height) * 0.72;
        const alpha = Math.max(0, 1 - shockwaveRadius / maxRadius);

        ctx.save();
        ctx.beginPath();
        ctx.ellipse(serverGround.x, serverGround.y, shockwaveRadius, shockwaveRadius * 0.52, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.9})`;
        ctx.lineWidth = 3;
        ctx.stroke();

        if (shockwaveRadius > 40) {
          ctx.beginPath();
          ctx.ellipse(serverGround.x, serverGround.y, shockwaveRadius - 32, (shockwaveRadius - 32) * 0.52, 0, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(6, 182, 212, ${alpha * 0.65})`;
          ctx.lineWidth = 1.8;
          ctx.stroke();
        }
        ctx.restore();

        if (shockwaveRadius >= maxRadius) {
          isShockwaveActive = false;
          if (onSimulationComplete) {
            onSimulationComplete();
          }
        }
      }

      animFrameRef.current = requestAnimationFrame(render);
    };

    render();

    // Canvas click & hover selection detection
    const handleCanvasClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const scaleVal = Math.min(width * 0.42, height * 0.54);
      const pitchVal = 0.56 + currentTiltRef.current.y * 0.14;
      const yawVal = currentTiltRef.current.x * 0.22;

      for (const node of DEFAULT_NODES) {
        const p = project(node.wx, node.wy, 18, scaleVal, pitchVal, yawVal);
        const dist = Math.hypot(mx - p.x, my - p.y);
        if (dist < 50) {
          if (onSelectNode) onSelectNode(node.id);
          return;
        }
      }
    };

    const handleCanvasMouseMove = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const scaleVal = Math.min(width * 0.42, height * 0.54);
      const pitchVal = 0.56 + currentTiltRef.current.y * 0.14;
      const yawVal = currentTiltRef.current.x * 0.22;

      let found: string | null = null;
      for (const node of DEFAULT_NODES) {
        const p = project(node.wx, node.wy, 18, scaleVal, pitchVal, yawVal);
        const dist = Math.hypot(mx - p.x, my - p.y);
        if (dist < 50) {
          found = node.id;
          break;
        }
      }

      hoveredNodeRef.current = found;
      canvas.style.cursor = found ? "pointer" : "default";
    };

    canvas.addEventListener("click", handleCanvasClick);
    canvas.addEventListener("mousemove", handleCanvasMouseMove);

    return () => {
      window.removeEventListener("resize", handleResize);
      canvas.removeEventListener("click", handleCanvasClick);
      canvas.removeEventListener("mousemove", handleCanvasMouseMove);
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isSimulatingRound, selectedNodeId, mouseTilt, onSelectNode, onSimulationComplete]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full select-none z-0"
    />
  );
}
