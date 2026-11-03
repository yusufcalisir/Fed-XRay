"use client";

import React, { useEffect, useRef } from "react";

export interface Hospital3DNode {
  id: string;
  name: string;
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
  onUpdateProjectedPositions?: (
    nodes: { id: string; x: number; y: number; topY: number }[],
    server: { x: number; y: number; topY: number }
  ) => void;
}

interface CipherPacket {
  tunnelIndex: number;
  progress: number;
  speed: number;
  label: string;
  size: number;
}

const DEFAULT_NODES: Hospital3DNode[] = [
  {
    id: "node-1",
    name: "BCN-20000 Skin Cancer Hub",
    location: "Barcelona, ES",
    cityCode: "BCN",
    modality: "Dermoscopy",
    dpEpsilon: 1.42,
    scans: 40,
    wx: -0.62,
    wy: -0.48,
    themeColor: "#10b981",
    glowColor: "rgba(16, 185, 129, 0.45)",
  },
  {
    id: "node-3",
    name: "ViDIR Dermatopathology Institute",
    location: "Vienna, AT",
    cityCode: "VIE",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    scans: 40,
    wx: 0.62,
    wy: -0.48,
    themeColor: "#06b6d4",
    glowColor: "rgba(6, 182, 212, 0.45)",
  },
  {
    id: "node-4",
    name: "Queensland Oncology Center",
    location: "Brisbane, AU",
    cityCode: "BNE",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    scans: 40,
    wx: -0.62,
    wy: 0.48,
    themeColor: "#10b981",
    glowColor: "rgba(16, 185, 129, 0.45)",
  },
  {
    id: "node-2",
    name: "Beth Israel Deaconess Medical Center",
    location: "Boston, USA",
    cityCode: "BOS",
    modality: "Chest Radiography",
    dpEpsilon: 1.38,
    scans: 40,
    wx: 0.62,
    wy: 0.48,
    themeColor: "#38bdf8",
    glowColor: "rgba(56, 189, 248, 0.45)",
  },
];

const PACKET_LABELS = [
  ["Enc(B_1)", "[c0, c1]", "RLWE 128b", "eps=1.42"],
  ["Enc(B_3)", "CKKS-Poly", "Zero-Plain", "eps=1.55"],
  ["Enc(B_4)", "[c0, c1]", "FFA-LoRA", "eps=1.48"],
  ["Enc(B_2)", "SecAgg+", "RLWE 128b", "eps=1.38"],
];

export default function Consortium3DCanvas({
  isSimulatingRound = false,
  selectedNodeId = "node-1",
  mouseTilt = { x: 0, y: 0 },
  onSelectNode,
  onSimulationComplete,
  onUpdateProjectedPositions,
}: Consortium3DCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);
  const hoveredNodeRef = useRef<string | null>(null);

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
      return; // Safe guard for headless test environments
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

    // Initialize 3D Ciphertext flow packets
    const packets: CipherPacket[] = [];
    for (let t = 0; t < 4; t++) {
      for (let p = 0; p < 4; p++) {
        packets.push({
          tunnelIndex: t,
          progress: (p * 0.25 + t * 0.1) % 1,
          speed: 0.0035 + Math.random() * 0.0015,
          label: PACKET_LABELS[t][p % PACKET_LABELS[t].length],
          size: 3.5 + (p % 2),
        });
      }
    }

    if (isSimulatingRound) {
      shockwaveRadius = 15;
      isShockwaveActive = true;
    }

    // -------------------------------------------------------------
    // 3D Projection Mathematics with Isometric Tilt
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
      const cy = height / 2 + 10;

      // Yaw rotation (horizontal azimuth)
      const cosY = Math.cos(yawVal);
      const sinY = Math.sin(yawVal);
      const rx = wx * cosY - wy * sinY;
      const ry = wx * sinY + wy * cosY;

      // Pitch rotation (elevation angle)
      const cosP = Math.cos(pitchVal);
      const sinP = Math.sin(pitchVal);

      const screenX = cx + rx * scaleVal;
      const screenY = cy + ry * scaleVal * sinP - wz * cosP;
      const depth = ry * cosP + wz * sinP;

      return { x: screenX, y: screenY, depth };
    };

    // Helper: Draw 3D Isometric Extruded Box
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
      style: {
        topFill: string;
        leftFill: string;
        rightFill: string;
        stroke: string;
        lineWidth?: number;
      }
    ) => {
      // 8 Vertices
      const p0 = project(bx - bw / 2, by - bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p1 = project(bx + bw / 2, by - bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p2 = project(bx + bw / 2, by + bd / 2, bz, scaleVal, pitchVal, yawVal);
      const p3 = project(bx - bw / 2, by + bd / 2, bz, scaleVal, pitchVal, yawVal);

      const p4 = project(bx - bw / 2, by - bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p5 = project(bx + bw / 2, by - bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p6 = project(bx + bw / 2, by + bd / 2, bz + bh, scaleVal, pitchVal, yawVal);
      const p7 = project(bx - bw / 2, by + bd / 2, bz + bh, scaleVal, pitchVal, yawVal);

      ctx.save();
      ctx.lineWidth = style.lineWidth || 1;
      ctx.strokeStyle = style.stroke;

      // Left Wall Face
      ctx.beginPath();
      ctx.moveTo(p3.x, p3.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p7.x, p7.y);
      ctx.closePath();
      ctx.fillStyle = style.leftFill;
      ctx.fill();
      ctx.stroke();

      // Right Wall Face
      ctx.beginPath();
      ctx.moveTo(p1.x, p1.y);
      ctx.lineTo(p2.x, p2.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p5.x, p5.y);
      ctx.closePath();
      ctx.fillStyle = style.rightFill;
      ctx.fill();
      ctx.stroke();

      // Top Roof Face
      ctx.beginPath();
      ctx.moveTo(p4.x, p4.y);
      ctx.lineTo(p5.x, p5.y);
      ctx.lineTo(p6.x, p6.y);
      ctx.lineTo(p7.x, p7.y);
      ctx.closePath();
      ctx.fillStyle = style.topFill;
      ctx.fill();
      ctx.stroke();

      ctx.restore();
      return { roofCenter: { x: (p4.x + p6.x) / 2, y: (p4.y + p6.y) / 2 } };
    };

    // -------------------------------------------------------------
    // Main Render Loop (60 FPS)
    // -------------------------------------------------------------
    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2 + 10;
      const scaleVal = Math.min(width * 0.44, height * 0.58);
      const pitchVal = 0.52 + mouseTilt.y * 0.18;
      const yawVal = mouseTilt.x * 0.26;

      // -------------------------------------------------------------
      // 1. 3D Perspective Cyber Floor Grid
      // -------------------------------------------------------------
      ctx.save();
      ctx.strokeStyle = "rgba(16, 185, 129, 0.06)";
      ctx.lineWidth = 1;

      const gridRange = 1.1;
      const gridStep = 0.22;
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

      // Project Central Server position
      const serverGround = project(0, 0, 0, scaleVal, pitchVal, yawVal);
      const serverTop = project(0, 0, 95, scaleVal, pitchVal, yawVal);

      // Project 4 Hospital anchor positions
      const projectedNodes = DEFAULT_NODES.map((node) => {
        const ground = project(node.wx, node.wy, 0, scaleVal, pitchVal, yawVal);
        const top = project(node.wx, node.wy, 55, scaleVal, pitchVal, yawVal);
        return {
          id: node.id,
          x: ground.x,
          y: ground.y,
          topY: top.y,
        };
      });

      // Notify parent component of exact 2D coordinates for crisp HUD positioning
      if (onUpdateProjectedPositions && Math.floor(time * 50) % 6 === 0) {
        onUpdateProjectedPositions(projectedNodes, {
          x: serverGround.x,
          y: serverGround.y,
          topY: serverTop.y,
        });
      }

      // -------------------------------------------------------------
      // 2. Pedestal Anchor Rings & Soft Contact Shadows
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node) => {
        const p = project(node.wx, node.wy, 0, scaleVal, pitchVal, yawVal);
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNodeRef.current;

        ctx.save();
        // Ground Contact Shadow
        const shadowGrad = ctx.createRadialGradient(p.x, p.y, 2, p.x, p.y, 45);
        shadowGrad.addColorStop(0, "rgba(0, 0, 0, 0.7)");
        shadowGrad.addColorStop(1, "rgba(0, 0, 0, 0)");
        ctx.fillStyle = shadowGrad;
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, 48, 26, 0, 0, Math.PI * 2);
        ctx.fill();

        // Target Pedestal Ring
        ctx.beginPath();
        ctx.ellipse(p.x, p.y, 44, 24, time * 0.2, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected || isHovered
          ? node.themeColor
          : "rgba(16, 185, 129, 0.22)";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash([4, 6]);
        ctx.lineDashOffset = time * 6;
        ctx.stroke();

        // Engraved City Code
        ctx.font = '9px "JetBrains Mono", monospace';
        ctx.fillStyle = isSelected ? node.themeColor : "rgba(255, 255, 255, 0.4)";
        ctx.textAlign = "center";
        ctx.fillText(node.cityCode, p.x, p.y + 22);
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 3. 3D Transmission Conduits (Laser Beams)
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node, idx) => {
        const isSelected = node.id === selectedNodeId;
        const from = project(node.wx, node.wy, 24, scaleVal, pitchVal, yawVal);
        const to = project(0, 0, 32, scaleVal, pitchVal, yawVal);

        const midX = (from.x + to.x) / 2 + (idx % 2 === 0 ? -18 : 18);
        const midY = (from.y + to.y) / 2 - 20;

        ctx.save();
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.quadraticCurveTo(midX, midY, to.x, to.y);

        ctx.strokeStyle = isSelected || isSimulatingRound
          ? "rgba(16, 185, 129, 0.65)"
          : "rgba(16, 185, 129, 0.22)";
        ctx.lineWidth = isSelected || isSimulatingRound ? 2.5 : 1.2;
        ctx.setLineDash([6, 6]);
        ctx.lineDashOffset = -time * 16;
        ctx.stroke();

        if (isSelected || isSimulatingRound) {
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.quadraticCurveTo(midX, midY, to.x, to.y);
          ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
          ctx.lineWidth = 4;
          ctx.stroke();
        }
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 4. 3D Ciphertext Flowing Packets
      // -------------------------------------------------------------
      packets.forEach((p) => {
        const node = DEFAULT_NODES[p.tunnelIndex];
        const currentSpeed = isSimulatingRound ? p.speed * 3.5 : p.speed;
        p.progress = (p.progress + currentSpeed) % 1;

        const u = p.progress;
        const from = project(node.wx, node.wy, 24, scaleVal, pitchVal, yawVal);
        const to = project(0, 0, 32, scaleVal, pitchVal, yawVal);

        const midX = (from.x + to.x) / 2 + (p.tunnelIndex % 2 === 0 ? -18 : 18);
        const midY = (from.y + to.y) / 2 - 20;

        // Quadratic Bezier interpolation
        const px = (1 - u) * (1 - u) * from.x + 2 * (1 - u) * u * midX + u * u * to.x;
        const py = (1 - u) * (1 - u) * from.y + 2 * (1 - u) * u * midY + u * u * to.y;

        ctx.save();
        const scaleDepth = 0.8 + u * 0.4;
        ctx.fillStyle = node.themeColor;
        ctx.shadowColor = node.themeColor;
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(px, py, p.size * scaleDepth, 0, Math.PI * 2);
        ctx.fill();

        // Cipher label
        if (u > 0.15 && u < 0.85) {
          ctx.shadowBlur = 0;
          ctx.font = `${Math.floor(8.5 * scaleDepth)}px "JetBrains Mono", monospace`;
          ctx.fillStyle = "rgba(255, 255, 255, 0.85)";
          ctx.fillText(p.label, px + 8, py - 4);
        }
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 5. Render the 4 3D Hospital Nodes (Institutions)
      // -------------------------------------------------------------
      DEFAULT_NODES.forEach((node) => {
        const isSelected = node.id === selectedNodeId;
        const isHovered = node.id === hoveredNodeRef.current;
        const levitation = isSelected ? Math.sin(time * 3) * 3 + 6 : isHovered ? 4 : 0;

        const bw = 0.16;
        const bd = 0.14;
        const bh = 38;

        // Base 3D Hospital Wing
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
            topFill: isSelected ? "rgba(16, 185, 129, 0.28)" : "#1e293b",
            leftFill: isSelected ? "rgba(15, 23, 42, 0.95)" : "#0f172a",
            rightFill: isSelected ? "rgba(9, 13, 22, 0.95)" : "#080c14",
            stroke: isSelected ? node.themeColor : "rgba(255, 255, 255, 0.15)",
            lineWidth: isSelected ? 2 : 1,
          }
        );

        // Illuminated Diagnostic Floor Bands on Front Wall
        const pLeftMid1 = project(node.wx - bw / 2, node.wy + bd / 2, levitation + 12, scaleVal, pitchVal, yawVal);
        const pRightMid1 = project(node.wx + bw / 2, node.wy + bd / 2, levitation + 12, scaleVal, pitchVal, yawVal);
        const pLeftMid2 = project(node.wx - bw / 2, node.wy + bd / 2, levitation + 24, scaleVal, pitchVal, yawVal);
        const pRightMid2 = project(node.wx + bw / 2, node.wy + bd / 2, levitation + 24, scaleVal, pitchVal, yawVal);

        ctx.save();
        ctx.strokeStyle = node.themeColor;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(pLeftMid1.x, pLeftMid1.y);
        ctx.lineTo(pRightMid1.x, pRightMid1.y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(pLeftMid2.x, pLeftMid2.y);
        ctx.lineTo(pRightMid2.x, pRightMid2.y);
        ctx.stroke();

        // Medical Cross Emblem on Front Wall
        const crossCenter = project(node.wx, node.wy + bd / 2 + 0.005, levitation + 18, scaleVal, pitchVal, yawVal);
        ctx.strokeStyle = isSelected ? "#ffffff" : node.themeColor;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.moveTo(crossCenter.x - 5, crossCenter.y);
        ctx.lineTo(crossCenter.x + 5, crossCenter.y);
        ctx.moveTo(crossCenter.x, crossCenter.y - 5);
        ctx.lineTo(crossCenter.x, crossCenter.y + 5);
        ctx.stroke();

        // Rooftop Architecture & Beacon
        const roofPeak = project(node.wx, node.wy, levitation + bh + 12, scaleVal, pitchVal, yawVal);
        ctx.fillStyle = node.themeColor;
        ctx.shadowColor = node.themeColor;
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(roofPeak.x, roofPeak.y, isSelected ? 4.5 : 3.5, 0, Math.PI * 2);
        ctx.fill();

        // Beacon Laser Tether
        const roofCenter = project(node.wx, node.wy, levitation + bh, scaleVal, pitchVal, yawVal);
        ctx.strokeStyle = node.themeColor;
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 2]);
        ctx.beginPath();
        ctx.moveTo(roofCenter.x, roofCenter.y);
        ctx.lineTo(roofPeak.x, roofPeak.y);
        ctx.stroke();
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 6. Render the 3D Central Server Core (in the middle!)
      // -------------------------------------------------------------
      const serverLev = Math.sin(time * 2.5) * 2;
      const sw = 0.20;
      const sd = 0.18;

      // Tier 1: Foundation Server Pedestal
      draw3DBox(0, 0, serverLev, sw * 1.3, sd * 1.3, 16, scaleVal, pitchVal, yawVal, {
        topFill: "#1e293b",
        leftFill: "#0f172a",
        rightFill: "#0a0e17",
        stroke: "rgba(16, 185, 129, 0.4)",
        lineWidth: 1.5,
      });

      // Tier 2: 3D Server Mainframe Tower
      draw3DBox(0, 0, serverLev + 16, sw, sd, 54, scaleVal, pitchVal, yawVal, {
        topFill: "rgba(16, 185, 129, 0.35)",
        leftFill: "#0f172a",
        rightFill: "#070b12",
        stroke: "#10b981",
        lineWidth: 2,
      });

      // Blinking Server Blade LEDs on Mainframe Face
      ctx.save();
      for (let rack = 0; rack < 3; rack++) {
        const rz = serverLev + 22 + rack * 14;
        const ledPos = project(0, sd / 2 + 0.005, rz, scaleVal, pitchVal, yawVal);
        const isBlinking = Math.floor(time * 6 + rack) % 2 === 0;

        ctx.fillStyle = isBlinking ? "#10b981" : "#06b6d4";
        ctx.shadowColor = ctx.fillStyle;
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(ledPos.x - 7, ledPos.y, 1.8, 0, Math.PI * 2);
        ctx.arc(ledPos.x, ledPos.y, 1.8, 0, Math.PI * 2);
        ctx.arc(ledPos.x + 7, ledPos.y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Tier 3: Cryptographic Chamber & Floating CKKS Core
      const coreZ = serverLev + 80;
      const corePos = project(0, 0, coreZ, scaleVal, pitchVal, yawVal);

      ctx.save();
      // Glowing Energy Core
      const coreGrad = ctx.createRadialGradient(corePos.x, corePos.y, 4, corePos.x, corePos.y, 35);
      coreGrad.addColorStop(0, "rgba(255, 255, 255, 0.95)");
      coreGrad.addColorStop(0.3, "rgba(16, 185, 129, 0.8)");
      coreGrad.addColorStop(0.7, "rgba(6, 182, 212, 0.3)");
      coreGrad.addColorStop(1, "rgba(16, 185, 129, 0)");
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(corePos.x, corePos.y, 35, 0, Math.PI * 2);
      ctx.fill();

      // Rotating Orbital Ring 1 (CKKS 128-Bit)
      ctx.beginPath();
      ctx.ellipse(corePos.x, corePos.y, 48, 22, time * 0.8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.55)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 8]);
      ctx.lineDashOffset = time * 18;
      ctx.stroke();

      // Rotating Orbital Ring 2 (FedOpt Momentum)
      ctx.beginPath();
      ctx.ellipse(corePos.x, corePos.y, 36, 16, -time * 1.1, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.65)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 6]);
      ctx.lineDashOffset = -time * 14;
      ctx.stroke();
      ctx.restore();

      // -------------------------------------------------------------
      // 7. Consensus Shockwave Expansion (Triggered Round)
      // -------------------------------------------------------------
      if (isShockwaveActive) {
        shockwaveRadius += 6;
        const maxRadius = Math.max(width, height) * 0.75;
        const alpha = Math.max(0, 1 - shockwaveRadius / maxRadius);

        ctx.save();
        ctx.beginPath();
        ctx.ellipse(serverGround.x, serverGround.y, shockwaveRadius, shockwaveRadius * 0.55, 0, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.85})`;
        ctx.lineWidth = 3.5;
        ctx.stroke();

        if (shockwaveRadius > 45) {
          ctx.beginPath();
          ctx.ellipse(serverGround.x, serverGround.y, shockwaveRadius - 35, (shockwaveRadius - 35) * 0.55, 0, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(6, 182, 212, ${alpha * 0.6})`;
          ctx.lineWidth = 2;
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

    // Canvas click & hover detection
    const handleCanvasClick = (e: MouseEvent) => {
      const rect = canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      const scaleVal = Math.min(width * 0.44, height * 0.58);
      const pitchVal = 0.52 + mouseTilt.y * 0.18;
      const yawVal = mouseTilt.x * 0.26;

      for (const node of DEFAULT_NODES) {
        const p = project(node.wx, node.wy, 18, scaleVal, pitchVal, yawVal);
        const dist = Math.hypot(mx - p.x, my - p.y);
        if (dist < 46) {
          if (onSelectNode) onSelectNode(node.id);
          return;
        }
      }
    };

    canvas.addEventListener("click", handleCanvasClick);

    return () => {
      window.removeEventListener("resize", handleResize);
      canvas.removeEventListener("click", handleCanvasClick);
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isSimulatingRound, selectedNodeId, mouseTilt, onSelectNode, onSimulationComplete, onUpdateProjectedPositions]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full select-none cursor-pointer z-0"
    />
  );
}
