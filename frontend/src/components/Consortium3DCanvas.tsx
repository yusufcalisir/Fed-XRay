"use client";

import React, { useEffect, useRef } from "react";

interface Consortium3DCanvasProps {
  isSimulatingRound?: boolean;
  selectedNodeId?: string;
  mouseTilt?: { x: number; y: number };
  onSimulationComplete?: () => void;
}

interface Particle {
  tunnelIndex: number;
  progress: number;
  speed: number;
  label: string;
  size: number;
}

const TUNNEL_LABELS = [
  ["Enc(B_1)", "[c0, c1]", "RLWE 128b", "eps=1.42"],
  ["Enc(B_2)", "CKKS-Poly", "Zero-Plain", "eps=1.38"],
  ["Enc(B_3)", "[c0, c1]", "FFA-LoRA", "eps=1.55"],
  ["Enc(B_4)", "SecAgg+", "RLWE 128b", "eps=1.48"],
];

export default function Consortium3DCanvas({
  isSimulatingRound = false,
  selectedNodeId = "node-1",
  mouseTilt = { x: 0, y: 0 },
  onSimulationComplete,
}: Consortium3DCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const animFrameRef = useRef<number | null>(null);

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
      return; // Graceful fallback in headless / test environments
    }

    let width = 0;
    let height = 0;
    let time = 0;
    let shockwaveRadius = 0;
    let isShockwaveActive = false;

    // Responsive canvas resizing
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

    // Initialize flowing ciphertext particles
    const particles: Particle[] = [];
    for (let t = 0; t < 4; t++) {
      for (let p = 0; p < 4; p++) {
        particles.push({
          tunnelIndex: t,
          progress: (p * 0.25 + t * 0.08) % 1,
          speed: 0.004 + Math.random() * 0.0015,
          label: TUNNEL_LABELS[t][p % TUNNEL_LABELS[t].length],
          size: 3.5 + (p % 2),
        });
      }
    }

    // Trigger shockwave on simulation round
    if (isSimulatingRound) {
      shockwaveRadius = 15;
      isShockwaveActive = true;
    }

    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      // 3D Tilt Vanishing Point
      const tiltOffsetX = mouseTilt.x * 28;
      const tiltOffsetY = mouseTilt.y * 20;

      // Anchor positions corresponding to the 4 corner pods
      const nodes = [
        { x: width * 0.17, y: height * 0.16, id: "node-1" }, // Top-Left (Barcelona)
        { x: width * 0.83, y: height * 0.16, id: "node-3" }, // Top-Right (Vienna)
        { x: width * 0.17, y: height * 0.84, id: "node-4" }, // Bottom-Left (Brisbane)
        { x: width * 0.83, y: height * 0.84, id: "node-2" }, // Bottom-Right (Boston)
      ];

      const coreX = cx + tiltOffsetX;
      const coreY = cy + tiltOffsetY;

      // -------------------------------------------------------------
      // 1. 3D Isometric Perspective Floor Grid
      // -------------------------------------------------------------
      ctx.save();
      ctx.strokeStyle = "rgba(16, 185, 129, 0.05)";
      ctx.lineWidth = 1;
      const gridCount = 9;
      for (let i = -gridCount; i <= gridCount; i++) {
        const gx = cx + i * (width / (gridCount * 2.2)) + tiltOffsetX * 0.6;
        ctx.beginPath();
        ctx.moveTo(gx, 0);
        ctx.lineTo(gx + i * 16, height);
        ctx.stroke();
      }
      for (let j = 0; j <= 7; j++) {
        const gy = height * 0.15 + j * (height * 0.12);
        ctx.beginPath();
        ctx.moveTo(0, gy);
        ctx.lineTo(width, gy);
        ctx.stroke();
      }
      ctx.restore();

      // -------------------------------------------------------------
      // 2. 3D Pedestal Rings beneath each Hospital Node
      // -------------------------------------------------------------
      nodes.forEach((node) => {
        const isSelected = node.id === selectedNodeId;
        ctx.save();
        ctx.beginPath();
        ctx.ellipse(node.x, node.y, 44, 28, time * 0.3, 0, Math.PI * 2);
        ctx.strokeStyle = isSelected
          ? "rgba(16, 185, 129, 0.45)"
          : "rgba(16, 185, 129, 0.15)";
        ctx.lineWidth = isSelected ? 2 : 1;
        ctx.setLineDash([4, 6]);
        ctx.lineDashOffset = time * 8;
        ctx.stroke();

        // Glowing center anchor
        ctx.fillStyle = isSelected ? "#10b981" : "rgba(16, 185, 129, 0.4)";
        ctx.beginPath();
        ctx.arc(node.x, node.y, 3, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      // -------------------------------------------------------------
      // 3. Inter-Hospital Byzantine Verification Mesh (Elliptical Orbit)
      // -------------------------------------------------------------
      ctx.save();
      const ringRadiusX = width * 0.38;
      const ringRadiusY = height * 0.35;
      ctx.beginPath();
      ctx.ellipse(cx + tiltOffsetX * 0.5, cy + tiltOffsetY * 0.5, ringRadiusX, ringRadiusY, 0, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.22)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 8]);
      ctx.lineDashOffset = -time * 14;
      ctx.stroke();

      // Floating Byzantine tokens along the outer ring
      const meshAngle = time * 0.5;
      for (let m = 0; m < 4; m++) {
        const ma = meshAngle + (m * Math.PI) / 2;
        const mx = cx + tiltOffsetX * 0.5 + Math.cos(ma) * ringRadiusX;
        const my = cy + tiltOffsetY * 0.5 + Math.sin(ma) * ringRadiusY;
        ctx.fillStyle = "#06b6d4";
        ctx.shadowColor = "#06b6d4";
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(mx, my, 3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // -------------------------------------------------------------
      // 4. Central Aggregator 3D Orbital Rings & Glow Beacon
      // -------------------------------------------------------------
      ctx.save();
      // Outer 3D revolving ring
      ctx.beginPath();
      ctx.ellipse(coreX, coreY, 78, 56, time * 0.6, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.35)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 10]);
      ctx.lineDashOffset = time * 20;
      ctx.stroke();

      // Inner 3D counter-revolving ring
      ctx.beginPath();
      ctx.ellipse(coreX, coreY, 60, 42, -time * 0.8, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([6, 6]);
      ctx.lineDashOffset = -time * 16;
      ctx.stroke();

      // Radial Core Ambient Glow
      const coreGrad = ctx.createRadialGradient(coreX, coreY, 10, coreX, coreY, 75);
      coreGrad.addColorStop(0, "rgba(16, 185, 129, 0.3)");
      coreGrad.addColorStop(0.5, "rgba(6, 182, 212, 0.12)");
      coreGrad.addColorStop(1, "rgba(16, 185, 129, 0)");
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(coreX, coreY, 75, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // -------------------------------------------------------------
      // 5. 3D Transmission Tunnels (Curved Energy Beams)
      // -------------------------------------------------------------
      nodes.forEach((node, idx) => {
        const isSelected = node.id === selectedNodeId;
        ctx.save();

        const midX = (node.x + coreX) / 2 + (idx % 2 === 0 ? -20 : 20);
        const midY = (node.y + coreY) / 2 + (idx < 2 ? -15 : 15);

        // Core conduit line
        ctx.beginPath();
        ctx.moveTo(node.x, node.y);
        ctx.quadraticCurveTo(midX, midY, coreX, coreY);
        ctx.strokeStyle = isSelected
          ? "rgba(16, 185, 129, 0.6)"
          : "rgba(16, 185, 129, 0.22)";
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.setLineDash([6, 6]);
        ctx.lineDashOffset = -time * 16;
        ctx.stroke();

        // Neon Glow on active tunnels
        if (isSelected || isSimulatingRound) {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.quadraticCurveTo(midX, midY, coreX, coreY);
          ctx.strokeStyle = "rgba(6, 182, 212, 0.45)";
          ctx.lineWidth = 4;
          ctx.stroke();
        }

        ctx.restore();
      });

      // -------------------------------------------------------------
      // 6. 3D Encrypted Ciphertext Packets (Stream Animation)
      // -------------------------------------------------------------
      particles.forEach((p) => {
        const node = nodes[p.tunnelIndex];
        const currentSpeed = isSimulatingRound ? p.speed * 3.2 : p.speed;
        p.progress = (p.progress + currentSpeed) % 1;

        const u = p.progress;
        const midX = (node.x + coreX) / 2 + (p.tunnelIndex % 2 === 0 ? -20 : 20);
        const midY = (node.y + coreY) / 2 + (p.tunnelIndex < 2 ? -15 : 15);

        // Quadratic Bezier interpolation
        const px = (1 - u) * (1 - u) * node.x + 2 * (1 - u) * u * midX + u * u * coreX;
        const py = (1 - u) * (1 - u) * node.y + 2 * (1 - u) * u * midY + u * u * coreY;

        ctx.save();

        // 3D depth scaling: particles grow slightly as they converge towards the center
        const scale = 0.85 + u * 0.4;

        // Glowing packet orb
        ctx.fillStyle = "#10b981";
        ctx.shadowColor = "#10b981";
        ctx.shadowBlur = 12;
        ctx.beginPath();
        ctx.arc(px, py, p.size * scale, 0, Math.PI * 2);
        ctx.fill();

        // Trailing phosphorescent tail
        const trailU = Math.max(0, u - 0.05);
        const tx = (1 - trailU) * (1 - trailU) * node.x + 2 * (1 - trailU) * trailU * midX + trailU * trailU * coreX;
        const ty = (1 - trailU) * (1 - trailU) * node.y + 2 * (1 - trailU) * trailU * midY + trailU * trailU * coreY;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(tx, ty);
        ctx.strokeStyle = "rgba(6, 182, 212, 0.55)";
        ctx.lineWidth = 2;
        ctx.stroke();

        // Encrypted ciphertext mathematical label
        if (u > 0.12 && u < 0.88) {
          ctx.shadowBlur = 0;
          ctx.font = `${Math.floor(9 * scale)}px "JetBrains Mono", monospace`;
          ctx.fillStyle = "rgba(16, 185, 129, 0.95)";
          ctx.fillText(p.label, px + 8, py - 4);
        }

        ctx.restore();
      });

      // -------------------------------------------------------------
      // 7. Consensus Shockwave Expansion (Triggered Round Effect)
      // -------------------------------------------------------------
      if (isShockwaveActive) {
        shockwaveRadius += 5;
        const maxRadius = Math.max(width, height) * 0.75;
        const alpha = Math.max(0, 1 - shockwaveRadius / maxRadius);

        ctx.save();
        ctx.beginPath();
        ctx.arc(coreX, coreY, shockwaveRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.8})`;
        ctx.lineWidth = 3.5;
        ctx.stroke();

        // Secondary cyan pulse
        if (shockwaveRadius > 45) {
          ctx.beginPath();
          ctx.arc(coreX, coreY, shockwaveRadius - 35, 0, Math.PI * 2);
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

    return () => {
      window.removeEventListener("resize", handleResize);
      if (animFrameRef.current) {
        cancelAnimationFrame(animFrameRef.current);
      }
    };
  }, [isSimulatingRound, selectedNodeId, mouseTilt, onSimulationComplete]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none select-none z-0"
    />
  );
}
