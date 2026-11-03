"use client";

import React, { useEffect, useRef } from "react";

interface Consortium3DCanvasProps {
  is3DMode?: boolean;
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
  is3DMode = true,
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
          speed: 0.0035 + Math.random() * 0.0015,
          label: TUNNEL_LABELS[t][p % TUNNEL_LABELS[t].length],
          size: 3 + (p % 2),
        });
      }
    }

    // Trigger shockwave on simulation round
    if (isSimulatingRound) {
      shockwaveRadius = 10;
      isShockwaveActive = true;
    }

    const render = () => {
      time += 0.02;
      ctx.clearRect(0, 0, width, height);

      const cx = width / 2;
      const cy = height / 2;

      // 3D Tilt Vanishing Offset
      const tiltOffsetX = is3DMode ? mouseTilt.x * 24 : 0;
      const tiltOffsetY = is3DMode ? mouseTilt.y * 18 : 0;

      // Node coordinate positions (percentage of container width/height)
      const nodes = [
        { x: width * 0.22, y: height * 0.19, id: "node-1" }, // Top-Left (Barcelona)
        { x: width * 0.78, y: height * 0.19, id: "node-3" }, // Top-Right (Vienna)
        { x: width * 0.22, y: height * 0.81, id: "node-4" }, // Bottom-Left (Brisbane)
        { x: width * 0.78, y: height * 0.81, id: "node-2" }, // Bottom-Right (Boston)
      ];

      // -------------------------------------------------------------
      // 1. 3D Isometric Ground Grid
      // -------------------------------------------------------------
      if (is3DMode) {
        ctx.save();
        ctx.strokeStyle = "rgba(16, 185, 129, 0.04)";
        ctx.lineWidth = 1;
        const gridLines = 8;
        for (let i = -gridLines; i <= gridLines; i++) {
          const gx = cx + i * (width / (gridLines * 2.2)) + tiltOffsetX * 0.5;
          ctx.beginPath();
          ctx.moveTo(gx, 0);
          ctx.lineTo(gx + (i * 12), height);
          ctx.stroke();
        }
        for (let j = 0; j <= 6; j++) {
          const gy = height * 0.25 + j * (height * 0.12);
          ctx.beginPath();
          ctx.moveTo(0, gy);
          ctx.lineTo(width, gy);
          ctx.stroke();
        }
        ctx.restore();
      }

      // -------------------------------------------------------------
      // 2. Inter-Hospital Byzantine Verification Mesh (Elliptical Ring)
      // -------------------------------------------------------------
      ctx.save();
      const ringRadiusX = width * 0.38;
      const ringRadiusY = height * 0.34;
      ctx.beginPath();
      ctx.ellipse(cx + tiltOffsetX * 0.4, cy + tiltOffsetY * 0.4, ringRadiusX, ringRadiusY, 0, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.18)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 8]);
      ctx.lineDashOffset = -time * 12;
      ctx.stroke();

      // Floating Byzantine validation tokens on the mesh perimeter
      const meshAngle = time * 0.4;
      for (let m = 0; m < 4; m++) {
        const ma = meshAngle + (m * Math.PI) / 2;
        const mx = cx + tiltOffsetX * 0.4 + Math.cos(ma) * ringRadiusX;
        const my = cy + tiltOffsetY * 0.4 + Math.sin(ma) * ringRadiusY;
        ctx.fillStyle = "rgba(6, 182, 212, 0.6)";
        ctx.beginPath();
        ctx.arc(mx, my, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // -------------------------------------------------------------
      // 3. Central Aggregator 3D Orbital Rings & Glow Beacon
      // -------------------------------------------------------------
      ctx.save();
      const coreX = cx + tiltOffsetX;
      const coreY = cy + tiltOffsetY;

      // Outer revolving cipher ring
      ctx.beginPath();
      ctx.ellipse(coreX, coreY, 68, 52, time * 0.5, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(16, 185, 129, 0.3)";
      ctx.lineWidth = 1.5;
      ctx.setLineDash([8, 12]);
      ctx.lineDashOffset = time * 18;
      ctx.stroke();

      // Inner counter-revolving ring
      ctx.beginPath();
      ctx.ellipse(coreX, coreY, 54, 38, -time * 0.7, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(6, 182, 212, 0.35)";
      ctx.lineWidth = 1.2;
      ctx.setLineDash([6, 6]);
      ctx.lineDashOffset = -time * 14;
      ctx.stroke();

      // Ambient radial core glow
      const coreGrad = ctx.createRadialGradient(coreX, coreY, 8, coreX, coreY, 64);
      coreGrad.addColorStop(0, "rgba(16, 185, 129, 0.25)");
      coreGrad.addColorStop(0.6, "rgba(6, 182, 212, 0.08)");
      coreGrad.addColorStop(1, "rgba(16, 185, 129, 0)");
      ctx.fillStyle = coreGrad;
      ctx.beginPath();
      ctx.arc(coreX, coreY, 64, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // -------------------------------------------------------------
      // 4. Transmission Tunnels (Hospital Node -> Central Gateway)
      // -------------------------------------------------------------
      nodes.forEach((node, idx) => {
        const isSelected = node.id === selectedNodeId;
        ctx.save();

        // Curved spline path through an elevated control point
        const midX = (node.x + coreX) / 2 + (idx % 2 === 0 ? -15 : 15);
        const midY = (node.y + coreY) / 2 + (idx < 2 ? -10 : 10);

        // Background glowing tunnel line
        ctx.beginPath();
        ctx.moveTo(node.x, node.y);
        ctx.quadraticCurveTo(midX, midY, coreX, coreY);
        ctx.strokeStyle = isSelected
          ? "rgba(16, 185, 129, 0.5)"
          : "rgba(16, 185, 129, 0.2)";
        ctx.lineWidth = isSelected ? 2.5 : 1.5;
        ctx.setLineDash([5, 5]);
        ctx.lineDashOffset = -time * 15;
        ctx.stroke();

        // Subtle glow blur on active tunnels
        if (isSelected || isSimulatingRound) {
          ctx.beginPath();
          ctx.moveTo(node.x, node.y);
          ctx.quadraticCurveTo(midX, midY, coreX, coreY);
          ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
          ctx.lineWidth = 4;
          ctx.stroke();
        }

        ctx.restore();
      });

      // -------------------------------------------------------------
      // 5. Flowing Encrypted Ciphertext Packets (3D Stream)
      // -------------------------------------------------------------
      particles.forEach((p) => {
        const node = nodes[p.tunnelIndex];
        const currentSpeed = isSimulatingRound ? p.speed * 2.8 : p.speed;
        p.progress = (p.progress + currentSpeed) % 1;

        const u = p.progress;
        const midX = (node.x + coreX) / 2 + (p.tunnelIndex % 2 === 0 ? -15 : 15);
        const midY = (node.y + coreY) / 2 + (p.tunnelIndex < 2 ? -10 : 10);

        // Quadratic Bezier interpolation: B(u) = (1-u)^2 * P0 + 2*(1-u)*u * P1 + u^2 * P2
        const px = (1 - u) * (1 - u) * node.x + 2 * (1 - u) * u * midX + u * u * coreX;
        const py = (1 - u) * (1 - u) * node.y + 2 * (1 - u) * u * midY + u * u * coreY;

        ctx.save();

        // 3D perspective scale: packets grow slightly as they approach center
        const scale = is3DMode ? 0.85 + u * 0.35 : 1.0;

        // Glowing packet capsule
        ctx.fillStyle = "#10b981";
        ctx.shadowColor = "#10b981";
        ctx.shadowBlur = 10;
        ctx.beginPath();
        ctx.arc(px, py, p.size * scale, 0, Math.PI * 2);
        ctx.fill();

        // Glowing tail trail
        const trailU = Math.max(0, u - 0.04);
        const tx = (1 - trailU) * (1 - trailU) * node.x + 2 * (1 - trailU) * trailU * midX + trailU * trailU * coreX;
        const ty = (1 - trailU) * (1 - trailU) * node.y + 2 * (1 - trailU) * trailU * midY + trailU * trailU * coreY;
        ctx.beginPath();
        ctx.moveTo(px, py);
        ctx.lineTo(tx, ty);
        ctx.strokeStyle = "rgba(6, 182, 212, 0.4)";
        ctx.lineWidth = 1.5;
        ctx.stroke();

        // Ciphertext badge label (shows encrypted token text)
        if (u > 0.15 && u < 0.85) {
          ctx.shadowBlur = 0;
          ctx.font = `${Math.floor(8 * scale)}px "JetBrains Mono", monospace`;
          ctx.fillStyle = "rgba(16, 185, 129, 0.9)";
          ctx.fillText(p.label, px + 7, py - 4);
        }

        ctx.restore();
      });

      // -------------------------------------------------------------
      // 6. Expanding Consensus Shockwave (when round completes or pulses)
      // -------------------------------------------------------------
      if (isShockwaveActive) {
        shockwaveRadius += 4;
        const maxRadius = Math.max(width, height) * 0.7;
        const alpha = Math.max(0, 1 - shockwaveRadius / maxRadius);

        ctx.save();
        ctx.beginPath();
        ctx.arc(coreX, coreY, shockwaveRadius, 0, Math.PI * 2);
        ctx.strokeStyle = `rgba(16, 185, 129, ${alpha * 0.7})`;
        ctx.lineWidth = 3;
        ctx.stroke();

        // Secondary cyan pulse
        if (shockwaveRadius > 40) {
          ctx.beginPath();
          ctx.arc(coreX, coreY, shockwaveRadius - 30, 0, Math.PI * 2);
          ctx.strokeStyle = `rgba(6, 182, 212, ${alpha * 0.5})`;
          ctx.lineWidth = 1.5;
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
  }, [is3DMode, isSimulatingRound, selectedNodeId, mouseTilt, onSimulationComplete]);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full pointer-events-none select-none z-0"
    />
  );
}
