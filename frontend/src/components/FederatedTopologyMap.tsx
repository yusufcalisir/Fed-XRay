"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Server,
  Lock,
  ShieldCheck,
  CheckCircle2,
  Database,
  Layers,
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";

// ---------------------------------------------------------------------------
// Types & Domain Entities
// ---------------------------------------------------------------------------
export interface TopologyNode {
  id: string;
  shortName: string;
  fullName: string;
  city: string;
  countryCode: string;
  modality: string;
  dpEpsilon: number;
  scans: number;
  x: number;
  y: number;
  themeColor: string;
  badgeBgLight: string;
  badgeBgDark: string;
  badgeBorderLight: string;
  badgeBorderDark: string;
}

export interface FederatedTopologyMapProps {
  selectedNodeId?: string;
  isSimulatingRound?: boolean;
  onSelectNode?: (nodeId: string) => void;
  onSimulationComplete?: () => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Institutional Topology Nodes (Normalized 800x480 ViewBox)
// Dedicated Quadrants with Zero Overlap
// ---------------------------------------------------------------------------
const INSTITUTIONAL_NODES: TopologyNode[] = [
  {
    id: "node-1",
    shortName: "BCN-20000",
    fullName: "Skin Cancer Hub",
    city: "Barcelona",
    countryCode: "ES",
    modality: "Dermoscopy",
    dpEpsilon: 1.42,
    scans: 40,
    x: 165,
    y: 110,
    themeColor: "#059669",
    badgeBgLight: "rgba(5, 150, 105, 0.08)",
    badgeBgDark: "rgba(16, 185, 129, 0.12)",
    badgeBorderLight: "rgba(5, 150, 105, 0.25)",
    badgeBorderDark: "rgba(16, 185, 129, 0.35)",
  },
  {
    id: "node-3",
    shortName: "ViDIR",
    fullName: "Dermatopathology Inst.",
    city: "Vienna",
    countryCode: "AT",
    modality: "Histopathology",
    dpEpsilon: 1.55,
    scans: 40,
    x: 635,
    y: 110,
    themeColor: "#0284c7",
    badgeBgLight: "rgba(2, 132, 199, 0.08)",
    badgeBgDark: "rgba(56, 189, 248, 0.12)",
    badgeBorderLight: "rgba(2, 132, 199, 0.25)",
    badgeBorderDark: "rgba(56, 189, 248, 0.35)",
  },
  {
    id: "node-4",
    shortName: "Queensland",
    fullName: "Oncology Screening",
    city: "Brisbane",
    countryCode: "AU",
    modality: "Dermoscopy",
    dpEpsilon: 1.48,
    scans: 40,
    x: 165,
    y: 370,
    themeColor: "#0d9488",
    badgeBgLight: "rgba(13, 148, 136, 0.08)",
    badgeBgDark: "rgba(20, 184, 166, 0.12)",
    badgeBorderLight: "rgba(13, 148, 136, 0.25)",
    badgeBorderDark: "rgba(20, 184, 166, 0.35)",
  },
  {
    id: "node-2",
    shortName: "Beth Israel",
    fullName: "Deaconess Med Center",
    city: "Boston",
    countryCode: "USA",
    modality: "Chest X-Ray",
    dpEpsilon: 1.38,
    scans: 40,
    x: 635,
    y: 370,
    themeColor: "#2563eb",
    badgeBgLight: "rgba(37, 99, 235, 0.08)",
    badgeBgDark: "rgba(96, 165, 250, 0.12)",
    badgeBorderLight: "rgba(37, 99, 235, 0.25)",
    badgeBorderDark: "rgba(96, 165, 250, 0.35)",
  },
];

// Center Orchestrator Coordinates
const CENTER_X = 400;
const CENTER_Y = 240;

// ---------------------------------------------------------------------------
// Main Component: FederatedTopologyMap
// ---------------------------------------------------------------------------
export default function FederatedTopologyMap({
  selectedNodeId = "node-1",
  isSimulatingRound = false,
  onSelectNode,
  onSimulationComplete,
  className = "",
}: FederatedTopologyMapProps) {
  const { t } = useLanguage();
  const [animTime, setAnimTime] = useState(0);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [shockwaveProgress, setShockwaveProgress] = useState(0);
  const shockwaveActiveRef = useRef(false);

  // Smooth 60fps animation loop for particle streaming and shockwave ripples
  useEffect(() => {
    let animFrame: number;
    let lastTs = performance.now();

    const loop = (ts: number) => {
      const dt = (ts - lastTs) / 1000;
      lastTs = ts;
      setAnimTime((prev) => prev + dt);

      if (shockwaveActiveRef.current) {
        setShockwaveProgress((prev) => {
          const next = prev + dt * 0.9;
          if (next >= 1) {
            shockwaveActiveRef.current = false;
            onSimulationComplete?.();
            return 0;
          }
          return next;
        });
      }

      animFrame = requestAnimationFrame(loop);
    };

    animFrame = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animFrame);
  }, [onSimulationComplete]);

  // Trigger shockwave when simulation starts
  useEffect(() => {
    if (isSimulatingRound) {
      shockwaveActiveRef.current = true;
      setShockwaveProgress(0);
    }
  }, [isSimulatingRound]);

  const speedMultiplier = isSimulatingRound ? 3.5 : 1.0;

  return (
    <div
      className={`relative w-full h-full flex items-center justify-center select-none ${className}`}
    >
      <svg
        viewBox="0 0 800 480"
        className="w-full h-full max-h-[460px] overflow-visible"
        preserveAspectRatio="xMidYMid meet"
      >
        <defs>
          {/* Subtle grid pattern for precision architectural coordinate grid */}
          <pattern
            id="arch-blueprint-grid"
            width="40"
            height="40"
            patternUnits="userSpaceOnUse"
          >
            <path
              d="M 40 0 L 0 0 0 40"
              fill="none"
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-slate-300/40 dark:text-slate-800/40"
            />
          </pattern>

          {/* Linear gradient for conduits */}
          {INSTITUTIONAL_NODES.map((node) => (
            <linearGradient
              key={`grad-${node.id}`}
              id={`conduit-grad-${node.id}`}
              x1={node.x}
              y1={node.y}
              x2={CENTER_X}
              y2={CENTER_Y}
              gradientUnits="userSpaceOnUse"
            >
              <stop
                offset="0%"
                stopColor={node.themeColor}
                stopOpacity="0.8"
              />
              <stop
                offset="100%"
                stopColor="#059669"
                stopOpacity="0.85"
              />
            </linearGradient>
          ))}

          {/* Glow filter for active selection and shockwave */}
          <filter id="subtle-enclave-glow" x="-20%" y="-20%" width="140%" height="140%">
            <feGaussianBlur stdDeviation="3" result="blur" />
            <feComposite in="SourceGraphic" in2="blur" operator="over" />
          </filter>
        </defs>

        {/* Background Architectural Blueprint Grid */}
        <rect
          x="20"
          y="20"
          width="760"
          height="440"
          rx="16"
          fill="url(#arch-blueprint-grid)"
          className="opacity-60"
        />

        {/* Concentric Coordinate Radar Rings (Centered on Orchestrator) */}
        {[100, 200, 310].map((radius, i) => (
          <circle
            key={`range-${i}`}
            cx={CENTER_X}
            cy={CENTER_Y}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth="1"
            strokeDasharray={i === 1 ? "4 6" : undefined}
            className="text-slate-300/50 dark:text-slate-800/60"
          />
        ))}

        {/* Axis Crosshairs */}
        <line
          x1={CENTER_X - 320}
          y1={CENTER_Y}
          x2={CENTER_X + 320}
          y2={CENTER_Y}
          stroke="currentColor"
          strokeWidth="0.75"
          strokeDasharray="2 4"
          className="text-slate-300/40 dark:text-slate-800/40"
        />
        <line
          x1={CENTER_X}
          y1={CENTER_Y - 180}
          x2={CENTER_X}
          y2={CENTER_Y + 180}
          stroke="currentColor"
          strokeWidth="0.75"
          strokeDasharray="2 4"
          className="text-slate-300/40 dark:text-slate-800/40"
        />

        {/* ===================================================================
            LAYER 1: ENCRYPTED TENSOR CONDUITS (Quadratic Bezier Curves)
            =================================================================== */}
        {INSTITUTIONAL_NODES.map((node, i) => {
          const isSelected = selectedNodeId === node.id;
          const isHovered = hoveredNodeId === node.id;
          const isHighlighted = isSelected || isHovered;

          // Compute smooth curved conduit
          // Control point pulled slightly outward for aesthetic curvature
          const midX = (node.x + CENTER_X) / 2 + (i % 2 === 0 ? -25 : 25);
          const midY = (node.y + CENTER_Y) / 2 + (node.y < CENTER_Y ? -15 : 15);
          const pathD = `M ${node.x} ${node.y} Q ${midX} ${midY} ${CENTER_X} ${CENTER_Y}`;

          // Calculate 3 moving photon packets along the curve
          // Using quadratic bezier formula: B(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
          const packetOffsets = [0, 0.33, 0.66];

          return (
            <g key={`conduit-group-${node.id}`}>
              {/* Static Base Conduit Line */}
              <path
                d={pathD}
                fill="none"
                stroke="currentColor"
                strokeWidth={isHighlighted ? "2" : "1.25"}
                className={
                  isHighlighted
                    ? "text-emerald-500/80 dark:text-emerald-400/90"
                    : "text-slate-300 dark:text-slate-700/80"
                }
              />

              {/* Directional Signal Guide Dash */}
              <path
                d={pathD}
                fill="none"
                stroke={`url(#conduit-grad-${node.id})`}
                strokeWidth={isHighlighted ? "2" : "1.25"}
                strokeDasharray="6 14"
                strokeDashoffset={-animTime * 28 * speedMultiplier}
                className="opacity-70"
              />

              {/* Animated Encrypted Tensor Packets (Photons) */}
              {packetOffsets.map((offset, pIdx) => {
                const prog =
                  ((animTime * 0.22 * speedMultiplier + offset + i * 0.12) % 1 + 1) % 1;
                const inv = 1 - prog;
                const px =
                  inv * inv * node.x + 2 * inv * prog * midX + prog * prog * CENTER_X;
                const py =
                  inv * inv * node.y + 2 * inv * prog * midY + prog * prog * CENTER_Y;

                return (
                  <g key={`packet-${node.id}-${pIdx}`}>
                    {/* Outer glow ring */}
                    <circle
                      cx={px}
                      cy={py}
                      r={isHighlighted ? 4.5 : 3.5}
                      fill={node.themeColor}
                      className="opacity-30"
                    />
                    {/* Inner high-contrast bright core */}
                    <circle
                      cx={px}
                      cy={py}
                      r={isHighlighted ? 2.5 : 1.8}
                      fill="#ffffff"
                    />
                  </g>
                );
              })}

              {/* Mathematical Payload Annotations along Conduits */}
              {isHighlighted && (
                <text
                  x={midX}
                  y={midY - 8}
                  textAnchor="middle"
                  className="text-[9px] font-mono font-bold fill-slate-700 dark:fill-slate-300"
                >
                  Enc(B_{i + 1}) · A
                </text>
              )}
            </g>
          );
        })}

        {/* ===================================================================
            LAYER 2: CONSENSUS SHOCKWAVE WAVEFRONT RIPPLE
            =================================================================== */}
        {shockwaveProgress > 0 && (
          <g>
            <circle
              cx={CENTER_X}
              cy={CENTER_Y}
              r={shockwaveProgress * 340}
              fill="none"
              stroke="#059669"
              strokeWidth="2"
              opacity={(1 - shockwaveProgress) * 0.75}
            />
            {shockwaveProgress > 0.08 && (
              <circle
                cx={CENTER_X}
                cy={CENTER_Y}
                r={(shockwaveProgress - 0.08) * 340}
                fill="none"
                stroke="#0284c7"
                strokeWidth="1.5"
                strokeDasharray="4 6"
                opacity={(1 - (shockwaveProgress - 0.08)) * 0.55}
              />
            )}
          </g>
        )}

        {/* ===================================================================
            LAYER 3: CENTRAL CONSORTIUM ORCHESTRATOR ENCLAVE
            Multi-Tier Enterprise Pedestal with RLWE CKKS Cryptographic Core
            =================================================================== */}
        <g
          transform={`translate(${CENTER_X}, ${CENTER_Y})`}
          className="cursor-default"
        >
          {/* Ground Footprint Plate */}
          <rect
            x="-85"
            y="-48"
            width="170"
            height="96"
            rx="14"
            className="fill-slate-100/90 dark:fill-slate-900/95 stroke-slate-300 dark:stroke-slate-700/80 shadow-md"
            strokeWidth="1.5"
          />

          {/* Accent Inner Ring / Shield Rim */}
          <rect
            x="-80"
            y="-43"
            width="160"
            height="86"
            rx="10"
            fill="none"
            className="stroke-emerald-500/30 dark:stroke-emerald-500/40"
            strokeWidth="1"
            strokeDasharray="4 4"
          />

          {/* Central Cryptographic Lock Pip */}
          <g transform="translate(0, -20)">
            <circle
              cx="0"
              cy="0"
              r="15"
              className="fill-emerald-500/10 dark:fill-emerald-500/20 stroke-emerald-500/40 dark:stroke-emerald-400/50"
              strokeWidth="1.2"
            />
            <g transform="translate(-7, -7)">
              <Lock className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
            </g>
          </g>

          {/* Header Title: Consortium Orchestrator */}
          <text
            x="0"
            y="6"
            textAnchor="middle"
            className="text-[10.5px] font-mono font-bold tracking-wider fill-slate-900 dark:fill-slate-100 uppercase"
          >
            CONSORTIUM AGGREGATOR
          </text>

          {/* Security Standard: RLWE CKKS 128-Bit */}
          <text
            x="0"
            y="19"
            textAnchor="middle"
            className="text-[8.5px] font-mono font-semibold fill-emerald-600 dark:fill-emerald-400"
          >
            RLWE CKKS 128-Bit Enclave
          </text>

          {/* Status Metric: Zero Plaintext Transfer */}
          <text
            x="0"
            y="31"
            textAnchor="middle"
            className="text-[8px] font-sans font-medium fill-slate-500 dark:fill-slate-400"
          >
            Zero Plaintext Patient Data
          </text>
        </g>

        {/* ===================================================================
            LAYER 4: CLINICAL INSTITUTIONAL ENCLAVE PODS
            Zero Overlap, High-Contrast Ceramic/Titanium Architectural Blocks
            =================================================================== */}
        {INSTITUTIONAL_NODES.map((node) => {
          const isSelected = selectedNodeId === node.id;
          const isHovered = hoveredNodeId === node.id;
          const isHighlighted = isSelected || isHovered;

          // Pod card dimensions: 170px width, 72px height
          const W = 170;
          const H = 72;
          const left = node.x - W / 2;
          const top = node.y - H / 2;

          return (
            <g
              key={node.id}
              className="cursor-pointer transition-transform duration-200"
              onClick={() => onSelectNode?.(node.id)}
              onMouseEnter={() => setHoveredNodeId(node.id)}
              onMouseLeave={() => setHoveredNodeId(null)}
            >
              {/* Ground Anchor Shadow Plate */}
              <ellipse
                cx={node.x}
                cy={node.y + H / 2 + 6}
                rx="42"
                ry="7"
                className="fill-slate-400/20 dark:fill-black/40"
              />

              {/* Main Pod Chassis Card */}
              <rect
                x={left}
                y={top}
                width={W}
                height={H}
                rx="12"
                className={`transition-all duration-200 ${
                  isSelected
                    ? "fill-white dark:fill-slate-900 stroke-emerald-500 dark:stroke-emerald-400 shadow-lg"
                    : isHovered
                      ? "fill-white dark:fill-slate-900 stroke-slate-400 dark:stroke-slate-600 shadow-md"
                      : "fill-white/95 dark:fill-slate-900/90 stroke-slate-300 dark:stroke-slate-700/80 shadow-xs"
                }`}
                strokeWidth={isSelected ? "2" : "1.25"}
              />

              {/* Left Color-Coded Accent Indicator Strip */}
              <rect
                x={left}
                y={top + 8}
                width="3.5"
                height={H - 16}
                rx="1.75"
                fill={node.themeColor}
              />

              {/* Row 1: Status Dot + Short Name + Country Badge */}
              <g transform={`translate(${left + 12}, ${top + 18})`}>
                {/* Status Dot */}
                <circle
                  cx="4"
                  cy="-3"
                  r="3.5"
                  fill={node.themeColor}
                />
                <circle
                  cx="4"
                  cy="-3"
                  r="6"
                  fill="none"
                  stroke={node.themeColor}
                  strokeWidth="1"
                  className={isSelected ? "animate-ping opacity-60" : "opacity-30"}
                />

                {/* Institutional Code */}
                <text
                  x="14"
                  y="0"
                  className="text-[11px] font-mono font-bold tracking-tight fill-slate-900 dark:fill-slate-100"
                >
                  {node.shortName}
                </text>

                {/* Country Badge */}
                <rect
                  x="96"
                  y="-9"
                  width="44"
                  height="14"
                  rx="4"
                  className="fill-slate-100 dark:fill-slate-800 stroke-slate-200 dark:stroke-slate-700"
                  strokeWidth="0.8"
                />
                <text
                  x="118"
                  y="1.5"
                  textAnchor="middle"
                  className="text-[8px] font-mono font-bold fill-slate-600 dark:fill-slate-300"
                >
                  {node.countryCode}
                </text>
              </g>

              {/* Row 2: Location & Clinical Modality */}
              <text
                x={left + 14}
                y={top + 37}
                className="text-[9px] font-sans font-medium fill-slate-500 dark:fill-slate-400 truncate"
              >
                {node.city} · {node.modality}
              </text>

              {/* Row 3: Bottom Divider & Differential Privacy Budget */}
              <line
                x1={left + 12}
                y1={top + 48}
                x2={left + W - 12}
                y2={top + 48}
                className="stroke-slate-200 dark:stroke-slate-800"
                strokeWidth="0.8"
              />

              <g transform={`translate(${left + 14}, ${top + 61})`}>
                <text
                  x="0"
                  y="0"
                  className="text-[8.5px] font-mono fill-slate-400 dark:fill-slate-500"
                >
                  DP Budget:
                </text>
                <text
                  x="56"
                  y="0"
                  className="text-[8.5px] font-mono font-bold fill-emerald-600 dark:fill-emerald-400"
                >
                  ε = {node.dpEpsilon}
                </text>

                {/* Verified Biopsy/Cohort Indicator */}
                <text
                  x="112"
                  y="0"
                  className="text-[8px] font-mono font-semibold fill-slate-500 dark:fill-slate-400"
                >
                  {node.scans} scans
                </text>
              </g>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
