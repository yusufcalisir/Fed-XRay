"use client";

import React, { useState, useEffect } from "react";
import {
  Lock,
  Server,
  ShieldCheck,
  CheckCircle2,
  Activity,
  Layers,
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";

// ---------------------------------------------------------------------------
// Types & Domain Entities
// ---------------------------------------------------------------------------
export interface TopologyNode {
  id: string;
  shortName: string;
  fullNameKey: string;
  cityKey: string;
  countryCodeKey: string;
  modalityKey: string;
  dpEpsilon: number;
  scans: number;
  themeColor: string;
  accentBg: string;
  accentBorder: string;
}

export interface FederatedTopologyMapProps {
  selectedNodeId?: string;
  isSimulatingRound?: boolean;
  onSelectNode?: (nodeId: string) => void;
  onSimulationComplete?: () => void;
  className?: string;
}

// ---------------------------------------------------------------------------
// Institutional Topology Nodes
// ---------------------------------------------------------------------------
const INSTITUTIONS: TopologyNode[] = [
  {
    id: "node-1",
    shortName: "BCN-20000",
    fullNameKey: "topo_node1_fullname",
    cityKey: "topo_city_barcelona",
    countryCodeKey: "topo_country_es",
    modalityKey: "topo_mod_dermo",
    dpEpsilon: 1.42,
    scans: 40,
    themeColor: "#059669",
    accentBg: "bg-emerald-500/10",
    accentBorder: "border-emerald-500/30",
  },
  {
    id: "node-3",
    shortName: "ViDIR",
    fullNameKey: "topo_node3_fullname",
    cityKey: "topo_city_vienna",
    countryCodeKey: "topo_country_at",
    modalityKey: "topo_mod_histo",
    dpEpsilon: 1.55,
    scans: 40,
    themeColor: "#0284c7",
    accentBg: "bg-sky-500/10",
    accentBorder: "border-sky-500/30",
  },
  {
    id: "node-4",
    shortName: "Queensland",
    fullNameKey: "topo_node4_fullname",
    cityKey: "topo_city_brisbane",
    countryCodeKey: "topo_country_au",
    modalityKey: "topo_mod_dermo",
    dpEpsilon: 1.48,
    scans: 40,
    themeColor: "#0d9488",
    accentBg: "bg-teal-500/10",
    accentBorder: "border-teal-500/30",
  },
  {
    id: "node-2",
    shortName: "Beth Israel",
    fullNameKey: "topo_node2_fullname",
    cityKey: "topo_city_boston",
    countryCodeKey: "topo_country_us",
    modalityKey: "topo_mod_xray",
    dpEpsilon: 1.38,
    scans: 40,
    themeColor: "#2563eb",
    accentBg: "bg-blue-500/10",
    accentBorder: "border-blue-500/30",
  },
];

// ---------------------------------------------------------------------------
// Main Component: FederatedTopologyMap
// Hybrid High-Legibility Native HTML Nodes + SVG Encrypted Conduit Layer
// ---------------------------------------------------------------------------
export default function FederatedTopologyMap({
  selectedNodeId = "node-1",
  isSimulatingRound = false,
  onSelectNode,
  onSimulationComplete,
  className = "",
}: FederatedTopologyMapProps) {
  const { t } = useLanguage();
  const [animProgress, setAnimProgress] = useState(0);
  const [shockwave, setShockwave] = useState(0);
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);

  // Smooth continuous animation for encrypted packet travel
  useEffect(() => {
    let frameId: number;
    let lastTs = performance.now();

    const loop = (ts: number) => {
      const dt = (ts - lastTs) / 1000;
      lastTs = ts;
      const speed = isSimulatingRound ? 2.5 : 0.6;
      setAnimProgress((prev) => (prev + dt * speed) % 1);

      if (isSimulatingRound) {
        setShockwave((prev) => {
          const next = prev + dt * 0.9;
          if (next >= 1) {
            onSimulationComplete?.();
            return 0;
          }
          return next;
        });
      } else {
        setShockwave(0);
      }

      frameId = requestAnimationFrame(loop);
    };

    frameId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(frameId);
  }, [isSimulatingRound, onSimulationComplete]);

  const topNodes = [INSTITUTIONS[0], INSTITUTIONS[1]];
  const bottomNodes = [INSTITUTIONS[2], INSTITUTIONS[3]];

  return (
    <div
      className={`relative w-full h-full min-h-[350px] sm:min-h-[380px] flex flex-col justify-between py-2 sm:py-3 px-1.5 sm:px-4 select-none ${className}`}
    >
      {/* ===================================================================
          BACKGROUND SVG LAYER: ENCRYPTED TENSOR CONDUITS & SHOCKWAVE
          =================================================================== */}
      <svg
        className="absolute inset-0 w-full h-full pointer-events-none overflow-visible"
        preserveAspectRatio="none"
        viewBox="0 0 100 100"
      >
        <defs>
          <linearGradient id="conduit-grad-left" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stopColor="#059669" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#0284c7" stopOpacity="0.7" />
          </linearGradient>
          <linearGradient id="conduit-grad-right" x1="1" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0284c7" stopOpacity="0.7" />
            <stop offset="100%" stopColor="#059669" stopOpacity="0.7" />
          </linearGradient>
        </defs>

        {/* Dynamic Curved Conduits from 4 corners to center (50, 50) */}
        {[
          { x1: 22, y1: 18, cx: 34, cy: 30, x2: 50, y2: 50, id: "node-1" },
          { x1: 78, y1: 18, cx: 66, cy: 30, x2: 50, y2: 50, id: "node-3" },
          { x1: 22, y1: 82, cx: 34, cy: 70, x2: 50, y2: 50, id: "node-4" },
          { x1: 78, y1: 82, cx: 66, cy: 70, x2: 50, y2: 50, id: "node-2" },
        ].map((c) => {
          const isSelected = selectedNodeId === c.id;
          const isHovered = hoveredNodeId === c.id;
          const isHighlighted = isSelected || isHovered;
          const pathD = `M ${c.x1} ${c.y1} Q ${c.cx} ${c.cy} ${c.x2} ${c.y2}`;

          // Calculate moving packet point along quadratic bezier
          // P(t) = (1-t)^2 P0 + 2(1-t)t P1 + t^2 P2
          const t = animProgress;
          const inv = 1 - t;
          const px = inv * inv * c.x1 + 2 * inv * t * c.cx + t * t * c.x2;
          const py = inv * inv * c.y1 + 2 * inv * t * c.cy + t * t * c.y2;

          return (
            <g key={`conduit-${c.id}`}>
              {/* Base Conduit Path */}
              <path
                d={pathD}
                fill="none"
                stroke="currentColor"
                strokeWidth={isHighlighted ? "0.8" : "0.5"}
                className={
                  isHighlighted
                    ? "text-emerald-500/80 dark:text-emerald-400"
                    : "text-slate-300 dark:text-slate-800"
                }
              />

              {/* Animated Dashed Signal Stream */}
              <path
                d={pathD}
                fill="none"
                stroke={c.x1 < 50 ? "url(#conduit-grad-left)" : "url(#conduit-grad-right)"}
                strokeWidth={isHighlighted ? "0.8" : "0.5"}
                strokeDasharray="2 3"
                className="opacity-75"
              />

              {/* Traveling Encrypted Parameter Photon */}
              <circle
                cx={px}
                cy={py}
                r={isHighlighted ? 1.4 : 1.0}
                className="fill-emerald-500 dark:fill-emerald-400 shadow-sm"
              />
              <circle
                cx={px}
                cy={py}
                r={isHighlighted ? 0.7 : 0.5}
                fill="#ffffff"
              />
            </g>
          );
        })}

        {/* Concentric Center Range Rings */}
        <circle
          cx="50"
          cy="50"
          r="16"
          fill="none"
          stroke="currentColor"
          strokeWidth="0.35"
          strokeDasharray="1.5 2"
          className="text-slate-300/60 dark:text-slate-800/80"
        />
        <circle
          cx="50"
          cy="50"
          r="28"
          fill="none"
          stroke="currentColor"
          strokeWidth="0.35"
          className="text-slate-200/50 dark:text-slate-800/40"
        />

        {/* Consensus Shockwave Wavefront */}
        {shockwave > 0 && (
          <circle
            cx="50"
            cy="50"
            r={shockwave * 45}
            fill="none"
            stroke="#059669"
            strokeWidth="0.8"
            opacity={(1 - shockwave) * 0.8}
          />
        )}
      </svg>

      {/* ===================================================================
          ROW 1: TOP CLINICAL INSTITUTIONS (BCN-20000 & ViDIR)
          =================================================================== */}
      <div className="relative z-10 flex items-center justify-between gap-2 sm:gap-4 w-full">
        {topNodes.map((node) => (
          <InstitutionCard
            key={node.id}
            node={node}
            isSelected={selectedNodeId === node.id}
            onSelect={() => onSelectNode?.(node.id)}
            onHover={(h) => setHoveredNodeId(h ? node.id : null)}
          />
        ))}
      </div>

      {/* ===================================================================
          ROW 2: CENTRAL CONSORTIUM ORCHESTRATOR ENCLAVE
          High-Contrast Architectural Hub with Cryptographic Proof
          =================================================================== */}
      <div className="relative z-10 flex items-center justify-center w-full my-auto py-1.5">
        <div className="relative w-full max-w-[280px] sm:max-w-[340px] px-3 py-2 sm:py-2.5 rounded-2xl bg-white/95 dark:bg-slate-900/95 border-2 border-emerald-500/40 shadow-sm backdrop-blur-md text-center transition-all">
          {/* Subtle Outer Status Pip */}
          <div className="flex items-center justify-center gap-1.5 mb-0.5">
            <div className="w-5 h-5 rounded-lg bg-emerald-500/10 dark:bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center shrink-0">
              <Lock className="w-3 h-3 text-emerald-600 dark:text-emerald-400" />
            </div>
            <span className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-wider text-slate-900 dark:text-slate-100">
              {t("topo_hub_title")}
            </span>
          </div>

          {/* Cryptographic Standard */}
          <div className="font-mono text-[10px] sm:text-[11px] font-semibold text-emerald-600 dark:text-emerald-400">
            {t("topo_hub_crypto")}
          </div>

          {/* Privacy Guarantee */}
          <div className="text-[9.5px] sm:text-[10px] text-slate-600 dark:text-slate-400 font-medium mt-0.5">
            {t("topo_hub_privacy")}
          </div>
        </div>
      </div>

      {/* ===================================================================
          ROW 3: BOTTOM CLINICAL INSTITUTIONS (Queensland & Beth Israel)
          =================================================================== */}
      <div className="relative z-10 flex items-center justify-between gap-2 sm:gap-4 w-full">
        {bottomNodes.map((node) => (
          <InstitutionCard
            key={node.id}
            node={node}
            isSelected={selectedNodeId === node.id}
            onSelect={() => onSelectNode?.(node.id)}
            onHover={(h) => setHoveredNodeId(h ? node.id : null)}
          />
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-component: High-Legibility Native HTML Institution Card
// WCAG AAA Contrast, 100% Crisp Font Rendering across All Viewports
// ---------------------------------------------------------------------------
function InstitutionCard({
  node,
  isSelected,
  onSelect,
  onHover,
}: {
  node: TopologyNode;
  isSelected: boolean;
  onSelect: () => void;
  onHover: (hovered: boolean) => void;
}) {
  const { t } = useLanguage();

  return (
    <div
      onClick={onSelect}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
      className={`relative w-[calc(50%-4px)] sm:w-56 p-2.5 sm:p-3 rounded-xl transition-all cursor-pointer ${
        isSelected
          ? "bg-white dark:bg-slate-900 border-2 border-emerald-500 dark:border-emerald-400 shadow-md scale-[1.02]"
          : "bg-white/95 dark:bg-slate-900/90 border border-slate-200 dark:border-slate-800 shadow-xs hover:border-slate-300 dark:hover:border-slate-700 hover:shadow-sm"
      }`}
    >
      {/* Header: Status Dot + Code + Country Badge */}
      <div className="flex items-center justify-between gap-1 mb-1">
        <div className="flex items-center gap-1.5 min-w-0">
          <span
            className="w-2 h-2 rounded-full shrink-0 shadow-xs"
            style={{ backgroundColor: node.themeColor }}
          />
          <span className="font-mono text-xs sm:text-sm font-bold tracking-tight text-slate-900 dark:text-slate-100 truncate">
            {node.shortName}
          </span>
        </div>

        <span className="px-1.5 py-0.5 rounded text-[9.5px] sm:text-[10px] font-mono font-bold bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 shrink-0">
          {t(node.countryCodeKey)}
        </span>
      </div>

      {/* Sub: City & Clinical Modality */}
      <div className="text-[10.5px] sm:text-xs text-slate-600 dark:text-slate-400 font-medium truncate">
        {t(node.cityKey)} : {t(node.modalityKey)}
      </div>

      {/* Footer: Privacy Budget & Scans */}
      <div className="mt-1.5 pt-1.5 border-t border-slate-100 dark:border-slate-800/80 flex items-center justify-between text-[10px] sm:text-[11px] font-mono">
        <span className="text-slate-500 dark:text-slate-400">{t("topo_dp_budget")}</span>
        <span className="font-bold text-emerald-600 dark:text-emerald-400">
          ε = {node.dpEpsilon}
        </span>
      </div>
    </div>
  );
}
