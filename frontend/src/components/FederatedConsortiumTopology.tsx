"use client";

import React, { useState } from "react";
import {
  ShieldCheck,
  Lock,
  Activity,
  Server,
  AlertTriangle,
  TrendingDown,
  FileCheck,
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";

interface HospitalNode {
  id: string;
  name: string;
  location: string;
  scanner: string;
  scans: number;
  modalityKey: string;
  skewKey: string;
  adapter: string;
  dpEpsilon: number;
  status: "online" | "syncing";
}

const HOSPITAL_NODES: HospitalNode[] = [
  {
    id: "node-1",
    name: "BCN-20000 Skin Cancer Hub",
    location: "Barcelona, ES",
    scanner: "Siemens Sensation Optical Dermatoscope",
    scans: 54,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_melanoma",
    adapter: "FFA-LoRA (r=16, α=32)",
    dpEpsilon: 1.42,
    status: "online",
  },
  {
    id: "node-2",
    name: "Beth Israel Deaconess Medical Center",
    location: "Boston, USA",
    scanner: "GE Revolution Digital Radiography (MIMIC-CXR)",
    scans: 50,
    modalityKey: "topo_mod_xray",
    skewKey: "topo_skew_pneumonia",
    adapter: "FFA-LoRA (r=16, α=32)",
    dpEpsilon: 1.38,
    status: "online",
  },
  {
    id: "node-3",
    name: "ViDIR Dermatopathology Institute",
    location: "Vienna, AT",
    scanner: "Canon PRISM High-Throughput Digital Pathology",
    scans: 51,
    modalityKey: "topo_mod_histo",
    skewKey: "topo_skew_carcinoma",
    adapter: "FFA-LoRA (r=16, α=32)",
    dpEpsilon: 1.55,
    status: "online",
  },
  {
    id: "node-4",
    name: "Queensland Oncology Screening Center",
    location: "Brisbane, AU",
    scanner: "Philips Digital Dermatoscope Core",
    scans: 53,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_nevi",
    adapter: "FFA-LoRA (r=16, α=32)",
    dpEpsilon: 1.48,
    status: "online",
  },
];

export interface FederatedConsortiumTopologyProps {
  onLaunchDemo?: () => void;
  isModalOpen?: boolean;
}

export default function FederatedConsortiumTopology({
  onLaunchDemo,
}: FederatedConsortiumTopologyProps) {
  const { t } = useLanguage();
  const [activeView, setActiveView] = useState<"topology" | "crypto" | "telemetry">("topology");
  const [selectedNodeId, setSelectedNodeId] = useState<string>("node-1");

  const selectedNode = HOSPITAL_NODES.find((n) => n.id === selectedNodeId) || HOSPITAL_NODES[0];

  return (
    <div className="card relative h-[480px] sm:h-[520px] lg:h-[550px] flex flex-col justify-between overflow-hidden p-4 sm:p-5 select-none border border-[var(--border-card)] bg-[var(--bg-card)] shadow-xl rounded-2xl">
      {/* Top Console Navigation Bar */}
      <div className="relative z-20 flex items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981] shrink-0" />
          <span className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-wider text-[var(--text-heading)] whitespace-nowrap">
            {t("topo_badge_title")}
          </span>
        </div>

        {/* Console View Switcher */}
        <div className="flex items-center p-0.5 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[10px] sm:text-[11px] font-semibold shrink-0">
          <button
            type="button"
            onClick={() => setActiveView("topology")}
            className={`px-2 sm:px-2.5 py-1 rounded-md transition-all whitespace-nowrap ${
              activeView === "topology"
                ? "bg-emerald-500 text-white shadow-sm"
                : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
            }`}
          >
            {t("topo_tab_map")}
          </button>
          <button
            type="button"
            onClick={() => setActiveView("crypto")}
            className={`px-2 sm:px-2.5 py-1 rounded-md transition-all whitespace-nowrap ${
              activeView === "crypto"
                ? "bg-emerald-500 text-white shadow-sm"
                : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
            }`}
          >
            {t("topo_tab_crypto")}
          </button>
          <button
            type="button"
            onClick={() => setActiveView("telemetry")}
            className={`px-2 sm:px-2.5 py-1 rounded-md transition-all whitespace-nowrap ${
              activeView === "telemetry"
                ? "bg-emerald-500 text-white shadow-sm"
                : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
            }`}
          >
            {t("topo_tab_telemetry")}
          </button>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="relative flex-1 w-full my-2 flex items-center justify-center overflow-hidden">
        {/* VIEW 1: FEDERATED TOPOLOGY MAP */}
        {activeView === "topology" && (
          <div className="relative w-full h-full flex flex-col justify-between py-1">
            {/* SVG Connecting Enclave Tunnels */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
              <defs>
                <linearGradient id="streamGrad1" x1="0%" y1="0%" x2="50%" y2="50%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </linearGradient>
                <linearGradient id="streamGrad2" x1="100%" y1="0%" x2="50%" y2="50%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </linearGradient>
                <linearGradient id="streamGrad3" x1="0%" y1="100%" x2="50%" y2="50%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </linearGradient>
                <linearGradient id="streamGrad4" x1="100%" y1="100%" x2="50%" y2="50%">
                  <stop offset="0%" stopColor="#10b981" stopOpacity="0.5" />
                  <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.1" />
                </linearGradient>
              </defs>

              {/* Connecting Tunnels */}
              <line x1="22%" y1="20%" x2="50%" y2="50%" stroke="url(#streamGrad1)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="78%" y1="20%" x2="50%" y2="50%" stroke="url(#streamGrad2)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="22%" y1="80%" x2="50%" y2="50%" stroke="url(#streamGrad3)" strokeWidth="2" strokeDasharray="4 4" />
              <line x1="78%" y1="80%" x2="50%" y2="50%" stroke="url(#streamGrad4)" strokeWidth="2" strokeDasharray="4 4" />

              {/* Aggregation Center Glow Rings */}
              <circle cx="50%" cy="50%" r="58" fill="none" stroke="#10b981" strokeWidth="1" strokeOpacity="0.2" />
              <circle cx="50%" cy="50%" r="76" fill="none" stroke="#06b6d4" strokeWidth="1" strokeOpacity="0.12" strokeDasharray="6 6" />
            </svg>

            {/* Top Row: Nodes 1 & 2 */}
            <div className="grid grid-cols-2 gap-3 z-10">
              {HOSPITAL_NODES.slice(0, 2).map((node) => (
                <button
                  type="button"
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`text-left p-2.5 sm:p-3 rounded-xl border transition-all ${
                    selectedNodeId === node.id
                      ? "bg-[var(--bg-card)] border-emerald-500 shadow-md ring-1 ring-emerald-500/30"
                      : "bg-[var(--bg-card-inner)] hover:bg-[var(--bg-card)] border-[var(--border-subtle)]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400">
                      {node.location}
                    </span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  </div>
                  <div className="font-display text-xs font-bold text-[var(--text-heading)] truncate">
                    {node.name}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)] truncate mt-0.5">
                    {t(node.modalityKey)} · {node.scans.toLocaleString("en-US")} {t("topo_studies")}
                  </div>
                </button>
              ))}
            </div>

            {/* Central Aggregation Ring */}
            <div className="z-10 self-center text-center my-1 sm:my-2">
              <div className="inline-flex flex-col items-center p-3 sm:p-3.5 rounded-2xl bg-[var(--bg-card)] border border-emerald-500/40 shadow-xl backdrop-blur-md">
                <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-mono text-[10px] font-bold uppercase tracking-wider mb-1">
                  <ShieldCheck className="w-3.5 h-3.5" />
                  <span>{t("topo_aggregator_badge")}</span>
                </div>
                <div className="font-display text-xs sm:text-sm font-extrabold text-[var(--text-heading)]">
                  {t("topo_aggregator_title")}
                </div>
                <div className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5">
                  {t("topo_aggregator_sub")}
                </div>
              </div>
            </div>

            {/* Bottom Row: Nodes 3 & 4 */}
            <div className="grid grid-cols-2 gap-3 z-10">
              {HOSPITAL_NODES.slice(2, 4).map((node) => (
                <button
                  type="button"
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`text-left p-2.5 sm:p-3 rounded-xl border transition-all ${
                    selectedNodeId === node.id
                      ? "bg-[var(--bg-card)] border-emerald-500 shadow-md ring-1 ring-emerald-500/30"
                      : "bg-[var(--bg-card-inner)] hover:bg-[var(--bg-card)] border-[var(--border-subtle)]"
                  }`}
                >
                  <div className="flex items-center justify-between mb-1">
                    <span className="font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400">
                      {node.location}
                    </span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                  </div>
                  <div className="font-display text-xs font-bold text-[var(--text-heading)] truncate">
                    {node.name}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)] truncate mt-0.5">
                    {t(node.modalityKey)} · {node.scans.toLocaleString("en-US")} {t("topo_studies")}
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* VIEW 2: CRYPTOGRAPHIC ENCLAVE & DP SIMULATOR */}
        {activeView === "crypto" && (
          <div className="w-full h-full flex flex-col justify-between p-3.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
            <div>
              <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-2 mb-3">
                <div className="flex items-center gap-2">
                  <Lock className="w-4 h-4 text-sky-500" />
                  <span className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)]">
                    {t("topo_crypto_title")}
                  </span>
                </div>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-sky-500/10 text-sky-600 dark:text-sky-400 font-bold border border-sky-500/20">
                  {t("topo_crypto_std")}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    {t("topo_crypto_grad_title")}
                  </div>
                  <div className="font-mono text-xs font-bold text-emerald-600 dark:text-emerald-400 mt-1">
                    ΔW = Bk · A
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-1">
                    {t("topo_crypto_grad_desc")}
                  </div>
                </div>

                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    {t("topo_crypto_state_title")}
                  </div>
                  <div className="font-mono text-xs font-bold text-sky-600 dark:text-sky-400 mt-1">
                    (c0, c1) ∈ R_q^2
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-1">
                    {t("topo_crypto_state_desc")}
                  </div>
                </div>

                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    {t("topo_crypto_dp_title")}
                  </div>
                  <div className="font-mono text-xs font-bold text-violet-600 dark:text-violet-400 mt-1">
                    ε ≤ 2.0, δ = 10^-5
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-1">
                    {t("topo_crypto_dp_desc")}
                  </div>
                </div>
              </div>

              <div className="mt-3 p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px] leading-relaxed text-[var(--text-muted)]">
                <strong className="text-[var(--text-heading)]">{t("topo_crypto_math_title")}</strong>
                <span className="font-mono block mt-0.5 text-emerald-600 dark:text-emerald-400">
                  {"W_global = (1/K) · Σ Enc(B_k) · A  ==>  Dec(Σ Enc(B_k)) · A"}
                </span>
                {t("topo_crypto_math_desc")}
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] border-t border-[var(--border-subtle)] pt-2 mt-2">
              <span>{t("topo_crypto_byzantine")}</span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {t("topo_crypto_transit")}
              </span>
            </div>
          </div>
        )}

        {/* VIEW 3: CONSENSUS TELEMETRY & BYZANTINE DEFENSE */}
        {activeView === "telemetry" && (
          <div className="w-full h-full flex flex-col justify-between p-3.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
            <div>
              <div className="flex items-center justify-between border-b border-[var(--border-subtle)] pb-2 mb-3">
                <div className="flex items-center gap-2">
                  <Activity className="w-4 h-4 text-emerald-500" />
                  <span className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)]">
                    {t("topo_telem_title")}
                  </span>
                </div>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/20">
                  {t("topo_telem_converged")}
                </span>
              </div>

              <div className="grid grid-cols-1 sm:grid-cols-3 gap-2.5 text-xs">
                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    <TrendingDown className="w-3 h-3 text-emerald-500" />
                    <span>{t("topo_telem_payload_title")}</span>
                  </div>
                  <div className="font-mono text-sm font-bold text-emerald-600 dark:text-emerald-400 mt-1">
                    4.2 MB
                  </div>
                  <div className="text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5">
                    {t("topo_telem_payload_desc")}
                  </div>
                </div>

                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    <AlertTriangle className="w-3 h-3 text-amber-500" />
                    <span>{t("topo_telem_defense_title")}</span>
                  </div>
                  <div className="font-mono text-sm font-bold text-amber-600 dark:text-amber-400 mt-1">
                    Trimmed-Mean
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                    {t("topo_telem_defense_desc")}
                  </div>
                </div>

                <div className="card p-2.5 border border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-slate-500 uppercase font-semibold">
                    <FileCheck className="w-3 h-3 text-sky-500" />
                    <span>{t("topo_telem_auc_title")}</span>
                  </div>
                  <div className="font-mono text-sm font-bold text-sky-600 dark:text-sky-400 mt-1">
                    0.984
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                    {t("topo_telem_auc_desc")}
                  </div>
                </div>
              </div>

              <div className="mt-3 p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px] leading-relaxed text-[var(--text-muted)]">
                <div className="flex items-center justify-between mb-1.5">
                  <strong className="text-[var(--text-heading)]">{t("topo_telem_log_title")}</strong>
                  <span className="font-mono text-[10px] text-emerald-500 font-semibold">{t("topo_telem_loss")}</span>
                </div>
                <div className="flex items-center gap-2 font-mono text-[10px]">
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R1: 72.5%</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R2: 81.3%</span>
                  <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 font-bold">R3: 87.2% ({t("topo_telem_filtered")})</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R4: 91.6%</span>
                  <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-extrabold">R5: 94.8%</span>
                </div>
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] border-t border-[var(--border-subtle)] pt-2 mt-2">
              <span>{t("topo_telem_drift")}</span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {t("topo_telem_divergence")}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Bottom Node Inspector Bar */}
      {activeView === "topology" && (
        <div className="relative z-20 p-2.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-xs mt-1">
          <div className="flex items-center justify-between gap-2">
            {/* Left: Server icon + Hospital Name + Location */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <div className="w-6 h-6 rounded-lg bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-500 shrink-0">
                <Server className="w-3.5 h-3.5" />
              </div>
              <span className="font-display font-bold text-[var(--text-heading)] text-xs truncate">
                {selectedNode.name}
              </span>
              <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 shrink-0">
                {selectedNode.location}
              </span>
            </div>

            {/* Right: DP Budget & Simulate */}
            <div className="flex items-center gap-2 shrink-0 text-[10px] font-mono">
              <div className="hidden sm:inline-flex items-center px-2 py-0.5 rounded bg-[var(--bg-card)] border border-[var(--border-subtle)]">
                <span className="text-slate-400 mr-1">{t("topo_dp_budget")}</span>
                <span className="font-bold text-emerald-600 dark:text-emerald-400">ε={selectedNode.dpEpsilon}</span>
              </div>
              {onLaunchDemo && (
                <button
                  type="button"
                  onClick={onLaunchDemo}
                  className="px-2.5 py-1 rounded-md bg-emerald-500 hover:bg-emerald-600 text-white font-bold text-[10px] sm:text-[11px] transition-all shadow-sm"
                >
                  {t("topo_simulate")}
                </button>
              )}
            </div>
          </div>

          {/* Subline: Scanner · Skew */}
          <div className="text-[10px] font-mono text-[var(--text-muted)] truncate mt-1 pt-1 border-t border-[var(--border-subtle)] flex items-center gap-1.5">
            <span className="truncate">{selectedNode.scanner}</span>
            <span className="text-slate-400 shrink-0">·</span>
            <span className="text-amber-600 dark:text-amber-400 font-semibold shrink-0">
              {t("topo_skew_label")} {t(selectedNode.skewKey)}
            </span>
          </div>
        </div>
      )}
    </div>
  );
}
