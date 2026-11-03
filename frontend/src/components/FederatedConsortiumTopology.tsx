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
  Zap,
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import Consortium3DCanvas from "@/components/Consortium3DCanvas";

export interface HospitalNode {
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
    scans: 40,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_melanoma",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 1.42,
    status: "online",
  },
  {
    id: "node-2",
    name: "Beth Israel Deaconess Medical Center",
    location: "Boston, USA",
    scanner: "GE Revolution Digital Radiography (MIMIC-CXR)",
    scans: 40,
    modalityKey: "topo_mod_xray",
    skewKey: "topo_skew_pneumonia",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 1.38,
    status: "online",
  },
  {
    id: "node-3",
    name: "ViDIR Dermatopathology Institute",
    location: "Vienna, AT",
    scanner: "Canon PRISM High-Throughput Digital Pathology",
    scans: 40,
    modalityKey: "topo_mod_histo",
    skewKey: "topo_skew_carcinoma",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 1.55,
    status: "online",
  },
  {
    id: "node-4",
    name: "Queensland Oncology Screening Center",
    location: "Brisbane, AU",
    scanner: "Philips Digital Dermatoscope Core",
    scans: 40,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_nevi",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 1.48,
    status: "online",
  },
];

export interface FederatedConsortiumTopologyProps {
  onLaunchDemo?: () => void;
  isModalOpen?: boolean;
  hospitals?: HospitalNode[];
  activeEpsilon?: number;
  telemetryRounds?: {
    round_num: number;
    train_loss: number;
    train_accuracy: number;
    f1_score: number;
    bandwidth: string;
    threat_detected: boolean;
    status: string;
  }[];
  isLive?: boolean;
}

export default function FederatedConsortiumTopology({
  onLaunchDemo,
  hospitals,
  activeEpsilon,
  telemetryRounds,
  isLive = false,
}: FederatedConsortiumTopologyProps) {
  const { t } = useLanguage();
  const [activeView, setActiveView] = useState<"topology" | "crypto" | "telemetry">("topology");
  const [selectedNodeId, setSelectedNodeId] = useState<string>("node-1");
  const [is3DMode, setIs3DMode] = useState<boolean>(true);
  const [isSimulatingRound, setIsSimulatingRound] = useState<boolean>(false);
  const [mouseTilt, setMouseTilt] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isHovered, setIsHovered] = useState<boolean>(false);
  const [roundNotification, setRoundNotification] = useState<string | null>(null);

  const nodeList = hospitals && hospitals.length > 0 ? hospitals : HOSPITAL_NODES;
  const selectedNode = nodeList.find((n) => n.id === selectedNodeId) || nodeList[0];

  const triggerConsensusRound = () => {
    if (isSimulatingRound) return;
    setIsSimulatingRound(true);
    setRoundNotification(t("topo_simulating_round"));
    setTimeout(() => {
      setRoundNotification(t("topo_round_synced"));
      setTimeout(() => {
        setRoundNotification(null);
        setIsSimulatingRound(false);
      }, 2000);
    }, 2200);
  };

  return (
    <div className="card relative min-h-[520px] sm:min-h-[530px] lg:h-[560px] flex flex-col justify-between overflow-hidden p-3.5 sm:p-5 select-none border border-[var(--border-card)] bg-[var(--bg-card)] shadow-xl rounded-2xl">
      {/* Top Console Navigation Bar */}
      <div className="relative z-20 flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-[var(--border-subtle)] pb-2.5">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981] shrink-0" />
          <span className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-wider text-[var(--text-heading)]">
            {t("topo_badge_title")}
          </span>
        </div>

        {/* Console Controls: 3D Toggle, Action Button, View Switcher */}
        <div className="flex flex-wrap items-center gap-1.5 sm:gap-2 w-full sm:w-auto justify-between sm:justify-end">
          {activeView === "topology" && (
            <div className="flex items-center gap-1">
              {/* 3D Mode Toggle Button */}
              <button
                type="button"
                onClick={() => setIs3DMode((prev) => !prev)}
                className={`px-2 py-1 rounded-md text-[10px] font-mono font-bold transition-all border ${
                  is3DMode
                    ? "bg-emerald-500/15 border-emerald-500/40 text-emerald-600 dark:text-emerald-400 shadow-xs"
                    : "bg-[var(--bg-card-inner)] border-[var(--border-card)] text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
                title={is3DMode ? t("topo_view_2d") : t("topo_view_3d")}
              >
                {is3DMode ? "3D" : "2D"}
              </button>

              {/* Trigger Encrypted Consensus Round */}
              <button
                type="button"
                onClick={triggerConsensusRound}
                disabled={isSimulatingRound}
                className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs transition-all disabled:opacity-50 active:scale-95"
              >
                <Zap className={`w-3 h-3 ${isSimulatingRound ? "animate-spin text-amber-300" : ""}`} />
                <span className="hidden sm:inline">
                  {isSimulatingRound ? t("topo_simulating_round") : t("topo_trigger_round")}
                </span>
                <span className="sm:hidden">
                  {isSimulatingRound ? "Sync..." : "Round"}
                </span>
              </button>
            </div>
          )}

          {/* Console View Switcher */}
          <div className="grid grid-cols-3 sm:flex items-center p-0.5 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[10px] sm:text-[11px] font-semibold w-full sm:w-auto">
            <button
              type="button"
              onClick={() => setActiveView("topology")}
              className={`px-2 sm:px-2.5 py-1 text-center rounded-md transition-all whitespace-nowrap ${
                activeView === "topology"
                  ? "bg-emerald-500 text-white shadow-sm font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_map")}
            </button>
            <button
              type="button"
              onClick={() => setActiveView("crypto")}
              className={`px-2 sm:px-2.5 py-1 text-center rounded-md transition-all whitespace-nowrap ${
                activeView === "crypto"
                  ? "bg-emerald-500 text-white shadow-sm font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_crypto")}
            </button>
            <button
              type="button"
              onClick={() => setActiveView("telemetry")}
              className={`px-2 sm:px-2.5 py-1 text-center rounded-md transition-all whitespace-nowrap ${
                activeView === "telemetry"
                  ? "bg-emerald-500 text-white shadow-sm font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_telemetry")}
            </button>
          </div>
        </div>
      </div>

      {/* Main Interactive Stage */}
      <div className="relative flex-1 w-full my-2 flex items-center justify-center overflow-hidden">
        {/* VIEW 1: FEDERATED TOPOLOGY MAP (3D PERSPECTIVE + ENCRYPTED CANVAS) */}
        {activeView === "topology" && (
          <div
            className="relative w-full h-full flex flex-col justify-between py-1"
            style={{ perspective: "1100px" }}
            onMouseMove={(e) => {
              if (!is3DMode) return;
              const rect = e.currentTarget.getBoundingClientRect();
              const x = (e.clientX - rect.left) / rect.width - 0.5;
              const y = (e.clientY - rect.top) / rect.height - 0.5;
              setMouseTilt({ x, y });
            }}
            onMouseEnter={() => setIsHovered(true)}
            onMouseLeave={() => {
              setIsHovered(false);
              setMouseTilt({ x: 0, y: 0 });
            }}
          >
            {/* 3D Encrypted Canvas Engine */}
            <Consortium3DCanvas
              is3DMode={is3DMode}
              isSimulatingRound={isSimulatingRound}
              selectedNodeId={selectedNodeId}
              mouseTilt={mouseTilt}
              onSimulationComplete={() => setIsSimulatingRound(false)}
            />

            {/* Notification Toast during consensus synchronization */}
            {roundNotification && (
              <div className="absolute top-2 left-1/2 -translate-x-1/2 z-30 px-3 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-lg animate-in fade-in flex items-center gap-1.5 whitespace-nowrap">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-ping shrink-0" />
                <span>{roundNotification}</span>
              </div>
            )}

            {/* 3D Perspective Stage Container */}
            <div
              className="relative w-full h-full flex flex-col justify-between"
              style={{
                transform: is3DMode
                  ? `rotateX(${13 + mouseTilt.y * 14}deg) rotateY(${mouseTilt.x * 12}deg)`
                  : "none",
                transformStyle: "preserve-3d",
                transition: isHovered
                  ? "transform 0.08s ease-out"
                  : "transform 0.6s cubic-bezier(0.2, 0.8, 0.2, 1)",
              }}
            >
              {/* Top Row: Nodes 1 & 2 */}
              <div className="grid grid-cols-2 gap-2 sm:gap-3 z-10" style={{ transformStyle: "preserve-3d" }}>
                {nodeList.slice(0, 2).map((node) => {
                  const isSelected = selectedNodeId === node.id;
                  return (
                    <button
                      type="button"
                      key={node.id}
                      onClick={() => setSelectedNodeId(node.id)}
                      style={{
                        transform: is3DMode
                          ? isSelected
                            ? "translateZ(38px)"
                            : "translateZ(20px)"
                          : "none",
                        transition: "all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1)",
                      }}
                      className={`text-left p-2 sm:p-2.5 lg:p-3 rounded-xl border backdrop-blur-md transition-all ${
                        isSelected
                          ? "bg-[var(--bg-card)]/95 border-emerald-500 shadow-xl shadow-emerald-500/10 ring-2 ring-emerald-500/40"
                          : "bg-[var(--bg-card-inner)]/85 hover:bg-[var(--bg-card)] border-[var(--border-subtle)] hover:border-emerald-500/40"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400 truncate pr-1">
                          {node.location}
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {isSelected && (
                            <span className="text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/30">
                              Active
                            </span>
                          )}
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                        </div>
                      </div>
                      <div className="font-display text-xs font-bold text-[var(--text-heading)] truncate">
                        {node.name}
                      </div>
                      <div className="text-[10px] font-mono text-[var(--text-muted)] truncate mt-0.5 flex items-center justify-between">
                        <span>{t(node.modalityKey)} · {(node.scans ?? 0).toLocaleString("en-US")} {t("topo_studies")}</span>
                        <span className="text-emerald-600/70 dark:text-emerald-400/70 font-mono text-[9px] hidden sm:inline">
                          ε={node.dpEpsilon}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>

              {/* Central Aggregation Vault in 3D */}
              <div
                className="z-10 self-center text-center my-1"
                style={{
                  transform: is3DMode ? "translateZ(54px)" : "none",
                  transformStyle: "preserve-3d",
                  transition: "transform 0.4s ease-out",
                }}
              >
                <div className="relative inline-flex flex-col items-center p-3 sm:p-4 rounded-2xl bg-[var(--bg-card)]/95 border border-emerald-500/50 shadow-2xl backdrop-blur-lg ring-1 ring-emerald-500/20 group">
                  <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-mono text-[10px] font-bold uppercase tracking-wider mb-0.5">
                    <ShieldCheck className="w-4 h-4 shrink-0" />
                    <span>{t("topo_aggregator_badge")}</span>
                    <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse ml-0.5" />
                  </div>
                  <div className="font-display text-xs sm:text-sm font-extrabold text-[var(--text-heading)]">
                    {t("topo_aggregator_title")}
                  </div>
                  <div className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5">
                    {t("topo_aggregator_sub")}
                  </div>
                  <div className="mt-1.5 text-[9px] font-mono text-emerald-600 dark:text-emerald-400 px-2 py-0.5 rounded-full bg-emerald-500/10 border border-emerald-500/20">
                    {t("topo_stream_active")}
                  </div>
                </div>
              </div>

              {/* Bottom Row: Nodes 3 & 4 */}
              <div className="grid grid-cols-2 gap-2 sm:gap-3 z-10" style={{ transformStyle: "preserve-3d" }}>
                {nodeList.slice(2, 4).map((node) => {
                  const isSelected = selectedNodeId === node.id;
                  return (
                    <button
                      type="button"
                      key={node.id}
                      onClick={() => setSelectedNodeId(node.id)}
                      style={{
                        transform: is3DMode
                          ? isSelected
                            ? "translateZ(38px)"
                            : "translateZ(20px)"
                          : "none",
                        transition: "all 0.3s cubic-bezier(0.2, 0.8, 0.2, 1)",
                      }}
                      className={`text-left p-2 sm:p-2.5 lg:p-3 rounded-xl border backdrop-blur-md transition-all ${
                        isSelected
                          ? "bg-[var(--bg-card)]/95 border-emerald-500 shadow-xl shadow-emerald-500/10 ring-2 ring-emerald-500/40"
                          : "bg-[var(--bg-card-inner)]/85 hover:bg-[var(--bg-card)] border-[var(--border-subtle)] hover:border-emerald-500/40"
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className="font-mono text-[10px] font-bold text-emerald-600 dark:text-emerald-400 truncate pr-1">
                          {node.location}
                        </span>
                        <div className="flex items-center gap-1.5 shrink-0">
                          {isSelected && (
                            <span className="text-[9px] font-mono uppercase px-1 py-0.2 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/30">
                              Active
                            </span>
                          )}
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                        </div>
                      </div>
                      <div className="font-display text-xs font-bold text-[var(--text-heading)] truncate">
                        {node.name}
                      </div>
                      <div className="text-[10px] font-mono text-[var(--text-muted)] truncate mt-0.5 flex items-center justify-between">
                        <span>{t(node.modalityKey)} · {(node.scans ?? 0).toLocaleString("en-US")} {t("topo_studies")}</span>
                        <span className="text-emerald-600/70 dark:text-emerald-400/70 font-mono text-[9px] hidden sm:inline">
                          ε={node.dpEpsilon}
                        </span>
                      </div>
                    </button>
                  );
                })}
              </div>
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
                    <span>{telemetryRounds && telemetryRounds.length > 0 ? "Accuracy" : t("topo_telem_auc_title")}</span>
                  </div>
                  <div className="font-mono text-sm font-bold text-sky-600 dark:text-sky-400 mt-1">
                    {telemetryRounds && telemetryRounds.length > 0
                      ? `${telemetryRounds[telemetryRounds.length - 1].train_accuracy}%`
                      : "0.984"}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                    {telemetryRounds && telemetryRounds.length > 0
                      ? `Live Round ${telemetryRounds.length} Validation`
                      : t("topo_telem_auc_desc")}
                  </div>
                </div>
              </div>

              <div className="mt-3 p-3 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px] leading-relaxed text-[var(--text-muted)]">
                <div className="flex items-center justify-between mb-1.5">
                  <strong className="text-[var(--text-heading)]">
                    {Array.isArray(telemetryRounds) && telemetryRounds.length > 0
                      ? "Live Round Trajectory"
                      : t("topo_telem_log_title")}
                  </strong>
                  <span className="font-mono text-[10px] text-emerald-500 font-semibold">
                    {Array.isArray(telemetryRounds) && telemetryRounds.length > 0
                      ? `Loss: ${telemetryRounds[telemetryRounds.length - 1].train_loss}`
                      : t("topo_telem_loss")}
                  </span>
                </div>
                <div className="flex flex-wrap items-center gap-1.5 font-mono text-[10px]">
                  {Array.isArray(telemetryRounds) && telemetryRounds.length > 0 ? (
                    telemetryRounds.map((rd) => (
                      <span
                        key={rd.round_num}
                        className={`px-2 py-0.5 rounded font-bold ${
                          rd.threat_detected
                            ? "bg-amber-500/10 text-amber-600 dark:text-amber-400"
                            : "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                        }`}
                      >
                        R{rd.round_num}: {rd.train_accuracy}%
                      </span>
                    ))
                  ) : (
                    <>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R1: 72.5%</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R2: 81.3%</span>
                      <span className="px-2 py-0.5 rounded bg-amber-500/10 text-amber-600 dark:text-amber-400 font-bold">R3: 87.2% ({t("topo_telem_filtered")})</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">R4: 91.6%</span>
                      <span className="px-2 py-0.5 rounded bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-extrabold">R5: 94.8%</span>
                    </>
                  )}
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
              <div className="min-w-0 flex items-center gap-1.5 flex-wrap">
                <span className="font-display font-bold text-[var(--text-heading)] text-xs truncate max-w-[170px] sm:max-w-none">
                  {selectedNode.name}
                </span>
                <span className="text-[10px] font-mono px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 shrink-0">
                  {selectedNode.location}
                </span>
              </div>
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
                  onClick={() => {
                    triggerConsensusRound();
                    if (onLaunchDemo) onLaunchDemo();
                  }}
                  className="px-2.5 py-1 rounded-md bg-emerald-500 hover:bg-emerald-600 text-white font-bold text-[10px] sm:text-[11px] transition-all shadow-sm"
                >
                  {t("topo_simulate")}
                </button>
              )}
            </div>
          </div>

          {/* Subline: Scanner & Skew Details */}
          <div className="mt-2 pt-2 border-t border-[var(--border-subtle)] flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 text-[10px] font-mono">
            <span className="text-[var(--text-muted)] leading-tight break-words">
              {selectedNode.scanner}
            </span>
            <div className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-400 font-semibold bg-amber-500/10 px-2 py-0.5 rounded border border-amber-500/20 text-[10px] w-fit max-w-full">
              <span className="shrink-0">{t("topo_skew_label")}</span>
              <span className="break-words">{t(selectedNode.skewKey)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
