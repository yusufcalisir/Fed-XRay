"use client";

import React, { useState } from "react";
import {
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
  telemetryRounds,
}: FederatedConsortiumTopologyProps) {
  const { t } = useLanguage();
  const [activeView, setActiveView] = useState<"topology" | "crypto" | "telemetry">("topology");
  const [selectedNodeId, setSelectedNodeId] = useState<string>("node-1");
  const [isSimulatingRound, setIsSimulatingRound] = useState<boolean>(false);
  const [mouseTilt, setMouseTilt] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
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

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width - 0.5;
    const y = (e.clientY - rect.top) / rect.height - 0.5;
    setMouseTilt({ x, y });
  };

  return (
    <div className="relative w-full min-h-[500px] sm:min-h-[520px] lg:h-[550px] flex flex-col justify-between select-none bg-transparent">
      {/* Floating Top Console Navigation Bar */}
      <div className="relative z-30 flex flex-col sm:flex-row sm:items-center justify-between gap-2 p-2 sm:px-3.5 sm:py-2 rounded-2xl bg-[var(--bg-card)]/75 dark:bg-slate-900/75 border border-[var(--border-card)] shadow-xs backdrop-blur-md">
        <div className="flex items-center gap-2 min-w-0">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_10px_#10b981] shrink-0" />
          <div className="min-w-0">
            <span className="font-mono text-[11px] sm:text-xs font-bold uppercase tracking-wider text-[var(--text-heading)] block truncate">
              {t("topo_badge_title")}
            </span>
          </div>
        </div>

        {/* Action Controls & View Switcher */}
        <div className="flex flex-wrap items-center gap-2 w-full sm:w-auto justify-between sm:justify-end">
          {activeView === "topology" && (
            <button
              type="button"
              onClick={triggerConsensusRound}
              disabled={isSimulatingRound}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-[11px] font-mono font-bold bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs shadow-emerald-600/20 transition-all disabled:opacity-50 active:scale-95 border border-emerald-500/30"
            >
              <Zap className={`w-3.5 h-3.5 ${isSimulatingRound ? "animate-spin text-amber-200" : ""}`} />
              <span>
                {isSimulatingRound ? t("topo_simulating_round") : t("topo_trigger_round")}
              </span>
            </button>
          )}

          {/* View Mode Switcher */}
          <div className="grid grid-cols-3 sm:flex items-center p-1 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[10px] sm:text-[11px] font-semibold w-full sm:w-auto">
            <button
              type="button"
              onClick={() => setActiveView("topology")}
              className={`px-2.5 py-1 text-center rounded-lg transition-all whitespace-nowrap ${
                activeView === "topology"
                  ? "bg-emerald-600 text-white shadow-xs font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_map")}
            </button>
            <button
              type="button"
              onClick={() => setActiveView("crypto")}
              className={`px-2.5 py-1 text-center rounded-lg transition-all whitespace-nowrap ${
                activeView === "crypto"
                  ? "bg-emerald-600 text-white shadow-xs font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_crypto")}
            </button>
            <button
              type="button"
              onClick={() => setActiveView("telemetry")}
              className={`px-2.5 py-1 text-center rounded-lg transition-all whitespace-nowrap ${
                activeView === "telemetry"
                  ? "bg-emerald-600 text-white shadow-xs font-bold"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              {t("topo_tab_telemetry")}
            </button>
          </div>
        </div>
      </div>

      {/* Main Spatial Stage (Completely transparent background so page grid lines flow seamlessly!) */}
      <div className="relative flex-1 w-full my-1.5 flex items-center justify-center overflow-hidden bg-transparent">
        {/* VIEW 1: 3D CONSORTIUM (Authentic 3D Hospitals + 3D Central Server) */}
        {activeView === "topology" && (
          <div
            className="relative w-full h-[370px] sm:h-[400px] flex items-center justify-center overflow-hidden bg-transparent"
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setMouseTilt({ x: 0, y: 0 })}
          >
            {/* 3D Geometric Canvas: Transparent, sits directly on top of page background grid */}
            <Consortium3DCanvas
              isSimulatingRound={isSimulatingRound}
              selectedNodeId={selectedNodeId}
              mouseTilt={mouseTilt}
              onSelectNode={(id) => setSelectedNodeId(id)}
              onSimulationComplete={() => setIsSimulatingRound(false)}
            />

            {/* Notification Toast during consensus round */}
            {roundNotification && (
              <div className="absolute top-3 left-1/2 -translate-x-1/2 z-30 px-3.5 py-1 rounded-full text-[10px] font-mono font-bold bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 border border-emerald-500/40 backdrop-blur-md shadow-md flex items-center gap-1.5 whitespace-nowrap">
                <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping shrink-0" />
                <span>{roundNotification}</span>
              </div>
            )}
          </div>
        )}

        {/* VIEW 2: CRYPTOGRAPHIC ENCLAVE & DP PROTOCOL */}
        {activeView === "crypto" && (
          <div className="w-full h-full flex flex-col justify-between p-4 rounded-2xl bg-[var(--bg-card)]/90 dark:bg-slate-900/90 border border-[var(--border-card)] text-[var(--text-main)] shadow-sm">
            <div>
              <div className="flex items-center justify-between border-b border-[var(--border-card)] pb-2 mb-3">
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
                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
                    {t("topo_crypto_grad_title")}
                  </div>
                  <div className="font-mono text-xs font-bold text-emerald-600 dark:text-emerald-400 mt-1">
                    ΔW = Bk · A
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-1">
                    {t("topo_crypto_grad_desc")}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
                    {t("topo_crypto_state_title")}
                  </div>
                  <div className="font-mono text-xs font-bold text-sky-600 dark:text-sky-400 mt-1">
                    (c0, c1) ∈ R_q^2
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-1">
                    {t("topo_crypto_state_desc")}
                  </div>
                </div>

                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
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

              <div className="mt-3 p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[11px] leading-relaxed text-[var(--text-main)]">
                <strong className="text-[var(--text-heading)]">{t("topo_crypto_math_title")}</strong>
                <span className="font-mono block mt-0.5 text-emerald-600 dark:text-emerald-400">
                  {"W_global = (1/K) · Σ Enc(B_k) · A  ==>  Dec(Σ Enc(B_k)) · A"}
                </span>
                {t("topo_crypto_math_desc")}
              </div>
            </div>

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] border-t border-[var(--border-card)] pt-2 mt-2">
              <span>{t("topo_crypto_byzantine")}</span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {t("topo_crypto_transit")}
              </span>
            </div>
          </div>
        )}

        {/* VIEW 3: CONSENSUS TELEMETRY & BYZANTINE DEFENSE */}
        {activeView === "telemetry" && (
          <div className="w-full h-full flex flex-col justify-between p-4 rounded-2xl bg-[var(--bg-card)]/90 dark:bg-slate-900/90 border border-[var(--border-card)] text-[var(--text-main)] shadow-sm">
            <div>
              <div className="flex items-center justify-between border-b border-[var(--border-card)] pb-2 mb-3">
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
                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
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

                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
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

                <div className="p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                  <div className="flex items-center gap-1.5 text-[10px] font-mono text-[var(--text-muted)] uppercase font-semibold">
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

              <div className="mt-3 p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[11px] leading-relaxed text-[var(--text-main)]">
                <div className="flex items-center justify-between mb-1.5">
                  <strong className="text-[var(--text-heading)]">
                    {Array.isArray(telemetryRounds) && telemetryRounds.length > 0
                      ? "Live Round Trajectory"
                      : t("topo_telem_log_title")}
                  </strong>
                  <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
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
                            ? "bg-amber-500/10 text-amber-500"
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

            <div className="flex items-center justify-between text-[11px] text-[var(--text-muted)] border-t border-[var(--border-card)] pt-2 mt-2">
              <span>{t("topo_telem_drift")}</span>
              <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">
                {t("topo_telem_divergence")}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Docked Minimal Telemetry Inspector (Theme-Harmonized Glass Ribbon) */}
      {activeView === "topology" && (
        <div className="relative z-30 p-2.5 sm:p-3 rounded-2xl bg-[var(--bg-card)]/80 dark:bg-slate-900/80 border border-[var(--border-card)] text-xs backdrop-blur-md shadow-xs">
          <div className="flex items-center justify-between gap-2">
            {/* Left: Selected Hospital & Location */}
            <div className="flex items-center gap-2 min-w-0 flex-1">
              <div className="w-7 h-7 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shrink-0 shadow-xs">
                <Server className="w-4 h-4" />
              </div>
              <div className="min-w-0 flex items-center gap-1.5 flex-wrap">
                <span className="font-display font-bold text-[var(--text-heading)] text-xs truncate max-w-[170px] sm:max-w-none">
                  {selectedNode.name}
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 shrink-0">
                  {selectedNode.location}
                </span>
              </div>
            </div>

            {/* Right: DP Budget & Quick Trigger Action */}
            <div className="flex items-center gap-2 shrink-0 text-[10px] font-mono">
              <div className="hidden sm:inline-flex items-center px-2 py-0.5 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)]">
                <span className="text-[var(--text-muted)] mr-1">{t("topo_dp_budget")}</span>
                <span className="font-bold text-emerald-600 dark:text-emerald-400">ε={selectedNode.dpEpsilon}</span>
              </div>
              {onLaunchDemo && (
                <button
                  type="button"
                  onClick={() => {
                    triggerConsensusRound();
                    if (onLaunchDemo) onLaunchDemo();
                  }}
                  className="px-3 py-1 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-[10px] sm:text-[11px] transition-all shadow-sm shadow-emerald-600/20 active:scale-95 border border-emerald-500/30"
                >
                  {t("topo_simulate")}
                </button>
              )}
            </div>
          </div>

          {/* Subline: Scanner Hardware & Skew */}
          <div className="mt-2 pt-2 border-t border-[var(--border-card)] flex flex-col sm:flex-row sm:items-center justify-between gap-1.5 text-[10px] font-mono">
            <span className="text-[var(--text-muted)] leading-tight break-words truncate">
              {selectedNode.scanner}
            </span>
            <div className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-300 font-semibold bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20 text-[10px] w-fit max-w-full">
              <span className="shrink-0">{t("topo_skew_label")}</span>
              <span className="break-words">{t(selectedNode.skewKey)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
