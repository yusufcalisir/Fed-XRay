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
import FederatedTopologyMap from "@/components/FederatedTopologyMap";

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
  status: "online" | "syncing" | "standby";
}

const getLocalizedNodeName = (node: HospitalNode, t: (k: string) => string) => {
  const key = `topo_${node.id.replace("-", "")}_name`;
  const translated = t(key);
  return translated !== key ? translated : node.name;
};

const getLocalizedNodeLocation = (node: HospitalNode, t: (k: string) => string) => {
  const key = `topo_${node.id.replace("-", "")}_loc`;
  const translated = t(key);
  return translated !== key ? translated : node.location;
};

const getLocalizedNodeScanner = (node: HospitalNode, t: (k: string) => string) => {
  const key = `topo_${node.id.replace("-", "")}_scanner`;
  const translated = t(key);
  return translated !== key ? translated : node.scanner;
};

const HOSPITAL_NODES: HospitalNode[] = [
  {
    id: "node-1",
    name: "BCN-20000 Skin Cancer Hub",
    location: "Barcelona, ES",
    scanner: "Siemens Sensation Optical Dermatoscope",
    scans: 0,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_melanoma",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 0.0,
    status: "standby",
  },
  {
    id: "node-3",
    name: "ViDIR Dermatopathology Institute",
    location: "Vienna, AT",
    scanner: "Canon PRISM High-Throughput Digital Pathology",
    scans: 0,
    modalityKey: "topo_mod_histo",
    skewKey: "topo_skew_carcinoma",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 0.0,
    status: "standby",
  },
  {
    id: "node-4",
    name: "Queensland Oncology Screening Center",
    location: "Brisbane, AU",
    scanner: "Philips Digital Dermatoscope Core",
    scans: 0,
    modalityKey: "topo_mod_dermo",
    skewKey: "topo_skew_nevi",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 0.0,
    status: "standby",
  },
  {
    id: "node-2",
    name: "Beth Israel Deaconess Medical Center",
    location: "Boston, USA",
    scanner: "GE Revolution Digital Radiography (MIMIC-CXR)",
    scans: 0,
    modalityKey: "topo_mod_xray",
    skewKey: "topo_skew_pneumonia",
    adapter: "FFA-LoRA (r=8, alpha=16)",
    dpEpsilon: 0.0,
    status: "standby",
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
    uplink_bytes?: number;
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
    <div className="relative w-full flex flex-col justify-between select-none rounded-3xl p-2.5 sm:p-4 bg-[var(--bg-card)]/35 dark:bg-slate-900/40 border border-[var(--border-card)] shadow-xs backdrop-blur-xs">
      {/* Floating Top Console Navigation Bar */}
      <div className="relative z-30 flex items-center justify-between gap-1.5 sm:gap-2 p-1.5 sm:px-3.5 sm:py-2 rounded-2xl bg-[var(--bg-card)]/80 dark:bg-slate-900/80 border border-[var(--border-card)] shadow-xs backdrop-blur-md">
        <div className="flex items-center gap-1.5 sm:gap-2 min-w-0 shrink">
          <div className="w-2.5 h-2.5 rounded-full bg-emerald-500 animate-pulse shadow-[0_0_8px_#10b981] shrink-0" />
          <span className="font-mono text-[10px] sm:text-xs font-bold uppercase tracking-wider text-[var(--text-heading)] truncate">
            <span className="hidden sm:inline">{t("topo_badge_title")}</span>
            <span className="sm:hidden">{t("topo_badge_title_short")}</span>
          </span>
        </div>

        {/* Action Controls & View Switcher */}
        <div className="flex items-center gap-1 sm:gap-2 shrink-0">
          {activeView === "topology" && (
            <button
              type="button"
              onClick={triggerConsensusRound}
              disabled={isSimulatingRound}
              title={t("topo_trigger_round")}
              aria-label={t("topo_trigger_round")}
              className="w-7 h-7 sm:w-8 sm:h-8 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white shadow-xs shadow-emerald-600/20 transition-all disabled:opacity-50 active:scale-95 border border-emerald-500/30 flex items-center justify-center shrink-0 cursor-pointer"
            >
              <Zap className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isSimulatingRound ? "animate-spin text-amber-200" : "text-white"}`} />
            </button>
          )}

          {/* View Mode Switcher */}
          <div className="flex items-center p-0.5 sm:p-1 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[10px] sm:text-[11px] font-semibold">
            <button
              type="button"
              onClick={() => setActiveView("topology")}
              className={`px-1.5 sm:px-2.5 py-0.5 sm:py-1 text-center rounded-lg transition-all whitespace-nowrap ${
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
              className={`px-1.5 sm:px-2.5 py-0.5 sm:py-1 text-center rounded-lg transition-all whitespace-nowrap ${
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
              className={`px-1.5 sm:px-2.5 py-0.5 sm:py-1 text-center rounded-lg transition-all whitespace-nowrap ${
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

      {/* Consensus Round Status Ribbon (Zero Overlap, Guaranteed Visibility) */}
      <div className="relative z-20 w-full min-h-[26px] my-1 flex items-center justify-center">
        {roundNotification ? (
          <div className="px-3 py-0.5 rounded-full text-[10px] sm:text-[11px] font-mono font-bold bg-emerald-500/20 text-emerald-700 dark:text-emerald-300 border border-emerald-500/50 backdrop-blur-md shadow-xs flex items-center gap-1.5 whitespace-nowrap animate-in fade-in zoom-in-95 duration-150">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-ping shrink-0" />
            <span>{roundNotification}</span>
          </div>
        ) : (
          <div className="text-[9.5px] font-mono text-slate-400 dark:text-slate-500 flex items-center gap-1.5 opacity-80 select-none">
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-500/60" />
            <span>{t("topo_stream_active")}</span>
          </div>
        )}
      </div>

      {/* Main Spatial Stage */}
      <div className="relative w-full my-1 flex items-center justify-center bg-transparent">
        {/* VIEW 1: TOPOLOGY MAP */}
        {activeView === "topology" && (
          <div
            className="relative w-full flex items-center justify-center bg-transparent"
            onMouseMove={handleMouseMove}
            onMouseLeave={() => setMouseTilt({ x: 0, y: 0 })}
          >
            {/* Enterprise Architectural Topology Map */}
            <FederatedTopologyMap
              selectedNodeId={selectedNodeId}
              isSimulatingRound={isSimulatingRound}
              onSelectNode={(id) => setSelectedNodeId(id)}
            />
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
                    <span>{telemetryRounds && telemetryRounds.length > 0 ? t("topo_accuracy") : t("topo_telem_auc_title")}</span>
                  </div>
                  <div className="font-mono text-sm font-bold text-sky-600 dark:text-sky-400 mt-1">
                    {telemetryRounds && telemetryRounds.length > 0
                      ? `${telemetryRounds[telemetryRounds.length - 1].train_accuracy}%`
                      : "0.984"}
                  </div>
                  <div className="text-[10px] text-[var(--text-muted)] mt-0.5">
                    {telemetryRounds && telemetryRounds.length > 0
                      ? `${t("topo_live_round_prefix")} ${telemetryRounds.length} ${t("topo_validation_suffix")}`
                      : t("topo_telem_auc_desc")}
                  </div>
                </div>
              </div>

              <div className="mt-3 p-3 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[11px] leading-relaxed text-[var(--text-main)]">
                <div className="flex items-center justify-between mb-1.5">
                  <strong className="text-[var(--text-heading)]">
                    {Array.isArray(telemetryRounds) && telemetryRounds.length > 0
                      ? t("topo_telem_live_trajectory")
                      : t("topo_telem_log_title")}
                  </strong>
                  <span className="font-mono text-[10px] text-emerald-600 dark:text-emerald-400 font-semibold">
                    {Array.isArray(telemetryRounds) && telemetryRounds.length > 0
                      ? `${t("topo_loss_prefix")} ${telemetryRounds[telemetryRounds.length - 1].train_loss}`
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
        <div className="relative z-30 mt-2 p-2 sm:p-2.5 rounded-2xl bg-[var(--bg-card)]/80 dark:bg-slate-900/80 border border-[var(--border-card)] text-xs backdrop-blur-md shadow-xs">
          <div className="flex items-center justify-between gap-1.5 sm:gap-2">
            {/* Left: Selected Hospital & Location */}
            <div className="flex items-center gap-1.5 sm:gap-2 min-w-0 flex-1">
              <div className="w-6 h-6 sm:w-7 sm:h-7 rounded-xl bg-emerald-500/10 border border-emerald-500/25 flex items-center justify-center text-emerald-600 dark:text-emerald-400 shrink-0 shadow-xs">
                <Server className="w-3.5 h-3.5 sm:w-4 sm:h-4" />
              </div>
              <div className="min-w-0 flex items-center gap-1 sm:gap-1.5 flex-wrap">
                <span className="font-display font-bold text-[var(--text-heading)] text-[11px] sm:text-xs truncate max-w-[140px] xs:max-w-[180px] sm:max-w-none">
                  {getLocalizedNodeName(selectedNode, t)}
                </span>
                <span className="text-[9px] sm:text-[10px] font-mono px-1.5 sm:px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 shrink-0">
                  {getLocalizedNodeLocation(selectedNode, t)}
                </span>
              </div>
            </div>

            {/* Right: DP Budget & Quick Trigger Action */}
            <div className="flex items-center gap-1.5 sm:gap-2 shrink-0 text-[10px] font-mono">
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
                  className="px-2.5 sm:px-3 py-1 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-[10px] sm:text-[11px] transition-all shadow-sm shadow-emerald-600/20 active:scale-95 border border-emerald-500/30 cursor-pointer"
                >
                  {t("topo_simulate")}
                </button>
              )}
            </div>
          </div>

          {/* Subline: Scanner Hardware & Skew */}
          <div className="mt-1.5 pt-1.5 border-t border-[var(--border-card)] flex flex-col sm:flex-row sm:items-center justify-between gap-1 text-[10px] font-mono">
            <span className="text-[var(--text-muted)] leading-tight break-words truncate">
              {getLocalizedNodeScanner(selectedNode, t)}
            </span>
            <div className="inline-flex items-center gap-1 text-amber-600 dark:text-amber-300 font-semibold bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/20 text-[9.5px] sm:text-[10px] w-fit max-w-full">
              <span className="shrink-0">{t("topo_skew_label")}</span>
              <span className="break-words">{t(selectedNode.skewKey)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
