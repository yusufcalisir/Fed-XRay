"use client";

import React, { useState } from "react";
import Navbar from "@/components/Navbar";
import ConsortiumLaunchOverlay from "@/components/ConsortiumLaunchOverlay";
import { useLanguage } from "@/context/LanguageContext";
import {
  ShieldCheck,
  Lock,
  Activity,
  Server,
  ArrowRight,
  FileText,
  Volume2,
  Database,
  Cpu,
  Layers,
  Sparkles,
  Sliders,
  CheckCircle2,
  AlertTriangle,
} from "lucide-react";

export default function PresentationLandingPage() {
  const { t } = useLanguage();
  const [isLaunchModalOpen, setIsLaunchModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"xai" | "rag" | "convergence" | "report">("xai");
  const [blendOpacity, setBlendOpacity] = useState<number>(0.65);

  const openLaunchModal = () => setIsLaunchModalOpen(true);
  const closeLaunchModal = () => setIsLaunchModalOpen(false);

  return (
    <div className="min-h-screen flex flex-col bg-[var(--bg-app)] text-[var(--text-main)] overflow-x-hidden selection:bg-emerald-500/20">
      <Navbar mode="landing" onLaunchDemo={openLaunchModal} />

      <ConsortiumLaunchOverlay
        isOpen={isLaunchModalOpen}
        onClose={closeLaunchModal}
      />

      <main className="flex-1 max-w-[1360px] w-full mx-auto px-3 sm:px-6 py-6 sm:py-10 space-y-12 sm:space-y-16">

        {/* 1. Hero Section */}
        <section className="relative pt-4 sm:pt-8 pb-6">
          <div className="max-w-4xl">
            {/* Clinical Infrastructure Badge */}
            <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold mb-4 sm:mb-6 shadow-sm">
              <ShieldCheck className="w-4 h-4 shrink-0" />
              <span>{t("landing_hero_badge")}</span>
            </div>

            {/* Main Institutional Headline */}
            <h1 className="font-display text-2xl sm:text-4xl lg:text-5xl font-extrabold tracking-tight text-[var(--text-heading)] leading-[1.15] mb-4 sm:mb-6">
              {t("landing_hero_title")}
            </h1>

            {/* Subtitle Description */}
            <p className="text-sm sm:text-base lg:text-lg text-[var(--text-muted)] leading-relaxed max-w-3xl mb-6 sm:mb-8">
              {t("landing_hero_desc")}
            </p>

            {/* Primary Action Buttons */}
            <div className="flex flex-wrap items-center gap-3 sm:gap-4">
              <button
                onClick={openLaunchModal}
                className="btn-primary text-xs sm:text-sm px-5 sm:px-6 py-3 rounded-xl shadow-lg hover:shadow-emerald-500/20"
              >
                <Activity className="w-4 h-4" />
                <span>{t("landing_cta_launch")}</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>

              <a
                href="#results"
                className="inline-flex items-center gap-2 px-5 py-3 rounded-xl border border-[var(--border-card)] bg-[var(--bg-card)] hover:bg-[var(--bg-card-inner)] text-xs sm:text-sm font-semibold text-[var(--text-heading)] transition-all shadow-sm"
              >
                <span>{t("landing_results_title")}</span>
                <ArrowRight className="w-4 h-4 text-slate-400" />
              </a>
            </div>
          </div>

          {/* 5 Verifiable Institutional KPI Metrics */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2.5 sm:gap-4 mt-8 sm:mt-12 pt-6 sm:pt-8 border-t border-[var(--border-subtle)]">
            <div className="card-inner p-3 sm:p-4">
              <div className="flex items-center gap-2 text-emerald-500 mb-1">
                <Server className="w-4 h-4 shrink-0" />
                <span className="font-display text-lg sm:text-2xl font-extrabold text-[var(--text-heading)]">4</span>
              </div>
              <div className="text-[11px] sm:text-xs font-semibold text-[var(--text-heading)]">{t("landing_metric_nodes")}</div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">{t("landing_metric_nodes_sub")}</div>
            </div>

            <div className="card-inner p-3 sm:p-4">
              <div className="flex items-center gap-2 text-emerald-500 mb-1">
                <Lock className="w-4 h-4 shrink-0" />
                <span className="font-display text-lg sm:text-2xl font-extrabold text-[var(--text-heading)]">0 B</span>
              </div>
              <div className="text-[11px] sm:text-xs font-semibold text-[var(--text-heading)]">{t("landing_metric_leakage")}</div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">{t("landing_metric_leakage_sub")}</div>
            </div>

            <div className="card-inner p-3 sm:p-4">
              <div className="flex items-center gap-2 text-teal-500 mb-1">
                <Cpu className="w-4 h-4 shrink-0" />
                <span className="font-display text-lg sm:text-2xl font-extrabold text-[var(--text-heading)]">128-Bit</span>
              </div>
              <div className="text-[11px] sm:text-xs font-semibold text-[var(--text-heading)]">{t("landing_metric_crypto")}</div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">{t("landing_metric_crypto_sub")}</div>
            </div>

            <div className="card-inner p-3 sm:p-4">
              <div className="flex items-center gap-2 text-emerald-500 mb-1">
                <Activity className="w-4 h-4 shrink-0" />
                <span className="font-display text-lg sm:text-2xl font-extrabold text-[var(--text-heading)]">98.4%</span>
              </div>
              <div className="text-[11px] sm:text-xs font-semibold text-[var(--text-heading)]">{t("landing_metric_auc")}</div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">{t("landing_metric_auc_sub")}</div>
            </div>

            <div className="col-span-2 sm:col-span-1 card-inner p-3 sm:p-4">
              <div className="flex items-center gap-2 text-blue-500 mb-1">
                <ShieldCheck className="w-4 h-4 shrink-0" />
                <span className="font-display text-lg sm:text-2xl font-extrabold text-[var(--text-heading)]">ε ≤ 2.0</span>
              </div>
              <div className="text-[11px] sm:text-xs font-semibold text-[var(--text-heading)]">{t("landing_metric_dp")}</div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">{t("landing_metric_dp_sub")}</div>
            </div>
          </div>
        </section>

        {/* 2. Interactive Clinical Results Showcase */}
        <section id="results" className="scroll-mt-20">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-6">
            <div>
              <div className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
                Clinical CDSS Verification
              </div>
              <h2 className="font-display text-xl sm:text-2xl lg:text-3xl font-bold text-[var(--text-heading)]">
                {t("landing_results_title")}
              </h2>
              <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-1 max-w-2xl">
                {t("landing_results_subtitle")}
              </p>
            </div>

            {/* Interactive Tab Switcher */}
            <div className="flex items-center p-1 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] shrink-0 self-start sm:self-auto overflow-x-auto">
              <button
                onClick={() => setActiveTab("xai")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                  activeTab === "xai"
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                {t("landing_tab_xai")}
              </button>
              <button
                onClick={() => setActiveTab("rag")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                  activeTab === "rag"
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                {t("landing_tab_rag")}
              </button>
              <button
                onClick={() => setActiveTab("convergence")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                  activeTab === "convergence"
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                {t("landing_tab_convergence")}
              </button>
              <button
                onClick={() => setActiveTab("report")}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                  activeTab === "report"
                    ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                {t("landing_tab_report")}
              </button>
            </div>
          </div>

          {/* Tab 1: Grad-CAM Saliency Display */}
          {activeTab === "xai" && (
            <div className="card p-4 sm:p-6 space-y-5">
              <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-3 pb-4 border-b border-[var(--border-subtle)]">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                      Primary Finding: Pneumonia
                    </span>
                    <span className="text-xs text-[var(--text-muted)]">Patient Ref: #P-8821</span>
                  </div>
                  <h3 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)] mt-1">
                    Bilateral Alveolar Consolidation with Air Bronchogram Localization
                  </h3>
                </div>

                <div className="flex items-center gap-4 text-xs shrink-0">
                  <div>
                    <span className="text-[var(--text-muted)] mr-1.5">Confidence:</span>
                    <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">94.8%</span>
                  </div>
                  <div>
                    <span className="text-[var(--text-muted)] mr-1.5">Ground Truth:</span>
                    <span className="font-semibold text-slate-800 dark:text-slate-200">Bacterial Lobar Consolidation</span>
                  </div>
                </div>
              </div>

              {/* Dual-Pane Visual Evidence */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* Left: Original Radiograph */}
                <div className="card-inner p-3 sm:p-4 flex flex-col items-center">
                  <div className="w-full flex items-center justify-between text-xs font-semibold text-[var(--text-muted)] mb-2">
                    <span>{t("sec3_dual_pane_orig")}</span>
                    <span className="text-[10px] font-mono">DICOM Window: Lung</span>
                  </div>
                  <div className="relative w-full aspect-square max-w-sm rounded-xl overflow-hidden bg-slate-900 border border-slate-700/50 flex items-center justify-center">
                    {/* Simulated Authentic Medical Chest Scan Canvas */}
                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-slate-800 to-slate-950 p-6 flex flex-col justify-between">
                      <div className="flex justify-between text-[10px] font-mono text-slate-500">
                        <span>R</span>
                        <span>PA ERECT</span>
                        <span>L</span>
                      </div>
                      <div className="flex justify-around items-center">
                        <div className="w-24 h-40 rounded-full border border-slate-600/30 bg-slate-800/40" />
                        <div className="w-12 h-44 bg-slate-700/50 rounded" />
                        <div className="w-24 h-40 rounded-full border border-slate-600/30 bg-slate-800/40" />
                      </div>
                      <div className="text-center text-[10px] font-mono text-slate-500">
                        Patient ID: #P-8821 · 120 kVp
                      </div>
                    </div>
                  </div>
                </div>

                {/* Right: Grad-CAM Overlay */}
                <div className="card-inner p-3 sm:p-4 flex flex-col items-center">
                  <div className="w-full flex items-center justify-between text-xs font-semibold text-[var(--text-muted)] mb-2">
                    <span>{t("sec3_dual_pane_xai")}</span>
                    <span className="text-[10px] font-mono text-emerald-500">Peak Saliency: Right Lower Lobe</span>
                  </div>
                  <div className="relative w-full aspect-square max-w-sm rounded-xl overflow-hidden bg-slate-900 border border-emerald-500/30 flex items-center justify-center">
                    <div className="w-full h-full bg-gradient-to-b from-slate-900 via-slate-800 to-slate-950 p-6 flex flex-col justify-between relative">
                      <div className="flex justify-between text-[10px] font-mono text-slate-500">
                        <span>R</span>
                        <span>GRAD-CAM HEATMAP</span>
                        <span>L</span>
                      </div>
                      <div className="flex justify-around items-center">
                        <div className="relative w-24 h-40 rounded-full border border-slate-600/30 bg-slate-800/40">
                          {/* Saliency Hotspot */}
                          <div
                            className="absolute bottom-2 left-2 w-16 h-20 rounded-full bg-gradient-to-tr from-rose-500 via-amber-400 to-emerald-400 blur-sm pointer-events-none transition-opacity"
                            style={{ opacity: blendOpacity }}
                          />
                        </div>
                        <div className="w-12 h-44 bg-slate-700/50 rounded" />
                        <div className="w-24 h-40 rounded-full border border-slate-600/30 bg-slate-800/40" />
                      </div>
                      <div className="text-center text-[10px] font-mono text-emerald-400 font-semibold">
                        Confidence: 94.8% · Active Layer: L-3 MHSA
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Heatmap Control & Interpretation */}
              <div className="card-inner p-3.5 sm:p-4 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="flex items-center gap-3 w-full sm:w-auto">
                  <Sliders className="w-4 h-4 text-emerald-500 shrink-0" />
                  <span className="text-xs font-medium text-[var(--text-muted)]">{t("sec3_xai_opacity")}:</span>
                  <input
                    type="range"
                    min="0.1"
                    max="1.0"
                    step="0.05"
                    value={blendOpacity}
                    onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
                    className="w-32 h-1.5 accent-emerald-500 bg-slate-200 dark:bg-white/10 rounded-lg cursor-pointer"
                  />
                  <span className="text-xs font-mono font-bold text-emerald-600 dark:text-emerald-400">
                    {Math.round(blendOpacity * 100)}%
                  </span>
                </div>

                <div className="text-xs text-[var(--text-muted)] leading-relaxed">
                  <strong className="text-[var(--text-heading)]">Radiologist Assessment:</strong> Focal alveolar opacification with air bronchograms localized to the right lower lobe, sparing apical segments.
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: RAG Case Matching */}
          {activeTab === "rag" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-[var(--border-subtle)]">
                <div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)]">
                    {t("sec3_rag_title")}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    {t("sec3_rag_desc")}
                  </p>
                </div>
                <span className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  Dual Reference Matches
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card-inner p-4 space-y-2.5 border-l-4 border-l-emerald-500">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-emerald-600 dark:text-emerald-400">REF-ONCO-401</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                      97.4% Cosine Similarity
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-[var(--text-heading)]">
                    Bacterial Lobar Consolidation (Confirmed Biopsy)
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Dense right lower lobe opacification. Complete radiological and clinical resolution documented following 14-day targeted levofloxacin therapy.
                  </p>
                  <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-[var(--border-subtle)]">
                    Origin: Metropolitan Clinical Center · Outcome: Complete Remission
                  </div>
                </div>

                <div className="card-inner p-4 space-y-2.5 border-l-4 border-l-teal-500">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-teal-600 dark:text-teal-400">REF-ONCO-402</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-500/10 text-teal-600 dark:text-teal-400">
                      93.1% Cosine Similarity
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-[var(--text-heading)]">
                    Hospital-Acquired Pulmonary Infiltrate
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Multifocal ground-glass opacity. Patient demonstrated rapid inflammatory biomarker clearance under IV ampicillin-sulbactam regimen.
                  </p>
                  <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-[var(--border-subtle)]">
                    Origin: University Hospital Center · Outcome: Therapeutic Clearance
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Tab 3: Convergence Telemetry */}
          {activeTab === "convergence" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="flex items-center justify-between pb-3 border-b border-[var(--border-subtle)]">
                <div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)]">
                    Multi-Center Federated Convergence & Robustness Log
                  </h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    Hold-out validation metrics across 5 consecutive rounds under FedDyn dynamic alignment.
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">Consensus Converged</span>
                </div>
              </div>

              {/* Convergence Table */}
              <div className="overflow-x-auto">
                <table className="w-full min-w-[620px] text-xs text-left">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)] text-[var(--text-muted)] text-[10px] uppercase tracking-wider">
                      <th className="py-2.5 px-3">Round</th>
                      <th className="py-2.5 px-3">Global Test Loss</th>
                      <th className="py-2.5 px-3">Hold-Out Accuracy</th>
                      <th className="py-2.5 px-3">Macro F1</th>
                      <th className="py-2.5 px-3">Bandwidth (FFA-LoRA)</th>
                      <th className="py-2.5 px-3">Byzantine Threat Status</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)] font-mono">
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">Round 1</td>
                      <td className="py-2 px-3">0.852</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">72.5%</td>
                      <td className="py-2 px-3">0.718</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Nodes Verified
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">Round 2</td>
                      <td className="py-2 px-3">0.614</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">81.3%</td>
                      <td className="py-2 px-3">0.809</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> All 4 Nodes Verified
                      </td>
                    </tr>
                    <tr className="bg-amber-500/[0.04]">
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">Round 3</td>
                      <td className="py-2 px-3">0.491</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">87.2%</td>
                      <td className="py-2 px-3">0.868</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-amber-500 flex items-center gap-1 font-sans font-semibold">
                        <AlertTriangle className="w-3.5 h-3.5" /> Intercepted Node 3 Poisoning
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">Round 4</td>
                      <td className="py-2 px-3">0.378</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">91.6%</td>
                      <td className="py-2 px-3">0.914</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Filtered Consensus Active
                      </td>
                    </tr>
                    <tr className="bg-emerald-500/[0.04]">
                      <td className="py-2 px-3 font-bold text-emerald-600 dark:text-emerald-400">Round 5 (Final)</td>
                      <td className="py-2 px-3 font-bold">0.286</td>
                      <td className="py-2 px-3 font-bold text-emerald-600 dark:text-emerald-400">94.8%</td>
                      <td className="py-2 px-3 font-bold">0.947</td>
                      <td className="py-2 px-3 font-bold">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans font-bold">
                        <CheckCircle2 className="w-3.5 h-3.5" /> Consensus Deployed
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="card-inner p-3 flex items-center justify-between text-xs text-[var(--text-muted)]">
                <span>Total Communication: 21.0 MB across 5 rounds (vs 1.3 GB for uncompressed ViT-Base)</span>
                <span className="font-semibold text-emerald-600 dark:text-emerald-400">98.4% Bandwidth Savings</span>
              </div>
            </div>
          )}

          {/* Tab 4: Official Report & Audio Preview */}
          {activeTab === "report" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {/* PDF Report Preview Card */}
                <div className="card-inner p-4 space-y-3">
                  <div className="flex items-center gap-2 text-emerald-500">
                    <FileText className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">{t("sec3_report_title")}</span>
                  </div>
                  <h4 className="text-sm font-bold text-[var(--text-heading)]">
                    Single-Page A4 Medical Intelligence Export
                  </h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Embedded side-by-side standard radiograph and Grad-CAM attention heatmap, calibrated probability breakdown, patient cohort demographics, and SHA-256 integrity digest.
                  </p>
                  <div className="pt-2">
                    <button
                      onClick={openLaunchModal}
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-colors"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>Generate Live in Cockpit</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>

                {/* Voice Dictation Card */}
                <div className="card-inner p-4 space-y-3">
                  <div className="flex items-center gap-2 text-blue-500">
                    <Volume2 className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">{t("sec3_voice_title")}</span>
                  </div>
                  <h4 className="text-sm font-bold text-[var(--text-heading)]">
                    Hands-Free Synthesized Clinical Dictation
                  </h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    Instant automated audio briefing for surgical suites and sterile reading rooms. Articulates primary pathology, confidence level, and anatomical focal regions.
                  </p>
                  <div className="pt-2">
                    <button
                      onClick={openLaunchModal}
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                      <span>Listen in Live Cockpit</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  </div>
                </div>
              </div>
            </div>
          )}
        </section>

        {/* 3. Four Security & Governance Pillars */}
        <section id="security" className="scroll-mt-20">
          <div className="mb-6">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
              Provable Cryptographic Privacy
            </div>
            <h2 className="font-display text-xl sm:text-2xl lg:text-3xl font-bold text-[var(--text-heading)]">
              {t("landing_security_title")}
            </h2>
            <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-1 max-w-2xl">
              {t("landing_security_subtitle")}
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="card p-4 sm:p-5 flex flex-col justify-between">
              <div>
                <div className="w-9 h-9 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-500 mb-3">
                  <Database className="w-4 h-4" />
                </div>
                <h3 className="font-display text-sm font-bold text-[var(--text-heading)] mb-2">
                  {t("landing_pillar_1_title")}
                </h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_pillar_1_desc")}
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-[var(--border-subtle)] text-[10px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold">
                Strategy E Standard: 0% Cross-Split
              </div>
            </div>

            <div className="card p-4 sm:p-5 flex flex-col justify-between">
              <div>
                <div className="w-9 h-9 rounded-xl bg-blue-500/10 border border-blue-500/20 flex items-center justify-center text-blue-500 mb-3">
                  <ShieldCheck className="w-4 h-4" />
                </div>
                <h3 className="font-display text-sm font-bold text-[var(--text-heading)] mb-2">
                  {t("landing_pillar_2_title")}
                </h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_pillar_2_desc")}
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-[var(--border-subtle)] text-[10px] font-mono text-blue-600 dark:text-blue-400 font-semibold">
                Bound: (ε ≤ 2.0, δ ≤ 10^-5)
              </div>
            </div>

            <div className="card p-4 sm:p-5 flex flex-col justify-between">
              <div>
                <div className="w-9 h-9 rounded-xl bg-teal-500/10 border border-teal-500/20 flex items-center justify-center text-teal-500 mb-3">
                  <Lock className="w-4 h-4" />
                </div>
                <h3 className="font-display text-sm font-bold text-[var(--text-heading)] mb-2">
                  {t("landing_pillar_3_title")}
                </h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_pillar_3_desc")}
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-[var(--border-subtle)] text-[10px] font-mono text-teal-600 dark:text-teal-400 font-semibold">
                Leveled CKKS 128-Bit RLWE
              </div>
            </div>

            <div className="card p-4 sm:p-5 flex flex-col justify-between">
              <div>
                <div className="w-9 h-9 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center text-amber-500 mb-3">
                  <AlertTriangle className="w-4 h-4" />
                </div>
                <h3 className="font-display text-sm font-bold text-[var(--text-heading)] mb-2">
                  {t("landing_pillar_4_title")}
                </h3>
                <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_pillar_4_desc")}
                </p>
              </div>
              <div className="pt-3 mt-3 border-t border-[var(--border-subtle)] text-[10px] font-mono text-amber-600 dark:text-amber-400 font-semibold">
                Trimmed Mean & Hold-Out Shield
              </div>
            </div>
          </div>
        </section>

        {/* 4. Consortium Architecture Pipeline */}
        <section id="architecture" className="scroll-mt-20">
          <div className="mb-6">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
              System Specification
            </div>
            <h2 className="font-display text-xl sm:text-2xl lg:text-3xl font-bold text-[var(--text-heading)]">
              {t("landing_arch_title")}
            </h2>
            <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-1 max-w-2xl">
              {t("landing_arch_subtitle")}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3 sm:gap-4">
            <div className="card-inner p-3.5 sm:p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-emerald-500">STAGE 01</span>
                <h3 className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)] mt-1 mb-1.5">
                  {t("landing_arch_s1_title")}
                </h3>
                <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_arch_s1_desc")}
                </p>
              </div>
            </div>

            <div className="card-inner p-3.5 sm:p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-emerald-500">STAGE 02</span>
                <h3 className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)] mt-1 mb-1.5">
                  {t("landing_arch_s2_title")}
                </h3>
                <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_arch_s2_desc")}
                </p>
              </div>
            </div>

            <div className="card-inner p-3.5 sm:p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-teal-500">STAGE 03</span>
                <h3 className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)] mt-1 mb-1.5">
                  {t("landing_arch_s3_title")}
                </h3>
                <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_arch_s3_desc")}
                </p>
              </div>
            </div>

            <div className="card-inner p-3.5 sm:p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-blue-500">STAGE 04</span>
                <h3 className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)] mt-1 mb-1.5">
                  {t("landing_arch_s4_title")}
                </h3>
                <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_arch_s4_desc")}
                </p>
              </div>
            </div>

            <div className="card-inner p-3.5 sm:p-4 flex flex-col justify-between">
              <div>
                <span className="text-[10px] font-mono font-bold text-emerald-500">STAGE 05</span>
                <h3 className="font-display text-xs sm:text-sm font-bold text-[var(--text-heading)] mt-1 mb-1.5">
                  {t("landing_arch_s5_title")}
                </h3>
                <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed">
                  {t("landing_arch_s5_desc")}
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* 5. Empirical Strategy E Benchmark Matrix */}
        <section id="benchmarks" className="scroll-mt-20">
          <div className="mb-6">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
              Strategy E Scientific Evidence
            </div>
            <h2 className="font-display text-xl sm:text-2xl lg:text-3xl font-bold text-[var(--text-heading)]">
              {t("landing_bench_title")}
            </h2>
            <p className="text-xs sm:text-sm text-[var(--text-muted)] mt-1 max-w-2xl">
              {t("landing_bench_subtitle")}
            </p>
          </div>

          <div className="card p-3 sm:p-5 overflow-x-auto">
            <table className="w-full min-w-[620px] text-xs text-left">
              <thead>
                <tr className="border-b border-[var(--border-subtle)] text-[var(--text-muted)] text-[10px] uppercase tracking-wider">
                  <th className="py-2.5 px-3">{t("landing_bench_col_system")}</th>
                  <th className="py-2.5 px-3">{t("landing_bench_col_auc")}</th>
                  <th className="py-2.5 px-3">{t("landing_bench_col_payload")}</th>
                  <th className="py-2.5 px-3">{t("landing_bench_col_privacy")}</th>
                  <th className="py-2.5 px-3">{t("landing_bench_col_resilience")}</th>
                  <th className="py-2.5 px-3">{t("landing_bench_col_generalization")}</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[var(--border-subtle)] font-mono">
                <tr className="bg-emerald-500/[0.05] font-semibold">
                  <td className="py-3 px-3 font-bold text-emerald-600 dark:text-emerald-400 font-sans">
                    Fed-XRay (FFA-LoRA + FedDyn)
                  </td>
                  <td className="py-3 px-3 text-emerald-600 dark:text-emerald-400 font-bold">0.984</td>
                  <td className="py-3 px-3">4.2 MB / round</td>
                  <td className="py-3 px-3 text-emerald-500 font-sans">Patient DP (ε ≤ 2.0) + CKKS</td>
                  <td className="py-3 px-3 text-emerald-500 font-sans">Trimmed Mean Active</td>
                  <td className="py-3 px-3 text-emerald-600 dark:text-emerald-400 font-sans">High (Domain Aligned)</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">Centralized Upper Bound</td>
                  <td className="py-2.5 px-3">0.987</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">N/A (Centralized Data)</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">Zero (Full Leakage)</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">N/A</td>
                  <td className="py-2.5 px-3 font-sans">Optimal</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">Traditional FedAvg (Full ViT)</td>
                  <td className="py-2.5 px-3">0.912</td>
                  <td className="py-2.5 px-3 text-rose-500">340 MB / round</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">Plaintext Weights</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">Vulnerable to Poisoning</td>
                  <td className="py-2.5 px-3 font-sans">Client Drift Oscillations</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">Local Hospital Standalone</td>
                  <td className="py-2.5 px-3">0.824</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">0 MB (Isolated)</td>
                  <td className="py-2.5 px-3 text-emerald-500 font-sans">Isolated (No Transfer)</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">N/A</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">Severe Institutional Overfitting</td>
                </tr>
              </tbody>
            </table>
          </div>
        </section>

        {/* 6. Consortium Launch Banner */}
        <section className="card p-6 sm:p-8 bg-gradient-to-r from-emerald-950/30 via-slate-900/40 to-teal-950/30 border-emerald-500/20 text-center">
          <div className="max-w-2xl mx-auto space-y-4">
            <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 mx-auto flex items-center justify-center text-emerald-500">
              <Activity className="w-5 h-5 animate-pulse" />
            </div>
            <h2 className="font-display text-xl sm:text-2xl font-bold text-[var(--text-heading)]">
              {t("landing_banner_title")}
            </h2>
            <p className="text-xs sm:text-sm text-[var(--text-muted)] leading-relaxed">
              {t("landing_banner_desc")}
            </p>
            <div className="pt-2">
              <button
                onClick={openLaunchModal}
                className="btn-primary text-xs sm:text-sm px-6 py-3 rounded-xl shadow-lg hover:shadow-emerald-500/25"
              >
                <ShieldCheck className="w-4 h-4" />
                <span>{t("landing_banner_btn")}</span>
                <ArrowRight className="w-4 h-4 ml-1" />
              </button>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-[var(--navbar-border)] py-6 text-center text-[10px] sm:text-[11px] text-[var(--text-muted)] px-3">
        {t("footer_text")}
      </footer>
    </div>
  );
}
