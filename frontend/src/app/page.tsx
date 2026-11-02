"use client";

import React, { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import ConsortiumLaunchOverlay from "@/components/ConsortiumLaunchOverlay";
import LungViewer3D from "@/components/LungViewer3D";
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
  const [backendStatus, setBackendStatus] = useState<"checking" | "online" | "standby">("checking");

  useEffect(() => {
    let isMounted = true;
    const checkBackend = async () => {
      try {
        const res = await fetch("/api/health").catch(() => fetch("http://127.0.0.1:8000/api/health"));
        if (res && res.ok && isMounted) {
          setBackendStatus("online");
        } else if (isMounted) {
          setBackendStatus("standby");
        }
      } catch {
        if (isMounted) setBackendStatus("standby");
      }
    };
    checkBackend();
    return () => {
      isMounted = false;
    };
  }, []);

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
        <section className="relative pt-2 sm:pt-6 pb-6">
          <div className="absolute inset-x-0 -top-10 h-[620px] -z-10 bg-grid pointer-events-none" />
          <div className="grid grid-cols-1 lg:grid-cols-[1.02fr_0.98fr] gap-8 lg:gap-10 items-center">
            <div className="max-w-2xl">
              {/* Clinical Infrastructure Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold mb-4 sm:mb-6 shadow-sm">
                <ShieldCheck className="w-4 h-4 shrink-0" />
                <span>{t("landing_hero_badge")}</span>
              </div>

              {/* Main Institutional Headline */}
              <h1 className="font-display text-3xl sm:text-4xl lg:text-[44px] xl:text-5xl font-extrabold tracking-tight text-[var(--text-heading)] leading-[1.12] mb-4 sm:mb-6">
                {t("landing_hero_title")}
              </h1>

              {/* Subtitle Description */}
              <p className="text-sm sm:text-base lg:text-lg text-[var(--text-muted)] leading-relaxed max-w-xl mb-6 sm:mb-8">
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

              {/* Live Federation Node Connectivity Status */}
              <div className="mt-4 flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${backendStatus === "online" ? "bg-emerald-500 animate-pulse" : "bg-slate-400"}`} />
                <span className="font-mono text-[11px] text-[var(--text-muted)]">
                  {backendStatus === "online" ? t("landing_backend_online") : t("landing_backend_standby")}
                </span>
              </div>
            </div>

            <div className="relative">
              <div className="lung-viewer relative h-[380px] sm:h-[460px] lg:h-[540px]">
                <LungViewer3D className="absolute inset-0" mode="anatomy" />
                <div className="chip-float absolute left-4 top-5">
                  <Cpu className="w-3.5 h-3.5 text-emerald-300" />
                  <span>{t("landing_hero_chip_peft")}</span>
                </div>
                <div className="chip-float absolute right-4 top-1/3" style={{ animationDelay: "-2s" }}>
                  <Lock className="w-3.5 h-3.5 text-sky-300" />
                  <span>{t("landing_hero_chip_he")}</span>
                </div>
                <div className="chip-float absolute left-5 bottom-12" style={{ animationDelay: "-4s" }}>
                  <ShieldCheck className="w-3.5 h-3.5 text-violet-300" />
                  <span>{t("landing_hero_chip_dp")}</span>
                </div>
                <div className="pointer-events-none absolute left-4 bottom-3 font-mono text-[10px] uppercase tracking-wider text-white/45">
                  {t("landing_hero_viewer_caption")}
                </div>
              </div>
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
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-4">
            <div>
              <div className="flex flex-wrap items-center gap-2 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400">
                  {t("landing_sec_results_badge")}
                </span>
                <span className="px-2 py-0.5 rounded text-[10px] font-mono font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                  {t("landing_showcase_spec_badge")}
                </span>
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

          {/* Protocol Specification Banner Card */}
          <div className="card-inner p-3.5 sm:p-4 mb-6 border-l-4 border-l-blue-500 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
            <div className="space-y-0.5">
              <div className="text-xs font-bold text-[var(--text-heading)] flex items-center gap-1.5">
                <Sparkles className="w-3.5 h-3.5 text-blue-500" />
                <span>{t("landing_showcase_spec_badge")}</span>
              </div>
              <p className="text-xs text-[var(--text-muted)] leading-relaxed max-w-3xl">
                {t("landing_showcase_spec_desc")}
              </p>
            </div>
            <button
              onClick={openLaunchModal}
              className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-lg bg-blue-500 text-white text-xs font-semibold hover:bg-blue-600 transition-colors shrink-0 shadow-sm"
            >
              <span>{t("landing_showcase_spec_btn")}</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {/* Tab 1: 3D lung saliency workspace (anatomical model, no patient sample) */}
          {activeTab === "xai" && (
            <div className="card p-4 sm:p-6">
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 items-stretch">
                <div className="lung-viewer relative lg:col-span-3 h-[380px] sm:h-[460px]">
                  <LungViewer3D className="absolute inset-0" mode="anatomy" />
                  <div className="pointer-events-none absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-full border border-white/10 bg-black/40 px-2.5 py-1 font-mono text-[10px] text-emerald-200 backdrop-blur">
                    <Sparkles className="w-3 h-3" />
                    {t("lung3d_locked_preview")}
                  </div>
                </div>
                <div className="lg:col-span-2 flex flex-col gap-4">
                  <div>
                    <h3 className="font-display text-lg sm:text-xl font-bold text-[var(--text-heading)]">{t("landing_xai3d_title")}</h3>
                    <p className="mt-1 text-xs sm:text-sm text-[var(--text-muted)] leading-relaxed">{t("landing_xai3d_desc")}</p>
                  </div>
                  <ol className="space-y-3">
                  <li className="flex gap-3">
                    <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                      <Database className="w-3.5 h-3.5" />
                    </span>
                    <span>
                      <span className="block text-xs font-bold text-[var(--text-heading)]">{t("landing_xai3d_s1_t")}</span>
                      <span className="block text-[11px] text-[var(--text-muted)] leading-relaxed">{t("landing_xai3d_s1_d")}</span>
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                      <Activity className="w-3.5 h-3.5" />
                    </span>
                    <span>
                      <span className="block text-xs font-bold text-[var(--text-heading)]">{t("landing_xai3d_s2_t")}</span>
                      <span className="block text-[11px] text-[var(--text-muted)] leading-relaxed">{t("landing_xai3d_s2_d")}</span>
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                      <Layers className="w-3.5 h-3.5" />
                    </span>
                    <span>
                      <span className="block text-xs font-bold text-[var(--text-heading)]">{t("landing_xai3d_s3_t")}</span>
                      <span className="block text-[11px] text-[var(--text-muted)] leading-relaxed">{t("landing_xai3d_s3_d")}</span>
                    </span>
                  </li>
                  <li className="flex gap-3">
                    <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                      <CheckCircle2 className="w-3.5 h-3.5" />
                    </span>
                    <span>
                      <span className="block text-xs font-bold text-[var(--text-heading)]">{t("landing_xai3d_s4_t")}</span>
                      <span className="block text-[11px] text-[var(--text-muted)] leading-relaxed">{t("landing_xai3d_s4_d")}</span>
                    </span>
                  </li>
                  </ol>
                  <div className="card-inner p-3 text-[11px] leading-relaxed text-[var(--text-muted)] border-l-4 border-l-blue-500">
                    {t("landing_xai3d_note")}
                  </div>
                  <button
                    onClick={openLaunchModal}
                    className="btn-primary mt-auto self-start"
                  >
                    <Activity className="w-4 h-4" />
                    <span>{t("landing_xai3d_cta")}</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Tab 2: RAG Case Matching */}
          {activeTab === "rag" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border-subtle)]">
                <div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)]">
                    {t("sec3_rag_title")}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    {t("sec3_rag_desc")}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                    {t("landing_tab2_spec_badge")}
                  </span>
                  <span className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                    {t("landing_tab2_badge")}
                  </span>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card-inner p-4 space-y-2.5 border-l-4 border-l-emerald-500">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-emerald-600 dark:text-emerald-400">REF-ONCO-401</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400">
                      {t("landing_tab2_c1_sim")}
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-[var(--text-heading)]">
                    {t("landing_tab2_c1_diag")}
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    {t("landing_tab2_c1_desc")}
                  </p>
                  <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-[var(--border-subtle)]">
                    {t("landing_tab2_c1_meta")}
                  </div>
                </div>

                <div className="card-inner p-4 space-y-2.5 border-l-4 border-l-teal-500">
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-bold text-teal-600 dark:text-teal-400">REF-ONCO-402</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-teal-500/10 text-teal-600 dark:text-teal-400">
                      {t("landing_tab2_c2_sim")}
                    </span>
                  </div>
                  <div className="text-xs font-semibold text-[var(--text-heading)]">
                    {t("landing_tab2_c2_diag")}
                  </div>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    {t("landing_tab2_c2_desc")}
                  </p>
                  <div className="text-[10px] font-mono text-slate-500 pt-1 border-t border-[var(--border-subtle)]">
                    {t("landing_tab2_c2_meta")}
                  </div>
                </div>
              </div>

              {/* Tab 2 Interactive Action Footer */}
              <div className="card-inner p-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 pt-3 border-t border-[var(--border-subtle)]">
                <div className="text-xs text-[var(--text-muted)] flex items-center gap-1.5">
                  <Database className="w-3.5 h-3.5 text-teal-500" />
                  <span>{t("landing_showcase_spec_badge")}</span>
                </div>
                <button
                  onClick={openLaunchModal}
                  className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-lg bg-teal-500/10 border border-teal-500/20 text-teal-600 dark:text-teal-400 text-xs font-semibold hover:bg-teal-500/20 transition-colors shrink-0"
                >
                  <Database className="w-3.5 h-3.5" />
                  <span>{t("landing_tab_action_cdss")}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>
          )}

          {/* Tab 3: Convergence Telemetry */}
          {activeTab === "convergence" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border-subtle)]">
                <div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)]">
                    {t("landing_tab3_title")}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    {t("landing_tab3_desc")}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20">
                    {t("landing_tab3_spec_badge")}
                  </span>
                  <span className="w-2 h-2 rounded-full bg-emerald-500" />
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    {t("landing_tab3_status")}
                  </span>
                </div>
              </div>

              {/* Convergence Table */}
              <div className="overflow-x-auto">
                <table className="w-full min-w-[620px] text-xs text-left">
                  <thead>
                    <tr className="border-b border-[var(--border-subtle)] text-[var(--text-muted)] text-[10px] uppercase tracking-wider">
                      <th className="py-2.5 px-3">{t("landing_tab3_col_round")}</th>
                      <th className="py-2.5 px-3">{t("landing_tab3_col_loss")}</th>
                      <th className="py-2.5 px-3">{t("landing_tab3_col_acc")}</th>
                      <th className="py-2.5 px-3">{t("landing_tab3_col_f1")}</th>
                      <th className="py-2.5 px-3">{t("landing_tab3_col_bw")}</th>
                      <th className="py-2.5 px-3">{t("landing_tab3_col_threat")}</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-[var(--border-subtle)] font-mono">
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">{t("landing_tab3_col_round")} 1</td>
                      <td className="py-2 px-3">0.852</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">72.5%</td>
                      <td className="py-2 px-3">0.718</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> {t("landing_tab3_verified")}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">{t("landing_tab3_col_round")} 2</td>
                      <td className="py-2 px-3">0.614</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">81.3%</td>
                      <td className="py-2 px-3">0.809</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> {t("landing_tab3_verified")}
                      </td>
                    </tr>
                    <tr className="bg-amber-500/[0.04]">
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">{t("landing_tab3_col_round")} 3</td>
                      <td className="py-2 px-3">0.491</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">87.2%</td>
                      <td className="py-2 px-3">0.868</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-amber-500 flex items-center gap-1 font-sans font-semibold">
                        <AlertTriangle className="w-3.5 h-3.5" /> {t("landing_tab3_intercepted")}
                      </td>
                    </tr>
                    <tr>
                      <td className="py-2 px-3 font-bold text-[var(--text-heading)]">{t("landing_tab3_col_round")} 4</td>
                      <td className="py-2 px-3">0.378</td>
                      <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">91.6%</td>
                      <td className="py-2 px-3">0.914</td>
                      <td className="py-2 px-3">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                        <CheckCircle2 className="w-3.5 h-3.5" /> {t("landing_tab3_filtered")}
                      </td>
                    </tr>
                    <tr className="bg-emerald-500/[0.04]">
                      <td className="py-2 px-3 font-bold text-emerald-600 dark:text-emerald-400">{t("landing_tab3_col_round")} 5 (Final)</td>
                      <td className="py-2 px-3 font-bold">0.286</td>
                      <td className="py-2 px-3 font-bold text-emerald-600 dark:text-emerald-400">94.8%</td>
                      <td className="py-2 px-3 font-bold">0.947</td>
                      <td className="py-2 px-3 font-bold">4.2 MB</td>
                      <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans font-bold">
                        <CheckCircle2 className="w-3.5 h-3.5" /> {t("landing_tab3_deployed")}
                      </td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <div className="card-inner p-3 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-xs text-[var(--text-muted)]">
                <div>
                  <span>{t("landing_tab3_comm_summary")} · </span>
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">{t("landing_tab3_savings")}</span>
                </div>
                <button
                  onClick={openLaunchModal}
                  className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors shrink-0"
                >
                  <Server className="w-3.5 h-3.5" />
                  <span>{t("landing_tab_action_train")}</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
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
                    {t("landing_tab4_pdf_title")}
                  </h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    {t("landing_tab4_pdf_desc")}
                  </p>
                  <div className="pt-2">
                    <button
                      onClick={openLaunchModal}
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-colors"
                    >
                      <FileText className="w-3.5 h-3.5" />
                      <span>{t("landing_tab4_pdf_btn")}</span>
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
                    {t("landing_tab4_voice_title")}
                  </h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">
                    {t("landing_tab4_voice_desc")}
                  </p>
                  <div className="pt-2">
                    <button
                      onClick={openLaunchModal}
                      className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-600 dark:text-blue-400 text-xs font-semibold hover:bg-blue-500/20 transition-colors"
                    >
                      <Volume2 className="w-3.5 h-3.5" />
                      <span>{t("landing_tab4_voice_btn")}</span>
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
              {t("landing_sec_security_badge")}
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
                {t("landing_p1_tag")}
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
                {t("landing_p2_tag")}
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
                {t("landing_p3_tag")}
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
                {t("landing_p4_tag")}
              </div>
            </div>
          </div>
        </section>

        {/* 4. Consortium Architecture Pipeline */}
        <section id="architecture" className="scroll-mt-20">
          <div className="mb-6">
            <div className="text-xs font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 mb-1">
              {t("landing_sec_arch_badge")}
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
                <span className="text-[10px] font-mono font-bold text-emerald-500">{t("landing_arch_stage_label")} 01</span>
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
                <span className="text-[10px] font-mono font-bold text-emerald-500">{t("landing_arch_stage_label")} 02</span>
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
                <span className="text-[10px] font-mono font-bold text-teal-500">{t("landing_arch_stage_label")} 03</span>
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
                <span className="text-[10px] font-mono font-bold text-blue-500">{t("landing_arch_stage_label")} 04</span>
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
                <span className="text-[10px] font-mono font-bold text-emerald-500">{t("landing_arch_stage_label")} 05</span>
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
              {t("landing_sec_bench_badge")}
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
                  <td className="py-3 px-3 text-emerald-500 font-sans">{t("bench_val_trimmed_active")}</td>
                  <td className="py-3 px-3 text-emerald-600 dark:text-emerald-400 font-sans">{t("bench_val_high_aligned")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_centralized")}</td>
                  <td className="py-2.5 px-3">0.987</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">{t("bench_val_na_centralized")}</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_leakage_full")}</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">N/A</td>
                  <td className="py-2.5 px-3 font-sans">{t("bench_val_optimal")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_fedavg")}</td>
                  <td className="py-2.5 px-3">0.912</td>
                  <td className="py-2.5 px-3 text-rose-500">340 MB / round</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">{t("bench_val_plaintext")}</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_vulnerable")}</td>
                  <td className="py-2.5 px-3 font-sans">{t("bench_val_drift")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_standalone")}</td>
                  <td className="py-2.5 px-3">0.824</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">{t("bench_val_isolated")}</td>
                  <td className="py-2.5 px-3 text-emerald-500 font-sans">{t("bench_val_no_transfer")}</td>
                  <td className="py-2.5 px-3 text-slate-500 font-sans">N/A</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_overfitting")}</td>
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
