"use client";

import React, { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import ConsortiumLaunchOverlay from "@/components/ConsortiumLaunchOverlay";
import LungViewer3D from "@/components/LungViewer3D";
import FederatedConsortiumTopology from "@/components/FederatedConsortiumTopology";
import Footer from "@/components/Footer";
import FedXRayLogo from "@/components/FedXRayLogo";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
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
  CheckCircle2,
  AlertTriangle,
  Building2,
  Eye,
  Zap,
} from "lucide-react";

import { BackendSystemState } from "@/types";
import { fetchSystemState } from "@/lib/api";
import ClinicalErrorBoundary from "@/components/ClinicalErrorBoundary";

export default function PresentationLandingPage() {
  const { t, language } = useLanguage();
  const { apiUrl, isConnected } = useApi();
  const [isLaunchModalOpen, setIsLaunchModalOpen] = useState(false);
  const [activeTab, setActiveTab] = useState<"xai" | "rag" | "convergence" | "report">("xai");
  const [systemState, setSystemState] = useState<BackendSystemState | null>(null);

  useEffect(() => {
    let isMounted = true;
    if (!isConnected) {
      setSystemState(null);
      return;
    }

    const checkBackend = async () => {
      try {
        const target = (apiUrl || "http://127.0.0.1:8000").replace(/\/+$/, "");
        const stateData = await fetchSystemState(target).catch(() => null);
        if (stateData && isMounted) {
          setSystemState(stateData);
        }
      } catch {
        // Backend offline or unreachable
      }
    };
    checkBackend();
    return () => {
      isMounted = false;
    };
  }, [apiUrl, isConnected]);

  const openLaunchModal = () => setIsLaunchModalOpen(true);
  const closeLaunchModal = () => setIsLaunchModalOpen(false);

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <div className="min-h-screen flex flex-col text-[var(--text-main)] overflow-x-hidden selection:bg-emerald-500/20">
      <Navbar mode="landing" onLaunchDemo={openLaunchModal} />

      <ConsortiumLaunchOverlay
        isOpen={isLaunchModalOpen}
        onClose={closeLaunchModal}
      />

      <main className="flex-1 w-full">

        {/* ================================================================
            SECTION 1: HERO
            Clean, product-focused. Problem + Solution in one view.
            ================================================================ */}
        <section className="max-w-[1200px] mx-auto px-4 sm:px-6 pt-10 sm:pt-16 pb-12 sm:pb-20">

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-10 lg:gap-14 items-center">
            {/* Left: Value Proposition */}
            <div className="max-w-xl">
              {/* Single concise category tag */}
              <div className="inline-flex items-center gap-2 mb-5 text-[11px] font-semibold uppercase tracking-widest text-emerald-600 dark:text-emerald-400">
                <span className="w-4 h-px bg-emerald-500 shrink-0" />
                <span>{t("landing_hero_badge")}</span>
              </div>

              <h1 className="font-display text-3xl sm:text-4xl lg:text-[44px] font-extrabold tracking-tight text-[var(--text-heading)] leading-[1.12] mb-5">
                {t("landing_hero_title")}
              </h1>

              <p className="text-[15px] sm:text-base text-[var(--text-muted)] leading-relaxed mb-7 max-w-lg">
                {t("landing_hero_desc")}
              </p>

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3 mb-8">
                <button
                  type="button"
                  onClick={openLaunchModal}
                  className="w-full sm:w-auto h-11 sm:h-12 px-6 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs sm:text-sm inline-flex items-center justify-center gap-2 shadow-md shadow-emerald-600/20 border border-emerald-500/30 transition-all duration-150 active:scale-95 cursor-pointer whitespace-nowrap"
                >
                  <Activity className="w-4 h-4 shrink-0 text-emerald-100" />
                  <span className="whitespace-nowrap">{t("landing_cta_launch")}</span>
                  <ArrowRight className="w-4 h-4 shrink-0 text-emerald-200" />
                </button>

                <button
                  type="button"
                  onClick={() => scrollToSection("results")}
                  className="w-full sm:w-auto h-11 sm:h-12 px-6 rounded-xl bg-[var(--bg-card)]/90 hover:bg-[var(--bg-card-inner)] text-[var(--text-heading)] font-semibold text-xs sm:text-sm inline-flex items-center justify-center gap-2 border border-[var(--border-card)] shadow-xs backdrop-blur-md transition-all duration-150 active:scale-95 cursor-pointer whitespace-nowrap group"
                >
                  <Eye className="w-4 h-4 shrink-0 text-[var(--text-muted)] group-hover:text-emerald-500 transition-colors" />
                  <span className="whitespace-nowrap">{t("landing_cta_results")}</span>
                  <ArrowRight className="w-4 h-4 text-[var(--text-muted)] group-hover:translate-x-0.5 transition-all shrink-0" />
                </button>
              </div>

              {/* Three key facts - plain text with dot separators */}
              <div className="flex flex-col gap-2 text-[12px] text-[var(--text-muted)]">
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>{t("landing_hero_bullet_1")}</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>{t("landing_hero_bullet_2")}</span>
                </div>
                <div className="flex items-center gap-2">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span>{t("landing_hero_bullet_3")}</span>
                </div>
              </div>
            </div>

            {/* Right: Consortium Topology (the interactive network map) */}
            <div className="relative">
              <FederatedConsortiumTopology
                onLaunchDemo={openLaunchModal}
                isModalOpen={isLaunchModalOpen}
                hospitals={systemState?.hospitals}
                activeEpsilon={systemState?.privacy?.epsilon}
                telemetryRounds={systemState?.rounds}
                isLive={Boolean(systemState?.is_live)}
              />
            </div>
          </div>

          {/* Key Metrics Strip - horizontal rule + inline stats, no card grid */}
          <div className="mt-14 sm:mt-18 pt-8 border-t border-[var(--border-card)]">
            <div className="flex items-center gap-2 mb-5">
              <span
                className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                  systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                    ? "bg-emerald-500 animate-pulse"
                    : "bg-slate-400"
                }`}
              />
              <span className="font-mono text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)]">
                {systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                  ? `${t("landing_kpi_live_title")} - ${systemState.history_rounds} ${t("landing_kpi_rounds")} - ${systemState.model_type.toUpperCase()}`
                  : t("landing_kpi_ref_title")}
              </span>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-px border border-[var(--border-card)] rounded-2xl overflow-hidden bg-[var(--border-card)] shadow-xs">
              {[
                {
                  value: systemState?.hospitals?.length ? String(systemState.hospitals.length) : "4",
                  label: t("landing_metric_nodes"),
                  sub: systemState?.is_live ? t("landing_metric_nodes_live_sub") : t("landing_metric_nodes_sub"),
                  icon: Server,
                  accentColor: "text-emerald-500",
                },
                {
                  value: "0 B",
                  label: t("landing_metric_leakage"),
                  sub: t("landing_metric_leakage_sub"),
                  icon: Lock,
                  accentColor: "text-emerald-500",
                },
                {
                  value: "128-Bit",
                  label: t("landing_metric_crypto"),
                  sub: t("landing_metric_crypto_sub"),
                  icon: Cpu,
                  accentColor: "text-sky-500",
                },
                {
                  value:
                    systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? `${systemState.rounds[systemState.rounds.length - 1].train_accuracy}%`
                      : "98.4%",
                  label:
                    systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? t("landing_metric_val_acc")
                      : t("landing_metric_auc"),
                  sub:
                    systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? `${t("landing_kpi_round")} ${systemState.rounds.length} (${systemState.peft_mode ? systemState.peft_mode.toUpperCase() : "Full"})`
                      : t("landing_metric_auc_sub"),
                  icon: Activity,
                  accentColor: "text-emerald-500",
                },
                {
                  value:
                    systemState?.privacy?.epsilon && systemState.privacy.epsilon > 0
                      ? `\u03B5 = ${systemState.privacy.epsilon.toFixed(2)}`
                      : "\u03B5 \u2264 2.0",
                  label: t("landing_metric_dp"),
                  sub:
                    systemState?.privacy?.epsilon && systemState.privacy.epsilon > 0
                      ? t("landing_metric_renyi_dp")
                      : t("landing_metric_dp_sub"),
                  icon: ShieldCheck,
                  accentColor: "text-indigo-500",
                },
                {
                  value: "99.6%",
                  label: t("landing_metric_efficiency"),
                  sub: t("landing_metric_efficiency_sub"),
                  icon: Zap,
                  accentColor: "text-amber-500",
                },
              ].map((m, i) => {
                const IconComponent = m.icon;
                return (
                  <div
                    key={i}
                    className="bg-[var(--bg-card)]/85 backdrop-blur-sm p-4 sm:p-5 transition-colors hover:bg-[var(--bg-card-inner)] flex flex-col justify-between text-left items-start h-[132px] sm:h-[142px]"
                  >
                    {/* Row 1: Fixed Height Header (Icon + Label) */}
                    <div className="flex items-center gap-1.5 h-6 w-full">
                      <IconComponent className={`w-3.5 h-3.5 ${m.accentColor} shrink-0`} />
                      <span className="text-[11px] font-semibold text-[var(--text-heading)] leading-none truncate">
                        {m.label}
                      </span>
                    </div>

                    {/* Row 2: Fixed Baseline Metric Value */}
                    <div className="h-8 flex items-baseline font-display text-xl sm:text-2xl font-extrabold text-[var(--text-heading)] tracking-tight leading-none">
                      {m.value}
                    </div>

                    {/* Row 3: Fixed Height Subtitle with top subtle divider */}
                    <div className="h-5 flex items-center text-[10px] sm:text-[11px] text-[var(--text-muted)] font-mono leading-none truncate w-full border-t border-[var(--border-card)]/50 pt-1">
                      {m.sub}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        </section>

        {/* ================================================================
            SECTION 2: HOW IT WORKS
            3-step horizontal pipeline. Simple, clean, no cards.
            ================================================================ */}
        <section className="border-y border-[var(--border-card)] bg-[var(--bg-card-inner)]/60 backdrop-blur-sm">
          <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-12 sm:py-16">
            <div className="text-center mb-10">
              <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-2">
                {t("landing_how_badge")}
              </div>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)]">
                {t("landing_how_title")}
              </h2>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-8 md:gap-4 relative">
              {/* Connector lines (desktop only) */}
              <div className="hidden md:block absolute top-8 left-[33.3%] right-[33.3%] h-px bg-[var(--border-card)]" />

              {[
                { num: "01", icon: Building2, title: t("landing_how_s1_title"), desc: t("landing_how_s1_desc"), color: "emerald" },
                { num: "02", icon: Lock, title: t("landing_how_s2_title"), desc: t("landing_how_s2_desc"), color: "blue" },
                { num: "03", icon: Eye, title: t("landing_how_s3_title"), desc: t("landing_how_s3_desc"), color: "emerald" },
              ].map((step, i) => (
                <div key={i} className="text-center relative">
                  <div className={`w-16 h-16 mx-auto rounded-2xl flex items-center justify-center mb-4 ${
                    step.color === "emerald"
                      ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20"
                      : "bg-blue-500/10 text-blue-600 dark:text-blue-400 border border-blue-500/20"
                  }`}>
                    <step.icon className="w-6 h-6" />
                  </div>
                  <div className="font-mono text-[10px] font-bold text-[var(--text-muted)] mb-1">STEP {step.num}</div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)] mb-2">{step.title}</h3>
                  <p className="text-sm text-[var(--text-muted)] leading-relaxed max-w-xs mx-auto">{step.desc}</p>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ================================================================
            SECTION 3: PLATFORM CAPABILITIES (Tab-based Showcase)
            Show what the platform actually does with real UI previews.
            ================================================================ */}
        <section id="results" className="scroll-mt-20 max-w-[1200px] mx-auto px-4 sm:px-6 py-12 sm:py-16">
          <div className="flex flex-col sm:flex-row sm:items-end justify-between gap-3 mb-6">
            <div>
              <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-1">
                {t("landing_sec_results_badge")}
              </div>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)]">
                {t("landing_results_title")}
              </h2>
              <p className="text-sm text-[var(--text-muted)] mt-1 max-w-xl">
                {t("landing_results_subtitle")}
              </p>
            </div>

            {/* Tab Switcher: Responsive 2x2 Grid on Mobile (No Horizontal Scroll), Flex Pill Bar on Desktop */}
            <div className="grid grid-cols-2 sm:flex items-center gap-1.5 sm:gap-1 p-1 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-card)] w-full sm:w-auto shrink-0 overflow-hidden">
              {(["xai", "rag", "convergence", "report"] as const).map((tab) => (
                <button
                  key={tab}
                  type="button"
                  onClick={() => setActiveTab(tab)}
                  className={`w-full sm:w-auto px-2 sm:px-3 py-2 sm:py-1.5 rounded-lg text-[11px] sm:text-xs font-semibold text-center transition-all ${
                    activeTab === tab
                      ? "bg-[var(--text-heading)] text-[var(--bg-app)] shadow-sm font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)]/50"
                  }`}
                >
                  {tab === "xai" && t("landing_tab_xai")}
                  {tab === "rag" && t("landing_tab_rag")}
                  {tab === "convergence" && t("landing_tab_convergence")}
                  {tab === "report" && t("landing_tab_report")}
                </button>
              ))}
            </div>
          </div>

          {/* Tab Content Enclosed in Clinical Error Boundary */}
          <ClinicalErrorBoundary language={language}>
            {/* Tab 1: 3D Explainable Diagnostics */}
            {activeTab === "xai" && (
            <div className="card p-4 sm:p-6">
              <div className="grid grid-cols-1 lg:grid-cols-5 gap-5 items-stretch">
                <div className="relative lg:col-span-3 min-h-[380px] sm:min-h-[460px] h-[380px] sm:h-[460px] lg:h-full rounded-2xl overflow-hidden border border-[var(--border-card)] bg-[var(--bg-card-inner)] shadow-xs">
                  <LungViewer3D className="absolute inset-0" mode="anatomy" labels={false} />
                  <div className="pointer-events-none absolute left-3 top-3 inline-flex items-center gap-1.5 rounded-lg border border-[var(--border-card)] bg-[var(--bg-card)]/90 px-2.5 py-1 font-mono text-[10px] text-[var(--text-muted)] backdrop-blur">
                    <Eye className="w-3 h-3 text-emerald-500" />
                    <span>{t("lung3d_locked_preview")}</span>
                  </div>
                </div>
                <div className="lg:col-span-2 flex flex-col justify-between gap-4">
                  <div>
                    <h3 className="font-display text-lg font-bold text-[var(--text-heading)]">
                      {t("landing_xai3d_title")}
                    </h3>
                    <p className="mt-1 text-sm text-[var(--text-muted)] leading-relaxed">
                      {t("landing_xai3d_desc")}
                    </p>
                  </div>

                  <ol className="space-y-2">
                    {[
                      { icon: Database, title: t("landing_xai3d_s1_t"), desc: t("landing_xai3d_s1_d"), tag: "ViT-L/16" },
                      { icon: Activity, title: t("landing_xai3d_s2_t"), desc: t("landing_xai3d_s2_d"), tag: "MHSA L-3" },
                      { icon: Layers, title: t("landing_xai3d_s3_t"), desc: t("landing_xai3d_s3_d"), tag: "Spatial Map" },
                      { icon: CheckCircle2, title: t("landing_xai3d_s4_t"), desc: t("landing_xai3d_s4_d"), tag: "CDSS Audit" },
                    ].map((step, i) => (
                      <li key={i} className="flex gap-3 p-2.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)]">
                        <span className="mt-0.5 flex h-7 w-7 shrink-0 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                          <step.icon className="w-3.5 h-3.5" />
                        </span>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-bold text-[var(--text-heading)] truncate">{step.title}</span>
                            <span className="font-mono text-[9px] font-bold px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 shrink-0">{step.tag}</span>
                          </div>
                          <span className="block text-[11px] text-[var(--text-muted)] leading-relaxed mt-0.5">{step.desc}</span>
                        </div>
                      </li>
                    ))}
                  </ol>

                  <p className="text-[11px] text-[var(--text-muted)] p-3 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)]">
                    {t("landing_xai3d_note")}
                  </p>

                  <button
                    type="button"
                    onClick={openLaunchModal}
                    className="btn-primary self-start"
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
                  <p className="text-xs text-[var(--text-muted)]">{t("sec3_rag_desc")}</p>
                </div>
                <span className="px-2.5 py-1 rounded-md text-[10px] font-mono font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  {systemState?.reference_cases?.length
                    ? `${t("landing_rag_live_bank")} (${systemState.reference_cases.length} ${t("landing_rag_verified_cases")})`
                    : t("landing_tab2_badge")}
                </span>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {(Array.isArray(systemState?.reference_cases) && systemState.reference_cases.length > 0
                  ? systemState.reference_cases.slice(0, 2)
                  : [
                      {
                        case_id: "REF-ONCO-401",
                        diagnosis: t("landing_tab2_c1_diag"),
                        similarity: 96.4,
                        clinical_notes: t("landing_tab2_c1_desc"),
                        site_id: t("landing_tab2_c1_meta"),
                        biopsy_confirmed: true,
                        anatomical_site: "Thorax / Left Lobe",
                      },
                      {
                        case_id: "REF-ONCO-402",
                        diagnosis: t("landing_tab2_c2_diag"),
                        similarity: 92.1,
                        clinical_notes: t("landing_tab2_c2_desc"),
                        site_id: t("landing_tab2_c2_meta"),
                        biopsy_confirmed: true,
                        anatomical_site: "Right Lower Lobe",
                      },
                    ]
                ).map((c, idx) => (
                  <div
                    key={c.case_id}
                    className={`card-inner p-4 space-y-2.5 border-l-4 ${
                      idx === 0 ? "border-l-emerald-500" : "border-l-blue-500"
                    }`}
                  >
                    <div className="flex items-center justify-between">
                      <span
                        className={`font-mono text-xs font-bold ${
                          idx === 0
                            ? "text-emerald-600 dark:text-emerald-400"
                            : "text-blue-600 dark:text-blue-400"
                        }`}
                      >
                        {c.case_id}
                      </span>
                      <span
                        className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                          idx === 0
                            ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                            : "bg-blue-500/10 text-blue-600 dark:text-blue-400"
                        }`}
                      >
                        {c.similarity}% {t("landing_rag_match")}
                      </span>
                    </div>
                    <div className="text-xs font-semibold text-[var(--text-heading)]">
                      {language === "TR"
                        ? c.diagnosis?.toLowerCase().includes("melanoma")
                          ? "Malign Melanom"
                          : c.diagnosis?.toLowerCase().includes("basal")
                          ? "Bazal Hücreli Karsinom"
                          : c.diagnosis?.toLowerCase().includes("nevus")
                          ? "Benign Displastik Nevüs"
                          : c.diagnosis
                        : c.diagnosis}
                    </div>
                    <p className="text-xs text-[var(--text-muted)] leading-relaxed line-clamp-3">
                      {language === "TR" && (c.case_id.startsWith("PAT_ISIC") || c.clinical_notes?.toLowerCase().includes("melanoma"))
                        ? idx === 0
                          ? "Gövde bölgesinde yerleşik, histopatoloji (biyopsi) ile doğrulanmış Malign Melanom. Eksantrik hiperpigmentasyon ve periferik çizgilenmeler mevcut. Geniş lokal eksizyon endikedir."
                          : "Sırt bölgesinde yerleşik, konkomitant biyopsi ile doğrulanmış yüzeyel yayılan Malign Melanom. Atipik pigmentsel retikülasyon ve lezyon kenarlarında asimetri izlenmektedir."
                        : c.clinical_notes}
                    </p>
                    <div className="text-[10px] font-mono text-[var(--text-muted)] pt-1 border-t border-[var(--border-subtle)] flex items-center justify-between">
                      <span className="truncate">
                        {language === "TR" && c.site_id?.includes("BCN-20000")
                          ? "BCN-20000 Deri Kanseri Merkezi (Barselona)"
                          : c.site_id || (language === "TR" ? "ISIC Çok Merkezli Bankası" : "ISIC Multi-Center Bank")}
                      </span>
                      {c.biopsy_confirmed && (
                        <span className="text-emerald-600 dark:text-emerald-400 font-semibold shrink-0 ml-1">
                          {t("landing_rag_biopsy_yes")}
                        </span>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Tab 3: Training Monitor */}
          {activeTab === "convergence" && (
            <div className="card p-4 sm:p-6 space-y-4">
              <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-[var(--border-subtle)]">
                <div>
                  <h3 className="font-display text-base font-bold text-[var(--text-heading)]">
                    {systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? `${t("landing_tab3_live_title")} (${systemState.history_rounds} ${t("landing_kpi_rounds")} · ${systemState.model_type.toUpperCase()})`
                      : t("landing_tab3_title")}
                  </h3>
                  <p className="text-xs text-[var(--text-muted)]">
                    {systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? (language === "TR"
                          ? `${systemState.data_source} üzerinde ${systemState.peft_mode ? systemState.peft_mode.toUpperCase() : "tam konsensüs"} adaptörleri ile eğitildi.`
                          : `Trained on ${systemState.data_source} with ${systemState.peft_mode ? systemState.peft_mode.toUpperCase() : "full consensus"} adapters.`)
                      : t("landing_tab3_desc")}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <span
                    className={`w-2 h-2 rounded-full ${
                      systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                        ? "bg-emerald-500 animate-pulse"
                        : "bg-sky-500"
                    }`}
                  />
                  <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                    {systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                      ? t("landing_tab3_live_synced")
                      : t("landing_tab3_ref_target")}
                  </span>
                </div>
              </div>

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
                    {systemState?.rounds && Array.isArray(systemState.rounds) && systemState.rounds.length > 0 ? (
                      systemState.rounds.map((rd) => (
                        <tr
                          key={rd.round_num}
                          className={rd.threat_detected ? "bg-amber-500/[0.04]" : ""}
                        >
                          <td className="py-2 px-3 font-bold text-[var(--text-heading)]">
                            {t("landing_tab3_col_round")} {rd.round_num}
                          </td>
                          <td className="py-2 px-3">{rd.train_loss.toFixed(3)}</td>
                          <td className="py-2 px-3 text-emerald-600 dark:text-emerald-400">
                            {rd.train_accuracy}%
                          </td>
                          <td className="py-2 px-3">{rd.f1_score.toFixed(3)}</td>
                          <td className="py-2 px-3">{rd.bandwidth}</td>
                          <td className="py-2 px-3 text-emerald-500 flex items-center gap-1 font-sans">
                            <CheckCircle2 className="w-3.5 h-3.5" />{" "}
                            {rd.status === "completed" ? t("landing_tab3_verified") : rd.status}
                          </td>
                        </tr>
                      ))
                    ) : (
                      <>
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
                      </>
                    )}
                  </tbody>
                </table>
              </div>

              <div className="text-xs text-[var(--text-muted)] pt-2">
                {systemState?.is_live && systemState?.rounds && systemState.rounds.length > 0
                  ? `Active Session: ${systemState.rounds.length} rounds synchronized from server checkpoint.`
                  : `${t("landing_tab3_comm_summary")} - `}
                {!systemState?.is_live && (
                  <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                    {t("landing_tab3_savings")}
                  </span>
                )}
              </div>
            </div>
          )}

          {/* Tab 4: Reporting */}
          {activeTab === "report" && (
            <div className="card p-4 sm:p-6">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <div className="card-inner p-5 space-y-3">
                  <div className="flex items-center gap-2 text-emerald-500">
                    <FileText className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">{t("sec3_report_title")}</span>
                  </div>
                  <h4 className="text-sm font-bold text-[var(--text-heading)]">{t("landing_tab4_pdf_title")}</h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">{t("landing_tab4_pdf_desc")}</p>
                  <button
                    onClick={openLaunchModal}
                    className="inline-flex items-center gap-2 px-3.5 py-2 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-semibold hover:bg-emerald-500/20 transition-colors"
                  >
                    <FileText className="w-3.5 h-3.5" />
                    <span>{t("landing_tab4_pdf_btn")}</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="card-inner p-5 space-y-3">
                  <div className="flex items-center gap-2 text-blue-500">
                    <Volume2 className="w-4 h-4" />
                    <span className="text-xs font-bold uppercase tracking-wider">{t("sec3_voice_title")}</span>
                  </div>
                  <h4 className="text-sm font-bold text-[var(--text-heading)]">{t("landing_tab4_voice_title")}</h4>
                  <p className="text-xs text-[var(--text-muted)] leading-relaxed">{t("landing_tab4_voice_desc")}</p>
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
          )}
          </ClinicalErrorBoundary>
        </section>

        {/* ================================================================
            SECTION 4: SECURITY (Compact, icon-based, not verbose cards)
            ================================================================ */}
        <section id="security" className="scroll-mt-20 border-y border-[var(--border-card)] bg-[var(--bg-card-inner)]/60 backdrop-blur-sm">
          <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-12 sm:py-16">
            <div className="text-center mb-10">
              <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-1">
                {t("landing_sec_security_badge")}
              </div>
              <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)]">
                {t("landing_security_title")}
              </h2>
              <p className="text-sm text-[var(--text-muted)] mt-2 max-w-lg mx-auto">
                {t("landing_security_subtitle")}
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 items-stretch">
              {[
                { icon: Database, title: t("landing_pillar_1_title"), desc: t("landing_pillar_1_desc"), tag: t("landing_p1_tag"), accent: "emerald" },
                { icon: ShieldCheck, title: t("landing_pillar_2_title"), desc: t("landing_pillar_2_desc"), tag: t("landing_p2_tag"), accent: "blue" },
                { icon: Lock, title: t("landing_pillar_3_title"), desc: t("landing_pillar_3_desc"), tag: t("landing_p3_tag"), accent: "emerald" },
                { icon: AlertTriangle, title: t("landing_pillar_4_title"), desc: t("landing_pillar_4_desc"), tag: t("landing_p4_tag"), accent: "amber" },
              ].map((pillar, i) => (
                <div
                  key={i}
                  className="bg-[var(--bg-card)]/80 backdrop-blur-sm rounded-2xl p-5 border border-[var(--border-card)] flex flex-col h-full shadow-xs transition-all hover:border-emerald-500/30"
                >
                  <div className="flex-1 flex flex-col">
                    <div className={`w-10 h-10 rounded-xl flex items-center justify-center mb-3.5 ${
                      pillar.accent === "emerald" ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20" :
                      pillar.accent === "blue" ? "bg-sky-500/10 text-sky-600 dark:text-sky-400 border border-sky-500/20" :
                      "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20"
                    }`}>
                      <pillar.icon className="w-5 h-5" />
                    </div>
                    <h3 className="text-sm font-bold text-[var(--text-heading)] mb-2 min-h-[42px] flex items-start leading-snug">
                      {pillar.title}
                    </h3>
                    <p className="text-xs text-[var(--text-muted)] leading-relaxed flex-1">
                      {pillar.desc}
                    </p>
                  </div>

                  {/* Strictly Anchored Bottom Tag Bar: Identical height, padding and baseline across all 4 cards */}
                  <div className="mt-5 pt-3 border-t border-[var(--border-card)]/60 min-h-[44px] flex items-center">
                    <span className="text-[10px] sm:text-[11px] font-mono font-semibold text-[var(--text-muted)] leading-snug">
                      {pillar.tag}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* ================================================================
            SECTION 5: ARCHITECTURE PIPELINE (Compact horizontal flow)
            ================================================================ */}
        <section id="architecture" className="scroll-mt-20 max-w-[1200px] mx-auto px-4 sm:px-6 py-12 sm:py-16">
          <div className="mb-6">
            <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-1">
              {t("landing_sec_arch_badge")}
            </div>
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)]">
              {t("landing_arch_title")}
            </h2>
            <p className="text-sm text-[var(--text-muted)] mt-1 max-w-xl">
              {t("landing_arch_subtitle")}
            </p>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
            {[
              { num: "01", title: t("landing_arch_s1_title"), desc: t("landing_arch_s1_desc"), tag: "ViT Backbone" },
              { num: "02", title: t("landing_arch_s2_title"), desc: t("landing_arch_s2_desc"), tag: "FFA-LoRA" },
              { num: "03", title: t("landing_arch_s3_title"), desc: t("landing_arch_s3_desc"), tag: "CKKS 128b" },
              { num: "04", title: t("landing_arch_s4_title"), desc: t("landing_arch_s4_desc"), tag: "FedDyn" },
              { num: "05", title: t("landing_arch_s5_title"), desc: t("landing_arch_s5_desc"), tag: "CDSS" },
            ].map((stage, i) => (
              <div key={i} className="card-inner p-4 flex flex-col">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-[10px] font-mono font-bold text-emerald-500">{t("landing_arch_stage_label")} {stage.num}</span>
                  <span className="font-mono text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold">{stage.tag}</span>
                </div>
                <h3 className="text-xs font-bold text-[var(--text-heading)] mb-1">{stage.title}</h3>
                <p className="text-[11px] text-[var(--text-muted)] leading-relaxed">{stage.desc}</p>
              </div>
            ))}
          </div>
        </section>

        {/* ================================================================
            SECTION 6: BENCHMARK TABLE (Clean, data-dense)
            ================================================================ */}
        <section id="benchmarks" className="scroll-mt-20 max-w-[1200px] mx-auto px-4 sm:px-6 pb-12 sm:pb-16">
          <div className="mb-6">
            <div className="text-[11px] font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-1">
              {t("landing_sec_bench_badge")}
            </div>
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)]">
              {t("landing_bench_title")}
            </h2>
            <p className="text-sm text-[var(--text-muted)] mt-1 max-w-xl">
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
                  <td className="py-3 px-3 font-bold text-emerald-600 dark:text-emerald-400 font-sans">Fed-XRay (FFA-LoRA + FedDyn)</td>
                  <td className="py-3 px-3 text-emerald-600 dark:text-emerald-400 font-bold">0.984 *</td>
                  <td className="py-3 px-3">4.2 MB / round</td>
                  <td className="py-3 px-3 text-emerald-500 font-sans">Patient DP + CKKS</td>
                  <td className="py-3 px-3 text-emerald-500 font-sans">{t("bench_val_trimmed_active")}</td>
                  <td className="py-3 px-3 text-emerald-600 dark:text-emerald-400 font-sans">{t("bench_val_high_aligned")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_centralized")}</td>
                  <td className="py-2.5 px-3">0.987 *</td>
                  <td className="py-2.5 px-3 text-[var(--text-muted)] font-sans">{t("bench_val_na_centralized")}</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_leakage_full")}</td>
                  <td className="py-2.5 px-3 text-[var(--text-muted)] font-sans">N/A</td>
                  <td className="py-2.5 px-3 font-sans">{t("bench_val_optimal")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_fedavg")}</td>
                  <td className="py-2.5 px-3">0.912 *</td>
                  <td className="py-2.5 px-3 text-rose-500">340 MB / round</td>
                  <td className="py-2.5 px-3 text-[var(--text-muted)] font-sans">{t("bench_val_plaintext")}</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_vulnerable")}</td>
                  <td className="py-2.5 px-3 font-sans">{t("bench_val_drift")}</td>
                </tr>
                <tr>
                  <td className="py-2.5 px-3 font-sans text-[var(--text-heading)]">{t("bench_row_standalone")}</td>
                  <td className="py-2.5 px-3">0.824 *</td>
                  <td className="py-2.5 px-3 text-[var(--text-muted)] font-sans">{t("bench_val_isolated")}</td>
                  <td className="py-2.5 px-3 text-emerald-500 font-sans">{t("bench_val_no_transfer")}</td>
                  <td className="py-2.5 px-3 text-[var(--text-muted)] font-sans">N/A</td>
                  <td className="py-2.5 px-3 text-rose-500 font-sans">{t("bench_val_overfitting")}</td>
                </tr>
              </tbody>
            </table>
            <p className="mt-3 text-[10px] text-[var(--text-muted)] leading-relaxed border-t border-[var(--border-subtle)] pt-3">
              * {t("footer_bench_note")}
            </p>
          </div>
        </section>

        {/* ================================================================
            SECTION 7: FINAL CTA
            ================================================================ */}
        <section className="border-t border-[var(--border-card)] bg-[var(--bg-card-inner)]/60 backdrop-blur-sm">
          <div className="max-w-[1200px] mx-auto px-4 sm:px-6 py-14 sm:py-20 text-center">
            <h2 className="font-display text-2xl sm:text-3xl font-bold text-[var(--text-heading)] mb-3">
              {t("landing_banner_title")}
            </h2>
            <p className="text-sm text-[var(--text-muted)] max-w-lg mx-auto mb-6">
              {t("landing_banner_desc")}
            </p>
            <button
              onClick={openLaunchModal}
              className="btn-primary text-sm px-8 py-3.5 rounded-xl w-full sm:w-auto"
            >
              <ShieldCheck className="w-4 h-4" />
              <span>{t("landing_banner_btn")}</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          </div>
        </section>
      </main>

      {/* Footer */}
      <Footer maxWidth="max-w-[1200px]" />
    </div>
  );
}
