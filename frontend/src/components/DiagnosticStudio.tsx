"use client";

import React, { useState, useEffect, useRef } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { DiagnosisResult } from "@/types";
import { Stethoscope, Search, Activity, Lock, ArrowUp, Upload, Box, ShieldCheck } from "lucide-react";
import LungViewer3D from "@/components/LungViewer3D";
import type { LungViewMode } from "@/lib/scanTexture";

interface DiagnosticStudioProps {
  diagnosis: DiagnosisResult | null;
  onDiagnose: (classIndex?: number, opacity?: number, colormap?: string, imageBase64?: string) => void;
  isLoading: boolean;
  isLocked?: boolean;
  onScrollToTraining?: () => void;
}

const CLASS_COLORS_BAR = ["bg-accent-500", "bg-amber-500", "bg-red-500"];

export default function DiagnosticStudio({
  diagnosis,
  onDiagnose,
  isLoading,
  isLocked = false,
  onScrollToTraining,
}: DiagnosticStudioProps) {
  const { t } = useLanguage();
  const [opacity, setOpacity] = useState(0.55);
  const [colormap, setColormap] = useState("Hot");
  const [targetClass, setTargetClass] = useState<number | undefined>(undefined);
  const [viewMode, setViewMode] = useState<LungViewMode>("anatomy");
  const rawRef = useRef<HTMLCanvasElement | null>(null);
  const xaiRef = useRef<HTMLCanvasElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = () => {
      const b64 = reader.result as string;
      onDiagnose(undefined, opacity, colormap, b64);
    };
    reader.readAsDataURL(file);
    e.target.value = "";
  };

  useEffect(() => {
    if (!diagnosis) return;
    setViewMode(diagnosis.heatmap_available !== false && diagnosis.heatmap ? "gradcam" : "scan");
  }, [diagnosis]);

  // Render raw X-Ray
  useEffect(() => {
    if (!diagnosis || !rawRef.current) return;
    const canvas = rawRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const d = diagnosis.raw_image;
    const s = d.length;
    canvas.width = s; canvas.height = s;
    const img = ctx.createImageData(s, s);
    for (let r = 0; r < s; r++) for (let c = 0; c < s; c++) {
      const val = d[r][c];
      const norm = val < 0.0 ? val * 0.5 + 0.5 : val;
      const v = Math.floor(Math.min(1.0, Math.max(0.0, norm)) * 255);
      const i = (r * s + c) * 4;
      img.data[i] = v; img.data[i+1] = v; img.data[i+2] = v; img.data[i+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [diagnosis]);

  // Render Grad-CAM overlay
  useEffect(() => {
    if (!diagnosis || !diagnosis.heatmap || !xaiRef.current) return;
    const canvas = xaiRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const raw = diagnosis.raw_image;
    const heat = diagnosis.heatmap;
    const s = raw.length;
    canvas.width = s; canvas.height = s;
    const img = ctx.createImageData(s, s);
    for (let r = 0; r < s; r++) for (let c = 0; c < s; c++) {
      const rawVal = raw[r][c];
      const norm = rawVal < 0.0 ? rawVal * 0.5 + 0.5 : rawVal;
      const rv = Math.min(1.0, Math.max(0.0, norm)) * 255;
      const hv = heat[r][c];
      let hr = 0, hg = 0, hb = 0;
      if (colormap === "Hot") {
        hr = Math.min(255, hv * 3 * 255);
        hg = Math.max(0, Math.min(255, (hv - 0.33) * 3 * 255));
        hb = Math.max(0, Math.min(255, (hv - 0.66) * 3 * 255));
      } else {
        hr = Math.min(255, Math.max(0, 1.5 - Math.abs(hv * 4 - 3)) * 255);
        hg = Math.min(255, Math.max(0, 1.5 - Math.abs(hv * 4 - 2)) * 255);
        hb = Math.min(255, Math.max(0, 1.5 - Math.abs(hv * 4 - 1)) * 255);
      }
      const a = opacity;
      const i = (r * s + c) * 4;
      img.data[i] = Math.floor((1-a)*rv + a*hr);
      img.data[i+1] = Math.floor((1-a)*rv + a*hg);
      img.data[i+2] = Math.floor((1-a)*rv + a*hb);
      img.data[i+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [diagnosis, opacity, colormap]);

  const classLabels = [t("sec1_normal"), t("sec1_pneumonia"), t("sec1_covid")];
  const heatAvailable = !!diagnosis && diagnosis.heatmap_available !== false && !!diagnosis.heatmap;

  return (
    <section id="section-diagnosis" className="mb-5 sm:mb-6 scroll-mt-20">
      {/* Section Header */}
      <div className="section-bar mb-3.5 sm:mb-4 flex-col sm:flex-row items-stretch sm:items-center">
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center shrink-0 ${
            isLocked ? "bg-slate-200/50 dark:bg-navy-800 text-[var(--text-muted)]" : "bg-purple-500/10 text-purple-500 dark:text-purple-400"
          }`}>
            <Stethoscope className="w-4 h-4 sm:w-5 sm:h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">{t("sec3_title")}</h2>
              {isLocked && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                  <Lock className="w-3 h-3" />
                  {t("lock_diag_badge")}
                </span>
              )}
            </div>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)]">{t("sec3_subtitle")}</p>
          </div>
        </div>

        {!isLocked && diagnosis && (
          <button
            onClick={() => onDiagnose(targetClass, opacity, colormap)}
            disabled={isLoading}
            className="btn-primary w-full sm:w-auto mt-2.5 sm:mt-0"
          >
            <Search className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isLoading ? "animate-spin" : ""}`} />
            <span>{t("sec3_btn_run_diag")}</span>
          </button>
        )}
      </div>

      {/* Locked State Container */}
      {isLocked ? (
        <div className="card overflow-hidden border border-[var(--border-card)] bg-[var(--bg-card)]">
          <div className="grid grid-cols-1 md:grid-cols-2 items-stretch">
            {/* Left: Cryptographic Consensus Enclave Status (Clean, Medical SaaS) */}
            <div className="p-6 sm:p-8 flex flex-col justify-between bg-[var(--bg-card-inner)] border-b md:border-b-0 md:border-r border-[var(--border-subtle)]">
              <div>
                <div className="flex items-center justify-between mb-4">
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[10px] font-mono font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                    <Lock className="w-3 h-3" />
                    <span>Weights Locked · Standby</span>
                  </span>
                  <span className="font-mono text-[10px] text-[var(--text-muted)]">
                    Round 0 / 5
                  </span>
                </div>

                <div className="my-6 flex flex-col items-center justify-center text-center">
                  <div className="relative mb-4">
                    <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-amber-500/20 to-emerald-500/20 border border-amber-500/30 flex items-center justify-center shadow-lg">
                      <ShieldCheck className="w-8 h-8 text-amber-500" />
                    </div>
                    <span className="absolute -bottom-1 -right-1 w-4 h-4 rounded-full bg-amber-500 border-2 border-[var(--bg-card)] flex items-center justify-center">
                      <Lock className="w-2.5 h-2.5 text-white" />
                    </span>
                  </div>
                  <h4 className="font-display text-sm font-bold text-[var(--text-heading)]">
                    Federated Model Weights Awaiting Consensus
                  </h4>
                  <p className="text-[11px] text-[var(--text-muted)] max-w-xs mt-1 leading-relaxed">
                    Diagnostic inference and Grad-CAM attention require global foundation model adapters distributed across consortium nodes.
                  </p>
                </div>

                <div className="space-y-2 pt-2">
                  <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px]">
                    <span className="text-[var(--text-muted)]">Hospital Nodes Connected:</span>
                    <span className="font-mono font-bold text-emerald-600 dark:text-emerald-400">4 / 4 Nodes Ready</span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px]">
                    <span className="text-[var(--text-muted)]">Cryptographic Standard:</span>
                    <span className="font-mono font-bold text-sky-600 dark:text-sky-400">RLWE CKKS 128-Bit</span>
                  </div>
                  <div className="flex items-center justify-between p-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-subtle)] text-[11px]">
                    <span className="text-[var(--text-muted)]">Differential Privacy Budget:</span>
                    <span className="font-mono font-bold text-violet-600 dark:text-violet-400">ε ≤ 2.0, δ = 10^-5</span>
                  </div>
                </div>
              </div>

              <div className="pt-4 border-t border-[var(--border-subtle)] text-[10px] font-mono text-[var(--text-muted)] flex items-center justify-between">
                <span>Consortium Enclave Status:</span>
                <span className="text-amber-600 dark:text-amber-400 font-bold">Synchronizing Parameters</span>
              </div>
            </div>

            {/* Right: Step 3 Locked Action Panel */}
            <div className="flex flex-col items-start justify-center p-6 sm:p-8">
              <div className="w-11 h-11 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mb-3 text-amber-500 shadow-sm">
                <Lock className="w-5 h-5" />
              </div>
              <h3 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)] mb-1.5">
                {t("lock_diag_title")}
              </h3>
              <p className="text-xs text-[var(--text-muted)] max-w-md mb-5 leading-relaxed">
                {t("lock_diag_desc")}
              </p>
              {onScrollToTraining && (
                <button onClick={onScrollToTraining} className="btn-primary inline-flex items-center gap-2 shadow-md">
                  <ArrowUp className="w-4 h-4" />
                  <span>{t("lock_diag_btn")}</span>
                </button>
              )}
            </div>
          </div>
        </div>
      ) : (
        <>
          {/* Diagnostic Target & Scan Ingestion Toolbar */}
          <div className="card p-3 sm:p-3.5 mb-3.5 sm:mb-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <span className="text-xs text-[var(--text-muted)] font-medium mr-1 hidden sm:inline">
                {t("sec3_results_header")}:
              </span>
              {/* Target Pathology Filter Chips */}
              <div className="flex flex-wrap items-center gap-1 bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] p-1 rounded-lg text-xs">
                <button
                  type="button"
                  onClick={() => setTargetClass(undefined)}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    targetClass === undefined
                      ? "bg-purple-500/20 text-purple-600 dark:text-purple-300 font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                  }`}
                >
                  {t("sec3_filter_all")}
                </button>
                <button
                  type="button"
                  onClick={() => setTargetClass(0)}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    targetClass === 0
                      ? "bg-emerald-500/20 text-emerald-600 dark:text-emerald-300 font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                  }`}
                >
                  {t("sec1_normal")}
                </button>
                <button
                  type="button"
                  onClick={() => setTargetClass(1)}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    targetClass === 1
                      ? "bg-amber-500/20 text-amber-600 dark:text-amber-300 font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                  }`}
                >
                  {t("sec1_pneumonia")}
                </button>
                <button
                  type="button"
                  onClick={() => setTargetClass(2)}
                  disabled={isLoading}
                  className={`px-2.5 py-1 rounded-md font-medium transition-all ${
                    targetClass === 2
                      ? "bg-rose-500/20 text-rose-600 dark:text-rose-300 font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                  }`}
                >
                  {t("sec1_covid")}
                </button>
              </div>
            </div>

            {/* Custom Scan File Input */}
            <div className="flex items-center gap-2">
              <input
                type="file"
                ref={fileInputRef}
                accept="image/*"
                className="hidden"
                onChange={handleFileUpload}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                disabled={isLoading}
                className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-[var(--border-card)] bg-[var(--bg-card-inner)] hover:bg-black/5 dark:hover:bg-white/5 text-xs font-semibold text-[var(--text-heading)] transition-all cursor-pointer"
                title={t("sec3_upload_tooltip")}
              >
                <Upload className="w-3.5 h-3.5 text-purple-500 shrink-0" />
                <span>{t("sec3_btn_upload")}</span>
              </button>
            </div>
          </div>

          {diagnosis ? (
        <div className="space-y-3.5 sm:space-y-4">
          {/* Diagnosis Banner */}
          <div className={`card p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-l-4 ${
            diagnosis.predicted_class === 0 ? "border-l-accent-500" :
            diagnosis.predicted_class === 1 ? "border-l-amber-500" : "border-l-red-500"
          }`}>
            <div>
              <div className="flex items-center gap-2 mb-0.5 sm:mb-1">
                <span className="metric-label">{t("sec3_diagnosis")}</span>
                {diagnosis.patient_id && (
                  <span className="font-mono text-[10px] font-bold px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-white/10 text-slate-700 dark:text-slate-300 border border-slate-300/60 dark:border-white/10">
                    {diagnosis.patient_id}
                  </span>
                )}
                {diagnosis.image_source && (
                  <span className="text-[10px] font-medium text-[var(--text-muted)]">
                    ({diagnosis.image_source.startsWith("uploaded") ? t("sec3_source_custom") : t("sec3_source_cohort")})
                  </span>
                )}
              </div>
              <div className="font-display text-lg sm:text-xl font-extrabold text-[var(--text-heading)]">
                {classLabels[diagnosis.predicted_class] || diagnosis.predicted_name}
              </div>
            </div>
            <div className="text-left sm:text-right">
              <div className="metric-label mb-0.5 sm:mb-1">{t("sec3_confidence")}</div>
              <div className="font-display text-lg sm:text-xl font-black text-accent-600 dark:text-accent-400">
                {diagnosis.confidence > 1.0 ? diagnosis.confidence : Math.round(diagnosis.confidence * 1000) / 10}%
              </div>
            </div>
          </div>

          {/* 3D comparison workspace + evidence column */}
          <div className="grid grid-cols-1 xl:grid-cols-12 gap-3.5 sm:gap-4 items-stretch">
            <div className="xl:col-span-7 card p-3 sm:p-4 flex flex-col gap-3">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                    <Box className="w-3.5 h-3.5" />
                  </span>
                  <span className="font-display text-sm font-bold text-[var(--text-heading)]">{t("lung3d_title")}</span>
                </div>
                <div className="flex items-center gap-1 rounded-lg border border-[var(--border-subtle)] bg-[var(--bg-card-inner)] p-1">
                  <button
                    type="button"
                    onClick={() => setViewMode("anatomy")}
                    disabled={false}
                    className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-all disabled:opacity-40 ${
                      viewMode === "anatomy"
                        ? "bg-emerald-500 text-white shadow"
                        : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                    }`}
                  >
                    {t("lung3d_mode_anatomy")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode("scan")}
                    disabled={false}
                    className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-all disabled:opacity-40 ${
                      viewMode === "scan"
                        ? "bg-emerald-500 text-white shadow"
                        : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                    }`}
                  >
                    {t("lung3d_mode_scan")}
                  </button>
                  <button
                    type="button"
                    onClick={() => setViewMode("gradcam")}
                    disabled={!heatAvailable}
                    className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-all disabled:opacity-40 ${
                      viewMode === "gradcam"
                        ? "bg-emerald-500 text-white shadow"
                        : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                    }`}
                  >
                    {t("lung3d_mode_gradcam")}
                  </button>
                </div>
              </div>
              <div className="lung-viewer relative h-[400px] sm:h-[520px]">
                <LungViewer3D
                  className="absolute inset-0"
                  rawImage={diagnosis.raw_image}
                  heatmap={heatAvailable ? diagnosis.heatmap : null}
                  mode={viewMode}
                  opacity={opacity}
                  colormap={colormap}
                  labels={false}
                  showHint={false}
                />
              </div>
              {heatAvailable && viewMode === "gradcam" && (
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-card-inner)] p-3">
                  <label className="flex items-center gap-3 text-[11px] text-[var(--text-muted)]">
                    <span className="shrink-0">{t("sec3_xai_opacity")}</span>
                    <input type="range" min="0.1" max="1.0" step="0.05" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} className="w-full accent-emerald-500 h-1.5" />
                    <span className="font-mono font-bold text-[var(--text-heading)] w-9 text-right">{Math.round(opacity * 100)}%</span>
                  </label>
                  <label className="flex items-center justify-between gap-3 text-[11px] text-[var(--text-muted)]">
                    <span>{t("sec3_xai_colormap")}</span>
                    <select value={colormap} onChange={(e) => setColormap(e.target.value)} className="bg-[var(--bg-app)] text-[var(--text-heading)] text-[11px] font-semibold rounded-md px-2 py-1 border border-[var(--border-card)]">
                      <option value="Hot">Hot</option>
                      <option value="Jet">Jet</option>
                    </select>
                  </label>
                </div>
              )}
              <p className="text-[10px] leading-relaxed text-[var(--text-muted)]">{t("lung3d_note")}</p>
            </div>

            <div className="xl:col-span-5 flex flex-col gap-3.5 sm:gap-4">
              <div className="card p-3.5 sm:p-4">
                <div className="flex items-center gap-1.5 metric-label mb-3">
                  <Activity className="w-3.5 h-3.5 text-accent-500" />
                  <span>{t("sec3_results_header")}</span>
                </div>
                <div className="space-y-3">
                  {classLabels.map((lbl, idx) => {
                    const rawVal = diagnosis.probabilities[idx] ?? 0;
                    const pct = Math.min(100, Math.max(0, Math.round(rawVal > 1.0 ? rawVal : rawVal * 100)));
                    return (
                      <div key={idx}>
                        <div className="flex justify-between text-xs mb-1">
                          <span className="text-[var(--text-main)]">{lbl}</span>
                          <span className="font-mono font-bold text-[var(--text-heading)]">{pct}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-navy-800 overflow-hidden">
                          <div className={`h-full rounded-full ${CLASS_COLORS_BAR[idx]} transition-all duration-500`} style={{ width: `${pct}%` }} />
                        </div>
                      </div>
                    );
                  })}
                </div>
                <div className="mt-4 pt-3 border-t border-[var(--border-subtle)] text-[11px] text-[var(--text-muted)] leading-relaxed">
                  {diagnosis.findings}
                </div>
                <div className="mt-3 text-[11px] text-[var(--text-muted)] py-2 text-center rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)]">
                  {t("sec3_ground_truth")}:{" "}
                  <span className="text-[var(--text-heading)] font-semibold">
                    {diagnosis.true_class >= 0 && diagnosis.true_class < classLabels.length
                      ? classLabels[diagnosis.true_class]
                      : diagnosis.true_name}
                  </span>
                </div>
              </div>

              <div className="card p-3.5 sm:p-4">
                <div className="metric-label mb-3">{t("lung3d_2d_title")}</div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div className="text-[10px] font-semibold text-[var(--text-muted)] mb-1.5">{t("sec3_dual_pane_orig")}</div>
                    <div className="xray-frame aspect-square w-full flex items-center justify-center">
                      <canvas ref={rawRef} className="w-full h-full object-contain" />
                    </div>
                  </div>
                  <div>
                    <div className="text-[10px] font-semibold text-[var(--text-muted)] mb-1.5">{t("sec3_dual_pane_xai")}</div>
                    {heatAvailable ? (
                      <div className="xray-frame aspect-square w-full flex items-center justify-center">
                        <canvas ref={xaiRef} className="w-full h-full object-contain" />
                      </div>
                    ) : (
                      <div className="aspect-square w-full flex flex-col items-center justify-center p-3 text-center bg-slate-900/40 rounded-xl border border-dashed border-slate-700">
                        <div className="text-amber-400 text-[11px] font-semibold mb-1">{t("sec3_xai_unavailable")}</div>
                        <div className="text-[10px] text-[var(--text-muted)]">{diagnosis.heatmap_error || t("sec3_xai_unavailable_desc")}</div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : (
        <div className="card text-center py-10 sm:py-14 px-4">
          <h3 className="font-display text-sm sm:text-base font-bold text-[var(--text-heading)] mb-1">
            {t("sec3_ready_title")}
          </h3>
          <p className="text-xs text-[var(--text-muted)] max-w-md mx-auto mb-4 leading-relaxed">
            {t("sec3_ready_desc")}
          </p>
          <button
            onClick={() => onDiagnose(targetClass, opacity, colormap)}
            disabled={isLoading}
            className="btn-primary inline-flex items-center gap-2"
          >
            <Search className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
            <span>{t("sec3_btn_run_diag")}</span>
          </button>
        </div>
      )}
        </>
      )}
    </section>
  );
}
