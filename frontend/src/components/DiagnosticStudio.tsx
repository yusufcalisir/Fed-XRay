"use client";

import React, { useState, useEffect, useRef } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { DiagnosisResult } from "@/types";
import { Stethoscope, Search, Activity } from "lucide-react";

interface DiagnosticStudioProps {
  diagnosis: DiagnosisResult | null;
  onDiagnose: (classIndex?: number, opacity?: number, colormap?: string) => void;
  isLoading: boolean;
}

const CLASS_COLORS_BAR = ["bg-accent-500", "bg-amber-500", "bg-red-500"];

export default function DiagnosticStudio({ diagnosis, onDiagnose, isLoading }: DiagnosticStudioProps) {
  const { t } = useLanguage();
  const [opacity, setOpacity] = useState(0.55);
  const [colormap, setColormap] = useState("Hot");
  const rawRef = useRef<HTMLCanvasElement | null>(null);
  const xaiRef = useRef<HTMLCanvasElement | null>(null);

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
      const v = Math.floor(d[r][c] * 255);
      const i = (r * s + c) * 4;
      img.data[i] = v; img.data[i+1] = v; img.data[i+2] = v; img.data[i+3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [diagnosis]);

  // Render Grad-CAM overlay
  useEffect(() => {
    if (!diagnosis || !xaiRef.current) return;
    const canvas = xaiRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const raw = diagnosis.raw_image;
    const heat = diagnosis.heatmap;
    const s = raw.length;
    canvas.width = s; canvas.height = s;
    const img = ctx.createImageData(s, s);
    for (let r = 0; r < s; r++) for (let c = 0; c < s; c++) {
      const rv = raw[r][c] * 255;
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

  return (
    <section className="mb-5 sm:mb-6">
      {/* Section Header */}
      <div className="section-bar mb-3.5 sm:mb-4 flex-col sm:flex-row items-stretch sm:items-center">
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-purple-500/10 flex items-center justify-center shrink-0">
            <Stethoscope className="w-4 h-4 sm:w-5 sm:h-5 text-purple-500 dark:text-purple-400" />
          </div>
          <div>
            <h2 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">{t("sec3_title")}</h2>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)]">{t("sec3_subtitle")}</p>
          </div>
        </div>
        <button onClick={() => onDiagnose(undefined, opacity, colormap)} disabled={isLoading} className="btn-primary w-full sm:w-auto mt-2.5 sm:mt-0">
          <Search className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isLoading ? "animate-spin" : ""}`} />
          <span>{t("sec3_btn_run_diag")}</span>
        </button>
      </div>

      {diagnosis ? (
        <div className="space-y-3.5 sm:space-y-4">

          {/* Diagnosis Banner */}
          <div className={`card p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-l-4 ${
            diagnosis.predicted_class === 0 ? "border-l-accent-500" :
            diagnosis.predicted_class === 1 ? "border-l-amber-500" : "border-l-red-500"
          }`}>
            <div>
              <div className="metric-label mb-0.5 sm:mb-1">{t("sec3_diagnosis")}</div>
              <div className="font-display text-lg sm:text-xl font-extrabold text-[var(--text-heading)]">
                {classLabels[diagnosis.predicted_class]}
              </div>
            </div>
            <div className="text-left sm:text-right">
              <div className="metric-label mb-0.5 sm:mb-1">{t("sec3_confidence")}</div>
              <div className="font-display text-lg sm:text-xl font-black text-accent-600 dark:text-accent-400">{diagnosis.confidence}%</div>
            </div>
          </div>

          {/* Equal 3-Column Diagnostic Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3.5 sm:gap-4">

            {/* Original X-Ray */}
            <div className="card p-3.5 sm:p-4 flex flex-col">
              <div className="metric-label mb-2">{t("sec3_dual_pane_orig")}</div>
              <div className="xray-frame flex-1 flex items-center justify-center max-w-[280px] mx-auto w-full">
                <canvas ref={rawRef} />
              </div>
              <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-2 text-center">
                {t("sec3_ground_truth")}: <span className="text-[var(--text-heading)] font-semibold">{classLabels[diagnosis.true_class]}</span>
              </div>
            </div>

            {/* Grad-CAM Overlay */}
            <div className="card p-3.5 sm:p-4 flex flex-col">
              <div className="metric-label mb-2">{t("sec3_dual_pane_xai")}</div>
              <div className="xray-frame flex-1 flex items-center justify-center max-w-[280px] mx-auto w-full">
                <canvas ref={xaiRef} />
              </div>
              <div className="mt-2.5 space-y-2">
                <div className="flex items-center justify-between text-[10px] sm:text-[11px]">
                  <span className="text-[var(--text-muted)]">{t("sec3_xai_opacity")}</span>
                  <span className="font-mono text-[var(--text-heading)] font-bold">{Math.round(opacity * 100)}%</span>
                </div>
                <input type="range" min="0.1" max="1.0" step="0.05" value={opacity} onChange={(e) => setOpacity(parseFloat(e.target.value))} className="w-full accent-accent-500 h-1.5" />
                <div className="flex items-center justify-between text-[10px] sm:text-[11px]">
                  <span className="text-[var(--text-muted)]">{t("sec3_xai_colormap")}</span>
                  <select value={colormap} onChange={(e) => setColormap(e.target.value)} className="bg-[var(--bg-card-inner)] text-[var(--text-heading)] text-[10px] sm:text-[11px] font-semibold rounded-md px-2 py-1 border border-[var(--border-card)]">
                    <option value="Hot">Hot</option>
                    <option value="Jet">Jet</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Probabilities */}
            <div className="card p-3.5 sm:p-4 flex flex-col">
              <div className="flex items-center gap-1.5 metric-label mb-2.5 sm:mb-3">
                <Activity className="w-3.5 h-3.5 text-accent-500" />
                <span>{t("sec3_results_header")}</span>
              </div>
              <div className="space-y-2.5 sm:space-y-3 flex-1">
                {classLabels.map((lbl, idx) => {
                  const pct = Math.round((diagnosis.probabilities[idx] || 0) * 100);
                  return (
                    <div key={idx}>
                      <div className="flex justify-between text-[11px] sm:text-xs mb-1">
                        <span className="text-[var(--text-main)]">{lbl}</span>
                        <span className="font-mono font-bold text-[var(--text-heading)]">{pct}%</span>
                      </div>
                      <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-navy-800 overflow-hidden">
                        <div className={`h-full rounded-full ${CLASS_COLORS_BAR[idx]} transition-all duration-300`} style={{ width: `${pct}%` }} />
                      </div>
                    </div>
                  );
                })}
              </div>
              <div className="card-inner p-2.5 sm:p-3 mt-3 text-[10px] sm:text-[11px] text-[var(--text-muted)] leading-relaxed">
                {diagnosis.findings}
              </div>
            </div>

          </div>
        </div>
      ) : (
        <div className="card text-center py-12 sm:py-16 text-xs sm:text-sm text-[var(--text-muted)]">
          {t("sec3_subtitle")}
        </div>
      )}
    </section>
  );
}
