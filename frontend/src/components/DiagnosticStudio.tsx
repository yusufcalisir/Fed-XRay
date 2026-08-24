"use client";

import React, { useState, useEffect, useRef } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { DiagnosisResult } from "@/types";
import { Stethoscope, Search, Sliders, Activity, CheckCircle2, AlertTriangle } from "lucide-react";

interface DiagnosticStudioProps {
  diagnosis: DiagnosisResult | null;
  onDiagnose: (classIndex?: number, opacity?: number, colormap?: string) => void;
  isLoading: boolean;
}

export default function DiagnosticStudio({ diagnosis, onDiagnose, isLoading }: DiagnosticStudioProps) {
  const { t } = useLanguage();
  const [opacity, setOpacity] = useState<number>(0.55);
  const [colormap, setColormap] = useState<string>("Hot");
  const rawCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const xaiCanvasRef = useRef<HTMLCanvasElement | null>(null);

  // Render raw X-Ray canvas
  useEffect(() => {
    if (!diagnosis || !rawCanvasRef.current) return;
    const canvas = rawCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const imgData = diagnosis.raw_image;
    const size = imgData.length;
    canvas.width = size;
    canvas.height = size;
    const img = ctx.createImageData(size, size);

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const val = Math.floor(imgData[r][c] * 255);
        const idx = (r * size + c) * 4;
        img.data[idx] = val;
        img.data[idx + 1] = val;
        img.data[idx + 2] = val;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [diagnosis]);

  // Render Grad-CAM Heatmap Blend Canvas
  useEffect(() => {
    if (!diagnosis || !xaiCanvasRef.current) return;
    const canvas = xaiCanvasRef.current;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const raw = diagnosis.raw_image;
    const heat = diagnosis.heatmap;
    const size = raw.length;
    canvas.width = size;
    canvas.height = size;
    const img = ctx.createImageData(size, size);

    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const rawVal = raw[r][c] * 255;
        const heatVal = heat[r][c]; // 0 to 1

        // Simple JET / HOT heatmap color map
        let hr = 0, hg = 0, hb = 0;
        if (colormap === "Hot") {
          hr = Math.min(255, heatVal * 3 * 255);
          hg = Math.max(0, Math.min(255, (heatVal - 0.33) * 3 * 255));
          hb = Math.max(0, Math.min(255, (heatVal - 0.66) * 3 * 255));
        } else {
          // Jet Colormap
          hr = Math.min(255, Math.max(0, 1.5 - Math.abs(heatVal * 4 - 3)) * 255);
          hg = Math.min(255, Math.max(0, 1.5 - Math.abs(heatVal * 4 - 2)) * 255);
          hb = Math.min(255, Math.max(0, 1.5 - Math.abs(heatVal * 4 - 1)) * 255);
        }

        const alpha = opacity;
        const finalR = Math.floor((1 - alpha) * rawVal + alpha * hr);
        const finalG = Math.floor((1 - alpha) * rawVal + alpha * hg);
        const finalB = Math.floor((1 - alpha) * rawVal + alpha * hb);

        const idx = (r * size + c) * 4;
        img.data[idx] = finalR;
        img.data[idx + 1] = finalG;
        img.data[idx + 2] = finalB;
        img.data[idx + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [diagnosis, opacity, colormap]);

  const classLabels = [t("sec1_normal"), t("sec1_pneumonia"), t("sec1_covid")];
  const classColors = ["bg-emerald-500", "bg-amber-500", "bg-rose-500"];

  return (
    <section className="mb-12">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-brand-950 to-slate-900 text-white shadow-lg mb-6 border border-slate-800">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-brand-500/20 text-brand-400 border border-brand-500/30">
            <Stethoscope className="w-6 h-6" />
          </div>
          <div>
            <h2 className="font-display text-xl sm:text-2xl font-bold">{t("sec3_title")}</h2>
            <p className="text-xs sm:text-sm text-slate-400">{t("sec3_subtitle")}</p>
          </div>
        </div>

        <button
          onClick={() => onDiagnose(undefined, opacity, colormap)}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-clinical-cyan hover:from-brand-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-md hover:shadow-brand-500/25 transition-all disabled:opacity-50"
        >
          <Search className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          <span>{t("sec3_btn_run_diag")}</span>
        </button>
      </div>

      {diagnosis ? (
        <div className="space-y-6">
          
          {/* Diagnostic Outcome Banner */}
          <div className={`p-6 rounded-3xl border flex flex-col sm:flex-row sm:items-center justify-between gap-4 ${
            diagnosis.predicted_class === 0
              ? "bg-emerald-500/10 border-emerald-500/30 text-emerald-800 dark:text-emerald-300"
              : diagnosis.predicted_class === 1
              ? "bg-amber-500/10 border-amber-500/30 text-amber-800 dark:text-amber-300"
              : "bg-rose-500/10 border-rose-500/30 text-rose-800 dark:text-rose-300"
          }`}>
            <div>
              <span className="text-xs uppercase font-bold tracking-wider opacity-80">{t("sec3_diagnosis")}</span>
              <div className="font-display text-2xl sm:text-3xl font-extrabold flex items-center gap-2 mt-1">
                <span>{diagnosis.predicted_class === 0 ? "🟢" : diagnosis.predicted_class === 1 ? "🟠" : "🔴"}</span>
                <span>{classLabels[diagnosis.predicted_class]}</span>
              </div>
            </div>

            <div className="sm:text-right">
              <span className="text-xs uppercase font-bold tracking-wider opacity-80">{t("sec3_confidence")}</span>
              <div className="font-display text-2xl sm:text-3xl font-black">{diagnosis.confidence}%</div>
            </div>
          </div>

          {/* Dual-Pane Radiological Inspector Grid */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
            
            {/* Left Pane: Original Grayscale X-Ray */}
            <div className="lg:col-span-4 p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col items-center">
              <div className="w-full flex items-center justify-between mb-3 text-xs font-bold text-slate-700 dark:text-slate-300">
                <span>📷 {t("sec3_dual_pane_orig")}</span>
                <span className="text-slate-500 font-mono">28x28 DICOM</span>
              </div>

              <div className="w-full aspect-square max-w-[280px] rounded-2xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center shadow-inner">
                <canvas ref={rawCanvasRef} className="w-full h-full object-cover" />
              </div>

              <div className="mt-3 text-xs text-slate-500">
                {t("sec3_ground_truth")}: <strong className="text-slate-900 dark:text-white">{classLabels[diagnosis.true_class]}</strong>
              </div>
            </div>

            {/* Middle Pane: Grad-CAM Attention Saliency Overlay */}
            <div className="lg:col-span-4 p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col items-center">
              <div className="w-full flex items-center justify-between mb-3 text-xs font-bold text-slate-700 dark:text-slate-300">
                <span>🔬 {t("sec3_dual_pane_xai")}</span>
                <span className="text-brand-500 font-mono">Grad-CAM</span>
              </div>

              <div className="w-full aspect-square max-w-[280px] rounded-2xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center shadow-inner">
                <canvas ref={xaiCanvasRef} className="w-full h-full object-cover" />
              </div>

              {/* Opacity & Colormap Controls */}
              <div className="w-full mt-4 space-y-2 text-xs">
                <div className="flex items-center justify-between">
                  <span className="text-slate-500 font-medium">{t("sec3_xai_opacity")}:</span>
                  <span className="font-bold text-slate-900 dark:text-white">{Math.round(opacity * 100)}%</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={opacity}
                  onChange={(e) => setOpacity(parseFloat(e.target.value))}
                  className="w-full accent-brand-500"
                />

                <div className="flex items-center justify-between pt-1">
                  <span className="text-slate-500 font-medium">{t("sec3_xai_colormap")}:</span>
                  <select
                    value={colormap}
                    onChange={(e) => setColormap(e.target.value)}
                    className="p-1 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-800 dark:text-slate-200 text-xs border border-slate-300 dark:border-slate-700"
                  >
                    <option value="Hot">Hot Spectrum</option>
                    <option value="Jet">Jet Rainbow</option>
                  </select>
                </div>
              </div>
            </div>

            {/* Right Pane: Class Probability Breakdown */}
            <div className="lg:col-span-4 p-5 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between">
              <div>
                <div className="flex items-center gap-2 text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-4">
                  <Activity className="w-4 h-4 text-brand-500" />
                  <span>{t("sec3_results_header")}</span>
                </div>

                <div className="space-y-3">
                  {classLabels.map((lbl, idx) => {
                    const prob = diagnosis.probabilities[idx] || 0;
                    const pct = Math.round(prob * 100);
                    return (
                      <div key={idx} className="space-y-1">
                        <div className="flex justify-between text-xs font-semibold">
                          <span className="text-slate-700 dark:text-slate-300">{lbl}</span>
                          <span className="text-slate-900 dark:text-white font-mono">{pct}%</span>
                        </div>
                        <div className="w-full h-2 rounded-full bg-slate-100 dark:bg-slate-800 overflow-hidden">
                          <div
                            className={`h-full ${classColors[idx]} transition-all duration-300`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>

              {/* Clinical Morphological Findings */}
              <div className="mt-6 p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs text-slate-600 dark:text-slate-400">
                {diagnosis.findings}
              </div>
            </div>

          </div>

        </div>
      ) : (
        <div className="text-center py-12 p-8 rounded-3xl bg-white dark:bg-slate-900 border border-dashed border-slate-300 dark:border-slate-800">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {t("sec3_subtitle")}. Click &quot;{t("sec3_btn_run_diag")}&quot; to examine a patient study.
          </p>
        </div>
      )}
    </section>
  );
}
