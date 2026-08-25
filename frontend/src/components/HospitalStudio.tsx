"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { HospitalCohort } from "@/types";
import { Database, RefreshCw, Eye, CheckCircle2, ShieldCheck, Sparkles } from "lucide-react";

interface HospitalStudioProps {
  hospitals: HospitalCohort[];
  onGenerate: () => void;
  isLoading: boolean;
}

const CLASS_COLORS = ["bg-accent-500", "bg-amber-500", "bg-red-500"];
const CLASS_DOT_COLORS = ["bg-accent-500", "bg-amber-500", "bg-red-500"];

export default function HospitalStudio({ hospitals, onGenerate, isLoading }: HospitalStudioProps) {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState(0);
  const active = hospitals[activeTab];

  return (
    <section id="section-ingestion" className="mb-5 sm:mb-6 scroll-mt-20">
      {/* Section Header */}
      <div className="section-bar mb-3.5 sm:mb-4 flex-col sm:flex-row items-stretch sm:items-center">
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-accent-500/10 flex items-center justify-center shrink-0">
            <Database className="w-4 h-4 sm:w-5 sm:h-5 text-accent-600 dark:text-accent-400" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">{t("sec1_title")}</h2>
              {hospitals.length > 0 && (
                <span className="hidden sm:inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                  <CheckCircle2 className="w-3 h-3" />
                  Ready
                </span>
              )}
            </div>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)]">{t("sec1_subtitle")}</p>
          </div>
        </div>
        <button onClick={onGenerate} disabled={isLoading} className="btn-primary w-full sm:w-auto mt-2.5 sm:mt-0">
          <RefreshCw className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isLoading ? "animate-spin" : ""}`} />
          <span>{hospitals.length > 0 ? t("sec1_btn_generate") : "Ingest Multi-Hospital Cohorts"}</span>
        </button>
      </div>

      {/* Ingestion Success Banner */}
      {hospitals.length > 0 && (
        <div className="card-inner flex items-center justify-between p-3 mb-3.5 border-l-4 border-l-emerald-500 text-[11px] sm:text-xs text-emerald-600 dark:text-emerald-400">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-4 h-4 text-emerald-500 shrink-0" />
            <span className="font-semibold">{t("sec1_ingestion_complete")}</span>
          </div>
          <span className="text-[10px] font-mono text-[var(--text-muted)] hidden md:inline">
            Non-IID Patient Skew Calibrated
          </span>
        </div>
      )}

      {/* Content Card */}
      <div className="card p-3.5 sm:p-5">
        {hospitals.length > 0 && active ? (
          <>
            {/* Hospital Tabs - Equal 4-Column Row (No scroll, No line break) */}
            <div className="grid grid-cols-4 gap-1 sm:gap-2 mb-4 sm:mb-5 w-full">
              {hospitals.map((h, idx) => (
                <button
                  key={h.hospital_id}
                  onClick={() => setActiveTab(idx)}
                  className={`py-1.5 px-1 sm:px-3 rounded-lg text-center font-semibold transition-all truncate text-[10px] sm:text-xs ${
                    activeTab === idx
                      ? "bg-accent-500/15 text-accent-600 dark:text-accent-400 border border-accent-500/30 shadow-sm"
                      : "text-slate-500 hover:text-[var(--text-heading)] border border-transparent hover:bg-black/5 dark:hover:bg-white/5"
                  }`}
                >
                  <span className="hidden sm:inline">{t("sec1_hospital_prefix")} </span>
                  <span className="inline sm:hidden">Hosp. </span>
                  <span>{h.hospital_id}</span>
                </button>
              ))}
            </div>

            {/* Equal 2-Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 items-start">
              {/* Left: Stats & Demographics */}
              <div className="space-y-4">
                <div>
                  <div className="text-sm font-bold text-[var(--text-heading)] mb-0.5 truncate">{active.name}</div>
                  <div className="text-[11px] text-[var(--text-muted)]">
                    Node <code className="text-accent-600 dark:text-accent-400 font-bold">client_{active.hospital_id}</code> | {active.num_samples} Scans
                  </div>
                </div>

                <div className="pt-3 border-t border-[var(--border-subtle)]">
                  <div className="metric-label mb-3">{t("sec1_stats_title")}</div>
                  <div className="space-y-2.5">
                    {[
                      { key: "normal", label: t("sec1_normal"), count: active.counts.normal, dist: active.distribution[0], idx: 0 },
                      { key: "pneumonia", label: t("sec1_pneumonia"), count: active.counts.pneumonia, dist: active.distribution[1], idx: 1 },
                      { key: "covid", label: t("sec1_covid"), count: active.counts.covid, dist: active.distribution[2], idx: 2 },
                    ].map((item) => (
                      <div key={item.key} className="flex items-center justify-between py-1.5 border-b border-[var(--border-subtle)] last:border-0 gap-2">
                        <div className="flex items-center gap-2 text-[11px] sm:text-xs text-[var(--text-main)] min-w-0">
                          <span className={`w-2 h-2 rounded-full shrink-0 ${CLASS_DOT_COLORS[item.idx]}`} />
                          <span className="truncate">{item.label}</span>
                        </div>
                        <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
                          <div className="w-16 sm:w-24 h-1.5 rounded-full bg-slate-200 dark:bg-navy-800 overflow-hidden">
                            <div
                              className={`h-full rounded-full ${CLASS_COLORS[item.idx]}`}
                              style={{ width: `${Math.round(item.dist * 100)}%` }}
                            />
                          </div>
                          <span className="text-[11px] sm:text-xs font-mono font-bold text-[var(--text-heading)] w-10 sm:w-12 text-right">
                            {item.count}
                          </span>
                          <span className="text-[9px] sm:text-[10px] text-[var(--text-muted)] w-7 sm:w-8 text-right">
                            {Math.round(item.dist * 100)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </div>

              {/* Right: 3x3 Scan Gallery */}
              <div>
                <div className="flex items-center justify-between mb-2.5 sm:mb-3">
                  <div className="flex items-center gap-1.5 metric-label">
                    <Eye className="w-3.5 h-3.5 text-accent-500" />
                    <span>{t("sec1_sample_gallery")}</span>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2">
                  {active.sample_images.map((imgData, i) => (
                    <div key={i} className="xray-frame group hover:ring-2 hover:ring-accent-500/40 transition-all aspect-square">
                      <canvas
                        ref={(canvas) => {
                          if (!canvas) return;
                          const ctx = canvas.getContext("2d");
                          if (!ctx) return;
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
                        }}
                      />
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </>
        ) : (
          <div className="text-center py-10 sm:py-14 px-4">
            <div className="w-12 h-12 rounded-2xl bg-accent-500/10 border border-accent-500/20 flex items-center justify-center mx-auto mb-3 text-accent-500">
              <Sparkles className="w-6 h-6 animate-pulse" />
            </div>
            <h3 className="font-display text-sm sm:text-base font-bold text-[var(--text-heading)] mb-1">
              {t("sec1_empty_state_title")}
            </h3>
            <p className="text-xs text-[var(--text-muted)] max-w-md mx-auto mb-4 leading-relaxed">
              {t("sec1_empty_state_desc")}
            </p>
            <button
              onClick={onGenerate}
              disabled={isLoading}
              className="btn-primary inline-flex items-center gap-2"
            >
              <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
              <span>{t("sec1_btn_generate")}</span>
            </button>
          </div>
        )}
      </div>
    </section>
  );
}
