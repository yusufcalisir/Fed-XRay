"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { HospitalCohort } from "@/types";
import { Database, RefreshCw, Eye, CheckCircle2, ShieldCheck, Sparkles } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts";

interface HospitalStudioProps {
  hospitals: HospitalCohort[];
  onGenerate: () => void;
  isLoading: boolean;
}

const PIE_COLORS = ["#10b981", "#f59e0b", "#ef4444"];
const PIE_BG_COLORS = ["bg-emerald-500/10", "bg-amber-500/10", "bg-red-500/10"];
const PIE_BORDER_COLORS = ["border-emerald-500/30", "border-amber-500/30", "border-red-500/30"];
const PIE_TEXT_COLORS = ["text-emerald-600 dark:text-emerald-400", "text-amber-600 dark:text-amber-400", "text-red-600 dark:text-red-400"];

export default function HospitalStudio({ hospitals, onGenerate, isLoading }: HospitalStudioProps) {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState(0);
  const active = hospitals[activeTab];

  const pieData = active
    ? [
        {
          name: t("sec1_normal"),
          shortName: "Normal",
          value: active.counts.normal,
          percentage: Math.round(active.distribution[0] * 100),
          color: PIE_COLORS[0],
          bg: PIE_BG_COLORS[0],
          border: PIE_BORDER_COLORS[0],
          text: PIE_TEXT_COLORS[0],
        },
        {
          name: t("sec1_pneumonia"),
          shortName: "Pneumonia",
          value: active.counts.pneumonia,
          percentage: Math.round(active.distribution[1] * 100),
          color: PIE_COLORS[1],
          bg: PIE_BG_COLORS[1],
          border: PIE_BORDER_COLORS[1],
          text: PIE_TEXT_COLORS[1],
        },
        {
          name: t("sec1_covid"),
          shortName: "COVID-19",
          value: active.counts.covid,
          percentage: Math.round(active.distribution[2] * 100),
          color: PIE_COLORS[2],
          bg: PIE_BG_COLORS[2],
          border: PIE_BORDER_COLORS[2],
          text: PIE_TEXT_COLORS[2],
        },
      ]
    : [];

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
            Non-IID Patient Skew Calibrated (Strategy E Standard)
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
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-5 items-stretch">
              {/* Left Column: Modern Integrated Distribution Studio */}
              <div className="flex flex-col justify-between bg-[var(--bg-card-inner)]/50 p-4 sm:p-5 rounded-2xl border border-[var(--border-subtle)]">
                {/* Header info */}
                <div className="flex items-center justify-between">
                  <div className="min-w-0">
                    <h3 className="text-sm sm:text-base font-bold text-[var(--text-heading)] truncate">{active.name}</h3>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5">
                      Cohort Ingestion: <span className="font-semibold text-[var(--text-main)]">{active.num_samples} Scans</span>
                    </div>
                  </div>
                  <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-accent-500/10 text-accent-600 dark:text-accent-400 border border-accent-500/20 shrink-0">
                    client_{active.hospital_id}
                  </span>
                </div>

                {/* Prominent Donut Pie Chart */}
                <div className="my-3 sm:my-4 flex items-center justify-center relative">
                  <div className="w-40 h-40 sm:w-48 sm:h-48 relative">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <RechartsTooltip
                          contentStyle={{
                            background: "var(--bg-card)",
                            border: "1px solid var(--border-card)",
                            borderRadius: "10px",
                            fontSize: "11px",
                            color: "var(--text-main)",
                            boxShadow: "0 8px 24px rgba(0,0,0,0.2)",
                          }}
                          formatter={(value: any, name: any) => [`${value} Scans`, name]}
                        />
                        <Pie
                          data={pieData}
                          dataKey="value"
                          nameKey="name"
                          cx="50%"
                          cy="50%"
                          innerRadius={48}
                          outerRadius={74}
                          paddingAngle={4}
                          strokeWidth={0}
                        >
                          {pieData.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                      </PieChart>
                    </ResponsiveContainer>
                    {/* Centered Total Indicator */}
                    <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                      <span className="text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-bold">Total</span>
                      <span className="text-base sm:text-lg font-black font-mono text-[var(--text-heading)] leading-none mt-0.5">
                        {active.num_samples}
                      </span>
                      <span className="text-[9px] text-[var(--text-muted)] font-mono mt-0.5">scans</span>
                    </div>
                  </div>
                </div>

                {/* 3-Column Distribution Cards underneath the Donut Chart */}
                <div className="grid grid-cols-3 gap-2 sm:gap-2.5">
                  {pieData.map((d, i) => (
                    <div
                      key={i}
                      className={`p-2 sm:p-2.5 rounded-xl border ${d.border} ${d.bg} flex flex-col justify-between text-center transition-all hover:scale-[1.02]`}
                    >
                      <div className="flex items-center justify-center gap-1.5 mb-1">
                        <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: d.color }} />
                        <span className="text-[10px] sm:text-[11px] font-semibold text-[var(--text-main)] truncate">
                          {d.shortName}
                        </span>
                      </div>
                      <div className="flex items-baseline justify-center gap-1">
                        <span className="text-xs sm:text-sm font-black font-mono text-[var(--text-heading)]">
                          {d.value}
                        </span>
                        <span className={`text-[10px] font-bold font-mono ${d.text}`}>
                          ({d.percentage}%)
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right Column: 3x3 Scan Gallery */}
              <div className="flex flex-col justify-between bg-[var(--bg-card-inner)]/50 p-4 sm:p-5 rounded-2xl border border-[var(--border-subtle)]">
                <div>
                  <div className="flex items-center justify-between mb-3">
                    <div className="flex items-center gap-1.5 metric-label">
                      <Eye className="w-3.5 h-3.5 text-accent-500" />
                      <span>{t("sec1_sample_gallery")}</span>
                    </div>
                    <span className="text-[10px] text-[var(--text-muted)] font-mono">9 Multi-Class Crops</span>
                  </div>

                  <div className="grid grid-cols-3 gap-2 sm:gap-2.5">
                    {active.sample_images.map((imgData, i) => (
                      <div key={i} className="xray-frame group hover:ring-2 hover:ring-accent-500/40 transition-all aspect-square rounded-xl overflow-hidden">
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

                <div className="mt-3 pt-2.5 border-t border-[var(--border-subtle)] text-[10px] text-[var(--text-muted)] text-center font-mono">
                  Zero Patient Leakage Enforced (Strategy E Protocol)
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
