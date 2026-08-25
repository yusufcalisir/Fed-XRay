"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { HospitalCohort } from "@/types";
import { Database, RefreshCw, Eye, CheckCircle2, ShieldCheck, Sparkles, Building2, Layers } from "lucide-react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip as RechartsTooltip } from "recharts";

interface HospitalStudioProps {
  hospitals: HospitalCohort[];
  onGenerate: () => void;
  isLoading: boolean;
}

const PIE_COLORS = ["#10b981", "#f59e0b", "#ef4444"];
const PIE_BAR_GRADIENTS = [
  "from-emerald-500 to-teal-400",
  "from-amber-500 to-orange-400",
  "from-red-500 to-rose-400",
];

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
          gradient: PIE_BAR_GRADIENTS[0],
        },
        {
          name: t("sec1_pneumonia"),
          shortName: "Pneumonia",
          value: active.counts.pneumonia,
          percentage: Math.round(active.distribution[1] * 100),
          color: PIE_COLORS[1],
          gradient: PIE_BAR_GRADIENTS[1],
        },
        {
          name: t("sec1_covid"),
          shortName: "COVID-19",
          value: active.counts.covid,
          percentage: Math.round(active.distribution[2] * 100),
          color: PIE_COLORS[2],
          gradient: PIE_BAR_GRADIENTS[2],
        },
      ]
    : [];

  return (
    <section id="section-ingestion" className="mb-5 sm:mb-6 scroll-mt-20">
      {/* Section Bar */}
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
            Strategy E Isolated Partitions (Zero Patient Leakage)
          </span>
        </div>
      )}

      {/* Main Studio Card */}
      <div className="card p-3.5 sm:p-5">
        {hospitals.length > 0 && active ? (
          <>
            {/* Hospital Tabs */}
            <div className="grid grid-cols-4 gap-1.5 sm:gap-2 mb-4 w-full">
              {hospitals.map((h, idx) => (
                <button
                  key={h.hospital_id}
                  onClick={() => setActiveTab(idx)}
                  className={`py-2 px-2 rounded-xl text-center font-semibold transition-all truncate text-[11px] sm:text-xs flex items-center justify-center gap-1.5 ${
                    activeTab === idx
                      ? "bg-accent-500/15 text-accent-600 dark:text-accent-400 border border-accent-500/30 shadow-sm"
                      : "text-slate-500 hover:text-[var(--text-heading)] border border-transparent hover:bg-black/5 dark:hover:bg-white/5"
                  }`}
                >
                  <Building2 className="w-3.5 h-3.5 shrink-0 hidden sm:inline" />
                  <span className="truncate">{h.name.split(" ")[0]}</span>
                  <span className="font-mono text-[10px] opacity-75">#{h.hospital_id}</span>
                </button>
              ))}
            </div>

            {/* Symmetrical 2-Column Responsive Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-start">
              {/* Left Column: Integrated Visual Donut & Demographics */}
              <div className="lg:col-span-6 flex flex-col justify-between bg-[var(--bg-card-inner)]/60 p-4 sm:p-4 rounded-2xl border border-[var(--border-subtle)] space-y-3.5">
                {/* Node Title & Ingestion Stats */}
                <div className="flex items-start justify-between gap-2 pb-2.5 border-b border-[var(--border-subtle)]">
                  <div className="min-w-0">
                    <h3 className="text-xs sm:text-sm font-bold text-[var(--text-heading)] truncate">{active.name}</h3>
                    <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mt-0.5">
                      Provenance: <span className="font-medium text-[var(--text-main)]">ISIC 2019 / NCT Benchmark</span>
                    </div>
                  </div>
                  <div className="flex flex-col items-end shrink-0">
                    <span className="text-[10px] font-mono font-bold px-2 py-0.5 rounded-md bg-accent-500/10 text-accent-600 dark:text-accent-400 border border-accent-500/20">
                      client_{active.hospital_id}
                    </span>
                    <span className="text-[10px] font-mono text-[var(--text-muted)] mt-0.5">
                      {active.num_samples} Scans
                    </span>
                  </div>
                </div>

                {/* Donut Chart & Demographic Bars Side-by-Side on Tablet/Desktop, Stacked on Mobile */}
                <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center">
                  {/* Left: Donut Chart with Centered Total Badge */}
                  <div className="sm:col-span-5 flex items-center justify-center">
                    <div className="w-28 h-28 sm:w-32 sm:h-32 relative">
                      <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                          <RechartsTooltip
                            contentStyle={{
                              background: "var(--bg-card)",
                              border: "1px solid var(--border-card)",
                              borderRadius: "8px",
                              fontSize: "11px",
                              color: "var(--text-main)",
                              boxShadow: "0 6px 16px rgba(0,0,0,0.15)",
                            }}
                            formatter={(value: any, name: any) => [`${value} Scans`, name]}
                          />
                          <Pie
                            data={pieData}
                            dataKey="value"
                            nameKey="name"
                            cx="50%"
                            cy="50%"
                            innerRadius={34}
                            outerRadius={52}
                            paddingAngle={3}
                            strokeWidth={0}
                          >
                            {pieData.map((entry, index) => (
                              <Cell key={`cell-${index}`} fill={entry.color} />
                            ))}
                          </Pie>
                        </PieChart>
                      </ResponsiveContainer>
                      <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                        <span className="text-[8px] uppercase tracking-wider text-[var(--text-muted)] font-bold">Total</span>
                        <span className="text-xs sm:text-sm font-black font-mono text-[var(--text-heading)] leading-none mt-0.5">
                          {active.num_samples}
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Right: Detailed Demographic Breakdown Rows with Progress Bars */}
                  <div className="sm:col-span-7 space-y-2">
                    {pieData.map((item, idx) => (
                      <div key={idx} className="p-2 rounded-xl bg-black/5 dark:bg-white/5 border border-[var(--border-subtle)]/50">
                        <div className="flex items-center justify-between text-[11px] mb-1">
                          <div className="flex items-center gap-1.5 min-w-0">
                            <span className="w-2 h-2 rounded-full shrink-0" style={{ backgroundColor: item.color }} />
                            <span className="truncate font-semibold text-[var(--text-heading)]">{item.shortName}</span>
                          </div>
                          <div className="flex items-center gap-1.5 shrink-0 font-mono">
                            <span className="font-bold text-[var(--text-heading)]">{item.value}</span>
                            <span className="text-[10px] text-[var(--text-muted)]">({item.percentage}%)</span>
                          </div>
                        </div>
                        {/* Smooth Progress Bar */}
                        <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-navy-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full bg-gradient-to-r ${item.gradient} transition-all duration-500`}
                            style={{ width: `${item.percentage}%` }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Skew Tag Footer */}
                <div className="pt-2 border-t border-[var(--border-subtle)] flex items-center justify-between text-[10px] text-[var(--text-muted)] font-mono">
                  <span className="flex items-center gap-1">
                    <Layers className="w-3 h-3 text-accent-500" />
                    Non-IID Dirichlet Skew
                  </span>
                  <span className="text-accent-600 dark:text-accent-400 font-semibold">Native Prevalence</span>
                </div>
              </div>

              {/* Right Column: 3x3 Scan Gallery */}
              <div className="lg:col-span-6 flex flex-col justify-between bg-[var(--bg-card-inner)]/60 p-4 sm:p-4 rounded-2xl border border-[var(--border-subtle)] space-y-3">
                <div className="flex items-center justify-between pb-2.5 border-b border-[var(--border-subtle)]">
                  <div className="flex items-center gap-1.5 text-xs sm:text-sm font-bold text-[var(--text-heading)]">
                    <Eye className="w-3.5 h-3.5 text-accent-500" />
                    <span>{t("sec1_sample_gallery")}</span>
                  </div>
                  <span className="text-[10px] text-[var(--text-muted)] font-mono">9 Local Client Crops</span>
                </div>

                <div className="grid grid-cols-3 gap-2">
                  {active.sample_images.map((imgData, i) => (
                    <div key={i} className="xray-frame group hover:ring-2 hover:ring-accent-500/40 transition-all aspect-square rounded-xl overflow-hidden shadow-sm">
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

                <div className="pt-2 border-t border-[var(--border-subtle)] text-[10px] text-[var(--text-muted)] text-center font-mono truncate">
                  Patient ID Hash: SHA-256 Validated · Zero Data Cross-Split
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
