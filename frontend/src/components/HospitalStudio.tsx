"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { HospitalCohort } from "@/types";
import { Database, PieChart as PieIcon, Eye, RefreshCw } from "lucide-react";

interface HospitalStudioProps {
  hospitals: HospitalCohort[];
  onGenerate: () => void;
  isLoading: boolean;
}

export default function HospitalStudio({ hospitals, onGenerate, isLoading }: HospitalStudioProps) {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState<number>(0);

  const activeHospital = hospitals[activeTab] || hospitals[0];

  const getLabelBadge = (labelIndex: number) => {
    switch (labelIndex) {
      case 0:
        return <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">{t("sec1_normal")}</span>;
      case 1:
        return <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">{t("sec1_pneumonia")}</span>;
      case 2:
        return <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/20">{t("sec1_covid")}</span>;
      default:
        return null;
    }
  };

  return (
    <section className="mb-12">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-brand-950 to-slate-900 text-white shadow-lg mb-6 border border-slate-800">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-brand-500/20 text-brand-400 border border-brand-500/30">
            <Database className="w-6 h-6" />
          </div>
          <div>
            <h2 className="font-display text-xl sm:text-2xl font-bold">{t("sec1_title")}</h2>
            <p className="text-xs sm:text-sm text-slate-400">{t("sec1_subtitle")}</p>
          </div>
        </div>

        <button
          onClick={onGenerate}
          disabled={isLoading}
          className="flex items-center justify-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-clinical-cyan hover:from-brand-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-md hover:shadow-brand-500/25 transition-all disabled:opacity-50"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          <span>{t("sec1_btn_generate")}</span>
        </button>
      </div>

      {hospitals.length > 0 && activeHospital ? (
        <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          {/* Hospital Select Tabs */}
          <div className="flex items-center gap-2 border-b border-slate-200 dark:border-slate-800 pb-4 mb-6 overflow-x-auto">
            {hospitals.map((h, idx) => (
              <button
                key={h.hospital_id}
                onClick={() => setActiveTab(idx)}
                className={`px-4 py-2 rounded-xl text-xs sm:text-sm font-semibold transition-all whitespace-nowrap ${
                  activeTab === idx
                    ? "bg-brand-50 dark:bg-brand-950/80 text-brand-600 dark:text-brand-400 border border-brand-200 dark:border-brand-800 shadow-sm"
                    : "text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-white"
                }`}
              >
                🏥 {t("sec1_hospital_prefix")} {h.hospital_id}
              </button>
            ))}
          </div>

          {/* Active Hospital Details */}
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
            
            {/* Left Column: Demographics & Donut SVG */}
            <div className="lg:col-span-5 flex flex-col gap-4">
              <div className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <h3 className="font-display font-bold text-base text-slate-900 dark:text-white mb-1">
                  {activeHospital.name}
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Node ID: <code className="text-brand-500">client_node_{activeHospital.hospital_id}</code> | Cohort: <strong>{activeHospital.num_samples} Scans</strong>
                </p>
              </div>

              {/* Prevalence Stats List */}
              <div className="space-y-2.5 p-4 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
                <div className="text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-2">
                  {t("sec1_stats_title")}
                </div>
                
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-emerald-500"></span>
                    <span>{t("sec1_normal")}</span>
                  </span>
                  <span className="font-bold text-slate-900 dark:text-white">
                    {activeHospital.counts.normal} ({Math.round(activeHospital.distribution[0] * 100)}%)
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
                    <span>{t("sec1_pneumonia")}</span>
                  </span>
                  <span className="font-bold text-slate-900 dark:text-white">
                    {activeHospital.counts.pneumonia} ({Math.round(activeHospital.distribution[1] * 100)}%)
                  </span>
                </div>

                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
                    <span>{t("sec1_covid")}</span>
                  </span>
                  <span className="font-bold text-slate-900 dark:text-white">
                    {activeHospital.counts.covid} ({Math.round(activeHospital.distribution[2] * 100)}%)
                  </span>
                </div>
              </div>
            </div>

            {/* Right Column: 3x3 Medical Scan Gallery */}
            <div className="lg:col-span-7">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2 text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider">
                  <Eye className="w-4 h-4 text-brand-500" />
                  <span>{t("sec1_sample_gallery")}</span>
                </div>
                <span className="text-[11px] text-slate-500">3x3 High-Density Matrix</span>
              </div>

              {/* 3x3 Image Grid */}
              <div className="grid grid-cols-3 gap-2.5 p-3 rounded-2xl bg-slate-950 border border-slate-800">
                {activeHospital.sample_images.map((imgData, i) => (
                  <div
                    key={i}
                    className="relative group aspect-square rounded-xl overflow-hidden bg-black border border-slate-800 flex items-center justify-center transition-all hover:scale-105 hover:border-brand-500/50 hover:shadow-glow z-0 hover:z-10"
                  >
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
                      className="w-full h-full object-cover"
                    />

                    {/* Pathology Badge Overlay */}
                    <div className="absolute bottom-1 left-1 right-1 opacity-0 group-hover:opacity-100 transition-opacity">
                      {getLabelBadge(activeHospital.sample_labels[i])}
                    </div>
                  </div>
                ))}
              </div>
            </div>

          </div>
        </div>
      ) : (
        <div className="text-center py-12 p-8 rounded-3xl bg-white dark:bg-slate-900 border border-dashed border-slate-300 dark:border-slate-800">
          <p className="text-sm text-slate-500 dark:text-slate-400">
            {t("sec1_subtitle")}. Click &quot;{t("sec1_btn_generate")}&quot; to begin.
          </p>
        </div>
      )}
    </section>
  );
}
