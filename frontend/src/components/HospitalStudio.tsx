"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { HospitalCohort } from "@/types";
import { Database, RefreshCw, Eye } from "lucide-react";

interface HospitalStudioProps {
  hospitals: HospitalCohort[];
  onGenerate: () => void;
  isLoading: boolean;
}

const CLASS_COLORS = ["bg-accent-400", "bg-amber-400", "bg-red-400"];
const CLASS_DOT_COLORS = ["bg-accent-500", "bg-amber-500", "bg-red-500"];

export default function HospitalStudio({ hospitals, onGenerate, isLoading }: HospitalStudioProps) {
  const { t } = useLanguage();
  const [activeTab, setActiveTab] = useState(0);
  const active = hospitals[activeTab];

  return (
    <section className="mb-6">
      {/* Section Header */}
      <div className="section-bar mb-4 flex-col sm:flex-row">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-accent-500/10 flex items-center justify-center shrink-0">
            <Database className="w-5 h-5 text-accent-400" />
          </div>
          <div>
            <h2 className="font-display text-lg font-bold text-white">{t("sec1_title")}</h2>
            <p className="text-xs text-slate-500">{t("sec1_subtitle")}</p>
          </div>
        </div>
        <button onClick={onGenerate} disabled={isLoading} className="btn-primary mt-3 sm:mt-0">
          <RefreshCw className={`w-4 h-4 ${isLoading ? "animate-spin" : ""}`} />
          <span>{t("sec1_btn_generate")}</span>
        </button>
      </div>

      {/* Content Card */}
      <div className="card p-5">
        {hospitals.length > 0 && active ? (
          <>
            {/* Hospital Tabs */}
            <div className="flex items-center gap-2 mb-5 overflow-x-auto pb-1">
              {hospitals.map((h, idx) => (
                <button
                  key={h.hospital_id}
                  onClick={() => setActiveTab(idx)}
                  className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all whitespace-nowrap ${
                    activeTab === idx
                      ? "bg-accent-500/15 text-accent-400 border border-accent-500/20"
                      : "text-slate-500 hover:text-slate-300 border border-transparent"
                  }`}
                >
                  {t("sec1_hospital_prefix")} {h.hospital_id}
                </button>
              ))}
            </div>

            {/* Equal 2-Column Layout */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">

              {/* Left: Stats */}
              <div className="space-y-4">
                <div className="card-inner p-4">
                  <div className="text-sm font-bold text-white mb-0.5">{active.name}</div>
                  <div className="text-[11px] text-slate-500">
                    Node <code className="text-accent-400">client_{active.hospital_id}</code> | {active.num_samples} Scans
                  </div>
                </div>

                <div className="card-inner p-4">
                  <div className="metric-label mb-3">{t("sec1_stats_title")}</div>
                  {[
                    { key: "normal", label: t("sec1_normal"), count: active.counts.normal, dist: active.distribution[0], idx: 0 },
                    { key: "pneumonia", label: t("sec1_pneumonia"), count: active.counts.pneumonia, dist: active.distribution[1], idx: 1 },
                    { key: "covid", label: t("sec1_covid"), count: active.counts.covid, dist: active.distribution[2], idx: 2 },
                  ].map((item) => (
                    <div key={item.key} className="flex items-center justify-between py-2 border-b border-white/[0.04] last:border-0">
                      <div className="flex items-center gap-2 text-xs text-slate-300">
                        <span className={`w-2 h-2 rounded-full ${CLASS_DOT_COLORS[item.idx]}`} />
                        <span>{item.label}</span>
                      </div>
                      <div className="flex items-center gap-3">
                        <div className="w-24 h-1.5 rounded-full bg-navy-800 overflow-hidden">
                          <div
                            className={`h-full rounded-full ${CLASS_COLORS[item.idx]}`}
                            style={{ width: `${Math.round(item.dist * 100)}%` }}
                          />
                        </div>
                        <span className="text-xs font-mono font-bold text-white w-12 text-right">
                          {item.count}
                        </span>
                        <span className="text-[10px] text-slate-500 w-8 text-right">
                          {Math.round(item.dist * 100)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Right: 3x3 Scan Gallery */}
              <div>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-1.5 metric-label">
                    <Eye className="w-3.5 h-3.5 text-accent-400" />
                    <span>{t("sec1_sample_gallery")}</span>
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-2 card-inner p-3">
                  {active.sample_images.map((imgData, i) => (
                    <div key={i} className="xray-frame group hover:ring-1 hover:ring-accent-500/30 transition-all">
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
          <div className="text-center py-16 text-sm text-slate-500">
            {t("sec1_subtitle")}
          </div>
        )}
      </div>
    </section>
  );
}
