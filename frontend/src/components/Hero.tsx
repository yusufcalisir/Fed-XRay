"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Building2, Repeat, Users, Activity } from "lucide-react";

interface HeroProps {
  numHospitals: number;
  totalRounds: number;
  totalSamples: number;
  isTrained: boolean;
}

export default function Hero({ numHospitals, totalRounds, totalSamples, isTrained }: HeroProps) {
  const { t } = useLanguage();

  const metrics = [
    { icon: Building2, label: t("kpi_hospitals"), value: String(numHospitals), color: "text-accent-500" },
    { icon: Repeat, label: t("kpi_rounds"), value: String(totalRounds), color: "text-blue-500 dark:text-blue-400" },
    { icon: Users, label: t("kpi_samples"), value: totalSamples.toLocaleString(), color: "text-amber-500 dark:text-amber-400" },
    { icon: Activity, label: t("kpi_status"), value: isTrained ? t("kpi_status_trained") : t("kpi_status_ready"), color: isTrained ? "text-accent-500" : "text-slate-400" },
  ];

  return (
    <div className="mb-5 sm:mb-6">
      {/* Title Row */}
      <div className="mb-4 sm:mb-5 text-center px-2">
        <h1 className="font-display text-xl sm:text-2xl lg:text-3xl font-black text-[var(--text-heading)] tracking-tight mb-1 sm:mb-1.5 leading-tight">
          {t("hero_title")}
        </h1>
        <p className="text-xs sm:text-sm text-[var(--text-muted)] max-w-xl mx-auto leading-relaxed">
          {t("hero_subtitle")}
        </p>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2 sm:gap-3">
        {metrics.map((m, i) => (
          <div key={i} className="card p-3 sm:p-4 flex items-center gap-2.5 sm:gap-3">
            <div className="w-8 h-8 sm:w-9 sm:h-9 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] flex items-center justify-center shrink-0">
              <m.icon className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${m.color}`} />
            </div>
            <div className="min-w-0">
              <div className="metric-label truncate text-[10px] sm:text-[11px]">{m.label}</div>
              <div className="metric-value text-base sm:text-lg truncate">{m.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
