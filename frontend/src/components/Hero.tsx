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
    { icon: Building2, label: t("kpi_hospitals"), value: String(numHospitals), tone: "from-emerald-500/20 to-emerald-500/0", color: "text-emerald-500 dark:text-emerald-400" },
    { icon: Repeat, label: t("kpi_rounds"), value: String(totalRounds), tone: "from-sky-500/20 to-sky-500/0", color: "text-sky-500 dark:text-sky-400" },
    { icon: Users, label: t("kpi_samples"), value: totalSamples.toLocaleString("en-US"), tone: "from-amber-500/20 to-amber-500/0", color: "text-amber-500 dark:text-amber-400" },
    { icon: Activity, label: t("kpi_status"), value: isTrained ? t("kpi_status_trained") : t("kpi_status_ready"), tone: isTrained ? "from-emerald-500/20 to-emerald-500/0" : "from-slate-400/15 to-slate-400/0", color: isTrained ? "text-emerald-500 dark:text-emerald-400" : "text-slate-400" },
  ];

  return (
    <div className="mb-5 sm:mb-6">
      <div className="relative mb-4 sm:mb-5 px-1">
        <div className="absolute -left-2 top-0 h-full w-1 rounded-full bg-gradient-to-b from-emerald-400 via-sky-400 to-violet-400" />
        <h1 className="pl-4 font-display text-2xl sm:text-3xl lg:text-4xl font-black text-[var(--text-heading)] tracking-tight mb-1 leading-tight">
          {t("hero_title")}
        </h1>
        <p className="pl-4 text-xs sm:text-sm text-[var(--text-muted)] max-w-2xl leading-relaxed">
          {t("hero_subtitle")}
        </p>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-2.5 sm:gap-3">
        {metrics.map((m, i) => (
          <div key={i} className="card relative overflow-hidden p-3 sm:p-4 flex items-center gap-3">
            <div className={`pointer-events-none absolute inset-0 bg-gradient-to-br ${m.tone}`} />
            <div className="relative w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] flex items-center justify-center shrink-0">
              <m.icon className={`w-4 h-4 sm:w-[18px] sm:h-[18px] ${m.color}`} />
            </div>
            <div className="relative min-w-0 flex-1 overflow-hidden">
              <div className="metric-label text-[9px] sm:text-[10px] leading-tight mb-1 truncate">{m.label}</div>
              <div className="metric-value text-base sm:text-xl leading-none truncate">{m.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
