"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Sparkles, Building2, Repeat, Users, Activity } from "lucide-react";

interface HeroProps {
  numHospitals: number;
  totalRounds: number;
  totalSamples: number;
  isTrained: boolean;
}

export default function Hero({ numHospitals, totalRounds, totalSamples, isTrained }: HeroProps) {
  const { t } = useLanguage();

  return (
    <div className="relative overflow-hidden rounded-3xl border border-slate-200/80 dark:border-slate-800/80 bg-gradient-to-b from-white via-slate-50/50 to-slate-100/50 dark:from-slate-900 dark:via-slate-950 dark:to-slate-950 p-8 sm:p-12 mb-8 shadow-glass-elevated">
      {/* Background Radial Glow */}
      <div className="absolute -top-24 left-1/2 -translate-x-1/2 w-96 h-96 bg-brand-500/15 dark:bg-brand-500/20 blur-3xl rounded-full pointer-events-none" />
      
      <div className="relative z-10 text-center max-w-3xl mx-auto">
        {/* Pill Badge */}
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-brand-500/10 border border-brand-500/20 text-brand-600 dark:text-brand-400 text-xs font-semibold mb-6 tracking-wide">
          <Sparkles className="w-3.5 h-3.5" />
          <span>{t("app_badge")}</span>
        </div>

        {/* Hero Title */}
        <h1 className="font-display text-3xl sm:text-5xl font-black tracking-tight text-slate-900 dark:text-white mb-4 leading-tight">
          {t("hero_title")}
        </h1>

        {/* Subtitle */}
        <p className="text-slate-600 dark:text-slate-400 text-sm sm:text-base leading-relaxed mb-8">
          {t("hero_subtitle")}
        </p>

        {/* 4-Column Quick Metrics Grid */}
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4 text-left">
          <div className="p-4 sm:p-5 rounded-2xl bg-white/70 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 backdrop-blur-md shadow-sm hover:border-brand-500/40 transition-all">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <Building2 className="w-4 h-4 text-brand-500" />
              <span>{t("kpi_hospitals")}</span>
            </div>
            <div className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white">
              {numHospitals}
            </div>
          </div>

          <div className="p-4 sm:p-5 rounded-2xl bg-white/70 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 backdrop-blur-md shadow-sm hover:border-brand-500/40 transition-all">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <Repeat className="w-4 h-4 text-clinical-cyan" />
              <span>{t("kpi_rounds")}</span>
            </div>
            <div className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white">
              {totalRounds}
            </div>
          </div>

          <div className="p-4 sm:p-5 rounded-2xl bg-white/70 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 backdrop-blur-md shadow-sm hover:border-brand-500/40 transition-all">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <Users className="w-4 h-4 text-clinical-amber" />
              <span>{t("kpi_samples")}</span>
            </div>
            <div className="font-display text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-white">
              {totalSamples.toLocaleString()}
            </div>
          </div>

          <div className="p-4 sm:p-5 rounded-2xl bg-white/70 dark:bg-slate-900/70 border border-slate-200 dark:border-slate-800 backdrop-blur-md shadow-sm hover:border-brand-500/40 transition-all">
            <div className="flex items-center gap-2 text-slate-500 dark:text-slate-400 text-xs font-semibold uppercase tracking-wider mb-2">
              <Activity className="w-4 h-4 text-clinical-emerald" />
              <span>{t("kpi_status")}</span>
            </div>
            <div className={`font-display text-base sm:text-lg font-bold ${
              isTrained ? "text-clinical-emerald" : "text-brand-500"
            }`}>
              {isTrained ? `✓ ${t("kpi_status_trained")}` : t("kpi_status_ready")}
            </div>
          </div>
        </div>

      </div>
    </div>
  );
}
