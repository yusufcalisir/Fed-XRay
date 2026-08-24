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
    { icon: Building2, label: t("kpi_hospitals"), value: String(numHospitals), color: "text-accent-400" },
    { icon: Repeat, label: t("kpi_rounds"), value: String(totalRounds), color: "text-blue-400" },
    { icon: Users, label: t("kpi_samples"), value: totalSamples.toLocaleString(), color: "text-amber-400" },
    { icon: Activity, label: t("kpi_status"), value: isTrained ? t("kpi_status_trained") : t("kpi_status_ready"), color: isTrained ? "text-accent-400" : "text-slate-400" },
  ];

  return (
    <div className="mb-6">
      {/* Title Row */}
      <div className="mb-5 text-center">
        <h1 className="font-display text-2xl sm:text-3xl font-black text-white tracking-tight mb-1.5">
          {t("hero_title")}
        </h1>
        <p className="text-sm text-slate-500 max-w-xl mx-auto leading-relaxed">
          {t("hero_subtitle")}
        </p>
      </div>

      {/* KPI Strip */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
        {metrics.map((m, i) => (
          <div key={i} className="card p-4 flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-navy-800 flex items-center justify-center shrink-0">
              <m.icon className={`w-4 h-4 ${m.color}`} />
            </div>
            <div className="min-w-0">
              <div className="metric-label truncate">{m.label}</div>
              <div className="metric-value text-lg truncate">{m.value}</div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
