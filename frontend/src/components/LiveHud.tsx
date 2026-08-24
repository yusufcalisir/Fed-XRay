"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Cpu, ShieldCheck, CheckCircle2 } from "lucide-react";

interface LiveHudProps {
  numHospitals: number;
  isDefenseActive: boolean;
  isTrained: boolean;
}

export default function LiveHud({ numHospitals, isDefenseActive, isTrained }: LiveHudProps) {
  const { t } = useLanguage();

  return (
    <div className="card-inner flex flex-wrap items-center justify-between gap-x-4 sm:gap-x-6 gap-y-2 px-3.5 sm:px-5 py-2.5 sm:py-3 mb-5 sm:mb-6 text-[10px] sm:text-[11px] font-medium text-[var(--text-muted)]">
      <div className="flex items-center gap-1.5 sm:gap-2">
        <span className="pulse-dot" />
        <span className="text-[var(--text-heading)] font-semibold">{t("live_hud_active")}</span>
        <span className="text-slate-400 dark:text-slate-600">|</span>
        <span>{numHospitals}/{numHospitals} {t("live_hud_nodes")}</span>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2">
        <Cpu className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400 shrink-0" />
        <span className="truncate">{t("live_hud_model")}</span>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2">
        <ShieldCheck className="w-3.5 h-3.5 text-accent-500 dark:text-accent-400 shrink-0" />
        <span className="truncate">{isDefenseActive ? t("live_hud_shield") : t("sec2_shield_off")}</span>
      </div>
      <div className="flex items-center gap-1.5 sm:gap-2">
        <CheckCircle2 className={`w-3.5 h-3.5 ${isTrained ? "text-accent-500" : "text-slate-400"} shrink-0`} />
        <span>{isTrained ? t("kpi_status_trained") : t("kpi_status_ready")}</span>
      </div>
    </div>
  );
}
