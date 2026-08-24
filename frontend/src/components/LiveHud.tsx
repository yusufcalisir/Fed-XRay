"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Cpu, ShieldCheck, Network, CheckCircle2 } from "lucide-react";

interface LiveHudProps {
  numHospitals: number;
  isDefenseActive: boolean;
  isTrained: boolean;
}

export default function LiveHud({ numHospitals, isDefenseActive, isTrained }: LiveHudProps) {
  const { t } = useLanguage();

  return (
    <div className="w-full rounded-2xl bg-slate-900 border border-slate-800 p-4 mb-8 shadow-xl flex flex-wrap items-center justify-between gap-4 text-xs font-medium text-slate-300">
      <div className="flex items-center gap-2.5">
        <span className="pulse-dot"></span>
        <span className="text-white font-bold tracking-wide">
          {t("live_hud_active")}:
        </span>
        <span className="text-slate-400">
          {numHospitals} / {numHospitals} {t("live_hud_nodes")}
        </span>
      </div>

      <div className="flex items-center gap-2 text-slate-300">
        <Cpu className="w-4 h-4 text-brand-400" />
        <span>{t("live_hud_model")}</span>
      </div>

      <div className="flex items-center gap-2 text-slate-300">
        <ShieldCheck className="w-4 h-4 text-clinical-emerald" />
        <span>{isDefenseActive ? t("live_hud_shield") : t("sec2_shield_off")}</span>
      </div>

      <div className="flex items-center gap-2 text-slate-300">
        <CheckCircle2 className={`w-4 h-4 ${isTrained ? "text-clinical-emerald" : "text-brand-400"}`} />
        <span>
          {t("kpi_status")}: <strong className="text-white">{isTrained ? t("kpi_status_trained") : t("kpi_status_ready")}</strong>
        </span>
      </div>
    </div>
  );
}
