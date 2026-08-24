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
    <div className="card-inner flex flex-wrap items-center justify-between gap-x-6 gap-y-2 px-5 py-3 mb-6 text-[11px] font-medium text-slate-400">
      <div className="flex items-center gap-2">
        <span className="pulse-dot" />
        <span className="text-white font-semibold">{t("live_hud_active")}</span>
        <span className="text-slate-500">|</span>
        <span>{numHospitals}/{numHospitals} {t("live_hud_nodes")}</span>
      </div>
      <div className="flex items-center gap-2">
        <Cpu className="w-3.5 h-3.5 text-blue-400" />
        <span>{t("live_hud_model")}</span>
      </div>
      <div className="flex items-center gap-2">
        <ShieldCheck className="w-3.5 h-3.5 text-accent-400" />
        <span>{isDefenseActive ? t("live_hud_shield") : t("sec2_shield_off")}</span>
      </div>
      <div className="flex items-center gap-2">
        <CheckCircle2 className={`w-3.5 h-3.5 ${isTrained ? "text-accent-400" : "text-slate-500"}`} />
        <span>{isTrained ? t("kpi_status_trained") : t("kpi_status_ready")}</span>
      </div>
    </div>
  );
}
