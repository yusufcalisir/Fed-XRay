"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Cpu, ShieldCheck, CheckCircle2 } from "lucide-react";

interface LiveHudProps {
  numHospitals: number;
  isDefenseActive: boolean;
  isTrained: boolean;
  modelType?: string;
  peftMode?: string;
}

export default function LiveHud({
  numHospitals,
  isDefenseActive,
  isTrained,
  modelType = "vit_base",
  peftMode = "ffa_lora",
}: LiveHudProps) {
  const { t } = useLanguage();
  const [attestationDigest, setAttestationDigest] = React.useState<string | null>(null);

  React.useEffect(() => {
    try {
      if (typeof window !== "undefined") {
        const raw = window.sessionStorage.getItem("fedxray_attestation");
        if (raw) {
          const parsed = JSON.parse(raw);
          if (parsed?.attestation_digest) {
            setAttestationDigest(parsed.attestation_digest);
          }
        }
      }
    } catch {
      // Ignore session storage errors
    }
  }, []);

  const modelLabel = modelType
    ? `${modelType.toUpperCase()} (${peftMode ? peftMode.toUpperCase().replace("_", "-") : "Full"})`
    : t("live_hud_model");

  return (
    <div className="card-inner flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2.5 sm:gap-4 px-3.5 sm:px-5 py-2.5 sm:py-3 mb-5 sm:mb-6 text-[10px] sm:text-[11px] font-medium text-[var(--text-muted)]">
      {/* Top Header: Consortium Status & Connected Nodes */}
      <div className="flex items-center justify-between pb-2 sm:pb-0 border-b sm:border-b-0 border-[var(--border-subtle)] w-full sm:w-auto gap-2">
        <div className="flex items-center gap-2 shrink-0">
          <span className="pulse-dot shrink-0" />
          <span className="text-[var(--text-heading)] font-bold tracking-wide whitespace-nowrap" suppressHydrationWarning>
            {t("live_hud_active")}
          </span>
        </div>
        <div className="flex items-center gap-1.5 shrink-0">
          <span className="hidden sm:inline text-slate-400 dark:text-slate-600">|</span>
          <span className="font-mono text-[10px] sm:text-[11px] px-2 py-0.5 sm:px-0 sm:py-0 rounded-full sm:rounded-none bg-emerald-500/10 sm:bg-transparent text-emerald-600 dark:text-emerald-400 sm:text-[var(--text-muted)] border sm:border-0 border-emerald-500/20 font-semibold sm:font-normal whitespace-nowrap" suppressHydrationWarning>
            {numHospitals}/{numHospitals} {t("live_hud_nodes")}
          </span>
        </div>
      </div>

      {/* Attributes Grid: Model, Defense Shield, Attestation, Status */}
      <div className="grid grid-cols-2 sm:flex sm:items-center gap-2 sm:gap-4 sm:gap-x-6 w-full sm:w-auto">
        <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
          <Cpu className="w-3.5 h-3.5 text-blue-500 dark:text-blue-400 shrink-0" />
          <span className="truncate font-medium text-[var(--text-heading)] sm:text-[var(--text-muted)]" suppressHydrationWarning>
            {modelLabel}
          </span>
        </div>
        <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
          <ShieldCheck className="w-3.5 h-3.5 text-accent-500 dark:text-accent-400 shrink-0" />
          <span className="truncate" suppressHydrationWarning>
            {isDefenseActive ? t("live_hud_shield") : t("sec2_shield_off")}
          </span>
        </div>
        {attestationDigest && (
          <div className="flex items-center gap-1.5 sm:gap-2 text-emerald-600 dark:text-emerald-400 font-mono min-w-0">
            <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
            <span className="truncate" title={attestationDigest} suppressHydrationWarning>
              Attested ({attestationDigest.slice(0, 10)}...)
            </span>
          </div>
        )}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          <CheckCircle2 className={`w-3.5 h-3.5 ${isTrained ? "text-accent-500" : "text-slate-400"} shrink-0`} />
          <span suppressHydrationWarning>{isTrained ? t("kpi_status_trained") : t("kpi_status_ready")}</span>
        </div>
      </div>
    </div>
  );
}
