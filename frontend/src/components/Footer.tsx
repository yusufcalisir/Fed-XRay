"use client";

import React from "react";
import FedXRayLogo from "@/components/FedXRayLogo";
import { useLanguage } from "@/context/LanguageContext";
import { ShieldCheck, Lock, Activity } from "lucide-react";

interface FooterProps {
  maxWidth?: string;
}

export default function Footer({ maxWidth = "max-w-[1360px]" }: FooterProps) {
  const { t } = useLanguage();

  return (
    <footer className="border-t border-[var(--border-card)] bg-[var(--bg-card)]/40 backdrop-blur-xs py-8 px-4 sm:px-6 text-[var(--text-muted)] select-none">
      <div className={`${maxWidth} mx-auto space-y-5`}>
        {/* Top Tier: Identity & Security Credentials */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-xs shrink-0">
              <FedXRayLogo size={20} />
            </div>
            <div className="flex flex-col text-left">
              <div className="flex items-center gap-2">
                <span className="font-display font-extrabold text-sm text-[var(--text-heading)] tracking-tight">
                  Fed<span className="text-emerald-500 font-semibold">-</span>XRay
                </span>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20 font-bold">
                  v2.4.0 Enclave
                </span>
              </div>
              <span className="text-xs text-[var(--text-muted)] mt-0.5">
                {t("footer_text")}
              </span>
            </div>
          </div>

          {/* Privacy & Compliance Credentials Chips */}
          <div className="flex flex-wrap items-center gap-2 w-full md:w-auto">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-mono font-medium bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[var(--text-heading)]">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
              <span>{t("footer_egress")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-mono font-medium bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[var(--text-heading)]">
              <Lock className="w-3.5 h-3.5 text-sky-500 shrink-0" />
              <span>{t("footer_he")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-mono font-medium bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[var(--text-heading)]">
              <Activity className="w-3.5 h-3.5 text-violet-500 shrink-0" />
              <span>{t("footer_dp")}</span>
            </span>
          </div>
        </div>

        {/* Bottom Tier: Clinical Research Disclaimer & Copyright */}
        <div className="pt-4 border-t border-[var(--border-subtle)] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-[11px] leading-relaxed">
          <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] max-w-3xl">
            {t("footer_disclaimer")}
          </p>
          <div className="shrink-0 text-[10px] sm:text-[11px] font-mono text-[var(--text-muted)]">
            (c) 2026 Fed-XRay Consortium.
          </div>
        </div>
      </div>
    </footer>
  );
}

