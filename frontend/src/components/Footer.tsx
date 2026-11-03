"use client";

import React from "react";
import FedXRayLogo from "@/components/FedXRayLogo";
import { useLanguage } from "@/context/LanguageContext";

interface FooterProps {
  maxWidth?: string;
}

export default function Footer({ maxWidth = "max-w-[1360px]" }: FooterProps) {
  const { t } = useLanguage();

  return (
    <footer className="border-t border-[var(--border-card)] py-7 px-4 sm:px-6 text-[var(--text-muted)] select-none">
      <div className={`${maxWidth} mx-auto`}>
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-5">

          {/* Brand identity */}
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shrink-0">
              <FedXRayLogo size={16} />
            </div>
            <div>
              <span className="font-display font-extrabold text-sm text-[var(--text-heading)] tracking-tight">
                Fed<span className="text-emerald-500">-</span>XRay
              </span>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5 leading-tight">
                {t("footer_text")}
              </p>
            </div>
          </div>

          {/* Plain text links row */}
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-[var(--text-muted)]">
            <span>Fed-XRay</span>
            <span className="opacity-30">|</span>
            <span>Research Prototype - Not for Clinical Use</span>
            <span className="opacity-30">|</span>
            <span>Privacy-Preserving Federated Learning for Medical Imaging</span>
          </div>
        </div>

        {/* Disclaimer */}
        <div className="mt-5 pt-4 border-t border-[var(--border-subtle)] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-2 text-[10px] leading-relaxed">
          <p className="text-[var(--text-muted)] max-w-3xl">
            {t("footer_disclaimer")}
          </p>
          <span className="shrink-0 font-mono text-[var(--text-muted)]">
            (c) 2026 Fed-XRay Consortium.
          </span>
        </div>
      </div>
    </footer>
  );
}
