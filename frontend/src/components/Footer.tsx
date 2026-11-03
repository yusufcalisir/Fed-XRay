"use client";

import React from "react";
import Link from "next/link";
import FedXRayLogo from "@/components/FedXRayLogo";
import { useLanguage } from "@/context/LanguageContext";

interface FooterProps {
  maxWidth?: string;
}

export default function Footer({ maxWidth = "max-w-[1360px]" }: FooterProps) {
  const { t } = useLanguage();

  return (
    <footer className="border-t border-[var(--border-card)] bg-[var(--bg-app)] py-8 px-4 sm:px-6 text-[var(--text-muted)] select-none">
      <div className={`${maxWidth} mx-auto space-y-6`}>
        {/* Main Footer Row */}
        <div className="flex flex-col md:flex-row items-start md:items-center justify-between gap-6">

          {/* Brand Identity */}
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shrink-0">
              <FedXRayLogo size={18} />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-display font-extrabold text-[15px] text-[var(--text-heading)] tracking-tight">
                  Fed<span className="text-emerald-500">-</span>XRay
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  Enclave v2.4
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-0.5">
                {t("footer_text")}
              </p>
            </div>
          </div>

          {/* Navigation Links */}
          <nav className="flex flex-wrap items-center gap-x-5 gap-y-2 text-xs font-medium" aria-label="Footer Navigation">
            <Link
              href="/#results"
              className="text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors"
            >
              {t("nav_results")}
            </Link>
            <Link
              href="/#security"
              className="text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors"
            >
              {t("nav_security")}
            </Link>
            <Link
              href="/#architecture"
              className="text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors"
            >
              {t("nav_architecture")}
            </Link>
            <Link
              href="/#benchmarks"
              className="text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors"
            >
              {t("nav_benchmarks")}
            </Link>
            <Link
              href="/dashboard"
              className="text-emerald-600 dark:text-emerald-400 hover:underline transition-colors font-semibold"
            >
              {t("nav_dashboard_overview")}
            </Link>
          </nav>
        </div>

        {/* Disclaimer & Copyright Tier */}
        <div className="pt-5 border-t border-[var(--border-subtle)] flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 text-[11px] leading-relaxed">
          <p className="text-[var(--text-muted)] max-w-3xl">
            {t("footer_disclaimer")}
          </p>
          <span className="shrink-0 text-[var(--text-muted)] font-mono text-[10px] sm:text-[11px]">
            (c) 2026 Fed-XRay Consortium.
          </span>
        </div>
      </div>
    </footer>
  );
}
