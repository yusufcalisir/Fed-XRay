"use client";

import React from "react";
import Link from "next/link";
import FedXRayLogo from "@/components/FedXRayLogo";
import { useLanguage } from "@/context/LanguageContext";
import { ShieldCheck, Lock, Cpu, AlertCircle, ArrowUpRight } from "lucide-react";

interface FooterProps {
  maxWidth?: string;
}

export default function Footer({ maxWidth = "max-w-[1360px]" }: FooterProps) {
  const { t } = useLanguage();

  return (
    <footer className="border-t border-[var(--border-card)] bg-[var(--bg-app)] py-10 px-4 sm:px-6 text-[var(--text-muted)] select-none">
      <div className={`${maxWidth} mx-auto space-y-8`}>
        {/* Top Tier: Brand Identity & Privacy Pillars */}
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 pb-6 border-b border-[var(--border-subtle)]">
          {/* Brand Identity */}
          <div className="flex items-start sm:items-center gap-3.5">
            <FedXRayLogo size={32} className="shrink-0 mt-0.5 sm:mt-0" />
            <div>
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-display font-black text-base text-[var(--text-heading)] tracking-tight">
                  Fed<span className="text-emerald-500 font-semibold">-</span>XRay
                </span>
                <span className="text-[10px] font-mono font-bold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-md border border-emerald-500/20">
                  PPFL Core v2.4
                </span>
              </div>
              <p className="text-xs text-[var(--text-muted)] mt-1 max-w-md leading-relaxed">
                {t("footer_text")}
              </p>
            </div>
          </div>

          {/* Privacy & Cryptography Pillars */}
          <div className="flex flex-wrap items-center gap-2">
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[11px] font-medium text-[var(--text-heading)]">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-500" />
              <span>{t("footer_egress")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[11px] font-medium text-[var(--text-heading)]">
              <Lock className="w-3.5 h-3.5 text-sky-500" />
              <span>{t("footer_he")}</span>
            </span>
            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[11px] font-medium text-[var(--text-heading)]">
              <Cpu className="w-3.5 h-3.5 text-purple-500" />
              <span>{t("footer_dp")}</span>
            </span>
          </div>
        </div>

        {/* Middle Tier: Clean Structured Navigation Links */}
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <span className="text-[11px] font-mono font-semibold uppercase tracking-wider text-[var(--text-muted)]">
            {t("footer_nav_title")}
          </span>
          <nav className="grid grid-cols-2 sm:grid-cols-3 md:flex md:items-center gap-x-6 gap-y-3 text-xs font-semibold" aria-label="Footer Navigation">
            <Link
              href="/#results"
              className="text-[var(--text-muted)] hover:text-emerald-500 transition-colors"
            >
              {t("nav_results")}
            </Link>
            <Link
              href="/#security"
              className="text-[var(--text-muted)] hover:text-emerald-500 transition-colors"
            >
              {t("nav_security")}
            </Link>
            <Link
              href="/#architecture"
              className="text-[var(--text-muted)] hover:text-emerald-500 transition-colors"
            >
              {t("nav_architecture")}
            </Link>
            <Link
              href="/#benchmarks"
              className="text-[var(--text-muted)] hover:text-emerald-500 transition-colors"
            >
              {t("nav_benchmarks")}
            </Link>
            <Link
              href="/dashboard"
              className="inline-flex items-center gap-1 text-emerald-600 dark:text-emerald-400 hover:text-emerald-500 transition-colors font-bold col-span-2 sm:col-span-1"
            >
              <span>{t("nav_dashboard_overview")}</span>
              <ArrowUpRight className="w-3.5 h-3.5" />
            </Link>
          </nav>
        </div>

        {/* Regulatory Clinical Disclaimer Card */}
        <div className="p-3.5 sm:p-4 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] flex items-start gap-3">
          <AlertCircle className="w-4 h-4 text-amber-500 dark:text-amber-400 shrink-0 mt-0.5" />
          <div className="space-y-1 text-[11px] leading-relaxed">
            <p className="text-[var(--text-main)] font-medium">
              {t("footer_disclaimer")}
            </p>
            <p className="text-[var(--text-muted)] text-[10px]">
              {t("footer_bench_note")}
            </p>
          </div>
        </div>

        {/* Bottom Tier: Copyright & Consortium Verification */}
        <div className="pt-2 flex flex-col sm:flex-row items-center justify-between gap-3 text-[11px] font-mono text-[var(--text-muted)]">
          <span>(c) 2026 Fed-XRay Consortium. {t("footer_tagline")}</span>
          <span className="text-[10px] text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
            {t("footer_badge_strategy_e")}
          </span>
        </div>
      </div>
    </footer>
  );
}
