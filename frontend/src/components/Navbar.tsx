"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Sun, Moon, ArrowLeft, ArrowRight, ShieldCheck } from "lucide-react";
import FedXRayLogo from "@/components/FedXRayLogo";

interface NavbarProps {
  mode?: "landing" | "dashboard";
  onLaunchDemo?: () => void;
}

export default function Navbar({ mode = "landing", onLaunchDemo }: NavbarProps) {
  const { language, setLanguage, t } = useLanguage();
  const { isConnected, isChecking, reconnectAttempts } = useApi();
  const [isDark, setIsDark] = useState(false);

  useEffect(() => {
    let sessionTheme: string | null = null;
    try {
      if (typeof window !== "undefined") {
        sessionTheme = sessionStorage.getItem("theme");
      }
    } catch {
      // Ignore storage errors
    }

    if (sessionTheme === "dark") {
      setIsDark(true);
      document.documentElement.classList.add("dark");
    } else {
      // System always defaults to Light Theme on initial startup
      setIsDark(false);
      document.documentElement.classList.remove("dark");
      try {
        localStorage.setItem("theme", "light");
      } catch {
        // Ignore storage errors
      }
    }
  }, []);

  const toggleTheme = () => {
    const newDark = !isDark;
    setIsDark(newDark);
    if (newDark) {
      document.documentElement.classList.add("dark");
      try {
        sessionStorage.setItem("theme", "dark");
        localStorage.setItem("theme", "dark");
      } catch {
        // Ignore storage errors
      }
    } else {
      document.documentElement.classList.remove("dark");
      try {
        sessionStorage.setItem("theme", "light");
        localStorage.setItem("theme", "light");
      } catch {
        // Ignore storage errors
      }
    }
  };

  const connectionLabel = isConnected
    ? t("engine_connected")
    : isChecking
    ? t("engine_connecting")
    : reconnectAttempts > 0
    ? t("engine_reconnecting")
    : t("engine_offline");

  const mobileLabel = isConnected
    ? t("engine_mobile_live")
    : isChecking
    ? t("engine_mobile_probing")
    : t("engine_mobile_offline");

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full h-16 bg-[var(--navbar-bg)] backdrop-blur-xl border-b border-[var(--navbar-border)] transition-colors select-none">
      <div className="max-w-[1360px] mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-3">

        {/* Brand / Logo */}
        <div className="flex items-center gap-2.5 sm:gap-3 shrink-0">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-8 h-8 rounded-xl bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-xs shrink-0 group-hover:border-emerald-500/50 group-hover:shadow-[0_0_12px_rgba(16,185,129,0.25)] transition-all">
              <FedXRayLogo size={20} />
            </div>
            <div className="flex items-center gap-2">
              <span className="font-display font-extrabold text-[16px] sm:text-[17px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
                Fed<span className="text-emerald-500 font-semibold">-</span>XRay
              </span>
              <span className="hidden md:inline-flex items-center px-2 py-0.5 rounded-full text-[10px] font-bold tracking-wide uppercase bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/20">
                Clinical CDSS
              </span>
            </div>
          </Link>

          {mode === "dashboard" && (
            <Link
              href="/"
              className="hidden lg:inline-flex items-center gap-1.5 ml-2 px-3 py-1.5 rounded-full text-xs font-semibold text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card-inner)] transition-all border border-transparent hover:border-[var(--border-subtle)]"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{t("nav_dashboard_overview")}</span>
            </Link>
          )}
        </div>

        {/* Center Navigation Dock (Landing Mode Only, Smooth Scroll Without Hash) */}
        {mode === "landing" && (
          <nav className="hidden lg:flex items-center gap-1 p-1 rounded-full bg-[var(--bg-card-inner)] border border-[var(--border-card)] shadow-xs">
            <button
              type="button"
              onClick={() => scrollTo("results")}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)] transition-all focus:outline-none"
            >
              {t("nav_results")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("security")}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)] transition-all focus:outline-none"
            >
              {t("nav_security")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("architecture")}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)] transition-all focus:outline-none"
            >
              {t("nav_architecture")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("benchmarks")}
              className="px-3.5 py-1.5 rounded-full text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)] transition-all focus:outline-none"
            >
              {t("nav_benchmarks")}
            </button>
          </nav>
        )}

        {/* Right Action Controls Cluster */}
        <div className="flex items-center gap-2 sm:gap-2.5">

          {/* Clean Enterprise Engine Status Indicator */}
          <div
            className="hidden sm:inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-[11px] font-mono text-[var(--text-muted)] bg-[var(--bg-card-inner)] border border-[var(--border-card)] select-none shadow-xs"
          >
            <span
              className={`w-2 h-2 rounded-full shrink-0 ${
                isConnected
                  ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
                  : isChecking
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500 animate-pulse"
              }`}
            />
            <span className="font-medium tracking-wide">{connectionLabel}</span>
          </div>

          {/* Unified Utility Dock: Theme & Segmented Language Switcher */}
          <div className="flex items-center p-1 rounded-full bg-[var(--bg-card-inner)] border border-[var(--border-card)] shadow-xs gap-1">
            {/* Theme Toggle Button */}
            <button
              type="button"
              onClick={toggleTheme}
              aria-label="Toggle Theme"
              className="w-7 h-7 rounded-full flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card)] transition-colors shrink-0"
            >
              {isDark ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-slate-600" />}
            </button>

            <div className="h-3.5 w-px bg-[var(--border-card)]" />

            {/* Segmented Language Switcher (Zero Raw Slashes) */}
            <div className="flex items-center p-0.5 rounded-full bg-[var(--bg-card)]/60 text-[10px] font-mono font-bold">
              <button
                type="button"
                onClick={() => setLanguage("EN")}
                className={`px-2 py-0.5 rounded-full transition-all ${
                  language === "EN"
                    ? "bg-[var(--text-heading)] text-[var(--bg-app)] shadow-xs"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                EN
              </button>
              <button
                type="button"
                onClick={() => setLanguage("TR")}
                className={`px-2 py-0.5 rounded-full transition-all ${
                  language === "TR"
                    ? "bg-[var(--text-heading)] text-[var(--bg-app)] shadow-xs"
                    : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
                }`}
              >
                TR
              </button>
            </div>
          </div>

          {/* Launch Live Consortium Primary Action */}
          {mode === "landing" && (
            <button
              type="button"
              onClick={onLaunchDemo}
              className="inline-flex items-center gap-1.5 sm:gap-2 px-3.5 sm:px-4 py-2 rounded-full text-xs font-semibold bg-emerald-600 hover:bg-emerald-500 text-white border border-emerald-500/30 shadow-sm shadow-emerald-500/20 hover:shadow-emerald-500/30 transition-all shrink-0 active:scale-[0.98]"
            >
              <ShieldCheck className="w-3.5 h-3.5 shrink-0" />
              <span className="hidden sm:inline">{t("nav_launch_demo")}</span>
              <span className="sm:hidden">{t("nav_demo_short")}</span>
              <ArrowRight className="w-3.5 h-3.5 shrink-0 opacity-80" />
            </button>
          )}

          {/* Back to Overview Button (In Dashboard Mode on Mobile/Tablet) */}
          {mode === "dashboard" && (
            <Link
              href="/"
              className="lg:hidden inline-flex items-center gap-1 px-3 py-1.5 rounded-full text-xs font-medium text-[var(--text-muted)] border border-[var(--border-card)] hover:text-[var(--text-heading)]"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{t("nav_overview_short")}</span>
            </Link>
          )}

        </div>

      </div>
    </header>
  );
}
