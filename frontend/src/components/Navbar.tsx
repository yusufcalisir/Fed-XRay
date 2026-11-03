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
    <header className="sticky top-0 z-50 w-full h-14 bg-[var(--navbar-bg)] backdrop-blur-xl border-b border-[var(--navbar-border)] transition-colors">
      <div className="max-w-[1360px] mx-auto px-3 sm:px-5 h-full flex items-center justify-between gap-2">

        {/* Brand / Logo */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <Link href="/" className="flex items-center gap-2 sm:gap-3 group">
            <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-xs shrink-0 group-hover:border-emerald-500/40 transition-colors">
              <FedXRayLogo size={20} />
            </div>
            <div className="flex items-baseline gap-1.5 sm:gap-2">
              <span className="font-display font-extrabold text-[15px] sm:text-[16px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
                Fed<span className="text-emerald-500 font-semibold">-</span>XRay
              </span>
              <span className="hidden md:inline text-[9px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-white/[0.06] text-slate-600 dark:text-slate-400 border border-slate-300/60 dark:border-white/[0.06]">
                Clinical CDSS
              </span>
            </div>
          </Link>

          {mode === "dashboard" && (
            <Link
              href="/"
              className="hidden lg:inline-flex items-center gap-1.5 ml-3 px-2.5 py-1 rounded-lg text-xs font-semibold text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all border border-transparent hover:border-[var(--border-subtle)]"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{t("nav_dashboard_overview")}</span>
            </Link>
          )}
        </div>

        {/* Center Navigation Controls (Landing Mode Only, Smooth Scroll Without Hash) */}
        {mode === "landing" && (
          <nav className="hidden lg:flex items-center gap-6 text-xs font-medium text-[var(--text-muted)]">
            <button
              type="button"
              onClick={() => scrollTo("results")}
              className="hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_results")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("security")}
              className="hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_security")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("architecture")}
              className="hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_architecture")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("benchmarks")}
              className="hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_benchmarks")}
            </button>
          </nav>
        )}

        {/* Right Action Controls */}
        <div className="flex items-center gap-2 sm:gap-3">

          {/* Clean Enterprise Engine Status Indicator */}
          <div
            className="hidden sm:inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[11px] font-medium text-[var(--text-muted)] bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] select-none"
          >
            <span
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                isConnected
                  ? "bg-emerald-500 shadow-[0_0_6px_#10b981]"
                  : isChecking
                  ? "bg-amber-400 animate-pulse"
                  : "bg-rose-500 animate-pulse"
              }`}
            />
            <span className="font-mono text-[10px] tracking-wide">{connectionLabel}</span>
          </div>

          {/* Theme & Language Controls Strip */}
          <div className="flex items-center gap-1 pl-1">
            {/* Theme Toggle Button */}
            <button
              onClick={toggleTheme}
              aria-label="Toggle Theme"
              className="p-1.5 rounded-md text-[var(--text-muted)] hover:text-[var(--text-heading)] hover:bg-[var(--bg-card-inner)] transition-colors shrink-0"
            >
              {isDark ? <Sun className="w-3.5 h-3.5 text-amber-400" /> : <Moon className="w-3.5 h-3.5 text-slate-600" />}
            </button>

            {/* Language Switcher */}
            <div className="flex items-center gap-0.5 text-[10px] font-mono font-semibold text-[var(--text-muted)]">
              <button
                type="button"
                onClick={() => setLanguage("EN")}
                className={`px-1.5 py-0.5 rounded transition-all ${
                  language === "EN"
                    ? "text-[var(--text-heading)] font-bold bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] shadow-xs"
                    : "hover:text-[var(--text-heading)]"
                }`}
              >
                EN
              </button>
              <span className="text-[var(--border-card)]">/</span>
              <button
                type="button"
                onClick={() => setLanguage("TR")}
                className={`px-1.5 py-0.5 rounded transition-all ${
                  language === "TR"
                    ? "text-[var(--text-heading)] font-bold bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] shadow-xs"
                    : "hover:text-[var(--text-heading)]"
                }`}
              >
                TR
              </button>
            </div>
          </div>

          <div className="h-4 w-px bg-[var(--border-subtle)] hidden sm:block" />

          {/* Launch Live Consortium Primary Action */}
          {mode === "landing" && (
            <button
              onClick={onLaunchDemo}
              className="inline-flex items-center gap-1.5 px-3 sm:px-3.5 py-1.5 rounded-lg text-xs font-semibold bg-slate-900 hover:bg-slate-800 text-white dark:bg-emerald-600 dark:hover:bg-emerald-500 dark:text-white border border-slate-800 dark:border-emerald-500/40 shadow-xs hover:shadow transition-all shrink-0 active:scale-[0.98]"
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
              className="lg:hidden inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs font-medium text-[var(--text-muted)] border border-[var(--border-card)] hover:text-[var(--text-heading)]"
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
