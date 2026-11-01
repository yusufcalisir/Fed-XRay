"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Sun, Moon, Activity, ArrowLeft, ArrowRight, ShieldCheck } from "lucide-react";

interface NavbarProps {
  mode?: "landing" | "dashboard";
  onLaunchDemo?: () => void;
}

export default function Navbar({ mode = "landing", onLaunchDemo }: NavbarProps) {
  const { language, setLanguage, t } = useLanguage();
  const { isConnected, isChecking, reconnectAttempts } = useApi();
  const [isDark, setIsDark] = useState(true);

  useEffect(() => {
    const savedTheme = localStorage.getItem("theme");
    if (savedTheme) {
      const dark = savedTheme === "dark";
      setIsDark(dark);
      document.documentElement.classList.toggle("dark", dark);
    } else {
      const hasDarkClass = document.documentElement.classList.contains("dark");
      setIsDark(hasDarkClass);
    }
  }, []);

  const toggleTheme = () => {
    const newDark = !isDark;
    setIsDark(newDark);
    if (newDark) {
      document.documentElement.classList.add("dark");
      localStorage.setItem("theme", "dark");
    } else {
      document.documentElement.classList.remove("dark");
      localStorage.setItem("theme", "light");
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

  return (
    <header className="sticky top-0 z-50 w-full h-14 bg-[var(--navbar-bg)] backdrop-blur-xl border-b border-[var(--navbar-border)] transition-colors">
      <div className="max-w-[1360px] mx-auto px-3 sm:px-5 h-full flex items-center justify-between gap-2">

        {/* Brand / Logo */}
        <div className="flex items-center gap-2 sm:gap-3 shrink-0">
          <Link href="/" className="flex items-center gap-2 sm:gap-3 group">
            <div className="w-7 h-7 sm:w-8 sm:h-8 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] dark:text-accent-400 border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-sm shrink-0 group-hover:border-emerald-500/40 transition-colors">
              <Activity className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-400" />
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

        {/* Center Navigation Links (Landing Mode Only) */}
        {mode === "landing" && (
          <nav className="hidden lg:flex items-center gap-6 text-xs font-medium text-[var(--text-muted)]">
            <a href="#results" className="hover:text-[var(--text-heading)] transition-colors">
              {t("nav_results")}
            </a>
            <a href="#security" className="hover:text-[var(--text-heading)] transition-colors">
              {t("nav_security")}
            </a>
            <a href="#architecture" className="hover:text-[var(--text-heading)] transition-colors">
              {t("nav_architecture")}
            </a>
            <a href="#benchmarks" className="hover:text-[var(--text-heading)] transition-colors">
              {t("nav_benchmarks")}
            </a>
          </nav>
        )}

        {/* Right Action Controls */}
        <div className="flex items-center gap-1.5 sm:gap-2.5">

          {/* Secure Backend Connection Status Badge */}
          <div
            className={`flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 rounded-lg text-[11px] font-medium border select-none transition-all ${
              isConnected
                ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400"
                : isChecking
                ? "bg-amber-500/10 border-amber-500/20 text-amber-600 dark:text-amber-400"
                : "bg-rose-500/10 border-rose-500/20 text-rose-600 dark:text-rose-400"
            }`}
          >
            <span
              className={`w-1.5 h-1.5 rounded-full shrink-0 ${
                isConnected
                  ? "bg-emerald-500 shadow-[0_0_6px_#10b981]"
                  : isChecking
                  ? "bg-amber-400 animate-pulse shadow-[0_0_6px_#f59e0b]"
                  : "bg-rose-500 animate-pulse"
              }`}
            />
            <span className="hidden sm:inline font-semibold">{connectionLabel}</span>
            <span className="inline sm:hidden font-semibold">{mobileLabel}</span>
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="p-1.5 sm:p-2 rounded-lg border border-[var(--border-card)] text-slate-500 hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all shrink-0"
          >
            {isDark ? <Sun className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-amber-400" /> : <Moon className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-slate-700" />}
          </button>

          {/* Language Switcher */}
          <div className="flex items-center rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] p-0.5 shrink-0">
            <button
              onClick={() => setLanguage("EN")}
              className={`px-2 sm:px-2.5 py-1 rounded-md text-[10px] sm:text-[11px] font-bold transition-all ${
                language === "EN"
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-[var(--text-heading)]"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage("TR")}
              className={`px-2 sm:px-2.5 py-1 rounded-md text-[10px] sm:text-[11px] font-bold transition-all ${
                language === "TR"
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-[var(--text-heading)]"
              }`}
            >
              TR
            </button>
          </div>

          {/* Launch Live Demo Button (In Landing Mode) */}
          {mode === "landing" && (
            <button
              onClick={onLaunchDemo}
              className="inline-flex items-center gap-1.5 sm:gap-2 px-3 sm:px-3.5 py-1.5 bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white text-xs font-semibold rounded-lg shadow-sm hover:shadow transition-all shrink-0"
            >
              <ShieldCheck className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">{t("nav_launch_demo")}</span>
              <span className="sm:hidden">{t("nav_demo_short")}</span>
              <ArrowRight className="w-3.5 h-3.5" />
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
