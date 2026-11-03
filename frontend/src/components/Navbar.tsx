"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Sun, Moon, ArrowLeft } from "lucide-react";
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

  // Engine status dot color only - no text label in navbar
  const dotClass = isConnected
    ? "bg-emerald-500"
    : isChecking
    ? "bg-amber-400 animate-pulse"
    : reconnectAttempts > 0
    ? "bg-amber-400 animate-pulse"
    : "bg-rose-500";

  const scrollTo = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full h-14 bg-[var(--navbar-bg)] border-b border-[var(--navbar-border)] transition-colors select-none">
      <div className="max-w-[1360px] mx-auto px-4 sm:px-6 h-full flex items-center justify-between gap-4">

        {/* Brand / Logo */}
        <div className="flex items-center gap-3 shrink-0">
          <Link href="/" className="flex items-center gap-2.5 group">
            <div className="w-7 h-7 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shrink-0 group-hover:border-emerald-500/60 transition-colors">
              <FedXRayLogo size={18} />
            </div>
            <span className="font-display font-extrabold text-[15px] tracking-tight text-[var(--text-heading)] whitespace-nowrap">
              Fed<span className="text-emerald-500">-</span>XRay
            </span>
          </Link>

          {/* Clinical CDSS tag - always visible on md+ */}
          <span className="hidden md:inline text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)] border-l border-[var(--border-subtle)] pl-3">
            Clinical CDSS
          </span>

          {mode === "dashboard" && (
            <Link
              href="/"
              className="hidden lg:inline-flex items-center gap-1.5 ml-2 text-xs font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors"
            >
              <ArrowLeft className="w-3.5 h-3.5" />
              <span>{t("nav_dashboard_overview")}</span>
            </Link>
          )}
        </div>

        {/* Center Navigation - plain links, no pill container */}
        {mode === "landing" && (
          <nav className="hidden lg:flex items-center gap-6" aria-label="Site navigation">
            <button
              type="button"
              onClick={() => scrollTo("results")}
              className="text-[13px] font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_results")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("security")}
              className="text-[13px] font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_security")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("architecture")}
              className="text-[13px] font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_architecture")}
            </button>
            <button
              type="button"
              onClick={() => scrollTo("benchmarks")}
              className="text-[13px] font-medium text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors focus:outline-none"
            >
              {t("nav_benchmarks")}
            </button>
          </nav>
        )}

        {/* Right Controls */}
        <div className="flex items-center gap-3 shrink-0">

          {/* Engine status dot - minimal, no label text */}
          <span
            title={isConnected ? "Engine Connected" : isChecking ? "Connecting..." : "Engine Offline"}
            className={`hidden sm:block w-1.5 h-1.5 rounded-full shrink-0 ${dotClass}`}
          />

          {/* Theme toggle */}
          <button
            type="button"
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="w-8 h-8 flex items-center justify-center text-[var(--text-muted)] hover:text-[var(--text-heading)] transition-colors rounded-md hover:bg-[var(--bg-card-inner)]"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4" />}
          </button>

          {/* Language switcher - plain text, no pill container */}
          <div className="flex items-center gap-0 text-[12px] font-semibold font-mono">
            <button
              type="button"
              onClick={() => setLanguage("EN")}
              className={`px-1.5 py-0.5 transition-colors ${
                language === "EN"
                  ? "text-[var(--text-heading)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              EN
            </button>
            <span className="text-[var(--border-card)] select-none">/</span>
            <button
              type="button"
              onClick={() => setLanguage("TR")}
              className={`px-1.5 py-0.5 transition-colors ${
                language === "TR"
                  ? "text-[var(--text-heading)]"
                  : "text-[var(--text-muted)] hover:text-[var(--text-heading)]"
              }`}
            >
              TR
            </button>
          </div>

          {/* Launch CTA - solid rectangular button */}
          {mode === "landing" && (
            <button
              type="button"
              onClick={onLaunchDemo}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-md text-[13px] font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors shrink-0 active:scale-[0.98]"
            >
              <span className="hidden sm:inline">{t("nav_launch_demo")}</span>
              <span className="sm:hidden">{t("nav_demo_short")}</span>
            </button>
          )}

          {/* Back to Overview (dashboard mode, mobile) */}
          {mode === "dashboard" && (
            <Link
              href="/"
              className="lg:hidden inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-[var(--text-muted)] border border-[var(--border-card)] hover:text-[var(--text-heading)] transition-colors"
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
