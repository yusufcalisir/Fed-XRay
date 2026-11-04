"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Sun, Moon, ArrowLeft, Menu, X, ShieldCheck, ArrowRight } from "lucide-react";
import FedXRayLogo from "@/components/FedXRayLogo";

interface NavbarProps {
  mode?: "landing" | "dashboard";
  onLaunchDemo?: () => void;
}

export default function Navbar({ mode = "landing", onLaunchDemo }: NavbarProps) {
  const { language, setLanguage, t } = useLanguage();
  const { isConnected, isChecking, reconnectAttempts } = useApi();
  const [isDark, setIsDark] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

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

  // Close mobile menu on resize to desktop
  useEffect(() => {
    const handleResize = () => {
      if (window.innerWidth >= 1024) {
        setIsMobileMenuOpen(false);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => window.removeEventListener("resize", handleResize);
  }, []);

  // Lock body scroll when fullscreen mobile menu is open
  useEffect(() => {
    if (isMobileMenuOpen) {
      document.body.style.overflow = "hidden";
    } else {
      document.body.style.overflow = "";
    }
    return () => {
      document.body.style.overflow = "";
    };
  }, [isMobileMenuOpen]);

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

  // Engine status dot color
  const dotClass = isConnected
    ? "bg-emerald-500 shadow-[0_0_8px_#10b981]"
    : isChecking
    ? "bg-amber-400 animate-pulse"
    : reconnectAttempts > 0
    ? "bg-amber-400 animate-pulse"
    : "bg-rose-500 animate-pulse";

  const scrollTo = (id: string) => {
    setIsMobileMenuOpen(false);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  return (
    <>
      <header className="sticky top-0 z-40 w-full bg-[var(--navbar-bg)]/95 backdrop-blur-md border-b border-[var(--navbar-border)] transition-colors select-none">
        <div className="max-w-[1360px] mx-auto px-4 sm:px-6 h-14 flex items-center justify-between gap-3">

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

            {/* Clinical CDSS tag */}
            <span className="hidden md:inline text-[10px] font-semibold uppercase tracking-widest text-[var(--text-muted)] border-l border-[var(--border-subtle)] pl-3">
              {t("nav_clinical_cdss")}
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

          {/* Center Desktop Navigation */}
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
          <div className="flex items-center gap-2 sm:gap-3 shrink-0">

            {/* Engine status dot - desktop only */}
            <span
              title={connectionLabel}
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

            {/* Language switcher */}
            <div className="flex items-center text-[12px] font-semibold font-mono">
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

            {/* Desktop Launch CTA */}
            {mode === "landing" && (
              <button
                type="button"
                onClick={onLaunchDemo}
                className="hidden sm:inline-flex items-center gap-2 px-4 py-2 rounded-md text-[13px] font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-colors shrink-0 active:scale-[0.98]"
              >
                <span>{t("nav_launch_demo")}</span>
              </button>
            )}

            {/* Dashboard Mode: Desktop Overview Link */}
            {mode === "dashboard" && (
              <Link
                href="/"
                className="hidden sm:inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium text-[var(--text-muted)] border border-[var(--border-card)] hover:text-[var(--text-heading)] transition-colors"
              >
                <ArrowLeft className="w-3.5 h-3.5" />
                <span>{t("nav_overview_short")}</span>
              </Link>
            )}

            {/* Mobile Hamburger Menu Toggle Button */}
            <button
              type="button"
              onClick={() => setIsMobileMenuOpen((prev) => !prev)}
              aria-label={isMobileMenuOpen ? "Toggle Navigation Menu" : "Open Navigation Menu"}
              aria-expanded={isMobileMenuOpen}
              className="lg:hidden w-8 h-8 flex items-center justify-center text-[var(--text-heading)] rounded-md hover:bg-[var(--bg-card-inner)] border border-[var(--border-card)] transition-colors focus:outline-none"
            >
              {isMobileMenuOpen ? <X className="w-4 h-4" /> : <Menu className="w-4 h-4" />}
            </button>

          </div>

        </div>
      </header>

      {/* Fullscreen Mobile Navigation Menu Overlay */}
      {isMobileMenuOpen && (
        <div className="lg:hidden fixed inset-0 z-50 bg-[var(--bg-app)] flex flex-col justify-between p-5 sm:p-6 animate-in fade-in duration-200 overflow-y-auto">

          {/* Top Bar inside Overlay */}
          <div className="flex items-center justify-between pb-4 border-b border-[var(--border-subtle)] shrink-0">
            <Link
              href="/"
              onClick={() => setIsMobileMenuOpen(false)}
              className="flex items-center gap-2.5"
            >
              <div className="w-8 h-8 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shrink-0">
                <FedXRayLogo size={20} />
              </div>
              <div className="flex items-center gap-2">
                <span className="font-display font-extrabold text-base tracking-tight text-[var(--text-heading)]">
                  Fed<span className="text-emerald-500">-</span>XRay
                </span>
                <span className="text-[10px] font-semibold uppercase tracking-wider text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20">
                  CDSS
                </span>
              </div>
            </Link>

            <button
              type="button"
              onClick={() => setIsMobileMenuOpen(false)}
              aria-label="Close Navigation Menu"
              className="w-9 h-9 flex items-center justify-center text-[var(--text-heading)] rounded-lg hover:bg-[var(--bg-card-inner)] border border-[var(--border-card)] transition-colors focus:outline-none"
            >
              <X className="w-5 h-5" />
            </button>
          </div>

          {/* Main Navigation Links Tier */}
          <div className="flex-1 py-8 flex flex-col justify-center space-y-4">
            <div className="text-[10px] font-mono font-bold uppercase tracking-widest text-emerald-600 dark:text-emerald-400 mb-1">
              {t("nav_mobile_title")}
            </div>

            {mode === "landing" && (
              <div className="flex flex-col">
                {[
                  { id: "results", num: "01", label: t("nav_results") },
                  { id: "security", num: "02", label: t("nav_security") },
                  { id: "architecture", num: "03", label: t("nav_architecture") },
                  { id: "benchmarks", num: "04", label: t("nav_benchmarks") },
                ].map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    onClick={() => scrollTo(item.id)}
                    className="group w-full py-4 flex items-center justify-between text-left border-b border-[var(--border-subtle)] hover:border-emerald-500/50 transition-colors focus:outline-none"
                  >
                    <div className="flex items-center gap-3.5">
                      <span className="font-mono text-xs text-[var(--text-muted)] group-hover:text-emerald-500 transition-colors">
                        {item.num}
                      </span>
                      <span className="font-display text-xl sm:text-2xl font-bold text-[var(--text-heading)] group-hover:text-emerald-600 dark:group-hover:text-emerald-400 transition-colors">
                        {item.label}
                      </span>
                    </div>
                    <ArrowRight className="w-5 h-5 text-[var(--text-muted)] group-hover:text-emerald-500 group-hover:translate-x-1 transition-all" />
                  </button>
                ))}
              </div>
            )}

            {mode === "dashboard" && (
              <Link
                href="/"
                onClick={() => setIsMobileMenuOpen(false)}
                className="group w-full py-4 flex items-center justify-between text-left border-b border-[var(--border-subtle)]"
              >
                <div className="flex items-center gap-3.5">
                  <span className="font-mono text-xs text-[var(--text-muted)]">01</span>
                  <span className="font-display text-xl sm:text-2xl font-bold text-[var(--text-heading)] group-hover:text-emerald-500 transition-colors">
                    {t("nav_dashboard_overview")}
                  </span>
                </div>
                <ArrowLeft className="w-5 h-5 text-[var(--text-muted)]" />
              </Link>
            )}
          </div>

          {/* Bottom Dock Tier inside Fullscreen Menu */}
          <div className="pt-4 border-t border-[var(--border-subtle)] space-y-3.5 shrink-0">

            {/* Mobile Language Switcher */}
            <div className="flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-xs">
              <span className="text-[var(--text-muted)] font-medium">{language === "TR" ? "Arayüz Dili" : "Interface Language"}</span>
              <div className="flex items-center gap-1 font-mono font-semibold text-xs">
                <button
                  type="button"
                  onClick={() => setLanguage("EN")}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    language === "EN"
                      ? "bg-emerald-500 text-white font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)] bg-[var(--bg-app)]"
                  }`}
                >
                  EN
                </button>
                <button
                  type="button"
                  onClick={() => setLanguage("TR")}
                  className={`px-2.5 py-1 rounded-md transition-colors ${
                    language === "TR"
                      ? "bg-emerald-500 text-white font-bold"
                      : "text-[var(--text-muted)] hover:text-[var(--text-heading)] bg-[var(--bg-app)]"
                  }`}
                >
                  TR
                </button>
              </div>
            </div>

            {/* Quick Engine Status */}
            <div className="flex items-center justify-between px-3.5 py-2.5 rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-xs">
              <span className="text-[var(--text-muted)] font-medium">{t("nav_mobile_engine")}</span>
              <div className="flex items-center gap-2">
                <span className={`w-2 h-2 rounded-full ${dotClass}`} />
                <span className="font-mono text-[11px] font-semibold text-[var(--text-heading)]">
                  {connectionLabel}
                </span>
              </div>
            </div>

            {/* Launch Consortium Primary Action Button */}
            {mode === "landing" && (
              <button
                type="button"
                onClick={() => {
                  setIsMobileMenuOpen(false);
                  if (onLaunchDemo) onLaunchDemo();
                }}
                className="w-full py-3.5 px-4 rounded-xl text-sm font-semibold bg-emerald-600 hover:bg-emerald-500 text-white transition-all flex items-center justify-center gap-2 shadow-lg shadow-emerald-600/20 active:scale-[0.99]"
              >
                <ShieldCheck className="w-4 h-4 shrink-0" />
                <span>{t("nav_launch_demo")}</span>
                <ArrowRight className="w-4 h-4 shrink-0" />
              </button>
            )}

            {/* Security Guarantee Invariant */}
            <div className="text-center text-[10px] font-mono text-[var(--text-muted)] pt-1">
              {t("nav_mobile_security_guarantee")}
            </div>
          </div>

        </div>
      )}
    </>
  );
}
