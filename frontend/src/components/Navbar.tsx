"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Activity, ShieldCheck, Globe, Moon, Sun } from "lucide-react";

export default function Navbar() {
  const { language, setLanguage, t } = useLanguage();
  const [isDark, setIsDark] = React.useState(true);

  const toggleTheme = () => {
    setIsDark(!isDark);
    if (!isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  return (
    <header className="sticky top-0 z-50 w-full border-b border-slate-200/80 dark:border-slate-800/80 bg-white/80 dark:bg-slate-950/80 backdrop-blur-xl">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
        
        {/* Brand Logo & Tag */}
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-brand-600 to-clinical-cyan flex items-center justify-center text-white shadow-lg shadow-brand-500/20 font-bold text-xl">
            🫁
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-display font-extrabold text-xl tracking-tight bg-gradient-to-r from-slate-900 via-brand-600 to-clinical-cyan dark:from-white dark:via-brand-400 dark:to-clinical-cyan bg-clip-text text-transparent">
                Fed-XRay
              </span>
              <span className="text-[10px] uppercase font-bold tracking-widest px-2 py-0.5 rounded-full bg-brand-50 dark:bg-brand-950/80 text-brand-600 dark:text-brand-400 border border-brand-200 dark:border-brand-800">
                v2.0 CDSS
              </span>
            </div>
            <p className="text-xs text-slate-500 dark:text-slate-400 hidden sm:block">
              {t("app_badge")}
            </p>
          </div>
        </div>

        {/* Action Controls & Top-Right Language Switcher */}
        <div className="flex items-center gap-3">
          {/* Consortium Status Pill */}
          <div className="hidden md:flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-50 dark:bg-emerald-950/50 border border-emerald-200 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 text-xs font-semibold">
            <span className="pulse-dot"></span>
            <span>4 / 4 Nodes Online</span>
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-2 rounded-xl border border-slate-200 dark:border-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-900 transition-colors"
            title="Toggle Theme"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
          </button>

          {/* Top-Right Language Switcher (EN / TR) */}
          <div className="flex items-center p-1 rounded-xl bg-slate-100 dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
            <button
              onClick={() => setLanguage("EN")}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                language === "EN"
                  ? "bg-white dark:bg-brand-600 text-brand-600 dark:text-white shadow-sm"
                  : "text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
              }`}
            >
              🇺🇸 EN
            </button>
            <button
              onClick={() => setLanguage("TR")}
              className={`px-2.5 py-1 rounded-lg text-xs font-bold transition-all ${
                language === "TR"
                  ? "bg-white dark:bg-brand-600 text-brand-600 dark:text-white shadow-sm"
                  : "text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white"
              }`}
            >
              🇹🇷 TR
            </button>
          </div>

        </div>

      </div>
    </header>
  );
}
