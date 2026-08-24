"use client";

import React, { useEffect, useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Server, Sun, Moon, Activity, Shield } from "lucide-react";

export default function Navbar() {
  const { language, setLanguage } = useLanguage();
  const { apiUrl, setApiUrl, isConnected } = useApi();
  const [isDark, setIsDark] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [inputUrl, setInputUrl] = useState(apiUrl);

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

  const handleSaveUrl = (e: React.FormEvent) => {
    e.preventDefault();
    setApiUrl(inputUrl);
    setShowConfig(false);
  };

  return (
    <header className="sticky top-0 z-50 w-full h-14 bg-[var(--navbar-bg)] backdrop-blur-xl border-b border-[var(--navbar-border)] transition-colors">
      <div className="max-w-[1360px] mx-auto px-5 h-full flex items-center justify-between">

        {/* Brand / Logo */}
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] dark:text-accent-400 border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-sm">
            <Activity className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="flex items-baseline gap-2">
            <span className="font-display font-extrabold text-[16px] tracking-tight text-[var(--text-heading)]">
              Fed<span className="text-emerald-500 font-semibold">-</span>XRay
            </span>
            <span className="hidden sm:inline text-[9px] font-bold tracking-widest uppercase px-1.5 py-0.5 rounded bg-slate-200/80 dark:bg-white/[0.06] text-slate-600 dark:text-slate-400 border border-slate-300/60 dark:border-white/[0.06]">
              Clinical CDSS
            </span>
          </div>
        </div>

        {/* Right Action Controls */}
        <div className="flex items-center gap-2.5">

          {/* Backend Connection Indicator & Configuration */}
          <div className="relative">
            <button
              onClick={() => { setInputUrl(apiUrl); setShowConfig(!showConfig); }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-[11px] font-medium border transition-all ${
                isConnected
                  ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/15"
                  : "bg-rose-500/10 border-rose-500/20 text-rose-600 dark:text-rose-400 hover:bg-rose-500/15"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-emerald-500 shadow-[0_0_6px_#10b981]" : "bg-rose-500 animate-pulse"}`} />
              <span className="hidden sm:inline font-semibold">{isConnected ? "Engine Connected" : "Engine Offline"}</span>
              <Server className="w-3 h-3 opacity-60 ml-0.5" />
            </button>

            {showConfig && (
              <div className="absolute right-0 mt-2 w-84 p-4 card z-50 shadow-2xl animate-fade-in">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-[var(--text-heading)]">FastAPI Gateway</span>
                  <span className={`text-[10px] font-mono font-bold px-1.5 py-0.5 rounded ${isConnected ? "bg-emerald-500/15 text-emerald-400" : "bg-rose-500/15 text-rose-400"}`}>
                    {isConnected ? "200 OK" : "DISCONNECTED"}
                  </span>
                </div>
                <form onSubmit={handleSaveUrl} className="space-y-2.5">
                  <input
                    type="url"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="http://127.0.0.1:8000"
                    className="w-full px-3 py-2 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-xs text-[var(--text-heading)] focus:outline-none focus:border-emerald-500/50"
                  />
                  <div className="flex items-center justify-end gap-2">
                    <button
                      type="button"
                      onClick={() => setShowConfig(false)}
                      className="px-3 py-1 rounded-md text-[11px] text-slate-500 hover:text-[var(--text-heading)]"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-3.5 py-1.5 rounded-md bg-slate-900 dark:bg-white text-white dark:text-slate-900 text-[11px] font-bold shadow-sm"
                    >
                      Apply URL
                    </button>
                  </div>
                </form>
              </div>
            )}
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="p-2 rounded-lg border border-[var(--border-card)] text-slate-500 hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
          </button>

          {/* Language Switcher */}
          <div className="flex items-center rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] p-0.5">
            <button
              onClick={() => setLanguage("EN")}
              className={`px-2.5 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "EN"
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-[var(--text-heading)]"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage("TR")}
              className={`px-2.5 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "TR"
                  ? "bg-slate-900 text-white dark:bg-white dark:text-slate-900 shadow-sm"
                  : "text-slate-500 hover:text-[var(--text-heading)]"
              }`}
            >
              TR
            </button>
          </div>

        </div>
      </div>
    </header>
  );
}
