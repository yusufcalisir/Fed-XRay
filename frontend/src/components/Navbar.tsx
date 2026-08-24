"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Moon, Sun, Server, Check, RefreshCw, Link as LinkIcon } from "lucide-react";

export default function Navbar() {
  const { language, setLanguage, t } = useLanguage();
  const { apiUrl, setApiUrl, isConnected, isChecking, checkConnection } = useApi();
  const [isDark, setIsDark] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [inputUrl, setInputUrl] = useState(apiUrl);

  const toggleTheme = () => {
    setIsDark(!isDark);
    if (!isDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const handleSaveUrl = (e: React.FormEvent) => {
    e.preventDefault();
    setApiUrl(inputUrl);
    setShowConfig(false);
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
          
          {/* Backend Connection Status Pill & Config Trigger */}
          <div className="relative">
            <button
              onClick={() => {
                setInputUrl(apiUrl);
                setShowConfig(!showConfig);
              }}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full border text-xs font-semibold transition-all ${
                isConnected
                  ? "bg-emerald-50 dark:bg-emerald-950/50 border-emerald-300 dark:border-emerald-800 text-emerald-700 dark:text-emerald-400 hover:border-emerald-400"
                  : "bg-rose-50 dark:bg-rose-950/50 border-rose-300 dark:border-rose-800 text-rose-700 dark:text-rose-400 hover:border-rose-400 animate-pulse"
              }`}
              title="Backend Server Connection Settings"
            >
              <span
                className={`w-2 h-2 rounded-full ${
                  isConnected ? "bg-emerald-500" : "bg-rose-500"
                }`}
              />
              <span className="hidden sm:inline">
                {isChecking
                  ? "Checking Backend..."
                  : isConnected
                  ? "Backend Connected"
                  : "Backend Disconnected"}
              </span>
              <Server className="w-3.5 h-3.5 opacity-70" />
            </button>

            {/* Backend URL Config Popover */}
            {showConfig && (
              <div className="absolute right-0 mt-2 w-80 sm:w-96 p-4 rounded-2xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-2xl z-50 animate-in fade-in zoom-in-95 duration-150">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-xs font-bold text-slate-800 dark:text-white flex items-center gap-1.5">
                    <Server className="w-4 h-4 text-brand-500" />
                    <span>FastAPI Backend URL</span>
                  </span>
                  <button
                    onClick={() => checkConnection(apiUrl)}
                    className="text-xs text-brand-500 hover:underline flex items-center gap-1"
                  >
                    <RefreshCw className={`w-3 h-3 ${isChecking ? "animate-spin" : ""}`} />
                    <span>Test</span>
                  </button>
                </div>

                <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-3">
                  Enter your deployed Render URL (e.g. <code className="text-brand-500">https://fed-xray-api.onrender.com</code>) or local URL (<code className="text-brand-500">http://127.0.0.1:8000</code>).
                </p>

                <form onSubmit={handleSaveUrl} className="space-y-2">
                  <input
                    type="url"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="https://fed-xray-api.onrender.com"
                    className="w-full px-3 py-1.5 rounded-xl bg-slate-50 dark:bg-slate-950 border border-slate-300 dark:border-slate-700 text-xs text-slate-900 dark:text-white focus:outline-none focus:border-brand-500"
                    required
                  />
                  <div className="flex gap-2 justify-end">
                    <button
                      type="button"
                      onClick={() => setShowConfig(false)}
                      className="px-3 py-1 rounded-lg text-xs text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800"
                    >
                      Cancel
                    </button>
                    <button
                      type="submit"
                      className="px-3 py-1 rounded-lg bg-brand-600 hover:bg-brand-500 text-white font-semibold text-xs flex items-center gap-1"
                    >
                      <Check className="w-3 h-3" />
                      <span>Save & Connect</span>
                    </button>
                  </div>
                </form>
              </div>
            )}
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
