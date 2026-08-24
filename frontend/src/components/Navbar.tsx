"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Server, Sun, Moon, Check, RefreshCw } from "lucide-react";

export default function Navbar() {
  const { language, setLanguage, t } = useLanguage();
  const { apiUrl, setApiUrl, isConnected, isChecking, checkConnection } = useApi();
  const [isDark, setIsDark] = React.useState(true);
  const [showConfig, setShowConfig] = React.useState(false);
  const [inputUrl, setInputUrl] = React.useState(apiUrl);

  const toggleTheme = () => {
    setIsDark(!isDark);
    document.documentElement.classList.toggle("dark", !isDark);
  };

  const handleSaveUrl = (e: React.FormEvent) => {
    e.preventDefault();
    setApiUrl(inputUrl);
    setShowConfig(false);
  };

  return (
    <header className="sticky top-0 z-50 w-full h-14 bg-navy-900/90 backdrop-blur-xl border-b border-white/[0.04]">
      <div className="max-w-[1360px] mx-auto px-5 h-full flex items-center justify-between">

        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-600 to-accent-400 flex items-center justify-center text-white text-sm font-bold shadow-md">
            🫁
          </div>
          <span className="font-display font-extrabold text-[17px] tracking-tight text-white">
            Fed-XRay
          </span>
          <span className="hidden sm:inline text-[10px] font-semibold tracking-widest uppercase px-2 py-0.5 rounded-md bg-accent-500/10 text-accent-400 border border-accent-500/20">
            CDSS v2.0
          </span>
        </div>

        {/* Right Controls */}
        <div className="flex items-center gap-2">

          {/* Backend Connection Pill */}
          <div className="relative">
            <button
              onClick={() => { setInputUrl(apiUrl); setShowConfig(!showConfig); }}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-semibold border transition-all ${
                isConnected
                  ? "bg-accent-500/10 border-accent-500/20 text-accent-400"
                  : "bg-red-500/10 border-red-500/20 text-red-400 animate-pulse"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-accent-400" : "bg-red-400"}`} />
              <span className="hidden sm:inline">{isConnected ? "API Live" : "API Offline"}</span>
              <Server className="w-3 h-3 opacity-60" />
            </button>

            {showConfig && (
              <div className="absolute right-0 mt-2 w-80 p-4 card z-50 shadow-xl animate-fade-in">
                <p className="text-[11px] text-slate-400 mb-2">FastAPI Backend URL</p>
                <form onSubmit={handleSaveUrl} className="flex gap-2">
                  <input
                    type="url"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="https://fed-xray-api.onrender.com"
                    className="flex-1 px-2.5 py-1.5 rounded-lg bg-navy-950 border border-white/[0.06] text-[11px] text-white focus:outline-none focus:border-accent-500/40"
                  />
                  <button type="submit" className="px-3 py-1.5 rounded-lg bg-accent-600 text-white text-[11px] font-semibold">
                    Save
                  </button>
                </form>
              </div>
            )}
          </div>

          {/* Theme Toggle */}
          <button
            onClick={toggleTheme}
            className="p-1.5 rounded-lg border border-white/[0.06] text-slate-400 hover:text-white transition-colors"
          >
            {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
          </button>

          {/* Language Switcher */}
          <div className="flex items-center rounded-lg bg-navy-800 border border-white/[0.04] p-0.5">
            <button
              onClick={() => setLanguage("EN")}
              className={`px-2 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "EN" ? "bg-accent-600 text-white" : "text-slate-500 hover:text-white"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage("TR")}
              className={`px-2 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "TR" ? "bg-accent-600 text-white" : "text-slate-500 hover:text-white"
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
