"use client";

import React, { useEffect, useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Server, Sun, Moon } from "lucide-react";

export default function Navbar() {
  const { language, setLanguage } = useLanguage();
  const { apiUrl, setApiUrl, isConnected } = useApi();
  const [isDark, setIsDark] = useState(true);
  const [showConfig, setShowConfig] = useState(false);
  const [inputUrl, setInputUrl] = useState(apiUrl);

  useEffect(() => {
    // Sync initial theme
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

        {/* Logo */}
        <div className="flex items-center gap-2.5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-accent-600 to-accent-400 flex items-center justify-center text-white text-sm font-bold shadow-md">
            🫁
          </div>
          <span className="font-display font-extrabold text-[17px] tracking-tight text-[var(--text-heading)]">
            Fed-XRay
          </span>
          <span className="hidden sm:inline text-[10px] font-bold tracking-wider uppercase px-2 py-0.5 rounded-md bg-accent-500/10 text-accent-600 dark:text-accent-400 border border-accent-500/20">
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
                  ? "bg-accent-500/10 border-accent-500/20 text-accent-600 dark:text-accent-400"
                  : "bg-red-500/10 border-red-500/20 text-red-600 dark:text-red-400 animate-pulse"
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${isConnected ? "bg-accent-500" : "bg-red-500"}`} />
              <span className="hidden sm:inline">{isConnected ? "API Live" : "API Offline"}</span>
              <Server className="w-3 h-3 opacity-60" />
            </button>

            {showConfig && (
              <div className="absolute right-0 mt-2 w-80 p-4 card z-50 shadow-xl animate-fade-in">
                <p className="text-[11px] text-slate-500 dark:text-slate-400 mb-2">FastAPI Backend URL</p>
                <form onSubmit={handleSaveUrl} className="flex gap-2">
                  <input
                    type="url"
                    value={inputUrl}
                    onChange={(e) => setInputUrl(e.target.value)}
                    placeholder="https://fed-xray-api.onrender.com"
                    className="flex-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] text-[11px] text-[var(--text-heading)] focus:outline-none focus:border-accent-500/50"
                  />
                  <button type="submit" className="px-3 py-1.5 rounded-lg bg-accent-600 text-white text-[11px] font-semibold">
                    Save
                  </button>
                </form>
              </div>
            )}
          </div>

          {/* Theme Toggle Button */}
          <button
            onClick={toggleTheme}
            aria-label="Toggle Theme"
            className="p-2 rounded-lg border border-[var(--border-card)] text-slate-600 dark:text-slate-400 hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all"
          >
            {isDark ? <Sun className="w-4 h-4 text-amber-400" /> : <Moon className="w-4 h-4 text-slate-700" />}
          </button>

          {/* Language Switcher */}
          <div className="flex items-center rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-card)] p-0.5">
            <button
              onClick={() => setLanguage("EN")}
              className={`px-2 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "EN" ? "bg-accent-600 text-white" : "text-slate-500 hover:text-[var(--text-heading)]"
              }`}
            >
              EN
            </button>
            <button
              onClick={() => setLanguage("TR")}
              className={`px-2 py-1 rounded-md text-[11px] font-bold transition-all ${
                language === "TR" ? "bg-accent-600 text-white" : "text-slate-500 hover:text-[var(--text-heading)]"
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
