"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { Language, TRANSLATIONS } from "@/lib/translations";

interface LanguageContextType {
  language: Language;
  setLanguage: (lang: Language) => void;
  t: (key: string) => string;
}

const LanguageContext = createContext<LanguageContextType | undefined>(undefined);

export function LanguageProvider({ children }: { children: React.ReactNode }) {
  const [language, setLanguage] = useState<Language>("EN");

  useEffect(() => {
    let savedLang: string | null = null;
    try {
      if (typeof window !== "undefined") {
        savedLang = sessionStorage.getItem("fedxray_lang") || localStorage.getItem("fedxray_lang");
      }
    } catch {
      // Ignore storage errors
    }

    if (savedLang === "TR" || savedLang === "EN") {
      setLanguage(savedLang as Language);
    } else {
      setLanguage("EN");
    }
  }, []);

  const handleSetLanguage = (lang: Language) => {
    setLanguage(lang);
    try {
      sessionStorage.setItem("fedxray_lang", lang);
      localStorage.setItem("fedxray_lang", lang);
    } catch {
      // Ignore storage errors
    }
  };

  const t = (key: string): string => {
    const currentDict = TRANSLATIONS[language];
    return currentDict[key] || TRANSLATIONS.EN[key] || key;
  };

  return (
    <LanguageContext.Provider value={{ language, setLanguage: handleSetLanguage, t }}>
      {children}
    </LanguageContext.Provider>
  );
}

export function useLanguage() {
  const context = useContext(LanguageContext);
  if (!context) {
    throw new Error("useLanguage must be used within a LanguageProvider");
  }
  return context;
}
