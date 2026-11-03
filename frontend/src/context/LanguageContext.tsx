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
    let sessionLang: Language | null = null;
    try {
      if (typeof window !== "undefined") {
        sessionLang = sessionStorage.getItem("fedxray_lang") as Language;
      }
    } catch {
      // Ignore storage errors
    }

    if (sessionLang && (sessionLang === "EN" || sessionLang === "TR")) {
      setLanguage(sessionLang);
    } else {
      // System always defaults to English (EN) on initial startup
      setLanguage("EN");
      try {
        localStorage.setItem("fedxray_lang", "EN");
      } catch {
        // Ignore storage errors
      }
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
