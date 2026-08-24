"use client";

import React, { createContext, useContext, useState, useEffect } from "react";

interface ApiContextType {
  apiUrl: string;
  setApiUrl: (url: string) => void;
  isConnected: boolean;
  isChecking: boolean;
  checkConnection: (url?: string) => Promise<boolean>;
  lastError: string | null;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export function ApiProvider({ children }: { children: React.ReactNode }) {
  const [apiUrl, setApiUrlState] = useState<string>("http://127.0.0.1:8000");
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [lastError, setLastError] = useState<string | null>(null);

  useEffect(() => {
    const saved = localStorage.getItem("fedxray_api_url");
    const envUrl = process.env.NEXT_PUBLIC_API_URL;
    const initial = saved || envUrl || "http://127.0.0.1:8000";
    setApiUrlState(initial.replace(/\/$/, ""));
    checkConnection(initial.replace(/\/$/, ""));
  }, []);

  const setApiUrl = (url: string) => {
    const cleanUrl = url.trim().replace(/\/$/, "");
    setApiUrlState(cleanUrl);
    localStorage.setItem("fedxray_api_url", cleanUrl);
    checkConnection(cleanUrl);
  };

  const checkConnection = async (testUrl?: string): Promise<boolean> => {
    const target = (testUrl || apiUrl).replace(/\/$/, "");
    setIsChecking(true);
    setLastError(null);
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 6000);
      const res = await fetch(`${target}/api/health`, { signal: controller.signal });
      clearTimeout(timeoutId);
      if (res.ok) {
        setIsConnected(true);
        setIsChecking(false);
        return true;
      } else {
        setIsConnected(false);
        setLastError(`Backend HTTP ${res.status}`);
        setIsChecking(false);
        return false;
      }
    } catch (err: any) {
      setIsConnected(false);
      setLastError(err.message || "Connection refused");
      setIsChecking(false);
      return false;
    }
  };

  return (
    <ApiContext.Provider
      value={{
        apiUrl,
        setApiUrl,
        isConnected,
        isChecking,
        checkConnection,
        lastError,
      }}
    >
      {children}
    </ApiContext.Provider>
  );
}

export function useApi() {
  const context = useContext(ApiContext);
  if (!context) {
    throw new Error("useApi must be used within an ApiProvider");
  }
  return context;
}
