"use client";

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from "react";

interface ApiContextType {
  apiUrl: string;
  setApiUrl: (url: string) => void;
  isConnected: boolean;
  isChecking: boolean;
  checkConnection: (url?: string) => Promise<boolean>;
  autoDiscoverApi: () => Promise<string | null>;
  lastError: string | null;
  reconnectAttempts: number;
}

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export function ApiProvider({ children }: { children: React.ReactNode }) {
  const defaultUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "") || "http://127.0.0.1:8000";
  const [apiUrl, setApiUrlState] = useState<string>(defaultUrl);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [lastError, setLastError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);

  const apiUrlRef = useRef<string>(apiUrl);
  apiUrlRef.current = apiUrl;
  const isConnectedRef = useRef<boolean>(isConnected);
  isConnectedRef.current = isConnected;

  // Build ordered candidate URLs based on environment and network configuration
  const getCandidateUrls = useCallback((): string[] => {
    const candidates: string[] = [];

    // 1. Next.js Public Environment Variable (authoritative configuration)
    if (process.env.NEXT_PUBLIC_API_URL) {
      candidates.push(process.env.NEXT_PUBLIC_API_URL.trim().replace(/\/$/, ""));
    }

    // 2. Saved preference in localStorage
    if (typeof window !== "undefined") {
      try {
        const saved = localStorage.getItem("fedxray_api_url");
        if (saved && saved.trim()) {
          candidates.push(saved.trim().replace(/\/$/, ""));
        }
      } catch {
        // Ignore localStorage access restrictions
      }
    }

    // 3. Loopback IP and Localhost standards
    candidates.push("http://127.0.0.1:8000");
    candidates.push("http://localhost:8000");

    // 4. Dynamic hostname if running on LAN / local network IP
    if (typeof window !== "undefined" && window.location?.hostname) {
      const host = window.location.hostname;
      if (host !== "localhost" && host !== "127.0.0.1") {
        candidates.push(`http://${host}:8000`);
      }
    }

    // 5. Relative URL (leveraging Next.js rewrites proxy: /api/* -> backend)
    candidates.push("");

    return Array.from(new Set(candidates));
  }, []);

  // Ping a single endpoint with strict timeout
  const pingEndpoint = async (targetUrl: string, timeoutMs = 3000): Promise<boolean> => {
    try {
      const clean = targetUrl.replace(/\/$/, "");
      const fullUrl = clean ? `${clean}/api/health` : "/api/health";
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      const res = await fetch(fullUrl, {
        method: "GET",
        signal: controller.signal,
        headers: { Accept: "application/json" },
      });
      clearTimeout(timer);

      if (res.ok) {
        const data = await res.json().catch(() => null);
        return data && (data.status === "healthy" || data.status === "ok" || res.status === 200);
      }
      return false;
    } catch {
      return false;
    }
  };

  // Check connection to a specific URL (or the current apiUrl)
  const checkConnection = useCallback(async (testUrl?: string): Promise<boolean> => {
    const target = (testUrl !== undefined ? testUrl : apiUrlRef.current).trim().replace(/\/$/, "");
    setIsChecking(true);
    setLastError(null);

    const isHealthy = await pingEndpoint(target, 3500);
    if (isHealthy) {
      setIsConnected(true);
      setIsChecking(false);
      setLastError(null);
      setReconnectAttempts(0);
      return true;
    } else {
      setIsConnected(false);
      setIsChecking(false);
      setLastError("Connection failed or backend offline");
      return false;
    }
  }, []);

  // Auto-discover the first reachable candidate
  const autoDiscoverApi = useCallback(async (): Promise<string | null> => {
    const candidates = getCandidateUrls();
    setIsChecking(true);

    const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "");
    const saved = typeof window !== "undefined" ? localStorage.getItem("fedxray_api_url") : null;
    const prioritized = envUrl
      ? [envUrl, ...candidates.filter((c) => c !== envUrl)]
      : saved
      ? [saved.trim().replace(/\/$/, ""), ...candidates.filter((c) => c !== saved.trim().replace(/\/$/, ""))]
      : candidates;

    for (const cand of prioritized) {
      const ok = await pingEndpoint(cand, 2500);
      if (ok) {
        setApiUrlState(cand);
        if (typeof window !== "undefined") {
          try {
            localStorage.setItem("fedxray_api_url", cand);
          } catch {
            // Ignore storage quota or access errors
          }
        }
        setIsConnected(true);
        setIsChecking(false);
        setLastError(null);
        setReconnectAttempts(0);
        return cand;
      }
    }

    setIsConnected(false);
    setIsChecking(false);
    setLastError("Unable to reach FastAPI backend on any local or candidate port");
    return null;
  }, [getCandidateUrls]);

  // Set URL manually (e.g. from UI modal)
  const setApiUrl = (url: string) => {
    const cleanUrl = url.trim().replace(/\/$/, "");
    setApiUrlState(cleanUrl);
    if (typeof window !== "undefined") {
      try {
        localStorage.setItem("fedxray_api_url", cleanUrl);
      } catch {
        // Ignore storage quota errors
      }
    }
    checkConnection(cleanUrl);
  };

  // Initial startup: progressive discovery with rapid retries for cold start / reload
  useEffect(() => {
    let isMounted = true;

    const startupRoutine = async () => {
      // 1. Immediate attempt
      const found = await autoDiscoverApi();
      if (found || !isMounted) return;

      // 2. Retry at 1s if initial attempt failed (backend starting up)
      await new Promise((r) => setTimeout(r, 1000));
      if (!isMounted) return;
      const found2 = await autoDiscoverApi();
      if (found2 || !isMounted) return;

      // 3. Final rapid retry at 2.5s
      await new Promise((r) => setTimeout(r, 1500));
      if (!isMounted) return;
      await autoDiscoverApi();
    };

    startupRoutine();

    return () => {
      isMounted = false;
    };
  }, [autoDiscoverApi]);

  // Background Heartbeat & Auto-Reconnect Loop
  useEffect(() => {
    const intervalMs = isConnected ? 12000 : 3500;

    const timer = setInterval(async () => {
      if (!isConnectedRef.current) {
        setReconnectAttempts((prev) => prev + 1);
        await autoDiscoverApi();
      } else {
        const healthy = await pingEndpoint(apiUrlRef.current, 3000);
        if (!healthy) {
          setIsConnected(false);
          await autoDiscoverApi();
        }
      }
    }, intervalMs);

    return () => clearInterval(timer);
  }, [isConnected, autoDiscoverApi]);

  // Re-verify on window focus and network online events
  useEffect(() => {
    const handleRecheck = () => {
      if (!isConnectedRef.current) {
        autoDiscoverApi();
      } else {
        checkConnection();
      }
    };

    window.addEventListener("focus", handleRecheck);
    window.addEventListener("online", handleRecheck);

    return () => {
      window.removeEventListener("focus", handleRecheck);
      window.removeEventListener("online", handleRecheck);
    };
  }, [autoDiscoverApi, checkConnection]);

  return (
    <ApiContext.Provider
      value={{
        apiUrl,
        setApiUrl,
        isConnected,
        isChecking,
        checkConnection,
        autoDiscoverApi,
        lastError,
        reconnectAttempts,
      }}
    >
      {children}
    </ApiContext.Provider>
  );
}

export function useApi(): ApiContextType {
  const context = useContext(ApiContext);
  if (!context) {
    // Provide safe resilient default fallback if rendered outside ApiProvider (e.g. standalone tests)
    return {
      apiUrl: process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "") || "http://127.0.0.1:8000",
      setApiUrl: () => {},
      isConnected: false,
      isChecking: false,
      checkConnection: async () => false,
      autoDiscoverApi: async () => null,
      lastError: null,
      reconnectAttempts: 0,
    };
  }
  return context;
}
