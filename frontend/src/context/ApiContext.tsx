"use client";

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from "react";

export interface ApiContextType {
  apiUrl: string;
  setApiUrl: (url: string) => void;
  isConnected: boolean;
  isChecking: boolean;
  checkConnection: (url?: string) => Promise<boolean>;
  autoDiscoverApi: () => Promise<string | null>;
  lastError: string | null;
  reconnectAttempts: number;
  isTunnelExpired: boolean;
}

export const isEphemeralTunnel = (url: string): boolean => {
  try {
    if (!url) return false;
    const lower = url.toLowerCase();
    return (
      lower.includes(".trycloudflare.com") ||
      lower.includes(".loca.lt") ||
      lower.includes(".ngrok-free.app") ||
      lower.includes(".serveo.net") ||
      lower.includes("globe-celebration-poems-shakira")
    );
  } catch {
    return false;
  }
};

const ApiContext = createContext<ApiContextType | undefined>(undefined);

export function ApiProvider({ children }: { children: React.ReactNode }) {
  const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "") || "";
  // Do not initialize with known dead or unresolvable tunnel URLs
  const initialUrl = isEphemeralTunnel(envUrl) ? "http://127.0.0.1:8000" : envUrl || "http://127.0.0.1:8000";

  const [apiUrl, setApiUrlState] = useState<string>(initialUrl);
  const [isConnected, setIsConnected] = useState<boolean>(false);
  const [isChecking, setIsChecking] = useState<boolean>(true);
  const [lastError, setLastError] = useState<string | null>(null);
  const [reconnectAttempts, setReconnectAttempts] = useState<number>(0);
  const [isTunnelExpired, setIsTunnelExpired] = useState<boolean>(false);

  const apiUrlRef = useRef<string>(apiUrl);
  apiUrlRef.current = apiUrl;
  const isConnectedRef = useRef<boolean>(isConnected);
  isConnectedRef.current = isConnected;
  const deadTunnelsRef = useRef<Set<string>>(new Set());

  // Build ordered candidate URLs based on environment and network configuration
  const getCandidateUrls = useCallback((): string[] => {
    const candidates: string[] = [];

    // 1. Saved preference in localStorage (if not in dead tunnels)
    if (typeof window !== "undefined") {
      try {
        const saved = localStorage.getItem("fedxray_api_url")?.trim().replace(/\/$/, "");
        if (saved && !deadTunnelsRef.current.has(saved)) {
          candidates.push(saved);
        }
      } catch {
        // Ignore localStorage access restrictions
      }
    }

    // 2. Next.js Public Environment Variable (authoritative configuration)
    if (process.env.NEXT_PUBLIC_API_URL) {
      const env = process.env.NEXT_PUBLIC_API_URL.trim().replace(/\/$/, "");
      if (env && !deadTunnelsRef.current.has(env)) {
        candidates.push(env);
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

    return Array.from(new Set(candidates.filter(Boolean)));
  }, []);

  // Ping a single endpoint with strict timeout and clean failure handling
  const pingEndpoint = async (targetUrl: string, timeoutMs = 2500): Promise<boolean> => {
    try {
      const clean = targetUrl.replace(/\/+$/, "");
      const fullUrl = clean ? `${clean}/api/health` : "/api/health";
      const controller = new AbortController();
      const timer = setTimeout(() => controller.abort(), timeoutMs);

      const res = await fetch(fullUrl, {
        method: "GET",
        signal: controller.signal,
        headers: { Accept: "application/json" },
        cache: "no-store",
      }).catch(() => null);
      clearTimeout(timer);

      if (res && res.ok) {
        const data = await res.json().catch(() => null);
        return Boolean(data && (data.status === "healthy" || data.status === "ok" || res.status === 200));
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
      setIsTunnelExpired(false);
      return true;
    } else {
      setIsConnected(false);
      setIsChecking(false);
      setLastError("Connection failed or backend offline");
      if (isEphemeralTunnel(target)) {
        deadTunnelsRef.current.add(target);
        setIsTunnelExpired(true);
      }
      return false;
    }
  }, []);

  // Auto-discover the first reachable candidate
  const autoDiscoverApi = useCallback(async (): Promise<string | null> => {
    const candidates = getCandidateUrls();
    setIsChecking(true);

    const envUrlClean = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/+$/, "");
    const saved = typeof window !== "undefined" ? localStorage.getItem("fedxray_api_url")?.trim().replace(/\/+$/, "") : null;
    const prioritized = envUrlClean && !deadTunnelsRef.current.has(envUrlClean)
      ? [envUrlClean, ...candidates.filter((c) => c !== envUrlClean)]
      : saved && !deadTunnelsRef.current.has(saved)
      ? [saved, ...candidates.filter((c) => c !== saved)]
      : candidates;

    for (const cand of prioritized) {
      if (deadTunnelsRef.current.has(cand)) continue;
      const ok = await pingEndpoint(cand, 2000);
      if (ok) {
        setApiUrlState(cand);
        setIsTunnelExpired(false);
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
      } else {
        // If an ephemeral tunnel domain failed, blacklist it and prune from localStorage
        if (isEphemeralTunnel(cand)) {
          deadTunnelsRef.current.add(cand);
          setIsTunnelExpired(true);
          if (typeof window !== "undefined") {
            try {
              localStorage.removeItem("fedxray_api_url");
            } catch {
              // Ignore storage error
            }
          }
        }
      }
    }

    // Crucial fallback: ensure apiUrl does not remain set to a dead third-party domain
    setApiUrlState("http://127.0.0.1:8000");
    setIsConnected(false);
    setIsChecking(false);
    setLastError("Unable to reach FastAPI backend on candidate ports");
    return null;
  }, [getCandidateUrls]);

  // Set URL manually (e.g. from UI modal)
  const setApiUrl = (url: string) => {
    const cleanUrl = url.trim().replace(/\/$/, "");
    deadTunnelsRef.current.delete(cleanUrl);
    setIsTunnelExpired(false);
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

  // Background Heartbeat & Auto-Reconnect Loop with Exponential Backoff
  useEffect(() => {
    // When offline, stop polling after 3 consecutive attempts to avoid flooding network console
    if (!isConnected && reconnectAttempts >= 3) {
      return;
    }

    // Exponential backoff: 5s -> 10s -> 20s when offline; 20s when connected
    const delayMs = isConnected
      ? 20000
      : Math.min(25000, Math.round(5000 * Math.pow(2.0, Math.min(reconnectAttempts, 2))));

    const timer = setTimeout(async () => {
      if (!isConnectedRef.current) {
        setReconnectAttempts((prev) => prev + 1);
        await autoDiscoverApi();
      } else {
        const healthy = await pingEndpoint(apiUrlRef.current, 2500);
        if (!healthy) {
          setIsConnected(false);
          await autoDiscoverApi();
        }
      }
    }, delayMs);

    return () => clearTimeout(timer);
  }, [isConnected, reconnectAttempts, autoDiscoverApi]);

  // Re-verify on window focus and network online events (throttled to max once per 15s)
  useEffect(() => {
    let lastFocusCheck = 0;
    const handleRecheck = () => {
      const now = Date.now();
      if (now - lastFocusCheck < 15000) return;
      lastFocusCheck = now;

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
        isTunnelExpired,
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
      isTunnelExpired: false,
    };
  }
  return context;
}
