"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import {
  ShieldCheck,
  Lock,
  Activity,
  Server,
  ArrowRight,
  X,
  Building2,
  CheckCircle2,
  AlertTriangle,
  RefreshCw,
  Link2,
} from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { fetchEnclaveHandshake, fetchSystemState, fetchHospitalCohorts } from "@/lib/api";
import { EnclaveHandshakeResponse } from "@/types";

interface ConsortiumLaunchOverlayProps {
  isOpen: boolean;
  onClose: () => void;
}

export default function ConsortiumLaunchOverlay({
  isOpen,
  onClose,
}: ConsortiumLaunchOverlayProps) {
  const router = useRouter();
  const { t } = useLanguage();
  const { apiUrl, setApiUrl, autoDiscoverApi, isTunnelExpired } = useApi();
  const [attestationData, setAttestationData] = useState<EnclaveHandshakeResponse | null>(null);
  const [progress, setProgress] = useState(0);
  const [activeStage, setActiveStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [isVerifying, setIsVerifying] = useState(false);
  const [privacySummary, setPrivacySummary] = useState<string | null>(null);
  const [showUrlEditor, setShowUrlEditor] = useState(false);
  const [customUrlInput, setCustomUrlInput] = useState("");

  const runHandshake = useCallback(async (manualUrl?: string) => {
    if (!isOpen) return;
    setError(null);
    setIsVerifying(true);
    setProgress(15);
    setActiveStage(0);

    const candidates = [
      manualUrl,
      apiUrl,
      "",
      "http://127.0.0.1:8000",
      "http://localhost:8000",
    ].filter((u): u is string => typeof u === "string");

    let effectiveBase = (manualUrl || apiUrl || "").trim().replace(/\/+$/, "");

    try {
      // Step 1: Central Server Enclave Health Attestation across candidate endpoints
      let workingBase: string | null = null;
      let healthData: any = null;

      for (const cand of candidates) {
        try {
          const clean = cand.trim().replace(/\/+$/, "");
          const testUrl = clean ? `${clean}/api/health` : "/api/health";
          const controller = new AbortController();
          const tId = setTimeout(() => controller.abort(), 1200);
          const res = await fetch(testUrl, { signal: controller.signal, cache: "no-store" });
          clearTimeout(tId);
          if (res && res.ok) {
            const data = await res.json().catch(() => null);
            if (data && (data.status === "healthy" || data.status === "ok")) {
              workingBase = clean;
              healthData = data;
              break;
            }
          }
        } catch {
          // Probe next candidate
        }
      }

      if (workingBase !== null) {
        effectiveBase = workingBase;
        setApiUrl(workingBase);
      } else {
        // Fallback to auto-discovery if direct candidates failed
        const discovered = await autoDiscoverApi();
        if (discovered) {
          effectiveBase = discovered.trim().replace(/\/+$/, "");
          setApiUrl(effectiveBase);
          const res = await fetch(`${effectiveBase}/api/health`, { cache: "no-store" });
          if (res.ok) {
            healthData = await res.json();
          }
        }
      }

      if (!healthData || (healthData.status !== "healthy" && healthData.status !== "ok")) {
        throw new Error(
          `Consortium server unreachable at ${effectiveBase || "configured endpoints"}. Local backend offline or blocked.`
        );
      }

      await new Promise((r) => setTimeout(r, 260));
      setProgress(45);
      setActiveStage(1);

      // Step 2: Cryptographic Consortium Enclave Handshake (X25519 ECDH + CKKS + Patient DP)
      const handshake = await fetchEnclaveHandshake(effectiveBase);
      setAttestationData(handshake);

      try {
        if (typeof window !== "undefined") {
          window.sessionStorage.setItem("fedxray_attestation", JSON.stringify(handshake));
        }
      } catch {
        // Ignore session storage errors if restricted
      }

      await new Promise((r) => setTimeout(r, 300));
      setProgress(75);
      setActiveStage(2);

      // Step 3: Cryptographic Parameter & Patient-Level DP Handshake + Cohort Hydration Check
      const epsVal = typeof handshake.privacy.epsilon === "number" ? handshake.privacy.epsilon.toFixed(1) : "2.0";
      const cVal = typeof handshake.privacy.clipping_bound_C === "number" ? handshake.privacy.clipping_bound_C : "0.5";
      setPrivacySummary(`CKKS 128-Bit + Patient DP (eps <= ${epsVal}, C = ${cVal})`);

      try {
        const sysState = await fetchSystemState(effectiveBase);
        if (!sysState.cohorts || sysState.cohorts.length === 0) {
          await fetchHospitalCohorts(effectiveBase, 4, 40);
        }
      } catch {
        // Non-blocking: dashboard page also auto-hydrates
      }

      await new Promise((r) => setTimeout(r, 280));
      setProgress(100);
      setActiveStage(3);
      setIsVerifying(false);

      const timer = setTimeout(() => {
        router.push("/dashboard");
      }, 450);
      return () => clearTimeout(timer);
    } catch (err: any) {
      console.error("Enclave handshake failed:", err);
      setIsVerifying(false);
      setError(
        err?.message ||
        `Failed to establish secure enclave handshake with backend at ${effectiveBase}. Ensure local FastAPI backend is active or proceed in standby mode.`
      );
    }
  }, [isOpen, apiUrl, autoDiscoverApi, setApiUrl, router]);

  useEffect(() => {
    if (!isOpen) {
      setProgress(0);
      setActiveStage(0);
      setError(null);
      setIsVerifying(false);
      return;
    }

    runHandshake();
  }, [isOpen, runHandshake]);

  if (!isOpen) return null;

  const handleSkip = () => {
    router.push("/dashboard");
  };

  const steps = [
    {
      title: t("launch_modal_step1"),
      desc: t("launch_modal_step1_desc"),
      icon: Server,
    },
    {
      title: t("launch_modal_step2"),
      desc: t("launch_modal_step2_desc"),
      icon: Lock,
    },
    {
      title: t("launch_modal_step3"),
      desc: t("launch_modal_step3_desc"),
      icon: Activity,
    },
    {
      title: t("launch_modal_step4"),
      desc: t("launch_modal_step4_desc"),
      icon: ShieldCheck,
    },
  ];

  const node0 = attestationData?.nodes?.[0];
  const node1 = attestationData?.nodes?.[1];
  const node2 = attestationData?.nodes?.[2];
  const node3 = attestationData?.nodes?.[3];

  return (
    <div className="fixed inset-0 z-[99999] flex items-center justify-center p-3 sm:p-5 bg-black/80 backdrop-blur-md transition-all">
      <div className="relative w-full max-w-lg bg-[var(--bg-card)] border border-[var(--border-card)] rounded-2xl p-4 sm:p-6 shadow-2xl overflow-hidden select-none">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-3.5 right-3.5 sm:top-4 sm:right-4 p-1.5 rounded-lg text-slate-400 hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all z-30"
          aria-label="Close"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Modal Header */}
        <div className="flex items-center gap-2.5 sm:gap-3 mb-3 pr-6">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-500 shrink-0">
            <Activity className="w-4 h-4 sm:w-5 sm:h-5 animate-pulse" />
          </div>
          <div>
            <h3 className="font-display text-sm sm:text-base font-bold text-[var(--text-heading)] leading-snug">
              {t("launch_modal_title")}
            </h3>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)] leading-relaxed mt-0.5">
              {t("launch_modal_desc")}
            </p>
          </div>
        </div>

        {/* Dynamic Federated Learning Visual Enclave */}
        <div className="relative w-full h-[210px] sm:h-[230px] rounded-xl bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] overflow-hidden my-3 p-3 flex items-center justify-center">
          {/* Subtle Ambient Radial Glow */}
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(16,185,129,0.12)_0%,transparent_70%)] pointer-events-none" />

          {/* SVG Federated Tunnels & Active Data Pulses */}
          <svg className="absolute inset-0 w-full h-full pointer-events-none" xmlns="http://www.w3.org/2000/svg">
            <defs>
              <linearGradient id="flStream1" x1="15%" y1="15%" x2="50%" y2="50%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
              </linearGradient>
              <linearGradient id="flStream2" x1="85%" y1="15%" x2="50%" y2="50%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
              </linearGradient>
              <linearGradient id="flStream3" x1="15%" y1="85%" x2="50%" y2="50%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
              </linearGradient>
              <linearGradient id="flStream4" x1="85%" y1="85%" x2="50%" y2="50%">
                <stop offset="0%" stopColor="#10b981" stopOpacity="0.8" />
                <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.4" />
              </linearGradient>
            </defs>

            {/* Connecting Tunnels */}
            <line
              x1="22%" y1="22%" x2="50%" y2="50%"
              stroke="url(#flStream1)"
              strokeWidth={activeStage >= 1 ? "2.5" : "1.5"}
              strokeDasharray={activeStage >= 1 ? "6 4" : "4 4"}
              className={activeStage >= 1 ? "animate-pulse" : "opacity-40"}
            />
            <line
              x1="78%" y1="22%" x2="50%" y2="50%"
              stroke="url(#flStream2)"
              strokeWidth={activeStage >= 1 ? "2.5" : "1.5"}
              strokeDasharray={activeStage >= 1 ? "6 4" : "4 4"}
              className={activeStage >= 1 ? "animate-pulse" : "opacity-40"}
            />
            <line
              x1="22%" y1="78%" x2="50%" y2="50%"
              stroke="url(#flStream3)"
              strokeWidth={activeStage >= 1 ? "2.5" : "1.5"}
              strokeDasharray={activeStage >= 1 ? "6 4" : "4 4"}
              className={activeStage >= 1 ? "animate-pulse" : "opacity-40"}
            />
            <line
              x1="78%" y1="78%" x2="50%" y2="50%"
              stroke="url(#flStream4)"
              strokeWidth={activeStage >= 1 ? "2.5" : "1.5"}
              strokeDasharray={activeStage >= 1 ? "6 4" : "4 4"}
              className={activeStage >= 1 ? "animate-pulse" : "opacity-40"}
            />

            {/* Aggregation Rings */}
            <circle cx="50%" cy="50%" r="46" fill="none" stroke="#10b981" strokeWidth="1" strokeOpacity="0.25" strokeDasharray="3 3" />
            {activeStage >= 2 && (
              <circle cx="50%" cy="50%" r="62" fill="none" stroke="#06b6d4" strokeWidth="1.5" strokeOpacity="0.35" className="animate-ping" />
            )}
          </svg>

          {/* Node 1: Barcelona */}
          <div
            className={`absolute top-2.5 left-2.5 sm:top-3 sm:left-3 z-10 flex items-center gap-2 p-2 rounded-xl border transition-all ${
              activeStage === 0
                ? "bg-emerald-500/15 border-emerald-500 shadow-[0_0_14px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/40"
                : "bg-[var(--bg-card)] border-[var(--border-subtle)]"
            }`}
          >
            <div className="w-6 h-6 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-500 shrink-0">
              <Building2 className={`w-3.5 h-3.5 ${activeStage === 0 ? "animate-pulse" : ""}`} />
            </div>
            <div className="text-left min-w-0">
              <div className="font-mono text-[10px] font-bold text-[var(--text-heading)] leading-none truncate max-w-[85px]">
                {node0?.name ? node0.name.replace(" Skin Cancer Hub", "") : "BCN-20000"}
              </div>
              <div className="text-[8px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5 truncate max-w-[85px]">
                {node0?.peer_key_fingerprint ? node0.peer_key_fingerprint.replace("ecdh:x25519:", "pk:") : (activeStage === 0 ? "Local PEFT" : "Enc(Bk)")}
              </div>
            </div>
          </div>

          {/* Node 2: Boston */}
          <div
            className={`absolute top-2.5 right-2.5 sm:top-3 sm:right-3 z-10 flex items-center gap-2 p-2 rounded-xl border transition-all ${
              activeStage === 0
                ? "bg-emerald-500/15 border-emerald-500 shadow-[0_0_14px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/40"
                : "bg-[var(--bg-card)] border-[var(--border-subtle)]"
            }`}
          >
            <div className="text-right min-w-0">
              <div className="font-mono text-[10px] font-bold text-[var(--text-heading)] leading-none truncate max-w-[85px]">
                {node1?.name ? node1.name.replace(" Medical Center", "") : "Beth Israel"}
              </div>
              <div className="text-[8px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5 truncate max-w-[85px]">
                {node1?.peer_key_fingerprint ? node1.peer_key_fingerprint.replace("ecdh:x25519:", "pk:") : (activeStage === 0 ? "Local PEFT" : "Enc(Bk)")}
              </div>
            </div>
            <div className="w-6 h-6 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-500 shrink-0">
              <Building2 className={`w-3.5 h-3.5 ${activeStage === 0 ? "animate-pulse" : ""}`} />
            </div>
          </div>

          {/* Central Consortium Enclave Aggregator */}
          <div
            className={`z-20 flex flex-col items-center justify-center p-3 rounded-2xl border transition-all ${
              activeStage >= 1
                ? "bg-[var(--bg-card)] border-emerald-500 shadow-[0_0_24px_rgba(16,185,129,0.35)] ring-2 ring-emerald-500/30"
                : "bg-[var(--bg-card)] border-[var(--border-subtle)]"
            }`}
          >
            <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/30 flex items-center justify-center text-emerald-500 mb-1">
              {activeStage >= 2 ? (
                <ShieldCheck className="w-5 h-5 text-emerald-500 animate-pulse" />
              ) : activeStage === 1 ? (
                <Lock className="w-5 h-5 text-sky-500 animate-bounce" />
              ) : (
                <Server className="w-5 h-5 text-emerald-500" />
              )}
            </div>
            <div className="font-display text-[11px] sm:text-xs font-extrabold text-[var(--text-heading)] leading-tight text-center">
              Consortium Server
            </div>
            <div className="font-mono text-[8px] sm:text-[9px] text-emerald-600 dark:text-emerald-400 font-bold mt-0.5 truncate max-w-[130px]">
              {attestationData?.attestation_digest
                ? attestationData.attestation_digest.slice(0, 16) + "..."
                : activeStage >= 2
                ? "Consensus Deployed"
                : activeStage === 1
                ? "CKKS 128-Bit Aggregation"
                : "Enclave Sync"}
            </div>
          </div>

          {/* Node 3: Vienna */}
          <div
            className={`absolute bottom-2.5 left-2.5 sm:bottom-3 sm:left-3 z-10 flex items-center gap-2 p-2 rounded-xl border transition-all ${
              activeStage === 0
                ? "bg-emerald-500/15 border-emerald-500 shadow-[0_0_14px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/40"
                : "bg-[var(--bg-card)] border-[var(--border-subtle)]"
            }`}
          >
            <div className="w-6 h-6 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-500 shrink-0">
              <Building2 className={`w-3.5 h-3.5 ${activeStage === 0 ? "animate-pulse" : ""}`} />
            </div>
            <div className="text-left min-w-0">
              <div className="font-mono text-[10px] font-bold text-[var(--text-heading)] leading-none truncate max-w-[85px]">
                {node2?.name ? node2.name.replace(" Institute", "") : "ViDIR Pathology"}
              </div>
              <div className="text-[8px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5 truncate max-w-[85px]">
                {node2?.peer_key_fingerprint ? node2.peer_key_fingerprint.replace("ecdh:x25519:", "pk:") : (activeStage === 0 ? "Local PEFT" : "Enc(Bk)")}
              </div>
            </div>
          </div>

          {/* Node 4: Queensland */}
          <div
            className={`absolute bottom-2.5 right-2.5 sm:bottom-3 sm:right-3 z-10 flex items-center gap-2 p-2 rounded-xl border transition-all ${
              activeStage === 0
                ? "bg-emerald-500/15 border-emerald-500 shadow-[0_0_14px_rgba(16,185,129,0.3)] ring-1 ring-emerald-500/40"
                : "bg-[var(--bg-card)] border-[var(--border-subtle)]"
            }`}
          >
            <div className="text-right min-w-0">
              <div className="font-mono text-[10px] font-bold text-[var(--text-heading)] leading-none truncate max-w-[85px]">
                {node3?.name ? node3.name.replace(" Screening Center", "") : "Queensland"}
              </div>
              <div className="text-[8px] font-mono text-emerald-600 dark:text-emerald-400 font-semibold mt-0.5 truncate max-w-[85px]">
                {node3?.peer_key_fingerprint ? node3.peer_key_fingerprint.replace("ecdh:x25519:", "pk:") : (activeStage === 0 ? "Local PEFT" : "Enc(Bk)")}
              </div>
            </div>
            <div className="w-6 h-6 rounded-lg bg-emerald-500/10 flex items-center justify-center text-emerald-500 shrink-0">
              <Building2 className={`w-3.5 h-3.5 ${activeStage === 0 ? "animate-pulse" : ""}`} />
            </div>
          </div>
        </div>

        {/* Live FL Stage Banner & Progress Bar */}
        <div className="mb-3.5 p-2.5 rounded-xl bg-[var(--bg-card-inner)] border border-emerald-500/30">
          <div className="flex items-center justify-between text-xs mb-2">
            <div className="flex items-center gap-2 min-w-0 pr-2">
              <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse shrink-0" />
              <span className="font-semibold text-[var(--text-heading)] text-[11px] sm:text-xs truncate">
                {activeStage === 0 && t("launch_fl_phase1")}
                {activeStage === 1 && t("launch_fl_phase2")}
                {activeStage >= 2 && t("launch_fl_phase3")}
              </span>
            </div>
            <span className="font-mono text-[11px] font-bold text-emerald-600 dark:text-emerald-400 shrink-0">
              {progress}%
            </span>
          </div>

          {/* Progress Track */}
          <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-white/[0.08] overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-100 ease-out rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Error Alert if Enclave Verification Fails (Honest Failure Notification) */}
        {error && (
          <div className="mb-3.5 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-left">
            <div className="flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-rose-500 shrink-0 mt-0.5" />
              <div className="flex-1">
                <div className="text-xs font-bold text-rose-600 dark:text-rose-400">
                  Enclave Verification Handshake Failed
                </div>
                <div className="text-[11px] text-[var(--text-muted)] mt-0.5 leading-relaxed font-mono">
                  {error}
                </div>

                {isTunnelExpired && (
                  <div className="mt-2 p-1.5 rounded-md bg-amber-500/10 border border-amber-500/30 text-amber-500 dark:text-amber-400 text-[11px] font-sans flex items-center gap-1.5">
                    <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
                    <span>Configured cloud tunnel has expired. Switched to local backend (http://127.0.0.1:8000).</span>
                  </div>
                )}

                {/* Quick Action Buttons */}
                <div className="flex flex-wrap items-center gap-1.5 mt-2.5">
                  <button
                    onClick={handleSkip}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[11px] font-bold bg-emerald-600 text-white hover:bg-emerald-500 shadow-sm transition-all"
                  >
                    <span>Proceed to Dashboard (Standby Mode)</span>
                    <ArrowRight className="w-3.5 h-3.5" />
                  </button>

                  <button
                    onClick={() => runHandshake()}
                    disabled={isVerifying}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-md text-[11px] font-semibold bg-rose-500/80 text-white hover:bg-rose-500 transition-colors disabled:opacity-50"
                  >
                    <RefreshCw className={`w-3 h-3 ${isVerifying ? "animate-spin" : ""}`} />
                    <span>Retry Handshake</span>
                  </button>

                  <button
                    onClick={() => {
                      setApiUrl("http://127.0.0.1:8000");
                      runHandshake("http://127.0.0.1:8000");
                    }}
                    disabled={isVerifying}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-semibold bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 transition-colors disabled:opacity-50"
                  >
                    <span>Connect Localhost (8000)</span>
                  </button>

                  <button
                    onClick={() => setShowUrlEditor(!showUrlEditor)}
                    className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-[11px] font-semibold bg-slate-800/60 hover:bg-slate-700 text-slate-300 transition-colors"
                  >
                    <Link2 className="w-3 h-3" />
                    <span>Change API URL</span>
                  </button>
                </div>

                {/* Optional Custom Server URL Input */}
                {showUrlEditor && (
                  <div className="mt-2.5 pt-2 border-t border-rose-500/20 flex items-center gap-1.5">
                    <input
                      type="text"
                      placeholder="e.g. http://127.0.0.1:8000 or https://tunnel.trycloudflare.com"
                      value={customUrlInput}
                      onChange={(e) => setCustomUrlInput(e.target.value)}
                      className="flex-1 px-2 py-1 rounded-md bg-black/40 border border-rose-500/30 text-[11px] font-mono text-white placeholder:text-slate-500 focus:outline-hidden focus:border-emerald-500"
                    />
                    <button
                      onClick={() => {
                        if (customUrlInput.trim()) {
                          const clean = customUrlInput.trim().replace(/\/+$/, "");
                          setApiUrl(clean);
                          runHandshake(clean);
                        }
                      }}
                      className="px-2.5 py-1 rounded-md bg-emerald-600 text-white text-[11px] font-semibold hover:bg-emerald-700 transition-colors"
                    >
                      Connect
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {/* Verified Electronic Attestation Digest Badge */}
        {attestationData?.attestation_digest && (
          <div className="mb-3 px-2.5 py-1.5 rounded-lg bg-emerald-500/[0.08] border border-emerald-500/30 flex items-center justify-between text-[10px] font-mono">
            <div className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-semibold truncate">
              <ShieldCheck className="w-3.5 h-3.5 shrink-0 text-emerald-500" />
              <span className="truncate">Attestation: {attestationData.attestation_digest}</span>
            </div>
            <span className="text-[9px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-600 dark:text-emerald-400 font-bold shrink-0">
              ATTESTED
            </span>
          </div>
        )}

        {/* Compact 4-Step Handshake Milestones (Clean grid instead of tall walls of text) */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 mb-4">
          {steps.map((step, idx) => {
            const isCompleted = activeStage > idx || progress === 100;
            const isCurrent = activeStage === idx && progress < 100 && !error;

            return (
              <div
                key={idx}
                className={`flex items-center gap-2 p-2 rounded-lg border text-left transition-all ${
                  isCompleted
                    ? "bg-emerald-500/[0.08] border-emerald-500/30 text-[var(--text-heading)]"
                    : isCurrent
                    ? "bg-[var(--bg-card-inner)] border-emerald-500 text-[var(--text-heading)] shadow-sm ring-1 ring-emerald-500/20"
                    : "bg-transparent border-[var(--border-subtle)] text-[var(--text-muted)] opacity-50"
                }`}
              >
                <div
                  className={`w-4 h-4 rounded-full flex items-center justify-center shrink-0 text-[9px] font-mono font-bold ${
                    isCompleted
                      ? "bg-emerald-500 text-white"
                      : isCurrent
                      ? "bg-emerald-500/20 text-emerald-500 animate-pulse"
                      : "bg-slate-200 dark:bg-white/[0.08] text-slate-400"
                  }`}
                >
                  {isCompleted ? <CheckCircle2 className="w-3 h-3" /> : idx + 1}
                </div>
                <div className="text-[10px] sm:text-[11px] font-semibold truncate leading-tight flex-1" title={step.title}>
                  {step.title}
                </div>
              </div>
            );
          })}
        </div>

        {/* Hidden screen-reader accessible descriptions */}
        <div className="sr-only">
          <span>{t("launch_modal_handshake")}</span>
          {steps.map((s, i) => (
            <p key={i}>{s.desc}</p>
          ))}
        </div>

        {/* Footer Actions */}
        <div className="flex flex-wrap items-center justify-between gap-2 pt-2.5 border-t border-[var(--border-subtle)]">
          <span className="text-[10px] sm:text-[11px] text-[var(--text-muted)] font-mono">
            {privacySummary || t("launch_modal_footer_nodes")}
          </span>
          <button
            onClick={handleSkip}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold text-emerald-600 dark:text-emerald-400 hover:bg-emerald-500/10 transition-colors"
          >
            <span>{t("launch_modal_skip")}</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>
    </div>
  );
}
