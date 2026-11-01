"use client";

import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import { ShieldCheck, Lock, Activity, Server, ArrowRight, X } from "lucide-react";
import { useLanguage } from "@/context/LanguageContext";

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
  const [progress, setProgress] = useState(0);
  const [activeStage, setActiveStage] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      setProgress(0);
      setActiveStage(0);
      return;
    }

    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          setTimeout(() => {
            router.push("/dashboard");
          }, 200);
          return 100;
        }
        const next = prev + 10;
        if (next >= 25 && next < 55) setActiveStage(1);
        else if (next >= 55 && next < 85) setActiveStage(2);
        else if (next >= 85) setActiveStage(3);
        return next;
      });
    }, 110);

    return () => clearInterval(interval);
  }, [isOpen, router]);

  if (!isOpen) return null;

  const handleSkip = () => {
    router.push("/dashboard");
  };

  const steps = [
    {
      title: t("launch_modal_step1"),
      desc: "Node 1-4 enclaves online. Zero cross-institution patient leakage verified.",
      icon: Server,
    },
    {
      title: t("launch_modal_step2"),
      desc: "RLWE ciphertext ring initialized. Central server receives zero plaintext updates.",
      icon: Lock,
    },
    {
      title: t("launch_modal_step3"),
      desc: "Rank-16 low-rank adapters frozen and synchronized for non-IID drift control.",
      icon: Activity,
    },
    {
      title: t("launch_modal_step4"),
      desc: "Loading real-time orchestration cockpit, telemetry cockpit, and Grad-CAM CDSS.",
      icon: ShieldCheck,
    },
  ];

  return (
    <div className="fixed inset-0 z-[100] flex items-center justify-center p-3 sm:p-5 bg-black/75 backdrop-blur-md transition-all">
      <div className="relative w-full max-w-lg bg-[var(--bg-card)] border border-[var(--border-card)] rounded-2xl p-5 sm:p-7 shadow-2xl">
        {/* Close Button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 p-1.5 rounded-lg text-slate-400 hover:text-[var(--text-heading)] hover:bg-black/5 dark:hover:bg-white/5 transition-all"
          aria-label="Close"
        >
          <X className="w-4 h-4" />
        </button>

        {/* Header */}
        <div className="flex items-center gap-3 mb-4">
          <div className="w-10 h-10 rounded-xl bg-emerald-500/10 border border-emerald-500/20 flex items-center justify-center text-emerald-500 shrink-0">
            <Activity className="w-5 h-5 animate-pulse" />
          </div>
          <div>
            <h3 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">
              {t("launch_modal_title")}
            </h3>
            <p className="text-xs text-[var(--text-muted)]">
              {t("launch_modal_desc")}
            </p>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="mb-6">
          <div className="flex justify-between items-center text-[11px] font-semibold text-[var(--text-muted)] mb-1.5">
            <span>Federated Handshake</span>
            <span className="font-mono text-emerald-600 dark:text-emerald-400">{progress}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-200 dark:bg-white/[0.06] overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-emerald-500 to-teal-400 transition-all duration-150 ease-out rounded-full"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        {/* Animated Handshake Steps */}
        <div className="space-y-3 mb-6">
          {steps.map((step, idx) => {
            const Icon = step.icon;
            const isCompleted = activeStage > idx || progress === 100;
            const isCurrent = activeStage === idx && progress < 100;

            return (
              <div
                key={idx}
                className={`p-3 rounded-xl border transition-all ${
                  isCompleted
                    ? "bg-emerald-500/[0.04] border-emerald-500/20 text-slate-800 dark:text-slate-200"
                    : isCurrent
                    ? "bg-slate-100 dark:bg-white/[0.04] border-emerald-500/40 text-[var(--text-heading)] shadow-sm"
                    : "bg-transparent border-transparent text-slate-400 opacity-60"
                }`}
              >
                <div className="flex items-start gap-3">
                  <div
                    className={`w-6 h-6 rounded-lg flex items-center justify-center shrink-0 mt-0.5 transition-colors ${
                      isCompleted
                        ? "bg-emerald-500 text-white"
                        : isCurrent
                        ? "bg-emerald-500/20 text-emerald-500 animate-spin"
                        : "bg-slate-200 dark:bg-white/[0.06] text-slate-400"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="text-xs font-semibold leading-tight">{step.title}</div>
                    <div className="text-[11px] text-[var(--text-muted)] mt-0.5 leading-relaxed">
                      {step.desc}
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-2 border-t border-[var(--border-subtle)]">
          <span className="text-[11px] text-[var(--text-muted)]">
            4 Nodes Protected: Patient DP ε ≤ 2.0
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
