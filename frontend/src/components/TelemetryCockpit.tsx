"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { TelemetryRound } from "@/types";
import { Cpu, Play, ShieldAlert, ShieldCheck, Zap, TrendingUp, Lock, ArrowUp, SlidersHorizontal } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

function formatBytes(bytes: number): string {
  if (!bytes || bytes <= 0) return "n/a";
  const units = ["B", "KB", "MB", "GB"];
  let v = bytes;
  let i = 0;
  while (v >= 1024 && i < units.length - 1) {
    v /= 1024;
    i += 1;
  }
  return `${v.toFixed(i === 0 ? 0 : 2)} ${units[i]}`;
}

export interface TrainingConfig {
  modelType: string;
  algorithm: string;
  simulateAttack: boolean;
}

interface TelemetryCockpitProps {
  history: TelemetryRound[];
  onStartTraining: (config: TrainingConfig) => void;
  isTraining: boolean;
  currentRound: number;
  totalRounds: number;
  isLocked?: boolean;
  onUnlock?: () => void;
}

export default function TelemetryCockpit({
  history,
  onStartTraining,
  isTraining,
  currentRound,
  totalRounds,
  isLocked = false,
  onUnlock,
}: TelemetryCockpitProps) {
  const { t } = useLanguage();
  const [modelType, setModelType] = useState<string>("vit_tiny");
  const [algorithm, setAlgorithm] = useState<string>("FedDyn");
  const [simulateAttack, setSimulateAttack] = useState<boolean>(false);

  const safeHistory = Array.isArray(history) ? history : [];
  const latestRound = safeHistory.length > 0 ? safeHistory[safeHistory.length - 1] : undefined;
  const fullBytes = latestRound?.full_model_bytes ?? 0;
  const uplinkBytes = latestRound?.uplink_bytes_per_client ?? 0;
  const reductionPct = fullBytes > 0 ? (1 - uplinkBytes / fullBytes) * 100 : null;

  return (
    <section id="section-training" className="mb-5 sm:mb-6 scroll-mt-20">
      {/* Section Header */}
      <div className="section-bar mb-3.5 sm:mb-4 flex-col sm:flex-row items-stretch sm:items-center">
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className={`w-9 h-9 sm:w-10 sm:h-10 rounded-xl flex items-center justify-center shrink-0 ${
            isLocked ? "bg-slate-200/50 dark:bg-navy-800 text-[var(--text-muted)]" : "bg-blue-500/10 text-blue-500 dark:text-blue-400"
          }`}>
            <Cpu className="w-4 h-4 sm:w-5 sm:h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h2 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">{t("sec2_title")}</h2>
              {isLocked && (
                <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-bold bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/20">
                  <Lock className="w-3 h-3" />
                  {t("lock_training_badge")}
                </span>
              )}
            </div>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)]">{t("sec2_subtitle")}</p>
          </div>
        </div>

        {!isLocked && (
          <button
            onClick={() => onStartTraining({ modelType, algorithm, simulateAttack })}
            disabled={isTraining}
            className="btn-primary w-full sm:w-auto mt-2.5 sm:mt-0"
          >
            <Play className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isTraining ? "animate-spin" : ""}`} />
            <span>{t("sec2_btn_start")}</span>
          </button>
        )}
      </div>

      {/* Locked State Container */}
      {isLocked ? (
        <div className="card p-8 sm:p-12 text-center relative overflow-hidden border-dashed border-2 border-[var(--border-card)]">
          <div className="w-14 h-14 rounded-2xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center mx-auto mb-3.5 text-amber-500">
            <Lock className="w-7 h-7" />
          </div>
          <h3 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)] mb-1.5">
            {t("lock_training_title")}
          </h3>
          <p className="text-xs text-[var(--text-muted)] max-w-lg mx-auto mb-5 leading-relaxed">
            {t("lock_training_desc")}
          </p>
          {onUnlock && (
            <button
              onClick={onUnlock}
              className="btn-primary inline-flex items-center gap-2 mx-auto"
            >
              <ArrowUp className="w-4 h-4" />
              <span>{t("lock_training_btn")}</span>
            </button>
          )}
        </div>
      ) : (
        <>
          {/* Hyperparameter & Defense Toolbar */}
          <div className="card p-3 sm:p-3.5 mb-3.5 sm:mb-4 flex flex-col md:flex-row md:items-center justify-between gap-3">
            <div className="flex flex-col sm:flex-row flex-wrap items-stretch sm:items-center gap-2.5 sm:gap-3 w-full md:w-auto">
              {/* Model Architecture Selector */}
              <div className="flex items-center justify-between sm:justify-start gap-2 bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] px-2.5 py-1.5 rounded-lg text-xs w-full sm:w-auto">
                <div className="flex items-center gap-2">
                  <Cpu className="w-3.5 h-3.5 text-blue-500 shrink-0" />
                  <span className="text-[var(--text-muted)] font-medium">{t("sec2_model_label")}</span>
                </div>
                <select
                  value={modelType}
                  onChange={(e) => setModelType(e.target.value)}
                  disabled={isTraining}
                  className="bg-transparent text-[var(--text-heading)] font-semibold outline-none cursor-pointer text-xs"
                >
                  <option value="vit_base" className="bg-slate-900 text-slate-100">{t("sec2_model_vit_base")}</option>
                  <option value="vit_tiny" className="bg-slate-900 text-slate-100">{t("sec2_model_vit_tiny")}</option>
                  <option value="uni" className="bg-slate-900 text-slate-100">{t("sec2_model_uni")}</option>
                  <option value="swin_unetr" className="bg-slate-900 text-slate-100">{t("sec2_model_swin")}</option>
                  <option value="fedmedclip" className="bg-slate-900 text-slate-100">{t("sec2_model_fedmedclip")}</option>
                </select>
              </div>

              {/* FL Optimizer Selector */}
              <div className="flex items-center justify-between sm:justify-start gap-2 bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] px-2.5 py-1.5 rounded-lg text-xs w-full sm:w-auto">
                <div className="flex items-center gap-2">
                  <SlidersHorizontal className="w-3.5 h-3.5 text-emerald-500 shrink-0" />
                  <span className="text-[var(--text-muted)] font-medium">{t("sec2_algo_label")}</span>
                </div>
                <select
                  value={algorithm}
                  onChange={(e) => setAlgorithm(e.target.value)}
                  disabled={isTraining}
                  className="bg-transparent text-[var(--text-heading)] font-semibold outline-none cursor-pointer text-xs"
                >
                  <option value="FedDyn" className="bg-slate-900 text-slate-100">FedDyn</option>
                  <option value="FedAvg" className="bg-slate-900 text-slate-100">FedAvg</option>
                  <option value="FedProx" className="bg-slate-900 text-slate-100">FedProx</option>
                  <option value="FedCola" className="bg-slate-900 text-slate-100">FedCola</option>
                </select>
              </div>
            </div>

            {/* Byzantine Attack Simulation Toggle */}
            <label className={`flex items-center justify-center sm:justify-start gap-2 px-3 py-1.5 rounded-lg border text-xs font-semibold cursor-pointer select-none transition-all w-full md:w-auto ${
              simulateAttack
                ? "bg-rose-500/10 border-rose-500/30 text-rose-600 dark:text-rose-400 shadow-sm"
                : "bg-[var(--bg-card-inner)] border-[var(--border-subtle)] text-[var(--text-muted)] hover:text-[var(--text-heading)]"
            }`}>
              <input
                type="checkbox"
                checked={simulateAttack}
                onChange={(e) => setSimulateAttack(e.target.checked)}
                disabled={isTraining}
                className="w-3.5 h-3.5 rounded accent-rose-500 cursor-pointer"
              />
              <span className="whitespace-nowrap">{t("sec2_simulate_attack")}</span>
            </label>
          </div>

          {/* Progress Bar */}
          {isTraining && (
            <div className="card px-3.5 sm:px-4 py-2.5 sm:py-3 mb-3.5 sm:mb-4">
              <div className="flex items-center justify-between mb-1.5 text-[10px] sm:text-[11px]">
                <span className="flex items-center gap-1.5 sm:gap-2 text-accent-600 dark:text-accent-400 font-semibold">
                  <span className="pulse-dot" />
                  {t("sec2_round")} {currentRound} / {totalRounds}
                </span>
                <span className="text-[var(--text-muted)] font-mono">{Math.round((currentRound / totalRounds) * 100)}%</span>
              </div>
              <div className="w-full h-1.5 rounded-full bg-slate-200 dark:bg-navy-800 overflow-hidden">
                <div className="h-full rounded-full bg-gradient-to-r from-accent-600 to-accent-400 transition-all duration-300" style={{ width: `${(currentRound / totalRounds) * 100}%` }} />
              </div>
            </div>
          )}

          {/* Equal 2-Column Layout */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-3.5 sm:gap-4">
            {/* Left: Convergence Chart */}
            <div className="card p-3.5 sm:p-5 flex flex-col">
              <div className="flex items-center gap-2 mb-3 sm:mb-4">
                <TrendingUp className="w-4 h-4 text-accent-500" />
                <span className="text-xs font-bold text-[var(--text-heading)]">{t("sec2_convergence_title")}</span>
              </div>
              <div className="h-48 sm:h-56 w-full flex-1 min-h-[190px]">
                {safeHistory.length > 0 ? (
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={safeHistory} margin={{ top: 5, right: 10, left: -15, bottom: 0 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(148, 163, 184, 0.2)" />
                      <XAxis dataKey="round_num" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="acc" domain={[0, 100]} stroke="#64748b" tick={{ fontSize: 10 }} />
                      <YAxis yAxisId="loss" orientation="right" stroke="#64748b" tick={{ fontSize: 10 }} />
                      <Tooltip contentStyle={{ background: "var(--bg-card)", border: "1px solid var(--border-card)", borderRadius: "12px", fontSize: "11px", color: "var(--text-main)", boxShadow: "0 4px 12px rgba(0,0,0,0.1)" }} />
                      <Legend wrapperStyle={{ fontSize: "10px" }} />
                      <Line yAxisId="acc" type="monotone" dataKey="test_accuracy" name={t("sec2_acc")} stroke="#10b981" strokeWidth={2} dot={{ r: 3 }} />
                      <Line yAxisId="loss" type="monotone" dataKey="test_loss" name={t("sec2_loss")} stroke="#ef4444" strokeWidth={2} dot={{ r: 3 }} />
                    </LineChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)] text-center p-4">
                    {t("sec2_convergence_empty")}
                  </div>
                )}
              </div>
            </div>

            {/* Right: Shield + Economics */}
            <div className="flex flex-col gap-3.5 sm:gap-4">
              {/* Byzantine Shield */}
              <div className="card p-3.5 sm:p-5 flex-1 flex flex-col justify-between">
                <div className="flex items-center gap-2.5 sm:gap-3 mb-3">
                  {latestRound?.threat_detected ? (
                    <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-red-500/10 border border-red-500/20 flex items-center justify-center shrink-0">
                      <ShieldAlert className="w-4 h-4 sm:w-5 sm:h-5 text-red-500 animate-pulse" />
                    </div>
                  ) : (
                    <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-accent-500/10 border border-accent-500/20 flex items-center justify-center shrink-0">
                      <ShieldCheck className="w-4 h-4 sm:w-5 sm:h-5 text-accent-500" />
                    </div>
                  )}
                  <div>
                    <div className="text-xs sm:text-sm font-bold text-[var(--text-heading)]">{t("sec2_shield_title")}</div>
                    <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)]">{t("sec2_shield_subtitle")}</div>
                  </div>
                </div>
                
                <div className="pt-3 border-t border-[var(--border-subtle)] flex items-center gap-2 text-xs">
                  <span className={`w-2 h-2 rounded-full shrink-0 ${latestRound?.threat_detected ? "bg-red-500" : "bg-emerald-500 shadow-[0_0_8px_#10b981]"}`} />
                  <span className={`font-semibold text-[11px] sm:text-xs ${latestRound?.threat_detected ? "text-red-500" : "text-emerald-600 dark:text-emerald-400"}`}>
                    {latestRound?.threat_detected ? `${t("sec2_shield_alert")}: Node(s) ${latestRound.blocked_nodes.join(", ")}` : t("sec2_shield_secure")}
                  </span>
                </div>
              </div>

              {/* Communication Economics */}
              <div className="card p-3.5 sm:p-5 flex-1 flex flex-col justify-between">
                <div>
                  <div className="flex items-center gap-2 mb-2.5 sm:mb-3">
                    <Zap className="w-4 h-4 text-amber-500 shrink-0" />
                    <span className="metric-label">{t("sec2_economics_title")}</span>
                  </div>
                  <div className="space-y-2 text-[11px] sm:text-xs">
                    <div className="flex justify-between text-[var(--text-muted)]">
                      <span>{t("sec2_economics_full")}</span>
                      <span className="font-mono font-bold text-[var(--text-heading)]">{formatBytes(fullBytes)}</span>
                    </div>
                    <div className="flex justify-between text-accent-600 dark:text-accent-400 font-semibold">
                      <span>{t("sec2_economics_peft")}</span>
                      <span className="font-mono font-bold">{formatBytes(uplinkBytes)}</span>
                    </div>
                  </div>
                </div>

                <div className="mt-3 pt-2.5 border-t border-[var(--border-subtle)] flex items-center justify-between text-[11px] sm:text-xs">
                  <span className="text-[var(--text-muted)]">{t("sec2_optimization_label")}</span>
                  <span className="font-bold text-accent-600 dark:text-accent-400 bg-accent-500/10 px-2 py-0.5 rounded border border-accent-500/20">
                    {reductionPct === null
                      ? "n/a"
                      : `${reductionPct >= 0 ? reductionPct.toFixed(1) : `+${Math.abs(reductionPct).toFixed(1)}`}% ${reductionPct >= 0 ? t("sec2_economics_savings") : t("sec2_economics_overhead")}`}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </>
      )}
    </section>
  );
}
