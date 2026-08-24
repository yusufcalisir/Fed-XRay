"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { TelemetryRound } from "@/types";
import { Cpu, Play, ShieldAlert, ShieldCheck, Zap, TrendingUp } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface TelemetryCockpitProps {
  history: TelemetryRound[];
  onStartTraining: () => void;
  isTraining: boolean;
  currentRound: number;
  totalRounds: number;
}

export default function TelemetryCockpit({ history, onStartTraining, isTraining, currentRound, totalRounds }: TelemetryCockpitProps) {
  const { t } = useLanguage();
  const latestRound = history[history.length - 1];

  return (
    <section className="mb-5 sm:mb-6">
      {/* Section Header */}
      <div className="section-bar mb-3.5 sm:mb-4 flex-col sm:flex-row items-stretch sm:items-center">
        <div className="flex items-center gap-2.5 sm:gap-3">
          <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-blue-500/10 flex items-center justify-center shrink-0">
            <Cpu className="w-4 h-4 sm:w-5 sm:h-5 text-blue-500 dark:text-blue-400" />
          </div>
          <div>
            <h2 className="font-display text-base sm:text-lg font-bold text-[var(--text-heading)]">{t("sec2_title")}</h2>
            <p className="text-[11px] sm:text-xs text-[var(--text-muted)]">{t("sec2_subtitle")}</p>
          </div>
        </div>
        <button onClick={onStartTraining} disabled={isTraining} className="btn-primary w-full sm:w-auto mt-2.5 sm:mt-0">
          <Play className={`w-3.5 h-3.5 sm:w-4 sm:h-4 ${isTraining ? "animate-spin" : ""}`} />
          <span>{t("sec2_btn_start")}</span>
        </button>
      </div>

      {/* Progress Bar */}
      {isTraining && (
        <div className="card-inner px-3.5 sm:px-4 py-2.5 sm:py-3 mb-3.5 sm:mb-4">
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
            <span className="text-xs font-bold text-[var(--text-heading)]">Convergence (Loss & Accuracy)</span>
          </div>
          <div className="h-48 sm:h-56 w-full flex-1 min-h-[190px]">
            {history.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 5, right: 5, left: -25, bottom: 0 }}>
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
                Start a training round to view live convergence.
              </div>
            )}
          </div>
        </div>

        {/* Right: Shield + Economics */}
        <div className="flex flex-col gap-3.5 sm:gap-4">

          {/* Byzantine Shield */}
          <div className="card p-3.5 sm:p-5 flex-1">
            <div className="flex items-center gap-2.5 sm:gap-3 mb-3 sm:mb-4">
              {latestRound?.threat_detected ? (
                <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-red-500/10 flex items-center justify-center shrink-0">
                  <ShieldAlert className="w-4 h-4 sm:w-5 sm:h-5 text-red-500 animate-pulse" />
                </div>
              ) : (
                <div className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-accent-500/10 flex items-center justify-center shrink-0">
                  <ShieldCheck className="w-4 h-4 sm:w-5 sm:h-5 text-accent-500" />
                </div>
              )}
              <div>
                <div className="text-xs sm:text-sm font-bold text-[var(--text-heading)]">Byzantine Defense Shield</div>
                <div className="text-[10px] sm:text-[11px] text-[var(--text-muted)]">Hold-out Reference Validation</div>
              </div>
            </div>
            <div className="card-inner p-2.5 sm:p-3 text-[11px] sm:text-xs">
              {latestRound?.threat_detected ? (
                <span className="badge badge-danger">{t("sec2_shield_alert")}: Node(s) {latestRound.blocked_nodes.join(", ")}</span>
              ) : (
                <span className="badge badge-success">{t("sec2_shield_secure")}</span>
              )}
            </div>
          </div>

          {/* Communication Economics */}
          <div className="card p-3.5 sm:p-5 flex-1">
            <div className="flex items-center gap-2 mb-2.5 sm:mb-3">
              <Zap className="w-4 h-4 text-amber-500 shrink-0" />
              <span className="metric-label">{t("sec2_economics_title")}</span>
            </div>
            <div className="space-y-2 text-[11px] sm:text-xs">
              <div className="flex justify-between text-[var(--text-muted)]">
                <span>{t("sec2_economics_full")}</span>
                <span className="font-mono font-bold text-[var(--text-heading)]">2.43 GB</span>
              </div>
              <div className="flex justify-between text-accent-600 dark:text-accent-400 font-semibold">
                <span>{t("sec2_economics_peft")}</span>
                <span className="font-mono font-bold">1.60 MB</span>
              </div>
              <div className="card-inner p-2 sm:p-2.5 text-center text-accent-600 dark:text-accent-400 font-bold text-[10px] sm:text-[11px]">
                {t("sec2_economics_savings")}
              </div>
            </div>
          </div>

        </div>
      </div>
    </section>
  );
}
