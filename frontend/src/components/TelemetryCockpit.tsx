"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { TelemetryRound } from "@/types";
import { Cpu, Play, ShieldAlert, ShieldCheck, Zap, TrendingUp, BarChart2 } from "lucide-react";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend } from "recharts";

interface TelemetryCockpitProps {
  history: TelemetryRound[];
  onStartTraining: () => void;
  isTraining: boolean;
  currentRound: number;
  totalRounds: number;
}

export default function TelemetryCockpit({
  history,
  onStartTraining,
  isTraining,
  currentRound,
  totalRounds,
}: TelemetryCockpitProps) {
  const { t } = useLanguage();

  const latestRound = history[history.length - 1];

  return (
    <section className="mb-12">
      {/* Section Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 p-6 rounded-2xl bg-gradient-to-r from-slate-900 via-brand-950 to-slate-900 text-white shadow-lg mb-6 border border-slate-800">
        <div className="flex items-center gap-3.5">
          <div className="p-3 rounded-xl bg-brand-500/20 text-brand-400 border border-brand-500/30">
            <Cpu className="w-6 h-6" />
          </div>
          <div>
            <h2 className="font-display text-xl sm:text-2xl font-bold">{t("sec2_title")}</h2>
            <p className="text-xs sm:text-sm text-slate-400">{t("sec2_subtitle")}</p>
          </div>
        </div>

        <button
          onClick={onStartTraining}
          disabled={isTraining}
          className="flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-gradient-to-r from-brand-600 to-clinical-cyan hover:from-brand-500 hover:to-cyan-400 text-white font-semibold text-sm shadow-md hover:shadow-brand-500/25 transition-all disabled:opacity-50"
        >
          <Play className={`w-4 h-4 ${isTraining ? "animate-spin" : ""}`} />
          <span>{t("sec2_btn_start")}</span>
        </button>
      </div>

      {/* Training Progress Bar */}
      {isTraining && (
        <div className="mb-6 p-4 rounded-2xl bg-brand-950/60 border border-brand-800 text-brand-300 text-xs">
          <div className="flex items-center justify-between mb-2">
            <span className="font-bold flex items-center gap-2">
              <span className="pulse-dot"></span>
              {t("sec2_progress")}: Round {currentRound} / {totalRounds}
            </span>
            <span>{Math.round((currentRound / totalRounds) * 100)}%</span>
          </div>
          <div className="w-full h-2 rounded-full bg-slate-800 overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-brand-500 to-clinical-cyan transition-all duration-300"
              style={{ width: `${(currentRound / totalRounds) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Real-Time Telemetry Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        
        {/* Left Column: Live Convergence Chart */}
        <div className="lg:col-span-8 p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-2 text-sm font-bold text-slate-800 dark:text-slate-200">
              <TrendingUp className="w-4 h-4 text-brand-500" />
              <span>Telemetry Convergence (Loss & Accuracy)</span>
            </div>
            <span className="text-xs text-slate-500 font-mono">Dual-Axis Streaming</span>
          </div>

          <div className="h-64 sm:h-80 w-full">
            {history.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={history} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#334155" opacity={0.3} />
                  <XAxis dataKey="round_num" stroke="#94a3b8" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="acc" domain={[0, 100]} stroke="#10b981" tick={{ fontSize: 12 }} />
                  <YAxis yAxisId="loss" orientation="right" stroke="#ef4444" tick={{ fontSize: 12 }} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "#0f172a",
                      borderColor: "#334155",
                      borderRadius: "12px",
                      fontSize: "12px",
                      color: "#f8fafc",
                    }}
                  />
                  <Legend />
                  <Line
                    yAxisId="acc"
                    type="monotone"
                    dataKey="test_accuracy"
                    name={t("sec2_acc")}
                    stroke="#10b981"
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                  <Line
                    yAxisId="loss"
                    type="monotone"
                    dataKey="test_loss"
                    name={t("sec2_loss")}
                    stroke="#ef4444"
                    strokeWidth={3}
                    dot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="h-full flex items-center justify-center text-slate-400 text-xs">
                No active training telemetry. Start a round to view live curves.
              </div>
            )}
          </div>
        </div>

        {/* Right Column: Security Shield & Communication Economics */}
        <div className="lg:col-span-4 flex flex-col gap-6">
          
          {/* Byzantine Shield Card */}
          <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm flex flex-col justify-between">
            <div className="flex items-center gap-3 mb-4">
              {latestRound?.threat_detected ? (
                <div className="p-3 rounded-2xl bg-rose-500/10 text-rose-500 border border-rose-500/20 animate-pulse">
                  <ShieldAlert className="w-6 h-6" />
                </div>
              ) : (
                <div className="p-3 rounded-2xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
                  <ShieldCheck className="w-6 h-6" />
                </div>
              )}
              <div>
                <h4 className="font-display font-bold text-sm text-slate-900 dark:text-white">
                  Byzantine Defense Shield
                </h4>
                <p className="text-xs text-slate-500">Hold-out Reference Validation</p>
              </div>
            </div>

            <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 text-xs">
              {latestRound?.threat_detected ? (
                <span className="text-rose-600 dark:text-rose-400 font-semibold">
                  ⚠️ {t("sec2_shield_alert")}: Node(s) {latestRound.blocked_nodes.join(", ")}
                </span>
              ) : (
                <span className="text-emerald-600 dark:text-emerald-400 font-semibold">
                  ✅ {t("sec2_shield_secure")}
                </span>
              )}
            </div>
          </div>

          {/* Communication Economics Card */}
          <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm">
            <div className="flex items-center gap-2 text-xs font-bold text-slate-700 dark:text-slate-300 uppercase tracking-wider mb-3">
              <Zap className="w-4 h-4 text-brand-500" />
              <span>{t("sec2_economics_title")}</span>
            </div>

            <div className="space-y-2 text-xs">
              <div className="flex justify-between text-slate-500 dark:text-slate-400">
                <span>{t("sec2_economics_full")}</span>
              </div>
              <div className="flex justify-between font-bold text-emerald-600 dark:text-emerald-400">
                <span>{t("sec2_economics_peft")}</span>
              </div>
              <div className="mt-2 p-2 rounded-xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-700 dark:text-emerald-300 font-extrabold text-center">
                ✨ {t("sec2_economics_savings")}
              </div>
            </div>
          </div>

        </div>

      </div>
    </section>
  );
}
