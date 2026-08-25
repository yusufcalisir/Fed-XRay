"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Database, Cpu, Stethoscope, FileText, CheckCircle2, Lock, ArrowRight } from "lucide-react";

interface WorkflowStepperProps {
  hasIngested: boolean;
  isTraining: boolean;
  isTrained: boolean;
  hasDiagnosis: boolean;
  onScrollToStep?: (stepId: string) => void;
}

export default function WorkflowStepper({
  hasIngested,
  isTraining,
  isTrained,
  hasDiagnosis,
  onScrollToStep,
}: WorkflowStepperProps) {
  const { t } = useLanguage();

  const steps = [
    {
      id: "section-ingestion",
      num: 1,
      name: t("workflow_step_1"),
      icon: Database,
      isCompleted: hasIngested,
      isActive: !hasIngested,
      isLocked: false,
      statusLabel: hasIngested ? t("workflow_status_completed") : t("workflow_status_active"),
    },
    {
      id: "section-training",
      num: 2,
      name: t("workflow_step_2"),
      icon: Cpu,
      isCompleted: isTrained,
      isActive: hasIngested && !isTrained,
      isLocked: !hasIngested,
      statusLabel: isTrained
        ? t("workflow_status_completed")
        : hasIngested
        ? isTraining
          ? t("kpi_status_training")
          : t("workflow_status_active")
        : t("workflow_status_locked"),
    },
    {
      id: "section-diagnosis",
      num: 3,
      name: t("workflow_step_3"),
      icon: Stethoscope,
      isCompleted: hasDiagnosis,
      isActive: isTrained && !hasDiagnosis,
      isLocked: !isTrained,
      statusLabel: hasDiagnosis
        ? t("workflow_status_completed")
        : isTrained
        ? t("workflow_status_active")
        : t("workflow_status_locked"),
    },
    {
      id: "section-cdss",
      num: 4,
      name: t("workflow_step_4"),
      icon: FileText,
      isCompleted: hasDiagnosis,
      isActive: hasDiagnosis,
      isLocked: !hasDiagnosis,
      statusLabel: hasDiagnosis ? t("workflow_status_active") : t("workflow_status_locked"),
    },
  ];

  return (
    <div className="card p-3 sm:p-4 mb-5 sm:mb-6">
      <div className="flex items-center justify-between mb-2.5 sm:mb-3">
        <div className="text-[10px] sm:text-xs font-bold uppercase tracking-wider text-[var(--text-muted)] flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-accent-500 animate-pulse" />
          {t("workflow_pipeline_title")}
        </div>
        <div className="text-[10px] sm:text-[11px] font-mono text-[var(--text-muted)]">
          {hasDiagnosis
            ? "4 / 4 Completed"
            : isTrained
            ? "2 / 4 Completed"
            : hasIngested
            ? "1 / 4 Completed"
            : "0 / 4 Completed"}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
        {steps.map((s, idx) => {
          const Icon = s.icon;
          return (
            <div
              key={s.num}
              onClick={() => onScrollToStep && !s.isLocked && onScrollToStep(s.id)}
              className={`p-2.5 sm:p-3 rounded-xl border transition-all relative overflow-hidden flex flex-col justify-between ${
                s.isCompleted
                  ? "bg-accent-500/10 border-accent-500/30 text-accent-600 dark:text-accent-400 cursor-pointer hover:border-accent-500/50"
                  : s.isActive
                  ? "bg-[var(--bg-card-inner)] border-accent-500/40 text-[var(--text-heading)] ring-1 ring-accent-500/30 cursor-pointer"
                  : "bg-[var(--bg-card-inner)]/50 border-[var(--border-subtle)] text-[var(--text-muted)] opacity-60 cursor-not-allowed"
              }`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
                  <div
                    className={`w-6 h-6 sm:w-7 sm:h-7 rounded-lg flex items-center justify-center shrink-0 ${
                      s.isCompleted
                        ? "bg-accent-500 text-white"
                        : s.isActive
                        ? "bg-accent-500/20 text-accent-600 dark:text-accent-400"
                        : "bg-slate-200 dark:bg-navy-800 text-[var(--text-muted)]"
                    }`}
                  >
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="font-bold text-[11px] sm:text-xs truncate">{s.name}</span>
                </div>

                <div className="shrink-0 ml-1">
                  {s.isCompleted ? (
                    <CheckCircle2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-500" />
                  ) : s.isLocked ? (
                    <Lock className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-[var(--text-muted)]" />
                  ) : (
                    <span className="w-2 h-2 rounded-full bg-accent-500 animate-ping inline-block" />
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between text-[9px] sm:text-[10px] pt-1.5 border-t border-[var(--border-subtle)]">
                <span className="text-[var(--text-muted)]">Status</span>
                <span
                  className={`font-semibold uppercase tracking-wider ${
                    s.isCompleted
                      ? "text-emerald-600 dark:text-emerald-400 font-bold"
                      : s.isActive
                      ? "text-accent-600 dark:text-accent-400 font-bold"
                      : "text-[var(--text-muted)]"
                  }`}
                >
                  {s.statusLabel}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
