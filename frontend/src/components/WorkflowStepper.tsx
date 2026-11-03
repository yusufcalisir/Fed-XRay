"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { Database, Cpu, Stethoscope, FileText, CheckCircle2, Lock } from "lucide-react";

interface WorkflowStepperProps {
  hasIngested: boolean;
  isTraining: boolean;
  isTrained: boolean;
  hasDiagnosis: boolean;
  activeSection?: string;
  isHydrating?: boolean;
  onScrollToStep?: (stepId: string) => void;
}

export default function WorkflowStepper({
  hasIngested,
  isTraining,
  isTrained,
  hasDiagnosis,
  activeSection,
  isHydrating = false,
  onScrollToStep,
}: WorkflowStepperProps) {
  const { t } = useLanguage();

  const isTrainingDone = hasIngested && isTrained;
  const isDiagDone = isTrainingDone && hasDiagnosis;

  const completedCount: number = isDiagDone
    ? 4
    : isTrainingDone
    ? 2
    : hasIngested
    ? 1
    : 0;

  // Active step resolution:
  // If activeSection prop is supplied, use it; otherwise fallback to the most advanced available step
  const defaultActive = isDiagDone
    ? "section-cdss"
    : isTrainingDone
    ? "section-diagnosis"
    : hasIngested || isTraining
    ? "section-training"
    : "section-ingestion";

  const effectiveActive = activeSection || defaultActive;

  const steps = [
    {
      id: "section-ingestion",
      num: 1,
      name: t("workflow_step_1"),
      icon: Database,
      isCompleted: hasIngested,
      isActive: !isHydrating && effectiveActive === "section-ingestion",
      isLocked: false,
      statusLabel: hasIngested
        ? t("workflow_status_completed")
        : t("workflow_status_active"),
    },
    {
      id: "section-training",
      num: 2,
      name: t("workflow_step_2"),
      icon: Cpu,
      isCompleted: isTrainingDone,
      isActive: !isHydrating && effectiveActive === "section-training",
      isLocked: !hasIngested,
      statusLabel: isTrainingDone
        ? t("workflow_status_completed")
        : isTraining
        ? t("kpi_status_training")
        : !hasIngested
        ? t("workflow_status_locked")
        : effectiveActive === "section-training"
        ? t("workflow_status_active")
        : t("kpi_status_ready"),
    },
    {
      id: "section-diagnosis",
      num: 3,
      name: t("workflow_step_3"),
      icon: Stethoscope,
      isCompleted: isDiagDone,
      isActive: !isHydrating && effectiveActive === "section-diagnosis",
      isLocked: !isTrainingDone,
      statusLabel: isDiagDone
        ? t("workflow_status_completed")
        : !isTrainingDone
        ? t("workflow_status_locked")
        : effectiveActive === "section-diagnosis"
        ? t("workflow_status_active")
        : t("kpi_status_ready"),
    },
    {
      id: "section-cdss",
      num: 4,
      name: t("workflow_step_4"),
      icon: FileText,
      isCompleted: isDiagDone,
      isActive: !isHydrating && effectiveActive === "section-cdss",
      isLocked: !isDiagDone,
      statusLabel: !isDiagDone
        ? t("workflow_status_locked")
        : effectiveActive === "section-cdss"
        ? t("workflow_status_active")
        : t("workflow_status_completed"),
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
          {isHydrating ? (
            <span className="animate-pulse">Syncing...</span>
          ) : (
            `${completedCount} / 4 ${t("workflow_completed_count")}`
          )}
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 sm:gap-3">
        {steps.map((s) => {
          const Icon = s.icon;
          const isClickable = !s.isLocked && !isHydrating && onScrollToStep;

          const cardClass = isHydrating
            ? "bg-[var(--bg-card-inner)]/50 border-[var(--border-subtle)] text-[var(--text-muted)] animate-pulse cursor-wait"
            : s.isCompleted && s.isActive
            ? "bg-accent-500/15 border-accent-500 text-accent-600 dark:text-accent-400 ring-2 ring-accent-500/40 cursor-pointer shadow-xs"
            : s.isCompleted
            ? "bg-accent-500/10 border-accent-500/30 text-accent-600 dark:text-accent-400 cursor-pointer hover:border-accent-500/50"
            : s.isActive
            ? "bg-[var(--bg-card-inner)] border-accent-500 text-[var(--text-heading)] ring-2 ring-accent-500/40 cursor-pointer shadow-xs"
            : s.isLocked
            ? "bg-[var(--bg-card-inner)]/50 border-[var(--border-subtle)] text-[var(--text-muted)] opacity-60 cursor-not-allowed"
            : "bg-[var(--bg-card-inner)] border-[var(--border-subtle)] text-[var(--text-heading)] cursor-pointer hover:border-accent-500/30";

          const iconBgClass = isHydrating
            ? "bg-slate-200 dark:bg-navy-800 text-[var(--text-muted)]"
            : s.isCompleted
            ? "bg-accent-500 text-white"
            : s.isActive
            ? "bg-accent-500/20 text-accent-600 dark:text-accent-400"
            : "bg-slate-200 dark:bg-navy-800 text-[var(--text-muted)]";

          return (
            <div
              key={s.num}
              onClick={() => isClickable && onScrollToStep(s.id)}
              className={`p-2.5 sm:p-3 rounded-xl border transition-all relative overflow-hidden flex flex-col justify-between ${cardClass}`}
            >
              <div className="flex items-center justify-between mb-1.5">
                <div className="flex items-center gap-1.5 sm:gap-2 min-w-0">
                  <div className={`w-6 h-6 sm:w-7 sm:h-7 rounded-lg flex items-center justify-center shrink-0 ${iconBgClass}`}>
                    <Icon className="w-3.5 h-3.5" />
                  </div>
                  <span className="font-bold text-[11px] sm:text-xs truncate">{s.name}</span>
                </div>

                <div className="shrink-0 ml-1">
                  {isHydrating ? null : s.isCompleted ? (
                    <div className="flex items-center gap-1">
                      {s.isActive && (
                        <span className="w-1.5 h-1.5 rounded-full bg-accent-500 animate-ping inline-block mr-0.5" />
                      )}
                      <CheckCircle2 className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-emerald-500" />
                    </div>
                  ) : s.isLocked ? (
                    <Lock className="w-3 h-3 sm:w-3.5 sm:h-3.5 text-[var(--text-muted)]" />
                  ) : s.isActive ? (
                    <span className="w-2 h-2 rounded-full bg-accent-500 animate-ping inline-block" />
                  ) : (
                    <span className="w-1.5 h-1.5 rounded-full bg-slate-300 dark:bg-navy-700 inline-block" />
                  )}
                </div>
              </div>

              <div className="flex items-center justify-between text-[9px] sm:text-[10px] pt-1.5 border-t border-[var(--border-subtle)] gap-1">
                <span className="text-[var(--text-muted)] shrink-0">{t("workflow_status_col")}</span>
                <span
                  className={`font-semibold uppercase tracking-wider truncate text-right ${
                    isHydrating
                      ? "text-[var(--text-muted)]"
                      : s.isCompleted
                      ? "text-emerald-600 dark:text-emerald-400 font-bold"
                      : s.isActive
                      ? "text-accent-600 dark:text-accent-400 font-bold"
                      : "text-[var(--text-muted)]"
                  }`}
                >
                  {isHydrating ? "..." : s.statusLabel}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
