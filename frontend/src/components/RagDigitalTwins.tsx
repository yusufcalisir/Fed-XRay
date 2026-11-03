"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { RagCase } from "@/types";
import { Dna } from "lucide-react";

interface RagDigitalTwinsProps {
  twins: RagCase[];
}

export default function RagDigitalTwins({ twins }: RagDigitalTwinsProps) {
  const { t } = useLanguage();
  if (!twins || !Array.isArray(twins) || twins.length === 0) return null;

  return (
    <div className="card p-3.5 sm:p-5 mb-3.5 sm:mb-4">
      <div className="flex items-center gap-2 mb-1">
        <Dna className="w-4 h-4 text-accent-500" />
        <span className="text-xs sm:text-sm font-bold text-[var(--text-heading)]">{t("sec3_rag_title")}</span>
      </div>
      <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mb-3 sm:mb-4">{t("sec3_rag_desc")}</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {twins && Array.isArray(twins) && twins.length > 0 ? (
          twins.map((c) => (
          <div key={c.case_id} className="p-3 sm:p-4 rounded-xl border border-[var(--border-subtle)] bg-[var(--bg-app)] flex flex-col justify-between">
            <div>
              <div className="flex items-center justify-between mb-2">
                <div>
                  <div className="text-xs font-mono font-bold text-[var(--text-heading)]">{c.case_id}</div>
                  <div className="text-[11px] text-accent-600 dark:text-accent-400 font-semibold">{c.label_name}</div>
                </div>
                <span className="badge badge-info">{c.similarity}%</span>
              </div>
              <div className="text-[10px] text-[var(--text-muted)] font-mono mb-2 flex flex-wrap items-center gap-1.5">
                {c.site_id && (
                  <span className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-600 dark:text-sky-400 font-semibold border border-sky-500/20 truncate max-w-[200px]">
                    {c.site_id.split(" (")[0]}
                  </span>
                )}
                {c.anatomical_site && (
                  <span className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 border border-[var(--border-subtle)]">
                    {c.anatomical_site}
                  </span>
                )}
                {c.biopsy_confirmed && (
                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/20">
                    Histopathology Verified
                  </span>
                )}
              </div>
              <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] leading-relaxed border-t border-[var(--border-subtle)] pt-2 mt-1">
                {c.history}
              </p>
            </div>
            {c.therapeutic_outcome && (
              <div className="text-[10px] text-emerald-600 dark:text-emerald-400 mt-2.5 p-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15">
                <span className="font-bold">Clinical Outcome:</span> {c.therapeutic_outcome}
              </div>
            )}
          </div>
        ))
        ) : (
          <div className="col-span-2 text-center py-4 text-xs text-[var(--text-muted)] font-mono">
            No digital twin cases available.
          </div>
        )}
      </div>
    </div>
  );
}
