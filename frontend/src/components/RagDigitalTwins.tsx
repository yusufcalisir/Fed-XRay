"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { RagCase } from "@/types";
import { Dna, CheckCircle, ArrowUpRight } from "lucide-react";

interface RagDigitalTwinsProps {
  twins: RagCase[];
}

export default function RagDigitalTwins({ twins }: RagDigitalTwinsProps) {
  const { t } = useLanguage();

  if (!twins || twins.length === 0) return null;

  return (
    <div className="p-6 rounded-3xl bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 shadow-sm mb-6">
      <div className="flex items-center gap-2.5 text-sm font-bold text-slate-900 dark:text-white mb-1">
        <Dna className="w-5 h-5 text-brand-500" />
        <span>{t("sec3_rag_title")}</span>
      </div>
      <p className="text-xs text-slate-500 mb-4">{t("sec3_rag_desc")}</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {twins.map((caseItem) => (
          <div
            key={caseItem.case_id}
            className="p-4 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800 flex flex-col justify-between hover:border-brand-500/40 transition-all"
          >
            <div className="flex items-center justify-between mb-2">
              <div>
                <span className="text-xs font-mono font-bold text-slate-900 dark:text-white">{caseItem.case_id}</span>
                <p className="text-xs font-semibold text-brand-600 dark:text-brand-400">{caseItem.label_name}</p>
              </div>
              <span className="px-2.5 py-1 rounded-full text-xs font-extrabold bg-brand-500/10 text-brand-600 dark:text-brand-400 border border-brand-500/20">
                {caseItem.similarity}% {t("sec3_rag_sim")}
              </span>
            </div>

            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed mt-2 pt-2 border-t border-slate-200 dark:border-slate-800">
              {caseItem.history}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
