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
  if (!twins || twins.length === 0) return null;

  return (
    <div className="card p-5 mb-4">
      <div className="flex items-center gap-2 mb-1">
        <Dna className="w-4 h-4 text-accent-500" />
        <span className="text-sm font-bold text-[var(--text-heading)]">{t("sec3_rag_title")}</span>
      </div>
      <p className="text-[11px] text-[var(--text-muted)] mb-4">{t("sec3_rag_desc")}</p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        {twins.map((c) => (
          <div key={c.case_id} className="card-inner p-4">
            <div className="flex items-center justify-between mb-2">
              <div>
                <div className="text-xs font-mono font-bold text-[var(--text-heading)]">{c.case_id}</div>
                <div className="text-[11px] text-accent-600 dark:text-accent-400 font-semibold">{c.label_name}</div>
              </div>
              <span className="badge badge-info">{c.similarity}%</span>
            </div>
            <p className="text-[11px] text-[var(--text-muted)] leading-relaxed border-t border-[var(--border-subtle)] pt-2 mt-1">
              {c.history}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
