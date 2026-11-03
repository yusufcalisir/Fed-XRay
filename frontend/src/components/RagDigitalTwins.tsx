"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { RagCase } from "@/types";
import { Dna } from "lucide-react";

interface RagDigitalTwinsProps {
  twins: RagCase[];
}

export default function RagDigitalTwins({ twins }: RagDigitalTwinsProps) {
  const { t, language } = useLanguage();
  if (!twins || !Array.isArray(twins) || twins.length === 0) return null;

  const translateLabel = (label: string): string => {
    if (language !== "TR") return label;
    if (label.includes("Melanoma")) return "Malign Melanom";
    if (label.includes("Basal Cell")) return "Bazal Hücreli Karsinom";
    if (label.includes("Nevus") || label.includes("Nevüs")) return "Benign Displastik Nevüs";
    if (label.includes("Pneumonia")) return "Bakteriyel Pnömoni";
    if (label.includes("COVID")) return "COVID-19 Viral Konsolidasyon";
    if (label.includes("Normal")) return "Normal / Sağlıklı";
    return label;
  };

  const translateHistory = (history: string, label: string): string => {
    if (language !== "TR") return history;
    const lower = history.toLowerCase();
    if (lower.includes("bacterial lobar consolidation") || lower.includes("levofloxacin")) {
      return "Biyopsi ile doğrulanmış bakteriyel lober konsolidasyon. 14 günlük levofloksasin tedavisi ile tam remisyon sağlandı.";
    }
    if (lower.includes("hospital-acquired pneumonia") || lower.includes("ampicillin")) {
      return "Hastane kaynaklı pnömoni tablosu. IV ampisilin-sulbaktam protokolüne hızlı klinik yanıt verdi.";
    }
    if (lower.includes("melanoma") || label.includes("Melanoma")) {
      return "55 yaşındaki hastada gövde bölgesinde histopatoloji (biyopsi) ile doğrulanmış Malign Melanom. Eksantrik hiperpigmentasyon ve periferik çizgilenmeler mevcut. Geniş lokal eksizyon endikedir.";
    }
    if (lower.includes("basal cell") || label.includes("Basal")) {
      return "Histopatoloji ile doğrulanmış Bazal Hücreli Karsinom. Mohs mikrografik cerrahisi ile temiz cerrahi sınırlar elde edildi.";
    }
    if (lower.includes("nevus") || label.includes("Nevus")) {
      return "Histopatoloji ile doğrulanmış Displastik Nevüs. Dermoskopik takip ve koruyucu cerrahi eksizyon önerildi.";
    }
    return history;
  };

  const translateOutcome = (outcome?: string): string => {
    if (!outcome) return "";
    if (language !== "TR") return outcome;
    const lower = outcome.toLowerCase();
    if (lower.includes("antibiotic") || lower.includes("recovery")) {
      return "Antibiyotik tedavisi sonrası tam iyileşme";
    }
    if (lower.includes("excision") || lower.includes("pathological response")) {
      return "Geniş eksizyon sonrası tam patolojik yanıt";
    }
    return outcome;
  };

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
                  <div className="text-[11px] text-accent-600 dark:text-accent-400 font-semibold">{translateLabel(c.label_name)}</div>
                </div>
                <span className="badge badge-info">{c.similarity}%</span>
              </div>
              <div className="text-[10px] text-[var(--text-muted)] font-mono mb-2 flex flex-wrap items-center gap-1.5">
                {c.site_id && (
                  <span className="px-1.5 py-0.5 rounded bg-sky-500/10 text-sky-600 dark:text-sky-400 font-semibold border border-sky-500/20 truncate max-w-[200px]">
                    {language === "TR" && c.site_id.includes("BCN-20000")
                      ? "BCN-20000 Deri Kanseri Merkezi (Barselona)"
                      : c.site_id.split(" (")[0]}
                  </span>
                )}
                {c.anatomical_site && (
                  <span className="px-1.5 py-0.5 rounded bg-black/5 dark:bg-white/5 border border-[var(--border-subtle)]">
                    {language === "TR" && c.anatomical_site.toLowerCase().includes("trunk")
                      ? "Gövde"
                      : language === "TR" && c.anatomical_site.toLowerCase().includes("thorax")
                      ? "Toraks / Sol Lob"
                      : c.anatomical_site}
                  </span>
                )}
                {c.biopsy_confirmed && (
                  <span className="px-1.5 py-0.5 rounded bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-bold border border-emerald-500/20">
                    {t("rag_verified_tag")}
                  </span>
                )}
              </div>
              <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] leading-relaxed border-t border-[var(--border-subtle)] pt-2 mt-1">
                {translateHistory(c.history, c.label_name)}
              </p>
            </div>
            {c.therapeutic_outcome && (
              <div className="text-[10px] text-emerald-600 dark:text-emerald-400 mt-2.5 p-2 rounded-lg bg-emerald-500/5 border border-emerald-500/15">
                <span className="font-bold">{t("rag_outcome_label")}</span> {translateOutcome(c.therapeutic_outcome)}
              </div>
            )}
          </div>
        ))
        ) : (
          <div className="col-span-2 text-center py-4 text-xs text-[var(--text-muted)] font-mono">
            {t("rag_no_cases")}
          </div>
        )}
      </div>
    </div>
  );
}
