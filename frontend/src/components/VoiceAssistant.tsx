"use client";

import React from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Mic } from "lucide-react";
import { getVoiceBriefingUrl } from "@/lib/api";

interface VoiceAssistantProps { diagnosisName: string; confidence: number; }

export default function VoiceAssistant({ diagnosisName, confidence }: VoiceAssistantProps) {
  const { t } = useLanguage();
  const { apiUrl } = useApi();
  const audioUrl = getVoiceBriefingUrl(apiUrl, diagnosisName, confidence);

  return (
    <div className="card p-3.5 sm:p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Mic className="w-4 h-4 text-purple-500 dark:text-purple-400 shrink-0" />
          <span className="text-xs sm:text-sm font-bold text-[var(--text-heading)]">{t("sec3_voice_title")}</span>
        </div>
        <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mb-3 sm:mb-4">{t("sec3_voice_desc")}</p>
      </div>
      <audio controls src={audioUrl} className="w-full h-8 sm:h-9 rounded-lg" />
    </div>
  );
}
