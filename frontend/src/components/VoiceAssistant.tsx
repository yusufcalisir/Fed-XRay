"use client";

import React, { useState } from "react";
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
    <div className="card p-5">
      <div className="flex items-center gap-2 mb-1">
        <Mic className="w-4 h-4 text-purple-400" />
        <span className="text-sm font-bold text-white">{t("sec3_voice_title")}</span>
      </div>
      <p className="text-[11px] text-slate-500 mb-4">{t("sec3_voice_desc")}</p>
      <audio controls src={audioUrl} className="w-full h-9 rounded-lg" />
    </div>
  );
}
