"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Mic } from "lucide-react";
import { getVoiceBriefingUrl } from "@/lib/api";

interface VoiceAssistantProps {
  diagnosisName: string;
  confidence: number;
}

export default function VoiceAssistant({ diagnosisName, confidence }: VoiceAssistantProps) {
  const { t } = useLanguage();
  const { apiUrl } = useApi();
  const [isPlaying, setIsPlaying] = useState(false);
  const audioUrl = getVoiceBriefingUrl(apiUrl, diagnosisName, confidence);

  return (
    <div className="p-6 rounded-3xl bg-gradient-to-br from-indigo-900/40 via-slate-900 to-slate-900 border border-indigo-500/20 shadow-sm flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2.5 text-sm font-bold text-white mb-1">
          <Mic className="w-5 h-5 text-indigo-400" />
          <span>{t("sec3_voice_title")}</span>
        </div>
        <p className="text-xs text-slate-400 mb-4">{t("sec3_voice_desc")}</p>
      </div>

      <div className="pt-2">
        <audio
          controls
          src={audioUrl}
          className="w-full h-10 accent-indigo-500 rounded-xl"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
        />
      </div>
    </div>
  );
}
