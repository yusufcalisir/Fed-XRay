"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Mic, Volume2, AlertCircle } from "lucide-react";
import { getVoiceBriefingUrl } from "@/lib/api";

interface VoiceAssistantProps { diagnosisName: string; confidence: number; }

export default function VoiceAssistant({ diagnosisName, confidence }: VoiceAssistantProps) {
  const { t } = useLanguage();
  const { apiUrl } = useApi();
  const [audioError, setAudioError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

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

      {audioError ? (
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[10px] sm:text-[11px] text-amber-600 dark:text-amber-400">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>
            {diagnosisName} — {Math.round(confidence)}% confidence.{" "}
            <button
              onClick={() => setAudioError(false)}
              className="underline underline-offset-2 font-semibold hover:no-underline"
            >
              Retry
            </button>
          </span>
        </div>
      ) : (
        <audio
          key={audioUrl}
          controls
          src={audioUrl}
          className="w-full h-8 sm:h-9 rounded-lg"
          onPlay={() => setIsPlaying(true)}
          onPause={() => setIsPlaying(false)}
          onEnded={() => setIsPlaying(false)}
          onError={() => setAudioError(true)}
        />
      )}
    </div>
  );
}
