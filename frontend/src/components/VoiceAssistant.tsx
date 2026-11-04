"use client";

import React, { useState } from "react";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { Mic, Volume2, AlertCircle } from "lucide-react";
import { getVoiceBriefingUrl } from "@/lib/api";

interface VoiceAssistantProps { diagnosisName: string; confidence: number; }

export default function VoiceAssistant({ diagnosisName, confidence }: VoiceAssistantProps) {
  const { t, language } = useLanguage();
  const { apiUrl } = useApi();
  const [audioError, setAudioError] = useState(false);
  const [isPlaying, setIsPlaying] = useState(false);

  const langCode = (language || "EN").toLowerCase();
  const audioUrl = getVoiceBriefingUrl(apiUrl, diagnosisName, confidence, langCode);

  const isTr = langCode === "tr";
  const localizedDiagnosis = isTr
    ? diagnosisName.toLowerCase().includes("normal")
      ? "Normal Bulgular"
      : diagnosisName.toLowerCase().includes("pneumonia") || diagnosisName.toLowerCase().includes("zatürre")
      ? "Zatürre (Pnömoni)"
      : diagnosisName.toLowerCase().includes("covid")
      ? "COVID-19"
      : diagnosisName.toLowerCase().includes("nevus")
      ? "Benign Displastik Nevüs"
      : diagnosisName.toLowerCase().includes("melanoma")
      ? "Malign Melanom"
      : diagnosisName.toLowerCase().includes("carcinoma")
      ? "Bazal Hücreli Karsinom"
      : diagnosisName
    : diagnosisName;

  const spokenPreviewText = isTr
    ? `Yüzde ${Math.round(confidence)} model güveni ile ${localizedDiagnosis} teşhisi doğrulandı. Federe model analizi tamamlandı.`
    : `${diagnosisName} diagnosis with ${Math.round(confidence)} percent confidence. Federated model inference complete.`;

  return (
    <div className="card p-3.5 sm:p-5 flex flex-col justify-between">
      <div>
        <div className="flex items-center gap-2 mb-1">
          <Mic className="w-4 h-4 text-purple-500 dark:text-purple-400 shrink-0" />
          <span className="text-xs sm:text-sm font-bold text-[var(--text-heading)]">{t("sec3_voice_title")}</span>
        </div>
        <p className="text-[10px] sm:text-[11px] text-[var(--text-muted)] mb-3 sm:mb-4">{t("sec3_voice_desc")}</p>

        {/* Spoken Text Readout Banner */}
        <div className="p-2.5 rounded-lg bg-[var(--bg-card-inner)] border border-[var(--border-subtle)] text-[10px] sm:text-[11px] mb-3 sm:mb-4 leading-relaxed">
          <span className="text-[var(--text-muted)] font-medium">
            {t("sec3_voice_preview_label")}{" "}
          </span>
          <span className="font-semibold text-[var(--text-heading)]">
            &ldquo;{spokenPreviewText}&rdquo;
          </span>
        </div>
      </div>

      {audioError ? (
        <div className="flex items-center gap-2 p-2.5 rounded-lg bg-amber-500/10 border border-amber-500/20 text-[10px] sm:text-[11px] text-amber-600 dark:text-amber-400">
          <AlertCircle className="w-3.5 h-3.5 shrink-0" />
          <span>
            {isTr
              ? `${localizedDiagnosis} (%${Math.round(confidence)} güven). `
              : `${diagnosisName} (${Math.round(confidence)}% confidence). `}
            <button
              onClick={() => setAudioError(false)}
              className="underline underline-offset-2 font-semibold hover:no-underline"
            >
              {t("btn_retry")}
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
