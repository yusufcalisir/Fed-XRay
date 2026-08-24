"use client";

import React, { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import LiveHud from "@/components/LiveHud";
import HospitalStudio from "@/components/HospitalStudio";
import TelemetryCockpit from "@/components/TelemetryCockpit";
import DiagnosticStudio from "@/components/DiagnosticStudio";
import RagDigitalTwins from "@/components/RagDigitalTwins";
import VoiceAssistant from "@/components/VoiceAssistant";
import ReportDownload from "@/components/ReportDownload";
import { HospitalCohort, TelemetryRound, DiagnosisResult, RagCase } from "@/types";
import { fetchHospitalCohorts, fetchClinicalDiagnosis, fetchRagTwins } from "@/lib/api";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { AlertCircle, Server } from "lucide-react";

export default function DashboardPage() {
  const { t } = useLanguage();
  const { apiUrl, isConnected, isChecking, lastError } = useApi();

  // Consortium State
  const [numHospitals, setNumHospitals] = useState<number>(4);
  const [totalRounds, setTotalRounds] = useState<number>(5);
  const [samplesPerHospital, setSamplesPerHospital] = useState<number>(200);
  const [hospitals, setHospitals] = useState<HospitalCohort[]>([]);
  const [isCohortLoading, setIsCohortLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  // FL Telemetry State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentRound, setCurrentRound] = useState<number>(0);
  const [history, setHistory] = useState<TelemetryRound[]>([]);
  const [isTrained, setIsTrained] = useState<boolean>(false);

  // CDSS State
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagLoading, setIsDiagLoading] = useState<boolean>(false);
  const [ragTwins, setRagTwins] = useState<RagCase[]>([]);

  // 1. Generate Multi-Hospital Cohorts (Real FastAPI Call)
  const handleGenerateCohorts = async () => {
    setErrorMessage(null);
    try {
      setIsCohortLoading(true);
      const res = await fetchHospitalCohorts(apiUrl, numHospitals, samplesPerHospital);
      if (res.success && res.hospitals) {
        setHospitals(res.hospitals);
      }
    } catch (err: any) {
      console.error("Cohort generation error:", err);
      setErrorMessage(`Backend API Hatası: ${err.message || "Sunucuya bağlanılamadı"}. Lütfen sağ üstteki 'Backend' butonundan Render API adresinizi kontrol edin.`);
    } finally {
      setIsCohortLoading(false);
    }
  };

  // 2. Start Live Streaming Federated Learning (Real FastAPI SSE Stream)
  const handleStartTraining = () => {
    setErrorMessage(null);
    setIsTraining(true);
    setHistory([]);
    setCurrentRound(0);

    try {
      const sseUrl = `${apiUrl.replace(/\/$/, "")}/api/fl/train-stream?num_rounds=${totalRounds}&local_epochs=2&learning_rate=0.0001&simulate_attack=false&activate_defense=true`;
      const eventSource = new EventSource(sseUrl);

      eventSource.onmessage = (event) => {
        const data: TelemetryRound = JSON.parse(event.data);
        setCurrentRound(data.round_num);
        setHistory((prev) => [...prev, data]);

        if (data.status === "complete" || data.round_num >= totalRounds) {
          eventSource.close();
          setIsTraining(false);
          setIsTrained(true);
        }
      };

      eventSource.onerror = (err) => {
        console.error("SSE connection error:", err);
        eventSource.close();
        setIsTraining(false);
        setErrorMessage("Federe eğitim akışı kesildi veya sunucu yanıt vermedi. Lütfen backend bağlantısını kontrol edin.");
      };
    } catch (e: any) {
      setIsTraining(false);
      setErrorMessage(e.message || "Federe eğitim başlatılamadı");
    }
  };

  // 3. Execute CDSS Inference & RAG Twin Matching (Real FastAPI Call)
  const handleRunDiagnosis = async (classIndex?: number, opacity: number = 0.55, colormap: string = "Hot") => {
    setErrorMessage(null);
    try {
      setIsDiagLoading(true);
      const result = await fetchClinicalDiagnosis(apiUrl, classIndex, opacity, colormap);
      setDiagnosis(result);

      // Fetch RAG Twin Cases
      try {
        const twinRes = await fetchRagTwins(apiUrl);
        if (twinRes && twinRes.matched_cases) {
          setRagTwins(twinRes.matched_cases);
        }
      } catch (ragErr) {
        console.error("RAG fetch error:", ragErr);
      }
    } catch (err: any) {
      console.error("Diagnosis error:", err);
      setErrorMessage(`Teşhis API Hatası: ${err.message}. Lütfen backend sunucunuzun ayakta olduğundan emin olun.`);
    } finally {
      setIsDiagLoading(false);
    }
  };

  // Initial Attempt on Mount if connected
  useEffect(() => {
    if (isConnected) {
      handleGenerateCohorts();
    }
  }, [isConnected, apiUrl]);

  const totalCohortSamples = numHospitals * samplesPerHospital;
  const currentDiagnosisName = diagnosis ? diagnosis.predicted_name.split(" ")[0] : "Normal";
  const currentConfidence = diagnosis ? diagnosis.confidence : 95.0;

  return (
    <div className="min-h-screen flex flex-col bg-canvas text-slate-900 dark:text-slate-100">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* Backend Disconnected Warning Banner */}
        {!isConnected && !isChecking && (
          <div className="mb-6 p-4 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-800 dark:text-amber-300 text-xs flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3 shadow-sm">
            <div className="flex items-center gap-2.5">
              <AlertCircle className="w-5 h-5 text-amber-500 shrink-0" />
              <div>
                <strong>Backend Bağlantısı Bekleniyor:</strong> Şu anda <code className="bg-amber-500/20 px-1.5 py-0.5 rounded font-mono">{apiUrl}</code> adresine ulaşılamıyor.
                <p className="text-[11px] opacity-85 mt-0.5">Render ücretsiz sunucusu ilk açılışta uyanıyor olabilir (~30sn) veya sağ üstteki <strong>&quot;Backend&quot;</strong> butonundan Render API adresinizi güncelleyebilirsiniz.</p>
              </div>
            </div>
            <div className="text-right shrink-0">
              <span className="font-mono text-[11px] text-amber-600 dark:text-amber-400">FastAPI Port 8000 / Render</span>
            </div>
          </div>
        )}

        {/* Dynamic Error Toast */}
        {errorMessage && (
          <div className="mb-6 p-4 rounded-2xl bg-rose-500/10 border border-rose-500/30 text-rose-800 dark:text-rose-300 text-xs flex items-center justify-between">
            <span>⚠️ {errorMessage}</span>
            <button onClick={() => setErrorMessage(null)} className="font-bold underline text-rose-600 ml-3">Kapat</button>
          </div>
        )}

        {/* Hero Banner & KPI Counters */}
        <Hero
          numHospitals={numHospitals}
          totalRounds={totalRounds}
          totalSamples={totalCohortSamples}
          isTrained={isTrained}
        />

        {/* Live Clinical HUD */}
        <LiveHud
          numHospitals={numHospitals}
          isDefenseActive={true}
          isTrained={isTrained}
        />

        {/* Section 1: Decentralized Hospital Ingestion Studio */}
        <HospitalStudio
          hospitals={hospitals}
          onGenerate={handleGenerateCohorts}
          isLoading={isCohortLoading}
        />

        {/* Section 2: Live Federated Telemetry Cockpit */}
        <TelemetryCockpit
          history={history}
          onStartTraining={handleStartTraining}
          isTraining={isTraining}
          currentRound={currentRound}
          totalRounds={totalRounds}
        />

        {/* Section 3: AI Radiologist Diagnostic Studio (CDSS) */}
        <DiagnosticStudio
          diagnosis={diagnosis}
          onDiagnose={handleRunDiagnosis}
          isLoading={isDiagLoading}
        />

        {/* CDSS Contextual Intelligence: RAG Digital Twins */}
        {diagnosis && <RagDigitalTwins twins={ragTwins} />}

        {/* CDSS Media Actions: Voice Briefing & PDF Report Export */}
        {diagnosis && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-12">
            <VoiceAssistant
              diagnosisName={currentDiagnosisName}
              confidence={currentConfidence}
            />
            <ReportDownload
              diagnosisName={currentDiagnosisName}
              confidence={currentConfidence}
            />
          </div>
        )}

      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 dark:border-slate-800 py-6 text-center text-xs text-slate-500">
        {t("footer_text")}
      </footer>
    </div>
  );
}
