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
import { generateHospitalCohorts, runClinicalDiagnosis, fetchRagTwins } from "@/lib/api";
import { useLanguage } from "@/context/LanguageContext";

export default function DashboardPage() {
  const { t } = useLanguage();

  // Consortium State
  const [numHospitals, setNumHospitals] = useState<number>(4);
  const [totalRounds, setTotalRounds] = useState<number>(5);
  const [samplesPerHospital, setSamplesPerHospital] = useState<number>(200);
  const [hospitals, setHospitals] = useState<HospitalCohort[]>([]);
  const [isCohortLoading, setIsCohortLoading] = useState<boolean>(false);

  // FL Telemetry State
  const [isTraining, setIsTraining] = useState<boolean>(false);
  const [currentRound, setCurrentRound] = useState<number>(0);
  const [history, setHistory] = useState<TelemetryRound[]>([]);
  const [isTrained, setIsTrained] = useState<boolean>(false);

  // CDSS State
  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagLoading, setIsDiagLoading] = useState<boolean>(false);
  const [ragTwins, setRagTwins] = useState<RagCase[]>([]);

  // 1. Generate Multi-Hospital Cohorts
  const handleGenerateCohorts = async () => {
    try {
      setIsCohortLoading(true);
      const res = await generateHospitalCohorts(numHospitals, samplesPerHospital);
      if (res.success && res.hospitals) {
        setHospitals(res.hospitals);
      }
    } catch (err) {
      console.error("Cohort generation error:", err);
    } finally {
      setIsCohortLoading(false);
    }
  };

  // 2. Start Live Streaming Federated Learning
  const handleStartTraining = () => {
    setIsTraining(true);
    setHistory([]);
    setCurrentRound(0);

    const eventSource = new EventSource(
      `http://127.0.0.1:8000/api/fl/train-stream?num_rounds=${totalRounds}&local_epochs=2&learning_rate=0.0001&simulate_attack=false&activate_defense=true`
    );

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
    };
  };

  // 3. Execute CDSS Inference & RAG Twin Matching
  const handleRunDiagnosis = async (classIndex?: number, opacity: number = 0.55, colormap: string = "Hot") => {
    try {
      setIsDiagLoading(true);
      const result = await runClinicalDiagnosis(classIndex, opacity, colormap);
      setDiagnosis(result);

      // Fetch RAG Twin Cases
      try {
        const twinRes = await fetchRagTwins();
        if (twinRes && twinRes.matched_cases) {
          setRagTwins(twinRes.matched_cases);
        }
      } catch (ragErr) {
        console.error("RAG fetch error:", ragErr);
      }
    } catch (err) {
      console.error("Diagnosis error:", err);
    } finally {
      setIsDiagLoading(false);
    }
  };

  // Initial Auto-Generate on Mount
  useEffect(() => {
    handleGenerateCohorts();
  }, []);

  const totalCohortSamples = numHospitals * samplesPerHospital;
  const currentDiagnosisName = diagnosis ? diagnosis.predicted_name.split(" ")[0] : "Normal";
  const currentConfidence = diagnosis ? diagnosis.confidence : 95.0;

  return (
    <div className="min-h-screen flex flex-col bg-canvas text-slate-900 dark:text-slate-100">
      <Navbar />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
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
