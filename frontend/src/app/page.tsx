"use client";

import React, { useState, useEffect, useRef } from "react";
import Navbar from "@/components/Navbar";
import Hero from "@/components/Hero";
import LiveHud from "@/components/LiveHud";
import WorkflowStepper from "@/components/WorkflowStepper";
import HospitalStudio from "@/components/HospitalStudio";
import TelemetryCockpit from "@/components/TelemetryCockpit";
import DiagnosticStudio from "@/components/DiagnosticStudio";
import RagDigitalTwins from "@/components/RagDigitalTwins";
import VoiceAssistant from "@/components/VoiceAssistant";
import ReportDownload from "@/components/ReportDownload";
import { HospitalCohort, TelemetryRound, DiagnosisResult, RagCase } from "@/types";
import { fetchHospitalCohorts, fetchClinicalDiagnosis, fetchRagTwins, streamFederatedTraining } from "@/lib/api";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { AlertCircle } from "lucide-react";

export default function DashboardPage() {
  const { t } = useLanguage();
  const { apiUrl, isConnected, isChecking, reconnectAttempts } = useApi();

  const [numHospitals] = useState(4);
  const [totalRounds] = useState(5);
  const [samplesPerHospital] = useState(200);
  const [hospitals, setHospitals] = useState<HospitalCohort[]>([]);
  const [isCohortLoading, setIsCohortLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const [isTraining, setIsTraining] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [history, setHistory] = useState<TelemetryRound[]>([]);
  const [isTrained, setIsTrained] = useState(false);

  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagLoading, setIsDiagLoading] = useState(false);
  const [ragTwins, setRagTwins] = useState<RagCase[]>([]);

  const abortControllerRef = useRef<AbortController | null>(null);

  const handleGenerateCohorts = async () => {
    setErrorMessage(null);
    try {
      setIsCohortLoading(true);
      const res = await fetchHospitalCohorts(apiUrl, numHospitals, samplesPerHospital);
      if (res.success && res.hospitals) {
        setHospitals(res.hospitals);
      } else {
        throw new Error(res.error || "Cohort generation failed on backend");
      }
    } catch (err: any) {
      console.warn("Cohort generation error:", err.message);
      setHospitals([]);
      setErrorMessage(`Cohort generation failed: ${err.message}. Ensure backend service is reachable.`);
    } finally {
      setIsCohortLoading(false);
    }
  };

  // Pre-load cohorts when backend connection is verified
  useEffect(() => {
    if (isConnected) {
      handleGenerateCohorts();
      fetch(`${apiUrl.replace(/\/$/, "")}/api/state`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data && data.model_trained) {
            setIsTrained(true);
          }
        })
        .catch(() => {});
    }
  }, [apiUrl, isConnected]);

  const handleStartTraining = async () => {
    setErrorMessage(null);
    setIsTraining(true);
    setHistory([]);
    setCurrentRound(0);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    await streamFederatedTraining(
      apiUrl,
      {
        numRounds: totalRounds,
        localEpochs: 2,
        learningRate: 0.0001,
        simulateAttack: false,
        activateDefense: true,
        algorithm: "FedDyn",
        lossFn: "DAFL",
        modelType: "vit_tiny",
        peftMode: "ffa_lora",
      },
      (roundData) => {
        setCurrentRound(roundData.round_num);
        setHistory((prev) => [...prev, roundData]);
      },
      () => {
        setIsTraining(false);
        setIsTrained(true);
      },
      (err) => {
        setIsTraining(false);
        setErrorMessage(err);
      },
      abortControllerRef.current.signal
    );
  };

  const handleRunDiagnosis = async (classIndex?: number, opacity?: number, colormap?: string) => {
    setErrorMessage(null);
    try {
      setIsDiagLoading(true);
      const result = await fetchClinicalDiagnosis(apiUrl, classIndex, opacity, colormap);
      setDiagnosis(result);
      try {
        const twinRes = await fetchRagTwins(apiUrl, 2);
        if (twinRes?.matched_cases) setRagTwins(twinRes.matched_cases);
      } catch (ragErr: any) {
        console.warn("RAG retrieval warning:", ragErr.message);
        setRagTwins([]);
      }
    } catch (err: any) {
      console.error("Clinical diagnosis execution error:", err.message);
      setDiagnosis(null);
      setRagTwins([]);
      setErrorMessage(`Clinical diagnosis failed: ${err.message}. Ensure backend service is running and model is loaded.`);
    } finally {
      setIsDiagLoading(false);
    }
  };

  const scrollToSection = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const diagName = diagnosis ? diagnosis.predicted_name.split(" ")[0] : "";
  const diagConf = diagnosis ? diagnosis.confidence : 0;

  return (
    <div className="min-h-screen flex flex-col overflow-x-hidden">
      <Navbar />

      <main className="flex-1 max-w-[1360px] w-full mx-auto px-3 sm:px-5 py-4 sm:py-6">
        {/* Offline Banner */}
        {!isConnected && !isChecking && reconnectAttempts > 2 && (
          <div className="card-inner flex items-start gap-2.5 sm:gap-3 p-3.5 sm:p-4 mb-4 sm:mb-5 border-l-4 border-l-amber-500">
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5 animate-pulse" />
            <div className="text-[11px] sm:text-xs text-[var(--text-main)] leading-relaxed">
              <span className="text-amber-600 dark:text-amber-400 font-semibold">Backend Offline</span>: <code className="text-[var(--text-muted)] font-bold">{apiUrl}</code> hedefine otomatik yeniden bağlanılıyor. FastAPI servisinin (<code>uvicorn api.main:app</code>) çalıştığından emin olunuz.
            </div>
          </div>
        )}

        {/* Error Toast */}
        {errorMessage && (
          <div className="card-inner flex items-center justify-between p-3 mb-4 sm:mb-5 border-l-4 border-l-red-500 text-xs text-red-500">
            <span className="truncate">{errorMessage}</span>
            <button onClick={() => setErrorMessage(null)} className="text-[var(--text-muted)] hover:text-[var(--text-heading)] ml-3 font-bold">x</button>
          </div>
        )}

        <Hero
          numHospitals={hospitals.length}
          totalRounds={totalRounds}
          totalSamples={hospitals.reduce((acc, h) => acc + h.num_samples, 0)}
          isTrained={isTrained}
        />

        <LiveHud
          numHospitals={hospitals.length}
          isDefenseActive={true}
          isTrained={isTrained}
          modelType={history.length > 0 && history[history.length - 1].model_type ? history[history.length - 1].model_type : "vit_tiny"}
          peftMode={history.length > 0 && history[history.length - 1].peft_mode ? history[history.length - 1].peft_mode : "ffa_lora"}
        />

        {/* 4-Step Clinical Workflow Stepper */}
        <WorkflowStepper
          hasIngested={hospitals.length > 0}
          isTraining={isTraining}
          isTrained={isTrained}
          hasDiagnosis={diagnosis !== null}
          onScrollToStep={scrollToSection}
        />

        {/* Step 1: Ingestion */}
        <HospitalStudio
          hospitals={hospitals}
          onGenerate={handleGenerateCohorts}
          isLoading={isCohortLoading}
        />

        {/* Step 2: Training */}
        <TelemetryCockpit
          history={history}
          onStartTraining={handleStartTraining}
          isTraining={isTraining}
          currentRound={currentRound}
          totalRounds={totalRounds}
          isLocked={hospitals.length === 0}
          onUnlock={() => {
            scrollToSection("section-ingestion");
            handleGenerateCohorts();
          }}
        />

        {/* Step 3: Diagnostic Inference & Grad-CAM */}
        <DiagnosticStudio
          diagnosis={diagnosis}
          onDiagnose={handleRunDiagnosis}
          isLoading={isDiagLoading}
          isLocked={!isTrained}
          onScrollToTraining={() => scrollToSection("section-training")}
        />

        {/* Step 4: Digital Twins & Reports (Revealed once diagnosis is generated) */}
        {diagnosis && (
          <div id="section-cdss" className="scroll-mt-20">
            <RagDigitalTwins twins={ragTwins} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 sm:gap-4 mb-5 sm:mb-6">
              <VoiceAssistant diagnosisName={diagName} confidence={diagConf} />
              <ReportDownload
                diagnosisName={diagName}
                confidence={diagConf}
                predictedClass={diagnosis.predicted_class}
                findings={diagnosis.findings}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-[var(--navbar-border)] py-4 sm:py-5 text-center text-[10px] sm:text-[11px] text-[var(--text-muted)] px-3">
        {t("footer_text")}
      </footer>
    </div>
  );
}
