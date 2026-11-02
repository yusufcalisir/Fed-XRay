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
import FedXRayLogo from "@/components/FedXRayLogo";
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
  const [activeModel, setActiveModel] = useState<string>("vit_base");
  const [activePeft, setActivePeft] = useState<string>("ffa_lora");

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
        setIsTrained(false);
        setDiagnosis(null);
        setRagTwins([]);
        setHistory([]);
        setCurrentRound(0);
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
      fetch(`${apiUrl.replace(/\/$/, "")}/api/state`)
        .then((res) => (res.ok ? res.json() : null))
        .then((data) => {
          if (data) {
            setIsTrained(Boolean(data.model_trained));
            if (!data.has_test_set || data.hospitals_loaded === 0) {
              handleGenerateCohorts();
            }
          } else {
            handleGenerateCohorts();
          }
        })
        .catch(() => {
          handleGenerateCohorts();
        });
    }
  }, [apiUrl, isConnected]);

  const handleStartTraining = async (config?: {
    modelType: string;
    algorithm: string;
    simulateAttack: boolean;
  }) => {
    setErrorMessage(null);
    setIsTraining(true);
    setHistory([]);
    setCurrentRound(0);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const selectedModel = config?.modelType || "vit_base";
    const selectedAlgorithm = config?.algorithm || "FedDyn";
    const selectedAttack = config?.simulateAttack ?? false;
    setActiveModel(selectedModel);

    await streamFederatedTraining(
      apiUrl,
      {
        numRounds: totalRounds,
        localEpochs: 2,
        learningRate: 0.0001,
        simulateAttack: selectedAttack,
        activateDefense: true,
        algorithm: selectedAlgorithm,
        lossFn: "DAFL",
        modelType: selectedModel,
        peftMode: "ffa_lora",
      },
      (roundData) => {
        setCurrentRound(roundData.round_num);
        setHistory((prev) => [...prev, roundData]);
        if (roundData.model_type) setActiveModel(roundData.model_type);
        if (roundData.peft_mode) setActivePeft(roundData.peft_mode);
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

  const handleRunDiagnosis = async (
    classIndex?: number,
    opacity?: number,
    colormap?: string,
    imageBase64?: string
  ) => {
    if (!isTrained) {
      setErrorMessage("Federated model is not trained yet. Complete Step 2 before initiating diagnosis.");
      return;
    }
    setErrorMessage(null);
    try {
      setIsDiagLoading(true);
      const result = await fetchClinicalDiagnosis(apiUrl, classIndex, opacity, colormap, imageBase64);
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
      <Navbar mode="dashboard" />

      <main className="flex-1 max-w-[1360px] w-full mx-auto px-3 sm:px-5 py-4 sm:py-6">
        {/* Offline Banner */}
        {!isConnected && !isChecking && reconnectAttempts > 2 && (
          <div className="card-inner flex items-start gap-2.5 sm:gap-3 p-3.5 sm:p-4 mb-4 sm:mb-5 border-l-4 border-l-amber-500">
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5 animate-pulse" />
            <div className="text-[11px] sm:text-xs text-[var(--text-main)] leading-relaxed">
              <span className="text-amber-600 dark:text-amber-400 font-semibold">{t("offline_banner_title")}</span>: {t("offline_banner_reconnecting")}. {t("offline_banner_hint")}
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
          modelType={activeModel}
          peftMode={activePeft}
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

        {/* Step 4: Digital Twins & Reports */}
        {diagnosis && (
          <div id="section-cdss" className="scroll-mt-20">
            <RagDigitalTwins twins={ragTwins} />
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 sm:gap-4 mb-5 sm:mb-6">
              <VoiceAssistant diagnosisName={diagName} confidence={diagConf} />
              <ReportDownload
                diagnosisName={diagName}
                confidence={diagConf}
                patientId={diagnosis.patient_id}
                predictedClass={diagnosis.predicted_class}
                findings={diagnosis.findings}
              />
            </div>
          </div>
        )}
      </main>

      <footer className="border-t border-[var(--navbar-border)] py-6 px-4 text-[var(--text-muted)] select-none">
        <div className="max-w-[1360px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-7 h-7 rounded-lg bg-slate-900 text-white dark:bg-white/[0.08] border border-slate-700/50 dark:border-white/[0.12] flex items-center justify-center shadow-xs shrink-0">
              <FedXRayLogo size={18} />
            </div>
            <div className="flex flex-col text-left">
              <span className="font-display font-extrabold text-xs text-[var(--text-heading)] tracking-tight">
                Fed<span className="text-emerald-500 font-semibold">-</span>XRay Consortium Enclave
              </span>
              <span className="text-[10px] text-[var(--text-muted)] font-mono">
                Strategy E Medical AI · Multi-Center Clinical Execution Engine
              </span>
            </div>
          </div>

          <div className="text-center sm:text-right text-[10px] sm:text-[11px] text-[var(--text-muted)] leading-relaxed">
            <div>{t("footer_text")}</div>
          </div>
        </div>
      </footer>
    </div>
  );
}

