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
import Footer from "@/components/Footer";
import { HospitalCohort, TelemetryRound, DiagnosisResult, RagCase, TrainingConfig } from "@/types";
import { fetchHospitalCohorts, fetchClinicalDiagnosis, fetchRagTwins, streamFederatedTraining } from "@/lib/api";
import { useLanguage } from "@/context/LanguageContext";
import { useApi } from "@/context/ApiContext";
import { AlertCircle } from "lucide-react";
import ClinicalErrorBoundary from "@/components/ClinicalErrorBoundary";

export default function DashboardPage() {
  const { t, language } = useLanguage();
  const { apiUrl, isConnected, isChecking, reconnectAttempts } = useApi();

  const [numHospitals] = useState(4);
  const [totalRounds] = useState(5);
  const [samplesPerHospital] = useState(200);
  const [hospitals, setHospitals] = useState<HospitalCohort[]>([]);
  const [isCohortLoading, setIsCohortLoading] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isHydrating, setIsHydrating] = useState(true);
  const [activeSection, setActiveSection] = useState<string>("section-ingestion");

  const [isTraining, setIsTraining] = useState(false);
  const [currentRound, setCurrentRound] = useState(0);
  const [history, setHistory] = useState<TelemetryRound[]>([]);
  const [isTrained, setIsTrained] = useState(false);
  const [activeModel, setActiveModel] = useState<string>("vit_tiny");
  const [activePeft, setActivePeft] = useState<string>("ffa_lora");

  const [diagnosis, setDiagnosis] = useState<DiagnosisResult | null>(null);
  const [isDiagLoading, setIsDiagLoading] = useState(false);
  const [ragTwins, setRagTwins] = useState<RagCase[]>([]);

  const abortControllerRef = useRef<AbortController | null>(null);

  const sanitizeCohorts = (cohortList: any[]): HospitalCohort[] => {
    return cohortList.map((h: any) => ({
      ...h,
      sample_images: Array.isArray(h.sample_images) ? h.sample_images : [],
    }));
  };

  const handleGenerateCohorts = async () => {
    setErrorMessage(null);
    setActiveSection("section-ingestion");
    try {
      setIsCohortLoading(true);
      const res = await fetchHospitalCohorts(apiUrl, numHospitals, samplesPerHospital);
      if (res.success && res.hospitals) {
        setHospitals(sanitizeCohorts(res.hospitals));
        setIsTrained(false);
        setDiagnosis(null);
        setRagTwins([]);
        setHistory([]);
        setCurrentRound(0);
      } else {
        throw new Error(res.error || t("dashboard_cohort_error"));
      }
    } catch (err: any) {
      console.warn("Cohort generation error:", err.message);
      setHospitals([]);
      setErrorMessage(`${t("dashboard_cohort_error_prefix")}: ${err.message}. ${t("dashboard_cohort_error_suffix")}`);
    } finally {
      setIsCohortLoading(false);
    }
  };

  // Load state when backend is available
  useEffect(() => {
    let isCancelled = false;

    const syncState = async () => {
      const candidates = [
        apiUrl,
        "",
        "http://127.0.0.1:8000",
        "http://localhost:8000",
        "https://fed-xray-api.onrender.com",
      ].filter((c, idx, arr) => Boolean(c !== undefined && c !== null) && arr.indexOf(c) === idx);

      let fetchedData: any = null;
      let workingBase = apiUrl;

      for (const cand of candidates) {
        try {
          const cleanUrl = cand.replace(/\/+$/, "");
          const endpoint = cleanUrl ? `${cleanUrl}/api/state` : "/api/state";
          const controller = new AbortController();
          const timer = setTimeout(() => controller.abort(), 2000);
          const res = await fetch(endpoint, { signal: controller.signal, cache: "no-store" });
          clearTimeout(timer);
          if (res.ok) {
            fetchedData = await res.json();
            workingBase = cand;
            break;
          }
        } catch {
          // Probe next candidate
        }
      }

      if (isCancelled || !fetchedData) {
        if (!isCancelled) setIsHydrating(false);
        return;
      }

      // Hydrate cohorts from cohorts field or hospitals field
      const rawCohorts = (fetchedData.cohorts && Array.isArray(fetchedData.cohorts) && fetchedData.cohorts.length > 0)
        ? fetchedData.cohorts
        : (fetchedData.hospitals && Array.isArray(fetchedData.hospitals) && fetchedData.hospitals[0]?.sample_images)
        ? fetchedData.hospitals
        : null;

      if (rawCohorts && rawCohorts.length > 0) {
        setHospitals(sanitizeCohorts(rawCohorts));
      } else {
        // Auto-load authentic Strategy E cohorts if backend has not loaded them yet
        try {
          const res = await fetchHospitalCohorts(workingBase, 4, 50);
          if (!isCancelled) {
            if (res.success && res.hospitals) {
              setHospitals(sanitizeCohorts(res.hospitals));
            } else {
              setHospitals([]);
              setErrorMessage(res.error || t("dashboard_cohort_error"));
            }
          }
        } catch (err: any) {
          // Fail honestly: do not substitute fake mock data on backend failure
          if (!isCancelled) {
            console.warn("Auto-cohort hydration failed:", err.message);
            setHospitals([]);
            setErrorMessage(`${t("dashboard_cohort_error_prefix")}: ${err.message}. ${t("dashboard_cohort_error_suffix")}`);
          }
        }
      }

      if (isCancelled) return;
      // In the interactive clinical cockpit, a new session begins in clean un-trained state.
      // Do not auto-populate historical rounds as a completed training run.
      setHistory([]);
      setCurrentRound(0);
      setIsTrained(false);

      if (fetchedData.reference_cases && Array.isArray(fetchedData.reference_cases)) {
        setRagTwins(fetchedData.reference_cases);
      }
      if (fetchedData.model_type) setActiveModel(fetchedData.model_type);
      if (fetchedData.peft_mode) setActivePeft(fetchedData.peft_mode);
      if (!isCancelled) setIsHydrating(false);
    };

    syncState();

    return () => {
      isCancelled = true;
    };
  }, [apiUrl, isConnected]);

  // Track active section in viewport so the workflow stepper accurately highlights the stage in view
  useEffect(() => {
    if (typeof window === "undefined") return;

    const sectionIds = ["section-ingestion", "section-training", "section-diagnosis", "section-cdss"];

    const handleScroll = () => {
      // If user is near the top of the dashboard, step 1 is active
      if (window.scrollY < 200) {
        setActiveSection("section-ingestion");
        return;
      }

      // Check which section is currently centered/visible in the top third of viewport
      const scrollPos = window.scrollY + 220;

      let current = sectionIds[0];
      for (const id of sectionIds) {
        const el = document.getElementById(id);
        if (el) {
          const top = el.offsetTop;
          const height = el.offsetHeight;
          if (scrollPos >= top && scrollPos < top + height) {
            current = id;
            break;
          }
        }
      }
      setActiveSection(current);
    };

    window.addEventListener("scroll", handleScroll, { passive: true });
    handleScroll();

    return () => {
      window.removeEventListener("scroll", handleScroll);
    };
  }, [hospitals.length, history.length, diagnosis]);

  const handleStartTraining = async (config?: TrainingConfig) => {
    setErrorMessage(null);
    setIsTraining(true);
    setActiveSection("section-training");
    setHistory([]);
    setCurrentRound(0);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    const selectedModel = config?.modelType || "vit_tiny";
    const selectedAlgorithm = config?.algorithm || "FedDyn";
    const selectedAttack = config?.simulateAttack ?? false;
    const selectedPeft = config?.peftMode || "ffa_lora";
    const selectedLoss = config?.lossFn || "DAFL";
    const selectedSecAgg = config?.secureAggregation ?? false;
    const selectedCkks = config?.useCkks ?? false;
    const selectedDpMode = config?.dpMode || "distributed";
    const selectedClip = config?.dpClippingBound ?? 1.0;
    const selectedNoise = config?.privacyNoise ?? 0.01;
    const selectedEpochs = config?.localEpochs ?? 2;
    const selectedLr = config?.learningRate ?? 0.0001;
    const selectedDefense = config?.activateDefense ?? true;

    setActiveModel(selectedModel);
    setActivePeft(selectedPeft);

    await streamFederatedTraining(
      apiUrl,
      {
        numRounds: totalRounds,
        localEpochs: selectedEpochs,
        learningRate: selectedLr,
        simulateAttack: selectedAttack,
        activateDefense: selectedDefense,
        algorithm: selectedAlgorithm,
        lossFn: selectedLoss,
        modelType: selectedModel,
        peftMode: selectedPeft,
        secureAggregation: selectedSecAgg,
        useCkks: selectedCkks,
        dpMode: selectedDpMode,
        dpClippingBound: selectedClip,
        privacyNoise: selectedNoise,
      },
      (roundData: TelemetryRound) => {
        setCurrentRound(roundData.round_num);
        setHistory((prev) => [...prev, roundData]);
        if (roundData.model_type) setActiveModel(roundData.model_type);
        if (roundData.peft_mode) setActivePeft(roundData.peft_mode);
      },
      () => {
        setIsTraining(false);
        setIsTrained(true);
      },
      (err: string) => {
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
      setErrorMessage(t("dashboard_model_not_trained"));
      return;
    }
    setErrorMessage(null);
    setActiveSection("section-diagnosis");
    try {
      setIsDiagLoading(true);
      const result = await fetchClinicalDiagnosis(
        apiUrl,
        classIndex,
        opacity,
        colormap,
        imageBase64,
        language.toLowerCase()
      );
      setDiagnosis(result);
      setActiveSection("section-cdss");
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
    setActiveSection(id);
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  };

  const fullDiagName = diagnosis ? diagnosis.predicted_name : "";
  const diagConf = diagnosis ? diagnosis.confidence : 0;

  return (
    <div className="min-h-screen flex flex-col overflow-x-hidden">
      <Navbar mode="dashboard" />

      <main className="flex-1 max-w-[1360px] w-full mx-auto px-3 sm:px-5 py-4 sm:py-6">
        {/* Research Prototype Disclaimer */}
        <div className="card-inner flex items-start gap-2.5 p-3 mb-4 border-l-4 border-l-amber-400 bg-amber-50/5">
          <AlertCircle className="w-3.5 h-3.5 text-amber-500 shrink-0 mt-0.5" />
          <p className="text-[10px] text-[var(--text-muted)] leading-relaxed">
            <span className="font-semibold text-amber-600 dark:text-amber-400">{t("dashboard_disclaimer_badge")}:</span>{" "}
            {t("dashboard_disclaimer_text")}
          </p>
        </div>

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

        {/* Training completed invariant: requires authentic ingested cohorts, model_trained flag, AND recorded rounds */}
        {(() => {
          const isTrainingDone = hospitals.length > 0 && isTrained && history.length > 0;
          return (
            <>
              <Hero
                numHospitals={hospitals.length}
                totalRounds={totalRounds}
                totalSamples={(hospitals || []).reduce((acc, h) => acc + (h?.num_samples || 0), 0)}
                isTrained={isTrainingDone}
              />

              <LiveHud
                numHospitals={hospitals.length}
                isDefenseActive={true}
                isTrained={isTrainingDone}
                modelType={activeModel}
                peftMode={activePeft}
              />

              {/* 4-Step Clinical Workflow Stepper */}
              <WorkflowStepper
                hasIngested={hospitals.length > 0}
                isTraining={isTraining}
                isTrained={isTrainingDone}
                hasDiagnosis={isTrainingDone && diagnosis !== null}
                activeSection={activeSection}
                isHydrating={isHydrating}
                onScrollToStep={scrollToSection}
              />

              {/* Step 1: Ingestion */}
              <ClinicalErrorBoundary language={language}>
                <HospitalStudio
                  hospitals={hospitals}
                  onGenerate={handleGenerateCohorts}
                  isLoading={isCohortLoading}
                />
              </ClinicalErrorBoundary>

              {/* Step 2: Training */}
              <ClinicalErrorBoundary language={language}>
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
              </ClinicalErrorBoundary>

              {/* Step 3: Diagnostic Inference & Grad-CAM */}
              <ClinicalErrorBoundary language={language}>
                <DiagnosticStudio
                  diagnosis={diagnosis}
                  onDiagnose={handleRunDiagnosis}
                  isLoading={isDiagLoading}
                  isLocked={!isTrainingDone}
                  onScrollToTraining={() => scrollToSection("section-training")}
                />
              </ClinicalErrorBoundary>
            </>
          );
        })()}

        {/* Step 4: Digital Twins & Reports */}
        {diagnosis && (
          <ClinicalErrorBoundary language={language}>
            <div id="section-cdss" className="scroll-mt-20">
              <RagDigitalTwins twins={ragTwins} />
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3.5 sm:gap-4 mb-5 sm:mb-6">
                <VoiceAssistant diagnosisName={fullDiagName} confidence={diagConf} />
                <ReportDownload
                  diagnosisName={fullDiagName}
                  confidence={diagConf}
                  patientId={diagnosis.patient_id}
                  predictedClass={diagnosis.predicted_class}
                  findings={diagnosis.findings}
                />
              </div>
            </div>
          </ClinicalErrorBoundary>
        )}
      </main>

      <Footer maxWidth="max-w-[1360px]" />
    </div>
  );
}

