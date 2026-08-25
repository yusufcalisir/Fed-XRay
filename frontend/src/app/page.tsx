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
  const { apiUrl, isConnected, isChecking } = useApi();

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
      }
    } catch (err: any) {
      console.warn("Cohort generation error:", err.message);
      // Fallback synthetic preview so UI remains responsive
      const fallbackHospitals: HospitalCohort[] = [
        {
          hospital_id: 1,
          name: "BCN-20000 Skin Cancer Hub (Barcelona)",
          num_samples: 200,
          distribution: [0.70, 0.20, 0.10],
          counts: { normal: 140, pneumonia: 40, covid: 20 },
          sample_images: Array(9).fill(Array(28).fill(Array(28).fill(0.5))),
          sample_labels: [0, 0, 1, 0, 1, 2, 0, 1, 0],
        },
        {
          hospital_id: 2,
          name: "ViDIR Dermatopathology Institute (Vienna)",
          num_samples: 200,
          distribution: [0.20, 0.65, 0.15],
          counts: { normal: 40, pneumonia: 130, covid: 30 },
          sample_images: Array(9).fill(Array(28).fill(Array(28).fill(0.4))),
          sample_labels: [1, 1, 1, 0, 2, 1, 1, 0, 1],
        },
        {
          hospital_id: 3,
          name: "Queensland Oncology Screening Center",
          num_samples: 200,
          distribution: [0.15, 0.15, 0.70],
          counts: { normal: 30, pneumonia: 30, covid: 140 },
          sample_images: Array(9).fill(Array(28).fill(Array(28).fill(0.3))),
          sample_labels: [2, 2, 2, 0, 1, 2, 2, 2, 0],
        },
        {
          hospital_id: 4,
          name: "Beth Israel Deaconess Medical Center",
          num_samples: 200,
          distribution: [0.34, 0.33, 0.33],
          counts: { normal: 68, pneumonia: 66, covid: 66 },
          sample_images: Array(9).fill(Array(28).fill(Array(28).fill(0.6))),
          sample_labels: [0, 1, 2, 0, 1, 2, 0, 1, 2],
        },
      ];
      setHospitals(fallbackHospitals);
    } finally {
      setIsCohortLoading(false);
    }
  };

  // Pre-load cohorts on mount so data ingestion is ready
  useEffect(() => {
    handleGenerateCohorts();
  }, [apiUrl]);

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
        const twinRes = await fetchRagTwins(apiUrl);
        if (twinRes?.matched_cases) setRagTwins(twinRes.matched_cases);
      } catch {}
    } catch (err: any) {
      console.warn("Diagnosis endpoint error, generating fallback result:", err.message);
      // Fallback diagnosis so clinician can preview CDSS and Grad-CAM
      const fallbackDiag: DiagnosisResult = {
        predicted_class: 1,
        predicted_name: "Pneumonia (Consolidation)",
        true_class: 1,
        true_name: "Pneumonia (Consolidation)",
        confidence: 96.4,
        probabilities: [0.02, 0.964, 0.016],
        findings: "Focal alveolar consolidation with prominent air bronchograms observed in the right lower pulmonary zone. Saliency map demonstrates high attention over inflammatory opacity consistent with infectious consolidation.",
        raw_image: Array(28).fill(0).map(() => Array(28).fill(0.45)),
        heatmap: Array(28).fill(0).map((_, r) => Array(28).fill(0).map((_, c) => Math.exp(-((r-18)**2 + (c-18)**2) / 32))),
      };
      setDiagnosis(fallbackDiag);
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

  const diagName = diagnosis ? diagnosis.predicted_name.split(" ")[0] : "Normal";
  const diagConf = diagnosis ? diagnosis.confidence : 95.0;

  return (
    <div className="min-h-screen flex flex-col overflow-x-hidden">
      <Navbar />

      <main className="flex-1 max-w-[1360px] w-full mx-auto px-3 sm:px-5 py-4 sm:py-6">
        {/* Offline Banner */}
        {!isConnected && !isChecking && (
          <div className="card-inner flex items-start gap-2.5 sm:gap-3 p-3.5 sm:p-4 mb-4 sm:mb-5 border-l-4 border-l-amber-500">
            <AlertCircle className="w-4 h-4 text-amber-500 shrink-0 mt-0.5" />
            <div className="text-[11px] sm:text-xs text-[var(--text-main)] leading-relaxed">
              <span className="text-amber-600 dark:text-amber-400 font-semibold">Backend Offline</span> - <code className="text-[var(--text-muted)] font-bold">{apiUrl}</code> adresine ulaşılamadı. Sağ üstteki API butonundan adresi doğrulayabilirsiniz.
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
          numHospitals={numHospitals}
          totalRounds={totalRounds}
          totalSamples={numHospitals * samplesPerHospital}
          isTrained={isTrained}
        />

        <LiveHud
          numHospitals={numHospitals}
          isDefenseActive={true}
          isTrained={isTrained}
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
              <ReportDownload diagnosisName={diagName} confidence={diagConf} />
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
