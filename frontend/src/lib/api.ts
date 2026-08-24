import { HospitalCohort, DiagnosisResult, RagCase } from "@/types";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) return { status: "offline" };
    return await res.json();
  } catch (err) {
    return { status: "offline" };
  }
}

export async function generateHospitalCohorts(numHospitals: number = 4, samplesPerHospital: number = 200) {
  const res = await fetch(`${API_BASE}/api/cohorts/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_hospitals: numHospitals, samples_per_hospital: samplesPerHospital }),
  });
  if (!res.ok) throw new Error("Failed to generate cohorts");
  return await res.json();
}

export async function runClinicalDiagnosis(classIndex?: number, opacity: number = 0.55, colormap: string = "Hot"): Promise<DiagnosisResult> {
  const res = await fetch(`${API_BASE}/api/cdss/diagnose`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_index: classIndex, opacity, colormap }),
  });
  if (!res.ok) throw new Error("Diagnosis failed");
  return await res.json();
}

export async function fetchRagTwins(): Promise<{ query_class: number; matched_cases: RagCase[] }> {
  const res = await fetch(`${API_BASE}/api/cdss/rag-similar`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) throw new Error("RAG twin retrieval failed");
  return await res.json();
}

export function getVoiceBriefingUrl(diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${API_BASE}/api/cdss/voice?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}

export function getPdfReportUrl(diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${API_BASE}/api/cdss/report-pdf?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}
