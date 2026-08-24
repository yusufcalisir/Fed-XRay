import { HospitalCohort, DiagnosisResult, RagCase } from "@/types";

export async function fetchHospitalCohorts(baseUrl: string, numHospitals: number = 4, samplesPerHospital: number = 200) {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cohorts/generate`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_hospitals: numHospitals, samples_per_hospital: samplesPerHospital }),
  });
  if (!res.ok) {
    throw new Error(`Failed to generate cohorts from backend (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function fetchClinicalDiagnosis(baseUrl: string, classIndex?: number, opacity: number = 0.55, colormap: string = "Hot"): Promise<DiagnosisResult> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/diagnose`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_index: classIndex, opacity, colormap }),
  });
  if (!res.ok) {
    throw new Error(`Diagnosis failed (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function fetchRagTwins(baseUrl: string): Promise<{ query_class: number; matched_cases: RagCase[] }> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/rag-similar`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  if (!res.ok) {
    throw new Error(`RAG retrieval failed (HTTP ${res.status})`);
  }
  return await res.json();
}

export function getVoiceBriefingUrl(baseUrl: string, diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${baseUrl.replace(/\/$/, "")}/api/cdss/voice?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}

export function getPdfReportUrl(baseUrl: string, diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${baseUrl.replace(/\/$/, "")}/api/cdss/report-pdf?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}
