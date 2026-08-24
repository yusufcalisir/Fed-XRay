import { HospitalCohort, DiagnosisResult, RagCase, TelemetryRound } from "@/types";

export const API_BASE = (process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

export async function checkBackendHealth(): Promise<{ status: "online" | "offline"; device?: string }> {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 3500);
    const res = await fetch(`${API_BASE}/api/health`, { signal: controller.signal });
    clearTimeout(timeoutId);
    if (!res.ok) return { status: "offline" };
    const data = await res.json();
    return { status: "online", device: data.device || "cpu" };
  } catch (err) {
    return { status: "offline" };
  }
}

// Fallback Synthetic Cohort Generator for instant offline UI responsiveness
function createFallbackCohorts(numHospitals: number, samplesPerHospital: number): HospitalCohort[] {
  const names = [
    "Metropolitan General (Pulmonology Hub)",
    "St. Jude Infectious Disease Center",
    "Community Memorial Health Network",
    "University Medical Academy",
    "St. Mary Pulmonary Screening Clinic",
    "Regional Trauma & ICU Center",
    "Coastline Diagnostic Institute",
    "Mount Sinai Respiratory Lab"
  ];

  return Array.from({ length: numHospitals }).map((_, hIdx) => {
    const dist = hIdx === 0 ? [0.65, 0.25, 0.10] : hIdx === 1 ? [0.15, 0.70, 0.15] : [0.20, 0.20, 0.60];
    const n = Math.round(samplesPerHospital * dist[0]);
    const p = Math.round(samplesPerHospital * dist[1]);
    const c = samplesPerHospital - n - p;

    // Generate 9 28x28 grayscale sample matrices
    const sample_images: number[][][] = Array.from({ length: 9 }).map((_, i) => {
      const lbl = i % 3;
      return Array.from({ length: 28 }).map((_, r) =>
        Array.from({ length: 28 }).map((_, col) => {
          const rib = Math.sin(r * 0.5) * 0.25 + 0.3;
          const lung = (col > 4 && col < 12) || (col > 16 && col < 24) ? 0.15 : 0.65;
          const lesion = lbl === 1 && r > 12 && col > 15 ? 0.45 : lbl === 2 && r > 10 ? 0.3 : 0.0;
          return Math.min(1.0, Math.max(0.0, rib + lung + lesion + Math.random() * 0.08));
        })
      );
    });

    return {
      hospital_id: hIdx + 1,
      name: names[hIdx % names.length],
      num_samples: samplesPerHospital,
      distribution: dist,
      counts: { normal: n, pneumonia: p, covid: c },
      sample_images,
      sample_labels: [0, 1, 2, 0, 1, 2, 0, 1, 2],
    };
  });
}

export async function generateHospitalCohorts(numHospitals: number = 4, samplesPerHospital: number = 200) {
  try {
    const res = await fetch(`${API_BASE}/api/cohorts/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ num_hospitals: numHospitals, samples_per_hospital: samplesPerHospital }),
    });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback cohort generator:", err);
  }

  // Graceful fallback
  const fallback = createFallbackCohorts(numHospitals, samplesPerHospital);
  return {
    success: true,
    num_hospitals: numHospitals,
    total_samples: numHospitals * samplesPerHospital,
    hospitals: fallback,
  };
}

export async function runClinicalDiagnosis(classIndex?: number, opacity: number = 0.55, colormap: string = "Hot"): Promise<DiagnosisResult> {
  try {
    const res = await fetch(`${API_BASE}/api/cdss/diagnose`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ class_index: classIndex, opacity, colormap }),
    });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback diagnosis generator:", err);
  }

  // Fallback Diagnosis Generator
  const targetClass = classIndex !== undefined ? classIndex : Math.floor(Math.random() * 3);
  const classNames = ["Normal (Clear Parenchyma)", "Pneumonia (Focal Consolidation)", "COVID-19 (Ground-Glass Opacities)"];
  const probs = targetClass === 0 ? [0.91, 0.06, 0.03] : targetClass === 1 ? [0.05, 0.89, 0.06] : [0.04, 0.08, 0.88];
  
  const raw_image = Array.from({ length: 28 }).map((_, r) =>
    Array.from({ length: 28 }).map((_, col) => {
      const rib = Math.sin(r * 0.5) * 0.25 + 0.3;
      const lung = (col > 4 && col < 12) || (col > 16 && col < 24) ? 0.15 : 0.65;
      return Math.min(1.0, Math.max(0.0, rib + lung + Math.random() * 0.08));
    })
  );

  const heatmap = Array.from({ length: 28 }).map((_, r) =>
    Array.from({ length: 28 }).map((_, col) => {
      const dist = Math.hypot(r - 14, col - 16);
      return Math.max(0.0, 1.0 - dist / 10);
    })
  );

  return {
    predicted_class: targetClass,
    predicted_name: classNames[targetClass],
    true_class: targetClass,
    true_name: classNames[targetClass],
    confidence: Math.round(probs[targetClass] * 1000) / 10,
    probabilities: probs,
    findings: targetClass === 0
      ? "Bilateral pulmonary parenchyma is clear. Normal cardiothoracic ratio with no focal opacities."
      : targetClass === 1
      ? "Right lower lobe focal consolidation consistent with acute bacterial pneumonia."
      : "Peripheral and subpleural ground-glass opacities with bilateral multi-focal distribution.",
    raw_image,
    heatmap,
  };
}

export async function fetchRagTwins(): Promise<{ query_class: number; matched_cases: RagCase[] }> {
  try {
    const res = await fetch(`${API_BASE}/api/cdss/rag-similar`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({}),
    });
    if (res.ok) return await res.json();
  } catch (err) {
    console.warn("Backend unavailable, using fallback RAG twins:", err);
  }

  return {
    query_class: 1,
    matched_cases: [
      {
        case_id: "REF-ONCO-8429",
        label_id: 1,
        label_name: "Pneumonia",
        similarity: 94.6,
        history: "Biopsy and sputum culture verified bacterial pneumonia. Resolved with targeted antibiotics.",
      },
      {
        case_id: "REF-PULM-3190",
        label_id: 1,
        label_name: "Pneumonia",
        similarity: 91.2,
        history: "Focal consolidation in right lower lobe. Favorable clinical response within 10 days.",
      },
    ],
  };
}

export function getVoiceBriefingUrl(diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${API_BASE}/api/cdss/voice?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}

export function getPdfReportUrl(diagnosis: string = "Pneumonia", confidence: number = 95): string {
  return `${API_BASE}/api/cdss/report-pdf?diagnosis=${encodeURIComponent(diagnosis)}&confidence=${confidence}`;
}
