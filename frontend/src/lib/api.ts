import {
  HospitalCohort,
  DiagnosisResult,
  RagCase,
  TelemetryRound,
  BackendSystemState,
  EnclaveHandshakeResponse,
  FederatedTrainingOptions,
  TrainingCancelResponse,
} from "@/types";

const DIAG_TO_CLASS: Record<string, number> = {
  Normal: 0,
  Pneumonia: 1,
  COVID: 2,
  "COVID-19": 2,
  "Benign Dysplastic Nevus": 0,
  "Malignant Melanoma": 1,
  "Basal Cell Carcinoma": 2,
  "Actinic Keratosis": 3,
  "Benign Keratosis (Solar Lentigo)": 4,
  "Benign Keratosis": 4,
  Dermatofibroma: 5,
  "Vascular Lesion": 6,
  "Squamous Cell Carcinoma": 7,
  "Benign Displastik Nevüs": 0,
  "Malign Melanom": 1,
  "Bazal Hücreli Karsinom": 2,
  "Aktinik Keratoz": 3,
  "Benign Keratoz": 4,
  Dermatofibrom: 5,
  "Vasküler Lezyon": 6,
  "Skuamöz Hücreli Karsinom": 7,
  "Adipose Tissue": 0,
  Background: 1,
  Debris: 2,
  "Immune Lymphocytes": 3,
  Mucus: 4,
  "Smooth Muscle": 5,
  "Normal Colon Mucosa": 6,
  "Cancer-Associated Stroma": 7,
  "Colorectal Adenocarcinoma Epithelium": 8,
};

export async function fetchEnclaveHandshake(
  baseUrl: string = "http://127.0.0.1:8000"
): Promise<EnclaveHandshakeResponse> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/fl/enclave-handshake`;
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) {
    throw new Error(`Enclave handshake failed with HTTP ${res.status}`);
  }
  return await res.json();
}

export async function fetchSystemState(baseUrl: string = "http://127.0.0.1:8000"): Promise<BackendSystemState> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/state`;
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`Failed to fetch system state (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function fetchHospitalCohorts(baseUrl: string, numHospitals: number = 4, samplesPerHospital: number = 200) {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cohorts/generate`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_hospitals: numHospitals, samples_per_hospital: samplesPerHospital }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || body?.error || `Failed to generate cohorts from backend (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function fetchClinicalDiagnosis(
  baseUrl: string,
  classIndex?: number,
  opacity: number = 0.55,
  colormap: string = "Hot",
  imageBase64?: string,
  language: string = "en",
  metadata?: {
    anatomical_site?: string;
    biopsy_confirmed?: boolean;
    provenance?: string;
  }
): Promise<DiagnosisResult> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/diagnose`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      class_index: classIndex,
      image_base64: imageBase64,
      opacity,
      colormap,
      language,
      ...(metadata?.anatomical_site ? { anatomical_site: metadata.anatomical_site } : {}),
      ...(metadata?.biopsy_confirmed !== undefined ? { biopsy_confirmed: metadata.biopsy_confirmed } : {}),
      ...(metadata?.provenance ? { provenance: metadata.provenance } : {}),
    }),
  });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || body?.error || `Diagnosis failed (HTTP ${res.status})`);
  }
  return await res.json();
}

export async function fetchRagTwins(
  baseUrl: string,
  topK: number = 2
): Promise<{ query_class: number; matched_cases: RagCase[] }> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/rag-similar?top_k=${topK}`;
  const res = await fetch(url, { method: "GET" });
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new Error(body?.detail || body?.error || `RAG retrieval failed (HTTP ${res.status})`);
  }
  return await res.json();
}

/** Build query-string for training params (shared by SSE and polling endpoints). */
export function buildTrainQS(
  optionsOrNumRounds: FederatedTrainingOptions | number,
  localEpochs?: number,
  learningRate?: number,
  simulateAttack?: boolean,
  activateDefense?: boolean,
  algorithm?: string,
  lossFn?: string,
  modelType?: string,
  peftMode?: string,
  secureAggregation?: boolean,
  useCkks?: boolean,
  dpMode?: string,
  dpClippingBound?: number,
  privacyNoise?: number,
  pretrained?: boolean,
  strictCheckpoint?: boolean
): string {
  const opts: FederatedTrainingOptions =
    typeof optionsOrNumRounds === "object"
      ? optionsOrNumRounds
      : {
          numRounds: optionsOrNumRounds,
          localEpochs,
          learningRate,
          simulateAttack,
          activateDefense,
          algorithm,
          lossFn,
          modelType,
          peftMode,
          secureAggregation,
          useCkks,
          dpMode,
          dpClippingBound,
          privacyNoise,
          pretrained,
          strictCheckpoint,
        };

  const params = new URLSearchParams();
  if (opts.numRounds !== undefined) params.set("num_rounds", String(opts.numRounds));
  if (opts.localEpochs !== undefined) params.set("local_epochs", String(opts.localEpochs));
  if (opts.learningRate !== undefined) params.set("learning_rate", String(opts.learningRate));
  if (opts.privacyNoise !== undefined) params.set("privacy_noise", String(opts.privacyNoise));
  if (opts.simulateAttack !== undefined) params.set("simulate_attack", String(opts.simulateAttack));
  if (opts.activateDefense !== undefined) params.set("activate_defense", String(opts.activateDefense));
  if (opts.algorithm) params.set("algorithm", opts.algorithm);
  if (opts.lossFn) params.set("loss_fn", opts.lossFn);
  if (opts.modelType) params.set("model_type", opts.modelType);
  if (opts.peftMode) params.set("peft_mode", opts.peftMode);
  if (opts.secureAggregation !== undefined) params.set("secure_aggregation", String(opts.secureAggregation));
  if (opts.useCkks !== undefined) params.set("use_ckks", String(opts.useCkks));
  if (opts.pretrained !== undefined) params.set("pretrained", String(opts.pretrained));
  if (opts.dpMode) params.set("dp_mode", opts.dpMode);
  if (opts.dpClippingBound !== undefined) params.set("dp_clipping_bound", String(opts.dpClippingBound));
  if (opts.strictCheckpoint !== undefined) params.set("strict_checkpoint", String(opts.strictCheckpoint));

  return params.toString();
}

/**
 * Polling fallback: POST /api/fl/train-start, then poll /api/fl/train-status every 1.5 s.
 * Used automatically when the environment (Render proxy etc.) buffers SSE.
 */
async function pollingFallback(
  base: string, qs: string, numRounds: number,
  onRound: (d: TelemetryRound) => void,
  onComplete: () => void,
  onError: (e: string) => void,
  signal?: AbortSignal
) {
  const startRes = await fetch(`${base}/api/fl/train-start?${qs}`, { method: "POST", signal });
  if (!startRes.ok) { onError(`Training start failed (HTTP ${startRes.status})`); return; }

  let seen = 0;
  while (true) {
    if (signal?.aborted) return;
    await new Promise((r) => setTimeout(r, 1500));
    if (signal?.aborted) return;

    const statusRes = await fetch(`${base}/api/fl/train-status?from_round=${seen}`, { signal });
    if (!statusRes.ok) continue;

    const body = await statusRes.json();
    for (const rd of body.new_rounds as TelemetryRound[]) { onRound(rd); seen++; }
    if (body.error) { onError(body.error); return; }
    if (body.is_complete) { onComplete(); return; }
  }
}

export async function streamFederatedTraining(
  baseUrl: string,
  options: FederatedTrainingOptions,
  onRound: (roundData: TelemetryRound) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const resolvedOptions: FederatedTrainingOptions = {
    numRounds: options.numRounds ?? 5,
    localEpochs: options.localEpochs ?? 2,
    learningRate: options.learningRate ?? 0.0001,
    privacyNoise: options.privacyNoise ?? 0.01,
    simulateAttack: options.simulateAttack ?? false,
    activateDefense: options.activateDefense ?? true,
    algorithm: options.algorithm ?? "FedDyn",
    lossFn: options.lossFn ?? "DAFL",
    modelType: options.modelType ?? "vit_tiny",
    peftMode: options.peftMode ?? "ffa_lora",
    secureAggregation: options.secureAggregation ?? false,
    useCkks: options.useCkks ?? false,
    pretrained: options.pretrained ?? true,
    dpMode: options.dpMode ?? "distributed",
    dpClippingBound: options.dpClippingBound ?? 1.0,
    strictCheckpoint: options.strictCheckpoint ?? false,
  };

  const numRounds = resolvedOptions.numRounds ?? 5;
  const base   = baseUrl.replace(/\/$/, "");
  const qs     = buildTrainQS(resolvedOptions);
  const sseUrl = `${base}/api/fl/train-stream?${qs}`;

  try {
    const response = await fetch(sseUrl, { signal, headers: { Accept: "text/event-stream" } });
    if (!response.ok || !response.body) throw new Error(`SSE stream failed with HTTP ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer       = "";
    let receivedAny  = false;

    // Handshake timeout: If backend/proxy buffers SSE and doesn't send even the initial handshake within 8s, fallback to polling
    let handshakeTimer: NodeJS.Timeout | null = null;
    const handshakeTimeout = new Promise<"timeout">((resolve) => {
      handshakeTimer = setTimeout(() => resolve("timeout"), 8000);
    });

    const consumeSSE = async (): Promise<"done" | "timeout"> => {
      while (true) {
        let result: ReadableStreamReadResult<Uint8Array> | "timeout";
        if (!receivedAny) {
          result = await Promise.race([reader.read(), handshakeTimeout]);
          if (result === "timeout") {
            try { await reader.cancel(); } catch {}
            return "timeout";
          }
          if (handshakeTimer) clearTimeout(handshakeTimer);
        } else {
          result = await reader.read();
        }

        const { done, value } = result as ReadableStreamReadResult<Uint8Array>;
        if (done) return "done";
        buffer += decoder.decode(value, { stream: true });
        receivedAny = true;

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            const jsonStr = trimmed.replace(/^data:\s*/, "");
            try {
              const data: TelemetryRound = JSON.parse(jsonStr);
              if (data.round_num && data.round_num > 0) {
                onRound(data);
              }
              if (data.status === "complete" || (data.round_num && data.round_num >= (resolvedOptions.numRounds ?? 5))) {
                onComplete();
                return "done";
              }
            } catch (e) {
              console.error("Failed to parse SSE line:", trimmed, e);
            }
          }
        }
      }
    };

    const outcome = await consumeSSE();
    if (outcome === "timeout") {
      console.warn("SSE timeout: switching to polling fallback (Render proxy detected)");
      await pollingFallback(base, qs, numRounds, onRound, onComplete, onError, signal);
      return;
    }
    onComplete();
  } catch (err: any) {
    if (signal?.aborted) return;
    console.warn("SSE unavailable, switching to polling fallback:", err.message);
    try {
      await pollingFallback(base, qs, numRounds, onRound, onComplete, onError, signal);
    } catch (pollErr: any) {
      if (!signal?.aborted) onError(pollErr.message ?? String(pollErr));
    }
  }
}

/**
 * Dispatch an operator cancellation request to terminate an active federated training task.
 */
export async function cancelFederatedTraining(baseUrl: string): Promise<TrainingCancelResponse> {
  const base = baseUrl.replace(/\/$/, "");
  const res = await fetch(`${base}/api/fl/train-cancel`, { method: "POST" });
  if (!res.ok) {
    throw new Error(`Failed to cancel training: HTTP ${res.status}`);
  }
  return res.json();
}

// FIX: backend expects `text=` not `diagnosis=`
export function getVoiceBriefingUrl(
  baseUrl: string,
  diagnosis: string = "Normal",
  confidence: number = 95,
  lang: string = "en"
): string {
  const isTr = lang.toLowerCase() === "tr";
  let localizedDiag = diagnosis;
  if (isTr) {
    const dLower = diagnosis.toLowerCase();
    if (dLower.includes("normal")) localizedDiag = "Normal Bulgular";
    else if (dLower.includes("pneumonia") || dLower.includes("zatürre")) localizedDiag = "Zatürre";
    else if (dLower.includes("covid")) localizedDiag = "COVID-19";
    else if (dLower.includes("nevus") || dLower.includes("benign")) localizedDiag = "Benign Displastik Nevüs";
    else if (dLower.includes("melanoma")) localizedDiag = "Malign Melanom";
    else if (dLower.includes("carcinoma")) localizedDiag = "Bazal Hücreli Karsinom";
  }
  const text = isTr
    ? `Yüzde ${Math.round(confidence)} model güveni ile ${localizedDiag} teşhisi doğrulandı. Federe model analizi tamamlandı.`
    : `${diagnosis} diagnosis with ${Math.round(confidence)} percent confidence. Federated model inference complete.`;
  return `${baseUrl.replace(/\/$/, "")}/api/cdss/voice?text=${encodeURIComponent(text)}&lang=${isTr ? "tr" : "en"}`;
}

export interface VoiceEngineStatus {
  service: string;
  cloud_provider: string;
  cache_directory: string;
  cached_files_count: number;
  supported_languages: string[];
  client_offline_synthesis_supported: boolean;
  air_gapped_compatible: boolean;
  status: string;
}

export async function fetchVoiceEngineStatus(baseUrl: string): Promise<VoiceEngineStatus> {
  const res = await fetch(`${baseUrl.replace(/\/$/, "")}/api/cdss/voice/status`);
  if (!res.ok) {
    throw new Error(`Failed to fetch voice engine status: HTTP ${res.status}`);
  }
  return res.json();
}

export function getPdfReportUrl(
  baseUrl: string,
  diagnosis: string = "Normal",
  confidence: number = 95,
  patientId: string = "PT-EHR-7721",
  findings?: string,
  predictedClass?: number,
  lang: string = "en"
): string {
  const predClass = predictedClass ?? (DIAG_TO_CLASS[diagnosis] ?? 0);
  const reportFindings =
    findings ??
    `Federated AI diagnosis: ${diagnosis} (${Math.round(confidence)}% confidence). Evidence-grounded inference via FFA-LoRA + FedDyn.`;
  return (
    `${baseUrl.replace(/\/$/, "")}/api/cdss/report-pdf` +
    `?patient_id=${encodeURIComponent(patientId)}` +
    `&predicted_class=${predClass}` +
    `&confidence=${confidence.toFixed(1)}` +
    `&findings=${encodeURIComponent(reportFindings)}` +
    `&lang=${encodeURIComponent(lang || "en")}`
  );
}
