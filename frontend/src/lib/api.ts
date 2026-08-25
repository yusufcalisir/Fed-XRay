import { HospitalCohort, DiagnosisResult, RagCase, TelemetryRound } from "@/types";

const DIAG_TO_CLASS: Record<string, number> = {
  Normal: 0, Pneumonia: 1, COVID: 2, "COVID-19": 2,
};

export async function fetchHospitalCohorts(baseUrl: string, numHospitals: number = 4, samplesPerHospital: number = 200) {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cohorts/generate`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ num_hospitals: numHospitals, samples_per_hospital: samplesPerHospital }),
  });
  if (!res.ok) throw new Error(`Failed to generate cohorts from backend (HTTP ${res.status})`);
  return await res.json();
}

export async function fetchClinicalDiagnosis(
  baseUrl: string,
  classIndex?: number,
  opacity: number = 0.55,
  colormap: string = "Hot"
): Promise<DiagnosisResult> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/diagnose`;
  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ class_index: classIndex, opacity, colormap }),
  });
  if (!res.ok) throw new Error(`Diagnosis failed (HTTP ${res.status})`);
  return await res.json();
}

// FIX: backend expects GET, not POST
export async function fetchRagTwins(
  baseUrl: string,
  queryClass: number = 1
): Promise<{ query_class: number; matched_cases: RagCase[] }> {
  const url = `${baseUrl.replace(/\/$/, "")}/api/cdss/rag-similar?query_class=${queryClass}`;
  const res = await fetch(url, { method: "GET" });
  if (!res.ok) throw new Error(`RAG retrieval failed (HTTP ${res.status})`);
  return await res.json();
}

/** Build query-string for training params (shared by SSE and polling endpoints). */
function buildTrainQS(
  numRounds: number, localEpochs: number, learningRate: number,
  simulateAttack: boolean, activateDefense: boolean,
  algorithm: string, lossFn: string, modelType: string, peftMode: string
): string {
  return (
    `num_rounds=${numRounds}` +
    `&local_epochs=${localEpochs}` +
    `&learning_rate=${learningRate}` +
    `&simulate_attack=${simulateAttack}` +
    `&activate_defense=${activateDefense}` +
    `&algorithm=${algorithm}` +
    `&loss_fn=${lossFn}` +
    `&model_type=${modelType}` +
    `&peft_mode=${peftMode}`
  );
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
  options: {
    numRounds?: number; localEpochs?: number; learningRate?: number;
    simulateAttack?: boolean; activateDefense?: boolean;
    algorithm?: string; lossFn?: string; modelType?: string; peftMode?: string;
  },
  onRound: (roundData: TelemetryRound) => void,
  onComplete: () => void,
  onError: (error: string) => void,
  signal?: AbortSignal
): Promise<void> {
  const numRounds      = options.numRounds      ?? 5;
  const localEpochs    = options.localEpochs    ?? 2;
  const learningRate   = options.learningRate   ?? 0.0001;
  const simulateAttack = options.simulateAttack ?? false;
  const activateDefense= options.activateDefense?? true;
  const algorithm      = options.algorithm      ?? "FedDyn";
  const lossFn         = options.lossFn         ?? "DAFL";
  const modelType      = options.modelType      ?? "vit_tiny";
  const peftMode       = options.peftMode       ?? "ffa_lora";

  const base   = baseUrl.replace(/\/$/, "");
  const qs     = buildTrainQS(numRounds, localEpochs, learningRate, simulateAttack, activateDefense, algorithm, lossFn, modelType, peftMode);
  const sseUrl = `${base}/api/fl/train-stream?${qs}`;

  try {
    const response = await fetch(sseUrl, { signal, headers: { Accept: "text/event-stream" } });
    if (!response.ok || !response.body) throw new Error(`SSE stream failed with HTTP ${response.status}`);

    const reader  = response.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let buffer       = "";
    let receivedAny  = false;
    const sseTimeout = new Promise<"timeout">((res) => setTimeout(() => res("timeout"), 4000));

    const consumeSSE = async (): Promise<"done" | "timeout"> => {
      while (true) {
        const readPromise = reader.read();
        const result = await Promise.race([readPromise, sseTimeout]);

        if (result === "timeout") {
          if (!receivedAny) return "timeout";
          const { done, value } = await readPromise;
          if (done) return "done";
          buffer += decoder.decode(value, { stream: true });
        } else {
          const { done, value } = result as ReadableStreamReadResult<Uint8Array>;
          if (done) return "done";
          buffer += decoder.decode(value, { stream: true });
          receivedAny = true;
        }

        const lines = buffer.split("\n");
        buffer = lines.pop() || "";
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith("data:")) {
            const jsonStr = trimmed.replace(/^data:\s*/, "");
            try {
              const data: TelemetryRound = JSON.parse(jsonStr);
              onRound(data);
              if (data.status === "complete" || data.round_num >= numRounds) { onComplete(); return "done"; }
            } catch (e) { console.error("Failed to parse SSE line:", trimmed, e); }
          }
        }
      }
    }

    const outcome = await consumeSSE();
    if (outcome === "timeout") {
      console.warn("SSE timeout — switching to polling fallback (Render proxy detected)");
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

// FIX: backend expects `text=` not `diagnosis=`
export function getVoiceBriefingUrl(
  baseUrl: string,
  diagnosis: string = "Normal",
  confidence: number = 95,
  lang: string = "en"
): string {
  const text = `${diagnosis} diagnosis with ${Math.round(confidence)} percent confidence. Federated model inference complete.`;
  return `${baseUrl.replace(/\/$/, "")}/api/cdss/voice?text=${encodeURIComponent(text)}&lang=${lang}`;
}

// FIX: backend expects `predicted_class=` and `confidence=` not `diagnosis=`
export function getPdfReportUrl(
  baseUrl: string,
  diagnosis: string = "Normal",
  confidence: number = 95
): string {
  const predicted_class = DIAG_TO_CLASS[diagnosis] ?? 0;
  const findings = `Federated AI diagnosis: ${diagnosis} (${Math.round(confidence)}% confidence). Evidence-grounded inference via FFA-LoRA + FedDyn.`;
  return (
    `${baseUrl.replace(/\/$/, "")}/api/cdss/report-pdf` +
    `?predicted_class=${predicted_class}` +
    `&confidence=${confidence.toFixed(1)}` +
    `&findings=${encodeURIComponent(findings)}`
  );
}
