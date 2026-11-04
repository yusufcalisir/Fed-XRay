import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  getPdfReportUrl,
  getVoiceBriefingUrl,
  fetchHospitalCohorts,
  fetchClinicalDiagnosis,
  fetchRagTwins,
  buildTrainQS,
  streamFederatedTraining,
} from "@/lib/api";

describe("Frontend API Client Functions", () => {
  beforeEach(() => {
    vi.restoreAllMocks();
  });

  it("constructs correct PDF report URL with encoded parameters", () => {
    const url = getPdfReportUrl(
      "http://127.0.0.1:8000",
      "Pneumonia",
      95.2,
      "PAT-1234",
      "Bacterial opacity observed.",
      1,
      "en"
    );

    expect(url).toContain("http://127.0.0.1:8000/api/cdss/report-pdf");
    expect(url).toContain("patient_id=PAT-1234");
    expect(url).toContain("predicted_class=1");
    expect(url).toContain("confidence=95.2");
    expect(url).toContain("findings=Bacterial%20opacity%20observed.");
    expect(url).toContain("lang=en");
  });

  it("correctly maps ISIC dermoscopy and Turkish class names in getPdfReportUrl", () => {
    const urlBcc = getPdfReportUrl("http://127.0.0.1:8000", "Basal Cell Carcinoma", 88.0);
    expect(urlBcc).toContain("predicted_class=2");
    expect(urlBcc).toContain("lang=en");

    const urlTrBcc = getPdfReportUrl("http://127.0.0.1:8000", "Bazal Hücreli Karsinom", 88.0, "PAT-01", "", undefined, "tr");
    expect(urlTrBcc).toContain("predicted_class=2");
    expect(urlTrBcc).toContain("lang=tr");

    const urlMel = getPdfReportUrl("http://127.0.0.1:8000", "Malignant Melanoma", 72.5);
    expect(urlMel).toContain("predicted_class=1");

    const urlNev = getPdfReportUrl("http://127.0.0.1:8000", "Benign Displastik Nevüs", 96.0);
    expect(urlNev).toContain("predicted_class=0");
  });

  it("constructs correct voice briefing audio URL", () => {
    const url = getVoiceBriefingUrl("http://127.0.0.1:8000", "COVID-19", 91.5);
    expect(url).toContain("http://127.0.0.1:8000/api/cdss/voice");
    expect(url).toContain("text=COVID-19%20diagnosis%20with%2092%20percent%20confidence");
    expect(url).toContain("lang=en");

    const urlTr = getVoiceBriefingUrl("http://127.0.0.1:8000", "Pneumonia", 94.8, "tr");
    expect(urlTr).toContain("lang=tr");
    expect(urlTr).toContain(encodeURIComponent("Yüzde 95 model güveni"));
    expect(urlTr).toContain(encodeURIComponent("Zatürre"));
  });

  it("fetchHospitalCohorts throws descriptive error on network failure", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Network connection refused"));

    await expect(fetchHospitalCohorts("http://127.0.0.1:8000")).rejects.toThrow(
      /Network connection refused/i
    );
  });

  it("fetchClinicalDiagnosis sends POST with payload and receives diagnosis", async () => {
    const mockDiagnosisResult = {
      predicted_class: 0,
      predicted_name: "Normal Parenchyma",
      confidence: 98.2,
      true_class: 0,
      true_name: "Normal Parenchyma",
      raw_image: [[0.1, 0.2]],
      heatmap: [[0.5, 0.6]],
      heatmap_available: true,
      probabilities: [0.982, 0.01, 0.008],
      patient_id: "PAT-NORMAL-01",
      image_source: "holdout_cohort",
      findings: "Clear lung fields bilaterally.",
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockDiagnosisResult,
    } as any);

    const res = await fetchClinicalDiagnosis("http://127.0.0.1:8000", 0, 0.6, "Jet");
    expect(res.predicted_name).toBe("Normal Parenchyma");
    expect(res.confidence).toBe(98.2);
    expect(global.fetch).toHaveBeenCalledWith(
      "http://127.0.0.1:8000/api/cdss/diagnose",
      expect.objectContaining({
        method: "POST",
        headers: { "Content-Type": "application/json" },
      })
    );
  });

  it("fetchRagTwins queries similarity endpoint and retrieves matched twins", async () => {
    const mockTwins = {
      matched_cases: [
        {
          case_id: "REF-001",
          label: 0,
          label_name: "Normal Parenchyma",
          similarity: 99.1,
          history: "Healthy screening examination.",
        },
      ],
    };

    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      json: async () => mockTwins,
    } as any);

    const res = await fetchRagTwins("http://127.0.0.1:8000", 2);
    expect(res.matched_cases.length).toBe(1);
    expect(res.matched_cases[0].case_id).toBe("REF-001");
  });

  it("buildTrainQS serializes complete Option J cryptographic flags and hyperparameters", () => {
    const qs = buildTrainQS({
      numRounds: 10,
      localEpochs: 3,
      learningRate: 0.00005,
      simulateAttack: true,
      activateDefense: true,
      algorithm: "FedDyn",
      lossFn: "DAFL",
      modelType: "uni",
      peftMode: "ffa_lora",
      secureAggregation: true,
      useCkks: true,
      dpMode: "distributed",
      dpClippingBound: 1.5,
      privacyNoise: 0.05,
      pretrained: true,
      strictCheckpoint: false,
    });

    const parsed = new URLSearchParams(qs);
    expect(parsed.get("num_rounds")).toBe("10");
    expect(parsed.get("local_epochs")).toBe("3");
    expect(parsed.get("learning_rate")).toBe("0.00005");
    expect(parsed.get("simulate_attack")).toBe("true");
    expect(parsed.get("activate_defense")).toBe("true");
    expect(parsed.get("algorithm")).toBe("FedDyn");
    expect(parsed.get("loss_fn")).toBe("DAFL");
    expect(parsed.get("model_type")).toBe("uni");
    expect(parsed.get("peft_mode")).toBe("ffa_lora");
    expect(parsed.get("secure_aggregation")).toBe("true");
    expect(parsed.get("use_ckks")).toBe("true");
    expect(parsed.get("dp_mode")).toBe("distributed");
    expect(parsed.get("dp_clipping_bound")).toBe("1.5");
    expect(parsed.get("privacy_noise")).toBe("0.05");
    expect(parsed.get("pretrained")).toBe("true");
  });

  it("streamFederatedTraining transmits full cryptographic query string to training endpoints", async () => {
    let requestedUrl = "";
    global.fetch = vi.fn().mockImplementation((url: string) => {
      requestedUrl = url;
      return Promise.resolve({
        ok: false,
        status: 500,
        body: null,
      });
    });

    const onErrorMock = vi.fn();
    await streamFederatedTraining(
      "http://127.0.0.1:8000",
      {
        numRounds: 5,
        modelType: "fedmedclip",
        algorithm: "FedDyn",
        peftMode: "fedsa_lora",
        lossFn: "LDAM",
        secureAggregation: true,
        useCkks: true,
        dpMode: "distributed",
        dpClippingBound: 2.0,
      },
      vi.fn(),
      vi.fn(),
      onErrorMock
    );

    expect(requestedUrl).toContain("/api/fl/train-");
    expect(requestedUrl).toContain("secure_aggregation=true");
    expect(requestedUrl).toContain("use_ckks=true");
    expect(requestedUrl).toContain("peft_mode=fedsa_lora");
    expect(requestedUrl).toContain("loss_fn=LDAM");
    expect(requestedUrl).toContain("dp_mode=distributed");
    expect(requestedUrl).toContain("dp_clipping_bound=2");
  });
});
