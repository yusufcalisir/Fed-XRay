import { describe, it, expect, vi, beforeEach } from "vitest";
import {
  getPdfReportUrl,
  getVoiceBriefingUrl,
  fetchHospitalCohorts,
  fetchClinicalDiagnosis,
  fetchRagTwins,
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
});
