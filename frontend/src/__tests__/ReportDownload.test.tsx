import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@/test/test-utils";
import ReportDownload from "@/components/ReportDownload";
import { ApiProvider } from "@/context/ApiContext";

describe("ReportDownload Component", () => {
  it("renders report card and download link with encoded URL parameters", () => {
    render(
      <ApiProvider>
        <ReportDownload
          diagnosisName="Pneumonia"
          confidence={94.8}
          patientId="PAT-001"
          findings="Dense consolidation"
          predictedClass={1}
        />
      </ApiProvider>
    );

    expect(screen.getByText("Official Single-Page Medical Intelligence Report")).toBeInTheDocument();
    
    const downloadLink = screen.getByRole("link", { name: /Download Official Diagnostic Report/i });
    expect(downloadLink).toBeInTheDocument();
    expect(downloadLink).toHaveAttribute("target", "_blank");

    const href = downloadLink.getAttribute("href");
    expect(href).toContain("/api/cdss/report-pdf");
    expect(href).toContain("patient_id=PAT-001");
    expect(href).toContain("confidence=94.8");
    expect(href).toContain("predicted_class=1");
    expect(href).toContain("findings=Dense%20consolidation");
    expect(href).toContain("lang=en");
  });

  it("encodes ISIC dermoscopy diagnosis and Turkish language parameter properly", () => {
    render(
      <ApiProvider>
        <ReportDownload
          diagnosisName="Bazal Hücreli Karsinom"
          confidence={40.0}
          patientId="PAT-ISIC-0099"
          findings="Model tahmini: Bazal Hücreli Karsinom"
          predictedClass={2}
        />
      </ApiProvider>
    );

    const downloadLink = screen.getByRole("link", { name: /Download Official Diagnostic Report/i });
    expect(downloadLink).toBeInTheDocument();

    const href = downloadLink.getAttribute("href");
    expect(href).toContain("patient_id=PAT-ISIC-0099");
    expect(href).toContain("confidence=40.0");
    expect(href).toContain("predicted_class=2");
    expect(href).toContain("findings=Model%20tahmini%3A%20Bazal%20H%C3%BCcreli%20Karsinom");
  });

  it("provides sensible defaults when optional props are omitted", () => {
    render(
      <ApiProvider>
        <ReportDownload
          diagnosisName="Basal Cell Carcinoma"
          confidence={88.2}
        />
      </ApiProvider>
    );

    const downloadLink = screen.getByRole("link", { name: /Download Official Diagnostic Report/i });
    expect(downloadLink).toBeInTheDocument();

    const href = downloadLink.getAttribute("href");
    expect(href).toContain("patient_id=PAT-CDSS-LIVE");
    expect(href).toContain("predicted_class=2");
    expect(href).toContain("confidence=88.2");
  });
});
