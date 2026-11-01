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
    expect(href).toContain("patient_id=PAT-001");
  });
});
