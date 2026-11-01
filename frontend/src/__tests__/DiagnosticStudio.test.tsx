import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import DiagnosticStudio from "@/components/DiagnosticStudio";
import { DiagnosisResult } from "@/types";

const mockDiagnosis: DiagnosisResult = {
  predicted_class: 1,
  predicted_name: "Pneumonia",
  confidence: 94.8,
  true_class: 1,
  true_name: "Pneumonia",
  raw_image: Array(28).fill(Array(28).fill(0.6)),
  heatmap: Array(28).fill(Array(28).fill(0.85)),
  heatmap_available: true,
  probabilities: [0.03, 0.948, 0.022],
  patient_id: "PAT-TEST-001",
  image_source: "holdout_cohort",
  findings: "Localized dense alveolar consolidation in right lower lobe.",
};

describe("DiagnosticStudio Component", () => {
  it("renders locked state when model is not yet trained", () => {
    const onScrollMock = vi.fn();
    render(
      <DiagnosticStudio
        diagnosis={null}
        onDiagnose={vi.fn()}
        isLoading={false}
        isLocked={true}
        onScrollToTraining={onScrollMock}
      />
    );

    expect(screen.getByText("Step 3 Locked")).toBeInTheDocument();
    expect(screen.getByText(/Diagnostic Studio & Grad-CAM Locked/i)).toBeInTheDocument();

    const btn = screen.getByRole("button", { name: /Go to Federated Training/i });
    fireEvent.click(btn);
    expect(onScrollMock).toHaveBeenCalledTimes(1);
  });

  it("enforces ZERO duplicate button invariant in empty state", () => {
    const onDiagnoseMock = vi.fn();
    render(
      <DiagnosticStudio
        diagnosis={null}
        onDiagnose={onDiagnoseMock}
        isLoading={false}
        isLocked={false}
      />
    );

    // In empty state, exactly ONE run diagnosis CTA is rendered (in the center card)
    const runButtons = screen.getAllByRole("button", { name: /Execute Radiological AI Analysis/i });
    expect(runButtons.length).toBe(1);

    // Filter toolbar is accessible
    expect(screen.getByText("Random / Any")).toBeInTheDocument();
    expect(screen.getByText("Normal Parenchyma")).toBeInTheDocument();
    expect(screen.getByText("Pneumonia")).toBeInTheDocument();
    expect(screen.getByText("COVID-19")).toBeInTheDocument();
    expect(screen.getByText("Upload Scan")).toBeInTheDocument();

    fireEvent.click(runButtons[0]);
    expect(onDiagnoseMock).toHaveBeenCalledTimes(1);
  });

  it("renders dual-pane visual evidence and class probabilities when diagnosis exists", () => {
    render(
      <DiagnosticStudio
        diagnosis={mockDiagnosis}
        onDiagnose={vi.fn()}
        isLoading={false}
        isLocked={false}
      />
    );

    // Diagnosis Banner
    expect(screen.getAllByText("Pneumonia").length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText(/94.8%/i)).toBeInTheDocument();
    expect(screen.getByText("PAT-TEST-001")).toBeInTheDocument();

    // Dual-pane titles
    expect(screen.getByText("Standard Chest Radiograph")).toBeInTheDocument();
    expect(screen.getByText("Grad-CAM Saliency Attention Map")).toBeInTheDocument();

    // Clinical Findings
    expect(screen.getByText(/Localized dense alveolar consolidation/i)).toBeInTheDocument();
  });
});
