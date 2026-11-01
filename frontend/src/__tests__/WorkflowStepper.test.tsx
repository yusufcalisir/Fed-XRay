import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import WorkflowStepper from "@/components/WorkflowStepper";

describe("WorkflowStepper Component", () => {
  it("renders all 4 clinical pipeline stages", () => {
    render(
      <WorkflowStepper
        hasIngested={false}
        isTraining={false}
        isTrained={false}
        hasDiagnosis={false}
      />
    );
    expect(screen.getByText("1. Data Ingestion")).toBeInTheDocument();
    expect(screen.getByText("2. Federated Training")).toBeInTheDocument();
    expect(screen.getByText("3. Diagnostic Inference")).toBeInTheDocument();
    expect(screen.getByText("4. CDSS & Reports")).toBeInTheDocument();
  });

  it("indicates correct completion progress count", () => {
    const { rerender } = render(
      <WorkflowStepper hasIngested={false} isTraining={false} isTrained={false} hasDiagnosis={false} />
    );
    expect(screen.getByText(/0 \/ 4/i)).toBeInTheDocument();

    rerender(<WorkflowStepper hasIngested={true} isTraining={false} isTrained={false} hasDiagnosis={false} />);
    expect(screen.getByText(/1 \/ 4/i)).toBeInTheDocument();

    rerender(<WorkflowStepper hasIngested={true} isTraining={false} isTrained={true} hasDiagnosis={false} />);
    expect(screen.getByText(/2 \/ 4/i)).toBeInTheDocument();

    rerender(<WorkflowStepper hasIngested={true} isTraining={false} isTrained={true} hasDiagnosis={true} />);
    expect(screen.getByText(/4 \/ 4/i)).toBeInTheDocument();
  });

  it("calls onScrollToStep when clicking unlocked step", () => {
    const onScrollMock = vi.fn();
    render(
      <WorkflowStepper
        hasIngested={true}
        isTraining={false}
        isTrained={true}
        hasDiagnosis={false}
        onScrollToStep={onScrollMock}
      />
    );

    const step1 = screen.getByText("1. Data Ingestion");
    fireEvent.click(step1);
    expect(onScrollMock).toHaveBeenCalledWith("section-ingestion");
  });
});
