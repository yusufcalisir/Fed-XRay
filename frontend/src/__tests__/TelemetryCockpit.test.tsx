import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import TelemetryCockpit from "@/components/TelemetryCockpit";
import { TelemetryRound } from "@/types";

const mockTelemetry: TelemetryRound[] = [
  {
    round_num: 1,
    test_loss: 0.85,
    test_accuracy: 72.5,
    threat_detected: false,
    blocked_nodes: [],
    loss_reduction_pct: 12.0,
    model_type: "vit_base",
    peft_mode: "ffa_lora",
  },
  {
    round_num: 2,
    test_loss: 0.42,
    test_accuracy: 89.1,
    threat_detected: true,
    blocked_nodes: [3],
    loss_reduction_pct: 35.0,
    model_type: "vit_base",
    peft_mode: "ffa_lora",
  },
];

describe("TelemetryCockpit Component", () => {
  it("renders locked container with unlock action when isLocked is true", () => {
    const onUnlockMock = vi.fn();
    render(
      <TelemetryCockpit
        history={[]}
        onStartTraining={vi.fn()}
        isTraining={false}
        currentRound={0}
        totalRounds={5}
        isLocked={true}
        onUnlock={onUnlockMock}
      />
    );

    expect(screen.getByText("Step 2 Locked")).toBeInTheDocument();
    expect(screen.getByText(/Federated Training Cockpit Locked/i)).toBeInTheDocument();

    const unlockBtn = screen.getByRole("button", { name: /Ingest Hospital Cohorts/i });
    fireEvent.click(unlockBtn);
    expect(onUnlockMock).toHaveBeenCalledTimes(1);
  });

  it("renders dedicated hyperparameter toolbar card when unlocked", () => {
    const onStartMock = vi.fn();
    render(
      <TelemetryCockpit
        history={[]}
        onStartTraining={onStartMock}
        isTraining={false}
        currentRound={0}
        totalRounds={5}
        isLocked={false}
      />
    );

    // Header has start button
    const startBtn = screen.getByRole("button", { name: /Start Federated Training Round/i });
    expect(startBtn).toBeInTheDocument();

    // Toolbar has model, algo, and attack controls
    expect(screen.getByText("Model Backbone:")).toBeInTheDocument();
    expect(screen.getByText("Optimizer:")).toBeInTheDocument();
    expect(screen.getByText(/Simulate Byzantine Attack/i)).toBeInTheDocument();

    // Toggle attack and click start
    const attackCheckbox = screen.getByRole("checkbox");
    fireEvent.click(attackCheckbox);
    expect(attackCheckbox).toBeChecked();

    fireEvent.click(startBtn);
    expect(onStartMock).toHaveBeenCalledWith({
      modelType: "vit_base",
      algorithm: "FedDyn",
      simulateAttack: true,
    });
  });

  it("renders Byzantine defense threat alert when poisoning is detected", () => {
    render(
      <TelemetryCockpit
        history={mockTelemetry}
        onStartTraining={vi.fn()}
        isTraining={false}
        currentRound={2}
        totalRounds={5}
        isLocked={false}
      />
    );

    expect(screen.getByText(/ALERT: Byzantine Threat Intercepted/i)).toBeInTheDocument();
    expect(screen.getByText(/Node\(s\) 3/i)).toBeInTheDocument();
  });
});
