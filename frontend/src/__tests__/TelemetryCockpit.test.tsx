import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import TelemetryCockpit from "@/components/TelemetryCockpit";
import { TelemetryRound } from "@/types";

const mockTelemetry: TelemetryRound[] = [
  {
    round_num: 1,
    total_rounds: 5,
    train_loss: 0.95,
    train_accuracy: 70.0,
    test_loss: 0.85,
    test_accuracy: 72.5,
    precision: 0.73,
    recall: 0.72,
    f1_score: 0.725,
    threat_detected: false,
    blocked_nodes: [],
    status: "completed",
    model_type: "vit_base",
    peft_mode: "ffa_lora",
  },
  {
    round_num: 2,
    total_rounds: 5,
    train_loss: 0.48,
    train_accuracy: 88.0,
    test_loss: 0.42,
    test_accuracy: 89.1,
    precision: 0.89,
    recall: 0.89,
    f1_score: 0.89,
    threat_detected: true,
    blocked_nodes: [3],
    status: "completed",
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
      modelType: "vit_tiny",
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

