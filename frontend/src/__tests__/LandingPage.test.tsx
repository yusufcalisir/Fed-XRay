import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import PresentationLandingPage from "@/app/page";
import { ApiProvider } from "@/context/ApiContext";

describe("PresentationLandingPage Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderLandingPage = () => {
    return render(
      <ApiProvider>
        <PresentationLandingPage />
      </ApiProvider>
    );
  };

  it("renders hero headline, badge, and all 5 verifiable KPI metrics", () => {
    renderLandingPage();

    expect(screen.getByText("Federated Medical AI Platform")).toBeInTheDocument();
    expect(screen.getByText("Train AI Models Across Hospitals Without Sharing Patient Data")).toBeInTheDocument();

    // 5 KPI Metrics
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Hospital Nodes")).toBeInTheDocument();

    expect(screen.getByText("0 B")).toBeInTheDocument();
    expect(screen.getByText("Patient Data Shared")).toBeInTheDocument();

    expect(screen.getByText("128-Bit")).toBeInTheDocument();
    expect(screen.getByText("Encryption")).toBeInTheDocument();

    expect(screen.getByText("98.4%")).toBeInTheDocument();
    expect(screen.getAllByText("Diagnostic AUC")[0]).toBeInTheDocument();

    expect(screen.getByText("ε ≤ 2.0")).toBeInTheDocument();
    expect(screen.getByText("Privacy Budget")).toBeInTheDocument();
  });

  it("renders interactive results showcase tabs and switches between tabs", () => {
    renderLandingPage();

    // Default Tab: 3D lung workspace (anatomical model, no fabricated patient sample)
    expect(screen.getAllByText("Platform Capabilities")[0]).toBeInTheDocument();
    expect(screen.getAllByTestId("lung-viewer-3d").length).toBeGreaterThan(0);
    expect(screen.queryByText(/P-8821/)).not.toBeInTheDocument();

    // Switch to RAG Tab
    const ragTabBtn = screen.getByRole("button", { name: "Case Retrieval" });
    fireEvent.click(ragTabBtn);
    expect(screen.getByText(/Case-Based Reasoning: Digital Twin Matcher/i)).toBeInTheDocument();
    expect(screen.getByText("REF-ONCO-401")).toBeInTheDocument();
    expect(screen.getByText("REF-ONCO-402")).toBeInTheDocument();

    // Switch to Convergence Tab
    const convTabBtn = screen.getByRole("button", { name: "Training Monitor" });
    fireEvent.click(convTabBtn);
    expect(screen.getByText(/Multi-Center Federated Convergence & Robustness Log/i)).toBeInTheDocument();
    expect(screen.getByText("Round 1")).toBeInTheDocument();
    expect(screen.getByText(/Intercepted Node 3 Poisoning/i)).toBeInTheDocument();
    expect(screen.getByText("Round 5 (Final)")).toBeInTheDocument();

    // Switch to Report & Audio Tab
    const reportTabBtn = screen.getByRole("button", { name: "Reporting" });
    fireEvent.click(reportTabBtn);
    expect(screen.getByText("Single-Page A4 Medical Intelligence Export")).toBeInTheDocument();
    expect(screen.getByText("Hands-Free Synthesized Clinical Dictation")).toBeInTheDocument();
  });

  it("explains the 3D saliency pipeline with four steps and no static diagnosis", () => {
    renderLandingPage();

    expect(screen.getByText("Frozen medical foundation encoder")).toBeInTheDocument();
    expect(screen.getByText("Grad-CAM on layer L-3 attention")).toBeInTheDocument();
    expect(screen.getByText("Projection onto the lung model")).toBeInTheDocument();
    expect(screen.getByText("Clinician review")).toBeInTheDocument();
    expect(screen.queryByRole("slider")).not.toBeInTheDocument();
  });

  it("renders 4 security governance pillars and Strategy E benchmark matrix", () => {
    renderLandingPage();

    // 4 Security Pillars
    expect(screen.getByText("Zero Patient Data Leakage")).toBeInTheDocument();
    expect(screen.getByText("Patient-Level Differential Privacy")).toBeInTheDocument();
    expect(screen.getByText("Cryptographic Secure Aggregation")).toBeInTheDocument();
    expect(screen.getByText("Byzantine Resilience & Poison Defense")).toBeInTheDocument();

    // Strategy E Benchmark Matrix
    expect(screen.getByText("Performance Comparison")).toBeInTheDocument();
    expect(screen.getByText("Fed-XRay (FFA-LoRA + FedDyn)")).toBeInTheDocument();
    expect(screen.getByText("Centralized Upper Bound")).toBeInTheDocument();
    expect(screen.getByText("Traditional FedAvg (Full ViT)")).toBeInTheDocument();
  });

  it("opens consortium launch overlay when launch demo button is clicked", () => {
    renderLandingPage();

    // Multiple launch buttons exist (hero, navbar, banner)
    const launchBtns = screen.getAllByRole("button", { name: /Launch Live (Consortium|Demo)/i });
    expect(launchBtns.length).toBeGreaterThan(0);

    fireEvent.click(launchBtns[0]);

    // Modal should now be open
    expect(screen.getByText("Initializing Federated Consortium Enclave")).toBeInTheDocument();
    expect(screen.getByText("Federated Handshake")).toBeInTheDocument();
  });
});
