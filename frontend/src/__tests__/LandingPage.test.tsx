import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import PresentationLandingPage from "@/app/page";
import { ApiProvider } from "@/context/ApiContext";

describe("PresentationLandingPage Component", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn().mockResolvedValue({
      ok: true,
      status: 200,
      json: async () => ({
        status: "ok",
        version: "2.1.0",
        nodes_online: 4,
        total_nodes: 4,
      }),
    } as any);
  });

  const renderLandingPage = () => {
    return render(
      <ApiProvider>
        <PresentationLandingPage />
      </ApiProvider>
    );
  };

  it("renders hero headline, badge, and verifiable KPI metrics in standby state", () => {
    renderLandingPage();

    expect(screen.getByText("Privacy-Preserving Federated Learning (PPFL)")).toBeInTheDocument();
    expect(screen.getByText("Privacy-Preserving Federated Learning for Multi-Hospital Medical AI")).toBeInTheDocument();

    // Verifiable KPI Metrics in Standby Mode
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("Federated Nodes")).toBeInTheDocument();

    expect(screen.getByText("0 B")).toBeInTheDocument();
    expect(screen.getByText("Raw Data Transferred")).toBeInTheDocument();

    expect(screen.getByText("128-Bit")).toBeInTheDocument();
    expect(screen.getByText("Homomorphic Encryption")).toBeInTheDocument();

    // In standby mode, un-trained metrics must show standby indicators "--" without hardcoded values
    expect(screen.getByText("Consensus AUC (Standby)")).toBeInTheDocument();
    expect(screen.getByText("Bandwidth Efficiency (Standby)")).toBeInTheDocument();
    expect(screen.getAllByText("--").length).toBeGreaterThanOrEqual(2);
    expect(screen.queryByText("98.4%")).not.toBeInTheDocument();

    expect(screen.getByText("ε ≤ 2.0")).toBeInTheDocument();
    expect(screen.getByText("Privacy Budget")).toBeInTheDocument();
  });

  it("renders live consensus accuracy and bandwidth efficiency when live training rounds exist", async () => {
    vi.mocked(global.fetch).mockImplementation((url: any) => {
      const urlStr = String(url);
      if (urlStr.includes("/api/state")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            is_live: true,
            status: "training",
            history_rounds: 5,
            model_type: "vit",
            peft_mode: "ffa-lora",
            hospitals: [{ id: "h1" }, { id: "h2" }, { id: "h3" }, { id: "h4" }],
            rounds: [
              { round_num: 1, train_accuracy: 72.1 },
              { round_num: 5, train_accuracy: 94.8 },
            ],
            privacy: { epsilon: 1.45 },
          }),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          status: "ok",
          version: "2.1.0",
          nodes_online: 4,
          total_nodes: 4,
        }),
      } as any);
    });

    renderLandingPage();

    // Live training metrics should be rendered after state fetch
    expect(await screen.findByText("94.8%")).toBeInTheDocument();
    expect(screen.getByText("Validation Accuracy")).toBeInTheDocument();
    expect(screen.getByText("99.6%")).toBeInTheDocument();
    expect(screen.getByText("Bandwidth Efficiency")).toBeInTheDocument();
    expect(screen.getByText("ε = 1.45")).toBeInTheDocument();
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
    expect(screen.getByText("Federated Reference Bank Standby")).toBeInTheDocument();
    expect(screen.queryByText("REF-ONCO-401")).not.toBeInTheDocument();
    expect(screen.queryByText("REF-ONCO-402")).not.toBeInTheDocument();

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
    expect(screen.getByText("Initializing Federated Consortium Handshake")).toBeInTheDocument();
    expect(screen.getByText("Federated Handshake")).toBeInTheDocument();
  });

  it("renders authentic federated reference cases when reference_cases are returned from backend", async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      const urlStr = String(url);
      if (urlStr.includes("/api/state")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: "ok",
            version: "2.1.0",
            nodes_online: 4,
            total_nodes: 4,
            reference_cases: [
              {
                case_id: "REF-HIST-089",
                diagnosis: "Malignant Melanoma",
                similarity: 94.2,
                clinical_notes: "Atypical melanocytic proliferation at dermo-epidermal junction",
                site_id: "Hospital Clinic Barcelona (BCN_20000)",
                biopsy_confirmed: true,
              },
            ],
          }),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({
          status: "ok",
          version: "2.1.0",
          nodes_online: 4,
          total_nodes: 4,
        }),
      } as any);
    });

    renderLandingPage();

    const ragTabBtn = screen.getByRole("button", { name: "Case Retrieval" });
    fireEvent.click(ragTabBtn);

    expect(await screen.findByText("REF-HIST-089")).toBeInTheDocument();
    expect(screen.getByText("94.2% Match")).toBeInTheDocument();
    expect(screen.getByText("Malignant Melanoma")).toBeInTheDocument();
    expect(screen.getByText(/Atypical melanocytic proliferation/)).toBeInTheDocument();
  });
});
