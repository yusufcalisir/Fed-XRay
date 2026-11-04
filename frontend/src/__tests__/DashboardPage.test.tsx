import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent, waitFor } from "@/test/test-utils";
import DashboardPage from "@/app/dashboard/page";
import { ApiProvider } from "@/context/ApiContext";

describe("DashboardPage Component (AUDIT-DEF-02 Integrity)", () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  const renderDashboard = () => {
    return render(
      <ApiProvider>
        <DashboardPage />
      </ApiProvider>
    );
  };

  it("renders empty state and honest error notification when backend cohort generation fails", async () => {
    global.fetch = vi.fn().mockImplementation((url: string) => {
      const urlStr = String(url);
      if (urlStr.includes("/api/state")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: "ok",
            version: "2.1.0",
            cohorts: [],
            hospitals: [],
            rounds: [],
            reference_cases: [],
          }),
        } as any);
      }
      if (urlStr.includes("/api/cohorts/generate")) {
        return Promise.resolve({
          ok: false,
          status: 503,
          json: async () => ({
            detail: "Hospital node cluster offline or unreachable",
          }),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok" }),
      } as any);
    });

    renderDashboard();

    // Verify that the error message is displayed and failure is not masked with fake cohorts
    expect(
      await screen.findByText(/Cohort generation failed.*Hospital node cluster offline/i)
    ).toBeInTheDocument();

    // Verify empty state is rendered honestly instead of fabricated mock hospital cards
    expect(screen.getByText("Multi-Hospital Data Ingestion Required")).toBeInTheDocument();
  });

  it("renders authentic hospital cohorts when backend returns populated cohorts", async () => {
    const mockHospitals = [
      {
        hospital_id: 1,
        name: "Hospital Clinic Barcelona (BCN_20000)",
        num_samples: 250,
        distribution: [0.6, 0.25, 0.15],
        counts: { normal: 150, pneumonia: 62, covid: 38 },
        sample_images: [],
        sample_labels: [0],
      },
      {
        hospital_id: 2,
        name: "Medical University of Vienna (ViDIR)",
        num_samples: 300,
        distribution: [0.3, 0.5, 0.2],
        counts: { normal: 90, pneumonia: 150, covid: 60 },
        sample_images: [],
        sample_labels: [1],
      },
    ];

    global.fetch = vi.fn().mockImplementation((url: string) => {
      const urlStr = String(url);
      if (urlStr.includes("/api/state")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: "ok",
            version: "2.1.0",
            cohorts: mockHospitals,
            rounds: [],
            reference_cases: [],
          }),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok" }),
      } as any);
    });

    renderDashboard();

    expect(await screen.findByText("Hospital Clinic Barcelona (BCN_20000)")).toBeInTheDocument();
    expect(screen.getByText(/Hospital Data Ingestion Complete/i)).toBeInTheDocument();
    // With sample_images being empty, it should honestly show the empty thumbnail indicator
    expect(screen.getByText("No thumbnail samples available")).toBeInTheDocument();
  });

  it("allows retrying cohort generation from empty state button", async () => {
    let callCount = 0;
    global.fetch = vi.fn().mockImplementation((url: string) => {
      const urlStr = String(url);
      if (urlStr.includes("/api/state")) {
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            status: "ok",
            version: "2.1.0",
            cohorts: [],
            hospitals: [],
            rounds: [],
            reference_cases: [],
          }),
        } as any);
      }
      if (urlStr.includes("/api/cohorts/generate")) {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            ok: false,
            status: 500,
            json: async () => ({ detail: "Database locked on node cluster" }),
          } as any);
        }
        return Promise.resolve({
          ok: true,
          status: 200,
          json: async () => ({
            success: true,
            hospitals: [
              {
                hospital_id: 1,
                name: "Charite Berlin Center",
                num_samples: 180,
                distribution: [0.5, 0.3, 0.2],
                counts: { normal: 90, pneumonia: 54, covid: 36 },
                sample_images: [],
                sample_labels: [0],
              },
            ],
          }),
        } as any);
      }
      return Promise.resolve({
        ok: true,
        status: 200,
        json: async () => ({ status: "ok" }),
      } as any);
    });

    renderDashboard();

    // First auto-fetch fails
    expect(
      await screen.findByText(/Cohort generation failed.*Database locked/i)
    ).toBeInTheDocument();

    // User clicks the partition cohorts button to retry
    const retryBtn = screen.getByRole("button", { name: /Generate Multi-Hospital Cohorts/i });
    fireEvent.click(retryBtn);

    // Second call succeeds and renders authentic cohort
    expect(await screen.findByText("Charite Berlin Center")).toBeInTheDocument();
  });
});
