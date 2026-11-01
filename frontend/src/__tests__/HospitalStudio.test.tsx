import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import HospitalStudio from "@/components/HospitalStudio";
import { HospitalCohort } from "@/types";

const mockHospitals: HospitalCohort[] = [
  {
    hospital_id: 1,
    name: "General Hospital",
    num_samples: 300,
    distribution: [0.5, 0.33, 0.17],
    counts: { normal: 150, pneumonia: 100, covid: 50 },
    sample_images: [Array(28).fill(Array(28).fill(0.5))],
    sample_labels: [0],
  },
  {
    hospital_id: 2,
    name: "Metropolitan Hospital",
    num_samples: 400,
    distribution: [0.25, 0.625, 0.125],
    counts: { normal: 100, pneumonia: 250, covid: 50 },
    sample_images: [Array(28).fill(Array(28).fill(0.8))],
    sample_labels: [1],
  },
];

describe("HospitalStudio Component", () => {
  it("enforces ZERO duplicate button invariant in empty state", () => {
    const onGenerateMock = vi.fn();
    render(<HospitalStudio hospitals={[]} onGenerate={onGenerateMock} isLoading={false} />);

    // In empty state, exactly ONE primary call to action must be visible
    const buttons = screen.getAllByRole("button");
    const generateButtons = buttons.filter((btn) =>
      btn.textContent?.includes("Generate Multi-Hospital Cohorts") ||
      btn.textContent?.includes("Çok Merkezli Hasta Kohortlarını Üret")
    );
    expect(generateButtons.length).toBe(1);

    // Click triggers onGenerate
    fireEvent.click(generateButtons[0]);
    expect(onGenerateMock).toHaveBeenCalledTimes(1);
  });

  it("renders hospital tabs and demographics when cohorts are populated", () => {
    render(<HospitalStudio hospitals={mockHospitals} onGenerate={vi.fn()} isLoading={false} />);

    // Tabs display shortened names with node numbers
    expect(screen.getByText("General")).toBeInTheDocument();
    expect(screen.getByText("Metropolitan")).toBeInTheDocument();

    // Active Hospital Full Title
    expect(screen.getByText("General Hospital")).toBeInTheDocument();

    // Re-Ingest button in header
    expect(screen.getByText(/Re-Ingest Cohorts/i)).toBeInTheDocument();

    // Success banner
    expect(screen.getByText(/Hospital Data Ingestion Complete/i)).toBeInTheDocument();
  });

  it("switches active hospital tab on click", () => {
    render(<HospitalStudio hospitals={mockHospitals} onGenerate={vi.fn()} isLoading={false} />);

    const hospital2Tab = screen.getByText("Metropolitan");
    fireEvent.click(hospital2Tab);

    // Active hospital card updates to Metropolitan Hospital
    expect(screen.getByText("Metropolitan Hospital")).toBeInTheDocument();
  });
});
