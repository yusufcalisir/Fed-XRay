import React from "react";
import "@testing-library/jest-dom/vitest";
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import HospitalStudio from "../components/HospitalStudio";
import RagDigitalTwins from "../components/RagDigitalTwins";
import TelemetryCockpit from "../components/TelemetryCockpit";
import RootRouteError from "../app/error";
import GlobalError from "../app/global-error";
import { AllTheProviders } from "../test/test-utils";
import { HospitalCohort, RagCase } from "@/types";

describe("Client Subtree Isolation and Defensive Rendering Suite", () => {
  it("renders HospitalStudio without crashing when sample_images is null, undefined, or empty", () => {
    const malformedHospitals = [
      {
        id: 0,
        name: "Test Hospital",
        num_samples: 100,
        class_distribution: { 0: 50, 1: 50 },
        sample_images: undefined as unknown as number[][][],
        scanner_type: "Digital X-Ray",
        data_source: "isic_archive_v2_real_clinical",
        is_synthetic: false,
      },
    ];

    render(
      <AllTheProviders>
        <HospitalStudio
          hospitals={malformedHospitals as unknown as HospitalCohort[]}
          onGenerate={vi.fn()}
          isLoading={false}
        />
      </AllTheProviders>
    );

    expect(screen.getByText("Test Hospital")).toBeInTheDocument();
    expect(screen.getByText("No thumbnail samples available")).toBeInTheDocument();
  });

  it("handles RagDigitalTwins gracefully without throwing when twins is undefined, null, or empty", () => {
    const { container: container1 } = render(
      <AllTheProviders>
        <RagDigitalTwins twins={undefined as unknown as RagCase[]} />
      </AllTheProviders>
    );
    expect(container1.firstChild).toBeNull();

    const sampleTwin = [
      {
        case_id: "PAT-ISIC-0099",
        label_id: 1,
        label_name: "Malignant Melanoma",
        similarity: 94.5,
        site_id: "BCN_20000",
        history: "Biopsy verified",
        biopsy_confirmed: true,
      },
    ];

    const { getByText } = render(
      <AllTheProviders>
        <RagDigitalTwins twins={sampleTwin as unknown as RagCase[]} />
      </AllTheProviders>
    );
    expect(getByText("PAT-ISIC-0099")).toBeInTheDocument();
    expect(getByText("Malignant Melanoma")).toBeInTheDocument();
  });

  it("renders TelemetryCockpit without crashing when history is null or empty", () => {
    render(
      <AllTheProviders>
        <TelemetryCockpit
          history={undefined as unknown as []}
          onStartTraining={vi.fn()}
          isTraining={false}
          currentRound={0}
          totalRounds={5}
          isLocked={false}
          onUnlock={vi.fn()}
        />
      </AllTheProviders>
    );

    expect(screen.getByText(/Start a training round to view live convergence telemetry/i)).toBeInTheDocument();
  });

  it("renders Next.js app/error.tsx with clinical recovery buttons and reset handler", () => {
    const resetFn = vi.fn();
    const testError = new Error("Simulated Next.js router segment crash");

    render(<RootRouteError error={testError} reset={resetFn} />);

    expect(screen.getByText("Clinical Interface Protection Active")).toBeInTheDocument();
    expect(screen.getByText(/Simulated Next.js router segment crash/)).toBeInTheDocument();
    expect(screen.getByText("Recover View")).toBeInTheDocument();
  });

  it("renders Next.js app/global-error.tsx root recovery boundary", () => {
    const resetFn = vi.fn();
    const testError = new Error("Simulated root layout crash");

    render(<GlobalError error={testError} reset={resetFn} />);

    expect(screen.getByText("Fed-XRay Root Runtime Protection Active")).toBeInTheDocument();
    expect(screen.getByText("Recover Application")).toBeInTheDocument();
  });
});
