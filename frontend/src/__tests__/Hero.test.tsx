import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@/test/test-utils";
import Hero from "@/components/Hero";

describe("Hero Component", () => {
  it("renders consortium title and description", () => {
    render(<Hero numHospitals={4} totalRounds={5} totalSamples={1200} isTrained={false} />);
    expect(screen.getByText(/Fed-XRay/i)).toBeInTheDocument();
  });

  it("renders all 4 clinical KPI metric cards with correct values", () => {
    render(<Hero numHospitals={4} totalRounds={5} totalSamples={14250} isTrained={false} />);
    expect(screen.getByText("4")).toBeInTheDocument();
    expect(screen.getByText("5")).toBeInTheDocument();
    expect(screen.getByText("14,250")).toBeInTheDocument();
    expect(screen.getByText("Ready")).toBeInTheDocument();
  });

  it("updates status badge to Trained when isTrained is true", () => {
    render(<Hero numHospitals={4} totalRounds={5} totalSamples={14250} isTrained={true} />);
    expect(screen.getByText("Trained")).toBeInTheDocument();
  });
});
