import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import ClinicalErrorBoundary from "../components/ClinicalErrorBoundary";

const ThrowingComponent = ({ shouldThrow }: { shouldThrow: boolean }) => {
  if (shouldThrow) {
    throw new Error("Synthetic client render crash test");
  }
  return <div>Clinical View Running Smoothly</div>;
};

describe("ClinicalErrorBoundary Component", () => {
  it("renders children when no rendering exception occurs", () => {
    render(
      <ClinicalErrorBoundary>
        <ThrowingComponent shouldThrow={false} />
      </ClinicalErrorBoundary>
    );

    expect(screen.getByText("Clinical View Running Smoothly")).toBeInTheDocument();
  });

  it("catches rendering errors and displays medical grade recovery screen in English", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ClinicalErrorBoundary language="en">
        <ThrowingComponent shouldThrow={true} />
      </ClinicalErrorBoundary>
    );

    expect(screen.getByText("Clinical Interface Protection Active")).toBeInTheDocument();
    expect(screen.getByText(/Synthetic client render crash test/)).toBeInTheDocument();
    expect(screen.getByText("Retry Component")).toBeInTheDocument();

    consoleSpy.mockRestore();
  });

  it("displays medical grade recovery screen in Turkish when language is tr", () => {
    const consoleSpy = vi.spyOn(console, "error").mockImplementation(() => {});

    render(
      <ClinicalErrorBoundary language="tr">
        <ThrowingComponent shouldThrow={true} />
      </ClinicalErrorBoundary>
    );

    expect(screen.getByText("Klinik Arayüz Koruma Kalkanı Devrede")).toBeInTheDocument();
    expect(screen.getByText("Bileşeni Yeniden Dene")).toBeInTheDocument();

    consoleSpy.mockRestore();
  });
});
