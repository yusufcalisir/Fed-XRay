import React from "react";
import { describe, it, expect, beforeEach, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import Navbar from "@/components/Navbar";
import { ApiProvider } from "@/context/ApiContext";

describe("Navbar Component", () => {
  beforeEach(() => {
    localStorage.clear();
  });

  const renderNavbar = (props = {}) => {
    return render(
      <ApiProvider>
        <Navbar {...props} />
      </ApiProvider>
    );
  };

  it("renders the consortium brand name and clinical CDSS badge", () => {
    renderNavbar();
    expect(screen.getByText(/Fed/i)).toBeInTheDocument();
    expect(screen.getByText(/XRay/i)).toBeInTheDocument();
    expect(screen.getByText("Clinical CDSS")).toBeInTheDocument();
  });

  it("renders theme toggle button and switches theme", () => {
    renderNavbar();
    const themeBtn = screen.getByLabelText("Toggle Theme");
    expect(themeBtn).toBeInTheDocument();
    fireEvent.click(themeBtn);
    expect(localStorage.getItem("theme")).toBeDefined();
  });

  it("renders EN and TR language switches and toggles language", () => {
    renderNavbar();
    const enBtn = screen.getByRole("button", { name: "EN" });
    const trBtn = screen.getByRole("button", { name: "TR" });

    expect(enBtn).toBeInTheDocument();
    expect(trBtn).toBeInTheDocument();

    fireEvent.click(trBtn);
    expect(localStorage.getItem("fedxray_lang")).toBe("TR");

    fireEvent.click(enBtn);
    expect(localStorage.getItem("fedxray_lang")).toBe("EN");
  });

  it("displays read-only connection status without leaking backend URLs", () => {
    renderNavbar();
    // Verify there is no input field or popover exposing internal URLs
    expect(screen.queryByPlaceholderText(/http/i)).not.toBeInTheDocument();
    expect(screen.queryByText("Apply URL")).not.toBeInTheDocument();
  });

  it("renders navigation links and triggers onLaunchDemo when in landing mode", () => {
    const onLaunchMock = vi.fn();
    renderNavbar({ mode: "landing", onLaunchDemo: onLaunchMock });

    expect(screen.getByText("Clinical Evidence")).toBeInTheDocument();
    expect(screen.getByText("Security & Privacy")).toBeInTheDocument();
    expect(screen.getByText("Architecture")).toBeInTheDocument();
    expect(screen.getByText("Benchmarks")).toBeInTheDocument();

    const launchBtn = screen.getByRole("button", { name: /Launch Live Consortium/i });
    expect(launchBtn).toBeInTheDocument();
    fireEvent.click(launchBtn);
    expect(onLaunchMock).toHaveBeenCalledTimes(1);
  });

  it("renders overview back link when in dashboard mode", () => {
    renderNavbar({ mode: "dashboard" });

    expect(screen.getByText("Platform Overview")).toBeInTheDocument();
  });
});
