import React from "react";
import { describe, it, expect, vi, beforeEach } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import ConsortiumLaunchOverlay from "@/components/ConsortiumLaunchOverlay";
import * as nextNavigation from "next/navigation";

describe("ConsortiumLaunchOverlay Component", () => {
  const pushMock = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    global.fetch = vi.fn().mockImplementation(async (url: any) => {
      const u = typeof url === "string" ? url : url?.toString() || "";
      if (u.includes("/api/health")) {
        return {
          ok: true,
          status: 200,
          json: async () => ({ status: "healthy", version: "2.1.0" }),
        };
      }
      return {
        ok: true,
        status: 200,
        json: async () => ({
          status: "verified",
          nodes_online: 4,
          total_nodes: 4,
          dp_budget: { epsilon: 2.0, delta: 1e-5 },
        }),
      };
    });
    vi.spyOn(nextNavigation, "useRouter").mockReturnValue({
      push: pushMock,
      replace: vi.fn(),
      prefetch: vi.fn(),
      back: vi.fn(),
      forward: vi.fn(),
      refresh: vi.fn(),
    } as any);
  });

  it("renders null when isOpen is false", () => {
    const { container } = render(
      <ConsortiumLaunchOverlay isOpen={false} onClose={vi.fn()} />
    );
    expect(container).toBeEmptyDOMElement();
  });

  it("renders overlay title, description, and steps when isOpen is true", () => {
    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={vi.fn()} />
    );

    expect(screen.getByText("Initializing Federated Consortium Handshake")).toBeInTheDocument();
    expect(screen.getByText(/Establishing secure cryptographic communication channels/i)).toBeInTheDocument();
    expect(screen.getByText("Federated Handshake")).toBeInTheDocument();
  });

  it("calls onClose when the close button is clicked", () => {
    const onCloseMock = vi.fn();
    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={onCloseMock} />
    );

    const closeBtn = screen.getByRole("button", { name: "Close" });
    expect(closeBtn).toBeInTheDocument();
    fireEvent.click(closeBtn);
    expect(onCloseMock).toHaveBeenCalledTimes(1);
  });

  it("invokes router.push to /dashboard when skip button is clicked", () => {
    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={vi.fn()} />
    );

    const skipBtn = screen.getByText("Enter Immediately");
    expect(skipBtn).toBeInTheDocument();
    fireEvent.click(skipBtn);
    expect(pushMock).toHaveBeenCalledWith("/dashboard");
  });

  it("displays verified node enclaves and homomorphic key exchange steps", () => {
    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={vi.fn()} />
    );

    expect(screen.getByText("Hospital Cryptographic Nodes Verified (4/4 Nodes Online)")).toBeInTheDocument();
    expect(screen.getByText("128-bit RLWE CKKS Homomorphic Key Exchange Complete")).toBeInTheDocument();
    expect(screen.getByText("FFA-LoRA Parameter Subspace Synchronized")).toBeInTheDocument();
  });

  it("renders honest error alert and cloud / standby buttons when enclave probe fails in strict mode", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Network failure on all candidate endpoints"));

    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={vi.fn()} strict={true} />
    );

    const errorHeading = await screen.findByText("Enclave Verification Handshake Failed");
    expect(errorHeading).toBeInTheDocument();
    expect(screen.getByText("Proceed to Dashboard (Standby Mode)")).toBeInTheDocument();
    expect(screen.getByText("Connect Cloud Backend (Render)")).toBeInTheDocument();
    expect(screen.getByText("Connect Localhost (8000)")).toBeInTheDocument();
  });

  it("smoothly plays launch demo animation without displaying error barrier when offline in demo mode", async () => {
    global.fetch = vi.fn().mockRejectedValue(new Error("Network offline"));

    render(
      <ConsortiumLaunchOverlay isOpen={true} onClose={vi.fn()} strict={false} />
    );

    // Zero error banners in standard demo launch mode
    expect(screen.queryByText("Enclave Verification Handshake Failed")).not.toBeInTheDocument();
    expect(screen.getByText("Consortium Server")).toBeInTheDocument();
  });
});
