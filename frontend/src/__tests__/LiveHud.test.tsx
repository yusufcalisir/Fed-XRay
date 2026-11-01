import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@/test/test-utils";
import LiveHud from "@/components/LiveHud";

describe("LiveHud Component", () => {
  it("renders active consortium banner and node count", () => {
    render(<LiveHud numHospitals={4} isDefenseActive={true} isTrained={false} modelType="vit_base" peftMode="ffa_lora" />);
    expect(screen.getByText("CONSORTIUM ACTIVE")).toBeInTheDocument();
    expect(screen.getByText(/4\/4/i)).toBeInTheDocument();
  });

  it("renders model backbone and PEFT configuration tag", () => {
    render(<LiveHud numHospitals={4} isDefenseActive={true} isTrained={false} modelType="vit_base" peftMode="ffa_lora" />);
    expect(screen.getByText("VIT_BASE (FFA-LORA)")).toBeInTheDocument();
  });

  it("displays Byzantine Defense Shield status correctly", () => {
    const { rerender } = render(<LiveHud numHospitals={4} isDefenseActive={true} isTrained={false} />);
    expect(screen.getByText("Byzantine Defense Shield Active")).toBeInTheDocument();

    rerender(<LiveHud numHospitals={4} isDefenseActive={false} isTrained={false} />);
    expect(screen.getByText("Byzantine Defense Shield: OFF")).toBeInTheDocument();
  });
});
