import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@/test/test-utils";
import RagDigitalTwins from "@/components/RagDigitalTwins";
import { RagCase } from "@/types";

const mockTwins: RagCase[] = [
  {
    case_id: "REF-ONCO-401",
    label: 1,
    label_name: "Pneumonia",
    similarity: 97.4,
    history: "Biopsy confirmed bacterial lobar consolidation. Successful resolution under 14-day levofloxacin course.",
  },
  {
    case_id: "REF-ONCO-402",
    label: 1,
    label_name: "Pneumonia",
    similarity: 93.1,
    history: "Hospital-acquired pneumonia. Responded positively to IV ampicillin-sulbactam therapy.",
  },
];

describe("RagDigitalTwins Component", () => {
  it("returns null when no twins are provided", () => {
    const { container } = render(<RagDigitalTwins twins={[]} />);
    expect(container).toBeEmptyDOMElement();
  });

  it("renders verified digital twins with cosine similarity and clinical histories", () => {
    render(<RagDigitalTwins twins={mockTwins} />);

    expect(screen.getByText("Case-Based Reasoning: Digital Twin Matcher (Federated RAG)")).toBeInTheDocument();
    expect(screen.getByText("REF-ONCO-401")).toBeInTheDocument();
    expect(screen.getByText("97.4%")).toBeInTheDocument();
    expect(screen.getByText(/Biopsy confirmed bacterial lobar consolidation/i)).toBeInTheDocument();

    expect(screen.getByText("REF-ONCO-402")).toBeInTheDocument();
    expect(screen.getByText("93.1%")).toBeInTheDocument();
  });
});
