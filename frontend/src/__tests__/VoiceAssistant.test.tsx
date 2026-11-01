import React from "react";
import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import VoiceAssistant from "@/components/VoiceAssistant";
import { ApiProvider } from "@/context/ApiContext";

describe("VoiceAssistant Component", () => {
  it("renders voice assistant title and audio player element", () => {
    render(
      <ApiProvider>
        <VoiceAssistant diagnosisName="Pneumonia" confidence={94.8} />
      </ApiProvider>
    );

    expect(screen.getByText("Hands-Free Voice Diagnostic Assistant")).toBeInTheDocument();
    // HTML5 audio element
    const audioEl = document.querySelector("audio");
    expect(audioEl).toBeInTheDocument();
  });

  it("handles audio load error gracefully and renders retry action", () => {
    render(
      <ApiProvider>
        <VoiceAssistant diagnosisName="Pneumonia" confidence={94.8} />
      </ApiProvider>
    );

    const audioEl = document.querySelector("audio");
    expect(audioEl).toBeInTheDocument();

    // Trigger error event on audio
    if (audioEl) {
      fireEvent.error(audioEl);
    }

    // Verify retry button appears
    expect(screen.getByText("Retry")).toBeInTheDocument();
  });
});
