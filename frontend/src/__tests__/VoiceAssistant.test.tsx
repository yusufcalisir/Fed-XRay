import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen, fireEvent } from "@/test/test-utils";
import VoiceAssistant from "@/components/VoiceAssistant";
import { ApiProvider } from "@/context/ApiContext";
import { LanguageProvider } from "@/context/LanguageContext";

describe("VoiceAssistant Component", () => {
  it("renders voice assistant title and audio player element in English", () => {
    render(
      <ApiProvider>
        <VoiceAssistant diagnosisName="Pneumonia" confidence={94.8} />
      </ApiProvider>
    );

    expect(screen.getByText("Hands-Free Voice Diagnostic Assistant")).toBeInTheDocument();
    const audioEl = document.querySelector("audio");
    expect(audioEl).toBeInTheDocument();
    expect(audioEl?.getAttribute("src")).toContain("lang=en");
  });

  it("renders Turkish titles and audio url parameters when language is TR", () => {
    render(
      <LanguageProvider initialLanguage="TR">
        <ApiProvider>
          <VoiceAssistant diagnosisName="Pneumonia" confidence={94.8} />
        </ApiProvider>
      </LanguageProvider>
    );

    expect(screen.getByText("Sesli Klinik Asistan ve Dikte Desteği")).toBeInTheDocument();
    const audioEl = document.querySelector("audio");
    expect(audioEl).toBeInTheDocument();
    expect(audioEl?.getAttribute("src")).toContain("lang=tr");
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
    expect(screen.getByText("Speak via Local Browser")).toBeInTheDocument();
  });

  it("triggers local browser speech synthesis when speak button is clicked", () => {
    let spokenText = "";
    class MockUtterance {
      text: string;
      constructor(text: string) {
        this.text = text;
      }
    }
    const mockSpeak = (utterance: any) => {
      spokenText = utterance?.text || "";
    };
    (window as any).SpeechSynthesisUtterance = MockUtterance;
    (window as any).speechSynthesis = {
      speak: mockSpeak,
      cancel: () => {},
    };

    render(
      <ApiProvider>
        <VoiceAssistant diagnosisName="Melanoma" confidence={92.4} />
      </ApiProvider>
    );

    const audioEl = document.querySelector("audio");
    if (audioEl) {
      fireEvent.error(audioEl);
    }

    const speakBtn = screen.getByText("Speak via Local Browser");
    fireEvent.click(speakBtn);
    expect(spokenText).toContain("Melanoma diagnosis with 92 percent confidence");
  });
});
