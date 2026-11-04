import React from "react";
import { describe, it, expect } from "vitest";
import { render, screen } from "@/test/test-utils";
import FederatedTopologyMap from "@/components/FederatedTopologyMap";
import FederatedConsortiumTopology from "@/components/FederatedConsortiumTopology";
import { LanguageProvider } from "@/context/LanguageContext";

describe("FederatedTopologyMap Localization", () => {
  it("renders all four institutions and central aggregator enclave", () => {
    render(<FederatedTopologyMap />);

    // Institutions should be visible
    expect(screen.getByText("BCN-20000")).toBeInTheDocument();
    expect(screen.getByText("ViDIR")).toBeInTheDocument();
    expect(screen.getByText("Queensland")).toBeInTheDocument();
    expect(screen.getByText("Beth Israel")).toBeInTheDocument();

    // Aggregator Title
    expect(screen.getByText("CONSORTIUM AGGREGATOR")).toBeInTheDocument();
  });

  it("renders in Turkish when LanguageContext is set to TR", () => {
    // Force Turkish language in wrapper
    const TurkishWrapper = ({ children }: { children: React.ReactNode }) => {
      return <LanguageProvider initialLanguage="TR">{children}</LanguageProvider>;
    };

    render(
      <TurkishWrapper>
        <FederatedTopologyMap />
      </TurkishWrapper>
    );

    // Aggregator in Turkish
    expect(screen.getByText("KONSORSİYUM TOPLAYICI")).toBeInTheDocument();
    expect(screen.getByText("RLWE CKKS 128-Bit Güvenli Alan")).toBeInTheDocument();
    expect(screen.getByText("Sıfır Açık Metin Hasta Verisi Transferi")).toBeInTheDocument();

    // Localized modalities and cities
    expect(screen.getByText(/Barselona : Dermoskopi/)).toBeInTheDocument();
    expect(screen.getByText(/Viyana : Histopatoloji/)).toBeInTheDocument();

    // DP budget labels in Turkish
    const dpLabels = screen.getAllByText("DP Bütçesi:");
    expect(dpLabels.length).toBe(4);
  });
});

describe("FederatedConsortiumTopology Localization", () => {
  it("renders localized bottom telemetry ribbon in Turkish", () => {
    const TurkishWrapper = ({ children }: { children: React.ReactNode }) => {
      return <LanguageProvider initialLanguage="TR">{children}</LanguageProvider>;
    };

    render(
      <TurkishWrapper>
        <FederatedConsortiumTopology onLaunchDemo={() => {}} />
      </TurkishWrapper>
    );

    // Selected node details in Turkish
    expect(screen.getByText("BCN-20000 Cilt Kanseri Merkezi")).toBeInTheDocument();
    expect(screen.getByText("Barselona, ES")).toBeInTheDocument();
    expect(screen.getByText("Siemens Sensation Optik Dermatoskop")).toBeInTheDocument();
    expect(screen.getByText("Simüle Et")).toBeInTheDocument();
    expect(screen.getByText("Etiket Dağılımı:")).toBeInTheDocument();

    // Stream status indicator visible in normal state
    expect(screen.getByText("RLWE CKKS 128-Bit Şifreli Akış Aktif")).toBeInTheDocument();
  });
});
