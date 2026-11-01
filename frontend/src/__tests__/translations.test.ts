import { describe, it, expect } from "vitest";
import { TRANSLATIONS } from "@/lib/translations";

describe("Medical Localization and Translation Integrity", () => {
  it("should have identical key parity between English and Turkish", () => {
    const enKeys = Object.keys(TRANSLATIONS.EN).sort();
    const trKeys = Object.keys(TRANSLATIONS.TR).sort();

    expect(enKeys.length).toBeGreaterThan(0);
    expect(trKeys.length).toBeGreaterThan(0);

    const missingInTr = enKeys.filter((key) => !(key in TRANSLATIONS.TR));
    const missingInEn = trKeys.filter((key) => !(key in TRANSLATIONS.EN));

    expect(missingInTr).toEqual([]);
    expect(missingInEn).toEqual([]);
    expect(enKeys).toEqual(trKeys);
  });

  it("should have non-empty translation values for every key", () => {
    for (const [key, value] of Object.entries(TRANSLATIONS.EN)) {
      expect(value, `EN translation for ${key} should not be empty`).toBeTruthy();
      expect(value.trim().length, `EN translation for ${key} should have text`).toBeGreaterThan(0);
    }

    for (const [key, value] of Object.entries(TRANSLATIONS.TR)) {
      expect(value, `TR translation for ${key} should not be empty`).toBeTruthy();
      expect(value.trim().length, `TR translation for ${key} should have text`).toBeGreaterThan(0);
    }
  });

  it("should correctly define essential clinical terms in both languages", () => {
    expect(TRANSLATIONS.EN.sec1_normal).toBe("Normal Parenchyma");
    expect(TRANSLATIONS.TR.sec1_normal).toBe("Normal Parankim");

    expect(TRANSLATIONS.EN.sec1_pneumonia).toBe("Pneumonia");
    expect(TRANSLATIONS.TR.sec1_pneumonia).toContain("Pnömoni");

    expect(TRANSLATIONS.EN.sec1_covid).toBe("COVID-19");
    expect(TRANSLATIONS.TR.sec1_covid).toContain("COVID-19");
  });

  it("should define single-responsibility CTA buttons in both languages", () => {
    expect(TRANSLATIONS.EN.sec1_btn_generate).toBe("Generate Multi-Hospital Cohorts");
    expect(TRANSLATIONS.TR.sec1_btn_generate).toBe("Çok Merkezli Hasta Kohortlarını Üret");

    expect(TRANSLATIONS.EN.sec1_btn_reingest).toBe("Re-Ingest Cohorts");
    expect(TRANSLATIONS.TR.sec1_btn_reingest).toBe("Kohortları Yeniden Yükle");

    expect(TRANSLATIONS.EN.sec2_btn_start).toBe("Start Federated Training Round");
    expect(TRANSLATIONS.TR.sec2_btn_start).toBe("Federe Öğrenme Eğitim Turunu Başlat");

    expect(TRANSLATIONS.EN.sec3_btn_run_diag).toBe("Execute Radiological AI Analysis");
    expect(TRANSLATIONS.TR.sec3_btn_run_diag).toBe("Radyolojik Yapay Zeka Analizini Çalıştır");
  });
});
