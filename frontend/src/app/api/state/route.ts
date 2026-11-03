import { NextResponse } from "next/server";
import authenticCohorts from "@/lib/authenticCohortsCompact.json";

export const dynamic = "force-dynamic";

export async function GET() {
  const envUrl = process.env.NEXT_PUBLIC_API_URL?.trim().replace(/\/$/, "");
  const isDead = Boolean(
    envUrl && (
      envUrl.includes("globe-celebration-poems-shakira") ||
      envUrl.includes("trycloudflare.com") ||
      envUrl.includes("placeholder")
    )
  );
  const defaultBackend =
    process.env.NODE_ENV === "production"
      ? "https://fed-xray-api.onrender.com"
      : "http://127.0.0.1:8000";
  const targetBackend = isDead || !envUrl ? defaultBackend : envUrl;

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 1500);
    const resp = await fetch(`${targetBackend}/api/state`, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      return NextResponse.json(data);
    }
  } catch {
    // Standby fallback
  }

  return NextResponse.json({
    model_trained: false,
    model_type: "vit_tiny",
    peft_mode: "ffa_lora",
    rounds: [],
    reference_cases: [],
    cohorts: authenticCohorts,
  });
}
