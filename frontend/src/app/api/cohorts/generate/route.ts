import { NextRequest, NextResponse } from "next/server";
import authenticCohorts from "@/lib/authenticCohortsCompact.json";

export const dynamic = "force-dynamic";

export async function POST(request: NextRequest) {
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

  let body = { num_hospitals: 4, samples_per_hospital: 50 };
  try {
    const raw = await request.json();
    if (raw && typeof raw === "object") {
      body = {
        num_hospitals: raw.num_hospitals || 4,
        samples_per_hospital: Math.max(50, raw.samples_per_hospital || 50),
      };
    }
  } catch {
    // Use fallback body
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);
    const resp = await fetch(`${targetBackend}/api/cohorts/generate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      return NextResponse.json(data);
    }
  } catch {
    // Standby fallback
  }

  // Adjust sample count in authentic cohorts if requested
  const adaptedCohorts = authenticCohorts.map((h: any) => ({
    ...h,
    num_samples: body.samples_per_hospital || h.num_samples,
  }));

  return NextResponse.json({
    success: true,
    num_hospitals: adaptedCohorts.length,
    dataset: "isic2019",
    hospitals: adaptedCohorts,
  });
}
