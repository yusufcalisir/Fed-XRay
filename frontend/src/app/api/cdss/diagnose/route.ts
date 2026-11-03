import { NextRequest, NextResponse } from "next/server";

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

  let body = {};
  try {
    body = await request.json();
  } catch {
    // Empty body fallback
  }

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 60000);
    const resp = await fetch(`${targetBackend}/api/cdss/diagnose`, {
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
    const errBody = await resp.json().catch(() => null);
    return NextResponse.json(
      { detail: errBody?.detail || `Diagnosis inference failed with HTTP ${resp.status}` },
      { status: resp.status }
    );
  } catch (err: any) {
    return NextResponse.json(
      {
        detail: `CDSS inference backend unreachable at ${targetBackend}. If hosted on Render free tier, the container may be waking from cold sleep (takes ~45-60s). Please retry shortly.`,
      },
      { status: 503 }
    );
  }
}
