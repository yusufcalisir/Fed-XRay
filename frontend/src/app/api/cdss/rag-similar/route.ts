import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(request: NextRequest) {
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

  const searchParams = request.nextUrl.searchParams;
  const topK = searchParams.get("top_k") || "2";

  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 15000);
    const resp = await fetch(`${targetBackend}/api/cdss/rag-similar?top_k=${topK}`, {
      signal: controller.signal,
      cache: "no-store",
    });
    clearTimeout(timeout);
    if (resp.ok) {
      const data = await resp.json();
      return NextResponse.json(data);
    }
    const errBody = await resp.json().catch(() => null);
    return NextResponse.json(
      { detail: errBody?.detail || `RAG retrieval failed with HTTP ${resp.status}` },
      { status: resp.status }
    );
  } catch {
    return NextResponse.json(
      {
        query_class: 0,
        matched_cases: [],
      },
      { status: 200 }
    );
  }
}

