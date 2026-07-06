import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

// Upstash Redis REST API
const KV_URL = process.env.KV_REST_API_URL!;
const KV_TOKEN = process.env.KV_REST_API_TOKEN!;

// GitHub private repo
const GH_OWNER = process.env.SIGNS_GITHUB_OWNER!;
const GH_REPO = process.env.SIGNS_GITHUB_REPO!;
const GH_BRANCH = process.env.SIGNS_GITHUB_BRANCH || "main";
const GH_PAT = process.env.SIGNS_GITHUB_PAT!;

// Cache TTL: 7 days (signs don't change often)
const CACHE_TTL = 7 * 24 * 60 * 60;

// --- Redis helpers ---
async function kvGet(key: string): Promise<string | null> {
  try {
    const res = await fetch(`${KV_URL}/get/${encodeURIComponent(key)}`, {
      headers: { Authorization: `Bearer ${KV_TOKEN}` },
    });
    if (!res.ok) return null;
    const data = await res.json();
    return data.result || null;
  } catch {
    return null;
  }
}

async function kvSet(key: string, value: string, ttl: number): Promise<void> {
  try {
    await fetch(`${KV_URL}/set/${encodeURIComponent(key)}/${encodeURIComponent(value)}?EX=${ttl}`, {
      headers: { Authorization: `Bearer ${KV_TOKEN}` },
    });
  } catch {}
}

// --- GitHub raw content fetch ---
async function fetchFromGitHub(gloss: string): Promise<string | null> {
  const url = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${GH_BRANCH}/${gloss}.json`;
  try {
    const res = await fetch(url, {
      headers: { Authorization: `token ${GH_PAT}` },
    });
    if (!res.ok) return null;
    return await res.text();
  } catch {
    return null;
  }
}

// --- Route handler ---
export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ gloss: string }> }
) {
  const { gloss } = await params;
  const key = gloss.toUpperCase().replace(/[^A-Z0-9_-]/g, "");

  if (!key) {
    return NextResponse.json({ error: "Invalid gloss" }, { status: 400, headers: { "Access-Control-Allow-Origin": "*" } });
  }

  // 1. Check Redis cache
  const cached = await kvGet(`sign:${key}`);
  if (cached) {
    return new NextResponse(cached, {
      status: 200,
      headers: {
        "Content-Type": "application/json",
        "Cache-Control": "public, max-age=86400, s-maxage=604800",
        "X-Cache": "HIT",
        "Access-Control-Allow-Origin": "*",
      },
    });
  }

  // 2. Fetch from GitHub private repo
  const data = await fetchFromGitHub(key);
  if (!data) {
    return NextResponse.json({ error: "Sign not found" }, { status: 404, headers: { "Access-Control-Allow-Origin": "*" } });
  }

  // 3. Cache in Redis (async, don't block response)
  kvSet(`sign:${key}`, data, CACHE_TTL);

  return new NextResponse(data, {
    status: 200,
    headers: {
      "Content-Type": "application/json",
      "Cache-Control": "public, max-age=86400, s-maxage=604800",
      "X-Cache": "MISS",
      "Access-Control-Allow-Origin": "*",
    },
  });
}

// CORS preflight
export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      "Access-Control-Allow-Origin": "*",
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
