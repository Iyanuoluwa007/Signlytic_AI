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
// Distinguishes "this sign does not exist" from "GitHub is throttling us".
// Returning null for both made a rate limit look identical to a missing sign,
// so the client fingerspelled a word it has a perfectly good clip for, and the
// logs gave no hint that anything was wrong.
type FetchResult =
  | { ok: true; data: string }
  | { ok: false; throttled: boolean };

async function fetchFromGitHub(gloss: string): Promise<FetchResult> {
  const url = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${GH_BRANCH}/${gloss}.json`;
  const attempt = () =>
    fetch(url, { headers: { Authorization: `token ${GH_PAT}` } });

  try {
    let res = await attempt();

    for (let tries = 0; res.status === 429 && tries < 2; tries++) {
      const retryAfter = Number(res.headers.get("retry-after"));
      const waitMs =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? Math.min(retryAfter * 1000, 2000)
          : 400 * (tries + 1);
      await new Promise((r) => setTimeout(r, waitMs));
      res = await attempt();
    }

    if (res.ok) return { ok: true, data: await res.text() };
    if (res.status === 429) {
      console.error(`[ERR] signs upstream 429 for ${gloss}`);
      return { ok: false, throttled: true };
    }
    return { ok: false, throttled: false };
  } catch {
    return { ok: false, throttled: false };
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
  const result = await fetchFromGitHub(key);
  if (!result.ok) {
    // 503 for a throttle, so a caller can retry and so this is not mistaken
    // for a sign that does not exist. Never cached, either way.
    return result.throttled
      ? NextResponse.json(
          { error: "Sign store rate limited", retry: true },
          {
            status: 503,
            headers: {
              "Access-Control-Allow-Origin": "*",
              "Cache-Control": "no-store",
              "Retry-After": "5",
            },
          }
        )
      : NextResponse.json(
          { error: "Sign not found" },
          { status: 404, headers: { "Access-Control-Allow-Origin": "*" } }
        );
  }
  const data = result.data;

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
