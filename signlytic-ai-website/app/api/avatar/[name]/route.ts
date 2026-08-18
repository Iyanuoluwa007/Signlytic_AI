import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

// Avatar models live in a private GitHub repo rather than in this repo, so a
// ~33-50 MB binary never enters the app's git history. The browser cannot
// fetch them directly: the access token has to stay server-side, and GitHub's
// asset hosts send no CORS header. This route attaches the token, adds CORS,
// and streams the model back.
//
// The models live alongside the sign data rather than in their own repo,
// because the deployed token is scoped to the signs repo: pointing elsewhere
// returns 404 (GitHub hides private repos a token cannot see rather than
// returning 403). Owner/repo/branch and the token all default to the signs
// pipeline's values, so this needs no extra Vercel configuration.
const GH_OWNER  = process.env.AVATAR_GITHUB_OWNER  || process.env.SIGNS_GITHUB_OWNER  || "Iyanuoluwa007";
const GH_REPO   = process.env.AVATAR_GITHUB_REPO   || process.env.SIGNS_GITHUB_REPO   || "signlytic-signs-data";
const GH_BRANCH = process.env.AVATAR_GITHUB_BRANCH || process.env.SIGNS_GITHUB_BRANCH || "main";
const GH_PAT    = process.env.AVATAR_GITHUB_PAT    || process.env.SIGNS_GITHUB_PAT;
// Sign JSON sits at the repo root, so the models are kept in a subfolder.
const GH_DIR    = process.env.AVATAR_GITHUB_DIR    || "avatars";

// Allowlist: the path segment never reaches GitHub unvalidated.
const MODELS: Record<string, string> = {
  male: "Male.glb",
  female: "Female.glb",
};

// Models are immutable once published, so let the CDN hold them for a year.
// This is what keeps the large download off the origin after the first request.
const CACHE = "public, max-age=31536000, s-maxage=31536000, immutable";

const CORS = { "Access-Control-Allow-Origin": "*" };

export async function GET(
  req: NextRequest,
  { params }: { params: Promise<{ name: string }> }
) {
  const { name } = await params;
  const file = MODELS[String(name).toLowerCase()];

  if (!file) {
    return NextResponse.json(
      { error: "Unknown avatar" },
      { status: 404, headers: CORS }
    );
  }
  if (!GH_PAT) {
    return NextResponse.json(
      { error: "Avatar store not configured" },
      { status: 500, headers: CORS }
    );
  }

  const url = `https://raw.githubusercontent.com/${GH_OWNER}/${GH_REPO}/${GH_BRANCH}/${GH_DIR}/${file}`;

  // GitHub rate limits these fetches. The models are tens of megabytes, far
  // past what the CDN will hold, so the year-long s-maxage above never takes
  // effect and every cold visitor reaches raw.githubusercontent directly. A 429
  // is therefore routine rather than exceptional, and a single attempt turns a
  // throttle into a broken avatar for that visitor.
  //
  // Retrying briefly, honouring Retry-After when GitHub sends one, converts
  // most throttles into a slightly slower success. It does not address the
  // cause, which is the file size.
  const attempt = (): Promise<Response> =>
    fetch(url, {
      headers: { Authorization: `token ${GH_PAT}` },
      // Let the platform cache the upstream fetch as well as the response
      cache: "force-cache",
    });

  try {
    let upstream = await attempt();

    for (let tries = 0; upstream.status === 429 && tries < 2; tries++) {
      const retryAfter = Number(upstream.headers.get("retry-after"));
      const waitMs =
        Number.isFinite(retryAfter) && retryAfter > 0
          ? Math.min(retryAfter * 1000, 3000)
          : 500 * (tries + 1);
      await new Promise((r) => setTimeout(r, waitMs));
      upstream = await attempt();
    }

    if (!upstream.ok || !upstream.body) {
      console.error(`[ERR] avatar upstream ${upstream.status} for ${file}`);
      return NextResponse.json(
        {
          error: "Avatar unavailable",
          detail:
            upstream.status === 429
              ? "The avatar store is rate limited. Try again shortly."
              : undefined,
        },
        {
          status: 502,
          headers: {
            ...CORS,
            // Never let a transient failure be cached in place of the model.
            "Cache-Control": "no-store",
            ...(upstream.status === 429 ? { "Retry-After": "5" } : {}),
          },
        }
      );
    }

    // Stream rather than buffer: these are tens of megabytes.
    const headers: Record<string, string> = {
      "Content-Type": "model/gltf-binary",
      "Cache-Control": CACHE,
      ...CORS,
    };
    const len = upstream.headers.get("content-length");
    if (len) headers["Content-Length"] = len;

    return new NextResponse(upstream.body, { status: 200, headers });
  } catch (err) {
    console.error("[ERR] avatar route:", err);
    return NextResponse.json(
      { error: "Avatar fetch failed" },
      { status: 500, headers: CORS }
    );
  }
}

export async function OPTIONS() {
  return new NextResponse(null, {
    status: 204,
    headers: {
      ...CORS,
      "Access-Control-Allow-Methods": "GET, OPTIONS",
      "Access-Control-Allow-Headers": "Content-Type",
    },
  });
}
