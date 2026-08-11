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

  try {
    const upstream = await fetch(url, {
      headers: { Authorization: `token ${GH_PAT}` },
      // Let the platform cache the upstream fetch as well as the response
      cache: "force-cache",
    });

    if (!upstream.ok || !upstream.body) {
      console.error(`[ERR] avatar upstream ${upstream.status} for ${file}`);
      return NextResponse.json(
        { error: "Avatar unavailable" },
        { status: 502, headers: CORS }
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
