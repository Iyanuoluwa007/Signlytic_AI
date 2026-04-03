import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  try {
    const { glosses } = await req.json();

    if (!glosses || typeof glosses !== "string" || !glosses.trim()) {
      return NextResponse.json({ error: "Missing glosses" }, { status: 400 });
    }

    const apiKey = process.env.GROQ_API_KEY;
    if (!apiKey) {
      return NextResponse.json(
        { error: "GROQ_API_KEY not configured" },
        { status: 500 }
      );
    }

    const res = await fetch(
      "https://api.groq.com/openai/v1/chat/completions",
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${apiKey}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          model: "llama-3.3-70b-versatile",
          messages: [
            {
              role: "system",
              content:
                "You are a British Sign Language (BSL) gloss interpreter. " +
                "Convert BSL gloss sequences into natural English sentences. " +
                "BSL uses topic-comment order, different from English word order. " +
                "For example: TOMORROW MEETING WHAT TIME -> What time is the meeting tomorrow? " +
                "MY NAME SARAH -> My name is Sarah. " +
                "YESTERDAY I GO DOCTOR -> I went to the doctor yesterday. " +
                "Respond with ONLY the natural English sentence. No quotes, no explanation.",
            },
            {
              role: "user",
              content: `Convert these BSL glosses to natural English: ${glosses.trim().toUpperCase()}`,
            },
          ],
          temperature: 0.3,
          max_tokens: 200,
        }),
      }
    );

    if (!res.ok) {
      const errBody = await res.text().catch(() => "");
      console.error(`[ERR] Groq API ${res.status}: ${errBody}`);
      return NextResponse.json(
        { error: "Translation service error" },
        { status: 502 }
      );
    }

    const data = await res.json();
    const english = data.choices?.[0]?.message?.content?.trim() || "";

    if (!english) {
      return NextResponse.json(
        { error: "Empty response from model" },
        { status: 502 }
      );
    }

    return NextResponse.json({ english });
  } catch (err) {
    console.error("[ERR] glosses-to-english:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
