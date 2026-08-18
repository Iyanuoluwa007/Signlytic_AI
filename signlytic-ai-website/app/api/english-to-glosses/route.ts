import { NextRequest, NextResponse } from "next/server";

export const runtime = "edge";

export async function POST(req: NextRequest) {
  try {
    const { text } = await req.json();

    if (!text || typeof text !== "string" || !text.trim()) {
      return NextResponse.json({ error: "Missing text" }, { status: 400 });
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
          model: "openai/gpt-oss-120b",
          messages: [
            {
              role: "system",
              content:
                "You are a British Sign Language (BSL) translator. " +
                "Convert English sentences into BSL gloss sequences. " +
                "BSL uses topic-comment structure and drops articles/copulas. " +
                "Rules: " +
                "1. Output ONLY uppercase BSL glosses separated by spaces. " +
                "2. Drop articles (a, an, the), copulas (is, am, are, was, were). " +
                "3. Use BSL word order (topic first, then comment). " +
                "4. Time markers go first: TOMORROW MEETING WHAT TIME (not WHAT TIME MEETING TOMORROW). " +
                "5. Negation follows the verb: I UNDERSTAND NOT (not I NOT UNDERSTAND). " +
                "6. Use common BSL signs only. No invented glosses. " +
                "Examples: " +
                "What time is the meeting tomorrow? -> TOMORROW MEETING WHAT TIME " +
                "My name is Sarah -> MY NAME SARAH " +
                "I went to the doctor yesterday -> YESTERDAY I GO DOCTOR " +
                "I don't understand -> I UNDERSTAND NOT " +
                "Thank you very much -> THANK YOU MUCH " +
                "Respond with ONLY the gloss sequence. No quotes, no explanation, no punctuation.",
            },
            {
              role: "user",
              content: `Convert to BSL glosses: ${text.trim()}`,
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
    const raw = data.choices?.[0]?.message?.content?.trim() || "";

    if (!raw) {
      return NextResponse.json(
        { error: "Empty response from model" },
        { status: 502 }
      );
    }

    // Clean: strip any quotes or explanation the model might add
    const glosses = raw
      .replace(/^["']|["']$/g, "")
      .replace(/[.!?,;:]/g, "")
      .toUpperCase()
      .trim();

    return NextResponse.json({ glosses });
  } catch (err) {
    console.error("[ERR] english-to-glosses:", err);
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    );
  }
}
