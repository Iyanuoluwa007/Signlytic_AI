"use client";

import { useState, useRef, type ChangeEvent } from "react";
import Signable from "@/components/Signable";

// The local system, released from the main repo on its own version stream. The
// browser extension and the desktop app ship from the overlay repo and move
// independently, so their versions are deliberately not tied to this one.
const LOCAL_VERSION = "v1.2.0";
const LOCAL_RELEASE_URL =
  "https://github.com/Iyanuoluwa007/Signlytic_AI/releases/tag/v1.2.0";

/* =========================================
   LOCAL FALLBACK DICTIONARIES
   Used when API routes are unavailable
   ========================================= */
const G: Record<string, string> = {
  hello:"HELLO",hi:"HELLO",my:"MY",name:"NAME",is:"",what:"WHAT",time:"TIME",
  the:"",a:"",an:"",meeting:"MEETING",tomorrow:"TOMORROW",yesterday:"YESTERDAY",
  today:"TODAY",please:"PLEASE",thank:"THANK",thanks:"THANK",you:"YOU",your:"YOUR",
  i:"I",me:"I",we:"WE",go:"GO",going:"GO",went:"GO",want:"WANT",need:"NEED",
  help:"HELP",good:"GOOD",bad:"BAD",nice:"NICE",morning:"MORNING",afternoon:"AFTERNOON",
  evening:"EVENING",doctor:"DOCTOR",hospital:"HOSPITAL",understand:"UNDERSTAND",
  not:"NOT",no:"NO",yes:"YES",how:"HOW",where:"WHERE",when:"WHEN",who:"WHO",
  much:"MUCH",many:"MUCH",very:"MUCH",sorry:"SORRY",happy:"HAPPY",sad:"SAD",
  work:"WORK",school:"SCHOOL",home:"HOME",eat:"EAT",drink:"DRINK",sleep:"SLEEP",
  friend:"FRIEND",family:"FAMILY",man:"MAN",woman:"WOMAN",child:"CHILD",
  come:"COME",here:"HERE",there:"THERE",can:"CAN",know:"KNOW",think:"THINK",
  like:"LIKE",love:"LOVE",big:"BIG",small:"SMALL",old:"OLD","new":"NEW",
  young:"YOUNG",water:"WATER",food:"FOOD",money:"MONEY",phone:"PHONE",
  mother:"MOTHER",father:"FATHER",sister:"SISTER",brother:"BROTHER",
  learn:"LEARN",teach:"TEACH",read:"READ",write:"WRITE",give:"GIVE",
  take:"TAKE",start:"START",stop:"STOP",finish:"FINISH",wait:"WAIT",
};
const R: Record<string, string> = {
  HELLO:"Hello",MY:"my",NAME:"name",TOMORROW:"Tomorrow",MEETING:"meeting",
  WHAT:"what",TIME:"time",YESTERDAY:"Yesterday",GO:"go",DOCTOR:"doctor",
  THANK:"Thank",YOU:"you",MUCH:"very much",I:"I",NOT:"not",UNDERSTAND:"understand",
  PLEASE:"please",HELP:"help",NEED:"need",WANT:"want",GOOD:"good",MORNING:"morning",
  HOW:"how",WHERE:"where",WHEN:"when",WHO:"who",SORRY:"Sorry",HAPPY:"happy",
  WORK:"work",HOME:"home",SCHOOL:"school",FRIEND:"friend",COME:"come",HERE:"here",
  YES:"Yes",NO:"No",EAT:"eat",DRINK:"drink",SLEEP:"sleep",FAMILY:"family",
  MOTHER:"mother",FATHER:"father",SISTER:"sister",BROTHER:"brother",
  LOVE:"love",LIKE:"like",KNOW:"know",THINK:"think",LEARN:"learn",
  GIVE:"give",TAKE:"take",START:"start",STOP:"stop",FINISH:"finish",WAIT:"wait",
};
function localToGloss(t: string) {
  return t.toLowerCase().match(/[a-z']+/g)?.map(w => G[w] ?? w.toUpperCase()).filter(Boolean).filter((g,i,a) => i===0||a[i-1]!==g).join(" ") ?? "";
}
function localToEnglish(gl: string) {
  const g = gl.toUpperCase().split(/\s+/).filter(Boolean);
  if (!g.length) return "";
  let s = g.map(w => R[w] ?? w.toLowerCase()).join(" ");
  s = s[0].toUpperCase() + s.slice(1);
  if (!/[.?!]$/.test(s)) s += g.some(x => ["WHAT","WHERE","WHEN","HOW","WHO"].includes(x)) ? "?" : ".";
  return s;
}

/* =========================================
   UI COMPONENTS
   ========================================= */
const Label = ({ children }: { children: string }) => (
  <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.12em] mb-3 pl-2 border-l-2 border-[#0e7c6b]">{children}</div>
);
const ResultCard = ({ label, children, accent = "teal" }: { label: string; children: React.ReactNode; accent?: "teal" | "navy" }) => (
  <div className={`bg-white/[0.02] border border-white/[0.06] rounded-xl p-4 mb-3 ${accent === "teal" ? "border-l-2 border-l-[#0e7c6b]" : "border-l-2 border-l-[#3b82f6]"}`}>
    <div className="text-[9px] font-bold text-white/25 uppercase tracking-[0.12em] mb-1.5">{label}</div>
    {children}
  </div>
);
const Badge = ({ children, variant = "green" }: { children: string; variant?: "green" | "amber" | "red" }) => {
  const cls = variant === "green" ? "bg-[#0e7c6b]/10 text-[#5eead4] border-[#0e7c6b]/20"
    : variant === "red" ? "bg-red-500/10 text-red-400 border-red-500/20"
    : "bg-amber-500/10 text-amber-400 border-amber-500/20";
  return <span className={`text-[9px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full border ${cls}`}>{children}</span>;
};

/* =========================================
   PAGE
   ========================================= */
export default function DemoPage() {
  const [tab, setTab] = useState(0);

  /* --- Tab 1: BSL to English --- */
  const [d1Glosses, setD1Glosses] = useState("");
  const [d1DetectedGlosses, setD1DetectedGlosses] = useState("");
  const [d1Translation, setD1Translation] = useState("");
  const [d1Processing, setD1Processing] = useState(false);
  const [d1Source, setD1Source] = useState<"ai" | "local" | "">("");

  const handleD1Translate = async () => {
    const input = d1Glosses.trim();
    if (!input) return;
    setD1Processing(true);
    setD1DetectedGlosses(input.toUpperCase());
    setD1Translation("");
    setD1Source("");

    try {
      const res = await fetch("/api/glosses-to-english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ glosses: input }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      if (data.english) {
        setD1Translation(data.english);
        setD1Source("ai");
      } else {
        throw new Error("Empty response");
      }
    } catch {
      // Fallback to local dictionary
      setD1Translation(localToEnglish(input));
      setD1Source("local");
    } finally {
      setD1Processing(false);
    }
  };

  /* --- Tab 2: English to BSL --- */
  const [d2Text, setD2Text] = useState("");
  const [d2Echo, setD2Echo] = useState("");
  const [d2Glosses, setD2Glosses] = useState("");
  const [d2Processing, setD2Processing] = useState(false);
  const [d2Source, setD2Source] = useState<"ai" | "local" | "">("");

  const handleD2Convert = async () => {
    const input = d2Text.trim();
    if (!input) return;
    setD2Processing(true);
    setD2Echo(input);
    setD2Glosses("");
    setD2Source("");

    try {
      const res = await fetch("/api/english-to-glosses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: input }),
      });
      if (!res.ok) throw new Error(`API ${res.status}`);
      const data = await res.json();
      if (data.glosses) {
        setD2Glosses(data.glosses);
        setD2Source("ai");
      } else {
        throw new Error("Empty response");
      }
    } catch {
      // Fallback to local dictionary
      setD2Glosses(localToGloss(input));
      setD2Source("local");
    } finally {
      setD2Processing(false);
    }
  };

  const tabs = ["BSL to English", "English to BSL", "Help & Accessibility", "About & System"];

  return (
    <div className="min-h-screen bg-[#08090d] antialiased selection:bg-[#0e7c6b]/30 selection:text-white">
      {/* Nav */}
      <nav className="border-b border-white/[0.04] bg-[#08090d]/80 backdrop-blur-2xl sticky top-0 z-50">
        <div className="max-w-[1120px] mx-auto px-6 h-14 flex items-center justify-between">
          <a href="/" className="flex items-center gap-2">
            <div className="w-6 h-6 rounded-md bg-[#0e7c6b] flex items-center justify-center">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <span className="font-semibold text-[13px] text-white/80">Signlytic AI</span>
            <span className="text-white/15 text-[13px] mx-1">/</span>
            <span className="text-white/35 text-[13px]">Demo</span>
          </a>
          <div className="flex items-center gap-4">
            <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank" rel="noopener noreferrer" className="text-[11px] text-white/25 hover:text-white/50 transition-colors">GitHub</a>
            <a href="/" className="text-[12px] text-white/30 hover:text-white/60 transition-colors">&larr; Back</a>
          </div>
        </div>
      </nav>

      {/* Status Banner */}
      <div className="bg-[#0e7c6b]/[0.04] border-b border-[#0e7c6b]/10">
        <div className="max-w-[1120px] mx-auto px-6 py-2.5 flex items-center justify-center gap-2">
          <div className="w-1.5 h-1.5 rounded-full bg-[#5eead4] animate-pulse flex-shrink-0" />
          <p className="text-[11px] text-[#5eead4]/80">
            <span className="font-semibold">Live</span> &mdash; Text translation powered by Llama 3.3 70B. Video recognition, speech, and signing animation require <a href="https://github.com/Iyanuoluwa007/Signlytic_AI" target="_blank" rel="noopener noreferrer" className="underline underline-offset-2 hover:text-[#5eead4]">local GPU setup</a>.
          </p>
        </div>
      </div>

      {/* Hero banner */}
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_center,rgba(14,124,107,0.04)_0%,transparent_60%)]" />
        <div className="relative max-w-[1120px] mx-auto px-6 py-7">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-[#0e7c6b]/20 border border-[#0e7c6b]/30 flex items-center justify-center">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="#5eead4" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <div>
              <h1 className="text-[1.15rem] font-bold text-white tracking-[-0.02em]"><Signable text="Signlytic AI">Signlytic AI</Signable></h1>
              <p className="text-[12px] text-white/30"><Signable text="BSL Translation System. Interactive Demo.">BSL Translation System &middot; Interactive Demo</Signable></p>
            </div>
          </div>
        </div>
      </div>

      {/* App container */}
      <div className="max-w-[1120px] mx-auto px-6 pb-16">
        <div className="bg-white/[0.015] border border-white/[0.06] rounded-2xl overflow-hidden">

          {/* Tab bar */}
          <div className="flex border-b border-white/[0.06] bg-white/[0.02] overflow-x-auto">
            {tabs.map((t, i) => (
              <button key={t} onClick={() => setTab(i)} className={`px-5 py-3.5 text-[13px] font-semibold whitespace-nowrap transition-all border-b-2 ${tab === i ? "text-[#5eead4] border-[#0e7c6b] bg-white/[0.02]" : "text-white/30 border-transparent hover:text-white/50"}`}>
                {t}
              </button>
            ))}
          </div>

          <div className="p-6 md:p-8">

            {/* === TAB 1: BSL to English === */}
            {tab === 0 && (
              <div>
                <h2 className="text-[1.1rem] font-bold text-white/90 mb-1"><Signable text="Understand BSL Signs">Understand BSL Signs</Signable></h2>
                <p className="text-[13px] text-white/35 mb-6">
                  <Signable text="Type BSL glosses to see the English meaning.">
                    Type BSL glosses to see the English meaning.
                  </Signable>{" "}
                  Video and camera recognition require local GPU.
                </p>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <Label>Input</Label>

                    {/* Video / Camera - GPU required callout */}
                    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5 mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[13px] font-semibold text-white/70">Video or camera recognition</span>
                        <Badge variant="red">Local GPU only</Badge>
                      </div>
                      <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg px-4 py-4">
                        <p className="text-[12px] text-white/35 leading-relaxed mb-3">
                          BSL video recognition uses Video-SWIN-T with 5,203 pre-extracted sign features.
                          This requires a local NVIDIA GPU.
                        </p>
                        <a
                          href={LOCAL_RELEASE_URL}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 text-[12px] font-semibold text-[#5eead4] hover:text-[#5eead4]/80 transition-colors"
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
                          Clone repository and run locally
                          <span className="text-white/30 font-normal">{LOCAL_VERSION}</span>
                        </a>
                      </div>
                    </div>

                    {/* Divider */}
                    <div className="flex items-center gap-3 my-4">
                      <div className="flex-1 h-px bg-white/[0.06]" />
                      <span className="text-[10px] text-white/15 uppercase tracking-wider">or type glosses</span>
                      <div className="flex-1 h-px bg-white/[0.06]" />
                    </div>

                    {/* Text input */}
                    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[13px] font-semibold text-white/70">Type BSL glosses</span>
                        <Badge variant="green">Live</Badge>
                      </div>
                      <p className="text-[11px] text-white/25 mb-3">Type BSL sign glosses (e.g. TOMORROW MEETING WHAT TIME) and get natural English.</p>
                      <textarea value={d1Glosses} onChange={e => setD1Glosses(e.target.value)} placeholder="HELLO MY NAME SARAH" rows={3}
                        className="w-full bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2.5 text-[14px] text-white/80 placeholder:text-white/15 focus:outline-none focus:border-[#0e7c6b]/40 resize-none mb-3 font-[inherit]"
                        onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleD1Translate(); }}} />
                      <button onClick={handleD1Translate} disabled={!d1Glosses.trim() || d1Processing} className="bg-white text-[#08090d] font-semibold px-4 py-2 rounded-lg text-[12px] hover:shadow-[0_0_20px_rgba(255,255,255,0.05)] transition-all disabled:opacity-30">Translate to English</button>
                    </div>

                    {/* Examples */}
                    <div className="mt-4">
                      <div className="text-[10px] text-white/15 italic mb-2">BSL uses different word order than English. Try these:</div>
                      <div className="flex flex-wrap gap-1.5">
                        {["TOMORROW MEETING WHAT TIME","MY NAME SARAH","YESTERDAY I GO DOCTOR","THANK YOU MUCH","I NOT UNDERSTAND","PLEASE HELP ME","WHERE SCHOOL"].map(ex => (
                          <button key={ex} onClick={() => setD1Glosses(ex)} className="px-2.5 py-1 text-[11px] text-white/30 bg-white/[0.02] border border-white/[0.05] rounded-lg hover:border-white/[0.1] hover:text-white/50 transition-all">{ex}</button>
                        ))}
                      </div>
                    </div>
                  </div>

                  {/* Output column */}
                  <div>
                    <Label>Results</Label>
                    {d1Processing && (
                      <div className="flex items-center gap-2 mb-3 p-3 bg-white/[0.02] rounded-xl">
                        <div className="w-4 h-4 border-[1.5px] border-[#0e7c6b]/40 border-t-[#0e7c6b] rounded-full animate-spin" />
                        <span className="text-[12px] text-white/30">Translating via Llama 3.3 70B...</span>
                      </div>
                    )}
                    <ResultCard label="BSL glosses">
                      <div className="text-[14px] text-white/60 min-h-[40px] font-mono">{d1DetectedGlosses || <span className="text-white/15 italic font-sans">Waiting for input...</span>}</div>
                    </ResultCard>
                    <ResultCard label="English translation" accent="navy">
                      <div className="text-[15px] text-white/80 min-h-[60px] leading-relaxed">{d1Translation || <span className="text-white/15 italic">Translation will appear here</span>}</div>
                      {d1Source && (
                        <div className="mt-2">
                          <span className={`text-[9px] font-bold uppercase tracking-wider ${d1Source === "ai" ? "text-[#5eead4]/50" : "text-amber-400/50"}`}>
                            {d1Source === "ai" ? "Powered by Llama 3.3 70B via Groq" : "Offline fallback (dictionary)"}
                          </span>
                        </div>
                      )}
                    </ResultCard>
                    <ResultCard label="Speech output">
                      <div className="text-[12px] text-white/20 italic">Text-to-speech requires local Coqui XTTS v2 with GPU</div>
                    </ResultCard>
                  </div>
                </div>
              </div>
            )}

            {/* === TAB 2: English to BSL === */}
            {tab === 1 && (
              <div>
                <h2 className="text-[1.1rem] font-bold text-white/90 mb-1"><Signable text="Show Me in BSL">Show Me in BSL</Signable></h2>
                <p className="text-[13px] text-white/35 mb-6"><Signable text="Type in English to see the BSL gloss sequence.">Type in English to see the BSL gloss sequence.</Signable> Audio input requires local GPU.</p>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <Label>Input</Label>

                    {/* Audio - GPU required */}
                    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5 mb-3">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[13px] font-semibold text-white/70">Record or upload audio</span>
                        <Badge variant="red">Local GPU only</Badge>
                      </div>
                      <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg px-4 py-3 flex items-center gap-3">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white/15 flex-shrink-0"><path d="M12 1a3 3 0 0 0-3 3v8a3 3 0 0 0 6 0V4a3 3 0 0 0-3-3z"/><path d="M19 10v2a7 7 0 0 1-14 0v-2"/><line x1="12" y1="19" x2="12" y2="23"/></svg>
                        <span className="text-[11px] text-white/25">Requires OpenAI Whisper (local GPU)</span>
                      </div>
                    </div>

                    <div className="flex items-center gap-3 my-4">
                      <div className="flex-1 h-px bg-white/[0.06]" />
                      <span className="text-[10px] text-white/15 uppercase tracking-wider">or type in English</span>
                      <div className="flex-1 h-px bg-white/[0.06]" />
                    </div>

                    <div className="bg-white/[0.02] border border-white/[0.06] rounded-xl p-5">
                      <div className="flex items-center gap-2 mb-2">
                        <span className="text-[13px] font-semibold text-white/70">Type what you want to say</span>
                        <Badge variant="green">Live</Badge>
                      </div>
                      <textarea value={d2Text} onChange={e => setD2Text(e.target.value)} placeholder="What time is the meeting tomorrow?" rows={3}
                        className="w-full bg-white/[0.03] border border-white/[0.06] rounded-lg px-3 py-2.5 text-[14px] text-white/80 placeholder:text-white/15 focus:outline-none focus:border-[#0e7c6b]/40 resize-none mb-3 font-[inherit]"
                        onKeyDown={e => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); handleD2Convert(); }}} />
                      <button onClick={handleD2Convert} disabled={!d2Text.trim() || d2Processing} className="bg-white text-[#08090d] font-semibold px-4 py-2 rounded-lg text-[12px] hover:shadow-[0_0_20px_rgba(255,255,255,0.05)] transition-all disabled:opacity-30">Convert to BSL</button>
                    </div>

                    <div className="mt-4 flex flex-wrap gap-1.5">
                      {["Hello, my name is John.","What time is the meeting?","Thank you very much.","I need help please.","Where is the school?","I don't understand.","Good morning, how are you?"].map(ex => (
                        <button key={ex} onClick={() => setD2Text(ex)} className="px-2.5 py-1 text-[11px] text-white/30 bg-white/[0.02] border border-white/[0.05] rounded-lg hover:border-white/[0.1] hover:text-white/50 transition-all">{ex}</button>
                      ))}
                    </div>
                  </div>

                  <div>
                    <Label>Results</Label>
                    {d2Processing && (
                      <div className="flex items-center gap-2 mb-3 p-3 bg-white/[0.02] rounded-xl">
                        <div className="w-4 h-4 border-[1.5px] border-[#0e7c6b]/40 border-t-[#0e7c6b] rounded-full animate-spin" />
                        <span className="text-[12px] text-white/30">Converting via Llama 3.3 70B...</span>
                      </div>
                    )}
                    <ResultCard label="What you said">
                      <div className="text-[14px] text-white/60 min-h-[30px]">{d2Echo || <span className="text-white/15 italic">Waiting for input...</span>}</div>
                    </ResultCard>
                    <ResultCard label="BSL glosses" accent="navy">
                      <div className="text-[15px] text-white/80 min-h-[40px] font-mono leading-relaxed">{d2Glosses || <span className="text-white/15 italic font-sans">Glosses will appear here</span>}</div>
                      {d2Source && (
                        <div className="mt-2">
                          <span className={`text-[9px] font-bold uppercase tracking-wider ${d2Source === "ai" ? "text-[#5eead4]/50" : "text-amber-400/50"}`}>
                            {d2Source === "ai" ? "Powered by Llama 3.3 70B via Groq" : "Offline fallback (dictionary)"}
                          </span>
                        </div>
                      )}
                    </ResultCard>

                    {/* Signing animation preview */}
                    <ResultCard label="Signing animation">
                      <div className="border-2 border-dashed border-white/[0.06] rounded-lg h-40 flex flex-col items-center justify-center mb-2 px-3">
                        <a
                          href={LOCAL_RELEASE_URL}
                          target="_blank"
                          rel="noopener noreferrer"
                          className="inline-flex items-center gap-1.5 text-[12px] font-semibold text-[#5eead4] hover:text-[#5eead4]/80 transition-colors mb-3"
                        >
                          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/></svg>
                          Clone repository and run locally
                          <span className="text-white/30 font-normal">{LOCAL_VERSION}</span>
                        </a>
                        <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.2" className="text-white/10 mb-2"><circle cx="12" cy="5" r="2"/><path d="M12 7v6"/><path d="M8 11l4 2 4-2"/><path d="M8 21l4-6 4 6"/></svg>
                        <span className="text-[11px] text-white/20 mb-0.5">3D avatar signing preview</span>
                        <span className="text-[9px] text-white/12">Pose data deployment in progress</span>
                      </div>
                      <div className="bg-white/[0.02] border border-white/[0.06] rounded-lg px-3 py-2">
                        <p className="text-[10px] text-white/30 leading-relaxed">
                          Signing animation uses pose frame data for each gloss.
                          The 3D avatar will animate once pose data is deployed to CDN.
                          Run locally for full 2D skeleton and 3D character animation now.
                        </p>
                      </div>
                    </ResultCard>
                  </div>
                </div>
              </div>
            )}

            {/* === TAB 3: Help === */}
            {tab === 2 && (
              <div>
                <h2 className="text-[1.1rem] font-bold text-white/90 mb-1"><Signable text="How to Use This App">How to Use This App</Signable></h2>
                <p className="text-[13px] text-white/35 mb-6"><Signable text="Simple guides for BSL users and hearing users.">Simple guides for BSL users and hearing users.</Signable></p>
                <div className="space-y-3">
                  {[
                    { title: "BSL to English (Live)", body: "Go to the BSL to English tab. Type BSL glosses like TOMORROW MEETING WHAT TIME and click Translate. The system uses Llama 3.3 70B to produce natural English. Video and camera recognition require a local GPU." },
                    { title: "English to BSL (Live)", body: "Go to the English to BSL tab. Type an English sentence and click Convert to BSL. The AI generates proper BSL gloss ordering. Audio input requires a local GPU with Whisper." },
                    { title: "What are glosses?", body: "A \"gloss\" is the written name of a BSL sign. For example, TOMORROW MEETING WHAT TIME means \"What time is the meeting tomorrow?\" BSL uses a different word order than English." },
                    { title: "Full System (Local GPU)", body: "The complete system includes Video-SWIN-T recognition (5,203 signs), Cerebras gpt-oss-120b translation with a Groq fallback, Coqui XTTS v2 speech, and 3D avatar animation. Clone the GitHub repository and run locally with a GPU for the full experience." },
                    { title: "Accessibility", body: "All outputs include visible text. Nothing relies on audio. The interface supports keyboard navigation. Press Enter to submit in any text field." },
                  ].map(h => (
                    <div key={h.title} className="bg-white/[0.02] border border-white/[0.06] border-l-2 border-l-[#0e7c6b] rounded-xl p-5">
                      <h4 className="text-[14px] font-semibold text-white/80 mb-1">
                        <Signable text={h.title}>{h.title}</Signable>
                      </h4>
                      <p className="text-[13px] text-white/35 leading-relaxed">
                        <Signable text={h.body}>{h.body}</Signable>
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* === TAB 4: About === */}
            {tab === 3 && (
              <div className="max-w-[760px] mx-auto">
                <h2 className="text-[1.1rem] font-bold text-white/90 mb-1"><Signable text="System Overview">System Overview</Signable></h2>
                <p className="text-[13px] text-white/35 mb-6"><Signable text="Architecture, models, and performance.">Architecture, models, and performance.</Signable></p>

                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-8">
                  {[{ val: "100%", label: "Top-1 Accuracy" }, { val: "5,203", label: "BSL Signs" }, { val: "11,573+", label: "Glosses" }, { val: "92%", label: "ROUGE-L" }].map(m => (
                    <div key={m.label} className="bg-white/[0.02] border border-white/[0.06] border-t-2 border-t-[#0e7c6b] rounded-xl p-4 text-center">
                      <div className="text-[1.4rem] font-extrabold text-white/85 tracking-tight">{m.val}</div>
                      <div className="text-[10px] font-semibold text-white/25 uppercase tracking-wider mt-0.5">{m.label}</div>
                    </div>
                  ))}
                </div>

                <div className="rounded-xl overflow-hidden border border-white/[0.06] mb-6">
                  <table className="w-full text-[13px]">
                    <thead><tr className="bg-[#0e7c6b]/15">{["Component","Technology","Details"].map(h => <th key={h} className="text-left px-4 py-3 text-[10px] font-bold text-[#5eead4] uppercase tracking-[0.1em]">{h}</th>)}</tr></thead>
                    <tbody>
                      {[["Sign Recognition","Video-SWIN-T","5,203 pre-extracted 768-dim features"],["Speech Recognition","OpenAI Whisper","Base model, 16 kHz mono"],["Text-to-Speech","Coqui XTTS v2","Voice cloning with speaker reference"],["Language Model","Groq Llama 3.3 70B","Gloss to natural English (live on demo)"],["Signing Animation","3D Avatar + 2D Pose","Mixamo avatars + skeleton signing"],["Vocabulary","11,573+ glosses","BSL-1K + BSLDict datasets"]].map(([c,t,d]) => (
                        <tr key={c} className="border-t border-white/[0.04] hover:bg-white/[0.01]">
                          <td className="px-4 py-3 font-medium text-white/60">{c}</td>
                          <td className="px-4 py-3 font-mono text-[12px] text-[#5eead4]/70">{t}</td>
                          <td className="px-4 py-3 text-white/30">{d}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                <div className="rounded-xl overflow-hidden border border-white/[0.06]">
                  <table className="w-full text-[13px]">
                    <thead><tr className="bg-[#0e7c6b]/15">{["Model","Language","Top-1","Top-5"].map(h => <th key={h} className="text-left px-4 py-3 text-[10px] font-bold text-[#5eead4] uppercase tracking-[0.1em]">{h}</th>)}</tr></thead>
                    <tbody>
                      {[{m:"BSL Dict Retrieval",l:"British",t1:"100%",t5:"100%",hl:true},{m:"BSL-100",l:"British",t1:"72.34%",t5:"95.03%",hl:false},{m:"BSL-500",l:"British",t1:"59.26%",t5:"89.04%",hl:false},{m:"Pose Recognition",l:"ASL",t1:"44.44%",t5:"81.62%",hl:false}].map(r => (
                        <tr key={r.m} className="border-t border-white/[0.04] hover:bg-white/[0.01]">
                          <td className="px-4 py-3 font-medium text-white/60">{r.m}</td>
                          <td className="px-4 py-3 text-white/30">{r.l}</td>
                          <td className={`px-4 py-3 font-mono font-semibold ${r.hl ? "text-[#5eead4]" : "text-white/40"}`}>{r.t1}</td>
                          <td className={`px-4 py-3 font-mono font-semibold ${r.hl ? "text-[#5eead4]" : "text-white/40"}`}>{r.t5}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="text-center mt-8">
          <p className="text-white/15 text-[11px]">Signlytic AI &middot; Independent Robotics &amp; AI Systems Engineer &middot; v2.1 &middot; April 2026</p>
        </div>
      </div>

      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap");
        html { scroll-behavior: smooth; }
        body { font-family: "Outfit", system-ui, -apple-system, sans-serif; background: #08090d; color: #e8eaed; }
        .font-mono { font-family: "JetBrains Mono", monospace; }
        ::selection { background: rgba(14,124,107,0.3); color: #fff; }
      `}</style>
    </div>
  );
}


