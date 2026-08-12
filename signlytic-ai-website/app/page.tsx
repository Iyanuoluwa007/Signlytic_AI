"use client";

import { useState, useEffect, useRef, type ReactNode } from "react";
import Signable from "@/components/Signable";

// Published on the public overlay repo: the main repo is private, so its
// release assets are not publicly downloadable.
const DESKTOP_WINDOWS_URL =
  "https://github.com/Iyanuoluwa007/Signlytic-Overlay/releases/download/desktop-v0.3.5/Signlytic.AI.Setup.0.3.5.exe";

/* ═══════════════════════════════════════
   SCROLL ANIMATION SYSTEM
   ═══════════════════════════════════════ */

function useInView(threshold = 0.12) {
  const ref = useRef<HTMLDivElement>(null);
  const [visible, setVisible] = useState(false);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const obs = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) {
          setVisible(true);
          obs.disconnect();
        }
      },
      { threshold }
    );
    obs.observe(el);
    return () => obs.disconnect();
  }, [threshold]);
  return { ref, visible };
}

function Reveal({
  children,
  delay = 0,
  className = "",
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  const { ref, visible } = useInView();
  return (
    <div
      ref={ref}
      className={className}
      style={{
        opacity: visible ? 1 : 0,
        transform: visible ? "translateY(0)" : "translateY(20px)",
        transition: `opacity 0.8s cubic-bezier(0.16,1,0.3,1) ${delay}ms, transform 0.8s cubic-bezier(0.16,1,0.3,1) ${delay}ms`,
      }}
    >
      {children}
    </div>
  );
}

function Counter({ value, suffix = "" }: { value: number; suffix?: string }) {
  const [count, setCount] = useState(0);
  const { ref, visible } = useInView();
  useEffect(() => {
    if (!visible) return;
    let s = 0;
    const inc = value / 100;
    const t = setInterval(() => {
      s += inc;
      if (s >= value) {
        setCount(value);
        clearInterval(t);
      } else setCount(s);
    }, 16);
    return () => clearInterval(t);
  }, [visible, value]);
  return (
    <span ref={ref}>
      {Math.floor(count).toLocaleString()}
      {suffix}
    </span>
  );
}

/* ═══════════════════════════════════════
   INLINE DEMO — native translation widget
   ═══════════════════════════════════════ */

const GLOSS_MAP: Record<string, string> = {
  hello:"HELLO",hi:"HELLO",my:"MY",name:"NAME",is:"",what:"WHAT",time:"TIME",
  the:"",meeting:"MEETING",tomorrow:"TOMORROW",yesterday:"YESTERDAY",today:"TODAY",
  please:"PLEASE",thank:"THANK",thanks:"THANK",you:"YOU",your:"YOUR",i:"I",me:"I",
  go:"GO",going:"GO",went:"GO",want:"WANT",need:"NEED",help:"HELP",good:"GOOD",
  morning:"MORNING",doctor:"DOCTOR",understand:"UNDERSTAND",not:"NOT",no:"NO",
  yes:"YES",how:"HOW",where:"WHERE",when:"WHEN",who:"WHO",much:"MUCH",very:"MUCH",
  sorry:"SORRY",happy:"HAPPY",work:"WORK",school:"SCHOOL",home:"HOME",friend:"FRIEND",
  come:"COME",here:"HERE",can:"CAN",know:"KNOW",think:"THINK",like:"LIKE",love:"LOVE",
};

const REVERSE_MAP: Record<string, string> = {
  HELLO:"Hello",MY:"my",NAME:"name",TOMORROW:"Tomorrow",MEETING:"meeting",
  WHAT:"what",TIME:"time",YESTERDAY:"Yesterday",GO:"go",DOCTOR:"doctor",
  THANK:"Thank",YOU:"you",MUCH:"very much",I:"I",NOT:"not",UNDERSTAND:"understand",
  PLEASE:"please",HELP:"help",NEED:"need",WANT:"want",GOOD:"good",MORNING:"morning",
  HOW:"how",WHERE:"where",WHEN:"when",WHO:"who",SORRY:"Sorry",HAPPY:"happy",
  WORK:"work",HOME:"home",SCHOOL:"school",FRIEND:"friend",COME:"come",HERE:"here",
  YES:"Yes",NO:"No",
};

function textToGloss(text: string): string {
  return text.toLowerCase().match(/[a-z']+/g)?.map(w => GLOSS_MAP[w] ?? w.toUpperCase()).filter(Boolean).filter((g,i,a) => i===0 || a[i-1]!==g).join(" ") ?? "";
}

function glossToText(glosses: string): string {
  const g = glosses.toUpperCase().split(/\s+/);
  let s = g.map(w => REVERSE_MAP[w] ?? w.toLowerCase()).join(" ");
  if (s) { s = s[0].toUpperCase() + s.slice(1); if (!/[.?!]$/.test(s)) s += g.some(x=>["WHAT","WHERE","WHEN","HOW","WHO"].includes(x)) ? "?" : "."; }
  return s;
}

/* ═══════════════════════════════════════
   PAGE
   ═══════════════════════════════════════ */

export default function Home() {
  const [demoTab, setDemoTab] = useState<"bsl-to-en" | "en-to-bsl">("bsl-to-en");
  const [demoInput, setDemoInput] = useState("");
  const [demoOutput, setDemoOutput] = useState("");

  /* shared styles */
  const sectionCls = "relative py-28 md:py-36";
  const containerCls = "max-w-[1120px] mx-auto px-6";
  const labelCls =
    "text-[11px] font-bold uppercase tracking-[0.16em] text-[#0e7c6b] mb-4 block";
  const h2Cls =
    "text-[clamp(1.75rem,4vw,2.75rem)] font-bold text-white leading-[1.12] tracking-[-0.03em] mb-4";
  const subCls = "text-[#6b7280] text-base leading-relaxed max-w-xl mb-16";
  const dividerCls = "border-t border-white/[0.04]";

  return (
    <div className="min-h-screen bg-[#08090d] text-[#e8eaed] antialiased selection:bg-[#0e7c6b]/30 selection:text-white">
      {/* ═════════ NAV ═════════ */}
      <nav className="fixed top-0 w-full z-50">
        <div className="mx-auto max-w-[1120px] px-6 mt-4">
          <div className="flex items-center justify-between h-12 px-4 rounded-2xl bg-white/[0.03] backdrop-blur-2xl border border-white/[0.06] shadow-[0_0_0_1px_rgba(0,0,0,0.3),0_8px_40px_rgba(0,0,0,0.3)]">
            <a href="#" className="flex items-center gap-2">
              <div className="w-6 h-6 rounded-md bg-[#0e7c6b] flex items-center justify-center">
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="white"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <path d="M12 2L2 7l10 5 10-5-10-5z" />
                  <path d="M2 17l10 5 10-5" />
                  <path d="M2 12l10 5 10-5" />
                </svg>
              </div>
              <span className="font-semibold text-[13px] text-white/90 tracking-[-0.01em]">
                Signlytic AI
              </span>
            </a>
            <div className="hidden md:flex items-center gap-0.5">
              {["About", "Demo", "Performance", "Architecture"].map((s) => (
                <a
                  key={s}
                  href={`#${s.toLowerCase()}`}
                  className="px-3 py-1.5 text-[12px] font-medium text-white/40 hover:text-white/80 rounded-lg transition-colors"
                >
                  {s}
                </a>
              ))}
              <a
                href="https://github.com/Iyanuoluwa007/Signlytic_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="ml-2 px-3.5 py-1.5 text-[12px] font-semibold text-white/70 bg-white/[0.06] border border-white/[0.08] rounded-lg hover:bg-white/[0.1] transition-all"
              >
                GitHub
              </a>
            </div>
          </div>
        </div>
      </nav>

      {/* ═════════ HERO ═════════ */}
      <section className="relative min-h-screen flex items-center overflow-hidden">
        {/* Radial glow */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[900px] h-[600px] bg-[radial-gradient(ellipse_at_center,rgba(14,124,107,0.08)_0%,transparent_70%)] pointer-events-none" />
        {/* Grid texture */}
        <div
          className="absolute inset-0 opacity-[0.025]"
          style={{
            backgroundImage:
              "linear-gradient(rgba(255,255,255,1) 1px, transparent 1px), linear-gradient(90deg, rgba(255,255,255,1) 1px, transparent 1px)",
            backgroundSize: "80px 80px",
          }}
        />
        {/* Noise grain */}
        <div className="absolute inset-0 opacity-[0.015]" style={{ backgroundImage: "url(\"data:image/svg+xml,%3Csvg viewBox='0 0 256 256' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='4' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='1'/%3E%3C/svg%3E\")" }} />

        <div className={`relative ${containerCls} pt-36 pb-28`}>
          <Reveal>
            <div className="inline-flex items-center gap-2 border border-white/[0.06] rounded-full px-3.5 py-1 mb-10">
              <span className="w-1.5 h-1.5 rounded-full bg-[#0e7c6b] shadow-[0_0_6px_rgba(14,124,107,0.8)]" />
              <span className="text-[11px] font-medium text-white/40 tracking-wide uppercase">
                BSL Translation System
              </span>
            </div>
          </Reveal>

          <Reveal delay={60}>
            <h1 className="text-[clamp(2.8rem,7vw,5.5rem)] font-bold text-white leading-[1.02] tracking-[-0.04em] max-w-[800px] mb-7">
              <Signable text="Translate between BSL and English">
                Translate between
                <br />
                <span className="bg-gradient-to-r from-[#5eead4] via-[#0e7c6b] to-[#065f4e] bg-clip-text text-transparent">
                  BSL and English
                </span>
              </Signable>
            </h1>
          </Reveal>

          <Reveal delay={120}>
            <p className="text-lg text-white/40 leading-[1.7] max-w-[480px] mb-12">
              <Signable text="Upload a BSL video for English text. Speak or type English for BSL signing.">
                Upload a BSL video for English text. Speak or type English for BSL
                signing.
              </Signable>{" "}
              <Signable text="Built for Deaf communities, researchers, and accessibility.">
                Built for Deaf communities, researchers, and accessibility.
              </Signable>
            </p>
          </Reveal>

          <Reveal delay={180}>
            <div className="flex flex-wrap gap-3 mb-20">
              <a
                href="/demo"
                className="group inline-flex items-center gap-2 bg-white text-[#08090d] font-semibold px-6 py-3 rounded-xl text-[14px] hover:shadow-[0_0_40px_rgba(255,255,255,0.08)] transition-all hover:-translate-y-0.5"
              >
                Try the demo
                <span className="group-hover:translate-x-0.5 transition-transform text-base">
                  &#8594;
                </span>
              </a>
              <a
                href="https://github.com/Iyanuoluwa007/Signlytic_AI"
                target="_blank"
                rel="noopener noreferrer"
                className="inline-flex items-center gap-2 border border-white/[0.1] text-white/60 font-medium px-6 py-3 rounded-xl text-[14px] hover:bg-white/[0.03] hover:border-white/[0.16] transition-all"
              >
                View source
              </a>
            </div>
          </Reveal>

          <Reveal delay={240}>
            <div className="flex gap-16">
              {[
                { n: "5,203", l: "BSL signs" },
                { n: "100%", l: "accuracy" },
                { n: "11,573+", l: "glosses" },
              ].map((s) => (
                <div key={s.l}>
                  <div className="text-[2rem] font-bold text-white/90 tracking-tight leading-none">
                    {s.n}
                  </div>
                  <div className="text-[12px] text-white/25 font-medium mt-1.5 uppercase tracking-wider">
                    {s.l}
                  </div>
                </div>
              ))}
            </div>
          </Reveal>
        </div>

        {/* Bottom edge line */}
        <div className="absolute bottom-0 left-0 right-0 h-px bg-gradient-to-r from-transparent via-white/[0.06] to-transparent" />
      </section>

      {/* ═════════ ABOUT ═════════ */}
      <section id="about" className={sectionCls}>
        <div className={containerCls}>
          <Reveal>
            <span className={labelCls}>How it works</span>
            <h2 className={h2Cls}>
              <Signable text="Bidirectional BSL translation in one system">
                Bidirectional BSL translation
                <br className="hidden md:block" />
                in one system
              </Signable>
            </h2>
            <p className={subCls}>
              <Signable text="Vision, language, speech, and animation unified into a single pipeline.">
                Vision, language, speech, and animation unified into a single
                pipeline.
              </Signable>{" "}
              <Signable text="Every output is shown as text. Nothing relies on audio.">
                Every output is shown as text&mdash;nothing relies on audio.
              </Signable>
            </p>
          </Reveal>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-px bg-white/[0.04] rounded-2xl overflow-hidden border border-white/[0.04]">
            {[
              {
                title: "BSL Video to English",
                desc: "Upload signing video. Video-SWIN-T recognises signs across 5,203 classes and translates to natural English.",
              },
              {
                title: "English to BSL Signing",
                desc: "Type or speak English. The system generates BSL glosses and renders animated skeleton-signing video.",
              },
              {
                title: "AI-Powered Translation",
                desc: "Groq Llama 3.3 70B converts BSL glosses into fluent, grammatically correct English sentences.",
              },
              {
                title: "Voice Cloning TTS",
                desc: "Coqui XTTS v2 synthesises natural speech from translated text using voice cloning with speaker reference.",
              },
              {
                title: "5,203 BSL Signs",
                desc: "Trained on BSLDict with retrieval-based recognition achieving perfect Top-1 accuracy on dictionary signs.",
              },
              {
                title: "Accessibility First",
                desc: "Plain language, high contrast, keyboard navigation, text-first output. Designed for BSL users.",
              },
            ].map((card, i) => (
              <Reveal key={card.title} delay={i * 50}>
                <div className="bg-[#0c0d12] p-8 h-full group hover:bg-[#0f1017] transition-colors duration-500">
                  <div className="text-[11px] font-bold text-white/15 mb-5 tracking-wider">
                    {String(i + 1).padStart(2, "0")}
                  </div>
                  <h3 className="text-[15px] font-semibold text-white/90 mb-2.5 tracking-[-0.01em]">
                    <Signable text={card.title}>{card.title}</Signable>
                  </h3>
                  <p className="text-[13px] text-white/35 leading-[1.7]">
                    <Signable text={card.desc}>{card.desc}</Signable>
                  </p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
        <div className={`absolute bottom-0 left-0 right-0 ${dividerCls}`} />
      </section>

      {/* ═════════ DEMO ═════════ */}
      <section id="demo" className={sectionCls}>
        <div className={containerCls}>
          <Reveal>
            <span className={labelCls}>Interactive</span>
            <h2 className={h2Cls}><Signable text="Try it live">Try it live</Signable></h2>
            <p className={subCls}>
              <Signable text="Translate between BSL glosses and English instantly.">
                Translate between BSL glosses and English instantly.
              </Signable>{" "}
              <Signable text="For the full system with video recognition, speech, and signing animation, explore the full demo.">
                For the full system with video recognition, speech, and signing
                animation &mdash; explore the full demo.
              </Signable>
            </p>
          </Reveal>

          {/* Raycast-style app preview */}
          <Reveal delay={60}>
            <div className="max-w-[640px] mx-auto">
              {/* App window */}
              <div className="bg-white/[0.025] border border-white/[0.07] rounded-2xl overflow-hidden shadow-[0_20px_60px_rgba(0,0,0,0.4)]">
                {/* Mode switcher */}
                <div className="flex bg-white/[0.02] border-b border-white/[0.06]">
                  {([
                    { id: "bsl-to-en" as const, label: "BSL \u2192 English" },
                    { id: "en-to-bsl" as const, label: "English \u2192 BSL" },
                  ]).map(m => (
                    <button
                      key={m.id}
                      onClick={() => { setDemoTab(m.id); setDemoInput(""); setDemoOutput(""); }}
                      className={`flex-1 py-3 text-[12px] font-semibold transition-all ${
                        demoTab === m.id
                          ? "text-[#5eead4] border-b-2 border-[#0e7c6b] bg-white/[0.02]"
                          : "text-white/25 border-b-2 border-transparent hover:text-white/40"
                      }`}
                    >
                      {m.label}
                    </button>
                  ))}
                </div>

                {/* Input area */}
                <div className="p-5">
                  <textarea
                    value={demoInput}
                    onChange={(e) => { setDemoInput(e.target.value); setDemoOutput(""); }}
                    placeholder={demoTab === "bsl-to-en" ? "TOMORROW MEETING WHAT TIME" : "What time is the meeting tomorrow?"}
                    rows={2}
                    className="w-full bg-transparent text-[15px] text-white/80 placeholder:text-white/15 focus:outline-none resize-none leading-relaxed font-[inherit]"
                    onKeyDown={(e) => {
                      if (e.key === "Enter" && !e.shiftKey) {
                        e.preventDefault();
                        if (demoInput.trim()) {
                          const r = demoTab === "bsl-to-en" ? glossToText(demoInput) : textToGloss(demoInput);
                          setDemoOutput(r);
                        }
                      }
                    }}
                  />
                </div>

                {/* Translate bar */}
                <div className="flex items-center justify-between px-5 py-3 border-t border-white/[0.04]">
                  <span className="text-[10px] text-white/15">Press Enter to translate</span>
                  <button
                    onClick={() => {
                      if (demoInput.trim()) {
                        const r = demoTab === "bsl-to-en" ? glossToText(demoInput) : textToGloss(demoInput);
                        setDemoOutput(r);
                      }
                    }}
                    disabled={!demoInput.trim()}
                    className="bg-white text-[#08090d] font-semibold px-4 py-1.5 rounded-lg text-[12px] hover:shadow-[0_0_20px_rgba(255,255,255,0.05)] transition-all disabled:opacity-20"
                  >
                    Translate
                  </button>
                </div>

                {/* Output */}
                {demoOutput && (
                  <div className="border-t border-[#0e7c6b]/15 bg-[#0e7c6b]/[0.03] px-5 py-4">
                    <div className="text-[9px] font-bold text-[#0e7c6b]/50 uppercase tracking-[0.12em] mb-1">
                      {demoTab === "bsl-to-en" ? "English" : "BSL Glosses"}
                    </div>
                    <div className="text-[16px] text-white/80 font-medium">{demoOutput}</div>
                  </div>
                )}
              </div>

              {/* Example chips below */}
              <div className="mt-5 flex flex-wrap justify-center gap-1.5">
                {(demoTab === "bsl-to-en"
                  ? ["TOMORROW MEETING WHAT TIME", "MY NAME SARAH", "THANK YOU MUCH", "I NOT UNDERSTAND"]
                  : ["What time is the meeting?", "Hello, my name is Sarah.", "Thank you very much.", "I need help."]
                ).map(ex => (
                  <button
                    key={ex}
                    onClick={() => { setDemoInput(ex); setDemoOutput(""); }}
                    className="px-2.5 py-1 text-[11px] text-white/25 bg-white/[0.02] border border-white/[0.05] rounded-lg hover:text-white/45 hover:border-white/[0.1] transition-all"
                  >
                    {ex}
                  </button>
                ))}
              </div>
            </div>
          </Reveal>

          {/* CTA to full demo */}
          <Reveal delay={140}>
            <div className="mt-16 text-center">
              <a
                href="/demo"
                className="group inline-flex items-center gap-2 bg-white text-[#08090d] font-semibold px-7 py-3 rounded-xl text-[14px] hover:shadow-[0_0_40px_rgba(255,255,255,0.08)] transition-all hover:-translate-y-0.5"
              >
                Explore full demo
                <span className="group-hover:translate-x-0.5 transition-transform">&#8594;</span>
              </a>
              <p className="text-white/15 text-[11px] mt-3">
                Video recognition, camera input, speech output &amp; signing animation
              </p>
            </div>
          </Reveal>
        </div>
        <div className={`absolute bottom-0 left-0 right-0 ${dividerCls}`} />
      </section>

      {/* ═════════ PERFORMANCE ═════════ */}
      <section id="performance" className={sectionCls}>
        <div className={containerCls}>
          <Reveal>
            <span className={labelCls}>Benchmarks</span>
            <h2 className={h2Cls}><Signable text="Performance">Performance</Signable></h2>
            <p className={subCls}>
              <Signable text="Benchmark results across recognition models and translation components.">
                Benchmark results across recognition models and translation
                components.
              </Signable>
            </p>
          </Reveal>

          {/* Metric strip */}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-px bg-white/[0.04] rounded-2xl overflow-hidden border border-white/[0.04] mb-16">
            {[
              { val: 100, suf: "%", label: "Top-1 Accuracy", sub: "BSL Dict Retrieval" },
              { val: 5203, suf: "", label: "BSL Signs", sub: "Dictionary coverage" },
              { val: 11573, suf: "+", label: "Glosses", sub: "Vocabulary size" },
              { val: 92, suf: "%", label: "ROUGE-L", sub: "Gloss-to-text" },
            ].map((m, i) => (
              <Reveal key={m.label} delay={i * 60}>
                <div className="bg-[#0c0d12] p-8 text-center">
                  <div className="text-[2.5rem] font-bold text-white/90 tracking-tight leading-none mb-2 tabular-nums">
                    <Counter value={m.val} suffix={m.suf} />
                  </div>
                  <div className="text-[11px] font-bold text-white/30 uppercase tracking-[0.1em] mb-0.5">
                    {m.label}
                  </div>
                  <div className="text-[11px] text-white/15">{m.sub}</div>
                </div>
              </Reveal>
            ))}
          </div>

          {/* Table */}
          <Reveal delay={80}>
            <div className="rounded-2xl overflow-hidden border border-white/[0.04]">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-white/[0.02]">
                    {["Model", "Language", "Top-1", "Top-5"].map((h) => (
                      <th
                        key={h}
                        className="text-left px-6 py-4 text-[10px] font-bold text-white/25 uppercase tracking-[0.12em]"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { m: "BSL Dict Retrieval", l: "British", t1: "100%", t5: "100%", hl: true },
                    { m: "BSL-100", l: "British", t1: "72.34%", t5: "95.03%", hl: false },
                    { m: "BSL-500", l: "British", t1: "59.26%", t5: "89.04%", hl: false },
                    { m: "Pose Recognition", l: "ASL", t1: "44.44%", t5: "81.62%", hl: false },
                    { m: "Multi-Lingual", l: "ASL+LSF", t1: "20.95%", t5: "49.17%", hl: false },
                  ].map((r, i) => (
                    <tr
                      key={r.m}
                      className="border-t border-white/[0.03] hover:bg-white/[0.01] transition-colors"
                    >
                      <td className="px-6 py-4 font-medium text-white/70">
                        {r.m}
                      </td>
                      <td className="px-6 py-4 text-white/30">{r.l}</td>
                      <td
                        className={`px-6 py-4 font-mono font-semibold ${
                          r.hl ? "text-[#0e7c6b]" : "text-white/50"
                        }`}
                      >
                        {r.t1}
                      </td>
                      <td
                        className={`px-6 py-4 font-mono font-semibold ${
                          r.hl ? "text-[#0e7c6b]" : "text-white/50"
                        }`}
                      >
                        {r.t5}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>
        </div>
        <div className={`absolute bottom-0 left-0 right-0 ${dividerCls}`} />
      </section>

      {/* ═════════ ARCHITECTURE ═════════ */}
      <section id="architecture" className={sectionCls}>
        <div className={containerCls}>
          <Reveal>
            <span className={labelCls}>System</span>
            <h2 className={h2Cls}><Signable text="Technical Architecture">Technical Architecture</Signable></h2>
            <p className={subCls}>
              <Signable text="End-to-end pipeline unifying vision, language, speech, and animation.">
                End-to-end pipeline unifying vision, language, speech, and
                animation.
              </Signable>
            </p>
          </Reveal>

          {/* Pipeline table */}
          <Reveal delay={60}>
            <div className="rounded-2xl overflow-hidden border border-white/[0.04] mb-14">
              <table className="w-full text-[13px]">
                <thead>
                  <tr className="bg-white/[0.02]">
                    {["Component", "Technology", "Details"].map((h) => (
                      <th
                        key={h}
                        className="text-left px-6 py-4 text-[10px] font-bold text-white/25 uppercase tracking-[0.12em]"
                      >
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {[
                    { c: "Sign Recognition", t: "Video-SWIN-T", d: "Retrieval on 5,203 pre-extracted 768-dim features" },
                    { c: "Speech Recognition", t: "OpenAI Whisper", d: "Base model, 16 kHz mono" },
                    { c: "Text-to-Speech", t: "Coqui XTTS v2", d: "Voice cloning with speaker reference" },
                    { c: "Language Model", t: "Groq Llama 3.3 70B", d: "Gloss to natural English" },
                    { c: "Signing Animation", t: "2D Pose Animator", d: "Skeleton signing with MP4 export" },
                    { c: "Vocabulary", t: "11,573+ glosses", d: "BSL-1K + BSLDict datasets" },
                  ].map((r) => (
                    <tr
                      key={r.c}
                      className="border-t border-white/[0.03] hover:bg-white/[0.01] transition-colors"
                    >
                      <td className="px-6 py-4 font-medium text-white/70">
                        {r.c}
                      </td>
                      <td className="px-6 py-4 font-mono text-[12px] text-[#0e7c6b]/80">
                        {r.t}
                      </td>
                      <td className="px-6 py-4 text-white/30">{r.d}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </Reveal>

          {/* Dual flow */}
          <div className="grid md:grid-cols-2 gap-4 max-w-[760px] mx-auto">
            {[
              {
                title: "BSL to English",
                steps: [
                  "BSL Video / Camera",
                  "Video-SWIN-T Recognition",
                  "BSL Glosses Extracted",
                  "Groq LLM Translation",
                  "English Text + Speech",
                ],
              },
              {
                title: "English to BSL",
                steps: [
                  "English Speech / Text",
                  "Whisper Transcription",
                  "Text to BSL Glosses",
                  "Pose Animator Rendering",
                  "BSL Signing Video",
                ],
              },
            ].map((flow, fi) => (
              <Reveal key={flow.title} delay={fi * 70}>
                <div className="bg-white/[0.015] border border-white/[0.04] rounded-2xl p-8">
                  <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.14em] mb-7">
                    {flow.title}
                  </div>
                  <div className="space-y-4">
                    {flow.steps.map((step, i) => (
                      <div key={step} className="flex items-center gap-4">
                        <div className="w-7 h-7 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center text-[11px] font-bold text-white/25 flex-shrink-0">
                          {i + 1}
                        </div>
                        <div className="flex-1 h-px bg-white/[0.04]" />
                        <span className="text-[13px] text-white/50 font-medium">
                          {step}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
        <div className={`absolute bottom-0 left-0 right-0 ${dividerCls}`} />
      </section>

      {/* ═════════ INSIGHTS ═════════ */}
      <section className={sectionCls}>
        <div className={containerCls}>
          <Reveal>
            <span className={labelCls}>Research</span>
            <h2 className={h2Cls}><Signable text="Key Insights">Key Insights</Signable></h2>
          </Reveal>

          <div className="grid md:grid-cols-3 gap-px bg-white/[0.04] rounded-2xl overflow-hidden border border-white/[0.04] mt-16">
            {[
              {
                title: "Retrieval over classification",
                desc: "Cosine similarity on 768-dim SWIN features achieves perfect accuracy across 5,203 BSL dictionary signs with one sample per class.",
              },
              {
                title: "Fast feature extraction",
                desc: "Pre-computing features for all 5,203 videos takes ~1 hour on RTX 4060 (8 GB). Inference is near-instant after extraction.",
              },
              {
                title: "Unified pipeline",
                desc: "Vision, language, speech, and animation integrated in one system with consistent API patterns and shared vocabulary.",
              },
            ].map((item, i) => (
              <Reveal key={item.title} delay={i * 60}>
                <div className="bg-[#0c0d12] p-8 h-full">
                  <div className="text-[11px] font-bold text-white/15 mb-5 tracking-wider">
                    {String(i + 1).padStart(2, "0")}
                  </div>
                  <h3 className="text-[15px] font-semibold text-white/90 mb-2.5 tracking-[-0.01em]">
                    <Signable text={item.title}>{item.title}</Signable>
                  </h3>
                  <p className="text-[13px] text-white/35 leading-[1.7]">
                    <Signable text={item.desc}>{item.desc}</Signable>
                  </p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
        <div className={`absolute bottom-0 left-0 right-0 ${dividerCls}`} />
      </section>

      {/* ═════════ AUTHOR ═════════ */}
      <section className="relative py-28 md:py-36">
        <div className={containerCls}>
          <Reveal>
            <div className="max-w-lg mx-auto text-center">
              <span className={labelCls}>Developer</span>
              <h2 className="text-[clamp(1.5rem,3.5vw,2.25rem)] font-bold text-white leading-[1.15] tracking-[-0.03em] mb-3">
                Oke Iyanuoluwa Enoch
              </h2>
              <p className="text-white/30 text-[15px] mb-3">
                <Signable text="Independent Robotics and AI Systems Engineer">
                  Independent Robotics &amp; AI Systems Engineer
                </Signable>
              </p>
              <p className="text-white/20 text-[13px] leading-relaxed mb-10 max-w-md mx-auto">
                <Signable text="Signlytic AI is part of a portfolio of production AI systems spanning algorithmic trading, multi-agent frameworks, and accessibility technology.">
                  Signlytic AI is part of a portfolio of production AI systems
                  spanning algorithmic trading, multi-agent frameworks, and
                  accessibility technology.
                </Signable>
              </p>
              <div className="flex justify-center gap-2.5">
                {[
                  {
                    label: "GitHub",
                    href: "https://github.com/Iyanuoluwa007/Signlytic_AI",
                    primary: true,
                  },
                  {
                    label: "LinkedIn",
                    href: "https://www.linkedin.com/in/iyanuoluwa-enoch-oke/",
                    primary: false,
                  },
                  {
                    label: "Portfolio",
                    href: "https://signlytic-ai-website.vercel.app",
                    primary: false,
                  },
                ].map((link) => (
                  <a
                    key={link.label}
                    href={link.href}
                    target="_blank"
                    rel="noopener noreferrer"
                    className={`px-4 py-2 rounded-lg text-[12px] font-semibold transition-all ${
                      link.primary
                        ? "bg-white text-[#08090d] hover:shadow-[0_0_30px_rgba(255,255,255,0.06)]"
                        : "border border-white/[0.08] text-white/50 hover:text-white/70 hover:border-white/[0.14]"
                    }`}
                  >
                    {link.label}
                  </a>
                ))}
              </div>
            </div>
          </Reveal>
        </div>
      </section>

      {/* ═════════ FEEDBACK & EXTENSION ═════════ */}
      <section className="border-t border-white/[0.04] py-16">
        <div className={containerCls}>
          <div className="grid md:grid-cols-3 gap-4">
            {/* Feedback */}
            <Reveal>
              {/* The link is stretched over the whole card via after:inset-0
                  rather than wrapping it, so the sentences inside can be
                  their own controls. Signable text sits above that overlay
                  on z-10 and takes its own clicks. */}
              <div className="relative bg-white/[0.02] border border-white/[0.06] rounded-2xl p-7 hover:border-[#0e7c6b]/30 transition-all group h-full flex flex-col">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-[#0e7c6b]/10 border border-[#0e7c6b]/20 flex items-center justify-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5eead4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>
                  </div>
                  <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.14em]">Community</div>
                </div>
                <h3 className="text-[15px] font-semibold text-white/85 mb-1.5 group-hover:text-white transition-colors">
                  <span className="relative z-10">
                    <Signable text="Share your feedback">Share your feedback</Signable>
                  </span>
                </h3>
                <p className="text-[13px] text-white/30 leading-relaxed">
                  <span className="relative z-10">
                    <Signable text="Researchers, engineers, BSL interpreters, and deaf community members, your insights help shape the next version.">
                      Researchers, engineers, BSL interpreters, and deaf community members &mdash; your insights help shape the next version.
                    </Signable>
                  </span>
                </p>
                <a
                  href="https://forms.gle/oTy7Bi414fuThFc1A"
                  target="_blank"
                  rel="noopener noreferrer"
                  className="mt-4 text-[12px] text-[#0e7c6b]/60 font-medium group-hover:text-[#0e7c6b] transition-colors after:absolute after:inset-0 after:content-[''] after:rounded-2xl"
                >
                  Open feedback form &#8599;
                </a>
              </div>
            </Reveal>

            {/* Extension */}
            <Reveal delay={60}>
              <div className="relative bg-white/[0.02] border border-white/[0.06] rounded-2xl p-7 hover:border-white/[0.1] transition-all group h-full flex flex-col">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-white/[0.04] border border-white/[0.06] flex items-center justify-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-white/30"><circle cx="12" cy="12" r="10"/><path d="M2 12h20"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>
                  </div>
                  <div className="text-[10px] font-bold text-[#0e7c6b] uppercase tracking-[0.14em]">Beta Available</div>
                </div>
                <h3 className="text-[15px] font-semibold text-white/85 mb-1.5 group-hover:text-white transition-colors">
                  <span className="relative z-10">
                    <Signable text="Browser Extension">Browser Extension</Signable>
                  </span>
                </h3>
                <p className="text-[13px] text-white/30 leading-relaxed">
                  <span className="relative z-10">
                    <Signable text="Real-time BSL signing overlay for YouTube, BBC iPlayer, Netflix and more. Free to download.">
                      Real-time BSL signing overlay for YouTube, BBC iPlayer, Netflix and more. Free to download.
                    </Signable>
                  </span>
                </p>
                <a
                  href="/extension"
                  className="mt-4 text-[12px] text-white/25 font-medium group-hover:text-white/50 transition-colors after:absolute after:inset-0 after:content-[''] after:rounded-2xl"
                >
                  Download Beta &#8594;
                </a>
              </div>
            </Reveal>

            {/* Desktop app */}
            <Reveal delay={120}>
              <div className="relative bg-white/[0.02] border border-white/[0.06] rounded-2xl p-7 hover:border-[#0e7c6b]/30 transition-all group h-full flex flex-col">
                <div className="flex items-center gap-3 mb-3">
                  <div className="w-8 h-8 rounded-lg bg-[#0e7c6b]/10 border border-[#0e7c6b]/20 flex items-center justify-center">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#5eead4" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="3" width="20" height="14" rx="2"/><path d="M8 21h8"/><path d="M12 17v4"/></svg>
                  </div>
                  <div className="text-[10px] font-bold text-[#0e7c6b] uppercase tracking-[0.14em]">Beta Available</div>
                </div>
                <h3 className="text-[15px] font-semibold text-white/85 mb-1.5 group-hover:text-white transition-colors">
                  <span className="relative z-10">
                    <Signable text="Desktop App">Desktop App</Signable>
                  </span>
                </h3>
                <p className="text-[13px] text-white/30 leading-relaxed">
                  <span className="relative z-10">
                    <Signable text="Signs what your computer is saying, in any app, using Windows Live Captions.">
                      Signs what your computer is saying, in any app, using Windows Live Captions.
                    </Signable>
                  </span>
                </p>
                <div className="mt-auto pt-4 flex flex-col gap-1.5">
                  <a
                    href={DESKTOP_WINDOWS_URL}
                    className="text-[12px] text-[#0e7c6b]/70 font-medium group-hover:text-[#0e7c6b] transition-colors after:absolute after:inset-0 after:content-[''] after:rounded-2xl"
                  >
                    Download for Windows &#8594;
                  </a>
                  <span className="text-[11px] text-white/20">
                    Mac app upcoming &middot;{" "}
                    <span className="relative z-10 underline underline-offset-2 hover:text-white/40">
                      <a href="/extension#desktop">see details</a>
                    </span>
                  </span>
                </div>
              </div>
            </Reveal>
          </div>
        </div>
      </section>

      {/* ═════════ FOOTER ═════════ */}
      <footer className="border-t border-white/[0.04] py-12">
        <div className={`${containerCls} text-center`}>
          <div className="flex items-center justify-center gap-2 mb-5">
            <div className="w-5 h-5 rounded-[5px] bg-[#0e7c6b] flex items-center justify-center">
              <svg
                width="10"
                height="10"
                viewBox="0 0 24 24"
                fill="none"
                stroke="white"
                strokeWidth="3"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M12 2L2 7l10 5 10-5-10-5z" />
                <path d="M2 17l10 5 10-5" />
                <path d="M2 12l10 5 10-5" />
              </svg>
            </div>
            <span className="text-[13px] font-semibold text-white/60">
              Signlytic AI
            </span>
          </div>
          <p className="text-white/15 text-[12px] mb-5">
            Bridging communication between BSL users and hearing communities
          </p>
          <div className="flex justify-center gap-6 text-[11px] text-white/15 mb-6">
            {[
              {
                l: "GitHub",
                h: "https://github.com/Iyanuoluwa007/Signlytic_AI",
              },
              {
                l: "Demo",
                h: "/demo",
              },
              {
                l: "Give Feedback",
                h: "https://forms.gle/oTy7Bi414fuThFc1A",
              },
              {
                l: "Browser Extension",
                h: "/extension",
              },
              {
                l: "LinkedIn",
                h: "https://www.linkedin.com/in/iyanuoluwa-enoch-oke/",
              },
            ].map((a) => (
              <a
                key={a.l}
                href={a.h}
                target={a.h.startsWith("http") ? "_blank" : undefined}
                rel={a.h.startsWith("http") ? "noopener noreferrer" : undefined}
                className="hover:text-white/40 transition-colors"
              >
                {a.l}
              </a>
            ))}
          </div>
          <div className="text-white/[0.07] text-[10px] tracking-wider uppercase">
            Independent Robotics &amp; AI Systems Engineer &middot; v2.0 &middot;
            March 2026
          </div>
        </div>
      </footer>

      {/* ═════════ STYLES ═════════ */}
      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap");
        html {
          scroll-behavior: smooth;
          -webkit-font-smoothing: antialiased;
        }
        body {
          font-family: "Outfit", system-ui, -apple-system, sans-serif;
          background: #08090d;
        }
        .font-mono {
          font-family: "JetBrains Mono", monospace;
        }
        .tabular-nums {
          font-variant-numeric: tabular-nums;
        }
      `}</style>
    </div>
  );
}
