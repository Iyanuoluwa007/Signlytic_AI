"use client";

import { useState } from "react";

export default function ExtensionPage() {
  const [email, setEmail] = useState("");
  const [submitted, setSubmitted] = useState(false);

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
            <span className="text-white/35 text-[13px]">Browser Extension</span>
          </a>
          <a href="/" className="text-[12px] text-white/30 hover:text-white/60 transition-colors">&larr; Back to site</a>
        </div>
      </nav>

      {/* Hero */}
      <section className="relative pt-24 pb-20 overflow-hidden">
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-[800px] h-[500px] bg-[radial-gradient(ellipse_at_center,rgba(14,124,107,0.06)_0%,transparent_70%)] pointer-events-none" />

        <div className="relative max-w-[680px] mx-auto px-6 text-center">
          {/* Badge */}
          <div className="inline-flex items-center gap-2 border border-white/[0.06] rounded-full px-4 py-1.5 mb-8">
            <span className="w-1.5 h-1.5 rounded-full bg-amber-400 shadow-[0_0_6px_rgba(251,191,36,0.6)]" />
            <span className="text-[11px] font-medium text-white/35 tracking-wide uppercase">Coming Soon</span>
          </div>

          {/* Icon */}
          <div className="w-16 h-16 rounded-2xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mx-auto mb-8">
            <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#5eead4" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="10"/>
              <path d="M2 12h20"/>
              <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/>
            </svg>
          </div>

          <h1 className="text-[clamp(1.8rem,4.5vw,2.8rem)] font-bold text-white leading-[1.1] tracking-[-0.03em] mb-5">
            BSL signing from
            <br />
            <span className="text-[#5eead4]">any screen caption</span>
          </h1>

          <p className="text-white/35 text-[16px] leading-relaxed max-w-lg mx-auto mb-10">
            A Chrome extension that detects live captions on YouTube, Google Meet, Zoom, and Microsoft Teams — and translates them into BSL signing in real time.
          </p>

          {/* How it works */}
          <div className="flex items-center justify-center gap-2 mb-12 text-[12px] text-white/20">
            <span className="px-3 py-1.5 bg-white/[0.03] border border-white/[0.06] rounded-lg">Screen captions</span>
            <span>&#8594;</span>
            <span className="px-3 py-1.5 bg-white/[0.03] border border-white/[0.06] rounded-lg">Text to glosses</span>
            <span>&#8594;</span>
            <span className="px-3 py-1.5 bg-[#0e7c6b]/10 border border-[#0e7c6b]/20 rounded-lg text-[#5eead4]">BSL signing overlay</span>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="pb-20">
        <div className="max-w-[900px] mx-auto px-6">
          <div className="grid md:grid-cols-3 gap-px bg-white/[0.04] rounded-2xl overflow-hidden border border-white/[0.04]">
            {[
              {
                num: "01",
                title: "Caption Detection",
                desc: "Automatically detects and extracts live captions from YouTube subtitles, Google Meet, Zoom, and Microsoft Teams.",
              },
              {
                num: "02",
                title: "Real-Time Translation",
                desc: "Converts English captions to BSL glosses using the same translation engine powering the Signlytic AI application.",
              },
              {
                num: "03",
                title: "Signing Overlay",
                desc: "Displays a floating panel with 2D skeleton or 3D character signing each gloss in sequence alongside the video.",
              },
            ].map((c) => (
              <div key={c.title} className="bg-[#0c0d12] p-7 h-full">
                <div className="text-[11px] font-bold text-white/12 tracking-wider mb-5">{c.num}</div>
                <h3 className="text-[14px] font-semibold text-white/85 mb-2">{c.title}</h3>
                <p className="text-[12px] text-white/30 leading-[1.7]">{c.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Supported platforms */}
      <section className="border-t border-white/[0.04] py-16">
        <div className="max-w-[680px] mx-auto px-6 text-center">
          <div className="text-[10px] font-bold text-white/20 uppercase tracking-[0.14em] mb-6">Supported Platforms</div>
          <div className="flex justify-center gap-8">
            {["YouTube", "Google Meet", "Zoom", "Microsoft Teams"].map((p) => (
              <div key={p} className="text-[13px] text-white/25 font-medium">{p}</div>
            ))}
          </div>
        </div>
      </section>

      {/* Animation modes */}
      <section className="border-t border-white/[0.04] py-16">
        <div className="max-w-[680px] mx-auto px-6">
          <div className="text-center mb-10">
            <div className="text-[10px] font-bold text-[#0e7c6b] uppercase tracking-[0.14em] mb-3">Rendering</div>
            <h2 className="text-[1.4rem] font-bold text-white tracking-[-0.02em]">Two signing modes</h2>
          </div>
          <div className="grid md:grid-cols-2 gap-4">
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-4">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white/25"><circle cx="12" cy="5" r="2"/><path d="M12 7v6"/><path d="M8 11l4 2 4-2"/><path d="M8 21l4-6 4 6"/></svg>
              </div>
              <h3 className="text-[14px] font-semibold text-white/80 mb-1">2D Skeleton</h3>
              <p className="text-[12px] text-white/30 leading-relaxed">Lightweight stick-figure animation using the existing Pose Animator. Runs in browser Canvas with minimal resources.</p>
            </div>
            <div className="bg-white/[0.02] border border-white/[0.06] rounded-2xl p-6">
              <div className="w-10 h-10 rounded-xl bg-white/[0.03] border border-white/[0.06] flex items-center justify-center mb-4">
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" className="text-white/25"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
              </div>
              <h3 className="text-[14px] font-semibold text-white/80 mb-1">3D Character</h3>
              <p className="text-[12px] text-white/30 leading-relaxed">Full 3D avatar with pre-baked sign animations using Three.js or WebGL. More expressive but requires more resources.</p>
            </div>
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="border-t border-white/[0.04] py-16">
        <div className="max-w-[480px] mx-auto px-6 text-center">
          <h2 className="text-[1.2rem] font-bold text-white/85 mb-3 tracking-[-0.01em]">Want to be notified when it launches?</h2>
          <p className="text-[13px] text-white/30 mb-6">Leave your feedback and let us know this feature matters to you.</p>
          <a
            href="https://forms.gle/oTy7Bi414fuThFc1A"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-2 bg-white text-[#08090d] font-semibold px-6 py-2.5 rounded-xl text-[13px] hover:shadow-[0_0_30px_rgba(255,255,255,0.06)] transition-all hover:-translate-y-0.5"
          >
            Share feedback
            <span>&#8599;</span>
          </a>
        </div>
      </section>

      {/* Footer */}
      <footer className="border-t border-white/[0.04] py-8">
        <div className="max-w-[1120px] mx-auto px-6 text-center">
          <a href="/" className="inline-flex items-center gap-2 mb-3">
            <div className="w-5 h-5 rounded-[5px] bg-[#0e7c6b] flex items-center justify-center">
              <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="white" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round"><path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/></svg>
            </div>
            <span className="text-[13px] font-semibold text-white/50">Signlytic AI</span>
          </a>
          <p className="text-white/15 text-[11px]">Independent Robotics &amp; AI Systems Engineer &middot; v2.0 &middot; March 2026</p>
        </div>
      </footer>

      <style jsx global>{`
        @import url("https://fonts.googleapis.com/css2?family=Outfit:wght@400;500;600;700;800&display=swap");
        html { scroll-behavior: smooth; }
        body { font-family: "Outfit", system-ui, -apple-system, sans-serif; background: #08090d; color: #e8eaed; }
        ::selection { background: rgba(14,124,107,0.3); color: #fff; }
      `}</style>
    </div>
  );
}
