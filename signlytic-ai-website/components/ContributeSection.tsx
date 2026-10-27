"use client";

// The call to contribute, shown on the home page and at the foot of the demo.
// Two audiences: developers, who go to the code and its issues, and BSL users
// and accessibility specialists, whose feedback on the signing matters more
// than any code. The clip is the same one at the top of the README.
//
// The video never autoplays: a visitor starts it, which also suits anyone who
// has asked their system for reduced motion.
import Signable from "./Signable";

const REPO = "https://github.com/Iyanuoluwa007/Signlytic_AI";
const OVERLAY = "https://github.com/Iyanuoluwa007/Signlytic-Overlay";
const FEEDBACK_FORM = "https://forms.gle/oTy7Bi414fuThFc1A";

export default function ContributeSection({ compact = false }: { compact?: boolean }) {
  return (
    <div className={compact ? "max-w-[960px] mx-auto" : ""}>
      {compact && (
        <h2 className="text-[1.35rem] font-bold text-white/90 mb-6 tracking-[-0.02em]">
          <Signable text="Contribute to Signlytic AI">Contribute to Signlytic AI</Signable>
        </h2>
      )}
      <div className={`grid gap-10 items-center ${compact ? "md:grid-cols-[1.1fr_1fr]" : "md:grid-cols-[1.15fr_1fr]"}`}>
        <figure className="m-0">
          <video
            className="w-full rounded-2xl border border-white/[0.07] bg-black shadow-[0_20px_60px_rgba(0,0,0,0.4)]"
            src="/media/signlytic-demo.mp4"
            poster="/media/signlytic-demo-poster.jpg"
            controls
            muted
            playsInline
            preload="none"
            aria-describedby="contribute-clip-caption"
          />
          <figcaption id="contribute-clip-caption" className="mt-3 text-[12px] text-white/35 leading-relaxed">
            The 3D avatar and the 2D skeleton signing &ldquo;Do you take the bus to work?&rdquo;,
            then &ldquo;No.&rdquo; and &ldquo;Yes.&rdquo;. The English is shown above, and each
            BSL sign is named below as it is signed. There is no sound.
          </figcaption>
        </figure>

        <div>
          <p className="text-[15px] text-white/55 leading-relaxed mb-6">
            <Signable text="Signlytic AI is open source. You can work on it without an NVIDIA GPU or any paid API key.">
              Signlytic AI is open source. You can work on it without an NVIDIA GPU
              or any paid API key.
            </Signable>
          </p>

          <div className="flex flex-wrap gap-3 mb-3">
            <a
              href={`${REPO}/blob/main/CONTRIBUTING.md`}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 bg-white text-[#08090d] font-semibold px-5 py-2.5 rounded-xl text-[14px] transition-all hover:-translate-y-0.5 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#5eead4]"
            >
              Contribute code
            </a>
            <a
              href={OVERLAY}
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-2 font-semibold px-5 py-2.5 rounded-xl text-[14px] text-white/80 bg-white/[0.06] border border-white/[0.1] hover:bg-white/[0.1] transition-all focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#5eead4]"
            >
              Explore the apps
            </a>
          </div>
          <p className="text-[12px] text-white/35 mb-8">
            New here? Try a{" "}
            <a
              href={`${REPO}/issues?q=is%3Aopen+label%3A%22good+first+issue%22`}
              target="_blank"
              rel="noopener noreferrer"
              className="text-[#5eead4]/80 underline underline-offset-2 hover:text-[#5eead4]"
            >
              good first issue
            </a>
            .
          </p>

          <div className="border-l-2 border-[#0e7c6b] pl-4">
            <p className="text-[14px] font-semibold text-white/80 mb-1">
              <Signable text="BSL users and accessibility specialists">
                BSL users and accessibility specialists
              </Signable>
            </p>
            <p className="text-[13px] text-white/45 leading-relaxed">
              Your view of how the signing looks is the most valuable help this project
              can get. Tell us in the{" "}
              <a
                href={FEEDBACK_FORM}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#5eead4]/80 underline underline-offset-2 hover:text-[#5eead4]"
              >
                feedback form
              </a>
              {" "}(no account needed) or with the{" "}
              <a
                href={`${REPO}/issues/new?template=bsl_signing_feedback.yml`}
                target="_blank"
                rel="noopener noreferrer"
                className="text-[#5eead4]/80 underline underline-offset-2 hover:text-[#5eead4]"
              >
                BSL signing feedback form
              </a>
              {" "}on GitHub.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
