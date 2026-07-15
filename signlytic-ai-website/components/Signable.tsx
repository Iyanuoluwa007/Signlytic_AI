"use client";

// Wraps a single sentence and adds a trigger that requests its BSL
// translation in the shared BslSignPanel. Site-wide rollout means wrapping
// more sentences in this component; the panel is mounted once per page.
import { ReactNode } from "react";

export const BSL_SIGN_REQUEST_EVENT = "signlytic-sign-request";

export default function Signable({
  text,
  children,
}: {
  text: string;
  children: ReactNode;
}) {
  const trigger = () => {
    document.dispatchEvent(
      new CustomEvent(BSL_SIGN_REQUEST_EVENT, { detail: { text } })
    );
  };

  return (
    <span className="inline">
      {children}
      <button
        type="button"
        onClick={trigger}
        aria-label={"Play British Sign Language translation of: " + text}
        title="Play BSL translation"
        className="ml-1.5 align-middle inline-flex items-center px-1.5 py-px rounded border border-[#0e7c6b]/40 bg-[#0e7c6b]/10 text-[10px] font-semibold tracking-wide text-[#5eead4]/80 hover:bg-[#0e7c6b]/25 hover:text-[#5eead4] transition-colors"
      >
        BSL
      </button>
    </span>
  );
}
