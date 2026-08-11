"use client";

// Wraps a sentence and adds a trigger that plays its BSL translation in the
// shared BslSignPanel. Site-wide rollout means wrapping more sentences in
// this component; the panel itself is mounted once in the root layout.
import { ReactNode, useEffect, useState } from "react";

export const BSL_SIGN_REQUEST_EVENT = "signlytic-sign-request";
export const BSL_TOGGLE_EVENT = "signlytic-bsl-toggle";
export const BSL_ENABLED_KEY = "signlytic-bsl-enabled";

// Default on: this is an accessibility feature, so it should be present
// unless a visitor has deliberately turned it off.
export function readBslEnabled(): boolean {
  try {
    return localStorage.getItem(BSL_ENABLED_KEY) !== "off";
  } catch {
    return true;
  }
}

// Shared by Signable and the toggle so both react to the same signal.
export function useBslEnabled(): boolean {
  // Start false so server and first client render agree; the effect turns it
  // on immediately after mount. Avoids a hydration mismatch.
  const [enabled, setEnabled] = useState(false);
  useEffect(() => {
    setEnabled(readBslEnabled());
    const onToggle = (e: Event) => {
      const detail = (e as CustomEvent<{ enabled: boolean }>).detail;
      setEnabled(detail ? detail.enabled : readBslEnabled());
    };
    document.addEventListener(BSL_TOGGLE_EVENT, onToggle);
    return () => document.removeEventListener(BSL_TOGGLE_EVENT, onToggle);
  }, []);
  return enabled;
}

export default function Signable({
  text,
  children,
}: {
  text: string;
  children: ReactNode;
}) {
  const enabled = useBslEnabled();

  if (!enabled) return <>{children}</>;

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
        className="ml-1.5 align-middle inline-flex items-center px-1.5 py-px rounded border border-[#0e7c6b]/40 bg-[#0e7c6b]/10 text-[10px] font-semibold tracking-wide text-[#5eead4]/80 hover:bg-[#0e7c6b]/25 hover:text-[#5eead4] focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5eead4]/70 transition-colors"
      >
        BSL
      </button>
    </span>
  );
}
