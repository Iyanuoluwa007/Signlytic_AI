"use client";

// Wraps a sentence and makes the whole sentence the play target: hovering
// highlights it and offers playback, and clicking anywhere on it plays the
// BSL translation in the shared BslSignPanel. The small chip is a visual
// affordance only - the accessible control is the sentence itself.
//
// Site-wide rollout means wrapping more sentences in this component; the
// panel is mounted once in the root layout.
import { ReactNode, useEffect, useState } from "react";

export const BSL_SIGN_REQUEST_EVENT = "signlytic-sign-request";
export const BSL_TOGGLE_EVENT = "signlytic-bsl-toggle";
export const BSL_ENABLED_KEY = "signlytic-bsl-enabled";

// Default on: this is an accessibility feature, so it is present unless a
// visitor has deliberately turned it off.
export function readBslEnabled(): boolean {
  try {
    return localStorage.getItem(BSL_ENABLED_KEY) !== "off";
  } catch {
    return true;
  }
}

// Starts true so the server render and the first client render agree AND the
// affordance is present in the initial HTML. The effect then corrects it for
// the minority who have switched it off.
export function useBslEnabled(): boolean {
  const [enabled, setEnabled] = useState(true);
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
  const [active, setActive] = useState(false);

  if (!enabled) return <>{children}</>;

  const trigger = () => {
    document.dispatchEvent(
      new CustomEvent(BSL_SIGN_REQUEST_EVENT, { detail: { text } })
    );
  };

  return (
    <span
      role="button"
      tabIndex={0}
      onClick={trigger}
      onKeyDown={(e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          trigger();
        }
      }}
      onMouseEnter={() => setActive(true)}
      onMouseLeave={() => setActive(false)}
      onFocus={() => setActive(true)}
      onBlur={() => setActive(false)}
      aria-label={"Play British Sign Language translation of: " + text}
      className={
        "cursor-pointer rounded px-0.5 -mx-0.5 transition-colors duration-150 " +
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5eead4]/70 " +
        (active ? "bg-[#0e7c6b]/15 text-white/80" : "")
      }
    >
      {children}
      <span
        aria-hidden="true"
        className={
          "ml-1.5 align-middle inline-flex items-center whitespace-nowrap rounded border px-1.5 py-px " +
          "text-[10px] font-semibold tracking-wide transition-colors duration-150 " +
          (active
            ? "border-[#0e7c6b]/70 bg-[#0e7c6b]/30 text-[#5eead4]"
            : "border-[#0e7c6b]/40 bg-[#0e7c6b]/10 text-[#5eead4]/70")
        }
      >
        {active ? "Play BSL" : "BSL"}
      </span>
    </span>
  );
}
