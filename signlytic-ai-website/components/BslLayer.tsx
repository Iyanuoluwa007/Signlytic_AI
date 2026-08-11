"use client";

// Mounted once in the root layout. Provides the site-wide BSL on/off control
// and the single shared translation panel.
import { useEffect, useState } from "react";
import BslSignPanel from "./BslSignPanel";
import {
  BSL_ENABLED_KEY,
  BSL_TOGGLE_EVENT,
  readBslEnabled,
} from "./Signable";

function BslToggle() {
  const [enabled, setEnabled] = useState(true);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setEnabled(readBslEnabled());
    setMounted(true);
  }, []);

  const toggle = () => {
    const next = !enabled;
    setEnabled(next);
    try {
      localStorage.setItem(BSL_ENABLED_KEY, next ? "on" : "off");
    } catch {
      // non-persistent session is still usable
    }
    document.dispatchEvent(
      new CustomEvent(BSL_TOGGLE_EVENT, { detail: { enabled: next } })
    );
  };

  // Render nothing until mounted so the server and client markup match
  if (!mounted) return null;

  return (
    <button
      type="button"
      onClick={toggle}
      role="switch"
      aria-checked={enabled}
      aria-label="British Sign Language translations"
      title={
        enabled
          ? "BSL translations are on. Click to turn off."
          : "BSL translations are off. Click to turn on."
      }
      className={
        "fixed left-3 bottom-3 z-40 flex items-center gap-2 rounded-full border px-3 py-1.5 " +
        "text-[11px] font-semibold shadow-lg shadow-black/40 transition-colors " +
        "focus:outline-none focus-visible:ring-2 focus-visible:ring-[#5eead4]/70 " +
        (enabled
          ? "border-[#0e7c6b]/50 bg-[#0e7c6b]/20 text-[#5eead4] hover:bg-[#0e7c6b]/35"
          : "border-white/10 bg-[#0b0d13]/90 text-white/40 hover:text-white/70")
      }
    >
      <span
        aria-hidden="true"
        className={
          "inline-block h-1.5 w-1.5 rounded-full " +
          (enabled ? "bg-[#5eead4]" : "bg-white/30")
        }
      />
      BSL {enabled ? "on" : "off"}
    </button>
  );
}

export default function BslLayer() {
  return (
    <>
      <BslToggle />
      <BslSignPanel />
    </>
  );
}
