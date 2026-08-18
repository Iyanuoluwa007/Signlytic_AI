// Investigation tool, not shipped in the app.
//
// Answers one question: does the macOS Live Captions window expose its
// transcript text through the Accessibility API, the way the Windows Live
// Captions window exposes it through UI Automation?
//
// It does. The window is AXLiveCaptionsWindow and the text sits in AXStaticText
// children, one per line, already punctuated. This tool is kept so the finding
// can be re-checked against a new macOS release rather than trusted.
//
// Run it with audio actually playing. The window does not exist while there is
// nothing to caption, so a run against a silent machine reports no windows and
// reads exactly like a dead end. That is what it looked like on the first run
// here, and it was wrong.
//
// Build and run:
//   swiftc -O main/captions/mac/tools/ax-probe.swift -o /tmp/ax-probe
//   /tmp/ax-probe
//
// Needs Live Captions switched on in System Settings, Accessibility, and
// Accessibility permission granted to whatever runs this.
//
//   say "the weather is good today" & swiftc ... && /tmp/ax-probe

import Foundation
import ApplicationServices
import AppKit

let LIVE_CAPTIONS_BUNDLE_ID = "com.apple.accessibility.LiveTranscriptionAgent"

func attributeNames(_ element: AXUIElement) -> [String] {
    var names: CFArray?
    guard AXUIElementCopyAttributeNames(element, &names) == .success,
          let list = names as? [String] else { return [] }
    return list
}

func value(_ element: AXUIElement, _ attribute: String) -> CFTypeRef? {
    var out: CFTypeRef?
    guard AXUIElementCopyAttributeValue(element, attribute as CFString, &out) == .success else { return nil }
    return out
}

func describe(_ raw: CFTypeRef?) -> String {
    guard let raw = raw else { return "nil" }
    if let s = raw as? String { return "\"\(s)\"" }
    if let n = raw as? NSNumber { return n.stringValue }
    if let arr = raw as? [AnyObject] { return "[\(arr.count) items]" }
    return String(describing: raw)
}

// Attributes worth calling out: any of these carrying the transcript would be
// the macOS equivalent of CaptionsTextBlock.
let TEXT_ATTRIBUTES = [
    kAXValueAttribute, kAXTitleAttribute, kAXDescriptionAttribute,
    kAXHelpAttribute, kAXPlaceholderValueAttribute, kAXSelectedTextAttribute,
]

var foundText: [String] = []

func dump(_ element: AXUIElement, depth: Int, maxDepth: Int) {
    if depth > maxDepth { return }
    let pad = String(repeating: "  ", count: depth)
    let names = attributeNames(element)
    let role = (value(element, kAXRoleAttribute) as? String) ?? "?"
    let subrole = (value(element, kAXSubroleAttribute) as? String) ?? ""
    let identifier = (value(element, kAXIdentifierAttribute) as? String) ?? ""

    var line = "\(pad)\(role)"
    if !subrole.isEmpty { line += " [\(subrole)]" }
    if !identifier.isEmpty { line += " id=\(identifier)" }
    print(line)

    for attr in TEXT_ATTRIBUTES where names.contains(attr) {
        let v = value(element, attr)
        if let s = v as? String, !s.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            print("\(pad)  \(attr) = \"\(s)\"")
            foundText.append("\(role).\(attr): \(s)")
        }
    }

    // Print the full attribute list once per element so nothing non-obvious is
    // missed. A transcript could sit behind an attribute we did not guess.
    let others = names.filter { !TEXT_ATTRIBUTES.contains($0) && $0 != kAXChildrenAttribute }
    if !others.isEmpty {
        print("\(pad)  attrs: \(others.joined(separator: ", "))")
    }

    if let children = value(element, kAXChildrenAttribute) as? [AXUIElement] {
        for child in children { dump(child, depth: depth + 1, maxDepth: maxDepth) }
    }
}

// ---------------------------------------------------------------------------

print("AXIsProcessTrusted: \(AXIsProcessTrusted())")
if !AXIsProcessTrusted() {
    // Prompting here is what puts this binary into the Accessibility list, so
    // it can be ticked. Without that, every read below returns nothing and the
    // result would look like "no text exposed" when it is really "no access".
    let options = [kAXTrustedCheckOptionPrompt.takeUnretainedValue() as String: true] as CFDictionary
    _ = AXIsProcessTrustedWithOptions(options)
    print("")
    print("No Accessibility permission. A prompt should have appeared.")
    print("Grant it in System Settings, Privacy and Security, Accessibility,")
    print("then run this again. The entry will be named after the binary.")
    exit(2)
}

let running = NSWorkspace.shared.runningApplications
guard let app = running.first(where: { $0.bundleIdentifier == LIVE_CAPTIONS_BUNDLE_ID }) else {
    print("")
    print("Live Captions is not running.")
    print("Switch it on in System Settings, Accessibility, Live Captions,")
    print("then run this again.")
    print("")
    print("Running accessibility-related processes, for reference:")
    for a in running {
        if let b = a.bundleIdentifier, b.contains("accessibility") || b.contains("Transcription") {
            print("  \(b) pid=\(a.processIdentifier)")
        }
    }
    exit(3)
}

print("Live Captions pid: \(app.processIdentifier)")
let axApp = AXUIElementCreateApplication(app.processIdentifier)

print("")
print("Application-level attributes: \(attributeNames(axApp).joined(separator: ", "))")

if let windows = value(axApp, kAXWindowsAttribute) as? [AXUIElement] {
    print("windows: \(windows.count)")
} else {
    print("windows: none reported")
}

print("")
print("---- AX tree ----")
dump(axApp, depth: 0, maxDepth: 12)

print("")
print("---- verdict ----")
if foundText.isEmpty {
    print("No readable text found anywhere in the Live Captions AX tree.")
    print("Check that captions were actually on screen while this ran: the")
    print("window is not created at all while there is nothing to caption, and")
    print("an idle run looks identical to the route being unavailable.")
} else {
    print("Readable text found (\(foundText.count) strings):")
    for t in foundText { print("  \(t)") }
}
