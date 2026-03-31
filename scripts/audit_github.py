# scripts/audit_github.py
# Audits the git-tracked files to ensure no non-public data is present.
# Checks for: large files, video/model files, raw landmark data, API keys, credentials.

import subprocess
import sys
from pathlib import Path

BASE = Path(r"D:\Signlytic_AI\code\bsl_translation_project")

print("=" * 60)
print("SIGNLYTIC AI -- GITHUB AUDIT")
print("=" * 60)

# 1. Get all tracked files
result = subprocess.run(
    ["git", "ls-files"],
    capture_output=True, text=True, cwd=BASE
)
tracked = [Path(f) for f in result.stdout.strip().split("\n") if f]
print(f"\n[INFO] Total tracked files: {len(tracked)}")

# 2. Check for large files (>1MB)
print("\n--- Files over 1MB ---")
large = []
for f in tracked:
    full = BASE / f
    if full.exists() and full.stat().st_size > 1_000_000:
        mb = full.stat().st_size / 1_000_000
        large.append((mb, f))
if large:
    for mb, f in sorted(large, reverse=True):
        print(f"  {mb:.1f}MB  {f}")
else:
    print("  [OK] No files over 1MB")

# 3. Check for dangerous file extensions
print("\n--- Dangerous extensions (video/binary/model/data) ---")
BAD_EXTS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".pkl", ".pickle", ".h5", ".hdf5", ".pt", ".pth", ".onnx",
    ".npy", ".npz",
    ".glb", ".gltf",
    ".task",
    ".wav", ".mp3",
    ".zip", ".tar", ".gz",
}
found_bad = []
for f in tracked:
    if f.suffix.lower() in BAD_EXTS:
        full = BASE / f
        mb = full.stat().st_size / 1_000_000 if full.exists() else 0
        found_bad.append((f.suffix, mb, f))
if found_bad:
    for ext, mb, f in sorted(found_bad):
        print(f"  [{ext}] {mb:.1f}MB  {f}")
else:
    print("  [OK] No dangerous file types tracked")

# 4. Check for known non-public data paths
print("\n--- Known non-public data paths ---")
BANNED_PATHS = [
    "data/videos", "data/processed", "data/bsl_dict_features",
    "data/poses_bsldict", "data/bsldict", "data/BSL-1K",
    "data/mediapipe_models", "extension-data/signs",
    "signlytic-ai-website/public/signs",
]
for bp in BANNED_PATHS:
    matches = [f for f in tracked if str(f).replace("\\", "/").startswith(bp)]
    if matches:
        print(f"  [WARN] {bp}/ -> {len(matches)} files tracked!")
        for m in matches[:3]:
            print(f"         {m}")
        if len(matches) > 3:
            print(f"         ... and {len(matches)-3} more")
    else:
        print(f"  [OK]  {bp}/")

# 5. Scan for credentials / API keys in tracked text files
print("\n--- Credential scan (API keys, tokens, passwords) ---")
KEYWORDS = [
    "api_key", "api_secret", "password", "token", "secret",
    "ALPACA_API", "sk-", "Bearer ", "private_key",
    "firebase", "DATABASE_SECRET",
]
TEXT_EXTS = {".py", ".js", ".ts", ".json", ".md", ".txt", ".env", ".yaml", ".yml", ".sh"}
cred_hits = []
for f in tracked:
    if f.suffix.lower() not in TEXT_EXTS:
        continue
    full = BASE / f
    if not full.exists():
        continue
    try:
        content = full.read_text(encoding="utf-8", errors="ignore").lower()
        for kw in KEYWORDS:
            if kw.lower() in content:
                # Check it's not just a comment or placeholder
                lines = full.read_text(encoding="utf-8", errors="ignore").split("\n")
                for i, line in enumerate(lines, 1):
                    if kw.lower() in line.lower() and not line.strip().startswith("#"):
                        if any(x in line for x in ["=", ":", "Bearer"]):
                            cred_hits.append((f, i, line.strip()[:80]))
    except Exception:
        pass

if cred_hits:
    print(f"  [WARN] Potential credentials found in {len(cred_hits)} locations:")
    for f, line_no, line in cred_hits[:10]:
        print(f"  {f}:{line_no}  {line}")
else:
    print("  [OK] No credentials detected in tracked files")

# 6. GLB files check
print("\n--- GLB / Avatar files ---")
glb_files = [f for f in tracked if f.suffix.lower() in {".glb", ".gltf"}]
if glb_files:
    for f in glb_files:
        full = BASE / f
        mb = full.stat().st_size / 1_000_000 if full.exists() else 0
        print(f"  [WARN] {f} ({mb:.1f}MB) -- verify this is allowed to be public")
else:
    print("  [OK] No GLB files tracked")

# 7. Summary
print("\n" + "=" * 60)
print("AUDIT SUMMARY")
print("=" * 60)
issues = len(large) + len(found_bad) + len([f for bp in BANNED_PATHS
    for f in tracked if str(f).replace('\\','/').startswith(bp)]) + len(cred_hits) + len(glb_files)
if issues == 0:
    print("[CLEAN] No issues found. Safe to keep repo as-is.")
else:
    print(f"[ACTION REQUIRED] {issues} issue(s) found. Review above before making repo public.")
print()
