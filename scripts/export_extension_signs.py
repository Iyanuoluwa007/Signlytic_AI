# scripts/export_extension_signs.py
# Exports BSL dict pose landmarks as extension-ready JSON files
# Uses existing poses_bsldict/ first, runs MediaPipe on remainder
# Output: extension-data/signs/{GLOSS}.json  (top-200 also -> extension/data/signs/core/)

import json, os, sys, cv2, numpy as np

# Tee all print output to log file
import builtins
_LOG_PATH = "D:\\Signlytic_AI\\code\\bsl_translation_project\\scripts\\export_signs_log.txt"
_log_file = open(_LOG_PATH, "w", encoding="utf-8", buffering=1)
_orig_print = builtins.print
def _tee_print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    kwargs.pop("file", None)
    _orig_print(*args, file=_log_file, **kwargs)
builtins.print = _tee_print
print("[LOG] Output saving to: " + _LOG_PATH)

from pathlib import Path

BASE      = Path(r"D:\Signlytic_AI\code\bsl_translation_project")
POSES_DIR = BASE / "data" / "poses_bsldict" / "train"
VIDEO_DIR = BASE / "data" / "videos" / "bsl_signs"
EXT_CDN   = BASE / "extension-data" / "signs"
EXT_CORE  = BASE / "signlytic-extension" / "data" / "signs" / "core"

EXT_CDN.mkdir(parents=True, exist_ok=True)
EXT_CORE.mkdir(parents=True, exist_ok=True)

TOP_200 = {
    "HELLO","THANK","SORRY","PLEASE","YES","NO","HELP","WANT","NEED","LIKE",
    "LOVE","GOOD","BAD","BIG","SMALL","MORE","LESS","HAPPY","SAD","ANGRY",
    "TIRED","SICK","WELL","COME","GO","EAT","DRINK","SLEEP","WORK","PLAY",
    "LEARN","KNOW","UNDERSTAND","THINK","FEEL","SEE","HEAR","SPEAK","SIGN",
    "READ","WRITE","BUY","SELL","GIVE","TAKE","MAKE","USE","FIND","LOSE",
    "WIN","STOP","START","FINISH","WAIT","MEET","VISIT","LIVE","MOVE","STAY",
    "OPEN","CLOSE","TURN","LOOK","WATCH","LISTEN","WALK","RUN","SIT","STAND",
    "MAN","WOMAN","BOY","GIRL","CHILD","FAMILY","FRIEND","PEOPLE","PERSON","NAME",
    "HOME","HOUSE","SCHOOL","HOSPITAL","SHOP","MONEY","TIME","DAY","WEEK","MONTH",
    "YEAR","TODAY","TOMORROW","YESTERDAY","MORNING","AFTERNOON","NIGHT","NOW","LATER",
    "EARLY","LATE","ALWAYS","NEVER","SOMETIMES","OFTEN","AGAIN","FIRST","LAST","NEXT",
    "ONE","TWO","THREE","FOUR","FIVE","SIX","SEVEN","EIGHT","NINE","TEN",
    "HUNDRED","THOUSAND","MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY",
    "JANUARY","FEBRUARY","MARCH","APRIL","JUNE","JULY","AUGUST","SEPTEMBER","OCTOBER",
    "NOVEMBER","DECEMBER","RED","BLUE","GREEN","YELLOW","WHITE","BLACK","COLOUR","NUMBER",
    "QUESTION","ANSWER","PROBLEM","IDEA","REASON","IMPORTANT","DIFFERENT","SAME","NEW","OLD",
    "HOT","COLD","FAST","SLOW","EASY","HARD","TRUE","FALSE","POSSIBLE","CANNOT",
    "WILL","SHOULD","MUST","CAN","HOW","WHAT","WHERE","WHEN","WHO","WHY",
    "WHICH","BECAUSE","IF","BUT","AND","OR","NOT","ALSO","WITH","WITHOUT",
    "BEFORE","AFTER","DURING","BETWEEN","NEAR","FAR","HERE","THERE","EVERYWHERE","NOTHING",
}


def convert_pose_file(pose_path):
    with open(pose_path) as f:
        data = json.load(f)

    frames = []
    for p in data.get("poses", []):
        def to_list(lms):
            if not lms:
                return None
            if isinstance(lms, list):
                result = []
                for lm in lms:
                    if isinstance(lm, dict):
                        result.append([lm.get("x", 0), lm.get("y", 0), lm.get("z", 0)])
                    elif isinstance(lm, list):
                        result.append(lm[:3])
                return result
            return None

        body = to_list(p.get("pose"))
        lh   = to_list(p.get("left_hand"))
        rh   = to_list(p.get("right_hand"))

        if body or lh or rh:
            frames.append({"body": body, "lh": lh, "rh": rh})

    return frames


# Model files for MediaPipe 0.10.x Tasks API (downloaded once on first run)
MODELS_DIR = BASE / "data" / "mediapipe_models"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_URL   = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
HAND_URL   = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"


def ensure_models():
    import urllib.request
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for path, url in [(POSE_MODEL, POSE_URL), (HAND_MODEL, HAND_URL)]:
        if not path.exists():
            print(f"  [DL] Downloading {path.name}...")
            urllib.request.urlretrieve(url, path)
            print(f"  [DL] {path.name} saved ({path.stat().st_size // 1024}KB)")


def run_mediapipe(video_path):
    try:
        import mediapipe as mp
        from mediapipe.tasks.python import vision as mp_vision
        from mediapipe.tasks.python.core import base_options as mp_base
    except ImportError:
        print("  [WARN] mediapipe not installed -- pip install mediapipe")
        return []

    ensure_models()

    VisionRunningMode = mp_vision.RunningMode

    pose_opts = mp_vision.PoseLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=str(POSE_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    hand_opts = mp_vision.HandLandmarkerOptions(
        base_options=mp_base.BaseOptions(model_asset_path=str(HAND_MODEL)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    frames = []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    with mp_vision.PoseLandmarker.create_from_options(pose_opts) as pose_det,          mp_vision.HandLandmarker.create_from_options(hand_opts) as hand_det:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int(frame_idx * 1000 / fps)

            pose_result = pose_det.detect_for_video(mp_image, ts_ms)
            hand_result = hand_det.detect_for_video(mp_image, ts_ms)

            # Body landmarks
            body = None
            if pose_result.pose_landmarks:
                body = [[lm.x, lm.y, lm.z] for lm in pose_result.pose_landmarks[0]]

            # Hand landmarks -- map to left/right by handedness
            lh = rh = None
            if hand_result.hand_landmarks:
                for i, hand_lms in enumerate(hand_result.hand_landmarks):
                    lms_list = [[lm.x, lm.y, lm.z] for lm in hand_lms]
                    if i < len(hand_result.handedness):
                        label = hand_result.handedness[i][0].category_name.lower()
                        if label == "left":
                            lh = lms_list
                        else:
                            rh = lms_list

            if body or lh or rh:
                frames.append({"body": body, "lh": lh, "rh": rh})

            frame_idx += 1

    cap.release()
    return frames


# Build gloss -> pose file map from existing poses
existing = {}
for jf in sorted(POSES_DIR.glob("*.json")):
    gloss = jf.stem.split("_")[0].upper()
    existing[gloss] = jf

# Build full gloss list from video dir
video_map = {}
for vid in sorted(VIDEO_DIR.glob("*.mp4")):
    gloss = vid.stem.upper()
    video_map[gloss] = vid

print(f"[INFO] Existing pose files : {len(existing)}")
print(f"[INFO] Videos available    : {len(video_map)}")
print(f"[INFO] CDN output          : {EXT_CDN}")
print(f"[INFO] Core output         : {EXT_CORE}")
print()

done = skipped = mediapipe_ran = errors = 0
all_glosses = sorted(set(list(existing.keys()) + list(video_map.keys())))

for gloss in all_glosses:
    out_path = EXT_CDN / f"{gloss}.json"

    if out_path.exists():
        skipped += 1
        continue

    frames = []

    if gloss in existing:
        try:
            frames = convert_pose_file(existing[gloss])
        except Exception as e:
            print(f"  [ERR] {gloss} pose convert: {e}")
            errors += 1
            continue
    elif gloss in video_map:
        print(f"  [MP]  {gloss} -- running MediaPipe...")
        try:
            frames = run_mediapipe(video_map[gloss])
            mediapipe_ran += 1
        except Exception as e:
            print(f"  [ERR] {gloss} mediapipe: {e}")
            errors += 1
            continue

    if not frames:
        print(f"  [SKIP] {gloss} -- no frames extracted")
        skipped += 1
        continue

    with open(out_path, "w") as f:
        json.dump(frames, f, separators=(",", ":"))

    if gloss in TOP_200:
        core_path = EXT_CORE / f"{gloss}.json"
        with open(core_path, "w") as f:
            json.dump(frames, f, separators=(",", ":"))

    done += 1
    if done % 50 == 0:
        print(f"  [OK]  {done} signs exported so far...")

print()
print(f"[DONE] Exported  : {done}")
print(f"       Skipped   : {skipped}")
print(f"       MediaPipe : {mediapipe_ran}")
print(f"       Errors    : {errors}")
print()

core_files = list(EXT_CORE.glob("*.json"))
print(f"[CORE] {len(core_files)}/200 bundled signs in extension/data/signs/core/")
missing_core = TOP_200 - {f.stem for f in core_files}
if missing_core:
    print(f"[CORE] Missing {len(missing_core)}: {sorted(missing_core)[:10]}...")
else:
    print(f"[CORE] All top-200 core signs present.")
