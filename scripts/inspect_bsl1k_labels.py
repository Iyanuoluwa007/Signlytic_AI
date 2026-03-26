import pickle
import json
from pathlib import Path
import sys

# Redirect output to file
output_file = open('bsl1k_inspection.txt', 'w', encoding='utf-8')

def log(msg):
    print(msg)
    output_file.write(str(msg) + '\n')

log("="*70)
log("BSL-1K PSEUDO-LABELS INSPECTION")
log("="*70)

# 1. I3D Pseudo Labels
log("\n1. I3D PSEUDO LABELS:")
i3d_path = Path("data/BSL-1K/ISOLATED_SIGN-I3D_PSEUDO_LABELS/bobsl/v1.4/automatic_annotations/isolated_signs/i3d_pseudo_labels/i3d_pseudo_labels_spottings.pkl")
with open(i3d_path, 'rb') as f:
    i3d_labels = pickle.load(f)

log(f"   Type: {type(i3d_labels)}")
if isinstance(i3d_labels, dict):
    log(f"   Keys count: {len(i3d_labels)}")
    log(f"   Sample keys: {list(i3d_labels.keys())[:5]}")
    first_key = list(i3d_labels.keys())[0]
    val = i3d_labels[first_key]
    log(f"   Sample [{first_key}] type: {type(val)}")
    if isinstance(val, list) and val:
        log(f"   Sample [{first_key}] length: {len(val)}")
        log(f"   Sample [{first_key}][0]: {val[0]}")
elif isinstance(i3d_labels, list):
    log(f"   Items: {len(i3d_labels)}")
    if i3d_labels:
        log(f"   Sample[0]: {i3d_labels[0]}")

# 2. Dictionary Spottings
log("\n" + "="*70)
log("2. DICTIONARY SPOTTINGS (v2):")
dict_path = Path("data/BSL-1K/ISOLATED_SIGN-DICTIONARY/bobsl/v1.4/automatic_annotations/isolated_signs/dictionary/dictionary_spottings_v2.pkl")
with open(dict_path, 'rb') as f:
    dict_spottings = pickle.load(f)

log(f"   Type: {type(dict_spottings)}")
if isinstance(dict_spottings, dict):
    log(f"   Keys count: {len(dict_spottings)}")
    log(f"   Sample keys: {list(dict_spottings.keys())[:10]}")
    first_key = list(dict_spottings.keys())[0]
    val = dict_spottings[first_key]
    log(f"   Sample [{first_key}] type: {type(val)}")
    if isinstance(val, list) and val:
        log(f"   Length: {len(val)}")
        log(f"   First item type: {type(val[0])}")
        if hasattr(val[0], '__dict__'):
            log(f"   First item attrs: {val[0].__dict__}")
        else:
            log(f"   First item: {val[0]}")
elif isinstance(dict_spottings, list):
    log(f"   Items: {len(dict_spottings)}")

# 3. Mouthing Labels
log("\n" + "="*70)
log("3. MOUTHING LABELS:")
mouth_path = Path("data/BSL-1K/ISOLATED_SIGN-MOUTHING/bobsl/v1.4/automatic_annotations/isolated_signs/mouthing/mouthing_spottings_v2.pkl")
with open(mouth_path, 'rb') as f:
    mouthing = pickle.load(f)

log(f"   Type: {type(mouthing)}")
if isinstance(mouthing, dict):
    log(f"   Keys count: {len(mouthing)}")
    log(f"   Sample keys: {list(mouthing.keys())[:10]}")
    first_key = list(mouthing.keys())[0]
    val = mouthing[first_key]
    log(f"   Sample [{first_key}] type: {type(val)}")
    if isinstance(val, list) and val:
        log(f"   Length: {len(val)}")
elif isinstance(mouthing, list):
    log(f"   Items: {len(mouthing)}")

# 4. Count total labels
log("\n" + "="*70)
log("4. SUMMARY:")

total_i3d = sum(len(v) if isinstance(v, list) else 1 for v in i3d_labels.values()) if isinstance(i3d_labels, dict) else len(i3d_labels)
log(f"   I3D pseudo-labels: {total_i3d} spottings")

total_dict = sum(len(v) if isinstance(v, list) else 1 for v in dict_spottings.values()) if isinstance(dict_spottings, dict) else len(dict_spottings)
log(f"   Dictionary spottings: {total_dict} spottings")

total_mouth = sum(len(v) if isinstance(v, list) else 1 for v in mouthing.values()) if isinstance(mouthing, dict) else len(mouthing)
log(f"   Mouthing spottings: {total_mouth} spottings")

output_file.close()
print("\nSaved to bsl1k_inspection.txt")
