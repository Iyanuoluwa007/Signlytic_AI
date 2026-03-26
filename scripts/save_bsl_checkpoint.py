import torch
import numpy as np
from pathlib import Path
import json

print("Saving BSL Dict Recognizer checkpoint...")

features_dir = Path("data/bsl_dict_features")

# Load all features
with open(features_dir / "index.json", 'r') as f:
    glosses = json.load(f)

features_list = []
valid_glosses = []

for gloss in glosses:
    feat_path = features_dir / f"{gloss}.npy"
    if feat_path.exists():
        feat = np.load(feat_path).squeeze()
        features_list.append(feat)
        valid_glosses.append(gloss)

features = np.stack(features_list)

# Save as single checkpoint
save_dir = Path("models/bsl_dict_recognition")
save_dir.mkdir(parents=True, exist_ok=True)

torch.save({
    'glosses': valid_glosses,
    'features': torch.from_numpy(features),
    'num_classes': len(valid_glosses),
}, save_dir / 'retrieval_model.pt')

print(f"Saved: {len(valid_glosses)} BSL signs")
print(f"Location: {save_dir / 'retrieval_model.pt'}")
