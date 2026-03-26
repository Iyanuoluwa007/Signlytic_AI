from pathlib import Path
import torch
import json

print("="*70)
print("CHECKING BSL TRAINING STATUS")
print("="*70)

# Check for saved model
model_dir = Path("models/bsl_recognition")
print(f"\n1. Model directory: {model_dir}")

if model_dir.exists():
    files = list(model_dir.iterdir())
    print(f"   Files found: {len(files)}")
    for f in files:
        size_mb = f.stat().st_size / (1024*1024)
        print(f"   - {f.name}: {size_mb:.2f} MB")
        
        if f.name == "best_model.pt":
            try:
                checkpoint = torch.load(f, map_location='cpu', weights_only=False)
                print(f"\n   CHECKPOINT CONTENTS:")
                print(f"   - Keys: {list(checkpoint.keys())}")
                if 'val_acc' in checkpoint:
                    print(f"   - Val Accuracy: {checkpoint['val_acc']*100:.2f}%")
                if 'val_acc_top5' in checkpoint:
                    print(f"   - Val Top-5: {checkpoint['val_acc_top5']*100:.2f}%")
                if 'num_classes' in checkpoint:
                    print(f"   - Classes: {checkpoint['num_classes']}")
                if 'epoch' in checkpoint:
                    print(f"   - Saved at epoch: {checkpoint['epoch']+1}")
                print(f"\n   [MODEL FOUND - Training made progress!]")
            except Exception as e:
                print(f"   Error loading: {e}")
else:
    print("   Directory does not exist - training may not have started")

# Check all model directories
print("\n2. All model directories:")
models_root = Path("models")
if models_root.exists():
    for d in models_root.iterdir():
        if d.is_dir():
            files = list(d.glob("*.pt"))
            print(f"   {d.name}/: {len(files)} .pt files")

print("\n" + "="*70)
