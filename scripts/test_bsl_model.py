import torch
from pathlib import Path

print("="*70)
print("BSL MODEL - QUICK TEST")
print("="*70)

# Load model
checkpoint = torch.load("models/bsl_recognition/best_model.pt", map_location='cpu', weights_only=False)

print(f"Classes: {checkpoint['num_classes']}")
print(f"Val Top-1: {checkpoint['val_acc']*100:.2f}%")
print(f"Val Top-5: {checkpoint['val_acc_top5']*100:.2f}%")

# Show some BSL words it can recognize
words = list(checkpoint['label_map'].keys())[:20]
print(f"\nSample BSL words recognized:")
for i, w in enumerate(words):
    print(f"  {i+1}. {w}")

print("\n[BSL MODEL READY FOR DEPLOYMENT]")
