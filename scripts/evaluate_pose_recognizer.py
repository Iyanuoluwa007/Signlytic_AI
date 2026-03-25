import sys
from pathlib import Path
import torch
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_pose_recognizer import PoseRecognitionDataset
from src.models.enhanced_models import PoseSignRecognizer
from torch.utils.data import DataLoader

def evaluate_model(model_path: str = "models/pose_recognition/best_model.pt"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    print(f"Val Top-1: {checkpoint['val_acc']*100:.2f}%")
    print(f"Val Top-5: {checkpoint['val_acc_top5']*100:.2f}%")
    print(f"Num classes: {checkpoint['num_classes']}")
    
    # Load model
    model = PoseSignRecognizer(num_classes=checkpoint['num_classes']).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Load test dataset
    test_dataset = PoseRecognitionDataset(
        data_root="data/signavatars",
        dataset_name="wlasl",
        split="test",
        augment=False,
        min_samples_per_class=2
    )
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"\nTest set: {len(test_dataset)} samples")
    
    # Evaluate
    correct_top1 = 0
    correct_top5 = 0
    total = 0
    
    idx_to_label = checkpoint['idx_to_label']
    
    with torch.no_grad():
        for batch in test_loader:
            poses = batch['poses'].to(device)
            masks = batch['mask'].to(device)
            labels = batch['label'].to(device)
            
            logits = model(poses, masks)
            
            # Top-1
            preds = logits.argmax(dim=-1)
            correct_top1 += (preds == labels).sum().item()
            
            # Top-5
            k = min(5, logits.shape[-1])
            _, top5 = logits.topk(k, dim=-1)
            correct_top5 += (top5 == labels.unsqueeze(-1)).any(dim=-1).sum().item()
            
            total += labels.shape[0]
    
    test_top1 = correct_top1 / total * 100
    test_top5 = correct_top5 / total * 100
    
    print(f"\n{'='*50}")
    print(f"TEST RESULTS")
    print(f"{'='*50}")
    print(f"Test Top-1 Accuracy: {test_top1:.2f}%")
    print(f"Test Top-5 Accuracy: {test_top5:.2f}%")
    print(f"Random baseline: {100/checkpoint['num_classes']:.2f}%")
    print(f"Improvement over random: {test_top1 / (100/checkpoint['num_classes']):.1f}x")
    
    # Save results
    results = {
        'model_path': model_path,
        'num_classes': checkpoint['num_classes'],
        'val_top1': checkpoint['val_acc'] * 100,
        'val_top5': checkpoint['val_acc_top5'] * 100,
        'test_top1': test_top1,
        'test_top5': test_top5,
        'test_samples': len(test_dataset),
    }
    
    with open('models/pose_recognition/evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to models/pose_recognition/evaluation_results.json")
    
    return results

if __name__ == "__main__":
    evaluate_model()
