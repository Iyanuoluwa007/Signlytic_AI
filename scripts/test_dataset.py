import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_signavatars import PoseRecognitionDataset

print('Testing PoseRecognitionDataset...')
dataset = PoseRecognitionDataset(
    data_root='data/signavatars',
    dataset_name='wlasl',
    split='train',
    max_samples=100
)

print(f'Dataset size: {len(dataset)}')
print(f'Num classes: {dataset.num_classes}')

sample = dataset[0]
print(f'Sample poses shape: {sample["poses"].shape}')
print(f'Sample mask shape: {sample["mask"].shape}')
print(f'Sample label: {sample["label"]}')

print('')
print('[DATASET TEST PASSED]')
