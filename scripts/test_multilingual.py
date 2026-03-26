import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').absolute()))

from scripts.train_multilingual import MultiLingualPoseDataset

print('Testing MultiLingualPoseDataset...')
ds = MultiLingualPoseDataset(split='train', min_samples_per_class=2)
print(f'Train samples: {len(ds)}')
print(f'Classes: {ds.num_classes}')

sample = ds[0]
print(f'Sample shape: {sample["poses"].shape}')
print(f'Sample gloss: {sample["gloss"]}')
print('[OK]')
