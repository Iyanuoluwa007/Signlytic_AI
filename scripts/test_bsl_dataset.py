import sys
sys.path.insert(0, '.')
from scripts.train_bsl_recognizer import BSLDataset

print('Testing BSL Dataset...')
ds = BSLDataset(split='train', max_classes=100, min_samples_per_class=100)
print(f'Samples: {len(ds)}, Classes: {ds.num_classes}')

if len(ds) > 0:
    sample = ds[0]
    print(f'Sample features shape: {sample["features"].shape}')
    print(f'Sample word: {sample["word"]}')
    print('[OK]')
else:
    print('[FAILED - No samples]')
