import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.motion.signavatars_adapter import SignAvatarsAdapter

print('Testing SignAvatarsAdapter...')
adapter = SignAvatarsAdapter('data/signavatars')

print(f'Available datasets: {adapter.get_available_datasets()}')
info = adapter.get_dataset_info()
for name, data in info.items():
    print(f'  {name}: {data["count"]} samples')

# Load a sample
samples = adapter.load_dataset('wlasl', max_samples=5)
print(f'Loaded {len(samples)} samples')

sample = samples[0]
print(f'Sample ID: {sample.sample_id}')
print(f'Num frames: {sample.num_frames}')

features = adapter.get_recognition_features(sample)
print(f'Recognition features shape: {features.shape}')

print('')
print('[ADAPTER TEST PASSED]')
