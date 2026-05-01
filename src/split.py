import json
import random
from pathlib import Path
from sklearn.model_selection import train_test_split
from datetime import datetime 

DATA_DIR = Path('data')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'
SPLITS_DIR.mkdir(exist_ok=True)

def create_split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, random_seed=42):
    all_annotations = list(ANNOTATIONS_DIR.glob('*.json'))
    if not all_annotations:
        print("No annotations found. Run preprocess.py first.")
        return
    
    random.seed(random_seed)
    random.shuffle(all_annotations)
    # split
    n_total = len(all_annotations)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    train_files = all_annotations[:n_train]
    val_files = all_annotations[n_train:n_train+n_val]
    test_files = all_annotations[n_train+n_val:]
    split_info = {
        'train': [str(f) for f in train_files],
        'validation': [str(f) for f in val_files],
        'test': [str(f) for f in test_files],
        'stats': {'train': len(train_files), 'val': len(val_files), 'test': len(test_files)},
        'timestamp': datetime.now().isoformat()
    }
    with open(SPLITS_DIR / 'data_splits.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    print(f"Split created: train={len(train_files)}, val={len(val_files)}, test={len(test_files)}")

if __name__ == "__main__":
    create_split()