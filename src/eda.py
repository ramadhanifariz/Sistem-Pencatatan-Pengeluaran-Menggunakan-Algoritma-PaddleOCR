# eda.py
import json
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path('data')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
PROCESSED_DIR = DATA_DIR / 'processed'

def exploratory_data_analysis():
    all_annotations = list(ANNOTATIONS_DIR.glob('*.json'))
    if not all_annotations:
        print("No annotations found")
        return
    data_by_folder = {}
    for ann_file in all_annotations:
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            folder = data.get('folder', 'unknown')
            if folder not in data_by_folder:
                data_by_folder[folder] = []
            data_by_folder[folder].append(data)
    
    print("\n EDA RESULTS:")
    for folder, data_list in data_by_folder.items():
        total_amounts = [d.get('parsed_data', {}).get('total', 0) for d in data_list if d.get('parsed_data', {}).get('total')]
        total_items = [len(d.get('parsed_data', {}).get('items', [])) for d in data_list]
        confidences = [item['confidence'] for d in data_list for item in d.get('extracted_data', [])]
        print(f"\n {folder.upper()} SET ({len(data_list)} images):")
        if total_amounts:
            print(f"   - Avg total: Rp {np.mean(total_amounts):,.0f}")
        if confidences:
            print(f"   - Avg OCR confidence: {np.mean(confidences):.2%}")
    eda_report = {
        'timestamp': datetime.now().isoformat(),
        'total_images': len(all_annotations),
        'per_folder': {k: len(v) for k, v in data_by_folder.items()},
    }
    with open(PROCESSED_DIR / 'eda_report.json', 'w') as f:
        json.dump(eda_report, f, indent=2)
    print(f"\nEDA report saved to {PROCESSED_DIR / 'eda_report.json'}")

if __name__ == "__main__":
    exploratory_data_analysis()