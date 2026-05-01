import json
import sys
from pathlib import Path
from datetime import datetime
sys.path.append('src')
from utils import ReceiptExtractor, group_into_lines, preprocess_image  

DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)

def process_image_list(image_paths, folder_name, extractor, max_images=None):
    if max_images:
        image_paths = image_paths[:max_images]
    processed_data = []
    for i, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        print(f"   [{i+1}/{len(image_paths)}] {img_path.name}")
        processed_filename = f"processed_{folder_name}_{img_path.stem}.jpg"
        processed_path = PROCESSED_DIR / processed_filename
        try:
            preprocess_image(str(img_path), str(processed_path))
            extracted_data, parsed_data = extractor.process_image(str(processed_path))
            if extracted_data:
                print(f"      {len(extracted_data)} text blocks extracted")
                lines = group_into_lines(extracted_data, y_tolerance=10)
                print("      Extracted texts (grouped by line):")
                for idx, line in enumerate(lines[:30], 1):
                    display_line = line[:120] + '...' if len(line) > 120 else line
                    print(f"        {idx:2d}. {display_line}")
            else:
                print(f"      No text detected")
                
            annotation_data = {
                'filename': img_path.name,
                'folder': folder_name,
                'processed_path': str(processed_path),
                'original_path': str(img_path),
                'extracted_data': extracted_data,
                'parsed_data': parsed_data,
                'timestamp': datetime.now().isoformat()
            }
            annotation_file = ANNOTATIONS_DIR / f"{folder_name}_{img_path.stem}.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            processed_data.append(annotation_data)
            total = parsed_data.get('total')
            if total:
                print(f"      Total: Rp {total:,.0f}")
        except Exception as e:
            print(f"      Error: {str(e)[:80]}")
    return processed_data

def run_preprocess(max_train=400, max_test=100):
    # load images list
    images_list_path = DATA_DIR / 'splits' / 'images_list.json'
    if not images_list_path.exists():
        print("Please run data_collection.py first")
        return
    with open(images_list_path) as f:
        data_info = json.load(f)
    train_paths = data_info['train']
    test_paths = data_info['test']
    
    extractor = ReceiptExtractor(lang='id')
    
    print("Processing training images...")
    train_data = process_image_list(train_paths, 'train', extractor, max_train)
    print("Processing test images...")
    test_data = process_image_list(test_paths, 'test', extractor, max_test)
    
    split_info = {
        'train': [str(ANNOTATIONS_DIR / f'train_{Path(p).stem}.json') for p in train_paths[:max_train]],
        'test': [str(ANNOTATIONS_DIR / f'test_{Path(p).stem}.json') for p in test_paths[:max_test]],
        'stats': {'train_count': len(train_data), 'test_count': len(test_data)},
        'timestamp': datetime.now().isoformat()
    }
    SPLITS_DIR = DATA_DIR / 'splits'
    SPLITS_DIR.mkdir(exist_ok=True)
    with open(SPLITS_DIR / 'data_splits.json', 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"Preprocessing complete. Train: {len(train_data)}, Test: {len(test_data)}")

if __name__ == "__main__":
    run_preprocess()