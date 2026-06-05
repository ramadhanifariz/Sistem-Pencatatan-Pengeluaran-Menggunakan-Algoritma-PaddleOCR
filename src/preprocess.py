"""
preprocess.py - Preprocessing & OCR Extraction dengan Memory Management
"""

import json
import sys
import gc
import os
import cv2  # <--- TAMBAHKAN INI!
from pathlib import Path
from datetime import datetime

# Install psutil jika belum ada
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    print("⚠️ psutil not installed. Memory monitoring disabled.")
    print("   Install with: pip install psutil")

sys.path.append('src')
from utils import ReceiptExtractor, group_into_lines, preprocess_image

# ========== KONFIGURASI ==========
DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'

# Buat direktori
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
SPLITS_DIR.mkdir(parents=True, exist_ok=True)

# Konfigurasi
MAX_IMAGE_SIZE = 1500


def get_memory_usage():
    """Cek penggunaan memory saat ini (dalam MB)"""
    if HAS_PSUTIL:
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    return 0


def free_memory():
    """Force garbage collection untuk membersihkan memory"""
    gc.collect()
    gc.collect()


def validate_image(img_path):
    """
    Validasi gambar menggunakan OpenCV
    """
    try:
        # Cek file exist
        if not img_path.exists():
            return False, "File not found"
        
        # Cek ukuran file
        file_size = img_path.stat().st_size
        if file_size < 500:
            return False, f"File too small ({file_size} bytes)"
        
        # Baca dengan OpenCV
        img = cv2.imread(str(img_path))
        if img is None:
            return False, "Cannot read image (corrupted)"
        
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False, f"Image too small ({w}x{h})"
        
        return True, "OK"
        
    except Exception as e:
        return False, str(e)


def process_single_image(img_path, extractor):
    
    # Konversi ke Path jika perlu
    if isinstance(img_path, str):
        img_path = Path(img_path)
    
    # ========== VALIDASI GAMBAR ==========
    is_valid, msg = validate_image(img_path)
    if not is_valid:
        print(f"       {msg}")
        return [], {}, None
    
    try:
        # ========== PREPROCESS ==========
        processed_filename = f"processed_{img_path.stem}.jpg"
        processed_path = PROCESSED_DIR / processed_filename
        
        # Preprocess image (dari utils.py)
        preprocess_image(str(img_path), str(processed_path))
        
        # ========== OCR EXTRACTION ==========
        extracted_data, parsed_data = extractor.process_image(str(processed_path))
        
        return extracted_data, parsed_data, processed_path
        
    except Exception as e:
        error_msg = str(e)[:100]
        
        # Skip untuk error yang dikenal
        skip_errors = [
            "could not create a primitive",
            "bad allocation",
            "corrupt",
            "unable to allocate"
        ]
        
        for skip_error in skip_errors:
            if skip_error.lower() in error_msg.lower():
                print(f"       Skip - {skip_error}")
                return [], {}, None
        
        print(f"       Error: {error_msg}")
        return [], {}, None


def process_image_list(image_paths, folder_name, extractor, max_images=None):
    """
    Memproses daftar gambar
    """
    if max_images:
        image_paths = image_paths[:max_images]
    
    processed_data = []
    failed_images = []
    total = len(image_paths)
    
    print(f"\n   Total gambar: {total}")
    
    for idx, img_path in enumerate(image_paths, 1):
        # Konversi ke Path jika perlu
        if isinstance(img_path, str):
            img_path = Path(img_path)
        
        print(f"   [{idx}/{total}] {img_path.name}")
        
        # Proses gambar
        extracted_data, parsed_data, processed_path = process_single_image(
            img_path, extractor
        )
        
        if extracted_data and len(extracted_data) > 0:
            print(f"       {len(extracted_data)} text blocks extracted")
            
            # ========== TAMPILKAN SEMUA TEKS (TIDAK DIBATASI) ==========
            print(f"       ALL TEXT DETECTED ({len(extracted_data)} blocks):")
            for i, item in enumerate(extracted_data):
                text_preview = item['text'][:60] + '...' if len(item['text']) > 60 else item['text']
                print(f"         {i+1}. {text_preview} (conf: {item['confidence']:.2%})")
            
            # Simpan anotasi (tetap sama)
            annotation_data = {
                'filename': img_path.name,
                'folder': folder_name,
                'processed_path': str(processed_path) if processed_path else None,
                'original_path': str(img_path),
                'extracted_data': extracted_data,
                'parsed_data': parsed_data,
                'timestamp': datetime.now().isoformat()
            }
            
            annotation_file = ANNOTATIONS_DIR / f"{folder_name}_{img_path.stem}.json"
            with open(annotation_file, 'w', encoding='utf-8') as f:
                json.dump(annotation_data, f, indent=2, ensure_ascii=False)
            
            processed_data.append(annotation_data)
            
            total_val = parsed_data.get('total')
            if total_val:
                print(f"       Total: Rp {total_val:,.0f}")
            else:
                print(f"       Total not detected")
        
        # Bersihkan memory setiap 10 gambar
        if idx % 10 == 0:
            free_memory()
            if HAS_PSUTIL:
                print(f"       Memory: {get_memory_usage():.1f} MB")
    
    # Report failed images
    if failed_images:
        print(f"\n    {len(failed_images)} gambar gagal diproses:")
        for fname in failed_images[:10]:
            print(f"      - {fname}")
        if len(failed_images) > 10:
            print(f"      ... dan {len(failed_images) - 10} lainnya")
    
    return processed_data


def run_preprocess(max_train=50, max_test=20):
    """
    Menjalankan pipeline preprocessing
    """
    print("=" * 60)
    print(" PREPROCESSING & OCR EXTRACTION")
    print("=" * 60)
    
    if HAS_PSUTIL:
        print(f"\n Memory awal: {get_memory_usage():.1f} MB")
    
    # Load daftar gambar
    images_list_path = DATA_DIR / 'splits' / 'images_list.json'
    
    if not images_list_path.exists():
        print("\n images_list.json tidak ditemukan!")
        print("   Jalankan 'python src/prepare.py' terlebih dahulu")
        return
    
    with open(images_list_path, 'r', encoding='utf-8') as f:
        data_info = json.load(f)
    
    # Konversi string path ke Path object
    train_paths = [Path(p) for p in data_info.get('train', [])]
    test_paths = [Path(p) for p in data_info.get('test', [])]
    
    print(f"\n Data yang akan diproses:")
    print(f"   Training: {min(len(train_paths), max_train)}/{len(train_paths)} gambar")
    print(f"   Testing : {min(len(test_paths), max_test)}/{len(test_paths)} gambar")
    
    # Inisialisasi OCR extractor
    print("\n Initializing PaddleOCR...")
    print("    Ini mungkin memakan waktu 10-30 detik...")
    
    try:
        extractor = ReceiptExtractor(lang='id')
        print("   ✅ PaddleOCR siap!")
    except Exception as e:
        print(f"   ❌ Gagal inisialisasi PaddleOCR: {e}")
        return
    
    # Proses training images
    print("\n" + "=" * 40)
    print(" Processing TRAINING images...")
    print("=" * 40)
    train_data = process_image_list(train_paths, 'train', extractor, max_train)
    
    # Bersihkan memory
    free_memory()
    if HAS_PSUTIL:
        print(f"\n Memory setelah training: {get_memory_usage():.1f} MB")
    
    # Proses test images
    print("\n" + "=" * 40)
    print(" Processing TESTING images...")
    print("=" * 40)
    test_data = process_image_list(test_paths, 'test', extractor, max_test)
    
    # Simpan split info
    split_info = {
        'train': [str(ANNOTATIONS_DIR / f'train_{p.stem}.json') for p in train_paths[:max_train]],
        'test': [str(ANNOTATIONS_DIR / f'test_{p.stem}.json') for p in test_paths[:max_test]],
        'stats': {
            'train_count': len(train_data),
            'test_count': len(test_data),
            'total': len(train_data) + len(test_data)
        },
        'timestamp': datetime.now().isoformat()
    }
    
    with open(SPLITS_DIR / 'data_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
    
    print("\n" + "=" * 60)
    print(" PREPROCESSING COMPLETE!")
    print("=" * 60)
    print(f"   Training: {len(train_data)} images processed")
    print(f"   Testing : {len(test_data)} images processed")
    print(f"\n Hasil disimpan di:")
    print(f"   - Annotations: {ANNOTATIONS_DIR}")
    print(f"   - Split info: {SPLITS_DIR / 'data_splits.json'}")
    
    if HAS_PSUTIL:
        print(f"\n Memory akhir: {get_memory_usage():.1f} MB")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Preprocessing OCR untuk struk belanja')
    parser.add_argument('--train', type=int, default=50, help='Maksimal gambar training (default: 50)')
    parser.add_argument('--test', type=int, default=20, help='Maksimal gambar testing (default: 20)')
    parser.add_argument('--all', action='store_true', help='Proses semua gambar')
    
    args = parser.parse_args()
    
    if args.all:
        max_train = 999999
        max_test = 999999
    else:
        max_train = args.train
        max_test = args.test
    
    print(f"\n📌 Konfigurasi: max_train={max_train}, max_test={max_test}")
    
    run_preprocess(max_train, max_test)