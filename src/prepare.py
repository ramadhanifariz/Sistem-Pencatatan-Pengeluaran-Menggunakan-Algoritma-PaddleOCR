import json
from pathlib import Path

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
TRAIN_DIR = RAW_DIR / 'train'
TEST_DIR = RAW_DIR / 'test'
SPLITS_DIR = DATA_DIR / 'splits'

def find_images_in_folder(folder):
    
    images = []
    possible_subfolders = ['image', 'images', 'Image', 'Images', 'img']
    
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
        images.extend(folder.glob(ext))
    
    for sub in possible_subfolders:
        subfolder = folder / sub
        if subfolder.exists():
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']:
                images.extend(subfolder.glob(ext))
    
    return images

def main():
    print("="*50)
    print(" DATA COLLECTION")
    print("="*50)
    
    # Cek folder train
    if not TRAIN_DIR.exists():
        print(f" Folder {TRAIN_DIR} tidak ditemukan!")
        return
    
    # Cari gambar
    train_images = find_images_in_folder(TRAIN_DIR)
    test_images = find_images_in_folder(TEST_DIR) if TEST_DIR.exists() else []
    
    # Simpan daftar gambar ke JSON
    data_info = {
        'train': [str(p) for p in train_images],
        'test': [str(p) for p in test_images]
    }
    
    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    output_path = SPLITS_DIR / 'images_list.json'
    with open(output_path, 'w') as f:
        json.dump(data_info, f, indent=2)
    
    print(f"\n STATISTIK:")
    print(f"   Training images: {len(train_images)}")
    print(f"   Testing images : {len(test_images)}")
    print(f"\n Daftar gambar disimpan di: {output_path}")

if __name__ == "__main__":
    main()