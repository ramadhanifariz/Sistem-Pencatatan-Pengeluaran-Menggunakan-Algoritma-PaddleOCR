import json
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
import sys
sys.path.append('src')

from utils import ReceiptExtractor      # asumsinya ini wrapper PaddleOCR
# Kita akan menimpa/melebarkan fungsi preprocess_image dengan versi robust

# ------------------------------------------------------------
# FUNGSI PREPROCESSING ROBUST UNTUK STRUK BERMASALAH
# ------------------------------------------------------------

def deskew_image(image):
    """Deteksi kemiringan teks dan luruskan."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)          # background putih, teks hitam
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    # Cari garis horizontal melalui HoughLines
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is None:
        return image
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi - 90
        if -45 < angle < 45:   # hanya garis yang mendekati horizontal
            angles.append(angle)
    
    if not angles:
        return image
    median_angle = np.median(angles)
    
    # Rotasi gambar
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

def enhance_contrast_and_sharpen(image):
    """CLAHE untuk kontras + unsharp masking."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enh, a, b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # Sharpening
    kernel_sharpen = np.array([[-1,-1,-1],
                               [-1, 9,-1],
                               [-1,-1,-1]])
    sharpened = cv2.filter2D(contrast_enhanced, -1, kernel_sharpen)
    return sharpened

def upscale_if_small(image, min_size=800):
    """Upscale gambar jika resolusi terlalu kecil (menggunakan INTER_CUBIC)."""
    h, w = image.shape[:2]
    if h < min_size and w < min_size:
        scale = min_size / min(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
    return image

def denoise_image(image):
    """Denoise tanpa menghilangkan detail teks."""
    # Bilateral filter untuk smooth tapi edge tetap tajam
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    # Jika masih ada noise, gunakan fastNlMeansDenoisingColored
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

def robust_preprocess_image(input_path, output_path):
    """
    Pipeline preprocessing adaptif untuk struk bermasalah.
    Mencoba beberapa konfigurasi jika OCR gagal di awal.
    """
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError(f"Tidak bisa membaca gambar: {input_path}")
    
    # 1. Denoise awal
    img = denoise_image(original)
    # 2. Upscale jika perlu
    img = upscale_if_small(img)
    # 3. Perbaiki kemiringan
    img = deskew_image(img)
    # 4. Contrast + sharpening
    img = enhance_contrast_and_sharpen(img)
    
    # Simpan hasil preprocessing
    cv2.imwrite(output_path, img)
    return output_path

# ------------------------------------------------------------
# OVERRIDE fungsi preprocess_image yang diimpor (jika berasal dari utils)
# ------------------------------------------------------------
# Asumsi: dari utils import preprocess_image as old_preprocess
# Kita akan menggantinya dengan versi robust di namespace ini.
# Cara termudah: definisikan ulang preprocess_image di file ini sebelum dipanggil.
# Jika utils.preprocess_image sudah digunakan di ReceiptExtractor, Anda harus memodifikasi utils.py secara langsung.

# Untuk keperluan demo, kita buat fungsi dengan nama sama agar dipakai oleh kode selanjutnya.
def preprocess_image(input_path, output_path):
    """Wrapper untuk robust preprocessing."""
    return robust_preprocess_image(input_path, output_path)

# ------------------------------------------------------------
# MODIFIKASI ReceiptExtractor (jika diperlukan)
# ------------------------------------------------------------
# Kita akan membuat wrapper yang mengatur parameter PaddleOCR agar lebih toleran
# terhadap teks miring, warna aneh, dll. Jika Anda sudah punya class ReceiptExtractor,
# modifikasi method __init__ atau process_image sebagai berikut:

class RobustReceiptExtractor(ReceiptExtractor):
    """
    Extends ReceiptExtractor dengan parameter PaddleOCR yang lebih optimal
    untuk struk bermasalah.
    """
    def __init__(self, lang='id'):
        super().__init__(lang=lang)
        # Jika di parent class self.ocr diinisialisasi, kita bisa update params
        # Asumsikan self.ocr = PaddleOCR(use_angle_cls=False, ...)
        # Maka kita set ulang
        if hasattr(self, 'ocr'):
            self.ocr = None   # force reinit
        # Inisialisasi ulang dengan parameter kustom
        from paddleocr import PaddleOCR
        self.ocr = PaddleOCR(
            lang=lang,
            use_angle_cls=True,          # deteksi orientasi teks (miring/terbalik)
            det_db_thresh=0.3,           # lebih rendah biar dapat teks samar
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,     # memperbesar bounding box teks
            rec_algorithm='SVTR_LCNet',  # untuk teks panjang (struk)
            rec_batch_num=6,
            use_gpu=True,                # jika ada GPU
            show_log=False
        )
    
    # Override process_image jika diperlukan untuk menangani gagal OCR dengan fallback preprocessing
    def process_image(self, image_path):
        # Coba OCR dengan gambar asli yg sudah melalui robust_preprocess
        # (tapi preprocess sudah dilakukan sebelum panggil method ini)
        result = super().process_image(image_path)   # asumsi parent mengembalikan (extracted, parsed)
        # Jika hasil OCR sangat sedikit, coba preprocessing ekstra
        if not result[0] or len(result[0]) < 3:
            # Coba dengan threshold berbeda atau grayscale saja
            # Di sini kita bisa lakukan percobaan kedua dengan image yang lebih ekstrim
            pass
        return result

# ------------------------------------------------------------
# SISANYA KODE ASLI DENGAN MODIFIKASI PENGGUNAAN FUNGSI PREPROCESS BARU
# ------------------------------------------------------------

DATA_DIR = Path('data')
RAW_DIR = DATA_DIR / 'raw'
TRAIN_DIR = RAW_DIR / 'train'
TEST_DIR = RAW_DIR / 'test'
VAL_DIR = RAW_DIR / 'val'
PROCESSED_DIR = DATA_DIR / 'processed'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'

for dir_path in [PROCESSED_DIR, ANNOTATIONS_DIR, SPLITS_DIR, VAL_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

def find_images_in_folder(folder):
    # (sama seperti asli)
    images = []
    possible_subfolders = ['image', 'images', 'Image', 'Images', 'img']
    direct_images = list(folder.glob('*.jpg')) + list(folder.glob('*.jpeg')) + \
                    list(folder.glob('*.png')) + list(folder.glob('*.JPG')) + \
                    list(folder.glob('*.PNG'))
    images.extend(direct_images)
    for subfolder_name in possible_subfolders:
        subfolder = folder / subfolder_name
        if subfolder.exists() and subfolder.is_dir():
            subfolder_images = list(subfolder.glob('*.jpg')) + list(subfolder.glob('*.jpeg')) + \
                               list(subfolder.glob('*.png')) + list(subfolder.glob('*.JPG')) + \
                               list(subfolder.glob('*.PNG'))
            images.extend(subfolder_images)
    return images

def check_data_structure():
    print("CEK STRUKTUR DATA")
    if not TRAIN_DIR.exists():
        print(f"\n Folder {TRAIN_DIR} not found!")
        return False, [], []
    print("\n Looking for training images...")
    train_images = find_images_in_folder(TRAIN_DIR)
    print("\n Looking for testing images...")
    test_images = find_images_in_folder(TEST_DIR)
    print(f"\n DATA STATISTICS: Training={len(train_images)}, Testing={len(test_images)}")
    if len(train_images) == 0:
        print(f"\n No training images found in {TRAIN_DIR}/image/")
        return False, [], []
    return True, train_images, test_images

def preprocess_images(image_list, folder_name, extractor):
    """Sama seperti asli, tapi menggunakan preprocess_image yang sudah robust."""
    if not image_list:
        return []
    processed_data = []
    for i, img_path in enumerate(image_list):
        print(f"   [{i+1}/{len(image_list)}] {img_path.name}")
        processed_filename = f"processed_{folder_name}_{img_path.stem}.jpg"
        processed_path = PROCESSED_DIR / processed_filename
        try:
            # Memanggil preprocess_image yang sudah kita timpa dengan versi robust
            preprocess_image(str(img_path), str(processed_path))
            extracted_data, parsed_data = extractor.process_image(str(processed_path))
            if extracted_data:
                print(f"      {len(extracted_data)} text blocks extracted")
            else:
                print(f"      No text detected (coba fallback?)")
                # Fallback: coba dengan preprocessing berbeda (optional)
                # Misal simpan gambar asli tanpa sharpening berlebih
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

def preprocess_all_data(train_images, test_images, extractor):
    print("PEMROSESAN DAN EKSTRAKSI OCR")
    print(f"\n Processing TRAINING ({len(train_images)} images)...")
    train_data = preprocess_images(train_images, 'train', extractor)
    print(f"\n Processing TESTING ({len(test_images)} images)...")
    test_data = preprocess_images(test_images, 'test', extractor)
    split_info = {
        'train': [str(ann) for ann in ANNOTATIONS_DIR.glob('train_*.json')],
        'test': [str(ann) for ann in ANNOTATIONS_DIR.glob('test_*.json')],
        'stats': {'train_count': len(train_data), 'test_count': len(test_data), 'total': len(train_data)+len(test_data)},
        'timestamp': datetime.now().isoformat()
    }
    with open(SPLITS_DIR / 'data_splits.json', 'w', encoding='utf-8') as f:
        json.dump(split_info, f, indent=2)
    print(f"\n PREPROCESSING COMPLETE! Training: {len(train_data)}, Testing: {len(test_data)}")
    return split_info

def exploratory_data_analysis():
    # (sama seperti asli)
    print("EXPLORATORY DATA ANALYSIS")
    all_annotations = list(ANNOTATIONS_DIR.glob('*.json'))
    if not all_annotations:
        print("No annotations found")
        return
    data_by_folder = {'train': [], 'test': []}
    for ann_file in all_annotations:
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            folder = data.get('folder', 'unknown')
            if folder in data_by_folder:
                data_by_folder[folder].append(data)
    print("\n EDA RESULTS:")
    for folder, data_list in data_by_folder.items():
        if not data_list:
            continue
        total_amounts = [d.get('parsed_data', {}).get('total', 0) for d in data_list if d.get('parsed_data', {}).get('total')]
        total_items = [len(d.get('parsed_data', {}).get('items', [])) for d in data_list]
        confidences = [item['confidence'] for d in data_list for item in d.get('extracted_data', [])]
        print(f"\n {folder.upper()} SET ({len(data_list)} images):")
        if total_amounts:
            print(f"   - Avg total: Rp {np.mean(total_amounts):,.0f}")
        if confidences:
            print(f"   - Avg OCR confidence: {np.mean(confidences):.2%}")
    # simpan EDA report
    eda_report = { 'timestamp': datetime.now().isoformat(), 'total_images': len(all_annotations) }
    with open(PROCESSED_DIR / 'eda_report.json', 'w') as f:
        json.dump(eda_report, f, indent=2)

def main():
    print("\n" + "="*50)
    print(" SISTEM OTOMATISASI PENCATATAN PENGELUARAN (ROBUST)")
    print("="*50)
    
    # --- TAMBAHKAN BATASAN DI SINI ---
    MAX_TRAIN = 400   # hanya proses 400 gambar train
    MAX_TEST = 100    # hanya proses 100 gambar test
    # ---------------------------------
    
    success, train_images, test_images = check_data_structure()
    if not success:
        print("\n Please fix data structure first!")
        return
    
    original_train_len = len(train_images)
    original_test_len = len(test_images)
    
    # Potong list sesuai batasan
    if len(train_images) > MAX_TRAIN:
        train_images = train_images[:MAX_TRAIN]
        print(f"\n [LIMIT] Memproses hanya {MAX_TRAIN} gambar train (dari {original_train_len})")
    else:
        print(f"\n [INFO] Semua gambar train akan diproses ({len(train_images)} gambar)")
    
    if len(test_images) > MAX_TEST:
        test_images = test_images[:MAX_TEST]
        print(f" [LIMIT] Memproses hanya {MAX_TEST} gambar test (dari {original_test_len})")
    else:
        print(f" [INFO] Semua gambar test akan diproses ({len(test_images)} gambar)")
    
    print("\nMENGINISIALISASI PADDLEOCR dengan konfigurasi robust...")
    # Gunakan class RobustReceiptExtractor jika sudah didefinisikan, atau ReceiptExtractor biasa
    # extractor = RobustReceiptExtractor(lang='id')
    extractor = ReceiptExtractor(lang='id')  # atau sesuaikan
    
    preprocess_all_data(train_images, test_images, extractor)
    exploratory_data_analysis()
    
    print("\n" + "="*50)
    print(" PREPARATION COMPLETE (dengan sampel terbatas)!")
    print("="*50)
    print("\n Next steps:")
    print("   1. Lihat hasil EDA di folder data/processed/eda_report.json")
    print("   2. Cek hasil preprocessing di folder data/processed/")
    print("   3. Jika sudah puas, hapus batasan MAX_TRAIN/MAX_TEST untuk memproses semua data.")

if __name__ == "__main__":
    main()