"""
demo_pipeline.py
Menampilkan pipeline lengkap: preprocessing robust -> OCR -> parsing.
Gunakan untuk menguji satu gambar struk.
"""

import cv2
import matplotlib.pyplot as plt
import sys
from pathlib import Path

sys.path.append('src')
from utils import ReceiptExtractor, preprocess_image

def show_images(original, processed, title_orig="Original", title_proc="Preprocessed"):
    """Tampilkan dua gambar berdampingan."""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.title(title_orig)
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Jika processed grayscale, konversi ke RGB untuk ditampilkan
    if len(processed.shape) == 2:
        processed = cv2.cvtColor(processed, cv2.COLOR_GRAY2RGB)
    else:
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    plt.imshow(processed)
    plt.title(title_proc)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def run_pipeline(image_path):
    """
    Jalankan pipeline lengkap pada satu gambar.
    """
    print("="*60)
    print(f"Processing: {image_path}")
    print("="*60)
    
    # 1. Load gambar asli
    original = cv2.imread(str(image_path))
    if original is None:
        print(f"ERROR: Tidak bisa membaca {image_path}")
        return
    
    # 2. Preprocessing robust (menyimpan hasil ke file sementara)
    temp_proc_path = "temp_preprocessed.jpg"
    preprocess_image(str(image_path), temp_proc_path)
    processed = cv2.imread(temp_proc_path)
    
    # Tampilkan perbandingan visual
    show_images(original, processed, "Original Receipt", "Robust Preprocessing")
    
    # 3. Inisialisasi OCR
    print("\n[OCR] Initializing PaddleOCR...")
    extractor = ReceiptExtractor(lang='id')
    
    # 4. Ekstraksi teks dan parsing
    print("[OCR] Extracting text...")
    extracted_data, parsed_data = extractor.process_image(temp_proc_path)
    
    # 5. Tampilkan hasil OCR
    print("\n" + "-"*40)
    print("EXTRACTED TEXT (OCR RESULTS):")
    print("-"*40)
    if extracted_data:
        for i, item in enumerate(extracted_data, 1):
            print(f"{i:2d}. Text: {item['text']} (conf: {item['confidence']:.2f})")
    else:
        print("No text detected!")
    
    # 6. Tampilkan hasil parsing
    print("\n" + "-"*40)
    print("PARSED RECEIPT DATA:")
    print("-"*40)
    
    if parsed_data['items']:
        print(f"Items ({len(parsed_data['items'])}):")
        for item in parsed_data['items']:
            print(f"   - {item['quantity']} x {item['name']} = Rp {item['total']:,.0f}")
    else:
        print("Items: None detected")
    
    print(f"\nTotal      : Rp {parsed_data['total']:,.0f}" if parsed_data['total'] else "\nTotal      : Not detected")
    print(f"Tax        : Rp {parsed_data['tax']:,.0f}" if parsed_data['tax'] else "Tax        : Not detected")
    print(f"Discount   : Rp {parsed_data['discount']:,.0f}" if parsed_data['discount'] else "Discount   : Not detected")
    print(f"Cash       : Rp {parsed_data['cash']:,.0f}" if parsed_data['cash'] else "Cash       : Not detected")
    print(f"Change     : Rp {parsed_data['change']:,.0f}" if parsed_data['change'] else "Change     : Not detected")
    
    # Bersihkan file sementara
    Path(temp_proc_path).unlink(missing_ok=True)
    
    print("\n" + "="*60)
    print("Pipeline demo selesai.")
    print("="*60)

if __name__ == "__main__":
    # Ganti dengan path gambar struk Anda
    train_folder = Path("data/raw/train/image")
    test_folder = Path("data/raw/test/image")
    
    if train_folder.exists():
        images = list(train_folder.glob("*.jpg")) + list(train_folder.glob("*.jpeg")) + list(train_folder.glob("*.png"))
        if images:
            receipt_image = images[0]
            print(f"Menemukan gambar train: {receipt_image}")
    if receipt_image is None and test_folder.exists():
        images = list(test_folder.glob("*.jpg")) + list(test_folder.glob("*.jpeg")) + list(test_folder.glob("*.png"))
        if images:
            receipt_image = images[0]
            print(f"Menemukan gambar test: {receipt_image}")
    
    if receipt_image is None:
        print("Tidak ada gambar ditemukan di folder data/raw/train/image/ atau data/raw/test/image/")
        print("Pastikan Anda sudah meletakkan file gambar di folder tersebut.")
        sys.exit(1)
    
    run_pipeline(receipt_image)