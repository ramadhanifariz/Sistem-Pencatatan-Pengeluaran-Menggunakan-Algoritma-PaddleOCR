import cv2
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR


class ReceiptExtractor:
    
    def __init__(self, lang='id'):
        print(f" Initializing PaddleOCR with language: {lang}")
        # Parameter OCR lebih toleran untuk teks miring dan bounding box longgar
        self.ocr = PaddleOCR(
            use_angle_cls=True,          # deteksi orientasi teks (miring/terbalik)
            lang=lang,
            show_log=False,
            det_db_thresh=0.3,           # lebih rendah agar peka teks samar
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0      # memperbesar bounding box
        )
        print(" PaddleOCR initialized successfully")
    
    def extract_text(self, image_path):
        try:
            result = self.ocr.ocr(str(image_path), cls=True)
            extracted_data = []
            
            if not result or not result[0]:
                return []
            
            for line in result[0]:
                if line and len(line) >= 2:
                    text_info = line[1]
                    if text_info and len(text_info) >= 2:
                        extracted_data.append({
                            'text': str(text_info[0]),
                            'confidence': float(text_info[1]) if text_info[1] else 0.0,
                            'bbox': line[0] if len(line) > 0 else []
                        })
            return extracted_data
        except Exception as e:
            print(f" OCR Error: {str(e)[:100]}")
            return []
    
    def parse_receipt(self, extracted_data):
        receipt_data = {
            'items': [],
            'total': None,
            'tax': None,
            'discount': None,
            'cash': None,
            'change': None,
            'subtotal': None,
            'service_charge': None
        }
        
        if not extracted_data:
            return receipt_data
        
        # Combine all text
        full_text = ' '.join([item['text'] for item in extracted_data])
        
        #  EXTRACT TOTAL 
        total_patterns = [
            r'(?:TOTAL|Total|total)[\s:*]*([\d.,]+)',
            r'(?:GRAND TOTAL|Grand Total)[\s:*]*([\d.,]+)',
            r'\*\*TOTAL\*\*[\s:]*([\d.,]+)',
            r'Amount Due[\s:]*([\d.,]+)',
            r'DUE[\s:]*([\d.,]+)',
            r'=([\d.,]+)$',
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, full_text)
            if matches:
                for match in reversed(matches):
                    price = self._extract_price(match)
                    if price and price > 1000:
                        receipt_data['total'] = price
                        break
                if receipt_data['total']:
                    break
        
        # EXTRACT CASH & CHANGE 
        cash_patterns = [r'(?:CASH|Cash|cash|TUNAI|Tunai|tunai|Bayar|bayar)[\s:]*([\d.,]+)']
        for pattern in cash_patterns:
            match = re.search(pattern, full_text)
            if match:
                receipt_data['cash'] = self._extract_price(match.group(1))
                break
        
        change_patterns = [r'(?:CHANGE|Change|change|KEMBALI|Kembali|kembali)[\s:]*([\d.,]+)']
        for pattern in change_patterns:
            match = re.search(pattern, full_text)
            if match:
                receipt_data['change'] = self._extract_price(match.group(1))
                break
        
        # EXTRACT TAX 
        tax_patterns = [r'(?:TAX|Tax|tax|PPN|ppn|PAJAK|pajak|PB1)[\s:]*([\d.,]+)']
        for pattern in tax_patterns:
            match = re.search(pattern, full_text)
            if match:
                receipt_data['tax'] = self._extract_price(match.group(1))
                break
        
        # EXTRACT ITEMS (ENHANCED FOR MULTIPLE ITEMS) 
        # Get all text lines
        all_text_lines = [item['text'] for item in extracted_data]
        
        # Method 1: Try to parse each line as individual item
        for line in all_text_lines:
            item = self._parse_item_line_enhanced(line)
            if item and item['price'] and item['price'] > 0:
                if item['name'] and len(item['name']) > 1:
                    # Avoid duplicates
                    if not self._is_duplicate_item(receipt_data['items'], item):
                        receipt_data['items'].append(item)
        
        # Method 2: If no items found, try to parse from combined text
        if len(receipt_data['items']) == 0:
            # Split by common patterns
            lines = full_text.split('\n')
            for line in lines:
                # Look for patterns like "1 ITEM_NAME 15.000,00"
                pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+)'
                matches = re.findall(pattern, line)
                for match in matches:
                    quantity = int(match[0])
                    name = match[1].strip()
                    price = self._extract_price(match[2])
                    if price and price > 0 and name:
                        receipt_data['items'].append({
                            'name': name[:50],
                            'quantity': quantity,
                            'price': price,
                            'total': quantity * price
                        })
        
        # Method 3: Special handling for lines with multiple items
        if len(receipt_data['items']) == 0:
            receipt_data['items'] = self._extract_items_from_text(full_text)
        
        # Calculate total from items if total not detected
        if not receipt_data['total'] and receipt_data['items']:
            calculated_total = sum(item['total'] for item in receipt_data['items'])
            if calculated_total > 0:
                receipt_data['total'] = calculated_total
        
        return receipt_data
    
    def _parse_item_line_enhanced(self, line):
        original_line = line
        line = re.sub(r'\s+', ' ', line).strip()
        
        # Skip short lines
        if len(line) < 5:
            return None
        
        # Skip lines that are likely totals or headers
        line_lower = line.lower()
        skip_keywords = ['total', 'subtotal', 'cash', 'change', 'tax', 'discount', 
                        'service', 'item', 'qty', 'pcs', 'due', 'grand', 
                        'kembali', 'tunai', 'bayar', 'diskon', 'pajak']
        if any(keyword in line_lower for keyword in skip_keywords):
            return None
        
        # Find quantity (number at beginning)
        quantity = 1
        name = ""
        price = None
        
        # Pattern 1: "1 ITEM_NAME 15.000,00"
        pattern1 = re.match(r'^(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+)', line)
        if pattern1:
            quantity = int(pattern1.group(1))
            name = pattern1.group(2).strip()
            price = self._extract_price(pattern1.group(3))
            if price:
                return {
                    'name': name,
                    'quantity': quantity,
                    'price': price,
                    'total': quantity * price
                }
        
        # Pattern 2: "ITEM_NAME 15.000,00" (no quantity)
        pattern2 = re.match(r'^([A-Za-z\s]+?)\s+([\d.,]+)', line)
        if pattern2:
            name = pattern2.group(1).strip()
            price = self._extract_price(pattern2.group(2))
            if price and len(name) > 2:
                return {
                    'name': name,
                    'quantity': 1,
                    'price': price,
                    'total': price
                }
        
        # Pattern 3: "1x ITEM_NAME @15.000,00"
        pattern3 = re.match(r'^(\d+)[xX]\s+([A-Za-z\s]+?)\s*@?\s*([\d.,]+)', line)
        if pattern3:
            quantity = int(pattern3.group(1))
            name = pattern3.group(2).strip()
            price = self._extract_price(pattern3.group(3))
            if price:
                return {
                    'name': name,
                    'quantity': quantity,
                    'price': price,
                    'total': quantity * price
                }
        
        # Pattern 4: "ITEM_NAME @15.000,00"
        pattern4 = re.match(r'^([A-Za-z\s]+?)\s*@\s*([\d.,]+)', line)
        if pattern4:
            name = pattern4.group(1).strip()
            price = self._extract_price(pattern4.group(2))
            if price and len(name) > 2:
                return {
                    'name': name,
                    'quantity': 1,
                    'price': price,
                    'total': price
                }
        
        return None
    
    def _extract_items_from_text(self, text):
        items = []
        
        # Find all occurrences of quantity + name + price
        # Pattern: number, then words, then price
        pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+(?:\.\d{2})?)'
        matches = re.findall(pattern, text)
        
        for match in matches:
            quantity = int(match[0])
            name = match[1].strip()
            price_str = match[2]
            price = self._extract_price(price_str)
            
            if price and price > 0 and name and len(name) > 2:
                # Check if this item is not a total or header
                name_lower = name.lower()
                skip_words = ['total', 'subtotal', 'cash', 'change', 'tax', 'discount']
                if name_lower not in skip_words:
                    items.append({
                        'name': name[:50],
                        'quantity': quantity,
                        'price': price,
                        'total': quantity * price
                    })
        
        return items
    
    def _is_duplicate_item(self, items, new_item):
        for item in items:
            if item['name'].lower() == new_item['name'].lower():
                if abs(item['price'] - new_item['price']) < 100:
                    return True
        return False
    
    def _extract_price(self, price_str):
        if not price_str:
            return None
        
        price_str = str(price_str).strip()
        
        # Remove 'Rp' or 'rp'
        price_str = re.sub(r'[Rr]p\.?\s*', '', price_str)
        
        # Remove spaces
        price_str = price_str.replace(' ', '')
        
        # Handle "15.000,00" format (dot thousand, comma decimal)
        if ',' in price_str:
            # Replace comma with dot for decimal
            price_str = price_str.replace(',', '.')
            # Remove dots that are thousand separators
            parts = price_str.split('.')
            if len(parts) > 2:
                # Multiple dots means first ones are thousand separators
                price_str = ''.join(parts[:-1]) + '.' + parts[-1]
        
        # Handle "15.000" format (dot as thousand separator)
        elif '.' in price_str:
            parts = price_str.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                # This is thousand separator (e.g., 15.000)
                price_str = price_str.replace('.', '')
            elif len(parts) == 2 and len(parts[1]) <= 2:
                # This is decimal (e.g., 15.50)
                pass
            else:
                # Multiple dots, likely thousand separators
                price_str = price_str.replace('.', '')
        
        # Remove all non-numeric except dot
        price_str = re.sub(r'[^\d.]', '', price_str)
        
        # Remove trailing dot
        if price_str.endswith('.'):
            price_str = price_str[:-1]
        
        try:
            return float(price_str)
        except (ValueError, TypeError):
            return None
    
    def process_image(self, image_path):
        extracted_data = self.extract_text(image_path)
        # Jika hasil OCR sedikit, coba dengan preprocessing tambahan (misal invert)
        if len(extracted_data) < 3:
            # Coba preprocessing alternatif pada gambar yang sudah disimpan? 
            # Untuk sederhananya, kita tidak melakukan fallback di sini
            pass
        parsed_data = self.parse_receipt(extracted_data)
        return extracted_data, parsed_data


# ============================================================
# FUNGSI PREPROCESSING ROBUST UNTUK STRUK BERMASALAH
# ============================================================

def deskew_image(image):
    """
    Deteksi kemiringan teks menggunakan Hough Lines dan luruskan gambar.
    Returns gambar yang sudah dirotasi (dalam format BGR).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)  # background putih, teks hitam
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 100)
    if lines is None:
        return image
    
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi - 90
        if -45 < angle < 45:
            angles.append(angle)
    
    if not angles:
        return image
    
    median_angle = np.median(angles)
    if abs(median_angle) < 0.5:
        return image
    
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated


def auto_invert_if_needed(image):
    """
    Jika gambar didominasi gelap (teks terang), lakukan inversi.
    Menggunakan perbandingan intensitas mean.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < 127:
        # Mungkin gambar gelap dengan teks terang -> invert
        inverted = cv2.bitwise_not(image)
        # Cek apakah setelah invert lebih jelas (variance lebih tinggi)
        gray_inv = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        var_original = np.var(gray)
        var_inverted = np.var(gray_inv)
        if var_inverted > var_original:
            return inverted
    return image


def sharpen_image(image, strength=1.5):
    """Sharpening menggunakan unsharp masking."""
    blurred = cv2.GaussianBlur(image, (0,0), 3.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def is_blurry(image, threshold=80):
    """Deteksi blur berdasarkan variance of Laplacian."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold


def enhance_contrast_color(image):
    """
    Tingkatkan kontras pada gambar berwarna menggunakan CLAHE di ruang LAB.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enh, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def denoise_image(image):
    """Denoise dengan bilateral filter (edge-preserving) + fastNlMeans."""
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h=10, hColor=10,
                                               templateWindowSize=7, searchWindowSize=21)
    return denoised


def upscale_if_small(image, min_size=1000):
    """
    Perbesar gambar jika resolusi terlalu kecil.
    """
    h, w = image.shape[:2]
    if max(h, w) < min_size:
        scale = min_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        upscaled = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        return upscaled
    return image


def preprocess_image(image_path, output_path=None):
    """
    Pipeline preprocessing adaptif untuk struk belanja dengan berbagai masalah:
    - Kemiringan teks (deskew)
    - Blur (sharpening & denoise)
    - Resolusi rendah (upscale)
    - Kontras rendah (CLAHE warna)
    - Warna teks tidak standar (auto invert)
    """
    # Baca gambar asli
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Tidak bisa membaca gambar: {image_path}")
    
    # 1. Denoise awal
    img = denoise_image(img)
    
    # 2. Upscale jika perlu
    img = upscale_if_small(img, min_size=1000)
    
    # 3. Perbaiki kemiringan (deskew)
    img = deskew_image(img)
    
    # 4. Deteksi blur dan sharpening ekstra jika perlu
    if is_blurry(img, threshold=60):
        img = sharpen_image(img, strength=1.2)
    else:
        img = sharpen_image(img, strength=0.8)
    
    # 5. Tingkatkan kontras warna (CLAHE di LAB)
    img = enhance_contrast_color(img)
    
    # 6. Cek apakah perlu invert (teks terang di background gelap)
    img = auto_invert_if_needed(img)
    
    # 7. Denoise ringan terakhir
    img = denoise_image(img)
    
    # Simpan hasil preprocessing
    if output_path:
        cv2.imwrite(str(output_path), img)
    
    return img


# Jika dijalankan sebagai script, lakukan test sederhana
if __name__ == "__main__":
    print("=" * 50)
    print("Testing Enhanced Receipt Parser")
    print("=" * 50)
    
    extractor = ReceiptExtractor(lang='id')
    
    # Test with multiple items
    test_lines = [
        "1 BLACK PAPPER MEATBALL 15.000,00",
        "1 GREAN TEA 10.000,00",
        "2 ORIGINAL BREWED TEA 20.000,00",
        "TOTAL 45.000,00"
    ]
    
    test_data = [{'text': line, 'confidence': 0.95, 'bbox': []} for line in test_lines]
    parsed = extractor.parse_receipt(test_data)
    
    print(f"\n Parse Test Results:")
    print(f"   Total: Rp {parsed['total']:,.0f}" if parsed['total'] else "   Total: Not detected")
    print(f"   Items found: {len(parsed['items'])}")
    
    if parsed['items']:
        print("\n   Item Details:")
        total_from_items = 0
        for item in parsed['items']:
            print(f"      - {item['quantity']} x {item['name']} = Rp {item['total']:,.0f}")
            total_from_items += item['total']
        print(f"\n   Total from items: Rp {total_from_items:,.0f}")
    
    print("\n utils.py ready to use!")