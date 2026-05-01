import cv2
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR


class ReceiptExtractor:
    
    def __init__(self, lang='id'):
        print(f" Initializing PaddleOCR with language: {lang}")
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            det_db_thresh=0.3,
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0
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
        
        full_text = ' '.join([item['text'] for item in extracted_data])
        
        # Extract total, cash, change, tax 
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
        
        tax_patterns = [r'(?:TAX|Tax|tax|PPN|ppn|PAJAK|pajak|PB1)[\s:]*([\d.,]+)']
        for pattern in tax_patterns:
            match = re.search(pattern, full_text)
            if match:
                receipt_data['tax'] = self._extract_price(match.group(1))
                break
        
        all_text_lines = [item['text'] for item in extracted_data]
        receipt_data['items'] = self._extract_items_smart(all_text_lines)
        
        if not receipt_data['items']:
            for line in all_text_lines:
                item = self._parse_item_line_enhanced(line)
                if item and item['price'] and item['price'] > 0:
                    if item['name'] and len(item['name']) > 1:
                        if not self._is_duplicate_item(receipt_data['items'], item):
                            receipt_data['items'].append(item)
        
        if not receipt_data['items']:
            lines = full_text.split('\n')
            for line in lines:
                matches = re.findall(r'(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+)', line)
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
        
        if not receipt_data['items']:
            receipt_data['items'] = self._extract_items_from_text(full_text)
        
        if not receipt_data['total'] and receipt_data['items']:
            calculated_total = sum(item['total'] for item in receipt_data['items'])
            if calculated_total > 0:
                receipt_data['total'] = calculated_total
        
        return receipt_data
    
    def _extract_items_smart(self, all_text_lines):
        """
        Ekstrak items dengan menangani:
        - nama di baris i, dan angka (qty, price, total) di baris i+1
        - format "nama 2 17000 34000" dalam satu baris
        - mendeteksi quantity dari angka kecil, harga dari angka besar
        """
        items = []
        i = 0
        n = len(all_text_lines)

        def is_likely_price(value):
            # harga >= 1000
            return value >= 1000

        def is_likely_quantity(value):
            return value < 1000 and float(value).is_integer()

        while i < n:
            line = all_text_lines[i].strip()
            if not line:
                i += 1
                continue

            lower = line.lower()
            skip_keywords = ['subtotal', 'total', 'tax', 'ppn', 'pb1', 'service', 'charge', 
                            'cash', 'change', 'due', 'kembali', 'tunai', 'diskon', 'discount',
                            'qty', 'nama', 'harga', 'item', 'quantity', 'price']
            if any(kw in lower for kw in skip_keywords):
                i += 1
                continue

            numbers = re.findall(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+)', line)
            numeric_vals = [self._extract_price(n) for n in numbers if self._extract_price(n) is not None]

            if numeric_vals:
                qty_candidates = [v for v in numeric_vals if is_likely_quantity(v)]
                price_candidates = [v for v in numeric_vals if is_likely_price(v)]
                
                if len(price_candidates) >= 2:
                    price_value = price_candidates[0]
                    total_value = price_candidates[1]
                    quantity_value = qty_candidates[0] if qty_candidates else 1
                    if quantity_value == 1 and total_value and price_value:
                        quantity_value = int(round(total_value / price_value))
                    temp_name = line
                    for v in numeric_vals:
                        pattern = r'\b' + re.escape(str(int(v)) if v.is_integer() else f"{v:.2f}".replace('.', ',')) + r'\b'
                        temp_name = re.sub(pattern, '', temp_name)
                    full_name = temp_name.strip()
                    if full_name and price_value:
                        items.append({
                            'name': full_name[:60],
                            'quantity': quantity_value,
                            'price': price_value,
                            'total': quantity_value * price_value
                        })
                    i += 1
                    continue
                else:
                    price_value = price_candidates[0] if price_candidates else None
                    quantity_value = qty_candidates[0] if qty_candidates else 1
                    temp_name = line
                    for v in numeric_vals:
                        pattern = r'\b' + re.escape(str(int(v)) if v.is_integer() else f"{v:.2f}".replace('.', ',')) + r'\b'
                        temp_name = re.sub(pattern, '', temp_name)
                    full_name = temp_name.strip()
                    if price_value and full_name:
                        items.append({
                            'name': full_name[:60],
                            'quantity': quantity_value,
                            'price': price_value,
                            'total': quantity_value * price_value
                        })
                    i += 1
                    continue

            if i + 1 < n:
                next_line = all_text_lines[i+1].strip()
                next_numbers = re.findall(r'(\d{1,3}(?:[.,]\d{3})*(?:[.,]\d{2})?|\d+)', next_line)
                next_vals = [self._extract_price(n) for n in next_numbers if self._extract_price(n) is not None]
                if next_vals:
                    qty_candidates = [v for v in next_vals if is_likely_quantity(v)]
                    price_candidates = [v for v in next_vals if is_likely_price(v)]
                    price_value = price_candidates[0] if price_candidates else None
                    quantity_value = qty_candidates[0] if qty_candidates else 1
                    full_name = line.strip()
                    if price_value and full_name:
                        items.append({
                            'name': full_name[:60],
                            'quantity': quantity_value,
                            'price': price_value,
                            'total': quantity_value * price_value
                        })
                    i += 2  
                    continue
            i += 1

        return items
    
    def _parse_item_line_enhanced(self, line):
        line = re.sub(r'\s+', ' ', line).strip()
        if len(line) < 5:
            return None
        line_lower = line.lower()
        skip_keywords = ['total', 'subtotal', 'cash', 'change', 'tax', 'discount', 
                        'service', 'item', 'qty', 'pcs', 'due', 'grand', 
                        'kembali', 'tunai', 'bayar', 'diskon', 'pajak']
        if any(kw in line_lower for kw in skip_keywords):
            return None
        quantity = 1
        name = ""
        price = None
        pattern1 = re.match(r'^(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+)', line)
        if pattern1:
            quantity = int(pattern1.group(1))
            name = pattern1.group(2).strip()
            price = self._extract_price(pattern1.group(3))
            if price:
                return {'name': name, 'quantity': quantity, 'price': price, 'total': quantity*price}
        pattern2 = re.match(r'^([A-Za-z\s]+?)\s+([\d.,]+)', line)
        if pattern2:
            name = pattern2.group(1).strip()
            price = self._extract_price(pattern2.group(2))
            if price and len(name) > 2:
                return {'name': name, 'quantity': 1, 'price': price, 'total': price}
        pattern3 = re.match(r'^(\d+)[xX]\s+([A-Za-z\s]+?)\s*@?\s*([\d.,]+)', line)
        if pattern3:
            quantity = int(pattern3.group(1))
            name = pattern3.group(2).strip()
            price = self._extract_price(pattern3.group(3))
            if price:
                return {'name': name, 'quantity': quantity, 'price': price, 'total': quantity*price}
        pattern4 = re.match(r'^([A-Za-z\s]+?)\s*@\s*([\d.,]+)', line)
        if pattern4:
            name = pattern4.group(1).strip()
            price = self._extract_price(pattern4.group(2))
            if price and len(name) > 2:
                return {'name': name, 'quantity': 1, 'price': price, 'total': price}
        return None
    
    def _extract_items_from_text(self, text):
        items = []
        pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+([\d.,]+(?:\.\d{2})?)'
        matches = re.findall(pattern, text)
        for match in matches:
            quantity = int(match[0])
            name = match[1].strip()
            price = self._extract_price(match[2])
            if price and price > 0 and name and len(name) > 2:
                name_lower = name.lower()
                skip_words = ['total', 'subtotal', 'cash', 'change', 'tax', 'discount']
                if name_lower not in skip_words:
                    items.append({'name': name[:50], 'quantity': quantity, 'price': price, 'total': quantity*price})
        return items
    
    def _is_duplicate_item(self, items, new_item):
        for item in items:
            if item['name'].lower() == new_item['name'].lower() and abs(item['price'] - new_item['price']) < 100:
                return True
        return False
    
    def _extract_price(self, price_str):
        if not price_str:
            return None
        price_str = str(price_str).strip()
        price_str = re.sub(r'[Rr]p\.?\s*', '', price_str)
        price_str = price_str.replace(' ', '')
        if ',' in price_str:
            price_str = price_str.replace(',', '.')
            parts = price_str.split('.')
            if len(parts) > 2:
                price_str = ''.join(parts[:-1]) + '.' + parts[-1]
        elif '.' in price_str:
            parts = price_str.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                price_str = price_str.replace('.', '')
            elif len(parts) == 2 and len(parts[1]) <= 2:
                pass
            else:
                price_str = price_str.replace('.', '')
        price_str = re.sub(r'[^\d.]', '', price_str)
        if price_str.endswith('.'):
            price_str = price_str[:-1]
        try:
            return float(price_str)
        except:
            return None
    
    def process_image(self, image_path):
        extracted_data = self.extract_text(image_path)
        parsed_data = self.parse_receipt(extracted_data)
        return extracted_data, parsed_data


# PREPROCESSING ROBUST

def deskew_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
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
    rotated = cv2.warpAffine(image, rot_mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def auto_invert_if_needed(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_intensity = np.mean(gray)
    if mean_intensity < 127:
        inverted = cv2.bitwise_not(image)
        gray_inv = cv2.cvtColor(inverted, cv2.COLOR_BGR2GRAY)
        if np.var(gray_inv) > np.var(gray):
            return inverted
    return image

def sharpen_image(image, strength=1.5):
    blurred = cv2.GaussianBlur(image, (0,0), 3.0)
    sharpened = cv2.addWeighted(image, 1.0 + strength, blurred, -strength, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)

def is_blurry(image, threshold=80):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

def enhance_contrast_color(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l_enh = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enh, a, b))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

def denoise_image(image):
    denoised = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

def upscale_if_small(image, min_size=1000):
    h, w = image.shape[:2]
    if max(h, w) < min_size:
        scale = min_size / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    return image

def preprocess_image(image_path, output_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Tidak bisa membaca gambar: {image_path}")
    img = denoise_image(img)
    img = upscale_if_small(img, min_size=1000)
    img = deskew_image(img)
    img = sharpen_image(img, strength=1.2 if is_blurry(img, 60) else 0.8)
    img = enhance_contrast_color(img)
    img = auto_invert_if_needed(img)
    img = denoise_image(img)
    if output_path:
        cv2.imwrite(str(output_path), img)
    return img


# FUNGSI PENDUKUNG UNTUK GROUPING BARIS 
def group_into_lines(extracted_data, y_tolerance=20, x_tolerance=30):
    
    if not extracted_data:
        return []
    
    # Ekstrak koordinat dan teks
    items = []
    for data in extracted_data:
        bbox = data.get('bbox')
        if bbox and len(bbox) >= 2:
            y_center = (bbox[0][1] + bbox[2][1]) / 2
            x_left = bbox[0][0]
            x_right = bbox[2][0]
        else:
            y_center = 0
            x_left = 0
            x_right = 0
        items.append({
            'text': data['text'],
            'y': y_center,
            'x_left': x_left,
            'x_right': x_right
        })
    
    # Urutkan berdasarkan Y 
    items.sort(key=lambda x: x['y'])
    
    # Fungsi untuk mendeteksi apakah teks kemungkinan harga 
    def is_price(text):
        # Bersihkan dari tanda baca
        cleaned = re.sub(r'[Rp.,\s]', '', text, flags=re.IGNORECASE)
        # Jika mengandung minimal 3 digit angka dan tidak banyak huruf, dianggap harga
        if re.search(r'\d{3,}', cleaned):
            huruf = re.sub(r'[\d]', '', cleaned)
            if len(huruf) <= 3:
                return True
        return False
    
    lines_raw = []
    current_line = []
    last_y = None
    for it in items:
        if last_y is None or abs(it['y'] - last_y) <= y_tolerance:
            current_line.append(it)
        else:
            current_line.sort(key=lambda x: x['x_left'])
            lines_raw.append(current_line)
            current_line = [it]
        last_y = it['y']
    if current_line:
        current_line.sort(key=lambda x: x['x_left'])
        lines_raw.append(current_line)
    
    merged_lines = []
    i = 0
    while i < len(lines_raw):
        current = lines_raw[i]
        has_price_curr = any(is_price(it['text']) for it in current)
        
        if i + 1 < len(lines_raw):
            next_line = lines_raw[i+1]
            has_price_next = any(is_price(it['text']) for it in next_line)
            
            if not has_price_curr and not has_price_next:
                
                curr_x_left = min(it['x_left'] for it in current)
                curr_x_right = max(it['x_right'] for it in current)
                next_x_left = min(it['x_left'] for it in next_line)
                next_x_right = max(it['x_right'] for it in next_line)
                
                if (curr_x_left <= next_x_right + x_tolerance and 
                    next_x_left <= curr_x_right + x_tolerance):
                    
                    combined_text = ' '.join([it['text'] for it in current] + [it['text'] for it in next_line])
                    
                    new_line = [{
                        'text': combined_text,
                        'y': (current[0]['y'] + next_line[0]['y']) / 2,
                        'x_left': min(curr_x_left, next_x_left),
                        'x_right': max(curr_x_right, next_x_right)
                    }]
                    merged_lines.append(new_line)
                    i += 2
                    continue
        
        merged_lines.append(current)
        i += 1
    
    result = []
    for line in merged_lines:
        line.sort(key=lambda x: x['x_left'])
        line_text = ' '.join([it['text'] for it in line])
        result.append(line_text)
    
    return result


if __name__ == "__main__":
    print("=" * 50)
    print("Testing Enhanced Receipt Parser")
    print("=" * 50)
    extractor = ReceiptExtractor(lang='id')
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
        for item in parsed['items']:
            print(f"      - {item['quantity']} x {item['name']} = Rp {item['total']:,.0f}")
    print("\n utils.py ready to use!")