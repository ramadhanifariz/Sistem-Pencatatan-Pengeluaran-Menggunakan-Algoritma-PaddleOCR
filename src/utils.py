import cv2
import numpy as np
import re
from pathlib import Path
from paddleocr import PaddleOCR


class ReceiptExtractor:
   
    def __init__(self, lang='id', use_gpu=False):
        print(f" Initializing PaddleOCR with language: {lang}")
        
        self.ocr = PaddleOCR(
            use_angle_cls=True,
            lang=lang,
            show_log=False,
            det_limit_side_len=2500,
            det_db_thresh=0.3,              
            det_db_box_thresh=0.5,          
            det_db_unclip_ratio=1.2,         
            rec_batch_num=6,
            drop_score=0.1,                 
            max_text_length=50,
            use_space_char=True,
        )
        print(" PaddleOCR initialized")
    
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
            'items': [], 'total': None, 'tax': None, 'discount': None,
            'cash': None, 'change': None, 'subtotal': None, 'service_charge': None
        }
        
        if not extracted_data:
            return receipt_data
        
        # Method 1: Group lines by position
        lines = self._group_lines_advanced(extracted_data)
        
        # Method 2: Parse items with improved patterns
        receipt_data['items'] = self._parse_items_improved(lines)
        
        # Method 3: If still no items, try raw text approach
        if len(receipt_data['items']) == 0:
            raw_text = ' '.join([item['text'] for item in extracted_data])
            receipt_data['items'] = self._parse_items_from_raw(raw_text)
        
        # Parse totals from full text
        full_text = ' '.join(lines)
        receipt_data['total'] = self._extract_total_improved(full_text)
        receipt_data['subtotal'] = self._extract_subtotal_improved(full_text)
        receipt_data['tax'] = self._extract_tax_improved(full_text)
        receipt_data['cash'] = self._extract_cash_improved(full_text)
        receipt_data['change'] = self._extract_change_improved(full_text)
        receipt_data['discount'] = self._extract_discount_improved(full_text)
        
        return receipt_data
    
    def _group_lines_advanced(self, extracted_data, y_tolerance=20, x_tolerance=200):
        if not extracted_data:
            return []
        
        # Ekstrak koordinat
        items = []
        for data in extracted_data:
            bbox = data.get('bbox')
            if bbox and len(bbox) >= 2:
                y_center = (bbox[0][1] + bbox[2][1]) / 2
                x_left = bbox[0][0]
                x_right = bbox[2][0]
                items.append({
                    'text': data['text'],
                    'y': y_center,
                    'x_left': x_left,
                    'x_right': x_right,
                })
            else:
                items.append({
                    'text': data['text'],
                    'y': 0,
                    'x_left': 0,
                    'x_right': 0,
                })
        
        if not items:
            return []
        
        items.sort(key=lambda x: x['y'])
        
        # ========== KELOMPOKKAN BERDASARKAN Y ==========
        lines_raw = []
        current_line = []
        last_y = None
        
        for item in items:
            if last_y is None or abs(item['y'] - last_y) <= y_tolerance:
                current_line.append(item)
            else:
                # Urutkan berdasarkan X sebelum menyimpan
                current_line.sort(key=lambda x: x['x_left'])
                lines_raw.append(current_line)
                current_line = [item]
            last_y = item['y']
        
        if current_line:
            current_line.sort(key=lambda x: x['x_left'])
            lines_raw.append(current_line)
        
        # ========== GABUNGKAN TEKS DALAM SATU BARIS ==========
        result = []
        for line in lines_raw:
            # Gabungkan teks dalam satu baris
            combined = []
            previous_x_right = None
            
            for item in line:
                text = item['text'].strip()
                if not text:
                    continue
                
                if previous_x_right is not None:
                    gap = item['x_left'] - previous_x_right
                    if gap > x_tolerance:
                        combined.append(' ')  
                
                combined.append(text)
                previous_x_right = item['x_right']
            
            # Gabungkan menjadi satu string
            line_text = ' '.join(combined)
            # Bersihkan spasi berlebih
            line_text = re.sub(r'\s+', ' ', line_text).strip()
            result.append(line_text)
        
        return result
    def _merge_nearby_texts(self, extracted_data, distance_threshold=50):
        
        if not extracted_data:
            return extracted_data
        
        # Urutkan berdasarkan posisi X
        items = []
        for data in extracted_data:
            bbox = data.get('bbox')
            if bbox and len(bbox) >= 2:
                x_left = bbox[0][0]
                items.append({
                    'text': data['text'],
                    'x': x_left,
                    'confidence': data['confidence'],
                    'data': data
                })
        
        items.sort(key=lambda x: x['x'])
        
        # Gabungkan yang berdekatan
        merged = []
        i = 0
        while i < len(items):
            current = items[i]
            combined_text = current['text']
            combined_conf = current['confidence']
            j = i + 1
            
            while j < len(items):
                gap = items[j]['x'] - items[j-1]['x']
                if gap <= distance_threshold:
                    combined_text += " " + items[j]['text']
                    combined_conf = (combined_conf + items[j]['confidence']) / 2
                    j += 1
                else:
                    break
            
            merged.append({
                'text': combined_text,
                'confidence': combined_conf,
                'bbox': []
            })
            i = j
        
        return merged

    def _parse_items_improved(self, lines):
        
        items = []
        
        # Kata-kata yang harus di-skip
        skip_words = ['total', 'subtotal', 'service', 'pbi', 'tax', 'cash', 'change', 
                      'discount', 'grand', 'amount', 'due', 'tendered', 'kembali',
                      'tunai', 'bayar', 'diskon', 'pajak', 'ppn', 'edc', 'bca']
        
        # Pattern untuk berbagai format item
        patterns = [
            # Pattern 1: "1 x Nama Barang 75,000" atau "1x Nama Barang 75,000"
            (re.compile(r'(\d+)\s*[xX]\s*([A-Za-z0-9\s]+?)\s*([\d.,]+)', re.IGNORECASE), 'with_x'),
            
            # Pattern 2: "1 Nama Barang 75,000" (tanpa x)
            (re.compile(r'(\d+)\s+([A-Za-z0-9\s]+?)\s+([\d.,]+)$', re.IGNORECASE), 'with_qty'),
            
            # Pattern 3: "Nama Barang 75,000" (tanpa quantity)
            (re.compile(r'^([A-Za-z0-9\s]+?)\s+([\d.,]+)$', re.IGNORECASE), 'no_qty'),
            
            # Pattern 4: "75,000 Nama Barang" (harga di depan)
            (re.compile(r'^([\d.,]+)\s+([A-Za-z0-9\s]+)$', re.IGNORECASE), 'price_first'),
            
            # Pattern 5: "Nama Barang @75,000"
            (re.compile(r'^([A-Za-z0-9\s]+?)\s*@\s*([\d.,]+)$', re.IGNORECASE), 'with_at'),
        ]
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Skip line yang mengandung kata skip (tapi tetap cek angka)
            line_lower = line.lower()
            if any(word in line_lower for word in skip_words):
                if not re.search(r'\d{3,}', line):
                    continue
            
            for pattern, ptype in patterns:
                matches = pattern.findall(line)
                for match in matches:
                    try:
                        if ptype == 'with_x' and len(match) == 3:
                            qty = int(match[0])
                            name = self._clean_name_improved(match[1])
                            price = self._extract_price_improved(match[2])
                            if price and price > 0 and name and len(name) > 2:
                                items.append({
                                    'name': name[:50],
                                    'quantity': qty,
                                    'price': price,
                                    'total': qty * price
                                })
                        
                        elif ptype == 'with_qty' and len(match) == 3:
                            qty = int(match[0])
                            name = self._clean_name_improved(match[1])
                            price = self._extract_price_improved(match[2])
                            if price and price > 0 and name and len(name) > 2:
                                items.append({
                                    'name': name[:50],
                                    'quantity': qty,
                                    'price': price,
                                    'total': qty * price
                                })
                        
                        elif ptype == 'no_qty' and len(match) == 2:
                            name = self._clean_name_improved(match[0])
                            price = self._extract_price_improved(match[1])
                            if price and price > 0 and name and len(name) > 3:
                                if name.lower() not in skip_words:
                                    items.append({
                                        'name': name[:50],
                                        'quantity': 1,
                                        'price': price,
                                        'total': price
                                    })
                        
                        elif ptype == 'price_first' and len(match) == 2:
                            price = self._extract_price_improved(match[0])
                            name = self._clean_name_improved(match[1])
                            if price and price > 0 and name and len(name) > 2:
                                items.append({
                                    'name': name[:50],
                                    'quantity': 1,
                                    'price': price,
                                    'total': price
                                })
                        
                        elif ptype == 'with_at' and len(match) == 2:
                            name = self._clean_name_improved(match[0])
                            price = self._extract_price_improved(match[1])
                            if price and price > 0 and name and len(name) > 2:
                                items.append({
                                    'name': name[:50],
                                    'quantity': 1,
                                    'price': price,
                                    'total': price
                                })
                    except:
                        continue
        
        # Gabungkan item yang sama (quantity diakumulasi)
        merged_items = {}
        for item in items:
            key = (item['name'].lower(), item['price'])
            if key in merged_items:
                merged_items[key]['quantity'] += item['quantity']
                merged_items[key]['total'] = merged_items[key]['quantity'] * merged_items[key]['price']
            else:
                merged_items[key] = item.copy()
        
        return list(merged_items.values())
    
    def _parse_items_from_raw(self, text):
        """Fallback: parse langsung dari raw text"""
        items = []
        
        # Cari semua pola "angka x teks angka"
        pattern = re.compile(r'(\d+)\s*[xX]\s*([A-Za-z0-9\s]+?)\s*([\d.,]+)', re.IGNORECASE)
        matches = pattern.findall(text)
        
        for match in matches:
            try:
                qty = int(match[0])
                name = self._clean_name_improved(match[1])
                price = self._extract_price_improved(match[2])
                if price and price > 0 and name and len(name) > 2:
                    items.append({
                        'name': name[:50],
                        'quantity': qty,
                        'price': price,
                        'total': qty * price
                    })
            except:
                continue
        
        # Hapus duplikat
        unique = []
        seen = set()
        for item in items:
            key = (item['name'].lower(), item['price'])
            if key not in seen:
                seen.add(key)
                unique.append(item)
        
        return unique
    
    def _clean_name_improved(self, name):
        """Clean name tanpa menghilangkan angka penting"""
        if not name:
            return ""
        # Hapus karakter aneh tapi pertahankan huruf, angka, dan spasi
        name = re.sub(r'[^\w\s]', '', name)
        name = re.sub(r'\s+', ' ', name).strip()
        # Hapus angka di awal saja (bukan di tengah)
        name = re.sub(r'^\d+\s*', '', name)
        return name[:50]
    
    def _extract_price_improved(self, price_str):
        if not price_str:
            return None
        
        price_str = str(price_str).strip()
        price_str = re.sub(r'[Rr]p\.?\s*', '', price_str)
        price_str = re.sub(r'[^\d.,-]', '', price_str)
        
        # Handle format Indonesia
        if '.' in price_str and ',' in price_str:
            price_str = price_str.replace('.', '').replace(',', '.')
        elif '.' in price_str:
            parts = price_str.split('.')
            if len(parts) == 2 and len(parts[1]) == 3:
                price_str = price_str.replace('.', '')
        elif ',' in price_str:
            parts = price_str.split(',')
            if len(parts) == 2 and len(parts[1]) == 3:
                price_str = price_str.replace(',', '')
            else:
                price_str = price_str.replace(',', '.')
        
        price_str = re.sub(r'[^\d.]', '', price_str)
        
        try:
            return float(price_str)
        except:
            return None
    
    def _extract_total_improved(self, text):
        patterns = [
            r'(?:GRAND TOTAL|Grand Total|TOTAL|Total)[\s:*]*([\d.,]+)',
            r'Grand Totai?l?[\s:]*([\d.,]+)',
            r'Amount Due[\s:]*([\d.,]+)',
            r'([\d.,]+)\s*$',
        ]
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                for match in reversed(matches):
                    price = self._extract_price_improved(match)
                    if price and price > 1000:
                        return price
        return None
    
    def _extract_subtotal_improved(self, text):
        patterns = [
            r'(?:SUB TOTAL|Sub Total|SUBTOTAL|Subtotal)[\s:]*([\d.,]+)',
            r'Sub[\s-]?Total[\s:]*([\d.,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_price_improved(match.group(1))
        return None
    
    def _extract_tax_improved(self, text):
        patterns = [
            r'(?:PBI|PB1|PAJAK|TAX|PPN)[\s:]*([\d.,]+)',
            r'Tax[\s:]*([\d.,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_price_improved(match.group(1))
        return None
    
    def _extract_cash_improved(self, text):
        patterns = [
            r'(?:CASH|Cash|TUNAI|Tunai)[\s:]*([\d.,]+)',
            r'Bayar[\s:]*([\d.,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_price_improved(match.group(1))
        return None
    
    def _extract_change_improved(self, text):
        patterns = [
            r'(?:CHANGE|Change|KEMBALI|Kembali)[\s:]*([\d.,]+)',
            r'Kembalian[\s:]*([\d.,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_price_improved(match.group(1))
        return None
    
    def _extract_discount_improved(self, text):
        patterns = [
            r'(?:DISCOUNT|Discount|DISKON|Diskon)[\s:]*([\d.,]+)',
            r'Potongan[\s:]*([\d.,]+)',
            r'-([\d.,]+)',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return self._extract_price_improved(match.group(1))
        return None
    
    def process_image(self, image_path):
        extracted_data = self.extract_text(image_path)
        parsed_data = self.parse_receipt(extracted_data)
        return extracted_data, parsed_data


# ============================================================================
# PREPROCESS IMAGE
# ============================================================================

def preprocess_image(image_path, output_path=None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # Kembalikan ke rasio normal (tanpa manipulasi tinggi)
    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # Grayscale bersih
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if output_path:
        cv2.imwrite(str(output_path), gray)

    return gray


def group_into_lines(extracted_data, y_tolerance=15, x_tolerance=50):
    
    extractor = ReceiptExtractor()
    return extractor._group_lines_advanced(extracted_data)


def is_valid_image(image_path):
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return False, "Cannot read image"
        h, w = img.shape[:2]
        if h < 50 or w < 50:
            return False, f"Image too small: {w}x{h}"
        return True, "OK"
    except Exception as e:
        return False, str(e)


if __name__ == "__main__":
    print("=" * 60)
    print(" TESTING UTILS.PY")
    print("=" * 60)
    
    extractor = ReceiptExtractor(lang='id')
    
    test_lines = [
        "1 x AYAM BAKAR 55.000",
        "NASI GORENG 25.000",
        "2x ES TEH 10.000",
        "KOPI @15.000",
    ]
    
    print("\n Testing item extraction:")
    for line in test_lines:
        items = extractor._parse_items_improved([line])
        if items:
            for item in items:
                print(f"   '{line}' -> {item['quantity']} x {item['name']} = Rp {item['total']:,.0f}")
        else:
            print(f"   '{line}' -> NOT DETECTED")
    
    print("\n utils.py ready!")