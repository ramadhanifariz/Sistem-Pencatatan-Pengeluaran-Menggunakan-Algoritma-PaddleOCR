"""
app.py - FastAPI Deployment dengan MLflow Model Registry
"""

import os
import tempfile
import numpy as np
import cv2
import re
from pathlib import Path
from dotenv import load_dotenv
import mlflow
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
import uvicorn

# Load environment variables
load_dotenv()

# ========== KONFIGURASI ==========
MODEL_NAME = os.getenv("MODEL_NAME", "Receipt_Total_Predictor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

# Jika tidak ada tracking URI, coba gunakan DagsHub default
if not MLFLOW_TRACKING_URI:
    MLFLOW_TRACKING_URI = "https://dagshub.com/ramadhanifariz/Sistem-Pencatatan-Pengeluaran-Menggunakan-Algoritma-PaddleOCR.mlflow"

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Load model from registry by alias
print(f"🔄 Loading model: {MODEL_NAME} (alias: {MODEL_ALIAS})")
print(f"🔗 Tracking URI: {MLFLOW_TRACKING_URI}")

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"⚠️ Could not load from registry: {e}")
    model = None

# Load OCR
try:
    from paddleocr import PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang='id', show_log=False)
    print("✅ PaddleOCR loaded")
except Exception as e:
    print(f"⚠️ OCR not loaded: {e}")
    ocr = None

app = FastAPI(title="Receipt Extraction API")


def preprocess_image(image_bytes):
    """Preprocess image for OCR"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    if max(h, w) < 1000:
        scale = 1200 / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.fastNlMeansDenoising(enhanced, None, 10, 7, 21)
    
    return denoised


def extract_features(ocr_result):
    """Extract features from OCR result"""
    if not ocr_result or not ocr_result[0]:
        return None
    
    num_items = 0
    has_tax = 0
    has_discount = 0
    total_confidence = 0
    
    for line in ocr_result[0]:
        if line and len(line) >= 2:
            text = line[1][0] if line[1] else ""
            confidence = line[1][1] if len(line[1]) > 1 else 0
            total_confidence += confidence
            
            if re.search(r'\d{3,}', text):
                num_items += 1
            if 'tax' in text.lower() or 'pajak' in text.lower():
                has_tax = 1
            if 'discount' in text.lower() or 'diskon' in text.lower():
                has_discount = 1
    
    avg_confidence = total_confidence / len(ocr_result[0]) if ocr_result[0] else 0
    
    return np.array([[num_items, has_tax, has_discount, avg_confidence]])


# HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Receipt Extractor</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; cursor: pointer; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; }
        .result { margin-top: 20px; padding: 20px; background: #f0f0f0; border-radius: 10px; display: none; }
        .total { font-size: 24px; color: green; font-weight: bold; }
    </style>
</head>
<body>
    <h1>🛒 Ekstraksi Struk Belanja</h1>
    <div class="upload-area" id="dropZone">
        <p>📸 Klik atau drag & drop gambar struk di sini</p>
        <input type="file" id="fileInput" style="display:none">
    </div>
    <button onclick="uploadFile()">Ekstrak</button>
    <div class="result" id="result">
        <h3>📊 Hasil Ekstraksi</h3>
        <div id="resultContent"></div>
    </div>
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#007bff'; };
        dropZone.ondragleave = () => dropZone.style.borderColor = '#ccc';
        dropZone.ondrop = (e) => { e.preventDefault(); fileInput.files = e.dataTransfer.files; uploadFile(); };
        fileInput.onchange = () => uploadFile();
        
        async function uploadFile() {
            const file = fileInput.files[0];
            if (!file) return;
            const formData = new FormData();
            formData.append('file', file);
            const response = await fetch('/predict', { method: 'POST', body: formData });
            const data = await response.json();
            const resultDiv = document.getElementById('result');
            const resultContent = document.getElementById('resultContent');
            resultDiv.style.display = 'block';
            if (response.ok) {
                resultContent.innerHTML = `<p class="total">💰 Total: Rp ${data.total.toLocaleString('id-ID')}</p>
                                          <p>📦 Items detected: ${data.num_items}</p>
                                          <p>🎯 Confidence: ${(data.confidence * 100).toFixed(1)}%</p>`;
            } else {
                resultContent.innerHTML = `<p style="color:red">Error: ${data.error}</p>`;
            }
        }
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return HTML_TEMPLATE


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        
        # Preprocess
        processed = preprocess_image(contents)
        if processed is None:
            return JSONResponse(status_code=400, content={"error": "Cannot read image"})
        
        # Save temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            cv2.imwrite(tmp.name, processed)
            tmp_path = tmp.name
        
        # OCR
        if ocr is None:
            return JSONResponse(status_code=500, content={"error": "OCR not available"})
        
        result = ocr.ocr(tmp_path, cls=True)
        os.unlink(tmp_path)
        
        # Extract features
        features = extract_features(result)
        
        # Predict using model
        if model is not None and features is not None:
            prediction = model.predict(features)[0]
            total = float(prediction)
            num_items = int(features[0][0])
            confidence = float(features[0][3])
        else:
            # Fallback: extract from text
            total = None
            for line in result[0] if result else []:
                text = line[1][0] if line[1] else ""
                match = re.search(r'[\d.,]{5,}', text)
                if match:
                    price_str = match.group().replace(',', '').replace('.', '')
                    total = float(price_str)
                    break
            num_items = len(result[0]) if result else 0
            confidence = 0.8
        
        return JSONResponse(content={
            "total": total if total else 0,
            "num_items": num_items,
            "confidence": confidence
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
