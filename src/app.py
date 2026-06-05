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

load_dotenv()

MODEL_NAME = os.getenv("MODEL_NAME", "Receipt_Total_Predictor")
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion1")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")

if not MLFLOW_TRACKING_URI:
    MLFLOW_TRACKING_URI = "https://dagshub.com/ramadhanifariz/Sistem-Pencatatan-Pengeluaran-Menggunakan-Algoritma-PaddleOCR.mlflow"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

print(f" Loading model: {MODEL_NAME} (alias: {MODEL_ALIAS})")
print(f" Tracking URI: {MLFLOW_TRACKING_URI}")

try:
    model = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@{MODEL_ALIAS}")
    print(" Model loaded successfully")
except Exception as e:
    print(f" Could not load from registry: {e}")
    model = None

# Load OCR
try:
    from paddleocr import PaddleOCR
    
    ocr = PaddleOCR(use_angle_cls=True, lang='id', show_log=False, det_limit_side_len=2500, det_db_unclip_ratio=1.2)
    print(" PaddleOCR loaded")
except Exception as e:
    print(f" OCR not loaded: {e}")
    ocr = None

app = FastAPI(title="Receipt Extraction API")


def preprocess_image(image_bytes):
    
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    if w < 1000:
        scale = 1000 / w
        new_w = int(w * scale)
        new_h = int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return gray


def extract_features(ocr_result):
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


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Receipt Extractor</title>
    <style>
        body { font-family: Arial; max-width: 600px; margin: 50px auto; padding: 20px; }
        
        /* Modifikasi Jarak: margin-bottom ditambahkan agar tidak nempel dengan tombol */
        .upload-area { border: 2px dashed #ccc; padding: 40px; text-align: center; cursor: pointer; margin-bottom: 25px; border-radius: 10px; transition: 0.3s; }
        .upload-area:hover { background-color: #f9f9f9; }
        
        button { background: #007bff; color: white; padding: 12px 25px; border: none; cursor: pointer; border-radius: 5px; font-size: 16px; font-weight: bold; width: 100%; }
        button:hover { background: #0056b3; }
        
        .result { margin-top: 30px; padding: 20px; background: #f8f9fa; border-radius: 10px; display: none; border: 1px solid #e9ecef; }
        .total { font-size: 26px; color: #28a745; font-weight: bold; margin-bottom: 10px;}
        
        /* Styling untuk daftar teks OCR */
        .text-list { text-align: left; background: #fff; padding: 15px; border-radius: 8px; border: 1px solid #ddd; max-height: 350px; overflow-y: auto; font-family: monospace; font-size: 14px;}
        .text-list ul { padding-left: 0; margin: 0; list-style-type: none; }
        .text-list li { margin-bottom: 8px; border-bottom: 1px dashed #eee; padding-bottom: 5px; color: #333;}
    </style>
</head>
<body>
    <h1 style="text-align: center; color: #333;">🛒 Ekstraksi Struk Belanja</h1>
    <div class="upload-area" id="dropZone">
        <p style="font-size: 18px; color: #666;">📸 Klik atau drag & drop gambar struk di sini</p>
        <p style="font-size: 12px; color: #999;" id="fileNameDisplay"></p>
        <input type="file" id="fileInput" style="display:none" accept="image/*">
    </div>
    <button onclick="uploadFile()" id="extractBtn">Ekstrak Data Struk</button>
    <div class="result" id="result">
        <h3 style="border-bottom: 2px solid #007bff; padding-bottom: 10px;">📊 Hasil Ekstraksi</h3>
        <div id="resultContent"></div>
    </div>
    
    <script>
        const dropZone = document.getElementById('dropZone');
        const fileInput = document.getElementById('fileInput');
        const fileNameDisplay = document.getElementById('fileNameDisplay');
        const extractBtn = document.getElementById('extractBtn');
        
        dropZone.onclick = () => fileInput.click();
        dropZone.ondragover = (e) => { e.preventDefault(); dropZone.style.borderColor = '#007bff'; dropZone.style.backgroundColor = '#e9f5ff'; };
        dropZone.ondragleave = () => { dropZone.style.borderColor = '#ccc'; dropZone.style.backgroundColor = 'transparent'; };
        dropZone.ondrop = (e) => { 
            e.preventDefault(); 
            dropZone.style.borderColor = '#ccc';
            dropZone.style.backgroundColor = 'transparent';
            fileInput.files = e.dataTransfer.files; 
            showFileName();
        };
        
        fileInput.onchange = () => showFileName();
        
        function showFileName() {
            if(fileInput.files.length > 0) {
                fileNameDisplay.textContent = "File terpilih: " + fileInput.files[0].name;
                fileNameDisplay.style.color = "#28a745";
            }
        }
        
        async function uploadFile() {
            const file = fileInput.files[0];
            if (!file) {
                alert("Pilih gambar struk terlebih dahulu!");
                return;
            }
            
            extractBtn.textContent = "Sedang Mengekstrak...";
            extractBtn.disabled = true;
            extractBtn.style.background = "#6c757d";
            
            const formData = new FormData();
            formData.append('file', file);
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                resultDiv.style.display = 'block';
                
                if (response.ok) {
                    // Menyusun tampilan HTML
                    let htmlContent = `
                        <p class="total"> Total: Rp ${data.total.toLocaleString('id-ID')}</p>
                        <p> Items detected: ${data.num_items}</p>
                        <p> Confidence OCR: ${(data.confidence * 100).toFixed(1)}%</p>
                        <br>
                        <h4 style="color:#444; margin-bottom:10px;"> Detail Teks Terbaca:</h4>
                        <div class="text-list">
                            <ul>
                    `;
                    
                    // Looping data teks ekstraksi
                    if (data.extracted_texts && data.extracted_texts.length > 0) {
                        data.extracted_texts.forEach(text => {
                            htmlContent += `<li>${text}</li>`;
                        });
                    } else {
                        htmlContent += `<li style="color:red;">Tidak ada teks yang dapat dibaca.</li>`;
                    }
                    
                    htmlContent += `</ul></div>`;
                    resultContent.innerHTML = htmlContent;
                    
                } else {
                    resultContent.innerHTML = `<p style="color:red"><b>Error:</b> ${data.error}</p>`;
                }
            } catch (err) {
                document.getElementById('resultContent').innerHTML = `<p style="color:red">Terjadi kesalahan pada server.</p>`;
            } finally {
                extractBtn.textContent = "Ekstrak Data Struk";
                extractBtn.disabled = false;
                extractBtn.style.background = "#007bff";
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

        extracted_texts = []
        if result and result[0]:
            for line in result[0]:
                if line and len(line) >= 2:
                    text_string = line[1][0]
                    extracted_texts.append(text_string)
        
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
            "confidence": confidence,
            "extracted_texts": extracted_texts  
        })
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)