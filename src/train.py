import json
import numpy as np
from pathlib import Path
import pandas as pd
import mlflow
import mlflow.sklearn
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tempfile
import os
import sys

# ========== KONFIGURASI MLFLOW ==========
PROJECT_ROOT = Path(__file__).parent.parent
MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_DB_URI = f"sqlite:///{MLFLOW_DB_PATH}"

# Set tracking URI
mlflow.set_tracking_uri(MLFLOW_DB_URI)

# Set experiment
mlflow.set_experiment("Receipt_OCR_Extraction")

DATA_DIR = Path('data')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'


def extract_features_and_target(data):
    """
    Extract features from OCR results for model training
    """
    features = []
    targets = []
    
    for d in data:
        parsed = d.get('parsed_data', {})
        extracted = d.get('extracted_data', [])
        
        num_items = len(parsed.get('items', []))
        has_tax = 1 if parsed.get('tax') else 0
        has_discount = 1 if parsed.get('discount') else 0
        
        confs = [item['confidence'] for item in extracted]
        avg_conf = np.mean(confs) if confs else 0.0
        
        total = parsed.get('total')
        if total is None or total <= 0:
            continue
        
        features.append([num_items, has_tax, has_discount, avg_conf])
        targets.append(total)
    
    return np.array(features), np.array(targets)


def load_annotations():
    """
    Load all annotations from JSON files
    """
    all_annotations = list(ANNOTATIONS_DIR.glob('*.json'))
    data = []
    
    for ann_file in all_annotations:
        try:
            with open(ann_file, 'r', encoding='utf-8') as f:
                data.append(json.load(f))
        except Exception as e:
            print(f"⚠️ Error loading {ann_file}: {e}")
    
    return data


def evaluate_ocr_accuracy():
    """
    Evaluasi akurasi ekstraksi OCR
    """
    print("=" * 60)
    print("📊 EVALUASI AKURASI EKSTRAKSI OCR")
    print("=" * 60)
    
    all_annotations = list(ANNOTATIONS_DIR.glob('*.json'))
    
    if not all_annotations:
        print("❌ Tidak ada file anotasi!")
        return None
    
    results = []
    
    for ann_file in all_annotations:
        with open(ann_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        parsed = data.get('parsed_data', {})
        extracted_total = parsed.get('total')
        num_items = len(parsed.get('items', []))
        
        extracted_data = data.get('extracted_data', [])
        confidences = [item['confidence'] for item in extracted_data]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        results.append({
            'filename': data.get('filename', 'unknown'),
            'total_detected': extracted_total,
            'num_items_detected': num_items,
            'num_text_blocks': len(extracted_data),
            'avg_confidence': avg_confidence,
            'has_tax': parsed.get('tax') is not None,
            'has_discount': parsed.get('discount') is not None,
            'has_cash': parsed.get('cash') is not None,
            'has_change': parsed.get('change') is not None,
        })
    
    df = pd.DataFrame(results)
    
    print("\n📈 STATISTIK EKSTRAKSI OCR:")
    print("-" * 40)
    print(f"   Total gambar diproses: {len(df)}")
    print(f"   Rata-rata item terdeteksi: {df['num_items_detected'].mean():.1f}")
    print(f"   Rata-rata confidence OCR: {df['avg_confidence'].mean():.2%}")
    print(f"   Total terdeteksi: {(df['total_detected'].notna()).sum()} dari {len(df)} gambar")
    print(f"   Tax terdeteksi: {df['has_tax'].sum()} dari {len(df)}")
    print(f"   Discount terdeteksi: {df['has_discount'].sum()} dari {len(df)}")
    
    return df


def log_ocr_evaluation_to_mlflow(df_results):
    """
    Log hasil evaluasi OCR ke MLflow
    """
    if df_results is None or df_results.empty:
        print("⚠️ Tidak ada data evaluasi")
        return
    
    with mlflow.start_run(run_name="OCR_Evaluation") as run:
        
        # Log parameters
        mlflow.log_params({
            "ocr_engine": "PaddleOCR",
            "evaluation_type": "OCR_Extraction_Accuracy",
            "total_images": len(df_results)
        })
        
        # Log metrics
        metrics = {
            "avg_items_per_receipt": float(df_results['num_items_detected'].mean()),
            "avg_confidence": float(df_results['avg_confidence'].mean()),
            "total_detection_rate": float(df_results['total_detected'].notna().mean()),
            "tax_detection_rate": float(df_results['has_tax'].mean()),
            "discount_detection_rate": float(df_results['has_discount'].mean()),
            "cash_detection_rate": float(df_results['has_cash'].mean()),
            "change_detection_rate": float(df_results['has_change'].mean()),
        }
        
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
            print(f"   📈 {key}: {value:.4f}")
        
        # Log artifacts - menggunakan path biasa (bukan tempfile)
        artifact_dir = Path("mlflow_artifacts")
        artifact_dir.mkdir(exist_ok=True)
        
        # Save evaluation summary
        summary_path = artifact_dir / "ocr_evaluation_summary.csv"
        df_results.to_csv(summary_path, index=False)
        mlflow.log_artifact(str(summary_path), artifact_path="evaluation")
        
        print(f"\n✅ OCR Evaluation logged to MLflow")
        print(f"   Run ID: {run.info.run_id}")


def train_and_track_model():
    """
    Training model dengan MLflow tracking
    """
    print("=" * 60)
    print("🚀 TRAINING MODEL DENGAN MLflow TRACKING")
    print("=" * 60)
    
    # Load data
    data = load_annotations()
    if len(data) == 0:
        print("❌ Tidak ada data anotasi!")
        return None
    
    # Extract features
    X, y = extract_features_and_target(data)
    if len(X) == 0:
        print("❌ Tidak ada data valid untuk training!")
        return None
    
    print(f"\n📊 Data shape: X={X.shape}, y={y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # ========== TRAINING DENGAN MLflow ==========
    with mlflow.start_run(run_name="RandomForest_Receipt") as run:
        
        # Enable autologging
        mlflow.sklearn.autolog()
        
        # Log parameters manually
        mlflow.log_params({
            "model_type": "RandomForestRegressor",
            "n_estimators": 100,
            "max_depth": 10,
            "random_state": 42,
            "test_size": 0.2,
            "features": ["num_items", "has_tax", "has_discount", "avg_confidence"]
        })
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train, y_train)
        
        # Predict
        y_pred = model.predict(X_test)
        
        # Log additional metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        mlflow.log_metrics({
            "test_mae": mae,
            "test_rmse": rmse,
            "test_r2": r2
        })
        
        print(f"\n📈 Model Performance on Test Set:")
        print(f"   MAE : Rp {mae:,.0f}")
        print(f"   RMSE: Rp {rmse:,.0f}")
        print(f"   R²  : {r2:.4f}")
        
        # ========== LOG FEATURE IMPORTANCE (FIXED) ==========
        feature_names = ["num_items", "has_tax", "has_discount", "avg_confidence"]
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        })
        
        # Simpan ke file biasa (bukan tempfile) untuk menghindari PermissionError
        artifact_dir = Path("mlflow_artifacts")
        artifact_dir.mkdir(exist_ok=True)
        
        importance_path = artifact_dir / "feature_importance.csv"
        importance_df.to_csv(importance_path, index=False)
        mlflow.log_artifact(str(importance_path), artifact_path="feature_importance")
        
        # Log model
        mlflow.sklearn.log_model(model, "random_forest_model")
        
        # Register model ke Model Registry
        model_uri = f"runs:/{run.info.run_id}/random_forest_model"
        try:
            mlflow.register_model(model_uri, "Receipt_Total_Predictor")
            print(f"\n✅ Model registered to Model Registry: 'Receipt_Total_Predictor'")
        except Exception as e:
            print(f"\n⚠️ Model registration failed (maybe already exists): {e}")
        
        print(f"   Run ID: {run.info.run_id}")
        
        return model


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("📊 MLflow EXPERIMENT TRACKING (Sesuai Materi Dosen)")
    print("=" * 60)
    
    # Step 1: Evaluasi OCR
    print("\n📋 STEP 1: Evaluasi Akurasi OCR")
    df_results = evaluate_ocr_accuracy()
    
    if df_results is not None and len(df_results) > 0:
        # Step 2: Log OCR evaluation ke MLflow
        print("\n📋 STEP 2: Logging OCR Evaluation to MLflow")
        log_ocr_evaluation_to_mlflow(df_results)
    
    # Step 3: Train model dengan MLflow tracking
    print("\n📋 STEP 3: Training Model dengan MLflow Tracking")
    train_and_track_model()
    
    print("\n" + "=" * 60)
    print("✅ SEMUA PROSES SELESAI!")
    print("=" * 60)
    print("\n💡 Untuk melihat dashboard MLflow, jalankan di terminal:")
    print(f"   mlflow server --port 5000")
    print(f"   Kemudian buka: http://localhost:5000")