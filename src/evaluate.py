"""
evaluate.py - Evaluasi Model (TIDAK WAJIB untuk deployment)
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import sys

sys.path.append('src')
from utils import ReceiptExtractor

load_dotenv()

# Setup paths
DATA_DIR = Path('data')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)


def extract_features_from_annotation(data):
    """Extract features from annotation data"""
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


def load_test_data():
    """Load test data from splits"""
    with open(SPLITS_DIR / 'data_splits.json', 'r') as f:
        splits = json.load(f)
    
    test_data = []
    for ann_path in splits.get('test', []):
        p_path = Path(ann_path)
        if p_path.exists():
            with open(p_path, 'r') as f:
                test_data.append(json.load(f))
    
    return test_data


def evaluate_model():
    """Evaluate model on test data"""
    print("\n" + "=" * 50)
    print("📊 MODEL EVALUATION")
    print("=" * 50)
    
    test_data = load_test_data()
    if not test_data:
        print("⚠️ No test data found.")
        return
    
    # Extract features
    X_test, y_test = extract_features_from_annotation(test_data)
    
    if len(X_test) == 0:
        print("⚠️ No valid test samples.")
        return
    
    # Load model
    import joblib
    model_path = MODELS_DIR / 'receipt_model_advanced.pkl'
    if not model_path.exists():
        print("⚠️ Model not found. Run train.py first.")
        return
    
    model_info = joblib.load(model_path)
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Scale and predict
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"\n📈 Test Metrics:")
    print(f"   MAE : Rp {mae:,.0f}")
    print(f"   RMSE: Rp {rmse:,.0f}")
    print(f"   R²  : {r2:.4f}")
    
    # Save metrics
    metrics = {'mae': mae, 'rmse': rmse, 'r2': r2, 'test_samples': len(test_data)}
    with open(RESULTS_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    return metrics


if __name__ == "__main__":
    evaluate_model()