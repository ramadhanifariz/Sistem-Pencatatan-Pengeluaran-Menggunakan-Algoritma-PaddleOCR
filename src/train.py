import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import sys
sys.path.append('src')
from utils import ReceiptExtractor

# Setup paths
DATA_DIR = Path('data')
PROCESSED_DIR = DATA_DIR / 'processed'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'
MODELS_DIR = Path('models')
MODELS_DIR.mkdir(exist_ok=True)

def load_training_data():
    """Load training and validation data"""
    with open(SPLITS_DIR / 'data_splits.json', 'r') as f:
        splits = json.load(f)
    
    train_data = []
    val_data = []
    
    # Load training annotations
    for ann_path in splits['train']:
        with open(ann_path, 'r') as f:
            data = json.load(f)
            train_data.append(data)
    
    # Load validation annotations
    for ann_path in splits['validation']:
        with open(ann_path, 'r') as f:
            data = json.load(f)
            val_data.append(data)
    
    return train_data, val_data

def extract_features(data):
    """Extract features from receipt data"""
    features = []
    targets = []
    
    for d in data:
        parsed = d.get('parsed_data', {})
        
        # Features
        num_items = len(parsed.get('items', []))
        has_tax = 1 if parsed.get('tax') else 0
        has_discount = 1 if parsed.get('discount') else 0
        
        # Confidence scores from OCR
        extracted = d.get('extracted_data', [])
        confidences = [item['confidence'] for item in extracted]
        avg_confidence = np.mean(confidences) if confidences else 0
        min_confidence = np.min(confidences) if confidences else 0
        
        # Target (total amount)
        total = parsed.get('total', 0)
        
        features.append([
            num_items,
            has_tax,
            has_discount,
            avg_confidence,
            min_confidence
        ])
        targets.append(total)
    
    return np.array(features), np.array(targets)

def train_model():
    """Train a simple regression model"""
    print("\n" + "=" * 50)
    print("TRAINING MODEL")
    print("=" * 50)
    
    # Load data
    train_data, val_data = load_training_data()
    
    if not train_data:
        print("⚠️ Tidak ada data training. Jalankan prepare.py terlebih dahulu.")
        return None
    
    print(f"\n Data Training: {len(train_data)} samples")
    print(f" Data Validation: {len(val_data)} samples")
    
    # Extract features
    X_train, y_train = extract_features(train_data)
    X_val, y_val = extract_features(val_data)
    
    # Simple linear regression using numpy
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train model
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_r2 = r2_score(y_train, y_train_pred)
    
    val_mae = mean_absolute_error(y_val, y_val_pred)
    val_rmse = np.sqrt(mean_squared_error(y_val, y_val_pred))
    val_r2 = r2_score(y_val, y_val_pred)
    
    print(f"\n Training Metrics:")
    print(f"   - MAE: Rp {train_mae:,.0f}")
    print(f"   - RMSE: Rp {train_rmse:,.0f}")
    print(f"   - R² Score: {train_r2:.4f}")
    
    print(f"\n Validation Metrics:")
    print(f"   - MAE: Rp {val_mae:,.0f}")
    print(f"   - RMSE: Rp {val_rmse:,.0f}")
    print(f"   - R² Score: {val_r2:.4f}")
    
    # Save model
    import joblib
    model_info = {
        'model': model,
        'scaler': scaler,
        'features': ['num_items', 'has_tax', 'has_discount', 'avg_confidence', 'min_confidence'],
        'metrics': {
            'train': {'mae': train_mae, 'rmse': train_rmse, 'r2': train_r2},
            'validation': {'mae': val_mae, 'rmse': val_rmse, 'r2': val_r2}
        },
        'timestamp': datetime.now().isoformat()
    }
    
    joblib.dump(model_info, MODELS_DIR / 'receipt_model.pkl')
    print(f"\nModel saved to: {MODELS_DIR / 'receipt_model.pkl'}")
    
    return model_info

if __name__ == "__main__":
    train_model()