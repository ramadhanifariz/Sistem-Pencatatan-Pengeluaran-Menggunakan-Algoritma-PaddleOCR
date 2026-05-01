import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import sys
import re
import joblib
import os
import tempfile
import traceback

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV

# KONFIGURASI PATH YANG FLEKSIBEL 
current_file = Path(__file__).resolve()
src_dir = current_file.parent  
root_candidate = src_dir.parent

if (root_candidate / 'data').exists():
    PROJECT_ROOT = root_candidate
else:
    PROJECT_ROOT = src_dir
    print(f" Tidak ditemukan folder 'data' di {root_candidate}, menggunakan {PROJECT_ROOT} sebagai root.")

DATA_DIR = PROJECT_ROOT / 'data'
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'
MODELS_DIR = PROJECT_ROOT / 'models'
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MLFLOW_DB_PATH = PROJECT_ROOT / "mlflow.db"
MLFLOW_DB_URI = f"sqlite:///{MLFLOW_DB_PATH}"
mlflow.set_tracking_uri(MLFLOW_DB_URI)
mlflow.set_experiment("Receipt_Total_Prediction")

print(f" Root proyek: {PROJECT_ROOT}")
print(f" Anotasi: {ANNOTATIONS_DIR}")
print(f" MLflow DB: {MLFLOW_DB_PATH}")

# LOAD DATA 
def load_annotations_from_split(split_key):
    """Load annotations from data_splits.json, dengan fallback ke semua file anotasi jika split tidak ada."""
    splits_path = SPLITS_DIR / 'data_splits.json'
    if not splits_path.exists():
        print(f" {splits_path} tidak ditemukan. Mencoba menggunakan semua anotasi sebagai training.")
        all_ann = list(ANNOTATIONS_DIR.glob('*.json'))
        if not all_ann:
            print(f" Tidak ada file anotasi di {ANNOTATIONS_DIR}")
            return []
        data = []
        for p in all_ann:
            with open(p, 'r') as f:
                data.append(json.load(f))
        print(f" Fallback: memuat {len(data)} anotasi untuk training.")
        return data
    else:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        ann_paths = splits.get(split_key, [])
        if not ann_paths:
            print(f" Key '{split_key}' kosong di data_splits.json. Isi: {list(splits.keys())}")
        data = []
        for p in ann_paths:
            p_path = Path(p)
            if p_path.exists():
                with open(p_path, 'r') as f:
                    data.append(json.load(f))
            else:
                print(f" File anotasi tidak ditemukan: {p}")
        print(f" Loaded {len(data)} samples from '{split_key}' set")
        return data

def extract_features_and_target(data):
    features = []
    targets = []
    for d in data:
        parsed = d.get('parsed_data', {})
        extracted = d.get('extracted_data', [])
        
        num_items = len(parsed.get('items', []))
        has_tax = 1 if parsed.get('tax') else 0
        has_discount = 1 if parsed.get('discount') else 0
        has_service_charge = 1 if parsed.get('service_charge') else 0
        
        confs = [item['confidence'] for item in extracted]
        avg_conf = np.mean(confs) if confs else 0.0
        min_conf = np.min(confs) if confs else 0.0
        std_conf = np.std(confs) if confs else 0.0
        
        all_text = ' '.join([item['text'] for item in extracted])
        text_len = len(all_text)
        num_numbers = len(re.findall(r'\d+', all_text))
        
        items_total = sum(item.get('total', 0) for item in parsed.get('items', []))
        
        total = parsed.get('total')
        if total is None or np.isnan(total) or total <= 0:
            continue
        
        features.append([
            num_items, has_tax, has_discount, has_service_charge,
            avg_conf, min_conf, std_conf, text_len, num_numbers, items_total
        ])
        targets.append(total)
    
    if not features:
        return np.array([]), np.array([])
    X = np.array(features, dtype=np.float64)
    y = np.array(targets, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mask = ~(np.isnan(y) | np.isinf(y))
    return X[mask], y[mask]

# TRAINING 
def train_model():
    print("TRAINING RECEIPT REGRESSION MODEL ")
    
    if not ANNOTATIONS_DIR.exists():
        print(f" Folder anotasi tidak ditemukan: {ANNOTATIONS_DIR}")
        return None
    
    ann_files = list(ANNOTATIONS_DIR.glob('*.json'))
    print(f"\n Jumlah file anotasi: {len(ann_files)}")
    if len(ann_files) == 0:
        print(" Tidak ada file JSON anotasi. Pastikan preprocess.py sudah dijalankan dengan benar.")
        return None
    else:
        print(f"   Contoh: {ann_files[0].name}")
    
    train_data = load_annotations_from_split('train')
    val_data = load_annotations_from_split('validation')
    test_data = load_annotations_from_split('test')
    
    if len(train_data) == 0:
        print("\n Tidak ada data training. Split tidak valid.")
        return None
    
    # Ekstrak fitur
    X_train, y_train = extract_features_and_target(train_data)
    X_val, y_val = extract_features_and_target(val_data) if val_data else (np.array([]), np.array([]))
    X_test, y_test = extract_features_and_target(test_data) if test_data else (np.array([]), np.array([]))
    
    print(f"\n Jumlah sampel valid (total > 0):")
    print(f"   Training : {X_train.shape[0]}")
    if X_val.shape[0] > 0:
        print(f"   Validation: {X_val.shape[0]}")
    else:
        print("   Validation: (tidak ada, gunakan cross-validation saja)")
    if X_test.shape[0] > 0:
        print(f"   Test     : {X_test.shape[0]}")
    
    if X_train.shape[0] == 0:
        print(" Tidak ada sampel training yang valid. Periksa apakah kolom 'total' di anotasi ada dan >0.")
        return None
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if X_val.shape[0] > 0 else None
    X_test_scaled = scaler.transform(X_test) if X_test.shape[0] > 0 else None
    
    # Hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_scaled, y_train)
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    print(f" Best parameters: {best_params}")
    print(f"   Best CV MAE  : Rp {-grid_search.best_score_:,.0f}")
    
    # Evaluasi
    val_metrics = {}
    test_metrics = {}
    if X_val.shape[0] > 0:
        y_val_pred = best_model.predict(X_val_scaled)
        val_metrics = {
            'mae': mean_absolute_error(y_val, y_val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, y_val_pred)),
            'r2': r2_score(y_val, y_val_pred)
        }
        print(f"\n Validation set performance:")
        print(f"   MAE : Rp {val_metrics['mae']:,.0f}")
        print(f"   RMSE: Rp {val_metrics['rmse']:,.0f}")
        print(f"   R²  : {val_metrics['r2']:.4f}")
    
    if X_test.shape[0] > 0:
        y_test_pred = best_model.predict(X_test_scaled)
        test_metrics = {
            'mae': mean_absolute_error(y_test, y_test_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'r2': r2_score(y_test, y_test_pred)
        }
        print(f"\n Test set performance:")
        print(f"   MAE : Rp {test_metrics['mae']:,.0f}")
        print(f"   RMSE: Rp {test_metrics['rmse']:,.0f}")
        print(f"   R²  : {test_metrics['r2']:.4f}")
    
    # MLflow logging
    try:
        with mlflow.start_run(run_name="RandomForest_Tuned") as run:
            mlflow.log_params(best_params)
            mlflow.log_param("model_type", "RandomForestRegressor")
            mlflow.log_param("scaler", "StandardScaler")
            mlflow.log_metric("best_cv_mae", -grid_search.best_score_)
            if val_metrics:
                for k, v in val_metrics.items():
                    mlflow.log_metric(f"val_{k}", v)
            if test_metrics:
                for k, v in test_metrics.items():
                    mlflow.log_metric(f"test_{k}", v)
            
            mlflow.sklearn.log_model(best_model, "random_forest_model")
            
            # Log scaler
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                joblib.dump(scaler, f.name)
                mlflow.log_artifact(f.name, artifact_path="preprocessor")
            os.unlink(f.name)
            
            # Feature importance
            feature_names = ["num_items", "has_tax", "has_discount", "has_service_charge",
                             "avg_confidence", "min_confidence", "std_confidence",
                             "text_length", "num_numbers", "items_total"]
            importances = best_model.feature_importances_
            imp_df = pd.DataFrame({'feature': feature_names, 'importance': importances})
            imp_df = imp_df.sort_values('importance', ascending=False)
            imp_csv = "feature_importance.csv"
            imp_df.to_csv(imp_csv, index=False)
            mlflow.log_artifact(imp_csv)
            os.remove(imp_csv)
            
            print(f"\n MLflow run successful! Run ID: {run.info.run_id}")
            print(f" Artifact URI: {run.info.artifact_uri}")
    except Exception as e:
        print(f" Gagal logging ke MLflow: {e}")
        traceback.print_exc()
    
    # Simpan lokal
    model_info = {
        'model': best_model,
        'scaler': scaler,
        'features': feature_names,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat()
    }
    local_path = MODELS_DIR / 'receipt_model_advanced.pkl'
    joblib.dump(model_info, local_path)
    print(f"\n Model lokal disimpan ke: {local_path}")
    
    print(f"   mlflow ui --backend-store-uri {MLFLOW_DB_URI}")
    
    return best_model, scaler

if __name__ == "__main__":
    train_model()