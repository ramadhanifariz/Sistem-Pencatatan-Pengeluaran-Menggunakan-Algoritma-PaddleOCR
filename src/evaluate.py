import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import dagshub
import mlflow
sys.path.append('src')
from utils import ReceiptExtractor

load_dotenv

print(os.getenv("MLFLOW_TRACKING_USERNAME"))
# Setup paths
DATA_DIR = Path('data')
ANNOTATIONS_DIR = DATA_DIR / 'annotations'
SPLITS_DIR = DATA_DIR / 'splits'
MODELS_DIR = Path('models')
RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)

dagshub.init(
    repo_name="Sistem-Pencatatan-Pengeluaran-Menggunakan-Algoritma-PaddleOCR",
    repo_owner="ramadhanifariz",
    mlflow="true"
)

#print("TRACKING URI :", mlflow.get_tracking_uri())


def load_test_data():
    """Load test data"""
    with open(SPLITS_DIR / 'data_splits.json', 'r') as f:
        splits = json.load(f)
    
    test_data = []
    for ann_path in splits['test']:
        with open(ann_path, 'r') as f:
            data = json.load(f)
            test_data.append(data)
    
    return test_data

def evaluate_on_new_image(image_path):
    """Evaluate model on a single new image"""
    print(f"\n Evaluating on: {image_path}")
    
    # Initialize extractor
    extractor = ReceiptExtractor(lang='id')
    
    # Process image
    extracted, parsed = extractor.process_image(image_path)
    
    print(f"\n Extracted Receipt Data:")
    print(f"   - Items found: {len(parsed['items'])}")
    print(f"   - Total amount: Rp {parsed['total']:,.0f}" if parsed['total'] else "   - Total: Not found")
    print(f"   - Tax: Rp {parsed['tax']:,.0f}" if parsed['tax'] else "   - Tax: Not found")
    print(f"   - Discount: Rp {parsed['discount']:,.0f}" if parsed['discount'] else "   - Discount: Not found")
    print(f"   - Cash paid: Rp {parsed['cash']:,.0f}" if parsed['cash'] else "   - Cash: Not found")
    print(f"   - Change: Rp {parsed['change']:,.0f}" if parsed['change'] else "   - Change: Not found")
    
    return parsed

def evaluate_model():
    """Evaluate model on test data"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)
    
    # Load test data
    test_data = load_test_data()
    
    if not test_data:
        print("⚠️ Tidak ada data test. Jalankan prepare.py terlebih dahulu.")
        return
    
    print(f"\n Testing on {len(test_data)} samples")
    
    # Extract features and targets
    from train import extract_features
    X_test, y_test = extract_features(test_data)
    
    # Load model
    import joblib
    model_path = MODELS_DIR / 'receipt_model.pkl'
    if not model_path.exists():
        print("⚠️ Model tidak ditemukan. Jalankan train.py terlebih dahulu.")
        return
    
    model_info = joblib.load(model_path)
    model = model_info['model']
    scaler = model_info['scaler']
    
    # Scale features
    X_test_scaled = scaler.transform(X_test)
    
    # Predict
    y_pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
    
    print(f"\n Test Metrics:")
    print(f"   - MAE: Rp {mae:,.0f}")
    print(f"   - RMSE: Rp {rmse:,.0f}")
    print(f"   - R² Score: {r2:.4f}")
    print(f"   - MAPE: {mape:.2f}%")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('Actual Total (Rp)')
    axes[0, 0].set_ylabel('Predicted Total (Rp)')
    axes[0, 0].set_title(f'Actual vs Predicted (R² = {r2:.3f})')
    
    # Plot 2: Residuals
    residuals = y_test - y_pred
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5)
    axes[0, 1].axhline(y=0, color='r', linestyle='--')
    axes[0, 1].set_xlabel('Predicted Total (Rp)')
    axes[0, 1].set_ylabel('Residuals (Rp)')
    axes[0, 1].set_title('Residual Plot')
    
    # Plot 3: Error Distribution
    axes[1, 0].hist(residuals, bins=20, edgecolor='black')
    axes[1, 0].set_xlabel('Prediction Error (Rp)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Error Distribution')
    
    # Plot 4: Feature Importance
    feature_names = model_info['features']
    coefficients = model.coef_
    axes[1, 1].barh(feature_names, coefficients)
    axes[1, 1].set_xlabel('Coefficient Value')
    axes[1, 1].set_title('Feature Importance')
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / 'evaluation_plots.png', dpi=150)
    plt.show()
    
    # Save metrics
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'mape': mape,
        'test_samples': len(test_data)
    }
    
    with open(RESULTS_DIR / 'test_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f"\n Evaluation results saved to: {RESULTS_DIR}")
    
    return metrics

def demo_extraction():
    """Demo extraction on a sample receipt"""
    print("\n" + "=" * 50)
    print("DEMO: Receipt Extraction")
    print("=" * 50)
    
    # Find a test image
    test_images = list(Path('data/raw/test').glob('*.jpg')) + list(Path('data/raw/test').glob('*.png'))
    
    if test_images:
        print(f"\n Testing on sample receipt: {test_images[0].name}")
        evaluate_on_new_image(str(test_images[0]))
    else:
        print("\n No test images found. Please add some images to data/raw/test/")
        print("\nYou can test with a custom image path:")
        image_path = input("Enter image path (or press Enter to skip): ")
        if image_path and Path(image_path).exists():
            evaluate_on_new_image(image_path)

if __name__ == "__main__":
    evaluate_model()
    demo_extraction()