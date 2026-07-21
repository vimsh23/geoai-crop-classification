import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Add project root to path so we can import from src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import DataManager
from src.preprocessing_cloud_removal import DataPreprocessor

def get_rf_model(config):
    from sklearn.ensemble import RandomForestClassifier
    rf_config = config['models']['random_forest']
    return RandomForestClassifier(
        n_estimators=rf_config.get('n_estimators', 100),
        max_depth=rf_config.get('max_depth', 20),
        class_weight=rf_config.get('class_weight', 'balanced'),
        random_state=rf_config.get('random_state', 42),
        n_jobs=rf_config.get('n_jobs', -1)
    )

def train_and_evaluate_rf():
    data_manager = DataManager()
    config = data_manager.config
    
    # Ensure thresholds exist for the new preprocessor
    if 'preprocessing' not in config:
        config['preprocessing'] = {}
    
    # We define thresholds here so we don't have to modify config.yaml
    config['preprocessing']['cloud_threshold'] = 2500
    config['preprocessing']['shadow_threshold'] = 800
    
    preprocessor = DataPreprocessor(config)
    
    # Get train/test splits
    splits_dir = config['paths']['splits_dir']
    
    try:
        with open(os.path.join(splits_dir, "train_patches.txt"), "r") as f:
            train_patches = [int(line.strip()) for line in f if line.strip()]
        with open(os.path.join(splits_dir, "test_patches.txt"), "r") as f:
            test_patches = [int(line.strip()) for line in f if line.strip()]
    except FileNotFoundError:
        print("Splits not found. Please run src/train.py first to generate splits.")
        return
        
    print(f"Loaded {len(train_patches)} train patches and {len(test_patches)} test patches.")
    
    # Feature extraction helper
    def extract_features(patches, max_pixels=10000):
        X_list, y_list = [], []
        for pid in tqdm(patches):
            # Load original untouched .npy data
            s2_data, target_data = data_manager.load_patch_data(pid)
            
            # This uses our NEW cloud/shadow removal preprocessor
            features = preprocessor.prepare_patch_features(s2_data)
            
            X_patch, y_patch = preprocessor.flatten_for_rf(features, target_data)
            
            # Subsample to avoid memory explosion
            if len(y_patch) > max_pixels:
                idx = np.random.choice(len(y_patch), max_pixels, replace=False)
                X_patch, y_patch = X_patch[idx], y_patch[idx]
                
            X_list.append(X_patch)
            y_list.append(y_patch)
        return np.vstack(X_list), np.concatenate(y_list)

    print("\n[1/3] Extracting Training Features (with on-the-fly Cloud & Shadow Removal)...")
    X_train, y_train = extract_features(train_patches)
    
    print(f"\n[2/3] Training Random Forest on {len(X_train)} samples with {X_train.shape[1]} features...")
    model = get_rf_model(config)
    model.fit(X_train, y_train)
    
    print("\n[3/3] Extracting Test Features and Evaluating...")
    X_test, y_test = extract_features(test_patches)
    
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("\n" + "="*50)
    print("                RESULTS")
    print("="*50)
    print(f"Test Accuracy:    {acc:.4f}")
    print(f"Test Weighted F1: {f1:.4f}")
    print("="*50)
    
    # Generate report mapping IDs back to names
    labels = sorted(list(set(y_test) | set(y_pred)))
    target_names = [config['classes'][i] for i in labels]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, labels=labels, target_names=target_names))
    
    # Save the model
    model_dir = config['paths']['model_dir']
    model_path = os.path.join(model_dir, "rf_model_cloud_free.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\nSaved cloud-free RF model to {model_path}")
    print("Experiment complete! Original files were untouched.")

if __name__ == "__main__":
    train_and_evaluate_rf()
