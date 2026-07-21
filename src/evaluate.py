import os
import json
import numpy as np
import pandas as pd
import pickle
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import DataManager
from src.preprocessing import DataPreprocessor
from src.dataset import PASTISDataset
from src.model import get_unet_model, get_resunet_model, get_transunet_model

class ModelEvaluator:
    """Handles evaluation of crop classification models."""

    def __init__(self):
        self.data_manager = DataManager()
        self.config = self.data_manager.config
        self.preprocessor = DataPreprocessor(self.config)
        self.s2_dir = self.config['paths']['s2_dir']
        self.target_dir = self.config['paths']['annotations_dir']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _evaluate_ml_model(self, test_patches, model_path, prefix):
        """Helper to evaluate classical ML models."""
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
            
        y_true_all, y_pred_all = [], []
        
        print(f"Extracting features for {prefix.strip('_').upper()} evaluation...")
        for pid in tqdm(test_patches):
            s2_data, target_data = self.data_manager.load_patch_data(pid)
            features = self.preprocessor.prepare_patch_features(s2_data)
            
            import rasterio.features
            
            C = features.shape[0]
            X_full = features.reshape(C, -1).T
            
            if X_full.shape[0] > 0:
                y_pred_full = model.predict(X_full).reshape(128, 128)
                
                y_pred_sieve = rasterio.features.sieve(y_pred_full.astype(np.int16), size=25, connectivity=4)
                
                target_arr = target_data.numpy() if torch.is_tensor(target_data) else np.array(target_data)
                y_patch_flat = target_arr.reshape(-1)
                y_pred_flat = y_pred_sieve.reshape(-1)
                
                ignore_classes = self.config.get('ignore_classes', [])
                mask = np.ones_like(y_patch_flat, dtype=bool)
                for ic in ignore_classes:
                    mask &= (y_patch_flat != ic)
                    
                y_true_all.extend(y_patch_flat[mask])
                y_pred_all.extend(y_pred_flat[mask])
                
        self.compute_and_save_metrics(y_true_all, y_pred_all, prefix=prefix)

    def evaluate_rf(self, test_patches, model_path):
        self._evaluate_ml_model(test_patches, model_path, "rf_")

    def evaluate_lgbm(self, test_patches, model_path):
        self._evaluate_ml_model(test_patches, model_path, "lgbm_")

    def _evaluate_pytorch_model(self, test_patches, model_path, get_model_fn, prefix):
        """Helper to evaluate PyTorch segmentation models."""
        dataset = PASTISDataset(test_patches, self.config, self.s2_dir, self.target_dir)
        
        sample_feat, _ = dataset[0]
        in_channels = sample_feat.shape[0]
        model = get_model_fn(self.config, in_channels).to(self.device)
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        
        y_true_all, y_pred_all = [], []
        ignore_index = 255
        
        print(f"Evaluating {prefix.strip('_').upper()}...")
        with torch.no_grad():
            for i in tqdm(range(len(dataset))):
                x, y = dataset[i]
                x = x.unsqueeze(0).to(self.device)
                
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
                import rasterio.features
                preds = rasterio.features.sieve(preds.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
                
                y = y.numpy()
                
                mask = y != ignore_index
                
                y_true_all.extend(y[mask])
                y_pred_all.extend(preds[mask])
                
        self.compute_and_save_metrics(y_true_all, y_pred_all, prefix=prefix)

    def evaluate_unet(self, test_patches, model_path):
        self._evaluate_pytorch_model(test_patches, model_path, get_unet_model, "unet_")

    def evaluate_resunet(self, test_patches, model_path):
        self._evaluate_pytorch_model(test_patches, model_path, get_resunet_model, "resunet_")

    def evaluate_transunet(self, test_patches, model_path):
        self._evaluate_pytorch_model(test_patches, model_path, get_transunet_model, "transunet_")

    def compute_and_save_metrics(self, y_true, y_pred, prefix=""):
        """Compute and save all classification metrics."""
        metrics_dir = os.path.join(self.config['paths']['output_dir'], 'metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        
        classes_dict = self.config['classes']
        
        labels_present = sorted(list(set(y_true) | set(y_pred)))
        target_names = [classes_dict.get(int(l), str(l)) for l in labels_present]
        
        oa = accuracy_score(y_true, y_pred)
        weighted_f1 = f1_score(y_true, y_pred, average='weighted')
        macro_f1 = f1_score(y_true, y_pred, average='macro')
        
        cm = confusion_matrix(y_true, y_pred, labels=labels_present)
        intersection = np.diag(cm)
        ground_truth_set = cm.sum(axis=1)
        predicted_set = cm.sum(axis=0)
        union = ground_truth_set + predicted_set - intersection
        iou = intersection / (union + 1e-8)
        miou = np.mean(iou)
        
        summary = {
            "overall_accuracy": float(oa),
            "weighted_f1": float(weighted_f1),
            "macro_f1": float(macro_f1),
            "miou": float(miou)
        }
        
        with open(os.path.join(metrics_dir, f"{prefix}summary_metrics.json"), 'w') as f:
            json.dump(summary, f, indent=4)
            
        clf_report = classification_report(y_true, y_pred, labels=labels_present, target_names=target_names, output_dict=True)
        
        for i, target_name in enumerate(target_names):
            clf_report[target_name]['iou'] = float(iou[i])
            
        with open(os.path.join(metrics_dir, f"{prefix}classification_report.json"), 'w') as f:
            json.dump(clf_report, f, indent=4)
            
        df_metrics = pd.DataFrame(clf_report).transpose()
        df_metrics.to_csv(os.path.join(metrics_dir, f"{prefix}per_class_metrics.csv"))
        
        df_cm = pd.DataFrame(cm, index=target_names, columns=target_names)
        df_cm.to_csv(os.path.join(metrics_dir, f"{prefix}confusion_matrix.csv"))
        
        print(f"Metrics saved to {metrics_dir} with prefix '{prefix}'")
        print(f"Overall Accuracy: {oa:.4f} | mIoU: {miou:.4f} | Weighted F1: {weighted_f1:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate crop classification models.")
    parser.add_argument("--model", type=str, choices=["rf", "lgbm", "unet", "resunet", "transunet", "all"], default="all",
                        help="Which model to evaluate")
    args = parser.parse_args()

    evaluator = ModelEvaluator()
    config = evaluator.config
    splits_dir = config['paths']['splits_dir']
    
    with open(os.path.join(splits_dir, "test_patches.txt"), "r") as f:
        test_patches = [int(line.strip()) for line in f.readlines()]
        
    model_dir = config['paths']['model_dir']
    
    if args.model in ["rf", "all"]:
        rf_model_path = os.path.join(model_dir, "rf_model.pkl")
        if os.path.exists(rf_model_path):
            print("\nEvaluating Random Forest...")
            evaluator.evaluate_rf(test_patches, rf_model_path)
        else:
            print(f"\nRF model not found at {rf_model_path}")

    if args.model in ["lgbm", "all"]:
        lgbm_model_path = os.path.join(model_dir, "lgbm_model.pkl")
        if os.path.exists(lgbm_model_path):
            print("\nEvaluating LightGBM...")
            evaluator.evaluate_lgbm(test_patches, lgbm_model_path)
        else:
            print(f"\nLightGBM model not found at {lgbm_model_path}")
            
    if args.model in ["unet", "all"]:
        unet_model_path = os.path.join(model_dir, "unet_best.pth")
        if os.path.exists(unet_model_path):
            print("\nEvaluating U-Net...")
            evaluator.evaluate_unet(test_patches, unet_model_path)
        else:
            print(f"\nU-Net model not found at {unet_model_path}")

    if args.model in ["resunet", "all"]:
        resunet_model_path = os.path.join(model_dir, "resunet_best.pth")
        if os.path.exists(resunet_model_path):
            print("\nEvaluating Res-UNet...")
            evaluator.evaluate_resunet(test_patches, resunet_model_path)
        else:
            print(f"\nRes-UNet model not found at {resunet_model_path}")

    if args.model in ["transunet", "all"]:
        transunet_model_path = os.path.join(model_dir, "transunet_best.pth")
        if os.path.exists(transunet_model_path):
            print("\nEvaluating TransUNet...")
            evaluator.evaluate_transunet(test_patches, transunet_model_path)
        else:
            print(f"\nTransUNet model not found at {transunet_model_path}")
