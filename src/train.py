import os
import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loading import DataManager
from src.preprocessing import DataPreprocessor
from src.dataset import PASTISDataset
from src.model import get_rf_model, get_lgbm_model, get_unet_model, get_resunet_model, get_transunet_model

class ModelTrainer:
    """Handles the training pipeline for various crop classification models."""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.config = self.data_manager.config
        self.preprocessor = DataPreprocessor(self.config)
        self.s2_dir = self.config['paths']['s2_dir']
        self.target_dir = self.config['paths']['annotations_dir']
        self.model_dir = self.config['paths']['model_dir']
        os.makedirs(self.model_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def create_splits(self):
        """Create train/val/test splits based on folds and save to txt files."""
        metadata = self.data_manager.load_metadata()
        train_patches, val_patches, test_patches = [], [], []
        
        train_folds = self.config['split']['train_folds']
        val_folds = self.config['split']['val_folds']
        test_folds = self.config['split']['test_folds']
        
        for feat in metadata['features']:
            pid = int(feat['properties']['ID_PATCH'])
            fold = int(feat['properties']['Fold'])
            
            if fold in train_folds:
                train_patches.append(pid)
            elif fold in val_folds:
                val_patches.append(pid)
            elif fold in test_folds:
                test_patches.append(pid)
                
        splits_dir = self.config['paths']['splits_dir']
        os.makedirs(splits_dir, exist_ok=True)
        
        with open(os.path.join(splits_dir, "train_patches.txt"), "w") as f:
            f.write("\n".join(map(str, train_patches)))
        with open(os.path.join(splits_dir, "val_patches.txt"), "w") as f:
            f.write("\n".join(map(str, val_patches)))
        with open(os.path.join(splits_dir, "test_patches.txt"), "w") as f:
            f.write("\n".join(map(str, test_patches)))
            
        print(f"Splits created: Train={len(train_patches)}, Val={len(val_patches)}, Test={len(test_patches)}")
        return train_patches, val_patches, test_patches

    def _extract_ml_features(self, train_patches, max_pixels_per_patch=10000):
        """Helper to extract features for classical ML models (RF, LightGBM)."""
        X_train_list, y_train_list = [], []
        
        for pid in tqdm(train_patches):
            s2_data, target_data = self.data_manager.load_patch_data(pid)
            features = self.preprocessor.prepare_patch_features(s2_data)
            
            X_patch, y_patch = self.preprocessor.flatten_for_rf(features, target_data)
            
            if len(y_patch) > max_pixels_per_patch:
                idx = np.random.choice(len(y_patch), max_pixels_per_patch, replace=False)
                X_patch, y_patch = X_patch[idx], y_patch[idx]
                
            X_train_list.append(X_patch)
            y_train_list.append(y_patch)
            
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        return X_train, y_train

    def train_random_forest(self, train_patches):
        """Train Random Forest model."""
        print("Extracting features for Random Forest training...")
        X_train, y_train = self._extract_ml_features(train_patches)
        
        print(f"Training Random Forest on {len(X_train)} samples with {X_train.shape[1]} features...")
        model = get_rf_model(self.config)
        model.fit(X_train, y_train)
        
        model_path = os.path.join(self.model_dir, "rf_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"Random Forest saved to {model_path}")
        return model

    def train_lgbm(self, train_patches):
        """Train LightGBM model."""
        print("Extracting features for LightGBM training...")
        X_train, y_train = self._extract_ml_features(train_patches)
        
        print(f"Training LightGBM on {len(X_train)} samples with {X_train.shape[1]} features...")
        model = get_lgbm_model(self.config)
        model.fit(X_train, y_train)
        
        model_path = os.path.join(self.model_dir, "lgbm_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
            
        print(f"LightGBM saved to {model_path}")
        return model

    def _train_pytorch_model(self, model_name, get_model_fn, train_patches, val_patches, save_name):
        """Helper to train any PyTorch based segmentation model."""
        print(f"Training {model_name} on {self.device}")
        
        train_dataset = PASTISDataset(train_patches, self.config, self.s2_dir, self.target_dir)
        val_dataset = PASTISDataset(val_patches, self.config, self.s2_dir, self.target_dir)
        
        batch_size = self.config['models']['unet'].get('batch_size', 8)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        sample_feat, _ = train_dataset[0]
        in_channels = sample_feat.shape[0]
        
        model = get_model_fn(self.config, in_channels).to(self.device)
        
        criterion = nn.CrossEntropyLoss(ignore_index=255)
        optimizer = optim.Adam(model.parameters(), lr=self.config['models']['unet'].get('learning_rate', 1e-3))
        epochs = self.config['models']['unet'].get('epochs', 10)
        
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0.0
            for x, y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]"):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x, y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]"):
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = model(x)
                    loss = criterion(outputs, y)
                    val_loss += loss.item()
                    
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), os.path.join(self.model_dir, save_name))
                print(f"Saved new best {model_name} model.")

    def train_unet(self, train_patches, val_patches):
        """Train the U-Net model."""
        self._train_pytorch_model("U-Net", get_unet_model, train_patches, val_patches, "unet_best.pth")

    def train_resunet(self, train_patches, val_patches):
        """Train the advanced Res-UNet model."""
        self._train_pytorch_model("Res-UNet", get_resunet_model, train_patches, val_patches, "resunet_best.pth")

    def train_transunet(self, train_patches, val_patches):
        """Train the Transformer-based TransUNet model."""
        self._train_pytorch_model("TransUNet", get_transunet_model, train_patches, val_patches, "transunet_best.pth")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train crop classification models.")
    parser.add_argument("--model", type=str, choices=["rf", "lgbm", "unet", "resunet", "transunet", "all"], default="all",
                        help="Which model to train (rf, lgbm, unet, resunet, transunet, or all)")
    args = parser.parse_args()

    trainer = ModelTrainer()
    train_p, val_p, test_p = trainer.create_splits()
    
    if args.model in ["rf", "all"]:
        print("\n--- Training Random Forest ---")
        trainer.train_random_forest(train_p)
        
    if args.model in ["lgbm", "all"]:
        print("\n--- Training LightGBM ---")
        trainer.train_lgbm(train_p)
        
    if args.model in ["unet", "all"]:
        print("\n--- Training U-Net ---")
        trainer.train_unet(train_p, val_p)

    if args.model in ["resunet", "all"]:
        print("\n--- Training Res-UNet ---")
        trainer.train_resunet(train_p, val_p)

    if args.model in ["transunet", "all"]:
        print("\n--- Training TransUNet ---")
        trainer.train_transunet(train_p, val_p)
