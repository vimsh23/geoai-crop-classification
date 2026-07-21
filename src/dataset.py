import os
import numpy as np
import torch
from torch.utils.data import Dataset
from .data_loading import DataManager
from .preprocessing import DataPreprocessor

class PASTISDataset(Dataset):
    """
    PyTorch Dataset for PASTIS Sentinel-2 data.
    Designed for U-Net training. Outputs temporally aggregated features.
    """
    def __init__(self, patch_ids, config, s2_dir, target_dir):
        self.patch_ids = patch_ids
        self.config = config
        self.s2_dir = s2_dir
        self.target_dir = target_dir
        self.ignore_classes = config.get('ignore_classes', [])
        
        # Initialize OOP components
        self.data_manager = DataManager()
        # Override paths to use the ones provided (or we can just let data_manager use its config)
        self.data_manager.config['paths']['s2_dir'] = s2_dir
        self.data_manager.config['paths']['annotations_dir'] = target_dir
        
        self.preprocessor = DataPreprocessor(config)

    def __len__(self):
        return len(self.patch_ids)

    def __getitem__(self, idx):
        pid = self.patch_ids[idx]
        
        # Load raw data
        s2_data, target_data = self.data_manager.load_patch_data(pid)
        
        # Preprocess to get (C, H, W) features
        features = self.preprocessor.prepare_patch_features(s2_data)
        
        # Convert to tensors
        # Target shape is (1, H, W), we return (H, W) for CrossEntropyLoss
        target_tensor = torch.tensor(target_data[0], dtype=torch.long)
        
        # Map ignored classes to an ignore index (e.g., 255) for loss computation
        for ic in self.ignore_classes:
            target_tensor[target_tensor == ic] = 255
            
        feature_tensor = torch.tensor(features, dtype=torch.float32)
        
        return feature_tensor, target_tensor
