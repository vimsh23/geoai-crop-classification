import numpy as np

class DataPreprocessor:
    """Handles spectral index computation and feature extraction."""
    
    def __init__(self, config):
        self.config = config

    def compute_spectral_indices(self, s2_data):
        """
        Compute spectral indices like NDVI, NDWI from Sentinel-2 data.
        s2_data: (T, C, H, W) where C is the bands 0-9
        returns: (T, num_indices, H, W)
        """
        bands = self.config['dataset']['bands']
        
        indices_data = []
        
        eps = 1e-8
        b3 = s2_data[:, 1, :, :].astype(np.float32)
        b4 = s2_data[:, 2, :, :].astype(np.float32)
        b8 = s2_data[:, 6, :, :].astype(np.float32)
        
        for idx_name in self.config['preprocessing']['spectral_indices']:
            if idx_name == "NDVI":
                ndvi = (b8 - b4) / (b8 + b4 + eps)
                indices_data.append(ndvi)
            elif idx_name == "NDWI":
                ndwi = (b3 - b8) / (b3 + b8 + eps)
                indices_data.append(ndwi)
                
        if not indices_data:
            return np.empty((s2_data.shape[0], 0, s2_data.shape[2], s2_data.shape[3]))
            
        return np.stack(indices_data, axis=1)

    def extract_temporal_features(self, s2_data, indices_data):
        """
        Aggregate temporal data using statistical measures for Random Forest.
        s2_data: (T, C, H, W)
        indices_data: (T, C_idx, H, W)
        
        Returns: (C_total * num_stats, H, W)
        """
        if indices_data.shape[1] > 0:
            combined_data = np.concatenate([s2_data, indices_data], axis=1) 
        else:
            combined_data = s2_data
            
        features = []
        for stat in self.config['preprocessing']['temporal_aggregation']:
            if stat == "mean":
                features.append(np.mean(combined_data, axis=0))
            elif stat == "std":
                features.append(np.std(combined_data, axis=0))
            elif stat == "min":
                features.append(np.min(combined_data, axis=0))
            elif stat == "max":
                features.append(np.max(combined_data, axis=0))
            elif stat == "median":
                features.append(np.median(combined_data, axis=0))
                
        return np.concatenate(features, axis=0)

    def normalize_features(self, features):
        """Normalize features (e.g., standard scaling)."""
        norm_type = self.config['preprocessing']['normalization']
        
        if norm_type == "standardize":
            mean = np.mean(features, axis=(1, 2), keepdims=True)
            std = np.std(features, axis=(1, 2), keepdims=True)
            return (features - mean) / (std + 1e-8)
        
        return features

    def prepare_patch_features(self, s2_data):
        """Complete preprocessing pipeline for a single patch."""
        if self.config['preprocessing']['clip_negatives']:
            s2_data = np.clip(s2_data, 0, None)
            
        indices_data = self.compute_spectral_indices(s2_data)
        features = self.extract_temporal_features(s2_data, indices_data)
        features = self.normalize_features(features)
        
        return features

    def flatten_for_rf(self, features, target):
        """
        Flatten features and target for scikit-learn models.
        Removes pixels with ignored classes.
        """
        C = features.shape[0]
        X = features.reshape(C, -1).T
        
        y = target.reshape(-1)
        
        ignore_classes = self.config.get('ignore_classes', [])
        mask = np.ones_like(y, dtype=bool)
        for ic in ignore_classes:
            mask &= (y != ic)
            
        return X[mask], y[mask]

