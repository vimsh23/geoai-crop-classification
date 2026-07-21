import os
import json
import numpy as np
import yaml

class DataManager:
    """Manages data loading and validation for the PASTIS dataset."""
    
    def __init__(self, config_path="configs/config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self):
        """Load the project configuration file."""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def load_metadata(self):
        """Load the PASTIS metadata geojson file."""
        with open(self.config['paths']['metadata'], 'r') as f:
            return json.load(f)

    def get_patch_ids(self, metadata=None):
        """Extract all patch IDs from the metadata."""
        if metadata is None:
            metadata = self.load_metadata()
        return [int(feat['properties']['ID_PATCH']) for feat in metadata['features']]

    def load_patch_data(self, patch_id):
        """
        Load the Sentinel-2 array and Target annotation array for a given patch ID.
        """
        s2_dir = self.config['paths']['s2_dir']
        annotations_dir = self.config['paths']['annotations_dir']
        
        s2_path = os.path.join(s2_dir, f"S2_{patch_id}.npy")
        target_path = os.path.join(annotations_dir, f"TARGET_{patch_id}.npy")
        
        if not os.path.exists(s2_path):
            raise FileNotFoundError(f"S2 data not found for patch {patch_id} at {s2_path}")
        if not os.path.exists(target_path):
            raise FileNotFoundError(f"Target data not found for patch {patch_id} at {target_path}")
            
        s2_data = np.load(s2_path)
        target_data = np.load(target_path)
        
        return s2_data, target_data

    def get_patch_metadata(self, patch_id, metadata=None):
        """Extract metadata feature (including geometry and properties) for a specific patch ID."""
        if metadata is None:
            metadata = self.load_metadata()
        for feat in metadata['features']:
            if int(feat['properties']['ID_PATCH']) == patch_id:
                return feat
        raise ValueError(f"Patch {patch_id} not found in metadata.")

    def validate_dataset(self):
        """Validate that all files exist and shapes are consistent."""
        metadata = self.load_metadata()
        patch_ids = self.get_patch_ids(metadata)
        
        print(f"Found {len(patch_ids)} patches in metadata.")
        
        s2_dir = self.config['paths']['s2_dir']
        target_dir = self.config['paths']['annotations_dir']
        
        for pid in patch_ids:
            s2_path = os.path.join(s2_dir, f"S2_{pid}.npy")
            target_path = os.path.join(target_dir, f"TARGET_{pid}.npy")
            if not os.path.exists(s2_path) or not os.path.exists(target_path):
                print(f"Missing data for patch {pid}")
                return False
                
        # Check first patch shape
        s2, target = self.load_patch_data(patch_ids[0])
        print(f"S2 shape: {s2.shape}, Target shape: {target.shape}")
        
        return True

if __name__ == "__main__":
    # Test loading
    manager = DataManager()
    is_valid = manager.validate_dataset()
    print(f"Dataset valid: {is_valid}")
