import os
import sys
import pickle
import torch
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from tqdm import tqdm
from skimage.segmentation import slic
from scipy import stats
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loading import DataManager
from src.dataset import PASTISDataset
from src.model import get_unet_model, get_resunet_model, get_transunet_model
from src.preprocessing import DataPreprocessor
from src.preprocessing_cloud_removal import DataPreprocessor as DataPreprocessorCloudFree

def export_geotiff(img_data, output_path, min_x, min_y, max_x, max_y, crs):
    # For a single band (prediction class)
    height, width = img_data.shape
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=1,
        dtype=np.uint8,
        crs=crs,
        transform=transform,
    ) as dst:
        dst.write(img_data.astype(np.uint8), 1)

def apply_obia(pixel_preds, s2_data):
    # Create RGB for segmentation
    img = np.median(s2_data, axis=0)
    rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(np.float32)
    p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
    rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
    
    segments = slic(rgb, n_segments=300, compactness=10, sigma=1, start_label=1)
    obia_preds = np.zeros_like(pixel_preds)
    
    for seg_id in np.unique(segments):
        mask = (segments == seg_id)
        segment_pixels = pixel_preds[mask]
        if len(segment_pixels) > 0:
            majority_class = stats.mode(segment_pixels, keepdims=False).mode
            obia_preds[mask] = majority_class
            
    return obia_preds

def export_predictions(use_obia=False):
    data_manager = DataManager()
    config = data_manager.config
    preprocessor = DataPreprocessor(config)
    
    # Configure cloud-free preprocessor
    cf_config = config.copy()
    if 'preprocessing' not in cf_config:
        cf_config['preprocessing'] = {}
    cf_config['preprocessing']['cloud_threshold'] = 2500
    cf_config['preprocessing']['shadow_threshold'] = 800
    preprocessor_cf = DataPreprocessorCloudFree(cf_config)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = data_manager.load_metadata()
    crs = config['dataset']['crs']
    
    # Load test patches
    with open(os.path.join(config['paths']['splits_dir'], "test_patches.txt"), "r") as f:
        test_patches = [int(line.strip()) for line in f.readlines()]
        
    out_dir = os.path.join(config['paths']['output_dir'], 'geotiff', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    dataset = PASTISDataset(test_patches, config, config['paths']['s2_dir'], config['paths']['annotations_dir'])
    
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    
    # Load Models
    model_dir = config['paths']['model_dir']
    
    # RF
    rf_model = None
    if os.path.exists(os.path.join(model_dir, "rf_model.pkl")):
        with open(os.path.join(model_dir, "rf_model.pkl"), 'rb') as f:
            rf_model = pickle.load(f)
            
    # RF Cloud-Free
    rf_cf_model = None
    if os.path.exists(os.path.join(model_dir, "rf_model_cloud_free.pkl")):
        with open(os.path.join(model_dir, "rf_model_cloud_free.pkl"), 'rb') as f:
            rf_cf_model = pickle.load(f)
            
    # LightGBM
    lgbm_model = None
    if os.path.exists(os.path.join(model_dir, "lgbm_model.pkl")):
        with open(os.path.join(model_dir, "lgbm_model.pkl"), 'rb') as f:
            lgbm_model = pickle.load(f)
            
    # UNet
    unet_model = None
    if os.path.exists(os.path.join(model_dir, "unet_best.pth")):
        unet_model = get_unet_model(config, in_channels).to(device)
        unet_model.load_state_dict(torch.load(os.path.join(model_dir, "unet_best.pth"), map_location=device))
        unet_model.eval()
        
    # ResUNet
    resunet_model = None
    if os.path.exists(os.path.join(model_dir, "resunet_best.pth")):
        resunet_model = get_resunet_model(config, in_channels).to(device)
        resunet_model.load_state_dict(torch.load(os.path.join(model_dir, "resunet_best.pth"), map_location=device))
        resunet_model.eval()
        
    # TransUNet
    transunet_model = None
    if os.path.exists(os.path.join(model_dir, "transunet_best.pth")):
        transunet_model = get_transunet_model(config, in_channels).to(device)
        transunet_model.load_state_dict(torch.load(os.path.join(model_dir, "transunet_best.pth"), map_location=device))
        transunet_model.eval()
        
    print(f"Exporting predictions for {len(test_patches)} patches as GeoTIFF...")
    
    import shutil
    qml_src = os.path.join(config['paths']['output_dir'], 'geotiff', 'merge', 'mosaic_label', 'mosaic_label.qml')
    has_qml = os.path.exists(qml_src)
    
    for idx, pid in enumerate(tqdm(test_patches)):
        # Find bounds from metadata
        feat = next((f for f in metadata['features'] if int(f['properties']['ID_PATCH']) == pid), None)
        if not feat: continue
        
        coords = feat['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_x, max_x = min(lons), max(lons)
        min_y, max_y = min(lats), max(lats)
        
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)
        
        patch_dir = os.path.join(out_dir, str(pid))
        os.makedirs(patch_dir, exist_ok=True)
        
        # RF
        if rf_model:
            s2_data, _ = data_manager.load_patch_data(pid)
            features = preprocessor.prepare_patch_features(s2_data)
            C = features.shape[0]
            feats_flat = features.reshape(C, -1).T
            preds = rf_model.predict(feats_flat)
            pred = preds.reshape(128, 128)
            
            import rasterio.features
            pred_sieve = rasterio.features.sieve(pred.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
            out_path_sieve = os.path.join(patch_dir, f"rf_pred_sieve_{pid}.tif")
            export_geotiff(pred_sieve, out_path_sieve, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"rf_pred_sieve_{pid}.qml"))
                
            if use_obia:
                pred_obia = apply_obia(pred, s2_data)
                out_path_obia = os.path.join(patch_dir, f"rf_pred_obia_{pid}.tif")
                export_geotiff(pred_obia, out_path_obia, min_x, min_y, max_x, max_y, crs)
                if has_qml:
                    shutil.copy(qml_src, os.path.join(patch_dir, f"rf_pred_obia_{pid}.qml"))
                    
        # RF Cloud-Free
        if rf_cf_model:
            s2_data, _ = data_manager.load_patch_data(pid)
            features = preprocessor_cf.prepare_patch_features(s2_data)
            C = features.shape[0]
            feats_flat = features.reshape(C, -1).T
            preds = rf_cf_model.predict(feats_flat)
            pred = preds.reshape(128, 128)
            
            import rasterio.features
            pred_sieve = rasterio.features.sieve(pred.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
            out_path_sieve = os.path.join(patch_dir, f"rf_cf_pred_sieve_{pid}.tif")
            export_geotiff(pred_sieve, out_path_sieve, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"rf_cf_pred_sieve_{pid}.qml"))
                
            if use_obia:
                pred_obia = apply_obia(pred, s2_data)
                out_path_obia = os.path.join(patch_dir, f"rf_cf_pred_obia_{pid}.tif")
                export_geotiff(pred_obia, out_path_obia, min_x, min_y, max_x, max_y, crs)
                if has_qml:
                    shutil.copy(qml_src, os.path.join(patch_dir, f"rf_cf_pred_obia_{pid}.qml"))
                
        # LightGBM
        if lgbm_model:
            s2_data, _ = data_manager.load_patch_data(pid)
            features = preprocessor.prepare_patch_features(s2_data)
            C = features.shape[0]
            feats_flat = features.reshape(C, -1).T
            preds = lgbm_model.predict(feats_flat)
            pred = preds.reshape(128, 128)
            
            import rasterio.features
            pred_sieve = rasterio.features.sieve(pred.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
            out_path_sieve = os.path.join(patch_dir, f"lgbm_pred_sieve_{pid}.tif")
            export_geotiff(pred_sieve, out_path_sieve, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"lgbm_pred_sieve_{pid}.qml"))
                
            if use_obia:
                pred_obia = apply_obia(pred, s2_data)
                out_path_obia = os.path.join(patch_dir, f"lgbm_pred_obia_{pid}.tif")
                export_geotiff(pred_obia, out_path_obia, min_x, min_y, max_x, max_y, crs)
                if has_qml:
                    shutil.copy(qml_src, os.path.join(patch_dir, f"lgbm_pred_obia_{pid}.qml"))
        
        # UNet
        if unet_model:
            with torch.no_grad():
                outputs = unet_model(x)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            
            out_path = os.path.join(patch_dir, f"unet_pred_{pid}.tif")
            export_geotiff(pred, out_path, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"unet_pred_{pid}.qml"))
            
        # ResUNet
        if resunet_model:
            with torch.no_grad():
                outputs = resunet_model(x)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
            out_path = os.path.join(patch_dir, f"resunet_pred_{pid}.tif")
            export_geotiff(pred, out_path, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"resunet_pred_{pid}.qml"))
                
        # TransUNet
        if transunet_model:
            with torch.no_grad():
                outputs = transunet_model(x)
                pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
                
            out_path = os.path.join(patch_dir, f"transunet_pred_{pid}.tif")
            export_geotiff(pred, out_path, min_x, min_y, max_x, max_y, crs)
            if has_qml:
                shutil.copy(qml_src, os.path.join(patch_dir, f"transunet_pred_{pid}.qml"))
            
    print(f"\nSaved GeoTIFF predictions (and style files) to {out_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--obia", action="store_true", help="Apply OBIA (Object-Based Image Analysis) smoothing")
    args = parser.parse_args()
    
    export_predictions(use_obia=args.obia)
