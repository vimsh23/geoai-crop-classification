import os
import sys
import pickle
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loading import DataManager
from src.dataset import PASTISDataset
from src.model import get_unet_model, get_resunet_model, get_transunet_model
from src.preprocessing import DataPreprocessor

def get_rgb(s2_data, t_idx=20):
    # s2_data shape: (T, 10, H, W)
    # RGB bands are B4, B3, B2 which are indices 2, 1, 0 in standard Sentinel-2 10-band stack
    rgb = s2_data[t_idx, [2, 1, 0], :, :]
    # Normalize for visualization (percentile clip)
    rgb = np.clip(rgb / 3000.0, 0, 1)
    rgb = np.transpose(rgb, (1, 2, 0))
    return rgb

def visualize():
    data_manager = DataManager()
    config = data_manager.config
    preprocessor = DataPreprocessor(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load test patches
    with open(os.path.join(config['paths']['splits_dir'], "test_patches.txt"), "r") as f:
        test_patches = [int(line.strip()) for line in f.readlines()]
        
    # Take just the first 3 patches for visualization
    sample_patches = test_patches[:3]
    
    dataset = PASTISDataset(sample_patches, config, config['paths']['s2_dir'], config['paths']['annotations_dir'])
    
    sample_x, _ = dataset[0]
    in_channels = sample_x.shape[0]
    
    # Load Models
    model_dir = config['paths']['model_dir']
    
    # RF
    rf_model = None
    if os.path.exists(os.path.join(model_dir, "rf_model.pkl")):
        with open(os.path.join(model_dir, "rf_model.pkl"), 'rb') as f:
            rf_model = pickle.load(f)
            
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
        
    # Colormap
    tab20 = plt.get_cmap('tab20')
    colors = [tab20(i) for i in range(20)]
    colors[0] = (0, 0, 0, 1) # Background is black
    colors[19] = (1, 1, 1, 1) # Void is white
    cmap = ListedColormap(colors)
    
    out_dir = os.path.join(config['paths']['output_dir'], 'figures', 'predictions')
    os.makedirs(out_dir, exist_ok=True)
    
    for idx, pid in enumerate(sample_patches):
        s2_data, target_data = data_manager.load_patch_data(pid)
        rgb = get_rgb(s2_data, t_idx=20)
        
        gt = target_data[0].numpy() if torch.is_tensor(target_data) else target_data[0]
        
        # RF Pred
        rf_pred = np.zeros_like(gt)
        if rf_model:
            features = preprocessor.prepare_patch_features(s2_data)
            # Flatten to shape (H*W, features)
            H, W = gt.shape
            C = features.shape[0]
            feats_flat = features.reshape(C, -1).T
            preds = rf_model.predict(feats_flat)
            rf_pred = preds.reshape(H, W)
            
            import rasterio.features
            rf_pred = rasterio.features.sieve(rf_pred.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
            
        # LGBM Pred
        lgbm_pred = np.zeros_like(gt)
        if lgbm_model:
            features = preprocessor.prepare_patch_features(s2_data)
            H, W = gt.shape
            C = features.shape[0]
            feats_flat = features.reshape(C, -1).T
            preds = lgbm_model.predict(feats_flat)
            lgbm_pred = preds.reshape(H, W)
            
            import rasterio.features
            lgbm_pred = rasterio.features.sieve(lgbm_pred.astype(np.int16), size=25, connectivity=4).astype(np.uint8)
            
        # DL Preds
        x, _ = dataset[idx]
        x = x.unsqueeze(0).to(device)
        
        unet_pred = np.zeros_like(gt)
        if unet_model:
            with torch.no_grad():
                outputs = unet_model(x)
                unet_pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
        resunet_pred = np.zeros_like(gt)
        if resunet_model:
            with torch.no_grad():
                outputs = resunet_model(x)
                resunet_pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
        transunet_pred = np.zeros_like(gt)
        if transunet_model:
            with torch.no_grad():
                outputs = transunet_model(x)
                transunet_pred = torch.argmax(outputs, dim=1).squeeze(0).cpu().numpy()
                
        # Plot
        fig, axes = plt.subplots(1, 7, figsize=(28, 4))
        axes[0].imshow(rgb)
        axes[0].set_title(f"Patch {pid} (RGB t=20)")
        
        axes[1].imshow(gt, cmap=cmap, vmin=0, vmax=19)
        axes[1].set_title("Ground Truth")
        
        axes[2].imshow(rf_pred, cmap=cmap, vmin=0, vmax=19)
        axes[2].set_title("Random Forest")
        
        axes[3].imshow(lgbm_pred, cmap=cmap, vmin=0, vmax=19)
        axes[3].set_title("LightGBM")
        
        axes[4].imshow(unet_pred, cmap=cmap, vmin=0, vmax=19)
        axes[4].set_title("U-Net")
        
        axes[5].imshow(resunet_pred, cmap=cmap, vmin=0, vmax=19)
        axes[5].set_title("ResU-Net")
        
        axes[6].imshow(transunet_pred, cmap=cmap, vmin=0, vmax=19)
        axes[6].set_title("TransUNet")
        
        for ax in axes:
            ax.axis('off')
            
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"prediction_patch_{pid}.png")
        plt.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    visualize()
