import os
import sys
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'app'))

from src.data_loading import DataManager
from app.components.common import make_rgb, CROP_COLORS

def load_tif(path):
    if not os.path.exists(path):
        return np.zeros((128, 128), dtype=np.uint8)
    with rasterio.open(path) as src:
        return src.read(1)

def generate_slides():
    data_manager = DataManager()
    config = data_manager.config
    
    with open(os.path.join(config['paths']['splits_dir'], "test_patches.txt"), "r") as f:
        test_patches = [int(line.strip()) for line in f.readlines()]
        
    out_dir = os.path.join(config['paths']['output_dir'], 'geotiff', 'predictions')
    slides_dir = os.path.join(config['paths']['output_dir'], 'slides')
    os.makedirs(slides_dir, exist_ok=True)
    
    cmap = ListedColormap([CROP_COLORS.get(i, '#000000') for i in range(20)])
    
    col_titles = [
        "RGB", "GT", "RF (Sieve)", "RF (OBIA)", 
        "RF CF (Sieve)", "RF CF (OBIA)",
        "LGBM (Sieve)", "LGBM (OBIA)", 
        "U-Net", "ResU-Net", "TransUNet"
    ]
    
    print(f"Generating {len(test_patches)} individual slide images in {slides_dir}...")
    
    for pid in tqdm(test_patches):
        patch_dir = os.path.join(out_dir, str(pid))
        
        # Load S2 for RGB and Ground truth
        s2_data, target_data = data_manager.load_patch_data(pid)
        rgb = make_rgb(s2_data.numpy() if hasattr(s2_data, 'numpy') else s2_data)
        gt = target_data.numpy().squeeze() if hasattr(target_data, 'numpy') else np.array(target_data).squeeze()
        
        # Load all predictions including Sieve and OBIA
        rf_sieve = load_tif(os.path.join(patch_dir, f"rf_pred_sieve_{pid}.tif"))
        rf_obia = load_tif(os.path.join(patch_dir, f"rf_pred_obia_{pid}.tif"))
        rf_cf_sieve = load_tif(os.path.join(patch_dir, f"rf_cf_pred_sieve_{pid}.tif"))
        rf_cf_obia = load_tif(os.path.join(patch_dir, f"rf_cf_pred_obia_{pid}.tif"))
        lgbm_sieve = load_tif(os.path.join(patch_dir, f"lgbm_pred_sieve_{pid}.tif"))
        lgbm_obia = load_tif(os.path.join(patch_dir, f"lgbm_pred_obia_{pid}.tif"))
        unet = load_tif(os.path.join(patch_dir, f"unet_pred_{pid}.tif"))
        resunet = load_tif(os.path.join(patch_dir, f"resunet_pred_{pid}.tif"))
        transunet = load_tif(os.path.join(patch_dir, f"transunet_pred_{pid}.tif"))
        
        imgs = [rgb, gt, rf_sieve, rf_obia, rf_cf_sieve, rf_cf_obia, lgbm_sieve, lgbm_obia, unet, resunet, transunet]
        
        # Create a wide 1x11 figure perfect for a presentation slide
        fig, axes = plt.subplots(1, 11, figsize=(30, 3), facecolor='white')
        
        for j, img in enumerate(imgs):
            ax = axes[j]
            if j == 0:
                ax.imshow(img)
            else:
                ax.imshow(img, cmap=cmap, vmin=0, vmax=19)
                
            ax.axis('off')
            ax.set_title(col_titles[j], fontsize=14, fontweight='bold', pad=10)
                
        plt.tight_layout()
        save_path = os.path.join(slides_dir, f"slide_patch_{pid}.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
    print("All slides successfully saved!")

if __name__ == "__main__":
    generate_slides()
