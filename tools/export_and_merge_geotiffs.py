import os
import sys
import glob
import json
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.merge import merge
from tqdm import tqdm

# Add the project root to python path so we can import src
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loading import DataManager

def export_to_geotiff(img_data, output_path, min_x, min_y, max_x, max_y, crs, band_names=None):
    n_bands, height, width = img_data.shape
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    with rasterio.open(
        output_path,
        'w',
        driver='GTiff',
        height=height,
        width=width,
        count=n_bands,
        dtype=img_data.dtype,
        crs=crs,
        transform=transform,
    ) as dst:
        for i in range(n_bands):
            dst.write(img_data[i], i + 1)
            if band_names:
                dst.set_band_description(i + 1, band_names[i])

def main():
    data_manager = DataManager()
    config = data_manager.config
    metadata = data_manager.load_metadata()
    
    # Setup directories
    base_out = os.path.join(project_root, 'outputs', 'geotiff')
    images_dir = os.path.join(base_out, 'images')
    labels_dir = os.path.join(base_out, 'labels')
    merge_img_dir = os.path.join(base_out, 'merge', 'mosaic_image')
    merge_lbl_dir = os.path.join(base_out, 'merge', 'mosaic_label')
    
    for d in [images_dir, labels_dir, merge_img_dir, merge_lbl_dir]:
        os.makedirs(d, exist_ok=True)
        
    crs = config['dataset']['crs']
    band_names = config['dataset']['bands']
    
    print("STEP 1: Exporting individual GeoTIFFs...")
    all_dates = set()
    
    for feat in tqdm(metadata['features'], desc="Exporting Patches"):
        pid = int(feat['properties']['ID_PATCH'])
        dates_dict = feat['properties'].get('dates-S2', {})
        
        # Get geometry (bounding box)
        coords = feat['geometry']['coordinates'][0]
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        min_x, max_x = min(lons), max(lons)
        min_y, max_y = min(lats), max(lats)
        
        try:
            s2_data, target_data = data_manager.load_patch_data(pid)
        except Exception as e:
            print(f"Skipping patch {pid}: {e}")
            continue
            
        # Export Label
        label_out = os.path.join(labels_dir, f"Target_{pid}.tif")
        if not os.path.exists(label_out):
            export_to_geotiff(target_data[0:1], label_out, min_x, min_y, max_x, max_y, crs, ["Crop Class"])
            
        # Export Images
        for t in range(s2_data.shape[0]):
            date_str = str(dates_dict.get(str(t), f"t{t:02d}"))
            all_dates.add(date_str)
            
            img_out = os.path.join(images_dir, f"{date_str}_{pid}.tif")
            if not os.path.exists(img_out):
                export_to_geotiff(s2_data[t], img_out, min_x, min_y, max_x, max_y, crs, band_names)
                
    print(f"\nSTEP 2: Merging Labels...")
    label_files = glob.glob(os.path.join(labels_dir, "*.tif"))
    if label_files:
        out_path = os.path.join(merge_lbl_dir, "mosaic_label.tif")
        if not os.path.exists(out_path):
            print(f"Merging {len(label_files)} label files...")
            try:
                mosaic, out_trans = merge(label_files)
                with rasterio.open(label_files[0]) as src:
                    out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans
                })
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                print(f"Merge successful. Cleaning up {len(label_files)} individual label files...")
                for f in label_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            except Exception as e:
                print(f"Failed to merge labels: {e}")
                
    print(f"\nSTEP 3: Merging Images by Date...")
    for date_str in tqdm(sorted(list(all_dates)), desc="Mosaicing Dates"):
        date_files = glob.glob(os.path.join(images_dir, f"{date_str}_*.tif"))
        if not date_files: continue
        
        out_path = os.path.join(merge_img_dir, f"mosaic_image_{date_str}.tif")
        if not os.path.exists(out_path):
            try:
                mosaic, out_trans = merge(date_files)
                with rasterio.open(date_files[0]) as src:
                    out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": mosaic.shape[1],
                    "width": mosaic.shape[2],
                    "transform": out_trans
                })
                with rasterio.open(out_path, "w", **out_meta) as dest:
                    dest.write(mosaic)
                print(f"Cleaning up {len(date_files)} individual image files for {date_str}...")
                for f in date_files:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
            except Exception as e:
                print(f"Failed to merge images for date {date_str}: {e}")
                
    print(f"\nDone! All files saved in {base_out}")

if __name__ == "__main__":
    main()
