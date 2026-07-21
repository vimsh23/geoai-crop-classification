import os
import argparse
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from typing import Dict, Any

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loading import DataManager

def convert_patch_to_geotiff(patch_id: int, config: Dict[str, Any], t_idx: int = 20, export_target: bool = False, use_median: bool = False):
    """
    Convert a Sentinel-2 numpy patch to a GeoTIFF.
    Can extract a specific time-step, create a median cloud-free composite, or export the labels.
    Uses the bounding box from metadata to approximate georeferencing.
    """
    data_manager = DataManager()
    metadata = data_manager.load_metadata()
    patch_meta = data_manager.get_patch_metadata(patch_id, metadata)
    
    s2_data, target_data = data_manager.load_patch_data(patch_id)
    
    # Get geometry (bounding box)
    coords = patch_meta['geometry']['coordinates'][0]
    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]
    
    min_x, max_x = min(lons), max(lons)
    min_y, max_y = min(lats), max(lats)
    
    output_dir = os.path.join(config['paths']['output_dir'], 'geotiffs')
    os.makedirs(output_dir, exist_ok=True)
    
    if export_target:
        # Target data shape is usually (3, 128, 128). We only need the first band (class labels).
        _, height, width = target_data.shape
        n_bands = 1
        img_data = target_data[0:1] # (1, 128, 128)
        output_path = os.path.join(output_dir, f"Target_{patch_id}.tif")
        band_names = ["Crop Class"]
    elif use_median:
        # Calculate the median across all 46 time steps to create a cloud-free image
        _, n_bands, height, width = s2_data.shape
        img_data = np.median(s2_data, axis=0) # (10, 128, 128)
        output_path = os.path.join(output_dir, f"S2_{patch_id}_median.tif")
        band_names = config['dataset']['bands']
    else:
        # Sentinel-2 data shape: (46, 10, 128, 128)
        _, n_bands, height, width = s2_data.shape
        img_data = s2_data[t_idx] # (10, 128, 128)
        output_path = os.path.join(output_dir, f"S2_{patch_id}_t{t_idx}.tif")
        band_names = config['dataset']['bands']
    
    # Create affine transform from bounds
    transform = from_bounds(min_x, min_y, max_x, max_y, width, height)
    
    # Define CRS
    crs = config['dataset']['crs']
    
    print(f"Writing {output_path}...")
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
            dst.set_band_description(i + 1, band_names[i])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PASTIS .npy to GeoTIFF for QGIS exploration.")
    parser.add_argument("--patch_id", type=int, help="Patch ID to convert. Required unless --all is set.")
    parser.add_argument("--t_idx", type=int, default=20, help="Time index to extract (0-45). Ignored if --target or --median is set.")
    parser.add_argument("--target", action="store_true", help="If set, exports the target labels instead of the satellite image.")
    parser.add_argument("--median", action="store_true", help="If set, computes the median across all 46 time steps to create a cloud-free composite.")
    parser.add_argument("--all", action="store_true", help="Convert ALL patches in the metadata.")
    args = parser.parse_args()
    
    data_manager = DataManager()
    config = data_manager.config
    
    if args.all:
        metadata = data_manager.load_metadata()
        print(f"Starting batch conversion of {len(metadata['features'])} patches...")
        for feat in metadata['features']:
            pid = int(feat['properties']['ID_PATCH'])
            try:
                convert_patch_to_geotiff(pid, config, args.t_idx, args.target, args.median)
            except FileNotFoundError:
                print(f"Skipping patch {pid}: Numpy array file not found.")
            except Exception as e:
                print(f"Failed to process patch {pid}: {e}")
        print("Batch conversion complete.")
    elif args.patch_id:
        convert_patch_to_geotiff(args.patch_id, config, args.t_idx, args.target, args.median)
    else:
        print("Error: You must specify either --patch_id or --all")
