"""
vectorize_labels.py
--------------------
Converts the merged label GeoTIFF (mosaic_label.tif) into a vector
polygon layer (GeoJSON + Shapefile) where each polygon has the actual
crop class name as an attribute.

Outputs:
  outputs/geotiff/merge/mosaic_label/crop_labels.geojson
  outputs/geotiff/merge/mosaic_label/crop_labels.shp
"""

import os
import sys
import numpy as np
import rasterio
from rasterio.features import shapes
import geopandas as gpd
from shapely.geometry import shape

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from src.data_loading import load_config

# ── Class mapping ─────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0:  "Background",
    1:  "Meadow",
    2:  "Soft Winter Wheat",
    3:  "Corn",
    4:  "Winter Barley",
    5:  "Winter Rapeseed",
    6:  "Spring Barley",
    7:  "Sunflower",
    8:  "Grapevine",
    9:  "Beet",
    10: "Winter Triticale",
    11: "Winter Durum Wheat",
    12: "Fruits/Vegetables/Flowers",
    13: "Potatoes",
    14: "Leguminous Fodder",
    15: "Soybeans",
    16: "Orchard",
    17: "Mixed Cereal",
    18: "Sorghum",
    19: "Void",
}

def vectorize_labels():
    config = load_config()
    merge_lbl_dir = os.path.join(project_root, 'outputs', 'geotiff', 'merge', 'mosaic_label')
    input_tif = os.path.join(merge_lbl_dir, 'mosaic_label.tif')

    if not os.path.exists(input_tif):
        print(f"ERROR: {input_tif} not found.")
        print("Please run export_and_merge_geotiffs.py first.")
        return

    print(f"Reading: {input_tif}")
    with rasterio.open(input_tif) as src:
        data = src.read(1)       # first band = class IDs (uint8 / int16)
        transform = src.transform
        crs = src.crs

    print(f"Raster shape: {data.shape}, dtype: {data.dtype}")
    print(f"Unique class IDs: {np.unique(data)}")

    # Mask out Background (0) and Void (19) if desired
    # Comment these lines if you want to keep them
    mask_out = {0}               # Background: skip
    valid_mask = ~np.isin(data, list(mask_out)).astype(np.uint8)

    print("Vectorizing pixels into polygons ...")
    polygons = []
    class_ids = []

    for geom, value in shapes(data.astype(np.int32), mask=valid_mask, transform=transform):
        cid = int(value)
        if cid == 0:
            continue
        polygons.append(shape(geom))
        class_ids.append(cid)

    if not polygons:
        print("No valid polygons found.")
        return

    print(f"Found {len(polygons):,} polygons before dissolve.")

    gdf = gpd.GeoDataFrame(
        {
            "class_id":   class_ids,
            "crop_name":  [CLASS_NAMES.get(c, f"Class_{c}") for c in class_ids],
        },
        geometry=polygons,
        crs=crs,
    )

    # Optional: dissolve by crop type to reduce number of tiny polygons
    print("Dissolving by crop class ...")
    gdf_dissolved = gdf.dissolve(by="crop_name", as_index=False)
    gdf_dissolved["class_id"] = gdf_dissolved["crop_name"].map(
        {v: k for k, v in CLASS_NAMES.items()}
    )

    # ── Save GeoJSON ────────────────────────────────────────────────────────
    geojson_out = os.path.join(merge_lbl_dir, "crop_labels.geojson")
    gdf_dissolved.to_file(geojson_out, driver="GeoJSON")
    print(f"GeoJSON saved: {geojson_out}")

    # ── Save Shapefile ───────────────────────────────────────────────────────
    shp_dir = os.path.join(merge_lbl_dir, "crop_labels_shp")
    os.makedirs(shp_dir, exist_ok=True)
    shp_out = os.path.join(shp_dir, "crop_labels.shp")
    gdf_dissolved.to_file(shp_out, driver="ESRI Shapefile")
    print(f"Shapefile saved: {shp_out}")

    print("\n── Crop Class Summary ──────────────────")
    for _, row in gdf_dissolved.iterrows():
        print(f"  [{int(row['class_id']):>2}] {row['crop_name']}")

    print("\nDone!")

if __name__ == "__main__":
    vectorize_labels()
