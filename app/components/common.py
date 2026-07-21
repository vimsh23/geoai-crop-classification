"""
common.py — Shared utilities, CSS, color palettes, and caching for the CropAI Explorer.
"""
import os, sys, json, yaml, numpy as np
import streamlit as st

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, BASE)

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

# ── Crop class color palette (matching QGIS/QML exactly) ─────────────────
_tab20 = plt.get_cmap('tab20')
CROP_COLORS = {i: to_hex(_tab20(i)) for i in range(20)}
CROP_COLORS[0] = "#000000"  # Background
CROP_COLORS[19] = "#ffffff" # Void

# ── Sentinel-2 band metadata ────────────────────────────────────────────────
BAND_INFO = {
    0: {"name":"B2",  "label":"Blue",           "wavelength":"490 nm",  "resolution":"10 m", "color":"#2196F3",
        "desc":"Sensitive to water bodies, soil/vegetation discrimination, and atmospheric scattering."},
    1: {"name":"B3",  "label":"Green",          "wavelength":"560 nm",  "resolution":"10 m", "color":"#4CAF50",
        "desc":"Peak reflectance for healthy vegetation. Used in NDWI and true-color composites."},
    2: {"name":"B4",  "label":"Red",            "wavelength":"665 nm",  "resolution":"10 m", "color":"#F44336",
        "desc":"Strong chlorophyll absorption. Critical for NDVI computation and vegetation health."},
    3: {"name":"B5",  "label":"Red Edge 1",     "wavelength":"705 nm",  "resolution":"20 m", "color":"#E91E63",
        "desc":"Transition zone between red absorption and NIR reflectance. Sensitive to chlorophyll content."},
    4: {"name":"B6",  "label":"Red Edge 2",     "wavelength":"740 nm",  "resolution":"20 m", "color":"#9C27B0",
        "desc":"Captures vegetation stress and canopy structure variations at the red-edge inflection point."},
    5: {"name":"B7",  "label":"Red Edge 3",     "wavelength":"783 nm",  "resolution":"20 m", "color":"#673AB7",
        "desc":"Near the NIR plateau. Useful for LAI estimation and biomass mapping."},
    6: {"name":"B8",  "label":"NIR",            "wavelength":"842 nm",  "resolution":"10 m", "color":"#00BCD4",
        "desc":"High reflectance for vegetation. Core band for NDVI, LAI, and vegetation classification."},
    7: {"name":"B8A", "label":"Narrow NIR",     "wavelength":"865 nm",  "resolution":"20 m", "color":"#009688",
        "desc":"Narrower NIR band for improved vegetation characterization and water vapor estimation."},
    8: {"name":"B11", "label":"SWIR 1",         "wavelength":"1610 nm", "resolution":"20 m", "color":"#FF5722",
        "desc":"Sensitive to soil and vegetation moisture. Used for drought monitoring and burn mapping."},
    9: {"name":"B12", "label":"SWIR 2",         "wavelength":"2190 nm", "resolution":"20 m", "color":"#795548",
        "desc":"Complementary SWIR band for mineral mapping, soil moisture, and snow/ice discrimination."},
}

# ── Cached data loaders ─────────────────────────────────────────────────────
@st.cache_data
def load_config():
    with open(os.path.join(BASE, 'configs/config.yaml')) as f:
        return yaml.safe_load(f)

@st.cache_data
def load_metadata():
    c = load_config()
    with open(os.path.join(BASE, c['paths']['metadata'])) as f:
        return json.load(f)

@st.cache_data
def load_s2(patch_id):
    """Load Sentinel-2 array (46, 10, 128, 128)."""
    c = load_config()
    path = os.path.join(BASE, c['paths']['s2_dir'], f"S2_{patch_id}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data
def load_target(patch_id):
    """Load target labels (1, 128, 128)."""
    c = load_config()
    path = os.path.join(BASE, c['paths']['annotations_dir'], f"TARGET_{patch_id}.npy")
    if os.path.exists(path):
        return np.load(path)
    return None

@st.cache_data
def get_patch_ids():
    meta = load_metadata()
    return sorted([int(f['properties']['ID_PATCH']) for f in meta['features']])

@st.cache_data
def get_split_patches():
    """Return dict of train/val/test patch ID lists."""
    c = load_config()
    splits = {}
    for split in ['train', 'val', 'test']:
        path = os.path.join(BASE, c['paths']['splits_dir'], f"{split}_patches.txt")
        if os.path.exists(path):
            with open(path) as f:
                splits[split] = [int(l.strip()) for l in f.readlines() if l.strip()]
        else:
            splits[split] = []
    return splits

@st.cache_data
def load_metrics(prefix):
    """Load summary metrics JSON for a model prefix (rf_ or unet_)."""
    c = load_config()
    path = os.path.join(BASE, c['paths']['output_dir'], 'metrics', f"{prefix}summary_metrics.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None

@st.cache_data
def load_confusion_matrix(prefix):
    """Load confusion matrix CSV."""
    import pandas as pd
    c = load_config()
    path = os.path.join(BASE, c['paths']['output_dir'], 'metrics', f"{prefix}confusion_matrix.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_per_class_metrics(prefix):
    """Load per-class metrics CSV."""
    import pandas as pd
    c = load_config()
    path = os.path.join(BASE, c['paths']['output_dir'], 'metrics', f"{prefix}per_class_metrics.csv")
    if os.path.exists(path):
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data
def load_classification_report(prefix):
    """Load full classification report JSON."""
    c = load_config()
    path = os.path.join(BASE, c['paths']['output_dir'], 'metrics', f"{prefix}classification_report.json")
    if os.path.exists(path):
        return json.load(open(path))
    return None

def metrics_path(prefix, fname):
    c = load_config()
    return os.path.join(BASE, c['paths']['output_dir'], 'metrics', f"{prefix}{fname}")

# ── Image helpers ────────────────────────────────────────────────────────────
def hex_to_rgba(hex_color, alpha=180):
    h = hex_color.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4)) + (alpha,)

def make_rgb(s2, date_idx=None, percentile_stretch=True):
    """Create RGB image from S2 data. If date_idx is None, use median composite."""
    if date_idx is not None:
        img = s2[date_idx]  # (10, 128, 128)
    else:
        img = np.median(s2, axis=0)
    # B4(Red)=idx2, B3(Green)=idx1, B2(Blue)=idx0
    rgb = np.stack([img[2], img[1], img[0]], axis=-1).astype(np.float32)
    if percentile_stretch:
        p2, p98 = np.percentile(rgb, 2), np.percentile(rgb, 98)
        rgb = np.clip((rgb - p2) / (p98 - p2 + 1e-8) * 255, 0, 255)
    return rgb.astype(np.uint8)

def make_label_rgb(label_map, crop_colors=None):
    """Convert label array to RGB using crop color palette."""
    if crop_colors is None:
        crop_colors = CROP_COLORS
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for cid, hex_c in crop_colors.items():
        mask = label_map == cid
        if mask.any():
            r, g, b = hex_to_rgba(hex_c)[:3]
            rgb[mask] = [r, g, b]
    return rgb

def compute_ndvi(s2_single):
    """Compute NDVI from a single-date S2 image (10, H, W)."""
    nir = s2_single[6].astype(np.float32)
    red = s2_single[2].astype(np.float32)
    return (nir - red) / (nir + red + 1e-8)

def compute_ndwi(s2_single):
    """Compute NDWI from a single-date S2 image (10, H, W)."""
    green = s2_single[1].astype(np.float32)
    nir = s2_single[6].astype(np.float32)
    return (green - nir) / (green + nir + 1e-8)

def compute_evi(s2_single):
    """Compute EVI from a single-date S2 image."""
    nir = s2_single[6].astype(np.float32)
    red = s2_single[2].astype(np.float32)
    blue = s2_single[0].astype(np.float32)
    return 2.5 * (nir - red) / (nir + 6*red - 7.5*blue + 1e4 + 1e-8)

def compute_savi(s2_single, L=0.5):
    """Compute SAVI from a single-date S2 image."""
    nir = s2_single[6].astype(np.float32)
    red = s2_single[2].astype(np.float32)
    return ((nir - red) / (nir + red + L + 1e-8)) * (1 + L)

def compute_gndvi(s2_single):
    """Compute GNDVI (Green NDVI) from a single-date S2 image."""
    nir = s2_single[6].astype(np.float32)
    green = s2_single[1].astype(np.float32)
    return (nir - green) / (nir + green + 1e-8)

def compute_msavi(s2_single):
    """Compute MSAVI2 from a single-date S2 image."""
    nir = s2_single[6].astype(np.float32)
    red = s2_single[2].astype(np.float32)
    return (2*nir + 1 - np.sqrt((2*nir + 1)**2 - 8*(nir - red) + 1e-8)) / 2

def compute_ndbi(s2_single):
    """Compute NDBI from a single-date S2 image."""
    swir = s2_single[8].astype(np.float32)  # B11
    nir = s2_single[6].astype(np.float32)
    return (swir - nir) / (swir + nir + 1e-8)

INDEX_FUNCS = {
    "NDVI": compute_ndvi, "NDWI": compute_ndwi, "EVI": compute_evi,
    "SAVI": compute_savi, "GNDVI": compute_gndvi, "MSAVI": compute_msavi, "NDBI": compute_ndbi,
}

INDEX_INFO = {
    "NDVI":  {"formula": "(NIR − Red) / (NIR + Red)", "range": "−1 to 1", "desc": "Normalized Difference Vegetation Index — primary indicator of vegetation health and density."},
    "NDWI":  {"formula": "(Green − NIR) / (Green + NIR)", "range": "−1 to 1", "desc": "Normalized Difference Water Index — highlights water content in vegetation and open water bodies."},
    "EVI":   {"formula": "2.5 × (NIR − Red) / (NIR + 6R − 7.5B + 10⁴)", "range": "−1 to 1", "desc": "Enhanced Vegetation Index — reduces atmospheric and soil noise, better for dense canopy."},
    "SAVI":  {"formula": "(NIR − Red) / (NIR + Red + L) × (1+L)", "range": "−1 to 1", "desc": "Soil-Adjusted Vegetation Index — corrects for soil brightness in sparse vegetation areas."},
    "GNDVI": {"formula": "(NIR − Green) / (NIR + Green)", "range": "−1 to 1", "desc": "Green NDVI — sensitive to chlorophyll concentration variations in crop canopies."},
    "MSAVI": {"formula": "(2NIR + 1 − √((2NIR+1)² − 8(NIR−Red))) / 2", "range": "0 to 1", "desc": "Modified SAVI — self-adjusting soil factor, optimal for early-stage crops."},
    "NDBI":  {"formula": "(SWIR1 − NIR) / (SWIR1 + NIR)", "range": "−1 to 1", "desc": "Normalized Difference Built-up Index — highlights impervious surfaces and urban areas."},
}

# ── CSS ──────────────────────────────────────────────────────────────────────
GLOBAL_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* Base Theme */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"],
.main, .block-container, section[data-testid="stSidebar"] {
    background-color: #0f172a !important; /* Premium Dark Navy */
    color: #f8fafc !important; /* Off-white text */
    font-family: 'Inter', sans-serif !important;
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Typography */
*, p, span, label, li, td, th, h1, h2, h3, h4, h5, h6,
div[data-testid="stMarkdownContainer"] *,
div[data-testid="stMetricValue"], div[data-testid="stMetricLabel"] {
    font-family: 'Inter', sans-serif !important;
}

h1, h2, h3, h4, h5, h6 {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
}

/* Glassmorphism Metrics */
div[data-testid="stMetric"] {
    background: rgba(30, 41, 59, 0.7) !important;
    backdrop-filter: blur(10px) !important;
    -webkit-backdrop-filter: blur(10px) !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important; 
    border-radius: 12px !important; 
    padding: 20px !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}

div[data-testid="stMetric"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1) !important;
}

div[data-testid="stMetricValue"] { color: #38bdf8 !important; font-weight: 700 !important; }
div[data-testid="stMetricLabel"] { color: #94a3b8 !important; text-transform: uppercase !important; font-size: 0.8rem !important; font-weight: 600 !important; letter-spacing: 0.05em !important; }

/* Buttons */
div.stButton > button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: #ffffff !important;
    border-radius: 8px !important; 
    font-weight: 600 !important;
    padding: 0.5rem 1rem !important;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    transition: all 0.2s ease !important;
}
div.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -2px rgba(0, 0, 0, 0.1) !important;
    background: linear-gradient(135deg, #60a5fa 0%, #3b82f6 100%) !important;
}
div.stButton > button p { color: #ffffff !important; }

/* Tabs */
button[data-baseweb="tab"] { font-weight: 600 !important; color: #94a3b8 !important; }
button[data-baseweb="tab"][aria-selected="true"] { color: #38bdf8 !important; border-bottom: 3px solid #38bdf8 !important; }

/* File Uploader */
div[data-testid="stFileUploader"] {
    background: rgba(30, 41, 59, 0.5) !important;
    border: 2px dashed rgba(255, 255, 255, 0.2) !important;
    border-radius: 12px !important;
    transition: all 0.2s ease !important;
}
div[data-testid="stFileUploader"]:hover {
    background: rgba(30, 41, 59, 0.8) !important;
}

/* Chat inputs and Expanders */
div[data-testid="stChatInput"] textarea { background: rgba(30, 41, 59, 0.8) !important; color: #f8fafc !important; border: 1px solid rgba(255,255,255,0.1) !important; border-radius: 24px !important; }
div[data-testid="stChatMessage"] { background: rgba(30, 41, 59, 0.6) !important; border: 1px solid rgba(255,255,255,0.05) !important; border-radius: 12px !important; }

hr { border-color: rgba(255,255,255,0.1) !important; }
iframe { border-radius: 12px !important; background: #0f172a !important; border: 1px solid rgba(255,255,255,0.1) !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2) !important; }
div[data-testid="stIFrame"], div.stFolium { background: #0f172a !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: #475569; }

/* Slider */
div[data-testid="stSlider"] label { color: #94a3b8 !important; font-weight: 600 !important; }
div[data-baseweb="slider"] div { color: #38bdf8 !important; }

/* Expander */
details summary span { color: #e2e8f0 !important; font-weight: 600 !important; }
details { background: rgba(30, 41, 59, 0.7) !important; border: 1px solid rgba(255, 255, 255, 0.1) !important; border-radius: 12px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important; }

/* Dataframes */
[data-testid="stDataFrame"] { background: transparent !important; }
[data-testid="stDataFrame"] table { color: #e2e8f0 !important; }
[data-testid="stDataFrame"] th { background-color: rgba(30, 41, 59, 0.9) !important; color: #94a3b8 !important; border-bottom: 2px solid rgba(255, 255, 255, 0.1) !important; }
[data-testid="stDataFrame"] td { border-bottom: 1px solid rgba(255, 255, 255, 0.05) !important; }
</style>
"""
