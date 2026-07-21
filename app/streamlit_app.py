"""
PASTIS CropAI Intelligence Platform
====================================
A world-class GeoAI Explorer for crop classification using Sentinel-2 time series.

Modules:
  Dashboard    — Dataset overview, KPIs, class distribution, pipeline summary
  Crop Map     — Interactive Folium map with crop label overlays (QGIS-style)
  Temporal     — Time slider for 46 satellite observations, NDVI/NDWI time series
  Spectral     — Band explorer, spectral signatures, pixel inspector
  Vegetation   — Interactive index lab (NDVI, NDWI, EVI, SAVI, GNDVI, MSAVI, NDBI)
  Models       — Radar charts, confusion matrices, per-class deep dive
  Predict      — Upload → Preprocess → Inference → Visualization → Export

"""
import os, sys
# ── Windows DLL fix: register torch lib dir and import torch FIRST ──────────
# If Streamlit loads first, it causes DLL fragmentation/conflicts (WinError 1114)
def _register_torch_dll_dirs():
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.isdir(torch_lib):
                os.add_dll_directory(torch_lib)
    except (AttributeError, OSError):
        pass

if sys.platform == "win32":
    _register_torch_dll_dirs()

try:
    import torch
except Exception:
    pass

import streamlit as st

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PASTIS CropAI Platform",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── i18n Language Selection ──────────────────────────────────────────────────
from i18n import translations

# ── Import shared CSS ────────────────────────────────────────────────────────
from components.common import GLOBAL_CSS
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

# ── Header & Language Selector ───────────────────────────────────────────────
col1, col2, col3 = st.columns([1, 3, 1])

with col3:
    language = st.radio(
        "Language", 
        options=["English", "日本語"], 
        horizontal=True,
        label_visibility="collapsed"
    )
    st.session_state['language'] = language
    t = translations[language]

with col2:
    st.markdown(f"""
    <div style="text-align:center;padding:6px 0 2px">
        <span style="font-size:1.5rem;font-weight:800;color:#1a73e8">{t["brand_title"]}</span>
        <span style="font-size:1.5rem;font-weight:300;color:#5f6368"> {t["brand_subtitle"]}</span>
    </div>
    """, unsafe_allow_html=True)



# ── Tab Navigation ───────────────────────────────────────────────────────────
tabs = st.tabs([
    t["tab_dashboard"],
    t["tab_map"],
    t["tab_exploration"],
    t["tab_models"],
    t["tab_predict"],
])

# ── Page Routing ─────────────────────────────────────────────────────────────
with tabs[0]:
    from components.dataset_dashboard import render as render_dashboard
    render_dashboard()

with tabs[1]:
    from components.crop_map import render as render_crop_map
    render_crop_map()

with tabs[2]:
    st.markdown(t["eda_title"])
    eda_tabs = st.tabs([t["eda_temporal"], t["eda_spectral"], t["eda_vegetation"]])
    with eda_tabs[0]:
        from components.temporal_explorer import render as render_temporal
        render_temporal()
    with eda_tabs[1]:
        from components.spectral_explorer import render as render_spectral
        render_spectral()
    with eda_tabs[2]:
        from components.vegetation_lab import render as render_vegetation
        render_vegetation()

with tabs[3]:
    from components.model_comparison import render as render_models
    render_models()

with tabs[4]:
    from components.prediction_portal import render as render_predict
    render_predict()
