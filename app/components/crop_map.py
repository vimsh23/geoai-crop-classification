"""
crop_map.py — Existing interactive Folium map with crop label overlays (extracted from original app).
"""
import os, base64, numpy as np
import folium
from folium import raster_layers
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Polygon
from io import BytesIO
from PIL import Image
import streamlit as st
from .common import *
from i18n import translations

class CropMapExplorer:
    def __init__(self):
        self.config = load_config()
        self.meta = load_metadata()
        self.classes = self.config['classes']
        
    def render_sidebar(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        patch_ids = [t["cmap_all_patches"]] + get_patch_ids()
        self.selected_pid = st.selectbox(t["cmap_select_patch"], patch_ids, index=0, key="map_pid")

        st.markdown("---")
        self.show_labels = st.toggle(t["cmap_show_labels"], value=True, key="map_labels")
        self.show_median = st.toggle(t["cmap_show_median"], value=False, key="map_median")

        # Load the labels for this patch
        self.label_map = None
        self.unique_classes = []
        if self.selected_pid != t["cmap_all_patches"]:
            tgt = load_target(self.selected_pid)
            if tgt is not None:
                self.label_map = tgt[0]
                self.unique_classes = [int(u) for u in np.unique(self.label_map) if int(u) not in (0, 19)]
                st.markdown(t["cmap_crops_patch"])
                for uid in sorted(self.unique_classes):
                    name = self.classes.get(uid, f"Class {uid}")
                    clr = CROP_COLORS.get(uid, '#aaa')
                    pct = np.sum(self.label_map == uid) / self.label_map.size * 100
                    st.markdown(
                        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0">'
                        f'<div style="width:16px;height:16px;border-radius:4px;background:{clr};'
                        f'border:1px solid rgba(255,255,255,0.25)"></div>'
                        f'<span style="color:#e8f0fe;font-size:0.85rem"><b>{name}</b> — {pct:.1f}%</span></div>',
                        unsafe_allow_html=True)
                st.markdown("---")
                st.markdown(t["cmap_toggle_info"], unsafe_allow_html=True)
            else:
                st.warning(t["cmap_no_target"])

    def render_map(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        # Get patch geometry and reproject
        if self.selected_pid == t["cmap_all_patches"]:
            polys = []
            for feat in self.meta['features']:
                if 'geometry' in feat and feat['geometry']['type'] == 'Polygon':
                    coords = feat['geometry']['coordinates'][0]
                    polys.append(Polygon(coords))
            gdf = gpd.GeoDataFrame(crs="EPSG:2154", geometry=polys).to_crs("EPSG:4326")
            bounds = gdf.total_bounds
            img_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]
        else:
            feat = next(f for f in self.meta['features'] if int(f['properties']['ID_PATCH']) == self.selected_pid)
            coords = feat['geometry']['coordinates'][0]
            poly = Polygon(coords)
            gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:2154", geometry=[poly]).to_crs("EPSG:4326")
            bounds = gdf.total_bounds
            img_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

        m = folium.Map(
            location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2],
            zoom_start=15,
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri')

        folium.TileLayer('CartoDB dark_matter', name='Dark', overlay=False, control=True, show=False).add_to(m)

        # ── Median cloud-free composite overlay ──
        if self.show_median and self.selected_pid != t["cmap_all_patches"]:
            s2 = load_s2(self.selected_pid)
            if s2 is not None:
                rgb = make_rgb(s2)
                pil_rgb = Image.fromarray(rgb, 'RGB')
                buf = BytesIO()
                pil_rgb.save(buf, format='PNG')
                encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded}",
                    bounds=img_bounds, opacity=0.95,
                    name=t["cmap_median_name"], interactive=False, show=True
                ).add_to(m)

        # ── Patch boundary ──
        def poly_style(x):
            return {'fillColor':'transparent','color':'#00e5ff','weight':2,'dashArray':'6,4'}
            
        folium.GeoJson(
            gdf.to_json(), name=t["cmap_boundary_name"],
            style_function=poly_style
        ).add_to(m)

        # ── Crop label overlays ──
        if self.label_map is not None and self.show_labels:
            h, w = self.label_map.shape
            combined = np.zeros((h, w, 4), dtype=np.uint8)
            for uid in self.unique_classes:
                rgba = hex_to_rgba(CROP_COLORS.get(uid, '#aaaaaa'), alpha=160)
                combined[self.label_map == uid] = rgba
            pil_comb = Image.fromarray(combined, 'RGBA')
            buf = BytesIO()
            pil_comb.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded}",
                bounds=img_bounds, opacity=0.8,
                name=t["cmap_all_crops"], interactive=False, show=True
            ).add_to(m)

            for uid in sorted(self.unique_classes):
                crop_name = self.classes.get(uid, f"Class {uid}")
                rgba = hex_to_rgba(CROP_COLORS.get(uid, '#aaaaaa'), alpha=180)
                overlay_img = np.zeros((h, w, 4), dtype=np.uint8)
                overlay_img[self.label_map == uid] = rgba
                pil_img = Image.fromarray(overlay_img, 'RGBA')
                buf = BytesIO()
                pil_img.save(buf, format='PNG')
                encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
                raster_layers.ImageOverlay(
                    image=f"data:image/png;base64,{encoded}",
                    bounds=img_bounds, opacity=0.85,
                    name=f"{crop_name}", interactive=False, show=False
                ).add_to(m)

        m.fit_bounds(img_bounds)
        folium.LayerControl(collapsed=True).add_to(m)
        st_folium(m, width=None, height=650, returned_objects=[])

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["cmap_title"])
        st.markdown(t["cmap_desc"], unsafe_allow_html=True)

        ctrl1, ctrl2 = st.columns([1, 3])
        with ctrl1:
            self.render_sidebar()

        with ctrl2:
            self.render_map()
            
def render():
    CropMapExplorer().render()
