"""
spectral_explorer.py — Band switching, spectral signatures, pixel inspector, and interactive ROI Profiling.
"""
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import base64
from io import BytesIO
from PIL import Image, ImageDraw
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
import geopandas as gpd
from shapely.geometry import Polygon
from .common import *
from i18n import translations

class SpectralExplorer:
    def __init__(self):
        self.config = load_config()
        self.classes = self.config['classes']
        self.patch_ids = get_patch_ids()

    def render_band_explorer(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["spec_band_desc"], unsafe_allow_html=True)
        ctrl, viz = st.columns([1, 3])
        with ctrl:
            pid = st.selectbox(t["spec_patch"], self.patch_ids, key="be_pid")
            def format_band(x):
                return f"{BAND_INFO[x]['name']} — {BAND_INFO[x]['label']}"
                
            band_idx = st.selectbox(t["spec_band"], list(BAND_INFO.keys()),
                                     format_func=format_band,
                                     key="be_band")
            date_idx = st.slider(t["spec_date"], 0, 45, 23, key="be_date")
            info = BAND_INFO[band_idx]
            st.markdown(f"""
            <div style="background:#0d1b30;border:1px solid #1a3560;border-radius:10px;padding:14px;margin-top:12px">
                <div style="color:{info['color']};font-weight:700;font-size:1.1rem">{info['name']} · {info['label']}</div>
                <div style="color:#7a9cc6;font-size:0.8rem;margin:6px 0">λ = {info['wavelength']} · {info['resolution']}</div>
                <div style="color:#e8f0fe;font-size:0.82rem">{info['desc']}</div>
            </div>""", unsafe_allow_html=True)

        with viz:
            s2 = load_s2(pid)
            if s2 is None:
                st.error(t["spec_no_data"])
                return
            band_data = s2[date_idx, band_idx]
            fig = px.imshow(band_data, color_continuous_scale='Viridis',
                            title=f"{info['name']} ({info['label']}) — Patch {pid}, Date {date_idx+1}")
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                              coloraxis_colorbar_title="DN", height=500)
            st.plotly_chart(fig, use_container_width=True)



    def render_roi_profile(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["spec_roi_desc"], unsafe_allow_html=True)
        
        # Controls at the top
        c1, c2, c3 = st.columns([1, 1, 2])
        with c1:
            pid = st.selectbox(t["spec_patch"], self.patch_ids, key="roi_pid")
        with c2:
            stat_method = st.radio(t["spec_agg"], [t["spec_median"], t["spec_avg"]], horizontal=True)
        with c3:
            st.markdown("<br>", unsafe_allow_html=True) # spacing
            show_labels = st.toggle(t["spec_show_labels"], value=True, key="roi_labels")
            
        # Prepare Map
        meta = load_metadata()
        feat = next(f for f in meta['features'] if int(f['properties']['ID_PATCH']) == pid)
        coords = feat['geometry']['coordinates'][0]
        poly = Polygon(coords)
        gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:2154", geometry=[poly]).to_crs("EPSG:4326")
        bounds = gdf.total_bounds
        img_bounds = [[bounds[1], bounds[0]], [bounds[3], bounds[2]]]

        m = folium.Map(location=[(bounds[1]+bounds[3])/2, (bounds[0]+bounds[2])/2], zoom_start=15,
                       tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', attr='Esri')
        folium.TileLayer('CartoDB dark_matter', name='Dark', overlay=False, control=True).add_to(m)

        s2 = load_s2(pid)
        tgt = load_target(pid)
        if s2 is not None:
            rgb = make_rgb(s2)
            pil_rgb = Image.fromarray(rgb, 'RGB')
            buf = BytesIO()
            pil_rgb.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded}", bounds=img_bounds, opacity=0.85, name=t["spec_median_rgb"]
            ).add_to(m)
            
        if tgt is not None and show_labels:
            label_map = tgt[0]
            unique_classes = [int(u) for u in np.unique(label_map) if int(u) not in (0, 19)]
            h, w = label_map.shape
            combined = np.zeros((h, w, 4), dtype=np.uint8)
            for uid in unique_classes:
                rgba = hex_to_rgba(CROP_COLORS.get(uid, '#aaaaaa'), alpha=150)
                combined[label_map == uid] = rgba
            pil_comb = Image.fromarray(combined, 'RGBA')
            buf = BytesIO()
            pil_comb.save(buf, format='PNG')
            encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
            folium.raster_layers.ImageOverlay(
                image=f"data:image/png;base64,{encoded}", bounds=img_bounds, opacity=0.7, name=t["spec_crop_labels"]
            ).add_to(m)
            
        Draw(export=False, draw_options={
            'polygon': True, 'rectangle': True, 
            'polyline': False, 'circle': False, 'marker': False, 'circlemarker': False
        }).add_to(m)
        folium.LayerControl(collapsed=False).add_to(m)
        
        st.markdown(t["spec_draw_inst"])
        
        # Full width map
        st_data = st_folium(m, use_container_width=True, height=500)
        
        st.markdown("---")
        
        # Wrap the heavy processing and plotting in a submit button
        if st.button(t["spec_generate"], type="primary"):
            if st_data is None:
                st.warning(t["spec_warn_draw"])
                return
                
            drawings = st_data.get("all_drawings", [])
            if not drawings:
                st.warning(t["spec_warn_draw"])
                return
                    
            geom = drawings[-1]['geometry']
            if geom['type'] != 'Polygon':
                st.warning(t["spec_warn_poly"])
                return
                
            with st.spinner(t["spec_analyzing"]):
                poly_coords = geom['coordinates'][0]
                
                min_lon, min_lat, max_lon, max_lat = bounds
                shape = (128, 128)
                poly_pixels = []
                for lon, lat in poly_coords:
                    x = (lon - min_lon) / (max_lon - min_lon) * shape[1]
                    y = (max_lat - lat) / (max_lat - min_lat) * shape[0]
                    poly_pixels.append((x, y))
                    
                img_mask = Image.new('L', (shape[1], shape[0]), 0)
                ImageDraw.Draw(img_mask).polygon(poly_pixels, outline=1, fill=1)
                mask = np.array(img_mask, dtype=bool)
                
                num_pixels = mask.sum()
                if num_pixels == 0:
                    st.warning(t["spec_warn_small"])
                    return
                    
                st.markdown(f"{t['spec_roi_profile']} ({num_pixels} {t['spec_pixels_sel']})")
                
                # s2 shape: (46, 10, 128, 128) -> extract masked pixels
                masked_s2 = s2[:, :, mask] # Shape: (46, 10, num_pixels)
                if stat_method == "Median":
                    roi_ts = np.median(masked_s2, axis=2) # Shape: (46, 10)
                else:
                    roi_ts = np.mean(masked_s2, axis=2) # Shape: (46, 10)
                    
                # Plot 1: ERDAS-style Spectral Profile (Wavelength vs Reflectance)
                wavelengths = [float(BAND_INFO[i]['wavelength'].replace(' nm','')) for i in range(10)]
                band_names = [BAND_INFO[i]['name'] for i in range(10)]
                
                overall_spectral = np.median(roi_ts, axis=0) if stat_method == t["spec_median"] else np.mean(roi_ts, axis=0)
                
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(x=wavelengths, y=overall_spectral, mode='lines+markers', name=f'{t["spec_overall"]} {stat_method}',
                                          line=dict(color='#ffca28', width=3), marker=dict(size=8)))
                                          
                min_spectral = np.min(roi_ts, axis=0)
                max_spectral = np.max(roi_ts, axis=0)
                fig2.add_trace(go.Scatter(x=wavelengths + wavelengths[::-1], 
                                          y=list(max_spectral) + list(min_spectral[::-1]),
                                          fill='toself', fillcolor='rgba(255,202,40,0.15)', line=dict(color='rgba(0,0,0,0)'),
                                          showlegend=False, hoverinfo='skip'))
                                          
                fig2.update_layout(title=t["spec_avg_sig"],
                                   xaxis_title=t["spec_wave"], yaxis_title=t["spec_reflec"],
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font_color='#e8f0fe', height=330, margin=dict(t=40,b=30))
                fig2.update_xaxes(tickvals=wavelengths, ticktext=band_names, tickangle=45)
                st.plotly_chart(fig2, use_container_width=True)
                
                # Plot 2: Temporal Profile (NDVI, NDWI)
                import datetime
                dates_s2 = feat['properties']['dates-S2']
                actual_dates = [datetime.datetime.strptime(str(dates_s2[str(i)]), "%Y%m%d").strftime("%Y-%m-%d") for i in range(46)]
                
                roi_nir = roi_ts[:, 6]
                roi_red = roi_ts[:, 2]
                roi_green = roi_ts[:, 1]
                
                roi_ndvi = (roi_nir - roi_red) / (roi_nir + roi_red + 1e-8)
                roi_ndwi = (roi_green - roi_nir) / (roi_green + roi_nir + 1e-8)
                
                fig1 = go.Figure()
                fig1.add_trace(go.Scatter(x=actual_dates, y=roi_ndvi, mode='lines+markers', name='NDVI', line=dict(color='#66bb6a', width=2)))
                fig1.add_trace(go.Scatter(x=actual_dates, y=roi_ndwi, mode='lines+markers', name='NDWI', line=dict(color='#42a5f5', width=2)))
                fig1.update_layout(title=f"{t['spec_temp_prof']} ({stat_method} {t['spec_over_46']})",
                                   xaxis_title=t["temp_date"], yaxis_title=t["spec_index_val"],
                                   paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   font_color='#e8f0fe', height=300, margin=dict(t=40,b=30))
                st.plotly_chart(fig1, use_container_width=True)

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["spec_title"])

        sub_tab = st.radio(t["spec_mode"], [t["spec_mode_band"], t["spec_mode_roi"]],
                            horizontal=True, key="spec_mode")

        if sub_tab == t["spec_mode_band"]:
            self.render_band_explorer()
        else:
            self.render_roi_profile()

def render():
    SpectralExplorer().render()
