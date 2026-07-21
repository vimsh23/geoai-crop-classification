"""
temporal_explorer.py — Time slider to browse 46 temporal observations with RGB, NDVI, NDWI.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from .common import *
from i18n import translations

class TemporalExplorer:
    def __init__(self):
        self.config = load_config()
        self.classes = self.config['classes']
        self.patch_ids = get_patch_ids()

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["temp_title"])
        st.markdown(t["temp_desc"], unsafe_allow_html=True)

        ctrl, viz = st.columns([1, 3])
        with ctrl:
            pid = st.selectbox(t["temp_select_patch"], self.patch_ids, key="temp_pid")
            display_mode = st.radio(t["temp_display"], [t["temp_rgb"], t["temp_ndvi"], t["temp_ndwi"], t["temp_fc"]], key="temp_mode")

        s2 = load_s2(pid)
        tgt = load_target(pid)
        meta = load_metadata()

        if s2 is None or meta is None:
            st.error(f"{t['temp_no_map'].replace('...', str(pid))}") # Slightly hacky for "Data not found" but we'll just keep the original structure where possible. Wait, let me just use hardcoded translated f-string for this one.
            st.error(f"Data not found for patch {pid}")
            return
            
        feat = next(f for f in meta['features'] if int(f['properties']['ID_PATCH']) == pid)
        dates_s2 = feat['properties']['dates-S2']
        import datetime
        actual_dates = [datetime.datetime.strptime(str(dates_s2[str(i)]), "%Y%m%d").strftime("%Y-%m-%d") for i in range(46)]

        n_dates = s2.shape[0]

        with viz:
            date_idx = st.slider(t["temp_obs_date"], 0, n_dates - 1, n_dates // 2,
                                  format="%d", help=f"{t['temp_slide_help']} {n_dates} dates", key="temp_slider")
            selected_date = actual_dates[date_idx]

            img_col, map_col, stat_col = st.columns([1.2, 1.2, 1])
            with img_col:
                if display_mode == t["temp_rgb"]:
                    img = make_rgb(s2, date_idx)
                    st.image(img, caption=f"{t['temp_true_color']} — {selected_date} ({date_idx+1}/{n_dates})", use_container_width=True)
                elif display_mode == t["temp_ndvi"]:
                    ndvi = compute_ndvi(s2[date_idx])
                    fig = px.imshow(ndvi, color_continuous_scale='RdYlGn', zmin=0.0, zmax=0.85)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                                      coloraxis_showscale=False, margin=dict(t=0,b=0,l=0,r=0))
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{t['temp_ndvi']} — {selected_date}")
                elif display_mode == t["temp_ndwi"]:
                    ndwi = compute_ndwi(s2[date_idx])
                    fig = px.imshow(ndwi, color_continuous_scale='RdBu', zmin=-0.4, zmax=0.4)
                    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                                      coloraxis_showscale=False, margin=dict(t=0,b=0,l=0,r=0))
                    fig.update_xaxes(visible=False)
                    fig.update_yaxes(visible=False)
                    st.plotly_chart(fig, use_container_width=True)
                    st.caption(f"{t['temp_ndwi']} — {selected_date}")
                elif display_mode == t["temp_fc"]:
                    img_data = s2[date_idx]
                    fc = np.stack([img_data[6], img_data[2], img_data[1]], axis=-1).astype(np.float32)
                    p2, p98 = np.percentile(fc, 2), np.percentile(fc, 98)
                    fc = np.clip((fc - p2) / (p98 - p2 + 1e-8) * 255, 0, 255).astype(np.uint8)
                    st.image(fc, caption=f"{t['temp_false_color']} — {selected_date}", use_container_width=True)

            with map_col:
                if tgt is not None:
                    label_map = tgt[0]
                    pred_rgb = make_label_rgb(label_map)
                    st.image(pred_rgb, caption=t["temp_gt_map"], use_container_width=True)
                else:
                    st.info(t["temp_no_map"])

            with stat_col:
                st.markdown(t["temp_stat_title"])
                single = s2[date_idx]
                ndvi_val = compute_ndvi(single)
                ndwi_val = compute_ndwi(single)
                st.metric(t["temp_mean_ndvi"], f"{np.mean(ndvi_val):.3f}")
                st.metric(t["temp_mean_ndwi"], f"{np.mean(ndwi_val):.3f}")
                
                if tgt is not None:
                    unique = [int(u) for u in np.unique(label_map) if int(u) not in (0, 19)]
                    st.markdown(t["temp_crops_patch"])
                    for uid in sorted(unique):
                        clr = CROP_COLORS.get(uid, '#aaa')
                        name = self.classes.get(uid, str(uid))
                        st.markdown(f'<span style="color:{clr};font-size:0.82rem">● {name}</span>', unsafe_allow_html=True)

        # ── Temporal Profiles ──
        st.markdown("---")
        st.markdown(t["temp_prof_title"])

        tc1, tc2 = st.columns(2)
        with tc1:
            ndvi_ts = [float(np.mean(compute_ndvi(s2[t]))) for t in range(n_dates)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual_dates, y=ndvi_ts, mode='lines+markers',
                                      name=t['temp_mean_ndvi'], line=dict(color='#66bb6a', width=2),
                                      marker=dict(size=4)))
            fig.add_vline(x=selected_date, line_dash="dash", line_color="#42a5f5", annotation_text=t["temp_current"])
            fig.update_layout(title=t["temp_ndvi_ts"], xaxis_title=t["temp_date"], yaxis_title="NDVI",
                              paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                              font_color='#e8f0fe', height=300, margin=dict(t=40,b=40))
            st.plotly_chart(fig, use_container_width=True)

        with tc2:
            ndwi_ts = [float(np.mean(compute_ndwi(s2[t]))) for t in range(n_dates)]
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=actual_dates, y=ndwi_ts, mode='lines+markers',
                                       name=t['temp_mean_ndwi'], line=dict(color='#42a5f5', width=2),
                                       marker=dict(size=4)))
            fig2.add_vline(x=selected_date, line_dash="dash", line_color="#42a5f5", annotation_text=t["temp_current"])
            fig2.update_layout(title=t["temp_ndwi_ts"], xaxis_title=t["temp_date"], yaxis_title="NDWI",
                               paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                               font_color='#e8f0fe', height=300, margin=dict(t=40,b=40))
            st.plotly_chart(fig2, use_container_width=True)

def render():
    TemporalExplorer().render()
