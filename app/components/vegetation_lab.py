"""
vegetation_lab.py — Interactive vegetation index computation, comparison, histograms, maps.
"""
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import streamlit as st
from .common import *
from i18n import translations

class VegetationLab:
    def __init__(self):
        self.config = load_config()
        self.classes = self.config['classes']
        self.patch_ids = get_patch_ids()

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["veg_title"])
        st.markdown(t["veg_desc"], unsafe_allow_html=True)

        ctrl, viz = st.columns([1, 3])
        with ctrl:
            pid = st.selectbox(t["veg_patch"], self.patch_ids, key="vi_pid")
            date_idx = st.slider(t["veg_date"], 0, 45, 23, key="vi_date")
            selected_indices = st.multiselect(t["veg_indices"],
                                               list(INDEX_FUNCS.keys()),
                                               default=["NDVI", "NDWI", "EVI"],
                                               key="vi_indices")
            compare_mode = st.toggle(t["veg_compare"], value=False, key="vi_compare")

            # Index info cards
            for idx_name in selected_indices:
                info = INDEX_INFO[idx_name]
                st.markdown(f"""
                <div style="background:#0d1b30;border:1px solid #1a3560;border-radius:8px;padding:10px;margin:6px 0">
                    <div style="color:#42a5f5;font-weight:700">{idx_name}</div>
                    <div style="color:#7a9cc6;font-size:0.75rem">{info['formula']}</div>
                    <div style="color:#e8f0fe;font-size:0.78rem;margin-top:4px">{info['desc']}</div>
                </div>""", unsafe_allow_html=True)

        s2 = load_s2(pid)
        tgt = load_target(pid)
        if s2 is None:
            st.error(t["veg_no_data"])
            return

        with viz:
            if not selected_indices:
                st.info(t["veg_sel_one"])
                return

            single = s2[date_idx]

            if compare_mode and len(selected_indices) >= 2:
                # Side-by-side comparison
                idx1_name, idx2_name = selected_indices[0], selected_indices[1]
                v1 = INDEX_FUNCS[idx1_name](single)
                v2 = INDEX_FUNCS[idx2_name](single)
                c1, c2 = st.columns(2)
                with c1:
                    fig1 = px.imshow(v1, color_continuous_scale='RdYlGn', zmin=-0.3, zmax=0.8,
                                     title=f"{idx1_name} — Date {date_idx+1}")
                    fig1.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe', height=350)
                    st.plotly_chart(fig1, use_container_width=True)
                with c2:
                    fig2 = px.imshow(v2, color_continuous_scale='RdYlGn', zmin=-0.3, zmax=0.8,
                                     title=f"{idx2_name} — Date {date_idx+1}")
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe', height=350)
                    st.plotly_chart(fig2, use_container_width=True)

                # Scatter: index vs index
                st.markdown(t["veg_scatter_title"])
                flat1, flat2 = v1.flatten(), v2.flatten()
                sample_idx = np.random.choice(len(flat1), min(3000, len(flat1)), replace=False)
                fig_sc = go.Figure()
                fig_sc.add_trace(go.Scattergl(x=flat1[sample_idx], y=flat2[sample_idx],
                                               mode='markers', marker=dict(size=2, color='#42a5f5', opacity=0.4)))
                fig_sc.update_layout(xaxis_title=idx1_name, yaxis_title=idx2_name,
                                     paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                     font_color='#e8f0fe', height=350, margin=dict(t=20,b=40))
                st.plotly_chart(fig_sc, use_container_width=True)

            else:
                # Grid view of all selected indices
                n_idx = len(selected_indices)
                cols = st.columns(min(n_idx, 3))
                for i, idx_name in enumerate(selected_indices):
                    with cols[i % 3]:
                        val = INDEX_FUNCS[idx_name](single)
                        cscale = 'Blues' if 'W' in idx_name or 'B' in idx_name else 'RdYlGn'
                        fig = px.imshow(val, color_continuous_scale=cscale, zmin=-0.3, zmax=0.8,
                                        title=idx_name)
                        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                                          height=280, margin=dict(t=35,b=10))
                        st.plotly_chart(fig, use_container_width=True)

            # ── Histograms per class ──
            st.markdown("---")
            st.markdown(t["veg_hist_title"])
            hist_index = st.selectbox(t["veg_hist_idx"], selected_indices, key="vi_hist_idx")
            val_map = INDEX_FUNCS[hist_index](single)
            if tgt is not None:
                label_map = tgt[0]
                unique_cls = [int(u) for u in np.unique(label_map) if int(u) not in (0, 19)]
                fig_hist = go.Figure()
                for uid in sorted(unique_cls):
                    mask = label_map == uid
                    vals = val_map[mask]
                    name = self.classes.get(uid, str(uid))
                    clr = CROP_COLORS.get(uid, '#888')
                    fig_hist.add_trace(go.Histogram(x=vals, name=name, marker_color=clr,
                                                     opacity=0.6, nbinsx=50))
                fig_hist.update_layout(barmode='overlay',
                                        title=f"{hist_index} {t['veg_dist_by_crop']}",
                                        xaxis_title=hist_index, yaxis_title=t["veg_pixel_count"],
                                        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                        font_color='#e8f0fe', height=380, margin=dict(t=50,b=40),
                                        legend=dict(bgcolor='rgba(0,0,0,0)'))
                st.plotly_chart(fig_hist, use_container_width=True)

            # ── Per-class statistics table ──
            st.markdown(t["veg_stat_title"])
            if tgt is not None:
                rows = []
                for uid in sorted(unique_cls):
                    mask = label_map == uid
                    vals = val_map[mask]
                    rows.append({
                        t["veg_crop"]: self.classes.get(uid, str(uid)),
                        t["veg_mean"]: f"{np.mean(vals):.3f}",
                        t["veg_std"]: f"{np.std(vals):.3f}",
                        t["veg_min"]: f"{np.min(vals):.3f}",
                        t["veg_max"]: f"{np.max(vals):.3f}",
                        t["veg_pixels"]: int(np.sum(mask))
                    })
                import pandas as pd
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

def render():
    VegetationLab().render()
