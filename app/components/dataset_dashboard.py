"""
dataset_dashboard.py — Rich landing page with KPIs, class distribution, observation calendar, AOI info.
"""
import os, json, numpy as np, pandas as pd
import plotly.express as px, plotly.graph_objects as go
import streamlit as st
from .common import *
from i18n import translations

class DatasetDashboard:
    def __init__(self):
        self.config = load_config()
        self.meta = load_metadata()
        self.classes = self.config['classes']
        self.splits = get_split_patches()
        self.patch_ids = get_patch_ids()

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        # ── Hero ──
        st.markdown(f"""
        <div style="text-align:center;padding:30px 0 10px">
            <h1 style="font-size:2.6rem;font-weight:800;
                background:linear-gradient(135deg,#fff 0%,#42a5f5 60%,#7c4dff 100%);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                margin-bottom:8px">{t["dash_hero_title"]}</h1>
            <p style="color:#7a9cc6 !important;font-size:0.95rem;max-width:650px;margin:0 auto">
                {t["dash_hero_sub"]}</p>
        </div>
        """, unsafe_allow_html=True)

        # ── KPI Row ──
        k = st.columns(6)
        k[0].metric(t["dash_kpi_patches"], str(len(self.patch_ids)))
        k[1].metric(t["dash_kpi_time"], str(self.config['dataset']['n_dates']))
        k[2].metric(t["dash_kpi_bands"], str(self.config['dataset']['n_bands']))
        k[3].metric(t["dash_kpi_classes"], str(len([x for x in self.classes if int(x) not in self.config.get('ignore_classes',[])])))
        k[4].metric(t["dash_kpi_size"], f"{self.config['dataset']['patch_size']}×{self.config['dataset']['patch_size']}")
        k[5].metric(t["dash_kpi_crs"], self.config['dataset']['crs'].split(':')[1])

        st.markdown("---")

        # ── Split summary ──
        st.markdown(t["dash_split_title"])
        sc = st.columns(3)
        sc[0].metric(t["dash_split_train"], f"{len(self.splits.get('train',[]))} {t['dash_split_patches']}", t["dash_split_folds13"])
        sc[1].metric(t["dash_split_val"], f"{len(self.splits.get('val',[]))} {t['dash_split_patches']}", t["dash_split_fold4"])
        sc[2].metric(t["dash_split_test"], f"{len(self.splits.get('test',[]))} {t['dash_split_patches']}", t["dash_split_fold5"])

        st.markdown("---")

        # ── Class Distribution ──
        col_chart, col_legend = st.columns([2, 1])
        with col_chart:
            st.markdown(t["dash_dist_title"])
            # Count pixels per class across ALL patches (sample a few for speed)
            sample_pids = self.patch_ids[:20]
            class_pixels = {}
            for pid in sample_pids:
                tgt = load_target(pid)
                if tgt is not None:
                    lbl = tgt[0]
                    for uid in np.unique(lbl):
                        uid = int(uid)
                        if uid in self.config.get('ignore_classes', []):
                            continue
                        class_pixels[uid] = class_pixels.get(uid, 0) + int(np.sum(lbl == uid))

            if class_pixels:
                df = pd.DataFrame([
                    {"Class ID": cid, "Crop": self.classes.get(cid, str(cid)), "Pixels": cnt}
                    for cid, cnt in class_pixels.items()
                ]).sort_values("Pixels", ascending=True)
                colors = [CROP_COLORS.get(int(r['Class ID']), '#888') for _, r in df.iterrows()]
                fig = px.bar(df, x='Pixels', y='Crop', orientation='h',
                             color='Crop', color_discrete_sequence=colors)
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                    font_color='#e8f0fe', showlegend=False, height=450, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig, use_container_width=True)

        with col_legend:
            st.markdown(t["dash_legend_title"])
            for cid_str, name in self.classes.items():
                cid = int(cid_str)
                if cid in self.config.get('ignore_classes', []):
                    continue
                clr = CROP_COLORS.get(cid, '#888')
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:8px;margin:3px 0">'
                    f'<div style="width:14px;height:14px;border-radius:3px;background:{clr};'
                    f'border:1px solid rgba(255,255,255,0.15)"></div>'
                    f'<span style="font-size:0.82rem;color:#e8f0fe">{cid}: {name}</span></div>',
                    unsafe_allow_html=True)

        st.markdown("---")

        # ── Quick Results ──
        st.markdown(t["dash_perf_title"])
        models_to_show = [
            ("rf_", "Random Forest", "#42a5f5"),
            ("lgbm_", "LightGBM", "#26a69a"),
            ("unet_", "U-Net", "#7c4dff"),
            ("resunet_", "Res-UNet", "#ff7043"),
            ("transunet_", "TransUNet", "#00bfa5")
        ]
        
        # Display each model in a clean grid
        for prefix, mname, clr in models_to_show:
            sm = load_metrics(prefix)
            if sm:
                c1, c2, c3, c4, c5 = st.columns([2.5,1,1,1,1])
                with c1:
                    st.markdown(f'<span style="color:{clr};font-weight:700;font-size:1rem">{mname}</span>', unsafe_allow_html=True)
                c2.metric(t["dash_perf_acc"], f"{sm['overall_accuracy']:.1%}")
                c3.metric(t["dash_perf_miou"], f"{sm['miou']:.3f}")
                c4.metric(t["dash_perf_wf1"], f"{sm['weighted_f1']:.3f}")
                c5.metric(t["dash_perf_mf1"], f"{sm['macro_f1']:.3f}")

        st.markdown("---")

        # ── Pipeline Summary ──
        st.markdown("---")
        st.markdown(t["dash_pipe_title"])
        
        st.markdown("""
```mermaid
graph LR
    A[Sentinel-2 L2A<br>46 Dates x 10 Bands<br>EPSG:2154] --> B(Data Preprocessing)
    B --> C(Feature Engineering<br>Temporal Stats & Indices)
    C --> D{Modeling Suite}
    D -->|Machine Learning| E(Random Forest & LightGBM)
    D -->|Deep Learning| F(U-Net, Res-UNet, TransUNet)
    E --> G[Crop Map Output]
    F --> G
    
    style A fill:#0d1b30,stroke:#1a3560,color:#42a5f5
    style B fill:#1e293b,stroke:#334155,color:#f8fafc
    style C fill:#1e293b,stroke:#334155,color:#f8fafc
    style D fill:#1e293b,stroke:#334155,color:#f8fafc
    style E fill:#0d1b30,stroke:#1a3560,color:#26a69a
    style F fill:#0d1b30,stroke:#1a3560,color:#7c4dff
    style G fill:#0d1b30,stroke:#4caf50,color:#4caf50
```
        """)

def render():
    DatasetDashboard().render()
