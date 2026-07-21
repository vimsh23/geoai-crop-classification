"""
prediction_portal.py — Upload → Validate → Preprocess → Predict → Visualize → Export.
Uses existing trained models and preprocessing pipeline.
"""
import os, sys, pickle, time, json, numpy as np, pandas as pd
import plotly.express as px
import streamlit as st
from io import BytesIO
from PIL import Image
from .common import *
from i18n import translations

# ── Windows DLL fix: register torch lib dir before importing torch ──────────
# Streamlit runs scripts in a subprocess where the DLL search path is not
# automatically extended to include torch's bundled libraries (c10.dll, etc.)
# os.add_dll_directory() is only available on Windows Python 3.8+
def _register_torch_dll_dirs():
    try:
        import site
        for sp in site.getsitepackages():
            torch_lib = os.path.join(sp, "torch", "lib")
            if os.path.isdir(torch_lib):
                os.add_dll_directory(torch_lib)
    except (AttributeError, OSError):
        pass  # Not on Windows or already registered

if sys.platform == "win32":
    _register_torch_dll_dirs()

# Pre-import torch once at module level so DLL path is resolved at startup
try:
    import torch
    _TORCH_AVAILABLE = True
except OSError:
    _TORCH_AVAILABLE = False

class PredictionPortal:
    def __init__(self):
        self.config = load_config()
        self.classes = self.config['classes']
        self.uploaded_file = None
        self.s2_data = None
        self.features = None
        self.preds = None
        self.confidence = None
        self.preprocess_time = 0
        self.inf_time = 0

    def render_upload(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown("---")
        self.uploaded_file = st.file_uploader(t["pred_upload"], type=['npy'])

        if self.uploaded_file is None:
            st.info(t["pred_upload_info"])
            st.markdown(t["pred_supported"])
            return False
        return True

    def validate_upload(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        try:
            self.s2_data = np.load(BytesIO(self.uploaded_file.read()))
        except Exception as e:
            st.error(f"Failed to load file: {e}")
            return False

        st.markdown(t["pred_val_title"])
        expected = (46, 10, 128, 128)
        v1, v2, v3 = st.columns(3)
        v1.metric(t["pred_shape"], str(self.s2_data.shape))
        v2.metric(t["pred_dtype"], str(self.s2_data.dtype))
        v3.metric(t["pred_size"], f"{self.s2_data.nbytes / 1e6:.1f}")

        if self.s2_data.shape != expected:
            st.error(f"Shape mismatch: expected {expected}, got {self.s2_data.shape}")
            return False
            
        st.success(t["pred_val_success"])
        return True

    def run_inference(self, model_choice):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        with st.spinner(t["pred_run_pre"]):
            start = time.time()
            from src.preprocessing import DataPreprocessor
            preprocessor = DataPreprocessor(self.config)
            self.features = preprocessor.prepare_patch_features(self.s2_data)
            self.preprocess_time = time.time() - start

        st.success(f"{t['pred_pre_complete']} ({self.preprocess_time:.2f}s) — Shape: {self.features.shape}")

        with st.spinner(t["pred_run_inf"].replace("inference", f"{model_choice} inference").replace("推論", f"{model_choice} 推論")):
            inf_start = time.time()

            if model_choice == "Random Forest":
                model_path = os.path.join(BASE, self.config['paths']['model_dir'], "rf_model.pkl")
                if not os.path.exists(model_path):
                    st.error("RF model not found. Train it first.")
                    return False
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                C, H, W = self.features.shape
                X = self.features.reshape(C, -1).T
                self.preds = model.predict(X).reshape(H, W)
                proba = model.predict_proba(X)
                self.confidence = np.max(proba, axis=1).reshape(H, W)

            elif model_choice == "LightGBM":
                model_path = os.path.join(BASE, self.config['paths']['model_dir'], "lgbm_model.pkl")
                if not os.path.exists(model_path):
                    st.error("LightGBM model not found. Train it first.")
                    return False
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
                C, H, W = self.features.shape
                X = self.features.reshape(C, -1).T
                self.preds = model.predict(X).reshape(H, W)
                proba = model.predict_proba(X)
                self.confidence = np.max(proba, axis=1).reshape(H, W)

            elif model_choice == "U-Net":
                from src.model import get_unet_model
                model_path = os.path.join(BASE, self.config['paths']['model_dir'], "unet_best.pth")
                if not os.path.exists(model_path):
                    st.error("U-Net model not found. Train it first.")
                    return False
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = get_unet_model(self.config, self.features.shape[0]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    x = torch.tensor(self.features, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(x)
                    proba_t = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                    self.preds = np.argmax(proba_t, axis=0)
                    self.confidence = np.max(proba_t, axis=0)
                    
            elif model_choice == "Res-UNet":
                from src.model import get_resunet_model
                model_path = os.path.join(BASE, self.config['paths']['model_dir'], "resunet_best.pth")
                if not os.path.exists(model_path):
                    st.error("Res-UNet model not found. Train it first by modifying train.py!")
                    return False
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = get_resunet_model(self.config, self.features.shape[0]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    x = torch.tensor(self.features, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(x)
                    proba_t = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                    self.preds = np.argmax(proba_t, axis=0)
                    self.confidence = np.max(proba_t, axis=0)
                    
            elif model_choice == "TransUNet":
                from src.model import get_transunet_model
                model_path = os.path.join(BASE, self.config['paths']['model_dir'], "transunet_best.pth")
                if not os.path.exists(model_path):
                    st.error("TransUNet model not found. Train it first by running: python3 src/train.py --model transunet")
                    return False
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = get_transunet_model(self.config, self.features.shape[0]).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                with torch.no_grad():
                    x = torch.tensor(self.features, dtype=torch.float32).unsqueeze(0).to(device)
                    outputs = model(x)
                    proba_t = torch.softmax(outputs, dim=1).squeeze(0).cpu().numpy()
                    self.preds = np.argmax(proba_t, axis=0)
                    self.confidence = np.max(proba_t, axis=0)

            self.inf_time = time.time() - inf_start

        st.success(f"{t['pred_inf_complete']} ({self.inf_time:.2f}s)")
        return True

    def render_results(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown("---")
        st.markdown(t["pred_res_title"])
        t1, t2, t3 = st.columns(3)
        t1.metric(t["pred_pre"], f"{self.preprocess_time:.2f}s")
        t2.metric(t["pred_inf"], f"{self.inf_time:.2f}s")
        t3.metric(t["pred_mean_conf"], f"{np.mean(self.confidence):.1%}")

        # Visualizations
        c1, c2, c3 = st.columns(3)
            
        with c1:
            st.markdown(t["pred_rgb_comp"])
            rgb = make_rgb(self.s2_data)
            st.image(rgb, use_container_width=True)

        # Calculate Crop statistics first for legend
        unique_preds = np.unique(self.preds)
        rows = []
        for uid in unique_preds:
            uid = int(uid)
            mask = self.preds == uid
            rows.append({
                "Class ID": uid,
                "Crop": self.classes.get(str(uid), self.classes.get(uid, f"Class {uid}")),
                "Pixels": int(np.sum(mask)),
                "Coverage %": float(np.sum(mask) / self.preds.size * 100),
                "Avg Confidence": float(np.mean(self.confidence[mask]))
            })
        
        def sort_by_pixels(x):
            return x['Pixels']
            
        sorted_rows = sorted(rows, key=sort_by_pixels, reverse=True)

        with c2:
            st.markdown(t["pred_crop_map"])
            pred_rgb = make_label_rgb(self.preds)
            st.image(pred_rgb, use_container_width=True)
            
            st.markdown(f"<div style='margin-top:10px; font-size:0.85rem'><b>{t['pred_dom_cls']}</b><br>", unsafe_allow_html=True)
            for row in sorted_rows:
                if row['Pixels'] > 0:
                    clr = CROP_COLORS.get(row['Class ID'], '#ffffff')
                    st.markdown(f"<span style='color:{clr}'>■</span> {row['Crop']} ({row['Coverage %']:.1f}%)", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with c3:
            st.markdown(t["pred_conf_map"])
            fig = px.imshow(self.confidence, color_continuous_scale='RdYlGn', zmin=0, zmax=1)
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                              coloraxis_colorbar_title="Conf", height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown(t["pred_crop_stats"])
        df_stats = pd.DataFrame(sorted_rows)
        df_display = df_stats.copy()
        
        def format_cov(x):
            return f"{x:.1f}"
            
        def format_conf(x):
            return f"{x:.3f}"
            
        df_display['Coverage %'] = df_display['Coverage %'].apply(format_cov)
        df_display['Avg Confidence'] = df_display['Avg Confidence'].apply(format_conf)
        st.dataframe(df_display, use_container_width=True, hide_index=True)

        self.render_exports(rows)

    def render_exports(self, rows):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown("---")
        st.markdown(t["pred_exp_res"])
        ec1, ec2, ec3 = st.columns(3)
        with ec1:
            pred_buf = BytesIO()
            np.save(pred_buf, self.preds)
            st.download_button(t["pred_exp_pred"], pred_buf.getvalue(),
                                file_name="prediction.npy", mime="application/octet-stream")
        with ec2:
            conf_buf = BytesIO()
            np.save(conf_buf, self.confidence)
            st.download_button(t["pred_exp_conf"], conf_buf.getvalue(),
                                file_name="confidence.npy", mime="application/octet-stream")
        with ec3:
            csv_buf = pd.DataFrame(rows).to_csv(index=False)
            st.download_button(t["pred_exp_stat"], csv_buf,
                                file_name="prediction_stats.csv", mime="text/csv")

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["pred_title"])
        st.markdown(t["pred_desc"], unsafe_allow_html=True)

        if not self.render_upload():
            return
            
        if not self.validate_upload():
            return

        model_choice = st.selectbox(t["pred_sel_model"], ["Random Forest", "LightGBM", "U-Net", "Res-UNet", "TransUNet"], key="pred_model")

        if st.button(t["pred_run_pred"], type="primary"):
            if self.run_inference(model_choice):
                self.render_results()

def render():
    PredictionPortal().render()
