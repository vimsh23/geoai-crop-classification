"""
model_comparison.py — Deep model benchmarking: radar, bar charts, confusion matrices, per-class analysis.
"""
import os, json, numpy as np, pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
import streamlit as st
from .common import *
from i18n import translations

class ModelComparison:
    def __init__(self):
        self.config = load_config()
        self.classes = self.config['classes']

    def render_overview(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        results = {}
        for prefix, name in [("rf_", "Random Forest"), ("lgbm_", "LightGBM"), ("unet_", "U-Net"), ("resunet_", "Res-UNet"), ("transunet_", "TransUNet")]:
            sm = load_metrics(prefix)
            if sm:
                results[name] = sm

        if not results:
            st.info(t["mod_no_metrics"])
            return

        # Summary cards
        for name, vals in results.items():
            is_rf = "Forest" in name or "LightGBM" in name
            clr = "#42a5f5" if is_rf else "#7c4dff"
            st.markdown(f'<span style="color:{clr};font-weight:700;font-size:1.1rem">{name}</span>',
                        unsafe_allow_html=True)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric(t["mod_oa"], f"{vals['overall_accuracy']:.2%}")
            m2.metric(t["mod_miou"], f"{vals['miou']:.4f}")
            m3.metric(t["mod_wf1"], f"{vals['weighted_f1']:.4f}")
            m4.metric(t["mod_mf1"], f"{vals['macro_f1']:.4f}")
            st.markdown("")

        if len(results) >= 2:
            st.markdown("---")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(t["mod_radar"])
                labels = [t["mod_oa"], t["mod_wf1"], t["mod_mf1"], t["mod_miou"]]
                fig = go.Figure()
                for (name, vals), color in zip(results.items(), ['#42a5f5', '#26a69a', '#7c4dff', '#ff7043', '#00bfa5']):
                    v = [vals['overall_accuracy'], vals['weighted_f1'], vals['macro_f1'], vals['miou']]
                    fig.add_trace(go.Scatterpolar(
                        r=v + [v[0]], theta=labels + [labels[0]],
                        fill='toself', name=name, line_color=color, opacity=0.2, line_width=3
                    ))
                fig.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0,1], color='#555'),
                               bgcolor='rgba(0,0,0,0)', angularaxis=dict(color='#7a9cc6')),
                    paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                    legend=dict(bgcolor='rgba(0,0,0,0)'), height=450, margin=dict(t=30,b=30)
                )
                st.plotly_chart(fig, use_container_width=True)

            with r2:
                st.markdown(t["mod_heatmap"])
                dfs = {}
                for prefix, name in [("rf_", "Random Forest"), ("lgbm_", "LightGBM"), ("unet_", "U-Net"), ("resunet_", "Res-UNet"), ("transunet_", "TransUNet")]:
                    pcm = load_per_class_metrics(prefix)
                    if pcm is not None:
                        exc = ['accuracy', 'macro avg', 'weighted avg']
                        dfs[name] = pcm[~pcm.index.isin(exc)][['f1-score']].rename(columns={'f1-score': name})
                if len(dfs) >= 2:
                    # Outer join to include ALL crops, even those with NaN in some models
                    merged = list(dfs.values())[0]
                    for df in list(dfs.values())[1:]:
                        merged = merged.join(df, how='outer')
                    
                    # Fill NaNs with 0 for heatmap
                    merged = merged.fillna(0.0)
                    
                    fig2 = px.imshow(merged.values, x=merged.columns.tolist(), y=merged.index.tolist(),
                                     color_continuous_scale='Turbo', text_auto='.2f', aspect='auto')
                    
                    fig2.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                                       height=450, margin=dict(t=30,b=30,l=10,r=10))
                    st.plotly_chart(fig2, use_container_width=True)

            # Model comparison table
            st.markdown("---")
            st.markdown(t["mod_detailed"])
            rf_sm = results.get("Random Forest", {})
            lgbm_sm = results.get("LightGBM", {})
            unet_sm = results.get("U-Net", {})
            resunet_sm = results.get("Res-UNet", {})
            transunet_sm = results.get("TransUNet", {})
            comp_data = {
                t["mod_metric"]: [t["mod_oa"], t["mod_wf1"], t["mod_mf1"], t["mod_miou"],
                            t["mod_arch"], t["mod_train"], t["mod_spatial"], t["mod_interp"]],
                "Random Forest": [f"{rf_sm.get('overall_accuracy',0):.2%}", f"{rf_sm.get('weighted_f1',0):.4f}",
                                  f"{rf_sm.get('macro_f1',0):.4f}", f"{rf_sm.get('miou',0):.4f}",
                                  t["mod_t_ensemble"], t["mod_t_pixel"], t["mod_t_none"], t["mod_t_feat"]],
                "LightGBM": [f"{lgbm_sm.get('overall_accuracy',0):.2%}" if lgbm_sm else t["mod_not_trained"], 
                             f"{lgbm_sm.get('weighted_f1',0):.4f}" if lgbm_sm else "-",
                             f"{lgbm_sm.get('macro_f1',0):.4f}" if lgbm_sm else "-", 
                             f"{lgbm_sm.get('miou',0):.4f}" if lgbm_sm else "-",
                             t["mod_t_gbt"], t["mod_t_pixel"], t["mod_t_none"], t["mod_t_feat"]],
                "U-Net": [f"{unet_sm.get('overall_accuracy',0):.2%}", f"{unet_sm.get('weighted_f1',0):.4f}",
                          f"{unet_sm.get('macro_f1',0):.4f}", f"{unet_sm.get('miou',0):.4f}",
                          t["mod_t_cnn"], t["mod_t_e2e"], t["mod_t_full"], t["mod_t_limit"]],
                "Res-UNet": [f"{resunet_sm.get('overall_accuracy',0):.2%}" if resunet_sm else t["mod_not_trained"], 
                             f"{resunet_sm.get('weighted_f1',0):.4f}" if resunet_sm else "-",
                             f"{resunet_sm.get('macro_f1',0):.4f}" if resunet_sm else "-", 
                             f"{resunet_sm.get('miou',0):.4f}" if resunet_sm else "-",
                             t["mod_t_res"], t["mod_t_deep"], t["mod_t_full"], t["mod_t_limit"]],
                "TransUNet": [f"{transunet_sm.get('overall_accuracy',0):.2%}" if transunet_sm else t["mod_not_trained"], 
                             f"{transunet_sm.get('weighted_f1',0):.4f}" if transunet_sm else "-",
                             f"{transunet_sm.get('macro_f1',0):.4f}" if transunet_sm else "-", 
                             f"{transunet_sm.get('miou',0):.4f}" if transunet_sm else "-",
                             t["mod_t_hybrid"], t["mod_t_global"], t["mod_t_full_glob"], t["mod_t_limit"]]
            }
            st.dataframe(pd.DataFrame(comp_data), use_container_width=True, hide_index=True)


    def render_confusion(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        model = st.selectbox(t["mod_model_select"], ["Random Forest", "LightGBM", "U-Net", "Res-UNet", "TransUNet"], key="cm_model")
        prefix = "rf_" if "Forest" in model else ("lgbm_" if "Light" in model else ("resunet_" if "Res" in model else ("transunet_" if "Trans" in model else "unet_")))
        cm = load_confusion_matrix(prefix)
        if cm is None:
            st.info(t["mod_cm_not_found"])
            return

        fig = px.imshow(cm.values, x=cm.columns.tolist(), y=cm.index.tolist(),
                        color_continuous_scale='Turbo', text_auto=True,
                        title=f"{model} — {t['mod_confusion']}")
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font_color='#e8f0fe',
                          height=800, xaxis_title=t["mod_pred"], yaxis_title=t["mod_actual"])
        st.plotly_chart(fig, use_container_width=True)

        # Most confused pairs
        st.markdown(t["mod_confused_pairs"])
        cm_vals = cm.values.copy()
        np.fill_diagonal(cm_vals, 0)
        flat_idx = np.argsort(cm_vals.flatten())[::-1][:10]
        rows = []
        for idx in flat_idx:
            r, cc = divmod(idx, cm_vals.shape[1])
            if cm_vals[r, cc] > 0:
                rows.append({
                    t["mod_true_cls"]: cm.index[r],
                    t["mod_pred_as"]: cm.columns[cc],
                    t["mod_mis_pix"]: int(cm_vals[r, cc])
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


    def render_per_class(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        model = st.selectbox(t["mod_model_select"], ["Random Forest", "LightGBM", "U-Net", "Res-UNet", "TransUNet"], key="pcd_model")
        prefix = "rf_" if "Forest" in model else ("lgbm_" if "Light" in model else ("resunet_" if "Res" in model else ("transunet_" if "Trans" in model else "unet_")))
        report = load_classification_report(prefix)
        if report is None:
            st.info(t["mod_report_not_found"])
            return

        # Build per-class table with IoU
        rows = []
        for key, vals in report.items():
            if key in ['accuracy', 'macro avg', 'weighted avg'] or not isinstance(vals, dict):
                continue
            rows.append({
                t["mod_crop"]: key,
                t["mod_prec"]: f"{vals.get('precision',0):.3f}",
                t["mod_rec"]: f"{vals.get('recall',0):.3f}",
                t["mod_f1"]: f"{vals.get('f1-score',0):.3f}",
                t["mod_iou_table"]: f"{vals.get('iou',0):.3f}",
                t["mod_sup"]: int(vals.get('support',0))
            })
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Best and worst classes
        if rows:
            st.markdown("---")
            def get_f1(x):
                return float(x[t['mod_f1']])
                
            best = max(rows, key=get_f1)
            worst = min(rows, key=get_f1)
            b1, b2 = st.columns(2)
            b1.success(t["mod_best"].replace("{crop}", best[t['mod_crop']]).replace("{f1}", str(best[t['mod_f1']])))
            b2.error(t["mod_worst"].replace("{crop}", worst[t['mod_crop']]).replace("{f1}", str(worst[t['mod_f1']])))

    def render(self):
        t = translations.get(st.session_state.get('language', 'English'), translations['English'])
        st.markdown(t["mod_title"])

        sub = st.radio(t["mod_view"], [t["mod_overview"], t["mod_confusion"], t["mod_per_class"]], horizontal=True, key="mc_view")

        if sub == t["mod_overview"]:
            self.render_overview()
        elif sub == t["mod_confusion"]:
            self.render_confusion()
        else:
            self.render_per_class()

def render():
    ModelComparison().render()
