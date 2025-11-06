import json
import os
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# Imports from src/
# The app is run from the repo root. Ensure `src/` is importable.
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from preprocess import preprocess_features, FEATURE_COLS  # noqa: E402
try:
    # map_columns helps align uploaded CSV headers to canonical names used in training
    from preprocess import map_columns  # type: ignore  # noqa: E402
except Exception:  # pragma: no cover
    map_columns = None


# -----------------------------------------------------------------------------
# Page configuration and minimal styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Health Dashboard – Heart Risk",
    page_icon="❤️",
    layout="centered",
)

CUSTOM_CSS = """
<style>
/* Compact top padding */
.block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* Result card styling - Enhanced */
.result-card {
  border: 2px solid rgba(200,200,200,0.3);
  border-radius: 16px;
  padding: 24px;
  background: linear-gradient(135deg, rgba(255,255,255,0.1) 0%, rgba(255,255,255,0.05) 100%);
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
  margin: 20px 0;
  transition: transform 0.3s ease;
}

.result-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 12px rgba(0,0,0,0.15);
}

.metric-title { 
  font-weight: 600; 
  opacity: 0.9; 
  font-size: 1rem;
  margin-bottom: 8px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.metric-value { 
  font-size: 2.5rem; 
  font-weight: 700; 
  margin: 12px 0;
  text-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.risk-category {
  font-size: 1.2rem;
  font-weight: 600;
  margin-top: 8px;
  padding: 8px 16px;
  border-radius: 8px;
  display: inline-block;
  background: rgba(255,255,255,0.1);
}

/* Enhanced horizontal risk bar */
.risk-bar {
  width: 100%;
  height: 20px;
  border-radius: 999px;
  background: linear-gradient(90deg, #3CB371 0%, #FFD166 50%, #EF476F 100%);
  position: relative;
  margin: 16px 0;
  box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
}

.risk-indicator {
  position: absolute;
  top: -4px;
  width: 0; 
  height: 28px;
  border-left: 4px solid white;
  box-shadow: 0 2px 4px rgba(0,0,0,0.3);
  transition: left 0.3s ease;
}

.risk-indicator::after {
  content: '';
  position: absolute;
  top: 50%;
  left: -6px;
  transform: translateY(-50%);
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: white;
  border: 2px solid currentColor;
  box-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

/* Input section styling */
.input-section {
  background: rgba(240,240,240,0.5);
  padding: 20px;
  border-radius: 12px;
  margin: 20px 0;
}

.input-group {
  margin-bottom: 16px;
}

/* Footer */
.footer-note { 
  opacity: 0.7; 
  font-size: 0.9rem; 
  text-align: center;
  padding: 20px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { 
  gap: 8px; 
  background: rgba(240,240,240,0.3);
  border-radius: 8px;
  padding: 4px;
}

.stTabs [data-baseweb="tab"] { 
  padding: 8px 16px; 
  border-radius: 6px;
  transition: all 0.2s ease;
}

.stTabs [data-baseweb="tab"]:hover {
  background: rgba(255,255,255,0.5);
}

/* Success message styling */
.success-box {
  background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
  border: 1px solid #28a745;
  border-radius: 8px;
  padding: 12px;
  margin: 16px 0;
}

/* Info boxes */
.info-box {
  background: rgba(173,216,230,0.2);
  border-left: 4px solid #2196F3;
  padding: 12px 16px;
  border-radius: 4px;
  margin: 12px 0;
}

/* Feature importance cards */
.feature-card {
  background: rgba(255,255,255,0.6);
  border-radius: 8px;
  padding: 12px;
  margin: 8px 0;
  border-left: 4px solid #2196F3;
}

/* Hide Streamlit default elements */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
.stDeployButton { display: none; }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Artifact loading
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model_path = ROOT / "model" / "model.joblib"
    scaler_path = ROOT / "model" / "scaler.joblib"
    metrics_path = ROOT / "model" / "metrics.json"

    if not model_path.exists() or not scaler_path.exists():
        return None, None, None

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    metrics = None
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    return model, scaler, metrics


model, scaler, metrics = load_artifacts()

# Enhanced header with emoji and better styling
st.markdown("""
<div style="text-align: center; padding: 20px 0;">
    <h1 style="font-size: 2.5rem; margin-bottom: 10px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">
        ❤️ Health Dashboard
    </h1>
    <p style="font-size: 1.1rem; color: #666; margin-top: 10px;">
        Heart Disease Risk Prediction using AI
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

if model is None or scaler is None:
    st.error(
        "Model artifacts not found. Please run training first from the repo root: `python src/train.py`."
    )
    st.stop()


# -----------------------------------------------------------------------------
# Sidebar: Enhanced with better organization
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📊 Dashboard Controls")
    st.markdown("---")
    
    st.markdown("#### 📝 Input Summary")
    st.caption("Live view of your current input values.")
    summary_placeholder = st.empty()

    with st.expander("ℹ️ Dataset & Model Info", expanded=False):
        st.markdown("""
        **Dataset:** UCI Heart Disease  
        **Location:** `data/heart.csv`
        """)
        if metrics:
            roc_auc = metrics.get('roc_auc', None)
            accuracy = metrics.get('accuracy', None)
            precision = metrics.get('precision', None)
            recall = metrics.get('recall', None)
            f1 = metrics.get('f1', None)
            
            st.markdown("""
            **Model Performance:**
            """)
            if roc_auc is not None:
                st.markdown(f"- 🎯 ROC AUC: `{roc_auc:.3f}`")
            if accuracy is not None:
                st.markdown(f"- ✅ Accuracy: `{accuracy:.3f}`")
            if precision is not None:
                st.markdown(f"- 📊 Precision: `{precision:.3f}`")
            if recall is not None:
                st.markdown(f"- 🔄 Recall: `{recall:.3f}`")
            if f1 is not None:
                st.markdown(f"- ⚖️ F1 Score: `{f1:.3f}`")
        else:
            st.info("Model metrics not available.")
        st.markdown("""
        **Model Type:** RandomForestClassifier  
        **Preprocessing:** StandardScaler
        """)

    # Risk threshold controls
    st.markdown("---")
    st.markdown("#### ⚙️ Risk Thresholds")
    st.caption("Adjust how risk categories are determined")
    low_default, high_default = 0.33, 0.66
    low_thr = st.slider(
        "🟢 Low Risk Threshold",
        0.0,
        0.9,
        low_default,
        0.01,
        help="Probability below this value is considered Low Risk"
    )
    high_thr = st.slider(
        "🔴 High Risk Threshold",
        0.1,
        0.99,
        max(high_default, low_thr + 0.01),
        0.01,
        help="Probability above this value is considered High Risk"
    )
    if low_thr >= high_thr:
        st.warning("⚠️ Low threshold must be less than High threshold.")
    else:
        st.success(f"✅ Risk Categories: Low (<{low_thr:.0%}), Moderate ({low_thr:.0%}-{high_thr:.0%}), High (>{high_thr:.0%})")

    st.markdown("---")
    st.markdown("#### 📦 Batch Prediction")
    st.caption("Upload a CSV file to predict multiple patients at once")
    csv_file = st.file_uploader(
        "📄 Upload CSV File",
        type=["csv"],
        help="CSV should contain columns matching the feature names"
    )
    if csv_file:
        st.info(f"✅ File uploaded: {csv_file.name}")
    batch_run = st.button("🚀 Run Batch Prediction", type="secondary", use_container_width=True)


# -----------------------------------------------------------------------------
# Input widgets – Enhanced with tooltips and better organization
# -----------------------------------------------------------------------------
st.markdown("### 📋 Enter Patient Information")
st.markdown('<div class="info-box">Enter the patient\'s health metrics below. Hover over field labels for more information.</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)

# Reasonable defaults based on typical UCI value ranges
with col1:
    st.markdown("#### 👤 Demographics & Basic Info")
    age = st.number_input(
        "👤 Age",
        min_value=18,
        max_value=100,
        value=54,
        help="Patient's age in years"
    )
    sex = st.selectbox(
        "⚧️ Sex",
        options=["Male", "Female"],
        index=0,
        help="Patient's biological sex"
    )
    
    st.markdown("#### 🩺 Vital Signs")
    trestbps = st.number_input(
        "💓 Resting Blood Pressure (mmHg)",
        min_value=80,
        max_value=220,
        value=130,
        help="Normal range: 90-120 mmHg. Resting systolic blood pressure"
    )
    chol = st.number_input(
        "🧪 Total Cholesterol (mg/dl)",
        min_value=100,
        max_value=600,
        value=246,
        help="Normal range: <200 mg/dl. Serum cholesterol level"
    )
    thalach = st.number_input(
        "❤️ Maximum Heart Rate Achieved (bpm)",
        min_value=60,
        max_value=220,
        value=150,
        help="Maximum heart rate achieved during exercise. Typical max: 220 - age"
    )

with col2:
    st.markdown("#### 🏥 Clinical Features")
    fbs = st.selectbox(
        "🍬 Fasting Blood Sugar > 120 mg/dl",
        options=["No", "Yes"],
        index=0,
        help="Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)"
    )
    exang = st.selectbox(
        "🏃 Exercise Induced Angina",
        options=["No", "Yes"],
        index=0,
        help="Exercise induced angina (1 = yes; 0 = no)"
    )
    oldpeak = st.number_input(
        "📉 ST Depression Induced by Exercise",
        min_value=0.0,
        max_value=6.0,
        value=1.0,
        step=0.1,
        help="ST depression induced by exercise relative to rest (typically 0-6)"
    )
    
    # Quick reference card
    st.markdown("---")
    st.markdown("""
    <div style="background: rgba(240,240,240,0.5); padding: 12px; border-radius: 8px; font-size: 0.85rem;">
        <strong>💡 Quick Tips:</strong><br>
        • Normal BP: 90-120 mmHg<br>
        • Normal Cholesterol: <200 mg/dl<br>
        • Max HR: ~220 - age
    </div>
    """, unsafe_allow_html=True)


def build_input_df() -> pd.DataFrame:
    """Construct a single-row DataFrame in the order of FEATURE_COLS."""
    sex_bin = 1 if sex == "Male" else 0
    fbs_bin = 1 if fbs == "Yes" else 0
    exang_bin = 1 if exang == "Yes" else 0
    row = {
        "age": age,
        "sex": sex_bin,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs_bin,
        "thalach": thalach,
        "exang": exang_bin,
        "oldpeak": oldpeak,
    }
    # ensure column order matches trained model input
    return pd.DataFrame([[row[c] for c in FEATURE_COLS]], columns=FEATURE_COLS)


# Keep sidebar summary synced
summary_placeholder.json({c: v for c, v in build_input_df().iloc[0].items()})


# -----------------------------------------------------------------------------
# Prediction & Results
# -----------------------------------------------------------------------------
st.markdown("---")
predict_clicked = st.button("🔮 Predict Heart Disease Risk", type="primary", use_container_width=True)

if predict_clicked:
    try:
        user_df = build_input_df()
        X_proc, _, _ = preprocess_features(user_df, fit_scaler=False, scaler=scaler)
        proba = float(model.predict_proba(X_proc)[0, 1])

        # Risk bucket thresholds
        if proba < low_thr:
            bucket, color = "Low", "#3CB371"
        elif proba <= high_thr:
            bucket, color = "Moderate", "#FFD166"
        else:
            bucket, color = "High", "#EF476F"

        # Enhanced Result card
        risk_emoji = "🟢" if bucket == "Low" else "🟡" if bucket == "Moderate" else "🔴"
        st.markdown(
            f"""
            <div class="result-card">
                <div class="metric-title">Predicted Heart Disease Risk</div>
                <div class="metric-value" style="color:{color}">{proba*100:.1f}%</div>
                <div class="risk-category" style="color:{color}; border: 2px solid {color};">
                    {risk_emoji} {bucket} Risk
                </div>
                <div class="risk-bar" style="margin-top:16px;">
                    <div class="risk-indicator" style="left:{proba*100:.0f}%; color:{color};"></div>
                </div>
                <div style="margin-top: 12px; font-size: 0.9rem; opacity: 0.8;">
                    This prediction is based on the patient's input features and model analysis.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        
        # Success message
        st.markdown(f'<div class="success-box">✅ Prediction completed successfully!</div>', unsafe_allow_html=True)

        # Tabs: Global importances and SHAP with enhanced visualizations
        tabs = st.tabs(["📊 Feature Importance", "🔍 SHAP Explanation", "📈 Model Performance"])
        
        with tabs[0]:
            st.markdown("### Global Feature Importance")
            st.markdown("These are the overall importance of each feature in the model.")
            if hasattr(model, "feature_importances_"):
                fi = model.feature_importances_
                fi_df = pd.DataFrame({
                    "Feature": FEATURE_COLS, 
                    "Importance": fi
                }).sort_values("Importance", ascending=True)
                
                # Create interactive Plotly bar chart
                fig = px.bar(
                    fi_df,
                    x="Importance",
                    y="Feature",
                    orientation='h',
                    color="Importance",
                    color_continuous_scale="Blues",
                    title="Feature Importance (Higher = More Important)",
                    labels={"Importance": "Importance Score", "Feature": "Feature Name"}
                )
                fig.update_layout(
                    height=400,
                    showlegend=False,
                    xaxis_title="Importance Score",
                    yaxis_title="",
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Display top features
                top_features = fi_df.tail(3)
                st.markdown("**Top 3 Most Important Features:**")
                for idx, row in top_features.iterrows():
                    st.markdown(f"• **{row['Feature']}**: {row['Importance']:.4f}")
            else:
                st.info("This model does not expose feature_importances_.")

        with tabs[2]:
            st.markdown("### Model Performance Metrics")
            if metrics:
                # Create metrics visualization
                metric_names = ["Accuracy", "ROC AUC", "Precision", "Recall", "F1 Score"]
                metric_values = [
                    metrics.get("accuracy", 0),
                    metrics.get("roc_auc", 0),
                    metrics.get("precision", 0),
                    metrics.get("recall", 0),
                    metrics.get("f1", 0)
                ]
                
                # Gauge chart for ROC AUC
                fig_roc = go.Figure(go.Indicator(
                    mode = "gauge+number+delta",
                    value = metrics.get("roc_auc", 0) * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "ROC AUC Score"},
                    delta = {'reference': 70},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 50], 'color': "lightgray"},
                            {'range': [50, 70], 'color': "gray"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                fig_roc.update_layout(height=300)
                st.plotly_chart(fig_roc, use_container_width=True)
                
                # Bar chart for all metrics
                metrics_df = pd.DataFrame({
                    "Metric": metric_names,
                    "Score": metric_values
                })
                
                fig_metrics = px.bar(
                    metrics_df,
                    x="Metric",
                    y="Score",
                    color="Score",
                    color_continuous_scale="Greens",
                    title="Model Performance Metrics",
                    labels={"Score": "Score", "Metric": "Metric"}
                )
                fig_metrics.update_layout(
                    height=350,
                    showlegend=False,
                    yaxis_range=[0, 1],
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)"
                )
                fig_metrics.update_traces(texttemplate='%{y:.3f}', textposition='outside')
                st.plotly_chart(fig_metrics, use_container_width=True)
                
                # Display confusion matrix if available
                if "confusion_matrix" in metrics:
                    st.markdown("#### Confusion Matrix")
                    cm = np.array(metrics["confusion_matrix"])
                    fig_cm = px.imshow(
                        cm,
                        labels=dict(x="Predicted", y="Actual", color="Count"),
                        x=["No Disease", "Disease"],
                        y=["No Disease", "Disease"],
                        color_continuous_scale="Blues",
                        text_auto=True
                    )
                    fig_cm.update_layout(height=350)
                    st.plotly_chart(fig_cm, use_container_width=True)
            else:
                st.info("Model metrics not available.")
        
        with tabs[1]:
            # Check if SHAP is installed
            try:
                import shap
                shap_available = True
            except ImportError:
                shap_available = False
                st.warning("SHAP is not installed. Install it with: `pip install shap`")
                st.info("Showing global feature importances only.")
            
            if shap_available:
                try:
                    # Convert DataFrame to numpy array for SHAP
                    # Ensure it's 2D array with shape (n_samples, n_features)
                    if isinstance(X_proc, pd.DataFrame):
                        X_proc_array = X_proc.values  # Convert to numpy array
                    else:
                        X_proc_array = X_proc
                    
                    # Ensure it's 2D (even for single sample)
                    if X_proc_array.ndim == 1:
                        X_proc_array = X_proc_array.reshape(1, -1)
                    
                    # Use TreeExplainer for tree-based models (RandomForest)
                    explainer = shap.TreeExplainer(model)
                    sv_list = explainer.shap_values(X_proc_array)
                    
                    # Handle binary classifier output (list of arrays)
                    # For binary classification, SHAP returns [class0_values, class1_values]
                    if isinstance(sv_list, list) and len(sv_list) == 2:
                        # Use class 1 (disease) values - extract first sample
                        sv = np.array(sv_list[1])
                        if sv.ndim == 2:
                            sv = sv[0]  # Take first row (first sample)
                        elif sv.ndim > 2:
                            sv = sv.flatten()[:len(FEATURE_COLS)]  # Flatten and take first n features
                    else:
                        sv = np.array(sv_list)
                        if sv.ndim == 2:
                            sv = sv[0]  # Take first row
                        elif sv.ndim > 2:
                            sv = sv.flatten()[:len(FEATURE_COLS)]
                    
                    # Ensure sv is 1D array
                    sv = np.array(sv).flatten()
                    
                    # If still wrong length, might be concatenated - try to reshape
                    if len(sv) == len(FEATURE_COLS) * 2:
                        # Likely both classes concatenated, take second half (class 1)
                        sv = sv[len(FEATURE_COLS):]
                    elif len(sv) > len(FEATURE_COLS):
                        # Take first n features
                        sv = sv[:len(FEATURE_COLS)]
                    
                    # Final check
                    if len(sv) != len(FEATURE_COLS):
                        st.warning(f"SHAP values length ({len(sv)}) doesn't match features ({len(FEATURE_COLS)}). Using first {min(len(sv), len(FEATURE_COLS))} values.")
                        min_len = min(len(sv), len(FEATURE_COLS))
                        sv = sv[:min_len]
                        feature_cols_used = FEATURE_COLS[:min_len]
                    else:
                        feature_cols_used = FEATURE_COLS
                    
                    # Create enhanced SHAP visualization
                    shap_df = pd.DataFrame({
                        "Feature": feature_cols_used,
                        "Contribution": sv,
                        "Absolute Contribution": np.abs(sv),
                    }).sort_values("Absolute Contribution", ascending=True)
                    
                    # Interactive Plotly chart with color coding (positive/negative)
                    fig = px.bar(
                        shap_df,
                        x="Contribution",
                        y="Feature",
                        orientation='h',
                        color="Contribution",
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        title="SHAP Values: Feature Contributions to This Prediction",
                        labels={"Contribution": "SHAP Value", "Feature": "Feature Name"}
                    )
                    fig.add_vline(x=0, line_dash="dash", line_color="gray", opacity=0.5)
                    fig.update_layout(
                        height=400,
                        showlegend=False,
                        xaxis_title="SHAP Value (Red = Increases Risk, Blue = Decreases Risk)",
                        yaxis_title="",
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown("""
                    <div class="info-box">
                        <strong>Understanding SHAP Values:</strong><br>
                        • <span style="color: #EF476F;">Positive values (red)</span>: Increase risk of heart disease<br>
                        • <span style="color: #2196F3;">Negative values (blue)</span>: Decrease risk of heart disease<br>
                        • Larger absolute values indicate stronger influence on this prediction
                    </div>
                    """, unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"SHAP computation failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
                    st.info("This may be due to numba compatibility issues. Try: `pip install --upgrade shap numba`")
                    st.info("Showing global feature importances only.")

    except Exception as e:
        st.error(f"Prediction failed: {e}")


# -----------------------------------------------------------------------------
# Batch CSV prediction
# -----------------------------------------------------------------------------
if csv_file and batch_run:
    try:
        df_in = pd.read_csv(csv_file)
        # Normalize column names to match training
        if map_columns is not None:
            df_in = map_columns(df_in)
        # Ensure all expected features exist; create missing ones to allow imputation
        for c in FEATURE_COLS:
            if c not in df_in.columns:
                df_in[c] = np.nan
        # Preprocess using saved scaler
        Xb, _, _ = preprocess_features(df_in, fit_scaler=False, scaler=scaler)
        probas = model.predict_proba(Xb)[:, 1]
        labels = (probas >= 0.5).astype(int)

        def bucketize(p: float) -> str:
            if p < low_thr:
                return "Low"
            if p <= high_thr:
                return "Moderate"
            return "High"

        out = df_in.copy()
        out["probability"] = probas
        out["prediction"] = labels
        out["risk_bucket"] = [bucketize(p) for p in probas]

        st.markdown("### 📊 Batch Prediction Results")
        st.success(f"✅ Successfully processed {len(out)} patient(s)")
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Patients", len(out))
        with col2:
            st.metric("Low Risk", sum(out["risk_bucket"] == "Low"))
        with col3:
            st.metric("Moderate Risk", sum(out["risk_bucket"] == "Moderate"))
        with col4:
            st.metric("High Risk", sum(out["risk_bucket"] == "High"))
        
        # Risk distribution chart
        risk_counts = out["risk_bucket"].value_counts()
        fig_dist = px.pie(
            values=risk_counts.values,
            names=risk_counts.index,
            title="Risk Distribution",
            color_discrete_map={"Low": "#3CB371", "Moderate": "#FFD166", "High": "#EF476F"}
        )
        fig_dist.update_layout(height=300)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Results table
        st.markdown("#### Detailed Results")
        st.dataframe(
            out.head(100),
            use_container_width=True,
            height=400
        )
        
        if len(out) > 100:
            st.info(f"Showing first 100 of {len(out)} results. Download full results below.")

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Results CSV",
            data=csv_bytes,
            file_name="batch_predictions.csv",
            mime="text/csv",
            use_container_width=True
        )
    except Exception as e:
        st.error(f"Batch prediction failed: {e}")
        st.caption("Your CSV should include headers that map to: " + ", ".join(FEATURE_COLS) + ". Common variants are auto-mapped when possible.")


# -----------------------------------------------------------------------------
# Footer
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    """
    <div class="footer-note">
      <p style="margin-bottom: 8px;">
        <strong>❤️ Health Dashboard</strong> - Heart Disease Risk Prediction
      </p>
      <p style="font-size: 0.85rem; margin-bottom: 4px;">
        Built by <strong>vineeth</strong> | Powered by Machine Learning
      </p>
      <p style="font-size: 0.8rem; opacity: 0.6;">
        ⚠️ For educational purposes only. This is not a medical device and should not be used for medical diagnosis.
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)
