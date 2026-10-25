"""
app.py — Streamlit Interactive Web Application
==============================================
Interactive Dashboard for Battery SOH & RUL Prediction using
the Adaptive Memory-Window Deep Neural Network.
"""

import os
import sys
import json
import torch
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# Path configuration
APP_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(APP_DIR, '..'))
SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models', 'final')
DATA_PATH = os.path.join(PROJECT_ROOT, 'data', 'processed', 'cycle_features.csv')
FIGURES_DIR = os.path.join(PROJECT_ROOT, 'figures')

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from dnn_model import build_model
from preprocessing import AntiLeakageScaler, compute_soh, compute_rul, DEFAULT_EOL_THRESHOLD_AH
from memory_windows import create_memory_window_dataset

# Streamlit Page Config
st.set_page_config(
    page_title="Battery RUL Prognostics — Adaptive Memory DNN",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for rich modern UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4b5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0f172a;
    }
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-top: 4px;
    }
    .status-healthy { color: #16a34a; font-weight: 600; }
    .status-degraded { color: #d97706; font-weight: 600; }
    .status-critical { color: #dc2626; font-weight: 600; }
    .disclaimer-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        font-size: 0.88rem;
        color: #1e40af;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load trained model, scaler, and config metadata."""
    model_path = os.path.join(MODELS_DIR, 'final_model.pt')
    scaler_path = os.path.join(MODELS_DIR, 'scaler.joblib')
    config_path = os.path.join(MODELS_DIR, 'config.json')
    
    # Fallback to models root if final subdir not populated
    if not os.path.exists(model_path):
        model_path = os.path.join(PROJECT_ROOT, 'models', 'final_model.pt')
        scaler_path = os.path.join(PROJECT_ROOT, 'models', 'scaler.joblib')
        config_path = os.path.join(PROJECT_ROOT, 'models', 'config.json')
        
    with open(config_path, 'r') as f:
        config = json.load(f)
        
    scaler = AntiLeakageScaler('minmax').load(scaler_path)
    
    input_dim = config['input_dimension']
    model = build_model(input_dim=input_dim, hidden_dims=(128, 64, 32), dropout_rate=0.2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    
    return model, scaler, config


@st.cache_data
def load_battery_dataset():
    """Load preprocessed NASA battery dataset."""
    df = pd.read_csv(DATA_PATH)
    df = compute_soh(df)
    df = compute_rul(df)
    return df


def main():
    st.markdown('<div class="main-header">⚡ Lithium-Ion Battery Degradation & RUL Prognostics</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Adaptive Memory-Window Feed-Forward Deep Neural Network (NASA Li-ion Dataset Benchmark)</div>', unsafe_allow_html=True)
    
    # Load resources
    try:
        model, scaler, config = load_model_and_config()
        df = load_battery_dataset()
    except Exception as e:
        st.error(f"Error loading model resources: {e}")
        return
        
    window_size = config.get('selected_memory_window', 10)
    base_features = config.get('base_feature_names')
    
    # Sidebar
    st.sidebar.image("https://img.icons8.com/fluency/96/000000/battery-charging.png", width=64)
    st.sidebar.title("Configuration & Controls")
    
    app_mode = st.sidebar.radio("Navigation", ["🔍 Live Battery Diagnostics", "📊 Research Benchmark & Experiments", "📖 Project Documentation & Architecture"])
    
    if app_mode == "🔍 Live Battery Diagnostics":
        st.sidebar.subheader("Select Battery & Cycle")
        batteries = sorted(df['battery_id'].unique())
        selected_battery = st.sidebar.selectbox("Test Battery Cell", batteries, index=0)
        
        bat_df = df[df['battery_id'] == selected_battery].sort_values('cycle_number').reset_index(drop=True)
        max_cycles = len(bat_df)
        
        selected_cycle_idx = st.sidebar.slider(
            "Current Cycle Index (t)",
            min_value=window_size,
            max_value=max_cycles - 1,
            value=min(60, max_cycles - 1),
            step=1
        )
        
        current_cycle_num = bat_df.iloc[selected_cycle_idx]['cycle_number']
        initial_cap = bat_df.iloc[0]['discharge_capacity']
        
        # Extract memory window [selected_cycle_idx - window_size + 1 : selected_cycle_idx + 1]
        window_df = bat_df.iloc[selected_cycle_idx - window_size + 1 : selected_cycle_idx + 1]
        raw_window_matrix = window_df[base_features].values
        flattened_input = raw_window_matrix.flatten().reshape(1, -1)
        
        # Scale and Predict
        scaled_input = scaler.transform(flattened_input)
        with torch.no_grad():
            tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
            predicted_capacity = float(model(tensor_input).item())
            
        predicted_soh = (predicted_capacity / initial_cap) * 100.0
        
        # Compute ground truth
        actual_capacity = bat_df.iloc[selected_cycle_idx]['discharge_capacity']
        actual_soh = (actual_capacity / initial_cap) * 100.0
        actual_rul = bat_df.iloc[selected_cycle_idx]['rul_cycles']
        
        # Estimate RUL based on current degradation rate
        deg_rate_per_cycle = max(1e-5, (initial_cap - predicted_capacity) / max(1, current_cycle_num))
        estimated_rul = max(0, int((predicted_capacity - DEFAULT_EOL_THRESHOLD_AH) / deg_rate_per_cycle)) if predicted_capacity > DEFAULT_EOL_THRESHOLD_AH else 0
        
        # Metrics Row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{predicted_capacity:.4f} Ah</div>
                <div class="metric-label">Predicted Capacity</div>
                <small style="color: #64748b;">Ground Truth: {actual_capacity:.4f} Ah</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{predicted_soh:.2f} %</div>
                <div class="metric-label">Predicted SOH</div>
                <small style="color: #64748b;">Ground Truth: {actual_soh:.2f} %</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{estimated_rul} cycles</div>
                <div class="metric-label">Estimated RUL</div>
                <small style="color: #64748b;">Ground Truth: {actual_rul} cycles</small>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            if predicted_soh >= 85:
                status_html = '<span class="status-healthy">● Optimal Health (Good)</span>'
            elif predicted_soh >= 75:
                status_html = '<span class="status-degraded">▲ Moderate Degradation</span>'
            else:
                status_html = '<span class="status-critical">■ Critical / End-of-Life Warning</span>'
                
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value" style="font-size: 1.25rem; padding-top: 6px;">{status_html}</div>
                <div class="metric-label">Health Assessment</div>
                <small style="color: #64748b;">EOL Threshold: {DEFAULT_EOL_THRESHOLD_AH:.2f} Ah (70%)</small>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Interactive Degradation Trajectory
        st.subheader(f"📈 Battery {selected_battery} Degradation Trajectory & Memory Window Tracking")
        
        # Generate full trajectory prediction for this battery
        ds_bat = create_memory_window_dataset(bat_df, window_size=window_size)
        X_bat_all = scaler.transform(ds_bat['X'])
        with torch.no_grad():
            preds_all = model(torch.tensor(X_bat_all, dtype=torch.float32)).numpy().flatten()
            
        cycles_pred = ds_bat['metadata']['target_cycle'].values
        
        fig, ax = plt.subplots(figsize=(12, 5), dpi=300)
        ax.plot(bat_df['cycle_number'], bat_df['discharge_capacity'], 'o-', color='#1e293b', label='Actual NASA Capacity (Ah)', markersize=3, linewidth=1.5, alpha=0.75)
        ax.plot(cycles_pred, preds_all, 's--', color='#2563eb', label=f'DNN Memory-{window_size} Predicted Capacity (Ah)', markersize=3, linewidth=1.8)
        
        # Highlight active sliding window
        window_start_cycle = window_df.iloc[0]['cycle_number']
        window_end_cycle = window_df.iloc[-1]['cycle_number']
        ax.axvspan(window_start_cycle, window_end_cycle, color='#10b981', alpha=0.25, label=f'Active Historical Window (W={window_size} cycles)')
        
        # Highlight current predicted point
        ax.scatter([current_cycle_num], [predicted_capacity], color='#dc2626', s=130, zorder=6, label=f'Active Prediction at Cycle {int(current_cycle_num)}')
        
        # EOL line
        ax.axhline(y=DEFAULT_EOL_THRESHOLD_AH, color='#f59e0b', linestyle='-.', linewidth=1.6, label='EOL Threshold (1.40 Ah)')
        
        ax.set_title(f"Degradation Profile for NASA Battery {selected_battery}", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlabel("Cycle Number", fontsize=10)
        ax.set_ylabel("Discharge Capacity (Ah)", fontsize=10)
        ax.legend(loc='upper right', frameon=True, fontsize=9)
        ax.set_ylim(1.1, 2.05)
        
        st.pyplot(fig)
        
        # Historical memory window feature table
        with st.expander(f"📋 Inspect Active Historical Memory Window Data (Cycles {int(window_start_cycle)} to {int(window_end_cycle)})"):
            st.dataframe(window_df[['cycle_number', 'discharge_capacity', 'voltage_mean', 'voltage_min', 'voltage_max', 'temp_mean', 'temp_max', 'discharge_duration']], use_container_width=True)
            
    elif app_mode == "📊 Research Benchmark & Experiments":
        st.subheader("🔬 Phase 2 Research Experiments & Benchmarks")
        
        tab1, tab2, tab3, tab4 = st.tabs([
            "1. Memory Window Comparison",
            "2. Early-Life Prediction",
            "3. Feature Ablation",
            "4. Sensor Noise Robustness"
        ])
        
        with tab1:
            st.markdown("#### Systematic Comparison Across Memory Windows (W = 3, 5, 10, 20)")
            st.markdown("Evaluated under strict 4-Fold Leave-One-Battery-Out (LOBO) Cross-Validation across all NASA batteries.")
            
            mem_csv_path = os.path.join(PROJECT_ROOT, 'results', 'final', 'memory_comparison.csv')
            if os.path.exists(mem_csv_path):
                df_mem = pd.read_csv(mem_csv_path)
                st.dataframe(df_mem, use_container_width=True)
                
            fig_path = os.path.join(FIGURES_DIR, 'memory_window_comparison.png')
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Figure 1: Mean LOBO RMSE and RUL Error across Memory Horizons", use_container_width=True)
                
        with tab2:
            st.markdown("#### Early-Life Degradation & RUL Trajectory Forecasting")
            st.markdown("Forecasts full remaining battery life using only the first 20, 30, 40, or 50 cycles on an unseen battery.")
            
            early_csv_path = os.path.join(PROJECT_ROOT, 'results', 'final', 'early_life_prediction.csv')
            if os.path.exists(early_csv_path):
                df_early = pd.read_csv(early_csv_path)
                st.dataframe(df_early, use_container_width=True)
                
            fig_path = os.path.join(FIGURES_DIR, 'early_life_prediction.png')
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Figure 2: Early-Life Forecasted Trajectory on Unseen NASA Cell B0005", use_container_width=True)
                
        with tab3:
            st.markdown("#### Feature Group & Memory Ablation Study")
            st.markdown("Isolates the individual contributions of Voltage statistics, Temperature statistics, Discharge Duration, and Capacity history.")
            
            abl_csv_path = os.path.join(PROJECT_ROOT, 'results', 'final', 'ablation_study.csv')
            if os.path.exists(abl_csv_path):
                df_abl = pd.read_csv(abl_csv_path)
                st.dataframe(df_abl, use_container_width=True)
                
            fig_path = os.path.join(FIGURES_DIR, 'ablation_study.png')
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Figure 3: Ablation Study Comparison across Feature Subsets", use_container_width=True)
                
        with tab4:
            st.markdown("#### Sensor Measurement Noise Robustness")
            st.markdown("Evaluates stability when 1%, 2%, 5%, and 10% Gaussian sensor noise is added to the input features.")
            
            rob_csv_path = os.path.join(PROJECT_ROOT, 'results', 'final', 'robustness_study.csv')
            if os.path.exists(rob_csv_path):
                df_rob = pd.read_csv(rob_csv_path)
                st.dataframe(df_rob, use_container_width=True)
                
            fig_path = os.path.join(FIGURES_DIR, 'robustness_study.png')
            if os.path.exists(fig_path):
                st.image(fig_path, caption="Figure 4: Prediction Degradation Under Sensor Noise", use_container_width=True)
                
    elif app_mode == "📖 Project Documentation & Architecture":
        st.subheader("🏛️ Architecture & Project Specification")
        
        st.markdown("""
        ### Deep Neural Network Architecture
        ```
        Input Layer (W × 12 Features = 120 Inputs for W=10)
          ↓
        Linear Layer (120 → 128) + ReLU + He Normal Initialization
          ↓
        Dropout (p = 0.20)
          ↓
        Linear Layer (128 → 64) + ReLU
          ↓
        Linear Layer (64 → 32) + ReLU
          ↓
        Linear Output Layer (32 → 1) → Predicted Discharge Capacity (Ah)
        ```
        
        ### Global Mathematical Formulations
        1. **State of Health (SOH):**
           $$\\text{SOH}(t) = \\frac{C(t)}{C_{\\text{initial}}} \\times 100\\%$$
        2. **Remaining Useful Life (RUL):**
           $$\\text{RUL}(t) = \\max(0, t_{\\text{EOL}} - t)$$
           where $t_{\\text{EOL}}$ is the first cycle where capacity drops below 1.40 Ah (70% nominal rated capacity).
        3. **Leave-One-Battery-Out (LOBO) Anti-Leakage Protocol:**
           $$\\text{Scaler fitted on } \\mathcal{D}_{\\text{train}} = \\{B_j \\mid j \\neq i\\}, \\quad \\text{Evaluated on unseen } \\mathcal{D}_{\\text{test}} = \\{B_i\\}$$
        """)
        
    st.markdown("""
    <div class="disclaimer-box">
        <b>⚠️ Research Prototype Disclaimer:</b> This application demonstrates an adaptive memory-window deep learning prognostic model trained on the NASA Lithium-Ion Battery Aging dataset (B0005, B0006, B0007, B0018). Results are provided for academic research and educational demonstration only and are not intended for direct safety-critical hardware BMS deployment.
    </div>
    """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
