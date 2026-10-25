# Early Lithium-Ion Battery Degradation and Remaining Useful Life Prediction Using an Adaptive Memory-Window Deep Neural Network

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end research implementation and benchmarking framework investigating how historical memory window horizon influences the degradation trajectory and Remaining Useful Life (RUL) prediction of Lithium-Ion batteries when using a lightweight, pure feed-forward Deep Neural Network (DNN / MLP).

---

## 1. Project Overview & Research Question

### Research Question
> **How much historical battery-cycle information is required for a pure feed-forward DNN to accurately predict future battery degradation and RUL under battery-independent evaluation?**

While modern battery health prognostics frequently utilize complex recurrent architectures (RNN, LSTM, GRU) or heavy Transformer models, these approaches pose significant computational overhead and memory constraints when deployed on edge microcontrollers within Battery Management Systems (BMS). This project systematically analyzes whether a feed-forward Deep Neural Network supplied with an **Adaptive Memory Window** ($W \in \{3, 5, 10, 20\}$ past discharge cycles) can capture electrochemical degradation dynamics and capacity regeneration spikes with high fidelity.

---

## 2. NASA Battery Dataset

The project utilizes the benchmark [NASA Li-ion Battery Aging Dataset](https://data.nasa.gov/dataset/li-ion-battery-aging-datasets) (Ames Prognostics Center of Excellence):
- **Battery Cells:** B0005, B0006, B0007, B0018 (Gen 1 18650 LiCoO2 cells, 2.0 Ah nominal rated capacity).
- **Test Protocol:** Repeated CC-CV charging at 1.5A up to 4.2V, followed by CC discharge at 2.0A until cut-off voltages (2.7V, 2.5V, 2.2V, 2.5V) at ambient room temperature (24°C).
- **Cycle Count:** 636 total validated discharge cycles across 4 cells (168 cycles each for B0005, B0006, B0007; 132 cycles for B0018).

---

## 3. Preprocessing, Feature Engineering & Anti-Leakage Protocol

### Extracted Cycle-Level Features (12 per cycle)
1. **Voltage Profile (4):** Mean Voltage ($V_{\text{mean}}$), Minimum Voltage ($V_{\text{min}}$), Maximum Voltage ($V_{\text{max}}$), Voltage Standard Deviation ($V_{\text{std}}$).
2. **Current Profile (2):** Mean Discharge Current ($I_{\text{mean}}$), Current Standard Deviation ($I_{\text{std}}$).
3. **Thermal Profile (3):** Mean Temperature ($T_{\text{mean}}$), Peak Temperature ($T_{\text{max}}$), Temperature Range ($\Delta T$).
4. **Temporal & Capacity (3):** Discharge Duration ($t_{\text{dis}}$), Discharge Capacity ($C$), Cycle Number ($k$).

### Target Formulations
- **State of Health (SOH):**
  $$\text{SOH}(t) = \frac{C(t)}{C_{\text{initial}}} \times 100\%$$
- **Remaining Useful Life (RUL):**
  $$\text{RUL}(t) = \max(0, k_{\text{EOL}} - t)$$
  where $k_{\text{EOL}}$ is the first cycle index where $C(t) \le 1.40\text{ Ah}$ (70% nominal rated capacity).

### Anti-Leakage Protocol (Leave-One-Battery-Out)
To prevent data leakage, data is never split randomly across cycles. A 4-fold **Leave-One-Battery-Out (LOBO)** cross-validation is enforced:
$$\text{Scaler fitted on } \mathcal{D}_{\text{train}} = \{B_j \mid j \neq i\}, \quad \text{Tested strictly on unseen held-out } \mathcal{D}_{\text{test}} = \{B_i\}$$

---

## 4. Adaptive Memory-Window Architecture

For a memory window size $W$, the input vector $\mathbf{x}_t \in \mathbb{R}^{W \times 12}$ aggregates the preceding $W$ cycles to predict the capacity at cycle $t+1$:

```
Historical Memory Window (W cycles × 12 features)
  [t-W+1, ..., t] ──> Flatten (Dim = 12·W)
                          ↓
                    Linear (128) + ReLU + He Normal Init
                          ↓
                    Dropout (p = 0.20)
                          ↓
                    Linear (64) + ReLU
                          ↓
                    Linear (32) + ReLU
                          ↓
                    Linear (1) ──> Predicted Capacity C(t+1)
```

---

## 5. Experimental Results (LOBO Cross-Validation)

### Experiment 1: Memory-Window Comparison
| Memory Horizon | Input Dim | LOBO MAE (Ah) | LOBO RMSE (Ah) | LOBO $R^2$ | Mean MAPE (%) | RUL MAE (Cycles) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Memory-3** | 36 | 0.08180 | **0.09486** | **0.6794** | 5.22% | 12.91 |
| **Memory-5** | 60 | 0.10801 | 0.13176 | 0.3483 | 6.81% | 33.40 |
| **Memory-10** | 120 | **0.08037** | 0.09960 | 0.6512 | **5.21%** | **8.45** |
| **Memory-20** | 240 | 0.08861 | 0.10415 | 0.1382 | 5.82% | 12.96 |

> **Key Takeaway:** Memory-10 achieved the lowest overall RUL estimation error (**8.45 cycles**) and lowest MAE (**0.08037 Ah / 5.21% MAPE**) across all unseen batteries.

### Experiment 2: Early-Life Degradation & RUL Prediction
Using only early historical cycles on unseen battery **B0005**:
- **First 20 Cycles Observation:** Future RMSE = 0.05332 Ah, Future $R^2 = 0.9075$, Predicted EOL = Cycle 132 (True EOL = Cycle 124).
- **First 30 Cycles Observation:** Future RMSE = 0.04741 Ah, Future $R^2 = 0.9155$, Predicted EOL = Cycle 132 (True EOL = Cycle 124).
- **First 40 Cycles Observation:** Future RMSE = 0.04658 Ah, Future $R^2 = 0.9015$, Predicted EOL = Cycle 132 (True EOL = Cycle 124).

### Experiment 3: Feature Group Ablation Study
- **Operational Sensors Only (No Capacity History):** RMSE increases by +64.9% (RMSE = 0.16423 Ah, $R^2 = 0.1116$).
- **No Duration:** RMSE rises to 0.13343 Ah ($R^2 = 0.4440$).
- **No Voltage Statistics:** RMSE rises to 0.12362 Ah ($R^2 = 0.2496$).

### Experiment 4: Sensor Measurement Noise Robustness
- Clean: RMSE = 0.09960 Ah ($R^2 = 0.6512$)
- +1% Gaussian Noise: RMSE = 0.10361 Ah (+4.03% error)
- +5% Gaussian Noise: RMSE = 0.10819 Ah (+8.62% error)
- +10% Gaussian Noise: RMSE = 0.12014 Ah (+20.62% error)

---

## 6. Installation & Quickstart

### Setup Environment
```bash
git clone https://github.com/Vignesh-61/Early-Life-Lithium-Ion-Battery-Degradation-and-RUL-Prediction.git
cd Early-Life-Lithium-Ion-Battery-Degradation-and-RUL-Prediction
pip install -r requirements.txt
```

### Run Tests
```bash
pytest tests/ -v
```

### Run Full Research Experiments (Phase 2)
```bash
python src/phase2_experiments.py
```

### Train Final Production Model (Phase 3)
```bash
python src/train_final_model.py
```

### Launch Interactive Streamlit Dashboard
```bash
streamlit run app/app.py
```

---

## 7. Project Directory Structure

```
battery-dnn-project/
├── data/
│   ├── raw/                      # NASA B0005, B0006, B0007, B0018 .mat files
│   └── processed/                # cycle_features.csv (636 cycles)
├── src/
│   ├── data_loader.py            # .mat parser and cycle extractor
│   ├── feature_engineering.py    # 12-feature statistical cycle extractor
│   ├── preprocessing.py          # SOH/RUL calculator and AntiLeakageScaler
│   ├── memory_windows.py         # Adaptive memory window generator (3, 5, 10, 20)
│   ├── dnn_model.py              # Pure feed-forward PyTorch DNN
│   ├── training.py               # LOBO cross-validation engine
│   ├── evaluation.py             # Performance metrics and plot generator
│   ├── phase2_experiments.py     # All 5 research experiment runners
│   ├── train_final_model.py      # Final model training and serialization
│   └── prediction.py             # Inference pipeline
├── models/
│   └── final/                    # final_model.pt, scaler.joblib, config.json
├── results/
│   ├── phase1/                   # Phase 1 LOBO results and fold curves
│   └── final/                    # Phase 2 experiment CSVs and plots
├── figures/                      # High-resolution publication figures
├── app/
│   └── app.py                    # Interactive Streamlit dashboard
├── report/
│   ├── IEEE_Research_Paper.md    # Complete IEEE-style project paper
│   └── Review1_Presentation.md   # College Review-1 presentation slides
├── requirements.txt
└── README.md
```

---

## 8. Limitations & Future Work

### Limitations
1. **Limited Operating Diversity:** The NASA dataset comprises 4 cells tested under fixed CC-CV charging and constant discharge at 24°C; real-world EV drive cycles involve dynamic discharge profiles.
2. **Fixed EOL Assumption:** RUL estimation is evaluated relative to the standard 1.40 Ah (70% nominal) threshold; different applications (e.g. grid storage) define EOL at 80% or 60%.
3. **Research Prototype:** This model is designed for academic benchmarking and edge feasibility studies, not direct safety-critical BMS deployment.

### Future Work
- Evaluation on dynamic drive-cycle datasets (e.g. Oxford, CALCE, Stanford/MIT).
- Multi-temperature transfer learning and domain adaptation.
- Integration of physics-informed degradation constraints.
- Quantization (INT8) and deployment on ARM Cortex-M microcontrollers.

---

## 9. Citation & Base References

If using this implementation, please cite:

```bibtex
@article{tarar2023accurate,
  title={Accurate prediction of remaining useful life for lithium-ion battery using deep neural networks with memory features},
  author={Tarar, M. and others},
  journal={Frontiers in Energy Research},
  volume={10},
  pages={1059701},
  year={2023},
  publisher={Frontiers},
  doi={10.3389/fenrg.2023.1059701}
}
```
