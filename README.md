<div align="center">

<img src="https://img.shields.io/badge/%E2%9A%A1-Smart%20Grid%20AI-00C7B7?style=for-the-badge&labelColor=0d1117" alt="Smart Grid AI"/>

# DeepAR Probabilistic Electricity Load Forecasting

### 📉 *Quantifying Uncertainty in Power Grids with TensorFlow Probability* 📉

<br/>

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)
[![Dataset](https://img.shields.io/badge/Data-American%20Electric%20Power%20(AEP)-005288?style=for-the-badge)](https://www.aep.com/)

<br/>

<img src="https://img.shields.io/badge/Author-Mohammed%20Ezzeldin%20Babiker%20Abdullah-4A90D9?style=flat-square&logo=google-scholar&logoColor=white" alt="Author"/>

---

*"Deterministic point forecasts are insufficient for smart grids. Decision-makers need confidence intervals."*

</div>

---

## 🎯 Project Overview

This repository provides an academically rigorous implementation of the **DeepAR** model using **TensorFlow, Keras, and TensorFlow Probability (TFP)**. Specifically applied to **probabilistic electricity load forecasting** on the American Electric Power (AEP) dataset, this model predicts not just a single value, but an entire probability distribution.

### ✨ Key Features

| Feature | Description |
|:-------:|:------------|
| 📊 **Probabilistic Output** | Outputs parameterized Gaussian distributions via TFP lambda layers |
| 🛡️ **Zero Data Leakage** | Strict chronological window framing and isolated Min-Max scaling |
| 🧮 **Custom Quantile Loss** | Evaluation tracking for 10th, 50th (median), and 90th percentiles |
| 📉 **Negative Log-Likelihood** | Directly optimizes the Bayesian probability of the ground truth |
| 🎨 **4K Publication Charts** | Automated high-fidelity rendering of confidence bands (P10 to P90) |

---

## 🏗️ DeepAR Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   ⏱️ Input Sequence (Historical Window = 168 Hours / 7 Days) │
│       Target (MW) + Time Covariates (Hour, Day, Month)      │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Long Short-Term Memory (LSTM) Encoder        │          │
│  │  64 Units — Capturing historical seasonality  │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Context Vector + Future Covariates Mapping   │          │
│  │  (Concatenating deterministic future time)    │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  LSTM Decoder (Autoregressive step mapping)   │          │
│  │  64 Units — Future trajectory projection      │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  Distribution Parameters (Dense Layer)        │ μ, σ     │
│  │  Mean (μ) & Softplus scale (σ)                │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│  ┌───────────────────────▼───────────────────────┐          │
│  │  TensorFlow Probability Layer                 │ P(y|X)   │
│  │  tfd.Independent(tfd.Normal(loc=μ, scale=σ))  │          │
│  └───────────────────────┬───────────────────────┘          │
│                          │                                  │
│        📊 Probabilistic Forecasting (Next 24H)              │
│        Outputting Bayesian Confidence Intervals             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📂 Repository Structure

```
📦 DeepAR-Probabilistic-Load-Forecasting/
│
├── 📁 training_code/
│   └── 🧠 deepar_forecasting.py         # Full TFP DeepAR architecture
│
├── 📁 training_data/
│   └── 📊 AEP_hourly.csv                # AEP Energy Consumption Data
│
├── 📄 DeepAR_Probabilistic_Forecast_Paper.docx  # Full Academic Manuscript
├── 📋 requirements.txt
└── 📖 README.md
```

---

## 🚀 Quick Start

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the DeepAR Pipeline:**
   ```bash
   python training_code/deepar_forecasting.py
   ```

3. **Outputs Generated Automatically:**
   - 10 High-Fidelity 4K Confidence Plots saved to `DeepAR_Research_Charts/`
   - TensorFlow SavedModel Engine matching TFP extensions
   - Final Quantile Loss & RMSE statistical evaluations printed to the console

---

## 📚 Related Research Portfolio

<div align="center">

| # | Paper | Repository |
|:-:|:------|:----------:|
| 1 | Physics-Guided CNN-BiLSTM Solar Forecast | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Physics-Guided-CNN-BiLSTM-Solar) |
| 2 | Physics-Informed State Space Model (PISSM) | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PISSM-Solar-Forecasting) |
| 3 | PISSM Cross-Attention Networks | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/PISSM-CrossAttention-Solar) |
| 4 | Thermodynamic Liquid Manifold Networks | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/TLMN-Thermodynamic-Solar-Microgrids) |
| 5 | Industrial RUL Prediction Architecture | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/Industrial-RUL-Prediction-Architecture) |
| **6** | **DeepAR Probabilistic Forecasting** *(this repo)* 🌟 | [![Repo](https://img.shields.io/badge/Repo-181717?style=flat-square&logo=github)](https://github.com/Marco9249/DeepAR-Probabilistic-Load-Forecasting) |

</div>

---

## 📖 Citation

```bibtex
@misc{abdullah2026deepar,
  title   = {Probabilistic Electricity Load Forecasting using DeepAR Architectures},
  author  = {Mohammed Ezzeldin Babiker Abdullah},
  year    = {2026}
}
```

---

<div align="center">

### 👤 Author

**Mohammed Ezzeldin Babiker Abdullah**
*Researcher in Predictive Modeling, Renewable Energy & Deep Learning*

[![GitHub](https://img.shields.io/badge/GitHub-Marco9249-181717?style=for-the-badge&logo=github)](https://github.com/Marco9249)

</div>
