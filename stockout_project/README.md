# 📦 Stockout Risk Dashboard

Interactive Machine Learning dashboard untuk memprediksi **probabilitas SKU mengalami stockout dalam 30 hari ke depan**.

Project ini membantu tim inventory memahami:
- SKU yang memiliki risiko kehabisan stok.
- Faktor utama penyebab risiko.
- Simulasi kemungkinan stockout.
- Rekomendasi tindakan untuk menurunkan risiko.

Pipeline:
Predict → Simulate → Explain → What-If

---

## 🚀 Features

| Feature | Description |
|---|---|
| Prediction | Prediksi probabilitas stockout menggunakan XGBoost + probability calibration. |
| Simulation | Monte Carlo simulation untuk memperkirakan hari hingga stockout. |
| Explainability | SHAP untuk menjelaskan faktor yang mempengaruhi prediksi. |
| What-If Analysis | Rekomendasi perubahan safety stock atau lead time agar risiko turun. |

---

## 🏗️ Project Structure
stockout_dashboard/

├── app.py
│
├── dashboard/
│ ├── config.py
│ ├── loaders.py
│ ├── sidebar.py
│ ├── charts.py
│ ├── prediction.py
│ ├── simulation.py
│ ├── explainability.py
│ ├── whatif.py
│ └── utils.py
│
├── data_prep.py
├── train_model.py
├── inventory_dataset.csv
├── inventory_dataset_clean.csv
├── model_artifacts.joblib
├── requirements.txt
└── Dockerfile

---

## 🧠 Machine Learning Approach

Model:
XGBoost Classifier
|
v
Calibrated Probability
|
v
Stockout Risk Prediction

Teknik yang digunakan:

- Probability calibration menggunakan `CalibratedClassifierCV`.
- Class imbalance handling menggunakan `scale_pos_weight`.
- SHAP TreeExplainer untuk interpretasi model.
- Monte Carlo simulation dengan NumPy vectorization.

---

## 📊 Dataset Features

Input utama:

- `current_stock`
- `avg_daily_sales`
- `sales_volatility`
- `supplier_lead_time_days`
- `safety_stock_level`
- `reorder_point`
- `seasonality_index`
- `historical_stockout_count`

Target:
will_stockout_next_30_days

---

## ⚙️ Running Locally

Install dependencies:

```bash
pip install -r requirements.txt

Prepare data:

python data_prep.py

Train model:

python train_model.py

Run dashboard:

streamlit run app.py

Open:

http://localhost:8501
🐳 Docker

Build:

docker build -t stockout-dashboard .

Run:

docker run -p 8501:8501 stockout-dashboard