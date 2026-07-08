"""
app.py
Dashboard Stockout Probability: Predict -> Simulate -> Explain -> What-If.
Jalankan: streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st

st.set_page_config(page_title="Stockout Risk Dashboard", layout="wide")

DATA_PATH = "inventory_dataset.csv"
ARTIFACT_PATH = "model_artifacts.joblib"


# ---------------------------------------------------------------------------
# Loaders (cached)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    artifacts = joblib.load(ARTIFACT_PATH)
    return artifacts["calibrated_model"], artifacts["shap_model"], artifacts["features"]


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


calibrated_model, shap_model, FEATURES = load_artifacts()
df = load_data()

explainer = shap.TreeExplainer(shap_model)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Pilih SKU & Simulasi")

sku_id = st.sidebar.selectbox("SKU", df["sku_id"].tolist())
row = df[df["sku_id"] == sku_id].iloc[0]

st.sidebar.markdown("---")
st.sidebar.caption("Ubah parameter untuk simulasi what-if")

safety_stock = st.sidebar.slider(
    "Safety Stock Level", 0, 300, int(row["safety_stock_level"])
)
lead_time = st.sidebar.slider(
    "Supplier Lead Time (hari)", 1, 40, int(row["supplier_lead_time_days"])
)
avg_sales = st.sidebar.slider(
    "Avg Daily Sales", 0.0, 40.0, float(row["avg_daily_sales"]), step=0.1
)

# Bangun baris fitur hasil simulasi (fitur lain tetap dari data asli SKU)
sim_row = row.copy()
sim_row["safety_stock_level"] = safety_stock
sim_row["supplier_lead_time_days"] = lead_time
sim_row["avg_daily_sales"] = avg_sales
X_sim = pd.DataFrame([sim_row[FEATURES]])


def predict_proba(X):
    return calibrated_model.predict_proba(X)[:, 1][0]


proba = predict_proba(X_sim)

st.title(f"📦 Stockout Risk Dashboard — {sku_id}")
st.caption("Alur: Predict → Simulate → Explain → What-If Recommendation")

col_left, col_right = st.columns([1, 1])


# ---------------------------------------------------------------------------
# Konten A — Prediksi (Gauge)
# ---------------------------------------------------------------------------
with col_left:
    st.subheader("A. Calibrated Stockout Probability")

    gauge_color = "green" if proba < 0.3 else "orange" if proba < 0.6 else "red"
    fig_gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            number={"suffix": "%"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": gauge_color},
                "steps": [
                    {"range": [0, 30], "color": "#d4f4dd"},
                    {"range": [30, 60], "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
            },
            title={"text": "Risiko Stockout 30 Hari"},
        )
    )
    fig_gauge.update_layout(height=320, margin=dict(t=50, b=10))
    st.plotly_chart(fig_gauge, use_container_width=True)

    if proba >= 0.6:
        st.error("⚠️ Risiko tinggi — segera tinjau ulang stok.")
    elif proba >= 0.3:
        st.warning("⚠️ Risiko sedang — perlu dipantau.")
    else:
        st.success("✅ Risiko rendah — stok relatif aman.")


# ---------------------------------------------------------------------------
# Konten B — Simulasi Monte Carlo
# ---------------------------------------------------------------------------
with col_right:
    st.subheader("B. Simulasi Monte Carlo — Days Until Stockout")

    with st.spinner("Menjalankan simulasi Monte Carlo (1000 iterasi)..."):
        n_sims = 1000
        n_days = 60
        rng = np.random.default_rng(42)

        daily_mean = avg_sales * sim_row["seasonality_index"]
        daily_std = max(row["sales_volatility"], 0.01)

        # Vektorisasi penuh: matrix (n_sims x n_days), tanpa loop Python
        daily_sales_matrix = rng.normal(
            loc=daily_mean, scale=daily_std, size=(n_sims, n_days)
        )
        daily_sales_matrix = np.clip(daily_sales_matrix, 0, None)

        cum_sales = np.cumsum(daily_sales_matrix, axis=1)
        stock_curve = safety_stock + row["current_stock"] - cum_sales

        stockout_mask = stock_curve <= 0
        days_until_stockout = np.where(
            stockout_mask.any(axis=1),
            stockout_mask.argmax(axis=1) + 1,
            n_days + 1,  # tidak stockout dalam horizon simulasi
        )

    pct_stockout_30d = (days_until_stockout <= 30).mean() * 100

    fig_hist = go.Figure()
    fig_hist.add_trace(
        go.Histogram(
            x=days_until_stockout[days_until_stockout <= n_days],
            nbinsx=30,
            marker_color="steelblue",
            name="Distribusi",
        )
    )
    fig_hist.add_vline(
        x=30, line_dash="dash", line_color="red",
        annotation_text="Batas aman 30 hari", annotation_position="top",
    )
    fig_hist.update_layout(
        height=320,
        xaxis_title="Hari hingga stockout",
        yaxis_title="Jumlah simulasi",
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig_hist, use_container_width=True)
    st.caption(
        f"Dari {n_sims} simulasi, **{pct_stockout_30d:.1f}%** mengalami stockout dalam 30 hari."
    )


st.markdown("---")

col_c, col_d = st.columns([1, 1])


# ---------------------------------------------------------------------------
# Konten C — SHAP Explanation
# ---------------------------------------------------------------------------
with col_c:
    st.subheader("C. Kontribusi Fitur (SHAP)")

    shap_values = explainer.shap_values(X_sim)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    contrib = pd.DataFrame(
        {"feature": FEATURES, "shap_value": shap_values[0]}
    ).sort_values("shap_value", key=abs, ascending=True)

    fig_shap = go.Figure(
        go.Bar(
            x=contrib["shap_value"],
            y=contrib["feature"],
            orientation="h",
            marker_color=["#d9534f" if v > 0 else "#5cb85c" for v in contrib["shap_value"]],
        )
    )
    fig_shap.update_layout(
        height=350,
        xaxis_title="Kontribusi terhadap risiko (SHAP value)",
        margin=dict(t=20, b=10),
    )
    st.plotly_chart(fig_shap, use_container_width=True)
    st.caption("🔴 Merah = menaikkan risiko stockout · 🟢 Hijau = menurunkan risiko")


# ---------------------------------------------------------------------------
# Konten D — What-If Recommendation (pengganti DiCE, lebih stabil)
# ---------------------------------------------------------------------------
with col_d:
    st.subheader("D. Rekomendasi What-If")

    if proba <= 0.10:
        st.info("Probabilitas sudah di bawah 10% — tidak perlu tindakan tambahan.")
    else:
        st.write("Mencari kombinasi parameter yang menurunkan risiko di bawah 10%...")

        target_threshold = 0.10
        found = None

        try:
            # Grid search sederhana & cepat (vektor kecil, bukan iterasi eksternal library)
            safety_options = np.arange(safety_stock, safety_stock + 250, 10)
            lead_options = np.arange(max(1, lead_time - lead_time + 1), lead_time + 1, 1)[::-1]

            for ss in safety_options:
                candidate = sim_row.copy()
                candidate["safety_stock_level"] = ss
                X_cand = pd.DataFrame([candidate[FEATURES]])
                p = predict_proba(X_cand)
                if p < target_threshold:
                    found = ("safety_stock_level", int(ss), p)
                    break

            if found is None:
                for lt in lead_options:
                    candidate = sim_row.copy()
                    candidate["supplier_lead_time_days"] = lt
                    X_cand = pd.DataFrame([candidate[FEATURES]])
                    p = predict_proba(X_cand)
                    if p < target_threshold:
                        found = ("supplier_lead_time_days", int(lt), p)
                        break

            if found:
                param_name, param_value, new_proba = found
                label = "safety stock level" if param_name == "safety_stock_level" else "lead time supplier"
                st.success(
                    f"✅ Untuk menurunkan risiko di bawah 10%, naikkan **{label}** "
                    f"menjadi **{param_value}** "
                    f"({'unit' if param_name == 'safety_stock_level' else 'hari'}). "
                    f"Estimasi probabilitas baru: **{new_proba*100:.1f}%**."
                )
            else:
                st.warning(
                    "⚠️ Kondisi stok terlalu kritis — perubahan safety stock atau lead time saja "
                    "tidak cukup menurunkan risiko di bawah 10%. Pertimbangkan menambah "
                    "current_stock secara langsung atau mencari supplier alternatif."
                )
        except Exception as e:
            st.warning(
                "⚠️ Tidak dapat menemukan solusi what-if untuk SKU ini karena kondisi stok "
                "terlalu kritis. Silakan tinjau manual bersama tim procurement."
            )

st.markdown("---")
with st.expander("Lihat data mentah SKU ini"):
    st.dataframe(pd.DataFrame([row]))