# 📦 Stockout Risk Dashboard

Dashboard interaktif untuk memprediksi **probabilitas SKU kehabisan stok dalam 30 hari ke depan**, lengkap dengan simulasi, penjelasan model (explainability), dan rekomendasi what-if.

Alur kerja: **Predict → Simulate → Explain → What-If**

---

## Fitur

| Bagian | Deskripsi |
|---|---|
| **A. Prediksi** | Probabilitas stockout yang **terkalibrasi** (`CalibratedClassifierCV`), ditampilkan sebagai gauge chart hijau–merah. |
| **B. Simulasi** | Monte Carlo (1000 iterasi, vektorisasi NumPy) memproyeksikan distribusi "hari hingga stockout" berdasarkan variabilitas penjualan. |
| **C. Explain (SHAP)** | Bar chart kontribusi tiap fitur terhadap risiko stockout SKU yang dipilih. |
| **D. What-If** | Mencari nilai `safety_stock_level` atau `supplier_lead_time_days` minimal agar probabilitas turun di bawah 10%. |

Semua parameter simulasi (safety stock, lead time, avg daily sales) bisa diubah langsung lewat slider di sidebar.

---

## Struktur Proyek

```
stockout_dashboard/
├── inventory_dataset.csv         # Dataset mentah (input)
├── data_prep.py                  # Validasi & pembersihan data
├── inventory_dataset_clean.csv   # Dataset bersih (dihasilkan data_prep.py)
├── train_model.py                # Training model + kalibrasi probabilitas
├── model_artifacts.joblib        # Model terlatih (dihasilkan train_model.py)
├── app.py                        # Dashboard Streamlit
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Skema Data

| Kolom | Tipe | Keterangan |
|---|---|---|
| `sku_id` | string | ID unik produk |
| `current_stock` | int | Stok fisik saat ini |
| `avg_daily_sales` | float | Rata-rata penjualan harian |
| `sales_volatility` | float | Standar deviasi penjualan harian |
| `supplier_lead_time_days` | int | Waktu tunggu pengiriman supplier |
| `safety_stock_level` | int | Batas stok aman |
| `reorder_point` | int | Titik pemesanan ulang otomatis |
| `seasonality_index` | float | Faktor musiman (0.5–1.5) |
| `historical_stockout_count` | int | Jumlah stockout tahun lalu |
| `will_stockout_next_30_days` | 0/1 | **Target**: stockout dalam 30 hari? |

---

## Cara Menjalankan (Lokal)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Bersihkan & validasi data mentah
python data_prep.py

# 3. Latih model (menghasilkan model_artifacts.joblib)
python train_model.py

# 4. Jalankan dashboard
streamlit run app.py
```

Buka browser di `http://localhost:8501`.

---

## Cara Menjalankan (Docker)

```bash
docker build -t stockout-dashboard .
docker run -p 8501:8501 stockout-dashboard
```

Dockerfile sudah menjalankan `data_prep.py` dan `train_model.py` otomatis saat build image, jadi container langsung siap pakai.

---

## Catatan Teknis

- **Kalibrasi probabilitas**: Model dasar XGBoost dibungkus `CalibratedClassifierCV` (isotonic) agar `predict_proba()` merepresentasikan probabilitas statistik yang valid — bukan sekadar skor mentah. Ini penting untuk keputusan manajemen risiko (mis. keputusan pembelian).
- **Class imbalance**: Ditangani lewat `scale_pos_weight` di XGBoost, bukan oversampling, agar distribusi asli tetap terjaga untuk proses kalibrasi.
- **SHAP**: Menggunakan model XGBoost terpisah (tanpa kalibrasi) khusus untuk `TreeExplainer`, karena SHAP butuh akses langsung ke struktur pohon.
- **What-If (pengganti DiCE)**: Alih-alih library `dice-ml` (rawan konflik versi dengan scikit-learn/shap terbaru), rekomendasi what-if dihitung lewat grid search sederhana terhadap model yang sudah terlatih — lebih ringan dan stabil untuk deployment, dengan `try/except` fallback jika kondisi stok terlalu kritis untuk diselesaikan lewat safety stock/lead time saja.
- **Monte Carlo**: Sepenuhnya vektorisasi NumPy (matrix `n_sims x n_days`), tidak ada loop Python, sehingga 1000 iterasi berjalan cepat.

---

## Retrain dengan Data Baru

Ganti isi `inventory_dataset.csv` dengan data terbaru (skema kolom harus sama), lalu jalankan ulang:

```bash
python data_prep.py
python train_model.py
```

`data_prep.py` akan otomatis memvalidasi skema, membuang baris dengan nilai kosong/tidak wajar, dan melaporkan ringkasan pembersihan sebelum training dilanjutkan.