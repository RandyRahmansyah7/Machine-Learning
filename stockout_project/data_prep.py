"""
data_prep.py
Validasi & pembersihan dataset inventaris sebelum masuk ke training.

Tugas:
- Cek skema kolom & tipe data
- Cek missing values & duplikat sku_id
- Cek nilai yang tidak masuk akal (stok negatif, probabilitas non-binary, dll)
- Simpan dataset bersih siap latih

Jalankan: python data_prep.py
Input   : inventory_dataset.csv (raw)
Output  : inventory_dataset_clean.csv
"""

import sys

import pandas as pd

RAW_PATH = "inventory_dataset.csv"
CLEAN_PATH = "inventory_dataset_clean.csv"

EXPECTED_COLUMNS = {
    "sku_id": "object",
    "current_stock": "int64",
    "avg_daily_sales": "float64",
    "sales_volatility": "float64",
    "supplier_lead_time_days": "int64",
    "safety_stock_level": "int64",
    "reorder_point": "int64",
    "seasonality_index": "float64",
    "historical_stockout_count": "int64",
    "will_stockout_next_30_days": "int64",
}

NON_NEGATIVE_COLUMNS = [
    "current_stock",
    "avg_daily_sales",
    "sales_volatility",
    "supplier_lead_time_days",
    "safety_stock_level",
    "reorder_point",
    "historical_stockout_count",
]


def load_raw(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        sys.exit(f"[ERROR] File tidak ditemukan: {path}")
    return df


def check_schema(df: pd.DataFrame) -> None:
    missing_cols = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing_cols:
        sys.exit(f"[ERROR] Kolom hilang dari dataset: {missing_cols}")

    extra_cols = set(df.columns) - set(EXPECTED_COLUMNS)
    if extra_cols:
        print(f"[WARNING] Kolom tak dikenal ditemukan (akan diabaikan): {extra_cols}")

    print("[OK] Skema kolom sesuai spesifikasi.")


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    n_missing = df.isnull().sum().sum()
    if n_missing > 0:
        print(f"[WARNING] Ditemukan {n_missing} nilai kosong, baris terkait akan dibuang.")
        df = df.dropna()
    else:
        print("[OK] Tidak ada missing values.")
    return df


def check_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    n_dupes = df["sku_id"].duplicated().sum()
    if n_dupes > 0:
        print(f"[WARNING] Ditemukan {n_dupes} sku_id duplikat, hanya baris pertama yang dipertahankan.")
        df = df.drop_duplicates(subset="sku_id", keep="first")
    else:
        print("[OK] Tidak ada sku_id duplikat.")
    return df


def check_value_ranges(df: pd.DataFrame) -> pd.DataFrame:
    initial_len = len(df)

    # Nilai stok/waktu tidak boleh negatif
    for col in NON_NEGATIVE_COLUMNS:
        invalid = df[col] < 0
        if invalid.any():
            print(f"[WARNING] {invalid.sum()} baris dengan {col} negatif, akan dibuang.")
            df = df[~invalid]

    # Target harus biner (0/1)
    invalid_target = ~df["will_stockout_next_30_days"].isin([0, 1])
    if invalid_target.any():
        print(f"[WARNING] {invalid_target.sum()} baris dengan target non-biner, akan dibuang.")
        df = df[~invalid_target]

    # seasonality_index seharusnya berada di sekitar 0.5 - 1.5 sesuai spesifikasi
    out_of_range = ~df["seasonality_index"].between(0.3, 2.0)
    if out_of_range.any():
        print(f"[WARNING] {out_of_range.sum()} baris dengan seasonality_index di luar rentang wajar (0.3-2.0), akan dibuang.")
        df = df[~out_of_range]

    removed = initial_len - len(df)
    if removed == 0:
        print("[OK] Semua nilai berada dalam rentang wajar.")
    else:
        print(f"[INFO] Total {removed} baris dibuang karena nilai tidak valid.")

    return df


def main():
    print(f"Membaca dataset mentah: {RAW_PATH}")
    df = load_raw(RAW_PATH)
    print(f"Jumlah baris awal: {len(df)}\n")

    check_schema(df)
    df = check_missing(df)
    df = check_duplicates(df)
    df = check_value_ranges(df)

    df = df.reset_index(drop=True)

    print(f"\nJumlah baris setelah pembersihan: {len(df)}")
    df.to_csv(CLEAN_PATH, index=False)
    print(f"Dataset bersih tersimpan di: {CLEAN_PATH}")


if __name__ == "__main__":
    main()