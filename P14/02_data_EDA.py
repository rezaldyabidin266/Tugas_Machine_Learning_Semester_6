"""Tahap 02 - Exploratory Data Analysis | Mobile Price Classification.

Cara menjalankan:
    python 02_data_EDA.py

Semua grafik disimpan sebagai PNG, lalu ditampilkan dalam SATU jendela.
Gunakan tombol Previous / Next atau tombol panah kiri / kanan.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Mobile_Price_Classification" / "train.csv"
OUTPUT_DIR = BASE_DIR / "figures" / "02_eda"


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}\n"
            "Pastikan train.csv berada di folder Mobile_Price_Classification."
        )

    data = pd.read_csv(DATASET_PATH)
    required_columns = {"price_range", "ram", "battery_power", "px_height", "px_width", "mobile_wt"}
    missing_columns = required_columns.difference(data.columns)
    if missing_columns:
        raise ValueError(f"Kolom dataset tidak ditemukan: {sorted(missing_columns)}")
    return data


def save_chart(fig, filename: str) -> Path:
    """Simpan grafik lalu tutup figur sementara (tanpa memunculkan jendela baru)."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def open_gallery(charts: list[tuple[str, Path]]) -> None:
    """Navigasi gambar dengan Previous/Next pada satu jendela Matplotlib."""
    if not charts:
        return

    images = [plt.imread(path) for _, path in charts]
    index = 0

    fig, ax = plt.subplots(figsize=(12, 7.5))
    fig.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.17)
    ax.set_axis_off()
    title = fig.suptitle("", fontsize=13, fontweight="bold", y=0.96)
    prev_button = Button(fig.add_axes([0.28, 0.045, 0.18, 0.075]), "Previous")
    next_button = Button(fig.add_axes([0.54, 0.045, 0.18, 0.075]), "Next")
    # Simpan referensi tombol agar tetap aktif saat jendela non-blocking.
    fig._gallery_buttons = (prev_button, next_button)

    def render() -> None:
        ax.clear()
        ax.imshow(images[index])
        ax.set_axis_off()
        title.set_text(f"EDA  |  {index + 1}/{len(charts)}  |  {charts[index][0]}")
        fig.canvas.draw_idle()

    def previous(_event=None) -> None:
        nonlocal index
        index = (index - 1) % len(charts)
        render()

    def next_chart(_event=None) -> None:
        nonlocal index
        index = (index + 1) % len(charts)
        render()

    def on_key(event) -> None:
        if event.key == "left":
            previous()
        elif event.key == "right":
            next_chart()

    prev_button.on_clicked(previous)
    next_button.on_clicked(next_chart)
    fig.canvas.mpl_connect("key_press_event", on_key)
    render()
    print("\nNavigasi grafik: klik Previous / Next atau tekan panah kiri / kanan.")
    print("Tutup SATU jendela grafik setelah selesai melihat semuanya.\n")
    plt.show()


def detect_iqr_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Deteksi kandidat outlier tanpa mengubah atau menghapus data."""
    rows = []
    for feature in ["ram", "battery_power", "px_height", "px_width", "mobile_wt"]:
        q1, q3 = df[feature].quantile([0.25, 0.75])
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outlier_count = int(((df[feature] < low) | (df[feature] > high)).sum())
        rows.append(
            {
                "Fitur": feature,
                "Batas bawah": round(low, 2),
                "Batas atas": round(high, 2),
                "Jumlah outlier IQR": outlier_count,
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    df = load_data()
    print(f"Dataset: {DATASET_PATH}")
    print(f"Ukuran: {df.shape}")

    print("\n=== STATISTIK DESKRIPTIF ===")
    print(df.describe().T.round(2).to_string())

    correlations = df.corr(numeric_only=True)
    target_corr = (
        correlations["price_range"]
        .drop("price_range")
        .sort_values(key=lambda values: values.abs(), ascending=False)
    )
    print("\n=== KORELASI FITUR DENGAN PRICE_RANGE ===")
    print(target_corr.round(3).to_string())

    charts = []

    # Grafik 1: Sepuluh fitur dengan korelasi absolut tertinggi ke target.
    top_10 = target_corr.head(10).sort_values()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(top_10.index, top_10.values)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlim(-1, 1)
    ax.set_xlabel("Koefisien korelasi Pearson dengan price_range")
    ax.set_title("10 Fitur dengan Hubungan Terkuat ke Target")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    charts.append(("Korelasi target", save_chart(fig, "01_korelasi_target.png")))

    # Grafik 2: Heatmap dari 10 fitur teratas dan target.
    columns = target_corr.head(10).index.tolist() + ["price_range"]
    corr_matrix = df[columns].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10.5, 8.5))
    heatmap = ax.imshow(corr_matrix.to_numpy(), cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(columns)), labels=columns, rotation=60, ha="right")
    ax.set_yticks(np.arange(len(columns)), labels=columns)
    for row in range(len(columns)):
        for col in range(len(columns)):
            value = corr_matrix.iloc[row, col]
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if abs(value) > 0.65 else "black",
            )
    fig.colorbar(heatmap, ax=ax, shrink=0.8, label="Korelasi Pearson")
    ax.set_title("Heatmap Korelasi 10 Fitur Teratas dan Target", pad=14)
    fig.tight_layout()
    charts.append(("Heatmap korelasi", save_chart(fig, "02_heatmap_korelasi.png")))

    # Grafik 3: Distribusi RAM untuk tiap kategori harga.
    classes = sorted(df["price_range"].dropna().unique())
    groups = [df.loc[df["price_range"] == category, "ram"].dropna() for category in classes]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.boxplot(
        groups,
        tick_labels=[str(int(category)) for category in classes],
        patch_artist=True,
        boxprops={"facecolor": "lightsteelblue"},
    )
    ax.set_title("Perbandingan RAM pada Tiap Kategori Harga")
    ax.set_xlabel("Price range (0 = terendah, 3 = tertinggi)")
    ax.set_ylabel("RAM (MB)")
    ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    charts.append(("Boxplot RAM", save_chart(fig, "03_boxplot_ram.png")))

    # Grafik 4: Distribusi kapasitas baterai per kategori.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for category in classes:
        values = df.loc[df["price_range"] == category, "battery_power"].dropna()
        ax.hist(values, bins=22, alpha=0.45, label=f"Kelas {int(category)}")
    ax.set_title("Distribusi Battery Power Menurut Kategori Harga")
    ax.set_xlabel("Battery power (mAh)")
    ax.set_ylabel("Frekuensi")
    ax.legend()
    ax.grid(axis="y", alpha=0.18)
    fig.tight_layout()
    charts.append(("Histogram baterai", save_chart(fig, "04_histogram_battery.png")))

    print("\n=== KANDIDAT OUTLIER (METODE IQR) ===")
    print(detect_iqr_outliers(df).to_string(index=False))
    print("Catatan: kandidat outlier belum tentu data salah; belum ada data yang dihapus.")

    print("\n=== RINGKASAN UNTUK LAPORAN ===")
    print(f"- Korelasi absolut tertinggi dengan target: {target_corr.index[0]} ({target_corr.iloc[0]:.3f}).")
    print("- Korelasi menunjukkan hubungan statistik, bukan sebab-akibat.")
    print("- Keputusan penanganan kandidat outlier dilakukan pada tahap preprocessing.")
    print(f"- Grafik tersimpan di: {OUTPUT_DIR}")

    open_gallery(charts)


if __name__ == "__main__":
    main()
