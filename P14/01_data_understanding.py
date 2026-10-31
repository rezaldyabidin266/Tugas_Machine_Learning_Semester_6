"""Tahap 01 - Data Understanding | Mobile Price Classification.

Cara menjalankan:
    python 01_data_understanding.py

Semua grafik disimpan sebagai PNG, lalu ditampilkan dalam SATU jendela.
Gunakan tombol Previous / Next atau tombol panah kiri / kanan.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.widgets import Button


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "Mobile_Price_Classification" / "train.csv"
OUTPUT_DIR = BASE_DIR / "figures" / "01_data_understanding"
CLASS_LABELS = {
    0: "0 - Low",
    1: "1 - Medium",
    2: "2 - High",
    3: "3 - Very High",
}


def load_data() -> pd.DataFrame:
    if not DATASET_PATH.is_file():
        raise FileNotFoundError(
            f"Dataset tidak ditemukan: {DATASET_PATH}\n"
            "Pastikan train.csv berada di folder Mobile_Price_Classification."
        )

    data = pd.read_csv(DATASET_PATH)
    if "price_range" not in data.columns:
        raise ValueError("Kolom price_range tidak ditemukan. Gunakan train.csv.")
    return data


def save_chart(fig, filename: str) -> Path:
    """Simpan grafik dan tutup figur sementara agar tidak membuka banyak jendela."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / filename
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def open_gallery(charts: list[tuple[str, Path]]) -> None:
    """Satu jendela untuk semua grafik, dengan tombol dan keyboard navigasi."""
    if not charts:
        return

    # PNG sudah dibuat, jadi tidak perlu menggambar ulang grafik saat pindah.
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
        title.set_text(f"Data Understanding  |  {index + 1}/{len(charts)}  |  {charts[index][0]}")
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


def main() -> None:
    df = load_data()
    print(f"Dataset: {DATASET_PATH}")
    print(f"Ukuran: {df.shape}")

    print("\n=== 5 BARIS PERTAMA ===")
    print(df.head().to_string(index=False))
    print(f"Jumlah baris: {len(df):,}")
    print(f"Jumlah kolom: {df.shape[1]} (termasuk 1 target)")
    print(f"Jumlah fitur: {df.shape[1] - 1}")

    print("\n=== INFORMASI DATASET ===")
    df.info()
    print("\n=== JUMLAH KOLOM PER TIPE DATA ===")
    print(df.dtypes.value_counts().to_string())

    missing = df.isna().sum()
    duplicate_count = int(df.duplicated().sum())
    print("\n=== MISSING VALUE ===")
    print(missing.to_string())
    print(f"Total missing value: {int(missing.sum())}")
    print(f"Total baris duplikat: {duplicate_count}")

    counts = df["price_range"].value_counts().sort_index()
    summary = pd.DataFrame(
        {
            "Kategori": [CLASS_LABELS.get(int(k), str(k)) for k in counts.index],
            "Jumlah data": counts.values,
            "Proporsi (%)": (counts.values / len(df) * 100).round(2),
        }
    )
    print("\n=== DISTRIBUSI PRICE_RANGE ===")
    print(summary.to_string(index=False))

    charts = []

    # Grafik 1: Distribusi kelas.
    fig, ax = plt.subplots(figsize=(8.5, 5))
    bars = ax.bar(summary["Kategori"], summary["Jumlah data"])
    ax.bar_label(bars, padding=4)
    ax.set_title("Distribusi Kelas Harga Smartphone")
    ax.set_xlabel("Kategori price_range")
    ax.set_ylabel("Jumlah smartphone")
    ax.set_ylim(0, max(counts.values) * 1.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    charts.append(("Distribusi kelas", save_chart(fig, "01_distribusi_kelas.png")))

    # Grafik 2: Persentase kolom yang terisi (untuk pemeriksaan data kosong).
    completeness = (df.notna().mean() * 100).sort_values()
    fig, ax = plt.subplots(figsize=(9.5, 6.4))
    ax.barh(completeness.index, completeness.values)
    ax.set_xlim(0, 108)
    ax.set_xlabel("Persentase nilai terisi (%)")
    ax.set_title("Kelengkapan Setiap Kolom Dataset")
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    fig.tight_layout()
    charts.append(("Kelengkapan data", save_chart(fig, "02_kelengkapan_kolom.png")))

    print("\n=== RINGKASAN UNTUK LAPORAN ===")
    print(f"- Dataset memiliki {len(df)} baris, {df.shape[1] - 1} fitur dan 1 target.")
    print(f"- Missing value: {int(missing.sum())}; baris duplikat: {duplicate_count}.")
    print(f"- Distribusi kelas {'seimbang' if counts.nunique() == 1 else 'tidak seimbang sempurna'}.")
    print(f"- Grafik tersimpan di: {OUTPUT_DIR}")

    open_gallery(charts)


if __name__ == "__main__":
    main()
