"""Tahap 03 - Data Preprocessing | Mobile Price Classification.
VS Code: python 03_data_preprocessing.py
Google Colab: jalankan versi .ipynb, lalu upload train.csv.
Tidak melatih model; scaling hanya demonstrasi agar tidak bocor ke test set.
"""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

BASE = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
DATA = BASE / "Mobile_Price_Classification" / "train.csv"
FIGURES = BASE / "figures" / "03_preprocessing"
PROCESSED = BASE / "Mobile_Price_Classification" / "processed"
FIGURES.mkdir(parents=True, exist_ok=True)


def load_dataset():
    global DATA, PROCESSED
    if not DATA.exists():
        DATA = BASE / "train.csv"
    if not DATA.exists() and "google.colab" in sys.modules:
        from google.colab import files
        print("Upload train.csv dari Kaggle:")
        uploaded = files.upload()
        if "train.csv" not in uploaded:
            raise FileNotFoundError("Harus mengunggah train.csv")
        DATA = Path.cwd() / "train.csv"
    if not DATA.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {DATA}")
    PROCESSED = DATA.parent / "processed"
    df = pd.read_csv(DATA)
    if "price_range" not in df.columns:
        raise ValueError("Gunakan train.csv (wajib ada kolom price_range)")
    return df


def save_chart(fig, name, figures):
    path = FIGURES / name
    fig.savefig(path, dpi=160, bbox_inches="tight")
    figures.append(path)
    plt.close(fig)


def show_gallery(figures):
    """Satu window dengan Previous/Next; Colab menampilkan grafik inline."""
    if "google.colab" in sys.modules:
        from IPython.display import display, Image
        for path in figures:
            print(path.name)
            display(Image(filename=str(path)))
        return

    index = [0]
    window, ax = plt.subplots(figsize=(11, 7))
    window.subplots_adjust(bottom=0.16, top=0.92)
    ax.axis("off")
    image = ax.imshow(plt.imread(figures[0]))
    title = ax.set_title(f"1/{len(figures)} - {figures[0].stem}", fontsize=12)
    ax_prev = window.add_axes((0.32, 0.04, 0.15, 0.07))
    ax_next = window.add_axes((0.53, 0.04, 0.15, 0.07))
    prev_btn = Button(ax_prev, "Previous")
    next_btn = Button(ax_next, "Next")

    def move(step):
        index[0] = (index[0] + step) % len(figures)
        image.set_data(plt.imread(figures[index[0]]))
        image.set_extent((0, image.get_array().shape[1], image.get_array().shape[0], 0))
        ax.set_xlim(0, image.get_array().shape[1])
        ax.set_ylim(image.get_array().shape[0], 0)
        title.set_text(f"{index[0] + 1}/{len(figures)} - {figures[index[0]].stem}")
        window.canvas.draw_idle()

    prev_btn.on_clicked(lambda _: move(-1))
    next_btn.on_clicked(lambda _: move(1))
    window.canvas.mpl_connect(
        "key_press_event",
        lambda event: move(-1) if event.key == "left" else (
            move(1) if event.key == "right" else None
        ),
    )
    print("Navigasi: tombol Previous/Next atau panah kiri/kanan.")
    plt.show()


def main():
    df = load_dataset()
    figures = []
    target = "price_range"
    print("=== TAHAP 03 - DATA PREPROCESSING ===")
    print(f"Dataset: {DATA}")
    print(f"Ukuran: {df.shape[0]} baris, {df.shape[1]} kolom")

    missing = df.isna().sum()
    duplicated = int(df.duplicated().sum())
    print("\n=== PEMERIKSAAN DATA ===")
    print(f"Total missing value : {int(missing.sum())}")
    print(f"Baris duplikat      : {duplicated}")
    print("Distribusi kelas:\n", df[target].value_counts().sort_index())

    # Semua transformasi yang perlu fit dilakukan setelah split.
    X = df.drop(columns=[target])
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "X_train.csv": X_train, "X_test.csv": X_test,
        "y_train.csv": y_train.to_frame(), "y_test.csv": y_test.to_frame(),
    }.items():
        frame.to_csv(PROCESSED / name, index=False)
    print("\n=== PEMBAGIAN DATA ===")
    print(f"Training: {len(X_train)} data (80%)")
    print(f"Testing : {len(X_test)} data (20%)")

    # Hitung kandidat outlier IQR hanya dari data train.
    numeric = X_train.select_dtypes(include="number")
    continuous = [c for c in numeric.columns if numeric[c].nunique() > 10]
    q1, q3 = numeric[continuous].quantile(0.25), numeric[continuous].quantile(0.75)
    iqr = q3 - q1
    outlier = ((numeric[continuous] < (q1 - 1.5 * iqr)) |
               (numeric[continuous] > (q3 + 1.5 * iqr))).sum().sort_values(ascending=False)
    print("\n=== KANDIDAT OUTLIER (IQR, TRAIN SAJA) ===")
    print(outlier.to_string())
    print("Tidak otomatis menghapus outlier: nilai ekstrem dapat valid secara domain.")

    # Scaling hanya DEMO, fit hanya pada training. Untuk SVM gunakan Pipeline nanti.
    scaler = StandardScaler()
    ram_train_scaled = scaler.fit_transform(X_train[["ram"]]).ravel()
    ram_test_scaled = scaler.transform(X_test[["ram"]]).ravel()
    print("\n=== CONTOH STANDARD SCALING RAM ===")
    print(f"Mean RAM train setelah scaling: {ram_train_scaled.mean():.5f}")
    print(f"Std RAM train setelah scaling : {ram_train_scaled.std():.5f}")
    print(f"Jumlah RAM test yang ditransformasi: {len(ram_test_scaled)}")
    print("CSV yang disimpan tetap RAW; scaling di tahap modeling melalui Pipeline.")

    # Grafik 1 - missing value
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(missing.index, missing.values)
    ax.set(title="Missing Value per Kolom", ylabel="Jumlah missing value")
    ax.set_ylim(0, max(1, int(missing.max()) + 1))
    ax.tick_params(axis="x", labelrotation=75)
    fig.tight_layout()
    save_chart(fig, "01_missing_values.png", figures)

    # Grafik 2 - kelas (persentase, perbandingan dapat adil meski ukuran berbeda)
    fig, ax = plt.subplots(figsize=(8, 5))
    classes = sorted(y.unique())
    xs = np.arange(len(classes))
    w = 0.25
    for offset, labels, name in [(-w, y, "Semua"), (0, y_train, "Train"), (w, y_test, "Test")]:
        dist = labels.value_counts(normalize=True).reindex(classes, fill_value=0) * 100
        ax.bar(xs + offset, dist.values, width=w, label=name)
    ax.set_xticks(xs, [str(k) for k in classes])
    ax.set(title="Proporsi Kelas Setelah Stratified Split", xlabel="price_range", ylabel="Persentase (%)")
    ax.legend()
    fig.tight_layout()
    save_chart(fig, "02_distribusi_train_test.png", figures)

    # Grafik 3 - kandidat outlier
    fig, ax = plt.subplots(figsize=(9, 5))
    outlier.head(10).sort_values().plot.barh(ax=ax)
    ax.set(title="10 Fitur dengan Kandidat Outlier Terbanyak (IQR)", xlabel="Jumlah pada data train")
    fig.tight_layout()
    save_chart(fig, "03_outlier_iqr.png", figures)

    # Grafik 4 - RAM per kelas pada TRAIN
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot([X_train.loc[y_train == k, "ram"] for k in classes], tick_labels=[str(k) for k in classes])
    ax.set(title="Distribusi RAM Menurut Kelas (Train)", xlabel="price_range", ylabel="RAM (MB)")
    fig.tight_layout()
    save_chart(fig, "04_boxplot_ram.png", figures)

    # Grafik 5 - sebelum dan sesudah scaling pada train
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(X_train["ram"], bins=25)
    axes[0].set(title="RAM Asli - Train", xlabel="RAM (MB)", ylabel="Frekuensi")
    axes[1].hist(ram_train_scaled, bins=25)
    axes[1].set(title="RAM StandardScaler - Train", xlabel="Z-score", ylabel="Frekuensi")
    fig.tight_layout()
    save_chart(fig, "05_ram_sebelum_sesudah_scaling.png", figures)

    print(f"\nOutput grafik: {FIGURES}")
    print(f"Output split : {PROCESSED}")
    show_gallery(figures)


if __name__ == "__main__":
    main()
