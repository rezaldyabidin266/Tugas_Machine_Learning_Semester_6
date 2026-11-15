# -*- coding: utf-8 -*-
# Versi Python dari notebook dengan nama yang sama.
from IPython.display import display


# %% [markdown]
# <a href="https://colab.research.google.com/github/pradityaputra888-lab/latihan-machine-learning/blob/main/pertemuan2/latihan2MachineLearning.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown]
# ## Persiapan: Import library

# %% (sel 3)
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from IPython.display import display


# %% [markdown]
# ## 1. Memuat dataset

# %% (sel 5)
from pathlib import Path

# Mendukung script Python dan notebook yang dijalankan dari folder proyek/pertemuan.
BASE_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_DIR = next(
    (p for p in (BASE_DIR, BASE_DIR.parent) if (p / "datasets").is_dir()),
    BASE_DIR.parent,
)

# Dataset lokal; Google Colab tetap mendukung upload jika file belum tersedia
nama_file = "WA_Fn-UseC_-Telco-Customer-Churn.csv"
jalur_file = PROJECT_DIR / "datasets" / nama_file

# Jika file belum ada, minta pengguna mengunggah file CSV
if not os.path.exists(jalur_file):
    try:
        from google.colab import files
        print(f"Silakan unggah file: {nama_file}")
        berkas_diunggah = files.upload()
        kandidat_csv = [
            nama for nama in berkas_diunggah
            if nama.lower().endswith(".csv")
        ]
        if not kandidat_csv:
            raise FileNotFoundError(
                "Tidak ada file CSV yang diunggah. Unggah dataset Telco Customer Churn."
            )
        jalur_file = kandidat_csv[0]
    except ImportError as exc:
        raise FileNotFoundError(
            f"File {nama_file} tidak ditemukan. Letakkan file CSV di folder kerja."
        ) from exc

df = pd.read_csv(jalur_file)

# Ubah TotalCharges menjadi numerik; nilai kosong/berisi spasi menjadi NaN
if "TotalCharges" in df.columns:
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

if "Churn" not in df.columns:
    raise ValueError(
        "Kolom target 'Churn' tidak ditemukan. Pastikan file yang diunggah adalah dataset Telco Customer Churn."
    )

df = df.dropna(subset=["Churn"]).copy()

print("Ukuran dataset:", df.shape)
print("\nLima baris pertama:")
display(df.head())

print("\nDistribusi target Churn:")
print(df["Churn"].value_counts())
print("\nProporsi target (%):")
print((df["Churn"].value_counts(normalize=True) * 100).round(2))


# %% [markdown]
# ## 2. Menentukan fitur dan membagi data (split + stratify)

# %% (sel 7)
# Pisahkan fitur (X) dan target (y)
kolom_dihapus = ["Churn", "customerID"]
X = df.drop(columns=kolom_dihapus, errors="ignore").copy()

# Target: No = 0, Yes = 1
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Target Churn memiliki label di luar kategori 'No' dan 'Yes'.")

# Pembagian data: 80% train, 20% test dengan stratifikasi
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Jumlah seluruh data :", len(X))
print("Data training       :", X_train.shape)
print("Data testing        :", X_test.shape)

print("\nProporsi kelas pada seluruh data:")
print(y.value_counts(normalize=True).round(3))

print("\nProporsi kelas pada data training:")
print(y_train.value_counts(normalize=True).round(3))

print("\nProporsi kelas pada data testing:")
print(y_test.value_counts(normalize=True).round(3))


# %% [markdown]
# ## 3. Melatih Decision Tree dengan depth 1, 4, dan ∞

# %% (sel 9)
# Kenali tipe kolom dari data training
kolom_kategorikal = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

kolom_numerik = X_train.select_dtypes(
    exclude=["object", "category"]
).columns.tolist()

print("Kolom numerik:", kolom_numerik)
print("\nKolom kategorikal:", kolom_kategorikal)

def buat_preprocessing():
    # Membuat objek preprocessing baru untuk setiap model
    return ColumnTransformer(
        transformers=[
            (
                "numerik",
                SimpleImputer(strategy="median"),
                kolom_numerik
            ),
            (
                "kategorikal",
                Pipeline(steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]),
                kolom_kategorikal
            )
        ],
        remainder="drop"
    )

# Tiga tingkat kompleksitas sesuai instruksi praktikum
konfigurasi_model = [
    ("Depth 1", 1),
    ("Depth 4", 4),
    ("Depth ∞", None)
]

# Stratified 5-fold cross-validation pada data training
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

hasil = {}

for nama_model, depth in konfigurasi_model:
    model = Pipeline(steps=[
        ("preprocessing", buat_preprocessing()),
        ("tree", DecisionTreeClassifier(
            max_depth=depth,
            random_state=42
        ))
    ])

    # Fit pada seluruh data training untuk memperoleh skor train/test
    model.fit(X_train, y_train)
    prediksi_train = model.predict(X_train)
    prediksi_test = model.predict(X_test)

    akurasi_train = accuracy_score(y_train, prediksi_train)
    akurasi_test = accuracy_score(y_test, prediksi_test)

    # Learning curve dihitung hanya dengan X_train dan y_train.
    # Setiap fold akan mem-fit preprocessing dari bagian train fold-nya sendiri.
    train_sizes, train_scores, valid_scores = learning_curve(
        estimator=model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=cv,
        scoring="accuracy",
        n_jobs=-1
    )

    hasil[nama_model] = {
        "depth": depth,
        "train_sizes": train_sizes,
        "train_mean": train_scores.mean(axis=1),
        "train_std": train_scores.std(axis=1),
        "valid_mean": valid_scores.mean(axis=1),
        "valid_std": valid_scores.std(axis=1),
        "akurasi_train": akurasi_train,
        "akurasi_test": akurasi_test
    }

print("Pelatihan tiga model selesai.")


# %% [markdown]
# ## 4. Plot learning curve

# %% (sel 11)
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

for ax, (nama_model, _) in zip(axes, konfigurasi_model):
    data = hasil[nama_model]
    ukuran = data["train_sizes"]
    train_mean = data["train_mean"]
    train_std = data["train_std"]
    valid_mean = data["valid_mean"]
    valid_std = data["valid_std"]

    ax.plot(
        ukuran, train_mean,
        marker="o", linewidth=2,
        label="Training"
    )
    ax.fill_between(
        ukuran,
        np.maximum(0, train_mean - train_std),
        np.minimum(1, train_mean + train_std),
        alpha=0.15
    )

    ax.plot(
        ukuran, valid_mean,
        marker="o", linewidth=2,
        label="Validation"
    )
    ax.fill_between(
        ukuran,
        np.maximum(0, valid_mean - valid_std),
        np.minimum(1, valid_mean + valid_std),
        alpha=0.15
    )

    ax.set_title(f"Decision Tree — {nama_model}")
    ax.set_xlabel("Jumlah data training")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend()

axes[0].set_ylabel("Accuracy")
fig.suptitle("Learning Curve — Telco Customer Churn", fontsize=15)
plt.tight_layout()
plt.show()


# %% [markdown]
# ## 5. Tabel hasil dan diagnosis tiga model

# %% (sel 13)
baris_tabel = []

for nama_model, _ in konfigurasi_model:
    data = hasil[nama_model]
    train_akhir = float(data["train_mean"][-1])
    valid_akhir = float(data["valid_mean"][-1])
    gap = train_akhir - valid_akhir

    baris_tabel.append({
        "Model": nama_model,
        "Train accuracy": round(data["akurasi_train"], 4),
        "Test accuracy": round(data["akurasi_test"], 4),
        "Learning curve train akhir": round(train_akhir, 4),
        "Learning curve validation akhir": round(valid_akhir, 4),
        "Gap train-validation": round(gap, 4)
    })

tabel_hasil = pd.DataFrame(baris_tabel)
display(tabel_hasil)


# %% (sel 14)
def diagnosis_model(train_curve, valid_curve):
    train_akhir = float(train_curve[-1])
    valid_akhir = float(valid_curve[-1])
    gap = train_akhir - valid_akhir

    # Kriteria indikatif untuk membantu membaca pola learning curve.
    if train_akhir < 0.75 and valid_akhir < 0.75 and abs(gap) <= 0.05:
        return (
            "Indikasi UNDERFITTING / bias tinggi: "
            "akurasi training dan validation sama-sama relatif rendah dan berdekatan."
        )
    elif train_akhir >= 0.90 and gap >= 0.10:
        return (
            "Indikasi OVERFITTING / variance tinggi: "
            "akurasi training tinggi, tetapi validation tertinggal jauh."
        )
    elif abs(gap) <= 0.05 and valid_akhir >= 0.75:
        return (
            "Relatif SEIMBANG / generalisasi cukup baik: "
            "skor training dan validation berdekatan dan validation relatif tinggi."
        )
    else:
        return (
            "Pola CAMPURAN: periksa grafik, besar gap, dan kestabilan validation. "
            "Belum cukup untuk memberi label tegas hanya dari satu angka."
        )

print("DIAGNOSIS LEARNING CURVE")
print("=" * 70)

for nama_model, _ in konfigurasi_model:
    data = hasil[nama_model]
    train_akhir = data["train_mean"][-1]
    valid_akhir = data["valid_mean"][-1]
    gap = train_akhir - valid_akhir

    print(f"\n{nama_model}")
    print(f"Training curve akhir   : {train_akhir:.3f}")
    print(f"Validation curve akhir : {valid_akhir:.3f}")
    print(f"Gap train-validation   : {gap:.3f}")
    print(diagnosis_model(data["train_mean"], data["valid_mean"]))

print("\nCatatan:")
print("- Diagnosis didasarkan pada learning curve, bukan kedalaman saja.")
print("- Gunakan test accuracy sebagai evaluasi akhir; jangan terus memilih model berdasarkan test set.")
print("- Ambang di atas bersifat indikatif, sehingga tetap periksa grafik dan konteks.")

