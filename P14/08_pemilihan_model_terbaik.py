"""Tahap 08 — Pemilihan Model Terbaik (mandiri).
Jalankan python 08_pemilihan_model_terbaik.py; notebook .ipynb memiliki fungsi yang sama.
Dataset: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
import hashlib

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.widgets import Button
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier



def find_project_root():
    """Works for .py from any CWD, VS Code notebooks, and Google Colab."""
    if "__file__" in globals():
        return Path(__file__).resolve().parent
    here = Path.cwd()
    candidates = (here, here / "P14", here.parent / "P14")
    for candidate in candidates:
        if (candidate / "Mobile_Price_Classification" / "train.csv").is_file():
            return candidate
        if (candidate / "train.csv").is_file():
            return candidate
    if (here / "P14").is_dir():
        return here / "P14"
    return here


def get_dataset_path():
    """train.csv must contain price_range (Kaggle test.csv has no labels)."""
    candidates = (
        BASE / "Mobile_Price_Classification" / "train.csv",
        BASE / "train.csv",
    )
    found = next((candidate for candidate in candidates if candidate.is_file()), None)
    if found is None and "google.colab" in sys.modules:
        from google.colab import files
        print("Upload train.csv (bukan test.csv) dari Kaggle Mobile Price Classification:")
        uploaded = files.upload()
        if "train.csv" not in uploaded:
            raise FileNotFoundError("Nama file yang di-upload harus train.csv")
        found = Path.cwd() / "train.csv"
    if found is None:
        raise FileNotFoundError(
            "train.csv tidak ditemukan. Simpan pada:\n"
            f"  {BASE / 'Mobile_Price_Classification' / 'train.csv'}\n"
            "atau (khusus notebook/Colab) pada folder kerja saat ini.\n"
            "Dataset: https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification"
        )
    return found

BASE = find_project_root()
DATA_DIR = BASE / "Mobile_Price_Classification"
REPORT_DIR = BASE / "reports"
MODEL_DIR = BASE / "models"
FIGURE_DIR = BASE / "figures"
RANDOM_STATE = 42
TARGET = "price_range"

BASE_FEATURES = [
    "battery_power", "blue", "clock_speed", "dual_sim", "fc", "four_g",
    "int_memory", "m_dep", "mobile_wt", "n_cores", "pc", "px_height",
    "px_width", "ram", "sc_h", "sc_w", "talk_time", "three_g",
    "touch_screen", "wifi",
]
NEW_FEATURES = [
    "pixel_area", "screen_area", "battery_per_weight", "camera_total",
    "connectivity_count",
]
ALL_FEATURES = BASE_FEATURES + NEW_FEATURES
CLASS_LABELS = ["0 - Low", "1 - Medium", "2 - High", "3 - Very High"]
MODELS = {"Decision Tree": "decision_tree", "Random Forest": "random_forest", "SVM": "svm"}


def ensure_dirs():
    for directory in (REPORT_DIR, MODEL_DIR, FIGURE_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def load_raw_dataset():
    path = get_dataset_path()
    frame = pd.read_csv(path)
    missing = [name for name in BASE_FEATURES + [TARGET] if name not in frame.columns]
    if missing:
        raise ValueError("Kolom dataset tidak lengkap: " + ", ".join(missing))
    if frame[TARGET].isna().any():
        raise ValueError("Target price_range memiliki missing value; periksa dataset")
    frame = frame[BASE_FEATURES + [TARGET]].copy()
    frame[TARGET] = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    if not set(frame[TARGET].unique()).issubset({0, 1, 2, 3}):
        raise ValueError("price_range harus berisi kelas 0, 1, 2, dan 3")
    for column in BASE_FEATURES:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame


def load_split():
    """Use Stage 03 raw split, or recreate its exact deterministic split."""
    ensure_dirs()
    raw = load_raw_dataset()
    # Local P14: Stage 03 saved these CSVs. Colab without prior stages: regenerate.
    dataset_file = get_dataset_path()
    processed = dataset_file.parent / "processed"
    paths = {name: processed / f"{name}.csv" for name in ("X_train", "X_test", "y_train", "y_test")}
    if all(path.exists() for path in paths.values()):
        x_train = pd.read_csv(paths["X_train"])
        x_test = pd.read_csv(paths["X_test"])
        y_train = pd.read_csv(paths["y_train"])[TARGET]
        y_test = pd.read_csv(paths["y_test"])[TARGET]
        if (list(x_train.columns) == BASE_FEATURES and list(x_test.columns) == BASE_FEATURES
                and len(x_train) + len(x_test) == len(raw)
                and len(x_train) == len(y_train) and len(x_test) == len(y_test)):
            print("Menggunakan split Stage 03 dari:", processed)
            return x_train, x_test, y_train.astype(int), y_test.astype(int)
        print("Split Stage 03 tidak sesuai; membuat ulang split dari train.csv")
    x_train, x_test, y_train, y_test = train_test_split(
        raw[BASE_FEATURES], raw[TARGET], test_size=0.2, random_state=RANDOM_STATE,
        stratify=raw[TARGET],
    )
    processed.mkdir(parents=True, exist_ok=True)
    for name, part in {
        "X_train": x_train, "X_test": x_test,
        "y_train": y_train.to_frame(name=TARGET), "y_test": y_test.to_frame(name=TARGET),
    }.items():
        part.to_csv(paths[name], index=False)
    print("Split 80:20 dibuat dan disimpan:", processed)
    return x_train, x_test, y_train.astype(int), y_test.astype(int)


class MobileFeatureEngineer(BaseEstimator, TransformerMixin):
    """Stateless, row-by-row construction; safe inside cross-validation pipelines."""

    def fit(self, X, y=None):
        self._validate(X)
        return self

    def _validate(self, X):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("Model membutuhkan input DataFrame dengan nama 20 fitur asli")
        missing = set(BASE_FEATURES) - set(X.columns)
        if missing:
            raise ValueError("Input fitur tidak lengkap: " + ", ".join(sorted(missing)))

    def transform(self, X):
        self._validate(X)
        result = X[BASE_FEATURES].copy()
        for feature in BASE_FEATURES:
            result[feature] = pd.to_numeric(result[feature], errors="coerce")
        result["pixel_area"] = result["px_height"] * result["px_width"]
        result["screen_area"] = result["sc_h"] * result["sc_w"]
        result["battery_per_weight"] = result["battery_power"] / result["mobile_wt"].clip(lower=1)
        result["camera_total"] = result["fc"] + result["pc"]
        result["connectivity_count"] = result[["blue", "four_g", "three_g", "wifi"]].sum(axis=1, min_count=4)
        return result[ALL_FEATURES]

    def get_feature_names_out(self, input_features=None):
        return np.asarray(ALL_FEATURES, dtype=object)


def cv_strategy():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def model_pipeline(model_name):
    if model_name == "Decision Tree":
        estimator = DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE)
    elif model_name == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=150, n_jobs=1, random_state=RANDOM_STATE)
    elif model_name == "SVM":
        estimator = SVC(kernel="rbf", C=1.0, gamma="scale", random_state=RANDOM_STATE)
    else:
        raise ValueError("Algoritma tidak dikenal: " + model_name)
    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
    if model_name == "SVM":
        numeric_steps.append(("scaler", StandardScaler()))
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline(numeric_steps), ALL_FEATURES),
    ], remainder="drop")
    # Fitur turunan dihitung secara deterministik sebelum CV.
    # Fitting imputer/scaler tetap di dalam Pipeline untuk mencegah leakage.
    return Pipeline([
        ("preprocessor", preprocessor),
        ("model", estimator),
    ])


def parameter_grid(model_name):
    quick = os.environ.get("P14_FAST_TEST") == "1"
    if model_name == "Decision Tree":
        return {"model__max_depth": [6] if quick else [5, 8, 10, None],
                "model__min_samples_leaf": [1] if quick else [1, 3]}
    if model_name == "Random Forest":
        return {"model__n_estimators": [20] if quick else [100, 150],
                "model__max_depth": [None] if quick else [None, 12],
                "model__min_samples_leaf": [1] if quick else [1, 3]}
    if model_name == "SVM":
        return {"model__C": [1] if quick else [0.5, 1, 5, 10],
                "model__gamma": ["scale"] if quick else ["scale", 0.01]}
    raise ValueError("Algoritma tidak dikenal: " + model_name)


def train_baselines():
    """Stage 05 results; evaluate by CV on training only."""
    ensure_dirs()
    csv_file = REPORT_DIR / "05_baseline_cv.csv"
    meta_file = REPORT_DIR / "05_baseline_metadata.json"
    fingerprint = hashlib.sha256(get_dataset_path().read_bytes()).hexdigest()
    expected_meta = {"code_version": "P14_FIXED_20261102_v3", "data_sha256": fingerprint}
    if (csv_file.exists() and meta_file.is_file()
            and all((MODEL_DIR / f"05_baseline_{slug}.joblib").is_file() for slug in MODELS.values())):
        try:
            if json.loads(meta_file.read_text(encoding="utf-8")) == expected_meta:
                print("Hasil baseline tahap 05 sudah valid; memakai hasil yang tersimpan.")
                return pd.read_csv(csv_file)
        except (ValueError, OSError):
            pass
    X_train, _, y_train, _ = load_split()
    X_train = MobileFeatureEngineer().fit_transform(X_train)
    records = []
    for name, slug in MODELS.items():
        print(f"Baseline CV 5-fold: {name}")
        model = model_pipeline(name)
        scores = cross_validate(
            model, X_train, y_train, cv=cv_strategy(),
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
            n_jobs=1, error_score="raise", return_train_score=False,
        )
        record = {"model": name,
                  "f1_macro_cv_mean": float(scores["test_f1_macro"].mean()),
                  "f1_macro_cv_std": float(scores["test_f1_macro"].std()),
                  "accuracy_cv_mean": float(scores["test_accuracy"].mean()),
                  "balanced_accuracy_cv_mean": float(scores["test_balanced_accuracy"].mean()),
                  "fit_seconds_mean": float(scores["fit_time"].mean())}
        for i, value in enumerate(scores["test_f1_macro"], 1):
            record[f"f1_fold_{i}"] = float(value)
        model.fit(X_train, y_train)
        joblib.dump(model, MODEL_DIR / f"05_baseline_{slug}.joblib")
        records.append(record)
    table = pd.DataFrame(records).sort_values("f1_macro_cv_mean", ascending=False)
    table.to_csv(csv_file, index=False)
    meta_file.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return table


def tune_models():
    """Stage 06: GridSearchCV on training only; never inspect held-out test."""
    ensure_dirs()
    csv_file = REPORT_DIR / "06_tuned_cv.csv"
    meta_file = REPORT_DIR / "06_tuned_metadata.json"
    fingerprint = hashlib.sha256(get_dataset_path().read_bytes()).hexdigest()
    expected_meta = {"code_version": "P14_FIXED_20261102_v3", "data_sha256": fingerprint,
                     "fast_test": os.environ.get("P14_FAST_TEST") == "1"}
    if (csv_file.exists() and meta_file.is_file()
            and all((MODEL_DIR / f"06_tuned_{slug}.joblib").is_file() for slug in MODELS.values())):
        try:
            if json.loads(meta_file.read_text(encoding="utf-8")) == expected_meta:
                print("Model tuning tahap 06 sudah valid; memakai hasil yang tersimpan.")
                return pd.read_csv(csv_file)
        except (ValueError, OSError):
            pass
    X_train, _, y_train, _ = load_split()
    X_train = MobileFeatureEngineer().fit_transform(X_train)
    records = []
    for name, slug in MODELS.items():
        print(f"GridSearchCV 5-fold: {name}", flush=True)
        search = GridSearchCV(
            estimator=model_pipeline(name), param_grid=parameter_grid(name),
            scoring={"f1_macro": "f1_macro", "accuracy": "accuracy", "balanced_accuracy": "balanced_accuracy"},
            refit="f1_macro", cv=cv_strategy(), n_jobs=1, error_score="raise",
            return_train_score=False,
        )
        started = time.perf_counter()
        search.fit(X_train, y_train)
        duration = time.perf_counter() - started
        best = search.best_index_
        record = {"model": name,
                  "f1_macro_cv_mean": float(search.cv_results_["mean_test_f1_macro"][best]),
                  "f1_macro_cv_std": float(search.cv_results_["std_test_f1_macro"][best]),
                  "accuracy_cv_mean": float(search.cv_results_["mean_test_accuracy"][best]),
                  "balanced_accuracy_cv_mean": float(search.cv_results_["mean_test_balanced_accuracy"][best]),
                  "fit_seconds_mean": float(search.cv_results_["mean_fit_time"][best]),
                  "total_search_seconds": duration,
                  "best_params": json.dumps(search.best_params_, ensure_ascii=True),
                  "n_candidates": len(search.cv_results_["params"])}
        records.append(record)
        joblib.dump(search.best_estimator_, MODEL_DIR / f"06_tuned_{slug}.joblib")
        details = pd.DataFrame(search.cv_results_)
        keep = [k for k in details.columns if k.startswith("param_") or k in (
            "mean_test_f1_macro", "std_test_f1_macro", "mean_test_accuracy", "rank_test_f1_macro")]
        details[keep].sort_values("rank_test_f1_macro").to_csv(REPORT_DIR / f"06_grid_{slug}.csv", index=False)
        print(f"  -> F1 macro CV = {record['f1_macro_cv_mean']:.4f}; params = {search.best_params_}")
    table = pd.DataFrame(records).sort_values("f1_macro_cv_mean", ascending=False)
    table.to_csv(csv_file, index=False)
    meta_file.write_text(json.dumps(expected_meta, indent=2), encoding="utf-8")
    return table


def save_figure(figure, stage, filename, paths):
    folder = FIGURE_DIR / stage
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / filename
    figure.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(figure)
    paths.append(path)
    return path


def view_gallery(image_paths):
    if not image_paths:
        return
    if "google.colab" in sys.modules or "ipykernel" in sys.modules:
        from IPython.display import display, Image
        for p in image_paths:
            print(p.name)
            display(Image(filename=str(p)))
        return
    if os.environ.get("P14_NO_GUI") == "1" or matplotlib.get_backend().lower() == "agg":
        print(f"Grafik tersimpan ({len(image_paths)} PNG). Tidak membuka window pada mode tanpa GUI.")
        return
    index = [0]
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.subplots_adjust(bottom=0.17, top=0.91)
    ax.set_axis_off()
    first = plt.imread(image_paths[0])
    image = ax.imshow(first, interpolation="nearest")
    ax.set_xlim(0, first.shape[1])
    ax.set_ylim(first.shape[0], 0)
    title = ax.set_title(f"1/{len(image_paths)} - {image_paths[0].stem}")
    prev_axis = fig.add_axes((0.32, 0.045, 0.15, 0.065))
    next_axis = fig.add_axes((0.53, 0.045, 0.15, 0.065))
    previous = Button(prev_axis, "Previous")
    following = Button(next_axis, "Next")

    def move(by):
        index[0] = (index[0] + by) % len(image_paths)
        pixels = plt.imread(image_paths[index[0]])
        image.set_data(pixels)
        image.set_extent((0, pixels.shape[1], pixels.shape[0], 0))
        ax.set_xlim(0, pixels.shape[1])
        ax.set_ylim(pixels.shape[0], 0)
        title.set_text(f"{index[0]+1}/{len(image_paths)} - {image_paths[index[0]].stem}")
        fig.canvas.draw_idle()

    previous.on_clicked(lambda event: move(-1))
    following.on_clicked(lambda event: move(1))
    fig.canvas.mpl_connect("key_press_event", lambda event: move(-1) if event.key == "left" else (move(1) if event.key == "right" else None))
    print("Navigasi: klik Previous/Next atau gunakan tombol panah kiri/kanan.")
    plt.show()


def cli_summary(table, digits=4):
    with pd.option_context("display.max_columns", 12, "display.width", 125, "display.float_format", f"{{:.{digits}f}}".format):
        print(table.to_string(index=False))


# ==================== KODE TAHAP 08 ====================


"""Tahap 08: pilih dari CV TRAINING; evaluasi akhir sekali pada TEST holdout.

Output model terbaik (pipeline lengkap), confusion matrix, classification report,
precision/recall/F1 tiap kelas, permutation importance, ringkasan ilmiah.
"""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score,
    classification_report, confusion_matrix, f1_score, precision_score, recall_score,
)



def main():
    train_baselines()
    tuned = tune_models().sort_values(
        ["f1_macro_cv_mean", "accuracy_cv_mean"], ascending=[False, False]
    )
    winner = tuned.iloc[0]
    selected_name = winner["model"]
    selected_slug = MODELS[selected_name]
    trained_model = joblib.load(MODEL_DIR / f"06_tuned_{selected_slug}.joblib")

    # Only now load the held-out test partition to produce final evaluation.
    _, x_test, _, y_test = load_split()
    x_test_features = MobileFeatureEngineer().fit_transform(x_test)
    prediction = trained_model.predict(x_test_features)
    labels = sorted(y_test.unique())
    report = classification_report(y_test, prediction, labels=labels, zero_division=0, output_dict=True)
    metrics = {
        "model_terpilih": selected_name,
        "n_fitur_model": len(ALL_FEATURES),
        "catatan_inferensi": "Input model_terbaik.joblib harus memuat 25 fitur (20 asli + 5 fitur turunan).",
        "pemilihan_berdasarkan": "F1 macro 5-fold CV pada data train saja",
        "cv_f1_macro": float(winner["f1_macro_cv_mean"]),
        "cv_f1_std": float(winner["f1_macro_cv_std"]),
        "accuracy_test": float(accuracy_score(y_test, prediction)),
        "precision_macro_test": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
        "recall_macro_test": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        "f1_macro_test": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "balanced_accuracy_test": float(balanced_accuracy_score(y_test, prediction)),
        "n_train": int(len(load_split()[0])),
        "n_test": int(len(y_test)),
        "random_state": 42,
        "best_params": json.loads(winner["best_params"]),
        "label_kelas": {"0": "Low", "1": "Medium", "2": "High", "3": "Very High"},
        "catatan": "Kategori harga relatif, tidak merepresentasikan harga rupiah; data Kaggle memiliki fitur spesifikasi saja.",
    }
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_model, MODEL_DIR / "model_terbaik.joblib")
    (REPORT_DIR / "08_evaluasi_final.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    pd.DataFrame(report).transpose().to_csv(REPORT_DIR / "08_classification_report.csv")
    output_predictions = x_test.copy()
    output_predictions["actual_price_range"] = np.asarray(y_test)
    output_predictions["predicted_price_range"] = prediction
    output_predictions.to_csv(REPORT_DIR / "08_test_predictions.csv", index=False)

    print("=== 08 MODEL FINAL TERPILIH ===")
    print("Dipilih berdasarkan F1 macro CV:", selected_name)
    print("CV F1 Macro:", f"{metrics['cv_f1_macro']:.4f}")
    print("Holdout Test Accuracy:", f"{metrics['accuracy_test']:.4f}")
    print("Holdout Test F1 Macro:", f"{metrics['f1_macro_test']:.4f}")
    print("Holdout Test Precision Macro:", f"{metrics['precision_macro_test']:.4f}")
    print("Holdout Test Recall Macro:", f"{metrics['recall_macro_test']:.4f}")
    print("\nClassification report (test):\n")
    print(classification_report(y_test, prediction, labels=labels, zero_division=0))

    images = []
    confusion = confusion_matrix(y_test, prediction, labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=[CLASS_LABELS[k] for k in labels]).plot(
        ax=ax, cmap="Blues", values_format="d", colorbar=False,
    )
    ax.set_title(f"Confusion Matrix - {selected_name} (Test)")
    ax.tick_params(axis="x", labelrotation=20)
    fig.tight_layout()
    save_figure(fig, "08_pemilihan_model", "01_confusion_matrix_test.png", images)

    frame = pd.DataFrame({
        "kelas": [CLASS_LABELS[k] for k in labels],
        "precision": [report[str(k)]["precision"] for k in labels],
        "recall": [report[str(k)]["recall"] for k in labels],
        "f1": [report[str(k)]["f1-score"] for k in labels],
    })
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10, 5))
    for dx, key in [(-0.25, "precision"), (0, "recall"), (0.25, "f1")]:
        ax.bar(x + dx, frame[key], width=0.25, label=key.title())
    ax.set_xticks(x, frame["kelas"])
    ax.set(title="Precision / Recall / F1 Per Kelas (Test)", ylabel="Skor", ylim=(0, 1.08))
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "08_pemilihan_model", "02_metrik_per_kelas.png", images)

    # After model selection, interpret on held-out test. Do not use for re-selection.
    imp = permutation_importance(
        trained_model, x_test_features, y_test, n_repeats=4, random_state=42,
        scoring="f1_macro", n_jobs=1,
    )
    importance = pd.DataFrame({
        "feature": ALL_FEATURES,
        "mean_decrease_f1_macro": imp.importances_mean,
        "std": imp.importances_std,
    }).sort_values("mean_decrease_f1_macro", ascending=False)
    importance.to_csv(REPORT_DIR / "08_permutation_importance.csv", index=False)
    top = importance.head(12).sort_values("mean_decrease_f1_macro")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_decrease_f1_macro"], xerr=top["std"])
    ax.set(title="Permutation Importance - Penurunan F1 Macro (Test)", xlabel="Perubahan skor saat satu fitur diacak")
    fig.tight_layout()
    save_figure(fig, "08_pemilihan_model", "03_permutation_importance.png", images)

    fig, ax = plt.subplots(figsize=(9, 5))
    actual = pd.Series(y_test).value_counts().sort_index().reindex(labels, fill_value=0)
    predicted = pd.Series(prediction).value_counts().sort_index().reindex(labels, fill_value=0)
    ax.bar(x - 0.17, actual, width=0.34, label="Aktual")
    ax.bar(x + 0.17, predicted, width=0.34, label="Prediksi")
    ax.set_xticks(x, [CLASS_LABELS[k] for k in labels])
    ax.set(title="Distribusi Aktual vs Prediksi pada Test Set", ylabel="Jumlah smartphone")
    ax.legend()
    fig.tight_layout()
    save_figure(fig, "08_pemilihan_model", "04_aktual_vs_prediksi.png", images)

    report_text = (
        "RINGKASAN HASIL UNTUK LAPORAN ILMIAH\n"
        "================================\n"
        f"Model terpilih: {selected_name}\n"
        f"Dasar pemilihan: rerata 5-fold CV F1 macro = {metrics['cv_f1_macro']:.4f} "
        f"(std={metrics['cv_f1_std']:.4f}) dari data training.\n"
        f"Evaluasi test holdout: Accuracy={metrics['accuracy_test']:.4f}, "
        f"Precision-macro={metrics['precision_macro_test']:.4f}, "
        f"Recall-macro={metrics['recall_macro_test']:.4f}, "
        f"F1-macro={metrics['f1_macro_test']:.4f}.\n"
        "Keterbatasan: dataset Kaggle memuat label kategori harga (0-3), bukan harga "
        "dalam satuan rupiah. Model merupakan studi klasifikasi akademik.\n"
        "Model terbaik dilatih pada 25 fitur hasil feature engineering, "
        "pipeline menyimpan imputasi dan scaling (jika menggunakan SVM).\n"
    )
    (REPORT_DIR / "08_ringkasan_laporan.md").write_text(report_text, encoding="utf-8")
    print("\nHasil disimpan:", MODEL_DIR / "model_terbaik.joblib")
    print("File laporan:", REPORT_DIR / "08_ringkasan_laporan.md")
    view_gallery(images)


if __name__ == "__main__":
    main()
