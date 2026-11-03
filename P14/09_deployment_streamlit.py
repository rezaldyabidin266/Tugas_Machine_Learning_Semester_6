"""Tahap 09 - Aplikasi prediksi kategori harga HP menggunakan Streamlit.

Menjalankan (dari folder P14):
    python -m pip install streamlit
    python -m streamlit run 09_deployment_streamlit.py

Membaca model hasil Tahap 08 dari P14/models/model_terbaik.joblib.
Tidak membutuhkan train.csv, test.csv, p14_common.py, atau training ulang.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model_terbaik.joblib"
METRICS_PATH = BASE_DIR / "reports" / "08_evaluasi_final.json"

# URUTAN fitur asli sama dengan Tahap 03-08.
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
CLASS_NAMES = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}

# Hanya untuk contoh nilai formulir, bukan data training tambahan.
DEMO_INPUT = {
    "battery_power": 1450,
    "blue": 1,
    "clock_speed": 2.0,
    "dual_sim": 1,
    "fc": 5,
    "four_g": 1,
    "int_memory": 32,
    "m_dep": 0.6,
    "mobile_wt": 145,
    "n_cores": 4,
    "pc": 13,
    "px_height": 900,
    "px_width": 1440,
    "ram": 2200,
    "sc_h": 12,
    "sc_w": 6,
    "talk_time": 10,
    "three_g": 1,
    "touch_screen": 1,
    "wifi": 1,
}


def build_features(specifications: dict[str, int | float]) -> pd.DataFrame:
    """Persis transformasi 20 -> 25 fitur yang dipakai di Tahap 04-08."""
    missing = [feature for feature in BASE_FEATURES if feature not in specifications]
    if missing:
        raise ValueError("Spesifikasi tidak lengkap: " + ", ".join(missing))

    data = pd.DataFrame([{feature: specifications[feature] for feature in BASE_FEATURES}])
    for feature in BASE_FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="raise")

    # HARUS identik dengan MobileFeatureEngineer.transform() pada Tahap 08.
    data["pixel_area"] = data["px_height"] * data["px_width"]
    data["screen_area"] = data["sc_h"] * data["sc_w"]
    data["battery_per_weight"] = data["battery_power"] / data["mobile_wt"].clip(lower=1)
    data["camera_total"] = data["fc"] + data["pc"]
    data["connectivity_count"] = data[
        ["blue", "four_g", "three_g", "wifi"]
    ].sum(axis=1, min_count=4)
    return data.loc[:, ALL_FEATURES]


@st.cache_resource(show_spinner=False)
def load_trained_model(filename: str, modified_at_ns: int):
    """Baca hanya model lokal hasil Tahap 08 yang dipercaya."""
    return joblib.load(filename)


def read_metrics() -> dict:
    if not METRICS_PATH.is_file():
        return {}
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        return metrics if isinstance(metrics, dict) else {}
    except (OSError, ValueError):
        return {}


def show_sidebar(metrics: dict) -> None:
    with st.sidebar:
        st.header("Informasi model")
        st.success("Menggunakan model hasil Tahap 08")
        st.caption("Sumber: models/model_terbaik.joblib")
        st.write("Algoritma:", metrics.get("model_terpilih", "Lihat hasil Tahap 08"))
        if isinstance(metrics.get("accuracy_test"), (float, int)):
            st.metric("Accuracy test", f"{metrics['accuracy_test']:.2%}")
        if isinstance(metrics.get("f1_macro_test"), (float, int)):
            st.metric("F1-macro test", f"{metrics['f1_macro_test']:.4f}")
        st.divider()
        st.caption("20 fitur input + 5 fitur turunan otomatis")
        st.caption("Dataset: Kaggle Mobile Price Classification")
        st.caption("Model adalah alat klasifikasi akademik, bukan penaksir harga rupiah.")


def get_specifications_form() -> tuple[dict[str, int | float], bool]:
    """Form 20 fitur asli; fitur turunan tidak perlu diisi pengguna."""
    data: dict[str, int | float] = {}

    with st.form("mobile_price_form"):
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Performa, memori, dan baterai")
            data["ram"] = st.number_input(
                "RAM (MB)", min_value=1, max_value=16384, value=DEMO_INPUT["ram"], step=128,
            )
            data["int_memory"] = st.number_input(
                "Memori internal (GB)", min_value=1, max_value=1024,
                value=DEMO_INPUT["int_memory"], step=1,
            )
            data["battery_power"] = st.number_input(
                "Kapasitas baterai (mAh)", min_value=1, max_value=15000,
                value=DEMO_INPUT["battery_power"], step=50,
            )
            data["clock_speed"] = st.number_input(
                "Clock speed (GHz)", min_value=0.1, max_value=6.0,
                value=DEMO_INPUT["clock_speed"], step=0.1, format="%.1f",
            )
            data["n_cores"] = st.number_input(
                "Jumlah core prosesor", min_value=1, max_value=32,
                value=DEMO_INPUT["n_cores"], step=1,
            )
            data["mobile_wt"] = st.number_input(
                "Berat HP (gram)", min_value=1, max_value=500,
                value=DEMO_INPUT["mobile_wt"], step=1,
            )
            data["m_dep"] = st.number_input(
                "Ketebalan HP (cm)", min_value=0.1, max_value=3.0,
                value=DEMO_INPUT["m_dep"], step=0.1, format="%.1f",
            )
            data["talk_time"] = st.number_input(
                "Waktu bicara (jam)", min_value=1, max_value=48,
                value=DEMO_INPUT["talk_time"], step=1,
            )

        with right:
            st.subheader("Layar dan kamera")
            data["px_height"] = st.number_input(
                "Resolusi layar - tinggi (pixel)", min_value=0, max_value=6000,
                value=DEMO_INPUT["px_height"], step=1,
            )
            data["px_width"] = st.number_input(
                "Resolusi layar - lebar (pixel)", min_value=0, max_value=6000,
                value=DEMO_INPUT["px_width"], step=1,
            )
            data["sc_h"] = st.number_input(
                "Tinggi layar fisik (cm)", min_value=0, max_value=40,
                value=DEMO_INPUT["sc_h"], step=1,
            )
            data["sc_w"] = st.number_input(
                "Lebar layar fisik (cm)", min_value=0, max_value=40,
                value=DEMO_INPUT["sc_w"], step=1,
            )
            data["fc"] = st.number_input(
                "Kamera depan (MP)", min_value=0, max_value=200,
                value=DEMO_INPUT["fc"], step=1,
            )
            data["pc"] = st.number_input(
                "Kamera utama (MP)", min_value=0, max_value=200,
                value=DEMO_INPUT["pc"], step=1,
            )
            data["touch_screen"] = int(st.checkbox(
                "Touchscreen", value=bool(DEMO_INPUT["touch_screen"]),
            ))

        st.subheader("Konektivitas dan SIM")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            data["blue"] = int(st.checkbox("Bluetooth", value=bool(DEMO_INPUT["blue"])))
        with c2:
            data["dual_sim"] = int(st.checkbox("Dual SIM", value=bool(DEMO_INPUT["dual_sim"])))
        with c3:
            data["three_g"] = int(st.checkbox("3G", value=bool(DEMO_INPUT["three_g"])))
        with c4:
            data["four_g"] = int(st.checkbox("4G", value=bool(DEMO_INPUT["four_g"])))
        with c5:
            data["wifi"] = int(st.checkbox("Wi-Fi", value=bool(DEMO_INPUT["wifi"])))

        submitted = st.form_submit_button(
            "Prediksi kategori harga", type="primary", use_container_width=True,
        )

    return data, submitted


def main() -> None:
    st.set_page_config(
        page_title="Prediksi Harga HP | P14 Machine Learning",
        page_icon="📱",
        layout="wide",
    )
    st.title("Prediksi Kategori Harga Smartphone")
    st.caption("Mini Project Machine Learning - Tahap 09 | Deployment Streamlit")
    st.info("Isi spesifikasi HP di bawah. Aplikasi menggunakan model terbaik dari Tahap 08.")

    if not MODEL_PATH.is_file():
        st.error("Model Tahap 08 belum ditemukan: models/model_terbaik.joblib")
        st.write("Jalankan Tahap 08 lebih dahulu dari folder P14:")
        st.code("python 08_pemilihan_model_terbaik.py", language="bash")
        st.stop()

    try:
        model = load_trained_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime_ns)
    except Exception as exc:
        st.error("Model gagal dibuka. Periksa file dan versi scikit-learn/joblib.")
        st.exception(exc)
        st.stop()

    # Model Tahap 08 dilatih pada DataFrame berisi 25 fitur urut dan bernama.
    expected = getattr(model, "feature_names_in_", None)
    if expected is not None and list(expected) != ALL_FEATURES:
        st.error("Fitur pada model tidak sesuai format model Tahap 08 (25 fitur).")
        st.stop()

    show_sidebar(read_metrics())
    specifications, submitted = get_specifications_form()
    if not submitted:
        st.caption("Nilai awal hanya contoh untuk mencoba aplikasi, bukan hasil prediksi.")
        return

    try:
        model_features = build_features(specifications)
        label_code = int(model.predict(model_features)[0])
        if label_code not in CLASS_NAMES:
            raise ValueError(f"Model mengembalikan kelas tidak dikenal: {label_code}")
    except Exception as exc:
        st.error("Prediksi gagal. Periksa kecocokan model Tahap 08 dengan input.")
        st.exception(exc)
        return

    st.divider()
    st.subheader("Hasil prediksi")
    st.success("Prediksi berhasil dijalankan.")
    st.metric("Kategori harga yang diprediksi", f"{label_code} - {CLASS_NAMES[label_code]}")
    st.caption("Low sampai Very High adalah kelas relatif, bukan nominal harga rupiah.")

    # SVM Tahap 08 tidak dilatih dengan predict_proba; jangan membuat probabilitas palsu.
    if hasattr(model, "predict_proba"):
        try:
            probabilities = model.predict_proba(model_features)[0]
            classes = list(model.classes_)
            if len(classes) == len(probabilities):
                chart = pd.DataFrame({
                    "Kategori": [f"{int(c)} - {CLASS_NAMES.get(int(c), '?')}" for c in classes],
                    "Probabilitas": probabilities,
                }).set_index("Kategori")
                st.subheader("Estimasi probabilitas per kelas")
                st.bar_chart(chart)
                st.caption("Probabilitas merupakan perkiraan model, bukan kepastian.")
        except (AttributeError, ValueError, TypeError):
            pass

    with st.expander("Lihat 25 fitur yang dikirim ke model"):
        st.dataframe(
            model_features.T.rename(columns={0: "Nilai"}),
            use_container_width=True,
        )

    result = model_features.copy()
    result["predicted_price_range"] = label_code
    result["predicted_class"] = CLASS_NAMES[label_code]
    st.download_button(
        "Unduh hasil prediksi (.csv)",
        data=result.to_csv(index=False).encode("utf-8"),
        file_name="hasil_prediksi_mobile_price.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()
