"""Tahap 09 - Aplikasi prediksi kategori harga HP menggunakan Streamlit.

Menjalankan (dari folder P14):
    python -m pip install streamlit
    python -m streamlit run 09_deployment_streamlit.py

Membaca model hasil Tahap 08 dari P14/models/model_terbaik.joblib.
Tidak membutuhkan train.csv, test.csv, p14_common.py, atau training ulang.
"""
from __future__ import annotations

import json
from html import escape
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
CLASS_LABELS = {
    0: "Harga rendah",
    1: "Harga menengah",
    2: "Harga tinggi",
    3: "Harga sangat tinggi",
}
CLASS_DESCRIPTIONS = {
    0: "Spesifikasi yang dimasukkan paling sesuai dengan kelompok harga terendah dalam dataset latihan.",
    1: "Spesifikasi yang dimasukkan paling sesuai dengan kelompok harga menengah: di atas kelompok rendah dan di bawah kelompok tinggi.",
    2: "Spesifikasi yang dimasukkan paling sesuai dengan kelompok harga tinggi: di atas kelompok menengah dan di bawah kelompok sangat tinggi.",
    3: "Spesifikasi yang dimasukkan paling sesuai dengan kelompok harga tertinggi dari empat kategori dalam dataset latihan.",
}

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


def icon_svg(name: str) -> str:
    paths = {
        "phone": '<rect x="7" y="2" width="10" height="20" rx="2"/><path d="M10 5h4m-3 14h2"/>',
        "performance": '<rect x="5" y="5" width="14" height="14" rx="3"/><path d="M9 1v4m6-4v4M9 19v4m6-4v4M1 9h4m-4 6h4m14-6h4m-4 6h4M9 9h6v6H9z"/>',
        "screen": '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/>',
        "check": '<circle cx="12" cy="12" r="10"/><path d="m7 12 3 3 7-7"/>',
        "details": '<path d="M5 3h14v18H5zM9 7h6m-6 5h6m-6 5h4"/>',
    }
    return (
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" '
        'stroke="currentColor" stroke-width="1.8" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true">{paths[name]}</svg>'
    )


def apply_visual_theme() -> None:
    """Gaya presentasi; perhitungan fitur dan prediksi tetap memakai model Tahap 08."""
    st.html(
        """
        <style>
        .stApp { background: #f6f8fb; color: #172b4d; }
        [data-testid="stHeader"] { height: 0 !important; min-height: 0 !important; background: transparent !important; }
        [data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], [data-testid="stMainMenu"] { display: none; }
        [data-testid="stMain"] { margin-left: 252px; width: calc(100% - 252px); min-width: 0; }
        .block-container { max-width: 1320px; padding: 1.5rem; }
        .block-container > [data-testid="stVerticalBlock"] { gap: 1rem; }
        .block-container [data-testid="stElementContainer"]:has(.lab-sidebar) { position: absolute; width: 0; height: 0; }
        .lab-sidebar {
            position: fixed; inset: 0 auto 0 0; z-index: 10;
            width: 252px; height: 100dvh; box-sizing: border-box;
            padding: 1rem; overflow-y: auto;
            background: #fff; border-right: 1px solid #e6ebf2;
        }
        .sidebar-brand { margin-bottom: 1rem; }
        .brand-name { color: #17365d; font-size: 1rem; font-weight: 800; }
        .brand-caption { color: #5e718a; font-size: .8rem; margin-top: .2rem; }
        .model-status { background: #f3f5f8; border: 1px solid #e8edf4; border-radius: 11px; padding: .75rem; margin: 0 0 1rem; color: #465a77; font-size: .875rem; }
        .model-status strong { display: block; color: #304e70; margin-bottom: .25rem; }
        .status-dot { display: inline-block; width: 7px; height: 7px; background: #1eb980; border-radius: 50%; margin-right: .4rem; }
        .sidebar-metrics { border: 1px solid #e2e8f0; border-radius: 11px; padding: .25rem .75rem; margin: 0 0 1rem; }
        .sidebar-metrics div { display: flex; justify-content: space-between; align-items: center; gap: .5rem; padding: .65rem 0; }
        .sidebar-metrics span { font-size: .875rem; color: #60728b; }
        .sidebar-metrics strong { font-size: 1rem; color: #0671e8; white-space: nowrap; }
        .sidebar-metrics .accuracy { color: #128357; }
        .sidebar-nav { display: flex; align-items: center; gap: .6rem; padding: .75rem; margin: 0 0 1rem; border-radius: 10px; background: #e8f2ff; color: #075dbd; font-size: .9rem; font-weight: 700; }
        .lab-sidebar .sidebar-source, .sidebar-identity { color: #60728b; font-size: .8125rem; font-weight: 400; line-height: 1.6; overflow-wrap: anywhere; }
        .sidebar-source { margin: 0; }
        .sidebar-source strong, .sidebar-identity strong { font-weight: 600; }
        .sidebar-identity { margin-top: .75rem; padding-top: .75rem; border-top: 1px solid #e6ebf2; }
        .sidebar-identity p { margin: 0; }
        .sidebar-identity p + p { margin-top: .125rem; }
        .page-heading { padding: 0; }
        [data-testid="stMarkdownContainer"]:has(> .page-heading, > .form-section-title, > .feature-panel, > .feature-page-status) { margin-bottom: 0; }
        .page-heading h1 { color: #172b4d; font-size: clamp(1.4rem, 1.9vw, 1.75rem); line-height: 1.25; padding: 0; margin: 0; }
        [data-testid="stForm"] {
            background: #fff; border: 1px solid #dce4ee; border-radius: 15px;
            padding: 1.25rem; box-shadow: 0 3px 12px rgba(23,43,77,.025);
        }
        [data-testid="stForm"] [data-testid="stVerticalBlock"] { gap: .75rem; }
        .st-key-predict_specs { order: 1; }
        [data-testid="stForm"] [data-testid="stHorizontalBlock"]:has([data-testid="stNumberInput"]) { column-gap: 1.5rem; }
        .form-section-title { display: flex; align-items: center; gap: .5rem; color: #304e72; font-size: .875rem; font-weight: 800; letter-spacing: .025em; margin: 0 0 .25rem; }
        .form-section-title svg { color: #0671e8; flex-shrink: 0; }
        [data-testid="stWidgetLabel"] { min-height: 1.15rem; margin-bottom: .375rem; }
        [data-testid="stWidgetLabel"] p { color: #5d7089; font-size: .875rem; line-height: 1.25; }
        [data-testid="stNumberInputContainer"] { height: 2.5rem !important; min-height: 2.5rem !important; border-radius: 8px; background: #f3f5f8; }
        [data-testid="stNumberInputField"] { height: 100%; min-height: 0; padding: .45rem .75rem !important; font-size: 1rem; line-height: 1.2; color: #213b5c; text-align: left !important; box-sizing: border-box; }
        [data-testid="stNumberInputStepDown"], [data-testid="stNumberInputStepUp"] { width: 1.7rem !important; min-width: 1.7rem !important; color: #45627f; }
        [data-testid="stCheckbox"] label { padding: .25rem .3rem; border-radius: 7px; background: #f1f5fa; }
        [data-testid="stCheckbox"] label p { color: #304e70; font-size: .875rem; }
        [data-testid="stFormSubmitButton"] button {
            min-height: 40px; border: 1px solid #0671e8; border-radius: 9px;
            background: #0671e8; color: #fff; margin-top: .2rem;
        }
        [data-testid="stFormSubmitButton"] button:hover { background: #0564ce; border-color: #0564ce; color: #fff; }
        [data-testid="stFormSubmitButton"] button p { font-size: 1rem; font-weight: 700; }
        .st-key-reset_specs [data-testid="stFormSubmitButton"] button { min-height: 32px; margin-top: .5rem; border: 1px solid #dce4ee; border-radius: 8px; background: #fff; color: #45607e; }
        .st-key-reset_specs [data-testid="stFormSubmitButton"] button:hover { border-color: #0671e8; background: #edf5ff; color: #0671e8; }
        .st-key-reset_specs button p { font-size: .875rem; font-weight: 400; }
        .result-card {
            background: #fff; border: 1px solid #dce4ee; border-top: 4px solid #0671e8;
            border-radius: 15px; padding: 1.25rem; margin: 0 0 1rem;
            box-shadow: 0 3px 12px rgba(23,43,77,.025);
        }
        .result-position { display: inline-block; background: #e6f1ff; color: #075dbd; border-radius: 7px; padding: .3rem .6rem; font-size: .875rem; font-weight: 700; }
        .result-title { display: flex; align-items: center; gap: .75rem; margin: 0 0 .75rem; }
        .empty-result .result-title { margin-top: .75rem; }
        .result-title svg { color: #0671e8; flex-shrink: 0; width: 24px; height: 24px; }
        .result-card h2 { color: #17365d; font-size: 1.35rem; line-height: 1.3; padding: 0; margin: 0; }
        .result-card p { color: #5e718a; font-size: 1rem; line-height: 1.5; margin: 0 0 1rem; }
        .category-track { display: grid; grid-template-columns: 1fr 1fr; gap: .75rem; }
        .category-step { min-height: 44px; display: flex; align-items: center; padding: .65rem; border: 1px solid transparent; border-radius: 9px; color: #526780; background: #f2f4f7; font-size: .875rem; line-height: 1.3; }
        .category-step.is-active { color: #fff; background: #0671e8; border-color: #0671e8; font-weight: 800; }
        .empty-result { border-top-color: #b3d5ff; }
        .spec-summary { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: .75rem; margin: 0 0 1rem; }
        .spec-summary div { background: #fff; border: 1px solid #dce4ee; border-radius: 10px; padding: .75rem; }
        .spec-summary span { display: block; color: #60728b; font-size: .875rem; font-weight: 700; letter-spacing: .04em; }
        .spec-summary strong { display: block; color: #213b5c; font-size: 1rem; margin-top: .35rem; }
        .st-key-feature_details { background: #fff; border: 1px solid #dce4ee; border-radius: 13px; padding: 1.25rem; gap: .75rem; }
        .detail-heading { display: flex; justify-content: space-between; align-items: center; color: #304e70; font-size: .875rem; font-weight: 800; }
        .detail-heading span { display: flex; align-items: center; gap: .45rem; }
        .detail-heading svg { color: #0671e8; }
        .detail-count { color: #075dbd; font-size: .8rem; background: #eaf3ff; border-radius: 6px; padding: .2rem .4rem; }
        .detail-meta { margin-top: .5rem; color: #5e718a; font-size: .875rem; line-height: 1.6; }
        .feature-table-wrapper { margin-top: .5rem; border: 1px solid #e2e8f1; border-radius: 8px; }
        .feature-table-wrapper .feature-table { width: 100%; margin: 0; border-collapse: separate; border-spacing: 0; table-layout: fixed; }
        .feature-table th, .feature-table td { text-align: left !important; padding: .4rem .65rem; font-size: .875rem; line-height: 1.35; border-bottom: 1px solid #e6ecf4; font-variant-numeric: tabular-nums; overflow-wrap: anywhere; }
        .feature-table th { background: #f1f5fa; color: #61738d; font-weight: 500; }
        .feature-table th:first-child { width: 65%; }
        .feature-table td { color: #354f70; background: #fff; }
        .feature-table tr:nth-child(even) td { background: #f8fafe; }
        .feature-table tr:last-child td { border-bottom: 0; }
        .st-key-feature_details [data-testid="stHorizontalBlock"] { flex-wrap: nowrap; gap: .5rem; align-items: center; }
        .st-key-feature_details [data-testid="stColumn"] { min-width: 0 !important; }
        .st-key-feature_details button { min-height: 30px; padding: .2rem .4rem; border-radius: 7px; }
        .st-key-feature_details button p { font-size: .8125rem; }
        .feature-page-status { text-align: center; color: #60728b; font-size: .8125rem; line-height: 1.4; }
        [data-testid="stDownloadButton"] { margin-top: 0; }
        [data-testid="stDownloadButton"] button { background: #fff; color: #0671e8; border: 1px solid #dce4ee; border-radius: 10px; min-height: 44px; }
        [data-testid="stDownloadButton"] button:hover { background: #edf5ff; border-color: #0671e8; color: #0564ce; }
        @media (min-width: 1151px) {
            [data-testid="stFormSubmitButton"] button { min-height: 34px; margin-top: 0; }
            .result-card h2 { font-size: clamp(1.125rem, 1.6vw, 1.35rem); }
            .result-card p { font-size: .875rem; }
            .category-step { min-height: 34px; }
            [data-testid="stDownloadButton"] button { min-height: 34px; }
        }
        @media (max-width: 1150px) {
            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stForm"]) { flex-direction: column; align-items: stretch; gap: 1.25rem; }
            [data-testid="stHorizontalBlock"]:has(> [data-testid="stColumn"] [data-testid="stForm"]) > [data-testid="stColumn"] { width: 100% !important; flex: 1 1 100% !important; min-width: 0 !important; }
        }
        @media (max-width: 900px) {
            [data-testid="stMain"] { margin-left: 0; width: 100%; }
            .block-container [data-testid="stElementContainer"]:has(.lab-sidebar) { position: static; width: 100%; height: auto; }
            .lab-sidebar { position: static; width: 100%; height: auto; padding: 1rem; border: 1px solid #e6ebf2; border-radius: 13px; }
        }
        @media (max-width: 760px) {
            .block-container { padding: 1rem; }
            .lab-sidebar { margin-bottom: 1rem; }
            .spec-summary div { padding: .65rem .5rem; }
        }
        </style>
        """,
    )


def show_prediction_result(label_code: int, specifications: dict) -> None:
    """Terjemahkan kode kelas menjadi hasil yang dapat dipahami pengguna."""
    title = "Kategori " + CLASS_LABELS[label_code].title()
    categories = "".join(
        f'<div class="category-step{" is-active" if code == label_code else ""}"'
        f' aria-current="{"step" if code == label_code else "false"}">'
        f'{escape(label)}</div>'
        for code, label in CLASS_LABELS.items()
    )
    st.markdown(
        f"""<section class="result-card">
        <div class="result-title">{icon_svg('check')}<h2>{escape(title)}</h2></div>
        <p>{escape(CLASS_DESCRIPTIONS[label_code])}</p>
        <div class="category-track" aria-label="Empat kelompok harga">{categories}</div>
        </section>""",
        unsafe_allow_html=True,
    )
    show_specification_summary(specifications)


def show_specification_summary(specifications: dict) -> None:
    st.markdown(
        f"""<div class="spec-summary" aria-label="Ringkasan spesifikasi masukan">
        <div><span>RAM</span><strong>{specifications['ram']:g} MB</strong></div>
        <div><span>MEMORI</span><strong>{specifications['int_memory']:g} GB</strong></div>
        <div><span>BATERAI</span><strong>{specifications['battery_power']:g} mAh</strong></div>
        </div>""",
        unsafe_allow_html=True,
    )


def set_feature_page(page: int) -> None:
    st.session_state["feature_page"] = page


def show_feature_details(model_features: pd.DataFrame, label_code: int | None = None) -> None:
    """Tampilkan seluruh fitur lewat pagination tanpa scroll internal."""
    page_size = 5
    total = len(model_features.columns)
    page_count = max(1, (total + page_size - 1) // page_size)
    page = min(max(0, st.session_state.get("feature_page", 0)), page_count - 1)
    start = page * page_size
    if label_code is None:
        metadata = '<div class="detail-meta">Pratinjau fitur dari spesifikasi masukan.</div>'
    else:
        metadata = ""
    rows = []
    for name, value in model_features.iloc[0, start:start + page_size].items():
        formatted = f"{float(value):.4f}".rstrip("0").rstrip(".")
        rows.append(f'<tr><td>{escape(str(name))}</td><td>{escape(formatted)}</td></tr>')
    with st.container(key="feature_details"):
        st.markdown(
            f'''<section class="feature-panel">
            <div class="detail-heading"><span>{icon_svg("details")}DETAIL NILAI & FITUR</span>
            <span class="detail-count">{total} fitur</span></div>
            {metadata}<div class="feature-table-wrapper">
            <table class="feature-table"><thead><tr><th scope="col">Fitur</th><th scope="col">Nilai</th></tr></thead>
            <tbody>{''.join(rows)}</tbody></table></div></section>''',
            unsafe_allow_html=True,
        )
        previous, status, following = st.columns([1.2, 1, 1.2], gap="small", vertical_alignment="center")
        with previous:
            st.button("Sebelumnya", key="feature_previous", disabled=page == 0,
                      on_click=set_feature_page, args=(page - 1,), width="stretch")
        with status:
            st.markdown(f'<div class="feature-page-status" role="status" aria-label="Halaman {page + 1} dari {page_count}, fitur {start + 1} sampai {min(start + page_size, total)} dari {total}">{page + 1} / {page_count}</div>', unsafe_allow_html=True)
        with following:
            st.button("Berikutnya", key="feature_next", disabled=page == page_count - 1,
                      on_click=set_feature_page, args=(page + 1,), width="stretch")


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
    """Panel statis: tetap di kiri desktop, masuk alur halaman pada layar kecil."""
    algorithm = escape(str(metrics.get("model_terpilih", "Model Tahap 08")))
    metric_rows = []
    if isinstance(metrics.get("accuracy_test"), (float, int)):
        metric_rows.append(f'<div><span>Akurasi uji</span><strong class="accuracy">{metrics["accuracy_test"]:.2%}</strong></div>')
    if isinstance(metrics.get("f1_macro_test"), (float, int)):
        metric_rows.append(f'<div><span>F1-macro</span><strong>{metrics["f1_macro_test"]:.4f}</strong></div>')
    metrics_html = '<div class="sidebar-metrics">' + ''.join(metric_rows) + '</div>' if metric_rows else ''
    st.markdown(
        f'''<aside class="lab-sidebar" aria-label="Informasi model Smartphone Lab">
        <div class="sidebar-brand"><div class="brand-name">Smartphone Lab</div>
        <div class="brand-caption">Klasifikasi kelompok harga HP</div></div>
        <div class="model-status"><strong><span class="status-dot"></span>Model siap digunakan</strong>
        Algoritma: <b>{algorithm}</b></div>
        {metrics_html}
        <nav aria-label="Halaman aktif"><div class="sidebar-nav" aria-current="page">{icon_svg("details")}Prediksi Harga</div></nav>
        <p class="sidebar-source"><strong>Sumber data:</strong> Kaggle Mobile Price Classification</p>
        <div class="sidebar-identity" aria-label="Identitas mahasiswa">
        <p><strong>Nama:</strong> Rezaldy Abidin</p>
        <p><strong>NPM:</strong> 202343500253</p>
        <p><strong>Kelas:</strong> X7C</p>
        </div>
        </aside>''',
        unsafe_allow_html=True,
    )


def reset_specifications() -> None:
    """Identitas widget baru mengembalikan seluruh formulir ke nilai contoh."""
    st.session_state["mobile_form_version"] = st.session_state.get("mobile_form_version", 0) + 1
    st.session_state.pop("mobile_price_result", None)
    st.session_state.pop("feature_page", None)


def get_specifications_form() -> tuple[dict[str, int | float], bool]:
    """Form 20 fitur asli; fitur turunan tidak perlu diisi pengguna."""
    data: dict[str, int | float] = {}
    key_prefix = f"mobile_specs_{st.session_state.get('mobile_form_version', 0)}"

    with st.form("mobile_price_form"):
        # Submit utama tetap didaftarkan pertama untuk Enter; CSS menaruhnya di bawah.
        submitted = st.form_submit_button(
            "Lihat kategori harga", key="predict_specs", type="primary", width="stretch",
        )
        left, right = st.columns(2, gap="medium")
        with left:
            st.markdown(f'<div class="form-section-title" role="heading" aria-level="2">{icon_svg("performance")}PERFORMA & BATERAI</div>', unsafe_allow_html=True)
            data["ram"] = st.number_input(
                "RAM (MB)", min_value=1, max_value=16384, value=DEMO_INPUT["ram"], step=128,
                help="Masukkan dalam MB. Contoh: RAM 2 GB ditulis 2048 MB.",
                key=f"{key_prefix}_ram",
            )
            data["int_memory"] = st.number_input(
                "Memori internal (GB)", min_value=1, max_value=1024,
                value=DEMO_INPUT["int_memory"], step=1,
                key=f"{key_prefix}_int_memory",
            )
            data["battery_power"] = st.number_input(
                "Kapasitas baterai (mAh)", min_value=1, max_value=15000,
                value=DEMO_INPUT["battery_power"], step=50,
                key=f"{key_prefix}_battery_power",
            )
            data["clock_speed"] = st.number_input(
                "Clock speed (GHz)", min_value=0.1, max_value=6.0,
                value=DEMO_INPUT["clock_speed"], step=0.1, format="%.1f",
                key=f"{key_prefix}_clock_speed",
            )
            data["n_cores"] = st.number_input(
                "Jumlah core prosesor", min_value=1, max_value=32,
                value=DEMO_INPUT["n_cores"], step=1,
                key=f"{key_prefix}_n_cores",
            )
            data["mobile_wt"] = st.number_input(
                "Berat HP (gram)", min_value=1, max_value=500,
                value=DEMO_INPUT["mobile_wt"], step=1,
                key=f"{key_prefix}_mobile_wt",
            )
            data["m_dep"] = st.number_input(
                "Ketebalan HP (cm)", min_value=0.1, max_value=3.0,
                value=DEMO_INPUT["m_dep"], step=0.1, format="%.1f",
                key=f"{key_prefix}_m_dep",
            )
            data["talk_time"] = st.number_input(
                "Waktu bicara (jam)", min_value=1, max_value=48,
                value=DEMO_INPUT["talk_time"], step=1,
                key=f"{key_prefix}_talk_time",
            )

        with right:
            st.markdown(f'<div class="form-section-title" role="heading" aria-level="2">{icon_svg("screen")}LAYAR & KAMERA</div>', unsafe_allow_html=True)
            data["px_height"] = st.number_input(
                "Resolusi layar - tinggi (pixel)", min_value=0, max_value=6000,
                value=DEMO_INPUT["px_height"], step=1,
                key=f"{key_prefix}_px_height",
            )
            data["px_width"] = st.number_input(
                "Resolusi layar - lebar (pixel)", min_value=0, max_value=6000,
                value=DEMO_INPUT["px_width"], step=1,
                key=f"{key_prefix}_px_width",
            )
            data["sc_h"] = st.number_input(
                "Tinggi layar fisik (cm)", min_value=0, max_value=40,
                value=DEMO_INPUT["sc_h"], step=1,
                key=f"{key_prefix}_sc_h",
            )
            data["sc_w"] = st.number_input(
                "Lebar layar fisik (cm)", min_value=0, max_value=40,
                value=DEMO_INPUT["sc_w"], step=1,
                key=f"{key_prefix}_sc_w",
            )
            data["fc"] = st.number_input(
                "Kamera depan (MP)", min_value=0, max_value=200,
                value=DEMO_INPUT["fc"], step=1,
                key=f"{key_prefix}_fc",
            )
            data["pc"] = st.number_input(
                "Kamera utama (MP)", min_value=0, max_value=200,
                value=DEMO_INPUT["pc"], step=1,
                key=f"{key_prefix}_pc",
            )
            data["touch_screen"] = int(st.checkbox(
                "Touchscreen", value=bool(DEMO_INPUT["touch_screen"]),
                key=f"{key_prefix}_touch_screen",
            ))
            st.form_submit_button(
                "Reset Nilai", key="reset_specs", type="secondary",
                on_click=reset_specifications, width="content",
            )

        st.markdown('<div class="form-section-title" role="heading" aria-level="2">KONEKTIVITAS & SIM</div>', unsafe_allow_html=True)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            data["blue"] = int(st.checkbox("Bluetooth", value=bool(DEMO_INPUT["blue"]), key=f"{key_prefix}_blue"))
        with c2:
            data["dual_sim"] = int(st.checkbox("Dual SIM", value=bool(DEMO_INPUT["dual_sim"]), key=f"{key_prefix}_dual_sim"))
        with c3:
            data["three_g"] = int(st.checkbox("3G", value=bool(DEMO_INPUT["three_g"]), key=f"{key_prefix}_three_g"))
        with c4:
            data["four_g"] = int(st.checkbox("4G", value=bool(DEMO_INPUT["four_g"]), key=f"{key_prefix}_four_g"))
        with c5:
            data["wifi"] = int(st.checkbox("Wi-Fi", value=bool(DEMO_INPUT["wifi"]), key=f"{key_prefix}_wifi"))

    return data, submitted


def main() -> None:
    st.set_page_config(
        page_title="Prediksi Harga HP | P14 Machine Learning",
        page_icon="📱",
        layout="wide",
    )
    apply_visual_theme()

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
    st.markdown(
        '<div class="page-heading"><h1>Prediksi Kategori Harga HP</h1></div>',
        unsafe_allow_html=True,
    )

    input_column, result_column = st.columns([1.4, 1], gap="medium")
    with input_column:
        specifications, submitted = get_specifications_form()

    model_modified = MODEL_PATH.stat().st_mtime_ns
    if submitted:
        st.session_state["feature_page"] = 0
        try:
            model_features = build_features(specifications)
            label_code = int(model.predict(model_features)[0])
            if label_code not in CLASS_LABELS:
                raise ValueError(f"Model mengembalikan kelas tidak dikenal: {label_code}")
            st.session_state["mobile_price_result"] = {
                "specifications": specifications,
                "label_code": label_code,
                "model_modified": model_modified,
            }
        except Exception as exc:
            st.session_state.pop("mobile_price_result", None)
            with result_column:
                st.error("Prediksi gagal. Periksa kecocokan model Tahap 08 dengan input.")
                st.exception(exc)
            return

    with result_column:
        prediction = st.session_state.get("mobile_price_result")
        if prediction and prediction["model_modified"] != model_modified:
            st.session_state.pop("mobile_price_result", None)
            prediction = None
        if prediction is None:
            st.markdown(
                """<section class="result-card empty-result">
                <span class="result-position">Belum diprediksi</span>
                <div class="result-title"><h2>Kategori harga HP</h2></div>
                <p>Sesuaikan spesifikasi, lalu klik <strong>Lihat kategori harga</strong>.
                Kategori yang terpilih akan ditandai biru.</p>
                <div class="category-track" aria-label="Empat kelompok harga">
                <div class="category-step">Harga rendah</div><div class="category-step">Harga menengah</div>
                <div class="category-step">Harga tinggi</div><div class="category-step">Harga sangat tinggi</div>
                </div>
                </section>""",
                unsafe_allow_html=True,
            )
            show_specification_summary(specifications)
            show_feature_details(build_features(specifications))
            return

        label_code = prediction["label_code"]
        specifications = prediction["specifications"]
        model_features = build_features(specifications)
        show_prediction_result(label_code, specifications)

        # Tampilkan probabilitas hanya jika tersedia dari model yang dilatih.
        if hasattr(model, "predict_proba"):
            try:
                probabilities = model.predict_proba(model_features)[0]
                classes = list(model.classes_)
                if len(classes) == len(probabilities):
                    chart = pd.DataFrame({
                        "Kategori": [CLASS_LABELS[int(c)] for c in classes],
                        "Probabilitas": probabilities,
                    }).set_index("Kategori")
                    st.subheader("Estimasi probabilitas per kategori")
                    st.bar_chart(chart)
                    st.caption("Probabilitas merupakan perkiraan model, bukan kepastian.")
            except (AttributeError, ValueError, TypeError):
                pass

        show_feature_details(model_features, label_code)

        result = model_features.copy()
        result["predicted_price_range"] = label_code
        result["predicted_class"] = CLASS_NAMES[label_code]
        result["kategori_harga"] = CLASS_LABELS[label_code]
        result["arti_kategori"] = CLASS_DESCRIPTIONS[label_code]
        st.download_button(
            "Unduh hasil prediksi (.csv)",
            data=result.to_csv(index=False).encode("utf-8"),
            file_name="hasil_prediksi_mobile_price.csv",
            mime="text/csv",
            width="stretch",
        )


if __name__ == "__main__":
    main()
