import pandas as pd
import streamlit as st

from services import models, roles
from ui import charts, fmt, forms, guard, latih, layout

user = guard.require_role(roles.ADMIN)

P = "f_rf"
BAWAAN = models.default_params(models.RF)
PERSIST = {"persist_state": "session"}

STRUKTUR_FOLDER = """\
batuceper/                  <- nama folder = nama kecamatan yang dipilih
├── 2019-07-04/             <- satu subfolder per tanggal (YYYY-MM-DD)
│   ├── ..._B02_10m.tif
│   ├── ..._B03_10m.tif     (B02, B03, B04, B05, B06, B07, B08, B11, B12)
│   ├── ...
│   └── ..._SCL_10m.tif
├── 2019-09-12/
│   └── ...
└── ..."""


def _label(nama: str, param: str) -> str:
    return latih.labeled(nama, models.RF, param)


def _samakan_rasio(sumber: str, tujuan: str) -> None:
    """Train ratio + test ratio selalu 1: mengubah satu menyesuaikan yang lain"""
    st.session_state[tujuan] = round(1 - st.session_state[sumber], 2)


def _laporan_folder(laporan: dict, kecamatan_id: str | None) -> None:
    if kecamatan_id and laporan["akar"] and not laporan["akar_cocok"]:
        st.warning(f"Nama folder “{laporan['akar']}” tidak sama dengan kecamatan yang dipilih.",
                   icon=":material/folder_off:")
    if not laporan["baris"]:
        st.caption("Tidak ada subfolder tanggal yang dikenali.")
        return
    st.dataframe(
        pd.DataFrame([
            {"Tanggal": b["tanggal"], "Band": b["band"],
             "Status": "✅ valid" if b["status"] == "valid" else "❌ gagal"}
            for b in laporan["baris"]
        ]),
        hide_index=True, width="stretch", height=min(36 * len(laporan["baris"]) + 40, 260),
    )
    gagal = len(laporan["baris"]) - laporan["jumlah_valid"]
    st.caption(f"Folder “{laporan['akar']}” · {laporan['jumlah_valid']} tanggal valid · {gagal} gagal · "
               f"{len(laporan['diabaikan'])} berkas diabaikan. Tanggal yang gagal tidak ikut dilatih.")


def _identitas(errors: dict, nonaktif: bool) -> dict:
    c_nama, c_wil, c_kec = st.columns([1.2, 1, 1])
    nama = c_nama.text_input("Model", key=f"{P}_nama", placeholder="Nama model", disabled=nonaktif, **PERSIST)
    forms.field_error(errors, "nama", c_nama)
    _, kecamatan_id = forms.region_select(c_wil, c_kec, P, disabled=nonaktif, errors=errors)

    berkas = st.file_uploader(
        "Folder citra kecamatan", type=["tif", "tiff"], accept_multiple_files="directory",
        key=f"{P}_folder", disabled=nonaktif,
        help="Pilih satu folder kecamatan. Isinya subfolder per tanggal, masing-masing 9 band + SCL.",
    ) or []
    forms.field_error(errors, "folder")
    with st.expander("Struktur folder yang dibaca sistem", icon=":material/account_tree:"):
        st.code(STRUKTUR_FOLDER, language=None)
        st.caption("Band dikenali dari nama berkas (…_B02_…, …_SCL_…). Semua berkas harus berada pada grid yang "
                   "sama (CRS, resolusi 10 m, ukuran). Tanggal harus di dalam rentang arsip 2016–2025.")

    laporan = None
    if berkas:
        laporan = models.scan_folder([b.name for b in berkas], kecamatan_id)
        _laporan_folder(laporan, kecamatan_id)
    return {"nama": nama, "kecamatan_id": kecamatan_id, "berkas": berkas,
            "jumlah_tanggal": laporan["jumlah_valid"] if laporan else 0}


def _hyperparameter(errors: dict, nonaktif: bool, jumlah_tanggal: int) -> dict:
    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    n_trees = c1.number_input(_label("Jumlah decision tree", "n_trees"), min_value=1, max_value=500, step=5,
                              value=BAWAAN["n_trees"], key=f"{P}_trees", disabled=nonaktif, **PERSIST)
    max_depth = c2.number_input(_label("Max depth", "max_depth"), min_value=1, max_value=100, step=1,
                                value=BAWAAN["max_depth"], key=f"{P}_maxdepth", disabled=nonaktif, **PERSIST)
    min_split = c3.number_input(_label("Min samples split", "min_samples_split"), min_value=2, max_value=1000,
                                step=1, value=BAWAAN["min_samples_split"], key=f"{P}_minsplit", disabled=nonaktif,
                                **PERSIST)
    n_features = c4.number_input(_label("Jumlah fitur per split", "n_features"), min_value=1, max_value=7, step=1,
                                 value=BAWAAN["n_features"], key=f"{P}_mtry", disabled=nonaktif, **PERSIST)

    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    max_thr = c1.number_input(_label("Max thresholds", "max_thresholds"), min_value=2, max_value=1000, step=5,
                              value=BAWAAN["max_thresholds"], key=f"{P}_thr", disabled=nonaktif, **PERSIST)
    train = c2.number_input(_label("Train ratio", "train_ratio"), min_value=0.05, max_value=0.95, step=0.05,
                            format="%.2f", value=BAWAAN["train_ratio"], key=f"{P}_train", disabled=nonaktif,
                            on_change=_samakan_rasio, args=(f"{P}_train", f"{P}_test"), **PERSIST)
    test = c3.number_input(_label("Test ratio", "test_ratio"), min_value=0.05, max_value=0.95, step=0.05,
                           format="%.2f", value=BAWAAN["test_ratio"], key=f"{P}_test", disabled=nonaktif,
                           on_change=_samakan_rasio, args=(f"{P}_test", f"{P}_train"), **PERSIST)
    semua = st.session_state.get(f"{P}_semua", False)
    sampel = c4.number_input(_label("Sampel per tanggal", "sampel_per_tanggal"), min_value=10,
                             max_value=100_000, step=100, value=BAWAAN["sampel_per_tanggal"],
                             key=f"{P}_npd", disabled=nonaktif or semua, **PERSIST)
    forms.row_error(errors, "rasio", kolom=2)

    c_ket, c_semua = st.columns([3, 1])
    semua = c_semua.toggle("Pakai seluruh piksel", key=f"{P}_semua", disabled=nonaktif, **PERSIST)
    keterangan = ("*Memakai seluruh piksel membuat pelatihan jauh lebih lama, karena satu kecamatan bisa "
                  "berisi jutaan piksel. Prosesnya bisa sampai berjam-jam.")
    if jumlah_tanggal and not semua:
        keterangan += (f" Perkiraan sampel: {jumlah_tanggal} tanggal × {fmt.angka(sampel, 0)} piksel = "
                       f"{fmt.angka(jumlah_tanggal * sampel, 0)} piksel.")
    c_ket.caption(keterangan)

    return {
        "n_trees": int(n_trees),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_split),
        "n_features": int(n_features),
        "max_thresholds": int(max_thr),
        "train_ratio": round(float(train), 2),
        "test_ratio": round(float(test), 2),
        "sampel_per_tanggal": None if semua else int(sampel),
    }


def _evaluasi(calon: dict) -> None:
    ev = calon["evaluasi"]
    m = ev["metrik"]
    latih.evaluation_caption(calon, "sampel uji")
    c_cm, c_kanan = st.columns([1, 1.4], gap="large")
    with c_cm:
        layout.section("Confusion matrix")
        charts.confusion_matrix(ev["confusion"], ev["label_kelas"])
    with c_kanan:
        layout.metric_cards([
            ("Accuracy", fmt.skor(m["accuracy"])), ("Precision", fmt.skor(m["precision"])),
            ("Recall", fmt.skor(m["recall"])), ("F1-score", fmt.skor(m["f1"])),
        ])
        layout.section("Data latih dan uji")
        layout.facts([
            ("Tanggal citra", f"{ev['jumlah_tanggal']} tanggal"),
            ("Sampel latih", f"{fmt.angka(ev['n_latih'], 0)} piksel · {ev['blok_latih']} blok"),
            ("Sampel uji", f"{fmt.angka(ev['n_uji'], 0)} piksel · {ev['blok_uji']} blok"),
            ("Label terbangun (acuan NDBI)", fmt.persen(ev["persen_terbangun_acuan"])),
            ("Waktu latih", f"{fmt.angka(ev['waktu_latih_detik'], 1)} detik"),
        ])
        st.caption("Metrik mengukur kesesuaian dengan label acuan berbasis NDBI pada blok uji di kecamatan yang "
                   "sama, bukan kebenaran lapangan. Angka ini lebih optimis daripada uji pada kecamatan lain dan "
                   "tidak sebanding langsung dengan hasil notebook.")


layout.page_header("Klasifikasi")
with layout.card("klasifikasi", group="klasifikasi", role=user["role"]):
    berjalan, errors = latih.status(P)
    latih.general_error(errors)

    identitas = _identitas(errors, berjalan)
    params = _hyperparameter(errors, berjalan, identitas.pop("jumlah_tanggal"))

    latih.start_button(P, berjalan)
    latih.run_if_started(P, models.RF, {**identitas, "params": params})

    st.divider()
    layout.section("Evaluasi model")
    calon = latih.draft(P)
    if calon is None:
        layout.empty_state("Evaluasi muncul di sini setelah model selesai dilatih.", "query_stats")
    else:
        _evaluasi(calon)
        latih.save_discard(P, calon)
