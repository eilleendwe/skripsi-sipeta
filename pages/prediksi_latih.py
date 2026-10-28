import pandas as pd
import streamlit as st

from services import arsip, models, regions, roles
from ui import arsip as panel_arsip
from ui import charts, fmt, forms, guard, latih, layout

user = guard.require_role(roles.ADMIN)

P = "f_cl"
BAWAAN = models.default_params(models.CONVLSTM)
PERSIST = {"persist_state": "session"}


def _label(nama: str, param: str) -> str:
    return latih.labeled(nama, models.CONVLSTM, param)


def _tahun(daftar: list[int]) -> str:
    return f"{daftar[0]}–{daftar[-1]}" if len(daftar) > 1 else str(daftar[0])


def _identitas(errors: dict, nonaktif: bool) -> dict:
    c_nama, c_sumber = st.columns([1, 3])
    nama = c_nama.text_input("Model", key=f"{P}_nama", placeholder="Nama model", disabled=nonaktif, **PERSIST)
    forms.field_error(errors, "nama", c_nama)
    sumber = c_sumber.radio(
        "Sumber data latih", list(models.SUMBER_DATA), format_func=models.SUMBER_DATA.get, horizontal=True,
        key=f"{P}_sumber", disabled=nonaktif, **PERSIST,
        help="Arsip di server: pilih kecamatan, peta klasifikasi diambil dari arsip. Unggah folder: seperti "
             "Latih RF, folder peta klasifikasi diunggah dan hanya dipakai untuk pelatihan ini.",
    )
    data = _data_unggah(errors, nonaktif) if sumber == models.SUMBER_UNGGAH else _data_arsip(errors, nonaktif)
    return {"nama": nama, "sumber": sumber, **data}


def _data_arsip(errors: dict, nonaktif: bool) -> dict:
    panel_arsip.panel(nonaktif)
    c_wil, c_kec = st.columns([1, 2])
    wilayah_id, kecamatan_ids = forms.region_select(c_wil, c_kec, P, multi=True, semua_wilayah=True,
                                                    disabled=nonaktif, errors=errors)
    if kecamatan_ids:
        lengkap = arsip.lengkap(kecamatan_ids)
        kurang = [regions.nama_kecamatan(k) for k in kecamatan_ids if k not in lengkap]
        teks = (f"Arsip klasifikasi: {len(lengkap)} dari {len(kecamatan_ids)} kecamatan punya peta lengkap "
                f"{arsip.TAHUN_DATA[0]}–{arsip.TAHUN_DATA[-1]}.")
        if kurang:
            teks += " Belum lengkap, akan dilewati: " + ", ".join(kurang) + "."
        st.caption(teks)
    return {"wilayah_id": wilayah_id, "kecamatan_ids": list(kecamatan_ids)}


def _data_unggah(errors: dict, nonaktif: bool) -> dict:
    berkas = st.file_uploader(
        "Folder peta klasifikasi (RF_LABEL)", type=["tif", "tiff"], accept_multiple_files="directory",
        key=f"{P}_folder", disabled=nonaktif,
        help="Boleh folder kecamatan_masked_rf_classified, satu folder wilayah, atau satu folder kecamatan.",
    ) or []
    forms.field_error(errors, "folder")
    with st.expander("Struktur folder yang dibaca sistem", icon=":material/account_tree:"):
        st.code(panel_arsip.STRUKTUR, language=None)
    kecamatan_ids = []
    if berkas:
        laporan = panel_arsip.laporan_folder(berkas)
        kecamatan_ids = [b["kecamatan_id"] for b in laporan["baris"] if b["lengkap"]]
    st.caption(f"Seperti Latih RF, folder ini hanya dipakai untuk pelatihan ini lalu dihapus dari server. Yang dipakai "
               f"hanya kecamatan dengan peta lengkap {arsip.TAHUN_DATA[0]}–{arsip.TAHUN_DATA[-1]}. Prediksi Citra "
               "tetap membaca arsip di server, jadi kecamatan yang ingin diprediksi juga harus ada di arsip.")
    return {"wilayah_id": None, "kecamatan_ids": kecamatan_ids, "berkas": berkas}


def _hyperparameter(errors: dict, nonaktif: bool) -> dict:
    # Validation ratio = sisa dari train + test, jadi jumlah ketiganya selalu 1.
    k_val = f"{P}_val"
    train_kini = st.session_state.get(f"{P}_train", BAWAAN["train_ratio"])
    test_kini = st.session_state.get(f"{P}_test", BAWAAN["test_ratio"])
    st.session_state[k_val] = max(0.0, round(1 - train_kini - test_kini, 2))

    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    seq = c1.number_input(_label("Sequence length", "sequence_length"), min_value=2, max_value=4, step=1,
                          value=BAWAAN["sequence_length"], key=f"{P}_seq", disabled=nonaktif, **PERSIST)
    train = c2.number_input(_label("Train ratio", "train_ratio"), min_value=0.05, max_value=0.90, step=0.01,
                            format="%.2f", value=BAWAAN["train_ratio"], key=f"{P}_train", disabled=nonaktif, **PERSIST)
    test = c3.number_input(_label("Test ratio", "test_ratio"), min_value=0.05, max_value=0.90, step=0.01,
                           format="%.2f", value=BAWAAN["test_ratio"], key=f"{P}_test", disabled=nonaktif, **PERSIST)
    val = c4.number_input(_label("Validation ratio", "val_ratio"), min_value=0.0, max_value=1.0, format="%.2f",
                          key=k_val, disabled=True, help="Dihitung otomatis: 1 − train ratio − test ratio.")
    if train + test >= 1:
        errors = {**errors, "rasio": "Train ratio + test ratio harus kurang dari 1 agar tersisa data validasi."}
    forms.row_error(errors, "rasio", kolom=2)

    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    n_layers = c1.number_input(_label("Jumlah layer", "n_layers"), min_value=1, max_value=3, step=1,
                               value=BAWAAN["n_layers"], key=f"{P}_layers", disabled=nonaktif, **PERSIST)
    filters = c2.number_input(_label("Jumlah filter", "filters"), min_value=1, max_value=128, step=8,
                              value=BAWAAN["filters"], key=f"{P}_filters", disabled=nonaktif, **PERSIST)
    kernel = c3.number_input(_label("Kernel size", "kernel_size"), min_value=1, max_value=9, step=2,
                             value=BAWAAN["kernel_size"], key=f"{P}_kernel", disabled=nonaktif, **PERSIST)
    dropout = c4.number_input(_label("Dropout", "dropout"), min_value=0.0, max_value=0.9, step=0.05, format="%.2f",
                              value=BAWAAN["dropout"], key=f"{P}_dropout", disabled=nonaktif, **PERSIST)
    forms.row_error(errors, "kernel_size", kolom=3)

    c1, c2, c3, c4 = st.columns(4, vertical_alignment="bottom")
    epoch = c1.number_input(_label("Epoch maksimum", "max_epochs"), min_value=1, max_value=500, step=5,
                            value=BAWAAN["max_epochs"], key=f"{P}_epoch", disabled=nonaktif, **PERSIST)
    batch = c2.number_input(_label("Batch size", "batch_size"), min_value=1, max_value=128, step=1,
                            value=BAWAAN["batch_size"], key=f"{P}_batch", disabled=nonaktif, **PERSIST)
    lr = c3.number_input(_label("Learning rate", "learning_rate"), min_value=0.00001, max_value=1.0, step=0.0005,
                         format="%.5f", value=BAWAAN["learning_rate"], key=f"{P}_lr", disabled=nonaktif, **PERSIST)
    optimizer = c4.selectbox(_label("Optimizer", "optimizer"), list(models.OPTIMIZER), format_func=models.OPTIMIZER.get,
                             key=f"{P}_optimizer", disabled=nonaktif, **PERSIST)

    params = {
        "sequence_length": int(seq),
        "n_layers": int(n_layers),
        "filters": int(filters),
        "kernel_size": int(kernel),
        "dropout": round(float(dropout), 2),
        "learning_rate": float(lr),
        "optimizer": optimizer,
        "max_epochs": int(epoch),
        "batch_size": int(batch),
        "train_ratio": round(float(train), 2),
        "val_ratio": round(float(val), 2),
        "test_ratio": round(float(test), 2),
    }
    split = models.pembagian_tahun(params) if params["val_ratio"] > 0 else None
    if split:
        st.caption(f"Tahun target {_tahun(split['train'] + split['val'] + split['test'])} dibagi urut waktu: "
                   f"train {_tahun(split['train'])}, validasi {_tahun(split['val'])}, test {_tahun(split['test'])}. "
                   "Latih berjalan tiga kali (validasi dengan early stopping, uji, lalu model final sampai 2025); "
                   "di CPU bisa memakan waktu lama. Pemantauan berlanjut jika Anda pindah halaman lalu kembali, "
                   "selama tab browser ini tidak ditutup.")
    return params


def _skor(nilai) -> str:
    return "-" if nilai is None else fmt.skor(nilai)


def _tabel_metrik(m: dict) -> pd.DataFrame:
    """Tabel model vs persistensi untuk satu tahun test"""
    baris = [
        ("Accuracy", "akurasi", _skor), ("Precision", "precision", _skor), ("Recall", "recall", _skor),
        ("F1-score", "f1", _skor), ("IoU", "iou", _skor), ("FoM", "fom", _skor),
        ("MAE luas (km²)", "MAE_km2", lambda v: "-" if v is None else fmt.angka(v, 3)),
        ("RMSE luas (km²)", "RMSE_km2", lambda v: "-" if v is None else fmt.angka(v, 3)),
        ("MAPE luas", "MAPE_%", lambda v: "-" if v is None else fmt.persen(v, 2)),
        ("R² luas", "R2", _skor),
        ("Bias total luas", "bias_total_%", lambda v: "-" if v is None else fmt.persen(v, 2)),
    ]
    return pd.DataFrame(
        [{"Metrik": label, "Model": f(m.get(k)), "Persistensi": f(m.get(f"persist_{k}"))} for label, k, f in baris]
    )


def _evaluasi(calon: dict) -> None:
    ev = calon["evaluasi"]
    origin = ev["tahun"]["test"][0] - 1
    latih.evaluation_caption(calon, "sampel uji (kecamatan × tahun test)")

    pilihan = {u["tahun"]: u for u in ev["uji"]}
    tahun = st.segmented_control(
        "Tahun test", list(pilihan), default=list(pilihan)[0], key=f"{P}_tahun_uji",
        format_func=lambda t: f"Test {t} ({pilihan[t]['langkah']} langkah{', rekursif' if pilihan[t]['langkah'] > 1 else ''})",
    ) or list(pilihan)[0]
    u = pilihan[tahun]

    c_cm, c_tabel = st.columns([1, 1.2], gap="large")
    with c_cm:
        layout.section(f"Confusion matrix model, {tahun}")
        charts.confusion_matrix(u["confusion"]["model"], ev["label_kelas"])
        layout.metric_cards([
            ("Accuracy", _skor(u["metrik"]["akurasi"])), ("Precision", _skor(u["metrik"]["precision"])),
            ("Recall", _skor(u["metrik"]["recall"])), ("F1-score", _skor(u["metrik"]["f1"])),
        ], min_width=90)
    with c_tabel:
        layout.section(f"Model vs persistensi (peta {origin} disalin)")
        st.dataframe(_tabel_metrik(u["metrik"]), hide_index=True, width="stretch", height="content")

    c_pers, c_ubah = st.columns(2, gap="large")
    with c_pers:
        layout.section(f"Persistensi, {tahun}")
        charts.confusion_matrix(u["confusion"]["persistensi"], ev["label_kelas"], height=220)
    with c_ubah:
        layout.section(f"Model, hanya piksel nonterbangun di {origin}")
        charts.confusion_matrix(u["confusion"]["perubahan"], ["Tidak berubah", "Jadi terbangun"], height=220)

    c_loss, c_info = st.columns([1.4, 1], gap="large")
    with c_loss:
        layout.section("Kurva loss (latih vs validasi)")
        charts.loss_curve(ev["loss"], height=230)
    with c_info:
        layout.section("Data dan pelatihan")
        dipakai = len(calon["kecamatan_ids"])
        fakta = [
            ("Sumber data", models.SUMBER_DATA[ev["sumber"]]),
            ("Kecamatan dipakai", f"{dipakai} kecamatan"),
            ("Tahun target", f"train {_tahun(ev['tahun']['train'])} · validasi {_tahun(ev['tahun']['val'])} · "
                             f"test {_tahun(ev['tahun']['test'])}"),
            ("Sampel latih / validasi", f"{ev['n_latih']} / {ev['n_val']} urutan"),
            ("Epoch terbaik", f"{ev['best_epoch']} dari {ev['epoch_jalan']} epoch berjalan"),
            ("Epoch latih ulang (uji dan final)", str(ev["epochs"])),
            ("Waktu latih", f"{fmt.angka(ev['waktu_latih_detik'] / 60, 1)} menit"),
        ]
        if ev["kecamatan_dilewati"]:
            fakta.append(("Dilewati (arsip tidak lengkap)", ", ".join(ev["kecamatan_dilewati"])))
        layout.facts(fakta)

    st.caption(
        f"Test {ev['tahun']['test'][0]} diprediksi 1 langkah dari {origin} (setara prediksi 2026); tahun berikutnya "
        "secara rekursif (setara prediksi 2027). Sebagian besar piksel tidak berubah dari tahun ke tahun, jadi "
        "persistensi pun sudah akurat: selalu bandingkan model dengan persistensi. FoM hanya menilai piksel yang "
        "berubah dari nonterbangun menjadi terbangun (persistensi selalu 0). Label adalah peta klasifikasi berbasis "
        "NDBI yang sudah dibersihkan dengan aturan konsistensi waktu, bukan kebenaran lapangan."
    )


layout.page_header("Prediksi")
with layout.card("prediksi", group="prediksi", role=user["role"]):
    berjalan, errors = latih.status(P)
    latih.general_error(errors)

    identitas = _identitas(errors, berjalan)
    params = _hyperparameter(errors, berjalan)

    latih.start_button(P, berjalan)
    latih.run_if_started(P, models.CONVLSTM, {**identitas, "params": params})

    st.divider()
    layout.section("Evaluasi model")
    calon = latih.draft(P)
    if calon is None:
        layout.empty_state("Evaluasi muncul di sini setelah model selesai dilatih.", "query_stats")
    else:
        _evaluasi(calon)
        latih.save_discard(P, calon)
