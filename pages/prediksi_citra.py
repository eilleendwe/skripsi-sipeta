import streamlit as st

from services import models, predict, roles
from services.errors import ServiceError
from ui import fmt, forms, guard, hasil as hasil_ui, layout

user = guard.require_role(roles.USER)

P = "f_pc"
HASIL = "pc_hasil" 


@st.fragment
def _form() -> dict:
    c_model, c_wil, c_kec, c_tahun = st.columns([1.5, 1, 1, 1])
    _, kecamatan_id = forms.region_select(c_wil, c_kec, P)
    try:
        # Model ConvLSTM cuma dipakai untuk kecamatan yang ikut melatihnya.
        model, peringatan = forms.model_select(c_model, P, models.CONVLSTM, kecamatan_id, luar_cakupan=False)
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()

    tahun = forms.select(c_tahun, "Tahun", predict.TAHUN_PREDIKSI, f"{P}_tahun", placeholder="Pilih tahun")

    for pesan in peringatan:
        st.warning(pesan, icon=":material/warning:")
    if model:
        L = model["params"]["sequence_length"]
        st.caption(f"Masukan: peta tahunan {predict.TAHUN_AKTUAL_TERAKHIR - L + 1}–{predict.TAHUN_AKTUAL_TERAKHIR} "
                   f"kecamatan terpilih dari arsip klasifikasi (sequence length model = {L}).")
    return {"model_id": model["id"] if model else None, "kecamatan_id": kecamatan_id, "tahun": tahun}


def _hasil(hasil: dict) -> None:
    lebar, tinggi = hasil["ukuran_piksel"]
    acuan, tahun = hasil["tahun_acuan"], hasil["tahun"]
    keterangan = f"{lebar} × {tinggi} piksel · 1 piksel = {fmt.angka(hasil['resolusi_m'], 0)} m"
    with st.container(key="kanvas_pc"):
        c_aktual, c_pred, c_ubah = st.columns(3, gap="medium")
        with c_aktual:
            layout.section(f"Aktual {acuan} (peta tahunan)")
            st.image(hasil["citra_aktual"], width="stretch", caption=keterangan)
        with c_pred:
            layout.section(f"Prediksi {tahun} (ConvLSTM)")
            st.image(hasil["citra_prediksi"], width="stretch", caption=keterangan)
        with c_ubah:
            layout.section(f"Perubahan {acuan} → {tahun}")
            st.image(hasil["citra_perubahan"], width="stretch", caption=keterangan)
    layout.legend(predict.LEGENDA + predict.LEGENDA_PERUBAHAN)

    st.divider()
    c_fakta, c_luas = st.columns([1, 1.2], gap="medium")
    with c_fakta:
        masukan = f"peta tahunan {hasil['masukan'][0]}–{hasil['masukan'][-1]}"
        if hasil["rekursif"]:
            masukan += f", lalu prediksi {acuan + 1}"
        layout.facts([
            ("Kota/Kabupaten", hasil["wilayah_nama"]),
            ("Kecamatan", hasil["kecamatan_nama"]),
            ("Tahun", str(tahun)),
            ("Model", hasil["model_nama"]),
            ("Masukan", masukan),
        ])
        if hasil["rekursif"]:
            st.info(f"Akurasi prediksi {tahun} cenderung lebih rendah daripada {acuan + 1} karena memakai hasil "
                    f"prediksi {acuan + 1} sebagai masukan, sehingga kesalahan pada {acuan + 1} ikut terbawa.",
                    icon=":material/info:")
        hasil_ui.save_download("pc", hasil, predict, "Hasil prediksi disimpan.")
    with c_luas:
        layout.metric_cards([
            ("Luas Lahan Terbangun", fmt.km2(hasil["luas_terbangun"])),
            ("Luas Lahan Nonterbangun", fmt.km2(hasil["luas_nonterbangun"])),
        ], min_width=180)
        nada = "naik" if hasil["selisih"] > 0 else ("turun" if hasil["selisih"] < 0 else "tetap")
        st.html(
            f'<div class="sp-delta sp-delta-{nada}">{fmt.selisih_km2(hasil["selisih"])} '
            f'<span>dibanding {acuan} ({fmt.km2(hasil["luas_acuan"])})</span></div>'
        )
        layout.facts([
            ("Bertambah", fmt.km2(hasil["bertambah"])),
            ("Berkurang", fmt.km2(hasil["berkurang"])),
            ("Luas versi 64×64", f"{acuan}: {fmt.km2(hasil['luas_acuan_64'])} · "
                                 f"{tahun}: {fmt.km2(hasil['luas_terbangun_64'])}"),
        ])


layout.page_header("Prediksi")
with layout.card("prediksi", group="prediksi", role=user["role"]):
    isian = _form()

    if st.button("Prediksi", type="primary", icon=":material/play_arrow:", key="pc_jalankan"):
        bar = st.progress(0.0, text="Menyiapkan…")
        try:
            st.session_state[HASIL] = predict.jalankan(bar, **isian)
            bar.empty()
        except ServiceError as e:
            bar.empty()
            st.error(e.message, icon=":material/error:")

    st.divider()
    hasil = st.session_state.get(HASIL)
    if hasil is None:
        layout.empty_state("Pilih model, kecamatan, dan tahun, lalu tekan Prediksi untuk melihat hasil.",
                           "timeline")
    else:
        _hasil(hasil)
