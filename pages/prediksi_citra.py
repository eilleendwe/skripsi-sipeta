import streamlit as st

from services import models, predict, roles
from services.errors import ServiceError
from ui import badges, fmt, forms, guard, hasil as hasil_ui, layout

user = guard.require_role(roles.USER)

P = "f_pc"
HASIL = "pc_hasil" 
_LENCANA_REKURSIF = "prediksi rekursif — akurasi lebih rendah"


def _form() -> dict:
    c_model, c_wil, c_kec, c_tahun = st.columns([1.5, 1, 1, 1])
    _, kecamatan_id = forms.region_select(c_wil, c_kec, P)
    try:
        # Model ConvLSTM cuma dipakai untuk kecamatan yang ikut melatihnya.
        model, peringatan = forms.model_select(c_model, P, models.CONVLSTM, kecamatan_id, luar_cakupan=False)
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()

    tahun = forms.select(
        c_tahun, "Tahun", predict.TAHUN_PREDIKSI, f"{P}_tahun", placeholder="Pilih tahun",
        format_func=lambda t: f"{t} (rekursif)" if predict.is_recursive(t) else str(t),
    )
    if tahun and predict.is_recursive(tahun):
        c_tahun.html(badges.pill_html(_LENCANA_REKURSIF, "amber", wrap=True))

    for pesan in peringatan:
        st.warning(pesan, icon=":material/warning:")
    if model:
        L = model["params"]["sequence_length"]
        st.caption(f"Masukan: peta tahunan {predict.TAHUN_AKTUAL_TERAKHIR - L + 1}–{predict.TAHUN_AKTUAL_TERAKHIR} "
                   f"kecamatan terpilih dari arsip klasifikasi (sequence length model = {L}).")
    return {"model_id": model["id"] if model else None, "kecamatan_id": kecamatan_id, "tahun": tahun}


def _hasil(hasil: dict) -> None:
    lebar, tinggi = hasil["ukuran_piksel"]
    sel_h, sel_w = hasil["ukuran_sel_m"]
    acuan, tahun = hasil["tahun_acuan"], hasil["tahun"]
    keterangan = f"{lebar} × {tinggi} piksel · 1 piksel = {fmt.angka(hasil['resolusi_m'], 0)} m"
    c_aktual, c_pred, c_info = st.columns([1, 1, 0.9], gap="medium")

    with c_aktual:
        layout.section(f"Aktual {acuan} (peta tahunan)")
        with st.container(key="kanvas_pc_aktual"):
            st.image(hasil["citra_aktual"], width="stretch", caption=keterangan)

    with c_pred:
        perubahan = st.session_state.get("pc_perubahan", False)
        layout.section(f"Perubahan {acuan} → {tahun}" if perubahan else f"Prediksi {tahun} (ConvLSTM)")
        kanvas = st.container(key="kanvas_pc_prediksi")
        st.toggle("Tampilkan perubahan", key="pc_perubahan",
                  help=f"Piksel yang bertambah atau berkurang dibanding peta aktual {acuan}.")
        if perubahan:
            kanvas.image(hasil["citra_perubahan"], width="stretch", caption=keterangan)
            layout.legend(predict.LEGENDA_PERUBAHAN)
        else:
            kanvas.image(hasil["citra_prediksi"], width="stretch", caption=keterangan)

    with c_info:
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
            badges.pill(_LENCANA_REKURSIF, "amber")
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
        hasil_ui.save_download("pc", hasil, predict, "Hasil prediksi disimpan.")

    layout.legend(predict.LEGENDA)
    st.caption(
        f"ConvLSTM bekerja pada peta 64×64 (1 sel ≈ {sel_h} × {sel_w} m untuk kecamatan ini). Peta 10 m disusun "
        f"dari peta aktual {acuan}: sel yang menurut model berubah diganti seluruhnya, sel lain tetap memakai detail "
        f"asli {acuan}. Jadi detail halus berasal dari data {acuan}, sedangkan perubahan hasil prediksi berupa kotak "
        f"seukuran satu sel model. Luas utama dihitung dari peta 10 m; versi 64×64 hanya pembanding."
    )


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
