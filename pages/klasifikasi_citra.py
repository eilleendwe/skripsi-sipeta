import streamlit as st

from services import classify, models, roles
from services.errors import ServiceError
from ui import fmt, forms, guard, hasil as hasil_ui, layout, nav

user = guard.require_role(roles.USER)

P = "f_kc"
HASIL = "kc_hasil" 


def _isi_tanggal_dari_berkas() -> None:
    """Saat berkas berubah, isi tanggal dari nama berkas kalau terbaca (tetap bisa diubah)"""
    berkas = st.session_state.get(f"{P}_berkas") or []
    tanggal = classify.detect_upload([b.name for b in berkas])["tanggal"]
    if tanggal and classify.TANGGAL_MIN <= tanggal <= classify.TANGGAL_MAKS:
        st.session_state[f"{P}_tanggal_citra"] = tanggal


def _form() -> dict:
    c_model, c_wil, c_kec, c_tgl = st.columns([1.5, 1, 1, 1])
    _, kecamatan_id = forms.region_select(c_wil, c_kec, P)
    try:
        model, peringatan = forms.model_select(c_model, P, models.RF, kecamatan_id)
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()
    tanggal = c_tgl.date_input(
        "Tanggal citra", value=None, min_value=classify.TANGGAL_MIN, max_value=classify.TANGGAL_MAKS,
        format="DD/MM/YYYY", key=f"{P}_tanggal_citra",
        help="Tanggal akuisisi citra yang diunggah, di dalam rentang arsip 2016–2025. "
             "Diisi otomatis bila tanggal terbaca dari nama berkas.",
    )
    for pesan in peringatan:
        st.warning(pesan, icon=":material/warning:")

    berkas = st.file_uploader(
        "Berkas band (" + ", ".join(classify.BAND_WAJIB) + ")", accept_multiple_files=True,
        type=["tif", "tiff"], key=f"{P}_berkas", on_change=_isi_tanggal_dari_berkas,
        help="Unggah citra satu kecamatan: tujuh band fitur dan SCL sekaligus. "
             "Band dikenali dari nama berkas, mis. …_B02_10m.tif. Tanpa SCL, awan tidak bisa dideteksi.",
    ) or []
    _daftar_band(classify.detect_upload([b.name for b in berkas]))
    return {"model_id": model["id"] if model else None, "kecamatan_id": kecamatan_id, "tanggal": tanggal,
            "berkas": berkas}


def _daftar_band(deteksi: dict) -> None:
    keping = "".join(
        f'<span class="sp-pill sp-pill-{"green" if ada else "red"}">{s} {"✓" if ada else "✗"}</span>'
        for s, ada in deteksi["band"].items()
    )
    st.html(f'<div class="sp-chips">{keping}</div>')
    if deteksi["duplikat"]:
        st.caption("Band duplikat, diabaikan: " + ", ".join(deteksi["duplikat"]))
    if deteksi["tidak_dikenali"]:
        st.caption("Tidak dipakai, diabaikan: " + ", ".join(deteksi["tidak_dikenali"]))


def _hasil(hasil: dict) -> None:
    lebar, tinggi = hasil["ukuran_piksel"]
    keterangan = f"{lebar} × {tinggi} piksel · 1 piksel = {fmt.angka(hasil['resolusi_m'], 0)} m"
    c_asli, c_kelas, c_info = st.columns([1, 1, 0.9], gap="medium")

    with c_asli:
        judul = st.empty()
        kanvas_asli = st.container(key="kanvas_kc_asli")
        pchip = forms.mode_citra("kc_mode_citra")
        tampil_masker = st.toggle("Tampilkan masker awan", key="kc_masker",
                                  help="Piksel awan hasil cloud masking SCL (kelas 3, 8, 9, 10) + buffer 30 m.")
        tampil_overlay = st.toggle("Tampilkan overlay", key="kc_overlay")
        layout.section(("Citra setelah PCHIP" if pchip else "Citra asli") + " (B04-B03-B02)", judul)
        if tampil_masker:
            kanvas_asli.image(classify.dengan_masker_awan(hasil, pchip), width="stretch", caption=keterangan)
            layout.legend(classify.LEGENDA_AWAN)
        else:
            kanvas_asli.image(classify.citra(hasil, pchip), width="stretch", caption=keterangan)

    with c_kelas:
        layout.section("Citra hasil klasifikasi" + (" (overlay)" if tampil_overlay else ""))
        kanvas = st.container(key="kanvas_kc_kelas")
        if tampil_overlay:
            opasitas = st.slider("Opasitas overlay", 0, 100, 60, step=5, format="%d%%", key="kc_opasitas")
            kanvas.image(classify.overlay(hasil, opasitas / 100, pchip), width="stretch", caption=keterangan)
        else:
            kanvas.image(hasil["citra_klasifikasi"], width="stretch", caption=keterangan)

    with c_info:
        layout.facts([
            ("Kota/Kabupaten", hasil["wilayah_nama"]),
            ("Kecamatan", hasil["kecamatan_nama"]),
            ("Tanggal", fmt.tanggal(hasil["tanggal"])),
            ("Model", hasil["model_nama"]),
        ])
        layout.metric_cards([
            ("Luas Lahan Terbangun", fmt.km2(hasil["luas_terbangun"])),
            ("Luas Lahan Nonterbangun", fmt.km2(hasil["luas_nonterbangun"])),
        ], min_width=180)
        layout.facts([
            ("Piksel berawan", fmt.persen(hasil["persen_awan"], 2)),
            ("Terisi PCHIP", fmt.persen(hasil["persen_pchip"], 2)),
            ("Terisi backward/forward fill",
             fmt.persen(hasil["persen_backward"] + hasil["persen_forward"], 2)),
            ("Tidak terisi (NoData)", fmt.persen(hasil["persen_tidak_terisi"], 2)),
        ])
        if hasil["persen_terisi"] > classify.BATAS_TERISI:
            st.warning(
                f"Lebih dari {classify.BATAS_TERISI:.0f}% piksel tertutup awan dan diisi interpolasi. "
                "Pertimbangkan tanggal lain.", icon=":material/cloud:",
            )
        for pesan in hasil["peringatan"]:
            st.warning(pesan, icon=":material/cloud_off:")
        if hasil["di_luar_cakupan"]:
            st.warning("Model ini dilatih pada kecamatan lain. Penerapannya di kecamatan ini belum dievaluasi.",
                       icon=":material/warning:")

        hasil_ui.save_download("kc", hasil, classify, "Hasil disimpan dan tersedia di Pemantauan.")
        if classify.is_saved(hasil):
            st.page_link(nav.path("pemantauan"), label="Lihat di Pemantauan", icon=":material/monitoring:")

    layout.legend(classify.LEGENDA)


layout.page_header("Klasifikasi")
with layout.card("klasifikasi", group="klasifikasi", role=user["role"]):
    isian = _form()

    if st.button("Klasifikasi", type="primary", icon=":material/play_arrow:", key="kc_jalankan"):
        bar = st.progress(0.0, text="Menyiapkan…")
        try:
            st.session_state[HASIL] = classify.jalankan(bar, **isian)
            bar.empty()
        except ServiceError as e:
            bar.empty()
            st.error(e.message, icon=":material/error:")

    st.divider()
    hasil = st.session_state.get(HASIL)
    if hasil is None:
        layout.empty_state("Pilih model, kecamatan, dan tanggal, unggah citra, lalu tekan Klasifikasi.",
                           "satellite_alt")
    else:
        _hasil(hasil)
