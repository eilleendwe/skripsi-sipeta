import pandas as pd
import streamlit as st

from services import monitoring, roles
from services.errors import ServiceError
from ui import charts, fmt, forms, guard, layout, nav

user = guard.require_role(roles.USER)

P = "f_pm"
TAMPIL = "pm_tampil"  # ringkasan yang sedang tampil (hasil tombol Tampilkan Data)
PETA = "pm_peta"  # gambar ringkasan itu, supaya tidak dibaca ulang dari storage tiap rerun


def _link_klasifikasi() -> None:
    st.page_link(nav.path("klasifikasi_citra"), label="Buka Klasifikasi citra", icon=":material/satellite_alt:")


def _form() -> dict:
    c_wil, c_kec, c_model, c_awal, c_akhir = st.columns([1.15, 1, 1.3, 1, 1])
    _, kecamatan_id = forms.region_select(c_wil, c_kec, P)

    # Hasil yang dibandingkan harus dari model yang sama, jadi model dipilih dulu
    tersedia = {m["id"]: m for m in monitoring.models_with_results(kecamatan_id)} if kecamatan_id else {}
    model_id = forms.select(
        c_model, "Model", list(tersedia), f"{P}_model",
        format_func=lambda i: f"{tersedia[i]['nama']} · {tersedia[i]['jumlah']} hasil",
        placeholder=("Pilih model" if tersedia else "Belum ada hasil tersimpan") if kecamatan_id
        else "Pilih kecamatan dahulu",
    )

    # Hanya tanggal yang punya hasil tersimpan untuk model + kecamatan ini
    tanggal = monitoring.dates(model_id, kecamatan_id) if model_id else []
    awal = forms.select(c_awal, "Tanggal awal", tanggal[:-1], f"{P}_awal", format_func=fmt.tanggal,
                        placeholder="Pilih tanggal")
    akhir = forms.select(c_akhir, "Tanggal akhir", [t for t in tanggal if awal and t > awal], f"{P}_akhir",
                         format_func=fmt.tanggal, placeholder="Pilih tanggal")

    if kecamatan_id and not tersedia:
        st.info("Belum ada hasil klasifikasi tersimpan untuk kecamatan ini. "
                "Jalankan Klasifikasi citra lalu tekan Simpan hasil.", icon=":material/info:")
        _link_klasifikasi()
    elif model_id and len(tanggal) < 2:
        st.info("Model ini baru memiliki satu hasil tersimpan untuk kecamatan ini. "
                "Simpan hasil klasifikasi pada tanggal lain untuk membandingkan.", icon=":material/info:")
    return {"model_id": model_id, "kecamatan_id": kecamatan_id, "awal": awal, "akhir": akhir}


def _ringkasan(r: dict) -> None:
    st.caption(f"{r['kecamatan_nama']}, {r['wilayah_nama']} · model {r['model_nama']}")
    c_teks, c_delta = st.columns([1, 1.6], vertical_alignment="center")
    layout.facts([
        (f"Luas awal ({fmt.tanggal(r['awal'])})", fmt.km2(r["luas_awal"])),
        (f"Luas akhir ({fmt.tanggal(r['akhir'])})", fmt.km2(r["luas_akhir"])),
    ], container=c_teks)
    nada = "naik" if r["perubahan"] > 0 else ("turun" if r["perubahan"] < 0 else "tetap")
    tanda = "+" if r["perubahan_persen"] > 0 else ("−" if r["perubahan_persen"] < 0 else "")
    c_delta.html(
        '<div class="sp-section">Perubahan luas lahan terbangun</div>'
        f'<div class="sp-delta sp-delta-besar sp-delta-{nada}">{fmt.selisih_km2(r["perubahan"])} '
        f'<span>({tanda}{fmt.persen(abs(r["perubahan_persen"]))} dari luas awal)</span></div>'
    )


def _gambar(container, isi, judul: str) -> None:
    container.html(f'<div class="sp-section">{judul}</div>')
    if isi:
        container.image(isi, width="stretch")
    else:
        container.caption("Citra tidak tersedia untuk hasil ini.")


def _citra(container, peta: dict, kunci: str, tanggal: str, pchip: bool) -> None:
    """
    Citra satelit satu tanggal (`kunci`: awal/akhir). Hasil lama belum punya citra PCHIP,
    jadi yang tampil citra asli.
    """
    if pchip and peta[f"{kunci}_pchip"]:
        _gambar(container, peta[f"{kunci}_pchip"], f"Setelah PCHIP · {tanggal}")
        return
    _gambar(container, peta[f"{kunci}_asli"], f"Citra asli · {tanggal}")
    if pchip:
        container.caption("Hasil ini belum punya citra setelah PCHIP. Klasifikasi ulang tanggal ini lalu simpan lagi.")


def _peta(r: dict) -> None:
    """
    Baris atas: citra satelit kedua tanggal + angka perubahan. Baris bawah: peta klasifikasi kedua
    tanggal + peta perubahan. 1 piksel = 10 m.
    """
    awal, akhir = fmt.tanggal(r["awal"]), fmt.tanggal(r["akhir"])
    peta = st.session_state.get(PETA)
    if peta is None:
        try:
            peta = st.session_state[PETA] = monitoring.maps(r)
        except ServiceError as e:
            st.error(e.message, icon=":material/error:")
            return

    pchip = forms.mode_citra("pm_mode_citra")
    with st.container(key="kanvas_pm"):
        c1, c2, c3 = st.columns(3, gap="medium")
        _citra(c1, peta, "awal", awal, pchip)
        _citra(c2, peta, "akhir", akhir, pchip)
        with c3:
            layout.section("Perubahan piksel")
            layout.metric_cards([
                ("Bertambah", fmt.km2(peta["bertambah"])),
                ("Berkurang", fmt.km2(peta["berkurang"])),
                ("Perubahan bersih", fmt.selisih_km2(r["perubahan"])),
            ], min_width=200)
            st.caption("Bertambah dan berkurang dihitung dari piksel yang valid pada kedua tanggal, jadi "
                       "selisihnya bisa sedikit berbeda dari perubahan bersih jika ada piksel NoData.")

        c1, c2, c3 = st.columns(3, gap="medium")
        _gambar(c1, peta["awal"], f"Klasifikasi · {awal}")
        _gambar(c2, peta["akhir"], f"Klasifikasi · {akhir}")
        _gambar(c3, peta["perubahan"], f"Peta perubahan · {awal} → {akhir}")
    layout.legend(monitoring.LEGENDA_KELAS + monitoring.LEGENDA_PERUBAHAN)


def _riwayat(r: dict) -> None:
    riwayat = monitoring.history(r["model_id"], r["kecamatan_id"])
    layout.section("Luas lahan terbangun terhadap waktu")
    charts.area_trend(riwayat, sorot=(r["awal"], r["akhir"]))

    layout.section("Riwayat hasil tersimpan")
    st.dataframe(
        pd.DataFrame([
            {"Tanggal": fmt.tanggal(h["tanggal"]), "Model": h["model_nama"],
             "Luas terbangun (km²)": fmt.angka(h["luas_terbangun"]),
             "Luas nonterbangun (km²)": fmt.angka(h["luas_nonterbangun"])}
            for h in riwayat
        ]),
        hide_index=True, width="stretch",
    )


layout.page_header("Pemantauan")
with layout.card("pemantauan"):
    try:
        ada_hasil = monitoring.has_results()
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()
    if not ada_hasil:
        layout.empty_state("Belum ada hasil klasifikasi tersimpan. "
                           "Jalankan Klasifikasi citra lalu tekan Simpan hasil.", "monitoring")
        _link_klasifikasi()
        st.stop()

    isian = _form()
    if st.button("Tampilkan Data", type="primary", icon=":material/query_stats:", key="pm_tampilkan"):
        try:
            st.session_state[TAMPIL] = monitoring.compare(**isian)
            st.session_state.pop(PETA, None)
        except ServiceError as e:
            st.error(e.message, icon=":material/error:")

    ringkasan = st.session_state.get(TAMPIL)
    if ringkasan is None:
        st.divider()
        layout.empty_state("Pilih kecamatan, model, dan dua tanggal, lalu tekan Tampilkan Data.", "monitoring")
    else:
        st.divider()
        layout.section("Hasil")
        _ringkasan(ringkasan)
        st.divider()
        layout.section("Perbandingan spasial")
        _peta(ringkasan)
        st.divider()
        _riwayat(ringkasan)
