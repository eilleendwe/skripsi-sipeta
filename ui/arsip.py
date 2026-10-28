"""
Arsip klasifikasi di halaman Latih Model Prediksi: panel status + unggah ke arsip server, dan
tabel isi folder unggahan (dipakai juga oleh opsi "Unggah folder").
"""

import pandas as pd
import streamlit as st

from services import arsip, regions
from services.errors import ServiceError
from ui import layout

STRUKTUR = """\
kecamatan_masked_rf_classified/     <- boleh folder ini, satu folder wilayah, atau satu folder kecamatan
└── kota_tangerang/
    ├── batuceper/                  <- nama folder = id atau nama kecamatan
    │   ├── 2016-10-07/
    │   │   └── RF_LABEL_2016-10-07.tif   (0 = nonterbangun, 1 = terbangun, 255 = NoData)
    │   ├── 2017-07-04/
    │   └── ...
    └── ..."""
_VERSI = "_arsip_versi"  # dinaikkan setelah unggah supaya kotak unggah kosong lagi
_ERRORS = "_arsip_errors"


def _rentang(tahun: list[int]) -> str:
    return f"{tahun[0]}–{tahun[-1]}" if tahun else "-"


def _status(tahun: list[int]) -> str:
    if not tahun:
        return "belum ada"
    kurang = [t for t in arsip.TAHUN_DATA if t not in tahun]
    if not kurang:
        return "✅ lengkap"
    if len(kurang) <= 3:
        return "⚠️ tanpa " + ", ".join(map(str, kurang))
    return f"⚠️ {len(arsip.TAHUN_DATA) - len(kurang)} dari {len(arsip.TAHUN_DATA)} tahun"


def laporan_folder(berkas: list, tersedia: dict | None = None) -> dict:
    """
    Tabel isi folder unggahan per kecamatan. Kalau `tersedia` diisi, ada kolom kondisi arsip di server.
    """
    laporan = arsip.scan_folder([b.name for b in berkas])
    if laporan["baris"]:
        baris = []
        for b in laporan["baris"]:
            item = {"Kecamatan": b["kecamatan"], "Kota/Kabupaten": b["wilayah"], "Tanggal": b["tanggal"],
                    "Tahun": _rentang(b["tahun"]), "Status": _status(b["tahun"])}
            if tersedia is not None:
                item["Di server"] = _status(tersedia.get(b["kecamatan_id"], []))
            baris.append(item)
        st.dataframe(pd.DataFrame(baris), hide_index=True, width="stretch",
                     height="content" if len(baris) <= 8 else 320)
    catatan = [f"{len(laporan['baris'])} kecamatan dikenali"]
    if laporan["tak_dikenal"]:
        catatan.append("folder bukan nama kecamatan, diabaikan: " + ", ".join(laporan["tak_dikenal"]))
    if laporan["diabaikan"]:
        catatan.append(f"{laporan['diabaikan']} berkas di luar pola <tanggal>/RF_LABEL_*.tif diabaikan")
    st.caption(" · ".join(catatan) + ".")
    return laporan


def panel(nonaktif: bool) -> None:
    try:
        tersedia = arsip.tahun_tersedia()
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()
    for kecamatan, pesan in st.session_state.pop(_ERRORS, {}).items():
        st.warning(f"Arsip {kecamatan} tidak diunggah: {pesan}", icon=":material/warning:")

    semua = regions.list_kecamatan()
    n_lengkap = len(arsip.lengkap([k["id"] for k in semua]))
    with st.expander(f"Arsip klasifikasi di server: {n_lengkap} dari {len(semua)} kecamatan lengkap "
                     f"{_rentang(arsip.TAHUN_DATA)}", icon=":material/folder_open:"):
        layout.facts([
            (w["nama"], f"{len(arsip.lengkap([k['id'] for k in regions.list_kecamatan(w['id'])]))} dari "
                        f"{len(regions.list_kecamatan(w['id']))} kecamatan lengkap")
            for w in regions.list_wilayah()
        ])
        st.caption("Arsip dipakai opsi latih “Arsip di server” dan selalu menjadi masukan Prediksi Citra. "
                   "Unggah folder di sini untuk menyimpannya di server; tanggal yang sudah ada diganti, tanggal lain "
                   "tetap. Peta tahunan dibuat ulang otomatis setelah arsip berubah.")
        st.code(STRUKTUR, language=None)

        versi = st.session_state.get(_VERSI, 0)
        berkas = st.file_uploader(
            "Folder arsip klasifikasi", type=["tif", "tiff"], accept_multiple_files="directory",
            key=f"arsip_folder_{versi}", disabled=nonaktif,
            help="Kecamatan dikenali dari nama folder di atas folder tanggal. Berkas selain RF_LABEL_*.tif diabaikan.",
        ) or []
        if not berkas:
            return

        laporan = laporan_folder(berkas, tersedia)

        if st.button(f"Unggah {len(laporan['baris'])} kecamatan ke server", type="primary", icon=":material/upload:",
                     key="arsip_unggah", disabled=nonaktif or not laporan["baris"]):
            bar = st.progress(0.0, text="Menyiapkan…")
            try:
                for event in arsip.unggah(laporan, berkas):
                    bar.progress(event["progress"], text=event["label"])
                    errors = event["errors"]
            except ServiceError as e:
                bar.empty()
                st.error(e.message, icon=":material/error:")
                return
            berhasil = len(laporan["baris"]) - len(errors)
            st.session_state[_VERSI] = versi + 1
            st.session_state[_ERRORS] = errors
            layout.set_flash(f"Arsip {berhasil} kecamatan diunggah ke server.", ":material/check_circle:")
            st.rerun()
