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
peta_klasifikasi/                   <- boleh folder ini, satu folder wilayah, atau satu folder kecamatan
├── kota_tangerang/
│   ├── batuceper/                  <- nama folder = id atau nama kecamatan
│   │   ├── 2016-10-07/
│   │   │   └── peta.tif            <- satu file .tif per folder tanggal
│   │   ├── 2017-07-04/
│   │   └── ...
│   └── ...
└── klasifikasi_benda_2016-10-07.tif   <- hasil unduhan Klasifikasi Citra, boleh di folder mana saja

Nilai piksel: 0 = nonterbangun, 1 = terbangun, 255 = NoData."""


def _rentang(tahun: list[int]) -> str:
    return f"{tahun[0]}–{tahun[-1]}" if tahun else "-"


def _status(tahun: list[int]) -> str:
    if not tahun:
        return "belum ada"
    kurang = [t for t in arsip.TAHUN_DATA if t not in tahun]
    if not kurang:
        return "✓ lengkap"
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
        catatan.append(f"{laporan['diabaikan']} berkas bukan peta klasifikasi diabaikan "
                       "(folder tanggal yang berisi lebih dari satu .tif dianggap citra mentah)")
    st.caption(" · ".join(catatan) + ".")
    return laporan


def panel(nonaktif: bool = False) -> None:
    try:
        arsip.tahun_tersedia()
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()

    semua = regions.list_kecamatan()
    n_lengkap = len(arsip.lengkap([k["id"] for k in semua]))
    with st.expander(f"Arsip klasifikasi di server: {n_lengkap} dari {len(semua)} kecamatan lengkap "
                     f"{_rentang(arsip.TAHUN_DATA)}", icon=":material/folder_open:"):
        layout.facts([
            (w["nama"], f"{len(arsip.lengkap([k['id'] for k in regions.list_kecamatan(w['id'])]))} dari "
                        f"{len(regions.list_kecamatan(w['id']))} kecamatan lengkap")
            for w in regions.list_wilayah()
        ])
