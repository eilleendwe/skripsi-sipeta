"""Tombol Simpan dan Unduh untuk panel hasil Klasifikasi Citra dan Prediksi Citra"""

import streamlit as st

from services.errors import ServiceError
from ui import layout

_FORMAT = (("png", "PNG"), ("tif", "GeoTIFF"))


def save_download(prefix: str, hasil: dict, service, pesan_tersimpan: str) -> None:
    """
    Tombol Simpan hasil dan Unduh. `service` = services.classify atau services.predict
    (punya `is_saved`, `save`, dan `export`).
    """
    tersimpan = service.is_saved(hasil)
    with st.container(horizontal=True):
        if st.button("Tersimpan" if tersimpan else "Simpan hasil", type="primary", disabled=tersimpan,
                     icon=":material/check:" if tersimpan else ":material/save:", key=f"{prefix}_simpan"):
            try:
                service.save(hasil)
            except ServiceError as e:
                st.error(e.message, icon=":material/error:")
            else:
                layout.set_flash(pesan_tersimpan, ":material/check_circle:")
                st.rerun()
        with st.popover("Unduh", icon=":material/download:"):
            for kode, label in _FORMAT:
                try:
                    isi, nama, mime = service.export(hasil, kode)
                except ServiceError as e:
                    st.caption(f"{label}: {e.message}")
                    continue
                st.download_button(label, isi, file_name=nama, mime=mime, key=f"{prefix}_unduh_{kode}",
                                   on_click="ignore", width="stretch")
