"""Wilayah dan kecamatan, dibaca dari database."""

import re

import streamlit as st

from core import operasi
from services import backend


@st.cache_data(ttl=600, show_spinner=False)
def _wilayah() -> list[dict]:
    return backend.panggil(operasi.daftar_wilayah)


def list_wilayah() -> list[dict]:
    return [{"id": w["id"], "nama": w["nama"]} for w in _wilayah()]


def get_wilayah(wilayah_id: str) -> dict | None:
    for w in list_wilayah():
        if w["id"] == wilayah_id:
            return w
    return None


def list_kecamatan(wilayah_id: str | None = None) -> list[dict]:
    """Kecamatan di satu wilayah, atau semua kecamatan kalau `wilayah_id` None."""
    hasil = []
    for w in _wilayah():
        if wilayah_id and w["id"] != wilayah_id:
            continue
        for k in w["kecamatan"]:
            hasil.append({"id": k["id"], "nama": k["nama"], "wilayah_id": w["id"], "wilayah_nama": w["nama"]})
    return hasil


def get_kecamatan(kecamatan_id: str | None) -> dict | None:
    for k in list_kecamatan():
        if k["id"] == kecamatan_id:
            return k
    return None


def nama_kecamatan(kecamatan_id: str) -> str:
    k = get_kecamatan(kecamatan_id)
    return k["nama"] if k else kecamatan_id


def nama_sama(a: str, b: str) -> bool:
    """Bandingkan nama tanpa peduli huruf besar/kecil, spasi, dan tanda baca."""
    def bersih(s):
        return re.sub(r"[^a-z0-9]", "", s.lower())
    return bersih(a) == bersih(b)


def cari_kecamatan(nama_folder: str) -> dict | None:
    """Kecamatan yang id atau namanya sama dengan nama folder, cth. 'pondok_aren' atau 'Pondok Aren'."""
    for k in list_kecamatan():
        if nama_sama(nama_folder, k["id"]) or nama_sama(nama_folder, k["nama"]):
            return k
    return None
