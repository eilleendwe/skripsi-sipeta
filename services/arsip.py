"""
Arsip klasifikasi untuk ConvLSTM: peta 0/1 per tanggal (RF_LABEL_*.tif) per kecamatan.
Admin bisa mengunggahnya sebagai folder.
"""

import re
from pathlib import PurePosixPath
from typing import Iterator

import streamlit as st

from core import operasi
from services import auth, backend, regions, roles
from services.errors import ServiceError

TAHUN_DATA = list(range(2016, 2026))  # tahun yang dibutuhkan ConvLSTM per kecamatan
_TANGGAL_RE = re.compile(r"\d{4}-?\d{2}-?\d{2}")
_RF_LABEL_RE = re.compile(r"RF_LABEL_.*\.tif", re.IGNORECASE)


@st.cache_data(ttl=60, show_spinner=False)
def tahun_tersedia() -> dict[str, list[int]]:
    """{kecamatan_id: [tahun, ...]} yang ada di arsip server"""
    return backend.panggil(operasi.arsip_klasifikasi)


def lengkap(kecamatan_ids: list[str]) -> list[str]:
    """Kecamatan yang punya peta klasifikasi untuk semua tahun 2016–2025"""
    arsip = tahun_tersedia()
    return [k for k in kecamatan_ids if arsip.get(k) == TAHUN_DATA]


def scan_folder(nama_berkas: list[str]) -> dict:
    """
    Kenali isi folder unggahan dari nama berkasnya saja. Struktur yang dibaca:
    .../<kecamatan>/<YYYY-MM-DD>/RF_LABEL_*.tif, jadi yang dipilih boleh folder induk, satu folder
    wilayah, atau satu folder kecamatan. Isi berkasnya baru dicek saat diunggah.
    """
    per_kecamatan, tak_dikenal, diabaikan = {}, set(), 0
    for nama in nama_berkas:
        bagian = PurePosixPath(nama.replace("\\", "/")).parts
        if len(bagian) < 3 or not _TANGGAL_RE.fullmatch(bagian[-2]) or not _RF_LABEL_RE.fullmatch(bagian[-1]):
            diabaikan += 1
            continue
        kecamatan = regions.cari_kecamatan(bagian[-3])
        if kecamatan is None:
            tak_dikenal.add(bagian[-3])
            continue
        # Satu berkas per tanggal, yang dipakai berkas pertama.
        per_kecamatan.setdefault(kecamatan["id"], {}).setdefault(bagian[-2], nama)

    baris = []
    for kec_id, per_tanggal in sorted(per_kecamatan.items()):
        kecamatan = regions.get_kecamatan(kec_id)
        tahun = sorted({int(t[:4]) for t in per_tanggal})
        baris.append({"kecamatan_id": kec_id, "kecamatan": kecamatan["nama"], "wilayah": kecamatan["wilayah_nama"],
                      "tanggal": len(per_tanggal), "tahun": tahun, "lengkap": tahun == TAHUN_DATA})
    return {"baris": baris, "berkas": per_kecamatan, "tak_dikenal": sorted(tak_dikenal), "diabaikan": diabaikan}


def kemas_berkas(laporan: dict, berkas: list, kecamatan_id: str, folder_kecamatan: bool = False) -> list:
    """
    Berkas satu kecamatan sebagai (nama, berkas), dengan nama <tanggal>/<berkas>
    (atau <kecamatan>/<tanggal>/<berkas> kalau `folder_kecamatan`).
    """
    isi = {b.name: b for b in berkas}
    files = []
    for tanggal, nama in laporan["berkas"][kecamatan_id].items():
        dasar = PurePosixPath(nama.replace("\\", "/")).name
        path = f"{tanggal}/{dasar}"
        if folder_kecamatan:
            path = f"{kecamatan_id}/{path}"
        files.append((path, isi[nama]))
    return files


def unggah(laporan: dict, berkas: list) -> Iterator[dict]:
    """
    Unggah arsip per kecamatan sambil kirim event progress. Kecamatan yang ditolak dikumpulkan
    di `errors` pada event terakhir.
    """
    user = auth.current_user()
    if user is None or not roles.has_role(user["role"], roles.ADMIN):
        raise ServiceError("Hanya Admin yang dapat mengunggah arsip klasifikasi.")
    errors = {}
    for i, b in enumerate(laporan["baris"]):
        yield {"progress": i / len(laporan["baris"]), "label": f"Mengunggah {b['kecamatan']} "
                                                              f"({i + 1}/{len(laporan['baris'])})", "errors": None}
        try:
            backend.panggil(operasi.unggah_arsip_klasifikasi, b["kecamatan_id"],
                            kemas_berkas(laporan, berkas, b["kecamatan_id"]))
        except ServiceError as e:
            errors[b["kecamatan"]] = e.message  # kecamatan lain tetap diunggah
    tahun_tersedia.clear()
    yield {"progress": 1.0, "label": "Selesai", "errors": errors}
