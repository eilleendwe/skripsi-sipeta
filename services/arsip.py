"""
Arsip klasifikasi untuk ConvLSTM: peta 0/1 per tanggal per kecamatan.
Admin bisa mengunggahnya sebagai folder.
"""

import re
from collections import Counter
from pathlib import PurePosixPath
from typing import Iterator

import numpy as np
import rasterio
import streamlit as st
from rasterio.io import MemoryFile

from core import db, peta_tahunan, storage
from core.config import CACHE_TAHUNAN_DIR, LABEL_NODATA
from services import auth, regions, roles
from services.errors import ServiceError

TAHUN_DATA = list(range(2016, 2026))
_TANGGAL_RE = re.compile(r"\d{4}-?\d{2}-?\d{2}")
_TIF_RE = re.compile(r".+\.tiff?", re.IGNORECASE)
# nama hasil unduhan Klasifikasi Citra
_UNDUHAN_RE = re.compile(r"klasifikasi_(.+)_(\d{4}-\d{2}-\d{2})\.tiff?", re.IGNORECASE)


@st.cache_data(ttl=60, show_spinner=False)
def tahun_tersedia() -> dict[str, list[int]]:
    """{kecamatan_id: [tahun, ...]} yang ada di arsip server"""
    tersedia = peta_tahunan.tahun_arsip()
    rows = db.fetch_all("SELECT id, wilayah_id FROM kecamatan")
    return {r["id"]: sorted(tersedia.get((r["wilayah_id"], r["id"]), ())) for r in rows}


def lengkap(kecamatan_ids: list[str]) -> list[str]:
    """Kecamatan yang punya peta klasifikasi untuk semua tahun 2016–2025"""
    arsip = tahun_tersedia()
    return [k for k in kecamatan_ids if arsip.get(k) == TAHUN_DATA]


def scan_folder(nama_berkas: list[str]) -> dict:
    """
    Kenali isi folder unggahan dari nama berkasnya saja. Yang dibaca .../<kecamatan>/<YYYY-MM-DD>/*.tif
    (satu .tif per folder tanggal) atau hasil unduhan klasifikasi_<kecamatan>_<tanggal>.tif di folder mana saja.
    Isi berkasnya baru dicek saat diunggah.
    """
    per_kecamatan, tak_dikenal, diabaikan = {}, set(), 0
    paths = [PurePosixPath(nama.replace("\\", "/")) for nama in nama_berkas]
    # folder tanggal yang isinya banyak .tif kemungkinan citra mentah, jadi dilewati
    jumlah_per_folder = Counter(p.parent for p in paths)
    for nama, path in zip(nama_berkas, paths):
        bagian = path.parts
        unduhan = _UNDUHAN_RE.fullmatch(path.name)
        if unduhan:
            nama_kecamatan, tanggal = unduhan.groups()
        elif (len(bagian) >= 3 and _TANGGAL_RE.fullmatch(bagian[-2]) and _TIF_RE.fullmatch(bagian[-1])
              and jumlah_per_folder[path.parent] == 1):
            nama_kecamatan, tanggal = bagian[-3], bagian[-2]
        else:
            diabaikan += 1
            continue
        kecamatan = regions.cari_kecamatan(nama_kecamatan)
        if kecamatan is None:
            tak_dikenal.add(nama_kecamatan)
            continue
        # Satu berkas per tanggal, yang dipakai berkas pertama.
        per_kecamatan.setdefault(kecamatan["id"], {}).setdefault(tanggal, nama)

    baris = []
    for kec_id, per_tanggal in sorted(per_kecamatan.items()):
        kecamatan = regions.get_kecamatan(kec_id)
        tahun = sorted({int(t[:4]) for t in per_tanggal})
        baris.append({"kecamatan_id": kec_id, "kecamatan": kecamatan["nama"], "wilayah": kecamatan["wilayah_nama"],
                      "tanggal": len(per_tanggal), "tahun": tahun, "lengkap": tahun == TAHUN_DATA})
    return {"baris": baris, "berkas": per_kecamatan, "tak_dikenal": sorted(tak_dikenal), "diabaikan": diabaikan}


def kemas_berkas(laporan: dict, berkas: list, kecamatan_id: str) -> dict:
    """Berkas satu kecamatan dari hasil `scan_folder`: {tanggal: berkas}"""
    isi = {b.name: b for b in berkas}
    return {tanggal: isi[nama] for tanggal, nama in laporan["berkas"][kecamatan_id].items()}


def periksa_peta(kecamatan: dict, berkas: dict) -> dict[str, bytes]:
    """
    Isi peta klasifikasi satu kecamatan sebagai {tanggal: bytes}. Semua berkas dicek dulu: harus GeoTIFF
    bernilai 0/1/255 dengan ukuran yang sama. Kalau ada satu yang tidak valid, error.
    """
    hasil, ukuran = {}, None
    for tanggal, b in sorted(berkas.items()):
        label, isi = f"{kecamatan['nama']} {tanggal}", b.getvalue()
        try:
            with MemoryFile(isi) as mem, mem.open() as src:
                peta = src.read(1)
        except rasterio.errors.RasterioError:
            raise ServiceError(f"{label}: bukan GeoTIFF yang valid.")
        if not np.isin(peta, (0, 1, LABEL_NODATA)).all():
            raise ServiceError(f"{label}: nilai piksel harus 0, 1, atau 255.")
        if ukuran and peta.shape != ukuran:
            raise ServiceError(f"{label}: ukuran raster {peta.shape} berbeda dengan berkas lain {ukuran}.")
        ukuran, hasil[tanggal] = peta.shape, isi
    return hasil


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
            kecamatan = regions.get_kecamatan(b["kecamatan_id"])
            peta = periksa_peta(kecamatan, kemas_berkas(laporan, berkas, kecamatan["id"]))
        except ServiceError as e:
            errors[b["kecamatan"]] = e.message  # kecamatan lain tetap diunggah
            continue
        folder = peta_tahunan.folder_arsip(kecamatan["wilayah_id"], kecamatan["id"])
        for tanggal, isi in peta.items():
            rel = storage.relatif(folder / tanggal)
            storage.hapus(rel)  # tanggal yang sudah ada diganti
            (folder / tanggal).mkdir(parents=True)
            (folder / tanggal / f"klasifikasi_{tanggal}.tif").write_bytes(isi)
            storage.kirim(rel)
        # Arsip berubah, jadi cache peta tahunan kecamatan ini dihapus dan dibuat ulang nanti.
        (CACHE_TAHUNAN_DIR / f"{kecamatan['id']}.npz").unlink(missing_ok=True)
    tahun_tersedia.clear()
    yield {"progress": 1.0, "label": "Selesai", "errors": errors}
