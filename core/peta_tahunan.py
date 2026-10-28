"""
Peta tahunan untuk ConvLSTM, dipakai Latih Prediksi dan Prediksi Citra.
Masukannya peta klasifikasi 0/1 per tanggal (255 = NoData)
"""

import logging
import re
from pathlib import Path

import numpy as np
import rasterio

from core import storage
from core.config import ARSIP_KLASIFIKASI_DIR, CACHE_TAHUNAN_DIR, LABEL_NODATA, LUAS_PIKSEL_M2

log = logging.getLogger("sipeta")

PATCH_SIZE = 64
MAJORITY_MIN_FRAC = 0.5
KM2_PIKSEL = LUAS_PIKSEL_M2 / 1e6
TANGGAL_RE = re.compile(r"\d{4}-?\d{2}-?\d{2}")    # nama folder tanggal, mis. 2019-07-04 atau 20190704
RF_LABEL_RE = re.compile(r"RF_LABEL_.*\.tif", re.IGNORECASE)

# Disimpan di konfigurasi model, supaya ketahuan aturan apa yang dipakai saat latih.
ATURAN = {
    "composite_mode": "mayoritas",
    "majority_min_frac": MAJORITY_MIN_FRAC,
    "tie_rule": "seri = 0",
    "temporal_consistency": "terbangun di t-1 dan t-2 -> tetap terbangun di t",
    "patch_size": PATCH_SIZE,
    "resize": "nearest",
    "nodata_in": LABEL_NODATA,
    "nodata_model": -1,
}


def folder_arsip(wilayah_id: str, kecamatan_id: str) -> Path:
    return ARSIP_KLASIFIKASI_DIR / wilayah_id / kecamatan_id


def daftar_berkas(folder: Path) -> dict[int, list[Path]]:
    """{tahun: [berkas RF_LABEL, ...]} dari folder satu kecamatan (isinya folder tanggal), urut tanggal"""
    hasil = {}
    if not folder.is_dir():
        return hasil
    for d in sorted(folder.iterdir()):
        if not (d.is_dir() and TANGGAL_RE.match(d.name)):
            continue
        tifs = sorted(p for p in d.iterdir() if RF_LABEL_RE.fullmatch(p.name))
        if tifs:
            hasil.setdefault(int(d.name[:4]), []).append(tifs[0])
    return hasil


def tahun_arsip() -> dict[tuple[str, str], set[int]]:
    """
    {(wilayah_id, kecamatan_id): tahun yang punya peta} di seluruh arsip klasifikasi.
    Cukup dari daftar nama berkas, jadi di mode s3 petanya tidak perlu diunduh.
    """
    hasil = {}
    for rel in storage.daftar(storage.relatif(ARSIP_KLASIFIKASI_DIR)):
        bagian = rel.split("/")
        if len(bagian) == 5 and TANGGAL_RE.match(bagian[3]) and RF_LABEL_RE.fullmatch(bagian[4]):
            hasil.setdefault((bagian[1], bagian[2]), set()).add(int(bagian[3][:4]))
    return hasil


def komposit_mayoritas(paths: list[Path]) -> np.ndarray:
    """
    Gabung semua tanggal dalam setahun jadi satu peta: 1 kalau terbangun di lebih dari 50% tanggal
    valid (seri = 0), 0 kalau tidak, 255 kalau tidak pernah valid.
    """
    n_built = n_valid = None
    for path in paths:
        with rasterio.open(path) as src:
            arr = src.read(1)
        if n_built is None:
            n_built = np.zeros(arr.shape, np.uint16)
            n_valid = np.zeros(arr.shape, np.uint16)
        elif arr.shape != n_built.shape:
            log.warning("%s: ukuran %s berbeda, dilewati", path, arr.shape)
            continue
        valid = arr != LABEL_NODATA
        n_valid += valid
        n_built += valid & (arr == 1)
    frac = n_built / np.maximum(n_valid, 1)
    annual = (frac > MAJORITY_MIN_FRAC).astype(np.uint8)
    annual[n_valid == 0] = LABEL_NODATA
    return annual


def konsistensi_waktu(maps: list[np.ndarray]) -> list[np.ndarray]:
    """
    Piksel yang terbangun 2 tahun berturut-turut sebelumnya tetap dianggap terbangun.
    Cuma melihat tahun sebelumnya, tidak pernah tahun sesudahnya. NoData dibiarkan.
    """
    out = [m.copy() for m in maps]
    for t in range(2, len(out)):
        if out[t].shape != out[t - 1].shape or out[t].shape != out[t - 2].shape:
            continue
        keep = (out[t - 1] == 1) & (out[t - 2] == 1) & (out[t] == 0)
        out[t][keep] = 1
    return out


def ubah_ukuran(arr: np.ndarray, tinggi: int, lebar: int) -> np.ndarray:
    """Resize nearest neighbour, karena label tidak boleh diinterpolasi"""
    baris = np.minimum(((np.arange(tinggi) + 0.5) * arr.shape[0] / tinggi).astype(int), arr.shape[0] - 1)
    kolom = np.minimum(((np.arange(lebar) + 0.5) * arr.shape[1] / lebar).astype(int), arr.shape[1] - 1)
    return arr[np.ix_(baris, kolom)]


def ke_patch(peta: np.ndarray) -> np.ndarray:
    """
    Peta 10 m -> 64x64 untuk ConvLSTM. NoData jadi -1 (tidak ikut loss dan evaluasi).
    """
    kecil = ubah_ukuran(peta, PATCH_SIZE, PATCH_SIZE)
    out = (kecil == 1).astype(np.float32)
    out[kecil == LABEL_NODATA] = -1
    return out


def km2_per_sel(bentuk_asli) -> float:
    """Luas satu sel 64x64: (tinggi x lebar raster asli) piksel 10 m dibagi 64²"""
    h, w = bentuk_asli
    return h * w * KM2_PIKSEL / (PATCH_SIZE * PATCH_SIZE)


def _bangun(berkas: dict[int, list[Path]]) -> dict:
    """
    Peta tahunan dari daftar berkas: komposit mayoritas + konsistensi waktu, lalu diubah ke 64x64.

    Isinya: tahun, patch (n_tahun, 64, 64), piksel_terbangun per tahun (resolusi asli),
    asli_terakhir (peta 10 m tahun terakhir, untuk tampilan), dan berkas_acuan (untuk georeferensi).
    """
    tahun = sorted(berkas)
    maps = konsistensi_waktu([komposit_mayoritas(berkas[t]) for t in tahun])
    return {
        "tahun": tahun,
        "patch": np.stack([ke_patch(m) for m in maps]),
        "piksel_terbangun": np.array([int((m == 1).sum()) for m in maps]),
        "asli_terakhir": maps[-1],
        "berkas_acuan": str(berkas[tahun[-1]][0]),
        "jumlah_berkas": sum(len(v) for v in berkas.values()),
    }


def dari_folder(folder: Path) -> dict | None:
    """
    Peta tahunan dari folder unggahan satu kecamatan, tanpa cache. None kalau kosong.
    """
    berkas = daftar_berkas(folder)
    return _bangun(berkas) if berkas else None


def muat(wilayah_id: str, kecamatan_id: str) -> dict | None:
    """
    Peta tahunan satu kecamatan dari arsip. Pakai cache, dan dibuat ulang kalau cache belum ada
    atau jumlah berkas arsipnya berubah. None kalau arsipnya kosong.
    """
    folder = storage.ambil_folder(storage.relatif(folder_arsip(wilayah_id, kecamatan_id)))
    berkas = daftar_berkas(folder)
    jumlah = sum(len(v) for v in berkas.values())
    if jumlah == 0:
        return None

    path = CACHE_TAHUNAN_DIR / f"{kecamatan_id}.npz"
    if path.exists():
        with np.load(path) as f:
            cache = {k: f[k] for k in f.files}
        if int(cache["jumlah_berkas"]) == jumlah:
            cache["tahun"] = cache["tahun"].tolist()
            cache["berkas_acuan"] = str(cache["berkas_acuan"])
            return cache

    cache = _bangun(berkas)
    CACHE_TAHUNAN_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **cache)
    return cache
