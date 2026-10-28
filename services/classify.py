"""Klasifikasi citra dengan model Random Forest tersimpan (diproses sebagai job latar belakang)"""

import io
import re
from datetime import date

import numpy as np
from PIL import Image

from core import operasi
from services import auth, backend, models, regions
from services.errors import ServiceError

BAND_WAJIB = ["B02", "B03", "B04", "B05", "B06", "B07", "B12", "SCL"]
BATAS_TERISI = 30.0
TANGGAL_MIN, TANGGAL_MAKS = date(2016, 1, 1), date(2025, 12, 31)

_BAND_RE = re.compile(r"(?<![A-Za-z0-9])(B(?:0[1-9]|8A|1[0-2])|SCL)(?![A-Za-z0-9])", re.IGNORECASE)
_TANGGAL_RE = re.compile(r"(?<!\d)(20\d{2})[-_.]?(0[1-9]|1[0-2])[-_.]?(0[1-9]|[12]\d|3[01])(?!\d)")

LEGENDA = [
    ("Terbangun", (180, 88, 58)),
    ("Nonterbangun", (111, 168, 104)),
    ("Awan tidak terisi (NoData)", (140, 156, 170)),
]
LEGENDA_AWAN = [
    ("Awan terisi PCHIP / backward / forward fill", (0, 144, 138)),
    ("Awan tidak terisi", (214, 150, 30)),
]


def detect_upload(filenames: list[str]) -> dict:
    """Kenali band dan tanggal akuisisi dari nama berkas unggahan"""
    ditemukan: dict[str, str] = {}
    duplikat, tidak_dikenali, tanggal = [], [], None
    for nama in filenames:
        cocok = _BAND_RE.search(nama)
        band = cocok.group(1).upper() if cocok else None
        if band not in BAND_WAJIB:
            tidak_dikenali.append(nama)
            continue
        if band in ditemukan:
            duplikat.append(nama)
            continue
        ditemukan[band] = nama
        if tanggal is None:
            t = _TANGGAL_RE.search(nama)
            if t:
                try:
                    tanggal = date(int(t.group(1)), int(t.group(2)), int(t.group(3)))
                except ValueError:
                    pass
    kurang = [s for s in BAND_WAJIB if s not in ditemukan]
    return {
        "band": {s: s in ditemukan for s in BAND_WAJIB},
        "kurang": kurang,
        "duplikat": duplikat,
        "tidak_dikenali": tidak_dikenali,
        "tanggal": tanggal,
        "berkas": list(ditemukan.values()),
        "lengkap": not kurang,
    }


def jalankan(bar, model_id, kecamatan_id: str | None, tanggal: date | None, berkas: list | None) -> dict:
    """
    Klasifikasi satu citra, progresnya ditampilkan di `bar`. Hasilnya belum disimpan.
    """
    model = models.get(models.RF, model_id)
    if model is None or model["algoritma"] != models.RF:
        raise ServiceError("Pilih model Random Forest terlebih dahulu.")
    kecamatan = regions.get_kecamatan(kecamatan_id)
    if kecamatan is None:
        raise ServiceError("Pilih kecamatan terlebih dahulu.")
    if tanggal is None:
        raise ServiceError("Isi tanggal citra terlebih dahulu.")
    if not TANGGAL_MIN <= tanggal <= TANGGAL_MAKS:
        raise ServiceError(f"Tanggal harus di dalam rentang arsip ({TANGGAL_MIN.year}–{TANGGAL_MAKS.year}).")
    berkas = berkas or []
    deteksi = detect_upload([b.name for b in berkas])
    if not deteksi["lengkap"]:
        raise ServiceError("Band belum lengkap: " + ", ".join(deteksi["kurang"]) + ".")

    dipakai = set(deteksi["berkas"])
    files = [(b.name, b) for b in berkas if b.name in dipakai]
    bar.progress(0.0, text="Menyiapkan citra…")
    job_id = backend.panggil(operasi.mulai_klasifikasi, model["id"], kecamatan_id, tanggal.isoformat(), files)
    h = backend.tunggu_job(job_id, bar)

    return {
        "job_id": job_id,
        "hasil_id": None,  # terisi setelah disimpan
        "model_id": model["id"],
        "model_nama": model["nama"],
        "di_luar_cakupan": not models.covers(model, kecamatan_id),
        "kecamatan_id": kecamatan_id,
        "kecamatan_nama": kecamatan["nama"],
        "wilayah_nama": kecamatan["wilayah_nama"],
        "tanggal": tanggal,
        **{k: h[k] for k in ("luas_terbangun", "luas_nonterbangun", "persen_awan", "persen_pchip",
                             "persen_backward", "persen_forward", "persen_tidak_terisi", "peringatan",
                             "ukuran_piksel", "resolusi_m")},
        "persen_terisi": round(h["persen_pchip"] + h["persen_backward"] + h["persen_forward"], 2),
        "citra_asli": backend.panggil(operasi.berkas_klasifikasi, job_id, "asli.png"),
        "citra_pchip": backend.panggil(operasi.berkas_klasifikasi, job_id, "pchip.png"),
        "citra_klasifikasi": backend.panggil(operasi.berkas_klasifikasi, job_id, "klasifikasi.png"),
        "masker_awan": backend.panggil(operasi.berkas_klasifikasi, job_id, "awan.png"),
    }


def _rgba(png: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(png)).convert("RGBA"), dtype=np.float32)


def _tumpuk(bawah_png: bytes, atas_png: bytes, opasitas: float) -> np.ndarray:
    """Gambar `atas` ditumpuk di atas `bawah` (opasitas 0..1), tetap 1 piksel = 10 m"""
    bawah, atas = _rgba(bawah_png), _rgba(atas_png)
    ada = atas[..., 3:] > 0
    rgb = np.where(ada, bawah[..., :3] * (1 - opasitas) + atas[..., :3] * opasitas, bawah[..., :3])
    alpha = np.maximum(bawah[..., 3:], atas[..., 3:])
    return np.concatenate([rgb, alpha], axis=-1).astype(np.uint8)


def citra(hasil: dict, pchip: bool) -> bytes:
    """Citra asli, atau citra yang awannya sudah diisi PCHIP"""
    return hasil["citra_pchip"] if pchip else hasil["citra_asli"]


def overlay(hasil: dict, opasitas: float, pchip: bool = False) -> np.ndarray:
    """Hasil klasifikasi ditumpuk di atas citra"""
    return _tumpuk(citra(hasil, pchip), hasil["citra_klasifikasi"], opasitas)


def dengan_masker_awan(hasil: dict, pchip: bool = False) -> np.ndarray:
    """Citra dengan masker awan (SCL + buffer 30 m) di atasnya"""
    return _tumpuk(citra(hasil, pchip), hasil["masker_awan"], 0.7)


def is_saved(hasil: dict) -> bool:
    return hasil.get("hasil_id") is not None


def save(hasil: dict) -> dict:
    """
    Simpan hasil supaya muncul di Pemantauan. Model yang dipakai ikut dicatat.
    """
    user = auth.current_user()
    if user is None:
        raise ServiceError("Sesi berakhir. Silakan masuk kembali.")
    simpan = backend.panggil(operasi.simpan_hasil, hasil["job_id"], user["nama"])
    # `hasil` objek yang sama dengan yang ada di session_state, jadi tanda tersimpan tetap ada saat rerun.
    hasil["hasil_id"] = simpan["id"]
    return simpan


def export(hasil: dict, fmt: str) -> tuple[bytes, str, str]:
    """
    Berkas unduhan: (isi, nama berkas, tipe MIME). `fmt`: png atau tif (GeoTIFF bergeoreferensi).
    """
    dasar = f"klasifikasi_{hasil['kecamatan_id']}_{hasil['tanggal'].isoformat()}"
    if fmt == "png":
        return hasil["citra_klasifikasi"], f"{dasar}.png", "image/png"
    if fmt == "tif":
        if "_tif" not in hasil:
            hasil["_tif"] = (backend.panggil(operasi.unduh_hasil, hasil["hasil_id"]) if is_saved(hasil)
                             else backend.panggil(operasi.berkas_klasifikasi, hasil["job_id"], "klasifikasi.tif"))
        return hasil["_tif"], f"{dasar}.tif", "image/tiff"
    raise ServiceError("Format unduhan tidak dikenal.")
