"""Klasifikasi citra dengan model Random Forest tersimpan (diproses sebagai job latar belakang)"""

import io
import re
import shutil
from datetime import date

import numpy as np
from PIL import Image

from core import db, jobs, klasifikasi, storage
from core.config import CITRA_DIR, HASIL_DIR, TMP_DIR
from services import auth, models, regions, unggahan
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
    model_row = db.fetch_one("SELECT * FROM model_klasifikasi WHERE id = %s", (model["id"],))
    job_id = jobs.buat("klasifikasi")
    folder = TMP_DIR / job_id
    if unggahan.simpan_band(files, folder / "input", pakai_subfolder_tanggal=False) == 0:
        jobs.hapus(job_id)
        raise ServiceError("Tidak ada berkas band yang dikenali.")
    jobs.mulai(job_id, klasifikasi.jalankan, folder, model_row, kecamatan, tanggal.isoformat())
    h = jobs.tunggu(job_id, bar)

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
        "citra_asli": jobs.berkas(job_id, "asli.png"),
        "citra_pchip": jobs.berkas(job_id, "pchip.png"),
        "citra_klasifikasi": jobs.berkas(job_id, "klasifikasi.png"),
        "masker_awan": jobs.berkas(job_id, "awan.png"),
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


def path_pchip(rel_raster: str) -> str:
    """Citra setelah PCHIP disimpan di sebelah raster hasilnya: <tanggal>_model<id>_pchip.png"""
    return rel_raster.removesuffix(".tif") + "_pchip.png"


def save(hasil: dict) -> dict:
    """
    Simpan hasil supaya muncul di Pemantauan. Model yang dipakai ikut dicatat. Hasil lama dengan model,
    kecamatan, dan tanggal yang sama ditimpa.
    """
    user = auth.current_user()
    if user is None:
        raise ServiceError("Sesi berakhir. Silakan masuk kembali.")
    h = jobs.selesai(hasil["job_id"])["hasil"]
    folder = TMP_DIR / hasil["job_id"]

    path_raster = HASIL_DIR / h["kecamatan_id"] / f"{h['tanggal']}_model{h['model_id']}.tif"
    path_citra = CITRA_DIR / h["kecamatan_id"] / h["tanggal"]
    rel_raster, rel_citra = storage.relatif(path_raster), storage.relatif(path_citra)
    rel_pchip = path_pchip(rel_raster)
    path_raster.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / "klasifikasi.tif", path_raster)
    shutil.copy2(folder / "pchip.png", storage.lokal(rel_pchip))
    storage.hapus(rel_citra)  # citra masukan lama di tanggal ini diganti
    shutil.copytree(folder / "input", path_citra)
    for rel in (rel_raster, rel_pchip, rel_citra):
        storage.kirim(rel)

    simpan = db.fetch_one(
        "INSERT INTO hasil_klasifikasi (model_id, kecamatan_id, tanggal, luas_terbangun, luas_nonterbangun, "
        "persen_awan, persen_pchip, persen_backward, persen_forward, persen_tidak_terisi, path_raster, path_citra, "
        "disimpan_oleh) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (model_id, kecamatan_id, tanggal) DO UPDATE SET "
        "luas_terbangun = EXCLUDED.luas_terbangun, luas_nonterbangun = EXCLUDED.luas_nonterbangun, "
        "persen_awan = EXCLUDED.persen_awan, persen_pchip = EXCLUDED.persen_pchip, "
        "persen_backward = EXCLUDED.persen_backward, persen_forward = EXCLUDED.persen_forward, "
        "persen_tidak_terisi = EXCLUDED.persen_tidak_terisi, path_raster = EXCLUDED.path_raster, "
        "path_citra = EXCLUDED.path_citra, disimpan_oleh = EXCLUDED.disimpan_oleh, disimpan_pada = now() "
        "RETURNING id",
        (h["model_id"], h["kecamatan_id"], h["tanggal"], h["luas_terbangun"], h["luas_nonterbangun"],
         h["persen_awan"], h["persen_pchip"], h["persen_backward"], h["persen_forward"], h["persen_tidak_terisi"],
         rel_raster, rel_citra, user["nama"]),
    )
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
            if is_saved(hasil):
                row = db.fetch_one("SELECT path_raster FROM hasil_klasifikasi WHERE id = %s", (hasil["hasil_id"],))
                hasil["_tif"] = storage.ambil(row["path_raster"]).read_bytes()
            else:
                hasil["_tif"] = jobs.berkas(hasil["job_id"], "klasifikasi.tif")
        return hasil["_tif"], f"{dasar}.tif", "image/tiff"
    raise ServiceError("Format unduhan tidak dikenal.")
