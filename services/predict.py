"""Prediksi lahan terbangun dengan model ConvLSTM tersimpan (diproses sebagai job latar belakang)"""

from core import db, jobs
from core.config import TMP_DIR
from services import models, regions
from services.errors import ServiceError

TAHUN_AKTUAL_TERAKHIR = 2025
TAHUN_PREDIKSI = [2026, 2027]

LEGENDA = [
    ("Terbangun", (180, 88, 58)),
    ("Nonterbangun", (111, 168, 104)),
]
LEGENDA_PERUBAHAN = [
    ("Bertambah (menjadi terbangun)", (201, 73, 47)),
    ("Berkurang (menjadi nonterbangun)", (31, 120, 180)),
    ("Tetap terbangun", (74, 71, 67)),
    ("Tetap nonterbangun", (217, 221, 214)),
]


def is_recursive(tahun: int) -> bool:
    """Tahun yang lebih dari satu langkah ke depan diprediksi dari hasil prediksi sebelumnya"""
    return tahun > TAHUN_AKTUAL_TERAKHIR + 1


def jalankan(bar, model_id: int | None, kecamatan_id: str | None, tahun: int | None) -> dict:
    """
    Prediksi lahan terbangun satu kecamatan di `tahun`, progresnya ditampilkan di `bar`. Hasilnya
    tidak disimpan, cuma bisa diunduh.
    """
    model = models.get(models.CONVLSTM, model_id)
    if model is None:
        raise ServiceError("Pilih model ConvLSTM terlebih dahulu.")
    kecamatan = regions.get_kecamatan(kecamatan_id)
    if kecamatan is None:
        raise ServiceError("Pilih kecamatan terlebih dahulu.")
    if tahun not in TAHUN_PREDIKSI:
        raise ServiceError("Pilih tahun prediksi terlebih dahulu.")

    from core import prediksi  # memuat TensorFlow, jadi baru diimpor di sini
    model_row = db.fetch_one("SELECT * FROM model_prediksi WHERE id = %s", (model_id,))
    if kecamatan_id not in model_row["kecamatan_ids"]:
        raise ServiceError(f"Model ini tidak dilatih dengan kecamatan {kecamatan['nama']}.")
    job_id = jobs.buat("prediksi")
    folder = TMP_DIR / job_id
    folder.mkdir(parents=True)
    jobs.mulai(job_id, prediksi.jalankan, folder, model_row, kecamatan, tahun)
    h = jobs.tunggu(job_id, bar, jeda=0.5)

    return {
        **h,
        "job_id": job_id,
        "model_nama": model["nama"],
        "kecamatan_nama": kecamatan["nama"],
        "wilayah_nama": kecamatan["wilayah_nama"],
        "rekursif": is_recursive(tahun),
        "selisih": round(h["luas_terbangun"] - h["luas_acuan"], 4),
        "citra_prediksi": jobs.berkas(job_id, "prediksi.png"),
        "citra_aktual": jobs.berkas(job_id, "aktual.png"),
        "citra_perubahan": jobs.berkas(job_id, "perubahan.png"),
    }


def export(hasil: dict, fmt: str) -> tuple[bytes, str, str]:
    """
    Berkas unduhan: (isi, nama berkas, tipe MIME). `fmt`: png atau tif (GeoTIFF bergeoreferensi).
    """
    dasar = f"prediksi_{hasil['kecamatan_id']}_{hasil['tahun']}"
    if fmt == "png":
        return hasil["citra_prediksi"], f"{dasar}.png", "image/png"
    if fmt == "tif":
        if "_tif" not in hasil:
            hasil["_tif"] = jobs.berkas(hasil["job_id"], "prediksi.tif")
        return hasil["_tif"], f"{dasar}.tif", "image/tiff"
    raise ServiceError("Format unduhan tidak dikenal.")
