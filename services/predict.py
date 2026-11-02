"""Prediksi lahan terbangun dengan model ConvLSTM tersimpan (diproses sebagai job latar belakang)"""

import shutil

from core import db, jobs, storage
from core.config import PREDIKSI_DIR, TMP_DIR
from services import auth, models, regions
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
    belum disimpan.
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
        "hasil_id": None,  # terisi setelah disimpan
        "model_nama": model["nama"],
        "kecamatan_nama": kecamatan["nama"],
        "wilayah_nama": kecamatan["wilayah_nama"],
        "rekursif": is_recursive(tahun),
        "selisih": round(h["luas_terbangun"] - h["luas_acuan"], 4),
        "citra_prediksi": jobs.berkas(job_id, "prediksi.png"),
        "citra_aktual": jobs.berkas(job_id, "aktual.png"),
        "citra_perubahan": jobs.berkas(job_id, "perubahan.png"),
    }


def is_saved(hasil: dict) -> bool:
    return hasil.get("hasil_id") is not None


def save(hasil: dict) -> dict:
    """
    Simpan hasil prediksi (luas, GeoTIFF 10 m, dan peta model 64×64). Hasil lama dengan model,
    kecamatan, dan tahun yang sama ditimpa.
    """
    user = auth.current_user()
    if user is None:
        raise ServiceError("Sesi berakhir. Silakan masuk kembali.")
    h = jobs.selesai(hasil["job_id"])["hasil"]
    folder = TMP_DIR / hasil["job_id"]
    dasar = PREDIKSI_DIR / h["kecamatan_id"] / f"{h['tahun']}_model{h['model_id']}"
    dasar.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(folder / "prediksi.tif", dasar.with_suffix(".tif"))
    shutil.copy2(folder / "peta.npz", dasar.with_suffix(".npz"))
    rel_raster, rel_peta = storage.relatif(dasar.with_suffix(".tif")), storage.relatif(dasar.with_suffix(".npz"))
    storage.kirim(rel_raster)
    storage.kirim(rel_peta)

    simpan = db.fetch_one(
        "INSERT INTO hasil_prediksi (model_id, kecamatan_id, tahun, luas_terbangun, luas_nonterbangun, luas_acuan, "
        "luas_terbangun_64, path_raster, path_peta, disimpan_oleh) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (model_id, kecamatan_id, tahun) DO UPDATE SET "
        "luas_terbangun = EXCLUDED.luas_terbangun, luas_nonterbangun = EXCLUDED.luas_nonterbangun, "
        "luas_acuan = EXCLUDED.luas_acuan, luas_terbangun_64 = EXCLUDED.luas_terbangun_64, "
        "path_raster = EXCLUDED.path_raster, path_peta = EXCLUDED.path_peta, "
        "disimpan_oleh = EXCLUDED.disimpan_oleh, disimpan_pada = now() RETURNING id",
        (h["model_id"], h["kecamatan_id"], h["tahun"], h["luas_terbangun"], h["luas_nonterbangun"], h["luas_acuan"],
         h["luas_terbangun_64"], rel_raster, rel_peta, user["nama"]),
    )
    hasil["hasil_id"] = simpan["id"]  # objek yang sama dengan di session_state, jadi tandanya tetap ada
    return simpan


def export(hasil: dict, fmt: str) -> tuple[bytes, str, str]:
    """
    Berkas unduhan: (isi, nama berkas, tipe MIME). `fmt`: png atau tif (GeoTIFF bergeoreferensi).
    """
    dasar = f"prediksi_{hasil['kecamatan_id']}_{hasil['tahun']}"
    if fmt == "png":
        return hasil["citra_prediksi"], f"{dasar}.png", "image/png"
    if fmt == "tif":
        if "_tif" not in hasil:
            if is_saved(hasil):
                row = db.fetch_one("SELECT path_raster FROM hasil_prediksi WHERE id = %s", (hasil["hasil_id"],))
                hasil["_tif"] = storage.ambil(row["path_raster"]).read_bytes()
            else:
                hasil["_tif"] = jobs.berkas(hasil["job_id"], "prediksi.tif")
        return hasil["_tif"], f"{dasar}.tif", "image/tiff"
    raise ServiceError("Format unduhan tidak dikenal.")
