"""Prediksi lahan terbangun dengan model ConvLSTM tersimpan (diproses sebagai job latar belakang)"""

from core import operasi
from services import auth, backend, models, regions
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

    job_id = backend.panggil(operasi.mulai_prediksi, model_id, kecamatan_id, tahun)
    h = backend.tunggu_job(job_id, bar, jeda=0.5)

    return {
        **h,
        "job_id": job_id,
        "hasil_id": None,  # terisi setelah disimpan
        "model_nama": model["nama"],
        "kecamatan_nama": kecamatan["nama"],
        "wilayah_nama": kecamatan["wilayah_nama"],
        "rekursif": is_recursive(tahun),
        "selisih": round(h["luas_terbangun"] - h["luas_acuan"], 4),
        "citra_prediksi": backend.panggil(operasi.berkas_prediksi, job_id, "prediksi.png"),
        "citra_aktual": backend.panggil(operasi.berkas_prediksi, job_id, "aktual.png"),
        "citra_perubahan": backend.panggil(operasi.berkas_prediksi, job_id, "perubahan.png"),
    }


def is_saved(hasil: dict) -> bool:
    return hasil.get("hasil_id") is not None


def save(hasil: dict) -> dict:
    """Simpan hasil prediksi (luas, GeoTIFF 10 m, dan peta model 64×64)"""
    user = auth.current_user()
    if user is None:
        raise ServiceError("Sesi berakhir. Silakan masuk kembali.")
    simpan = backend.panggil(operasi.simpan_hasil_prediksi, hasil["job_id"], user["nama"])
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
            hasil["_tif"] = (backend.panggil(operasi.unduh_hasil_prediksi, hasil["hasil_id"]) if is_saved(hasil)
                             else backend.panggil(operasi.berkas_prediksi, hasil["job_id"], "prediksi.tif"))
        return hasil["_tif"], f"{dasar}.tif", "image/tiff"
    raise ServiceError("Format unduhan tidak dikenal.")
