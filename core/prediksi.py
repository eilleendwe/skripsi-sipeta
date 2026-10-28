"""
Prediksi lahan terbangun dengan model ConvLSTM yang sudah disimpan.
Model bekerja di peta 64x64. Peta yang ditampilkan dan dihitung luasnya versi 10 m.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import tensorflow as tf

from core import citra, convlstm, jobs, peta_tahunan, storage
from core.jobs import ProsesError


@lru_cache(maxsize=2)
def muat_model(path_model: str):
    # Layer ConvLSTMManual sudah terdaftar di Keras karena convlstm diimpor di atas.
    return tf.keras.models.load_model(storage.ambil(path_model), compile=False)


def muat_config(path_config: str) -> dict:
    return json.loads(storage.ambil(path_config).read_text(encoding="utf-8"))


def jalankan(job_id: str, folder: Path, model_row: dict, kecamatan: dict, tahun: int) -> dict:
    # Sequence length, threshold, dan tahun terakhir diambil dari konfigurasi model.
    config = muat_config(model_row["path_config"])
    L = config["hyperparameters"]["sequence_length"]
    threshold = config["threshold"]
    terakhir = config["forecast"]["last_observed_year"]

    jobs.update(job_id, 0.05, f"Menyiapkan peta tahunan {kecamatan['nama']}")
    cache = peta_tahunan.muat(kecamatan["wilayah_id"], kecamatan["id"])
    masukan = list(range(terakhir - L + 1, terakhir + 1))
    if cache is None or any(t not in cache["tahun"] for t in masukan):
        raise ProsesError(f"Arsip klasifikasi kecamatan {kecamatan['nama']} belum lengkap. Dibutuhkan peta "
                          f"tahun {masukan[0]}–{masukan[-1]}.")
    window = cache["patch"][[cache["tahun"].index(t) for t in masukan]]

    jobs.update(job_id, 0.4, "Memuat model ConvLSTM")
    model = muat_model(model_row["path_model"])
    langkah = tahun - terakhir
    jobs.update(job_id, 0.6, f"Prediksi {tahun}" + (f" ({langkah} langkah, rekursif)" if langkah > 1 else ""))
    probs, bins = convlstm.forecast_recursive(model, window[np.newaxis, ..., np.newaxis], langkah, threshold)
    biner = bins[-1][0]

    jobs.update(job_id, 0.85, "Menyusun peta 10 m dan menghitung luas")
    aktual = cache["asli_terakhir"]
    peta = convlstm.native_forecast(aktual, window[-1], biner)
    acuan = citra.profil(Path(cache["berkas_acuan"]))
    km2_piksel = acuan["res"][0] * acuan["res"][1] / 1e6
    km2_sel = peta_tahunan.km2_per_sel(aktual.shape)

    citra.tulis_geotiff(folder / "prediksi.tif", peta, acuan)
    (folder / "prediksi.png").write_bytes(citra.png_klasifikasi(peta))
    (folder / "aktual.png").write_bytes(citra.png_klasifikasi(aktual))
    (folder / "perubahan.png").write_bytes(citra.png_perubahan(aktual, peta))
    # Peta 64x64 untuk semua langkah: peluang 0-1 dan biner (-1 = NoData).
    np.savez_compressed(folder / "peta.npz",
                        **{f"peluang_{terakhir + i + 1}": p[0] for i, p in enumerate(probs)},
                        **{f"biner_{terakhir + i + 1}": b[0] for i, b in enumerate(bins)})

    def luas(nilai, kelas, km2):
        return round(float((nilai == kelas).sum()) * km2, 4)

    tinggi, lebar = aktual.shape
    return {
        "model_id": model_row["id"],
        "kecamatan_id": kecamatan["id"],
        "tahun": tahun,
        "masukan": masukan,
        "tahun_acuan": terakhir,
        "luas_terbangun": luas(peta, 1, km2_piksel),
        "luas_nonterbangun": luas(peta, 0, km2_piksel),
        "luas_acuan": luas(aktual, 1, km2_piksel),
        "bertambah": round(float(((aktual == 0) & (peta == 1)).sum()) * km2_piksel, 4),
        "berkurang": round(float(((aktual == 1) & (peta == 0)).sum()) * km2_piksel, 4),
        # Versi 64x64, cuma sebagai pembanding.
        "luas_terbangun_64": luas(biner, 1, km2_sel),
        "luas_acuan_64": luas(window[-1], 1, km2_sel),
        "ukuran_piksel": [lebar, tinggi],
        "resolusi_m": acuan["res"][0],
        "ukuran_sel_m": [round(tinggi / peta_tahunan.PATCH_SIZE * acuan["res"][1]),
                         round(lebar / peta_tahunan.PATCH_SIZE * acuan["res"][0])],
    }
