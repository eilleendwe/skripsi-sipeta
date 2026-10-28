"""Klasifikasi Citra: satu citra kecamatan (7 band + SCL) -> peta biner terbangun/nonterbangun"""

import json
import pickle
from functools import lru_cache
from pathlib import Path

import numpy as np

from core import citra, jobs, prapemrosesan as pra, storage
from core.config import LABEL_NODATA
from core.jobs import ProsesError


class _Unpickler(pickle.Unpickler):
    # Model yang dilatih sebelum folder api/ diganti nama jadi core/ masih menyimpan nama modul api.rf.
    def find_class(self, module, name):
        if module.startswith("api."):
            module = "core." + module[4:]
        return super().find_class(module, name)


@lru_cache(maxsize=4)
def muat_model(path_model: str):
    with open(storage.ambil(path_model), "rb") as f:
        return _Unpickler(f).load()


def muat_config(path_config: str) -> dict:
    return json.loads(storage.ambil(path_config).read_text(encoding="utf-8"))


def jalankan(job_id: str, folder: Path, model_row: dict, kecamatan: dict, tanggal: str) -> dict:
    # Parameter prapemrosesan diambil dari konfigurasi model, bukan dari nilai bawaan.
    config = muat_config(model_row["path_config"])
    fitur = config["features"]
    kelas_awan, buffer_m = config["scl_cloud_classes"], config["cloud_buffer_m"]

    jobs.update(job_id, 0.05, "Membaca citra")
    berkas = citra.index_folder(folder / "input", fitur)
    acuan = citra.periksa_grid(list(berkas.values()))
    res_x, res_y = acuan["res"]
    data = {band: citra.baca(berkas[band]) for band in fitur}
    awan, valid = pra.cloud_mask(citra.baca(berkas["SCL"]), res_x, res_y, kelas_awan, buffer_m)
    if not valid.any():
        raise ProsesError("Citra tidak memiliki piksel valid (SCL seluruhnya 0).")

    # Citra asli untuk ditampilkan, sebelum awannya diisi.
    cerah = valid & ~awan
    png_asli = citra.png_true_color(data["B04"], data["B03"], data["B02"], valid, cerah)

    metode = np.zeros(awan.shape, dtype=np.uint8)
    peringatan = []
    if awan.any():
        jobs.update(job_id, 0.15, "Mengambil anchor arsip")
        anchors = citra.muat_anchor(kecamatan["id"], tanggal, acuan, fitur, kelas_awan, buffer_m)
        if anchors:
            jobs.update(job_id, 0.3, f"Mengisi piksel berawan dengan PCHIP ({len(anchors)} anchor)")
            metode = pra.isi_awan(data, awan, tanggal, anchors, fitur)
        else:
            peringatan.append(
                f"Ada piksel berawan, tapi belum bisa diisi PCHIP karena arsip citra kecamatan "
                f"{kecamatan['nama']} (tanggal cerah sebagai donor) belum didaftarkan. "
                "Piksel awan ditandai abu-abu dan tidak ikut dihitung luasnya."
            )

    jobs.update(job_id, 0.6, "Klasifikasi Random Forest")
    dipakai = valid & (~awan | (metode > 0))
    X = np.stack([data[band][dipakai] for band in fitur], axis=1).astype(np.float32)
    peta = np.full(awan.shape, LABEL_NODATA, dtype=np.uint8)
    peta[dipakai] = muat_model(model_row["path_model"]).predict(X).astype(np.uint8)

    jobs.update(job_id, 0.9, "Menghitung luas dan menyimpan raster")
    km2_per_piksel = res_x * res_y / 1e6
    citra.tulis_geotiff(folder / "klasifikasi.tif", peta, acuan)
    (folder / "asli.png").write_bytes(png_asli)
    # Citra yang awannya sudah diisi (band di `data` ikut berubah oleh isi_awan).
    png_pchip = citra.png_true_color(data["B04"], data["B03"], data["B02"], valid, cerah) if metode.any() else png_asli
    (folder / "pchip.png").write_bytes(png_pchip)
    awan_terisi = awan & (metode > 0)
    awan_kosong = awan & (metode == 0)
    (folder / "klasifikasi.png").write_bytes(citra.png_klasifikasi(peta, awan_kosong))
    (folder / "awan.png").write_bytes(citra.png_masker_awan(awan_terisi, awan_kosong))

    return {
        "model_id": model_row["id"],
        "kecamatan_id": kecamatan["id"],
        "tanggal": tanggal,
        "luas_terbangun": round(float((peta == 1).sum()) * km2_per_piksel, 4),
        "luas_nonterbangun": round(float((peta == 0).sum()) * km2_per_piksel, 4),
        "ukuran_piksel": [acuan["width"], acuan["height"]],
        "resolusi_m": res_x,
        **pra.ringkas_metode(awan, metode, valid),
        "peringatan": peringatan,
    }
