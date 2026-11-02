"""
Pemantauan: membandingkan hasil klasifikasi tersimpan antar tanggal.
Membaca hasil dari klasifkasi citra `services.classify.save'.
"""

from datetime import date

import numpy as np

from core import citra, db, storage
from core.config import LABEL_NODATA
from services import classify, regions
from services.errors import ServiceError

LEGENDA_KELAS = [
    ("Terbangun", (180, 88, 58)),
    ("Nonterbangun", (111, 168, 104)),
]
LEGENDA_PERUBAHAN = [
    ("Bertambah (menjadi terbangun)", (201, 73, 47)),
    ("Berkurang (menjadi nonterbangun)", (31, 120, 180)),
    ("Tetap terbangun", (74, 71, 67)),
    ("Tetap nonterbangun", (217, 221, 214)),
]


def _results(kecamatan_id: str | None = None, model_id: int | None = None) -> list[dict]:
    rows = db.fetch_all(
        "SELECT h.id, h.model_id, m.nama AS model_nama, h.kecamatan_id, h.tanggal, h.luas_terbangun, "
        "h.luas_nonterbangun, h.persen_awan, h.persen_pchip, h.persen_backward, h.persen_forward, "
        "h.persen_tidak_terisi, h.disimpan_oleh, h.disimpan_pada "
        "FROM hasil_klasifikasi h JOIN model_klasifikasi m ON m.id = h.model_id "
        "WHERE (%(kec)s::text IS NULL OR h.kecamatan_id = %(kec)s) "
        "AND (%(model)s::int IS NULL OR h.model_id = %(model)s) ORDER BY h.tanggal",
        {"kec": kecamatan_id, "model": model_id},
    )
    for r in rows:
        r["persen_terisi"] = round((r["persen_pchip"] or 0) + (r["persen_backward"] or 0)
                                   + (r["persen_forward"] or 0), 2)
    return rows


def has_results() -> bool:
    return bool(_results())


def models_with_results(kecamatan_id: str) -> list[dict]:
    """Model yang punya hasil tersimpan untuk kecamatan ini: id, nama, jumlah hasil"""
    per_model: dict[int, dict] = {}
    for r in _results(kecamatan_id):
        m = per_model.setdefault(r["model_id"], {"id": r["model_id"], "nama": r["model_nama"], "jumlah": 0})
        m["jumlah"] += 1
    return sorted(per_model.values(), key=lambda m: (-m["jumlah"], m["nama"]))


def dates(model_id: int, kecamatan_id: str) -> list[date]:
    """Tanggal yang punya hasil tersimpan untuk model dan kecamatan ini"""
    return sorted(r["tanggal"] for r in _results(kecamatan_id, model_id))


def history(model_id: int, kecamatan_id: str) -> list[dict]:
    """Riwayat hasil tersimpan, urut tanggal"""
    return sorted(_results(kecamatan_id, model_id), key=lambda r: r["tanggal"])


def _find(model_id: int, kecamatan_id: str, tanggal: date) -> dict:
    for r in _results(kecamatan_id, model_id):
        if r["tanggal"] == tanggal:
            return r
    raise ServiceError("Tanggal yang dipilih tidak memiliki hasil tersimpan untuk model dan kecamatan ini.")


def _check_range(awal: date | None, akhir: date | None) -> None:
    if awal is None or akhir is None:
        raise ServiceError("Pilih tanggal awal dan tanggal akhir.")
    if awal >= akhir:
        raise ServiceError("Tanggal akhir harus setelah tanggal awal.")


def compare(model_id: int | None, kecamatan_id: str | None, awal: date | None, akhir: date | None) -> dict:
    """Ringkasan perubahan luas lahan terbangun antara dua tanggal"""
    kecamatan = regions.get_kecamatan(kecamatan_id)
    if kecamatan is None:
        raise ServiceError("Pilih kecamatan terlebih dahulu.")
    if not model_id:
        raise ServiceError("Pilih model terlebih dahulu.")
    _check_range(awal, akhir)
    a, b = _find(model_id, kecamatan_id, awal), _find(model_id, kecamatan_id, akhir)
    perubahan = round(b["luas_terbangun"] - a["luas_terbangun"], 2)
    return {
        "model_id": model_id,
        "model_nama": a["model_nama"],
        "kecamatan_id": kecamatan_id,
        "kecamatan_nama": kecamatan["nama"],
        "wilayah_nama": kecamatan["wilayah_nama"],
        "awal": awal,
        "akhir": akhir,
        "hasil_awal": a["id"],
        "hasil_akhir": b["id"],
        "luas_awal": a["luas_terbangun"],
        "luas_akhir": b["luas_terbangun"],
        "perubahan": perubahan,
        "perubahan_persen": round(100 * perubahan / a["luas_terbangun"], 1) if a["luas_terbangun"] else 0.0,
    }


def maps(ringkasan: dict) -> dict:
    """
    Citra asli, citra setelah PCHIP, dan peta klasifikasi kedua tanggal (PNG), peta perubahan, dan luas
    perubahannya. Citra bernilai None kalau berkasnya tidak tersimpan (citra PCHIP belum ada di hasil lama).
    """
    a, b = (db.fetch_one("SELECT * FROM hasil_klasifikasi WHERE id = %s", (i,))
            for i in (ringkasan["hasil_awal"], ringkasan["hasil_akhir"]))
    path_a, path_b = storage.ambil(a["path_raster"]), storage.ambil(b["path_raster"])
    peta_a, peta_b = citra.baca(path_a), citra.baca(path_b)
    if peta_a.shape != peta_b.shape:
        raise ServiceError("Ukuran raster kedua tanggal berbeda, jadi tidak bisa dibandingkan per piksel.")
    res = citra.profil(path_a)["res"]
    km2 = res[0] * res[1] / 1e6
    valid = (peta_a != LABEL_NODATA) & (peta_b != LABEL_NODATA)

    def asli(row) -> bytes | None:
        return citra.png_dari_folder_citra(storage.ambil_folder(row["path_citra"])) if row["path_citra"] else None

    def pchip(row) -> bytes | None:
        try:
            return storage.ambil(classify.path_pchip(row["path_raster"])).read_bytes()
        except FileNotFoundError:  # hasil yang disimpan sebelum citra PCHIP ikut disimpan
            return None

    return {
        "awal_asli": asli(a),
        "akhir_asli": asli(b),
        "awal_pchip": pchip(a),
        "akhir_pchip": pchip(b),
        "awal": citra.png_klasifikasi(peta_a),
        "akhir": citra.png_klasifikasi(peta_b),
        "perubahan": citra.png_perubahan(peta_a, peta_b),
        "bertambah": round(float(np.sum(valid & (peta_a == 0) & (peta_b == 1))) * km2, 4),
        "berkurang": round(float(np.sum(valid & (peta_a == 1) & (peta_b == 0))) * km2, 4),
    }
