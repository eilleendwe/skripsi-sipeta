"""
Pemantauan: membandingkan hasil klasifikasi tersimpan antar tanggal.
Membaca hasil dari klasifkasi citra `services.classify.save'.
"""

import base64
from datetime import date

from core import operasi
from services import backend, regions
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
    rows = backend.panggil(operasi.daftar_hasil, kecamatan_id, model_id)
    for r in rows:
        r["tanggal"] = date.fromisoformat(r["tanggal"])
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
    data = backend.panggil(operasi.bandingkan, ringkasan["hasil_awal"], ringkasan["hasil_akhir"])

    def png(kunci):
        return base64.b64decode(data[kunci]) if data.get(kunci) else None

    return {
        "awal_asli": png("awal_asli_png"),
        "akhir_asli": png("akhir_asli_png"),
        "awal_pchip": png("awal_pchip_png"),
        "akhir_pchip": png("akhir_pchip_png"),
        "awal": png("awal_png"),
        "akhir": png("akhir_png"),
        "perubahan": png("perubahan_png"),
        "bertambah": data["bertambah"],
        "berkurang": data["berkurang"],
    }
