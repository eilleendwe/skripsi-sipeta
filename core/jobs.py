"""
Job latar belakang untuk proses yang lama (latih model, klasifikasi, prediksi).
Job jalan di thread dalam proses, jadi tetap lanjut walau pengguna pindah halaman. Status job disimpan di memori, jadi hilang kalau app di-restart.
"""

import logging
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor

from core.config import TMP_DIR

log = logging.getLogger("sipeta")

_latih = ThreadPoolExecutor(max_workers=1)
_antrean = {"latih": _latih, "latih_prediksi": _latih,
            "klasifikasi": ThreadPoolExecutor(max_workers=1), "prediksi": ThreadPoolExecutor(max_workers=1)}
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


class ProsesError(Exception):
    """Error yang pesannya boleh langsung ditampilkan ke pengguna. `fields` = pesan per field form."""

    def __init__(self, pesan: str, fields: dict[str, str] | None = None):
        super().__init__(pesan)
        self.message = pesan
        self.fields = fields or {}


def buat(jenis: str) -> str:
    job_id = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[job_id] = {"id": job_id, "jenis": jenis, "status": "antre", "progress": 0.0,
                         "label": "Menunggu giliran…", "hasil": None, "pesan": None}
    return job_id


def mulai(job_id: str, fungsi, *args) -> None:
    """Masukkan `fungsi(job_id, *args)` ke antrean. Nilai kembaliannya jadi hasil job."""
    _antrean[get(job_id)["jenis"]].submit(_jalankan, job_id, fungsi, args)


def _jalankan(job_id, fungsi, args):
    update(job_id, 0.0, "Memulai…", status="berjalan")
    try:
        hasil = fungsi(job_id, *args)   # hasil = latih_prediksi.jalankan(job_id, folder, kecamatan, params)
    except ProsesError as e:
        _set(job_id, status="gagal", pesan=str(e))
    except Exception as e:  # error tak terduga: catat lengkap di log
        log.error("Job %s gagal:\n%s", job_id, traceback.format_exc())
        _set(job_id, status="gagal", pesan=f"Terjadi error saat memproses: {e}")
    else:
        _set(job_id, status="selesai", progress=1.0, label="Selesai", hasil=hasil)


def _set(job_id, **nilai):
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(nilai)


def update(job_id: str, progress: float, label: str, status: str | None = None) -> None:
    nilai = {"progress": round(min(max(progress, 0.0), 1.0), 4), "label": label}
    if status:
        nilai["status"] = status
    _set(job_id, **nilai)


def get(job_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def hapus(job_id: str) -> None:
    """Buang job beserta berkas sementaranya"""
    with _lock:
        _jobs.pop(job_id, None)
    shutil.rmtree(TMP_DIR / job_id, ignore_errors=True)


def selesai(job_id: str) -> dict:
    """Job yang sudah selesai. Error kalau tidak ada (mis. app sudah di-restart) atau belum selesai."""
    job = get(job_id)
    if job is None:
        raise ProsesError("Hasil proses tidak ditemukan. Mungkin app sudah dijalankan ulang.")
    if job["status"] != "selesai":
        raise ProsesError("Proses belum selesai.")
    return job


def berkas(job_id: str, nama: str) -> bytes:
    """Isi berkas hasil job yang belum disimpan, mis. klasifikasi.png"""
    selesai(job_id)
    return (TMP_DIR / job_id / nama).read_bytes()


def tunggu(job_id: str, bar, jeda: float = 1.0):
    """
    Cek status job tiap `jeda` detik sambil update progress bar. Kalau selesai, hasil job-nya dikembalikan.
    """
    while True:
        job = get(job_id)
        if job is None:
            raise ProsesError("Proses tidak ditemukan. Mungkin app sudah dijalankan ulang.")
        bar.progress(job["progress"], text=job["label"])
        if job["status"] == "selesai":
            return job["hasil"]
        if job["status"] == "gagal":
            raise ProsesError(job["pesan"])
        time.sleep(jeda)
