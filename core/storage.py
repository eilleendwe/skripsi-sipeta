"""
Penyimpanan berkas permanen (model, hasil, citra masukan, arsip).

Semua proses tetap baca/tulis di folder STORAGE, karena rasterio, pickle, dan Keras butuh path lokal.
- Mode "lokal": folder STORAGE satu-satunya tempat penyimpanan.
- Mode "s3": berkas permanen ada di bucket, dengan kunci = path relatif terhadap STORAGE (sama dengan
  kolom path_* di database). Folder STORAGE cuma cache; berkas yang belum ada diunduh saat dibutuhkan.

    ambil(rel)         path lokal satu berkas, diunduh dulu kalau belum ada
    ambil_folder(rel)  path lokal satu folder, isinya diunduh kalau belum ada
    kirim(rel)         unggah berkas/folder lokal ke bucket
    hapus(rel)         hapus berkas/folder di lokal dan di bucket
    daftar(rel)        path relatif semua berkas di dalam folder
"""

import os
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from core.config import (S3_BUCKET, S3_ENDPOINT, S3_KEY, S3_REGION, S3_SECRET, STORAGE,
                         STORAGE_MODE)

S3 = STORAGE_MODE == "s3"


def periksa() -> None:
    """
    Cek pengaturan storage lengkap. Dipanggil sekali saat app start.
    """
    if STORAGE_MODE not in ("lokal", "s3"):
        raise RuntimeError(f"SIPETA_STORAGE_MODE harus 'lokal' atau 's3', bukan '{STORAGE_MODE}'.")
    if S3:
        kurang = [n for n, v in (("SIPETA_S3_ENDPOINT", S3_ENDPOINT), ("SIPETA_S3_BUCKET", S3_BUCKET),
                                 ("SIPETA_S3_KEY", S3_KEY), ("SIPETA_S3_SECRET", S3_SECRET)) if not v]
        if kurang:
            raise RuntimeError(f"Mode storage s3 butuh pengaturan: {', '.join(kurang)}.")


def lokal(rel: str) -> Path:
    return STORAGE / rel


def relatif(path: Path) -> str:
    """Path lokal -> path relatif terhadap STORAGE (bentuk yang disimpan di database)"""
    return Path(path).relative_to(STORAGE).as_posix()


@lru_cache(maxsize=1)
def _klien():
    import boto3  # cuma dipakai di mode s3
    from botocore.config import Config
    # Path-style addressing didukung semua layanan S3-compatible. Checksum bawaan botocore versi baru
    # belum didukung semuanya, jadi hanya dipakai kalau diwajibkan.
    config = Config(s3={"addressing_style": "path"}, request_checksum_calculation="when_required",
                    response_checksum_validation="when_required", retries={"max_attempts": 5, "mode": "standard"})
    return boto3.client("s3", endpoint_url=S3_ENDPOINT, aws_access_key_id=S3_KEY,
                        aws_secret_access_key=S3_SECRET, region_name=S3_REGION, config=config)


def _objek(prefix: str) -> list[tuple[str, int]]:
    """(kunci, ukuran) semua objek di bucket yang diawali `prefix`"""
    hasil = []
    for halaman in _klien().get_paginator("list_objects_v2").paginate(Bucket=S3_BUCKET, Prefix=prefix):
        hasil += [(o["Key"], o["Size"]) for o in halaman.get("Contents", [])]
    return hasil


def _unduh(kunci: str, tujuan: Path) -> None:
    # Tulis ke nama sementara dulu, supaya thread lain tidak membaca berkas yang belum selesai.
    tujuan.parent.mkdir(parents=True, exist_ok=True)
    sementara = tujuan.with_name(f".{tujuan.name}.{uuid.uuid4().hex[:8]}")
    _klien().download_file(S3_BUCKET, kunci, str(sementara))
    os.replace(sementara, tujuan)


def ambil(rel: str) -> Path:
    path = lokal(rel)
    if S3 and not path.exists():
        from botocore.exceptions import ClientError
        try:
            _unduh(rel, path)
        except ClientError as e:
            raise FileNotFoundError(f"Berkas {rel} tidak ada di storage.") from e
    return path


def ambil_folder(rel: str) -> Path:
    folder = lokal(rel)
    if S3:
        for kunci, ukuran in _objek(rel.rstrip("/") + "/"):
            path = lokal(kunci)
            if not path.exists() or path.stat().st_size != ukuran:
                _unduh(kunci, path)
    return folder


def kirim(rel: str) -> None:
    if not S3:
        return
    path = lokal(rel)
    berkas = [p for p in path.rglob("*") if p.is_file()] if path.is_dir() else [path]
    for p in berkas:
        _klien().upload_file(str(p), S3_BUCKET, relatif(p))


def hapus(rel: str) -> None:
    path = lokal(rel)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    else:
        path.unlink(missing_ok=True)
    if S3:
        kunci = [k for k, _ in _objek(rel.rstrip("/") + "/")] + [rel]
        for i in range(0, len(kunci), 1000):  # delete_objects maksimal 1000 kunci sekali panggil
            _klien().delete_objects(Bucket=S3_BUCKET, Delete={"Objects": [{"Key": k} for k in kunci[i:i + 1000]],
                                                             "Quiet": True})


def daftar(rel: str) -> list[str]:
    if S3:
        return sorted(k for k, _ in _objek(rel.rstrip("/") + "/"))
    folder = lokal(rel)
    return sorted(relatif(p) for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
