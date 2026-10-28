"""
Unggah arsip dari laptop ke bucket cloud (mode s3).

anchor  Cuma tanggal anchor (citra_arsip.is_anchor = true) dan cuma 7 band + SCL, karena hanya itu
        yang dibaca Klasifikasi Citra.
folder  Satu folder apa adanya, mis. arsip_klasifikasi.

Berkas yang sudah ada di bucket dengan ukuran sama dilewati.
"""

import argparse
import sys
from pathlib import Path

from core import citra, db, storage
from core.config import ARSIP_DIR, DB_URL, FITUR, S3_BUCKET, S3_ENDPOINT, STORAGE, tujuan_db
from core.jobs import ProsesError


def _mb(n: int) -> str:
    return f"{n / 1024 ** 2:,.1f} MB"


def _berkas_anchor(kecamatan: list[str] | None) -> list[Path]:
    rows = db.fetch_all("SELECT wilayah_id, kecamatan_id, tanggal, path_folder FROM citra_arsip "
                        "WHERE is_anchor ORDER BY wilayah_id, kecamatan_id, tanggal")
    if kecamatan:
        tidak_ada = sorted(set(kecamatan) - {r["kecamatan_id"] for r in rows})
        if tidak_ada:
            print(f"[peringatan] tidak punya anchor di database: {', '.join(tidak_ada)}")
        rows = [r for r in rows if r["kecamatan_id"] in kecamatan]
    if not rows:
        sys.exit("Tidak ada anchor di tabel citra_arsip. Jalankan dulu: python -m core.daftar_arsip")

    berkas, per_wilayah = [], {}
    for r in rows:
        try:
            dipakai = citra.index_folder(ARSIP_DIR / r["path_folder"], FITUR)
        except ProsesError as e:
            print(f"[lewati] {r['path_folder']}: {e}")
            continue
        berkas += dipakai.values()
        w = per_wilayah.setdefault(r["wilayah_id"], {"tanggal": set(), "kecamatan": set(), "byte": 0})
        w["tanggal"].add(r["tanggal"])
        w["kecamatan"].add(r["kecamatan_id"])
        w["byte"] += sum(p.stat().st_size for p in dipakai.values())

    for wilayah, w in per_wilayah.items():
        print(f"{wilayah}: {len(w['tanggal'])} tanggal anchor, {len(w['kecamatan'])} kecamatan, {_mb(w['byte'])}")
    return berkas


def _unggah(berkas: list[Path], prefix: str) -> None:
    if not storage.S3:
        sys.exit("SIPETA_STORAGE_MODE bukan 's3'. Atur pengaturan bucket dulu (lihat PANDUAN_DEPLOY.md).")
    storage.periksa()
    sudah = dict(storage._objek(prefix))
    total = len(berkas)
    for i, p in enumerate(berkas, 1):
        kunci = storage.relatif(p)
        if sudah.get(kunci) == p.stat().st_size:
            continue
        print(f"[{i}/{total}] {kunci} ({_mb(p.stat().st_size)})")
        storage._klien().upload_file(str(p), storage.S3_BUCKET, kunci)
    print("Selesai.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="perintah", required=True)
    p_anchor = sub.add_parser("anchor", help="unggah tanggal anchor (7 band fitur + SCL)")
    p_anchor.add_argument("--ukur", action="store_true", help="hitung ukuran saja, tanpa mengunggah")
    p_anchor.add_argument("--kecamatan", nargs="+", metavar="ID", help="hanya kecamatan ini (id, mis. pondok_aren)")
    p_folder = sub.add_parser("folder", help="unggah satu subfolder storage/ apa adanya")
    p_folder.add_argument("nama", help="nama subfolder, mis. arsip_klasifikasi")
    p_folder.add_argument("--ukur", action="store_true", help="hitung ukuran saja, tanpa mengunggah")
    args = parser.parse_args()
    print(f"Database tujuan: {tujuan_db(DB_URL)}\nFolder storage : {STORAGE}\n"
          f"Bucket         : {f'{S3_BUCKET} @ {S3_ENDPOINT}' if storage.S3 else '(mode lokal, tidak mengunggah)'}\n")

    if args.perintah == "anchor":
        berkas, prefix = _berkas_anchor(args.kecamatan), "arsip/"
    else:
        folder = STORAGE / args.nama
        if not folder.is_dir():
            sys.exit(f"Folder {folder} tidak ada.")
        berkas, prefix = [p for p in sorted(folder.rglob("*")) if p.is_file()], f"{args.nama}/"
    print(f"Total: {len(berkas)} berkas, {_mb(sum(p.stat().st_size for p in berkas))}")
    if not args.ukur:
        _unggah(berkas, prefix)


if __name__ == "__main__":
    main()
