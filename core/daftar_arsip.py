"""Mendaftarkan arsip Sentinel-2 ke tabel citra_arsip dan menentukan tanggal anchor"""

import numpy as np
import rasterio

from core import citra, db, prapemrosesan as pra
from core.config import ARSIP_DIR, DB_URL, FITUR, tujuan_db
from core.jobs import ProsesError
from core.latih import TANGGAL_RE


def hitung_awan(scl_path):
    """(piksel valid, piksel awan dari SCL, piksel awan setelah buffer) untuk satu citra kecamatan"""
    with rasterio.open(scl_path) as src:
        scl = src.read(1)
        res_x, res_y = abs(src.transform.a), abs(src.transform.e)
    awan, valid = pra.cloud_mask(scl, res_x, res_y)
    awan_scl = np.isin(scl, pra.CLOUD_RELATED_SCL) & valid
    return int(valid.sum()), int(awan_scl.sum()), int(awan.sum())


def persen(bagian, total):
    return round(100 * bagian / total, 2) if total else None


def daftarkan_wilayah(folder_wilayah, kecamatan_wilayah):
    # awan tiap kecamatan per tanggal
    baris = []
    for folder_kec in sorted(p for p in folder_wilayah.iterdir() if p.is_dir()):
        if folder_kec.name not in kecamatan_wilayah:
            print(f"[lewati] {folder_wilayah.name}/{folder_kec.name}: tidak ada di tabel kecamatan wilayah ini")
            continue
        for folder in sorted(p for p in folder_kec.iterdir() if p.is_dir() and TANGGAL_RE.fullmatch(p.name)):
            try:
                berkas = citra.index_folder(folder, FITUR)  # cek 7 band + SCL lengkap
            except ProsesError as e:
                print(f"{folder_kec.name}/{folder.name} | dilewati: {e}")
                continue
            valid, awan_scl, awan_buffer = hitung_awan(berkas["SCL"])
            baris.append({"kecamatan": folder_kec.name, "tanggal": folder.name, "valid": valid,
                          "awan_scl": awan_scl, "awan_buffer": awan_buffer,
                          "path": f"{folder_wilayah.name}/{folder_kec.name}/{folder.name}"})

    # awan gabungan satu wilayah per tanggal, dari sini ketahuan anchor atau bukan
    anchor = {}
    for tanggal in sorted({b["tanggal"] for b in baris}):
        sama = [b for b in baris if b["tanggal"] == tanggal]
        valid = sum(b["valid"] for b in sama)
        pct_scl = persen(sum(b["awan_scl"] for b in sama), valid)
        pct_buffer = persen(sum(b["awan_buffer"] for b in sama), valid)
        anchor[tanggal] = (pct_buffer, pct_buffer is not None and pct_buffer <= pra.ANCHOR_CLOUD_MAX)
        if pct_buffer is None:
            print(f"{tanggal} | tidak ada piksel valid")
        else:
            print(f"{tanggal} | {len(sama)} kecamatan | SCL={pct_scl:.2f}% | +buffer={pct_buffer:.2f}% | "
                  f"{'ANCHOR' if anchor[tanggal][1] else 'perlu gap-fill'}")

    # satu baris per kecamatan per tanggal
    for b in baris:
        pct_wilayah, is_anchor = anchor[b["tanggal"]]
        db.execute(
            "INSERT INTO citra_arsip (wilayah_id, kecamatan_id, tanggal, path_folder, cloud_pct_scl, "
            "cloud_pct_buffer, cloud_pct_buffer_wilayah, is_anchor) VALUES (%s, %s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (kecamatan_id, tanggal) DO UPDATE SET path_folder = EXCLUDED.path_folder, "
            "cloud_pct_scl = EXCLUDED.cloud_pct_scl, cloud_pct_buffer = EXCLUDED.cloud_pct_buffer, "
            "cloud_pct_buffer_wilayah = EXCLUDED.cloud_pct_buffer_wilayah, is_anchor = EXCLUDED.is_anchor",
            (folder_wilayah.name, b["kecamatan"], b["tanggal"], b["path"], persen(b["awan_scl"], b["valid"]),
             persen(b["awan_buffer"], b["valid"]), pct_wilayah, is_anchor),
        )
    jumlah_kec = len({b["kecamatan"] for b in baris})
    print(f"Total {folder_wilayah.name}: {jumlah_kec} kecamatan, {len(anchor)} tanggal, "
          f"{sum(a for _, a in anchor.values())} anchor")


def main():
    print(f"Database tujuan: {tujuan_db(DB_URL)}\nFolder arsip   : {ARSIP_DIR}\n")
    kolom = {r["column_name"] for r in db.fetch_all(
        "SELECT column_name FROM information_schema.columns WHERE table_name = 'citra_arsip'")}
    if not {"kecamatan_id", "cloud_pct_buffer_wilayah"} <= kolom:
        print("Tabel citra_arsip masih versi lama (belum ada kolom kecamatan_id / cloud_pct_buffer_wilayah).\n"
              "Buat ulang tabelnya dengan query citra_arsip di core/schema.sql, lalu jalankan skrip ini lagi.")
        return
    kecamatan = db.fetch_all("SELECT id, wilayah_id FROM kecamatan")
    wilayah = {k["wilayah_id"] for k in kecamatan}
    for folder_wilayah in sorted(p for p in ARSIP_DIR.iterdir() if p.is_dir()):
        if folder_wilayah.name not in wilayah:
            print(f"[lewati] {folder_wilayah.name}: tidak ada di tabel wilayah")
            continue
        print(f"\n== {folder_wilayah.name}")
        daftarkan_wilayah(folder_wilayah, {k["id"] for k in kecamatan if k["wilayah_id"] == folder_wilayah.name})


if __name__ == "__main__":
    main()
