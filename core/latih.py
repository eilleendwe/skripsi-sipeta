"""
Latih model Random Forest untuk satu kecamatan.
Folder masukan berisi satu subfolder per tanggal (YYYY-MM-DD), masing-masing 9 band + SCL.
Alurnya: cloud mask -> label NDBI -> sampel piksel cerah -> split blok spasial -> Random Forest -> evaluasi
"""

import pickle
import re
import shutil
import time
from pathlib import Path

import numpy as np

from core import citra, jobs, prapemrosesan as pra, rf
from core.config import BANDS_LATIH, FITUR, TANGGAL_MAKS, TANGGAL_MIN
from core.jobs import ProsesError

TANGGAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
UKURAN_BLOK = 50     # sisi blok spasial untuk split latih/uji, dalam piksel
RANDOM_SEED = 42


def _konfigurasi(kecamatan_id: str, tanggal: list[str], params: dict, res_m: float) -> dict:
    """Konfigurasi yang disimpan bersama model, dipakai lagi saat klasifikasi"""
    return {
        "model_file": None,
        "preprocessing_version": pra.PROCESS_VERSION,
        "scl_cloud_classes": pra.CLOUD_RELATED_SCL,
        "cloud_buffer_m": pra.CLOUD_BUFFER_METERS,
        "anchor_cloud_max_pct": pra.ANCHOR_CLOUD_MAX,
        "gap_fill": "pchip_2before_2after",
        "resolution_m": res_m,
        "features": FITUR,
        "label_rule": "NDBI = (B11-B08)/(B11+B08) > 0",
        "training_kecamatan": [kecamatan_id],
        "training_dates": tanggal,
        "sampling": {"n_per_date": params["sampel_per_tanggal"] or "semua", "source": "cerah",
                     "seed": RANDOM_SEED},
        "hyperparameters": {**{k: params[k] for k in
                               ("n_trees", "max_depth", "min_samples_split", "n_features", "max_thresholds")},
                            "bootstrap": True},
        "evaluation": {"split": "blok_spasial", "block_size_px": UKURAN_BLOK,
                       "test_fraction": params["test_ratio"]},
    }


def jalankan(job_id: str, folder: Path, kecamatan: dict, params: dict) -> dict:
    try:
        return _latih(job_id, folder / "input", folder, kecamatan, params)
    finally:
        # Citra latih dibuang setelah selesai; yang disimpan cuma model + konfigurasi.
        shutil.rmtree(folder / "input", ignore_errors=True)


def _latih(job_id, folder_input, folder_job, kecamatan, params):
    # --- Cek masukan ----------------------------------------------------------
    jobs.update(job_id, 0.01, "Memvalidasi citra latih")
    tanggal_list = sorted(d.name for d in folder_input.iterdir() if d.is_dir() and TANGGAL_RE.fullmatch(d.name))
    if not tanggal_list:
        raise ProsesError("Tidak ada subfolder tanggal (YYYY-MM-DD) di folder citra.")
    di_luar = [t for t in tanggal_list if not TANGGAL_MIN <= t <= TANGGAL_MAKS]
    if di_luar:
        raise ProsesError(f"Tanggal di luar rentang arsip {TANGGAL_MIN[:4]}–{TANGGAL_MAKS[:4]}: {', '.join(di_luar)}.")

    berkas = {t: citra.index_folder(folder_input / t, BANDS_LATIH) for t in tanggal_list}
    acuan = citra.periksa_grid([p for b in berkas.values() for p in b.values()])
    res_x, res_y = acuan["res"]
    lebar = acuan["width"]

    rng = np.random.default_rng(RANDOM_SEED)
    n_blok_x = -(-lebar // UKURAN_BLOK)
    n_sampel = params["sampel_per_tanggal"]  # None = semua piksel cerah

    # --- Cloud mask, label NDBI, dan sampel piksel cerah per tanggal ----------
    X_list, y_list, blok_list = [], [], []
    for i, tanggal in enumerate(tanggal_list):
        jobs.update(job_id, 0.03 + 0.32 * i / len(tanggal_list),
                    f"Prapemrosesan {tanggal} ({i + 1}/{len(tanggal_list)}) · cloud masking · label NDBI")
        b = berkas[tanggal]
        awan, valid = pra.cloud_mask(citra.baca(b["SCL"]), res_x, res_y)
        data = {band: citra.baca(b[band]) for band in BANDS_LATIH}

        b08 = data["B08"].astype(np.float32)
        b11 = data["B11"].astype(np.float32)
        idx = np.flatnonzero(valid & ~awan & ((b11 + b08) != 0))
        if idx.size == 0:
            continue

        pilih = idx if n_sampel is None else rng.choice(idx, size=min(n_sampel, idx.size), replace=False)
        b08, b11 = b08.reshape(-1)[pilih], b11.reshape(-1)[pilih]
        ndbi = (b11 - b08) / (b11 + b08)
        baris, kolom = np.divmod(pilih, lebar)

        X_list.append(np.stack([data[f].reshape(-1)[pilih] for f in FITUR], axis=1).astype(np.float32))
        y_list.append((ndbi > 0).astype(np.int64))  # label dari NDBI: 1 = terbangun
        blok_list.append((baris // UKURAN_BLOK) * n_blok_x + kolom // UKURAN_BLOK)

    if not X_list:
        raise ProsesError("Tidak ada piksel valid untuk dijadikan sampel latih.")
    X, y, blok = np.concatenate(X_list), np.concatenate(y_list), np.concatenate(blok_list)

    # --- Split blok spasial: semua tanggal dari satu blok masuk ke sisi yang sama -------
    blok_unik = rng.permutation(np.unique(blok))
    if len(blok_unik) < 5:
        raise ProsesError(f"Hanya ada {len(blok_unik)} blok spasial. Perkecil ukuran blok.")
    n_blok_uji = max(1, int(round(len(blok_unik) * params["test_ratio"])))
    uji = np.isin(blok, blok_unik[:n_blok_uji])
    X_latih, y_latih, X_uji, y_uji = X[~uji], y[~uji], X[uji], y[uji]
    if len(np.unique(y_latih)) < 2:
        raise ProsesError("Data latih hanya berisi satu kelas, model tidak bisa dilatih.")

    # --- Latih Random Forest -------------------------------------------------
    np.random.seed(RANDOM_SEED) 
    model = rf.RandomForest(
        n_trees=params["n_trees"],
        max_depth=params["max_depth"],
        min_samples_split=params["min_samples_split"],
        n_features=params["n_features"],
        max_thresholds=params["max_thresholds"],
        bootstrap=True,
    )
    t0 = time.time()

    def progres(selesai, total):
        jobs.update(job_id, 0.35 + 0.6 * selesai / total, f"Melatih Random Forest · pohon {selesai}/{total}")

    jobs.update(job_id, 0.35, f"Melatih Random Forest · {len(y_latih):,} sampel".replace(",", "."))
    model.fit(X_latih, y_latih, progress=progres)
    waktu_latih = time.time() - t0

    # --- Evaluasi pada blok uji ------------------------------------------------
    jobs.update(job_id, 0.96, "Evaluasi pada blok uji")
    hasil = rf.evaluasi(y_uji, model.predict(X_uji))

    with open(folder_job / "model.pkl", "wb") as f:
        pickle.dump(model, f)

    return {
        "kecamatan_id": kecamatan["id"],
        "params": params,
        "tanggal_citra": tanggal_list,
        "config": _konfigurasi(kecamatan["id"], tanggal_list, params, res_x),
        "evaluasi": {
            **hasil,
            "label_kelas": ["Nonterbangun", "Terbangun"],
            "n_latih": int(len(y_latih)),
            "n_uji": int(len(y_uji)),
            "blok_latih": int(len(blok_unik) - n_blok_uji),
            "blok_uji": int(n_blok_uji),
            "jumlah_tanggal": len(tanggal_list),
            "persen_terbangun_acuan": round(100 * float(y.mean()), 2),
            "waktu_latih_detik": round(waktu_latih, 1),
        },
    }
