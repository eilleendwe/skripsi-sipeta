"""
Latih model ConvLSTM.

Alurnya: peta tahunan -> sampel (L tahun -> tahun berikutnya) -> latih dengan validasi dan early
stopping -> latih ulang sampai tahun sebelum test, lalu uji (1 langkah dan rekursif) -> latih ulang
sampai 2025 sebagai model final yang dipakai halaman Prediksi.
"""

import math
import shutil
import time
from pathlib import Path

import numpy as np
import tensorflow as tf

from core import convlstm, jobs, peta_tahunan
from core.config import TAHUN_DATA
from core.jobs import ProsesError

SEQ_MAKS = 4               # tahun target selalu 2020-2025, berapa pun sequence length-nya
PATIENCE = 15
START_FROM_EPOCH = 20      # setelah 20 epoch, early stopping baru bekerja
RANDOM_STATE = 42
LABEL_KELAS = ["Nonterbangun", "Terbangun"]


class _Progres(tf.keras.callbacks.Callback):
    """Melaporkan epoch yang selesai ke status job"""

    def __init__(self, job_id, awal, akhir, label, total):
        super().__init__()
        self.job_id, self.awal, self.akhir, self.label, self.total = job_id, awal, akhir, label, total

    def on_epoch_end(self, epoch, logs=None):
        selesai = epoch + 1
        jobs.update(self.job_id, self.awal + (self.akhir - self.awal) * selesai / self.total,
                    f"{self.label} · epoch {selesai}/{self.total} · loss {logs['loss']:.4f}")


def split_years_by_ratio(years, train_ratio, val_ratio, test_ratio):
    """
    Potong daftar tahun target menurut urutan waktu. Val dan test masing-masing minimal 1 tahun kalau rasionya > 0.
    """
    n = len(years)
    tot = train_ratio + val_ratio + test_ratio
    n_test = max(1, round(n * test_ratio / tot)) if test_ratio > 0 else 0
    n_val = max(1, round(n * val_ratio / tot)) if val_ratio > 0 else 0
    n_train = n - n_val - n_test
    if n_train < 1:
        raise ProsesError("Rasio terlalu kecil untuk train: tidak ada tahun target yang tersisa.")
    return years[:n_train], years[n_train:n_train + n_val], years[n_train + n_val:]


def _peta_tahun(peta: dict, kec_ids: list[str], tahun: int) -> np.ndarray:
    """Peta 64x64 satu tahun untuk semua kecamatan -> (K, H, W)"""
    return np.stack([peta[k]["patch"][TAHUN_DATA.index(tahun)] for k in kec_ids])


def make_samples(peta: dict, kec_ids: list[str], target_years, seq_len):
    """
    Satu sampel = (seq_len tahun berurutan) -> (tahun berikutnya), urut per kecamatan lalu tahun.
    Return X (N, seq_len, H, W, 1), y (N, H, W, 1).
    """
    X, y = [], []
    for k in kec_ids:
        for ty in target_years:
            X.append(peta[k]["patch"][[TAHUN_DATA.index(t) for t in range(ty - seq_len, ty)]])
            y.append(peta[k]["patch"][TAHUN_DATA.index(ty)])
    return np.array(X, dtype=np.float32)[..., np.newaxis], np.array(y, dtype=np.float32)[..., np.newaxis]


def _model_baru(p: dict):
    tf.keras.utils.set_random_seed(RANDOM_STATE)
    return convlstm.build_convlstm_model(p["sequence_length"], peta_tahunan.PATCH_SIZE, p["filters"],
                                         p["kernel_size"], p["learning_rate"], n_layers=p["n_layers"],
                                         dropout=p["dropout"], optimizer=p["optimizer"])


def _latih_tanpa_validasi(job_id, p, peta, kec_ids, target_years, epochs, progres):
    """Latih dengan semua target yang diberikan, tanpa validasi, jumlah epoch tetap"""
    X, y = make_samples(peta, kec_ids, target_years, p["sequence_length"])
    model = _model_baru(p)
    model.fit(X, y, epochs=epochs, batch_size=p["batch_size"], verbose=0,
              callbacks=[_Progres(job_id, *progres, epochs)])
    return model


def _tanpa_nan(metrik: dict) -> dict:
    """NaN (mis. presisi tanpa prediksi positif) tidak bisa dikirim sebagai JSON"""
    return {k: None if isinstance(v, float) and math.isnan(v) else v for k, v in metrik.items()}


def jalankan(job_id: str, folder: Path, kecamatan_list: list[dict], p: dict) -> dict:
    """Data latih dari folder unggahan (folder/input/<kecamatan>/...) kalau ada, kalau tidak dari arsip"""
    try:
        return _latih(job_id, folder, kecamatan_list, p)
    finally:
        # Data unggahan dibuang, yang disimpan cuma model + konfigurasi.
        shutil.rmtree(folder / "input", ignore_errors=True)


def _latih(job_id, folder, kecamatan_list, p):
    t0 = time.time()
    unggahan = folder / "input"
    sumber = "unggah" if unggahan.is_dir() else "arsip"

    # --- Peta tahunan (cache per kecamatan) ------------------------------------
    peta, dilewati = {}, []
    for i, k in enumerate(kecamatan_list):
        jobs.update(job_id, 0.1 * i / len(kecamatan_list),
                    f"Menyiapkan peta tahunan {k['nama']} ({i + 1}/{len(kecamatan_list)})")
        if sumber == "unggah":
            cache = peta_tahunan.dari_folder(unggahan / k["id"])
        else:
            cache = peta_tahunan.muat(k["wilayah_id"], k["id"])
        if cache is None or cache["tahun"] != TAHUN_DATA:
            dilewati.append(k["nama"])  # kecamatan yang tahunnya tidak lengkap tidak dipakai
        else:
            peta[k["id"]] = cache
    if not peta:
        raise ProsesError(f"Tidak ada kecamatan yang punya peta klasifikasi lengkap "
                          f"{TAHUN_DATA[0]}–{TAHUN_DATA[-1]}.")
    # Diurutkan wilayah__kecamatan supaya urutan sampelnya tetap.
    kec_ids = [k["id"] for k in sorted(kecamatan_list, key=lambda k: (k["wilayah_id"], k["id"])) if k["id"] in peta]

    # --- Pembagian tahun target: train, validasi, test (urut waktu) -------------
    L = p["sequence_length"]
    kandidat = list(range(TAHUN_DATA[0] + SEQ_MAKS, TAHUN_DATA[-1] + 1))
    train_y, val_y, test_y = split_years_by_ratio(kandidat, p["train_ratio"], p["val_ratio"], p["test_ratio"])
    origin = test_y[0] - 1  # tahun terakhir yang "terlihat" saat uji

    # --- Latih dengan validasi untuk mencari epoch terbaik -----------------------
    X_tr, y_tr = make_samples(peta, kec_ids, train_y, L)
    X_va, y_va = make_samples(peta, kec_ids, val_y, L)
    jobs.update(job_id, 0.1, f"Latih dengan validasi · {len(X_tr)} sampel latih, {len(X_va)} sampel validasi")
    model = _model_baru(p)
    hist = model.fit(
        X_tr, y_tr, validation_data=(X_va, y_va), epochs=p["max_epochs"], batch_size=p["batch_size"], verbose=0,
        callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=PATIENCE,
                                                    restore_best_weights=True, start_from_epoch=START_FROM_EPOCH),
                   _Progres(job_id, 0.1, 0.5, "Latih dengan validasi", p["max_epochs"])],
    )
    if not np.isfinite(hist.history["loss"]).all():
        raise ProsesError("Loss menjadi NaN saat latih (pelatihan tidak stabil). Coba learning rate yang lebih kecil.")
    best_epoch = int(np.argmin(hist.history["val_loss"])) + 1
    epochs = max(5, best_epoch)
    loss = [{"epoch": i + 1, "latih": round(float(a), 5), "validasi": round(float(b), 5)}
            for i, (a, b) in enumerate(zip(hist.history["loss"], hist.history["val_loss"]))]

    # --- Uji: latih ulang sampai `origin`, prediksi tahun test secara rekursif ------
    test_model = _latih_tanpa_validasi(job_id, p, peta, kec_ids, list(range(TAHUN_DATA[0] + L, origin + 1)),
                                       epochs, (0.5, 0.72, "Latih ulang untuk uji"))
    jobs.update(job_id, 0.72, f"Uji tahun {', '.join(map(str, test_y))}")
    window = np.stack([_peta_tahun(peta, kec_ids, t) for t in range(origin - L + 1, origin + 1)], axis=1)
    probs, _ = convlstm.forecast_recursive(test_model, window[..., np.newaxis], len(test_y))
    before = _peta_tahun(peta, kec_ids, origin)
    uji = []
    for step, ty in enumerate(test_y):
        y_true = _peta_tahun(peta, kec_ids, ty)
        pred = probs[step] > convlstm.THRESHOLD
        uji.append({
            "tahun": ty,
            "langkah": step + 1,
            "metrik": _tanpa_nan(convlstm.evaluate_predictions(before, y_true, probs[step])),
            "confusion": convlstm.confusion_from_maps(before, y_true, pred),
        })

    # --- Model final: semua target sampai 2025, epoch sama -----------------------
    final = _latih_tanpa_validasi(job_id, p, peta, kec_ids, list(range(TAHUN_DATA[0] + L, TAHUN_DATA[-1] + 1)),
                                  epochs, (0.72, 0.98, "Latih model final"))
    final.save(folder / "model.keras")

    m = uji[0]["metrik"]
    tahun_split = {"train": train_y, "val": val_y, "test": test_y}
    return {
        "kecamatan_ids": kec_ids,
        "params": p,
        "config": {
            "model_file": None,  # diisi saat model disimpan
            "model_type": "ConvLSTMManual (Persamaan 2.39-2.44, tanpa peephole)",
            "framework": f"TensorFlow {tf.__version__}",
            "input": "peta klasifikasi 0/1 per tanggal (kecamatan_masked_rf_classified)",
            "data_source": "folder unggahan" if sumber == "unggah" else "arsip server",
            "annual_map": peta_tahunan.ATURAN,
            "training_kecamatan": kec_ids,
            "years": {"data": [TAHUN_DATA[0], TAHUN_DATA[-1]], **tahun_split},
            "hyperparameters": {**{k: p[k] for k in ("sequence_length", "n_layers", "filters", "kernel_size",
                                                     "dropout", "learning_rate", "optimizer", "batch_size")},
                                "epochs": epochs},
            "early_stopping": {"monitor": "val_loss", "patience": PATIENCE, "start_from_epoch": START_FROM_EPOCH,
                               "max_epochs": p["max_epochs"]},
            "threshold": convlstm.THRESHOLD,
            "forecast": {"last_observed_year": TAHUN_DATA[-1], "years": [TAHUN_DATA[-1] + 1, TAHUN_DATA[-1] + 2],
                         "mode": "rekursif"},
            "evaluation_test": {str(u["tahun"]): {k: u["metrik"][k] for k in ("akurasi", "precision", "recall", "f1")}
                                for u in uji},
        },
        "evaluasi": {
            # Ringkasan tahun test pertama (1 langkah), dipakai label dropdown model.
            "metrik": {"accuracy": m["akurasi"], "precision": m["precision"], "recall": m["recall"],
                       "f1": m["f1"], "iou": m["iou"], "fom": m["fom"]},
            "uji": uji,
            "loss": loss,
            "label_kelas": LABEL_KELAS,
            "tahun": tahun_split,
            "best_epoch": best_epoch,
            "epoch_jalan": len(hist.epoch),
            "epochs": epochs,
            "n_latih": int(len(X_tr)),
            "n_val": int(len(X_va)),
            "n_uji": len(kec_ids) * len(test_y),
            "kecamatan_dilewati": dilewati,
            "sumber": sumber,
            "waktu_latih_detik": round(time.time() - t0, 1),
        },
    }
