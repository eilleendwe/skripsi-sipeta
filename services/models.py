"""
Model tersimpan dan pelatihan model.

Random Forest (Klasifikasi) dan ConvLSTM (Prediksi) sama-sama dilatih sebagai job latar belakang.
"""

import json
import re
import shutil
from datetime import date
from pathlib import Path, PurePosixPath

import streamlit as st

from core import db, jobs, latih, storage
from core.config import MODEL_DIR, TMP_DIR
from services import arsip, auth, regions, roles, unggahan
from services.errors import ServiceError

RF = "rf"
CONVLSTM = "convlstm"

OPTIMIZER = {"adam": "Adam", "adamw": "AdamW", "nadam": "Nadam", "rmsprop": "RMSprop", "sgd": "SGD"}
SUMBER_ARSIP, SUMBER_UNGGAH = "arsip", "unggah"
SUMBER_DATA = {SUMBER_ARSIP: "Arsip di server", SUMBER_UNGGAH: "Unggah folder"}  # data latih ConvLSTM

BANDS_LATIH = ["B02", "B03", "B04", "B05", "B06", "B07", "B08", "B11", "B12", "SCL"]
TANGGAL_MIN, TANGGAL_MAKS = date(2016, 1, 1), date(2025, 12, 31)
_BAND_RE = re.compile(r"(?<![A-Za-z0-9])(B0[2-8]|B1[12]|SCL)(?![A-Za-z0-9])", re.IGNORECASE)
_TANGGAL_RE = re.compile(r"\d{4}-\d{2}-\d{2}")


def default_params(algoritma: str) -> dict:
    if algoritma == RF:
        # Nilai awal hasil grid search (384.000 sampel dari 40 kecamatan).
        return {
            "n_trees": 30,
            "max_depth": 16,
            "min_samples_split": 10,
            "n_features": 3,
            "max_thresholds": 20,
            "train_ratio": 0.8,
            "test_ratio": 0.2,
            "sampel_per_tanggal": 1000,  # None = semua piksel cerah
        }
    # Konfigurasi terbaik hasil grid search.
    return {
        "sequence_length": 4,
        "n_layers": 2,
        "filters": 40,
        "kernel_size": 5,
        "dropout": 0.0,
        "learning_rate": 0.0005,
        "optimizer": "adam",
        "max_epochs": 100,
        "batch_size": 8,
        "train_ratio": 0.5,
        "val_ratio": 0.17,
        "test_ratio": 0.33,
    }


# Keterangan di bawah label hyperparameter di form Latih Model: (penjelasan singkat, nilai rekomendasi).
REKOMENDASI = {
    RF: {
        "n_trees": ("Banyaknya pohon keputusan dalam model", "30"),
        "max_depth": ("Kedalaman maksimum tiap pohon", "16"),
        "min_samples_split": ("Sampel minimum untuk memecah simpul", "10"),
        "n_features": ("Band yang dicoba di tiap percabangan", "3"),
        "max_thresholds": ("Batas nilai yang dicoba per band", "20"),
        "train_ratio": ("Porsi blok spasial untuk melatih model", "0,8"),
        "test_ratio": ("Porsi blok spasial untuk menguji model", "0,2"),
        "sampel_per_tanggal": ("Piksel acak yang diambil dari tiap tanggal", "1.000"),
    },
    CONVLSTM: {
        "sequence_length": ("Jumlah peta tahunan berurutan sebagai masukan", "4"),
        "n_layers": ("Banyaknya layer ConvLSTM", "2"),
        "filters": ("Jumlah filter di tiap layer ConvLSTM", "40"),
        "kernel_size": ("Ukuran jendela konvolusi, dalam sel 64×64", "5"),
        "dropout": ("Porsi masukan yang dinolkan acak saat latih", "0"),
        "learning_rate": ("Besar langkah pembaruan bobot", "0,0005"),
        "optimizer": ("Algoritma pembaruan bobot", "Adam"),
        "max_epochs": ("Batas epoch; early stopping bisa berhenti lebih awal", "100"),
        "batch_size": ("Jumlah sampel per langkah latih", "8"),
        "train_ratio": ("Porsi tahun target untuk melatih", "0,5"),
        "val_ratio": ("Porsi tahun target untuk memilih epoch", "0,17"),
        "test_ratio": ("Porsi tahun target untuk menguji", "0,33"),
    },
}


# --- Baca -------------------------------------------------------------------

# Tabel model per algoritma: (query kolom, syarat "model mencakup kecamatan X")
_TABEL = {
    RF: ("SELECT id, nama, kecamatan_id, evaluasi, hyperparameter, dibuat_oleh, dibuat_pada FROM model_klasifikasi ",
         "kecamatan_id = %s"),
    CONVLSTM: ("SELECT id, nama, kecamatan_ids, evaluasi, hyperparameter, dibuat_oleh, dibuat_pada FROM model_prediksi ",
               "%s = ANY(kecamatan_ids)"),
}


def _dari_baris(algoritma: str, row: dict) -> dict:
    """Baris tabel model_klasifikasi / model_prediksi -> dict model yang dipakai halaman"""
    return {
        "id": row["id"],
        "nama": row["nama"],
        "algoritma": algoritma,
        "wilayah_id": None,
        "kecamatan_ids": [row["kecamatan_id"]] if algoritma == RF else row["kecamatan_ids"],
        "evaluasi": row["evaluasi"],
        "params": row["hyperparameter"],
        "tanggal_latih": row["dibuat_pada"].date(),
        "dibuat_oleh": row["dibuat_oleh"],
    }


def covers(model: dict, kecamatan_id: str) -> bool:
    return kecamatan_id in model["kecamatan_ids"]


def list_models(algoritma: str, kecamatan_id: str | None = None) -> list[dict]:
    """
    Model tersimpan, terbaru dulu. Kalau `kecamatan_id` diisi, cuma model yang mencakup kecamatan itu.
    """
    return [_dari_baris(algoritma, r) for r in _daftar(algoritma, kecamatan_id)]


@st.cache_data(ttl=300, show_spinner=False)
def _daftar(algoritma: str, kecamatan_id: str | None) -> list[dict]:
    sql, cakupan = _TABEL[algoritma]
    if kecamatan_id:
        return db.fetch_all(sql + f"WHERE {cakupan} ORDER BY dibuat_pada DESC", (kecamatan_id,))
    return db.fetch_all(sql + "ORDER BY dibuat_pada DESC")


def get(algoritma: str, model_id: int | None) -> dict | None:
    if model_id is None:
        return None
    row = db.fetch_one(_TABEL[algoritma][0] + "WHERE id = %s", (model_id,))
    return _dari_baris(algoritma, row) if row else None


def coverage_label(model: dict) -> str:
    """Ringkasan cakupan kecamatan sebuah model"""
    ids = model["kecamatan_ids"]
    if len(ids) == len(regions.list_kecamatan()):
        return "semua kecamatan"
    if model.get("wilayah_id"):
        wilayah = regions.get_wilayah(model["wilayah_id"])
        if wilayah and len(ids) == len(regions.list_kecamatan(model["wilayah_id"])):
            return f"{wilayah['nama']} ({len(ids)} kecamatan)"
    if len(ids) <= 2:
        return ", ".join(regions.nama_kecamatan(i) for i in ids)
    return f"{len(ids)} kecamatan"


# --- Folder citra latih (Random Forest) ----------------------------------------

def scan_folder(nama_berkas: list[str], kecamatan_id: str | None) -> dict:
    """
    Cek struktur folder citra latih dari nama file.
    Struktur yang diharapkan: <kecamatan>/<YYYY-MM-DD>/<...>_B02_10m.tif (9 band + SCL).
    """
    akar, per_tanggal, diabaikan = set(), {}, []
    for nama in nama_berkas:
        bagian = PurePosixPath(nama.replace("\\", "/")).parts
        akar.add(bagian[0] if len(bagian) > 1 else "")
        cocok = _BAND_RE.search(bagian[-1])
        if len(bagian) < 3 or not _TANGGAL_RE.fullmatch(bagian[-2]) or not cocok:
            diabaikan.append(nama)
            continue
        per_tanggal.setdefault(bagian[-2], {}).setdefault(cocok.group(1).upper(), []).append(nama)

    baris, berkas_valid = [], []
    for tanggal, bands in sorted(per_tanggal.items()):
        kurang = [b for b in BANDS_LATIH if b not in bands]
        ganda = [b for b, daftar in bands.items() if len(daftar) > 1]
        try:
            t = date.fromisoformat(tanggal)
        except ValueError:
            t = None
        if t is None:
            status, ket = "gagal", "nama folder bukan tanggal yang valid"
        elif not TANGGAL_MIN <= t <= TANGGAL_MAKS:
            status, ket = "gagal", f"di luar rentang arsip {TANGGAL_MIN.year}–{TANGGAL_MAKS.year}"
        elif kurang:
            status, ket = "gagal", f"{', '.join(kurang)} tidak ada"
        elif ganda:
            status, ket = "gagal", f"{', '.join(ganda)} ganda"
        else:
            status, ket = "valid", "9 band + SCL"
            berkas_valid += [daftar[0] for b, daftar in bands.items() if b in BANDS_LATIH]
        baris.append({"tanggal": tanggal, "band": ket, "status": status})

    nama_akar = next(iter(akar)) if len(akar) == 1 else None
    kecamatan = regions.get_kecamatan(kecamatan_id)
    akar_cocok = bool(nama_akar and kecamatan and (regions.nama_sama(nama_akar, kecamatan["id"])
                                                    or regions.nama_sama(nama_akar, kecamatan["nama"])))
    return {
        "akar": nama_akar,
        "akar_cocok": akar_cocok,
        "baris": baris,
        "jumlah_valid": sum(b["status"] == "valid" for b in baris),
        "berkas_valid": berkas_valid,
        "diabaikan": diabaikan,
    }


# --- Pembagian tahun (ConvLSTM) -----------------------------------------------------

def pembagian_tahun(p: dict) -> dict[str, list[int]] | None:
    """
    Bagi tahun target 2020–2025 urut waktu sesuai rasio. None kalau tidak ada tahun tersisa
    untuk train.
    """
    tahun = list(range(arsip.TAHUN_DATA[0] + 4, arsip.TAHUN_DATA[-1] + 1))
    total = p["train_ratio"] + p["val_ratio"] + p["test_ratio"]
    n_test = max(1, round(len(tahun) * p["test_ratio"] / total))
    n_val = max(1, round(len(tahun) * p["val_ratio"] / total))
    n_train = len(tahun) - n_val - n_test
    if n_train < 1:
        return None
    return {"train": tahun[:n_train], "val": tahun[n_train:n_train + n_val], "test": tahun[n_train + n_val:]}


# --- Validasi ---------------------------------------------------------------

def _name_taken(algoritma: str, nama: str) -> bool:
    kunci = nama.strip().lower()
    return any(m["nama"].lower() == kunci for m in list_models(algoritma))


def validate(algoritma: str, spec: dict) -> dict[str, str]:
    """
    Pesan error per field (kunci = nama field). Dict kosong berarti valid.
    """
    errors: dict[str, str] = {}
    p = spec["params"]

    nama = (spec.get("nama") or "").strip()
    if not nama:
        errors["nama"] = "Nama model wajib diisi."
    elif _name_taken(algoritma, nama):
        errors["nama"] = "Nama model sudah dipakai."

    if algoritma == RF:
        if not spec.get("kecamatan_id"):
            errors["kecamatan"] = "Pilih kecamatan."
        laporan = scan_folder([f.name for f in spec.get("berkas") or []], spec.get("kecamatan_id"))
        if not spec.get("berkas"):
            errors["folder"] = "Unggah folder citra kecamatan."
        elif spec.get("kecamatan_id") and not laporan["akar_cocok"]:
            errors["folder"] = "Nama folder harus sama dengan kecamatan yang dipilih."
        elif laporan["jumlah_valid"] == 0:
            errors["folder"] = "Tidak ada subfolder tanggal yang lengkap (9 band + SCL)."
        if abs(p["train_ratio"] + p["test_ratio"] - 1) > 1e-6:
            errors["rasio"] = "Jumlah train ratio dan test ratio harus 1."
        return errors

    if spec.get("sumber") == SUMBER_UNGGAH:
        laporan = arsip.scan_folder([f.name for f in spec.get("berkas") or []])
        if not spec.get("berkas"):
            errors["folder"] = "Unggah folder peta klasifikasi."
        elif not any(b["lengkap"] for b in laporan["baris"]):
            errors["folder"] = (f"Tidak ada kecamatan di folder yang punya peta lengkap "
                               f"{arsip.TAHUN_DATA[0]}–{arsip.TAHUN_DATA[-1]}.")
    elif not spec.get("kecamatan_ids"):
        errors["kecamatan"] = "Pilih minimal satu kecamatan."
    elif not arsip.lengkap(spec["kecamatan_ids"]):
        errors["kecamatan"] = (f"Belum ada kecamatan terpilih yang punya arsip klasifikasi lengkap "
                              f"{arsip.TAHUN_DATA[0]}–{arsip.TAHUN_DATA[-1]}.")
    if min(p["train_ratio"], p["test_ratio"], p["val_ratio"]) <= 0:
        errors["rasio"] = "Train, validation, dan test ratio harus lebih dari 0."
    elif abs(p["train_ratio"] + p["test_ratio"] + p["val_ratio"] - 1) > 1e-6:
        errors["rasio"] = "Jumlah train, validation, dan test ratio harus 1."
    if p["kernel_size"] % 2 == 0:
        errors["kernel_size"] = "Kernel size harus ganjil."
    return errors


def _require_admin() -> dict:
    user = auth.current_user()
    if user is None or not roles.has_role(user["role"], roles.ADMIN):
        raise ServiceError("Hanya Admin yang dapat melatih dan menyimpan model.")
    return user


# --- Latih dan simpan -------------------------------------------------------

def mulai_latih(algoritma: str, spec: dict, bar) -> str:
    """
    Kirim data latih dan mulai job pelatihan. Yang dikembalikan job_id-nya.
    """
    _require_admin()
    errors = validate(algoritma, spec)
    if errors:
        raise ServiceError("Periksa kembali isian form.", errors)
    if algoritma == RF:
        files = _berkas_rf(spec)
        bar.progress(0.0, text=f"Menyiapkan {len(files)} berkas citra…")
        params = {k: v for k, v in spec["params"].items() if k != "train_ratio"}
        job_id = jobs.buat("latih")
        folder = TMP_DIR / job_id
        if unggahan.simpan_band(files, folder / "input", pakai_subfolder_tanggal=True) == 0:
            jobs.hapus(job_id)
            raise ServiceError("Tidak ada berkas citra yang dikenali di folder unggahan.")
        jobs.mulai(job_id, latih.jalankan, folder, regions.get_kecamatan(spec["kecamatan_id"]), params)
        return job_id

    from core import latih_prediksi
    job_id = jobs.buat("latih_prediksi")
    folder = TMP_DIR / job_id
    folder.mkdir(parents=True)
    if spec.get("sumber") == SUMBER_UNGGAH:
        # Cuma kecamatan yang lengkap 2016–2025 yang dipakai. Petanya ditaruh di folder job, tidak masuk arsip.
        laporan = arsip.scan_folder([f.name for f in spec["berkas"]])
        kecamatan = [regions.get_kecamatan(b["kecamatan_id"]) for b in laporan["baris"] if b["lengkap"]]
        bar.progress(0.0, text=f"Memeriksa peta klasifikasi {len(kecamatan)} kecamatan…")
        try:
            for k in kecamatan:
                for tanggal, isi in arsip.periksa_peta(k, arsip.kemas_berkas(laporan, spec["berkas"], k["id"])).items():
                    tujuan = folder / "input" / k["id"] / tanggal
                    tujuan.mkdir(parents=True)
                    (tujuan / f"klasifikasi_{tanggal}.tif").write_bytes(isi)
        except ServiceError:
            jobs.hapus(job_id)
            raise
    else:
        kecamatan = [regions.get_kecamatan(k) for k in spec["kecamatan_ids"]]
    # panggil fungsi core/latih_prediksi
    jobs.mulai(job_id, latih_prediksi.jalankan, folder, kecamatan, spec["params"])
    return job_id


def tunggu(job_id: str, bar) -> dict:
    """Tunggu job latih selesai sambil update progress bar, lalu kembalikan hasilnya"""
    return jobs.tunggu(job_id, bar)


def buat_draft(algoritma: str, spec: dict, job_id: str, hasil: dict) -> dict:
    """Calon model dari job latih yang sudah selesai (belum disimpan)"""
    draft = {
        "nama": spec["nama"].strip(),
        "algoritma": algoritma,
        "job_id": job_id,
        "wilayah_id": spec.get("wilayah_id"),
        "params": spec["params"],
        "evaluasi": hasil["evaluasi"],
    }
    if algoritma == RF:
        draft["kecamatan_ids"] = [spec["kecamatan_id"]]
    else:
        draft["kecamatan_ids"] = hasil["kecamatan_ids"]
    return draft


def _berkas_rf(spec: dict) -> list:
    """
    Berkas valid dari folder unggahan sebagai (tanggal/nama berkas, berkas). Objek berkasnya
    diteruskan apa adanya (bukan getvalue()), supaya folder ratusan MB tidak disalin lagi di memori.
    """
    dipakai = set(scan_folder([f.name for f in spec["berkas"]], spec["kecamatan_id"])["berkas_valid"])
    return [
        ("/".join(PurePosixPath(f.name.replace("\\", "/")).parts[-2:]), f)
        for f in spec["berkas"] if f.name in dipakai
    ]


def _slug(teks: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", teks.lower()).strip("_") or "model"


def _simpan_berkas_model(job_id: str, berkas: str, dasar: str, config: dict) -> tuple[str, str]:
    """
    Salin model + konfigurasi (.json) dari folder job ke folder model. Mengembalikan path relatifnya.
    """
    path_model = MODEL_DIR / f"{dasar}{Path(berkas).suffix}"
    path_config = MODEL_DIR / f"{dasar}.json"
    shutil.copy2(TMP_DIR / job_id / berkas, path_model)
    path_config.write_text(json.dumps({**config, "model_file": path_model.name}, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    rel_model, rel_config = storage.relatif(path_model), storage.relatif(path_config)
    storage.kirim(rel_model)
    storage.kirim(rel_config)
    return rel_model, rel_config


def save(draft: dict) -> dict:
    """Simpan calon model hasil latih: berkas model ke storage, datanya ke tabel model"""
    user = _require_admin()
    algoritma, nama = draft["algoritma"], draft["nama"].strip()
    tabel = "model_klasifikasi" if algoritma == RF else "model_prediksi"
    if not nama or len(nama) > 100:
        raise ServiceError("Nama model wajib diisi, maksimal 100 karakter.")
    if db.fetch_one(f"SELECT 1 FROM {tabel} WHERE lower(nama) = lower(%s)", (nama,)):
        raise ServiceError("Nama model sudah dipakai.")

    hasil = jobs.selesai(draft["job_id"])["hasil"]
    m = hasil["evaluasi"]["metrik"]
    if algoritma == RF:
        rel_model, rel_config = _simpan_berkas_model(draft["job_id"], "model.pkl",
                                                     f"rf_{_slug(nama)}_{date.today():%Y%m%d}", hasil["config"])
        row = db.fetch_one(
            "INSERT INTO model_klasifikasi (nama, kecamatan_id, path_model, path_config, akurasi, presisi, recall, "
            "f1, evaluasi, hyperparameter, tanggal_citra, dibuat_oleh) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
            (nama, hasil["kecamatan_id"], rel_model, rel_config, m["accuracy"], m["precision"], m["recall"],
             m["f1"], db.json(hasil["evaluasi"]), db.json(hasil["params"]),
             [date.fromisoformat(t) for t in hasil["tanggal_citra"]], user["nama"]),
        )
    else:
        rel_model, rel_config = _simpan_berkas_model(draft["job_id"], "model.keras",
                                                     f"convlstm_{_slug(nama)}_{date.today():%Y%m%d}", hasil["config"])
        row = db.fetch_one(
            "INSERT INTO model_prediksi (nama, kecamatan_ids, path_model, path_config, akurasi, presisi, recall, "
            "f1, evaluasi, hyperparameter, dibuat_oleh) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) "
            "RETURNING id",
            (nama, hasil["kecamatan_ids"], rel_model, rel_config, m["accuracy"], m["precision"], m["recall"],
             m["f1"], db.json(hasil["evaluasi"]), db.json(hasil["params"]), user["nama"]),
        )
    jobs.hapus(draft["job_id"])
    _daftar.clear()  # biar model baru langsung muncul di dropdown
    return get(algoritma, row["id"])


def discard(draft: dict) -> None:
    """Buang calon model beserta berkas sementaranya"""
    if draft.get("job_id"):
        jobs.hapus(draft["job_id"])
