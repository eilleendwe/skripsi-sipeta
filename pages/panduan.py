import pandas as pd
import streamlit as st

from services import regions, roles
from services.errors import ServiceError
from ui import guard, layout

guard.require_role(roles.USER)

_KLASIFIKASI = """
**Melatih model Random Forest (khusus Admin)**

1. Buka **Klasifikasi → Latih model**.
2. Isi **Model** dengan nama yang belum dipakai, lalu pilih **Kota/Kabupaten** dan satu **Kecamatan**.
3. Pada **Folder citra kecamatan**, pilih folder yang namanya sama dengan kecamatan terpilih (lihat tab **Penamaan berkas & kecamatan**). Isinya satu subfolder per tanggal (`YYYY-MM-DD`, 2016–2025), masing-masing berisi B02, B03, B04, B05, B06, B07, B08, B11, B12, dan SCL. Periksa tabel validasi; tanggal yang gagal tidak ikut dilatih.
4. Atur hyperparameter bila perlu; penjelasan dan nilai yang direkomendasikan tertulis di bawah tiap nama field. Train ratio dan test ratio selalu berjumlah 1. **Sampel per tanggal** menentukan jumlah piksel cerah acak yang diambil dari tiap tanggal (bawaan 1.000). **Pakai seluruh piksel** memakai semua piksel cerah, tetapi latih bisa berjam-jam.
5. Tekan **Latih model** dan tunggu progres selesai: unggah citra, cloud masking, label NDBI, pelatihan Random Forest, dan evaluasi. Anda boleh pindah halaman lalu kembali; pemantauan berlanjut selama tab browser tidak ditutup.
6. Periksa **Evaluasi model** (accuracy, precision, recall, F1-score, confusion matrix, serta jumlah sampel dan blok latih/uji), lalu tekan **Simpan model** atau **Buang**. Metrik dihitung pada blok uji di kecamatan yang sama terhadap label acuan berbasis NDBI.

**Mengklasifikasi citra**

1. Buka **Klasifikasi → Klasifikasi citra**.
2. Pilih **Kota/Kabupaten** dan **Kecamatan**, lalu pilih **Model**. Daftar model hanya memuat model yang dilatih untuk kecamatan itu; centang **Tampilkan model di luar cakupan** untuk memakai model kecamatan lain.
3. Unggah berkas band B02, B03, B04, B05, B06, B07, B12, dan SCL dari citra kecamatan tersebut, pastikan semua bertanda ✓, lalu isi **Tanggal citra** (2016–2025). Tanggal terisi otomatis jika terbaca dari nama berkas.
4. Tekan **Klasifikasi**. Piksel berawan diisi PCHIP dari arsip anchor, lalu hasil menampilkan citra satelit (bisa dipilih **Citra asli** atau **Setelah PCHIP**), peta klasifikasi beresolusi 10 m, dan luas lahan terbangun dan nonterbangun dalam km².
5. Aktifkan **Tampilkan masker awan** untuk melihat piksel awan yang terdeteksi, atau **Tampilkan overlay** untuk menumpuk hasil di atas citra asli. Perhatikan persentase piksel berawan dan cara pengisiannya; di atas 30% sebaiknya pakai tanggal lain.
6. Tekan **Simpan hasil** agar hasil muncul di **Pemantauan**, atau **Unduh** untuk mengambil PNG atau GeoTIFF.

**Memantau perubahan**

1. Buka **Pemantauan**, pilih **Kota/Kabupaten**, **Kecamatan**, dan **Model**.
2. Pilih **Tanggal awal** dan **Tanggal akhir**. Hanya tanggal yang hasilnya sudah disimpan untuk model itu yang bisa dipilih.
3. Tekan **Tampilkan Data** untuk melihat citra satelit dan peta klasifikasi kedua tanggal, peta perubahan, luas yang bertambah dan berkurang, grafik, dan tabel riwayat.
"""

_PREDIKSI = """
Masukan ConvLSTM adalah **peta klasifikasi 0/1 per tanggal** (`RF_LABEL_*.tif`, hasil notebook Random Forest), bukan citra Sentinel-2. Per kecamatan, sistem membuat satu peta per tahun 2016–2025 (komposit mayoritas + konsistensi waktu), lalu mengecilkannya ke 64×64 untuk model.

**Menyiapkan arsip klasifikasi di server (khusus Admin)**

1. Buka **Prediksi → Latih model**, lalu buka bagian **Arsip klasifikasi di server**. Di sana terlihat berapa kecamatan per kota/kabupaten yang petanya sudah lengkap 2016–2025.
2. Pada **Folder arsip klasifikasi**, pilih folder `kecamatan_masked_rf_classified`, satu folder wilayah, atau satu folder kecamatan. Periksa tabelnya: jumlah tanggal, rentang tahun, status lengkap, dan kondisi arsip di server.
3. Tekan **Unggah … kecamatan ke server**. Berkas diperiksa dulu (nilai 0/1/255, ukuran seragam); kecamatan yang berkasnya tidak valid ditolak dan diberi pesan. Tanggal yang sudah ada di server diganti, tanggal lain tetap.

Arsip ini dipakai oleh opsi latih **Arsip di server** dan **selalu** menjadi masukan **Prediksi citra**. Cukup diunggah sekali, atau lagi saat ada data tahun baru.

**Melatih model ConvLSTM (khusus Admin)**

1. Buka **Prediksi → Latih model**, lalu isi **Model** dengan nama yang belum dipakai.
2. Pilih **Sumber data latih**:
   - **Arsip di server**: pilih **Kota/Kabupaten** (atau semua) dan **Kecamatan**. Keterangan di bawahnya menunjukkan berapa kecamatan yang lengkap; yang belum lengkap dilewati.
   - **Unggah folder**: seperti Latih RF, pilih folder peta klasifikasi. Folder ini hanya dipakai untuk pelatihan ini lalu dihapus dari server; yang dipakai hanya kecamatan yang lengkap 2016–2025.
3. Atur hyperparameter bila perlu: sequence length (2–4 tahun masukan), train dan test ratio (validation ratio dihitung otomatis), jumlah layer, jumlah filter, kernel size, dropout, epoch maksimum, batch size, learning rate, dan optimizer. Pembagian tahun target tertulis di bawah form, misalnya train 2020–2022, validasi 2023, test 2024–2025.
4. Tekan **Latih model**. Pelatihan berjalan tiga kali: dengan validasi dan early stopping (mencari epoch terbaik), latih ulang untuk uji, lalu model final sampai 2025. Di komputer tanpa GPU, satu kali latih untuk semua kecamatan bisa memakan waktu 1–2 jam. Anda boleh pindah halaman lalu kembali; jangan menutup atau me-refresh tab browser.
5. Periksa **Evaluasi model**. Pilih **Test 2024** (1 langkah, setara prediksi 2026) atau **Test 2025** (rekursif, setara prediksi 2027). Tersedia confusion matrix, tabel model vs persistensi (accuracy, precision, recall, F1-score, IoU, FoM, dan error luas per kecamatan), confusion matrix persistensi dan piksel yang nonterbangun di tahun awal, serta kurva loss. Selalu bandingkan model dengan persistensi: sebagian besar piksel tidak berubah dari tahun ke tahun.
6. Tekan **Simpan model** atau **Buang**.

**Memprediksi lahan terbangun**

1. Buka **Prediksi → Prediksi citra**.
2. Pilih **Kota/Kabupaten** dan **Kecamatan**, lalu pilih **Model**. Hanya model yang dilatih dengan kecamatan itu yang muncul.
3. Pilih **Tahun**: 2026 atau 2027. Tahun 2027 adalah prediksi rekursif (memakai hasil prediksi 2026 sebagai masukan), sehingga akurasinya lebih rendah.
4. Tekan **Prediksi**. Tidak perlu mengunggah apa pun: masukannya peta tahunan beberapa tahun terakhir (sebanyak sequence length model, misalnya 2022–2025) dari arsip klasifikasi di server. Jika arsip kecamatan itu belum lengkap, muncul pesan.
5. Hasil menampilkan peta aktual 2025 dan peta prediksi berdampingan (10 m), luas lahan terbangun dan nonterbangun, selisih terhadap 2025, luas yang bertambah dan berkurang, serta luas versi 64×64 sebagai pembanding. Aktifkan **Tampilkan perubahan** untuk melihat lokasi yang berubah. Detail halus peta berasal dari peta 2025; perubahan hasil prediksi berupa kotak seukuran satu sel model.
6. Tekan **Simpan hasil** atau **Unduh** (PNG atau GeoTIFF).
"""

_PENAMAAN = """
**Nama folder kecamatan**

- Pakai **nama folder** pada tabel di bawah: huruf kecil, spasi diganti garis bawah, misalnya `pondok_aren` atau `serpong_utara`.
- Sistem juga mengenali nama biasa (`Pondok Aren`, `PondokAren`) karena huruf besar/kecil, spasi, dan tanda baca diabaikan. Ejaannya harus sama; folder yang tidak dikenali akan diabaikan dan disebut di laporan.
- Folder wilayah, bila dipakai, bernama `kota_tangerang`, `kota_tangerang_selatan`, atau `kabupaten_tangerang`. Saat mengunggah, folder wilayah boleh ada atau tidak; kecamatan dikenali dari nama foldernya sendiri.
- Folder tanggal memakai format `YYYY-MM-DD` (misalnya `2019-07-04`), rentang 2016–2025.

**Citra Sentinel-2 (Klasifikasi)**

```text
batuceper/                              <- nama folder kecamatan
├── 2019-07-04/                         <- tanggal akuisisi
│   ├── T48MXU_20190704T025551_B02_10m.tif
│   ├── ...                             (B02, B03, B04, B05, B06, B07, B08, B11, B12)
│   └── T48MXU_20190704T025551_SCL_10m.tif
└── 2019-09-12/
```

- Band dikenali dari kode di nama berkas (`_B02_`, `_B11_`, `_SCL_`, …), jadi kode harus dipisahkan tanda baca dari teks lain.
- Latih model butuh 9 band + SCL per tanggal; Klasifikasi citra cukup B02–B07, B12, dan SCL dari satu tanggal. Tanggal dibaca dari nama berkas (`20190704` atau `2019-07-04`) untuk mengisi **Tanggal citra** otomatis.
- Semua citra kecamatan harus dipotong dari raster wilayah yang sama dengan arsip (grid sejajar, 10 m).

**Peta klasifikasi (Prediksi)**

```text
kecamatan_masked_rf_classified/         <- boleh juga satu folder wilayah atau satu folder kecamatan
└── kota_tangerang/
    └── batuceper/
        ├── 2016-10-07/
        │   └── RF_LABEL_2016-10-07.tif
        └── ...
```

- Nama berkas diawali `RF_LABEL_` dan berakhiran `.tif`; satu berkas per folder tanggal.
- Nilai piksel hanya 0 (nonterbangun), 1 (terbangun), atau 255 (NoData), dan ukuran raster sama untuk semua tanggal dalam satu kecamatan.
- Kecamatan baru bisa dipakai jika setiap tahun 2016–2025 punya minimal satu tanggal.
"""

layout.page_header("Panduan Pengguna")
with layout.card("panduan"):
    tab_klasifikasi, tab_prediksi, tab_nama = st.tabs(["Klasifikasi", "Prediksi", "Penamaan berkas & kecamatan"])
    tab_klasifikasi.markdown(_KLASIFIKASI)
    tab_prediksi.markdown(_PREDIKSI)
    with tab_nama:
        st.markdown(_PENAMAAN)
        st.markdown("**Daftar kecamatan**")
        try:
            kecamatan = regions.list_kecamatan()
        except ServiceError as e:
            st.error(e.message, icon=":material/cloud_off:")
        else:
            nama_wilayah = {w["id"]: w["nama"] for w in regions.list_wilayah()}
            pilihan = st.segmented_control("Kota/Kabupaten", list(nama_wilayah), default=next(iter(nama_wilayah)),
                                           format_func=nama_wilayah.get, key="panduan_wilayah") or next(iter(nama_wilayah))
            daftar = [k for k in kecamatan if k["wilayah_id"] == pilihan]
            tabel = pd.DataFrame([{"Kecamatan": k["nama"], "Nama folder kecamatan": k["id"],
                                   "Folder wilayah": k["wilayah_id"]} for k in daftar])
            tabel.index = pd.RangeIndex(1, len(tabel) + 1, name="No")  # kolom indeks tidak ikut melebar
            st.dataframe(tabel, width="stretch", height="content")
            st.caption(f"{len(daftar)} kecamatan di {nama_wilayah[pilihan]} · total {len(kecamatan)} kecamatan "
                       "di tiga kota/kabupaten.")
