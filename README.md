# <img src="ui/logo.svg" alt="Logo SIPETA" width="36" align="absmiddle"> SIPETA: Sistem Pemantauan Tutupan Lahan Terbangun

SIPETA adalah web untuk memantau perubahan dan memprediksi tutupan lahan terbangun di Kota Tangerang, Kota Tangerang Selatan, dan Kabupaten Tangerang pada tingkat kecamatan, berbasis citra satelit Sentinel-2 Level-2A.

## Metodologi

- **Dataset**: citra Sentinel-2 Level-2A dari tahun 2016-2025, beresolusi spasial 10 m, diunduh dari [Copernicus Browser](https://browser.dataspace.copernicus.eu/). Band yang dipakai:
  - B02, B03, B04 (Blue, Green, Red), B05, B06, B07 (Red Edge), dan B12 (SWIR 2) sebagai fitur Random Forest
  - B08 (NIR) dan B11 (SWIR 1) untuk label NDBI saat melatih model
  - SCL (Scene Classification Layer) untuk mendeteksi awan (_cloud masking_)
- **Prapemrosesan**: _cloud masking_ dan pengisian piksel berawan dengan _Piecewise Cubic Hermite Interpolating Polynomial_ (PCHIP) dengan menggunakan data historis dari citra tersebut.
- **Klasifikasi**: Random Forest untuk mendeteksi dan mengklasifikasikan piksel lahan terbangun dan non-terbangun.
- **Prediksi**: Convolutional Long Short-Term Memory (ConvLSTM) untuk memprediksi perubahan lahan terbangun pada tahun berikutnya (2026-2027).

## Fitur

| Halaman          | Isi                                                                                                        |
| ---------------- | ---------------------------------------------------------------------------------------------------------- |
| Beranda          | Gambaran umum sistem dan wilayah kajian                                                                    |
| Klasifikasi      | Melatih model Random Forest dan mengklasifikasi citra satu kecamatan                                       |
| Prediksi         | Melatih model ConvLSTM dan memprediksi peta lahan terbangun tahun berikutnya                               |
| Pemantauan       | Perbandingan spasial antartanggal, grafik luas lahan terbangun terhadap waktu, dan riwayat hasil tersimpan |
| Panduan Pengguna | Cara memakai setiap fitur dan format berkas masukan                                                        |
| Tentang          | Informasi penelitian, penulis, dan sumber data                                                             |
| Kelola Pengguna  | Mengatur peran akun pengguna                                                                               |

## Peran pengguna

- **Pengguna**: klasifikasi citra, prediksi citra, pemantauan, panduan, dan tentang.
- **Admin**: semua akses Pengguna, ditambah melatih model klasifikasi dan prediksi.
- **Super Admin**: semua akses Admin, ditambah mengelola peran pengguna.

## Teknologi

| Bagian             | Teknologi                                                         |
| ------------------ | ----------------------------------------------------------------- |
| Bahasa pemrograman | Python 3.11, Streamlit                                            |
| Framework web      | Streamlit                                                         |
| Klasifikasi        | NumPy (Random Forest)                                             |
| Prediksi           | TensorFlow/Keras (ConvLSTM)                                       |
| Prapemrosesan      | NumPy (PCHIP), SciPy (buffer awan), Rasterio (baca/tulis GeoTIFF) |
| Data dan grafik    | Pandas, Altair, Pillow                                            |
| Database           | Supabase (PostgreSQL), Psycopg                                    |
| Penyimpanan berkas | Backblaze B2 (S3-compatible), Boto3                               |

## Struktur folder

```text
app.py        masuk aplikasi: navigasi, sidebar, dan pembatasan akses per peran
pages/        halaman modul
services/     penghubung halaman dengan logika di core/
core/         logika utama: PCHIP, Random Forest, ConvLSTM, job latar belakang, database, penyimpanan berkas
ui/           komponen tampilan bersama: sidebar, tab, kartu, form, grafik, CSS
data/         isi halaman Tentang dan gambar peta wilayah di Beranda
.streamlit/   tema tampilan
```
