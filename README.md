# SIPETA: Sistem Pemantauan Tutupan Lahan Terbangun

Perancangan sistem pemantauan perubahan dan prediksi tutupan lahan terbangun di Kota Tangerang, Kota Tangerang Selatan, dan Kabupaten Tangerang.

**Metodologi:**
- **Data:** Menggunakan citra Sentinel-2 Level-2A.
- **Prapemrosesan:** Menggunakan *Piecewise Cubic Hermite Interpolation Polynomial* (PCHIP) untuk *cloud masking* dan mengisi *gap* tutupan awan.
- **Klasifikasi:** Menggunakan Random Forest untuk klasifikasi lahan terbangun dan non-terbangun.
- **Prediksi:** Menggunakan Convolutional LSTM untuk memprediksi perubahan lahan terbangun pada periode berikutnya pada tingkat kecamatan.
