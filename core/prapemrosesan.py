"""
Cloud masking SCL + buffer dan pengisian awan pakai PCHIP, dipakai pada modul Latih dan Klasifikasi.
Kode metode isi: 0 = tidak terisi, 1 = PCHIP, 2 = backward fill, 3 = forward fill.
"""

from datetime import datetime

import numpy as np
from scipy.ndimage import distance_transform_edt

PROCESS_VERSION = "pchip_manual_cloudbuffer30m_v2"

# SCL: 3 = cloud shadow, 8 = cloud medium prob., 9 = cloud high prob., 10 = thin cirrus
CLOUD_RELATED_SCL = [3, 8, 9, 10]
CLOUD_BUFFER_METERS = 30.0
ANCHOR_CLOUD_MAX = 6.0  # batas % awan (sesudah buffer) supaya tanggal jadi anchor


def date_to_ordinal(date_str):
    """'YYYY-MM-DD' -> nomor hari (float), dipakai sebagai sumbu waktu PCHIP"""
    return float(datetime.strptime(date_str, "%Y-%m-%d").toordinal())


# ============================================================
# Cloud mask: SCL 3/8/9/10
# ============================================================
def cloud_mask(scl, res_x, res_y, cloud_classes=CLOUD_RELATED_SCL, buffer_m=CLOUD_BUFFER_METERS):
    """
    Return (awan, valid). awan = piksel berawan setelah buffer, valid = SCL bukan 0 (NoData).
    """
    valid = scl != 0
    base_cloud = np.isin(scl, cloud_classes) & valid
    if np.any(base_cloud):
        # jarak (meter) setiap piksel ke piksel awan terdekat
        distance_m = distance_transform_edt(~base_cloud, sampling=(res_y, res_x))
        awan = (distance_m <= buffer_m) & valid
    else:
        awan = np.zeros_like(base_cloud, dtype=bool)
    return awan, valid


def buffer_halo(res_x, res_y, buffer_m=CLOUD_BUFFER_METERS):
    """Jumlah piksel tambahan"""
    return int(np.ceil(buffer_m / min(res_x, res_y)))


# ============================================================
# PCHIP
# ============================================================
def pchip_internal_slope(h_prev, h_next, d_prev, d_next):
    """
    Turunan di titik interior: rata-rata harmonik berbobot dari dua secant.
    Bernilai 0 jika secant berlawanan tanda atau salah satunya nol
    """
    slope = np.zeros_like(d_prev, dtype=np.float64)
    same_direction = (d_prev != 0) & (d_next != 0) & (np.sign(d_prev) == np.sign(d_next))
    if np.any(same_direction):
        hp, hn = h_prev[same_direction], h_next[same_direction]
        dp, dn = d_prev[same_direction], d_next[same_direction]
        w1, w2 = 2.0 * hn + hp, hn + 2.0 * hp
        slope[same_direction] = (w1 + w2) / (w1 / dp + w2 / dn)
    return slope


def pchip_edge_slope(h0, h1, d0, d1):
    """Turunan di titik ujung: formula tiga titik satu sisi"""
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    wrong_direction = np.sign(slope) != np.sign(d0)
    limited = (~wrong_direction) & (np.sign(d0) != np.sign(d1)) & (np.abs(slope) > 3.0 * np.abs(d0))
    slope[wrong_direction] = 0.0
    slope[limited] = 3.0 * d0[limited]
    return slope


def pchip_manual_from_anchors(anchor_times, anchor_values, anchor_valid, target_time):
    """
    Isi nilai pada `target_time` untuk banyak piksel sekaligus (vektorisasi).

    anchor_times  : (n_anchor,)          waktu anchor (ordinal hari)
    anchor_values : (n_anchor, n_pixel)  nilai reflektansi anchor
    anchor_valid  : (n_anchor, n_pixel)  True jika anchor valid (tidak berawan) pada piksel tsb.
    Return        : output (n_pixel,), method (n_pixel,)
                    method: 0 = tidak terisi, 1 = PCHIP, 2 = backward fill, 3 = forward fill
    """
    x = np.asarray(anchor_times, dtype=np.float64)
    y = np.asarray(anchor_values, dtype=np.float64)
    valid = np.asarray(anchor_valid, dtype=bool)
    n_anchor, n_pixel = y.shape
    output = np.full(n_pixel, np.nan, dtype=np.float64)
    method = np.zeros(n_pixel, dtype=np.uint8)
    anchor_index = np.arange(n_anchor, dtype=np.int32)[:, None]

    # --- 2 anchor valid terdekat sebelum (j0 < j1) dan sesudah (j2 < j3) target, per piksel ---
    before_mask = valid & (x[:, None] < target_time)
    after_mask = valid & (x[:, None] > target_time)
    j1 = np.where(before_mask, anchor_index, -1).max(axis=0)
    j2 = np.where(after_mask, anchor_index, n_anchor).min(axis=0)
    j0 = np.where(before_mask & (anchor_index < j1[None, :]), anchor_index, -1).max(axis=0)
    j3 = np.where(after_mask & (anchor_index > j2[None, :]), anchor_index, n_anchor).min(axis=0)
    has_before, has_after = j1 >= 0, j2 < n_anchor

    # --- Hanya ada anchor sesudah -> backward fill (nilai anchor sesudah terdekat) ---
    only_after = (~has_before) & has_after
    if np.any(only_after):
        cols = np.where(only_after)[0]
        output[cols] = y[j2[cols], cols]
        method[cols] = 2

    # --- Hanya ada anchor sebelum -> forward fill (nilai anchor sebelum terdekat) ---
    only_before = has_before & (~has_after)
    if np.any(only_before):
        cols = np.where(only_before)[0]
        output[cols] = y[j1[cols], cols]
        method[cols] = 3

    # --- Ada anchor sebelum & sesudah -> PCHIP ---
    cols = np.where(has_before & has_after)[0]
    if len(cols) == 0:
        return output, method

    k0, k1, k2, k3 = j0[cols], j1[cols], j2[cols], j3[cols]
    x1, x2 = x[k1], x[k2]
    y1, y2 = y[k1, cols], y[k2, cols]
    h = x2 - x1
    d = (y2 - y1) / h
    m1, m2 = d.copy(), d.copy()
    has0, has3 = k0 >= 0, k3 < n_anchor

    def secant(a, b, c):
        """Jarak waktu dan kemiringan antara anchor a -> b pada kolom piksel c"""
        hab = x[b] - x[a]
        return hab, (y[b, c] - y[a, c]) / hab

    # Turunan m1 di anchor sebelum (x1)
    use = has0  # interior: x0, x1, x2
    if np.any(use):
        c = cols[use]
        h_prev, d_prev = secant(k0[use], k1[use], c)
        h_next, d_next = secant(k1[use], k2[use], c)
        m1[use] = pchip_internal_slope(h_prev, h_next, d_prev, d_next)
    use = (~has0) & has3  # endpoint kiri: x1, x2, x3
    if np.any(use):
        c = cols[use]
        h0, d0 = secant(k1[use], k2[use], c)
        h1, d1 = secant(k2[use], k3[use], c)
        m1[use] = pchip_edge_slope(h0, h1, d0, d1)

    # Turunan m2 di anchor sesudah (x2)
    use = has3  # interior: x1, x2, x3
    if np.any(use):
        c = cols[use]
        h_prev, d_prev = secant(k1[use], k2[use], c)
        h_next, d_next = secant(k2[use], k3[use], c)
        m2[use] = pchip_internal_slope(h_prev, h_next, d_prev, d_next)
    use = (~has3) & has0  # endpoint kanan: x2, x1, x0 (arah terbalik)
    if np.any(use):
        c = cols[use]
        h0, d0 = secant(k1[use], k2[use], c)
        h1, d1 = secant(k0[use], k1[use], c)
        m2[use] = pchip_edge_slope(h0, h1, d0, d1)

    # Basis cubic Hermite pada interval [x1, x2]
    s = (target_time - x1) / h
    h00 = 2 * s**3 - 3 * s**2 + 1
    h10 = s**3 - 2 * s**2 + s
    h01 = -2 * s**3 + 3 * s**2
    h11 = s**3 - s**2
    output[cols] = h00 * y1 + h10 * h * m1 + h01 * y2 + h11 * h * m2
    method[cols] = 1
    return output, method


def cast_values_to_dtype(values, dtype):
    """Bulatkan & clip hasil float ke dtype raster asli"""
    dtype = np.dtype(dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(values), info.min, info.max).astype(dtype)
    return values.astype(dtype)


# ============================================================
# Isi awan satu citra pakai anchor arsip
# ============================================================
def isi_awan(citra, awan, tanggal, anchors, bands):
    """
    Isi piksel berawan di `citra` pakai anchor arsip per band.

    Tiap anchor berisi tanggal, masker cerah, dan band yang mengembalikan raster asli anchor
    (NaN di piksel NoData). Hasilnya kode metode per piksel; piksel baru dianggap terisi kalau semua
    band-nya terisi
    """
    metode = np.zeros(awan.shape, dtype=np.uint8)
    cloud_pixels = np.flatnonzero(awan)
    if cloud_pixels.size == 0 or not anchors:
        return metode

    anchors = sorted(anchors, key=lambda a: a["tanggal"])
    anchor_times = np.array([date_to_ordinal(a["tanggal"]) for a in anchors], dtype=np.float64)
    cerah = np.stack([a["cerah"].reshape(-1)[cloud_pixels] for a in anchors], axis=0)
    target_time = date_to_ordinal(tanggal)

    terisi_semua = np.ones(cloud_pixels.size, dtype=bool)
    metode_flat = None
    for band in bands:
        values = np.stack([a["baca"](band).reshape(-1)[cloud_pixels] for a in anchors], axis=0)
        valid = cerah & np.isfinite(values)

        filled_values, method = pchip_manual_from_anchors(anchor_times, values, valid, target_time)
        filled = np.isfinite(filled_values)
        out_flat = citra[band].reshape(-1) 
        out_flat[cloud_pixels[filled]] = cast_values_to_dtype(filled_values[filled], citra[band].dtype)

        terisi_semua &= filled
        if metode_flat is None:
            metode_flat = method

    metode_flat = np.where(terisi_semua, metode_flat, 0).astype(np.uint8)
    metode.reshape(-1)[cloud_pixels] = metode_flat
    return metode


def ringkas_metode(awan, metode, valid):
    """Persentase piksel berawan dan cara pengisiannya terhadap piksel valid"""
    total = max(int(valid.sum()), 1)

    def pct(mask):
        return round(100 * int(mask.sum()) / total, 2)

    return {
        "persen_awan": pct(awan),
        "persen_pchip": pct(awan & (metode == 1)),
        "persen_backward": pct(awan & (metode == 2)),
        "persen_forward": pct(awan & (metode == 3)),
        "persen_tidak_terisi": pct(awan & (metode == 0)),
    }
