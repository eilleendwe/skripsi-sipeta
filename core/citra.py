"""Baca/tulis raster, tau band dari nama berkas, cek grid, ambil anchor arsip, save GeoTIFF dan PNG"""

import io
import re
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.windows import Window

from core import db, prapemrosesan, storage
from core.config import ARSIP_DIR, LABEL_NODATA
from core.jobs import ProsesError

# Tau band dari nama file, cth: T48MXU_20190704_B02_10m.tif -> B02
_BAND_RE = re.compile(r"(?<![A-Za-z0-9])(B0[2-8]|B1[12]|SCL)(?![A-Za-z0-9])", re.IGNORECASE)

WARNA_TERBANGUN = (180, 88, 58)
WARNA_NONTERBANGUN = (111, 168, 104)
WARNA_AWAN_KOSONG = (140, 156, 170)   # awan yang tidak bisa diisi (NoData)
WARNA_AWAN_TERISI = (0, 144, 138)     # awan yang diisi PCHIP / backward / forward fill
WARNA_AWAN_MASKER_KOSONG = (214, 150, 30)
WARNA_PERUBAHAN = {
    "tetap_nonterbangun": (217, 221, 214),
    "tetap_terbangun": (74, 71, 67),
    "bertambah": (201, 73, 47),
    "berkurang": (31, 120, 180),
}


def kenali_band(nama: str) -> str | None:
    cocok = _BAND_RE.search(Path(nama).name)
    return cocok.group(1).upper() if cocok else None


def index_folder(folder: Path, bands: list[str]) -> dict[str, Path]:
    """
    {band: berkas .tif} di satu folder. Semua `bands` + SCL wajib ada.
    """
    berkas = {}
    for path in sorted(folder.glob("*.tif*")):
        band = kenali_band(path.name)
        if band is None or (band not in bands and band != "SCL"):
            continue
        if band in berkas:
            raise ProsesError(f"Band {band} ada lebih dari satu di folder {folder.name}.")
        berkas[band] = path
    kurang = [b for b in [*bands, "SCL"] if b not in berkas]
    if kurang:
        raise ProsesError(f"Folder {folder.name}: band {', '.join(kurang)} tidak ditemukan.")
    return berkas


def profil(path: Path) -> dict:
    with rasterio.open(path) as src:
        return {"crs": src.crs, "transform": src.transform, "width": src.width, "height": src.height,
                "res": (abs(src.transform.a), abs(src.transform.e)), "profile": src.profile.copy()}


def periksa_grid(paths: list[Path]) -> dict:
    """
    Cek semua raster punya ukuran, CRS, dan transform yang sama. Mengembalikan profil raster pertama.
    """
    acuan = profil(paths[0])
    for path in paths[1:]:
        with rasterio.open(path) as src:
            sama = (src.width == acuan["width"] and src.height == acuan["height"]
                    and src.crs == acuan["crs"] and src.transform.almost_equals(acuan["transform"]))
        if not sama:
            raise ProsesError(f"Grid berkas {path.parent.name}/{path.name} berbeda dengan "
                              f"{paths[0].parent.name}/{paths[0].name} (ukuran, CRS, atau transform).")
    return acuan


def baca(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return src.read(1)


# --- Arsip anchor ------------------------------------------------------------

def jendela_di_arsip(src, acuan: dict, halo: int = 0) -> Window:
    """Posisi citra kecamatan (`acuan`) di grid raster arsip `src`, ditambah `halo` piksel di tiap sisi"""
    if src.crs != acuan["crs"] or not np.allclose((abs(src.transform.a), abs(src.transform.e)), acuan["res"]):
        raise ProsesError("CRS atau resolusi citra berbeda dengan arsip.")
    col = (acuan["transform"].c - src.transform.c) / src.transform.a
    row = (acuan["transform"].f - src.transform.f) / src.transform.e
    if abs(col - round(col)) > 0.01 or abs(row - round(row)) > 0.01:
        raise ProsesError("Citra tidak sejajar dengan grid arsip. Potong citra kecamatan dari raster wilayah "
                          "yang sama dengan arsip.")
    return Window(round(col) - halo, round(row) - halo, acuan["width"] + 2 * halo, acuan["height"] + 2 * halo)


def _baca_jendela(src, window: Window) -> np.ndarray:
    di_dalam = (window.col_off >= 0 and window.row_off >= 0
                and window.col_off + window.width <= src.width and window.row_off + window.height <= src.height)
    # Bagian di luar raster arsip diisi 0 (NoData).
    return src.read(1, window=window, boundless=not di_dalam, fill_value=0)


def _pembaca_band(berkas: dict[str, Path], acuan: dict):
    def baca_band(band: str) -> np.ndarray:
        with rasterio.open(berkas[band]) as src:
            nilai = _baca_jendela(src, jendela_di_arsip(src, acuan)).astype(np.float32)
            if src.nodata is not None:
                nilai[nilai == src.nodata] = np.nan
        return nilai
    return baca_band


def muat_anchor(kecamatan_id: str, kecuali_tanggal: str | None, acuan: dict, bands: list[str],
                cloud_classes: list[int], buffer_m: float) -> list[dict]:
    """Anchor dari arsip kecamatan yang sama, dipotong sesuai posisi citra masukan (`acuan`)"""
    baris = db.fetch_all(
        "SELECT tanggal, path_folder FROM citra_arsip "
        "WHERE kecamatan_id = %s AND is_anchor AND tanggal <> %s ORDER BY tanggal",
        (kecamatan_id, kecuali_tanggal or "1900-01-01"),
    )
    anchors = []
    for r in baris:
        folder = storage.ambil_folder(storage.relatif(ARSIP_DIR / r["path_folder"]))
        berkas = index_folder(folder, bands)
        with rasterio.open(berkas["SCL"]) as src:
            res_x, res_y = abs(src.transform.a), abs(src.transform.e)
            halo = prapemrosesan.buffer_halo(res_x, res_y, buffer_m)
            scl = _baca_jendela(src, jendela_di_arsip(src, acuan, halo))
        awan, valid = prapemrosesan.cloud_mask(scl, res_x, res_y, cloud_classes, buffer_m)
        inti = (slice(halo, halo + acuan["height"]), slice(halo, halo + acuan["width"]))
        anchors.append({
            "tanggal": r["tanggal"].isoformat(),
            "cerah": valid[inti] & ~awan[inti],
            "baca": _pembaca_band(berkas, acuan),
        })
    return anchors


# --- Keluaran -----------------------------------------------------------------

def tulis_geotiff(path: Path, peta: np.ndarray, acuan: dict) -> None:
    profile = acuan["profile"].copy()
    profile.update(driver="GTiff", count=1, dtype="uint8", nodata=LABEL_NODATA, compress="lzw")
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(peta.astype(np.uint8), 1)


def _png(rgba: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def png_klasifikasi(peta: np.ndarray, awan_kosong: np.ndarray | None = None) -> bytes:
    """
    1 piksel PNG = 1 piksel raster (10 m). Di luar kecamatan transparan, awan yang tidak bisa
    diisi diberi warna abu-abu.
    """
    rgba = np.zeros((*peta.shape, 4), dtype=np.uint8)
    rgba[peta == 1] = (*WARNA_TERBANGUN, 255)
    rgba[peta == 0] = (*WARNA_NONTERBANGUN, 255)
    if awan_kosong is not None:
        rgba[awan_kosong] = (*WARNA_AWAN_KOSONG, 255)
    return _png(rgba)


def png_masker_awan(terisi: np.ndarray, kosong: np.ndarray) -> bytes:
    """Masker awan (setelah buffer) untuk distack di atas citra asli"""
    rgba = np.zeros((*terisi.shape, 4), dtype=np.uint8)
    rgba[terisi] = (*WARNA_AWAN_TERISI, 255)
    rgba[kosong] = (*WARNA_AWAN_MASKER_KOSONG, 255)
    return _png(rgba)


def png_dari_folder_citra(folder: Path) -> bytes | None:
    """Citra warna asli dari folder citra masukan yang disimpan (B04-B03-B02 + SCL)"""
    try:
        berkas = index_folder(folder, ["B02", "B03", "B04"])
    except (ProsesError, OSError):
        return None
    res_x, res_y = profil(berkas["SCL"])["res"]
    awan, valid = prapemrosesan.cloud_mask(baca(berkas["SCL"]), res_x, res_y)
    return png_true_color(baca(berkas["B04"]), baca(berkas["B03"]), baca(berkas["B02"]), valid, valid & ~awan)


def png_true_color(merah, hijau, biru, valid, acuan_stretch=None) -> bytes:
    """
    Citra warna asli B04-B03-B02, di-stretch persentil 2-98.

    Persentil dihitung dari piksel `acuan_stretch` (bawaan: semua piksel valid). Citra asli dan citra
    setelah PCHIP sama-sama pakai piksel cerah, jadi warnanya sama dan awan tidak bikin citra jadi gelap.
    """
    acuan = valid if acuan_stretch is None or not acuan_stretch.any() else acuan_stretch
    rgba = np.zeros((*valid.shape, 4), dtype=np.uint8)
    for i, band in enumerate((merah, hijau, biru)):
        nilai = band.astype(np.float32)
        if acuan.any():
            bawah, atas = np.percentile(nilai[acuan], (2, 98))
            nilai = (nilai - bawah) / max(atas - bawah, 1e-6)
        rgba[..., i] = np.clip(nilai * 255, 0, 255).astype(np.uint8)
    rgba[..., 3] = np.where(valid, 255, 0)
    return _png(rgba)


def png_perubahan(awal: np.ndarray, akhir: np.ndarray) -> bytes:
    valid = (awal != LABEL_NODATA) & (akhir != LABEL_NODATA)
    rgba = np.zeros((*awal.shape, 4), dtype=np.uint8)
    rgba[valid & (awal == 0) & (akhir == 0)] = (*WARNA_PERUBAHAN["tetap_nonterbangun"], 255)
    rgba[valid & (awal == 1) & (akhir == 1)] = (*WARNA_PERUBAHAN["tetap_terbangun"], 255)
    rgba[valid & (awal == 0) & (akhir == 1)] = (*WARNA_PERUBAHAN["bertambah"], 255)
    rgba[valid & (awal == 1) & (akhir == 0)] = (*WARNA_PERUBAHAN["berkurang"], 255)
    return _png(rgba)
