"""Berkas unggahan: list (nama berkas + subfolder, berkas), cth. ("2019-07-04/T48_B02_10m.tif", berkas)"""

import shutil
from pathlib import Path, PurePosixPath

from core import citra, latih


def bagian(nama: str) -> tuple[str, ...]:
    """Potongan path nama berkas, cth. ('batuceper', '2019-07-04', 'B02.tif')"""
    return PurePosixPath((nama or "").replace("\\", "/")).parts


def simpan_band(files: list, folder: Path, pakai_subfolder_tanggal: bool) -> int:
    """
    Tulis berkas band ke folder job, mengembalikan jumlah berkas yang ditulis. Yang dipakai cuma nama
    berkasnya (+ subfolder tanggal untuk latih). Berkas disalin per 1 MB supaya folder ratusan MB tidak
    disalin lagi di memori.
    """
    jumlah = 0
    for nama, berkas in files:
        b = bagian(nama)
        if not b or citra.kenali_band(b[-1]) is None:
            continue
        tujuan = folder
        if pakai_subfolder_tanggal:
            if len(b) < 2 or not latih.TANGGAL_RE.fullmatch(b[-2]):
                continue
            tujuan = folder / b[-2]
        tujuan.mkdir(parents=True, exist_ok=True)
        berkas.seek(0)
        with open(tujuan / Path(b[-1]).name, "wb") as out:
            shutil.copyfileobj(berkas, out, 1024 * 1024)
        jumlah += 1
    return jumlah
