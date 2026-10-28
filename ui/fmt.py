"""Format tampilan tanggal, dan label model"""

from datetime import date

_BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


def angka(nilai: float, desimal: int = 2) -> str:
    """1234.5 -> '1.234,50'"""
    teks = f"{nilai:,.{desimal}f}"
    return teks.replace(",", "#").replace(".", ",").replace("#", ".")


def km2(nilai: float) -> str:
    return f"{angka(nilai)} km²"


def selisih_km2(nilai: float) -> str:
    """Selisih bertanda: '+0,42 km²' atau '−0,10 km²'"""
    tanda = "+" if nilai > 0 else ("−" if nilai < 0 else "")
    return f"{tanda}{angka(abs(nilai))} km²"


def persen(nilai: float, desimal: int = 1) -> str:
    return f"{angka(nilai, desimal)}%"


def skor(nilai: float) -> str:
    """Metrik 0..1 dengan tiga desimal: '0,912'"""
    return angka(nilai, 3)


def tanggal(d: date | None) -> str:
    return f"{d.day} {_BULAN[d.month - 1]} {d.year}" if d else "-"


def rentang(awal: date | None, akhir: date | None, tahun_saja: bool = False) -> str:
    if not awal or not akhir:
        return "-"
    if tahun_saja:
        return f"{awal.year}–{akhir.year}"
    return f"{tanggal(awal)} – {tanggal(akhir)}"


def model_label(model: dict, cakupan: str) -> str:
    """Opsi dropdown model: nama · cakupan · metrik utama · tanggal latih"""
    metrik = model["evaluasi"]["metrik"]
    utama = f"F1 {skor(metrik['f1'])}" if "f1" in metrik else f"IoU {skor(metrik['iou'])}"
    return f"{model['nama']} · {cakupan} · {utama} · {tanggal(model['tanggal_latih'])}"
