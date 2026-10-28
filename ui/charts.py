"""
Grafik Altair yang dipakai halaman.

Warna: satu gradasi untuk besaran (confusion matrix), dua warna tetap untuk dua seri
(latih = teal, validasi = terakota). Teks selalu warna tinta, bukan warna seri.
"""

import altair as alt
import pandas as pd
import streamlit as st

TEAL = "#00908A"
TERAKOTA = "#B8552F"
TINTA = "#1E2A32"
TINTA_REDUP = "#55636E"
GARIS = "#E1E6E9"

_AXIS = dict(labelColor=TINTA_REDUP, titleColor=TINTA_REDUP, gridColor=GARIS, domainColor=GARIS,
              tickColor=GARIS, labelFontSize=12, titleFontSize=12, titleFontWeight="normal")


def _show(chart: alt.Chart, height: int) -> None:
    chart = (
        chart.properties(height=height, background="transparent")
        .configure_view(stroke=None)
        .configure_axis(**_AXIS)
        .configure_legend(labelColor=TINTA, titleColor=TINTA_REDUP, orient="top", title=None,
                          symbolStrokeWidth=3, labelFontSize=12)
    )
    st.altair_chart(chart, width="stretch")


def _id(nilai: float, desimal: int = 0) -> str:
    return f"{nilai:,.{desimal}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def confusion_matrix(cm: list[list[int]], labels: list[str], height: int = 250) -> None:
    """
    Heatmap 2x2. Baris = kelas aktual, kolom = kelas prediksi. Persentase dihitung per baris.
    """
    rows = [
        {"Aktual": labels[i], "Prediksi": labels[j], "Jumlah": cm[i][j],
         "Teks": _id(cm[i][j]), "Proporsi": f"{_id(100 * cm[i][j] / max(sum(cm[i]), 1), 1)}%"}
        for i in range(2) for j in range(2)
    ]
    df = pd.DataFrame(rows)
    ambang = max(r["Jumlah"] for r in rows) * 0.55
    dasar = alt.Chart(df).encode(
        x=alt.X("Prediksi:N", sort=labels, title="Prediksi", axis=alt.Axis(labelAngle=0, orient="top")),
        y=alt.Y("Aktual:N", sort=labels, title="Aktual"),
        tooltip=["Aktual", "Prediksi", alt.Tooltip("Teks", title="Jumlah piksel"),
                 alt.Tooltip("Proporsi", title="Persen dari kelas aktual")],
    )
    kotak = dasar.mark_rect(stroke="white", strokeWidth=2, cornerRadius=4).encode(
        color=alt.Color("Jumlah:Q", scale=alt.Scale(range=["#E3F1EF", "#0B5F5C"]), legend=None)
    )
    angka = dasar.mark_text(fontSize=15, fontWeight=600, dy=-6).encode(
        text="Teks", color=alt.condition(alt.datum.Jumlah > ambang, alt.value("white"), alt.value(TINTA))
    )
    porsi = dasar.mark_text(fontSize=11, dy=12).encode(
        text="Proporsi", color=alt.condition(alt.datum.Jumlah > ambang, alt.value("white"), alt.value(TINTA_REDUP))
    )
    _show(kotak + angka + porsi, height)


def loss_curve(rows: list[dict], height: int = 250) -> None:
    """Kurva loss latih vs validasi per epoch"""
    df = pd.DataFrame(rows).rename(columns={"latih": "Latih", "validasi": "Validasi"})
    panjang = df.melt("epoch", var_name="Seri", value_name="Loss")
    dekat = alt.selection_point(nearest=True, on="pointerover", fields=["epoch"], empty=False)
    garis = alt.Chart(panjang).mark_line(strokeWidth=2).encode(
        x=alt.X("epoch:Q", title="Epoch", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("Loss:Q", title="Loss", scale=alt.Scale(zero=False)),
        color=alt.Color("Seri:N", scale=alt.Scale(domain=["Latih", "Validasi"], range=[TEAL, TERAKOTA])),
    )
    penunjuk = alt.Chart(df).mark_rule(color=TINTA_REDUP, strokeWidth=1).encode(
        x="epoch:Q",
        opacity=alt.condition(dekat, alt.value(0.6), alt.value(0)),
        tooltip=[alt.Tooltip("epoch", title="Epoch"), alt.Tooltip("Latih", format=".4f"),
                 alt.Tooltip("Validasi", format=".4f")],
    ).add_params(dekat)
    _show(garis + penunjuk, height)


def area_trend(rows: list[dict], sorot: tuple | None = None, height: int = 280) -> None:
    """
    Luas lahan terbangun terhadap waktu. `sorot` = (tanggal awal, tanggal akhir) yang sedang
    dibandingkan, digambar sebagai titik terisi.
    """
    df = pd.DataFrame(rows)
    df["tanggal"] = pd.to_datetime(df["tanggal"])
    df["Teks"] = df["luas_terbangun"].map(lambda v: f"{_id(v, 2)} km²")
    df["Label"] = df["tanggal"].dt.strftime("%d-%m-%Y")
    df["Dipilih"] = df["tanggal"].isin(pd.to_datetime(list(sorot))) if sorot else False
    dasar = alt.Chart(df).encode(
        x=alt.X("tanggal:T", title=None, axis=alt.Axis(format="%Y", tickCount="year"),
                scale=alt.Scale(padding=36)),
        y=alt.Y("luas_terbangun:Q", title="Luas lahan terbangun (km²)", scale=alt.Scale(zero=False, nice=True)),
        tooltip=[alt.Tooltip("Label", title="Tanggal"), alt.Tooltip("Teks", title="Luas terbangun"),
                 alt.Tooltip("model_nama", title="Model")],
    )
    garis = dasar.mark_line(color=TEAL, strokeWidth=2)
    titik = dasar.mark_point(size=90, strokeWidth=2, color=TEAL).encode(
        fill=alt.condition(alt.datum.Dipilih, alt.value(TEAL), alt.value("white"))
    )
    label = dasar.transform_filter(alt.datum.Dipilih).mark_text(dy=-14, color=TINTA, fontSize=12).encode(text="Teks")
    _show(garis + titik + label, height)
