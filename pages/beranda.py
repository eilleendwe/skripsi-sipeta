from html import escape
from pathlib import Path

import streamlit as st

from services import classify, predict, regions, roles
from services.errors import ServiceError
from ui import guard, layout, nav

user = guard.require_role(roles.USER)

_PETA_DIR = Path(__file__).resolve().parent.parent / "data" / "wilayah"

# Tiga modul utama
_MODUL = [
    ("layers", "Klasifikasi",
     "Memetakan tiap piksel citra menjadi lahan terbangun atau nonterbangun dengan Random Forest. "
     "Piksel berawan diisi PCHIP lebih dulu.",
     [("Latih model", "Random Forest dari arsip citra kecamatan"),
      ("Klasifikasi citra", "peta 10 m dan luas lahan terbangun")],
     [("klasifikasi_latih", "Latih model"), ("klasifikasi_citra", "Klasifikasi citra")]),
    ("timeline", "Prediksi",
     "Memperkirakan sebaran lahan terbangun tahun berikutnya dari urutan peta tahunan dengan "
     "Convolutional LSTM.",
     [("Latih model", "ConvLSTM dari arsip peta klasifikasi"),
      ("Prediksi citra", "peta lahan terbangun tahun mendatang")],
     [("prediksi_latih", "Latih model"), ("prediksi_citra", "Prediksi citra")]),
    ("monitoring", "Pemantauan",
     "Membandingkan dua hasil klasifikasi tersimpan untuk melihat di mana lahan terbangun bertambah "
     "atau berkurang.",
     [("Perbandingan", "peta perubahan antara dua tanggal"),
      ("Riwayat", "grafik dan tabel luas lahan terbangun")],
     [("pemantauan", "Buka Pemantauan")]),
]

# Bagian sekilas peta
def _5w1h(jumlah_kec: int) -> list[tuple[str, str, str, str]]:
    tahun = f"{classify.TANGGAL_MIN.year}–{classify.TANGGAL_MAKS.year}"
    prediksi = " dan ".join(str(t) for t in predict.TAHUN_PREDIKSI)
    return [
        ("travel_explore", "Apa", "What",
         "Sistem informasi geografis berbasis web untuk memetakan dan memantau lahan terbangun dari citra "
         "Sentinel-2 Level-2A."),
        ("trending_up", "Mengapa", "Why",
         "Lahan terbangun di Tangerang terus bertambah, sementara citra satelit sering tertutup awan. "
         "Awan perlu diisi dulu agar peta antartanggal bisa dibandingkan."),
        ("groups", "Siapa", "Who",
         "Perencana wilayah, pemerintah daerah, peneliti, dan mahasiswa. Pengguna memakai hasilnya, Admin "
         "melatih model, Super Admin mengelola akun."),
        ("location_on", "Di mana", "Where",
         f"Kota Tangerang, Kota Tangerang Selatan, dan Kabupaten Tangerang, disajikan per kecamatan "
         f"({jumlah_kec} kecamatan)."),
        ("calendar_month", "Kapan", "When",
         f"Arsip citra {tahun}, diklasifikasi per tanggal akuisisi. Prediksi untuk tahun {prediksi}."),
        ("account_tree", "Bagaimana", "How",
         "Awan diisi interpolasi PCHIP, citra diklasifikasi Random Forest, lalu urutan peta tahunan "
         "diprediksi ConvLSTM."),
    ]


def _hero(jumlah_wilayah: int, jumlah_kec: int) -> None:
    angka = [
        (str(jumlah_wilayah), "Kota/kabupaten"),
        (str(jumlah_kec), "Kecamatan"),
        (f"{classify.TANGGAL_MIN.year}–{classify.TANGGAL_MAKS.year}", "Arsip citra"),
        ("10 m", "Resolusi piksel"),
    ]
    with st.container(key="beranda_hero"):
        st.html(
            '<div class="sp-hero-kicker">Sentinel-2 Level-2A · Random Forest · ConvLSTM</div>'
            '<div class="sp-beranda-judul">Sistem Pemantauan Tutupan Lahan Terbangun</div>'
            '<p class="sp-body">SIPETA memetakan lahan terbangun di Tangerang dari citra satelit dan memantau '
            "perubahannya dari waktu ke waktu, sampai tingkat kecamatan.</p>"
            '<div class="sp-statistik">' + "".join(
                f"<div><b>{escape(nilai)}</b><span>{escape(label)}</span></div>" for nilai, label in angka
            ) + "</div>"
        )


def _modul(no: int, ikon: str, nama: str, keterangan: str, fungsi: list, tautan: list) -> None:
    with st.container(key=f"beranda_modul_{no}"):
        st.html(
            f'<div class="sp-modul-head"><span class="sp-modul-ic material-symbols-rounded">{ikon}</span>'
            f'<div><div class="sp-modul-no">Modul {no:02d}</div><b>{escape(nama)}</b></div></div>'
            f'<p class="sp-modul-teks">{escape(keterangan)}</p>'
        )
        st.html('<ul class="sp-modul-sub">' + "".join(
            f"<li><b>{escape(label)}</b>{escape(teks)}</li>" for label, teks in fungsi
        ) + "</ul>")
        with st.container(key=f"beranda_buka_{no}", horizontal=True, gap="medium"):
            for key, label in tautan:
                if nav.allowed(key, user["role"]):
                    st.page_link(nav.path(key), label=label, icon=":material/arrow_forward:")


def _baris_info(key: str, ikon: str, judul: str, teks: str, halaman: str, label: str) -> None:
    """Baris tipis berisi keterangan singkat dan tautan ke satu halaman"""
    with st.container(key=key):
        c_teks, c_buka = st.columns([3, 1], vertical_alignment="center")
        c_teks.html(f'<div class="sp-baris-info"><span class="material-symbols-rounded">{ikon}</span>'
                    f"<b>{escape(judul)}</b><span>{escape(teks)}</span></div>")
        if nav.allowed(halaman, user["role"]):
            c_buka.page_link(nav.path(halaman), label=label, icon=":material/arrow_forward:")


def _wilayah_kajian(wilayah: list[dict]) -> None:
    nama = {w["id"]: w["nama"] for w in wilayah}
    pilih = st.segmented_control("Wilayah", list(nama), format_func=nama.get, default=wilayah[0]["id"],
                                 key="beranda_wilayah", label_visibility="collapsed") or wilayah[0]["id"]
    kecamatan = regions.list_kecamatan(pilih)

    c_peta, c_info = st.columns([1.7, 1], gap="large")
    with c_peta.container(key="beranda_peta"):
        peta = _PETA_DIR / f"{pilih}.jpg"
        if peta.is_file():
            st.image(str(peta), width="stretch")
        else:
            layout.empty_state("Peta wilayah belum tersedia.", "map")
    with c_info:
        st.html(f'<div class="sp-lead">{escape(nama[pilih])}</div>')
        layout.metric_cards([("Jumlah kecamatan", str(len(kecamatan)))], min_width=140)
        layout.section("Kecamatan")
        st.html('<div class="sp-kec-list">'
                + "".join(f"<span>{escape(k['nama'])}</span>" for k in kecamatan) + "</div>")


layout.page_header("Beranda")
with layout.card("beranda"):
    try:
        wilayah = regions.list_wilayah()
        jumlah_kec = len(regions.list_kecamatan())
    except ServiceError as e:
        st.error(e.message, icon=":material/cloud_off:")
        st.stop()

    _hero(len(wilayah), jumlah_kec)

    layout.section("Sekilas SIPETA")
    st.html('<div class="sp-w-grid">' + "".join(
        f'<div class="sp-w"><div class="sp-w-head"><span class="sp-w-ic material-symbols-rounded">{ikon}</span>'
        f"<b>{tanya}</b><small>{en}</small></div><p>{escape(jawab)}</p></div>"
        for ikon, tanya, en, jawab in _5w1h(jumlah_kec)
    ) + "</div>")

    st.divider()
    layout.section("Modul dan alur pengolahan")
    kolom = st.columns(len(_MODUL), gap="medium")
    for no, (c, modul) in enumerate(zip(kolom, _MODUL), start=1):
        with c:
            _modul(no, *modul)
    _baris_info("beranda_admin", "manage_accounts", "Kelola Pengguna",
                "Mengatur akun yang terdaftar beserta perannya, khusus Super Admin.",
                "kelola_pengguna", "Buka Kelola Pengguna")
    _baris_info("beranda_panduan", "menu_book", "Cara memakai SIPETA",
                "Langkah pemakaian tiap modul dan aturan penamaan berkas ada di Panduan Pengguna.",
                "panduan", "Buka Panduan Pengguna")

    st.divider()
    layout.section("Wilayah kajian")
    if wilayah:
        _wilayah_kajian(wilayah)
