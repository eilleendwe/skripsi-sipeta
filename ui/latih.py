"""
Bagian yang sama di kedua halaman Latih Model (Random Forest dan ConvLSTM):
jalankan pelatihan dengan progress bar, lalu Simpan/Buang.
"""

import streamlit as st

from services import backend, models
from services.errors import ServiceError
from ui import fmt, layout


def _k(prefix: str, nama: str) -> str:
    return f"{prefix}_{nama}"


def status(prefix: str) -> tuple[bool, dict]:
    """
    (sedang berjalan, error validasi terakhir). Job yang masih dipantau juga dihitung berjalan.
    """
    berjalan = st.session_state.get(_k(prefix, "berjalan"), False) or _k(prefix, "job") in st.session_state
    return berjalan, st.session_state.get(_k(prefix, "errors"), {})


def draft(prefix: str) -> dict | None:
    """Calon model dari pelatihan terakhir yang belum disimpan atau dibuang"""
    return st.session_state.get(_k(prefix, "draft"))


def start_button(prefix: str, disabled: bool) -> None:
    def _mulai():
        st.session_state[_k(prefix, "berjalan")] = True
        st.session_state.pop(_k(prefix, "errors"), None)

    st.button("Latih model", type="primary", icon=":material/model_training:", key=_k(prefix, "latih"),
              disabled=disabled, on_click=_mulai)


def run_if_started(prefix: str, algoritma: str, spec: dict) -> None:
    """
    Mulai pelatihan kalau tombol baru ditekan, atau lanjut memantau job yang belum selesai (mis.
    habis pindah halaman). Form dikunci selama jalan, hasilnya disimpan di session_state.
    """
    k_jalan, k_job = _k(prefix, "berjalan"), _k(prefix, "job")
    job = st.session_state.get(k_job)  # (job_id, spec) pelatihan yang sudah dimulai
    if not st.session_state.get(k_jalan) and job is None:
        return
    st.session_state[k_jalan] = False

    if job is None:
        try:
            errors = models.validate(algoritma, spec)
        except ServiceError as e:
            errors = {"umum": e.message}
        if errors:
            st.session_state[_k(prefix, "errors")] = errors
            st.rerun()

    bar = st.progress(0.0, text="Menyiapkan…")
    try:
        if job is None:
            job_id = models.mulai_latih(algoritma, spec, bar)
            # disimpan supaya pemantauan bisa lanjut kalau halaman ditinggal lalu dibuka lagi
            st.session_state[k_job] = (job_id, spec)
        else:
            job_id, spec = job
        hasil = backend.tunggu_job(job_id, bar)
    except ServiceError as e:
        st.session_state.pop(k_job, None)
        st.session_state[_k(prefix, "errors")] = e.fields or {"umum": e.message}
        st.rerun()
    st.session_state.pop(k_job, None)
    st.session_state[_k(prefix, "draft")] = models.buat_draft(algoritma, spec, job_id, hasil)
    st.rerun()


def general_error(errors: dict) -> None:
    if errors.get("umum"):
        st.error(errors["umum"], icon=":material/error:")
    elif errors:
        st.error("Periksa kembali isian yang ditandai.", icon=":material/error:")


def labeled(nama: str, algoritma: str, param: str) -> str:
    """Label field hyperparameter + keterangan kecil di bawahnya (penjelasan dan nilai rekomendasi)"""
    penjelasan, nilai = models.REKOMENDASI[algoritma][param]
    return f"{nama}  \n:small[:gray[{penjelasan} (Direkomendasikan: {nilai})]]"


def evaluation_caption(calon: dict, satuan: str) -> None:
    """Keterangan di atas panel evaluasi: nama model, cakupan, jumlah data uji, status simpan"""
    status = "Sudah tersimpan." if calon.get("tersimpan") else "Belum tersimpan."
    st.caption(
        f"Hasil untuk model “{calon['nama']}” · {models.coverage_label(calon)} · "
        f"{fmt.angka(calon['evaluasi']['n_uji'], 0)} {satuan}. {status}"
    )


def save_discard(prefix: str, calon: dict) -> None:
    """
    Tombol Simpan model dan Buang. Setelah disimpan, evaluasinya tetap tampil (ditandai tersimpan)
    sampai model lain dilatih.
    """
    k_draft = _k(prefix, "draft")
    if calon.get("tersimpan"):
        st.button("Tersimpan", type="primary", icon=":material/check:", key=_k(prefix, "simpan"), disabled=True)
        return
    with st.container(horizontal=True):
        if st.button("Simpan model", type="primary", icon=":material/save:", key=_k(prefix, "simpan")):
            try:
                model = models.save(calon)
            except ServiceError as e:
                st.error(e.message, icon=":material/error:")
            else:
                st.session_state[k_draft] = {**calon, "tersimpan": True}
                layout.set_flash(f"Model “{model['nama']}” disimpan.", ":material/check_circle:")
                st.rerun()
        if st.button("Buang", icon=":material/delete:", key=_k(prefix, "buang")):
            models.discard(calon)
            st.session_state.pop(k_draft, None)
            layout.set_flash("Hasil pelatihan dibuang.")
            st.rerun()
