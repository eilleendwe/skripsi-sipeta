"""
Model ConvLSTM dan fungsi evaluasinya.

Layer ConvLSTM-nya ditulis manual, sisanya (konvolusi, Adam, BatchNormalization, Conv2D, fit)
tetap pakai TensorFlow/Keras.
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from core import peta_tahunan

tf.get_logger().setLevel("ERROR")  # sembunyikan warning deprecation dari dalam Keras
from core.config import LABEL_NODATA

THRESHOLD = 0.5
BATCH_PREDIKSI = 8


@tf.keras.utils.register_keras_serializable(package="sipeta")
class ConvLSTMManual(layers.Layer):
    """
    Layer ConvLSTM dengan rumus manual (rumus 2.39–2.44 di skripsi).

    Konvolusinya pakai tf.nn.conv2d padding same, jadi peta tetap 64x64. H dan C mulai dari nol dan
    bobotnya dipakai bareng untuk semua tahun. Nilai awal bobot dan dropout-nya disamain dengan
    ConvLSTM2D bawaan Keras.
    """

    GATES = ("i", "f", "c", "o")    # input gate, forget gate, kandidat cell state, output gate

    def __init__(self, filters, kernel_size, return_sequences=False, dropout=0.0, **kwargs):
        super().__init__(**kwargs)
        self.filters = int(filters)
        self.kernel_size = int(kernel_size)
        self.return_sequences = bool(return_sequences)
        self.dropout = float(dropout)

    def get_config(self):           # supaya model .keras bisa disimpan dan dimuat ulang
        return {**super().get_config(), "filters": self.filters, "kernel_size": self.kernel_size,
                "return_sequences": self.return_sequences, "dropout": self.dropout}

    def build(self, input_shape):   # input_shape = (batch, waktu, tinggi, lebar, kanal)
        k, f, c_in = self.kernel_size, self.filters, int(input_shape[-1])
        # Nilai awal 4 gate dibuat sekaligus lalu dipotong per gate, supaya sebarannya sama dengan ConvLSTM2D.
        W_x_awal = tf.keras.initializers.GlorotUniform()((k, k, c_in, 4 * f))
        W_h_awal = tf.keras.initializers.Orthogonal()((k, k, f, 4 * f))
        self.W_x, self.W_h, self.b = {}, {}, {}
        for j, g in enumerate(self.GATES):
            sl = slice(j * f, (j + 1) * f)
            self.W_x[g] = self.add_weight(name=f"W_x{g}", shape=(k, k, c_in, f),
                                          initializer=lambda shape, dtype=None, v=W_x_awal[..., sl]: v)
            self.W_h[g] = self.add_weight(name=f"W_h{g}", shape=(k, k, f, f),
                                          initializer=lambda shape, dtype=None, v=W_h_awal[..., sl]: v)
            self.b[g] = self.add_weight(name=f"b_{g}", shape=(f,), initializer="ones" if g == "f" else "zeros")

    def conv(self, x, w):
        """Konvolusi 2D, tanda * di rumus"""
        return tf.nn.conv2d(x, w, strides=1, padding="SAME")

    def step(self, X_t, H_prev, C_prev, masks=None):
        """Satu langkah waktu (rumus 2.39–2.44)"""
        x = {g: (X_t * masks[g] if masks else X_t) for g in self.GATES}
        i_t = tf.sigmoid(self.conv(x["i"], self.W_x["i"]) + self.conv(H_prev, self.W_h["i"]) + self.b["i"])      # (2.39)
        f_t = tf.sigmoid(self.conv(x["f"], self.W_x["f"]) + self.conv(H_prev, self.W_h["f"]) + self.b["f"])      # (2.40)
        C_tilde = tf.tanh(self.conv(x["c"], self.W_x["c"]) + self.conv(H_prev, self.W_h["c"]) + self.b["c"])     # (2.41)
        C_t = f_t * C_prev + i_t * C_tilde                                                                       # (2.42)
        o_t = tf.sigmoid(self.conv(x["o"], self.W_x["o"]) + self.conv(H_prev, self.W_h["o"]) + self.b["o"])      # (2.43)
        H_t = o_t * tf.tanh(C_t)                                                                                 # (2.44)
        return H_t, C_t

    def call(self, X, training=None):
        shape = tf.shape(X)
        H = tf.zeros((shape[0], shape[2], shape[3], self.filters), dtype=X.dtype)   # H_0 = 0
        C = tf.zeros_like(H)                                                         # C_0 = 0
        masks = None
        if training and self.dropout > 0:
            ones = tf.ones_like(X[:, 0])
            masks = {g: tf.nn.dropout(ones, rate=self.dropout) for g in self.GATES}
        outputs = []
        for t in range(X.shape[1]):              # baca tahun satu per satu, dari yang tertua
            H, C = self.step(X[:, t], H, C, masks)
            outputs.append(H)
        # return_sequences=True: keluarkan H semua tahun (untuk layer ConvLSTM berikutnya)
        # return_sequences=False: keluarkan H tahun terakhir saja (untuk Conv2D output)
        return tf.stack(outputs, axis=1) if self.return_sequences else H


def masked_binary_crossentropy(y_true, y_pred):
    """BCE yang tidak menghitung piksel nodata (-1)"""
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true)

    bce = tf.keras.losses.binary_crossentropy(y_true_safe, y_pred)  # shape (batch,H,W)
    mask_2d = tf.squeeze(mask, axis=-1)

    return tf.reduce_sum(bce * mask_2d) / (tf.reduce_sum(mask_2d) + 1e-8)


def masked_accuracy(y_true, y_pred):
    """Accuracy tanpa piksel nodata, buat log selama fit()"""
    y_true = tf.cast(y_true, tf.float32)
    mask = tf.cast(tf.not_equal(y_true, -1.0), tf.float32)
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_safe = tf.where(tf.equal(y_true, -1.0), tf.zeros_like(y_true), y_true)

    correct = tf.cast(tf.equal(y_true_safe, y_pred_bin), tf.float32) * mask
    return tf.reduce_sum(correct) / (tf.reduce_sum(mask) + 1e-8)


OPTIMIZERS = {
    "adam": lambda lr: tf.keras.optimizers.Adam(learning_rate=lr),
    "adamw": lambda lr: tf.keras.optimizers.AdamW(learning_rate=lr),
    "nadam": lambda lr: tf.keras.optimizers.Nadam(learning_rate=lr),
    "rmsprop": lambda lr: tf.keras.optimizers.RMSprop(learning_rate=lr),
    "sgd": lambda lr: tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9),
}


def build_convlstm_model(seq_len, patch_size, filters=40, kernel_size=3, learning_rate=0.001,
                         n_layers=2, dropout=0.0, optimizer="adam"):
    """
    Tumpukan ConvLSTMManual + BatchNormalization, ditutup Conv2D sigmoid untuk peluang terbangun.
    Layer ConvLSTM terakhir cuma mengeluarkan H tahun terakhir.
    """
    k = int(kernel_size)
    inputs = layers.Input(shape=(seq_len, patch_size, patch_size, 1))

    x = inputs
    for i in range(n_layers):
        last = (i == n_layers - 1)
        x = ConvLSTMManual(filters, k, return_sequences=not last, dropout=dropout, name=f"convlstm_{i + 1}")(x)
        x = layers.BatchNormalization(name=f"batchnorm_{i + 1}")(x)

    outputs = layers.Conv2D(filters=1, kernel_size=(k, k), padding="same", activation="sigmoid",
                            name="output_peluang")(x)

    model = models.Model(inputs, outputs)
    model.compile(
        optimizer=OPTIMIZERS[optimizer.lower()](learning_rate),
        loss=masked_binary_crossentropy,
        metrics=[masked_accuracy]
    )
    return model


# --- Evaluasi ------------------------------------------------------------------

def _div(a, b):
    return float(a) / float(b) if b > 0 else float("nan")


def _spatial_scores(t, p):
    tp = int(np.sum(p & t)); fp = int(np.sum(p & ~t))
    fn = int(np.sum(~p & t)); tn = int(np.sum(~p & ~t))
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "oa": _div(tp + tn, tp + fp + fn + tn),
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "f1": _div(2 * tp, 2 * tp + fp + fn),
        "iou": _div(tp, tp + fp + fn),
    }


def evaluate_predictions(before, y_true, y_prob, threshold=THRESHOLD):
    """
    Metrik peta model. `before` = peta tahun terakhir yang teramati, -1 = nodata.
    FoM cuma menilai piksel yang berubah 0 -> 1.
    """
    y_pred = (y_prob > threshold)
    valid = (y_true != -1)

    t = (y_true == 1)[valid]
    m = _spatial_scores(t, y_pred[valid])

    valid_fom = (before != -1) & valid
    hits = int(np.sum((before == 0) & (y_true == 1) & y_pred & valid_fom))
    misses = int(np.sum((before == 0) & (y_true == 1) & ~y_pred & valid_fom))
    false_alarms = int(np.sum((before == 0) & (y_true == 0) & y_pred & valid_fom))
    fom = _div(hits, hits + misses + false_alarms)

    return {
        "akurasi": m["oa"], "precision": m["precision"], "recall": m["recall"],
        "f1": m["f1"], "iou": m["iou"],
        "hits": hits, "misses": misses, "false_alarms": false_alarms, "fom": fom,
        "tp": m["tp"], "fp": m["fp"], "fn": m["fn"], "tn": m["tn"],
    }


def confusion_from_maps(before, y_true, y_pred_bin):
    """
    Confusion matrix piksel [[TN, FP], [FN, TP]], baris = aktual. Nodata tidak dihitung.
    """
    valid = (y_true != -1) & (before != -1)
    t = (y_true == 1)[valid]
    p = (y_pred_bin == 1)[valid]
    return [[int((~t & ~p).sum()), int((~t & p).sum())],
            [int((t & ~p).sum()), int((t & p).sum())]]


def forecast_recursive(model, window, n_steps, threshold=THRESHOLD):
    """
    Prediksi beberapa tahun ke depan dari `window` (peta tahun-tahun terakhir). Tahun kedua dst.
    pakai peta hasil prediksi sebelumnya. Hasilnya list peluang dan list peta biner per tahun.
    """
    nodata = (window[:, -1, :, :, 0] == -1)

    probs, bins = [], []
    for _ in range(n_steps):
        prob = model.predict(window, batch_size=BATCH_PREDIKSI, verbose=0)[..., 0]   # (K,H,W)
        binary = (prob > threshold).astype(np.float32)
        binary[nodata] = -1.0
        probs.append(prob)
        bins.append(binary)
        window = np.concatenate([window[:, 1:], binary[:, np.newaxis, :, :, np.newaxis]], axis=1)
    return probs, bins


def native_forecast(native_last, coarse_last, coarse_pred):
    """
    Peta prediksi 10 m: mulai dari peta aktual terakhir, lalu sel 64x64 yang menurut model berubah
    diganti semua. Sel yang tidak berubah tetap pakai detail aslinya.
    """
    up_last = peta_tahunan.ubah_ukuran(coarse_last, *native_last.shape)
    up_pred = peta_tahunan.ubah_ukuran(coarse_pred, *native_last.shape)
    out = native_last.copy()
    valid = native_last != LABEL_NODATA
    out[valid & (up_last == 0) & (up_pred == 1)] = 1
    out[valid & (up_last == 1) & (up_pred == 0)] = 0
    return out
