-- Skema database SIPETA

-- Akun pengguna
CREATE TABLE pengguna (
    id             SERIAL       PRIMARY KEY,
    nama           VARCHAR(100) NOT NULL,
    email          VARCHAR(150) NOT NULL UNIQUE,
    password_hash  TEXT         NOT NULL,
    peran          VARCHAR(20)  NOT NULL DEFAULT 'user' CHECK (peran IN ('user', 'admin', 'superadmin')),
    tanggal_daftar DATE         NOT NULL DEFAULT CURRENT_DATE
);

-- Akun default
INSERT INTO pengguna (nama, email, password_hash, peran) VALUES
    ('Pemilik Sistem', 'superadmin@sipeta.test',
     'pbkdf2_sha256$200000$732b32128ca06811b91bea287f6e36fd$76d773bbbe697702c840594753a585610f202debbc4ca4f581ba5b11dfc2bf38', 'superadmin'),
    ('Admin SIPETA', 'admin@sipeta.test',
     'pbkdf2_sha256$200000$2798d8bf0dcdc821b58e08ea50e4e9a6$46134468b1c4cd65e3797575c6ab67af0bfa9229184e70e806a49e8fdd673d8b', 'admin'),
    ('Pengguna Demo', 'user@sipeta.test',
     'pbkdf2_sha256$200000$7c48bc8b2fcffd4a2917653a0f5dbc7a$4dbf2bbca2d8b45b6dac89b2539c2cb88e49bbc144c85de14758a67f4d849bdf', 'user');

CREATE TABLE wilayah (
    id   VARCHAR(40)  PRIMARY KEY,
    nama VARCHAR(100) NOT NULL
);

CREATE TABLE kecamatan (
    id         VARCHAR(40)  PRIMARY KEY,
    nama       VARCHAR(100) NOT NULL,
    wilayah_id VARCHAR(40)  NOT NULL REFERENCES wilayah (id)
);

-- Metadata arsip Sentinel-2 per kecamatan 
-- Status anchor ditentukan per wilayah dari cloud_pct_buffer_wilayah
CREATE TABLE citra_arsip (
    id                       SERIAL      PRIMARY KEY,
    wilayah_id               VARCHAR(40) NOT NULL REFERENCES wilayah (id),
    kecamatan_id             VARCHAR(40) NOT NULL REFERENCES kecamatan (id),
    tanggal                  DATE        NOT NULL,
    path_folder              TEXT        NOT NULL,
    cloud_pct_scl            REAL,          -- % awan kecamatan, SCL saja
    cloud_pct_buffer         REAL,          -- % awan kecamatan setelah buffer 30 m
    cloud_pct_buffer_wilayah REAL,          -- % awan gabungan semua kecamatan di wilayah 
    is_anchor                BOOLEAN     NOT NULL DEFAULT FALSE,
    UNIQUE (kecamatan_id, tanggal)
);

CREATE TABLE model_klasifikasi (
    id             SERIAL       PRIMARY KEY,
    nama           VARCHAR(100) NOT NULL,
    kecamatan_id   VARCHAR(40)  NOT NULL REFERENCES kecamatan (id),
    path_model     TEXT         NOT NULL,
    path_config    TEXT         NOT NULL,
    akurasi        REAL,
    presisi        REAL,
    recall         REAL,
    f1             REAL,
    evaluasi       JSONB        NOT NULL,
    hyperparameter JSONB        NOT NULL,
    tanggal_citra  DATE[]       NOT NULL,
    dibuat_oleh    VARCHAR(100),
    dibuat_pada    TIMESTAMP    NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX model_klasifikasi_nama_unik ON model_klasifikasi (lower(nama));

CREATE TABLE hasil_klasifikasi (
    id                  SERIAL      PRIMARY KEY,
    model_id            INTEGER     NOT NULL REFERENCES model_klasifikasi (id) ON DELETE CASCADE,
    kecamatan_id        VARCHAR(40) NOT NULL REFERENCES kecamatan (id),
    tanggal             DATE        NOT NULL,
    luas_terbangun      REAL        NOT NULL,  -- km²
    luas_nonterbangun   REAL        NOT NULL,  -- km²
    persen_awan         REAL,
    persen_pchip        REAL,
    persen_backward     REAL,
    persen_forward      REAL,
    persen_tidak_terisi REAL,
    path_raster         TEXT        NOT NULL,
    path_citra          TEXT,
    disimpan_oleh       VARCHAR(100),
    disimpan_pada       TIMESTAMP   NOT NULL DEFAULT now(),
    UNIQUE (model_id, kecamatan_id, tanggal)
);

-- Model ConvLSTM (satu model untuk banyak kecamatan)
CREATE TABLE model_prediksi (
    id             SERIAL       PRIMARY KEY,
    nama           VARCHAR(100) NOT NULL,
    kecamatan_ids  TEXT[]       NOT NULL,
    path_model     TEXT         NOT NULL,
    path_config    TEXT         NOT NULL,
    akurasi        REAL,
    presisi        REAL,
    recall         REAL,
    f1             REAL,
    evaluasi       JSONB        NOT NULL,
    hyperparameter JSONB        NOT NULL,
    dibuat_oleh    VARCHAR(100),
    dibuat_pada    TIMESTAMP    NOT NULL DEFAULT now()
);
CREATE UNIQUE INDEX model_prediksi_nama_unik ON model_prediksi (lower(nama));

CREATE TABLE hasil_prediksi (
    id                SERIAL      PRIMARY KEY,
    model_id          INTEGER     NOT NULL REFERENCES model_prediksi (id) ON DELETE CASCADE,
    kecamatan_id      VARCHAR(40) NOT NULL REFERENCES kecamatan (id),
    tahun             SMALLINT    NOT NULL,
    luas_terbangun    REAL        NOT NULL,  -- km², peta 10 m
    luas_nonterbangun REAL        NOT NULL,  -- km², peta 10 m
    luas_acuan        REAL,                  -- km², peta tahunan aktual terakhir (10 m)
    luas_terbangun_64 REAL,                  -- km², peta model 64x64
    path_raster       TEXT        NOT NULL,  -- GeoTIFF 10 m
    path_peta         TEXT        NOT NULL,  -- .npz peluang + biner 64x64
    disimpan_oleh     VARCHAR(100),
    disimpan_pada     TIMESTAMP   NOT NULL DEFAULT now(),
    UNIQUE (model_id, kecamatan_id, tahun)
);

-- Data wilayah dan kecamatan
INSERT INTO wilayah (id, nama) VALUES
    ('kabupaten_tangerang',    'Kabupaten Tangerang'),
    ('kota_tangerang',         'Kota Tangerang'),
    ('kota_tangerang_selatan', 'Kota Tangerang Selatan');

INSERT INTO kecamatan (id, nama, wilayah_id) VALUES
    ('balaraja',      'Balaraja',      'kabupaten_tangerang'),
    ('cikupa',        'Cikupa',        'kabupaten_tangerang'),
    ('cisauk',        'Cisauk',        'kabupaten_tangerang'),
    ('cisoka',        'Cisoka',        'kabupaten_tangerang'),
    ('curug',         'Curug',         'kabupaten_tangerang'),
    ('gunung_kaler',  'Gunung Kaler',  'kabupaten_tangerang'),
    ('jambe',         'Jambe',         'kabupaten_tangerang'),
    ('jayanti',       'Jayanti',       'kabupaten_tangerang'),
    ('kelapa_dua',    'Kelapa Dua',    'kabupaten_tangerang'),
    ('kemiri',        'Kemiri',        'kabupaten_tangerang'),
    ('kosambi',       'Kosambi',       'kabupaten_tangerang'),
    ('kresek',        'Kresek',        'kabupaten_tangerang'),
    ('kronjo',        'Kronjo',        'kabupaten_tangerang'),
    ('legok',         'Legok',         'kabupaten_tangerang'),
    ('mauk',          'Mauk',          'kabupaten_tangerang'),
    ('mekar_baru',    'Mekar Baru',    'kabupaten_tangerang'),
    ('pagedangan',    'Pagedangan',    'kabupaten_tangerang'),
    ('pakuhaji',      'Pakuhaji',      'kabupaten_tangerang'),
    ('panongan',      'Panongan',      'kabupaten_tangerang'),
    ('pasar_kemis',   'Pasar Kemis',   'kabupaten_tangerang'),
    ('rajeg',         'Rajeg',         'kabupaten_tangerang'),
    ('sepatan',       'Sepatan',       'kabupaten_tangerang'),
    ('sepatan_timur', 'Sepatan Timur', 'kabupaten_tangerang'),
    ('sindang_jaya',  'Sindang Jaya',  'kabupaten_tangerang'),
    ('solear',        'Solear',        'kabupaten_tangerang'),
    ('sukadiri',      'Sukadiri',      'kabupaten_tangerang'),
    ('sukamulya',     'Sukamulya',     'kabupaten_tangerang'),
    ('teluknaga',     'Teluknaga',     'kabupaten_tangerang'),
    ('tigaraksa',     'Tigaraksa',     'kabupaten_tangerang'),
    ('batuceper',     'Batuceper',     'kota_tangerang'),
    ('benda',         'Benda',         'kota_tangerang'),
    ('cibodas',       'Cibodas',       'kota_tangerang'),
    ('ciledug',       'Ciledug',       'kota_tangerang'),
    ('cipondoh',      'Cipondoh',      'kota_tangerang'),
    ('jatiuwung',     'Jatiuwung',     'kota_tangerang'),
    ('karang_tengah', 'Karang Tengah', 'kota_tangerang'),
    ('karawaci',      'Karawaci',      'kota_tangerang'),
    ('larangan',      'Larangan',      'kota_tangerang'),
    ('neglasari',     'Neglasari',     'kota_tangerang'),
    ('periuk',        'Periuk',        'kota_tangerang'),
    ('pinang',        'Pinang',        'kota_tangerang'),
    ('tangerang',     'Tangerang',     'kota_tangerang'),
    ('ciputat',       'Ciputat',       'kota_tangerang_selatan'),
    ('ciputat_timur', 'Ciputat Timur', 'kota_tangerang_selatan'),
    ('pamulang',      'Pamulang',      'kota_tangerang_selatan'),
    ('pondok_aren',   'Pondok Aren',   'kota_tangerang_selatan'),
    ('serpong',       'Serpong',       'kota_tangerang_selatan'),
    ('serpong_utara', 'Serpong Utara', 'kota_tangerang_selatan'),
    ('setu',          'Setu',          'kota_tangerang_selatan');
