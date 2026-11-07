"""
Dashboard Interaktif Clustering Gempa Bumi (K-Means) menggunakan Streamlit.

Cara menjalankan (di terminal / Command Prompt, pada folder yang sama dengan notebook):
    streamlit run app.py

Berkas yang dibutuhkan (dihasilkan oleh notebook):
    - Hasil_Clustering_KMeans.csv
    - Data_Siap_Clustering_Final.csv
    - Evaluasi_KMeans.csv
    - config_clustering.json
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score

st.set_page_config(
    page_title="Dashboard Clustering Gempa Bumi",
    page_icon=":earth_asia:",
    layout="wide",
)

BASE_DIR = Path(__file__).resolve().parent
RANDOM_STATE = 42
sns.set_theme(style="whitegrid")

KETERANGAN_FITUR = {
    "latitude": "Lintang episentrum gempa (derajat)",
    "longitude": "Bujur episentrum gempa (derajat)",
    "depth": "Kedalaman hiposentrum (km)",
    "mag": "Magnitudo gempa",
    "nst": "Jumlah stasiun seismik yang digunakan",
    "dmin": "Jarak horizontal ke stasiun terdekat",
    "rms": "Galat rata-rata waktu tempuh gelombang (detik)",
    "gap": "Celah azimuth antar stasiun (derajat)",
}


# ---------------------------------------------------------------------------
# Fungsi bantu: pemuatan data dan komputasi
# ---------------------------------------------------------------------------
@st.cache_data
def muat_data():
    hasil = pd.read_csv(BASE_DIR / "Hasil_Clustering_KMeans.csv")
    scaled = pd.read_csv(BASE_DIR / "Data_Siap_Clustering_Final.csv")
    evaluasi = pd.read_csv(BASE_DIR / "Evaluasi_KMeans.csv")
    config = json.loads(
        (BASE_DIR / "config_clustering.json").read_text(encoding="utf-8")
    )
    return hasil, scaled, evaluasi, config


@st.cache_data
def jalankan_kmeans(k, kolom_fitur, _scaled_values):
    """Menjalankan K-Means pada data terstandardisasi dan menghitung metrik evaluasi."""
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    label = model.fit_predict(_scaled_values)
    return {
        "label": label,
        "inertia": float(model.inertia_),
        "silhouette": float(silhouette_score(_scaled_values, label)),
        "dbi": float(davies_bouldin_score(_scaled_values, label)),
    }


@st.cache_data
def hitung_pca(kolom_fitur, _scaled_values):
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    hasil_pca = pca.fit_transform(_scaled_values)
    return hasil_pca, pca.explained_variance_ratio_


def cari_elbow(daftar_k, inertia):
    """Titik siku = titik dengan jarak terjauh ke garis penghubung titik awal dan akhir kurva."""
    k_arr = np.array(daftar_k, dtype=float)
    i_arr = np.array(inertia, dtype=float)
    k_norm = (k_arr - k_arr.min()) / (k_arr.max() - k_arr.min())
    i_norm = (i_arr - i_arr.min()) / (i_arr.max() - i_arr.min())
    dx, dy = k_norm[-1] - k_norm[0], i_norm[-1] - i_norm[0]
    jarak = np.abs(dy * (k_norm - k_norm[0]) - dx * (i_norm - i_norm[0])) / np.hypot(dx, dy)
    return int(k_arr[np.argmax(jarak)])


def eta_kuadrat(nilai, label):
    """Proporsi variansi suatu fitur yang dijelaskan oleh pembagian klaster (0 s.d. 1)."""
    nilai = pd.Series(np.asarray(nilai, dtype=float))
    label = pd.Series(np.asarray(label))
    rata = nilai.mean()
    ss_total = ((nilai - rata) ** 2).sum()
    if ss_total == 0:
        return 0.0
    g = nilai.groupby(label)
    ss_antar = (g.count() * (g.mean() - rata) ** 2).sum()
    return float(ss_antar / ss_total)


# ---------------------------------------------------------------------------
# Fungsi bantu: penafsiran angka menjadi kalimat
# ---------------------------------------------------------------------------
def kategori_skew(nilai):
    a = abs(nilai)
    arah = "kanan" if nilai > 0 else "kiri"
    if a < 0.5:
        return "relatif simetris"
    if a < 1:
        return f"menceng sedang ke {arah}"
    return f"menceng kuat ke {arah}"


def persen_outlier_iqr(seri):
    q1, q3 = seri.quantile([0.25, 0.75])
    iqr = q3 - q1
    return float(((seri < q1 - 1.5 * iqr) | (seri > q3 + 1.5 * iqr)).mean() * 100)


def kekuatan_korelasi(r):
    a = abs(r)
    if a < 0.1:
        return "sangat lemah"
    if a < 0.3:
        return "lemah"
    if a < 0.5:
        return "sedang"
    if a < 0.7:
        return "kuat"
    return "sangat kuat"


def kategori_silhouette(nilai):
    if nilai >= 0.7:
        return "struktur klaster sangat kuat"
    if nilai >= 0.5:
        return "struktur klaster baik dan cukup jelas"
    if nilai >= 0.25:
        return "struktur klaster lemah sampai sedang (klaster saling berdekatan)"
    return "struktur klaster lemah dan banyak tumpang tindih"


def kategori_pemisahan(eta):
    if eta >= 0.6:
        return "terpisah jelas"
    if eta >= 0.3:
        return "terpisah cukup baik"
    return "banyak tumpang tindih"


def kategori_kedalaman(d):
    if d < 70:
        return "dangkal (< 70 km)"
    if d <= 300:
        return "menengah (70-300 km)"
    return "dalam (> 300 km)"


def tampilkan_analisis(baris):
    """Menampilkan kotak analisis berisi daftar poin."""
    st.info("**Analisis**\n\n" + "\n".join(f"- {b}" for b in baris))


# ---------------------------------------------------------------------------
# Memuat berkas
# ---------------------------------------------------------------------------
berkas_wajib = [
    "Hasil_Clustering_KMeans.csv",
    "Data_Siap_Clustering_Final.csv",
    "Evaluasi_KMeans.csv",
    "config_clustering.json",
]
berkas_hilang = [b for b in berkas_wajib if not (BASE_DIR / b).exists()]
if berkas_hilang:
    st.error(
        "Berkas berikut belum ditemukan: "
        + ", ".join(berkas_hilang)
        + ". Jalankan seluruh sel pada notebook terlebih dahulu, "
        "lalu pastikan app.py berada di folder yang sama."
    )
    st.stop()

hasil, scaled, evaluasi, config = muat_data()
FITUR_AKTIF = config["final_features"]
FITUR_ASLI = config["original_features"]
NAMA_SET = config["feature_set"]
K_DEFAULT = int(config["k_optimal"])

# ---------------------------------------------------------------------------
# Panel samping (opsi interaktif)
# ---------------------------------------------------------------------------
st.sidebar.header("Pengaturan Interaktif")

k_min = int(evaluasi["K"].min())
k_max = int(evaluasi["K"].max())
k_pilihan = st.sidebar.slider(
    "Jumlah klaster (K)",
    min_value=k_min,
    max_value=k_max,
    value=min(max(K_DEFAULT, k_min), k_max),
    help="Mengubah K akan menjalankan ulang K-Means pada fitur yang digunakan.",
)
st.sidebar.caption(f"K hasil notebook (Elbow Method): {K_DEFAULT}")

fitur_visual = st.sidebar.multiselect(
    "Fitur yang divisualisasikan",
    options=FITUR_ASLI,
    default=FITUR_ASLI,
    help="Berlaku untuk histogram/boxplot, heatmap korelasi, dan grafik karakteristik klaster.",
)
if not fitur_visual:
    st.warning("Pilih minimal satu fitur pada panel samping.")
    st.stop()

st.sidebar.markdown("**Fitur yang dipakai model K-Means**")
st.sidebar.write(", ".join(FITUR_AKTIF))

hitung = jalankan_kmeans(k_pilihan, tuple(FITUR_AKTIF), scaled[FITUR_AKTIF].values)
data = hasil.copy()
data["cluster"] = hitung["label"]
semua_cluster = sorted(data["cluster"].unique())

cluster_terpilih = st.sidebar.multiselect(
    "Klaster yang ditampilkan", options=semua_cluster, default=semua_cluster
)
if not cluster_terpilih:
    st.warning("Pilih minimal satu klaster pada panel samping.")
    st.stop()
data_tampil = data[data["cluster"].isin(cluster_terpilih)]

warna = sns.color_palette("tab10" if k_pilihan <= 10 else "tab20", n_colors=k_pilihan)
palet = {c: warna[i] for i, c in enumerate(semua_cluster)}

# ---------------------------------------------------------------------------
# Judul dan deskripsi proyek
# ---------------------------------------------------------------------------
st.title("Dashboard Clustering Gempa Bumi Global dengan K-Means")
st.markdown(
    f"""
Proyek ini mengelompokkan **{len(data):,} kejadian gempa bumi** di seluruh dunia menggunakan
algoritma **K-Means Clustering**. Pengelompokan dilakukan berdasarkan fitur
**{", ".join(FITUR_AKTIF)}** (kombinasi fitur: *{NAMA_SET}*), di mana koordinat `x, y, z` adalah
representasi 3D dari latitude dan longitude agar posisi di permukaan bumi terukur dengan benar.
Dashboard ini menampilkan ringkasan data, eksplorasi fitur, penentuan jumlah klaster,
hasil pengelompokan, dan karakteristik setiap klaster. Jumlah klaster dan fitur yang
divisualisasikan dapat diubah melalui panel samping.
"""
)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Jumlah Klaster (K)", k_pilihan)
m2.metric("Inertia (SSE)", f"{hitung['inertia']:,.1f}")
m3.metric("Silhouette Score", f"{hitung['silhouette']:.4f}")
m4.metric("Davies-Bouldin Index", f"{hitung['dbi']:.4f}")

tab_data, tab_eda, tab_elbow, tab_visual, tab_karakter = st.tabs(
    [
        "1. Ringkasan Dataset",
        "2. Distribusi dan Korelasi",
        "3. Elbow dan Silhouette",
        "4. Hasil Clustering",
        "5. Karakteristik Klaster",
    ]
)

# ===========================================================================
# TAB 1: Ringkasan dataset
# ===========================================================================
with tab_data:
    st.subheader("Ringkasan Dataset")

    jumlah_missing = int(data[FITUR_ASLI].isnull().sum().sum())
    jumlah_duplikat = int(data.drop(columns=["cluster"]).duplicated().sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Data", f"{len(data):,}")
    c2.metric("Jumlah Kolom", data.shape[1] - 1)
    c3.metric("Fitur Numerik Dianalisis", len(FITUR_ASLI))
    c4.metric("Missing Value (fitur)", jumlah_missing)

    st.markdown("**Daftar fitur numerik**")
    st.dataframe(
        pd.DataFrame(
            {
                "Fitur": FITUR_ASLI,
                "Keterangan": [KETERANGAN_FITUR.get(f, "-") for f in FITUR_ASLI],
                "Dipakai langsung oleh model": [
                    "Ya" if f in FITUR_AKTIF else "Tidak (latitude/longitude diwakili x, y, z)"
                    if f in ("latitude", "longitude") else "Tidak"
                    for f in FITUR_ASLI
                ],
            }
        )
    )

    st.markdown("**Statistik deskriptif**")
    deskripsi = data[FITUR_ASLI].describe().T
    st.dataframe(deskripsi.round(3))

    st.markdown("**Contoh data (10 baris pertama)**")
    kolom_contoh = [
        c for c in ["place", "time", "latitude", "longitude", "depth", "mag"] if c in data.columns
    ]
    st.dataframe(data[kolom_contoh + ["cluster"]].head(10))

    rentang = (deskripsi["max"] - deskripsi["min"]).sort_values()
    tampilkan_analisis(
        [
            f"Dataset memuat {len(data):,} kejadian gempa dengan {len(FITUR_ASLI)} fitur numerik "
            f"yang dianalisis. Terdapat {jumlah_missing} nilai kosong pada fitur tersebut dan "
            f"{jumlah_duplikat} baris duplikat.",
            f"Skala antar fitur sangat berbeda: rentang nilai terkecil adalah fitur "
            f"**{rentang.index[0]}** ({rentang.iloc[0]:,.2f}), sedangkan terbesar adalah "
            f"**{rentang.index[-1]}** ({rentang.iloc[-1]:,.2f}). Karena itu seluruh fitur "
            f"distandarisasi sebelum clustering agar tidak ada fitur yang mendominasi jarak "
            f"Euclidean pada K-Means.",
            *[
                f"Fitur **{f}** memiliki rata-rata {deskripsi.loc[f, 'mean']:,.2f} dan median "
                f"{deskripsi.loc[f, '50%']:,.2f}"
                + (
                    ". Rata-rata yang jauh lebih besar daripada median menandakan sebaran "
                    "dengan ekor nilai besar di sisi kanan."
                    if deskripsi.loc[f, "mean"] > 1.3 * deskripsi.loc[f, "50%"] > 0
                    else "."
                )
                for f in ["depth"]
                if f in deskripsi.index
            ],
        ]
    )

# ===========================================================================
# TAB 2: Distribusi fitur dan korelasi
# ===========================================================================
with tab_eda:
    st.subheader("Distribusi Fitur")
    jenis_dist = st.radio("Jenis grafik", ["Histogram", "Boxplot"], horizontal=True)

    n_fitur = len(fitur_visual)
    n_kolom = min(3, n_fitur)
    n_baris = int(np.ceil(n_fitur / n_kolom))
    fig, axes = plt.subplots(n_baris, n_kolom, figsize=(5.2 * n_kolom, 3.6 * n_baris))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, fitur_visual):
        if jenis_dist == "Histogram":
            sns.histplot(data=data, x=f, kde=True, ax=ax, color="steelblue")
            ax.set_ylabel("Frekuensi")
        else:
            sns.boxplot(x=data[f], ax=ax, color="lightsteelblue")
        ax.set_title(f"{jenis_dist} {f}")
    for ax in axes[n_fitur:]:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    baris_dist = []
    for f in fitur_visual:
        s = data[f]
        baris_dist.append(
            f"**{f}**: rata-rata {s.mean():,.2f}, median {s.median():,.2f}, distribusi "
            f"{kategori_skew(s.skew())} (skewness = {s.skew():.2f}); sekitar "
            f"{persen_outlier_iqr(s):.1f}% data berada di luar batas IQR (nilai ekstrem)."
        )
    baris_dist.append(
        "Fitur yang menceng kuat dan memiliki banyak nilai ekstrem perlu ditransformasi "
        "(log) dan ditangani (winsorizing) sebelum clustering, karena K-Means sensitif "
        "terhadap pencilan. Hal ini sudah dilakukan pada tahap pra-pemrosesan di notebook."
    )
    tampilkan_analisis(baris_dist)

    st.subheader("Korelasi Antar Fitur")
    if len(fitur_visual) < 2:
        st.warning("Pilih minimal dua fitur pada panel samping untuk menampilkan korelasi.")
    else:
        korelasi = data[fitur_visual].corr()
        fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(fitur_visual)), max(5, 0.9 * len(fitur_visual))))
        sns.heatmap(korelasi, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
        ax.set_title("Correlation Heatmap Antar Fitur")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        pasangan = (
            korelasi.where(np.triu(np.ones(korelasi.shape), k=1).astype(bool))
            .stack()
            .rename("r")
            .reset_index()
        )
        pasangan["abs"] = pasangan["r"].abs()
        pasangan = pasangan.sort_values("abs", ascending=False)
        teratas = pasangan.iloc[0]
        kuat = pasangan[pasangan["abs"] >= 0.5]

        baris_kor = [
            f"Pasangan fitur dengan hubungan terkuat adalah **{teratas.iloc[0]}** dan "
            f"**{teratas.iloc[1]}** (r = {teratas['r']:.2f}, tergolong "
            f"{kekuatan_korelasi(teratas['r'])}, "
            f"{'searah' if teratas['r'] > 0 else 'berlawanan arah'}).",
        ]
        if len(kuat) == 0:
            baris_kor.append(
                "Tidak ada pasangan fitur dengan korelasi kuat (|r| >= 0,5), sehingga setiap "
                "fitur membawa informasi yang relatif berbeda dan tidak ada yang sangat redundan."
            )
        else:
            baris_kor.append(
                f"Terdapat {len(kuat)} pasangan fitur dengan korelasi kuat (|r| >= 0,5). Fitur "
                f"yang saling berkorelasi tinggi membawa informasi yang tumpang tindih sehingga "
                f"dapat membuat sebuah aspek data berbobot ganda pada perhitungan jarak K-Means."
            )
        baris_kor.append(
            "Korelasi yang mendekati nol menunjukkan tidak adanya hubungan linear, bukan "
            "berarti tidak ada hubungan sama sekali."
        )
        tampilkan_analisis(baris_kor)

# ===========================================================================
# TAB 3: Elbow dan silhouette
# ===========================================================================
with tab_elbow:
    st.subheader("Penentuan Jumlah Klaster Optimal")
    eval_set = evaluasi[evaluasi["Fitur"] == NAMA_SET].sort_values("K").reset_index(drop=True)
    k_elbow = cari_elbow(eval_set["K"], eval_set["Inertia"])
    k_sil_terbaik = int(eval_set.loc[eval_set["Silhouette"].idxmax(), "K"])

    kiri, kanan = st.columns(2)
    with kiri:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eval_set["K"], eval_set["Inertia"], marker="o")
        ax.axvline(k_pilihan, color="crimson", linestyle="--", label=f"K dipilih = {k_pilihan}")
        if k_elbow != k_pilihan:
            ax.axvline(k_elbow, color="gray", linestyle=":", label=f"Titik siku = {k_elbow}")
        ax.set_title("Elbow Method (Inertia)")
        ax.set_xlabel("Jumlah Klaster (K)")
        ax.set_ylabel("Inertia")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
    with kanan:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(eval_set["K"], eval_set["Silhouette"], marker="o", color="darkgreen")
        ax.axvline(k_pilihan, color="crimson", linestyle="--", label=f"K dipilih = {k_pilihan}")
        ax.set_title("Silhouette Score")
        ax.set_xlabel("Jumlah Klaster (K)")
        ax.set_ylabel("Silhouette Score")
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)

    inertia_k = eval_set.set_index("K")["Inertia"]
    sil_k = eval_set.set_index("K")["Silhouette"]
    baris_elbow = [
        f"**Elbow Method:** inertia turun tajam pada K kecil lalu melandai. Titik siku "
        f"(penurunan mulai tidak signifikan) berada pada **K = {k_elbow}**."
    ]
    if k_pilihan - 1 in inertia_k.index:
        turun_sebelum = (1 - inertia_k[k_pilihan] / inertia_k[k_pilihan - 1]) * 100
        baris_elbow.append(
            f"Menambah klaster dari K = {k_pilihan - 1} ke K = {k_pilihan} menurunkan inertia "
            f"sebesar {turun_sebelum:.1f}%."
        )
    if k_pilihan + 1 in inertia_k.index:
        turun_sesudah = (1 - inertia_k[k_pilihan + 1] / inertia_k[k_pilihan]) * 100
        baris_elbow.append(
            f"Menambah klaster dari K = {k_pilihan} ke K = {k_pilihan + 1} hanya menurunkan "
            f"inertia sebesar {turun_sesudah:.1f}%, sehingga "
            + ("manfaat tambahan relatif kecil." if turun_sesudah < 10 else
               "penambahan klaster masih memberi perbaikan yang cukup berarti.")
        )
    baris_elbow.append(
        f"**Silhouette Score:** nilai tertinggi pada rentang K = {int(eval_set['K'].min())} "
        f"sampai {int(eval_set['K'].max())} adalah {sil_k.max():.4f} pada K = {k_sil_terbaik}. "
        f"Pada K = {k_pilihan}, nilainya {sil_k[k_pilihan]:.4f} (data sampel pada tabel "
        f"evaluasi) dan {hitung['silhouette']:.4f} (seluruh data), tergolong "
        f"{kategori_silhouette(hitung['silhouette'])}."
    )
    if k_pilihan == k_elbow:
        baris_elbow.append("K yang dipilih sama dengan titik siku, yaitu keseimbangan antara kekompakan klaster dan jumlah klaster yang tidak berlebihan.")
    else:
        baris_elbow.append(
            f"K yang dipilih ({k_pilihan}) berbeda dari titik siku (K = {k_elbow}). Nilai K yang "
            "lebih besar membuat klaster lebih kompak tetapi lebih terfragmentasi, sedangkan "
            "yang lebih kecil membuat klaster lebih umum."
        )
    baris_elbow.append(
        "Silhouette cenderung berubah seiring K sehingga sebaiknya dibaca bersama kurva elbow, "
        "bukan sebagai satu-satunya penentu K."
    )
    tampilkan_analisis(baris_elbow)

    st.markdown("**Tabel evaluasi untuk setiap nilai K**")
    st.dataframe(eval_set.drop(columns=["Fitur"]).round(4))

# ===========================================================================
# TAB 4: Hasil clustering
# ===========================================================================
with tab_visual:
    st.subheader("Visualisasi Hasil Clustering")

    pilihan_sumbu = [c for c in FITUR_ASLI]
    default_x = pilihan_sumbu.index("longitude") if "longitude" in pilihan_sumbu else 0
    default_y = pilihan_sumbu.index("latitude") if "latitude" in pilihan_sumbu else min(1, len(pilihan_sumbu) - 1)
    sx, sy = st.columns(2)
    sumbu_x = sx.selectbox("Fitur sumbu X", pilihan_sumbu, index=default_x)
    sumbu_y = sy.selectbox("Fitur sumbu Y", pilihan_sumbu, index=default_y)

    fig, ax = plt.subplots(figsize=(11, 6))
    sns.scatterplot(
        data=data_tampil, x=sumbu_x, y=sumbu_y, hue="cluster",
        palette=palet, alpha=0.7, s=18, ax=ax,
    )
    pusat = data_tampil.groupby("cluster")[[sumbu_x, sumbu_y]].median()
    for c, baris in pusat.iterrows():
        ax.text(
            baris[sumbu_x], baris[sumbu_y], str(c), fontsize=12, fontweight="bold",
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.85),
        )
    ax.set_title(f"Sebaran Klaster: {sumbu_x} vs {sumbu_y} (K = {k_pilihan})")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

    eta_x = eta_kuadrat(data[sumbu_x], data["cluster"])
    eta_y = eta_kuadrat(data[sumbu_y], data["cluster"])
    lebih_baik = sumbu_x if eta_x >= eta_y else sumbu_y
    baris_scatter = [
        f"Setiap titik adalah satu gempa dan warna menunjukkan label klaster; angka pada kotak "
        f"putih adalah label klaster pada posisi median kelompoknya.",
        f"Pada sumbu **{sumbu_x}**, pembagian klaster menjelaskan {eta_x * 100:.1f}% variasi "
        f"(klaster {kategori_pemisahan(eta_x)}). Pada sumbu **{sumbu_y}**, nilainya "
        f"{eta_y * 100:.1f}% (klaster {kategori_pemisahan(eta_y)}). Sumbu yang lebih "
        f"membedakan klaster adalah **{lebih_baik}**.",
    ]
    tidak_dipakai = [
        s for s in (sumbu_x, sumbu_y)
        if s not in FITUR_AKTIF and s not in ("latitude", "longitude")
    ]
    if tidak_dipakai:
        baris_scatter.append(
            f"Fitur {', '.join(tidak_dipakai)} tidak digunakan dalam pembentukan klaster, sehingga "
            "klaster tampak bercampur pada sumbu tersebut. Itu wajar dan bukan berarti klaster salah."
        )
    else:
        baris_scatter.append(
            "Tumpang tindih visual dapat terjadi karena klaster dibentuk dari beberapa dimensi "
            "sekaligus (termasuk kedalaman), sedangkan grafik ini hanya menampilkan dua dimensi."
        )
    tampilkan_analisis(baris_scatter)

    st.markdown("---")
    st.subheader("Proyeksi PCA 2 Dimensi")
    nilai_pca, variansi = hitung_pca(tuple(FITUR_AKTIF), scaled[FITUR_AKTIF].values)
    df_pca = pd.DataFrame(nilai_pca, columns=["PC1", "PC2"])
    df_pca["cluster"] = data["cluster"].values
    df_pca = df_pca[df_pca["cluster"].isin(cluster_terpilih)]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    sns.scatterplot(
        data=df_pca, x="PC1", y="PC2", hue="cluster", palette=palet, alpha=0.7, s=18, ax=ax
    )
    ax.set_xlabel(f"PC1 ({variansi[0] * 100:.2f}%)")
    ax.set_ylabel(f"PC2 ({variansi[1] * 100:.2f}%)")
    ax.set_title("Visualisasi Klaster Menggunakan PCA")
    ax.legend(title="Cluster", bbox_to_anchor=(1.01, 1), loc="upper left")
    st.pyplot(fig)
    plt.close(fig)

    total_var = variansi.sum() * 100
    tampilkan_analisis(
        [
            f"PCA meringkas {len(FITUR_AKTIF)} fitur model menjadi dua sumbu. PC1 menjelaskan "
            f"{variansi[0] * 100:.1f}% dan PC2 menjelaskan {variansi[1] * 100:.1f}% variasi data "
            f"(total {total_var:.1f}%).",
            (
                "Karena dua komponen sudah menjelaskan sebagian besar variasi, grafik ini cukup "
                "mewakili struktur data dan pemisahan klaster pada grafik dapat dipercaya."
                if total_var >= 75
                else "Sisa variasi yang tidak tergambar pada dua sumbu ini dapat membuat klaster yang "
                "sebenarnya terpisah tampak saling menimpa, sehingga grafik ini dibaca sebagai "
                "gambaran kasar."
            ),
            f"Dengan Silhouette Score {hitung['silhouette']:.2f}, klaster memiliki "
            f"{kategori_silhouette(hitung['silhouette'])}.",
        ]
    )

    st.markdown("---")
    st.subheader("Jumlah Data pada Setiap Klaster")
    jumlah = data["cluster"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(jumlah.index.astype(str), jumlah.values, color=[palet[c] for c in jumlah.index])
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Jumlah Data")
    ax.set_title("Jumlah Data per Klaster")
    st.pyplot(fig)
    plt.close(fig)

    terbesar, terkecil = jumlah.idxmax(), jumlah.idxmin()
    tampilkan_analisis(
        [
            f"Klaster terbesar adalah **klaster {terbesar}** dengan {jumlah.max():,} data "
            f"({jumlah.max() / len(data) * 100:.1f}%), sedangkan terkecil adalah **klaster "
            f"{terkecil}** dengan {jumlah.min():,} data ({jumlah.min() / len(data) * 100:.1f}%).",
            f"Rasio ukuran klaster terbesar terhadap terkecil adalah {jumlah.max() / jumlah.min():.1f} kali. "
            + (
                "Ukuran klaster relatif seimbang."
                if jumlah.max() / jumlah.min() < 3
                else "Ukuran klaster cukup timpang, mencerminkan wilayah dengan aktivitas gempa yang "
                "jauh lebih padat dibanding wilayah lain."
            ),
        ]
    )

# ===========================================================================
# TAB 5: Karakteristik klaster
# ===========================================================================
with tab_karakter:
    st.subheader("Tabel Ringkasan Karakteristik Setiap Klaster")
    statistik = st.radio("Ukuran pemusatan", ["Mean", "Median"], horizontal=True)

    agregat = data.groupby("cluster")[FITUR_ASLI]
    tabel = agregat.mean() if statistik == "Mean" else agregat.median()
    tabel.insert(0, "jumlah_data", data["cluster"].value_counts().sort_index())
    st.dataframe(tabel.round(3))

    # Profil otomatis tiap klaster
    data["wilayah"] = data["place"].astype(str).str.split(",").str[-1].str.strip() if "place" in data.columns else "-"
    narasi = []
    for c in semua_cluster:
        sub = data[data["cluster"] == c]
        wilayah_top = sub["wilayah"].value_counts().head(3)
        teks_wilayah = ", ".join(f"{w} ({n / len(sub) * 100:.0f}%)" for w, n in wilayah_top.items())
        d_med = sub["depth"].median() if "depth" in sub.columns else np.nan
        kalimat = f"**Klaster {c}** ({len(sub):,} data, {len(sub) / len(data) * 100:.1f}%)"
        if not np.isnan(d_med):
            kalimat += f": kedalaman {kategori_kedalaman(d_med)} dengan median {d_med:.1f} km"
        if "mag" in sub.columns:
            kalimat += f", magnitudo median {sub['mag'].median():.1f}"
        if "place" in data.columns:
            kalimat += f". Wilayah dominan: {teks_wilayah}"
        narasi.append(kalimat + ".")

    tampilkan_analisis(
        [
            f"Tabel menunjukkan {statistik.lower()} setiap fitur pada tiap klaster dalam satuan "
            "aslinya sehingga mudah ditafsirkan.",
            "Median lebih tahan terhadap nilai ekstrem dibanding rata-rata. Jika keduanya sangat "
            "berbeda pada suatu fitur, klaster tersebut memiliki sebaran yang menceng.",
            *narasi,
        ]
    )

    st.markdown("---")
    st.subheader(f"{statistik} Fitur per Klaster")
    n_fitur = len(fitur_visual)
    n_kolom = min(3, n_fitur)
    n_baris = int(np.ceil(n_fitur / n_kolom))
    fig, axes = plt.subplots(n_baris, n_kolom, figsize=(5.2 * n_kolom, 3.4 * n_baris))
    axes = np.atleast_1d(axes).ravel()
    for ax, f in zip(axes, fitur_visual):
        ax.bar(tabel.index.astype(str), tabel[f].values, color=[palet[c] for c in tabel.index])
        ax.set_title(f"{statistik} {f}")
        ax.set_xlabel("Cluster")
    for ax in axes[n_fitur:]:
        ax.axis("off")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    baris_bar = []
    for f in fitur_visual:
        tinggi, rendah = tabel[f].idxmax(), tabel[f].idxmin()
        baris_bar.append(
            f"**{f}**: tertinggi pada klaster {tinggi} ({tabel[f].max():,.2f}) dan terendah pada "
            f"klaster {rendah} ({tabel[f].min():,.2f}); pembagian klaster menjelaskan "
            f"{eta_kuadrat(data[f], data['cluster']) * 100:.1f}% variasi fitur ini."
        )
    baris_bar.append(
        "Fitur dengan persentase variasi yang dijelaskan klaster tinggi adalah fitur yang paling "
        "membedakan satu klaster dari klaster lain. Fitur dengan persentase rendah relatif "
        "sama pada semua klaster."
    )
    tampilkan_analisis(baris_bar)

    st.markdown("---")
    st.subheader("Perbandingan Relatif Antar Klaster (Heatmap)")
    rata_klaster = data.groupby("cluster")[fitur_visual].mean()
    z = (rata_klaster - data[fitur_visual].mean()) / data[fitur_visual].std()

    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(fitur_visual)), max(3.5, 0.6 * len(z) + 2)))
    sns.heatmap(z, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Rata-rata Fitur per Klaster (z-score terhadap seluruh data)")
    ax.set_ylabel("Cluster")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    baris_z = [
        "Nilai positif (merah) berarti klaster memiliki rata-rata fitur di atas rata-rata seluruh "
        "data, nilai negatif (biru) berarti di bawahnya. Satuan nilai adalah simpangan baku."
    ]
    for c in z.index:
        f_utama = z.loc[c].abs().idxmax()
        nilai = z.loc[c, f_utama]
        baris_z.append(
            f"Klaster {c} paling menonjol pada **{f_utama}** (z = {nilai:+.2f}), yaitu "
            f"{'lebih tinggi' if nilai > 0 else 'lebih rendah'} dari rata-rata keseluruhan."
        )
    tampilkan_analisis(baris_z)

    st.download_button(
        "Unduh hasil clustering (CSV)",
        data=data.drop(columns=["wilayah"]).to_csv(index=False).encode("utf-8"),
        file_name=f"hasil_clustering_k{k_pilihan}.csv",
        mime="text/csv",
    )