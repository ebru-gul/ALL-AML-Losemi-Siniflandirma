# -*- coding: utf-8 -*-
"""
Ara rapor düzeltmeleri için ek analizler (ALL/AML dönem projesi).

  1) Holm düzeltmesi   : PC1–PC5 eğitim/test, sınıf içi eğitim/test ve doku (KI/PK) testleri
  2) Duyarlılık analizi: 238 ortak gen ve r = 0,61 sonucunu ETİKETSİZ prob seçimiyle yeniden hesaplar
                         (en yüksek varyans, en yüksek ortalama, probların ortalaması)
  3) GSE13159 log öncesi/sonrası dağılım şekli + beta(1,2; 3) moment karşılaştırması
  4) GSE13159 örnek düzeyinde aykırı değer kontrolünün İLK taslağı (tüm örneklerde PCA–Mahalanobis,
     ortalama korelasyon; dosyanın sonundaki ornek_aykiri_kontrolu). RAPORDA KULLANILMADI:
     Bölüm IV-G'deki sonuç (209 / 4 örnek), sınıf içi MCD tabanlı sürümle elde edildi; o sürüm
     notebooks/GSE13159_EDA_v3.ipynb defterinin son hücresindedir ve çıktısı
     results/GSE13159_aykiri_ornek_ozet.json dosyasındadır. Golub karşılığı: src/golub_ek_betimsel_aykiri.py

Kullanım (defterden):
    from ek_analizler import *
    # --- 1) Holm ---
    p = pc_welch(pc_skorlari_golub[:, :5], set_etiketi == "test")          # dict PC1..PC5 -> p
    print(holm_tablosu(p))
    # --- 2) Duyarlılık ---
    tablo = duyarlilik_analizi(Xg, yg, gpl80_map, Xm, ym, gpl570_map)
    print(tablo); print(latex_satirlari(tablo))
    # --- 3) Log şekli ---
    gse_log_sekli(Xm_raw, c=<defterde kullandığın c>, dosya="sekil12_log.pdf")

Girdiler:
    Xg : Golub, prob x örnek DataFrame (20–16.000 kırpma + log2 SONRASI, sıfır varyanslılar atılmış)
    Xm : GSE13159, prob x örnek DataFrame (seri matris, 45 boş prob atılmış; log YOK)
    yg, ym : 1 = ALL, 0 = AML (sütun sırasıyla aynı)
    gpl80_map, gpl570_map : pd.Series, index = prob kimliği, değer = gen sembolü ("A /// B" olabilir)

Yalnızca numpy, pandas, scipy, matplotlib gerekir. Tohum: 42.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

TOHUM = 42


# ---------------------------------------------------------------------------
# 1) Çoklu test düzeltmeleri
# ---------------------------------------------------------------------------
def holm(p):
    """Holm–Bonferroni düzeltilmiş p değerleri (aynı sırayla döner)."""
    p = np.asarray(p, float)
    m = p.size
    sira = np.argsort(p)
    duz = np.empty(m)
    enb = 0.0
    for i, j in enumerate(sira):
        enb = max(enb, (m - i) * p[j])
        duz[j] = min(1.0, enb)
    return duz


def bh(p):
    """Benjamini–Hochberg q değerleri."""
    p = np.asarray(p, float)
    m = p.size
    sira = np.argsort(p)[::-1]
    q = np.empty(m)
    enk = 1.0
    for k, j in enumerate(sira):
        enk = min(enk, p[j] * m / (m - k))
        q[j] = enk
    return q


def pc_welch(skorlar, grup, maske=None, adlar=None):
    """Her bileşende grup True / False arasında Welch t-testi p değeri."""
    skorlar = np.asarray(skorlar, float)
    grup = np.asarray(grup, bool)
    if maske is not None:
        maske = np.asarray(maske, bool)
        skorlar, grup = skorlar[maske], grup[maske]
    adlar = adlar or [f"PC{i+1}" for i in range(skorlar.shape[1])]
    return {a: stats.ttest_ind(skorlar[grup, i], skorlar[~grup, i], equal_var=False).pvalue
            for i, a in enumerate(adlar)}


def holm_tablosu(pdict, alfa=0.05):
    t = pd.DataFrame({"p": pd.Series(pdict)})
    t["p_Holm"] = holm(t["p"].values)
    t["anlamlı (Holm)"] = t["p_Holm"] < alfa
    return t


# ---------------------------------------------------------------------------
# 2) Prob düzeyi istatistikler ve gen düzeyine indirgeme
# ---------------------------------------------------------------------------
def prob_istatistikleri(X: pd.DataFrame, y) -> pd.DataFrame:
    """Welch p, BH q, Cohen d (ALL - AML, havuzlanmış s.s.), varyans ve ortalama."""
    y = np.asarray(y).astype(bool)
    A, B = X.values[:, y], X.values[:, ~y]
    n1, n2 = A.shape[1], B.shape[1]
    m1, m2 = A.mean(1), B.mean(1)
    v1, v2 = A.var(1, ddof=1), B.var(1, ddof=1)
    sp = np.sqrt(((n1 - 1) * v1 + (n2 - 1) * v2) / (n1 + n2 - 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (m1 - m2) / sp
    p = stats.ttest_ind(A, B, axis=1, equal_var=False).pvalue
    out = pd.DataFrame({"d": d, "p": p,
                        "var": X.values.var(1, ddof=1),
                        "ort": X.values.mean(1)}, index=X.index)
    out = out[np.isfinite(out["p"]) & (out["var"] > 0)]
    out["q"] = bh(out["p"].values)
    return out


def _patlat(harita: pd.Series) -> pd.Series:
    """'A /// B' gibi çok genli probları her gen için ayrı satıra açar."""
    s = harita.dropna().astype(str).str.split(r"\s*///\s*")
    s = s.explode().str.strip()
    return s[(s != "") & (s != "---")]


def gen_duzeyi(X: pd.DataFrame, y, harita: pd.Series, kural: str) -> pd.DataFrame:
    """
    kural:
      'maks_d'   : en yüksek |d|'li prob   (rapordaki mevcut, ETİKETE BAĞLI seçim)
      'maks_var' : en yüksek varyanslı prob (etiketsiz)
      'maks_ort' : en yüksek ortalama ifadeli prob (etiketsiz)
      'ortalama' : genin tüm problarının ortalaması alınır, test yeniden yapılır (etiketsiz)
    Dönen: gen -> d, q
    """
    ist = prob_istatistikleri(X, y)
    eslesme = _patlat(harita)
    eslesme = eslesme[eslesme.index.isin(ist.index)]
    df = ist.loc[eslesme.index].copy()
    df["gen"] = eslesme.values

    if kural == "ortalama":
        Xg = X.loc[eslesme.index].copy()
        Xg["gen"] = eslesme.values
        Xg = Xg.groupby("gen").mean()
        g = prob_istatistikleri(Xg, y)
        # q burada gen düzeyinde yeniden hesaplanır
        return g[["d", "q"]]

    anahtar = {"maks_d": df["d"].abs(), "maks_var": df["var"], "maks_ort": df["ort"]}[kural]
    df["_k"] = anahtar.values
    sec = df.sort_values("_k", ascending=False).groupby("gen", sort=False).head(1)
    return sec.set_index("gen")[["d", "q"]]


def platformlar_arasi(ga: pd.DataFrame, gb: pd.DataFrame, q_esik=0.01, d_esik=1.0):
    ortak = ga.index.intersection(gb.index)
    a, b = ga.loc[ortak], gb.loc[ortak]
    sa = (a["q"] < q_esik) & (a["d"].abs() > d_esik)
    sb = (b["q"] < q_esik) & (b["d"].abs() > d_esik)
    N, K, n = len(ortak), int(sa.sum()), int(sb.sum())
    k = int((sa & sb).sum())
    ayni_yon = int((np.sign(a["d"][sa & sb]) == np.sign(b["d"][sa & sb])).sum())
    beklenen = K * n / N if N else np.nan
    p_hg = stats.hypergeom.sf(k - 1, N, K, n) if k > 0 else 1.0
    r = stats.pearsonr(a["d"], b["d"])[0]
    rho = stats.spearmanr(a["d"], b["d"])[0]
    return dict(ortak_gen=N, golub_ayirt=K, gse_ayirt=n, kesisim=k, ayni_yon=ayni_yon,
                beklenen=beklenen, kat=k / beklenen if beklenen else np.nan,
                p_hipergeom=p_hg, pearson_r=r, spearman_rho=rho)


def duyarlilik_analizi(Xg, yg, harita_g, Xm, ym, harita_m,
                       kurallar=("maks_d", "maks_var", "maks_ort", "ortalama"),
                       ortak_genler=None):
    """
    ortak_genler: raporda kullanılan 5.829 genlik liste verilirse karşılaştırma onunla sınırlanır
    (sayıların rapordakiyle birebir tutması için önerilir).
    """
    satirlar = {}
    for kural in kurallar:
        ga = gen_duzeyi(Xg, yg, harita_g, kural)
        gb = gen_duzeyi(Xm, ym, harita_m, kural)
        if ortak_genler is not None:
            ga = ga[ga.index.isin(ortak_genler)]
            gb = gb[gb.index.isin(ortak_genler)]
        satirlar[kural] = platformlar_arasi(ga, gb)
    return pd.DataFrame(satirlar).T


def latex_satirlari(tablo: pd.DataFrame) -> str:
    adlar = {"maks_d": "En yüksek $|d|$ (etikete bağlı; Bölüm IV-K)",
             "maks_var": "En yüksek varyans (etiketsiz)",
             "maks_ort": "En yüksek ortalama ifade (etiketsiz)",
             "ortalama": "Probların ortalaması (etiketsiz)"}

    def tr(x, n=2):
        return "—" if not np.isfinite(x) else f"{x:.{n}f}".replace(".", "{,}")

    def bilimsel(p):
        if p == 0:
            return "$<10^{-300}$"
        e = int(np.floor(np.log10(p)))
        return f"${tr(p / 10**e, 1)}\\times10^{{{e}}}$"

    out = []
    for k, s in tablo.iterrows():
        out.append(f"{adlar.get(k, k)} & {int(s.golub_ayirt)} & {int(s.gse_ayirt)} & "
                   f"{int(s.kesisim)} & {tr(s.kat, 1)} & {bilimsel(s.p_hipergeom)} & "
                   f"{int(s.ayni_yon)}/{int(s.kesisim)} & {tr(s.pearson_r)} \\\\")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# 3) GSE13159: log öncesi / sonrası dağılım
# ---------------------------------------------------------------------------
def gse_log_sekli(Xm: pd.DataFrame, c: float, dosya="sekil12_log_oncesi_sonrasi.pdf",
                  n_ornek=2_000_000):
    import matplotlib.pyplot as plt
    rng = np.random.default_rng(TOHUM)
    v = Xm.values.ravel()
    v = v[np.isfinite(v)]
    if v.size > n_ornek:
        v = rng.choice(v, n_ornek, replace=False)
    lv = np.log2(v + c)
    b = stats.beta(1.2, 3)
    print(f"Ham     : ort {v.mean():.4f}  s.s. {v.std(ddof=1):.4f}  medyan {np.median(v):.4f}  "
          f"çarpıklık {stats.skew(v):.3f}")
    print(f"beta(1,2;3) kuramsal: ort {b.mean():.4f}  s.s. {b.std():.4f}  medyan {b.median():.4f}  "
          f"çarpıklık {float(b.stats('s')):.3f}")
    print(f"log2(x+{c}): çarpıklık {stats.skew(lv):.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    ax[0].hist(v, bins=100, density=True, color="#4C72B0", alpha=.75, label="veri")
    xx = np.linspace(0, 1, 400)
    ax[0].plot(xx, b.pdf(xx), "k--", lw=1.2, label="beta(1,2; 3)")
    ax[0].set(title=f"(a) DQN değerleri (çarpıklık {stats.skew(v):.2f})".replace(".", ","),
              xlabel="İfade (0–1)", ylabel="Yoğunluk")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].hist(lv, bins=100, density=True, color="#DD8452", alpha=.75)
    ax[1].set(title=f"(b) log$_2$(x + {c}) (çarpıklık {stats.skew(lv):.2f})".replace(".", ","),
              xlabel="log$_2$ ifade", ylabel="Yoğunluk")
    fig.tight_layout()
    fig.savefig(dosya, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Raporda zaten yazan p değerleriyle Holm (kodla yeniden üretip doğrula)
# ---------------------------------------------------------------------------
def rapordaki_degerlerle_holm():
    """ek_analiz_sonuclar.json'daki Welch p'leri (aile = PC1–PC5)."""
    aileler = {
        "Golub eğitim/test": [0.004235, 0.392800, 0.490869, 0.000136, 0.023216],
        "Golub yalnız ALL": [0.004582, 0.730161, 0.823240, 0.020504, 0.000618],
        "Golub yalnız AML": [0.168206, 0.962559, 0.035542, 0.001598, 0.391976],
        "GSE doku ALL": [0.002367, 0.054642, 0.277480, 0.001623, 0.529872],
        "GSE doku AML": [0.809457, 0.486782, 0.011910, 0.228855, 0.037245],
    }
    for ad, p in aileler.items():
        print(ad); print(holm_tablosu(dict(zip([f"PC{i}" for i in range(1, 6)], p))).round(4), "\n")


# =============================================================================
# EK: GSE13159 – ÖRNEK DÜZEYİNDE AYKIRI DEĞER KONTROLÜ (Bölüm IV-G)
# Bu blok ek_analizler.py dosyasının sonuna eklenecektir.
# DURUM: İlk taslak; gerçek veride çalıştırılmadı ve raporda KULLANILMADI. Rapordaki sonuç
#        sınıf içi MCD tabanlı sürümden gelir (notebooks/GSE13159_EDA_v3.ipynb, son hücre).
#
# Yapılanlar (seed = 42):
#   1) En değişken 5.000 prob üzerinde PCA; ilk 10 bileşende her örneğin
#      Mahalanobis uzaklığı (D^2). D^2 > ki-kare(sd=10) dağılımının 0,999
#      kantili olan örnekler sayılır.
#   2) Örnekler arası Pearson korelasyon matrisinde her örneğin diğer
#      örneklerle ortalama korelasyonu; medyan − 3×MAD altında kalan
#      örnekler sayılır.
#   3) Bulunan örnekler ÇIKARILMAZ; yalnızca sayıları ile sınıf (ALL/AML) ve
#      doku (kemik iliği / periferik kan) dağılımları raporlanır.
#
# Girdi:
#   X    : pandas.DataFrame, satır = örnek (1.292 ALL/AML), sütun = prob.
#          45 tamamen boş prob çıkarılmış, ek log dönüşümü UYGULANMAMIŞ DQN
#          değerleri (Bölüm IV-G ile aynı ön işleme).
#   sinif: pandas.Series (index = X.index), "ALL"/"AML"
#   doku : pandas.Series (index = X.index), örn. "KI"/"PK"
#
# Varsayımlar (raporda belirtilmeli):
#   - PCA, Bölüm IV-H'deki gibi en değişken 5.000 prob üzerinde, yalnızca
#     merkezlenmiş veriyle yapılır (prob bazında ölçekleme yok).
#   - PCA bileşen skorları birbirinden bağımsız olduğundan ilk 10 bileşende
#     Mahalanobis uzaklığı D^2 = Σ_j skor_j^2 / varyans_j biçiminde hesaplanır.
#   - Korelasyon tüm 54.630 geçerli prob üzerinden hesaplanır.
#   - MAD ölçeklenmemiş medyan mutlak sapmadır (scipy varsayılanı, scale=1).
#     Normal tutarlı MAD (×1,4826) istenirse MAD_OLCEK = "normal" yapın.
# =============================================================================
import json

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA

SEED = 42
N_DEGISKEN_PROB = 5000
N_BILESEN = 10
KANTIL = 0.999
MAD_KAT = 3.0
MAD_OLCEK = 1.0  # "normal" -> 1.4826 ile ölçeklenmiş MAD


def _dagilim(maske, seri):
    """Bayraklı örneklerin bir değişkene göre sayıları (tüm düzeyler, 0 dahil)."""
    duzeyler = sorted(seri.dropna().unique())
    sayim = seri[maske].value_counts().reindex(duzeyler, fill_value=0)
    return {str(k): int(v) for k, v in sayim.items()}


def ornek_aykiri_kontrolu(X, sinif, doku, seed=SEED):
    np.random.seed(seed)
    assert X.index.equals(sinif.index) and X.index.equals(doku.index)
    assert not X.isna().any().any(), "Önce 45 boş prob çıkarılmalı."

    # ---- 1) PCA + Mahalanobis -------------------------------------------
    varyans = X.var(axis=0, ddof=1)
    en_degisken = varyans.sort_values(ascending=False).index[:N_DEGISKEN_PROB]
    Xv = X[en_degisken].to_numpy(dtype=np.float64)

    pca = PCA(n_components=N_BILESEN, svd_solver="full", random_state=seed)
    skor = pca.fit_transform(Xv)                       # merkezlenmiş skorlar
    d2 = np.sum(skor ** 2 / pca.explained_variance_, axis=1)
    esik_d2 = stats.chi2.ppf(KANTIL, df=N_BILESEN)
    maske_mah = pd.Series(d2 > esik_d2, index=X.index)

    # ---- 2) Ortalama örnekler arası Pearson korelasyonu -----------------
    Z = X.to_numpy(dtype=np.float64)
    Z = Z - Z.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    R = Z @ Z.T                                         # n × n Pearson r
    n = R.shape[0]
    ort_kor = (R.sum(axis=1) - np.diag(R)) / (n - 1)    # kendisi hariç
    ort_kor = pd.Series(ort_kor, index=X.index)

    olcek = "normal" if MAD_OLCEK == "normal" else MAD_OLCEK
    mad = stats.median_abs_deviation(ort_kor.to_numpy(), scale=olcek)
    medyan = float(np.median(ort_kor))
    esik_kor = medyan - MAD_KAT * mad
    maske_kor = ort_kor < esik_kor

    # ---- 3) Raporlama (örnekler çıkarılmaz) ------------------------------
    maske_iki = maske_mah & maske_kor
    maske_herhangi = maske_mah | maske_kor

    sonuc = {
        "seed": seed,
        "n_ornek": int(n),
        "pca": {
            "n_prob": int(len(en_degisken)),
            "n_bilesen": N_BILESEN,
            "aciklanan_varyans_toplam": float(pca.explained_variance_ratio_.sum()),
            "esik_chi2_0.999_sd10": float(esik_d2),
            "n_aykiri": int(maske_mah.sum()),
            "sinif": _dagilim(maske_mah, sinif),
            "doku": _dagilim(maske_mah, doku),
            "ornekler": list(map(str, X.index[maske_mah])),
        },
        "ortalama_korelasyon": {
            "medyan": medyan,
            "mad": float(mad),
            "mad_olcek": str(MAD_OLCEK),
            "esik_medyan_eksi_3mad": float(esik_kor),
            "n_aykiri": int(maske_kor.sum()),
            "sinif": _dagilim(maske_kor, sinif),
            "doku": _dagilim(maske_kor, doku),
            "ornekler": list(map(str, X.index[maske_kor])),
        },
        "iki_olcutte_birden": {
            "n": int(maske_iki.sum()),
            "sinif": _dagilim(maske_iki, sinif),
            "doku": _dagilim(maske_iki, doku),
            "ornekler": list(map(str, X.index[maske_iki])),
        },
        "en_az_bir_olcutte": {
            "n": int(maske_herhangi.sum()),
            "sinif": _dagilim(maske_herhangi, sinif),
            "doku": _dagilim(maske_herhangi, doku),
        },
        # Karşılaştırma için tüm alt kümedeki dağılımlar
        "tum_alt_kume": {
            "sinif": _dagilim(pd.Series(True, index=X.index), sinif),
            "doku": _dagilim(pd.Series(True, index=X.index), doku),
        },
    }
    ayrinti = pd.DataFrame({
        "mahalanobis_d2": d2, "mah_aykiri": maske_mah,
        "ortalama_kor": ort_kor, "kor_aykiri": maske_kor,
        "sinif": sinif, "doku": doku,
    }, index=X.index)
    return sonuc, ayrinti


def ozet_yazdir(s):
    p, k, b = s["pca"], s["ortalama_korelasyon"], s["iki_olcutte_birden"]
    print(f"PCA ({p['n_prob']} prob, ilk {p['n_bilesen']} bileşen, "
          f"açıklanan varyans {p['aciklanan_varyans_toplam']:.3f}); "
          f"eşik D² > {p['esik_chi2_0.999_sd10']:.2f}: {p['n_aykiri']} örnek | "
          f"sınıf {p['sinif']} | doku {p['doku']}")
    print(f"Ortalama korelasyon: medyan {k['medyan']:.4f}, MAD {k['mad']:.4f}, "
          f"eşik {k['esik_medyan_eksi_3mad']:.4f}: {k['n_aykiri']} örnek | "
          f"sınıf {k['sinif']} | doku {k['doku']}")
    print(f"İki ölçütte birden: {b['n']} örnek | sınıf {b['sinif']} | doku {b['doku']}")


# --- Kullanım (değişken adlarını kendi defterinizdekilerle eşleyin) ---------
# sonuc, ayrinti = ornek_aykiri_kontrolu(X_gse, y_gse, doku_gse, seed=42)
# ozet_yazdir(sonuc)
# ayrinti.to_csv("gse13159_ornek_aykiri_kontrolu.csv")
# with open("ek_analiz_sonuclar.json", encoding="utf-8") as f:
#     tum = json.load(f)
# tum["gse13159_ornek_aykiri_kontrolu"] = sonuc
# with open("ek_analiz_sonuclar.json", "w", encoding="utf-8") as f:
#     json.dump(tum, f, ensure_ascii=False, indent=2)
