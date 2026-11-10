# -*- coding: utf-8 -*-
"""Golub ALL/AML — set ve sınıf bazında betimsel istatistikler (rapor Tablo VII) ve
sınıf içi aykırı örnek kontrolü (rapor Ek D, "Golub aykırı örnek kontrolü").

Kullanım:
    python src/golub_ek_betimsel_aykiri.py <veri_klasoru> <cikti_klasoru>

Girdi : <veri_klasoru>/data_set_ALL_AML_train.csv, data_set_ALL_AML_independent.csv, actual.csv
Çıktı : <cikti_klasoru>/golub_betimsel_tablo.csv
        <cikti_klasoru>/golub_aykiri_ornek_kontrolu.csv
        <cikti_klasoru>/golub_ek_betimsel_aykiri_sonuclar.json

Yöntem (GSE13159 defterindeki aykırı örnek hücresiyle aynı mantık, Golub boyutlarına uyarlanmış):
  Betimsel tablo
    - Ham MAS average difference değerlerinde her örnek için medyan, Q1 ve Q3 hesaplanır;
      tabloda bunların grup (set x sınıf) içi ortalaması, medyan için ayrıca en küçük-en büyük verilir.
    - Negatif, < 20 ve > 16.000 oranları grup içindeki tüm ölçümler üzerinden hesaplanır.
    - "Dönüşüm sonrası": 20-16.000 kırpma + log2, 72 örnekte sabit kalan problar çıkarıldıktan
      sonra (6.832 prob) örnek ortalamalarının grup içi ortalaması ve standart sapması (ddof=1).
  Aykırı örnek kontrolü (sınıf içinde; örnekler ÇIKARILMAZ)
    1) En değişken 1.000 prob, prob bazında z-skoru; sınıf içi PCA'nın ilk 5 bileşeninde
       MCD (Minimum Covariance Determinant) tabanlı sağlam Mahalanobis uzaklığı.
       Eşik: ki-kare(sd=5) dağılımının 0,999 kantili (20,5).
    2) 6.832 probun tamamıyla, sınıf içindeki diğer örneklerle ortalama Pearson korelasyonu.
       Eşik: medyan - 3 x MAD. Rapordaki sonuç ÖLÇEKLENMEMİŞ MAD ile (scipy varsayılanı,
       scale=1) elde edilmiştir: MAD_OLCEK = 1.0. GSE13159 defterindeki hücre normal tutarlı
       MAD (x1,4826) kullanır; karşılaştırma için bu seçenekle elde edilen sonuç da JSON'a
       "normal_tutarli_mad" anahtarıyla yazılır (yalnızca 21 numaralı hasta işaretlenir).
  Etiketler yalnızca keşif amacıyla kullanılır; model eğitilmez.

Sabit tohum: 42. Kütüphaneler: numpy, pandas, scipy, scikit-learn.
"""
import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.covariance import MinCovDet
from sklearn.decomposition import PCA

SEED = 42
ALT, UST = 20, 16000          # kırpma sınırları
N_DEGISKEN_PROB = 1000        # Mahalanobis için en değişken prob sayısı
K = 5                         # Mahalanobis için bileşen sayısı
KANTIL = 0.999
MAD_KAT = 3.0
MAD_OLCEK = 1.0               # 1.0: ölçeklenmemiş MAD (rapor); "normal": x1,4826


def golub_oku(veri):
    """Eğitim ve test CSV'lerini okur; hasta numarası üzerinden etiketleri eşler.
    Dönen: X (örnek x prob, ham değerler), y (ALL/AML), set_ (egitim/test)."""
    etiket = pd.read_csv(os.path.join(veri, "actual.csv")).set_index("patient")["cancer"]
    parca, setler = [], []
    for ad, dosya in (("egitim", "data_set_ALL_AML_train.csv"),
                      ("test", "data_set_ALL_AML_independent.csv")):
        df = pd.read_csv(os.path.join(veri, dosya))
        hasta = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
        X = df[hasta].apply(pd.to_numeric).T
        X.index = [int(h) for h in hasta]
        X.columns = df["Gene Accession Number"].values
        parca.append(X)
        setler += [ad] * len(hasta)
    X = pd.concat(parca)
    assert X.shape == (72, 7129), X.shape
    assert X.index.is_unique, "Eğitim ve test setinde ortak hasta numarası var."
    y = etiket.reindex(X.index)
    assert y.notna().all(), "Etiketi bulunmayan hasta var."
    return X, y, pd.Series(setler, index=X.index, name="set")


def betimsel_tablo(X, y, set_, Xp):
    satirlar = []
    ort_donusum = Xp.mean(axis=1)                       # örnek ortalaması (kırpma + log2)
    for s in ("egitim", "test"):
        for c in ("ALL", "AML"):
            m = ((set_ == s) & (y == c)).to_numpy()
            V = X.to_numpy()[m]
            med = np.median(V, axis=1)
            q1 = np.percentile(V, 25, axis=1)
            q3 = np.percentile(V, 75, axis=1)
            satirlar.append({
                "set": s, "sinif": c, "n": int(m.sum()),
                "ornek_medyani_ort": float(med.mean()),
                "ornek_medyani_min": float(med.min()),
                "ornek_medyani_maks": float(med.max()),
                "q1_ort": float(q1.mean()), "q3_ort": float(q3.mean()),
                "negatif_yuzde": float(100 * (V < 0).mean()),
                "lt20_yuzde": float(100 * (V < ALT).mean()),
                "gt16000_yuzde": float(100 * (V > UST).mean()),
                "donusum_sonrasi_ort": float(ort_donusum[m].mean()),
                "donusum_sonrasi_ss": float(ort_donusum[m].std(ddof=1)),
            })
    return pd.DataFrame(satirlar)


def aykiri_ornek_kontrolu(Xp, y, set_, mad_olcek=MAD_OLCEK):
    """Sınıf içi MCD-Mahalanobis ve ortalama korelasyon ölçütleri. Xp: örnek x prob (kırpma + log2)."""
    M = Xp.to_numpy(dtype=np.float64)                   # örnek x prob
    var = M.var(axis=0)
    top = np.argsort(var)[::-1][:N_DEGISKEN_PROB]
    Zs = (M[:, top] - M[:, top].mean(0)) / M[:, top].std(0)
    esik_d2 = float(stats.chi2.ppf(KANTIL, df=K))
    satirlar = []
    for cls in ("ALL", "AML"):                          # sınıf içinde: diğer sınıf aykırı sayılmasın
        m = (y == cls).to_numpy()
        Pc = PCA(n_components=K, random_state=SEED).fit_transform(Zs[m])
        d2 = MinCovDet(random_state=SEED).fit(Pc).mahalanobis(Pc)
        C = np.corrcoef(M[m])
        np.fill_diagonal(C, np.nan)
        ort_r = np.nanmean(C, axis=1)
        med = float(np.median(ort_r))
        mad = float(stats.median_abs_deviation(ort_r, scale=mad_olcek))
        esik_r = med - MAD_KAT * mad
        for hasta, s, a, b in zip(Xp.index[m], set_[m], d2, ort_r):
            satirlar.append({"hasta": int(hasta), "sinif": cls, "set": s,
                             "mahal_d2": float(a), "mahal_aykiri": bool(a > esik_d2),
                             "ort_r": float(b), "r_medyan": med, "r_mad": mad,
                             "r_esik": float(esik_r), "kor_aykiri": bool(b < esik_r)})
    df = pd.DataFrame(satirlar)
    df["iki_olcut"] = df.mahal_aykiri & df.kor_aykiri
    return df, esik_d2


def main(veri, cikti):
    os.makedirs(cikti, exist_ok=True)
    X, y, set_ = golub_oku(veri)

    # Ön işleme (yalnızca keşif): kırpma + log2, sabit probların çıkarılması
    Xp = np.log2(X.clip(ALT, UST))
    Xp = Xp.loc[:, (Xp.max() - Xp.min()) > 0]
    assert Xp.shape[1] == 6832, Xp.shape

    tablo = betimsel_tablo(X, y, set_, Xp)
    tablo.to_csv(os.path.join(cikti, "golub_betimsel_tablo.csv"), index=False)

    df, esik_d2 = aykiri_ornek_kontrolu(Xp, y, set_)
    df.to_csv(os.path.join(cikti, "golub_aykiri_ornek_kontrolu.csv"), index=False)

    ozet = df.groupby("sinif")[["mahal_aykiri", "kor_aykiri", "iki_olcut"]].sum().astype(int)
    ozet["n"] = df.groupby("sinif").size()
    df_n, _ = aykiri_ornek_kontrolu(Xp, y, set_, mad_olcek="normal")   # duyarlılık: normal tutarlı MAD
    sonuc = {
        "seed": SEED,
        "n_prob_donusum_sonrasi": int(Xp.shape[1]),
        "mahalanobis": {"n_degisken_prob": N_DEGISKEN_PROB, "K": K,
                        "esik_chi2_0.999": esik_d2},
        "ozet": ozet.reset_index().to_dict("records"),
        "r_esik": df.groupby("sinif").r_esik.first().to_dict(),
        "kor_aykiri_hastalar": df[df.kor_aykiri][["hasta", "sinif", "set", "ort_r", "r_esik"]].to_dict("records"),
        "iki_olcut_hastalar": df[df.iki_olcut][["hasta", "sinif", "set", "mahal_d2", "ort_r", "r_esik"]].to_dict("records"),
        "mad_olcek": str(MAD_OLCEK),
        "normal_tutarli_mad": {
            "r_esik": df_n.groupby("sinif").r_esik.first().to_dict(),
            "kor_aykiri_hastalar": df_n[df_n.kor_aykiri].hasta.tolist(),
            "iki_olcut_hastalar": df_n[df_n.iki_olcut].hasta.tolist(),
        },
        "betimsel_tablo": tablo.to_dict("records"),
    }
    with open(os.path.join(cikti, "golub_ek_betimsel_aykiri_sonuclar.json"), "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=1)

    pd.set_option("display.width", 200)
    print(tablo.round(2).to_string(index=False))
    print()
    print(ozet)
    print("Mahalanobis eşiği (ki-kare, sd=%d, %.3f): %.2f" % (K, KANTIL, esik_d2))
    print("Korelasyon eşikleri:", {k: round(v, 3) for k, v in sonuc["r_esik"].items()})
    print("Korelasyon ölçütünde işaretlenenler:")
    print(df[df.kor_aykiri][["hasta", "sinif", "set", "mahal_d2", "ort_r", "r_esik", "iki_olcut"]].round(3).to_string(index=False))
    print("Normal tutarlı MAD ile işaretlenenler:", sonuc["normal_tutarli_mad"]["kor_aykiri_hastalar"],
          "| iki ölçüt:", sonuc["normal_tutarli_mad"]["iki_olcut_hastalar"])


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Kullanım: python golub_ek_betimsel_aykiri.py <veri_klasoru> <cikti_klasoru>")
    main(sys.argv[1], sys.argv[2])
