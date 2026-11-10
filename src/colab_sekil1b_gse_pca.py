# -*- coding: utf-8 -*-
"""Şekil 1b (GSE13159 PCA + scree) — sütun genişliğinde okunur yazı boyutuyla yeniden çizim.

Hesap notebooks/GSE13159_EDA_v3.ipynb defterindeki "8. PCA" hücresiyle aynıdır: tüm örneklerde
sonlu olan problar, prob bazında z-skoru, en değişken 5.000 prob, PCA (random_state=42).
Yalnızca şekil boyutu ve yazı boyutları değişir; sayılar değişmez.

Kullanım (Colab, defterin sonunda yeni bir hücrede):
    exec(open('/content/drive/MyDrive/Donem_Projesi_Losemi/02_Colab/colab_sekil1b_gse_pca.py', encoding='utf-8').read())
Defter baştan çalıştırıldıysa bellekteki P, evr ve y kullanılır; değilse işlenmiş alt küme Drive'dan okunur.
Çıktı: 03_Ikinci_Veri_Seti_GSE13159/GSE13159_03_PCA_sutun.png (300 dpi)

NOT: Bu hücre 10 Kasım 2026'da yazıldı; çizim kısmı sentetik veriyle sınandı, GSE13159 verisinde
çalıştırılması ve çıkan eksen yüzdelerinin rapordaki %17,6 / %6,9 ile karşılaştırılması gerekir.
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

RANDOM_STATE = 42
PALET = {"ALL": "#1f77b4", "AML": "#d9541e"}          # Şekil 1a ile aynı renkler


def gse_pca_hesapla(Xt):
    """Xt: prob x örnek. Defterdeki PCA hücresiyle aynı adımlar. Dönen: skorlar, açıklanan varyans oranı."""
    Xt = np.asarray(Xt, dtype=np.float32)
    sonlu = np.isfinite(Xt).all(axis=1)
    mu = np.nanmean(Xt, axis=1, keepdims=True)
    sd = np.nanstd(Xt, axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        Z = (Xt - mu) / sd
    var = np.nanvar(Xt, axis=1)
    var_ok = np.where(sonlu & np.isfinite(var) & np.isfinite(Z).all(axis=1), var, -np.inf)
    top = np.argsort(var_ok)[::-1][:min(5000, int(np.isfinite(var_ok).sum()))]
    pca = PCA(n_components=min(50, Xt.shape[1] - 1), random_state=RANDOM_STATE).fit(Z[top].T)
    return pca.transform(Z[top].T), pca.explained_variance_ratio_


def sekil_1b(P, evr, y, dosya):
    y = np.asarray(y)
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5, "xtick.labelsize": 7.5,
                         "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
    for cls in ("ALL", "AML"):
        m = y == cls
        ax[0].scatter(P[m, 0], P[m, 1], s=3, alpha=.6, c=PALET[cls], linewidths=0, label=f"{cls} (n={m.sum()})")
    v = lambda x: f"{100 * x:.1f}".replace(".", ",")
    ax[0].set(xlabel=f"PC1 (%{v(evr[0])})", ylabel=f"PC2 (%{v(evr[1])})", title="GSE13159 PCA (1.292 örnek)")
    k = min(20, len(evr))
    ax[1].bar(range(1, k + 1), 100 * evr[:k], color="gray")
    ax[1].plot(range(1, k + 1), 100 * np.cumsum(evr[:k]), "o-", color=PALET["ALL"], ms=2.5, lw=1)
    ax[1].set(xlabel="Bileşen", ylabel="Açıklanan varyans (%)", title="Scree", xticks=[1, 5, 10, 15, 20][:1 + k // 5])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(.5, -.09), markerscale=2.5,
               columnspacing=.8, handletextpad=.1)
    fig.tight_layout(w_pad=.6)
    fig.savefig(dosya, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Kaydedildi:", dosya, "| PC1 %" + v(evr[0]), "PC2 %" + v(evr[1]),
          "| ilk 10 PC kümülatif %" + v(evr[:10].sum()))


if __name__ == "__main__" or "get_ipython" in globals():
    g = globals()
    if all(k in g for k in ("P", "evr", "y", "OUT")):          # defter baştan çalıştırılmış
        _P, _evr, _y, _out = g["P"], g["evr"], g["y"], g["OUT"]
    else:
        from google.colab import drive
        drive.mount("/content/drive")
        _out = glob.glob("/content/drive/MyDrive/**/03_Ikinci_Veri_Seti_GSE13159", recursive=True)[0]
        expr = pd.read_parquet(os.path.join(_out, "GSE13159_ALL_AML_expression.parquet"))
        sel = pd.read_csv(os.path.join(_out, "GSE13159_ALL_AML_labels.csv"))
        expr = expr[sel.geo_accession]
        _y = sel.ALL_AML_group.values
        _P, _evr = gse_pca_hesapla(expr.values)                  # GSE13159'a ek dönüşüm uygulanmıyor
    sekil_1b(_P, _evr, _y, os.path.join(_out, "GSE13159_03_PCA_sutun.png"))
