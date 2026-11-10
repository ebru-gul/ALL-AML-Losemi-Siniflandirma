# -*- coding: utf-8 -*-
"""Ara rapordaki Golub şekillerini ve iki seti birlikte gösteren Şekil 3-4'ü teslim boyutunda
(basılı genişlikte okunur yazı boyutu, 300 dpi) yeniden üretir.

Kullanım:
    python src/sekilleri_yeniden_uret.py <golub_veri_klasoru> <cikti_klasoru> [<gse_tablo_klasoru>]

Girdi : <golub_veri_klasoru>/data_set_ALL_AML_train.csv, data_set_ALL_AML_independent.csv, actual.csv
        (isteğe bağlı) <gse_tablo_klasoru>/gen_duzeyi_iki_set.csv         -> Şekil 3
        (isteğe bağlı) <gse_tablo_klasoru>/GSE13159_ALL_AML_labels.csv    -> Şekil 4b-c
Çıktı : sekil01a ... sekil19 PNG dosyaları ve sekil_kontrol_sayilari.json

Üretilen şekiller (rapor numarasıyla):
    1a  Golub PCA + scree            6   Ham ve kırpma + log2 dağılımları
    2   En ayırt edici 50 prob        7   t-SNE
    3   İki sette Cohen d             8a  Volkan grafiği
    4   Sınıf dağılımları             9   Etiketsiz Ward kümelemesi
    5   Örnek bazında kutu grafiği    10  PC1 / PC4 eğitim-test kayması
                                      19  Şekil 2'nin tüm problar etiketli sürümü
Şekil 1b, 8b, 11-18 GSE13159 ifade matrisini gerektirir; notebooks/ altındaki defterlerle üretilir.

Hesaplar src/golub_eda_ek.py ile aynıdır (kırpma 20-16.000, log2, sabit probların çıkarılması,
z-skoru, PCA, Welch t + BH + Cohen d, ortalama bağlantı / korelasyon uzaklığı, Ward). Betik,
çizdiği sayıları sekil_kontrol_sayilari.json dosyasına yazar; bu sayılar rapordakilerle karşılaştırılabilir.
Sabit tohum: 42.
"""
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage, set_link_color_palette
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

SEED = 42
PAL = {"ALL": "#1f77b4", "AML": "#d9541e"}
SEMBOL = {"U46499_at": "MGST1", "M84526_at": "CFD", "L09209_s_at": "APLP2", "M27891_at": "CST3",
          "M23197_at": "CD33", "M19507_at": "MPO", "X95735_at": "ZYX", "M31523_at": "TCF3",
          "M11722_at": "DNTT", "U05259_rna1_at": "CD79A", "M89957_at": "CD79B",
          "M84371_rna1_s_at": "CD19", "M28170_at": "CD19", "M92287_at": "CCND3",
          "Z49194_at": "POU2AF1", "X82240_rna1_at": "TCL1A"}        # GPL80 anotasyonundaki semboller
SEKIL2_ETIKET = ["MPO", "APLP2", "ZYX", "CST3", "CFD", "MGST1", "CD33", "CD19", "CD79A", "CD79B",
                 "CCND3", "TCF3", "DNTT"]
SEKIL3_ETIKET = ["DNTT", "TCF3", "CD79A", "CD19", "MPO", "CD33", "CST3", "CFD", "MGST1"]

plt.rcParams.update({"savefig.dpi": 300, "font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8.5,
                     "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def tr(x, nd=2):
    """Ondalık virgül: 0.69 -> '0,69'."""
    return f"{x:.{nd}f}".replace(".", ",")


def kaydet(fig, cikti, ad):
    fig.savefig(os.path.join(cikti, ad), bbox_inches="tight")
    plt.close(fig)


def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0)
    n = p.size
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def golub_oku(veri):
    etiket = pd.read_csv(os.path.join(veri, "actual.csv")).set_index("patient")["cancer"]
    parca, aciklama = [], None
    for dosya in ("data_set_ALL_AML_train.csv", "data_set_ALL_AML_independent.csv"):
        df = pd.read_csv(os.path.join(veri, dosya))
        hasta = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
        X = df[hasta].apply(pd.to_numeric).T
        X.index = [int(h) for h in hasta]
        X.columns = df["Gene Accession Number"].values
        parca.append(X)
        aciklama = df.set_index("Gene Accession Number")["Gene Description"]
    Xtr, Xte = parca
    X = pd.concat(parca)
    assert X.shape == (72, 7129) and X.index.is_unique
    y = etiket.reindex(X.index)
    assert y.notna().all()
    egitim = np.array([i in set(Xtr.index) for i in X.index])
    return X, y, egitim, aciklama


def ad(prob, aciklama, uzun=False):
    """Prob etiketi: sembol biliniyorsa sembol, yoksa Hu6800 açıklamasının ilk 34 karakteri."""
    s = SEMBOL.get(prob) or str(aciklama.get(prob, prob))[:34]
    return f"{s} ({prob})" if uzun else s


def main(veri, cikti, gse=None):
    os.makedirs(cikti, exist_ok=True)
    K = {}                                                   # kontrol sayıları
    X, y, egitim, aciklama = golub_oku(veri)
    yv = y.to_numpy()
    ham = X.to_numpy().ravel()

    # ---- Ortak hesaplar (golub_eda_ek.py ile aynı) -------------------------------------
    Xp = np.log2(X.clip(20, 16000))
    Xp = Xp.loc[:, (Xp.max() - Xp.min()) > 0]
    Z = StandardScaler().fit_transform(Xp)
    pca = PCA(n_components=20, random_state=SEED).fit(Z)
    P = pca.transform(Z)
    ev = pca.explained_variance_ratio_ * 100
    mA = yv == "ALL"
    A, B = Xp[mA].to_numpy(), Xp[~mA].to_numpy()
    _, p_t = stats.ttest_ind(A, B, axis=0, equal_var=False)
    q = bh(p_t)
    sp = np.sqrt(((A.shape[0] - 1) * A.var(0, ddof=1) + (B.shape[0] - 1) * B.var(0, ddof=1)) / (len(yv) - 2))
    d = (A.mean(0) - B.mean(0)) / sp
    de = pd.DataFrame({"prob": Xp.columns, "d": d, "p": p_t, "q": q}).sort_values("p")
    K.update(n_prob=int(Xp.shape[1]), pca_ev=[round(float(e), 2) for e in ev[:5]],
             pca_kum10=round(float(ev[:10].sum()), 1), pca_kum20=round(float(ev[:20].sum()), 1),
             n_all_yuksek=int(((q < .01) & (d > 1)).sum()), n_aml_yuksek=int(((q < .01) & (d < -1)).sum()))

    def sacilim(ax, xs, ys, s=26):
        for cls in ("ALL", "AML"):
            m = yv == cls
            ax.scatter(xs[m & egitim], ys[m & egitim], c=PAL[cls], marker="o", s=s, edgecolor="k",
                       linewidth=.3, label=f"{cls} eğitim")
            ax.scatter(xs[m & ~egitim], ys[m & ~egitim], c=PAL[cls], marker="^", s=s + 6, edgecolor="k",
                       linewidth=.3, label=f"{cls} test")

    # ---- Şekil 1a: PCA + scree ---------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(3.45, 2.1), gridspec_kw={"width_ratios": [1.35, 1]})
    sacilim(ax[0], P[:, 0], P[:, 1], s=16)
    ax[0].set(xlabel=f"PC1 (%{tr(ev[0], 1)})", ylabel=f"PC2 (%{tr(ev[1], 1)})", title="Golub PCA (72 örnek)")
    ax[1].bar(range(1, 21), ev, color="gray")
    ax[1].plot(range(1, 21), np.cumsum(ev), "o-", color=PAL["ALL"], ms=2.5, lw=1)
    ax[1].set(xlabel="Bileşen", ylabel="Açıklanan varyans (%)", title="Scree", xticks=[1, 5, 10, 15, 20])
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(.5, -.09),
               columnspacing=.8, handletextpad=.1)
    fig.tight_layout(w_pad=.6)
    kaydet(fig, cikti, "sekil01a_golub_pca.png")

    # ---- Şekil 2 ve 19: en ayırt edici 50 prob -----------------------------------------
    top50 = de.head(50).prob.tolist()
    H = pd.DataFrame(Z, index=X.index, columns=Xp.columns)[top50]
    Ls = linkage(H.to_numpy(), method="average", metric="correlation")
    Lg = linkage(H.T.to_numpy(), method="average", metric="correlation")
    cl2 = fcluster(Ls, 2, criterion="maxclust")
    K["ari_top50_k2"] = round(float(adjusted_rand_score(yv, cl2)), 2)
    for dosya, uzun, boyut, yazi in (("sekil02_golub_isi_haritasi.png", False, (3.45, 4.6), 6.5),
                                     ("sekil19_golub_isi_haritasi_etiketli.png", True, (7.0, 6.6), 6)):
        if uzun:
            etiket = [ad(g, aciklama, uzun=True) for g in top50]
        else:
            etiket = [SEMBOL[g] if SEMBOL.get(g) in SEKIL2_ETIKET and g != "M28170_at" else "" for g in top50]
        g = sns.clustermap(H.T, row_linkage=Lg, col_linkage=Ls, cmap="RdBu_r", vmin=-3, vmax=3, figsize=boyut,
                           col_colors=[PAL[c] for c in yv], yticklabels=etiket, xticklabels=False,
                           dendrogram_ratio=(.12, .075), colors_ratio=.025,
                           cbar_pos=(.32, .957, .3, .011), cbar_kws={"orientation": "horizontal"})
        g.ax_col_dendrogram.set_visible(False)
        g.ax_cbar.set_title("z-skoru", fontsize=yazi + .5, pad=2)
        g.ax_cbar.tick_params(labelsize=yazi)
        g.ax_heatmap.tick_params(axis="y", labelsize=yazi, length=0 if uzun else 2)
        g.ax_heatmap.set_xlabel("72 hasta (üst şerit: mavi ALL, turuncu AML)", fontsize=yazi + 1.5)
        g.ax_heatmap.set_ylabel("")
        g.savefig(os.path.join(cikti, dosya))
        plt.close("all")

    # ---- Şekil 3: iki sette Cohen d ----------------------------------------------------
    yol = os.path.join(gse, "gen_duzeyi_iki_set.csv") if gse else None
    if yol and os.path.exists(yol):
        gd = pd.read_csv(yol)
        iki = gd.golub_ayirt & gd.gse_ayirt
        bir = gd.golub_ayirt ^ gd.gse_ayirt
        r = stats.pearsonr(gd.golub_d, gd.gse_d)[0]
        K.update(ortak_gen=int(len(gd)), iki_sette_ayirt=int(iki.sum()), cohen_d_pearson=round(float(r), 2),
                 ayni_yon=int((np.sign(gd.golub_d[iki]) == np.sign(gd.gse_d[iki])).sum()))
        fig, ax = plt.subplots(figsize=(3.45, 3.1))
        ax.scatter(gd.golub_d[~iki & ~bir], gd.gse_d[~iki & ~bir], s=3, c="#cfcfcf", linewidths=0,
                   label="hiçbirinde ayırt edici değil")
        ax.scatter(gd.golub_d[bir], gd.gse_d[bir], s=4, c="#6b6b6b", linewidths=0, label="yalnız bir sette ayırt edici")
        ax.scatter(gd.golub_d[iki], gd.gse_d[iki], s=7, c="#7b2d9b", linewidths=0,
                   label=f"iki sette ayırt edici ({int(iki.sum())})")
        for v in (-1, 1):
            ax.axvline(v, ls=":", c="k", lw=.6)
            ax.axhline(v, ls=":", c="k", lw=.6)
        sec = gd.set_index("gen").reindex(SEKIL3_ETIKET).dropna(subset=["golub_d"])
        for sag in (True, False):                              # etiketleri dikeyde aralıklı yerleştir
            alt = sec[(sec.golub_d > 0) == sag].sort_values("gse_d", ascending=False)
            yer = np.linspace(2.75, 1.55, len(alt)) if sag else np.linspace(-1.55, -3.15, len(alt))
            for (gen, s_), yy in zip(alt.iterrows(), yer):
                ax.annotate(gen, (s_.golub_d, s_.gse_d), fontsize=6.5, ha="left" if sag else "right", va="center",
                            xytext=(3.55 if sag else -3.95, yy), textcoords="data",
                            arrowprops=dict(arrowstyle="-", lw=.3, color="gray"))
        ax.set(xlabel="Golub Cohen d (ALL − AML)", ylabel="GSE13159 Cohen d (ALL − AML)",
               title=f"{len(gd):,}".replace(",", ".") + f" ortak gen; Pearson r = {tr(r)}", xlim=(-4.9, 4.9), ylim=(-3.5, 3.1))
        ax.legend(frameon=False, loc="lower right", fontsize=6.3, markerscale=1.6, handletextpad=.1)
        kaydet(fig, cikti, "sekil03_cohen_d_iki_set.png")

    # ---- Şekil 4: sınıf dağılımları ----------------------------------------------------
    yol = os.path.join(gse, "GSE13159_ALL_AML_labels.csv") if gse else None
    etk = pd.read_csv(yol) if yol and os.path.exists(yol) else None
    fig, ax = plt.subplots(1, 3 if etk is not None else 1, figsize=(7.0, 2.7) if etk is not None else (2.6, 2.7),
                           gridspec_kw={"width_ratios": [1.25, .9, 1.15]} if etk is not None else None)
    ax = np.atleast_1d(ax)
    say = pd.DataFrame({"Eğitim (n=38)": y[egitim].value_counts(), "Test (n=34)": y[~egitim].value_counts()}).loc[["ALL", "AML"]]
    say.plot.bar(ax=ax[0], color=["#4c72b0", "#8c8c8c"], rot=0, edgecolor="k", linewidth=.4, width=.75)
    for p_ in ax[0].patches:
        ax[0].annotate(f"{int(p_.get_height())}", (p_.get_x() + p_.get_width() / 2, p_.get_height()), ha="center", va="bottom", fontsize=7)
    ax[0].set(ylabel="Hasta sayısı", xlabel="", title="(a) Golub", ylim=(0, 38))
    ax[0].legend(frameon=False, loc="upper right")
    K["golub_sinif"] = {k: {s: int(v) for s, v in say[k].items()} for k in say.columns}
    if etk is not None:
        n = etk.ALL_AML_group.value_counts().loc[["ALL", "AML"]]
        ax[1].bar(n.index, n.values, color=[PAL["ALL"], PAL["AML"]], edgecolor="k", linewidth=.4, width=.7)
        for i, v in enumerate(n.values):
            ax[1].text(i, v + 10, f"{v}\n(%{tr(100 * v / n.sum(), 1)})", ha="center", va="bottom", fontsize=7)
        ax[1].set(ylabel="Örnek sayısı", title="(b) GSE13159", ylim=(0, 1000), xlim=(-.6, 1.6))
        alt = etk.groupby(["ALL_AML_group", "leukemia_class"]).size().reset_index(name="n")
        alt = pd.concat([alt[alt.ALL_AML_group == g].sort_values("n", ascending=False) for g in ("AML", "ALL")])
        yy = np.arange(len(alt))[::-1]
        ax[2].barh(yy, alt.n, color=[PAL[g] for g in alt.ALL_AML_group], edgecolor="k", linewidth=.4)
        for a, v in zip(yy, alt.n):
            ax[2].text(v + 5, a, str(v), va="center", fontsize=6.5)
        ax[2].set_yticks(yy)
        ax[2].set_yticklabels(alt.leukemia_class, fontsize=6.3)
        ax[2].set(xlabel="Örnek sayısı", title="(c) GSE13159 alt tipleri", xlim=(0, 400))
        K["gse_sinif"] = {k: int(v) for k, v in n.items()}
        K["gse_alt_tip"] = {r_.leukemia_class: int(r_.n) for r_ in alt.itertuples()}
    if etk is not None:                                      # uzun alt tip adları için elle yerleşim
        for a, konum in zip(ax, ([.06, .17, .22, .70], [.385, .17, .13, .70], [.80, .17, .18, .70])):
            a.set_position(konum)
    else:
        fig.tight_layout()
    kaydet(fig, cikti, "sekil04_sinif_dagilimlari.png")

    # ---- Şekil 5: örnek bazında kutu grafiği -------------------------------------------
    logX = np.sign(X) * np.log2(1 + np.abs(X))
    fig, ax = plt.subplots(figsize=(7.0, 2.3))
    bp = ax.boxplot([logX.loc[h].to_numpy() for h in X.index], showfliers=False, patch_artist=True, widths=.75,
                    medianprops=dict(color="k", lw=.6), whiskerprops=dict(lw=.5), capprops=dict(lw=.5))
    for kutu, h in zip(bp["boxes"], X.index):
        kutu.set(facecolor=PAL[y[h]], linewidth=.4)
    ax.axvline(38.5, color="k", ls="--", lw=.8)
    ax.text(19.5, 15.8, "eğitim", ha="center")
    ax.text(55.5, 15.8, "bağımsız test", ha="center")
    ax.set(xticks=[], ylim=(-15, 17), ylabel="İşaretli log$_2$(1+|x|)",
           xlabel="72 hasta, dosya sırasıyla (mavi ALL, turuncu AML)")
    kaydet(fig, cikti, "sekil05_golub_ornek_kutu.png")

    # ---- Şekil 6: ham ve kırpma + log2 dağılımları -------------------------------------
    kirp = np.log2(X.clip(20, 16000))
    K.update(lt20_yuzde=round(100 * float((ham < 20).mean()), 1), gt16000_yuzde=round(100 * float((ham > 16000).mean()), 2),
             negatif_yuzde=round(100 * float((ham < 0).mean()), 1))
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.4))
    ax[0].hist(ham, bins=200, color="#4c72b0")
    ax[0].axvline(0, color="k", lw=.6)
    ax[0].set(yscale="log", xlabel="Ham ifade (MAS average difference)", ylabel="Ölçüm sayısı (log)",
              title="(a) Ham değerler, 72 örnek")
    for cls in ("ALL", "AML"):
        ax[1].hist(kirp[mA if cls == "ALL" else ~mA].to_numpy().ravel(), bins=100, density=True, histtype="step",
                   color=PAL[cls], lw=1, label=cls)
    ax[1].set(ylim=(0, .30), xlabel="log$_2$(20–16.000 aralığına kırpılmış ifade)", ylabel="Yoğunluk",
              title="(b) Kırpma + log$_2$ sonrası, sınıfa göre")
    ax[1].annotate(f"log$_2$ 20 alt sınırında birikim\n(%{tr(K['lt20_yuzde'], 1)} ölçüm; tepe eksen dışında)",
                   xy=(np.log2(20) + .05, .295), xytext=(5.6, .245), fontsize=7,
                   arrowprops=dict(arrowstyle="->", lw=.6))
    ax[1].legend(frameon=False, loc="center right")
    fig.tight_layout()
    kaydet(fig, cikti, "sekil06_golub_dagilim.png")

    # ---- Şekil 7: t-SNE ----------------------------------------------------------------
    ts = TSNE(n_components=2, perplexity=15, init="pca", random_state=SEED, learning_rate="auto").fit_transform(P[:, :20])
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    sacilim(ax, ts[:, 0], ts[:, 1])
    ax.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title="Golub t-SNE (ilk 20 PC, perplexity 15)")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(.5, -.2), ncol=4, columnspacing=.8, handletextpad=.1)
    kaydet(fig, cikti, "sekil07_golub_tsne.png")

    # ---- Şekil 8a: volkan --------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(3.6, 3.0))
    renk = np.where((q < .01) & (d > 1), PAL["ALL"], np.where((q < .01) & (d < -1), PAL["AML"], "#cfcfcf"))
    ax.scatter(d, -np.log10(p_t), c=renk, s=4, linewidths=0)
    ilk = de.head(10)
    for yon in (1, -1):
        alt = ilk[np.sign(ilk.d) == yon].sort_values("p")
        for i, s in enumerate(alt.itertuples()):
            ax.annotate(ad(s.prob, aciklama), (s.d, -np.log10(s.p)), fontsize=6.5, va="center",
                        ha="left" if yon > 0 else "right",
                        xytext=(yon * 3.15, max(-np.log10(alt.p)) + .6 - 1.25 * i), textcoords="data",
                        arrowprops=dict(arrowstyle="-", lw=.3, color="gray"))
    for v in (-1, 1):
        ax.axvline(v, ls=":", c="k", lw=.6)
    ax.scatter([], [], c=PAL["ALL"], s=10, label=f"ALL'de yüksek ({K['n_all_yuksek']})")
    ax.scatter([], [], c=PAL["AML"], s=10, label=f"AML'de yüksek ({K['n_aml_yuksek']})")
    ax.set(xlabel="Cohen d (ALL − AML)", ylabel="−log$_{10}$ p (Welch)", xlim=(-4.9, 4.9),
           title="Golub: q < 0,01 ve |d| > 1 renkli")
    ax.legend(frameon=False, loc="upper center", fontsize=6.5, handletextpad=.1)
    kaydet(fig, cikti, "sekil08a_golub_volkan.png")
    K["volkan_etiketleri"] = [ad(pb, aciklama) for pb in ilk.prob]

    # ---- Şekil 9: etiketsiz Ward kümelemesi --------------------------------------------
    vi = np.argsort(Xp.var().to_numpy())[::-1][:1000]
    Lw = linkage(Z[:, vi], "ward")
    ari = {k: round(float(adjusted_rand_score(yv, fcluster(Lw, k, "maxclust"))), 2) for k in (2, 3, 4)}
    K["ari_ward_1000"] = ari
    K["ward_k2_capraz"] = pd.crosstab(yv, fcluster(Lw, 2, "maxclust")).to_dict()
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    set_link_color_palette(["#ff7f0e", "#2ca02c"])
    dendrogram(Lw, ax=ax, labels=[str(h) for h in X.index], leaf_font_size=5.5,
               color_threshold=(Lw[-1, 2] + Lw[-2, 2]) / 2, above_threshold_color="#333333")
    for lbl in ax.get_xmajorticklabels():
        lbl.set_color(PAL[y[int(lbl.get_text())]])
    ax.set(ylabel="Ward uzaklığı", xlabel="Hasta no (etiket rengi gerçek sınıf: mavi ALL, turuncu AML; test hastaları 39–72)",
           title=f"Etiketsiz Ward kümelemesi, en değişken 1.000 prob; dal renkleri k = 2 kesimi (ARI = {tr(ari[2])})")
    kaydet(fig, cikti, "sekil09_golub_ward.png")

    # ---- Şekil 10: PC1 / PC4 eğitim-test kayması ---------------------------------------
    gruplar = [("ALL eğitim", mA & egitim, "ALL"), ("AML eğitim", ~mA & egitim, "AML"),
               ("ALL test", mA & ~egitim, "ALL"), ("AML test", ~mA & ~egitim, "AML")]
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.5))
    K["pc_egitim_test_welch_p"] = {}
    for a, k in zip(ax, (0, 3)):
        p_w = stats.ttest_ind(P[egitim, k], P[~egitim, k], equal_var=False).pvalue
        K["pc_egitim_test_welch_p"][f"PC{k + 1}"] = float(p_w)
        bp = a.boxplot([P[m, k] for _, m, _ in gruplar], patch_artist=True, showfliers=False, widths=.55,
                       medianprops=dict(color="k", lw=.8))
        for kutu, (_, _, c) in zip(bp["boxes"], gruplar):
            kutu.set(facecolor=PAL[c], alpha=.75)
        for i, (_, m, _) in enumerate(gruplar, 1):
            a.scatter(i + rng.uniform(-.12, .12, m.sum()), P[m, k], s=4, c="k", zorder=3, linewidths=0)
        a.set_xticks(range(1, 5))
        a.set_xticklabels([f"{g}\nn={m.sum()}" for g, m, _ in gruplar])
        us = int(np.floor(np.log10(p_w)))
        a.set(ylabel=f"PC{k + 1} skoru", title=f"PC{k + 1}: eğitim–test Welch p = {tr(p_w / 10 ** us, 1)} × 10$^{{{us}}}$")
    fig.tight_layout()
    kaydet(fig, cikti, "sekil10_golub_pc_kayma.png")

    with open(os.path.join(cikti, "sekil_kontrol_sayilari.json"), "w", encoding="utf-8") as f:
        json.dump(K, f, ensure_ascii=False, indent=1, default=str)
    print(json.dumps(K, ensure_ascii=False, indent=1, default=str))


if __name__ == "__main__":
    if len(sys.argv) not in (3, 4):
        sys.exit("Kullanım: python sekilleri_yeniden_uret.py <golub_veri_klasoru> <cikti_klasoru> [<gse_tablo_klasoru>]")
    main(*sys.argv[1:])
