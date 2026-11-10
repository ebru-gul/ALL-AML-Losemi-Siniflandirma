"""Golub ALL/AML — ara rapor için tamamlayıcı EDA (rehber §4 madde 4). v5: gen sembolleri (ZYX vb.), etiket kaydırma, ondalık virgül.
Kullanım: python golub_eda_ek.py <veri_klasoru> <cikti_klasoru>
Girdi: <veri_klasoru>/data_set_ALL_AML_train.csv, data_set_ALL_AML_independent.csv, actual.csv
Çıktı: <cikti_klasoru>/golub_fig1..7.png, golub_eda_sonuclar.json, golub_ayirt_edici_genler.csv
Sabit tohum: 42. Etiketler keşif amaçlı kullanılır; modelleme yapılmaz.
Kütüphaneler: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn.
"""
import re, json, sys, os
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_rand_score

SEED = 42
D = sys.argv[1]; OUT = sys.argv[2]; os.makedirs(OUT, exist_ok=True)
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)
PAL = {"ALL": "#1f77b4", "AML": "#d9541e"}
R = {}
# --- v5 ekleri: Türkçe sayı biçimi ve gen sembolü eşlemesi (Şekil 2/5 tutarlılığı) ---
def tr(x, nd=3):
    """Ondalık virgül: 0.504 -> '0,504'."""
    return f"{x:.{nd}f}".replace(".", ",")
SEMBOL = {"U46499_at": "MGST1", "M84526_at": "CFD", "L09209_s_at": "APLP2", "M27891_at": "CST3", "M23197_at": "CD33",
          "M19507_at": "MPO", "X95735_at": "ZYX", "M31523_at": "TCF3", "M11722_at": "DNTT", "U05259_rna1_at": "CD79A",
          "M89957_at": "CD79B", "M84371_rna1_s_at": "CD19", "M92287_at": "CCND3", "Y12670_at": "LEPR", "X17042_at": "SRGN",
          "M96326_rna1_at": "AZU1", "U22376_cds2_s_at": "MYB", "M63138_at": "CTSD", "D88422_at": "CSTA", "X04085_rna1_at": "CAT",
          "M55150_at": "FAH", "U50136_rna1_at": "LTC4S", "M28130_rna1_s_at": "IL8", "X14008_rna1_f_at": "LYZ", "M16038_at": "LYN"}  # GPL80 anotasyonuyla doğrulanmalı
def sembol(prob, desc_str):
    """Prob için kısa gen sembolü; eşleme yoksa açıklamanın ilk sözcüğü."""
    s = SEMBOL.get(prob, "")
    return s if s else str(desc_str).split(" ")[0][:14]

train = pd.read_csv(f"{D}/data_set_ALL_AML_train.csv")
test = pd.read_csv(f"{D}/data_set_ALL_AML_independent.csv")
actual = pd.read_csv(f"{D}/actual.csv")
lab = actual.set_index("patient")["cancer"]

def split(df):
    pats = [c for c in df.columns if re.fullmatch(r"\d+", str(c))]
    X = df[pats].apply(pd.to_numeric).T
    X.index = [int(p) for p in pats]
    X.columns = df["Gene Accession Number"].values
    calls = df[[c for c in df.columns if str(c).startswith("call")]].to_numpy()
    return X, calls, [int(p) for p in pats]

Xtr, Ctr, ptr = split(train); Xte, Cte, pte = split(test)
assert Xtr.shape == (38, 7129) and Xte.shape == (34, 7129)
# etiket-sütun eşleştirme doğrulaması
ytr = lab.reindex(Xtr.index); yte = lab.reindex(Xte.index)
assert ytr.notna().all() and yte.notna().all() and not set(ptr) & set(pte)
R["train_order_sorted"] = bool(ptr == sorted(ptr)); R["test_order_sorted"] = bool(pte == sorted(pte))
R["train_patient_order_head"] = ptr[:10] + ["..."] + ptr[-12:]
R["train_counts"] = ytr.value_counts().to_dict(); R["test_counts"] = yte.value_counts().to_dict()
R["n_nan"] = int(train.isna().sum().sum() + test.isna().sum().sum())
R["n_inf"] = int(np.isinf(Xtr.to_numpy()).sum() + np.isinf(Xte.to_numpy()).sum())

# call sütunları
allc = np.concatenate([Ctr.ravel(), Cte.ravel()])
vc = pd.Series(allc).value_counts(); R["call_counts"] = vc.to_dict()
R["call_pct"] = (100 * vc / vc.sum()).round(1).to_dict()
# A çağrısı alan değerlerin negatif olma oranı
neg_by_call = {}
for name, X, C in [("train", Xtr, Ctr), ("test", Xte, Cte)]:
    V = X.T.to_numpy()
    for c in ["P", "M", "A"]:
        m = C == c
        neg_by_call.setdefault(c, []).append((int((V[m] < 0).sum()), int(m.sum())))
R["neg_frac_by_call"] = {c: round(100 * sum(a for a, b in v) / sum(b for a, b in v), 1) for c, v in neg_by_call.items()}

# betimsel + aykırı değerler
Xall = pd.concat([Xtr, Xte]); yall = pd.concat([ytr, yte])
vals = Xall.to_numpy().ravel()
R["all_desc"] = {k: float(v) for k, v in pd.Series(vals).describe().items()}
R["neg_pct_all"] = round(100 * float((vals < 0).mean()), 1)
R["pct_gt_16000"] = round(100 * float((vals > 16000).mean()), 3)
R["pct_lt_20"] = round(100 * float((vals < 20).mean()), 1)
R["n_gt_16000"] = int((vals > 16000).sum()); R["n_lt_20"] = int((vals < 20).sum())
# örnek bazlı IQR aykırılık (örnek medyanı)
med = Xall.median(axis=1); q1, q3 = med.quantile([.25, .75]); iqr = q3 - q1
out_s = med[(med < q1 - 1.5 * iqr) | (med > q3 + 1.5 * iqr)]
R["sample_median_outliers"] = {int(k): float(v) for k, v in out_s.items()}
R["sample_median_range"] = [float(med.min()), float(med.max())]
# düşük varyans / ifade filtresi öngörüsü
clipped = Xall.clip(20, 16000)
present_frac = (Xall > 100).mean(axis=0)
R["n_genes_present_lt_10pct"] = int((present_frac < 0.10).sum())
R["n_genes_maxmin_filter"] = int(((clipped.max() / clipped.min() > 5) & ((clipped.max() - clipped.min()) > 500)).sum())

# ---- Şekil 1: sınıf dağılımı + örnek dağılımları (eğitim vs test)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
cnt = pd.DataFrame({"Eğitim (n=38)": ytr.value_counts(), "Bağımsız test (n=34)": yte.value_counts()}).loc[["ALL", "AML"]]
cnt.plot.bar(ax=ax[0], color=["#4c72b0", "#8c8c8c"], rot=0, edgecolor="black", linewidth=.5)
for p in ax[0].patches:
    ax[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=10)
ax[0].set_ylabel("Hasta sayısı"); ax[0].set_xlabel(""); ax[0].set_title("(a) Sınıf dağılımı"); ax[0].set_ylim(0, 32)
ax[0].legend(frameon=False, fontsize=9)
logX = np.sign(Xall) * np.log2(1 + np.abs(Xall))
df_box = pd.DataFrame({"deger": logX.to_numpy().ravel(), "hasta": np.repeat(Xall.index, Xall.shape[1])})
order = list(Xtr.index) + list(Xte.index)
bp = sns.boxplot(data=df_box, x="hasta", y="deger", order=order, ax=ax[1], fliersize=0, linewidth=.5,
                 palette=[PAL[yall[h]] for h in order], hue="hasta", legend=False)
ax[1].axvline(37.5, color="k", ls="--", lw=1); ax[1].text(18, 15.5, "eğitim", ha="center"); ax[1].text(54, 15.5, "bağımsız test", ha="center")
ax[1].set_xticks([])
ax[1].set_xlabel("72 hasta, dosya sırasında (mavi: ALL, turuncu: AML)"); ax[1].set_ylabel("İşaretli log$_2$ ifade"); ax[1].set_title("(b) Örnek başına dağılım")
ax[1].set_ylim(-14, 17)
plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig1_sinif_ornek.png", dpi=300); plt.close()

# ---- Şekil 2: ham vs log dağılım
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].hist(vals, bins=200, color="#4c72b0"); ax[0].set_yscale("log"); ax[0].set_xlabel("Ham ifade (MAS average difference)"); ax[0].set_ylabel("Ölçüm sayısı (log)")
ax[0].set_title("(a) Ham değerler, 72 örnek"); ax[0].axvline(0, color="k", lw=.8)
for cls in ["ALL", "AML"]:
    v = logX.loc[yall == cls].to_numpy().ravel()
    ax[1].hist(v, bins=120, alpha=.6, density=True, color=PAL[cls], label=cls)
ax[1].set_xlabel("İşaretli log$_2$(1+|x|)"); ax[1].set_ylabel("Yoğunluk"); ax[1].set_title("(b) İşaretli log$_2$ dönüşümü sonrası, sınıfa göre"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig2_dagilim.png", dpi=300); plt.close()

# ---- Ön işleme (keşif): kırpma + log2 + z-skor (tüm 72, yalnız EDA)
Xp = np.log2(Xall.clip(20, 16000))
Xp = Xp.loc[:, (Xp.max() - Xp.min()) > 0]  # ortamdan bağımsız sıfır varyans filtresi
Z = StandardScaler().fit_transform(Xp)
R["n_genes_after_zero_var"] = int(Xp.shape[1])

# ---- Şekil 3: PCA + scree
pca = PCA(n_components=20, random_state=SEED).fit(Z); P = pca.transform(Z)
ev = pca.explained_variance_ratio_ * 100
R["pca_ev"] = [round(float(e), 2) for e in ev[:5]]; R["pca_cum10"] = round(float(ev[:10].sum()), 1); R["pca_cum20"] = round(float(ev[:20].sum()), 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1.3, 1]})
is_tr = np.array([i in set(Xtr.index) for i in Xall.index])
for cls in ["ALL", "AML"]:
    m = (yall == cls).to_numpy()
    ax[0].scatter(P[m & is_tr, 0], P[m & is_tr, 1], c=PAL[cls], marker="o", s=45, label=f"{cls} – eğitim", edgecolor="k", linewidth=.4)
    ax[0].scatter(P[m & ~is_tr, 0], P[m & ~is_tr, 1], c=PAL[cls], marker="^", s=55, label=f"{cls} – bağımsız test", edgecolor="k", linewidth=.4)
ax[0].set_xlabel(f"PC1 (%{ev[0]:.1f})"); ax[0].set_ylabel(f"PC2 (%{ev[1]:.1f})"); ax[0].set_title("(a) PCA, 72 örnek"); ax[0].legend(frameon=False, fontsize=8)
ax[1].bar(range(1, 21), ev, color="gray"); ax[1].plot(range(1, 21), np.cumsum(ev), "o-", color="#1f77b4", label="kümülatif")
ax[1].set_xlabel("Bileşen"); ax[1].set_ylabel("Açıklanan varyans (%)"); ax[1].set_title("(b) Scree"); ax[1].legend(frameon=False)
plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig3_pca.png", dpi=300); plt.close()
# batch: PC'lerde eğitim vs test farkı (Welch t + Mann-Whitney), PC1-5
pc_batch = []
for k in range(5):
    t, p = stats.ttest_ind(P[is_tr, k], P[~is_tr, k], equal_var=False)
    u, pu = stats.mannwhitneyu(P[is_tr, k], P[~is_tr, k])
    pc_batch.append({"PC": k + 1, "p_welch": float(p), "p_mwu": float(pu)})
R["pc_batch"] = pc_batch
# batch: kaç gen eğitim vs test arasında farklı (BH)
def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1.0); n = p.size; o = np.argsort(p); q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]; out = np.empty(n); out[o] = np.minimum(q, 1); return out
tb, pb = stats.ttest_ind(Xp[is_tr], Xp[~is_tr], equal_var=False); qb = bh(pb)
R["batch_genes_q05"] = int((qb < .05).sum()); R["batch_genes_q05_pct"] = round(100 * float((qb < .05).mean()), 1)
# sınıf içi (yalnız ALL) eğitim-test farkı
mA = (yall == "ALL").to_numpy()
tA, pA = stats.ttest_ind(Xp[is_tr & mA], Xp[~is_tr & mA], equal_var=False); R["batch_genes_q05_withinALL"] = int((bh(pA) < .05).sum())
# örnek medyan farkı ham veri
R["median_raw_train"] = float(np.median(Xtr.to_numpy())); R["median_raw_test"] = float(np.median(Xte.to_numpy()))
R["mean_sample_median_train"] = float(med.loc[Xtr.index].mean()); R["mean_sample_median_test"] = float(med.loc[Xte.index].mean())
R["neg_pct_train"] = round(100 * float((Xtr.to_numpy() < 0).mean()), 1); R["neg_pct_test"] = round(100 * float((Xte.to_numpy() < 0).mean()), 1)

# ---- Şekil 4: t-SNE
ts = TSNE(n_components=2, perplexity=15, init="pca", random_state=SEED, learning_rate="auto").fit_transform(P[:, :20])
fig, ax = plt.subplots(figsize=(5.2, 4.4))
for cls in ["ALL", "AML"]:
    m = (yall == cls).to_numpy()
    ax.scatter(ts[m & is_tr, 0], ts[m & is_tr, 1], c=PAL[cls], marker="o", s=45, label=f"{cls} – eğitim", edgecolor="k", linewidth=.4)
    ax.scatter(ts[m & ~is_tr, 0], ts[m & ~is_tr, 1], c=PAL[cls], marker="^", s=55, label=f"{cls} – test", edgecolor="k", linewidth=.4)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2"); ax.set_title("t-SNE (20 PC girdi, perplexity 15)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig4_tsne.png", dpi=300); plt.close()

# ---- Ayırt edici genler: Welch t + MWU + BH + Cohen d (tüm 72; keşif)
A = Xp[mA].to_numpy(); B = Xp[~mA].to_numpy()
t, p_t = stats.ttest_ind(A, B, axis=0, equal_var=False); q = bh(p_t)
u, p_u = stats.mannwhitneyu(A, B, axis=0); q_u = bh(p_u)
sp = np.sqrt(((A.shape[0] - 1) * A.var(0, ddof=1) + (B.shape[0] - 1) * B.var(0, ddof=1)) / (A.shape[0] + B.shape[0] - 2))
d = (A.mean(0) - B.mean(0)) / np.where(sp == 0, np.nan, sp)
desc = train.set_index("Gene Accession Number")["Gene Description"]
de = pd.DataFrame({"prob": Xp.columns, "aciklama": desc.reindex(Xp.columns).values, "ALL_ort": A.mean(0), "AML_ort": B.mean(0),
                   "cohen_d": d, "p_welch": p_t, "q_welch": q, "p_mwu": p_u, "q_mwu": q_u}).sort_values("p_welch")
de.to_csv(f"{OUT}/golub_ayirt_edici_genler.csv", index=False)
R["de_q05"] = int((q < .05).sum()); R["de_q01"] = int((q < .01).sum()); R["de_q05_d1"] = int(((q < .05) & (np.abs(d) > 1)).sum())
R["de_q05_mwu"] = int((q_u < .05).sum())
R["top20"] = de.head(20)[["prob", "aciklama", "cohen_d", "q_welch"]].to_dict("records")
R["n_up_ALL_q01_d1"] = int(((q < .01) & (d > 1)).sum()); R["n_up_AML_q01_d1"] = int(((q < .01) & (d < -1)).sum())
# Golub (1999) imzasındaki bilinen genlerin sıralamadaki yeri
golub50 = ["M27891_at", "M84526_at", "X95735_at", "M19507_at", "M23197_at", "U46499_at", "M11722_at", "X17042_at", "M96326_rna1_at", "Y12670_at", "U22376_cds2_s_at"]
R["known_genes_rank"] = {g: {"rank": int(np.where(de.prob.values == g)[0][0]) + 1, "d": float(de.set_index("prob").loc[g, "cohen_d"]), "desc": str(desc.get(g, ""))[:50]} for g in golub50 if g in set(de.prob)}

# ---- Şekil 5: volkan
fig, ax = plt.subplots(figsize=(5.4, 4.4))
col = np.where((q < .01) & (d > 1), PAL["ALL"], np.where((q < .01) & (d < -1), PAL["AML"], "lightgray"))
ax.scatter(d, -np.log10(p_t), c=col, s=7, alpha=.7, linewidths=0)
# Etiket çakışmasını önlemek için (IGB/OBF-1 vb.) her etikete el ile kaydırma; sembol adları Şekil 6 ile aynı
kaydir = {}
for i, (_, r) in enumerate(de.head(8).iterrows()):
    kaydir[r.prob] = (6, 4 + 9 * (i % 3)) if r.cohen_d > 0 else (-6, 4 + 9 * (i % 3))
for _, r in de.head(8).iterrows():
    dx, dy = kaydir[r.prob]
    ax.annotate(sembol(r.prob, r.aciklama), (r.cohen_d, -np.log10(r.p_welch)), fontsize=7.5, xytext=(dx, dy), textcoords="offset points",
                ha="left" if dx > 0 else "right", arrowprops=dict(arrowstyle="-", lw=.4, color="gray"))
ax.axvline(1, ls=":", c="k", lw=.7); ax.axvline(-1, ls=":", c="k", lw=.7)
ax.set_xlabel("Cohen's d (ALL − AML)"); ax.set_ylabel("−log$_{10}$ p (Welch)"); ax.set_title("Volkan grafiği (72 örnek; q<0,01 ve |d|>1 renkli)"); ax.tick_params(labelsize=9)
ax.scatter([], [], c=PAL["ALL"], label=f"ALL'da yüksek ({R['n_up_ALL_q01_d1']})"); ax.scatter([], [], c=PAL["AML"], label=f"AML'de yüksek ({R['n_up_AML_q01_d1']})"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig5_volkan.png", dpi=300); plt.close()

# ---- Şekil 6: en iyi 50 gen ısı haritası + hiyerarşik kümeleme (örnekler)
top50 = de.head(50).prob.tolist()
H = pd.DataFrame(Z, index=Xall.index, columns=Xp.columns)[top50]
Lsamp = linkage(H.to_numpy(), method="average", metric="correlation")
Lgene = linkage(H.T.to_numpy(), method="average", metric="correlation")
cl2 = fcluster(Lsamp, 2, criterion="maxclust")
R["ari_top50_k2"] = round(float(adjusted_rand_score(yall, cl2)), 3)
R["ct_top50"] = pd.crosstab(yall.values, cl2).to_dict()
labels_row = [sembol(g, desc.get(g, g)) if g in SEMBOL else f"{desc.get(g, g)}"[:28] for g in top50]
g = sns.clustermap(H.T, row_linkage=Lgene, col_linkage=Lsamp, cmap="RdBu_r", vmin=-3, vmax=3, figsize=(11, 9.5),
                   col_colors=[PAL[c] for c in yall.loc[H.index]], yticklabels=labels_row, xticklabels=False,
                   cbar_kws={"label": "z-skoru"}, dendrogram_ratio=(.08, .12), cbar_pos=(.02, .83, .02, .12))
g.ax_heatmap.set_xlabel("Hastalar (üst şerit: mavi ALL, turuncu AML)"); g.ax_heatmap.set_ylabel("")
g.ax_heatmap.tick_params(axis="y", labelsize=7)
g.fig.suptitle("En ayırt edici 50 gen — hiyerarşik kümeleme (ortalama bağlantı, korelasyon)", y=1.0, fontsize=11)
g.savefig(f"{OUT}/golub_fig6_isi_haritasi.png", dpi=300, bbox_inches="tight"); plt.close("all")

# ---- Şekil 7: denetimsiz kümeleme (en değişken 1000 gen, Ward) + dendrogram
var_idx = np.argsort(Xp.var().to_numpy())[::-1][:1000]
Zv = Z[:, var_idx]; Lw = linkage(Zv, "ward")
aris = {k: round(float(adjusted_rand_score(yall, fcluster(Lw, k, "maxclust"))), 3) for k in [2, 3, 4]}
R["ari_var1000"] = aris; R["ct_var1000_k2"] = pd.crosstab(yall.values, fcluster(Lw, 2, "maxclust")).to_dict()
fig, ax = plt.subplots(figsize=(11, 4))
dn = dendrogram(Lw, ax=ax, labels=[f"{h}" for h in Xall.index], leaf_font_size=7, color_threshold=Lw[-1, 2] * .7)
for lbl in ax.get_xmajorticklabels():
    lbl.set_color(PAL[yall[int(lbl.get_text())]])
ax.set_title(f"Hiyerarşik kümeleme (Ward, en değişken 1.000 gen, etiket kullanılmadı) — k=2 ARI = {tr(aris[2])}")
ax.set_ylabel("Uzaklık"); ax.set_xlabel("Hasta no (mavi: ALL, turuncu: AML; test hastaları 39–72)")
ax.tick_params(axis="y", labelsize=9); plt.tight_layout(); plt.savefig(f"{OUT}/golub_fig7_dendrogram.png", dpi=300); plt.close()

json.dump(R, open(f"{OUT}/golub_eda_sonuclar.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(R, indent=1, ensure_ascii=False, default=str))
