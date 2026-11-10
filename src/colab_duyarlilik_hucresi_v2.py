# Duyarlılık analizi (rapor Tablo VI) ve Şekil 12 — v2
# v1'den fark: Golub'da sabit problar nunique ile çıkarılır (6.832 prob); sayılar Bölüm IV-K ile birebir aynıdır.
# Çalıştırma (Colab): from google.colab import drive; drive.mount('/content/drive');
#   exec(open('/content/drive/MyDrive/Donem_Projesi_Losemi/02_Colab/colab_duyarlilik_hucresi_v2.py', encoding='utf-8').read())
import sys, gzip, numpy as np, pandas as pd
KOK = "/content/drive/MyDrive/Donem_Projesi_Losemi"
sys.path.append(f"{KOK}/02_Colab")
from ek_analizler import duyarlilik_analizi, latex_satirlari, gse_log_sekli

G = f"{KOK}/01_Ana_Veri_Seti_Golub"
M = f"{KOK}/03_Ikinci_Veri_Seti_GSE13159"
OUT = f"{KOK}/06_Ara_Rapor/ek_analiz_cikti"


def gpl_oku(yol):
    with gzip.open(yol, "rt", encoding="latin-1") as f:
        satirlar = f.read().splitlines()
    bas = next(i for i, s in enumerate(satirlar) if s.startswith("!platform_table_begin")) + 1
    son = next(i for i, s in enumerate(satirlar) if s.startswith("!platform_table_end"))
    from io import StringIO
    t = pd.read_csv(StringIO("\n".join(satirlar[bas:son])), sep="\t", dtype=str)
    return t.set_index("ID")["Gene symbol"]


def golub_oku():
    tr = pd.read_csv(f"{G}/data_set_ALL_AML_train.csv")
    te = pd.read_csv(f"{G}/data_set_ALL_AML_independent.csv")
    etiket = pd.read_csv(f"{G}/actual.csv").set_index("patient")["cancer"]
    parca = []
    for df in (tr, te):
        df = df.set_index("Gene Accession Number")
        df = df[[c for c in df.columns if c.strip().isdigit()]]
        parca.append(df)
    X = pd.concat(parca, axis=1)
    X.columns = X.columns.astype(int)
    X = np.log2(X.clip(20, 16000))
    X = X[X.nunique(axis=1) > 1]   # sabit problar (297) çıkarılır -> 6.832 prob
    y = (etiket.loc[X.columns] == "ALL").astype(int).values
    return X, y


Xg, yg = golub_oku()
print("Golub:", Xg.shape, "ALL =", yg.sum())          # beklenen: (6832, 72), 47

Xm = pd.read_parquet(f"{M}/GSE13159_ALL_AML_expression.parquet")
if not str(Xm.index[0]).endswith("_at"):
    Xm = Xm.T
Xm = Xm.dropna(how="all")                              # 45 boş prob
lab = pd.read_csv(f"{M}/GSE13159_ALL_AML_labels.csv")
sinif_sut = next(c for c in lab.columns if set(lab[c].astype(str).unique()) <= {"ALL", "AML"})
ornek_sut = next(c for c in lab.columns if lab[c].astype(str).str.startswith("GSM").all())
lab = lab.set_index(ornek_sut)[sinif_sut]
Xm = Xm[[c for c in Xm.columns if c in lab.index]]
ym = (lab.loc[Xm.columns] == "ALL").astype(int).values
print("GSE13159:", Xm.shape, "ALL =", ym.sum())       # beklenen: (54630, 1292), 750

h80 = gpl_oku(f"{M}/GPL80.annot.gz")
h570 = gpl_oku(f"{M}/GPL570.annot.gz")

ref = pd.read_csv(f"{KOK}/06_Ara_Rapor/ek_analiz_cikti/gen_duzeyi_iki_set.csv")
gsut = next((c for c in ref.columns if str(c).lower() in ("gen","gene","sembol","symbol","gen_sembolu","gene_symbol")), ref.columns[0])
ortak = set(ref[gsut].astype(str))
print("Evren:", len(ortak))                            # beklenen: 5829

tablo = duyarlilik_analizi(Xg, yg, h80, Xm, ym, h570, ortak_genler=ortak)
pd.set_option("display.width", 200)
print(tablo.round(3))   # maks_d: 359 / 749 / 238 / 237, r = 0,61
tablo.to_csv(f"{OUT}/duyarlilik_tablosu_v2.csv")
print(latex_satirlari(tablo))

gse_log_sekli(Xm, c=1e-3, dosya=f"{OUT}/sekil12_log_oncesi_sonrasi.png")
