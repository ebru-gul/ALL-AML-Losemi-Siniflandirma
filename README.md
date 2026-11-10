# Mikrodizi Gen İfadesi Verisinden ALL/AML Lösemi Alt Tipi Sınıflandırması

**Ebru Gül** (503020260037) · Dönem Projesi, Ara Rapor

Bu depoda ara rapordaki keşifsel veri analizinin (EDA) kodları yer alır.

## Veri

| Veri seti | Kaynak | Platform | Örnek | Prob | Sınıf dağılımı |
|---|---|---|---|---|---|
| Golub ve ark. (1999) | Kaggle `crawford/gene-expression` | Hu6800 (GPL80) | 72 | 7.129 | ALL 47 / AML 25 |
| MILE (GSE13159) | NCBI GEO | HG-U133 Plus 2.0 (GPL570) | 1.292 | 54.675 | ALL 750 / AML 542 |

Ham veri depoda bulunmaz. İndirme bağlantıları ve klasör yapısı `data/README.md` dosyasındadır.

## Klasör yapısı

```
src/         Golub EDA, betimsel istatistikler, aykırı örnek kontrolü ve şekil betikleri
notebooks/   GSE13159 EDA ve iki veri setini birlikte kullanan analizler (Colab)
data/        Veri indirme açıklaması (ham veri yok)
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
python src/golub_eda_ek.py data/golub results/golub
python src/golub_ek_betimsel_aykiri.py data/golub results
python src/sekilleri_yeniden_uret.py data/golub results/sekiller results
```

GSE13159 analizleri `notebooks/` altındaki defterlerle Google Colab'da çalıştırılır.

## Rapor ile eşleşme

| Rapor bölümü | Kod |
|---|---|
| IV-A – IV-F (Golub EDA) | `src/golub_eda_ek.py` |
| Tablo VII, Ek D | `src/golub_ek_betimsel_aykiri.py` |
| Şekil 1a, 2–10, 19 | `src/sekilleri_yeniden_uret.py` |
| IV-G – IV-J (GSE13159 EDA) | `notebooks/GSE13159_EDA_v3.ipynb` |
| IV-K, Tablo V–VI | `notebooks/EDA_Ek_Analizler.ipynb`, `src/ek_analizler.py` |

Tüm rastgele işlemlerde tohum değeri 42'dir.

## Yapay zekâ kullanımı

Yapay zekâ araçlarının kullanımı raporun Ek E bölümünde beyan edilmiştir.
