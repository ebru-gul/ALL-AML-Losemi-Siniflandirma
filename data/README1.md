# Veri

Ham veri depoda bulunmaz. Dosyaları indirip aşağıdaki yapıya yerleştirin:

```
data/
  golub/      data_set_ALL_AML_train.csv, data_set_ALL_AML_independent.csv, actual.csv
  gse13159/   GSE13159_series_matrix.txt.gz
```

- **Golub:** https://www.kaggle.com/datasets/crawford/gene-expression
- **GSE13159:** https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE13159

GSE13159 ifade matrisi ve etiket dosyası `notebooks/GSE13159_EDA_v3.ipynb` içinde seri matrisinden üretilir.
