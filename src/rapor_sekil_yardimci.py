"""rapor_sekil_yardimci.py — v5 şekil düzeltmeleri için yardımcılar (GSE13159_EDA_v3.ipynb ve EDA_Ek_Analizler.ipynb).

Kullanım (defterin ilk kod hücresine, importlardan sonra):
    from rapor_sekil_yardimci import *
    rapor_stili()           # savefig.dpi=300 + büyütülmüş yazı boyutları (madde 8, 12, 13)

Sağladıkları:
    tr(x, nd)               ondalık virgül:      tr(0.504) -> '0,504'   (madde 14)
    tr_bilim(x)             bilimsel gösterim:   4.2e-3 -> 4,2×10⁻³ (mathtext)
    binlik(n)               binlik ayırıcı:      binlik(1292) -> '1.292' (madde 15)
    etiket_yerlestir(...)   volkan grafiğinde çakışmayan etiketler (madde 11; PNKD/CTSA/FNDC3B)
    log2_panel(...)         Şekil 12'ye log2(x+c) sonrası paneli ekler ve çarpıklıkları başlığa yazar (madde 6)
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats


def rapor_stili(taban=11):
    """savefig.dpi=300; eksen/başlık/etiket yazılarını büyütür (Şekil 11, 14, 15, 16, 17 için)."""
    matplotlib.rcParams.update({
        "savefig.dpi": 300,
        "figure.dpi": 100,
        "font.size": taban,
        "axes.titlesize": taban + 1,
        "axes.labelsize": taban,
        "xtick.labelsize": taban - 1,
        "ytick.labelsize": taban - 1,
        "legend.fontsize": taban - 1,
        "figure.titlesize": taban + 2,
    })


def tr(x, nd=3):
    """0.504 -> '0,504'; bilimsel gösterim için tr_bilim kullanın."""
    return f"{x:.{nd}f}".replace(".", ",")


def tr_bilim(x, nd=1):
    """4.2e-03 -> '4,2×10⁻³' (matplotlib mathtext: $4{,}2\\times10^{-3}$)."""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10 ** e
    return f"${m:.{nd}f}".replace(".", "{,}") + r"\times10^{" + str(e) + "}$"


def binlik(n):
    """1292 -> '1.292'."""
    return f"{int(n):,}".replace(",", ".")


def etiket_yerlestir(ax, xs, ys, metinler, min_dy=0.04, fontsize=8, sag_taraf=None):
    """Volkan grafiğinde etiketleri dikeyde iterek çakışmayı önler ve ince bir çizgiyle noktaya bağlar.
    xs, ys: veri koordinatları; sag_taraf: True ise etiketler sağa, False ise sola; None ise x işaretine göre.
    min_dy: eksen oranı cinsinden en az dikey aralık."""
    ymin, ymax = ax.get_ylim()
    xmin, xmax = ax.get_xlim()
    pts = sorted(zip(ys, xs, metinler), reverse=True)
    yerler = []
    for y, x, m in pts:
        yy = y
        for (yp, _) in yerler:
            if abs(yy - yp) < min_dy * (ymax - ymin):
                yy = yp - min_dy * (ymax - ymin)
        yerler.append((yy, (x, y, m)))
    for yy, (x, y, m) in yerler:
        sag = (x > 0) if sag_taraf is None else sag_taraf
        dx = 0.04 * (xmax - xmin) * (1 if sag else -1)
        ax.annotate(m, (x, y), xytext=(x + dx, yy), fontsize=fontsize, ha="left" if sag else "right", va="center",
                    arrowprops=dict(arrowstyle="-", lw=.4, color="gray"))


def log2_panel(degerler_all, degerler_aml, c=1e-3, dosya=None, renk=("#1f77b4", "#d9541e")):
    """Şekil 12 (GSE13159 dağılım): sol panel ham DQN değerleri, sağ panel log2(x+c) sonrası.
    Başlığa çarpıklıkları (örn. 0,697 -> −0,905) yazar. degerler_*: 1 boyutlu numpy dizileri (havuzlanmış)."""
    ham = np.concatenate([degerler_all, degerler_aml])
    lg_all, lg_aml = np.log2(degerler_all + c), np.log2(degerler_aml + c)
    sk_ham, sk_log = stats.skew(ham), stats.skew(np.concatenate([lg_all, lg_aml]))
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for a, (v1, v2), xl in [(ax[0], (degerler_all, degerler_aml), "DQN sinyali (0–1)"), (ax[1], (lg_all, lg_aml), f"log$_2$(x + {c:g})")]:
        a.hist(v1, bins=120, density=True, alpha=.6, color=renk[0], label="ALL")
        a.hist(v2, bins=120, density=True, alpha=.6, color=renk[1], label="AML")
        a.set_xlabel(xl); a.set_ylabel("Yoğunluk"); a.legend(frameon=False)
    ax[0].set_title(f"(a) Kantil normalize değerler (çarpıklık {tr(sk_ham)})")
    ax[1].set_title(f"(b) log$_2$(x+c) sonrası (çarpıklık {tr(sk_log)})")
    fig.suptitle(f"GSE13159: log$_2$ dönüşümü çarpıklığı {tr(sk_ham)}'den {tr(sk_log)}'e çevirir; uygulanmadı", y=1.02)
    plt.tight_layout()
    if dosya:
        plt.savefig(dosya, dpi=300, bbox_inches="tight")
    return fig, (sk_ham, sk_log)
