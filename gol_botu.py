"""Haftalık gol analizi botu.

football-data.org verisiyle, gol bol ligler için önümüzdeki 7 günün maçlarında
2.5+ / 3.5+ / 4.5+ / 5+ gol olasılığını Poisson modeliyle hesaplar ve
rapor.html üretir. Sadece standart kütüphane kullanır.

Çalıştırma:  python gol_botu.py
"""
import json
import math
import os
import sys
import time
import urllib.request
from datetime import datetime, timedelta, timezone

KLASOR = os.path.dirname(os.path.abspath(__file__))
ANAHTAR = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
if not ANAHTAR:
    ANAHTAR = open(os.path.join(KLASOR, "anahtar.txt"), encoding="utf-8").read().strip()
API = "https://api.football-data.org/v4"

LIGLER = {
    "BL1": "Bundesliga",
    "DED": "Eredivisie",
    "PL": "Premier Lig",
    "ELC": "Championship",
    "PD": "La Liga",
    "FL1": "Ligue 1",
    "SA": "Serie A",
    "PPL": "Primeira Liga",
    "BSA": "Brezilya Serie A",
    "CL": "Şampiyonlar Ligi",
}
TAKVIM_YILI_SEZON = {"BSA"}   # sezonu Ocak-Aralık olan ligler
ESIKLER = [2.5, 3.5, 4.5, 6]   # 6 => 6 ve üzeri gol (5+ zaten 4.5+ ile aynıdır)
ALT_ESIKLER = [1.5, 2.5]        # alt listeleri: 1.5 alt = en fazla 1 gol, 2.5 alt = en fazla 2 gol
GUN_ILERI = 7
YARI_OMUR_GUN = 120            # eski maçların ağırlığı bu sürede yarıya iner
SHRINK = 6.0                   # az maçı olan takımı lig ortalamasına çeker
RAPOR_ILK = 10

# backtest.py'nin bulduğu ayarlar (varsa)
PERFORMANS = {}
ALT_KALIB = {}
try:
    _a = json.load(open(os.path.join(KLASOR, "ayarlar.json"), encoding="utf-8"))
    SHRINK, YARI_OMUR_GUN = _a["shrink"], _a["yari_omur"]
    PERFORMANS = _a.get("performans", {})
    ALT_KALIB = _a.get("alt_kalibrasyon", {})
except (OSError, KeyError, ValueError):
    pass


def api_get(yol):
    for deneme in range(3):
        try:
            req = urllib.request.Request(API + yol, headers={"X-Auth-Token": ANAHTAR})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(65)
                continue
            if e.code in (403, 404):
                return None
            raise
    return None


def sezon_maclari(kod, sezon):
    onbellek = os.path.join(KLASOR, "onbellek", f"{kod}_{sezon}.json")
    if os.path.exists(onbellek) and time.time() - os.path.getmtime(onbellek) < 6 * 3600:
        return json.load(open(onbellek, encoding="utf-8"))
    veri = api_get(f"/competitions/{kod}/matches?season={sezon}")
    time.sleep(7)  # ücretsiz paket: dakikada 10 istek
    maclar = veri["matches"] if veri else []
    os.makedirs(os.path.dirname(onbellek), exist_ok=True)
    json.dump(maclar, open(onbellek, "w", encoding="utf-8"))
    return maclar


def alt_olasilik(p_ust, e):
    """Alt olasılığı. Backtest'te doğrulanan eşiklerde (1.5, 2.5) düzeltme uygulanır."""
    ab = ALT_KALIB.get(str(e))
    p = min(max(1 - p_ust, 1e-6), 1 - 1e-6)
    if not ab:
        return p
    return 1 / (1 + math.exp(-(ab[0] + ab[1] * math.log(p / (1 - p)))))


def ust_olasilik(p_ust, e):
    """Üst olasılığı; alt ile tutarlı olsun diye aynı düzeltmeyi kullanır."""
    if str(e) in ALT_KALIB:
        return 1 - alt_olasilik(p_ust, e)
    return p_ust


def pois_pmf(lam, kmax=15):
    p = [math.exp(-lam)]
    for k in range(1, kmax + 1):
        p.append(p[-1] * lam / k)
    return p


def toplam_gol_dagilimi(lh, la):
    ph, pa = pois_pmf(lh), pois_pmf(la)
    d = [0.0] * 31
    for i, a in enumerate(ph):
        for j, b in enumerate(pa):
            d[i + j] += a * b
    return d


def olasilik_ustu(d, esik):
    en_az = math.floor(esik) + 1 if esik != int(esik) else int(esik)
    return sum(d[en_az:])


def model_kur(maclar, simdi):
    """Takım atak/defans güçleri (ev/deplasman ayrı), zaman ağırlıklı."""
    biten = [m for m in maclar if m["status"] == "FINISHED"
             and m["score"]["fullTime"]["home"] is not None]
    if len(biten) < 30:
        return None
    atilan_ev = atilan_dep = w_toplam = 0.0
    kayit = []
    for m in biten:
        tarih = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
        w = 0.5 ** ((simdi - tarih).days / YARI_OMUR_GUN)
        h, a = m["score"]["fullTime"]["home"], m["score"]["fullTime"]["away"]
        atilan_ev += w * h
        atilan_dep += w * a
        w_toplam += w
        kayit.append((m["homeTeam"]["id"], m["awayTeam"]["id"], h, a, w))
    ort_ev, ort_dep = atilan_ev / w_toplam, atilan_dep / w_toplam

    # Takım başına ağırlıklı toplamlar
    t = {}
    for ev, dep, h, a, w in kayit:
        e = t.setdefault(ev, dict(wh=0, gf_h=0, ga_h=0, wa=0, gf_a=0, ga_a=0))
        e["wh"] += w; e["gf_h"] += w * h; e["ga_h"] += w * a
        d = t.setdefault(dep, dict(wh=0, gf_h=0, ga_h=0, wa=0, gf_a=0, ga_a=0))
        d["wa"] += w; d["gf_a"] += w * a; d["ga_a"] += w * h

    def guc(toplam, w, ort):
        # lig ortalamasına SHRINK maçlık sanal veriyle çekilir
        return (toplam + SHRINK * ort) / (w + SHRINK) / ort

    guclar = {}
    for k, v in t.items():
        guclar[k] = dict(
            atak_ev=guc(v["gf_h"], v["wh"], ort_ev),
            def_ev=guc(v["ga_h"], v["wh"], ort_dep),
            atak_dep=guc(v["gf_a"], v["wa"], ort_dep),
            def_dep=guc(v["ga_a"], v["wa"], ort_ev),
            ort_gf=(v["gf_h"] + v["gf_a"]) / max(v["wh"] + v["wa"], 1e-9),
            ort_ga=(v["ga_h"] + v["ga_a"]) / max(v["wh"] + v["wa"], 1e-9),
            mac=round(v["wh"] + v["wa"]),
        )
    return dict(ort_ev=ort_ev, ort_dep=ort_dep, guclar=guclar, n=len(biten))


def main():
    simdi = datetime.now(timezone.utc)
    bitis = simdi + timedelta(days=GUN_ILERI)
    tahminler = []

    for kod, ad in LIGLER.items():
        print(f"{ad} verisi çekiliyor...", flush=True)
        if kod in TAKVIM_YILI_SEZON:
            sezon_bu = simdi.year
        else:
            sezon_bu = simdi.year if simdi.month >= 7 else simdi.year - 1
        bu = sezon_maclari(kod, sezon_bu)
        onceki = sezon_maclari(kod, sezon_bu - 1)
        hepsi = [m for m in onceki + bu]
        model = model_kur(hepsi, simdi)
        if not model:
            print(f"  {ad}: yeterli veri yok, atlandı")
            continue
        for m in bu:
            if m["status"] not in ("SCHEDULED", "TIMED"):
                continue
            tarih = datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))
            if not (simdi <= tarih <= bitis):
                continue
            ev, dep = m["homeTeam"]["id"], m["awayTeam"]["id"]
            g = model["guclar"]
            if ev not in g or dep not in g:
                continue
            lh = model["ort_ev"] * g[ev]["atak_ev"] * g[dep]["def_dep"]
            la = model["ort_dep"] * g[dep]["atak_dep"] * g[ev]["def_ev"]
            d = toplam_gol_dagilimi(lh, la)
            tahminler.append(dict(
                lig=ad,
                tarih=tarih,
                ev=m["homeTeam"]["shortName"] or m["homeTeam"]["name"],
                dep=m["awayTeam"]["shortName"] or m["awayTeam"]["name"],
                lh=lh, la=la,
                p={e: olasilik_ustu(d, e) for e in sorted(set(ESIKLER) | set(ALT_ESIKLER))},
                ev_gf=g[ev]["ort_gf"], ev_ga=g[ev]["ort_ga"],
                dep_gf=g[dep]["ort_gf"], dep_ga=g[dep]["ort_ga"],
                mac_ev=g[ev]["mac"], mac_dep=g[dep]["mac"],
            ))

    if not tahminler:
        print("Önümüzdeki günlerde analiz edilecek maç bulunamadı.")
        sys.exit(1)
    ort = sum(t["lh"] + t["la"] for t in tahminler) / len(tahminler)
    print(f"Ortalama beklenen toplam gol: {ort:.2f}")
    rapor_yaz(tahminler, simdi)
    print(f"\n{len(tahminler)} maç analiz edildi -> rapor.html")


def rapor_yaz(tahminler, simdi):
    bolumler = []
    # (eşik, yön): önce üst listeleri, sonra alt listeleri
    bolum_listesi = [(e, "ust") for e in ESIKLER] + [(e, "alt") for e in ALT_ESIKLER]
    for e, yon in bolum_listesi:
        alt = yon == "alt"
        ad = f"{e} ALT (en fazla {int(e)} gol)" if alt else ("6+ gol" if e == 6 else f"{e}+ gol")
        pv = (lambda x: alt_olasilik(x["p"][e], e)) if alt else (lambda x: ust_olasilik(x["p"][e], e))
        sirali = sorted(tahminler, key=pv, reverse=True)[:RAPOR_ILK]
        satirlar = []
        for i, t in enumerate(sirali, 1):
            yerel = t["tarih"].astimezone(timezone(timedelta(hours=3)))
            karsit = (f"<br><small>Üst %{ust_olasilik(t['p'][e], e)*100:.0f}</small>" if alt
                      else f"<br><small>Alt %{alt_olasilik(t['p'][e], e)*100:.0f}</small>")
            satirlar.append(
                f"<tr><td>{i}</td><td>{t['ev']} - {t['dep']}<br>"
                f"<small>{t['lig']} · {yerel:%d.%m %H:%M}</small></td>"
                f"<td class='y'>%{pv(t)*100:.0f}{karsit}</td>"
                f"<td>{t['lh']+t['la']:.2f}</td>"
                f"<td><small>{t['ev']}: {t['ev_gf']:.1f} at / {t['ev_ga']:.1f} ye<br>"
                f"{t['dep']}: {t['dep_gf']:.1f} at / {t['dep_ga']:.1f} ye</small></td></tr>")
        p = PERFORMANS.get(("alt" if alt else "") + str(e))
        if p:
            nadir = ""
            if not alt and e >= 6:
                nadir = " Bu eşik çok nadirdir ve model burada gerçekte olandan yaklaşık 2 kat iyimser kalır; sıralamaya bak, yüzdeye fazla güvenme."
            elif alt:
                nadir = " Alt tahminleri düzeltilmiştir; yine de en düşük gollü maçı seçmek üst tahmininden daha zordur."
            not_ = (f"<p class='not'>Geçmiş performans ({p['hafta']} hafta test): haftanın ilk 10 maçından "
                    f"ortalama <b>{p['haftada']:.1f}</b> tanesi tuttu (%{p['isabet']*100:.0f}).{nadir}</p>")
        else:
            not_ = ""
        bolumler.append(
            f"<h2>{ad}</h2>{not_}<table><tr><th>#</th><th>Maç</th><th>Olasılık</th>"
            f"<th>Beklenen gol</th><th>Maç başı ort. (atılan / yenen)</th></tr>"
            + "".join(satirlar) + "</table>")

    html = f"""<!doctype html><html lang="tr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="robots" content="noindex,nofollow">
<title>Haftalık Gol Analizi</title>
<style>
body{{font-family:system-ui,sans-serif;max-width:900px;margin:0 auto;padding:16px;background:#0f1419;color:#e6e6e6}}
h1{{margin-bottom:4px}} h2{{margin-top:32px;color:#7fd1ae}}
table{{width:100%;border-collapse:collapse;font-size:15px}}
th,td{{padding:8px;border-bottom:1px solid #2a323c;text-align:left;vertical-align:top}}
th{{color:#9aa7b4;font-weight:600;font-size:13px}} small{{color:#9aa7b4}}
.not{{color:#9aa7b4;font-size:14px;margin:4px 0 10px}}
@media(max-width:640px){{th:nth-child(4),td:nth-child(4),th:nth-child(5),td:nth-child(5){{display:none}}body{{padding:10px}}th,td{{padding:6px 4px}}}}
.y small{{font-size:12px;font-weight:500;color:#9aa7b4}}
.y{{font-weight:700;font-size:18px;color:#ffd166}}
.uyari{{background:#2a1f1f;border-left:4px solid #e05d5d;padding:12px;margin:16px 0;font-size:14px}}
</style></head><body>
<h1>Haftalık Gol Analizi</h1>
<small>Üretildi: {simdi.astimezone(timezone(timedelta(hours=3))):%d.%m.%Y %H:%M} · {len(tahminler)} maç tarandı · Poisson modeli</small>
<div class="uyari"><b>Önemli:</b> Bunlar kesinlik değil olasılıktır. Bir maç listenin tepesinde olsa
bile o eşiğin tutma ihtimali genelde %100'ün çok altındadır; özellikle 4.5+ ve 6+ gol nadir olaylardır.
Bahis oranları bu bilgiyi zaten fiyatlar, bu rapor kâr garantisi vermez. Sadece analiz aracıdır.</div>
{''.join(bolumler)}
</body></html>"""
    with open(os.path.join(KLASOR, "rapor.html"), "w", encoding="utf-8") as f:
        f.write(html)


if __name__ == "__main__":
    main()
