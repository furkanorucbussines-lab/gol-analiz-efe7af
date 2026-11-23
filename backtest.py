"""Geriye dönük test + ayarlama.

Her hafta için "o gün elimde olan veriyle" tahmin yapar, gerçek sonuçla
karşılaştırır. En iyi SHRINK / YARI_OMUR değerlerini bulur, olasılıklar için
kalibrasyon (düzeltme) katsayılarını çıkarır ve kalibrasyon.json'a yazar.

Çalıştırma:  python backtest.py
"""
import json
import math
import os
from datetime import datetime, timedelta, timezone

import gol_botu as g

ESIKLER = g.ESIKLER


def tarih_of(m):
    return datetime.fromisoformat(m["utcDate"].replace("Z", "+00:00"))


def veri_yukle():
    d = {}
    for kod in g.LIGLER:
        for s in (2024, 2025, 2026):
            d[(kod, s)] = g.sezon_maclari(kod, s)
    return d


def tahmin_uret(veri, test_sezon, shrink, omur, atla_gun=45):
    """Test sezonunun biten maçları için (lig, tarih, lh, la, toplam_gol) döndürür."""
    g.SHRINK, g.YARI_OMUR_GUN = shrink, omur
    sonuc = []
    for kod in g.LIGLER:
        gecmis = veri[(kod, test_sezon - 1)] + veri[(kod, test_sezon)]
        test = [m for m in veri[(kod, test_sezon)]
                if m["status"] == "FINISHED" and m["score"]["fullTime"]["home"] is not None]
        if not test:
            continue
        test.sort(key=tarih_of)
        sezon_baslangic = tarih_of(test[0])
        # haftalık gruplar
        hafta = {}
        for m in test:
            t = tarih_of(m)
            if (t - sezon_baslangic).days < atla_gun:
                continue
            hafta.setdefault((t - sezon_baslangic).days // 7, []).append(m)
        for _, grup in sorted(hafta.items()):
            sinir = min(tarih_of(m) for m in grup)
            sinir = sinir.replace(hour=0, minute=0)
            onceki = [m for m in gecmis if tarih_of(m) < sinir]
            model = g.model_kur(onceki, sinir)
            if not model:
                continue
            for m in grup:
                ev, dep = m["homeTeam"]["id"], m["awayTeam"]["id"]
                gu = model["guclar"]
                if ev not in gu or dep not in gu:
                    continue
                lh = model["ort_ev"] * gu[ev]["atak_ev"] * gu[dep]["def_dep"]
                la = model["ort_dep"] * gu[dep]["atak_dep"] * gu[ev]["def_ev"]
                top = m["score"]["fullTime"]["home"] + m["score"]["fullTime"]["away"]
                sonuc.append((kod, tarih_of(m), lh, la, top))
    return sonuc


def olasiliklar(lh, la):
    d = g.toplam_gol_dagilimi(lh, la)
    return {e: g.olasilik_ustu(d, e) for e in sorted(set(ESIKLER) | set(g.ALT_ESIKLER))}


def gerceklesen(top, e):
    return 1.0 if top > e else 0.0


def skorlar(tahminler, kalib=None):
    """Eşik başına Brier ve log-loss."""
    out = {}
    for e in ESIKLER:
        br = ll = 0.0
        for _, _, lh, la, top in tahminler:
            p = olasiliklar(lh, la)[e]
            if kalib:
                p = kalib_uygula(p, kalib[str(e)])
            p = min(max(p, 1e-6), 1 - 1e-6)
            y = gerceklesen(top, e)
            br += (p - y) ** 2
            ll += -(y * math.log(p) + (1 - y) * math.log(1 - p))
        n = len(tahminler)
        out[e] = (br / n, ll / n)
    return out


def logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def kalib_uygula(p, ab):
    a, b = ab
    return sigmoid(a + b * logit(p))


def kalib_fit(ps, ys, iters=50):
    """p' = sigmoid(a + b*logit(p)) — Newton yöntemiyle lojistik regresyon."""
    a, b = 0.0, 1.0
    xs = [logit(p) for p in ps]
    for _ in range(iters):
        ga = gb = haa = hab = hbb = 0.0
        for x, y in zip(xs, ys):
            q = sigmoid(a + b * x)
            r = q - y
            w = q * (1 - q) + 1e-9
            ga += r; gb += r * x
            haa += w; hab += w * x; hbb += w * x * x
        det = haa * hbb - hab * hab
        if abs(det) < 1e-12:
            break
        da = (hbb * ga - hab * gb) / det
        db = (-hab * ga + haa * gb) / det
        a -= da; b -= db
        if abs(da) + abs(db) < 1e-9:
            break
    return a, b


def guvenirlik_tablosu(tahminler, kalib=None, e=4.5):
    kutular = [(0, .1), (.1, .2), (.2, .3), (.3, .4), (.4, .5), (.5, .6), (.6, 1.01)]
    satir = []
    for lo, hi in kutular:
        ps, ys = [], []
        for _, _, lh, la, top in tahminler:
            p = olasiliklar(lh, la)[e]
            if kalib:
                p = kalib_uygula(p, kalib[str(e)])
            if lo <= p < hi:
                ps.append(p); ys.append(gerceklesen(top, e))
        if ps:
            satir.append(f"   tahmin %{lo*100:.0f}-{min(hi,1)*100:.0f}: n={len(ps):4d}  "
                         f"ort. tahmin %{sum(ps)/len(ps)*100:.1f}  gerçek %{sum(ys)/len(ys)*100:.1f}")
    return "\n".join(satir)


def main():
    veri = veri_yukle()

    print("== 1) Parametre taraması (test: 2025 sezonu, her hafta geçmişle tahmin) ==")
    en_iyi = None
    for shrink in (3, 6, 10, 15, 25):
        for omur in (60, 120, 240, 100000):
            t = tahmin_uret(veri, 2025, shrink, omur)
            sk = skorlar(t)
            toplam_ll = sum(v[1] for v in sk.values())
            print(f"SHRINK={shrink:3d} YARI_OMUR={omur:6d}  maç={len(t)}  toplam log-loss={toplam_ll:.4f}")
            if en_iyi is None or toplam_ll < en_iyi[0]:
                en_iyi = (toplam_ll, shrink, omur)
    _, shrink, omur = en_iyi
    print(f"\nEn iyi: SHRINK={shrink}, YARI_OMUR_GUN={omur}\n")

    # kalibrasyon: 2025 (eğitim) -> 2026 sezonu (bağımsız test)
    egit = tahmin_uret(veri, 2025, shrink, omur)
    test = tahmin_uret(veri, 2026, shrink, omur, atla_gun=30)
    print(f"Kalibrasyon eğitim maçı: {len(egit)}, bağımsız test maçı (2026 sezonu): {len(test)}")

    kalib = {}
    for e in ESIKLER:
        ps = [olasiliklar(lh, la)[e] for _, _, lh, la, _ in egit]
        ys = [gerceklesen(top, e) for *_, top in egit]
        kalib[str(e)] = kalib_fit(ps, ys)

    print("\n== 2) Bağımsız testte (2026) kalibrasyon etkisi — düşük = iyi ==")
    once = skorlar(test)
    sonra = skorlar(test, kalib)
    for e in ESIKLER:
        print(f"{e:>4}+ : Brier {once[e][0]:.4f} -> {sonra[e][0]:.4f}   "
              f"log-loss {once[e][1]:.4f} -> {sonra[e][1]:.4f}   "
              f"(gerçek oran %{sum(gerceklesen(t[4], e) for t in test)/len(test)*100:.1f}, "
              f"ort. tahmin önce %{sum(olasiliklar(t[2], t[3])[e] for t in test)/len(test)*100:.1f} "
              f"sonra %{sum(kalib_uygula(olasiliklar(t[2], t[3])[e], kalib[str(e)]) for t in test)/len(test)*100:.1f})")

    print("\n== 3) 4.5+ güvenilirlik tablosu, bağımsız test, KALİBRASYONSUZ ==")
    print(guvenirlik_tablosu(test, None, 4.5))
    print("\n== 3b) 4.5+ güvenilirlik tablosu, KALİBRASYONLU ==")
    print(guvenirlik_tablosu(test, kalib, 4.5))
    print("\n== 3c) 6+ KALİBRASYONLU ==")
    print(guvenirlik_tablosu(test, kalib, 6))

    # Final: kalibrasyon KULLANILMIYOR (bağımsız testte fayda göstermedi).
    # Ayarlar + haftalık "ilk 10" geçmiş isabet oranları kaydedilir.
    perf = ilk10_performans(veri, shrink, omur)
    alt_kalib = {}
    for e in g.ALT_ESIKLER:
        ps = [1 - olasiliklar(lh, la)[e] for _, _, lh, la, _ in egit + test]
        ys = [1.0 if t <= e else 0.0 for *_, t in egit + test]
        alt_kalib[str(e)] = kalib_fit(ps, ys)
    kupon = kupon_gecmis(veri, shrink, omur)
    print("KUPON GEÇMİŞİ:", kupon)
    cikti = dict(shrink=shrink, yari_omur=omur, performans=perf, alt_kalibrasyon=alt_kalib, kupon_gecmis=kupon,
                 tarih=datetime.now(timezone.utc).isoformat())
    with open(os.path.join(g.KLASOR, "ayarlar.json"), "w", encoding="utf-8") as f:
        json.dump(cikti, f, indent=2)
    print("\nayarlar.json yazıldı:", json.dumps(cikti, indent=2))


def kupon_gecmis(veri, shrink, omur):
    """Aynı kupon kuralı geçmiş haftalarda uygulansaydı kaç seçim tutardı?"""
    hepsi = {}
    for sezon, atla in ((2025, 45), (2026, 30)):
        for kod, tar, lh, la, top in tahmin_uret(veri, sezon, shrink, omur, atla):
            hepsi.setdefault((sezon,) + tar.isocalendar()[:2], []).append((lh, la, top))
    sayim = {"hafta": 0, "isabet_toplam": 0, "en_az_6": 0, "en_az_7": 0, "en_az_8": 0}
    for l in hepsi.values():
        adaylar = []
        for lh, la, top in l:
            pm = olasiliklar(lh, la)
            bg = g.kupon_bacagi(pm)
            if bg:
                adaylar.append((bg[1], top > bg[0]))
        if len(adaylar) < g.KUPON_MAC or len(l) < 15:
            continue
        secim = sorted(adaylar, key=lambda x: -x[0])[:g.KUPON_MAC]
        tutan = sum(1 for _, ok in secim if ok)
        sayim["hafta"] += 1
        sayim["isabet_toplam"] += tutan
        for k in (6, 7, 8):
            if tutan >= k:
                sayim[f"en_az_{k}"] += 1
    sayim["ort_isabet"] = round(sayim["isabet_toplam"] / max(sayim["hafta"], 1), 2)
    return sayim


def ilk10_performans(veri, shrink, omur):
    """2025 + 2026 haftalarında, haftanın ilk 10 maçının gerçek isabet oranı."""
    hepsi = {}
    for sezon, atla in ((2025, 45), (2026, 30)):
        for kod, tar, lh, la, top in tahmin_uret(veri, sezon, shrink, omur, atla):
            hepsi.setdefault((sezon,) + tar.isocalendar()[:2], []).append((lh, la, top))
    out = {}
    for e, alt in [(e, False) for e in ESIKLER] + [(e, True) for e in g.ALT_ESIKLER]:
        isabet = n = 0
        haftalar = 0
        ort_tahmin = 0.0
        for l in hepsi.values():
            if len(l) < 15:
                continue
            ps = []
            for lh, la, top in l:
                p = olasiliklar(lh, la)[e]
                if alt:
                    ps.append((1 - p, top <= e))
                else:
                    ps.append((p, top > e))
            ps = sorted(ps, key=lambda x: -x[0])[:10]
            isabet += sum(1 for _, ok in ps if ok)
            ort_tahmin += sum(p for p, _ in ps)
            n += len(ps)
            haftalar += 1
        out[("alt" if alt else "") + str(e)] = dict(
            isabet=round(isabet / n, 4), hafta=haftalar,
            haftada=round(isabet / haftalar, 2), tahmin=round(ort_tahmin / n, 4))
    return out


if __name__ == "__main__":
    main()
