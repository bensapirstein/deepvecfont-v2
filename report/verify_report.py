"""Check every number quoted in REPORT.md against RESULTS.csv.

Usage:  python report/verify_report.py     (exit 0 = all claims verified)

Nothing in the report is typed by hand from memory. If RESULTS.csv changes,
rerun this and fix whatever it flags.
"""
import csv, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # repo root, from report/
R = list(csv.DictReader(open(os.path.join(ROOT, "RESULTS.csv"))))
FAIL = []


def rows(**kw):
    out = []
    for r in R:
        ok = True
        for k, v in kw.items():
            if k == "gtsvg":
                if ("gtsvg" in r["checkpoint"]) != v: ok = False
            elif str(r.get(k)) != str(v): ok = False
        if ok: out.append(r)
    return out


def one(**kw):
    m = rows(**kw); assert len(m) == 1, (kw, len(m)); return m[0]


f = lambda r, k: float(r[k])
mean = lambda x: sum(x) / len(x)
spread = lambda x: max(x) - min(x)


def chk(label, got, want, tol=6e-5):
    ok = abs(got - want) < tol
    print(f"{'ok  ' if ok else 'FAIL'}  {label:46s} {got:+.4f}  report says {want:+.4f}")
    if not ok: FAIL.append(label)


cb   = [one(name_exp=f"seedfloor_{s}_chn", epoch=150, n_samples=50) for s in (1111, 2222, 3333)]
cb3  = [one(name_exp=f"seedfloor_{s}_chn", epoch=150, n_samples=3)  for s in (1111, 2222, 3333)]
ce   = [one(name_exp=n, epoch=150, n_samples=50) for n in
        ("e9_sigma050_chn", "e9_sigma050_2222_chn", "e9_sigma050_3333_chn")]
eb   = [one(name_exp=n, epoch=e, n_samples=50) for n, e in
        (("eng_seedfloor_1111", 640), ("eng_seedfloor_2222", 580), ("eng_seedfloor_3333", 640))]
ee   = [one(name_exp=f"e9_sigma050_{s}_eng", epoch=e, n_samples=50) for s, e in
        ((1111, 640), (2222, 580), (3333, 640))]

print("\n-- section 3, reconstruction --")
chk("chn reconstruction mean L1",  mean([f(r,'l1')    for r in cb]), 0.1621)
chk("chn reconstruction s-IoU",    mean([f(r,'s_iou') for r in cb]), 0.2681)
chk("chn reconstruction SSIM",     mean([f(r,'ssim')  for r in cb]), 0.4425)
chk("eng reconstruction mean L1",  mean([f(r,'l1')    for r in eb]), 0.0597)
chk("eng reconstruction s-IoU",    mean([f(r,'s_iou') for r in eb]), 0.7309)
chk("eng reconstruction SSIM",     mean([f(r,'ssim')  for r in eb]), 0.7374)
oc = one(name_exp="official_chn", epoch=600, n_samples=50, gtsvg=False)
oe = one(name_exp="official_eng", epoch=600, n_samples=50, n_fonts=34, gtsvg=False)
chk("released chn 600 L1",    f(oc,'l1'),    0.1629)
chk("released chn 600 s-IoU", f(oc,'s_iou'), 0.3225)
chk("released chn 600 SSIM",  f(oc,'ssim'),  0.4373)
chk("released eng 600 L1",    f(oe,'l1'),    0.0658)
chk("released eng 600 s-IoU", f(oe,'s_iou'), 0.7029)
chk("released eng 600 SSIM",  f(oe,'ssim'),  0.7181)
chk("ours vs released, chn",  abs(mean([f(r,'l1') for r in cb]) - f(oc,'l1')), 0.0008)
chk("600-epoch own runs",     mean([f(one(name_exp=f"seedfloor600_{s}_chn", epoch=e, n_samples=50,
                                          gtsvg=False),'l1') for s, e in
                                    ((1111,200),(2222,150),(3333,200))]), 0.1583)
chk("rasterizer gap, chn",    f(oc,'l1') - f(one(name_exp="official_chn", epoch=600,
                                                 n_samples=50, gtsvg=True),'l1'), 0.0455)
chk("rasterizer gap, eng",    f(oe,'l1') - f(one(name_exp="official_eng", epoch=600, n_samples=50,
                                                 n_fonts=34, gtsvg=True),'l1'), 0.0074)
chk("N=50 vs N=10, eng",      f(one(name_exp="eng_seedfloor_1111", epoch=640, n_samples=10),'l1')
                              - f(one(name_exp="eng_seedfloor_1111", epoch=640, n_samples=50),'l1'), 0.0027)

print("\n-- section 4, the floor --")
chk("chn floor (screening)",  spread([f(r,'l1')    for r in cb3]), 0.0097)
chk("chn s-IoU floor",        spread([f(r,'s_iou') for r in cb3]), 0.0315)
chk("eng floor",              spread([f(r,'l1')    for r in eb]),  0.0038)
chk("chn s-IoU floor, n=50",  spread([f(r,'s_iou') for r in cb]),  0.0236)

print("\n-- section 5, the sweep --")
cands = [f(r,'l1') for r in R if r['language']=='chn' and r['n_samples']=='3'
         and r['batch'] in ('tier1','tier2','tier3a') and not r['name_exp'].startswith('seedfloor')]
chk("candidate count", len(cands), 26, tol=.5)
chk("candidate spread", spread(cands), 0.0101)
print("\n-- section 5.3, E9 chinese --")
for i, s in enumerate((1111, 2222, 3333)):
    chk(f"E9 chn seed {s} L1",    f(ce[i],'l1'),    [0.1588,0.1531,0.1623][i])
    chk(f"E9 chn seed {s} dL1",   f(ce[i],'l1')   - f(cb[i],'l1'),    [-0.0074,-0.0038,-0.0009][i])
    chk(f"E9 chn seed {s} ds-IoU",f(ce[i],'s_iou')- f(cb[i],'s_iou'), [ 0.0325, 0.0454, 0.0035][i])
chk("E9 chn mean L1",    mean([f(r,'l1')    for r in ce]), 0.1581)
chk("E9 chn mean s-IoU", mean([f(r,'s_iou') for r in ce]), 0.2952)
chk("E9 chn mean SSIM",  mean([f(r,'ssim')  for r in ce]), 0.4479)
chk("E9 chn delta L1",   mean([f(r,'l1')    for r in ce]) - mean([f(r,'l1')    for r in cb]), -0.0040)
chk("E9 chn delta s-IoU",mean([f(r,'s_iou') for r in ce]) - mean([f(r,'s_iou') for r in cb]),  0.0271)

print("\n-- section 5.4, E9 english --")
chk("E9 eng mean L1",    mean([f(r,'l1')    for r in ee]), 0.0601)
chk("E9 eng mean s-IoU", mean([f(r,'s_iou') for r in ee]), 0.7305)
chk("E9 eng mean SSIM",  mean([f(r,'ssim')  for r in ee]), 0.7358)
signs = [f(a,'l1') - f(b,'l1') for a, b in zip(ee, eb)]
print(f"{'ok  ' if not (all(s>0 for s in signs) or all(s<0 for s in signs)) else 'FAIL'}"
      f"  E9 eng mixed sign as reported          {[round(s,4) for s in signs]}")
if all(s > 0 for s in signs) or all(s < 0 for s in signs): FAIL.append("E9 eng mixed sign")

print("\n-- section 5.5, category representatives --")
base = {s: one(name_exp=f"seedfloor_{s}_chn", epoch=150, n_samples=3) for s in (1111, 2222, 3333)}
for k, pre, cl, ci in (("E2","c_e2_batchnorm",-0.0066,0.0221), ("E4","c_e4_ngf32",0.0059,0.0173),
                       ("E5","c_e5_bneck256",-0.0010,0.0203),  ("E7","c_e7_aux01",-0.0010,-0.0053),
                       ("E11","c_e11_adamw",-0.0011,0.0017),   ("E3","c_e3_refine2",-0.0004,0.0007),
                       ("E16","e16_depth8",0.0025,-0.0045),    ("E17","e17_dff2048",0.0005,0.0051)):
    g = lambda s: one(name_exp=f"{pre}_{s}_chn", epoch=150, n_samples=3)
    chk(f"{k} mean dL1",    mean([f(g(s),'l1')   -f(base[s],'l1')    for s in base]), cl)
    chk(f"{k} mean ds-IoU", mean([f(g(s),'s_iou')-f(base[s],'s_iou') for s in base]), ci)

print("\n-- section 5.5, E1 --")
o1 = {1111:("e1_norm_chn",150), 2222:("e1_norm_2222_chn",125), 3333:("e1_norm_3333_chn",100)}
chk("E1 as first read", mean([f(one(name_exp=n,epoch=e,n_samples=3),'s_iou')-f(base[s],'s_iou')
                              for s,(n,e) in o1.items()]), -0.0760)
m1 = {1111:"e1_norm_chn", 2222:"a_e1_norm_2222_chn", 3333:"a_e1_norm_3333_chn"}
chk("E1 at matched epoch", mean([f(one(name_exp=n,epoch=150,n_samples=3),'s_iou')-f(base[s],'s_iou')
                                 for s,n in m1.items()]), -0.0376)

print("\n-- section 6.5, quantization oracle --")
# oracle_chn.csv is one row per font; every figure quoted is the column mean over 34 fonts.
path = os.path.join(ROOT, "oracle_chn.csv")
if os.path.exists(path):
    d = list(csv.DictReader(open(path)))
    col = lambda c: mean([float(r[c]) for r in d])
    chk("oracle floor, unlimited precision", col("l1_inf"), 0.1422)
    chk("oracle at the released 128 grid",   col("l1_128"), 0.1443)
    chk("what quantization costs",           col("l1_128") - col("l1_inf"), 0.0021)
    chk("ceiling on E13, 128 -> 256 bins",   col("l1_128") - col("l1_256"), 0.0016)

print(f"\n{'='*64}\n{len(R)} rows in RESULTS.csv.  {len(FAIL)} failed claim(s).")
for x in FAIL: print("   -", x)
sys.exit(1 if FAIL else 0)
