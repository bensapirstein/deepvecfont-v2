#!/usr/bin/env python3
"""Self-test for render_val.py. No torch, no cairosvg, no GPU, no dataset.

Verifies the parts that do not need a GPU: the ground-truth mask convention, the L1 and
s-IoU arithmetic, renderability accounting, best-of-N selection, and train/eval mode
restoration. Everything here is hand-computable, so a wrong answer is unambiguous.
"""
import io, os, sys, types
import numpy as np
from PIL import Image

REPO = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---- stub torch -----------------------------------------------------------------
torch = types.ModuleType('torch')
class _NoGrad:
    def __call__(self, fn):
        def wrapper(*a, **k):
            return fn(*a, **k)
        return wrapper
    def __enter__(self): return self
    def __exit__(self, *a): return False
torch.no_grad = _NoGrad
sys.modules['torch'] = torch

# ---- stub cairosvg: the svg string is "ink=<0/1 grid>" ---------------------------
cairosvg = types.ModuleType('cairosvg')
def svg2png(bytestring=None, output_width=None, output_height=None, **kw):
    s = bytestring.decode()
    if s == 'BOOM':
        raise RuntimeError('cairosvg failed on this outline')
    grid = np.array([int(c) for c in s], dtype=np.uint8).reshape(output_height, output_width)
    rgba = np.zeros((output_height, output_width, 4), dtype=np.uint8)
    rgba[..., 3] = grid * 255            # alpha 255 = ink
    buf = io.BytesIO()
    Image.fromarray(rgba, 'RGBA').save(buf, format='PNG')
    return buf.getvalue()
cairosvg.svg2png = svg2png
sys.modules['cairosvg'] = cairosvg

# ---- stub data_utils.svg_utils.render --------------------------------------------
du = types.ModuleType('data_utils'); du.__path__ = []
svg_utils = types.ModuleType('data_utils.svg_utils')
def render(arr):
    flat = np.asarray(arr).ravel()
    if flat[0] < 0:                       # sentinel for "this sequence is malformed"
        raise ValueError('render() failed')
    return ''.join(str(int(v)) for v in flat)
svg_utils.render = render
sys.modules['data_utils'] = du
sys.modules['data_utils.svg_utils'] = svg_utils

sys.path.insert(0, REPO)
import render_val

# ---- tiny tensor stand-in --------------------------------------------------------
class T:
    def __init__(self, a): self.a = np.asarray(a)
    def __gt__(self, v): return T(self.a > v)
    def __getitem__(self, i): return T(self.a[i])
    def __iter__(self): return (T(x) for x in self.a)
    def cpu(self): return self
    def cuda(self): return self
    def numpy(self): return self.a
    def detach(self): return self
    def clone(self): return T(self.a.copy())
    def reshape(self, *s): return T(self.a.reshape(*s))

class Opts:
    char_num, img_size = 4, 4

# 4x4 glyphs. ink pattern per char, as the GT.
GT = [
    np.array([[1,1,0,0],[1,1,0,0],[0,0,0,0],[0,0,0,0]]),   # 4 ink px
    np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),   # 4 ink px
    np.zeros((4,4), int),                                   # empty
    np.ones((4,4), int),                                    # full
]

def font_batch(gt_list):
    # loader convention: 1 - raw/255, so ink sits at 1.0
    rendered = np.stack([g.astype(float) for g in gt_list])[None, ...]
    return {'rendered': T(rendered)}

class StubModel:
    """Returns a fixed set of decoded sequences, plus an image-decoder output."""
    def __init__(self, per_sample_preds, img_out):
        self.per_sample_preds, self.img_out = per_sample_preds, img_out
        self.call, self.training = 0, True
    def eval(self): self.training = False
    def train(self): self.training = True
    def __call__(self, data, mode=None):
        assert mode == 'test', mode
        preds = self.per_sample_preds[self.call % len(self.per_sample_preds)]
        self.call += 1
        return ({'svg': {'sampled_2': T(np.array(preds))},
                 'img': {'out': T(np.asarray(self.img_out, dtype=float))}}, {})

def l1(a, b): return float(np.mean(np.abs(a.astype(float) - b.astype(float))))
def iou(a, b):
    u = np.sum(a + b)
    return 1.0 if u == 0 else float(np.sum(a * b) / u)

fail = 0
def check(name, got, want, tol=1e-9):
    global fail
    ok = abs(got - want) < tol if isinstance(want, float) else got == want
    print(f"  [{'ok  ' if ok else 'FAIL'}] {name}: got {got!r}, want {want!r}")
    if not ok: fail += 1

print("1. perfect reconstruction -> L1 0, s-IoU 1 (empty glyph counted as IoU 1)")
perfect = [g.ravel() for g in GT]
m = render_val.rendered_val_metrics(
    StubModel([perfect], np.zeros((4,4,4))), [font_batch(GT)], Opts(), n_samples=1)
check("l1", m['l1'], 0.0)
check("siou", m['siou'], 1.0)
check("renderability", m['renderability'], 1.0)
check("n_glyphs (1 font x char_num 4)", m["n_glyphs"], 4)

print("\n2. known imperfect prediction -> hand-computed L1 and s-IoU")
pred = [
    np.array([[1,1,0,0],[1,0,0,0],[0,0,0,0],[0,0,0,0]]),   # 1 px short
    np.array([[1,1,1,1],[1,0,0,0],[0,0,0,0],[0,0,0,0]]),   # 1 px extra
    np.zeros((4,4), int),
    np.ones((4,4), int),
]
want_l1 = np.mean([l1(p, g) for p, g in zip(pred, GT)])
want_iou = np.mean([iou(p.astype(bool), g.astype(bool)) for p, g in zip(pred, GT)])
m = render_val.rendered_val_metrics(
    StubModel([[p.ravel() for p in pred]], np.zeros((4,4,4))), [font_batch(GT)], Opts(), n_samples=1)
check("l1", m['l1'], float(want_l1))
check("siou", m['siou'], float(want_iou))
print(f"        (hand check: L1 = mean(1/16, 1/16, 0, 0) = {want_l1:.6f})")

print("\n3. a glyph that fails to render is excluded, not scored as zero")
broken = [p.ravel().astype(float) for p in pred]
broken[1] = np.array([-1.0] * 16)          # render() sentinel
m = render_val.rendered_val_metrics(
    StubModel([broken], np.zeros((4,4,4))), [font_batch(GT)], Opts(), n_samples=1)
check("renderability", m['renderability'], 0.75)
check("l1 over the 3 that rendered", m['l1'],
      float(np.mean([l1(pred[i], GT[i]) for i in (0, 2, 3)])))

print("\n4. best-of-N picks the candidate closest to the IMAGE DECODER, not the GT")
# sample A matches the GT exactly; sample B matches the image-decoder output exactly.
# test_few_shot.py selects on the decoder output, so B must win -- and the resulting
# L1 must therefore be WORSE than sample A's, which is the whole point.
dec_out = [
    np.array([[1,0,0,0],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),
    np.array([[1,1,1,1],[0,0,0,0],[0,0,0,0],[0,0,0,0]]),
    np.zeros((4,4), int),
    np.ones((4,4), int),
]
sample_A = [g.ravel() for g in GT]
sample_B = [d.ravel() for d in dec_out]
m = render_val.rendered_val_metrics(
    StubModel([sample_A, sample_B], np.stack([d.astype(float) for d in dec_out])),
    [font_batch(GT)], Opts(), n_samples=2)
want = np.mean([l1(dec_out[i], GT[i]) for i in range(4)])
check("l1 reflects the decoder-selected candidate", m['l1'], float(want))
check("  (and is worse than the GT-matching sample's 0.0)", m['l1'] > 0.0, True)

print("\n5. max_fonts caps the pass; training mode is restored")
model = StubModel([perfect], np.zeros((4,4,4)))
m = render_val.rendered_val_metrics(model, [font_batch(GT)] * 5, Opts(),
                                    n_samples=1, max_fonts=2)
check("n_glyphs with max_fonts=2", m['n_glyphs'], 8)
check("model returned to training mode", model.training, True)

model2 = StubModel([perfect], np.zeros((4,4,4)))
model2.eval()
render_val.rendered_val_metrics(model2, [font_batch(GT)], Opts(), n_samples=1)
check("a model already in eval stays in eval", model2.training, False)

print("\n6. best_checkpoint_file returns the best SURVIVING checkpoint")
import csv, tempfile, checkpoint_log
tmp = tempfile.mkdtemp()
logs, ckpts = os.path.join(tmp, 'logs'), os.path.join(tmp, 'checkpoints')
os.makedirs(logs); os.makedirs(ckpts)
with open(os.path.join(logs, 'checkpoint_metrics.csv'), 'w', newline='') as fh:
    w = csv.DictWriter(fh, fieldnames=checkpoint_log.MANIFEST_FIELDS); w.writeheader()
    for ep, rl1 in ((25, 0.19), (50, 0.21), (75, 0.23)):
        w.writerow({'epoch': ep, 'step': ep, 'checkpoint': f'{ep}_x.ckpt',
                    'val_metric': '3.0', 'val_l1': '0', 'val_vggpt': '0',
                    'val_svg_total': '0', 'val_svg_para_total': '0',
                    'val_svg_aux': '0', 'val_svg_para_aux': '0',
                    'val_render_l1': f'{rl1}', 'val_render_siou': '0.7',
                    'val_render_renderability': '1.0'})
# epoch 25 scores best but was pruned; 50 and 75 survive
for ep in (50, 75):
    open(os.path.join(ckpts, f'{ep}_x.ckpt'), 'w').close()
got = checkpoint_log.best_checkpoint_file(tmp, 'val_render_l1')
check("falls through to the best on-disk checkpoint", got, '50_x.ckpt')

print(f"\n{'ALL PASS' if not fail else str(fail) + ' FAILURES'}")
sys.exit(1 if fail else 0)
