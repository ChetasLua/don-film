"""ドン: brushed sound lettering, drawn with the sumi engine from our own strokes.

    from op import don
    lay = don.don(f, 'slam', x=960, y=520, size=.8)          # premult RGBA (1080,1920,4); f = frames since the hit (f<0 or past life: empty)
    img = don.put(img, f, 'slam', ...)                        # same, composited, with a one-frame paper flash and shake already in the mark
kinds: 'slam'  ドン!   black brush, paper keyline, halftone in `accent` (b5, c15)      'full'  ドン!! filling the frame (a3)
       'small' ドン!   no halftone, thin key (asides, c-cards)
slam timing (frames): 0 huge + smear, 1 1.16, 2 squash .93, 3 1.04, 4 .99, 5.. 1.0 with a dying tremble; the last `out` frames flake away as dry ink.
write(pg, ...) puts the strokes on a sumi Page with your own times (g4).   alpha(kind) = the finished ink mask, 1080x1920.
"""
import numpy as np, cv2, math, os
from . import W, H
from .paint import rgb, f32, smooth, grid, fbm
from .sumi import Page, page_save, page_load, splash, tooth
from .letters import S_, tate
from . import letters as LT
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_b = dict(dry=.42, deplete=.85, load=.72, split=.78)
DO = [
    tate(384, 66, 404, 936, 176, head=1.14, waist=.78, end='tome', tail=.96, tip=1.0, **{**_b, 'seed': 11}),
    S_([(448, 350, 24, .8), (488, 384, 158, .62), (612, 452, 120, 1.25), (758, 556, 152, .6), (816, 626, 128, .42), (776, 672, 36, .6)], **{**_b, 'seed': 12}),
    S_([(592, 34, 20, .8), (622, 72, 126, .55), (668, 168, 110, 1.15), (706, 270, 24, 1.4)], **{**_b, 'dry': .34, 'seed': 13}),
    S_([(782, -14, 20, .8), (812, 22, 126, .55), (860, 120, 110, 1.15), (900, 224, 22, 1.4)], **{**_b, 'dry': .34, 'seed': 14})]
N_ = [
    S_([(88, 140, 24, .8), (126, 178, 172, .55), (250, 262, 130, 1.25), (376, 366, 160, .5), (398, 424, 44, .6)], **{**_b, 'seed': 15}),
    S_([(58, 818, 24, .7), (88, 900, 188, .42), (152, 920, 180, .6), (332, 832, 148, 1.1), (560, 642, 124, 1.45), (760, 432, 98, 1.7), (912, 212, 50, 1.9), (968, 112, 6, 2.1)],
       dry=.44, deplete=1.15, load=.60, split=.92, fork=1., speed=1.15, seed=16)]
STEM = S_([(452, 4, 40, .8), (500, 62, 216, .5), (500, 220, 186, 1.15), (488, 420, 122, 1.5), (474, 590, 60, 1.5), (466, 690, 6, 1.8)], dry=.40, deplete=1.0, load=.66, split=.85, fork=.8, speed=1.1, seed=17)
DOT = S_([(416, 818, 60, .6), (452, 858, 186, .4), (500, 884, 172, .4), (478, 916, 64, .5)], dry=.22, deplete=.4, split=.5, seed=18)
EX = [STEM, DOT]
LAY = dict(slam=[(DO, (-150, 110, 830)), (N_, (636, 176, 780)), (EX, (1236, 70, 860))],
           full=[(DO, (-268, 4, 1076)), (N_, (640, 140, 890)), (EX, (1092, -10, 1040), .78), (EX, (1306, 16, 990), .78)])
LAY['small'] = LAY['slam']
_pg, _al = {}, {}


def write(pg, glyph, box, times=None, t0=0., seed=0, wt=1.0, rot=0., **over):
    """glyph: DO | N_ | EX | any stroke list.  box=(x,y,size).  -> Marks"""
    return LT.write(pg, None, box, t0, times, wt, seed, strokes=glyph, rot=rot, **over)


def page(kind='slam'):
    k = 'slam' if kind == 'small' else kind
    if k in _pg: return _pg[k]
    c = os.path.join(ROOT, 'cache', f'don_{k}_4.npz')
    if os.path.exists(c): _pg[k] = page_load(c); return _pg[k]
    pg = Page(); t = 0.
    for g, box, *wt in LAY[k]:
        rot = .06 if (k == 'full' and box[0] == 1306) else 0.
        for i, q in enumerate(g): write(pg, [q], box, [(t, t + 1)], seed=len(pg.marks), wt=wt[0] if wt else 1., rot=rot); t += 1
    for (x, y, R, a, n) in ((300, 930, 190, 1.9, 90), (1330, 250, 220, -.9, 110), (640, 900, 150, 2.6, 60), (1600, 820, 170, .9, 80)):
        pg.add(splash(x, y, R, seed=int(x), ang=a, spread=.9, n=n, big=15, fingers=6, t0=t, t1=t + 1, mist=.6))
    os.makedirs(os.path.dirname(c), exist_ok=True); page_save(pg, c); _pg[k] = pg
    return pg


def alpha(kind='slam', t=1e9):
    k = 'slam' if kind == 'small' else kind
    if t > 1e8:
        if k not in _al: _al[k] = page(k).ink()[0]
        return _al[k]
    return page(k).ink(t)[0]


def halftone(dark, pitch=15., ang=.62):
    """round manga dots whose size follows `dark` 0..1 -> coverage"""
    h, w = dark.shape; x, y = grid(h, w); c, s = math.cos(ang), math.sin(ang)
    u = (x * c + y * s) / pitch; v = (-x * s + y * c) / pitch; d = np.sqrt((u - np.round(u)) ** 2 + (v - np.round(v)) ** 2)
    return np.clip((np.sqrt(np.clip(dark, 0, 1) * .62) - d) * pitch + .5, 0, 1) * (dark > .015)


_S = (1.62, 1.16, .93, 1.045, .985, 1.008)
_fl = {}


def _flake():
    if 'n' not in _fl: n = fbm(H, W, (90, 30, 9, 3), (1, .8, .6, .5), 91); _fl['n'] = (n - n.min()) / np.ptp(n)
    return _fl['n']


def don(f, kind='slam', x=None, y=None, size=1.0, rot=None, ink='#0d0b0e', paper='#f7f4ec', accent='#c8202c', key=13., life=28, out=7, seed=0, tone=True, deep='#3a0a12', pitch=15., trem=1.0):
    """-> premultiplied RGBA layer.  x,y = centre, size 1 = 1500 px wide, f = frames since the hit"""
    if f < 0 or f >= life: return np.zeros((H, W, 4), f32)
    x = W / 2 if x is None else x; y = H / 2 if y is None else y; rot = (-.06 if kind != 'full' else 0.) if rot is None else rot
    a = alpha(kind); fi = int(f); s = size * (_S[fi] if fi < len(_S) else 1. + .0015 * (f - 5))
    r = np.random.default_rng(seed * 771 + fi); j = r.normal(0, 1, 2) * 9 * math.exp(-f / 3.4) * size * trem; rr = rot + r.normal(0, .008) * math.exp(-f / 4.) * trem
    k = np.clip((f - (life - out)) / max(1, out), 0, 1)
    sx, sy = (s * 1.05, s * .95) if fi == 2 else (s * .98, s * 1.02) if fi == 3 else (s, s)
    def M(q=1., dx=0., dy=0.):
        c, sn = math.cos(rr), math.sin(rr)
        return np.float32([[c * sx * q, -sn * sy * q, x + j[0] + dx - (c * sx * q * W / 2 - sn * sy * q * H / 2)], [sn * sx * q, c * sy * q, y + j[1] + dy - (sn * sx * q * W / 2 + c * sy * q * H / 2)]])
    fl = cv2.INTER_AREA if s < .55 else cv2.INTER_LINEAR
    src = cv2.GaussianBlur(a, (0, 0), .45 / max(s, .2)) if s < .8 else a
    T = lambda m: cv2.warpAffine(src, m, (W, H), flags=cv2.INTER_LINEAR)
    m = sum(T(M(q)) for q in np.linspace(.86, 1.06, 9)) / 9. if fi == 0 else T(M())
    if k > 0:
        n = _flake(); keep = smooth(k * 1.15 - .12, k * 1.15 - .02, n); m = m * keep
    q = cv2.resize(m, (W // 2, H // 2), interpolation=cv2.INTER_AREA); kk = key * size / 2
    kl = cv2.resize(smooth(.05, .16, cv2.GaussianBlur(q, (0, 0), max(.8, kk * .55))), (W, H), interpolation=cv2.INTER_LINEAR)
    t = tooth(H, W); kl = smooth(.42, .60, kl + (t - .5) * .28) * (1 - smooth(.15, .9, k))
    out_ = np.zeros((H, W, 4), f32)
    if tone and kind != 'small':
        hm = cv2.resize(cv2.GaussianBlur(smooth(.02, .5, cv2.GaussianBlur(q, (0, 0), 9 * size)), (0, 0), 15 * size), (W, H)); hm = cv2.warpAffine(hm, np.float32([[1, 0, 26 * size], [0, 1, 34 * size]]), (W, H))
        n0 = min(1., (f + .4) / 3.); hd = halftone(np.clip(hm * .95 * n0, 0, .95), pitch * max(.7, size)) * (1 - smooth(.1, .8, k)); out_ = np.dstack([rgb(accent) * hd[..., None], hd])
    c = kl[..., None]; out_ = out_ * (1 - c) + np.concatenate([rgb(paper) * c, c], -1)
    mi = smooth(.06, .80, m)[..., None]; yy = grid()[1]; g = np.clip((yy - (y - 330 * size)) / (760 * size), 0, 1)[..., None] ** 2
    col = rgb(ink) * (1 - g * .8) + rgb(deep) * g * .8
    out_ = out_ * (1 - mi) + np.concatenate([col * mi, mi], -1)
    return out_.astype(f32)


def put(img, f, kind='slam', flash=True, **k):
    """composite on a frame; on the hit frame the picture behind jumps to high-contrast paper tones (one frame)"""
    if f < 0: return img
    if flash and int(f) == 0:
        L = (img @ np.array([.3, .59, .11], f32))[..., None]; img = img * .35 + (rgb('#f7f4ec') * smooth(.30, .62, L)) * .65
    l = don(f, kind, **k); return img * (1 - l[..., 3:4]) + l[..., :3]
