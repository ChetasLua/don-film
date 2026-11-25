"""OLD ERA worlds: the iron sky (back-lit cumulonimbus), the execution tower, crowd planes, gulls, spears."""
import os
from ._old import *
from op import clouds as CL
from op.kit import zoomblur, contours
from op.comp import screen, radial, bokeh
from op.paint import ramp, mottle, repaint
CACHE = os.path.join(os.path.dirname(__file__), '..', 'cache')
PW, PH = 2240, 1500                                      # sky plate
SUN = (1235., 430.)                                      # sun in plate px (behind the great cloud)


def cached(name, fn):
    p = os.path.join(CACHE, name + '.npy')
    if os.path.exists(p): return np.load(p, mmap_mode='r')
    a = fn(); os.makedirs(CACHE, exist_ok=True); t = p + '.%d.tmp.npy' % os.getpid(); np.save(t, np.asarray(a, np.float16)); os.replace(t, p); return np.load(p, mmap_mode='r')


def _sky():
    h, w = PH, PW; x, y = grid(h, w); r = np.random.default_rng(4)
    base = ramp(y / h, [(0, '#1b2a3f'), (.22, '#2f4760'), (.48, '#6f8798'), (.66, '#b3b3a6'), (.80, '#d8c79e'), (1, '#c9b48c')])
    d = np.sqrt((x - SUN[0]) ** 2 + ((y - SUN[1]) * 1.1) ** 2); g = np.exp(-(d / 420) ** 1.3)[..., None]
    base = base * (1 - g * .9) + rgb('#fff0c4') * g * .9
    S = CL.Sky(h, w, base, light=(-.30, -.80, -.30), seed=3)
    S.strata(40, 420, 50, ('#4c6278', '#70838f', '#33485f'), seed=5, w=(.10, .40), th=(1.5, 5.), op=(.25, .6))
    S.bank(1235, 960, 1900, 640, ('#fbebc2', '#dccb9f', '#a9a99c', '#8593a0'), '#d6c9a2', seed=11, profile=CL.peak(.30, .25, .45, .75, .6), detail=.9, tiers=2, n=7, rim='#fffae0', rim_amt=.9, lit=.26, half=.34, deep=.10)
    S.bank(1790, 1290, 1850, 1480, ('#e9cf98', '#7d8b97', '#3e5169', '#22334a'), '#75838f', seed=63, profile=CL.peak(.40, .20, .46, .80, .74, .18), detail=1.0, tiers=3, n=5, rim='#fff6d4', rim_amt=1., lit=.07, half=.20, deep=.30, light=(-.80, -.30, -.24), contrast=1.25)
    k1 = S.k
    S.bank(300, 1280, 1640, 1120, ('#e6cb94', '#7c8994', '#41536a', '#25364d'), '#7c878d', seed=31, profile=CL.peak(.36, .22, .34, .82, .5), detail=1.0, tiers=3, n=5, rim='#fff3cc', rim_amt=1., lit=.07, half=.20, deep=.28, light=(.80, -.36, -.24), contrast=1.2)
    k2 = S.k
    S.bank(1230, 1300, 1900, 260, ('#f2d49a', '#8b9297', '#55667a', '#3b4c62'), '#a3a498', seed=41, detail=.9, tiers=2, n=8, rim='#ffeec0', rim_amt=.9, lit=.12, half=.3, deep=.2, light=(-.1, -.85, -.3))
    k3 = S.k; lab = S.lab.copy()
    img = S.paint(seed=2, sizes=(16, 8, 4), amt=.85)
    n = fbm(h, w, (60, 22), (1, .6), 77)
    for k, reach in ((k1, 900.), (k2, 800.), (k3, 700.)):                                 # hot gold edge toward the sun, wide near it
        m = (lab >= k).astype(f32) if k == k1 else (lab == k).astype(f32); m = cv2.GaussianBlur(m, (0, 0), 1.2); gx, gy = (SUN[0] - x) / (d + 1), (SUN[1] - y) / (d + 1)
        for wd, col, am in ((30., '#ffc978', .55), (13., '#ffe6a8', .8), (5.5, '#fffbe6', 1.)):
            q = cv2.remap(m, x + gx * wd, y + gy * wd, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); rr = cv2.GaussianBlur(np.clip(m - q, 0, 1), (0, 0), wd * .22 + .5) * (m > .3) * np.exp(-(d / reach) ** 1.6) * np.clip(.75 + .5 * n, 0, 1.3); img = img + (rgb(col) - img) * np.clip(rr * am, 0, 1)[..., None]
    L = lum_(img); hot = np.clip((L - .80) / .2, 0, 1)[..., None] * img
    q = cv2.resize(hot, (w // 4, h // 4), interpolation=cv2.INTER_AREA); ray = zoomblur(q, SUN[0] / 4, SUN[1] / 4, 1.6, 40); ray = cv2.resize(ray, (w, h), interpolation=cv2.INTER_LINEAR)
    ang = np.arctan2(y - SUN[1], x - SUN[0]); n = np.interp((ang + math.pi) / (2 * math.pi) * 720, np.arange(721), cv2.GaussianBlur(r.standard_normal((1, 721)).astype(f32), (0, 0), 3.0)[0] * 2.2)
    beam = np.clip(.55 + .6 * n, 0, 1.4) * np.exp(-(d / 820) ** 1.5) * smooth(-120, 200, y - SUN[1] + 60)
    img = screen(img, ray * rgb('#ffe2a8') * .42 + beam[..., None] * rgb('#ffdf9c') * .30)
    img = screen(img, (np.exp(-(d / 120) ** 2) * .7)[..., None] * rgb('#fff6dc'))
    return np.clip(img, 0, 1.3)


def lum_(i): return i[..., 0] * .299 + i[..., 1] * .587 + i[..., 2] * .114


def sky(): return cached('old_sky_iron', _sky)


# ------------------------------------------------------------------ spears
def spear(cel, x, y, ang, L, k, col='#1b1b24', steel='#8f98a2', steel_hi='#f6f2dd', wood='#3a2a26', w=9., prongs=False):
    """long spear: shaft from (x,y) along ang (radians, 0 = right), blade at the end"""
    d = A(math.cos(ang), math.sin(ang)); n = A(-d[1], d[0]); p0 = A(x, y); p1 = p0 + d * L * .74; tip = p0 + d * L
    cel.fill(np.vstack([p0 + n * w * k, p1 + n * w * .85 * k, p1 - n * w * .85 * k, p0 - n * w * k]), wood); cel.fill(np.vstack([p0 - n * w * k, p1 - n * w * .85 * k, p1 - n * w * .3 * k, p0 - n * w * .35 * k]), col)
    bl = L * .26; b = [p1 - d * bl * .06, p1 + n * bl * .07 - d * bl * .02, p1 + n * bl * .11 + d * bl * .22, p1 + n * bl * .075 + d * bl * .6, tip, p1 - n * bl * .075 + d * bl * .6, p1 - n * bl * .11 + d * bl * .22, p1 - n * bl * .07 - d * bl * .02]
    B = spl(b, True, (4,)); cel.fill(B, steel); cel.fill(spl([b[0], b[5] * 0 + p1 + d * bl * .3, tip, b[5], b[6], b[7]], True, (2,)), col, .55); cel.fill(np.array([p1 + d * bl * .02 + n * 1.2, tip, p1 + d * bl * .02 - n * 1.2]), steel_hi)
    cel.fill(np.vstack([p1 + n * w * 1.7 * k - d * bl * .10, p1 + n * w * 1.7 * k - d * bl * .02, p1 - n * w * 1.7 * k - d * bl * .02, p1 - n * w * 1.7 * k - d * bl * .10]), '#b08a3c')
    return tip


# ------------------------------------------------------------------ the tower (metric 3-D, op.cam.Camera3D)
TX, TY, TZ = 3.05, 12.0, 8.25


def tower(cam, f=0, dark='#1d1f2a', mid='#333846', lite='#4b4a4c', rimc='#ffe9b0'):
    """execution tower as a layer, plus the projected deck points for the figures"""
    C = Cel(); r = np.random.default_rng(3); B = []
    def P(x, y, z): return A(TX + x, TY + y, z)
    b0, b1 = 2.15, 1.55; lev = [0.0, 2.7, 5.4, TZ - .25]
    hw = lambda z: b0 + (b1 - b0) * z / TZ
    for sx in (-1, 1):
        for sy in (-1, 1): B.append((P(sx * hw(0), sy * hw(0) * .8, 0), P(sx * hw(TZ), sy * hw(TZ) * .8, TZ), .24))
    for i, z in enumerate(lev[1:]):
        h = hw(z)
        for sy in (-1, 1): B.append((P(-h - .22, sy * h * .8, z), P(h + .22, sy * h * .8, z), .20))
        for sx in (-1, 1): B.append((P(sx * h, -h * .8 - .15, z), P(sx * h, h * .8 + .15, z), .17))
    for a, b in zip(lev[:-1], lev[1:]):
        ha, hb = hw(a), hw(b)
        for sy in (-1, 1): B += [(P(-ha, sy * ha * .8, a), P(hb, sy * hb * .8, b), .13), (P(ha, sy * ha * .8, a), P(-hb, sy * hb * .8, b), .13)]
        for sx in (-1, 1): B.append((P(sx * ha, -ha * .8, a), P(sx * hb, hb * .8, b), .12))
    for i in range(16): B.append((P(-.35, -b1 * .8 - .32 + (i / 16) * 0, .45 * i + .3), P(.35, -b1 * .8 - .32, .45 * i + .3), .06))      # ladder
    for sx in (-.36, .36): B.append((P(sx, -b1 * .8 - .34, 0), P(sx, -b1 * .8 - .30, TZ - .1), .08))
    def depth(b): return -cam.project((b[0] + b[1])[None] / 2)[1][0]
    for q, (a, b, th) in enumerate(sorted(B, key=depth)):
        pr, z = cam.project(np.stack([a, b]));
        if z.min() < .2: continue
        w0, w1 = th * cam.f / z[0], th * cam.f / z[1]; d = (pr[1] - pr[0]); L = np.hypot(*d) + 1e-6; n = A(-d[1], d[0]) / L; n = n if n[0] < 0 else -n
        p0, p1 = pr; far = np.clip((z.mean() - 11) / 5, 0, 1); col = mix3(dark, mid, far * .8)
        C.fill(np.array([p0 + n * w0 / 2, p1 + n * w1 / 2, p1 - n * w1 / 2, p0 - n * w0 / 2]), col); C.fill(np.array([p0 + n * w0 / 2, p1 + n * w1 / 2, p1 + n * w1 * .12, p0 + n * w0 * .12]), mix3(lite, mid, far))
        for j in range(2 if L > 60 else 0):
            u = r.uniform(-.32, .3); e0, e1 = r.uniform(0, .4), r.uniform(.55, 1.); C.line(spl([p0 + d * e0 + n * w0 * u, p0 + d * (e0 + e1) / 2 + n * w0 * (u + .04), p0 + d * e1 + n * w1 * u], n=6), 1.2, dark, taper=(.4, .5), seed=q * 3 + j)
        C.line(np.stack([p0 + n * w0 / 2, p1 + n * w1 / 2]), 1.9, rimc, taper=(.3, .3), press=.5, seed=q, alpha=.55 * (1 - far * .6)); C.line(np.stack([p0 - n * w0 / 2, p1 - n * w1 / 2]), 1.4, '#0e0f16', seed=q + 9)
        for e, pt, wd in ((0, p0, w0), (1, p1, w1)):
            if wd > 9: a_ = np.linspace(0, 6.28, 9)[:-1]; C.fill(np.c_[pt[0] + 1.8 * np.cos(a_), pt[1] + 1.8 * np.sin(a_)], '#0d0e14')
    h = b1 + .45; dk = [P(-h, -h * .85, TZ), P(h, -h * .85, TZ), P(h, h * .85, TZ), P(-h, h * .85, TZ)]; lo = [v - A(0, 0, .26) for v in dk]
    pu = cam.project(np.stack(lo))[0]; pt = cam.project(np.stack(dk))[0]
    C.fill(pu, '#14151d')
    for i in range(9):
        u = (i + .5) / 9; a, b = lo[0] * (1 - u) + lo[1] * u, lo[3] * (1 - u) + lo[2] * u; C.line(cam.project(np.stack([a, b]))[0], 2.0, '#2c303c', taper=(.1, .3), seed=70 + i)
    for (i, j) in ((3, 0), (0, 1), (1, 2)):
        C.fill(np.array([pu[i], pu[j], pt[j], pt[i]]), '#2a2b34' if j else '#39383c')
    C.fill(np.array([pu[0], pu[1], pt[1], pt[0]]), '#2b2c35')
    for i in range(12):
        u = i / 12; a, b = lo[0] * (1 - u) + lo[1] * u, dk[0] * (1 - u) + dk[1] * u; C.line(cam.project(np.stack([a, b]))[0], 1.6, '#0f1016', seed=90 + i)
    C.line(np.stack([pt[0], pt[1]]), 2.6, rimc, taper=(.02, .02), seed=99, alpha=.9); C.line(np.stack([pu[0], pu[1]]), 2.0, '#0c0d13', seed=98)
    return C.layer(), P


def mix3(a, b, t): return rgb(a) * (1 - t) + rgb(b) * t


def condemned(cel, x, y, u, f=0, col=('#381d24', '#55272d', '#0f0e14')):
    """kneeling man on the deck, coat in the wind, wrists in wooden stocks.  u = px per metre, (x,y) = knees"""
    t = f // 2 * 2; w = lambda i, a: a * math.sin(t * .55 - i * .95)
    p = lambda L: np.array([(x + a * u, y - b * u) for a, b in L]); red, red2, blk = col; F = lambda L, c, cr=(): cel.fill(spl(p(L), True, cr), c)
    F([(-.40, 1.14), (.40, 1.16), (.70, 1.02 + w(0, .05)), (1.05, .98 + w(1, .09)), (1.42, .86 + w(2, .13)), (1.95, .92 + w(3, .20)), (1.66, .58 + w(3.3, .16)), (1.22, .42 + w(2.4, .10)), (.82, .20 + w(1.5, .06)), (.56, .0), (-.62, .0), (-.70, .30), (-.62, .80)], red, (0, 5, 9, 10))
    F([(.40, 1.10), (.8, .94 + w(.6, .06)), (1.2, .74 + w(2, .10)), (1.70, .72 + w(3, .17)), (1.24, .56 + w(2.4, .10)), (.8, .34), (.50, .1), (.44, .6)], red2, (0, 3)); F([(-.40, 1.10), (-.60, .8), (-.64, .30), (-.50, .05), (-.40, .05), (-.44, .6)], red2)
    F([(.46, .98), (.95, .70 + w(1.6, .08)), (1.36, .62 + w(2.8, .12)), (.95, .62 + w(1.6, .08))], '#2a1218', (0, 2))
    F([(-.56, 0), (-.58, .20), (-.34, .40), (.34, .40), (.58, .20), (.56, 0)], '#14151d', (0, 5)); F([(-.02, .38), (.0, .05), (.04, .38)], '#05060a', (1,)); F([(-.27, 1.08), (.27, 1.08), (.24, .36), (-.24, .36)], '#1b1e28', (0, 1, 2, 3)); cel.fill(p([(-.12, 1.10), (.12, 1.10), (.02, .72), (-.02, .72)]), '#6b4a3c')
    for e in (-1, 1):
        F([(e * .42, 1.12), (e * .24, 1.16), (e * .16, .9), (e * .20, .58), (e * .40, .50), (e * .46, .80)], red if e < 0 else red2); F([(e * .20, .60), (e * .40, .56), (e * .30, .40), (e * .16, .36), (e * .12, .46)], '#23262f')
        F([(e * .34, 1.19), (e * .60, 1.12), (e * .64, 1.0), (e * .42, 1.06)], '#8a6a34', (1,))
    cel.fill(p([(-.46, .28), (.46, .30), (.46, .46), (-.46, .44)]), '#57432f'); cel.fill(p([(-.46, .28), (.46, .30), (.46, .34), (-.46, .32)]), '#2c211c')
    for sx in (-.17, .17): F([(sx - .07, .32), (sx - .08, .44), (sx, .50), (sx + .08, .44), (sx + .07, .32), (sx, .27)], '#7d5744'); cel.line(p([(sx - .05, .30), (sx, .42)]), 1.2, '#1a1210')
    F([(-.16, 1.34), (-.16, 1.20), (-.10, 1.08), (0, 1.04), (.10, 1.08), (.16, 1.20), (.16, 1.34)], '#80563f'); F([(.03, 1.04), (.13, 1.1), (.16, 1.34), (.07, 1.34), (.05, 1.2)], '#4c2f2c')
    F([(-.22, 1.10), (-.27, 1.32), (-.20, 1.52), (0, 1.60), (.18, 1.55), (.34, 1.50 + w(1, .03)), (.58, 1.46 + w(2, .06)), (.36, 1.36), (.54, 1.26 + w(2, .05)), (.28, 1.22), (.40, 1.10 + w(2.4, .04)), (.19, 1.14), (.17, 1.32), (.06, 1.27), (0, 1.36), (-.08, 1.27), (-.15, 1.34), (-.16, 1.16)], blk, (6, 8, 10, 13, 15))
    F([(0, 1.17), (.11, 1.185), (.22, 1.19), (.32, 1.34), (.24, 1.13), (.09, 1.09), (0, 1.115), (-.09, 1.09), (-.24, 1.13), (-.32, 1.34), (-.22, 1.19), (-.11, 1.185)], blk, (3, 9)); cel.fill(p([(-.07, 1.085), (.07, 1.085), (.04, 1.06), (-.04, 1.06)]), '#e8dcc2')


def guard(cel, x, y, u, f=0, sg=1, lean=.04, c=('#1c212d', '#2a3140', '#12151d')):
    """armoured guard, long coat, spear planted at arm's length"""
    p = lambda L: np.array([(x + a * u * sg, y - b * u) for a, b in L]); t = f // 2 * 2; w = .05 * math.sin(t * .5 + sg * 1.3); F = lambda L, col, cr=(): cel.fill(spl(p(L), True, cr), col); dk, md, bk = c
    s0, s1 = p([(.56, -.02)])[0], p([(.56 + lean, 3.75)])[0]; spear(cel, s0[0], s0[1], math.atan2(s1[1] - s0[1], s1[0] - s0[0]), np.hypot(*(s1 - s0)), u / 130., w=5.)
    cel.fill(spl(p([(.60, 2.72), (.86, 2.66 + w), (1.12, 2.72 + w * 2), (1.06, 2.56 + w), (.84, 2.50 + w), (.60, 2.52)]), True, (0, 2, 5)), '#3a4152')
    for e in (-1, 1): F([(e * .05, .0), (e * .08, .55), (e * .06, .9), (e * .26, .9), (e * .24, .5), (e * .22, .06), (e * .30, .0)], bk, (0, 5, 6))
    F([(-.26, 1.58), (-.34, 1.2), (-.34, .9), (-.52 - w, .34), (-.30, .28), (-.10, .40), (0, .86), (.10, .40), (.28, .28), (.46 + w, .34), (.32, .9), (.33, 1.2), (.26, 1.58)], dk, (3, 6, 9)); F([(0, 1.5), (.22, 1.5), (.28, 1.1), (.28, .8), (.40, .36), (.28, .32), (.06, .8)], bk)
    cel.fill(p([(-.33, 1.02), (.33, 1.02), (.33, .92), (-.33, .92)]), bk)
    for e in (-1, 1): F([(e * .16, 1.60), (e * .40, 1.58), (e * .48, 1.36), (e * .30, 1.32), (e * .18, 1.44)], md, (1, 2))
    F([(-.44, 1.40), (-.52, 1.1), (-.46, .80), (-.34, .78), (-.34, 1.1), (-.30, 1.38)], md); F([(.40, 1.46), (.50, 1.26), (.66, 1.40), (.64, 1.54), (.56, 1.54), (.50, 1.44), (.40, 1.58)], md)
    F([(-.13, 1.58), (-.15, 1.76), (-.10, 1.90), (.10, 1.90), (.15, 1.76), (.13, 1.58), (0, 1.52)], '#4a3a38'); F([(0, 1.52), (.13, 1.58), (.15, 1.76), (.12, 1.88), (.03, 1.86)], '#2d2730')
    F([(-.24, 1.76), (-.17, 1.80), (-.15, 1.98), (0, 2.06), (.15, 1.98), (.17, 1.80), (.24, 1.76), (.10, 1.74), (-.10, 1.74)], bk, (0, 6)); F([(-.015, 2.04), (0, 2.20), (.06, 2.28 + w), (.16, 2.24 + w), (.06, 2.14), (.02, 2.04)], md, (3,))


# ------------------------------------------------------------------ crowd
KINDS = ('bare', 'bald', 'bandana', 'tricorn', 'cap', 'wide', 'top', 'bun', 'tail', 'hood', 'curly', 'scarf', 'spike', 'turban', 'horn', 'bare', 'cap', 'curly')


def person(cel, x, y, s, kind, col, dk, r, arms=False, sg=1, acc=None, det=False):
    """back view bust silhouette with a few inner shapes.  (x,y) = top of the skull, s = head height px"""
    q = lambda L: np.array([(x + a * s * sg, y + b * s) for a, b in L]); F = lambda L, c=col, cr=(), cl=True: cel.fill(spl(q(L), True, cr), c); j = lambda k=.04: r.normal(0, k)
    sw = r.uniform(.88, 1.2)
    F([(-.15, .85), (-.17, 1.0), (-.45 * sw, 1.10), (-.88 * sw, 1.28), (-1.0 * sw, 1.7), (-1.16 * sw, 14.), (1.16 * sw, 14.), (1.0 * sw, 1.7), (.88 * sw, 1.28), (.45 * sw, 1.10), (.17, 1.0), (.15, .85)], col if acc is None else acc, (5, 6))
    F([(-.2, 1.06), (0, 1.16), (.2, 1.06), (.12, 2.0), (.0, 3.6), (-.1, 2.0)], dk); F([(.5 * sw, 1.2), (.86 * sw, 1.34), (1.0 * sw, 1.8), (1.12 * sw, 14.), (.6 * sw, 14.), (.72 * sw, 2.)], dk)
    if det:
        ln = lambda L, w_, c=dk: cel.line(spl(q(L), n=8), w_, c, taper=(.2, .6), press=.3, seed=int(r.integers(1 << 20))); v = r.integers(4); ac = rgb(col) if isinstance(col, str) else col
        if v == 0:
            for i in range(7): cel.line(spl(q([(-1.1 * sw, 1.75 + i * .42), (0, 1.62 + i * .42), (1.1 * sw, 1.75 + i * .42)]), n=8), .16 * s, dk, taper=(.02, .02), seed=i)
        elif v == 1: ln([(-.6, 1.2), (.1, 2.2), (.7, 3.4)], .22 * s, ac * .62); ln([(-.6, 1.14), (.1, 2.1), (.74, 3.3)], 2.)
        elif v == 2: F([(-.55, 1.1), (0, 1.26), (.55, 1.1), (.62, 1.34), (.2, 1.6), (0, 2.05), (-.2, 1.6), (-.62, 1.34)], ac * 1.25, (4,))
        for e in (-1, 1): ln([(e * .78 * sw, 1.42), (e * .70, 2.2), (e * .78, 3.4)], .08 * s)
    F([(-.36, .40), (-.34, .16), (-.2, .02), (0, 0), (.2, .02), (.34, .16), (.36, .40), (.30, .72), (.16, .95), (-.16, .95), (-.30, .72)])
    for e in (-1, 1): F([(e * .34, .42), (e * .43, .40), (e * .42, .56), (e * .33, .60)])
    if kind in ('bare', 'curly', 'spike'):
        n = 9 if kind != 'spike' else 11; L = []
        for i in range(n):
            a = math.pi * (1.15 - 1.3 * i / (n - 1)); R0 = .46 + (.06 if kind == 'bare' else .13) * (i % 2) + j(.03); L.append((R0 * math.cos(a) * (.95 if kind != 'spike' else 1.05), .38 - R0 * math.sin(a) * (1 + .25 * (kind == 'spike') * (i % 2))))
        F(L + [(.30, .86), (.12, .80), (0, .92), (-.12, .82), (-.30, .88)], dk, tuple(range(1, n, 2)) if kind == 'spike' else ())
    elif kind == 'bandana': F([(-.40, .44), (-.36, .10), (0, -.07), (.36, .10), (.40, .44), (.1, .52), (-.1, .52)], dk); F([(.30, .46), (.62, .40 + j()), (.58, .62 + j()), (.34, .56)], dk); F([(.30, .50), (.56, .72), (.42, .84), (.28, .58)], dk)
    elif kind == 'tricorn': F([(-.86, .16), (-.5, .10), (-.34, -.16), (0, -.30), (.34, -.16), (.5, .10), (.86, .16), (.6, .40), (0, .30), (-.6, .40)], dk, (0, 6))
    elif kind == 'cap': F([(-.40, .34), (-.46, .06), (-.2, -.10), (.2, -.10), (.46, .06), (.40, .34), (0, .42)], dk)
    elif kind == 'wide': F([(-1.0, .34 + j()), (-.5, .16), (-.36, -.18), (-.1, -.26), (.1, -.26), (.36, -.18), (.5, .16), (1.0, .30 + j()), (.5, .46), (0, .50), (-.5, .46)], dk, (0, 7))
    elif kind == 'top': F([(-.60, .30), (-.40, .20), (-.42, -.62), (0, -.70), (.42, -.62), (.40, .20), (.60, .30), (0, .42)], dk, (0, 2, 4, 6))
    elif kind in ('bun', 'tail'):
        F([(-.40, .6), (-.42, .2), (-.2, -.04), (.2, -.04), (.42, .2), (.40, .6), (.2, .9), (-.2, .9)], dk)
        F([(-.2, -.02), (-.22, -.26), (0, -.38), (.22, -.26), (.2, -.02)] if kind == 'bun' else [(-.14, .1), (.14, .1), (.22, .7), (.12, 1.5), (0, 1.8), (-.1, 1.4), (-.2, .7)], dk)
    elif kind in ('hood', 'scarf'): F([(-.50, 1.10), (-.50, .42), (-.34, .02), (0, -.16), (.34, .02), (.50, .42), (.50, 1.10), (0, 1.25)], dk, (3,) if kind == 'hood' else ())
    elif kind == 'turban': F([(-.46, .34), (-.50, .05), (-.2, -.22), (.2, -.22), (.50, .05), (.46, .34), (0, .42)], dk); F([(.1, -.2), (.3, -.4), (.34, -.1)], dk)
    elif kind == 'horn':
        F([(-.42, .4), (-.40, .05), (0, -.12), (.40, .05), (.42, .4), (0, .5)], dk)
        for e in (-1, 1): F([(e * .34, .16), (e * .62, .06), (e * .72, -.34), (e * .86, .04), (e * .6, .34)], col, (2,))
    if arms: F([(.62, 1.35), (.92, .4), (.98, -.5), (1.10, -.72), (1.28, -.74), (1.30, -.45), (1.18, -.2), (1.22, .6), (.98, 1.5)], col if acc is None else acc)


def plate(d, seed, n, col, dk, yh, ez, w, h, oy=0, focal=960., rows=1, sel=None, hgt=1.70, accs=(), arm=.1, x0=-200, x1=2200, skip=(), rimc='#ffe6b2', ramt=.9, rimpx=3.):
    """one crowd plane at distance d metres as a uint8 premultiplied RGBA plate (w x h, frame y offset oy); sel = 0/1 -> half of the people"""
    r = np.random.default_rng(seed); C = Cel(w, h, 0, oy, ss=1); u = focal / d; xs = np.linspace(x0, x1, n) + r.normal(0, (x1 - x0) / n * .26, n); order = r.permutation(n)
    for i in order:
        kind = KINDS[int(r.integers(len(KINDS)))]; hh = hgt + r.normal(0, .07); arms = r.random() < arm; sg = 1 if r.random() < .5 else -1; acc = accs[int(r.integers(len(accs)))] if accs and r.random() < .6 else None; rr = np.random.default_rng(seed * 77 + int(i))
        if (sel is not None and i % 2 != sel) or any(a < xs[i] < b for a, b in skip): continue
        person(C, xs[i], yh + u * (ez - hh), .235 * u, kind, col, dk, rr, arms, sg, acc, det=d < 7.)
    L = C.layer(h, w) if False else C.rgba(); L = rim(L, rimpx * .25, rimpx, rimc, ramt, .6); yy = np.arange(h, dtype=f32)[:, None, None] + oy; L[..., :3] *= 1 - .62 * smooth(yh - 40 + u * .12, yh + 130 + u * .62, yy) * (d < 10); return L


def u8(L): return (np.clip(L, 0, 1) * 255 + .5).astype(np.uint8)


def put8(img, P8, dx, dy, op=1.0):
    """premultiplied uint8 plate over frame, sub-pixel shift; plate origin at frame (dx,dy)"""
    L = cv2.warpAffine(P8, np.float32([[1, 0, dx], [0, 1, dy]]), (W, HH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT).astype(f32) / 255.
    return img * (1 - L[..., 3:4] * op) + L[..., :3] * op


def town(w, h, yh, seed=5, cols=('#8d979b', '#6d7a86', '#4f5d6c')):
    """hazy half-timbered houses both sides of the square, watchers on balconies, masts behind"""
    r = np.random.default_rng(seed); C = Cel(w, h, ss=1)
    for i in range(9):
        x = r.uniform(60, w - 60); C.line(np.array([(x, yh), (x + r.normal(0, 4), yh - r.uniform(260, 400))]), 4, '#9aa29f', taper=(0, .4), seed=i)
        for yy in (yh - r.uniform(160, 220), yh - r.uniform(250, 320)): C.line(np.array([(x - 46, yy + 4), (x + 46, yy - 4)]), 2.4, '#9aa29f', seed=i + 9)
        C.fill(spl(A((x + 4, yh - 300), (x + 44, yh - 292), (x + 40, yh - 270), (x + 4, yh - 276)), True, (0, 3)), '#9aa29f')
    for lay, (c, hh) in enumerate(((cols[0], 180), (cols[1], 270), (cols[2], 380))):
        c = rgb(c); d, e = c * .84, c * .70
        for side in (-1, 1):
            x = w / 2 + side * (250, 420, 640)[lay]
            while -300 < x < w + 300:
                bw = r.uniform(110, 200) * (1 + lay * .30); top = yh - hh * r.uniform(.6, 1.12) - max(0, abs(x - w / 2) - 500) * .12 * lay; x0, x1 = (x, x + bw) if side > 0 else (x - bw, x); g = bw * r.uniform(.5, .95); m = (x0 + x1) / 2 + r.normal(0, bw * .10); lit = -side
                C.fill(np.array([(x0, h), (x0, top), (m, top - g), (x1, top), (x1, h)]), c); sx = x0 if side < 0 else x1; C.fill(np.array([(sx, h), (sx, top), (sx - side * bw * .30, top - g * .55), (sx - side * bw * .30, h)]), d)
                for q0, q1 in (((x0 - 16, top + 18), (m, top - g - 8)), ((x1 + 16, top + 18), (m, top - g - 8))):
                    q0, q1 = A(*q0), A(*q1); C.fill(np.array([q0, q1, q1 + A(0, 20 + lay * 6), q0 + A(0, 16 + lay * 4)]), e); C.line(np.stack([q0, q1]), 2., rgb('#efe2c2') * .95, taper=(.1, .1), seed=int(r.integers(999)), alpha=.7)
                for k in range(3):                                                               # timber frame
                    yy = top + 14 + k * 78; C.line(np.array([(x0, yy), (x1, yy)]), 5., e, taper=(0, 0), seed=k)
                    for v in np.linspace(x0 + 4, x1 - 4, 4): C.line(np.array([(v, yy), (v + r.normal(0, 2), yy + 78)]), 4., e, taper=(0, 0), seed=k + 4)
                    if r.random() < .6: v = r.uniform(x0, x1 - 50); C.line(np.array([(v, yy + 76), (v + 46, yy + 2)]), 4., e, taper=(0, 0), seed=k + 7)
                    for v in np.linspace(x0 + bw * .2, x1 - bw * .2, 2 + (bw > 180)):
                        C.fill(np.array([(v - 11, yy + 20), (v + 11, yy + 20), (v + 11, yy + 54), (v - 11, yy + 54)]), c * .50); C.line(np.array([(v, yy + 20), (v, yy + 54)]), 2., e); C.fill(np.array([(v - 15, yy + 54), (v + 15, yy + 54), (v + 15, yy + 59), (v - 15, yy + 59)]), e)
                if r.random() < .7: cx = r.uniform(x0 + 16, x1 - 36); C.fill(np.array([(cx, top), (cx, top - g * .95), (cx + 22, top - g * .95), (cx + 22, top)]), d); C.fill(np.array([(cx - 4, top - g * .95), (cx - 4, top - g * .95 - 8), (cx + 26, top - g * .95 - 8), (cx + 26, top - g * .95)]), e)
                if lay == 2 and r.random() < .9:
                    for k in range(6): person(C, r.uniform(x0 + 6, x1 - 6), top + 64 + r.uniform(-6, 6), 17, KINDS[int(r.integers(len(KINDS)))], c * .56, c * .46, r, arms=r.random() < .25)
                    C.fill(np.array([(x0 - 6, top + 92), (x1 + 6, top + 92), (x1 + 6, top + 150), (x0 - 6, top + 150)]), d); C.fill(np.array([(x0 - 8, top + 88), (x1 + 8, top + 88), (x1 + 8, top + 97), (x0 - 8, top + 97)]), e)
                    for v in np.arange(x0, x1, 13): C.line(np.array([(v, top + 96), (v, top + 150)]), 3., e, taper=(0, 0))
                if lay and r.random() < .5: v = r.uniform(x0, x1); C.fill(spl(A((v, top + 30), (v + 34, top + 34), (v + 30, top + 120), (v + 17, top + 108), (v + 2, top + 120)), True, (0, 1, 2, 3, 4)), c * A(.78, .62, .62))
                x += side * bw * r.uniform(.86, 1.0)
        C.ctx.save(); C.ctx.restore()
    L = C.rgba(); L = rim(L, 0., 3., '#fff0c8', .5, .6); n = fbm(h, w, (40, 14, 5), (1, .6, .5), 9)[..., None]; L[..., :3] *= 1 + .035 * n; L = para(L, (0, yh - 460), (0, yh + 20), [(0, '#f4ece2'), (1, '#ffffff')], 'mul'); hz = np.clip((np.arange(h, dtype=f32) - (yh - 300)) / 330, 0, 1)[:, None, None] ** 1.3 * .55 * L[..., 3:4]; L[..., :3] = L[..., :3] * (1 - hz) + rgb('#c9bea0') * hz; return L


def gull(cel, x, y, s, ph, col='#232733', hi='#f3ead2', bank=0.):
    """a gull from below, wings on a flap phase (radians)"""
    a = .55 * math.sin(ph); b = .75 * math.sin(ph - .7); c, sn = math.cos(bank), math.sin(bank)
    R = lambda L: np.array([(x + (p * c - q * sn) * s, y + (p * sn + q * c) * s) for p, q in L])
    for sg in (-1, 1):
        w = [(0, -.06), (sg * .34, -.10 - a * .35), (sg * .62, -.06 - a * .55 - b * .12), (sg * 1.02, .06 - a * .5 - b * .45), (sg * .64, .06 - a * .5 - b * .10), (sg * .32, .10 - a * .28), (0, .10)]; W_ = spl(R(w), True, (3,)); cel.fill(W_, col); cel.fill(spl(R([(0, -.06), (sg * .34, -.10 - a * .35), (sg * .62, -.06 - a * .55 - b * .12), (sg * .9, .02 - a * .5 - b * .36), (sg * .6, -.01 - a * .53 - b * .1), (sg * .32, -.02 - a * .3)]), True, (3,)), hi)
    cel.fill(spl(R([(-.07, -.1), (0, -.30), (.07, -.1), (.06, .2), (.12, .40), (0, .34), (-.12, .40), (-.06, .2)]), True, (1, 4, 6)), col)
