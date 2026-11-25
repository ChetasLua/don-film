"""SEA lane extras: gulls from above, top-down cloud puffs, tiny crew figures (whoop/stand), hat-brim foreground for g1/g2."""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, fbm, grid, ramp, repaint, lum
from .ink import Cel, spl, resample
from .kit import outline, shape, spike
from . import clouds

A_ = np.asarray


def shear_pose(ps, L, z=0.):
    """pose that flattens a model onto the plane z along light L: its cast shadow on the sea"""
    M = A_([[1, 0, -L[0] / L[2]], [0, 1, -L[1] / L[2]], [0, 0, 1e-3]]); return M @ ps[0], M @ (ps[1] - [0, 0, z]) + [0, 0, z]


def gull_top(cel, x, y, s, rot, flap=0., col='#fbfbf6', shade='#b9c9dc', tip='#2a2c3a', lw=1.6, ink='#1c2436'):
    """gull from above, s = half wing span px, flap -1..1 folds the span"""
    k = 1 - .28 * abs(flap); X = lambda P: A_([[x + (p[0] * k * math.cos(rot) - p[1] * math.sin(rot)) * s, y + (p[0] * k * math.sin(rot) + p[1] * math.cos(rot)) * s] for p in P])
    sw = .10 * flap
    for sg in (-1, 1):
        wg = [(0, -.14), (sg * .22, -.24), (sg * .50, -.22 - sw), (sg * .80, -.06 - sw), (sg * 1.0, .16 + sw), (sg * .74, .06), (sg * .50, .04), (sg * .25, .10), (0, .12)]
        P = spl(X(wg), True, corners=(4,), n=5); cel.fill(P, col); cel.clip(P); cel.fill(X([(0, .02), (sg * .4, -.02), (sg * 1.1, .14), (sg * .9, .4), (0, .3)]), shade); cel.fill(X([(sg * .80, -.2), (sg * 1.1, -.1), (sg * 1.1, .4), (sg * .82, .2)]), tip); cel.unclip()
        if lw: outline(cel, P, lw, ink, seed=3 + sg)
    B = spl(X([(0, -.36), (.07, -.26), (.09, 0), (.05, .24), (.10, .42), (-.10, .42), (-.05, .24), (-.09, 0), (-.07, -.26)]), True, n=5); cel.fill(B, col)
    if lw: outline(cel, B, lw, ink, seed=9)
    cel.fill(X([(-.03, -.34), (0, -.47), (.03, -.34)]), '#f0a83a')


def puff(w, h, seed=0, n=150, tones_=('#ffffff', '#eaf3fb', '#bcd5ea', '#93b7d8'), light=(-.6, -.8), spread=(.30, .24), r=(.016, .135), crisp=.05):
    """cloud from above -> premultiplied RGBA plate"""
    rr = np.random.default_rng(seed); img = np.zeros((h, w, 3), f32) + rgb(tones_[2]); A = np.zeros((h, w), f32); ang = np.zeros((h, w), f32); L = []
    ax = rr.uniform(-.3, .3)
    for i in range(n):
        q = i / n; a = rr.uniform(0, 6.283); d = abs(rr.normal(0, 1)) * (1 - .45 * q); x = w * (.5 + math.cos(a) * d * spread[0]); y = h * (.5 + math.sin(a) * d * spread[1] + (x / w - .5) * ax); rad = w * (r[1] + (r[0] - r[1]) * q ** .55) * rr.uniform(.55, 1.2) * max(.35, 1 - d * .24); L.append((x, y, rad, rad * rr.uniform(.74, .95)))
    L = A_(L); pad = L[:, 2].max() * 1.1; L[:, 0] = np.clip(L[:, 0], L[:, 2] * 1.3, w - L[:, 2] * 1.3); L[:, 1] = np.clip(L[:, 1], L[:, 3] * 1.3, h - L[:, 3] * 1.3)
    clouds.lobes(img, A, ang, L, clouds.tones(*tones_), light=light, crisp=crisp, fade=.75, wob=.24, seed=seed)
    img = repaint(img, (14, 7), seed=seed, angle=ang + fbm(h, w, (60,), None, seed) * .3, thr=(0, .012), density=1.0, aspect=3.0, jit=(.003, .01, .010), op=.8)
    A = smooth(.05, .9, A); return np.dstack([img * A[..., None], A]).astype(f32)


# ------------------------------------------------------------------ tiny crew (30..120 px), facing lens or away
CREW = dict(luffy=dict(h=1.74, skin='#f0b98d', top='#c9302c', leg='#3c62a8', hair='#17151a', hat='#e3b557', bare=.45), zoro=dict(h=1.81, skin='#d9986a', top='#2f5a3c', leg='#22262c', hair='#5fae4e', sash='#b3302f', coat='#274c34'),
            nami=dict(h=1.70, skin='#f3c7a0', top='#3f9ad0', leg='#39507a', hair='#ee8a2b', long=1), usopp=dict(h=1.76, skin='#c98e5e', top='#c98a3a', leg='#6a7a3a', hair='#1c1a1c', curly=1, hat='#8a7a4a'),
            sanji=dict(h=1.80, skin='#f0c9a2', top='#1b1b22', leg='#1b1b22', hair='#f0d46a'), chopper=dict(h=.92, skin='#b98558', top='#b98558', leg='#b23a4a', hair='#b98558', hat='#e86a8a', round=1),
            robin=dict(h=1.88, skin='#e6b48a', top='#5a3f80', leg='#3a2c58', hair='#17151f', long=1), franky=dict(h=2.4, skin='#e0a878', top='#d0473a', leg='#e0a878', hair='#4fb4e0', wide=1.9), brook=dict(h=2.6, skin='#f3efe2', top='#22202a', leg='#22202a', hair='#17151a', afro=1, thin=.55),
            jinbe=dict(h=3.0, skin='#4f86b8', top='#c9532f', leg='#8a3a2a', hair='#15151c', wide=1.7))


def mini(cel, x, y, px, who, f=0, pose='whoop', seed=0, rot=0., lw=1.2, dark=0., ink='#17131a', wind=1.):
    """tiny full figure, (x,y) feet, px pixels per metre.  pose whoop (floating, arms up) | stand"""
    c = CREW[who]; h = c['h'] * px; t = f / 24.; w = h * .13 * c.get('wide', 1.) * c.get('thin', 1.); sh = (lambda k: tuple(rgb(k) * (1 - dark))); hd = h * (.115 if who != 'chopper' else .24); ca, sa = math.cos(rot), math.sin(rot)
    X = lambda P: A_([[x + p[0] * ca - p[1] * sa, y + p[0] * sa + p[1] * ca] for p in P]); ph = seed * 1.7; sw = math.sin(t * 9 + ph); hy, sy = -h * .46, -h * .80
    def limb(P, wd, col):
        P = spl(X(P), n=5); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; u = np.linspace(1, .68, len(P))[:, None] * wd / 2; Q = np.vstack([P + N * u, (P - N * u)[::-1]]); cel.fill(Q, sh(col)); outline(cel, Q, lw * .8, ink, seed=seed)
    up = pose == 'whoop'
    for s in (-1, 1):
        kx = s * w * (.5 + (.75 if up else .1)); ky = hy * .5; fx = s * w * (.6 + (1.3 + .5 * sw * s if up else .15)); fy = (-h * .10 + sw * s * h * .06) if up else 0
        limb([(s * w * .35, hy), (kx, ky + (0 if not up else -h * .02)), (fx, fy)], w * .62, c['leg'])
    if 'coat' in c or who in ('brook', 'jinbe'): limb([(0, sy * .9), (w * (.4 - .5 * wind), hy * .8), (w * (.2 - 1.0 * wind) + sw * w * .3, hy * .28)], w * 2.0, c.get('coat', c['top']))
    T = spl(X([(-w * .62, hy), (-w * .95, sy * .92), (-w * .5, sy), (w * .5, sy), (w * .95, sy * .92), (w * .62, hy)]), True, n=4); cel.fill(T, sh(c['top'])); outline(cel, T, lw, ink, seed=seed)
    if 'sash' in c: limb([(-w * .6, hy), (w * .6, hy)], w * .4, c['sash'])
    for s in (-1, 1):
        if up: e = [(s * w * .9, sy * .95), (s * w * 1.7, sy * 1.04 - h * .04), (s * w * (1.6 + .35 * math.sin(t * 11 + ph + s)), sy * 1.04 - h * .24)]
        else: e = [(s * w * .9, sy * .95), (s * w * 1.25, hy * 1.25), (s * w * 1.15, hy * .88)]
        limb(e, w * .55 * (1.5 if who == 'franky' else 1), c['skin'] if who in ('luffy', 'nami', 'usopp', 'franky', 'chopper') else c['top'])
        q = X([e[-1]])[0]; cel.fill(np.c_[q[0] + np.cos(np.linspace(0, 6.28, 10)) * w * .30, q[1] + np.sin(np.linspace(0, 6.28, 10)) * w * .30], sh('#f3efe2' if who == 'brook' else c['skin']))
    hc = X([(0, sy - hd * .95)])[0]; a = np.linspace(0, 6.283, 22); cel.fill(np.c_[hc[0] + np.cos(a) * hd, hc[1] + np.sin(a) * hd], sh(c['skin']))
    if c.get('long'): P = spl(X([(-hd, sy - hd * 1.2), (0, sy - hd * 2.1), (hd, sy - hd * 1.2), (hd * 1.0 - wind * hd * .8, sy + hd * .9), (-wind * hd * 1.8 + sw * hd * .3, sy + hd * 1.6), (-hd * 1.1 - wind * hd * .6, sy + hd * .2)]), True, n=5); cel.fill(P, sh(c['hair'])); outline(cel, P, lw * .8, ink, seed=seed); cel.fill(np.c_[hc[0] + np.cos(a) * hd * .72, hc[1] + hd * .2 + np.sin(a) * hd * .66], sh(c['skin']))
    elif c.get('afro') or c.get('curly'): r_ = hd * (1.9 if c.get('afro') else 1.35); cel.fill(np.c_[hc[0] + np.cos(a) * r_ * (1 + .06 * np.sin(a * 9)), hc[1] - hd * .35 + np.sin(a) * r_ * (.92 + .05 * np.sin(a * 11))], sh(c['hair'])); cel.fill(np.c_[hc[0] + np.cos(a) * hd * .75, hc[1] + hd * .35 + np.sin(a) * hd * .7], sh(c['skin']))
    else:
        cr, sr = math.cos(rot), math.sin(rot); P = []
        for i in range(13): b = math.pi + math.pi * i / 12; r_ = hd * (1.32 if i % 2 else 1.02); P.append((math.cos(b) * r_, sy - hd * .95 + math.sin(b) * r_))
        P = X(P + [(hd * .8, sy - hd * 1.1), (-hd * .8, sy - hd * 1.1)]); cel.fill(P, sh(c['hair']))
    if up and hd > 3: m = X([(0, sy - hd * .55)])[0]; cel.fill(np.c_[m[0] + np.cos(a) * hd * .38, m[1] + np.sin(a) * hd * .30], '#3a1418')
    if up and hd > 4:
        for s in (-1, 1): m = X([(s * hd * .38, sy - hd * 1.10)])[0]; cel.fill(np.c_[m[0] + np.cos(a) * hd * .13, m[1] + np.sin(a) * hd * .15], ink)
    outline(cel, np.c_[hc[0] + np.cos(a) * hd, hc[1] + np.sin(a) * hd], lw * .7, ink, seed=seed)
    if 'hat' in c: b = X([(-hd * 1.75, sy - hd * 1.35), (-hd * .8, sy - hd * 1.62), (-hd * .72, sy - hd * 2.2), (0, sy - hd * 2.5), (hd * .72, sy - hd * 2.2), (hd * .8, sy - hd * 1.62), (hd * 1.75, sy - hd * 1.35), (0, sy - hd * 1.2)]); b = spl(b, True, corners=(0, 6), n=5); cel.fill(b, sh(c['hat'])); outline(cel, b, lw * .8, ink, seed=seed)


# ------------------------------------------------------------------ the near wing (a3 tail / a4 head)
def near_wing(u, f=0, blur_=3.0, speed=1.0):
    """painted gull wing almost touching the lens.  u 0 = it fills the frame, 1 = gone off the upper right.  Ten primaries, a comb of secondaries,
    three rows of coverts, warm light through the feathers from below, cool shade under every overlap, soft lens blur + travel smear -> premult RGBA"""
    from .kit import smear, para
    from .comp import blur
    hw, hh = W // 2, H // 2; c = Cel(hw, hh, ss=2); rot = .24 + .42 * u; s = 1.32 + .06 * u; ox, oy = 2080 + 1300 * u, 235 - 1950 * u; cr, sr = math.cos(rot), math.sin(rot)
    X = lambda P: (np.atleast_2d(A_(P, float)) * s @ A_([[cr, sr], [-sr, cr]]) + [ox, oy]) / 2; r = np.random.default_rng(12)

    def feather(b, ang, L, w, base, tip, dark=None, bend=.10, shaft=True, seed=0):
        b = A_(b, float); d = A_([math.cos(ang), math.sin(ang)]); n = A_([-d[1], d[0]]); v = np.linspace(0, 1, 9); ax = b + d * (v * L)[:, None] + n * (bend * L * v ** 2)[:, None]; hwid = w * np.interp(v, [0, .10, .35, .75, .93, 1], [.35, .46, .52, .50, .34, .06])
        P = np.vstack([ax + n * hwid[:, None] * .82, (ax - n * hwid[:, None] * 1.12)[::-1]]); Q = spl(X(P), True, n=4); c.fill(Q, base); c.clip(Q)
        for k in range(5): q = .30 + .14 * k; c.fill(X(np.vstack([ax[int(q * 8):] + n * w, (ax[int(q * 8):] - n * w)[::-1]])), tip, .28)
        c.fill(X(np.vstack([ax - n * hwid[:, None] * .42, (ax - n * w * 1.5)[::-1]])), '#b9c8de', .62)                              # lee half of the vane, under its neighbour
        c.fill(X(np.vstack([ax + n * hwid[:, None] * .45, (ax + n * w * 1.5)[::-1]])), '#fff0d0', .50)                                # sun leaking through the thin edge
        if dark: c.fill(X(np.vstack([ax[6:] + n * w, (ax[6:] - n * w)[::-1]])), dark, .85); c.fill(X(np.vstack([ax[5:7] + n * w, (ax[5:7] - n * w)[::-1]])), dark, .35)
        for k in range(16): q = r.uniform(.08, .96); i = min(7, int(q * 8)); p = ax[i] + (ax[i + 1] - ax[i]) * (q * 8 - i); sd = 1 if k % 2 else -1; c.line(X([p, p + d * w * .5 + n * sd * w * .5]), 1.1, '#8fa3c2' if sd < 0 else '#e2cfae', taper=(.05, .7), alpha=.45, seed=seed * 20 + k)
        if shaft: c.line(X(ax), 3.4, '#a7b6cf', taper=(.02, .5), alpha=.9, seed=seed); c.line(X(ax + n * 5), 1.6, '#ffffff', taper=(.02, .5), alpha=.8, seed=seed + 1)
        c.unclip(); c.line(Q, 2.0, '#93a5c0', taper=(.2, .3), press=.5, alpha=.65, seed=seed + 2)

    # wing bones: shoulder (0,0) -> elbow -> wrist, leading edge upward (-y), feathers hang down (+y)
    wr = A_([-1500., 60.]); sh = A_([700., -120.])
    for i in range(10): a = math.radians(188 - 9.6 * i); feather(wr + A_([46. * i, 30 + 10 * i]), a, 1500 - 46 * i, 235, '#eef3fa', '#ffffff', '#4a5166' if i < 4 else None, -.07, True, i)
    for i in range(13): q = i / 12.; feather(wr + (sh - wr) * (.10 + .92 * q) + [0, 120], math.radians(92 - 10 * (q - .4)), 1060 - 120 * q, 245, '#e6eef8', '#fffdf6', None, -.05, True, 20 + i)
    for row, (off, L, w, b, t) in enumerate(((60, 620, 190, '#dfe8f4', '#fff9ea'), (-30, 400, 150, '#e9eff8', '#fffaf0'), (-120, 250, 120, '#f3f6fb', '#ffffff'))):
        n_ = 17 + 4 * row
        for i in range(n_): q = i / (n_ - 1.); feather(wr + (sh - wr) * (-.06 + 1.12 * q) + [r.normal(0, 10), off + r.normal(0, 10)], math.radians(100 - 16 * (q - .3)) + r.normal(0, .03), L * r.uniform(.9, 1.1), w, b, t, None, -.04, row < 2, 60 + row * 30 + i)
    arm = spl(X([(-1760, 60), (-1560, -60), (-1100, -170), (-400, -330), (300, -380), (1500, -700), (2600, -1500), (2600, 300), (900, 60), (200, -40), (-500, -10), (-1200, 60), (-1560, 130)]), True, n=8); c.fill(arm, '#fbfaf4'); c.clip(arm)
    for i in range(150): p = A_([r.uniform(-1700, 1500), 0]); p[1] = -120 - .16 * (p[0] + 1500) + r.uniform(-160, 170); c.line(X([p, p + [r.uniform(-40, -10), r.uniform(26, 70)]]), r.uniform(2, 5), ['#d5dfec', '#e9eef6', '#f3e7cf'][i % 3], taper=(.1, .7), alpha=.7, seed=i)
    c.fill(X([(-1800, -400), (2600, -1600), (2600, -900), (200, -270), (-1200, -40), (-1800, 60)]), '#ffffff', .55); c.unclip()
    lay = cv2.resize(c.layer(hh, hw), (W, H), interpolation=cv2.INTER_LINEAR)
    lay = para(lay, (1500, -100), (420, 1150), [(0, '#e3ebf6'), (.45, '#fbfaf6'), (1, '#ffe7bf')], 'mul', .9); lay = blur(lay, blur_)
    v = speed * (40 + 190 * u); return smear(lay, v, -v * .70, 9)
