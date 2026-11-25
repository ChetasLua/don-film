"""f5 (90): the dragon wraps itself in fire and dives; an arm grows past the size of the island; they meet, hold, and the fire gives."""
import numpy as np, cv2, math
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp
from op.ink import Cel, spl, resample, noise1
from op import bolt, kit, hands as HD, fx_fire as FF
from op.clouds import Sky
from op.comp import blur, radial, screen
from op.cam import Camera3D
from op.fx_core import h1, noise, put, FX, lin01, R as RNG, band, leaf
from chars import kaido as K
from shots import dragonkit as DK

INK = '#160e18'
SK_N = dict(skin='#c49a8a', shade='#5c4670', line=INK, nail='#d6b0a0', nail_shade='#70587c', light='#e6ecff')
SK_F = dict(skin='#d08a5e', shade='#5a2f52', line='#180a12', nail='#e0a070', nail_shade='#6e3c56', light='#ffe0a0')


def seaplate():
    h, w = 1240, 2300; hz = 520
    g = np.vstack([DK.vgrad(hz + 60, w, [(0, '#060a1c'), (.45, '#10193c'), (.8, '#2a3468'), (1., '#545a94')]), DK.vgrad(h - hz - 60, w, [(0, '#3a4378'), (1., '#161d40')])]); x, y = grid(h, w); d = np.hypot(x - 560, y - 190)
    g = g + (np.exp(-(d / 380) ** 1.4) * .40)[..., None] * rgb('#7f9be0')
    S = Sky(h, w, g, light=(-.35, -.80, .45), seed=3); S.strata(300, 500, 40, ('#5a64a0', '#2a3468'), 2); r = np.random.default_rng(6)
    for x_, w_, h_ in ((120, 700, 200), (900, 620, 130), (1500, 800, 240), (2150, 760, 300)): S.bank(x_, hz + 26, w_, h_, ('#8f9cd6', '#555f9c', '#303a70', '#1c244c'), '#3f487e', seed=int(x_), n=6)
    for k in range(8):
        q = (k / 7.) ** 1.9; yb = hz + 46 + q * 640; m = int(11 - 5 * q)
        a, b = np.array([rgb(c) for c in ('#98a4da', '#5a66a4', '#343e78', '#20284f')]), np.array([rgb(c) for c in ('#aab6ea', '#4f5ca0', '#262f66', '#121836')]); T = a + (b - a) * q
        for i in range(m): S.bank(w * (i + .5 + r.uniform(-.4, .4)) / m, yb + r.uniform(-14, 14) * (1 + 2 * q), w / m * r.uniform(1.2, 2.0), (44 + 170 * q) * r.uniform(.7, 1.25), T, rgb('#3d467c') * (1 - .4 * q), seed=k * 50 + i, tiers=3, n=6, levels=3 if q < .5 else 3, soft=1.4 * q, ss=1 if q > .6 else 2)
    return S.paint(3, amt=.75)


def townplate():
    """what lies under the cloud: black sea, the island, a festival town full of lights"""
    h, w = 620, 1500; x, y = grid(h, w); g = DK.vgrad(h, w, [(0, '#141a40'), (.5, '#0c1130'), (1., '#070a20')]); n = fbm(h, w, (60, 14, 5), (1, .7, .4), 2, aspect=5.); g = g * (1 + .22 * n[..., None])
    c = Cel(w, h); land = spl([(220, 420), (330, 330), (470, 300), (620, 250), (800, 262), (960, 300), (1120, 318), (1250, 390), (1180, 470), (980, 520), (700, 540), (420, 510)], True, n=10); c.fill(land, '#1a1c38'); c.clip(land); c.fill(land + [0, 60], '#10122a'); c.unclip(); kit.outline(c, land, 3, '#060716')
    DK.skull_dome(c, 760, 300, 250, ('#4a4a7a', '#30315e', '#1d1f46', '#0c0d26'), lw=2.2, seed=2)
    for yy, hh, n_, sd in ((360, 34, 34, 1), (404, 40, 36, 2), (452, 48, 30, 3)): DK.roofs(c, 330, 1190, yy, hh, sd, n=n_, lw=1.4)
    l = c.layer(h, w); g = g * (1 - l[..., 3:4]) + l[..., :3]; r = np.random.default_rng(4); gl = np.zeros((h, w), f32)
    for i in range(420): a, b = r.uniform(300, 1200), 330 + 180 * r.random(); cv2.circle(gl, (int(a), int(b)), int(r.integers(1, 3)), float(r.uniform(.4, 1.)), -1, cv2.LINE_AA)
    g = g + (gl * .9 + cv2.GaussianBlur(gl, (0, 0), 5) * 1.6 + cv2.GaussianBlur(gl, (0, 0), 26) * 2.6)[..., None] * rgb('#ff9a3a')
    return g


KN, WR, FC, AX = np.array([1619., 720.]), np.array([799., 1028.]), np.array([1235., 730.]), np.array([-.924, .383])


def fist_layer(light, pal, res=1500, fire=False, ink=False):
    """the fist alone, knuckles to the right and at us.  2000x1500 canvas, s = 820 px per hand unit, cut at the wrist (cloud cuff goes there)"""
    w, h = 2000, 1500; cam = HD.Cam(1250, 700, 820, 3.4); R_ = HD.Ry(HD.D(46)) @ HD.Rz(HD.D(-6)) @ HD.euler(180, 72, 0)
    hd = HD.Hand('fist', 'L', R_, (0, 0, 0), HD.HEAVY, 1.0); g = HD.render(hd, cam, light=light, res=res, clip=(0, 0, w, h), thr=.10)
    cel = Cel(w, h); P = HD.PAL['day'] | pal; HD.fill_flat(cel, g, P); HD.fill_shadow(cel, g, P); HD.fill_overlays(cel, g, P, 12.)
    if not ink:
        HD.fill_polys(cel, g, HD.depth_polys(g, light, -.34), '#331d3c', .85); HD.fill_polys(cel, g, HD.rim_polys(g, (.95, .25, 0.) if fire else (-.65, .75, 0.), lo=.22, graze=.66, eps=.02), '#ffcf7a' if fire else '#dfe8ff', .95)
    HD.hatch(cel, g, None, .034, 4.6, pal['line'], 3, 1.0, .85); HD.feather(cel, g, .034, .17, 5.2, pal['line'], 4); HD.brush_lines(cel, g, 17., pal['line'], 2, .55)
    l = cel.rgba()
    if not ink: l = kit.para(l, (1600, 500), (800, 1000), [(0, '#8a6a86' if not fire else '#a87474'), (.35, '#c8b4c4'), (1., '#ffffff')])
    x, y = grid(h, w); d = (x - WR[0]) * AX[0] + (y - WR[1]) * AX[1]; return l * smooth(52, 44, d)[..., None]


def cuff(cx, cy, u, ax, f, pal=('#ffffff', '#d9d4ec', '#9c93c4'), ink='#1a1426', lw=4.5, seed=0):
    """a collar of cloud round the wrist; u = half width px, ax = direction of the arm"""
    n = np.array([-ax[1], ax[0]]); n = n if n[1] < 0 else -n; cel = Cel(); L = []
    for row, (o, m, rr) in enumerate(((.16, 8, .30), (-.10, 9, .34))):
        for i in range(m):
            v = -1.08 + 2.16 * (i + .5 * (row % 2)) / (m - 1); wob = .05 * math.sin(f * .45 + i * 1.9 + row); p = np.array([cx, cy]) + n * u * v * 1.04 + ax * u * (o + .30 * (1 - v * v) * -.5 + wob); L.append((p, u * rr * (.75 + .5 * h1(seed, i, row)) * (1 + wob)))
    for k, (p, r) in enumerate(L[:3] + L[-3:]):                           # licks of steam leaving the collar
        q = p - n * 0 + ax * u * .2; t = lin01(14); P = q[None] + ax[None] * (u * (.6 + .7 * h1(seed, k, 5)) * t)[:, None] + n[None] * (u * (.16 * np.sin(t * 5 - f * .5 + k) * t + (.5 if k < 3 else -.5) * t ** 2))[:, None]; b = band(P, lambda v: r * .62 * leaf(v, .25), seed + k, 0, 40, .25); cel.fill(b, pal[1]); kit.outline(cel, b, lw * .7, ink, seed=k)
    a = np.linspace(0, 6.283, 34, endpoint=False); c = cel.ctx
    for p, r in L: cel.fill(np.c_[p[0] + np.cos(a) * (r + lw), p[1] + np.sin(a) * (r + lw)], ink)
    for p, r in L: cel.fill(np.c_[p[0] + np.cos(a) * r, p[1] + np.sin(a) * r], pal[2])
    for p, r in L: cel.fill(np.c_[p[0] - r * .14 + np.cos(a) * r * .88, p[1] - r * .20 + np.sin(a) * r * .84], pal[1])
    for p, r in L: cel.fill(np.c_[p[0] - r * .24 + np.cos(a) * r * .66, p[1] - r * .34 + np.sin(a) * r * .60], pal[0])
    for k, (p, r) in enumerate(L):
        if h1(seed, k, 8) < .45: th = np.linspace(.4, .4 + 4.6, 22); rd = r * np.linspace(.70, .12, 22); cel.line(np.c_[p[0] + np.cos(th) * rd, p[1] + np.sin(th) * rd], lw * .7, ink, taper=(.15, .6), seed=k)
    return cel.layer()


class Shot:
    id = 'f5'; n = 90; finish = dict(bloom=.36, vig=.16)
    EVENTS = [(0, 'growl', dict(size=1.)), (7, 'fire', dict(size=1.2, note='ignites tail to head', dur=8)), (14, 'roar', dict(size=1., note='dives')), (18, 'whoosh', dict(size=1.2, note='fire over the lens')), (22, 'stretch', dict(size=.6, note='first swell')), (30, 'stretch', dict(size=.9, note='second swell')), (37, 'stretch', dict(size=1.2, note='third swell, black lightning')), (24, 'crackle', dict(note='black lightning', dur=50)),
              (42, 'whoosh', dict(size=1.3, note='launch')), (45, 'impact', dict(size=1.6, note='fist meets fire')), (46, 'stop', dict(note='hit stop 4 f')), (50, 'strain', dict(note='both forces hold', dur=24)), (60, 'drum', dict(size=1., note='second pulse runs down the arm')), (75, 'impact', dict(size=1.8, note='the fire gives')), (80, 'white', dict(note='white, ringing', dur=10))]

    def setup(self):
        self.sky = DK.full('night', DK.night_plate); self.sea = DK.full('f5sea', seaplate); self.town = DK.full('f5town', townplate)
        a = DK.full('f5fistN4', lambda: fist_layer((-.55, .62, .50), SK_N)); b = DK.full('f5fistF4', lambda: fist_layer((.80, .05, .50), SK_F, fire=True)); self.fN, self.fF = a, b
        self.lan = DK.Lanterns(330, 5, box=((-150, 150), (46, 360), (-62, 70)), size=2.6); self.n2 = fbm(H, W, (90, 34, 12), (1, .7, .4), 12); self.camL = Camera3D((0, -10, 6), (0, 100, 4), 1250)
        self.lf = DK.Lanterns(26, 11, box=((-30, 30), (14, 34), (-16, 16)), size=1.7, rise=(2., 3.4))

    # ------------------------------------------------------------ A: fire and dive
    def A(self, f):
        t = f / 24.; cam = Camera3D((0, -92, -6 + .15 * f), (0, 60, 6), 1300, roll=-.06 + .004 * f); img = DK.skydome(self.sky, cam, 1.30, (.60, -.50)); img = DK.moon(img, 700, 250, 120)
        rear = smooth(0, 8, f) * (1 - smooth(13, 15, f)); dv = smooth(13.5, 19.5, f) ** 1.25; i = np.arange(17); fl = .10 * f + 1.2 * dv; a = 1.9 - i * .60 + fl * .5; Rr = (52 - 5 * rear) * (1 - .012 * i)
        P = np.c_[Rr * np.cos(a), 66 + 24 * np.sin(a) + 2.5 * i, 14 - 3.6 * i + 20 * np.sin(a * 1.0 + .6) * .55 + 2. * np.sin(i * 1.3 - t * 5)]
        hp = np.array([-6, 34, 30.]) + rear * np.array([-5, 16, 13]) + dv * np.array([8., -104, -38]); nk = np.array([-2, 50, 36.]) + rear * np.array([-2, 8, 10]) + dv * np.array([6., -74, -26]); n2 = np.array([10, 62, 33.]) + dv * np.array([6., -30, -10]); P = np.vstack([hp, nk, n2, P])
        D = K.Dragon(P, R=6.6, n=620, around=32, legs=(.12, .45), mane_step=.36); bn = 1.35 * smooth(6, 15, f) ** .8
        o = D.render(cam, f, dict(K.LIT['storm'], rw=12., L=(-.5, -.1, .8)), burn=bn, eyes=1 + .3 * smooth(12, 15, f), furn=.4 + .7 * smooth(8, 15, f), jaw=.12 + .10 * rear + .85 * smooth(13, 17, f), breath=1. * smooth(13, 16, f), wet=.4, wind=(.0, .5 * dv + .1, .25 + .3 * smooth(6, 12, f)), hd=(0., .10 * rear, 0), tongues=int(230 * smooth(6, 14, f)), fire=bn > 1.0, scales=36, sm_c=(0, 60, 15), sm_r=75., fog=('#33427c', 120, 330, .5), grad=((0, 0), (0, H), [(0, '#ffffff'), (1., '#b9b4da')]), look=(0, -.1))
        # the coil lights the cloud from inside
        fr = np.clip((bn - .15) / 1.0, 0, 1); img = screen(img, (radial(960, 560, 1100, 800, 1.6) * .55 * fr)[..., None] * rgb('#ff6a1e')) * (1 - .18 * fr); L = o['layer']; img = img * (1 - L[..., 3:4]) + L[..., :3]
        if o['light'] is not None: img = img + o['light']
        if f >= 6: e = FF.embers(f, 110, 4, wind=(40., -240.), size=1.4, origin=(960, 620), spread=620.); img = put(img, e)
        if f in (10, 11, 12): img = put(img, bolt.storm(f, (1660, -30), (1400, 640), 12, 3, t0=10, life=3))
        if 14 <= f < 18: img = kit.zoomblur(img, o['pts']['mouth'][0], o['pts']['mouth'][1], .05, 7)
        if f >= 17:                                                   # into the furnace: fire rolls out of the jaws and over the lens
            m = o['pts']['throat']; RR = (0, 330, 900, 2300)[f - 16]; hx, hy = grid(H // 2, W // 2); hx = hx * 2 - m[0]; hy = (hy * 2 - m[1]) * 1.12; r_ = np.hypot(hx, hy) + 1; an = np.arctan2(hy, hx); lr = np.log(r_ / 60.)
            w1 = noise(np.cos(an) * 1.3, np.sin(an) * 1.3, lr * 1.2 - t * 3.0, 5); an2 = an + .55 * w1 + .18 * lr; n1 = noise(np.cos(an2) * 2.6, np.sin(an2) * 2.6, lr * 1.15 - t * 5.0 + .6 * w1, 7); n2_ = noise(np.cos(an2) * 8.5, np.sin(an2) * 8.5, lr * 1.5 - t * 8.0, 9)
            v = cv2.resize((1.05 - r_ / RR + .46 * n1 + .44 * n2_).astype(f32), (W, H), interpolation=cv2.INTER_CUBIC); v = cv2.GaussianBlur(v, (0, 0), 1.2)
            for th, cc in ((.10, '#7a100c'), (.22, '#c8260c'), (.44, '#f2650e'), (.74, '#ffa62a'), (1.02, '#ffe07a'), (1.30, '#fff6d0')): a_ = smooth(th - .012, th + .012, v)[..., None]; img = img * (1 - a_) + rgb(cc) * a_
            for th in (.22, .44, .74): ln = smooth(.016, .006, np.abs(v - th)) * smooth(-.2, .25, self.n2); img = img * (1 - ln[..., None] * .8) + rgb('#5a0f08') * ln[..., None] * .8
            q_ = cv2.GaussianBlur((np.clip(v, 0, 1.3) * smooth(.1, .5, v))[::4, ::4], (0, 0), 26); img = img + cv2.resize(q_, (W, H))[..., None] * rgb('#ff6a1c') * .30
        o_ = kit.shake(f, 3 + 8 * (f >= 14), 2); return cv2.warpAffine(img.astype(f32), np.float32([[1, 0, o_[0]], [0, 1, o_[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    # ------------------------------------------------------------ pieces
    def seabg(self, f, ox=0., oy=0., z=1., hole=0., hc=(960, 830), warm=0.):
        M = np.float32([[z, 0, W / 2 - z * (1150 + ox)], [0, z, H / 2 - z * (600 + oy)]]); img = cv2.warpAffine(self.sea, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        if hole > 0:
            x, y = grid(); d = np.sqrt(((x - hc[0]) / 1.0) ** 2 + ((y - hc[1]) / .30) ** 2); rr = hole * 760; edge = d + 120 * self.n2 - rr; m = smooth(14, -14, edge); lip = np.exp(-((edge - 40) / 60) ** 2)
            tw = cv2.warpAffine(self.town, np.float32([[1.02, 0, hc[0] - 765], [0, .92, hc[1] - 350]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
            img = cv2.remap(img, (x - (x - hc[0]) / (d + 1) * 60 * np.exp(-np.maximum(0, edge) / 160) * hole).astype(f32), (y - 40 * np.exp(-np.maximum(0, edge) / 120) * hole * (y > hc[1] - 260)).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
            img = img * (1 - m[..., None]) + tw * m[..., None]; img = screen(img, (lip * .55 * hole)[..., None] * rgb('#ff8a3a') * (.5 + .5 * warm)); sh = smooth(0, 70, -edge) * smooth(200, 60, -edge) * m; img = img * (1 - .4 * sh[..., None])
        return img

    def fist(self, img, src, cx, cy, s, rot=0., sx=1., sy=1., smear=0., rimc=None):
        """(cx, cy) = middle of the fist"""
        kx, ky = FC; c, n = math.cos(rot), math.sin(rot); M = np.float32([[c * s * sx, -n * s * sy, 0], [n * s * sx, c * s * sy, 0]]); M[:, 2] = [cx - M[0, 0] * kx - M[0, 1] * ky, cy - M[1, 0] * kx - M[1, 1] * ky]
        a = src if s > .6 else cv2.GaussianBlur(src, (0, 0), .35 / s); l = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)
        if smear: l = kit.smear(l, smear, 0, 29)
        return l, M

    def arm(self, P, Wd, pal, f, pulse=None, lw=5.5, lit=-1, rimc='#dfe8ff', seed=0):
        """the giant rubber arm as a drawn cel: P centre line from fist to far end, Wd half widths.  pulse = position 0..1 of the travelling bulge"""
        P = resample(spl(np.asarray(P, float), n=20), 10.); v = lin01(len(P)); w = np.interp(v, lin01(len(Wd)), Wd); w = w * (1 + .05 * np.sin(v * 17 - f * .5))
        if pulse is not None: w = w * (1 + .34 * np.exp(-((v - pulse) / .07) ** 2))
        T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; 
        if N[:, 1].mean() > 0: N = -N
        A, B = P + N * w[:, None], P - N * w[:, None]; body = np.vstack([A, B[::-1]]); c = Cel(); c.fill(body, pal['skin']); c.clip(body); r = np.random.default_rng(seed)
        wob = .10 * noise1(len(P), 16, r); lo = P - N * (w * (.02 + wob))[:, None]; c.fill(np.vstack([lo, (B - N * 8)[::-1]]), pal['shade']); lo2 = P - N * (w * (.60 + .6 * wob))[:, None]; c.fill(np.vstack([lo2, (B - N * 8)[::-1]]), '#33203f')
        n = len(P)
        for i in range(0, n - 3, 2):                                   # hatch off the shadow, wrapped round the form
            if h1(seed, i, 4) < .62: q = lo[i]; L = w[i] * (.16 + .26 * h1(seed, i, 5)); c.line(spl([q, q + N[i] * L * .6 + T[i] * L * .12, q + N[i] * L + T[i] * L * .4], n=5), lw * .42, pal['line'], taper=(.05, .75), seed=i, alpha=.9)
        for k in range(5):                                             # tendons and stretched folds
            o = -.55 + .28 * k; i0 = int(n * (.02 + .2 * h1(seed, k, 6))); i1 = int(n * (.45 + .45 * h1(seed, k, 7))); c.line(P[i0:i1] + N[i0:i1] * (w[i0:i1] * (o + .06 * np.sin(v[i0:i1] * 9 + k)))[:, None], lw * .5, pal['line'], taper=(.15, .6), seed=k + 40)
        rp = np.vstack([A + N * 6, (A - N * np.maximum(5, w * .07)[:, None])[::-1]]); c.fill(rp, rimc, .95)
        c.unclip(); c.line(A, lw * 1.05, pal['line'], taper=(.02, .02), seed=1, press=.4); c.line(B, lw * 1.25, pal['line'], taper=(.02, .02), seed=2, press=.4); return c.layer(), P, w

    # ------------------------------------------------------------ B: the arm grows
    def B(self, f):
        k = f - 20; t = f / 24.; img = self.seabg(f, -40 + 3.2 * k, 4 + 1.1 * k, 1.03 + .002 * k); img = DK.moon(img, 380 - 3 * k, 170 - k, 74)
        c = Cel(); DK.skull_dome(c, 1440 - 2.2 * k, 705 - 1.0 * k, 210, ('#59618f', '#3a4270', '#232a54', '#0e1230'), lw=2.2, seed=5); DK.roofs(c, 1250 - 2.2 * k, 1640 - 2.2 * k, 738 - k, 22, 3, n=22, lw=1.2); l = c.layer(); y = grid()[1]; l = l * (1 - smooth(690 - k, 752 - k, y + 18 * self.n2))[..., None]; img = img * (1 - l[..., 3:4]) + l[..., :3]
        img = screen(img, radial(1440, 720, 330, 110, 1.8)[..., None] * rgb('#ff8a30') * .55); img = self.lan.draw(img, self.camL, t, zmin=120)
        # growth: three pumps, each overshoots and settles
        g = .10
        for t0, a, b in ((22, .10, .36), (30, .36, .66), (37, .66, 1.0)):
            if f >= t0: q = f - t0; u = min(1, q / 3.); e = 1 - (1 - u) ** 2; g = a + (b - a) * (e + .30 * math.exp(-max(0, q - 2) / 2.2) * math.sin(max(0, q - 1) * 1.25) * (q > 1))
            elif t0 - 3 <= f < t0: g *= 1 - .07 * smooth(t0 - 3, t0, f)          # squeeze before each swell
        pull = smooth(39.5, 42.4, f); go = max(0, f - 42.4) / 1.6; fx, fy = 300 + 470 * g - 120 * pull + 1300 * go ** 1.6, 560 - 170 * g - 10 * pull + 40 * go; sc = .05 + 1.02 * g; sq = 1 + (.08 * math.sin(f * 1.3) * math.exp(-((f - 22) % 8) / 2.5)) + .10 * go - .05 * pull
        fig = np.array([1180 + 1.2 * k, 985 - 1.2 * k]); fig_h = fig + [-44, -112]
        rot = -.10 + .05 * math.sin(f * .5) - .12 * pull
        l, M = self.fist(img, self.fN, fx, fy, sc, rot, 1 * sq, 1 / sq, 130 * go if go > 0 else 0); wr = M[:, :2] @ WR + M[:, 2]; w0 = 186 * sc; ax = M[:, :2] @ AX; ax /= np.hypot(*ax)
        back = np.array([wr - ax * 60 * sc, wr + ax * 260 * sc + [-10, 6], [min(150, wr[0] - 300 * sc) - 150 * g, 690 + 150 * g], [130 - 190 * g, 890 + 40 * g], [360, 985 + 40 * g], [780, 1010 + 20 * g], fig + [-120, -70], fig_h])
        pl = None
        for t0 in (22, 30, 37):
            if t0 - 5 <= f < t0 + 2: pl = 1 - (f - t0 + 5) / 6.
        al, P, w = self.arm(back, [w0, w0 * 1.22, w0 * 1.12, 110 * g + 14, 46 * g + 9, 26 * g + 8, 10, 6], SK_N, f, pl, lw=max(2.2, 6 * sc), seed=3)
        img = img * (1 - al[..., 3:4]) + al[..., :3]; img = img * (1 - l[..., 3:4]) + l[..., :3]
        cf = cuff(wr[0] + ax[0] * 20 * sc, wr[1] + ax[1] * 20 * sc, 228 * sc, ax, f, lw=max(1.6, 5.5 * sc), seed=2); img = img * (1 - cf[..., 3:4]) + cf[..., :3]
        bx = bolt.web(f, P[:int(len(P) * .75)], n=int(2 + 4 * g), w=10 * g + 3, seed=2, off=150 * sc, amt=g, edge=4.5)
        if bx is not None and f >= 24: img = put(img, bx)
        q = bolt.crackle(f, fx, fy, 470 * sc, 440 * sc, n=int(4 + 9 * g), seed=4, amt=min(1, .3 + g), w=24 * g + 4, lw=3 + 3 * g)
        if q is not None and f >= 23: img = put(img, q)
        for t0 in (22, 30, 37):
            if t0 + 1 <= f < t0 + 9: c = Cel(); bolt.ring(c, fx, fy, (f - t0) / 9., 420 * sc, 1000 * sc + 240, .92, 0, 26 * sc + 6, t0, '#f2f0ff', '#6a64a8'); rl = c.layer(); img = img * (1 - rl[..., 3:4]) + rl[..., :3]; img = bolt.bend(img, fx, fy, 260 * sc + (f - t0) * 90, 110, 26 * (1 - (f - t0) / 9.), 1.)
        from chars import gear5 as G
        fl = G.figure(fig[0], fig[1], 31, 'reach', f=f // 2 * 2, light='night', hA=(-1.5, 3.6), hB=(1.3, 2.2), expr='laugh', rims=((-.6, -.9), (.9, .2)), lagT=np.array([[0, 0], [4, 2], [8, 4], [13, 6], [18, 8], [22, 10]], float)); img = screen(img, radial(fig[0], fig[1] - 80, 260, power=2.)[..., None] * rgb('#dfe4ff') * .5); img = img * (1 - fl[..., 3:4]) + fl[..., :3]
        img = self.lan.draw(img, self.camL, t, zmin=30, zmax=120); img = self.lf.draw(img, self.camL, t, zmin=8)
        o = kit.shake(f, 3.5 * g + (9 if f in (22, 23, 30, 31, 37, 38) else 0), 6); img = cv2.warpAffine(img.astype(f32), np.float32([[1, 0, o[0]], [0, 1, o[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        if go > 0: img = kit.smear(img, 60 * go, -6, 7)
        return img

    # ------------------------------------------------------------ C: the meeting
    def push(self, f):
        """where the fist stands (px): hit, hit stop, forced back, the pulse, the creep on sticks and slips, the break"""
        if f < 45: return -420 + 420 * ((f - 42.) / 3.) ** 2
        if f < 49: return 0.
        x = -78 * smooth(49, 57, f) + 5 * math.sin(f * 2.9) * smooth(49, 53, f)
        if f >= 58: st = np.array([60, 63, 65, 68, 70, 72, 74]); x += sum(22 * smooth(s - .4, s + 1.2, f) for s in st)
        if f >= 75: x += 60 * (f - 75) + 46 * (f - 75) ** 2.2
        return x

    def C(self, f, clean=False):
        t = f / 24.; px = self.push(f); give = smooth(74.5, 78, f); z = 1.02 + .0016 * (f - 45); cx, cy = 1010 + px, 478 - .12 * px; hole = smooth(45, 56, f) * .72 + .28 * smooth(73, 80, f)
        img = self.seabg(f, 40 + .2 * px, 10, 1.02, hole, (930, 880), 1.); rs = math.sin(f * 2.1) * .04; warm = .75 + rs + .25 * smooth(58, 75, f)
        img = img * (1 - .20 * smooth(45, 50, f)); img = screen(img, (radial(1300, 400, 1250, 820, 1.5) * .46 * warm)[..., None] * rgb('#ff6420')); img = screen(img, (radial(500, 330, 700, 560, 1.8) * .30)[..., None] * rgb('#7a6cff'))
        # lanterns below and far, pushed away from the blow
        wv = 60 * smooth(45, 52, f)
        suck = smooth(49, 74, f) * (1 - smooth(74.5, 76.5, f)); P0 = np.array([(cx - 960) / 1250. * 110, 100, 6 - (cy - 540) / 1250. * 110])
        def pf(P):                                                     # the blow pushes them out, then the strain drinks them in
            d = P - P0; l = np.linalg.norm(d * [1, .6, 1], axis=1)[:, None] + 1e-6; tw = np.stack([-d[:, 2], 0 * l[:, 0], d[:, 0]], 1); g = np.exp(-(l / 105) ** 2)
            return d / l * wv * np.exp(-l / 70) * (1 - smooth(48, 56, f)) - d * g * .62 * suck + tw * g * .55 * suck + d / l * 90 * smooth(74.5, 78, f)
        img = self.lan.draw(img, self.camL, t, zmin=95, push=pf)
        # the dragon: a river of fire behind its jaws
        cam = Camera3D((0, -100, 0), (0, 40, 0), 1300); wx = (cx - 960) / 9.28; wz = -(cy - 540) / 9.28; hs = 18.0; tr = .5 * math.sin(f * 2.7) * smooth(46, 50, f); crush = give
        C0 = np.array([wx + 41 + 12 * crush + tr, 47., wz + 6 + 4 * crush + .4 * math.sin(f * 3.3)]); fw = np.array([-.94, -.34 + .25 * crush, -.06 - .25 * crush]); fw /= np.linalg.norm(fw); up = np.array([.06 * math.sin(f * 1.1), -.10, 1.]); xr = np.cross(fw, up); xr /= np.linalg.norm(xr); Rm = np.stack([xr, fw, np.cross(xr, fw)], 1)
        w_ = lambda i, a: a * math.sin(i * 1.2 - t * 6)
        P = np.array([C0 - fw * 6, C0 - fw * 22 + [0, 2, 2 + w_(1, 1.5)], C0 - fw * 40 + [0, 8, 12 + w_(2, 3)], [wx * .6 + 112, 78, 40 + w_(3, 4)], [112, 100, 70 + w_(4, 4)], [70, 118, 78 + w_(5, 4)], [34, 130, 62 + w_(6, 3)], [26, 150, 36 + w_(7, 3)], [60, 176, 22 + w_(8, 3)], [112, 200, 34 + w_(9, 4)], [170, 220, 60.]])
        bn = 1.4 if f < 75 else max(.05, 1.4 - .42 * (f - 74)); fire = bn > .9
        D = K.Dragon(P, R=16.6, n=420, around=34, legs=(), mane_step=.30, head=1.08)
        jaw = .62 + .05 * math.sin(f * 1.9) - .10 * smooth(50, 57, f) + .22 * smooth(60, 74, f) + .20 * crush
        o = D.render(cam, f, dict(K.LIT['fire'], L=(-.80, -.35, .35), rim='#fff0b8', rd=(-1., .1)), burn=bn, furn=1., eyes=1.38, whz=-.2, jaw=jaw, wind=(.75, .0, .16), tongues=260 if fire else 60, fire=fire, Rm=Rm, hC=C0, scales=34, look=(0, -.1), shadow=not fire, whisk=1.0, lw=(5.6, 2.8, 1.8), clouds=False, hair=1.25)
        L = o['layer']; img = img * (1 - L[..., 3:4]) + L[..., :3]
        if o['light'] is not None: img = img + o['light'] * .42
        # the arm and the fist
        hit = math.exp(-max(0, f - 45) / 3.) if f >= 45 else 0.; sq = 1 - .10 * hit * math.cos(max(0, f - 45) * 1.1) - .04 * smooth(50, 57, f) + .10 * give; s = (.80 + .30 * give) * z; sh = kit.shake(f, 2.4 if f >= 50 else 0, 9)
        l, M = self.fist(img, self.fF, cx + sh[0] - 350 * s * sq, cy + sh[1], s, -.02 + .02 * math.sin(f * .9) + .05 * give, sq, 1 / sq, 160 if f in (42, 43, 44) else 60 * give)
        wr = M[:, :2] @ WR + M[:, 2]; w0 = 186 * s; ax = M[:, :2] @ AX; ax /= np.hypot(*ax); back = np.array([wr - ax * 60 * s, wr + ax * 260 * s, wr + ax * 560 * s + [0, 14 * math.sin(f * .7)], wr + ax * 1000 * s])
        pl = None if not 53 <= f < 61 else 1 - (f - 53) / 7.5
        al, Pa, w = self.arm(back, [w0, w0 * 1.22, w0 * 1.15, w0 * .95], SK_F, f, pl, lw=6., rimc='#8a7ad0', seed=5); img = img * (1 - al[..., 3:4]) + al[..., :3]; cf = cuff(wr[0] + ax[0] * 20 * s, wr[1] + ax[1] * 20 * s, 228 * s, ax, f, ('#ffe9d0', '#d8a8b0', '#8a6a9a'), seed=2)
        # fire parts round the fist like water round a stone: behind it first
        c = Cel(); str_ = smooth(44.6, 46.5, f) * (1 - smooth(76, 79, f)); kp = np.array([cx + 20 * s * sq, cy - 10])
        def stream(c, i, sd, th0, w_, ln):
            a = th0; d0 = np.array([math.cos(a), math.sin(a)]); p0 = kp + d0 * 40; up_ = 1 if math.sin(a) < 0 else -1; sp = spl(np.array([p0 + [90, 0], kp + [10, -up_ * 120 * abs(math.sin(a))], [cx - 190, cy - up_ * (430 + 60 * i) * s], [cx - 420 - 60 * i, cy - up_ * (520 + 64 * i + 20 * math.sin(f * .8 + i)) * s], [cx - 420 - ln, cy - up_ * (590 + 120 * i + 40 * math.sin(f * .6 + i * 2)) * s]]), n=14)
            FF.flame_stream(c, sp, w_, t + i * .3, seed=sd, smooth=.75, speed=1500., slivers=False)
        if str_ > .05:
            for i in range(2): stream(c, i + 1, 20 + i, -1.2, 56 * str_, 520 + 160 * i); stream(c, i + 1, 30 + i, 1.2, 46 * str_, 480 + 160 * i)
            sl = c.layer(); img = img * (1 - sl[..., 3:4]) + sl[..., :3]; img = img + cv2.resize(cv2.GaussianBlur(sl[::4, ::4, :3], (0, 0), 14), (W, H)) * .22
        img = img * (1 - l[..., 3:4]) + l[..., :3]
        # the skin itself: knuckles flatten on the jaws, ripples run back down hand and arm
        if 45 <= f < 78:
            x, y = grid(); sg_ = smooth(44.5, 46, f) * (1 - give); dx_, dy_ = x - kp[0], y - kp[1]; g = np.exp(-(dx_ / 210) ** 2 - (dy_ / 300) ** 2); amp = (30 + 16 * math.sin(f * 1.25) + 46 * hit) * sg_
            rp_ = np.sin(np.hypot(dx_, dy_) / 44. - f * 1.35) * np.exp(-np.hypot(dx_, dy_) / 620.) * (x < kp[0] + 30) * (4.2 + 9 * hit) * sg_; u = np.hypot(dx_, dy_) + 1.
            pu = 0.
            if 53 <= f < 62: pc = wr[0] + (kp[0] - wr[0]) * np.clip((f - 56) / 5., -.6, 1.); pu = np.exp(-((x - pc) / 120) ** 2) * 26 * np.sign(dy_) * np.clip(np.abs(dy_) / 260, 0, 1)
            mx, my = x + amp * g + rp_ * dx_ / u, y - .42 * amp * g * dy_ / 160. + rp_ * dy_ / u - pu; mk = cv2.GaussianBlur(np.clip(l[..., 3] + al[..., 3] + cf[..., 3], 0, 1)[::2, ::2], (0, 0), 14); mk = cv2.resize(mk, (W, H))[..., None] * (x < kp[0] + 70)[..., None]
            wp = cv2.remap(img.astype(f32), mx.astype(f32), my.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); img = img * (1 - mk) + wp * mk
        # fire light eats the knuckles; the contact is white
        st = smooth(48, 55, f) * (1 - give); pw = .6 + .4 * math.sin(f * 1.25)
        if st > .02:
            c = Cel(); V = lambda P: np.asarray(P, float) @ M[:, :2].T + M[:, 2]                                                # veins stand up on the back of the hand and down the forearm
            hand_v = [[(860, 960), (960, 880), (1040, 800), (1150, 740), (1250, 640)], [(960, 880), (1010, 920), (1090, 930), (1150, 980)], [(1040, 800), (1060, 720), (1140, 620), (1230, 560)], [(1150, 740), (1230, 760), (1300, 740)]]
            an_ = np.array([ax[1], -ax[0]]); an_ = an_ if an_[1] < 0 else -an_; arm_v = [[wr + ax * a_ * s + an_ * w0 * (o_ + .14 * math.sin(a_ / 58. + q + f * .3 * 0)) for a_ in range(150, 760, 75)] for q, o_ in enumerate((.62, .18, -.22))]
            for q, P in enumerate([V(p) for p in hand_v] + arm_v): P = spl(np.asarray(P), n=8); wv_ = (15 if q < 4 else 19) * s * (.75 + .35 * pw * st); c.line(P + [1.5, 4], wv_ * 1.25, '#2a1230', taper=(.12, .45), seed=q, alpha=.95 * st); c.line(P, wv_ * .85, '#8a4a5e' if q < 4 else '#a8685a', taper=(.12, .45), seed=q, alpha=.95 * st); c.line(P - [1, 3], wv_ * .25, '#ffb27a', taper=(.2, .5), seed=q + 9, alpha=.9 * st)
            vl = c.layer(); vl = vl * np.maximum(l[..., 3], al[..., 3])[..., None]; vb = vl * (1 - cf[..., 3:4]); img = img * (1 - vb[..., 3:4]) + vb[..., :3]
        img = img * (1 - cf[..., 3:4]) + cf[..., :3]; kk = l[..., 3] * np.clip(radial(kp[0], kp[1], 520, 560, 1.6) * 1.1, 0, 1); img = img * (1 - .45 * kk[..., None]) + rgb('#ffb04a') * (.50 * kk * warm)[..., None]
        if str_ > .05:
            c = Cel()
            stream(c, 0, 50, -.9, 40 * str_, 360); stream(c, 0, 60, .9, 36 * str_, 330)
            n_ = 34; th = np.linspace(0, 6.283, n_ * 2, endpoint=False); r0 = (40 + 12 * math.sin(f * 2.3)) * str_; rr = r0 * np.where(np.arange(n_ * 2) % 2, 1., np.array([1.6 + 2.6 * h1(f // 2, i, 9) ** 2 for i in range(n_ * 2)])); bp = np.c_[kp[0] + np.cos(th) * rr * .62, kp[1] + np.sin(th) * rr * 1.25]; c.fill(kp + (bp - kp) * 1.16, '#ff7a1e'); c.fill(bp, '#fffbe6')
            sl = c.layer(); img = img * (1 - sl[..., 3:4]) + sl[..., :3]; img = img + cv2.resize(cv2.GaussianBlur(sl[::4, ::4, :3], (0, 0), 14), (W, H)) * .25
            img = img + (radial(kp[0], kp[1], 420, 420, 2.4) * .34 * str_ * (1 + rs * 3))[..., None] * rgb('#ffd9a0')
        if 46 <= f < 78:                                               # sparks shear off sideways, up and down the line of contact, and are dragged back
            c = Cel(); rs_ = RNG(f, 44); ns = int(70 + 60 * smooth(58, 74, f))
            for i in range(ns):
                up_ = -1 if rs_.random() < .55 else 1; th = up_ * (1.5708 + rs_.normal(0, .34)) ; d0 = 30 + 560 * rs_.random() ** 1.5; L_s = (40 + 170 * rs_.random() ** 2) * (1.1 - d0 / 900.); u_ = np.array([math.cos(th), math.sin(th)]); bd = -.0011 * d0; p = kp + u_ * d0 + [-d0 * d0 * .00062, 0]; u2 = np.array([u_[0] - d0 * .0013, u_[1]]); u2 /= np.hypot(*u2); n_s = np.array([-u2[1], u2[0]]) * (1.2 + 4.2 * rs_.random() ** 2); q_ = np.array([p - u2 * L_s * .5, p + n_s, p + u2 * L_s * .5, p - n_s]); c.fill(p + (q_ - p) * 1.35, '#ff6a1a'); c.fill(q_, '#fff6cf')
            sl = c.layer(); img = img * (1 - sl[..., 3:4]) + sl[..., :3]; img = img + cv2.resize(cv2.GaussianBlur(sl[::4, ::4, :3], (0, 0), 5), (W, H)) * .5
        # black lightning
        for q in (bolt.crackle(f, cx - 330 * s, cy, 500 * s, 470 * s, n=11, seed=8, amt=1., w=24., lw=5.2, out=.6), bolt.web(f, Pa, n=7, w=13, seed=6, off=w0 * .9, edge=4.2)):
            if q is not None and f >= 44: img = put(img, q)
        if f >= 45:
            for i in range(3):
                t0 = 45 + 5 * ((f - 45) // 5) + i; q = bolt.black(f, kp + [0, -30 + 30 * i], (kp[0] + (-900, 300, 820)[i] * (.7 + .5 * h1(t0, i)), (-60, 1120, -80)[i]), 26, seed=int(t0) * 3 + i, t0=t0, life=4, kids=6, edge=6)
                if q is not None: img = put(img, q)
        e = FF.embers(f, 120, 9, wind=(-260., -120.), size=1.6, origin=(kp[0], kp[1]), spread=240., life=1.2, swirl=90., near=.03)
        if f >= 45: img = put(img, e)
        img = self.lan.draw(img, self.camL, t, zmin=34, zmax=72, push=pf); img = self.lf.draw(img, self.camL, t, zmin=8, push=lambda P: pf(P) * .25)
        # shock
        for t0, am, mx in ((45, 60, 1700), (60, 26, 1200), (75, 80, 2000)):
            if t0 <= f < t0 + 10:
                q = (f - t0 + .5) / 10.; img = bolt.bend(img, kp[0], kp[1], 80 + mx * q ** .7, 150, am * (1 - q), .85); c = Cel(); bolt.ring(c, kp[0], kp[1], q, 120, mx, .85, 0, 34 if t0 != 60 else 18, t0, '#fff4e0', '#b03a3a'); rl = c.layer(); img = img * (1 - rl[..., 3:4]) + rl[..., :3]
        if clean: return img
        if f == 45: L_ = (img @ np.array([.3, .59, .11], f32))[..., None]; m = smooth(.34, .40, L_); img = rgb('#0a0509') * (1 - m) + rgb('#fff6e4') * m
        if f == 46: img = kit.impact(img, 'red', .22, .58, accent='#e02414') * .85 + img * .15
        if f >= 75: img = kit.zoomblur(img, kp[0], kp[1], .05 + .03 * (f - 75), 9); c = Cel(); kit.speed_lines(c, .05, 60, 10, (500, 1500), '#fff2d8', f, alpha=.7); sl = c.layer(); img = img * (1 - sl[..., 3:4]) + sl[..., :3]
        if f >= 77: w_ = smooth(76.2, 80, f); g = np.clip(radial(kp[0] - 120, kp[1], 500 + 2600 * w_, power=.8) * (1.4 + 2 * w_), 0, 1)[..., None]; img = img * (1 - g) + rgb('#fffaf0') * g * 1.05
        o_ = kit.shake(f, 22, 3, 3., 45) + kit.shake(f, 3.2 + 5.5 * smooth(58, 74, f), 4) * (f >= 50) + kit.shake(f, 18, 5, 3., 75) * (f >= 75)
        hd_ = smooth(49, 74, f); zz = 1.03 + .085 * hd_; an = (.0035 * (1 if f % 2 else -1)) * smooth(56, 74, f) - .012 * smooth(49, 57, f) + .036 * smooth(58, 74, f) + .003 * math.sin(f * 1.7) * hd_; M = cv2.getRotationMatrix2D((kp[0], kp[1]), math.degrees(an), zz).astype(f32); M[0, 2] += o_[0] + float(np.clip((960 - kp[0]) * .5, -50, 50)) * hd_; M[1, 2] += o_[1]; return cv2.warpAffine(img.astype(f32), M, (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def white(self, f):
        """f80-90: white.  Rays of the paper, the ring going out, ash of light"""
        k = f - 80; x, y = grid(); d = np.hypot(x - 1000, (y - 460) * 1.1); img = np.ones((H, W, 3), f32) * rgb('#fffaf0'); a = np.arctan2(y - 460, x - 1000); n = .5 + .5 * np.sin(a * 23 + 2 * np.sin(a * 7)) * np.sin(a * 41 + 1); r = smooth(.55, .9, n) * smooth(200, 900, d) * .07 * (1 - k / 13.); img = img - r[..., None] * np.array([.10, .35, .70], f32)
        c = Cel(); bolt.ring(c, 1000, 460, min(.98, .15 + k / 12.), 200, 1900, .85, 0, 22, 3, '#f4dcc0', None); bolt.ring(c, 1000, 460, min(.98, k / 14.), 100, 1500, .85, 0, 12, 8, '#f2cfae', None); r_ = np.random.default_rng(4)
        for i in range(60): q = r_.uniform(0, 6.283); dd = r_.uniform(200, 900) * (1 + .06 * k); p = np.array([1000 + math.cos(q) * dd * 1.2, 460 + math.sin(q) * dd * .9 - 4 * k]); L = r_.uniform(4, 22) * (1 - k / 14.); u = np.array([math.cos(q), math.sin(q)]); c.line([p, p + u * L], r_.uniform(1, 4), '#f0c49a', taper=(.3, .6), seed=i)
        l = c.layer(); return (img * (1 - l[..., 3:4]) + l[..., :3]).astype(f32)

    def frame(self, f):
        if f < 20: return self.A(f)
        if f < 44: return self.B(f)
        if f < 80: return self.C(f)
        return self.white(f)
