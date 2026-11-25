"""d7: three cups; the same stump in the rain; a scrap of paper that burns out in a palm."""
import numpy as np, math, cv2
from op import W, H, comp
from op.paint import rgb, f32, smooth, grid, ramp, fbm, repaint, Canvas
from op.comp import over, screen, blur, bokeh, lerp_keys, ease, ease_in, ease_out, radial, cam2d
from op.ink import Cel, spl
from op.kit import para, outline, rim, flare
from op import hands as HD, fx_core as FC
from op.fx_air import rain, splash, motes, ash
from chars import scaffold as SC
from chars._old import *

unit = HD.unit


def frame_of(y, zback):
    """rotation: fingers along y, back of the hand toward zback (world x right, y up, z viewer)"""
    y = unit(y); z = np.asarray(zback, float); z = unit(z - y * np.dot(z, y)); return np.stack([np.cross(y, z), y, z], -1)


CRADLE = HD.pose((34, 46, 24, -3), (38, 50, 26, 0), (42, 52, 28, 3), (46, 54, 30, 7), thumb=(44, 40, 6, 22, 14), wrist=(8, 0), cup=.35)
KIDB = HD.build(len=.90, wid=1.06, boxy=.30, tip=.40, knuckle=.82, palm=.98, wrist=.90, meat=1.12)
CUPS = dict(o=(300, 160))


def cup(C, x, y, w, tilt=0., sake=1., wet=True, crack=0., dull=0., rimc='#e8c072', lw=4.):
    """red lacquer sake dish seen a little from above"""
    c, s = math.cos(tilt), math.sin(tilt); R = lambda L: np.array([(x + (a * c - b * s) * w, y + (a * s + b * c) * w) for a, b in L]); a = np.linspace(0, 2 * math.pi, 60)[:-1]; m = lambda u, v: tuple(rgb(u) * (1 - dull) + rgb(v) * dull)
    red, dk, ins, ins2 = m('#c5242c', '#6a4a52'), m('#741420', '#3d3040'), m('#9c1b26', '#55404a'), m('#e0474a', '#7a5a62'); body = np.vstack([spl(R([(-.50, 0), (-.44, .12), (-.30, .215), (-.17, .25)])), R([(-.155, .295)]), spl(R([(-.155, .295), (0, .315), (.155, .295)])), spl(R([(.17, .25), (.30, .215), (.44, .12), (.50, 0)])), spl(R([(.50, 0), (0, .06), (-.50, 0)]))]); C.fill(body, red); C.clip(body); C.fill(spl(R([(.02, .02), (.2, .2), (.12, .34), (.5, .34), (.6, -.05)]), True), dk); C.fill(spl(R([(-.44, .06), (-.36, .14), (-.24, .20), (-.28, .14), (-.36, .08)]), True), ins2); C.unclip(); outline(C, body, lw, '#1d0d10', 1)
    el = np.array([(.5 * math.cos(t), .155 * math.sin(t)) for t in a]); C.fill(R(el), ins); C.clip(R(el)); C.fill(R(el * [.96, .9] + [.05, -.06]), m('#7c1420', '#40303c'))
    if sake > 0:
        lv = .42 + .04 * sake; C.fill(R(el * [lv / .5, lv / .5] + [0, .018]), '#d9c9a8'); C.fill(R(el * [lv / .5 * .9, lv / .5 * .74] + [.02, .035]), '#f6ecd0'); C.fill(R(el * [.34, .26] + [-.16, -.01]), '#ffffff'); C.line(R(el * [lv / .5, lv / .5] + [0, .018]), lw * .5, '#4a2a20', closed=True, taper=(0, 0))
    C.unclip(); C.line(np.vstack([R(el), R(el)[:1]]), lw * 1.5, m(rimc, '#8a7a70'), taper=(0, 0), press=.2, seed=2); C.line(np.vstack([R(el * 1.015), R(el * 1.015)[:1]]), lw * .8, '#1d0d10', taper=(0, 0), seed=3)
    if crack:
        for k, L in enumerate(([(-.06, -.155), (-.02, -.06), (-.07, .02), (.0, .12), (-.03, .19), (.02, .30)], [(-.07, .02), (-.17, .05), (-.22, .11)], [(.0, .12), (.10, .15), (.13, .21)])): C.line(R(L), lw * (1.3 if k == 0 else .8), '#0d0608', taper=(.1, .5), press=.5, seed=5 + k)
        C.fill(R([(.30, -.13), (.38, -.10), (.36, -.02), (.30, -.05), (.25, -.10)]), '#1a1218')


class Shot:
    id = 'd7'; n = 60; finish = dict(bloom=.30, vig=.22)
    EVENTS = [(0, 'forest', dict(note='cicadas, leaves', len=24)), (10, 'clink', dict(size=.9, note='three lacquer cups meet, sake slops')), (14, 'drip', dict(size=.3, note='sake drops on wood')), (24, 'cut_rain', dict(note='rain, everything else gone', len=8)), (32, 'ember', dict(note='paper smoulder, tiny crackle', len=14)), (45, 'ember_out', dict(size=.4, note='last glow dies, one breath of ash')), (52, 'fist', dict(size=.4, note='hand closes on nothing, skin and a held breath'))]

    def setup(self):
        self.x, self.y = grid(); self.day = np.asarray(SC.cached('d7_day', lambda: self.world(False)), f32); self.wet = np.asarray(SC.cached('d7_wet', lambda: self.world(True)), f32); self._h = {}; self._p = {}
        self.pn = fbm(520, 620, (150, 60, 24, 10), (1, .8, .55, .35), 9); self.pf = fbm(520, 620, (9, 4, 2.2), (1, .7, .5), 4); self.pm = fbm(520, 620, (30, 16), (1, .6), 14); self.ml = radial(500, 260, 1500, 1000, 1.)

    # ------------------------------------------------------------ forest, stump
    def world(self, wet):
        x, y = self.x, self.y; r = np.random.default_rng(6); im = ramp(y / H, [(0, '#a8c66a'), (.35, '#6f9c4e'), (.62, '#3e6a3c'), (1, '#27452f')]) if not wet else ramp(y / H, [(0, '#5f7480'), (.35, '#465c66'), (.62, '#2f424c'), (1, '#1f2d36')]); n = fbm(H, W, (120, 46, 18), (1, .7, .5), 3)[..., None]; im = im * (1 + .12 * n); C = Cel()
        for i in range(26):
            z = r.random(); tx = r.uniform(-60, W + 60); tw = 16 + 90 * z ** 2; c = mix3('#8a8f5a', '#33271f', .25 + .75 * z) if not wet else mix3('#5c6c74', '#151b22', .25 + .75 * z); C.fill(np.array([(tx - tw / 2, 700 + 40 * z), (tx - tw * .42 + r.normal(0, 8), -20), (tx + tw * .42, -20), (tx + tw / 2 + 4, 700 + 40 * z)]), c, .55 + .45 * z)
            if not wet: C.fill(np.array([(tx - tw / 2, 700), (tx - tw * .42, -20), (tx - tw * .26, -20), (tx - tw * .34, 700)]), '#e2d68e', .55)
        im = over(im, cv2.GaussianBlur(C.layer(), (0, 0), 6)); cv = Canvas(H, W, img=im); m = 900; px, py = r.uniform(-40, W + 40, m), np.r_[r.uniform(-20, 260, m // 2), r.uniform(560, 760, m - m // 2)]; col = np.stack([mix3('#4d7a3c', '#c2d878', v) if not wet else mix3('#22343a', '#5d7780', v * .7) for v in r.random(m)]).astype(f32); cv.strokes(arcs_(px, py, r.uniform(0, 6.28, m), r.uniform(14, 46, m)), r.uniform(5, 15, m), col, op=.8, dry=.2, seed=4, shape='leaf'); im = cv2.GaussianBlur(cv.img, (0, 0), 4.5)
        if not wet:
            for i in range(60): bx, by, br = r.uniform(0, W), r.uniform(-20, 520), r.uniform(14, 52); d = np.hypot(x - bx, y - by); im = screen(im, (smooth(br, br - 3, d) * (.55 + .45 * smooth(br * .6, br, d)) * r.uniform(.10, .34))[..., None] * rgb('#f4f0a8'))
            for a, b, w, k in ((200, 700, 120, .30), (560, 1150, 160, .24), (1150, 1560, 90, .18)): bm = np.exp(-((x - a - (b - a) * (y / 820.)) / w) ** 2); im = screen(im, (bm * k * (1 - smooth(600, 900, y)))[..., None] * rgb('#fff0a8'))
        # the stump
        S = Cel(); ln = '#1b1310'; cx, cy, rx, ry = 960, 792, 760, 178; a = np.linspace(0, 2 * math.pi, 90)[:-1]; wb = 1 + .018 * np.sin(5 * a + 1) + .012 * np.sin(11 * a); top = np.c_[cx + rx * np.cos(a) * wb, cy + ry * np.sin(a) * wb]; bark, bark2, wood, wood2 = ('#5d4330', '#3b2a22', '#d3a566', '#a9763f') if not wet else ('#3f3c40', '#26262c', '#7c7878', '#585860')
        side = np.vstack([top[(a > -.1) & (a < math.pi + .1)], A((cx - rx - 70, 1140), (cx + rx + 70, 1140))]); S.fill(side, bark); S.clip(side)
        for i in range(90): u = r.uniform(-1, 1); x0 = cx + rx * u * 1.03; y0 = cy + ry * math.sqrt(max(0, 1 - u * u)); S.line(spl(A((x0, y0 + r.uniform(0, 30)), (x0 + r.normal(0, 8) + 20 * u, y0 + 120), (x0 + r.normal(0, 12) + 50 * u, 1130)), n=8), r.uniform(4, 18), bark2 if i % 3 else mix3(bark, '#ffffff', .12), taper=(.2, .3), press=.5, seed=i)
        S.unclip(); S.fill(top, wood2); S.fill(np.c_[cx + (rx - 26) * np.cos(a) * wb, cy - 3 + (ry - 12) * np.sin(a) * wb], wood); S.clip(top)
        for i in range(17): q = (i + .6) / 17.; w_ = 1 + .03 * np.sin(3 * a + i * .4) + .02 * np.sin(7 * a + i); rr = np.c_[cx + 60 + (rx - 30) * q * np.cos(a) * w_ - 60 * q, cy - 6 + (ry - 14) * q * np.sin(a) * w_ + 6 * q]; S.line(np.vstack([rr, rr[:1]]), 2.0 + 2.2 * (i % 3 == 0), wood2, taper=(0, 0), press=.6, seed=i)
        for i in range(9): th = r.uniform(0, 6.28); q0, q1 = r.uniform(.05, .4), r.uniform(.6, 1.); S.line(spl(A((cx + rx * q0 * math.cos(th), cy + ry * q0 * math.sin(th)), (cx + rx * (q0 + q1) / 2 * math.cos(th + .03), cy + ry * (q0 + q1) / 2 * math.sin(th + .03)), (cx + rx * q1 * math.cos(th), cy + ry * q1 * math.sin(th))), n=8), r.uniform(3, 7), ln, taper=(.5, .2), press=.4, seed=40 + i, alpha=.75)
        S.fill(np.c_[cx + 330 + rx * .95 * np.cos(a), cy + 60 + ry * .9 * np.sin(a)], bark2, .28); S.unclip(); S.line(np.vstack([top, top[:1]]), 6., ln, taper=(0, 0), press=.3, seed=80)
        for i in range(150): u = r.uniform(-1.06, 1.06); y0 = cy + ry * math.sqrt(max(0, 1 - min(1, u * u))) + r.uniform(60, 260); x0 = cx + rx * u * 1.04; g = ('#4f7a35', '#7fa54a', '#2f5230') if not wet else ('#34474a', '#4e6468', '#22322f'); S.line(spl(A((x0, y0), (x0 + r.normal(0, 8), y0 - 22), (x0 + r.normal(0, 18), y0 - r.uniform(30, 60))), n=6), r.uniform(4, 9), g[int(r.integers(3))], taper=(.1, .8), seed=100 + i)
        L = S.layer(); L = para(L, (300, 600), (1650, 1080), [(0, '#fff2c0'), (.5, '#e9d2b4'), (1, '#7d7a80')] if not wet else [(0, '#dfe9ee'), (1, '#7a8894')]); L = rim(L, 2, 5, '#fff0b0' if not wet else '#cfe0ea', .6, 1.2); im = over(im, L); lab = repaint(im, (14, 7), seed=4, density=.9, op=.7); im = im * .55 + lab * .45
        return np.clip(im, 0, 1.3)

    # ------------------------------------------------------------ hands
    def hand(self, key, fn):
        if key not in self._h: self._h[key] = fn()
        return self._h[key]

    def kid_hand(self, who, x, y, rot, tilt, sake):
        """one hand with its cup, as a layer.  who 0 left, 1 right, 2 back"""
        pal = dict(HD.PAL['day'], skin=('#f0b98d', '#e8ac7c', '#f2c49c')[who], shade=('#c07a58', '#b06a48', '#c48862')[who], line='#1d1210', cuff='#2a3f70', cuff_shade='#1a2748', cuff_light='#5a78b0', band='#c43a2a')
        key = (who, round(rot, 2))
        def mk():
            c, s = math.cos(rot), math.sin(rot); tl = .42; vert = A(0, math.cos(tl), math.sin(tl))
            if who == 0: y_, zb, side = A(c, s, -.16), -vert, 'R'
            elif who == 1: y_, zb, side = A(-c, s, -.16), -vert, 'L'
            else: y_, zb, side = A(.10, -.50, .86), A(0, -.82, -.50), 'R'
            style = dict(cuff=dict(kind='uniform', y=.34), arm=5.) if who == 2 else dict(bands=((.42, .52),), arm=6.) if who == 1 else dict(arm=6.)
            h = HD.Hand(CRADLE, side, frame_of(y_, zb), (0, 0, 0), KIDB, pivot=(0, .75, 0), **style); cam = HD.Cam(0, 0, 236, 6.5 if who < 2 else 8.); g = HD.render([h], cam, (-.5, .8, .35), res=560, clip=(-1200, -700, 1200, 700)); C = Cel(2400, 1400, -1200, -700); HD.ink(C, g, pal, 5.2, seed=3 + who); T = Cel(2400, 1400, -1200, -700)
            for G in g.groups:
                if G['name'] == 'thumb' and who != 1: [T.fill(p, '#ffffff') for p in G['polys']]
            lay = C.rgba(); lay = sleeve_up(lay, g) if who == 2 else lay; lay = para_roi(lay, [(0, '#fff6dc'), (.5, '#f6d8b8'), (1, '#b88a80')]); lay = rim(lay, 4, 7, '#ffe9a6', .9, .8); pts = g.pts[0] if isinstance(g.pts, list) else g.pts[list(g.pts)[0]]; return lay, lay * T.rgba()[..., 3:4], np.array(pts['palm'][:2], float)
        lay, th, pm = self.hand(key, mk); out = np.zeros((H, W, 4), f32); ox, oy = int(round(x - pm[0])) - 1200, int(round(y + 58 - pm[1])) - 700; put(out, lay, ox, oy); C = Cel(); cup(C, x, y - 46, 400 if who != 2 else 350, tilt, sake, lw=5.); CL_ = C.layer(); CL_ = rim(CL_, 3, 5, '#ffe9a6', .85, .8); out = over_l(out, CL_); put(out, th, ox, oy); return out

    # ------------------------------------------------------------ the burning scrap
    def scrap(self, img, f, quad, glow):
        """paper (or ash) in the palm: quad = its 4 corners on screen"""
        h, w = self.pn.shape; u, v = np.meshgrid(np.linspace(0, 1, w), np.linspace(0, 1, h)); n = self.pn * .070 + self.pm * .010; B = np.hypot((u - 1.12) / 1.15, (v - .86) / 1.05); T = float(lerp_keys(f, [(30, .34), (40, .80), (45, 1.08), (47, 1.20), (60, 1.22)], lambda t: t)) + .012 * math.sin(f * 1.9)
        s = B + n - T                                                # >0 paper, 0 front, <0 burnt
        ed = np.minimum(np.minimum(u, 1 - u), np.minimum(v, 1 - v)) + self.pn * .010 + cv2.GaussianBlur(self.pf, (0, 0), 2.5) * .006; pa = smooth(.030, .036, ed)
        paper = np.full((h, w, 3), rgb('#e9dcc2'), f32); paper *= (1 + .02 * cv2.GaussianBlur(self.pf, (0, 0), 2))[..., None]; paper *= (1 - .75 * (smooth(.030, .034, ed) * (1 - smooth(.040, .046, ed))))[..., None]; fold = np.exp(-((u - .47 - .02 * np.sin(v * 5)) / .006) ** 2) * .25; paper *= (1 - fold)[..., None]; br = np.clip(1 - s / .20, 0, 1)[..., None]; paper = paper * (1 - br * .55) + rgb('#8a5426') * br * .55; ch = smooth(.075, .0, s)[..., None]; paper = paper * (1 - ch) + rgb('#1c1110') * ch
        hot = np.exp(-((s + .006) / .012) ** 2) * (.55 + .45 * np.clip(FCn(u * 26, v * 26, f * .30), -1, 1)); hot2 = np.exp(-((s + .012) / .026) ** 2); crawl = np.clip(FCn(u * 60, v * 60, f * .16) - .45, 0, 1) * smooth(-.07, -.005, s) * (s < 0)
        life = float(lerp_keys(f, [(30, 1), (42, 1), (45, .85), (47, .0), (60, 0)]))
        age = np.clip(-s / .22, 0, 2); lace = smooth(.30, .38, .72 - .80 * age + .22 * self.pn + .20 * self.pm); ashc = np.full((h, w, 3), rgb('#1f1a1c'), f32) + (np.clip(age, 0, 1) * .16 + .03 * self.pm * (age > .1))[..., None]
        al = np.where(s > 0, 1., lace) * pa * np.where(s < 0, 1 - smooth(.85, 1.3, age), 1.); rgb_ = np.where((s > 0)[..., None], paper, ashc); em = (hot[..., None] * rgb('#ffb23c') * 1.5 + (np.exp(-((s + .004) / .007) ** 2) * np.clip(hot * 1.6, 0, 1))[..., None] * rgb('#fff2c0') * 1.2 + hot2[..., None] * rgb('#e83c10') * .60 + crawl[..., None] * rgb('#ff7a2a') * .7) * life * pa[..., None]
        M = cv2.getPerspectiveTransform(np.float32([(0, 0), (w, 0), (w, h), (0, h)]), np.float32(quad)); wp = lambda a: cv2.warpPerspective(a.astype(f32), M, (W, H), flags=cv2.INTER_AREA, borderMode=cv2.BORDER_CONSTANT)
        A_ = wp(al)[..., None]; C_ = wp(rgb_ * al[..., None]); E_ = wp(em); sh = cv2.GaussianBlur(cv2.warpAffine(A_[..., 0], np.float32([[1, 0, 10], [0, 1, 16]]), (W, H)), (0, 0), 7)[..., None]; img = img * (1 - sh * .55); litp = np.clip(.34 + glow * .46, 0, 1.); gm = min(1., glow * 1.15); img = img * (1 - A_) + C_ * litp * (rgb('#ffcf98') * 1.10 * gm + rgb('#9fb2e6') * (1 - gm))
        # drawn edge of the paper
        img = img + E_ + blur(E_, 8) * .45 + blur(E_, 34) * .55; return img, (A_[..., 0], T)

    def palm(self, f, close, glow, tr):
        key = ('palm', round(close, 2))
        def mk():
            a, b = HD.POSES['palm_up'], HD.pose((64, 78, 46, 0), (68, 82, 48, 0), (72, 82, 48, 0), (74, 80, 46, 2), thumb=(26, 52, -20, 24, 34), cup=.5); ps = HD.lerp_pose(a, b, close) if close > 0 else dict(a); R = frame_of((.34, .50, .62), (-.10, -.80, .55) if False else (.02, -.74, .62)) if False else HD.euler(172, -44, 18)
            h = HD.Hand(ps, 'R', R, (0, 0, 0), HD.MALE, pivot=(0, .70, 0)); cam = HD.Cam(1000, 590, 610, 5.5); g = HD.render([h], cam, (-.35, .30, .95), res=640, clip=(0, 0, W, H), thr=.02); pal = dict(HD.PAL['day'], skin='#d89a70', shade='#8c5444', line='#160d10', nail='#e6b08c', nail_shade='#a8705c'); C = Cel(); HD.ink(C, g, pal, 7.0, seed=5); pts = g.pts[0] if isinstance(g.pts, list) else g.pts[list(g.pts)[0]]; return C.layer(), np.array(pts['palm'][:2], float)
        return self.hand(key, mk)

    def frame(self, f):
        if f < 24: return self.f_cups(f)
        if f < 32: return self.f_rain(f)
        return self.f_palm(f)

    def f_cups(self, f):
        t = f if 6 <= f <= 15 else f // 2 * 2; z = 1 + .0012 * f; img = cam2d(self.day, W / 2, H / 2 + 6 - f * .25, z); k = float(lerp_keys(t, [(0, 1.), (5, 1.16), (6, 1.14), (9, .08), (10, -.02), (12, .10), (15, .035), (23, .055)], lambda q: ease_in(q, 1.6) if False else ease(q)))
        kick = math.exp(-(t - 10) / 2.6) if t >= 10 else 0.; jit = lambda i: A(math.sin(t * 2.6 + i * 2) * 7 * kick, math.cos(t * 2.2 + i) * 6 * kick); bob = lambda i: 3 * math.sin(t * .22 + i * 2.1)
        pos = [(A(764, 560) + A(-560, 150) * k + jit(0) + A(0, bob(0)), .08 + .10 * k, .20 * k - .16 * kick), (A(1158, 566) + A(560, 140) * k + jit(1) + A(0, bob(1)), .06 + .10 * k, -.20 * k + .16 * kick), (A(962, 398) + A(20, -420) * k + jit(2) + A(0, bob(2)), 0., -.06 * math.sin(t * 2.1) * kick)]
        for who in (2, 0, 1):
            p, rot, tilt = pos[who]; L = self.kid_hand(who, p[0], p[1], round(rot * 10) / 10., tilt, 1.); img = over(img, np.clip(shadow(L, 30, 150, 16), 0, 1)) if False else img * (1 - blur(cv2.warpAffine(L[..., 3], np.float32([[1, 0, 24], [0, 1, 210]]), (W, H)), 22)[..., None] * .22 * smooth(640, 720, self.y)[..., None]); img = over(img, L)
        if t >= 10:
            C = Cel(); r = np.random.default_rng(5); q = (t - 10) / 24.
            for i in range(26):
                a = r.uniform(-2.9, -.25); v = r.uniform(240, 760); p = A(960 + r.normal(0, 60), 440) + A(math.cos(a), math.sin(a)) * v * q + A(0, 1500 * q * q); R = r.uniform(4, 12) * (1 - .3 * q); vel = A(math.cos(a), math.sin(a)) * v + A(0, 3000 * q); d = vel / (np.hypot(*vel) + 1e-6); n = A(-d[1], d[0])
                if p[1] < 830: C.fill(spl([p + d * R * 1.1, p + n * R * .8, p - d * R * 2.2, p - n * R * .8], True, (2,)), '#f3e6c2'); C.fill(np.array([p + n * R * .4, p + d * R * .4, p - d * R]), '#ffffff'); outline(C, spl([p + d * R * 1.1, p + n * R * .8, p - d * R * 2.2, p - n * R * .8], True, (2,)), 1.8, '#4a2a1a', i)
            if t <= 13:
                for i in range(14): a = i / 14. * 2 * math.pi + .2; r0, r1 = 110 + 46 * (t - 10), 150 + 70 * (t - 10) + 40 * (i % 2); C.fill(np.array([A(960 + math.cos(a) * r0, 452 + math.sin(a) * r0 * .6), A(960 + math.cos(a + .022) * (r0 + r1) / 2, 452 + math.sin(a + .022) * (r0 + r1) / 2 * .6), A(960 + math.cos(a) * r1, 452 + math.sin(a) * r1 * .6), A(960 + math.cos(a - .022) * (r0 + r1) / 2, 452 + math.sin(a - .022) * (r0 + r1) / 2 * .6)]), '#fff6d0', .9)
            img = over(img, C.layer())
        if t == 10: img = screen(img, radial(960, 440, 420, 300, 2.)[..., None] * rgb('#fff3c0') * .8)
        return img + motes(f, 60, seed=8, light=self.ml, col='#fff4b0', drift=(10, -8), size=1.4, amt=.7)

    def f_rain(self, f):
        img = self.wet.copy(); C = Cel(); t = f // 2 * 2
        cup(C, 800, 716, 300, .02, 0., dull=.35); cup(C, 1176, 742, 300, -.10, 0., crack=1., dull=.45); a = np.linspace(0, 2 * math.pi, 50); C.line(np.c_[985 + 74 * np.cos(a), 690 + 18 * np.sin(a)], 5., '#3a3a42', taper=(0, 0), press=.8, seed=4, alpha=.7); L = C.layer(); L = para(L, (500, 500), (1400, 900), [(0, '#e6eef2'), (1, '#7a8896')]); L = rim(L, 0, 5, '#dbe8f0', .7, 1.)
        img = img * (1 - blur(cv2.warpAffine(L[..., 3], np.float32([[1, 0, 14], [0, 1, 30]]), (W, H)), 12)[..., None] * .45); img = over(img, L); S = Cel()
        for i in range(34): u, v, p = FC.h1(3, i, 1), FC.h1(3, i, 2), int(FC.h1(3, i, 3) * 9); splash(S, 960 + (u * 2 - 1) * 700 * math.sqrt(max(0, 1 - (v * 2 - 1) ** 2)), 790 + (v * 2 - 1) * 168, 12 + 12 * v, (f + p) % 9, seed=i, col='#dce8f0', shade='#5f7380')
        for i in range(6): x0 = 800 + (i - 2.5) * 46 if i < 3 else 1176 + (i - 4) * 60; splash(S, x0, 700 + 8 * (i % 2), 11, (f + i * 3) % 8, seed=60 + i, col='#eef4f8', shade='#5f7380')
        img = over(img, S.layer()); img = over(img, rain(f, n=(300, 130, 22), wind=-.06, seed=9, alpha=.85, col='#cddce6')); return img * (1 - .10 * smooth(300, 0, self.y)[..., None])

    def f_palm(self, f):
        t = f // 2 * 2; x, y = self.x, self.y; n = fbm(H, W, (260, 100, 40), None, 12)[..., None]; img = ramp(y / H, [(0, '#0d0d16'), (.6, '#1a1820'), (1, '#231a1c')]) * (1 + .18 * n)
        glow = float(lerp_keys(f, [(32, .95), (40, .85), (44, .62), (45, .70), (47, .06), (49, .0), (60, 0.)])) * (1 + .10 * math.sin(f * 2.1) * (f < 46)); cl = float(lerp_keys(f, [(47, 0.), (49, .04), (52, 1.0), (54, .92), (56, 1.0), (59, 1.0)])); cl = round(cl * 20) / 20.; tr = A(math.sin(f * 2.7), math.cos(f * 3.1)) * (1.2 if f < 50 else 2.4); far = blur(radial(1060, 520, 900, 700, 1.6), 30)[..., None]; img = img + far * rgb('#ff8a3a') * .10 * glow + far * rgb('#5a6a90') * .10
        L, pm = self.palm(f, cl, glow, tr); L = np.roll(L, (int(round(tr[1])), int(round(tr[0]))), (0, 1)); lum = (L[..., 0] * .35 + L[..., 1] * .45 + L[..., 2] * .2)[..., None]; cold = L.copy(); cold[..., :3] = lum * rgb('#8fa8e6') * .92 + L[..., :3] * .10; d = np.sqrt((x - 1000) ** 2 + ((y - 610) * 1.05) ** 2); g = min(1.15, glow * 1.22); wm = (np.clip(1.22 - (d / 900) ** 1.6, .0, 1.) * min(1, g))[..., None]; hotc = np.exp(-(d / 300) ** 2)[..., None]
        warm = L[..., :3] * (rgb('#f99a58') * (1 - hotc) + rgb('#ffc284') * hotc) * (1.02 + .34 * hotc); warm = warm + L[..., 3:4] * rgb('#ff5a14') * .07; lay = cold.copy(); lay[..., :3] = cold[..., :3] * (1 - wm) + warm * wm * (.80 + .20 * g); e_ = (y < 1050).astype(f32); lay = rim(lay, 6, 7, '#c9d6ff', .85 * (1 - .55 * min(1, g)), .8, where=e_); lay = rim(lay, -5, -7, '#ffd08a', .9 * min(1, g), 2., where=e_); img = img + far * rgb('#ff7a2a') * .20 * g
        img = over(img, lay)
        if cl < .10:
            q = [A(770, 452), A(1150, 478), A(1210, 742), A(752, 728)]; dr = 4 * math.sin(f * .25); q = [p + tr + A(0, dr * (i % 2) * .5) for i, p in enumerate(q)]; img, _ = self.scrap(img, f, q, glow)
        # smoke, ash
        S = Cel(ss=1); sm = float(lerp_keys(f, [(32, .7), (44, 1.), (50, .9), (59, .5)]))
        for i in range(5):
            u = np.linspace(0, 1, 40); ph = f / 24. * 1.3 + i * 1.9; bx = 1030 + 60 * math.sin(i * 2.3) - 10 * i; X = bx + (30 + 150 * u) * np.sin(u * (4.2 + i * .4) - ph * 2 + i) * u ** .8 + 60 * u * u; Y = 580 - u * (560 + 60 * i); S.line(np.c_[X, Y], 26 + 10 * i, '#9aa0b4', taper=(.1, .8), press=.5, seed=70 + i, alpha=.20 * sm, wfun=lambda q_: .35 + 1.6 * q_)
        sl = cv2.GaussianBlur(S.layer(), (0, 0), 9); sl[..., :3] = sl[..., :3] + sl[..., 3:4] * rgb('#ff9a4a') * .5 * glow * np.exp(-((y - 560) / 160) ** 2)[..., None]; img = over(img, sl); A_ = Cel()
        for i in range(34):
            h = lambda k: FC.h1(11, i, k); b = 32 + 17 * h(1); age = (f - b) / 24.
            if age < 0 or age > .95: continue
            p = A(850 + 300 * h(2), 560 + 90 * h(3)) + A((110 + 120 * h(4)) * age + 34 * math.sin(age * 4 + i), -(260 + 420 * h(5)) * age); s_ = 3 + 8 * h(6); an = age * 5 * (h(7) - .5) + i; sq = abs(math.sin(age * 5 + i)) * .8 + .2; P = np.array([(math.cos(a_ + an) * s_, math.sin(a_ + an) * s_ * sq) for a_ in (0, 1.3, 2.6, 4.2, 5.2)]) + p; A_.fill(P, '#575059', 1 - .8 * age); 
            if age < .35: A_.fill((P - p) * .55 + p, '#ff9a3a', (1 - age / .35) * min(1, glow + .2))
        img = over(img, A_.layer()); return img


def sleeve_up(L, g):
    """carry the blue sleeve on up the arm, above its cuff"""
    h, w = L.shape[:2]; c = L[..., :3]; blue = (c[..., 2] > c[..., 0] + .06) & (L[..., 3] > .9); ys = np.nonzero(blue.any(1))[0]
    if not len(ys): return L
    cut = int(ys.min() + (ys.max() - ys.min()) * .35); o = L.copy(); m = (np.arange(h)[:, None] < cut) & (c[..., 0] > c[..., 2] + .10); t = np.where(c[..., 1] > .55, 1., .62)[..., None]; o[..., :3] = np.where(m[..., None], rgb('#2a3f70') * t * L[..., 3:4], c); return o


def para_roi(L, stops):
    h, w = L.shape[:2]; x, y = np.meshgrid(np.arange(w, dtype=f32), np.arange(h, dtype=f32)); t = np.clip((x / w * .4 + y / h * .9) - .15, 0, 1); g = ramp(t, stops); o = L.copy(); o[..., :3] *= g; return o


def put(dst, src, ox, oy):
    h, w = dst.shape[:2]; sh, sw = src.shape[:2]; x0, y0, x1, y1 = max(0, ox), max(0, oy), min(w, ox + sw), min(h, oy + sh)
    if x1 > x0 and y1 > y0: s = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; d = dst[y0:y1, x0:x1]; dst[y0:y1, x0:x1] = d * (1 - s[..., 3:4]) + s


def mix3(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, f32); b = rgb(b) if isinstance(b, str) else np.asarray(b, f32); return a * (1 - t) + b * t


def arcs_(x, y, a, l):
    from op.paint import arcs
    return arcs(x, y, a, l, .4, 5)


def FCn(u, v, t):
    """cheap animated value noise in -1..1 on an image grid"""
    return np.sin(u * 1.7 + 2.1 * np.sin(v * .9 + t * 1.3)) * np.sin(v * 1.9 + 1.7 * np.sin(u * .8 - t)) * .7 + .4 * np.sin(u * 3.1 - v * 2.7 + t * 2.)
