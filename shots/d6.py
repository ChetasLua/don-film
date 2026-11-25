"""d6  Nothing happened.  One brush picture: dawn ruins in black, bone and red; he is still standing.  A dropped cigarette.  Only steam moves."""
from chars.swkit import *
from op import sumi
BK = '#0d0a0c'; BONE = '#ecdfc6'; RED = '#b3131b'


class Shot:
    id = 'd6'; n = 30; finish = dict(bloom=.16, vig=.20)
    EVENTS = [(0, 'silence', dict(note='wind, one far drip')), (11, 'cigarette', dict(size=.15, note='falls on stone, tiny hiss')), (13, 'tick', dict(size=.08))]

    def paint_(self):
        x, y = grid(); g = fbm(H, W, (320, 110, 36, 10), None, 3); im = rgb(BONE) * (1 + .03 * g[..., None]) + fbm(H, W, (3, 1.5), (1, .6), 4)[..., None] * .016; hz = 706
        sun = (1130, 600); d = np.hypot((x - sun[0]), (y - sun[1])); im = im * (1 - .0) ; w = smooth(1250, 120, d) * .34; im = im * (1 - w[..., None]) + rgb('#f7ecd2') * w[..., None]
        pg = sumi.Page(seed=3); r = np.random.default_rng(2)                  # sky: dry sweeps of grey and one of red
        wash = np.zeros((H, W), f32)
        for k in range(9): P = [(r.uniform(-200, 300), 60 + 52 * k + r.normal(0, 14)), (700, 40 + 54 * k + r.normal(0, 26)), (1300, 50 + 50 * k + r.normal(0, 26)), (r.uniform(1700, 2200), 40 + 52 * k)]; L = drybrush_(P, 34 + 20 * r.random(), k + 11, .78); wash = np.maximum(wash, L * (.75 - .06 * k))
        im = im * (1 - .30 * wash[..., None]) + rgb('#5b4c48') * (.30 * wash)[..., None]
        for k, (yy, wd, al) in enumerate(((560, 30, .85), (610, 44, .95), (654, 26, .9), (520, 16, .6))): P = [(-60, yy + 10), (520, yy - 8), (1120, yy - 12), (1990, yy + 6)]; L = drybrush_(P, wd, 40 + k, .55); im = im * (1 - al * .9 * L[..., None]) + rgb(RED) * (al * .9 * L)[..., None]
        m = smooth(122, 118, d) * (y < hz); im = im * (1 - m[..., None]) + rgb('#fbf3dd') * m[..., None]; ring = np.exp(-((d - 121) / 3.) ** 2) * (y < hz) * (.5 + .5 * smooth(-.5, .6, g)); im = im * (1 - .7 * ring[..., None]) + rgb('#8e0f18') * (.7 * ring)[..., None]
        # red lake of light
        gm = smooth(hz - 2, hz + 2, y + 5 * fbm(H, W, (90, 24), None, 8, aspect=6.)); t = np.clip((y - hz) / (H - hz), 0, 1); lake = ramp(t, [(0, '#e0402e'), (.15, '#c91d20'), (.55, '#9c0f1a'), (1, '#5d0913')]); lake = lake * (1 + .07 * fbm(H, W, (140, 36, 9), None, 9, aspect=7.)[..., None]); im = im * (1 - gm[..., None]) + lake * gm[..., None]
        c = Cel()
        for k in range(150):
            v = r.uniform(0, 1) ** 1.3; yy = hz + 8 + v * 366; wide = 30 + 300 * v; xx = sun[0] + r.normal(0, 34 + 90 * v) if k < 90 else r.uniform(150, 1800); L_ = (10 + 130 * v) * r.uniform(.3, 1.1); c.line(np.array([(xx - L_, yy), (xx - L_ * .2, yy + r.normal(0, 1.5)), (xx + L_, yy + r.normal(0, 2))]), 1.4 + 6 * v * r.uniform(.4, 1), '#f8d8b4' if k < 70 else ('#ff7a5c' if k % 2 else '#4a0710'), taper=(.35, .5), press=.7, alpha=.9, seed=k)
        im = over(im, c.layer())
        # ---- ruins, all one black, edges brushed
        c = Cel(); R = lambda P: c.fill(np.array(P, float), '#ffffff')
        def wall(P, seed, j=2.6):
            rr = np.random.default_rng(seed); Q = resample(np.array(P, float), 22); Q = Q + rr.normal(0, j, Q.shape) * (np.hypot(*(Q - Q[0]).T) > 1)[:, None]; c.fill(Q, '#ffffff')
        wall([(-60, 1120), (-60, 90), (70, 60), (120, 150), (170, 130), (196, 260), (240, 280), (250, 420), (300, 470), (310, 640), (380, 700), (470, 760), (560, 800), (300, 840), (200, 1120)], 1)
        wall([(330, 720), (340, 300), (322, 250), (402, 236), (420, 300), (414, 520), (424, 740)], 2, 2); wall([(318, 252), (312, 226), (414, 214), (422, 246)], 3, 1.5)
        wall([(520, 720), (526, 480), (560, 430), (592, 470), (598, 724)], 4, 2); wall([(640, 716), (648, 600), (690, 572), (704, 640), (716, 716)], 5, 2.5)
        wall([(1980, 1120), (1980, 30), (1880, 70), (1850, 190), (1800, 170), (1790, 300), (1730, 340), (1742, 470), (1690, 560), (1650, 640), (1560, 700), (1500, 742), (1700, 800), (1800, 1120)], 6)
        wall([(1480, 716), (1486, 560), (1520, 520), (1556, 570), (1560, 716)], 7, 2); wall([(1330, 714), (1340, 660), (1400, 640), (1440, 680), (1452, 714)], 8, 2.5)
        mk = c.rgba()[..., 0]; c = Cel()                                     # pointed window left standing on both sides
        for q in (spl([(60, 420), (64, 290), (104, 220), (142, 300), (146, 430), (150, 560), (62, 560)], True, (2, 5, 6), n=6), spl([(1836, 430), (1840, 320), (1876, 250), (1912, 330), (1916, 450), (1916, 540), (1836, 540)], True, (2, 5, 6), n=6)): c.fill(q, '#ffffff')
        win = c.rgba()[..., 0]; mk = np.clip(mk - win, 0, 1); ink = sumi.inkify(mk, .45, 5, sig=1.1, warp=1.4)
        im = sumi.lay(im, ink, col=BK)
        c = Cel(); rr = np.random.default_rng(9)                             # bone scratches: broken light on stone
        for row in range(26):                                                # courses of stone catch the light in broken bone lines
            yy = 110 + row * 38 + rr.normal(0, 3)
            for xx in np.arange(-20, W, 74) + (37 if row % 2 else 0):
                if not (xx < 470 or xx > 1480): continue
                px = xx + rr.normal(0, 6)
                if rr.random() < .42 and mk[int(np.clip(yy, 0, H - 1)), int(np.clip(px + 25, 0, W - 1))] > .9: c.line(np.array([(px, yy), (px + 30, yy + rr.normal(0, 2)), (px + rr.uniform(40, 66), yy + rr.normal(0, 2))]), rr.uniform(1.4, 4.), BONE if rr.random() < .78 else '#c42a28', taper=(.1, .6), press=.7, alpha=.75, seed=int(px) + row)
                if rr.random() < .20 and mk[int(np.clip(yy + 18, 0, H - 1)), int(np.clip(px, 0, W - 1))] > .9: c.line(np.array([(px, yy), (px + rr.normal(0, 1.5), yy + rr.uniform(16, 34))]), rr.uniform(1.3, 3.), BONE, taper=(.1, .6), alpha=.7, seed=int(px) + 99)
        for q in ([(404, 262), (412, 420), (410, 700)], [(384, 264), (390, 400), (388, 690)], [(590, 474), (594, 600), (592, 716)], [(566, 440), (572, 560), (570, 700)], [(310, 640), (380, 700), (470, 760), (556, 800)], [(250, 420), (300, 470), (312, 636)], [(196, 262), (244, 284)], [(1850, 192), (1802, 172)], [(1788, 302), (1732, 342), (1742, 468), (1692, 560), (1652, 640), (1562, 700)], [(1486, 562), (1520, 524)]): c.line(spl(q, n=8), 5., BONE, taper=(.1, .5), press=.7, seed=int(q[0][0]))
        im = over(im, c.layer()); return np.clip(im, 0, 1.2)

    def man(self):
        """he stands: arms crossed, chin level, three swords; a woodcut lit by the sun at his left shoulder"""
        from chars import zoro
        c = Cel(); h = 92.; X0, Y0 = 1000, 216; P = lambda q: np.asarray(q, float) * h + [X0, Y0]; S = lambda q, cor=(), n=8: spl(P(q), True, cor, n)
        def part(Q, rim=8., extra=(), lines=(), seed=0):
            c.fill(Q, BK); c.clip(Q); c.fill([Q, Q + [-rim, -rim * .25]], BONE, evenodd=True)
            for e in extra: c.fill(e, BONE)
            for q, w in lines: c.line(spl(P(q), n=8), w, BONE, taper=(.1, .6), press=.6, seed=seed + int(w * 7))
            c.unclip(); outline(c, Q, 2.6, BK, seed)
        L = lambda q: S(q)
        # scabbards at the right hip (screen left)
        for j, (a, b) in enumerate((((-.42, 4.10), (-2.15, 6.35)), ((-.34, 4.22), (-2.28, 5.75)), ((-.26, 4.34), (-2.30, 5.20)))):
            A, B = P([a, b]); d = (B - A) / np.hypot(*(B - A)); m = A_(-d[1], d[0]); part(np.array([A + m * 11, B + m * 8, B + d * 10, B - m * 9, A - m * 11]), 4.5, [np.array([A + d * 22 + m * 12, A + d * 30 + m * 12, A + d * 30 - m * 12, A + d * 22 - m * 12])], seed=j)
        for j, (a, b) in enumerate((((-.30, 4.05), (.62, 2.98)), ((-.24, 4.16), (.80, 3.28)), ((-.16, 4.30), (.94, 3.62)))):      # hilts cross in front of the sash
            pass
        # legs planted wide
        for sg in (-1, 1): part(S([(sg * .04, 4.30), (sg * .66, 4.25), (sg * .88, 5.1), (sg * .86, 5.9), (sg * .98, 6.6), (sg * .92, 7.45), (sg * .50, 7.50), (sg * .46, 6.7), (sg * .34, 5.8), (sg * .10, 4.9)], (0, 1, 5, 6)), 9 if sg > 0 else 5, [], [([(sg * .30, 5.75), (sg * .55, 5.95), (sg * .84, 5.88)], 4.), ([(sg * .36, 6.05), (sg * .6, 6.18)], 3.), ([(sg * .2, 4.8), (sg * .5, 5.2), (sg * .62, 5.6)], 3.4), ([(sg * .56, 6.7), (sg * .70, 7.1), (sg * .68, 7.4)], 3.)], 10 + sg)
        for sg in (-1, 1): part(S([(sg * .44, 7.30), (sg * .96, 7.28), (sg * 1.02, 7.62), (sg * 1.26, 7.86), (sg * 1.24, 8.0), (sg * .42, 8.0)], (0, 1, 4, 5)), 8 if sg > 0 else 4, [], [([(sg * .46, 7.42), (sg * .7, 7.48), (sg * .98, 7.40)], 3.4)], 20 + sg)
        # torso: wide shoulders, narrow waist, torn shirt gone
        part(S([(-.28, 1.18), (-.62, 1.36), (-1.06, 1.50), (-1.10, 1.95), (-.82, 2.7), (-.62, 3.5), (-.58, 4.0), (.58, 4.0), (.62, 3.5), (.82, 2.7), (1.10, 1.95), (1.06, 1.50), (.62, 1.36), (.28, 1.18)], (6, 7)), 9, [S([(.36, 1.72), (.80, 1.66), (.92, 1.95), (.70, 2.14), (.44, 2.10)]), S([(.20, 3.16), (.52, 3.05), (.50, 3.35), (.22, 3.42)]), S([(.18, 3.52), (.48, 3.44), (.46, 3.72), (.20, 3.76)])], [([(-.80, 1.52), (-.40, 1.58), (-.06, 1.66)], 4.), ([(.0, 1.70), (-.02, 2.0), (.0, 2.3)], 4.), ([(0, 3.0), (-.02, 3.4), (0, 3.9)], 3.4), ([(-.46, 3.1), (-.24, 3.2), (-.06, 3.16)], 3.), ([(-.44, 3.46), (-.24, 3.54), (-.06, 3.5)], 3.)], 30)
        part(S([(-.64, 3.72), (.64, 3.72), (.72, 4.42), (-.72, 4.42)], (0, 1, 2, 3)), 8, [], [([(-.66, 3.92), (0, 3.98), (.66, 3.92)], 3.6), ([(-.68, 4.10), (0, 4.17), (.68, 4.10)], 3.6), ([(-.70, 4.27), (0, 4.34), (.70, 4.27)], 3.6)], 33)
        for j, (a, b) in enumerate((((-.60, 4.02), (-.16, 3.42)), ((-.52, 4.16), (.02, 3.62)), ((-.44, 4.30), (.14, 3.84)))):
            A, B = P([a, b]); d = (B - A) / np.hypot(*(B - A)); m = A_(-d[1], d[0]); part(np.array([A + m * 9, B + m * 9, B + d * 7, B - m * 9, A - m * 9]), 5, [np.array([A + d * k * 13 + m * 9, A + d * (k * 13 + 5) - m * 9, A + d * (k * 13 + 8) - m * 9, A + d * (k * 13 + 3) + m * 9]) for k in range(4)], seed=40 + j); g = A + d * 4; part(np.array([g + m * 19 - d * 4, g + m * 19 + d * 4, g - m * 19 + d * 4, g - m * 19 - d * 4]), 3, seed=44)
        # upper arms, then the crossed forearms
        for sg in (-1, 1): part(S([(sg * .72, 1.40), (sg * 1.10, 1.44), (sg * 1.36, 1.80), (sg * 1.34, 2.30), (sg * 1.22, 2.86), (sg * .92, 2.96), (sg * .86, 2.4), (sg * .80, 1.9)], (5,)), 10 if sg > 0 else 6, [S([(1.0, 1.60), (1.22, 1.72), (1.22, 2.0), (1.04, 1.9)])] if sg > 0 else [], [([(sg * .84, 2.02), (sg * 1.02, 2.14), (sg * 1.26, 2.02)], 3.6), ([(sg * .92, 2.2), (sg * 1.0, 2.5), (sg * .96, 2.75)], 3.)], 50 + sg)
        part(S([(1.24, 2.66), (1.20, 3.02), (.60, 2.98), (-.10, 2.72), (-.70, 2.40), (-.62, 2.14), (.0, 2.34), (.66, 2.54)], (0, 1)), 6, [S([(.95, 2.74), (1.10, 2.80), (.6, 2.80), (.1, 2.60), (.4, 2.58)])], [([(.9, 2.92), (.4, 2.80), (-.1, 2.62)], 3.)], 60)
        part(S([(-1.24, 2.62), (-1.22, 2.98), (-.66, 3.04), (.10, 2.86), (.80, 2.52), (.86, 2.28), (.70, 2.22), (.0, 2.50), (-.64, 2.56)], (0, 1)), 8, [S([(-.9, 2.70), (-.4, 2.70), (.1, 2.58), (.3, 2.62), (-.3, 2.80)])], [([(-1.0, 2.92), (-.5, 2.92), (.1, 2.76)], 3.6), ([(-.2, 2.62), (.2, 2.56)], 2.6)], 63)
        part(S([(.64, 2.20), (.90, 2.10), (1.12, 2.16), (1.18, 2.36), (1.02, 2.50), (.78, 2.54), (.62, 2.40)]), 6, [], [([(.82, 2.14), (.90, 2.28), (1.14, 2.22)], 2.6), ([(.80, 2.26), (.92, 2.38), (1.16, 2.32)], 2.6), ([(.78, 2.38), (.90, 2.46), (1.10, 2.42)], 2.6)], 66)
        part(S([(-.60, 2.18), (-.86, 2.12), (-1.06, 2.20), (-1.02, 2.38), (-.74, 2.42)]), 4, [], [([(-.80, 2.14), (-.90, 2.26), (-1.05, 2.24)], 2.4), ([(-.78, 2.26), (-.9, 2.34), (-1.0, 2.34)], 2.4)], 68)
        part(S([(-.30, .80), (-.30, 1.16), (-.62, 1.40), (.62, 1.40), (.30, 1.16), (.30, .80)], (2, 3)), 6, [], [([(.16, 1.0), (.10, 1.2), (.04, 1.4)], 3.4), ([(-.2, 1.0), (-.10, 1.25), (-.02, 1.4)], 3.)], 70)
        hx, hy = P([(0, .50)])[0]; BP = dict(base=BONE, shade=BK, deep=BK, light='#fbf3dd', line=BK, term=BK, bounce=BK); HPAL = dict(deep=BK, shade=BK, base=BK, light=BONE, hot=BONE, line=BONE)
        zoro.draw(c, hx, hy, 47, .10, -.06, .02, 'glare', key=1, young=True, body=None, pal=BP, hairpal=HPAL, L=(.75, .45, .25), kick=0, seed=3, earrings=False, look=(0, .01))
        for q, w in (([(-.18, .22), (-.24, .60), (-.16, .95)], 6), ([(.10, 1.5), (.02, 2.0), (.10, 2.35)], 9), ([(-.5, 1.6), (-.56, 1.9), (-.48, 2.2)], 8), ([(.30, 3.1), (.36, 3.4), (.28, 3.7)], 8), ([(-1.16, 2.0), (-1.2, 2.4), (-1.1, 2.75)], 8), ([(.60, 4.6), (.70, 5.2), (.62, 5.7)], 9), ([(-.50, 4.7), (-.58, 5.4), (-.52, 6.0)], 8), ([(1.02, 2.0), (1.06, 2.3)], 7)): c.line(spl(P(q), n=8), w, '#c4161c', taper=(.08, .7), press=.8, seed=int(abs(q[0][0]) * 100))
        L_ = c.layer(); a = sumi.inkify(L_[..., 3], .30, 8, sig=.7, warp=.9); t = np.clip(sumi.tooth(H, W, 5), 0, 1); col = L_[..., :3] / np.maximum(L_[..., 3:4], 1e-3); lit = smooth(.35, .6, lum(col)) * (grid()[1] > 330); sp = lit * smooth(.80, .92, t) * .8; col = col * (1 - sp[..., None]) + rgb(BK) * sp[..., None]; return np.dstack([col * a[..., None], a])

    def setup(self):
        os.makedirs('cache', exist_ok=True); p = 'cache/d6_bg.npy'
        if not os.path.exists(p): np.save(p + '.tmp.npy', self.paint_().astype(np.float16)); os.replace(p + '.tmp.npy', p)
        self.bg = np.load(p).astype(f32); M = self.man(); y = grid()[1]
        fl = cv2.warpAffine(M[..., 3], np.float32([[1, .10, -94], [0, -.62, 948 + .62 * 948]]), (W, H)); fl = sumi.inkify(blur(fl, 2.5), .8, 12, sig=2.2, warp=7.) * smooth(1120, 940, y)                    # his long shadow comes to us across the red
        im = self.bg * (1 - .88 * fl[..., None]) + rgb('#1a0508') * (.88 * fl)[..., None]; halo = np.clip(blur(M[..., 3], 26) * 1.3 - M[..., 3], 0, 1); im = im + (halo * .16)[..., None] * rgb('#fff0d0'); im = over(im, M)
        c = Cel(); r = np.random.default_rng(5)                              # rubble in the near corner
        for q in ([(-40, 1120), (-40, 984), (90, 962), (240, 972), (380, 1010), (560, 1030), (700, 1074), (760, 1120)],): Q = resample(np.array(q, float), 20); c.fill(Q + r.normal(0, 5, Q.shape), '#ffffff')
        nk = sumi.inkify(c.rgba()[..., 0], .5, 21, sig=1.4, warp=3.); near = np.dstack([rgb(BK) * nk[..., None], nk]); c = Cel()
        for q, w in (([(-20, 998), (90, 976), (236, 985), (372, 1022)], 4.5), ([(250, 990), (262, 1030), (240, 1080)], 3.), ([(400, 1026), (520, 1042), (640, 1076)], 3.5), ([(80, 980), (110, 1020), (90, 1060)], 2.6)): c.line(spl(q, n=8), w, '#d9b79a', taper=(.1, .5), press=.6, seed=int(q[0][0]) + 50)
        for i in range(16): px = r.uniform(0, 480); c.line(np.array([(px, 1000 + px * .06 + r.uniform(0, 40)), (px + r.uniform(12, 50), 1006 + px * .06 + r.uniform(0, 44))]), 2., '#8d1218' if i % 3 else '#d9b79a', taper=(.1, .7), alpha=.8, seed=i)
        self.near = over_rgba(near, c.layer()); self.im = im

    def steam(self, f):
        c = Cel()
        for i in range(9):
            r = np.random.default_rng(40 + i); x0, y0 = 1010 + r.uniform(-120, 120), 936 - 1.12 * r.uniform(420, 640); t = np.linspace(0, 1, 30); L_ = r.uniform(120, 260); q = np.c_[x0 + np.sin(t * r.uniform(4, 7) - f * .16 + i) * (4 + 26 * t) + t * 30, y0 - t * L_ - (f * 2.2 + 40 * i) % 70]; c.line(q, r.uniform(5, 13), '#f4ead6', taper=(.25, .75), press=.8, alpha=.34, seed=i)
        return blur(c.layer(), 1.6)

    def frame(self, f):
        z = 1.015 + .0012 * f; img = over(self.im, self.steam(f)); img = cam2d(img, 960 + .5 * f, 548 - .3 * f, z, interp=cv2.INTER_CUBIC); nz = 1.02 + .0022 * f
        near = self.near; c = Cel(); ang, x, y, sq = .18, 330, 972, 1.
        if f < 11: t = f / 11.; x, y = 470 - 140 * t, -120 + 1092 * t ** 1.9; ang = -2.4 + 2.58 * t + 1.0
        elif f == 11: y, ang = 974, .30
        elif f < 14: y, x, ang = 972 - 16 * math.sin((f - 11) / 3. * math.pi), 330 + 5 * (f - 11), .30 - .07 * (f - 11)
        else: x, ang, y = 346, .12, 973
        if 2 < f:
            u = A_(math.cos(ang), math.sin(ang)); m = A_(-u[1], u[0]); o = A_(x, y); P = np.array([o - u * 74 + m * 10, o + u * 60 + m * 10, o + u * 66, o + u * 60 - m * 10, o - u * 74 - m * 10]); cloth(c, P, '#f4ead6', '#a89a8a', [np.array([o - u * 80 + m * 1, o + u * 66 + m * 1, o + u * 66 + m * 12, o - u * 80 + m * 12])], w=3.4, ink=BK); c.fill(np.array([o - u * 74 + m * 10, o - u * 36 + m * 10, o - u * 36 - m * 10, o - u * 74 - m * 10]), '#b98a52'); e = o + u * 62; c.fill(np.array([e + m * 10 - u * 16, e + m * 8, e + u * 8, e - m * 8, e - m * 10 - u * 14]), '#e8331c'); c.fill(np.array([e + m * 4 - u * 8, e + m * 4, e - m * 4, e - m * 5 - u * 8]), '#ffb45a')
            t = np.linspace(0, 1, 34)
            if f >= 11:
                g = min(1., (f - 10) / 12.); q = np.c_[e[0] + 14 * np.sin(t * 6.5 - f * .18) * t + 46 * np.sin(t * 2.6 - f * .05) * t, e[1] - 10 - t * 430 * g]; c.line(q, 8, '#efe4d0', taper=(.03, .85), press=.5, alpha=.62, seed=3); c.line(q + [10, 4], 3.4, '#efe4d0', taper=(.03, .85), press=.5, alpha=.5, seed=4)
            if 11 <= f < 17:
                for i in range(9): a = -2.6 + i * .25; d = (f - 10) * (6 + 3 * (i % 3)); c.fill(np.c_[e[0] + math.cos(a) * d + np.cos(np.linspace(0, 6.3, 8)) * 2.4, e[1] + math.sin(a) * d * .8 + .6 * (f - 10) ** 2 + np.sin(np.linspace(0, 6.3, 8)) * 2.4], '#ff6a3a')
        lay = over_rgba(near, c.layer()); lay = cam2d(lay, 960 - .6 * f, 540, nz, border=cv2.BORDER_REPLICATE); img = over(img, lay); e = 1 - .04 * max(0, 3 - f) * 0; return np.maximum(img, 0)


def drybrush_(P, width, seed, dry):
    cv = Canvas(H, W, color='#000000', layer=True); Q = resample(spl(np.array(P, float), n=10), 24.); ii = np.linspace(0, len(Q) - 1, 40).astype(int); cv.strokes(Q[ii][None], width, rgb('#000000')[None], op=.95, dry=dry, seed=seed, shape='flat', bristle=.35); return cv.a


def over_rgba(a, b):
    k = b[..., 3:4]; return a * (1 - k) + b
