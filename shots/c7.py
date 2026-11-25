"""c7 Franky: cyan / red card cut open by a weld torch; a hull under construction, Tom laughing in the light; arms swing up, the star closes: SUPER."""
from chars.gang_shot import *
from chars import franky as FR

ST = np.array([960., 344.])


def rib(k, cx, cy, th, sd=0):
    """one U-shaped frame of the hull, inner and outer contour"""
    O = [(-700, -520), (-760, -200), (-700, 120), (-480, 360), (-200, 470), (0, 500), (200, 470), (480, 360), (700, 120), (760, -200), (700, -520)]; I = [(u * (1 - th / 760.), v * (1 - th / 620.) if v > 0 else v) for u, v in O][::-1]
    return spn([(cx + u * k, cy + v * k) for u, v in O] + [(cx + u * k, cy + v * k) for u, v in I], True, (0, 10, 11, 21), 8), np.array([(cx + u * k, cy + v * k) for u, v in spn(I, False, (), 8)]), np.array([(cx + u * k, cy + v * k) for u, v in spn(O, False, (), 8)])


def make_bg():
    o = np.array([OW / 2 - 960, OH / 2 - 540]); cx, cy = OW / 2 + 60, OH / 2 - 40; x, y = grid(OH, OW); d = np.hypot((x - cx) / 900., (y - cy) / 560.); g = ramp(d, [(0, '#ffffff'), (.22, '#d7fbff'), (.5, '#7fdcea'), (.85, '#2f93b3'), (1.4, '#0d3f5c')]); c = Cel(OW, OH, ss=1); r = np.random.default_rng(6); n = 9
    for i in range(n):
        k = .20 * 1.30 ** i; t = i / (n - 1.); P, I, O_ = rib(k, cx, cy + 30 * (1 - t), 92); col = mix(rgb('#8fd3dc'), rgb('#1b2c3f'), t ** .8); warm = mix(rgb('#b9b0a0'), rgb('#6d2a22'), t)
        if i < n - 1:                                        # planking between the frames
            k2 = .20 * 1.30 ** (i + 1); P2, I2, O2 = rib(k2, cx, cy + 30 * (1 - (i + 1) / (n - 1.)), 92); m = len(O_)
            for j in range(0, m - 6, 6):
                if r.random() < (.95 if abs(j / m - .5) < .33 else .45): q = np.array([O_[j], O_[j + 5], O2[j + 5], O2[j]]); base = mix(col, warm, .55 if j < m * .5 else .15) * r.uniform(.72, 1.02); c.fill(q, base); c.fill(np.array([O_[j], O_[j + 1], O2[j + 1], O2[j]]), base * .6)
        c.fill(P, col); c.fill(np.vstack([I, (I * .96 + np.array([cx, cy]) * .04)[::-1]]), mix(rgb('#f4ffff'), rgb('#5fd0e6'), t) * (1 - .25 * t)); hh = len(I) // 2; c.fill(np.vstack([I[hh:], (I[hh:] * .955 + np.array([cx, cy]) * .045)[::-1]]), mix(rgb('#ffd9b0'), rgb('#f0682e'), t))
    for (x0, y0, x1, y1, w) in ((-200, 250, 900, 120, 46), (1300, 90, 2500, 300, 50)): c.fill(band_poly(np.array([[x0, y0], [x1, y1]], float) + 0, w), '#16273a'); c.fill(band_poly(np.array([[x0, y0 + w * .42], [x1, y1 + w * .40]], float), 7), '#ff8a44')
    for xx in (480, 1780, 2060): c.f = c.line(np.array([[xx, -20], [xx + 12, 300], [xx - 10, 620]]), 7, '#12202f', taper=(.05, .05))
    im = comp.over(g, c.rgba()); im = im * (1 - .40 * smooth(OH * .66, OH, y))[..., None]; im = painted(np.clip(im, 0, 1), 8, (24, 12, 6)) * .85 + im * .15
    b = np.zeros((OH, OW), f32)
    for a in (-.9, -.55, -.2, .25, .6): q = np.array([[cx - 20, cy - 40], [cx + 20, cy - 40], [cx + math.sin(a) * 1500 + 130, cy + 1000], [cx + math.sin(a) * 1500 - 130, cy + 1000]]); b += comp.shaft(OH, OW, q, 18, int(abs(a * 10)), .3) * .5
    return np.clip(im + (np.clip(b, 0, 1) * np.exp(-((y - cy) / 700.) ** 2))[..., None] * rgb('#8fe6f2') * .22, 0, 1.2)


def make_tom():
    """Tom: a belly like a boat, a back like a door, one fist on his chest, laughing with the whole mouth"""
    o = np.array([OW / 2 - 960, OH / 2 - 540]); D, D2, D3 = '#0f4158', '#175d78', '#23809c'; Q = lambda P, co=(): spn([(1478 + u, 650 + v) for u, v in P], True, co) + o; L = []
    L.append((Q([(-290, 740), (-400, 600), (-436, 430), (-410, 250), (-352, 90), (-290, -20), (-214, -86), (-120, -120), (120, -120), (214, -86), (300, -14), (368, 100), (424, 260), (448, 440), (408, 600), (300, 740)], (0, 15)), D))
    L.append((Q([(-128, -96), (-206, -150), (-236, -220), (-222, -300), (-160, -366), (-70, -396), (70, -396), (160, -366), (222, -300), (236, -220), (206, -150), (128, -96), (0, -64)]), D))
    for sd in (-1, 1): L.append((Q([(sd * 92, -380), (sd * 112, -408), (sd * 140, -424), (sd * 162, -414), (sd * 160, -388), (sd * 142, -362)]), D))
    L.append((Q([(310, -10), (420, 70), (476, 200), (470, 330), (420, 400), (352, 380), (330, 280), (350, 170), (300, 70)]), D2)); L.append((Q([(-300, -8), (-424, 80), (-470, 220), (-420, 320), (-300, 300), (-160, 190), (-110, 120), (-150, 60), (-260, 140), (-316, 150)]), D2))
    L.append((Q([(-268, 330), (-170, 270), (0, 250), (170, 268), (280, 330), (330, 450), (300, 590), (180, 690), (0, 720), (-180, 690), (-296, 590), (-322, 450)]), '#185f7a')); L.append((Q([(-186, 52), (-80, 0), (26, 30), (54, 116), (8, 196), (-106, 212), (-190, 146)]), D3))
    P = plate_rgba(L, 18, (18, 9, 4)); c = Cel(OW, OH, ss=2); S = lambda Pts, w, col='#c9f8ff', tp=(.2, .3): c.line(spn([(1478 + u, 650 + v) for u, v in Pts], False) + o, w, col, taper=tp)
    MO = Q([(-176, -262), (-120, -240), (0, -232), (120, -240), (176, -262), (164, -186), (104, -120), (0, -94), (-104, -120), (-164, -186)], (0, 4)); c.fill(MO, '#071e2b'); c.clip(MO); c.fill(Q([(-190, -280), (190, -280), (170, -214), (0, -192), (-170, -214)], (0, 1)), '#f2fdff'); c.fill(Q([(-90, -118), (0, -150), (90, -118), (70, -80), (-70, -80)]), '#e8745c'); c.unclip(); kit.outline(c, MO, 7, '#c9f8ff', 5)
    for sd in (-1, 1): S([(sd * 44, -310), (sd * 84, -338), (sd * 128, -306)], 11); S([(sd * 40, -356), (sd * 92, -378), (sd * 140, -350)], 12, '#0a2c3e'); c.fill(ell(1478 + sd * 176 + o[0], 650 - 278 + o[1], 24, 17, 20), '#e8745c', .45); S([(sd * 184, -270), (sd * 204, -246), (sd * 196, -214)], 6)
    S([(-34, -300), (-32, -268), (0, -254), (32, -268), (34, -300)], 9); [S([(u, -392 + abs(u) * .10), (u * 1.2, -424 + abs(u) * .2)], 12, D) for u in (-46, -16, 16, 46)]
    c.fill(Q([(-60, -92), (0, -80), (60, -92), (40, -50), (0, -14), (-40, -50)], (0, 2, 4)), '#0a2c3e')
    [S([(-64 + 46 * j, 22 + 12 * abs(j - 1)), (-68 + 46 * j, 84), (-54 + 46 * j, 132)], 9) for j in range(3)]; S([(-186, 140), (-104, 214), (8, 198), (54, 116)], 8); S([(-186, 56), (-80, 2), (26, 30), (54, 110)], 8); S([(-250, 400), (-150, 470), (0, 492), (150, 470), (250, 400)], 7, '#3aa0bb'); S([(-8, 560), (4, 576), (14, 562)], 8, '#3aa0bb'); S([(-214, -80), (-130, -20), (-40, 10)], 7, '#58c4d8'); S([(60, 20), (150, -30), (220, -80)], 7, '#58c4d8'); S([(-424, 316), (-300, 300), (-170, 200)], 6); S([(350, 380), (300, 470), (312, 600)], 7, '#58c4d8'); S([(60, 40), (130, 180), (140, 330)], 6, '#58c4d8')
    for fx_, fy_ in ((-330, 420), (-360, 560), (290, 200), (370, 480), (250, 90), (360, 600)): c.fill(Q([(fx_ + math.cos(a_) * 30 * r_, fy_ + math.sin(a_) * 30 * r_) for a_, r_ in zip(np.arange(10) * .628, [1, .45] * 5)]), '#2a8ba8')
    l = over4(P, c.rgba()); l = kit.rim(l, 9, 8, '#d8ffff', .95, 1.2); l = kit.rim(l, -11, 4, '#ff9a5a', .85, 1.2); return l


class Shot:
    id = 'c7'; n = 30; finish = dict(bloom=.36, vig=.20, halation='#d0f4ff')
    EVENTS = [(0, 'card', dict(note='cyan/red card')), (1, 'sizzle', dict(note='weld torch cuts the card f1-5')), (5, 'clang', dict(size=.4, note='plates fall apart')), (10, 'servo', dict(note='arms wide, wind-up')), (18, 'whoosh', dict(note='arms swing up')), (24, 'clack', dict(size=.3, note='shades drop')), (25, 'impact', dict(size=1.0, note='forearms clash, steel DON')), (25, 'laugh', dict(note='deep laugh behind')), (26, 'whoosh', dict(note='star wipe'))]

    def setup(self):
        self.bg = cached('c7_bg', make_bg); self.tom = cached('c7_tom', make_tom); self.cA = cached('c7_cyan', lambda: card('#2aa5bf', '#7fe0ea', '#136a8c', 71, -.5)); self.cB = cached('c7_red', lambda: card('#c9352b', '#f07a4a', '#7a141f', 72, -.5)); self.nxt = cached('c7_next', lambda: card('#1c2460', '#3f52ad', '#0c1034', 81, -.4))

    def pose(self, f, t=None):
        t = t if t is not None else f if f >= 12 or 4 <= f <= 9 else f // 2 * 2; tr = np.random.default_rng(60 + f // 2).normal(0, 1, 2) * (1 if 13 <= f <= 17 else 0)
        # per arm (viewer left): elbow, wrist end; mirrored for the right arm.  scale = foreshortening toward the lens
        El = key(t, [(0, (590, 800)), (5, (560, 800)), (9, (440, 700)), (12, (420, 720)), (16, (520, 840)), (18, (430, 700)), (20, (470, 430)), (22, (640, 440)), (24, (800, 508)), (25, (846, 512)), (26, (840, 506)), (29, (846, 512))], eout if False else ease)
        an = math.radians(key(t, [(0, 32.), (3, 24.), (9, 222.), (12, 216.), (13, 205.), (14, 170.), (15, 110.), (16, 52.), (17, 56.), (18, 150.), (20, 213.), (22, 250.), (24, 267.), (25, 270.4), (29, 270.)])); Ln = key(t, [(0, 205.), (4, 196.), (9, 258.), (12, 264.), (16, 190.), (18, 262.), (20, 322.), (22, 326.), (24, 300.), (25, 280.), (26, 306.), (29, 298.)]); Wr = El + Ln * np.array([math.cos(an), math.sin(an)])
        sc = key(t, [(0, 1.0), (9, 1.16), (12, 1.18), (16, 1.34), (18, 1.4), (20, 1.18), (22, 1.04), (25, 1.0)])
        hd = key(t, [(0, 20.), (5, 26.), (10, -6.), (13, -2.), (16, 66.), (18, 60.), (21, -26.), (24, -10.), (25, 26.), (26, -8.), (28, 2.)])
        lean = key(t, [(0, .02), (9, -.03), (16, .05), (21, -.02), (25, -.10), (29, -.085)])
        return t, El + tr * 3, Wr + tr * 4, sc, hd, lean

    def arcsmear(self, f, sd, M, s):
        """one smear drawing: the path of the steel forearm since the last frame, a solid comet head and a dry-brushed tail (no ink line: it is air, not a thing)"""
        p = Part(Rig(960, 600, s), None, 5780 + sd * 10, k=5.0); n = 24; i1 = 11; C = FR.C; ro = []; r = np.random.default_rng(f * 7 + sd + 2)
        for q in np.linspace(f - 1.5, f, n): _, El, Wr, sc, _, _ = self.pose(f, q); e, w = M(El), M(Wr); L = np.hypot(*(w - e)) + 1e-6; ro.append((e, (w - e) / L, L))
        at = lambda v, a, fn: np.array([e + d * (L * 1.02 + v * 185 * fn(i)) for i, (e, d, L) in enumerate(ro) if i >= a]); hd = lambda i: (max(0, i - i1) / (n - 1. - i1)) ** .6; tl = lambda i: .72 + .28 * i / (n - 1.)
        for v in np.linspace(-.96, .98, 17): p.f.line(at(v + r.normal(0, .03), int(r.integers(0, 10)), tl), r.uniform(5, 17), [C['redS'], C['red'], C['redS'], '#ffae90', C['redD'], C['red'], C['star']][int(r.integers(0, 7))], taper=(.92, .02))
        p.f.fill(np.vstack([at(1., i1, hd), at(-1., i1, hd)[::-1]]), C['redS']); p.f.fill(np.vstack([at(.95, i1, hd), at(-.10, i1, hd)[::-1]]), C['red'])
        for v, w_, col in ((.80, 13, '#ffae90'), (.52, 8, C['redD']), (.30, 20, C['star']), (.06, 6, C['white']), (-.40, 12, C['redD'])): p.f.line(at(v, 4, tl), w_, col, taper=(.9, .02))
        return p

    def figure(self, f, li):
        t, El, Wr, sc, hd, lean = self.pose(f); hx, hy = 960, 700 + hd; s = 152; st = Stack(); c_, s_ = math.cos(lean), math.sin(lean); R = lambda p: np.array([960 + (p[0] - 960) * c_ - (p[1] - 900) * s_, 900 + (p[0] - 960) * s_ + (p[1] - 900) * c_]); C = FR.C
        p = Part(Rig(960, 600, s), None, 5600, k=4.6); sy = hd * .55
        TB = spn([R(q) for q in ((842, 820 + sy), (760, 840 + sy), (660, 852 + sy), (590, 900 + sy), (600, 990 + sy), (690, 1060 + sy), (760, 1180), (1160, 1180), (1230, 1060 + sy), (1320, 990 + sy), (1330, 900 + sy), (1260, 852 + sy), (1160, 840 + sy), (1078, 820 + sy))], True, (6, 7)); B = p.shape(TB, C['skin'], 1.2); p.shade(spn([R(q) for q in ((1000, 800), (1040, 900 + sy), (960, 1000 + sy), (1010, 1200), (1400, 1200), (1400, 800))], True), C['sh'], B); p.shade(spn([R(q) for q in ((800, 800), (900, 900 + sy), (960, 912 + sy), (1020, 900 + sy), (1120, 800))], True), C['dk'], B, .8)
        p.line(np.array([R((960, 940 + sy)), R((964, 1010 + sy)), R((958, 1100))]), .5); p.line(spn([R((690, 1010 + sy)), R((820, 1062 + sy)), R((948, 1030 + sy))], False), .7); p.line(spn([R((1230, 1010 + sy)), R((1100, 1062 + sy)), R((972, 1030 + sy))], False), .7)
        for sd in (-1, 1):
            X = lambda u: 960 + sd * (u - 960) * -1 if False else 960 + sd * (u); Q = p.shape(spn([R((X(86), 838 + sy)), R((X(190), 832 + sy)), R((X(330), 846 + sy)), R((X(392), 900 + sy)), R((X(360), 1000 + sy)), R((X(300), 1180)), R((X(150), 1180)), R((X(196), 1040 + sy)), R((X(150), 930 + sy))], True, (0, 5, 6)), C['shirt'], 1.1); p.shade(Q + [-60 if sd < 0 else 56, 26], C['shirtS'], Q, .0 if sd < 0 else 1.)
            for (u, v, r) in ((240, 880, 26), (320, 950, 30), (230, 1010, 24)): q = R((X(u), v + sy)); [p.shade(ell(q[0] + math.cos(a) * r * .7, q[1] + math.sin(a) * r * .7, r * .5, r * .3, 14, rot=a), '#d9452f', Q) for a in np.arange(5) * 1.2566]; p.shade(ell(q[0], q[1], r * .28, r * .28, 12), '#fff0c8', Q)
            p.line(np.array([R((X(190), 832 + sy)), R((X(250), 920 + sy)), R((X(240), 1000 + sy))]), .45)
        st.add(p, li, 8); squash = 1.06 if f == 25 else 1.; parts = []
        for sd in (-1, 1):
            M = lambda q: R((960 + sd * (960 - q[0]) * 1., q[1])); e, w = M(El), M(Wr); sh = R((960 + sd * 300, 892 + sy)); p = Part(Rig(960, 600, s), None, 5700 + sd * 50, k=5.4 * min(1.25, sc)); ua = union(capsule(sh, e, 82, 62 * sc, 34)); ua = p.shape(ua, C['skin'], 1.0); p.shade(ua + [44, 22], C['sh'], ua); p.shade(ua + [86, 40], C['dk'], ua, .7); dd = (e - sh) / (np.hypot(*(e - sh)) + 1e-6); nn = np.array([-dd[1], dd[0]]); LL = np.hypot(*(e - sh)); p.line(spn([sh + dd * LL * .34 + nn * 70, sh + dd * LL * .50 + nn * 26, sh + dd * LL * .74 + nn * 54], False), .6); p.line(spn([sh + dd * LL * .40 - nn * 78, sh + dd * LL * .56 - nn * 40, sh + dd * LL * .80 - nn * 50], False), .5); p.hatch(spn([sh + dd * LL * .36 + nn * 60, sh + dd * LL * .52 + nn * 24, sh + dd * LL * .72 + nn * 56, sh + dd * LL * .55 + nn * 84], True), math.atan2(dd[1], dd[0]) + 1.1, 11, .3, fade=.3, within=ua); st.add(p, li, 8)
            p = Part(Rig(960, 600, s), None, 5750 + sd * 50, k=5.2); bl = p.shape(ell(sh[0], sh[1], 112, 106, 40), C['red'], 1.1); p.shade(bl + [38, 34], C['redS'], bl); p.shade(bl + [76, 70], C['redD'], bl, .8); p.shape(FR.star(sh[0] - 8, sh[1] - 4, 62, lean), C['white'], .5); p.shape(FR.star(sh[0] - 8, sh[1] - 4, 48, lean), C['star'], .4); p.f.clip(bl); p.f.line(ell(sh[0], sh[1], 94, 88, 24, 3.5, 4.7), 10, '#ffae90', taper=(.3, .4)); p.f.unclip(); [p.f.fill(ell(sh[0] + math.cos(a_) * 88, sh[1] + math.sin(a_) * 83, 6, 6, 10), C['ironD']) for a_ in np.arange(0, 6.2, .785)]; st.add(p, li, 9)
            if f in (6, 7, 14, 15, 16): parts.append((self.arcsmear(f, sd, M, s), li))
            p = Part(Rig(960, 600, s), None, 5800 + sd * 50, k=5.6 * min(1.25, sc)); d = w - e; inner = 1 if d[0] * (344. - e[1]) - d[1] * (960. - e[0]) > 0 else -1; FR.forearm(p, e, w, 216 * sc * squash, 228 * sc * squash, 1, 7 + sd, inner, s); parts.append((p, li))
        low = t < 19
        hl = FR.draw(Rig(R((hx, hy))[0], R((hx, hy))[1], s, -.10 + .12 * (t < 18) + (.0 if t < 25 else -.06), .10 if t > 19 else -.10 * (hd > 40), lean * 1.2, .12, sx=1 + (.03 if f == 25 else 0), sy=1 - (.04 if f == 25 else 0)), li, pick(FR.EX, 'calm' if t < 8 else 'grin' if t < 18 else ('grin', 'roar', min(1., (t - 18) / 6.)) if t < 25 else 'roar'), f, flip=key(f, [(0, 0.), (21, -.06), (23, .30), (24, 1.14), (25, .96), (27, 1.)], lambda v: v), flop={22: .16, 23: .12, 24: .02, 25: -.26, 26: .14, 27: -.06, 28: .03}.get(f, .10 * math.sin(t * .5) * (t < 16)))
        if low: st.a = over4(st.a, hl); [st.add(p, l, 9) for p, l in parts]
        else: [st.add(p, l, 9) for p, l in parts]; st.a = over4(st.a, hl)
        return st.a

    def sparks(self, f, seed, o, n, v0, ang, spread, life, col='#fff6d0', col2='#8ff2ff', t0=0):
        c = Cel(ss=1); r = np.random.default_rng(seed)
        for i in range(n):
            b = r.uniform(0, 1) * 8 if t0 is None else 0; a = ang + r.normal(0, spread); v = v0 * r.uniform(.4, 1.2); lf = life * r.uniform(.5, 1.); age = ((f - b) % 8.) if t0 is None else f - t0
            if age < 0 or age > lf: continue
            q = lambda s: np.array([o[0] + math.cos(a) * v * s * (1 - s * .03), o[1] + math.sin(a) * v * s * (1 - s * .03) + 1.6 * s * s]); k = 1 - age / lf; c.line(np.array([q(max(0, age - 1.3)), q(max(0, age - .6)), q(age)]), 2 + 5 * k * r.random(), col if r.random() < .6 else col2, taper=(.7, .1), press=0)
        return c.layer()

    def cola(self, f):
        """three bottles in a slatted crate, standing on the dock planks; they jump with the hit"""
        st = Stack(); p = Part(None, (0, 620, 760, 1080), 5900, k=4.2); j = {25: -16, 26: -26, 27: -18, 28: -4}.get(f, 0); W0, W1, W2 = '#9a5f38', '#5f3522', '#2f1a18'
        DK = np.array([[-10, 1000], [300, 984], [760, 1010], [760, 1090], [-10, 1090]], float); p.f.fill(DK, '#243247'); [p.f.fill(np.array([[-10, y0], [760, y0 + 8], [760, y0 + 12], [-10, y0 + 4]], float), '#0d1826') for y0 in (1024, 1052)]; p.l.line(DK[:3], .8 * p.k, INK, taper=(.02, .3)); p.f.line(DK[:3] + [0, 5], 4, '#8fe6f2', taper=(.02, .5))
        p.f.fill(spn([(330, 1066), (470, 1040), (640, 1034), (700, 1046), (640, 1068), (420, 1080)], True), '#070f1a', .75)
        p.f.fill(np.array([[70, 856], [392, 846], [404, 900], [60, 910]], float), '#120c12'); st.add(p, None); p = Part(None, (0, 520, 760, 1080), 5920, k=3.8)
        for i, (x, h, a) in enumerate(((130, 270, -.04), (230, 292, .02), (330, 266, .05))):
            y = 1010 + j * (1 + .3 * i); c, s_ = math.cos(a + (.05 * (i - 1) if j else 0)), math.sin(a + (.05 * (i - 1) if j else 0)); Q = lambda P: [(x + u * h * c - v * h * s_, y + u * h * s_ + v * h * c) for u, v in P]; b = [(-.15, 0), (-.16, -.30), (-.12, -.46), (-.155, -.60), (-.07, -.78), (-.055, -.95), (.055, -.95), (.07, -.78), (.155, -.60), (.12, -.46), (.16, -.30), (.15, 0)]; B = p.shape(spn(Q(b), True, (0, 11)), '#5b1c16', 1.); p.shade(B + [h * .10, 0], '#2a0c10', B); p.shape(spn(Q([(-.145, -.50), (-.135, -.66), (-.08, -.74), (.08, -.74), (.135, -.66), (.145, -.50), (0, -.46)]), True, (0, 5)), '#e9efe6', .5); p.f.fill(FR.star(*Q([(0, -.59)])[0], h * .085, a), '#3f74d6'); p.f.line(spn(Q([(-.10, -.40), (-.06, -.76), (-.035, -.9)]), False), h * .022, '#ffb08a', taper=(.3, .3)); p.shape(spn(Q([(-.07, -.94), (-.075, -1.01), (.075, -1.01), (.07, -.94)]), True, (0, 1, 2, 3)), '#d8a53a', .6)
        st.add(p, light('weld'), 5); p = Part(None, (0, 820, 760, 1080), 5940, k=4.2)
        for y0, y1, col in ((878, 948, W0), (962, 1040, W0)): q = p.shape(np.array([[54, y0 + 8], [410, y0 - 4], [414, y1 - 4], [50, y1 + 6]], float), col, 1.); p.shade(q + [230, 6], W1, q); [p.l.line(np.array([[70 + r_ * 5, y0 + 18 + r_ * 13], [230, y0 + 12 + r_ * 13], [396, y0 + 10 + r_ * 13]], float), .26 * p.k, W2, taper=(.3, .5)) for r_ in range(1, 4)]
        for x0, x1 in ((48, 92), (374, 420)): q = p.shape(np.array([[x0, 866], [x1, 862], [x1 + 2, 1058], [x0 - 2, 1062]], float), W1, 1.); p.shade(q + [22, 0], W2, q); [p.f.fill(ell((x0 + x1) / 2, yy, 5, 5, 10), '#c9d2de') for yy in (910, 1002)]
        p.shape(FR.star(232, 998, 30, .04), '#e9efe6', .4); st.add(p, light('weld'), 6); return st.a

    def frame(self, f):
        z = 1.02 + .004 * f + (.025 if f >= 25 else 0); sk = kit.shake(f, 12, 7, 2.4, 25) * [1, 1] if f >= 25 else np.zeros(2); sk = np.array([sk[0], -abs(sk[1])]); li = light('weld'); im = view(self.bg, sk[0] * .5, -10 + sk[1] * .5 + f * .5, z); lb = 10 * abs(math.sin(f // 2 * 1.1)) * (1 if f < 25 else 2.2); im = comp.over(im, view(self.tom, 10 + f * .6 + sk[0] * .6, lb + sk[1] * .6, z * 1.01)); im = im + comp.radial(1010, 470, 760, 520, 2.)[..., None] * rgb('#8ff0ff') * (.16 + .03 * math.sin(f * 2.1))
        for i, (o, a) in enumerate((((330, 300), 1.2), ((1780, 210), 2.0), ((1660, 820), -2.0))): im = comp.over(im, self.sparks(f, 300 + i, o, 26, 26, a, .5, 7, t0=None)); im = kit.flare(im, o[0], o[1], 70, '#c8fbff', .8 + .4 * math.sin(f * 2.7 + i), .25)
        if f < 10: im = self.open(im, f)
        fg = self.figure(f, li)
        if f in (18, 19, 20, 21): v = {18: (0, 60), 19: (0, 110), 20: (30, 120), 21: (40, 60)}[f]; m = np.clip(1.25 - comp.radial(960, 800, 420, 420, .6) * 2.2, 0, 1)[..., None]; fg = fg * (1 - m * .75) + sm(fg, v[0], v[1], 31) * m * .75
        fg = kit.shifted(fg, -sk[0], -sk[1]) if f >= 25 else fg; im = kit.cast(im, fg, 18, 24, '#06202e', .35, 14); im = comp.over(im, fg); im = comp.over(im, kit.shifted(self.cola(f), -abs(sk[0]), -sk[1]))
        if f >= 25: im = self.hit(im, f, sk)
        return im

    def open(self, im, f):
        n = np.array([.44, .898]); x, y = grid(); u = (x - 960) * n[0] + (y - 540) * n[1]; al = ((x - 960) * .898 - (y - 540) * .44); A, B = np.asarray(self.cA, f32), np.asarray(self.cB, f32); g = 0 if f < 5 else [0, 34, 130, 330, 640, 1100][min(5, f - 4)]; sh = lambda C_, s: cv2.warpAffine(C_, np.float32([[1, 0, n[0] * s], [0, 1, n[1] * s]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); jag = 5 * fbm(H, W, (22, 8), (1, .5), 5); mA = smooth(1, -1, u + g + jag); mB = smooth(-1, 1, u - g + jag); im = lerp2(lerp2(im, sh(A, -g), mA), sh(B, g), mB); front = -1150 + 2300 * min(1., (f + .3) / 4.2); on = smooth(front + 6, front - 6, al)
        for sg, e in ((-1, u + g + jag), (1, u - g + jag)): w = np.exp(-np.abs(e) / (5. if f < 5 else 9.)) * on * (1 if f < 8 else .5); im = im + w[..., None] * rgb('#ffffff') * 1.1 + (np.exp(-np.abs(e) / 34.) * on)[..., None] * rgb('#ff8a3a') * (.8 if f < 7 else .4)
        if f < 5: p0 = np.array([960 + .898 * front, 540 - .44 * front]); im = kit.flare(im, p0[0], p0[1], 220, '#d8fbff', 1.3, .5); im = comp.over(im, self.sparks(f, 500 + f, p0, 60, 46, 1.9, .8, 5, t0=f - 2))
        return im

    def hit(self, im, f, sk):
        q = f - 25; c0 = ST - sk
        if f == 25: L = np.clip(im, 0, 1); im = kit.impact(im, 'bw', .20, .58, '#0a2a44', '#f2ffff')
        c = Cel(); kit.focus_lines(c, c0[0], c0[1], 300 + 60 * q, 1700, 150, 11, '#ffffff' if q else '#d02a22', 90 + f, ry=.8); im = comp.over(im, c.layer(), .85 - .15 * q); im = comp.over(im, self.sparks(f, 700, c0, 150, 74, -1.57, 1.5, 9, t0=24)); im = kit.flare(im, c0[0], c0[1], 520, '#dffcff', [.9, .7, .5, .3, .2][q], .8)
        R = [215, 420, 840, 1600, 3600][q]; rot = -.06 - .10 * q
        for k, col in ((1.22, '#ffffff'), (1.08, '#3f74d6')): c = Cel(ss=1); c.fill([FR.star(c0[0], c0[1], R * k, rot), FR.star(c0[0], c0[1], R * (k - .10), rot)[::-1]] if q < 2 else FR.star(c0[0], c0[1], R * k, rot), col, evenodd=True); im = comp.over(im, c.layer(), 1. if q else .9)
        if q >= 1: m = mask_soft(FR.star(c0[0], c0[1], R * .96, rot)); im = lerp2(im, np.asarray(self.nxt, f32), m)
        return np.asarray(self.nxt, f32).copy() if q == 4 else im
