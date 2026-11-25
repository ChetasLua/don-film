"""c9 Jinbe: deep-blue card drains like a lens; he hauls the wheel under a wave tower, draws the palm back, and the sea answers with a hand of its own; water takes the lens."""
from chars.gang_shot import *
from chars import jinbe as J
from op import fx_air

PAL = dict(skin=J.C['skin'], shade=J.C['sh'], nail='#d7e6ee', nail_shade='#9ab4c8', light='#bfe2f2', cuff=J.C['haori'], cuff_shade=J.C['haoriS'], cuff_light='#f08a4a')
WPAL = dict(skin='#2f86b4', shade='#134c7e', line='#072440', nail='#8fd6ea', nail_shade='#2f86b4', nail_dark='#8fd6ea', light='#dff8ff', cuff='#2f86b4', cuff_shade='#134c7e', cuff_light='#8fd6ea')


def make_sky():
    x, y = grid(OH, OW); n = fbm(OH, OW, (240, 100, 40, 16), (1, .7, .5, .3), 8, 2.2); lx, ly = OW * .30, OH * .22; d = np.hypot((x - lx) / 1.5, y - ly); v = n * .34 + 1.05 * np.exp(-d / 420.) - .35 * smooth(0, OH, y); v = v - .25 * np.sin((x * .6 + y * 1.3) / 190. + n * .8); g = ramp(v, [(-.9, '#050d1c'), (-.4, '#0b1c34'), (-.05, '#12324f'), (.25, '#215a78'), (.6, '#5aa3b4'), (1.0, '#c9eee6')]); return painted(np.clip(g, 0, 1), 4, (48, 24, 11), -.5 + .6 * fbm(OH, OW, (200, 70), (1, .5), 5), bristle=.09)


def make_wave(side):
    """wave tower that rears and curls toward the middle"""
    o = np.array([OW / 2 - 960, OH / 2 - 540]); fx = (lambda P: [(1920 - u, v) for u, v in P]) if side > 0 else (lambda P: P)
    S = fx([(-460, 1500), (-460, 300), (-260, 80), (0, -60), (270, -90), (470, -20), (590, 120), (610, 262), (560, 330), (548, 250), (480, 190), (400, 200), (360, 290), (372, 430), (430, 600), (540, 780), (720, 930), (980, 1030), (980, 1500)]); L = [(spn(S, True, (0, 8, 9, 18)) + o, '#14507c')]
    g = flat(L); c, a = split(g); x, y = grid(OH, OW); X, Y = x - o[0], y - o[1]; X = 1920 - X if side > 0 else X; u = np.clip((X + 200) / 800., -1, 1.5); col = ramp(np.clip(.32 * u + .55 * smooth(900, 50, Y) + .16 * fbm(OH, OW, (90, 34, 13), (1, .6, .4), 9 + side, .5), 0, 1), [(0, '#05182e'), (.25, '#0b3358'), (.48, '#175f8c'), (.70, '#2f93b4'), (.86, '#7fd2d8'), (1, '#e0fbf4')]); an = np.arctan2(1.0 + .0 * X, .55 * np.cos((Y - 100) / 260.)) * (1 if side < 0 else -1); col = painted(np.clip(col, 0, 1), 12 + side, (30, 15, 7), an, bristle=.10); am = paint.warp_by(a, 3, 20, 4); P = np.dstack([col * am[..., None], am])
    k = Cel(OW, OH, ss=1); T = lambda Pts: spn(fx(Pts), False) + o; r = np.random.default_rng(7)
    for i in range(26): q = i / 26.; Pts = [(-420 + 820 * q + r.normal(0, 14), 1300), (-330 + 640 * q, 800), (-260 + 520 * q, 420), (-120 + 420 * q, 120), (80 + 340 * q, 0 + 60 * q), (330 + 220 * q, 30 + 120 * q)]; k.line(T(Pts), r.uniform(3, 10), ['#7fd2d8', '#0a2a4a', '#bff0ee', '#1b6f98'][i % 4], taper=(.5, .4), seed=i, alpha=.75, press=.6)
    ed = spn(fx([(-200, 40), (0, -60), (270, -90), (470, -20), (590, 120), (612, 262), (566, 332)]), False, (), 14) + o
    for j, (w, col) in enumerate(((58, '#8fd6dc'), (40, '#e4fbf6'), (24, '#ffffff'))): k.fill(scallop(band_poly(ed, [w * .2, w * .8, w, w * 1.1, w * 1.2, w * .7]), 34 - 7 * j, 15 - 3 * j, 5 + j), col)
    for i in range(120): e = ed[int(r.uniform(.1, 1) * (len(ed) - 1))]; p = e + np.array([r.normal(0, 34) + (26 if side < 0 else -26), abs(r.normal(0, 40)) * (1 if r.random() < .7 else -1)]); s = r.uniform(2, 11) * r.random() + 1.5; k.fill(ell(p[0], p[1], s, s * .9, 10), '#f4fffb')
    for i in range(9): e = ed[int((.35 + .07 * i) * (len(ed) - 1))]; sg = 1 if side < 0 else -1; k.line(np.array([e, e + [sg * 30, 44 + 12 * (i % 3)], e + [sg * 16, 84 + 18 * (i % 4)]]), 18, '#f4fffb', taper=(.1, .7))      # fingers of foam
    return over4(P, k.rgba())


def make_fore():
    """stern rail and deck edge, low"""
    o = np.array([OW / 2 - 960, OH / 2 - 540]); L = [(spn([(-300, 1500), (-300, 1010), (400, 985), (960, 975), (1520, 985), (2220, 1010), (2220, 1500)], True, (0, 1, 5, 6)) + o, '#3c2a2c')]
    for x in range(-260, 2300, 150): L.append((np.array([[x, 900], [x + 44, 900], [x + 52, 1030], [x - 8, 1030]]) + o, '#6b3f2e')); L.append((np.array([[x + 26, 900], [x + 44, 900], [x + 52, 1030], [x + 26, 1030]]) + o, '#2b1a20'))
    L.append((spn([(-300, 915), (-300, 868), (960, 846), (2220, 868), (2220, 915), (960, 892)], True, (0, 1, 3, 4)) + o, '#8a5236')); L.append((spn([(-300, 915), (-300, 898), (960, 876), (2220, 898), (2220, 915), (960, 892)], True, (0, 1, 3, 4)) + o, '#3a2224')); P = plate_rgba(L, 15, (16, 8, 4)); P[..., :3] *= rgb('#9fb4cc'); return kit.rim(P, 4, 5, '#dff4ff', .9, 1.)


class Shot:
    id = 'c9'; n = 30; finish = dict(bloom=.30, vig=.22, halation='#d8f2ff')
    EVENTS = [(0, 'card', dict(note='deep blue card')), (1, 'water', dict(note='card drains down the lens f1-8')), (3, 'storm', dict(note='wind and rain bed')), (5, 'creak', dict(note='wheel hauled over f5-12')), (9, 'thunder', dict(size=.6, note='lightning')), (13, 'inhale', dict(note='palm drawn back, sea goes quiet')), (21, 'whoosh', dict(note='palm thrust')), (25, 'impact', dict(size=1.0, note='palm lands, DON, wave bends round a hand of water')), (26, 'water', dict(size=1.0, note='water wipes the lens'))]

    def setup(self):
        self.sky = cached('c9_sky', make_sky); self.wL = cached('c9_waveL', lambda: make_wave(-1)); self.wR = cached('c9_waveR', lambda: make_wave(1)); self.fg = cached('c9_fore', make_fore); self.card = cached('c9_card', lambda: card('#0d3158', '#1f6a9c', '#061a34', 91, -.2)); self._hx = (505., 800.)

    def wheel(self, p, cx, cy, R, th):
        W_, WS, WH = '#9a5c34', '#54291f', '#e0a45e'; Q = lambda r, a: np.array([cx + math.cos(a) * r, cy + math.sin(a) * r * .94])
        for i in range(8):
            a = th + i * math.pi / 4; sp = band_poly(np.array([Q(R * .16, a), Q(R * .62, a), Q(R * 1.0, a), Q(R * 1.10, a), Q(R * 1.28, a)]), [26, 30, 26, 22, 34], 6); p.shape(sp, W_, .8); p.shade(sp + [9, 9], WS, sp); kn = ell(*Q(R * 1.29, a), 20, 20, 18); p.shape(kn, '#d7a03c', .7); p.shade(kn + [8, 8], '#7a4a1c', kn)
        for r0, r1 in ((R * .99, R * .80),):
            a = np.linspace(0, 2 * math.pi, 80, endpoint=False); O = np.array([Q(r0, t) for t in a]); I = np.array([Q(r1, t) for t in a])[::-1]; p.f.fill([O, I], W_, evenodd=True); p.f.ctx.save(); p.f._path([O, I]); p.f.ctx.set_fill_rule(1); p.f.ctx.clip(); p.f.fill([O + [14, 18], I + [14, 18]], WS, evenodd=True); p.f.line(np.array([Q((r0 + r1) / 2 + 12, t) for t in np.linspace(3.3, 5.0, 24)]), 9, WH, taper=(.3, .3)); p.f.ctx.restore(); p.ol(O, .9); p.ol(I, .7); M = np.array([Q((r0 + r1) / 2, t) for t in a]); p.l.line(np.vstack([M, M[:1]]), .28 * p.k, INK, taper=(0, 0))
            for i in range(16): p.f.fill(ell(*Q((r0 + r1) / 2, th + (i + .5) * math.pi / 8), 7, 7, 10), '#e8c060')
        hb = ell(cx, cy, R * .22, R * .21, 40); p.shape(hb, '#e7b94a', 1.); p.shade(hb + [14, 14], '#8f5a20', hb); ln = [p.shape(spn([Q(R * .21, th + k_), Q(R * .30, th + k_ + .19), Q(R * .21, th + k_ + .38)], True, (0, 1, 2)), '#d8662c', .5) for k_ in np.arange(0, 6.2, .392)]; p.shape(ell(cx, cy, R * .17, R * .16, 30), '#f2cf62', .7); [p.f.fill(ell(cx + sd * R * .06, cy - R * .03, 5, 7, 10), INK) for sd in (-1, 1)]; p.l.line(spn([(cx - R * .08, cy + R * .04), (cx, cy + R * .08), (cx + R * .08, cy + R * .04)], False), .5 * p.k, INK)

    def jinbe(self, f, sk, li):
        t = f if f >= 19 else f // 2 * 2; C = J.C; st = Stack(); tr = np.random.default_rng(30 + f // 2).normal(0, 1, 2) * (1.6 if 6 <= f <= 20 else 0)
        th = key(t, [(0, .1), (4, .22), (6, .10), (12, -1.22), (14, -1.10), (24, -1.10), (25, -1.02), (27, -1.12)]); ln = key(t, [(0, 0.), (4, .05), (9, -.13), (12, -.10), (18, .16), (21, .18), (23, -.02), (25, -.15), (28, -.10)]); cr = key(t, [(0, 0.), (4, 14), (9, 26), (13, 4), (18, 44), (21, 48), (25, -20), (28, -8)])
        hx, hy = 985 + ln * 190 + tr[0], 322 + cr + tr[1]; s = 136; R = lambda u, v: np.array([960 + (u - 960) * math.cos(ln * .6) - (v - 900) * math.sin(ln * .6), 900 + (u - 960) * math.sin(ln * .6) + (v - 900) * math.cos(ln * .6) + cr * .6 * (1 if v < 900 else 0)]); g = 16 * math.sin(f * .8)
        p = Part(Rig(960, 500, s), None, 9200, k=4.8)
        # haori back panels flying
        for sd in (-1, 1):
            w1, w2, w3 = (30 * math.sin(f * .75 + sd * 1.4 + j * 1.1) for j in range(3)); kx_ = .80 if sd < 0 else 1.06; R0 = R; R = (lambda u, v, R0=R0, kx_=kx_: R0(960 + (u - 960) * kx_, v)); B = p.shape(spn([R(960 + sd * 200, 462), R(960 + sd * 420, 500), R(960 + sd * 540 + w1 * .5, 600 + w1), R(960 + sd * 690 + w2, 690 + w2), R(960 + sd * 800 + w3, 700 + w3 * 1.2), R(960 + sd * 770 + w2, 800 + w3), R(960 + sd * 640 + w2, 860 + w2), R(960 + sd * 560 + w1, 1000), R(960 + sd * 400, 1130), R(960 + sd * 200, 1130)], True, (4, 9)), C['haoriD'], 1.); p.shade(spn([R(960 + sd * 560 + w1, 640 + w1), R(960 + sd * 690 + w2, 760 + w2), R(960 + sd * 600 + w1, 830 + w1), R(960 + sd * 520, 960), R(960 + sd * 420, 1130), R(960 + sd * 300, 1130), R(960 + sd * 430, 800)], True), '#8f2f1f', B); p.line(spn([R(960 + sd * 430, 620), R(960 + sd * 580 + w1, 720 + w1), R(960 + sd * 740 + w2, 760 + w2)], False), .5); p.line(spn([R(960 + sd * 420, 760), R(960 + sd * 520 + w1, 850), R(960 + sd * 540 + w1, 960)], False), .45); R = R0
        TB = spn([R(700, 500), R(630, 560), R(640, 720), R(720, 900), R(760, 1140), R(1200, 1140), R(1240, 900), R(1320, 720), R(1330, 560), R(1260, 500), R(1100, 470), R(860, 470)], True, (4, 5)); B = p.shape(TB, C['skin'], 1.2); p.shade(spn([R(1010, 440), R(1070, 600), R(1010, 720), R(1100, 840), R(1040, 1200), R(1500, 1200), R(1500, 440)], True), C['sh'], B); p.shade(spn([R(1240, 440), R(1260, 640), R(1190, 860), R(1180, 1200), R(1500, 1200), R(1500, 440)], True), C['dk'], B, .8); p.shade(spn([R(760, 440), R(820, 520), R(980, 560), R(1130, 520), R(1200, 440)], True), C['dk'], B, .85)
        p.line(spn([R(700, 660), R(820, 716), R(950, 690)], False), .7); p.line(spn([R(1250, 660), R(1120, 716), R(970, 690)], False), .7); p.line(np.array([R(958, 560), R(962, 640), R(958, 700)]), .5); p.ticks((0, 0), (.01, 0), 1, .01, .001) if 0 else None
        c0 = R(962, 742); [p.f.ctx.save() for _ in (0,)]; p.f._path(B); p.f.ctx.clip(); [p.f.fill(q, C['sun']) for q in J.sunmark(c0[0], c0[1], 150, .1 + ln)]; p.f.fill(J.sunmark(c0[0] + 12, c0[1] + 10, 150, .1 + ln)[0], '#9c2222', .55); p.f.ctx.restore(); [p.l.line(np.vstack([q, q[:1]]), .4 * p.k, INK) for q in J.sunmark(c0[0], c0[1], 150, .1 + ln)]
        for sd in (-1, 1):                                   # front panels + wide sleeves root
            Q = p.shape(spn([R(960 + sd * 120, 452), R(960 + sd * 230, 446), R(960 + sd * 390, 520), R(960 + sd * 430, 640), R(960 + sd * 400, 800), R(960 + sd * 350, 1130), R(960 + sd * 190, 1130), R(960 + sd * 250, 860), R(960 + sd * 230, 640)], True, (5, 6)), C['haori'], 1.1); p.shade(Q + [46, 26] if sd < 0 else Q + [66, 10], C['haoriS'], Q); T2 = p.shape(spn([R(960 + sd * 112, 452), R(960 + sd * 168, 446), R(960 + sd * 276, 640), R(960 + sd * 300, 860), R(960 + sd * 240, 1130), R(960 + sd * 190, 1130), R(960 + sd * 250, 860), R(960 + sd * 226, 640)], True, (4, 5)), C['trim'], .8); p.shade(T2 + [16 * 1, 8], C['goldS'], T2); [p.l.line(np.array([R(960 + sd * u, v), R(960 + sd * (u + 34), v + 8)]), .4 * p.k, '#8a2a1c') for u, v in ((150, 520), (196, 580), (226, 640), (244, 720), (246, 800), (238, 890))]; p.line(spn([R(960 + sd * 330, 520), R(960 + sd * 310, 640), R(960 + sd * 350, 760)], False), .5); p.line(spn([R(960 + sd * 380, 580), R(960 + sd * 360, 700), R(960 + sd * 380, 800)], False), .4)
        st.add(p, li, 9)
        # head
        ex = 'calm' if t < 4 else ('calm', 'strain', min(1, (t - 4) / 4.)) if t < 14 else ({'base': 'calm', 'eo': .34, 'bi': .08, 'bo': -.06}) if t < 21 else ('strain', 'roar', [0, .3, .6, .9, 1.][min(4, f - 21)]) if f < 25 else 'roar'
        e = pick(J.EX, ex); hT = Rig(hx, hy, s, key(t, [(0, .05), (8, -.18), (14, -.10), (20, .12), (25, -.12)]), key(t, [(0, .06), (9, -.06), (19, -.10), (25, .08)]), ln * .8, .12, r=(.92, 1.14, .84), sx=1.02 if f == 25 else 1, sy=.97 if f == 25 else 1); st.a = over4(st.a, J.draw(hT, li, e, f, wind=-.15 + .12 * math.sin(f * .7)))
        # wheel, both hands / one hand
        p = Part(Rig(960, 500, s), None, 9400, k=4.6); wc = np.array([1000., 968.]); self.wheel(p, wc[0], wc[1], 265, th); st.add(p, li, 7)
        wq = lambda i: wc + np.array([math.cos(th + i * math.pi / 4) * 265 * 1.12, math.sin(th + i * math.pi / 4) * 265 * 1.05])
        def sleeve(p, sh, el, hx_, sd, seed):
            """wide haori sleeve: shoulder cap, elbow, hanging mouth with a wave-crest border"""
            d = (hx_ - el) / (np.hypot(*(hx_ - el)) + 1e-6); u = (el - sh) / (np.hypot(*(el - sh)) + 1e-6); m = np.array([-d[1], d[0]]); n = np.array([-u[1], u[0]]); n = n if n @ np.array([sd, -.5]) > 0 else -n; m = m if m @ np.array([sd * .4, -1.]) > 0 else -m; c = el * .52 + hx_ * .48; fl = 14 * math.sin(f * .8 + sd)
            k1, k2 = c + m * 104, c + np.array([fl, 176.]); Q = p.shape(union(capsule(sh + np.array([-sd * 60., -6.]), sh + u * 40, 112, 126), capsule(sh + u * 30, el, 124, 116, 6), capsule(el, c, 116, 102), spn([el + np.array([-sd * 30., 40.]), c + m * 60, c + d * 18 + np.array([0, 40.]), k2, c * .55 + el * .45 + np.array([fl * .7, 210.]), el + np.array([fl * .4, 168.])], True, (3,))), C['haori'], 1.15)
            p.shade(spn([sh + np.array([-sd * 160., 40.]), sh * .55 + el * .45 + np.array([0, 36.]), el + np.array([-sd * 20, 36.]), c + np.array([0, 34.]), k2 + [60, -30], k2 + [0, 80], el + np.array([0, 300.]), sh + np.array([-sd * 180., 300.])], True), C['haoriS'], Q); p.shade(spn([el + np.array([-sd * 40, 96.]), c + np.array([0, 104.]), k2 + [60, 60], el + np.array([0, 320.]), sh + np.array([-sd * 160., 320.]), sh + np.array([-sd * 120., 110.])], True), C['haoriD'], Q, .85)
            for j, (q0, q1, q2, w_) in enumerate(((el + n * 104 - u * 10, el + n * 40 + d * 6, el - n * 30 + d * 60, 34), (el + n * 96 - u * 60, el + n * 34 - u * 56, el - n * 30 - u * 70, 24), (sh + n * 112 + u * 70, sh + n * 66 + u * 104, sh + u * 160 + n * 26, 28), (k2 + [0, -40], c * .6 + el * .4 + np.array([0., 84.]), el + np.array([0, 24.]), 30))): B_ = band_poly(np.array([q0, q1, q2]), [2, w_, 2], 8); p.shade(B_, C['haoriS'], Q); p.f.clip(Q); p.f.line(np.array([q0, q1, q2]) - [4, 8], 7, '#f59a58', taper=(.3, .6)); p.f.unclip(); p.line(np.array([q0, q1, q2]), .5, taper=(.1, .7)) if j in (0, 3) else None
            tb = band_poly(np.array([k1 + m * 10, c + d * 22 + np.array([fl * .4, 46.]), k2 + [0, 16]]) - d * 12, 46, 10); p.f.clip(Q); p.f.fill(tb, C['trim']); p.f.fill(tb + [12, 14], C['goldS']); p.f.unclip(); p.l.line(spn([k1 - d * 38, c - d * 26 + np.array([fl * .4, 46.]), k2 - d * 36], False), .5 * p.k, INK)
            for t_ in np.linspace(.12, .88, 6): w0 = (k1 * (1 - t_) ** 2 + 2 * (c + d * 12 + np.array([0, 46.])) * t_ * (1 - t_) + k2 * t_ * t_) - d * 18; p.f.line(ell(w0[0], w0[1], 13, 13, 10, 2.6, 6.2), 4, '#a3321f', taper=(.1, .1))
            return Q
        def arm(sd, hand_xy, w1, seed, el=None, hs=116, pose='hammer', Rr=None, dist=6.):
            sh = R(960 + sd * 330, 590); el = el if el is not None else (sh + hand_xy) / 2 + np.array([sd * 130, 40.]); p = Part(Rig(960, 500, s), None, seed, k=4.8); d = (hand_xy - el) / (np.hypot(*(hand_xy - el)) + 1e-6); m = np.array([-d[1], d[0]]); fa = capsule(el, hand_xy - d * 6, 70, w1, 16); p.shape(fa, C['skin'], 1.); p.shade(fa + [26, 30], C['sh'], fa); [p.f.fill(ell(*(el + (hand_xy - el) * u + m * v), 8, 7, 10), C['belly'], .7) for u, v in ((.68, 20), (.76, -16), (.84, 18))]; p.line(spn([el + (hand_xy - el) * .62 + m * 34, el + (hand_xy - el) * .76 + m * 14, el + (hand_xy - el) * .9 + m * 22], False), .4)
            st.add(p, li, 8); p = Part(Rig(960, 500, s), None, seed + 40, k=5.0); sleeve(p, sh, el, hand_xy, sd, seed); st.add(p, li, 9); p = Part(Rig(960, 500, s), None, seed + 60, k=4.6); cc = ell(hand_xy[0] + d[0] * hs * .42, hand_xy[1] + d[1] * hs * .42, hs * 1.12, hs * 1.12, 40); p.f.clip(cc); p.l.clip(cc); hand(p, pose, *hand_xy, hs, Rr if Rr is not None else HD.euler(20 * sd, 30, -10 * sd), PAL, 'L' if sd > 0 else 'R', dist, w=4.4, res=340, bld=HD.HEAVY); p.f.unclip(); p.l.unclip(); st.add(p, li, 8); return p
        arm(1, wq(round((-.35 - th) / (math.pi / 4))), 50, 9500, pose='hammer', Rr=HD.euler(165, 10, 108))
        if t < 13: arm(-1, wq(round((-2.75 - th) / (math.pi / 4))), 50, 9600, pose='hammer', Rr=HD.euler(195, 10, -108))
        elif t < 22:                                         # on ones: two frames of wind-up into the wheel, then the palm comes round on a forearm that rises from below
            u = key(f, [(13, 0.), (15, -.04), (16, .20), (17, .52), (18, .84), (19, 1.02), (20, 1.0), (21, 1.06)], lambda v: v); uc = float(np.clip(u, 0, 1)); wa = {14: .7, 15: 1., 16: .35}.get(f, 0.)
            hxy = wq(round((-2.75 - th) / (math.pi / 4))) * (1 - u) + np.array([516., 776.]) * u + np.array([42., 22.]) * wa; self._hx = hxy; el = ((R(630, 590) + hxy) / 2 + np.array([-130, 40.]) + np.array([-44., -56.]) * wa) * (1 - uc) + R(400, 930) * uc
            arm(-1, hxy, 60 + 14 * uc, 9600, el=el, hs=116 + 74 * u, pose=dict(HD.blend('hammer', 'open', uc), wrist=(-52 * uc, 0)), Rr=HD.euler(188, -6 + 14 * u, -100 + 84 * u), dist=5.)
        else: self.palm(st, f, R, li)
        if 14 <= f < 23:
            p = Part(None, None, 9650, k=2.2); r = np.random.default_rng(44); o = np.asarray(self._hx) if f < 22 else np.array([590., 660.])
            for i in range(60):
                ph = (f - 14) / 8.5 * r.uniform(.8, 1.3) + r.uniform(-.45, .15)
                if not 0 < ph < 1: continue
                a = r.uniform(0, 6.28) + 2.4 * ph; d_ = (1 - ph) ** 1.4 * r.uniform(200, 640) + 46; bead(p, o[0] + math.cos(a) * d_, o[1] + math.sin(a) * d_ * .8, 4 + 8 * r.random(), a + math.pi + .6, 2.4, '#e6fbff', '#6fc4dc', .4)
            for i in range(4): a0 = i * 1.57 + f * .5; R_ = 150 - 9 * (f - 14) + 26 * i; p.f.line(ell(o[0], o[1], R_, R_ * .8, 18, a0, a0 + 1.5), 6, '#c9f4f6', taper=(.6, .1))
            st.add(p)
        return st.a

    def palm(self, st, f, R, li):
        q = f - 22; c = [(566, 700), (604, 630), (640, 580), (656, 556), (650, 562), (652, 560), (652, 560), (652, 560)][q]; sc = [170, 250, 322, 384, 358, 366, 364, 364][q]; C = J.C; sh = R(650, 590); ce = np.array(c, float) + [26, sc * .30]; R1 = 130 + sc * .36; p = Part(Rig(960, 500, 136), None, 9700, k=5.8); fl = 18 * math.sin(f * 1.1)
        sl = union(capsule(sh + [40, -20], sh, 130, 130), capsule(sh, ce, 128, R1), spn([tuple(ce + v) for v in ((-R1 * .98, 0), (-R1 * .9, R1 * .9), (-R1 * .5 + fl, R1 * 1.45), (R1 * .3 + fl, R1 * 1.38), (R1 * .95, R1 * .8), (R1, 0))], True)); Q = p.shape(sl, C['haori'], 1.2); p.shade(Q + [60, 60], C['haoriS'], Q)
        for a_ in (2.4, 2.9, 3.5, 4.1): q0 = ce + np.array([math.cos(a_), math.sin(a_)]) * R1 * 1.0; q1 = sh * .5 + ce * .5 + np.array([math.cos(a_), math.sin(a_)]) * R1 * .7; p.f.clip(Q); p.f.fill(band_poly(np.array([q0, (q0 + q1) / 2 + [8, 10], q1]), [26, 16, 2], 8), C['haoriS']); p.f.unclip(); p.line(np.array([q0, (q0 + q1) / 2 + [8, 10], q1]), .5, taper=(.1, .7))
        mo = ell(ce[0] + 4, ce[1] + 10, R1 * .96, R1 * .92, 60); p.shape(mo, C['trim'], 1.1); p.shade(mo + [26, 30], C['goldS'], mo); mi = ell(ce[0] + 6, ce[1] + 14, R1 * .84, R1 * .80, 60); p.shape(mi, '#3a0f14', .9); p.shade(mi + [-30, -60], '#1c070c', mi); [p.f.line(ell(ce[0] + math.cos(t_) * R1 * .90, ce[1] + 12 + math.sin(t_) * R1 * .86, R1 * .06, R1 * .06, 10, t_ + 1.2, t_ + 4.9), 5, '#a3321f', taper=(.1, .1)) for t_ in np.arange(0, 6.2, .45)]
        fa = capsule(ce + [0, 10], np.array(c, float) + [8, sc * .30], R1 * .46, sc * .34); p.shape(fa, C['skin'], 1.1); p.shade(fa + [46, 30], C['sh'], fa); st.add(p, li, 10); p = Part(Rig(960, 500, 136), None, 9720, k=6.4)
        g = hand(p, dict(HD.POSES['open'], wrist=(-62, 0)), c[0], c[1], sc, HD.euler(180, 10, 8), PAL, 'R', 3.0, w=6.2, res=460, bld=HD.HEAVY)
        pts = g.pts[0] if not isinstance(g.pts, dict) else g.pts.get(0, list(g.pts.values())[0])
        F_ = ('index', 'middle', 'ring', 'little'); PT = lambda k_, j: np.asarray(pts[(k_, j)], float).ravel()[:2]
        try:                                                 # webs between the fingers
            for a, b in zip(F_[:-1], F_[1:]): A0, B0, A1, B1 = PT(a, 'mcp'), PT(b, 'mcp'), PT(a, 'pip'), PT(b, 'pip'); e0, e1 = A1 * .92 + A0 * .08, B1 * .92 + B0 * .08; md = (e0 + e1) / 2 * .66 + (A0 + B0) / 2 * .34; W_ = np.vstack([spn([e0, md, e1], False), [B0 * .7 + B1 * .3, A0 * .7 + A1 * .3]]); p.f.fill(W_, C['sh']); p.l.line(spn([e0, md, e1], False), .7 * p.k, INK, taper=(.2, .2)); p.l.line(np.array([md, (A0 + B0) / 2 * .5 + md * .5]), .35 * p.k, INK, taper=(.1, .7))
        except Exception as ex_: print('web', ex_)
        rr = np.random.default_rng(50 + f); pc = np.array(c, float)
        for k_ in F_ + ('thumb',):                           # water shed from the fingertips
            tp = PT(k_, 'tip'); dv = (tp - pc) / (np.hypot(*(tp - pc)) + 1e-6); an_ = math.atan2(dv[1], dv[0])
            for i in range(4): w_ = tp + dv * rr.uniform(10, 34 + 26 * q) + np.array([-dv[1], dv[0]]) * rr.normal(0, sc * .06); bead(p, w_[0], w_[1], sc * rr.uniform(.016, .040), an_ + rr.normal(0, .25), 2.0, '#e6fbff', '#6fc4dc', .4)
            side_ = np.array([-dv[1], dv[0]]); p.f.line(spn([tuple(tp - dv * sc * .34 + side_ * sc * .10), tuple(tp - dv * sc * .10 + side_ * sc * .13), tuple(tp + dv * sc * .05 + side_ * sc * .05)], False), sc * .022, '#e9fbff', taper=(.3, .4)); p.f.line(spn([tuple(tp - dv * sc * .42 - side_ * sc * .09), tuple(tp - dv * sc * .20 - side_ * sc * .12)], False), sc * .014, '#bff0f4', taper=(.4, .5))
        st.add(p, li, 11)

    def whand(self, f):
        """the hand of water: a giant open palm rising behind him, drawn with the hand engine in sea colours"""
        q = max(0, f - 19); k = [0., .25, .48, .70, .88, .98, 1.10, 1.04, 1.05, 1.06, 1.07][min(10, q)]
        if k <= 0: return None
        key_ = f
        p = Part(Rig(960, 500, 300), None, 9800, k=6.5, ink='#0a2c4c'); sc = 400 * k; cx, cy = 1385, 470 - 60 * k + 380 * (1 - k); g = hand(p, dict(HD.POSES['open'], wrist=(-8, 0)), cx, cy, sc, HD.euler(180, -6 + 10 * k, 12), WPAL, 'R', 5., w=7., res=420, bld=HD.HEAVY); st = Stack(); st.add(p, dict(rim=[(UL, '#eafffb', 1., 1.5), (R_, '#8fe0ea', .7, 1.)], air=(UL, '#9fe8f0', .5, 90)), 11); L = st.a; c = Cel(); r = np.random.default_rng(6); a = L[..., 3]
        for i in range(34): x = cx + r.uniform(-1, 1) * sc * .9; y = cy + r.uniform(-1, 1) * sc; ln = r.uniform(60, 190) * k; ph = (f * .2 + r.random()) % 1; c.line(np.array([[x, y + ln * .5 - ph * 40], [x + r.normal(0, 10), y - ph * 40], [x + r.normal(0, 16), y - ln * .5 - ph * 40]]), r.uniform(4, 12), ['#8fe0ea', '#dffaf6', '#0f4a7a'][i % 3], taper=(.5, .5), seed=i)
        l = c.layer() * cv2.erode(a, np.ones((9, 9)))[..., None] * .75; L = over4(L, l); cs = kit.contours(a, min_len=300); c = Cel(); 
        if cs: e = max(cs, key=len); b = scallop(e, 46, 15, 4); c.fill(b, '#dffaf6'); c.fill(e, '#ffffff', 0) if 0 else None; foam = c.layer(); foam = foam * np.clip(1 - cv2.erode(a, np.ones((9, 9))), 0, 1)[..., None]; L = over4(foam, L) if False else over4(L, foam * .9)
        c = Cel(); rr = np.random.default_rng(90 + f)
        for i in range(70): an = rr.uniform(0, 6.28); d = sc * rr.uniform(.8, 1.45); c.fill(ell(cx + math.cos(an) * d * .8, cy + math.sin(an) * d, *(rr.uniform(2, 9, 2)), 10), '#eafffb', .9)
        return over4(L, c.layer())

    def frame(self, f):
        z = f / 29.; sk = kit.shake(f, 13, 9, 2.4, 25) if f >= 25 else kit.shake(f // 2, 3, 9); sk = np.array([sk[0], -abs(sk[1])]); li = light('storm'); roll = key(f, [(0, .0), (4, .02), (12, -.06), (20, -.03), (25, .03), (29, .02)]); tw = key(f, [(0, 0.), (13, .75), (20, 1.0), (23, .85), (25, -.55), (29, -.95)], eout)
        im = view(self.sky, sk[0] * .3, sk[1] * .3 + 10 * z, 1.04, roll * .6); fl = {9: 1., 10: .25, 11: .6, 12: .15}.get(f, 0.)
        if fl: im = im + comp.radial(420, 140, 1100, 800, 1.6)[..., None] * rgb('#dff4ff') * fl * .55; im = self.bolt(im, f) if f in (9, 11) else im
        im = comp.over(im, view(self.wR, -190 + 70 * tw + sk[0] * .5, 70 + 130 * (1 - z) - 120 * max(0, tw) + 260 * max(0, -tw) + sk[1] * .5, 1.02, roll - .06 * tw)); wl = view(self.wL, 60 - 120 * tw + sk[0] * .6, 150 * (1 - z) - 110 * max(0, tw) + 220 * max(0, -tw) + sk[1] * .6, 1.05, roll + .05 * tw); im = comp.over(im, wl); im = comp.over(im, fx_air.mist(f, 3, 620, 1000, '#9fd2e0', .45, 60.))
        wh = self.whand(f)
        if wh is not None: im = kit.aura(im, wh, 30, '#7fd8ea', .35); im = comp.over(im, kit.shifted(wh, -sk[0], -sk[1]))
        rn = fx_air.rain(f, n=(240, 110, 0), wind=-.55, seed=5, col='#cfeaf6', near=False); im = comp.over(im, rn * .8)
        if f < 9: im = self.open(im, f)
        jb = self.jinbe(f, sk, li)
        jb = kit.shifted(jb, -sk[0] * 1.2, -sk[1] * 1.2); im = kit.cast(im, jb, 16, 22, '#04101e', .4, 14)
        fg = view(self.fg, sk[0], sk[1] - 30, 1.0, roll * .5); im = comp.over(im, jb) if f >= 0 else im
        im = self.refract(im, f) if 21 <= f <= 24 else im
        im = comp.over(im, fx_air.rain(f, n=(0, 40, 30), wind=-.55, seed=8, col='#e6f6ff')); im = self.spray(im, f)
        if f >= 25: im = self.hit(im, f, sk)
        return im

    def bolt(self, im, f):
        r = np.random.default_rng(3 + f); P = [np.array([520., -20.])]
        for i in range(12): P.append(P[-1] + [r.normal(-12, 46), 46])
        c = Cel(); c.line(np.array(P), 8, '#ffffff', taper=(.05, .7), press=.5); Q = P[4:7] + [P[6] + [r.normal(70, 30), 50 * i] for i in range(1, 4)]; c.line(np.array(Q), 4, '#ffffff', taper=(.1, .8)); L = c.layer(); return im + blur(L[..., 3], 18)[..., None] * rgb('#9fd8ff') * .9 + L[..., :3]

    def refract(self, im, f):
        """the push itself: a curved lens of water running out from the palm"""
        q = f - 21; cx, cy = [(545, 720), (566, 700), (604, 630), (640, 580)][q]; R = [120, 250, 390, 520][q]; x, y = grid(); dx, dy = x - cx, (y - cy) * 1.08; d = np.hypot(dx, dy) + 1e-3; a = np.arctan2(dy, dx); Rw = R * (1 + .045 * np.sin(a * 5 + f * 1.3) + .03 * np.sin(a * 9 - f)); u = (d - Rw) / (34. + 10 * q); am = -(20 + 5 * q) * u * np.exp(-u * u); im = cv2.remap(im.astype(f32), (x + dx / d * am).astype(f32), (y + dy / d * am).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); seg = np.clip(.25 + .9 * np.sin(a * 3 + 1.2 + f * .4), 0, 1); hi = np.exp(-((d - Rw - 16) / 4.2) ** 2) * seg + .55 * np.exp(-((d - Rw + 20) / 3.) ** 2) * (1 - seg); sh = np.exp(-((d - Rw + 8) / 12.) ** 2) * .20; im = im * (1 - sh[..., None]) + hi[..., None] * rgb('#e9fbff') * .8; c = Cel(); r = np.random.default_rng(60)
        for i in range(34): a_ = i * .185 + r.normal(0, .05); rr = R * (1.02 + .16 * r.random()); s_ = r.uniform(3, 12); P = np.array([[cx + math.cos(a_) * rr, cy + math.sin(a_) * rr / 1.08], [cx + math.cos(a_ + .05) * (rr + s_ * 2), cy + math.sin(a_ + .05) * (rr + s_ * 2) / 1.08], [cx + math.cos(a_ + .07) * (rr + s_ * 4.5), cy + math.sin(a_ + .07) * (rr + s_ * 4.5) / 1.08]]); c.fill(band_poly(P, [1., s_, s_ * .5], 6), ['#dff8fb', '#7fd0de', '#f6fffd'][i % 3], .9)
        return comp.over(im, c.layer())

    def spray(self, im, f):
        c = Cel(ss=1); r = np.random.default_rng(12)
        for i in range(90):
            p = (f * r.uniform(.05, .11) + r.random()) % 1.; x = -100 + 2200 * ((r.random() + p * .5) % 1); y = 1080 - 560 * p * r.uniform(.2, 1) + 260 * p * p; s = r.uniform(1.5, 7) * (1 - p * .5); c.fill(ell(x, y, s * 1.4, s, 9, rot=-.5), '#eafaff', .75 * (1 - p))
        return comp.over(im, c.layer())

    def open(self, im, f):
        """water level falls: the card is the sea on the lens"""
        cd = np.asarray(self.card, f32); lv = [-80, -40, 60, 210, 420, 660, 880, 1060, 1240][f]; x, y = grid(); s = lv + 34 * np.sin(x / 170. + f * .9) + 16 * np.sin(x / 63. - f * 1.6) + 70 * np.sin(x / 520. + f * .35)
        if f == 0: return cd.copy()
        c = Cel(ss=1); r = np.random.default_rng(8)
        for i in range(46):                                  # runs left on the glass
            xx = r.uniform(0, W); l0 = s[0, int(xx)] if s.ndim == 2 else lv; ln = r.uniform(40, 260); wd = r.uniform(5, 20); c.fill(band_poly(np.array([[xx, l0 + 10], [xx + r.normal(0, 8), l0 - ln * .5], [xx + r.normal(0, 8), l0 - ln]]), [wd * 1.6, wd, 2.]), '#2f86b4', .55); c.fill(ell(xx, l0 - ln - wd, wd * .55, wd * .8, 12), '#8fd6ea', .6)
        im = comp.over(im, c.layer() * (f < 8)); m = smooth(-1.5, 1.5, y - s); e = np.exp(-np.abs(y - s) / 5.); e2 = np.exp(-np.maximum(0, y - s) / 40.) * m; fo = smooth(.2, 1.0, fbm(H, W, (26, 9), (1, .6), 4)) * np.exp(-np.maximum(0, y - s) / 26.) * m; return lerp2(im, cd, m) + (e * .45 + fo * .55)[..., None] * rgb('#e6fbff') + e2[..., None] * rgb('#3f9cc4') * .28

    def hit(self, im, f, sk):
        q = f - 25; cx, cy = 652, 560
        if q == 0: im = kit.impact(im, 'bw', .14, .46, '#061a34', '#e6fbff'); c = Cel(); kit.focus_lines(c, cx, cy, 420, 1700, 150, 12, '#0d3158', 5, ry=.9); return comp.over(im, c.layer(), .8)
        R = [0, 300, 700, 1300, 2100][q]; p = Part(None, None, 9900, k=6, ink='#0a3a60'); r = np.random.default_rng(14)
        for j, (kk, col) in enumerate(((1.0, '#0f4f7e'), (.95, '#1a6f9e'), (.88, '#2f93b8'), (.74, '#7fd0de'), (.52, '#c9f4f2'), (.28, '#f6fffd'))):
            B = blob(cx + 30 * j * q * .3, cy - 10 * j, R * kk, 15 + j, spike=.30, lob=8, ry=.86); B = scallop(B, 70 - 6 * j, 24 - 2 * j, 4 + j)
            if j == 0: p.shape(B, col, .9)
            else: p.f.fill(B, col, .95)
        for i in range(70): a = r.uniform(0, 6.28); d = R * r.uniform(.5, 1.5); s_ = r.uniform(4, 30) * r.random() + 3; bead(p, cx + math.cos(a) * d, cy + math.sin(a) * d * .86, s_, a, 2.2, '#dff8fb', '#5fb4d0', .3)
        for i in range(40): a = r.uniform(0, 6.28); u0, u1 = R * r.uniform(.15, .5), R * r.uniform(.6, 1.0); p.f.line(np.array([[cx + math.cos(a) * u0, cy + math.sin(a) * u0 * .86], [cx + math.cos(a + .05) * (u0 + u1) / 2, cy + math.sin(a + .05) * (u0 + u1) / 2 * .86], [cx + math.cos(a + .12) * u1, cy + math.sin(a + .12) * u1 * .86]]), r.uniform(5, 22), ['#ffffff', '#1f7fac', '#8fe0ea'][i % 3], taper=(.4, .6))
        a_, ox, oy = p.done(); 
        if q < 2: ba = cv2.GaussianBlur(a_, (0, 0), .6)
        return comp.over(im, a_) if q < 4 else comp.over(im, a_)
