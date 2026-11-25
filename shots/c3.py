"""c3 Usopp: mustard card, shot open; the slingshot comes up into the lens, the pull, the release, the pellet takes the frame."""
from chars.gang_shot import *
from chars import usopp as U

PAL = dict(skin=U.C['skin'], shade=U.C['sh'], nail='#f0c99a', nail_shade='#b98a5c', light='#ffe2b0', cuff='#6b3a24', cuff_shade='#3d1f17', cuff_light='#9a5a34')
CX, CY = 960, 470


def mask_face(x, y, R, rot, k=1):
    """the sniper king's sun mask"""
    c, s = math.cos(rot), math.sin(rot); Q = lambda P: np.array([(x + (u * c - v * s) * R, y + (u * s + v * c) * R) for u, v in P]); out = []
    ray = []
    for i in range(32): a = i * math.pi / 16; r = 1.42 if i % 4 == 0 else 1.20 if i % 2 == 0 else .98; ray.append((math.cos(a) * r, math.sin(a) * r))
    out.append((Q(ray), '#d9582a')); out.append((Q([(u * .97, v * .97) for u, v in ell(0, 0, 1, 1, 48)]), '#7a2f18')); out.append((Q(ell(0, 0, .93, .93, 48)), '#f3dfa6')); cr = np.vstack([ell(0, 0, .925, .925, 24, -1.30, 1.30), spn([(.25, .89), (.52, .5), (.42, .05), (.56, -.5), (.25, -.89)], False)]); out.append((Q(cr), '#cfa862'))
    for sd in (-1, 1): out.append((Q(spn([(sd * .16, -.12), (sd * .34, -.28), (sd * .58, -.20), (sd * .50, -.02), (sd * .28, .02)], True)), '#2a1712')); out.append((Q(spn([(sd * .12, -.34), (sd * .40, -.52), (sd * .66, -.36), (sd * .40, -.44)], True)), '#8f4a22'))
    out.append((Q(ell(0, .12, .13, .15, 20)), '#2a1712')); out.append((Q(spn([(-.46, .42), (-.2, .52), (.2, .52), (.46, .42), (.3, .70), (0, .76), (-.3, .70)], True, (0, 3))), '#7a2f18')); return out


def make_sky():
    x, y = grid(OH, OW); cx, cy = OW / 2, OH / 2 - 60; a = np.arctan2(y - cy, x - cx); d = np.hypot(x - cx, (y - cy) * 1.15) / 900.
    ray = smooth(-.25, .25, np.sin(a * 22 + .5 * np.sin(a * 3))); c1 = ramp(d, [(0, '#fff3c0'), (.35, '#f7d06a'), (.8, '#dd9f2c'), (1.5, '#9a5d18')]); c2 = ramp(d, [(0, '#fff0b0'), (.35, '#eeb742'), (.8, '#bf7d1f'), (1.5, '#74400f')]); g = lerp2(c1, c2, ray)
    x2 = (d - .55) / .7; g = lerp2(g, g * np.array([.80, .55, .42], f32), np.clip(x2, 0, 1) * .75); hz = OH / 2 + 330; sea = ramp((y - hz) / 320., [(0, '#8fd0c0'), (.25, '#3f9aa0'), (1, '#175468')]); gl = smooth(.2, 1.2, fbm(OH, OW, (9, 4), (1, .5), 6, 6.)) * np.exp(-((x - cx) / 420.) ** 2); sea = sea + gl[..., None] * rgb('#fff0b0') * .6; g = lerp2(g, sea, smooth(hz - 2, hz + 2, y)); an = np.where(y > hz, 0., a + .05 * fbm(OH, OW, (160, 60), (1, .5), 4))
    return g * .38 + painted(g, 3, (52, 24), an, bristle=.06) * .62


def make_far():
    L = []
    hill = [(1240, 1300), (1320, 1010), (1480, 915), (1660, 880), (1840, 900), (2060, 870), (2400, 860), (2400, 1400)]; L.append((spn(hill, True, (0, 6, 7)) + [OW / 2 - 960, OH / 2 - 540], '#5d6a2a')); L.append((spn([(1300, 1300), (1380, 1080), (1560, 990), (1780, 960), (2100, 950), (2400, 930), (2400, 1400)], True, (0, 5, 6)) + [OW / 2 - 960, OH / 2 - 540], '#3d4a22')); L.append((spn([(-200, 1300), (-200, 980), (60, 940), (300, 985), (460, 1080), (540, 1300)], True, (0, 1, 5)) + [OW / 2 - 960, OH / 2 - 540], '#4c5a26'))
    def sh(c, a): x, y = grid(OH, OW); return np.clip(c * (1.12 - .25 * smooth(OH * .6, OH, y))[..., None] + .03 * fbm(OH, OW, (60, 20), (1, .5), 8)[..., None], 0, 1)
    P = plate_rgba(L, 7, (22, 11, 5), sh); c = Cel(OW, OH, ss=2); o = [OW / 2 - 960, OH / 2 - 540]
    for i, (p, col) in enumerate(mask_face(290, 400, 225, -.22)): c.fill(p + o, col); kit.outline(c, p + o, 4.5 if i < 3 else 3., '#4a1710', 40 + i)
    kit.hatch(c, mask_face(290, 400, 225, -.22)[3][0] + o, .9, 13, 2.2, '#8a5a2a', 3, 60, .5); l = c.rgba(); t = 1 + .05 * fbm(OH, OW, (40, 9, 3), (1, .6, .4), 9)[..., None]; l[..., :3] *= t; l[..., :3] = l[..., :3] * .84 + rgb('#f0b850') * .16 * l[..., 3:4]
    return over4(l, P)


class Shot:
    id = 'c3'; n = 30; finish = dict(bloom=.30, vig=.18)
    EVENTS = [(0, 'card', dict(note='mustard card')), (3, 'splat', dict(size=.5, note='card shot open, paint')), (7, 'whoosh', dict(note='slingshot swings up to the lens')), (9, 'creak', dict(note='rubber stretching f9-22, wood creak')), (14, 'crowd', dict(note='eight tiny people cheer')), (20, 'gulp', dict(note='sweat drop')), (25, 'impact', dict(size=1.0, note='twang + shout, DON')), (25, 'twang', dict(note='bands snap')), (26, 'whoosh', dict(note='pellet to lens, fills at f28'))]

    def setup(self):
        self.sky = cached('c3_sky', make_sky); self.far = cached('c3_far', make_far); self.card = cached('c3_card', lambda: card('#d9a12e', '#f0c65a', '#a36a1a', 31)); self.nxt = cached('c3_next', lambda: card('#15110e', '#b4842a', '#050404', 32, -.35))

    # ---- poses
    def pose(self, f):
        t = f if (f >= 22 or 5 <= f <= 9) else f // 2 * 2; tr = np.random.default_rng(40 + f // 2).normal(0, 1, 4) * (1 if 14 <= f <= 24 else 0)
        up = key(t, [(0, .0), (3, -.10), (5, .10), (7, .82), (9, 1.12), (12, 1.0)], lambda v: v)              # sling up to lens
        pull = key(t, [(9, 0.), (12, -.06), (17, .95), (20, 1.0), (24, 1.16)]) if t < 25 else {25: -.80, 26: .10, 27: -.16}.get(t, 0.)
        lean = key(t, [(0, .25), (3, .42), (8, -.10), (10, -.02), (12, -.10), (18, .34), (24, .62), (25, -.40), (28, -.2)])
        return dict(t=t, up=up, pull=pull, lean=lean, tr=tr)

    def people(self, p, f):
        T0 = f // 2
        for i in range(8):
            x = 1392 + i * 62 + (9 if i % 2 else -6); y0 = 944 - 42 * math.sin(min(1, i / 5.) * 1.5) + (i % 3) * 5; h = 44 + (i * 7) % 13; j = abs(math.sin(T0 * .95 + i * 1.7)) * 16; y = y0 - j; col = ['#5b2c1c', '#7a3a20', '#4a2a2e'][i % 3]; lean = .12 * math.sin(T0 * .9 + i)
            p.fill(spn([(x - h * .17, y - h * .62), (x + h * .17, y - h * .62), (x + h * .15, y - h * .25), (x + h * .12, y), (x + h * .03, y - h * .2), (x - h * .03, y - h * .2), (x - h * .12, y), (x - h * .15, y - h * .25)], True, (0, 1, 3, 6)), col); p.fill(ell(x + lean * 4, y - h * .80, h * .15, h * .17, 14), '#e7b47e'); p.fill(ell(x + lean * 4, y - h * .88, h * .16, h * .10, 12), col)
            for sd in (-1, 1): a = -1.1 - .5 * math.sin(T0 * 1.3 + i + sd); p.f.line(np.array([[x + sd * h * .15, y - h * .58], [x + sd * h * .32, y - h * .58 + math.sin(a) * h * .3], [x + sd * (h * .30 + 5 * math.cos(T0 + i)), y - h * .62 + math.sin(a) * h * .52]]), h * .10, col, taper=(0, .2))
            if i in (2, 5): fx = x + h * .34; wv = math.sin(T0 * 1.1 + i) * 6; p.f.line(np.array([[fx, y - h * .9], [fx + 2, y - h * 1.9]]), 3, '#3a2415'); p.fill(spn([(fx + 2, y - h * 1.9), (fx + 24, y - h * 1.86 + wv), (fx + 46, y - h * 1.78 - wv), (fx + 44, y - h * 1.42 - wv), (fx + 22, y - h * 1.46 + wv), (fx + 2, y - h * 1.4)], True, (0, 2, 3, 5)), '#c7392b'); p.fill(ell(fx + 23, y - h * 1.64, 6, 7, 12), '#f6e6b8')

    def usopp(self, f, q, li):
        t, up, pull, lean, tr = q['t'], q['up'], q['pull'], q['lean'], q['tr']; st = Stack(); u = max(0., min(1.2, up)); rel = f >= 25
        hx, hy = 1010 + 46 * lean + tr[0] * 1.5, 395 + 80 * (1 - min(1, u)) + 24 * lean + tr[1] * 1.5 - (22 if rel else 0); s = 204 + 8 * u; yw = key(t, [(0, -.30), (4, -.34), (8, -.52), (12, -.60), (24, -.68), (25, -.50), (28, -.56)]); T = Rig(hx, hy, s, yw, .02 + .06 * lean, .04 + .10 * lean)
        ro = .06 + .10 * lean; R = lambda x, y: np.array([hx + (x * math.cos(ro) - y * math.sin(ro)) * s, hy + (x * math.sin(ro) + y * math.cos(ro)) * s]); p = Part(T, None, 1500)
        # torso (local units of the head), turned away: chest far, shoulders wide
        B = p.shape([(-.40, 1.10), (-.95, 1.26), (-1.42, 1.50), (-1.50, 2.0), (-1.22, 2.6), (-1.1, 3.6), (1.15, 3.6), (1.25, 2.6), (1.50, 2.0), (1.42, 1.50), (.95, 1.26), (.40, 1.10)], U.C['skin'], 1.1, z=-.5); p.shade([(.1, 1.1), (.30, 1.7), (.1, 2.3), (.3, 3.8), (1.9, 3.8), (1.9, 1.2)], U.C['sh'], B, z=-.5)
        for sd in (-1, 1): Q = p.shape([(sd * .72, 1.16), (sd * 1.02, 1.26), (sd * .92, 2.3), (sd * .80, 3.7), (sd * .48, 3.7), (sd * .62, 2.3)], '#a3292b', .8, z=-.48, co=(0, 1, 3, 4)); p.shade([(sd * .9, 1.1), (sd * 1.2, 1.2), (sd * 1.0, 3.8), (sd * .7, 3.8)], '#64151f', Q, z=-.48)
        p.line([(-.30, 1.34), (-.08, 1.46), (-.01, 1.42)], .6, z=-.5); p.line([(.30, 1.34), (.08, 1.46), (.03, 1.42)], .6, z=-.5); p.ticks((.35, 1.7), (.55, 2.3), 7, .3, .18); p.line([(0, 1.8), (.02, 2.6)], .4, z=-.5)
        # pulling arm (viewer right): shoulder -> elbow out and up -> pouch
        pouch = self.pouch(q); sh = R(1.18, 1.52); el = sh * .5 + pouch * .5 + np.array([300 + 60 * max(0, pull), 150 - 190 * max(0, pull)]) if u > .3 else sh + [120, 250]
        wr = pouch + np.array([52, 14]) if u > .3 else np.array([hx + 60, hy + 360])
        if rel: wr = np.array([1350., 520.]) + [10 * (f - 25), -8 * (f - 25)]; el = sh * .45 + wr * .55 + [190, 110]
        ua = union(capsule(sh, el, s * .34, s * .21, s * .05), capsule(el, el * .5 + wr * .5, s * .21, s * .22)); ua = p.shape(ua, U.C['skin'], 1.1); p.shade(ua + [s * .05, s * .20], U.C['sh'], ua); p.line(np.array([sh * .75 + el * .25 + [-10, -s * .10], sh * .5 + el * .5 + [0, -s * .02], sh * .28 + el * .72 + [8, -s * .06]]), .45)
        st.add(p, li)
        # head
        ex = 'worry' if t < 4 else {'base': 'calm', 'bi': .02, 'mw': .26} if t < 8 else dict(U.EX['aim'], tongue_out=1.) if t < 24 else ({'base': 'shout'})
        e = pick(U.EX, ex if not isinstance(ex, dict) or 'base' in ex else 'aim')
        if 8 <= t < 24: e['tongue_out'] = 1.; e['md'] = .05; e['mt'] = .6; e['lx'] = .55; e['ly'] = .10; e['eo'] = .70 if t < 18 else .86; e['bi'] = .07 + .03 * max(0, pull)
        wk = -1. if 9 <= t < 25 else 0.
        wind = key(f, [(0, .1), (8, .55), (14, .15), (22, -.05), (24, -.35), (25, 1.5), (26, 1.8), (29, 1.5)]) + .12 * math.sin(f // 2 * .8)
        lay = over4(st.a, U.draw(T, li, e, f, wink=wk, wind=wind, nose=1.34)); st = Stack(); st.a = lay; p = Part(T, None, 1600)
        # sweat
        if 10 <= f < 25: k_ = ((f - 10) // 2 * 2) / 14.; b0 = R(.30 + .06 * k_, -.10 + .50 * k_); bead(p, b0[0], b0[1], 11 + 7 * k_, math.pi / 2, 1.8, '#e8f7fb')
        # forearm + pulling hand
        fa = capsule(el, wr, s * .205, s * .135, s * .055); fa = p.fill(fa, U.C['skin']); p.ol(fa, 1.05, gaps=[(.60, .80)]); p.shade(fa + [0, s * .13], U.C['sh'], fa); p.ticks((0, 0), (.01, .01), 1, .01, .001) if 0 else None; d = (wr - el) / (np.hypot(*(wr - el)) + 1e-6); cu = wr - d * s * .22
        WB = p.shape(band_poly(np.array([cu - d * s * .16, cu, cu + d * s * .14]), [s * .40, s * .36, s * .33]), '#6b3a24', .9); p.shade(WB + [0, s * .12], '#3d1f17', WB)
        p.ticks((.0, 0), (.0, .01), 1, .01, .001) if 0 else None
        if u > .25: hand(p, 'fist' if not rel else 'open', wr[0] - 10, wr[1] - 4, s * .34, HD.euler(60 if not rel else 110, 12, 78 + 12 * pull), PAL, 'L', 6., w=s * .017, res=300)
        st.add(p, li); return st.a

    def geom(self, q):
        u = max(.0, min(1.25, q['up'])); k = .30 + .70 * u; rel = q['t'] >= 25; k *= 1.10 if rel else 1.
        cx, cy = 770 + 190 * min(1, u) + q['tr'][2] * 2, 960 - 70 * u + q['tr'][3] * 2 + (26 if rel else 0); rot = -.27 - .10 * (1 - min(1, u)) + (.05 if rel else 0.)
        c, s = math.cos(rot), math.sin(rot); P = lambda x, y: np.array([cx + (x * c - y * s) * k, cy + (x * s + y * c) * k]); return P, k

    def pouch(self, q):
        P, k = self.geom(q); a = (P(-500, -600) + P(500, -600)) / 2 + np.array([190., 250.]) * min(1., k) ** 2; tgt = np.array([1196., 566.]) + q['tr'][:2] * 2; n = max(-.7, q['pull']); r = np.array([960., 470.]); return a * (1 - n) + tgt * n if n >= 0 else a + np.array([-230., -190.]) * min(1., -n * 1.25)

    def sling(self, f, q, li):
        """the fork, huge at the lens"""
        P, k = self.geom(q); p = Part(Rig(960, 540, 240 * min(1, k)), None, 1700, k=7.0 * min(1., k) ** .6); WD, WS, WH = '#5a3a22', '#2c1a14', '#a9753f'; pouch = self.pouch(q); rel = q['t'] >= 25; st = Stack()
        tipL, tipR = P(-524, -596), P(524, -596); qv = 0. if rel else (math.sin(f * 2.4) * 5 * max(0, q['pull']) if 14 <= f <= 24 else 0.)
        # bands behind the fork front: near tips wide, pouch narrow
        for sd, tp in ((-1, tipL), (1, tipR)):
            en = pouch + [sd * 14, 0]; dv = en - tp; nv = np.array([-dv[1], dv[0]]) / (np.hypot(*dv) + 1e-6); tt = np.linspace(0, 1, 9 if rel else 3); amp = {25: 62., 26: 46., 27: 30., 28: 20., 29: 12.}.get(f, 0.) * k; wv = (amp * np.sin(tt * 9.4 + f * 2.1 + sd) * np.sin(np.pi * tt) ** .8) if rel else np.array([0., qv, 0.]); BP = tp + dv * tt[:, None] + nv * wv[:, None]; wd_ = np.interp(tt, [0, .5, 1], [64 * k, 44 * k, 22 if not rel else 34 * k]) * (1.25 if rel else 1.); B = band_poly(BP, list(wd_), 8); p.shape(B, '#e8d9a8', .8); p.shade(B + [0, 13 * k], '#a98f55', B); p.f.line(spn([tuple(v) for v in BP], False) + [0, -7 * k], 5 * k, '#fffbe2', taper=(.1, .4))
            if rel and f < 28: [p.l.line(spn([tuple(v) for v in BP], False) + nv * g_, 2.2, '#7a5a24', taper=(.3, .5), alpha=.55) for g_ in (-44 * k, 46 * k)]
        if not rel: p.shape(spn([pouch + [-40, -26], pouch + [0, -34], pouch + [44, -22], pouch + [40, 26], pouch + [0, 34], pouch + [-40, 24]], True), '#7a4424', .8); p.shape(ell(pouch[0] - 8, pouch[1], 24, 24, 24), '#2a2a30', .6); p.f.fill(ell(pouch[0] - 15, pouch[1] - 8, 7, 6, 12), '#c9d2de')
        st.add(p, li); p = Part(Rig(960, 540, 240 * min(1, k)), None, 1720, k=7.0 * min(1., k) ** .6)
        for sd in (-1, 1):
            A = band_poly(np.array([P(sd * 20, -120), P(sd * 190, -215), P(sd * 420, -300), P(sd * 528, -450), P(sd * 520, -660)]), [104 * k, 116 * k, 122 * k, 132 * k, 124 * k], 12); p.shape(A, WD, 1.1); p.shade(A + [26 * k, 22 * k], WS, A); p.f.clip(A); p.f.line(spn([P(sd * 180 - 20, -235), P(sd * 410 - 30, -322), P(sd * 506 - 34, -460), P(sd * 498 - 34, -640)], False), 13 * k, WH, taper=(.3, .3)); [p.l.line(spn([P(sd * (200 + 20 * j), -205 + 16 * j), P(sd * (400 + 18 * j), -300 + 24 * j), P(sd * (520 + 16 * j), -470), P(sd * (516 + 12 * j), -560)], False), .30 * p.k, INK, taper=(.4, .5), alpha=.8) for j in (-1, 1)]; p.f.unclip()
            for j, yy in enumerate((-590, -556, -524)): tb = band_poly(np.array([P(sd * 526 - 76, yy + 4), P(sd * 526, yy + 16), P(sd * 526 + 76, yy - 4)]), 34 * k); p.shape(tb, '#efe2b8' if j != 1 else '#d8c48c', .55)
            C_ = p.shape(spn([P(sd * 520 - 68, -640), P(sd * 520 - 56, -706), P(sd * 520, -734), P(sd * 520 + 56, -706), P(sd * 520 + 68, -640), P(sd * 520, -620)], True, (0, 4)), '#d7b65c', .9); p.shade(C_ + [24 * k, 12 * k], '#8a5f22', C_)
        S_ = band_poly(np.array([P(0, -180), P(0, 20), P(6, 420)]), [150 * k, 118 * k, 126 * k], 8); p.shape(S_, WD, 1.1); p.shade(S_ + [34 * k, 0], WS, S_); J = p.shape(spn([P(-120, -226), P(0, -176), P(120, -226), P(88, -86), P(0, -62), P(-88, -86)], True, (0, 2)), '#d7b65c', .9); p.shade(J + [30 * k, 16 * k], '#8a5f22', J)
        [p.l.line(np.array([P(-20 + 24 * j, -50), P(-18 + 22 * j, 60)]), .3 * p.k, INK) for j in range(3)]
        st.add(p, li, 9); p = Part(Rig(960, 540, 240), None, 1740, k=6.5 * min(1., k) ** .5)
        # forearm from the shoulder up to the lens, then the fist round the grip
        wr_ = P(-92, 190); fa = capsule([752., 790.], wr_, 84, 150 * k, 22); fs = sides([752., 790.], wr_, 84, 150 * k, 22)
        if k > .5: p.fill(fa, U.C['skin']); [p.line(q_[3:], 1.1, taper=(.35, .1)) for q_ in fs]; p.shade(fa + [74 * k, -40], U.C['sh'], fa); p.hatch(fa + [100 * k, -60], 1.0, 13, .28, fade=.6, within=fa); cu = band_poly(np.array([P(-250, 120), P(-150, 60), P(-40, -20)]) + [0, 0], 86 * k, 10); cu = p.shape(cu, '#6b3a24', 1.); p.shade(cu + [16 * k, 30 * k], '#3d1f17', cu); [p.f.fill(ell(*P(x, y), 11 * k, 11 * k, 12), '#d7b65c') for x, y in ((-205, 94), (-128, 44))]
        hand(p, 'fist', *P(0, 140), 250 * k, HD.euler(180, 74, 84), PAL, 'R', 4.0, w=6.4 * min(1, k), res=460, pivot=None) if 0 else hand(p, 'fist', *P(-6, 66), 262 * k, HD.euler(180, 72, 86), PAL, 'L', 4.2, w=6.2 * min(1, k) ** .7, res=460)
        st.add(p, li, 9); return st.a

    def frame(self, f):
        q = self.pose(f); li = light('sun'); z = 1.0 + .0022 * f + (.02 if f >= 25 else 0); sk = kit.shake(f, 9, 3, 2.2, 25) if f >= 25 else np.zeros(2)
        rot = .0012 * f; im = view(self.sky, sk[0], sk[1], z, rot); fl = view(self.far, 14 * (1 - f / 29.) + sk[0], sk[1], z * (1 + .0015 * f)); im = comp.over(im, fl); pp = Part(None, (1330, 760, 1920, 1080), 1800, k=2.2); self.people(pp, f); a, ox, oy = pp.done(); L = np.zeros((H, W, 4), f32); L[oy:oy + a.shape[0], ox:ox + a.shape[1]] = a; im = comp.over(im, warp_layer(L, [[z, 0, 960 * (1 - z) - sk[0]], [0, z, 540 * (1 - z) - sk[1]]]))
        if f < 9:                                             # card shot open
            cd = np.asarray(self.card, f32); R = [0, 0, 0, 150, 430, 760, 1080, 1420, 1800][f]
            if R == 0: im = cd.copy()
            else: B = blob(1000, 500, R, 5, ry=.8); m = mask_soft(B); B2 = blob(1000, 500, R + 34, 5, ry=.8); m2 = mask_soft(B2); e = np.clip(m2 - m, 0, 1); cd2 = cd * (1 - .25 * e[..., None]) + rgb('#fff3c8') * (e * .75)[..., None]; im = lerp2(cd2, im, m)
        else:
            m = radial_cache(); im = im * (1 + .10 * m[..., None])
        us = self.usopp(f, q, li); im = kit.cast(im, us, 26, 30, '#4a2a10', .35, 10); im = comp.over(im, us); sl = self.sling(f, q, li)
        if f in (5, 6, 7): sl = over4(sm(sl, 30, 170), sl * .62)
        im = kit.cast(im, sl, 14, 34, '#3a1c10', .30, 16); im = comp.over(im, sl)
        if 14 <= f < 25: c = Cel(); kit.focus_lines(c, 1000, 480, 760 - 12 * (f - 14), 1500, 70, 6, '#7a4410', 60 + f // 2, ry=.62); l = c.layer(); im = comp.over(im, l, .55)
        if 12 <= f < 25: zz = 1 + .0065 * (f - 12); im = comp.cam2d(im, 960, 470 + 70 / zz, zz); im = im * (1 - .16 * (f - 12) / 12. * (1 - radial_cache()[..., None]))
        if f >= 25: im = self.release(im, f, q)
        return im

    def release(self, im, f, q):
        P, k = self.geom(q); c = Cel(); kit.focus_lines(c, 900, 520, 620 if f == 25 else 520, 1700, 190, 12, '#fff4cc' if f > 25 else '#17131a', 80 + f, ry=.7); im = comp.over(im, c.layer(), .9); r = [86, 236, 640, 1280, 2600][f - 25]; x, y = [(760, 560), (820, 560), (900, 550), (960, 540), (960, 540)][f - 25]
        if f == 25: im = kit.impact(im, 'bw', .22, .60, '#1a0f08', '#ffe9a8')
        nx = np.asarray(self.nxt, f32)
        if f >= 28: B = mask_soft(ell(x, y, r, r, 90)); im = lerp2(im, nx, B); return im
        p = Part(None, None, 1900, k=7); B = ell(x, y, r, r, 90); p.shape(B, '#1b1a20', 1.2); p.shade(ell(x - r * .12, y - r * .14, r * .92, r * .90, 80), '#2d2c36', B); p.shade(ell(x - r * .20, y - r * .22, r * .70, r * .66, 80), '#43424f', B); p.f.clip(B); p.f.fill(ell(x - r * .36, y - r * .40, r * .20, r * .13, 30, rot=-.7), '#fdfdff'); p.f.line(ell(x, y, r * .94, r * .94, 40, .2, 1.6), r * .05, '#e8b24a', taper=(.4, .4)); p.f.unclip()
        a, ox, oy = p.done(); im = im + comp.radial(x, y, r * 1.5, power=1.6)[..., None] * rgb('#ffcf6a') * .5; im = comp.over(im, a)
        for i in range(3): rg = ring_poly(x, y, r * (1.25 + .28 * i), r * .06, 1., 11 + i); cc = Cel(); cc.fill(rg, '#fff6d8', evenodd=True); im = comp.over(im, cc.layer(), .7 - .2 * i)
        return im


_rc = None
def radial_cache():
    global _rc
    if _rc is None: _rc = comp.radial(960, 420, 900, power=1.5)
    return _rc
