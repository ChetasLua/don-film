"""c5 Chopper: pink card swept away by petals; the drum peak and its pink cloud stand like one enormous cherry tree; he looks up, wipes his eyes, grins."""
from chars.gang_shot import *
from chars import chopper as K
from op import fx_air

SNOW = dict(lit='#fff0f4', mid='#f6a9c2', dark='#c86f9c', ink='#7a3360'); VIO = dict(lit='#c9a8f0', mid='#7d55b8', dark='#4a2a80', ink='#2a1450')
TREE = (1262, 372)


def make_sky():
    x, y = grid(OH, OW); t = y / OH; g = ramp(t, [(0, '#160d2e'), (.30, '#34194f'), (.58, '#7a3a74'), (.78, '#d0749a'), (.90, '#f7b4bd'), (1, '#ffd9cf')]); r = comp.radial(OW / 2 + TREE[0] - 960, OH / 2 + TREE[1] - 540, 1000, 700, 1.4, OH, OW); g = g + r[..., None] * rgb('#ff8fb8') * .55
    g = painted(np.clip(g, 0, 1), 5, (50, 24), .15 * np.sin(y / 180.) + .2 * fbm(OH, OW, (200,), None, 3)) * .65 + g * .35; rr = np.random.default_rng(4); c = Cel(OW, OH, ss=1)
    for _ in range(150): xx, yy = rr.uniform(0, OW), rr.uniform(0, OH * .5) ** 1; s = rr.uniform(.6, 2.2) ** 1.3; c.fill(ell(xx, yy, s, s, 8), '#ffe6f0', rr.uniform(.3, .9))
    return comp.over(np.clip(g, 0, 1.2), c.rgba())


def make_canopy():
    """cumulus of pink powder = the crown of a cherry tree, lit from inside"""
    o = np.array([OW / 2 - 960, OH / 2 - 540]); r = np.random.default_rng(12); cx, cy = TREE + o; L = []
    for lev, n, (r0, r1) in ((0, 34, (120, 190)), (1, 110, (60, 110)), (2, 260, (24, 56))):
        for _ in range(n):
            a = r.uniform(0, 6.28); q = math.sqrt(r.random()) * (.92 if lev == 0 else 1.0); X = cx + math.cos(a) * 560 * q * (1.02 + .05 * math.sin(a * 5)); Y = cy - 70 + math.sin(a) * (250 if math.sin(a) < 0 else 120) * q + 26 * math.sin(a * 4); L.append((lev, X, Y, r.uniform(r0, r1) * (1.15 - .35 * q)))
    img = np.zeros((OH, OW, 3), f32); A = np.zeros((OH, OW), f32); x, y = grid(OH, OW); Z = np.full((OH, OW), -1e9, f32); NX = np.zeros((OH, OW), f32); NY = np.zeros((OH, OW), f32)
    for lev, X, Y, R in L:
        x0, x1, y0, y1 = max(0, int(X - R)), min(OW, int(X + R) + 1), max(0, int(Y - R)), min(OH, int(Y + R) + 1)
        if x1 <= x0 or y1 <= y0: continue
        u = (x[y0:y1, x0:x1] - X) / R; v = (y[y0:y1, x0:x1] - Y) / (R * .88); d = u * u + v * v; z = np.sqrt(np.maximum(0, 1 - d)) * R + lev * 60 + (Y - cy) * .25; m = (d < 1) & (z > Z[y0:y1, x0:x1]); Z[y0:y1, x0:x1][m] = z[m]; NX[y0:y1, x0:x1][m] = u[m]; NY[y0:y1, x0:x1][m] = v[m]; A[y0:y1, x0:x1] = np.maximum(A[y0:y1, x0:x1], (d < 1) * 1.)
    core = np.exp(-(((x - cx) / 300.) ** 2 + ((y - cy + 20) / 150.) ** 2)); lit = -.62 * NY - .40 * NX + .38 * core + .10 * fbm(OH, OW, (40, 14), (1, .5), 8); lit = cv2.GaussianBlur(lit.astype(f32), (0, 0), 1.6)
    c = ramp(lit, [(-.8, '#6a2c70'), (-.35, '#8f3f84'), (-.30, '#b4558f'), (.10, '#d9709f'), (.16, '#f3a0bd'), (.55, '#fbbccd'), (.62, '#ffe0e6'), (1.2, '#fff6f0')]); a = cv2.GaussianBlur(A, (0, 0), 1.2); a = paint.warp_by(a, 5, 26, 3)
    base = cv2.inpaint((np.clip(c[::4, ::4], 0, 1) * 255).astype(np.uint8), (A[::4, ::4] < .5).astype(np.uint8), 4, cv2.INPAINT_TELEA); c = np.where(A[..., None] > .5, c, cv2.resize(base, (OW, OH)).astype(f32) / 255.); c = painted(np.clip(c, 0, 1).astype(f32), 9, (20, 10, 5)) * .7 + c * .3
    # the doctor, in light: tall hat, coat thrown open, cup up
    dx, dy = cx + 262, cy + 46; S = 1.62; D = lambda P: np.array([(dx + u * S, dy + v * S) for u, v in P]); ce = Cel(OW, OH, ss=1)
    body = spn([(-28, -22), (-46, -8), (-54, 40), (-72, 96), (70, 100), (52, 40), (60, 2), (96, -34), (112, -74), (98, -80), (82, -50), (44, -20), (26, -24), (14, -32), (-14, -32)], True, (0, 3, 4, 8, 9, 14)); head_ = spn([(-26, -36), (-30, -62), (-26, -86), (26, -86), (30, -62), (24, -38), (0, -28)], True); hat = spn([(-58, -84), (-34, -94), (-30, -150), (-36, -176), (0, -184), (36, -176), (30, -150), (34, -94), (58, -84), (0, -76)], True, (0, 3, 5, 8)); cup = spn([(86, -84), (124, -84), (116, -100), (124, -112), (86, -112), (94, -100)], True, (0, 1, 3, 4)); crossL = spn([(-30, -66), (-66, -62), (-64, -50), (-30, -50)], True, (0, 1, 2, 3)); crossR = spn([(30, -66), (66, -62), (64, -50), (30, -50)], True, (0, 1, 2, 3))
    for P in (body, head_, hat, cup, crossL, crossR): ce.fill(D(P), '#ffffff')
    m = ce.rgba()[..., 3]; m = paint.warp_by(m, 2, 26, 6); soft = cv2.GaussianBlur(m, (0, 0), 2.0); tex = smooth(-1.6, .4, fbm(OH, OW, (30, 10, 5), (1, .6, .4), 5)); ha = cv2.GaussianBlur(m, (0, 0), 22); c = c * np.where(True, 1, 1) + (rgb('#c8588f') - c) * (np.clip(ha * 1.2, 0, 1) * .35)[..., None]; c = c * (1 - (np.clip(cv2.GaussianBlur(m, (0, 0), 40) * 1.5, 0, 1) * .42)[..., None]); c = c + (rgb('#fffaf0') - c) * (soft * (.70 + .28 * tex))[..., None]
    f2 = Cel(OW, OH, ss=1); f2.line(D(spn([(-16, -46), (0, -37), (18, -47)], False)), 7, '#c8688f', taper=(.2, .2)); f2.line(D(spn([(-18, -64), (-11, -70), (-4, -64)], False)), 5, '#c8688f'); f2.line(D(spn([(6, -64), (13, -70), (20, -64)], False)), 5, '#c8688f'); f2.line(D(spn([(-44, -88), (0, -80), (44, -88)], False)), 5, '#d98aa8'); f2.line(D(spn([(-30, 10), (-10, 60), (-14, 96)], False)), 5, '#e0a0b8'); f2.line(D(spn([(24, 6), (14, 60), (20, 96)], False)), 5, '#e0a0b8'); c = comp.over(c, f2.rgba(), .8)
    return np.dstack([c * a[..., None], a])


def make_peaks():
    o = np.array([OW / 2 - 960, OH / 2 - 540]); L = []
    def drum(cx, top, bot, w0, w1, cols, sd):
        r = np.random.default_rng(sd); t = np.linspace(0, 1, 9); n1 = r.normal(0, w0 * .035, 9); Lf = np.c_[cx - (w0 + (w1 - w0) * t ** 2.2) + n1, top + (bot - top) * t]; Rt = np.c_[cx + (w0 + (w1 - w0) * t ** 2.2) + n1[::-1], top + (bot - top) * t]; L.append((np.vstack([Lf, Rt[::-1]]) + o, cols[1])); k = int(5 + w0 / 22)
        for i in range(k): u = -1 + 2 * (i + .5) / k; ww = w0 * 2 / k * r.uniform(.25, .7); xx = cx + u * w0; y1 = top + (bot - top) * r.uniform(.6, 1); L.append((np.array([[xx - ww / 2, top + 10], [xx + ww / 2, top + 10], [xx + ww / 2 + (w1 - w0) * u * .5, y1], [xx - ww / 2 + (w1 - w0) * u * .5, y1]]) + o, cols[2] if u < .1 else cols[0]))
        L.append((np.array([[cx + w0 * .30, top], [cx + w0, top], [cx + w1, bot], [cx + w0 * .55, bot]]) + o, cols[0])) if 0 else None; L.append((spn([(cx - w0 * 1.08, top + 8), (cx - w0 * .9, top - 14), (cx, top - 20), (cx + w0 * .9, top - 14), (cx + w0 * 1.08, top + 8), (cx + w0 * .7, top + 26), (cx + w0 * .3, top + 14), (cx - w0 * .1, top + 34), (cx - w0 * .5, top + 14), (cx - w0 * .8, top + 30)], True) + o, '#ffd6de'))
        for i in range(7): yy = top + (bot - top) * r.uniform(.12, .85); xx = cx + r.uniform(-.8, .8) * w0; L.append((spn([(xx - 40, yy), (xx, yy - 8), (xx + 44, yy + 2), (xx + 6, yy + 10)], True) + o, '#f0a9bf'))
    drum(250, 640, 1180, 120, 190, ('#8a4a86', '#5f2f6e', '#48235c'), 2); drum(610, 700, 1180, 70, 120, ('#9a5590', '#6f3a7c', '#552a68'), 5); drum(1830, 610, 1180, 110, 180, ('#8a4a86', '#5c2d6b', '#452158'), 4); drum(TREE[0], 455, 1200, 135, 265, ('#b0608f', '#5d2a62', '#3f1b4c'), 3)
    # the round castle on the trunk
    cx = TREE[0]; by = 440
    for dx, w, h, roof, col in ((-84, 26, 70, 44, '#6f3a78'), (84, 26, 64, 44, '#b7709c'), (0, 74, 92, 0, '#8f4e86'), (-34, 22, 132, 46, '#7f4480'), (36, 22, 120, 46, '#c487a8'), (0, 24, 168, 60, '#a5628f')):
        X = cx + dx; L.append((np.array([[X - w, by - h], [X + w, by - h], [X + w * 1.06, by + 8], [X - w * 1.06, by + 8]]) + o, col)); L.append((np.array([[X - w, by - h], [X - w * .2, by - h], [X - w * .2, by + 8], [X - w * 1.06, by + 8]]) + o, '#5f2f6c'))
        if roof: L.append((np.array([[X - w * 1.3, by - h + 3], [X, by - h - roof], [X + w * 1.3, by - h + 3]]) + o, '#4a2060')); L.append((np.array([[X + 2, by - h], [X, by - h - roof], [X + w * 1.3, by - h + 3]]) + o, '#ffc3d3')); 
        else: L.append((spn([(X - w * 1.05, by - h), (X, by - h - 34), (X + w * 1.05, by - h)], True, (0, 2)) + o, '#ffd0da'))
        for yy in range(int(by - h + 18), by - 8, 30): L.append((ell(X + w * .3, yy, 4, 6, 8) + o, '#fff0b8'))
    L.append((np.array([[cx - 2, by - 230], [cx + 1, by - 230], [cx + 1, by - 300], [cx - 2, by - 300]]) + o, '#3a1a48')); L.append((spn([(cx, by - 300), (cx + 36, by - 292), (cx + 64, by - 300), (cx + 60, by - 270), (cx + 30, by - 264), (cx, by - 272)], True, (0, 2, 3, 5)) + o, '#f48fb0'))
    return plate_rgba(L, 14, (18, 9, 4), None, 1.2, np.full((OH, OW), 1.5, f32))


def make_ground():
    o = np.array([OW / 2 - 960, OH / 2 - 540]); L = [(spn([(-300, 1400), (-300, 900), (0, 868), (300, 905), (700, 876), (1100, 918), (1500, 880), (1960, 900), (2300, 870), (2300, 1400)], True, (0, 1, 8, 9)) + o, '#e79ab8'), (spn([(-300, 1400), (-300, 968), (200, 940), (640, 985), (1000, 950), (1500, 990), (2300, 940), (2300, 1400)], True, (0, 1, 6, 7)) + o, '#c478a8'), (spn([(-300, 1400), (-300, 1060), (300, 1010), (900, 1048), (1500, 1020), (2300, 1070), (2300, 1400)], True, (0, 1, 5, 6)) + o, '#f3b6c8')]; r = np.random.default_rng(9)
    for x, b, h in ((60, 900, 250), (150, 915, 190), (-20, 930, 330), (360, 890, 130), (1700, 905, 210), (1790, 915, 300), (1880, 925, 250), (1960, 940, 340), (1120, 895, 90), (1040, 892, 70)):
        for j in range(5):
            t0 = j / 5.; w = h * (.10 + .20 * t0); yy = b - h + h * t0 * .95; L.append((np.array([[x, yy - h * .05], [x - w, yy + h * .26], [x - w * .4, yy + h * .22], [x, yy + h * .28], [x + w * .5, yy + h * .22], [x + w, yy + h * .26]]) + o, '#3a1c4e')); L.append((np.array([[x + 2, yy - h * .05], [x + w * .25, yy + h * .10], [x + w, yy + h * .26], [x + w * .5, yy + h * .14], [x + w * .1, yy + h * .1]]) + o, '#ffc9d6'))
    def sh(c, a): x, y = grid(OH, OW); return np.clip(c * (1.08 - .30 * smooth(OH * .72, OH * .95, y))[..., None] + .018 * fbm(OH, OW, (50, 18), (1, .5), 8, 3.)[..., None], 0, 1)
    return plate_rgba(L, 15, (22, 10, 5), sh)


class Shot:
    id = 'c5'; n = 30; finish = dict(bloom=.34, vig=.20, halation='#ffd0dc')
    EVENTS = [(0, 'card', dict(note='pink card, hush')), (2, 'wind', dict(note='petals sweep the card away f2-8')), (5, 'breath', dict(note='small gasp, looks up')), (11, 'cloth', dict(note='forearm wipe 1')), (16, 'cloth', dict(note='forearm wipe 2, sniff')), (20, 'sniff', {}), (25, 'impact', dict(size=.7, note='grin, hat jumps, soft DON; doctor cup clinks')), (25, 'clink', dict(note='two cups, far')), (26, 'wind', dict(note='petal gust to lens, turns violet'))]

    def setup(self):
        self.sky = cached('c5_sky', make_sky); self.can = cached('c5_can', make_canopy); self.pk = cached('c5_peaks', make_peaks); self.gr = cached('c5_ground', make_ground); self.card = cached('c5_card', lambda: card('#d7799f', '#f3aac2', '#a04f85', 51, -.35)); self.nxt = cached('c5_next', lambda: card('#3d2470', '#7a4fb8', '#22123f', 61, -.6))

    def ik(self, a, b, l1, l2):
        """elbow of a two-bone arm, always bending out and down to the viewer's left"""
        d = b - a; D = min(np.hypot(*d), (l1 + l2) * .985); d = d / (np.hypot(*d) + 1e-6); c = np.clip((l1 * l1 + D * D - l2 * l2) / (2 * l1 * D), -1, 1); u = l1 * c; v = l1 * math.sqrt(1 - c * c); n = np.array([-d[1], d[0]]); n = n if n @ np.array([-1., .35]) > 0 else -n; return a + d * u + n * v

    def chopper(self, f, hx, hy, s, li):
        t = f if f >= 20 else f // 2 * 2; st = Stack(); look = key(t, [(0, 0.), (4, -.15), (8, 1.), (10, 1.)]) * (1 if t < 12 else key(t, [(12, 1.), (15, .45), (19, .25), (21, -.25), (24, .55), (27, .45)]))
        w = key(t, [(9, 0.), (11, .0), (12, .30), (15, 1.)]) if t < 16 else key(t, [(15, 1.), (19, .12), (20, .05)]); armup = key(t, [(8, 0.), (11, 1.)]) if t < 20 else key(t, [(20, 1.), (22, .0)])
        b = {24: -.05, 25: .16, 26: .10, 27: -.03, 28: .02, 29: 0.}.get(f, 0.); dip = key(t, [(0, .20), (5, .24), (9, -.02), (11, .02), (13, .08), (19, .10), (21, .16), (22, .18), (24, -.06), (26, .0)])
        yaw = key(t, [(0, -.1), (8, -.22), (12, -.05), (15, .16), (19, -.18), (22, -.05), (25, -.12)]); pit = .30 * look - .30 * dip; roll = key(t, [(0, .03), (8, -.06), (12, -.02), (15, .08), (19, -.07), (22, .02), (25, -.07), (29, -.05)])
        ex = 'calm' if t < 4 else ('calm', 'brim', min(1., (t - 4) / 4.)) if t < 10 else 'cry' if t < 20 else ('cry', 'grin', [.0, .3, .5, .8, 1.][min(4, f - 20)]) if f < 25 else 'grin'
        e = pick(K.EX, ex)
        if 10 <= t < 20: e['md'] = .22 + .04 * (t % 4); e['mw'] = .34; e['snot'] = 1.; e['tear'] = .8
        if t < 4: e['ly'] = .5; e['eo'] = .75
        if f >= 25: e['md'] = .44 - .03 * (f - 25) * (f < 28); e['tear'] = .9
        sq = (1 + .04 * (f == 24) - .03 * (f == 25), 1 - .05 * (f == 24) + .05 * (f == 25)); T = Rig(hx, hy + dip * s * .25, s, yaw, pit, roll, .10, r=(1.0, 1.08, .86), o=(0, .1), sx=sq[0], sy=sq[1]); Y0 = hy + dip * s * .25
        B = lambda x, y: np.array([hx + x * s * .92 + roll * s * .3 * (y - 1), hy + s * .96 + (y - 1) * s * .95]); hit = 1 if f >= 24 else 0
        # body: pack, legs, shorts, trunk
        p = Part(T, None, 3400); bt = B(0, 1.0)
        for sd in (-1, 1):
            lg = capsule(B(sd * .32, 1.95), B(sd * .36, 2.75), s * .24, s * .20); lg = p.shape(fur(lg, s * .2, s * .022, 5 + sd), K.C['fur'], .95); p.shade(lg + [-s * .10, s * .04], K.C['sh'], lg); hf = p.shape(spn([B(sd * .36 - .24, 2.66), B(sd * .36 + .24, 2.66), B(sd * .36 + .32, 2.98), B(sd * .36, 2.92), B(sd * .36 - .32, 2.98)], True, (0, 1, 2, 3, 4), 6), K.C['hoof'], .9)
        for sd in (-1, 1): Q = p.shape(spn([B(sd * .52, 1.12), B(sd * .78, 1.10), B(sd * .88, 1.5), B(sd * .74, 1.9), B(sd * .5, 1.9)], True), K.C['pack'], .85); p.shade(Q + [-s * .12, s * .12], K.C['packS'], Q)
        tr = spn([B(-.34, .80), B(-.56, 1.15), B(-.66, 1.6), B(-.60, 2.0), B(0, 2.12), B(.60, 2.0), B(.66, 1.6), B(.56, 1.15), B(.34, .80)], True); Q = p.shape(fur(tr, s * .24, s * .022, 9), K.C['fur'], 1.); p.shade(Q + [-s * .40, s * .10], K.C['sh'], Q); p.shade(spn([B(-.7, .8), B(.7, .8), B(.5, 1.18), B(-.1, 1.30), B(-.5, 1.12)], True), K.C['dk'], Q, .85)
        S_ = spn([B(-.64, 1.72), B(0, 1.82), B(.64, 1.72), B(.72, 2.28), B(.10, 2.30), B(0, 2.16), B(-.10, 2.30), B(-.72, 2.28)], True, (0, 2, 3, 4, 5, 6, 7), 6); Q = p.shape(S_, K.C['shorts'], 1.); p.shade(Q + [-s * .42, s * .02], K.C['shortsS'], Q); p.line(np.array([B(-.2, 1.9), B(-.28, 2.2)]), .4); p.line(np.array([B(.3, 1.9), B(.36, 2.2)]), .4)
        for sd in (-1, 1): Q = p.shape(band_poly(np.array([B(sd * .46, .95), B(sd * .50, 1.3), B(sd * .44, 1.72)]), s * .13), K.C['pack'], .7); p.shade(Q + [-s * .05, 0], K.C['packS'], Q)
        # resting arm (his left): hangs, then throws up with the grin
        la = key(t, [(0, 0.), (20, 0.), (22, -.12), (24, .95), (25, 1.12), (27, 1.)]); sh = B(.52, 1.10); el = sh * 1 + np.array([s * (.36 + .22 * la), s * (.50 - .62 * la)]); ho = el + np.array([s * (-.10 + .30 * la), s * (.44 - 1.02 * la)]); K.hoof_arm(p, [sh, el, ho], [s * .36, s * .33, s * .30], -1, s * .9)
        st.add(p, li); lay = over4(st.a, K.draw(T, li, e, f, hat='old', bounce=b, flop=key(t, [(0, .12), (8, -.02), (12, .14), (20, .16), (24, -.14), (26, .04), (29, 0)]), sway=.0))
        # wiping arm (his right) over the face
        p = Part(T, None, 3500); sh = B(-.52, 1.12)
        if armup > .05:
            hx_ = -.62 + 1.20 * w; face = T.at(hx_, .12 - .05 * math.sin(w * math.pi), 1.0); rest = B(-.86, 1.92); arc = np.array([-s * .42, 0]) * math.sin(armup * math.pi); ho = rest * (1 - armup) + face * armup + arc; el = self.ik(sh, ho, s * .80, s * .84)
        else:
            fl = key(t, [(20, .0), (22, .25), (24, 1.), (25, 1.12), (27, 1.)]) if t >= 20 else 0.; el = sh + np.array([-s * (.40 + .20 * fl), s * (.44 - .60 * fl)]); ho = el + np.array([-s * (-.06 + .34 * fl), s * (.44 - 1.0 * fl)]); el = self.ik(sh, ho, s * .62, s * .70)
        K.hoof_arm(p, [sh, el, ho], [s * .36, s * .33, s * .30], -1, s * .9)
        if 12 <= t < 20:                                     # tears pushed out by the forearm
            r = np.random.default_rng(70 + t)
            for i in range(5): a = (-.5 if w > .5 else math.pi + .5) + r.normal(0, .35); d = s * r.uniform(.28, .62); c0 = T.at(.62 if w > .5 else -.70, .10, .9); bead(p, c0[0] + math.cos(a) * d, c0[1] + math.sin(a) * d, s * r.uniform(.035, .07), a, 1.8)
        if 21 <= f <= 25:
            r = np.random.default_rng(90)
            for i in range(9): a = math.pi + r.uniform(-.9, .7); d = s * (.4 + .42 * (f - 20)) * r.uniform(.6, 1.1); c0 = T.at(-.6, .25, .8); bead(p, c0[0] + math.cos(a) * d, c0[1] + math.sin(a) * d - s * .1 * (f - 20) + s * .03 * (f - 20) ** 2, s * r.uniform(.04, .08), a, 1.7)
        q = Stack(); q.add(p, li); return over4(lay, q.a), T

    def frame(self, f):
        z = key(f, [(0, 0.), (3, .03), (11, .46), (21, .95), (25, 1.), (29, 1.04)]); sk = kit.shake(f, 5, 5, 2., 25) if f >= 25 else np.zeros(2); li = light('pink')
        im = view(self.sky, 18 * z + sk[0] * .3, -26 + 40 * z + sk[1] * .3, 1 + .05 * z); im = comp.over(im, view(self.can, 30 * z + sk[0] * .4, -24 + 16 * z, 1 + .08 * z)); im = comp.over(im, view(self.pk, 34 * z + sk[0] * .5, -20 + 22 * z, 1 + .10 * z))
        mt = fx_air.mist(f, 5, 800, 1010, '#f7a8c4', .5, 18.); im = comp.over(im, mt) if mt.shape[-1] == 4 else im
        im = comp.over(im, view(self.gr, 60 * z + sk[0], 40 * z - 8, 1 + .30 * z))
        tx, ty = TREE; gl = comp.radial(tx - 30 * z + 40, ty - 46 * z, 620, 420, 1.6); im = im + gl[..., None] * rgb('#ff9ec2') * (.10 + .05 * math.sin(f * .5) + (.22 if f >= 25 else 0) * math.exp(-(max(0, f - 25)) / 3.))
        back = fx_air.petals(f, 110, 3, SNOW, wind=(-150., 95.), size=(4, 15), near=0, step=2, lw=.7); im = comp.over(im, back)
        s = 66 + 178 * z; hx, hy = 640 - 60 * z, 752 - 112 * z
        ch, T = self.chopper(f, hx + sk[0], hy + sk[1], s, li)
        if f >= 25: im = comp.over(im, fx_air.petals(f - 23, 90, 17, SNOW, origin=(hx, hy - s * .2), burst=2300., wind=(0, 60), size=(8, 26), near=0, lw=.9, step=1))
        im = self.open(im, f, None) if f < 10 else im
        sh = np.zeros((H, W), f32); cv2.ellipse(sh, (int(hx), int(hy + s * 2.86)), (int(s * 1.25), int(s * .22)), 0, 0, 360, 1., -1, cv2.LINE_AA); im = im * (1 - blur(sh, 8)[..., None] * .5); im = kit.aura(im, ch, 16, '#ff9cc4', .28); im = comp.over(im, ch)
        if 6 <= f <= 12: p = Part(None, None, 5, k=3); u = (f - 6) / 6.; c0 = T.at(.0, .75, 1.); [p.f.fill(ell(c0[0] - s * (.35 + .5 * u + .14 * i), c0[1] - s * (.05 + .25 * u * i), s * (.10 + .14 * u) * (1 + .3 * i), s * (.08 + .10 * u), 20), '#ffe8ee', .38 * (1 - u)) for i in range(3)]; a, ox, oy = p.done(); im = comp.over(im, a)
        if f >= 25:                                           # the clink in the cloud
            q = f - 25; cxx, cyy = tx + 262 + 170 + 12 * z, ty + 46 - 160 - 4 * z; im = kit.flare(im, cxx, cyy, 90 + 30 * q, '#fff3d8', [1.1, .8, .5, .3, .2][q], .5, .2); pass
        fp = fx_air.petals(f, 30, 23, SNOW, wind=(-260., 150.), size=(12, 30), near=4, step=1, lw=1.); im = comp.over(im, fp * (1 - np.clip(comp.radial(hx, hy + s * .3, s * 1.7, power=1.) * 2.4, 0, 1))[..., None])
        if 2 <= f < 10: im = comp.over(im, self.swarm(150, 21, 1.15 * ein((f - 2) / 7., 1.1), SNOW, 24, 78, -.55, .9))
        if f >= 26: im = self.wipe(im, f)
        return im

    def swarm(self, n, seed, prog, pal, s0, s1, ang, spread=1.):
        """a wall of drawn petals crossing the frame; prog = front position along ang (0 enters .. 1 gone)"""
        c = Cel(); d = np.array([math.cos(ang), math.sin(ang)]); m = np.array([-d[1], d[0]]); r = np.random.default_rng(seed); P = []
        for i in range(n):
            lag = r.beta(1.2, 1.2) * spread; v = r.uniform(-1, 1) * 760; s = r.uniform(s0, s1); u = -1300 + 2600 * (prog - lag + .5) + r.normal(0, 60); P.append((s, np.array([960, 540]) + d * u + m * (v + 60 * math.sin(prog * 6 + i)), i, r.uniform(0, 6.28, 3), r.uniform(-1, 1, 3)))
        for s, xy, i, a, w in sorted(P, key=lambda q: q[0]):
            A = a + w * prog * 9; Q, nz = fx_air._tumble(fx_air.SAKURA, xy[0], xy[1], s, A[0], A[1], A[2], .3); fc = abs(nz); c.fill(Q, pal['lit'] if fc > .55 and nz > 0 else pal['mid'] if fc > .25 else pal['dark']); c.line(np.vstack([Q, Q[:1]]), 2.0, pal['ink'], taper=(.2, .4), seed=i, alpha=.8)
        return c.layer()

    def open(self, im, f, ch):
        """the card itself breaks into petals blown to the upper right"""
        cd = np.asarray(self.card, f32)
        if f < 2: return comp.over(cd, ch) if ch is not None else cd
        x, y = grid(); pr = (f - 2) / 7.; u = (x * .86 - y * .5 + 540 * .5) / 1920. + .16 * fbm(H, W, (170, 60, 24), (1, .7, .5), 12) * .5; m = smooth(-.03, .03, u - (-.35 + 1.9 * ein(pr, 1.25))); im = lerp2(im, cd, m)
        return im

    def wipe(self, im, f):
        q = f - 25; pr = [0, .16, .40, .70, 1.0][q]; x, y = grid(); u = (x * .80 - y * .6 + 648) / 2180. + .13 * fbm(H, W, (190, 70, 26), (1, .7, .5), 31) * .5; m = 1 - smooth(-.03, .03, u - (-.40 + 1.75 * pr)); nx = np.asarray(self.nxt, f32); im = lerp2(im, nx, m)
        im = comp.over(im, self.swarm(190, 33, .30 + .50 * pr, SNOW if q < 2 else VIO, 40, 150, -.62, .55)) if q < 4 else comp.over(nx, self.swarm(40, 33, 1.0, VIO, 40, 110, -.62, .5)); return im
