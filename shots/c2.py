"""c2 Nami, orange into storm blue: staff spin opens the card, cloud builds, bolt on 25, grin; the bolt rips the next card open."""
from chars.lkit import *
from chars import nami as N
from op import clouds
from op.comp import over, screen, radial, blur, cam2d, bokeh
from op.kit import speed_lines, flare, shake
from op.fx_air import petals, rain
from op.paint import ramp, fbm
from op.ink import ribbon

HX, HY, S = 672, 512, 246
HAND0 = np.array([1276., 566.])
LEAF = dict(lit='#8fbf4a', mid='#4f8a34', dark='#2c5a2a', ink='#15301a')


def staff(cel, c, ang, L=372, w=24, seed=0, charge=0., f=0):
    d = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-d[1], d[0]]); n = n if n[1] < 0 else -n
    for sgn in (-1, 1):
        a = np.asarray(c, float); b = a + d * L * sgn; h = w * .5; R = np.array([a + n * h, b + n * h, b - n * h, a - n * h]); cel.fill(R, '#8fd3f2'); cel.fill(np.array([a - n * h * .05, b - n * h * .05, b - n * h, a - n * h]), '#3f7fbb'); cel.fill(np.array([a + n * h * .70, b + n * h * .70, b + n * h * .36, a + n * h * .36]), '#eefbff'); stroke(cel, R, 4.6)
        for k in (.30, .64, .96):
            p = a + d * L * k * sgn; q = p + d * w * .9 * sgn; h2 = w * .66; J = np.array([p + n * h2, q + n * h2, q - n * h2, p - n * h2]); cel.fill(J, '#f0b93a'); cel.fill(np.array([p - n * h2 * .1, q - n * h2 * .1, q - n * h2, p - n * h2]), '#a86a1c'); cel.fill(np.array([p + n * h2 * .7, q + n * h2 * .7, q + n * h2 * .4, p + n * h2 * .4]), '#fff1b0'); stroke(cel, J, 4.0)
        t = b + d * 14 * sgn; a_ = np.linspace(0, TAU, 30, endpoint=False); O = t + np.c_[np.cos(a_), np.sin(a_)] * 19; cel.fill(O, '#d8f3ff'); cel.shade(O + [4, 7], '#6db4e2', O, .7); cel.fill(t + [-6, -7] + np.c_[np.cos(a_), np.sin(a_)] * 5, '#ffffff'); stroke(cel, O, 4.0)
    return np.asarray(c) + d * (L + 14), np.asarray(c) - d * (L + 14)


def bolt_path(p0, p1, seed, jag=.10, depth=6):
    r = rs(seed, 9); P = [np.asarray(p0, float), np.asarray(p1, float)]; a = np.hypot(*(P[1] - P[0])) * jag
    for _ in range(depth):
        Q = [P[0]]
        for u, v in zip(P[:-1], P[1:]): d = v - u; n = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6); Q += [(u + v) / 2 + n * r.normal(0, 1) * a, v]
        P = Q; a *= .52
    return np.array(P)


def bolt_layers(paths):
    """paths [(P, w)] -> (cel layer with ink edge, glow mask)"""
    c = Cel(); polys = []
    for P, w in paths:
        n = len(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; nn = np.c_[-T[:, 1], T[:, 0]]; u = np.linspace(0, 1, n); ww = (w * (1 - .72 * u) * (.75 + .5 * rs(n, int(w)).random(n)))[:, None]; polys.append((P, nn, ww))
    for P, nn, ww in polys: c.fill(np.vstack([P + nn * (ww + 9), (P - nn * (ww + 9))[::-1]]), '#24306e')
    for P, nn, ww in polys: c.fill(np.vstack([P + nn * (ww + 4.5), (P - nn * (ww + 4.5))[::-1]]), '#ffc21e')
    for P, nn, ww in polys: c.fill(np.vstack([P + nn * ww, (P - nn * ww)[::-1]]), '#fff6c4')
    for P, nn, ww in polys: c.fill(np.vstack([P + nn * ww * .5, (P - nn * ww * .5)[::-1]]), '#ffffff')
    return c.layer()


def tangerine(s):
    c = Cel(int(s * 3), int(s * 3)); m = s * 1.5; a = np.linspace(0, TAU, 60, endpoint=False); P = np.c_[m + np.cos(a) * s, m + np.sin(a) * s * .88 - s * .06 * np.exp(-((a - 4.71) / .4) ** 2) * -1]; c.fill(P, '#f98f1f'); c.shade(P + [s * .24, s * .30], '#c9541a', P, .0); c.clip(P); c.fill(spl(np.array([(m + s * 1.1, m - s * .4), (m + s * .45, m + s * .02), (m + s * .1, m + s * .52), (m - s * .7, m + s * .74), (m + s * .2, m + s * 1.3), (m + s * 1.3, m + s * .6)]), True, n=6), '#c9541a')
    r = rs(4)
    for _ in range(26): x, y = r.normal(0, .5, 2) * s; c.fill(np.c_[m + x + np.cos(a) * s * .022, m + y + np.sin(a) * s * .022], '#d9691a')
    c.fill(spl(np.array([(m - s * .62, m - s * .28), (m - s * .36, m - s * .60), (m - s * .18, m - s * .56), (m - s * .44, m - s * .24)]), True, n=6), '#ffd58a'); c.unclip(); stroke(c, P, s * .085)
    c.line(np.array([(m, m - s * .80), (m + s * .03, m - s * 1.04)]), s * .13, '#3c5a22', taper=(0, 0)); L = spl(np.array([(m + s * .02, m - s * .94), (m + s * .34, m - s * 1.26), (m + s * .86, m - s * 1.18), (m + s * .50, m - s * .86)]), True, corners=(0, 2), n=7); c.fill(L, '#5f9a3c'); c.shade(L + [0, s * .16], '#376a2c', L, .9); c.line(np.array([L[0], (m + s * .45, m - s * 1.06), (m + s * .8, m - s * 1.16)]), s * .04, '#1d3a18'); stroke(c, L, s * .06); return c.rgba()


class Shot:
    id = 'c2'; n = 30; finish = dict(bloom=.34, vig=.14)
    EVENTS = [(0, 'whoosh', dict(size=.4, note='card in, staff already spinning')), (2, 'spin', dict(size=.7, note='staff whirr, f0-10')), (4, 'paint', dict(note='card swept open by the spin')), (12, 'whip', dict(size=.6, note='staff stops, points at the sky')), (13, 'crackle', dict(size=.3)), (16, 'thunder', dict(size=.35, note='far rumble in the cloud')),
              (19, 'thunder', dict(size=.45)), (23, 'whip', dict(size=.7, note='staff swung down')), (25, 'impact', dict(size=1.0, note='bolt: DON')), (25, 'thunderclap', dict(size=1.0)), (26, 'rip', dict(size=.8, note='bolt tears the picture open to mustard'))]

    def setup(self):
        self.card = card(H, W, '#f07c16', '#c9520c', '#ffa63c', 2, -.45); self.next = card(H, W, '#d8a11e', '#a8741a', '#f4c94c', 3, -.4)
        import os
        p = 'cache/c2_sky.npy'
        if os.path.exists(p): Z = np.load(p, mmap_mode='r'); self.sky = np.array(Z['sky'] if False else Z[..., :3], f32); self.cl = np.array(np.load('cache/c2_cloud.npy', mmap_mode='r'), f32)
        else:
            hh, ww = 1240, 2240; y = np.linspace(0, 1, hh)[:, None].repeat(ww, 1); g = ramp(y, [(0, '#16305e'), (.45, '#2f659c'), (.74, '#78aecb'), (.86, '#cfe4df'), (1, '#8fb9cc')]); sk = clouds.Sky(hh, ww, g, light=(.62, -.70, .36), seed=5)
            sk.bank(300, 1120, 900, 330, ('#e6f1ee', '#a9c6d8', '#6d93b8', '#46699a'), '#a8c8d4', 3); sk.bank(1250, 1150, 1100, 260, ('#f3f3e6', '#b7cede', '#7aa0c0', '#4d74a4'), '#b6d2da', 4); sk.bank(2000, 1140, 800, 420, ('#e6f1ee', '#9fc0d6', '#628bb4', '#3d6296'), '#a8c8d4', 6); self.sky = sk.paint(3).astype(f32); np.save(p, self.sky.astype(np.float16))
            k2 = clouds.Sky(900, 2400, rgb('#1b2a52'), light=(.55, -.60, .50), seed=8, layered=True)
            for x, b, w_, h_, sd in ((500, 560, 1100, 430, 1), (1900, 560, 1100, 440, 2), (1200, 700, 1600, 620, 3), (700, 820, 1000, 420, 5), (1700, 820, 1000, 420, 6)): k2.bank(x, b, w_, h_, ('#8fa6cc', '#4c5f92', '#2b3868', '#161c40'), '#1a2350', sd, pool=0., foot=.16)
            L = k2.layer(k2.paint(5)); self.cl = L[::-1].copy().astype(f32); np.save('cache/c2_cloud.npy', self.cl.astype(np.float16))     # hangs from the sky: flip
        st = N.style('storm'); pal = dict(skin=st['skin'], shade=st['shade']); self.hand, g = hand_sprite('grip', 'L', HD.euler(-12, 6, -21), 104, HD.FEMALE, pal, light=(.55, .7, .5), w=4.0, rimc='#e6f6ff', rimd=(-.80, .60), rw=6, style=dict(nails='short'), dist=8.); self.crop_hand(g)
        self.tang = mips(tangerine(90), 4)
        # swirl reveal time: angle round the hand + radius, torn by bristles
        x, y = grid(); dx, dy = x - HAND0[0], y - HAND0[1]; r = np.sqrt(dx * dx + dy * dy); a = (np.arctan2(dy, dx) + math.pi * .75) % TAU / TAU; streak = cv2.resize(np.random.default_rng(3).standard_normal((1, 420)).astype(f32), (1500, 1), interpolation=cv2.INTER_LINEAR)[0]; br = streak[np.clip(r, 0, 1499).astype(int)]
        self.rev = (.55 * a + .42 * r / 1500. + .040 * br + .016 * fbm(H, W, (60, 18, 6), None, 4)).astype(f32)
        self.N_ = None

    def crop_hand(self, g):
        """keep the hand and a short stub of wrist: the arm is one drawn contour, the log pose sits on the join"""
        h, w = self.hand.shape[:2]; R = sorted(g.wrist[0], key=lambda q: abs(q['y'])); far = np.asarray(R[-1]['P'], float)[:, :2].mean(0) - [w / 2., h / 2.]; u = far / np.hypot(*far); nn = np.array([-u[1], u[0]]); near = min(R, key=lambda q: abs(abs(q['y']) - .22)); pr = (np.asarray(near['P'], float)[:, :2] - [w / 2., h / 2.]) @ nn; self.wh = float(np.ptp(pr) / 2.); x, y = np.meshgrid(np.arange(w) - w / 2., np.arange(h) - h / 2.); self.hand = (self.hand * np.clip(26 - (x * u[0] + y * u[1]), 0, 1)[..., None]).astype(f32)

    def arm(self, c, p, st, lw, dy):
        S_ = p + np.array([S * 1.06, S * 1.74]); Wr = np.asarray(self.hd_, float); E = HAND0 + [-96, 240 + dy * .55]; d = (E - S_) / np.hypot(*(E - S_)); n = np.array([-d[1], d[0]]); v = (Wr - E) / np.hypot(*(Wr - E)); mm = np.array([-v[1], v[0]]); mm = mm if mm[0] > 0 else -mm; Lf = np.hypot(*(Wr - E)); wh = self.wh
        out_ = spl([S_ + n * 54 - d * 20, S_ + d * 70 + n * 52, E - d * 52 + n * 43, E + d * 2 + n * 40, E + d * 34 + n * 22, E + v * 46 + mm * 40, E + v * Lf * .42 + mm * 38, E + v * Lf * .74 + mm * (wh + 6), Wr - v * 6 + mm * wh], n=9); in_ = spl([S_ - n * 50 - d * 20, S_ + d * 70 - n * 49, E - d * 74 - n * 41, E - d * 40 - n * 40 + v * 2, E + v * 58 - mm * 36, E + v * Lf * .48 - mm * 37, E + v * Lf * .76 - mm * (wh + 5), Wr - v * 6 - mm * wh], n=9)
        A = np.vstack([out_, in_[::-1]]); c.fill(A, st['skin']); c.clip(A); c.fill(np.vstack([spl([S_ - d * 30 + n * 6, S_ + d * 80 + n * 10, E - d * 40 + n * 12, E - d * 4 + n * 2, E + v * 60 - mm * 12, E + v * Lf * .5 - mm * 12, Wr - mm * 6], n=8), (Wr - mm * 90)[None], (E - mm * 80 + v * 20)[None], (E - d * 60 - n * 4)[None], (E - d * 50 + n * 120)[None], (S_ + n * 140 - d * 30)[None]]), st['shade']); m = lerp(S_, E, .64) - n * 6
        # tattoo: tangerine + pinwheel
        a_ = np.linspace(0, TAU, 40, endpoint=False); blue = '#2c4fa0'; R = 21
        for k in range(4): th = k * math.pi / 2 + .4; q = m + [0, 4]; arc = spl([q, q + R * 1.0 * np.array([math.cos(th), math.sin(th)]), q + R * 1.5 * np.array([math.cos(th + .75), math.sin(th + .75)]), q + R * .82 * np.array([math.cos(th + 1.05), math.sin(th + 1.05)])], True, n=5); c.fill(arc, blue)
        c.fill(m + [0, -30] + np.c_[np.cos(a_) * 10, np.sin(a_) * 9], blue); c.fill(m + [-2, -32] + np.c_[np.cos(a_) * 4, np.sin(a_) * 3], st['skin']); c.line(spl([m + [2, -38], m + [10, -48], m + [22, -48]], n=5), 4, blue, taper=(0, .5)); c.unclip(); c.line(out_, lw, INK, taper=(.02, .02), press=.4, seed=4); c.line(in_, lw, INK, taper=(.02, .02), press=.4, seed=5); c.line(spl([E - d * 46 - n * 34, E - d * 30 - n * 20, E - d * 26 - n * 4], n=6), lw * .5, INK, taper=(.05, .7))
        # sleeve over it
        B = spl([S_ - n * 66 - d * 70, S_ - n * 60 + d * 10, S_ + d * 74 - n * 58, S_ + d * 86, S_ + d * 70 + n * 62, S_ + n * 78, S_ + n * 92 - d * 70], True, n=6); c.fill(B, st['cloth']['base']); c.clip(B)
        for k in range(-1, 3): o = S_ + d * (k * 48 + 4); c.fill(spl([o - n * 90 - d * 10, o + d * 12, o + n * 100 - d * 6, o + n * 100 + d * 18, o + d * 36, o - n * 90 + d * 14], True, n=5), st['cloth']['stripe'])
        c.fill(B + n * 54, st['cloth']['shade'], .55); c.unclip(); outline(c, B, lw, INK, 7); fold(c, [S_ - d * 40 + n * 10, S_ + d * 6 + n * 6, S_ + d * 40 - n * 4], lw * .5); fold(c, [S_ - d * 30 + n * 46, S_ + d * 10 + n * 34, S_ + d * 36 + n * 26], lw * .5)

    def logpose(self, c, w, lw):
        u = np.array([math.cos(math.radians(-69)), math.sin(math.radians(-69))]); n = np.array([-u[1], u[0]]); B = spl([w - n * 33 - u * 16, w - u * 10, w + n * 33 - u * 16, w + n * 33 + u * 12, w + u * 18, w - n * 33 + u * 12], True, n=5); c.fill(B, '#5a3320'); c.shade(B + n * 22, '#2f1810', B, .8); outline(c, B, lw, INK)
        P = w + spl([(-34, -15), (34, -15), (40, 0), (34, 15), (-34, 15), (-40, 0)], True, n=4); 
        a = np.linspace(0, TAU, 30, endpoint=False); pl = np.array([w - n * 34 - u * 13, w + n * 34 - u * 13, w + n * 34 + u * 13, w - n * 34 + u * 13]); c.fill(pl, '#e9b945'); c.fill(np.array([w + n * 4 - u * 13, w + n * 34 - u * 13, w + n * 34 + u * 13, w + n * 4 + u * 13]), '#a8741e'); stroke(c, pl, lw * .7)
        for k in (-1, 0, 1):
            q = w + n * 21 * k - u * 2; G = q + np.c_[np.cos(a), np.sin(a)] * 15.5; c.fill(G, '#bfe6f2'); c.shade(G + [4, 6], '#6aa6cc', G, .75); c.line(np.array([q - [6, -7], q + [6, -7]]), 3.4, ('#d8203a', '#2c4fa0', '#d8203a')[k + 1], taper=(.3, .3)); c.fill(q + [-5, -6] + np.c_[np.cos(a) * 4.5, np.sin(a) * 3], '#ffffff'); stroke(c, G, lw * .62)

    def frame(self, f):
        d = f // 2 * 2; t = f / 24.; build = float(sm((f - 8) / 14.)); hit = f >= 25; flash = math.exp(-(f - 25) / 2.5) if hit else 0.; flick = {16: .35, 17: .15, 19: .5, 20: .2, 22: .45, 23: .15}.get(f, 0.); z = 1. + .0016 * f + (.05 * flash); sx, sy = (shake(f, 16, 3, 2.5, 25) if hit else (0, 0))
        world = cam2d(self.sky, 1120 - 3.2 * f, 640 - f * 1.4, 1.06 * z, -.015).astype(f32); world = world * (1 - .42 * build) + rgb('#101b40') * .42 * build * .6
        M = np.float32([[1.0 + .10 * build, 0, -250 - f * 5.5], [0, .62 + .50 * build, -420 + 300 * build]]); cl = cv2.warpAffine(self.cl, M, (W, H), flags=cv2.INTER_LINEAR); L = cl.copy(); L[..., :3] *= (1.05 - .25 * build); world = over(world, L)
        for k, (x, y, r) in enumerate(((420, 210, 520), (1180, 120, 640), (1640, 260, 420))): world = world + (radial(x, y, r, r * .55, 2.) * cl[..., 3] * (flick * (1.2 if (f + k) % 3 else .3) + flash * 1.1))[..., None] * rgb('#cfe0ff')
        if f > 10: R = rain(f, wind=-.22, n=(130, 60, 0), seed=3, near=False, alpha=.55 * build); world = over(world, R)
        # reveal
        k = (f - 2.5) / 7.5; m = np.clip((k * 1.25 - self.rev) * 70, 0, 1)[..., None] if f < 13 else 1.
        if f < 13: e = np.clip(1 - np.abs((k * 1.25 - self.rev) * 70 + 1.2) / 1.6, 0, 1)[..., None] * (f > 2); base = self.card * (1 - m) + world * m; base = base * (1 - .75 * e) + rgb('#ffe0a0') * .75 * e
        else: base = world
        P = lambda q: np.array([HX + float(keys(q, [(0, -46.), (5, 0., out3), (10, -4.), (12, 10.), (14, -14., out3), (18, -8.), (22, -14.), (24, 4.), (25, 22., out3), (28, 14.), (30, 12.)])), HY + float(keys(q, [(0, 26.), (5, 0., out3), (10, 8.), (12, 22.), (14, -22., out3), (17, -12.), (22, -14.), (24, 14.), (25, -6., out3), (28, 4.), (30, 4.)]))])
        p = P(f) + [sx * .5, sy * .5]; yaw = float(keys(f, [(0, .10), (6, .22), (11, .30), (14, .34), (19, .30), (23, .08), (25, -.06, out3), (30, -.08)])); pitch = float(keys(f, [(0, -.02), (10, -.02), (12, -.08), (14, .16, out3), (19, .10), (23, -.06), (25, -.02, out3), (30, .0)])); roll = float(keys(f, [(0, .06), (10, -.02), (12, .05), (14, -.10, out3), (20, -.05), (24, .06), (25, .10, out3), (30, .08)]))
        if f < 11: ex = dict(EXPR['smirk'], o=.95, gaze=(.75, .15), bo=.06, msx=1.15)
        elif f < 17: ex = dict(EXPR['smirk'], o=1.0, gaze=(.65, .7), M='smile', bo=.05, bi=.02)
        elif f in (18, 19): ex = dict(EXPR['smirk'], o=.12, M='smirk')
        elif f < 25: ex = dict(EXPR['smirk'], o=.88, gaze=(.15, -.02), M=mmix('smirk', 'grin', .35 if f > 21 else 0.), teeth=(.05, 0))
        else: ex = dict(o=(1.0, 0.0) if f < 28 else (1.0, .0), shut='tight', M=mmix('grin', 'laugh', .55), msx=1.04, msy=1.02, mc=(.02, -.575), teeth=(.085, 0.), tongue=.50, gaze=(.12, .02), bi=-.02, bo=.07, sq=.22, blush=.35)
        ws = float(keys(f, [(0, .55), (10, .60), (14, .80), (22, .85), (25, 1.0), (30, 1.0)])); sway = np.array([-34 - 22 * math.sin(d * .9), -6 - 14 * (f > 13)]) * (1 + .9 * flash)
        st = N.style('storm'); dy = float(keys(f, [(0, 0.), (10, 10.), (12, 34.), (13, -34., out3), (15, -18.), (22, -22.), (23, -40.), (25, 30., out3), (28, 18.), (30, 18.)])); hd = HAND0 + [0, dy + (sy if hit else 0)]; self.hd_ = hd
        def body(c, F, s_): bust(c, p[0], p[1] + (P(f)[1] - HY) * -.4, S, roll * .35, s_, 'tee', .18, 4.8, 1, f=f, breath=math.sin(f * .3)); self.arm(c, p * [1, 0] + [0, HY], s_, 4.6, dy)
        D = N.draw(p[0], p[1], S, yaw, pitch, roll, None, 'storm', d, wind=(math.pi * .985 - .12 * math.sin(d * .5), ws), ex=ex, body=body, posfn=P, sway=sway, length=1.18)
        lay_ = stack(D['back'], D['body'], D['face'], D['front']); lay_ = para(lay_, (1300, 100), (300, 1080), [(0, '#ffffff'), (.5, '#f6ecf4'), (1, '#8f8fc2')]); 
        if build > 0: lay_[..., :3] *= (1 - .10 * build)
        if flick + flash > 0: lay_ = lit(lay_, (.35, .94) if hit else (-.5, .86), '#f4f9ff', 3., 12 + 16 * flash, min(1., flick + flash), .6)
        wind_l = over(base, petals(f, 46, 5, LEAF, wind=(-900., 60.), size=(9, 26), spin=5., sway=70., near=0))
        # fruit on the wind behind her
        for k, (x0, y0, v, sc, sd) in enumerate(((2050, 260, 58, .34, 1), (2300, 820, 74, .52, 2), (2600, 420, 96, .70, 3), (2450, 985, 50, .26, 4), (2900, 930, 150, 1.5, 5))):
            if k in (3, 4): continue
            x = x0 - v * (f + 3) - 4; y = y0 + 40 * math.sin(f * .3 + k) - (f * 3 if k % 2 else 0); tmp = wind_l
            if k == 4:
                L = np.zeros((H, W, 4), f32); putm(L, self.tang, x, y, sc, f * .4); L = blur(L, 9); L = L + 0; wind_l = over(wind_l, L)
            else: putm(wind_l, self.tang, x, y, sc, f * (.35 + .1 * k), tintc='#27406e', ta=.25 if sc < .4 else .08)
        base = wind_l; base = over(base, lay_)
        # staff + hand
        c = Cel(); a0 = -2.30; spin = f < 11; ang = a0 - (11 - f) * .82 if spin else float(keys(f, [(11, a0 + .0), (12, a0 + .35), (13, a0 - .25, out3), (15, a0 - .05), (22, a0), (23, a0 + .42), (24, a0 - .50, lambda q: q), (25, a0 - 1.02, out3), (27, a0 - .92), (30, a0 - .94)])); squash = .62 + .38 * min(1, f / 11.)
        hc = hd + [14, -62]
        if spin or f in (13, 24):
            sp = 1.5 if spin else .9; sg = 1 if spin else (1 if f == 13 else 1); u = np.linspace(0, 1, 26)
            for e in (0, math.pi):
                th = ang + e - sg * u * sp * (-1 if not spin else 1) * (1 if spin else -1); r0, r1 = 385, 150 + 190 * u; A = hc + np.c_[np.cos(th), np.sin(th)] * 392; B_ = hc + np.c_[np.cos(th), np.sin(th)] * (392 - 235 * (1 - u))[:, None]; c.fill(np.vstack([A, B_[::-1]]), '#cdeeff', .72); c.fill(np.vstack([A, (hc + np.c_[np.cos(th), np.sin(th)] * (392 - 60 * (1 - u))[:, None])[::-1]]), '#ffffff', .85)
                for q in (330, 270, 205): c.line(hc + np.c_[np.cos(th), np.sin(th)] * q, 3.4, '#2c5f9c', taper=(.02, .8), alpha=.8)
        tp = staff(c, hc, ang, w=23, f=f); layS = lit(c.layer(), (-.8, .6), '#ffffff', 3.4, 4, .7); HL_ = np.zeros((H, W, 4), f32); put(HL_, self.hand, hd[0], hd[1], 1., float(keys(f, [(0, .05), (12, .10), (13, -.08), (22, 0), (23, .12), (25, -.16, out3), (30, -.12)]))); c = Cel(); self.logpose(c, hd + [-6, 22], 4.0); layS = lay(lay(layS, HL_), lit(c.layer(), (-.8, .6), '#f4fbff', 3, 4, .8)); base = over(base, para(layS, (1300, 100), (900, 1080), [(0, '#ffffff'), (1, '#b9b4d8')]))
        # charge at the tip, the bolt
        tip = tp[0] if tp[0][1] < tp[1][1] else tp[1]
        if 13 <= f < 25:
            c = Cel(); r = rs(f, 5)
            for k in range(5): a = r.uniform(0, TAU); q = tip + np.array([math.cos(a), math.sin(a)]) * r.uniform(40, 110 + 60 * build); P_ = bolt_path(tip, q, f * 7 + k, .18, 4); c.line(P_, 7, '#24306e', taper=(.05, .5)); c.line(P_, 3.6, '#fff3a0', taper=(.05, .5))
            base = over(base, c.layer()); base = flare(base, tip[0], tip[1], 190 + 60 * (f % 2), '#ffe98a', .30 + .25 * build)
        if f >= 25:
            s0 = 41 + (f - 25) // 2; main = bolt_path((1000, -60), (1075, 1140), s0, .085, 7); paths = [(main, 34 * (.6 + .6 * flash))]
            for k in (20, 44, 66, 88): q = main[k]; a = math.pi * (.35 + .5 * rs(s0, k).random()); e = q + np.array([math.cos(a) * -1, math.sin(a)]) * rs(s0, k, 2).uniform(300, 620) * [(-1 if k % 3 else 1), 1]; paths.append((bolt_path(q, e, s0 + k, .13, 5), 12))
            paths.append((bolt_path((1130, -60), tip, s0 + 5, .12, 6), 17)); BL = bolt_layers(paths); back_ = base; m_ = blur(BL[..., 3], 34); base = base + m_[..., None] * rgb('#ffe9a0') * .75 * max(flash, .35); ch = np.maximum(lay_[..., 3], layS[..., 3]); BL2 = BL * (1 - ch[..., None]); base = over(base, BL2); base = over(base, bolt_layers(paths[-1:]))
        # wind debris: tangerines, near leaves
        for k, (x0, y0, v, sc, sd) in enumerate(((2050, 260, 58, .34, 1), (2300, 820, 74, .52, 2), (2600, 420, 96, .70, 3), (2450, 985, 50, .26, 4), (2900, 930, 150, 1.5, 5))):
            if k not in (3, 4): continue
            x = x0 - v * (f + 3) - 4; y = y0 + 40 * math.sin(f * .3 + k) - (f * 3 if k % 2 else 0); tmp = base
            if k == 4:
                L = np.zeros((H, W, 4), f32); putm(L, self.tang, x, y, sc, f * .4); L = blur(L, 9); L = L + 0; base = over(base, L)
            else: putm(base, self.tang, x, y, sc, f * (.35 + .1 * k), tintc='#27406e', ta=.25 if sc < .4 else .08)
        base = over(base, petals(f + 7, 14, 9, LEAF, wind=(-1500., 90.), size=(24, 46), spin=6., sway=60., near=3))
        if f == 25: base = base * .55 + (1 - np.clip(base, 0, 1)) * 0 + rgb('#fff8dc') * .52
        if f >= 26:                                                   # the bolt rips the picture open to the next card
            wd = (0, 46, 260, 900)[f - 26]; G = bolt_path((1000, -80), (1075, 1160), 7, .075, 5); n = len(G); jw = rs(8).uniform(.75, 1.3, n); A = G - np.c_[wd * jw, 0 * jw]; B_ = G + np.c_[wd * jw[::-1], 0 * jw]; Lf = cv2.warpAffine(base, np.float32([[1, 0, -wd * .35], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE); Rt = cv2.warpAffine(base, np.float32([[1, 0, wd * .35], [0, 1, 0]]), (W, H), borderMode=cv2.BORDER_REPLICATE)
            x, y = grid(); gx = np.interp(y[:, 0], G[:, 1], G[:, 0])[:, None]; base = np.where((x < gx)[..., None], Lf, Rt)
            if wd: c = Cel(); gp = np.vstack([A, B_[::-1]]); c.fill(gp, '#ffffff'); mk = c.layer()[..., 3:4]; nx = self.next; base = base * (1 - mk) + nx * mk; c = Cel(); [(c.line(E, 26, '#24306e', taper=(0, 0)), c.line(E, 17, '#ffc21e', taper=(0, 0)), c.line(E, 8, '#ffffff', taper=(0, 0))) for E in (A, B_)]; LL = c.layer(); base = over(base + blur(LL[..., 3], 22)[..., None] * rgb('#ffe08a') * .6, LL)
        if hit and f < 29: base = cv2.warpAffine(base.astype(f32), np.float32([[1, 0, sx * .6], [0, 1, sy * .6]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        return np.clip(base, 0, 2.5).astype(f32)
