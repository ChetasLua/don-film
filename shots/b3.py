"""b3  The hat is too big.  A lip, a tear with a ship in it, and the howl."""
import numpy as np, cv2, math, os
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp, repaint, mix
from op.comp import over, screen, blur, lerp_keys, radial, cam2d
from op.ink import Cel, spl, resample
from op.kit import rim, shake, smear
from op import hat as HT, fx_core as FC
from chars import luffy as L
from shots.d8 import cached

BW, BH = 2240, 1260
SUN = (1560, 500)                                       # plate px


def ship(c, x, y, s, k=('#2a1426', '#4a1f2e', '#7a2c34'), rimc='#ffce86'):
    """the Red Force going away: dragon prow to the right, three masts; s = hull length px"""
    P = lambda q: np.array(q, float) * s + [x, y]
    for mx, h, w in ((-.24, .62, .20), (.02, .74, .26), (.26, .54, .18)):
        c.fill(P([(mx - .008, -.06), (mx - .004, -h), (mx + .004, -h), (mx + .008, -.06)]), k[0])
        for j, (a, b, ww) in enumerate(((.16, .40, 1.), (.42, .60, .74), (.62, .72, .5))):
            if b > h / .74 * .72: continue
            hw = w * ww; c.fill(P(spl([(mx - hw * .5, -b * h / .74), (mx, -b * h / .74 - .006), (mx + hw * .5, -b * h / .74), (mx + hw * .62, -(a + b) / 2 * h / .74), (mx + hw * .5, -a * h / .74), (mx, -a * h / .74 + .022), (mx - hw * .5, -a * h / .74), (mx - hw * .40, -(a + b) / 2 * h / .74)], True, n=5)), k[2]); c.line(P([(mx + hw * .5, -b * h / .74), (mx + hw * .62, -(a + b) / 2 * h / .74), (mx + hw * .5, -a * h / .74)]), max(1., s * .006), rimc, taper=(.2, .2))
    c.fill(P(spl([(-.50, -.16), (-.46, -.02), (-.36, .04), (.30, .04), (.42, -.02), (.50, -.12), (.40, -.09), (.0, -.06), (-.34, -.09), (-.38, -.17)], True, corners=(0, 5, 9), n=5)), k[0]); c.fill(P([(-.40, -.02), (.36, -.02), (.34, .0), (-.38, .0)]), k[1])
    c.fill(P(spl([(.46, -.08), (.54, -.12), (.56, -.21), (.61, -.26), (.69, -.25), (.70, -.22), (.63, -.20), (.60, -.14), (.54, -.05)], True, n=4)), k[0]); c.fill(P(spl([(.595, -.26), (.585, -.32), (.625, -.27)], True, n=3)), k[0]); c.line(P([(.02, -.74), (.02, -.82)]), max(1, s * .006), k[0]); c.fill(P([(.02, -.82), (-.07, -.80), (.02, -.765)]), k[0])


def plate(blurred=True):
    x, y = np.meshgrid(np.arange(BW, dtype=f32), np.arange(BH, dtype=f32)); n = fbm(BH, BW, (70, 30, 12), seed=3, aspect=5.5); n2 = fbm(BH, BW, (160, 60, 24), seed=8, aspect=2.2); hz = 620
    g = ramp(np.clip(y / hz, 0, 1), [(0, '#4d4c8c'), (.30, '#8d5f95'), (.55, '#d8748a'), (.78, '#ff9d62'), (.93, '#ffd487'), (1, '#fff0b8')]); sd = np.hypot((x - SUN[0]) / 1.25, y - SUN[1])
    g = screen(g, np.exp(-(sd / 330) ** 1.3)[..., None] * rgb('#ffcf7a') * .75)
    cl = smooth(.20, .55, n2 + .35 * n - .25) * (y < hz); lit = smooth(-.1, .9, np.roll(n2, (14, -22), (0, 1)) * -1 + n2 + .2); g = mix(g, mix(rgb('#6b4a7a') * np.ones_like(g), rgb('#ffbf86') * np.ones_like(g), np.clip(lit + np.exp(-(sd / 500) ** 2), 0, 1)[..., None]), (cl * .80)[..., None])
    s = ramp(np.clip((y - hz) / (BH - hz), 0, 1), [(0, '#ffc27c'), (.05, '#d47a6c'), (.22, '#8c4f72'), (.6, '#4a3566'), (1, '#2a2350')]); wv = fbm(BH, BW, (9, 4), seed=12, aspect=7); s = s * (1 + .10 * wv)[..., None]
    gl = np.exp(-((x - SUN[0]) / (60 + (y - hz) * .42)) ** 2) * (wv > .1) * (y > hz); s = s + gl[..., None] * rgb('#ffd98a') * .65
    g = np.where((y < hz)[..., None], g, s); g = repaint(np.clip(g, 0, 1.3).astype(f32), (36, 16, 8), seed=5)
    d = np.clip((74 - np.hypot(x - SUN[0], y - SUN[1])) * .5, 0, 1)[..., None]; g = g * (1 - d) + rgb('#fffbe4') * 1.12 * d; g = screen(g, np.exp(-(np.hypot(x - SUN[0], y - SUN[1]) / 170) ** 1.2)[..., None] * rgb('#ffd27a') * .6)
    c = Cel(BW, BH, ss=1)
    c.fill(spl([(-40, 625), (0, 560), (120, 520), (260, 548), (330, 530), (440, 590), (560, 622)], False), '#5a3a66'); c.fill(np.array([(300, 532), (306, 492), (318, 492), (324, 536)]), '#4a2f58')
    for a in (.4, 1.97, 3.54, 5.11): d_ = np.array([math.cos(a), math.sin(a)]); c.line(np.array([(312, 494), np.array([312, 494]) + d_ * 46]), 5, '#4a2f58', taper=(.1, 0))
    if not blurred: ship(c, 800, 612, 420)
    q = c.rgba(); g = g * (1 - q[..., 3:4]) + q[..., :3]
    return (blur(g, 6.5) if blurred else g).astype(f32)


def caustic(f, h=270, w=480, sc=.021):
    y, x = np.mgrid[0:h, 0:w].astype(f32); p0, p1 = (x * sc * 4) % 1e6, (y * sc * 4.6); t = f * .045 + 23; i0, i1 = p0.copy(), p1.copy(); c = np.ones((h, w), f32)
    for n in range(4):
        tt = t * (1 - 3.5 / (n + 1)); a = p0 + np.cos(tt - i0) + np.sin(tt + i1); b = p1 + np.sin(tt - i1) + np.cos(tt + i0); i0, i1 = a, b
        c += 1. / np.sqrt((p0 / (np.sin(i0 + tt) / .28 + 1e-3)) ** 2 + (p1 / (np.cos(i1 + tt) / .28 + 1e-3)) ** 2 + 1e-4) * 0 + .42 * (1 - np.abs(np.sin((i0 + i1 * .7) * .8)))
    c = c / c.max(); return cv2.resize(np.clip((c - .52) / .30, 0, 1) ** 2.2, (W, H), interpolation=cv2.INTER_CUBIC)


DAP = [(-.60, -.40, .060, .028), (-.42, -.30, .034, .018), (-.24, -.50, .026, .016), (.06, -.34, .022, .014), (.30, -.44, .050, .026), (.52, -.34, .036, .020), (.64, -.52, .030, .030), (-.52, -.62, .022, .018), (.10, -.26, .03, .012), (.42, -.60, .020, .014), (-.12, -.30, .030, .014)]


LEAK = dict(HT.LIGHTS['dusk'], glow=.95, leak=.8)


class Shot:
    id = 'b3'; n = 60; finish = dict(bloom=.34, vig=.16)
    EVENTS = [(0, 'tear_slide', dict(size=.1, note='small wet trail; pier water, gulls far')), (10, 'sniff', dict(size=.4)), (14, 'lip_tremble', dict(size=.2, dur=12)), (26, 'tear_drop', dict(size=.4, note='leaves the chin')), (28, 'inhale', dict(size=.5, note='caught breath')), (30, 'howl', dict(size=1., note='child cry breaks open, brim flops')),
              (36, 'sob_hitch', dict(size=.6)), (44, 'sob_hitch', dict(size=.6)), (52, 'sob_hitch', dict(size=.6))]

    def setup(self):
        self.bg = cached('b3_bg', plate); self.src = cached('b3_src', lambda: plate(False))

    def state(self, f):
        g = f if 26 <= f <= 34 else f // 2 * 2; K = lambda k: float(lerp_keys(g, k))
        z = float(lerp_keys(f, [(0, 1.0), (6, 1.01), (24, 1.34), (28, 1.36), (31, 1.02), (35, 1.08), (59, 1.24)]))
        trem = (.6 + 1.6 * smooth(6, 22, g)) * (g < 27); tr = math.sin(g * 2.6) * trem
        if g < 27: e = dict(L.EXPR['lip'], mw=.175 - .015 * math.sin(g * .9) - .02 * smooth(8, 11, g) * (1 - smooth(11, 16, g)), yc=-.66 + .004 * tr, uy=-.605 + .006 * tr, ly=-.612 + .006 * tr, wob=.010 + .012 * smooth(8, 22, g), sk=.006 * tr, tear=0.)
        elif g < 30: e = dict(L.EXPR['lip'], mw=.14, yc=-.67, uy=-.60, ly=-.612, wob=.02, fore=1., tear=0., lip=1.)
        else:
            op = K([(30, .55), (31, 1.18), (33, .90), (35, 1.02)]) if g < 36 else 1 - .34 * max(0, math.sin((g - 36) / 8 * 2 * math.pi)) ** 1.0 * (1 if (g - 36) % 16 < 8 else .6)
            e = ('lip', dict(L.EXPR['howl'], tear=float(np.clip((g - 29) / 5, 0, 1)), snot=.35 + .65 * smooth(32, 40, g) + 1.6 * 0), op)
        hitch = 0.
        for t0 in (10, 36, 44, 52):
            if g >= t0: hitch += .11 * math.exp(-(g - t0) / 2.6) * (1 if t0 != 10 else .6)
        pitch = K([(0, -.07), (8, -.08), (10, -.02), (14, -.08), (24, -.10), (27, -.17), (29, -.20), (30, .02), (31, .13), (34, .07), (44, .05), (52, .09), (58, .06)]) + .25 * hitch * (g > 30)
        cy = K([(0, 400), (26, 404), (29, 420), (30, 384), (31, 366), (34, 386), (59, 390)]) - 60 * hitch + tr * .4; st = K([(0, 0), (27, 0), (29, -.06), (30, .10), (31, .06), (33, -.02), (35, 0)]) - .25 * hitch * (g > 34)
        return g, z, e, pitch, cy, st, hitch

    def frame(self, f):
        g, z, e, pitch, cy, st, hitch = self.state(f); s0 = 400.; A = np.array([960., 700.]); s = s0 * z; cx, cy = A[0] + (960 - A[0]) * z, A[1] + (cy - A[1]) * z
        zb = 1 + (z - 1) * .40; img = cam2d(self.bg, BW / 2 - 10 + f * .35, BH / 2 - 10 - (z - 1) * 60, zb * .98, 0)
        # glitter that never stops + the ship, soft, leaving
        r = np.random.default_rng(40); c = Cel(W, H, ss=1); t = f
        for j in range(150):
            d = r.random() ** 1.6; y = 612 + d * 420; x = 1420 + r.normal(0, 1) * (70 + d * 420); ph = r.uniform(0, 6.28); k = max(0, math.sin(t * (.25 + .3 * r.random()) + ph)) ** 3
            if k > .08: w_ = (5 + 24 * d) * k; c.fill(np.array([(x - w_, y), (x, y - 1.2 - 3 * d * k), (x + w_, y), (x, y + 1.2 + 3 * d * k)]), '#fff0c0', .85)
        gl = c.rgba(); img = screen(img, cv2.GaussianBlur(gl[..., :3], (0, 0), 3.2) * 1.2)
        c = Cel(W, H, ss=1); ship(c, 205 - f * .5, 566 + 2 * math.sin(f * .12) - (z - 1) * 40, 330 * zb); img = over(img, cv2.GaussianBlur(c.rgba(), (0, 0), 5.0))
        r = np.random.default_rng(6); c = Cel(W, H, ss=1)
        for j in range(5): x = 300 + 260 * j + f * (1.6 + .4 * j); y = 190 + 60 * math.sin(j * 2.1) + 12 * math.sin(f * .1 + j); fl = 10 * math.sin(f * .55 + j * 1.4); c.line(spl([(x - 30, y - fl), (x - 12, y - 6 - fl * .3), (x, y), (x + 12, y - 6 - fl * .3), (x + 30, y - fl)], n=6), 5, '#5a3c62', taper=(.3, .3))
        img = over(img, cv2.GaussianBlur(c.rgba(), (0, 0), 3.0))
        # the child
        hk = dict(w=3.62, tilt=-.25 + .03 * math.sin(f * .17) - .35 * pitch * 0, up=-.115, lean=.03); sway = math.sin(f * .16)
        dap = [(x + .012 * sway + .006 * math.sin(f * .3 + i), y + .006 * math.cos(f * .21 + i * 2), a * (.75 + .35 * math.sin(f * .23 + i * 1.9)), b * (.8 + .3 * math.sin(f * .19 + i))) for i, (x, y, a, b) in enumerate(DAP)]
        kx = dict(st=st)
        R, Hd, E, Lt = L.record(cx, cy, s, -.10 + .012 * math.sin(f * .2), pitch, .035 + .01 * math.sin(g * .5) * (g >= 30), e, 'dusk', f, kid=True, hat=True, kx=kx, wind=.9, dapple=dap, body_dy=-hitch * .9 + (.05 if g in (28, 29) else 0))
        b = HT.settle(f, 30, .55) + HT.settle(f, 36, .28) + HT.settle(f, 44, .28) + HT.settle(f, 52, .28) + HT.settle(f, 10, .12) - (.18 if 27 <= f < 30 else 0)
        stt = HT.seat(Hd, **hk); kb = dict(wear=.32, light=LEAK, f=f, wind=.9, bounce=b); lay = HT.hat(**stt, part='back', **kb).copy(); lay = R.cel(Lt, s, lay)
        # deep shade under the brim so that the eyes are only guessed
        ce = Hd.P(np.array([(0., -.06)]))[0]; face = Hd.face(.7); m = Cel(W, H, ss=1); m.fill(spl([ce + [-s * 1.0, -s * .9], ce + [s * 1.0, -s * .9], ce + [s * .95, s * .02], ce + [s * .45, s * .20], ce + [0, s * .235], ce + [-s * .45, s * .20], ce + [-s * .95, s * .02]], True), '#ffffff'); mk = cv2.GaussianBlur(m.rgba()[..., 3], (0, 0), 2.0)[..., None] * .62
        lay[..., :3] = lay[..., :3] * (1 - mk) + lay[..., :3] * rgb('#5a2c3c') * mk
        # water light dancing up under the brim
        ca = caustic(f); gy = smooth(ce[1] - s * .35, ce[1] + s * .1, grid()[1]) * (1 - .45 * smooth(ce[1] + s * .9, ce[1] + s * 1.6, grid()[1])); hb = np.clip(1.15 - np.abs(grid()[0] - cx) / (s * 1.6), 0, 1)
        mc = Hd.P(np.array([(.0, -.64)]))[0]; gx_, gy_ = grid(); hb = hb * (1 - .96 * np.exp(-(np.abs(gx_ - mc[0]) / (s * .50)) ** 4 - (np.abs(gy_ - mc[1]) / (s * .42)) ** 4)) * (1 - .62 * smooth(mc[1] + s * .34, mc[1] + s * .62, gy_))      # nose, mouth and chin stay clean skin
        lay[..., :3] += (ca * gy * hb * lay[..., 3])[..., None] * rgb('#ffd890') * .33
        lay = L.over4(lay, HT.hat(**stt, part='front', **kb))
        img = over(img, lay)
        img = self.tear(img, f, Hd, s)
        if f >= 31:                                    # drops shaken loose
            c = Cel(); r = np.random.default_rng(3)
            for j in range(20):
                t0 = 31 + (j % 4) * 7; k = f - t0 - r.uniform(0, 2); sd = -1 if j % 2 else 1; a = r.uniform(-.2, .9); v = r.uniform(14, 30)
                if 0 < k < 11:
                    p = Hd.P(np.array([(sd * .55, -.45)]))[0] + np.array([sd * math.cos(a) * v * k, -math.sin(a) * v * k + 1.7 * k * k]); R_ = r.uniform(5, 13); an = math.atan2(-math.sin(a) * v + 3.4 * k, sd * math.cos(a) * v); D = spl(np.array([(1.5, 0), (.5, .62), (-.6, .66), (-1.0, 0), (-.6, -.66), (.5, -.62)]) * R_, True, n=5) @ np.array([[math.cos(an), math.sin(an)], [-math.sin(an), math.cos(an)]]) + p
                    c.fill(D, '#fff3df', .9); c.line(D, 2.0, '#4a2430', closed=True); c.fill(p + (D - p) * .35 - [R_ * .2, R_ * .3], '#ffffff')
            img = over(img, c.layer())
        a = 9 * math.exp(-(f - 30) / 3.) if f >= 30 else 0
        if a > .3: sx, sy = shake(f, a, 5); img = cam2d(img, 960 + sx, 540 + sy, 1.015, 0)
        if f == 30: img = smear(img, 0, -34, 9)
        return img.astype(f32)

    def tear(self, img, f, Hd, s):
        """rolls to the chin, hangs, carries the ship upside down, falls at f26"""
        path = np.array([(.50, -.36), (.56, -.52), (.50, -.72), (.34, -.90), (.16, -1.0), (.05, -1.02)]); P = Hd.P(np.clip(path, -.995, 9), .012); P = spl(P, n=12); d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        if f < 17: u = float(lerp_keys(f, [(0, 0.0), (6, .20), (10, .30), (17, 1.0)])); p = np.array([np.interp(u * d[-1], d, P[:, i]) for i in (0, 1)]); r = s * (.034 + .012 * u); drop = p; stx = 1.15
        else: p = P[-1]; k = smooth(16, 24, f); r = s * (.046 + .040 * k); drop = p + [0, r * (.55 + .75 * k)]; stx = 1.15 + .40 * k + .05 * math.sin(f * 1.1)
        if f >= 26: k = f - 25; drop = p + [0, r * 1.3 + 9.0 * k + 8.5 * k * k]; stx = 1.25 if k > 1 else 1.7
        if drop[1] - r > H + 60: return img
        # wet trail
        if f < 30:
            u = 1. if f >= 17 else np.interp(f, [0, 6, 10, 17], [0, .2, .3, 1]); n = max(3, int(u * len(P))); c = Cel(); w = np.full(n, s * .013); T = P[:n]; c.line(T + [1.5, 0], s * .030, '#fff0d8', taper=(.1, .1), alpha=.40, press=.4, seed=3); c.line(T + [-s * .008, 0], s * .008, '#ffffff', taper=(.3, .2), alpha=.9, press=.6, seed=4); img = over(img, c.layer())
        cx, cy = float(drop[0]), float(drop[1]); h = r * stx; t = np.linspace(0, 2 * math.pi, 60, endpoint=False); yy = np.sin(t); xx = np.cos(t) * (1 - .36 * np.clip(-yy, 0, 1) ** 1.3); Q = np.c_[cx + xx * r, cy + np.where(yy < 0, yy * h * 1.25, yy * h * .92)]
        x0, y0, x1, y1 = [int(v) for v in (max(0, cx - r * 1.4), max(0, cy - h * 1.6), min(W, cx + r * 1.4), min(H, cy + h * 1.4))]
        if x1 - x0 < 4 or y1 - y0 < 4: return img
        c = Cel(x1 - x0, y1 - y0, x0, y0); c.fill(Q, '#ffffff'); a = c.rgba()[..., 3:4]; Y, X = np.mgrid[y0:y1, x0:x1].astype(f32); nx, ny = (X - cx) / r, (Y - cy) / (h * 1.05); d2 = nx * nx + ny * ny; k = 1 + .9 * d2
        u, v = 900 - nx * 400 * k, 560 - ny * 330 * k; q = cv2.remap(self.src, u.astype(f32), v.astype(f32), cv2.INTER_AREA if 0 else cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        ed = smooth(.60, 1.02, np.sqrt(d2))[..., None]; col = q * 1.05 * (1 - ed * .55) + rgb('#5a2c48') * ed * .42; bt = np.exp(-((nx + .10) / .55) ** 2 - ((ny - .74) / .20) ** 2)[..., None]; col = col + bt * rgb('#ffe9b0') * .75
        sp = np.clip((1 - np.sqrt(((nx + .38) / .20) ** 2 + ((ny + .30) / .34) ** 2)) * 5, 0, 1)[..., None]; col = col * (1 - sp) + sp * 1.1
        if f >= 27: ky = 26
        img = img.copy(); img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - a) + col * a; c = Cel(x1 - x0, y1 - y0, x0, y0); c.line(Q, max(2., s * .0062), '#3a1826', closed=True, press=.6, seed=2); L_ = c.rgba(); img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - L_[..., 3:4]) + L_[..., :3]
        return img
