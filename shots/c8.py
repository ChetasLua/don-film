"""c8 Brook: indigo card, the moon opens like an eye; one long bow stroke on the yard; a scarred whale rises behind him with a straw-hat skull on its brow; the bow flick cuts the night in two."""
from chars.gang_shot import *
from chars import brook as BR
from op import fx_air

MOON = (1228, 392, 318); HZ = 800


def make_sky():
    x, y = grid(OH, OW); o = np.array([OW / 2 - 960, OH / 2 - 540]); mx, my = MOON[0] + o[0], MOON[1] + o[1]; t = y / OH; g = ramp(t, [(0, '#04051a'), (.30, '#0b0f36'), (.55, '#182260'), (.70, '#34458f'), (.74, '#5568b0')]); d = np.hypot(x - mx, y - my); g = g + (np.exp(-d / 420.) * .40)[..., None] * rgb('#8fa8ff') + (np.exp(-d / 1100.) * .12)[..., None] * rgb('#5a7af0'); n = fbm(OH, OW, (150, 60, 24, 10), (1, .7, .5, .3), 8, 3.2); band = np.exp(-((y - my - 150 + (x - mx) * .10) / 74.) ** 2) + .8 * np.exp(-((y - my + 250 - (x - mx) * .06) / 50.) ** 2) + .9 * np.exp(-((y - my - 330) / 46.) ** 2); cl = smooth(.15, 1.0, n * .55 + band * 1.25 - .35)
    g = painted(np.clip(g, 0, 1), 4, (46, 22), .08 * np.sin(x / 300.)) * .7 + np.clip(g, 0, 1) * .3; c = Cel(OW, OH, ss=1); r = np.random.default_rng(3)
    for _ in range(260): xx, yy = r.uniform(0, OW), r.uniform(0, OH * .62); s = .5 + 1.9 * r.random() ** 3; c.fill(ell(xx, yy, s, s, 8), '#eaf0ff', r.uniform(.25, .95))
    g = comp.over(g, c.rgba()); m = smooth(MOON[2] + 1.5, MOON[2] - 1.5, d); mar = fbm(OH, OW, (110, 46, 18), (1, .6, .3), 21); mc = ramp(np.clip(.62 + .22 * np.tanh(mar * 1.2) - .14 * smooth(.3, 1., d / MOON[2]), 0, 1), [(.3, '#8f9cc8'), (.5, '#bcc6e4'), (.62, '#e4e4e6'), (.9, '#f6f2e2')]); g = lerp2(g, mc * .93, m); lit = np.exp(-d / 460.); cc = ramp(np.clip(lit * 1.5, 0, 1), [(0, '#1b2462'), (.4, '#4557ab'), (.75, '#a9b8ea'), (1, '#eaf0ff')]); return lerp2(g, cc, cl * (.75 - .15 * m))


def make_sea():
    x, y = grid(OH, OW); o = np.array([OW / 2 - 960, OH / 2 - 540]); hz = HZ + o[1]; v = np.clip((y - hz) / 380., 0, 1.3); g = ramp(v, [(0, '#5f74c0'), (.10, '#2b3c8e'), (.45, '#141d5c'), (1, '#080b30')]); w = fbm(OH, OW, (26, 10, 5), (1, .7, .5), 5, 7.) ; g = g * (1 + .16 * w[..., None]); gl = smooth(.30, 1.35, fbm(OH, OW, (12, 6, 3.2), (1, .7, .5), 12, 6.) + .55 - .25 * v) * np.exp(-((x - MOON[0] - o[0]) / (120 + 360 * v)) ** 2); g = g + gl[..., None] * rgb('#eef2ff') * (.95 - .35 * v)[..., None]; a = smooth(hz - 1, hz + 1, y); return np.dstack([painted(np.clip(g, 0, 1.2), 6, (12, 6), np.zeros((OH, OW), f32), aspect=7.) * .6 + g * .4, np.ones_like(a)]) * a[..., None]


WH = [(590, 1500), (572, 1080), (576, 880), (600, 700), (660, 540), (772, 416), (940, 336), (1150, 304), (1380, 322), (1590, 394), (1780, 516), (1940, 690), (2070, 900), (2160, 1150), (2220, 1500)]


def make_whale():
    o = np.array([OW / 2 - 960, OH / 2 - 540]); A = spn(WH, True, (0, 14), 14) + o; c = Cel(OW, OH, ss=1); c.fill(A, '#232b5e'); g, a = split(c.rgba()); x, y = grid(OH, OW); X, Y = x - o[0], y - o[1]
    dx, dy = (X - 1270) / 720., (Y - 900) / 640.; nz = np.sqrt(np.maximum(0, 1 - dx * dx - dy * dy)); lit = .55 * np.clip(-dy, 0, 1) + .20 * (1 - nz) - .12 * dx + .05 * fbm(OH, OW, (120, 40), (1, .5), 3, 1.6); g = ramp(lit, [(-.2, '#0b0f30'), (.10, '#131a48'), (.22, '#1d2763'), (.34, '#2b3a82'), (.44, '#3f54a4'), (.62, '#6f88d2')]); g = painted(np.clip(g, 0, 1).astype(f32), 9, (34, 16, 8), np.arctan2(dx, -dy) * 1.0 + .0, bristle=.08) * .75 + g * .25
    pale = smooth(-6, 6, Y - (968 + 60 * ((X - 1270) / 600.) ** 2 - 40)); gro = .5 + .5 * np.sin((X - 1270) / (19 + .010 * (Y - 960)) + .0012 * (Y - 960) * (X - 1270) / 40.); bel = ramp(np.clip(.25 + .45 * gro + .25 * (-dx) + .1, 0, 1), [(0, '#1b2358'), (.4, '#33417e'), (.8, '#6676b4'), (1, '#93a2d6')]); g = lerp2(g, bel, pale)
    P = np.dstack([g * a[..., None], a]); c = Cel(OW, OH, ss=2); S = lambda Pts, w, col, tp=(.25, .35), **k: c.line(spn([(u, v) for u, v in Pts], False) + o, w, col, taper=tp, **k)
    S([(640, 930), (820, 1000), (1060, 960), (1290, 948), (1520, 968), (1740, 1010), (1920, 940)], 9, '#060820'); S([(650, 940), (820, 1012), (1060, 972), (1290, 960)], 3, '#8fa0dc')
    for u in np.linspace(700, 1860, 16): S([(u, 1010 + 22 * math.sin(u / 90.)), (1270 + (u - 1270) * 1.04, 1200), (1270 + (u - 1270) * 1.06, 1400)], 4, '#0b0f34')
    r = np.random.default_rng(8)                             # scars of years of butting the mountain
    for i in range(86):
        a0 = r.uniform(-1.35, 1.3); q = r.uniform(.28, .94); cx, cy = 1270 + math.sin(a0) * 640 * q, 900 - math.cos(a0) * 580 * q
        if (abs(cx - 1140) < 240 and abs(cy - 600) < 215) or (cx < 880 and cy > 700): continue
        th = a0 + r.normal(0, .6) + (1.57 if r.random() < .35 else 0); L = r.uniform(40, 150); d = np.array([math.cos(th), math.sin(th)]); m = np.array([-d[1], d[0]]); b = r.normal(0, 10); Pts = [tuple(np.array([cx, cy]) - d * L / 2), tuple(np.array([cx, cy]) + m * b), tuple(np.array([cx, cy]) + d * L / 2)]; S([(u + 2, v + 4) for u, v in Pts], r.uniform(9, 17), '#0a0e2e'); S(Pts, r.uniform(6, 11), '#93a6e2'); S([(u - 1, v - 2) for u, v in Pts], 2.6, '#e2eaff')
        for t in (-.3, 0., .3) if L > 95 else (): e = np.array([cx, cy]) + d * L * t; S([tuple(e - m * 16), tuple(e + m * 16)], 3.4, '#8fa2dc')
    # the promise: a badly painted skull in a straw hat
    ex = lambda Pts: [(1140 + u * 1.12, 610 + v * 1.12) for u, v in Pts]; pw, pd = '#f4f0e0', '#0e1238'
    for sd in (-1, 1): S(ex([(-sd * 150, -70), (-sd * 40, 30), (sd * 60, 92), (sd * 152, 130)]), 34, pw, (.04, .04), tremor=6., press=.5); [c.fill(ell(*(np.array(ex([(q, w)])[0]) + o), 25, 22, 18), pw) for q, w in ((-sd * 160, -88), (-sd * 168, -58), (sd * 160, 118), (sd * 152, 146))]
    c.fill(spn(ex([(-112, -18), (-120, -70), (-90, -118), (0, -140), (92, -116), (120, -66), (108, -10), (70, 34), (62, 84), (30, 100), (-34, 98), (-66, 82), (-72, 34)]), True) + o, pw)
    for sd in (-1, 1): c.fill(spn(ex([(sd * 24, -22), (sd * 50, -50), (sd * 84, -34), (sd * 80, 6), (sd * 46, 18)]), True) + o, pd)
    c.fill(ell(*(np.array(ex([(2, 30)])[0]) + o), 14, 11, 16), pd); S(ex([(-62, 56), (0, 72), (62, 54)]), 7, pd, (.1, .1)); [S(ex([(u, 60 + 6 * abs(u) / 60 * -1 + 4), (u + 1, 96)]), 6, pd, (.1, .1)) for u in (-40, -14, 14, 40)]
    c.fill(spn(ex([(-196, -78), (-120, -104), (-100, -164), (-40, -206), (40, -208), (102, -168), (118, -106), (198, -84), (120, -66), (0, -58), (-120, -62)]), True, (0, 7)) + o, '#e9c65a'); c.fill(spn(ex([(-118, -104), (0, -84), (118, -106), (112, -136), (0, -116), (-110, -136)]), True, (0, 2, 3, 5)) + o, '#c0392f'); S(ex([(-192, -78), (-60, -62), (60, -62), (194, -84)]), 6, pd, (.1, .1)); S(ex([(-100, -164), (-40, -206), (40, -208), (102, -168)]), 5, pd, (.2, .2))
    for u, v, L in ((-150, 130, 60), (-60, 100, 84), (34, 104, 110), (150, 148, 52), (-196, -70, 40)): S(ex([(u, v), (u + 2, v + L)]), 9, pw, (.02, .6))
    # the eye, low on the near side, bright with water
    ey = lambda P, co=(): spn([(742 + (u - 722) * 1.0, 770 + (v - 800) * 1.0) for u, v in P], True, co) + o; es = lambda P, w, col, tp=(.25, .35): S([(742 + (u - 722), 770 + (v - 800)) for u, v in P], w, col, tp)
    c.fill(ey([(624, 806), (664, 758), (736, 742), (806, 768), (842, 812), (776, 850), (694, 852)]), '#060820'); c.fill(ey([(656, 806), (692, 780), (744, 772), (796, 792), (816, 812), (766, 832), (700, 832)]), '#dfe7ff'); EM = ey([(656, 806), (692, 780), (744, 772), (796, 792), (816, 812), (766, 832), (700, 832)]); c.clip(EM); c.fill(ey([(690, 806), (706, 776), (744, 766), (780, 780), (790, 808), (766, 836), (716, 838)]), '#0a1030'); c.fill(ey([(716, 818), (744, 806), (776, 818), (756, 840), (728, 840)]), '#2f4aa8'); c.fill(ey([(620, 740), (860, 740), (840, 800), (780, 786), (736, 782), (690, 790), (640, 812)], (0, 1)), '#141c4a'); c.fill(ell(726 + 20 + o[0], 794 - 30 + o[1], 13, 11, 18), '#ffffff'); c.fill(ell(764 + 20 + o[0], 818 - 30 + o[1], 5, 5, 12), '#ffffff'); c.unclip(); es([(648, 812), (690, 786), (736, 779), (786, 786), (826, 808)], 9, '#060820', (.1, .2)); es([(700, 838), (750, 842), (804, 822)], 5, '#bfe0ff'); es([(640, 776), (700, 738), (770, 742), (836, 774)], 6, '#0a0e2e'); es([(660, 866), (740, 884), (820, 858)], 5, '#0a0e2e'); es([(668, 862), (740, 878), (806, 856)], 2.5, '#8fa2dc')
    c.fill(ey([(700, 838), (716, 862), (712, 900), (694, 914), (678, 898), (682, 862)], (0,)), '#cfe6ff'); c.fill(ey([(692, 880), (700, 896), (690, 904), (684, 892)]), '#ffffff')
    l = c.rgba(); m = cv2.erode(a, np.ones((5, 5))); l = l * m[..., None]; tex = np.clip(1.02 - .12 * np.maximum(0, fbm(OH, OW, (18, 6), (1, .6), 12)), .6, 1.05)[..., None]; l[..., :3] *= tex * (rgb('#d4dcff')); l *= np.clip(1.15 - .30 * np.maximum(0, fbm(OH, OW, (9, 4), (1, .6), 13) - .9), 0, 1)[..., None]
    P = over4(P, l); P = kit.rim(P, 3, 13, '#eef4ff', 1., 1.6); P = kit.rim(P, -6, 3, '#6f8ae8', .6, 1.4); s2 = blur(np.clip(a - kit.shifted(a, 6, 46), 0, 1), 16) * a; P[..., :3] += (s2 * .5)[..., None] * rgb('#9fb8ff'); return P


def make_yard():
    o = np.array([OW / 2 - 960, OH / 2 - 540]); L = []; yd = lambda x: 906 + (x - 600) * .035
    L.append((np.array([[-400, yd(-400) - 34], [2400, yd(2400) - 26], [2400, yd(2400) + 34], [-400, yd(-400) + 46]]) + o, '#4a2f2c')); L.append((np.array([[-400, yd(-400) + 10], [2400, yd(2400) + 8], [2400, yd(2400) + 34], [-400, yd(-400) + 46]]) + o, '#22151f')); L.append((np.array([[96, -300], [196, -300], [212, 1500], [84, 1500]]) + o, '#3a2427')); L.append((np.array([[150, -300], [196, -300], [212, 1500], [160, 1500]]) + o, '#6f4a44'))
    r = np.random.default_rng(5); xs = np.arange(-400, 2400, 128)
    for i, x in enumerate(xs): w = r.uniform(120, 150); L.append((spn([(x - 10, yd(x) + 26), (x + w * .5, yd(x) + 18), (x + w + 10, yd(x) + 30), (x + w * .9, yd(x) + 96 + r.uniform(0, 26)), (x + w * .5, yd(x) + 118 + r.uniform(0, 20)), (x + w * .1, yd(x) + 92)], True) + o, ['#5c66a4', '#6b76b4', '#525b98'][i % 3])); L.append((spn([(x + w * .5, yd(x) + 36), (x + w, yd(x) + 34), (x + w * .9, yd(x) + 96), (x + w * .5, yd(x) + 116), (x + w * .62, yd(x) + 70)], True) + o, '#2c3168'))
    for x in xs: L.append((band_poly(np.array([[x, yd(x) - 36], [x + 10, yd(x) + 40], [x - 6, yd(x) + 120]], float), 13) + o, '#b09a86'))
    P = plate_rgba(L, 16, (16, 8, 4), None, .7); P[..., :3] *= rgb('#8088b8'); P = kit.rim(P, 1., 4.5, '#e8efff', .9, .8); c = Cel(OW, OH, ss=2)
    for a, b in (((150, -200), (-300, 800)), ((146, 520), (330, 905)), ((2060, -60), (2260, 905)), ((1980, -60), (2180, 905))): c.line(np.array([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 20), b], float) + o, 3, '#0a0d2a', taper=(.02, .02), press=0)
    return over4(c.rgba(), P)


class Shot:
    id = 'c8'; n = 30; finish = dict(bloom=.34, vig=.22, halation='#cfdcff')
    EVENTS = [(0, 'card', dict(note='indigo card')), (0, 'violin', dict(note='one long bow stroke f0-20')), (5, 'glint', dict(note='tone dial (spiral shell) on the yard by his foot catches the moon; the lone voice plays from it')), (2, 'whoosh', dict(size=.3, note='moon iris opens')), (5, 'rumble', dict(note='sea lifts, whale rises f5-24')), (9, 'water', dict(note='sheets of water off the brow')), (18, 'whale', dict(note='low whale call answers the violin')), (25, 'impact', dict(size=.8, note='bow flick, DON, white arc')), (25, 'slash', {}), (26, 'split', dict(note='night slides apart, deep blue'))]

    def setup(self):
        self.sky = cached('c8_sky', make_sky); self.sea = cached('c8_sea', make_sea); self.wh = cached('c8_whale', make_whale); self.yd = cached('c8_yard', make_yard); self.card = cached('c8_card', lambda: card('#1c2460', '#3f52ad', '#0c1034', 81, -.4)); self.nxt = cached('c8_next', lambda: card('#0d3158', '#1f6a9c', '#061a34', 91, -.2))

    def rise(self, f): return key(f, [(0, 760.), (5, 690.), (12, 270.), (18, 30.), (23, -86.), (25, -70.), (29, -80.)], lambda t: t * t * (3 - 2 * t) if t < 0 else 1 - (1 - t) ** 1.7)

    def water(self, f, dy, dx, dz):
        """water running off the brow and the boil where the body breaks the sea"""
        c = Cel(ss=1); r = np.random.default_rng(40); L = spn(WH[1:13], False, (), 24); x0 = 960; wl = HZ + 60 + dz; gush = math.exp(-max(0, f - 9) / 9.) * (1 if f > 5 else 0)
        for i in range(64):
            u = r.uniform(.04, .96); p = L[int(u * (len(L) - 1))] + np.array([dx, dy]); a = abs(u - .5) * 2; ph = r.uniform(0, 1); sp = r.uniform(60, 120); ln = r.uniform(140, 420) * (.5 + .8 * a); s0 = ((f * sp / 400. + ph) % 1.)
            p = p * [1, 1] + np.array([(1270 - p[0]) * r.uniform(.0, .30), (960 - p[1]) * r.uniform(0., .22) * (1 if abs(p[0] - 1270) > 260 else .3)]); on = r.random() < .30 + gush
            if p[1] > wl - 20 or not on: continue
            x = p[0] + r.normal(0, 16); y0 = max(p[1] + 8, p[1] + s0 * ln * .8 - 60); y1 = min(wl + 20, y0 + ln * (.5 + .5 * gush)); sl = (x - 1270) * .0002
            if y1 - y0 < 30: continue
            w = r.uniform(8, 44) * (.35 + .65 * gush); t = np.linspace(0, 1, 7); c.fill(band_poly(np.c_[x + sl * (y1 - y0) * t * t + 4 * np.sin(t * 7 + i), y0 + (y1 - y0) * t], [w * .25, w, w * .8, 1.]), ['#dfe9ff', '#b7c9f6', '#8fa6e6'][i % 3], .30 + .28 * gush)
        l = c.layer(); c = Cel(ss=1)
        hw = 420 + 320 * min(1, (760 - dy) / 760.); bo = 1 - max(0, f - 14) / 26.
        for lay, (col, k, a_) in enumerate((('#4a5fae', 1.0, .9), ('#8fa4e2', .78, .92), ('#dfe8ff', .52, .95), ('#ffffff', .30, 1.))):
            m = 26; top = []
            for i in range(m + 1): u = -1 + 2 * i / m; n = .35 + .65 * abs(math.sin(i * 2.3 + lay * 1.7)) * abs(math.sin(i * .9 + lay + f // 2 * .45)); top.append((1275 + dx + u * hw * (1.04 - .05 * lay), wl + 10 - (26 + 120 * bo) * k * n * (1 - u ** 4)))
            P_ = scallop(spn(top, False, (), 6), 46 - 7 * lay, 15 - 2 * lay, 6 + lay + f // 2, closed=False); P_ = P_ if P_[:, 1].mean() < wl + 40 else spn(top, False, (), 6); c.fill(np.vstack([P_, [[1275 + dx + hw, wl + 26], [1275 + dx + hw * .5, wl + 50], [1275 + dx, wl + 58], [1275 + dx - hw * .5, wl + 50], [1275 + dx - hw, wl + 26]]]), col, a_)
        for i in range(90):
            t0 = r.uniform(5, 22); age = f - t0
            if age < 0 or age > 13: continue
            a = -math.pi / 2 + r.normal(0, .55); v = r.uniform(14, 46); px = 1275 + dx + r.uniform(-1, 1) * hw * .95 + math.cos(a) * v * age; py = wl - 20 + math.sin(a) * v * age + 1.9 * age * age; s = r.uniform(1.5, 7.5); c.fill(ell(px, py, s * .8, s, 10), '#f4f8ff' if i % 3 else '#b9caf6', .9)
        fo = c.layer(); y = grid()[1]; lace = smooth(-.9, .9, fbm(H, W, (46, 17, 7), (1, .7, .5), 60 + f // 3, 3.0)); fade = np.clip(1 - smooth(wl - 6, wl + 40, y) * .96 + smooth(wl - 6, wl + 30, y) * 0, 0, 1); fo = fo * (fade + (1 - fade) * lace * .55)[..., None]; return over4(l, fo)

    def spout(self, f, wy, dx):
        """the answer: breath and sea thrown up against the moon"""
        g = f - 17
        if g < 0: return None
        c = Cel(ss=1); r = np.random.default_rng(77); x0, y0 = 1180 + dx, 306 + wy
        for i in range(150):
            t0 = r.uniform(0, 7); age = g - t0
            if age < 0: continue
            a = -math.pi / 2 + r.normal(0, .22) * (1 + .4 * r.random()); v = r.uniform(26, 62); v_ = lambda s: np.array([x0 + math.cos(a) * v * s * 1.1 + .22 * np.sign(math.cos(a)) * s * s, y0 + math.sin(a) * v * s + 2.0 * s * s]); s = r.uniform(3, 15) * r.random() + 2.2; p = v_(age)
            if p[1] > y0 + 60: continue
            c.line(np.array([v_(max(0, age - 1.2)), v_(max(0, age - .5)), p]), s * 1.3, ['#f4f8ff', '#cfdcff', '#9fb6f2'][i % 3], taper=(.7, .1), press=0, alpha=.9)
        h = [60, 150, 230, 290, 330, 350, 350, 340, 330, 320, 310, 300, 300][min(g, 12)]; m = scallop(spn([(x0 - 26, y0 + 8), (x0 - 34, y0 - h * .35), (x0 - 84, y0 - h * .72), (x0 - 120, y0 - h * .92), (x0 - 40, y0 - h * 1.06), (x0 + 46, y0 - h * 1.04), (x0 + 124, y0 - h * .88), (x0 + 78, y0 - h * .66), (x0 + 32, y0 - h * .34), (x0 + 24, y0 + 8)], True), 40, 14, 3); l = c.layer(); c = Cel(ss=1); c.fill(m, '#dfe8ff', .55); c.fill(m * .62 + np.array([x0, y0 - h * .45]) * .38, '#ffffff', .55); return over4(blur(c.layer(), 3.), l)

    def dial(self, f, ox, oy, k):
        """a tone dial left on the yard by his foot: the crew's song is inside it"""
        x, y = ox + 118 * k, oy - 22 * k; R = 25 * k; p = Part(Rig(x, y, 60), (x - 90, y - 90, x + 90, y + 70), 8600, k=2.1); B = p.shape(spn([(x - R * 1.15, y + R * .80), (x - R * 1.25, y + R * .1), (x - R * .9, y - R * .75), (x - R * .1, y - R * 1.05), (x + R * .75, y - R * .72), (x + R * 1.15, y - R * .05), (x + R * 1.75, y + R * .50), (x + R * 2.3, y + R * .62), (x + R * 1.75, y + R * .86), (x + R * .6, y + R * .95)], True, (0, 7)), '#f0dfc6', .9); p.shade(B + [-R * .35, R * .42], '#a3869a', B); p.shade(B + [-R * .75, R * .85], '#5f4d78', B)
        t = np.linspace(0, 3.3 * math.pi, 60); sp = np.c_[x - R * .10 + R * .82 * (1 - t / 11.2) * np.cos(t + 2.9), y - R * .08 + R * .76 * (1 - t / 11.2) * np.sin(t + 2.9)]; p.l.line(sp, 2.2 * k, '#5a2f3c', taper=(.05, .3)); p.f.line(sp + [1.5 * k, -1.5 * k], 2.4 * k, '#fff6e6', taper=(.1, .4)); [p.l.line(np.array([[x + R * (.95 + .22 * j), y - R * .05 + R * .20 * j], [x + R * (.90 + .24 * j), y + R * .88 - R * .03 * j]]), 1.3 * k, '#6a3c48', taper=(.2, .3)) for j in range(4)]; p.f.fill(ell(x - R * .35, y + R * 1.0 - R * .35, R * .20, R * .13, 12), '#c2573f'); p.ol(ell(x - R * .35, y + R * .65, R * .20, R * .13, 12), .3); a, ax, ay = p.done(light('moon', rim=[(UR, '#f4f8ff', 1., 1.2)]), 3.2); L = np.zeros((H, W, 4), f32); L[ay:ay + a.shape[0], ax:ax + a.shape[1]] = a; return L, x + R * .35, y - R * .62

    def brook(self, f, ox, oy, k, li):
        t = f if f >= 20 else f // 2 * 2; C = BR.C; st = Stack(); X = lambda u, v: np.array([ox + u * k, oy + v * k]); P = lambda L, co=(): spn([tuple(X(u, v)) for u, v in L], True, co, 8)
        br = key(t, [(0, 0.), (10, 1.), (20, -.1), (23, -.55), (25, 1.35), (27, 1.1), (29, 1.2)]); sw = 9 * math.sin(t * .30); lean = key(t, [(0, -6), (10, 22), (19, -16), (23, -38), (25, 46), (28, 30)]); g = 14 * math.sin(f * .62) + (40 if f >= 25 else 0); wd = lambda i: g * (.5 + .5 * math.sin(f * .55 + i * 1.3)) + 10 * math.sin(f * .4 + i)
        hipx, hipy = 0, -308; shx, shy = lean * .8 + sw * .4, -468
        p = Part(Rig(ox, oy, 84 * k), None, 8200, k=2.7)
        # coat tails behind
        for i, (u0, u1, L_) in enumerate(((-58, 6, 268), (-10, 62, 250))): e = -120 - 46 * i + wd(i) * 1.6 - lean; Q = p.shape(P([(u0, hipy - 60), (u1, hipy - 60), (u1 + 8 + wd(i + 1) * .3, hipy + 90), (e + 96, hipy + L_ * .82), (e + 20, hipy + L_ + 12 + wd(i) * .3), (e + 34, hipy + L_ * .72), (u0 - 20 + wd(i) * .5, hipy + 100)], (0, 1, 4)), C['coat'], 1.); p.shade(Q + [16 * k, 12 * k], C['coatS'], Q); p.line(np.array([X((u0 + u1) / 2, hipy - 20), X((u0 + u1) / 2 - 16 + wd(i) * .3, hipy + 100), X(e + 50, hipy + L_ * .75)]), .5)
        # legs
        for i, (hx, kx, ky, fx, fy) in enumerate(((-20, -48 - lean * .2, -156, -62, -10), (24, 64 - lean * .3, -166, 58, -4))):
            Q = p.shape(union(capsule(X(hx, hipy), X(kx, ky), 27 * k, 17 * k, 4 * k), capsule(X(kx, ky), X(fx, fy - 14), 17 * k, 13 * k, 4 * k), P([(fx - 13, fy - 30), (fx + 14, fy - 30), (fx + 22, fy), (fx - 18, fy)], (0, 1, 2, 3))), '#3a3550', 1.); p.shade(Q + [13 * k, 2 * k], '#1b1830', Q); p.line(np.array([X(kx - 10, ky - 4), X(kx + 8, ky + 6)]), .4); Q = p.shape(P([(fx - 16, fy - 16), (fx + 12, fy - 18), (fx + 40, fy - 2), (fx + 44, fy + 9), (fx - 18, fy + 9)], (0, 3, 4)), '#15121f', .9)
        # torso: long coat, waistcoat, cravat
        T_ = [(shx - 78, shy + 10), (shx - 40, shy - 14), (shx + 44, shy - 14), (shx + 80, shy + 10), (shx + 62, shy + 90), (46, hipy - 30), (52, hipy + 26), (0, hipy + 40), (-50, hipy + 26), (-44, hipy - 30), (shx - 60, shy + 90)]; Q = p.shape(P(T_), C['coat'], 1.05); p.shade(Q + [44 * k, 16 * k], C['coatS'], Q); W_ = p.shape(P([(shx - 20, shy - 8), (shx + 24, shy - 8), (shx + 20, shy + 70), (10, hipy + 10), (-10, hipy + 10), (shx - 18, shy + 70)], (0, 1)), '#e8e4ee', .7); p.shade(W_ + [18 * k, 0], '#8f88ac', W_)
        for sd in (-1, 1): p.shape(P([(shx + sd * 22, shy - 14), (shx + sd * 56, shy + 4), (shx + sd * 38, shy + 60), (shx + sd * 12, shy + 120), (shx + sd * 16, shy + 50)], (0, 3)), '#3d3858', .7)
        for i, (u, v, w) in enumerate(((0, -6, 26), (0, 16, 32), (-2, 40, 28))): q = P([(shx - w, shy + v), (shx, shy + v + 8), (shx + w, shy + v - 2), (shx + w * .8, shy + v + 22), (shx, shy + v + 30), (shx - w * .8, shy + v + 22)], (0, 2)); p.shape(q, C['crav'], .6); p.shade(q + [10 * k, 8 * k], C['cravS'], q)
        st.add(p, li, 8.); p = Part(Rig(ox, oy, 84 * k), None, 8300, k=2.6)
        vo = X(shx + 96, shy - 40); vi = BR.violin(p, vo, .10 + lean * .002, 158 * k)
        # left arm holds the neck (viewer right)
        s2, e2, h2 = X(shx + 74, shy + 12), X(shx + 150, shy + 108), vi['neck'] + [-8 * k, 12 * k]; Q = union(capsule(s2, e2, 27 * k, 20 * k, 4 * k), capsule(e2, h2 + [-6 * k, 22 * k], 20 * k, 14 * k)); p.shape(Q, C['coat'], 1.); p.shade(Q + [12 * k, 10 * k], C['coatS'], Q); cf = P([tuple((h2 - X(0, 0)) / k + np.array(v)) for v in ((-24, 22), (-8, 2), (14, 10), (10, 34), (-12, 40))]); p.shape(scallop(cf, 13 * k, 4 * k, 5), '#f3ecef', .6)
        for j in range(4): p.f.line(np.array([h2 + [(-10 + 8 * j) * k, 4 * k], h2 + [(-12 + 8 * j) * k, -12 * k], h2 + [(-8 + 8 * j) * k, -24 * k]]), 5.6 * k, C['bone'], taper=(.05, .3)); p.l.line(np.array([h2 + [(-14 + 8 * j) * k, 2 * k], h2 + [(-16 + 8 * j) * k, -12 * k], h2 + [(-10 + 8 * j) * k, -26 * k]]), .4 * p.k, INK)
        st.add(p, li, 7.)
        # head: tilted on the violin, then lifted
        ht = key(t, [(0, .26), (8, .30), (16, .16), (22, .08), (25, -.16), (28, -.10)]); hx, hy = shx + 14 + ht * 30, shy - 104 - (10 if t > 15 else 0); e = pick(BR.EX, 'hush' if t < 12 else ('hush', 'sing', min(1, (t - 12) / 8.)) if t < 25 else 'laugh'); e['tear'] = key(f, [(0, .0), (15, .0), (20, .7), (29, 1.0)]); hT = Rig(*X(hx, hy), 80 * k, key(t, [(0, .42), (14, .38), (20, .50), (25, .20)]), key(t, [(0, -.14), (14, -.06), (20, .16), (25, .10)]), ht, .10, r=(.78, 1.1, .74)); st.a = over4(st.a, BR.draw(hT, li, e, f, hat='top', hats=.88, wind=-.12 + .06 * math.sin(f * .5) + (-.12 if f >= 25 else 0), squash={24: .05, 25: -.08, 26: .05}.get(f, .015 * math.sin(f * .6)), hatdy={25: -.16, 26: -.10, 27: .02}.get(f, 0.)))
        # bow arm
        p = Part(Rig(ox, oy, 84 * k), None, 8400, k=2.6); sh = X(shx - 72, shy + 16)
        if f < 24: d = np.array([math.cos(-.98 + .10 * br), math.sin(-.98 + .10 * br)]); fr = vi['bridge'] + d * 0; hd = fr - d * (60 + 170 * max(-.3, br)) * k; tip = hd + d * 330 * k
        elif f == 24: hd = X(shx - 190, shy + 190); d = np.array([-.55, .83]); tip = hd + d * 320 * k
        else: a = {25: -.75, 26: -1.05, 27: -1.16, 28: -1.12, 29: -1.1}[f]; d = np.array([math.cos(a), math.sin(a)]); hd = X(shx + (60 if f == 25 else 20), shy - 190 + (0 if f == 25 else -30)); tip = hd + d * 340 * k
        el = (sh + hd) / 2 + (np.array([-74, 34]) * k if f < 24 else np.array([-40, 10]) * k); Q = union(capsule(sh, el, 27 * k, 20 * k, 4 * k), capsule(el, hd, 20 * k, 13 * k)); p.shape(Q, C['coat'], 1.); p.shade(Q + [12 * k, 10 * k], C['coatS'], Q); dd = (hd - el) / (np.hypot(*(hd - el)) + 1e-6); cf = scallop(ell(*(hd - dd * 6 * k), 20 * k, 17 * k, 24), 12 * k, 4 * k, 6); p.shape(cf, '#f3ecef', .6)
        m = np.array([-d[1], d[0]]); p.f.line(np.array([hd - d * 26 * k, hd + d * 170 * k - m * 12 * k, tip]), 6.5 * k, '#7a4426', taper=(.05, .3), press=0); p.l.line(np.array([hd - d * 26 * k, hd + d * 170 * k - m * 12 * k, tip]) - m * 3 * k, .4 * p.k, INK); p.f.line(np.array([hd - d * 20 * k + m * 9 * k, tip + m * 5 * k]), 2.4 * k, '#fbf6e8', taper=(.0, .0), press=0)
        for j in range(4): b = hd + d * (-10 + 8 * j) * k; p.f.line(np.array([b - m * 9 * k + dd * 2 * k, b + m * 4 * k, b + m * 11 * k + d * 3 * k]), 6 * k, C['bone'], taper=(.05, .3)); p.l.line(np.array([b - m * 9 * k + d * 3 * k, b + m * 6 * k + d * 3 * k]), .4 * p.k, INK)
        st.add(p, li, 7.); return st.a, vi, hd, tip

    def frame(self, f):
        z = key(f, [(0, 0.), (29, 1.)]); sk = kit.shake(f, 8, 8, 2., 25) if f >= 25 else (kit.shake(f, 1.6, 8) if 8 <= f < 20 else np.zeros(2)); li = light('moon'); dy = self.rise(f)
        im = view(self.sky, 10 * z + sk[0] * .3, -16 * z + sk[1] * .3, 1.03); wy = dy - 30 + 10 * z; im = comp.over(im, view(self.sea, 14 * z, -14 * z, 1.03)) if False else im
        wh = view(self.wh, 18 * z + sk[0] * .5, -wy + sk[1] * .5, 1.02); wl = HZ + 60 - 8 * z; y = grid()[1]; vis = smooth(wl + 40, wl + 10, y)[..., None]; far = view(self.sea, 14 * z, -14 * z, 1.03); im = comp.over(im, far)
        if f > 2: im = comp.over(im, wh * vis); ref = cv2.flip(wh, 0); sh_ = int(2 * wl - H); ref = cv2.warpAffine(ref, np.float32([[1, 0, 0], [0, 1, sh_]]), (W, H), borderMode=cv2.BORDER_CONSTANT); ref = kit.smear(ref, 26, 0, 5); rp = np.sin(y * .42 + f * .9) * 6; ref = cv2.remap(ref, grid()[0] + rp.astype(f32), y, cv2.INTER_LINEAR); im = comp.over(im, ref * smooth(wl, wl + 14, y)[..., None] * .42)
        sp = self.spout(f, wy, -18 * z)
        if sp is not None: im = comp.over(im, sp)
        im = comp.over(im, self.water(f, wy, -18 * z, -10 * z)); hz_ = fx_air.mist(f, 9, 730, 900, '#a9bcf4', .38, 22.); im = comp.over(im, hz_)
        mo = comp.radial(MOON[0] - 10 * z, MOON[1] + 16 * z, 760, power=2.2)[..., None]; im = im + mo * rgb('#8fa8ff') * .10
        if f < 9: im = self.open(im, f)
        yd = view(self.yd, 40 * z + sk[0], -98 - 10 * z + sk[1], 1.02 + .02 * z); k = 1.31 + .03 * z; ox, oy = 520 - 40 * z - sk[0], 1012 + 10 * z - sk[1]; bk, vi, hd, tip = self.brook(f, ox, oy, k, li)
        dl, gx, gy = self.dial(f, ox, oy, k)
        im = kit.aura(im, bk, 10, '#8fa8ff', .25); im = comp.over(im, yd) if f >= 4 else comp.over(im, yd * min(1., 1.)); im = comp.over(im, dl); im = comp.over(im, bk)
        gl = {3: .12, 4: .55, 5: 1.25, 6: .85, 7: .45, 8: .2, 9: .08}.get(f, 0.)
        if gl: im = kit.flare(im, gx, gy, 70, '#f4f8ff', gl, .9); c = Cel(ss=1); rr = 30 * min(1.2, gl); c.fill(np.array([[gx - rr, gy], [gx - 3, gy - 3], [gx, gy - rr], [gx + 3, gy - 3], [gx + rr, gy], [gx + 3, gy + 3], [gx, gy + rr], [gx - 3, gy + 3]]), '#ffffff'); im = comp.over(im, c.layer())
        # the note itself: light off the string, dust of rosin
        if f < 24: im = im + (np.exp(-((grid()[0] - vi['bridge'][0]) ** 2 + (grid()[1] - vi['bridge'][1]) ** 2) / 5200.))[..., None] * rgb('#fff1c8') * (.28 + .10 * math.sin(f * 2.2)); im = im + fx_air.motes(f, 70, 4, np.clip(comp.radial(vi['bridge'][0] + 60, vi['bridge'][1] - 60, 420, power=1.), 0, 1), '#fff0cf', (14., -22.), 1.1, .9)
        if f >= 25: im = self.cut(im, f)
        return im

    def open(self, im, f):
        """the moon is the iris"""
        cd = np.asarray(self.card, f32); R = [0, 0, 40, 150, 290, 480, 800, 1250, 2200][f]; x, y = grid(); d = np.hypot(x - MOON[0] + 4, y - MOON[1] - 4) + 7 * fbm(H, W, (40, 14), (1, .5), 6)
        if R == 0: return cd.copy()
        m = smooth(R + 2, R - 2, d); e = np.exp(-np.abs(d - R) / 8.) * (R < 1200); g = np.exp(-np.maximum(0, d - R) / 60.); return lerp2(cd, im, m) + e[..., None] * rgb('#fffbe6') * .9 + (g * (1 - m))[..., None] * rgb('#8fa8ff') * .35

    def cut(self, im, f):
        q = f - 25; t = np.linspace(0, 1, 60); P0, P1, P2 = np.array([-260., 1010.]), np.array([820., 1010.]), np.array([2180., -60.]); A = (1 - t)[:, None] ** 2 * P0 + 2 * ((1 - t) * t)[:, None] * P1 + (t * t)[:, None] * P2; T_ = np.gradient(A, axis=0); T_ /= np.hypot(*T_.T)[:, None]; N_ = np.c_[-T_[:, 1], T_[:, 0]]; gap = [0, 16, 90, 320, 1500][q]; nx = np.asarray(self.nxt, f32)
        if f == 25: im = kit.impact(im, 'inv', .16, .5, '#f4f6ff', '#0b1040') * np.array([1., 1., 1.], f32)
        up = np.vstack([A, [[W + 400, -400], [-400, -400]]]); mu = mask_soft(up); x, y = grid()
        if q >= 1:
            n = np.array([.42, .91]); sh = lambda s: cv2.warpAffine(im, np.float32([[1, 0, n[0] * s], [0, 1, n[1] * s]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); ms = lambda m, s: cv2.warpAffine(m, np.float32([[1, 0, n[0] * s], [0, 1, n[1] * s]]), (W, H), borderMode=cv2.BORDER_REPLICATE); m1 = ms(mu, -gap); m2 = ms(1 - mu, gap); out = nx.copy(); out = lerp2(out, sh(gap), m2); out = lerp2(out, sh(-gap), m1); im = out if q < 4 else nx.copy()
        if q < 3:
            w = [64, 30, 12][q]; c = Cel(); hd = min(1., .55 + .5 * q) if q else 1.; pr = np.sin(np.pi * np.clip(t, 0, 1)) ** .55; c.fill(np.vstack([A + N_ * (w * pr)[:, None], (A - N_ * (w * .55 * pr)[:, None])[::-1]]), '#ffffff'); L = c.layer(); im = im + blur(L[..., 3], 26)[..., None] * rgb('#8fb4ff') * .9 + blur(L[..., 3], 8)[..., None] * rgb('#dfeaff') * .6; im = comp.over(im, L); c = Cel(); r = np.random.default_rng(5)
            for i in range(60): u = int(r.uniform(5, 54)); v = r.normal(0, 1); p = A[u] + N_[u] * v * (30 + 40 * q) + T_[u] * r.uniform(20, 120) * (1 + q); s = r.uniform(2, 8); c.fill(band_poly(np.array([p - T_[u] * s * 6, p + T_[u] * s * 4]), lambda tt: s * np.sin(np.pi * tt) + .3), '#f4f8ff')
            im = comp.over(im, c.layer())
        return im
