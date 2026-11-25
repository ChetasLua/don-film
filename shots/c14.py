"""c14  Torn paper: three strips rip open (right to left, like a manga page): Zoro f0, Luffy f15, Sanji f30.  Under a parchment painted with the crew's X."""
from chars.swkit import *
from chars import zoro, sanji
import importlib, functools

T1 = tear_line((712, -50), (578, 1130), 21, 12); T2 = tear_line((1392, -50), (1258, 1130), 22, 12)
L0 = np.array([(-60, -50), (-60, 1130)], float); R0 = np.array([(1990, -50), (1990, 1130)], float)
STR = dict(S=(L0, T1), L=(T1, T2), Z=(T2, R0)); RIP = dict(Z=0, L=15, S=30); CX = dict(S=(330, 520), L=(965, 540), Z=(1600, 520)); DIR = dict(Z=1, L=-1, S=1)
ZD = dict(key=-1, rims=[((-9, 5), '#e6ffb8', 1.), ((5, 3), '#fff0c8', .5)], g=(('#fffaf0', '#cfae9c')), L=(-.5, .6, .6))
SD = dict(key=1, rims=[((9, 5), '#fff3c0', 1.), ((-5, 4), '#fff2cf', .5)], g=(('#fffaf0', '#d8b29a')), L=(.28, .52, .8))


def luffy_layer(f, blink):
    try:
        from chars import luffy
        for e in ('glare', 'angry', 'calm'):
            try: return luffy.head(965, 590, 262, yaw=-.10, pitch=-.07, roll=.06, expr=e, light='day', f=f), True
            except (KeyError, ValueError): continue
    except Exception as ex: print('luffy placeholder', repr(ex)[:80])
    return placeholder(), False


@functools.lru_cache(1)
def placeholder():
    c = Cel(); a = np.linspace(0, 6.283, 80); P = spl([(700, 560), (800, 470), (965, 440), (1130, 470), (1240, 560), (965, 620)], True, n=10)
    H_ = zoro.make(True); H_.pose(965, 590, 262, -.1, -.07, .06); F = H_.face(); c.fill(F, '#f0b98d'); outline(c, F, 6)
    shape(c, spl([(560, 500), (780, 400), (965, 375), (1160, 400), (1390, 520), (1160, 470), (965, 455), (760, 470)], True, n=10), '#e3b557', 6); shape(c, spl([(770, 410), (800, 270), (965, 215), (1130, 270), (1165, 410), (965, 385)], True, n=10), '#e3b557', 6)
    return light_layer(c.layer(), 'day', 965, 590, 262)


class Shot:
    id = 'c14'; n = 60; finish = dict(bloom=.34, vig=.16)
    EVENTS = [(0, 'paper_rip', dict(size=1.0, note='Zoro strip, top to bottom', pan=.6)), (5, 'eye_snap', dict(note='steel ting + earrings', who='zoro')), (15, 'paper_rip', dict(size=1.0, note='Luffy strip, bottom to top', pan=0)), (20, 'eye_snap', dict(who='luffy')),
              (30, 'paper_rip', dict(size=1.0, note='Sanji strip', pan=-.6)), (35, 'eye_snap', dict(who='sanji', note='cigarette crackle')), (45, 'impact', dict(size=.8, note='rim lights flare, brass swell')), (56, 'paper_rip', dict(size=1.4, note='whole page torn off, right to left'))]

    def setup(self):
        p = 'cache/c14_plates.npy'
        if not os.path.exists(p):
            os.makedirs('cache', exist_ok=True); out = []
            for cols, ang, sd, kind in ((['#1a1004', '#4a2e08', '#8f5f10', '#d9a021', '#f7cf4e'], -.72, 5, 'S'), (['#200406', '#5a0a10', '#a3141c', '#d92a26', '#ff6a3c'], .62, 8, 'L'), (['#04140c', '#0c3a20', '#176a38', '#2fa050', '#8ce070'], -.58, 3, 'Z')):
                im = brush_card(cols, ang, sd, dark='#0a0508' if kind != 'Z' else '#030c08'); e = Cel(ss=1); x, y = CX[kind]
                if kind == 'S': emblem_cutlery(e, x - 6, y + 40, 470, '#ffffff')
                elif kind == 'L': emblem_skull(e, x + 6, y + 10, 500, '#ffffff', '#000000')
                else: emblem_swords(e, x - 20, y + 10, 560, '#ffffff')
                m = e.rgba()[..., 0]; g = fbm(H, W, (60, 22, 7, 3), None, sd + 9); dry = cv2.warpAffine(g, cv2.getRotationMatrix2D((960, 540), math.degrees(-ang), 1.), (W, H), borderMode=cv2.BORDER_REFLECT_101); dry = cv2.GaussianBlur(dry, (0, 0), sigmaX=9, sigmaY=.6)
                m = m * smooth(-.35, .30, dry / (dry.std() + 1e-6) + .45 * g); im = im * (1 - .50 * m[..., None]) + rgb(cols[-1]) * 1.12 * (.50 * m)[..., None]
                d = np.hypot((grid()[0] - x) / 520, (grid()[1] - y) / 720)[..., None]; im = im * (1.06 - .42 * np.clip(d - .25, 0, 1)); out.append(im.astype(np.float16))
            # parchment with the painted X
            g = fbm(H, W, (260, 90, 24, 6), None, 61)[..., None]; pa = rgb('#e2cda2') * (1 + .055 * g) + fbm(H, W, (3, 1.4), (1, .6), 4)[..., None] * .018; x_, y_ = grid()
            edge = np.clip(np.maximum(np.abs(x_ / 960 - 1), np.abs(y_ / 540 - 1)) - .55, 0, 1)[..., None]; pa = pa * (1 - .42 * edge ** 1.4 * np.array([.8, 1., 1.25], f32))
            for fx in (640, 1290): pa = pa * (1 - .10 * np.exp(-np.abs(x_ - fx) / 5.))[..., None] + .05 * np.exp(-np.abs(x_ - fx - 7) / 4.)[..., None]
            pa = pa * (1 - .10 * np.exp(-np.abs(y_ - 545) / 5.))[..., None]
            cv = Canvas(H, W, color='#17100e', layer=True); r = np.random.default_rng(4)
            for P in ([(330, 70), (700, 380), (1180, 700), (1610, 1010)], [(1600, 60), (1200, 400), (760, 720), (300, 1015)]):
                Q = resample(spl(P, n=12), 30.); ii = np.linspace(0, len(Q) - 1, 30).astype(int)
                for o, wd, dr in ((0, 62, .22), (-46, 20, .6), (44, 24, .6), (60, 9, .7), (-62, 8, .7)): cv.strokes((Q[ii] + [o * .62, -o * .62 if P[0][0] < 900 else o * .62])[None], wd, rgb('#1a1210')[None], op=.96, dry=dr, seed=int(wd * 7), shape='flat', bristle=.3)
            c = Cel(ss=1)
            for q in ((960, 540, 260), (420, 130, 120), (1560, 980, 150), (1480, 150, 110)): splatter(c, *q, 46, 11, '#1a1210', int(q[0]))
            pa = over(over(pa.astype(f32), cv.layer()), c.rgba()); out.append(np.clip(pa, 0, 1).astype(np.float16)); np.save(p + '.tmp.npy', np.array(out)); os.replace(p + '.tmp.npy', p)
        A = np.load(p, mmap_mode='r'); self.pl = dict(S=A[0].astype(f32), L=A[1].astype(f32), Z=A[2].astype(f32)); self.cover = A[3].astype(f32)
        self.poly = {k: np.vstack([a, b[::-1]]) for k, (a, b) in STR.items()}; self.m = {k: poly_mask(v) for k, v in self.poly.items()}
        self.e1, o1 = fibre_edge(T1, -1, 3); self.e2, o2 = fibre_edge(T2, -1, 4); self.e1b, _ = fibre_edge(T1, 1, 13, wid=(3, 10)); self.e2b, _ = fibre_edge(T2, 1, 14, wid=(3, 10))
        self.backp = np.clip(rgb('#f4efe4') * (1 + .03 * fbm(H, W, (90, 24, 5), None, 77)[..., None]), 0, 1); self._c = {}

    def char(self, k, f):
        """character layer for strip k at frame f (drawn on twos, eyes on ones round the snap)"""
        t = f - RIP[k]; snap = 5; d = f // 2; lid = 0. if t < snap - 1 else (.45 if t == snap - 1 else 1.12 if t == snap else 1.04 if t == snap + 1 else 1.)
        bl = 1.
        if 43 <= f <= 44: bl = .72                                           # a tightening before the flare
        u = max(0, t - snap); rec = .045 * math.exp(-u / 4.5) * math.cos(u * .9) if t >= snap else -.05
        key = (k, d, lid, bl, t if abs(t - snap) < 3 else -1)
        if key in self._c: return self._c[key]
        fl = 0. if f < 45 else .55 + .9 * math.exp(-(f - 45) / 5.); boost = 1. + fl * .25; rk = 1 + .45 * fl
        if k == 'Z':
            D = dict(ZD, rims=[((-6 * rk, 3 * rk), '#d9ffac' if f < 45 else '#efffd0', .9), ((5 * rk, 3 * rk), '#fff0c8', min(1., .5 + .5 * fl))]); sw = .22 * math.exp(-max(0, t) / 14.) * math.cos(t * .55) + .03 * math.sin(f * .16)
            L = zoro.head(1606, 486, 322, -.42, -.065 + rec, .03 - rec * .3, 'glare', D, d, broll=0., swing=sw, lid=lid * bl if lid else 0., look=(0, 0) if t >= snap else (.06, -.03), wind=.25 + (1.2 * math.exp(-max(0, t) / 3.) if t >= 0 else 0), boost=boost, hairpal=None)
        elif k == 'S':
            D = dict(SD, rims=[((9 * rk, 5 * rk), '#fff3c0' if f < 45 else '#fffbe2', 1.), ((-5 * rk, 4 * rk), '#fff2cf', min(1., .5 + .5 * fl))])
            L = sanji.head(318, 492, 310, .22, -.045 + rec, -.05 + rec * .4, 'glare', D, d, broll=0., lid=lid * bl if lid else 0., wind=.10 * math.sin(f * .11) + (-.5 * math.exp(-max(0, t) / 3.) if t >= 0 else 0), boost=boost, ember=1. if f >= 45 else .4)
        else:
            L, ok = luffy_layer(d, lid)
            if fl > 0:
                for dx, dy, c in ((-8, 5, '#fff3d2'), (8, 5, '#fff3d2')): L = rim2(L, dx * rk, dy * rk, c, min(1., .75 * fl), .9)
        self._c = {q: v for q, v in self._c.items() if q[0] != k}; self._c[key] = L; return L

    def strip(self, k, f):
        t = f - RIP[k]; cx, cy = CX[k]; rate = dict(Z=.00065, L=.00115, S=.0016)[k]; z = 1.0 + rate * max(0, t); jolt = 1 + .030 * math.exp(-max(0, t) / 2.2) + (.018 * math.exp(-(f - 45) / 4.) if f >= 45 else 0)
        bg = cam2d(self.pl[k], cx + (960 - cx) * 0, cy, 1.0, w=W, h=H) if False else self.pl[k]
        M = np.float32([[1 + (z - 1) * .45, 0, cx * (1 - (1 + (z - 1) * .45)) - 2 + .10 * t], [0, 1 + (z - 1) * .45, cy * (1 - (1 + (z - 1) * .45))]]); bg = cv2.warpAffine(bg, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        L = self.char(k, f); zz = z * jolt; M = np.float32([[zz, 0, cx * (1 - zz)], [0, zz, cy * (1 - zz)]]); L = cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        col = dict(Z='#8cff9a', L='#ff8a5a', S='#ffd860')[k]; fl = 0. if f < 45 else .35 + .8 * math.exp(-(f - 45) / 5.); bg = cast(bg, L, 16 if k != 'S' else -16, 14, '#020202', .55, 10); bg = aura(bg, L, 26, col, .20 + .55 * fl)
        out = over(bg, L); y = grid()[1]; dk = (.50 * smooth(640, 1090, y))[..., None]; out = out * (1 - dk * (1 - rgb(dict(Z='#0f3a2a', L='#5a0a14', S='#5a3208')[k]) * .6)); hx, hy = dict(Z=(1500, 420), L=(965, 520), S=(430, 430))[k]; lit = radial(hx, hy, 430, 520, 1.6)[..., None]; return out * (1 + .16 * lit) + rgb(col) * lit * .02

    def rip(self, img, k, f):
        """cover sheet still hiding strip k, peeling"""
        t = f - RIP[k]; a, b = STR[k]
        if t >= 6: return img, None
        if t < 0: m = self.m[k][..., None]; return img * (1 - m) + self.cover * m, None
        q = (.16, .42, .74, 1.02, 1.3, 1.6)[t]; dn = DIR[k] > 0; yf = -60 + q * 1220 if dn else 1140 - q * 1220; r = np.random.default_rng(90 + t + RIP[k])
        n = 40; u = np.linspace(0, 1, n); slope = 90 if dn else -90; fr = lambda y, w: np.interp(y, w[:, 1], w[:, 0])
        Y = yf + slope * (u - .5) + noise1(n, 5, r) * 9; X = fr(yf, a) + (fr(yf, b) - fr(yf, a)) * u; FR = np.c_[X, Y]
        far = [(fr(1130, b), 1200), (fr(1130, a), 1200)] if dn else [(fr(-50, b), -120), (fr(-50, a), -120)]
        keep = poly_mask(np.vstack([FR[:1] - [400, 0], FR, FR[-1:] + [400, 0], [(far[0][0] + 400, far[0][1]), (far[1][0] - 400, far[1][1])]])) * self.m[k]; img = img * (1 - keep[..., None]) + self.cover * keep[..., None]
        L_ = (150, 250, 300, 240, 150, 90)[t]; sg = 1 if dn else -1; wv = noise1(n, 7, r); FL = np.c_[X + (u - .5) * -40 + 30 * sg * q, Y + sg * (L_ * (.75 + .25 * np.sin(u * 3.1)) + wv * 16)]
        P = np.vstack([FR - [0, sg * 10], FL[::-1]]); msk = poly_mask(P); sh = blur(shifted(msk, 10, sg * 34), 16); img = img * (1 - .55 * (sh * (1 - msk))[..., None])
        y = grid()[1]; v = np.clip((y - (yf + slope * ((grid()[0] - X[0]) / (X[-1] - X[0] + 1e-6) - .5))) * sg / L_, -.1, 1.1); tone = ramp(v, [(-.1, '#b9b0a6'), (.05, '#ffffff'), (.18, '#f4efe6'), (.45, '#c9c0b4'), (.75, '#e9e3d8'), (1.1, '#9e948a')])
        img = img * (1 - msk[..., None]) + tone * self.backp * msk[..., None]
        c = Cel(); c.line(FR - [0, sg * 10], 3.2, '#8d857e', taper=(.1, .1), press=.5); outline(c, P, 2.6, '#3a3230', 4, press=.5)
        for i in range(26): j = int(r.integers(1, n - 1)); q0 = FL[j]; c.line(np.array([q0, q0 + [r.normal(0, 8), sg * r.uniform(6, 26)]]), 1.4, '#ffffff', taper=(.1, .7), seed=i)
        return over(img, c.layer()), None

    def scraps(self, img, f):
        c = Cel()
        for k, t0 in RIP.items():
            t = f - t0
            if not 0 <= t < 22: continue
            a, b = STR[k]
            for i in range(26):
                r = np.random.default_rng(1000 + t0 * 7 + i); y0 = r.uniform(0, 1080); e = (a if r.random() < .5 else b); x0 = float(np.clip(np.interp(y0, e[:, 1], e[:, 0]), -20, 1940)); tt = t - r.uniform(0, 3)
                if tt < 0: continue
                vx, vy = r.normal(0, 26), r.normal(-9, 14) * DIR[k]; x = x0 + vx * 5 * (1 - math.exp(-tt / 5.)) + 8 * math.sin(tt * .5 + i); y = y0 + vy * 5 * (1 - math.exp(-tt / 5.)) + .55 * tt ** 2; s = r.uniform(4, 22) * (1 if r.random() < .85 else 1.8); rot = r.uniform(0, 6) + tt * r.normal(0, .35); sq = abs(math.cos(tt * .4 + i)) * .85 + .15; m = 7; an = np.linspace(0, 6.283, m, endpoint=False); R_ = s * r.uniform(.6, 1.2, m)
                P = np.c_[np.cos(an) * R_, np.sin(an) * R_ * sq]; P = P @ np.array([[math.cos(rot), math.sin(rot)], [-math.sin(rot), math.cos(rot)]]) + [x, y]; lit = math.cos(tt * .4 + i) > 0; c.fill(P + [4, 6], '#000000', .30); c.fill(P, '#fbf7ee' if lit else '#c2b7a8'); outline(c, P, 1.6, '#4a403a', i)
        return over(img, c.layer())

    def frame(self, f):
        img = np.zeros((H, W, 3), f32)
        for k in ('S', 'L', 'Z'):
            m = self.m[k][..., None]
            img = img + (self.strip(k, f) if f >= RIP[k] else self.cover) * m
        sh = np.zeros((H, W), f32)
        for k in ('Z', 'L', 'S'): img, _ = self.rip(img, k, f)
        # torn edges appear with the first strip that touches them; the raised sheet throws a shadow on the opened one
        for E, e, eb, ks in ((T2, self.e2, self.e2b, ('Z', 'L')), (T1, self.e1, self.e1b, ('L', 'S'))):
            t = max(f - RIP[q] for q in ks)
            if t < 0: continue
            g = np.ones((H, W), f32)
            if t < 5:
                q = (.16, .42, .74, 1.02, 1.3)[t]; k0 = min(ks, key=lambda z: RIP[z]); g = smooth(0, 30, (-60 + q * 1220) - grid()[1]) if DIR[k0] > 0 else smooth(0, 30, grid()[1] - (1140 - q * 1220))
            for lay, dx in ((e, -9), (eb, 8)): img = img * (1 - .62 * (blur(shifted(lay[..., 3], dx, 12), 9) * g)[..., None])
            img = over(img, eb * g[..., None]); img = over(img, e * g[..., None])
        if f >= 45:                                                          # a blade of light runs down the two tears
            q = (f - 45) / 8.; y = grid()[1]
            for e in (self.e1, self.e2): img = img + (e[..., 3] * np.exp(-((y - q * 1300) / 130.) ** 2))[..., None] * rgb('#fff4d0') * .9
        img = self.scraps(img, f)
        if f in (RIP['Z'], RIP['L'], RIP['S']) or f - 1 in RIP.values() or f == 45:
            d = shake(f, 7); img = cv2.warpAffine(img, np.float32([[1.012, 0, d[0] - 11], [0, 1.012, d[1] - 6]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        if f >= 56: img = self.tearoff(img, f)
        return np.maximum(img, 0)

    def tearoff(self, img, f):
        t = f - 56; q = (.14, .42, .76, 1.12)[min(t, 3)]; x, y = grid(); xf_ = 2050 - q * 2500; r = np.random.default_rng(700 + t); n = 60; u = np.linspace(0, 1, n); X = xf_ + 300 * u + noise1(n, 7, r) * 16 + noise1(n, 2, r) * 5; Y = -40 + 1160 * u
        sky = ramp(y / H, [(0, '#3f9fe6'), (.55, '#a9dcf7'), (.8, '#f4fbff'), (1, '#8fd0f0')]); m = poly_mask(np.vstack([np.c_[X, Y], [(2600, 1120), (2600, -40)]]))
        out = img * (1 - m[..., None]) + sky * m[..., None]; L_ = (260, 420, 420, 300)[min(t, 3)]; B = np.c_[X - L_ * (.7 + .3 * np.sin(u * 3)) + noise1(n, 8, r) * 22, Y - 40 * (u - .5)]; P = np.vstack([np.c_[X + 14, Y], B[::-1]]); msk = poly_mask(P)
        out = out * (1 - .6 * (blur(shifted(msk, -46, 20), 20) * (1 - msk) * (1 - m))[..., None]); v = np.clip((np.interp(y, Y, X)[..., None][..., 0] - x) / L_, -.1, 1.1); tone = ramp(v, [(-.1, '#a8a098'), (.04, '#ffffff'), (.2, '#f2ede4'), (.5, '#c6bdb2'), (.8, '#ece6dc'), (1.1, '#978d84')]); out = out * (1 - msk[..., None]) + tone * self.backp * msk[..., None]
        c = Cel(); outline(c, P, 3., '#3a3230', 6, press=.5)
        for i in range(70): j = int(r.integers(1, n - 1)); c.line(np.array([[X[j], Y[j]], [X[j] + r.uniform(6, 30), Y[j] + r.normal(0, 9)]]), 1.5, '#ffffff', taper=(.1, .7), seed=i)
        return over(out, c.layer())
