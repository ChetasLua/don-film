"""c1  Zoro.  Green card; the white hilt in his teeth; a rain of cannonballs, one breath, and everything is already cut."""
from chars.swkit import *
from chars import zoro, swbody as BD
from op.clouds import Sky, tones

ZL = dict(key=-1, rims=[((-8, 4), '#eaffc2', 1.), ((6, 4), '#fff3d0', .7)], g=(('#fffaf0', '#cfb29a')), L=(-.5, .6, .6))
FL = dict(key=-1, rims=[((6, 5), '#fff3c8', 1.), ((-5, 3), '#d8ffd0', .6)], g=(('#fffaf0', '#c8b8a8')), L=(-.5, .65, .55))
BALLS = [(1560, 250, 150, -.50, 2), (330, 300, 128, .62, 0), (820, 170, 92, -.30, 1), (1190, 120, 66, .40, 3), (1800, 520, 110, .9, 4), (120, 600, 84, -.8, 5), (600, 90, 46, .2, 6), (1420, 60, 40, -.6, 7), (1010, 300, 38, .1, 8), (1690, 840, 190, -.25, 9), (200, 930, 235, .35, 10)]


def world():
    x, y = grid(); sky = ramp(y / 700., [(0, '#1d78c8'), (.45, '#58b4ec'), (.8, '#b9e6f6'), (1, '#f1fbf4')]); S = Sky(H, W, sky, light=(-.55, -.75, .4), seed=3)
    for q in ((300, 640, 620, 330, 1), (1080, 650, 520, 220, 2), (1640, 645, 720, 420, 3), (820, 655, 300, 110, 4)): S.bank(q[0], q[1], q[2], q[3], ('#ffffff', '#eef8fb', '#c4def0', '#93b9dc', '#6d97c6'), '#dff2f8', seed=q[4])
    im = S.paint(2); sea = ramp((y - 640) / 160., [(0, '#bfeae6'), (.08, '#3f9fb8'), (.5, '#1f7aa0'), (1, '#155a82')]); m = smooth(638, 641, y)[..., None]; im = im * (1 - m) + sea * m; c = Cel(ss=1); r = np.random.default_rng(6)
    for k in range(130): yy = 645 + 150 * r.uniform(0, 1) ** 1.5; xx = r.uniform(-40, W + 40); L = 6 + 60 * (yy - 640) / 150; c.line(np.array([(xx - L, yy), (xx, yy - 1), (xx + L, yy)]), 1 + 2.2 * (yy - 640) / 150, '#d5f4f2', taper=(.4, .4), alpha=.75, seed=k)
    for sx, s in ((1250, 1.), (1395, .7), (560, .8)):                        # the ships that fired
        c.fill(np.array([(sx - 70 * s, 628), (sx + 74 * s, 628), (sx + 58 * s, 652), (sx - 54 * s, 652)]), '#33506a'); [c.fill(spl([(sx + o * s, 530 - 10 * s), (sx + (o + 34) * s, 560), (sx + (o + 30) * s, 612), (sx + o * s, 616)], True, n=5), '#e6f0f2') for o in (-50, -6, 34)]; [c.fill(np.array([(sx + o * s - 1.5, 628), (sx + o * s - 1.5, 515), (sx + o * s + 1.5, 515), (sx + o * s + 1.5, 628)]), '#33506a') for o in (-50, -6, 34)]
        for k in range(6): a = np.linspace(0, 6.283, 18); R_ = r.uniform(10, 24) * s; c.fill(np.c_[sx + r.uniform(-70, 70) * s + np.cos(a) * R_, 622 - r.uniform(0, 34) * s + np.sin(a) * R_ * .8], '#eef3f0', .85)
    im = over(im, c.rgba()); hill = spl([(-40, 905), (260, 850), (620, 818), (1000, 812), (1400, 836), (1980, 880), (1980, 1120), (-40, 1120)], True, (6, 7, 0, 5), n=8); hm = poly_mask(hill); g = ramp((y - 810) / 280., [(0, '#b9e05a'), (.25, '#6fb83e'), (.7, '#2f7d36'), (1, '#174f2c')]); im = im * (1 - hm[..., None]) + g * hm[..., None]
    cv = Canvas(H, W, img=im); n = 5200; xx = r.uniform(-20, W + 20, n); yy = 812 + 280 * r.uniform(0, 1, n) ** .85; k = hm[np.clip(yy, 0, H - 1).astype(int), np.clip(xx, 0, W - 1).astype(int)] > .5; xx, yy = xx[k], yy[k]; o = np.argsort(yy); xx, yy = xx[o], yy[o]; t = (yy - 812) / 280; col = ramp(np.clip(t + r.normal(0, .18, len(t)), 0, 1), [(0, '#d6f07a'), (.3, '#7cc644'), (.65, '#3a8a38'), (1, '#1c5a30')])
    cv.strokes(arcs(xx, yy - 8 - 22 * t, -1.35 + r.normal(0, .25, len(t)), 18 + 60 * t, r.normal(.5, .4, len(t)), 6), 1.4 + 4.2 * t, col, op=.95, dry=.1, seed=4, shape='leaf'); im = cv.img
    c = Cel()                                                                # the signpost that sends you both ways
    P = lambda q: np.asarray(q, float) + [366, 600]
    cloth(c, P([(-11, -60), (9, -60), (13, 300), (-13, 300)]), '#8a623a', '#573820', [P([(0, -60), (12, -60), (14, 300), (2, 300)])], w=4.5, folds=[P([(-4, -20), (-5, 120), (-3, 260)])])
    for q, sg in (([(-190, -34), (-150, -66), (34, -58), (36, -4), (-152, 0)], -1), ([(-34, 22), (150, 14), (196, 52), (152, 84), (-32, 78)], 1)):
        B = P(q); cloth(c, B, '#c99a5e', '#8e6234', [np.vstack([B[[0 if sg < 0 else 4]], (B[-1] + [0, 0])[None], (B[3])[None], (B[3] + [0, -16])[None], (B[-1] + [0, -18])[None]])], '#e9c98e', [], w=5., folds=[[B[1] + [10 * sg * -1, 12], B[2] + [-12, 10]]], fw=1.8); m = B.mean(0) + [8 * sg, 0]
        emblem_swords(c, m[0], m[1] + 2, 22, '#3b2414')
    im = over(im, c.layer()); return im.astype(f32)


def ball(c, x, y, r, a, gap=0., slide=0., cut=True, hot=0.):
    u = A_(math.cos(a), math.sin(a)); n = A_(-u[1], u[0]); o = A_(x, y); lw = max(2., r * .030); B, D, M, Lt = '#2b2e3a', '#12131b', '#4d5364', '#aebccb'
    if not cut:
        t = np.linspace(0, 6.283, 50); P = np.c_[x + np.cos(t) * r, y + np.sin(t) * r]; cloth(c, P, B, D, [spl(np.array([(x + .2 * r, y - r), (x + .45 * r, y), (x - .1 * r, y + .5 * r), (x - r, y + .5 * r), (x - r * .4, y + 1.2 * r), (x + 1.3 * r, y + 1.1 * r), (x + 1.2 * r, y - r)]), True, n=5)], Lt, [spl(np.array([(x - .62 * r, y - .40 * r), (x - .30 * r, y - .70 * r), (x - .18 * r, y - .58 * r), (x - .50 * r, y - .28 * r)]), True, n=5)], w=lw); return
    for sg in (-1, 1):
        q = o + n * sg * gap + u * sg * slide; t = np.linspace(0, math.pi, 30) * sg; arc = q + np.outer(np.cos(t), u) * r + np.outer(np.sin(t), n) * r; cel_ = arc
        cloth(c, cel_, B, D, [spl(np.array([q + u * r * .2 + n * sg * r * .0, q + u * r * .3 + n * sg * r * .6, q + u * r * .1 + n * sg * r * 1.2, q + u * r * 1.2 + n * sg * r * 1.0, q + u * r * 1.2]), True, n=5)], Lt, [spl(np.array([q - u * r * .70 + n * sg * r * .30, q - u * r * .42 + n * sg * r * .66, q - u * r * .34 + n * sg * r * .54, q - u * r * .58 + n * sg * r * .22]), True, n=5)], w=lw)
        if sg > 0:
            tt = np.linspace(0, 6.283, 44); E = lambda k: q + np.outer(np.cos(tt), u) * r * k * .995 + np.outer(np.sin(tt), n) * r * .24 * k
            c.fill(E(1.), '#8f9bb0'); c.clip(E(1.)); c.fill(q + np.outer(np.cos(tt), u) * r * .9 + np.outer(np.sin(tt), n) * r * .2 + u * r * .35, '#c9d6e4'); c.fill(np.array([q - u * r - n * r * .3, q - u * r * .2 - n * r * .3, q + u * r * .05 + n * r * .3, q - u * r * .6 + n * r * .3]), '#f4faff'); c.unclip(); c.line(E(.70), lw * .55, '#5d6780', closed=True); c.line(E(.36), lw * .5, '#5d6780', closed=True); c.line(E(1.), lw, INK, closed=True)
            if hot > 0: c.line(E(1.)[20:44], lw * 1.6, '#ffffff', taper=(.2, .2), alpha=hot)


def boom(c, x, y, r, t, seed):
    """one burst: drawn lobes, hot core, ink"""
    rg = np.random.default_rng(seed); g = 1 - (1 - min(1, t)) ** 2.2
    for lay, (k, col, ink) in enumerate(((1.0, '#4a2a3c', '#140a12'), (.90, '#b5301a', None), (.80, '#f2701c', '#3a0f0a'), (.60, '#ffb52e', None), (.42, '#fff2a0', None), (.24, '#ffffff', None))):
        L = []
        for i in range(11): a = i / 11 * 6.283 + rg.uniform(-.2, .2); d = r * g * k * rg.uniform(.45, .80); R_ = r * g * k * rg.uniform(.28, .46); th = np.linspace(0, 6.283, 26); L.append(np.c_[x + math.cos(a) * d + np.cos(th) * R_, y + math.sin(a) * d * .9 - r * .12 * g * lay + np.sin(th) * R_ * .95])
        th = np.linspace(0, 6.283, 30); L.append(np.c_[x + np.cos(th) * r * g * k * .6, y + np.sin(th) * r * g * k * .55 - r * .12 * g * lay])
        for q in union(L, 1., sig=.8): c.fill(q, col); (outline(c, q, max(2.4, r * .02), ink, seed + lay) if ink else None)
    for i in range(16): a = rg.uniform(0, 6.283); d0, d1 = r * g * rg.uniform(.6, .9), r * g * rg.uniform(1.1, 1.7); u = A_(math.cos(a), math.sin(a)); m = A_(-u[1], u[0]) * r * .025; c.fill(np.array([[x, y] + u * d0 + m, [x, y] + u * d1, [x, y] + u * d0 - m]), '#ffd86a')


def crescent(c, cx, cy, R_, a0, a1, w, col='#ffffff', core=None, asp=.6, rot=0.):
    t = np.linspace(a0, a1, 60); v = np.linspace(0, 1, 60); wd = w * np.sin(np.pi * v ** 1.4) ** .9; O = np.c_[np.cos(t) * (R_ + wd), np.sin(t) * (R_ + wd) * asp]; I = np.c_[np.cos(t) * (R_ - wd * .35), np.sin(t) * (R_ - wd * .35) * asp]; P = xf(np.vstack([O, I[::-1]]), 1, 1, rot, cx, cy); c.fill(P, col); return P


class Shot:
    id = 'c1'; n = 30; finish = dict(bloom=.32, vig=.14)
    EVENTS = [(0, 'hit', dict(size=.4, note='card slap')), (4, 'eye_snap', dict(who='zoro')), (3, 'cannon', dict(note='distant volley, balls whistling in')), (9, 'sword_draw', dict(size=1.0)), (10, 'slash', dict(n=3)), (12, 'silence', dict(note='cut metal sings')), (25, 'impact', dict(size=1.0, note='balls burst, DON')), (26, 'slash', dict(note='picture splits, wipe'))]

    def setup(self):
        os.makedirs('cache', exist_ok=True); p = 'cache/c1_world.npy'
        if not os.path.exists(p): np.save(p + '.tmp.npy', world().astype(np.float16)); os.replace(p + '.tmp.npy', p)
        self.w = np.load(p).astype(f32); p = 'cache/c1_card.npy'
        if not os.path.exists(p):
            a = brush_card(['#0b3a22', '#157a3c', '#23a04c', '#39c060', '#8fe682'], -.42, 11, dark='#0d4a2a'); e = Cel(ss=1); emblem_swords(e, 820, 560, 620, '#ffffff'); m = e.rgba()[..., 0] * smooth(-.5, .4, fbm(H, W, (60, 18, 5), None, 4, aspect=3.)); a = a * (1 - .26 * m[..., None]) + rgb('#0a3d22') * (.26 * m)[..., None]; b = brush_card(['#b5480c', '#e8701a', '#f58a24', '#ffa53a', '#ffd06a'], -.35, 12, dark='#d9601a'); np.save(p + '.tmp.npy', np.array([a, b]).astype(np.float16)); os.replace(p + '.tmp.npy', p)
        q = np.load(p).astype(f32); self.card, self.next = q[0], q[1]; self._c = {}

    def bust(self, f):
        d = f if f in (3, 4, 5) else f // 2 * 2; k = d
        if k in self._c: return self._c[k]
        pit = float(lerp_keys(f, [(0, -.26), (2, -.28), (4, .05), (5, .03), (7, -.02), (10, -.02)])); dy = float(lerp_keys(f, [(0, 70), (2, 78), (4, -14), (5, -6), (7, 0), (10, 0)])); lid = 0. if f < 3 else (.4 if f == 3 else 1.12 if f == 4 else 1.)
        c = Cel(W + 500, H + 300, -250, -100); H_ = zoro.draw(c, 900, 500 + dy, 338, -.34, pit, -.06, 'bite', key=-1, f=d, body='coat', broll=-.02, bandana=True, tails=.6, swing=.55 * math.sin(f * .9) * math.exp(-max(0, f - 2) / 9.), lid=lid, wind=.6, seed=2); BD.mouth_sword(c, H_, 78, 104, None if f < 4 else (f - 4) / 4.5, 5.2, L=4.2, refl=f * .02); L = light_layer(c.layer(), ZL, 900, 500, 338)
        self.eye = H_.P([(-.345, -.005)])[0]; self._c = {k: L}; return L

    def frame(self, f):
        if f < 9:
            img = cam2d(self.card, 960 - 8 * f, 540 + 3 * f, 1.06 + .004 * f); L = self.bust(f); img = cast(img, L, 46, 40, '#04200f', .5, 22); img = aura(img, L, 22, '#c8ff9a', .35); z = 1.0 + .006 * f; img = over(img, cam2d(L, 960, 540, z, border=cv2.BORDER_CONSTANT))
            if f in (4, 5): e = self.eye; img = flare(img, e[0], e[1], 120 if f == 4 else 60, '#f0ffd0', .8, 1.6, .1)
            c = Cel()
            for i in range(26):                                              # wind of leaves and dust over the card
                r = np.random.default_rng(i); x = (r.uniform(0, 2300) + f * r.uniform(60, 130)) % 2300 - 200; y = r.uniform(0, H) + 18 * math.sin(f * .5 + i); s = r.uniform(5, 20); a = f * .3 + i; c.fill(spl([(x - s, y), (x, y - s * .45 * math.sin(a)), (x + s, y + s * .2), (x, y + s * .3 * math.sin(a))], True, (0, 2), n=4), '#d9ffae' if i % 3 else '#06301a', .9)
            img = over(img, c.layer())
            if f == 8: c = Cel(); crescent(c, 1300, 1500, 1350, -2.75, -1.2, 64, '#ffffff', asp=.9); img = over(img * .55, c.layer())
            return img
        return self.battle(f)

    def balls(self, f, pre=False):
        c = Cel(); t = max(0, f - 10)
        for x, y, r, a, k in sorted(BALLS, key=lambda q: q[2]):
            ap = 1 + .020 * (f - 10) * (r / 100.) ** .6; dx, dy = (x - 960) * (ap - 1), (y - 480) * (ap - 1) + 1.0 * t; e = 1 - math.exp(-t / 5.5); sl = r * (.10 * e + .026 * t); gp = r * (.03 + .015 * e + .011 * t)
            ball(c, x + dx, y + dy, r * ap, a, gp, sl, f >= 11, max(0, 1 - t / 6.))
        return c.layer()

    def battle(self, f):
        img = cam2d(self.w, 960 + 6 * math.sin(f * .2), 540 - .6 * f, 1.05 + .0030 * (f - 9)); ballsL = self.balls(f); t = f - 25
        if f in (9, 10):                                                     # the draw, two smear drawings
            img = img * .45 + rgb('#0a3d24') * .35; c = Cel(); speed_lines(c, -.40, 70, 7, (400, 1700), '#eaffd8', 5 + f, alpha=.8); speed_lines(c, -.40, 40, 6, (300, 1500), INK, 15 + f)
            img = over(img, ballsL); crescent(c, 820, 700, 560 + 160 * (f - 9), 2.6, 5.6, 90, '#ffffff', asp=.55, rot=-.4); crescent(c, 1120, 640, 620 + 200 * (f - 9), -2.4, .5, 110, '#eaffd0', asp=.5, rot=.35); crescent(c, 960, 560, 900 + 200 * (f - 9), -2.9, -.2, 70, '#b6ff9a', asp=.34, rot=.02)
            c2 = Cel(); BD.zoro_crouch(c2, f, 0, 0, None, face=True); S = c2.layer(); S = smear(S, 260 if f == 9 else 150, -120 if f == 9 else -70, 15); S = cam2d(S, 960 - (90 if f == 9 else 30), 560 + (40 if f == 9 else 12), 1.12 if f == 9 else 1.08, border=cv2.BORDER_CONSTANT); img = over(img, S, 1.); return over(img, c.layer())
        ov = {11: (1.085, 24), 12: (.975, -12), 13: (1.012, 5), 14: (.997, -2)}.get(f, (1., 0)); c = Cel(W, H); gl = ((f - 12) / 9.) if f < 22 else None; BD.zoro_crouch(c, f, 0, 2.2 * math.sin(f // 2 * .7), gl, wave=1.5); L = light_layer(c.layer(), FL, 960, 560, 150)
        ps = 1.0 + .0052 * max(0, f - 12); M = np.float32([[(2 - ov[0]) * ps, 0, 960 * (1 - (2 - ov[0]) * ps)], [0, ov[0] * ps, 990 * (1 - ov[0] * ps) + 12 * (ps - 1) * 10]]); L = cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_CUBIC)
        far = Cel()
        img = over(img, self.mask_balls(ballsL, f, 0)); a = L[..., 3]; sh = np.zeros_like(a); sh = cv2.warpAffine(a, np.float32([[1, -.9, 900], [0, -.10, 1080]]), (W, H)); img = img * (1 - .42 * blur(sh, 7)[..., None] * (grid()[1] > 900)[..., None])
        if f >= 25:
            c = Cel()
            for i, (x, y, r, an, k) in enumerate(BALLS): boom(c, x + (x - 960) * .1, y + (y - 480) * .1, r * 1.9, .55 + .18 * (f - 25), 40 + i)
            E = c.layer(); img = img * .8 + rgb('#ff9a3a') * .18; img = over(img, E); img = img + blur(E[..., 3], 60)[..., None] * rgb('#ff7a1e') * .35
        L2 = rim2(L, 7, 5, '#ffd27a', 1., .9) if f >= 25 else L
        if f == 11 or f == 12: img = img * (.7 if f == 11 else .9)
        img = over(img, L2)
        if f < 25: img = over(img, self.mask_balls(ballsL, f, 1))
        c = Cel()                                                            # cut lines hang in the air, sparks fall, grass blades fly
        if 11 <= f < 20:
            q = 1 - (f - 11) / 9.
            for P in ([(40, 1010), (760, 660), (1800, 150)], [(120, 120), (900, 560), (1880, 930)], [(20, 640), (900, 590), (1900, 640)]): B, *_ = limb(P, [0, 6 * q, 0], [0, 6 * q, 0]); c.fill(B, '#ffffff', min(1., q * 1.4))
        for i in range(46):
            r = np.random.default_rng(60 + i); tt = f - 11 + r.uniform(0, 3); x = r.uniform(0, W) + (tt * r.uniform(8, 26)); y = r.uniform(200, 1050) - tt * r.uniform(-6, 9); s = r.uniform(6, 24); a = tt * r.uniform(-.5, .5) + i; u = A_(math.cos(a), math.sin(a)) * s; m = A_(-u[1], u[0]) * .16; c.fill(np.array([[x, y] - u, [x, y] + m, [x, y] + u, [x, y] - m]), '#c8f07a' if i % 2 else '#3f8f3a')
        img = over(img, c.layer())
        if f == 25: return impact(img, 'bw', .25, .6, ink='#07140c', paper='#f4ffe8')
        if f >= 26: img = self.split(img, f)
        elif f in (11, 12): d = shake(f, 10); img = cam2d(img, 960 + d[0], 540 + d[1], 1.02)
        return img

    def mask_balls(self, L, f, front):
        y = grid()[1]; x = grid()[0]; m = np.zeros((H, W), f32)
        for q in ((1690, 840, 260), (200, 930, 300)): m = np.maximum(m, (np.hypot(x - q[0], y - q[1]) < q[2] * 1.15).astype(f32))
        m = m if front else 1 - m; return L * m[..., None]

    def split(self, img, f):
        t = f - 26; off = (16, 110, 330, 760)[t]; sl = (30, 130, 300, 520)[t]; a, b = A_(-40, 900), A_(1960, 130); d = (b - a) / np.hypot(*(b - a)); n = A_(d[1], -d[0]); n = n if n[1] < 0 else -n; x, y = grid(); sd = (x - a[0]) * n[0] + (y - a[1]) * n[1]; out = cam2d(self.next, 960, 540, 1.05 - .01 * t); c = Cel()
        for sg in (1, -1):
            o = n * sg * off + d * sg * sl * .5; m = smooth(-1, 1, ((x - o[0] - a[0]) * n[0] + (y - o[1] - a[1]) * n[1]) * sg); im2 = cv2.warpAffine(img, np.float32([[1, 0, o[0]], [0, 1, o[1]]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); lay = np.dstack([im2 * m[..., None], m]); sh = blur(lay[..., 3], 18); out = out * (1 - .45 * (np.clip(sh - lay[..., 3], 0, 1))[..., None]); out = over(out, lay)
            e0, e1 = a + o, b + o; w = (12, 9, 7, 5)[t]; B, *_ = limb([e0, (e0 + e1) / 2, e1], [w, w * 1.3, w], [w, w * 1.3, w]); c.fill(B, '#f4ffe0')
        if t == 0: B, *_ = limb([a, (a + b) / 2, b], [4, 30, 4], [4, 30, 4]); c.fill(B, '#ffffff')
        return over(out, c.layer())
