"""g4 (84): cream.  The sun contracts into the dot of the "!", a brush writes ドン, the last stroke is the last drum, a straw hat comes home to the dot."""
import numpy as np, cv2, math, os
from op import W, H, don, kit, sumi
from op.paint import rgb, f32, smooth, grid, fbm
from op.ink import Cel, spl, resample
from op.comp import radial
from op.fx_core import h1
from shots import dragonkit as DK

PW, PH = 2440, 1420
SUN = np.array([1470., 815.]); D0 = np.array([960., 486.]) - SUN
T_DO = [(15, 20), (22, 25.2), (27, 28.6), (29.6, 31.2)]; T_N = [(33, 35.6), (38.4, 45)]; T_EX = (58.4, 60.4)
CREAM = '#f5e8cb'


class Shot:
    id = 'g4'; n = 84; finish = dict(bloom=.10, vig=.12, soft=.05)
    EVENTS = [(0, 'sun', dict(note='sun contracts', dur=18)), (16, 'pop', dict(size=.3, note='becomes the dot')), (15, 'brush', dict(size=.8, note='do 1', dur=5)), (22, 'brush', dict(size=.6, note='do 2', dur=3)), (27, 'brush', dict(size=.4, note='dakuten')), (30, 'brush', dict(size=.4, note='dakuten')),
              (33, 'brush', dict(size=.5, note='n 1', dur=2)), (38, 'brush', dict(size=.9, note='n 2, long sweep', dur=7)), (52, 'cloth', dict(size=.2, note='hat flutters down', dur=14)), (60, 'impact', dict(size=1., note='last stroke, last drum')), (62, 'heart', dict(size=.7)), (67, 'heart', dict(size=.7)), (67, 'straw', dict(size=.3, note='the hat lands'))]

    def setup(self):
        c = os.path.join(DK.ROOT, 'cache', 'dr_g4_page_4.npz')
        if not os.path.exists(c):
            pg = sumi.Page(PH, PW, seed=3)
            for g, box, ts in ((don.DO, (92, 150, 800), T_DO), (don.N_, (700, 290, 660), T_N), ([don.STEM], (1078, 160, 800), [T_EX])):
                for q, t in zip(g, ts): don.write(pg, [q], box, [t], seed=len(pg.marks) + 2, wt=.96)
            for (x, y, R, a, n, t) in ((470, 900, 120, 1.8, 60, 19.6), (1350, 350, 150, -.8, 70, 44.6), (1470, 200, 230, -1.6, 80, 60.), (1480, 560, 260, None, 120, 60.2)): pg.add(sumi.splash(x, y, R, seed=int(x + y), ang=a, spread=.9 if a is not None else math.pi, n=n, big=12, fingers=7, t0=t, t1=t + 1.6, mist=.7, h=PH, w=PW))
            sumi.page_save(pg, c)
        self.pg = sumi.page_load(c, PH, PW)
        for m in self.pg.marks: m.d = m.d.astype(f32); m.wet = m.wet.astype(f32)
        self.paper = DK.full('g4_paper', lambda: sumi.paper(PH, PW, 23, CREAM, 1.25))
        # sun texture: pigment dragged round by a wide dry brush
        S = 900; x, y = grid(S, S); x = (x - S / 2) / (S / 2); y = (y - S / 2) / (S / 2); r = np.sqrt(x * x + y * y); a = np.arctan2(y, x); n = fbm(S, S, (120, 50, 18), (1, .6, .4), 4); nf = fbm(S, S, (9, 3.5), (1, .8), 5)
        pr = cv2.remap(fbm(S, S, (3.2, 60), (1, .5), 9, aspect=26.), ((a / 6.2832 + .5) * (S - 1)).astype(f32), ((r * .9) * (S - 1)).astype(f32), cv2.INTER_LINEAR)
        self.sun_a = smooth(.985, .955, r + .012 * n + .010 * nf + .010 * pr); dry = smooth(1.0, 1.9, pr + .5 * nf + 1.2 * smooth(.75, .98, r) * smooth(-.4, 1., np.cos(a - .6))); self.sun_a = self.sun_a * (1 - .75 * dry)
        t = np.clip(.5 + .20 * n + .16 * pr + .32 * y, 0, 1)[..., None]; self.sun_c = (rgb('#f0522c') * (1 - t) + rgb('#c81e1c') * t) * (1 - .22 * smooth(.86, .97, r))[..., None] * (1 + .03 * nf[..., None])
        self.tooth = sumi.tooth(PH, PW)
        c = Cel(700, 700); r_ = np.random.default_rng(2)
        self.rays = [(i / 28. * 6.283 + r_.normal(0, .03), r_.uniform(1.16, 1.24), r_.uniform(1.42, 1.95), r_.uniform(10, 24)) for i in range(28)]

    def R(self, f):
        """radius of the sun: breath in, contraction, snap, settle; then the two heartbeats and the jolt of the last stroke"""
        if f < 4: r = 352 + 12 * smooth(0, 4, f)
        elif f < 16: q = (f - 4) / 12.; r = 364 + (50 - 364) * (q * q * (3 - 2 * q)) ** 1.15
        else: r = 66 - 16 * math.exp(-(f - 16) / 2.2) * math.cos((f - 16) * 1.25)
        for t0, a in ((62, .20), (67, .17)):
            if f >= t0: k = f - t0; r *= 1 + a * math.exp(-k / 2.2) * math.cos(k * 1.05)
        if f >= 60: k = f - 60; r *= 1 + .16 * math.exp(-k / 1.4) * math.sin(k * 2.2 + 1.2)
        return r

    def off(self, f):
        k = smooth(3, 36, f); k = 1 - (1 - k) ** 1.35; return D0 * (1 - k)

    # ---- the brush
    def tip(self, f):
        """(x, y, lift 0..1, angle, spread) of the brush on the page"""
        seg = [(m, t) for m, t in zip(self.pg.marks[:7], T_DO + T_N + [T_EX])]
        prev = None
        for m, (a, b) in seg:
            I = m.info
            if a <= f <= b: u = (f - a) / (b - a); i = int(np.searchsorted(I['t'], u)); i = min(i, len(I['x']) - 1); return I['x'][i], I['y'][i], 0., I['w'][i]
            if f < a:
                p1 = np.array([I['x'][0], I['y'][0]])
                if prev is None: p0, t0 = p1 + [520, -980], a - 7
                else: p0, t0 = prev
                u = np.clip((f - t0) / max(.01, a - t0), 0, 1); big = m is self.pg.marks[6]
                if big:
                    hi = np.array([1480, 40.]); v = smooth(0, .55, u); q = p0 + (hi - p0) * v; w_ = smooth(.86, 1., u); q = q + (p1 - q) * w_; lf = (1 + 1.3 * smooth(.1, .7, u)) * (1 - w_ ** 2) if u < 1 else 0.; return q[0], q[1], max(0.02, min(2.2, lf * smooth(0, .2, u))), 60.
                e = u * u * (3 - 2 * u); q = p0 + (p1 - p0) * e; return q[0], q[1], math.sin(math.pi * u) ** .7 * (.35 if prev is not None else 1.0) + (1.4 * (1 - u) if prev is None else 0), 50.
            prev = (np.array([I['x'][-1], I['y'][-1]]), b)
        u = np.clip((f - T_EX[1]) / 5.5, 0, 1.4); return prev[0][0] + 120 * u + 360 * u * u, prev[0][1] - 300 * u - 900 * u * u, u * 1.8, 40.

    def brush(self, cel, sh, x, y, lift, w):
        """a fat sumi brush: wet black belly, pale root hair, bamboo; tilted to the upper right, shadow falls lower left"""
        th = -1.02 - .10 * math.sin(lift * 2); d = np.array([math.cos(th), math.sin(th)]); n = np.array([-d[1], d[0]]); sp = 1 + .22 * (lift < .02) * np.clip(w / 130., 0, 1.3); z = np.array([26., -60.]) * lift
        p = np.array([x, y]) + z; X = lambda u, v: p + d * u + n * v; b = 46 * sp
        hair = spl([X(0, -2), X(34, -b * .62), X(90, -b), X(150, -b * .86), X(186, -30), X(186, 30), X(150, b * .86), X(90, b), X(34, b * .55), X(0, 4)], True, n=8); fer = np.array([X(180, -31), X(236, -33), X(236, 33), X(180, 31)]); pole = np.array([X(232, -24), X(1500, -30), X(1500, 30), X(232, 24)])
        s0 = np.array([-70., 66.]) * (.35 + 1.6 * lift); q = np.array([x, y]) + s0; Y = lambda P: (P - p) @ np.array([[.94, .10], [.02, .84]]) + q
        for P in (hair, fer, pole): sh.fill(Y(np.asarray(P)), '#3a2a22', 1.)
        cel.fill(pole, '#c79a55'); cel.fill(np.array([X(232, 4), X(1500, 5), X(1500, 30), X(232, 24)]), '#8d6230'); cel.fill(np.array([X(232, -19), X(1500, -22), X(1500, -13), X(232, -11)]), '#eccf8f')
        for k in (430, 700, 1010): cel.fill(np.array([X(k, -27), X(k + 12, -29), X(k + 12, 29), X(k, 27)]), '#a8793a'); cel.line([X(k + 13, -28), X(k + 13, 28)], 3, '#4a2f18', taper=(.1, .1))
        kit.outline(cel, pole, 4.2, '#1b130f', seed=1); cel.fill(fer, '#7c1a20'); cel.fill(np.array([X(180, 6), X(236, 8), X(236, 33), X(180, 31)]), '#4e0f16'); kit.outline(cel, fer, 3.6, '#1b130f', seed=2)
        cel.fill(hair, '#15121a'); cel.clip(hair); cel.fill(spl([X(130, -60), X(128, -14), X(140, 10), X(126, 60), X(200, 60), X(200, -60)], True, n=6), '#e9dcc0'); cel.fill(spl([X(150, 6), X(140, 14), X(130, 60), X(200, 60), X(200, 8)], True, n=6), '#b59c80')
        for k in range(9): v = -b * .8 + k * b * .2; cel.line(spl([X(178, v * .6), X(120, v), X(58, v * .72)], n=8), 2.2, '#2c2733' if k % 2 else '#0b090e', taper=(.1, .6), seed=k)
        cel.fill(spl([X(60, -b * .62), X(104, -b * .80), X(124, -b * .66), X(90, -b * .52)], True, n=6), '#8b93a8', .8); cel.unclip(); kit.outline(cel, hair, 4.2, '#0d0b10', seed=3)

    def hat(self, f):
        """pendulum fall, landing on the dot at 67"""
        if f < 50: return None
        u = np.clip((f - 50) / 17., 0, 1); sw = math.cos(u * 3.2 * math.pi) * (1 - u) ** .8
        end = SUN + [6, -50]; p = end + np.array([300 * sw + 120 * (1 - u), -1000 * (1 - u) ** 1.15 + 60 * abs(sw) * (1 - u)]); roll = -.62 * math.sin(u * 3.2 * math.pi + .5) * (1 - u) ** .6 - .16 * u; b = 0.
        if f >= 67: k = f - 67; roll = -.16 + .17 * math.exp(-k / 4.2) * math.sin(k * .95); b = .9 * math.exp(-k / 2.4) * math.cos(k * 1.5); p = end + [0, 5 * math.exp(-k / 2.) * math.cos(k * 1.3)]
        return p, roll, b, u

    def frame(self, f):
        ink, fr, po = sumi.page_state(self.pg, float(f) if f < 61 else f, 16.)
        img = sumi.lay(self.paper, ink, None, po, col='#100c10', thin='#4a3628') + sumi.wet_light(ink, fr, po, .85, f * .02)
        # the sun
        r = self.R(f); sx, sy = (1., 1.)
        if 14 <= f < 22: q = math.exp(-(f - 14) / 2.5) * math.sin((f - 14) * 1.5); sx, sy = 1 + .10 * q, 1 - .10 * q
        wash = np.exp(-((np.hypot(*(np.meshgrid(np.arange(PW, dtype=f32) - SUN[0], np.arange(PH, dtype=f32) - SUN[1]))) / (r * 2.1 + 40)) ** 2)) * (.34 * (1 - smooth(2, 18, f)) + .05); img = img * (1 - wash[..., None]) + rgb('#f9a05c') * wash[..., None] * 1.0
        k = r * 2 / 900 / .97; M = np.float32([[k * sx, 0, SUN[0] - 450 * k * sx], [0, k * sy, SUN[1] - 450 * k * sy]]); fl = cv2.INTER_AREA if k < .5 else cv2.INTER_LINEAR; sa, sc = self.sun_a, self.sun_c
        if k < .5: sa, sc = cv2.GaussianBlur(sa, (0, 0), .4 / k), cv2.GaussianBlur(sc, (0, 0), .4 / k)
        a = cv2.warpAffine(sa.astype(f32), M, (PW, PH), flags=cv2.INTER_LINEAR); c = cv2.warpAffine(sc.astype(f32), M, (PW, PH), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); a = a * (1 - .18 * self.tooth * smooth(120, 60, r)); img = img * (1 - a[..., None]) + c * a[..., None]
        if f < 14:
            ry = Cel(PW, PH, ss=1); q = smooth(1, 13, f)
            for i, (th, r0, r1, w) in enumerate(self.rays):
                d = np.array([math.cos(th + .02 * f), math.sin(th + .02 * f)]); a0 = r * (r0 - .06 * q); a1 = a0 + (352 * (r1 - r0)) * (1 - q) ** 1.2 * (1 + .08 * math.sin(f * .9 + i)); 
                if a1 - a0 > 6: ry.line(np.array([SUN + d * a0, SUN + d * (a0 + a1) * .5, SUN + d * a1]), w * (1 - .6 * q), '#d8391f', taper=(.1, .75), press=.4, seed=i, alpha=.85)
            l = ry.layer(PH, PW); l = l * smooth(.12, .5, self.tooth + .28)[..., None]; img = img * (1 - l[..., 3:4]) + l[..., :3]
        o = self.off(f); j = kit.shake(f, 11, 4, 1.8, 60) if f >= 60 else np.zeros(2)
        h = self.hat(f); cel = Cel(); sh = Cel(ss=1); t = o + j
        M = np.float32([[1, 0, t[0]], [0, 1, t[1]]]); img = cv2.warpAffine(img.astype(f32), M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        # heart: the picture breathes
        z = 1.
        for t0, am in ((62, .012), (67, .010)):
            if f >= t0: z += am * math.exp(-(f - t0) / 2.6) * math.cos((f - t0) * .8)
        if f >= 60: z += .010 * math.exp(-(f - 60) / 1.5)
        bx, by, lf, w = self.tip(f)
        if 7 < f < 67:
            self.brush(cel, sh, bx + t[0], by + t[1], lf, w); s = sh.layer()[..., 3]; s = cv2.GaussianBlur(s, (0, 0), 5 + 14 * min(lf, 1.6)); img = img * (1 - s[..., None] * .30 * (1 - .35 * min(lf, 1.4))) + rgb('#3a2a22') * (s * .30 * (1 - .35 * min(lf, 1.4)))[..., None] * .5; l = cel.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if h is not None:
            from op import hat as HT
            p, roll, b, u = h; q = p + t
            if q[1] > -200:
                l = HT.hat(q[0], q[1], 250, yaw=.35 + (1 - u) * 3.6, pitch=-.16 + .34 * (1 - u) * math.sin(u * 9), roll=roll, light='day', wear=.7, bounce=b, f=f, wind=.5); s = kit.shifted(l[..., 3], -22 - 90 * (1 - u), 26 + 150 * (1 - u)); s = cv2.GaussianBlur(s, (0, 0), 5 + 20 * (1 - u)); img = img * (1 - s[..., None] * .26); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if f >= 60 and f < 66:
            sp = Cel(); r_ = np.random.default_rng(6)
            for i in range(40):
                a = r_.uniform(0, 6.283); v = r_.uniform(60, 420); k_ = (f - 60 + .6) / 24.; p = np.array([1478, 230.]) + t + np.array([math.cos(a), math.sin(a) * .8]) * v * k_ * 5 + [0, 900 * k_ * k_]; s = r_.uniform(2, 10) * (1 - (f - 60) / 6.5); sp.fill(np.c_[p[0] + np.cos(np.linspace(0, 6.28, 12)) * s, p[1] + np.sin(np.linspace(0, 6.28, 12)) * s * 1.2], '#100c10')
            l = sp.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if abs(z - 1) > 1e-4: c_ = SUN + t; img = cv2.warpAffine(img, np.float32([[z, 0, c_[0] * (1 - z)], [0, z, c_[1] * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return img.astype(f32)
