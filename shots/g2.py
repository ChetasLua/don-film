"""g2: under the brim.  Cream sky, a huge red sun, the lion ship at anchor (its reflection is the little sheep ship), ten backs on the headland,
Luffy bareheaded in the middle.  f30 wind; f45-75 one after another the left arms rise: X."""
import numpy as np, cv2, math, os
from op import W, H, ships, sea
from op.cam import Camera3D
from op.sea import Sea, DAWN
from op.paint import *
from op.ink import Cel, spl, resample
from op.kit import outline, rim, para, hatch
from op.comp import blur, ease, lerp_keys, screen, radial, over
from op.fx_core import put, noise, h1
from op.fx_air import petals
from op.seakit import gull_top
A_ = np.asarray
SUN, SR, HOR = A_([960., 486.]), 268., 606.
INK = '#1c1417'
PAL = dict(DAWN, deep='#a4907c', body='#c2ae95', mid='#d9c7aa', light='#ecdcc0', sky='#f8ecd2', crest='#dcb89a', foam='#fbf3e2', foamsh='#cdb9a0', far='#eadbc2')
from op.backs import CAST, DRAW, Back, gust, INK


class Shot:
    id = 'g2'; n = 90; finish = dict(bloom=.22, vig=.20)
    EVENTS = [(0, 'sea', dict(note='calm shore, soft', size=.5)), (8, 'gull', dict(size=.3)), (29, 'wind', dict(note='gust arrives left to right')), (33, 'cloth', dict(note='coats crack'))] + [(q[4], 'arm', dict(note=q[0] + ' raises the left arm (cloth snap on the hit)', size=1. if q[0] == 'luffy' else .5)) for q in sorted(CAST, key=lambda q: q[4])] + [(78, 'hush', dict(note='all ten up')), (72, 'secret', dict(note='far headland left: a girl and a duck raise arm and wing', size=.2))]

    def setup(self):
        self.S = Sea(31, hs=.30, length=13, chop=.6, wind=(.4, -1)); p = 'cache/g2_sky3.npy'
        if not os.path.exists(p):
            x, y = grid(); n = fbm(H, W, (340, 120, 40), (1, .5, .3), 4, 2.4); d = np.hypot(x - SUN[0], (y - SUN[1]) * 1.1); g = ramp(np.clip(y / HOR, 0, 1), [(0, '#e9dcc2'), (.5, '#f1e5cc'), (1, '#f3dfc0')]); g = g + (np.exp(-(d / 520) ** 2) * .20)[..., None] * (rgb('#f6b98c') - g) * 1.6; g *= 1 + .018 * n[..., None]
            # flat brushed cloud bars
            g = repaint(g, (26, 12, 6), seed=5, density=.9, jit=(.005, .012, .012)); c = Cel(); r = np.random.default_rng(19)
            for i, (cx, cy, L, th) in enumerate(((150, 452, 330, 17), (470, 388, 210, 10), (60, 330, 190, 11), (1780, 420, 330, 18), (1460, 352, 190, 9), (1890, 318, 190, 12), (420, 520, 300, 8), (1560, 506, 330, 10), (250, 566, 260, 6), (1760, 570, 260, 7))):
                m = np.linspace(-1, 1, 90); bump = np.zeros(90)
                for _ in range(9): q = r.uniform(-.8, .8); bump = np.maximum(bump, r.uniform(.4, 1.3) * np.sqrt(np.clip(1 - ((m - q) / r.uniform(.10, .30)) ** 2, 0, 1)) * (1 - abs(q) * .7))
                env = np.clip(1 - np.abs(m), 0, 1) ** .5; top = cy - th * (.35 * env + bump); bot = cy + th * .45 * env * (1 + .3 * np.sin(m * 14 + i)); P = np.vstack([np.c_[cx + m * L, top], np.c_[cx + m[::-1] * L, bot[::-1]]]); c.fill(P, '#cdbba2'); c.clip(P); c.fill(np.vstack([np.c_[cx + m * L, cy - th * (.35 * env + bump) * .35], [[cx + L, cy + 60], [cx - L, cy + 60]]]), '#b9a68f', .8); c.unclip()
            l = c.rgba(); g = g * (1 - l[..., 3:4]) + l[..., :3]
            # the sun: flat vermilion with a dry-brush breath, hard edge
            a = np.clip((SR - np.hypot(x - SUN[0], y - SUN[1]) + .45 * fbm(H, W, (60, 22), (1, .5), 2)) / 1.4 + .5, 0, 1)[..., None]; sc = ramp(np.clip((y - SUN[1] + SR) / (2 * SR), 0, 1), [(0, '#e8573a'), (.5, '#df4329'), (1, '#e9683f')]) * (1 + .030 * fbm(H, W, (160, 40, 9), (1, .6, .5), 12, 3.)[..., None]); g = g * (1 - a) + sc * a
            c = Cel(ss=1)                                                                    # thin bars crossing the disc
            for i, (yy, x0, x1, th) in enumerate(((548, 640, 1010, 7), (572, 930, 1330, 9), (520, 1060, 1300, 4), (588, 560, 900, 6))): m = np.linspace(0, 1, 20); c.fill(np.vstack([np.c_[x0 + (x1 - x0) * m, yy - th * np.sin(math.pi * m) * (1 + .4 * np.sin(m * 19 + i))], np.c_[x0 + (x1 - x0) * m[::-1], yy + th * .4 * np.sin(math.pi * m[::-1])]]), '#e7b592', .85)
            l = c.rgba(); g = g * (1 - l[..., 3:4]) + l[..., :3]; np.save(p, g.astype(np.float16))
        self.sky = np.load(p).astype(f32)
        # far hills
        c = Cel(); xs = np.linspace(-20, W + 20, 260); r = np.random.default_rng(3)
        for (a, b, hh, col, sd) in ((-20, 470, 92, '#c3b09a', 1), (1420, 1960, 120, '#bba892', 2), (-20, 330, 140, '#9f8d7c', 3), (1620, 1960, 168, '#93816f', 4)):
            q = np.linspace(a, b, 120); u = (q - a) / (b - a); e = np.sin(math.pi * (u if sd % 2 == 0 else 1 - u) * .5 + (0 if sd % 2 == 0 else 0)) ** 1.5; nz = fbm(1, 120, (30, 11, 4), (1, .6, .4), sd)[0]; c.fill(np.vstack([np.c_[q, HOR + 3 - hh * e * (1 + .22 * nz)], [[b, HOR + 6], [a, HOR + 6]]]), col)
        self.hill = c.layer()
        self.rock = self._rock()

    def _rock(self):
        x = np.linspace(-40, W + 40, 400); n = fbm(1, 400, (60, 20, 7, 3), (1, .6, .4, .25), 8)[0]; top = 974 - 14 * np.exp(-((x - 960) / 300) ** 2) + 8 * n - 16 * smooth(500, 900, np.abs(x - 960)); c = Cel(); P = np.vstack([np.c_[x, top], [[W + 40, H + 40], [-40, H + 40]]]); c.fill(P, '#2c2324'); r = np.random.default_rng(4); c.clip(P)
        for i in range(120): x0 = r.uniform(-40, W); y0 = np.interp(x0, x, top) + abs(r.normal(0, 1)) * 60 + 4; L = r.uniform(40, 190); c.fill(spl([(x0, y0), (x0 + L * .4, y0 - r.uniform(2, 12)), (x0 + L, y0 + r.uniform(0, 14)), (x0 + L * .5, y0 + r.uniform(8, 24))], True, n=5), ['#3a2d2c', '#453432', '#211a1c', '#4f3a34'][i % 4], .75)
        for i in range(260): x0 = r.uniform(-40, W); y0 = np.interp(x0, x, top) + r.uniform(4, 110); c.line([(x0, y0), (x0 + r.uniform(8, 34), y0 + r.uniform(8, 30))], r.uniform(.8, 1.8), '#120d10', taper=(.1, .6), seed=i, alpha=.6)
        c.unclip(); outline(c, np.c_[x, top], 2.6, '#150f12', closed=False, seed=3); l = c.layer(); self.top = (x, top); return rim(l, 0, 3.5, '#e07a4c', .55, 1.0)

    def far(self, f):
        """thirty pixels of secret: a long-haired girl and a big duck on the far headland, left arm and wing up after the others"""
        c = Cel(170, 160, 0, 380, ss=4); tops = np.argmax(self.hill[:, 70:112, 3] > .5, 0); xp = 70 + int(tops.argmin()); gy = float(tops.min()) + 7.; u = float(lerp_keys(f, [(60, 0), (66, -.1), (70, 1.12), (74, .96), (78, 1.)])); w = gust(f, -.5); P = lambda q, o=(float(xp) - 7, 0.): A_(q, float) * [1, -1] + [o[0], gy + o[1]]; k = lambda q, cc, cor=(), n=4: c.fill(spl(P(q), True, corners=cor, n=n), cc)
        # the girl: 30 px with the arm down
        k([(-3.4, 0), (-4.2, 9), (-3.0, 17), (-3.6, 21), (3.2, 21), (2.8, 16), (4.4, 8), (5.2 + 2 * w, 0)], '#6f5a50'); k([(-3.6, 12), (-3.2, 21), (3.2, 21), (3.0, 12)], '#4d5c66'); e = np.linspace(0, 6.283, 14)[:-1]; k([(math.cos(a) * 3.1, 25.4 + math.sin(a) * 3.4) for a in e], '#6e4d3d'); k([(-3.3, 24), (-2.6, 29.2), (1.6, 29.6), (3.6, 26), (4.6 + 2 * w, 20), (9 + 7 * w, 14), (5 + 4 * w, 13), (2.6, 18), (1.2, 22)], '#3d5568', (5,))
        a = math.radians(-100 - 166 * u); S = A_([-3.2, 20.]); E = S + A_([math.cos(a), math.sin(a)]) * 9.5; c.line(P([S, (S + E) / 2, E]), 2.5, '#6e4d3d', taper=(0, .1), tmin=.7); c.line(P([S + (E - S) * .62, S + (E - S) * .85]), 2.8, '#fff4df', taper=(0, 0))
        # the duck
        k([(16, 0), (13, 9), (13.5, 16), (17, 20), (23, 19), (27, 13), (30 + w, 14.5), (27, 6), (24, 0)], '#8a6432', (6,)); k([(math.cos(q) * 5.0 + 17.5, 23 + math.sin(q) * 4.8) for q in e], '#8a6432'); k([(13, 25), (13.6, 29), (18, 30.4), (22, 28), (22.4, 24.6)], '#3d5568'); k([(13.4, 22.6), (8.4, 21.6), (9.0, 19.6), (13.6, 20.2)], '#b9802f', (1, 2)); wg = math.radians(-70 - 150 * u); W0 = A_([16., 14.]); T = W0 + A_([math.cos(wg), math.sin(wg)]) * 12; n = A_([-math.sin(wg), math.cos(wg)]); k([W0 + n * 2.4, T + n * 1.6, T + [0, 1.4], T - n * 1.8, W0 - n * 2.6], '#76552a', (2,))
        l = c.layer(); return rim(para(l, (0, gy - 34), (0, gy), [(0, '#ffffff'), (1, '#d8c8bc')], 'mul', .8), -.9, .4, '#f0764a', .8, .3)

    def grass(self, f):
        c = Cel(); r = np.random.default_rng(9); g = (f // 2) * 2
        for i in range(340):
            x0 = r.uniform(-20, W + 20); y0 = np.interp(x0, *self.top) + r.uniform(-2, 70) ** 1.; L = r.uniform(16, 50) * (.7 + .6 * (y0 - 940) / 130) * (1 - .45 * math.exp(-((x0 - 952) / 210.) ** 2)); w = gust(g, (x0 - 200) / 420.) * (1 + .3 * math.sin(g * .7 + x0 * .03)); ln = r.normal(0, .25); tp = (x0 + L * (ln + .95 * w), y0 - L * (1 - .40 * w * w)); md = (x0 + L * (ln * .3 + .20 * w), y0 - L * .62); P = spl([(x0, y0 + 4), md, tp], n=6); c.line(P, r.uniform(2.0, 4.4), ['#241c1c', '#33282a', '#463430', '#1a1416'][i % 4], taper=(.02, .85), seed=i, press=.1, tremor=.1)
        return rim(c.layer(), 0, 1.6, '#e3875a', .5, .5)

    def frame(self, f):
        t = f / 24.; ez = ease(f / 110.); zoom = 1 + .035 * ez; img = self.sky.copy(); x, y = grid()
        G = Cel()
        for i in range(6): q = A_([1260 + 46 * i + 30 * t * (1 + .1 * i), 300 - 16 * i + 14 * math.sin(i * 2.1) - 10 * t]); fl = math.sin(f * .55 + i * 1.3); s = 15 - i; G.line(spl([(q[0] - s, q[1] - 5 * fl), (q[0] - s * .45, q[1] - 5 - 3 * fl), q, (q[0] + s * .45, q[1] - 5 - 3 * fl), (q[0] + s, q[1] - 5 * fl)], n=5), 2.2, '#3a2c2a', taper=(.4, .4), seed=i)
        img = put(img, G.layer()); img = put(img, self.hill); img = put(img, self.far(f))
        cam = Camera3D((0, 0, 5.4), (0, 100, 5.4), 1500, cx=960, cy=HOR); shipx = -64.; D = 300.
        # reflection plate: mirrored sky + mirrored MERRY (never the lion)
        mir = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, -1, 2 * HOR]]), (W, H), borderMode=cv2.BORDER_REFLECT); mir4 = np.dstack([mir, np.ones((H, W), f32)])
        bz = .18 * math.sin(t * 1.2); me = ships.merry(t * .6, wind=.45, sails=True, flag=True); ships.sheep(me, (11.8, 0, 7.6), 1.75)
        ps = ships.pose(shipx + 3., D - 6, bz, -.62, .006 * math.sin(t * 1.1), .012 * math.sin(t * .9), 2.2); dm = ships.draw([(me, ships.mirror(ps))], cam, (0, 1, -.35), ss=2, ink='edge', lw=1.2, shadow=False, dark=.16, sil="#5a342c", amb=(.84, .64, .54), edge_col='#3a2320', texture=0)
        mir4 = dm.layer + mir4 * (1 - dm.layer[..., 3:4])
        w = self.S.render(cam, 5 + t, PAL, sun=(0, 1, .22), step=5, mirror=mir4, mirror_amt=.95, mirror_k=.15, foam_amt=0, crest=0, haze=.25, tone=.75); wt = w.rgb
        img = w.over(img); ln = np.exp(-((y - HOR) / 3.) ** 2)[..., None]; img = img * (1 - .18 * ln)
        # glitter strokes along the sun road
        gl = Cel(); r = np.random.default_rng(6)
        for i in range(150): q = r.random(); yy = HOR + 6 + 330 * q ** 1.6; xx = SUN[0] + r.normal(0, 1) * (70 + 150 * q); L = (8 + 60 * q) * r.uniform(.5, 1.2) * max(0., math.sin(t * r.uniform(2, 4) + i)); gl.line([(xx - L, yy), (xx, yy + 1), (xx + L, yy)], 1.2 + 3 * q, '#f9dcae' if i % 3 else '#f2a67a', taper=(.5, .5), seed=i, alpha=.75) if L > 3 else None
        img = put(img, gl.layer())
        # the lion ship at anchor, sails furled
        su = ships.sunny(t * .8, wind=.4, sails=False)
        for (xx, zz, hw) in ((4.2, 25.2, 8.8), (-8.4, 25.0, 6.8)):
            ships.yard(su, xx, zz + .4, hw * 1.08)
            for yy in np.linspace(-hw * .86, hw * .86, 7): su.ball((xx + .5, yy, zz - .25), (.50, hw * .16, .62), '#e9dcc2', f'furl{xx}', n=(7, 10), grp=f'furl{xx}', lw=1.4)
        ds = ships.draw([(su, ships.pose(shipx, D, bz, -.42, .006 * math.sin(t * 1.1), .012 * math.sin(t * .9), 1.55))], cam, (.55, .9, .4), lw=.62, shadow=False, dark=.46, sil='#4a2a26', amb=(.86, .72, .62), rim=('#ffb27a', 1.6, .8), edge_col='#2a1715', texture=0)
        img = sea.under(img, ds.layer, ds.z, w)
        for n, cx, gy, k, hit, rest, tone in CAST:
            s = Back(cx, gy, k, f, hit, rest, len(n) * 7 + int(cx), tone); DRAW[n](s); c = s.done(3.2 if n not in ('luffy',) else 3.6); l = c.rgba(); dx, dy = SUN[0] - cx, SUN[1] + 120 - (gy - 200); m = math.hypot(dx, dy) + 1e-6; dy = min(dy, -.35 * m); m = math.hypot(dx, dy); ox, oy = c.ox, c.oy
            l = para(l, (0, gy - 520 * k - oy), (0, gy - oy), [(0, '#fff0e2'), (.5, '#e6cfc6'), (1, '#a88f90')], 'mul', .9); nz = fbm(l.shape[0], l.shape[1], (34, 12, 5), (1, .6, .35), int(cx)); wh = smooth(-1.1, .35, nz + .5); l[..., :3] *= (1 + .055 * nz)[..., None]; l = rim(l, -dx / m * 4.6, -dy / m * 4.6, "#e9582f", .92, .8, where=wh); l = rim(l, -dx / m * 1.7, -dy / m * 1.7, "#ffc890", .85, .5, where=wh)
            y0, y1, x0, x1 = max(0, oy), min(H, oy + l.shape[0]), max(0, ox), min(W, ox + l.shape[1]); p = l[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; img[y0:y1, x0:x1] = img[y0:y1, x0:x1] * (1 - p[..., 3:4]) + p[..., :3]
        img = put(img, self.rock)
        img = put(img, self.grass(f))
        if f >= 24: img = put(img, petals(f - 24, 34, 5, pal=dict(lit='#d0603c', mid='#a5482f', dark='#6b3a2a', line='#4a2418', ink='#4a2418'), wind=(560. * min(1, (f - 24) / 8.), -20.), size=(5, 15), box=(-400, 420, W, 1050), near=0, lw=1.0, op=min(1., (f - 24) / 6.)))
        img = cv2.warpAffine(img, np.float32([[zoom, 0, SUN[0] * (1 - zoom)], [0, zoom, SUN[1] * (1 - zoom)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        img = img * (1 + .012 * fbm(H, W, (3.2, 1.6), (1, .7), 5)[..., None])              # paper tooth
        try:
            from op.hat import under_brim
            l = under_brim(f, float(lerp_keys(f, [(0, 1.05), (4, .86), (12, .50), (18, .46), (30, .46), (34, .40), (38, .48), (90, .44)])), stars=float(lerp_keys(f, [(0, 1.), (14, .55), (90, .5)])), sun=(960, 300)); l = blur(l, 3.2); img = img * (1 - l[..., 3:4]) + l[..., :3]
        except Exception as e: print('g2 brim', repr(e)[:100])
        return img
