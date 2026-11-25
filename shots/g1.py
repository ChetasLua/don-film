"""g1: POV on the deck.  Eyelids open (two blinks), a ring of ten faces, mast, sail, black flag, the sun behind the crow's nest."""
import numpy as np, cv2, math, os, importlib
from op import W, H, ships
from op.cam import Camera3D
from op.paint import *
from op.ink import Cel, spl
from op.kit import outline, rim, para, flare
from op.comp import blur, ease, lerp_keys, screen, radial, over
from op.seakit import puff, gull_top
from op.fx_core import put
from op.fx_air import motes
A_ = np.asarray
ORDER = ['robin', 'chopper', 'jinbe', 'brook', 'sanji', 'nami', 'usopp', 'franky', 'zoro', 'luffy']
SUN = (960., 486.)


class Shot:
    id = 'g1'; n = 60; finish = dict(bloom=.36, vig=.16)
    EVENTS = [(0, 'breath', dict(note='a breath in: eyes open, ears full of sea')), (3, 'blink', {}), (7, 'blink', dict(note='second blink')), (10, 'sea', dict(note='world comes in: sail, rope creak, gulls')), (12, 'rope', {}), (20, 'laugh', dict(size=1., note='ten laughs break at once')), (36, 'cloth', dict(note='he takes off the hat')), (44, 'whoosh', dict(size=.4, note='hat swings over the lens')), (56, 'straw', dict(note='straw settles on a forehead'))]

    def setup(self):
        p = 'cache/g1_sky2.npy'
        if not os.path.exists(p):
            x, y = grid(); d = np.hypot((x - SUN[0]), (y - SUN[1])); n = fbm(H, W, (260, 90, 34), (1, .6, .35), 5); g = ramp(np.clip(d / 1150 + .03 * n, 0, 1), [(0, '#cfeeff'), (.10, '#8ed0f8'), (.30, '#469fe8'), (.7, '#2272c8'), (1, '#1652a4')])
            th = np.arctan2(y - SUN[1], x - SUN[0]); ci = smooth(.2, 1.5, fbm(H, W, (120, 50, 20), (1, .6, .4), 11, 2.6) + .5 * np.sin(th * 3 + 1)); g = g + (1 - g) * (ci * .42)[..., None]
            for i, (s, X, Y) in enumerate(((1.0, 1180, 560), (.8, 1250, 120), (.62, 560, 620), (.7, 640, -60))): T = puff(900, 620, 60 + i, n=70, tones_=('#ffffff', '#f4f9fd', '#d2e6f6', '#aecdea'), light=((SUN[0] - X - 300) / 600., (SUN[1] - Y - 200) / 500.)); l = cv2.warpAffine(T, np.float32([[s, 0, X], [0, s * .9, Y]]), (W, H)); g = put(g, l, .92)
            np.save(p, repaint(g, (30, 14, 7), seed=3, density=.95, jit=(.004, .012, .012)).astype(np.float16))
        self.sky = np.load(p).astype(f32); self.mods = {}
        for n in ORDER:
            try: self.mods[n] = importlib.import_module('chars.' + n).g1
            except Exception as e: print('g1: no layer from', n, repr(e)[:120])

    def deck(self, f):
        t = f / 24.; sw = .012 * math.sin(t * 1.6); cam = Camera3D((4.2 - 2.15, -1.25, 4.75), (4.2 - 2.15 + sw * 10, -1.25 + .006 * math.sin(t * 1.1) * 10, 14.75), 930, cx=1010, cy=520, roll=sw * 1.2)
        m = ships.sunny(t, wind=.85, flag=False); m.parts = [p for p in m.parts if p.name not in ('gaff', 'bench')]
        m.flag((4.2, 0, 37.8), 0, 5.6, 3.6, t, dirn=(-.55, -.85, 0), amp=.13)
        L = A_([.12, .10, 1.]); d = ships.draw([(m, ships.pose())], cam, L, lw=1.9, shadow=False, shade_amt=.95, texture=.02, rim=None); lay = d.layer
        # sun through the cloth: warm and bright near the disc, cool at the edges
        x, y = grid(); r = np.hypot(x - SUN[0], y - SUN[1]); g = ramp(np.clip(r / 950, 0, 1), [(0, '#fff6dc'), (.22, '#f6e2bf'), (.55, '#d9cdd0'), (1, '#8e9cc8')]); lay[..., :3] *= g
        return rim(lay, -3.5 * 1, -2.0, '#fff6d6', .8, .7), d

    def frame(self, f):
        t = f / 24.; img = self.sky.copy(); dx, dy = 6 * math.sin(t * 1.6), 4 * math.sin(t * 1.1); img = cv2.warpAffine(img, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REFLECT)
        x, y = grid(); r = np.hypot(x - SUN[0], y - SUN[1]); img = img + (np.exp(-(r / 210.) ** 1.5) * .50 + np.clip((112 - r) / 5, 0, 1) * 1.0)[..., None] * rgb('#fff3cf')
        G = Cel()
        for i in range(3): gull_top(G, 1130 + 90 * i + 40 * t * (i + 1), 330 - 60 * i + 30 * t, 22 - 4 * i, .9 + .4 * i, math.sin(t * 8 + i * 2), lw=1.)
        img = put(img, G.layer()); lay, d = self.deck(f); img = put(img, lay)
        # halo spilling round the nest
        img = img + (np.exp(-(r / 120.) ** 1.6) * .26)[..., None] * rgb('#ffe9b0')
        th = np.arctan2(y - SUN[1], x - SUN[0]); ray = np.clip(np.sin(th * 7 + .6 * np.sin(th * 3) + .05 * f) * .6 + np.sin(th * 13 - .04 * f) * .4, 0, 1) ** 2; img = img + (ray * np.exp(-r / 420.) * .30 * smooth(60, 150, r))[..., None] * rgb('#fff2cc')
        img = img * (1 - .20 * smooth(420, 1000, r)[..., None] * A_([1, .8, .5], f32))
        for n in ORDER:
            fn = self.mods.get(n)
            if fn is None: continue
            try: l = fn(f)
            except Exception as e:
                print('g1', n, repr(e)[:140]); self.mods[n] = None; continue
            if n != 'luffy':                                                             # contact shade where one head overlaps another
                s = blur(l[..., 3], 14); s = cv2.warpAffine(s, np.float32([[1, 0, 0], [0, 1, 8]]), (W, H))[..., None]; img = img * (1 - .30 * s * A_([.8, .7, .5], f32))
            img = img * (1 - l[..., 3:4]) + l[..., :3]
        # light falling into the ring
        k = 1 - smooth(36, 54, f); img = img + (np.exp(-(r / 520.) ** 1.2) * .08 * k)[..., None] * rgb('#fff0c8') + (ray * np.exp(-r / 620.) * .10 * k)[..., None] * rgb('#fff2cc')
        mo = motes(f, 70, 4, amt=.5 * max(.2, k)); img = img + mo if isinstance(mo, np.ndarray) and mo.shape[2] == 3 else put(img, mo)
        return self.lids(img, f)

    def lids(self, img, f):
        if f >= 13: return img
        o = lerp_keys(f, [(0, .03), (1, .10), (2, .24), (3, .26), (4, .04), (5, .10), (6, .46), (7, .50), (8, .12), (9, .62), (10, .88), (11, 1.05), (12, 1.3)]); s = lerp_keys(f, [(0, 22), (3, 14), (7, 7), (10, 2.5), (12, 0.)])
        if s > .4: img = blur(img, float(s))
        img = img * (1 + .55 * (1 - ease(f / 12.)))                                       # the eye has not closed down yet
        xs = np.linspace(-80, W + 80, 40); u = (xs - 960) / 1040.; arc = 1 - u * u; tp = 500 - (560 * o) * (.35 + .65 * np.clip(arc, 0, 1) ** .8) - 30 * (u > 0) * u; bt = 560 + (470 * o) * (.35 + .65 * np.clip(arc, 0, 1) ** .8)
        c = Cel(); T = np.vstack([np.c_[xs, tp], [[W + 80, -200], [-80, -200]]]); B = np.vstack([np.c_[xs, bt], [[W + 80, H + 200], [-80, H + 200]]]); c.fill(T, '#140708'); c.fill(B, '#140708'); r = np.random.default_rng(3)
        for i in range(46):
            q = (i + r.uniform(-.3, .3)) / 45.; x0 = -40 + (W + 80) * q; y0 = np.interp(x0, xs, tp); ln = r.uniform(70, 150) * (.6 + .6 * math.sin(math.pi * q)); lean = (q - .5) * 150; c.line(spl([(x0, y0 - 10), (x0 + lean * .4, y0 + ln * .35), (x0 + lean, y0 + ln * .55)], n=8), r.uniform(6, 12), '#120607', taper=(.05, .75), seed=i)
            if i % 2: y1 = np.interp(x0, xs, bt); c.line(spl([(x0, y1 + 8), (x0 + lean * .3, y1 - ln * .18), (x0 + lean * .6, y1 - ln * .26)], n=8), r.uniform(4, 7), '#120607', taper=(.05, .75), seed=i)
        l = blur(c.layer(), 7.); e = blur(l[..., 3], 46); gl = np.clip(e * (1 - l[..., 3]) * 1.6, 0, 1)[..., None]; img = img * (1 - gl * .55) + rgb('#7a2418') * gl * .45
        l[..., :3] += (np.clip(blur(1 - l[..., 3], 40), 0, 1) * l[..., 3])[..., None] * rgb('#9a3420') * .55
        return img * (1 - l[..., 3:4]) + l[..., :3]
