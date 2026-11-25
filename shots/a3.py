"""a3 (30): the blades.  Six ink drawings, a held breath, ドン!!, gulls, and the paper eats the picture from the cut."""
import numpy as np, cv2, math
from op import W, H, don, kit
from op.paint import rgb, f32, smooth, grid, fbm
from op.ink import Cel, spl
from op import sumi
from op.fx_core import h1, R
from shots import dragonkit as DK

INK, PAPER, RED, IRON, GOLD = '#0b0a0d', '#f7f4ec', '#b3121f', '#1c222d', '#c9984a'


def _wing(cel, X, S, E, Wr, T, back, fill, shade, ink, lw, s, seed, k0=1.):
    """a wing built of feathers: secondaries, black-tipped primaries, coverts over their roots"""
    bone = spl([S, E, Wr, T], n=8); n = len(bone); do = np.asarray(T, float) - Wr; do /= np.hypot(*do); di = np.asarray(back, float); di /= np.hypot(*di); N = 13 if s > 34 else 7; tips = []
    for k in range(N):
        u = k / (N - 1.); r = bone[int(u * (n - 1) * .97)]; d = di * (1 - u ** 1.7) + (do * .95 + di * .22) * u ** 1.7; d /= np.hypot(*d); m = np.array([-d[1], d[0]]); L = k0 * (.24 + .12 * u + .13 * max(0, u - .55)); w = k0 * (.040 + .012 * (1 - u)) * 13. / N
        P = X(spl([r - d * .02, r + d * L * .45 + m * w, r + d * L * .88 + m * w * .55, r + d * L, r + d * L * .86 - m * w * .6, r + d * L * .45 - m * w], True, corners=(3,), n=5)); tips.append(r + d * L)
        cel.fill(P, fill if (k + seed) % 3 else shade)
        if u > .60: cel.clip(P); cel.fill(X([r + d * L * (.50 - .25 * (u - .6)) + m * .2, r + d * L * 1.1 + m * .2, r + d * L * 1.1 - m * .2, r + d * L * (.50 - .25 * (u - .6)) - m * .2]), ink); cel.unclip()
        if s > 26: kit.outline(cel, P, lw * .62, ink, seed=seed * 7 + k)
        if s > 70: cel.line(X([r + d * L * .2, r + d * L * .8]), lw * .45, ink, taper=(.1, .6), seed=k)
    cv = np.vstack([bone - np.array([0, .028]), np.array([bone[int(u * (n - 1))] + di * .13 * k0 * (1 - .45 * u) for u in np.linspace(1, 0, 10)])]); C = X(cv); cel.fill(C, fill); cel.clip(C); cel.fill(C + np.array(X([(0, .0)]) - X([(0, -.065)]))[0], shade, .85); cel.unclip()
    if s > 44:
        for q in range(2):
            for j in range(9): u = (j + .5 + .5 * q) / 9.6; c0 = bone[int(u * (n - 1))] + di * (.045 + .05 * q) * k0 * (1 - .4 * u); cel.line(X(spl([c0 + [-.026, -.006], c0 + [0, .020], c0 + [.026, -.006]], n=5)), lw * .48, ink, taper=(.2, .3), seed=j)
    kit.outline(cel, C, lw, ink, seed=seed)


def gull(cel, x, y, s, ph, rot=0., fill='#f7f4ec', shade='#aeb4c4', ink=INK, lw=2.2, flip=1, seed=0):
    """an inked gull in three-quarter flight: feathered wings, toned belly, fanned tail.  ph 0..1 wing beat; s = half span px"""
    up = math.sin(ph * 6.283); c, n = math.cos(rot), math.sin(rot); X = lambda P: np.atleast_2d(np.asarray(P, float)) * [s * flip, s] @ np.array([[c, n], [-n, c]]) + [x, y]
    tip = -.60 * up - .12; el = -.36 * up - .18; b = .10 * up; lw = max(.9, lw)
    _wing(cel, X, (.08, -.03 + b), (.24, el * .92 + b), (.48, (el + tip) * .60 + b), (.84, tip * .9 + b), (.05, 1.), shade, '#8990a6', ink, lw * .85, s * .8, seed + 1, .80)
    for k in range(4): a = .10 + .16 * (k - 1.5); d = np.array([math.cos(a + .25), math.sin(a + .25)]); P = X(spl([(.20, .03 + b), np.array([.20, .03 + b]) + d * .14 + [-d[1] * .03, d[0] * .03], np.array([.20, .03 + b]) + d * .27, np.array([.20, .03 + b]) + d * .14 - [-d[1] * .03, d[0] * .03]], True, corners=(2,), n=5)); cel.fill(P, fill); kit.outline(cel, P, lw * .7, ink, seed=k)
    body = spl([(-.33, .01 + b), (-.27, -.065 + b), (-.17, -.075 + b), (.0, -.10 + b), (.20, -.02 + b), (.28, .06 + b), (.10, .13 + b), (-.08, .14 + b), (-.22, .09 + b), (-.30, .06 + b)], True, n=7); B = X(body); cel.fill(B, fill); cel.clip(B); cel.fill(X(spl([(-.30, .06 + b), (-.16, .03 + b), (.04, .06 + b), (.30, .04 + b), (.3, .3 + b), (-.3, .3 + b)], True, n=6)), shade); cel.fill(X(spl([(-.12, .10 + b), (.02, .09 + b), (.22, .08 + b), (.2, .3 + b), (-.1, .3 + b)], True, n=6)), '#7f879c', .8)
    if s > 40:
        for k in range(7): q = np.array([-.18 + .06 * k, .05 + .012 * (k % 2) + b]); cel.line(X([q, q + [.035, .045]]), lw * .5, ink, taper=(.1, .7), seed=k)
    cel.unclip(); kit.outline(cel, B, lw, ink, seed=seed + 2)
    bk = X(spl([(-.315, -.015 + b), (-.40, .015 + b), (-.425, .055 + b), (-.385, .050 + b), (-.31, .045 + b)], True, corners=(2,), n=5)); cel.fill(bk, '#e2982e'); cel.fill(X([(-.39, .030 + b), (-.372, .022 + b), (-.368, .046 + b), (-.386, .048 + b)]), '#b3121f'); kit.outline(cel, bk, lw * .7, ink, seed=9); cel.line(X([(-.315, .020 + b), (-.40, .034 + b)]), lw * .5, ink)
    t = np.linspace(0, 6.28, 14); cel.fill(X(np.c_[-.262 + .020 * np.cos(t), -.018 + b + .021 * np.sin(t)]), ink); cel.fill(X(np.c_[-.267 + .006 * np.cos(t), -.024 + b + .006 * np.sin(t)]), '#ffffff')
    _wing(cel, X, (.04, -.05 + b), (-.18, el + b), (-.50, (el + tip) * .64 + b), (-.98, tip + b), (.10, 1.), fill, shade, ink, lw, s, seed, 1.)
    return cel


def blade(c, p, ang, s=1., grey='#8d929c'):
    """a spear blade, white on black, edge on the diagonal: ridge, bevel, hatch, socket and shaft"""
    R_ = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]]); X = lambda P: np.asarray(P, float) * s @ R_ + p
    out = spl([(-640, -34), (-560, -40), (-520, -120), (-400, -150), (-120, -170), (260, -104), (620, 0), (260, 104), (-120, 170), (-400, 150), (-520, 120), (-560, 40), (-640, 34)], True, corners=(0, 2, 6, 10, 12), n=8); c.fill(X([(-3000, -27), (-630, -27), (-630, 27), (-3000, 27)]), '#d8d4ca'); c.fill(X([(-3000, 6), (-630, 6), (-630, 27), (-3000, 27)]), '#5a5d66')
    c.fill(X(out), PAPER); c.clip(X(out)); c.fill(X([(-560, 4), (620, 0), (260, 130), (-120, 190), (-560, 150)]), grey); c.fill(X([(-560, 92), (-120, 118), (300, 60), (620, 0), (260, 140), (-120, 200), (-560, 170)]), '#33363f')
    for k in range(26): q = -500 + k * 40; c.line(X(spl([(q, 4), (q + 26, 44), (q + 70, 96 - abs(k - 10) * 3)], n=5)), 3.6, INK, taper=(.05, .75), seed=k)
    c.unclip(); c.line(X([(-560, 3), (0, 2), (600, 0)]), 5, INK, taper=(.02, .5), seed=1); c.fill(X([(-700, -46), (-590, -52), (-590, 52), (-700, 46)]), '#c9a24e'); c.fill(X([(-700, 8), (-590, 10), (-590, 52), (-700, 46)]), '#6e4f1e')
    for k in (-690, -600): c.line(X([(k, -48), (k, 48)]), 5, INK, taper=(.02, .02))
    for k in range(9): o = -60 + 15 * k; c.line(X(spl([(-640, o * .7), (-760, o * 1.6 - 30), (-900 - 30 * (k % 3), o * 2.2 - 90)], n=8)), 13, '#b3121f', taper=(.1, .5), seed=k)


class Shot:
    id = 'a3'; n = 30; finish = dict(bloom=.12, vig=.10, soft=.06)
    EVENTS = [(0, 'blade', dict(size=1., note='first spear falls')), (2, 'blade', dict(size=1., note='second spear')), (5, 'stop', dict(note='silence, breath held')), (10, 'impact', dict(size=1.4, note='DON!! letters slam')), (10, 'gulls', dict(size=1., note='gulls burst')), (14, 'paper', dict(note='white eats the frame', dur=15)), (24, 'wing', dict(size=.8, note='wing over the lens'))]

    def setup(self):
        import os
        c = os.path.join(DK.ROOT, 'cache', 'dr_a3_slash_4.npz')
        if not os.path.exists(c):
            pg = sumi.Page(); kw = dict(dry=.30, deplete=.95, load=.9, split=.9, fork=1., speed=1.2)
            pg.stroke([(-160, 40, 60, 1.2), (200, 190, 330, 1.2), (900, 480, 270, 1.6), (1500, 730, 150, 1.8), (2080, 968, 10, 2.)], t0=0, t1=1, seed=4, **kw)
            pg.stroke([(2080, -30, 50, 1.2), (1720, 170, 310, 1.2), (1050, 520, 250, 1.6), (420, 850, 130, 1.8), (-120, 1130, 8, 2.)], t0=0, t1=1, seed=9, **kw); sumi.page_save(pg, c)
        pg = sumi.page_load(c); self.pg = pg; self.s1, self.s2 = pg.ink(1e9, only=(0,))[0], pg.ink(1e9, only=(1,))[0]
        self.tooth = sumi.tooth(H, W); self.nz = fbm(H, W, (150, 60, 22, 8), (1, .7, .5, .3), 5); self.nf = fbm(H, W, (40, 14, 5), (1, .7, .5), 8)
        self.bst, self.bd = sumi.burst(H, W, 960, 540, 3, r0=160.)
        x, y = grid(); d = np.array([2240., 928.]); d /= np.hypot(*d); self.dn = ((x + 160) * -d[1] + (y - 40) * d[0]); e2 = np.array([-2200., 1160.]); e2 /= np.hypot(*e2); self.dn2 = ((x - 2080) * -e2[1] + (y + 30) * e2[0]); self.al = (x + 160) * d[0] + (y - 40) * d[1]; self.nv = np.array([-d[1], d[0]]); self.dv = d
        r = np.random.default_rng(21); self.g = []
        for i in range(44):
            a = r.uniform(0, 6.283); sp = r.uniform(.55, 1.5); self.g.append(dict(a=a, sp=sp, s=r.uniform(20, 54) * (1.7 if i % 7 == 0 else 1), ph=r.random(), hz=r.uniform(2.4, 3.6), rot=r.uniform(-.4, .4), r0=r.uniform(20, 340), fl=-1 if math.cos(a) > 0 else 1, st=int(r.integers(0, 4))))


    def black(self, m, red=0., rimw=None):
        img = np.ones((H, W, 3), f32) * rgb(INK) * (1 + .5 * np.clip(self.nf, -1, 1)[..., None] * .25)
        if red: g = blur_(m, 26); img = img * (1 - (g * red)[..., None]) + rgb(RED) * (g * red)[..., None]
        return img * (1 - m[..., None]) + rgb(PAPER) * m[..., None]

    def frame(self, f):
        pg = self.pg; tm = lambda i, t: pg.ink(t, only=(i,))[0]
        if f < 6:
            if f == 0: m = tm(0, .52)
            elif f == 1: m = self.s1
            elif f == 2: m = np.maximum(self.s1 * smooth(.2, .5, self.tooth + .42), tm(1, .55))
            elif f == 3: m = np.maximum(self.s1 * smooth(.15, .6, self.tooth + .18), self.s2)
            else:
                z = 1 + .05 * (f - 3); q = np.maximum(self.s1, self.s2); q = cv2.warpAffine(q, np.float32([[z, 0, 960 * (1 - z)], [0, z, 500 * (1 - z)]]), (W, H)); b = self.bst * smooth(60, 520, self.bd) * smooth(1300, 700, self.bd); m = np.clip(np.maximum(q, b * (.9 if f == 4 else .55)), 0, 1)
            img = self.black(m.astype(f32), .85 if f in (3, 4) else .4)
            c = Cel(); r = R(f // 2, 3)
            for i in range(26 if f > 1 else 14): t = r.random(); p = np.array([-160, 40]) + np.array([2240, 928]) * t + self.nv * r.normal(0, 60); c.line([p, p + self.dv * r.uniform(40, 260)], r.uniform(1, 4.5), PAPER, taper=(.2, .7), seed=i)
            l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
            if f < 4:
                c = Cel(); A1, A2 = math.atan2(928, 2240), math.atan2(1160, -2200)
                if f < 2: blade(c, np.array([-160, 40]) + np.array([2240, 928]) * (.34, .74)[f], A1, 1.12)
                else: blade(c, np.array([2080, -30]) + np.array([-2200, 1160]) * (.36, .76)[f - 2], A2, 1.12)
                l = c.layer(); l = kit.smear(l, *((90, 37) if f < 2 else (-85, 45)), 13) * .55 + l * .45; img = img * (1 - l[..., 3:4]) + l[..., :3]
            o = kit.shake(f, 9 if f in (1, 3, 4) else 0, 3); return cv2.warpAffine(img, np.float32([[1, 0, o[0]], [0, 1, o[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101).astype(f32)
        if f < 10:                                                     # the breath: one thread of light dies
            w = (150., 92., 46., 20.)[f - 6]; k = (1., 1., 1., 1.)[f - 6]; rg = 1 + .05 * self.nf; L1 = np.clip(w * .5 * rg - np.abs(self.dn + 2 * self.nz) + .5, 0, 1); L2 = np.clip(w * .42 * rg - np.abs(self.dn2 + 2 * self.nz) + .5, 0, 1); ln = np.maximum(L1, L2) * np.maximum(smooth(.25, .5, self.tooth + .30), .55 * (f < 8))
            img = np.ones((H, W, 3), f32) * rgb(INK) * (1 + .12 * np.clip(self.nf, -1, 1))[..., None]; g = np.maximum(np.exp(-np.abs(self.dn) / (w * 1.1 + 20)), np.exp(-np.abs(self.dn2) / (w * 1.1 + 20))) * .62; img = img + g[..., None] * rgb('#8a0f1a'); col = rgb(PAPER)
            img = img * (1 - ln[..., None]) + col * ln[..., None]
            c = Cel()
            for i in range(30): t = h1(i, 4); q = np.array([-160, 40]) + np.array([2240, 928]) * t + self.nv * (12 + 90 * h1(i, 5)) * (1 if i % 2 else -1) * (.4 + .25 * (f - 6)); s = 1.5 + 4 * h1(i, 6) ** 2; c.fill(np.c_[q[0] + np.cos(np.linspace(0, 6.28, 9)) * s, q[1] + np.sin(np.linspace(0, 6.28, 9)) * s * .6], '#c8222c', .8 - .15 * (f - 6))
            l = c.layer(); return (img * (1 - l[..., 3:4]) + l[..., :3]).astype(f32)
        # ---- DON!!
        k = f - 10
        b = np.clip(self.bst * smooth(120 + 30 * k, 520 + 60 * k, self.bd), 0, 1); base = rgb(IRON) * (1 + .10 * np.clip(self.nz, -1.5, 1.5)[..., None]); img = base * (1 - b[..., None] * .9) + rgb(GOLD) * (1 + .12 * self.nf[..., None]) * b[..., None] * .9
        t = np.clip(1 - self.bd / 1300., 0, 1) ** 2; img = img + (t * .18)[..., None] * rgb('#ffcf80')
        if f == 10: img = rgb(PAPER) * (1 - b[..., None]) + rgb(INK) * b[..., None]
        z = 1 + .010 * k; img = cv2.warpAffine(img.astype(f32), np.float32([[z, 0, 960 * (1 - z)], [0, z, 540 * (1 - z)]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)
        c = Cel()
        self.gulls(c, f, lambda g: g['s'] < 46, '#20242e', '#12141c')
        l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        lay = don.don(k, 'full', accent='#a3161f' if f != 10 else '#e01f2c', life=40, out=1, pitch=17., key=15, rot=0.); img = img * (1 - lay[..., 3:4]) + lay[..., :3]
        # white eats the frame from the cut
        if f >= 14:
            e = (f - 14) / 14.; R_ = -60 + 900 * e ** 1.35 + 60 * e; edge = np.abs(self.dn) + 70 * self.nz + 22 * self.nf * (.5 + e) - 90 * smooth(.55, 1., np.sin(self.al / 130. + 2.2 * self.nz + f * .35))
            m = smooth(6, -6, edge - R_); ik = smooth(12, 2, np.abs(edge - R_ - 5)) * (.75 if e < .9 else .2) * smooth(.2, .6, self.tooth + .3); img = img * (1 - ik[..., None]) + rgb(INK) * ik[..., None]
            pw = rgb('#fbf9f4') * (1 + .010 * self.nf[..., None] - .012 * np.clip(self.nz, 0, 2)[..., None]); img = img * (1 - m[..., None]) + pw * m[..., None]
        c = Cel(); self.gulls(c, f, lambda g: g['s'] >= 46, '#fbf9f4', '#a9b0c2')
        # a flock rides the front of the white
        if f >= 14:
            for i in range(26):
                u = h1(i, 41); sg = 1 if i % 2 else -1; q = np.array([-160, 40]) + np.array([2240, 928]) * u + self.nv * sg * (max(0, R_) * (.70 + .34 * h1(i, 42)) + 20) + self.dv * (f - 14) * (h1(i, 43) - .5) * 26; s = (30 + 52 * h1(i, 44) ** 2) * (.6 + .9 * e)
                gull(c, q[0], q[1], s, (f // 2 * 2) / 24. * 3.1 + h1(i, 45), .5 * (h1(i, 46) - .5), '#fbf9f4', '#a9b0c2', lw=1.6 + s * .02, flip=1 if h1(i, 47) > .5 else -1, seed=i)
        if f >= 20:                                                    # the one that comes at us: its wing becomes the white of a4
            q = (f - 20) / 7.; s = 150 * 18 ** (q ** 1.1); gull(c, 1180 + 560 * q ** 1.6, 330 + 420 * q ** 2, s, .86 - .50 * q, -.25 + .55 * q, '#fbf9f4', '#c4cad8', lw=min(12, 2 + s * .006), flip=1, seed=77)
        l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if f >= 26: w = smooth(24.6, 27.8, f); img = img * (1 - w) + rgb('#fdfbf7') * w
        o = kit.shake(f, 16, 8, 2.6, 10); return cv2.warpAffine(img, np.float32([[1, 0, o[0]], [0, 1, o[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101).astype(f32)

    def gulls(self, c, f, sel, fill, shade):
        k = f - 10
        for i, g in enumerate(self.g):
            if not sel(g) or k < g['st']: continue
            t = (k - g['st']) / 24.; d = g['r0'] + 2300 * g['sp'] * (t - .55 * t * t); x, y = 960 + math.cos(g['a']) * d * 1.1, 560 + math.sin(g['a']) * d * .62 - 240 * t
            s = g['s'] * (.45 + 2.4 * t * g['sp'])
            gull(c, x, y, s, g['ph'] + (f // 2 * 2) / 24. * g['hz'], g['rot'], fill, shade, ink=INK if fill[1] == 'f' else '#0a0b10', lw=1.2 + s * .022, flip=g['fl'], seed=i)


def blur_(m, s): return cv2.resize(cv2.GaussianBlur(cv2.resize(m, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), s / 4.), (W, H))
