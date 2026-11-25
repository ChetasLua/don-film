"""a4: the gull's wing lifts off lens: 118 ships leave a harbour town, their wakes are the rays of a sun."""
import numpy as np, cv2, math, os
from op import W, H, ships, sea
from op.cam import Camera3D
from op.sea import Sea, TROPIC, pnoise, _samp
from op.paint import *
from op.ink import Cel, spl, resample
from op.kit import outline, smear, rim, para
from op.comp import blur, ease, ease_out, lerp_keys
from op.seakit import shear_pose, gull_top, puff, near_wing
A_ = np.asarray
L_ = A_([-.50, .42, .72]); L_ = L_ / np.linalg.norm(L_)
M_, N_ = 250., 2400
TH = np.linspace(0, 2 * math.pi, 420); BAY = (A_([58., -62.]), 47.)                        # crescent bay bitten out of the south-east shore


def _nz(a, s): return sum(np.sin(a * q + r0) * w for q, w, r0 in ((2, .6, s), (3, .7, s * 2.1), (5, .4, s * 3.3), (9, .20, s * 4.1), (14, .12, s)))


def _shore():
    R = 80 + 6.0 * _nz(TH, 1.); out = []
    for a, r in zip(TH, R):
        d = A_([math.cos(a), math.sin(a)]); c, rb = BAY; b = d @ c; disc = b * b - (c @ c - rb * rb)
        if disc > 0 and b > 0: r = min(r, max(26., b - math.sqrt(disc)))
        out.append(r)
    k = np.hanning(7); k /= k.sum(); return np.convolve(np.r_[out[-4:-1], out, out[1:4]], k, 'same')[3:-3]


SH_ = _shore()
def rad(a): return np.interp(np.mod(a, 2 * math.pi), TH, SH_)


def town():
    """harbour town from above -> premult RGBA plate N_ px, M_ metres: hill + lighthouse, quays, two stone moles, houses with walls and shadows, the square"""
    k = N_ / M_; c = Cel(N_, N_, ss=1); r = np.random.default_rng(4); X = lambda P: np.atleast_2d(A_(P, float)) * [k, -k] + N_ / 2; e = np.linspace(0, 6.283, 18); SD = A_([.62, -.52])         # shadow direction per metre of height
    disc = lambda p, rr, ry=None: A_(p) + np.c_[np.cos(e) * rr, np.sin(e) * (ry or rr)]; ring = lambda m: np.c_[np.cos(TH) * (SH_ + m), np.sin(TH) * (SH_ + m)]
    c.fill(X(ring(9)), '#d9f3e0', .30); c.fill(X(ring(4.5)), '#f1edcf', .65)
    # ------- piers on the open coast, then the two curved moles that close the bay
    def slab(P, w, col, sh=1.4, lw=1.6, seed=0):
        S = spl(A_(P, float), n=14); T = np.gradient(S, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; n = np.c_[-T[:, 1], T[:, 0]]; Q = np.vstack([S + n * w, (S - n * w)[::-1]]); c.fill(X(Q + SD * sh), '#0a5f7c', .5); c.fill(X(Q + [0, -.55]), '#6b5a44'); c.fill(X(Q), col); outline(c, X(Q), lw, '#4a3524', seed=seed); return S, n
    for i in range(24):
        a = i / 24 * 6.283 + r.uniform(-.08, .08); R0 = rad(a); d = A_([math.cos(a), math.sin(a)])
        if np.hypot(*(d * R0 - BAY[0])) < BAY[1] + 8: continue
        d2 = d + A_([-d[1], d[0]]) * r.uniform(-.25, .25); d2 /= np.hypot(*d2); S, n = slab([d * (R0 - 5), d * R0 + d2 * 4, d * R0 + d2 * r.uniform(9, 24)], r.uniform(.8, 1.4), '#c9975a', seed=i)
        for j in range(4, len(S) - 2, 3): c.line(X([S[j] + n[j] * 1.1, S[j] - n[j] * 1.1]), 1.1, '#7b5430', taper=(0, 0), alpha=.5)
    c.fill(X(ring(0)), '#cdb88c'); c.fill(X(ring(0) + [0, -1.2]), '#8c7653'); c.fill(X(ring(0)), '#dcc89a'); outline(c, X(ring(0)), 3.2, '#5c4630', seed=2)
    hb, rb = BAY; ha = np.linspace(0, 2 * math.pi, 200); quay = [hb + A_([math.cos(q), math.sin(q)]) * rb for q in np.linspace(math.radians(70), math.radians(200), 60)]
    c.line(X(A_(quay) + (A_(quay) - hb) / rb * 2.2), 58, '#efe4c6', taper=(0, 0), press=.04); c.line(X(A_(quay) + (A_(quay) - hb) / rb * .1), 4.2, '#8c7653', taper=(0, 0))
    for j in range(0, 60, 2): q = A_(quay[j]); d = (q - hb) / rb; c.line(X([q + d * .4, q + d * 4.4]), 1., '#b39c72', taper=(0, 0), alpha=.6)
    for (a0, a1, bow, sd) in ((68, 12, 1.06, 5), (204, 262, 1.04, 8)):
        P = [hb + A_([math.cos(math.radians(q)), math.sin(math.radians(q))]) * rb * (1 + (bow - 1) * v) for v, q in zip(np.linspace(0, 1, 9), np.linspace(a0, a1, 9))]; S, n = slab(P, 2.0, '#e9dfc4', 2.2, 2.0, sd)
        for j in range(2, len(S) - 1, 4): c.line(X([S[j] + n[j] * 1.8, S[j] - n[j] * 1.8]), 1.1, '#a8916a', taper=(0, 0), alpha=.6)
        t = S[-1]; c.fill(X(disc(t, 3.4)), '#e9dfc4'); outline(c, X(disc(t, 3.4)), 1.8, '#4a3524', seed=sd); c.fill(X(np.vstack([t + [-1.2, 0], t + [1.2, 0], t + SD * 9 + [.7, 0], t + SD * 9 - [.7, 0]])), '#2a3f4c', .45); c.fill(X([t + [-1.2, -.6], t + [1.2, -.6], t + [.8, 4.6], t + [-.8, 4.6]]), '#fbf6e8' if sd == 8 else '#c9483a'); outline(c, X([t + [-1.2, -.6], t + [1.2, -.6], t + [.8, 4.6], t + [-.8, 4.6]]), 1.5, '#3a2a20', seed=sd); c.fill(X(disc(t + [0, 5.0], 1.1, .9)), '#e2a64a')
    for i in range(26):                                                              # boats asleep in the basin
        p = hb + A_([math.cos(q := r.uniform(0, 6.28)), math.sin(q)]) * rb * math.sqrt(r.random()) * .86
        if np.hypot(*p) < rad(math.atan2(p[1], p[0])) + 4: continue
        an = r.uniform(0, 6.28); u = A_([math.cos(an), math.sin(an)]); n = A_([-u[1], u[0]]); L = r.uniform(1.6, 3.6); B = spl(A_([p - u * L - n * L * .22, p + n * L * .30, p + u * L * 1.2, p - n * L * .30, p - u * L + n * L * .22]), True, corners=(2,), n=5); c.fill(X(B + SD * .9), '#0a6b86', .45); c.fill(X(B), ['#a9703c', '#c9483a', '#e9dfc4', '#3f6a8a'][i % 4]); outline(c, X(B), 1.3, '#3a2416', seed=i)
        if i % 2: c.fill(X([p + n * L * .8, p + u * L * .4, p - n * L * .8 + u * .2]), '#f6efdc'); outline(c, X([p + n * L * .8, p + u * L * .4, p - n * L * .8 + u * .2]), 1.1, '#3a2416', seed=i + 1)
    # ------- the hill, in three tiers, a path, trees, the lighthouse
    hc = A_([-38., 40.]); tiers = [(27, 0, '#6f9a4e'), (20, 2.5, '#7fab56'), (12.5, 5.5, '#93bf62')]
    for i, (R, up, col) in enumerate(tiers):
        P = np.c_[hc[0] + np.cos(TH) * (R + 1.6 * _nz(TH, 3. + i)) * 1.12, hc[1] + up + np.sin(TH) * (R + 1.4 * _nz(TH, 5. + i)) * .86]; c.fill(X(P + SD * 4), '#3f5a3a', .45); c.fill(X(P + [0, -1.8]), '#5a4f3c' if i else '#7a6a48'); c.fill(X(P), col); outline(c, X(P), 1.8, '#33482a', seed=i)
        for j in range(26): q = r.uniform(0, 6.28); p = hc + [math.cos(q) * R * r.uniform(.5, .95), up + math.sin(q) * R * .8 * r.uniform(.4, .95)]; c.line(X([p, p + [r.uniform(.8, 2.4), r.uniform(-.8, .2)]]), 1.4, '#4f7a3c', alpha=.6, seed=j)
    pth = spl(A_([(-14, 24), (-24, 30), (-44, 26), (-54, 38), (-40, 44), (-30, 42), (-38, 50)]), n=12); c.line(X(pth), 13, '#e9d9a8', taper=(.02, .02)); lh = hc + [0, 7.]; c.fill(X([lh + [-2.4, 0], lh + [2.4, 0], lh + SD * 24 + [1.6, 1], lh + SD * 24 - [1.6, 1]]), '#2c4030', .55); c.fill(X(disc(lh, 4.2, 3.2)), '#efe4c6'); outline(c, X(disc(lh, 4.2, 3.2)), 1.6, '#4a3524', seed=4)
    tw = [lh + [-2.5, -.6], lh + [2.5, -.6], lh + [1.7, 12.6], lh + [-1.7, 12.6]]; c.fill(X(tw), '#fbf6e8'); c.clip(X(tw)); c.fill(X([lh + [.4, -2], lh + [3, -2], lh + [3, 14], lh + [.2, 14]]), '#bfc2cc'); [c.fill(X([lh + [-3, y0], lh + [3, y0], lh + [3, y0 + 1.9], lh + [-3, y0 + 1.9]]), '#c9483a') for y0 in (2.2, 7.4)]; c.unclip(); outline(c, X(tw), 2.0, '#3a2a20', seed=5); c.fill(X(disc(lh + [0, 13.4], 2.3, 1.5)), '#e9dfc4'); c.fill(X(disc(lh + [0, 14.4], 1.5, 1.5)), '#f2c14a'); outline(c, X(disc(lh + [0, 14.4], 1.5, 1.5)), 1.6, '#3a2a20', seed=6); c.fill(X([lh + [-1.7, 15.2], lh + [1.7, 15.2], lh + [0, 17.2]]), '#c9483a')
    # ------- streets that converge on the square
    ST = []
    for a, bow in ((.25, 8), (1.05, -9), (1.75, 7), (2.9, -7), (3.5, 9), (4.35, -9), (5.0, 7), (5.95, -7)): d = A_([math.cos(a), math.sin(a)]); n = A_([-d[1], d[0]]); R = rad(a) - 2; ST.append(spl(A_([d * 8, d * R * .4 + n * bow * .8, d * R * .7 - n * bow * .5, d * R]), n=12))
    for fr, sd in ((.52, 1.), (.86, 2.7)): ST.append(np.c_[np.cos(TH) * (SH_ * fr + 1.6 * _nz(TH, sd)), np.sin(TH) * (SH_ * fr + 1.6 * _nz(TH, sd))])
    for S in ST: c.line(X(S), 27, '#c2aa78', taper=(0, .05), press=.1, tmin=.8)
    for S in ST: c.line(X(S), 23, '#f2e5c0', taper=(0, .05), press=.1, tmin=.8)
    SP = np.vstack([resample(S, 1.) for S in ST]); HL = []; roofs = (['#d4643a', '#c9552f', '#de7a45'], ['#e0a64c', '#d6973e', '#ebb964'], ['#5a7f9a', '#4d7090', '#6f94aa']); walls = ['#fbf4e4', '#f4e3bf', '#e7c587', '#fffaf0', '#ecd2a0']
    for _ in range(14000):
        a = r.uniform(0, 6.283); R = 80 * math.sqrt(r.random()); p = A_([math.cos(a) * R, math.sin(a) * R])
        if R < 15 or R > rad(a) - 5.2 or ((p - hc - [0, 3]) @ ((p - hc - [0, 3]) * [1, 1.6])) < 24 ** 2: continue
        ds = np.hypot(*(SP - p).T); j = int(ds.argmin()); sz = r.uniform(1.9, 3.5) * (1.35 if r.random() < .10 else 1)
        if ds[j] < 1.3 + sz * .66 or any((p[0] - h[0]) ** 2 + (p[1] - h[1]) ** 2 < ((sz + h[2]) * .83) ** 2 for h in HL): continue
        t = SP[min(j + 2, len(SP) - 1)] - SP[max(j - 2, 0)]; HL.append((p[0], p[1], sz, math.atan2(t[1], t[0]) + r.normal(0, .06) + (math.pi / 2 if r.random() < .25 else 0), r.uniform(2.2, 4.6)))
        if len(HL) > 560: break
    hue = lambda p: 0 if r.random() < .60 else 1 if math.sin(p[0] * .06 + p[1] * .04) > -.2 else 2
    for p_ in HL: p = A_(p_[:2]); u = A_([math.cos(p_[3]), math.sin(p_[3])]); n = A_([-u[1], u[0]]); L, B = p_[2] * 1.1, p_[2] * .74; P = A_([p + u * L + n * B, p - u * L + n * B, p - u * L - n * B, p + u * L - n * B]); Q = P + SD * p_[4]; c.fill(X(cv2.convexHull(np.vstack([P, Q]).astype(np.float32))[:, 0]), '#3c3448', .46)
    for i, (x, y, sz, an, hh) in enumerate(sorted(HL, key=lambda h: -h[1])):
        p = A_([x, y])
        if r.random() < .13: c.fill(X(disc(p, sz * .66) + SD * 1.6), '#4a4a38', .45); c.fill(X(disc(p, sz * .66)), '#477f3f'); c.fill(X(disc(p + [-.3, .6], sz * .46)), '#6aa84f'); c.fill(X(disc(p + [-.5, .9], sz * .24)), '#9acb6a'); outline(c, X(disc(p, sz * .66)), 1.3, '#274a26', seed=i); continue
        u = A_([math.cos(an), math.sin(an)]); n = A_([-u[1], u[0]]); L, B = sz * r.uniform(.95, 1.25), sz * r.uniform(.64, .80); P = A_([p + u * L + n * B, p - u * L + n * B, p - u * L - n * B, p + u * L - n * B]); v = A_([0, hh * .62]); T = P + v; wc = rgb(walls[int(r.integers(0, 5))]); col = rgb(roofs[hue(p)][int(r.integers(0, 3))]) * r.uniform(.93, 1.06)
        for q in range(4):
            a, b = P[q], P[(q + 1) % 4]; m = (a + b) / 2 - p
            if m[1] < -.05: f = A_([a, b, b + v, a + v]); lit = m[0] < 0; c.fill(X(f), tuple(np.clip(wc * (1. if lit else A_([.66, .60, .68])), 0, 1))); [c.fill(X(a + (b - a) * w0 + v * .32 + A_([[-.22, 0], [.22, 0], [.22, .75], [-.22, .75]])), '#4a3a40') for w0 in (.28, .68)]; outline(c, X(f), 1.2, '#4a3428', seed=i * 4 + q)
        rd0, rd1 = p + v - u * L, p + v + u * L; rg = A_([0, sz * .24]); lit = n @ [-.5, .42] > 0; s0, s1 = A_([T[0], T[1], rd0 + rg, rd1 + rg]), A_([T[2], T[3], rd1 + rg, rd0 + rg]); c.fill(X(s0), tuple(np.clip(col * (1.06 if lit else A_([.70, .60, .66])), 0, 1))); c.fill(X(s1), tuple(np.clip(col * (A_([.70, .60, .66]) if lit else 1.06), 0, 1)))
        for q in (.2, .5, .8): c.line(X([s0[0] + (s0[1] - s0[0]) * q, s0[3] + (s0[2] - s0[3]) * q]), .9, '#5a2c20', alpha=.4); c.line(X([s1[0] + (s1[1] - s1[0]) * q, s1[3] + (s1[2] - s1[3]) * q]), .9, '#5a2c20', alpha=.4)
        c.line(X([rd0 + rg, rd1 + rg]), 2.0, '#fff3d4', taper=(.05, .05), alpha=.8); outline(c, X(T), 1.5, '#43241c', seed=i)
        if sz > 3.5: t = p + v; c.fill(X([t + [-.9, 0], t + [.9, 0], t + [.9, 4.4], t + [-.9, 4.4]]), '#fbf4e4'); c.fill(X([t + [.2, 0], t + [.9, 0], t + [.9, 4.4], t + [.2, 4.4]]), '#b9b4c0'); c.fill(X([t + [-1.2, 4.2], t + [1.2, 4.2], t + [0, 7.2]]), '#4d7090'); outline(c, X([t + [-.9, 0], t + [.9, 0], t + [1.2, 4.2], t + [0, 7.2], t + [-1.2, 4.2]]), 1.4, '#3a2a28', seed=i)
    # ------- the square: crowd, the tiny tower
    sq = np.c_[np.cos(TH) * (12.5 + .8 * _nz(TH, 2.)), np.sin(TH) * (12.5 + .8 * _nz(TH, 2.))]; c.fill(X(sq), '#f7e9c2'); outline(c, X(sq), 1.8, '#a9824a', seed=3)
    for rr in (6, 9.5): c.line(X(np.c_[np.cos(TH), np.sin(TH)] * rr), 1.2, '#d8c08c', closed=True, alpha=.8)
    for i in range(900):
        a, d = r.uniform(0, 6.28), 12 * math.sqrt(r.random())
        if d < 2.8: continue
        p = A_([math.cos(a), math.sin(a)]) * d; c.fill(X(p + [.3, -.25] + np.c_[np.cos(e), np.sin(e)] * .30), '#8a6a44', .5); c.fill(X(p + np.c_[np.cos(e), np.sin(e)] * .30), ['#5a4a52', '#8a4a3a', '#4a5a78', '#e9dcc0', '#7a3a3a'][i % 5])
    bx = lambda m, y: A_([[-m, -m * .8 + y], [m, -m * .8 + y], [m, m * .8 + y], [-m, m * .8 + y]]); c.fill(X(np.vstack([bx(1.7, 0)[[0, 2]], bx(1.7, 0)[[2]] + SD * 13, bx(1.7, 0)[[0]] + SD * 13])), '#3c3448', .55)
    for q in (-1.5, 1.5): c.line(X([[q, -1], [q * .8, 8]]), 5, '#5a3a22', taper=(0, 0)); c.line(X([[q, 2], [-q, 6]]), 2.6, '#5a3a22', taper=(0, 0))
    c.fill(X(bx(2.0, 8.6)), '#8a5a34'); outline(c, X(bx(2.0, 8.6)), 1.8, '#2a1a10', seed=1)
    for q in (-.9, .9): c.line(X([[q, 8.6], [q, 10.6]]), 1.6, '#17131a'); c.line(X([[q * .2, 10.8], [q * .9 + .2 * q, 9.6]]), 1.2, '#17131a')
    lay = c.rgba(); n = fbm(N_, N_, (90, 30, 8), None, 3)[..., None]; lay[..., :3] *= 1 + .045 * n; return lay


class Shot:
    id = 'a4'; n = 60; finish = dict(bloom=.30, vig=.22)
    EVENTS = [(0, 'whoosh', dict(size=1.0, note='gull wing sweeps off the lens, feathers f0-9')), (3, 'gull', dict(note='near cry')), (6, 'sea', dict(note='harbour opens: wind, 118 hulls, distant bells')), (14, 'bell', dict(note='harbour bell far below')), (22, 'gull', dict(note='two gulls cross')), (30, 'wind', dict(size=.8, note='camera climbing, cloud shadows')), (44, 'air', dict(note='near cloud slides past lens'))]

    def setup(self):
        p = 'cache/a4_town3.npy'
        if not os.path.exists(p): np.save(p, town().astype(np.float16))
        self.town = np.load(p).astype(f32); self.S = Sea(5, hs=.55, length=22, wind=(.8, .5), chop=.8); self.cl = []
        for i, (w, h, s) in enumerate(((900, 640, 3), (820, 620, 8), (1000, 660, 21))):
            p = f'cache/a4_puffc{i}.npy'
            if not os.path.exists(p): np.save(p, puff(w, h, s).astype(np.float16))
            self.cl.append(np.load(p).astype(f32))
        r = np.random.default_rng(11); self.sh = []; N = 118
        for i in range(N):
            a = i / N * 2 * math.pi + r.normal(0, .042) + .05 * math.sin(i * .9); u = (i * .61803 + r.uniform(-.12, .12)) % 1; R = float(rad(a)); d = max(R + 26, 100) + 340 * u ** .85; hd = a + r.normal(0, .085)
            m = ships.generic(i + 40, 0., 0, rk=.62, big=1.28); self.sh.append((m, a, d, r.uniform(7.5, 14.5) * (1.1 - .3 * u), m.L, R, hd, r.uniform(.45, 1.0) ** .8, r.uniform(.65, 1.75)))

    def cam(self, f):
        e = 1 - (1 - f / 59.) ** 1.7; h = 330 + 330 * e; ro = math.radians(-5 + 14 * ease(f / 59.)); y = -130 * (1 - ease(min(1, f / 46.)))
        return Camera3D((0, y, h), (0, 0, 0), 1500, cy=486, roll=ro)

    def frame(self, f):
        if f == 0: return np.zeros((H, W, 3), f32) + rgb('#fdfdf9') * (1 - .02 * fbm(H, W, (300,), None, 3)[..., None])
        t = f / 24.; cam = self.cam(f); poses = []; wk = []
        for m, a, d, v, L, R, hd, ln, wf in self.sh:
            q = d + v * t; x, y = math.cos(a) * d + math.cos(hd) * v * t, math.sin(a) * d + math.sin(hd) * v * t; poses.append((m, ships.pose(x, y, 0, hd))); W_ = L * .16
            wk.append(dict(x=x - math.cos(hd) * L * .40, y=y - math.sin(hd) * L * .40, head=hd, L=max(40., (q - R - 6) * ln), W=W_ * wf, amt=.99, dec=.80, arm=.9))
        def depth(X, Y): r = np.hypot(X, Y); return smooth(62, 400, r + 26 * _samp(pnoise(), X, Y, 420.) + 9 * _samp(pnoise(), X, Y, 120.))
        wt = self.S.render(cam, 2 + t, TROPIC, sun=L_, step=13, depth=depth, wakes=wk, foam_amt=0, lace_m=60, tone=.62, haze=0, crest=0); img = wt.rgb
        rf = smooth(.55, 1.0, _samp(pnoise(), wt.X + 60, wt.Y, 150.)) * smooth(70, 90, np.hypot(wt.X, wt.Y)) * (1 - smooth(120, 220, np.hypot(wt.X, wt.Y))); img = img * (1 - .30 * rf[..., None] * (1 - wt.foam[..., None]) * A_([1, .5, .45], f32))
        # a thread of surf round the island
        lay = cam.warp_plane(self.town, (-M_ / 2, M_ / 2, .4), (1, 0, 0), (0, -1, 0), M_, M_); sf = blur(lay[..., 3], 3.2) * (1 - lay[..., 3]); sf = np.clip(sf * 2.6 - .5 + .25 * np.sin(wt.X * .25 + wt.Y * .21 + t * 3), 0, 1)[..., None]; img = img + (1 - img) * sf * .7
        img = img * (1 - lay[..., 3:4]) + lay[..., :3]
        sd = ships.draw([(m, shear_pose(ps, L_)) for m, ps in poses], cam, L_, ss=1, ink=None, shadow=False, decals=False, texture=0); s = blur(sd.layer[..., 3], 1.1)[..., None]; img = img * (1 - s * .50 * A_([.95, .72, .55], f32))
        d = ships.draw(poses, cam, L_, ss=2, ink='edge', lw=1.15, shadow=False, texture=0, rim=('#fff6d8', 1.6, .55)); img = img * (1 - d.layer[..., 3:4]) + d.layer[..., :3]
        cl = [(0, -330 + 34 * t, 150, 60, 260), (1, 330 - 30 * t, -150, 90, 240), (2, 60 + 40 * t, 235, 150, 250), (1, 360 - 42 * t, 130, 130, 230), (0, -330 + 52 * t, -150, 210, 220), (2, -90 + 30 * t, -235, 120, 260)]
        for k, x, y, z, sz in cl:
            T = self.cl[k]; mh = sz * T.shape[0] / T.shape[1]; ox, oy = -L_[0] / L_[2] * z, -L_[1] / L_[2] * z; a = cam.warp_plane(T, (x + ox - sz / 2, y + oy + mh / 2, 0), (1, 0, 0), (0, -1, 0), sz, mh)[..., 3]; a = blur(a, 5)[..., None]; img = img * (1 - a * .34 * A_([1., .72, .42], f32))
        G = Cel()
        for i in range(7):
            z = 60 + 26 * i; x = -170 + 60 * i + (26 + 5 * i) * t * (1 if i % 2 else -1); y = 90 - 38 * i + 16 * t; p, zz = cam.project((x, y, z)); s = 5.2 * cam.f / zz; gull_top(G, p[0], p[1], s, (1.2 if i % 2 else -1.1) + .1 * math.sin(t * 2 + i), math.sin(t * 7 + i * 2), lw=1.1)
        g = G.layer(); img = img * (1 - g[..., 3:4]) + g[..., :3]
        for k, x, y, z, sz in sorted(cl, key=lambda q: q[3]):
            T = self.cl[k]; zz = z + 60; mh = sz * T.shape[0] / T.shape[1]; lay = cam.warp_plane(T, (x - sz / 2, y + mh / 2, zz), (1, 0, 0), (0, -1, 0), sz, mh)
            if z > 120: lay = blur(lay, 2.0)
            img = img * (1 - lay[..., 3:4] * .96) + lay[..., :3] * .96
        if f > 34:
            u = (f - 34) / 26.; T = self.cl[2]; s = 2.6; M2 = np.float32([[s, 0, -2700 + 1500 * u], [0, s, 260 - 120 * u]]); lay = cv2.warpAffine(T, M2, (W, H)); lay = blur(lay, 9); img = img * (1 - lay[..., 3:4] * .85) + lay[..., :3] * .85
        x, y = grid(); g = np.exp(-(((x - 960) ** 2 + (y - 486) ** 2) / (430. ** 2))); img = img + g[..., None] * rgb('#ffe6a8') * .09
        if f < 12:
            u = (f / 11.) ** 1.2; lay = near_wing(u, f); sh = blur(cv2.warpAffine(lay[..., 3], np.float32([[1, 0, -70], [0, 1, 110]]), (W, H)), 34)[..., None]; img = img * (1 - .34 * sh * A_([1., .7, .45], f32)); img = img * (1 - lay[..., 3:4]) + lay[..., :3]
            k = float(lerp_keys(f, [(0, 1.), (1, .62), (2, .34), (4, .0)])); img = img * (1 - k) + rgb('#fdfdf9') * k
        return img
