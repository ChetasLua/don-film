"""f1 60: the heart restarts as a drum seen through the chest, the hair bleaches and lifts into flame, the eyes open, the grin splits."""
import numpy as np, cv2, math
from op import W, H as HH
from op.paint import rgb, f32, smooth, ramp, grid, fbm, mix
from op.ink import Cel, spl, resample
from op.kit import Turn, para, hatch, outline, shape, flare, shifted
from op.comp import over, screen, blur, radial, ease, lerp_keys
from op.fx_core import h1, lin01, noise, noise_img, put, twos
from op.fx_fire import flames, embers, PAL as FP
from chars import gear5 as G
from chars.gear5 import ss, lay, back_out, rim

PULSE = [(0, 1.), (10, .55), (15, .6), (20, .7), (30, 1.), (40, .5), (45, .6), (50, 1.)]          # DON . do t to . | DON
HEART = (.17, 2.60)
eo = lambda t, p=2.6: 1 - (1 - np.clip(t, 0, 1)) ** p


def K(f, keys):
    """keys (frame, value...) with eased spans; 4th style: ('o' ease-out) as last element"""
    f0 = keys[0][0]
    if f <= f0: return np.asarray(keys[0][1], float)
    for a, b in zip(keys[:-1], keys[1:]):
        if f < b[0]:
            t = (f - a[0]) / (b[0] - a[0]); t = eo(t) if len(b) > 2 and b[2] == 'o' else t * t if len(b) > 2 and b[2] == 'i' else t * t * (3 - 2 * t)
            return np.asarray(a[1], float) * (1 - t) + np.asarray(b[1], float) * t
    return np.asarray(keys[-1][1], float)


def pulse(f, tau=3.4):
    return sum(a * math.exp(-(f - k) / tau) for k, a in PULSE if f >= k)


CAM = [(0, (.14, 2.66, 372)), (10, (.13, 2.56, 388)), (17, (.10, 2.05, 372)), (27, (.02, .62, 352)), (33, (0., .20, 400)), (35, (0., .21, 410)), (38, (0., .25, 600), 'o'), (42, (0., .27, 622)), (49, (0., .29, 600)), (52, (.02, -.16, 372), 'o'), (55, (.02, -.12, 340)), (60, (.02, -.12, 350))]
HP = dict(pitch=[(0, -.24), (28, -.28), (33, -.36), (36, .05, 'o'), (39, .02), (42, .10, 'o'), (45, .05), (49, -.14), (52, .36, 'o'), (55, .24), (60, .33)],
          yaw=[(0, .22), (33, .24), (37, .12, 'o'), (49, .13), (52, .20, 'o'), (60, .22)], roll=[(0, .10), (33, .13), (36, -.03, 'o'), (42, -.07, 'o'), (49, .02), (52, -.18, 'o'), (56, -.12), (60, -.16)])
EYE = [(35, 0), (36, .5), (37, 1.34), (39, 1.22), (42, 1.16), (49, 1.10), (51, 1.34), (54, 1.20), (60, 1.16)]
CHEEK = [(37, 0), (39, .06), (42, .44), (45, .34), (49, .56), (51, .04), (54, .30), (57, .52), (60, .46)]
BURST = lambda f: (.50 * math.exp(-(f - 42) / 3.5) if 42 <= f < 50 else 0.) + (1. * math.exp(-(f - 50) / 4.5) if f >= 50 else 0.)
IRIS = [(35, 1.7), (36, 1.6), (37, .50), (39, 1.02), (42, .88), (60, .90)]


class Shot:
    id = 'f1'; n = 60; finish = dict(bloom=.42, vig=.18)
    EVENTS = [(0, 'drum', dict(size=1.0, note='DON, light ring out of the chest, white light from e2')), (10, 'drum', dict(size=.5, note='do')), (14, 'fire', dict(size=.6, note='hair roots ignite, whoosh rising to f34')), (15, 'drum', dict(size=.55, note='t')), (20, 'drum', dict(size=.65, note='to')),
              (30, 'drum', dict(size=1.0, note='DON, fire behind')), (36, 'sting', dict(size=.9, note='eyes snap open, concentric red')), (38, 'whoosh', dict(size=.4, note='push-in lands')), (40, 'drum', dict(size=.5)), (45, 'drum', dict(size=.55)), (47, 'breath', dict(size=.5, note='wound-up giggle, breath in')),
              (50, 'drum', dict(size=1.0, note='DON: the laugh bursts, hair explodes')), (51, 'stretch', dict(size=.6, note='rubber mouth')), (34, 'stretch', dict(size=.3, note='mouth corners already pulling')), (42, 'laugh', dict(size=.6, note='breaks into the unhinged grin, hair bursts')), (51, 'laugh', dict(size=1.0, note='head rears back, open laugh; jaw beats f52 f54 f56 f58'))]

    def setup(self):
        self.gx, self.gy = grid()
        self.smoke = fbm(270, 480, (120, 50, 22, 9), None, 41)

    # ------------------------------------------------------------------ plates
    def bg(self, f, g):
        t = f / 24.; h, w = 270, 480
        y = np.linspace(0, 1, h, dtype=f32)[:, None]; n1 = noise_img(h, w, 74, t * .7, 5, 4, 1, .62, oy=t * 250); n2 = noise_img(h, w, 150, t * .35, 9, 3, 1, .8, oy=t * 120)
        x = np.linspace(-1, 1, w, dtype=f32)[None]
        v = .50 + .42 * n1 + .38 * n2 - .62 * (1 - y) ** 1.2 + .26 * np.abs(x) ** 1.6 - .04
        v = v + (g - 1) * .75
        col = ramp(np.clip(v, 0, 1), [(0, '#0a0406'), (.20, '#1d0709'), (.36, '#4c0d0d'), (.52, '#a11d0e'), (.66, '#e2491a'), (.80, '#ff8d2c'), (.95, '#ffd063')])
        img = cv2.resize(col, (W, HH), interpolation=cv2.INTER_CUBIC)
        if g > .12:
            cel = Cel(W // 2, HH // 2, ss=1); flames(cel, np.array([(-80, 575), (240, 590), (480, 600), (720, 590), (1040, 575)], float), 470 * g, t, 3, n=9, hot=False, ink=False, rows=2, lw=1., hz=1.3, spread=.16, core=True)
            L = cel.rgba(); B = cv2.GaussianBlur(L, (0, 0), 5.5); B = cv2.resize(B, (W, HH), interpolation=cv2.INTER_LINEAR)
            img = over(img, B, .78 * min(1, g * 1.4))
            self._near = cv2.resize(cv2.GaussianBlur(L[::-1, ::-1][::-1], (0, 0), 13), (W, HH), interpolation=cv2.INTER_LINEAR)
        else: self._near = None
        return img

    def body(self, T, S, f, sh_r):
        """-> (skin layer, shirt layer with the red / white split, heart-light masks helper)"""
        pal = G.PAL['fire']; ink = pal['ink']; lw = S * .0125; il = lw * .55
        zf = lambda P: .42 * np.clip(1 - (np.asarray(P, float)[:, 0] / 2.1) ** 2, 0, 1)
        X = lambda P: T(P, zf(P)); sx = lambda P, n=8, **k: X(spl(P, n=n, **k))
        cel = Cel(); tor = sx([(-.40, .55), (-.43, 1.16), (-.66, 1.50), (-1.28, 1.70), (-1.78, 1.96), (-2.12, 2.6), (-2.1, 3.6), (-1.6, 4.0), (-1.42, 5.6), (1.42, 5.6), (1.6, 4.0), (2.1, 3.6), (2.12, 2.6), (1.78, 1.96), (1.28, 1.70), (.66, 1.50), (.43, 1.16), (.40, .55)], corners=(0, 17))
        cel.fill(tor, pal['sk']); cel.clip(tor)
        for P in ([(.42, 1.1), (.30, 1.62), (.62, 2.5), (.74, 3.3), (.60, 4.2), (.66, 5.6), (2.6, 5.6), (2.6, 1.2)],                              # right side plane
                  [(-1.1, 2.98), (-.86, 3.16), (-.52, 3.28), (-.14, 3.16), (-.16, 3.04), (-.5, 3.16), (-.84, 3.02)], [(.16, 3.04), (.5, 3.16), (.9, 3.0), (.84, 3.16), (.5, 3.28), (.12, 3.16)],
                  [(-1.3, 1.80), (-.6, 1.76), (-.22, 1.84), (-.5, 1.92), (-1.0, 1.94)], [(.22, 1.84), (.6, 1.76), (1.3, 1.80), (1.0, 1.94), (.5, 1.92)], [(-.06, 2.0), (.04, 2.0), (.06, 3.1), (-.03, 3.1)]):
            cel.fill(sx(P + P[:1], 6), pal['s1'])
        cel.fill(sx([(-.46, .5), (-.40, 1.20), (-.20, 1.44), (.02, 1.36), (.30, 1.60), (.60, 1.48), (.46, .5)], 8), pal['s2'])                              # throat in shade
        for a, b, sd in (((-1.04, 1.98), (.98, 3.62), 1), ((1.02, 1.94), (-.98, 3.66), 2)):                                                           # burn scar
            a, b = np.array(a), np.array(b); d = b - a; n_ = np.array([-d[1], d[0]]) / np.hypot(*d); m = 16; e = []
            for q in (1, -1): e.append(np.array([a + d * k / m + n_ * q * (.17 + .055 * (h1(sd, k, q + 5) - .4) * 2) * (1 - .25 * abs(k / m - .5)) * 1.1 for k in range(m + 1)]))
            S_ = X(np.vstack([e[0], e[1][::-1]])); cel.fill(S_, '#8f3f35', .60)
            for q in e: cel.line(X(q), il * 1.05, pal['s2'], taper=(.15, .15), press=.6, seed=sd, tremor=.5)
            th = math.atan2(d[1], d[0]); hatch(cel, S_, th + .12, S * .05, il * .62, pal['s2'], sd, length=S * .20, jit=.4, alpha=.8)
        for P, w_ in (([(-.14, 1.76), (-.42, 1.64), (-.92, 1.60), (-1.34, 1.72)], 1.), ([(.14, 1.76), (.42, 1.64), (.92, 1.60), (1.34, 1.72)], 1.), ([(-.33, .98), (-.24, 1.34), (-.09, 1.70)], .9), ([(.33, .98), (.26, 1.34), (.09, 1.70)], .9),
                      ([(-.07, 1.72), (0, 1.80), (.07, 1.72)], .7), ([(-1.12, 2.90), (-.84, 3.14), (-.50, 3.24), (-.12, 3.10)], 1.1), ([(1.12, 2.90), (.86, 3.14), (.50, 3.24), (.12, 3.10)], 1.1), ([(0, 1.95), (.02, 2.5), (.0, 3.06)], .7), ([(.0, 3.3), (.02, 4.2), (0, 5.4)], .8),
                      ([(-.50, 3.5), (-.22, 3.62), (-.04, 3.56)], .7), ([(.50, 3.5), (.22, 3.62), (.04, 3.56)], .7), ([(-.52, 4.1), (-.22, 4.2), (-.04, 4.14)], .7), ([(.52, 4.1), (.22, 4.2), (.04, 4.14)], .7), ([(-1.0, 3.4), (-.74, 3.6)], .6), ([(-1.02, 3.62), (-.76, 3.84)], .6), ([(1.0, 3.4), (.74, 3.6)], .6)):
            cel.line(sx(P, 8), lw * .62 * w_, ink, taper=(.3, .35), press=.4, seed=int(abs(P[0][0] * 100)) + 3)
        for j in range(7): cel.line(X([(.36 + .05 * j, 2.02 + .05 * j), (.50 + .05 * j, 2.15 + .05 * j)]), il * .6, ink, taper=(.2, .6), seed=j)
        for j in range(6): cel.line(X([(-.98 + .07 * j, 2.86 + .045 * j), (-.88 + .07 * j, 2.99 + .04 * j)]), il * .6, ink, taper=(.2, .6), seed=j + 20)
        for g in (-1, 1): cel.fill(sx([(g * .70, 2.78), (g * .76, 2.74), (g * .82, 2.78), (g * .76, 2.82), (g * .70, 2.78)], 5), pal['s2'], .7)
        cel.unclip(); outline(cel, tor, lw * 1.1, ink, 4)
        skin = cel.layer()
        cel = Cel(); sway = .03 * math.sin(f * .33)
        for g in (-1, 1):
            inn = [(g * .58, 1.22), (g * .82, 1.86), (g * (.98 + sway), 2.5), (g * (1.02 + sway), 3.3), (g * (1.0 + sway * 2), 4.2), (g * (1.10 + sway * 3), 5.8)]; out = [(g * 3.0, 5.8), (g * 2.70, 3.6), (g * 2.36, 2.45), (g * 1.86, 1.86), (g * 1.30, 1.52), (g * .84, 1.22)]
            P = np.vstack([sx(inn), sx(out)]); cel.fill(P, pal['cl']); cel.clip(P)
            if g > 0: cel.fill(sx([(1.10, 1.5), (1.34, 2.2), (1.42, 3.2), (1.30, 4.4), (1.5, 5.8), (3.4, 5.8), (3.0, 1.5)]), pal['c1'])
            else: cel.fill(sx([(-1.86, 1.8), (-1.72, 2.5), (-1.90, 3.4), (-1.78, 4.4), (-2.0, 5.8), (-3.4, 5.8), (-3.0, 1.8)]), pal['c1'])
            for F in ([(1.02, 1.6), (1.30, 2.1), (1.36, 2.8)], [(1.84, 1.94), (1.62, 2.4), (1.60, 3.1)], [(2.1, 2.4), (1.9, 3.0), (1.95, 3.8)], [(1.12, 3.4), (1.34, 3.9), (1.40, 4.6)]): cel.line(sx([(g * a, b) for a, b in F]), lw * .55, ink, taper=(.15, .65), press=.4, seed=int(F[0][0] * 50))
            cel.unclip(); outline(cel, P, lw * 1.15, ink, 6 + g, press=.45, breaks=1)
            col = np.vstack([sx([(g * .50, 1.10), (g * .64, 1.50), (g * .84, 1.94), (g * .98, 2.42), (g * 1.02, 2.9)]), sx([(g * 1.22, 2.74), (g * 1.24, 2.3), (g * 1.10, 1.80), (g * .88, 1.40), (g * .62, 1.04)])]); shape(cel, col, pal['cl'] if g < 0 else pal['c1'], lw * .9, ink, 8 + g, press=.4)
            for q in (3.25, 3.9): c = X([(g * 1.14 + sway, q)])[0]; a = np.linspace(0, 6.283, 18); shape(cel, c + np.c_[np.cos(a) * .055 * S * .8, np.sin(a) * .055 * S], pal['sash'], lw * .55, ink, 3)
        shirt = cel.layer(); hc = X([HEART])[0]; d = np.hypot(self.gx - hc[0], self.gy - hc[1]) / S + .18 * np.sin(np.arctan2(self.gy - hc[1], self.gx - hc[0]) * 9 + 1)
        m = smooth(sh_r + .04, sh_r - .04, d)[..., None]; red = shirt[..., :3] * (np.array([.80, .135, .13], f32)); edge = np.exp(-((d - sh_r) / .07) ** 2)[..., None] * (0 < sh_r < 3.6)
        shirt = np.dstack([shirt[..., :3] * m + red * (1 - m) + shirt[..., 3:4] * edge * rgb('#ffb04a') * 1.1, shirt[..., 3]])
        return skin, shirt, hc, X

    def light(self, f, X, S, hc):
        """heart seen through the flesh (half res): ribs and heart in silhouette, gold core -> (through, rings) additive RGB"""
        h, w = HH // 2, W // 2; y, x = np.mgrid[0:h, 0:w].astype(f32); c = hc / 2; d = np.hypot(x - c[0], (y - c[1]) * 1.04) / (S / 2); p = pulse(f); on = .44 + .62 * min(1., p)
        I = on * (1.20 * np.exp(-(d / .34) ** 1.6) + .82 * np.exp(-(d / (.62 + .35 * p)) ** 1.4) + .25 * np.exp(-d / 1.1))
        m = np.zeros((h, w), np.uint8)
        for i in range(6):
            for g in (-1, 1):
                C = spl([(g * .10, 2.02 + .30 * i), (g * .50, 1.94 + .31 * i), (g * .98, 2.12 + .32 * i), (g * 1.34, 2.46 + .33 * i)], n=8); P = X(np.vstack([C - [0, .058], (C + [0, .058])[::-1]])) / 2
                cv2.fillPoly(m, [np.round(P * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        st = X(spl([(-.10, 1.9), (.10, 1.9), (.085, 2.9), (.0, 3.25), (-.085, 2.9)], True, n=6)) / 2; cv2.fillPoly(m, [np.round(st * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        rib = cv2.GaussianBlur(m, (0, 0), 2.2).astype(f32) / 255.; I = I * (1 - .62 * rib)
        cs = (1 - .20 * p + .06 * math.sin(f * .9)); sq = 1 + .10 * p
        hs = spl(np.array([(-.11, -.30), (-.02, -.23), (.09, -.33), (.17, -.22), (.27, -.16), (.31, .04), (.22, .26), (.03, .44), (-.13, .26), (-.24, .05), (-.23, -.16)]) * [cs * sq, cs], True, n=6); P = (hc + hs * S * .92) / 2
        m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [np.round(P * 8).astype(np.int32)], 255, cv2.LINE_AA, 3)
        for a, b, r in (((-.10, -.26), (-.22, -.62), .05), ((.07, -.28), (.10, -.66), .06), ((.16, -.20), (.36, -.48), .045)): cv2.line(m, tuple(int(v) for v in (hc + np.array(a) * S * .92 * cs) / 2), tuple(int(v) for v in (hc + np.array(b) * S * .92) / 2), 255, max(2, int(r * S)), cv2.LINE_AA)
        hm = cv2.GaussianBlur(m, (0, 0), 2.).astype(f32) / 255.; inner = cv2.GaussianBlur(hm, (0, 0), S * .045)
        I = I * (1 - .50 * hm) + hm * inner ** 2 * (.30 + .5 * p)
        ln = Cel(w, h, ss=1)                                             # the drum: lacing over the heart, hoop and tacks
        B = (hs * S * .92 / 2); n = len(B)
        for k in range(7):
            a, b = B[int(n * (.02 + k * .066)) % n], B[int(n * (.98 - k * .066)) % n]; a2 = B[int(n * (.02 + (k + 1) * .066)) % n]; ln.line([c + a * .96, c + b * .96], 1.5, '#ffffff', taper=(.1, .1)); ln.line([c + b * .96, c + a2 * .96], 1.2, '#ffffff', taper=(.1, .1))
        a = np.linspace(0, 6.283, 120); hoop = lambda r, sd: c + np.c_[np.cos(a), np.sin(a)] * (r * S / 2 * (1 + .010 * np.sin(5 * a + sd) + .006 * np.sin(11 * a + sd * 2)))[:, None]
        for r, ww in ((.60 * (1 + .05 * p), 2.6), (.655 * (1 + .06 * p), 1.2)): ln.line(hoop(r, 1.), ww, '#ffffff', taper=(0, 0), press=.5)
        for k in range(26): th = k / 26 * 6.283; q = c + np.array([math.cos(th), math.sin(th)]) * .627 * S / 2 * (1 + .055 * p); ln.fill(q + np.c_[np.cos(a[::8]), np.sin(a[::8])] * 2.2 * S / 360, '#ffffff')
        L0 = ln.rgba()[..., 3] * (.25 + .75 * min(1, p))
        col = ramp(np.clip(I, 0, 1.2) / 1.2, [(0, '#000000'), (.16, '#3c0504'), (.36, '#b12408'), (.56, '#f77a1a'), (.76, '#ffc75a'), (1, '#fff7dc')]) * np.clip(I * 1.6, 0, 1)[..., None]
        col = col + L0[..., None] * rgb('#ffe2a0') * .8 + cv2.GaussianBlur(L0, (0, 0), 4)[..., None] * rgb('#ff8a2a') * .6
        rg = Cel(w, h, ss=1)
        for i, (k, am) in enumerate(PULSE):
            age = f - k
            if 0 <= age < 15:
                r = .66 + .42 * age * (1 - .022 * age); al = am * (1 - age / 15.) ** 1.2
                rg.line(hoop(r, i + age * .3), (5.5 - .22 * age) * (.6 + .6 * am), '#ffffff', taper=(0, 0), press=.9, seed=i, alpha=al); rg.line(hoop(r * .955, i * 2.), 1.4, '#ffffff', taper=(0, 0), press=.8, alpha=al * .8)
                if age < 9:
                    for j in range(30): th = j / 30 * 6.283 + i; u = np.array([math.cos(th), math.sin(th)]); rg.line([c + u * r * S / 2 * 1.035, c + u * r * S / 2 * (1.075 + .07 * h1(i, j))], 2.0, '#ffffff', taper=(.1, .8), alpha=al)
        R_ = rg.rgba()[..., 3]; ring = R_[..., None] * rgb('#fff0c4') * .95 + cv2.GaussianBlur(R_, (0, 0), 5)[..., None] * rgb('#ffab3c') * .9 + cv2.GaussianBlur(R_, (0, 0), 22)[..., None] * rgb('#ff5a1e') * .9
        up = lambda q: cv2.resize(q, (W, HH), interpolation=cv2.INTER_LINEAR)
        return up(col.astype(f32)), up(ring.astype(f32)), p

    def frame(self, f):
        cx_, cy_, S = K(f, CAM); pf = pulse(f, 2.0); sh = np.array([math.sin(f * 5.1), math.cos(f * 4.3)]) * 5 * pf * (f < 34) + np.array([0., -7. * pf])
        if 36 <= f < 41: sh = sh + np.array([(-1) ** f * 7., 4 * (-1) ** (f // 2)]) * (41 - f) / 5
        S = S * (1 + .012 * pf); ox, oy = 960 - cx_ * S + sh[0], 540 - cy_ * S + sh[1]
        lau = (math.cos((f - 52) * math.pi) * .5 + .5) * .085 * (f >= 52) * (.6 + .4 * float(ss(52, 57, f)))                # shishishi: shoulders on twos
        fire_g = float(ss(12, 32, f)); amb = float(ss(10, 36, f))
        pitch, yaw, roll = [float(K(f, HP[k])) for k in ('pitch', 'yaw', 'roll')]
        hx, hy = ox + (3.5 * (-1) ** f if 44 <= f < 50 else 0.), oy - S * lau * .6 - S * .07 * float(ss(49.5, 52, f)); hy += S * .05 * (math.sin(min(1, max(0, (f - 28) / 5.)) * math.pi))
        lagT = []
        for k in range(8):                                                   # follow-through of the hair: where the head was k frames ago (world, camera removed)
            g = max(0, f - k); p_, r_ = float(K(g, HP['pitch'])), float(K(g, HP['roll'])); lagT.append(((r_ - roll) * S * -.9, (p_ - pitch) * S * -1.0))
        lagT = np.array(lagT)
        bl = G.bleach(f, 15, 37) if f < 37 else 1.; wb = 1. if f >= 26 else 0.; spir = float(back_out(np.clip((f - 28) / 10., 0, 1), 1.4))
        e = dict(eye=float(K(f, EYE)), iris=float(K(f, IRIS)), ring=float(np.clip((46 - f) / 8., 0, 1)) if f >= 37 else 0., brow=float(K(f, [(30, -.03), (35, -.06), (37, .13, 'o'), (41, .06), (48, .02), (51, .16, 'o'), (60, .10)])), tilt=0.)
        e['cheek'] = float(K(f, CHEEK)) + .05 * (-1) ** (f // 2) * (f >= 44); e['gaze'] = tuple(K(f, [(37, (0, 0)), (40, (0, 0)), (42, (-.30, .02), 'o'), (45, (-.30, .02)), (47, (.02, 0), 'o'), (60, (0, 0))]))
        if f < 38: e.update(mouth='line', mw=float(K(f, [(0, .15), (32, .15), (34, .13), (36, .20), (38, .27)])), smile=float(K(f, [(0, -.01), (32, -.01), (34, .02), (36, .075), (38, .12)])))
        elif f < 50:                                                      # the grin pulls itself open: 38 crack, 42 wide, trembles, 49 wound up
            gw = float(K(f, [(38, .24), (40, .42), (42, .68, 'o'), (44, .60), (47, .63), (49, .54)])) + .014 * (-1) ** f * (f > 43); op = float(K(f, [(38, .25), (40, .62), (42, 1.14, 'o'), (44, .96), (47, 1.04), (49, .80)])); e.update(mouth='grin', gw=gw, gc=-.50 + .145 * op, gt=-.60 + .05 * op, gb=-.66 - .19 * op, puff=float(np.clip(.02 + (gw - .5) * .62, 0, .2)), jaw=.07 * op)
        else:                                                             # head back, open laugh, jaw beating on twos
            mw = float(K(f, [(50, .46), (51, .66, 'o'), (53, .52), (60, .54)])); jw = float(K(f, [(50, .12), (51, .44), (52, .66, 'o'), (54, .34), (60, .34)])) + (.085 * (-1) ** (f // 2) if f > 53 else 0.); e.update(mouth='laugh', mw=mw, mc=-.40, mt=-.465, jaw=jw, puff=float(np.clip(.04 + (mw - .5) * .6, 0, .2)), iris=.62 if f < 55 else .78)
        brim = None if f >= 31 else float(K(f, [(0, -.30), (19, -.30), (26, .30), (30, .78), (37, .78)]))
        bu_ = BURST(f); size = (1. + .10 * pf * (f > 22)) * float(K(f, [(0, .94), (42, .94), (44, 1.0), (50, 1.0), (52, 1.12), (60, 1.10)])) * (1 + .30 * bu_)
        sway = math.sin(f * .21) * .22 - .28
        P = G.head_parts(hx, hy, S, yaw, pitch, roll, e, 'fire', f, flow=(sway * 1.7 - .25, -.85), fs=(.58 + .06 * math.sin(f * .31)) * (1 - .72 * min(1, bu_)), bl=bl, lagT=lagT, amp=1.0 + .7 * bu_, spir=max(0., spir), wb=wb, brim=brim, size=size, seed=1, socket=.55, shade_side=-1, hz=1.0, lw=S * .0115)
        T = Turn(ox, oy + S * lau, S, yaw=-yaw * .55, pitch=0., roll=.02 + roll * .12, persp=.10)
        skin, shirt, hc, X = self.body(T, S, f, float(K(f, [(0, -.5), (11, -.3), (30, 4.2)])))
        scarf = []
        if f > 36:                                                            # the cloud scarf condenses out of the fire
            g_ = float(back_out(np.clip((f - 36) / 11., 0, 1), 1.2)); scarf = G.scarf(T([(0, 1.50)])[0], np.array([1., 0.]), np.array([0., -1.]), S * 1.25, f, lagT, 950, g_)
        scl = G.paint_hair(scarf, P['pal'], S, lw=S * .0075, seed=5, close=.06, dim=.95) if scarf else None
        bg = self.bg(f, fire_g)
        # ---- light: the fire behind gives a hot rim on both sides, the heart lights flesh from inside
        w = S * .020; ru = fire_g
        dark = mix(np.array([.46, .27, .29], f32), np.array([1., 1., 1.], f32), amb)
        def cel(L, rimk=1., up=.35):
            L = L.copy(); L[..., :3] *= dark
            a = L[..., 3]; wr = w * rimk
            if ru > .02:
                L = G.hotrim(L, -1., .30, wr, '#fff0c0', '#ffb04e', ru); L = G.hotrim(L, .8, .25, wr * .8, '#ffc877', '#ff8030', .9 * ru)
            return L
        back, face, front, skin, shirt = cel(P['back'], 1.2), cel(P['face']), cel(P['front'], 1.1), cel(skin, .8), cel(shirt)
        lower = smooth(hy + S * .2, hy + S * 2.2, self.gy)[..., None]
        fig_b = lay(back, skin); 
        if scl is not None: scl = cel(scl)
        fig_b = lay(fig_b, shirt) if scl is None else lay(lay(fig_b, shirt), scl)
        through, ring, p = self.light(f, X, S, hc)
        fade = 1. if f < 34 else max(.0, 1 - (f - 34) / 6.)
        sa = skin[..., 3:4] * (1 - shirt[..., 3:4] * .72)
        fig_b[..., :3] += through * sa * fade * (1 - .0 * amb)
        # chin and jaw lit from below on every pulse
        kf = face.copy(); dy = np.clip((self.gy - (hy - S * .1)) / (S * 1.3), 0, 1) ** 1.6; upl = (dy * min(1.2, .25 + p) * fade)[..., None] * (1 - .55 * amb)
        kf[..., :3] += face[..., 3:4] * upl * rgb('#ff7a24') * .62; fr = front.copy(); fr[..., :3] += front[..., 3:4] * upl * rgb('#ff6a1c') * .25
        fig = lay(lay(fig_b, kf), fr)
        # gradient over the whole cel: warm from below, red-violet at the crown
        fig = para(fig, (960, hy - S * 1.6), (960, hy + S * 1.5), [(0, mix(rgb('#e8a9a2'), np.ones(3, f32), .35)), (.45, '#ffffff'), (1, '#ffd9b4')], amt=.75 * amb)
        img = bg * (1 - .35 * (1 - fire_g))
        if f < 34: img = img + (radial(hc[0], hc[1], S * 2.4, power=2.2) * .16 * (.3 + p) * fade)[..., None] * rgb('#ff6a24')
        img = screen(img, blur(fig[..., 3], 34)[..., None] * rgb('#ff6a26') * .34 * fire_g)
        img = over(img, fig)
        img = img + ring * (1. if f < 34 else .34)
        # burn line climbing the locks
        if P['glow']:
            gl = Cel(W // 2, HH // 2, ss=1); 
            for q, ww in P['glow']: gl.line(q / 2, max(1.6, ww * .05), '#ffffff', taper=(.1, .1), seed=3)
            m = gl.rgba()[..., 3]; g2 = m[..., None] * rgb('#fff2c0') * 1.1 + cv2.GaussianBlur(m, (0, 0), 3.5)[..., None] * rgb('#ff9a30') * 1.3 + cv2.GaussianBlur(m, (0, 0), 14)[..., None] * rgb('#ff4a18') * 1.6
            img = img + cv2.resize(g2.astype(f32), (W, HH))
        if fire_g > .2: img = put(img, embers(f, n=int(46 * fire_g), seed=4, wind=(30., -260.), size=1.25, swirl=60.))
        if self._near is not None:
            m = (np.clip(np.abs(self.gx - 960) / 960 - .45, 0, 1) * np.clip((self.gy - 420) / 660, 0, 1) * 1.6)[..., None] * fire_g; img = over(img, self._near * np.minimum(m, 1), .85)
        if 37 <= f < 40: img = img + (radial(960, 540, 1300, power=1.2) * (.16 - .05 * (f - 37)))[..., None] * rgb('#ff5a3a')
        if f < 4: k = [.94, .62, .30, .10][f]; img = img * (1 - k) + rgb('#fff6df') * k
        return img.astype(f32)
