"""f2 60: Nika dances before a sun that fills the sky; the ground is a drum skin, roofs and rocks leap on the beat, a mountain of an elephant turns its head."""
import numpy as np, cv2, math, os
from op import W, H as HH
from op.paint import rgb, f32, smooth, ramp, grid, fbm, mix, repaint, mottle
from op.ink import Cel, spl, resample
from op.kit import outline, shape, hatch, para, speed_lines
from op.comp import over, screen, blur, radial, ease, cam2d
from op.clouds import Sky
from op.fx_core import h1, lin01, noise, c2, put
from op.fx_break import rock
from op.fx_fire import embers
from chars import gear5 as G
from chars.gear5 import ss, lay, back_out, rim, hotrim, inked, lock, paint_hair
from shots.g5kit import Keys, Rub, PROPS, SUNP

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
CX, GY, RX, RY = 960., 946., 1230., 214.                                           # drum head
U = 138.
BEATS = (0, 15, 30, 45)
S_ = lambda d: {k: (-v if np.isscalar(v) else (-v[0], v[1])) for k, v in d.items()}
K = Keys(dict(
    py={0: .98, 2: .78, 3.5: .92, 6: 2.15, 8: 2.80, 10: 2.95, 12.5: 2.45, 15: 1.62, 17: 1.16, 19: 1.62, 21: 2.45, 24: 2.90, 26: 2.85, 28: 2.35, 30: 1.70, 31.5: 1.46, 33.5: 1.52, 35: 1.85, 37: 2.55, 39.5: 2.90, 42: 2.20, 45: 1.18, 46.5: .84, 48.5: .98, 51: 2.25, 54: 2.95, 59: 3.40},
    px={0: 0, 6: .02, 9: -.12, 13: -.22, 15: -.32, 17: -.42, 20: -.30, 24: .10, 28: .28, 30: .30, 33: .38, 37: .12, 40: -.14, 45: 0, 59: .05},
    sq={0: .74, 2: .58, 3.5: .70, 6: 1.36, 8: 1.16, 10: 1.02, 13: 1.12, 15: .84, 17: .62, 19: 1.0, 21: 1.34, 24: 1.06, 28: 1.12, 30: .90, 31.5: .80, 33.5: .88, 35: 1.06, 37: 1.32, 40: 1.04, 43: 1.14, 45: .74, 46.5: .56, 48.5: .72, 51: 1.40, 54: 1.16, 59: 1.06},
    lean={0: 0, 2: .05, 6: -.06, 9: .20, 13: .26, 15: .34, 17: .48, 19: .34, 22: -.10, 25: -.32, 28: -.30, 30: -.32, 33: -.44, 35: -.3, 38: .16, 40: .30, 43: .10, 45: 0, 59: 0},
    roll={0: 0, 5: 0, 8: -.22, 13: .06, 15: .16, 17.5: .40, 21: .10, 25: -.30, 28: -.16, 30: -.18, 33: -.36, 36: -.06, 40: .36, 43: .10, 45: 0, 48: .06, 52: -.10, 55: .14, 59: -.10},
    yaw={0: 0, 8: -.22, 15: .28, 22: .1, 26: -.36, 30: -.26, 37: .1, 40: .36, 45: 0, 59: .08}, pitch={0: -.40, 3: -.46, 6: .18, 10: .26, 15: -.06, 17: -.22, 22: .24, 30: -.04, 32: -.14, 37: .26, 43: .05, 45: -.24, 47: -.34, 50: .05, 54: .34, 59: .28},
    st={0: 0, 8: -.35, 13: .40, 15: .62, 17: .70, 21: .10, 25: -.45, 28: -.62, 30: -.70, 33.5: -.62, 38: .20, 41: .55, 45: 0, 59: 0}, ht={0: 0, 8: .30, 13: -.25, 15: -.60, 17: -.70, 21: 0, 25: .40, 28: .55, 30: .70, 33.5: .66, 38: -.2, 41: -.55, 45: 0, 59: 0},
    sink={0: .9, 2: 1.15, 4: .6, 6: -.25, 9: 0, 15: .35, 17: 1.1, 19: .25, 21: -.25, 25: 0, 30: .25, 31.5: .6, 34: .3, 36: -.2, 40: 0, 45: .8, 46.5: 1.2, 48.5: .7, 51: -.3, 55: 0, 59: 0},
    hA={0: (-.72, .30), 5: (-.74, .26), 8: (-1.55, 1.25), 10.5: (-1.65, 3.5), 13: (-.95, 5.1), 15: (-1.60, 4.98), 17: (-1.88, 4.45), 19: (-1.62, 4.25), 22: (-1.95, 3.0), 24.5: (-1.05, 1.6), 27: (-1.95, 1.8), 30: (-2.10, 2.32), 32: (-2.0, 1.95), 34: (-2.04, 2.05), 37: (-1.5, 1.25), 40: (-2.25, 3.7), 42: (-1.4, 4.5), 43.5: (-.95, 3.2), 45: (-.42, 1.50), 47: (-.54, 1.08), 49: (-.95, 1.5), 52: (-1.35, 1.5), 55: (-1.75, 3.8), 57: (-2.2, 5.2), 59: (-2.45, 5.6)},
    hB={0: (.72, .30), 5: (.74, .26), 8: (1.55, 1.25), 10.5: (2.0, 2.9), 13: (2.2, 2.15), 15: (2.10, 2.10), 17: (1.95, 1.82), 19: (2.1, 2.3), 22: (1.45, 1.15), 24.5: (2.3, 3.5), 27: (1.2, 5.15), 30: (1.68, 4.92), 32: (1.88, 4.5), 34: (1.74, 4.6), 37: (2.0, 3.1), 40: (1.45, 1.4), 42: (1.7, 3.2), 43.5: (.95, 3.0), 45: (.42, 1.50), 47: (.54, 1.08), 49: (.95, 1.5), 52: (1.35, 1.5), 55: (1.75, 3.8), 57: (2.2, 5.2), 59: (2.45, 5.6)},
    fA={0: (-1.08, 0), 5.6: (-.82, 0), 7: (-.55, .45), 9: (-.22, 1.5), 11: (-.35, 1.35), 13: (-.5, .55), 15: (-.56, 0), 19.6: (-.56, 0), 21: (-.5, .35), 23: (-.95, 1.2), 25: (-1.50, 1.95), 28: (-1.65, 2.0), 30: (-1.54, 1.74), 31.5: (-1.50, 1.56), 33.5: (-1.52, 1.62), 36: (-1.2, 1.3), 39: (-.50, .8), 42: (-.9, .4), 45: (-1.18, 0), 49.6: (-1.12, 0), 51: (-.75, .35), 53: (-.42, 1.5), 55: (-.7, 1.9), 57: (-1.4, 1.95), 59: (-1.85, 2.2)},
    fB={0: (1.08, 0), 5.6: (.82, 0), 7: (.75, .6), 9: (1.2, 1.9), 11: (1.45, 2.05), 13: (1.55, 1.92), 15: (1.54, 1.74), 17: (1.46, 1.48), 19: (1.50, 1.62), 21.5: (1.2, 1.3), 24: (.55, .75), 27: (.55, .35), 30: (.56, 0), 34.6: (.56, 0), 36.5: (.5, .4), 38.5: (1.25, 1.4), 40.5: (1.8, 1.6), 42.5: (1.35, .6), 45: (1.18, 0), 49.6: (1.12, 0), 51: (.75, .35), 53: (.42, 1.5), 55: (.7, 1.9), 57: (1.4, 1.95), 59: (1.85, 2.2)}))
FACE = [(0, 'fierce'), (6, 'roar'), (12, 'joy'), (15, 'arcs'), (22, 'wink'), (30, 'teeth'), (36, 'cres'), (45, 'arcs'), (51, 'crazy')]


def pose(f):
    """every channel on its own spline; hips and shoulders counter-rotate, the head follows 1.4 f late, hands whip in two frames after the body"""
    g = lambda k, d=0.: K(k, max(0, f - d)); px, py, sq = g('px'), g('py'), g('sq'); ln = g('lean', .6); tl = 1.04 * (.52 + .48 * sq) if sq < 1 else 1.04 * sq ** .55
    hA, hB, fA, fB = K('hA', f), K('hB', f), K('fA', f), K('fB', f); che = np.array([px + ln, py + tl]); rl = g('roll', 1.4); sk = g('sink', .8)
    hd = che + np.array([ln * .55 + math.sin(rl) * .32, .98 - .34 * sk]); hd[1] += .055 * math.sin(f * 1.7) * (f > 6)
    ex = [e for k, e in FACE if f >= k][-1]; hit = min([f - b for b in BEATS if f >= b]); jaw0 = G.EX[ex].get('jaw', 0.); E = {} if ex in ('fierce', 'teeth') else dict(jaw=max(.06, jaw0 + .10 * math.sin(f * 2.1) + .16 * float(np.clip(g('sq', 1.4) - 1, 0, .4))))
    if 6 <= f: E['eye'] = G.EX[ex].get('eye', 1.)
    air = float(ss(.1, .9, min(fA[1], fB[1]))); vA, vB = [float(np.hypot(*(K(k, f + .5) - K(k, f - .5)))) for k in ('hA', 'hB')]
    spir = lambda v: float(np.clip((v - .28) / .50, 0, 1)) * float(ss(5.5, 8, f)); kn = .34 + .52 * max(0., 1.75 - py) + .28 * air
    rip = hit / 8. if hit <= 8 else None
    return dict(arms='back' if 16 <= f < 20 else 'front', pel=(px, py), che=che, hd=hd, roll=rl * .8, yaw=g('yaw', 1.5), pitch=g('pitch', 1.2), hA=hA, hB=hB, bA=.34 + .10 * math.sin(f * .7), bB=.34 + .10 * math.sin(f * .7 + 1), fA=fA, fB=fB, kA=kn, kB=kn, back=(-ln * .95 + .16 * math.sin(f * .42), 0), st=g('st'), ht=g('ht', -.6), sink=max(0, sk),
                spA=spir(vA), spB=spir(vB), rip=rip, aA=math.pi - .75 * float(np.clip(fA[1], 0, 1)), aB=.75 * float(np.clip(fB[1], 0, 1)), kA_h='fist' if f < 5 else 'flat' if 43 <= f <= 48 else 'open', kB_h='fist' if f < 5 else 'flat' if 43 <= f <= 48 else 'open',
                sq=float(np.clip(.35 + .65 * g('sq', 1.2), .68, 1.24)), belly=float(ss(40, 45, f) * (1 - ss(49, 54, f))) * (1 + .12 * math.sin(f * 2.4) * (f >= 45)), expr=ex, E=E, amp=1.15, hdA=2.2 if f < 5 else None, hdB=.94 if f < 5 else None)


DEP = Keys(dict(g={0: 1.0, 15: 1.0, 19: 1.0, 25: 1.14, 30: 1.25, 34: 1.25, 38: 1.14, 45: 1.02, 49: 1.02, 54: 1.20, 59: 1.46}))
def depth(f): return DEP('g', f)
def foot_y(f): return GY - 22 + (depth(f) - 1) * 190
_FT = {}


def feet(f):
    """[(x, plane y, press 0..1)] of everything that pushes on the skin now"""
    k = int(round(f * 4))
    if k in _FT: return _FT[k]
    p = pose(max(0, f)); g = depth(f); out = []
    for q in ('fA', 'fB') + (('hA', 'hB') if f < 5 else ()):
        h = p[q][1] - (.28 if q[0] == 'h' else 0)
        if h < .10:
            age = 0
            while age < 12 and f - age > 0 and pose(f - age - 1)[q][1] - (.28 if q[0] == 'h' else 0) < .10: age += 1
            out.append((960 + p[q][0] * U * g, foot_y(f) + 8, (1 - h / .10) * (.34 + .66 * math.exp(-age / 3.2))))
    _FT[k] = out; return out


def skin_wave(r, f):
    """vertical travel of the drum skin (px, + = down) at radius r 0..1"""
    z = 0.
    for b in BEATS:
        a = f - b
        if 0 <= a < 15: z = z + (1 - r * r) * 40 * math.exp(-a / 3.8) * math.cos(a * .95)
        if 0 <= a < 19: z = z - 20 * np.exp(-((r - a * RV) / .045) ** 2) * math.exp(-a / 12.) * np.clip(a / 2., 0, 1)
    return z


RV = .052                                                                                      # ring speed: a stomp reaches the houses exactly one beat later and kicks them


def G_(r, th, f):
    r = np.asarray(r, float); x = CX + RX * r * np.cos(th); y = GY + RY * r * np.sin(th); z = skin_wave(r, f)
    for fx, fy, w in feet(f): z = z + 40 * w * np.exp(-(((x - fx) / 150.) ** 2 + ((y - fy) / 34.) ** 2))
    return c2(x, y + z)


OBJ = [                                                                                        # kind, radius, azimuth deg, size, period (beats), height px, phase beat, spin
    ('pagoda', .80, -128, .78, 2, 400, 0, .16), ('house', .74, -158, .80, 1, 170, 0, -.10), ('torii', .84, -112, .74, 2, 330, 1, -.14), ('hut', .86, -70, .62, 1, 190, 0, .12), ('house', .78, -34, .84, 1, 190, 0, .10), ('pine', .90, -76, .62, 1, 150, 0, .2),
    ('rock', .60, -172, .62, 1, 250, 0, 1.2), ('lantern', .55, -24, .62, 1, 230, 0, .5), ('barrel', .42, -150, .58, 1, 260, 0, -.8), ('rock', .52, -2, .80, 2, 470, 0, -1.4), ('pine', .95, -145, .70, 1, 160, 0, -.2), ('hut', .92, -20, .72, 2, 360, 0, -.16),
    ('rock', .66, 160, .95, 2, 430, 1, .9), ('barrel', .40, -22, .56, 1, 290, 0, .9), ('rock', .60, 30, 1.0, 1, 250, 0, -1.1), ('house', 1.0, 172, 1.30, 2, 300, 1, .12), ('torii', 1.02, 9, 1.35, 2, 330, 0, -.12)]


class Shot:
    id = 'f2'; n = 60; finish = dict(bloom=.34, vig=.14)
    EVENTS = [(0, 'drum', dict(size=1.0, note='both fists on the drum-skin ground')), (5, 'whoosh', dict(size=.5, note='springs up')), (15, 'drum', dict(size=1.0, note='lands on the left foot; houses, rocks land and bounce')), (20, 'wood', dict(size=.4, note='roofs rattle')), (24, 'creak', dict(size=.5, note='elephant turns, very low and far')), (30, 'drum', dict(size=1.0, note='lands on the right foot')),
              (37, 'whoosh', dict(size=.4)), (45, 'drum', dict(size=1.0, note='both palms slap the drum belly: boing skin')), (50, 'drum', dict(size=.5, note='belly rebound')), (51, 'whoosh', dict(size=.7, note='leaps to lens')), (8, 'laugh', dict(size=1.0, note='laughs to the end'))]

    def setup(self):
        self.gx, self.gy = grid(); p = os.path.join(ROOT, 'cache/f2_sky3.npy')
        if os.path.exists(p): self.sky = np.load(p).astype(f32)
        else: self.sky = self.paint_sky(); np.save(p, self.sky.astype(np.float16))
        p = os.path.join(ROOT, 'cache/f2_bank2.npy')
        if os.path.exists(p): self.bank = np.load(p).astype(f32)
        else:
            s = Sky(HH, W + 120, '#b06aa0', (0., -.8, .5), 3, layered=True); T = ('#ffe9bc', '#f7a98a', '#b5699f', '#6b4393')
            for x, b, w_, h_, sd in ((120, 800, 760, 250, 2), (1780, 815, 900, 300, 5), (560, 810, 620, 120, 7), (1330, 812, 640, 130, 9), (960, 830, 900, 90, 11), (300, 850, 900, 120, 13), (1700, 858, 900, 120, 15)): s.bank(x, b, w_, h_, T, '#e39a9c', sd, pool=.35, foot=.3)
            L = s.layer(s.paint(2)); self.bank = L.astype(f32); np.save(p, self.bank.astype(np.float16))
        self.skin = fbm(HH, W, (160, 60, 22, 8), None, 12, aspect=3.4); self.fib = fbm(HH, W, (9, 4), (1, .6), 15, aspect=9.)
        self.ang = np.arctan2(self.gy - 486, self.gx - 960); self.rad = np.hypot(self.gx - 960, self.gy - 486)

    def paint_sky(self):
        x, y = grid(); r = np.hypot(x - 960, y - 486); a = np.arctan2(y - 486, x - 960); n = fbm(HH, W, (220, 80, 30), None, 3)
        sky = ramp(np.clip((r - 500) / 900 + .05 * n, 0, 1), [(0, '#ffb45a'), (.10, '#f57f63'), (.30, '#c9589a'), (.62, '#7a3f9c'), (1, '#3f2a7c')])
        w = np.sin(a * 22 + .5 * n + 3 * np.exp(-r / 900)) * .5 + .5; sky = sky + (smooth(.45, .75, w) * np.clip(1.2 - r / 1300, 0, 1))[..., None] * rgb('#ff9a5c') * .16
        d = ramp(np.clip(r / 505 + .035 * n, 0, 1), [(0, '#ffd474'), (.30, '#ffb84a'), (.62, '#fd9534'), (.86, '#f5762d'), (1, '#e8562a')])
        for k, rr in enumerate((190, 300, 400)): d = d + (np.exp(-((r - rr - 14 * n) / (10 + 4 * k)) ** 2))[..., None] * rgb('#fff0b0') * (.10 - .02 * k)
        sg = repaint(sky, (40, 18, 9), seed=4, angle=a + .15 * n, density=1.0, aspect=6., jit=(.012, .03, .03)); dg = repaint(d, (34, 15, 8), seed=5, angle=a + math.pi / 2 + .3 * n, density=1.0, aspect=5., jit=(.01, .02, .022))
        m = smooth(507, 503.5, r + 3.2 * fbm(HH, W, (60, 20), (1, .5), 8))[..., None]; edge = np.exp(-((r - 498) / 5.) ** 2)[..., None]
        out = sg * (1 - m) + dg * m + edge * rgb('#ffe9a0') * .30
        return out.astype(f32)

    # ------------------------------------------------------------------ the elephant: one 3/4 silhouette, hazy, bigger than the sun's edge
    def elephant(self, f, dx, dy):
        cel = Cel(); hz = dict(b='#9c74b0', s='#80599f', d='#6d4b92', ink='#6a5096'); lw = 2.3; X = lambda P: np.asarray(P, float) + [dx, dy]
        t = float(ease(np.clip((f - 15) / 30., 0, 1))) + (.05 * math.sin((f - 45) * .62) * math.exp(-(f - 45) / 5.) if f > 45 else 0.); tl = float(ease(np.clip((f - 19) / 34., 0, 1))); a = np.linspace(0, 6.283, 40)

        def leg(x, w_, sh):
            kn = 700; C = spl([(x, 500), (x - 8, 600), (x + 10, kn), (x - 4, 800), (x + 2, 980)], n=10); m = len(C); v = np.linspace(0, 1, m); hw = w_ * (1.15 - .62 * ss(0, .42, v) + .20 * np.exp(-((v - .42) / .07) ** 2) + .16 * v); return np.vstack([C - c2(hw, 0 * hw), (C + c2(hw, 0 * hw))[::-1]]), C, hw
        for x, w_ in ((1720, 60), (2010, 64)):
            P, C, hw = leg(x, w_, 1); inked(cel, X(P), hz['s'], lw, hz['ink']); cel.clip(X(P)); cel.fill(X(P + [w_ * .8, 0]), hz['d']); cel.unclip()
        body = spl([(1500, 330), (1540, 215), (1640, 128), (1800, 78), (1990, 66), (2160, 110), (2160, 660), (1980, 612), (1800, 622), (1650, 600), (1560, 548), (1500, 450)], True, n=10)
        near = [leg(1612, 72, 0), leg(1880, 76, 0)]
        piv = np.array([1585., 350.]); rot = -.26 + .34 * t; cs, sn = math.cos(rot), math.sin(rot); sq_ = 1 - .16 * t; off = np.array([-34., -16.]) * t
        def Hd(P, k=1.): P = np.asarray(P, float).copy(); P[:, 0] = np.where(P[:, 0] < -90, -90 + (P[:, 0] + 90) * sq_, P[:, 0]); return c2(P[:, 0] * cs - P[:, 1] * sn, P[:, 0] * sn + P[:, 1] * cs) + piv + off
        head = Hd(spl([(96, -150), (10, -200), (-70, -212), (-118, -196), (-172, -206), (-232, -168), (-262, -96), (-280, -10), (-282, 70), (-230, 128), (-180, 172), (-110, 178), (-40, 150), (60, 150), (130, 120)], True, n=9))
        r0 = Hd([(-236., 10.)])[0]; bd = (t - tl) * 230 + 10 * math.sin(f * .13); sw = 16 * math.sin(f * .15); C = spl([r0 + [40, -80], r0 + [4, 0], r0 + [-30, 110], r0 + [-22 + bd * .4, 230], r0 + [26 + bd * .9, 340], r0 + [30 + bd * 1.3 + sw, 430], r0 + [-26 + bd * 1.6 + sw * 1.6, 486], r0 + [-74 + bd * 1.7 + sw * 2, 478]], n=9)
        m = len(C); v = np.linspace(0, 1, m); hw = (84 * (1 - .80 * v ** .72)) * (.55 + .45 * ss(0, .10, v)) + 6; T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = c2(-T[:, 1], T[:, 0]); trunk = np.vstack([C + N * hw[:, None], (C - N * hw[:, None])[::-1]])
        for dx_, al in ((-26, .0), (0, 1.)):                                                   # tusks: far one peeks out as the head turns
            if al == 0 and t < .25: continue
            o = Hd([(-196. + dx_ * t * 1.6, 112.)])[0]; tk = np.vstack([spl([o + [-18, -16], o + [-62, 62], o + [-128, 96], o + [-188, 66]], n=7), spl([o + [-188, 66], o + [-120, 66], o + [-70, 30], o + [16, -8]], n=7)])
            if al == 0: inked(cel, X(tk), '#d9b6bd', lw, hz['ink'])
            else: tk_ = tk
        U_ = [X(body), X(near[0][0]), X(near[1][0]), X(head), X(trunk)]; inked(cel, U_, hz['b'], lw * 1.25, hz['ink'])
        cel.clip(U_[:3]); cel.fill(X(spl([(1520, 640), (1580, 500), (1740, 430), (1960, 400), (2200, 420), (2200, 760), (1520, 760)], True, n=8)), hz['s'])
        for P, C_, h_ in near: cel.fill(X(P + [h_[0] * .62, 0]), hz['s'])
        cel.unclip(); cel.clip(U_[3]); cel.fill(X(head + [86, 92]), hz['s']); cel.unclip(); cel.clip(U_[4]); cel.fill(X(np.vstack([C - N * (hw * .05)[:, None], (C - N * (hw * 1.1)[:, None])[::-1]])), hz['s']); cel.unclip()
        for k in range(m // 4, m - 3, 3): cel.line(X(spl([C[k] + N[k] * hw[k] * .92, C[k] + T[k] * 7, C[k] - N[k] * hw[k] * .92], n=4)), 1.3, hz['ink'], taper=(.2, .3), seed=k)
        inked(cel, X(tk_), '#f0d6cc', lw, hz['ink'])
        fl = 1 - .22 * t + .07 * math.sin(f * .17) + .10 * math.sin(max(0, f - 40) * .5) * math.exp(-max(0, f - 40) / 7.)       # near ear
        ep = np.array([(-50, -128), (14, -168), (110, -150), (186, -70), (190, 50), (146, 160), (84, 232), (34, 206), (0, 120), (-34, 50), (-56, -40)], float); ep[:, 0] = -50 + (ep[:, 0] + 50) * fl; ep[:, 1] += 12 * (1 - fl) * 4 * np.clip(ep[:, 1] / 200, 0, 1)
        E = X(Hd(spl(ep, True, n=8))); inked(cel, E, hz['b'], lw, hz['ink']); cel.clip(E); cel.fill(E + [40, 46], hz['s']); cel.fill(E + [96, 104], hz['d']); cel.unclip()
        for q in (.25, .5, .75): cel.line(X(Hd(spl([(-30, -60 + 60 * q), (40 * fl, -40 + 110 * q), (110 * fl, -20 + 200 * q)], n=6))), 1.4, hz['ink'], taper=(.2, .6), seed=int(q * 9))
        e0 = Hd([(-168. + 34 * t, -62. + 4 * t)])[0]; ew = 24 + 5 * t; cel.fill(X(e0 + c2(np.cos(a) * ew * 1.5, np.sin(a) * 24)), hz['s']); inked(cel, X(e0 + c2(np.cos(a) * ew, np.sin(a) * 12)), '#ffe9bc', lw * .8, hz['ink']); cel.fill(X(e0 + [-9 + 7 * t, 1] + c2(np.cos(a) * 8, np.sin(a) * 10)), '#4a2860'); cel.fill(X(e0 + [-12 + 7 * t, -3] + c2(np.cos(a) * 2.4, np.sin(a) * 2.4)), '#fff6e0')
        lid = spl([(-1.25 * ew, 0), (0, -17), (1.25 * ew, -2)], n=6); cel.fill(X(e0 + np.vstack([lid, (lid + [0, -14])[::-1]])), hz['s']); cel.line(X(e0 + lid), 3., hz['ink'], taper=(.2, .2))
        for j in range(3): cel.line(X(e0 + spl([(-1.1 * ew, 12 + 8 * j), (0, 18 + 8 * j), (ew, 11 + 8 * j)], n=4)), 1.4, hz['ink'], taper=(.3, .4))
        for j in range(4): cel.line(X(Hd(spl([(-206 + 10 * j, -150 + 22 * j), (-150 + 8 * j, -158 + 22 * j), (-110, -146 + 20 * j)], n=5))), 1.3, hz['ink'], taper=(.3, .5), seed=j)
        for k in range(8): cel.line(X(spl([(1660 + 52 * k, 170 + 9 * (k % 3)), (1668 + 52 * k, 300), (1654 + 52 * k, 420)], n=6)), 1.3, hz['ink'], taper=(.4, .6), seed=k)
        for P, C_, h_ in near:
            for j in range(5): q = C_[10 + 4 * j]; w_ = h_[10 + 4 * j]; cel.line(X(spl([q - [w_ * .7, 0], q + [0, 7], q + [w_ * .5, -2]], n=4)), 1.3, hz['ink'], taper=(.3, .5), seed=j)
        yb = lambda x: np.interp(x, [1500, 1540, 1640, 1800, 1990, 2160], [330, 215, 128, 78, 66, 110])
        for k in range(24): x = 1600 + 22 * k; r = 9 + 12 * h1(k, 3); cel.fill(X(c2(x + np.cos(a) * r, yb(x) - 3 + np.sin(a) * r * .9)), hz['s'] if k % 3 else hz['d'])
        for x, h_, w_ in ((1708, 46, 9), (1742, 74, 12), (1776, 56, 10), (1830, 40, 11), (1868, 60, 10)): y = yb(x); cel.fill(X([(x - w_, y), (x - w_ * .7, y - h_), (x, y - h_ - 18), (x + w_ * .7, y - h_), (x + w_, y)]), hz['d'])
        L = cel.layer(); L = hotrim(L, .95, .22, 3.4, '#ffe6ac', '#f6a27a', 1.)
        al = L[..., 3:4]; nearsun = np.exp(-((self.gx - 960) / 560.) ** 2)[..., None]; L[..., :3] = L[..., :3] * .64 + al * rgb('#dd97b0') * .36; L[..., :3] += al * nearsun * rgb('#ff9a4c') * .26
        mist = smooth(610, 830, self.gy - dy)[..., None] * al; L[..., :3] = L[..., :3] * (1 - mist * .95) + rgb('#f0b0a6') * mist * .95; return L

    def wisps(self, f):
        """thin cloud drifting across the endless legs"""
        c = Cel(W // 2, HH // 2, ss=1); a = np.linspace(0, 6.283, 60)
        for i, (x, y, w_, h_) in enumerate(((1700, 600, 250, 15), (1480, 668, 200, 18), (1900, 690, 270, 22), (1620, 738, 300, 24), (1360, 752, 220, 18))): q = 1 + .22 * np.abs(np.sin(a * 5 + i)); c.fill(c2((x - f * (.5 + .2 * i) + np.cos(a) * w_) / 2, (y + np.sin(a) * h_ * q) / 2), '#ffd9b8' if i % 2 else '#f6bda6', .62)
        return cv2.resize(cv2.GaussianBlur(c.rgba(), (0, 0), 3.2), (W, HH), interpolation=cv2.INTER_LINEAR)

    # ------------------------------------------------------------------ the drum
    def drum(self, f, oy):
        cel = Cel(); a = np.linspace(0, 6.283, 220); E = lambda r: G_(np.full(220, r), a, f) + [0, oy]
        rim_o = E(1.045); cel.fill(rim_o, '#4a2440'); cel.fill(E(1.02), '#7c3a44'); cel.fill(E(1.0), '#e9c79a')
        cel.fill(E(.985), '#e3bb8e'); cel.fill(E(.93), '#efcfa2'); cel.fill(E(.91), '#e7c192')
        hub = lambda r, th: G_(r, th, f) + [0, oy]                                          # three-comma tomoe, worn vermilion
        for k in range(3):
            o = k * 2.0944 + .5; v = np.linspace(0, 1, 60); th = o + v * 2.5; rr = .40 + .20 * v; w_ = .115 * (1 - v) ** .75; h_ = np.linspace(math.pi, 2 * math.pi, 24); head_c = np.array([.40 * math.cos(o), .40 * math.sin(o)])
            hc = head_c + .118 * np.c_[np.cos(np.linspace(0, 6.283, 40)), np.sin(np.linspace(0, 6.283, 40))]; pol = lambda Q: hub(np.hypot(Q[:, 0], Q[:, 1]), np.arctan2(Q[:, 1], Q[:, 0])); tl = np.vstack([np.c_[np.cos(th), np.sin(th)] * (rr + w_)[:, None], (np.c_[np.cos(th), np.sin(th)] * (rr - w_)[:, None])[::-1]])
            cel.fill(pol(hc), '#c8503f', .82); cel.fill(pol(tl), '#c8503f', .82)
        cel.clip(E(1.0))
        for fx, fy, w in feet(f):                                                            # dimples: dark near lip, lit far slope, stretched contour lines
            fy = fy + oy + 30 * w + float(skin_wave(0., f)) * .0; e = lambda kx, ky, sx=0, sy=0: c2(fx + sx + np.cos(a) * kx, fy + sy + np.sin(a) * ky)
            cel.fill(e(150, 36), '#b5787a', .34 * w); cel.fill(e(118, 27, 0, 5), '#8a4c66', .42 * w); cel.fill(e(96, 19, 0, -9), '#ffe6b4', .50 * w)
            for q, (kx, ky) in enumerate(((168, 41), (136, 32), (104, 23))): cel.line(e(kx, ky), 2.6 - .5 * q, '#7a3f5c', taper=(0, 0), press=.6, seed=q, alpha=min(1., 1.1 * w)); cel.line(e(kx * 1.03, ky * 1.05, 0, -4), 1.6, '#fff3cf', taper=(0, 0), press=.6, seed=q + 5, alpha=.8 * w)
        for b in BEATS:                                                                      # the ring wave runs out to the houses: lit crest, dark trough
            g = f - b
            if 0 < g < 19:
                al = math.exp(-g / 12.) * min(1., g / 2.); r = RV * g
                if r > .985: continue
                wv = .022 + .010 * g / 15.; cel.fill(np.vstack([E(r + wv), E(max(.004, r))[::-1]]), '#a35a68', .46 * al); cel.fill(np.vstack([E(r), E(max(.003, r - wv * 1.2))[::-1]]), '#fff0c4', .74 * al)
                cel.line(E(r), 2.6, '#7a3f5c', taper=(0, 0), press=.6, seed=b, alpha=.8 * al); r2 = max(.01, r - .075)
                cel.line(E(r2), 2.2, '#a35a68', taper=(0, 0), press=.6, seed=b + 1, alpha=.6 * al); cel.line(E(max(.008, r2 - .012)), 2.2, '#fff0c4', taper=(0, 0), press=.6, seed=b + 2, alpha=.7 * al)
        cel.unclip()
        for k in range(64):                                                                  # bronze tacks round the edge
            th = k / 64 * 6.283; q = hub(np.array([.958]), np.array([th]))[0]; s_ = 7 + 6.5 * (math.sin(th) * .5 + .5); e = np.linspace(0, 6.283, 14); inked(cel, c2(q[0] + np.cos(e) * s_ * 1.2, q[1] + np.sin(e) * s_ * .55), '#b5783a', 1.4, '#3a1c2c'); cel.fill(c2(q[0] - s_ * .3 + np.cos(e) * s_ * .4, q[1] - s_ * .2 + np.sin(e) * s_ * .16), '#ffdf9a')
        cel.line(E(1.0), 2.6, '#3a1c2c', taper=(0, 0)); cel.line(E(1.045), 3., '#2a1226', taper=(0, 0))
        L = cel.layer(); al = L[..., 3:4]; m = (1 + .050 * self.skin + .030 * self.fib)[..., None]; L[..., :3] = np.clip(L[..., :3] * m, 0, 1.2)
        far = smooth(GY + 60, GY - RY, self.gy)[..., None]; cen = np.exp(-((self.gx - CX) / 520) ** 2)[..., None]
        L[..., :3] = L[..., :3] * (mix(rgb('#b98fae'), np.ones(3, f32), .25) * (1 - far * .7) + np.array([1.10, .99, .86], f32) * far * .7) + al * far * cen * rgb('#ffb860') * .34
        return L

    def obj(self, cel, shc, f, o, ox, oy, front):
        kind, r, az, sz, per, Hh, ph, spin = o; T = 15 * per; th = math.radians(az); tau = ((f - 15 * ph) / T) % 1.; k = math.floor((f - 15 * ph) / T)
        g = G_(np.array([r]), np.array([th]), f)[0]; dep = .62 + .55 * (g[1] - (GY - RY)) / (2 * RY); h = 4 * Hh * dep * tau * (1 - tau); vy = 1 - 2 * tau
        c = min(tau, 1 - tau) * T                                                              # frames from contact
        sq = 1 + .20 * abs(vy) * ss(.5, 3, c) - .36 * math.exp(-(c / 1.1) ** 2); sgn = 1 if (k % 2 == 0) else -1
        rot = spin * sgn * math.sin(math.pi * tau) * (1 if abs(spin) < .6 else 2.2); bend = -.55 * vy * spin * 3 * ss(0, 3, c) + .16 * math.sin(f * .9 + az) * math.exp(-c / 4)
        dx = (30 * sgn * math.sin(math.pi * tau) * (.5 + .5 * h1(int(az) + 400, 3))); x, y = g[0] + ox + dx, g[1] + oy - h; s = sz * dep * 1.5
        e = np.linspace(0, 6.283, 30); kk = 1 - .55 * h / (Hh * dep + 1); shc.fill(c2(g[0] + ox + dx + np.cos(e) * 70 * s * (.55 + .45 * kk), g[1] + oy + 14 * s + np.sin(e) * 20 * s * (.55 + .45 * kk)), '#5a2350', .16 + .34 * kk)
        if kind == 'rock':
            rp = dict(lit='#cbb0c0', mid='#8a6f98', dark='#513e72', ink='#24152c'); rock(cel, x, y - 34 * s * sq, 46 * s, (f * .09 * spin, f * .11 * spin + az, f * .07 * spin), seed=int(abs(az)) % 24, pal=rp, light=(0, -.9, .2), lw=2.4 * max(.8, dep)); return
        PROPS[kind](cel, Rub(x, y, s, rot, sq, bend), SUNP, lw=2.5 * max(.8, dep), seed=int(abs(az)), face=float(ss(.1, .35, tau) * (1 - ss(.75, .95, tau))))
        if c < 2.5 and tau < .5:                                                               # dust lips on take-off
            for j in (-1, 1): q = (c + 1) * 12 * s; cel.line(spl([(x + j * 46 * s, g[1] + oy), (x + j * (58 * s + q), g[1] + oy - 10 - q * .3), (x + j * (72 * s + q), g[1] + oy - 2)], n=5), 3., '#fff0d0', taper=(.3, .6))

    def frame(self, f):
        hit = min([f - b for b in BEATS if f >= b]); jolt = 15 * math.exp(-hit / 2.6) * math.cos(hit * 1.25); z = 1 + .045 * float(ease(f / 60.)) + .012 * math.exp(-hit / 3.)
        dxc = 10 * math.sin(f * .05)
        zs = 1.0 + (z - 1) * .35 + .02; base = cam2d(self.sky, 960 + dxc * .15, 486 + 54 / zs + jolt * .18, zs)
        # corona: flame-cloud licks round the sun, breathing on the beat
        br = math.exp(-hit / 4.); t = f / 24.; LK = []
        for i in range(30):
            th = i / 30. * 6.283 + .06 * t; R0 = 496 * zs + 8 * br; big = h1(i, 4); p0 = np.array([960 - dxc * .15 + math.cos(th) * R0, 486 + math.sin(th) * R0])
            if p0[1] > 860: continue
            LK += lock(p0, th + .25, (62 + 115 * big) * (1 + .22 * br), 19 + 20 * big, t, 4000 + i, wave=.45, k=.8, hz=.9, bend=.5, curl=2.1 + .8 * h1(i, 5), cu=.4, shed=big > .5, r0=.9, belly=.2, n=24)
        cp = dict(h0='#f5702c', h1='#ff9a3e', h2='#ffc765', h3='#ffe9a8', hl='#b5402e')
        cor = paint_hair(LK, cp, 300, lw=1.8, light=(0, -1), seed=3, close=.05); rays = smooth(.1, .9, np.sin(self.ang * 18 + t * 1.1) * .5 + .5) * np.clip((self.rad - 500) / 500, 0, 1) * np.clip(1.5 - self.rad / 1400, 0, 1)
        img = base + rays[..., None] * rgb('#ff9860') * (.10 + .16 * br); img = over(img, cor, .92)
        img = over(img, self.elephant(f, -dxc * .4 + 10, jolt * .25), .97); img = over(img, self.wisps(f))
        B = self.bank[:, 60 + int(dxc * .4):60 + int(dxc * .4) + W]; img = over(img, cam2d(B, 960, 540 - jolt * .3, 1., border=cv2.BORDER_CONSTANT))
        img = img + (np.exp(-((self.gy - 748) / 70.) ** 2) * .30)[..., None] * rgb('#ffbf80')
        oy = jolt * .6; D = self.drum(f, oy); img = over(img, D)
        shc = Cel(ss=1); cb, cf = Cel(), Cel(); objs = sorted(OBJ, key=lambda o: math.sin(math.radians(o[2])) * o[1])
        for o in objs:
            fr = o[1] * math.sin(math.radians(o[2])) > -.05; self.obj(cf if fr else cb, shc, f, o, -dxc * .8, oy, fr)
        sand = Cel(ss=1)                                                                       # grains that hop on the skin
        for i in range(130):
            r = math.sqrt(h1(i, 1)) * .9; th = h1(i, 2) * 6.283; g = G_(np.array([r]), np.array([th]), f)[0]; dly = r * 4.; tau = ((f - dly) % 15) / 15.
            if f < dly: continue
            h = 4 * (26 + 70 * h1(i, 3)) * tau * (1 - tau) * (1.1 - .5 * r); s_ = 1.6 + 2.8 * h1(i, 4); e = np.linspace(0, 6.283, 8); sand.fill(c2(g[0] + np.cos(e) * s_, g[1] + oy - h + np.sin(e) * s_), '#fff0c8' if i % 3 else '#b5503f')
        # figure
        p = pose(f); hp = np.array([p['hd'][0], -p['hd'][1]]) * U; lagT = []
        for k in range(10):
            q = pose(max(0, f - k))['hd']; lagT.append(np.array([q[0], -q[1]]) * U - hp)
        gr = depth(f); lagT = np.array(lagT) * gr; gyf = foot_y(f); fy = gyf + float(skin_wave(0., f)) + oy + 34 * max([w for _, _, w in feet(f)] + [0.])
        for q in ('fA', 'fB'): e = np.linspace(0, 6.283, 40); shc.fill(c2(960 + p[q][0] * U * gr + np.cos(e) * 62 * gr, fy + 6 + np.sin(e) * 12 * gr), '#4a1a44', .55 / (1 + 5 * p[q][1]))
        S = shc.rgba(); img = over(img, cv2.GaussianBlur(S, (0, 0), 5), .95)
        obk = cb.layer(); obk = hotrim(obk, .05, 1., 2.6, '#fff0bc', '#ffae60', 1.); obk = para(obk, (0, 300), (0, 900), [(0, '#ffffff'), (1, '#ffe6d8')]); img = over(img, obk)
        top = fy - p['hd'][1] * U * gr; fy += max(0., 338 + 60 * float(ss(48, 54, f)) - top); sil = float(ss(28.4, 29.8, f) * (1 - ss(33.2, 36.5, f)))
        fig = G.figure(960, fy, U * gr, p, f, 'sun', lagT=lagT, l=(.15, -.95), rims=(), lw=3.1 * (.7 + .3 * gr), head_kw=dict(fs=.40, flow=(-.15 * math.sin(f * .2), -1.), size=1.0 + .10 * sil, ldir=(0, -1), socket=.35))
        # his own shadow dances toward us on the skin: the silhouette thrown over the ground line
        k = .30; d = self.gy - fy; mx = 960 + (self.gx - 960) / (1 + .0016 * np.clip(d, 0, 400)); my = fy - d / k; sh = cv2.remap(fig[..., 3], mx.astype(f32), my.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT); sh = cv2.GaussianBlur(sh * (d > -2), (0, 0), 3.2) * D[..., 3] * np.clip(1.06 - d / 420, .2, 1)
        img = img * (1 - sh[..., None] * .58) + (sh[..., None] * rgb('#5a1f58')) * .58
        fig = para(fig, (0, fy - 640 * gr), (0, fy), [(0, '#ead3ea'), (.55, '#f4dfec'), (1, '#ffcfa2')], amt=.85)      # back-lit: violet shade in front, warm bounce off the skin
        if sil > .01: g_ = fig[..., :3]; al = fig[..., 3:4]; lu = (g_ @ np.array([.3, .59, .11], f32))[..., None]; hy_ = fy - p['hd'][1] * U * gr; L_ = np.clip(lu[..., 0] / np.maximum(al[..., 0], .05), 0, 1); hm = (smooth(.50, .78, L_) * smooth(hy_ + .30 * U * gr, hy_ - .16 * U * gr, self.gy) * smooth(1.7 * U * gr, 1.2 * U * gr, np.abs(self.gx - 960 - p['hd'][0] * U * gr - .0)))[..., None]; hc_ = ramp(L_, [(0, '#4a1650'), (.50, '#a83a70'), (.72, '#f0703a'), (.90, '#ffb04a'), (1, '#ffe49a')]) * al; tg = (rgb('#2c0f45') * al + rgb('#7a2f7a') * lu * .30) * (1 - hm) + hc_ * hm; fig[..., :3] = g_ * (1 - .92 * sil) + tg * .92 * sil
        w_ = 3.0 * gr * (1 + .5 * sil)
        for dx_, dy_ in ((.10, 1.), (1., .25), (-1., .25)): fig = hotrim(fig, dx_, dy_, w_, '#fff6cf', '#ffb648', 1., .8)
        wr = np.clip(1 - blur(fig[..., 3], 11 * gr), 0, 1) * fig[..., 3]; fig[..., :3] += wr[..., None] * rgb('#ffb04a') * (.34 + .30 * sil)                                                    # light wrap
        img = screen(img, blur(fig[..., 3], 22)[..., None] * rgb('#ff9a3c') * .26); img = img + (np.exp(-(self.rad / 430.) ** 2) * .34 * sil)[..., None] * rgb('#ffd27a')
        img = over(img, sand.rgba()); img = over(img, fig)
        ofr = cf.layer(); ofr = hotrim(ofr, .05, 1., 3.2, '#ffedb4', '#ff9c58', 1.); ofr = para(ofr, (0, 600), (0, 1080), [(0, '#ffffff'), (1, '#c9a6c4')]); img = over(img, ofr)
        if hit < 2: sl = Cel(ss=1); speed_lines(sl, -math.pi / 2, 26, 3.2, (80, 330), '#fff4d4', seed=f, box=(0, 560, W, 960), alpha=.65); img = over(img, sl.rgba())
        img = img + (np.exp(-((self.rad) / 620.) ** 2) * (.0 + .07 * br))[..., None] * rgb('#ffb04e')
        zz = 1 + .055 * float(ease(f / 59.)) + .014 * math.exp(-hit / 3.2) * (hit < 12); img = cam2d(img, 960, 486 + 54 / zz, zz, .0075 * math.exp(-hit / 4.) * math.sin(hit * .9) * (1 if (f // 15) % 2 else -1))
        if f < 2: k = (.30, .10)[f]; img = img * (1 - k) + rgb('#fff3d0') * k
        return img.astype(f32)
