"""b1: one hat, four heads.  The hat is locked to the pixel; heads, light and weather change under it every 15 f."""
import numpy as np, math, cv2
from op import W, H, comp, clouds as CL
from op.cam import Camera3D
from op.paint import rgb, f32, smooth, grid, ramp, fbm, repaint
from op.comp import over, screen, blur, bokeh, lerp_keys, ease, radial, cam2d
from op.ink import Cel, spl
from op.kit import para, outline, rim, flare
from op import hat as HAT, sea as SEA
from op.fx_air import rain, motes, petals
from chars import roger as RG, shanks as SK, scaffold as SC
from chars._old import *

YAW, PIT, ROL = .16, .05, -.05
HK = dict(cx=960., cy=300., w=900., yaw=YAW, pitch=PIT, roll=ROL)
WEAR = (.04, .20, .50, .78); HLIGHT = ('storm', 'lamp', 'lamp', 'day')


def under(Hd, w, up=0.):
    """head placement so that op.hat.seat gives exactly HK"""
    Hd.pose(0, 0, 100., YAW, PIT, ROL); k = HAT.seat(Hd, w=w, up=up); s = 100. * HK['w'] / k['w']; Hd.pose(0, 0, s, YAW, PIT, ROL); k = HAT.seat(Hd, w=w, up=up); return HK['cx'] - k['cx'], HK['cy'] - k['cy'], s


def bounce(lf): return 0. if lf < 1 else .62 * math.exp(-(lf - 1) / 3.0) * math.sin(2 * math.pi * lf / 6.4)


def bust(fx, kind, pal, f, k):
    """shoulders under the chin, in head units"""
    C = Cel(); t = f // 2 * 2; w = lambda i, a: a * math.sin(t * .6 + i * 1.1); P = lambda L: fx.H.P3(np.array([(a, b, c) for a, b, c in L])); ln = pal['line']
    def sh(L, col, cr=(), lw=4.2, sd=0): S = spl(P(L), True, cr); C.fill(S, col); outline(C, S, lw, ln, sd); return S
    if kind == 'roger':
        S = sh([(-.50, -1.35, .1), (-.9, -1.62, 0), (-1.1, -2.6, 0), (1.1, -2.6, 0), (.9, -1.62, 0), (.50, -1.35, .1), (.2, -1.75, .35), (-.2, -1.75, .35)], '#27324a', (0, 2, 3, 5)); C.fill(P([(-.16, -1.6, .35), (.16, -1.6, .35), (.02, -2.5, .4)]), pal['shade'])
        for sg in (-1, 1):
            E = [(sg * .46, -1.22, .12), (sg * .66, -1.16 + w(0, .02), .16), (sg * 1.02, -1.30 + w(1, .03), .1), (sg * 1.55, -1.52 + w(2, .04), 0), (sg * 2.1, -1.66, 0), (sg * 2.2, -2.7, 0), (sg * .62, -2.7, .2), (sg * .44, -2.0, .3), (sg * .36, -1.6, .3)]
            S = sh(E, '#7a2028' if sg < 0 else '#541823', (0, 4, 5, 6)); C.clip(S); C.fill(spl(P([(sg * .5, -1.4, .2), (sg * .9, -1.7, .1), (sg * 1.1, -2.7, 0), (sg * .5, -2.7, .2)]), True), '#3a1019', .8)
            for i in range(4): C.line(spl(P([(sg * (.75 + .3 * i), -1.36 - .05 * i, .1), (sg * (.8 + .32 * i), -1.8, .05), (sg * (.78 + .3 * i), -2.5, 0)])), 4., ln, taper=(.2, .6), seed=i)
            C.unclip(); e = spl(P(E[:5]), n=10); C.line(e + A(0, 7), 17, '#7e5a2a', taper=(.05, .02), seed=3); C.line(e + A(0, 5), 11, '#c9a04a', taper=(.05, .02), seed=3); C.line(e, 4.2, ln, taper=(.1, .02), seed=4)
    elif kind == 'boy':
        S = sh([(-.46, -1.30, .1), (-.95, -1.52, 0), (-1.25, -1.8, 0), (-1.4, -2.7, 0), (1.4, -2.7, 0), (1.25, -1.8, 0), (.95, -1.52, 0), (.46, -1.30, .1), (0, -1.62, .35)], '#86705e', (0, 3, 4, 7)); C.clip(S)
        for i in range(6): C.line(spl(P([(-1.5, -1.66 - i * .20, 0), (-.5, -1.76 - i * .20, .3), (.5, -1.76 - i * .20, .3), (1.5, -1.66 - i * .20, 0)])), 22, '#4c5a78', taper=(.02, .02), seed=i)
        C.fill(spl(P([(.05, -1.5, .35), (.7, -1.5, .1), (1.5, -1.8, 0), (1.6, -2.8, 0), (.3, -2.8, .2)]), True), '#362e3c', .62); edge_hatch(C, spl(P([(.05, -1.6, .35), (.12, -2.0, .33), (.28, -2.5, .25)])), 12, 46, 2.6, 2.2, ln, 5); C.unclip()
        for sg in (-1, 1):                                                                   # soaked open shirt over the stripes
            E = [(sg * .42, -1.20, .14), (sg * .62, -1.24, .14), (sg * 1.0, -1.46, .04), (sg * 1.34, -1.78, 0), (sg * 1.55, -2.8, 0), (sg * .66, -2.8, .22), (sg * .60 + w(1, .01), -2.2, .28), (sg * .44, -1.70, .34)]; G = sh(E, '#b8a07c' if sg < 0 else '#75604f', (0, 4, 5, 7), 4.4, 12 + sg); C.clip(G)
            C.fill(spl(P([(sg * .62, -1.4, .2), (sg * .78, -1.9, .18), (sg * .70, -2.8, .2), (sg * .95, -2.8, .1), (sg * .98, -2.0, .1)]), True), '#6c573f' if sg < 0 else '#45373a', .85); C.fill(spl(P([(sg * 1.02, -1.6, .04), (sg * 1.18, -2.1, 0), (sg * 1.12, -2.8, 0), (sg * 1.3, -2.8, 0), (sg * 1.32, -2.0, 0)]), True), '#6c573f' if sg < 0 else '#45373a', .8)
            for i in range(6): x0 = .52 + .16 * i; C.line(spl(P([(sg * x0, -1.36 - .07 * i, .16), (sg * (x0 + .05), -1.8, .14), (sg * (x0 + .01 + .03 * (i % 2)), -2.4, .1)])), 3.4, ln, taper=(.2, .7), press=.45, seed=20 + i)
            edge_hatch(C, spl(P([(sg * .62, -1.5, .2), (sg * .76, -1.9, .18), (sg * .70, -2.5, .2)])), 12, 38, .4 if sg < 0 else 2.7, 2., ln, 30 + sg); edge_hatch(C, spl(P([(sg * 1.04, -1.7, .04), (sg * 1.16, -2.1, 0), (sg * 1.12, -2.5, 0)])), 9, 34, .4 if sg < 0 else 2.7, 2., ln, 33 + sg)
            for i in range(7): x0 = .56 + .13 * i; y0 = -1.42 - .10 * i - .12 * (i % 2); C.line(spl(P([(sg * x0, y0, .18), (sg * (x0 + .03), y0 - .16, .17), (sg * (x0 + .02), y0 - .34, .16)])), 5.5, '#ffe3ae' if sg < 0 else '#8fa0c8', taper=(.5, .6), press=.4, seed=40 + i, alpha=.85 if sg < 0 else .6)
            C.unclip(); K = [(sg * .40, -1.18, .16), (sg * .66, -1.16 + w(0, .01), .18), (sg * .86, -1.36, .12), (sg * .62, -1.56, .26), (sg * .42, -1.46, .34)]; G = sh(K, '#cdb58e' if sg < 0 else '#8a725c', (0, 2, 3), 4.2, 50 + sg); C.line(spl(P([(sg * .50, -1.26, .2), (sg * .62, -1.36, .22), (sg * .66, -1.48, .22)])), 3., ln, taper=(.3, .6), seed=52)
    elif kind == 'grown':
        S = sh([(-.48, -1.32, .1), (-.95, -1.58, 0), (-1.2, -2.7, 0), (1.2, -2.7, 0), (.95, -1.58, 0), (.48, -1.32, .1), (.06, -2.1, .4)], '#f0e4d0', (0, 2, 3, 5)); C.fill(P([(-.42, -1.40, .2), (.42, -1.40, .2), (.04, -2.2, .42)]), pal['skin']); C.fill(spl(P([(.1, -1.5, .35), (.46, -1.38, .2), (.36, -1.8, .35), (.06, -2.2, .42)]), True), pal['shade']); C.line(spl(P([(-.30, -1.62, .36), (-.12, -1.66, .4)])), 3.4, ln); C.line(spl(P([(.30, -1.62, .36), (.12, -1.66, .4)])), 3.4, ln)
        for sg in (-1, 1):
            C.line(spl(P([(sg * .48, -1.32, .1), (sg * .26, -1.74, .36), (.05, -2.16, .42)])), 5., ln, seed=6)
            E = [(sg * .50, -1.18, .0), (sg * .84, -1.12, .05), (sg * 1.35, -1.40, 0), (sg * 2.0, -1.62, 0), (sg * 2.2, -2.7, 0), (sg * .72, -2.7, .2), (sg * .74, -1.9, .2), (sg * .56, -1.5, .1)]; S = sh(E, '#231e2a', (0, 3, 4, 5)); C.clip(S)
            for i in range(4): C.line(spl(P([(sg * (.8 + .3 * i), -1.30 - .05 * i, .1), (sg * (.84 + .32 * i), -1.8, .05), (sg * (.9 + .3 * i), -2.5, 0)])), 16 - 2 * i, '#120f18', taper=(.6, .2), seed=i); C.line(spl(P([(sg * (.84 + .3 * i), -1.32 - .05 * i, .1), (sg * (.89 + .32 * i), -1.8, .05), (sg * (.96 + .3 * i), -2.5, 0)])), 3., '#4a4054', taper=(.3, .6), seed=i)
            C.unclip()
    return C.layer()


class Shot:
    id = 'b1'; n = 60; finish = dict(bloom=.32, vig=.18)
    EVENTS = [(0, 'cut', dict(size=.7, note='don 1')), (0, 'storm', dict(note='rain and sea', len=15)), (2, 'thunder', dict(size=.9)), (0, 'laugh', dict(note='Roger, huge', len=15)), (1, 'hat_flop', dict(size=.4)), (15, 'cut', dict(size=.6, note='don 2')), (15, 'wind_night', dict(note='lantern creak, rope, night sea', len=15)), (16, 'hat_flop', dict(size=.4)), (20, 'sob', dict(note='boy, held breath')),
              (30, 'cut', dict(size=.6, note='don 3')), (30, 'bar', dict(note='glasses, warm laughter', len=15)), (31, 'hat_flop', dict(size=.4)), (30, 'laugh', dict(note='Shanks, bright', len=15)), (45, 'cut', dict(size=.8, note='don 4')), (45, 'day', dict(note='breeze, gulls', len=15)), (46, 'hat_flop', dict(size=.4)), (49, 'kidlaugh', dict(note='shishishi'))]

    def setup(self):
        self.x, self.y = grid(); self.S = SEA.Sea(7, hs=3.4, wind=(.7, .6)); from chars import luffy as LU; self.LU = LU
        self.day = np.asarray(SC.cached('b1_day', self._day), f32); self.bar = np.asarray(SC.cached('b1_bar', self._bar), f32); self.night = np.asarray(SC.cached('b1_night', self._night), f32)

    def _day(self):
        sky = ramp(self.y / H, [(0, '#1f7fc4'), (.45, '#58b8de'), (.8, '#c9ecec'), (1, '#e9f5e2')]); S = CL.Sky(H, W, sky, light=(-.55, -.75, .35), seed=3)
        S.bank(300, 900, 1100, 560, ('#ffffff', '#e2eef4', '#a9cfe0', '#86b4d0'), '#cdeaf0', seed=5, profile=CL.peak(.3, .25, .4)); S.bank(1600, 940, 1100, 700, ('#ffffff', '#e6f0f4', '#aed2e2', '#8ab8d2'), '#cdeaf0', seed=9, profile=CL.peak(.6, .22, .4)); S.bank(1000, 960, 1500, 180, ('#ffffff', '#e2eef2', '#b4d4e0', '#98c0d4'), '#d6eef0', seed=12)
        im = S.paint(seed=3); C = Cel(); C.fill(np.array([(0, 900), (W, 900), (W, H), (0, H)]), '#2f9ec0'); r = np.random.default_rng(3)
        for i in range(240): y = 902 + 170 * r.random() ** 1.6; x = r.uniform(0, W); ln = (y - 890) * r.uniform(.3, 1.2); C.line(np.array([(x, y), (x + ln, y + r.normal(0, 1))]), 1 + (y - 900) * .02, ('#8fdbe4', '#1c7fae', '#e8fbf6')[int(r.integers(3))], taper=(.4, .4), seed=i)
        C.fill(spl(A((-40, 910), (140, 800), (330, 770), (520, 860), (660, 912), (640, 1100), (-60, 1100)), True, (5, 6)), '#4f9c58'); C.fill(spl(A((300, 780), (520, 860), (660, 912), (640, 1100), (340, 1100), (420, 900)), True), '#35784a')
        for i in range(60): x, y = r.uniform(-20, 600), r.uniform(830, 1060); C.fill(spl(A((x - 26, y + 8), (x - 14, y - 12), (x + 6, y - 16), (x + 26, y + 8)), True), ('#3f8a4c', '#68b463', '#2c6e44')[i % 3])
        for x, y, s in ((250, 792, 1.), (395, 806, .7)):
            C.fill(np.array([(x - 22 * s, y), (x - 14 * s, y - 96 * s), (x + 14 * s, y - 96 * s), (x + 22 * s, y)]), '#f1e6cf'); C.fill(np.array([(x + 2 * s, y), (x + 2 * s, y - 96 * s), (x + 14 * s, y - 96 * s), (x + 22 * s, y)]), '#bfae9e'); C.fill(np.array([(x - 18 * s, y - 94 * s), (x, y - 122 * s), (x + 18 * s, y - 94 * s)]), '#b5483a')
        l = C.layer(); im = over(im, l); return cv2.GaussianBlur(np.clip(im, 0, 1.2), (0, 0), 3.2)

    def _bar(self):
        x, y = self.x, self.y; im = ramp(np.clip(np.sqrt(((x - 330) / 1500) ** 2 + ((y - 330) / 1000) ** 2), 0, 1), [(0, '#e79a4a'), (.35, '#a85f30'), (.7, '#5c3022'), (1, '#2a1718')]); C = Cel(); r = np.random.default_rng(6)
        for i in range(14): C.line(np.array([(i * 150 - 20, 0), (i * 150 - 10, H)]), 5, '#2c1614', taper=(0, 0), alpha=.5, seed=i)
        for sy in (150, 400, 650):
            C.fill(np.array([(1080, sy), (W, sy - 20), (W, sy + 14), (1080, sy + 30)]), '#3a1d16')
            for i in range(17):
                bx = 1100 + i * 50 + r.uniform(-8, 8); h = r.uniform(80, 150); c = ('#6d3a24', '#3f5a3a', '#8a5a28', '#5a2a2e', '#36506a', '#a87a3a')[int(r.integers(6))]; C.fill(spl(A((bx - 16, sy), (bx - 17, sy - h * .6), (bx - 6, sy - h * .74), (bx - 6, sy - h), (bx + 6, sy - h), (bx + 6, sy - h * .74), (bx + 17, sy - h * .6), (bx + 16, sy)), True, (0, 3, 4, 7)), c); C.line(np.array([(bx - 10, sy - 8), (bx - 10, sy - h * .55)]), 5, '#ffd79a', taper=(.3, .3), alpha=.55)
        for i in range(5): SC.person(C, 120 + i * 130 + r.uniform(-20, 20), 560 + r.uniform(-30, 40), 110, SC.KINDS[int(r.integers(len(SC.KINDS)))], '#5a2c22', '#44201c', r, arms=i == 2)
        im = over(im, bokeh(C.layer(), 9, 1.)); im = screen(im, radial(330, 250, 900, 700, 2.0)[..., None] * rgb('#ffb85a') * .6); return np.clip(im, 0, 1.3)

    def _night(self):
        x, y = self.x, self.y; im = ramp(y / H, [(0, '#0d1428'), (.6, '#1d2c4a'), (1, '#1a2038')]); C = Cel(); r = np.random.default_rng(2)
        for i in range(8): a = 100 + i * 250; C.line(np.array([(a, 0), (a + r.uniform(-200, 200), H)]), 5, '#0a0f1d', taper=(0, 0), seed=i)
        for yy in (740, 800): C.fill(np.array([(0, yy), (W, yy - 40), (W, yy - 12), (0, yy + 30)]), '#241c22')
        for i in range(22): C.fill(np.array([(i * 92, 742), (i * 92 + 26, 741), (i * 92 + 30, H), (i * 92 + 4, H)]), '#1f1a24')
        C.fill(np.array([(1500, 0), (1620, 0), (1640, H), (1490, H)]), '#17141f'); im = over(im, bokeh(C.layer(), 7, 1.)); return np.clip(im, 0, 1.3)

    def chest(self, f):
        """an open chest, and the fruit in it"""
        C = Cel(); ln = '#1c0f10'; o = A(1560, 640)
        q = lambda L: A(*L) + o
        S = np.array(q([(-250, -60), (240, -90), (262, -210), (214, -290), (-210, -262), (-268, -180)])); S = spl(S, True, (0, 1)); C.fill(S, '#4a2a1c'); outline(C, S, 4.5, ln, 1); C.fill(q([(-236, -70), (228, -98), (244, -190), (-244, -164)]), '#2a1613')
        a = np.linspace(0, 2 * math.pi, 60)[:-1]; fc = o + A(0, -74); R = 112.
        F = np.c_[fc[0] + R * np.cos(a) * 1.02, fc[1] + R * np.sin(a) * (1 + .06 * np.cos(2 * a))]; C.fill(F, '#6b3a96'); C.clip(F); C.fill(np.c_[fc[0] + 40 + R * np.cos(a), fc[1] + 34 + R * np.sin(a)], '#4c2474')
        for (x, y, s) in ((-52, -48, 44), (34, -58, 40), (-6, 24, 46), (74, 10, 36), (-84, 34, 36), (60, 78, 34), (-40, 92, 34)):
            t = np.linspace(0, 1, 60); th = t * 2 * math.pi * 1.9; rr = s * (1 - .93 * t); C.line(np.c_[fc[0] + x + rr * np.cos(th), fc[1] + y + rr * np.sin(th)], 5.2, '#2b1048', taper=(.1, .3), seed=int(s)); C.line(np.c_[fc[0] + x - 3 + rr * .86 * np.cos(th), fc[1] + y - 3 + rr * .86 * np.sin(th)][:28], 3., '#b184dc', taper=(.3, .4), seed=int(s) + 1)
        C.unclip(); outline(C, F, 5., ln, 2); C.line(spl(fc + A((0, -R), (-6, -R - 26), (16, -R - 48), (42, -R - 40), (40, -R - 20), (24, -R - 22))), 10, '#4f8a3a', taper=(.05, .3), seed=3); C.line(spl(fc + A((-4, -R), (-11, -R - 26), (14, -R - 54), (48, -R - 43))), 3., ln, seed=4)
        sd = q([(-262, -52), (-246, 224), (-330, 184), (-348, -66)]); C.fill(sd, '#a26a3c'); C.clip(sd); [C.line(q([(-350, 0 + i * 56), (-250, 8 + i * 60)]), 4, ln, taper=(0, 0), seed=i) for i in range(4)]; C.fill(q([(-316, -70), (-288, -62), (-272, 220), (-300, 206)]), '#b09a68'); C.unclip(); outline(C, sd, 4.5, ln, 7); ls = q([(-258, -56), (-268, -180), (-210, -262), (-282, -250), (-344, -176), (-348, -68)]); C.fill(spl(ls, True, (0, 2, 5)), '#6a3d26'); outline(C, spl(ls, True, (0, 2, 5)), 4.5, ln, 8)
        S = q([(-262, -52), (252, -86), (232, 210), (-246, 224)]); C.fill(S, '#7a4a28'); C.clip(S); C.fill(q([(40, -80), (260, -90), (240, 220), (60, 220)]), '#55301e')
        for i in range(4): C.line(q([(-270, 8 + i * 58), (260, -14 + i * 60)]), 4, ln, taper=(0, 0), seed=i)
        for xx in (-190, 0, 180): C.fill(q([(xx - 20, -90), (xx + 20, -92), (xx + 16, 230), (xx - 18, 230)]), '#8c7a52'); C.fill(q([(xx + 4, -90), (xx + 20, -92), (xx + 16, 230), (xx + 4, 230)]), '#5c4d38')
        C.unclip(); outline(C, S, 5., ln, 5); C.fill(q([(-34, -60), (34, -64), (34, 20), (0, 36), (-34, 22)]), '#c9a04a'); outline(C, q([(-34, -60), (34, -64), (34, 20), (0, 36), (-34, 22)]), 4., ln, 6); C.fill(spl(fc * 0 + q([(-8, -34), (8, -34), (10, -14), (5, 8), (-5, 8), (-10, -14)]), True), ln)
        L = para(C.layer(), (1300, 400), (1800, 900), [(0, '#ffe6bc'), (1, '#8a5a50')]); L = rimlit(L, 1., .25, LIGHT['lamp'], 1., 1., 14, 5); K = Cel(); r = np.random.default_rng(3); yb, yf = (lambda x: 842 - .026 * x), (lambda x: 944 - .016 * x)
        K.fill(np.array([(-20, yb(-20)), (1940, yb(1940)), (1940, yf(1940)), (-20, yf(-20))]), '#a8642f'); K.clip(np.array([(-20, yb(-20)), (1940, yb(1940)), (1940, yf(1940)), (-20, yf(-20))])); K.fill(spl(A((-40, 830), (300, 828), (760, 850), (560, 900), (100, 950), (-40, 950)), True), '#e09a4e', .75); K.fill(spl(A((1500, 780), (1960, 770), (1960, 960), (1300, 960)), True), '#6a3a22', .7)
        for i in range(5): u = (i + .5) / 5.; K.line(np.array([(-20, yb(-20) * (1 - u) + yf(-20) * u), (1940, yb(1940) * (1 - u) + yf(1940) * u)]), 2.6, '#5a2f1c', taper=(0, 0), press=.5, seed=i)
        for i in range(60): u = r.random(); x0 = r.uniform(-20, 1900); ln_ = r.uniform(40, 170); yy = yb(x0) * (1 - u) + yf(x0) * u; K.line(np.array([(x0, yy), (x0 + ln_, yy - .02 * ln_)]), 1.6, '#6e3a20', taper=(.4, .5), seed=20 + i, alpha=.7)
        K.unclip(); K.fill(np.array([(-20, yf(-20)), (1940, yf(1940)), (1940, 1100), (-20, 1100)]), '#4a2619'); K.fill(np.array([(-20, yf(-20)), (1940, yf(1940)), (1940, yf(1940) + 20), (-20, yf(-20) + 20)]), '#7a4226'); K.line(np.array([(-20, yf(-20)), (1940, yf(1940))]), 5, '#ffcf86', taper=(0, .1), press=.4, seed=9)
        for i in range(12): K.line(np.array([(i * 170, yf(i * 170) + 22), (i * 170 + 4, 1100)]), 4, '#2a1411', taper=(0, 0), seed=i)
        Sx = Cel(ss=1); Sx.fill(spl(A((1206, 846), (1312, 872), (1800, 856), (1930, 872), (1930, 902), (1500, 916), (1300, 900)), True), '#1c0b0c', .85); Kl = cv2.GaussianBlur(K.layer(), (0, 0), 1.8); Kl = over_l(Kl, cv2.GaussianBlur(Sx.layer(), (0, 0), 9) * Kl[..., 3:4]); T = Cel(); T.fill(spl(A((1300, 862), (1312, 870), (1800, 854), (1800, 848)), True, (0, 1, 2, 3)), '#0f0608', .9); Kl = over_l(Kl, cv2.GaussianBlur(T.layer(), (0, 0), 2.5)); return over_l(Kl, cv2.GaussianBlur(L, (0, 0), 1.5))


    def frame(self, f):
        i, lf = f // 15, f % 15; t = f // 2 * 2; hk = dict(HK, light=HLIGHT[i], wear=WEAR[i], bounce=bounce(lf), f=f, wind=.4)
        if i == 0:
            flash = {2: 1., 3: .55, 4: .25, 9: .5, 10: .2}.get(lf, 0.); cam = Camera3D((0, 0, 7.), (.2, 10, 7.9), 1150., roll=.05 * math.sin(f * .2)); sky = ramp(self.y / H, [(0, '#0b1224'), (.5, '#1c2c48'), (1, '#33496a')]); n = fbm(H, W, (210, 80, 30), None, 5)[..., None]; sky = sky * (1 + .22 * n) + rgb('#8fb0e8') * flash * .55 * np.clip(.6 + .4 * n, 0, 1)
            w = self.S.render(cam, 4 + f / 24., SEA.STORM, sun=(-.5, .6, .5), tone=.8 + .6 * flash) if False else self.S.render(cam, 4 + f / 24., SEA.STORM, sun=(-.5, .6, .5)); img = w.over(sky) * (.75 + .5 * flash)
            if lf in (2, 3, 9):
                C = Cel(); r = np.random.default_rng(8 + lf); p = A(1500 if lf < 5 else 330, -20.); P = [p]
                for k in range(16): p = p + A(r.normal(-14 if lf < 5 else 10, 46), 46); P.append(p)
                for wd, c, al in ((26, '#8fb4ff', .35), (9, '#e8f2ff', 1.)): C.line(np.array(P), wd, c, taper=(.02, .6), press=.4, seed=3, alpha=al)
                for k in (4, 7, 10): C.line(spl([P[k], P[k] + A(60, 70), P[k] + A(90 + r.normal(0, 30), 170)]), 5, '#dce8ff', taper=(.1, .8), seed=k)
                img = over(img, C.layer())
            img = over(blur(img, 4.5), rain(f, n=(260, 110, 0), wind=-.34, seed=4, alpha=.8))
            fx = RG.FX(RG.PROPS, 0, 0, 1); cx, cy, s = under(fx.H, 2.60, .02); jaw = (1., .86, .72, .9, 1., .8, .7, .95)[lf // 2]; e = dict(RG.EXPR['laugh']); e.update(ly=-.94 - .06 * jaw, drop=.10 + .09 * jaw, sq=.8); ev = np.array([e[q] for q in RG.KEY])
            LY = RG.draw(cx, cy, s, YAW, PIT, ROL, expr=ev, light='storm', f=f, hat=True, wind=(1.4, 0), rimdir=(-.8, .6), lw=1.75, gust=.25, rim_amt=1. + flash, dirty=False); bd = bust(LY['fx'], 'roger', LY['pal'], f, 1.); bd = rimlit(bd, -.8, .6, LY['pal'], 1., .9, 14, 5)
            pr = lambda L: para(L, (1500, 200), (500, 1050), [(0, '#ffffff'), (.5, '#b8bedb'), (1, '#6670a4')]); parts = [pr(LY['back']), pr(bd), pr(LY['face']), pr(LY['hair']), pr(LY['stache'])]
        elif i == 1:
            fl = .88 + .12 * math.sin(f * 2.3) * math.sin(f * .9); img = self.night.copy(); img = screen(img, radial(250, 800, 1150, 900, 2.0)[..., None] * rgb('#ff9a3e') * .75 * fl); L = Cel(); sw = 10 * math.sin(t * .35); q = lambda P_: A(*P_) + A(232 + sw, 700)
            L.fill(q([(-60, -90), (60, -90), (74, 90), (-74, 90)]), '#ffdf8e'); L.fill(spl(q([(-18, 70), (-24, 10), (0, -50), (24, 10), (18, 70)]), True), '#fff8de')
            for xx in (-66, -22, 22, 66): L.line(q([(xx * .86, -92), (xx, 92)]), 8, '#1e1412', taper=(0, 0))
            for yy in (-100, 96): L.fill(q([(-84, yy - 14), (84, yy - 14), (84, yy + 14), (-84, yy + 14)]), '#2a1c18')
            L.line(spl(q([(-60, -110), (0, -190), (60, -110)])), 9, '#1e1412'); img = over(img, cv2.GaussianBlur(L.layer(), (0, 0), 3.0)); img = screen(img, radial(232 + sw, 700, 520, 520, 2.2)[..., None] * rgb('#ffc46a') * .8 * fl)
            fx = SK.FX(SK.BOY, 0, 0, 1); cx, cy, s = under(fx.H, 2.95, .0); LY = SK.draw(cx, cy, s, YAW, PIT, ROL, age='boy', expr='cry', light='lantern', f=f, side=1, rimdir=(1., -.35), lw=1.85, tears=1., tf=.42 + lf / 18., blow=1.2, look=(.0, -.01), hairlen=.85); bd = bust(LY['fx'], 'boy', LY['pal'], f, 1.); bd = rimlit(bd, 1., -.3, LY['pal'], 1., 1., 16, 6)
            pr = lambda L_: para(L_, (300, 900), (1500, 300), [(0, '#fff0cc'), (.45, '#d89a80'), (1, '#4b4a80')]); parts = [pr(LY['back']), pr(bd), pr(LY['face']), pr(LY['hair'])]
        elif i == 2:
            img = self.bar.copy(); sw = math.sin(t * .2); img = screen(img, radial(330 + 8 * sw, 200, 560, 480, 2.2)[..., None] * rgb('#ffd88a') * .55); img = over(img, self.chest(f)); fx = SK.FX(SK.GROWN, 0, 0, 1); cx, cy, s = under(fx.H, 2.66, .02)
            jaw = (1., .80, .62, .85, 1.05, .8, .62, .9)[lf // 2]; LY = SK.draw(cx, cy, s, YAW, PIT, ROL, age='grown', expr='laugh', light='lamp', f=f, side=1, rimdir=(1., .25), lw=1.75, jaw=jaw, blow=.5); bd = bust(LY['fx'], 'grown', LY['pal'], f, 1.); bd = rimlit(bd, 1., .25, LY['pal'], 1., 1., 16, 6)
            pr = lambda L_: para(L_, (300, 250), (1500, 1000), [(0, '#fff2d2'), (.45, '#eebc98'), (1, '#7a5068')]); parts = [pr(LY['back']), pr(bd), pr(LY['face']), pr(LY['hair'])]
        else:
            img = cam2d(self.day, W / 2 + lf * 1.2, H / 2, 1.04); C = Cel(ss=1)
            for mx, my, s_ in ((250, 700, 1.), (395, 738, .7)):
                for k in range(4): a = f * .09 + k * math.pi / 2; d = A(math.cos(a), math.sin(a)); n_ = A(-d[1], d[0]); C.fill(np.array([A(mx, my), A(mx, my) + d * 100 * s_, A(mx, my) + d * 100 * s_ + n_ * 20 * s_, A(mx, my) + d * 22 * s_ + n_ * 20 * s_]), '#f6efdf')
            img = over(img, cv2.GaussianBlur(C.layer(), (0, 0), 3.)); Hd = self.LU.LHead(**self.LU.KID); cx, cy, s = under(Hd, 3.45, -.06)
            ex = ('smile', 'kidgrin', ease(np.clip((lf - 1) / 3., 0, 1))) if lf < 4 else 'kidgrin'; R, Hd, E, L = self.LU.record(cx, cy + 0, s, YAW, PIT, ROL, ex, 'day', f, kid=True, hat=True); kb = HAT.hat(part='back', **hk)
            img = over(img, self.LU.over4(kb, R.cel(L, s, np.zeros((H, W, 4), f32)))) if False else over(img, R.cel(L, s, kb.copy())); img = over(img, HAT.hat(part='front', **hk)); return img + motes(f, 40, seed=4, light=None, col='#ffffff', drift=(50, -10), size=1.2, amt=.4)
        al = np.clip(sum(p[..., 3] for p in parts), 0, 1); img = over(img, HAT.hat(part='back', **hk))
        for p in parts: img = over(img, p)
        fr = HAT.hat(part='front', **hk); img = over(img, fr)
        if i == 0: img = over(img, rain(f, n=(0, 60, 26), wind=-.34, seed=14, alpha=.9))
        if i == 1:                                                                           # rain runs off the brim
            D = Cel()
            for j in range(11):
                u = (j + .3 + .4 * ((j * 7) % 5) / 5.) / 11.; x0 = 530 + 860 * u; yb = 490 - 56 * u + 14 * math.sin(math.pi * u); age = ((lf + j * 5) % 12) / 12.; g = max(0., age - .42); sz = 4 + 7 * min(1., age / .42); y0 = yb + 1500 * g * g * 1.6; st = 1 + 3.4 * g; D.fill(spl(A((x0, y0 - sz * 1.2 * st), (x0 + sz * .72, y0 + sz * .3), (x0, y0 + sz), (x0 - sz * .72, y0 + sz * .3)), True, (0,)), '#ffe0a4', .9); D.fill(spl(A((x0 - sz * .2, y0 - sz * .4), (x0 + sz * .14, y0 + sz * .1), (x0 - sz * .2, y0 + sz * .4)), True), '#ffffff', .95)
            img = over(img, D.layer())
        if i == 1: img = over(img, rain(f, n=(60, 24, 0), wind=-.2, seed=24, alpha=.35, col='#ffd8a0'))
        if i == 2: img = img + motes(f, 60, seed=6, light=radial(330, 250, 1500, 1100, 1.), col='#ffe0a0', drift=(8, -8), size=1.3, amt=.6)
        return img
