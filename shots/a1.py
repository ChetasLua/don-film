"""a1: the square at Loguetown.  Low in the crowd, 18 mm; timber tower against a back-lit cumulonimbus; the one warm hat."""
import numpy as np, math, cv2
from op import W, H, comp
from op.cam import Camera3D
from op.paint import rgb, f32, smooth, grid, ramp
from op.comp import over, screen, blur, bokeh, lerp_keys, ease, radial, cam2d
from op.ink import Cel, spl
from op.kit import para, outline, rim, shifted
from op.fx_air import motes
from op import hat as HAT
from chars import roger as RG, scaffold as SC
from chars._old import *

EZ0, EZ1, D_BOY = 1.36, 1.60, 1.27
TILT = math.radians(16)
YH = 540 + 960 * math.tan(TILT)
PLANES = [  # depth m, seed, people, flat, inner, blur
    (26., 1, 190, '#858d90', '#788187', 1.4), (18., 2, 130, '#68747f', '#5a6673', 1.0), (12.6, 3, 92, '#4c5968', '#3f4b5a', .7), (9., 4, 62, '#394453', '#2c3644', .5), (6.2, 5, 42, '#2b3341', '#1f2631', .6), (4.2, 6, 25, '#1f2530', '#151a23', 1.3), (2.9, 7, 13, '#151a24', '#0e121a', 2.8)]


def heart(f):
    """two soft thumps each: 0,5 and 30,35"""
    z = 0.
    for t0, a in ((0, 1.), (5, .6), (30, 1.), (35, .6)):
        t = f - t0
        if t >= 0: z += a * (t / 1.2 * math.exp(1 - t / 1.2) if t < 12 else 0)
    return z


class Shot:
    id = 'a1'; n = 60; finish = dict(bloom=.30, vig=.20, soft=.16)
    EVENTS = [(0, 'heart', dict(size=.5, note='don')), (5, 'heart', dict(size=.3, note='-don')), (30, 'heart', dict(size=.5)), (35, 'heart', dict(size=.3)), (0, 'crowd', dict(note='huge hushed crowd, murmur', len=60)), (0, 'wind', dict(note='high wind, coat far up, spear pennant', len=60)), (12, 'gull', dict(pan=-.4)), (40, 'gull', dict(pan=.5)), (22, 'creak', dict(note='timber tower creak', size=.3))]

    def setup(self):
        self.sky = np.asarray(SC.sky(), f32); w, h, oy = 2240, 1180, 300; self.pl = []
        def mk():
            out = []
            for d, sd, n, c, k, bl in PLANES:
                for sel in (0, 1):
                    skip = ((-400, 1180),) if d < 3.5 else ((300, 980),) if d < 5 else ()
                    L = SC.plate(d, sd, n, c, k, YH, EZ0, w, h, oy, sel=sel, x0=0, x1=2240, skip=skip, accs=(rgb(c) * A(1.06, .94, .92), rgb(c) * A(.90, 1.0, 1.10), rgb(c) * A(1.02, 1.0, .88)), rimpx=2.2 if d > 10 else 3.4 if d > 4 else 5., ramt=.55 if d > 15 else .85)
                    if d == 9. and sel == 0: L = self.secret(L, oy)
                    if bl: L = cv2.GaussianBlur(L, (0, 0), bl)
                    out.append(L)
            t = SC.town(2240, 1180, YH); out.append(cv2.GaussianBlur(t, (0, 0), 1.0)); return np.stack(out)
        self.pl = [SC.u8(np.asarray(v, f32)) for v in SC.cached('a1_planes', mk)]; self.oy = oy
        self.x, self.y = grid(); self.rays = None
        self.ml = (radial(1130, 330, 1300, 1000, 1.0)).astype(f32)

    @staticmethod
    def secret(L, oy):
        """Pandaman, far right, looking straight at us; a feathered wide hat; a hooded man"""
        C = Cel(L.shape[1], L.shape[0], 0, oy); P = lambda cx, cy, s, Q: np.array([(cx + a * s, cy + b * s) for a, b in Q]); cx, cy, s = 1965, 792, 30
        SC.person(C, cx, cy - 4, 25, 'bald', '#39424f', '#2c3542', np.random.default_rng(1))
        for e in (-1, 1): C.fill(spl(P(cx, cy, s, [(e * .34, -.30), (e * .50, -.52), (e * .68, -.40), (e * .62, -.12)]), True), '#171b24')
        C.fill(spl(P(cx, cy, s, [(-.60, .15), (-.46, -.30), (0, -.45), (.46, -.30), (.60, .15), (.40, .55), (0, .68), (-.40, .55)]), True), '#a9adab')
        for e in (-1, 1): C.fill(spl(P(cx, cy, s, [(e * .14, .0), (e * .30, -.14), (e * .42, .06), (e * .28, .22)]), True), '#171b24'); C.fill(spl(P(cx, cy, s, [(e * .24, .0), (e * .29, -.05), (e * .33, .02), (e * .28, .07)]), True), '#e8e6da')
        C.fill(spl(P(cx, cy, s, [(-.08, .22), (.08, .22), (0, .30)]), True), '#171b24'); C.line(P(cx, cy, s, [(-.16, .40), (0, .44), (.16, .40)]), 1.4, '#171b24')
        # hawk eye: plumed wide hat and the tall cross hilt over the shoulder
        x, y, s = 1590, 742, 30; SC.person(C, x, y, 27, 'bare', '#333d4c', '#28313f', np.random.default_rng(4))
        C.fill(spl(P(x, y, s, [(-1.25, .28), (-.6, .10), (-.42, -.26), (.1, -.40), (.46, -.24), (.6, .10), (1.25, .20), (.6, .36), (0, .42), (-.6, .38)]), True, (0, 6)), '#222a37'); C.fill(spl(P(x, y, s, [(-.4, -.2), (-.1, -.62), (.5, -.95), (1.2, -.92), (1.75, -.55), (1.2, -.62), (.6, -.5), (.1, -.1)]), True, (4,)), '#8a9298'); C.fill(P(x, y, s, [(.86, 1.3), (1.28, -1.6), (1.40, -1.6), (1.02, 1.4)]), '#1f2733'); C.fill(P(x, y, s, [(.82, -1.02), (1.72, -.90), (1.70, -.78), (.80, -.90)]), '#1f2733')
        x, y, s = 1060, 770, 30; SC.person(C, x, y, 27, 'hood', '#313b49', '#26303d', np.random.default_rng(8)); C.fill(spl(P(x, y, s, [(-.62, 1.3), (-.56, .30), (-.30, -.20), (.06, -.34), (.40, -.14), (.58, .36), (.62, 1.3)]), True), '#26303c'); C.line(P(x, y, s, [(-.28, .0), (.0, .3), (-.06, .9)]), 2., '#1a222d')
        n = rim(C.rgba(), .8, 2.6, '#ffe6b2', .8, .6); return L * (1 - n[..., 3:4]) + n

    def cam(self, f):
        e = float(ease(f / 59. * .92 + .04)); ez = EZ0 + (EZ1 - EZ0) * e; ex = .10 * e
        return ex, ez

    def boys(self, f, dx, dy):
        """left third, sharp.  Boy in the straw hat from behind; a boy with a round nose and a bandana beside him"""
        t = f // 2 * 2; sway = 2.5 * math.sin(t * .13); look = float(lerp_keys(t, [(0, 0), (14, 0), (26, 1), (59, 1)])); blow = math.sin(t * .5)
        ox, oy = dx, dy; layers = []
        # ---- the round nose boy: body, head, nose, bandana are separate cels, each with its own light from the sun (right, above)
        bx, by = 196 + ox, 742 + oy + sway; p = dict(LIGHT['gold']); p.update(skin='#cf9c78', shade='#835a56', deep='#443244', lite='#ffd09a', line='#2a1c24', hair='#2b4a66', hair2='#1b2f47', hl='#3c6080', hi='#7fa3ba', white='#e8decb', white_sh='#8d8480', tooth='#efe6d2', tooth_sh='#b09a8c', mouth='#43202a', tongue='#9c5654', gum='#7a4246')
        def lit(Cq, amt=.42, wide=11., hot=4.6):
            Lq = para(Cq.layer(), (bx + 175, by - 130), (bx - 200, by + 170), [(0, '#fff6e2'), (.36, '#eecab0'), (.68, '#a78a94'), (1, '#635d7c')], amt=amt); return rim(rim(Lq, -wide, wide * .55, '#ffbf6c', .62, 4.5), -hot, hot * .5, '#fff2cc', 1., .7)
        C = Cel(); bd = [(-30, 150), (-84, 166), (-168, 210), (-214, 282), (-236, 520), (250, 520), (222, 300), (188, 224), (96, 176), (52, 150)]; S = spl(A(*bd) + A(bx, by), True, (4, 5)); C.fill(S, '#4a5160'); C.clip(S)
        for i in range(7): C.line(spl(A((-240, 250 + i * 40), (-120, 232 + i * 40), (0, 240 + i * 40), (120, 234 + i * 40), (260, 276 + i * 40)) + A(bx, by), n=10), 17, '#7a6468', taper=(.02, .02), seed=i)
        C.fill(spl(A((-60, 160), (-180, 200), (-250, 320), (-260, 540), (-20, 540), (-96, 400), (-60, 300), (-104, 236)) + A(bx, by), True), '#2b2f40', .78); C.fill(spl(A((150, 560), (128, 420), (170, 320), (150, 262), (196, 330), (190, 430), (214, 560)) + A(bx, by), True, (0,)), '#2b2f40', .6)
        for i, L_ in enumerate(([(-70, 190), (-110, 280), (-92, 390)], [(60, 190), (84, 270), (62, 360)], [(150, 232), (134, 320), (150, 420)], [(-150, 240), (-158, 330), (-140, 430)], [(0, 222), (14, 300), (-6, 390)])): C.line(spl(A(*L_) + A(bx, by), n=10), 3.2, p['line'], taper=(.2, .7), press=.4, seed=i + 5)
        edge_hatch(C, spl(A((-60, 300), (-96, 400), (-60, 500)) + A(bx, by)), 12, 46, .2, 1.8, p['line'], 7); C.unclip(); outline(C, S, 4.0, p['line'], 2, gaps=[(.52, .60)])
        N = spl(A((-52, 60), (-66, 130), (-96, 178), (0, 206), (84, 186), (52, 150), (40, 90)) + A(bx, by), True); C.fill(N, p['skin']); C.clip(N); C.fill(spl(A((-110, 40), (80, 60), (70, 118), (28, 142), (-6, 158), (-30, 210), (-120, 210)) + A(bx, by), True), p['shade']); C.fill(spl(A((-110, 40), (80, 50), (58, 96), (4, 120), (-60, 150), (-120, 160)) + A(bx, by), True), p['deep'], .8); edge_hatch(C, spl(A((58, 100), (4, 124), (-40, 150)) + A(bx, by)), 9, 30, 1.4, 1.7, p['line'], 8); C.unclip(); outline(C, N, 3.4, p['line'], 3); lay = lit(C)
        C = Cel(); fx = FX(dict(a=[(-1.0, .28), (-.96, .42), (-.88, .54), (-.74, .65), (-.56, .725), (-.35, .77), (-.1, .785), (.12, .785), (.35, .77), (.6, .72), (.8, .57), (.93, .35), (1., .05)], cheek=.06, p=2.1), bx, by - 8, 136, yaw=.96, pitch=.26 + .07 * look, roll=-.06, drop=.02 + .03 * look); C.fill(fx.H.skull(), p['hair'])
        for i, (a, tp, b) in enumerate((((-.7, -.05, -.2), (-.98, -.50, -.5), (-.6, -.45, -.35)), ((-.5, -.2, -.6), (-.62, -.85, -.8), (-.3, -.5, -.6)), ((-.75, .15, -.3), (-1.15, -.16, -.6), (-.75, -.15, -.3)), ((-.62, -.1, -.45), (-.86, -.72, -.7), (-.46, -.4, -.5)))):
            q = fx.H.P3(A(a, tp, b)); clump(C, q[0], q[1] + A(-3 * blow, 2 * blow), q[2], p, .62, 5 + i, .2, .1, 4, .6)
        Fc = fx.face(); C.fill(Fc, p['skin']); C.clip(Fc); fr = lambda L_, far: np.vstack([fx.J(fx.S(L_)), fx.H.P3(A((.2 * far, -1.5, .3), (1.5 * far, -1.5, 0), (1.5 * far, 1.2, 0)))]); se = [(.04, .9), (-.02, .42), (-.16, .10), (-.12, -.10), (-.02, -.24), (-.14, -.42), (-.10, -.62), (.02, -.80), (.12, -1.02)]
        C.fill(fr(se, -1), p['shade']); C.fill(fr([(.46, .9), (.44, .4), (.36, .12), (.46, -.06), (.40, -.26), (.44, -.48), (.36, -.70), (.30, -.95)], 1), p['lite']); C.fill(fx.S([(-.5, .22), (-.2, .30), (.1, .34), (.4, .36), (.7, .40), (.8, .7), (-.6, .7)], 0, True), p['deep'], .62)
        C.fill(fx.S([(-.36, -.20), (-.24, -.30), (-.10, -.30), (-.14, -.22), (-.26, -.18)], 0, True), p['deep'], .35); edge_hatch(C, fx.S(se[2:5]), 8, 30, 2.7, 1.7, p['line'], 30); edge_hatch(C, fx.J(fx.S(se[5:8])), 8, 30, 2.7, 1.7, p['line'], 31); edge_hatch(C, fx.S([(-.02, -.22), (.10, -.26), (.20, -.24)]), 6, 22, 1.9, 1.5, p['line'], 32); nz = fx.H.P3(A((.02, -.24, .98)))[0]; a = np.linspace(0, 2 * math.pi, 40)[:-1]; R = 33.; C.fill(np.c_[nz[0] - 18 + R * 1.02 * np.cos(a), nz[1] + 20 + R * .9 * np.sin(a)], p['shade']); C.unclip()
        e = fx.H.ear(-1); C.fill(e['outer'], p['skin']); C.fill(e['inner'], p['shade']); outline(C, e['outer'], 2.8, p['line'], 4); C.line(e['inner'], 1.8, p['line'], seed=5)
        eye(C, fx, 1, p, .92, .05, .02, .02, r=.05, w=.16, h=.185, x=.31, seed=12, bag=False, lw=1.7); eye(C, fx, -1, p, .92, .05, .07, .03, r=.05, w=.17, h=.185, x=.26, seed=13, bag=False, lw=1.7); brow(C, fx, -1, p, [(.10, .25), (.11, .32), (.30, .40), (.50, .30), (.30, .335)], 1.1, 14, col=p['hair'])
        laugh(C, fx, .135, -.655, -.625, -.755 - .05 * look, p, 20, lw=1.15, up=1.7, lo=1.8, tongue=.5); C.line(fx.S([(-.20, -.50), (-.23, -.60), (-.20, -.70)]), 2.0, p['line'], taper=(.5, .6), seed=22); outline(C, Fc, 3.0, p['line'], 15, gaps=[(.02, .10)]); lay = over_l(lay, lit(C))
        C = Cel(); nc = np.c_[nz[0] + R * np.cos(a), nz[1] + R * np.sin(a)]; C.fill(nc, '#6e2f38'); C.clip(nc); C.fill(np.c_[nz[0] + 9 + R * .92 * np.cos(a), nz[1] - 9 + R * .90 * np.sin(a)], '#9c474a'); C.fill(np.c_[nz[0] + 14 + R * .52 * np.cos(a), nz[1] - 13 + R * .48 * np.sin(a)], '#c46e62'); edge_hatch(C, np.c_[nz[0] - 6 + R * .62 * np.cos(a[14:26]), nz[1] + 6 + R * .62 * np.sin(a[14:26])], 7, 18, .7, 1.5, p['line'], 17); C.unclip(); C.fill(np.c_[nz[0] + 15 + 6 * np.cos(a), nz[1] - 16 + 4.5 * np.sin(a)], '#ffe9cc'); outline(C, nc, 3.0, p['line'], 16); lay = over_l(lay, lit(C, .6, 7., 3.))
        C = Cel(); bn = [(-.80, .22, .10), (-.84, .55, -.1), (-.62, .92, -.15), (0, 1.10, 0.), (.50, .96, .2), (.66, .62, .45), (.58, .36, .62), (.30, .40, .80), (0, .42, .88), (-.34, .40, .78), (-.62, .30, .50)]; Bq = spl(fx.H.P3(A(*bn)), True); kn = fx.H.P3(A((-.80, .28, -.35)))[0]
        for i, (u, v) in enumerate(((-74, -14), (-66, 34))): clump(C, kn + A(2, -8), kn + A(u, v) + A(-5 * blow, 8 * math.sin(t * .5 + i)), kn + A(2, 12), dict(p, hair='#59606c', hl='#39404c'), .7, 50 + i, .3, .1, 2)
        C.fill(Bq, '#59606c'); C.clip(Bq)
        for i in range(5): C.line(spl(fx.H.P3(A((-.9, .36 + i * .16, .2), (-.4, .52 + i * .15, .6), (.2, .54 + i * .15, .7), (.7, .5 + i * .14, .5))), n=8), 10, '#3c4350', seed=40 + i)
        C.fill(spl(fx.H.P3(A((-.95, .1, .1), (-.9, .7, -.1), (-.5, 1.1, -.1), (-.42, .8, .3), (-.52, .55, .45), (-.36, .40, .75), (-.7, .2, .4))), True), '#262b3a', .7)
        for i, (x0, x1) in enumerate(((-.6, -.2), (-.5, .05), (-.3, .3))): C.line(spl(fx.H.P3(A((x0, .40, .5), ((x0 + x1) / 2, .62 + .06 * i, .65), (x1, .92, .35))), n=8), 2.2, p['line'], taper=(.2, .7), seed=44 + i)
        C.unclip(); outline(C, Bq, 3.4, p['line'], 17, gaps=[(.34, .46)]); lay = over_l(lay, lit(C)); layers.append(lay)
        # ---- the hat boy
        C = Cel(); hx, hy = 596 + ox, 628 + oy - sway * .8; q = lambda L: A(*L) + A(hx, hy); p = dict(LIGHT['gold']); p.update(skin='#9c6d55', shade='#5b4040', hair='#6b2a2a', hair2='#431a20', hl='#8a3a34', hi='#a84c3c')
        S = spl(q([(-58, 205), (-84, 226), (-160, 258), (-206, 330), (-226, 560), (232, 560), (212, 330), (166, 258), (84, 226), (58, 205)]), True, (4, 5)); C.fill(S, '#556070'); C.clip(S)
        C.fill(spl(q([(60, 220), (170, 258), (220, 330), (240, 560), (40, 560), (92, 430), (96, 320)]), True), '#39424f'); C.fill(spl(q([(-60, 240), (0, 270), (60, 240), (40, 320), (6, 420), (-10, 320)]), True), '#39424f')
        C.line(spl(q([(-150, 262), (-10, 400), (150, 560)]), n=10), 30, '#2a211f', taper=(.02, .02), seed=81); C.line(spl(q([(-138, 258), (4, 398), (164, 556)]), n=10), 4., '#8a6a44', taper=(.1, .1), seed=82)
        for i, L in enumerate(([(-150, 280), (-120, 360), (-128, 470)], [(-40, 262), (-60, 360), (-40, 460)], [(60, 262), (70, 360), (50, 470)], [(140, 290), (120, 380), (132, 470)])): C.line(spl(q(L), n=10), 3.4, p['line'], taper=(.2, .7), seed=60 + i)
        C.unclip(); outline(C, S, 5., p['line'], 61); N = spl(q([(-58, 120), (-60, 218), (0, 250), (60, 218), (58, 120)]), True, (0, 4)); C.fill(N, p['skin']); C.fill(spl(q([(0, 130), (46, 130), (44, 222), (0, 246), (14, 190)]), True, (0, 1)), p['shade']); C.line(q([(-8, 170), (-4, 222)]), 2.6, p['line']); outline(C, N, 4.2, p['line'], 62)
        Hd = spl(q([(-112, 60), (-116, 0), (-60, -60), (60, -60), (116, 0), (112, 60), (90, 130), (40, 178), (-40, 178), (-90, 130)]), True); C.fill(Hd, p['skin'])
        for e in (-1, 1):
            E = spl(q([(e * 106, 64), (e * 134, 50), (e * 140, 84), (e * 124, 122), (e * 104, 116)]), True); C.fill(E, p['skin']); outline(C, E, 3.6, p['line'], 63 + e)
        BH = spl(q([(-122, -10), (-118, 70), (-96, 124), (-52, 150), (0, 162), (52, 150), (96, 124), (118, 70), (122, -10)]), True, (0, 8)); C.fill(BH, p['hair2'])
        for i, (a, tp, b) in enumerate((((-118, 20), (-112, 148), (-60, 30)), ((-92, 30), (-74, 178), (-30, 30)), ((-54, 40), (-30, 200), (6, 40)), ((-14, 40), (20, 206), (44, 40)), ((26, 40), (66, 188), (84, 30)), ((62, 30), (104, 158), (116, 20)), ((-128, 0), (-138, 112), (-92, 20)), ((92, 10), (140, 104), (128, 0)))):
            C.fill(spike(q([a])[0], q([(tp[0] + 6 * blow, tp[1] * 1.16)])[0], q([b])[0], .2, .05), p['hair']); clump(C, q([a])[0], q([(tp[0] + 5 * blow, tp[1] * 1.16)])[0], q([b])[0], p, .75, 70 + i, .2, .06 * (-1) ** i, 3, .5, edge=p['line'])
        layers.append(C.layer())
        return layers, (hx, hy - 20)

    def frame(self, f):
        ex, ez = self.cam(f); dz = ez - EZ0; th = heart(f); t = f // 2 * 2
        shift = lambda d: (-960 * ex / d, 960 * dz / d)
        img = cam2d(self.sky, 1066 + 60 * ex, 654 - dz * 60, 1.02)
        sx, sy = shift(60.); img = SC.put8(img, self.pl[14], -160 + sx, sy)
        low = smooth(520, 900, self.y)[..., None]; img = img * (1 - low * .25) + rgb('#cfc3a2') * low * .25
        cam = Camera3D((ex, 0, ez), (ex, math.cos(TILT), ez + math.sin(TILT)), 960.)
        for k in range(0, 2):
            for sel in (0, 1):
                d = PLANES[k][0]; a, b = shift(d); img = SC.put8(img, self.pl[k * 2 + sel], -160 + a, self.oy + b + 1.0 * math.sin(t * .22 + sel * 2 + k))
            img = screen(img, (smooth(700, 940, self.y) * .12)[..., None] * rgb('#ffe2ab'))
        tw, P = SC.tower(cam, f); tw = para(tw, (0, 120), (0, 960), [(0, '#ffffff'), (.5, '#f2ece4'), (1, '#d9d2c8')]); hz = (smooth(380, 980, self.y) * .42)[..., None] * tw[..., 3:4]; tw[..., :3] = tw[..., :3] * (1 - hz) + rgb('#a9a793') * hz
        img = screen(img, blur(tw[..., 3], 26)[..., None] * rgb('#ffe9b8') * .16); img = over(img, tw)
        G = Cel(); u = cam.scale(P(0, 0, SC.TZ)[None])[0] * 1.32; pt = lambda x, y: cam.project(P(x, y, SC.TZ)[None])[0][0]
        g0, g1, k0 = pt(-1.22, .2), pt(1.22, .2), pt(.0, -.5); SC.guard(G, g0[0], g0[1] + 4, u * 1.02, f, -1, .06); SC.guard(G, g1[0], g1[1] + 4, u * 1.02, f, 1, .06); SC.condemned(G, k0[0], k0[1] + 3, u * 1.06, f)
        gl = G.layer(); gl = rim(gl, 2.6, 1.6, '#fff3cb', 1., .5); gl = rim(gl, -2.2, 1.8, '#ffe0a0', .9, .5); img = screen(img, blur(gl[..., 3], 12)[..., None] * rgb('#fff1cc') * .35); img = over(img, gl)
        B = Cel()
        for i in range(8):
            z = SC.np.random.default_rng(50 + i); v = z.uniform(.6, 1.4); q = f / 24.; x = 300 + i * 205 + 70 * math.sin(q * .7 * v + i) + q * 34 * v; y = 150 + 46 * (i * 7 % 5) + 22 * math.sin(q * 1.1 + i * 2) - (0 if i % 3 else 40); s = 12 + 5 * (i * 3 % 4); fl = math.sin(q * 1.3 + i); SC.gull(B, x, y + dz * 20, s, q * 9 * v + i if fl > -.2 else .4, bank=.25 * math.sin(q * .9 + i))
        z = f / 24.; SC.gull(B, 1560 + z * 130, 560 - z * 60, 40, z * 11, bank=-.22); SC.gull(B, 820 - z * 40, 430 - z * 26, 24, z * 10 + 2, bank=.18)
        img = over(img, B.layer())
        for k in range(2, 6):
            d = PLANES[k][0]; a, b = shift(d)
            for sel in (0, 1): img = SC.put8(img, self.pl[k * 2 + sel], -160 + a, self.oy + b + (1.2 + k * .35) * math.sin(t * .20 + sel * 2.4 + k * 1.3))
            img = screen(img, (smooth(640, 1000, self.y) * .07)[..., None] * rgb('#ffdca0'))
        dx, dy = shift(D_BOY); (L1, L2), hc = self.boys(f, dx * .75, dy)
        for L in (L1, L2):
            L = para(L, (0, 500 + dy), (0, 1080), [(0, '#ffffff'), (.5, '#c9c2c6'), (1, '#6a6f86')]); L = rim(L, -7, 9, '#ffc978', .6, 4.); L = rim(L, -3.6, 4.0, '#fff0c4', 1., .7)
            img = over(img, L)
        bn = HAT.settle(f, 5, .10) + HAT.settle(f, 35, .10) + .05 * math.sin(t * .33)
        img = over(img, HAT.hat(hc[0], hc[1], 432, yaw=math.pi + .22, pitch=-.36, roll=.05, light='dusk', wear=.25, bounce=bn, f=f, wind=.8))
        d = PLANES[6][0]; a, b = shift(d)
        for sel in (0, 1): img = SC.put8(img, self.pl[12 + sel], -160 + a, self.oy + b + 3 * math.sin(t * .17 + sel * 3))
        # light through dust
        q = np.clip((self.x - 1010) * .45 + (self.y - 330), -4000, 4000); beam = np.exp(-((self.x - 1080 + (self.y - 330) * .62) / 190) ** 2) * smooth(330, 520, self.y) * (1 - smooth(760, 1080, self.y) * .6); beam2 = np.exp(-((self.x - 1260 - (self.y - 330) * .52) / 150) ** 2) * smooth(330, 560, self.y) * (1 - smooth(760, 1080, self.y) * .7)
        img = screen(img, (beam * .16 + beam2 * .12)[..., None] * rgb('#ffdf9f')); img = img + motes(f, 90, seed=3, light=self.ml, col='#ffeec4', drift=(26., -14.), size=1.2, amt=.55)
        zoom = 1 + .015 * th
        if th > .002: img = cam2d(img, 960 + 0, 540, zoom) if False else cv2.warpAffine(img.astype(f32), np.float32([[zoom, 0, 1010 * (1 - zoom)], [0, zoom, 470 * (1 - zoom)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        return img
