"""b2: the pier at Foosha, late sun.  A hand, a hat, a child.  He rises and walks past the lens; the cloak wipes the frame."""
import numpy as np, math, cv2
from op import W, H, comp, clouds as CL, ships as SH, sea as SEA, hands as HD
from op.cam import Camera3D
from op.paint import rgb, f32, smooth, grid, ramp, fbm, repaint, mottle, Canvas
from op.comp import over, screen, blur, bokeh, lerp_keys, ease, ease_in, ease_out, radial, cam2d
from op.ink import Cel, spl
from op.kit import para, outline, rim, flare, shifted, spike, smear
from op import hat as HAT
from op.fx_air import motes
from chars import shanks as SK, scaffold as SC
from chars._old import *

HZ, U, GY = 612., 430., 946.                    # horizon, px per metre at the figures, their ground line
OX, BX = 772., 1062.                            # Shanks / the boy
Z = 1.0                                         # push-in of the figure plane this frame (set in frame)
U0, GY0, OX0, BX0, ZX = U, GY, OX, BX, 890.


def dolly(z):
    """walk the camera in: the figures' plane scales about (ZX, horizon)"""
    global U, GY, OX, BX, Z
    Z = z; U = U0 * z; GY = HZ + (GY0 - HZ) * z; OX = ZX + (OX0 - ZX) * z; BX = ZX + (BX0 - ZX) * z


def path(f):
    """his walk to the lens from 45: one distance, so size and place run on from where he stood.  -> ox, oy, px per metre"""
    t = f - 45.; d = .085 * t * t / 6. if t <= 3 else .085 * (t - 1.5); u = U / (1 - min(max(d, 0.), .88)); e = float(np.clip((f - 46) / 8., 0, 1)); return ZX + (-.2244 - .28 * e * e * (3 - 2 * e)) * u, HZ + (GY0 - HZ) * u / U0, u


def zw(L, z, border=cv2.BORDER_REFLECT_101):
    return cv2.warpAffine(np.ascontiguousarray(L, f32), np.float32([[z, 0, ZX * (1 - z)], [0, z, HZ * (1 - z)]]), (W, H), flags=cv2.INTER_LINEAR, borderMode=border)

SUN = (948., 566.)
PAL = LIGHT['sunset']
TEAL = dict(SEA.SUNSET, deep='#1d4660', body='#2f6d7a', mid='#6f9c90', light='#f0b97c', sky='#ffe3a0', crest='#58a294', foam='#fff3dc', foamsh='#b9b294', far='#f2c48c', ink='#123048')


def stage(L, x0=.0):
    return L


class Shot:
    id = 'b2'; n = 60; finish = dict(bloom=.36, vig=.18, soft=.18)
    EVENTS = [(0, 'sea_calm', dict(note='pier water, late gulls, rigging of the big ship', len=60)), (0, 'windmill', dict(note='slow sails', len=60)), (0, 'cloth', dict(note='empty sleeve + cloak', len=60)), (15, 'press', dict(size=.8, note='hat pressed on: soft don, straw crackle, knees give')), (22, 'sob', dict(note='child, one hitch')), (30, 'sob', dict(note='child')), (37, 'cloth', dict(size=.5, note='he stands')), (48, 'step', dict(size=.5, note='sandal on planks')), (53, 'step', dict(size=.8)), (22, 'sleeve', dict(size=.3, note='empty sleeve cracks in the wind')), (43, 'sleeve', dict(size=.4)), (46, 'cloth', dict(size=.5, note='cloak comes round as he turns to the lens')), (53, 'cloak_wipe', dict(size=.9, note='heavy cloak hem sweeps past the lens', len=7))]

    def setup(self):
        self.x, self.y = grid(); self.bg = np.asarray(SC.cached('b2_bg', self._bg), f32); self.cam = Camera3D((0, 0, 1.95), (0, 10, 1.95 + 10 * (HZ - 540) / 1250.), 1250.); self.S = SEA.Sea(5, hs=.35, length=14); self.ml = radial(SUN[0], SUN[1], 1400, 900, 1.); self.pier = np.asarray(SC.cached('b2_pier', self._pier), f32); self._hands = {}
        ps = SH.pose(27., 47., -.2, math.pi * .90, .01, .0, 1.0); self.ps = ps

    def _bg(self):
        sky = ramp(self.y / HZ, [(0, '#2f8f98'), (.30, '#63b3ac'), (.60, '#c9cfa0'), (.82, '#f7c98a'), (1., '#ffdf9e')]); d = np.hypot(self.x - SUN[0], (self.y - SUN[1]) * 1.3); sky = screen(sky, np.exp(-(d / 420) ** 1.2)[..., None] * rgb('#ffcf7e') * .9); S = CL.Sky(H, W, sky, light=(.1, .55, -.5), seed=3)
        S.bank(1500, 600, 1300, 520, ('#ffe9bd', '#f2b48c', '#c98a86', '#9a7486'), '#f6cf9c', seed=15, profile=CL.peak(.55, .22, .35), lit=.16, half=.30, deep=.16, rim='#fff2cf', rim_amt=1., light=(-.75, .25, -.2)); S.bank(260, 600, 1100, 420, ('#ffe6b8', '#eeb48e', '#c08a88', '#917488'), '#f4cb98', seed=22, profile=CL.peak(.35, .25, .35, .8, .5), lit=.16, half=.30, deep=.16, rim='#fff0c8', rim_amt=1., light=(.75, .25, -.2)); S.bank(960, 606, 2100, 120, ('#ffedc4', '#f6c795', '#dba08c', '#b98c90'), '#ffd9a2', seed=31, lit=.2, half=.3, deep=.1)
        S.strata(60, 360, 30, ('#8fc4b8', '#e9c9a0', '#f2b690'), seed=8, w=(.1, .34), th=(1.5, 4.), op=(.25, .6)); im = S.paint(seed=5, amt=.8); g = np.exp(-(d / 60) ** 4); im = im * (1 - g[..., None]) + rgb('#fffbe6') * 1.25 * g[..., None]
        C = Cel(); r = np.random.default_rng(4)
        C.fill(spl(A((-40, 640), (-40, 470), (90, 420), (240, 408), (360, 448), (470, 520), (560, 590), (620, 640)), True, (0, 7)), '#4f8a66'); C.fill(spl(A((170, 412), (240, 408), (360, 448), (470, 520), (560, 590), (620, 640), (330, 640), (300, 520)), True), '#8fae6a'); C.fill(spl(A((-40, 560), (120, 540), (300, 580), (420, 640), (-40, 640)), True, (0, 4)), '#386a58')
        for i in range(120): x, y = r.uniform(-30, 560), r.uniform(440, 640); h_ = 420 + max(0, x - 240) * .55; s = r.uniform(7, 20); (y > h_ + 10) and C.fill(spl(A((x - s, y + s * .4), (x - s * .5, y - s * .5), (x + s * .3, y - s * .6), (x + s, y + s * .4)), True), ('#2f6050', '#6f9c62', '#a9bd74')[int(r.integers(3))], .8)
        for i in range(12): x = 60 + i * 38 + r.uniform(-10, 10); y = 560 + r.uniform(-24, 34); w = r.uniform(26, 44); C.fill(np.array([(x, y), (x + w, y), (x + w, y - 24), (x, y - 24)]), '#efd2a2'); C.fill(np.array([(x - 5, y - 22), (x + w * .5, y - 44), (x + w + 5, y - 22)]), ('#b45a44', '#9a4c46')[i % 2]); C.fill(np.array([(x, y), (x + w * .4, y), (x + w * .4, y - 24), (x, y - 24)]), '#b98a78')
        for x, y, s in self.MILLS: C.fill(np.array([(x - 20 * s, y), (x - 12 * s, y - 84 * s), (x + 12 * s, y - 84 * s), (x + 20 * s, y)]), '#b98f80'); C.fill(np.array([(x + 4 * s, y), (x + 4 * s, y - 84 * s), (x + 12 * s, y - 84 * s), (x + 20 * s, y)]), '#ffe2ae'); C.fill(np.array([(x - 16 * s, y - 82 * s), (x, y - 108 * s), (x + 16 * s, y - 82 * s)]), '#8a4a44'); C.fill(np.array([(x - 5 * s, y), (x - 5 * s, y - 20 * s), (x + 5 * s, y - 20 * s), (x + 5 * s, y)]), '#5c4048')
        L = C.layer(); hz = (smooth(420, 620, self.y) * .30)[..., None] * L[..., 3:4]; L[..., :3] = L[..., :3] * (1 - hz) + rgb('#f6cf98') * hz; L = rim(L, -2, 4, '#fff0c0', .7, 1.); return np.clip(over(im, L), 0, 1.4)

    MILLS = ((150, 428, 1.), (268, 418, .8), (372, 470, .6))

    def _pier(self):
        """planks running to the horizon; posts, rope, a crate on the back edge"""
        C = Cel(W, 560, 0, 520); r = np.random.default_rng(9); K = U / (GY - HZ) * 1.0; by = 770.; px = lambda X, y: 960 + X * (y - HZ) * 1.29
        ks = np.arange(-12, 13) * .24
        for i, X in enumerate(ks):
            a, b = X + .006, X + .234; c = mix3(('#b98452', '#a8744a', '#c79260', '#9a6a46')[int(r.integers(4))], '#7a4a48', r.uniform(0, .25)); C.fill(np.array([(px(a, by), by), (px(b, by), by), (px(b, 1100), 1100), (px(a, 1100), 1100)]), c)
            for j in range(9): u = r.uniform(a + .01, b - .01); y0 = by + (1090 - by) * r.random() ** 2; y1 = min(1095, y0 + r.uniform(30, 170)); C.line(np.array([(px(u, y0), y0), (px(u + r.normal(0, .004), (y0 + y1) / 2), (y0 + y1) / 2), (px(u, y1), y1)]), r.uniform(1., 2.6), mix3(c, '#3a2020', .55), taper=(.4, .5), press=.5, seed=i * 11 + j, alpha=.8)
            if r.random() < .5: y0 = r.uniform(by + 20, 1060); C.line(np.array([(px(a, y0), y0), (px(b, y0 + 1), y0 + 1)]), 3., '#3a1f20', taper=(.05, .05)); 
            for y0 in (by + 10, 1030): C.fill(np.c_[px(X + .12, y0) + 3 * np.cos(np.linspace(0, 6, 8)), y0 + 2 * np.sin(np.linspace(0, 6, 8))], '#3a2422')
            C.line(np.array([(px(b, by), by), (px(b, 1100), 1100)]), 4.2, '#341c20', taper=(0, 0), press=.5, seed=i); C.line(np.array([(px(a, by) + 1, by), (px(a, 1100) + 3, 1100)]), 2.2, '#f2c68a', taper=(0, .1), press=.6, seed=i, alpha=.55)
        C.fill(np.array([(0, by - 2), (W, by - 2), (W, by + 8), (0, by + 8)]), '#5a3432'); C.line(np.array([(0, by - 2), (W, by - 2)]), 3.5, '#ffdca0', taper=(0, 0))
        for X in (150, 520, 1400, 1790):
            S = np.array([(X - 22, by + 20), (X - 20, by - 78), (X - 14, by - 86), (X + 14, by - 86), (X + 20, by - 78), (X + 22, by + 20)]); C.fill(S, '#7c5238'); C.fill(np.array([(X + 2, by + 20), (X + 2, by - 84), (X + 14, by - 86), (X + 20, by - 78), (X + 22, by + 20)]) if X < 950 else np.array([(X - 2, by + 20), (X - 2, by - 84), (X - 14, by - 86), (X - 20, by - 78), (X - 22, by + 20)]), '#4a2c2a'); outline(C, S, 4., '#231014', int(X)); C.fill(spl(A((X - 18, by - 84), (X, by - 92), (X + 18, by - 84), (X, by - 78)), True), '#c79262')
            for k in range(4): C.line(spl(A((X - 23, by - 54 + k * 10), (X, by - 47 + k * 10), (X + 23, by - 54 + k * 10))), 7.5, '#d8b878', taper=(.05, .05), seed=k); C.line(spl(A((X - 23, by - 50 + k * 10), (X, by - 43 + k * 10), (X + 23, by - 50 + k * 10))), 2., '#4a2f20', seed=k)
        for a, b in ((150, 520), (1400, 1790)): C.line(spl(A((a, by - 48), ((a + b) / 2, by + 4), (b, by - 48))), 8, '#d0ac70', taper=(.02, .02), seed=1); C.line(spl(A((a, by - 44), ((a + b) / 2, by + 8), (b, by - 44))), 2.4, '#4a2f20', taper=(.02, .02), seed=2)
        L = C.layer(); m = np.zeros((H, W, 4), f32); L2 = L.copy(); gl = np.exp(-((self.x - SUN[0] - (self.y - HZ) * .0) / (110 + (self.y - by) * .9)) ** 2) * smooth(1120, by - 10, self.y); L2 = para(L2, (960, 770), (960, 1080), [(0, '#fff0cf'), (.4, '#e8bc98'), (1, '#8f6a72')]); L2[..., :3] += L2[..., 3:4] * (gl * .30)[..., None] * rgb('#ffcf80'); n = fbm(H, W, (50, 14, 5), (1, .6, .4), 4)[..., None]; L2[..., :3] *= 1 + .035 * n; return L2

    # ------------------------------------------------------------ figures
    def child(self, f):
        """small boy, 3/4 back, hat on (from f 12): hunched, fists, the head going down under the hat's weight, sobs that jump the shoulders"""
        t = f // 2 * 2 if not 10 <= f <= 22 else f; c = float(lerp_keys(t, [(0, .0), (13, -.04), (15, .9), (16, 1.1), (19, .55), (22, .72), (35, .70), (38, -.06), (42, .12), (59, .08)])); hun = float(lerp_keys(t, [(0, .42), (9, .34), (13, .30), (15, 1.05), (16, 1.22), (19, .86), (23, 1.), (59, 1.)])); self._hun = hun if f >= 15 else max(0., hun - .3)
        ph_ = (t - 18) % 10; sob = {0: .050, 2: .036, 4: .010, 6: -.010, 8: -.004}[ph_ if ph_ in (0, 2, 4, 6, 8) else 8] * (1. if t < 46 else .7) if t >= 18 else .007 * ((t // 4) % 2)
        C = Cel(); u, ox, oy = U, BX, GY; q = lambda L: np.array([(ox + a * u, oy - b * u) for a, b in L]); ln = PAL['line']; dn = -.050 * c; sh = dn + sob; up = .030 * hun; w = lambda i, a: a * math.sin(t * .6 + i); lx = lambda y: -.038 * hun * float(np.clip((y - .50) / .45, 0, 1.2)); sk, sd_ = PAL['skin'], PAL['shade']; tee, tee2, tee3 = '#e4ebea', '#94a7c2', '#6c7fa4'
        def seg(a, b, wa, wb, col, cs, sd, lw_=3.4, bul=.0):
            """round limb: no line across its ends, so joints flow into each other"""
            a, b = A(*a), A(*b); d = b - a; ax = d / (np.hypot(*d) + 1e-6); n = A(-ax[1], ax[0]); mw = (wa + wb) / 2; md = (a + b) / 2; S = spl(q([a - ax * wa * .55, a + n * wa, md + n * (mw + bul), b + n * wb, b + ax * wb * .7, b - n * wb, md - n * (mw + bul * .4), a - n * wa]), True); C.fill(S, col); C.clip(S); m = 1 if n[0] > 0 else -1; C.fill(spl(q([a + n * wa * .10 * m - ax * wa, md + n * m * wa * .24, b + n * m * wb * .05 + ax * wb, b + n * m * wb * 2.4, a + n * m * wa * 2.4]), True, (0, 2, 3, 4)), cs); C.unclip()
            for e, bb in ((1, bul), (-1, bul * .4)): C.line(spl(q([a + n * wa * e * .99, md + n * (mw + bb) * e, b + n * wb * e])), lw_, ln, taper=(.20, .16), press=.3, seed=sd + e)
            return S
        for sg in (1, -1):                                                                  # knock-kneed, toes in, knees giving
            hp_, kn, an = (sg * .078, .47 + dn), (sg * (.072 - .012 * hun + .010 * c) - .010 * c, .255 + dn * .55), (sg * .094, .055); seg(kn, an, .052, .038, sk, sd_, 3 + sg, 3.6, .012); seg(hp_, kn, .066, .054, sk, sd_, 6 + sg, 3.6, .004)
            C.line(spl(q([(kn[0] - .034, kn[1] + .012), (kn[0], kn[1] - .006), (kn[0] + .034, kn[1] + .010)])), 2.6, ln, taper=(.3, .4), seed=2); C.line(spl(q([(kn[0] - .016, kn[1] - .03), (kn[0] - .010, kn[1] - .075)])), 2.0, ln, taper=(.2, .7), seed=3); C.line(spl(q([(kn[0] + .020, kn[1] - .03), (kn[0] + .016, kn[1] - .07)])), 2.0, ln, taper=(.2, .7), seed=4); C.line(spl(q([(an[0] - .02, .10), (an[0] - .012, .05)])), 2.2, ln, taper=(.3, .6), seed=5)
            F = spl(q([(an[0] - .050, .050), (an[0] - .052, .004), (an[0] - .02 - .05 * sg, -.016), (an[0] + .02 - .05 * sg, -.016), (an[0] + .052, .004), (an[0] + .048, .050), (an[0], .062)]), True, (1, 4)); C.fill(F, sk if sg < 0 else sd_); outline(C, F, 3.2, ln, 8); C.fill(q([(an[0] - .060, .008), (an[0] - .03 - .05 * sg, -.028), (an[0] + .03 - .05 * sg, -.028), (an[0] + .060, .008), (an[0] + .05, -.004), (an[0] - .05, -.004)]), '#4a2e22'); C.line(q([(an[0] - .05, .042), (an[0] + .05, .030)]), 4.5, '#5a3826', taper=(.05, .05), seed=9)
        S = spl(q([(-.158, .625 + dn), (-.196, .42 + dn), (-.186, .315 + dn), (-.010, .300 + dn), (.0, .42 + dn), (.016, .300 + dn), (.190, .315 + dn), (.200, .42 + dn), (.165, .625 + dn)]), True, (0, 2, 3, 4, 5, 6, 8)); C.fill(S, '#41639a'); C.clip(S); C.fill(spl(q([(.02, .66), (.0, .48), (.05, .28), (.3, .28), (.3, .66)]), True), '#26396a'); C.fill(q([(-.2, .345 + dn), (.2, .345 + dn), (.2, .29 + dn), (-.2, .29 + dn)]), '#2c4274', .55)
        for i, L in enumerate(([(-.15, .56), (-.10, .48), (-.14, .38)], [(-.06, .60), (-.03, .50), (-.05, .42)], [(.07, .60), (.05, .50), (.08, .40)], [(.15, .56), (.11, .48), (.15, .36)], [(-.11, .36), (-.06, .40), (-.02, .44)], [(.12, .36), (.06, .40), (.02, .44)])): C.line(spl(q([(a, b + dn) for a, b in L])), 2.4, ln, taper=(.2, .7), press=.4, seed=20 + i)
        C.unclip(); outline(C, S, 3.8, ln, 6)
        T_ = [(-.152, .585 + dn), (-.172 + lx(.7), .715 + sh), (-.186 + lx(.84), .835 + sh + up), (-.148 + lx(.9), .902 + sh + up * 1.2), (-.07 + lx(.93), .930 + sh + up), (.08 + lx(.93), .930 + sh + up), (.152 + lx(.9), .900 + sh + up * 1.2), (.190 + lx(.84), .830 + sh + up), (.178 + lx(.7), .712 + sh), (.168 + w(0, .012), .555 + dn), (.06, .575 + dn), (-.04, .548 + dn)]; S = spl(q(T_), True, (0, 9)); C.fill(S, tee); C.clip(S); C.fill(spl(q([(.05 + lx(.9), .98 + sh), (.10 + lx(.8), .80 + sh), (.02, .68 + sh), (.05, .52), (.4, .5), (.4, 1.)]), True), tee2); C.fill(spl(q([(-.15 + lx(.9), .90 + sh + up), (-.03 + lx(.9), .865 + sh), (.10 + lx(.9), .90 + sh + up), (.03 + lx(.8), .80 + sh), (-.03 + lx(.8), .78 + sh)]), True, (0, 2)), tee2); C.fill(spl(q([(.12, .95 + sh), (.17, .75 + sh), (.12, .55), (.4, .5), (.4, 1.)]), True), tee3, .8)
        for i, L in enumerate(([(-.165, .86), (-.09, .83), (-.035, .77)], [(-.175, .80), (-.10, .775), (-.05, .72)], [(.165, .86), (.09, .83), (.02, .77)], [(.175, .80), (.11, .77), (.05, .71)], [(-.015, .84), (-.02, .74), (-.03, .64)], [(-.13, .66), (-.06, .62), (-.09, .57)], [(.13, .66), (.08, .62), (.11, .57)], [(-.10, .60), (-.04, .585)])): C.line(spl(q([(a + lx(b), b + sh) for a, b in L])), 2.6, ln, taper=(.2, .7), press=.45, seed=8 + i)
        edge_hatch(C, spl(q([(.10 + lx(.8), .80 + sh), (.02, .68 + sh), (.05, .56 + dn)])), 10, 24, 2.7, 1.6, ln, 41); C.unclip(); outline(C, S, 3.8, ln, 7)
        for sg in (-1, 1):                                                                  # elbows out, fists hard against the thighs
            tr = .007 * math.sin(f * 2.4 + sg) * (f > 12); s0 = (sg * .162 + lx(.85), .862 + sh + up); el = (sg * (.232 + .010 * hun) + lx(.7), .690 + sh + up * .5); wr = (sg * .196 + tr + lx(.5), .545 + sh * .7 + up * .6); seg(el, wr, .046, .038, sk if sg < 0 else mix3(sk, sd_, .45), sd_, 12 + sg, 3.4, .008); seg(s0, el, .060, .046, sk if sg < 0 else mix3(sk, sd_, .45), sd_, 10 + sg, 3.4)
            sl = A((el[0] - s0[0]), (el[1] - s0[1])); sl = sl / np.hypot(*sl); sn = A(-sl[1], sl[0]) * (1 if sg < 0 else -1); hm = A(*s0) + sl * .118; nk = A(sg * .085 + lx(.92), .922 + sh + up); S = spl(q([nk, A(*s0) + A(sg * .050, .030), hm + A(sg * .078, .012), hm + A(sg * .030, -.030), hm - A(sg * .058, .030), A(*s0) - A(sg * .055, .075)]), True, (2, 4)); C.fill(S, tee if sg < 0 else tee2); C.clip(S); C.fill(spl(q([A(*s0) + A(.015, .05), hm + A(.02, 0), hm + A(.0, -.06), hm + A(.16, -.06), A(*s0) + A(.14, .06)]), True), tee2 if sg < 0 else tee3); C.unclip(); C.line(spl(q([nk, A(*s0) + A(sg * .050, .030), hm + A(sg * .078, .012)])), 3.6, ln, taper=(.3, .08), seed=30 + sg); C.line(spl(q([hm + A(sg * .078, .012), hm + A(sg * .030, -.028), hm - A(sg * .058, .030)])), 3.2, ln, taper=(.1, .2), seed=32 + sg); C.line(spl(q([A(*s0) + A(sg * .03, -.01), A(*s0) + sl * .05 - A(sg * .01, 0), hm - A(sg * .02, -.01)])), 2.2, ln, taper=(.3, .7), seed=34); hx_, hy_ = wr[0] - sg * .004, wr[1] - .055; H_ = spl(q([(hx_ - .050, hy_ + .044), (hx_ - .062, hy_ - .004), (hx_ - .046, hy_ - .052), (hx_ + .006, hy_ - .062), (hx_ + .050, hy_ - .046), (hx_ + .060, hy_ + .004), (hx_ + .044, hy_ + .046), (hx_, hy_ + .056)]), True); C.fill(H_, sk if sg < 0 else mix3(sk, sd_, .5)); C.clip(H_); C.fill(spl(q([(hx_ + .0, hy_ + .07), (hx_ + .012, hy_), (hx_ - .004, hy_ - .07), (hx_ + .09, hy_ - .07), (hx_ + .09, hy_ + .07)]), True), sd_); C.unclip(); outline(C, H_, 3.4, ln, 16 + sg)
            for k in range(3): x_ = hx_ - .028 + .028 * k; C.line(spl(q([(x_, hy_ - .056), (x_ + .004, hy_ - .030), (x_ - .002, hy_ - .008)])), 2.4, ln, taper=(.1, .6), seed=k)
            C.line(spl(q([(hx_ - sg * .052, hy_ + .006), (hx_ - sg * .018, hy_ + .018), (hx_ + sg * .012, hy_ + .004)])), 2.8, ln, taper=(.2, .4), seed=5); C.line(spl(q([(wr[0] - .03, wr[1] + .012), (wr[0] + .01, wr[1] + .018)])), 2.2, ln, taper=(.3, .5), seed=6)
        hx0 = lx(1.0) - .012 * hun; hy = 1.0 + dn + sob * .22 - .052 * hun; h_ = lambda a, b: (a + hx0, hy + b); S = spl(q([h_(-.158, .06), h_(-.156, -.05), h_(-.112, -.125), h_(-.03, -.150), h_(.07, -.138), h_(.138, -.07), h_(.152, .06), h_(.0, .16)]), True); C.fill(S, sk); C.clip(S); C.fill(spl(q([h_(.0, .2), h_(-.02, 0), h_(-.05, -.16), h_(.3, -.2), h_(.3, .2)]), True), sd_); C.unclip(); outline(C, S, 3.8, ln, 16); C.line(spl(q([h_(-.152, -.045), h_(-.135, -.085), h_(-.105, -.105)])), 2.4, ln, taper=(.4, .5), seed=19)
        E = spl(q([h_(.068, .02), h_(.112, .032), h_(.118, -.02), h_(.088, -.058), h_(.066, -.03)]), True); C.fill(E, sk); outline(C, E, 3., ln, 17); C.line(spl(q([h_(.084, .012), h_(.100, -.008), h_(.086, -.032)])), 2., ln)
        tp_ = .25 if f < 13 else .13; HT = spl(q([h_(-.17, .02), h_(-.185, .12 * tp_ * 4), h_(-.12, tp_ * .88), h_(0, tp_), h_(.13, tp_ * .84), h_(.185, .4 * tp_), h_(.165, .0), h_(.08, .03), h_(0, -.03), h_(-.08, .02)]), True, (0, 6)); C.fill(HT, '#1c1a22'); outline(C, HT, 3., ln, 18); hp = dict(PAL, hair='#1c1a22', hair2='#0e0d13', hl='#3c3946', hi='#4a4a60')
        for i, (a, tp, b) in enumerate((((-.16, .08), (-.180, -.07), (-.09, .07)), ((-.10, .08), (-.105, -.115), (-.02, .06)), ((-.04, .07), (-.025, -.150), (.05, .06)), ((.02, .07), (.050, -.125), (.09, .06)), ((.09, .08), (.148, -.065), (.16, .06)), ((-.165, .10), (-.212, .0), (-.13, .12)), ((.12, .10), (.208, .0), (.165, .12)), ((-.07, .06), (-.066, -.13), (-.01, .05)))): clump(C, q([h_(*a)])[0], q([h_(tp[0] + w(i, .008), tp[1])])[0], q([h_(*b)])[0], hp, .62, 30 + i, .2, .06 * (-1) ** i, 2, .4, edge=ln)
        if f > 18:                                                                          # tears fall from the hidden face
            for k in range(4):
                p = ((f - 18) + k * 6) % 23; a = q([h_(-.150 - .006 * k, -.09)])[0] + A(-p * 1.5, .9 * p * p * .50); sz = 1.3 * Z
                if p < 17: C.fill(spl([a + A(0, -9 * sz), a + A(4.5 * sz, 2 * sz), a + A(0, 6 * sz), a + A(-4.5 * sz, 2 * sz)], True, (0,)), '#fff6dc'); C.fill(spl([a + A(1.5 * sz, -2 * sz), a + A(3.2 * sz, 2 * sz), a + A(1. * sz, 4 * sz)], True), '#9cc4d8')
        return C.layer(), (ox + hx0 * u, oy - (hy + .128) * u), c

    def walker(self, f):
        """him, turned to the lens, black cloak, growing out of the frame"""
        ox, oy, u = path(f); ph = (f - 48) * math.pi / 5.; bob = .03 * abs(math.sin(ph)); C = Cel(); q = lambda L: np.array([(ox + a * u, oy - (b + bob) * u) for a, b in L]); ln = PAL['line']; w = lambda i, a: a * math.sin(f * .55 - i * .9); lw = min(6., 3.6 * (u / U) ** .4)
        for sg in (-1, 1):
            sw = math.sin(ph + (0 if sg > 0 else math.pi)); S = spl(q([(sg * .14 - .10, .92), (sg * .13 - .09, .5 + .04 * sw), (sg * .13 - .07, .08 - bob + .08 * max(0, sw)), (sg * .13 + .07, .08 - bob + .08 * max(0, sw)), (sg * .13 + .09, .5 + .04 * sw), (sg * .14 + .10, .92)]), True, (2, 3)); C.fill(S, '#57433a' if sw > 0 else '#3a2c2a'); outline(C, S, lw, ln, 3); F = spl(q([(sg * .13 - .07, .09 - bob + .08 * max(0, sw)), (sg * .13 - .08, -bob + .06 * max(0, sw)), (sg * .13 + .08, -bob + .06 * max(0, sw)), (sg * .13 + .07, .09 - bob + .08 * max(0, sw))]), True, (1, 2)); C.fill(F, PAL['shade']); outline(C, F, lw, ln, 5)
        S = spl(q([(-.16, 1.56), (.16, 1.56), (.18, 1.2), (.17, .86), (-.17, .86), (-.18, 1.2)]), True, (0, 1, 3, 4)); C.fill(S, '#c9ab94'); C.fill(q([(-.09, 1.58), (.09, 1.58), (.01, 1.18)]), PAL['shade']); C.fill(spl(q([(-.18, 1.0), (.18, 1.0), (.19, .88), (-.19, .88)]), True, (0, 1, 2, 3)), '#7a2422'); C.clip(S)
        for i in range(9): x0 = -.17 + .042 * i; C.line(spl(q([(x0, 1.5 - .02 * i), (x0 + .03, 1.25), (x0 + .01 + w(i, .006), .98)])), max(2., lw * .6), ln, taper=(.2, .7), press=.4, seed=20 + i)
        C.fill(spl(q([(.03, 1.6), (.08, 1.3), (.04, .9), (.3, .9), (.3, 1.6)]), True), '#8a6a66', .6); C.unclip(); C.fill(spl(q([(-.19, .985), (.0, .955), (.19, .985), (.19, .90), (0, .87), (-.19, .90)]), True, (0, 2, 3, 5)), '#93302a'); C.line(spl(q([(-.19, .945), (0, .915), (.19, .945)])), lw * .7, ln); C.fill(spl(q([(.10, .95), (.17, .92), (.15, .62), (.09, .56), (.08, .7)]), True, (0, 3)), '#7a2422'); C.line(q([(-.30, .98), (-.02, .80)]), lw * 2.2, '#c9a04a', taper=(.1, .1)); C.line(q([(-.34, 1.0), (-.28, .965)]), lw * 5, '#3a2a30', taper=(.1, .1)); outline(C, S, lw, ln, 6)
        for sg in (-1, 1):
            E = [(sg * .06, 1.66), (sg * .20, 1.70), (sg * .34, 1.55), (sg * .44, 1.15), (sg * (.50) + w(1, .02), .70), (sg * .58 + w(2, .04), .28 + w(2.4, .02)), (sg * .40 + w(3, .03), .22 + w(3, .02)), (sg * .13 + w(4, .03), .27), (sg * .06 + w(4.4, .02), .7), (sg * .05, 1.15), (sg * .10, 1.5)]; S = spl(q(E), True, (0, 5, 7)); C.fill(S, '#1d1822'); C.clip(S)
            for i in range(4): x = sg * (.20 + .07 * i); P = spl(q([(x * .8, 1.58), (x * 1.05, 1.1), (x * 1.2 + w(i + 2, .02), .6), (x * 1.35 + w(i + 3, .03), .2)]), n=10); C.line(P, .05 * u, '#0f0c14', taper=(.6, .1), press=.3, seed=i); C.line(P + A(.022 * u, 0), max(2., lw * .6), '#4a3d4a', taper=(.3, .4), seed=9 + i)
            C.unclip(); outline(C, S, lw * 1.2, ln, 8)
        wv = lambda i, a: a * (math.sin(f * .92 - i) + .5 * math.sin(f * 1.9 - i * 1.8)); S = spl(q([(.36, 1.44), (.50, 1.40 + wv(1, .03)), (.66, 1.36 + wv(2, .05)), (.84, 1.38 + wv(3, .08)), (.82, 1.25 + wv(3.2, .08)), (.64, 1.22 + wv(2.2, .05)), (.48, 1.18 + wv(1.2, .03)), (.38, 1.22)]), True, (0, 3, 4, 7)); C2 = Cel(); C2.fill(S, '#e6d3bb'); outline(C2, S, lw, ln, 44); C2.line(spl(q([(.44, 1.36), (.60, 1.31 + wv(1.6, .04)), (.72, 1.31 + wv(2.4, .05))])), lw * .7, ln, taper=(.2, .7))
        L = over_l(C2.layer(), C.layer()); L = over_l(L, self.sweep(f)) if f <= 49 else L; hx, hy = q([(.0, 1.80)])[0]
        if hy > -900 and u < 2400:
            LY = SK.draw(hx, hy, .150 * u, float(lerp_keys(f, [(46, .08), (51, -.16)])), -.04, .03, expr='warm', light='sunset', f=f, hat=False, side=-1, lw=min(2.6, 2.6 * 430 / u * 1.6 + .6), rimdir=(-1., .25)); L = over_l(over_l(over_l(LY['back'], L), LY['face']), LY['hair'])
        return (smear(L, {47: 30., 48: 22., 49: 12.}[f], -3., 9) if f <= 49 else L), u

    def sweep(self, f):
        """45-49: the cloak comes round with his turn, across the body from his back to his left side, hem flung out, lining at the leading edge"""
        K = lambda *v: float(np.interp(f, (45, 46, 47, 48, 49), v)); lead, trail, lh, th, tr = K(.0, .24, .58, .70, .64), K(-.66, -.66, -.58, -.02, .10), K(.24, .34, .40, .44, .32), K(.16, .15, .18, .25, .27), K(.0, .14, .24, .30, .32); tx = -.22 + .26 * float(np.clip((trail + .58) / .6, 0, 1)); ox, oy, u = path(f); C = Cel(); ln = PAL['line']; q = lambda L: np.array([(ox + a * u, oy - b * u) for a, b in L])
        v = np.linspace(0, 1, 11); hem = [tuple(p) for p in np.c_[lead + (trail - lead) * v, lh + (th - lh) * v + .030 * np.sin(v * 13 + f * 1.7) - .05 * np.sin(np.pi * v)]]; edge = [(tr, 1.64), (tr + .02 + lead * .16, 1.36), (.02 + lead * .80, .95), (lead * 1.04, .62)]; back = [(trail * .86, .62), (trail * .66 + tx * .2, 1.05), (tx - .05, 1.42)]
        S = spl(q([(tx, 1.62), (tx + (tr - tx) * .5, 1.68)] + edge + hem + back), True, (0, 2, 6, 16)); C.fill(S, '#1d1822'); C.clip(S)
        for i, k in enumerate(np.linspace(.10, .92, 7)): h = hem[int(round(k * 10))]; a = (tx + (tr - tx) * (1 - k), 1.58); P = spl(q([a, (a[0] * .5 + h[0] * .5 + .05 * lead, 1.58 * .5 + h[1] * .5), h]), n=10); C.line(P, .050 * u, '#0f0c14', taper=(.7, .1), press=.3, seed=i); C.line(P + A(.022 * u, 0), 2.6, '#4a3d4a', taper=(.35, .3), seed=9 + i)
        I = [(edge[1][0] - .01, 1.36), (edge[2][0] - .05, .95), (edge[3][0] - .085, .62), (lead - .11, lh + .02)]; C.fill(spl(q(edge[1:] + hem[:1] + I[::-1]), True, (0, 3, 4)), '#4b1d22'); C.line(spl(q(I)), 4., '#0a070c', taper=(.4, .1), seed=31)
        C.unclip(); outline(C, S, 4.4, ln, 8); return smear(C.layer(), K(6., 26., 58., 40., 14.) * u / 450., K(0., -4., -8., -6., 0.), 17)

    def shadow(self, img, L, amt=.42):
        """long shadow thrown to the lens (sun behind): the figure's alpha flipped about its ground line"""
        s = cv2.warpAffine(L[..., 3], np.float32([[1, -.10, .10 * GY], [0, -.62, GY * 1.62]]), (W, H)); s = blur(s, 5) * smooth(GY - 12, GY + 10, self.y); return img * (1 - s[..., None] * amt * rgb('#7a9a9a') ** .2) + rgb('#3a1c38') * s[..., None] * amt * .35

    def hand_on(self, x, y, rot):
        """right hand resting on the crown, seen from outside"""
        zk = round(Z * 25) / 25.; key = (round(rot, 1), zk)
        if key not in self._hands:
            a = key[0]; y_ = A(.90, -.40 - a * .3, .10); zb = A(.36, .86, .34); from shots.d7 import frame_of
            h = HD.Hand(HD.POSES['rest'], 'R', frame_of(y_, zb), (0, 0, 0), HD.MALE, pivot=(0, .6, 0), arm=.30); cam = HD.Cam(200, 160, 62 * zk, 8); g = HD.render([h], cam, (.5, .5, .2), res=420, clip=(0, 0, 400, 320)); C = Cel(400, 320); HD.ink(C, g, dict(HD.PAL['day'], skin=PAL['skin'], shade=PAL['shade'], line=PAL['line'], nail='#f0c090', nail_shade='#b07050'), 3.4, seed=2); self._hands[key] = C.rgba()
        L = np.zeros((H, W, 4), f32); from shots.d7 import put
        put(L, self._hands[key], int(x - 200), int(y - 160)); return L

    def frame(self, f):
        z = float(lerp_keys(f, [(0, 1.0), (34, 1.06), (46, 1.03), (60, 1.06)])); dolly(z); img = self.bg.copy(); x, y = self.x, self.y; tm = f / 24.; t = f // 2 * 2; C = Cel()
        for mx, my, s in self.MILLS:
            hx, hy = mx, my - 86 * s
            for k in range(4): a = f * .035 + k * math.pi / 2 + mx; d = A(math.cos(a), math.sin(a)); n = A(-d[1], d[0]); C.fill(np.array([A(hx, hy), A(hx, hy) + d * 92 * s, A(hx, hy) + d * 92 * s + n * 17 * s, A(hx, hy) + d * 18 * s + n * 17 * s]), '#f6dcae'); C.line(np.array([A(hx, hy), A(hx, hy) + d * 94 * s]), 2.6, '#6a4240', taper=(0, 0)); [C.line(np.array([A(hx, hy) + d * (20 + 18 * j) * s, A(hx, hy) + d * (20 + 18 * j) * s + n * 17 * s]), 1.2, '#8a5a50', taper=(0, 0)) for j in range(4)]
        img = over(img, C.layer()); w = self.S.render(self.cam, 3 + tm, TEAL, sun=(-.02, 1., .12)); img = w.over(img)
        m = SH.red_force(tm); d = SH.draw([(m, self.ps)], self.cam, light=(-.75, .55, .30), rim=('#fff0c0', 3, .8), bounce='#e0906a', dark=.30, sil='#4a2f48', lw=.9); lay = d.layer; hz = .26; lay = lay.copy(); lay[..., :3] = lay[..., :3] * (1 - hz) + rgb('#f2b684') * hz * lay[..., 3:4]; img = SEA.under(img, lay, d.z, w); img = over(img, self.rail(f))
        img = img + SEA.glitter(self.cam, tm, (-.01, 1., .10), 620, spread=.16) * 1.0; img = screen(img, radial(SUN[0], SUN[1], 760, 420, 2.2)[..., None] * rgb('#ffc673') * .55)
        B = Cel()
        for i in range(5): q = tm * (1 + .2 * i); SC.gull(B, 560 + 190 * i + 60 * math.sin(q + i) + 30 * q, 190 + 46 * (i * 3 % 4) + 20 * math.sin(q * 1.3 + i), 12 + 3 * (i % 3), q * 9 + i, col='#6a5a66', hi='#fff0d0', bank=.2 * math.sin(q + i))
        img = over(img, B.layer()); img = zw(img, 1 + .42 * (z - 1)); img = over(img, zw(self.pier, z, cv2.BORDER_CONSTANT) if z > 1.001 else self.pier)
        # ---- the two of them
        CL_, (hx, hy), cc = self.child(f); L = np.zeros((H, W, 4), f32); hp = None
        if f < 47:
            tt = f if 10 <= f <= 20 or 35 <= f <= 47 else f // 2 * 2; r = float(lerp_keys(tt, [(0, 0), (35, 0), (36, -.05), (41, 1.), (59, 1.)])); st = float(lerp_keys(tt, [(38, 0), (44, 1.02), (45, 1.), (59, 1.)])); p = SK.lerp_body(SK.lerp_body(SK.KNEEL, SK.RISE, r), SK.STAND, st); lean = .03 * float(lerp_keys(tt, [(0, -.3), (10, -.6), (15, 1.4), (19, .9), (35, .9), (40, 0)])); [p.__setitem__(k, p[k] + A(lean * v, -.4 * abs(lean) * v)) for k, v in (('chest', 1.), ('neck', 1.3), ('head', 1.6))]
            hatx, haty = (hx - OX) / U, (GY - hy) / U; hk = [(0, (hatx - .30, haty + .52)), (5, (hatx - .30, haty + .55)), (10, (hatx - .14, haty + .34)), (13, (hatx - .02, haty + .13)), (15, (hatx, haty - .004)), (59, (hatx, haty))]; hp = lerp_keys(tt, hk, lambda v: ease(v)); hr = float(lerp_keys(tt, [(0, -.62), (6, -.66), (13, -.16), (15, .02), (59, .02)])); go = float(lerp_keys(tt, [(0, 0), (34, 0), (37, .5), (40, 1.)])); hand = hp + A(-.035, .115) if go < .02 else hp * (1 - go) + (p['chest'] + A(.16 - .05 * st, -.30 - .24 * st)) * go + A(-.035 * (1 - go), .115 * (1 - go) + .16 * math.sin(go * math.pi))
            C = Cel(); wr, dv = SK.body(C, p, OX, GY, U, f, hand=hand + (A(-.075, .036) if go < .05 else A(0, 0)), lw=Z, arm=True if go < .35 else 'under', snap=1. + .5 * (f > 33), slen=.74 + .26 * float(ease((f - 42) / 4.))); Lb = C.layer(); s = 57. * Z * float(lerp_keys(tt, [(35, 1.), (43, 1.13)])); hx0, hy0 = OX + p['head'][0] * U, GY - p['head'][1] * U; tu = float(lerp_keys(tt, [(0, 0), (35, 0), (37, -.06), (41, 1.04), (43, 1.)])); tv = float(lerp_keys(tt, [(42, 0), (46, 1.)])); LY = SK.draw(hx0 + 6, hy0 + 6, s, .92 - .60 * tu - .22 * tv, -.20 + .19 * tu, .10 - .04 * tu, expr='warm' if tt >= 38 else 'soft', light='sunset', f=f, hat=False, side=-1, lw=2.7 * (1 - .12 * tu), rimdir=(-1., .25), look=(.03, -.03)); Lh = over_l(over_l(LY['back'], LY['face']), LY['hair'])
            NK = Cel(); Lb = over_l(Lh * 0, Lb) if f < 45 else over_l(Lb, self.sweep(f)); L = over_l(over_l(LY['back'], Lb), over_l(LY['face'], LY['hair'])); L = smear(L, 14., -4., 7) if f == 46 else L; hb = (1.25 * math.exp(-(f - 15) / 6.5) * math.cos((f - 15) * 2 * math.pi / 8.) if f >= 15 else 0.) + (-.50 * math.exp(-(f - 36) / 4.) * math.cos((f - 36) * .9) if f >= 36 else 0.) + (.16 * ease((f - 16) / 8.) if 16 < f < 36 else 0.) + .05 * math.sin(t * .4); hat_kw = dict(cx=OX + hp[0] * U, cy=GY - hp[1] * U, w=236 * Z, yaw=-2.45, pitch=-.16 + .25 * hr + .24 * self._hun * float(ease((f - 12) / 3.)), roll=hr + .06 - .07 * self._hun * float(ease((f - 12) / 3.)), light='dusk', wear=.45, bounce=hb, f=f, wind=.8); HD_ = self.hand_on(OX + hand[0] * U + 6, GY - hand[1] * U + 2, go)
        else:
            L, u = self.walker(f); hb = .06 * math.sin(t * .4); hat_kw = dict(cx=hx, cy=hy, w=236 * Z, yaw=-2.45, pitch=-.16 + .24 * self._hun, roll=.06 - .07 * self._hun, light='dusk', wear=.45, bounce=hb, f=f, wind=.8)
        pr = lambda Q, k=1.: para(Q, (960, 480), (960, 1000), [(0, '#fff0d6'), (.5, '#e2b4a0'), (1, '#8a6276')], amt=k); grp = over_l(L, CL_) if f < 47 else CL_; img = self.shadow(img, grp); img = self.shadow(img, L, .42 * (1 - (f - 46) / 6.)) if 47 <= f < 52 else img
        ll = rim(rim(pr(L), -9, 5, '#ffb45a', .55, 4.), -3.4, 2.4, '#fff3cc', 1., .6); cl = rim(rim(pr(CL_), 9, 5, '#ffb45a', .55, 4.), 3.4, 2.4, '#fff3cc', 1., .6); img = screen(img, blur(np.maximum(L[..., 3], CL_[..., 3]), 30)[..., None] * rgb('#ffbe6c') * .14); hat_l = HAT.hat(**hat_kw)
        if f < 47:
            img = over(img, ll); img = over(img, cl); img = over(img, hat_l)
            if go < .05: img = over(img, rim(pr(HD_), -2.6, 3, '#fff3cc', 1., .6))
        else:
            img = over(img, cl); img = over(img, hat_l); wk = rim(rim(pr(L, max(.8, 1 - .05 * (f - 46))), -10, 5, '#ffb45a', .55, 5.), -4, 3, '#fff3cc', 1., .6); bl = max(0., (u / U - 2.2) * 1.4); img = over(img, blur(wk, min(9., bl)) if bl > .4 else wk)
            if f >= 53: img = over(img, self.wipe(f))
        img = screen(img, radial(SUN[0], SUN[1], 320, 240, 2.4)[..., None] * rgb('#ffe2a0') * float(lerp_keys(f, [(46, .45), (50, .30), (52, .30), (55, 0.)])) * (1 - .8 * L[..., 3:4] * (f >= 47))); return img + motes(f, 70, seed=5, light=self.ml, col='#fff0b8', drift=(34, -8), size=1.3, amt=.6) * (1 if f < 55 else .2)

    def wipe(self, f):
        """7 f: the cloak's hem swings across the lens on a diagonal.  Folds fan from the shoulder, lining shows where the hem turns, sun on the edge"""
        from op.kit import smear
        k = (f - 53) / 6.; e = k * k * (3 - 2 * k); pv = A(-980, -820); R0 = 1860 + 1880 * e; C = Cel(ss=1); sw = .9 * e; TH = np.linspace(-.14, 1.74, 320); ph = TH * 25.5 + sw * 2.2 + .9 * np.sin(TH * 5. + sw); Rr = R0 + 64 * np.sin(ph) + 26 * np.sin(ph * 2.1 + 1.) + 40 * np.sin(TH * 3. + 2 * sw)
        pt = lambda r, th: pv + r * A(math.cos(th), math.sin(th)); arr = lambda r, th: pv + np.c_[r * np.cos(th), r * np.sin(th)]; hem = arr(Rr, TH); C.fill(np.vstack([hem, pv[None]]), '#1f1a25'); C.clip(np.vstack([hem, pv[None]]))
        n0, n1 = int(math.ceil((ph[0] - 4) / 6.283)), int((ph[-1] + 4) / 6.283)
        def blade(j, p0, p1, col, r0=.12, al=1., bow=.0):
            a0, a1 = np.interp([6.283 * j + p0, 6.283 * j + p1], ph, TH); rr = np.linspace(r0, 1.04, 22); mid = (a0 + a1) / 2; wob = .016 * np.sin(rr * 7 + j * 1.9 + sw * 2) * np.clip(1 - rr, 0, 1) ** .5; hw = (a1 - a0) / 2 * (.35 + .65 * rr) * np.sin(np.clip((rr - r0) / .3, 0, 1) * 1.5708); C.fill(np.vstack([arr(rr * R0, mid - hw + wob), arr(rr * R0, mid + hw + wob)[::-1]]), col, al); return arr(rr * R0, mid - hw + wob), arr(rr * R0, mid + hw + wob)
        for j in range(n0, n1 + 1):
            blade(j, .5, 3.5, '#0b0910'); blade(j, 4.0, 5.6, '#3a3242'); l, r_ = blade(j, 4.35, 5.0, '#584a58', .40); C.line(l[4:], 7, '#ffbf74', taper=(.95, .02), press=.3, seed=j, alpha=.75); d, _ = blade(j, .5, .62, '#050408'); C.line(d, 6, '#050408', taper=(.6, .05), seed=20 + j)
        for j in range(n0, n1 + 1):                                                          # the hem turns under: lining, then its own dark
            a0, a1 = np.interp([6.283 * j + 3.3, 6.283 * j + 6.4], ph, TH)
            if a1 - a0 < .02: continue
            th = np.linspace(a0, a1, 20); rr = np.interp(th, TH, Rr); dp = 150 * np.sin(np.linspace(0, math.pi, 20)) ** .8; C.fill(np.vstack([arr(rr, th), arr(rr - dp, th)[::-1]]), '#4b1d22'); C.fill(np.vstack([arr(rr - dp * .35, th), arr(rr - dp, th)[::-1]]), '#2a1116'); C.line(arr(rr - dp, th), 8, '#0a070c', taper=(.15, .15), press=.4, seed=60 + j)
        C.unclip(); C.line(hem, 9, '#0a070c', taper=(0, 0), press=.3, seed=90); L = C.layer(); L = rim(L, -44, -30, '#ff9a42', .62, 14.); L = rim(L, -15, -10, '#ffe2a4', 1., 2.5); L = rim(L, -6, -4, '#fffbe8', 1., 1.); v = 46 * (1 - abs(2 * k - 1)) + 8; return smear(L, v * .8, v * .55, 7)

    def rail(self, f):
        """silhouettes on the rail of the ship: one smoking, one round with meat on the bone, one with a long rifle"""
        C = Cel(); t = f / 24.; b = 2.2 * math.sin(t * 1.4); c = '#3a2436'; y0 = RAIL[1] + b; x0 = RAIL[0]; s = RAIL[2]; q = lambda ox, L: np.array([(x0 + ox + a * s, y0 - bb * s) for a, bb in L]); F = lambda ox, L, cr=(), col=c: C.fill(spl(q(ox, L), True, cr), col)
        torso = [(-.22, 0), (-.28, .5), (-.24, .82), (-.08, .92), (.08, .92), (.24, .82), (.28, .5), (.22, 0)]; head = [(-.12, 1.02), (-.13, 1.18), (-.06, 1.30), (.06, 1.30), (.13, 1.18), (.11, 1.02), (0, .94)]
        F(0, torso, (0, 7)); F(0, head); F(0, [(-.10, 1.20), (-.20, 1.10), (-.26, .80), (-.18, .84), (-.10, 1.04)]); C.line(q(0, [(.11, 1.06), (.25, 1.08)]), 2.2, '#fff0d8', taper=(0, 0)); C.fill(q(0, [(.24, 1.06), (.27, 1.06), (.27, 1.10), (.24, 1.10)]), '#ff9a4a'); F(0, [(.2, .8), (.36, .55), (.30, .40), (.16, .6)])
        F(1.35 * s, [(-.40, 0), (-.52, .4), (-.42, .78), (-.1, .92), (.1, .92), (.42, .78), (.52, .4), (.40, 0)], (0, 7)); F(1.35 * s, [(-.20, .96), (-.22, 1.16), (-.10, 1.30), (.10, 1.30), (.22, 1.16), (.20, .96), (0, .88)]); ch = .03 * math.sin(t * 9); F(1.35 * s, [(.38, .80), (.52, .92), (.40, 1.06 + ch), (.28, 1.0 + ch)]); F(1.35 * s, [(.14, 1.06 + ch), (.22, .92 + ch), (.42, .96 + ch), (.46, 1.12 + ch), (.32, 1.20 + ch)], (), '#7a3a2e'); C.line(q(1.35 * s, [(.44, 1.06 + ch), (.62, 1.14 + ch)]), 4., '#f2dcb8', taper=(0, 0))
        F(2.55 * s, torso, (0, 7)); F(2.55 * s, head); F(2.55 * s, [(-.15, 1.14), (-.10, 1.36), (.08, 1.38), (.16, 1.16), (.0, 1.20)]); C.line(q(2.55 * s, [(.30, -.05), (-.34, 1.95)]), 5., c, taper=(0, .2)); C.line(q(2.55 * s, [(.36, .02), (.16, .58)]), 9., c, taper=(.1, .2)); F(2.55 * s, [(-.26, .8), (-.34, .6), (-.10, .56), (.10, .66), (.02, .74)])
        for k in range(4): u = np.linspace(0, 1, 24); ph = t * 2 + k * 1.6; C.line(np.c_[x0 + .27 * s + 60 * u + 9 * np.sin(u * 7 - ph) * u, y0 - 1.1 * s - 46 * u - k * 4], 5., '#fff0da', taper=(.1, .8), alpha=.22, seed=k, wfun=lambda v: .4 + 1.6 * v)
        L = C.layer(); return rim(L, 2, 1.6, '#fff0c0', .9, .5)


RAIL = (1290., 452., 62.)


def mix3(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a, f32); b = rgb(b) if isinstance(b, str) else np.asarray(b, f32); return a * (1 - t) + b * t
