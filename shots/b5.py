"""b5  the sea king punch.  f0-29 the shadow / f30-39 the eruption / f40-59 the arm past the lens / f60-65 the hit / f66-74 it topples / f75-89 the hat."""
import numpy as np, math, cv2
from op import W, H, comp, kit, sea, ships, don, fleet3 as F3, splash as SP
from op.paint import rgb, f32, grid, smooth, ramp, fbm
from op.ink import Cel, spl, over_roi
from op.cam import Camera3D, rot_z
from op.comp import over, screen, lerp_keys, ease
from op.fig_pose import P, POSES, seq, mix_pose, add_pose, Rz, Rx, Ry, unit
from chars.act import Actor, rubber, View, Pen, small_hand, SKIN, MAT, hull, union
from chars.seaking import seaking, head_pt

K = np.array([-11., 26., 0.]); _KD = K / np.linalg.norm(K); SE = (.02, .71, .70); SUN = (.70, .25, .66); UP = np.array([0, 0, 1.])
TROP = dict(sea.TROPIC); GLASS = dict(sea.TROPIC, deep='#0a86a0', body='#1aa9b8', mid='#3cc4c6', light='#86e6da')
DEEP = dict(sea.DAY, deep='#0a5c86', body='#1583a8', mid='#2fa6c0', light='#74d0da', sky='#d6f6f4', crest='#46d6c4', far='#8fd2e4', ink='#063a56')
SIT = P(root=(0, 0, -.30), pel=(-10, 0, 0), wst=(16, 0, 0), cst=(12, 0, 4), nck=(4, 0, 0), hed=(-8, 0, 0), hL=(.30, .10, -.86), hR=(.02, .30, .62), eL=(.8, -1, .1), eR=(1, .1, .3), gR='open', gL='relax', fL=(.09, .24, .10), fR=(.07, .20, .16), kL=(.3, 1, .6), kR=(.1, 1, .8))
LEAN = P(SIT, root=(.16, .02, -.28), pel=(20, 24, 0), wst=(20, 14, 0), cst=(18, 12, 0), nck=(16, 8, 0), hed=(14, 6, 0), hL=(.2, .3, -.7), hR=(.55, .25, -.55), gR='grip', eR=(.9, -.4, .5))
UPP = P(root=(0, -.02, -.16), pel=(10, 0, -8), wst=(10, 0, -4), cst=(4, 0, -6), nck=(-12, 0, 6), hed=(-16, 0, 6), hL=(.10, .08, .60), eL=(1, .1, .3), gL='open', hR=(.62, .20, -.45), eR=(.8, -.8, 0), gR='splay', fL=(.11, .13, .043), fR=(.11, -.14, .05), kL=(.4, 1, 0), kR=(.5, 1, 0))
WIND = P('punch_wind', hL=(.10, .26, .52), eL=(1, -.1, .2), gL='open', wL=(-30, 20, 0), nck=(-2, -4, 40), hed=(-8, -4, 26))
HITP = P('punch_hit', hL=(.10, .10, .50), eL=(1, -.2, .1), gL='open')


def MI(p):
    """segE only: mirror across the upright plane Luffy -> king, so the punch keeps reading left to right"""
    p = np.asarray(p, float); q = 2 * (p @ _KD)[..., None] * _KD - p; q[..., 2] = p[..., 2]; return q


def put(img, t): l, ox, oy = t; return over_roi(img, l, ox, oy)


def king_rest(sw=0., e=1.):
    return np.array([[-11, 28, -13], [-11.3, 27, -4], [-12.2 - sw * .4, 27.5, 4.5 * e], [-12.0 + sw * .5, 28.2, 11.5 * e], [-10.6 + sw, 26.2, 17.0 * e], [-9.6 + sw, 23.4, 19.2 * e]], float)


def frameR(F, tilt=0.):
    F = unit(np.asarray(F, float)); L = unit(np.cross(UP, F)); U = np.cross(F, L); c, s = math.cos(tilt), math.sin(tilt); return np.stack([F, L * c + U * s, U * c - L * s], 1)


def axis_rot(ax, a):
    ax = unit(np.asarray(ax, float)); Kx = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]]); return np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx


class Shot:
    id = 'b5'; n = 90; finish = dict(bloom=.30, vig=.15)
    EVENTS = [(0, 'water', dict(note='calm lapping, creak of the dinghy')), (12, 'bubble', dict(note='bubbles rise')), (20, 'rumble', dict(note='the shadow widens, low swell', dur=10)), (30, 'eruption', dict(size=1.0, note='sea king breaks the surface')), (36, 'roar', dict(size=1., dur=20)),
              (42, 'step', dict(note='foot planted, hull knocks')), (44, 'whoosh', dict(note='arm whips back past the lens')), (46, 'stretch', dict(note='rubber creak under tension', dur=11)), (57, 'snap', dict(note='release, smear')), (60, 'impact', dict(size=1.2, note='ドン! hit stop 5 f, ring shock on the sea')),
              (65, 'impact', dict(size=.6, note='head thrown, teeth fly')), (67, 'whip', dict(note='arm snaps home')), (73, 'splash', dict(size=1.2, note='topples, wall of spray', dur=16)), (79, 'catch', dict(note='hat caught by the brim')), (84, 'pat', dict(note='hat on')), (86, 'laugh', dict(note='shishishi'))]

    def setup(self):
        self.S0 = sea.Sea(5, hs=.20, length=10, chop=.6); self.S = sea.Sea(3, hs=1.0, wind=(.6, 1.)); self.pano = SP.Pano('day'); self.L = Actor('luffy', 'vest'); self.king_cache = {}
        from op import hands as HD; self.HD = HD

    # ---------------------------------------------------------------- A
    def segA(self, f):
        t = f / 24.; e = (f / 29.) ** 1.5; cam = Camera3D((1.2 + 3.2 * e, -6.6 - 10.5 * e, 6.4 + 17 * e), (.1, .4 + 1.6 * e, .3), 1450); sw = smooth(17, 29, f) ** 1.5
        Xp, Yp, _ = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); rr = np.hypot(Xp - .3, Yp - 1.5); foam = np.clip(np.exp(-((rr - 7) / 5.) ** 2) * smooth(24, 29, f) * .62, 0, 1).astype(f32)
        rings = [(0, 0, 1.6 + ((f * .12 + k * 1.7) % 5.2), .5, .05, 0.) for k in range(3)] + ([(.5, 2, 1.0, 9., 2.0 * sw, 0.)] if sw > 0 else [])
        w = self.S0.render(cam, 2 + t, GLASS, sun=SUN, step=8, rings=rings, foam=foam, haze=.3); img = w.rgb.copy()
        # ---- what rises under it: head, throat, coils; edges sharpen as it nears the surface
        k = .05 + .95 * (f / 29.) ** 2.0; q = np.array([[-.5, 2.2], [.8, -.6], [3.4, -2.2], [6.2, -.4], [6.0, 3.8], [2.4, 6.6], [-3.4, 6.4], [-7.4, 2.6], [-7.4, -3.4], [-3.2, -7.8], [3.6, -9.0], [10., -6.0]]) * (1.2 + 2.1 * k); q = spl(q + [.5, 1.4], n=7); ww = np.interp(np.linspace(0, 1, len(q)), [0, .04, .10, .16, .3, .6, 1], [1.0, 2.2, 2.6, 1.9, 2.0, 1.7, .8]) * (.35 + 2.0 * k)
        sx, sy = Xp[::4, ::4] + .35 * np.sin(Yp[::4, ::4] * 1.1 + t * 5), Yp[::4, ::4] + .35 * np.sin(Xp[::4, ::4] * 1.0 - t * 4); d = np.full(sx.shape, 1e9, f32)
        for a, b, wa in zip(q[:-1], q[1:], ww):
            v = b - a; tt = np.clip(((sx - a[0]) * v[0] + (sy - a[1]) * v[1]) / (v @ v), 0, 1); d = np.minimum(d, np.hypot(sx - a[0] - v[0] * tt, sy - a[1] - v[1] * tt) - wa)
        
        for sg in (-1, 1):                                            # fan fins and jaw spread out of the head
            hb = q[3]; dv0 = q[0] - q[6]; dv0 = dv0 / np.hypot(*dv0); nv0 = np.array([-dv0[1], dv0[0]]) * sg
            for j_ in range(5):
                an = .35 + .34 * j_; e_ = hb + (nv0 * math.sin(an + .5) + dv0 * math.cos(an + 1.1)) * (3.4 + .5 * (j_ % 2)) * (.35 + 1.7 * k); v = e_ - hb; tt = np.clip(((sx - hb[0]) * v[0] + (sy - hb[1]) * v[1]) / (v @ v), 0, 1); d = np.minimum(d, np.hypot(sx - hb[0] - v[0] * tt, sy - hb[1] - v[1] * tt) - (.10 + .55 * k * (1 - tt)))
        soft = 2.2 - 1.8 * k; m = cv2.resize(smooth(soft, -soft, d), (W, H), interpolation=cv2.INTER_CUBIC).clip(0, 1) * min(1, f / 5. + .15); m2 = cv2.resize(smooth(soft * .5, -soft - 1.2, d), (W, H)).clip(0, 1)
        dk = rgb('#06425e'); img = img * (1 - (m * (.40 + .42 * k))[..., None]) + dk * (m * (.40 + .42 * k))[..., None]; img = img * (1 - (m2 * .35 * k)[..., None]) + rgb('#031f38') * (m2 * .35 * k)[..., None]
        if f >= 17:                                                   # two eyes open in it
            c = Cel(ss=2); op = smooth(17, 21, f); hd = q[0]; dv = unit(q[0] - q[3]); nv = np.array([-dv[1], dv[0]])
            for sg in (-1, 1):
                a = np.linspace(0, 6.283, 20); e0 = hd + nv * sg * 1.15 * (.35 + 1.4 * k) + dv * .2; E = e0 + np.outer(np.cos(a) * .55 * (.4 + k), dv) + np.outer(np.sin(a) * .26 * op * (.4 + k), nv); p = cam.project(np.c_[E, 0 * a])[0]; c.fill(p, '#ffe27a', .85); P2 = e0 + np.outer(np.cos(a) * .10 * (.4 + k), dv) + np.outer(np.sin(a) * .22 * op * (.4 + k), nv); c.fill(cam.project(np.c_[P2, 0 * a])[0], '#1a1420', .9)
            tk = smooth(21, 28, f)
            if tk > 0:                                                # the mouth opening under the boat: two rows of teeth through the water
                for sg in (-1, 1):
                    for j_ in range(13):
                        a_ = -1.25 + 2.5 * j_ / 12; rr_ = (1.1 + 2.3 * tk) * (1 + .06 * math.sin(j_ * 2.3)); p0 = np.array([.3, 1.6]) + dv * sg * math.cos(a_) * rr_ * 1.25 + nv * math.sin(a_) * rr_ * 1.15; p1 = p0 - (p0 - np.array([.3, 1.6])) * (.22 + .10 * (j_ % 3)) * tk; m_ = (p1 - p0)[::-1] * [1, -1] * .22; c.fill(cam.project(np.c_[np.array([p0 + m_, p1, p0 - m_]), [0, 0, 0]])[0], '#dff2dc', .62 * tk)
            l = c.layer(); l = cv2.GaussianBlur(l, (0, 0), 1.0 + 2.4 * (1 - k)); img = over(img, l, .85); img = screen(img, cv2.GaussianBlur(l[..., :3], (0, 0), 16) * .6)
        # sun flecks that slide on the glass
        img = screen(img, sea.glitter(cam, t, (.25, .9, .5), 260, seed=2, rmin=2., rmax=60., spread=.6, size=.5) * .55)
        c = Cel(ss=2); r = np.random.default_rng(6)                   # bubbles
        for i in range(46):
            b0 = 8 + r.uniform(0, 18); th, ra, sz = r.uniform(0, 6.28), r.uniform(1.2, 8.5), r.uniform(.05, .22); wob = r.uniform(0, 6)
            if f < b0: continue
            u = (f - b0) / 7.; 
            if u > 1.3: continue
            p, z = cam.project(np.array([ra * math.cos(th) + .12 * math.sin(f * .9 + wob), 1 + ra * math.sin(th), 0])); R = sz * cam.f / z * (.35 + .65 * min(1, u)); a = np.linspace(0, 6.283, 18); o = p + np.c_[np.cos(a) * R, np.sin(a) * R * .8]
            if u < 1: c.fill(o, '#e6fffb', .30); c.line(o, 1.6, '#f6fffd', taper=(0, 0), alpha=.9); c.fill(p + [-R * .3, -R * .3] + np.c_[np.cos(a), np.sin(a)] * R * .25, '#ffffff')
            else: [c.line(np.array([p + np.array([math.cos(h), math.sin(h)]) * R * .9, p + np.array([math.cos(h), math.sin(h)]) * R * 1.6]), 1.6, '#f6fffd', alpha=.8) for h in np.linspace(0, 6.28, 7)[:-1]]
        img = over(img, c.layer())
        # ---- dinghy, barrel, boy
        zb = .12 + float(self.S0.height(0, 0, 2 + t)) + 2.0 * sw; pit, rol = .03 * math.sin(t * 2.2) - .10 * sw, .05 * math.sin(t * 1.7 + 1) + .16 * sw * math.sin(f * .9)
        ps = ships.pose(0, 0, zb, 1.95, pit, rol); cs = cv2.GaussianBlur(np.exp(-((Xp + .55) / 1.3) ** 2 - ((Yp + .25) / 2.0) ** 2).astype(f32), (0, 0), 3); img = img * (1 - .34 * cs[..., None] * np.array([1, .7, .55], f32))
        d_ = ships.draw([(ships.dinghy(t, barrel_=True), ps)], cam, light=SUN, lw=1.1); img = sea.under(img, d_.layer, d_.z, w)
        pose = seq(f // 2 * 2, [(0, SIT, 'hold'), (10, SIT, 'io'), (15, LEAN, 'hold'), (22, LEAN, 'out'), (26, UPP, 'hold')]); R0, t0 = ps[0], ps[1]
        o = t0 + R0 @ np.array([-.25, 0, .42]); hd = 1.95 - math.pi / 2
        img = put(img, self.L.pose(pose).place(o, hd + (.9 if 12 < f < 24 else .1)).draw(cam, f, light=SUN, rim=dict(col='#fffbe0', w=2.4, dir=SUN, amt=.95), expr='calm', wind=(.3, .1, 0), w=1.7))
        fl = smooth(26, 29, f) * .25; return img * (1 - fl) + fl * rgb('#eafffa')

    # ---------------------------------------------------------------- B
    def king(self, f, seg):
        t = f / 24.
        if seg == 'B':
            e = float(np.interp(f, [30, 31, 32, 33, 34, 36, 38, 40], [.30, .56, .78, .94, 1.04, 1.07, 1.01, 1.0])); nose = float(smooth(31, 39, f)); pt_ = 1.295 * (1 - nose) - .167 * nose; hz_ = unit(np.array([.37, -.93, 0]) * (1 - nose) + np.array([.42, -.907, 0]) * nose); F = unit(hz_ * math.cos(pt_) + UP * math.sin(pt_)); return dict(spine=king_rest(.8 * (1 - nose), e), R=frameR(F, .10 * math.sin(t * 8)), jaw=float(np.interp(f, [30, 32, 33, 34, 35, 36, 37, 38, 39], [.15, .16, .24, .36, .50, .66, .82, .96, 1.05])), flare=nose * 1.4)
        lean = smooth(45, 57, f); sw = .5 * math.sin(t * 4.5); q = king_rest(sw, .80); q[3:] += np.outer(np.array([.25, .75, 1.0]), np.array([2.6, -8.0, -1.6])) * lean + np.outer(np.array([.1, .3, .5]), [0, 0, 1.]) * .35 * math.sin(f * 1.9); q[4:] += np.array([0, 0, .35 * math.sin(t * 5)]); F = unit(np.array([.40, -.86, -.10 - .10 * lean])); return dict(spine=q, R=frameR(F, .06 * math.sin(t * 5)), jaw=float(.95 + .20 * lean + .05 * math.sin(f * 1.9)), flare=1.4 + .3 * math.sin(f * 1.3))

    def segB(self, f):
        t = f / 24.; u = (f - 30) / 9.; tb = (f - 29.2) / 24.; sh = kit.shake(f, 15 * math.exp(-u * 1.6) + 3, 5); cam = Camera3D((1.8, -14.5 + .6 * u, 1.35 + .4 * u), (-5.0 - .3 * u, 12.7, 3.6 + 3.3 * comp.ease(min(1, u * 1.35))), 1300, cx=960 + sh[0], cy=560 + sh[1], roll=-.05 + .02 * u)
        sky = self.pano.sky(cam, .95); Xp, Yp, z0 = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); rr = np.hypot(Xp - K[0], Yp - K[1]); foam = (np.exp(-((rr - 5 - 9 * tb) / (4 + 10 * tb)) ** 2) * (z0 > 0)).astype(f32)
        Rw = 5 + 46 * tb; rings = [(K[0], K[1], Rw, 5.5, 4.2 * math.exp(-tb * 1.1), 1.)]; w = self.S.render(cam, 3 + t, DEEP, sun=SUN, rings=rings, foam=foam); img = w.over(sky); img = img + sea.glitter(cam, t, (.35, 1, .3), 260) * .5
        k = self.king(f, 'B'); G = seaking(k['spine'], (.4, -.9, 0), k['R'], k['jaw'], t, flare=k['flare'], whisk=(0, 0, -.6 + u)); r = F3.draw(cam, [G], SUN, fog=('#bfe9ee', 25., 1 / 260.), lw=.05, lwmax=4.6)
        lay = kit.rim(r.rgba, -8, 5, '#fffbe2', .9, soft=1.0); lay = kit.para(lay, (0, 900), (0, 250), [(0, '#8fb2c8'), (.6, '#ffffff'), (1, '#ffffff')]); img = sea.under(img, lay, r.z, w)
        # water torn up with it
        for j, (a0, a1, hh, sd) in enumerate(((.2, 2.0, 12, 1), (2.2, 4.2, 9, 2), (4.1, 6.3, 14, 3), (3.6, 5.6, 18, 4), (5.2, 7.2, 10, 6))):
            a = np.linspace(a0, a1, 26); z = np.linspace(.5, 1, 26) * 0 + hh * min(1, .35 + u * 1.2) ; rt = np.stack([K[0] + np.cos(a) * 3.4, K[1] + np.sin(a) * 3.4, z + 1.2 * np.sin(a * 5 + sd)], -1); v = np.stack([np.cos(a) * (6 + sd), np.sin(a) * (6 + sd), np.full(26, 5. - 2 * sd)], -1)
            img = over(img, sea.sheet(cam, SP.curtain(rt, v * [1, 1, .8], .55 + tb * 2.4, seed=sd), seed=sd, pal=DEEP, su=10, sv=3., flow=-tb * 2, holes=.55 + .45 * u, thin=.12))
        Pc, Dp, Dv, rd = sea.splash_ring(cam, K[0], K[1], .05 + tb, r0=4.2, v=10, vz=20, n=60, seed=4, drop=.18); img = over(img, sea.sheet(cam, Pc, seed=7, pal=DEEP, su=20, sv=4, holes=.45 + .5 * u)); img = over(img, sea.drops(cam, Dp, Dv, rd, DEEP, maxpx=9))
        rg = np.random.default_rng(8); n = 70; a = rg.uniform(0, 6.28, n); v = np.stack([np.cos(a) * rg.uniform(2, 14, n), np.sin(a) * rg.uniform(2, 12, n) - 4, rg.uniform(8, 24, n)], -1); p0 = np.array([K[0], K[1] - 2, 1.5]) + rg.normal(0, 2, (n, 3)); tt = .1 + tb * 1.3; img = over(img, sea.drops(cam, p0 + v * tt + np.array([0, 0, -4.9 * tt * tt]), v - np.array([0, 0, 9.8 * tt]), rg.uniform(.04, .16, n), DEEP, shutter=1 / 30., maxpx=9))
        bump = 2.2 * math.exp(-((Rw - 28.2) / 7.) ** 2); zb = .15 + float(self.S.height(0, 0, 3 + t)) * .6 + bump; ps = ships.pose(0, 0, zb, 1.98, .05 + .42 * math.exp(-((Rw - 24) / 7.) ** 2), .06 * math.sin(f)); dl = ships.draw([(ships.dinghy(t, barrel_=True), ps)], cam, light=SUN, lw=1.1); img = sea.under(img, dl.layer, dl.z, w)
        img = put(img, self.L.pose(add_pose(UPP, pel=(-6 * u, 0, 0))).place(ps[1] + ps[0] @ np.array([.35, 0, .42]), .41).draw(cam, f, light=SUN, rim=dict(col='#fff6d0', w=3, dir=SUN), wind=(.5, -1.6, .2), expr='calm', hair=1.))
        lens = np.random.default_rng(3); c = Cel(ss=2)                 # drops on the lens
        for i in range(16):
            x, y, s = lens.uniform(60, W - 60), lens.uniform(40, H - 40), lens.uniform(5, 18)
            if u > .25 + .02 * i: a = np.linspace(0, 6.28, 18); y2 = y + (u * 9 - 2) * s * .6; c.fill(np.c_[x + np.cos(a) * s, y2 + np.sin(a) * s * 1.2], '#e9fbff', .28); c.fill(np.c_[x - s * .3 + np.cos(a) * s * .25, y2 - s * .4 + np.sin(a) * s * .25], '#ffffff', .8)
        img = over(img, c.layer()); return img if f > 30 else img * .55 + .45 * rgb('#f3fffc')

    # ---------------------------------------------------------------- C
    def camC(self, f):
        u = (f - 40) / 19.; sh = kit.shake(f, 1.2 + 7 * math.exp(-abs(f - 46) / 1.5) * (f >= 45) + (14 if f >= 57 else 0), 9); a = .06 * u; return Camera3D((2.05 + .12 * u, -2.30 + .20 * u, 1.62 - .10 * u), (-2.15 + a, 5.0, 2.80 + .06 * u), 1000 + 110 * smooth(46, 57, f), cx=960 + sh[0], cy=575 + sh[1], roll=-.04 - .05 * smooth(44, 47, f) + .05 * (f >= 57))

    def segC(self, f):
        t = f / 24.; cam = self.camC(f); sky = self.pano.sky(cam, .95); Xp, Yp, z0 = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); rr = np.hypot(Xp - K[0], Yp - K[1]); foam = (np.exp(-((rr - 5.5) / 4.) ** 2) * .9 * (z0 > 0)).astype(f32)
        w = self.S.render(cam, 3 + t, DEEP, sun=SUN, foam=foam); img = w.over(sky); img = img + sea.glitter(cam, t, (.2, 1, .3), 200) * .4
        k = self.king(f, 'C'); G = seaking(k['spine'], (.4, -.9, 0), k['R'], k['jaw'], t, flare=k['flare'], whisk=(.3 * math.sin(f), 0, 0)); r = F3.draw(cam, [G], SUN, fog=('#bfe9ee', 25., 1 / 240.), lw=.05, lwmax=4.2); lay = kit.rim(r.rgba, -8, 5, '#fffbe2', .9, soft=1.0); lay = kit.para(lay, (0, 900), (0, 250), [(0, '#8fb2c8'), (.6, '#ffffff'), (1, '#ffffff')]); img = sea.under(img, lay, r.z, w)
        rg = np.random.default_rng(12); n = 60; a = rg.uniform(0, 6.28, n); ph = (t * 1.1 + rg.random(n)) % 1.; o = head_pt(k['R'], k['spine'][-1], (7, 0, -1)); v = k['R'][:, 0] * 16 + rg.normal(0, 4, (n, 3)); img = over(img, sea.drops(cam, o + v * ph[:, None] + np.array([0, 0, -4.9]) * (ph ** 2)[:, None], v, rg.uniform(.04, .16, n), DEEP, shutter=1 / 30.))
        sway = .05 * math.sin(t * 3); push = smooth(43, 46, f); ps = ships.pose(0, 0, .20 + .05 * math.sin(t * 4), 1.98, .05 - .10 * push + .16 * (f >= 57), sway + .06 * push); dl = ships.draw([(ships.dinghy(t, barrel_=True), ps)], cam, light=SUN, lw=1.5, rim=('#fff2c8', 4, .7)); img = sea.under(img, dl.layer, dl.z, w)
        st = 0 if f < 57 else 1; fp = f if f < 48 or f >= 57 else 48 + (f - 48) // 2 * 2
        pose = seq(fp, [(40, UPP, 'io'), (43, add_pose(WIND, pel=(4, 0, 24), cst=(0, 0, 10), root=(0, .03, -.02)), 'out'), (46, add_pose(WIND, pel=(0, 0, -10), cst=(-4, 0, -14), root=(0, -.02, -.03)), 'io'), (48, WIND, 'io'), (56, add_pose(WIND, root=(0, -.015, -.02), pel=(0, 0, -4)), 'snap'), (57, mix_pose(WIND, HITP, .75), 'lin'), (59, HITP, 'hold')])
        ext = float(np.interp(f, [42, 43, 44, 45, 46, 60], [0, .10, .50, .92, 1, 1])); R_, D_, Fw = cam.R[0], cam.R[1], cam.R[2]; q = .010 * math.sin(f * 2.7) if f > 47 else 0
        def arm(pen, s, ua, fa, S):
            if s != 'R' or f < 43: return False
            sh = S.sh['R']; E = cam.eye + R_ * .36 + D_ * .13 - Fw * .55
            if st == 0:
                end = sh + (E - sh) * ext; mid1 = sh + (end - sh) * .30 + D_ * (.22 - .05 * ext + q) + R_ * .16; mid2 = sh + (end - sh) * .66 + D_ * (.05 - q) + R_ * (.20 + q); P3 = spl(np.array([sh, mid1, mid2, end]), n=14) if False else np.array([(1 - u) ** 3 * sh + 3 * (1 - u) ** 2 * u * mid1 + 3 * (1 - u) * u * u * mid2 + u ** 3 * end for u in np.linspace(0, 1, 46)]); u = np.linspace(0, 1, 46); rad = np.interp(u, [0, .06, .2, .5, 1], [.052, .060, .054, .050, .056]); amp = .30 * math.exp(-max(0, f - 45) / 3.2) + .05 + .03 * (f > 52)
                if 43 <= f <= 45: hR = np.stack([D_ * -1, unit(end - sh), -R_], 1); small_hand(pen, end, np.stack([unit(np.cross(unit(end - sh), R_)), unit(end - sh), R_], 1), .20, 'R', 'fist', SKIN, pen.w * 1.3)
                rubber(pen, P3, rad, SKIN, ripple=amp, k=5.5, phase=(f - 43) * .55, rings=8, seed=2, w=pen.w * 1.25, zmin=.07); return True
            kk = [.22, .60, 1.0][min(2, f - 57)]; tgt = head_pt(k['R'], k['spine'][-1], (4.4, 3.2, .0)); a0 = E if f == 57 else sh; P3 = np.array([a0 + (tgt - sh) * u * kk + (sh - a0) * min(1, u * 3) for u in np.linspace(0, 1, 40)]); rad = np.interp(np.linspace(0, 1, 40), [0, .5, 1], [.10 if f > 57 else .16, .11, .30]); rubber(pen, P3, rad, SKIN, ripple=.18, k=4, phase=f * .6, rings=10, seed=2, w=pen.w * 1.2)
            e_ = P3[-1]; c2, zz = cam.project(e_); rr_ = [.62, .95, 1.35][min(2, f - 57)] * cam.f / zz; a = np.linspace(0, 6.28, 20); pen.c.fill(np.c_[c2[0] + np.cos(a) * rr_, c2[1] + np.sin(a) * rr_ * .9], SKIN['shade']); pen.c.fill(np.c_[c2[0] + rr_ * .12 + np.cos(a) * rr_ * .8, c2[1] - rr_ * .14 + np.sin(a) * rr_ * .72], SKIN['base']); kit.outline(pen.c, np.c_[c2[0] + np.cos(a) * rr_, c2[1] + np.sin(a) * rr_ * .9], pen.w); return True
        ex = 'grit' if f < 57 else 'shout'
        if f >= 57:
            c = Cel(ss=1); kit.speed_lines(c, -2.45, 46, 5., (300, 1100), '#ffffff', f, alpha=.75); img = over(img, c.layer())
        o = ps[1] + ps[0] @ np.array([.30, 0, .42]); l = self.L.pose(pose).place(o, .41).draw(cam, f, light=SUN, rim=dict(col='#fff4cc', w=8, dir=SUN, glow=.5), wind=(.5, -1.9, .2), expr=ex, hair=1.2, arm_fn=arm, full=True)
        img = put(img, l)
        if f == 57: img = kit.smear(img, 70, -90, 31)
        if f == 58: img = kit.smear(img, 26, -34, 17)
        if f in (44, 45): pass
        vg = smooth(50, 53, f) * (f < 57) * .16; yy = grid()[1] / H; img = img * (1 - vg * (.4 + .6 * np.abs(yy - .5) * 2))[..., None]; return img

    # ---------------------------------------------------------------- D
    def segD(self, f):
        t = f / 24.; ff = min(f, 64); hold = f < 65; q = king_rest(.5, 1.); q[3:] += np.outer(np.array([.3, .8, 1.0]), np.array([1.6, -4.0, -1.4])); F0 = unit(np.array([.40, -.86, -.16])); L0 = unit(np.cross(UP, F0)); dv = unit(-F0 * .62 - L0 * .78 + UP * .16); push = [.35, .95, 1.0, 1.0, 1.0, 1.25][min(5, f - 60)]; q[4] += dv * 1.0 * push; q[5] += dv * 2.2 * push
        tw = .55 * push; F = unit(F0 - L0 * tw + UP * .10 * push); R = frameR(F, -.42 * push); o = q[-1]; hp = head_pt(R, o, (4.2, 3.0, -.2)); am = 2.5 * push
        sh = kit.shake(f, 11 if hold else 22, 31); base = head_pt(frameR(F0), q[-1] - dv * 2.2 * push, (3.4, 0, 0)); zoom = 1 + .012 * (f - 60); ey = base + L0 * 15.5 + F0 * 10.0 + UP * -2.6; cam = Camera3D(ey, base + L0 * .6 + UP * .3, 1230 * zoom, cx=960 + sh[0], cy=560 + sh[1], roll=-.10)
        sky = self.pano.sky(cam, .4); w = self.S.render(cam, 3 + t, DEEP, sun=SUN); img = w.over(sky) if w.a.max() > 0 else sky
        G = seaking(q, (.4, -.9, 0), R, 1.10 + .12 * push, t, hit=1., dent=(hp - dv * .4, dv, am, 2.0), ripple=(.95 - .05 * (f - 60), .10 * push, .10), flare=-1.2, whisk=dv * 6); r = F3.draw(cam, [G], (.45, -.75, .5), fog=('#bfe9ee', 60., 1 / 400.), lw=.030, lwmax=5.2); lay = kit.rim(r.rgba, -10, 6, '#fffbe2', .95, soft=1.2); lay = kit.para(lay, (0, 1000), (0, 200), [(0, '#86a6c6'), (.55, '#ffffff'), (1, '#ffffff')])
        pc, pz = cam.project(hp); inv = np.array([pc[0] - 960, pc[1] - 540])
        c = Cel(ss=2); SP.burst_lines(c, pc[0], pc[1], 330, 1700, 120, 3 + f // 2, '#ffffff', 10, .80); img = over(img, c.layer()); img = over(img, lay)
        # creases round the dent
        c = Cel(ss=2); rg = np.random.default_rng(5); E = [pc + np.array([math.cos(a), math.sin(a)]) * 190 for a in np.linspace(0, 6.28, 24)]
        pen = Pen(c, View(cam), (-.3, -.95), 5.6, 4)
        # arm + fist (forced scale)
        arm_d = cam.project(hp - dv * 9 - UP * 3.4)[0] - pc; arm_d = arm_d / np.hypot(*arm_d); P2 = [pc + arm_d * s_ + np.array([0, 26 * math.sin(s_ * .006 + 1)]) * min(1, s_ / 300) for s_ in np.linspace(250, 1700, 40)]
        fc = Camera3D((0, -10, 0), (0, 0, 0), 10000.); fp = lambda p: np.array([(p[0] - 960) / 1000, 0, -(p[1] - 540) / 1000]); P3 = np.array([fp(p) for p in P2]); rad = np.interp(np.linspace(0, 1, 40), [0, .12, .3, 1], [.062, .074, .080, .105]); pn = Pen(c, View(fc), (-.25, -.95), 6., 4); rubber(pn, P3, rad, SKIN, ripple=.16 if hold else .3, k=5, phase=(f - 60) * .5, rings=12, seed=6, w=6.)
        img = over(img, c.layer())
        HD = self.HD; vd = np.array([-arm_d[0], arm_d[1], -.10]); vd /= np.linalg.norm(vd); bk0 = np.array([.92, .05, .42]); bk = unit(bk0 - vd * (bk0 @ vd)); Rf = np.stack([np.cross(vd, bk), vd, bk], 1); c = Cel(ss=2); wr = pc + arm_d * 190
        pal = dict(skin=SKIN['base'], shade=SKIN['shade'], line='#17131a', light='#ffe8c8'); hh_ = HD.Hand('fist', 'R', Rf, (0, 0, 0), HD.HEAVY, pivot=(0, .95, 0), arm=.5)
        g = HD.render([hh_], HD.Cam(pc[0] + arm_d[0] * 10, pc[1] + arm_d[1] * 10, 265, 4.2), (-.4, .85, .5), 520, clip=(0, 0, W, H)); HD.ink(c, g, pal, 6.5, 3); fl = c.layer(); fl = kit.rim(fl, 8, 7, '#fffbe6', .9, soft=1.); img = over(img, fl)
        c = Cel(ss=2)                                                  # wrinkles of flesh rolling over the knuckles
        for k_ in range(9):
            a = math.atan2(-arm_d[1], -arm_d[0]) + (k_ - 4) * .30; e1 = np.array([math.cos(a), math.sin(a)]); p0 = pc + e1 * (178 + 14 * (k_ % 2)); m_ = np.array([-e1[1], e1[0]]); L_ = spl([p0 - m_ * 46, p0 + e1 * 26, p0 + m_ * 46], n=8); c.line(L_, 5.5, '#17131a', taper=(.4, .4)); c.line(L_ + e1 * 30, 3., '#17131a', taper=(.4, .4))
        for k_ in range(12):                                           # teeth and spit
            a = rg.uniform(-2.4, .2); v = rg.uniform(160, 520); tt = (f - 60 + .5) * (.25 if hold else .5); mo = cam.project(head_pt(R, o, (7.5, 0, -1.6)))[0]; p = mo + np.array([math.cos(a), math.sin(a)]) * v * tt * .55 + [0, 12 * tt * tt]; th = a + tt * 1.5 + k_; s_ = rg.uniform(10, 26); tri = p + np.array([[math.cos(th) * s_ * 1.6, math.sin(th) * s_ * 1.6], [math.cos(th + 2.3) * s_, math.sin(th + 2.3) * s_], [math.cos(th - 2.3) * s_, math.sin(th - 2.3) * s_]])
            if k_ % 2: kit.shape(c, spl(tri, True, n=4), '#fbf6e2', 3.)
            else: kit.shape(c, spl(tri, True, n=5), '#e6fbff', 2.4)
        img = over(img, c.layer()); img = kit.flare(img, pc[0] - 60, pc[1], 420, '#fff3c8', .14 if hold else .08)
        c = Cel(ss=2); SP.burst_lines(c, pc[0], pc[1], 240, 900, 34, 9 + f, '#fffdf0', 16, .95); SP.burst_lines(c, pc[0], pc[1], 260, 620, 26, 19 + f, '#17131a', 7, .9); l = c.layer(); img = over(img, l)
        if f == 60: img = kit.impact(img, 'red', .22, .60, accent='#d8142a')
        if f == 65: img = kit.smear(img, -70, 24, 9)
        return don.put(img, f - 61, 'slam', flash=False, x=1430, y=250, size=.52, life=14, out=5, rot=-.10)

    # ---------------------------------------------------------------- E
    def segE(self, f):
        t = f / 24.; u = (f - 66) / 8.; sh = kit.shake(f, 5, 41); cam = Camera3D(MI((-20 - 1.0 * u, -13, 3.0)), MI((-7.8 - 2.0 * u, 20, 9.2 - 2.6 * u)), 965, cx=960 + sh[0], cy=560 + sh[1], roll=-.03); sky = self.pano.sky(cam, .2)
        th = float(np.interp(f, [66, 68, 70, 72, 73, 74, 75], [.20, .46, .84, 1.28, 1.46, 1.58, 1.64])); ax = np.cross(UP, unit(MI((-.9, .44, 0)))); q0 = MI(king_rest(0, 1.)); pv = MI((-11.4, 27, -1.)); q = np.array([pv + axis_rot(ax, th * (.10 + .90 * max(0, (k_ - 1) / 4.))) @ (p - pv) if k_ > 1 else p for k_, p in enumerate(q0)]); F = axis_rot(ax, th * 1.0 + .25) @ unit(MI((.1, -.86, .05))); R = frameR(F, -.5)
        tip = q[-1]; fc_ = MI((-27., 34., 0)); Xp, Yp, z0 = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); foam = ((np.exp(-((np.hypot(Xp - K[0], Yp - K[1]) - 5) / 4.) ** 2) * .8 + np.exp(-(np.hypot(Xp - fc_[0], Yp - fc_[1]) / (9 + 4 * u)) ** 2) * smooth(72, 74, f)) * (z0 > 0)).astype(f32); Rw = 8 + (f - 60) * 2.6; rings = [(K[0], K[1], Rw, 4, 1.1 * math.exp(-(f - 60) / 14.), 1.)]
        w = self.S.render(cam, 3 + t, DEEP, sun=SE, rings=rings, foam=foam); img = w.over(sky) + sea.glitter(cam, t, tuple(MI((.5, 1, .3))), 200)[..., :3] * .4
        sc = 1.02 - .12 * (f - 65); G = seaking(q, MI((.4, -.9, 0)), R, .95, t, hit=1., ripple=(sc, .20, .11), flare=-.6, whisk=MI((3, -1, 2))); r = F3.draw(cam, [G], SE, fog=('#bfe9ee', 30., 1 / 300.), lw=.05, lwmax=3.6); lay = kit.rim(r.rgba, -6, 4, '#fffbe2', .9, soft=.8); img = sea.under(img, lay, r.z, w)
        if f >= 72:
            tb = (f - 71.6) / 24.; a = np.linspace(0, 1, 40); rt = MI(np.stack([-14 - 22 * a, 27 + 10 * a, 0 * a], -1)); img = over(img, sea.sheet(cam, SP.curtain(rt + MI((0, 3, 0)), MI(np.stack([np.full(40, -2.), np.full(40, 2.), 15 + 8 * np.sin(a * 3.1)], -1)), tb, seed=2), seed=3, pal=DEEP, su=20, sv=5)); img = over(img, sea.sheet(cam, SP.curtain(rt - MI((0, 4, 0)), MI(np.stack([np.full(40, 1.), np.full(40, -5.), 13 + 6 * np.sin(a * 3.1)], -1)), tb, seed=5), seed=6, pal=DEEP, su=20, sv=5))
        ps = ships.pose(0, 0, .2 + .5 * math.exp(-((Rw - 28) / 7.) ** 2), 1.98, .10, .1); dl = ships.draw([(ships.dinghy(t, barrel_=True), ps)], cam, light=SE, lw=.9); img = sea.under(img, dl.layer, dl.z, w)
        o = ps[1] + ps[0] @ np.array([.3, 0, .42]); back = float(np.interp(f, [66, 67, 69, 71, 72], [1, .97, .55, .10, 0])); hit_p = MI((-9.4, 19.5, 15.2)); 
        def arm(pen, s, ua, fa, S):
            if s != 'R' or back <= 0: return False
            a0 = S.sh['R']; v = (hit_p - a0) * back; U = np.linspace(0, 1, 50); wig = np.sin(U * 16 - f * 2.1) * np.sin(U * math.pi) * (1.4 if f > 66 else .15) * back; P3 = a0 + v * U[:, None] + np.outer(wig, UP) + np.outer(wig * .5, MI((1, 0, 0))) + np.outer(np.sin(U * math.pi) * -1.2 * (f > 67), UP); rubber(pen, P3, np.interp(U, [0, .6, 1], [.10, .13, .30]), SKIN, rings=4, veins=False, w=1.8, seed=1); fd_ = unit(P3[-1] - P3[-4]); sd_ = unit(np.cross(fd_, UP)); small_hand(pen, P3[-1], np.stack([sd_, fd_, np.cross(sd_, fd_)], 1), 1.5, 'R', 'fist', SKIN, 2.0); return True
        l = self.L.pose(mix_pose(HITP, POSES['contra'], .0 if f < 70 else .5)); self.L.hat = f < 66; img = put(img, l.place(o, .41).draw(cam, f, light=SE, rim=dict(col='#fff6d0', w=2, dir=SE), arm_fn=arm, full=True, w=1.7, expr='shout', wind=tuple(MI((1.5, -1, .3))), hair=1.2)); self.L.hat = True
        from op import hat as HT                                        # the hat is blown off
        hp = o + MI((.5 + .25 * (f - 66), -.6 - .5 * (f - 66) * .6, 2.0 + .62 * (f - 66) - .022 * (f - 66) ** 2)); p, z = cam.project(hp)
        if f >= 66 and 0 < p[0] < W: img = over(img, HT.hat(float(p[0]), float(p[1]), .42 * cam.f / z * 1.5, yaw=f * .7, pitch=.5, roll=.5 * math.sin(f), wear=.5, f=f))
        return img

    # ---------------------------------------------------------------- F
    def segF(self, f):
        from chars import luffy as LU
        from op import hat as HT
        t = f / 24.; u = (f - 75) / 14.; tb = (f - 72) / 24.; cam = Camera3D((3.0, -5.0 + .4 * u, 1.5), (-10, 26, 6.2 + .5 * u), 1250, roll=.02); sky = self.pano.sky(cam, .6); Xp, Yp, z0 = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); foam = (np.exp(-((Yp - 24) / 7.) ** 2) * np.clip(tb * 2, 0, 1) * (z0 > 0)).astype(f32)
        w = self.S.render(cam, 3 + t, DEEP, sun=SUN, rings=[(-16, 30, 4 + 34 * tb, 6, 2.6, 1.)], foam=foam); img = w.over(sky) + sea.glitter(cam, t, (-.2, 1, .3), 220)[..., :3] * .45
        for j, (y0, vz, sd, hw) in enumerate(((34, 27, 2, 34), (28, 23, 5, 28), (22, 17, 9, 24))):
            a = np.linspace(0, 1, 56); rt = np.stack([-16 + (a - .5) * 2 * hw, y0 + 2 * np.sin(a * 9 + sd), 0 * a], -1); env = np.sin(math.pi * a) ** .6 * (1 + .28 * np.sin(a * 13 + sd)); img = over(img, sea.sheet(cam, SP.curtain(rt, np.stack([(a - .5) * 14, np.full(56, -3. - j * 2), vz * env], -1), tb, seed=sd), seed=sd, pal=DEEP, su=26, sv=7, holes=.38 + .30 * u, thin=.05 * j))
        from op import fx_break as FB
        c = Cel(ss=2); PW = dict(lit='#ffffff', mid='#d9f3f5', dark='#8fd0dc', ink='#3a93ac'); L_ = []; rq = np.random.default_rng(17)   # foam heads roll over the top of the wall
        for k_ in range(46):
            a_ = (k_ + rq.uniform(-.4, .4)) / 46; env = math.sin(math.pi * a_) ** .6 * (1 + .28 * math.sin(a_ * 13 + 2)); tt_ = tb * rq.uniform(.55, .98); q_, z_ = cam.project(np.array([-16 + (a_ - .5) * 60 + (a_ - .5) * 14 * tt_, 30 - 4 * tt_, max(.2, 25 * env * tt_ - 4.9 * tt_ * tt_)])); rr_ = (14 + 62 * rq.random() ** 1.5) * min(1.4, .25 + tb * 1.9); L_.append((float(q_[0]), float(q_[1]), rr_, 300 + k_, f * .10 * (1 if k_ % 2 else -1) + k_, .0)); L_.append((float(q_[0]) + rr_ * .8, float(q_[1]) + rr_ * .5, rr_ * .6, 400 + k_, -f * .08 + k_, .0))
        FB.puffs(c, sorted(L_, key=lambda q: q[1]), PW, (.6, -.8), 1.6, 5); l_ = c.layer(); img = over(img, l_, .92); img = screen(img, cv2.GaussianBlur(l_[..., :3], (0, 0), 22) * .30)
        rb = smooth(77, 82, f); img = screen(img, SP.rainbow(760, 1180, 900, 30, .62 * rb) * (smooth(1000, 300, grid()[1]) * smooth(0, 500, grid()[0] + 200))[..., None])
        rg = np.random.default_rng(21); n = 170; p0 = np.stack([rg.uniform(-18, 16, n), rg.uniform(-2, 24, n), rg.uniform(9, 19, n)], -1); v = np.stack([rg.normal(0, 1, n), rg.normal(-2, 1, n), np.full(n, -3.)], -1); tt = max(0., tb - .1) * 1.0; img = over(img, sea.drops(cam, p0 + v * tt + np.array([0, 0, -4.9 * tt * tt]), v + [0, 0, -9.8 * tt], rg.uniform(.02, .09, n), DEEP, shutter=1 / 40.))
        # ---- Luffy
        on = f >= 84; hitch = (0, -7, -10, -4)[(f - 85) % 4] if f >= 85 else 0; ex = 'smile' if f < 79 else ('smile', 'grin', comp.ease((f - 79) / 4.)) if f < 84 else 'grin'
        bob = 5 * math.sin(t * 5); cx, cy = 985, 520 + bob + hitch + (6 if f == 84 else 0) - (10 if f == 79 else 0); yaw = float(np.interp(f, [75, 78, 80, 84, 89], [-.42, -.42, -.20, .16, .20])); pit = float(np.interp(f, [75, 79, 82, 84, 86], [.30, .30, .12, .02, .06])); roll = float(np.interp(f, [75, 79, 82, 85, 89], [-.10, -.12, -.04, .07, .06]))
        l = LU.head(cx, cy, 168, yaw, pit, roll, expr=ex, light='day', f=f, hat=True if on else False, shirt='vest', scar=False, body=True, wind=1.6, **(dict(hk=dict(bounce=HT.settle(f, 84, .8))) if on else {}))
        hx = float(np.interp(f, [75, 77, 79, 80, 81, 84], [470, 560, 690, 688, 720, cx - 20])); hy = float(np.interp(f, [75, 76, 77, 78, 79, 80, 81, 82, 83, 84], [-160, -70, 30, 130, 214, 226, 214, 230, 252, cy - 190])); hr = float(np.interp(f, [75, 79, 80, 84], [-2.4, -.42, -.36, 0])); hw = 560
        hatl = HT.hat(hx, hy, hw, yaw=float(np.interp(f, [75, 79, 84], [3.4, .3, yaw])), pitch=float(np.interp(f, [75, 79, 82, 84], [-.6, .34, .16, pit])), roll=hr, light='day', wear=.5, bounce=HT.settle(f, 79, .9), f=f) if not on else None
        armv = f >= 77
        def arm_layer():
            """one drawn contour, shoulder to knuckles; the fingers go over the brim, the thumb under it"""
            ink, bs, sd = '#17131a', SKIN['base'], SKIN['shade']; L_ = np.array([.62, -.78]); k = 1.06; rs = float(np.interp(f, [76, 77, 78, 79], [0, .25, .74, 1.])); opn = float(np.interp(f, [77, 78, 79], [1., .75, 0.]))
            def tip(al, y0): ys, xs = np.where(al > .5); x0 = xs.min(); return np.array([x0, y0 + ys[xs < x0 + 8].mean()], float)
            if on: y0 = max(0, int(cy - 205)); T, ang, hm = tip(l[y0:int(cy - 85), :int(cx), 3], y0), roll + .05, l[..., 3]
            elif f >= 79: T, ang, hm = tip(hatl[..., 3], 0), hr, hatl[..., 3]
            else: T, ang, hm = np.array([526., 411.]), -.42, None
            b_, m_ = np.array([math.cos(ang), math.sin(ang)]), np.array([math.sin(ang), -math.cos(ang)]); T = T + b_ * 15; T = T + (np.array([cx - 300., cy + 560.]) - T) * (1 - rs); Hp = lambda p, q: T + k * (b_ * p + m_ * q)
            W_ = Hp(-74, -96); S = np.array([cx - 212., cy + 288.]); E = np.array([S[0] - 228 - .24 * max(0., 735 - W_[0]), S[1] + 64 + 420 * (1 - rs)])
            e1, e2 = unit(S - E), unit(W_ - E); l1, l2 = np.linalg.norm(S - E), np.linalg.norm(W_ - E); bis = unit(e1 + e2); half = math.acos(float(np.clip(e1 @ e2, -.95, .95))) / 2; d = min(57 / math.sin(half), 150.); C = E + bis * d; al_ = d * math.cos(half); n1 = unit(bis - e1 * (bis @ e1)); n2 = unit(bis - e2 * (bis @ e2)); tc, t2 = 1 - al_ / l1, al_ / l2; tb_ = max(.30, tc)
            U = lambda t_, w_: S + (E - S) * t_ + n1 * w_; F_ = lambda t_, w_: E + (W_ - E) * t_ + n2 * w_; fr = lambda q: t2 + (1 - t2) * q
            up = spl([U(-.16, 14), U(-.07, 44), U(.05, 64), U(.18, 67), U(.20 + (tb_ - .20) * .42, 68), U(.20 + (tb_ - .20) * .80, 67), C], n=10); fo = spl([C, F_(fr(.12), 63), F_(fr(.36), 59), F_(fr(.64), 47), F_(fr(.86), 38), F_(1., 34), Hp(-27, -90), Hp(-6, -73)], n=10)
            ou = spl([Hp(-26, 66), Hp(-56, 54), Hp(-84, 12), Hp(-100, -42), F_(.96, -36), F_(.84, -41), F_(.66, -50), F_(.46, -59), F_(.28, -62), F_(.13, -56), E - bis * 53, U(.86, -50), U(.60, -57), U(.32, -65), U(.06, -70), U(-.14, -66)], n=10); web = spl([Hp(-6, -73), Hp(-8, -52), Hp(-12, -32), Hp(-16, -18), Hp(-14, -2), Hp(-4, 14), Hp(0, 40), Hp(-26, 66)], n=8); body = np.vstack([up, fo, web, ou])
            def digit(P, R_, rot=0.):
                P = np.array(P, float); cr, sr = math.cos(rot), math.sin(rot); P = P[0] + (P - P[0]) @ np.array([[cr, sr], [-sr, cr]]); Q = spl([Hp(*q) for q in P], n=12); n_ = len(Q); tg = np.gradient(Q, axis=0); tg /= np.linalg.norm(tg, axis=1)[:, None]; nr = np.c_[tg[:, 1], -tg[:, 0]]; w_ = k * np.interp(np.linspace(0, 1, n_), np.linspace(0, 1, len(R_)), R_); a = np.linspace(0, math.pi, 12)[1:-1]; cap = Q[-1] + w_[-1] * (np.cos(a)[:, None] * nr[-1] + np.sin(a)[:, None] * tg[-1] * 1.05); return np.vstack([Q + nr * w_[:, None], cap, (Q - nr * w_[:, None])[::-1]])
            FG = [digit([(-32, 47), (10, 65), (54, 65), (92, 44)], (17, 17, 16, 12.5), opn * .55), digit([(-36, 37), (10, 55), (58, 53), (99, 28)], (18.5, 18, 17, 13), opn * .40), digit([(-36, 24), (8, 42), (52, 39), (90, 14)], (19, 18.5, 17, 13), opn * .24), digit([(-34, 12), (4, 27), (40, 23), (71, 3)], (19, 18, 16.5, 12.5), opn * .08)]; TH = digit([(-30, -54), (4, -50), (36, -36), (67, -16)], (24, 20, 17.5, 13), -opn * .55)
            c = Cel(ss=2)
            for Q in [TH] + FG: c.fill(Q, sd); c.clip(Q); c.fill(Q + L_ * 9, bs); c.unclip(); c.line(Q, 4.4, ink, taper=(.02, .02))
            c.fill(body, sd); c.clip(body); c.fill(body + L_ * 26, bs)
            c.fill(spl([U(.16, 4), U((.2 + tb_) / 2, -6), U(tb_ + .02, 22), U((.2 + tb_) / 2, 12)], True, n=8), sd, .85)          # under the biceps
            c.fill(spl([C + bis * 4, C - bis * 22 + e1 * 20, C - bis * 46 + e1 * 4, C - bis * 24 - e1 * 8], True, n=6), sd); c.unclip()
            c.line(np.vstack([up[:-1], C[None]]), 5.2, ink, taper=(.05, .12)); c.line(fo, 5.0, ink, taper=(.10, .10)); c.line(ou, 5.4, ink, taper=(.08, .05)); c.line(web[int(len(web) * .29):int(len(web) * .66)], 4.2, ink, taper=(.10, .20))
            c.line(spl([C, C - bis * 24 + e1 * 3, C - bis * 50 + e1 * 10], n=8), 4.4, ink, taper=(.02, .85)); c.line(spl([C - bis * 16 - e1 * 14, C - bis * 34 - e1 * 14, C - bis * 46 - e1 * 8], n=8), 2.6, ink, taper=(.2, .8))
            c.line(spl([U(tb_ + .01, 30), U((.24 + tb_) / 2, 13), U(.20, 22)], n=10), 2.8, ink, taper=(.2, .7)); c.line(spl([U(.22, 62), U(.17, 36), U(.10, 12)], n=8), 3., ink, taper=(.1, .8)); c.line(spl([F_(fr(.06), 12), F_(fr(.34), 22), F_(fr(.62), 20)], n=10), 2.5, ink, taper=(.3, .7)); pe = np.array([-bis[1], bis[0]]); c.line(spl([E - bis * 30 + pe * 22, E - bis * 24, E - bis * 32 - pe * 20], n=8), 3., ink, taper=(.4, .5))
            for q in (17, 29, 42): c.line(spl([Hp(4, q + 2), Hp(-10, q - 2), Hp(-26, q - 7)], n=6), 3., ink, taper=(.03, .85))
            for q, p in ((34, 22), (49, 30), (62, 34)): c.line(spl([Hp(p - 7, q + 5), Hp(p, q + 8), Hp(p + 8, q + 5)], n=5), 2.2, ink, taper=(.4, .4))
            c.line(spl([Hp(57, -27), Hp(65, -23), Hp(71, -15)], n=5), 2.2, ink, taper=(.3, .4)); c.line(spl([Hp(-48, -88), Hp(-34, -66), Hp(-30, -44)], n=8), 2.4, ink, taper=(.3, .7)); arm_ = kit.rim(c.layer(), -3, 2, '#fff4cc', .45, soft=.6)
            if hm is not None:                                             # the fingers' own shadow on the straw
                c = Cel(ss=2)
                for Q in FG: c.fill(Q + np.array([-5., 13.]), '#4a2a10', 1.)
                s_ = c.layer() * hm[..., None] * .42; arm_ = arm_ + s_ * (1 - arm_[..., 3:])
            c = Cel(ss=2); c.fill(np.array([(cx - 275, cy + 350), (cx - 190, cy + 350), (cx - 170, cy + 640), (cx - 275, cy + 640)]), sd, 1.); lat = c.layer() * l[..., 3:]     # the raised arm's shadow on the flank
            c = Cel(ss=2); vs = np.array([(cx - 128, cy + 226), (cx - 176, cy + 250), (cx - 222, cy + 300), (cx - 236, cy + 372), (cx - 214, cy + 452), (cx - 190, cy + 640), (cx - 120, cy + 640), (cx - 120, cy + 300)]); Q = spl(vs, True, corners=(0, 5, 6), n=6); c.fill(Q, MAT['red']['base']); c.clip(Q); c.fill(spl([(cx - 270, cy + 336), (cx - 204, cy + 346), (cx - 164, cy + 400), (cx - 156, cy + 480), (cx - 172, cy + 660), (cx - 270, cy + 660)], True, n=6), MAT['red']['shade']); c.unclip(); c.line(spl(vs[:6], n=8), 5.2, ink, taper=(.2, .2)); c.line(spl([(cx - 196, cy + 292), (cx - 182, cy + 372), (cx - 168, cy + 450)], n=8), 2.6, '#5f1020', taper=(.2, .5)); c.line(np.array([(cx - 121, cy + 292), (cx - 121, cy + 460), (cx - 121, cy + 640)], float), 3.6, '#3d0a16', taper=(.1, .02))
            if f >= 82:
                for q in (343, 428, 513): a = np.linspace(0, 6.28, 18); kit.shape(c, np.c_[cx - 133 + np.cos(a) * 7.5, cy + q + np.sin(a) * 7.5], '#f0b63c', 2.2)
            v_ = c.layer(); arm_ = arm_ + lat * (1 - arm_[..., 3:]); return v_ + arm_ * (1 - v_[..., 3:])
        if hatl is not None and f < 79: img = over(img, hatl)
        if armv and f < 82: img = over(img, l); img = over(img, hatl) if f >= 79 else img; img = over(img, arm_layer())
        elif armv and not on: img = over(img, l); img = over(img, hatl); img = over(img, arm_layer())
        elif on: img = over(img, l); img = over(img, arm_layer()) if f < 88 else over(img, arm_layer())
        else: img = over(img, l)
        c = Cel(ss=2); rg = np.random.default_rng(33)                   # near rain, out of focus
        for i in range(34):
            x, y0, s = rg.uniform(0, W), rg.uniform(-900, 500), rg.uniform(3, 9); y = y0 + (f - 76) * (70 + 8 * s)
            if f > 76 and -60 < y < H + 60: c.fill(spl([(x, y - s * 5), (x + s * .6, y), (x, y + s), (x - s * .6, y)], True, n=5), '#f2feff', .55)
        img = over(img, cv2.GaussianBlur(c.layer(), (0, 0), 1.6)); return img

    def frame(self, f):
        f = int(f); im = self.segA(f) if f < 30 else self.segB(f) if f < 40 else self.segC(f) if f < 60 else self.segD(f) if f < 66 else self.segE(f) if f < 75 else self.segF(f)
        return im.astype(f32)
