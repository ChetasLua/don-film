"""c15  one camera through a marine fleet: the burning heel (f15) / the mast run and the laser (f40) / the hull-sized fist (f60) / back to back on the lion (f70 75 80)."""
import numpy as np, math, cv2
from op import W, H, comp, kit, sea, ships, don, fleet3 as F3, splash as SP, fx_fire as FF, fx_break as FB
from op.paint import rgb, f32, grid, smooth
from op.ink import Cel, spl, over_roi
from op.cam import Camera3D, rot_z, rot_x, rot_y
from op.comp import over, screen, lerp_keys
from op.fig_pose import P, POSES, seq, mix_pose, add_pose, sprint, Rz, Rx, Ry, unit
from chars.act import Actor, rubber, View, Pen, SKIN, MAT, M

SUN = np.array([.72, .12, .62]); FOG = ('#c3e4ee', 110., 1 / 620.); UP = np.array([0, 0, 1.])
PAL = dict(sea.DAY, deep='#0a3a78', body='#1260a8', mid='#2a86c6', light='#63b6e2', haze=1 / 700.)
BPOS = np.array([4., 92., 0.]); SPOT = dict(Z=(20.5, -.64, 12.0), S=(20.5, .66, 11.97), L=(20.60, .02, 11.92)); PV = np.array([0, .25, 9.]); Z3 = None
AXU = P(root=(0, -.02, .05), pel=(-44, 0, 26), wst=(14, 0, -4), cst=(18, 0, -6), nck=(10, 0, 0), hed=(6, 0, 0), hL=(.80, -.25, -.12), hR=(.82, -.20, .10), gL='splay', gR='fist', fR=(.14, .40, .98), fL=(.06, -.06, .12), kR=(.2, .2, 1), kL=(.3, 1, .3), tR=(0, 40, 0), tL=(10, -30, 0))
AXD = P(root=(0, .04, .06), pel=(22, -8, -10), wst=(14, 0, 4), cst=(12, 0, 6), nck=(8, 0, 0), hed=(10, 0, 0), hL=(.86, -.30, .30), hR=(.80, -.45, .34), gL='fist', gR='splay', fR=(.02, .34, .02), fL=(.12, -.26, .42), kR=(.1, 1, .2), kL=(.6, .8, .3), tR=(0, 42, 0), tL=(20, -50, 0))
TUCK = P('twist_tuck'); RUNARMS = dict(gL='grip', gR='grip', hL=(.62, -.52, -.40), hR=(.62, -.52, -.40), eL=(.6, -1, .1), eR=(.6, -1, .1), wL=(-60, 10, 0), wR=(-60, 10, 0))
CROSS = P('run_fly', lead='chest', gL='grip', gR='grip', hL=(-.30, .52, .10), hR=(-.30, .56, .24), eL=(1, -.1, -.2), eR=(1, -.1, -.2), wL=(-90, 0, 0), wR=(-90, 0, 0), cst=(10, 0, 0), wst=(14, 0, 0))
SLASH = P('leap', gL='grip', gR='grip', lead='chest', hL=(.95, .10, -.10), hR=(.95, .12, .22), eL=(.6, -.4, -.6), eR=(.6, -.4, -.6), wL=(-20, 10, 0), wR=(-20, 10, 0), cst=(-12, 0, 0), wst=(4, 0, 0), pel=(10, 0, 0), nck=(0, 0, 0), hed=(-4, 0, 0))
BITE = P(root=(0, 0, .12), pel=(6, 0, 10), wst=(6, 0, 4), cst=(4, 0, 8), nck=(4, 0, 16), hed=(2, -4, 14), hR=(-.27, .40, .17), eR=(1, .1, -.8), gR='fist', wR=(-30, -50, 0), hL=(.5, .3, -.5), eL=(.8, -.6, 0), gL='fist', fL=(.10, .22, .34), fR=(.06, -.30, .22), kL=(.3, 1, .3), kR=(.2, 1, -.4), tL=(10, -30, 0), tR=(4, -60, 10))
BLOW = P(BITE, cst=(-14, 0, -10), wst=(-6, 0, -6), nck=(-6, 0, 14), hed=(-10, 0, 14))
COCK = P('punch_wind', lead='root', hL=(.30, .80, .05), gL='splay', root=(0, 0, .12), hR=(.30, -.70, .62), eR=(1, .2, .5), fL=(.10, .24, .30), fR=(.06, -.36, .20), kL=(.2, 1, .2), kR=(.2, 1, -.4), nck=(2, -4, 40), hed=(6, -4, 30))
THROW = P('dive_punch', hL=(.5, -.6, .3), nck=(-4, 4, -30), hed=(6, 4, -26))
STZ = P('contra', hL=(.42, .34, -.66), hR=(.30, .34, .30), eR=(.9, -.2, -.6), gL='grip', gR='grip', wL=(-50, 0, 0), wR=(-20, 30, 0), fL=(.17, .10, .043), fR=(.15, -.14, .043), kL=(.5, 1, 0), kR=(.6, 1, 0), cst=(-6, 0, 10), nck=(0, 0, 6), hed=(4, 0, 12), root=(0, 0, -.075))
STS = P('pockets', fL=(.06, .02, .043), fR=(.13, .20, .10), kR=(.5, 1, .2), tR=(16, 30, 0), cst=(0, 0, -10), hed=(4, 6, 8), root=(-.02, 0, -.045))
STL = P(root=(0, 0, -.085), pel=(4, 0, 6), wst=(2, 0, 0), cst=(-6, 0, -10), nck=(2, 0, 4), hed=(6, 0, 4), hL=(.06, .20, .62), eL=(1, .1, .3), gL='open', wL=(-20, 20, 0), hR=(.30, .62, -.22), eR=(1, -.3, -.5), gR='fist', wR=(-60, -6, 0), fL=(.19, .08, .043), fR=(.17, -.12, .043), kL=(.6, 1, 0), kR=(.7, 1, 0), tL=(28, 0, 0), tR=(30, 0, 0))


def fallR(th): c, s = math.cos(th), math.sin(th); return np.array([[1, 0, 0], [0, c, s], [0, -s, c]])


def theta(f): return float(np.interp(f, [18, 20, 24, 28, 32, 36, 40, 44, 48], [0, .03, .13, .28, .47, .70, .94, 1.14, 1.24]))


def K_(f, keys): return np.array(lerp_keys(f, keys, e=lambda t: t * t * (3 - 2 * t)))


def smoothpath(f, keys):
    from scipy.interpolate import PchipInterpolator
    k = np.array([q[0] for q in keys], float); v = np.array([q[1] for q in keys], float); return PchipInterpolator(k, v)(np.clip(f, k[0], k[-1]))


class Shot:
    id = 'c15'; n = 90; finish = dict(bloom=.34, vig=.14)
    EVENTS = [(0, 'whoosh', dict(note='falling with Sanji, wind, fire roar', dur=14)), (2, 'cannon', dict(size=.5, note='far guns')), (9, 'cannon', dict(size=.4)), (15, 'impact', dict(size=1.0, note='burning heel on the mast')), (16, 'crack', dict(note='mast splits lengthwise, six frames', dur=6)), (20, 'impact', dict(size=.7, note='deck bursts')),
              (22, 'creak', dict(note='mast half falls', dur=22)), (24, 'run', dict(note='boots on raw wood, every 3 f', dur=18)), (33, 'charge', dict(note='laser whine')), (35, 'laser', dict(note='beam past the lens', dur=9)), (40, 'slash', dict(size=1.0, note='beam cut in two')), (43, 'explosion', dict(size=.6, note='the halves land far off')), (44, 'whoosh', dict(note='whip pan')),
              (48, 'bite', dict(note='thumb')), (49, 'blow', dict(note='cheeks like a bellows, the swell runs from the hand up the arm', dur=4)), (53, 'swell', dict(note='the whole arm bursts into a black hull-sized fist', dur=4)), (57, 'stretch', dict(note='cocked')), (58, 'whoosh', dict(size=1.)), (60, 'impact', dict(size=1.4, note='ドン! warship driven into the sea, hit stop 5 f')), (65, 'explosion', dict(size=1.2, note='hull folds, column of water')), (70, 'land', dict(note='Zoro')), (75, 'land', dict(note='Sanji')), (80, 'land', dict(note='Luffy; ring wave lifts the bow')), (83, 'debris', dict(note='wreckage rain', dur=7))]

    def setup(self):
        self.pano = SP.Pano('day'); self.S = sea.Sea(3, hs=1.7, wind=(1., .4)); self.A0 = F3.warship(1, main=False); self.stump = F3.Geo().cyl((0, 0, 3.6), (0, 0, 9.4), .95, .90, 'mast', 12)
        self.mast = F3.mast(F3.Geo(), 0, 36, yards=((.50, 18, .40), (.92, 13.5, .30)), r=.95, seed=3, beam=11)
        far = [F3.warship(k, lod=1) for k in (5, 6, 7)]; R = rot_z
        self.back = [far[0].at(R(.35), (-58, -78, 0)), far[1].at(R(-.5), (44, -92, 0)), far[2].at(R(.15), (108, -70, 0)), far[0].at(R(1.0), (-120, -150, 0)), far[1].at(R(-.2), (8, -185, 0)), far[2].at(R(.6), (70, -210, 0)), far[0].at(R(-.8), (-150, -40, 0)), far[1].at(R(.3), (160, -160, 0))]
        self.front = [far[1].at(R(.3), (-70, 150, 0)), far[2].at(R(-.4), (96, 150, 0)), far[0].at(R(.8), (-140, 110, 0)), far[1].at(R(2.6), (150, 96, 0)), far[2].at(R(.1), (10, 220, 0)), far[0].at(R(-.2), (64, 250, 0))]
        self.B1 = F3.warship(2, s1=.52, jag=1.3); self.B2 = F3.warship(2, s0=.52, jag=1.3, main=False); self.Bw = F3.warship(2)
        self.sanji, self.zoro, self.luffy, self.mar = Actor('sanji'), Actor('zoro', swords=True), Actor('luffy', 'grown'), Actor('marine')
        self.D1 = F3.Debris((0, .5, 33), 46, 3, vel=(3, 13), up=(-2, 12), size=(.25, 1.7), spread=(.5, .5, 2.5), t0=15 / 24.); self.D2 = F3.Debris((0, 0, 6), 70, 4, vel=(4, 15), up=(6, 22), size=(.4, 3.0), spread=(1.6, 1.6, .5), t0=20 / 24.)
        self.D3 = F3.Debris(BPOS + [0, 0, 5], 150, 5, vel=(4, 22), up=(8, 33), size=(.6, 5.0), spread=(7, 4, 1.5), t0=0.); self.D4 = F3.Debris((2, 50, 34), 34, 8, vel=(1, 6), up=(-10, 2), size=(.5, 2.6), spread=(9, 5, 5), t0=0., spin=7)
        from op import hands as HD; self.HD = HD

    # ---------------------------------------------------------------- camera
    def ride(self, f):
        d, _ = self.zoro_state(f); D = float(np.interp(f, [22, 24, 28, 34, 38, 43, 45], [9, 6.4, 4.0, 3.1, 2.9, 2.3, 2.1])); Rm = fallR(theta(f)); return PV + Rm @ np.array([.95, -1.42, d + D]), PV + Rm @ np.array([-.02, -float(np.interp(f, [34, 37, 40, 44], [.92, 1.25, 1.50, 1.9])), d - .6])

    def camera(self, f):
        if f < 15.01:
            p = self.sanji_state(f); e = p + K_(f, [(0, (2.0, 2.7, 2.2)), (6, (1.45, 2.25, 1.0)), (11, (1.5, 2.3, .2)), (15, (2.5, 3.3, 1.5))]); t = p + K_(f, [(0, (.1, 0, .2)), (7, (0, 0, .55)), (12, (0, 0, .7)), (15, (.1, .1, .2))])
        elif f < 26.5: e = smoothpath(f, [(15, (2.7, 4.1, 38.6)), (17, (2.5, 3.9, 38.2)), (20, (2.3, 3.2, 35.5)), (26.5, tuple(self.ride(26.5)[0]))]); t = smoothpath(f, [(15, (0, .5, 37.0)), (16, (0, .4, 35)), (19, (0, .3, 24)), (22, (0, .4, 19)), (26.5, tuple(self.ride(26.5)[1]))])
        elif f < 45: e, t = self.ride(f)
        else:
            e = smoothpath(f, [(45, tuple(self.ride(45)[0])), (45.7, (-2.6, 42, 22.7)), (46.6, (-1.02, 48.75, 23.3)), (49.5, (-.82, 48.95, 23.28)), (52.2, (-.75, 49.8, 23.3)), (54, (1.4, 54.6, 24.4)), (55.4, (9.0, 51.0, 26.4)), (57, (13.5, 33.5, 29.5)), (58, (14.6, 36, 28.5)), (60, (15.5, 63, 18.5)), (64, (15.0, 63.6, 18.2)), (67, (11, 50, 15)), (70, (3.0, 40.2, 11.4)), (76, (1.9, 41.6, 12.0)), (81, (.80, 43.4, 12.6)), (90, (-.12, 46.2, 12.45))])
            t = smoothpath(f, [(45, tuple(self.ride(45)[1])), (45.7, (.5, 47.2, 23.3)), (46.6, (.42, 47.1, 22.72)), (48, (.44, 47.1, 22.92)), (49.5, (.45, 47.1, 23.08)), (52.2, (.75, 47.15, 23.25)), (54, (2.3, 46.6, 25.3)), (55.4, (2.2, 45.0, 26.2)), (57, (1.4, 58, 18.5)), (58, (2.5, 72, 12)), (60, (4, 91, 6.6)), (64, (4, 91, 6.8)), (67, (2, 72, 9)), (70, (-.6, 52, 11.7)), (76, (-.6, 50, 12.5)), (81, (-.75, 50, 13.0)), (90, (-.8, 50, 12.30))])
        fo = float(np.interp(f, [0, 15, 24, 45, 46.6, 49.5, 52.2, 54, 55.4, 57, 60, 66, 70, 76, 81, 90], [1000, 1000, 900, 880, 1500, 1640, 1250, 860, 800, 860, 1060, 960, 1080, 1180, 1420, 1840])); ro = float(np.interp(f, [0, 14, 20, 28, 40, 45, 48, 52, 57, 60, 65, 70, 90], [.22, .05, -.08, .05, -.13, -.20, .05, .12, -.16, -.08, -.05, .03, .0])); sh = np.zeros(2)
        for f0, amp, dc in ((15, 16, 3.), (20, 12, 2.5), (40, 13, 2.5), (60, 20, 3.2), (65, 18, 4), (70, 5, 1.5), (75, 5, 1.5), (80, 8, 2)): sh = sh + kit.shake(f, amp, int(f0), dc, f0)
        if f > 62: lf_ = 2.6 * math.exp(-((9 + 46 * ((f - 65) / 24. + .06) - 42) / 8.) ** 2) * .6 * .85 * smooth(64, 68, f); e = np.asarray(e, float) + [0, 0, lf_]; t = np.asarray(t, float) + [0, 0, lf_]
        return Camera3D(e, t, fo, cx=960 + sh[0], cy=540 + sh[1], roll=ro)

    # ---------------------------------------------------------------- pieces
    def mast_geo(self, f):
        if f < 15.5: return [self.mast, self.stump]
        sp = float(np.interp(f, [15, 16, 18, 20], [.05, .30, .80, 1.])); bd = float(np.interp(f, [15, 18, 22, 30], [.25, .9, 1.1, .5])); th = theta(f); out = [self.stump]
        for sd in (1, -1):
            g = F3.splitmast(0, 27.6, .92, sd, sp, bd, part_char=1.); r = .5
            for zy, w, yr in ((8.5, 18, .40), (21.2, 13.5, .30)):
                op = np.clip((zy / 27.6 - (1 - sp)) / sp, 0, 1) ** 1.6 * bd; a, b = np.array([1.0, sd * (.1 + op), zy]), np.array([1.0, sd * (w / 2 + op), zy]); g.cyl(a, b, yr, yr * .8, 'yard', 8); g.cyl(a, a - [0, sd * .5, 0], yr * .8, .04, 'raw', 6, caps=False)
            swg = math.sin(f * .6) * .5
            for z0, z1, w0, w1 in ((-5.2, 8.0, 9.0, 8.6), (9.0, 20.7, 8.2, 6.4)):
                n = 10; u = np.linspace(0, 1, n); v = np.linspace(0, 1, n); Pq = np.zeros((n, n, 3)); U = np.zeros((n, n, 2))
                for j, vv in enumerate(v):
                    z = z1 + (z0 - z1) * vv; op = np.clip((max(z1, 0) / 27.6 - (1 - sp)) / sp, 0, 1) ** 1.6 * bd; wd = w1 + (w0 - w1) * vv
                    for i, a in enumerate(u): rag = .5 * math.sin(vv * 23 + z0) * (a < .05) * vv; Pq[j, i] = (1.3 + (2.6 + swg) * math.sin(math.pi * min(1, vv * .9 + .05)) * (1 - .5 * a * a) + .4 * vv * math.sin(f * .7 + a * 4), sd * (.15 + op + a * wd + rag), z + .8 * (1 - a) * vv * (f > 18)); U[j, i] = (sd * a * wd, (vv - .5) * (z0 - z1))
                g.grid(Pq, 'gull' if z0 < 0 else 'sail', uv=U, two=True)
            out.append(g.at(fallR(th if sd > 0 else -th * .9), PV * [1, sd, 1]))
        return out

    def sanji_state(self, f):
        if f < 15: z = 50.4 - 13.6 * (f / 15.) ** 1.7; y = 2.6 - 2.2 * (f / 15.); x = -1.4 + 1.2 * (f / 15.)
        else: z = float(np.interp(f, [15, 16, 18, 20, 22], [36.8, 33.5, 21, 9.5, 7.2])); y, x = .45, -.15
        return np.array([x, y, z])

    def zoro_state(self, f): d = float(np.interp(f, [20, 24, 30, 37, 38, 41, 43, 46], [5.6, 8.8, 12.8, 17.6, 18.0, 19.4, 21.8, 30])); lift = float(np.interp(f, [37, 38, 40, 42, 43, 45], [0, .22, .40, .62, 1.1, 2.6])); return d, lift

    def luffy_pos(self, f): return smoothpath(f, [(44, (.4, 46.4, 20.6)), (49, (.5, 47.1, 22.6)), (53, (.5, 47.2, 22.9)), (57, (.5, 46.6, 22.8)), (60, (.5, 49.5, 21.6)), (65, (.5, 49.5, 21.4)), (70, (0, 49.4, 23.4)), (74, (-.6, 49.6, 22)), (80, (-1.0, 50.0, 12.4))])

    def fist_state(self, f):
        C = smoothpath(f, [(50, (.8, 47.9, 23.4)), (51, (2.0, 47.2, 24.4)), (52, (3.0, 46.6, 25.6)), (53, (4.0, 45.6, 27.0)), (54, (4.6, 44.4, 28.2)), (55, (4.2, 43.0, 29.2)), (57, (2.2, 39.6, 30.4)), (58, (2.0, 58, 24)), (59, (3.0, 79, 13.5)), (60, (4.0, 89.5, 7.6)), (64, (4.0, 90.0, 7.0)), (65, (4., 90.5, 5.6)), (67, (3.5, 84, 8)), (69, (2.5, 66, 16)), (71, (1, 52, 22))]); r = float(np.interp(f, [50, 51, 52, 53, 54, 55, 56, 57, 60, 65, 67, 69, 71], [.16, .75, 1.45, 2.2, 3.2, 4.3, 5.3, 6.0, 7.6, 7.6, 4.6, 1.6, .25])); wob = .05 * math.sin(f * 2.2) * (f < 58); return C, r * (1 + wob)

    def bigfist(self, cam, C, r, fdir, squash=1.0, w=6.):
        c, z = cam.project(C); z = float(z)
        if z < .6: return None
        px = r * cam.f / z; v = lambda q: np.array([cam.R[0] @ q, -cam.R[1] @ q, -cam.R[2] @ q]); y = unit(v(np.asarray(fdir, float))); zz = v(unit(np.array([.55, -.1, .9]) if fdir[2] > -.02 and fdir[1] < .9 else np.array([.80, -.25, .60]))); zz = unit(zz - y * (zz @ y)); Rf = np.stack([np.cross(y, zz), y, zz], 1); m = int(px * 2.6) + 60; x0, y0, x1, y1 = max(0, int(c[0] - m)), max(0, int(c[1] - m)), min(W, int(c[0] + m)), min(H, int(c[1] + m))
        if x1 - x0 < 8 or y1 - y0 < 8: return None
        cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=2); L = v(SUN); HD = self.HD; pal = dict(skin='#43355a', shade='#171020', line='#0a070d', light='#6c5890', nail='#3c3048', nail_shade='#221828', nail_dark='#0c080f')
        hh_ = HD.Hand('fist', 'R', Rf, (0, 0, 0), HD.HEAVY, pivot=(0, .95, 0), arm=.16)
        g = HD.render([hh_], HD.Cam(float(c[0]), float(c[1]), px / .62, max(2.4, z / r * .62)), tuple(L), 560, clip=(x0, y0, x1, y1)); HD.ink(cel, g, pal, float(np.clip(px * .020, 1.6, w)), 3)
        sp = px / .62; lit2 = np.array([L[0], -L[1]]); lit2 = lit2 / (np.hypot(*lit2) + 1e-6); pts = g.pts[0]; cel.clip([q_ for G_ in g.groups for q_ in G_['polys']])
        try: HD.fill_polys(cel, g, HD.rim_polys(g, v(unit(np.array([.62, .66, .2]))), lo=.42), '#8f76c4', .85)
        except Exception: pass
        for fi in ('index', 'middle', 'ring', 'little'):                # gloss bars on each finger, bone caps on each knuckle
            a_, b_, c_ = [np.asarray(pts[fi, k_]) for k_ in ('mcp', 'pip', 'dip')]
            for q0, q1, wd in ((a_, b_, .085), (b_, c_, .065)):
                d_ = q1 - q0; ln = np.hypot(*d_) + 1e-6
                if ln < sp * .06: continue
                n_ = np.array([-d_[1], d_[0]]) / ln; n_ = n_ * (1 if n_ @ lit2 > 0 else -1); m0, m1 = q0 + d_ * .24 + n_ * sp * .045, q0 + d_ * .78 + n_ * sp * .045; cel.fill(spl([m0, (m0 + m1) / 2 + n_ * sp * wd * .55, m1, (m0 + m1) / 2 - n_ * sp * wd * .20], True, n=6), '#b7a0e8', .85); cel.fill(spl([m0 * .8 + m1 * .2, (m0 + m1) / 2 + n_ * sp * wd * .30, m0 * .3 + m1 * .7], True, n=6), '#f3ecff', .95)
            th_ = np.linspace(0, 2 * math.pi, 22); cap = a_ + np.c_[np.cos(th_), np.sin(th_)] * sp * .105; cel.fill(cap + lit2 * sp * .035, '#6a5590', .9); cel.fill(a_ + lit2 * sp * .06 + np.c_[np.cos(th_) * .055, np.sin(th_) * .040] * sp, '#efe6ff', .95); arc = np.linspace(math.atan2(-lit2[1], -lit2[0]) - 1.15, math.atan2(-lit2[1], -lit2[0]) + 1.15, 14); cel.line(a_ + np.c_[np.cos(arc), np.sin(arc)] * sp * .13, max(1.6, px * .022), '#0a070d', taper=(.3, .3))
        wr_ = np.asarray(pts['wrist'])
        for j_, fi in enumerate(('index', 'middle', 'ring', 'little')):  # cords and veins run from the wrist to the knuckles
            k0 = np.asarray(pts[fi, 'mcp']); m_ = (wr_ * .45 + k0 * .55) + np.array([math.sin(j_ * 2.1), math.cos(j_ * 1.7)]) * sp * .05; L_ = spl([wr_ * .75 + k0 * .25, m_, k0 + (wr_ - k0) * .16], n=10); cel.line(L_, max(1.4, px * .018), '#0a070d', taper=(.3, .5), seed=j_); cel.line(L_ + lit2 * max(1.2, px * .016), max(1.2, px * .012), '#9c84cf', taper=(.3, .5), seed=j_, alpha=.9)
            if j_ < 3: f0_ = L_[len(L_) // 2]; cel.line(spl([f0_, f0_ + (k0 - wr_) * .10 + np.array([-(k0 - wr_)[1], (k0 - wr_)[0]]) * .10, f0_ + (k0 - wr_) * .22 + np.array([-(k0 - wr_)[1], (k0 - wr_)[0]]) * .16], n=6), max(1.2, px * .013), '#0a070d', taper=(.1, .7), seed=j_ + 9)
        cel.unclip()
        lay = cel.rgba(); a2 = np.array([y[0], -y[1]]); al = float(np.hypot(*a2)); xg, yg = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1)); cut = smooth(-1.16, -1.0, ((xg - c[0]) * a2[0] + (yg - c[1]) * a2[1]) / max(al, .25) / (px / .62 * max(al, .25))) if al > .5 else 1.; lay = lay * (np.asarray(cut, f32)[..., None] if al > .5 else 1.); d2 = np.array([L[0], -L[1]]); d2 /= np.hypot(*d2) + 1e-6; lay = kit.rim(lay, d2[0] * max(3, px * .030), d2[1] * max(3, px * .030), '#8b5cff', .95, soft=1.4); lay = kit.rim(lay, -d2[0] * max(3, px * .035), -d2[1] * max(3, px * .035), '#ffe9b8', .95, soft=1.2); return lay, x0, y0, z

    # ---------------------------------------------------------------- frame
    def frame(self, f):
        f = int(f); t = f / 24.; cam = self.camera(f); sky = self.pano.sky(cam, -.35 if f < 45 else .55); lookY = cam.R[2][1]
        hs = 60 <= f < 65; tB = (.05 if hs else (f - 65) / 24. + .06) if f >= 60 else 0.
        Xp, Yp, z0 = cam.uv((0, 0, 0), (1, 0, 0), (0, 1, 0)); foam = np.zeros((H, W), f32)
        def blob(x, y, r, a): return a * np.exp(-((Xp - x) ** 2 + (Yp - y) ** 2) / r ** 2) * (z0 > 0)
        foam += (np.exp(-(np.maximum(0, np.hypot(Xp / 26., Yp / 6.2) - 1) * 6.) ** 2) * .55 * (z0 > 0)).astype(f32)
        rings = []
        if f >= 60: Rw = 9 + 46 * tB; rings = [(BPOS[0], BPOS[1], Rw, 6., 6.0 * math.exp(-tB * 1.2), 1.)]; foam += blob(BPOS[0], BPOS[1], 16 + 30 * tB, 1.)
        wakes = [dict(x=-120 + 4 * t, y=-150, head=1.0, L=70, W=5)]
        w = self.S.render(cam, 5 + t, PAL, sun=tuple(SUN), foam=np.clip(foam, 0, 1).astype(f32), rings=rings); img = w.over(sky); img = img + sea.glitter(cam, t, tuple(SUN), 260) * .5
        # ---- geometry
        geos = []; glow = []
        if f < 47: geos += [self.A0] + self.mast_geo(f) + self.back
        else: geos += [self.A0, self.stump]
        if f >= 44: geos += self.front
        if f >= 44:
            if f < 60: geos.append(self.Bw.at(rot_z(.12), BPOS))
            else:
                a = .22 if hs else float(np.interp(f, [65, 68, 74, 82, 90], [.30, .62, .95, 1.12, 1.18])); zz = -.6 if hs else float(np.interp(f, [65, 70, 80, 90], [-1.2, -2.0, -5., -9.])); d = 1. if hs else float(np.interp(f, [65, 90], [1.5, 6.5]))
                geos += [self.B1.at(rot_z(.12) @ rot_y(a), BPOS + [-d, 0, zz]), self.B2.at(rot_z(.12) @ rot_y(-a), BPOS + [d, 0, zz]), self.D3.geo(tB)]
        if f >= 15: geos.append(self.D1.geo(t))
        if f >= 20 and f < 50: geos.append(self.D2.geo(t))
        if f >= 72: geos.append(self.D4.geo((f - 72) / 24.))
        if f < 24: p = self.sanji_state(f); glow.append((p, 3.4 + .4 * math.sin(f * 2), '#ff8a2a', .55))
        if 33 <= f < 46: d, lf = self.zoro_state(f); glow.append((PV + fallR(theta(f)) @ np.array([0, -1.5, d]), 5., '#ffd05a', .55 * smooth(33, 36, f)))
        near = (f >= 46 and f < 66)
        r = F3.draw(cam, geos, SUN, fog=FOG, glow=glow, lw=.034, lwmax=4.2); lay = kit.rim(r.rgba, -4, 3, '#fff3cf', .55, soft=.8); img = sea.under(img, lay, r.z, w); zmap = np.minimum(r.z, w.z)
        if f >= 44:
            sb = .5 * math.sin(t * 2.2); lift = 2.6 * math.exp(-((9 + 46 * tB - 42) / 8.) ** 2) if f >= 60 else 0; sp = ships.pose(-15.4, 67.2, -.8 + sb * .3 + lift * .6, -.872, .02 + .05 * lift, .02 * math.sin(t * 2)); self.sp = sp
            mdl = ships.sunny(t, 1.0, lod=2 if f >= 64 else 1); dl = ships.draw([(mdl, sp)], cam, light=tuple(SUN), lw=1.1, rim=('#fff3cf', 4, .6)); dm = (dl.z < zmap + .1).astype(f32); dm = np.where(dl.z > 1e8, cv2.dilate(dm * (dl.z < 1e8), np.ones((5, 5), np.uint8)), dm); L_ = dl.layer * dm[..., None]; img = over(img, L_); zmap = np.minimum(zmap, dl.z); self.dlz = dl.z
        # ---- water thrown by the hit
        if f >= 60:
            for k_, (r0, v, vz, sd) in enumerate(((7, 11, 27, 2), (10, 20, 14, 5))):
                Pc, Dp, Dv, rd = sea.splash_ring(cam, BPOS[0], BPOS[1], .10 + tB * .95, r0=r0, v=v, vz=vz, n=64, seed=sd, drop=.20); img = self.zover(img, sea.sheet(cam, Pc, seed=sd, pal=PAL, su=40, sv=9, holes=.55 + .5 * tB, z_out=True), zmap); img = over(img, sea.drops(cam, Dp, Dv, rd, PAL, zmask=zmap, maxpx=9))
            for x, y, age, s in self.D3.hits(tB)[:26]: img = self.small_splash(img, cam, x, y, age, s, zmap)
        # ---- far gunfire
        img = self.guns(img, cam, f)
        # ---- actors
        if f < 23: img = self.do_sanji(img, cam, f, t)
        if 20 <= f < 46: img = self.do_zoro(img, cam, f, t, zmap)
        if 45 <= f < 72: img = self.do_luffy(img, cam, f, t)
        if f >= 64: img = self.do_trio(img, cam, f, t)
        # ---- marines in the air
        if 16 <= f < 30: img = self.marines(img, cam, (f - 16) / 24., (0, 1, 8), 5, 3, zmap)
        if f >= 60: img = self.marines(img, cam, tB, BPOS + [0, -2, 7], 8, 9, zmap)
        # ---- hit drawings
        if f in (15, 16, 17): p = cam.project(np.array([0, .4, 36.4 - 3 * (f - 15)]))[0]; c = Cel(ss=2); SP.burst_lines(c, p[0], p[1], 130, 1300, 80, f, '#fff6d8', 11, .9); SP.burst_lines(c, p[0], p[1], 110, 620, 44, f + 9, '#ff9a2e', 12, .95); SP.burst_lines(c, p[0], p[1], 100, 420, 26, f + 19, '#17131a', 6, .9); img = over(img, c.layer()); img = kit.flare(img, p[0], p[1], 520 - 80 * (f - 15), '#ffca6a', .8 - .2 * (f - 15))
        if f == 15: img = kit.impact(img, 'red', .25, .62, accent='#ff7a1e')
        if 60 <= f < 65:
            p = cam.project(BPOS + [0, -1, 7])[0]; c = Cel(ss=2); SP.burst_lines(c, p[0], p[1], 330, 1700, 130, 3 + f // 2, '#ffffff', 11, .85); SP.burst_lines(c, p[0], p[1], 300, 900, 40, 13 + f, '#17131a', 8, .9); img = over(img, c.layer())
            if f == 60: img = kit.impact(img, 'red', .24, .60, accent='#d8142a')
        # ---- whip blur
        if f > 0 and f not in (15, 60, 61, 62, 63, 64):
            sht = .30 if 44 <= f <= 46 else .13; c0, c1 = self.camera_still(f - sht), self.camera_still(f + sht); q = cam.eye + cam.R[2] * 60.; a, b = c0.project(q)[0], c1.project(q)[0]; dv = b - a; m = np.hypot(*dv)
            if m > 30 and np.isfinite(m): dv = dv * min(1, (420 if 44 <= f <= 46 else 16 if 47 <= f <= 57 else 50) / m); img = self.mblur(img, dv[0], dv[1])
        if 58 <= f < 60: img = kit.zoomblur(img, 960, 560, .10, 11)
        if f >= 60: img = don.put(img, f - 60, 'slam', flash=False, x=1370, y=850, size=.50, life=12, out=4, rot=-.08)
        return img.astype(f32)

    def camera_still(self, f): return self.camera_ns(f)

    def camera_ns(self, f):
        s = kit.shake; kit.shake = lambda *a, **k: np.zeros(2)
        try: return self.camera(f)
        finally: kit.shake = s

    def mblur(self, img, dx, dy):
        m = math.hypot(dx, dy); n = int(m) | 1; k = np.zeros((int(abs(dy)) | 1, int(abs(dx)) | 1), np.float32); h, w = k.shape; sx, sy = (1 if dx * dy >= 0 else -1), 1; cv2.line(k, (0 if sx > 0 else w - 1, 0), (w - 1 if sx > 0 else 0, h - 1), 1., 1, cv2.LINE_AA); k /= k.sum(); return cv2.filter2D(img.astype(f32), -1, k, borderType=cv2.BORDER_REFLECT_101)

    def zover(self, img, lz, zmap):
        l, z = lz; vis = cv2.GaussianBlur((z < zmap + .5).astype(f32) + (z > 1e8), (0, 0), .8).clip(0, 1); return img * (1 - l[..., 3:4] * vis[..., None]) + l[..., :3] * vis[..., None]

    def small_splash(self, img, cam, x, y, age, s, zmap):
        Pc, Dp, Dv, rd = sea.splash_ring(cam, x, y, age, r0=.6 + s * .4, v=2.6, vz=5 + s * 2, n=18, seed=int(x * 7) & 255, drop=.12); return over(img, sea.drops(cam, Dp[::2], Dv[::2], rd[::2], PAL, zmask=zmap))

    def guns(self, img, cam, f):
        c = Cel(ss=2); fl = []
        for k, (f0, p) in enumerate(((1, (-50, -76, 5)), (3, (-54, -76, 5)), (8, (50, -90, 5)), (11, (100, -70, 5)), (21, (-60, -78, 5)), (26, (40, -92, 5)), (31, (6, -183, 5)), (50, (100, 148, 5)), (53, (-70, 148, 5)), (62, (146, 96, 5)), (72, (92, 150, 5)))):
            a = f - f0
            if not 0 <= a < 22: continue
            q, z = cam.project(np.array(p, float))
            if z < 5 or not (-200 < q[0] < W + 200 and -200 < q[1] < H + 200): continue
            s = cam.f / z
            if a < 2: fl.append((q, s))
            FB.puffs(c, FB.billow(a / 24. + .03, q[0], q[1], 7. * s, k + 3, 7, .6, 1.2, 1.5, drift=20), dict(lit='#f6f4ee', mid='#cfd4dc', dark='#9aa8ba', ink='#4a5468'), (.6, -.8), 1.2, k + 3)
        img = over(img, c.layer())
        for q, s in fl: img = kit.flare(img, q[0], q[1], 16 * s, '#ffd27a', 1.0)
        return img

    def marines(self, img, cam, t, o, n, seed, zmap):
        r = np.random.default_rng(seed)
        for i in range(n):
            v = np.array([r.normal(0, 9), r.normal(0, 7) - 3, r.uniform(8, 24)]); p = np.asarray(o, float) + r.normal(0, 2.4, 3) * [2, 1, .3] + v * t + np.array([0, 0, -4.9 * t * t]); a, ax = r.uniform(0, 6) + t * r.uniform(3, 9), r.integers(0, 3); q, z = cam.project(p)
            if p[2] < 0 or z < 1 or not (-100 < q[0] < W + 100 and -100 < q[1] < H + 100): continue
            R = Rz(r.uniform(0, 6)) @ (Rx, Ry, Rz)[ax](a); l = self.mar.pose(('thrown', 'float', 'twist_open')[i % 3]).hang(p, R).draw(cam, int(t * 24), light=SUN, expr='shock', rim=dict(col='#fff6d8', w=1.4), hatch=False, pad=30); xi, yi = int(np.clip(q[0], 0, W - 1)), int(np.clip(q[1], 0, H - 1))
            if zmap[yi, xi] > z - 1: img = over_roi(img, *l)
        return img

    # ---------------------------------------------------------------- Sanji
    def do_sanji(self, img, cam, f, t):
        p = self.sanji_state(f); pose = seq(f, [(0, TUCK, 'io'), (7, AXU, 'hold'), (12, AXU, 'in'), (15, AXD, 'hold'), (23, AXD, 'hold')]); spin = float(np.interp(f, [0, 7, 9, 23], [-5.2, -.25, 0, 0])); R = Rz(-1.72) @ Rx(-spin) if f < 9 else Rz(-1.72) @ Rx(.34 * (f < 12) - .12 * (f >= 13))
        A = self.sanji.pose(pose).hang(p + [0, 0, .9 if f < 15 else 1.0], R); S = A.build(); heel = S.an['R']; hist = []
        for k in range(1, 9):
            g = f - k * .8
            if g < -6: break
            q = self.sanji_state(max(0, g)); hist.append(q + (heel - p) * (1 if f >= 12 else .0) + (np.array([0, 0, 0]) if f >= 12 else (heel - p)) + np.array([.15 * math.sin(k * 1.3 + f), .15 * math.cos(k * 1.7), (.16 * k if f < 12 else .55 * k)]))
        pts = np.vstack([heel] + hist); pr, z = cam.project(pts); z = np.maximum(z, .3); hz = float(z[0]); wpx = float(np.clip(.34 * cam.f / hz, 8, 190)); c = Cel(ss=2)
        pth = spl(pr, n=6); ln = np.hypot(*np.diff(pth, axis=0).T).sum()
        if ln > 30: FF.flame_stream(c, pth, wpx * (1.15 if f > 11 else .85), t, 3, speed=700, lw=1.8, smooth=.35, core=True)
        if f >= 15:                                                  # fire left along the split
            zt = np.linspace(min(36, p[2] + 14), p[2] + 1, 12); q = cam.project(np.c_[0 * zt, .35 + 0 * zt, zt])[0]; FF.flame_stream(c, q[::-1], float(np.clip(.5 * cam.f / hz, 8, 120)), t, 9, speed=500, lw=1.6, smooth=.2)
        lay = c.layer(); img = over(img, lay); img = screen(img, cv2.GaussianBlur(lay[..., :3], (0, 0), 26) * .55)
        rim = dict(col='#ffb25a', dir=(.1, .2, -1.), glow=.8, w=max(3., 1.8 * cam.f / hz * .012)); l = A.draw(cam, f, light=SUN, expr='glare' if f < 11 else 'grit', rim=rim, wind=(0, 0, 2.), hair=1.6, pad=200)
        if f in (13, 14): sm = cv2.GaussianBlur(l[0], (0, 0), sigmaX=.1, sigmaY=22); img = over_roi(img, sm * .6, l[1], l[2])
        img = over_roi(img, *l)
        hp, _ = cam.project(heel); c = Cel(ss=2); FF.flames(c, np.array([hp + [-wpx * .55, wpx * .3], hp + [0, wpx * .5], hp + [wpx * .55, wpx * .3]]), wpx * 1.7, t * 1.3, 5, lw=1.6, up=-math.pi / 2); l2 = c.layer(); img = over(img, l2, .92); img = screen(img, cv2.GaussianBlur(l2[..., :3], (0, 0), 18) * .7); img = FF.put(img, FF.embers(f, 60, 3, origin=(float(hp[0]), float(hp[1])), spread=300, wind=(0, -420), life=1.0)); return img

    # ---------------------------------------------------------------- Zoro
    def do_zoro(self, img, cam, f, t, zmap):
        d, lift = self.zoro_state(f); Rm = fallR(theta(f)); bd_ = float(np.interp(f, [15, 18, 22, 30], [.25, .9, 1.1, .5])); face = lambda z: (min(max(z, 0.), 27) / 27.6) ** 1.6 * bd_; pos = PV + Rm @ np.array([.0, face(d) - lift, d]); B = np.array([[1, 0, 0], [0, 0, -1.], [0, 1, 0]]); u = (f - 20) / 7.0
        run = P(sprint(u), **RUNARMS); pose = run if f < 35 else seq(f, [(35, run, 'io'), (37, CROSS, 'hold'), (38, CROSS, 'snap'), (40, SLASH, 'hold'), (46, add_pose(SLASH, pel=(10, 0, 0)), 'hold')])
        A = self.zoro.pose(pose); A.xf = ('ground', pos, Rm @ B); S_ = A.build(); gs, ft = {}, {}
        for s in 'LR':                                               # clearance of each sole over the split face
            V_ = [Rm.T @ (S_.an[s] + (S_.Rfoot[s][:, 1] * b_ + S_.Rfoot[s][:, 2] * d_) * S_.H - PV) for b_, d_ in ((-.034, -.046), (.04, -.048), (.108, -.046))]; gs[s] = min(face(q[2]) - q[1] for q in V_); ft[s] = V_[1]
        gap = min(gs.values()); dn = float(np.interp(f, [37, 38], [1, 0])) * (gap - min(.40 * max(gap, 0.), .08)); pos = pos + Rm @ np.array([0, dn, 0]); A.xf = ('ground', pos, Rm @ B); beam = 35 <= f < 46; kb = smooth(33, 36, f)
        zc = cam.project(pos + Rm @ np.array([0, -1., 0])); zp = zc[0]
        # foot dust and shadow on the raw wood
        c = Cel(ss=2)
        if f < 38:
            a = np.linspace(0, 6.28, 28); v = View(cam); ell = lambda x, z, rx, rz: v(np.array([PV + Rm @ np.array([x + rx * math.cos(q), face(z + rz * math.sin(q)) - .012, z + rz * math.sin(q)]) for q in a])); lo_ = min('LR', key=lambda s: gs[s]); mx, mz = ft[lo_][0] * .65 + (ft['L'][0] + ft['R'][0]) * .175, ft[lo_][2] * .65 + (ft['L'][2] + ft['R'][2]) * .175
            c.fill(ell(mx - .05, mz - .06, .36, .34), '#5a3820', .45)
            for s in 'LR':
                g_ = gs[s] - dn
                if g_ < .16: c.fill(ell(ft[s][0], ft[s][2], .15, .19), '#341c0c', .62 * (1 - max(g_, 0.) / .16))
            if int(u * 2) != int((u - 1 / 7.) * 2) or True:
                rr = np.random.default_rng(f // 3)
                for k in range(7): q, z = cam.project(pos + Rm @ np.array([rr.normal(0, .42), -rr.uniform(0, .34), -rr.uniform(.2, 1.5)])); s2 = cam.f / z * rr.uniform(.03, .10); kit.shape(c, q + np.c_[np.cos(a) * s2 * 1.6, np.sin(a) * s2 * .6], '#f0cf94', 1.)
        img = over(img, c.layer())
        tgt = cam.project(pos + Rm @ np.array([.1, -1.15, .9]))[0]
        if beam and f < 40: img = self.beam(img, f, np.array([1990., -120.]), tgt, 300, 46)
        rim = dict(col='#ffe9a0' if kb > .2 else '#fff3cf', dir=(.8, .3, .5), glow=.4 + .6 * kb); l = A.draw(cam, f, light=SUN if kb < .5 else (.3, .9, .5), expr='grit' if f < 38 else 'shout', rim=rim, wind=tuple(Rm @ np.array([0, -.3, -2.2])), sword_mouth=True, hair=.4, pad=260, tone=None)
        if f == 39: sm = cv2.GaussianBlur(l[0], (0, 0), sigmaX=26, sigmaY=4); img = over_roi(img, sm * .55, l[1], l[2])
        img = over_roi(img, *l)
        if 38 <= f <= 42:                                            # the cut: white crescents
            c = Cel(ss=2); k = (f - 38) / 4.; s = cam.f / zc[1]; a = np.linspace(-2.75 + .4 * k, .1 + .5 * k, 40)
            for rr_, wd, col in ((1.75, .34, '#ffffff'), (1.45, .12, '#d8f2e0'), (2.15, .10, '#ffffff')):
                th = wd * s * np.sin(np.linspace(0, math.pi, 40)) ** .8 * (1 - .7 * k); o = zp + np.c_[np.cos(a) * rr_ * s * 1.25, np.sin(a) * rr_ * s * .62 + .1 * s]; n_ = np.c_[np.cos(a), np.sin(a)]; c.fill(np.vstack([o + n_ * th[:, None], (o - n_ * th[:, None] * .15)[::-1]]), col, .95 - .4 * k)
            lay = c.layer(); img = over(img, lay); img = screen(img, cv2.GaussianBlur(lay[..., :3], (0, 0), 16) * .6)
        if 40 <= f < 46:
            for dx, dy in ((-1500, 330), (760, 560)): img = self.beam(img, f, tgt + np.array([0, 0]), tgt + np.array([dx, dy]) * 1.7, 40, 96 - 8 * (f - 40), False)
            img = self.beam(img, f, np.array([1990., -120.]), tgt, 300 * (1 - .15 * (f - 40)), 46)
            img = kit.flare(img, tgt[0], tgt[1], 260 - 30 * (f - 40), '#ffe9a0', .55 - .1 * (f - 40), streak=.5, ang=.2)
        if f == 40: img = kit.impact(img, 'red', .45, .86, accent='#f2a92a')
        if f >= 43:
            q, z = cam.project(np.array([30., -60., 0]))
            if z > 5: c = Cel(ss=2); FB.puffs(c, FB.billow((f - 42) / 24., q[0], q[1], 26 * cam.f / z, 5, 11, 1., 1.2, 1.4), FB.HOTSMOKE, (.6, -.8), 1.8, 5); img = over(img, c.layer()); img = kit.flare(img, q[0], q[1], 260, '#ffb04a', .6)
        return img

    def beam(self, img, f, a, b, wa, wb, taper=True):
        a, b = np.asarray(a, float), np.asarray(b, float); d = unit(np.r_[b - a, 0])[:2]; n = np.array([-d[1], d[0]]); r = np.random.default_rng(f * 3 + int(wa)); m = 18; u = np.linspace(0, 1, m); c = Cel(ss=2)
        for k, col in ((1.35, '#ff9a1e'), (1.0, '#ffc93a'), (.66, '#fff08a'), (.34, '#fffdf0')):
            wd = (wa + (wb - wa) * u) * k * .5; up = a + (b - a) * u[:, None] + n * (wd * (1 + r.normal(0, .07, m)))[:, None]; dn = a + (b - a) * u[:, None] - n * (wd * (1 + r.normal(0, .07, m)))[:, None]; c.fill(np.vstack([spl(up, n=4), spl(dn, n=4)[::-1]]), col, .55 if k > 1.2 else 1.)
        l = c.layer(); img = over(img, l); return screen(img, cv2.GaussianBlur(l[..., :3], (0, 0), 34) * .75)

    # ---------------------------------------------------------------- Luffy
    def do_luffy(self, img, cam, f, t):
        from chars import luffy as LU
        from op.head3 import Head
        from chars import act as AC
        p = self.luffy_pos(f); bite = 1 if f == 48 else 0; pose = seq(f, [(45, POSES['leap'], 'io'), (46, BITE, 'hold'), (49, add_pose(BITE, hed=(-6, 0, 0), cst=(-4, 0, 0)), 'io'), (51, add_pose(BITE, hed=(-10, 0, 2), cst=(-9, 0, -2), cR=(2, 0)), 'io'), (53, BLOW, 'io'), (55, COCK, 'hold'), (57, add_pose(COCK, root=(0, -.02, -.02), cst=(0, 0, -6)), 'snap'), (59, THROW, 'hold'), (72, THROW, 'hold')]); C, r = self.fist_state(f); big = f >= 53; trem = .02 * math.sin(f * 2.9) * (50 < f < 57); cu = 46 <= f < 53
        A = self.luffy.pose(pose).hang(p + np.array([0, 0, -.02 * bite]), Rz(.0 + trem) @ Rx(-.10 if f < 56 else -.35)); z_l = cam.project(p)[1]; k_ = smooth(57, 58.6, f); sh0 = p + np.array([.2, 0, .5])
        def arm(pen, s, ua, fa, S): return s == 'R' and (big or cu)
        def armlayer(S):
            sh = S.sh['R']; d = C - sh; L = np.linalg.norm(d); fd = unit(unit(C - sh) * (1 - k_) + unit(BPOS + [0, 0, 6] - C + np.array([0, 4., 0])) * k_); e = C - fd * r * 1.05; u = np.linspace(0, 1, 60); c1 = sh + d / L * L * .35 + np.array([.5, 0, .5]) * r * .12; c2 = e - fd * r * (1.2 if f < 58 else .4); P3 = np.array([(1 - q) ** 3 * sh + 3 * (1 - q) ** 2 * q * c1 + 3 * (1 - q) * q * q * c2 + q ** 3 * e for q in u]); rad = .085 + (r * (.32 if f < 58 else .36) - .085) * np.clip((u - .06) / .94, 0, 1) ** (1.3 if f < 58 else 1.1)
            c = Cel(ss=2); pen = Pen(c, View(cam), View(cam).l2(SUN), 4.6, 3); rubber(pen, P3, rad, SKIN, ripple=.12 if f < 57 else .06, k=3.2, phase=f * .35, rings=10, w=4.6, seed=3, zmin=.8); n = 36; rubber(pen, P3[n:], rad[n:] * 1.015, M('#43355a', '#171020', '#0a070d', '#8f78b8'), rings=4, w=4.6, seed=4, veins=False, hi=.6, zmin=.8); lay = c.layer(); lay = kit.rim(lay, -6, -6, '#8b5cff', .6, soft=1.); return kit.rim(lay, 6, 6, '#ffe9b8', .85, soft=1.)
        if cu:                                                        # close: thumb in the teeth, then the swell runs from the hand up the arm
            fp = f // 2 * 2 if f < 48 else f; l = A.draw(cam, f, light=SUN, expr='grit', rim=dict(col='#fff0c4', w=7, dir=SUN, glow=.5), wind=(.5, -1.2, .6), hair=1., arm_fn=arm, pad=500, nohead=True); img = over_roi(img, *l); S = A.S; v = View(cam)
            e = unit(cam.eye - S.head); xv = unit(v.right - e * (v.right @ e)); yv = np.cross(e, xv); hx, hy, hz = -S.Rh[:, 0], S.Rh[:, 2], S.Rh[:, 1]; R = np.array([[q @ a_ for a_ in (hx, hy, hz)] for q in (xv, yv, e)]); yaw, pit = math.atan2(R[0, 2], R[2, 2]), math.asin(np.clip(R[1, 2], -1, 1)); rol = -math.atan2(R[1, 0], R[1, 1]); c0 = v(S.head + S.Rh[:, 1] * .02); us = S.u * v.s(S.head) * 1.22
            yaw, pit, rol = float(np.clip(yaw, -.75, .75)), float(np.clip(pit, -.35, .35)), float(np.clip(rol, -.5, .5)); hl = LU.head(float(c0[0]), float(c0[1]) - 3 * bite, us, yaw, pit, rol, expr='glare', light='day', f=f, hat=True, body=False, wind=1.4, hk=dict(bounce=.35 * math.sin(f * 1.7))); img = over(img, hl)
            Hd = Head().pose(float(c0[0]), float(c0[1]), us, yaw, pit, rol); bl = smooth(49, 51, f); c = Cel(ss=2); pen = Pen(c, v, v.l2(SUN), 5., 5); th = np.linspace(0, 2 * math.pi, 26, endpoint=False)
            if bl > 0:                                                # cheeks like a bellows
                pu = bl * (1 + .10 * math.sin(f * 2.4))
                for sg in (-1, 1):
                    q = Hd.P(np.c_[sg * .46 + np.cos(th) * .30 * pu, -.46 + np.sin(th) * .27 * pu]); m_ = q.mean(0); q = m_ + (q - m_) * (1 + .10 * pu) + np.array([sg * us * .10 * pu, 0]); c.fill(q, SKIN['shade']); c.clip(q); c.fill(m_ + (q - m_) * .82 + pen.L * us * .07, '#f7b793'); c.fill(m_ + pen.L * us * .12 + np.c_[np.cos(th), np.sin(th)] * us * .07, '#ffe6cf'); c.unclip(); kit.outline(c, q, 4., '#17131a', 3 + sg, press=.5); [c.line(np.vstack([m_ + [(-.12 + .12 * j) * us, .03 * us], m_ + [(-.08 + .12 * j) * us, .12 * us]]), 2.4, '#9a4a3c', taper=(.2, .5)) for j in range(3)]
            sh, el, wr = S.sh['R'], S.el['R'], S.wr['R']; mth = v(wr); swl = float(np.interp(f, [48, 49, 50, 51, 52, 53], [1.15, .96, .74, .50, .26, .10])); amp = float(np.interp(f, [48, 49, 50, 51, 52, 53], [0, .6, 1.25, 1.6, 1.8, 1.9])); u = np.linspace(0, 1, 54); P3 = np.where(u[:, None] < .52, sh + (el - sh) * (u / .52)[:, None], el + (wr - el) * ((u - .52) / .48)[:, None]); ker = np.hanning(13)[1:-1]; ker /= ker.sum(); Pp = np.vstack([np.repeat(P3[:1], 5, 0), P3, np.repeat(P3[-1:], 5, 0)]); P3 = np.stack([np.convolve(Pp[:, i], ker, 'valid') for i in range(3)], -1)
            base = np.interp(u, [0, .12, .35, .50, .62, .80, 1], [.060, .066, .060, .048, .056, .046, .036]); tail = .055 * amp * (u > swl); rad = base + .088 * amp * np.exp(-((u - swl) / .125) ** 2) + tail * .55 * np.clip((1 - u) / .18, 0, 1)
            n0 = 20; rubber(pen, P3[n0 - 3:], rad[n0 - 3:], SKIN, ripple=.0, rings=int(5 + 4 * amp), w=5., seed=6, zmin=.2); sl = rad[:n0] * 1.16; rubber(pen, P3[:n0], sl, MAT['red'], rings=3, veins=False, w=5., seed=7, zmin=.2)
            for j, uu in enumerate((.52, .60, .68, .78, .88)):           # cords stand out as it passes
                i = int(uu * 53); k2 = max(0., 1 - abs(uu - swl) / .35) * min(1, amp)
                if k2 > .2: a0 = v(P3[i]); a1 = v(P3[min(53, i + 5)]); n_ = np.array([-(a1 - a0)[1], (a1 - a0)[0]]); n_ /= np.hypot(*n_) + 1e-6; rr = rad[i] * v.s(P3[i]); c.line(spl([a0 + n_ * rr * .35, (a0 + a1) / 2 + n_ * rr * (.15 - .1 * j % 2), a1 - n_ * rr * .1], n=8), 3.4, '#9a5646', taper=(.3, .5), seed=j)
            hs = A.S.H * (.100 + .016 * min(1, amp)) * (1 + .05 * math.sin(f * 2)); AC.small_hand(pen, wr, S.Rhand['R'], hs, 'R', 'fist', SKIN, 5.); th0 = S.Rhand['R'][:, 0]; tip = v(S.head + S.Rh @ np.array([0, .145, -.075])); b0 = v(wr + S.Rhand['R'][:, 1] * hs * .45 + th0 * hs * .30); d_ = tip - b0; n_ = np.array([-d_[1], d_[0]]) / (np.hypot(*d_) + 1e-6) * us * .10; c.fill(spl([b0 + n_ * 1.2, b0 * .4 + tip * .6 + n_, tip + n_ * .7, tip - n_ * .7, b0 * .4 + tip * .6 - n_, b0 - n_ * 1.2], True, n=6), SKIN['base']); kit.outline(c, spl([b0 + n_ * 1.2, b0 * .4 + tip * .6 + n_, tip + n_ * .7, tip - n_ * .7, b0 * .4 + tip * .6 - n_, b0 - n_ * 1.2], n=6), 4.4, '#17131a', 5, closed=False)
            c.fill(spl([tip + [-us * .16, -us * .05], tip + [0, -us * .09], tip + [us * .16, -us * .05], tip + [us * .14, us * .05], tip + [0, us * .02], tip + [-us * .14, us * .05]], True, n=6), '#fbf8ee'); c.line(spl([tip + [-us * .18, -us * .06], tip + [0, -us * .10], tip + [us * .18, -us * .06]], n=8), 4., '#17131a', taper=(.2, .2)); [c.line(np.array([tip + [x * us, -us * .08], tip + [x * us, us * .03]]), 2., '#17131a', taper=(0, 0)) for x in (-.08, 0., .08)]
            lay = c.layer(); lay = kit.rim(lay, -View(cam).l2(SUN)[0] * 7, -View(cam).l2(SUN)[1] * 7, '#fff0c4', .9, soft=.8); img = over(img, lay); c = Cel(ss=2)
            if f in (48, 49): SP.burst_lines(c, tip[0], tip[1], us * .45, us * 1.5, 15, f, '#ffffff', 8, .95); SP.burst_lines(c, tip[0], tip[1], us * .4, us * .9, 9, f + 5, '#17131a', 4, .9)
            if f >= 49:                                               # air drawn in toward the mouth, stretch marks beside the swell
                for k in range(12): a_ = k * .55 + f * .4; r1 = us * (2.6 + .5 * (k % 3)); q0 = tip + np.array([math.cos(a_), math.sin(a_) * .8]) * r1; q1 = tip + np.array([math.cos(a_ + .35), math.sin(a_ + .35) * .8]) * r1 * .62; c.line(spl([q0, (q0 + q1) / 2 + (q1 - q0)[::-1] * [.12, -.12], q1], n=8), 4.5, '#ffffff', taper=(.5, .1), alpha=.75)
                i = int(np.clip(swl, 0, 1) * 53); q = v(P3[i]); rr = rad[i] * v.s(P3[i]); [c.line(q + np.c_[np.cos(np.linspace(a_, a_ + .8, 10)), np.sin(np.linspace(a_, a_ + .8, 10))] * rr * (1.18 + .10 * k), 4.2, '#ffffff', taper=(.4, .4), alpha=.9) for k, a_ in enumerate((-2.4, .6, 2.6))]
            return over(img, c.layer())
        lF = self.bigfist(cam, C, r, unit(unit(C - sh0) * (1 - k_) + unit(BPOS + [0, 0, 6] - C + np.array([0, 4., 0])) * k_), w=7.) if big else None
        ex = 'grit' if f < 47 else 'pout' if f < 56 else 'shout'
        l = A.draw(cam, f, light=SUN, expr=ex, rim=dict(col='#fff0c4', w=6, dir=SUN, glow=.5), wind=(.4, -1.8 if f >= 57 else -.8, .5), hair=1., arm_fn=arm, pad=700, puff=(1.15 if 50 <= f < 56 else 0.))
        first_f = lF is not None and lF[3] > z_l
        if lF is not None and first_f: img = over_roi(img, lF[0], lF[1], lF[2])
        if f < 58 and big:
            if first_f: img = over(img, armlayer(A.S)); img = over_roi(img, lF[0], lF[1], lF[2]) if lF is not None else img; img = over_roi(img, *l)
            else:
                img = over_roi(img, *l); img = over(img, armlayer(A.S)); img = over_roi(img, lF[0], lF[1], lF[2]) if lF is not None else img
        elif big and first_f: img = over(img, armlayer(A.S)); img = over_roi(img, *l)
        else:
            if z_l > .3: img = over_roi(img, *l)
            if big: img = over(img, armlayer(A.S))
            if lF is not None: img = over_roi(img, lF[0], lF[1], lF[2])
        q, z = cam.project(C); rp = r * cam.f / max(float(z), .5)
        if 53 <= f < 58:                                             # it keeps swelling: breathing contour strokes, weight dragging the air
            c = Cel(ss=2)
            for k in range(10): a_ = k * .63 + f * .25; c.line(q + np.c_[np.cos(a_ + np.linspace(0, .45, 10)), np.sin(a_ + np.linspace(0, .45, 10))] * (rp * (1.30 + .06 * (k % 3) + .03 * (f % 2))), 5., '#ffffff', taper=(.4, .4), alpha=.85)
            img = over(img, c.layer())
        if 57.5 <= f < 66 and z > 1:                                  # the air squeezed flat in front of it
            from op import bolt as BO
            c0 = cam.project(C)[0]; c1 = cam.project(C + unit(BPOS + [0, 0, 6] - C + np.array([0, 4., 0])) * r * 1.5)[0] if f < 60 else cam.project(C + np.array([.2, 1., -.35]) * r)[0]; dv = c1 - c0; an = math.atan2(dv[1], dv[0]); c = Cel(ss=2)
            for k, (kr, w_) in enumerate(((1.22, 15), (1.48, 10), (1.80, 7), (2.2, 4))):
                sp_ = 1.05 - .12 * k; a_ = np.linspace(an - sp_, an + sp_, 40); wd = w_ * np.sin(np.linspace(0, math.pi, 40)) ** .7; o = q + np.c_[np.cos(a_), np.sin(a_)] * rp * kr * (1 + .02 * math.sin(f * 3 + k)); n_ = np.c_[np.cos(a_), np.sin(a_)]; c.fill(np.vstack([o + n_ * wd[:, None], (o - n_ * wd[:, None] * .35)[::-1]]), '#ffffff', .85 - .14 * k); c.fill(np.vstack([o - n_ * wd[:, None] * .35, (o - n_ * wd[:, None] * .75)[::-1]]), '#8b5cff', .45 - .08 * k)
            img = over(img, c.layer()); img = BO.bend(img, float(q[0] + math.cos(an) * rp * .4), float(q[1] + math.sin(an) * rp * .4), float(rp * 1.6), wd=max(30., rp * .25), amt=22.)
        if 58 <= f < 61: c = Cel(ss=1); SP.burst_lines(c, q[0], q[1], 260, 1800, 90, f, '#ffffff', 9, .8); img = over(img, c.layer())
        return img

    # ---------------------------------------------------------------- trio
    def do_trio(self, img, cam, f, t):
        from op.fig_pose import idle; R0, t0 = self.sp[0], self.sp[1]; gust = (1.6 * math.exp(-max(0, f - 66) / 14.) + 1.25 * math.exp(-((f - 83.5) / 4.5) ** 2) + .55) * (1 + .22 * math.sin(f * 1.9) + .12 * math.sin(f * 3.1)); W_ = np.array([-.55 * gust, -2.0 * gust, .55 * gust]); LP = P('land'); it = []; shd = []; vw = View(cam); loc = lambda p: R0.T @ (np.asarray(p, float) - t0); wld = lambda x, y, z: t0 + R0 @ np.array([x, y, z]); seat = lambda o: wld(*o)      # SPOT = tops of the mane petals in ship coordinates
        def drop(A, f0, off, hd, stand, frm, air, ex, **k):
            if f < f0 - 6: return
            q = seat(off); g0 = loc(q); fl_ = 1. if f >= f0 else max(.35, 1 - (f0 - f) / 6.)
            if f < f0: u = 1 - (f0 - f) / 6.; q = q + np.array(frm) * (1 - u) + np.array([0, 0, 9. * (1 - u ** 2)]); po = air
            else: L0 = P(LP, **{g: stand.get(g, 'fist') for g in ('gL', 'gR')}); po = seq(f, [(f0, add_pose(L0, root=(0, 0, .07)), 'out'), (f0 + 1, add_pose(L0, root=(0, 0, -.055)), 'hold'), (f0 + 2, add_pose(L0, root=(0, 0, -.075)), 'hold'), (f0 + 4, add_pose(L0, root=(0, 0, -.04)), 'io'), (f0 + 6, add_pose(L0, root=(0, 0, .03)), 'out'), (f0 + 8, add_pose(stand, root=(0, 0, .016), cst=(-4, 0, 0)), 'io'), (f0 + 11, add_pose(stand, root=(0, 0, -.022), pel=(4, 0, 0)), 'io'), (f0 + 14, stand, 'hold')])
            if f >= f0 + 8: k_ = -5.5 * (gust - .6); fb = unit(np.array([-math.sin(hd), math.cos(hd)])); fw_ = fb[1]; po = idle(add_pose(po, pel=(k_ * fw_ * 1.2, 0, 0), wst=(k_ * fw_ * .5, 0, 0), root=(0, 0, -.018 * gust), hed=(-k_ * fw_ * .6, 0, 0)), f, 1.4, f0)
            S_ = A.pose(po).place(q, hd).build()
            if f >= f0:
                for s in 'LR': l_ = loc(S_.an[s] + S_.Rfoot[s][:, 1] * .03 * S_.H); shd.append((l_[0], l_[1], .16, .14, g0[2] + .02, '#3c1a04', .66))
            shd.append((g0[0] - .04 * fl_, g0[1] - .12 * fl_, .26 * fl_, .42 * fl_, g0[2], '#7a3a06', .46 * fl_))
            it.append((cam.project(q)[1], f >= f0, lambda: A.pose(po).place(q, hd).draw(cam, f, light=SUN, expr=ex, rim=dict(col='#fff3cf', w=6, dir=SUN, glow=.5), wind=tuple(W_ + (np.array([0, 0, 2.2]) if f < f0 else 0)), hair=.55 + .55 * gust, pad=260, **k)))
        drop(self.zoro, 70, SPOT['Z'], math.pi / 2 + .55, STZ, (-5, 1, 3), P('leap', gL='grip', gR='grip'), 'glare', sword_mouth=False)
        drop(self.sanji, 75, SPOT['S'], -math.pi / 2 - .50, STS, (5, 1, 3), AXD, 'glare')
        drop(self.luffy, 80, SPOT['L'], math.pi, STL, (0, 0, 4), P('float'), 'shout' if f < 80 else 'grin')
        c = Cel(ss=2)                                                   # contact and cast shadows lie on the head
        for x, y, rx, ry, z_, col, al in sorted(shd, key=lambda q: q[6]):
            P3 = np.array([wld(x + rx * math.cos(a), y + ry * math.sin(a), z_) for a in np.linspace(0, 6.28, 26)]); c.fill(vw(P3), col, al)
        sl = c.layer(); img = over(img, sl * (self.dlz < 1e8)[..., None])
        for zq, dn, fn in sorted(it, key=lambda q: -q[0]):
            l_ = fn(); L_ = kit.rim(l_[0], 5, 4, '#dff4ff', .75, soft=.8)
            if dn: ox, oy = int(l_[1]), int(l_[2]); Z_ = np.full(L_.shape[:2], 1e9); sb_ = self.dlz[oy:oy + L_.shape[0], ox:ox + L_.shape[1]]; Z_[:sb_.shape[0], :sb_.shape[1]] = sb_; L_ = L_ * cv2.GaussianBlur((Z_ > zq - .45).astype(f32), (0, 0), .8)[..., None]
            img = over_roi(img, L_, l_[1], l_[2])
        c = Cel(ss=2)
        for f0, off in ((70, SPOT['Z']), (75, SPOT['S']), (80, SPOT['L'])):     # landing dust
            a = f - f0
            if 0 <= a < 9:
                q, z = cam.project(seat(off) + [0, 0, .05]); s = cam.f / z
                for k in range(10): th = k / 10 * 6.28; rr = (.25 + .10 * a) * s; FB.puffs(c, [(q[0] + math.cos(th) * rr * 1.5, q[1] + math.sin(th) * rr * .3 - a * 1.5, max(2.6, s * .10 * (1 - a / 9.)), k + f0, a * .3, a / 12.)], dict(lit='#fff6e0', mid='#e6cfa6', dark='#b89468', ink='#5a4030'), (.6, -.8), 1.4, k)
        return over(img, c.layer())
