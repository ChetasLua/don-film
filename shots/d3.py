"""d3: the little sheep ship burns in the snow.  One slow push at sea level: black water, a boat of silhouettes, a lowered torch,
a small raincoat on the yard that waves once, and melted snow that runs from the sheep's eye."""
import numpy as np, cv2, math, os
from op import W, H, ships, sea
from op.cam import Camera3D
from op.sea import Sea, NIGHT
from op.paint import *
from op.ink import Cel, spl, resample
from op.kit import outline, rim, para, shape, flare
from op.comp import blur, ease, ease_out, lerp_keys, screen, over, radial
from op.fx_fire import flames, embers, PAL
from op.fx_air import snow
from op.fx_core import put, noise, h1
A_ = np.asarray
HEAD = math.pi + .70
FP = dict(rim='#9c2410', body='#d9541a', amber='#ee8a2c', hot='#ffd27a', core='#fff0c4', ink='#2e0c0a', dark='#1c0506', maroon='#4a0f0c')
HOR = 775.


def capsule(c, P, w0, w1, col):
    P = spl(P, n=6); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; u = np.linspace(0, 1, len(P))[:, None]; w = (w0 + (w1 - w0) * u) / 2; c.fill(np.vstack([P + N * w, (P - N * w)[::-1]]), col)


def crew(c, X, f, col='#0d0c12'):
    """eight backs in a rowing boat, metres (x right, y up) -> Cel.  twos."""
    t = (f // 2) * 2 / 24.; hitch = lambda p: .018 * max(0, math.sin(t * 9 + p)) ** 3; fl = lambda p, a=1.: a * (.5 * math.sin(t * 5 + p) + .3 * math.sin(t * 8.3 + p * 2)); e = np.linspace(0, 6.283, 26)
    ell = lambda x, y, rx, ry, rot=0.: X(np.c_[x + np.cos(e) * rx * math.cos(rot) - np.sin(e) * ry * math.sin(rot), y + np.cos(e) * rx * math.sin(rot) + np.sin(e) * ry * math.cos(rot)])
    P = lambda q, cor=(), n=6: c.fill(spl(X(q), True, corners=cor, n=n), col); cap = lambda q, a, b: capsule(c, X(q), a * X.k, b * X.k, col)
    # Franky: huge, pompadour sunk, forearm across his eyes
    y = hitch(1.); P([(-2.62, .2), (-2.86, 1.0), (-2.78, 1.62), (-2.45, 1.86), (-2.05, 1.84), (-1.72, 1.60), (-1.66, 1.0), (-1.92, .2)]); c.fill(ell(-2.16, 2.06 + y, .19, .23), col); P([(-2.36, 2.10 + y), (-2.40, 2.36 + y), (-2.10, 2.48 + y), (-1.72, 2.42 + y), (-1.66, 2.24 + y), (-1.94, 2.20 + y)]); cap([(-1.78, 1.52), (-1.60, 1.12), (-1.64, .55)], .34, .46)
    # Robin: straight, long hair moving, one hand at the heart
    w = fl(0, .05); P([(-1.36, .2), (-1.42, .95), (-1.36, 1.34), (-1.14, 1.42), (-.92, 1.34), (-.86, .95), (-.92, .2)]); c.fill(ell(-1.12, 1.66, .125, .155), col); P([(-1.28, 1.72), (-1.12, 1.86), (-.96, 1.70), (-.98, 1.40), (-1.04 - w, 1.05), (-1.18 - w * 2, .86), (-1.34 - w * 2, .92), (-1.36 - w, 1.2)])
    # Usopp: folded over the gunwale, nose, curls, shoulders hitching
    y = hitch(0.) * 2; P([(-.86, .1), (-.94, .62 + y), (-.76, .98 + y), (-.46, 1.02 + y), (-.28, .74), (-.26, .1)]); c.fill(ell(-.46, 1.12 + y, .13, .14), col); cap([(-.36, 1.12 + y), (-.16, 1.05 + y)], .06, .035)
    for k in range(9): a = k / 9 * 6.283; c.fill(ell(-.52 + math.cos(a) * .13, 1.19 + y + math.sin(a) * .12, .085, .08), col)
    cap([(-.36, .88 + y), (-.18, .80 + y), (-.30, 1.02 + y)], .14, .10)
    # Nami: kneeling, both hands to her face
    y = hitch(2.2) * 1.6; P([(-.16, .1), (-.22, .62), (-.14, .92 + y), (.06, 1.0 + y), (.24, .90 + y), (.28, .6), (.24, .1)]); c.fill(ell(.08, 1.16 + y, .115, .135), col); w = fl(1.5, .03); P([(-.06, 1.22 + y), (.06, 1.36 + y), (.21, 1.24 + y), (.20, 1.08 + y), (.06 - w, .96 + y), (-.10 - w, 1.02 + y)]); cap([(.22, .86 + y), (.36, .80 + y), (.20, 1.10 + y)], .10, .07)
    # Sanji: standing, head down, hand in pocket
    P([(.50, .2), (.46, 1.0), (.50, 1.42), (.68, 1.50), (.90, 1.42), (.94, 1.0), (.88, .2)]); c.fill(ell(.76, 1.68, .12, .15, -.2), col); P([(.60, 1.70), (.66, 1.86), (.82, 1.86), (.92, 1.72), (.93, 1.58), (.84, 1.60), (.74, 1.56), (.62, 1.56)]); cap([(.52, 1.34), (.40, 1.0), (.52, .74)], .15, .12); w = fl(.4, .04); P([(.50, .7), (.40 - w, .2), (.60, .2)])
    # Zoro: arms crossed, hilts
    P([(1.06, .2), (.98, 1.0), (1.02, 1.46), (1.26, 1.56), (1.52, 1.46), (1.54, 1.0), (1.46, .2)]); c.fill(ell(1.30, 1.74, .125, .15), col)
    P([(1.16, 1.74), (1.15, 1.88), (1.22, 1.86), (1.24, 1.96), (1.31, 1.89), (1.38, 1.95), (1.40, 1.86), (1.47, 1.86), (1.44, 1.72)], tuple(range(9))); cap([(1.02, 1.36), (.94, 1.12), (1.30, 1.10)], .17, .14)
    for k in range(3): cap([(1.12, .62 - .07 * k), (.80, .94 - .05 * k), (.66, 1.08 - .04 * k)], .045, .06)
    # Chopper: on the gunwale, hat and antlers, shaking
    y = hitch(3.4) * 2; jx = .012 * math.sin(t * 22); P([(1.60 + jx, .30), (1.58 + jx, .56), (1.70 + jx, .68 + y), (1.90 + jx, .66 + y), (2.00 + jx, .5), (1.96 + jx, .30)]); c.fill(ell(1.80 + jx, .82 + y, .19, .17), col); P([(1.56 + jx, .90 + y), (1.62 + jx, .98 + y), (1.64 + jx, 1.20 + y), (1.80 + jx, 1.26 + y), (1.96 + jx, 1.20 + y), (1.98 + jx, .98 + y), (2.06 + jx, .90 + y), (1.8 + jx, .86 + y)], (0, 6))
    for s in (-1, 1): cap([(1.80 + jx + s * .20, 1.0 + y), (1.80 + jx + s * .36, 1.06 + y), (1.80 + jx + s * .40, 1.22 + y)], .06, .04); cap([(1.80 + jx + s * .34, 1.05 + y), (1.80 + jx + s * .46, 1.10 + y)], .05, .03)
    # Luffy at the bow: hat, the torch arm hanging
    bow = .05 * ease((f - 14) / 14.); b = fl(2., .012); P([(2.22, .2), (2.16, .98), (2.20, 1.40), (2.42, 1.48), (2.64, 1.40), (2.66, .98), (2.60, .2)]); hx, hy = 2.46 + bow * 1.2, 1.66 - bow * .9; c.fill(ell(hx, hy, .13, .155), col)
    P([(hx - .20, hy + .02), (hx - .17, hy + .16), (hx, hy + .20), (hx + .17, hy + .13), (hx + .16, hy - .02), (hx + .05, hy - .10), (hx - .10, hy - .14)]); r = -.10 - bow * 2; R = lambda q: [(hx + (x * math.cos(r) - y * math.sin(r)), hy + (x * math.sin(r) + y * math.cos(r))) for x, y in q]
    P(R([(-.40, .10 + b), (-.16, .15), (-.15, .30), (0, .35), (.15, .30), (.16, .15), (.40, .09 - b), (.18, .05), (-.18, .06)]), (0, 6)); cap([(2.58, 1.34), (2.78, 1.02), (2.92 - .04, .70)], .17, .12); w = fl(.9, .05); P([(2.22, .95), (2.06 - w, .55), (2.10 - w, .25), (2.3, .3)])
    return (2.92, .72)


class Shot:
    id = 'd3'; n = 60; finish = dict(bloom=.34, vig=.30)
    EVENTS = [(0, 'fire', dict(note='ship fire bed, soft, far; snow hush', size=.7)), (6, 'wood', dict(note='a timber settles')), (18, 'bell', dict(note='one tiny hammer tap: the raincoat figure lifts his arm', size=.3)), (24, 'wind', dict(note='snow gust')), (40, 'drip', dict(note='melt gathers at the eye', size=.3)), (46, 'drip', dict(note='it runs', size=.3)), (54, 'drip', dict(note='it falls on the water', size=.4))]

    def setup(self):
        self.S = Sea(17, hs=.16, length=7, chop=.5, wind=(1, .3)); p = 'cache/d3_sky2.npy'
        if not os.path.exists(p):
            x, y = grid(H, W); n = fbm(H, W, (420, 170, 70, 28), (1, .6, .4, .25), 6, aspect=2.2); g = ramp(np.clip(y / HOR + .05 * n, 0, 1), [(0, '#0d0e15'), (.50, '#1d1c26'), (.80, '#3a3138'), (1, '#6b4e48')])
            lo = np.exp(-((x - 1280) / 620) ** 2 - ((y - 620) / 420) ** 2); sm = smooth(-.3, 1.1, n + lo * 1.0); g = g + (sm * lo)[..., None] * rgb('#7a3a24') * .55; g = g * (1 + .10 * smooth(0, 1.4, n)[..., None])
            np.save(p, repaint(g, (40, 20, 9), seed=2, density=.9, jit=(.006, .02, .02)).astype(np.float16))
        self.sky = np.load(p).astype(f32)

    def cam(self, f):
        e = ease(f / 62.) * 1.0; return Camera3D((-.15 + .55 * e, -17.3 + 2.5 * e, 1.75 + .12 * e), (-.15 + .55 * e, 60, 1.75 + .12 * e), 1300, cx=960, cy=HOR)

    def ship(self, t):
        m = ships.merry(t * .5, wind=.25, sails=False, flag=False)
        ships.yard(m, 1.2, 13.3, 5.2)
        # scorched remains of the sail + snow the fire has not reached
        for (y0, y1, z, s) in ((-5.0, -3.6, 8.6, 1), (2.2, 3.2, 10.4, 2), (4.2, 5.0, 9.4, 3)): m.sail((1.75, y0, 13.15), (1.75, y1, 13.15), (1.75, y0 + .5, z), (1.75, (y0 + y1) / 2, z - .8), .5, t, '#6b5a50', f'rag{s}', s, n=(8, 12), wind=.7, ripple=.05)
        for q, r in (((11.55, 0, 8.33), (.78, .62, .26)), ((11.1, .0, 8.28), (.60, .70, .22)), ((12.75, 0, 7.66), (.34, .42, .10))): m.ball(q, r, '#e9edf4', 'snowcap', n=(8, 14), grp='snowcap', lw=1.6, sh=(.56, .58, .72))
        u = np.linspace(.78, .995, 16)
        for sd in (1, -1): m.tube(m.hull.pt(u, 1., sd) + [0, 0, .24], .17, '#e4e9f2', 'snowrail', n=6, grp='snowrail', lw=1.3, sh=(.56, .58, .72))
        m.tube([(6.1, -2.8, 3.95), (6.1, 2.8, 3.95)], .16, '#e4e9f2', 'snowb', n=6, lw=1.)
        return m

    def firebases(self, cam, ps, flip=1):
        """(screen polyline, height px, seed) rows, far to near, from the ship's own geometry"""
        f3 = lambda P: cam.project(ships.at(ps, A_(P, float)))[0]; px = lambda p: float(cam.scale(ships.at(ps, A_(p, float)))); u = np.linspace(0, 1, 9)[:, None]; L = lambda a, b: A_(a, float) + (A_(b, float) - A_(a, float)) * u
        back = [(f3(L((-10.4, -2.6, 5.2), (6.0, -3.0, 3.4))), 8.6 * px((-2, -3, 4)), 3), (f3(L((-10.2, -.4, 4.4), (6.4, 0, 2.4))), 10.2 * px((-2, 0, 3)), 11)]
        mid = [(f3(L((1.2, 0, 5.0), (1.2, 0, 11.4))), 3.6 * px((1.2, 0, 9)), 17, .55), (f3(L((1.45, -5.0, 13.2), (1.45, 1.6, 13.35))), 2.2 * px((1.4, 0, 13.7)), 23, 1.1), (f3(L((-5.6, 0, 7), (-5.6, 0, 12))), 2.8 * px((-5.6, 0, 9)), 29, .5)]
        front = [(f3(L((-9.0, 2.9, 4.8), (-2.0, 3.5, 3.2))), 3.8 * px((-4, 3, 4)), 31), (f3(L((-8.6, 1.6, 6.5), (-4.9, 1.6, 6.5))), 3.6 * px((-6, 2, 6)), 37), (f3(L((2.2, 3.3, 2.9), (5.6, 3.0, 3.0))), 2.6 * px((4, 3, 3)), 41)]
        return back, mid, front

    def frame(self, f):
        t = f / 24.; cam = self.cam(f); gx = 1.05; z0 = self.S.height(5, 4, t) * .4; ps = ships.pose(4.75, 6.9, z0 - .28 + .04 * math.sin(t * 1.3), HEAD, .012 * math.sin(t * 1.5), .020 * math.sin(t * 1.1 + 1)); m = self.ship(t)
        x, y = grid(); fk = 1 + .07 * noise(t * 4.2, 3., 0, 5) + .04 * noise(t * 11, 1., 0, 6)
        img = self.sky.copy(); img = img + (np.exp(-((x - 1240) / 640) ** 2 - ((y - 560) / 430) ** 2) * .20 * fk)[..., None] * rgb('#ff7a36')
        # smoke leaning away
        img = self.smoke(img, f)
        Lf = A_([.55, .80, .35])
        back, mid, front = self.firebases(cam, ps); FB = Cel(); FM = Cel(); FF = Cel(); tt = (f // 2) / 12.
        for b, h, s in back: flames(FB, b, h * (.96 + .04 * fk), tt * .8, s, pal=FP, hot=True, rows=2, spread=.10, b=.17, hz=1.0, sway=.17, lean=-.10, lw=1.2)
        for b, h, s, dn in mid: flames(FM, b, h, tt * .85, s, pal=FP, hot=True, rows=1, hz=1.2, sway=.18, lean=-.10, lw=1.0, dens=dn)
        for b, h, s in front: flames(FF, b, h, tt * .9, s, pal=FP, hot=False, rows=1, hz=1.3, sway=.18, lean=-.08, lw=1.0, dens=.8)
        fb, fm, ff = FB.layer(), FM.layer(), FF.layer()
        d = ships.draw([(m, ps)], cam, Lf, lw=1.05, shadow=False, amb=(.47, .43, .52), rim=('#ffb768', 7, 1.0), bounce='#8a3a24', edge_col='#0c0910', shade_amt=.9, noink=())
        lay = para(d.layer, (1500, 300), (420, 820), [(0, '#ffb072'), (.35, '#d89a84'), (.7, '#b4a8bc'), (1, '#9a9cb8')], 'mul', .5)
        kl = self.klabauter(cam, ps, f)
        # ---------------- above-water picture pieces
        mw = np.zeros((H, W, 4), f32)
        pm = ships.mirror(ps); dm = ships.draw([(m, pm)], cam, Lf * [1, 1, -1], ss=1, ink=None, shadow=False, amb=(.34, .30, .38), decals=False); mw = dm.layer
        fl = np.maximum(fb, 0); allf = fb * (1 - fm[..., 3:4]) + fm
        # mirrored flames: each row flips about its own waterline
        wl = cam.project(A_([[ps[1][0], ps[1][1], 0]]))[0][0, 1]
        mf = cv2.warpAffine(allf, np.float32([[1, 0, 0], [0, -1, 2 * wl]]), (W, H)); mw = mf * (1 - mw[..., 3:4]) + mw; mff = cv2.warpAffine(ff, np.float32([[1, 0, 0], [0, -1, 2 * wl]]), (W, H)); mw = mw * (1 - mff[..., 3:4]) + mff
        skyr = cv2.warpAffine(img, np.float32([[1, 0, 0], [0, -1, 2 * HOR]]), (W, H), borderMode=cv2.BORDER_REFLECT); sk4 = np.dstack([skyr * .62, np.ones((H, W), f32)]); mw = sk4 * (1 - mw[..., 3:4]) + mw
        bp = ships.pose(-5.0, -2.9, self.S.height(-5, -3, t) + .06, 0.05, .012 * math.sin(t * 1.9), .03 * math.sin(t * 1.4)); bx = self.boat(cam, bp, f)
        bw = cam.project(A_([[-5.0, -2.9, 0]]))[0][0, 1]; mb = cv2.warpAffine(bx[0], np.float32([[1, 0, 0], [0, -1, 2 * bw - 4]]), (W, H)); mw = mw * (1 - mb[..., 3:4]) + mb
        col = ships.collar(m, ps, cam, 7, .42)
        w = self.S.render(cam, 3 + t, NIGHT, sun=(.5, .8, .2), mirror=mw, mirror_amt=1.05, mirror_k=.55, foam_amt=0, foam=None, tone=.9, haze=.6, crest=0)
        img = w.over(img)
        # ---------------- composite
        img = put(img, fb); img = img + blur(fb[..., :3], 34) * .55
        img = put(img, kl[0]); img = img + kl[1]
        img = sea.under(img, lay, d.z, w); img = put(img, fm); img = put(img, ff); img = img + blur(fm[..., :3] + ff[..., :3], 26) * .38 + blur(fb[..., :3], 90) * .30
        img = self.tear(img, cam, ps, f)
        # fire path on the water: drawn flecks
        img = img + sea.glitter(cam, t, (4.4, 24., .2), 300, 5, '#ff9a4a', 4, 34, .17, .85, 1.3, frac=.40, asp=.13) * .55 * fk
        bl = bx[0]; img = img * (1 - bl[..., 3:4]) + bl[..., :3]; img = img + bx[1]
        # snow and embers cross
        em = embers(f, 78, 4, box=(760, 120, 1800, 900), wind=(-34., -118.), pal=FP, near=.08, size=.9, life=2.6); img = put(img, em)
        img = put(img, snow(f, n=(420, 190, 46, 9), wind=-46., fall=74., seed=8, col='#e6e8f0', shade='#8f93a8', size=.95, op=.92))
        g = lum(np.clip(img, 0, 1))[..., None]; img = g + (img - g) * .84                # the film is losing colour
        return img

    def smoke(self, img, f):
        t = f / 24.; n = fbm(H // 2, W // 2, (150, 60, 24), (1, .6, .35), 9, aspect=1.2); x, y = grid(H // 2, W // 2)
        mx, my = x + 34 * t + 10 * np.sin(y * .02 + t), y + 46 * t
        a = cv2.remap(n, np.mod(mx, W // 2 - 1), np.mod(my, H // 2 - 1), cv2.INTER_LINEAR); ax = 640 - (430 - y) * .42; m = np.exp(-((x - ax) / (95 + (430 - y) * .42)) ** 2) * smooth(470, 250, y); a = smooth(-.25, 1.3, a + m * 1.5 - .35) * m
        a = cv2.resize(a, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]; un = smooth(60, 720, grid()[1])[..., None]
        return img * (1 - a * .70) + a * .70 * (rgb('#15131a') * (1 - un) + rgb('#5a2a1c') * un)

    def klabauter(self, cam, ps, f):
        """raincoat, hammer, a wave: f14 lift, f18-32 three sways, f38 down.  -> (layer, light)"""
        p3 = ships.at(ps, (1.5, -3.0, 13.45)); p, z = cam.project(p3); s = cam.f / z * 1.62; ff = (f // 2) * 2; k = lerp_keys(ff, [(10, 0), (13, -.10), (18, 1.0), (21, .93), (34, .97), (42, 0.)]); wv = math.sin((ff - 18) / 14. * 2 * math.pi * 1.5) * (18 <= ff <= 34) * .42
        c = Cel(400, 400, int(p[0]) - 200, int(p[1]) - 300); X = lambda q: A_(q, float) * [s, -s] + p; fl = .02 * math.sin(f * .9); col = '#f7dca6'; sh = '#d99a5a'
        body = spl(X([(-.19, 0), (-.24 - fl, .10), (-.20, .40), (-.15, .62), (0, .68), (.15, .62), (.20, .40), (.25 + fl, .10), (.20, 0), (.07, .03), (0, .0), (-.07, .03)]), True, n=6); hood = spl(X([(-.20, .66), (-.22, .82), (-.13, .98), (.02, 1.04), (.15, .97), (.21, .82), (.18, .66), (0, .60)]), True, n=6)
        sh0 = A_([.15, .58]); a = -1.15 + k * 2.15 + wv; e = sh0 + A_([math.cos(a - .35), math.sin(a - .35)]) * .17; h_ = e + A_([math.cos(a + .15 + wv * .6), math.sin(a + .15 + wv * .6)]) * .17
        capsule(c, X([(-.15, .58), (-.25, .42), (-.21, .28)]), .12 * s, .09 * s, col); capsule(c, X([(-.21, .28), (-.30, .02)]), .03 * s, .03 * s, '#6a4224'); c.fill(X([(-.39, .05), (-.25, .09), (-.22, -.03), (-.36, -.07)]), '#8a8088')
        c.fill(body, col); c.shade(X([(.02, 0), (.04, .7), (.4, .7), (.4, 0)]), sh, body, .8); capsule(c, X([sh0, e, h_]), .13 * s, .09 * s, col); a_ = np.linspace(0, 6.283, 14); c.fill(np.c_[X(h_)[0] + np.cos(a_) * .055 * s, X(h_)[1] + np.sin(a_) * .055 * s], '#fff0d0')
        c.fill(hood, col); fc = spl(X([(-.12, .72), (-.13, .84), (0, .92), (.13, .84), (.12, .72), (0, .66)]), True, n=5); c.fill(fc, '#7a3f2a')
        for q in (-.045, .05): c.fill(np.c_[X((q, .80))[0] + np.cos(a_) * .016 * s, X((q, .80))[1] + np.sin(a_) * .024 * s], '#fff3d6')
        c.line(X([(-.05, .725), (0, .705), (.055, .728)]), 1.2, '#fff3d6')
        for i, q in enumerate((body, hood)): outline(c, q, 1.5, '#5a2414', seed=i)
        l = c.layer() * .88; g = blur(l[..., 3], 16)[..., None] * rgb('#ffd08a') * .50; return l, g

    def boat(self, cam, bp, f):
        d = ships.draw([(ships.dinghy(f / 24., L=6.4, oars=False), bp)], cam, (.7, .6, .3), lw=1.0, shadow=False, dark=.90, sil='#0b0a10', ink=None, rim=('#d9783a', 2.2, .8)); o, z = cam.project(bp[1] + A_([0, .1, .18])); k = cam.f / z * 1.02
        X = lambda q: A_(q, float) * [k, -k] + o; X.k = k; c = Cel(); r = math.sin(f / 24. * 1.4) * .03; hand = crew(c, X, f)
        l = rim(c.layer(), -4.2, 1.8, '#ffa050', .95, .7); l = rim(l, 1.6, 1.2, '#4c5270', .35, .8)
        hp = X([hand])[0]; sw = .05 * math.sin(f * .16); tip = hp + A_([.66 + sw, .10]) * k; tc = Cel(); capsule(tc, [hp - A_([.06, .12]) * k, tip], .050 * k, .066 * k, '#120d0e')
        flames(tc, np.c_[np.linspace(tip[0] - .07 * k, tip[0] + .09 * k, 5), np.full(5, tip[1] + 2)], .62 * k, (f // 2) / 12., 5, pal=FP, hot=True, n=3, rows=1, hz=2.0, sway=.2, lean=-.15, lw=.9)
        tl = tc.layer(); b = d.layer; lay = l * (1 - b[..., 3:4]) + b; lay = lay * (1 - tl[..., 3:4]) + tl
        x, y = grid(); g = np.exp(-((x - tip[0]) ** 2 + (y - tip[1] + 8) ** 2) / (2 * (46 + 3 * math.sin(f * 1.7)) ** 2))[..., None] * rgb('#ff9a44') * .42; return lay, g

    def tear(self, img, cam, ps, f):
        """f34-42 a bead swells under the snow cap at the eye, f42-53 it runs down the cheek, f54 it is gone"""
        c = Cel(); L = lambda p: cam.project(ships.at(ps, A_(p, float)))[0]
        # the path over the face (local metres on the port side of the head)
        h = A_([11.3, 0, 7.0]); s = 1.05; path = [h + A_(q) * s for q in ((1.00, .70, .72), (1.10, .72, .52), (1.16, .74, .30), (1.18, .74, .06), (1.12, .72, -.22), (1.02, .66, -.48))]; P = spl(A_([L(q) for q in path]), n=10); P = resample(P, 1.5); n = len(P)
        if f >= 34:
            u = lerp_keys(f, [(34, 0.0), (41, .24), (43, .26), (49, .74), (53, 1.0)], lambda t: t * t * (3 - 2 * t)); i = max(3, int(u * (n - 1))); a0 = 0
            if i > 6: c.line(P[a0:i], 4.6, '#cfd8ea', taper=(.1, .05), press=.15, seed=3, alpha=.85); c.line(P[a0:i] - [1.1, 0], 1.3, '#fff4e0', taper=(.2, .1), seed=5, alpha=.9)
            if f < 56:
                r = lerp_keys(f, [(34, 1.6), (42, 7.2), (48, 7.8), (54, 8.4)]); p = P[min(i, n - 1)].copy()
                if f > 53: p[1] += (f - 53) ** 2 * 5.
                a = np.linspace(0, 6.283, 22); dr = np.c_[p[0] + np.cos(a) * r * .84, p[1] + np.sin(a) * r * (1 + .42 * (np.sin(a) < 0))]; c.fill(np.c_[dr[:, 0], dr[:, 1]] + (dr - p) * .22, '#1a1520'); c.fill(dr, '#bcc8e0'); c.fill(p + (dr - p) * [.62, .62] + [r * .16, r * .14], '#f0a860'); c.fill(p + (dr - p) * .30 + [-r * .28, -r * .34], '#ffffff')
        l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if f >= 34: img = img + blur(l[..., :3], 5) * .25
        return img
