"""f6 (60): two ink drawings; a long shape falls through the ring it tore in the clouds; dawn; he falls laughing; the hat fills the sky."""
import numpy as np, cv2, math
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp
from op.ink import Cel, spl, resample
from op import bolt, kit, sumi
from op.clouds import Sky
from op.comp import blur, radial, screen, shaft
from op.cam import Camera3D
from op.fx_core import h1, noise, put, FX, lin01
from chars import kaido as K
from shots import dragonkit as DK

SW, SH, OV = 4000, 780, 420


def strip():
    """clouds whose tops face the hole, lit from it; wraps round"""
    g = DK.vgrad(SH, SW + OV, [(0, '#fff0cf'), (.16, '#ffd9a4'), (.34, '#f0a892'), (.6, '#7d5f96'), (1., '#2c2a58')]); S = Sky(SH, SW + OV, g, light=(.0, -.92, .28), seed=4); r = np.random.default_rng(12)
    rows = ((300, 13, 130, ('#fff6dc', '#ffd29a', '#e39a98', '#a470a0'), '#e8a69a'), (420, 12, 190, ('#ffedc6', '#f8b68a', '#c07d98', '#765a94'), '#b07a98'), (560, 11, 250, ('#ffdcae', '#d08c8c', '#85609a', '#4c4280'), '#6d548e'), (700, 10, 280, ('#e0a094', '#8a6496', '#54477e', '#302c5a'), '#473f74'), (860, 9, 300, ('#8a6a9a', '#54487e', '#322e5c', '#1c1c40'), '#2c2a56'))
    for yb, m, hh, T, mist in rows:
        for i in range(m):
            x = SW * (i + .5 + r.uniform(-.3, .3)) / m; w_ = SW / m * r.uniform(1.3, 1.9); h_ = hh * r.uniform(.7, 1.3); sd = int(r.integers(9999))
            for o in (-SW, 0, SW):
                if -w_ < x + o < SW + OV + w_: S.bank(x + o, yb, w_, h_, T, mist, seed=sd, n=6, ss=1, pool=.35)
    p = S.paint(2, amt=.6); a = np.linspace(1, 0, OV)[None, :, None] ** 1.; p[:, :OV] = p[:, :OV] * (1 - a) + p[:, SW:SW + OV] * a; return p[:, :SW]


class Shot:
    id = 'f6'; n = 60; finish = dict(bloom=.34, vig=.10)
    EVENTS = [(0, 'impact', dict(size=2., note='ink impact, black on white')), (3, 'impact', dict(size=1., note='second drawing, inverted')), (6, 'ring', dict(size=1.2, note='the ring tears open', dur=14)), (8, 'fall', dict(size=1., note='the dragon falls', dur=22)), (14, 'bells', dict(size=.6, note='dawn pours in')), (28, 'rumble', dict(size=1., note='it hits the sea far below')), (30, 'laugh', dict(size=1., note='he falls laughing')), (34, 'whoosh', dict(size=.6, note='hair goes black, nape to fringe', dur=10)), (45, 'whoosh', dict(size=.8, note='the hat, spinning, grows', dur=14)), (58, 'straw', dict(size=.5))]

    def setup(self):
        self.st = DK.full('f6strip', strip); self.n1 = fbm(H, W, (120, 46, 16), (1, .6, .4), 3); self.lan = DK.Lanterns(170, 9, box=((-120, 120), (40, 260), (-60, 60)), size=2.4); self.camL = Camera3D((0, -10, -20), (0, 100, 14), 1200)
        c = Cel(); DK.skull_dome(c, 330, 985, 168, ('#4a3a62', '#33284e', '#221b3c', '#120e26'), '#0c0a1c', '#ffcf7a', lw=2.4, seed=4); DK.roofs(c, 120, 900, 990, 44, 6, ('#2c2348', '#1e1838', '#120e26'), n=34, lw=1.6); DK.roofs(c, 1100, 1980, 1010, 60, 8, ('#2c2348', '#1e1838', '#120e26'), n=26, lw=1.8); self.isl = c.layer()
        self.ink = None

    # ------------------------------------------------------------ the two drawings
    def impacts(self, f):
        from shots import f5 as F5
        IL = dict(pal=K.palette(over={m: ('#ffffff', '#f7f2e8', '#0d0b0e', '#0d0b0e') for m in K.BASE} | {K.M_GUM: ('#0d0b0e',) * 4, K.M_MANE: ('#f4efe4', '#0d0b0e', '#0d0b0e', '#0d0b0e'), K.M_TONGUE: ('#ffffff', '#f4efe4', '#0d0b0e', '#0d0b0e')}), L=(-.50, -.80, .42), rim='#ffffff', rd=(-1., -.4), rw=10., line='#0b090c')
        fk = DK.full('f5fistI', lambda: F5.fist_layer((.75, .25, .50), dict(skin='#f6f1e6', shade='#0d0b0e', line='#0b090c', nail='#ffffff', nail_shade='#0d0b0e', light='#ffffff'), ink=True)); two = f >= 3; cx, cy = (1010, 520) if not two else (1120, 490); b, d = sumi.burst(H, W, 1000, 510, 5, r0=200.)
        hk = dict(yaw=.72, pitch=.10, roll=-.16, jaw=.85) if not two else dict(yaw=.60, pitch=.26, roll=-.44, jaw=1.05)
        o = K.head(cx + (410 if not two else 400), cy - (40 if not two else 70), 158 if not two else 166, light=IL, f=3, full=True, wind=(.95, .3, .30), lw=(8.5, 3.8, 2.2), persp=7., term=-.30, hatch=.5, shadow=False, rim=0., **hk)
        c = Cel(); r = np.random.default_rng(4 + two); pts = np.array([[-200, 900], [-200, 560]])
        s = .98 if not two else 1.10; M = np.float32([[s * 1.06, 0, cx - 60 - F5.KN[0] * s * 1.06], [0, s * .95, cy + 20 - F5.KN[1] * s * .95]]); fl = cv2.warpAffine(fk, M, (W, H)); wr = M[:, :2] @ F5.WR + M[:, 2]; ax = F5.AX; n = np.array([ax[1], -ax[0]]); n = n if n[1] < 0 else -n; w0 = 190 * s
        arm = np.array([wr + n * w0 - ax * 40, wr + n * w0 * 1.2 + ax * 420, wr + n * w0 * 1.3 + ax * 1300, wr - n * w0 * 1.3 + ax * 1300, wr - n * w0 * 1.25 + ax * 420, wr - n * w0 - ax * 40]); c.fill(arm, '#f6f1e6'); c.fill(np.array([wr + n * w0 * .05, wr + n * w0 * .2 + ax * 420, wr + n * w0 * .2 + ax * 1300, arm[3], arm[4], arm[5]]), '#0d0b0e'); c.line(arm[:3], 13, '#0b090c', taper=(.02, .02)); c.line(arm[3:], 15, '#0b090c', taper=(.02, .02))
        for i in range(26): p = wr + n * w0 * .12 + ax * (30 + i * 46); c.line(spl([p, p + n * w0 * (.2 + .3 * r.random()) - ax * 20], n=4), 5, '#0b090c', taper=(.05, .8), seed=i)
        al = c.layer(); cf = F5.cuff(wr[0] + ax[0] * 20, wr[1] + ax[1] * 20, 232 * s, ax, 0, ('#ffffff', '#ffffff', '#cfcac0'), '#0b090c', 8., 2)
        t = sumi.tooth(H, W); rays = b * smooth(340, 760, d); sp = sumi.splash(cx - 10, cy + 10, 480, seed=4 + two, n=170, big=20, fingers=14, core=0.); ink = np.zeros((H, W), f32); x0, y0 = max(0, sp.ox), max(0, sp.oy); h, w = sp.d.shape; h, w = min(h, H - y0), min(w, W - x0); ink[y0:y0 + h, x0:x0 + w] = sp.d[:h, :w]
        g = np.ones((H, W, 3), f32) * rgb('#f7f2e6'); k = np.clip(rays, 0, 1)[..., None]; g = g * (1 - k) + rgb('#0b090c') * k
        for l in (o['layer'], al, fl, cf): g = g * (1 - l[..., 3:4]) + l[..., :3]
        fa = np.clip(o['layer'][..., 3] + fl[..., 3] + al[..., 3] + cf[..., 3], 0, 1)
        c = Cel()
        for i in range(34):                                             # the white star of the blow
            th = (i + r.random()) / 34. * 6.283; ln = r.uniform(90, 300) * (1.6 if i % 5 == 0 else 1); q = np.array([math.cos(th), math.sin(th)]); n2 = np.array([-q[1], q[0]]) * r.uniform(5, 15); p = np.array([cx - 14., cy + 16.]); c.fill(np.array([p - n2, p + q * ln, p + n2]), '#ffffff')
        for i in range(16 if not two else 26):                         # teeth and splinters
            a = r.uniform(-2.6, 1.2); dd = r.uniform(160, 620); p = np.array([cx + math.cos(a) * dd, cy + math.sin(a) * dd * .8]); q = r.uniform(10, 34); th = r.uniform(0, 6.28); u = np.array([math.cos(th), math.sin(th)]); v = np.array([-u[1], u[0]]); P = np.array([p - u * q + v * q * .35, p + u * q * 1.3, p - u * q - v * q * .35]); c.fill(P, '#ffffff'); kit.outline(c, P, 4, '#0b090c', seed=i)
        l = c.layer()
        if not two:
            g = g * (1 - l[..., 3:4]) + l[..., :3]; k = (ink * (1 - .85 * smooth(.3, .9, fa)))[..., None]; g = g * (1 - k) + rgb('#0b090c') * k
        else:                                                           # second drawing: night and blood behind, the figures still burn white
            bg = rgb('#0a0709') * (1 - np.clip(rays, 0, 1)[..., None]) + rgb('#c1141f') * np.clip(rays, 0, 1)[..., None]; halo = cv2.resize(cv2.dilate(fa[::2, ::2], cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))), (W, H)); halo = smooth(.3, .7, halo + .25 * (t - .5))[..., None]; bg = bg * (1 - halo) + rgb('#0a0709') * halo
            m = smooth(.4, .6, fa)[..., None]; fg = g; L = (fg @ np.array([.3, .6, .1], f32))[..., None]; fg = rgb('#fff6ea') * smooth(.35, .6, L) + rgb('#6d0c16') * (1 - smooth(.35, .6, L)); g = bg * (1 - m) + fg * m
            k = (ink * (1 - fa * .9))[..., None]; g = g * (1 - k) + rgb('#fff6ea') * k; g = g * (1 - l[..., 3:4]) + l[..., :3]
        z = 1.02 + .02 * (f % 3); g = cv2.warpAffine(g, np.float32([[z, 0, cx * (1 - z)], [0, z, cy * (1 - z)]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); o_ = kit.shake(f, 14, 2); return cv2.warpAffine(g.astype(f32), np.float32([[1, 0, o_[0]], [0, 1, o_[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def ring(self, cx, cy, r0, asp=.62, spin=0., rmax=1500., tilt=0.):
        """polar cloud ring round a dawn hole"""
        x, y = grid(); c, s = math.cos(tilt), math.sin(tilt); dx, dy = (x - cx) * c + (y - cy) * s, (-(x - cx) * s + (y - cy) * c) / asp; r = np.hypot(dx, dy); a = np.arctan2(dy, dx)
        u = ((a / 6.28318 + .25 + spin) % 1.) * (SW - 1); v = np.where(r > r0, ((r - r0) / (rmax - r0)).clip(0, 1) ** .62, 0) * (SH - 160) + 150 - 150 * np.clip((r0 - r) / (r0 * .35 + 1), 0, 1)
        img = cv2.remap(self.st, u.astype(f32), v.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP); q = np.clip(r / (r0 + 1), 0, 1); core = ramp(q, [(0, '#fffdf0'), (.5, '#fff3cc'), (.85, '#ffe2ae'), (1., '#ffe8bd')]); m = smooth(1.02, .62, q + .05 * self.n1)[..., None]; sky = core + (np.clip(-dy / (r0 + 1), -1, 1) * .08)[..., None] * np.array([-.9, .1, 1.2], f32); img = img * (1 - m) + sky * m
        k = np.clip((r - r0) / rmax, 0, 1)[..., None]; return img * (1 - .25 * k)

    def wide(self, f):
        t = f / 24.; op = 1 - (1 - smooth(5, 24, f)) ** 1.8; r0 = 60 + 300 * op; cx, cy = 1010, 300 + 14 * (1 - op); img = self.ring(cx, cy, r0, .56, .010 * t, 1500 + 200 * (1 - op)); dim = .45 + .55 * op; img = img * np.array([dim * .96, dim * .95, dim ** .8], f32)
        x, y = grid(); hz = 880; sea = np.clip((y - hz) / (H - hz), 0, 1); m = smooth(hz - 40, hz + 6, y + 10 * self.n1)[..., None]; sc = ramp(sea, [(0, '#c98c8e'), (.2, '#7a5a86'), (1., '#22204a')]) * dim; gl = smooth(.35, 1.3, fbm_(self, f)) * np.exp(-((x - 1010) / (170 + 500 * sea)) ** 2) * op; sc = sc + gl[..., None] * rgb('#ffd9a0') * .8; hzl = np.exp(-((y - hz) / 60) ** 2)[..., None] * rgb('#ffb890') * .35 * op; img = img * (1 - m) + sc * m + hzl
        for i, (a, b, w0, w1, am) in enumerate(((1010, 640, 130, 420, .34), (960, 1010, 110, 300, .42), (1060, 1420, 120, 380, .32))): q = shaft(H, W, [(a - w0 * op, cy + 40), (a + w0 * op, cy + 40), (b + w1 * op, 1060), (b - w1 * op, 1060)], 34, i + 2); img = screen(img, (q * am * op * (.9 + .1 * math.sin(t * 5 + i)))[..., None] * rgb('#ffe2ac'))
        img = img + (radial(cx, cy, 760, 470, 2.2) * .45 * op)[..., None] * rgb('#ffe0a6')
        # the long shape
        fall = np.clip((f - 6) / 24., 0, 1.1); zt = 104 - 160 * fall ** 1.7; cam = Camera3D((0, -170, -30), (0, 60, 18), 1150); n = 13; i = np.arange(n); lag = i * 8.4
        lo = np.array([8 * math.sin(t * 5.4 + .6), 5 * math.cos(t * 4.6), 0]); sw = np.exp(-i / 1.6)                       # dead weight: waves run up the body, the neck swings a beat late
        P = np.c_[8 + (6 + 1.15 * i) * 1.15 * np.sin(i * .92 - t * 7.0 + .9) * (1 - np.exp(-i / 1.2)) + .6 * i, 70 + (4 + .9 * i) * np.cos(i * .78 - t * 5.8), zt - 40 + lag * 1.62 + 9 * np.sin(i * 1.05 - t * 6.2)]; P[:, 0] += lo[0] * sw * 1.2; P[:, 1] += lo[1] * sw; D = K.Dragon(P, R=6.4, n=380, around=26, legs=(.14, .46), mane_step=.5, up=(0, -1, .2))
        o = D.render(cam, f, 'dawn', jaw=.42 + .34 * math.sin(t * 5.4 - .8), wind=(-.5 * math.sin(t * 5.4), .0, 2.1), hd=(.55 * math.sin(t * 5.4 - .5), .28 + .30 * math.sin(t * 4.3 + .6), .70 * math.sin(t * 5.4 + .3)), scales=30, ss=1.25, lw=(4.2, 2., 1.2), clouds=False, sil=('#2c1d3e', .50 - .14 * op), rd=(.15, -1.), grip=(.1, .2, .1, .2), pose=((-.55, .75 + .2 * math.sin(t * 6)), (-.4, .6 + .2 * math.cos(t * 5)), (-.6, .8 + .2 * math.cos(t * 6)), (-.3, .6)), whisk=1.5, shadow=False)
        c = Cel(ss=1); pr, zz = cam.project(D.P[::9])
        for k, p in [(k, p) for k, p in enumerate(pr) if p[1] < 880]:    # smoke it leaves
            for j in range(2): a = h1(k, j, 3) * 6.28; rr = (14 + 26 * h1(k, j, 4)) * (1 + .5 * fall); q = p + np.array([math.cos(a) * 18 + 10 * math.sin(t * 2 + k), -26 - 60 * h1(k, j, 5)]); th = np.linspace(0, 6.283, 22); rad = rr * (1 + .12 * np.sin(th * 5 + k)); c.fill(np.c_[q[0] + np.cos(th) * rad, q[1] + np.sin(th) * rad * .86], ('#5a4468', '#7a5c78', '#c08a86')[j + int(h1(k, 6) > .6)], .40)
        l = c.layer(); l = cv2.GaussianBlur(l, (0, 0), 2.); img = img * (1 - l[..., 3:4]) + l[..., :3]; img = _splash(img, o['layer'], f, self.n1, x, y)
        img = put(img, _emb(f)); l = self.isl * np.array([dim, dim, dim, 1], f32); img = img * (1 - l[..., 3:4]) + l[..., :3]; img = screen(img, np.exp(-((y - 985) / 60) ** 2)[..., None] * rgb('#ffb070') * .20)
        ex = 46 * smooth(6, 20, f)
        def pf(P): d = P - np.array([0, 90, 6]); l = np.linalg.norm(d * [1, .4, 1], axis=1)[:, None] + 1e-6; return d / l * ex * np.array([1.3, .4, .8])
        img = self.lan.draw(img, self.camL, t * 1.4, zmin=14, push=pf, dim=1 - .35 * op)
        if 6 <= f < 17: q = (f - 6) / 11.; img = bolt.bend(img, cx, cy, 120 + 1500 * q ** .8, 160, 40 * (1 - q), .56); c = Cel(); bolt.ring(c, cx, cy, q, 160, 1700, .56, 0, 34, 2, '#fff6e2', '#b98aa8'); rl = c.layer() * .9; img = img * (1 - rl[..., 3:4]) + rl[..., :3]
        if f == 6: img = img * .16 + rgb('#fffaf0') * .90
        o_ = kit.shake(f, 10, 3, 5., 6) + (kit.shake(f, 13, 4, 3., 25) if f >= 25 else 0); return cv2.warpAffine(img.astype(f32), np.float32([[1, 0, o_[0]], [0, 1, o_[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def fallshot(self, f):
        """he falls on his back, laughing, the ring of morning over him"""
        from chars import gear5 as G
        k = f - 30; t = f / 24.; img = self.ring(1180 + 26 * math.sin(k * .22), 190 - 5 * k, 330 - 7 * k, .74, .02 + .012 * k, 1900, .05 + .012 * k); img = img + (radial(1180, 190, 900, 620, 2.) * .35)[..., None] * rgb('#fff0c8'); img = img * ramp(np.clip(grid()[1] / H, 0, 1), [(0, '#ffffff'), (.5, '#e9d6e6'), (1., '#9a86b8')]); c = Cel(); r = np.random.default_rng(5)
        for i in range(46): x = r.uniform(0, W); q = (h1(i, 2) + t * (1.4 + 2.2 * h1(i, 3))) % 1.; y = H * 1.25 * (1 - q) - 60; L = 120 + 380 * h1(i, 4); w = .8 + 3.2 * h1(i, 5) ** 2; c.fill(np.array([[x, y], [x + w, y + L * .5], [x, y + L], [x - w, y + L * .5]]), '#fff6e2', .30 + .4 * h1(i, 6))
        l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]; img = self.lan.draw(img, self.camL, t * 6., zmin=40, dim=.8)
        bl = lambda tier, i: 1 - float(np.clip(((f - 34) / 8. - (dict(back=0., top=.22, front=.46).get(tier, .2) + .16 * h1(i, 71))) / .40, 0, 1))
        hp = np.array([960 + 40 * math.sin(k * .20 + .5), 1010 - 4 * k + 14 * math.sin(k * .5)]); la = np.array([[0, 0], [10, -18], [20, -40], [30, -64], [40, -86], [48, -104]], float) * 1.3; kick = .22 * math.sin(k * 1.05)
        l = G.figure(hp[0], hp[1], 176, 'fall', f=f // 2 * 2, light='sun', lagT=la, head_kw=dict(bl=bl), pel=(0, 2.0), che=(.06, 2.95), hd=(.14, 3.95), roll=.18 + .05 * math.sin(k * 1.25), hA=(-1.9, 4.5 + kick), hB=(1.9, 4.2 - kick), bA=-.25, bB=.25, fA=(-1.35, 2.55 + kick), fB=(1.25, 2.95 - kick), kA=.75, kB=-.75, rims=((.6, -1.), (-.8, -.6)), expr='laugh', amp=1.3, scarf=1.0 - smooth(34, 43, f) * .8)
        an = 64 - 1.6 * k + 5 * math.sin(k * .45); M = cv2.getRotationMatrix2D((hp[0], hp[1] - 470), an, 1.06 + .012 * k); M[0, 2] += 40; M[1, 2] += 30; l = cv2.warpAffine(l, M, (W, H), flags=cv2.INTER_LINEAR); l = kit.para(l, (1200, 200), (800, 900), [(0, '#fff6e6'), (.45, '#f2dfe6'), (1., '#a892bc')]); l = kit.rim(l, -7, 9, '#fff3c4', .85, 1.2)
        gl = cv2.resize(cv2.GaussianBlur(l[::4, ::4, 3], (0, 0), 14), (W, H)); img = screen(img, (gl * .30)[..., None] * rgb('#fff3d0')); img = img * (1 - l[..., 3:4]) + l[..., :3]
        # first sign of the hat: a spinning speck out of the light
        from op import hat as HT
        s = 26 + 8.5 * k ** 1.15; h = HT.hat(1560 - 10 * k, 130 + 9 * k, s * 1.25, yaw=k * .9, pitch=.5 * math.sin(k * .7), roll=k * .55, light='dusk', f=f); img = img * (1 - h[..., 3:4]) + h[..., :3]
        img = self.lan.draw(img, self.camL, t * 6., zmin=12, zmax=40, dim=.9); o_ = kit.shake(f, 2.5, 8); return cv2.warpAffine(img.astype(f32), np.float32([[1, 0, o_[0]], [0, 1, o_[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def hatshot(self, f):
        """looking up: it comes home"""
        from op import hat as HT
        k = f - 45; u = k / 14.; img = self.ring(960, 470, 560 + 16 * k, .92, .02 + .03 * u, 2000, .3 * u); img = img + (radial(960, 470, 900, 800, 2.) * .40)[..., None] * rgb('#fff0c8'); t = f / 24.
        c = Cel(); r = np.random.default_rng(7)
        for i in range(60): a = h1(i, 1) * 6.283; q = (h1(i, 2) + t * (1. + 1.5 * h1(i, 3))) % 1.; d0 = 80 + 1200 * (1 - q) ** 1.6; L = 30 + 160 * (1 - q); p = np.array([960 + math.cos(a) * d0 * 1.2, 470 + math.sin(a) * d0]); d = np.array([math.cos(a), math.sin(a)]); n = np.array([-d[1], d[0]]) * (.8 + 2.4 * h1(i, 4)); c.fill(np.array([p, p + d * L * .5 + n, p + d * L, p + d * L * .5 - n]), '#fff6e2', .5)
        l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]; img = self.lan.draw(img, self.camL, t * 2., zmin=60, dim=.7)
        w = 250 * 13.5 ** (u ** 1.35); sp = 7.2 * (1 - (1 - u) ** 1.6); wob = (1 - u) ** .7; y = 250 + 290 * u ** .8; x = 1060 - 100 * u + 46 * math.sin(k * .8) * (1 - u)
        h = HT.hat(x, y, w, yaw=sp, pitch=(-.62 * math.cos(u * 3.14159 * 2.2) * (1 - u) ** .5 + .95 * smooth(.45, 1., u)), roll=.34 * math.sin(k * .95 + .6) * wob, light='dusk', f=f, bounce=-.35 * math.sin(k * 1.1) * wob, wear=.7)
        if f < 57: h = h * .0 + _smear_rot(h, x, y, .05 * (1 - u))
        s = kit.shifted(h[..., 3], 0, 0); g = cv2.resize(cv2.GaussianBlur(s[::4, ::4], (0, 0), 20), (W, H)); img = screen(img, (g * .25 * (1 - u))[..., None] * rgb('#ffe2a0')); img = img * (1 - h[..., 3:4]) + h[..., :3]
        img = img * (1 - .80 * smooth(54.5, 59, f) * np.array([.92, .96, 1.], f32)); o_ = kit.shake(f, 2. + 4 * u, 9); return cv2.warpAffine(img.astype(f32), np.float32([[1, 0, o_[0]], [0, 1, o_[1]]]), (W, H), borderMode=cv2.BORDER_REFLECT_101)

    def frame(self, f):
        if f < 6: return self.impacts(f)
        if f < 30: return self.wide(f)
        if f < 45: return self.fallshot(f)
        return self.hatshot(f)


_fc = {}


def fbm_(s, f):
    if 'a' not in _fc: _fc['a'] = fbm(H, W, (16, 6, 2.6), (1, .8, .5), 8, aspect=6.); _fc['b'] = fbm(H, W, (14, 5, 2.4), (1, .8, .5), 9, aspect=6.)
    k = (f // 2 * 2) * 3; return np.roll(_fc['a'], k, 1) * .7 + np.roll(_fc['b'], -k, 1) * .7


def _emb(f):
    from op.fx_fire import embers
    return embers(f, 60, 5, wind=(20., 160.), size=1.2, origin=(1010, 420), spread=420., life=2.)


def _smear_rot(l, x, y, a, n=5):
    if a < .004: return l
    o = np.zeros_like(l)
    for t in np.linspace(-.5, .5, n): M = cv2.getRotationMatrix2D((x, y), math.degrees(a * t), 1.); o += cv2.warpAffine(l, M, (W, H))
    return o / n


def _wall(X, Y, s, g, seed, front):
    """half of the water crown: a flared sheet torn into fingers, lit through from above"""
    r = np.random.default_rng(seed); nf = 9 if front else 13; n = 560; t = np.linspace(0, 1, n); th = math.pi * t if front else 2 * math.pi - math.pi * t; w = r.uniform(.55, 1.5, nf); ed = np.r_[0, np.cumsum(w) / w.sum()]; j = np.clip(np.searchsorted(ed, t, 'right') - 1, 0, nf - 1); v = (t - ed[j]) / (ed[j + 1] - ed[j]); sh = (1 - np.abs(2 * v - 1)).clip(0, 1) ** 1.75; rh = r.uniform(.30, 1., nf); rh[1::2] *= .72; sn = np.sin(math.pi * t)
    hh = s * g * ((.11 + .30 * rh[j] * sh * (1 - .45 * sn)) if front else (.16 + .06 * sn + .60 * rh[j] * sh * (.55 + .45 * sn))); bs = s * .56; cs, si = np.cos(th), np.sin(th); sk = (r.uniform(-.5, .5, nf)[j] + cs * .9) * sh ** 1.5 * s * .10
    top = np.c_[X + cs * (bs + .55 * hh) + sk, Y + si * s * .085 - hh]; base = np.c_[X + cs * bs * 1.04, Y + si * s * .085 + 5 * np.sin(t * 40)]; mid = base + (top - base) * np.c_[(.46 + .10 * np.sin(t * 37 + seed))]; c = Cel(); c.fill(np.vstack([top, base[::-1]]), '#ffffff'); c.fill(np.vstack([mid, base[::-1]]), '#cfaebb' if front else '#e9cdc4', .85)
    for q in range(nf):                                               # flow lines up every finger, beads thrown off the tips
        a = int(n * ed[q]); b = min(n - 1, int(n * ed[q + 1])); m = (a + b) // 2
        for z in (-.22, .16): e = int(np.clip(m + z * (b - a), 0, n - 1)); c.line(spl([base[e], (base[e] + top[e]) / 2 + [cs[e] * s * .012, 0], top[int(m + z * .25 * (b - a))]], n=8), max(1.6, s * .011), '#9d7a9c', taper=(.5, .5), seed=q, alpha=.75)
        d = top[m] - base[m]; d = d / (np.hypot(*d) + 1e-6); a_ = np.linspace(0, 6.283, 20)
        for z in range(2): rr = s * (.020 - .008 * z) * (.6 + .6 * h1(seed, q, z)); p = top[m] + d * s * (.045 + .055 * z) * (1 + h1(seed, q, z + 3)); c.fill(np.c_[p[0] + np.cos(a_) * rr, p[1] + np.sin(a_) * rr * 1.15], '#ffffff')
    c.line(top, max(1.8, s * .0075), '#7d5f92', taper=(.02, .02), seed=seed, alpha=.9); l = c.layer(); H_ = s * g * (.45 if front else .85)
    return kit.para(l, (X, Y - H_), (X, Y + s * .07), [(0, '#fffdf4'), (.45, '#fff0da'), (.8, '#f0c4b0'), (1., '#a98aae')] if front else [(0, '#fffaf0'), (.5, '#ffe9cc'), (1., '#d8a9ac')])


def _splash(img, L, f, n1, x, y):
    """the dragon meets the sea (f24): ring wave, a crown that rises behind and before the body, spray that hangs, mist over the waterline"""
    ov = lambda im, l: im * (1 - l[..., 3:4]) + l[..., :3]; L = L * smooth(940, 862, y + 30 * n1)[..., None]
    if f < 24: return ov(img, L)
    k = min(1., (f - 24) / 5.5); e = 1 - (1 - k) ** 1.8; s = 180 + 340 * e; g = .42 + .58 * (1 - (1 - k) ** 2); X, Y = 985., 908.; a = np.linspace(0, 6.283, 22)
    sh = np.exp(-((x - X) / (s * 1.25)) ** 2 - ((y - Y - 24) / (14 + .06 * s)) ** 2) * smooth(884, 902, y); img = img * (1 - (.55 * sh)[..., None] * np.array([.60, .72, .40], f32)); c = Cel(ss=1); a2 = np.linspace(-.02, 6.27, 200)
    for q, (m, w_) in enumerate(((1., 9.), (.74, 5.5), (1.28, 3.5))):     # ring waves, flattened by the grazing view
        A = (s * 1.05 + 560 * e) * m; B = A * .085; si = np.sin(a2); wb = 1 + .02 * np.sin(a2 * 11 + q * 2 + f * .6); P = np.c_[X + np.cos(a2) * A * wb, Y + 5 + np.where(si > 0, si * B, si * min(B * .3, 17.)) * wb]; c.line(P + [0, w_ * .9], w_ * 1.5, '#45366a', taper=(.02, .02), seed=q, alpha=.6); c.line(P, w_, '#fff2dc', taper=(.02, .02), seed=q, alpha=.92 - .22 * q)
    img = ov(img, c.layer()); bk = _wall(X, Y, s, g, 11, False); gl = cv2.resize(cv2.GaussianBlur(bk[::4, ::4, 3], (0, 0), 12), (W, H)); img = screen(img, (gl * .35)[..., None] * rgb('#ffe6c0')); img = ov(img, bk * .96)
    def mist(im, w_, h_, dy, am): m = (np.exp(-((x - X) / w_) ** 2 - ((y - Y + dy) / h_) ** 2) * (.82 + .5 * n1) * am).clip(0, .9)[..., None]; return im * (1 - m) + rgb('#ffe8d4') * m
    img = mist(img, s * .95, s * .34, s * .22, .45); img = ov(img, L); img = mist(img, s * .80, 26 + s * .13, s * .08, .95); img = ov(img, _wall(X, Y + 4, s, g, 23, True)); c = Cel(); r = np.random.default_rng(7); Q = []
    for i in range(30): th = math.pi * (i + r.random()) / 30.; Q.append((X + math.cos(th) * s * .60 * r.uniform(.94, 1.12), Y + 8 + math.sin(th) * s * .085, s * r.uniform(.030, .072)))
    for g_, col, d in ((4, '#6b5688', 0), (0, '#e3b6ac', 0), (-.18, '#ffe8d0', -.14), (-.5, '#fffaf0', -.26)):
        for px, py, q in Q: rr = q + g_ if g_ >= 0 else q * (1 + g_); c.fill(np.c_[px + d * q + np.cos(a) * rr * 1.25, py + d * q * 1.1 + np.sin(a) * rr * .78], col)
    dk = 1 - (1 - min(1., (f - 23.2) / 6.)) ** 2.6                        # spray: thrown hard, then it hangs
    for i in range(130):
        th = r.uniform(.12, math.pi - .12); d = r.uniform(.25, 1.) ** .8 * 690; q = 1.6 + 6.5 * r.random() ** 3; vx, vy = math.cos(th) * 1.05, -math.sin(th) * r.uniform(.45, 1.); p = np.array([X + r.uniform(-.4, .4) * s + vx * d * dk, Y - 10 + vy * d * dk + 34 * k * k]); st = 1 + 2.2 * (1 - dk); u = np.array([vx, vy]) / math.hypot(vx, vy); nn = np.array([-u[1], u[0]]); C = p + np.cos(a)[:, None] * u * q * st + np.sin(a)[:, None] * nn * q
        if p[1] < Y: c.fill(C + [.0, q * .35], '#a987a6', .9) if q > 3.5 else None; c.fill(C, '#fff6e6', .95)
    img = ov(img, c.layer()); return mist(img, s * 1.5, 20 + s * .05, -6, .55)
