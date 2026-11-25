"""c13: the lion ship leaps.  f0-10 stern cannon drinks the air, f10 blast, f11-15 it tears the sea into a wall, f15-31 the camera races
beside the hull, f30-35 through the cloud deck, f45 apex: weightless."""
import numpy as np, cv2, math, os
from op import W, H, ships, sea
from op.cam import Camera3D
from op.sea import Sea, DAY, sheet, drops, _samp, pnoise
from op.paint import *
from op.ink import Cel, spl, resample
from op.kit import outline, rim, para, flare, smear, zoomblur, shake, speed_lines, focus_lines
from op.comp import blur, ease, ease_in, ease_out, lerp_keys, screen, radial, cam2d
from op.fx_core import put, noise, h1
from op.seakit import puff, mini, gull_top
from op import clouds
A_ = np.asarray
SUN = A_([.42, -.60, .68]); SUN = SUN / np.linalg.norm(SUN)
P2 = dict(DAY, foam='#ffffff', crest='#35cfd0')


def ship_at(f):
    """(x, z, pitch) of the ship"""
    if f <= 10: k = ease(f / 10.); return -1.8 * k + (.10 * math.sin(f * 2.4) * k), -.55 * k, .030 * k
    u = (f - 10) / 35.; x = 250 * (1 - math.exp(-1.9 * u)); z = 172 * (1 - (1 - u) ** 2) if u < 1 else 172 - 18 * (u - 1) ** 2
    return -1.8 + x, z - .55, lerp_keys(f, [(10, .03), (12, .34), (16, .46), (30, .40), (45, .10), (59, -.05)])


class Shot:
    id = 'c13'; n = 60; finish = dict(bloom=.34, vig=.16)
    EVENTS = [(0, 'suck', dict(note='stern cannon drinks air, rising swirl f0-9', size=.8)), (4, 'creak', dict(note='hull squats')), (10, 'blast', dict(size=1.3, note='coup de burst, 1 white frame')), (11, 'water', dict(size=1.2, note='wall of sea torn')), (15, 'whoosh', dict(size=1., note='camera races beside the hull, water streaming off')), (22, 'sail', dict(note='canvas cracks')), (30, 'burst', dict(size=.8, note='cloud deck')), (35, 'air', dict(note='silence opens, wind thin')), (38, 'cheer', dict(note='ten voices whooping, far and small')), (45, 'apex', dict(note='weightless: everything floats, one bell')), (50, 'clink', dict(note='barrels, tangerines drifting'))]

    def setup(self):
        self.S = Sea(21, hs=1.5, length=34, wind=(1, .35), chop=1.0); p = 'cache/c13_sky2.npy'
        if not os.path.exists(p):
            h, w = 1400, 2400; y = np.arange(h, dtype=f32)[:, None]; g = ramp(np.clip(y / 720., 0, 1), [(0, '#0f55b8'), (.45, '#2f8be0'), (.82, '#8fd0f6'), (1, '#e6f5fb')]); g = np.broadcast_to(g, (h, w, 3)).copy(); g *= 1 + .012 * fbm(h, w, (300, 90), None, 2, 3.)[..., None]
            S = clouds.Sky(h, w, g, light=(.55, -.74, .38), seed=5); S.strata(560, 700, 26, ('#ffffff', '#e2f0fb'), 3); far = clouds.tones('#ffffff', '#f3f8fd', '#cfe2f3', '#b4d0ea'); near = clouds.tones('#ffffff', '#e6f0fa', '#93b9e2', '#5f8fc8')
            S.bank(330, 800, 560, 330, far, '#dbeaf7', 7, detail=1.1); S.bank(1980, 790, 640, 400, far, '#dbeaf7', 9, detail=1.1); S.sea(716, 1420, far, near, '#e0eef9', '#a9cbec', seed=12, rows=9, size=.46, cover=.92)
            np.save(p, S.paint(1, amt=.75).astype(np.float16))
        self.sky = np.load(p).astype(f32); self.pf = []
        for i, s in enumerate((5, 14, 33)):
            q = f'cache/c13_puffb{i}.npy'
            if not os.path.exists(q): np.save(q, puff(900, 620, s, n=80, tones_=('#ffffff', '#f2f7fc', '#c9def0', '#a4c4e2'), light=(.5, -.8)).astype(np.float16))
            self.pf.append(np.load(q).astype(f32))
        r = np.random.default_rng(5); self.fl = [(r.uniform(-12, 18), r.uniform(-7, 5), r.uniform(8, 20), r.uniform(0, 6, 3), r.uniform(-1.2, 1.2, 3), i % 3) for i in range(17)]

    # ------------------------------------------------------------------ phases
    def frame(self, f):
        if f < 15: img = self.launch(f)
        elif f < 33: img = self.chase(f)
        else: img = self.apex(f)
        if f == 10: img = self.hit(img)
        if 27 <= f <= 37: img = self.cloud(img, f)
        return img

    def hit(self, img):
        L = lum(np.clip(img, 0, 1)); t = smooth(.34, .40, cv2.GaussianBlur(L, (0, 0), 1.2))[..., None]; o = rgb('#0b2a6a') * (1 - t) + rgb('#f8fdff') * t; c = Cel(); focus_lines(c, 420, 640, 330, 2400, 120, 16., '#ffffff', 3, ry=.7); l = c.layer(); o = put(o, l); return screen(o, radial(420, 640, 900, power=1.6)[..., None] * rgb('#d8fbff'))

    def launch(self, f):
        t = f / 24.; x, z, pt = ship_at(f); sk = shake(f, 1.5 + 4 * ease_in(f / 10.) if f < 10 else 15 * math.exp(-(f - 10) / 3.), 3); e = ease(f / 12.)
        tx, tz = lerp_keys(f, [(0, -8.), (10, -9.), (14, 26.)]), lerp_keys(f, [(0, 9.4), (10, 8.4), (14, 27.)])
        cam = Camera3D((-43 + 2.4 * e, -19 + 1.2 * e, 2.5), (tx, 0, tz), 1120, roll=-.05 - .02 * e, cx=960 + sk[0], cy=540 + sk[1])
        hy = cam.project(cam.eye + A_([10000., 4000, -cam.eye[2]]))[0][1]; y = grid()[1]; img = ramp(np.clip((y - hy + 900) / 900, 0, 1), [(0, '#1b66c6'), (.6, '#58ace8'), (1, '#d6f0fb')])
        s = 1.35; M = np.float32([[s, 0, -300 - 8 * f], [0, s * .55, hy - 716 * s * .55 - 8]]); bg = cv2.warpAffine(self.sky, M, (W, H), borderMode=cv2.BORDER_REFLECT); a = smooth(hy - 320, hy - 120, y)[..., None]; img = img * (1 - a * .9) + bg * a * .9
        m = ships.sunny(t + (0 if f < 10 else (f - 10) * .06), wind=lerp_keys(f, [(0, .75), (9, .45), (11, 1.6), (14, 1.3)]), cannon=ease(f / 8.)); ps = ships.pose(x, 0, z + (self.S.height(x, 0, t) * .4 if f <= 10 else 0), 0, pt, .02 * math.sin(f * .9))
        rings = [] if f < 10 else [(-22, 0, 6 + 11 * (f - 10), 3.2, 2.6 * math.exp(-(f - 10) / 6.), 1.2)]
        fo = ships.collar(m, ps, cam, 22, .75) if f <= 10 else None
        if f <= 10:                                                           # water drawn into a swirl at the cannon
            c0, zc = cam.project(A_([-23.5, -1, 0])); k = ease(f / 9.); xx, yy = grid(); dx, dy = (xx - c0[0]) / (420 * 1.0), (yy - c0[1]) / 95.; r = np.hypot(dx, dy); th = np.arctan2(dy, dx); sp = np.sin(3 * th + 7 * np.log(r + .08) + t * 26); fo = np.maximum(fo, np.clip((1.05 - r) * k, 0, 1) * (.35 + .55 * smooth(-.1, .7, sp)))
        w = self.S.render(cam, t, DAY, sun=SUN, rings=rings, foam=fo, wakes=[] if f <= 10 else [dict(x=min(x - 18, 60), y=0, head=0., L=120, W=9, dec=.4, amt=1.1)]); img = w.over(img)
        d = ships.draw([(m, ps)], cam, SUN, lw=1.15, rim=('#fff3cc', 4, .8), bounce='#4fb4e8'); lay = para(d.layer, (960, 100), (960, 1000), [(0, '#ffffff'), (.6, '#f3f0f6'), (1, '#b9cdea')], 'mul', .8)
        if f <= 10: img = sea.under(img, lay, d.z, w)
        mz = ships.at(ps, (-22.6, 0, 1.9)); mp, mzz = cam.project(mz); kk = cam.f / mzz
        if f <= 10: img = self.swirl(img, mp, kk, f)
        else:
            n = f - 10; xs = x - 19; Ns, Na = 70, 26; sg = np.linspace(0, 1, Ns); a = np.linspace(0, 1, Na); rr = np.random.default_rng(3); vk = 1 + .20 * np.interp(np.linspace(0, 16, Ns), np.arange(17), rr.normal(0, 1, 17)); backs, fronts = [], []
            for side, st in ((1, backs), (-1, fronts)):
                X = -24 + (xs + 24) * sg; T = (n / 24. + .17) * (1.0 - .55 * sg) * 1.35; P = np.zeros((Ns, Na, 3))
                for j, q in enumerate(a): tau = T * q; P[:, j] = np.stack([X + 10 * tau, side * (2.6 + 17 * vk * tau), np.maximum(60 * vk * tau - 34 * tau * tau, -.3)], -1)
                st.append(P)
            for P in backs: img = put(img, sheet(cam, P, 4 + f, P2, flow=-.9 * n, su=8, sv=2.6, light=SUN, lw=2.0, holes=.80, frag=1.35))
            img = self.jet(img, mp, kk, f, cam)
            img = img * (1 - lay[..., 3:4]) + lay[..., :3]
            for P in fronts: img = put(img, sheet(cam, P, 9 + f, P2, flow=-.9 * n, su=8, sv=2.6, light=SUN, lw=2.2, holes=.80, frag=1.35))
            nd = 340; r = np.random.default_rng(77); p0 = np.stack([r.uniform(-28, max(xs, -10), nd), r.choice([-1, 1], nd) * r.uniform(2, 5, nd), 0 * r.random(nd)], -1); v = np.stack([r.uniform(-6, 16, nd), p0[:, 1] * r.uniform(1.5, 5.5, nd), r.uniform(12, 42, nd)], -1); tau = np.clip((n / 24. + .15) * r.uniform(.5, 1.4, nd), 0, 9)[:, None]; q = p0 + v * tau; q[:, 2] -= 12 * tau[:, 0] ** 2; ok = q[:, 2] > 0
            img = put(img, drops(cam, q[ok], (v * 1.0)[ok], (.035 + .30 * r.random(nd) ** 4)[ok], P2, shutter=1 / 20., zmask=None))
            img = zoomblur(img, mp[0] + 200, mp[1] - 200, .030 * math.exp(-(n - 1) / 2.), 9)
        img = flare(img, 1560, 70, 520, '#fff2c4', .30)
        return img

    def swirl(self, img, mp, kk, f):
        """spiral of wind and drops pulled into the muzzle; the core lights up"""
        k = ease(f / 10.); c = Cel(); r = np.random.default_rng(2); R0 = 11 * kk
        for i in range(64):
            ph = (r.random() + f * (.055 + .05 * k) * (1 + .5 * r.random())) % 1; rad = R0 * (1 - ph) ** 1.3 * (.5 + .7 * r.random()) + 1.3 * kk; a0 = r.uniform(0, 6.283) + f * .30 + 5.5 * ph; th = np.linspace(a0, a0 + .55 + .9 * ph, 16); q = rad * np.linspace(1, .82, 16); P = np.c_[mp[0] - 1.2 * kk * (1 - ph) * 5 + np.cos(th) * q * .62, mp[1] + np.sin(th) * q * .80]
            c.line(P, (1.4 + 3.6 * r.random()) * (.5 + k), ['#ffffff', '#dff6ff', '#9fe0f6'][i % 3], taper=(.4, .3), seed=i, alpha=.85 * k * min(1, ph * 4))
        img = put(img, c.layer()); g = radial(mp[0], mp[1], (2.0 + 4.6 * k) * kk, power=2.2)[..., None] * rgb('#8ff0ff') * 1.3 * k; img = img + g + radial(mp[0], mp[1], (.8 + 1.5 * k) * kk, power=1.4)[..., None] * rgb('#ffffff') * 1.4 * k ** 2; return img

    def jet(self, img, mp, kk, f, cam):
        """air cannon blast: ragged cone of white pressure, drawn"""
        n = f - 10; c = Cel(); r = np.random.default_rng(f * 3); d0 = cam.project(A_([-23., 0, 1.5]))[0]; ax = d0 - mp; L = np.hypot(*ax) + 1; u = ax / L; nn = A_([-u[1], u[0]]); far = mp + u * (L + 900 * math.exp(-n / 5.) + 200)
        for k, (wd, col, al) in enumerate(((2.6, '#bfeafc', .75), (1.8, '#e8faff', .9), (1.0, '#ffffff', 1.))):
            pts = []; w0, w1 = .8 * kk * wd * .6, (3.6 + 2.2 * n) * kk * wd * .35
            for sd in (1, -1):
                q = []
                for i in range(19): v = i / 18.; q.append(mp + (far - mp) * v + nn * sd * (w0 + (w1 - w0) * v ** .8) * (1 + .30 * (r.random() - .5) * (1 if i % 2 else 2.1)))
                pts += q if sd > 0 else q[::-1]
            c.fill(spl(pts, True, corners=(0, 18, 19, 37), n=4), col, al * math.exp(-max(0, n - 2) / 3.))
        for i in range(30): v = r.random(); p = mp + (far - mp) * v + nn * r.normal(0, 1) * (1 + 3 * v) * kk; c.line(A_([p, p + u * r.uniform(60, 260)]), r.uniform(1.5, 5), '#ffffff', taper=(.2, .6), seed=i, alpha=.8)
        img = put(img, c.layer()); return img + radial(mp[0], mp[1], 420 * math.exp(-n / 4.) + 120, power=2)[..., None] * rgb('#c8f6ff') * .8 * math.exp(-n / 4.)

    def chase(self, f):
        t = f / 24.; x, z, pt = ship_at(f); u = (f - 15) / 17.; ps = ships.pose(x, 0, z, 0, pt, lerp_keys(f, [(15, -.10), (24, .05), (33, .02)]) + .012 * math.sin(f * 2.1)); sk = shake(f, 5.5 * (1 - .4 * u), 9)
        off = lerp_keys(f, [(15, (-30., -15., -7.)), (22, (-9., -16.5, -5.)), (32, (19., -19., -1.))]); tg = lerp_keys(f, [(15, (2., 0, 6.5)), (22, (5., 0, 7.)), (32, (9., 0, 9.))]); cam = Camera3D(ships.at(ps, off), ships.at(ps, tg), lerp_keys(f, [(15, 820.), (32, 930.)]), roll=lerp_keys(f, [(15, -.22), (32, -.05)]), cx=960 + sk[0], cy=540 + sk[1])
        rd = cam.rays(); el = rd[..., 2] / np.linalg.norm(rd, axis=-1); img = ramp(np.clip(el * .5 + .5, 0, 1), [(0, '#0d4fa6'), (.48, '#cfeaf8'), (.56, '#86c8f2'), (.75, '#2f86da'), (1, '#1558b4')])
        if cam.eye[2] > .5:
            w = self.S.render(cam, t, DAY, sun=SUN, step=8, rings=[(-22, 0, 40 + 7 * (f - 10), 6, 1.2, .9), (-22, 0, 16 + 5 * (f - 10), 4, .8, .7)], wakes=[dict(x=75, y=0, head=0., L=110, W=14, dec=.3, amt=1.1)], lace_m=13, haze=.55); img = w.over(img)
        img = smear(img, 46, 30, 7)
        # clouds rushing past
        for i in range(7):
            q = (h1(3, i) + u * (1.1 + .5 * h1(5, i))) % 1.25 - .12; T = self.pf[i % 3]; s = .5 + 1.2 * h1(7, i); X = 200 + 1700 * h1(9, i) - 500 * q; Y = -500 + 1900 * q; lay = cv2.warpAffine(T, np.float32([[s, 0, X - 450 * s], [0, s, Y - 310 * s]]), (W, H)); lay = smear(lay, -60, 190, 7); img = put(img, lay, (.35 + .4 * u) * (.5 + .5 * h1(2, i)))
        c = Cel(); dr = cam.project(ships.at(ps, (0, -8, 5)))[0] - cam.project(ships.at(ps, (30, -8, 5)))[0]; an = math.atan2(dr[1], dr[0]); speed_lines(c, an, 46, 3.2, (300, 1300), '#ffffff', 40 + f // 2, alpha=.5); img = put(img, c.layer())
        m = ships.sunny(t * 1.4, wind=1.25, cannon=1.); d = ships.draw([(m, ps)], cam, SUN, lw=1.5, rim=('#fff3cc', 6, .85), bounce='#4fb4e8'); lay = para(d.layer, (1200, -100), (700, 1150), [(0, '#ffffff'), (.5, '#f4f1f6'), (1, '#a9c4ea')], 'mul', .9); img = img * (1 - lay[..., 3:4]) + lay[..., :3]
        # the sea still hanging on the hull: near-side skirts peel aft
        hl = m.hull; Ns, Na = 80, 22; sg = np.linspace(.02, .97, Ns); a = np.linspace(0, 1, Na); rr = np.random.default_rng(8); k = 1 + .30 * np.interp(np.linspace(0, 22, Ns), np.arange(23), rr.normal(0, 1, 23)); die = 1 - .55 * u
        for lay_i, (q0, out, dn, al) in enumerate(((.36, 3.2, 8.0, 1.), (.50, 1.0, 4.2, .9))):
            base = hl.pt(sg, q0 + .05 * np.sin(sg * 21 + t * 5), -1); P = np.zeros((Ns, Na, 3))
            for j, v in enumerate(a): tau = v * die; P[:, j] = base + np.stack([-(11 + 5 * k) * tau - 5 * tau ** 2, -(.25 + out * tau * k), -dn * tau ** 1.5 * k + .4 * np.sin(sg * 40 + v * 6 - t * 22) * v], -1)
            img = put(img, sheet(cam, ships.at(ps, P), 6 + lay_i, P2, flow=-f * .34, su=13, sv=2.6, light=SUN, lw=1.8, holes=1.0 + .35 * u))
        nd = 260; r = np.random.default_rng(31); life = (r.random(nd) + f * .075) % 1; s0 = r.uniform(.05, .95, nd); b0 = hl.pt(s0, r.uniform(.2, .7, nd), -1); p = b0 + np.stack([-26 * life - 10 * life ** 2, -r.uniform(.3, 4, nd) * life, -9 * life ** 2 + r.uniform(-1, 3, nd) * life], -1); v = np.stack([-26 - 20 * life, -2 + 0 * life, -18 * life], -1); R_ = ps[0]; img = put(img, drops(cam, ships.at(ps, p), v @ R_.T, (.03 + .22 * r.random(nd) ** 3), P2, shutter=1 / 24., zmask=d.z + .6))
        return img if f > 16 else smear(img, 110 if f == 15 else 46, -64 if f == 15 else -24, 37)

    def cloud(self, img, f):
        k = lerp_keys(f, [(27, 0.), (30, .72), (32, 1.), (33, 1.), (37, 0.)]); x, y = grid(); n = cv2.resize(fbm(H // 4, W // 4, (70, 26), (1, .5), 40 + f // 2, 1.6), (W, H), interpolation=cv2.INTER_CUBIC); base = ramp(np.clip(y / H + .10 * n, 0, 1), [(0, '#ffffff'), (.6, '#eef6fc'), (1, '#cfe3f4')]); a = np.clip(k * 1.6 - .3 + .32 * n * (k < 1), 0, 1)[..., None] * (k > 0); return img * (1 - a) + base * a

    def giant(self, img, f):
        """for twelve frames a colossal man's shadow lies on the cloud sea, then the light has moved on"""
        k = float(lerp_keys(f, [(44, 0.), (48, 1.), (53, 1.), (57, 0.)])); g = f - 44; U = [(0, 1.0), (.055, .985), (.075, .93), (.05, .875), (.035, .855), (.16, .83), (.205, .77), (.25, .62), (.285, .50), (.255, .47), (.215, .58), (.165, .70), (.135, .60), (.125, .47), (.145, .28), (.15, .02), (.06, .02), (.035, .25), (0, .42)]; P = A_([(-a, b) for a, b in U] + [(a, b) for a, b in U[::-1]])
        ft, hd = A_([640. + 2 * g, 1040.]), A_([210. + 3.5 * g, 706.]); t = P[:, 1:2] ** .72; w = 700 * (1 - t) + 290 * t; Q = ft + (hd - ft) * t + np.c_[P[:, :1] * w, P[:, :1] * w * .16]; c = Cel(ss=1); c.fill(spl(Q, True, n=6), '#ffffff'); L = lum(np.clip(img, 0, 1)); a = blur(c.layer()[..., 3], 9) * smooth(.25, .7, L) * k * .46; return img * (1 - a[..., None]) + rgb('#4a74b6') * a[..., None]

    def apex(self, f):
        g = f - 33; t = 1.4 + g / 24. * .42; e = ease(min(1, g / 26.)); x, z, pt = ship_at(f); bob = 2.2 * (1 - ((f - 45) / 12.) ** 2); ps = ships.pose(0, 0, bob, 0, pt + .06, -.10 + .05 * e)
        an = math.radians(-31 - 13 * e); dist = lerp_keys(f, [(33, 54.), (40, 47.), (59, 42.)]); cam = Camera3D((7 + math.cos(an) * dist, math.sin(an) * dist, 3.5 + 2 * e), (2.5, 0, 14.4), 1150, roll=.07 - .05 * e)
        zm = 1.02 + .05 * e; img = cam2d(self.sky, 1200 - 46 * e + g * 1.2, 636 + lerp_keys(f, [(33, 150), (45, 0), (59, -26)]), zm, -.07 + .05 * e).copy()
        sunp = A_([1418., 150.]); img = img + (radial(sunp[0], sunp[1], 760, power=2.6)[..., None] * rgb('#fff0c8') * .75); img = img + radial(sunp[0], sunp[1], 120, power=.9)[..., None] * rgb('#ffffff') * 1.2
        # far cloud wisps sliding up behind
        for i in range(4): T = self.pf[i % 3]; s = .42 + .16 * i; lay = cv2.warpAffine(T, np.float32([[s, 0, -260 + 560 * i - 5 * g], [0, s * .55, 800 + 34 * i - lerp_keys(f, [(33, -120), (45, 0), (59, 24)]) * (.6 + .2 * i)]]), (W, H)); img = put(img, lay, .8)
        if 44 <= f <= 57: img = self.giant(img, f)
        m = ships.sunny(t * 1.2, wind=.55, swing=1.15 + .18 * math.sin(t * 3), cannon=.4); M2 = ships.Model(); tt = g / 24.
        for i, (bx, by, bz, rot, av, kind) in enumerate(self.fl):
            p = A_([bx, by, bz + 1.6 * ease(min(1, g / 14.)) + .4 * math.sin(tt * 1.7 + i)]); r3 = rot + av * tt * 1.6; cy_, sy_, cx_, sx_ = math.cos(r3[0]), math.sin(r3[0]), math.cos(r3[1]), math.sin(r3[1]); R3 = A_([[cy_, -sy_, 0], [sy_, cy_, 0], [0, 0, 1]]) @ A_([[1, 0, 0], [0, cx_, -sx_], [0, sx_, cx_]])
            if kind == 0: ships.barrel(M2, p, .62, 1.6, R3, f'b{i}')
            else:
                M2.ball(p, .40, '#f58a1f', f'tg{i}', n=(10, 14), lw=1.6, sh=(.78, .52, .50)); M2.ball(p + R3 @ A_([.1, 0, .46]), A_([.26, .12, .07]), '#4f9a3a', f'tl{i}', R=R3, n=(6, 10), lw=1.2)
        d = ships.draw([(m, ps), (M2, ships.pose(0, 0, bob * .3))], cam, SUN, lw=1.35, rim=('#fff1c2', 6, .95), bounce='#8ec8f2'); lay = para(d.layer, (1500, 60), (640, 1050), [(0, '#fff6dc'), (.45, '#fbf6f4'), (1, '#9fbfe8')], 'mul', .95)
        crew = [('luffy', (22.0, 0, 13.0), 0), ('usopp', (13.5, -5.2, 10.2), .6), ('chopper', (10.2, -5.8, 12.4), -.9), ('nami', (6.4, -6.6, 8.0), .25), ('sanji', (2.6, -7.0, 9.8), 1.0), ('zoro', (-1.4, -6.8, 7.2), -.2), ('robin', (-5.0, -6.8, 7.6), 0.15), ('franky', (-9.4, -6.6, 9.6), -.5), ('brook', (-13.6, -5.6, 13.2), 1.5), ('jinbe', (-18.2, -4.4, 12.4), .1)]
        front, back = Cel(), Cel()
        for i, (w, p, ro) in enumerate(crew):
            p = A_(p, float) + [0, 0, 1.5 * ease(min(1, g / 12.)) + .45 * math.sin(tt * 2 + i * 1.3)]; P, zc = cam.project(ships.at(ps, p)); ix, iy = int(np.clip(P[0], 0, W - 1)), int(np.clip(P[1] - 14, 0, H - 1)); mini(front if d.z[iy, ix] > zc else back, P[0], P[1], cam.f / zc * 1.5, w, f, 'whoop', i, ro * .6 + .12 * math.sin(tt * 2.2 + i), lw=1.6)
        img = put(img, rim(back.layer(), -2.5, 2.5, '#fff1c2', .8, .6)); img = img * (1 - lay[..., 3:4]) + lay[..., :3]; img = put(img, rim(front.layer(), -2.5, 2.5, '#fff1c2', .8, .6))
        # last water leaving the keel, hanging as beads
        nd = 150; r = np.random.default_rng(3); p = np.stack([r.uniform(-22, 22, nd), r.uniform(-9, 9, nd), r.uniform(-8, 3, nd) - tt * r.uniform(1, 5, nd)], -1); img = put(img, drops(cam, p, None, r.uniform(.05, .24, nd), P2))
        G = Cel()
        for i in range(5): q = A_([300 + 300 * i + 14 * g * (1 if i % 2 else -.7), 330 + 60 * ((i * 7) % 3) - 2 * g]); gull_top(G, q[0], q[1], 24 + 7 * (i % 3), 1.2 - .5 * i, math.sin(g * .7 + i * 2), lw=1.2)
        img = put(img, G.layer())
        # flare
        img = flare(img, sunp[0], sunp[1], 620, '#fff0cc', .34, streak=.35, ang=.06); v = A_([960., 540.]) - sunp
        for k, rr, c, al in ((.45, 34, '#ffd9a0', .20), (.78, 18, '#a8e4ff', .22), (1.25, 62, '#ffe9c0', .13), (1.6, 110, '#9fd4ff', .09)): p = sunp + v * k * 1.25; q = radial(p[0], p[1], rr, power=.35)[..., None]; img = img + q * rgb(c) * al
        return img
