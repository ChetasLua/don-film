"""f3 (60): the night storm.  The camera flies along a body that never ends; f45 the jaws open."""
import numpy as np, cv2, math
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm
from op.ink import Cel, spl
from op import bolt, kit
from op.comp import blur, radial, screen
from op.cam import Camera3D
from op.fx_core import h1, noise, put, FX
from chars import kaido as K
from shots import dragonkit as DK

CT = np.array([(-3, 226, 47), (3, 243, 52), (22, 262, 50), (52, 262, 36), (66, 238, 12), (48, 208, -14), (14, 194, -20), (-26, 182, -10), (-50, 158, 14), (-38, 138, 38), (0, 128, 50), (34, 118, 36), (46, 98, 10), (30, 70, -8), (6, 44, -13), (-6, 14, -13), (-14, -20, -14), (-40, -60, -22)], float)
EYE = DK.path_keys([(0, (-9, -8, 1.5)), (12, (1, 28, 3.5)), (24, (10, 72, 9)), (34, (3, 114, 20)), (41, (1, 150, 27)), (45, (2, 160, 29)), (48, (2, 154, 29.4)), (60, (2.6, 160, 30))])
TGT = DK.path_keys([(0, (6, 50, -9)), (12, (32, 92, 2)), (24, (2, 150, 40)), (34, (0, 214, 37)), (41, (-2, 226, 44)), (45, (-1.5, 226, 40)), (60, (-1.5, 226, 40.6))])


LITE = dict(K.LIT['storm'], L=(-.30, .16, .90), rw=13., rd=(-.8, -1.), pal=K.palette((.82, .88, .98), (.64, .64, .90)))


class Shot:
    id = 'f3'; n = 60; finish = dict(bloom=.34, vig=.20)
    EVENTS = [(0, 'whoosh', dict(note='camera skims the scales', dur=40)), (8, 'thunder', dict(size=.7, note='far bolt')), (13, 'whoosh', dict(note='claw and flame cloud pass')), (27, 'thunder', dict(size=.6)), (30, 'whoosh', dict(note='under the coil')),
              (38, 'boing', dict(size=.4, note='white figure bounces up')), (40, 'inhale', dict(size=1., note='dragon draws breath')), (45, 'roar', dict(size=1.5, note='jaws open, furnace')), (45, 'impact', dict(size=1., note='air ring')), (52, 'fire', dict(size=.8, note='furnace roar'))]

    def setup(self):
        self.sky = DK.full('night', DK.night_plate)
        self.cl = DK.full('ncl', lambda: DK.cloud_sprites(('#8194d2', '#46558f', '#262f60', '#141a3c'), 3, 5))
        r = np.random.default_rng(8); self.cp = [(r.uniform(-75, 75), r.uniform(20, 330), r.uniform(-46, 66), r.uniform(26, 64), int(r.integers(0, 5))) for _ in range(34)]; self.cp = [c for c in self.cp if not (abs(c[0]) < 24 and c[1] < 215 and -14 < c[2] < 44)]

    def cam(self, f):
        sh = kit.shake(f, .16 if f < 44 else 0, 3) + (kit.shake(f, 1.5, 5, 6., 45) if f >= 45 else 0); e = EYE(f) + np.array([sh[0] * .4, 0, sh[1] * .4]); fl = 1180 if f < 38 else 1180 + 260 * smooth(36, 46, f)
        return Camera3D(e, TGT(f) + np.array([sh[0], 0, sh[1]]), fl, roll=.16 * math.sin(f / 60 * 6.2 + .4) * (1 - smooth(30, 42, f)))

    def frame(self, f):
        cam = self.cam(f); vel = (EYE(f + .5) - EYE(f - .5)) * 24.; t = f / 24.; fl1 = 1.0 if f in (8, 9) else .5 if f in (10, 27) else .0
        img = DK.skydome(self.sky, cam, 1.30, (.74, -.36)) * (1 + .35 * fl1)
        p, z = cam.project(cam.eye + np.array([[.02, 1., .37]]) * 3000)
        if z[0] > 0: img = DK.moon(img, p[0, 0], p[0, 1], 92 * cam.f / 1180)
        # island far below, left
        c = Cel(); q, z = cam.project(np.array([[-520, 1500, -235.]])); s = cam.f * 165 / z[0]
        if z[0] > 0 and s < 900:
            DK.skull_dome(c, q[0, 0], q[0, 1], s, ('#42507f', '#2b3462', '#1a2146', '#0d122c'), lw=max(1.1, s / 160.)); l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]; img = screen(img, (radial(q[0, 0], q[0, 1] + s * .25, s * .9, s * .35, 2.)[..., None]) * rgb('#ff9a3c') * .35)
        # clouds: those behind the dragon first
        if f in (8, 9, 10): img = put(img, bolt.storm(f, (1500, -40), (1240, 620), 13, 21, t0=8, life=4, kids=8))
        if f in (27, 28, 29): img = put(img, bolt.storm(f, (330, -40), (520, 560), 11, 33, t0=27, life=4, kids=7))
        CT2 = CT.copy(); w = np.arange(len(CT)); amp = np.minimum(1, w / 3.); CT2[:, 2] += amp * 2.0 * np.sin(w * 1.1 - t * 2.4); CT2[:, 0] += amp * 2.0 * np.cos(w * .9 - t * 2.0)
        inh = smooth(36, 42, f) * (1 - smooth(43, 45, f)); ro = smooth(42.5, 45.2, f); k = math.exp(-max(0, f - 45) / 5.) * (f >= 45)
        turn = 1 - smooth(22, 40, f); CT2[0] += [-16 * turn, 10 * turn + 7 * inh - 9 * ro + 2 * k, 10 * turn + 7 * inh - 5 * ro]; CT2[1] += [-5 * turn, 4 * inh - 3 * ro, 5 * turn + 5 * inh]
        D = K.Dragon(CT2, R=7., n=760, around=36, legs=(.13, .47, .775), mane_step=.34)
        jaw = .10 + .04 * math.sin(t * 5) - .08 * inh + (1.02 - .20 * smooth(47, 52, f) + .05 * math.sin(f * 2.4) * (f > 46)) * smooth(42.8, 46, f); fu = max(.35 * smooth(37, 42, f), 1.12 * smooth(43, 46, f)) * (1 + .06 * math.sin(f * 2.1))
        wind = np.array([.10, .10 + .55 * smooth(44, 48, f), .12 + .10 * smooth(44, 48, f)])
        o = D.render(cam, f, LITE, furn=fu, jaw=jaw, wet=.75, wind=wind, hd=(.5 * turn, -.10 - .16 * inh + .50 * ro, .22 * turn + .03 * math.sin(f * 1.9) * k), fog=('#33427c', 60, 420, .72), scales=40, sm_c=cam.eye + (TGT(f) - cam.eye) * (.55 if f < 40 else .9), sm_r=95. if f < 38 else 60., ss=1.5, lw=(5.8, 2.8, 1.9), look=(0, -.06), rim=1.15, term=.16, hatch=.34, breath=1.15 * smooth(45.5, 49, f), eyes=1 + .38 * smooth(42, 45.5, f), whz=1 - 1.5 * smooth(41, 45, f), grad=((0, 0), (0, 1080), [(0, '#ffffff'), (.6, '#e6e4f4'), (1., '#8f8fbe')]), whisk=1.15, grip=(.1, .5, .2, .6))
        L = o['layer']; iz = o['iz']
        eye_g = []
        if f >= 42:
            for q in ('eyeL', 'eyeR'):
                w3 = o['info'][q]; pe, ze = cam.project(w3[None]); pe = pe[0]; ze = float(ze[0]); xi, yi = int(pe[0]), int(pe[1])
                if 8 < xi < W - 8 and 8 < yi < H - 8 and iz[yi - 6:yi + 7, xi - 6:xi + 7].max() < 1. / (ze - 5.2): eye_g.append((pe, cam.f * 7.56 / ze))
        near = []
        for (x, y, zz, s, k_) in self.cp:
            q, z = cam.project(np.array([[x + t * 2.5, y, zz]])); z = float(z[0]); sc = cam.f * s / max(z, .1) / 620.
            if z < 6 or sc < .05 or sc > 3.4: continue
            a = smooth(6, 26, z) * .85; near.append((z, q[0, 0], q[0, 1], sc, k_, a))
        for (z, x, y, sc, k_, a) in sorted(near, reverse=True):
            K.sprite(L, iz, self.cl[k_] * a, x, y, z, sc)
        if fl1: L[..., :3] *= 1 + .22 * fl1
        img = img * (1 - L[..., 3:4]) + L[..., :3]
        if o['light'] is not None: img = img + o['light'] * (1. if f < 44 else 1.15)
        if eye_g:                                                      # the eye burns: a hot core, flame dragged off its outer corner
            k = smooth(42.5, 47, f) * (1 + .10 * math.sin(f * 2.3)); c = Cel()
            for pe, un in eye_g:
                d = np.array([.80, -.60])
                for j, (ln, wd, col) in enumerate(((1.15, .155, '#d8290a'), (.90, .115, '#ff8a1e'), (.62, .075, '#ffe27a'))):
                    for m in range(3):
                        v = np.linspace(0, 1, 14); th = -.35 + .35 * m; u_ = np.array([d[0] * math.cos(th) - d[1] * math.sin(th), d[0] * math.sin(th) + d[1] * math.cos(th)]); nn = np.array([-u_[1], u_[0]]); P = pe + d * un * .16 + u_[None] * (un * ln * (1 - .22 * abs(m - 1)) * v)[:, None] + nn[None] * (un * .13 * np.sin(v * 5.5 - f * 1.5 + m * 2) * v)[:, None]; c.fill(K.band(P, lambda q: un * wd * K.tongue(q, .18, 1.2), f // 2 + m, 0, 40, .4), col, .95)
            l = c.layer() * k; l = l * np.clip(1 - np.exp(-np.hypot(*(np.stack(grid()) - np.array([eye_g[0][0][0], eye_g[0][0][1]])[:, None, None])) / (eye_g[0][1] * .10)) * 1.4, 0, 1)[..., None] if len(eye_g) == 1 else l
            img = img * (1 - l[..., 3:4]) + l[..., :3]
            for pe, un in eye_g: img = img + (radial(pe[0], pe[1], un * 1.25, power=2.4) * .75 * k)[..., None] * rgb('#ff8a30') + (radial(pe[0], pe[1], un * .34, power=1.6) * .55 * k)[..., None] * rgb('#fff0b0')
        img = self.white(img, cam, D, f, t, iz)
        m = o['pts']['mouth'] if 'mouth' in o['pts'] else None
        # rain: the roar bends it
        fr = None
        if f >= 44:
            mw = o['info']['throat']; fw = o['info']['fwd']; front = (f - 44.5) * 7.; st = 150 * min(1, (f - 43) / 3.)
            def fr(P):
                d = P - mw; l = np.linalg.norm(d, axis=1)[:, None] + 1e-6; u = d / l; a = np.clip((u @ fw), 0, 1)[:, None] ** 1.5; return u * st * a * np.exp(-np.maximum(0, l - front) / 5.) * (1 / (1 + (l / 60) ** 2)) + u * 90 * np.exp(-((l - front) / 6.) ** 2)
        r = DK.rain3d(cam, f, vel, 1500, 4, wind=(9, 2, -44.), force=fr, shutter=1 / 34. if f < 44 else 1 / 22., alpha=.72, segs=8, thick=1.25); img = img * (1 - r[..., 3:4]) + r[..., :3]
        if f < 36:                                                   # flight: the edges stream
            z = kit.zoomblur(img, W / 2, H / 2, .045, 7); x, y = grid(); e = smooth(.45, 1.05, np.sqrt(((x - W / 2) / 960) ** 2 + ((y - H / 2) / 540) ** 2))[..., None] * .85 * (1 - smooth(24, 36, f)); img = img * (1 - e) + z * e
        if f >= 45 and m is not None:
            for t0, am in ((45, 46), (50, 34), (55, 30)):
                if t0 <= f < t0 + 9: q = (f - t0) / 9.; img = bolt.bend(img, m[0], m[1], 90 + 1150 * q ** .75, 130, am * (1 - q), .9)
            if f < 49:
                c = Cel(); bolt.ring(c, m[0], m[1], (f - 45) / 8. + .05, 120, 1500, .86, 0, 30, 4, '#eaf2ff', '#6d7cc0'); l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
            if f == 45: L_ = (img @ np.array([.3, .59, .11], f32))[..., None]; img = img * .5 + (rgb('#08091a') * (1 - smooth(.28, .5, L_)) + rgb('#fff1cf') * smooth(.28, .5, L_)) * .5
            img = kit.zoomblur(img, m[0], m[1], .035 * math.exp(-(f - 45) / 4.), 7) if f < 52 else img
        return img.astype(f32)

    def white(self, img, cam, D, f, t, iz):
        """the tiny white figure: skips along the back, later bounces up before the jaws"""
        from chars import gear5 as G
        if f < 22:
            ph = (f % 10) / 10.; hop = 3.2 * 4 * ph * (1 - ph); u = .80 - .034 * (f // 10 + (ph if True else 0)) * 1.0; p, T, U, S, r = D.at(u); P = p + U * (r * 1.02 + hop); q, z = cam.project(P[None]); z = float(z[0]); Uu = cam.f * 1.9 * 1.55 / max(z, 1) / 5.2
            if z > 4 and Uu > 2.5:
                pose = 'squat' if ph < .18 or ph > .9 else 'leap' if ph < .62 else 'reach'; l = G.figure(q[0, 0], q[0, 1], Uu, pose, f=f // 2 * 2, light='night', roll=-.15); img = screen(img, radial(q[0, 0], q[0, 1] - Uu * 2.6, Uu * 5, power=2.)[..., None] * rgb('#cfd8ff') * .5); img = img * (1 - l[..., 3:4]) + l[..., :3]
                if ph < .3 and f > 3:
                    c = Cel()
                    for i in range(6): a = -.4 + i * .8 - 1.5; rr = Uu * (.35 + .9 * ph) * (1.2 - .5 * h1(i, 3)); c.fill(np.c_[q[0, 0] + math.cos(a) * Uu * (1 + 4 * ph) + np.cos(np.linspace(0, 6.28, 16)) * rr, q[0, 1] + hop * cam.f / z - abs(math.sin(a)) * Uu * .4 + np.sin(np.linspace(0, 6.28, 16)) * rr * .8], '#eef0fb', .9 * (1 - ph / .3))
                    l = c.layer(); img = img * (1 - l[..., 3:4]) + l[..., :3]
        if f >= 37:
            k = np.clip((f - 37) / 8., 0, 1); arc = math.sin(min(1, k * 1.0) * 1.5708); zz = 24 + 11.2 * arc + (.9 * math.sin((f - 45) * .42) * math.exp(-(f - 45) / 12.) if f > 45 else 0); P = np.array([[-12.6, 203.5, zz - 3.4]]) - np.array([[0, 1.3 * (1 - math.exp(-max(0, f - 45) / 3.)) * 3, 0]]); q, z = cam.project(P); Uu = cam.f * 3.4 / float(z[0]) / 5.2
            pose = 'leap' if f < 43 else 'rise'; sq = 1.35 if f < 41 else .80 if f in (44, 45) else 1.; blow = 0 if f < 45 else 1; la = np.array([[0, 0], [2, 3], [5, 8], [9, 13], [13, 18], [18, 22]], float) * Uu * .055 * (1 + blow * 1.1)
            l = G.figure(q[0, 0], q[0, 1], Uu, pose, f=f // 2 * 2, light='night', lagT=la, sq=sq, expr='laugh', rims=((.2, 1.), (-.9, .2)), amp=1 + .8 * blow); gl = radial(q[0, 0], q[0, 1] - Uu * 2.6, Uu * 4.6, power=2.)[..., None] * rgb('#dfe6ff') * .35
            img = screen(img, gl); img = img * (1 - l[..., 3:4]) + l[..., :3]
        return img
