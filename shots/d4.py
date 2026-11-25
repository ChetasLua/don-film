"""d4 Robin: "I want to live".  Composure, a lip that will not hold, the head goes down, and on 30 the whole face breaks open; a flag sweeps in on 42 and burns through to a small masked archer on a roof."""
from chars.lkit import *
from chars import robin as R_
from op import clouds
from op.comp import over, radial, blur, cam2d, screen
from op.kit import shake, flare, focus_lines, speed_lines
from op.fx_fire import embers, flames
from op.fx_core import put as fput
from op.fx_ink import paper_burn
from op.paint import ramp, fbm, repaint
import os

BX, BY, S = 905, 512, 328


class Shot:
    id = 'd4'; n = 60; finish = dict(bloom=.34, vig=.18)
    EVENTS = [(0, 'wind', dict(size=.6, note='sea wind, hair whipping, high')), (6, 'breath', dict(size=.3, note='Robin: breath catches')), (12, 'tear', dict(size=.3, note='eyes brim')), (18, 'sob', dict(size=.4, note='first tear goes')), (22, 'breath', dict(size=.8, note='head goes down, inhale through teeth f22-29')), (30, 'impact', dict(size=.9, note='DON + the high note: I want to live')), (30, 'voice', dict(size=1.0, note='Robin cry f30-42')), (36, 'sob', dict(size=.7, note='she looks at us')),
              (40, 'cloth', dict(size=1.0, note='flag sweeps across the lens, f40-42')), (43, 'ignite', dict(size=.7, note='flag catches in the centre')), (46, 'fire', dict(size=1.0, note='burn through f46-60')), (52, 'cape', dict(size=.4, note='cape snaps; the slingshot is lowered'))]

    def setup(self):
        p = 'cache/d4_sky.npy'
        if os.path.exists(p): self.sky = np.array(np.load(p, mmap_mode='r'), f32); self.roof = np.array(np.load('cache/d4_roof.npy', mmap_mode='r'), f32)
        else: self.sky = self.paint_sky(); self.roof = self.paint_roof(); np.save(p, self.sky.astype(np.float16)); np.save('cache/d4_roof.npy', self.roof.astype(np.float16))
        self.flag = self.paint_flag(); self.ring = None
        # tear particles: (birth, side, speed, angle, radius, near)
        r = np.random.default_rng(4); T = []
        for i in range(36): T.append((30 + r.uniform(0, 1.2), -1 if i % 2 else 1, r.uniform(24, 78), r.uniform(-.55, 1.30), r.uniform(8, 22), False))
        for i in range(26): T.append((31.5 + r.uniform(0, 10.5), -1 if i % 2 else 1, r.uniform(22, 60), r.uniform(-.50, 1.15), r.uniform(7, 18), False))
        for i in range(4): T.append((30 + i * 2.2, -1 if i % 2 else 1, r.uniform(80, 120), r.uniform(-.1, 1.1), r.uniform(15, 24), True))
        for i in range(9): T.append((18 + i * 1.35, -1 if i % 3 else 1, r.uniform(7, 14), r.uniform(-.1, .25), r.uniform(6, 10), False))
        self.T = T

    def paint_sky(self):
        h, w = 1300, 2300; y = np.linspace(0, 1, h)[:, None].repeat(w, 1); gx, gy = grid(h, w); g = ramp(y, [(0, '#3c4a5e'), (.35, '#6d7d8c'), (.68, '#b3b5ae'), (1, '#d8cdb4')]); g = g + np.exp(-(((gx - 420) / 620) ** 2 + ((gy - 420) / 460) ** 2))[..., None] * rgb('#fff0cc') * .30
        sk = clouds.Sky(h, w, g, light=(-.70, -.45, -.10), seed=14)
        for x, b, ww, hh, sd in ((420, 900, 1300, 620, 1), (1900, 820, 1200, 560, 2), (1200, 1150, 1700, 520, 3), (150, 1260, 1100, 420, 4), (2150, 1280, 1000, 460, 5)): sk.bank(x, b, ww, hh, ('#f3e6c8', '#a9a8a4', '#6f7984', '#49535f'), '#9aa0a0', sd, rim='#fff6da', rim_amt=.9)
        sk.strata(120, 420, 26, ('#c9c4b6', '#8f97a0'), 3); return sk.paint(5).astype(f32)

    def paint_flag(self):
        h, w = 1300, 2300; im = card(h, w, '#35629c', '#24487a', '#4f80bc', 9, .1); c = Cel(w, h); cx, cy = w / 2, h / 2; a = np.linspace(0, TAU, 90); ring = lambda x, y, r: np.c_[x + np.cos(a) * r, y + np.sin(a) * r]
        for P in (np.array([(cx - 470, cy - 34), (cx + 470, cy - 34), (cx + 470, cy + 34), (cx - 470, cy + 34)]), np.array([(cx - 34, cy - 430), (cx + 34, cy - 430), (cx + 34, cy + 430), (cx - 34, cy + 430)])): c.fill(P, '#eef0ea')
        for x, y, r in ((cx, cy, 150), (cx - 470, cy, 104), (cx + 470, cy, 104), (cx, cy - 430, 104), (cx, cy + 430, 104)): c.fill(ring(x, y, r), '#eef0ea'); c.fill(ring(x, y, r * .62), '#35629c'); c.fill(ring(x, y, r * .30), '#eef0ea')
        L = c.layer(h, w); im = im * (1 - L[..., 3:4]) + L[..., :3]; x, y = grid(h, w); wv = 1 + .028 * np.sin(x * 1.1) * np.sin(y * 1.1) + .02 * fbm(h, w, (3, 1.6), (1, .7), 4); im = im * wv[..., None]
        c = Cel(w, h); r = rs(6)
        for k in range(3): yy = 220 + k * 430; [c.line(np.array([(xx, yy), (xx + 12, yy + r.normal(0, 1))]), 2.2, '#1c3860', taper=(.1, .1), alpha=.6) for xx in range(0, w, 22)]
        L = c.layer(h, w); return (im * (1 - L[..., 3:4]) + L[..., :3]).astype(f32)

    def paint_roof(self):
        h, w = 1200, 2120; y = np.linspace(0, 1, h)[:, None].repeat(w, 1); gx, gy = grid(h, w); g = ramp(y, [(0, '#86b6d2'), (.40, '#c4dde0'), (.62, '#f4ecd2'), (1, '#e8cfa4')]); g = g + np.exp(-(((gx - 1060) / 520) ** 2 + ((gy - 470) / 400) ** 2))[..., None] * rgb('#fff4d2') * .5
        sk = clouds.Sky(h, w, g, light=(.0, -.6, -.2), seed=24)
        for x, b, ww, hh, sd in ((300, 860, 1000, 420, 1), (1850, 870, 1000, 460, 2), (1060, 900, 1300, 220, 3)): sk.bank(x, b, ww, hh, ('#fff8e4', '#dfe3de', '#a8c0cc', '#7c9cb4'), '#e6e2d2', sd, rim='#fffdf0', rim_amt=.7)
        im = sk.paint(6).astype(f32); c = Cel(w, h)
        for x, wd, top in ((190, 150, 640), (470, 90, 720), (1700, 120, 690), (1960, 170, 610)): c.fill(np.array([(x - wd / 2, h), (x - wd / 2, top + 90), (x - wd * .62, top + 90), (x, top - 40), (x + wd * .62, top + 90), (x + wd / 2, top + 90), (x + wd / 2, h)]), '#8fa6b4'); c.fill(np.array([(x, top - 40), (x + wd * .62, top + 90), (x + wd / 2, top + 90), (x + wd / 2, h), (x, h)]), '#728a9c')
        ridge = 760; c.fill(np.array([(1060 - 820, h), (1060 - 470, ridge + 70), (1060 - 120, ridge), (1060 + 120, ridge), (1060 + 470, ridge + 70), (1060 + 820, h)]), '#44546a'); c.fill(np.array([(1060, ridge), (1060 + 120, ridge), (1060 + 470, ridge + 70), (1060 + 820, h), (1060, h)]), '#344258'); r = rs(3)
        for j in range(13):
            yy = ridge + 18 + j * 33 + j * j * .6; half = 120 + (yy - ridge) * 1.6
            for i in range(26): x = 1060 + (i - 12.5) * (half * 2 / 25.) + (j % 2) * half / 25.; s = half / 25. * 1.05; c.line(spl(np.array([(x - s, yy - 4), (x - s * .8, yy + s * .5), (x, yy + s * .9), (x + s * .8, yy + s * .5), (x + s, yy - 4)]), n=4), 2.6, '#222c3e', taper=(.1, .1)); 
            c.line(np.array([(1060 - half, yy - 8), (1060, yy - 10)]), 2.0, '#9db0c0', alpha=.5, taper=(.1, .3))
        c.fill(np.array([(1060 - 150, ridge - 16), (1060 + 150, ridge - 16), (1060 + 138, ridge + 14), (1060 - 138, ridge + 14)]), '#59697c'); c.line(np.array([(1060 - 150, ridge - 15), (1060 + 150, ridge - 15)]), 6, '#fff2cf', taper=(.05, .05)); c.line(np.array([(1060 - 138, ridge + 14), (1060 + 138, ridge + 14)]), 4, INK, taper=(.05, .05)); L = c.layer(h, w); a = L[..., 3]; pt = L[..., :3] / np.maximum(a, 1e-3)[..., None]
        pt = repaint(np.where(a[..., None] > .02, pt, im).astype(f32), (9, 5, 3), seed=6, jit=(.006, .02, .022), thr=(0, .012, .016), op=.7, dry=.12, mask=a) * .6 + pt * .4; return (im * (1 - a[..., None]) + pt * a[..., None]).astype(f32)

    # ------------------------------------------------------------ small masked archer, feet at (x, y), height h
    def archer(self, x, y, h, f):
        c = Cel(); t = f / 24.; Q = lambda P: np.asarray(P, float) * [h, -h] + [x, y]; low = float(keys(f, [(44, 0.), (51, 0.), (55, 1.06, out3), (57, 1.)]))
        lw = 3.0 * (h / 292.) ** .62; fl = lambda k: .035 * math.sin(t * 13 + k * 1.3)                # cape
        cp = spl(np.array([(-.06, .80), (.06, .80), (.22, .74 + fl(0) * .3), (.42, .70 + fl(1)), (.66, .74 + fl(2) * 1.6), (.84, .62 + fl(3) * 1.8), (.70, .50 + fl(3.4)), (.56, .40 + fl(4) * 1.4), (.34, .32 + fl(5)), (.16, .16 + fl(6) * .6), (-.10, .20), (-.14, .50)]), True, n=6); P = Q(cp); c.fill(P, '#b3202a'); c.clip(P)
        for k, q in enumerate(([(.04, .76), (.30, .60 + fl(2)), (.62, .64 + fl(3) * 1.4)], [(.04, .70), (.26, .46 + fl(3)), (.54, .46 + fl(4))], [(.02, .60), (.16, .34), (.30, .34 + fl(5))])): S_ = spl(np.array(q), n=8); S2 = S_ + np.c_[0 * S_[:, 0], -.02 - .09 * np.linspace(0, 1, len(S_))]; c.fill(Q(np.vstack([S_, S2[::-1]])), '#73101e'); c.line(Q(S_), lw * .6, INK, taper=(.2, .4))
        c.unclip(); outline(c, P, lw, INK, 3)
        for k in range(2):                                             # scarf tails on the wind
            u_ = np.linspace(0, 1, 16); T_ = np.c_[.05 + u_ * (.34 - .08 * k), .80 - .03 * k + .05 * u_ + .035 * np.sin(u_ * 7 - t * 12 + k * 1.4) * u_]; wd = .030 * (1 - .3 * u_); q = Q(np.vstack([T_ + np.c_[0 * u_, wd], (T_ - np.c_[0 * u_, wd])[::-1]])); c.fill(q, '#efe6cf' if k == 0 else '#cfc2a2'); outline(c, q, lw * .8, INK, 11 + k)
        for sd in (-1, 1): L = Q(spl(np.array([(sd * .03, .46), (sd * .13, .44), (sd * .15, .22), (sd * .155, .03), (sd * .20, .0), (sd * .07, .0), (sd * .075, .22), (sd * .0, .40)]), True, n=5)); c.fill(L, '#6a4a2c'); c.clip(L); c.fill(L + [h * .05, 0], '#402a1a'); c.unclip(); outline(c, L, lw, INK, sd + 3)
        B = Q(spl(np.array([(-.12, .44), (-.135, .60), (-.15, .78), (-.05, .83), (.05, .83), (.15, .78), (.135, .60), (.12, .44)]), True, n=5)); c.fill(B, '#d9c7a0'); c.clip(B); c.fill(B + [h * .12, 0], '#9a8258'); c.fill(Q([(-.2, .50), (.2, .50), (.2, .43), (-.2, .43)]), '#3a2a1e'); [c.line(Q(q), lw * 3.4, '#4a3020', taper=(0, 0), press=0) for q in ([(-.085, .80), (-.05, .46)], [(.085, .80), (.05, .46)])]; [c.line(Q(q), lw * 1.8, '#7a5434', taper=(0, 0), press=0) for q in ([(-.085, .80), (-.05, .46)], [(.085, .80), (.05, .46)])]; fold(c, Q([(-.03, .74), (.01, .64), (-.01, .55)]), lw * .6); fold(c, Q([(.11, .56), (.04, .53), (-.04, .54)]), lw * .6); c.unclip(); outline(c, B, lw, INK, 9); pk = Q(spl(np.array([(.06, .50), (.15, .49), (.155, .38), (.06, .375)]), True, corners=(0, 1, 2, 3), n=4)); c.fill(pk, '#8a5f36'); c.clip(pk); c.fill(pk + [h * .04, h * .03], '#5a3a20'); c.unclip(); outline(c, pk, lw * .9, INK, 21)
        # staff slingshot (five prongs) held up, then lowered
        a = lerp(-1.05, -.30, low); hd_ = Q([(-.22 - .06 * low, .74 - .22 * low)])[0]; d = np.array([math.cos(a) * -1, math.sin(a)]); s0, s1 = hd_ - d * h * .46, hd_ + d * h * .40; c.line(np.array([s0, s1]), h * .050, INK, taper=(0, 0), press=0); c.line(np.array([s0, s1]), h * .030, '#8a5a2c', taper=(0, 0), press=0); n_ = np.array([-d[1], d[0]])
        for k in range(5): e = s1 + d * h * (.13 + .05 * (2 - abs(k - 2))) + n_ * h * (k - 2) * .075; c.line(np.array([s1, e]), h * .038, INK, taper=(0, 0), press=0); c.line(np.array([s1, e]), h * .020, '#d8a94a', taper=(0, 0), press=0)
        for sd in (-1, 1):
            shd = np.array([sd * .14, .77]); tg = ((hd_ - [x, y]) / [h, -h]) if sd < 0 else np.array([.10 - .02 * low, .56 - .06 * low]) if low > .3 else ((hd_ - d * h * .2 - [x, y]) / [h, -h]); el = (shd + tg) / 2 + [sd * .05, -.05]; from shots.d2 import limb; limb(c, Q([shd, el, tg]), [.085 * h, .07 * h, .06 * h], '#c98e5c', '#85552f', lw); hh = np.linspace(0, TAU, 16); g = Q([tg])[0]; c.fill(np.c_[g[0] + np.cos(hh) * h * .035, g[1] + np.sin(hh) * h * .035], '#c98e5c'); stroke(c, np.c_[g[0] + np.cos(hh) * h * .035, g[1] + np.sin(hh) * h * .035], lw * .8)
        mt = Q(spl(np.array([(-.09, .845), (-.185, .805), (-.245, .70), (-.235, .615), (-.165, .655), (-.10, .72), (-.02, .755), (.02, .755), (.10, .72), (.165, .655), (.235, .615), (.245, .70), (.185, .805), (.09, .845)]), True, corners=(3, 10), n=6)); c.fill(mt, '#b3202a'); c.clip(mt); c.fill(mt + [h * .06, h * .05], '#73101e'); c.fill(Q([(-.16, .80), (-.20, .74), (-.13, .77)]), '#e2575a'); c.unclip(); outline(c, mt, lw, INK, 17); [fold(c, Q(q), lw * .6) for q in ([(-.12, .82), (-.17, .75), (-.19, .68)], [(.10, .82), (.15, .75), (.18, .68)], [(-.06, .81), (-.08, .77), (-.07, .75)])]
        sc_ = Q(spl(np.array([(-.11, .85), (0, .87), (.11, .85), (.125, .80), (.06, .765), (0, .755), (-.06, .765), (-.125, .80)]), True, n=5)); c.fill(sc_, '#efe6cf'); c.clip(sc_); c.fill(sc_ + [h * .03, h * .035], '#b5a682'); c.unclip(); outline(c, sc_, lw * .9, INK, 13); fold(c, Q([(-.09, .82), (-.02, .80), (.06, .805)]), lw * .6); fold(c, Q([(-.07, .785), (.0, .775), (.06, .785)]), lw * .6)
        # sun-ray mask, long nose
        hx, hy = .0, .945; K = 14; a_ = np.linspace(0, TAU, K * 2 + 1)[:-1] + .11; R0, R1 = .100, .172; Sx = np.c_[hx + np.cos(a_) * np.where(np.arange(K * 2) % 2 == 0, R1, R0), hy + np.sin(a_) * np.where(np.arange(K * 2) % 2 == 0, R1, R0) * 1.02]; P = Q(Sx); c.fill(P, '#e0a52c'); c.clip(P); c.fill(P + [h * .06, h * .05], '#a86a1a'); c.unclip(); outline(c, P, lw * .9, INK, 2); e = np.linspace(0, TAU, 40); D = Q(np.c_[hx + np.cos(e) * .098, hy + np.sin(e) * .104]); c.fill(D, '#f7e6b0'); c.clip(D); c.fill(D + [h * .075, h * .02], '#c9a25a'); c.unclip(); stroke(c, D, lw * .9)
        for sd in (-1, 1): E = Q(np.c_[hx + sd * .042 + np.cos(e) * .024, hy + .014 + np.sin(e) * .017]); c.fill(E, INK); c.fill(Q(np.c_[hx + sd * .042 + np.cos(e) * .007, hy + .016 + np.sin(e) * .006]), '#ffffff')
        for sd in (-1, 1): c.line(Q(spl(np.array([(hx + sd * .016, hy + .045), (hx + sd * .045, hy + .058), (hx + sd * .075, hy + .040)]), n=5)), lw * .8, '#8a4a16', taper=(.2, .4)); c.line(Q([(hx + sd * .070, hy - .030), (hx + sd * .052, hy - .062)]), lw * .7, '#8a4a16', taper=(.2, .4))
        c.line(Q([(hx - .035, hy - .05), (hx, hy - .060), (hx + .035, hy - .05)]), lw * .9, INK, taper=(.1, .1)); N_ = Q(spl(np.array([(hx - .016, hy + .008), (hx - .09, hy - .010), (hx - .155, hy - .018), (hx - .15, hy - .045), (hx - .09, hy - .040), (hx + .012, hy - .032)]), True, n=5)); c.fill(N_, '#d8a94a'); c.clip(N_); c.fill(N_ + [0, h * .02], '#9a6a20'); c.unclip(); stroke(c, N_, lw * .8)
        L = c.layer(); L = para(L, (x, y - h), (x, y), [(0, '#ffffff'), (1, '#b7a4a4')]); return lit(L, (.15, .98), '#fffbe6', 2.5, 5, 1., .5)

    def flagframe(self, f):
        t = f / 24.; x, y = grid(); px = (x - 960) / 1000.; w1 = np.sin(6.4 * px - 9.5 * t + y * .0021 + .6); w2 = np.sin(13.5 * px - 15. * t + y * .0045); mx = x * 1.02 + 190 - 22 + 24 * w1 + 8 * w2; my = y * 1.02 + 100 + 26 * np.sin(5.2 * px - 9.5 * t + 1.1) + 9 * w2; im = cv2.remap(self.flag, mx.astype(f32), my.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); sl = np.cos(6.4 * px - 9.5 * t + y * .0021 + .6) * .62 + .38 * np.cos(13.5 * px - 15. * t + y * .0045)
        sh = smooth(.10, .30, sl)[..., None]; deep = smooth(.62, .74, sl)[..., None]; hi = smooth(.55, .85, -sl)[..., None]; im = im * (1 - .34 * sh - .14 * deep); im = im + (rgb('#fff2d0') - im) * hi * .30; return (im * ramp(np.clip((x + y * .6) / 2600., 0, 1), [(0, '#fff3dc'), (.5, '#f1efee'), (1, '#8f9cb8')])).astype(f32)

    def frame(self, f):
        if f >= 43: return self.burn(f)
        base = self.robin(f)
        if f >= 40:                                                    # the flag's edge sweeps through the lens
            fl = self.flagframe(f); x, y = grid(); e = {40: 1500., 41: 620., 42: -260.}[f] + 120 * np.sin(y / 150. + f * 2.) + 46 * np.sin(y / 61. - f); m = np.clip((x - e) / 1.5, 0, 1)[..., None]; hem = np.clip(1 - np.abs(x - e - 46) / 46., 0, 1)[..., None]; fl = fl * (1 - hem * .35); sd = np.exp(-np.maximum(e - x, 0) / 90.)[..., None] * (x < e)[..., None] * .5; b = base * (1 - sd); ln = np.exp(-((x - e) / 3.2) ** 2)[..., None] + np.exp(-((x - e - 92) / 2.0) ** 2)[..., None] * .8; o = b * (1 - m) + fl * m; o = o * (1 - np.clip(ln, 0, 1) * .85); k = np.ones((1, 61), f32) / 61; base = cv2.filter2D(o.astype(f32), -1, k) if f < 42 else o
        return np.clip(base, 0, 2.5).astype(f32)

    def burn(self, f):
        k = f - 43; z = float(keys(f, [(43, 1.10), (48, 1.16), (60, 2.30)], lambda q: q * q * (3 - 2 * q))); cyc = lerp(600., 590., min(1, k / 10.)); rf = cam2d(self.roof, 1060, cyc, z).astype(f32); rf = blur(rf, 1.5 + 2.2 * (z - 1.1)) * np.array([.80, .86, .92], f32); rf = rf * (1 - .16 * radial(960, 560, 520, 520, 1.2)[..., None]); rf = over(rf, self.archer(960, 540 + (748 - cyc) * z, 292 * z, f))
        x, y_ = grid(); fl = cam2d(self.flagframe(f), 960, 540, 1 + .34 * float(sm(k / 16.))).astype(f32); tb = float(keys(f, [(43, 0.0), (45, .055), (50, .22), (55, .44), (60, .68)], lambda q: q))
        sc = np.exp(-(((x - 960) / 420) ** 2 + ((y_ - 540) / 300) ** 2))[..., None] * min(1, (k + 1) / 5.); fl = fl * (1 - .30 * sc) + rgb('#ff9a3c') * sc * .26
        out = paper_burn(fl, tb, 5, 960, 540, rf, f, 150., '#ff6a14', '#140b08', '#4a2a16', 1.15)
        if tb > 0:                                                     # tongues licking along the true edge of the hole
            hole = (np.abs(out - rf).sum(-1) < .02).astype(np.uint8); hole = cv2.morphologyEx(cv2.resize(hole, (W // 2, H // 2), interpolation=cv2.INTER_AREA), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)); cs, _ = cv2.findContours(hole, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            if cs:
                C_ = max(cs, key=len)[:, 0].astype(float) * 2; c = Cel(); ctr = np.array([960., 560.]); an = np.arctan2(C_[:, 1] - ctr[1], C_[:, 0] - ctr[0]); K = 26
                for j in range(K):
                    a0 = -math.pi + j * TAU / K; q = C_[(an >= a0) & (an < a0 + TAU / K * 1.1)]
                    if len(q) < 24: continue
                    q = q[::max(1, len(q) // 14)]; kk = np.ones(3) / 3; q = np.c_[np.convolve(np.pad(q[:, 0], 1, 'edge'), kk, 'valid'), np.convolve(np.pad(q[:, 1], 1, 'edge'), kk, 'valid')]
                    if not ((q[:, 0] > 4) & (q[:, 0] < W - 4) & (q[:, 1] > 4) & (q[:, 1] < H - 4)).any(): continue
                    q = q[np.argsort(q[:, 0])]; flames(c, spl(q, n=3), 44 + 70 * rs(j, 2).random(), t=f / 24. + j * .37, seed=j, spread=.22, rows=1, lw=1.5, dens=.8)
                out = over(out, c.layer()); g = blur(cv2.resize(hole.astype(f32), (W, H)), 40); e = np.clip(g * (1 - g) * 4, 0, 1); out = out + e[..., None] * rgb('#ff7a24') * .38
        out = fput(out, embers(f, 90, 7, wind=(70., -210.), size=1.2, near=.10))
        return np.clip(out, 0, 2.5).astype(f32)

    def robin(self, f):
        t = f / 24.; d = f // 2 * 2; hit = f >= 30; sh = shake(f, 11, 9, 2.6, 30) if hit else np.zeros(2); z = float(keys(f, [(0, 1.0), (29, 1.05), (30, 1.13, out3), (32, 1.17, out3), (43, 1.23)]))
        bg = cam2d(self.sky, 1180 - f * 2.6, 640 + f * 1.1, 1.08 * (1 + (z - 1) * .5)).astype(f32); bg = bg + (radial(250, 260, 1000, 800, 1.5) * .26)[..., None] * rgb('#ffe9bc')
        def P(q): return np.array([BX + float(keys(q, [(0, -16.), (10, -4.), (18, 4.), (21, 8.), (26, 16.), (29, 22.), (30, -18., out3), (32, -4.), (43, 4.)])) + (3.2 * math.sin(q * 2.4) if q > 30.5 else 0), BY + float(keys(q, [(0, 14.), (10, 2.), (14, -4.), (18, 2.), (22, 16.), (26, 62.), (29, 74.), (30, -34., out3), (32, -10.), (35, -18.), (36, 0., out3), (39, -14.), (43, -6.)])) + (2.2 * math.sin(q * 3.1) if q > 30.5 else 0)])
        pz = lambda q: (P(q) - [960, 540]) * z + [960, 540] + sh; p = pz(f); s = S * z
        pitch = float(keys(f, [(0, -.10), (8, -.06), (14, .12), (18, .14), (22, .02), (26, -.24), (29, -.30), (30, .36, out3), (32, .27), (36, .22, out3), (43, .26)])); yaw = float(keys(f, [(0, .26), (14, .20), (22, .16), (29, .10), (30, .12), (36, .16), (43, .14)])); roll = float(keys(f, [(0, -.03), (10, -.07), (18, -.04), (26, .06), (29, .08), (30, -.13, out3), (32, -.07), (36, -.04), (43, -.07)]))
        lipf = ('tremble', 'press', 'tremble', 'calm')[(d // 2) % 4]
        if f < 4: ex = dict(o=.50, M='calm', gaze=(-.25, -.6), bi=.0)
        elif f < 8: ex = dict(o=.48, M='press', gaze=(-.25, -.6), bi=.02, bk=.1)
        elif f < 10: ex = dict(o=.16, M='press', bi=.04, bk=.2)
        elif f < 14: ex = dict(o=1.04, ir=.92, M='press', gaze=(-.05, .1), bi=.07, bo=-.01, bk=.2, pool=.25 + .05 * (f - 10), wob=1., blush=.25)
        elif f < 18: ex = dict(o=1.14, ir=.86, M=lipf, gaze=(-.05, .2), bi=.11, bo=-.03, bk=.35, pool=.60, wob=1., blush=.4, fore=.4, rope=.03 if f >= 16 else 0)
        elif f < 22: ex = dict(o=.92 if f < 21 else .7, ir=.88, M=mmix(lipf, 'sob', .25 + .15 * (f - 18)), clench=True, gaze=(-.05, .1), bi=.13, bo=-.05, bk=.55, pool=.72, wob=1., blush=.5, fore=.7, rope=.065)
        elif f < 30: q = (f - 22) / 8.; ex = dict(o=(.40 if f < 24 else 0.), shut='tight', M='clench', clench=True, msx=.95 + .12 * q, msy=1. + .15 * q, bi=.13, bo=-.07, bk=.9, blush=.6, fore=1., rope=.085, jit=.004, mseed=d, pool=.6, flare=.4 * q)
        elif f == 30: ex = dict(o=0., shut='tight', M='sobopen', mc=(0, -.588), lip2=.044, folds=.45, msx=1.06, msy=1.18, teeth=(.090, .028), tongue=.42, jaw=.46, bi=.15, bo=-.07, bk=1., fore=1., blush=.6, rope=.12, rope2=1, flare=1.)
        elif f < 36: ex = dict(o=0., shut='tight', M='sobopen', mc=(0, -.588), lip2=.044, folds=.45, msx=1.0 + .03 * math.sin(f * 2.1), msy=1.08 + .05 * math.sin(f * 2.7), teeth=(.086, .028), tongue=.46 + .06 * math.sin(f * 2.3), jaw=.40 + .03 * math.sin(f * 2.7), bi=.14, bo=-.07, bk=1., fore=1., blush=.6, rope=.125, rope2=1, snot=.7, flare=1., jit=.002, mseed=f)
        else: ex = dict(o=.98 if f > 36 else .45, ir=1.0, M='sobopen', mc=(0, -.588), lip2=.044, folds=.45, msx=.96 + .03 * math.sin(f * 2.1), msy=1.0 + .06 * math.sin(f * 2.5), teeth=(.086, .028), tongue=.50 + .06 * math.sin(f * 2.3), jaw=.36 + .035 * math.sin(f * 2.5), gaze=(0, .05), pool=.60, wob=1.5, bi=.16, bo=-.07, bk=.8, fore=1., blush=.6, rope=.135, rope2=1, snot=.9, flare=.9, jit=.002, mseed=f)
        ws = float(keys(f, [(0, .72), (22, .80), (29, .66), (30, 1.), (43, 1.)])); wa = float(keys(f, [(0, .10), (29, .18), (31, -.42, out3), (36, -.18), (43, -.10)])) + .16 * math.sin(d * .55); ups = float(keys(f, [(0, .06), (29, -.02), (31, .30, out3), (34, .16), (43, .12)]))
        st = dict(R_.style('fire'), cloth=R_.BLACK, brow={'in': (.125, .195), 'mid': (.38, .262), 'out': (.645, .160), 't': .046}, glow='#ffe6c4', glowa=.34, shadeshape=[(.50, .66), (.32, .48), (.36, .30), (.50, .14), (.47, -.02), (.40, -.16), (.43, -.30), (.37, -.44), (.31, -.58), (.27, -.68), (.22, -.78), (.12, -.90), (.0, -1.0), (.2, -1.10), (.9, -1.0), (1.05, 0), (.95, .6)]); shl = float(keys(f, [(0, .0), (18, .06), (24, .22), (29, .34), (30, .04, out3), (33, .12), (36, .26, out3), (39, .1), (43, .18)]))
        def body(c, F, s_):
            b = np.array([960 + (BX - 960) * z, 540 + (BY + 22 - 540) * z]) + sh * .6 + (P(f) - [BX, BY]) * [.4, .12]
            hd = F.h; ch = F.chin(); wn = s * .34; c.fill(np.array([ch + [-wn, -s * .5], ch + [wn * 1.05, -s * .5], b + [s * .36, s * 1.3], b + [-s * .34, s * 1.3]]), s_['shade']); U = F.w(hd.under()); c.fill(U + [0, s * .05], '#6b4038'); c.fill(spl(np.array([ch + [-wn * 1.1, -s * .3], ch + [-wn * .5, s * .10], ch + [0, s * .17], ch + [wn * .6, s * .12], ch + [wn * 1.2, -s * .3]]), True, n=6), '#5a342e'); bust(c, b[0], b[1], s, roll * .25, dict(s_, skin=s_['shade'], shade='#6b4038'), 'jacket', .10, 6., -1, sh=shl, length=3., f=f); c.clip(np.array([ch + [-wn * 1.3, -s * .6], ch + [wn * 1.3, -s * .6], b + [s * .40, s * 1.12], b + [-s * .40, s * 1.12]])); c.fill(spl(np.array([ch + [-wn * 1.2, -s * .5], ch + [-wn * .9, s * .02], ch + [0, s * .17], ch + [wn * .8, s * .06], ch + [wn * 1.2, -s * .5]]), True, n=6), '#5a342e'); c.unclip()
        cut = float(keys(f, [(0, .30), (8, .26), (14, .55), (43, .6)]))
        def post(c, F, C):                                             # hair shadow over the eyes while she still hides
            if f < 13: q = float(keys(f, [(0, -.16), (7, -.14), (13, .42)])); Q_ = np.vstack([spl(F.X([(-.9, q - .02), (-.45, q + .05), (-.1, q - .03), (.25, q + .05), (.6, q - .01), (.9, q + .06)]), n=6), F.X([(1.0, 1.1), (-1.0, 1.1)])]); c.fill(Q_, '#2a1a24', .50)
        D = R_.draw(p[0], p[1], s, yaw, pitch, roll, None, 'fire', d if f < 29 else f, hair='short', wind=(wa, ws), ex=ex, body=body, posfn=pz, step=2 if f < 29 else 1, st=st, upsway=ups, post=post, length=1.15); F = D['F']
        bg = over(bg, np.zeros((H, W, 4), f32)); lay_ = stack(D['back'], D['body'], D['face'], D['front'])
        # loose strands thrown off the right side of the head
        c = Cel(); tt = d / 24. if f < 29 else t; rt = [F.X3([q])[0] for q in ((.60, .62, .2), (.70, .30, .1), (.72, -.10, .05), (.40, .92, .1), (.66, .48, .1))]; ls = [Lock(strand(rt[k], s * (.8 + .16 * k), .25 - .3 * (f >= 30), -.05 - .30 * (f >= 30) + .12 * k, tt, amp=.36, waves=1.0, speed=1.5, ph=k * .4, curl=-.6 + .2 * k), s * (.050 - .006 * k), k=1.7, n=30, inner=0, hi=[(.2, .6, .1)]) for k in range(5)]; draw_mass(c, ls, st['hair'], 1.6, 1.6, 4, (-.6, -.8)); lay_ = lay(lit(c.layer(), st['d'], st['rim'], 1.5, 3, .7), lay_)
        lay_ = para(lay_, (p[0] - s * 1.3, p[1] - s), (p[0] + s * 1.5, p[1] + s * 1.6), [(0, '#fff6e4'), (.45, '#f3e6dc'), (1, '#7f8098')]); base = bg
        if hit and f < 37: c = Cel(); focus_lines(c, p[0], p[1] + 40, s * 1.75 + 30 * (f % 2), 1700, 110, 9., '#f6ecd4', seed=f // 2 if f > 31 else f, ry=.8); base = over(base, c.layer(), .5 * math.exp(-(f - 30) / 5.))
        base = over(base, lay_)
        # tears in the wind
        c = Cel(); cn = Cel(); up = -F.dn
        for i, (t0, sd, v, a, rad, near) in enumerate(self.T):
            q = f - t0
            if not 0 <= q < 16: continue
            if t0 >= 30 and q < (3.5 if near else 1.6): continue
            j = rs(i, 12); F0 = F.X([(sd * (.50 + .10 * j.random()), .06 - .55 * j.random())], .03)[0] if t0 >= 30 else F.X([(sd * .36, -.80)], .03)[0]; dirx = sd * math.cos(a) * .95 + .30; diry = -math.sin(a) if t0 >= 30 else .5; n_ = math.hypot(dirx, diry); vx, vy = dirx / n_ * v, diry / n_ * v; kd = .085; k = (1 - math.exp(-kd * q)) / kd; X = F0[0] + vx * k + 1.3 * q * q * .5; Y = F0[1] + vy * k + .55 * q * q; ax, ay = vx * math.exp(-kd * q) + 1.3 * q, vy * math.exp(-kd * q) + 1.1 * q; g = rad * z * min(1., .55 + q * .2) * (1 + (.16 * q if near else 0)); bead(cn if near else c, X, Y, g, math.atan2(ay, ax), (.45 + min(1.1, math.hypot(ax, ay) / 34.)) * (.45 if near else 1.), 2.6 if not near else 5.)
        base = over(base, c.layer()); L = cn.layer(); base = over(base, blur(L, 7.)) if L[..., 3].max() > 0 else base
        if f == 30: base = base * .62 + rgb('#fff8e8') * .38
        return base
