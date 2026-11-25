"""d5  FEAR.  the palm fills the lens; friend after friend: there, dented, gone (f10 20 28 34 39 43); dropped things; one round eye."""
import numpy as np, math, cv2, importlib
from op import W, H, comp, kit
from op.paint import rgb, f32, grid, smooth, ramp, fbm, mottle, lum
from op.ink import Cel, spl, xf, over_roi
from op.kit import outline, hatch, shape
from op.comp import over, screen
from chars import kuma as KU

INK = '#0c0a10'; PWD = 7400; PI5 = math.pi / 2
VAN = [10, 20, 28, 34, 39, 43]
# name, panorama x, head px, (screen offset), yaw, pitch, roll, expr
CAST = [('zoro', 800, 150, 'glare', .30, .06, .10), ('brook', 1820, 150, 'terror', -.16, .10, -.06), ('chopper', 2840, 158, 'brim', -.06, .12, -.05), ('sanji', 3800, 150, 'glare', -.30, -.04, -.12), ('franky', 4960, 138, 'roar', .24, .04, .06), ('usopp', 6060, 150, 'shout', .10, .12, .08)]


class Shot:
    id = 'd5'; n = 60; finish = dict(bloom=.22, vig=.34, sat=.96)
    EVENTS = [(0, 'air', dict(note='air sucked away, sub rumble', dur=9)), (4, 'glint', dict(note='round glasses')), (9, 'paw', dict(size=1.0, note='palm fills the lens, pressure thump'))] + [(f, 'vanish', dict(size=.6 + .08 * i, note=n + ' gone: paw-shaped pop')) for i, (f, n) in enumerate(zip(VAN, ('Zoro', 'Brook', 'Chopper', 'Sanji', 'Franky', 'Usopp')))] + \
             [(14, 'drop', dict(note='sword sticks in the ground')), (24, 'drop', dict(note='cane clatters')), (13, 'whip', {}), (23, 'whip', {}), (31, 'whip', {}), (36, 'whip', {}), (40, 'whip', {}), (17, 'gasp', dict(note='Brook: the jaw drops, bones rattle')), (26, 'cry', dict(note='Chopper reaches out')), (30, 'drop', dict(note='a small hat, soft')), (40, 'drop', dict(note='wrench')), (45, 'drop', dict(note='goggles, glass')), (46, 'silence', dict(note='everything stops; high ringing')), (49, 'shrink', dict(note='pupil to a dot')), (55, 'heartbeat', dict(size=.8))]

    # ---------------------------------------------------------------- plates
    def grove(self):
        p = 'cache/d5_grove2.npy'
        import os
        if os.path.exists(p): return np.load(p, mmap_mode='r')
        r = np.random.default_rng(5); y, x = grid(H + 120, PWD); sky = ramp(y / (H + 120), [(0, '#27402f'), (.30, '#5f8050'), (.55, '#b5bf6c'), (.68, '#dfe09a'), (.74, '#8f9f58'), (1, '#26382a')]); sky = sky * (1 + .05 * fbm(H + 120, PWD, (260, 90, 30), None, 3)[..., None]); c = Cel(PWD, H + 120, ss=1); c.fill(np.array([(0, 0), (1, 0), (1, 1)]), '#000000', 0.)
        def trunk(x0, w, col, sh, hi, lw, base, seed, stripes=5):
            rr = np.random.default_rng(seed); ys = np.linspace(-40, base, 12); ln = rr.normal(0, w * .03, 12); fl = 1 + 1.5 * np.clip((ys - base + w * 1.1) / (w * 1.1), 0, 1) ** 2.2 + .25 * np.clip((120 - ys) / 160, 0, 1); L = np.c_[x0 - w / 2 * fl + ln, ys]; R = np.c_[x0 + w / 2 * fl + ln, ys]; roots = []
            q = [np.array([x0 + w * .5 * fl[-1], base])]
            for k in range(7): u = 1 - k / 6.; q += [np.array([x0 + (u - .5) * w * fl[-1] + w * .06, base + w * rr.uniform(.10, .28)]), np.array([x0 + (u - .5) * w * fl[-1] - w * .10, base - w * rr.uniform(.0, .06)])]
            Q = np.vstack([spl(L, n=4), np.array(q)[::-1], spl(R, n=4)[::-1]]); c.fill(Q, col); c.clip(Q); m = spl(np.c_[x0 + w * .08 + ln * 2 + w * .05 * np.sin(ys * .02), ys], n=4); c.fill(np.vstack([m, spl(R, n=4)[::-1]]), sh); c.fill(np.vstack([spl(np.c_[x0 - w * .5 * fl + ln, ys], n=4), spl(np.c_[x0 - w * .36 * fl + ln + w * .03 * np.sin(ys * .03), ys], n=4)[::-1]]), hi)
            for k in range(stripes): a = -.5 + (k + .5) / stripes; st = spl(np.c_[x0 + a * w * fl + ln + w * .02 * np.sin(ys * .015 + k), ys], n=4); c.line(st, lw * rr.uniform(.6, 1.6), INK, taper=(.1, .3), seed=seed + k, alpha=.85); 
            for k in range(int(9 * stripes / 5)): yy = rr.uniform(0, base); xx = x0 + rr.uniform(-.1, .5) * w; hatch(c, np.array([(xx, yy), (xx + w * .22, yy + 20), (xx + w * .18, yy + w * .7), (xx - w * .05, yy + w * .6)]), 1.1, max(5, w * .035), max(1., lw * .5), INK, seed * 9 + k, length=w * .16, fade=.5)
            c.unclip(); outline(c, Q, lw * 1.3, INK, seed, press=.5)
        for k in range(26): trunk(PWD * (k + r.uniform(0, .8)) / 26, r.uniform(44, 90), '#a3ad66', '#8a985a', '#c6ca84', 1.0, r.uniform(735, 770), 100 + k, 3)
        for k in range(44):                                           # bubbles
            bx, by, R = r.uniform(0, PWD), r.uniform(60, 760), r.uniform(24, 130) * (1 + (k % 9 == 0)); a = np.linspace(0, 6.283, 60); o = np.c_[bx + np.cos(a) * R, by + np.sin(a) * R]; c.fill(o, '#dfe8b0', .16); c.line(o, max(1.4, R * .022), '#f2f6cc', taper=(0, 0), alpha=.75); b = np.linspace(3.5, 4.6, 16); c.line(np.c_[bx + np.cos(b) * R * .80, by + np.sin(b) * R * .80], R * .10, '#fbffe0', taper=(.4, .4), alpha=.85); b = np.linspace(.4, 1.0, 10); c.line(np.c_[bx + np.cos(b) * R * .82, by + np.sin(b) * R * .82], R * .06, '#e6efb4', taper=(.4, .4), alpha=.6)
        for k in range(11): trunk(PWD * (k + .25 + r.uniform(0, .5)) / 11, r.uniform(170, 330), '#5d7444', '#2b3d2e', '#a9b766', 2.2, r.uniform(800, 850), 200 + k, 7)
        g = spl(np.c_[np.linspace(-20, PWD + 20, 60), 838 + r.normal(0, 9, 60)], n=4); c.fill(np.vstack([g, [(PWD + 20, 1300), (-20, 1300)]]), '#2b3f2d'); 
        for k in range(150): gx, gy = r.uniform(0, PWD), 860 + 330 * r.random() ** 1.4; wd = r.uniform(60, 340) * (gy / 900) ** 3; c.fill(spl([(gx - wd, gy), (gx, gy - wd * .045), (gx + wd, gy), (gx, gy + wd * .05)], True, n=5), ('#48603a', '#1a281f', '#6d8046')[k % 3], .9)
        for k in range(900): gx, gy = r.uniform(0, PWD), 845 + 340 * r.random(); h_ = r.uniform(10, 26) * (gy / 850) ** 3; c.line(np.array([(gx, gy), (gx + h_ * .15, gy - h_ * .6), (gx + h_ * r.normal(.2, .4), gy - h_)]), 1.6 + h_ * .03, INK if k % 3 else '#8d9c50', taper=(.02, .8), seed=k)
        for k in range(5): trunk(PWD * (k + .05 + r.uniform(0, .25)) / 5 + 420, r.uniform(260, 380), '#1f2d25', '#0e1614', '#3d5236', 3.0, 1010, 300 + k, 6)
        L = c.rgba(); im = sky * (1 - L[..., 3:4]) + L[..., :3]; im = mottle(np.clip(im, 0, 1), .03, 4); os.makedirs('cache', exist_ok=True); np.save(p + '.tmp.npy', im.astype(np.float16)); os.replace(p + '.tmp.npy', p); return np.load(p, mmap_mode='r')

    def setup(self):
        self.G = self.grove(); self.fr = {}

    def fear(self, L, light=(-1, -.4)):
        """a colour cel -> three-tone fear: salmon flesh, navy cloth, sick light, big blacks"""
        a = L[..., 3:4]; c = np.where(a > 1e-3, L[..., :3] / np.maximum(a, 1e-3), 0); l = lum(c); warm = smooth(.06, .16, c[..., 0] - c[..., 2]) * smooth(-.10, .06, c[..., 0] - c[..., 1] + .02); e = lambda t, k=.02: smooth(t - k, t + k, l)[..., None]
        W_ = rgb('#1a0c14') + (rgb('#7c2836') - rgb('#1a0c14')) * e(.22); W_ = W_ + (rgb('#e9705f') - W_) * e(.45); W_ = W_ + (rgb('#ffb597') - W_) * e(.76)
        K = rgb('#080a12') + (rgb('#122a4a') - rgb('#080a12')) * e(.13); K = K + (rgb('#33604e') - K) * e(.34); K = K + (rgb('#c3cf86') - K) * e(.66)
        o = K * (1 - warm[..., None]) + W_ * warm[..., None]; return np.dstack([o * a, L[..., 3]])

    def friend(self, i, var=0):
        """finished fear cel of friend i in a 1920x1080 frame (cached): lane head if it exists, our own black body under it"""
        if (i, var) in self.fr: return self.fr[i, var]
        n, _, s, ex, yaw, pit, rol = CAST[i]; cx, cy = 960, 430; c = Cel(ss=2); lw = 5.5; q = lambda P: np.asarray(P, float) * s + [cx, cy]; P = lambda pts: q(spl(pts, True, n=6)); r = np.random.default_rng(i)
        def sh(pts, col, seed=1, dark=None, hat=None, w=lw):
            Q = P(pts); c.fill(Q, col); c.clip(Q)
            if dark is not None: D = P(dark); c.fill(D, INK if len(dark) > 99 else '#0b0c12'); hatch(c, D + np.array([-46, 0]), 1.15, 10, 2.4, INK, seed, length=54, fade=.35)
            c.unclip(); outline(c, Q, w, INK, seed, press=.5); return Q
        def ln(pts, w=2.6, col=INK): c.line(q(spl(pts, n=8)), w, col, taper=(.2, .5), seed=int(abs(pts[0][0]) * 91))
        hands = []; hands_b = []
        if n == 'zoro':
            sh([(1.5, 1.5), (2.4, 1.9), (3.0, 3.2), (3.1, 4.6), (2.7, 5.6), (2.1, 5.6), (2.2, 4.0), (1.8, 3.0)], '#d9986a', 3, [(2.5, 1.6), (2.6, 3.2), (2.5, 5.6), (3.6, 5.6), (3.4, 2)])
            sh([(-2.2, 5.6), (-2.0, 3.6), (-2.2, 2.6), (-1.9, 1.8), (-1.1, 1.35), (-.45, 1.1), (.45, 1.1), (1.1, 1.35), (1.9, 1.7), (2.2, 2.6), (2.0, 3.6), (2.2, 5.6)], '#e6e2d2', 4, [(.5, 1.0), (.9, 2.2), (.4, 3.4), (.8, 5.8), (3, 5.8), (3, 1)]); [ln([(-2.0, 2.5 + .42 * k), (0, 2.8 + .42 * k), (2.0, 2.4 + .42 * k)]) for k in range(5)]; sh([(-2.1, 4.5), (0, 4.8), (2.1, 4.4), (2.3, 6), (-2.3, 6)], '#2f6b48', 5, [(.6, 4.4), (.8, 6), (2.6, 6), (2.6, 4.2)])
            sh([(-1.5, 1.5), (-2.4, 1.8), (-3.0, 2.9), (-2.8, 3.9), (-2.1, 3.9), (-1.6, 2.8)], '#d9986a', 6, [(-2.2, 3.0), (-1.4, 2.4), (-1.4, 4), (-2.4, 4)]); hands.append(('grip', 'R', (150, 20, -30), (-1.75, 2.95), 96, 5))
            c.line(q([(-2.35, 4.4), (-.4, 1.2), (.9, -1.0)]), 26, INK, taper=(0, .02)); c.line(q([(-2.35, 4.4), (-.4, 1.2), (.86, -.93)]), 16, '#dfe8c0', taper=(0, .02)); c.line(q([(-2.25, 4.4), (-.3, 1.2), (.9, -.9)]), 4, '#33504a', taper=(0, .1))
        elif n == 'brook':
            sh([(-1.6, 5.6), (-1.5, 3.2), (-1.3, 2.0), (-.9, 1.6), (-.3, 1.25), (.3, 1.25), (.9, 1.6), (1.3, 2.0), (1.5, 3.2), (1.7, 5.6)], '#23202c', 4, [(.3, 1.2), (.5, 5.8), (2, 5.8), (2, 1.2)]); sh([(-.5, 1.3), (0, 1.15), (.5, 1.3), (.55, 1.9), (.2, 2.7), (0, 3.0), (-.2, 2.7), (-.55, 1.9)], '#e6ecd0', 5); ln([(-.3, 1.6), (0, 2.1), (.3, 1.6)]); ln([(0, 2.1), (0, 2.8)])
            for sg in (-1, 1):
                sh([(sg * 1.0, 1.75), (sg * 1.7, 1.9), (sg * 2.7, 3.2), (sg * 2.5, 3.75), (sg * 2.0, 3.6), (sg * 1.2, 2.8)], '#23202c', 6 + sg, [(sg * 1.5, 2.5), (sg * 2.5, 3.4), (sg * 2.3, 4.0), (sg * 1.2, 3.1)]); sh([(sg * 2.75, 3.3), (sg * 2.3, 3.85), (sg * 1.35, 2.0), (sg * 1.25, 1.35), (sg * 1.75, 1.25), (sg * 2.0, 1.9)], '#23202c', 9 + sg, [(sg * 2.3, 3.0), (sg * 2.0, 3.9), (sg * 1.2, 2.0), (sg * 1.5, 1.9)]); hands_b.append(sg)
        elif n == 'usopp':
            sh([(-1.7, 5.6), (-1.6, 3.4), (-1.5, 2.3), (-1.0, 1.6), (-.4, 1.2), (.4, 1.2), (1.0, 1.6), (1.6, 2.2), (1.8, 3.4), (1.9, 5.6)], '#e0ae74', 4, [(.3, 1.2), (.7, 2.6), (.3, 4), (.7, 5.8), (2.2, 5.8), (2.2, 1.2)]); ln([(-.9, 2.5), (-.3, 2.8), (0, 2.6), (.3, 2.8), (.8, 2.5)]); ln([(0, 2.7), (0, 5)]); [ln([(-.7, 3.4 + .6 * k), (-.2, 3.5 + .6 * k)]) for k in range(3)]
            for sg, yy in ((-1, 1.8), (1, 1.6)): sh([(sg * .9, 1.5), (sg * 1.5, 1.5), (sg * 1.1, 3.2), (sg * 1.0, 5.6), (sg * .5, 5.6), (sg * .7, 3.2)], '#8a5f34', 6 + sg)
            sh([(1.4, 2.0), (2.1, 2.3), (2.6, 3.4), (2.3, 4.6), (1.7, 4.4), (1.6, 3.2)], '#e0ae74', 9, [(2.0, 2.2), (2.2, 3.4), (2.0, 4.8), (2.9, 4.6), (2.9, 2.2)]); hands += [('splay', 'L', (190, 46, 12), (-1.35, 2.2), 178, 2.5)]
        elif n == 'sanji':
            sh([(1.3, 1.6), (2.0, 1.8), (2.7, 2.8), (2.4, 4.2), (1.8, 4.2), (1.7, 3.0)], '#20202a', 3); hands.append(('reach', 'L', (160, 10, 40), (2.05, 2.0), 104, 4))
        elif n == 'franky':
            sh([(-3.0, 5.6), (-2.6, 3.4), (-3.2, 2.6), (-2.9, 1.5), (-1.6, 1.2), (-.6, 1.0), (.6, 1.0), (1.6, 1.2), (2.9, 1.5), (3.2, 2.6), (2.6, 3.4), (3.0, 5.6)], '#e5ab80', 4, [(.4, 1.0), (.9, 2.4), (.3, 3.8), (.7, 5.8), (3.4, 5.8), (3.4, 1)]); ln([(-1.6, 2.4), (-.6, 2.9), (0, 2.6), (.6, 2.9), (1.6, 2.4)]); ln([(0, 2.7), (0, 5.2)]); [sh([(sg * .9, 1.1), (sg * 1.9, 1.2), (sg * 2.2, 3.2), (sg * 1.9, 5.6), (sg * 1.1, 5.6), (sg * 1.4, 3.2)], '#3f9ac0', 6) for sg in (-1, 1)]
            for sg in (-1, 1): sh([(sg * 2.6, 3.6), (sg * 3.0, 1.4), (sg * 3.5, .4), (sg * 4.6, .4), (sg * 5.1, 1.6), (sg * 4.9, 3.6), (sg * 4.1, 4.4), (sg * 3.2, 4.4)], '#d33a2c', 8, [(sg * 3.9, .2) if sg > 0 else (-3.2, .2), (sg * 4.3, 2.2) if sg > 0 else (-3.0, 2.2), (sg * 4.0, 4.6) if sg > 0 else (-2.6, 4.6), (sg * 5.6, 4.6) if sg > 0 else (-2.2, 4), (sg * 5.6, .2) if sg > 0 else (-2.4, 1)]); a5 = np.linspace(-PI5, 2 * math.pi - PI5, 11)[:-1]; sh([(sg * 3.9 + math.cos(t_) * (.62 if k % 2 == 0 else .28), 2.5 + math.sin(t_) * (.62 if k % 2 == 0 else .28)) for k, t_ in enumerate(a5)], '#f0f0da', 9, w=3.5)
        else:
            sh([(-1.0, 5.0), (-1.15, 3.2), (-1.0, 2.0), (-.6, 1.3), (.6, 1.3), (1.0, 2.0), (1.15, 3.2), (1.0, 5.0)], '#c18a55', 4, [(.2, 1.2), (.4, 5.2), (1.6, 5.2), (1.4, 1.2)]); sh([(-.62, 2.0), (0, 1.8), (.62, 2.0), (.7, 3.6), (0, 4.1), (-.7, 3.6)], '#e6bd8a', 5)
            sh([(.7, 1.7), (1.4, 1.75), (2.0, 2.5), (2.0, 3.4), (1.5, 3.6), (1.1, 2.6)], '#c18a55', 7, [(1.2, 2.0), (1.9, 2.8), (1.8, 3.7), (1.2, 3.2)]); reach_ch = True
        L = c.layer()
        try:
            m = importlib.import_module('chars.' + n); kw = dict(zoro=dict(young=True, body=None), chopper=dict(hat='old')).get(n, {})
            if n == 'brook': m.EX['terror'] = m.E(md=.34 + .30 * var, ml=-.14, wob=.8 + .4 * var, tear=.9 * var); m.EX['dread'] = m.E(md=.14, ml=-.10, wob=.7); ex = 'terror' if var else 'dread'
            if n == 'chopper': ex = 'cry' if var else 'brim'
            try: h = m.head(cx, cy, s, yaw, pit, rol, expr=ex, f=0, **kw)
            except TypeError: h = m.head(cx, cy, s, yaw, pit, rol, expr=ex, f=0)
        except Exception as e:
            print('d5 fallback', n, e); c = Cel(ss=2); shape(c, q(np.c_[np.cos(KU.A) * .8, np.sin(KU.A) * .95 - .1]), '#e9a07a', lw); h = c.layer()
        L = over_layer(L, h)
        c = Cel(ss=2)
        for sg in hands_b:                                            # bone fingers dug into the cheeks, sweat, cold lines of dread down the brow
            for k in range(4): x0 = sg * (1.22 - .13 * k); pts = [(x0 + sg * .16, 1.42 - .05 * k), (x0 + sg * .05, 1.0 - .06 * k), (x0 - sg * .10, .62 - .07 * k), (x0 - sg * .26, .34 - .06 * k)]; c.line(q(spl(pts, n=5)), 23, INK, taper=(0, .1)); c.line(q(spl(pts, n=5)), 15, '#e9ecd0', taper=(0, .1)); [c.fill(q([pts[j]]) + np.c_[np.cos(KU.A), np.sin(KU.A)] * 11, '#e9ecd0') for j in (1, 2)]; [c.line(q([pts[j]]) + np.c_[np.cos(KU.A[:14]), np.sin(KU.A[:14])] * 12, 2.4, INK) for j in (1, 2)]
        if n == 'brook':
            for x, y, z_ in ((-.62, -.35, .16), (.66, -.2, .20), (.1, -.62, .13), (-.2, .36, .12), (.78, .30, .17)): d = spl([(x, y - z_ * 1.9), (x + z_ * .62, y), (x, y + z_ * .7), (x - z_ * .62, y)], True, corners=(0,), n=6); kit.shape(c, q(d), '#eaf4d8', 3.5, INK)
            for k in range(9): x = -.62 + .155 * k; c.line(q([(x, -.72), (x + .01, -.52 + .05 * (k % 3))]), 3., INK, taper=(.05, .7), seed=k)
        if n == 'chopper':                                            # the hoof comes at us, huge
            gr = 1. + .18 * var; arm_ = [(-.50, 1.60), (-.95, 1.35), (-1.75 * gr, 1.05), (-2.45 * gr, 2.05), (-1.55 * gr, 2.75 * gr), (-.60, 2.75)]; Q = P(arm_); c.fill(Q, '#c18a55'); c.clip(Q); c.fill(P([(-.3, 2.3), (-1.0, 2.35), (-1.7, 2.6), (-1.2, 3.6), (0, 3.4)]), '#0b0c12'); hatch(c, P([(-.7, 1.7), (-1.5, 1.8), (-1.6, 2.5), (-.8, 2.5)]), .8, 11, 2.4, INK, 4, length=52, fade=.4); c.unclip(); outline(c, Q, 6, INK, 3, press=.5)
            hx_, hy_ = -1.78 * gr, 1.62; R_ = .92 * gr; rim_ = np.c_[hx_ + np.cos(KU.A) * R_ * 1.02, hy_ + np.sin(KU.A) * R_ * .92]; kit.shape(c, q(rim_), '#8a5a34', 6, INK, 5)
            for sg in (-1, 1): lob = spl([(hx_ + sg * .06 * R_, hy_ - .72 * R_), (hx_ + sg * .62 * R_, hy_ - .66 * R_), (hx_ + sg * .88 * R_, hy_ - .05 * R_), (hx_ + sg * .66 * R_, hy_ + .62 * R_), (hx_ + sg * .10 * R_, hy_ + .70 * R_), (hx_ + sg * .16 * R_, hy_)], True, n=7); Q = q(lob); c.fill(Q, '#4a3030'); c.clip(Q); c.fill(Q + [26, -22], '#7a4a40'); c.fill(q(spl([(hx_ + sg * .30 * R_, hy_ - .45 * R_), (hx_ + sg * .55 * R_, hy_ - .40 * R_), (hx_ + sg * .60 * R_, hy_ - .15 * R_), (hx_ + sg * .40 * R_, hy_ - .18 * R_)], True, n=6)), '#e8b0a0'); hatch(c, Q + [-20, 22], .9, 11, 2.4, INK, 8 + sg, length=46, fade=.5); c.unclip(); outline(c, Q, 5.5, INK, 6 + sg, press=.5)
        L = over_layer(L, c.layer())
        if hands:
            from op import hands as HD
            c = Cel(ss=2); pal = dict(skin=dict(zoro='#d9986a', usopp='#e0ae74', sanji='#f3c39b')[n], shade='#8a4a3a', line=INK, light='#ffe0bd')
            for pose, side, ang, (hx, hy), hs, dist in hands: pp = q((hx, hy)); g = HD.render([HD.Hand(pose, side, HD.euler(*ang), (0, 0, 0), HD.MALE, pivot=(0, .75, 0))], HD.Cam(float(pp[0]), float(pp[1]), hs, dist), (-.3, .55, .9), 480, clip=(0, 0, W, H)); HD.ink(c, g, pal, 5.5, 3)
            L = over_layer(L, c.layer())
        L = self.fear(L); L = kit.rim(L, 9, 4, '#e6ec9c', .85, soft=1.2); ink = kit.rim(L, -10, -3, '#080a12', .9, soft=.8); self.fr[i, var] = ink; return ink

    # ---------------------------------------------------------------- dropped things
    def item(self, c, kind, x, y, s, rot):
        T = lambda P: xf(np.asarray(P, float), s, s, rot, x, y); w = max(2., s * .05)
        if kind == 'sword':
            shape(c, T([(-.06, -2.0), (.06, -2.0), (.07, .8), (0, 1.1), (-.07, .8)]), '#c9d6a0', w * .6, INK); c.fill(T([(0, -2.0), (.06, -2.0), (.07, .8), (0, 1.1)]), '#33504a'); shape(c, T([(-.30, -2.06), (.30, -2.06), (.30, -1.96), (-.30, -1.96)]), '#e9705f', w * .5, INK); shape(c, T([(-.08, -2.9), (.08, -2.9), (.08, -2.06), (-.08, -2.06)]), '#e6ecc0', w * .6, INK)
        elif kind == 'cane': c.line(T([(0, -2.2), (0, 0), (0, 2.2)]), s * .17, INK, taper=(0, 0)); c.line(T([(0, -2.2), (0, 2.1)]), s * .09, '#c3cf86', taper=(0, 0)); shape(c, T(spl([(-.1, -2.2), (-.1, -2.7), (.25, -2.95), (.6, -2.7), (.6, -2.4), (.4, -2.4), (.3, -2.65), (.1, -2.5), (.1, -2.2)], True, n=5)), '#33604e', w * .6, INK)
        elif kind == 'wrench': shape(c, T(spl([(-.18, 1.9), (-.16, -1.2), (-.60, -1.55), (-.62, -2.2), (-.28, -2.45), (-.26, -1.9), (.26, -1.9), (.28, -2.45), (.62, -2.2), (.60, -1.55), (.16, -1.2), (.18, 1.9), (0, 2.15)], True, n=5)), '#c3cf86', w, INK); c.fill(T([(0, -1.6), (.5, -1.6), (.14, -1.1), (.15, 1.9), (0, 2.0)]), '#22403f')
        elif kind == 'cig': c.line(T([(-.6, 0), (.6, 0)]), s * .30, INK, taper=(0, 0)); c.line(T([(-.55, 0), (.4, 0)]), s * .20, '#f0eecb', taper=(0, 0)); c.line(T([(.4, 0), (.58, 0)]), s * .20, '#ff6a4a', taper=(0, 0))
        elif kind == 'goggles':
            c.line(T([(-1.6, .1), (-.5, -.1), (.5, -.1), (1.6, .1)]), s * .22, INK); [shape(c, T(np.c_[sg * .5 + np.cos(KU.A) * .46, np.sin(KU.A) * .40]), '#e9705f', w, INK) for sg in (-1, 1)]; [c.fill(T(np.c_[sg * .5 - .12 + np.cos(KU.A) * .14, -.12 + np.sin(KU.A) * .10]), '#ffd9c4') for sg in (-1, 1)]
        else:                                                         # the small pink hat with the X
            br = T(spl([(-1.5, .35), (-.8, .62), (0, .70), (.8, .62), (1.5, .35), (.9, .10), (0, .02), (-.9, .10)], True, n=6)); cr = T(spl([(-.86, .30), (-.95, -.5), (-.86, -1.15), (-.3, -1.35), (.4, -1.30), (.92, -1.05), (.95, -.4), (.86, .30), (0, .44)], True, n=6)); shape(c, cr, '#e9705f', w, INK, 3); c.clip(cr); c.fill(T(spl([(.2, -1.5), (.36, -.4), (.2, .6), (1.4, .6), (1.4, -1.5)], True, n=5)), '#7c2836'); X = [T([(-.42, -.95), (.30, -.10), (.16, .02), (-.56, -.83)]), T([(.30, -.95), (.44, -.83), (-.30, .02), (-.44, -.10)])]; [c.fill(p, '#fff2d6') for p in X]; c.unclip(); [outline(c, p, w * .6, INK) for p in X]; shape(c, br, '#c8514f', w, INK, 5); hatch(c, cr, .9, s * .12, w * .4, INK, 5, length=s * .4, fade=.6)

    # ---------------------------------------------------------------- A: palm
    def palm(self, f):
        ff = f; f = int(round(f))
        x0 = 120; bg = np.array(self.G[60:60 + H, x0:x0 + W], f32); g = lum(bg)[..., None]; bg = bg * .55 + g * rgb('#8fa04e') * .45; e = (ff / 9.); s = 88 * math.exp(2.98 * e ** 1.25); cx = 960 + 190 * (1 - e) - 10 * e; cy = 640 - 110 * e + 20 * (f == 9); bg = cv2.GaussianBlur(bg, (0, 0), 3 + 7 * e); bg *= (1 - .30 * e); bg = screen(bg, np.exp(-(((grid()[0] - 860) / 760) ** 2 + ((grid()[1] - 380) / 480) ** 2))[..., None] * rgb('#dbe68a') * .55)
        img = over(bg, KU.bust(800 + 20 * e, 268 - 24 * e, 132 + 34 * e, f, glint=max(0., 1 - abs(f - 4) / 2.2)))
        c = Cel(ss=1); r = np.random.default_rng(f // 2); kit.focus_lines(c, cx, cy + s * .2, 330 + 420 * e, 1700, 120, 10, '#0a0d10', f // 2, ry=.8); img = over(img, c.layer(), .35 + .45 * e)
        if f >= 2:                                                    # arm behind the palm
            c = Cel(ss=2); a = np.array([cx - s * .75, cy + s * .9]); b = np.array([cx + s * .8, cy + s * .9]); q = np.array([a, b, [1180, 640], [1000, 600]]); c.fill(q, '#0e1513'); L = c.layer(); img = over(img, L)
        p = KU.paw(cx, cy, s, 1.0 + .10 * math.sin(f), -.10 + .12 * e, f, thrust=.15 + .6 * e).layer()
        p = kit.rim(p, -max(3, s * .02), max(3, s * .016), '#e6ee9a', .9, soft=1.4); p = kit.para(p, (0, cy + s * 1.4), (0, cy - s * 1.6), [(0, '#4a5a4a'), (.5, '#ffffff'), (1, '#ffffff')]); img = over(img, p)
        for k in range(3):                                            # pressure rings
            rr = s * (1.6 + .55 * k) + 20 * k; x, y = grid(); d = np.hypot(x - cx, (y - cy - s * .2)) ; img = img * (1 + .10 * np.exp(-((d - rr) / (10 + s * .03)) ** 2) * e)[..., None]
        img = kit.zoomblur(img, cx, cy + .2 * s, .015 + .05 * e, 7); return img if ff < 8.9 else img * .55 + .45 * rgb('#eaf6b4')

    # ---------------------------------------------------------------- B: whip to each friend
    def pov(self, f):
        # which friend are we on
        xs = [CAST[i][1] for i in range(6)]
        def camx(g):
            k = [(8, xs[0] - 16), (12.2, xs[0] + 10), (13.7, xs[1] - 30), (21.9, xs[1] + 12), (23.4, xs[2] - 24), (30.5, xs[2] + 10), (31.8, xs[3] - 18), (35.2, xs[3] + 6), (36.4, xs[4] - 16), (39.9, xs[4] + 4), (41.0, xs[5] - 8), (46, xs[5] + 2)]; return float(np.interp(g, [q[0] for q in k], [q[1] for q in k]))
        cxw = camx(f); v = camx(f + .5) - camx(f - .5); i = int(np.argmin([abs(cxw - x) for x in xs])); fv = VAN[i]; zoom = 1.06 + .012 * (f - fv) + (.10 if f == fv else 0.) + .05 * i / 5.; rol = (.05, -.05, .04, -.11, .07, -.09)[i] + .008 * math.sin(f * 1.3); sh = kit.shake(f, 3 + (16 if f in (fv, fv + 1) else 0), 5)
        M = np.float32([[1, 0, cxw - W / 2 - 80], [0, 1, 40]]); x0 = int(np.clip(cxw - W / 2 - 60, 0, PWD - W - 120)); plate = np.array(self.G[:, x0:x0 + W + 120], f32); img = comp.cam2d(plate, cxw - x0 + sh[0], 600 + sh[1], zoom, rol)
        g = lum(img)[..., None]; img = img * .74 + g * rgb('#a6b55a') * .40; img = cv2.GaussianBlur(img, (0, 0), 2.2); px0 = W / 2 + (xs[i] - cxw) * zoom; img = screen(img, np.exp(-(((grid()[0] - px0) / 560) ** 2 + ((grid()[1] - 430) / 420) ** 2))[..., None] * rgb('#cfdc80') * .42)
        ss_ = 1.0
        for j in range(6):
            dx = xs[j] - cxw
            if abs(dx) > 1500: continue
            a = f - VAN[j]
            if a > 0: continue
            L = self.friend(j, int((j == 1 and a >= -3) or (j == 2 and a >= -2))); z = zoom * (1.0, 1.16, 1.14, 1.12, 1.0, 1.22)[j] * (1.0 + (.040, .022, .050, .045, .035, .05)[j] * max(0, 6 + a)); jit = np.array([0., 0.]); g2 = (f // 2) * 2 - VAN[j]; ro_ = (-.05, .022 * (1 if f % 4 < 2 else -1), .035 * math.sin(f * 1.1), -.07, .04, .06)[j] * (1 if j in (1, 2) else max(0., 1 + g2 / 8.)); dy_ = (0, 4 * (f % 2), -12 * abs(math.sin(f * .8)), 0, -10 * max(0, 6 + g2), 0)[j]
            if a > -8: jit = kit.shake(f // 2, 2.5, 9 + j)
            cr, sr = z * math.cos(ro_), z * math.sin(ro_); M = np.float32([[cr, -sr, W / 2 - cr * W / 2 + sr * 620 + dx * zoom + jit[0]], [sr, cr, 560 + 150 * (z - 1) - sr * W / 2 - cr * 620 + 150 + jit[1] + dy_]])
            if a == 0: sx = z * .80; M = np.float32([[sx, .06, W / 2 - sx * W / 2 + dx * zoom + 20], [0, z * 1.06, 560 - z * 1.06 * 470]])
            lay = cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_LINEAR)
            if a == 0: lay = kit.smear(lay, 130, -50, 9)
            img = over(img, lay)
        px, py = W / 2 + (xs[i] - cxw) * zoom, 470
        if f == fv - 1: img = KU.dent(img, px, py + 30, 560, .10, .16)
        if f == fv: img = KU.dent(img, px, py + 20, 400, 1.2, 1.); c = Cel(ss=1); kit.focus_lines(c, px, py + 40, 520, 1700, 90, 9, '#0a0d10', f, ry=.8); img = over(img, c.layer(), .8)
        if f == fv + 1:
            img = KU.dent(img, px, py + 10, 520, .9, .5); c = Cel(ss=2)
            for p in KU.pads(px, py + 10, 520): c.line(np.vstack([p, p[:1]]), 9, '#f4ffd0', taper=(0, 0), alpha=.9)
            img = over(img, c.layer())
        if fv + 1 <= f <= fv + 4:                                      # the air rushing back: a thinning paw and torn wisps
            a = f - fv; c = Cel(ss=2); r = np.random.default_rng(i)
            for k in range(16):
                th = k / 16 * 6.28 + r.uniform(-.2, .2); r0 = 250 + 170 * a + r.uniform(-30, 60); p0 = np.array([px + math.cos(th) * r0, py + 40 + math.sin(th) * r0 * .8]); p1 = p0 + np.array([math.cos(th + .5), math.sin(th + .5) * .8]) * (120 - 22 * a); c.line(spl([p0, (p0 + p1) / 2 + r.normal(0, 14, 2), p1], n=8), 12 - 2.4 * a, '#eef6bc', taper=(.2, .7), seed=k, alpha=.8)
            img = over(img, c.layer())
        # what they were holding
        it = [(10, 'sword', 760, 420, 92, 13), (20, 'cane', 1730, 430, 74, 24), (20, 'tophat', 0, 0, 0, 0), (28, 'hat', 2860, 300, 130, 34), (34, 'cig', 3780, 470, 58, 37), (39, 'wrench', 5000, 420, 92, 42), (43, 'goggles', 6080, 330, 150, 47)]
        c = Cel(ss=2)
        for f0, kind, wx, wy, s, fl in it:
            a = f - f0
            if a < 1 or kind == 'tophat': continue
            t = a / 24.; yy = wy + 120 * t + (3200 if kind != 'hat' else 900) * t * t * .5; gy = 905
            if yy > gy: b = t; yy = gy - abs(math.sin((a - 6) * 1.1)) * 34 * math.exp(-.4 * max(0, a - 6)); 
            rot = dict(sword=.10, cane=.5 + a * .30, goggles=a * .25, cig=a * .7, wrench=-.4 + a * .36, hat=-.35 + .10 * a)[kind]
            if kind == 'sword': yy = min(yy, 800)
            X = W / 2 + (wx - cxw) * zoom
            if -300 < X < W + 300: self.item(c, kind, X, yy, s * zoom, rot)
        img = over(img, kit.rim(c.layer(), 5, 3, '#eaf0a6', .8, soft=.8))
        if abs(v) > 60:
            n = min(900, abs(v) * .75); k = np.ones((1, int(n) | 1), np.float32) / (int(n) | 1); img = cv2.blur(img, (int(n) | 1, 1)); c = Cel(ss=1); kit.speed_lines(c, 0., 70, 5, (500, 1600), '#0a0d10', f, alpha=.55); kit.speed_lines(c, 0., 40, 3, (500, 1600), '#e6ee9a', f + 50, alpha=.5); img = over(img, c.layer())
        dk = .10 + .06 * i; y, x = grid()[1], grid()[0]; vg = np.clip(np.hypot((x - 960) / 1150, (y - 520) / 700) - .55, 0, 1); img = img * (1 - np.clip(vg * (.5 + dk * 2.2), 0, .95))[..., None]
        if f == 8: img = img * .72 + .28 * rgb('#eaf6b4')
        return img

    # ---------------------------------------------------------------- C: the eye
    def eye(self, f):
        a = f - 46; z = 1.0 + .012 * a + .002 * a * a; q = kit.shake(f // 2, 3.2 if a > 2 else 0, 8); hold = f // 2; c = Cel(ss=2)
        NV, S0, S1, S2, S3 = '#0d1f46', '#f08a7a', '#c44d4c', '#7a2434', '#ffbfa8'; c.fill(np.array([(0, 0), (W, 0), (W, H), (0, H)]), NV); r = np.random.default_rng(3)
        for k in range(150):                                          # scribbled dark
            x, y = r.uniform(-100, W + 100), r.uniform(-60, H + 60); an = -.9 + r.normal(0, .18); ln = r.uniform(160, 520); d = np.array([math.cos(an), math.sin(an)]); c.line(spl([(x, y), np.array([x, y]) + d * ln * .5 + r.normal(0, 14, 2), np.array([x, y]) + d * ln], n=8), r.uniform(3, 12), ('#081430', '#14305e', '#050c20')[k % 3], taper=(.1, .4), seed=k, alpha=.9)
        E = np.array([1130., 470.])
        face = spl([(540, -60), (590, 160), (566, 330), (604, 430), (560, 560), (486, 712), (470, 770), (506, 806), (580, 812), (566, 862), (520, 930), (500, 1010), (560, 1120), (1500, 1120), (1690, 960), (1780, 760), (1800, 480), (1770, 200), (1700, -60)], True, n=8); c.fill(face, S0); c.clip(face)
        c.fill(spl([(1480, -60), (1560, 200), (1500, 420), (1580, 640), (1420, 860), (1500, 1140), (1900, 1140), (1900, -60)], True, n=8), S1); c.fill(spl([(1640, -60), (1690, 240), (1650, 520), (1700, 760), (1560, 1140), (1900, 1140), (1900, -60)], True, n=8), S2)
        c.fill(spl([(500, -60), (700, -60), (640, 150), (700, 300), (640, 400), (612, 470), (560, 430), (500, 300)], True, n=8), INK)     # the far half of the face is gone into black
        c.fill(spl([(300, 900), (560, 880), (760, 960), (1000, 985), (1240, 1010), (1300, 1140), (300, 1140)], True, n=8), INK)
        c.fill(spl([(590, 600), (668, 700), (650, 800), (560, 800), (520, 780), (560, 700)], True, n=7), S1); c.line(spl([(604, 430), (560, 560), (486, 712), (470, 770), (510, 806), (600, 800)], n=8), 9, INK, taper=(.3, .3)); c.line(spl([(650, 716), (676, 760), (652, 806)], n=8), 6, INK)
        hatch(c, spl([(1380, 120), (1560, 160), (1500, 640), (1400, 900), (1320, 860)], True, n=6), 1.05, 13, 3.0, INK, 5, length=90, fade=.4); hatch(c, spl([(560, 160), (760, 130), (700, 330), (620, 420)], True, n=6), 2.2, 13, 3.2, INK, 6, length=80, fade=.3); hatch(c, spl([(700, 780), (1180, 830), (1300, 1000), (760, 960)], True, n=6), .5, 13, 3., INK, 7, length=90, fade=.35); pass
        for k in range(7): y0 = 742 + 13 * k; c.line(spl([(830 + 16 * k, y0 + 14), (960, y0 + 38), (1150, y0 + 34), (1330 - 10 * k, y0 - 4)], n=8), 3.2 - .2 * k, INK if k % 2 else S2, taper=(.4, .5), seed=k)
        dk_ = np.c_[250 + np.cos(KU.A) * 170, 480 + np.sin(KU.A) * 190]
        # eyeball
        R = 285 + 10 * math.exp(-a / 2.) ; th = np.linspace(0, 2 * math.pi, 90, endpoint=False); rr = np.random.default_rng(40 + (hold % 3)); wob = 1 + .010 * np.sin(th * 2 + hold) + .006 * np.sin(th * 3 + hold * 2); eb = E + np.c_[np.cos(th) * R * 1.02, np.sin(th) * R] * wob[:, None]
        c.fill(E + (eb - E) * 1.10, S2); c.fill(eb, '#ffd8c9'); c.clip(eb); c.fill(eb + np.array([-36, 40]), '#ffece0'); c.fill(spl([(E[0] - 320, E[1] - 300), (E[0] + 320, E[1] - 300), (E[0] + 300, E[1] - 150), (E[0] + 40, E[1] - 205), (E[0] - 300, E[1] - 120)], True, n=8), '#f0988a')
        # reflection: an empty grove, a horizon, one paw dissolving
        ro = smooth(49, 53, f) * .75
        if ro > 0:
            c.fill(spl([(E[0] - 300, E[1] + 150), (E[0], E[1] + 128), (E[0] + 300, E[1] + 150), (E[0] + 260, E[1] + 300), (E[0] - 260, E[1] + 300)], True, n=8), '#c0605c', .55 * ro)
            for k in range(6): x = E[0] - 220 + 84 * k + 14 * math.sin(k * 3); c.fill(np.array([(x - 10, E[1] - 190), (x + 8, E[1] - 190), (x + 18, E[1] + 150), (x - 22, E[1] + 150)]), '#d8796e', .35 * ro)
            for p in KU.pads(E[0] + 128, E[1] - 92, 58): c.fill(p, '#a83c46', .60 * ro * (1 - smooth(55, 59, f)))
        pr = float(np.interp(f, [46, 47, 48, 49, 50, 52, 60], [74, 78, 40, 13, 21, 18, 16])); pj = kit.shake(f, 3.5 if a > 3 else 0, 3); pc = E + [-20, 6] + pj; c.fill(pc + np.c_[np.cos(th) * pr, np.sin(th) * pr * 1.06], INK); c.fill(pc + [-pr * .3, -pr * .35] + np.c_[np.cos(th), np.sin(th)] * pr * .22, '#ffffff')
        for k in range(26):                                           # veins of the white
            t0 = k / 26 * 6.283 + .1 * rr.normal(); ln = rr.uniform(26, 90) * (1 if a > 2 else .3); p0 = E + np.array([math.cos(t0), math.sin(t0)]) * R * .99; p1 = E + np.array([math.cos(t0 + .06), math.sin(t0 + .06)]) * (R - ln); c.line(spl([p0, (p0 + p1) / 2 + rr.normal(0, 5, 2), p1], n=6), 3.4, '#b5404a', taper=(.02, .9), seed=k)
        c.unclip()
        for j, (kk, w_) in enumerate(((1.0, 15), (1.022, 7), (1.055, 4), (1.10, 5))): outline(c, E + (eb - E) * kk, w_, INK, 50 + j + hold % 3, press=.7, tremor=1.3, breaks=3 * j)
        c.line(spl([E + [-R * 1.05, -R * .30], E + [-R * .6, -R * .98], E + [0, -R * 1.16], E + [R * .7, -R * .92], E + [R * 1.16, -R * .2]], n=10), 30, INK, taper=(.3, .3), press=.4)
        c.unclip(); ff_ = .30 + .05 * math.sin(f * .9); c.fill(dk_, '#3a1420', .9); c.fill(dk_ * .93 + dk_.mean(0) * .07, '#8a3a44', ff_ + .25); c.fill(np.array([258., 486.]) + pj + np.c_[np.cos(th), np.sin(th)] * pr * .9, INK); outline(c, dk_, 7, '#c44d4c', 61, press=.8, breaks=6, tremor=1.5); c.clip(face)
        # brow pulled up and broken, lid lines
        c.line(spl([(770, 190), (900, 84), (1090, 46), (1300, 92), (1480, 220)], n=10), 44, INK, taper=(.35, .5), press=.5, seed=2); c.line(spl([(820, 250), (980, 150), (1160, 122), (1340, 180)], n=8), 7, INK, taper=(.4, .4)); c.line(spl([(800, 700), (930, 800), (1120, 832), (1310, 780)], n=8), 8, INK, taper=(.4, .5))
        # scar: one line and two stitches
        c.line(spl([(930, 884), (1080, 908), (1240, 892), (1350, 856)], n=8), 11, '#7a2434', taper=(.3, .3)); [c.line(np.array([(x, y - 34), (x + 6, y), (x + 4, y + 36)]), 9, '#7a2434', taper=(.2, .2)) for x, y in ((1060, 900), (1200, 898))]
        c.unclip(); outline(c, face, 11, INK, 4, press=.6)
        # sweat
        for k, (sx, sy, sz, sp) in enumerate(((700, 330, 34, 1.4), (1560, 640, 40, 2.2), (1470, 300, 26, 1.0), (860, 880, 24, 2.6), (1640, 260, 30, 1.8))):
            yy = sy + sp * a * 1.6 + 1.5 * math.sin(f + k); d = spl([(sx, yy - sz * 1.7), (sx + sz * .62, yy), (sx + sz * .3, yy + sz * .62), (sx - sz * .3, yy + sz * .62), (sx - sz * .62, yy)], True, n=6); c.fill(d, '#ffe9de'); c.clip(d); c.fill(d + [sz * .28, sz * .2], '#f4a796'); c.unclip(); outline(c, d, 5, INK, k); c.fill(np.c_[sx - sz * .18 + np.cos(th) * sz * .12, yy - sz * .1 + np.sin(th) * sz * .2], '#ffffff')
        k = smooth(52, 56, f)                                         # water gathers on the lid
        if k > 0: w_ = spl([(920, 712), (1010, 768 - 14 * k), (1130, 790 - 26 * k), (1250, 764 - 16 * k), (1330, 706), (1290, 790), (1130, 838 + 12 * k), (980, 806)], True, n=7); c.fill(w_, '#eaf4ff', .92); outline(c, w_, 5, INK, 8); c.fill(spl([(1010, 786), (1100, 800), (1180, 796), (1100, 816)], True, n=6), '#ffffff')
        # hair: black clumps across the brow and the dead side
        for j, (a_, t_, b_) in enumerate((((180, -80), (640, 560), (520, -80)), ((420, -80), (860, 330), (760, -80)), ((700, -80), (1040, 250), (960, -80)), ((930, -80), (1380, 200), (1240, -80)), ((1200, -80), (1620, 300), (1500, -80)), ((1440, -80), (1830, 470), (1760, -80)), ((1620, -80), (1960, 700), (2000, -80)), ((60, -80), (430, 820), (300, -80)), ((-60, 100), (250, 980), (160, 60)))):
            sw = 6 * math.sin(f * .35 + j); s_ = kit.spike(a_, (t_[0] + sw, t_[1] + abs(sw)), b_, .10, .05 * (1 if j % 2 else -1), 16); c.fill(s_, INK); m = len(s_) // 2; c.line(s_[int(m * .15):int(m * .8)] + [22, 0], 5, S1 if j % 3 else '#1d3a70', taper=(.2, .7), seed=j)
        img = c.layer()[..., :3]; sh = cv2.GaussianBlur(img, (0, 0), 30); x, y = grid(); g = np.exp(-(((x - 1240) / 900) ** 2 + ((y - 380) / 620) ** 2)); img = img * (.80 + .28 * g[..., None]); img = img + (np.exp(-(((x - E[0] + 110) / 130) ** 2 + ((y - E[1] + 130) / 60) ** 2)) * .10)[..., None]
        img = comp.cam2d(img, 960 + q[0] + 90 * (1 - 1 / z), 540 + q[1] - 40 * (1 - 1 / z), z, .02 * math.sin(a * .2)); d = smooth(50, 60, f) * .55; vg = np.clip(np.hypot((x - 1080) / 1050, (y - 500) / 640) - (.80 - .25 * d), 0, 1); img = img * (1 - np.clip(vg * 2.2, 0, 1) * (.35 + d))[..., None]
        if f == 46: img = kit.impact(img, 'bw', .2, .5, paper='#f4c0b0')
        return img

    def frame(self, f):
        f = int(f); return (self.palm(f * 9 / 7.) if f < 8 else self.pov(f) if f < 46 else self.eye(f)).astype(f32)


def over_layer(a, b): return a * (1 - b[..., 3:4]) + b
