"""c6 Robin, violet: a page turns and the card becomes the world; arms cross; hundreds of arms bloom into a great tree; giant hands clap on 25."""
from chars.lkit import *
from chars import robin as R_
from op import clouds
from op.comp import over, radial, blur, cam2d
from op.kit import shake, flare, focus_lines, smear
from op.fx_air import petals, motes
from op.paint import ramp, fbm, Canvas, arcs, repaint
import os

CX, CY = 960., 520.
SEAT = np.array([960., 712.]); S0 = 66.
PET = dict(lit='#ffe9f6', mid='#f3a9d2', dark='#b06ab4', ink='#5c2a66')


def glyph(c, x, y, s, seed, sk=0.):
    """invented ponegliff sign in a square cell: recess + lit lower edge"""
    r = rs(seed, 2); Q = lambda P: np.array([(x + (u + v * sk) * s, y + v * s) for u, v in P]); k = int(r.integers(7)); L = []
    a = np.linspace(0, TAU, 22)
    if k in (0, 4): L.append(np.c_[.5 + .32 * np.cos(a), .5 + .32 * np.sin(a)])
    if k in (1, 5): L += [np.array([(.2, .2), (.8, .2), (.8, .8), (.2, .8), (.2, .2)])]
    if k in (2, 6): L += [np.array([(.5, .12), (.88, .84), (.12, .84), (.5, .12)])]
    for _ in range(int(r.integers(2, 4))): p = r.integers(0, 4, 4) / 3. * .72 + .14; L.append(np.array([(p[0], p[1]), (p[0], p[3]), (p[2], p[3])]) if r.random() < .6 else np.array([(p[0], p[1]), (p[2], p[3])]))
    if r.random() < .5: L.append(np.c_[.5 + .10 * np.cos(a), .5 + .10 * np.sin(a)])
    for q in L: c.line(Q(q) + [1.6, 2.2], s * .13, '#7f7fc4', taper=(0, 0), press=.2)
    for q in L: c.line(Q(q), s * .13, '#0f1030', taper=(0, 0), press=.2)


class Shot:
    id = 'c6'; n = 30; finish = dict(bloom=.36, vig=.15)
    EVENTS = [(0, 'cloth', dict(note='quiet, wind, one page rustle')), (4, 'page', dict(size=.7, note='page turns and takes the card with it')), (9, 'book', dict(size=.5, note='book claps shut')), (11, 'cloth', dict(size=.6, note='arms cross')), (12, 'bloom', dict(size=.6, note='first arms: soft flower pops, rising run f12-21')), (16, 'bloom', dict(size=.9)), (20, 'bloom', dict(size=1.0, note='tree complete, glass shimmer')), (22, 'creak', dict(size=.5, note='giant hands draw back')), (25, 'impact', dict(size=1.0, note='giant hands clap: DON')), (26, 'gust', dict(size=.8, note='petal shock wipes to cyan'))]

    def setup(self):
        self.card = card(H, W, '#7d4fb4', '#4f2c86', '#a67ad8', 6, -.3); self.next = card(H, W, '#29bccd', '#12808f', '#74e0e4', 7, .3)
        if os.path.exists('cache/c6_sky.npy'): self.sky = np.array(np.load('cache/c6_sky.npy', mmap_mode='r'), f32); self.cube = np.array(np.load('cache/c6_cube.npy', mmap_mode='r'), f32)
        else: self.sky = self.paint_sky(); self.cube = self.paint_cube(); np.save('cache/c6_sky.npy', self.sky.astype(np.float16)); np.save('cache/c6_cube.npy', self.cube.astype(np.float16))
        st = R_.style('violet'); pal = dict(skin='#ffbfb4', shade='#b85a9c', line='#54204e'); K = 46; self.K = K; self.arms = []; BLD = HD.build(len=.96, wid=1.10, boxy=.34, tip=.25, knuckle=.85, palm=1.06, wrist=1.25, meat=1.1)
        for j, (po, sd, e) in enumerate((('open', 'R', (180, 0, 4)), ('splay', 'L', (170, 10, -6)), ('relaxed', 'R', (20, 8, 4)), ('flat', 'L', (0, 0, 0)), ('reach', 'R', (150, 14, 8)), ('palm_up', 'L', (186, -18, 4)), ('open', 'L', (0, 6, -8)), ('slap', 'R', (120, 0, 0)), ('rest', 'L', (14, 24, 6)), ('splay', 'R', (195, -6, 8)))):
            a, _ = hand_sprite(po, sd, HD.euler(*e), K, BLD, pal, w=2.0, res=230, rimc='#ffe2f6', rimd=(.1 if j % 2 else -.1, .99), rw=4, marks=.5, size=K * 6); self.arms.append(mips(a, 5))
        self.bud, _ = hand_sprite('fist', 'R', HD.euler(170, 0, 0), K, BLD, pal, w=2.0, res=230, rimc='#ffe2f6', rimd=(0, 1), rw=4, marks=.4, size=K * 6); self.bud = mips(self.bud, 5)
        G = 178; self.G = G; pg = dict(skin='#f3c9b4', shade='#8e568a', line='#1a0e1f'); sty = dict(nails='short')
        self.gl, _ = hand_sprite('slap', 'L', HD.euler(62, 10, -16), G, HD.FEMALE, pg, light=(.1, .75, .6), w=5.2, res=520, rimc='#ffe2f6', rimd=(-.65, .75), rw=10, size=G * 5.6, style=sty); self.gr, _ = hand_sprite('slap', 'R', HD.euler(-62, 10, 16), G, HD.FEMALE, pg, light=(-.1, .75, .6), w=5.2, res=520, rimc='#ffe2f6', rimd=(.65, .75), rw=10, size=G * 5.6, style=sty)
        self.fl, _ = hand_sprite('flat', 'L', HD.euler(8, 0, 0), 46, HD.FEMALE, pal, w=2.2, res=260, rimc='#ffe2f6', rimd=(.78, .62), rw=3, size=46 * 5.4, style=sty); self.fr_, _ = hand_sprite('flat', 'R', HD.euler(-8, 0, 0), 46, HD.FEMALE, pal, w=2.2, res=260, rimc='#ffe2f6', rimd=(.78, .62), rw=3, size=46 * 5.4, style=sty)
        fc = Cel(160, 160); a = np.linspace(0, TAU, 40)
        for k in range(5): th = k * TAU / 5 - 1.57; q = np.array([80 + math.cos(th) * 40, 80 + math.sin(th) * 40]); n_ = np.array([-math.sin(th), math.cos(th)]); d_ = np.array([math.cos(th), math.sin(th)]); P = spl(np.array([[80, 80], q + n_ * 30 - d_ * 8, q + n_ * 22 + d_ * 24, q + d_ * 20, q - n_ * 22 + d_ * 24, q - n_ * 30 - d_ * 8]), True, corners=(0,), n=6); fc.fill(P, '#ffb7d6'); fc.clip(P); fc.fill(spl(np.array([q + n_ * 40 + d_ * 4, q + d_ * 10, q - n_ * 40 + d_ * 4, q - n_ * 40 + d_ * 50, q + n_ * 40 + d_ * 50]), True, n=4), '#ffe3ee'); fc.unclip(); stroke(fc, P, 2.4, '#7a2a64'); fc.line(np.array([[80, 80], q + d_ * 6]), 2.2, '#c9508c', taper=(.1, .6))
        fc.fill(np.c_[80 + np.cos(a) * 12, 80 + np.sin(a) * 12], '#ffe08a'); stroke(fc, np.c_[80 + np.cos(a) * 12, 80 + np.sin(a) * 12], 2, '#7a2a64'); self.flower = mips(fc.rgba(), 4)
        self.tree()

    # ---------------------------------------------------------------- plates
    def paint_sky(self):
        h, w = 1240, 2200; y = np.linspace(0, 1, h)[:, None].repeat(w, 1); x = np.linspace(0, 1, w)[None].repeat(h, 0); g = ramp(y, [(0, '#2a1c5c'), (.35, '#5c3c98'), (.62, '#b06cb8'), (.80, '#f0a8b8'), (1, '#7a4a8c')]); sk = clouds.Sky(h, w, g, light=(.0, -.5, -.3), seed=4)
        for i, (bx, by, bw, bh, sd) in enumerate(((250, 980, 900, 420, 1), (1950, 980, 900, 450, 2), (700, 1010, 800, 240, 3), (1500, 1010, 800, 250, 4), (1100, 1060, 1500, 170, 5))): sk.bank(bx, by, bw, bh, ('#ffd9dc', '#d894c4', '#8a5aaa', '#4c327e'), '#c88ac0', sd, rim='#fff0e4', rim_amt=.5)
        im = sk.paint(3).astype(f32); gx, gy = grid(h, w); d = np.sqrt((gx - 1100) ** 2 + (gy - 520) ** 2); disc = np.clip((372 - d) / 1.6, 0, 1); n = fbm(h, w, (90, 30, 10), None, 3); im = im * (1 - disc[..., None]) + ramp(np.clip((gy - 150) / 750 + n * .02, 0, 1), [(0, '#fff3ea'), (.55, '#ffd3dc'), (1, '#f5a4c4')]) * disc[..., None]
        im = im + np.exp(-np.maximum(d - 372, 0) / 170)[..., None] * rgb('#ff9ecb') * .32
        # far broken columns on the horizon
        c = Cel(w, h); r = rs(12)
        for i in range(17):
            bx = r.uniform(0, w); 
            if abs(bx - 1100) < 330: continue
            hh = r.uniform(60, 250); ww = hh * .16 + 8; by = 1010 + r.uniform(-8, 30); P = np.array([(bx - ww * .6, by), (bx - ww * .5, by - hh), (bx - ww * .1, by - hh * r.uniform(.92, 1.06)), (bx + ww * .5, by - hh * r.uniform(.8, .98)), (bx + ww * .6, by)]); c.fill(P, '#6a4796'); c.fill(P * [.35, 1] + [bx * .65 + (ww * .4 if bx < 1100 else -ww * .4), 0], '#c58cc4', .8)
        L = c.layer(h, w); im = im * (1 - L[..., 3:4]) + L[..., :3]
        # ground sea of broken stones
        c = Cel(w, h); r = rs(5)
        for i in range(90): q = r.random() ** .6; y0 = 1010 + q * 220; s = 14 + q * 90; bx = r.uniform(0, w); P = spl(np.array([(bx - s, y0), (bx - s * .7, y0 - s * r.uniform(.3, .7)), (bx + s * .2, y0 - s * r.uniform(.4, .8)), (bx + s, y0 - s * .2), (bx + s * 1.1, y0 + s * .2), (bx - s * .9, y0 + s * .2)]), True, n=3); c.fill(P, mix(rgb('#7a52a0'), rgb('#2c1f58'), q)); c.fill(np.vstack([P[4:11], P[4:11][::-1] + [0, 5 + s * .05]]), mix(rgb('#f0b4d0'), rgb('#9a6cc0'), q), .8)
        L = c.layer(h, w); return (im * (1 - L[..., 3:4]) + L[..., :3]).astype(f32)

    def paint_cube(self):
        """front face of the great stone + its top, 1920x1080 at zoom 1 (painted once, bigger than the frame below)"""
        h, w = 1080, 1920; F = np.array([(470, 742), (1450, 742), (1490, 1180), (430, 1180)], float); T = np.array([(470, 742), (1450, 742), (1330, 694), (590, 694)], float); c = Cel(w, h); c.fill(F, '#34386e'); c.fill(T, '#8e78c0'); c.clip(F)
        c.fill(np.array([(960, 742), (1450, 742), (1490, 1180), (1080, 1180)]), '#2a2c5c'); r = rs(4)
        for i in range(60): x, y = r.uniform(440, 1480), r.uniform(750, 1080); s = r.uniform(10, 60); c.fill(spl(np.array([(x - s, y), (x, y - s * .3), (x + s, y + s * .05), (x + s * .2, y + s * .34)]), True, n=4), mix(rgb('#23244e'), rgb('#4a4e8e'), r.random()), .55)
        for j in range(5):
            y = 768 + j * 66; k = 1 + j * .012
            for i in range(15): x = 960 + (i - 7.5) * 63 * k + 5; glyph(c, x, y, 50, j * 31 + i)
            c.line(np.array([(460, y - 9), (1470, y - 9)]), 2.2, '#14143a', alpha=.7, taper=(0, 0)); c.line(np.array([(460, y - 6), (1470, y - 6)]), 1.4, '#6f70b8', alpha=.6, taper=(0, 0))
        for i in range(9): x, y = r.uniform(470, 1450), r.uniform(745, 900); P = [np.array([x, y])]; 
        for x0 in (610, 1290, 1400):
            P = [np.array([x0, 742.])]
            for _ in range(7): P.append(P[-1] + [r.normal(0, 22), r.uniform(18, 46)])
            c.line(np.array(P) + [2, 1], 2.5, '#7a7cc4', taper=(.1, .6)); c.line(np.array(P), 4., '#0d0d2a', taper=(.1, .6))
        c.unclip(); outline(c, T, 3., INK); outline(c, F, 5., INK, 3); c.line(np.array([(474, 745), (1446, 745)]), 7, '#ffd9ee', taper=(.02, .02), press=.6); 
        for x0, s in ((470, 1), (1450, -1)): c.fill(np.array([(x0 - 8 * s, 736), (x0 + 60 * s, 738), (x0 + 22 * s, 770), (x0 - 4 * s, 800)]), '#5a4a98'); 
        for i in range(26): x = r.uniform(480, 1440); c.fill(spl(np.array([(x - 22, 746), (x - 6, 735), (x + 16, 741), (x + 24, 748), (x + 8, 760 + r.uniform(0, 22)), (x - 10, 754)]), True, n=3), ('#6a8a4e', '#4c6a44', '#87a45c')[i % 3], .95)
        return c.layer()

    # ---------------------------------------------------------------- the tree
    def tree(self):
        r = np.random.default_rng(17); A = []                          # (depth, x, y, ang, scale, birth, sprite, flip, tone)
        base = np.array([960., 730.]); up = -math.pi / 2
        for row in range(9):                                           # trunk: rows of arms climbing over one another
            y = 735 - row * 34; wd = 118 - row * 5.5 + 46 * math.exp(-row / 1.3)
            for k in range(8):
                u = (k + .5) / 8 * 2 - 1 + r.normal(0, .06); A.append((.30 + .02 * row + .10 * (1 - abs(u)), 960 + u * wd, y + r.normal(0, 8) + abs(u) * 14, up + u * (.30 - row * .02) + r.normal(0, .10), r.uniform(.60, .78), 10.6 + row * .55 + r.uniform(0, .8), int(r.integers(10)), r.random() < .5, .28 * abs(u) + .04))
        def limb(p, ang, L, w, n, t0, lv, dep, curve):
            q = np.array(p, float); st = L / n; a = ang; dt = .62 if lv == 0 else .55
            for i in range(n):
                a += curve / n + r.normal(0, .05); f_ = i / float(n); t = t0 + i * dt
                for k in range(3 if lv == 0 and f_ < .6 else 2 if lv == 0 or f_ < .5 else 1): o = (k - .5) * w * (1 - f_ * .6); nrm = np.array([-math.sin(a), math.cos(a)]); A.append((dep + .01 * i, *(q + nrm * o * r.uniform(.6, 1.2)), a + r.normal(0, .12), (.62 - .14 * lv) * (1 - .25 * f_) * r.uniform(.85, 1.1), t + r.uniform(0, .5), int(r.integers(10)), r.random() < .5, .30 * (lv > 0) + (.25 if dep < .3 else 0)))
                q = q + np.array([math.cos(a), math.sin(a)]) * st
                if lv == 0 and i >= 1:
                    for sgn in ((-1, 1) if i % 2 else (1,)):
                        if r.random() < .70: limb(q, a + sgn * r.uniform(.55, 1.0), L * r.uniform(.28, .44), 16, 3, t + .5, 1, dep + r.choice([-.06, .12]), sgn * r.uniform(-.2, .5))
                if lv == 1 or (lv == 0 and i == n - 1): ros(q, a, t + .4, dep + .05)
        def ros(q, a, t, dep):
            n = int(r.integers(5, 8)); sc = r.uniform(.34, .56)
            for _ in range(int(r.integers(2, 5))): A.append((dep + .015, *(q + r.normal(0, 22, 2)), r.uniform(0, 6), r.uniform(.22, .46), t + r.uniform(.8, 2.2), -1, False, .40 if dep < .25 else .0))
            for k in range(n): A.append((dep + .02 + r.uniform(0, .03), *(q + r.normal(0, 6, 2)), a + (k / float(n) * 2 - 1) * r.uniform(1.6, 2.6) + r.normal(0, .12), sc * r.uniform(.8, 1.2), t + r.uniform(.2, 1.2), int(r.choice([0, 1, 4, 5, 6, 9])), r.random() < .5, .45 if dep < .25 else .05))
        for k, (ang, L, cv, dep) in enumerate(((-2.95, 620, .30, .34), (-2.62, 640, -.10, .22), (-2.28, 520, .25, .36), (-1.92, 430, -.15, .2), (-1.57, 400, .05, .33), (-1.20, 430, .15, .2), (-.86, 520, -.25, .36), (-.52, 640, .10, .22), (-.19, 620, -.30, .34))): limb((960 + math.cos(ang) * 46, 452 + math.sin(ang) * 20 + (18 if abs(ang + 1.57) > 1 else 0)), ang, L, 34, 7, 13.2 + .12 * abs(k - 4), 0, dep, cv)
        for k in range(22): t = r.uniform(-1, 1); ros(np.array([960 + t * 800, 330 - 210 * math.cos(t * 1.4) + r.normal(0, 60)]), -1.57 + t * .9, 16.5 + abs(t) * 2.2, .10)
        for k in range(16): u = r.uniform(-1, 1); A.append((.8, 960 + u * 430 + 60 * np.sign(u), 742 + r.normal(0, 5), -1.57 + u * .6 + r.normal(0, .2), r.uniform(.30, .55), 11 + abs(u) * 3 + r.uniform(0, 1), int(r.integers(10)), r.random() < .5, 0.))
        B = []                                                         # inside the moon only the limbs and a few open hands stay: gaps let the disc through
        for e in A:
            dep, x, y, a, sc, t0, sp, fl, tone = e; rr = math.hypot(x - 960, y - 443)
            if rr < 366 and dep < .75 and y < 600 and (sp < 0 and r.random() < .72 or sp >= 0 and sc < .60 and r.random() < .66): continue
            B.append(e)
        B.sort(key=lambda q: q[0]); self.A = B

    def draw_tree(self, dst, f, Z, front=False, shk=(0, 0), oy=0.):
        K = self.K; an = (K * 3., K * 3. + 2.30 * K); t = f + .0; n = 0
        for dep, x, y, a, sc, t0, sp, fl, tone in self.A:
            if (dep >= .75) != front: continue
            u = (t - t0) / 4.
            if u <= 0: continue
            g = float(back(min(u, 1), 2.4)) if u < 1 else 1 + .02 * math.sin(f * .5 + x); sway = (.05 * math.sin(f * .45 + x * .01) + (.5 * (1 - min(u, 1)))) * (1 if fl else -1); X = CX + (x - CX) * Z + shk[0] * (.5 + dep); Y = CY + (y - CY) * Z + shk[1] * (.5 + dep) + oy
            if sp < 0: putm(dst, self.flower, X, Y, sc * g * Z, a + .06 * math.sin(f * .5 + x), False, 1., '#7a4aa4', tone); n += 1; continue
            ms = self.bud if u < .30 else self.arms[sp if u > .6 else 2]
            rr = math.hypot(x - 960, y - 443); ins = rr < 378 and dep < .75; tc = '#34174f' if ins else '#5a3a9c'; tone = max(tone, .52) if ins else tone; putm(dst, ms, X, Y, sc * g * Z, a + math.pi / 2 + sway, fl, 1., tc, tone, an); n += 1
        return n

    # ---------------------------------------------------------------- Robin on the stone
    def robin(self, f, Z, shk):
        s = S0 * Z; head = lambda q: np.array([CX + shk[0], CY + (SEAT[1] - 3.55 * S0 - CY) * Z + shk[1] + float(keys(q, [(0, 4.), (8, 0.), (10, 5.), (12, -5.), (15, 0.), (24, 2.), (25, 7.), (28, 0.)])) * Z * .6]); p = head(f); lw = float(np.clip(s * .0215, 2.0, 5.2)); d = f // 2 * 2; cross = float(keys(f, [(9, 0.), (11, 0.12), (13, 1.06, out3), (15, 1.)])); t = f / 24.
        yaw = float(keys(f, [(0, -.24), (6, -.20), (9, .10), (12, .0), (30, .02)])); pitch = float(keys(f, [(0, -.20), (5, -.16), (9, .06), (11, -.10), (13, .05, out3), (24, .0), (25, .08), (30, .04)])); roll = float(keys(f, [(0, -.05), (9, .04), (11, -.03), (14, .0), (30, 0)]))
        if f < 5: ex = dict(o=.42, M='calm', gaze=(-.2, -.9), bi=.01)
        elif f in (5, 6): ex = dict(o=.14, M='calm')
        elif f < 10: ex = dict(o=.84, M='smile', gaze=(.1, .05), sq=.08)
        elif f < 13: ex = dict(o=0., shut='soft', M='calm')
        else: ex = dict(o=.86, M='smile', gaze=(0, 0), bi=-.02, bo=.03, ir=1.0, sq=.1)
        st = R_.style('violet'); cl = st['cloth']; Q0 = lambda P: loc(p[0], p[1], s, roll * .3, P)
        def body(c, F, s_):
            fl = math.sin(t * 5.2) * .06; sk = Q0(spl([(-1.0, 2.9), (-1.42, 3.55), (-1.62 - fl, 4.6), (-1.75 + fl, 5.7), (-1.2 - fl, 5.95), (-.8, 5.4), (-.62, 4.3), (0, 3.9), (.62, 4.3), (.8, 5.5), (1.25 + fl, 6.0), (1.8 - fl, 5.6), (1.62 + fl, 4.6), (1.42, 3.55), (1.0, 2.9)], True, n=7)); c.fill(sk, cl['base']); c.clip(sk); c.fill(sk + [s * .5, s * .1], cl['shade']); c.unclip(); outline(c, sk, lw * 1.1, INK)
            for x in (-1.35, -1.05, 1.1, 1.4): fold(c, Q0([(x * .8, 3.6), (x, 4.5), (x * 1.08, 5.4)]), lw * .6, cl['line'])
            for kx, sg in ((-.46, 1), (.46, -1)):                       # shins, ankles crossed
                L = Q0(spl([(kx - .32, 3.62), (kx + .30, 3.58), (kx + .34 * .9 + sg * .12, 4.3), (sg * .40, 5.55), (sg * .10, 5.65), (kx - .30 + sg * .05, 4.4)], True, n=7)); c.fill(L, '#2a2648'); c.clip(L); c.fill(L + [s * .18, 0], '#17142c'); c.unclip(); outline(c, L, lw, INK, int(kx * 10) + 8); fold(c, Q0([(kx - .2, 3.75), (kx + .05, 3.85), (kx + .25, 3.72)]), lw * .6, '#0b0a18')
                B = Q0(spl([(sg * .42, 5.40), (sg * .06, 5.48), (sg * .10, 5.85), (sg * .34, 6.20), (sg * .72, 6.28), (sg * .74, 6.10), (sg * .46, 5.86)], True, n=6)); c.fill(B, '#5a2f4e'); c.clip(B); c.fill(B + [s * .10, s * .14], '#341a30'); c.unclip(); outline(c, B, lw, INK, 4)
            lap = Q0(spl([(-1.02, 2.8), (-1.02, 3.35), (-.80, 3.86), (-.20, 3.92), (0, 3.78), (.20, 3.92), (.80, 3.86), (1.02, 3.35), (1.02, 2.8)], True, n=6)); c.fill(lap, '#2a2648'); c.clip(lap); c.fill(Q0(spl([(.1, 2.6), (.2, 3.3), (.45, 4.1), (1.4, 4.1), (1.4, 2.6)], True, n=5)), '#17142c'); c.unclip(); outline(c, lap, lw, INK, 3)
            bust(c, p[0], p[1], s, roll * .3, s_, 'jacket', 0., lw, -1, length=3.05, f=f, breath=math.sin(f * .3), torso=True)
            def sleeve(P, w0, w1, dark): P = resample(spl(Q0(P), n=10), 3.); n = len(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; w = (lerp(w0, w1, np.linspace(0, 1, n)) * (.45 + .55 * sm(np.linspace(0, 1, n) / .22)))[:, None] * s * .5 * (1 + .06 * np.sin(np.linspace(0, 9, n)))[:, None]; A, B = P + N * w, P - N * w; q = np.vstack([A, B[::-1]]); c.fill(q, cl['base'] if not dark else lerp(rgb(cl['base']), rgb(cl['shade']), .5)); lo = A if A[:, 1].mean() > B[:, 1].mean() else B; hi = B if lo is A else A; c.fill(np.vstack([lo, (lo + (hi - lo) * .42)[::-1]]), cl['shade']); c.line(A, lw, INK, taper=(.02, .02)); c.line(B, lw, INK, taper=(.02, .02)); i = int(n * .45); fold(c, [A[i], P[i + 2], P[min(i + 6, n - 1)]], lw * .55, cl['line']); e = q[[n - 1, n]]; c.line(e, lw, INK, taper=(0, 0)); return P[-1]
            if cross < .5:
                k = cross * 2; bk = float(keys(f, [(0, 0), (7, 0), (9, 1.)])); wy = 2.58 - .25 * k
                for sg in (-1, 1): sleeve([(sg * .92, 1.50), (sg * 1.18, 1.95), (sg * 1.26, 2.72), (sg * (.62 - .1 * bk), wy + .10)], .56, .40, sg > 0)
                op_ = .78 * (1 - .72 * bk); C_ = Q0([(-op_, 1.92 + .05 * bk), (0, 2.12), (op_, 1.92 + .05 * bk), (op_, 2.86), (0, 3.08), (-op_, 2.86)]); pg = Q0([(-op_ * .95, 1.86), (-op_ * .4, 1.93), (0, 2.06), (op_ * .4, 1.93), (op_ * .95, 1.86), (op_ * .95, 2.2), (-op_ * .95, 2.2)]); c.fill(pg, '#f6ecd8'); outline(c, pg, lw * .7, INK); c.fill(C_, '#2d5a4e'); c.fill(np.array([C_[1], C_[2], C_[3], C_[4]]), '#1d4038'); outline(c, C_, lw, INK, 2); c.line(C_[[1, 4]], lw * 1.4, '#14302a', taper=(0, 0))
                for q in ((C_[0] * .86 + C_[1] * .14, C_[5] * .86 + C_[4] * .14), (C_[2] * .86 + C_[1] * .14, C_[3] * .86 + C_[4] * .14)): c.line(np.array(q), lw * .7, '#d9b75a', taper=(0, 0))
                for sg in (-1, 1):
                    e = np.array([sg * op_, 2.62])
                    for k2 in range(4): y = e[1] - .30 + k2 * .115; Fg = Q0(spl([(e[0] + sg * .10, y - .05), (e[0] - sg * .22, y - .045 + .01 * k2), (e[0] - sg * .30, y), (e[0] - sg * .22, y + .05), (e[0] + sg * .10, y + .06)], True, n=5)); c.fill(Fg, st['skin']); c.clip(Fg); c.fill(Fg + [0, s * .06], st['shade']); c.unclip(); outline(c, Fg, lw * .7, INK, k2)
                    Pm = Q0(spl([(e[0] + sg * .02, 2.22), (e[0] + sg * .20, 2.30), (e[0] + sg * .26, 2.60), (e[0] + sg * .10, 2.86), (e[0] + sg * .0, 2.72)], True, n=6)); c.fill(Pm, st['shade'] if sg > 0 else st['skin']); outline(c, Pm, lw * .8, INK)
            else:
                k = np.clip((cross - .5) * 2, 0, 1.12)
                for sg in (1, -1):
                    w_ = lerp(np.array([sg * .50, 2.62]), np.array([-sg * .66, 1.78]), k); sleeve([(sg * .92, 1.50), (sg * 1.16, 1.95), (sg * 1.24, 2.66), (sg * .62, 2.60 - .12 * k), tuple(w_)], .56, .40, sg > 0); self._hands.append((Q0([tuple(w_)])[0], -sg))
        self._hands = []; D = R_.draw(p[0], p[1], s, yaw, pitch, roll, None, 'violet', d, wind=(.35 + .2 * math.sin(d * .4), float(keys(f, [(0, .18), (12, .25), (16, .75), (25, .95), (30, 1.)]))), ex=ex, body=body, posfn=head, sway=(s * .10, 0))
        L = lay(D['back'], D['body']); Lh = np.zeros((H, W, 4), f32)
        for q, sg in self._hands: put(Lh, self.fl if sg < 0 else self.fr_, q[0], q[1], s * .47 / 46., sg * -.30)
        out = stack(L, Lh, D['face'], D['front']); return para(out, (p[0], p[1] - s * 1.2), (p[0], p[1] + s * 5), [(0, '#fff6fb'), (.4, '#f2dcf0'), (1, '#8a78c0')])

    def frame(self, f):
        t = f / 24.; Z = float(keys(f, [(0, 2.95), (6, 2.80), (9, 2.45), (15, 1.10, out3), (22, 1.0), (25, .985), (30, 1.03)])); hit = f >= 25; sh = shake(f, 15, 5, 2.4, 25) if hit else np.zeros(2); oy = 128. * float(np.clip(1 - (Z - 1) / 1.9, 0, 1)); so = sh + [0, oy]
        Zs = 1 + (Z - 1) * .22; bg = cam2d(self.sky, 1100 + (f - 15) * 1.2, 560 - oy * .55, 1.02 * Zs, 0.).astype(f32); bg = bg + radial(960, 470, 1200, 760, 1.6)[..., None] * rgb('#ff9ad0') * .10
        bg = bg * (1 - .10 * np.clip((f - 12) / 10., 0, 1)); bg = over(bg, petals(f, 60, 12, PET, wind=(90., 40.), size=(5, 14), near=0))
        T = np.zeros((H, W, 4), f32)
        if f > 10:
            self.draw_tree(T, f, Z, False, sh, oy); T[..., :3] *= np.array([1.0, .90, 1.0], f32); T = lit(T, (0, 1.), '#fff0f4', 1.5, 4, .9, .5); gm = np.clip((430 - np.hypot(grid()[0] - 960, grid()[1] - 571)) / 60., 0, 1); T = lit(lit(T, (.97, .2), '#ffd9ea', 1., 3.5, .95, .5, where=gm), (-.97, .2), '#ffd9ea', 1., 3.5, .95, .5, where=gm); g = radial(960 + sh[0], CY + oy + (430 - CY) * Z, 700 * Z, 520 * Z, 1.4); T[..., :3] *= (1 - .18 * (1 - g))[..., None]; bg = bg + blur(T[..., 3], 26)[..., None] * rgb('#ff8ecb') * .10; bg = over(bg, T)
            bg = over(bg, petals(f - 11, 150, 21, PET, wind=(60., -20.), size=(7, 20), origin=(960, 420 + oy), burst=1500., near=0))
        # giant hands
        if f >= 17:
            k = float(keys(f, [(17, 0.), (21, .72, out3), (23, .58), (24, .80, lambda q: q), (25, 1.0)])); rise = float(keys(f, [(17, 620.), (21, 20., out3), (24, 0.), (25, 0)])); L = np.zeros((H, W, 4), f32); Lz = []; gap = float(keys(f, [(17, 660.), (20, 620.), (23, 760., out3), (24, 360., lambda q: q), (25, 96., lambda q: q), (26, 86.), (28, 96.), (30, 100.)])); y = 560 + rise + (14 if f == 25 else 0); tilt = float(keys(f, [(17, .34), (23, .42), (24, .16), (25, 0.), (30, .02)]))
            for sgn, sp in ((-1, self.gl), (1, self.gr)):
                L = np.zeros((H, W, 4), f32); rot = sgn * tilt; w0 = np.array([960 + sgn * gap + sh[0], y + sh[1]]); put(L, sp, w0[0], w0[1], 1., rot)
                x = 960 + sgn * gap + sh[0]; rot = sgn * tilt; w0 = np.array([x, y + sh[1]]); dn = np.array([sgn * .30 * math.cos(rot) - .95 * math.sin(rot), sgn * .30 * math.sin(rot) + .95 * math.cos(rot)])
                for m in range(34): r = rs(m, 8 + sgn); q = w0 + dn * (250 + m * 13) + [r.normal(0, 56), 0]; putm(L, self.arms[int(r.integers(10))], q[0], q[1], r.uniform(.55, .95), math.atan2(-dn[0], dn[1]) * -1 * 0 + math.atan2(dn[0], -dn[1]) + math.pi + r.normal(0, .30) + (.06 * math.sin(f * .7 + m)), r.random() < .5, 1., '#6a3f9c', .22, (self.K * 3., self.K * 5.3))
                if f == 24: L = lay(smear(L, 420., 0, 31) * 1.0, cv2.warpAffine(L, np.float32([[1, 0, -sgn * 90], [0, 1, 0]]), (W, H)) * .85)
                Lz.append(L)
            L = lay(Lz[0], Lz[1])
            L = para(L, (960, 100), (960, 1000), [(0, '#ffffff'), (.5, '#f6d8ee'), (1, '#7a5aa8')]); bg = over(bg, L)
        M = np.float32([[Z, 0, CX * (1 - Z) + sh[0]], [0, Z, CY * (1 - Z) + so[1]]]); cube = cv2.warpAffine(self.cube, M, (W, H), flags=cv2.INTER_LINEAR); cs = cube.copy(); cs[..., :3] *= (1.05 - .25 * np.clip((np.arange(H)[:, None, None] - 742 - oy) / 340., 0, 1)); world = over(bg, cs)
        if f > 10: Fr = np.zeros((H, W, 4), f32); self.draw_tree(Fr, f, Z, True, sh, oy); world = over(world, lit(Fr, (0, 1.), '#fff0f4', 1.5, 3, .8, .4))
        rb = self.robin(f, Z, so)
        # ground shadow on the stone
        world = over(world, rb)
        if f > 11: world = over(world, petals(f + 3, 26, 33, PET, wind=(160., 30.), size=(16, 34), near=6))
        base = world
        if f < 11:                                                    # the page with the violet card turns
            u = float(keys(f, [(3, 0.), (5, .10), (10, 1.06, lambda q: q * q * (3 - 2 * q))])); card_ = over(self.card.copy(), rb) if f < 9 else over(self.card.copy(), rb)
            if u < 1.05:
                x, y = grid(); edge = W * (1.02 - 1.16 * u) + 130 * np.sin((y / H - .1) * 2.2) * u + 26 * np.sin(y / 190. + f); cw = 40 + 420 * math.sin(math.pi * min(u, 1.)) ; m = np.clip((edge - x) / 1.5, 0, 1)[..., None]; q = np.clip((x - edge) / cw, 0, 1); under = ((x > edge) & (x < edge + cw)).astype(f32)
                shd = np.exp(-np.maximum(x - edge - cw * .8, 0) / 120.) * (x > edge) * .55 * (u > .02); out = world * (1 - shd[..., None]); pg = ramp(q, [(0, '#cdb6e6'), (.35, '#efe4f6'), (.7, '#b79ad6'), (1, '#7d5aa8')]); lines = (np.abs(((y + x * .06) / 34.) % 1 - .5) < .05) * ((x - edge) > 30) * ((x - edge) < cw - 30); pg = pg * (1 - .18 * lines[..., None]); out = out * (1 - under[..., None]) + pg * under[..., None]; out = out * (1 - m) + card_ * m; ln = np.exp(-((x - edge) / 2.2) ** 2) + np.exp(-((x - edge - cw) / 2.2) ** 2) * (u > .02); base = out * (1 - np.clip(ln, 0, 1)[..., None] * .8) if u > .02 else card_
                if f < 4: base = card_
        if hit:
            k = f - 25
            c = Cel(); 
            if k < 3: focus_lines(c, 960, 420, 300 + 160 * k, 1700, 90, 10., '#fff4fb', seed=f, ry=.7); base = over(base, c.layer(), .9)
            base = flare(base, 960, 420, 420 - 60 * k, '#ffd9ec', .8 * math.exp(-k / 1.6)); base = over(base, petals(k * 1.6 + .6, 130, 51, PET, wind=(0., 0.), size=(16, 46), origin=(960, 420), burst=3800., near=8, spin=5.))
            if f == 25: base = base * .5 + rgb('#fff2fa') * .5
            if f >= 26:                                               # petal-scalloped shock opens the next card
                rad = (0, 190, 560, 1250, 2000)[f - 25]; a = np.linspace(0, TAU, 900, endpoint=False); nP = 15; rr = rad * (.84 + .16 * np.abs(np.sin(a * nP * .5 + k)) ** .55 + .02 * np.sin(a * 5 + f)); P = np.c_[960 + np.cos(a) * rr, 420 + np.sin(a) * rr * .86]; m = mask(P); base = base * (1 - m) + self.next * m; c = Cel(); c.line(P, 20, '#5c2a66', closed=True, press=.4); c.line(P, 12, '#ffd3ea', closed=True, press=.4); base = over(base, c.layer())
        if f > 5: base = base + motes(f, 70, 5, None, '#ffd9ee', (20, -16), 1.2, .5)
        return np.clip(base, 0, 2.5).astype(f32)


def mask(P):
    c = Cel(); c.fill(P, '#ffffff'); return c.layer()[..., 3:4]
