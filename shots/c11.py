"""c11 Bonney: a slice of pizza, a rubber hand, shark-tooth fury, one plucked string of cheese."""
from chars.lkit import *
from chars import bonney as B
from op.comp import over, radial, blur
from op.kit import focus_lines, splatter

BX, BY, S = 1300, 500, 228


def slice_(cel, x, y, s, rot, seed=3, lw=3.6, drip=1., bite=0.):
    """pizza slice, tip toward +x at rot 0, (x, y) = middle"""
    Q = lambda P: xf(np.asarray(P, float), s, s, rot, x, y)
    for k, (px, L, w) in enumerate(((-.25, .42, .075), (.18, .60, .06), (.52, .36, .05), (-.05, .26, .05))):
        L = L * drip; D = Q([(0, 0)])[0]; p = Q([(px, .40 - .34 * (px + .5))])[0]; w2 = w * s; P = np.array([p + [-w2, -s * .1], p + [-w2 * .55, s * L * .5], p + [-w2 * .75, s * L * .92], p + [0, s * L * 1.02], p + [w2 * .75, s * L * .92], p + [w2 * .5, s * L * .5], p + [w2, -s * .1]]); P = spl(P, n=6); cel.fill(P, '#f7cb4b'); stroke(cel, P, lw * .7)
    T = Q(spl([(-.62, -.50), (-.20, -.38), (.35, -.20), (.86, .02), (.30, .25), (-.15, .42), (-.58, .56)], False, n=6)); cel.fill(T, '#fbd65c'); cel.clip(T); cel.fill(Q(spl([(-.7, .20), (-.2, .24), (.4, .10), (.9, .0), (.9, .6), (-.7, .8)], True, n=5)), '#e0a032'); cel.unclip(); outline(cel, T, lw, INK, seed)
    for px, py, r in ((-.28, -.14, .155), (.12, .04, .14), (-.30, .26, .135), (.46, -.02, .10)):
        a = np.linspace(0, TAU, 26, endpoint=False); P = Q(np.c_[px + np.cos(a) * r, py + np.sin(a) * r * .86]); cel.fill(P, '#cf4427'); cel.shade(P + [s * .05, s * .06], '#8f2417', P, .8); cel.fill(Q(np.c_[px - r * .3 + np.cos(a) * r * .28, py - r * .32 + np.sin(a) * r * .16]), '#f58a5e'); stroke(cel, P, lw * .55)
    for px, py in ((-.05, -.22), (.28, .12), (-.10, .30)): P = Q(spl([(px - .06, py), (px, py - .05), (px + .07, py + .01), (px, py + .035)], True, n=5)); cel.fill(P, '#4f8f3a'); stroke(cel, P, lw * .35)
    C = Q(spl([(-.60, -.62), (-.78, -.52), (-.86, 0), (-.78, .58), (-.60, .68), (-.50, .52), (-.56, 0), (-.50, -.48)], True, n=6)); cel.fill(C, '#d08d3e'); cel.clip(C); cel.fill(Q(spl([(-.62, -.7), (-.66, 0), (-.62, .7), (-.4, .7), (-.4, -.7)], True, n=5)), '#9a5c22'); cel.fill(Q(spl([(-.80, -.36), (-.79, -.1), (-.82, .14), (-.76, -.08)], True, n=5)), '#f4c67c'); cel.unclip(); outline(cel, C, lw, INK, seed + 2)


def rubber(cel, P, w0, w1, lw, skin='#f0b98d', shade='#c9805c'):
    """stretched arm along P (n,2)"""
    P = resample(spl(P, n=12), 6.); n = len(P); u = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None]; N = np.c_[-T[:, 1], T[:, 0]]; w = lerp(w0, w1, u)[:, None] * .5; A, D = P + N * w, P - N * w; cel.fill(np.vstack([A, D[::-1]]), skin); cel.fill(np.vstack([A, (A - N * w * .75)[::-1]]), shade); cel.line(A, lw, INK, taper=(0, 0), press=.4); cel.line(D, lw, INK, taper=(0, 0), press=.4, seed=2)
    for k in range(6): i = int(n * (.08 + .15 * k)); cel.line((P + N * w * (.1 - .5 * (k % 2)))[i:i + int(n * .10)], lw * .55, '#8a4a30', taper=(.4, .5))


class Shot:
    id = 'c11'; n = 30; finish = dict(bloom=.22, vig=.10)
    EVENTS = [(0, 'cloth', dict(note='coat and hair, happy breath in')), (3, 'whoosh', dict(size=.2, note='slice lifted')), (9, 'voice', dict(note='Bonney: aaa- (mouth opens)')), (11, 'whip', dict(size=.6, note='rubber arm shoots in')), (12, 'grab', dict(size=.7, note='hand slaps the slice')), (14, 'creak', dict(note='cheese stretches')),
              (16, 'snap', dict(size=.9, note='slice gone, arm snaps back')), (16, 'chomp', dict(size=.7, note='teeth close on air')), (18, 'voice', dict(note='Bonney fury shriek')), (18, 'impact', dict(size=.6, note='shark-tooth fury')), (25, 'pizzicato', dict(size=1.0, note='cheese string snaps: plucked DON')), (27, 'splat', dict(size=.4, note='cheese on the nose'))]

    def setup(self):
        self.card = card(H, W, '#1440c4', '#0a2480', '#2f63ee', 4, -.5)
        P = dict(POSE := HD.POSES['pinch']); pal = dict(skin=B.PAL['skin'], shade=B.PAL['shade']); self.hand, _ = hand_sprite('pinch', 'R', HD.euler(60, 12, -58), 92, HD.FEMALE, pal, w=3.4, rimc='#fff2f6', rimd=(-.78, .62), rw=5, style=dict(nails='short', cuff=dict(kind='uniform', y=.34, flare=.10))if False else dict(nails='short'))
        pl = dict(skin='#f0b98d', shade='#c9805c'); self.rh0, _ = hand_sprite('splay', 'R', HD.euler(20, 35, -90), 112, HD.MALE, pl, w=4.0, rimc='#fff2f6', rimd=(-.78, .62), style=dict(nails='short')); self.rh1, _ = hand_sprite('grip', 'R', HD.euler(30, 20, -90), 112, HD.MALE, pl, w=4.0, rimc='#fff2f6', rimd=(-.78, .62), style=dict(nails='short'))

    def frame(self, f):
        d = f // 2 * 2; bg = self.card.copy(); fury = float(f >= 18) * (1.0 if f < 25 else .85); cel = Cel()
        # cream ovals, drifting on the same wind
        for k, (x, y, rx, ry, a) in enumerate(((300, 260, 250, 150, -.28), (560, 690, 200, 150, .5), (820, 190, 120, 88, -.5), (120, 820, 150, 190, .3), (1790, 300, 120, 170, .5), (1700, 860, 90, 120, -.4))):
            x -= f * (2.6 + k * .7); a0 = np.linspace(0, TAU, 60, endpoint=False); wob = 1 + .015 * np.sin(a0 * 3 + k + d * .3); P = np.c_[np.cos(a0) * rx * wob, np.sin(a0) * ry * wob]; P = xf(P, 1, 1, a + .02 * math.sin(d * .2 + k), x, y); cel.fill(P, '#fbf1d6'); cel.shade(P + [16, 22], '#f1dfb4', P, .0); cel.line(P[4:34], 3.0, '#e6cf9a', taper=(.3, .4)); cel.line(P[38:56], 2., '#e6cf9a', taper=(.3, .4))
        base = over(bg, cel.layer())
        if f >= 18:
            c = Cel(); focus_lines(c, BX, BY + 20, 520 + 30 * (f % 2), 1700, 120, 9., '#0a1550', seed=f // 2, ry=.8); base = over(base, c.layer(), .85); base = base * (1 - .25 * radial(BX, BY, 1500, 1100, 1.2)[..., None] * 0)
        # acting
        def pos(q): return np.array([BX + float(keys(q, [(0, 8.), (8, 0.), (10, 10.), (12, -10., out3), (15, -34.), (16, 26., out3), (18, -6., out3), (19, -14.), (22, 0.), (25, 4.), (26, 18., out3), (28, 4.)])), BY + float(keys(q, [(0, 14.), (6, 4.), (10, -10.), (12, 4.), (15, 8.), (16, -6.), (17, 4.), (18, -26., out3), (20, -8.), (25, 6.), (26, 12.), (29, 4.)]))])
        p = pos(f); yaw = float(keys(f, [(0, -.16), (8, -.26), (12, -.30), (15, -.34), (16, -.10), (18, -.22), (30, -.22)])); pitch = float(keys(f, [(0, -.06), (9, .12), (12, .10), (16, -.04), (18, .04), (25, .0), (30, -.03)])); roll = float(keys(f, [(0, .05), (9, -.05), (12, -.02), (15, -.12), (16, .06), (18, -.04), (25, 0.), (30, .03)])); sh = 0.; vib = 0.; scale = 1.
        if f < 6: ex = dict(o=1.0, M='smile', gaze=(-.5, -.5), wob=.8, blush=.5, bi=.05, bo=.04)
        elif f < 9: ex = dict(o=.55, M=mmix('oh', 'bite', .45), teeth=(.06, .03), tongue=.4, gaze=(-.5, -.5), blush=.6, bi=.05, bo=.04)
        elif f < 12: ex = dict(o=0., shut='tight', M='bite', msx=1.06, msy=1.12, teeth=(.09, .07), tongue=.55, blush=.7, jaw=.16, bi=.06, bo=.06)
        elif f < 16: ex = dict(o=1.25, ir=.55, M='bite', msx=.96, teeth=(.10, .07), tongue=.45, gaze=(-.9, -.6), jaw=.12, bi=.12, bo=.08); vib = 2.
        elif f < 18: ex = dict(o=1.2, ir=.30, M='clench', clench=True, gaze=(-.2, 0), bi=.10, bo=.02)
        elif f < 25: ex = dict(o=1.30, ir=.001, M='howl', msx=1.32, msy=1.30, shark=True, tongue=.40, jaw=.20, bi=-.16, bo=.12, bk=1.2, flare=1., mseed=f, jit=.004); vib = 5. * math.exp(-(f - 18) / 5.) + 1.5; scale = 1 + .10 * math.exp(-(f - 18) / 2.2)
        else: ex = dict(o=.62, ir=.34, M='press', gaze=(0, .1), bi=-.02, bo=.0, nonose=False); scale = 1.
        sh = float(keys(f, [(0, .03), (9, .12), (12, .0), (16, .05), (18, .22), (25, .05), (30, 0.)])); j = rs(f, 3).normal(0, 1, 2) * vib; q = p + j
        st = B.style()
        D = B.draw(q[0], q[1], S * scale, yaw, pitch, roll, None, st, d if f < 18 else f, wind=(math.pi * 1.05, 1.), ex=ex, body=lambda c, F, s_: B.coat(c, q[0] - 10 + (p[0] - BX) * -.5, q[1] + 6, S, roll * .4, s_, 4.4, f, sh), posfn=pos, fury=fury, pop=float(keys(f, [(0, 0), (17, 0), (18, .30, out3), (21, .10), (24, .12), (25, .0), (26, -.02), (30, 0)])), caproll=.10 * math.sin(f * 1.1) * (f >= 18) * math.exp(-(f - 18) / 8.))
        lay_ = stack(D['back'], D['body'], D['face'], D['front'], D['cap']); F = D['F']; M = F.X([(0, -.66)], .02)[0]
        lay_ = para(lay_, (0, 0), (500, 1080), [(0, '#ffffff'), (.6, '#fff4f6'), (1, '#d9b2d6')]); base = over(base, lay_)
        # hand, slice, cheese
        cel = Cel(); hx, hy = keys(f, [(0, (820., 985.)), (3, (860., 900.)), (8, (930., 790.)), (11, (965., 748.)), (12, (955., 756.)), (15, (905., 772.)), (16, (975., 800.), out3), (17, (960., 930.)), (18, (960., 1200.)), (30, (960., 1400.))]); rot = float(keys(f, [(0, -.50), (8, -.16), (11, -.10), (30, -.1)])); sx, sy = hx + 110, hy - 70; held = f < 16; arm = None
        if f >= 11:
            tip = keys(f, [(11, (260., 690.)), (12, (905., 700.), out3), (13, (900., 702.)), (15, (800., 712.)), (16, (330., 640.), lambda t: t), (17, (-260., 560.), lambda t: t)]); wav = 26 * math.sin(f * 2.2) if 12 <= f < 16 else 70 * math.sin(f * 1.9); arm = [(-160, 860), (lerp(-160, tip[0], .35), lerp(850, tip[1], .35) + wav), (lerp(-160, tip[0], .7), lerp(840, tip[1], .7) - wav * .7), (tip[0] - 60, tip[1] + 2)]
        string = None
        if f >= 12:                                                   # cheese from her teeth to the slice / off screen, then it snaps
            e = (sx + 60, sy + 30) if f < 16 else (-60., 610.); m0 = M + [-18, 6]; u = np.linspace(0, 1, 40)
            if f < 25: amp = (30 * (1 - (f - 12) / 5.) if f < 16 else 26 * math.sin(f * 2.6) * math.exp(-(f - 16) / 5.) + 4 * math.sin(f * 3.)); string = np.c_[lerp(m0[0], e[0], u), lerp(m0[1], e[1], u) + np.sin(np.pi * u) * (amp if f >= 16 else abs(amp)) + (6 * np.sin(TAU * 3 * u + f) if f >= 16 else 0)]
            else: k = (f - 25) / 4.; L = lerp(980, 150, float(out3(k))); string = np.c_[m0[0] - u * L, m0[1] - np.sin(TAU * (2.4 * u)) * 60 * u * (1 - .6 * k) - 130 * u * k]
        if string is not None:
            ww = 9. if f < 16 else 5.5 if f < 25 else 8.; from op.ink import ribbon
            R = ribbon(string, ww + 5, (0, .1), .5, 3, 0.); cel.fill(R, INK); R = ribbon(string, ww, (0, .1), .5, 3, 0.); cel.fill(R, '#fbd65c'); cel.line(string - [0, ww * .18], ww * .28, '#fff3bd', taper=(.1, .1))
        if f < 16: slice_(cel, sx, sy, 138, rot, drip=1 + .25 * math.sin(d * .5) + (.5 * (f - 12) / 4. if f >= 12 else 0))
        elif f == 16: slice_(cel, float(tip[0]) + 80, float(tip[1]) + 10, 138, .3, drip=1.5)
        base = over(base, cel.layer())
        L = np.zeros((H, W, 4), f32); put(L, self.hand, hx, hy + 22, 1., rot * .5 + (.06 * math.sin(f * 2.) if 12 <= f < 16 else 0))
        if arm is not None and f <= 17: put(L, self.rh0 if f == 11 else self.rh1, float(tip[0]) - 20, float(tip[1]), 1., .05); cA = Cel(); arm[-1] = (float(tip[0]) - 190, float(tip[1]) - 9); rubber(cA, arm, 128, 66, 4.4); L = lay(L, lit(cA.layer(), (-.2, .98), '#fff2f6', 4, 6, .8))
        # her sleeve: fur cuff + coat
        if hy < 1150:
            c = Cel(); w_ = np.array([hx - 70, hy + 190]); sl = spl([w_ + [-70, -30], w_ + [60, 10], w_ + [190, 240], w_ + [-190, 260]], True, n=6); c.fill(sl, '#d2912a'); c.clip(sl); c.fill(sl + [46, 30], '#a96a1a', .95); c.unclip(); outline(c, sl, 4.4); fold(c, [w_ + [-60, 60], w_ + [0, 90], w_ + [50, 80]], 2.4); fold(c, [w_ + [-80, 110], w_ + [-10, 150], w_ + [70, 140]], 2.4); B.fur(c, [w_ + [-78, -26], w_ + [-8, 8], w_ + [66, 16]], 44, 4., 8); L2 = lit(c.layer(), (-.78, .62), '#fff2f6', 4, 6, .9); L = lay(L2, L) if False else lay(L, L2)
        if f in (11, 12, 16, 17):
            sm_ = np.zeros_like(L); w0 = 150 if f in (12, 17) else 90
            for k in np.linspace(-.5, .5, 9): sm_ += cv2.warpAffine(L, np.float32([[1, 0, w0 * k], [0, 1, 0]]), (W, H)) / 9.
            kx = np.clip((np.arange(W)[None, :, None] - float(tip[0]) - 60) / 120., 0, 1); L = sm_ * (1 - kx) + L * kx if f != 12 else L * .65 + sm_ * .35
        base = over(base, L); fx = Cel()
        if 18 <= f < 25:                                              # vein + spit + steam
            k = 1 + .25 * math.sin((f - 18) * 1.6); v = np.array([BX + 170, BY - 250]) + j
            for a in (0, 1, 2, 3): th = a * math.pi / 2 + .3; P = spl([v + xf([(14, 14)], k, k, th)[0], v + xf([(22, 34)], k, k, th)[0], v + xf([(16, 62)], k, k, th)[0]], n=6); fx.line(P, 15 * k, '#0a0a12', taper=(.2, .4)); fx.line(P, 9 * k, '#f0243c', taper=(.2, .4))
            r = rs(7)
            for i in range(9): a = math.pi * (.75 + .5 * r.random()); D_ = 60 + (f - 18) * r.uniform(26, 60); x, y = M + [math.cos(a) * D_, -math.sin(a) * D_ * .7 + (f - 18) ** 2 * 1.2]; bead(fx, x, y, r.uniform(5, 11), a + math.pi, .5, 2.)
        if f >= 25:
            nz = F.X3(F.h.nose_profile())[4]
            if f >= 27: P = spl([nz + [-34, -4], nz + [-14, -24], nz + [18, -20], nz + [36, 2], nz + [26, 26], nz + [10, 22], nz + [4, 54 + 4 * (f - 27)], nz + [-10, 24], nz + [-28, 22]], True, n=6); fx.fill(P, '#fbd65c'); fx.shade(P + [8, 14], '#e0a032', P, .0); stroke(fx, P, 3.6); fx.fill(nz + spl([(-18, -8), (-6, -14), (2, -8), (-8, -4)], True, n=5), '#fff3bd')
            for i in range(10): a = i * .628 + .2; r0, r1 = 300 + 26 * (f - 25), 360 + 60 * (f - 25) * (1 + .4 * (i % 2)); c_ = np.array([BX, BY + 40]); fx.line(np.array([c_ + [math.cos(a) * r0, math.sin(a) * r0 * .9], c_ + [math.cos(a) * r1, math.sin(a) * r1 * .9]]), 9, '#fbf1d6', taper=(.1, .6))
        base = over(base, fx.layer())
        return np.clip(base, 0, 2).astype(f32)
