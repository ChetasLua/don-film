"""d1  The vow.  f0-29 the white back, the black cross, one flash, red as brush ink only.  f30-89 from straight above: hand over streaming eyes, the mouth that will not close, the white sword."""
from chars.swkit import *
from chars import zoro
from op import hands as HD, sumi
from op.fx_core import noise as nz1
import functools

Y = dict(base='#f2b680', shade='#b5663f', deep='#70301f', light='#ffdcae', bounce='#dc9a68')
HP = dict(skin='#f2b680', shade='#b5663f', line=INK, nail='#ffd9b4', nail_shade='#c98a64', light='#ffe9c8')
SY = dict(zoro.HAIR, base='#6cbf5c', light='#a6e684', hot='#efffc8')


# ------------------------------------------------------------------ A: the platform
def far_plate():
    y, x = grid()[1], grid()[0]; im = ramp(y / H, [(0, '#6d6455'), (.22, '#a88f66'), (.42, '#dcb878'), (.57, '#f3dba4'), (.60, '#f8e8bf'), (.605, '#a08a63'), (.68, '#6e6450'), (1, '#3d3c38')])
    g = fbm(H, W, (340, 120, 40, 14), None, 5, aspect=2.6); im = np.clip(im * (1 + .045 * g[..., None]), 0, 1); sun = np.exp(-(((x - 1250) / 300) ** 2 + ((y - 600) / 190) ** 2)); im = im + sun[..., None] * rgb('#ffe0a0') * .35
    cl = smooth(.45, 1.3, fbm(H, W, (260, 90, 34, 12), None, 12, aspect=3.2) + .9 * np.exp(-((y - 250) / 170) ** 2) - .25) * (y < 640); im = im * (1 - .40 * cl[..., None]) + rgb('#6b5c4c') * (.40 * cl[..., None])
    im = repaint(im, (26, 12, 6), 4, angle=np.full((H, W), .02, f32), jit=(.010, .03, .03))
    c = Cel(); r = np.random.default_rng(8)
    for k in range(150):
        yy = 652 + 130 * r.uniform(0, 1) ** 1.6; xx = r.uniform(-40, W + 40) if r.random() < .55 else r.normal(1250, 130); L = (8 + 90 * (yy - 648) / 130) * r.uniform(.5, 1.4); c.line(np.array([(xx - L, yy), (xx, yy + r.normal(0, 1)), (xx + L, yy)]), 1 + 2.6 * (yy - 648) / 130, '#f6dfae' if abs(xx - 1250) < 260 else '#8d8166', taper=(.4, .4), alpha=.8, seed=k)
    # the floating restaurant: round hull, fish-head prow, dorsal roofs, sail fins (far, soft silhouette)
    S = '#4a4334'; hull = spl([(150, 560), (190, 655), (330, 678), (520, 672), (610, 640), (668, 600), (700, 560), (676, 548), (640, 566), (560, 560), (330, 566)], True, (0, 6, 7), n=8); c.fill(hull, S)
    c.fill(spl([(640, 566), (664, 520), (712, 506), (752, 528), (758, 552), (726, 560), (752, 566), (728, 590), (668, 600)], True, (4, 5, 6), n=6), S); a = np.linspace(0, 6.283, 20); c.fill(np.c_[712 + np.cos(a) * 8, 532 + np.sin(a) * 8], '#d7bd88')
    for q in ([(230, 562), (240, 470), (300, 430), (380, 428), (440, 470), (452, 562)], [(300, 432), (310, 380), (345, 350), (382, 380), (388, 432)]): c.fill(spl(q, True, (0, 5), n=6), S)
    for mx, top in ((344, 210), (540, 300)): c.fill(np.array([(mx - 4, 560), (mx - 2, top), (mx + 2, top), (mx + 4, 560)]), S); c.fill(spl([(mx, top + 20), (mx + 86, top + 110), (mx + 92, top + 190), (mx, top + 160)], True, (0, 2, 3), n=6), '#5d5340')
    for k in range(9): c.fill(np.array([(262 + k * 21, 500), (272 + k * 21, 500), (272 + k * 21, 516), (262 + k * 21, 516)]), '#f0cf86', .9)
    c.fill(spl([(150, 560), (110, 520), (96, 450), (140, 500), (182, 530)], True, (2,), n=6), S)
    return over(im, c.layer())


def coffin(c, f):
    b = math.sin(f * .12) * 2; K = '#2c2923'; x, y = 1660, 676 + b
    c.fill(np.array([(x - 84, y - 10), (x - 60, y + 22), (x + 64, y + 22), (x + 96, y - 4), (x + 74, y - 26), (x - 60, y - 26)]), K); c.fill(np.array([(x - 4, y - 26), (x - 3, y - 230), (x + 3, y - 230), (x + 5, y - 26)]), K); c.fill(np.array([(x - 66, y - 190), (x + 66, y - 196), (x + 66, y - 188), (x - 66, y - 182)]), K)
    c.fill(spl([(x - 4, y - 182), (x + 60, y - 180), (x + 70, y - 110), (x + 6, y - 70)], True, (0, 1, 2, 3), n=6), '#3a352c')
    for dx in (-76, 84):
        c.fill(np.array([(x + dx - 3, y - 12), (x + dx - 3, y - 44), (x + dx + 3, y - 44), (x + dx + 3, y - 12)]), '#8d8a74'); fl = 1 + .25 * math.sin(f * .9 + dx); c.fill(spl([(x + dx - 5, y - 46), (x + dx - 2, y - 58 * 1.0 - 8 * fl), (x + dx + 1, y - 64 - 8 * fl), (x + dx + 5, y - 46), (x + dx, y - 42)], True, (2,), n=6), '#3fe08a'); c.fill(spl([(x + dx - 2, y - 46), (x + dx, y - 58), (x + dx + 2, y - 46)], True, n=5), '#d8ffe6')


def mihawk(c, x, y, s, up=0., f=0, col='#17150f'):
    """far swordsman: wide plumed hat, long coat, the black cross.  up 0: blade planted before him; 1: raised"""
    P = lambda q: np.asarray(q, float) * s + [x, y]; sw = math.sin(f * .3) * .03
    c.fill(spl(P([(-.16, -1.25), (-.30, -.9), (-.46 - sw, -.10), (-.60 - sw, .0), (-.22, -.06), (-.14, .02), (.0, -.04), (.14, .02), (.24, -.06), (.62 - sw, -.02), (.46, -.14), (.30, -.9), (.16, -1.25)]), True, (3, 4, 5, 7, 8, 9), n=6), col)
    c.fill(spl(P([(-.34, -1.36), (-.12, -1.42), (-.10, -1.56), (.10, -1.56), (.12, -1.42), (.36, -1.36), (.0, -1.30)]), True, (0, 5), n=6), col); c.fill(spl(P([(-.09, -1.40), (-.10, -1.22), (.10, -1.22), (.09, -1.40)]), True, n=6), col)
    for j in range(3): c.line(spl(P([(.08, -1.50), (.30 + .06 * j, -1.66 + .05 * j), (.48 + .06 * j, -1.52 + .09 * j), (.58 + .05 * j, -1.34 + .09 * j + sw)]), n=8), s * .075, col, taper=(.2, .6), seed=j)
    o = P([(-.62 + .30 * up, -1.02 - .72 * up)])[0]; th = 2.2 * up; d = A_(-math.sin(th) * .9, math.cos(th)); d = d / np.hypot(*d); m = A_(-d[1], d[0])
    c.fill(np.array([o - d * s * .42 - m * s * .03, o - d * s * .42 + m * s * .03, o + d * s * .0 + m * s * .05, o + d * s * .90 + m * s * .10, o + d * s * 1.05, o + d * s * .90 - m * s * .07, o + d * s * .0 - m * s * .05]), col)
    c.fill(np.array([o - m * s * .40 - d * s * .03, o - m * s * .40 + d * s * .03, o + m * s * .40 + d * s * .03, o + m * s * .40 - d * s * .03]), col)
    for sg in (-1, 1): B, *_ = limb([P([(sg * .18, -1.18)])[0], P([(-.16 + sg * .20, -1.02 - .30 * up)])[0], o - d * s * .20], [s * .11, s * .08, s * .05], [s * .11, s * .08, s * .05]); c.fill(B, col)


def back_fig(c, x, y, s, arm, dip=0., lean=0., f=0, wind=1., col=None):
    """Zoro from behind, unmarked white back, arms opening (arm -1 hanging .. 1 level, >1 above)"""
    K = dict(tee='#c9b792', teed='#86755c', skin='#8a5c3c', skind='#56371f', trs='#22241f', trsd='#131512', sash='#3f5a34', line='#120f0c', rim='#ffe6a6'); K.update(col or {})
    R_ = lambda P: xf(np.asarray(P, float) * s + [0, dip * s], 1, 1, lean, x, y); w = s / 110.
    H_ = zoro.make(True); hp = dict(deep='#141f14', shade='#22351f', base='#3a5a33', light='#5d8a4a', hot='#d9e08a', line='#101a10'); q = R_([(0, -6.78 + .05 * max(0, arm))])[0]; hs = s * .50
    # legs, sash, scabbards
    for sg in (-1, 1): L, *_ = limb(R_([(sg * .38, -3.75), (sg * .62, -2.3), (sg * .78, -.6), (sg * .82, .4)]), [.50 * s, .42 * s, .34 * s], [.50 * s, .44 * s, .34 * s]); cloth(c, L, K['trs'], K['trsd'], [L[len(L) // 2:] if sg < 0 else L[:len(L) // 2]], w=4 * w, ink=K['line'], folds=[R_([(sg * .2, -3.0), (sg * .55, -2.4)]), R_([(sg * .4, -2.2), (sg * .9, -1.9)])], fw=2.4 * w)
    for j, kd in enumerate(('wado', 'kitetsu', 'enma')):
        a, b = R_([(.62 + .05 * j, -3.95 + .16 * j), (1.35 + .16 * j, -1.6 + .30 * j)]); scabbard(c, a, b, 20 * w, kd, 2.6 * w, K['line']); h = a + (a - b) / np.hypot(*(a - b)) * s * .72
        if kd != 'wado': pass
    T = spl(R_([(-.36, -6.16), (-.98, -5.98), (-1.12, -5.60), (-1.00, -5.15), (-.84, -4.6), (-.76, -4.0), (-.82, -3.70), (.82, -3.70), (.76, -4.0), (.84, -4.6), (1.00, -5.15), (1.12, -5.60), (.98, -5.98), (.36, -6.16)]), True, (6, 7), n=6)
    # arms behind the tee sleeves
    for sg in (-1, 1):
        th = math.radians(78 * (1 - min(arm, 1)) * 1.0) if arm <= 1 else -math.radians(16 * (arm - 1) * 4); th = math.radians(76) * (1 - arm) if arm < 1 else -math.radians(24) * (arm - 1); e1 = th - (.10 if arm > .7 else -.15)
        S_ = A_(sg * 1.0, -5.72); E = S_ + A_(sg * math.cos(th), math.sin(th)) * 1.42; e2 = th - (.16 if arm > .7 else .05); Wp = E + A_(sg * math.cos(e2), math.sin(e2)) * 1.32; P = R_([S_, (S_ + E) / 2 + A_(0, .02), E, (E * .6 + Wp * .4), Wp])
        L, a, b, cc = limb(P, np.array([.30, .34, .30, .24, .27, .25, .17, .145]) * s, np.array([.30, .30, .24, .22, .24, .20, .15, .145]) * s); lo = b if (b.mean(0)[1] > a.mean(0)[1]) else a
        cloth(c, L, K['skin'], K['skind'], [np.vstack([cc, lo[::-1]])], w=4 * w, ink=K['line'], folds=[cc[len(cc) * 45 // 100:len(cc) * 58 // 100] + [0, .05 * s]], fw=2.2 * w)
        u = A_(sg * math.cos(e2), math.sin(e2)); m = A_(-u[1], u[0]); hb = [Wp + u * .02 + m * .14, Wp + u * .38 + m * .22, Wp + u * .42 - m * .18, Wp + u * .02 - m * .14]
        for j, (o, ln) in enumerate(((.18, .30), (.07, .36), (-.05, .34), (-.16, .26))): F, *_ = limb(R_([Wp + u * .36 + m * o, Wp + u * (.42 + ln * .6) + m * (o * 1.5), Wp + u * (.42 + ln) + m * (o * 1.75)]), [.062 * s, .055 * s, .04 * s], [.062 * s, .055 * s, .04 * s]); cloth(c, F, K['skin'], w=2.6 * w, ink=K['line'])
        F, *_ = limb(R_([Wp + u * .12 - m * .10 * 1, Wp + u * .30 - m * .34, Wp + u * .42 - m * .44]), [.075 * s, .06 * s, .045 * s], [.075 * s, .06 * s, .045 * s]); cloth(c, F, K['skind'], w=2.6 * w, ink=K['line'])
        cloth(c, spl(R_(hb), True, n=6), K['skin'], K['skind'], [R_([Wp + u * .02 - m * .02, Wp + u * .42 - m * .06, Wp + u * .42 - m * .2, Wp - m * .2])], w=3.2 * w, ink=K['line'])
        sl = spl(R_([S_ + A_(-sg * .18, -.34), S_ + A_(sg * math.cos(th), math.sin(th)) * .62 + A_(-sg * math.sin(th) * -.38 * sg * sg, -.40 * math.cos(th)) * 1., S_ + A_(sg * math.cos(th), math.sin(th)) * .66 + A_(-sg * math.sin(th) * .42, .40 * math.cos(th)), S_ + A_(-sg * .05, .52)]), True, (1, 2), n=8)
        cloth(c, sl, K['tee'], K['teed'], [spl(R_([S_ + A_(sg * .1, .0), S_ + A_(sg * math.cos(th), math.sin(th)) * .62 + A_(0, .05), S_ + A_(sg * math.cos(th), math.sin(th)) * .7 + A_(0, .5), S_ + A_(0, .6)]), True, n=6)], w=3.6 * w, ink=K['line'])
    N = spl(R_([(-.24, -6.6), (-.26, -6.28), (-.52, -6.08), (.52, -6.08), (.26, -6.28), (.24, -6.6)]), True, (2, 3), n=6); cloth(c, N, K['skin'], K['skind'], [R_([(-.1, -6.7), (.02, -6.2), (-.6, -6.0), (-.4, -6.7)])], w=3 * w, ink=K['line'], folds=[R_([(-.07, -6.5), (-.09, -6.16)]), R_([(.08, -6.5), (.10, -6.16)])])
    cloth(c, T, K['tee'], K['teed'], [spl(R_([(-1.2, -5.95), (-.70, -5.80), (-.18, -5.60), (-.50, -5.30), (-.30, -5.0), (-.52, -4.7), (-.28, -4.3), (-.46, -3.6), (-1.2, -3.6)]), True, (2, 4, 6, 7, 8), n=6), spl(R_([(.10, -5.70), (.40, -5.46), (.86, -5.36), (.60, -5.16), (.22, -5.20), (.06, -4.7)]), True, (0, 2, 5), n=6)], '#f7e7bd', [spl(R_([(.40, -6.02), (.96, -5.90), (1.04, -5.62), (.74, -5.72)]), True, n=6)], w=4.6 * w, ink=K['line'],
          folds=[R_([(0, -5.9), (-.03, -5.2), (.02, -4.4), (0, -3.9)]), R_([(-.95, -5.35), (-.55, -5.15), (-.30, -5.25)]), R_([(.95, -5.35), (.6, -5.0), (.3, -5.05)]), R_([(-.8, -4.5), (-.4, -4.3), (-.1, -4.4)]), R_([(.8, -4.45), (.45, -4.2), (.15, -4.3)]), R_([(-.75, -4.05), (-.3, -3.95)])], fw=2.4 * w)
    cloth(c, spl(R_([(-.86, -4.12), (.86, -4.12), (.90, -3.48), (-.90, -3.48)]), True, (0, 1, 2, 3), n=6), K['sash'], '#22331d', [R_([(-.9, -4.1), (-.2, -4.1), (-.3, -3.4), (-.9, -3.4)])], w=3.6 * w, ink=K['line'], folds=[R_([(-.8, -3.9), (0, -3.84), (.8, -3.92)]), R_([(-.8, -3.68), (0, -3.62), (.8, -3.70)])], fw=2. * w)
    H_.pose(q[0], q[1], hs, math.pi + .05, -.06 * 0, lean); T_ = zoro.table(H_); fac = (T_['n'] @ H_.R.T)[:, 2]
    for sd in (-1, 1): e = H_.ear(sd); cloth(c, e['outer'], K['skin'], w=2.4 * w, ink=K['line'])
    c.fill(H_.skull(), hp['deep']); P = crop_draw(c, H_, T_, fac > -.45, hp, (.5, .5, .5), ink=K['line'], lw=1.4, wind=lambda t: np.c_[np.sin(t['u'] * 6 + f * .6) * .03 + .05 * wind, 0 * t['u'], 0 * t['u']])
    for v in union(P + [H_.skull()], 1., min_len=20): outline(c, v, 3. * w, K['line'], 2)
    for i in range(3): o = H_.P3(np.array([(.82, -.36 + .03 * i, -.2)]))[0]; c.line(np.array([o, o + A_(math.sin(f * .7 + i) * .05, .16) * s]), 5 * w, '#f0c24a', taper=(.2, .1))


# ------------------------------------------------------------------ B: planks from above
PW, PH = 2112, 1188
def plank_plate():
    r = np.random.default_rng(14); y, x = np.mgrid[0:PH, 0:PW].astype(f32); ph = 168.; ky = (y + 30 * np.sin(x / 2100.)) / ph; k = np.floor(ky).astype(int); fr = ky - k
    tone = r.uniform(-.5, .5, 40); hue = r.uniform(-.5, .5, 40); base = np.stack([ramp(np.clip(.5 + tone[k % 40] * .7, 0, 1), [(0, '#a98a67'), (.5, '#c4a47c'), (1, '#d8bb93')])], 0)[0]; base = base + hue[k % 40][..., None] * np.array([.014, 0, -.016], f32)
    g = np.zeros((PH, PW), f32)
    for j in range(int(k.min()), int(k.max()) + 1): m = k == j; n = fbm(PH, PW, (9, 4, 2), (1, .8, .5), 100 + j, aspect=26.); w_ = fbm(PH, PW, (70, 24), (1, .5), 200 + j, aspect=7.); g = np.where(m, n * .6 + np.sin((fr * 9 + w_ * 1.6) * 6.283) * .55, g)
    im = base * (1 - .052 * g[..., None] * np.array([.9, 1., 1.15], f32)); im = im * (1 + .03 * fbm(PH, PW, (220, 80), None, 9)[..., None])
    gap = np.minimum(fr, 1 - fr) * ph; im = im * (1 - .80 * smooth(5.5, 1.2, gap))[..., None]; im = im + (.12 * smooth(13, 5, fr * ph) * (fr * ph > 4))[..., None] * rgb('#ffe9c4'); im = im * (1 - .25 * smooth(22, 4, (1 - fr) * ph))[..., None]
    c = Cel(PW, PH, ss=1)
    for j in range(-1, 9):
        for q in range(7): xx = r.uniform(0, PW); yy = j * ph + r.uniform(18, ph - 18); c.line(np.array([(xx, yy), (xx + r.uniform(30, 190), yy + r.normal(0, 4))]), r.uniform(1.5, 3.6), '#5c4128', taper=(.2, .6), alpha=.75, seed=int(r.integers(9999)))
    im = over(im, c.rgba()); lab = (k + 2).astype(np.int32); im = repaint(im, (15, 7, 4), 6, angle=np.zeros((PH, PW), f32) + .01, lab=lab, thr=(.02, .018, .018), jit=(.008, .02, .02), op=.7, dry=.3, start=im)
    c = Cel(PW, PH, ss=2)
    for j in range(-1, 9):
        for q in range(2):
            bx = (j * 733 + q * 1190 + 260) % PW; yy = j * ph - 30 * math.sin(bx / 2100.); c.line(np.array([(bx, yy + 4), (bx + 2, yy + ph - 4)]), 5.5, '#2e1d10', taper=(.02, .02)); c.line(np.array([(bx + 6, yy + 6), (bx + 7, yy + ph - 8)]), 2.2, '#f2d8ae', taper=(.1, .1), alpha=.7)
            for sx in (-24, 26):
                for sy in (36, ph - 38): a = np.linspace(0, 6.28, 18); c.fill(np.c_[bx + sx + np.cos(a) * 6.5, yy + sy + np.sin(a) * 6], '#3a2a1e'); c.fill(np.c_[bx + sx - 1.5 + np.cos(a) * 3, yy + sy - 2 + np.sin(a) * 2.6], '#8d7a66')
    im = over(im, c.rgba()); d = np.hypot((x - 900) / 1250, (y - 560) / 820)[..., None]; return np.clip(im * (1.07 - .30 * np.clip(d - .25, 0, 1)), 0, 1).astype(f32)


@functools.lru_cache(8)
def cover_hand(kind, sd):
    """left hand pressed over the eyes, back of the hand to us, forearm from the top of frame"""
    ps = dict(rest=HD.pose((4, 14, 8, -7), (5, 16, 10, 0), (7, 18, 10, 6), (10, 22, 12, 13), thumb=(50, 20, 24, 4, 6)), clench=HD.pose((4, 38, 24, -10), (4, 42, 26, 0), (6, 44, 26, 8), (10, 46, 26, 16), thumb=(52, 24, 16, 8, 24)))[kind]
    ps = HD.tremble(ps, sd, 2.2) if sd else ps; h = HD.Hand(ps, 'L', R=HD.euler(8, 4, 160), bld=HD.HEAVY, cuff=None); g = HD.render([h], HD.Cam(818, 470, 222, 7), light=(.45, .65, .6), res=520, clip=(0, 0, W, H)); c = Cel(); HD.ink(c, g, HP, 6.6, 3); return c.layer()


SWG, SWT, SWP = (1478, 628), (1352, -80), (1572, 1060)                        # guard, tip, pommel of the raised Wado
@functools.lru_cache(40)
def sword_layer(gl, hand=True):
    c = Cel(W + 400, H + 900, -100, -300)
    katana(c, SWG, SWT, 92, 104, 'wado', .030, 1, None if gl < 0 else gl / 100., 5.6, refl=max(0, gl) / 260.); hilt(c, SWG, SWP, 88, 'wado', 5.6, asp=.46, n=7)
    if hand:
        h = HD.Hand('hammer', 'R', R=HD.euler(168, 14, 74), bld=HD.HEAVY); g = HD.render([h], HD.Cam(1532, 838, 272, 3.6), light=(.45, .65, .6), res=640, clip=(700, 100, 2600, 1600)); HD.ink(c, g, HP, 9., 5)
    return c.rgba()


def place(lay, dx, dy, rot, sc, px, py):
    """sword layer (origin -100,-300) placed with a turn about the pivot"""
    co, si = math.cos(rot) * sc, math.sin(rot) * sc; ox, oy = -100, -300
    M = np.float32([[co, -si, px + dx - co * (px - ox) + si * (py - oy)], [si, co, py + dy - si * (px - ox) - co * (py - oy)]]); return cv2.warpAffine(lay, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


class Shot:
    id = 'd1'; n = 90; finish = dict(bloom=.30, vig=.18, sat=.93)
    EVENTS = [(0, 'ambience', dict(note='sea under a platform, wood, one gull')), (8, 'cloth', dict(note='arms thrown open')), (15, 'dash', dict(note='black blade')), (16, 'impact', dict(size=1.0, note='one flash, everything stops')), (17, 'brush', dict(note='wet red ink stroke')), (24, 'body_fall', dict(size=.5, note='slow, then water')),
              (30, 'ambience', dict(note='hull creak, small waves, tight breathing')), (34, 'sob', dict(size=.4)), (45, 'sob', dict(size=.6)), (54, 'breath_in', dict(note='before the vow')), (60, 'cry', dict(size=1.0, note='ma-ke-na-i: 60 65 70 75; earrings jingle')), (61, 'earrings', dict()), (62, 'sword_lift', dict(note='steel sings up')), (71, 'shadow', dict(note='the shadow crosses his face')), (78, 'glint', dict(size=1.0, note='sun on the tip, flare')), (84, 'sob', dict(size=.5))]

    def setup(self):
        os.makedirs('cache', exist_ok=True)
        for nm, fn in (('far', far_plate), ('planks', plank_plate)):
            p = f'cache/d1_{nm}.npy'
            if not os.path.exists(p): np.save(p + '.tmp.npy', fn().astype(np.float16)); os.replace(p + '.tmp.npy', p)
            setattr(self, nm, np.load(p).astype(f32))
        y, x = grid(); im = np.zeros((H, W, 4), f32); c = Cel(); c.fill(np.array([(-10, 1090), (-10, 800), (700, 744), (1930, 780), (1930, 1090)]), '#4b3f30')
        for k in range(-14, 22): c.line(np.array([(960 + k * 190, 1100), (1010 + k * 34, 752)]), 4.5, '#1e1810', taper=(.02, .3), seed=k + 40)
        c.line(np.array([(-10, 800), (700, 744), (1930, 780)]), 6, '#171209', taper=(.02, .02)); c.fill(np.array([(1060, 752), (1320, 766), (1780, 1090), (1180, 1090)]), '#8a6f48', .35); self.deck = c.layer()
        self.paper = np.clip(rgb('#e6d9bd') * (1 + .035 * fbm(H, W, (240, 80, 22, 5), None, 31)[..., None] + .015 * fbm(H, W, (3, 1.5), (1, .6), 6)[..., None]), 0, 1).astype(f32)
        pg = sumi.Page(seed=5); pg.stroke([(430, 50, 46, 1.6), (640, 235, 92, 1.9), (860, 450, 112, 2.2), (1090, 690, 96, 2.4), (1330, 960, 40, 2.8)], seed=12, load=.9, dry=.30, deplete=.9, split=.8, fork=.8, t0=0., t1=3.); pg.add(sumi.splash(860, 450, 190, 8, ang=.78, spread=.9, n=170, t0=1., t1=4.)); self.pg = pg

    # ---- A
    def partA(self, f):
        if f < 17:
            img = self.far.copy(); c = Cel(); coffin(c, f); up = 0. if f < 11 else float(ease((f - 10) / 4.)); 
            if f < 15: mihawk(c, 1372, 792, 235, up, f // 2)
            img = over(img, c.layer())
            for gx in (1584, 1744): img = glow(img, gx, 622, 120, '#3fe08a', .55 + .1 * math.sin(f * .9 + gx), 2.4)
            img = over(img, self.deck); c = Cel(); d = f // 2
            arm = float(lerp_keys(f, [(0, .02), (5, .02), (7, -.10), (9, 1.22), (10, 1.14), (12, .97), (16, 1.0)], lambda t: t)); dip = float(lerp_keys(f, [(0, 0), (4, .01), (7, .13), (9, -.07), (12, .02), (16, 0)])); br = .012 * math.sin(d * .9)
            back_fig(c, 800, 1226 + br * 110, 116, arm, dip, 0., d); L = c.layer(); L = para(L, (700, 300), (820, 1100), [(0, '#fff4dc'), (1, '#8d8678')]); L = rim2(L, -8, 5, '#ffe9ae', 1., .9); L = rim2(L, -3, 3, '#fff6d8', .8, .6)
            if f in (8, 9): q = smear(L, 0, 46 if f == 8 else 22, 9); L = np.where(L[..., 3:] > .5, L, q * .9)
            img = cast(img, L, 60, 30, '#140f08', .4, 22); img = over(img, L)
            if f >= 13 and f < 15: img = over(img * .9, Cel().layer())
            if f >= 15:                                                      # the black blade: already through
                c = Cel(); q = [(1500, 560), (1180, 620), (760, 700), (140, 930)] if f == 15 else [(1920, 480), (1300, 600), (640, 720), (-60, 960)]; B, *_ = limb(q, [4, 40, 70, 16], [2, 22, 60, 8]); c.fill(B, '#0a090b'); B, *_ = limb(q, [1, 10, 22, 2], [0, -4, -6, 0]); c.fill(B, '#f7f2e2'); speed_lines(c, math.atan2(230, -900), 60, 5, (200, 900), '#0a090b', f); img = over(img, c.layer())
            z = 1.02 + .0022 * f; s = shake(f, 5) if f == 15 else (0, 0); return cam2d(img, 960 + s[0], 560 + s[1], z)
        t = f - 17
        if f == 16 or t < 0: return np.ones((H, W, 3), f32) * rgb('#fbf7ea')
        lean = -.015 * ease_in(t / 12.) * 6; k = 1 + .03 * ease_in(t / 12.) * 3; c = Cel(); back_fig(c, 800 - 4 * t, 1226 + 40 * ease_in(t / 12.), 116 * k, 1.0 + .10 * ease(t / 8.), 0, lean, f // 2, 0., dict(tee='#17120f', teed='#17120f', skin='#17120f', skind='#17120f', trs='#17120f', trsd='#17120f', sash='#17120f', line='#3a322a')); a = c.layer()[..., 3]
        c2 = Cel(); back_fig(c2, 800 - 4 * t, 1226 + 40 * ease_in(t / 12.), 116 * k, 1.0 + .10 * ease(t / 8.), 0, lean, f // 2, 0., dict(tee='#e6d9bd', teed='#17120f', skin='#17120f', skind='#17120f', trs='#17120f', trsd='#17120f', sash='#17120f', line='#17120f')); L2 = c2.layer()
        m = sumi.inkify(a, .5, 3); lum_ = lum(L2[..., :3]) / np.maximum(L2[..., 3], 1e-3); white = smooth(.45, .6, lum_) * a; ink = np.clip(m - white * .95, 0, 1)
        y = grid()[1]; img = sumi.lay(self.paper, np.clip(ink, 0, 1)); gr = smooth(760, 1000, y + 60 * fbm(H, W, (200, 60), None, 3)) * .85; img = img * (1 - gr[..., None] * .8)
        ik, wet = self.pg.ink(min(4.2, .9 + t * 1.15)); img = sumi.lay(img, np.clip(ik, 0, 1), col='#b5101c', thin='#d8545a')
        dk = ease_in((f - 24) / 5.) if f > 24 else 0; img = img * (1 - dk) + rgb('#0a0708') * dk; return cam2d(img, 960, 540, 1.02 + .006 * t + (.02 if t < 2 else 0))

    # ---- B
    def acting(self, f):
        u = f - 30; h = lambda t0, a, d: a * math.exp(-(u - t0) / d) * math.sin((u - t0) * 1.25) if u >= t0 else 0.
        hitch = h(4, .030, 4) + h(15, .045, 5) + h(30, .07, 4) + h(35, .05, 3) + h(40, .05, 3) + h(45, .05, 6) + h(54, .04, 5)
        ph = (f // 2) * 1.7
        if f < 40: m = dict(o=.115, w=.25, ph=ph, grit=1) if f >= 34 else dict(o=.13, w=.20, ph=ph)
        elif f < 50: m = dict(o=.10, w=.27, ph=ph, grit=1)
        elif f < 58: q = ease((f - 49) / 8.); m = dict(o=.10 + .17 * q, w=.27 - .02 * q, ph=ph, grit=0 if f > 51 else 1)
        elif f < 60: m = dict(o=.06, w=.21, ph=ph)
        elif f < 65: m = dict(o=(.40, .46, .43, .42, .41)[f - 60], w=.335, ph=ph)
        elif f < 70: m = dict(o=(.22, .26, .27, .26, .25)[f - 65], w=.385, ph=ph)
        elif f < 75: m = dict(o=(.36, .42, .40, .39, .37)[f - 70], w=.34, ph=ph)
        else: m = dict(o=.185 - .01 * ((f // 2) % 2), w=.405 - .012 * ((f // 2) % 2), ph=ph * 1.3, grit=1)
        drop = max(0., m['o'] - .16) * .62; return hitch, m, drop

    def person(self, f):
        hitch, m, drop = self.acting(f); fd = f if f in (58, 59, 60, 61, 65, 70, 75) else f // 2 * 2; hitch2, m2, drop2 = self.acting(fd); c = Cel(W, H); cx, cy, s = 842 + hitch2 * 46, 566 - hitch2 * 26, 318; roll = -1.46 + hitch2 * .20
        g = (.10 + .2 * hitch2, -1.); lidv = .30 if f < 34 or 44 < f < 58 else (.0 if f in (34, 35, 58, 59) or 75 <= f < 80 else .42); H_ = zoro.draw(c, cx, cy, s, 0., .10 + drop2 * .25, roll, 'cry', key=1, f=fd, young=True, body='tee', broll=-1.46 + hitch2 * .04, swing=.5 * hitch2 * 8, mouth_=('box', dict(m2, cx=.0, top=-.565 + drop2 * .2)), pal=Y, hairpal=SY, drop=drop2, grav=g, seed=3, L=(.45, .6, .65), kick=.0, lid=lidv)
        # one long streak of brush ink on the shirt (no wound)
        P = [(1370, 250), (1520, 420), (1690, 560), (1990, 780)]; B, *_ = limb(P, [2, 26, 34, 24, 30, 10], [2, 20, 24, 34, 22, 8]); r = np.random.default_rng(3); C_ = Cel(W, H); C_.fill(B, '#a3131c'); splatter(C_, 1560, 540, 70, 34, 13, '#a3131c', 5, .7, 1.6); L_ = C_.layer(); t = sumi.tooth(H, W, 5); L_ = L_ * smooth(.18, .42, t + .22)[..., None]
        base = c.layer(); base = over(np.zeros((H, W, 3), f32), base), base[..., 3:4]; rgb_, a = base; ins = L_ * a; lay = np.dstack([rgb_ * (1 - ins[..., 3:4]) + ins[..., :3], a])
        # tears: clear streams to both temples, fed under the hand
        wt = Water(); u = fd * .11; big = 1.15 if f >= 58 else 1.
        for sg, p0 in ((1, 0.), (-1, .4)):
            E3 = lambda *q: H_.P3(np.array(q, float) * [sg, 1, 1]); E = lambda *q: H_.P(np.array(q, float) * [sg, 1])
            wt.stream(spl(np.vstack([E((.30, -.085), (.42, -.115), (.54, -.135), (.63, -.15)), E3((.72, -.16, .18), (.83, -.17, -.10), (.97, -.19, -.30))]), n=8), 25 * big, 5 + sg, u + p0)
            wt.stream(spl(np.vstack([E((.36, -.10), (.50, -.19), (.60, -.24)), E3((.69, -.28, .14), (.80, -.31, -.10))]), n=8), 15 * big, 21 + sg, u * 1.2 + p0 + .5)
            wt.stream(spl(np.vstack([E((.17, -.10), (.22, -.24), (.33, -.34), (.47, -.40), (.57, -.43)), E3((.67, -.46, .15), (.78, -.50, -.05))]), n=8), 20 * big, 8 + sg, u + p0 + .3)
            wt.stream(spl(np.vstack([E((.46, .00), (.58, .03), (.655, .05)), E3((.72, .07, .05), (.82, .08, -.25))]), n=8), 16 * big, 11 + sg, u * 1.1 + p0 + .6)
            wt.stream(spl(np.vstack([E((.12, -.14), (.14, -.30), (.22, -.43), (.36, -.52), (.46, -.56)), E3((.56, -.60, .10))]), n=8), 13 * big, 14 + sg, u + p0 + .8)
            wt.stream(E((.33, -.335), (.36, -.43), (.43, -.50), (.45, -.555)), 9 * big, 31 + sg, u * 1.3 + p0)
        tears = wt; kind = 'clench' if (f >= 58 or 33 < f < 40 or 44 < f < 50) else 'rest'; hd = cover_hand(kind, 1 + (f // 2) % 3 if kind == 'clench' else 0); hd = shifted(hd, hitch2 * 30, -hitch2 * 18 + (5 if kind == 'clench' else 0))
        return lay, tears, hd, H_

    def sea(self, f):
        x, y = grid(); im = ramp(np.clip(y / H, 0, 1), [(0, '#2f8f9c'), (.5, '#1f6f84'), (1, '#1b5a74')]); c = Cel(ss=1); d = f // 2
        for i in range(90):
            r = np.random.default_rng(i); xx, yy = r.uniform(-40, 520), r.uniform(-20, H + 20); L = r.uniform(16, 70); b = math.sin(d * .35 + i) * 7; c.line(spl([(xx - L, yy + b), (xx - L * .3, yy - 5 - b * .3), (xx + L * .3, yy + 4), (xx + L, yy - b)], n=6), r.uniform(2, 5), '#bdeef0' if i % 3 else '#0f4a62', taper=(.4, .4), alpha=.75, seed=i)
        return over(im, c.layer())

    def partB(self, f):
        lay, tears, hd, H_ = self.person(f); img = self.sea(f); P = self.planks[54:54 + H, 96:96 + W]
        bow = np.vstack([spl([(2000, -420), (1200, -300), (620, -90), (300, 120), (90, 340), (30, 540), (90, 740), (300, 960), (620, 1170), (1200, 1390), (2000, 1500)], n=10)]); m = poly_mask(bow)[..., None]
        sh = blur(shifted(m[..., 0], -14, 18), 14)[..., None]; img = img * (1 - .5 * sh); img = img * (1 - m) + P * m
        c = Cel(); c.line(bow, 64, '#7a5836', taper=(.01, .01), press=.1); c.line(bow + [9, 0], 16, '#e0bf90', taper=(.01, .01), press=.3); c.line(bow + [-34, 0], 5, INK, taper=(.01, .01)); c.line(bow + [33, 0], 5, INK, taper=(.01, .01)); g = c.layer(); img = cast(img, g, 20, 14, '#2a1a0e', .45, 10); img = over(img, g)
        grow = float(ease((f - 30) / 55.)); c = Cel(); r = np.random.default_rng(4)                     # pools of tears soak the wood
        for sg in (1, -1):
            o = H_.P3(np.array([(sg * .98, -.20, -.4)]))[0]
            for i in range(9): rr = (26 + 70 * grow) * r.uniform(.35, 1.0); q = o + A_(r.normal(0, 46), sg * -r.uniform(-30, 52)); a = np.linspace(0, 6.283, 18, endpoint=False); c.fill(spl(np.c_[q[0] + np.cos(a) * rr * 1.5 * (1 + .12 * np.sin(a * 3 + i)), q[1] + np.sin(a) * rr], True, n=4), '#4f371f')
        pl = c.layer()[..., 3]; pl = np.clip(pl, 0, 1); img = img * (1 - .46 * pl[..., None] * np.array([.8, 1., 1.15], f32)); ed = np.clip(pl - shifted(pl, 4, 5), 0, 1); img = img + (ed * .30)[..., None] * rgb('#fff6e0')
        img = cast(img, lay, -18, 26, '#2c1a10', .62, 9); img = over(img, lay); img = tears.put(img)
        hs = blur(shifted(hd[..., 3], -20, 26), 6)[..., None] * lay[..., 3:4]; img = img * (1 - hs * (1 - rgb('#a5583a'))); img = over(img, hd)
        for sg in (1, -1):                                                   # tears that break over the fingers
            if f >= 58 and sg > 0: img = Water().stream(spl([(866 + 14, 566 - 70), (900, 572 - 96), (926, 566 - 128)], n=8), 14, 3, f * .12).stream(spl([(870, 640), (902, 652), (930, 690)], n=8), 14, 4, f * .12 + .4).put(img)
        if f >= 60 and f < 90:                                               # beads shaken loose on each syllable
            wt = Water()
            for t0 in (60, 65, 70, 75, 84):
                t = f - t0
                if 0 <= t < 12:
                    for i in range(9): r = np.random.default_rng(t0 * 10 + i); sg = 1 if i % 2 else -1; a = -math.pi / 2 * sg + r.normal(0, .5); sp = r.uniform(16, 44) * 5 * (1 - math.exp(-t / 5.)); o = H_.P([(sg * .5, -.1)])[0]; wt.drop(o[0] + math.cos(a) * sp + 20, o[1] + math.sin(a) * sp, r.uniform(5, 13) * (1 - t / 14.) + 1.5, a)
            img = wt.put(img)
        k = (f - 30) / 60.; z = 1.075 + .075 * k + (.028 * math.exp(-(f - 60) / 4.) if f >= 60 else 0) + (.03 * math.exp(-(f - 30) / 2.5)) + (.016 * math.exp(-(f - 75) / 4.) if f >= 75 else 0); img = cam2d(img, 960 - 10 * k, 548 + 6 * k, z, -.035 + .065 * k, interp=cv2.INTER_CUBIC)
        # sun patch and the sword
        if f >= 61:
            q = float(lerp_keys(f, [(61, 0.), (64, .16), (70, 1.06), (73, .985), (76, 1.0), (90, 1.0)])); dy = 1180 * (1 - q); rot = .62 * (1 - q) + (.004 * math.sin(f * 2.1) if f > 74 else 0); gl = int(np.clip((f - 71) / 7., 0, 1.05) * 100) if f >= 71 else -1
            sl = sword_layer(gl); L = place(sl, 190 * (1 - q), dy, rot, 1 + .10 * (1 - q), 1500, 800)
            if 62 <= f <= 69: v = 60 if f in (64, 65, 66, 67) else 26; L = .55 * L + .45 * smear(L, 10, v, 9)
            sp = float(lerp_keys(f, [(61, -.2), (66, .25), (72, .88), (76, 1.0), (90, 1.02)])); A = A_(1180 - 260 * sp, -60); B = A_(330 + 420 * (1 - sp), 1130); d = B - A; nn = A_(-d[1], d[0]) / np.hypot(*d); c = Cel(ss=1); wv = 17
            c.fill(np.array([A + nn * wv * 1.5, A + d * .45 + nn * wv + [26, 0], B + nn * wv * .6, B - nn * wv * .6, A + d * .45 - nn * wv + [26, 0], A - nn * wv * 1.5]), '#000000'); s_ = blur(c.layer()[..., 3], 2.2)[..., None] * min(1., max(0., q)) ; img = img * (1 - s_ * (1 - rgb('#5f4658') * .95))
            s2 = blur(shifted(L[..., 3], -90, 70), 26)[..., None]; img = img * (1 - .28 * s2); L = rim2(L, 5, 6, '#fffbe6', .8, .8); img = over(img, L)
        if f >= 71:
            fl = float(lerp_keys(f, [(71, 0), (75, .25), (78, 1.25), (81, .75), (90, .55)])); tx, ty = 1380, 46; img = flare(img, tx, ty, 620, '#fff3d0', .62 * fl, .9 * min(1, fl), .06)
            for dd, rr, cc, aa in ((.42, 46, '#ffd9a0', .10), (.62, 88, '#a0e8d0', .07), (.80, 30, '#ffb0a0', .12), (1.15, 130, '#ffe6b8', .06), (1.4, 60, '#b8d8ff', .07)): x, y = tx + (800 - tx) * dd, ty + (620 - ty) * dd; a = smooth(rr, rr - 5, np.hypot(grid()[0] - x, grid()[1] - y)); img = img + (a * aa * fl)[..., None] * rgb(cc)
        warm = np.clip(1.0 - .0016 * max(0, f - 30), .85, 1); return img * np.array([1.0, .985, .955], f32)

    def frame(self, f): return np.maximum(self.partA(f) if f < 30 else self.partB(f), 0)
