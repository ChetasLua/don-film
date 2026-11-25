"""Franky: blue pompadour, iron nose, plate jaw, sunglasses, giant red forearms with the blue-white star."""
from .gang_kit import *

C = dict(skin='#e5ab80', sh='#a9694c', dk='#6d3d34', hair='#7fd0ea', hairS='#3a86b8', hairD='#1f4f84', iron='#aab2c0', ironS='#5d6578', ironD='#343a4b', red='#d33a2c', redS='#8d1c22', redD='#55101b', star='#3f74d6', starS='#26468f', white='#fbf6ea', whiteS='#b9b6cc', gl='#1b1a24', gl2='#4a5a80',
         shirt='#f0b43a', shirtS='#b56d22', tooth='#f7f3e6')
EX = dict(
    calm=E(eo=.62, el=.1, et=.18, pr=.26, bi=.06, bo=-.04, mw=.34, ml=.05), grin=E(eo=.66, el=.18, et=.22, pr=.24, bi=.07, bo=-.06, mw=.50, mu=.02, md=.20, ml=.14, mt=.98), worry=E(eo=.95, et=-.1, pr=.22, bi=-.10, bo=.06, mw=.40, mu=.03, md=.10, ml=-.07, mt=.98, wob=.5),
    roar=E(eo=.70, el=.2, et=.28, pr=.20, bi=.10, bo=-.10, mw=.50, mu=.08, md=.42, ml=.12, mt=.24, mb=.12, tg=.7), laugh=E(eo=0., arc=1, bi=-.04, bo=-.02, mw=.52, mu=.06, md=.46, ml=.14, mt=.28, tg=.8, tear=1., snot=.0), cry=E(eo=0., arc=-1, bi=-.10, bo=.06, mw=.46, mu=.08, md=.30, ml=-.06, mt=.4, mb=.2, tg=.4, wob=.8, tear=1., snot=1.))


def pomp(p, T, k, flop=0., wind=0.):
    """k = side the roll hangs over: a wave that rises from the hairline, rolls forward, sweeps back into blades"""
    s = T.s; h = k; w = wind; y = -flop
    BK = [(-h * .60, -.24, .1), (-h * .80, -.50, -.1), (-h * 1.06 - w * .2, -.60, -.2), (-h * .90, -.76, -.2), (-h * 1.26 - w * .3, -1.02, -.25), (-h * .96, -1.06, -.2), (-h * 1.16 - w * .3, -1.44, -.25), (-h * .72, -1.34, -.2), (-h * .60 - w * .2, -1.66, -.2), (-h * .10, -1.48, -.1), (h * .5, -1.2, 0), (h * .66, -.5, .1), (h * .62, -.24, .1), (0, -.44, .6)]
    Q = p.shape(BK, C['hair'], 1., co=(0, 2, 3, 4, 5, 6, 7, 8, 9, 12), n=7); p.shade([(-h * .1, -1.7, 0), (-h * .30, -1.2, .2), (-h * .50, -.8, .3), (-h * .46, -.3, .4), (h * 1., -.2, .3), (h * 1., -1.6, 0)], C['hairS'], Q)
    for a_, b_, c_ in (((-h * .56, -.50, .2), (-h * .76, -.62, 0), (-h * .94, -.64, -.1)), ((-h * .50, -.74, .3), (-h * .80, -.92, 0), (-h * 1.08, -1.0, -.1)), ((-h * .40, -.98, .3), (-h * .74, -1.18, 0), (-h * .98, -1.36, -.1)), ((-h * .24, -1.18, .3), (-h * .42, -1.36, 0), (-h * .56, -1.52, -.1))): p.line([a_, b_, c_], .5, taper=(.1, .6))
    R_ = [(-h * .60, -.34, .50), (-h * .66, -.62, .55), (-h * .58, -.98 + y * .4, .60), (-h * .24, -1.32 + y * .8, .70), (h * .26, -1.46 + y, .80), (h * .74, -1.34 + y, .95), (h * 1.08, -1.02 + y, 1.10), (h * 1.14 + w * .1, -.68 + y, 1.15), (h * .96, -.46 + y, 1.12), (h * .74, -.48 + y * .9, 1.05), (h * .70, -.62 + y * .8, 1.0), (h * .60, -.58 + y * .4, .9), (h * .34, -.40, .85), (0, -.46, .9), (-h * .30, -.40, .8)]
    Q = p.shape(R_, C['hair'], 1.1, co=(0, 9, 10, 12, 13), n=8)
    p.shade([(-h * .7, -.30, .6), (-h * .46, -.60, .7), (-h * .05, -.86 + y, .9), (h * .34, -.94 + y, 1.0), (h * .66, -.90 + y, 1.05), (h * .90, -.74 + y, 1.1), (h * 1.2, -.64 + y, 1.1), (h * 1.1, -.3, 1.), (h * .3, -.2, .9), (-h * .4, -.2, .8)], C['hairS'], Q); p.shade([(-h * .40, -.3, .8), (-h * .10, -.56, .9), (h * .34, -.66 + y * .6, 1.0), (h * .66, -.74 + y * .8, 1.0), (h * .80, -.56 + y, 1.05), (h * .98, -.40, 1.1), (h * .4, -.2, .9)], C['hairD'], Q)
    for a_, b_, c_ in (((-h * .44, -.44, .8), (-h * .46, -.80, .75), (-h * .12, -1.24 + y * .7, .75)), ((-h * .14, -.48, .9), (-h * .16, -.88 + y * .4, .85), (h * .26, -1.36 + y, .85)), ((h * .16, -.50, .92), (h * .20, -.90 + y * .6, .95), (h * .68, -1.24 + y, 1.0)), ((h * .42, -.50, .95), (h * .52, -.86 + y * .8, 1.05), (h * .96, -1.0 + y, 1.1))): p.line([a_, b_, c_], .48, taper=(.1, .6))
    p.line([(h * .98, -.52 + y, 1.14), (h * 1.04, -.72 + y, 1.15), (h * .92, -.86 + y, 1.12), (h * .80, -.76 + y, 1.1), (h * .84, -.64 + y, 1.1)], .6)
    for a_, b_, c_ in (((-h * .34, -1.04 + y * .6, .75), (-h * .10, -1.24 + y * .8, .8), (h * .20, -1.34 + y, .85)), ((h * .40, -1.30 + y, .95), (h * .62, -1.26 + y, 1.0), (h * .80, -1.12 + y, 1.1))): p.f.line(T.P3(spn([a_, b_, c_], False)), s * .05, '#e2f9ff', taper=(.3, .5))


def draw(T, li, e, f=0, seed=500, shades='up', flip=None, flop=0., wind=0.):
    """flip 0 = glasses up in the hair, 1 = down over the eyes"""
    s = T.s; st = Stack(); c0 = T.c; box = (c0[0] - s * 2.0, c0[1] - s * 2.3, c0[0] + s * 2.0, c0[1] + s * 2.2); n = lambda sd: Part(T, box, sd); L = -1 if T.yaw >= -.05 else 1; k = -L
    if flip is None: flip = 1. if shades == 'down' else 0.
    p = n(seed); j = e['md'] * .20
    NK = p.fill([(-.56, .60, -.1), (-.64, 1.06, -.1), (-.86, 1.40, -.2), (0, 1.56, -.2), (.86, 1.40, -.2), (.64, 1.06, -.1), (.56, .60, -.1)], C['skin'], co=(0, 6)); p.shade([(-.7, .6, 0), (-.62, 1.16, 0), (-.1, 1.34, .1), (.05, 1.6, 0), (1., 1.6, 0), (.7, .6, 0)], C['sh'], NK); p.shade([(-.7, .6, 0), (-.5, 1.14, 0), (0, 1.22, .1), (.7, 1.26, 0), (.7, .6, 0)], C['dk'], NK); p.line([(-.58, .8, -.1), (-.65, 1.06, -.1), (-.90, 1.36, -.2)], .9, taper=(.1, .5)); p.line([(.58, .8, -.1), (.65, 1.06, -.1), (.90, 1.36, -.2)], .9, taper=(.1, .5)); p.line([(-.36, 1.16), (-.14, 1.44)], .5); p.line([(.36, 1.16), (.14, 1.44)], .5)
    fc = [(-.40, -.78), (-.62, -.52), (-.67, -.16), (-.68, .10), (-.755, .36), (-.75, .70 + j * .6), (-.58, .98 + j), (-.28, 1.09 + j), (0, 1.04 + j), (.28, 1.09 + j), (.58, .98 + j), (.75, .70 + j * .6), (.755, .36), (.68, .10), (.67, -.16), (.62, -.52), (.40, -.78)]
    F = p.shape(fc, C['skin'], 1., limb=True, co=(8,))
    p.shade([(k * .9, -.7), (k * .48, -.40), (k * .52, -.10), (k * .46, .12), (k * .56, .40), (k * .54, .68), (k * .36, .92), (k * .10, 1.06 + j), (k * .3, 1.4), (k * 1.1, 1.1)], C['sh'], F); p.shade([(-.9, -.9), (-.8, -.24), (-.4, -.20), (0, -.28), (.4, -.20), (.8, -.24), (.9, -.9)], C['sh'], F); p.shade([(k * .95, .3), (k * .70, .42), (k * .68, .74), (k * .50, .98), (k * .25, 1.09 + j), (k * .9, 1.2)], C['dk'], F, .75)
    for sd in (-1, 1):                                       # plate seams, sideburns, ears
        p.line([(sd * .58, .16), (sd * .50, .40), (sd * .54, .70), (sd * .40, .96 + j)], .42); p.line([(sd * .57, .50), (sd * .47, .51)], .34); p.line([(sd * .56, .62), (sd * .47, .63)], .34)
        Q = p.shape([(sd * .62, -.50), (sd * .70, -.20), (sd * .69, .16), (sd * .60, .30), (sd * .58, -.02), (sd * .52, -.34)], C['hairS'], .7, co=(3,), z=.25)
        Q = p.shape([(sd * .68, -.14, .05), (sd * .84, -.20, -.02), (sd * .87, .02, -.02), (sd * .76, .22, .02), (sd * .67, .16, .06)], C['skin'], .85); p.shade(Q + [k * s * .04, s * .05], C['sh'], Q); p.line([(sd * .74, -.08, .05), (sd * .81, -.08, .03), (sd * .78, .08, .05)], .35)
    p.line([(0, .96 + j), (.0, 1.04 + j)], .6); p.ticks((k * .30, .86 + j * .6), (k * .50, .72), 6, .26, .12); p.ticks((-k * .56, .30), (-k * .42, .34), 5, .24, .10)
    for x in np.linspace(-.26, .26, 9): p.line([(x, .955 + j + .02 * math.cos(x * 9)), (x * 1.04, .995 + j)], .22, a=.7)
    for sd in (-1, 1):
        eye(p, sd * .30, .0, .16, .115, e, sd, C['skin'], C['sh'], lidw=1.1); p.line([(sd * .16, .15), (sd * .28, .18), (sd * .42, .13)], .3)
        brow(p, [(sd * .12, -.15 + e['bi']), (sd * .30, -.235 + (e['bi'] + e['bo']) * .4), (sd * .54, -.20 + e['bo'])], [.07, .10, .03], C['hairS'], lw=.5)
    if e['bi'] > .05: p.line([(-.06, -.16), (-.04, -.06)], .45); p.line([(.06, -.16), (.04, -.06)], .45)
    mouth(p, 0, .66, e, lines=7 if e['mt'] > .9 else 0, w=.95)
    zt = .80
    N = [(-.085, -.02, .74), (.085, -.02, .74), (.20, .32, zt + .10), (.235, .44, zt + .10), (.12, .52, zt + .22), (0, .535, zt + .26), (-.12, .52, zt + .22), (-.235, .44, zt + .10), (-.20, .32, zt + .10)]
    Q = p.T.P3(spn(N, True, (0, 1, 3, 7), 8)); p.shade(Q + [k * s * .06, s * .10], C['dk'], F, .8); p.shape(Q, C['iron'], .9); p.shade([(k * .02, -.1, .8), (k * .05, .30, zt + .26), (k * .02, .6, zt + .2), (k * .5, .6, zt), (k * .4, -.1, .7)], C['ironS'], Q); p.shade([(-.3, .42, zt + .1), (0, .46, zt + .26), (.3, .42, zt + .1), (.2, .7, zt), (-.2, .7, zt)], C['ironD'], Q); p.hl([(-k * .05, .02, .8), (-k * .02, .02, .8), (-k * .05, .34, zt + .24), (-k * .12, .36, zt + .2)], '#f4f8ff', 1., Q)
    for x, y in ((-.16, .34), (.16, .34), (0, .06)): p.f.fill(ell(*T.at(x, y, zt + .2), s * .020, s * .020, 10), C['ironD'])
    if e['tear'] > 0:
        ph = f // 2 * .8; t = e['tear']
        for sd in (-1, 1): rope(p, [(sd * .34, .08), (sd * .46, .24), (sd * .49, .30 + .34 * t), (sd * (.47 + .02 * math.sin(ph)), .30 + .60 * t)], [s * .035, s * .08, s * .09, s * .16])
    if e['snot'] > 0: rope(p, [(.08, .50), (.10, .60), (.07, .72)], [s * .05, s * .06, s * .08], dz=.2)
    st.add(p, li); p = n(seed + 40); pomp(p, T, k, flop, wind); st.add(p, li); p = n(seed + 60)
    y = -.50 + (.52) * flip; zz = .92 - .06 * flip
    for sd in (-1, 1):
        G = [(sd * .05, y - .10, zz), (sd * .30, y - .17, zz - .05), (sd * .56, y - .13, zz - .25), (sd * .57, y + .04, zz - .25), (sd * .40, y + .15, zz - .12), (sd * .16, y + .13, zz), (sd * .06, y + .02, zz)]; Q = p.shape(G, C['gl'], .7); p.shade([(sd * .30 - .16, y - .2, zz), (sd * .30 - .05, y - .2, zz), (sd * .30 - .18, y + .2, zz), (sd * .30 - .30, y + .2, zz)], C['gl2'], Q); p.shade([(sd * .30 + .05, y - .2, zz), (sd * .30 + .09, y - .2, zz), (sd * .30 - .06, y + .2, zz), (sd * .30 - .10, y + .2, zz)], C['gl2'], Q)
        p.line([(sd * .56, y - .08, zz - .25), (sd * .74, y - .06, .05)], .7)
    p.line([(-.06, y - .06, zz), (0, y - .10, zz + .03), (.06, y - .06, zz)], .8)
    st.add(p, li); return st.a


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light_='day', f=0, **kw):
    li = kw.pop('light', light_); li = light(li) if isinstance(li, str) else li; sq = kw.pop('sq', (1., 1.))
    return draw(Rig(cx, cy, s, yaw, pitch, roll, kw.pop('persp', .12), sx=sq[0], sy=sq[1]), li, pick(EX, expr), f, **kw)


def star(cx, cy, R, rot=0., ri=.42, n=5):
    a = np.arange(2 * n) * math.pi / n - math.pi / 2 + rot; r = np.where(np.arange(2 * n) % 2 == 0, R, R * ri); return np.c_[cx + np.cos(a) * r, cy + np.sin(a) * r]


def forearm(p, w, e, wd0, wd1, k, seed=0, half=0, sc=100, knuck=1.):
    """one giant forearm: wrist end at e (fist side), elbow at w (screen px).  half -1/1: the half of a star printed on that edge"""
    w, e = np.asarray(w, float), np.asarray(e, float); d = e - w; L = np.hypot(*d); d /= L; n = np.array([-d[1], d[0]]); P = lambda u, v: w + d * u * L + n * v
    A = spn([P(-.06, -wd0 * .34), P(.12, -wd0 * .50), P(.45, -wd1 * .56), P(.80, -wd1 * .54), P(.98, -wd1 * .46), P(1.0, 0), P(.98, wd1 * .46), P(.80, wd1 * .54), P(.45, wd1 * .56), P(.12, wd0 * .50), P(-.06, wd0 * .34)], True, (4, 6), 10)
    Q = p.shape(A, C['red'], 1.15); sv = n * k * wd1 * .30 if (n @ [k, .4]) > 0 else -n * k * wd1 * .30
    p.shade(Q + n * (wd1 * .34 * (1 if (n @ [k, .5]) > 0 else -1)), C['redS'], Q); p.shade(Q + n * (wd1 * .70 * (1 if (n @ [k, .5]) > 0 else -1)), C['redD'], Q, .8)
    for u in (.10, .94):                                   # iron rings with bolts
        B = np.array([P(u - .03, -wd1 * .56), P(u + .03, -wd1 * .56), P(u + .045, 0), P(u + .03, wd1 * .56), P(u - .03, wd1 * .56), P(u - .015, 0)]); B = spn(B, True, (0, 1, 3, 4), 8); p.f.clip(Q); p.f.fill(B, C['iron']); p.f.fill(B + sv * 1.3, C['ironS']); p.f.unclip(); p.l.line(np.array([P(u - .03, -wd1 * .54), P(u - .012, 0), P(u - .03, wd1 * .54)]), .5 * p.k, INK); p.l.line(np.array([P(u + .03, -wd1 * .54), P(u + .045, 0), P(u + .03, wd1 * .54)]), .5 * p.k, INK)
        for v in (-.32, 0., .32): p.f.fill(ell(*P(u + .006, wd1 * v), wd1 * .035, wd1 * .035, 10), C['ironD'])
    hi = -1 if (n @ [k, .5]) > 0 else 1; p.f.clip(Q); p.f.line(np.array([P(.18, wd1 * .40 * hi), P(.5, wd1 * .44 * hi), P(.84, wd1 * .40 * hi)]), wd1 * .06, '#ff9a7a', taper=(.3, .5)); p.f.unclip()
    if half:
        c = P(.54, wd1 * .54 * half); S = star(c[0], c[1], wd1 * .52, math.atan2(d[1], d[0]) + math.pi / 2); p.f.clip(Q); p.l.ctx.save(); p.l._path(Q); p.l.ctx.clip()
        p.f.fill(star(c[0], c[1], wd1 * .64, math.atan2(d[1], d[0]) + math.pi / 2), C['white']); p.f.fill(S, C['star']); p.f.fill(S + sv * .8, C['starS'], .0); kit.outline(p.l, star(c[0], c[1], wd1 * .64, math.atan2(d[1], d[0]) + math.pi / 2), .6 * p.k, INK, seed); kit.outline(p.l, S, .5 * p.k, INK, seed + 1); p.f.unclip(); p.l.ctx.restore()
    # the hand: back plate, four finger blocks, thumb
    if knuck:
        fw = wd1 * .92; o = e + d * wd1 * .02; R2 = lambda u, v: o + d * u * fw + n * v * fw; lit = 1 if (n @ [k, .5]) > 0 else -1
        for j in range(4):
            v0 = -.50 + .25 * j; F_ = spn([R2(.30, v0 + .015), R2(.62, v0 + .01), R2(.74, v0 + .06), R2(.76, v0 + .125), R2(.74, v0 + .19), R2(.62, v0 + .24), R2(.30, v0 + .235)], True, (0, 6), 6); q = p.shape(F_, C['red'], .9); p.shade(q + n * fw * .10 * lit + d * fw * .0, C['redS'], q); p.l.line(np.array([R2(.56, v0 + .03), R2(.585, v0 + .125), R2(.56, v0 + .22)]), .45 * p.k, INK, taper=(.2, .2)); p.f.clip(q); p.f.line(np.array([R2(.66, v0 + .05), R2(.70, v0 + .11), R2(.67, v0 + .17)]), fw * .03, '#ffae90'); p.f.unclip()
        BP = spn([R2(-.02, -.48), R2(.16, -.54), R2(.40, -.53), R2(.46, -.28), R2(.43, 0), R2(.46, .28), R2(.40, .53), R2(.16, .54), R2(-.02, .48)], True, (0, 8), 7); q = p.shape(BP, C['red'], 1.1); p.shade(q + n * fw * .34 * lit, C['redS'], q); p.shade(q + n * fw * .70 * lit, C['redD'], q, .7)
        for j in range(4): v0 = -.375 + .25 * j; p.l.line(np.array([R2(.18, v0 * .94), R2(.42, v0)]), .5 * p.k, INK, taper=(.6, .1)); p.f.fill(ell(*R2(.35, v0), fw * .045, fw * .045, 14), C['iron']); p.l.line(np.vstack([ell(*R2(.35, v0), fw * .045, fw * .045, 12)] * 1), .25 * p.k, INK)
        th = spn([R2(.10, -.56 * lit), R2(.36, -.66 * lit), R2(.60, -.60 * lit), R2(.66, -.44 * lit), R2(.52, -.38 * lit), R2(.34, -.44 * lit)], True, (0,), 6); q = p.shape(th, C['red'], .9); p.f.clip(q); p.f.line(np.array([R2(.28, -.60 * lit), R2(.50, -.57 * lit)]), fw * .03, '#ffae90'); p.f.unclip()
    return Q


def body_g1(T, li, seed=560):
    p = Part(T, (T.c[0] - T.s * 4.6, T.c[1] - T.s * 3.6, T.c[0] + T.s * 4.6, T.c[1] + T.s * 3.8), seed)
    B = p.shape([(-.62, 1.10), (-1.3, 1.36), (-1.9, 1.8), (-1.6, 3.2), (-1.1, 4.6), (1.1, 4.6), (1.6, 3.2), (1.9, 1.8), (1.3, 1.36), (.62, 1.10)], C['skin'], 1.1, z=-.5); p.shade([(.1, 1.1), (.4, 1.9), (.2, 3.0), (.4, 4.8), (2.2, 4.8), (2.2, 1.2)], C['sh'], B, z=-.5); p.line([(0, 1.6), (.03, 2.4), (0, 4.4)], .5, z=-.5); p.line([(-1.2, 2.3), (-.5, 2.62), (-.05, 2.45)], .6, z=-.5); p.line([(1.2, 2.3), (.5, 2.62), (.05, 2.45)], .6, z=-.5)
    for sd in (-1, 1):
        p.line([(sd * .9, 1.4), (sd * .7, 2.0), (sd * .95, 2.6)], .45, z=-.5); Q = p.shape([(sd * 1.5, 1.35), (sd * 2.0, 1.15), (sd * 2.6, 1.35), (sd * 2.9, 1.95), (sd * 2.7, 2.6), (sd * 2.2, 2.85), (sd * 1.65, 2.5), (sd * 1.4, 1.9)], C['red'], 1.1, z=-.3); p.shade([(sd * 2.2 + .2, 1.0), (sd * 2.2 + .1, 1.9), (sd * 2.2 - .1, 3.0), (sd * 2.2 + 1.4, 3.0), (sd * 2.2 + 1.4, 1.0)], C['redS'], Q, z=-.3)
        S = star(*T(np.array([[sd * 2.2, 1.95]]), -.2)[0], T.s * .44, T.roll); p.shape(star(*T(np.array([[sd * 2.2, 1.95]]), -.2)[0], T.s * .56, T.roll), C['white'], .5); p.shape(S, C['star'], .4)
    return p.done(li)


def g1(f):
    x, y = 1330, 880; t, b = g1_t(f, 3); ro = crown_roll(x, y); li = light('back', math.atan2(500 - y, 960 - x)); e = pick(EX, ('worry', 'laugh', t))
    T = Rig(x, y, 138, -.10, -.08 - .06 * b, ro - .03 * b, .14, sx=1 - .02 * b, sy=1 + .03 * b); st = Stack(); st.add(body_g1(T, li)); return over4(st.a, draw(T, li, e, f, flop=.06 * b))
