"""Brook: skull with the crack, enormous afro, top hat (hat='top') or little crown (hat='crown'), feather boa, violin."""
from .gang_kit import *

C = dict(bone='#f1ebdc', sh='#b3a8b8', dk='#70647f', hole='#15101a', hair='#191520', hair2='#2c2636', hat='#232030', hatS='#131019', band='#4a78c8', boa='#f3ecef', boaS='#b79fc0', coat='#26222f', coatS='#14111a', crav='#3f76c9', cravS='#234685', crown='#f2c84a', crownS='#a8722a', wood='#b5622a', woodS='#6f3218', woodD='#3b1a12')
EX = dict(
    calm=E(md=.04, ml=.04), sing=E(md=.26, ml=.05), worry=E(md=.10, ml=-.05, wob=.6, tear=.0), laugh=E(md=.36, ml=.10, tear=.35), hush=E(md=.02, ml=.02, tear=.5))


def skull(p, T, k, e, f=0, seed=0, glasses=False):
    s = T.s; j = e['md']
    # jaw first
    J = [(-.50, .52), (-.50, .74 + j * .7), (-.38, .98 + j), (-.16, 1.08 + j), (.16, 1.08 + j), (.38, .98 + j), (.50, .74 + j * .7), (.50, .52)]
    QJ = p.shape(J, C['bone'], 1., co=(0, 7), limb=True); p.shade([(k * .6, .4), (k * .24, .6), (k * .26, .90 + j), (k * .05, 1.10 + j), (k * .8, 1.3 + j)], C['sh'], QJ); p.shade([(-.6, .4), (-.45, .86 + j * .3), (0, .92 + j * .5), (.45, .86 + j * .3), (.6, .4)], C['hole'] if j > .08 else C['sh'], QJ)
    u = np.linspace(-1, 1, 15)
    if j > .03:
        LT = T(np.r_[np.c_[.40 * u, .80 + j - .05 * (1 - u * u) * 0 + .05 * (1 - np.abs(u) ** 2) - .06 * np.abs(u) ** 2], np.c_[.41 * u, .95 + j * .98 + .05 * (1 - u * u) - .05 * u * u][::-1]]); p.fill(LT, C['bone']); p.ol(LT, .45)
        for a in np.linspace(-.8, .8, 7): p.line([(.40 * a, .80 + j + .05 * (1 - a * a) - .06 * a * a), (.41 * a, .95 + j + .05 * (1 - a * a) - .05 * a * a)], .4)
    # cranium + cheek
    K = [(-.38, .72), (-.50, .52), (-.58, .40), (-.70, .20), (-.72, -.14), (-.76, -.50), (-.60, -.90), (-.26, -1.08), (.26, -1.08), (.60, -.90), (.76, -.50), (.72, -.14), (.70, .20), (.58, .40), (.50, .52), (.38, .72), (0, .77)]
    Q = p.shape(K, C['bone'], 1., limb=True, co=(0, 15)); p.shade([(k * .9, -1.1), (k * .36, -.9), (k * .56, -.4), (k * .52, -.1), (k * .60, .12), (k * .40, .34), (k * .44, .56), (k * .16, .84), (k * .9, .9)], C['sh'], Q); p.shade([(k * .95, -.6), (k * .68, -.5), (k * .66, -.1), (k * .68, .14), (k * .52, .40), (k * .9, .5)], C['dk'], Q, .75); p.shade([(-.9, -1.2), (-.8, -.52), (-.3, -.62), (0, -.56), (.3, -.62), (.8, -.50), (.9, -1.2)], C['sh'], Q)
    for sd in (-1, 1):                                       # sockets, cheekbones
        O = [(sd * .12, .0), (sd * .16, -.18), (sd * .32, -.25 + e['ml'] * -.2), (sd * .50, -.17), (sd * .53, .03), (sd * .42, .19), (sd * .24, .18)]; Qo = p.fill(O, C['hole']); p.ol(Qo, .8); p.shade(off(O, -k * .035, -.02, .80) if 0 else [(sd * .20, .10), (sd * .36, .13), (sd * .48, .04), (sd * .44, .16), (sd * .26, .16)], '#3a3146', Qo)
        p.line([(sd * .60, .26), (sd * .48, .34), (sd * .40, .46), (sd * .42, .58)], .5); p.line([(sd * .10, -.30 + e['wob'] * -.08), (sd * .32, -.37), (sd * .56, -.28 + e['wob'] * .06)], .5)
        p.shade([(sd * .62, .20), (sd * .44, .30), (sd * .40, .50), (sd * .48, .52), (sd * .60, .38)], C['sh'], Q)
    N = [(0, .22), (.11, .36), (.10, .45), (.03, .42), (0, .39), (-.03, .42), (-.10, .45), (-.11, .36)]; Qn = p.fill(N, C['hole']); p.ol(Qn, .5)
    UT = T(np.r_[np.c_[.42 * u, .565 + .05 * (1 - u * u) - .03 * u * u], np.c_[.41 * u, .715 + .06 * (1 - u * u) - .06 * u * u][::-1]]); p.fill(UT, C['bone']); p.shade(UT + [k * s * .14, s * .02], C['sh'], UT); p.ol(UT, .45)
    for a in np.linspace(-.8, .8, 8): p.line([(.42 * a, .565 + .05 * (1 - a * a) - .03 * a * a), (.41 * a, .715 + .06 * (1 - a * a) - .06 * a * a)], .4)
    if j <= .03:
        LT = T(np.r_[np.c_[.41 * u, .725 + .06 * (1 - u * u) - .06 * u * u], np.c_[.40 * u, .87 + .06 * (1 - u * u) - .06 * u * u][::-1]]); p.fill(LT, C['bone']); p.ol(LT, .45); [p.line([(.41 * a, .725 + .06 * (1 - a * a) - .06 * a * a), (.40 * a, .87 + .06 * (1 - a * a) - .06 * a * a)], .4) for a in np.linspace(-.8, .8, 7)]
    p.line([(.46, -.98), (.38, -.80), (.48, -.68), (.34, -.54), (.40, -.44), (.28, -.34)], .6, taper=(.05, .6), co=(1, 2, 3, 4), n=3); p.line([(.48, -.68), (.58, -.62)], .35)      # the crack over his left eye
    p.ticks((k * .30, -.80), (k * .48, -.42), 6, .26, .12); p.ticks((k * .30, .86 + j * .8), (k * .42, .70 + j * .6), 4, .24, .09)
    if glasses:
        for sd in (-1, 1): G = p.shape([(sd * .06, -.10), (sd * .20, -.24), (sd * .40, -.24), (sd * .60, -.10), (sd * .50, .10), (sd * .30, .22), (sd * .12, .10)], '#c9d6ee', .7)
    if e['tear'] > 0:
        t = e['tear']
        for sd in (-1, 1): rope(p, [(sd * .30, .17), (sd * .36, .30), (sd * .40, .34 + .40 * t)], [s * .04, s * .075, s * .06, s * .09])
    return Q


def afro(p, T, k, seed, wind=0., squash=0.):
    s = T.s; A = [(-.66, .30), (-1.10, .34), (-1.50 + wind * .1, .0), (-1.74 + wind * .2, -.52), (-1.62 + wind * .3, -1.12), (-1.16 + wind * .35, -1.62 + squash), (-.5 + wind * .35, -1.90 + squash), (.2 + wind * .35, -1.92 + squash), (.9 + wind * .32, -1.74 + squash), (1.46 + wind * .28, -1.26), (1.70 + wind * .2, -.66), (1.56 + wind * .1, -.04), (1.14, .34), (.66, .30)]
    C0 = T(spn(A, True, (), 8), -.2); Q = scallop(C0, s * .24, s * .10, seed); p.shape(Q, C['hair'], 1.)
    for dx, dy, kk, col in ((-.16 * k, -.10, .84, C['hair2']), (-.34 * k, -.26, .52, '#3b3349')): p.shade(scallop(T(spn(off(A, dx, dy, kk), True, (), 8), -.2), s * .22, s * .09, seed + 4), col, Q)
    curls(p, off(A, 0, -.02, .92), 60, .13, '#51475f', seed + 2, -.2, .42); curls(p, off(A, 0, 0, .96), 44, .095, '#08060a', seed + 7, -.2, .42); return Q


def tophat(p, T, k, x=.30, y=-1.72, tilt=.16, sc=1.):
    c, s_ = math.cos(tilt), math.sin(tilt); q = lambda L: [(x + (u * c - v * s_) * sc, y + (u * s_ + v * c) * sc, z) for u, v, z in L]
    Q = p.shape(q([(-.40, .02, .1), (-.38, -.40, .1), (-.46, -.86, .1), (0, -.93, .2), (.46, -.86, .1), (.38, -.40, .1), (.40, .02, .1), (0, .08, .4)]), C['hat'], 1., co=(0, 2, 4, 6)); p.shade(q([(k * .06, .2, .3), (k * .12, -.95, .2), (k * .7, -.95, 0), (k * .7, .2, 0)]), C['hatS'], Q); p.line(q([(-.44, -.84, .12), (0, -.76, .4), (.44, -.84, .12)]), .4)
    B = p.shape(q([(-.395, -.02, .12), (-.39, -.17, .12), (0, -.11, .42), (.39, -.17, .12), (.395, -.02, .12), (0, .05, .42)]), C['band'], .5, co=(0, 1, 3, 4)); p.shade(q([(k * .06, .2, .4), (k * .10, -.3, .4), (k * .6, -.3, .1), (k * .6, .2, .1)]), '#23406f', B)
    R_ = p.shape(q([(-.74, .04, 0), (-.50, -.06, .2), (-.40, .0, .25), (0, .07, .5), (.40, .0, .25), (.50, -.06, .2), (.74, .04, 0), (.42, .16, .4), (0, .20, .6), (-.42, .16, .4)]), C['hat'], 1., co=(0, 6)); p.shade(q([(-.8, .10, 0), (0, .13, .5), (.8, .10, 0), (0, .3, .5)]), C['hatS'], R_)


def crown(p, T, k, x=.10, y=-1.86, tilt=.1):
    c, s_ = math.cos(tilt), math.sin(tilt); q = lambda L: [(x + u * c - v * s_, y + u * s_ + v * c, z) for u, v, z in L]
    P = [(-.46, .06, .1), (-.56, -.44, .1), (-.28, -.18, .2), (-.18, -.58, .25), (0, -.22, .3), (.18, -.58, .25), (.28, -.18, .2), (.56, -.44, .1), (.46, .06, .1), (0, .14, .35)]; Q = p.shape(q(P), C['crown'], .9, co=tuple(range(9)), n=4); p.shade(q([(k * .05, .3, .3), (k * .12, -.6, .2), (k * .8, -.6, 0), (k * .8, .3, 0)]), C['crownS'], Q)
    for u, v in ((-.56, -.46), (-.18, -.60), (.18, -.60), (.56, -.46)): w = q([(u, v, .2)])[0]; p.shape(ell(*T.at(*w), T.s * .06, T.s * .06, 12), '#e0523f', .4)


def boa(p, T, seed=0, f=0, z=.1):
    for sd in (-1, 1):
        P = T(spn([(sd * .30, 1.10), (sd * .80, 1.18), (sd * 1.15, 1.6), (sd * 1.1, 2.3), (sd * .95, 3.2)], False, (), 8), z); B = scallop(band_poly(P, [T.s * v for v in (.34, .46, .50, .46, .42)]), T.s * .20, T.s * .09, seed + sd); p.shape(B, C['boa'], .8); p.shade(B + [T.s * .10, T.s * .10], C['boaS'], B)
        for i in range(0, len(P), 5): p.l.line(np.array([P[i], P[i] + [T.s * .06, -T.s * .10]]), .35 * p.k, '#8b6f99')


def draw(T, li, e, f=0, seed=700, hat='top', wind=0., squash=0., neck=True, hatdy=0., hats=1.):
    s = T.s; st = Stack(); c0 = T.c; box = (c0[0] - s * 2.6, c0[1] - s * 3.2, c0[0] + s * 2.6, c0[1] + s * 2.2); n = lambda sd: Part(T, box, sd); L = -1 if T.yaw >= -.05 else 1; k = -L
    p = n(seed); afro(p, T, k, seed, wind, squash); st.add(p, li); p = n(seed + 20)
    if hat == 'top': tophat(p, T, k, y=-1.72 + .10 * (hats < 1) + hatdy, sc=hats)
    elif hat == 'crown': crown(p, T, k)
    st.add(p, li); p = n(seed + 40)
    if neck:
        for y in (.98, 1.12, 1.26): p.shape([(-.14, y), (-.18, y + .06), (-.14, y + .13), (.14, y + .13), (.18, y + .06), (.14, y)], C['sh'], .5, z=-.1)
    skull(p, T, k, e, f, seed, False); st.add(p, li); p = n(seed + 60); fr = [(-.74, -.36, .3), (-.84, -.70, .2), (-.50, -1.02, .5), (0, -1.12, .6), (.50, -1.02, .5), (.84, -.70, .2), (.74, -.36, .3), (.60, -.62, .5), (.30, -.82, .66), (0, -.86, .7), (-.30, -.82, .66), (-.60, -.62, .5)]
    Q = p.fill(scallop(T.P3(spn(fr, True, (), 6)), s * .17, s * .07, seed + 1), C['hair']); curls(p, fr, 14, .08, '#51475f', seed + 12, .4, .4); p.ol(T.P3(spn([fr[0]] + fr[7:][::-1] + [fr[6]], False, (), 8))[:1].repeat(5, 0), .01) if 0 else None
    st.add(p, dict(li, rim=(), air=None)); return st.a


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light_='day', f=0, **kw):
    li = kw.pop('light', light_); li = light(li) if isinstance(li, str) else li; sq = kw.pop('sq', (1., 1.))
    return draw(Rig(cx, cy, s, yaw, pitch, roll, kw.pop('persp', .10), r=(.78, 1.1, .74), sx=sq[0], sy=sq[1]), li, pick(EX, expr), f, **kw)


def g1(f):
    x, y = 270, 300; t, b = g1_t(f, 4); ro = crown_roll(x, y); li = light('back', math.atan2(500 - y, 960 - x)); e = pick(EX, ('worry', 'laugh', t)); e['tear'] = 0.
    T = Rig(x, y, 122, .06, -.06 - .05 * b, ro + .04 * b, .12, r=(.78, 1.1, .74), sx=1 - .02 * b, sy=1 + .03 * b); st = Stack(); p = Part(T, None, 760)
    B = p.shape([(-.3, 1.1, -.4), (-1.1, 1.35, -.4), (-1.5, 2.0, -.4), (-1.5, 4.2, -.4), (1.5, 4.2, -.4), (1.5, 2.0, -.4), (1.1, 1.35, -.4), (.3, 1.1, -.4)], C['coat'], 1.); p.shade([(.1, 1.1, -.4), (.5, 2.0, -.4), (.3, 4.3, -.4), (2, 4.3, -.4), (1.8, 1.2, -.4)], C['coatS'], B); boa(p, T, 770, f, -.1)
    for a, b_ in ((1.35, .30), (1.55, .36), (1.75, .40)): p.shape([(-b_, a, 0), (0, a + .10, .05), (b_, a, 0), (b_ * .9, a + .16, 0), (0, a + .30, .05), (-b_ * .9, a + .16, 0)], C['crav'], .6)
    st.add(p, li); return over4(st.a, draw(T, li, e, f, hat='crown', squash=-.06 * b, wind=.1 * math.sin(f // 2 * .35)))


def violin(p, o, ang, sc, flip=1):
    """violin body centred at o, neck along ang (screen), sc = body length px.  returns dict of useful points"""
    c, s_ = math.cos(ang), math.sin(ang); R = lambda u, v: (o[0] + (u * c - v * s_ * flip) * sc, o[1] + (u * s_ + v * c * flip) * sc)
    P = [(-.50, 0), (-.48, .16), (-.34, .235), (-.16, .20), (-.08, .13), (.04, .125), (.12, .175), (.26, .19), (.40, .13), (.44, 0)]; B = [R(u, v) for u, v in P] + [R(u, -v) for u, v in P[-2:0:-1]]; Q = p.shape(spn(B, True, (), 8), C['wood'], .85); p.shade(Q + [sc * .05, sc * .06], C['woodS'], Q); p.f.clip(Q); p.f.line(np.array([R(-.40, -.16), R(-.1, -.13), R(.30, -.12)]), sc * .035, '#f0a860'); p.f.unclip()
    Q = p.shape(np.array([R(.30, -.035), R(1.05, -.028), R(1.05, .028), R(.30, .035)]), C['woodD'], .6); sc2 = [R(1.02, -.035), R(1.14, -.05), R(1.18, .0), R(1.12, .05), R(1.02, .035)]; p.shape(spn(sc2, True, (), 6), C['woodS'], .6)
    for v in (-.02, -.007, .007, .02): p.l.line(np.array([R(-.34, v * 1.4), R(1.02, v)]), max(.8, .16 * p.k), '#e9dcc0')
    p.shape(np.array([R(-.12, -.06), R(-.09, -.06), R(-.09, .06), R(-.12, .06)]), '#e8c99a', .4); p.shape(np.array([R(-.48, -.03), R(-.30, -.045), R(-.30, .045), R(-.48, .03)]), C['woodD'], .4)
    for sg in (-1, 1): p.l.line(spn([R(-.04, sg * .10), R(-.10, sg * .085), R(-.16, sg * .105)], False), .5 * p.k, INK)
    return dict(bridge=np.array(R(-.02, 0)), neck=np.array(R(.92, 0)), chin=np.array(R(-.44, .05)), R=R)
