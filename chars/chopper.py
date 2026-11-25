"""Chopper: small reindeer, blue nose, pink-red hat with the white X (hat='old'), blue-white cap over it (hat='new'), antlers, the mended left one."""
from .gang_kit import *

C = dict(fur='#c18a55', sh='#87553a', dk='#573427', muz='#e6bd8a', muzS='#b48660', ear='#e79a96', ant='#a9744a', antS='#6c4431', hat='#dd4f6b', hatS='#942a4d', hatD='#5f1838', x='#fbf6ea', xS='#c3b4cb', nose='#3f7fd0', noseS='#1f4788',
         cap='#58b7dd', capS='#2c6ea6', dot='#fbf8ee', brim='#ef9aa8', brimS='#ad5673', metal='#c9ced8', metalS='#7d8494', shorts='#6d3b7c', shortsS='#42224f', pack='#7fb3d8', packS='#4a7399', hoof='#4a3030')
EX = dict(
    calm=E(eo=1., pr=.46, mw=.20, ml=.06), look=E(eo=1., pr=.50, ly=-.55, bi=-.06, bo=-.02, mw=.10, mu=.02, md=.05, ml=-.02), worry=E(eo=1., pr=.44, bi=-.16, bo=.05, mw=.26, mu=.03, md=.06, ml=-.08, wob=1., tear=.25),
    brim=E(eo=1., pr=.54, ly=-.30, bi=-.17, bo=.03, mw=.22, mu=.02, md=.04, ml=-.07, wob=1., tear=.8), cry=E(eo=0., arc=-1, bi=-.18, bo=.06, mw=.40, mu=.08, md=.36, ml=-.06, mt=.16, tg=.6, wob=.8, tear=1., snot=1.),
    grin=E(eo=0., arc=1, bi=-.12, bo=-.02, mw=.50, mu=.03, md=.44, ml=.16, mt=.26, tg=.75, tear=.6, blush=1.), laugh=E(eo=0., arc=1, bi=-.12, bo=-.03, mw=.48, mu=.04, md=.52, ml=.14, mt=.22, tg=.8, tear=1., blush=1., snot=.0))


def hat_old(p, T, k, bounce=0.):
    """tall pink-red hat, white X"""
    b = bounce; cr = [(-.80, -.40, .0), (-.84, -.80 - b * .3, .0), (-.95, -1.20 - b, .0), (-.86, -1.48 - b, .0), (-.45, -1.62 - b, .0), (0, -1.66 - b, .0), (.45, -1.62 - b, .0), (.86, -1.48 - b, .0), (.95, -1.20 - b, .0), (.84, -.80 - b * .3, .0), (.80, -.40, .0), (0, -.30, .6)]
    Q = p.shape(cr, C['hat'], 1., co=(0, 10)); p.shade([(k * .9, -.3, 0), (k * .36, -.34, .5), (k * .44, -.85, .5), (k * .52, -1.25 - b, .4), (k * .30, -1.7 - b, .2), (k * 1.1, -1.7 - b, 0), (k * 1.1, -.5, 0)], C['hatS'], Q); p.shade([(k * .9, -.3, 0), (k * .70, -.5, .3), (k * .72, -1.1 - b, .3), (k * .84, -1.5 - b, 0), (k * 1.2, -1.2 - b, 0)], C['hatD'], Q, .8)
    p.line([(-.90, -1.30 - b, .1), (-.45, -1.40 - b, .5), (0, -1.42 - b, .62), (.45, -1.40 - b, .5), (.90, -1.30 - b, .1)], .45); p.ticks((k * .40, -.95 - b * .5), (k * .50, -1.30 - b), 5, .26, .12)
    for a in (-.72, .72):                                  # the X
        c, s_ = math.cos(a), math.sin(a); w, l = .085, .37; cy = -1.06 - b * .55; X = [(c * u - s_ * v, cy + s_ * u + c * v, .78) for u, v in ((-l, -w * 1.2), (0, -w * .9), (l, -w * 1.2), (l * 1.02, w * 1.1), (0, w * .9), (-l * 1.02, w * 1.1))]; Qx = p.shape(X, C['x'], .5, co=(0, 2, 3, 5)); p.shade([(k * .05, -.6, .8), (k * .10, -1.3 - b, .8), (k * .6, -1.3 - b, .8), (k * .6, -.6, .8)], C['xS'], Qx)
    return Q


def brim_old(p, T, k, dip=0.):
    """wide soft brim: far half, underside, front roll"""
    t = np.linspace(0, 2 * math.pi, 30)[:-1]; rx, rz = 1.24, .92; y = lambda a: -.52 + .045 * math.cos(a * 2) + dip * max(0, math.sin(a)) + .022 * math.sin(a * 5 + 1) + .20 * math.sin(a)
    O = [(rx * math.cos(a), y(a), rz * math.sin(a) * .8) for a in t]; Q = p.shape(O, C['hat'], 1.05)
    p.shade([(rx * math.cos(a), y(a) + .085, rz * math.sin(a) * .8) for a in t], C['hatS'], Q); p.shade([(k * .28, -.6, .4), (k * .44, -.1, .9), (k * 1.5, -.1, .4), (k * 1.5, -.7, 0)], C['hatS'], Q); p.shade([(rx * math.cos(a) * 1.0, y(a) + .175, rz * math.sin(a) * .8) for a in t], C['hatD'], Q, .9)
    p.line([(-.84, -.62, .2), (-.5, -.48, .62), (0, -.43, .72), (.5, -.48, .62), (.84, -.62, .2)], .5); p.line([(-1.10, -.46, .40), (-.5, -.335, .74), (0, -.305, .80), (.5, -.335, .74), (1.10, -.46, .40)], .3); return Q


def hat_new(p, T, k, b=0.):
    """blue cap with white dots over the old hat, pink brim + X patch"""
    cr = [(-.92, -.30, 0), (-1.16, -.66, 0), (-1.18, -1.06 - b, 0), (-.92, -1.42 - b, 0), (-.45, -1.60 - b, 0), (0, -1.64 - b, 0), (.45, -1.60 - b, 0), (.92, -1.42 - b, 0), (1.18, -1.06 - b, 0), (1.16, -.66, 0), (.92, -.30, 0), (0, -.36, .7)]
    Q = p.shape(cr, C['cap'], 1., co=(0, 10))
    p.shade([(k * 1.0, -.2, 0), (k * .46, -.4, .5), (k * .62, -.9, .5), (k * .50, -1.4 - b, .3), (k * .2, -1.7 - b, 0), (k * 1.3, -1.6 - b, 0), (k * 1.4, -.5, 0)], C['capS'], Q)
    for x, y, r in ((-.86, -.66, .12), (-.92, -1.08, .11), (-.62, -1.38, .10), (.86, -.66, .12), (.92, -1.08, .11), (.62, -1.38, .10), (-.22, -1.50, .10), (.26, -1.52, .10)):
        D = p.fill([(x + r * math.cos(a), y - b * .6 + r * .95 * math.sin(a), .75 * math.sqrt(max(0, 1 - (x / 1.2) ** 2))) for a in np.linspace(0, 6.28, 13)[:-1]], C['dot']); p.ol(D, .3); p.shade(D + [k * T.s * .06, T.s * .05], C['xS'], D)
    P = [(-.66, -.46, .72), (-.70, -.95 - b * .5, .62), (-.44, -1.24 - b * .8, .55), (0, -1.32 - b * .9, .58), (.44, -1.24 - b * .8, .55), (.70, -.95 - b * .5, .62), (.66, -.46, .72), (0, -.40, .9)]; Qp = p.shape(P, C['hat'], .7, co=(0, 6)); p.shade([(k * .25, -.3, .8), (k * .30, -1.3, .6), (k * .9, -1.3, .5), (k * .9, -.3, .6)], C['hatS'], Qp)
    for a in (-.72, .72):
        c, s_ = math.cos(a), math.sin(a); w, l = .075, .32; cy = -.86 - b * .5; X = [(c * u - s_ * v, cy + s_ * u + c * v, .88) for u, v in ((-l, -w * 1.2), (0, -w * .9), (l, -w * 1.2), (l * 1.02, w * 1.1), (0, w * .9), (-l * 1.02, w * 1.1))]; Qx = p.shape(X, C['x'], .5, co=(0, 2, 3, 5)); p.shade([(k * .05, -.5, .8), (k * .10, -1.3, .8), (k * .6, -1.3, .8), (k * .6, -.5, .8)], C['xS'], Qx)
    B = [(-1.10, -.36, .25), (-.60, -.56, .70), (0, -.60, .86), (.60, -.56, .70), (1.10, -.36, .25), (1.16, -.14, .30), (.64, -.22, .86), (0, -.22, 1.02), (-.64, -.22, .86), (-1.16, -.14, .30)]; Qb = p.shape(B, C['brim'], 1., co=(0, 4, 5, 9)); p.shade([(-1.3, -.1, .4), (-.6, -.34, .9), (0, -.36, 1.), (.6, -.34, .9), (1.3, -.1, .4), (0, 0, 1.)], C['brimS'], Qb); p.line([(-1.0, -.24, .4), (-.5, -.34, .88), (0, -.35, 1.0), (.5, -.34, .88), (1.0, -.24, .4)], .34)


def antler(p, T, sd, k, mend, sway=0., new=False):
    a = [(sd * .80, -.62, -.1), (sd * 1.12, -.86, -.1), (sd * 1.36, -1.20, -.08), (sd * (1.44 + sway), -1.60, -.05)]; W0 = [T.s * v for v in (.22, .24, .20, .05)]
    br = [(sd * 1.22, -.98, -.1), (sd * 1.60, -1.06, -.05), (sd * 1.90, -1.34, 0)]; b2 = [(sd * 1.30, -1.12, -.08), (sd * 1.14, -1.42, 0), (sd * 1.10, -1.66, .05)]
    for q, w in ((br, (.17, .17, .04)), (b2, (.15, .13, .04)), (a, None)):
        Q = band_poly(T.P3(spn(q, False, (), 8)), W0 if w is None else [T.s * v for v in w]); p.shape(Q, C['ant'], .85); p.shade(Q + [k * T.s * .05, T.s * .06], C['antS'], Q)
    # knit the joints
    for q in ((sd * 1.22, -.98), (sd * 1.31, -1.13)): p.f.fill(ell(*T.at(q[0], q[1], -.08), T.s * .085, T.s * .085, 16), C['ant']); c = p.l.ctx; c.set_operator(cairo.OPERATOR_DEST_OUT); p.l._path(ell(*T.at(q[0], q[1], -.08), T.s * .075, T.s * .075, 16)); c.set_source_rgba(0, 0, 0, 1); c.fill(); c.set_operator(cairo.OPERATOR_OVER)
    p.line([(sd * 1.0, -.74, -.05), (sd * 1.2, -.98, 0), (sd * 1.33, -1.3, 0)], .28, a=.7)
    if mend:
        M = [(sd * .84, -.78, 0), (sd * .98, -.92, 0), (sd * 1.16, -.76, 0), (sd * 1.02, -.60, 0)]; Q = p.shape(M, C['metal'], .65, co=(0, 1, 2, 3)); p.shade(Q + [T.s * .06 * k, T.s * .06], C['metalS'], Q)
        for u, v in ((.95, -.80), (1.05, -.71)): p.fill(ell(*T.at(sd * u, v, .02), T.s * .024, T.s * .024, 10), '#3a3d48')


def ear(p, T, sd, k, flop=0.):
    E_ = [(sd * .78, -.26, .1), (sd * 1.12, -.44 + flop * .4, .05), (sd * 1.56, -.34 + flop, .0), (sd * 1.74, -.12 + flop * 1.3, 0), (sd * 1.46, -.02 + flop, .05), (sd * 1.06, -.02 + flop * .4, .1), (sd * .80, -.02, .15)]; Q = p.shape(E_, C['fur'], .95, co=(0, 3, 6)); p.shade(off(E_, 0, .11), C['sh'], Q)
    I = [(sd * .96, -.20, .15), (sd * 1.22, -.31 + flop * .6, .08), (sd * 1.52, -.20 + flop, .04), (sd * 1.36, -.10 + flop, .06), (sd * 1.06, -.10 + flop * .4, .12)]; p.shape(I, C['ear'], .45, co=(0, 2)); p.ticks((sd * 1.0, -.18), (sd * 1.4, -.18 + flop * .8), 4, .24, (.10 * sd, -.03))


def draw(T, li, e, f=0, seed=300, hat='old', bounce=0., flop=0., sway=0., tears=True, after=None):
    s = T.s; st = Stack(); c0 = T.c; box = (c0[0] - s * 2.6, c0[1] - s * 2.6, c0[0] + s * 2.6, c0[1] + s * 2.6); n = lambda sd: Part(T, box, sd); L = -1 if T.yaw >= -.05 else 1; k = -L
    p = n(seed); antler(p, T, -1, k, False, sway); antler(p, T, 1, k, True, sway); st.add(p, li)
    p = n(seed + 30); ear(p, T, -1, k, flop); ear(p, T, 1, k, flop)
    jw = e['md'] * .16; fc = [(-.40, -.72), (-.80, -.48), (-.92, -.08), (-.98, .28), (-.86, .66), (-.56, .94 + jw), (0, 1.04 + jw), (.56, .94 + jw), (.86, .66), (.98, .28), (.92, -.08), (.80, -.48), (.40, -.72)]
    Q0 = T.limb(spn(fc, True, (), 10), .25); lo = np.clip((Q0 - Q0.mean(0)) @ np.array([math.sin(T.roll) * -1, math.cos(T.roll)]) / s, -1, 1); F = p.shape(fur(Q0, s * .24, s * .026, seed + 3, .6), C['fur'], 1.)
    p.shade([(k * 1.1, -.8), (k * .60, -.30), (k * .74, .05), (k * .66, .36), (k * .72, .60), (k * .42, .86), (k * .10, 1.05 + jw), (k * .3, 1.4), (k * 1.3, 1.1)], C['sh'], F); p.shade([(-1.1, -.9), (-1.0, -.16), (-.5, -.12), (0, -.16), (.4, -.10), (.7, -.14), (1.1, -.02), (1.1, -.9)], C['sh'], F, .92)
    M = [(-.60, .30), (-.40, .14), (0, .10), (.40, .14), (.60, .30), (.66, .58), (.48, .88 + jw), (0, .99 + jw), (-.48, .88 + jw), (-.66, .58)]; Qm = p.fill(M, C['muz']); p.shade([(k * .10, 1.1 + jw), (k * .36, .80), (k * .46, .5), (k * .40, .2), (k * .9, .2), (k * .9, 1.1)], C['muzS'], Qm); p.shade([(-.34, .40), (0, .48), (.34, .40), (.22, .30), (-.22, .30)], C['muzS'], Qm)
    p.ticks((k * .66, .34), (k * .80, .60), 6, .26, .14); p.ticks((-k * .78, .30), (-k * .64, .42), 4, .24, .10)
    if e['blush']:
        for sd in (-1, 1): p.shade([(sd * .52, .30), (sd * .66, .25), (sd * .82, .30), (sd * .80, .42), (sd * .62, .45)], '#e07f6a', F, .55 * e['blush']); p.ticks((sd * .56, .36), (sd * .78, .35), 4, .24, (-.03, .09), '#9b3f38')
    for sd in (-1, 1):
        eye(p, sd * .385, .045, .215, .255, e, sd, C['fur'], C['sh'], big=True, lidw=1.05, shw='#c2bdd2')
        brow(p, [(sd * .20, -.24 + e['bi']), (sd * .38, -.295 + (e['bi'] + e['bo']) * .4), (sd * .54, -.24 + e['bo'])], [.04, .07, .02], C['dk'], lw=.3)
    Qm2 = mouth(p, 0, .62 - e['md'] * .10, e, w=.8)
    p.line([(0, .40), (0, .50), (.0, .62 - e['mu'] - e['ml'] * 0 - e['md'] * .10)], .4)
    N = [(-.15, .24, .86), (0, .185, .92), (.15, .24, .86), (.13, .36, .92), (0, .42, .95), (-.13, .36, .92)]; Q = p.shape(N, C['nose'], .85); p.shade(Q + [k * s * .06, s * .07], C['noseS'], Q); p.hl(ell(*T.at(-.05 * k, .265, .96), s * .035, s * .022, 12), '#d7ecff', 1., Q)
    if e['tear'] > 0 and tears:
        ph = (f // 2) * .8; t = e['tear']
        for sd in (-1, 1):
            y0 = .20 if e['eo'] > .12 else .13; rope(p, [(sd * .42, y0), (sd * .56, y0 + .14), (sd * .63, y0 + .18 + .28 * t), (sd * (.62 + .02 * math.sin(ph)), y0 + .18 + .52 * t)], [s * .045, s * .10, s * .11, s * .19], dz=.02)
            if t > .5: p.shade([(sd * .30, y0 + .06), (sd * .42, y0 - .01), (sd * .52, y0 + .04), (sd * .42, y0 + .12)], '#dff3fb', F, .9)
    if e['snot'] > 0: rope(p, [(.06, .40), (.09, .54), (.06, .68)], [s * .05, s * .06, s * .08], dz=.05)
    st.add(p, li); p = n(seed + 60)
    if hat == 'old': hat_old(p, T, k, bounce); brim_old(p, T, k, bounce * .3 + .62 * max(0., math.sin(T.pitch)))
    else: hat_new(p, T, k, bounce)
    st.add(p, li); return st.a


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light_='day', f=0, **kw):
    li = kw.pop('light', light_); li = light(li) if isinstance(li, str) else li; sq = kw.pop('sq', (1., 1.)); kw.pop('wind', None)
    return draw(Rig(cx, cy, s, yaw, pitch, roll, kw.pop('persp', .10), r=(1.0, 1.08, .86), o=(0, .1), sx=sq[0], sy=sq[1]), li, pick(EX, expr), f, **kw)


def hoof_arm(p, J, Wd, k, sc, hoof_ang=None):
    """furry arm from screen joints, ends in a split hoof"""
    from .gang_shot import capsule, union
    Q = fur(union(capsule(J[0], J[1], Wd[0] * .5, Wd[1] * .5), capsule(J[1], J[2], Wd[1] * .5, Wd[2] * .5)), sc * .22, sc * .03, p.nx(), .5); p.shape(Q, C['fur'], .9); p.shade(Q + [k * sc * .07, sc * .09], C['sh'], Q)
    e = np.asarray(J[-1], float); d = e - np.asarray(J[-2], float); a = math.atan2(d[1], d[0]) if hoof_ang is None else hoof_ang; c, s_ = math.cos(a), math.sin(a); R = lambda u, v: (e[0] + (u * c - v * s_) * sc, e[1] + (u * s_ + v * c) * sc)
    H_ = spn([R(-.10, -.19), R(.10, -.20), R(.26, -.13), R(.19, 0), R(.26, .13), R(.10, .20), R(-.10, .19)], True, (3,), 8); H_ = np.asarray(H_); p.shape(H_, C['hoof'], .9); p.shade(H_ + [k * sc * .05, sc * .06], '#2c1b1e', H_); p.f.fill(np.array([R(.02, -.12), R(.12, -.13), R(.14, -.08), R(.04, -.08)]), '#8a6258'); p.l.line(np.array([R(.20, 0), R(.02, 0.01)]), .5 * p.k, INK)


def g1(f):
    x, y = 585, 150; t, b = g1_t(f, 2); ro = crown_roll(x, y); li = light('back', math.atan2(500 - y, 960 - x)); e = pick(EX, ('worry', 'laugh', t)); e['tear'] = max(.45, t)
    T = Rig(x, y, 132, -.08, -.06 - .05 * b, ro + .02 * b, .12, r=(1.0, 1.08, .86), o=(0, .1), sx=1 - .03 * b, sy=1 + .05 * b); st = Stack(); p = Part(T, None, 340)
    d = np.array([math.cos(ro + math.pi / 2), math.sin(ro + math.pi / 2)]); n_ = np.array([-d[1], d[0]]); c0 = np.array([x, y], float)
    B = p.shape([(-.5, .8, -.4), (-1.0, 1.4, -.4), (-1.1, 2.8, -.4), (1.1, 2.8, -.4), (1.0, 1.4, -.4), (.5, .8, -.4)], C['fur'], 1.); p.shade([(-.2, .8, -.3), (.2, 1.3, -.3), (.1, 2.9, -.3), (1.4, 2.9, -.3), (1.2, .8, -.3)], C['sh'], B); st.add(p, li)
    lay = over4(st.a, draw(T, li, e, f, hat='new', bounce=.05 * b, flop=.10 * (1 - t) - .10 * b, sway=.02 * b)); p = Part(T, None, 360)
    for sd in (-1, 1):                                      # hooves gripping the edge of the picture of himself: up to his cheeks, shaking
        wb = .03 * math.sin(f // 2 * 1.3 + sd) * (1 + t); hoof_arm(p, [T.at(sd * 1.15, 1.5, -.2), T.at(sd * 1.5, 1.0, .1), T.at(sd * (1.06 - .10 * t), .62 + wb - .06 * t, .5)], [132 * .34, 132 * .33, 132 * .30], 1, 132 * .9)
    q = Stack(); q.add(p, li); return over4(lay, q.a)
