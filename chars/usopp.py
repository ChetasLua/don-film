"""Usopp: long nose, thick lips, curls, yellow cap, goggles, headphones."""
from .gang_kit import *

C = dict(skin='#e0ae74', sh='#ad7443', dk='#744a2c', lip='#d39c65', hair='#1c171d', hair2='#3d3442', cap='#e8bb4c', capS='#a8742c', capD='#774b22', gog='#efe6cf', gogS='#a6946e', lens='#d4472c', lensS='#7d1d1c', phone='#f1e6c6', phoneS='#b09a6c')
EX = dict(
    calm=E(eo=.92, pr=.30, mw=.24, ml=.02), aim=E(eo=.80, el=.10, et=.10, pr=.26, lx=.0, ly=.1, bi=.07, bo=-.03, mw=.22, mu=.02, md=.06, ml=.02, mt=.8, ms=-.03),
    shout=E(eo=1., pr=.18, bi=-.05, bo=-.10, mw=.40, mu=.16, md=.46, ml=.04, mt=.22, mb=.14, tg=.7), worry=E(eo=1., pr=.20, bi=-.16, bo=.06, mw=.27, mu=.05, md=.06, ml=-.07, mt=.98, wob=.7, ly=.15),
    grin=E(eo=.8, pr=.3, mw=.40, mu=.02, md=.17, ml=.12, mt=.98, bi=-.02), laugh=E(eo=0., arc=1, bi=-.12, bo=-.04, mw=.42, mu=.12, md=.50, ml=.10, mt=.22, tg=.8, tear=1.), cry=E(eo=0., arc=-1, bi=-.20, bo=.08, mw=.36, mu=.10, md=.30, ml=-.10, mt=.35, mb=.2, tg=.4, wob=1., tear=1., snot=1.))


def draw(T, li, e, f=0, seed=100, wink=0., wind=0., nose=1., gog_up=1., hair=1., flap=0.):
    """list of parts in paint order on a Stack"""
    s = T.s; st = Stack(); c0 = T.c; box = (c0[0] - s * 2.3, c0[1] - s * 2.3, c0[0] + s * 2.3, c0[1] + s * 2.3); n = lambda sd: Part(T, box, sd)
    L = -1 if T.yaw >= -.05 else 1
    # --- hair mass behind
    p = n(seed); sw = wind
    hp = [(-.70, -.45), (-1.02, -.34), (-1.22 + sw * .16, .02), (-1.30 + sw * .36, .42 - abs(sw) * .04), (-1.18 + sw * .58, .86 - abs(sw) * .12), (-.86 + sw * .72, 1.16 - abs(sw) * .20), (-.4 + sw * .6, 1.22 - abs(sw) * .20), (0 + sw * .3, 1.0), (.4 + sw * .6, 1.22 - abs(sw) * .20), (.86 + sw * .74, 1.16 - abs(sw) * .26), (1.18 + sw * .62, .86 - abs(sw) * .24), (1.30 + sw * .40, .42 - abs(sw) * .14), (1.22 + sw * .18, .02), (1.02, -.34), (.70, -.45)]
    Q = T(spn(hp, True, (), 8), -.25); S = scallop(Q, s * .40, s * .12, seed, jit=.3); p.shape(S, C['hair'], 1.)
    r = np.random.default_rng(seed + 1)
    for sd in (-1, 1):                                       # long wavy locks: lit ribbons and pen lines that follow the fall
        for i in range(7):
            x0 = sd * (.74 + .07 * i); x1 = sd * (.50 + .12 * i); ph = r.uniform(0, 6.28); am = .045; t_ = np.linspace(0, 1, 8); sp = [(x0 * (1 - v) + x1 * v + am * math.sin(ph + v * 7.5) + sw * .34 * v * v + sd * .10 * math.sin(v * 3.1), -.36 + (1.38 - .05 * i) * v, -.25) for v in t_]
            if i % 2 == 0: p.f.ctx.save(); p.f._path(S); p.f.ctx.clip(); p.f.fill(band_poly(T.P3(spn(sp, False)), [s * .012, s * .055, s * .05, s * .012]), C['hair2']); p.f.ctx.restore()
            p.line(sp, .46, '#6a5d70' if i % 2 else '#0a070b', (.2, .5))
    st.add(p, li)
    # --- neck + face
    p = n(seed + 20); NK = p.fill([(-.30, .70, -.1), (-.33, 1.12, -.1), (-.42, 1.42, -.2), (0, 1.52, -.2), (.42, 1.42, -.2), (.33, 1.12, -.1), (.30, .70, -.1)], C['skin'], co=(0, 6)); p.shade([(-.5, .6, 0), (-.40, 1.14, 0), (-.05, 1.30, .1), (.10, 1.6, 0), (.6, 1.6, 0), (.5, .6, 0)], C['sh'], NK); p.shade([(-.5, .6, 0), (-.36, 1.06, 0), (0, 1.14, .1), (.5, 1.2, 0), (.5, .6, 0)], C['dk'], NK); p.line([(-.31, .85, -.1), (-.34, 1.14, -.1), (-.46, 1.38, -.2)], .8, taper=(.1, .5)); p.line([(.31, .85, -.1), (.34, 1.14, -.1), (.46, 1.38, -.2)], .8, taper=(.1, .5)); p.line([(-.20, 1.16, 0), (-.06, 1.42, 0)], .4)
    jaw = e['md'] * .22; fc = sym([(0, 1.02 + jaw), (.19, .99 + jaw), (.41, .84 + jaw * .8), (.585, .57 + jaw * .3), (.675, .24), (.69, -.08), (.67, -.52), (.36, -.78)])[::-1] + []
    fc = [(-.36, -.78), (-.67, -.52), (-.69, -.08), (-.675, .24), (-.585, .57 + jaw * .3), (-.41, .84 + jaw * .8), (-.19, .99 + jaw), (0, 1.02 + jaw), (.19, .99 + jaw), (.41, .84 + jaw * .8), (.585, .57 + jaw * .3), (.675, .24), (.69, -.08), (.67, -.52), (.36, -.78)]
    F = p.shape(fc, C['skin'], 1., limb=True)
    k = -L; side = [(k * .80, -.6), (k * .40, -.36), (k * .50, -.10), (k * .455, .12), (k * .52, .30), (k * .40, .50), (k * .46, .66), (k * .26, .86), (k * .04, 1.00 + jaw), (k * .1, 1.2 + jaw), (k * .9, 1.0)]
    p.shade(side, C['sh'], F); p.shade([(-.8, -.9), (-.8, -.22), (-.45, -.14), (-.1, -.20), (.15, -.15), (.45, -.20), (.8, -.10), (.8, -.9)], C['sh'], F)
    p.shade([(k * .80, .2), (k * .62, .30), (k * .60, .60), (k * .42, .86), (k * .16, 1.02 + jaw), (k * .9, 1.2)], C['dk'], F, .75)
    p.ticks((k * .40, .34), (k * .54, .62), 6, .28, .13); p.ticks((-k * .50, .26), (-k * .34, .24), 4, .26, .10)
    p.hl([(-k * .40, .20), (-k * .22, .16), (-k * .30, .30), (-k * .44, .32)], '#e6b07c', .8, F)
    zb = float(T.z(0, .20)); B0, A0 = T.at(0, .17, zb), T.at(0, .18, zb + .9 * nose); p.shade(band_poly(np.array([B0 + [k * s * .06, s * .16], A0 * .55 + B0 * .45 + [k * s * .10, s * .20]]), [s * .30, s * .26]), C['sh'], F)
    # --- eyes, brows
    for sd in (-1, 1):
        q = dict(e)
        if wink and sd == (1 if wink > 0 else -1): q['eo'] = 0.; q['arc'] = -.6; q['et'] = .25
        ex, ey = sd * .30, .0 + (.015 if q['eo'] < .12 else 0)
        eye(p, ex, ey, .175, .205, q, sd, C['skin'], C['sh'], lash=3, lidw=1.05)
        if wink and sd == (1 if wink > 0 else -1): p.ticks((ex - .12, ey + .13), (ex + .14, ey + .13), 5, .26, (.02, .07)); p.line([(ex + sd * .2, ey - .06), (ex + sd * .31, ey - .10)], .4); p.line([(ex + sd * .2, ey + .02), (ex + sd * .32, ey + .04)], .4)
        bi, bo = e['bi'], e['bo']
        if wink and sd == (1 if wink > 0 else -1): bi, bo = bi + .05, bo + .05
        brow(p, [(sd * .13, -.285 + bi), (sd * .30, -.335 + (bi + bo) * .4), (sd * .50, -.275 + bo)], [.05, .105, .03], C['hair'])
    if e['bi'] < -.1: p.ticks((-.10, -.34), (.12, -.34), 4, .26, (.015, .09))
    # --- mouth: thick lips
    my = .64; lw_, lu, ld = e['mw'] + .052, e['mu'] + .062, e['md'] + .082; u = np.linspace(-1, 1, 17); lf = -e['ml'] * np.abs(u) ** 1.5 + e['ms'] * u + e['wob'] * np.sin(u * 8.5 + 1) * .05
    LP = T(np.r_[np.c_[lw_ * u, my + lf - lu * (1 - np.abs(u) ** 2.2) - .015 * np.cos(u * 9)], np.c_[lw_ * u, my + lf + ld * (1 - np.abs(u) ** 2.4)][::-1]])
    p.fill(LP, C['lip']); p.ol(LP, .62); p.shade(LP + [0, s * .05], C['sh'], LP); Qm = mouth(p, 0, my, e, w=.85)
    p.shade([(-lw_ * .5, my + ld + .05), (0, my + ld + .03), (lw_ * .5, my + ld + .05), (0, my + ld + .10)], C['sh'], F); p.f.line(T(spn([(-lw_ * .4, my + ld * .72 + e['md'] * .15), (-.02, my + ld * .80 + e['md'] * .12), (lw_ * .2, my + ld * .74 + e['md'] * .13)], False)), s * .022, '#e4ab7e')
    if e.get('tongue_out'):
        tx = .16; p.shape([(tx - .06, my - .0), (tx - .02, my + .13), (tx + .06, my + .19), (tx + .14, my + .13), (tx + .12, my - .02)], '#e07a78', .55, dz=.05); p.line([(tx + .03, my + .02), (tx + .06, my + .12)], .3, '#8a2f3a', dz=.06)
    st.add(p, li)
    # --- phones, front curls, cap, goggles
    p = n(seed + 60)
    for sd in (-1, 1):
        if sd * T.yaw > .25: continue
        b = [(sd * .62, -.30, .2), (sd * .84, -.26, .05), (sd * .90, -.02, .0), (sd * .80, .10, .1), (sd * .66, -.02, .25)]; p.shape(scallop(T(spn(b, True, (), 6)), s * .12, s * .05, seed + sd), C['hair'], .8); curls(p, off(b, 0, 0, .98), 3, .07, '#5a4f63', seed + 5 + sd, .15, .4)
        e1 = [(sd * .76 + .13 * math.cos(a), .10 + .25 * math.sin(a), .04) for a in np.linspace(0, 6.28, 14)[:-1]]; Q = p.shape(e1, C['phone'], .9); p.shade([(sd * .76 + (.13 if sd * k > 0 else .13) * math.cos(a) + k * .07, .16 + .25 * math.sin(a), .04) for a in np.linspace(0, 6.28, 14)[:-1]], C['phoneS'], Q)
        e2 = [(sd * .785 + .07 * math.cos(a), .10 + .14 * math.sin(a), .08) for a in np.linspace(0, 6.28, 12)[:-1]]; p.shape(e2, C['gogS'], .45); p.line([(sd * .785 - .03, .02, .08), (sd * .785 + .03, .16, .08)], .35); p.line([(sd * .785 + .03, .04, .08), (sd * .785 - .03, .15, .08)], .35)
    cr = [(-.79, -.26, .05), (-.86, -.58, 0), (-.72, -1.00, 0), (-.34, -1.26, 0), (.06, -1.32, 0), (.44, -1.22, 0), (.76, -.96, 0), (.87, -.56, 0), (.79, -.26, .05), (.50, -.38, .52), (0, -.45, .80), (-.50, -.38, .52)]
    Q = p.shape(cr, C['cap'], 1., co=(0, 8)); p.shade([(k * 1.0, -.2, 0), (k * .50, -.40, .4), (k * .46, -.72, .4), (k * .62, -1.0, .2), (k * .3, -1.34, 0), (k * 1.1, -1.2, 0)], C['capS'], Q); p.shade([(-.9, -.2, .4), (-.45, -.52, .5), (0, -.60, .8), (.45, -.52, .5), (.9, -.2, .4), (0, -.2, .8)], C['capS'], Q)
    p.line([(-.02, -1.30, .05), (-.10, -1.0, .5), (-.08, -.66, .74)], .4); p.line([(-.60, -1.08, .1), (-.52, -.84, .45), (-.50, -.56, .55)], .34); p.line([(.58, -1.10, .1), (.52, -.84, .45), (.50, -.56, .55)], .34)
    p.ticks((k * .52, -.95), (k * .64, -.62), 6, .26, .12)
    # strap + goggles
    sy = -.30 if not gog_up else -.60; bd = [(-.88, sy + .04, .0), (-.60, sy - .02, .48), (0, sy - .06, .84), (.60, sy - .02, .48), (.88, sy + .04, .0), (.87, sy + .17, .0), (.60, sy + .11, .50), (0, sy + .08, .86), (-.60, sy + .11, .50), (-.87, sy + .17, .0)]
    Q = p.shape(bd, C['capD'], .6, co=(0, 4, 5, 9))
    vz = [(-.62, -.40, .55), (-.60, -.28, .86), (-.3, -.19, 1.06), (.0, -.17, 1.12), (.3, -.19, 1.06), (.60, -.28, .86), (.62, -.40, .55), (0, -.50, .84)]; Q = p.shape(vz, C['cap'], 1., co=(0, 6)); p.shade([(-.7, -.25, .9), (0, -.12, 1.1), (.7, -.25, .9), (.6, -.34, .9), (0, -.27, 1.05), (-.6, -.34, .9)], C['capS'], Q); p.line([(-.52, -.34, .80), (0, -.25, 1.08), (.52, -.34, .80)], .3)
    for sd in (-1, 1):
        gx, gy, gz = sd * .285, sy + .01, .86; R_o, R_i = .225, .165
        Q = p.shape([(gx + R_o * math.cos(a), gy + R_o * math.sin(a), gz) for a in np.linspace(0, 6.28, 18)[:-1]], C['gog'], .95); p.shade([(gx + R_o * math.cos(a) + k * .06, gy + R_o * math.sin(a) + .06, gz) for a in np.linspace(0, 6.28, 18)[:-1]], C['gogS'], Q)
        Q = p.shape([(gx + R_i * math.cos(a), gy + R_i * math.sin(a), gz + .06) for a in np.linspace(0, 6.28, 18)[:-1]], C['lens'], .6); p.shade([(gx + R_i * math.cos(a) + k * .05, gy + R_i * math.sin(a) + .07, gz + .06) for a in np.linspace(0, 6.28, 18)[:-1]], C['lensS'], Q)
        p.hl([(gx - .10, gy - .02, gz + .08), (gx - .02, gy - .12, gz + .08), (gx + .03, gy - .10, gz + .08), (gx - .07, gy + .03, gz + .08)], '#ffd9b0', 1., Q)
    p.shape([(-.07, sy - .04, .9), (.07, sy - .04, .9), (.07, sy + .06, .9), (-.07, sy + .06, .9)], C['gogS'], .5, co=(0, 1, 2, 3))
    st.add(p, li); p = n(seed + 80)
    # --- nose: a tube at the lens
    zb = float(T.z(0, .20)); Ln = 1.02 * nose; B, M_, A = T.at(0, .235, zb - .05), T.at(0, .24, zb + Ln * .55), T.at(0, .245, zb + Ln); rb, ra = .112 * s, .128 * s * (1 + .10 * Ln)
    d = A - B; ln = np.hypot(*d); an = math.atan2(d[1], d[0]) if ln > 2 else 0.; a1 = np.linspace(an - math.pi / 2, an + math.pi / 2, 16); a2 = np.linspace(an + math.pi / 2, an + 1.5 * math.pi, 12)
    N_ = np.r_[np.c_[A[0] + ra * np.cos(a1), A[1] + ra * np.sin(a1)], np.c_[B[0] + rb * np.cos(a2), B[1] + rb * np.sin(a2)]]
    p.fill(N_, C['skin']); p.shade(N_ + [s * .075 * k, s * .105], C['sh'], N_); p.shade(N_ + [s * .15 * k, s * .19], C['dk'], N_, .7)
    if ln > rb * .6:
        p.ol(np.r_[N_[-4:-1], N_[:19]], 1.05, closed=False); p.hl(band_poly(np.array([B + (A - B) * .35 + [0, -rb * .45], A + [-ra * .2 * k, -ra * .45]]), [s * .02, s * .05]), '#f0c290', 1., N_)
    else: p.ol(N_[:16] if ln > 4 else N_, 1., closed=ln <= 4); p.hl(ell(A[0] - k * ra * .35, A[1] - ra * .42, ra * .30, ra * .20, 14), '#f3c897', 1., N_)
    if ln > rb * .6: a3 = np.linspace(an + 2.0, an + 4.2, 8); p.l.line(np.c_[A[0] + ra * .86 * np.cos(a3) - math.cos(an) * ra * .1, A[1] + ra * .86 * np.sin(a3) - math.sin(an) * ra * .1], .3 * p.k, INK, alpha=.8)
    # --- tears
    if e['tear'] > 0:
        ph = (f // 2) * .9
        for sd in (-1, 1):
            t = e['tear']; rope(p, [(sd * .36, .10), (sd * .47, .30 + .02 * math.sin(ph)), (sd * .50, .30 + .40 * t), (sd * (.47 + .02 * math.sin(ph + 1)), .30 + .62 * t)], [s * .035, s * .075, s * .085, s * .15])
    if e['snot'] > 0: rope(p, [(.07, .30), (.09, .45), (.07, .60 + .03 * math.sin(f // 2))], [s * .05, s * .05, s * .07])
    st.add(p, li); return st.a


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light_='day', f=0, **kw):
    li = kw.pop('light', light_); li = light(li) if isinstance(li, str) else li; sq = kw.pop('sq', (1., 1.))
    return draw(Rig(cx, cy, s, yaw, pitch, roll, kw.pop('persp', .12), sx=sq[0], sy=sq[1]), li, pick(EX, expr), f, **kw)


def body_g1(T, li, seed=150):
    """bare shoulders and straps running off frame (local units of the head)"""
    p = Part(T, (T.c[0] - T.s * 4.6, T.c[1] - T.s * 3.6, T.c[0] + T.s * 4.2, T.c[1] + T.s * 3.6), seed)
    B = p.shape([(-.42, 1.15), (-1.05, 1.36), (-1.62, 1.62), (-1.95, 2.3), (-2.1, 4.2), (2.1, 4.2), (1.95, 2.3), (1.62, 1.62), (1.05, 1.36), (.42, 1.15)], C['skin'], 1.1, z=-.5); p.shade([(.2, 1.1), (.5, 1.7), (.3, 2.6), (.6, 4.4), (2.6, 4.4), (2.2, 1.3)], C['sh'], B, z=-.5)
    for sd in (-1, 1): Q = p.shape([(sd * .86, 1.24), (sd * 1.14, 1.34), (sd * 1.0, 2.4), (sd * .8, 4.4), (sd * .5, 4.4), (sd * .74, 2.4)], '#a3292b', .8, z=-.5, co=(0, 1, 3, 4)); p.shade([(sd * .95, 1.2), (sd * 1.2, 1.3), (sd * 1.0, 4.4), (sd * .75, 4.4)], '#64151f', Q, z=-.5)
    p.line([(-.35, 1.36), (-.1, 1.50), (-.02, 1.44)], .6, z=-.5); p.line([(.35, 1.36), (.1, 1.50), (.04, 1.44)], .6, z=-.5); p.line([(0, 1.9), (.02, 3.0)], .4, z=-.5); p.ticks((.5, 1.7), (.7, 2.3), 6, .3, .18)
    return p.done(li)


def g1(f):
    x, y = 215, 610; t, b = g1_t(f, 1); ro = crown_roll(x, y); li = light('back', math.atan2(500 - y, 960 - x)); wob = (.03 * math.sin(f // 2 * .5) * (1 - t)); e = ('worry', 'laugh', t)
    T = Rig(x, y, 148, .10 + wob, -.10 - .06 * b, ro + .03 * b, .14, sx=1 - .02 * b, sy=1 + .035 * b); st = Stack(); st.add(body_g1(T, li)); lay = draw(T, li, pick(EX, e), f, wind=.2 * math.sin(f // 2 * .4)); return over4(st.a, lay)
