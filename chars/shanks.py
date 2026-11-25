"""Shanks, boy and grown: red hair, three parallel scars over the left eye, stubble; plus the body for the pier (b2)."""
from ._old import *

GROWN = dict(a=[(-1.0, .18), (-.965, .27), (-.89, .39), (-.76, .495), (-.58, .575), (-.38, .63), (-.12, .668), (.12, .69), (.35, .70), (.6, .68), (.8, .55), (.93, .34), (1.0, .05)], p=2.2, cheek=.04, socket=.045, ear=(.69, .10, -.36, -.12))
BOY = dict(a=[(-1.0, .22), (-.96, .38), (-.88, .52), (-.75, .65), (-.6, .735), (-.4, .79), (-.2, .81), (.05, .81), (.35, .79), (.6, .73), (.8, .58), (.93, .36), (1.0, .06)], p=2.1, cheek=.05, socket=.01, nose=(.08, -.20, -.38, -.43, .11), ear=(.79, .02, -.38, -.10))


def draw(cx, cy, s, yaw=0., pitch=0., roll=0., age='grown', expr='laugh', light='lamp', f=0, anchor=None, hat=True, side=1, rimdir=(1., .2), lw=1., jaw=1., seed=300, tears=0., tf=0., pal=None, stitches=False, scars=None, blow=1., look=(0., 0.), hairlen=1.):
    """-> dict(back, face, hair), fx.  side = which side of the face is in shadow (+1 = viewer's right); tf 0..1 runs the tears"""
    pal = dict(pal or LIGHT[light]); kid = age == 'boy'; scars = (not kid) if scars is None else scars; t = f // 2 * 2; laughing = expr == 'laugh'; drop = (.15 * jaw if laughing else .0)
    fx = FX(BOY if kid else GROWN, cx, cy, s, yaw, pitch, roll, drop, .03 if laughing else 0, anchor); k = fx.k * lw; sway = lambda i, a=1.: A(math.sin(t * .45 + i * 1.7) + .5 * math.sin(t * .95 + i * 2.6), .5 * math.sin(t * .6 + i)) * a * s * .022 * blow
    # back: hair, neck
    B = Cel(); r = np.random.default_rng(seed)
    core = fx.H.P3(A((-.72, .5, -.2), (-.80, .1, -.3), (-.76, -.4 * hairlen, -.3), (-.5, -.72 * hairlen, -.2), (.5, -.72 * hairlen, -.2), (.76, -.4 * hairlen, -.3), (.80, .1, -.3), (.72, .5, -.2))); B.fill(spl(core, True), pal['hair2'])
    for i in range(16):
        u = i / 15.; x = -.82 + 1.64 * u; y0 = -.05 - .35 * (1 - abs(2 * u - 1)); ln = r.uniform(.34, .58) * hairlen * (1.15 - .3 * abs(2 * u - 1)); q = fx.H.P3(A((x - .10, y0 + .30, -.32), (x * 1.06 + (.10 + .06 * blow) * (u > .3), y0 - ln, -.30), (x + .12, y0 + .30, -.32))); clump(B, q[0], q[1] + sway(i, 1.4), q[2], pal, k, seed + i, .16, .08 * (-1) ** i, 3, .5 * (i % 2), fill=pal['hair2'] if i % 2 else pal['hair'])
    w = .36 if kid else .42; NK = spl(fx.H.P3(A((-w, -.4, -.1), (-w, -.9, -.05), (-w - .02, -1.25, -.05), (-.62, -1.6, 0), (-1.1, -1.9, 0), (1.1, -1.9, 0), (.62, -1.6, 0), (w + .02, -1.25, -.05), (w, -.9, -.05), (w, -.4, -.1))), True, (0, 4, 5, 9)); B.fill(NK, pal['skin']); B.clip(NK)
    B.fill(fx.H.P3(A((-.8, -.5, 0), (-.42, -1.12 - drop, .1), (0, -1.28 - drop, .2), (.40, -1.30 - drop, .1), (.5, -1.7, .1), (.45, -2.4, 0), (1.4, -2.4, 0), (1.2, -.5, 0)) if side > 0 else A((.8, -.5, 0), (.42, -1.12 - drop, .1), (0, -1.28 - drop, .2), (-.40, -1.30 - drop, .1), (-.5, -1.7, .1), (-.45, -2.4, 0), (-1.4, -2.4, 0), (-1.2, -.5, 0))), pal['shade'])
    for sg in (() if kid else (-1, 1)): B.line(spl(fx.H.P3(A((sg * .30, -1.05, .2), (sg * .20, -1.35, .26), (sg * .08, -1.62, .25)))), 3.2 * k, pal['line'], taper=(.35, .5), seed=seed + 50 + sg)
    if not kid: edge_hatch(B, fx.H.P3(A((-.36, -1.14 - drop, .1), (0, -1.30 - drop, .2), (.36, -1.32 - drop, .1))), 12, 80 * k, 1.75, 2.2 * k, pal['line'], seed + 55); B.unclip(); outline(B, NK, 4. * k, pal['line'], seed + 33)
    # face
    C = Cel(); F = fx.face(); sh = pal['shade']
    for e in (fx.H.ear(-1), fx.H.ear(1)):
        if e['vis'] > 0: C.fill(e['outer'], pal['skin']); C.fill(e['inner'], sh); outline(C, e['outer'], 4. * k, pal['line'], seed + 1); C.line(e['inner'], 2.4 * k, pal['line'], seed=seed + 2)
    C.fill(F, pal['skin']); C.clip(F)
    edge = mir([(.30, .9), (.36, .50), (.33, .30), (.50, .16), (.57, .0), (.52, -.14), (.46, -.24), (.53, -.36), (.585, -.50), (.54, -.66), (.44, -.82), (.30, -.96), (.18, -1.08)], side)
    C.fill(np.vstack([fx.J(fx.S(edge)), fx.H.P3(A((.2 * side, -1.5, .5), (1.5 * side, -1.5, .2), (1.5 * side, 1.2, .2)))]), sh)
    lit = mir([(.48, .95), (.55, .5), (.52, .25), (.585, .05), (.56, -.15), (.53, -.28), (.58, -.42), (.55, -.6), (.45, -.8), (.32, -.95)], -side); C.fill(np.vstack([fx.J(fx.S(lit)), fx.H.P3(A((-.2 * side, -1.5, .5), (-1.5 * side, -1.5, .2), (-1.5 * side, 1.2, .2)))]), pal['lite'])
    if hat:                                                                                  # the brim's shadow across the forehead
        C.fill(np.vstack([fx.S([(-.8, .20), (-.5, .165), (-.2, .15), (.1, .155), (.45, .18), (.8, .24)]), fx.H.P3(A((1.2, 1.2, .2), (-1.2, 1.2, .2)))]), pal['deep'], .78)
    for sg in (() if kid else (-1, 1)): C.fill(fx.S(mir([(.06, .09), (.12, .02), (.15, -.04), (.10, -.09), (.055, -.14), (.045, 0)], sg), .02, True), sh)
    C.unclip()
    edge_f = lambda a, b, n, L, an, sd: edge_hatch(C, fx.J(fx.S(edge[a:b])), n, L * k, an, 2.2 * k, pal['line'], seed + sd)
    C.clip(F); ha = 2.55 if side > 0 else .6; edge_f(5, 8, 8, 60, ha, 150); edge_f(8, 11, 8, 62, ha, 151); edge_f(2, 5, 6, 50, ha, 152); C.unclip()
    nose(C, fx, pal, lw, 1.05, 1, seed + 90) if not kid else kidnose(C, fx, pal, lw, seed + 90)
    if scars:                                                                                # three parallel claw marks over HIS left eye (viewer's right)
        for i in range(3):
            x0 = .205 + .095 * i; a = fx.S([(x0 + .055, .265 - .02 * i), (x0 + .025, .09), (x0 - .010, -.08), (x0 - .045, -.215 + .02 * i)], .006); C.line(a, 8.5 * k, mix3(pal['shade'], '#8f2f2a', .55), taper=(.45, .45), press=.25, seed=seed + 60 + i); C.line(a + A(2.6, 1.) * k, 2.2 * k, pal['line'], taper=(.4, .5), seed=seed + 64 + i); C.line(a - A(3., .5) * k, 2.0 * k, pal['lite'], taper=(.4, .6), seed=seed + 68 + i)
    if stitches:
        a = fx.S([(.20, -.185), (.30, -.20), (.40, -.185)]); C.line(a, 4. * k, pal['line'], taper=(.2, .2), seed=seed + 70)
        for x in (.26, .345): C.line(fx.S([(x - .006, -.155), (x + .004, -.235)]), 3.4 * k, pal['line'], taper=(.2, .2), seed=seed + 71)
    # eyes / brows / mouth by expression
    hb = dict(pal, hair=mix3(pal['hair'], '#1a0d10', .45), hl=pal['hair2'])
    if expr == 'laugh':
        for sg in (-1, 1):
            arc_eye(C, fx, sg, pal, y=-.02, h=.10, w=.155, lw=lw, seed=seed + 100); brow(C, fx, sg, hb, [(.10, .15), (.11, .205), (.32, .275), (.55, .18), (.33, .225)], lw, seed + 120)
            C.line(fx.S(mir([(.145, -.335), (.25, -.35), (.36, -.38), (.50, -.46), (.52, -.60)], sg)), 4.2 * k, pal['line'], taper=(.3, .5), seed=seed + 80 + sg); C.line(fx.S(mir([(.22, -.165), (.34, -.15), (.46, -.18)], sg)), 2.8 * k, pal['line'], taper=(.5, .5), seed=seed + 84 + sg); C.fill(fx.S(mir([(.40, -.42), (.50, -.46), (.52, -.60), (.485, -.58), (.465, -.48)], sg), 0, True, (0, 2)), sh)
        laugh(C, fx, .40, -.46, -.50, -.50 - .42 * jaw, pal, seed + 130, low=not kid, lw=lw, gap_tooth=kid)
    elif expr == 'cry':
        wb = .010 * math.sin(t * 2.1) ; ly = look[1]
        for sg in (-1, 1):
            eye(C, fx, sg, pal, .86, .05, look[0], ly, r=.058, w=.205, h=.205, x=.345, y=-.05, slant=-.06, lw=lw * 1.1, seed=seed + 100, bag=False); C.line(fx.S(mir([(.20, -.245), (.35, -.275), (.50, -.225)], sg)), 2.6 * k, pal['line'], taper=(.5, .5), seed=seed + 108 + sg)
            brow(C, fx, sg, hb, [(.10, .255), (.10, .315), (.33, .285), (.57, .165), (.34, .232)], lw, seed + 120); C.line(fx.S(mir([(.06, .22), (.045, .15)], sg), .03), 2.6 * k, pal['line'], taper=(.3, .5), seed=seed + 128 + sg)
            C.fill(fx.S(mir([(.19, -.17), (.34, -.215), (.50, -.15), (.46, -.11), (.34, -.135), (.23, -.12)], sg), .01, True), '#e4f1fa', .92); C.line(fx.S(mir([(.25, -.145), (.38, -.155)], sg), .01), 4.2 * k, '#ffffff', taper=(.3, .3), seed=1)
        mw = .24; pts = [(-mw, -.640 - .02), (-.17, -.600 + wb), (-.09, -.618 - wb), (0, -.590 + wb), (.09, -.618 - wb), (.17, -.600 + wb), (mw, -.645 - .02)]; C.line(fx.S(pts), 6. * k, pal['line'], taper=(.2, .2), seed=seed + 130); C.line(fx.S([(-.12, -.72), (0, -.70 + wb), (.12, -.72)]), 3.4 * k, pal['line'], taper=(.4, .4), seed=seed + 131)
        for i, x in enumerate((-.10, -.03, .05, .11)): C.line(fx.J(fx.S([(x, -.80), (x + .012, -.84 + wb)])), 2.6 * k, pal['line'], taper=(.4, .4), seed=seed + 134 + i)
    elif expr == 'warm':                                                                     # eyes shut, the big brother's smile
        for sg in (-1, 1):
            arc_eye(C, fx, sg, pal, y=-.01, h=.085, w=.15, lw=lw * 1.1, seed=seed + 100); brow(C, fx, sg, hb, [(.10, .165), (.11, .215), (.32, .285), (.55, .185), (.33, .235)], lw, seed + 120); C.line(fx.S(mir([(.145, -.335), (.25, -.37), (.36, -.43), (.43, -.54)], sg)), 4.0 * k, pal['line'], taper=(.3, .5), seed=seed + 80 + sg)
        grin(C, fx, .375, -.485, -.575, -.700, pal, asym=.014, seed=seed + 130, lw=lw)
    else:                                                                                    # soft smile
        for sg in (-1, 1):
            eye(C, fx, sg, pal, .52, .35, look[0], look[1], r=.034, w=.165, h=.10, lw=lw * 1.1, seed=seed + 100); brow(C, fx, sg, hb, [(.09, .13), (.09, .185), (.32, .25), (.55, .15), (.33, .195)], lw, seed + 120)
            C.line(fx.S(mir([(.145, -.335), (.23, -.39), (.31, -.50)], sg)), 3.4 * k, pal['line'], taper=(.3, .6), seed=seed + 80 + sg)
        C.line(fx.S([(-.27, -.585), (-.15, -.635), (0, -.650), (.16, -.63), (.29, -.565)]), 6. * k, pal['line'], taper=(.2, .2), seed=seed + 130); C.line(fx.S([(.27, -.60), (.315, -.56), (.30, -.52)]), 3.4 * k, pal['line'], taper=(.4, .5), seed=seed + 131); C.line(fx.S([(-.11, -.735), (0, -.75), (.11, -.735)]), 3.2 * k, pal['line'], taper=(.4, .4), seed=seed + 132)
    if not kid:                                                                              # stubble
        C.clip(F); sr = np.random.default_rng(seed + 5)
        for i in range(150):
            u = sr.uniform(-1, 1); y = -.80 - .18 * sr.random() + .34 * abs(u) ** 2.2; x = u * .55
            if sr.random() < .22: x, y = sr.uniform(-.24, .24), -.455 + sr.uniform(-.03, .02)
            q = fx.J(fx.P([(x, y)]))[0]; C.line(np.stack([q, q + A(sr.normal(0, 1.5), 7 + sr.random() * 5) * k]), 2.0 * k, pal['line'], taper=(.1, .6), seed=i, alpha=.75)
        C.unclip()
    C.line(fx.J(fx.S([(-.10, -.915), (0, -.94), (.11, -.915)])), 3. * k, pal['line'], taper=(.45, .45), seed=seed + 145)
    if kid:
        for sg in (-1, 1):
            for i in range(4): C.line(fx.S(mir([(.36 + i * .045, -.27), (.34 + i * .045, -.335)], sg)), 2.4 * k, mix3(pal['shade'], '#a02a2a', .5), taper=(.3, .5), seed=seed + 150 + i)
    if tears > 0:
        for sg in (-1, 1):
            for j, (x0, x1, w_, ph) in enumerate(((.30, .34, 30, 0.), (.50, .575, 22, .4))):
                q = [(x0, -.20), (x0 + (x1 - x0) * .35, -.34), (x1, -.52), (x1 - .02, -.72), (x1 - .10, -.90)]; n = len(q); u = min(1., max(0.1, (tf * 1.25 - ph * .4))); P = fx.S(mir(q, sg), .02, n=8); m = max(4, int(len(P) * u)); tear(C, P[:m], w_ * k * (1.25 if sg * side < 0 else 1.), seed=seed + 160 + j + sg)
    if kid and expr == 'cry':                                                                # pen work + rain-wet shine
        C.clip(F); sr = np.random.default_rng(seed + 9)
        for L_, n_, l_, an, sd_ in (([(-.58, -.30), (-.46, -.36), (-.32, -.35)], 8, 44, 1.9, 180), ([(.30, -.35), (.44, -.36), (.56, -.30)], 8, 44, 1.2, 181), ([(-.14, -.80), (0, -.86), (.16, -.80)], 9, 40, 1.6, 182), ([(.10, .06), (.13, -.06), (.12, -.16)], 6, 34, 2.5, 183), ([(-.44, .15), (-.15, .12), (.15, .13), (.44, .17)], 16, 36, 1.7, 184), ([(.46, -.50), (.40, -.66), (.30, -.80)], 10, 44, 2.6, 185)): edge_hatch(C, fx.J(fx.S(L_)), n_, l_ * k, an, 2.0 * k, pal['line'], seed + sd_)
        for L_ in ([(-.64, -.10), (-.655, -.28), (-.63, -.46)], [(-.56, -.60), (-.48, -.74), (-.38, -.85)], [(-.10, -.295), (-.06, -.315)], [(-.20, -.925), (-.05, -.955)], [(-.60, .08), (-.645, -.02)], [(-.36, -.40), (-.30, -.43)]): C.line(fx.S(L_, .01), 6.5 * k, '#fff4d6', taper=(.4, .55), press=.45, seed=int(sr.integers(999)), alpha=.92)
        C.unclip(); a_ = np.linspace(0, 2 * math.pi, 18)[:-1]
        for x, y, r_ in ((-.66, .06, .026), (-.52, .02, .018), (-.22, .085, .022), (.10, .10, .017), (.64, -.10, .02), (-.40, -.50, .019), (.22, -.48, .016), (.50, -.62, .018), (-.14, -.74, .015), (.04, -1.0, .024)):
            c_ = fx.J(fx.P([(x, y)], .01))[0]; R_ = r_ * s; Q = np.c_[c_[0] + R_ * .8 * np.cos(a_), c_[1] + R_ * np.sin(a_) * np.where(np.sin(a_) < 0, 1.35, .9)]; C.fill(Q, '#ffe9c4', .75); C.fill(Q * .5 + (c_ + A(R_ * .22, R_ * .3)) * .5, pal['shade'], .5); C.fill(np.c_[c_[0] - R_ * .3 + R_ * .26 * np.cos(a_), c_[1] - R_ * .35 + R_ * .30 * np.sin(a_)], '#ffffff'); outline(C, Q, 1.8 * k, pal['line'], seed + 190)
    outline(C, F, 6.0 * k, pal['line'], seed + 170); face = C.layer()
    # front hair (under the brim, or free)
    Hc = Cel(); y0 = .50 if hat else .98; L = []
    for i, (x, tx, ty, w_) in enumerate(((-.56, -.62, -.02, .22), (-.36, -.44, .12, .22), (-.16, -.16, .08, .24), (.06, .12, .16, .22), (.26, .36, .06, .22), (.46, .60, .10, .20))): L.append((A(x - w_ / 2, y0, .50), A(tx, ty - (0 if hat else .06), .72 - .3 * abs(tx)), A(x + w_ / 2, y0, .50), .18, -.10 if i % 2 else .08))
    sl = (.62 if kid else 1.0) * hairlen
    for sg in (-1, 1):
        for i, (x, tx, ty) in enumerate(((.66, .70, -.40), (.74, .80, -.62), (.70, .66, -.16))): L.append((A(sg * (x - .10), .48, .20 - .1 * i), A(sg * tx + .05 * blow, ty * sl, .14 - .12 * i), A(sg * (x + .06), .42, .02 - .1 * i), .16, .07 * sg))
    for i, (a, tp, b, bu, be) in enumerate(L): q = fx.H.P3(np.stack([a, tp, b])); clump(Hc, q[0], q[1] + sway(i + 30, 1.2), q[2], pal, k, seed + 200 + i, bu, be, 4, .7 if i % 2 == 0 else .35, edge=pal['line'], w=3.0)
    if kid and expr == 'cry':                                                                # soaked strands: long glints, a bead on each tip
        for i, (a, tp, b, bu, be) in enumerate(L):
            q = fx.H.P3(np.stack([a, tp, b])); e_ = q[1] + sway(i + 30, 1.2); m_ = (q[0] + q[2]) / 2; Hc.line(spl([m_ + (e_ - m_) * .22 + (q[0] - m_) * .45, m_ + (e_ - m_) * .56 + (q[0] - m_) * .25, m_ + (e_ - m_) * .93]), 4.0 * k, '#ffd9a0', taper=(.3, .6), press=.4, seed=i, alpha=.85); Hc.line(spl([m_ + (e_ - m_) * .30 + (q[2] - m_) * .35, m_ + (e_ - m_) * .60 + (q[2] - m_) * .18, m_ + (e_ - m_) * .88]), 2.4 * k, '#ff9a5a', taper=(.3, .6), seed=20 + i, alpha=.8); a_ = np.linspace(0, 6.283, 16)[:-1]; Hc.fill(np.c_[e_[0] + 5 * k * np.cos(a_), e_[1] + 4 * k + 7 * k * np.sin(a_)], '#fff0cc', .9)
    LY = dict(back=B.layer(), face=face, hair=Hc.layer(), fx=fx, pal=pal)
    for n_, a in (('back', .8), ('face', 1.), ('hair', 1.)): LY[n_] = rimlit(LY[n_], rimdir[0], rimdir[1], pal, k, a, 20, 8)
    return LY


def mix3(a, b, t):
    a = rgb(a) if isinstance(a, str) else np.asarray(a); b = rgb(b) if isinstance(b, str) else np.asarray(b); return a * (1 - t) + b * t


def flat(LY, order=('back', 'face', 'hair')):
    o = np.zeros((HH, W, 4), f32)
    for n in order: o = over_l(o, LY[n])
    return o


def head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='laugh', light='day', f=0, **kw): return flat(draw(cx, cy, s, yaw, pitch, roll, expr=expr, light=light, f=f, **kw))


# ------------------------------------------------------------------ the body on the pier: joint poses in metres (x right, y up)
KNEEL = dict(hip=(-.06, .52), chest=(.05, .98), neck=(.10, 1.10), head=(.165, 1.245), kn=(.06, .07), fn=(-.46, .07), kf=(.42, .56), ff=(.42, .03), hem=[(-.14, 1.10), (-.34, .98), (-.46, .68), (-.60, .36), (-.82, .10), (-1.0, -.01), (-.62, -.03), (-.30, .0), (-.14, .06), (-.06, .34), (-.02, .64), (.02, .92), (.06, 1.08)])
RISE = dict(hip=(-.02, .74), chest=(.14, 1.22), neck=(.20, 1.35), head=(.29, 1.48), kn=(.14, .34), fn=(-.30, .06), kf=(.34, .52), ff=(.30, .03), hem=[(-.06, 1.35), (-.28, 1.22), (-.42, .90), (-.56, .56), (-.74, .26), (-.86, .10), (-.56, .06), (-.30, .10), (-.14, .20), (-.06, .5), (.0, .86), (.08, 1.16), (.14, 1.34)])
STAND = dict(hip=(0, 1.0), chest=(.02, 1.50), neck=(.03, 1.63), head=(.07, 1.80), kn=(-.04, .54), fn=(-.10, .03), kf=(.14, .54), ff=(.18, .03), hem=[(-.16, 1.64), (-.34, 1.54), (-.42, 1.14), (-.48, .70), (-.56, .34), (-.60, .22), (-.40, .18), (-.22, .20), (-.12, .28), (-.10, .6), (-.08, 1.0), (-.06, 1.40), (-.02, 1.62)])


def lerp_body(a, b, t): return {k: np.asarray(a[k], float) * (1 - t) + np.asarray(b[k], float) * t for k in a}


def body(cel, p, ox, oy, u, f=0, hand=(.62, 1.10), pal=None, sleeve=True, lw=1., blow=1., part='all', arm=True, snap=1., slen=.80):
    """Shanks on one knee .. standing.  (ox, oy) ground point px, u px per metre.  hand = right hand target (m)"""
    pal = pal or LIGHT['sunset']; t = f // 2 * 2; q = lambda L: np.array([(ox + a * u, oy - b * u) for a, b in L]); w = lambda i, a: a * math.sin(t * .55 - i * .9) * blow; ln = pal['line']
    cl, cl2, cl3 = '#241f2a', '#16121b', '#3c3442'; sh_, shs = '#efe2cf', '#b79a88'; tr, trs = '#6a5240', '#43312c'
    def limb(a, b, wa, wb, c, cs, sd, w_=3.6):
        a, b = A(*a), A(*b); d = b - a; n = A(-d[1], d[0]) / (np.hypot(*d) + 1e-6); P = [a + n * wa - d * .06, a + d * .3 + n * wa * 1.05, b + n * wb, b + d * .05, b - n * wb, a + d * .3 - n * wa * 1.0, a - n * wa - d * .06]; S = spl(q(P), True, (0, 6)); cel.fill(S, c); cel.clip(S); cel.fill(spl(q([a - n * wa * .1, a + d * .4 - n * wa * .25, b - n * wb * .2, b - n * wb * 2, a - n * wa * 2]), True, (0, 2, 3, 4)), cs)
        for i in range(3): v = .2 + .28 * i; cel.line(spl(q([a + d * v - n * wa * .9, a + d * (v + .06) - n * wa * .2, a + d * (v + .02) + n * wa * .4]), n=8), 2.2 * lw, ln, taper=(.2, .7), seed=sd + i)
        cel.unclip(); outline(cel, S, w_ * lw, ln, sd)
    def sandal(x, y, sg=1):
        S = spl(q([(x - .06 * sg, y + .10), (x - .08 * sg, y), (x + .02 * sg, y - .02), (x + .17 * sg, y - .02), (x + .19 * sg, y + .03), (x + .10 * sg, y + .07), (x + .04 * sg, y + .11)]), True, (1, 4)); cel.fill(S, pal['skin']); outline(cel, S, 3.0 * lw, ln, 5); cel.fill(q([(x - .09 * sg, y - .01), (x + .20 * sg, y - .01), (x + .20 * sg, y - .035), (x - .09 * sg, y - .035)]), '#3b2a22'); cel.line(q([(x + .02 * sg, y + .09), (x + .13 * sg, y + .0)]), 4. * lw, '#2a1c1a')
    hip, chest, neck = A(*p['hip']), A(*p['chest']), A(*p['neck'])
    # far leg, near leg
    limb(hip + A(.06, .02), p['kf'], .115, .085, trs, '#2f2322', 10); limb(p['kf'], A(*p['ff']) + A(0, .08), .088, .06, trs, '#2f2322', 14); sandal(p['ff'][0], p['ff'][1] + .01)
    limb(hip, p['kn'], .125, .095, tr, trs, 18); limb(p['kn'], A(*p['fn']) + A(0, .04), .095, .06, tr, trs, 22); sandal(p['fn'][0] + .0, p['fn'][1] + .0, 1 if p['fn'][1] < .05 else -1)
    # torso: white shirt, sash
    ax = chest - hip; n = A(-ax[1], ax[0]) / np.hypot(*ax); n = n if n[0] < 0 else -n
    S = spl(q([neck + n * .07, chest + n * .20 + ax * .12, hip + n * .16 + ax * .5, hip + n * .13, hip - n * .13, hip - n * .17 + ax * .5, chest - n * .20 + ax * .08, neck - n * .06]), True, (0, 3, 4, 7)); cel.fill(S, sh_); cel.clip(S)
    cel.fill(spl(q([neck, chest + n * .02, hip + ax * .5 + n * .05, hip, hip + n * .3, chest + n * .4, neck + n * .2]), True), shs)
    for i in range(6): v = .1 + .15 * i; cel.line(spl(q([hip + ax * v - n * .16, hip + ax * (v + .08) - n * .05, hip + ax * (v + .05) + n * .07]), n=8), 2.4 * lw, ln, taper=(.2, .7), seed=30 + i)
    cel.fill(q([neck - n * .05, neck - n * .02 - ax * .0, chest - n * .06 - ax * .30, chest - n * .12 - ax * .02]), pal['skin']); cel.unclip(); outline(cel, S, 3.8 * lw, ln, 36)
    cel.fill(spl(q([hip + n * .17 + ax * .16, hip - n * .17 + ax * .14, hip - n * .16 - ax * .02, hip + n * .16 - ax * .0]), True, (0, 1, 2, 3)), '#8e2a24'); cel.line(q([hip + n * .16 + ax * .08, hip - n * .16 + ax * .06]), 2.4 * lw, ln)
    if sleeve:                                                                               # the EMPTY left sleeve, flapping in the gap
        ws = lambda i, a: a * snap * (math.sin(t * .92 - i * 1.05) + .55 * math.sin(t * 1.9 - i * 1.9)); s0 = chest - n * .16 + ax * .05; E = [s0 + A(0, .07), s0 + A(.16, .04 + ws(1, .02)), s0 + A(.32, -.02 + ws(2, .05)), s0 + A(.46, .02 + ws(3, .08)), s0 + A(.62, -.02 + ws(4, .11)), s0 + A(.60, -.13 + ws(4.3, .10)), s0 + A(.44, -.11 + ws(3.2, .07)), s0 + A(.28, -.15 + ws(2.2, .05)), s0 + A(.10, -.16), s0 + A(-.02, -.10)]; w = ws
        E = [s0 + (e - s0) * A(slen if i not in (0, 8, 9) else 1., 1.) for i, e in enumerate(E)]; S = spl(q(E), True, (0, 4, 5, 9)); cel.fill(S, sh_); cel.clip(S); cel.fill(spl(q([s0 + A(0, -.06), s0 + A(.16, -.10 + w(1.5, .03)), s0 + A(.34, -.12 + w(3, .07)), s0 + A(.5, -.17 + w(4, .1)), s0 + A(.4, -.4), s0 + A(0, -.4)]), True), shs); cel.unclip(); outline(cel, S, 3.6 * lw, ln, 40); cel.line(spl(q([s0 + A(.585 * slen, .0 + ws(4, .11)), s0 + A(.575 * slen, -.06 + ws(4.1, .105)), s0 + A(.57 * slen, -.12 + ws(4.3, .10))])), 3. * lw, ln, taper=(.3, .3), seed=41)
        for i in range(3): cel.line(spl(q([s0 + A(.06 + .1 * i, .02 - .03 * i + w(1 + i, .03)), s0 + A(.12 + .1 * i, -.06 + w(1 + i, .03)), s0 + A(.15 + .11 * i, -.13 + w(1.6 + i, .04))]), n=8), 2.4 * lw, ln, taper=(.2, .7), seed=42 + i)
    def do_arm():                                                                            # right arm: rolled sleeve, forearm
        sh0 = chest + n * .02 + ax * .06; hd = A(*hand); d = hd - sh0; L = np.hypot(*d); up_, fo = .30, .30; Lc = min(L, up_ + fo - .005); ca = (up_ ** 2 + Lc ** 2 - fo ** 2) / (2 * up_ * Lc + 1e-6); aa = math.atan2(d[1], d[0]) - math.acos(np.clip(ca, -1, 1)); el = sh0 + A(math.cos(aa), math.sin(aa)) * up_
        if L > up_ + fo: hd = sh0 + d / L * (up_ + fo - .005)
        limb(sh0, el, .085, .074, sh_, shs, 70, 4.); limb(el, hd, .062, .046, pal['skin'], pal['shade'], 76, 3.8); fd = (hd - el) / (np.hypot(*(hd - el)) + 1e-6); fn_ = A(-fd[1], fd[0]); S = spl(q([el - fd * .05 + fn_ * .085, el + fd * .07 + fn_ * .088, el + fd * .09 - fn_ * .088, el - fd * .05 - fn_ * .085]), True, (1, 2)); cel.fill(S, sh_); cel.clip(S); cel.fill(q([el - fd * .1 - fn_ * .01, el + fd * .15 - fn_ * .01, el + fd * .15 - fn_ * .2, el - fd * .1 - fn_ * .2]), shs); cel.unclip(); outline(cel, S, 3.4 * lw, ln, 79); cel.line(spl(q([el + fd * .045 + fn_ * .086, el + fd * .055, el + fd * .05 - fn_ * .086])), 2.4 * lw, ln); return hd, fd
    if arm == 'under': do_arm()
    w = lambda i, a: a * math.sin(t * .55 - i * .9) * blow
    # the cloak over the back
    hm = np.array(p['hem'], float).copy()
    for i in range(3, 10): hm[i] += A(w(i * .8, .035), w(i * .8 + 1, .022)) * (1 if 3 < i < 9 else .5)
    S = spl(q(hm), True, (0, 5, 6, 12)); cel.fill(S, cl); cel.clip(S); m = hm.mean(0)
    for i, (a, b, wd) in enumerate(((2, 4, 30), (1, 5, 40), (12, 6, 34), (11, 7, 30), (10, 8, 22))):
        P = spl(q([hm[0] * .6 + hm[a] * .4, (hm[a] + hm[b]) / 2 * .7 + m * .3, hm[b] * .96 + m * .04]), n=12); cel.line(P, wd * u / 430., cl2, taper=(.8, .1), press=.3, seed=50 + i); cel.line(P + A(wd * .5 * u / 430., 0), 2.4 * lw, cl3, taper=(.4, .4), seed=56 + i)
    cel.unclip(); outline(cel, S, 4.4 * lw, ln, 60); c0 = [neck + A(-.17, .10), neck + A(-.02, .12), neck + A(.09, .04), neck + A(.06, -.07), neck + A(-.12, -.09), neck + A(-.21, .0)]; S = spl(q(c0), True, (0, 2)); cel.fill(S, cl3); outline(cel, S, 4. * lw, ln, 61)
    if arm and arm != 'under': return do_arm()
    return A(*hand), A(1, 0)


def kidnose(C, fx, pal, lw, seed):
    k = fx.k * lw; N = fx.N; C.fill(spl(N([(-.05, -.30, .06), (.0, -.335, .09), (.06, -.30, .06), (.04, -.345, .05), (-.03, -.35, .05)]), True), pal['shade']); C.line(spl(N([(-.045, -.275, .07), (-.035, -.315, .10), (.01, -.335, .11), (.05, -.305, .08)])), 3.8 * k, pal['line'], taper=(.5, .4), seed=seed)
