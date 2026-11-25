"""Nami.  head(...) -> layer, draw(...) -> dict of layers (back, body, face, front, rig), g1(f) -> layer.  hair='long' (c2, g1) or 'short' (d2)."""
from .lkit import *

PROPS = dict(a=[(-1.0, .080), (-.965, .135), (-.89, .245), (-.78, .372), (-.62, .55), (-.42, .665), (-.18, .745), (.08, .765), (.35, .76), (.6, .715), (.8, .575), (.93, .345), (1.0, .05)], nose=(.060, -.17, -.305, -.365, .118))
EYE = dict(up=[(-1, -.16), (-.87, .40), (-.47, .86), (.04, 1.02), (.54, .90), (.91, .48), (1.06, .08)], lo=[(-1, -.16), (-.87, -.46), (-.50, -.74), (-.04, -.88), (.44, -.78), (.85, -.40), (1.06, .08)], c=(.380, -.015), w=.216, h=.205, iris=.60, irisv=.76, lid=.046, lash=[(.94, .28, .14), (.85, .22, .30), (.74, .10, .34)], lash2=[(.74, .10, -.16)], col=('#4a200c', '#9c5a22', '#e8a552'))
BROW = {'in': (.150, .335), 'mid': (.40, .470), 'out': (.655, .350), 't': .042}
LIGHT = dict(
    day=dict(skin='#f7c79b', shade='#d08a62', hair=dict(base='#f18a2b', shade='#c2551b', deep='#8c3613', deep2='#6a260e', light='#ffc364', line='#4a1a0c', contour=INK), rim='#fff0c2', d=(.74, .67)),
    storm=dict(skin='#f3c4a1', shade='#a9708a', hair=dict(base='#f5862a', shade='#b84a28', deep='#7a2c34', deep2='#54203a', light='#ffc66e', line='#42160f', contour=INK), rim='#e6f6ff', d=(-.80, .60)),
    dusk=dict(skin='#f8c894', shade='#c67c4e', hair=dict(base='#e27f27', shade='#a44818', deep='#742e12', deep2='#54200e', light='#ffc05c', line='#42160b', contour=INK), rim='#fff0b4', d=(-.82, .57)),
    g1=dict(skin='#f0bd93', shade='#b97a5e', hair=dict(base='#ee8a2e', shade='#b5501f', deep='#833515', deep2='#60260f', light='#ffca72', line='#48190c', contour=INK), rim='#fff6d8', d=(.95, .32)))


def style(light):
    p = dict(LIGHT[light]) if isinstance(light, str) else dict(light)
    p.update(nosea=3, eye=EYE, brow=BROW, browcol='#5e260e', mouth='#661222', line2='#8c4630', blush='#d65a3a', lip=None); p.setdefault('cloth', dict(base='#f3efe6', stripe='#2f6db4', shade='#6d6a9a', sa=.55)); return p


HAIRLINE = [(-.74, -.10), (-.70, .26), (-.56, .52), (-.20, .66), (.16, .74), (.46, .60), (.66, .34), (.74, -.06)]


def fp(h, P): return np.array([(x, y, float(h.z(x, y)) + d) for x, y, d in P])


def fringe(F, t, sway, amp, hair, loose=0.):
    h = F.h; s = F.s; H3 = [(.22, .40, .3), (.50, .74, .42)]; K = dict(inner=2)
    L = []
    sh = h.yaw * .22
    def a(P, w, ph, sk=1., **o): q = dict(K); q.update(o); P = [(x + sh * float(sm((abs(x) - .45) / .3)), y, d) for x, y, d in P]; L.append(lock3(F, fp(h, P), w * s, t, np.asarray(sway) * sk, amp * sk, ph, **q))
    if hair == 'long':
        a([(.05, .97, .0), (-.42, .87, .08), (-.72, .52, .12), (-.83, .08, .10), (-.80, -.38, .12), (-.86, -.78, .14), (-.74, -1.12, .16)], .34, .1, 1.5, hi=[(.20, .42, .3)], root=.5)
        a([(.24, .97, .05), (.54, .83, .10), (.76, .48, .12), (.83, .04, .10), (.78, -.40, .12), (.84, -.78, .12), (.72, -1.08, .14)], .31, .5, 1.5, hi=[(.24, .40, .3)], root=.5)
    else:
        a([(.05, .97, .0), (-.42, .87, .08), (-.72, .52, .12), (-.82, .08, .10), (-.78, -.34, .12), (-.80, -.62, .10), (-.96, -.80, .04)], .33, .1, 1.1, hi=[(.20, .42, .3)], root=.5)
        a([(.24, .97, .05), (.54, .83, .10), (.75, .48, .12), (.82, .04, .10), (.76, -.34, .12), (.80, -.60, .10), (.95, -.74, .04)], .30, .5, 1.1, hi=[(.24, .40, .3)], root=.5)
        a([(.00, .95, .0), (-.50, .80, .06), (-.76, .42, .06), (-.82, .0, .04), (-.74, -.40, .06), (-.62, -.72, .12)], .20, .8, 1.0, hi=None, tone=1)
    B_ = dict(hiw=.12, r0=.10)
    a([(.36, .96, .06), (.54, .82, .15), (.67, .56, .17), (.73, .30, .15), (.80, .06, .12), (.76, -.22, .12)], .36, .65, .9, hi=[(.22, .46, .2)], root=.8, k=2.2, **B_)
    a([(.26, .985, .05), (-.04, .90, .16), (-.36, .70, .20), (-.60, .42, .19), (-.74, .12, .16), (-.80, -.16, .13), (-.70, -.44, .13)], .46, .25, 1.15, hi=[(.20, .42, .30), (.48, .62, .2)], root=.8, k=2.0, inner=3, **B_)
    a([(.30, .97, .09), (.08, .85, .19), (-.18, .64, .22), (-.42, .40, .20), (-.60, .20, .16), (-.74, .06, .12)], .40, .4, 1.0, hi=[(.22, .44, .28), (.52, .68, .1)], root=.8, k=2.0, inner=3, **B_)
    a([(.32, .95, .11), (.16, .80, .20), (-.04, .58, .22), (-.24, .38, .19), (-.42, .24, .15), (-.54, .18, .11)], .34, .15, 1.0, hi=[(.24, .48, .24)], root=.8, k=2.0, **B_)
    a([(.34, .94, .12), (.24, .77, .20), (.10, .56, .21), (-.06, .38, .18), (-.22, .27, .14)], .27, .9, 1.0, hi=[(.26, .50, .2)], root=.8, k=1.9, **B_)
    a([(.36, .93, .12), (.44, .76, .18), (.50, .57, .18), (.52, .42, .15), (.58, .28, .12)], .24, .5, .9, hi=[(.28, .52, .1)], root=.8, k=1.9, **B_)
    a([(.35, .93, .13), (.31, .75, .20), (.24, .56, .20), (.14, .41, .16), (.06, .30, .13)], .17, .6, 1.0, hi=None, inner=1, root=.8, k=1.8, r0=.10)
    if loose: a([(.18, .96, .12), (.24, .76, .18), (.25, .54, .18), (.20, .36, .15)], .07, .2, 1.1, hi=None, inner=0)
    return L


def back_hair(F, t, wa, ws, hair, lagfn=None, seed=0, length=1.):
    h = F.h; s = F.s; L = []; g = math.pi / 2 + h.roll
    if hair == 'long':
        R = [(-.60, .30, -.35), (-.66, -.05, -.25), (-.46, -.15, -.55), (-.18, -.30, -.65), (.12, -.30, -.65), (.42, -.18, -.55), (.64, .0, -.28), (.58, .32, -.35), (-.30, .10, -.8), (.28, .10, -.8)]
        for j, q in enumerate(R):
            r = rs(seed, j); root = F.X3([q])[0]; a0 = g + (-.42 * q[0]) * (1 - .6 * ws); sp_ = (j % 5 - 2) * .13; a1 = lerp(g + r.normal(0, .10), wa + sp_ + r.normal(0, .06), ws); ln = s * r.uniform(2.5, 3.3) * length
            P = strand(root, ln, a0, a1, t, amp=.24 + .12 * ws, waves=r.uniform(.85, 1.1), speed=.62 + .5 * ws, ph=j * .23 + r.random() * .15, curl=(1 if j % 2 else -1) * r.uniform(.4, 1.0), lagfn=lagfn, n=34); L.append(Lock(P, s * r.uniform(.62, .84), root=.7, k=1.7, belly=.36, n=44, inner=3, hi=[(.12, .34, .30), (.40, .58, .24), (.66, .78, .1)], hiw=.11, sh=.58, tone=1 if j >= 8 else 0))
        for j in range(5):                                            # second tier: narrower ribbons that overshoot
            r = rs(seed, j, 77); q = R[(j * 3) % len(R)]; root = F.X3([q])[0]; P = strand(root, s * r.uniform(2.6, 3.5) * length, g, lerp(g, wa + (j - 2) * .2, ws), t, amp=.34, waves=1.1, speed=.8 + .5 * ws, ph=r.random(), curl=(1 if j % 2 else -1) * r.uniform(.7, 1.3), lagfn=lagfn, n=34); L.append(Lock(P, s * r.uniform(.24, .34), root=.6, k=1.5, belly=.3, n=40, inner=2, hi=[(.15, .45, .25)], hiw=.13))
    else:
        for j, q in enumerate([(-.62, .25, -.3), (-.58, -.1, -.35), (-.35, -.25, -.55), (-.05, -.3, -.6), (.25, -.28, -.6), (.52, -.12, -.4), (.62, .22, -.3)]):
            r = rs(seed, j); root = F.X3([q])[0]; sd = -1 if q[0] < -.02 else 1; P = strand(root, s * r.uniform(.95, 1.15), g - .30 * q[0], lerp(g - sd * .25, wa, ws * .7), t, amp=.12, waves=.8, speed=.7, ph=j * .3, curl=-sd * r.uniform(.5, 1.0), lagfn=lagfn, n=28); L.append(Lock(P, s * r.uniform(.40, .50), root=.8, k=3.4, belly=.4, inner=3, hi=[(.2, .5, .3)], tone=0 if abs(q[0]) > .4 else 1))
    return L


def earring(cel, F, sd, lw):
    e = F.h.ear(sd); p = e['outer'][0]; q = p + F.dn * F.s * .05; a = np.linspace(0, TAU, 20); cel.fill(np.c_[q[0] + np.cos(a) * F.s * .045, q[1] + np.sin(a) * F.s * .045], '#f2c14a'); stroke(cel, np.c_[q[0] + np.cos(a) * F.s * .045, q[1] + np.sin(a) * F.s * .045], lw * .5)


def draw(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, hair='long', wind=(math.pi / 2, 0.), ex=None, body=None, posfn=None, step=2, st=None, sway=None, length=1., seed=11, crown=True, loose=0., post=None, under=None, browtop=True):
    """-> dict(back, body, face, front) premult layers, F rig.  body(cel, F, st) paints the bust between back hair and face.  wind=(angle, strength 0..1)"""
    st = st or style(light); pal = st['hair']; h = Head(**PROPS).pose(cx, cy, s, yaw, pitch, roll); e = dict(EXPR[expr]) if expr else {}; e.update(ex or {}); F = Face(h, e.get('jaw', 0.) - .11); lw = float(np.clip(s * .0215, 2.0, 6.6)); t = (f // step * step) / 24.; ld = -np.asarray(st['d']); sgl = -1 if ld[0] < 0 else 1
    wa, ws = wind; lf = lag(posfn, f, 5.) if posfn else None; sw = np.array([math.cos(wa), math.sin(wa) * .5]) * s * (.30 * ws) if sway is None else np.asarray(sway, float)
    sw = sw * (1 + .18 * math.sin(t * 6.1)); lg = (lf(np.array([.6]))[0] * .6 if lf else 0); out = {}
    c = Cel(); bl = back_hair(F, t, wa, ws, hair, lf, seed, length); draw_mass(c, bl, pal, lw * .62, lw * .44, seed, ld); out['back'] = lit(c.layer(), st['d'], st['rim'], lw * .5, max(3, s * .030), 1., .5)
    if body is not None: c = Cel(); body(c, F, st); out['body'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .028), .9)
    c = Cel(); C, D = face_base(c, F, st, lw, sgl, seed); fr = fringe(F, t, sw + lg, s * (.03 + .07 * ws), hair, loose)
    shadow_of(c, fr, C, (-ld[0] * s * .05, s * .085), st['shade'])
    if under: c.clip(C); under(c, F, C); c.unclip()
    c.clip(C); feats(c, F, st, e, lw, f, sgl, seed, (lambda: post(c, F, C)) if post else None); c.unclip(); contour_line(c, D, lw, seed); cheek(c, F, st, -sgl, .55, 4, lw, '#8c4630', y=-.10, x=.50)
    out['face'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .032), 1., .45)
    c = Cel()
    if crown: cap(c, F, pal, HAIRLINE, 1.10, lw * .9, seed, ld, part=(.2, .985))
    draw_mass(c, fr, pal, lw * .42, lw * .42, seed + 3, ld, off=(.6, .9))
    if browtop:
        for sd in (-1, 1): brow(c, F, sd, st['brow'], e.get('bi', 0.), e.get('bo', 0.), e.get('bk', 0.), st['browcol'], None, lw)
    out['front'] = lit(c.layer(), st['d'], st['rim'], lw * .5, max(3, s * .030), 1., .5); out['F'] = F; out['st'] = st; out['lw'] = lw; out['fr'] = fr; return out


def head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='calm', light='day', f=0, **kw):
    d = draw(cx, cy, s, yaw, pitch, roll, expr, light, f, **kw); return stack(*[d[k] for k in ('back', 'body', 'face', 'front') if k in d])


def g1(f):
    """crew ring: head at (1600,720), crown toward (960,500); worry -> laugh at f20"""
    lf = f >= 20; hold = f // 2 * 2; b = keys(f, [(0, 0.), (14, 0.), (18, -10.), (20, 16., out3), (22, 26.), (26, 12.), (30, 18.), (40, 14.)]) + (4. * math.sin(f * .9) if 22 <= f < 44 else 1.5 * math.sin(f * .25)); u = np.array([-.945, -.325]); p = np.array([1606, 722]) + u * b; roll = -1.24 + (.05 * math.sin(f * .8) * math.exp(-(f - 20) / 14.) if lf else 0)
    ex = {} if lf else dict(gaze=(-.35, .25 + .1 * math.sin(f * .2))); ex.update(dict(o=.0) if f in (18, 19) else {})
    def body(c, F, st): bust(c, p[0], p[1], 168, roll, st, 'tee', .25, F.s * .0215, 1, length=4.2, breath=math.sin(f * .3))
    return head(p[0], p[1], 168, -.12, .10, roll, 'laugh' if lf else 'worry', 'g1', f, wind=(.35, .35), ex=ex, body=body, sway=(6, 10), length=.9)
