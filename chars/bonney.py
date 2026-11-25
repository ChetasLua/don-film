"""Jewelry Bonney, chest-up, coat closed, a glutton gag.  head(...), draw(...); fury 0..1 lifts the hair like flames."""
from .lkit import *
from .nami import fp

PROPS = dict(a=[(-1.0, .115), (-.965, .185), (-.89, .31), (-.78, .435), (-.62, .565), (-.42, .675), (-.18, .745), (.08, .76), (.35, .755), (.6, .715), (.8, .575), (.93, .345), (1.0, .05)], nose=(.068, -.15, -.345, -.40, .115))
EYE = dict(ALMOND, up=[(-1, -.28), (-.84, .30), (-.44, .80), (.06, .98), (.55, .88), (.91, .56), (1.08, .24)], lo=[(-1, -.28), (-.86, -.50), (-.50, -.74), (-.04, -.84), (.44, -.70), (.85, -.28), (1.08, .24)], c=(.375, -.005), w=.215, h=.180, iris=.56, irisv=.80, lid=.054, lash=[(.95, .28, .16), (.87, .24, .34), (.76, .12, .40)], lash2=[(.70, .08, -.20), (.84, .14, -.12)], col=('#2c1248', '#7a3fb2', '#cf9af6'))
BROW = {'in': (.14, .300), 'mid': (.38, .385), 'out': (.64, .310), 't': .040}
PAL = dict(skin='#fbd3b4', shade='#dc9a86', hair=dict(base='#f672a8', shade='#cc4585', deep='#98296a', deep2='#741c54', light='#ffb1d0', line='#4d0f33', contour='#2a0a20'), rim='#fff2f6', d=(-.78, .62))


def style(light='card'):
    p = dict(PAL) if isinstance(light, str) else dict(light); p.update(eye=EYE, brow=BROW, browcol='#8a2458', mouth='#661026', line2='#a5523f', blush='#e0605a', lip=None); p.setdefault('cloth', dict(base='#e3a22b', shade='#a96a1a', light='#ffd36e', line='#3a1e0a', zip='#ffffff')); return p


def fur(cel, P, r, lw, seed=0, pal=('#4a2818', '#2a140c', '#7d4e30')):
    P = resample(spl(np.asarray(P, float), n=10), r * .42); rr = rs(seed, 3); B = []
    for i, p in enumerate(P):
        a = np.linspace(0, TAU, 33)[:-1]; k = 7 + i % 3; B.append(p + np.c_[np.cos(a), np.sin(a)] * (r * (.92 + .10 * np.abs(np.sin(a * k * .5 + i * 1.7)) + .05 * rr.normal()))[:, None])
    for b in B: stroke(cel, b, lw * 2.1)
    for b in B: cel.fill(b, pal[0])
    for i, (p, b) in enumerate(zip(P, B)): cel.shade(b + [-r * .30, -r * .42], pal[0], b); cel.clip(b); cel.fill(p + (b - p) * .98 + [r * .16, r * .62], pal[1], .85); cel.unclip()
    for i, p in enumerate(P):
        for j in range(4):
            a = rr.uniform(0, TAU); q = p + np.array([math.cos(a), math.sin(a)]) * r * rr.uniform(.2, .75); L = spl([q + [-r * .16, r * .05], q + [0, -r * .10], q + [r * .16, r * .06]], n=5); cel.line(L, lw * .55, pal[2] if q[1] < p[1] else '#140a06', taper=(.3, .5), seed=i * 9 + j)
        if i % 2 == 0: soft(cel, p[0] - r * .35, p[1] - r * .45, r * .4, pal[2], .55)
    return B


def cap_(cel, F, st, lw, pop=0., roll=0.):
    h = F.h; s = F.s; y = h.yaw; Q = lambda P: loc(h.cx + math.sin(y) * s * .10, h.cy - pop * s, s, h.roll + roll, P); g, gd, gl, dk = '#3fa07c', '#257058', '#8fe0b8', '#123a30'; b = y * .9
    C = Q(spl([(-.80, -.50), (-.99, -.70), (-1.00, -1.02), (-.70, -1.30), (-.20, -1.43), (.36, -1.40), (.82, -1.22), (1.03, -.92), (.96, -.64), (.80, -.50), (.40, -.62), (0, -.66), (-.40, -.62)], True, n=8)); cel.fill(C, g); cel.clip(C)
    cel.fill(Q(spl([(1.2, -1.3), (.62, -1.16), (.50, -.90), (.10, -.80), (-.30, -.84), (-.62, -.74), (-1.2, -.6), (-1.2, -.2), (1.2, -.2)], True, n=6)), gd); cel.fill(Q(spl([(-.78, -.98), (-.62, -1.20), (-.30, -1.32), (-.42, -1.18), (-.64, -.98)], True, n=6)), gl, .9)
    for e in ((-.52, -.62), (b * .4, -.64), (.56, -.60)): cel.line(Q(spl([(.06 + b * .2, -1.40), (e[0] * 1.25 + .03, -1.06), e], n=8)), lw * .62, dk, taper=(.1, .2))
    for k, (x, yy) in enumerate(((-.30, -.80), (.34, -.80), (.70, -.84))): cel.line(Q(spl([(x - .10, yy), (x, yy + .04), (x + .13, yy - .05)], n=5)), lw * .5, dk, taper=(.2, .5))
    cel.unclip(); stroke(cel, C, lw * 1.0)
    bt = Q(spl([(b * .2 - .06, -1.43), (b * .2 + .06, -1.50), (b * .2 + .18, -1.43), (b * .2 + .06, -1.37)], True, n=5)); cel.fill(bt, g); stroke(cel, bt, lw * .8)
    V = Q(spl([(-.70 + b, -.56), (-.55 + b, -.40), (-.10 + b * 1.3, -.335), (.40 + b * 1.2, -.36), (.72 + b, -.46), (.74 + b, -.58), (.30 + b, -.665), (-.25 + b, -.67)], True, n=7)); cel.fill(V, '#2f8668'); cel.clip(V); cel.fill(Q(spl([(-.8 + b, -.30), (-.6 + b, -.46), (0 + b, -.41), (.5 + b, -.43), (.9 + b, -.5), (.9 + b, -.2)], True, n=6)), gd); cel.line(Q(spl([(-.60 + b, -.485), (-.1 + b, -.43), (.4 + b, -.44), (.68 + b, -.52)], n=8)), lw * .45, gl, taper=(.1, .3)); cel.unclip(); stroke(cel, V, lw * 1.0)
    return C


def back_hair(F, t, wa, ws, fury=0., lagfn=None, seed=5):
    h = F.h; s = F.s; L = []; n = 17
    for j in range(n):
        r = rs(seed, j); q = [(-.60, .45, -.3), (-.66, .05, -.3), (-.50, -.2, -.5), (-.10, -.3, -.6), (.3, -.25, -.6), (.6, .0, -.3), (-.4, .3, -.7), (.5, .4, -.4)][j % 8]; root = F.X3([q])[0]; spread = (j / (n - 1.) - .5); a0 = math.pi * (.92 - .30 * q[0]); a1 = wa + spread * 1.25
        if fury: a0 = lerp(a0, -math.pi / 2 + (q[0] * .9) + math.pi * 2 * 0, fury * .0); a1 = lerp(a1, math.pi * 1.28 + spread * 1.35, fury)
        ln = s * r.uniform(3.2, 5.4) * (1 - .25 * fury); P = strand(root, ln, a0, a1, t, amp=.34 * (1 + .25 * fury), waves=r.uniform(1.2, 1.7), speed=.55 + 1.1 * fury, ph=j * .19 + r.random() * .2, curl=r.choice([-1, 1]) * r.uniform(.7, 1.9) * (1 - .6 * fury), lagfn=lagfn, n=44); L.append(Lock(P, s * r.uniform(.52, .86), root=.6, k=2.2, belly=.42, n=54, inner=5, hi=[(.14, .34, .34), (.45, .62, .2), (.62, .78, -.2)], sh=.58, tone=1 if j % 8 in (3, 6) else 0))
    for j in range(16):
        r = rs(seed, j, 31); root = F.X3([(-.5 + r.uniform(-.1, .4), r.uniform(-.2, .4), -.4)])[0]; P = strand(root, s * r.uniform(3.2, 6.0), math.pi * .92, lerp(wa + r.normal(0, .5), math.pi * 1.28 + r.normal(0, .5), fury), t, amp=.42, waves=1.6, speed=.7 + fury, ph=r.random(), curl=r.choice([-1, 1]) * r.uniform(.8, 2.4), lagfn=lagfn, n=44); L.append(Lock(P, s * r.uniform(.07, .19), root=.5, k=2., n=46, inner=1, hi=[(.2, .5, .1)]))
    return L


def fringe(F, t, sway, amp, fury=0.):
    h = F.h; s = F.s; L = []; sh = h.yaw * .22; up = fury
    def a(P, w, ph, sk=1., **o): q = dict(inner=2, root=.8, r0=.2); q.update(o); P = [(x + sh * float(sm((abs(x) - .45) / .3)), y, d) for x, y, d in P]; L.append(lock3(F, fp(h, P), w * s, t, np.asarray(sway) * sk, amp * sk, ph, **q))
    a([(-.40, .72, .10), (-.70, .44, .14), (-.80, .05, .10), (-.80, -.40, .10), (-.86 - .3 * up, -.82 + .5 * up, .10), (-.96 - .5 * up, -1.15 + 1.0 * up, .06)], .36, .1, 1.6, hi=[(.16, .38, .3)])
    a([(.44, .72, .10), (.72, .42, .14), (.82, .04, .10), (.80, -.42, .10), (.84 + .2 * up, -.80 + .5 * up, .10), (.74 + .5 * up, -1.12 + 1.0 * up, .12)], .33, .5, 1.5, hi=[(.16, .38, .3)])
    a([(-.22, .66, .12), (-.46, .50, .17), (-.64, .24, .15), (-.70, -.02, .12), (-.62, -.22, .11)], .27, .3, 1., hi=[(.28, .56, .2)])
    a([(.26, .66, .12), (.46, .48, .16), (.60, .28, .15), (.72, .08, .11)], .24, .7, 1., hi=[(.28, .56, .2)])
    a([(-.04, .66, .12), (-.20, .52, .17), (-.34, .32, .15), (-.44, .16, .11)], .22, .9, 1., hi=[(.3, .6, .1)])
    a([(.10, .66, .12), (.12, .50, .17), (.06, .34, .15), (-.05, .20, .12)], .15, .4, 1., inner=1)
    return L


def draw(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='card', f=0, wind=(math.pi, 1.), ex=None, body=None, posfn=None, step=2, fury=0., pop=0., caproll=0., seed=5, st=None):
    st = st or style(light); pal = st['hair']; h = Head(**PROPS).pose(cx, cy, s, yaw, pitch, roll); e = dict(EXPR[expr]) if expr else {}; e.update(ex or {}); F = Face(h, e.get('jaw', 0.)); lw = float(np.clip(s * .0215, 2.0, 6.6)); t = (f // step * step) / 24.; ld = -np.asarray(st['d']); sgl = -1 if ld[0] < 0 else 1; lf = lag(posfn, f, 5.) if posfn else None; lg = (lf(np.array([.6]))[0] * .6 if lf else 0); out = {}
    c = Cel(); draw_mass(c, back_hair(F, t, wind[0], wind[1], fury, lf, seed), pal, lw * .70, lw * .44, seed, ld, contour=pal['contour']); out['back'] = lit(c.layer(), st['d'], st['rim'], lw * .5, max(3, s * .022), .8, .3)
    if body is not None: c = Cel(); body(c, F, st); out['body'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .03), .95)
    c = Cel(); C, D = face_base(c, F, st, lw, sgl, seed); fr = fringe(F, t, np.array([-.10, .02]) * s + lg, s * .05, fury)
    c.clip(C); c.fill(loc(h.cx, h.cy, s, roll, spl([(-1.2, -.6), (-.6, -.30), (0, -.22), (.6, -.28), (1.2, -.5), (1.2, -1.5), (-1.2, -1.5)], True, n=6)) + [0, s * .02], st['shade']); c.unclip(); shadow_of(c, fr, C, (s * .05, s * .07), st['shade'])
    c.clip(C); feats(c, F, st, e, lw, f, sgl, seed)
    if not e.get('nopierce'):
        p = F.X([(.40, -.255)], .02)[0]; a = np.linspace(0, TAU, 20); r = s * .026
        for dx in (-.055, 0, .055): q = F.X([(.40 + dx, -.262 - .10 * abs(dx))], .02)[0]; c.fill(np.c_[q[0] + np.cos(a) * r, q[1] + np.sin(a) * r], '#f7cf4f'); stroke(c, np.c_[q[0] + np.cos(a) * r, q[1] + np.sin(a) * r], lw * .42); c.fill(np.c_[q[0] - r * .3 + np.cos(a) * r * .3, q[1] - r * .3 + np.sin(a) * r * .3], '#ffffff')
    c.unclip(); contour_line(c, D, lw, seed); out['face'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .032), 1., .45)
    c = Cel(); cap(c, F, pal, [(-.74, -.10), (-.70, .26), (-.56, .50), (-.2, .60), (.2, .60), (.5, .50), (.68, .28), (.74, -.06)], 1.08, lw * .9, seed, ld, hiy=None); draw_mass(c, fr, pal, lw * .42, lw * .42, seed + 3, ld, off=(.6, .9), contour=pal['contour']); out['front'] = lit(c.layer(), st['d'], st['rim'], lw * .5, max(3, s * .026), .9, .4)
    c = Cel(); cap_(c, F, st, lw, pop, caproll); out['cap'] = lit(c.layer(), st['d'], '#eafff4', lw, max(3, s * .032), .95, .45); out['F'] = F; out['st'] = st; out['lw'] = lw; return out


def head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='calm', light='card', f=0, **kw):
    d = draw(cx, cy, s, yaw, pitch, roll, expr, light, f, **kw); return stack(*[d[k] for k in ('back', 'body', 'face', 'front', 'cap') if k in d])


def coat(cel, cx, cy, s, roll, st, lw, f=0, sh=0.):
    Q = bust(cel, cx, cy, s, roll, st, 'coat', .05, lw, -1, sh=sh, f=f); c = st['cloth']
    for x0, y0, r in ((.34, 2.35, .07), (.34, 2.85, .07)): a = np.linspace(0, TAU, 24); P = Q(np.c_[x0 + np.cos(a) * r, y0 + np.sin(a) * r], .3); cel.fill(P, '#f6d36a'); stroke(cel, P, lw * .7)
    cel.line(Q(spl([(.10, 1.75), (.16, 2.3), (.12, 3.2)], n=8), .3), lw * .9, INK, taper=(.05, .05))
    fur(cel, Q([(.66, .98), (.98, 1.28), (.72, 1.66), (.30, 1.86), (.10, 2.05), (.06, 2.6), (.04, 3.2)], .25), s * .215, lw, 3); fur(cel, Q([(-.66, .98), (-.98, 1.28), (-.74, 1.64), (-.30, 1.86), (.04, 2.04)], .25), s * .225, lw, 4)
    return Q
