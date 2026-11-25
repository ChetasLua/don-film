"""Nico Robin.  hair='long' (swept back, glasses on the head: c6, g1) or 'short' (straight fringe, collar length: d4).  head(...), draw(...), g1(f)."""
from .lkit import *

PROPS = dict(a=[(-1.0, .100), (-.965, .165), (-.89, .275), (-.78, .380), (-.62, .480), (-.42, .588), (-.20, .668), (.06, .680), (.35, .680), (.6, .650), (.8, .530), (.93, .335), (1.0, .05)], nose=(.072, -.04, -.420, -.478, .168), cheek=.046)
EYE = dict(up=[(-1, -.36), (-.80, .16), (-.40, .64), (.08, .82), (.56, .76), (.92, .52), (1.13, .30)], lo=[(-1, -.36), (-.82, -.54), (-.46, -.74), (0, -.80), (.48, -.64), (.88, -.20), (1.13, .30)], c=(.338, -.010), w=.212, h=.128, iris=.46, irisv=.98, lid=.064, hood=.40, lash=[(.97, .40, .10), (.90, .38, .38), (.80, .26, .60), (.68, .12, .62), (.54, .02, .48)], lash2=[(.60, .02, -.30), (.76, .10, -.28), (.88, .16, -.18)], col=('#0c2348', '#2668b4', '#7fd0f4'))
BROW = {'in': (.125, .232), 'mid': (.38, .300), 'out': (.640, .200), 't': .036}
LIGHT = dict(
    day=dict(skin='#f4cba9', shade='#c58c74', hair=dict(base='#262434', shade='#14121d', deep='#0c0a12', deep2='#08070c', light='#6d7fb4', line='#07060b', contour='#0b0a10'), rim='#fff1d0', d=(.74, .67)),
    violet=dict(skin='#f1c6ab', shade='#a9748a', hair=dict(base='#282340', shade='#161228', deep='#0e0b1b', deep2='#090713', light='#8f86d8', line='#08060f', contour='#0c0a14'), rim='#ffd9f2', d=(.78, .62)),
    fire=dict(skin='#d8a17b', shade='#8c5a4c', hair=dict(base='#232130', shade='#110f18', deep='#0a090f', deep2='#07060a', light='#7c89ac', line='#07060a', contour='#0a090e'), rim='#ffe6b8', d=(.80, .60)),
    g1=dict(skin='#eec3a2', shade='#b27f6c', hair=dict(base='#282536', shade='#15131f', deep='#0d0b13', deep2='#08070c', light='#7b8cc2', line='#08070c', contour='#0b0a10'), rim='#fff6d8', d=(-.08, 1.)))


def style(light):
    p = dict(LIGHT[light]) if isinstance(light, str) else dict(light)
    p.update(bridge=True, cheekbone=.34, eye=EYE, brow=BROW, browcol='#15121a', mouth='#4d0e1e', line2='#84463a', blush='#c9574a', lip='#b8605a'); p.setdefault('cloth', dict(base='#4b3a86', shade='#2a2052', light='#8f7cd0', line='#120c26', zip='#e7d7a0')); return p


BLACK = dict(base='#201f29', shade='#0f0e15', light='#6a7088', line='#050508', zip='#c9c4bb')
HL = [(-.70, -.12), (-.655, .20), (-.565, .42), (-.36, .575), (-.12, .610), (0, .565), (.12, .610), (.36, .575), (.565, .42), (.655, .20), (.70, -.12)]


def fp(h, P): return np.array([(x, y, float(h.z(x, y)) + d) for x, y, d in P])


def fringe(F, t, sway, amp, hair):
    h = F.h; s = F.s; L = []; sh = h.yaw * .20
    def a(P, w, ph, sk=1., **o): q = dict(inner=3, root=.75); q.update(o); P = [(x + sh * (abs(x) > .66), y, d) for x, y, d in P]; L.append(lock3(F, fp(h, P), w * s, t, np.asarray(sway) * sk, amp * sk, ph, **q))
    if hair == 'long':
        a([(-.42, .86, .06), (-.72, .52, .10), (-.80, .08, .06), (-.80, -.42, .04), (-.80, -.95, .10), (-.74, -1.45, .16)], .30, .1, 1.4, hi=[(.14, .34, .3), (.46, .62, .2)], k=2.4)
        a([(.42, .86, .06), (.72, .52, .10), (.80, .08, .06), (.80, -.42, .04), (.78, -.95, .10), (.72, -1.45, .14)], .28, .5, 1.4, hi=[(.14, .34, .3), (.46, .62, .2)], k=2.4)
        a([(-.30, .70, .06), (-.52, .56, .13), (-.645, .28, .12), (-.66, -.05, .10), (-.62, -.40, .10)], .075, .3, 1.3, inner=0, k=2.)
        a([(-.10, .62, .05), (-.30, .52, .12), (-.46, .34, .13), (-.53, .14, .11)], .05, .7, 1.3, inner=0, k=2.)
        return L
    a([(-.40, .90, .02), (-.70, .60, .10), (-.80, .12, .08), (-.78, -.40, .08), (-.76, -.80, .10), (-.74, -1.16, .12)], .30, .1, 1.5, hi=[(.16, .36, .3)], root=.8, k=5.)
    a([(.40, .90, .02), (.70, .60, .10), (.80, .12, .08), (.78, -.40, .08), (.76, -.80, .10), (.73, -1.10, .12)], .28, .5, 1.5, hi=[(.16, .36, .3)], root=.8, k=5.)
    for j, (x0, x1, tip, w) in enumerate([(-.34, -.62, .19, .25), (.36, .64, .21, .24), (-.16, -.36, .22, .24), (.20, .40, .24, .23), (.02, .02, .25, .23)]):
        a([(x0 * .6, .95, .05), (x0 * 1.05, .84, .13), (lerp(x0, x1, .75), .66, .15), (x1, .45, .13), (x1 * 1.03, tip, .11)], w, j * .31, .30, hi=[(.30, .52, .1)], root=.8, k=2.6, belly=.4, inner=2)
    for j in range(13):
        r = rs(j, 41); x1 = -.62 + 1.24 * (j + .5) / 13 + r.normal(0, .02); x0 = x1 * .55; tip = .155 + .12 * r.random() + .05 * abs(x1); w = r.uniform(.085, .15); a([(x0 * .6, .96, .06), (x0 * 1.1, .86, .145), (lerp(x0, x1, .75), .68, .165), (x1, .43, .15), (x1 * 1.04 + r.normal(0, .025), tip, .125)], w, r.random(), .34 + .16 * r.random(), hi=[(.32, .50, .05)] if j % 2 else None, root=.7, k=1.9, belly=.45, inner=1)
    return L


def back_hair(F, t, wa, ws, hair, lagfn=None, seed=0, length=1.):
    h = F.h; s = F.s; L = []; g = math.pi / 2 + h.roll; lg = hair == 'long'
    R = [(-.60, .30, -.35), (-.64, -.05, -.25), (-.45, -.18, -.55), (-.18, -.32, -.65), (.12, -.32, -.65), (.42, -.20, -.55), (.62, .0, -.28), (.57, .32, -.35), (-.30, .10, -.8), (.28, .10, -.8)]
    for j, q in enumerate(R):
        r = rs(seed, j, 4); root = F.X3([q])[0]; ln = s * (r.uniform(2.6, 3.3) if lg else r.uniform(1.45, 1.75)) * length; a1 = lerp(g + r.normal(0, .06), wa + r.normal(0, .18), ws)
        P = strand(root, ln, g - .25 * q[0] * (1 - .6 * ws), a1, t, amp=.12 + .26 * ws, waves=.9, speed=.6 + .9 * ws, ph=j * .21 + r.random() * .2, curl=r.normal(0, .25) + ws * r.normal(0, .5), lagfn=lagfn, n=34); L.append(Lock(P, s * r.uniform(.40, .55), root=.8, k=3. if lg else 3.6, belly=.4, n=40, inner=5, hi=[(.22, .42, .3), (.55, .68, .15)], sh=.5, tone=1 if j >= 8 else 0))
    for j in range(12):
        r = rs(seed, j, 79); q = R[j % len(R)]; root = F.X3([q])[0]; P = strand(root, s * (r.uniform(2.4, 3.6) if lg else r.uniform(1.5, 2.2)) * length, g, lerp(g, wa + r.normal(0, .32), ws), t, amp=.16 + .34 * ws, waves=1.2, speed=.8 + .9 * ws, ph=r.random(), curl=r.normal(0, .8), lagfn=lagfn, n=34); L.append(Lock(P, s * r.uniform(.05, .13), root=.6, k=2.2, n=36, inner=0, hi=[(.2, .5, .1)]))
    return L


def glasses(cel, F, lw):
    h = F.h; a = np.linspace(0, TAU, 40)
    for x in (-.31, .31):
        P = h.P3(np.array([scalp3(h, math.atan2(x + np.cos(q) * .235, .50 - np.sin(q) * .02) * 1.0, .835 + np.sin(q) * .105, .19) for q in a])); cel.fill(P, '#c9572b'); cel.clip(P); c = P.mean(0); cel.fill(P + [0, F.s * .075], '#7c2a1c'); cel.fill(np.array([c + [-F.s * .20, -F.s * .02], c + [-F.s * .10, -F.s * .14], c + [-F.s * .02, -F.s * .14], c + [-F.s * .14, F.s * .06]]), '#ffd9a6', .9); cel.unclip(); stroke(cel, P, lw * 1.5, '#2a1a16')
    cel.line(h.P3(scalp3(h, np.linspace(-.17, .17, 8), np.full(8, .84), .21)), lw * 1.6, '#2a1a16', taper=(0, 0))
    for sg in (-1, 1): cel.line(h.P3(scalp3(h, sg * np.linspace(.80, 1.75, 14), np.linspace(.82, .30, 14), .16)), lw * 1.3, '#2a1a16', taper=(0, .2))


def draw(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, hair='long', wind=(math.pi / 2, 0.), ex=None, body=None, posfn=None, step=2, st=None, sway=None, length=1., seed=23, under=None, browtop=None, upsway=0., post=None):
    st = st or style(light); pal = st['hair']; h = Head(**PROPS).pose(cx, cy, s, yaw, pitch, roll); e = dict(EXPR[expr]) if expr else {}; e.update(ex or {}); F = Face(h, e.get('jaw', 0.) + .07); lw = float(np.clip(s * .0215, 2.0, 6.6)); t = (f // step * step) / 24.; ld = -np.asarray(st['d']); sgl = -1 if ld[0] < 0 else 1
    wa, ws = wind; lf = lag(posfn, f, 5.) if posfn else None; sw = np.array([math.cos(wa), math.sin(wa) * .5]) * s * (.34 * ws) if sway is None else np.asarray(sway, float)
    sw = sw * (1 + .22 * math.sin(t * 7.3)) + [0, -upsway * s]; lg = (lf(np.array([.6]))[0] * .6 if lf else 0); out = {}
    c = Cel(); bl = back_hair(F, t, wa, ws, hair, lf, seed, length); draw_mass(c, bl, pal, lw * .62, lw * .40, seed, ld, contour=pal['contour']); out['back'] = lit(c.layer(), st['d'], st['rim'], lw * .4, max(3, s * .026), 1., .5)
    if body is not None: c = Cel(); body(c, F, st); out['body'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .030), .95)
    c = Cel(); C, D = face_base(c, F, st, lw, sgl, seed); fr = fringe(F, t, sw + lg, s * (.02 + .09 * ws), hair)
    shadow_of(c, fr, C, (-ld[0] * s * .05, s * .075), st['shade'])
    if under: c.clip(C); under(c, F, C); c.unclip()
    c.clip(C); feats(c, F, st, e, lw, f, sgl, seed, (lambda: post(c, F, C)) if post else None); c.unclip(); contour_line(c, D, lw, seed); cheek(c, F, st, -sgl, .6, 4, lw, '#84463a', y=-.10, x=.46)
    out['face'] = lit(c.layer(), st['d'], st['rim'], lw, max(3, s * .030), 1., .45)
    c = Cel(); cap(c, F, pal, HL if hair == 'long' else [(-.72, -.15), (-.68, .24), (-.54, .50), (-.2, .62), (.2, .62), (.54, .50), (.68, .24), (.72, -.15)], 1.10, lw * .9, seed, ld, part=(0, .99), nlines=15, hiy=(.66, .78) if hair == 'long' else (.74, .84))
    if hair == 'long': glasses(c, F, lw)
    draw_mass(c, fr, pal, lw * .42, lw * .40, seed + 3, ld, off=(.6, .9), contour=pal['contour'])
    if browtop:
        for sd in (-1, 1): brow(c, F, sd, st['brow'], e.get('bi', 0), e.get('bo', 0), e.get('bk', 0), '#0a0810', browtop, lw)
    out['front'] = lit(c.layer(), st['d'], st['rim'], lw * .4, max(3, s * .026), .95, .5); out['F'] = F; out['st'] = st; out['lw'] = lw; out['fr'] = fr; return out


def head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='calm', light='day', f=0, **kw):
    d = draw(cx, cy, s, yaw, pitch, roll, expr, light, f, **kw); return stack(*[d[k] for k in ('back', 'body', 'face', 'front') if k in d])


def g1(f):
    """crew ring: upside down at (930,120); worry -> a warm laugh at f20"""
    b = keys(f, [(0, 0.), (15, 0.), (19, -8.), (22, 14., out3), (26, 6.), (40, 8.)]) + (2.5 * math.sin(f * .8) if 23 <= f < 42 else 1.2 * math.sin(f * .22)); p = np.array([930, 152 + b]); roll = 3.06 + (.03 * math.sin(f * .6) * math.exp(-(f - 22) / 16.) if f >= 22 else 0)
    ex = dict(gaze=(.05, -.3), M='calm', bi=.09, bo=-.03, o=1.04, bk=.2) if f < 18 else dict(o=.25, M='smile') if f < 21 else {}
    def body(c, F, st): bust(c, p[0], p[1], 140, roll, st, 'jacket', 0., F.s * .0215, -1, length=4., breath=math.sin(f * .3))
    return head(p[0], p[1], 140, .06, .10, roll, 'worry' if f < 21 else 'chuckle', 'g1', f, wind=(-math.pi / 2 + .5, .55), ex=ex, body=body, sway=(8, 14), length=.8)
