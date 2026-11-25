"""Sanji (fan art).  head(cx,cy,s,yaw,pitch,roll,expr,light,f,**kw) -> layer; draw(cel, ...) -> Head; g1(f).   expr: calm glare laugh worry smirk.
After the two years: hair over the RIGHT eye, the LEFT brow curls at its inner end.  kw: cig, look, lid, mouth_, body, wind, key, smoke"""
from .swkit import *
from .zoro import OHead, ring_pose, jaw_cast

PAL = dict(base='#f9cba0', shade='#cb8460', deep='#85432f', light='#ffeccc', line=INK, term='#ec8a55', bounce='#e0a680')
HAIR = dict(deep='#7d4710', shade='#d5922a', base='#fcd24c', light='#ffee96', hot='#fffdea', line='#8a5414')
SUIT = dict(base='#23222c', shade='#12111a', deep='#0a090f', light='#454457', shirt='#f0bf3a', shirtd='#b07a1c', btn='#e7c25a')


def make():
    return OHead(a=[(-1.0, .090), (-.965, .165), (-.87, .290), (-.72, .415), (-.56, .525), (-.36, .610), (-.12, .660), (.12, .69), (.35, .705), (.6, .685), (.8, .55), (.93, .34), (1.0, .05)], nose=(.10, -.06, -.41, -.47, .175), cheek=.03, socket=.045)


# swept hair guides (front coordinates + z), parting on his left: root .. tip
FR = [[(.30, .90, .60), (.23, .99, .80), (.06, .70, .98), (-.08, .34, 1.03), (-.165, .06, 1.02), (-.175, -.20, .98)],
      [(.30, .94, .50), (.12, 1.04, .80), (-.14, .72, .97), (-.28, .34, 1.00), (-.36, .02, .97), (-.375, -.34, .90)],
      [(.29, .98, .40), (-.02, 1.08, .70), (-.32, .75, .87), (-.49, .34, .87), (-.56, -.02, .81), (-.565, -.44, .70)],
      [(.27, 1.02, .28), (-.12, 1.11, .56), (-.49, .79, .70), (-.67, .36, .66), (-.73, -.04, .57), (-.715, -.50, .45)],
      [(.24, 1.05, .14), (-.22, 1.13, .36), (-.60, .80, .46), (-.765, .36, .40), (-.80, -.06, .30), (-.755, -.50, .20)],
      [(.20, 1.07, -.02), (-.29, 1.12, .10), (-.68, .78, .14), (-.83, .30, .06), (-.845, -.14, -.04), (-.775, -.55, -.12)],
      [(.14, 1.06, -.2), (-.32, 1.07, -.16), (-.70, .72, -.20), (-.83, .22, -.26), (-.82, -.22, -.34), (-.71, -.68, -.42)]]
LS = [[(.33, .90, .60), (.43, .88, .74), (.56, .58, .76), (.66, .28, .68), (.715, .04, .60), (.70, -.16, .56)],
      [(.33, .95, .48), (.49, .94, .60), (.68, .62, .58), (.79, .26, .46), (.83, -.04, .36), (.80, -.32, .30)],
      [(.32, 1.0, .34), (.52, 1.02, .42), (.77, .68, .36), (.89, .24, .22), (.905, -.12, .12), (.86, -.46, .06)],
      [(.29, 1.04, .14), (.50, 1.06, .16), (.80, .70, .10), (.92, .18, -.06), (.92, -.22, -.16), (.84, -.60, -.24)],
      [(.24, 1.06, -.1), (.46, 1.05, -.12), (.79, .66, -.20), (.89, .12, -.34), (.87, -.30, -.42), (.74, -.74, -.48)]]


def _interp(G, u):
    G = np.asarray(G, float); u = float(np.clip(u, 0, len(G) - 1.001)); i = int(u); t = u - i; return G[i] * (1 - t) + G[i + 1] * t


# (guide, width, length change, sideways S, tip flick) back to front: big shapes first, a few thin escapes last
RSET = [(6.0, .36, .0, .02, -.02), (5.2, .36, -.06, -.02, -.04), (5.65, .15, .12, .03, -.06), (4.4, .38, .05, .03, -.03), (3.55, .36, -.10, -.03, .0), (4.0, .15, .13, .03, -.05), (2.75, .38, .06, .035, .02), (1.95, .36, -.06, -.03, .04), (2.4, .13, .16, .03, .03), (1.2, .36, .09, .035, .05), (.50, .34, -.02, -.02, .06), (.85, .10, -.22, .03, .05), (.05, .22, -.14, .03, .05)]
LSET = [(4.0, .36, .0, .0, .04), (3.2, .34, -.06, .02, .06), (2.4, .34, .06, .02, .05), (2.8, .12, .14, -.02, .08), (1.6, .32, -.05, -.02, .06), (.9, .26, .08, .02, .05), (1.2, .10, -.2, .0, .07), (.25, .20, -.04, -.02, .05)]


def _build(G, S, seed):
    out = []
    for j, (u, w, dl, sx, fl) in enumerate(S):
        P = _interp(G, u).copy(); P[-1] += (P[-1] - P[-2]) * dl * 2.0; P[2] += A_(sx, 0, .012); P[4] -= A_(sx, 0, 0); P[-1] += A_(fl, abs(fl) * .25, 0); out.append((P, w, w / .3, seed * 100 + j * 7))
    return out


SPROF = lambda t: np.clip(.42 + 1.1 * t, 0, 1) * np.clip((1 - t) / .52, 0, 1) ** .86 * (1 + .06 * np.sin(t * 8))
_TB = {}
def tables():
    if not _TB: _TB['R'] = _build(FR, RSET, 4); _TB['L'] = _build(LS, LSET, 9)
    return _TB


BROW = dict(calm=[(.66, .120), (.56, .200), (.42, .230), (.28, .200), (.205, .165)], glare=[(.66, .165), (.56, .225), (.42, .215), (.29, .150), (.205, .100)], worry=[(.64, .06), (.54, .155), (.41, .215), (.29, .235), (.205, .215)], laugh=[(.66, .135), (.55, .235), (.42, .270), (.29, .245), (.205, .205)])
BROW['smirk'] = BROW['calm']
EYE = dict(glare=([(.155, -.030), (.25, .040), (.39, .072), (.565, .058)], [(.55, .004), (.42, -.066), (.28, -.072), (.165, -.048)], (.335, .000)),
           calm=([(.155, -.018), (.25, .052), (.39, .078), (.56, .040)], [(.545, -.004), (.42, -.074), (.28, -.078), (.165, -.044)], (.345, -.004)),
           worry=([(.155, .0), (.25, .105), (.38, .145), (.55, .060)], [(.54, .0), (.42, -.09), (.28, -.096), (.165, -.045)], (.345, .020)))
EYE['smirk'] = EYE['calm']; EYE['laugh'] = EYE['calm']


def spiral(x, y, r, turns=1.55, a0=-2.2, n=44):
    t = np.linspace(0, 1, n); a = a0 - t * turns * 2 * math.pi; q = r * (1 - .80 * t); return np.c_[x + np.cos(a) * q, y + np.sin(a) * q]


def draw(cel, cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', key=-1, f=0, cig=True, look=(0, 0), lid=1., mouth_=None, body=True, broll=0., wind=0., seed=1, pal=None, hairpal=None, L=None, drop=0., hush=0., smoke=True, detail=True, ember=.6, cig_dir=(.12, -.10, .42)):
    pal = dict(PAL, **(pal or {})); hp = dict(HAIR, **(hairpal or {})); H_ = make(); H_.drop = drop; H_.pose(cx, cy, s, yaw, pitch, roll); p = Pen(cel, H_, s, pal['line'], seed); k = -key; c = cel; lw = p.w; near = 1 if yaw < -.02 else (-1 if yaw > .02 else 0)
    L = L or (key * .5, .62, .6); TB = tables(); ln = pal['line']
    def sway(P, u):
        P = P.copy(); t = np.linspace(0, 1, len(P))[:, None] ** 1.6; P += t * (A_(wind * .34, abs(wind) * .10, 0) + A_(.030 * math.sin(f * .5 + u * 5), .012 * math.cos(f * .43 + u * 7), 0)); return P
    if body: body_suit(c, H_, cx, cy, s, pal, k, broll, yaw, seed, p, f)
    polys = []
    Bk = H_.P3(np.array([(-.86, .55, -.5), (-.95, 0, -.45), (-.86, -.5, -.45), (-.62, -.86, -.5), (-.2, -.84, -.6), (.2, -.84, -.6), (.62, -.88, -.5), (.86, -.5, -.45), (.95, 0, -.45), (.86, .55, -.5), (.4, 1.05, -.4), (-.4, 1.05, -.4)])); Bk = spl(Bk, True, n=6)
    c.fill(Bk, hp['deep']); c.clip(Bk)
    for j in range(16): x = -.8 + 1.6 * j / 15; q = H_.P3(np.array([(x, -.1, -.5), (x * 1.02, -.5, -.55), (x * .95, -.9, -.5)])); c.line(spl(q, n=8), lw(2.0), '#5e3610', taper=(.4, .3), seed=j)
    c.unclip(); outline(c, Bk, lw(4.2), ln, 3)
    c.fill(H_.skull(), hp['deep'])
    F = H_.face(); c.fill(F, pal['base']); c.clip(F)
    side = [(.28, .84), (.33, .60), (.40, .42), (.50, .28), (.45, .15), (.575, .06), (.54, -.06), (.42, -.13), (.475, -.23), (.43, -.36), (.335, -.47), (.30, -.58), (.335, -.68), (.26, -.82), (.145, -.92), (.08, -1.03), (.6, -1.1), (1.0, -.8), (1.0, .85)]
    p.poly([((x - .018) * k, y - .004) for x, y in side], pal['term']); p.poly([(x * k, y) for x, y in side], pal['shade'])
    q = [(.045, .07), (.14, .09), (.25, .135), (.41, .185), (.59, .165), (.58, .075), (.40, .070), (.26, .035), (.15, -.03), (.07, -.02)]
    for sg in (1, -1): p.poly([(x * sg, y) for x, y in q], pal['shade'])
    p.poly([(-k * x, y) for x, y in [(.57, -.18), (.48, -.29), (.40, -.43), (.39, -.57), (.46, -.52), (.56, -.36)]], pal['shade'])
    p.poly([(-.10, -.69), (.12, -.688), (.06, -.742), (-.05, -.742)], pal['shade']); p.poly([(-k * x, y) for x, y in [(.28, -.14), (.46, -.13), (.49, -.20), (.38, -.26), (.27, -.21)]], pal['light'], .85)
    # shadow the curtain of hair throws on the cheek
    for P, w, big, sd in TB['R'][-30:]:
        S = strand(H_, sway(P, sd % 97 / 97.), w, prof=SPROF); q = S[0] + A_(math.cos(roll), math.sin(roll)) * s * .075 + A_(-math.sin(roll), math.cos(roll)) * s * .05; c.fill(q, pal['shade'])
    c.fill([F, F + A_(k * .030, .008) * s], '#fff4dc', .92, evenodd=True); c.unclip(); outline(c, F, lw(5.0), ln, seed, press=.42, breaks=2)
    nose(p, pal, k, near or k, tip=.175, w=.108)
    E = EYE.get(expr, EYE['calm'])
    if expr == 'laugh': p.line([(.16, -.015), (.27, .055), (.41, .060), (.56, -.02)], 6.4, taper=(.12, .16)); p.line([(.585, .015), (.64, .045)], 2)
    else:
        eye(p, 1, E[0], E[1], (E[2][0] + look[0], E[2][1] + look[1]), .050, shade='#b59d94', lw=8., lid=.06, open_=lid, crease=[(.18, .06), (.34, .122), (.55, .105)]); p.line([(.545, .035), (.60, .075)], 3., taper=(.1, .5)); p.line([(.50, -.082), (.38, -.125), (.26, -.122)], 1.8, taper=(.4, .5))
    br = BROW.get(expr, BROW['calm']); w0 = lambda t: .008 + .026 * math.sin(math.pi * min(1, .06 + t * .86)) ** .7; ribbon_fill(p, br, w0, '#6f4310'); p.line(br, 2.4, taper=(.4, .05))
    ex, ey = br[-1]; sp = spiral(ex - .062, ey + .062, .088); p.line(np.vstack([[br[-2]], sp]), 4.6, '#5b350c', taper=(.04, .35), press=.2, n=3); p.line(np.vstack([[br[-2]], sp]), 2.2, taper=(.04, .35), n=3)
    if expr == 'glare': p.line([(-.02, .055), (-.045, .125)], 2.2); p.line([(.06, .03), (.085, .085)], 2.0)
    kind = mouth_[0] if mouth_ else dict(glare='flat', calm='flat', worry='worry', laugh='laugh', smirk='smirk').get(expr, 'flat'); prm = mouth_[1] if mouth_ else {}
    if kind == 'flat':
        p.line([(-.165, -.672), (-.04, -.655), (.09, -.655), (.185, -.672), (.205, -.695)], 3.8, taper=(.2, .25), wfun=lambda t: .75 + .45 * np.sin(np.pi * t)); p.line([(-.06, -.755), (.07, -.752)], 2., taper=(.4, .4))
    elif kind == 'smirk': p.line([(-.14, -.682), (-.02, -.672), (.10, -.655), (.20, -.615), (.235, -.585)], 3.8, taper=(.2, .25)); p.line([(.245, -.555), (.265, -.60)], 2.2); p.line([(-.04, -.755), (.09, -.745)], 2., taper=(.4, .4))
    elif kind == 'worry': p.line([(-.15, -.70), (-.04, -.672), (.07, -.672), (.17, -.70)], 3.6, taper=(.2, .25)); p.line([(-.06, -.765), (.07, -.765)], 2., taper=(.4, .4))
    elif kind == 'laugh':
        o = prm.get('o', .20); mouth_open(p, [(-.30, -.545), (-.16, -.58), (.0, -.592), (.16, -.58), (.31, -.54)], [(.31, -.54), (.25, -.65 - o * .55), (.0, -.69 - o), (-.24, -.65 - o * .55), (-.30, -.545)], tu=.085, tl=.0, tongue=[(-.15, -.70 - o), (-.13, -.64 - o * .6), (0, -.62 - o * .55), (.13, -.64 - o * .6), (.15, -.70 - o)], lw=4.2, tm=.7); p.line([(.315, -.505), (.345, -.55), (.325, -.595)], 2.4)
    elif kind == 'o': a = np.linspace(0, 6.283, 24); w = prm.get('w', .07); p.fill(np.c_[.05 + np.cos(a) * w, -.675 + np.sin(a) * w * .8][::3], '#43121a'); p.line(np.c_[.05 + np.cos(a) * w, -.675 + np.sin(a) * w * .8], 3.4, n=2)
    # moustache shadow, chin tuft, jaw stubble
    for j in range(9): t = j / 8.; p.line([(-.12 + .26 * t, -.575 - .012 * abs(t - .5)), (-.135 + .29 * t, -.625)], 1.3, '#7a4f1e', taper=(.1, .6), alpha=.85)
    if kind != 'laugh' or prm.get('o', .2) < .12:
        p.fill([(-.075, -.885), (.0, -.868), (.075, -.882), (.06, -.97), (.015, -1.03), (-.03, -1.03), (-.06, -.97)], '#9a6a1e')
    for j in range(11): t = j / 10.; x = -.095 + .19 * t; p.line([(x, -.865 - .02 * math.sin(t * 9)), (x * .9, -.95), (x * .62 + .01, -1.045 - .02 * math.sin(t * 13))], 1.9, '#6f4310', taper=(.15, .5), n=5)
    for sg in (1, -1):
        for j in range(9): t = j / 8.; y = -.52 - .40 * t; x = sg * (np.interp(y, H_.yy, H_.A) - .05 - .02 * (j % 2)); p.line([(x, y), (x - sg * .015, y - .05)], 1.2, '#7a4f1e', taper=(.1, .6), alpha=.75)
    if detail and s > 120:
        p.hatch((k * .41, -.14), (k * .335, -.46), 7, (-k * .10, -.015), 1.5, ln, seed=3); p.hatch((k * .31, -.60), (k * .22, -.84), 5, (-k * .08, .02), 1.4, ln, seed=4); p.hatch((k * .05, -.12), (k * .07, -.30), 4, (k * .04, -.012), 1.3, ln, seed=8); p.hatch((.25, -.105), (.45, -.11), 5, (.012, -.045), 1.2, ln, seed=7); p.hatch((-.07, -.557), (.07, -.557), 4, (0, -.026), 1.1, ln, seed=9)
    # ---- hair and ear
    def lay(tab, idx=None):
        for P, w, big, sd in tab:
            S = strand(H_, sway(P, sd % 97 / 97.), w, prof=SPROF); polys.append(S[0]); strand_draw(c, H_, S, hp, L, ln, 2.2, seed + sd, inner=2 if big > .6 else 1, band=(.22, .36))
    order = [('L', 1), ('R', -1)] if near != 1 else [('R', -1), ('L', 1)]
    def ear(sd):
        e = H_.ear(sd); c.fill(e['outer'], pal['base'] if sd != k else pal['shade']); c.clip(e['outer']); c.fill(e['inner'], pal['shade'] if sd != k else pal['deep']); c.fill(e['tragus'], pal['deep']); c.unclip(); outline(c, e['outer'], lw(3.6), ln, seed + 7); p.line(e['inner'], 2.0, raw=True, taper=(.1, .3))
    if near == 1: ear(1)
    for q in front_mass(H_, lambda th: np.interp(np.abs(th), [0, .5, .9, 1.2, 1.5, 2., 3.15], [.80, .80, .66, .30, .05, -.1, -.4]), .03): c.fill(q, hp['shade']); polys.append(q)
    lay(TB['L']); lay(TB['R'])
    for q in union(polys, 1., sig=.7, min_len=40): outline(c, q, lw(4.4), ln, seed + 11, press=.4)
    if cig: H_.tip = cigarette(c, H_, s, lw, f, smoke, ember, kind, cig_dir)
    return H_


def back_head(c, cx, cy, s, yaw, roll=0., f=0, wind=0., key=-1, seed=3):
    """head from behind or in lost profile: hair only (and the edge of a cheek)"""
    H_ = make(); H_.pose(cx, cy, s, yaw, 0., roll); hp = HAIR; G = []
    F = H_.face(); c.fill(F, PAL['shade']); outline(c, F, max(2., s * .014), INK, 2)
    for th in np.linspace(-math.pi, math.pi, 23)[:-1]:
        x, z = math.sin(th), math.cos(th)
        if z > .55: continue
        G.append(([(.16 + x * .12, 1.05, -.05 + z * .1), (.10 + x * .55, 1.0, z * .55 - .1), (x * .88, .55, z * .86 - .10), (x * .95 + wind * .10, .0, z * .92 - .12), (x * .92 + wind * .24, -.42, z * .88 - .14), (x * .80 + wind * .42, -.80 - .06 * math.sin(th * 5), z * .78 - .16)], .42))
    for P, w, big, sd in tables()['R'] + tables()['L']: G.append((np.array(P) + np.linspace(0, 1, 6)[:, None] ** 1.6 * A_(wind * .34, 0, 0), w))
    G = sorted(G, key=lambda g: float(np.mean(H_.depth(np.array(g[0], float))[1:5]))); polys = []
    for j, (P, w) in enumerate(G): S = strand(H_, P, w, prof=SPROF); polys.append(S[0]); strand_draw(c, H_, S, hp, (key * .5, .55, .6), INK, 2.0, seed + j * 7, inner=1, band=(.22, .36))
    for q in union(polys, 1., sig=.7, min_len=30): outline(c, q, max(2.4, s * .016), INK, 5, press=.4)
    return H_


def cigarette(c, H_, s, lw, f=0, smoke=True, ember=.6, kind='flat', cd=(.12, -.10, .42)):
    y = -.655 if kind != 'laugh' else -.60; x0 = .11 if cd[0] > 0 else -.06; a = H_.P3(np.array([(x0, y, float(H_.z(x0, y)) - .01)]))[0]; b = H_.P3(np.array([(x0 + cd[0], y + cd[1] + .012 * math.sin(f * .6), float(H_.z(x0, y)) + cd[2])]))[0]; d = b - a; L_ = np.hypot(*d); u = d / L_; m = A_(-u[1], u[0]); w = s * .034
    P = np.array([a + m * w, b + m * w, b - m * w, a - m * w]); c.fill(P, '#fbf7ee'); c.fill(np.array([a, b, b - m * w, a - m * w]), '#c3b8b4'); e = a + d * .30; c.fill(np.array([a + m * w, e + m * w, e - m * w, a - m * w]), '#d99a3c')
    t = b - d * .10; c.fill(np.array([t + m * w, b + m * w * .8, b + u * w * .6, b - m * w * .8, t - m * w]), '#ff8a2c'); c.fill(np.array([t + m * w * .6, b + m * w * .3, b - m * w * .5, t - m * w * .5]), '#ffe59a' if ember > .5 else '#ffb45a'); outline(c, P, lw(2.6), INK, 4)
    if smoke:
        tt = np.linspace(0, 1, 26)
        for j in range(2): sg = 1 if d[0] >= 0 else -1; q = b + np.c_[sg * tt ** 1.2 * s * (.50 + .16 * j) + np.sin(tt * 6.5 - f * .22 + j * 1.2) * (2 + 30 * tt) * s / 300, -tt * s * 1.25]; c.line(q, lw(6.5 - 2.5 * j), '#f6f2ea', taper=(.02, .8), press=.6, seed=j + 2, alpha=.5)
    return b


def body_suit(c, H_, cx, cy, s, pal, k, broll, yaw, seed, p, f=0):
    off = -yaw * .28; B = lambda P: xf(np.asarray(P, float) * [1, 1] + [off, -.20], s, s, broll, cx, cy); lw = p.w; ln = pal['line']
    N = spl(B([(-.40, .55), (-.415, 1.0), (-.50, 1.25), (-.8, 1.44), (-.5, 2.4), (.5, 2.4), (.8, 1.42), (.50, 1.22), (.41, .98), (.40, .55)]), True, (4, 5), n=6); c.fill(N, pal['base']); c.clip(N)
    jaw_cast(c, H_, s, broll, k, pal); c.fill(spl(B([(k * .10, .9), (k * .26, 1.3), (k * .06, 1.58), (k * .3, 2.4), (k * 1.1, 2.4), (k * 1.1, .8)]), True, n=5), pal['shade']); c.unclip()
    for q, w in (([(-.38, .80), (-.24, 1.2), (-.07, 1.56)], 2.8), ([(.38, .76), (.26, 1.18), (.09, 1.56)], 2.8), ([(-.08, 1.58), (-.30, 1.50), (-.55, 1.48)], 2.4), ([(.10, 1.58), (.34, 1.49), (.58, 1.46)], 2.4), ([(.0, 1.08), (.04, 1.16), (.0, 1.26)], 2.)): c.line(spl(B(q), n=8), lw(w), ln, taper=(.12, .4), press=.35, seed=seed + int(w * 10))
    for sg in (-1, 1):                                                       # jacket, shirt, sharp collar, wide lapel
        sd = sg == k
        J = spl(B([(sg * .60, 1.16), (sg * 1.0, 1.30), (sg * 1.6, 1.50), (sg * 2.05, 1.74), (sg * 2.3, 2.5), (sg * 2.3, 4.2), (-sg * .15, 4.2), (sg * .02, 3.0), (sg * .34, 2.3)]), True, (0, 3, 5, 6, 7), n=6)
        cloth(c, J, SUIT['base'] if not sd else SUIT['shade'], SUIT['shade'] if not sd else SUIT['deep'], [spl(B([(sg * 1.2, 1.6), (sg * 1.45, 2.2), (sg * 1.2, 2.9), (sg * 1.5, 4.2), (sg * .2, 4.2), (sg * .5, 2.6)]), True, (3, 4), n=6)], SUIT['light'] if not sd else SUIT['base'], [spl(B([(sg * 1.1, 1.36), (sg * 1.65, 1.56), (sg * 1.55, 1.72), (sg * 1.15, 1.52)]), True, n=6)], w=lw(5.), folds=[B([(sg * 1.2, 1.5), (sg * 1.4, 2.1), (sg * 1.32, 2.8)]), B([(sg * 1.7, 1.7), (sg * 1.75, 2.3), (sg * 1.6, 3.0)])], fw=lw(2.2), seed=seed + sg)
        S_ = spl(B([(sg * .46, 1.10), (sg * .62, 1.12), (sg * .70, 1.40), (sg * .36, 2.5), (sg * .02, 3.0), (sg * .06, 2.05), (sg * .30, 1.60)]), True, (0, 1, 4), n=6); cloth(c, S_, SUIT['shirt'], SUIT['shirtd'], [spl(B([(sg * .30, 1.5), (sg * .22, 2.2), (0, 3.0), (sg * .06, 1.9)]), True, n=5)] + ([S_] if sd else []), w=lw(3.6), seed=seed)
        C_ = spl(B([(sg * .40, .98), (sg * .56, .96), (sg * .82, 1.26), (sg * .92, 1.60), (sg * .70, 2.02), (sg * .24, 1.66), (sg * .36, 1.30)]), True, (0, 1, 4, 5), n=6); cloth(c, C_, SUIT['shirt'] if not sd else '#cf9a2a', SUIT['shirtd'], [spl(B([(sg * .56, 1.5), (sg * .62, 1.96), (sg * .32, 1.64), (sg * .40, 1.32)]), True, (1, 2), n=5)], '#ffe48c', [spl(B([(sg * .52, 1.06), (sg * .60, 1.06), (sg * .72, 1.34), (sg * .66, 1.5)]), True, n=5)], w=lw(4.2), seed=seed + 3)
        Lp = spl(B([(sg * .74, 1.22), (sg * 1.05, 1.34), (sg * 1.16, 1.78), (sg * .96, 1.94), (sg * 1.04, 2.12), (sg * .30, 3.1), (sg * .12, 2.9), (sg * .58, 2.1)]), True, (0, 1, 2, 3, 4, 5, 6), n=6); cloth(c, Lp, SUIT['light'] if not sd else SUIT['base'], SUIT['base'] if not sd else SUIT['shade'], [spl(B([(sg * .84, 1.4), (sg * .92, 1.9), (sg * .6, 2.6), (sg * .2, 3.1), (sg * .3, 2.6), (sg * .6, 2.0)]), True, n=6)], w=lw(4.2), seed=seed + 5)
        for j in range(2):
            q = B([(sg * .62, 2.7 + j * .7)])[0]; a = np.linspace(0, 6.283, 20); R_ = s * .085; c.fill(q + np.c_[np.cos(a), np.sin(a)] * R_, SUIT['btn']); c.fill(q + np.c_[np.cos(a), np.sin(a)] * R_ * .8 + [R_ * .2, R_ * .2], '#a0792c'); c.line(q + np.c_[np.cos(a), np.sin(a)] * R_, lw(2.4), ln, closed=True)


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, **kw):
    x0, y0 = int(max(0, cx - 4.4 * s)), int(max(0, cy - 4.4 * s)); x1, y1 = int(min(W, cx + 4.4 * s)), int(min(H, cy + 4.4 * s))
    if x1 - x0 < 8 or y1 - y0 < 8: return np.zeros((H, W, 4), f32)
    kw = dict(kw); rims = kw.pop('rims', None); boost = kw.pop('boost', 1.); D = LIGHTS.get(light, LIGHTS['day']) if isinstance(light, str) else light; kw.setdefault('key', D['key']); kw.setdefault('broll', roll)
    c = Cel(x1 - x0, y1 - y0, x0, y0); draw(c, cx, cy, s, yaw, pitch, roll, expr, f=f, **kw); return light_layer(c.layer(), D, cx, cy, s, boost, rims)


def g1(f):
    x, y = 1690, 360; ux, uy = (960 - x), (500 - y); n = math.hypot(ux, uy); ux, uy = ux / n, uy / n; roll = math.atan2(ux, -uy); dx, dy, dr, k = ring_pose(x, y, f, 4.1); lg = f >= 20
    o = .13 + .06 * math.sin((f // 2) * 1.25 + 1) if lg else 0; D = dict(key=-1, rims=[((-ux * 10, -uy * 10), '#fff6d2', 1.)], g=(('#fffaf2', '#cfa9a0')), L=(-.5, .6, .55))
    return head(x + dx - dy * .9, y + dy * .2, 150 * k, -.08, -.10, roll + dr, 'laugh' if f >= 21 else 'worry', D, f // 2, mouth_=('laugh', dict(o=o)) if f >= 21 else None, wind=.12 * math.sin(f * .2), look=(0, -.02))
