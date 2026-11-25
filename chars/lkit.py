"""Ladies lane kit (Nami, Robin, Bonney): face rig on head3, Oda eyes / brows / mouths, tears (pool, rope, bead), long hair with real
flow (root, mass, secondary strands that lag on the wind), cloth, inside-the-line rim light.  All screen px; heads in head3 front units."""
import numpy as np, cv2, math, cairo
from op import W, H
from op.paint import rgb, mix, smooth, f32, grid
from op.ink import Cel, spl, resample, noise1, xf
from op.kit import rim, para, tint, hatch, outline, shifted
from op.head3 import Head
from op import hands as HD
INK = '#17131a'
TAU = 2 * math.pi


def lay(dst, src, op=1.):
    """premultiplied src over premultiplied dst"""
    if op != 1.: src = src * op
    return src + dst * (1 - src[..., 3:4])


def rs(*k): return np.random.default_rng([abs(int(v)) + 1 for v in k])
def lerp(a, b, t): return a + (b - a) * t
def sm(t): t = np.clip(t, 0., 1.); return t * t * (3 - 2 * t)
def keys(f, K, e=sm):
    """scalar/array keys with eased inbetweens; key = (f, v) or (f, v, easefn)"""
    if f <= K[0][0]: return np.asarray(K[0][1], float)
    for a, b in zip(K[:-1], K[1:]):
        if f < b[0]: return lerp(np.asarray(a[1], float), np.asarray(b[1], float), (b[2] if len(b) > 2 else e)((f - a[0]) / float(b[0] - a[0])))
    return np.asarray(K[-1][1], float)
def out3(t): return 1 - (1 - np.clip(t, 0, 1)) ** 3
def back(t, k=1.9): t = np.clip(t, 0, 1) - 1; return 1 + (k + 1) * t ** 3 + k * t * t


def stroke(cel, P, w, col=INK, alpha=1., closed=True):
    c = cel.ctx; P = np.asarray(P, float); c.new_path(); c.move_to(*P[0])
    for p in P[1:]: c.line_to(p[0], p[1])
    if closed: c.close_path()
    cel._src(col, alpha); c.set_line_width(w); c.set_line_join(cairo.LINE_JOIN_ROUND); c.set_line_cap(cairo.LINE_CAP_ROUND); c.stroke()


def soft(cel, x, y, r, col, a=.5, within=None, sy=1., rot=0.):
    """airbrush spot"""
    if within is not None: cel.clip(within)
    c = cel.ctx; c.save(); c.translate(x, y); c.rotate(rot); c.scale(1, sy); k = [float(v) for v in rgb(col)]
    g = cairo.RadialGradient(0, 0, 0, 0, 0, r); g.add_color_stop_rgba(0, *k, a); g.add_color_stop_rgba(.45, *k, a * .55); g.add_color_stop_rgba(1, *k, 0)
    c.set_source(g); c.arc(0, 0, r, 0, TAU); c.fill(); c.restore()
    if within is not None: cel.unclip()


def outside(cel, poly):
    """clip to everything OUTSIDE poly (undo with cel.unclip())"""
    c = cel.ctx; c.save(); c.new_path(); c.rectangle(-4000, -4000, 12000, 12000); p = np.asarray(poly, float); c.move_to(*p[0])
    for q in p[1:]: c.line_to(q[0], q[1])
    c.close_path(); c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); c.clip(); c.set_fill_rule(cairo.FILL_RULE_WINDING)


def lit(layer, d, col, lw=4., w=8., amt=1., inner=.35, soft_=.6, where=None):
    """hot rim inside the ink line.  d = unit vector FROM the light (screen).  hard band w px + soft inner falloff"""
    a = layer[..., 3]; dx, dy = d; A0 = shifted(a, dx * lw, dy * lw) * a; c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer
    A1 = cv2.GaussianBlur(shifted(a, dx * (lw + w * 2.4), dy * (lw + w * 2.4)), (0, 0), w * .9); m2 = np.clip(A0 - A1, 0, 1) * inner
    m = np.clip(A0 - shifted(a, dx * (lw + w), dy * (lw + w)), 0, 1)
    if soft_: m = cv2.GaussianBlur(m, (0, 0), soft_) * np.minimum(A0 * 1.0 + .0, 1)
    m = np.clip(np.maximum(m, m2), 0, 1) * amt
    if where is not None: m = m * where
    o = layer.copy(); o[..., :3] = layer[..., :3] * (1 - m[..., None]) + c * (m * a)[..., None]; return o


def loc(cx, cy, s, roll, pts):
    """local body units (x right, y DOWN from a pivot) -> screen"""
    return xf(np.asarray(pts, float), s, s, roll, cx, cy)


# ================================================================= face
class Face:
    def __init__(self, hd, jaw=0., jy=-.46):
        self.h = hd; self.s = hd.s; self.jaw = jaw; d = np.array([-hd.R[0, 1], hd.R[1, 1]]); self.dn = d / (np.hypot(*d) + 1e-9); self.v0 = hd.P([(0, jy)])[0] @ self.dn
    def w(self, P):
        P = np.asarray(P, float)
        if not self.jaw: return P
        e = np.maximum(0, P @ self.dn - self.v0); return P + (self.jaw * e)[..., None] * self.dn
    def X(self, xy, dz=0.): return self.w(self.h.P(np.asarray(xy, float), dz))
    def X3(self, Q): return self.w(self.h.P3(np.asarray(Q, float)))
    def contour(self, top=.75):
        C = self.w(self.h.face(top)); v = C @ self.dn; vm = v.max(); lat = np.array([-self.dn[1], self.dn[0]]); l = C @ lat; lc = l[v > vm - .05 * self.s].mean()
        b = np.clip(1 - ((l - lc) / (.31 * self.s)) ** 2, 0, 1) * sm((v - (vm - .20 * self.s)) / (.20 * self.s)); return C + self.dn * (b * (.018 + .13 * min(self.jaw, .4)) * self.s)[:, None]
    def chin(self): return self.X([(0, -1.)])[0]


def scalp3(hd, th, y, lift=.10):
    th, y = np.broadcast_arrays(np.asarray(th, float), np.asarray(y, float)); A = np.interp(y, hd.yy, hd.A) + lift * .8
    Z = np.where(np.cos(th) >= 0, np.interp(y, hd.yy, hd.F), np.interp(y, hd.yy, hd.B)) + lift; p = hd.k['p']; sx, cz = np.sin(th), np.cos(th); r = (np.abs(sx) ** p + np.abs(cz) ** p) ** (-1 / p)
    return np.stack([A * sx * r, y + lift * np.clip(y, 0, 1) * .9, Z * cz * r], -1)


# normalised eye curves, inner -> outer
ROUND = dict(up=[(-1, -.14), (-.86, .40), (-.46, .84), (.04, 1.0), (.52, .86), (.88, .44), (1.04, .04)], lo=[(-1, -.14), (-.86, -.40), (-.52, -.66), (-.06, -.80), (.42, -.72), (.82, -.42), (1.04, .04)])
ALMOND = dict(up=[(-1, -.34), (-.82, .20), (-.42, .70), (.06, .90), (.54, .82), (.90, .50), (1.08, .22)], lo=[(-1, -.34), (-.84, -.52), (-.48, -.74), (-.02, -.82), (.46, -.68), (.86, -.28), (1.08, .22)])


def eye(cel, F, sd, E, o=1., sq=0., gaze=(0, 0), ir=1., shut='soft', pool=0., lw=4., skin='#f2c39c', f=0, lidshade='#b9866e', wob=0.):
    cx, cy = E['c']; ew, eh = E['w'], E['h']; U = spl(np.array(E['up'], float), n=7); L = spl(np.array(E['lo'], float), n=7)
    vc = (lambda u: -.34 - .30 * (1 - u * u) + .10 * u) if shut == 'soft' else (lambda u: -.30 + .52 * (1 - u * u) - .06 * u)
    k = float(np.clip(1 - o, 0, 1)); U = U.copy(); L = L.copy(); U[:, 1] = U[:, 1] * max(o, 1.) * (1 - k) + vc(U[:, 0]) * k; L[:, 1] = L[:, 1] * (max(o, 1.) ** .5) * (1 - k * .8) + vc(L[:, 0]) * k * .8
    L[:, 1] += sq * .42 * np.clip(1 - L[:, 0] ** 2, 0, 1); L[:, 1] = np.minimum(L[:, 1], U[:, 1] + .02)
    q = lambda P: np.c_[(cx + P[:, 0] * ew) * sd, cy + P[:, 1] * eh]
    X = lambda P, dz=.004: F.X(q(P), dz)
    n = len(U); t = np.linspace(0, 1, n); lid = E.get('lid', .05) / eh
    if o < .14:                                                       # closed: one brush-thick line
        C = np.c_[U[:, 0], vc(U[:, 0])]; th = lid * .72 * np.sin(np.pi * np.clip(t * .9 + .07, 0, 1)) ** .7; sg = 1 if shut == 'soft' else .8
        cel.fill(np.vstack([X(C + np.c_[0 * t, th * .55]), X(C - np.c_[0 * t, th * .55])[::-1]]), INK)
        for tt, du, dv in E.get('lash', []):
            i = int(tt * (n - 1)); b0, b1 = C[max(0, i - 2)], C[min(n - 1, i + 2)]; tip = C[i] + np.array([du, (dv * -.55 if shut == 'soft' else dv * .2)]) * .9
            cel.fill(X(spl([b0, (b0 + tip) / 2 + [0, .03], tip, (b1 + tip) / 2, b1], n=4)), INK)
        if shut != 'soft':
            for a, b in (((-1.02, -.30), (-1.36, -.02)), ((-1.0, -.42), (-1.30, -.40)), ((1.02, -.30), (1.40, -.10)), ((1.0, -.42), (1.34, -.52))): cel.line(X(np.linspace(a, b, 6)), lw * .42, INK, taper=(.1, .6))
            cel.line(X(spl([(-.7, .52), (-.1, .86), (.6, .66)], n=8)), lw * .40, INK, taper=(.35, .4))
        else: cel.line(X(np.c_[U[:, 0], vc(U[:, 0]) + .62][4:-4]), lw * .30, INK, taper=(.4, .4))
        return dict(low=q(np.c_[L[:, 0], vc(L[:, 0])]), c=F.X([(cx * sd, cy)])[0])
    S = np.vstack([U, L[::-1]]); SP = X(S)
    cel.fill(SP, E.get('white', '#fffdf8')); cel.clip(SP)
    icx = cx * sd + gaze[0] * ew * .46 - sd * ew * .02; icy = cy + gaze[1] * eh * .42 + eh * .03; a = np.linspace(0, TAU, 40, endpoint=False); rx, ry = E['iris'] * ew * ir, E['irisv'] * eh * ir
    el = lambda kx, ky, ox=0., oy=0.: F.X(np.c_[icx + np.cos(a) * rx * kx + ox * rx, icy + np.sin(a) * ry * ky + oy * ry], .004)
    cd, cm, cl = E['col']
    cel.fill(el(1, 1), INK); cel.fill(el(.91, .93), cd); cel.fill(el(.84, .84, 0, -.12), cm); cel.fill(el(.66, .46, 0, -.44), cl); cel.fill(el(.52, .56, 0, .02), INK)
    for j in range(7):                                                # iris rays
        b = -math.pi * (.14 + .72 * j / 6.); cel.line(F.X(np.array([(icx + math.cos(b) * rx * .58, icy + math.sin(b) * ry * .58), (icx + math.cos(b) * rx * .84, icy + math.sin(b) * ry * .84)]), .004), lw * .22, cd, alpha=.8)
    cel.fill(np.vstack([X(U), X(U - np.c_[0 * t, .20 + .16 * np.sin(np.pi * t)])[::-1]]), '#56465e', .50)     # lid shadow on the eyeball
    if pool > 0:
        lv = np.minimum(L[:, 1] + (pool * 1.05 + .06 * np.sin(L[:, 0] * 7 + f * 1.1)) * np.clip(1.06 - np.abs(L[:, 0]) ** 2.4, 0.1, 1), U[:, 1]); top = np.c_[L[:, 0], lv]
        cel.fill(np.vstack([X(L), X(top)[::-1]]), '#cfeaf8', .52); cel.line(X(top), lw * .34, '#ffffff', taper=(.1, .1)); cel.line(X(top - [0, .07]), lw * .20, '#5f97bd', taper=(.2, .2), alpha=.8)
    j1, j2 = (math.sin(f * 1.7) * .05 * wob, math.cos(f * 1.3) * .05 * wob)
    cel.fill(el(.30 + .10 * wob, .27 + .08 * wob, -.30 + j1, .44 + j2), '#ffffff'); cel.fill(el(.15 + .08 * wob, .13 + .06 * wob, .38 - j2, -.42 + j1), '#ffffff', .95)
    if wob: cel.fill(el(.12, .16, -.5 - j1, -.30), '#ffffff', .8)
    cel.unclip()
    i0 = int(n * .12); cel.line(X(L)[i0:], lw * .46, INK, taper=(.45, .15), press=.2)
    for tt, du, dv in E.get('lash2', []): i = int(tt * (n - 1)); cel.line(X(np.array([L[i], L[i] + [du, dv]])), lw * .34, INK, taper=(.05, .7))
    th = lid * (.30 + .70 * np.sin(np.pi * np.clip(t * .78 + .10, 0, 1)) ** .8) * (1. if o > .4 else .85)
    cel.fill(np.vstack([X(U - np.c_[0 * t, th * .25]), X(U + np.c_[.04 * t ** 4, th])[::-1]]), INK)
    for tt, du, dv in E.get('lash', []):
        i = int(tt * (n - 1)); b0, b1 = U[max(0, i - 2)] + [0, th[i] * .3], U[min(n - 1, i + 2)] + [0, th[i] * .2]; tip = U[i] + np.array([du, dv * (.35 + .65 * min(o, 1.))])
        m = (b0 + b1) / 2; cel.fill(X(np.vstack([spl([b0, lerp(b0, tip, .55) + [0, .05], tip], n=5), spl([tip, lerp(b1, tip, .5) + [0, -.01], b1], n=5)])), INK)
    if E.get('hood') and o > .2:                                      # hooded lid: soft shadow between lash line and fold
        hd_ = E['hood'] * (1.25 - .25 * min(o, 1.1)); cel.fill(np.vstack([X(U + np.c_[0 * t, th * .8]), X(U + np.c_[0 * t, th + hd_ * np.sin(np.pi * np.clip(t * .9 + .06, 0, 1)) ** .7])[::-1]]), lidshade, .40)
    if E.get('crease', True) and o > .3: cel.line(X((U + np.c_[0 * t, th + .20 + .10 * (1 - min(o, 1.))])[int(n * .16):int(n * .86)]), lw * .36, INK, taper=(.4, .45))
    return dict(low=q(L), c=F.X([(cx * sd, cy)])[0])


def brow(cel, F, sd, B, bi=0., bo=0., bk=0., col=INK, edge=None, lw=3.):
    x0, y0 = B['in']; xm, ym = B['mid']; x1, y1 = B['out']; C = spl([(x0 - bk * .06, y0 + bi - bk * .05), (lerp(x0, xm, .5), lerp(y0, ym, .55) + bi * .72 + bo * .18), (xm, ym + bi * .40 + bo * .50), (x1, y1 + bo)], n=8)
    t = np.linspace(0, 1, len(C)); th = B['t'] * np.minimum(.45 + 1.6 * t, 1) * (1 - t ** 2.2) ** .8; P = np.vstack([C + np.c_[0 * t, th * .5], (C - np.c_[0 * t, th * .5])[::-1]]) * [sd, 1]; Q = F.X(P, .01); cel.fill(Q, col)
    if edge: stroke(cel, Q, lw * .3, edge)


def _pts(x, y): return np.c_[np.asarray(x, float), np.asarray(y, float)]
_L9 = np.linspace(-1, 1, 9)
MOUTHS = dict(
    calm=(_pts(_L9 * .15, [.006, .001, -.006, -.010, -.007, -.010, -.006, .001, .006]),) * 2,
    smile=(_pts(_L9 * .23, .062 * _L9 ** 2 - .018),) * 2,
    smirk=(_pts(_L9 * .20, .03 * _L9 ** 2 - .012 + .035 * np.clip(_L9, 0, 1) ** 1.5),) * 2,
    press=(_pts(_L9 * .125, [-.026, -.006, .008, .013, .012, .013, .008, -.006, -.026]),) * 2,
    tremble=(_pts(_L9 * .175, [-.034, -.002, .020, -.002, .016, -.006, .020, -.004, -.038]),) * 2,
    sob=(_pts(_L9 * .18, [-.015, .028, .046, .026, .040, .024, .048, .026, -.018]), _pts(_L9 * .18, [-.015, -.060, -.084, -.066, -.086, -.068, -.084, -.056, -.018])),
    clench=(_pts(_L9 * .25, [-.030, .026, .050, .030, .046, .028, .052, .024, -.034]), _pts(_L9 * .25, [-.030, -.085, -.104, -.082, -.102, -.084, -.106, -.082, -.034])),
    howl=(_pts([-.345, -.325, -.225, -.105, 0, .105, .225, .325, .345], [-.075, .045, .130, .108, .134, .106, .132, .045, -.075]), _pts([-.345, -.350, -.265, -.130, 0, .130, .265, .350, .345], [-.075, -.215, -.315, -.335, -.318, -.338, -.312, -.212, -.075])),
    wail=(_pts([-.335, -.315, -.220, -.10, 0, .10, .220, .315, .335], [-.085, .040, .140, .114, .142, .112, .142, .040, -.085]), _pts([-.335, -.335, -.250, -.125, 0, .125, .250, .335, .335], [-.085, -.190, -.262, -.276, -.258, -.278, -.260, -.188, -.085])),
    grin=(_pts(_L9 * .31, [.110, .060, .030, .016, .012, .016, .030, .060, .110]), _pts(_L9 * .31, [.110, -.015, -.095, -.140, -.155, -.140, -.095, -.015, .110])),
    laugh=(_pts(_L9 * .32, [.095, .076, .066, .060, .058, .060, .066, .076, .095]), _pts([-.32, -.295, -.22, -.12, 0, .12, .22, .295, .32], [.095, -.090, -.215, -.275, -.290, -.275, -.215, -.090, .095])),
    oh=(_pts(_L9 * .10, [0, .032, .045, .050, .050, .050, .045, .032, 0]), _pts(_L9 * .10, [0, -.045, -.066, -.076, -.078, -.076, -.066, -.045, 0])),
    sobopen=(_pts([-.300, -.268, -.190, -.095, 0, .095, .190, .268, .300], [-.110, -.004, .080, .112, .101, .112, .080, -.004, -.110]), _pts([-.300, -.318, -.272, -.152, 0, .152, .272, .318, .300], [-.110, -.168, -.218, -.236, -.228, -.236, -.218, -.168, -.110])),
    bite=(_pts(_L9 * .30, [-.02, .07, .11, .125, .13, .125, .11, .07, -.02]), _pts([-.30, -.30, -.24, -.13, 0, .13, .24, .30, .30], [-.02, -.16, -.26, -.30, -.31, -.30, -.26, -.16, -.02])),
)


def mmix(a, b, t):
    A, B = MOUTHS[a] if isinstance(a, str) else a, MOUTHS[b] if isinstance(b, str) else b
    return (A[0] * (1 - t) + B[0] * t, A[1] * (1 - t) + B[1] * t)


def mouth(cel, F, M, st, lw=4., teeth=(0., 0.), tongue=0., clench=False, c=(0, -.62), sx=1., sy=1., seed=0, shark=False, jit=0., lip2=0.):
    up, lo = MOUTHS[M] if isinstance(M, str) else M
    if jit: r = rs(seed, 71); up = up + r.normal(0, jit, up.shape); lo = lo + r.normal(0, jit, lo.shape); up[0] = lo[0] = (up[0] + lo[0]) / 2; up[-1] = lo[-1] = (up[-1] + lo[-1]) / 2
    up = spl(up * [sx, sy] + c, n=6); lo = spl(lo * [sx, sy] + c, n=6); gap = float(np.max(up[:, 1] - lo[:, 1])); m = len(up); t = np.linspace(0, 1, m)
    if gap < .022:
        cel.line(F.X(up, .01), lw * .86, INK, taper=(.28, .28), press=.35, seed=seed)
        w = (up[-1, 0] - up[0, 0])
        for sgn, e in ((-1, up[0]), (1, up[-1])): cel.line(F.X(np.array([e + [0, .014], e + [sgn * .014, -.004], e + [sgn * .012, -.020]]), .01), lw * .48, INK, taper=(.4, .5))
        y = c[1] - .080; cel.line(F.X(spl([(-w * .20, y + .006), (0, y - .004), (w * .20, y + .006)], n=6), .01), lw * .42, st['line2'], taper=(.4, .4))
        if st.get('lip'): cel.fill(F.X(np.vstack([spl([(-w * .30, c[1] - .018), (0, c[1] - .052), (w * .30, c[1] - .018)], n=8), spl([(w * .30, c[1] - .018), (0, c[1] - .026), (-w * .30, c[1] - .018)], n=8)]), .01), st['lip'], .55)
        return
    poly = np.vstack([up, lo[::-1]]); P = F.X(poly, .01); cel.fill(P, st['mouth']); cel.clip(P)
    hw = (up[-1, 0] - up[0, 0]) / 2; mid = (up[m // 2] + lo[m // 2]) / 2
    cel.fill(F.X(spl([(-hw * .62, mid[1] + gap * .16), (0, mid[1] + gap * .44), (hw * .62, mid[1] + gap * .16), (0, mid[1] - gap * .10)], True, n=8), .01), '#2c0713')
    if tongue > 0:
        b = lo[m // 2, 1]; T = spl([(-hw * .70, b - .03), (-hw * .60, b + gap * tongue * .55), (-hw * .22, b + gap * tongue * .98), (0, b + gap * tongue * .88), (hw * .22, b + gap * tongue * .98), (hw * .60, b + gap * tongue * .55), (hw * .70, b - .03)], n=8)
        TP = F.X(T, .01); cel.fill(TP, '#d4606a'); cel.shade(F.X(np.vstack([T[:len(T) // 2 + 6] + [.0, -gap * .13], T[:1] + [0, -.1]]), .01), '#a53c52', TP, .8); cel.line(TP, lw * .5, '#4a0f1e', taper=(.1, .1))
        cel.line(F.X(spl([(0, b + gap * tongue * .72), (.004, b + gap * tongue * .45), (0, b + gap * tongue * .2)], n=6), .01), lw * .4, '#7c2438', taper=(.2, .6)); soft(cel, *F.X([(-hw * .22, b + gap * tongue * .78)], .01)[0], F.s * gap * .13, '#ffd9d4', .85)
    if shark:
        K = 9; tw = 2 * hw / K
        for row, cur, sg, hgt in ((0, up, -1, min(.12, gap * .48)), (1, lo, 1, min(.11, gap * .46))):
            for k in range(K):
                x0 = up[0, 0] + tw * (k + (.0 if row == 0 else -.5)); yb = lambda x: np.interp(x, cur[:, 0], cur[:, 1]); xm = x0 + tw / 2; T = np.array([(x0, yb(x0) - sg * .06), (x0, yb(x0)), (xm, yb(xm) + sg * hgt * (1 - .3 * abs(xm) / hw)), (x0 + tw, yb(x0 + tw)), (x0 + tw, yb(x0 + tw) - sg * .06)]); TP = F.X(T, .012); cel.fill(TP, '#fdfaf2'); cel.line(TP[1:4], lw * .55, INK, taper=(.05, .05))
    elif clench:
        cel.fill(P, '#fdf8ee'); cel.shade(F.X(np.vstack([lo, (lo + [0, gap * .26])[::-1]]), .01), '#cdb9b4', P); md = (up + lo) / 2; cel.line(F.X(md + np.c_[0 * t, .008 * np.sin(t * 19)], .01), lw * .55, '#4a2a2c', taper=(.1, .1))
        for k in np.linspace(.12, .88, 8): i = int(k * (m - 1)); cel.line(F.X(np.array([lerp(up[i], md[i], .25), lerp(up[i], lo[i], .78)]), .01), lw * .26, '#6d4a48', taper=(.3, .3), alpha=.8)
    else:
        for th, cur, sg in ((teeth[0], up, -1), (teeth[1], lo, 1)):
            if th <= 0: continue
            e = cur + np.c_[0 * t, sg * th * (.55 + .45 * np.sin(np.pi * t))]; B = F.X(np.vstack([cur - np.c_[0 * t, 0 * t + sg * .05], e[::-1]]), .01); cel.fill(B, '#fdfaf2')
            cel.fill(F.X(np.vstack([e - np.c_[0 * t, 0 * t + sg * th * .26], e[::-1]]), .01), '#cfc0bd' if sg < 0 else '#b9a6a8'); cel.line(F.X(e, .01), lw * .42, '#3c2226', taper=(.08, .08))
            for k in np.linspace(.16, .84, 7): i = int(k * (m - 1)); cel.line(F.X(np.array([cur[i], lerp(cur[i], e[i], .8)]), .01), lw * .20, '#8a7270', taper=(.5, .5), alpha=.55)
    cel.unclip()
    if lip2:                                                         # lower lip pulled wide
        e = lo - np.c_[0 * t, lip2 * np.sin(np.pi * np.clip(t, .06, .94)) ** .6]; L_ = F.X(np.vstack([lo, e[::-1]]), .012); cel.fill(L_, st.get('lip') or '#b8605a'); cel.clip(L_); cel.fill(F.X(np.vstack([lerp(lo, e, .55), e[::-1]]), .012), '#8a3d40', .8); cel.unclip(); i = int(m * .34); cel.line(F.X(lerp(lo, e, .34)[i:i + int(m * .30)], .014), lw * .5, '#ffd9cf', taper=(.4, .4), alpha=.9); cel.line(F.X(e, .012), lw * .62, INK, taper=(.15, .15), press=.3, seed=seed + 4)
    cel.line(F.X(up, .01), lw * .95, INK, taper=(.12, .12), press=.35, seed=seed); cel.line(F.X(lo, .01), lw * .80, INK, taper=(.12, .12), press=.35, seed=seed + 1)
    y = lo[m // 2, 1] - .055 * (1 + gap * .5); cel.line(F.X(spl([(-hw * .28, y + .008), (0, y - .006), (hw * .30, y + .010)], n=6), .01), lw * .42, st['line2'], taper=(.4, .4))


def nose(cel, F, st, lw, light=-1, snot=0., f=0, flare=0.):
    h = F.h; N = h.nose_profile(); w, yt, ytip, yb, d = h.k['nose']; y = h.yaw; sg = -1 if light < 0 else 1
    P = F.X3(N); side = 1 if light < 0 else -1; z0 = float(h.z(0, yb))
    # shadow plane under / beside the tip
    S = F.X3(np.array([N[3], N[4], N[5], N[6], (side * (w + .035 + flare * .02), yb + .02, z0 + d * .12), (side * (w + .01), ytip + .05, z0 + d * .25)])); cel.fill(spl(S, True, n=5), st['shade'])
    if st.get('bridge'):                                             # long straight nose: shadow plane down one side, full bridge line
        zf = lambda x, yy: float(h.z(x, yy)); sd_ = [(side * .022, .10), (side * .042, -.08), (side * .060, -.26), (side * (w + .018), yb + .05)]
        Q = np.vstack([F.X3(N[:6]), F.X3(np.array([(x, yy, zf(x, yy)) for x, yy in sd_]))[::-1]]); cel.fill(Q, st['shade'], .66); cel.line(spl(P[[0, 1, 2, 3, 4]], n=8), lw * .56, INK, taper=(.80, .02), press=.2); t_ = F.X3(np.array([N[3] + [-side * .022, 0, 0], N[4] + [-side * .020, .01, 0]])); cel.line(t_, lw * .7, '#fff4e2', taper=(.4, .4), alpha=.8)
    a = st['nosea'] if 'nosea' in st else 2 if abs(y) > .12 else 3
    cel.line(spl(P[a:7], n=8), lw * .74, INK, taper=(.62, .25), press=.3)
    for s in (-1, 1):
        if s * y < -.42: continue
        q = F.X3(np.array([(s * (w * .35), yb + .012, z0 + d * .42), (s * (w * .78), yb + .030, z0 + d * .26), (s * (w + .012 + flare * .03), yb + .052, z0 + d * .12)])); cel.line(q, lw * .40, INK, taper=(.2, .55))
    if snot > 0:
        s = -1 if y > 0 else 1; b = F.X3(np.array([(s * w * .55, yb + .01, z0 + d * .32)]))[0]; L = F.s * .16 * snot * (1 + .08 * math.sin(f * .9)); dn = F.dn; nr = np.array([-dn[1], dn[0]]); ww = F.s * .030
        Q = spl([b - nr * ww * .5, b + dn * L * .5 - nr * ww * .8, b + dn * L * .95 - nr * ww * .6, b + dn * L * 1.05, b + dn * L * .95 + nr * ww * .7, b + dn * L * .5 + nr * ww * .7, b + nr * ww * .5], True, n=5); cel.fill(Q, '#dcecf0', .85); stroke(cel, Q, lw * .34, '#4b6474'); cel.line(np.array([b + dn * L * .25 - nr * ww * .2, b + dn * L * .8 - nr * ww * .2]), lw * .4, '#ffffff', taper=(.3, .3))


# ================================================================= tears
def bead(cel, x, y, r, ang=1.57, st=.6, lw=2., fill='#d6eefa', a=.92):
    """flying drop: round nose along ang, pulled tail; outline, shade crescent, two highlights"""
    t = np.linspace(0, TAU, 30, endpoint=False); c, s = np.cos(t), np.sin(t); px = np.where(c > 0, c, c * (1 + 1.9 * st)); py = np.where(c > 0, s, s * np.abs(s) ** (.5 * min(st, 1)))
    R = lambda X, Y: np.c_[x + (X * math.cos(ang) - Y * math.sin(ang)) * r, y + (X * math.sin(ang) + Y * math.cos(ang)) * r]
    B = R(px, py); cel.fill(B, fill, a); cel.shade(R(px * .95 + .12, py * .9 + .42), '#8dbfe0', B, .75); cel.fill(R(c * .24 + .12, s * .30 - .34), '#ffffff'); cel.fill(R(c * .10 + .52, s * .10 + .28), '#ffffff', .9); stroke(cel, B, lw, '#20344c')


def rope(cel, F, path, w, f=0, seed=0, lw=2.4, spill=.0, dz=.02, n=40, rate=.9):
    """fat tear rope on the face surface: path in front units, w px; bulges travel down it"""
    P = resample(spl(F.X(np.asarray(path, float), dz), n=10), 3.); P = P[np.linspace(0, len(P) - 1, n).round().astype(int)]; u = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    ww = w * (.62 + .26 * np.sin(TAU * (1.7 * u - f / 24. * rate * 2.2 + seed * .37)) + .12 * np.sin(TAU * (4.3 * u - f / 24. * rate * 3.1 + seed * .71))) * np.minimum(1, .80 + u * 1.2); ww[-3:] *= [1., .9, .6]
    A, B = P + N * ww[:, None] * .5, P - N * ww[:, None] * .5; Q = np.vstack([A, B[::-1]]); cel.fill(Q, '#cfe9f8', .90)
    cel.fill(np.vstack([B, (B + N * ww[:, None] * .34)[::-1]]), '#86b9dc', .75); cel.line((A - N * ww[:, None] * .28)[2:-3], max(1.2, w * .15), '#ffffff', taper=(.2, .3), press=.6, seed=seed)
    cel.line(A, lw, '#20344c', taper=(.05, .05), press=.4, seed=seed); cel.line(B, lw, '#20344c', taper=(.05, .05), press=.4, seed=seed + 1); return P[-1], T[-1]


def tearpath(sd, low=None, w=0., bend=0., k=.58):
    x0, y0 = (.40, -.13) if low is None else (abs(low[int(len(low) * k)][0]), low[int(len(low) * k)][1] - .02); b = x0 - .40 + bend
    return np.array([(x0 * sd, y0), ((.445 + b) * sd, -.26), ((.470 + b) * sd, -.40), ((.455 + b) * sd, -.56), ((.395 + b * .7) * sd, -.72), ((.31 + b * .5) * sd, -.86), ((.24 + b * .4) * sd, -.96)])


# ================================================================= face assembly
def face_base(cel, F, st, lw, light=-1, seed=0, contour=None, shade=None, blush=0., neck=True):
    """flat + hard shadow + contour"""
    C = F.contour() if contour is None else contour; cel.fill(C, st['skin']); sg = 1 if light < 0 else -1
    shade = st.get('shadeshape') if shade is None else shade
    S = np.array(shade if shade is not None else [(.68, .66), (.52, .46), (.565, .26), (.63, .08), (.585, -.06), (.555, -.20), (.50, -.34), (.425, -.47), (.345, -.585), (.315, -.68), (.285, -.78), (.17, -.90), (.02, -1.0), (.2, -1.10), (.9, -1.0), (1.05, 0), (.95, .6)], float)
    k = 13 if shade is None else len(S) - 4
    Q = np.vstack([spl(F.X(S[:k] * [sg, 1]), n=6), F.X(S[k:] * [sg, 1])]); cel.shade(Q, st['shade'], C)
    q = F.X([(-sg * .30, .10)])[0]; soft(cel, q[0], q[1], F.s * .80, st.get('glow', '#fff3dd'), st.get('glowa', .30), within=C); q = F.X([(sg * .05, -1.02)])[0]; soft(cel, q[0], q[1], F.s * .62, st['shade'], .50, within=C, sy=.7); q = F.X([(sg * .70, .05)])[0]; soft(cel, q[0], q[1], F.s * .55, st.get('sh2', st['shade']), .45, within=C)
    for k in range(4):                                               # pen hatching on the shadow edge of the cheek
        y = .02 - k * .062; x = .610 - .022 * k - (.006 * k * k if k > 3 else 0); cel.line(F.X(np.array([(sg * (x + .05), y + .045), (sg * (x - .030 - .012 * (k % 2)), y - .030)]), .004), lw * .36, st['line2'], taper=(.1, .7), alpha=.6)
    if st.get('cheekbone'):
        for sd in (-1, 1): cel.shade(F.X(spl([(sd * .62, -.12), (sd * .52, -.185), (sd * .42, -.30), (sd * .36, -.46), (sd * .45, -.43), (sd * .56, -.30)], True, n=5), .004), st['shade'], C, st['cheekbone'])
    u = F.dn; i = int(np.argmin(C @ u)); D = np.vstack([C[i:], C[:i + 1]]); n = len(D); return C, D


def contour_line(cel, D, lw, seed=0, a=.14, b=.86):
    n = len(D); cel.line(D[int(n * a):int(n * b)], lw, INK, taper=(.08, .08), press=.45, seed=seed, tremor=.35)


def cheek(cel, F, st, sd, amt=1., n=5, lw=3., col=None, y=-.20, x=.36):
    for k in range(n): cel.line(F.X(np.array([(sd * (x + k * .045), y + .035 * math.sin(k)), (sd * (x - .045 + k * .045), y - .10 - .02 * (k % 2))]), .01), lw * .40, col or st['blush'], taper=(.2, .5), alpha=amt)


def feats(cel, F, st, ex, lw, f=0, light=-1, seed=0, mid=None):
    """eyes, brows, nose, mouth, tears of an expression dict"""
    LOW = {}; E = st['eye']; o = ex.get('o', 1.); o = o if isinstance(o, (tuple, list)) else (o, o); sides = (-1, 1) if F.h.yaw <= 0 else (1, -1)
    if ex.get('blush'):
        for sd in (-1, 1): p = F.X([(sd * .44, -.27)], .01)[0]; soft(cel, p[0], p[1], F.s * .26, st['blush'], .45 * ex['blush'], within=None, sy=.6, rot=F.h.roll); cheek(cel, F, st, sd, .75 * ex['blush'], lw=lw)
    if not ex.get('nonose'): nose(cel, F, st, lw, light, ex.get('snot', 0.), f, ex.get('flare', 0.))
    if not ex.get('nomouth'): mouth(cel, F, ex.get('M', 'calm'), st, lw, ex.get('teeth', (0, 0)), ex.get('tongue', 0.), ex.get('clench', False), seed=ex.get('mseed', seed), c=ex.get('mc', (0, -.62)), sx=ex.get('msx', 1.), sy=ex.get('msy', 1.), shark=ex.get('shark', False), jit=ex.get('jit', 0.), lip2=ex.get('lip2', 0.))
    for sd in sides:
        oo = o[0] if sd < 0 else o[1]
        LOW[sd] = eye(cel, F, sd, E, oo, ex.get('sq', 0.), ex.get('gaze', (0, 0)), ex.get('ir', 1.), ex.get('shut', 'soft'), ex.get('pool', 0.), lw, st['skin'], f, st['shade'], ex.get('wob', 0.))['low']
        brow(cel, F, sd, st['brow'], ex.get('bi', 0.), ex.get('bo', 0.) if sd * ex.get('bside', sd) > 0 else ex.get('bo2', ex.get('bo', 0.)), ex.get('bk', 0.), st['browcol'], INK, lw)
    if ex.get('bk', 0) > .35:
        for sd in (-1, 1): cel.line(F.X(np.array([(sd * .045, .17), (sd * .060, .25), (sd * .052, .33)]), .01), lw * .46 * ex['bk'], INK, taper=(.35, .5))
    if ex.get('folds'):                                              # sobbing folds nose wing -> mouth corner, puckered chin
        for sd in (-1, 1): cel.line(F.X(spl([(sd * .120, -.385), (sd * .235, -.455), (sd * .345, -.575), (sd * .385, -.715)], n=8), .012), lw * .52 * ex['folds'], INK, taper=(.35, .6), press=.3, seed=sd + 3)
        for k in (-1, 0, 1): cel.line(F.X(np.array([(k * .055, -.945), (k * .050 + .012, -.972)]), .01), lw * .38, st['line2'], taper=(.3, .6))
    if ex.get('fore', 0) > 0:                                        # brow pain wrinkles
        for k, y in enumerate((.40, .48)): cel.line(F.X(spl([(-.20 + k * .03, y), (-.08, y + .045), (0, y + .030), (.08, y + .045), (.20 - k * .03, y)], n=6), .01), lw * .34 * ex['fore'], INK, taper=(.4, .4))
    if mid: mid()
    rw = ex.get('rope', 0.)
    if rw > 0:
        for sd in (-1, 1):
            if F.h.facing(sd * .42, -.4) < .03 or sd not in LOW: continue
            rope(cel, F, tearpath(sd, LOW[sd], bend=ex.get('ropebend', 0.)), rw * F.s, f, seed + (3 if sd > 0 else 0), max(2., lw * .55))
            if ex.get('rope2'): rope(cel, F, tearpath(sd, LOW[sd], bend=-.13 + ex.get('ropebend', 0.), k=.60), rw * F.s * .62, f + 5, seed + (13 if sd > 0 else 10), max(1.8, lw * .45))
            low = LOW[sd]; n = len(low); t = np.linspace(0, 1, n); hh = (.022 + rw * .55 * np.exp(-((t - .58) / .26) ** 2)) * np.sin(np.pi * np.clip(t, .02, .98)) ** .5 * (1 + .12 * np.sin(t * 11 + f * .9)); up = low + np.c_[0 * t, .012 + 0 * t]; B = F.X(np.vstack([up, (low - np.c_[0 * t, hh])[::-1]]), .012); cel.fill(B, '#d9effa', .94)
            cel.fill(F.X(np.vstack([low - np.c_[0 * t, hh * .62], (low - np.c_[0 * t, hh])[::-1]]), .013), '#86b9dc', .7); stroke(cel, B, max(1.6, lw * .40), '#20344c'); i = int(n * .36); cel.line(F.X(low[i:i + int(n * .34)] - np.c_[0 * t[i:i + int(n * .34)], hh[i:i + int(n * .34)] * .30], .016), lw * .55, '#ffffff', taper=(.3, .4))


# ================================================================= hair
def strand(root, L, a0, a1, t, n=26, amp=.16, waves=1.1, speed=.8, ph=0., curl=0., lagfn=None):
    """flowing centre line.  angles in screen rad (0 right, pi/2 down).  a travelling wave runs root -> tip; lagfn(u)->(n,2) px adds the follow-through of a moving head"""
    u = np.linspace(0, 1, n); th = a0 + (a1 - a0) * u ** .7 + amp * (.18 + .82 * u) * np.sin(TAU * (waves * u - speed * t + ph)) + .45 * amp * u * np.sin(TAU * (2.3 * waves * u - 1.7 * speed * t + ph * 1.7)) + curl * u ** 2.6
    d = np.c_[np.cos(th), np.sin(th)] * (L / (n - 1.)); P = np.asarray(root, float) + np.vstack([[0, 0], np.cumsum(d, 0)[:-1]])
    if lagfn is not None: P = P + lagfn(u)
    return P


def lag(posfn, f, frames=5.):
    """follow-through offset for strand(): points at u show where the root was `frames`*u frames ago"""
    p0 = np.asarray(posfn(f), float); return lambda u: np.array([np.asarray(posfn(f - frames * q), float) - p0 for q in u])


def wp(u, k=3.2, root=.5, belly=.35): return (root + (1 - root) * sm(u / belly)) * (1 - u ** k) ** .85 * (.35 + .65 * sm(u / .10))


class Lock:
    __slots__ = ('P', 'w', 'A', 'B', 'poly', 'N', 'o')
    def __init__(self, P, width, root=.5, k=3.2, belly=.35, n=30, **o):
        P = resample(np.asarray(P, float), 2.); P = P[np.linspace(0, len(P) - 1, n).round().astype(int)]; u = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9
        self.P = P; self.N = np.c_[-T[:, 1], T[:, 0]]; self.w = width * wp(u, k, root, belly); self.A = P + self.N * self.w[:, None] * .5; self.B = P - self.N * self.w[:, None] * .5; self.poly = np.vstack([self.A, self.B[::-1]]); self.o = o


def draw_lock(cel, L, pal, lw, seed=0, light=(-.6, -.8)):
    o = L.o; n = len(L.P); u = np.linspace(0, 1, n); r = rs(seed, 5); tone = o.get('tone', 0); base, shd = (pal['base'], pal['shade']) if tone == 0 else (pal['shade'], pal['deep']) if tone == 1 else (pal['deep'], pal['deep2'])
    cel.fill(L.poly, base); g = L.N @ np.asarray(light, float); k = o.get('sh', .50); wav = 1 + .28 * np.clip(noise1(n, 9, r), -1.4, 1.4)
    for E, I, s in ((L.B, L.A, 1), (L.A, L.B, -1)):
        fr = np.clip(sm((s * g + .25) / 1.1) * k * 1.35 * wav, 0, .92) * np.clip(1.12 - .35 * u, .5, 1)
        if fr.max() < .06: continue
        cel.fill(np.vstack([E, (E + (I - E) * fr[:, None])[::-1]]), shd)
    rs0 = o.get('rootsh', .0)
    if rs0 > 0: q = int(n * rs0); m = (L.A[:q] + L.B[:q]) / 2; cel.fill(np.vstack([L.A[:q], m[-1:] + (m[-1:] - m[-2:-1]) * 2.0, L.B[:q][::-1]]), shd)
    hi = o.get('hi', None)
    if hi is not None and tone == 0:
        for j, (a, b, off) in enumerate(hi):
            i0, i1 = int(a * (n - 1)), int(b * (n - 1))
            if i1 - i0 < 4: continue
            lg = 1 if g[i0:i1].mean() > 0 else -1; c = L.P[i0:i1] + L.N[i0:i1] * (lg * off * L.w[i0:i1] * .5)[:, None]; tt = np.linspace(0, 1, len(c)); hw = L.w[i0:i1] * o.get('hiw', .085) * np.sin(np.pi * tt) ** .7
            cel.fill(np.vstack([c + L.N[i0:i1] * hw[:, None], (c - L.N[i0:i1] * hw[:, None])[::-1]]), pal['light'])
    for j in range(int(o.get('inner', 2))):
        a = r.uniform(-.62, .62); i0 = int(n * r.uniform(0, .30)); i1 = int(n * r.uniform(.5, .94))
        if i1 - i0 > 5: cel.line((L.P + L.N * (a * L.w * .5)[:, None])[i0:i1], lw * r.uniform(.5, .9), pal.get('line', INK), taper=(.25, .55), press=.35, seed=int(r.integers(1 << 20)))
    e = o.get('edge', 1.)
    q = int(n * o.get('r0', .14) * .8)
    if e: cel.line(L.A[q:], lw * 1.15 * e, pal.get('line', INK), taper=(.16, .30), press=.45, seed=seed); cel.line(L.B[q:], lw * 1.15 * e, pal.get('line', INK), taper=(.16, .30), press=.45, seed=seed + 1)


def draw_mass(cel, locks, pal, lo, li, seed=0, light=(-.6, -.8), off=(1.0, 1.6), contour=INK):
    """thick silhouette round the whole group, thin lines between the locks"""
    for L in locks:
        if lo > 0 and L.o.get('sil', 1): i0 = int(len(L.P) * L.o.get('r0', .14)); stroke(cel, np.vstack([L.A[i0:], L.B[i0:][::-1]]) + np.asarray(off) * L.o.get('sil', 1), lo * 2 * L.o.get('sil', 1), contour, closed=False)
    for i, L in enumerate(locks): draw_lock(cel, L, pal, li, seed * 101 + i * 7, light)


def wisps(cel, roots, L, a, t, lw, col=INK, seed=0, amp=.3, speed=1., n=10, hi=None):
    """loose single hairs"""
    for i in range(n):
        r = rs(seed, i); P = strand(roots[i % len(roots)], L * r.uniform(.5, 1.15), a + r.normal(0, .25), a + r.normal(0, .5), t, amp=amp, waves=r.uniform(.8, 1.5), speed=speed * r.uniform(.8, 1.3), ph=r.random(), curl=r.normal(0, .6), n=20)
        cel.line(P, lw * r.uniform(.6, 1.2), hi if (hi and r.random() < .3) else col, taper=(.1, .8), press=.3, seed=i)


def cap(cel, F, pal, hairline, grow=1.12, lw=5., seed=0, light=(-.6, -.8), part=(.2, .97), nlines=9, hiy=(.70, .80), hir=(-1.25, .75), c0=(0, .02, 0), down=3.):
    """hair on the skull: everything outside the face window; crescent shadow, strand lines from the part, jagged highlight ring"""
    h = F.h; sk = h.skull(); c = h.P3([c0])[0]; big = c + (sk - c) * grow; win = F.X(spl(hairline, n=6)); dn = F.dn * F.s * down; window = np.vstack([win, win[-1:] + dn, win[:1] + dn]); outside(cel, window)
    cel.fill(big, pal['shade']); L = np.asarray(light, float); sh = big + L * F.s * .10; sh = c + (sh - c) * .985; cel.clip(big); cel.fill(sh, pal['base'])
    r = rs(seed, 9)
    for i in range(nlines):
        th1 = lerp(-2.0, 2.0, (i + .5 + r.normal(0, .18)) / nlines); v = np.linspace(.04, r.uniform(.55, .95), 14); th = part[0] + (th1 - part[0]) * sm(v * 1.15); y = lerp(part[1], r.uniform(.15, .45), v ** 1.5); cel.line(h.P3(scalp3(h, th, y, .12)), lw * r.uniform(.28, .46), pal.get('line', INK), taper=(.25, .5), press=.4, seed=i)
    if hiy:                                                          # highlight ring: lens streaks of uneven length along an arc
        K = 19
        for k in range(K):
            th = lerp(hir[0], hir[1], (k + .5 + r.normal(0, .12)) / K); gap = abs(math.sin(k * 2.4)) < .22
            if gap: continue
            y0 = hiy[0] - r.uniform(0, .05); y1 = hiy[1] + r.uniform(-.03, .02); dth = (hir[1] - hir[0]) / K * r.uniform(.30, .62); v = np.linspace(0, 1, 9); wv = np.sin(np.pi * v ** .8) ** .8 * dth; Y = lerp(y1, y0, v); cel.fill(np.vstack([h.P3(scalp3(h, th - wv * .5, Y, .125)), h.P3(scalp3(h, th + wv * .5, Y, .125))[::-1]]), pal['light'])
    cel.unclip(); stroke(cel, big, lw * 1.0, pal.get('contour', INK)); cel.unclip(); return big


def lock3(F, pts, width, t=0., sway=(0., 0.), amp=0., ph=0., speed=.9, n=30, **o):
    """fringe / side lock from 3-D control points + wind sway (px vector at the tip) + flutter (px)"""
    P = spl(F.X3(np.asarray(pts, float)), n=10); P = resample(P, 2.); P = P[np.linspace(0, len(P) - 1, n).round().astype(int)]; u = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    P = P + np.asarray(sway, float) * (u ** 1.7)[:, None] + N * (amp * u ** 1.4 * np.sin(TAU * (.9 * u - speed * t + ph)))[:, None]; return Lock(P, width, n=n, **o)


def shadow_of(cel, locks, within, d, col, alpha=1.):
    """hard cast shadow of fringe locks on the face"""
    cel.clip(within)
    for L in locks: cel.fill(L.poly + np.asarray(d, float), col, alpha)
    cel.unclip()


# ================================================================= cloth / body
def fold(cel, P, w=2.4, col=INK, seed=0): cel.line(spl(P, n=10), w, col, taper=(.12, .7), press=.4, seed=seed)


def hand_sprite(pose, side, R, s, bld=None, pal=None, light=(-.5, .75, .6), w=3.4, size=None, seed=3, res=420, dist=7., style=None, rimc=None, rimd=(.7, .5), rw=5, marks=.7, props=(), pivot=(0, 0, 0)):
    """hand + forearm as a premultiplied RGBA sprite; anchor (pivot) at the sprite centre"""
    S = int(size or s * 5.4) // 2 * 2; cel = Cel(S, S); h = HD.Hand(pose, side, R, (0, 0, 0), bld or HD.FEMALE, pivot=pivot, **(style or {})); g = HD.render([h], HD.Cam(S / 2, S / 2, s, dist), light, res=res, clip=(0, 0, S, S), props=list(props)); HD.ink(cel, g, pal, w=w, seed=seed, marks=marks); L = cel.rgba()
    if rimc: L = lit(L, rimd, rimc, w * .7, rw, 1.)
    return L, g


def put(dst, spr, x, y, scale=1., rot=0., flip=False, op=1., anchor=None, tintc=None, ta=0.):
    """premultiplied sprite over premultiplied/RGB dst, centre (or anchor px) at x,y; in place"""
    h, w = spr.shape[:2]; ax, ay = (w / 2., h / 2.) if anchor is None else anchor; c, s = math.cos(rot) * scale, math.sin(rot) * scale; fx = -1. if flip else 1.
    M = np.array([[c * fx, -s, 0], [s * fx, c, 0]], float); cor = np.array([(0, 0), (w, 0), (w, h), (0, h)], float) - [ax, ay]; q = cor @ M[:, :2].T + [x, y]; x0, y0 = np.floor(q.min(0)).astype(int) - 1; x1, y1 = np.ceil(q.max(0)).astype(int) + 2
    X0, Y0, X1, Y1 = max(x0, 0), max(y0, 0), min(x1, dst.shape[1]), min(y1, dst.shape[0])
    if X1 <= X0 or Y1 <= Y0: return dst
    M[:, 2] = np.array([x, y]) - M[:, :2] @ [ax, ay] - [X0, Y0]; sc = abs(scale); fl = cv2.INTER_AREA if sc < .6 else cv2.INTER_LINEAR
    p = cv2.warpAffine(spr, np.float32(M), (X1 - X0, Y1 - Y0), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if ta and tintc is not None: p = p.copy(); p[..., :3] = p[..., :3] * (1 - ta) + rgb(tintc) * p[..., 3:4] * ta
    if op != 1.: p = p * op
    d = dst[Y0:Y1, X0:X1]; n = dst.shape[2]; dst[Y0:Y1, X0:X1] = p[..., :n] + d * (1 - p[..., 3:4]); return dst


def mips(spr, n=4):
    out = [spr]
    for _ in range(n - 1): s = out[-1]; out.append(cv2.resize(s, (max(2, s.shape[1] // 2), max(2, s.shape[0] // 2)), interpolation=cv2.INTER_AREA))
    return out


def putm(dst, ms, x, y, scale=1., rot=0., flip=False, op=1., tintc=None, ta=0., anchor=None):
    """put() from a mip list"""
    k = 0
    while scale < .5 and k < len(ms) - 1: scale *= 2; k += 1
    a = None if anchor is None else (anchor[0] / 2 ** k, anchor[1] / 2 ** k)
    return put(dst, ms[k], x, y, scale, rot, flip, op, a, tintc, ta)


def card(h, w, col, dark, light, seed=1, ang=-.35):
    """flat colour card with gouache brush texture up to the edges"""
    from op.paint import Canvas, arcs, mottle
    cv = Canvas(h, w, color=col); r = rs(seed, 3)
    for hw, n, a in ((70, 260, .10), (34, 520, .12), (14, 500, .16)):
        x, y = r.uniform(-50, w + 50, n), r.uniform(-50, h + 50, n); t = r.random(n)[:, None]; c = np.where(t < .5, mix(rgb(col), rgb(dark), t * 1.1), mix(rgb(col), rgb(light), (t - .5) * 1.1)).astype(f32)
        cv.strokes(arcs(x, y, ang + r.normal(0, .18, n), hw * r.uniform(5, 11, n), r.normal(0, .25, n)), hw * r.uniform(.6, 1.2, n), c, op=r.uniform(.25, .7, n), dry=r.uniform(.1, .7, n), seed=seed + hw, bristle=.06)
    return mottle(cv.img, .012, seed)


# ================================================================= shared expressions, bust, assembly
EXPR = dict(
    calm=dict(o=.92, M='calm'), smile=dict(o=.94, M='smile', sq=.08), smirk=dict(o=.84, M='smirk', bo=.03, bi=-.015, sq=.1),
    grin=dict(o=.90, sq=.22, M='grin', teeth=(.075, .0), bi=-.02, bo=.03, blush=.25),
    worry=dict(o=1.10, ir=.86, bi=.10, bo=-.035, M='oh', bk=.15), hope=dict(o=1.08, ir=.94, bi=.08, bo=-.02, M='oh', wob=.6),
    laugh=dict(o=0., shut='tight', M='laugh', teeth=(.075, 0.), tongue=.52, bi=.05, bo=.045, blush=.55),
    chuckle=dict(o=0., shut='tight', M='grin', teeth=(.06, 0.), tongue=.3, msx=.72, msy=.78, bi=.03, bo=.03, blush=.4),
    cry=dict(o=.0, shut='tight', M='clench', clench=True, bi=.10, bo=-.06, bk=.6, rope=.07, fore=.6, blush=.5),
    howl=dict(o=.0, shut='tight', M='howl', teeth=(.075, .06), tongue=.5, bi=.13, bo=-.06, bk=.7, rope=.085, fore=1., blush=.6, flare=.6))


def bust(cel, cx, cy, s, roll, st, kind='tee', by=0., lw=4., light=-1, sh=0., length=3.2, f=0, seed=0, breath=0., wide=1., torso=False):
    """neck + shoulders in head units hung under a head at (cx,cy).  by = body yaw (rad), sh = shoulder lift, kinds: tee (striped, crew neck), jacket (high collar), coat (fur collar)"""
    cb, sb = math.cos(by), math.sin(by)
    def Q(P, z=.0):
        P = np.array(P, float); zz = z * np.clip(1 - np.abs(P[:, 0]) / 1.3, 0, 1); x = P[:, 0] * wide * .86 * cb + zz * sb; y = P[:, 1].copy(); k = sm((np.abs(P[:, 0]) - .35) / .7); y = y - sh * k - breath * .03 * sm((y - 1.2) / .6); return loc(cx, cy, s, roll, np.c_[x, y])
    sg = 1 if light < 0 else -1; c = st['cloth']
    nk = Q([(-.31, .55), (-.30, .95), (-.34, 1.14), (-.56, 1.32), (0, 1.75), (.56, 1.32), (.34, 1.14), (.30, .95), (.31, .55)], .12); cel.fill(nk, st['skin'])
    cel.clip(nk); cel.fill(Q([(-.5, .5), (-.36, 1.02), (-.10, 1.20), (.16, 1.24), (.40, 1.10), (.6, .5)], .12), st['shade']); cel.fill(Q(spl([(sg * .10, 1.18), (sg * .17, 1.30), (sg * .36, 1.36), (sg * .7, 1.5), (sg * .7, .6), (sg * .22, .6)], True, n=5), .12), st['shade']); cel.unclip()
    for x in (-1, 1): cel.line(Q(spl([(x * .30, .70), (x * .295, .95), (x * .335, 1.13), (x * .52, 1.29)], n=8), .12), lw * .9, INK, taper=(.2, .1), seed=seed + x)
    if kind == 'tee':
        for x in (-1, 1): cel.line(Q(spl([(x * .20, 1.02), (x * .12, 1.24), (x * .06, 1.40)], n=6), .14), lw * .42, INK, taper=(.5, .3))
        nl = spl([(-.56, 1.265), (-.36, 1.46), (0, 1.56), (.36, 1.46), (.56, 1.265)], n=8)
        B = np.vstack([nl, spl([(.56, 1.265), (.92, 1.37), (1.24, 1.52), (1.48, 1.86), (1.62, 2.4), (1.60, length)], n=8), spl([(-1.60, length), (-1.62, 2.4), (-1.48, 1.86), (-1.24, 1.52), (-.92, 1.37), (-.56, 1.265)], n=8)])
        BP = Q(B, .28); cel.fill(BP, c['base']); cel.clip(BP)
        for k in range(9):
            y = 1.60 + k * .27; xs = np.linspace(-1.9, 1.9, 40); sag = .16 * np.cos(np.clip(xs, -1.5, 1.5) * 1.05) + .05 * np.sin(xs * 2.3 + k) + .09 * np.sin(xs * 5 + k * 2) * np.exp(-((np.abs(xs) - .95) / .22) ** 2); cel.fill(Q(np.vstack([np.c_[xs, y - .12 + sag], np.c_[xs, y + sag + .012 * np.sin(xs * 9)][::-1]]), .28), c['stripe'])
        S = spl([(sg * .18, 1.56), (sg * .40, 1.78), (sg * .62, 1.92), (sg * .70, 2.3), (sg * .58, 2.8), (sg * .55, 3.6)], n=8); cel.fill(Q(np.vstack([S, [(sg * 2.4, 3.6), (sg * 2.4, 1.2), (sg * .8, 1.2)]]), .28), c['shade'], c.get('sa', .62)); cel.unclip()
        cel.fill(Q(np.vstack([nl, (nl + [0, .075])[::-1]]), .28), c['stripe']); cel.line(Q(nl + [0, .078], .28), lw * .6, INK, taper=(.1, .1))
        cel.line(Q(nl, .28), lw * .9, INK, taper=(.1, .1), seed=seed)
        for x in (-1, 1): cel.line(Q(spl([(x * .56, 1.265), (x * .92, 1.37), (x * 1.24, 1.52), (x * 1.48, 1.86), (x * 1.62, 2.4), (x * 1.60, length)], n=8), .28), lw * 1.05, INK, taper=(.1, .1), seed=seed + 5); cel.line(Q(spl([(x * 1.16, 1.50), (x * 1.06, 1.78), (x * 1.10, 2.1), (x * 1.16, 2.45)], n=8), .28), lw * .62, INK, taper=(.1, .5)); fold(cel, Q([(x * 1.10, 1.84), (x * .84, 1.90), (x * .62, 1.86)], .28), lw * .55); fold(cel, Q([(x * 1.12, 2.02), (x * .90, 2.16), (x * .74, 2.2)], .28), lw * .5); fold(cel, Q([(x * 1.2, 1.62), (x * 1.34, 1.78), (x * 1.38, 2.0)], .28), lw * .5)
        fold(cel, Q([(.10, 1.66), (.22, 1.80), (.40, 1.88)], .28), lw * .45); fold(cel, Q([(-.14, 1.68), (-.30, 1.84), (-.46, 1.9)], .28), lw * .45)
    else:
        col = kind == 'jacket'
        OL = [(.44, 1.10), (.60, 1.26), (.94, 1.36), (1.28, 1.54), (1.54, 1.95), (1.70, 2.5), (1.72, length)] if not torso else [(.44, 1.10), (.60, 1.26), (.94, 1.36), (1.20, 1.50), (1.28, 1.75), (1.08, 2.05), (.92, 2.6), (1.0, length)]
        B = np.vstack([spl([(-a, b) for a, b in OL], n=8), spl(OL[::-1], n=8)]); BP = Q(B, .28); cel.fill(BP, c['base']); cel.clip(BP)
        S = spl([(sg * .05, 1.30), (sg * .22, 1.60), (sg * .52, 1.86), (sg * .60, 2.3), (sg * .46, 2.9), (sg * .44, 3.6)], n=8); cel.fill(Q(np.vstack([S, [(sg * 2.4, 3.6), (sg * 2.4, 1.1), (sg * .5, 1.1)]]), .28), c['shade'])
        for x in (-1, 1): cel.fill(Q(spl([(x * 1.12, 1.62), (x * 1.0, 1.90), (x * 1.08, 2.5), (x * 1.16, 3.4), (x * 1.0, 2.6), (x * .92, 2.0)], True, n=6), .28), c['shade'])
        L = spl([(-sg * .70, 1.36), (-sg * 1.06, 1.50), (-sg * 1.30, 1.74), (-sg * 1.14, 1.64), (-sg * .86, 1.48)], True, n=6); cel.fill(Q(L, .28), c['light']); cel.unclip()
        for x in (-1, 1): cel.line(Q(spl([(x * a, b) for a, b in OL[1:]], n=8), .28), lw * 1.1, INK, taper=(.1, .1), seed=seed + 5); cel.line(Q(spl([(x * 1.16, 1.52), (x * 1.04, 1.80), (x * 1.08, 2.3), (x * 1.14, 3.0)], n=8), .28), lw * .7, c['line'], taper=(.1, .4)); fold(cel, Q([(x * 1.08, 1.86), (x * .82, 1.94), (x * .58, 1.90)], .28), lw * .55, c['line']); fold(cel, Q([(x * 1.10, 2.10), (x * .86, 2.24), (x * .62, 2.26)], .28), lw * .5, c['line']); (None if torso else fold(cel, Q([(x * 1.2, 1.62), (x * 1.36, 1.82), (x * 1.42, 2.1)], .28), lw * .5, c['line']))
        if col:
            for x in (-1, 1):
                K = Q(spl([(x * .05, 1.60), (x * .10, 1.40), (x * .30, 1.22), (x * .40, 1.02), (x * .50, .98), (x * .62, 1.22), (x * .56, 1.44), (x * .34, 1.62), (x * .12, 1.74)], True, n=6), .30); cel.fill(K, c['base'] if x * sg < 0 else c['shade']); cel.clip(K); cel.fill(Q(spl([(x * .40, 1.02), (x * .52, .98), (x * .60, 1.20), (x * .50, 1.10)], True, n=5), .30), c['light']); cel.unclip(); outline(cel, K, lw * .95, INK, seed + x + 3)
            cel.line(Q([(0, 1.62), (-.01, 2.2), (0, length)], .30), lw * .8, c['line'], taper=(.05, .05)); cel.line(Q([(.035, 1.66), (.03, 2.2), (.035, length)], .30), lw * .45, c['zip'], taper=(.05, .05))
            z = Q([(-.05, 1.72), (.06, 1.72), (.05, 1.90), (-.04, 1.90)], .32); cel.fill(z, c['zip']); outline(cel, z, lw * .5, INK)
    return Q


def stack(*Ls):
    o = Ls[0]
    for L in Ls[1:]:
        if L is not None: o = lay(o, L)
    return o
