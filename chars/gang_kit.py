"""Gang kit (GANG lane): 2.5-D head rig, two-plane inking (flats under, pen on top), motivated light, eyes, mouths, tears, curls.
Local face units: x right, y DOWN, z to viewer; eye line y=0, chin y=1."""
import numpy as np, math, cv2, cairo
from op import W, H
from op.paint import rgb, f32, smooth, fbm, grid, mix, ramp, Canvas
from op.ink import Cel, resample, ribbon, noise1
from op import kit
from op.comp import blur

INK = '#17131a'


# ---------------------------------------------------------------- splines (N-D)
def _crn(P, n, closed):
    m = len(P)
    if m < 3 and not closed:
        t = np.linspace(0, 1, max(2, n))[:, None]; return P[0] * (1 - t) + P[-1] * t
    out = []
    for i in range(m if closed else m - 1):
        if closed: p0, p1, p2, p3 = P[(i - 1) % m], P[i], P[(i + 1) % m], P[(i + 2) % m]
        else:
            p1, p2 = P[i], P[i + 1]; p0 = P[i - 1] if i > 0 else 2 * p1 - p2; p3 = P[i + 2] if i + 2 < m else 2 * p2 - p1
        d1 = max(np.linalg.norm(p1 - p0), 1e-5) ** .5; d2 = max(np.linalg.norm(p2 - p1), 1e-5) ** .5; d3 = max(np.linalg.norm(p3 - p2), 1e-5) ** .5
        m1 = (p2 - p1) + d2 * ((p1 - p0) / d1 - (p2 - p0) / (d1 + d2)); m2 = (p2 - p1) + d2 * ((p3 - p2) / d3 - (p3 - p1) / (d2 + d3))
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append((2 * t ** 3 - 3 * t ** 2 + 1) * p1 + (t ** 3 - 2 * t ** 2 + t) * m1 + (-2 * t ** 3 + 3 * t ** 2) * p2 + (t ** 3 - t ** 2) * m2)
    out = np.concatenate(out)
    return out if closed else np.vstack([out, P[-1]])


def spn(P, closed=False, corners=(), n=8):
    P = np.asarray(P, float); c = sorted(set(int(i) % len(P) for i in corners))
    if not c: return _crn(P, n, closed)
    out = []
    if closed:
        for a, b in zip(c, c[1:] + [c[0] + len(P)]): out.append(_crn(P[[k % len(P) for k in range(a, b + 1)]], n, False)[:-1])
        return np.concatenate(out)
    cs = [0] + [k for k in c if 0 < k < len(P) - 1] + [len(P) - 1]
    for a, b in zip(cs[:-1], cs[1:]): out.append(_crn(P[a:b + 1], n, False)[:-1])
    return np.vstack(out + [P[-1:]])


def mir(P): return [(-p[0],) + tuple(p[1:]) for p in P][::-1]
def sym(P): return list(P) + mir([p for p in P if abs(p[0]) > 1e-6])
def off(P, dx=0, dy=0, k=1.): return [(p[0] * k + dx, p[1] * k + dy) + tuple(p[2:]) for p in P]
def ell(cx, cy, rx, ry, n=24, a0=0., a1=2 * math.pi, rot=0.):
    t = np.linspace(a0, a1, n, endpoint=abs(a1 - a0) < 6.28); x, y = rx * np.cos(t), ry * np.sin(t); c, s = math.cos(rot), math.sin(rot)
    return np.c_[cx + x * c - y * s, cy + x * s + y * c]


def band_poly(P, w, n=8, closed=False, corners=()):
    """ribbon polygon round a spine; w = scalar, list at control points or fn(t)"""
    P = np.asarray(P, float); C = resample(spn(P, False, corners, n), 2.) if len(P) > 2 else resample(P, 2.); t = np.linspace(0, 1, len(C))
    ww = w(t) if callable(w) else (np.full(len(C), float(w)) if np.isscalar(w) else np.interp(t, np.linspace(0, 1, len(w)), w))
    T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    return np.vstack([C + N * ww[:, None] / 2, (C - N * ww[:, None] / 2)[::-1]])


def scallop(P, pitch, amp, seed=0, closed=True, jit=.35):
    """curly outline: outward bumps with cusps along a screen path"""
    P = np.asarray(P, float); Q = resample(np.vstack([P, P[:1]]) if closed else P, 1.6)
    A = .5 * np.sum(Q[:-1, 0] * Q[1:, 1] - Q[1:, 0] * Q[:-1, 1]); sg = 1. if A > 0 else -1.
    T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]] * sg
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; r = np.random.default_rng(seed); k = max(3, int(L[-1] / pitch)); e = np.cumsum(r.uniform(1 - jit, 1 + jit, k)); e = np.r_[0, e / e[-1]] * L[-1]
    i = np.clip(np.searchsorted(e, L, side='right') - 1, 0, k - 1); u = (L - e[i]) / (e[i + 1] - e[i] + 1e-9); a = r.uniform(.65, 1.25, k)
    d = amp * a[i] * np.sin(np.pi * np.clip(u, 0, 1)) ** .55 - amp * .15
    return Q + N * d[:, None]


# ---------------------------------------------------------------- rig
class Rig:
    """yaw > 0 turns to the viewer's right, pitch > 0 looks up, roll > 0 clockwise on screen"""
    def __init__(s, cx, cy, sc, yaw=0., pitch=0., roll=0., persp=.10, r=(.74, 1.08, .78), o=(0., 0.), sx=1., sy=1.):
        s.c = np.array([cx, cy], float); s.s = sc; s.r = r; s.o = o; s.sx, s.sy = sx, sy; s.k = persp; s.yaw, s.pitch, s.roll = yaw, pitch, roll
        cy_, sy_ = math.cos(yaw), math.sin(yaw); cp, sp = math.cos(pitch), math.sin(pitch); cr, sr = math.cos(roll), math.sin(roll)
        s.R = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy_, 0, sy_], [0, 1, 0], [-sy_, 0, cy_]])
        s.S = np.array([[cr, -sr], [sr, cr]])

    def z(s, x, y):
        u = np.clip(((x - s.o[0]) / s.r[0]) ** 2 + ((y - s.o[1]) / s.r[1]) ** 2, 0, 1); return s.r[2] * np.sqrt(1 - u)

    def P3(s, Q):
        Q = np.asarray(Q, float) * [s.sx, s.sy, 1.]; q = Q @ s.R.T; f = 1 / (1 - np.clip(q[..., 2] * s.k, -.6, .6))
        return np.stack([q[..., 0] * f * s.s, q[..., 1] * f * s.s], -1) + s.c

    def __call__(s, P, z=None, dz=0.):
        P = np.asarray(P, float)
        if P.shape[-1] == 3: return s.P3(P)
        zz = s.z(P[..., 0], P[..., 1]) + dz if z is None else np.broadcast_to(np.asarray(z, float), P.shape[:-1])
        return s.P3(np.concatenate([P, zz[..., None]], -1))

    def limb(s, P, chin=.42):
        """outer contour: keeps its width when the head turns, the chin swings forward"""
        P = np.asarray(P, float); c = math.cos(s.yaw); k = math.sqrt(c * c + (s.r[2] / s.r[0]) ** 2 * math.sin(s.yaw) ** 2) / max(c, .2)
        return s.P3(np.c_[P[:, 0] * k, P[:, 1], chin * smooth(.15, 1.0, P[:, 1])])

    def at(s, x, y, z=0.): return s.P3(np.array([[x, y, z]]))[0]


# ---------------------------------------------------------------- drawing
class Part:
    def __init__(s, T, box=None, seed=0, k=None, ink=INK, ss=2, pad=0):
        b = box or (0, 0, W, H); x0, y0, x1, y1 = max(0, int(b[0] - pad)), max(0, int(b[1] - pad)), min(W, int(b[2] + pad)), min(H, int(b[3] + pad)); s.ok = x1 - x0 > 3 and y1 - y0 > 3
        if not s.ok: x0, y0, x1, y1 = 0, 0, 4, 4
        s.T, s.sd, s.ink = T, seed, ink; s.k = k if k is not None else (max(1.6, min(6.6, T.s * .027)) if T is not None else 4.); s.f = Cel(x1 - x0, y1 - y0, x0, y0, ss); s.l = Cel(x1 - x0, y1 - y0, x0, y0, ss)

    def q(s, P, z=None, dz=0., cl=True, co=(), n=8, limb=False):
        if isinstance(P, np.ndarray): return P
        C = spn(P, cl, co, n)
        return s.T.limb(C) if limb else s.T(C, z, dz)

    def nx(s): s.sd += 1; return s.sd

    def fill(s, P, col, a=1., **q):
        P = [s.q(p, **q) for p in P] if (isinstance(P, list) and len(P) and isinstance(P[0], np.ndarray)) else s.q(P, **q)
        s.f.fill(P, col, a)
        if a >= 1:
            c = s.l.ctx; c.set_operator(cairo.OPERATOR_DEST_OUT); s.l._path(P); c.set_source_rgba(0, 0, 0, 1); c.set_fill_rule(cairo.FILL_RULE_WINDING); c.fill(); c.set_operator(cairo.OPERATOR_OVER)
        return P

    def ol(s, P, w=1., col=None, closed=True, **o):
        kit.outline(s.l, P, w * s.k, col or s.ink, s.nx(), closed, **o); return P

    def shape(s, P, col, w=1., lc=None, o=None, **q):
        P = s.fill(P, col, **q)
        if w > 0:
            for p in (P if isinstance(P, list) else [P]): s.ol(p, w, lc, **(o or {}))
        return P

    def shade(s, P, col, within=None, a=1., **q): s.f.shade(s.q(P, **q), col, within, a); return s

    def line(s, P, w=.5, col=None, taper=(.25, .35), z=None, dz=0., co=(), n=8, a=1., press=.25, wfun=None):
        P = s.q(P, z, dz, False, co, n); s.l.line(P, w * s.k, col or s.ink, taper=taper, seed=s.nx(), press=press, alpha=a, wfun=wfun); return s

    def hl(s, P, col, a=1., within=None, **q): s.f.shade(s.q(P, **q), col, within, a); return s

    def hatch(s, P, ang, gap=None, w=.28, n=0, col=None, length=None, fade=.5, within=None, a=1., **q):
        Q = s.q(P, **q); g = gap or max(4.5, s.k * 1.7)
        if within is not None: s.l.clip(within)
        kit.hatch(s.l, Q, ang, g, max(.9, w * s.k), col or s.ink, s.nx(), length, fade, alpha=a)
        if within is not None: s.l.unclip()
        return s

    def ticks(s, A, B, n=6, w=.28, ln=.10, col=None, jit=.3, a=1.):
        """cluster of short parallel strokes from local A to local B (Oda cheek hatching); ln = local stroke vector or length"""
        r = np.random.default_rng(s.nx()); A, B = np.asarray(A, float), np.asarray(B, float); d = np.asarray(ln, float) if np.ndim(ln) else np.array([-.45, .9]) * ln
        for i in range(n):
            t = (i + .5 + r.normal(0, jit * .4)) / n; p = A + (B - A) * t; k = r.uniform(.7, 1.15) * (1 - .35 * abs(t - .5) * 2)
            s.line([tuple(p - d * k / 2), tuple(p + d * k / 2 + r.normal(0, .004, 2))], w, col, (.15, .45), n=6, a=a)
        return s

    def done(s, li=None, rw=None):
        a = s.f.rgba(); b = s.l.rgba()
        if li: a = relight(a, li, s.f.ox, s.f.oy, rw if rw is not None else (max(3.5, min(12., s.T.s * .042)) if s.T is not None else 7.))
        if li and li.get('ink') is not None: b = kit.tint(b, li['ink'], .75, 'over')
        return a * (1 - b[..., 3:4]) + b, s.f.ox, s.f.oy


def relight(a, li, ox=0, oy=0, rw=8.):
    """flats -> graded, airbrushed, hot-rimmed.  li: mul, rim=[(ang, col, amt, widthmul)], low=(hex, x0,y0,x1,y1), air=(ang,col,amt,r)"""
    out = a.copy(); al = a[..., 3]
    if li.get('mul') is not None: out[..., :3] *= np.asarray(rgb(li['mul']) if isinstance(li['mul'], str) else li['mul'], f32)
    if li.get('low') is not None:
        col, x0, y0, x1, y1 = li['low']; h, w = al.shape; x, y = grid(h, w); d = np.array([x1 - x0, y1 - y0], float); t = np.clip(((x + ox - x0) * d[0] + (y + oy - y0) * d[1]) / (d @ d), 0, 1)[..., None]
        out[..., :3] *= 1 + (rgb(col) - 1) * t
    if li.get('air') is not None:
        ang, col, amt, r = li['air']; m = np.clip(al - blur(kit.shifted(al, -math.cos(ang) * r, -math.sin(ang) * r), r * .6), 0, 1); m = blur(m, r * .25) * al
        c = rgb(col); out[..., :3] = out[..., :3] + (al[..., None] * c - out[..., :3]) * (m * amt)[..., None]
    for ang, col, amt, wm in li.get('rim', ()):
        w = rw * wm; dx, dy = -math.cos(ang) * w, -math.sin(ang) * w
        m = np.clip(al - kit.shifted(al, dx, dy), 0, 1); soft = blur(np.clip(al - kit.shifted(al, dx * 2.6, dy * 2.6), 0, 1), w * .9) * al; c = rgb(col)
        out[..., :3] = out[..., :3] + (al[..., None] * c - out[..., :3]) * (soft * amt * .42)[..., None]
        m = cv2.GaussianBlur(m, (0, 0), .7) * (al > .02); out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * al)[..., None] * amt
    return out


class Stack:
    """full-frame premultiplied RGBA built from parts"""
    def __init__(s): s.a = np.zeros((H, W, 4), f32)
    def add(s, p, li=None, rw=None):
        if isinstance(p, Part):
            if not p.ok: return s
            src, ox, oy = p.done(li, rw)
        else: src, ox, oy = p
        h, w = src.shape[:2]; x0, y0, x1, y1 = max(0, ox), max(0, oy), min(W, ox + w), min(H, oy + h)
        if x1 <= x0 or y1 <= y0: return s
        S = src[y0 - oy:y1 - oy, x0 - ox:x1 - ox]; b = s.a[y0:y1, x0:x1]; s.a[y0:y1, x0:x1] = b * (1 - S[..., 3:4]) + S; return s


def over4(b, l): return b * (1 - l[..., 3:4]) + l


# ---------------------------------------------------------------- light presets (ang = where the light comes FROM, screen radians, y down)
UL, UR, L_, R_, UP = -2.36, -.78, math.pi, 0., -math.pi / 2
LIGHTS = dict(
    day=dict(rim=[(UL, '#fff6d8', .95, 1.)], air=(UL, '#ffe9b8', .22, 60)),
    sun=dict(rim=[(UR, '#fffbe6', 1., 1.55), (math.pi * 1.04, '#ff9f3a', .55, .7)], air=(UR, '#ffe7a6', .42, 80)),
    moon=dict(mul=(.60, .66, .98), rim=[(UR, '#eef6ff', 1., 1.1), (L_, '#4f6fd0', .45, .8)], air=(UR, '#9fc0ff', .25, 60)),
    pink=dict(mul=(.86, .70, .86), rim=[(UR, '#fff0f4', 1., 1.35), (L_, '#8f6ae0', .5, .8)], air=(UR, '#ffb3cf', .30, 60)),
    weld=dict(mul=(.94, .91, 1.), rim=[(UP, '#f2ffff', 1., 1.4), (math.pi * 1.04, '#ff8a3a', .85, 1.)], air=(UP, '#9ff0ff', .25, 60)),
    storm=dict(mul=(.80, .84, .98), rim=[(UL, '#f4fdff', 1., 1.5), (R_, '#58c4e8', .6, .9)], air=(UL, '#a8dcff', .34, 80)),
    back=dict(mul=(.90, .88, .95), rim=[(UP, '#fff8dc', 1., 1.15)], air=(UP, '#ffe2a0', .30, 60)))


def light(li, ang=None, **kw):
    d = dict(LIGHTS[li] if isinstance(li, str) else li)
    if ang is not None:
        a0 = d['rim'][0][0]; d['rim'] = [(r[0] + ang - a0,) + tuple(r[1:]) for r in d['rim']]
        if d.get('air'): d['air'] = (ang,) + tuple(d['air'][1:])
    d.update(kw); return d


# ---------------------------------------------------------------- features
def eye(p, x, y, rx, ry, e, side=1, skin='#000', lid='#000', lash=0, white='#fdfbf4', shw='#b9b4cc', lidw=1., pcol=INK, big=False):
    """e: eo open, el lower lid push, et tilt (inner end down +), lx ly look, pr pupil, arc for closed (+1 happy, -1 sad)"""
    eo = e['eo']
    if eo < .12:
        k = e.get('arc', 1); u = np.linspace(-1, 1, 9); h = ry * .55
        C = [(x + rx * 1.05 * a, y + (ry * .10 if k > 0 else -ry * .05) - k * h * (1 - a * a) + k * h * .35 + e['et'] * a * side * ry * .3) for a in u]
        p.line(C, lidw * 1.15, taper=(.15, .15))
        if lash:
            for a, l in ((.55, .30), (.85, .36), (1.05, .40))[:lash]:
                xx = x + rx * a * side; yy = y + ry * .10 - k * h * (1 - a * a) + k * h * .35; p.line([(xx, yy), (xx + rx * l * .9 * side, yy - k * ry * l)], .5, taper=(.02, .6))
        return None
    u = np.linspace(-1, 1, 25); w = np.sqrt(np.maximum(0, 1 - np.abs(u) ** 2.2)); lidy = -(2 * eo - 1) + e['et'] * -u * side; top = np.maximum(-w, lidy); bot = np.minimum(w, 1 - 2 * e.get('el', 0) + .12 * u * u); top = np.minimum(top, bot)
    full = np.r_[np.c_[x + rx * u, y - ry * w], np.c_[x + rx * u, y + ry * w][::-1]]; S = np.r_[np.c_[x + rx * u, y + ry * top], np.c_[x + rx * u, y + ry * bot][::-1]]
    QF = p.T(full); p.fill(QF, lid); Q = p.T(S); p.fill(Q, white)
    p.shade(p.T(np.r_[np.c_[x + rx * u, y + ry * top], np.c_[x + rx * u, y + ry * (top + .34 * (1 - u * u) + .10)][::-1]]), shw, Q)
    pr = e['pr']; px, py = x + e['lx'] * rx * .55, y + e['ly'] * ry * .55
    P = p.T(ell(px, py, rx * pr, rx * pr * (1.12 if not big else 1.2), 20)); p.shade(P, pcol, Q)
    if big: p.shade(p.T(ell(px - rx * pr * .30, py - rx * pr * .42, rx * pr * .34, rx * pr * .38, 14)), '#ffffff', Q); p.shade(p.T(ell(px + rx * pr * .36, py + rx * pr * .42, rx * pr * .16, rx * pr * .16, 12)), '#ffffff', Q)
    p.line([(x + rx * a, y + ry * t) for a, t in zip(u[::2], top[::2])], lidw, taper=(.22, .18), wfun=lambda t: .65 + .6 * np.sin(np.pi * t) ** .6)
    p.line([(x + rx * a, y + ry * b) for a, b in zip(u[3:-3:2], bot[3:-3:2])], .38, taper=(.4, .4))
    if eo < .92: p.line([(x + rx * a, y - ry * b) for a, b in zip(u[2:-2:2], w[2:-2:2])], .30, taper=(.4, .4))
    if lash:
        for a, l in ((.35, .34), (.66, .42), (.92, .46))[:lash]:
            j = int((a * side + 1) * 12); p.line([(x + rx * a * side, y + ry * top[j]), (x + rx * (a + l * .55) * side, y + ry * (top[j] - l))], .55, taper=(.02, .6))
    return Q


def brow(p, P, w, col, shd=None, lw=.55):
    Q = p.T(spn(P, False, (), 10)); B = band_poly(Q, [v * p.T.s for v in w]); p.shape(B, col, lw)
    if shd: p.shade(B + [p.T.s * .012, p.T.s * .03], shd, B)
    return B


def mouth(p, x, y, e, lipc=None, inn='#3e1019', gum='#b0414a', tong='#d9696a', teeth='#f7f3e6', tsh='#b9aebb', lines=0, w=1., seg=True):
    """e: mw half width, mu up height, md down height, ml corner lift, mt upper tooth band 0..1, mb lower band, tg tongue, wob, ms skew"""
    mw, mu, md, ml = e['mw'], e['mu'], e['md'], e['ml']; sk = e.get('ms', 0.); wob = e.get('wob', 0.); lift = lambda a: -ml * np.abs(a) ** 1.5
    u = np.linspace(-1, 1, 21); wv = wob * np.sin(u * 8.5 + 1.0) * .05
    top = y + lift(u) - mu * (1 - np.abs(u) ** 2.4) + wv + sk * u; bot = y + lift(u) + md * (1 - np.abs(u) ** 2.2) ** .8 + wv * .6 + sk * u
    if md + mu < .035:
        p.line([(x + mw * a, t) for a, t in zip(u, (top + bot) / 2)], .85, taper=(.15, .15)); s_ = .05
        p.line([(x - mw * 1.02, y - ml + sk * -1 - s_), (x - mw * 1.04, y - ml - sk + s_ * .6)], .4); p.line([(x + mw * 1.02, y - ml + sk - s_), (x + mw * 1.04, y - ml + sk + s_ * .6)], .4); return None
    P = np.r_[np.c_[x + mw * u, top], np.c_[x + mw * u, bot][::-1]]; Q = p.T(P); p.fill(Q, inn)
    if e['tg'] > 0: hgt = (mu + md) * .55 * e['tg']; p.shade(p.T(ell(x, y + md * 1.0 - .02, mw * .62, hgt, 24)), tong, Q); p.f.line(p.T(ell(x, y + md * 1.0 - .02, mw * .62, hgt, 16, math.pi * 1.1, math.pi * 1.9)), .4 * p.k, gum); p.f.ctx.save(); p.f.ctx.restore()
    H_ = mu + md
    if e['mt'] > 0:
        b = np.minimum(top + H_ * e['mt'] * (1 - .25 * u * u), bot); B = p.T(np.r_[np.c_[x + mw * u, top], np.c_[x + mw * u, b][::-1]]); p.shade(B, teeth, Q); p.shade(p.T(np.r_[np.c_[x + mw * u, top], np.c_[x + mw * u, top + H_ * .07][::-1]]), tsh, Q)
        if e['mt'] < .97: p.line([(x + mw * a, v) for a, v in zip(u[1:-1:2], b[1:-1:2])], .38)
    if e['mb'] > 0:
        b = np.maximum(bot - H_ * e['mb'] * (1 - .25 * u * u), top); p.shade(p.T(np.r_[np.c_[x + mw * u, b], np.c_[x + mw * u, bot][::-1]]), teeth, Q); p.line([(x + mw * a, v) for a, v in zip(u[2:-2:2], b[2:-2:2])], .34)
    if lines:
        for a in np.linspace(-.86, .86, lines):
            j = int((a + 1) * 10); t0 = top[j]; b0 = top[j] + (bot[j] - top[j]) * (e['mt'] if e['mt'] < .97 else 1.); p.line([(x + mw * a, t0 + .01), (x + mw * a, b0 - .01)], .26, a=.8)
        if e['mt'] >= .97: m = (top + bot) / 2 + .01; p.line([(x + mw * a, v) for a, v in zip(u[1:-1:2], m[1:-1:2])], .42)
    p.ol(Q, w * .85, corners=())
    return Q


def rope(p, P, w, z=None, dz=.04, col='#dff3fb', sh='#8fc3de', screen=False, lw=.5, tip=True):
    """fat drawn tear: P local spine (or screen when screen=True), w px widths"""
    Q = np.asarray(P, float) if screen else p.T(spn(P, False, (), 10), z, dz)
    wk = np.full(2, float(w)) if np.isscalar(w) else np.asarray(w, float); L_ = np.hypot(*np.diff(Q, axis=0).T).sum() + 1e-6; e0, e1 = min(.45, wk[0] * .9 / L_), min(.45, wk[-1] * .52 / L_)
    wf = lambda t: np.interp(t, np.linspace(0, 1, len(wk)), wk) * np.sqrt(np.clip(1 - (1 - np.clip(t / e0, 0, 1)) ** 2, 0, 1)) * np.sqrt(np.clip(1 - (1 - np.clip((1 - t) / e1, 0, 1)) ** 2, 0, 1)) + .4
    B = resample(np.vstack([band_poly(resample(Q, 4.), wf)]), 1.5); p.shape(B, col, lw, lc='#3d6f93'); p.shade(B + np.array([np.max(w) * .22, 0]), sh, B); p.f.line(Q - [np.max(w) * .16, 0], max(1.2, np.max(w) * .16), '#ffffff', taper=(.3, .5)); return B


def bead(p, x, y, r, ang=math.pi / 2, st=1.5, col='#dff3fb', sh='#8fc3de', lw=.45):
    """tear drop flying along ang (screen px)"""
    a = np.linspace(0, 2 * math.pi, 22, endpoint=False); c, s = math.cos(ang), math.sin(ang); X = np.cos(a) * r * np.where(np.cos(a) < 0, st * (1.1 - .1 * np.abs(np.sin(a))), 1.); Y = np.sin(a) * r * np.where(np.cos(a) < 0, np.clip(1 + np.cos(a) * .72, .1, 1), 1.)
    Q = np.c_[x + X * c - Y * s, y + X * s + Y * c]; p.shape(Q, col, lw, lc='#3d6f93'); p.shade(Q + [r * .35, r * .3], sh, Q); p.f.fill(ell(x - r * .25, y - r * .3, r * .28, r * .22, 12), '#ffffff'); return Q


def curls(p, P, n, r, col, seed=0, z=0., w=.4, a=1., turns=1.2):
    """spiral hair strokes inside a local polygon (positions fixed in local space)"""
    g = np.random.default_rng(seed); C = spn(P, True, (), 6)[:, :2].astype(np.float32); lo, hi = C.min(0), C.max(0); k = 0
    for _ in range(n * 6):
        c = g.uniform(lo, hi)
        if cv2.pointPolygonTest(C, (float(c[0]), float(c[1])), True) < r * .5: continue
        t0 = g.uniform(0, 6.28); sg = 1 if g.random() < .5 else -1; t = np.linspace(0, 6.28 * turns, 14); rr = r * g.uniform(.6, 1.15) * np.linspace(1, .25, 14)
        p.line([(c[0] + q * math.cos(t0 + sg * v), c[1] + q * math.sin(t0 + sg * v)) for q, v in zip(rr, t)], w, col, (.3, .3), z=z, n=4, a=a); k += 1
        if k >= n: break
    return p


def fur(P, pitch, amp, seed=0, lean=.5, closed=True, jit=.4):
    """tufted contour: pointed outward teeth leaning along the path (screen px)"""
    P = np.asarray(P, float); Q = resample(np.vstack([P, P[:1]]) if closed else P, 1.6); A = .5 * np.sum(Q[:-1, 0] * Q[1:, 1] - Q[1:, 0] * Q[:-1, 1]); sg = 1. if A > 0 else -1.
    T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]] * sg
    L = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; r = np.random.default_rng(seed); k = max(3, int(L[-1] / pitch)); e = np.cumsum(r.uniform(1 - jit, 1 + jit, k)); e = np.r_[0, e / e[-1]] * L[-1]
    i = np.clip(np.searchsorted(e, L, side='right') - 1, 0, k - 1); u = np.clip((L - e[i]) / (e[i + 1] - e[i] + 1e-9), 0, 1); a = r.uniform(.5, 1.3, k)[i]
    d = amp * a * np.where(u < .72, (u / .72) ** 1.6, ((1 - u) / .28) ** .8); return Q + N * d[:, None] + T * (amp * lean * a * u * (u < .72))[:, None]


def limb_poly(J, Wd, n=8):
    """tapered limb from joints J (screen) and widths Wd; widths may list extra bulges along it"""
    return band_poly(np.asarray(J, float), [float(v) for v in Wd], n)


def mixp(a, b, t):
    return {k: (a[k] * (1 - t) + b[k] * t if isinstance(a[k], (int, float)) and isinstance(b.get(k), (int, float)) else (a[k] if t < .5 else b.get(k, a[k]))) for k in a}


def pick(EX, expr):
    """'calm' | dict overrides | (a, b, t)"""
    if isinstance(expr, str): return dict(EX[expr])
    if isinstance(expr, tuple): return mixp(pick(EX, expr[0]), pick(EX, expr[1]), expr[2])
    d = dict(EX[expr.get('base', 'calm')]); d.update({k: v for k, v in expr.items() if k != 'base'}); return d


def E(**k):
    d = dict(eo=1., el=0., et=0., lx=0., ly=0., pr=.30, arc=1, bi=0., bo=0., mw=.25, mu=0., md=0., ml=0., mt=0., mb=0., tg=0., wob=0., ms=0., tear=0., snot=0., blush=0.); d.update(k); return d


def g1_t(f, seed=0):
    """worry -> laugh curve for g1: tiny hold, dip (anticipation) f18, snap f20-23, overshoot, settle; breathing laugh after.  returns (t, bounce)"""
    g = f // 2 * 2; d = seed * .6
    t = 0. if g < 20 + d else min(1., (g - 20 - d) / 3. + .4); antic = math.exp(-((g - 18 - d) / 1.6) ** 2) * .5; over = math.exp(-((g - 23 - d) / 2.) ** 2); lau = (.5 + .5 * math.sin((g - 24) * 2 * math.pi / 6. + seed)) * (g > 24) * math.exp(-(g - 24) / 60.)
    return t, -antic + over * 1.0 + .45 * lau


def crown_roll(x, y, cx=960, cy=500): return math.atan2(cx - x, -(cy - y))


# ---------------------------------------------------------------- painted card and helpers for the roll call
def card(base, streak, dark=None, seed=0, ang=-.5, n=220, h=H, w=W):
    """flat colour card with big dry-brush passes (refs 3, 6)"""
    r = np.random.default_rng(seed); cv = Canvas(h, w, color=base); b = rgb(base)
    for col, k, wd, op, dr in ((dark or base, .5, (40, 120), .55, .55), (streak, .5, (26, 90), .65, .75)):
        m = int(n * k); P = []; a = ang + r.normal(0, .12, m); x, y = r.uniform(-100, w + 100, m), r.uniform(-60, h + 60, m); L = r.uniform(260, 900, m); t = np.linspace(-.5, .5, 9)
        for i in range(m):
            bow = r.normal(0, .06) * L[i]; P.append(np.c_[x[i] + np.cos(a[i]) * L[i] * t - np.sin(a[i]) * bow * (1 - 4 * t * t), y[i] + np.sin(a[i]) * L[i] * t + np.cos(a[i]) * bow * (1 - 4 * t * t)])
        c = rgb(col)[None] * (1 + r.normal(0, .05, (m, 1)).astype(f32)); cv.strokes(np.array(P), r.uniform(*wd, m) / 2, np.clip(c, 0, 1), op=op * r.uniform(.4, 1, m), dry=dr * r.uniform(.6, 1.2, m), seed=seed + 3, bristle=.12)
    g = fbm(h, w, (3.2, 1.5), (1, .6), seed + 4); m = fbm(h, w, (300, 90), None, seed + 5)
    return np.clip(cv.img * (1 + .02 * g[..., None] + .03 * m[..., None]), 0, 1).astype(f32)


def ease(t): t = min(1., max(0., t)); return t * t * (3 - 2 * t)
def eout(t, p=3.): return 1 - (1 - min(1., max(0., t))) ** p
def ein(t, p=3.): return min(1., max(0., t)) ** p
def key(f, K, e=ease):
    """eased keys of scalars / arrays"""
    if f <= K[0][0]: return np.asarray(K[0][1], float) if np.ndim(K[0][1]) else float(K[0][1])
    for (a, va), (b, vb) in zip(K[:-1], K[1:]):
        if f < b: t = e((f - a) / float(b - a)); r = np.asarray(va, float) * (1 - t) + np.asarray(vb, float) * t; return r if np.ndim(va) else float(r)
    return np.asarray(K[-1][1], float) if np.ndim(K[-1][1]) else float(K[-1][1])
