"""SWORDS lane kit: faces drawn through head3 (Pen), cropped and swept hair, tears, real steel, torn paper, brush cards, cloth."""
import numpy as np, math, cv2, os, cairo
from op import W, H
from op.paint import rgb, f32, fbm, smooth, mix, grid, ramp, lum, Canvas, arcs, repaint
from op.ink import Cel, spl, resample, ribbon, noise1, xf, paste, over_roi
from op.kit import spike, outline, shape, contours, rim, para, tint, aura, cast, hatch, splatter, shifted, focus_lines, speed_lines, flare, smear, zoomblur, impact, shake, screentone
from op.comp import over, screen, blur, radial, cam2d, ease, ease_in, ease_out, lerp_keys, keyed, glow, bokeh
from op.fx_core import h1

INK = '#17131a'
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def A_(*p): return np.array(p, float)


def cr3(P, n=8):
    """uniform Catmull-Rom in any dimension"""
    P = np.asarray(P, float); m = len(P); out = []
    for i in range(m - 1):
        p1, p2 = P[i], P[i + 1]; p0 = P[i - 1] if i else 2 * p1 - p2; p3 = P[i + 2] if i + 2 < m else 2 * p2 - p1
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(.5 * (2 * p1 + (p2 - p0) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t * t + (3 * p1 - p0 - 3 * p2 + p3) * t ** 3))
    return np.vstack(out + [P[-1:]])


def union(polys, k=1.0, pad=10, sig=.8, min_len=24):
    """outer contours of a set of screen polygons -> list of polylines (screen px)"""
    polys = [np.asarray(p, float) for p in polys if len(p) > 2]
    if not polys: return []
    allp = np.vstack(polys); lo = allp.min(0) - pad; hi = allp.max(0) + pad
    w, h = np.ceil((hi - lo) * k).astype(int) + 2
    if w < 4 or h < 4 or w > 6000 or h > 6000: return []
    m = np.zeros((h, w), np.uint8)
    for p in polys: cv2.fillPoly(m, [np.round((p - lo) * k * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    if sig: m = cv2.GaussianBlur(m, (0, 0), sig)
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE); out = []
    kr = np.hanning(9); kr /= kr.sum()
    for c in cs:
        c = c[:, 0].astype(float)
        if len(c) < min_len: continue
        cc = np.vstack([c[-8:], c, c[:8]]); out.append(np.stack([np.convolve(cc[:, i], kr, 'same') for i in (0, 1)], -1)[8:-8] / k + lo)
    return out


def front_mass(H_, hem, infl=.0, ny=16, nf=72, top=.985):
    """contour(s) of the part of the scalp above hem(theta) that faces the camera"""
    th = np.linspace(-math.pi, math.pi, nf); h0 = hem(th); t = np.linspace(0, 1, ny)[:, None]; Y = h0[None] + (top - h0[None]) * t
    R, n = root3(H_, np.repeat(th[None], ny, 0), Y, infl); q = H_.P3(R); z = (n @ H_.R.T)[..., 2]; P = []
    for i in range(ny - 1):
        for j in range(nf - 1):
            if max(z[i, j], z[i + 1, j], z[i, j + 1], z[i + 1, j + 1]) > -.02: P.append(np.array([q[i, j], q[i, j + 1], q[i + 1, j + 1], q[i + 1, j]]))
    return union(P, 1., pad=4, sig=.6)


# ------------------------------------------------------------------ pen: draw in head-front coordinates
class Pen:
    def __init__(self, cel, H_, s, ink=INK, seed=0, ref=262.):
        self.c, self.H, self.s, self.ink, self.seed, self.k, self.ref = cel, H_, s, ink, seed, 0, ref
    def w(self, v): return max(.8, v * (self.s / self.ref) ** .85)
    def pts(self, P, n=8, corners=(), closed=False, dz=0.):
        P = np.asarray(P, float)
        if P.shape[1] == 3:                                       # third column = dz per point
            Q = cr3(P, n) if not closed else cr3(np.vstack([P, P[:1]]), n)
            return self.H.P(Q[:, :2], Q[:, 2])
        return self.H.P(spl(P, closed, corners, n), dz)
    def line(self, P, w=2.6, col=None, taper=(.18, .35), press=.3, corners=(), n=8, dz=0., wfun=None, alpha=1., tmin=.12, raw=False):
        self.k += 1; q = np.asarray(P, float) if raw else self.pts(P, n, corners, dz=dz)
        self.c.line(q, self.w(w), col or self.ink, taper, press, self.seed * 131 + self.k, .25, alpha, wfun=wfun, tmin=tmin); return self
    def fill(self, P, col, corners=(), n=8, dz=0., alpha=1., raw=False):
        self.c.fill(np.asarray(P, float) if raw else self.pts(P, n, corners, True, dz), col, alpha); return self
    def poly(self, P, col, alpha=1., dz=0., n=5):
        """straight-sided (planes) shape following the surface"""
        P = np.asarray(P, float); Q = np.vstack([P[:, :2], P[:1, :2]]); Z = np.r_[P[:, 2], P[0, 2]] if P.shape[1] == 3 else np.zeros(len(Q))
        t = np.linspace(0, len(Q) - 1, (len(Q) - 1) * n + 1); i = np.arange(len(Q))
        q = np.c_[np.interp(t, i, Q[:, 0]), np.interp(t, i, Q[:, 1])]; self.c.fill(self.H.P(q, np.interp(t, i, Z) + dz), col, alpha); return self
    def hatch(self, A, B, n, d, w=1.5, col=None, jit=.15, alpha=1., dz=0., seed=0, bow=.02):
        """n strokes whose roots run A->B, each going by vector d (front coordinates)"""
        r = np.random.default_rng(seed + 17); A, B, d = A_(*A), A_(*B), A_(*d)
        for i in range(n):
            t = (i + .5 + r.uniform(-.25, .25)) / n; p = A + (B - A) * t; dd = d * r.uniform(1 - jit * 2, 1 + jit) * (1 - .25 * abs(2 * t - 1))
            nn = A_(-dd[1], dd[0]) * r.uniform(-1, 1) * bow * 6
            self.line([p, p + dd * .5 + nn, p + dd], w * r.uniform(.75, 1.2), col, taper=(.06, .65), n=6, alpha=alpha, dz=dz)
        return self


def eye(p, side, up, lo, pupil, pr=.048, white='#fdf8f0', shade='#b7a3a4', lw=8., lid=.055, open_=1., crease=None, low=2.2, gl=True, prx=None):
    """almond eye.  up: inner->outer, lo: outer->inner (authored on +x side); pupil = centre; open_ squeezes it toward its lower lid"""
    m = A_(side, 1); U = spl(np.array(up, float) * m, n=10); L = spl(np.array(lo, float) * m, n=10)
    if open_ < 1:
        yl = np.interp(np.abs(U[:, 0]), np.abs(L[::-1, 0]) if abs(L[-1, 0]) < abs(L[0, 0]) else np.abs(L[:, 0]), L[::-1, 1] if abs(L[-1, 0]) < abs(L[0, 0]) else L[:, 1])
        U = U.copy(); U[:, 1] = yl + (U[:, 1] - yl) * max(open_, 0.)
    H_, c = p.H, p.c; poly = H_.P(np.vstack([U, L]))
    if open_ > .12:
        c.fill(poly, white); c.clip(poly)
        c.fill(H_.P(np.vstack([U, (U - [0, lid])[::-1]])), shade)
        a = np.linspace(0, 2 * math.pi, 28); px, py = pupil[0] * side, pupil[1]
        c.fill(H_.P(np.c_[px + np.cos(a) * (prx or pr), py + np.sin(a) * pr]), p.ink)
        if gl: c.fill(H_.P(np.c_[px - .34 * pr + np.cos(a) * pr * .34, py + .34 * pr + np.sin(a) * pr * .34]), '#ffffff')
        c.unclip()
    p.line(H_.P(U), lw * (1 if open_ > .12 else .72), raw=True, taper=(.10, .16), press=.15, wfun=lambda t: .62 + .55 * np.sin(np.pi * np.clip(t * .8 + .18, 0, 1)), tmin=.3)
    if open_ > .12: p.line(H_.P(L), low, raw=True, taper=(.25, .5), press=.35)
    if crease is not None: p.line(np.array(crease, float) * m, 1.9, taper=(.3, .5))
    return p


def ribbon_fill(p, P, wd, col, dz=0., n=8):
    """tapering brow-like shape: centre line P in front coords, wd(t) half-width in units"""
    Q = spl(np.asarray(P, float), n=n); T = np.gradient(Q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    t = np.linspace(0, 1, len(Q)); hw = np.array([wd(v) for v in t])[:, None]
    P2 = np.vstack([Q + N * hw, (Q - N * hw)[::-1]]); p.c.fill(p.H.P(P2, dz), col); return P2


def mouth_open(p, up, lo, inner='#43121a', teeth='#fdf9ef', tshade='#cfc0bb', tu=.10, tl=.06, tongue=None, tcol='#d9605f', lw=4.2, gum=None, seams=0, lip=True, tm=.82):
    """up: left->right upper lip, lo: right->left lower lip (front coords).  tu/tl = tooth band heights"""
    U = spl(np.asarray(up, float), n=10); L = spl(np.asarray(lo, float), n=10); H_, c = p.H, p.c; poly = H_.P(np.vstack([U, L]))
    c.fill(poly, inner); c.clip(poly)
    if tongue is not None: p.fill(tongue, tcol); p.line([(np.mean([q[0] for q in tongue]), max(q[1] for q in tongue) - .02), (np.mean([q[0] for q in tongue]), np.mean([q[1] for q in tongue]))], 1.8, '#8d2f3a')
    for E, hgt, sg in ((U, tu, -1), (L, tl, 1)):
        if hgt <= 0: continue
        t = np.linspace(0, 1, len(E)); e = np.clip(np.minimum(t, 1 - t) / (1 - tm + 1e-6) * 1.0, 0, 1) ** .5
        B = E + np.c_[np.zeros(len(E)), sg * hgt * (.35 + .65 * e)]
        c.fill(H_.P(np.vstack([E, B[::-1]])), teeth)
        if gum: c.fill(H_.P(np.vstack([E, (E + np.c_[np.zeros(len(E)), np.full(len(E), sg * gum)])[::-1]])), '#c9565c')
        p.line(H_.P(B), 1.8, '#5a3134', raw=True, taper=(.1, .1))
        for k in range(1, seams + 1):
            i = int(len(E) * k / (seams + 1)); p.line(H_.P(np.array([E[i] + (B[i] - E[i]) * .25, B[i]])), 1.2, '#a88f8c', raw=True)
        c.fill(H_.P(np.vstack([B, (B - np.c_[np.zeros(len(E)), np.full(len(E), sg * hgt * .28)])[::-1]])), tshade, .7)
    c.unclip()
    p.line(H_.P(U), lw, raw=True, taper=(.12, .12), press=.3, tmin=.35); p.line(H_.P(L), lw * .85, raw=True, taper=(.12, .12), press=.3, tmin=.35)
    return poly


def nose(p, pal, k=1, near=None, hi=True, lw=3.2, tip=.165, w=.128, long=0.):
    """straight Oda nose: shadow plane on side k, ridge line on the far side, nostril wing"""
    yaw = p.H.yaw; near = near if near is not None else (1 if yaw < -.02 else (-1 if yaw > .02 else k)); y = -long
    p.poly([(0, .02, .03), (0, -.16 + y * .5, .085), (0, -.385 + y, tip), (k * .05, -.445 + y, .10), (k * w, -.455 + y, .02), (k * (w - .01), -.36 + y, .015), (k * .055, -.14, .0), (k * .06, .02, 0)], pal['shade'])
    p.poly([(-w * .9, -.452 + y, .03), (-.03, -.475 + y, .07), (.03, -.475 + y, .07), (w * .9, -.452 + y, .03), (k * .075, -.535 + y, 0), (k * .0, -.515 + y, 0)], pal['shade'])
    if hi: p.poly([(-k * .010, -.05, .06), (-k * .004, -.28 + y, .13), (-k * .012, -.375 + y, tip), (-k * .032, -.33 + y, .12), (-k * .030, -.10, .05)], pal['light'])
    f = -near
    R = [(f * .018, -.05, .05), (f * .012, -.18 + y * .5, .095), (f * .004, -.30 + y, .14), (f * .012, -.378 + y, tip), (f * .018, -.425 + y, tip * .86), (near * .015, -.468 + y, .085)]
    p.line(R, lw, taper=(.5, .12), n=7)
    p.line([(near * .085, -.375 + y, .06), (near * (w + .012), -.418 + y, .035), (near * (w + .004), -.458 + y, .03), (near * .085, -.470 + y, .05), (near * .05, -.452 + y, .07)], lw * .78, taper=(.4, .2), n=7)
    p.fill([(near * .04, -.458 + y, .075), (near * .075, -.452 + y, .06), (near * .085, -.466 + y, .05), (near * .05, -.474 + y, .065)], p.ink, n=5)
    if abs(yaw) < .5: p.line([(f * .07, -.462 + y, .05), (f * (w - .012), -.452 + y, .03)], lw * .6, taper=(.3, .5))


# ------------------------------------------------------------------ tears
def rope(cel, P, w, seed=0, col='#eaf7ff', sh='#8fc4e6', line='#1d3350', lw=2.2, end=1.0):
    """fat outlined tear rope down a screen polyline; w = width px (scalar or f(t))"""
    P = resample(np.asarray(P, float), 3.); n = len(P)
    if n < 4: return cel
    t = np.linspace(0, 1, n); r = np.random.default_rng(seed); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    wd = (np.array([w(v) for v in t]) if callable(w) else np.full(n, float(w))) * (1 + .10 * noise1(n, 20, r)) * .5
    wd[:4] *= np.linspace(.5, 1, 4); body = np.vstack([P + N * wd[:, None], (P - N * wd[:, None])[::-1]]); e = P[-1]
    a = np.linspace(0, 2 * math.pi, 24); bead = e + np.c_[np.cos(a), np.sin(a)] * wd[-1] * 1.06 * end
    C = union([body, bead], 2., sig=.9, min_len=10)
    for c in C: cel.fill(c, col)
    cel.fill(np.vstack([P + N * wd[:, None] * .1, (P - N * wd[:, None])[::-1]]), sh, .75)
    for c in C: cel.line(np.vstack([c, c[:1]]), lw, line, taper=(.05, .05), press=.4, seed=seed)
    q = P + N * wd[:, None] * .42; i0, i1 = int(n * .12), int(n * .82)
    if i1 - i0 > 4: cel.line(q[i0:i1], max(1.6, wd.mean() * .5), '#ffffff', taper=(.3, .4), press=.6, seed=seed + 3)
    cel.fill(e + np.c_[np.cos(a), np.sin(a)] * wd[-1] * .30 + N[-1] * wd[-1] * .5, '#ffffff'); return cel


def bead(cel, x, y, r, ang=math.pi / 2, st=1.35, lw=2.2, col='#eaf7ff', sh='#8fc4e6', line='#1d3350'):
    """flying tear drop, tail pointing back along ang"""
    c, s = math.cos(ang), math.sin(ang); a = np.linspace(-2.3, 2.3, 22); P = np.c_[np.cos(a) * r, np.sin(a) * r]; P = np.vstack([P, [(-r * st * 1.5, 0)]])
    R_ = np.array([[c, -s], [s, c]]); Q = P @ R_.T + [x, y]; cel.fill(Q, col)
    cel.fill((np.c_[np.cos(a) * r * .9 + .12 * r, np.sin(a) * r * .9 + .22 * r][8:] @ R_.T) + [x, y], sh, .8)
    cel.line(np.vstack([Q, Q[:1]]), lw, line, taper=(.02, .02), press=.3); b = np.linspace(0, 6.283, 14)
    cel.fill(np.c_[x - r * .3 + np.cos(b) * r * .26, y - r * .34 + np.sin(b) * r * .22], '#ffffff'); return cel


class Water:
    """clear tears: streams and drops gather in a mask; put() bends the picture behind them, darkens their edge and lights them"""
    def __init__(self): self.m, self.k = Cel(), Cel()
    def stream(self, P, w, seed=0, phase=0., drop=1.15):
        P = resample(np.asarray(P, float), 3.); n = len(P)
        if n < 6: return self
        t = np.linspace(0, 1, n); r = np.random.default_rng(seed); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; s = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]
        wd = .5 * w * (1 + .26 * np.sin(6.283 * (s / 74. - phase)) + .10 * noise1(n, 16, r)) * np.clip(.45 + 2.2 * t, 0, 1); body = np.vstack([P + N * wd[:, None], (P - N * wd[:, None])[::-1]]); e = P[-1]; a = np.linspace(0, 6.283, 26)
        bd = [body] + ([e + np.c_[np.cos(a), np.sin(a)] * wd[-1] * drop] if drop else [])
        for c in union(bd, 2., sig=.8, min_len=8): self.m.fill(c, '#ffffff'); self.k.line(np.vstack([c, c[:1]]), max(1.8, w * .13), '#16324e', taper=(.03, .03), press=.7, seed=seed)
        i0, i1 = int(n * .06), int(n * .94); sh = P - N * (wd * .52)[:, None]; self.k.line(sh[i0:i1], max(1.4, w * .16), '#2c6a98', taper=(.3, .3), press=.8, seed=seed + 1, alpha=.55)
        hl = P + N * (wd * .40)[:, None]
        for q in range(int(s[-1] / 74.) + 2):                                # highlights ride on the bulges and travel with them
            c0 = (q + .25 + phase % 1.) * 74.; ii = np.nonzero(np.abs(s - c0) < 23)[0]
            if len(ii) > 4: self.k.line(hl[ii], max(1.8, w * .26), '#ffffff', taper=(.4, .4), press=.4, seed=seed + q)
        self.k.fill(e + np.c_[np.cos(a) * .26, np.sin(a) * .26] * wd[-1] + N[-1] * wd[-1] * .45 - T[-1] * wd[-1] * .2, '#ffffff'); return self
    def drop(self, x, y, r, ang=math.pi / 2, st=1.3):
        c, s = math.cos(ang), math.sin(ang); a = np.linspace(-2.3, 2.3, 22); P = np.vstack([np.c_[np.cos(a) * r, np.sin(a) * r], [(-r * st * 1.5, 0)]]) @ np.array([[c, -s], [s, c]]).T + [x, y]; self.m.fill(P, '#ffffff'); self.k.line(np.vstack([P, P[:1]]), max(1.5, r * .17), '#16324e', taper=(.02, .02), press=.6); b = np.linspace(0, 6.283, 12); self.k.fill(np.c_[x - r * .32 + np.cos(b) * r * .26, y - r * .34 + np.sin(b) * r * .22], '#ffffff'); q = np.linspace(.3, 1.6, 10); self.k.line(np.c_[x + np.cos(q) * r * .62, y + np.sin(q) * r * .62], max(1.2, r * .14), '#e8f7ff', taper=(.3, .3)); return self
    def put(self, img, amt=13.):
        m = self.m.rgba()[..., 3]
        if m.max() < .01: return img
        b = cv2.GaussianBlur(m, (0, 0), 4.5); gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, scale=.125); gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, scale=.125); x, y = grid(); ref = cv2.remap(img.astype(f32), x - gx * amt * 22, y - gy * amt * 22, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        edge = np.clip(1 - b * 1.25, 0, 1)[..., None]; w = ref * (1.10 - .34 * edge) * np.array([.93, 1.0, 1.06], f32) + rgb('#cfeaff') * .12; lit = np.clip(-gx - gy, 0, 1)[..., None] * 2.2; w = w + lit * .5
        out = img * (1 - m[..., None]) + w * m[..., None]; out = out * (1 - .30 * (np.clip(shifted(b, 5, 7) - m, 0, 1))[..., None]); return over(out, self.k.layer())


# ------------------------------------------------------------------ hair
def root3(H_, th, y, lift=0.):
    A = np.interp(y, H_.yy, H_.A); c = np.cos(th); B = np.where(c >= 0, np.interp(y, H_.yy, H_.F), np.interp(y, H_.yy, H_.B)); p = H_.k['p']
    sx = np.sin(th); r = (np.abs(sx) ** p + np.abs(c) ** p) ** (-1 / p); R = np.stack([A * sx * r, y, B * c * r], -1)
    n = np.stack([sx / np.maximum(A, .12), (y - .1) * .95, c / np.maximum(B, .12)], -1); n /= np.linalg.norm(n, axis=-1, keepdims=True)
    return R + n * lift, n


def crop_set(H_, hairline, flow, seed=3, dmin=.078, cand=6400, L=(.20, .42), WD=(.085, .155)):
    """pose-independent table of short clumps on the skull (blue-noise roots)"""
    r = np.random.default_rng(seed); th = r.uniform(-math.pi, math.pi, cand); hl = hairline(th); y = hl + (.99 - hl) * r.uniform(0, 1, cand) ** .9
    R, n = root3(H_, th, y, .0); keep = []; K = np.zeros((0, 3))
    edge = np.abs(y - hl) < .04
    for i in np.argsort(~edge):
        d = dmin * (.75 if edge[i] else 1.)
        if len(K) and (np.einsum('ij,ij->i', K - R[i], K - R[i]) < d * d).any(): continue
        keep.append(i); K = np.vstack([K, R[i]])
    k = np.array(keep); th, y, R, n, edge = th[k], y[k], R[k], n[k], edge[k]; m = len(k); z = r.uniform(0, 1, m)
    size = np.where(z < .2, 1., np.where(z < .6, .74, .52)) * r.uniform(.88, 1.12, m)
    d, lk = flow(th, y, n, edge, r); d /= np.linalg.norm(d, axis=1, keepdims=True)
    e = np.cross(d, n); e += r.normal(0, .02, e.shape); e /= np.linalg.norm(e, axis=1, keepdims=True) + 1e-9
    ln = (L[0] + (L[1] - L[0]) * size) * lk; wd = (WD[0] + (WD[1] - WD[0]) * size) * (.6 + .4 * np.minimum(lk, 1))
    return dict(th=th, y=y, R=R, n=n, d=d, e=e, ln=ln, wd=wd, size=size, u=r.uniform(0, 1, m), bend=r.normal(0, .10, m), edge=edge)


def crop_draw(cel, H_, T, which, pal, L=(-.55, .6, .55), ink=INK, lw=1.7, wind=None, clipout=None, seed=0):
    """draw a subset of clumps back to front; returns their polygons.  pal: deep shade base light hot"""
    s = H_.s; R, d, e = T['R'], T['d'], T['e']; wv = 0 if wind is None else wind(T)
    tip = R + d * T['ln'][:, None] + wv; a = R - e * T['wd'][:, None] * .5; b = R + e * T['wd'][:, None] * .5
    A, B, Q = H_.P3(a), H_.P3(b), H_.P3(tip); dep = H_.depth((R + tip) / 2); nv = T['n'] @ H_.R.T; l = np.asarray(L, float) / np.linalg.norm(L); lam = nv @ l
    idx = np.nonzero(which)[0]; idx = idx[np.argsort(dep[idx])]; polys = []; k = (s / 330.) ** .8
    for i in idx:
        P = spike(A[i], Q[i], B[i], .055, T['bend'][i], 9); polys.append(P); lv = lam[i] + (T['u'][i] - .5) * .34
        c0, c1 = (pal['base'], pal['shade']) if lv > .30 else ((pal['shade'], pal['deep']) if lv > -.12 else (pal['deep'], pal['deep']))
        if lv > .74: c0 = pal['light']; c1 = pal['base']
        cel.fill(P, c0); m = (A[i] + B[i]) / 2; ax = Q[i] - m; nn = A_(-ax[1], ax[0]); sg = 1 if nn @ A_(-l[0], l[1]) > 0 else -1
        far = B[i] if (B[i] - m) @ nn * sg > 0 else A[i]; near = A[i] if far is B[i] else B[i]
        cel.fill(spike(m + (near - m) * .35, m + ax * .86, far + (far - m) * .05, .05, T['bend'][i], 7), c1)
        if lv > .38 and T['size'][i] > .6:
            q0 = near + (Q[i] - near) * .22; cel.fill(spike(q0, near + (Q[i] - near) * .93 + (m - near) * .04, q0 + (far - near) * .20, .06, T['bend'][i], 6), pal['hot'] if lv > .68 else pal['light'])
        if s > 60:
            E = P[:len(P) // 2 + 1] if far is A[i] else P[len(P) // 2:][::-1]
            cel.line(E[int(len(E) * (.15 if T['size'][i] > .6 else .4)):], lw * k * (1.15 if T['size'][i] > .85 else .85), ink if lv < .55 else pal['line'], taper=(.5, .1), press=.35, seed=int(i) + seed)
    return polys


def strand(H_, P3, w, n=7, prof=None, off=0.):
    """swept clump along a 3-D guide -> (poly, centre, left, right, centre3)"""
    C = cr3(P3, n); q = H_.P3(C); T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    t = np.linspace(0, 1, len(C)); pf = prof(t) if prof else np.clip(.35 + 1.9 * t, 0, 1) * np.clip((1 - t) / .55, 0, 1) ** .72
    hw = (w * H_.s * .5 * pf)[:, None]; l, r = q + N * hw, q - N * hw
    return np.vstack([l, r[::-1]]), q, l, r, C


def strand_draw(cel, H_, S, pal, L=(-.5, .6, .6), ink=INK, lw=2.0, seed=0, band=(.16, .58), inner=2, c0=(0, .15, 0)):
    """one swept clump: flat, under-side shadow, root accent, highlight ribbons, ink"""
    poly, q, l, r, C = S; n = len(q); l3 = np.asarray(L, float) / np.linalg.norm(L); nv = (C - A_(*c0)); nv /= np.linalg.norm(nv, axis=1, keepdims=True); lam = (nv @ H_.R.T) @ l3
    m = float(np.mean(lam[n // 4:n * 3 // 4])); rg = np.random.default_rng(seed); k = (H_.s / 330.) ** .8
    base, sh = (pal['base'], pal['shade']) if m > .15 else ((pal['shade'], pal['deep']) if m > -.35 else (pal['deep'], pal['deep']))
    cel.fill(poly, base)
    ld = A_(-l3[0], l3[1]); g = np.mean((r - l) @ ld); dk, lt = (r, l) if g > 0 else (l, r)
    t0, t1 = int(n * rg.uniform(.0, .08)), int(n * rg.uniform(.80, .98)); mix_ = rg.uniform(.30, .5)
    e = q + (dk - q) * (1 - 2 * mix_)
    u = np.sin(np.linspace(0, np.pi, t1 - t0)) ** .5; E = dk[t0:t1] + (e[t0:t1] - dk[t0:t1]) * u[:, None]
    cel.fill(np.vstack([dk[t0:t1], E[::-1]]), sh)
    a = int(n * rg.uniform(.10, .22)); cel.fill(np.vstack([l[:a], q[a:a + 1] + (q[min(n - 1, a + 3):min(n - 1, a + 3) + 1] - q[a:a + 1]) * .5, r[:a][::-1]]), sh)
    wpx = np.hypot(*(l - r)[n // 3])
    if m > -.05 and wpx > 14 * k:                                             # zigzag highlight band, the same height on every lock
        b0 = band[0] + rg.uniform(-.025, .025); b1 = b0 + (band[1] - band[0]) * rg.uniform(.7, 1.1); z = 4 if wpx > 50 * k else 3; at = lambda t, o: (q + (lt - q) * o)[int(np.clip(t * (n - 1), 0, n - 1))] if o >= 0 else (q + (dk - q) * -o)[int(np.clip(t * (n - 1), 0, n - 1))]
        os_ = np.linspace(.92, -.55, z * 2 + 1); up = [at(b0 + (.030 if i % 2 else -.045) + .03 * rg.normal(), o) for i, o in enumerate(os_)]; dn = [at(b1 + (.075 if i % 2 else -.06) + .02 * rg.normal(), o) for i, o in enumerate(os_[::-1])]
        cel.fill(np.array(up + dn), pal['light'])
        if m > .22:
            os2 = np.linspace(.80, .05, 5); up = [at(b0 + .035 + (.02 if i % 2 else -.02), o) for i, o in enumerate(os2)]; dn = [at(b1 - .05 + (.05 if i % 2 else -.05), o) for i, o in enumerate(os2[::-1])]; cel.fill(np.array(up + dn), pal['hot'])
    i0 = int(n * rg.uniform(.20, .42)); cel.line(dk[i0:], lw * k * (1.25 if wpx > 30 * k else .8), ink, taper=(.5, .30), press=.35, seed=seed + 9, tmin=.06)
    if wpx > 26 * k: i0 = int(n * rg.uniform(.50, .74)); cel.line(lt[i0:], lw * k * .85, pal['line'], taper=(.55, .35), press=.35, seed=seed + 10, tmin=.06)
    for j in range(inner):
        o = rg.uniform(-.6, .6); i0 = int(n * rg.uniform(.02, .62)); i1 = min(n, i0 + int(n * rg.uniform(.16, .36)))
        if i1 - i0 > 4: cel.line(q[i0:i1] + (dk - q)[i0:i1] * o, lw * k * .55, pal['line'], taper=(.45, .4), press=.4, seed=seed + 20 + j)
    return poly


def cap(H_, hem, infl=.06, ny=26, nf=90, top=.985):
    """contour and mesh of a cloth cap hugging the skull above hem(theta)"""
    th = np.linspace(-math.pi, math.pi, nf); h0 = hem(th); t = np.linspace(0, 1, ny)[:, None]; Y = h0[None] + (top - h0[None]) * t
    R, n = root3(H_, np.repeat(th[None], ny, 0), Y, 0.); c = A_(0, .10, 0); M = c + (R - c) * (1 + infl)
    return H_._sil([M]), M, th


# ------------------------------------------------------------------ steel
STEEL = dict(
    wado=dict(dk='#56637a', md='#9db0c6', lt='#dbe8f4', ed='#ffffff', wrap='#f7f3e8', wrapd='#b9b2a6', ray='#8f8a86', guard='#e2b23c', guardd='#8a5f18', saya='#f5f1e8', sayad='#a9a5b2', band='#d8a832'),
    kitetsu=dict(dk='#4e4f63', md='#a3a0ae', lt='#e6dfe2', ed='#fff6ee', wrap='#8a2a26', wrapd='#4a1214', ray='#d8b46a', guard='#b8862c', guardd='#5c3a10', saya='#7a1f1f', sayad='#3c0c12', band='#c89434'),
    enma=dict(dk='#0f0b18', md='#2b2140', lt='#5c4490', ed='#d9c2ff', wrap='#4a3a78', wrapd='#1c1436', ray='#0f0b16', guard='#8b8296', guardd='#3a3448', saya='#241a38', sayad='#0e0a18', band='#8b7aa8'),
    yoru=dict(dk='#0b0a12', md='#20202c', lt='#4a4c60', ed='#aeb8d0', wrap='#2a2026', wrapd='#120c10', ray='#0b0a0c', guard='#d0a43c', guardd='#6a4a14', saya='#1a1420', sayad='#0a080c', band='#c89434'))


def _blade_pts(g, t, w0, w1, curve, edge, n=60):
    g, t = A_(*g), A_(*t); d = t - g; L = np.hypot(*d); u = d / L; nrm = A_(u[1], -u[0]) * edge               # nrm = side of the cutting edge
    v = np.linspace(0, 1, n)[:, None]; c = g + d * v + nrm * (curve * L * 4 * v * (1 - v)); w = w0 + (w1 - w0) * v
    T = np.gradient(c, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = np.c_[T[:, 1], -T[:, 0]] * edge
    k = .91; tip = np.clip((v - k) / (1 - k), 0, 1); fe = np.sqrt(np.clip(1 - tip ** 2, 0, 1)); back = c - N * w * .5; ed = c + N * w * (.5 * fe + -.5 * (1 - fe))
    return v[:, 0], back, ed, c, N, w


def katana(cel, g, t, w0, w1=None, kind='wado', curve=.03, edge=1, glint=None, lw=2.6, ink=INK, seed=0, refl=0., hi=1.0):
    """blade from guard point g to tip t (px).  edge=+1: cutting edge on the left-hand side when looking along g->t on screen (y down).  glint = 0..1 travelling light"""
    w1 = w0 * .78 if w1 is None else w1; K = STEEL[kind]; v, b, e, c, N, w = _blade_pts(g, t, w0, w1, curve, edge); n = len(v)
    P = np.vstack([b, e[::-1]]); cel.fill(P, K['md']); cel.clip(P)
    X = lambda q: b + (e - b) * q                                                              # q 0 back .. 1 edge
    cel.fill(np.vstack([X(-.02), X(.30)[::-1]]), K['dk'])                                      # bevel above the ridge line
    cel.fill(np.vstack([X(.30), X(.355)[::-1]]), K['lt'])                                      # shinogi catch
    r = np.random.default_rng(seed + 11); wav = noise1(n * 3, 5, r); vv = np.linspace(0, 1, n * 3)
    XX = lambda q: np.c_[np.interp(vv, v, b[:, 0]), np.interp(vv, v, b[:, 1])] * (1 - q[:, None]) + np.c_[np.interp(vv, v, e[:, 0]), np.interp(vv, v, e[:, 1])] * q[:, None]
    amp = .085 if kind == 'kitetsu' else .045; ham = .70 + amp * wav * (1 - vv ** 8) + (.03 * np.sin(vv * 60) if kind != 'kitetsu' else .05 * np.abs(np.sin(vv * 34)))
    cel.fill(np.vstack([XX(ham - .05), XX(np.ones_like(ham))[::-1]]), K['lt'], .55); cel.fill(np.vstack([XX(ham), XX(np.ones_like(ham))[::-1]]), K['ed'])
    for q0, q1, a in ((.18, .23, .5), (.46, .62, .55), (.78, .80, .6)):                         # room reflections, sliding
        q0, q1 = (q0 + refl) % 1.1, (q1 + refl) % 1.1
        if q1 < q0: continue
        i0, i1 = int(q0 * (n - 1)), max(int(q0 * (n - 1)) + 1, int(min(q1, 1) * (n - 1)))
        if i1 < n: cel.fill(np.array([X(.36)[i0], X(.36)[i1], X(.66)[min(n - 1, i1 + 2)], X(.66)[min(n - 1, i0 + 2)]]), K['lt'], a * hi)
    if glint is not None and -.1 < glint < 1.1:
        for wd, al in ((.11, .55), (.045, 1.)):
            q0, q1 = glint - wd, glint + wd; ii = np.nonzero((v >= q0) & (v <= q1))[0]
            if len(ii) > 1: cel.fill(np.vstack([X(0)[ii], X(1)[np.minimum(ii + 2, n - 1)][::-1]]), '#ffffff', al)
    cel.unclip(); cel.line(b, lw, ink, taper=(.04, .12), press=.2, seed=seed); cel.line(e, lw * .62, ink, taper=(.04, .12), press=.3, seed=seed + 1, alpha=.85)
    i = int(n * .91); cel.line(np.array([X(.0)[i], X(.35)[i + 1], X(.95)[i - 1]]), lw * .45, K['dk'], taper=(.2, .4))
    return c


def hilt(cel, g, p, w, kind='wado', lw=2.6, ink=INK, guard=1.0, asp=.34, seed=0, hab=True, n=None, blade=None):
    """guard at g, pommel at p (px), w = hilt thickness.  blade = direction of the blade (default g-p)"""
    K = STEEL[kind]; g, p = A_(*g), A_(*p); d = p - g; L = np.hypot(*d); u = d / L; m = A_(-u[1], u[0]); v = np.linspace(0, 1, 24)[:, None]
    prof = (1 - .10 * np.sin(np.pi * v)) * .5 * w; c = g + d * v; A, B = c + m * prof, c - m * prof; P = np.vstack([A, B[::-1]]); cel.fill(P, K['ray']); cel.clip(P)
    n = n or max(4, int(L / (w * .62))); X = lambda a, q: g + d * a + m * (w * .56 * q)
    for j in range(-1, n + 1):
        for sg in (1, -1):
            a0, a1 = j / n, (j + 1) / n; th = .36 / n
            cel.fill(np.array([X(a0 - th, sg), X(a0 + th, sg), X(a1 + th, -sg), X(a1 - th, -sg)]), K['wrap'])
            cel.fill(np.array([X(a0 + th * .35, sg), X(a0 + th, sg), X(a1 + th, -sg), X(a1 + th * .35, -sg)]), K['wrapd'])
            cel.line(np.array([X(a0 + th, sg), X(a1 + th, -sg)]), lw * .42, ink, taper=(.05, .05))
    cel.fill(np.vstack([c - m * w * .12, (c - m * w * .6)[::-1]]), '#1a1226', .28)
    cel.fill(np.vstack([c + m * w * .30, (c + m * w * .40)[::-1]]), '#ffffff', .30); cel.unclip()
    outline(cel, P, lw, ink, seed)
    e = np.array([p + m * w * .56 - u * w * .18, p + m * w * .5 + u * w * .30, p + m * w * .22 + u * w * .46, p - m * w * .22 + u * w * .46, p - m * w * .5 + u * w * .30, p - m * w * .56 - u * w * .18])
    cel.fill(spl(e, True, (0, 5)), K['guard']); cel.fill(spl(np.vstack([e[3:], p + u * w * .05]), True), K['guardd']); outline(cel, spl(e, True, (0, 5)), lw * .9, ink, seed + 1)
    if guard:
        a = np.linspace(0, 2 * math.pi, 40); R_ = w * 1.12 * guard; E = lambda o, k=1.: g + o * u + (np.cos(a)[:, None] * m * R_ * k) + (np.sin(a)[:, None] * u * R_ * asp * k)
        cel.fill(E(w * .16), K['guardd']); cel.line(E(w * .16), lw, ink, closed=True, press=.25); cel.fill(E(0), K['guard']); cel.fill(E(0, .74), K['guardd'], .55)
        cel.line(E(0, .74), lw * .5, ink, closed=True); cel.line(E(0), lw, ink, closed=True, press=.3)
        q = np.linspace(-2.6, -1.4, 12); cel.line(g + np.cos(q)[:, None] * m * R_ * .87 + np.sin(q)[:, None] * u * R_ * asp * .87, lw * 1.2, '#fff3bd', taper=(.3, .3))
        if hab:
            hb = np.array([g + m * w * .36, g - m * w * .36, g - m * w * .33 - u * w * .55, g + m * w * .33 - u * w * .55]); cel.fill(hb, K['guard']); cel.fill(hb[[0, 3]].tolist() + (hb[[3, 0]] - m * w * .22).tolist(), K['guardd']); outline(cel, hb, lw * .7, ink, seed + 2)
    return cel


def scabbard(cel, a, b, w, kind='wado', lw=2.4, ink=INK, curve=.03, side=1):
    K = STEEL[kind]; a, b = A_(*a), A_(*b); d = b - a; L = np.hypot(*d); u = d / L; m = A_(-u[1], u[0]) * side; v = np.linspace(0, 1, 30)[:, None]
    c = a + d * v + m * curve * L * 4 * v * (1 - v); A, B = c + m * w * .5, c - m * w * .42; P = np.vstack([A, B[::-1], [b + u * w * .25]]); cel.fill(P, K['saya']); cel.fill(np.vstack([c - m * w * .05, B[::-1]]), K['sayad'])
    cel.fill(np.vstack([c + m * w * .24, (c + m * w * .36)[::-1]]), '#ffffff', .5 if kind != 'wado' else .9)
    for q in (.04, .22, .30):
        i = int(q * 29); cel.fill(np.array([A[i], A[i + 1], B[i + 1], B[i]]), K['band'])
    outline(cel, P, lw, ink); return cel


# ------------------------------------------------------------------ painted cards and paper
def brush_card(cols, ang=-.5, seed=0, h=H, w=W, dark=None, n=(46, 110, 260), wd=(95, 46, 15), splat=90, keep=None):
    """dry-brush colour card: cols = list of hex, dark to light, dark ground first"""
    r = np.random.default_rng(seed); cv = Canvas(h, w, color=dark or cols[0]); C = np.array([rgb(c) for c in cols]); L = math.hypot(w, h)
    for k, (m, b) in enumerate(zip(n, wd)):
        x = r.uniform(-.1 * w, 1.1 * w, m); y = r.uniform(-.1 * h, 1.1 * h, m); ln = L * r.uniform(.10, .42, m) * (1 - .22 * k)
        P = arcs(x, y, ang + r.normal(0, .10 + .03 * k, m), ln, r.normal(0, .22, m), K=14); ci = np.clip((r.uniform(0, 1, m) ** (1.3 - .25 * k) * len(C)).astype(int), 0, len(C) - 1)
        cv.strokes(P, b * r.uniform(.5, 1.25, m), C[ci] * r.uniform(.9, 1.08, (m, 1)), op=r.uniform(.5, .95, m), dry=r.uniform(.25, .75, m), seed=seed + k * 31, bristle=.22)
    img = cv.img
    if splat:
        c = Cel(w, h, ss=1)
        for _ in range(splat):
            q = r.uniform(0, 1, 2); splatter(c, q[0] * w, q[1] * h, 26, 5, 7, cols[int(r.integers(0, len(cols)))], int(r.integers(1 << 30)), ang + math.pi * (r.random() > .5), 1.2)
        img = over(img, c.rgba())
    g = fbm(h, w, (5, 2.2), (1, .7), seed + 3); return np.clip(img * (1 + .035 * g[..., None]), 0, 1).astype(f32)


def tear_line(a, b, seed=0, amp=13., n=240):
    """ragged paper tear from a to b: big wander, medium bites, small teeth -> polyline"""
    r = np.random.default_rng(seed); a, b = A_(*a), A_(*b); d = b - a; L = np.hypot(*d); u = d / L; m = A_(-u[1], u[0]); t = np.linspace(0, 1, n)
    o = noise1(n, 46, r) * amp * 1.6 + noise1(n, 13, r) * amp * .75 + noise1(n, 3.2, r) * amp * .30
    s = noise1(n, 9, r) * amp * .4; return a + d * t[:, None] + m * o[:, None] + u * s[:, None]


def fibre_edge(E, side, seed=0, wid=(9, 26), h=H, w=W, sh=.8):
    """white torn paper core along polyline E, growing to `side` (+1 = left of travel on screen).  premult layer"""
    r = np.random.default_rng(seed + 5); n = len(E); T = np.gradient(E, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]] * side
    wv = wid[0] + (wid[1] - wid[0]) * np.clip(.5 + .45 * noise1(n, 16, r) + .18 * noise1(n, 4, r), .05, 1.25); O = E + N * wv[:, None]
    c = Cel(w, h, ss=2); body = np.vstack([E, O[::-1]]); c.fill(body, '#f6f3ec')
    c.fill(np.vstack([E, (E + N * (wv * .34)[:, None])[::-1]]), '#bdb7b4', .75); c.fill(np.vstack([E, (E + N * (wv * .14)[:, None])[::-1]]), '#8d8688', .7)
    for i in range(n * 5):                                                      # fibres
        j = int(r.integers(1, n - 1)); q = E[j] + N[j] * wv[j] * r.uniform(.2, 1.0) + T[j] * r.normal(0, 3); a = r.normal(0, .8); dv = N[j] * math.cos(a) + T[j] * math.sin(a); ln = r.uniform(3, 15) * (1.6 if r.random() < .08 else 1)
        c.line(np.array([q, q + dv * ln * .5 + T[j] * r.normal(0, 1.5), q + dv * ln + T[j] * r.normal(0, 3)]), r.uniform(.6, 1.5), '#ffffff' if r.random() < .7 else '#d4cec8', taper=(.1, .8), alpha=r.uniform(.5, 1.))
    for i in range(n):
        j = int(r.integers(0, n)); q = E[j] + N[j] * wv[j] * r.uniform(.25, .95); R_ = r.uniform(.6, 2.2); a = np.linspace(0, 6.283, 8); c.fill(np.c_[q[0] + np.cos(a) * R_ * 1.6, q[1] + np.sin(a) * R_], '#ffffff' if r.random() < .6 else '#cfc8c2', .8)
    return c.layer(h, w), O


def poly_mask(P, h=H, w=W, soft=0.):
    k = 2; m = np.zeros((h * k, w * k), np.uint8); cv2.fillPoly(m, [np.round(np.asarray(P, float) * k * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    m = cv2.resize(m, (w, h), interpolation=cv2.INTER_AREA).astype(f32) / 255.
    return blur(m, soft) if soft else m


def emblem_swords(cel, cx, cy, s, col, alpha=1., n=3):
    for a in ((-.62, .62, 1.571) if n == 3 else (-.62, .62)):
        c, q = math.cos(a), math.sin(a); d = A_(c, q); m = A_(-q, c)
        o = A_(cx, cy); L = np.array([o - d * s + m * s * .018, o + d * s * .50 + m * s * .05, o + d * s * .50 - m * s * .05, o - d * s * 1.08 - m * s * .03])
        cel.fill(L, col, alpha); cel.fill(np.array([o + d * s * .52 + m * s * .15, o + d * s * .58 + m * s * .15, o + d * s * .58 - m * s * .15, o + d * s * .52 - m * s * .15]), col, alpha)
        cel.fill(np.array([o + d * s * .60 + m * s * .045, o + d * s * .98 + m * s * .05, o + d * s * .98 - m * s * .05, o + d * s * .60 - m * s * .045]), col, alpha)


def emblem_skull(cel, cx, cy, s, col, hole, hat=True, alpha=1.):
    a = np.linspace(0, 6.283, 60)
    for q in (.72, -.72):
        c, n = math.cos(q), math.sin(q); d = A_(c, n); m = A_(-n, c); o = A_(cx, cy + s * .18); B = [o - d * s * 1.12 + m * s * .07, o + d * s * 1.12 + m * s * .07, o + d * s * 1.12 - m * s * .07, o - d * s * 1.12 - m * s * .07]
        cel.fill(np.array(B), col, alpha)
        for sg in (1, -1):
            for e in (.11, -.11): cel.fill(o + d * s * 1.14 * sg + m * s * e + np.c_[np.cos(a), np.sin(a)] * s * .13, col, alpha)
    cel.fill(np.c_[cx + np.cos(a) * s * .62, cy + np.sin(a) * s * .58], col, alpha); cel.fill(spl([(cx - s * .34, cy + s * .40), (cx - s * .32, cy + s * .80), (cx + s * .32, cy + s * .80), (cx + s * .34, cy + s * .40)], True, n=8), col, alpha)
    for sg in (1, -1): cel.fill(np.c_[cx + sg * s * .25 + np.cos(a) * s * .17, cy + s * .08 + np.sin(a) * s * .19], hole, alpha)
    cel.fill(np.array([(cx, cy + s * .28), (cx - s * .06, cy + s * .40), (cx + s * .06, cy + s * .40)]), hole, alpha)
    for k in range(-2, 3): cel.fill(np.array([(cx + k * s * .11 - s * .012, cy + s * .52), (cx + k * s * .11 + s * .012, cy + s * .52), (cx + k * s * .11 + s * .012, cy + s * .78), (cx + k * s * .11 - s * .012, cy + s * .78)]), hole, alpha)
    if hat:
        cel.fill(np.c_[cx + np.cos(a) * s * .98, cy - s * .24 + np.sin(a) * s * .17], col, alpha); cel.fill(spl([(cx - s * .56, cy - s * .26), (cx - s * .50, cy - s * .74), (cx, cy - s * .92), (cx + s * .50, cy - s * .74), (cx + s * .56, cy - s * .26)], True, (0, 4), n=8), col, alpha)
        cel.fill(np.array([(cx - s * .56, cy - s * .30), (cx - s * .55, cy - s * .44), (cx + s * .55, cy - s * .44), (cx + s * .56, cy - s * .30)]), hole, alpha)


def emblem_cutlery(cel, cx, cy, s, col, alpha=1., hole=None):
    def part(P, ang, sx=1):
        P = np.asarray(P, float) * [sx, 1]; c, q = math.cos(ang), math.sin(ang); return P @ np.array([[c, q], [-q, c]]) * s + [cx, cy]
    knife = spl([(-.03, -1.0), (.06, -.92), (.11, -.55), (.10, -.10), (.03, -.06), (.035, .2), (.06, .85), (.0, .98), (-.06, .85), (-.035, .2), (-.03, -.06)], True, (0, 4), n=6)
    cel.fill(part(knife, .72), col, alpha)
    fork = [spl([(-.035, -.10), (-.13, -.30), (-.14, -.55), (-.14, -1.0), (-.10, -1.0), (-.085, -.62), (-.055, -.60), (-.035, -1.02), (.0, -1.02), (.0, -.60)], False, (3, 4, 7, 8), n=5)]
    F = np.vstack([fork[0], (fork[0] * [-1, 1])[::-1]]); F = np.vstack([F, [(.035, -.10), (.035, .2), (.06, .85), (0, .98), (-.06, .85), (-.035, .2)]]); cel.fill(part(F, -.72), col, alpha)


# ------------------------------------------------------------------ bodies: cloth limbs with asymmetric profiles
def limb(P, wl, wr, n=10):
    """closed shape around a spline centre line; wl / wr = half widths sampled along it (left / right of travel on screen)"""
    C = resample(spl(np.asarray(P, float), n=n), 3.); k = len(C); t = np.linspace(0, 1, k); T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[T[:, 1], -T[:, 0]]
    f = lambda w: np.interp(t, np.linspace(0, 1, len(w)), w) if len(w) > 2 else np.linspace(w[0], w[-1], k)
    ker = np.hanning(9); ker /= ker.sum(); sm = lambda v: np.convolve(np.pad(v, 4, 'edge'), ker, 'valid')
    a, b = C + N * sm(f(wl))[:, None], C - N * sm(f(wr))[:, None]; return np.vstack([a, b[::-1]]), a, b, C


def cloth(cel, P, base, shade=None, sh=(), light=None, lt=(), ink=INK, w=3., folds=(), fw=1.8, seed=0, **k):
    """flat + clipped shadow shapes + light shapes + fold lines + ink"""
    cel.fill(P, base); cel.clip(P)
    for q in sh: cel.fill(np.asarray(q, float), shade)
    for q in lt: cel.fill(np.asarray(q, float), light)
    cel.unclip()
    for i, q in enumerate(folds): cel.line(spl(np.asarray(q, float), n=8) if len(q) > 2 else np.asarray(q, float), fw, ink, taper=(.08, .6), press=.35, seed=seed + i)
    outline(cel, P, w, ink, seed, **k); return cel


LIGHTS = dict(
    day=dict(key=-1, rims=[((8, 6), '#fff4d6', .9)], g=(('#fffaf0', '#e8cdb8')), L=(-.55, .6, .55)),
    card=dict(key=-1, rims=[((-9, 4), '#fff7d8', 1.), ((5, 5), '#ffe9c4', .45)], g=(('#fffaf2', '#dcb3a2')), L=(-.5, .55, .6)),
    sun=dict(key=-1, rims=[((9, 8), '#fff1b8', 1.)], g=(('#fff8e6', '#e6b99a')), L=(-.5, .7, .5)),
    fire=dict(key=-1, rims=[((9, -4), '#ffd27a', 1.), ((-6, 5), '#7a5cff', .25)], g=(('#ffe6c4', '#c8885a')), L=(-.7, -.1, .6)),
    flat=dict(key=-1, rims=[], g=None, L=(-.4, .6, .65)),
    over=dict(key=-1, rims=[], g=None, L=(-.3, .5, .8)))


def rim2(layer, dx, dy, col, amt=1.0, soft=.6):
    """kit.rim without false light along the picture border"""
    a = layer[..., 3]; h, w = a.shape; sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = np.clip(a - sh, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy(); out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt; return out


def light_layer(layer, name, cx, cy, s, boost=1., rims=None):
    """airbrush gradient + hot rims on a finished character layer"""
    D = LIGHTS[name] if isinstance(name, str) else name
    if D.get('g'): layer = para(layer, (cx - .5 * s, cy - 1.5 * s), (cx + .4 * s, cy + 2.2 * s), [(0, D['g'][0]), (1, D['g'][1])])
    k = max(.45, s / 300.)
    for (dx, dy), c, a in (rims if rims is not None else D['rims']):
        soft = rim2(layer, dx * k * 1.9, dy * k * 1.9, c, min(1., a * .34 * boost), soft=5 * k); layer = rim2(soft, dx * k, dy * k, c, min(1., a * .95 * boost), soft=.9)
    return layer
