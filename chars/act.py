"""Full-body action figures drawn as inked cels from a posed 3-D skeleton: Luffy (vest17 / grown), Sanji, Zoro, marines.

    A = Actor('luffy', coat='grown').pose(POSES['leap'])               # pose dict of op.fig_pose (any P(...))
    A.place((x, y, z), heading=0.)            feet frame on the ground    or  A.hang((x, y, z), R)   pelvis at the point, free rotation
    lay, ox, oy = A.draw(cam, f, light=(-.5,-.6,.6), expr='shout', rim=dict(col='#fff0c0', w=6), wind=(1,0,0))
    over_roi(img, lay, ox, oy);   A.j = joints px, A.S = skeleton
Limbs are anatomical tubes (front / back / outer / inner radii) -> silhouettes by union, ONE hard light shape, pen contour, edge hatching, hot rim.
rubber(cel, view, P3, rad, ...) draws a stretched arm as a tapering tube with travelling ripples.  bigfist(...) = hero fist (op.hands)."""
import numpy as np, math, cv2
from op import W, H as FH
from op.paint import rgb, f32
from op.ink import Cel, spl, resample, over_roi, noise1
from op import kit
from op.kit import outline, spike
from op.fig_pose import Skel, P, POSES, mix_pose, add_pose, mirror, seq, sprint, cycle, unit, Rx, Ry, Rz, STAND
from op.head3 import Head

INK = '#17131a'
PI = math.pi
ANG = np.linspace(0, 2 * PI, 18, endpoint=False)


def M(base, shade, deep=None, hi=None):
    return dict(base=base, shade=shade, deep=deep or shade, hi=hi or base)


SKIN = M('#f0b98d', '#c9805c', '#9a5646', '#ffe0bd')
TAN = M('#d9986a', '#a9643f', '#7c432f', '#f6c597')
FAIR = M('#f5c9a0', '#cf8d69', '#9d5d4c', '#ffe8cc')
MAT = dict(
    red=M('#d0302b', '#8d1a24', '#5f1020', '#ef5a44'), blue=M('#3a66b8', '#234083', '#182b60', '#6f95d8'), fur=M('#f4f0e6', '#b9b3c4', '#8f88a4', '#ffffff'),
    yellow=M('#f2bf3a', '#b97a1c', '#8a5414', '#ffe27a'), straw=M('#e3b557', '#a9742c', '#7b4f1f', '#f8da8c'), ribbon=M('#b3202a', '#6f1220', '#4c0c18', '#d84a42'),
    sandal=M('#c08a4a', '#7d5228', '#58381c', '#e0b070'), black=M('#2b2a35', '#16151d', '#0f0e14', '#5a5c78'), suit=M('#2c2b38', '#17161f', '#0f0e15', '#565876'),
    shirt=M('#e9b93c', '#a8741f', '#7d5216', '#ffdd73'), green=M('#2f6b48', '#173f2c', '#0f2b20', '#55946a'), sash=M('#b3262c', '#70141f', '#4a0d18', '#d94b44'),
    boot=M('#3a2c2a', '#1e1618', '#140f12', '#6a5148'), steel=M('#eef3fa', '#a3afc4', '#6c7890', '#ffffff'), navy=M('#2e4f8f', '#1d3263', '#132347', '#5878b8'),
    white=M('#f3f5f7', '#b3bcd0', '#8791ad', '#ffffff'), hairk=M('#24222b', '#141218', '#0e0c12', '#5d6486'), hairg=M('#5fae4e', '#33773a', '#1f5230', '#9bdc72'),
    hairb=M('#f0cb52', '#b98a26', '#8a6118', '#fff0a0'), haki=M('#33263c', '#150f1b', '#0c080f', '#7a5a92'), nail=M('#ffe9d2', '#e0a584'))

BODY = dict(
    luffy=dict(H=1.74, heads=6.15, sw=.100, hw=.038, arm=.168, fore=.158, hipz=.535, kneez=.292, waistz=.615, limb=.90, foot=1.05),
    sanji=dict(H=1.80, heads=6.5, sw=.104, hw=.040, hipz=.555, kneez=.302, waistz=.625, limb=.96, foot=1.0),
    zoro=dict(H=1.81, heads=6.4, sw=.112, hw=.043, limb=1.12, foot=1.05),
    marine=dict(H=1.72, heads=5.6, sw=.102, hw=.042, limb=1.0, foot=1.0))
T_THIGH = [(0, .050, .054, .050, .048), (.2, .057, .052, .052, .050), (.45, .053, .046, .047, .044), (.72, .045, .040, .040, .038), (.92, .041, .036, .037, .037), (1, .039, .034, .036, .036)]
T_CALF = [(0, .038, .034, .036, .036), (.10, .037, .041, .038, .039), (.30, .032, .050, .041, .042), (.52, .028, .043, .036, .034), (.78, .025, .030, .027, .025), (.94, .024, .025, .023, .022), (1, .024, .026, .024, .024)]
T_UARM = [(-.06, .030, .030, .034, .022), (.10, .038, .037, .043, .032), (.34, .038, .036, .037, .032), (.62, .036, .035, .030, .030), (.9, .029, .028, .028, .028), (1, .028, .028, .028, .028)]
T_FARM = [(0, .028, .028, .028, .028), (.16, .035, .032, .033, .032), (.36, .033, .030, .031, .029), (.72, .025, .023, .024, .022), (.95, .020, .019, .022, .020), (1.02, .020, .019, .022, .020)]


def lerp(a, b, t): return a + (b - a) * t


class View:
    def __init__(self, cam):
        self.cam = cam; self.right, self.down, self.fwd = cam.R[0], cam.R[1], cam.R[2]

    def __call__(self, P3):
        c = self.cam.cam(P3); z = np.maximum(c[..., 2], .06); f = self.cam.f
        return np.clip(np.stack([c[..., 0] / z * f + self.cam.cx, c[..., 1] / z * f + self.cam.cy], -1), -7000, 9000)

    def z(self, P3): return self.cam.cam(P3)[..., 2]

    def s(self, P3): return self.cam.f / max(float(self.z(P3)), .06)

    def l2(self, L3):
        v = np.array([self.right @ L3, self.down @ L3]); return v / (np.hypot(*v) + 1e-6)


def hull(p): return cv2.convexHull(np.asarray(p, np.float32))[:, 0].astype(float)


def union(polys, soft=1.1, up=2, pad=6, smooth=7, min_len=10):
    """union of (convex) polygons -> smooth contours (screen px), clipped a little outside the frame"""
    if not polys: return []
    A = np.vstack(polys); lo = np.maximum(np.floor(A.min(0)) - pad, (-160, -160)); hi = np.minimum(np.ceil(A.max(0)) + pad, (W + 160, FH + 160))
    w, h = ((hi - lo) * up).astype(int)
    if w < 4 or h < 4: return []
    m = np.zeros((h, w), np.uint8)
    for p in polys: cv2.fillPoly(m, [np.round((np.clip(p, -3000, 5000) - lo) * up * 16).astype(np.int32)], 255, cv2.LINE_8, 4)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft * up)
    cs, _ = cv2.findContours((m > 127).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        c = c[:, 0].astype(float)
        if len(c) < min_len * up: continue
        k = smooth * up // 2 * 2 + 1; k = min(k, len(c) // 3 * 2 - 1)
        if k >= 5:
            kr = np.hanning(k + 2)[1:-1]; kr /= kr.sum(); g = k // 2
            cc = np.vstack([c[-g:], c, c[:g]]); c = np.stack([np.convolve(cc[:, i], kr, 'valid') for i in (0, 1)], -1)
        out.append(c[::2] / up + lo)
    return out


class Tube:
    """stations: C (n,3), F front, O outer unit vectors (n,3), rad (n,4) front back outer inner, metres"""
    def __init__(self, v, C, F, O, rad, n=18):
        self.v = v; self.C, self.F, self.O, self.rad = [np.asarray(q, float) for q in (C, F, O, rad)]
        cs, sn = np.cos(ANG), np.sin(ANG); r = self.rad
        A = np.where(cs[None] > 0, r[:, :1], r[:, 1:2]) * cs[None]; B = np.where(sn[None] > 0, r[:, 2:3], r[:, 3:4]) * sn[None]
        self.R3 = self.C[:, None] + A[..., None] * self.F[:, None] + B[..., None] * self.O[:, None]
        self.px = v(self.R3); self.c2 = v(self.C); self.r2 = np.hypot(*np.moveaxis(self.px - self.c2[:, None], -1, 0)).mean(1)
        self.z = v.z(self.C)

    def hulls(self, k=1., off=(0, 0), amt=0., mod=None):
        q = self.px
        if k != 1 or amt:
            m = np.ones(len(q)) if mod is None else mod
            q = self.c2[:, None] + (q - self.c2[:, None]) * k + (np.asarray(off) * amt)[None, None] * (self.r2 * m)[:, None, None]
        ok = self.z > .05
        return [hull(np.vstack([q[i], q[i + 1]])) for i in range(len(q) - 1) if ok[i] and ok[i + 1]]

    def surf(self, t, ph):
        """(t in station index units 0..n-1, phi degrees: 0 front, 90 outer) -> 3-D points, facing cosine"""
        t = np.clip(np.asarray(t, float), 0, len(self.C) - 1.001); i = t.astype(int); u = (t - i)[:, None]; ph = np.radians(np.asarray(ph, float)) * np.ones(len(t))
        mix = lambda A: A[i] * (1 - u) + A[i + 1] * u
        C, F, O, r = mix(self.C), mix(self.F), mix(self.O), mix(self.rad); cs, sn = np.cos(ph), np.sin(ph)
        rf = np.where(cs > 0, r[:, 0], r[:, 1]); ro = np.where(sn > 0, r[:, 2], r[:, 3])
        Q = C + (cs * rf)[:, None] * F + (sn * ro)[:, None] * O; N = (cs / rf)[:, None] * F + (sn / ro)[:, None] * O
        N /= np.linalg.norm(N, axis=1)[:, None]; e = self.v.cam.eye - Q; e /= np.linalg.norm(e, axis=1)[:, None]
        return Q, (N * e).sum(1)

    def band(self, t0, t1, ph0, ph1, nt=14, nph=16):
        """front-facing quads of the patch ph0(t)..ph1(t) (degrees, functions or numbers)"""
        fn = lambda q: q if callable(q) else (lambda t, q=q: q)
        ph0, ph1 = fn(ph0), fn(ph1); T = np.linspace(t0, t1, nt); G = []; K = []
        for t in T:
            a = np.linspace(ph0(t), ph1(t), nph); Q, k = self.surf(np.full(nph, t), a); G.append(self.v(Q)); K.append(k)
        G, K = np.array(G), np.array(K); out = []
        for i in range(nt - 1):
            for j in range(nph - 1):
                if (K[i, j] + K[i + 1, j + 1]) > -.04: out.append(hull(np.array([G[i, j], G[i, j + 1], G[i + 1, j + 1], G[i + 1, j]])))
        return out


def joint_tube(v, a, b, front, out, table, k, tmin=None, tmax=None):
    d = unit(b - a); f = unit(front - d * (front @ d)); o = np.cross(d, f); o = o * (1 if o @ out >= 0 else -1)
    T = np.array(table, float)
    if tmax is not None:
        tt = np.linspace(T[0, 0] if tmin is None else tmin, tmax, 7); T = np.column_stack([tt] + [np.interp(tt, T[:, 0], T[:, i]) for i in range(1, 5)])
    C = a + (b - a) * T[:, :1]; n = len(T)
    return Tube(v, C, np.repeat(f[None], n, 0), np.repeat(o[None], n, 0), T[:, 1:] * k)


def runs(ok):
    out = []; i = 0; n = len(ok)
    while i < n:
        if ok[i]:
            j = i
            while j + 1 < n and ok[j + 1]: j += 1
            out.append((i, j)); i = j + 1
        else: i += 1
    return out


class Pen:
    """a Cel ROI with light direction and line scale"""
    def __init__(self, cel, v, L2, w, seed=0, ink=INK):
        self.c, self.v, self.L, self.w, self.seed, self.ink = cel, v, np.asarray(L2, float), w, seed, ink

    def part(self, tubes, mat, lk=.80, lo=.34, w=None, hi=0., edge=True, soft=.9, hatch=0, mods=None, extra=(), seed=0, breaks=1, deep=False):
        polys = [h for t in tubes for h in t.hulls()] + list(extra); C = union(polys, soft)
        if not C: return []
        self.c.fill(C, mat['shade'], evenodd=True)
        lit = union([h for j, t in enumerate(tubes) for h in t.hulls(lk, self.L, lo, None if mods is None else mods[j])] + [e + self.L * 2 for e in extra], soft * .7, smooth=4)
        self.c.clip(C)
        if lit: self.c.fill(lit, mat['base'], evenodd=True)
        if hi: self.c.fill(union([h for t in tubes for h in t.hulls(.34, self.L, .50)], .7, smooth=4), mat['hi'], hi, evenodd=True)
        if hatch and lit: self.edge_hatch(lit, hatch, seed)
        self.c.unclip()
        if edge: self.contour(C, w, seed, breaks)
        return C

    def contour(self, C, w=None, seed=0, breaks=1):
        for i, c in enumerate(C): outline(self.c, c, self.w if w is None else w, self.ink, self.seed * 13 + seed * 7 + i, press=.5, breaks=breaks, tremor=.2)

    def patch(self, polys, mat, w=None, soft=.8, shade=None, seed=0, breaks=1, clip=None, edge=True):
        C = union(polys, soft) if len(polys) and np.ndim(polys[0]) == 2 else polys
        if not C: return []
        if clip: self.c.clip(clip)
        self.c.fill(C, mat['base'], evenodd=True)
        if shade:
            S = union(shade, soft * .7, smooth=4)
            if S: self.c.clip(C); self.c.fill(S, mat['shade'], evenodd=True); self.c.unclip()
        if edge: self.contour(C, w, seed, breaks)
        if clip: self.c.unclip()
        return C

    def line3(self, Q, ok=None, w=None, col=None, taper=(.2, .4), seed=0):
        p = self.v(Q); ok = np.ones(len(p), bool) if ok is None else ok
        for a, b in runs(ok):
            if b - a >= 2: self.c.line(p[a:b + 1], self.w * .55 if w is None else w, col or self.ink, taper=taper, press=.4, seed=seed + a)

    def surf(self, tube, pts, w=None, seed=0, thr=.08, col=None):
        """pen line through (t, phi) points on a tube"""
        q = spl(np.asarray(pts, float), n=7) if len(pts) > 2 else np.linspace(pts[0], pts[1], 9); Q, k = tube.surf(q[:, 0], q[:, 1]); self.line3(Q, k > thr, w, col, seed=seed)

    def edge_hatch(self, lit, n, seed):
        r = np.random.default_rng(seed + 77)
        for c in lit:
            if len(c) < 24: continue
            T = np.roll(c, -2, 0) - np.roll(c, 2, 0); N = np.c_[T[:, 1], -T[:, 0]]; N /= np.hypot(*N.T)[:, None] + 1e-6
            if ((c + N * 3 - c.mean(0)) ** 2).sum() < ((c - c.mean(0)) ** 2).sum(): N = -N
            idx = np.nonzero(N @ self.L < -.35)[0]
            if len(idx) < 12: continue
            for _ in range(n):
                i0 = int(r.choice(idx)); m = r.integers(5, 9); ln = self.w * r.uniform(2.6, 4.4)
                for k in range(m):
                    i = (i0 + int(k * max(1, self.w * .8))) % len(c)
                    if N[i] @ self.L > -.2: continue
                    a = c[i] - N[i] * ln * (.6 + .5 * math.sin(PI * (k + .5) / m)) + T[i] * .3; self.c.line(np.array([c[i] + N[i] * self.w * .4, (c[i] + a) / 2 + T[i] * .12, a]), self.w * .42, self.ink, taper=(.05, .8), seed=int(r.integers(1 << 20)))


# ------------------------------------------------------------------ cloth strip (coat tails, sash, scarf)
def strip_grid(v, roots, length, wind, f, seed=0, n=8, grav=1.0, wave=.35, hz=2.2, down=None):
    """roots (m,3) hang points -> (n+1, m, 3) grid blown by wind (world vec, magnitude ~0..3)"""
    roots = np.asarray(roots, float); m = len(roots); G = [roots]; g = np.array([0, 0, -1.]) * grav if down is None else np.asarray(down, float); wind = np.asarray(wind, float); t = f / 24.
    side = unit(roots[-1] - roots[0] + 1e-6); wn = np.linalg.norm(wind); perp = unit(np.cross(g + wind + 1e-6, side))
    for k in range(n):
        u = (k + 1) / n; ph = t * hz * 2 * PI - u * 4.2 + seed + np.arange(m) * .55
        d = g[None] * (1 - .25 * u) + wind[None] * (.35 + .65 * u) + perp[None] * (np.sin(ph) * wave * (.2 + u) * (.4 + min(wn, 1.6)))[:, None] + side[None] * ((np.arange(m) - (m - 1) / 2) * .06)[:, None]
        G.append(G[-1] + d / np.linalg.norm(d, axis=1)[:, None] * length / n)
    return np.array(G)


def draw_strip(pen, G, mat, w=None, seed=0, hem=None, notch=0.):
    """cloth grid -> inked cloth: panels alternate light / shade by the direction they face"""
    v = pen.v; p = v(G); n, m = p.shape[:2]; q = []; sh = []; inner = []
    for i in range(n - 1):
        for j in range(m - 1):
            quad = np.array([p[i, j], p[i, j + 1], p[i + 1, j + 1], p[i + 1, j]]); hq = hull(quad); q.append(hq)
            N = np.cross(G[i, j + 1] - G[i, j], G[i + 1, j] - G[i, j]); N /= np.linalg.norm(N) + 1e-9; e = v.cam.eye - G[i, j]
            L3 = v.right * pen.L[0] + v.down * pen.L[1] - v.fwd * .5; s = (N @ L3) * (1 if N @ e > 0 else -1)
            if s < .12 + .25 * math.sin(j * 2.1 + seed + i * .5): sh.append(hq)
    C = pen.patch(q, mat, w, shade=sh, seed=seed)
    for j in range(1, m - 1):
        if m > 3: pen.c.line(spl(p[1 + (j + seed) % 2:-1 - (j % 2), j], n=5), pen.w * .5, pen.ink, taper=(.1, .5), seed=seed + j)
    return C


# ------------------------------------------------------------------ small heads
HP = dict(luffy=dict(a=[(-1.0, .12), (-.95, .25), (-.84, .40), (-.70, .53), (-.52, .63), (-.30, .70), (-.08, .735), (.12, .74), (.35, .73), (.6, .69), (.8, .55), (.93, .34), (1.0, .05)], nose=(.085, -.10, -.36, -.41, .13)),
          sanji=dict(nose=(.09, -.06, -.42, -.48, .16)), zoro=dict(a=[(-1.0, .17), (-.95, .28), (-.84, .42), (-.70, .53), (-.52, .61), (-.30, .67), (-.08, .70), (.12, .715), (.35, .72), (.6, .69), (.8, .55), (.93, .34), (1.0, .05)], nose=(.10, -.06, -.42, -.48, .16)), marine={})
SKINOF = dict(luffy=SKIN, sanji=FAIR, zoro=TAN, marine=FAIR)


def _hair(who):
    """(theta deg from front (+ = viewer right of a facing head), y, length, dir(x,y,z), width, front?)"""
    r = np.random.default_rng(dict(luffy=3, sanji=5, zoro=7, marine=9)[who]); S = []
    if who == 'luffy':
        for th, y, ln, dx, dy, w in [(-58, .46, .62, -.28, -1, .34), (-38, .60, .58, -.10, -1, .30), (-18, .68, .50, .12, -1, .28), (2, .70, .54, -.10, -1, .28), (22, .66, .50, .05, -1, .28), (42, .58, .60, .22, -1, .30), (60, .44, .62, .34, -1, .34)]: S.append((th, y, ln, (dx, dy, .18), w, 1))
        for k in range(15):
            th = 75 + k * 14; y = .55 - .55 * (k % 3) / 2; a = math.radians(th); S.append((th, y, r.uniform(.5, .72), (math.sin(a) * .75, -.62 - .3 * r.random(), math.cos(a) * .75), .36, 0))
        for k in range(9): th = k * 40 + 10; a = math.radians(th); S.append((th, .80, .42, (math.sin(a), .25 * r.random(), math.cos(a)), .34, 0))
    elif who == 'zoro':
        for k in range(34):
            th = k * 137.5 % 360; y = .98 - .95 * ((k * .618) % 1) * (1 if 65 < th < 295 else .40); a = math.radians(th); S.append((th, y, .40 + .08 * (k % 3), (math.sin(a) * .8, .60, math.cos(a) * .8), .27, 1 if th < 80 or th > 280 else 0))
        for th in (-52, -34, -16, 0, 16, 34, 52): a = math.radians(th); S.append((th, .60, .40, (math.sin(a) * .5, .75, .6), .24, 1))
    elif who == 'sanji':
        for th, y, ln, dx, dy, w in [(8, .88, 1.05, .30, -1, .40), (28, .84, 1.20, .16, -1, .40), (46, .74, 1.15, .10, -1, .38), (64, .62, 1.05, .02, -1, .36), (-12, .90, .52, -.65, -.9, .34), (-34, .82, .72, -.55, -1, .34), (-58, .62, .85, -.30, -1, .34)]: S.append((th, y, ln, (dx, dy, .22), w, 1))
        for k in range(14): th = 80 + k * 14.5; a = math.radians(th); S.append((th, .30 - .25 * (k % 2), .66, (math.sin(a) * .35, -1, math.cos(a) * .35), .36, 0))
    else:
        for k in range(12): th = 80 + k * 17; a = math.radians(th); S.append((th, .2, .4, (math.sin(a) * .5, -.8, math.cos(a) * .5), .3, 0))
    return S


HAIR = {k: _hair(k) for k in BODY}


def _skullpt(th, y, k=1.06):
    a = math.radians(th); yy = min(max(y, -.6), .99); r = math.sqrt(max(0., 1 - max(0., (yy - .25) / .80) ** 2)); return np.array([math.sin(a) * .76 * r * k, yy * k, (math.cos(a) * (.80 if math.cos(a) > 0 else .92)) * r * k])


def small_head(pen, S, who, expr='calm', hat=False, look=(0, 0), f=0, wind=(0, 0, 0), puff=0., sword=False, hairfly=.0, hatpos=None):
    """action-size head from the skeleton.  expr: calm grit shout grin pout blank shock"""
    v, c = pen.v, pen.c; cam = v.cam; o = S.head; e = unit(cam.eye - o); xv = unit(v.right - e * (v.right @ e)); yv = np.cross(e, xv)
    hx, hy, hz = -S.Rh[:, 0], S.Rh[:, 2], S.Rh[:, 1]; s = S.u * v.s(o); cx, cy = v(o)
    h = Head(**HP[who]).pose(cx, cy, s, 0, 0, 0, persp=min(.22, S.u / max(v.z(o), .3) * .9)); R = np.array([[q @ a for a in (hx, hy, hz)] for q in (xv, yv, e)]); h.R = R
    h.yaw = math.atan2(R[0, 2], R[2, 2]); h.pitch = math.asin(np.clip(R[1, 2], -1, 1)); h.roll = -math.atan2(R[1, 0], R[0, 0]) if R[2, 2] > 0 else 0.
    if s < 9:
        a_ = np.c_[np.cos(ANG), np.sin(ANG)] * s; c.fill(np.array([cx, cy]) + a_ * 1.08, MAT['white']['base'] if who == 'marine' else MAT[dict(luffy='straw', sanji='hairb', zoro='hairg')[who]]['base']); c.fill(np.array([cx, cy + s * .35]) + a_ * .8, SKINOF[who]['base']); return h
    w = pen.w; mat = SKINOF[who]; hm = MAT[dict(luffy='hairk', sanji='hairb', zoro='hairg', marine='hairk')[who]]; face_on = R[2, 2] > -.20
    Lh = np.array([pen.L[0], -pen.L[1], .45]); Lh = Lh / np.linalg.norm(Lh)
    wl = np.array([np.asarray(wind, float) @ a for a in (hx, hy, hz)]) * hairfly; t = f / 24.
    def spikes(front):
        A = []
        for k, (th, y, ln, d, wd, fr) in enumerate(HAIR[who]):
            root = _skullpt(th, y); n = root / np.linalg.norm(root); dd = np.asarray(d, float) + wl * (.6 + .4 * math.sin(t * 9 + k * 1.7)) + n * .10; dd /= np.linalg.norm(dd); tip = root + dd * ln
            sd = np.cross(dd, n); sd = sd / (np.linalg.norm(sd) + 1e-6) * wd / 2; zz = h.depth((root + tip) / 2)
            vis = (fr and R[2, 2] > -.2) or zz > .05
            if bool(vis) == bool(front): p = h.P3(np.stack([root - sd, tip, root + sd])); A.append((zz, spike(p[0], p[1], p[2], .16, .05 * math.sin(k * 2.1 + t * 5)), k))
        return sorted(A, key=lambda q: q[0])
    cap = []
    for th in range(0, 360, 15):
        for y in np.linspace(-.35 if 90 < th < 270 else (.18 if 60 < th < 300 else .56), 1., 6): cap.append(_skullpt(th, y, 1.08))
    cap = np.array(cap); zc = h.depth(cap); back = hull(h.P3(cap)); capf = cap[(zc > -.15)]; frontc = hull(h.P3(capf)) if len(capf) > 3 else back
    # ---- behind
    bs = spikes(False)
    def ink_hair(cont, sp, wk=.8):
        C = union([cont] + [q for _, q, _ in sp], .5, smooth=3)
        if not C: return
        c.fill(C, hm['shade'], evenodd=True); c.clip(C); c.fill([q + pen.L * s * .10 for q in C], hm['base'], evenodd=True); c.unclip(); pen.contour(C, w * wk, 3, 0)
        for zz, q, k in sp:
            if k % 2 == 0: c.line(q[:len(q) // 2] * .86 + q[len(q) // 2] * .14, w * .34, hm['deep'] if who != 'luffy' else '#3b3a4c', taper=(.1, .6), seed=k)
    ink_hair(back if not hat else hull(h.P3(cap[cap[:, 1] < .7])), bs)
    if face_on:
        for sg in (-1, 1):
            er = h.ear(sg)
            if er.get('vis', 0) > 0: kit.shape(c, er['outer'], mat['base'], w * .75); c.line(er['inner'], w * .4, pen.ink)
        F = h.face(); c.fill(F, mat['base'])
        # one hard shadow: side away from the light + under brim / fringe + under chin
        c.clip(F); sx = -1 if pen.L[0] > 0 else 1
        sh = h.P([(sx * .90, 1.), (sx * .42, .70), (sx * .36, .12), (sx * .50, -.12), (sx * .30, -.34), (sx * .36, -.62), (sx * .10, -.92), (sx * .2, -1.2), (sx * 1.2, -1.)]); c.fill(sh, mat['shade'])
        if hat or who == 'luffy': c.fill(h.P([(-.9, 1.), (-.9, .40), (-.55, .30), (-.2, .38), (.1, .28), (.4, .38), (.9, .34), (.9, 1.)]), mat['shade'])
        c.unclip(); outline(c, F, w, pen.ink, 5, press=.5)
        _features(pen, h, who, expr, s, w, mat, look, f, puff, sword)
        ink_hair(frontc if who != 'marine' else frontc, spikes(True), .75) if not hat else [ink_hair(h.P3(np.array([_skullpt(th, y, 1.08) for th in range(-85, 86, 17) for y in (.62, .8)]))[:3], spikes(True), .7)]
    else:
        ink_hair(back, spikes(True))
    if hat: draw_hat(pen, h, s, w, f, hatpos)
    if who == 'zoro' and True:                                      # three gold drops, left ear (+x in front coordinates)
        for k in range(3):
            q = np.array([.735, -.20, .02 - .07 * k]); n = h.depth(q)
            if n > -.25: a = h.P3(q); sw = math.sin(t * 11 + k) * .05; b = h.P3(q + np.array([.03, -.26, sw])); c.line(np.array([a, (a + b) / 2, b]), w * 1.5, pen.ink, taper=(0, 0)); c.line(np.array([a, (a + b) / 2, b]), w * .9, '#f0c24a', taper=(0, 0))
    if who == 'marine':
        pts = np.array([_skullpt(th, y, 1.14) for th in range(0, 360, 20) for y in (.45, .7, .95)]); pen.patch([hull(h.P3(pts))], MAT['white'], w * .9); br = np.array([(math.sin(a) * .86, .46, math.cos(a) * (1.25 if math.cos(a) > 0 else .9)) for a in np.linspace(-1.3, 1.3, 11)]); pen.patch([hull(h.P3(br))], MAT['navy'], w * .7)
    return h


def _features(pen, h, who, expr, s, w, mat, look, f, puff, sword):
    c = pen.c; ink = pen.ink; P_ = h.P; line = lambda q, ww, **k: c.line(P_(spl(np.asarray(q, float), n=6)), ww, ink, **k)
    sharp = expr in ('grit', 'shout', 'glare'); R = h.R
    for sg in (-1, 1):
        ex = .31 * sg
        if h.facing(ex * .95, 0.) < .10: continue
        if who == 'sanji' and sg > 0: continue                       # hair covers it
        if who == 'zoro' and sg > 0:
            line([(ex - .17, -.03), (ex, -.06), (ex + .16, -.02)], w * .9); line([(ex + .0, .30), (ex + .03, .05), (ex + .01, -.28)], w * .6); line([(ex - .2, .16), (ex, .17), (ex + .2, .26)], w * 1.5); continue
        rx, ry = (.185, .205) if who == 'luffy' else (.16, .095) if who == 'sanji' else (.17, .10) if who == 'zoro' else (.14, .15)
        if expr in ('shock', 'blank'): rx, ry = max(rx, .17) * 1.1, max(ry, .17) * 1.12
        top = .10 * (1 if sharp else 0); a = np.linspace(0, 2 * PI, 22, endpoint=False); E = np.c_[ex + np.cos(a) * rx, np.sin(a) * ry + .01]
        if sharp or who in ('zoro', 'sanji'):                         # lid cuts down toward the nose
            lim = .5 * ry + (E[:, 0] - ex) * sg * .42 * (1 if who == 'luffy' else .55); E[:, 1] = np.minimum(E[:, 1], lim + .02)
        if expr == 'grin' and who == 'luffy' : pass
        q = P_(E); c.fill(q, '#fdfcf6'); c.clip(q)
        c.fill(P_(np.c_[E[:, 0], E[:, 1]][E[:, 1] > ry * .25]) if (E[:, 1] > ry * .25).sum() > 2 else q[:3], '#cfc9d8', .55); c.unclip()
        u = E[np.argsort(E[:, 0])]; up = u[u[:, 1] > .0 * ry]
        if len(up) > 2: c.line(P_(spl(up, n=3)), w * 1.15, ink, taper=(.15, .15), press=.3)
        outline(c, q, w * .55, ink, 9 + sg)
        pr = (.052 if who == 'luffy' else .040) * (.62 if expr == 'shock' else 1); px, py = ex + look[0] * rx * .45 - sg * .02, look[1] * ry * .5 - (.01 if not sharp else .03); c.fill(P_(np.c_[px + np.cos(a) * pr, py + np.sin(a) * pr * 1.15]), ink)
        bh = .30 if who == 'luffy' else .21; tl = (.13 if sharp else .06 if who == 'zoro' else -.05 if expr in ('shock', 'blank') else .02)
        if who == 'sanji': line([(ex - sg * -.20, bh + .02), (ex, bh + .05), (ex - sg * .16, bh - .03)], w * 1.1); th = np.linspace(0, 7.2, 22); line(np.c_[ex + sg * (.23) + np.cos(th) * (.085 - .010 * th), bh + .02 + np.sin(th) * (.085 - .010 * th)], w * .6)
        else: line([(ex + sg * .20, bh + .02 + tl * .35), (ex + sg * .02, bh + .05 - tl * .2), (ex - sg * .18, bh - tl * 1.15 + .01)], w * 1.35, taper=(.3, .2))
        if who == 'luffy' and sg > 0:                                 # scar: one line, two stitches
            line([(ex - .13, -.285), (ex, -.31), (ex + .14, -.285)], w * .62); line([(ex - .05, -.25), (ex - .045, -.345)], w * .5); line([(ex + .055, -.25), (ex + .06, -.345)], w * .5)
    if sharp and R[2, 2] > .3: line([(-.07, .16), (-.03, .06)], w * .5); line([(.07, .16), (.03, .06)], w * .5)
    np_ = h.nose_profile()
    if abs(h.yaw) < 1.05: c.line(h.P3(np_[2:7]), w * .75, ink, taper=(.4, .2), press=.4)
    if puff > 0:
        for sg in (-1, 1): q = P_(spl(np.c_[sg * .42 + np.cos(np.linspace(0, 2 * PI, 14)) * .22 * puff, -.42 + np.sin(np.linspace(0, 2 * PI, 14)) * .24 * puff])); c.fill(q, '#f2a98a', .7); c.line(q[3:11], w * .5, ink)
    teeth = '#fbf8ee'
    def mouth(top, bot, kind):
        m = np.vstack([spl(top, n=6), spl(bot, n=6)[::-1]]); q = P_(m)
        if h.facing(0, -.6) < .02: return
        c.fill(q, '#5a1420' if kind != 'teeth' else teeth); c.clip(q)
        if kind == 'open':
            c.fill(P_(np.vstack([spl(top, n=6), (spl(top, n=6) - np.array([0, .085]))[::-1]])), teeth); c.line(P_(spl(top, n=6) - np.array([0, .085])), w * .45, ink, taper=(0, 0))
            b = spl(bot, n=6); c.fill(P_(np.vstack([b * [.62, 1] + [0, .0], (b * [.62, 1] + [0, .11])[::-1]])), '#d8625e')
        if kind == 'teeth':
            t_ = spl(top, n=6); b_ = spl(bot, n=6); c.line(P_((t_ + b_[np.linspace(0, len(b_) - 1, len(t_)).astype(int)]) / 2), w * .5, ink, taper=(0, 0))
            for x in np.linspace(top[0][0], top[-1][0], 9)[1:-1]: c.line(P_([(x, -.2), (x, -.9)]), w * .32, ink, taper=(0, 0))
        c.unclip(); outline(c, q, w * .8, ink, 21)
    if expr == 'shout': mouth([(-.36, -.46), (0, -.42), (.36, -.46)], [(-.36, -.46), (-.27, -.80), (0, -.88), (.27, -.80), (.36, -.46)], 'open')
    elif expr == 'grin': mouth([(-.50, -.40), (-.2, -.47), (.2, -.47), (.50, -.40)], [(-.50, -.40), (-.36, -.66), (0, -.78), (.36, -.66), (.50, -.40)], 'teeth' if who == 'luffy' else 'teeth')
    elif expr == 'grit': mouth([(-.34, -.56), (0, -.50), (.36, -.50)], [(-.34, -.56), (-.24, -.70), (.1, -.74), (.36, -.50)], 'teeth')
    elif expr == 'pout': c.fill(P_(spl(np.c_[np.cos(np.linspace(0, 2 * PI, 12)) * .07, -.62 + np.sin(np.linspace(0, 2 * PI, 12)) * .06])), '#8c3a34'); line([(-.10, -.60), (0, -.55), (.10, -.60)], w * .7)
    elif expr in ('shock',): mouth([(-.16, -.52), (0, -.48), (.16, -.52)], [(-.16, -.52), (-.15, -.84), (0, -.90), (.15, -.84), (.16, -.52)], 'open')
    elif not sword: line([(-.17, -.63), (0, -.645 + (.02 if who == 'sanji' else 0)), (.19, -.60)], w * .85)
    if who == 'sanji':
        a = P_([(.10, -.63)])[0]; b = h.P3(np.array([.30, -.70, 1.12])); c.line(np.array([a, (a + b) / 2, b]), w * 1.7, ink, taper=(0, 0)); c.line(np.array([a, (a + b) / 2, b]), w * 1.05, '#f6f2ea', taper=(0, 0)); c.line(np.array([a * .12 + b * .88, b, b + (b - a) * .02]), w * 1.1, '#ff8a3a', taper=(0, 0))
        line([(-.05, -.86), (0, -.97), (.05, -.86)], w * .8); c.fill(P_([(-.06, -.84), (.06, -.84), (.02, -.99), (-.02, -.99)]), '#8a6118')
    if sword:                                                         # white hilt across the teeth
        hy = h.P3(np.array([[.95, -.60, .62], [-.35, -.60, .78]])); c.line(hy, w * 3.4, ink, taper=(0, 0)); c.line(hy, w * 2.5, '#f5f2e8', taper=(0, 0))
        for k in range(6): q = hy[0] * (1 - k / 5.5) + hy[1] * (k / 5.5); c.fill(np.array([q + [0, -w * .9], q + [w, 0], q + [0, w * .9], q + [-w, 0]]), '#b9b4c6')
        g = h.P3(np.array([[-.36, -.44, .78], [-.38, -.76, .78]])); c.line(g, w * 2.2, ink, taper=(0, 0)); c.line(g, w * 1.2, '#e6c04a', taper=(0, 0)); b = h.P3(np.array([[-.38, -.60, .78], [-2.2, -.57, .50], [-3.6, -.46, .1]])); blade(c, b, w * 2.2, w)


def blade(c, pts, bw, w, flash=0.):
    """sword blade: polyline hilt -> tip"""
    p = spl(np.asarray(pts, float), n=10); t = np.linspace(0, 1, len(p)); T = np.gradient(p, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; k = bw * np.clip((1.02 - t) * 9, 0, 1)[:, None]
    q = np.vstack([p + N * k * .55, (p - N * k * .55)[::-1]]); c.fill(q, '#8f9ab2'); c.fill(np.vstack([p + N * k * .55, (p + N * k * .02)[::-1]]), '#f6fafe'); c.line(p + N * k * .02, w * .3, '#5c6680', taper=(0, .3)); outline(c, q, w * .6, INK, 3)


def draw_hat(pen, h, s, w, f=0, pos=None, rot=None):
    """small straw hat on a Head frame (head units)"""
    c = pen.c; t = f / 24.; a = np.linspace(0, 2 * PI, 48, endpoint=False); dy = 0. if pos is None else pos
    brim = np.c_[np.sin(a) * (1.36 + .03 * np.sin(3 * a + 1)), .50 + dy + .055 * np.sin(2 * a + .6 + t * 5) + .03 * np.sin(5 * a + t * 8), np.cos(a) * 1.30 - .06]; inn = np.c_[np.sin(a) * .84, np.full(48, .60 + dy), np.cos(a) * .86 - .06]
    B, I = h.P3(brim), h.P3(inn); under = h.R[1, 2] > .06
    pen_shape = lambda q, m, ww: (c.fill(q, m['shade'] if under else m['base']), None)
    c.fill(B, MAT['straw']['shade'] if under else MAT['straw']['base'])
    if not under: c.clip(B); c.fill(hull(I) - pen.L * s * .16, MAT['straw']['shade']); c.unclip()
    for rr in (.95, 1.08, 1.21):
        q = h.P3(np.c_[np.sin(a) * rr, .52 + dy + .05 * np.sin(2 * a + .6 + t * 5), np.cos(a) * rr * .96 - .06])
        for k in range(0, 48, 3):
            if (k // 3 + int(rr * 10)) % 2 == 0: c.line(q[[k, (k + 1) % 48, (k + 2) % 48]], w * .36, MAT['straw']['deep'], taper=(.2, .2), seed=k)
    outline(c, B, w * .9, pen.ink, 31, press=.5)
    dome = [inn]
    for y, r in ((.70, .84), (.86, .80), (1.0, .68), (1.10, .48), (1.15, .2)): dome.append(np.c_[np.sin(a) * r, np.full(48, y + dy), np.cos(a) * r * 1.02 - .06])
    if h.R[1, 1] > -.2 or under is False:
        D = hull(h.P3(np.vstack(dome))); c.fill(D, MAT['straw']['shade']); c.clip(D); c.fill(hull(h.P3(np.vstack(dome[1:]))) * .86 + D.mean(0) * .14 + pen.L * s * .16, MAT['straw']['base'])
        rb = hull(h.P3(np.vstack([dome[0], np.c_[np.sin(a) * .845, np.full(48, .74 + dy), np.cos(a) * .855 - .06]]))); c.fill(rb, MAT['ribbon']['shade']); c.clip(rb); c.fill(rb + pen.L * s * .20, MAT['ribbon']['base']); c.unclip()
        for y, r in ((.88, .79), (1.0, .68), (1.08, .5)):
            q = h.P3(np.c_[np.sin(a) * r, np.full(48, y + dy), np.cos(a) * r - .06]); zz = h.depth(np.c_[np.sin(a) * r, np.full(48, y + dy), np.cos(a) * r - .06])
            for k in range(0, 46, 3):
                if zz[k] > -.1: c.line(q[k:k + 3], w * .34, MAT['straw']['deep'], taper=(.2, .2), seed=k)
        c.unclip(); outline(c, D, w * .9, pen.ink, 33, press=.5)


# ------------------------------------------------------------------ hands / feet at figure size
_HX = [(.33, .0), (.11, .02), (-.11, .0), (-.31, -.05)]
HANDP = dict(fist=([(88, 100, 60)] * 4, (30, 60)), grip=([(70, 80, 50)] * 4, (20, 60)), open=([(8, 10, 6), (4, 8, 5), (6, 10, 6), (10, 14, 8)], (48, 12)), splay=([(-6, 12, 8), (-10, 10, 6), (-8, 12, 8), (-4, 16, 10)], (64, 10)),
             relax=([(18, 28, 14), (26, 32, 16), (32, 36, 18), (36, 40, 20)], (30, 24)), claw=([(-10, 55, 30)] * 4, (50, 30)), flat=([(0, 2, 2)] * 4, (16, 8)), point=([(2, 4, 3), (80, 95, 50), (84, 96, 50), (86, 96, 50)], (30, 60)))
for k_ in ('hold', 'soft', 'two', 'thumb'): HANDP[k_] = HANDP['grip' if k_ == 'hold' else 'relax']


def small_hand(pen, o, Rh, size, side, pose='fist', mat=SKIN, w=None):
    """hand from capsules: Rh columns = thumb side, finger direction, back; size = hand length m"""
    v, c = pen.v, pen.c; w = pen.w * .8 if w is None else w; k = size; th, fd, bk = Rh[:, 0], Rh[:, 1], Rh[:, 2]; F, TH = HANDP.get(pose, HANDP['relax']); px = k * v.s(o)
    pt = lambda x, y, z: o + (th * x + fd * y + bk * z) * k
    cap = lambda a, b, ra, rb: hull(np.vstack([v(a) + np.c_[np.cos(ANG), np.sin(ANG)] * ra * k * v.s(a), v(b) + np.c_[np.cos(ANG), np.sin(ANG)] * rb * k * v.s(b)]))
    parts = [(v.z(pt(0, .25, 0)), 'palm', [hull(np.vstack([v(pt(x, y, z)) for x, y in ((.20, 0), (-.20, 0), (.40, .34), (-.36, .34), (.42, .52), (-.40, .46)) for z in (-.14, .13)])[None][0])])]
    for i, (x, dy) in enumerate(_HX):
        a = pt(x, .52 + dy, .03); ang = 0.; seg = []
        for j, (ln, r) in enumerate(((.27, .115), (.19, .105), (.15, .09))):
            ang += math.radians(F[i][j]); d = fd * math.cos(ang) - bk * math.sin(ang) + th * x * .16 * (1 if pose in ('splay', 'open') else .0); b = a + d * ln * k * (1.05 if i == 1 else .86 if i == 3 else 1); seg.append(cap(a, b, r, r * .93)); a = b
        parts.append((v.z(a) * .5 + v.z(pt(x, .6, 0)) * .5, 'f%d' % i, seg))
    sp, cu = math.radians(TH[0]), math.radians(TH[1]); a = pt(.28, .12, -.04); d1 = fd * math.cos(sp) + th * math.sin(sp) - bk * .25; b = a + unit(d1) * .28 * k; d2 = fd * math.cos(sp * .4) + th * (math.sin(sp) - math.sin(cu)) - bk * math.sin(cu) * .8; e = b + unit(d2) * .24 * k
    parts.append(((v.z(a) + v.z(e)) / 2 - .01 * k, 'th', [cap(a, b, .15, .12), cap(b, e, .12, .095)]))
    parts.sort(key=lambda q: -q[0]); allp = [p for q in parts for p in q[2]]; C = union(allp, .6, smooth=3); det = px > 26
    c.fill(C, mat['shade'], evenodd=True)
    for z, n, pl in parts:
        U = union(pl, .6, smooth=3)
        c.fill(U, mat['shade'], evenodd=True); c.clip(U); c.fill([u + pen.L * px * .07 for u in U], mat['base']); c.unclip()
        if det: pen.contour(U, w * (.62 if n != 'palm' else .5), 3, 0)
    pen.contour(C, w, 5, 0); return C


def small_foot(pen, an, Rf, H, side, kind='sandal', mat=SKIN, w=None):
    v, c = pen.v, pen.c; w = pen.w if w is None else w; x, y, z = Rf[:, 0], Rf[:, 1], Rf[:, 2]; k = H; sg = 1 if side == 'R' else -1
    pt = lambda a, b, d: an + (x * a * sg + y * b + z * d) * k
    top = [(.0, -.024, .012), (.02, -.012, .012), (-.02, -.012, .012), (0, .02, .022)]; sole = [(.020, -.034, -.034), (-.018, -.034, -.034), (.030, .030, -.036), (-.024, .03, -.036), (.036, .080, -.036), (-.026, .088, -.036), (.026, .104, -.034), (-.010, .112, -.034), (.03, .08, -.012), (-.022, .088, -.010), (.014, .106, -.016), (.022, .03, .004), (-.02, .03, .006)]
    body = hull(v(np.array([pt(*q) for q in top + sole])))
    if kind == 'sandal':
        S = hull(v(np.array([pt(a * 1.12, b * 1.06, d) for a, b, _ in sole for d in (-.036, -.050)]))); kit.shape(c, S, MAT['sandal']['base'], w * .8)
        C = pen.patch([body], mat, w, shade=[body - pen.L * k * v.s(an) * .030 - (pen.L * 0)], soft=.8) if False else None
        c.fill(body, mat['shade']); c.clip(body); c.fill(body + pen.L * k * v.s(an) * .016, mat['base']); c.unclip(); outline(c, body, w, pen.ink, 41, press=.45)
        for a, b in (((.026, .045, -.016), (.004, .088, -.012)), ((-.022, .045, -.014), (.004, .088, -.012))): c.line(v(np.array([pt(*a), pt((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, -.002), pt(*b)])), w * 1.3, '#3b2a22', taper=(.05, .05))
        for j in range(3): c.line(v(np.array([pt(.018 - .014 * j, .100, -.020), pt(.019 - .014 * j, .112, -.030)])), w * .45, pen.ink)
    else:
        m = MAT['black' if kind == 'shoe' else 'boot']; S = hull(v(np.array([pt(a * 1.1, b * 1.04, d) for a, b, _ in sole for d in (-.036, -.048)]))); kit.shape(c, S, m['deep'], w * .7); c.fill(body, m['shade']); c.clip(body); c.fill(body + pen.L * k * v.s(an) * .018, m['base']); c.fill(body * .5 + body.mean(0) * .5 + pen.L * k * v.s(an) * .02, m['hi'], .6); c.unclip(); outline(c, body, w, pen.ink, 41, press=.45)


# ------------------------------------------------------------------ the actor
class Actor:
    def __init__(self, who, coat=None, scale=1.0, hat=None, swords=False):
        self.who, self.b, self.scale = who, BODY[who], scale; self.coat = coat or dict(luffy='vest', sanji='suit', zoro='coat', marine='uni')[who]; self.hat = who == 'luffy' if hat is None else hat; self.swords = swords; self._p = P(); self.place((0, 0, 0))

    def pose(self, p): self._p = POSES[p] if isinstance(p, str) else p; return self

    def place(self, pos, heading=0.): self.xf = ('ground', np.asarray(pos, float), Rz(heading)); return self

    def hang(self, pos, R): self.xf = ('pel', np.asarray(pos, float), np.asarray(R, float)); return self

    def build(self):
        S = Skel(self.b, self._p, 0, 0, 0, 0., self.scale); kind, pos, R = self.xf; piv = S.pel.copy() if kind == 'pel' else np.zeros(3); mv = lambda q: pos + R @ (q - piv)
        for k in ('pel', 'waist', 'chest', 'neck', 'headb', 'head'): setattr(S, k, mv(getattr(S, k)))
        for k in ('Rp', 'Rw', 'Rc', 'Rn', 'Rh', 'Rr'): setattr(S, k, R @ getattr(S, k))
        for d in (S.sh, S.el, S.wr, S.hip, S.kn, S.an):
            for s in d: d[s] = mv(d[s])
        for d in (S.e1a, S.e1l):
            for s in d: d[s] = R @ d[s]
        for d in (S.Rhand, S.Rfoot):
            for s in d: d[s] = R @ d[s]
        self.S = S; return S

    def torso(self, v, S, k=1., kl=1.):
        H = S.H; x = lambda R: R[:, 0]; y = lambda R: R[:, 1]; z = lambda R: R[:, 2]; mz = dict(luffy=.95, sanji=.98, zoro=1.16, marine=1.0)[self.who] * k
        st = [(S.pel - z(S.Rp) * .050 * H, S.Rp, .076, .058, .066), (S.pel, S.Rp, .086, .064, .072), (S.pel * .45 + S.waist * .55, S.Rp, .080, .062, .066), (S.waist, S.Rw, .071, .058, .056), (S.waist * .5 + S.chest * .5, S.Rw, .083, .068, .064),
              (S.chest, S.Rc, .101, .076, .072), (S.chest + z(S.Rc) * .055 * H, S.Rc, .108, .073, .074), (S.chest + z(S.Rc) * .095 * H, S.Rc, .092, .054, .060), (S.neck, S.Rc, .046, .036, .040)]
        return Tube(v, [q[0] for q in st], [y(q[1]) for q in st], [x(q[1]) for q in st], np.array([[q[3], q[4], q[2], q[2]] for q in st]) * H * mz)

    def draw(self, cam, f=0, light=(-.5, -.6, .6), expr='calm', rim=None, wind=(0, 0, 0), look=(0, 0), w=None, hide=(), arm_fn=None, hair=.6, puff=0., hatpos=None, sword_mouth=False, flashblade=0., pad=90, tone=None, after=None, hatch=True, sm=MAT, full=False, nohead=False):
        v = View(cam); S = self.build(); H = S.H; who = self.who; pal = SKINOF[who]; kl = H * self.b['limb']
        J = np.array([S.pel, S.head, S.chest] + [d[s] for d in (S.sh, S.el, S.wr, S.kn, S.an) for s in 'LR']); px = v(J); self.px = {n: p for n, p in zip(['pel', 'head', 'chest'] + [a + s for a in ('sh', 'el', 'wr', 'kn', 'an') for s in 'LR'], px)}
        sc = v.s(S.chest); self.sc = sc; m = H * sc * .40 + pad; m = min(m, 700)
        lo = np.maximum(np.floor(px.min(0) - m), (0, 0)).astype(int); hi = np.minimum(np.ceil(px.max(0) + m), (W, FH)).astype(int)
        if full: lo, hi = np.array([0, 0]), np.array([W, FH])
        if (hi - lo).min() < 4: return np.zeros((4, 4, 4), f32), 0, 0
        cel = Cel(hi[0] - lo[0], hi[1] - lo[1], lo[0], lo[1], ss=2); L3 = np.asarray(light, float); L2 = v.l2(L3); hp = H * sc
        w = float(np.clip(hp * .0062, .9, 5.6)) if w is None else w; pen = Pen(cel, v, L2, w, 3); hn = int(hatch and hp > 330)
        tor = self.torso(v, S); G = []; zc = v.z(S.chest); wind = np.asarray(wind, float); out = {s: S.Rp[:, 0] * (1 if s == 'R' else -1) for s in 'LR'}; outc = {s: S.Rc[:, 0] * (1 if s == 'R' else -1) for s in 'LR'}
        big = hp > 240
        # ---- legs
        for s in 'LR':
            th = joint_tube(v, S.hip[s], S.kn[s], S.e1l[s], out[s], T_THIGH, kl); cf = joint_tube(v, S.kn[s], S.an[s], S.e1l[s], out[s], T_CALF, kl)
            def leg(s=s, th=th, cf=cf):
                kind = dict(luffy='sandal', sanji='shoe', zoro='boot', marine='shoe')[who]; foot = lambda: small_foot(pen, S.an[s], S.Rfoot[s], H * self.b['foot'], s, kind, pal)
                if who == 'luffy':
                    foot(); pen.part([th, cf], pal, hatch=hn, seed=2, mods=[None, np.array([1, 1.3, 1.5, 1, .8, .8, 1])])
                    if big: pen.surf(cf, [(1.2, 150), (2.2, 165), (3.2, 170)], seed=4); pen.surf(cf, [(.2, -40), (.6, 0), (.2, 40)], seed=5); pen.surf(cf, [(6.2 - .3, 100), (5.4, 80)], seed=6)
                    sh = joint_tube(v, S.hip[s], S.kn[s], S.e1l[s], out[s], [(-.1, .062, .066, .058, .056), (.3, .070, .066, .064, .058), (.65, .070, .066, .066, .060), (.90, .072, .070, .068, .062)], kl * 1.06)
                    cu = joint_tube(v, S.hip[s], S.kn[s], S.e1l[s], out[s], [(.84, .078, .075, .074, .068), (.93, .084, .080, .080, .073), (1.02, .078, .075, .074, .068)], kl * 1.06)
                    pen.part([cu], sm['fur'], seed=8, soft=1.4); pen.part([sh], sm['blue'], hatch=hn, seed=9, mods=[np.array([1, .7, 1.3, 1])])
                    if big:
                        for q in ([(0.3, 20), (1.2, 55), (2.2, 80)], [(.4, -10), (1.0, -40), (2.6, -55)], [(2.6, 10), (2.0, 25)]): pen.surf(sh, q, seed=11)
                else:
                    tm = sm[dict(sanji='suit', zoro='black', marine='navy')[who]]; k = 1.14 if who != 'zoro' else 1.28
                    a = joint_tube(v, S.hip[s], S.kn[s], S.e1l[s], out[s], T_THIGH, kl * k); tb = [(0, .040, .038, .038, .038), (.3, .036, .046, .042, .042), (.7, .034, .038, .038, .036), (.98, .036, .040, .038, .036)] if who != 'zoro' else [(0, .042, .040, .040, .040), (.3, .042, .050, .046, .046), (.56, .046, .050, .050, .048)]
                    b = joint_tube(v, S.kn[s], S.an[s], S.e1l[s], out[s], tb, kl * k); foot()
                    if who == 'zoro': pen.part([joint_tube(v, S.kn[s], S.an[s], S.e1l[s], out[s], [(.50, .046, .050, .046, .046), (.56, .050, .054, .05, .05), (.7, .040, .046, .042, .042), (1., .034, .036, .034, .034)], kl * 1.1)], sm['boot'], hi=.5, seed=12)
                    pen.part([a, b], tm, hi=.55 if who != 'marine' else 0, hatch=0, seed=13)
                    if big:
                        for q in ([(.2, 30), (1.6, 15), (4, 10)], [(4.6, 150), (4.2, 60)], [(.5, 70), (1.5, 100)]): pen.surf(a, q, seed=14, col=tm['hi'] if who in ('sanji', 'zoro') else None)
                        pen.surf(b, [(.1, 20), (1, 8), (2.4, 0)], seed=15, col=tm['hi']); pen.surf(b, [(.2, 160), (.5, 90)], seed=16, col=tm['hi'])
            G.append((float((th.z.mean() + cf.z.mean()) / 2) + .02, leg))
        # ---- arms
        for s in 'LR':
            if 'arm' + s in hide: continue
            ua = joint_tube(v, S.sh[s], S.el[s], -S.e1a[s], outc[s], T_UARM, kl * (1.12 if who == 'zoro' else 1.)); fa = joint_tube(v, S.el[s], S.wr[s], -S.e1a[s], outc[s], T_FARM, kl * (1.1 if who == 'zoro' else 1.))
            def arm(s=s, ua=ua, fa=fa):
                hsz = H * .108; gp = S.grip[s]
                sleeve = who in ('sanji', 'zoro', 'marine') or self.coat == 'grown'
                far = (v.z(S.wr[s]) > v.z(S.el[s]))
                hand = lambda: self._hand(pen, S, s, hsz, gp, pal, flashblade)
                if arm_fn is not None and arm_fn(pen, s, ua, fa, S): return
                if far and not sleeve: hand()
                if not sleeve:
                    pen.part([ua, fa], pal, hatch=hn, seed=20, mods=[np.array([1, 1.3, .8, 1.2, .9, 1]), np.array([1, 1.4, 1.1, .9, .9, 1])])
                    if big: pen.surf(ua, [(1.6, 20), (2.1, 55), (1.4, 100)], seed=21); pen.surf(ua, [(2.2, -20), (3.4, -12)], seed=22); pen.surf(fa, [(.4, 40), (1.6, 60), (2.4, 60)], seed=23)
                    if not far: hand()
                else:
                    m = sm[dict(sanji='suit', zoro='green', marine='white', luffy='red')[who]]; k = 1.2 if who != 'zoro' else 1.34
                    if who != 'sanji': hand()
                    else: hand()
                    if who == 'marine': pen.part([fa], pal, seed=24)
                    T2 = [(a, *(q * k for q in r)) for a, *r in T_UARM]; tf = [(0, .034, .034, .034, .034), (.3, .040, .040, .038, .037), (.62, .038, .040, .038, .036), (.96 if who == 'sanji' else .84, .038, .044, .040, .038)]
                    a = joint_tube(v, S.sh[s], S.el[s], -S.e1a[s], outc[s], T2, kl); b = joint_tube(v, S.el[s], S.wr[s], -S.e1a[s], outc[s], tf, kl * (1.0 if who != 'zoro' else 1.25)); tub = [a] if who == 'marine' else [a, b]
                    if who == 'sanji': pen.part([joint_tube(v, S.el[s], S.wr[s], -S.e1a[s], outc[s], [(.9, .030, .03, .03, .03), (1.04, .030, .03, .03, .03)], kl)], sm['shirt'], seed=26)
                    pen.part(tub, m, hi=.5 if who == 'sanji' else 0, hatch=hn if who != 'sanji' else 0, seed=25)
                    if big:
                        cl = m['hi'] if who in ('sanji',) else None
                        for q in ([(.8, 60), (1.8, 20), (2.6, 0)], [(4.4, 170), (4.0, 60)], [(4.7, 120), (4.3, 20)]): pen.surf(a, q, seed=27, col=cl)
                        if who != 'marine': pen.surf(b, [(.2, 150), (.6, 60)], seed=28, col=cl); pen.surf(b, [(.6, 20), (1.6, 60), (2.6, 70)], seed=29, col=cl)
            G.append((float((ua.z.mean() + fa.z.mean()) / 2 - .015), arm))
        # ---- torso
        def torso():
            nk = joint_tube(v, S.neck - S.Rc[:, 2] * .02 * H, S.headb + S.Rh[:, 2] * .02 * H, S.Rn[:, 1], S.Rn[:, 0], [(0, .036, .040, .040, .040), (.5, .031, .034, .033, .033), (1, .031, .034, .032, .032)], H * (1.15 if who == 'zoro' else .95))
            if who in ('luffy', 'zoro'):
                pen.part([tor, nk], pal, hatch=hn, seed=30, mods=[np.array([1, 1, .8, .8, 1.1, 1.3, 1.2, 1, 1]), None])
                if big:
                    for q in ([(5.2, 14), (4.95, 30), (5.2, 52), (5.8, 62)], [(5.2, -14), (4.95, -30), (5.2, -52), (5.8, -62)], [(5.2, 0), (4.6, 0), (3.3, 0), (2.5, 0)], [(4.6, 8), (4.5, 26)], [(4.6, -8), (4.5, -26)], [(3.9, 8), (3.8, 24)], [(3.9, -8), (3.8, -24)], [(3.2, 8), (3.15, 22)], [(3.2, -8), (3.15, -22)],
                              [(7.5, 12), (7.1, 40), (7.15, 64)], [(7.5, -12), (7.1, -40), (7.15, -64)], [(4.5, 34), (3.4, 38), (2.4, 24)], [(4.5, -34), (3.4, -38), (2.4, -24)]): pen.surf(tor, q, w * .5, seed=31)
                if who == 'luffy' and self.coat == 'grown':                    # the X
                    for q in ([(6.5, -40), (5.4, -10), (4.0, 30)], [(6.5, 40), (5.4, 10), (4.0, -30)]):
                        a = [np.array(p) + [.0, 5] for p in q]; b = [np.array(p) - [.0, 5] for p in q]; qa, ka = tor.surf(*np.array(spl(a, n=6)).T); qb, kb = tor.surf(*np.array(spl(b, n=6)).T); ok = (ka > .05) & (kb > .05)
                        if ok.sum() > 4: pl = np.vstack([v(qa[ok]), v(qb[ok])[::-1]]); cel.fill(pl, '#b5604c'); outline(cel, pl, w * .4, '#7a3430', 33, tremor=.8)
                if who == 'zoro': pen.surf(tor, [(6.9, -48), (5.4, -4), (3.2, 42)], w * .8, seed=34, col='#8a4534'); [pen.surf(tor, [(6.9 - q, -48 + q * 24 - 5), (6.75 - q, -48 + q * 24 + 7)], w * .4, seed=35, col='#8a4534') for q in (.6, 1.3, 2.0, 2.7)]
            else:
                pen.part([nk], pal, seed=36)
            # pelvis cloth
            pm = sm[dict(luffy='blue', sanji='suit', zoro='black', marine='navy')[who]]; pel = Tube(v, tor.C[:4], tor.F[:4], tor.O[:4], tor.rad[:4] * np.array([1.13, 1.1, 1.05, 1.03])[:, None])
            pen.part([Tube(v, pel.C[:3], pel.F[:3], pel.O[:3], pel.rad[:3])], pm, hi=.4 if who in ('sanji', 'zoro') else 0, seed=37)
            if who == 'luffy':
                m = sm['red']; a0 = 24 if self.coat == 'vest' else 44; ph = lambda t: a0 + (10 if t > 6.5 else 0) + 14 * max(0, t - 6) + (8 * (4 - t) if t < 4 else 0); vt = Tube(v, tor.C[1:], tor.F[1:], tor.O[1:], tor.rad[1:] * 1.07); vt2 = lambda q0, q1: vt.band(1.0, 6.15 if self.coat == 'vest' else 6.5, q0, q1, nph=22)
                q = vt2(lambda t: ph(t + 1), lambda t: 360 - ph(t + 1)); d = vt2(lambda t: 105 + 8 * math.sin(t * 3) if L2[0] < 0 else 190, lambda t: 170 + 10 * math.sin(t * 2.4) if L2[0] < 0 else 255 + 8 * math.sin(t * 3)); C = pen.patch(q, m, shade=d, seed=38)
                if big and C:
                    for sg in (1, -1):
                        for pq in ([(5.4, sg * (a0 + 34)), (4.2, sg * (a0 + 14)), (2.6, sg * (a0 + 18))], [(3.4, sg * 120), (2.2, sg * 95), (1.3, sg * 80)], [(5.6, sg * 80), (4.6, sg * 70)]): pen.surf(vt, pq, seed=39)
                        for j in range(3):
                            Q, k = vt.surf([4.6 - 1.1 * j], [sg * (ph(5.6 - 1.1 * j) + 7)])
                            if k[0] > .1 and sg > 0: b = v(Q)[0]; cel.fill(b + np.c_[np.cos(ANG), np.sin(ANG)] * w * 1.25, pen.ink); cel.fill(b + np.c_[np.cos(ANG), np.sin(ANG)] * w * .85, '#f0c648')
                if self.coat == 'grown': self._sash(pen, S, tor, v, f, wind, sm['yellow'], w)
            elif who == 'sanji':
                jt = Tube(v, np.vstack([[S.pel - S.Rp[:, 2] * .10 * H], tor.C]), np.vstack([tor.F[:1], tor.F]), np.vstack([tor.O[:1], tor.O]), np.vstack([tor.rad[:1] * 1.34, tor.rad * np.array([1.26, 1.18, 1.12, 1.10, 1.10, 1.10, 1.10, 1.08, 1.0])[:, None]])); m = sm['suit']
                pen.part([jt], m, hi=.5, seed=40); pen.patch(jt.band(5.6, 8.6, lambda t: -3 - 9 * (t - 5.6), lambda t: 3 + 9 * (t - 5.6)), sm['shirt'], w * .7, seed=41); pen.patch(jt.band(5.8, 8.2, -3.5, 3.5), sm['black'], w * .4, seed=42)
                for sg in (1, -1): pen.surf(jt, [(8.6, sg * 28), (7.0, sg * 34), (6.4, sg * 26), (5.6, sg * 4)], w * .8, seed=43, col=m['hi']); [pen.surf(jt, [(y_, sg * 17), (y_ + .12, sg * 20)], w * 1.6, seed=44, col='#e6b84a', thr=.0) for y_ in (4.9, 4.0, 3.1)]
                pen.surf(jt, [(5.6, -4), (4.2, -8), (2.4, -6), (.2, -14)], w * .8, seed=45, col=m['hi'])
            elif who == 'zoro':
                ct = Tube(v, tor.C[1:], tor.F[1:], tor.O[1:], tor.rad[1:] * 1.12); ph = lambda t: 42 + 12 * max(0, t - 5.2); q = ct.band(1.4, 6.6, ph, lambda t: 360 - ph(t), nph=22); pen.patch(q, sm['green'], shade=ct.band(1.4, 6.6, lambda t: 120, lambda t: 240, nph=14), seed=46)
                for sg in (1, -1): pen.surf(ct, [(6.6, sg * 62), (4.5, sg * 50), (1.5, sg * 46)], w * .9, seed=47, col='#8fb04a')
                self._sash(pen, S, tor, v, f, wind, sm['sash'], w);
                for k in range(3):                                           # scabbards at the right hip
                    o = S.pel + S.Rp @ (np.array([.098, .02 - .03 * k, .02 + .018 * k]) * H); d = S.Rp @ unit(np.array([.35, -1, -.35 - .12 * k])); p = v(np.array([o + d * -.10 * H, o + d * .42 * H]))
                    if not np.isfinite(p).all(): continue
                    col = ('#ece7dd', '#7a2f2a', '#33243f')[k]; cel.line(np.linspace(p[0], p[1], 8), w * 3.4, pen.ink, taper=(0, .02), press=0); cel.line(np.linspace(p[0], p[1], 8), w * 2.3, col, taper=(0, .02), press=0)
            else:
                ut = Tube(v, tor.C, tor.F, tor.O, tor.rad * 1.10); pen.part([ut], sm['white'], seed=48); pen.patch(ut.band(6.9, 8.4, -70, 70), sm['navy'], w * .7, seed=49)
        # coat tails hang from the waist: depth by which side of the body they are
        if who == 'zoro':
            for lo_, hi_, sd in ((52, 130, 1), (130, 230, 2), (230, 308, 3)):
                Q, _ = Tube(v, tor.C[1:], tor.F[1:], tor.O[1:], tor.rad[1:] * 1.14).surf(np.full(6, 1.5), np.linspace(lo_, hi_, 6)); g = strip_grid(v, Q, .44 * H, wind, f, sd, n=7, wave=.4); G.append((float(v.z(g).mean()) + (.05 if sd == 2 else 0), lambda g=g, sd=sd: draw_strip(pen, g, sm['green'], seed=sd)))
        G.append((float(zc), torso))
        if who in ('sanji', 'luffy') and self.coat in ('suit', 'grown'):          # open hems that the wind takes
            m_ = sm['suit' if who == 'sanji' else 'red']; sp_ = ((20, 62, 1), (62, 118, 2), (118, 242, 3), (242, 298, 4), (298, 340, 5)) if who == 'sanji' else ((52, 120, 1), (120, 240, 3), (240, 308, 5))
            for lo_, hi_, sd in sp_:
                Q, _ = Tube(v, tor.C, tor.F, tor.O, tor.rad * (1.30 if who == 'sanji' else 1.16)).surf(np.full(5, .6 if who == 'sanji' else 2.3), np.linspace(lo_, hi_, 5)); g = strip_grid(v, Q, (.10 if who == 'sanji' else .15) * H, wind * .8, f, sd, n=5, wave=.4, hz=3.0); G.append((float(v.z(g).mean()) + (.06 if sd == 3 else -.01), lambda g=g, sd=sd: draw_strip(pen, g, m_, seed=sd)))
        if who == 'sanji' and np.linalg.norm(wind) > 1:
            q0 = S.chest + S.Rc @ np.array([0, .085, .06]) * H; g = strip_grid(v, np.stack([q0 - S.Rc[:, 0] * .012 * H, q0, q0 + S.Rc[:, 0] * .012 * H]), .20 * H, wind * 1.2, f, 9, n=7, wave=.6, hz=3.4); G.append((float(v.z(g).mean()) - .08, lambda g=g: draw_strip(pen, g, sm['black'], w * .7, seed=3)))
        hd_z = float(v.z(S.head)); up_ = float((v.cam.eye - S.head) @ S.Rh[:, 2])
        if not nohead: G.append((hd_z - .06, lambda: small_head(pen, S, who, expr, self.hat, look, f, wind, puff, sword_mouth, hair, hatpos)))
        for _, fn in sorted(G, key=lambda q: -q[0]): fn()
        if after: after(pen, S, v)
        L = cel.rgba()
        if tone is not None: L = L.copy(); L[..., :3] *= rgb(tone)
        if rim is not None:
            rw = rim.get('w', max(1.6, hp * .012)); d = v.l2(np.asarray(rim.get('dir', light), float)); L = kit.rim(L, -d[0] * rw, -d[1] * rw, rim.get('col', '#fff2cf'), rim.get('amt', .9), soft=.6)
            if rim.get('glow', 0): g = cv2.GaussianBlur(L[..., 3] - kit.shifted(L[..., 3], -d[0] * rw * 2.2, -d[1] * rw * 2.2), (0, 0), rw * 1.1); L[..., :3] = 1 - (1 - L[..., :3]) * (1 - (np.clip(g, 0, 1) * L[..., 3] * rim['glow'])[..., None] * rgb(rim.get('col', '#fff2cf')))
        return L, int(lo[0]), int(lo[1])

    def _sash(self, pen, S, tor, v, f, wind, m, w):
        t2 = Tube(v, tor.C[:4], tor.F[:4], tor.O[:4], tor.rad[:4] * 1.17); pen.part([Tube(v, t2.C[1:4], t2.F[1:4], t2.O[1:4], t2.rad[1:4])], m, seed=50)
        for q in ([(1.3, -70), (1.9, 0), (1.5, 70)], [(2.5, -80), (2.2, 0), (2.6, 80)]): pen.surf(t2, q, w * .5, seed=51)
        Q, k = t2.surf(np.full(3, 1.7), np.array([60, 72, 84]))
        g = strip_grid(v, Q, .26 * S.H, wind, f, 5, n=6, wave=.5); draw_strip(pen, g, m, w * .8, seed=4)

    def _hand(self, pen, S, s, hsz, gp, pal, flash):
        v = pen.v
        if self.swords and gp in ('grip', 'hold'):
            th, fd, bk = S.Rhand[s].T; o = S.wr[s] + fd * hsz * .42 - bk * hsz * .22; L_ = .46 * S.H; hilt = v(np.array([o - th * hsz * .9, o + th * hsz * .7])); bl = v(np.array([o + th * hsz * .75, o + th * (hsz * .75 + L_ * .5) + fd * L_ * .03, o + th * (hsz * .75 + L_) + fd * L_ * .10])); w = pen.w
            blade(pen.c, bl, max(3., hsz * v.s(o) * .22), w); pen.c.line(np.linspace(hilt[0], hilt[1], 6), hsz * v.s(o) * .26 + w, INK, taper=(0, 0), press=0); pen.c.line(np.linspace(hilt[0], hilt[1], 6), hsz * v.s(o) * .20, '#5c2430' if s == 'R' else '#2a2236', taper=(0, 0), press=0)
            g = hilt[1]; d = unit(hilt[1] - hilt[0]); n = np.array([-d[1], d[0]]) * hsz * v.s(o) * .24; kit.shape(pen.c, np.array([g + n, g + n * .8 + d * w * 1.5, g - n * .8 + d * w * 1.5, g - n]), '#e6c04a', w * .5)
        small_hand(pen, S.wr[s], S.Rhand[s], hsz, s, gp if gp in HANDP else 'relax', pal)


# ------------------------------------------------------------------ rubber
def rubber(pen, P3, rad, mat=SKIN, ripple=0., k=9., phase=0., rings=14, seed=0, w=None, zmin=.16, veins=True, lk=.80, lo=.34, hi=.0):
    """stretched limb along a 3-D polyline: rad (m) per point; ripple = amplitude as a share of the radius travelling along it"""
    v = pen.v; P3 = np.asarray(P3, float); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P3, axis=0), axis=1))]; n = max(40, int(len(P3) * 2)); s = np.linspace(0, d[-1], n); C = np.stack([np.interp(s, d, P3[:, i]) for i in range(3)], -1); r = np.interp(s, d, np.broadcast_to(rad, len(P3)))
    u = s / d[-1]; r = r * (1 + ripple * np.sin(2 * PI * (k * u - phase)) * np.sin(PI * u) ** .6); ok = v.z(C) > zmin; C, r, u = C[ok], r[ok], u[ok]
    if len(C) < 3: return []
    c2 = v(C); zz = v.z(C); sc = v.cam.f / zz; R2 = r * sc; circ = np.c_[np.cos(ANG), np.sin(ANG)]
    H_ = lambda kk, off: [hull(np.vstack([c2[i] + circ * R2[i] * kk + pen.L * off * R2[i], c2[i + 1] + circ * R2[i + 1] * kk + pen.L * off * R2[i + 1]])) for i in range(len(c2) - 1)]
    C_ = union(H_(1., 0), .8); cel = pen.c
    if not C_: return []
    cel.fill(C_, mat['shade'], evenodd=True); lit = union(H_(lk, lo), .7); cel.clip(C_); cel.fill(lit, mat['base'], evenodd=True)
    if hi: cel.fill(union(H_(.22, .52), .7), mat['hi'], hi, evenodd=True)
    T = np.gradient(c2, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; rr = np.random.default_rng(seed + 5)
    for j in range(rings):                                             # rings bow with the roundness; they bunch where the ripple bunches
        i = int((j + .5 + .3 * rr.uniform(-1, 1)) / rings * (len(c2) - 1)); a = np.linspace(-.95, .15 + .5 * rr.random(), 12); sg = -1 if N[i] @ pen.L > 0 else 1; q = c2[i] + N[i] * sg * R2[i] * np.sin(a[:, None] * 1.5) * -1 + T[i] * R2[i] * .28 * np.cos(a[:, None] * 1.5)
        cel.line(q, max(.8, min(pen.w * .6, R2[i] * .05)), pen.ink, taper=(.05, .8), seed=j)
    if veins:
        for j in range(3):
            a0 = rr.uniform(.05, .6); ii = np.arange(int(a0 * len(c2)), int(min(1, a0 + .3) * len(c2) - 1));
            if len(ii) > 5: q = c2[ii] + N[ii] * (R2[ii] * (.25 * (j - 1) + .12 * np.sin(u[ii] * 30 + j)))[:, None]; cel.line(q, max(.8, pen.w * .42), mat['deep'], taper=(.3, .5), seed=j)
    cel.unclip(); pen.contour(C_, w, seed, 2); return C_


def bigfist(cam2, R, side='R', res=560, pal=None, pose='fist', light=(-.5, .75, .6), w=5, box=None, seed=3, pos=(0, 0, 0)):
    """hero fist (op.hands) -> (rgba, ox, oy).  cam2 = hands.Cam"""
    from op import hands as HD
    x0, y0, x1, y1 = box or (0, 0, W, FH); cel = Cel(x1 - x0, y1 - y0, x0, y0, ss=2); g = HD.render([HD.Hand(pose, side, R, pos, HD.HEAVY)], cam2, light, res, clip=(x0, y0, x1, y1)); HD.ink(cel, g, pal or 'day', w, seed); HD.hatch(cel, g, None, w=max(1.2, w * .35), seed=seed) if False else None
    return cel.rgba(), x0, y0, g
