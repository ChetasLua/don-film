"""SHIPS: 3-D hulls, masts, sails, rigging drawn as inked cels at any angle (z-buffered flats, hard shadow, cast shadows, pen contours + feature lines).
Local axes: +X bow, +Y port, +Z up, z=0 waterline, metres.

    from op import ships
    m = ships.sunny(t, wind=1.)          # also merry  red_force  marine  dinghy  barrel  generic(seed)   (rebuild per frame: sails/flags move with t seconds)
    d = ships.draw([(m, ships.pose(x,y,z,heading,pitch,roll))], cam, light=(-.5,.6,.6))     # -> d.layer (premult RGBA)  d.z (camera depth, 1e9 empty)
    img = sea.under(img, d.layer, d.z, water)         # water hides the lower hull
  draw(..., ss=2, ink='pen'|'edge'|None, lw=1., shadow=True, rim=(col,px,amt), detail=1., dark=0..1 (toward silhouette colour `sil`), bounce='#hex')
  mirror of a ship for water: ships.draw([(m, ships.mirror(pose))], cam, light, ink=None, shadow=False, ss=1)
  fleets: ships.draw(list of 100+ (generic(...), pose), cam, ink='edge', shadow=False)
  Model API: grid/lathe/ball/box/tube/cyl/sail/flag/rope/line/decal. m.anchor['mast'|'prow'|'deck'|'nest'|...] -> local points; pose @ point via ships.at(pose,p)
  emblem(), gull(), scar_emblem() -> [(poly unit coords y-down, colour, ink w)]
"""
import numpy as np, cv2, math
from . import W, H
from .paint import f32, rgb, smooth, fbm
from .ink import Cel, spl, resample
from .kit import outline, rim as kit_rim
from .sea import Raster, grid_faces

A_ = np.asarray
SH = (.60, .55, .70)


def pose(x=0., y=0., z=0., head=0., pitch=0., roll=0., s=1.):
    """(R, t, s): heading about Z (0 = +X), pitch bow-up, roll to starboard-down"""
    c, n = math.cos(head), math.sin(head); Rz = A_([[c, -n, 0], [n, c, 0], [0, 0, 1.]])
    c, n = math.cos(-pitch), math.sin(-pitch); Ry = A_([[c, 0, n], [0, 1, 0], [-n, 0, c]])
    c, n = math.cos(roll), math.sin(roll); Rx = A_([[1, 0, 0], [0, c, -n], [0, n, c]])
    return Rz @ Ry @ Rx * s, A_([x, y, z], float)


def mirror(ps): return np.diag([1., 1, -1]) @ ps[0], ps[1] * [1, 1, -1]
def at(ps, p): return A_(p, float) @ ps[0].T + ps[1]


class Part:
    __slots__ = ('V', 'F', 'N', 'col', 'sh', 'sheet', 'grp', 'lw', 'name', 'thr', 'em')


class Model:
    def __init__(self): self.parts = []; self.lines = []; self.decals = []; self.anchor = {}; self.ng = 0; self._g = {}

    def grp(self, name):
        if name not in self._g: self._g[name] = len(self._g)
        return self._g[name]

    def add(self, V, F, N, col, name='', grp=None, sh=SH, sheet=False, lw=2.2, thr=.12, em=0.):
        p = Part(); p.V = A_(V, float); p.F = A_(F); p.N = A_(N, float); p.col = rgb(col) if isinstance(col, str) else A_(col, f32); p.sh = A_(sh, f32); p.sheet = sheet; p.name = name
        p.grp = self.grp(grp or name or f'_{len(self.parts)}'); p.lw = lw; p.thr = thr; p.em = em; self.parts.append(p); return len(self.parts) - 1

    def grid(self, P, col, name='', out=None, wrap=False, **k):
        """P (n,m,3) grid; out = point/None the normals point away from"""
        P = A_(P, float); n, m = P.shape[:2]
        da = np.gradient(P, axis=0); db = np.gradient(P, axis=1)
        if wrap: db = (np.roll(P, -1, 1) - np.roll(P, 1, 1)) / 2
        N = np.cross(da, db); ln = np.linalg.norm(N, axis=-1, keepdims=True); bad = ln[..., 0] < 1e-9
        N = N / (ln + 1e-12)
        if out is not None:
            s = (N * (P - A_(out, float))).sum(-1); N = np.where((np.median(s[~bad]) < 0), -N, N)
            if bad.any(): q = P - A_(out, float); N[bad] = (q / (np.linalg.norm(q, axis=-1, keepdims=True) + 1e-9))[bad]
        return self.add(P.reshape(-1, 3), grid_faces(n, m, 0, wrap), N.reshape(-1, 3), col, name, **k)

    def ball(self, c, r, col, name='', R=None, n=(14, 22), **k):
        r = np.broadcast_to(A_(r, float), (3,)); th = np.linspace(1e-3, math.pi - 1e-3, n[0]); ph = np.linspace(0, 2 * math.pi, n[1])[:-1]
        U = np.stack([np.sin(th)[:, None] * np.cos(ph), np.sin(th)[:, None] * np.sin(ph), np.cos(th)[:, None] + 0 * ph], -1)
        P = U * r; N = U / r; N /= np.linalg.norm(N, axis=-1, keepdims=True)
        if R is not None: P = P @ A_(R).T; N = N @ A_(R).T
        P = P + A_(c, float); return self.add(P.reshape(-1, 3), grid_faces(n[0], n[1] - 1, 0, True), N.reshape(-1, 3), col, name, **k)

    def tube(self, path, r, col, name='', n=9, cap=True, **k):
        P = A_(path, float); r = np.broadcast_to(A_(r, float), (len(P),)); T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=-1, keepdims=True) + 1e-9
        up = A_([0, 0, 1.]) if abs(T[0, 2]) < .85 else A_([1., 0, 0]); rings, Ns = [], []
        a = np.linspace(0, 2 * math.pi, n + 1)[:-1]
        for p, t, q in zip(P, T, r):
            u = np.cross(t, up); u /= np.linalg.norm(u) + 1e-9; v = np.cross(u, t); up = v
            d = np.cos(a)[:, None] * u + np.sin(a)[:, None] * v; rings.append(p + d * q); Ns.append(d)
        if cap: rings = [np.repeat(P[:1], n, 0)] + rings + [np.repeat(P[-1:], n, 0)]; Ns = [np.repeat(-T[:1], n, 0)] + Ns + [np.repeat(T[-1:], n, 0)]
        G = A_(rings); return self.add(G.reshape(-1, 3), grid_faces(len(G), n, 0, True), A_(Ns).reshape(-1, 3), col, name, **k)

    def cyl(self, a, b, r0, r1, col, name='', n=14, **k):
        a, b = A_(a, float), A_(b, float); u = np.linspace(0, 1, 4)[:, None]; return self.tube(a + (b - a) * u, r0 + (r1 - r0) * u[:, 0], col, name, n, **k)

    def lathe(self, prof, c, col, name='', n=22, axis='z', **k):
        """prof [(r, h)...] spun round a vertical axis through c"""
        prof = A_(prof, float); a = np.linspace(0, 2 * math.pi, n + 1)[:-1]
        P = np.stack([prof[:, :1] * np.cos(a), prof[:, :1] * np.sin(a), np.repeat(prof[:, 1:], n, 1)], -1) if axis == 'z' else np.stack([np.repeat(prof[:, 1:], n, 1), prof[:, :1] * np.cos(a), prof[:, :1] * np.sin(a)], -1)
        P = P + A_(c, float); cc = P.reshape(-1, 3).mean(0); return self.grid(P, col, name, out=cc, wrap=True, **k)

    def box(self, lo, hi, col, name='', taper=1., R=None, o=(0, 0, 0), **k):
        lo, hi = A_(lo, float), A_(hi, float); m = (lo + hi) / 2; V, F, N = [], [], []
        c = A_([[x, y, z] for z in (0, 1) for y in (0, 1) for x in (0, 1)]) * (hi - lo) + lo
        c[4:, :2] = m[:2] + (c[4:, :2] - m[:2]) * taper
        for q in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
            p = c[list(q)]; nn = np.cross(p[1] - p[0], p[2] - p[0]); nn /= np.linalg.norm(nn) + 1e-9
            if nn @ (p.mean(0) - m) < 0: nn = -nn
            b = len(V) * 4; V.append(p); N.append(np.repeat(nn[None], 4, 0)); F += [[b, b + 1, b + 2], [b, b + 2, b + 3]]
            self.lines.append((np.vstack([p, p[:1]]), 1.0, None, None)) if k.get('lw', 2) > 0 else None
        V = np.vstack(V); N = np.vstack(N)
        return self.add(V, F, N, col, name, **k)

    def line(self, P, w=1.2, col=None, al=1.): self.lines.append((A_(P, float), w, col, al))
    def decal(self, part, polys): self.decals.append((part, polys))

    def sail(self, tl, tr, bl, br, belly, t, col='#f3ead6', name='sail', seed=0, n=(20, 18), ripple=.05, roach=.10, lw=2.4, wind=1., bdir=None, furl=1., **k):
        """cloth between a head (tl-tr) and foot (bl-br); belly metres along bdir (default +X).  returns surf(u,v) -> 3-D"""
        tl, tr, bl, br = [A_(q, float) for q in (tl, tr, bl, br)]; u, v = np.meshgrid(np.linspace(0, 1, n[0]), np.linspace(0, 1, n[1]), indexing='ij'); U, Vv = u[..., None], v[..., None]
        P = (tl * (1 - U) + tr * U) * (1 - Vv) + (bl * (1 - U) + br * U) * Vv; hgt = np.linalg.norm(bl - tl); bd = A_([1., 0, 0]) if bdir is None else A_(bdir, float)
        sh = np.sin(math.pi * u) ** .62 * (np.sin(math.pi * np.clip(v * .86 + .05, 0, 1)) ** .7)
        gust = 1 + .09 * math.sin(t * 2.1 + seed) + .05 * math.sin(t * 5.3 + seed * 2)
        rp = ripple * hgt * (.25 + .75 * v) * (np.sin(6.0 * u + 7.5 * v - t * 6.4 + seed) * .5 + np.sin(13. * u - 5. * v - t * 9.1 + seed * 1.7) * .28) * np.clip(np.abs(u - .5) * 2 + .25 * v, 0, 1) * wind
        P = P + bd * (belly * gust * wind * sh + rp)[..., None]
        P[..., 2] += roach * hgt * np.sin(math.pi * u) * v ** 2
        S = lambda uu, vv: np.stack([cv2.remap(P[..., i].astype(f32), (A_(vv, f32).reshape(-1, 1) * (n[1] - 1)), (A_(uu, f32).reshape(-1, 1) * (n[0] - 1)), cv2.INTER_LINEAR)[:, 0] for i in range(3)], -1).astype(float)
        i = self.grid(P, col, name, sheet=True, lw=lw, thr=.02, sh=k.pop('sh', (.70, .68, .80)), **k)
        return S, i, P

    def cloth_lines(self, S, seed=0, n=5, w=1.1, al=.6):
        """tension folds from the corners + seams"""
        r = np.random.default_rng(seed)
        for cu, cv, su, sv in ((0, 0, 1, 1), (1, 0, -1, 1), (0, 1, 1, -1), (1, 1, -1, -1)):
            for j in range(n):
                a = r.uniform(.15, 1.35); L = r.uniform(.12, .34); q = np.linspace(.03, 1, 9); self.line(S(cu + su * np.cos(a) * L * q, cv + sv * np.sin(a) * L * q), w * r.uniform(.7, 1.2), None, al)

    def stamp(self, S, part, emb, c=(.5, .5), su=.3, sv=.3):
        """emblem polygons (unit, y down) on a sail surface"""
        self.decal(part, [(S(c[0] + A_(p)[:, 0] * su, c[1] + A_(p)[:, 1] * sv), col, w) for p, col, w in emb])

    def flag(self, p0, drop, length, hgt, t, col='#17151a', emb=True, seed=0, dirn=(-1., 0, 0), name='flag', amp=.16, n=22):
        d = A_(dirn, float); d /= np.linalg.norm(d); up = A_([0, 0, 1.]); sd = np.cross(d, up); u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, 7), indexing='ij')
        w = amp * length * (u ** .8) * np.sin(7.2 * u - t * 9. + seed + .9 * v); w2 = .10 * length * u ** 1.4 * np.sin(4.1 * u - t * 5.5 + seed * 1.3) - .10 * length * u ** 2
        P = A_(p0, float) + d * (u * length * .94)[..., None] + up * (-v * hgt + w2 - drop)[..., None] + sd * w[..., None]
        i = self.grid(P, col, name, sheet=True, lw=2., thr=.0, sh=(.55, .55, .68))
        S = lambda uu, vv: np.stack([cv2.remap(P[..., k].astype(f32), (A_(vv, f32).reshape(-1, 1) * 6), (A_(uu, f32).reshape(-1, 1) * (n - 1)), cv2.INTER_LINEAR)[:, 0] for k in range(3)], -1).astype(float)
        if emb: self.stamp(S, i, emblem() if emb is True else emb, (.47, .5), .30 * hgt / length * 1.05, .38)
        return S

    def rope(self, a, b, sag=.0, w=1.1, col=None, n=12, al=.9):
        a, b = A_(a, float), A_(b, float); u = np.linspace(0, 1, n)[:, None]; P = a + (b - a) * u; P[:, 2] -= sag * np.sin(math.pi * u[:, 0]); self.line(P, w, col, al)

    def shrouds(self, top, feet, rungs=12, w=1.0):
        for q in feet: self.rope(q, top, 0, w)
        if rungs and len(feet) > 1:
            for k in range(1, rungs):
                u = k / (rungs + 1.5); self.line(A_([A_(q) + (A_(top) - A_(q)) * u for q in feet]), w * .75, None, .8)


# ------------------------------------------------------------------ emblems (unit, y down)
_el = lambda cx, cy, rx, ry, n=26, a0=0., a1=2 * math.pi: np.c_[cx + rx * np.cos(np.linspace(a0, a1, n)), cy + ry * np.sin(np.linspace(a0, a1, n))]


def _bone(a, b, w, gr=0.):
    a, b = A_(a, float), A_(b, float); w0 = w; w = w + gr; d = b - a; L = np.hypot(*d); d /= L; n = A_([-d[1], d[0]]); o = []
    for p, s in ((a, -1), (b, 1)):
        th = math.atan2(d[1], d[0]) if s > 0 else math.atan2(-d[1], -d[0])
        for k in (-1, 1): q = p + n * w0 * .95 * k * 1 + d * s * w0 * .5; o.append(_el(q[0], q[1], w * 1.1, w * 1.1, 14))
    return [np.array([a + n * w * .62, b + n * w * .62, b - n * w * .62, a - n * w * .62])] + o


def emblem(hat=True, bone='#f6f2e6', ink='#17131a', scale=1.0):
    e = []
    for a, b in (((-.82, -.62), (.82, .72)), ((.82, -.62), (-.82, .72))):
        for p in _bone(a, b, .105, .045): e.append((p, ink, 0))
        for p in _bone(a, b, .105): e.append((p, bone, 0))
    sk = np.vstack([_el(0, -.02, .52, .48, 30, math.pi * .83, math.pi * 2.17), [[.40, .34], [.31, .40], [.30, .64], [.17, .70], [0, .72], [-.17, .70], [-.30, .64], [-.31, .40], [-.40, .34]]])
    e.append((sk, bone, 1.5))
    for s in (-1, 1): e.append((_el(s * .205, .06, .135, .155), ink, 0))
    e.append((A_([[0, .20], [.05, .29], [0, .27], [-.05, .29]]), ink, 0))
    e.append((A_([[-.30, .45], [.30, .45], [.30, .475], [-.30, .475]]), ink, 0)); e.append((A_([[-.27, .565], [.27, .565], [.27, .585], [-.27, .585]]), ink, 0))
    for x in (-.18, -.06, .06, .18): e.append((A_([[x - .010, .39], [x + .010, .39], [x + .010, .68], [x - .010, .68]]), ink, 0))
    if hat:
        e.append((spl([(-.80, -.20), (-.46, -.30), (0, -.33), (.46, -.30), (.80, -.20), (.60, -.09), (0, -.03), (-.60, -.09)], True, n=6), '#e9b94c', 1.3))
        e.append((np.vstack([_el(0, -.28, .46, .46, 22, math.pi, 2 * math.pi), [[.46, -.22], [0, -.14], [-.46, -.22]]]), '#e9b94c', 1.3))
        e.append((A_([[-.465, -.34], [0, -.28], [.465, -.34], [.465, -.20], [0, -.13], [-.465, -.20]]), '#c02a2a', .8))
    return [(A_(p) * scale, c, w) for p, c, w in e]


def scar_emblem():
    e = emblem(False)
    for k in (-.07, .0, .07): e.append((A_([[.10 + k, -.16], [.14 + k, -.16], [.24 + k, .22], [.20 + k, .22]]) + [.02, 0], '#c02a2a', 0))
    return e


def gull(col='#2c5a96'):
    w = spl([(0, .10), (-.20, -.20), (-.52, -.36), (-.95, -.20), (-.60, -.18), (-.34, .04), (-.10, .40), (0, .52), (.10, .40), (.34, .04), (.60, -.18), (.95, -.20), (.52, -.36), (.20, -.20)], True, corners=(3, 7, 11, 0), n=8)
    return [(w, col, 0), (_el(0, .18, .10, .13), col, 0)]


# ------------------------------------------------------------------ hulls
class Hull:
    def __init__(s, L, B, rail, keel=-2.6, rake=2.2, stern=.74, full=.5, tum=.06, srake=1.2, bowpow=2.3):
        s.L, s.B, s.rail, s.keel, s.rake, s.stern, s.full, s.tum, s.sr, s.bp = L, B, rail, keel, rake, stern, full, tum, srake, bowpow

    def b(s, u):
        u = A_(u, float); a = s.stern + (1 - s.stern) * smooth(0, .34, u); c = np.clip(1 - np.clip((u - s.full) / (1 - s.full), 0, 1) ** s.bp, 0, 1) ** .62; return s.B * a * c + .03

    def pt(s, u, q, side=1):
        u, q = np.broadcast_arrays(A_(u, float), A_(q, float)); v = smooth(.58, 1., u) * .8 + .25 * (1 - smooth(0, .2, u))
        pr = (1 - (1 - np.clip(q, 0, 1)) ** 2.3) ** .62 * (1 - v) + np.clip(q, 0, 1) ** .95 * v; y = s.b(u) * pr * (1 - s.tum * np.clip((q - .6) / .4, 0, 2) ** 2)
        kz = s.keel * (1 - .75 * smooth(.80, 1., u) - .35 * (1 - smooth(0, .14, u))); z = kz + (s.rail(u) - kz) * q
        x = -s.L / 2 + s.L * u + s.rake * u ** 7 * (q - .25) - s.sr * (1 - u) ** 7 * (q - .3); return np.stack([x, y * side, z], -1)

    def q_at(s, u, z): k = s.keel * (1 - .75 * smooth(.80, 1., u) - .35 * (1 - smooth(0, .14, u))); return (z - k) / (s.rail(u) - k)


def hull_parts(m, h, bands, ns=44, nq=7, rail=None, planks=0, lod=2, deckz=None, deckcol='#b98d58', name='hull'):
    """bands [(q0,q1,col)], rail (col, r); deckz fn(u)->z"""
    u = np.r_[np.linspace(0, .7, ns * 6 // 10), np.linspace(.7, 1, ns * 4 // 10 + 1)[1:]]; ids = {}
    for side in (1, -1):
        for k, (q0, q1, col) in enumerate(bands):
            q = np.linspace(q0, q1, max(3, int(nq * (q1 - q0) * 2) + 2)); P = h.pt(u[:, None], q[None], side); ids[side, k] = m.grid(P, col, f'{name}{side}{k}', out=(0, 0, h.rail(.5) * .4), grp=f'{name}{side}')
            if lod > 0 and k: m.line(h.pt(u, q0 + 0 * u, side), 1.3)
        if rail: m.tube(h.pt(u, 1.0 + 0 * u, side) + [0, 0, rail[1] * .4], rail[1], rail[0], f'rail{side}', n=6, lw=1.6)
        if planks and lod > 1:
            r = np.random.default_rng(5 + side)
            for j in range(planks):
                qq = .36 + .60 * (j + .5) / planks; a = r.uniform(0, .1)
                while a < .98: b = min(.995, a + r.uniform(.05, .26)); uu = np.linspace(a, b, 14); m.line(h.pt(uu, qq + 0 * uu, side), .75, None, .42); a = b + r.uniform(.01, .10)
    q = np.linspace(0, 1, 10); G = np.stack([h.pt(0 * q, q, 1), h.pt(0 * q, q, -1)]); G = G[0][None] * (1 - np.linspace(0, 1, 5)[:, None, None]) + G[1][None] * np.linspace(0, 1, 5)[:, None, None]
    tr = m.grid(G, bands[min(1, len(bands) - 1)][2], 'transom', out=(0, 0, 2)); ids['transom'] = tr
    if deckz:
        a = np.linspace(0, 1, 5); y = np.linspace(-1, 1, 7); zz = deckz(u); P = np.zeros((len(u), 7, 3)); ed = h.pt(u, np.clip(h.q_at(u, zz), 0, .98), 1); P[..., 0] = ed[:, None, 0]; P[..., 1] = ed[:, None, 1] * y * .985; P[..., 2] = zz[:, None]
        ids['deck'] = m.grid(P, deckcol, 'deck', out=(0, 0, -50))
    return ids


def mast(m, x, z0, z1, r0, r1, col='#b27a42', y=0., name='mast'): return m.cyl((x, y, z0), (x, y, z1), r0, r1, col, name + str(x), lw=2.0)


def yard(m, x, z, half, r=.16, col='#a06a38', y=0., dx=.45): m.tube([(x + dx, y - half, z), (x + dx, y - half * .5, z), (x + dx, y, z), (x + dx, y + half * .5, z), (x + dx, y + half, z)], [r * .55, r, r * 1.15, r, r * .55], col, f'yard{x}{z}', n=7, lw=1.5)


def sq_sail(m, x, z0, z1, w0, w1, belly, t, seed=0, col='#f4ecd9', emb=None, esz=.30, wind=1., name='sail', stripes=None, lod=2, dx=.55, ec=(.5, .50), rk=0.):
    """square sail: head at z1 (width w1) foot at z0 (width w0)"""
    hx = x + dx + rk * (z1 - z0); S, i, P = m.sail((hx, -w1, z1), (hx, w1, z1), (x + dx, -w0, z0), (x + dx, w0, z0), belly, t, col, name + str(x) + str(z1), seed, wind=wind)
    if stripes:
        c, n = stripes
        for k in range(0, n, 2):
            uu = np.linspace(0, 1, 12); a, b = k / n, (k + 1) / n
            m.decal(i, [(np.vstack([S(0 * uu + a, uu), S(0 * uu + b, uu[::-1])]), c, 0)])
    elif lod > 1:
        for k in range(1, 6): uu = np.linspace(.02, .98, 16); m.line(S(0 * uu + k / 6, uu), .8, None, .28)
    if emb: asp = (w0 + w1) / abs(z1 - z0); m.stamp(S, i, emb, ec, esz, esz * asp)
    if lod > 1: m.cloth_lines(S, seed)
    if lod > 0:
        yard(m, x, z1 + .15, w1 * 1.06, dx=dx * .6)
        for sg in (-1, 1): m.rope((x + dx, sg * w0, z0), (x - 5, sg * w0 * .9, max(2., z0 * .45)), .3, .9); m.rope((x + dx * .6, sg * w1, z1), (x, 0, z1 + w1 * .42), 0, .9)
    return S, i


# ------------------------------------------------------------------ the lion ship
def lion(m, c, s=1.0, tilt=.10):
    c = A_(c, float); ca, sa = math.cos(tilt), math.sin(tilt); R = A_([[ca, 0, -sa], [0, 1, 0], [sa, 0, ca]])
    L = lambda p: A_(p, float) * s @ R.T + c
    m.tube(L([(-5.4, 0, -2.9), (-3.6, 0, -1.9), (-1.8, 0, -.7), (-.4, 0, 0)]), A_([1.7, 1.7, 1.8, 1.9]) * s, '#d8963f', 'neck', n=14)
    for k in range(16):                                                      # sun-ray mane
        a = k / 16 * 2 * math.pi + math.pi / 16; d = A_([0, math.sin(a), math.cos(a)]); tn = A_([0, math.cos(a), -math.sin(a)])
        v = np.linspace(0, 1, 9); wv = .74 * np.sin(math.pi * (.10 + .86 * v ** .8)) ** .7; th = np.linspace(0, 2 * math.pi, 11)[:-1]
        P = np.stack([np.stack([-.25 - .28 * v[j] + .30 * wv[j] * np.sin(th), d[1] * (2.05 + 1.75 * v[j]) + tn[1] * wv[j] * np.cos(th), d[2] * (2.05 + 1.75 * v[j]) + tn[2] * wv[j] * np.cos(th)], -1) for j in range(9)])
        m.grid(L(P), '#ef8a24' if k % 2 else '#f39a2e', f'mane{k}', out=L(P.reshape(-1, 3).mean(0)), wrap=True, lw=2.0, sh=(.66, .50, .55))
    m.ball(L((0, 0, 0)), A_([1.35, 2.35, 2.35]) * s, '#f6c244', 'lface', R=R, n=(18, 28), sh=(.80, .62, .52))
    f = lambda y, z, o=0.: L(A_([1.35 * math.sqrt(max(0., 1 - (y / 2.35) ** 2 - (z / 2.35) ** 2)) + o, y, z]))
    for sg in (-1, 1):
        m.ball(f(sg * .92, .52), A_([.34, .50, .50]) * s, '#fffdf2', f'leye{sg}', R=R, n=(10, 14), lw=1.8); m.ball(f(sg * .90, .50, .27), A_([.16, .23, .25]) * s, '#17131a', f'lpup{sg}', R=R, n=(8, 10), lw=0)
        m.ball(L((-.2, sg * 1.75, 2.0)), A_([.35, .62, .62]) * s, '#f0b038', f'lear{sg}', R=R, n=(9, 12))
        m.ball(f(sg * .42, -.62, .05), A_([.62, .70, .56]) * s, '#fdebb8', f'lmuz{sg}', R=R, n=(12, 16), sh=(.82, .70, .64), grp='lmuz', lw=1.6)
        u = np.linspace(0, 1, 14); m.line(A_([f(sg * (.10 + 1.35 * q), -1.02 - .42 * math.sin(math.pi * q * .62) + .78 * q * q, .05) for q in u]), 2.2)
        for k in range(3): m.line(A_([f(sg * (1.0 + .55 * q), -.48 - .18 * k + .10 * q * (1 - k), .04) for q in np.linspace(0, 1, 5)]), 1.2)
        m.line(A_([f(sg * (.55 + .75 * q), 1.12 + .22 * math.sin(math.pi * q), .03) for q in np.linspace(0, 1, 8)]), 2.0)
    m.ball(L((1.95, 0, -.32)), A_([.30, .44, .30]) * s, '#4a2a1e', 'lnose', R=R, n=(10, 14), lw=1.6)
    for sg in (-1, 1):                                                        # crossed bones under the chin
        a, b = L((.35, sg * 2.5, -3.55)), L((.35, -sg * 2.1, -1.9)); m.tube([a, (a + b) / 2, b], .30 * s, '#f6f0e0', f'bone{sg}', n=8, lw=1.6)
        for e in (a, b):
            for k in (-1, 1): m.ball(e + A_([0, .25 * k * s * sg, .28 * k * s]), .40 * s, '#f6f0e0', f'bknob{sg}', n=(8, 10), grp=f'bone{sg}', lw=1.6)


def tree(m, c, h, r, col='#3f8f3a', seed=0, trunk='#8a5a34', fruit=None, blobs=7):
    c = A_(c, float); m.tube([c, c + [.05 * h, 0, h * .35], c + [0, .04 * h, h * .62]], [r * .20, r * .14, r * .10], trunk, f'trunk{seed}', n=7, lw=1.4); rr = np.random.default_rng(seed + 3)
    for k in range(blobs):
        a = k / blobs * 6.283 + rr.uniform(0, .5); q = r * .55 * (k > 0); m.ball(c + [math.cos(a) * q, math.sin(a) * q, h * .72 + rr.uniform(-.12, .22) * h * (k > 0) + (k == 0) * h * .14], A_([1, 1, .82]) * r * rr.uniform(.45, .62), col, f'leaf{seed}', n=(8, 12), grp=f'leaf{seed}', lw=1.7, sh=(.52, .62, .62))
    if fruit:
        for k in range(12): a = rr.uniform(0, 6.283); z = rr.uniform(-.2, .3); m.ball(c + [math.cos(a) * r * .92, math.sin(a) * r * .92, h * .74 + z * h], r * .11, fruit, f'fruit{seed}', n=(6, 8), lw=1.0)


def sunny(t=0., wind=1., lod=2, sails=True, swing=None, seed=0, flag=True, cannon=0., crewless=True):
    """the lion ship.  swing = (forward angle) of the deck swing; cannon 0..1 opens the stern cannon.  L 40 m"""
    m = Model(); h = Hull(40, 6.9, lambda u: 5.6 + 2.6 * np.abs((A_(u, float) - .46) / .54) ** 2.1 + .9 * (1 - smooth(0, .3, u)), keel=-3.4, rake=2.0, stern=.80, full=.50, tum=.05, srake=1.6, bowpow=2.5)
    dz = lambda u: 4.3 + 2.0 * smooth(.735, .745, u) + 2.6 * (1 - smooth(.255, .265, u))
    ids = hull_parts(m, h, [(0, .34, '#8a3f2a'), (.34, .42, '#f4ead2'), (.42, .80, '#dc9b45'), (.80, 1.0, '#f4ead2')], rail=('#cf4a2b', .34), planks=8, lod=lod, deckz=dz, deckcol='#6cb54c', name='hull')
    m.parts[ids['transom']].col = rgb('#dc9b45')
    for side in (1, -1):                                                     # portholes + the round dock hatch
        for uu in (.14, .22, .30, .62, .70, .78):
            c = h.pt(uu, .62, side); e1 = h.pt(uu + .01, .62, side) - c; e1 /= np.linalg.norm(e1); e2 = h.pt(uu, .66, side) - c; e2 /= np.linalg.norm(e2); n = np.cross(e1, e2); n *= np.sign(n[1] * side)
            for rr, cc, w in ((.64, '#f1e4c4', 1.4), (.44, '#2a5b86', 1.0)): a = np.linspace(0, 6.283, 22); m.decal(ids[side, 2], [(c + n * .04 + np.outer(np.cos(a), e1) * rr + np.outer(np.sin(a), e2) * rr, cc, w)])
            m.decal(ids[side, 2], [(c + n * .05 + np.outer([-.28, -.05, -.22], e1) + np.outer([.05, .30, .28], e2), '#bfe4f4', 0)])
        c = h.pt(.46, .60, side); e1 = h.pt(.47, .60, side) - c; e1 /= np.linalg.norm(e1); e2 = h.pt(.46, .64, side) - c; e2 /= np.linalg.norm(e2); a = np.linspace(0, 6.283, 40); C = lambda r: c + np.outer(np.cos(a), e1) * r + np.outer(np.sin(a), e2) * r
        pol = [(C(1.75), '#cf4a2b', 1.6), (C(1.48), '#f4ead2', 1.2), (C(1.10), '#dc9b45', 1.1), (C(.34), '#cf4a2b', 1.0)]
        for k in range(8): b = k * math.pi / 4; pol.append((c + np.outer([.42 * math.cos(b - .2), 1.02 * math.cos(b - .10), 1.02 * math.cos(b + .10), .42 * math.cos(b + .2)], e1) + np.outer([.42 * math.sin(b - .2), 1.02 * math.sin(b - .10), 1.02 * math.sin(b + .10), .42 * math.sin(b + .2)], e2), '#f4ead2', .7))
        m.decal(ids[side, 2], pol)
        # rail balusters
        if lod > 1:
            for k in range(22):
                uu = .06 + .86 * k / 21; m.line(h.pt(uu, A_([.80, .995]), side), 1.0, None, .7)
    # bulkheads + aft house + roof + stairs
    b1 = 5.9; m.box((-9.6, -b1, 4.3), (-9.3, b1, 6.9), '#e7c98c', 'bulk1', lw=1.6); m.box((9.3, -b1 * .93, 4.3), (9.6, b1 * .93, 6.3), '#e7c98c', 'bulk2', lw=1.6)
    hs = m.box((-18.2, -4.3, 6.9), (-11.2, 4.3, 10.4), '#f3e6c6', 'house', lw=2.0); m.box((-18.9, -4.9, 10.4), (-10.6, 4.9, 10.95), '#cf4a2b', 'roof', lw=1.8, taper=.94); m.lathe([(4.2, 0), (4.0, .8), (3.0, 1.6), (1.6, 2.1), (0, 2.2)], (-14.6, 0, 10.9), '#c2432a', 'dome', n=20)
    m.box((-11.25, -.9, 6.9), (-11.10, .9, 9.3), '#8a5a34', 'door', lw=1.3)
    for y in (-2.9, 2.9):
        a = np.linspace(0, 6.283, 20); m.decal(hs, [(np.c_[0 * a - 11.18, y + np.cos(a) * .75, 8.5 + np.sin(a) * .75], '#8a5a34', 1.2), (np.c_[0 * a - 11.17, y + np.cos(a) * .55, 8.5 + np.sin(a) * .55], '#2f6b95', .9)])
    for sg in (-1, 1):
        for xx in (-16.4, -14.7, -13.0): a = np.linspace(0, 6.283, 20); m.decal(hs, [(np.c_[xx + np.cos(a) * .62, 0 * a + sg * 4.31, 8.7 + np.sin(a) * .62], '#8a5a34', 1.1), (np.c_[xx + np.cos(a) * .44, 0 * a + sg * 4.32, 8.7 + np.sin(a) * .44], '#2f6b95', .8)])
        for k in range(7): m.box((-9.3 + k * .42, sg * 4.0 - .8, 4.3), (-8.9 + k * .42, sg * 4.0 + .8, 6.7 - k * .36), '#c89658', f'st{sg}', grp=f'stairs{sg}', lw=.9)
    # stern: cannon + windows
    r = .62 + .5 * cannon; m.lathe([(.9 + r * .55, 0), (1.15 + r * .6, -.5), (1.1 + r * .5, -1.5), (.95 + r * .5, -2.2 - cannon), (.62 + r * .45, -2.2 - cannon), (.5 + r * .4, -1.2)], (-19.6, 0, 1.9), '#6f6a72', 'coup', axis='x', n=22, lw=2.2, sh=(.5, .5, .62))
    a = np.linspace(0, 6.283, 24)
    for y in (-3.1, 3.1): m.decal(ids['transom'], [(np.c_[0 * a - 20.4, y + np.cos(a) * .85, 5.4 + np.sin(a) * .85], '#f4ead2', 1.3), (np.c_[0 * a - 20.45, y + np.cos(a) * .6, 5.4 + np.sin(a) * .6], '#2a5b86', 1.0)])
    # lawn: mast bench, tree, swing, flower boxes
    tree(m, (-4.6, 3.2, 4.3), 8.2, 2.5, '#4f9e3f', seed); m.anchor['tree'] = (-4.6, 3.2, 9.6)
    sw = .18 * math.sin(t * 2.2) if swing is None else swing; tp = A_([-4.6, 1.4, 9.0]); br = tp + A_([math.sin(sw), 0, -math.cos(sw)]) * 3.6
    for k in (-.5, .5): m.rope(tp + [0, 0, 0] + A_([0, k, 0]), br + [0, k, 0], 0, 1.3, '#6b4a28')
    m.box(tuple(br - [.35, .7, .07]), tuple(br + [.35, .7, .07]), '#b5783c', 'swing', lw=1.4); m.anchor['swing'] = br
    m.lathe([(1.5, 0), (1.5, .5), (0, .52)], (4.2, 0, 4.3), '#a8703a', 'bench', n=18, lw=1.4)
    # masts, nest, sails
    mast(m, 4.2, 3.5, 34.5, .62, .34); mast(m, -8.4, 4, 29, .52, .26, name='mizzen')
    m.lathe([(0, 0), (1.5, .3), (2.3, 1.1), (2.35, 3.2), (2.65, 3.35), (2.55, 3.7), (1.7, 4.7), (.6, 5.3), (0, 5.4)], (4.2, 0, 28.2), '#f0dfb8', 'nest', n=22, lw=2.2); m.anchor['nest'] = (4.2, 0, 30.2)
    m.lathe([(2.75, 0), (2.62, .34), (1.75, 1.35), (.6, 2.0), (0, 2.1)], (4.2, 0, 31.5), '#cf4a2b', 'nestroof', n=22, lw=1.8); m.lathe([(2.5, 0), (2.62, .22), (2.5, .42)], (4.2, 0, 29.05), '#b5783c', 'nestring', n=22, lw=1.4)
    for k in range(8):
        b = k * math.pi / 4 + .2; e = A_([-math.sin(b), math.cos(b), 0]); c = A_([4.2 + math.cos(b) * 2.37, math.sin(b) * 2.37, 30.3]); a = np.linspace(0, 6.283, 16); nest = [i for i, p in enumerate(m.parts) if p.name == 'nest'][0]
        m.decal(nest, [(c + np.outer(np.cos(a), e) * .48 + np.outer(np.sin(a), [0, 0, 1]) * .48, '#2f6b95', 1.0)])
    m.anchor.update(mast=(4.2, 0, 34.5), prow=(23.2, 0, 8.3), deck=(0, 0, 4.4), stern=(-21, 0, 1.9), lawn=(-1, -2, 4.4), house=(-14, 0, 11), head=(22.4, 0, 12.2))
    if sails:
        S, i = sq_sail(m, 4.2, 10.2, 26.4, 10.4, 9.2, 4.6, t, 1 + seed, emb=emblem(), esz=.29, wind=wind, lod=lod, ec=(.5, .52))
        S2, i2 = sq_sail(m, -8.4, 13.2, 26.2, 8.2, 7.0, 3.4, t, 4 + seed, wind=wind, lod=lod, stripes=('#e8b7a0', 9), col='#f6efe0')
        S3, i3, _ = m.sail((-8.8, 0, 13), (-9.2, 0, 24.6), (-9.2, 0, 11.8), (-17.8, .5, 12.6), .9, t, '#f2e2c4', 'gaff', 7 + seed, bdir=(0, 1., 0), wind=wind, ripple=.03)
    for sg in (-1, 1):
        m.shrouds((4.2, 0, 28.6), [h.pt(uu, 1.0, sg) for uu in (.50, .54, .58, .62)], 14 if lod > 1 else 0); m.shrouds((-8.4, 0, 25.), [h.pt(uu, 1.0, sg) for uu in (.20, .24, .28)], 11 if lod > 1 else 0)
        m.rope((4.6, sg * 10.4, 10.2), (-8, sg * 5.6, 7.5), .5)
    m.rope((4.2, 0, 33), (19.5, 0, 8.6), .5); m.rope((4.2, 0, 27.5), (-8.4, 0, 26.5), .7); m.rope((-8.4, 0, 28), (-20, 0, 9.5), .6)
    if flag: m.flag((4.2, 0, 37.6), 0, 4.6, 3.0, t, seed=seed); m.cyl((4.2, 0, 34), (4.2, 0, 37.9), .16, .08, '#8a5a34', 'fpole', lw=1.2); m.flag((-8.4, 0, 29.6), 0, 2.6, 1.7, t, seed=seed + 2)
    lion(m, (21.2, 0, 8.6), 1.0, .06)
    m.waterline = np.vstack([h.pt(np.linspace(0, 1, 40), h.q_at(np.linspace(0, 1, 40), 0.), 1), h.pt(np.linspace(1, 0, 40), h.q_at(np.linspace(1, 0, 40), 0.), -1)]); m.hull = h
    return m


def sheep(m, c, s=1.0):
    c = A_(c, float); L = lambda p: A_(p, float) * s + c; W_ = '#f4efe2'
    m.tube(L([(-3.0, 0, -2.6), (-1.9, 0, -1.4), (-.9, 0, -.4), (0, 0, .1)]), A_([.62, .60, .62, .78]) * s, '#b98a52', 'neck', n=12, lw=2.)
    m.ball(L((.35, 0, .3)), A_([1.22, .98, 1.06]) * s, W_, 'sface', n=(18, 26), sh=(.76, .74, .84), lw=2.4); m.ball(L((1.28, 0, .02)), A_([.62, .66, .60]) * s, W_, 'smuz', n=(14, 20), grp='sface', sh=(.76, .74, .84), lw=2.4)
    for sg in (-1, 1):
        th = np.linspace(-.4, 5.6, 30); rr = .78 * (1 - .115 * th * .9); P = np.stack([-.05 - np.sin(th) * rr * .95, sg * (.86 + .13 * th * .55), .42 + np.cos(th) * rr], -1)
        m.tube(L(P), (.30 * (1 - np.linspace(0, 1, 30) * .78)) * s, '#a9763e', f'horn{sg}', n=9, lw=1.9)
        for k in range(3, 27, 3): m.line(L(P[k] + A_([[0, sg * .06, .0]]) + A_([[0, sg * .22 * (1 - k / 40) * math.sin(q), .22 * (1 - k / 40) * math.cos(q)] for q in np.linspace(.2, 2.2, 5)])), 1.0, None, .7)
        m.ball(L((1.04, sg * .60, .55)), A_([.17, .17, .22]) * s, '#17131a', f'seye{sg}', n=(9, 12), lw=0); m.ball(L((1.17, sg * .63, .63)), A_([.05, .05, .07]) * s, '#ffffff', f'sgl{sg}', n=(6, 8), lw=0, em=1.)
        m.ball(L((-.25, sg * 1.05, .25)), A_([.22, .38, .20]) * s, W_, f'sear{sg}', n=(8, 10), lw=1.6)
        u = np.linspace(0, 1, 12); m.line(L(np.stack([1.80 - .35 * u ** 2 - .0 * u, sg * (.06 + .52 * u), -.10 - .16 * np.sin(math.pi * u * .6) + .22 * u * u], -1)), 1.8)
        m.ball(L((1.80, sg * .16, .16)), A_([.06, .07, .06]) * s, '#5a3a2a', f'snos{sg}', n=(6, 8), lw=0)
    m.anchor['eyeL'] = tuple(L((1.10, .62, .50))); m.anchor['eyeR'] = tuple(L((1.10, -.62, .50))); m.anchor['head'] = tuple(L((.35, 0, 1.4)))


def merry(t=0., wind=1., lod=2, sails=True, seed=0, flag=True, burn=0.):
    """the little sheep ship, L 22 m.  burn 0..1 chars it and tears the sails"""
    m = Model(); h = Hull(22, 4.1, lambda u: 3.0 + 2.2 * np.abs((A_(u, float) - .44) / .56) ** 2.0 + 1.1 * (1 - smooth(0, .3, u)), keel=-2.0, rake=1.8, stern=.72, full=.48, srake=1.0)
    dz = lambda u: 2.3 + 1.5 * smooth(.765, .775, u) + 1.9 * (1 - smooth(.285, .295, u))
    br = '#a56a3a'; ids = hull_parts(m, h, [(0, .40, '#8a5230'), (.40, .46, '#e9d9b2'), (.46, .80, br), (.80, 1.0, '#f0e4c6')], rail=('#8a5230', .20), planks=6, lod=lod, deckz=dz, deckcol='#c69a62', ns=36)
    for side in (1, -1):
        for uu in (.50, .58, .66, .74):
            c = h.pt(uu, .63, side); e1 = h.pt(uu + .01, .63, side) - c; e1 /= np.linalg.norm(e1); e2 = h.pt(uu, .67, side) - c; e2 /= np.linalg.norm(e2); a = np.linspace(0, 6.283, 20)
            m.decal(ids[side, 2], [(c + np.outer(np.cos(a), e1) * .40 + np.outer(np.sin(a), e2) * .40, '#e9d9b2', 1.2), (c + np.outer(np.cos(a), e1) * .27 + np.outer(np.sin(a), e2) * .27, '#2c3f52', .9)])
        if lod > 1:
            for k in range(13): uu = .10 + .80 * k / 12; m.line(h.pt(uu, A_([.81, .99]), side), .9, None, .7)
    m.box((-4.9, -3.1, 2.3), (-4.7, 3.1, 4.2), '#d8b987', 'bulk', lw=1.2); m.box((6.0, -2.9, 2.3), (6.15, 2.9, 3.8), '#d8b987', 'bulk2', lw=1.2)
    hs = m.box((-9.2, -2.4, 4.2), (-5.2, 2.4, 6.2), '#f2e6cb', 'house', lw=1.8); m.box((-9.6, -2.8, 6.2), (-4.8, 2.8, 6.5), '#8a4a30', 'roof', lw=1.5); m.box((-5.22, -.5, 4.2), (-5.1, .5, 5.7), '#7b4a2a', 'door', lw=1.0)
    for y in (-1.6, 1.6): a = np.linspace(0, 6.283, 16); m.decal(hs, [(np.c_[0 * a - 5.18, y + np.cos(a) * .42, 5.2 + np.sin(a) * .42], '#7b4a2a', 1.0), (np.c_[0 * a - 5.17, y + np.cos(a) * .3, 5.2 + np.sin(a) * .3], '#f0c25a', .7)])
    for k, (x, y) in enumerate(((-8.4, 1.5), (-8.4, -1.5), (-7.0, 0.))): tree(m, (x, y, 6.4), 2.5, .95, '#3f8f3a', 20 + k, fruit='#f28a1e', blobs=5)
    mast(m, 1.2, 2, 18.6, .40, .20); mast(m, -5.6, 4, 13.6, .26, .14, name='mizzen')
    m.lathe([(.7, 0), (1.15, .25), (1.2, 1.1), (1.08, 1.1), (1.0, .4)], (1.2, 0, 16.2), '#b27a42', 'nest', n=16, lw=1.8); m.tube([(9.4, 0, 5.2), (11.2, 0, 5.8), (13.2, 0, 6.5)], [.22, .18, .12], '#a06a38', 'bowsprit', n=8, lw=1.5)
    if sails:
        c = '#f4ecd9'
        sq_sail(m, 1.2, 5.6, 14.6, 5.6, 4.9, 2.5, t, 1 + seed, col=c, emb=emblem(), esz=.29, wind=wind, lod=lod, ec=(.5, .52))
        S, i, _ = m.sail((-5.4, 0, 6.6), (-2.6, 0, 5.4), (-9.8, 0, 13.4), (-12.2, .3, 6.9), .9, t, '#f4ecd9', 'lateen', 3 + seed, bdir=(0, 1., 0), wind=wind, ripple=.03)
        for k in range(0, 8, 2): uu = np.linspace(0, 1, 10); m.decal(i, [(np.vstack([S(uu, 0 * uu + k / 8), S(uu[::-1], 0 * uu + (k + 1) / 8)]), '#d4584a', 0)])
        m.tube([(-2.2, 0, 4.8), (-6, 0, 9.2), (-10.4, 0, 14.0)], [.10, .16, .08], '#a06a38', 'lyard', n=7, lw=1.2)
    for sg in (-1, 1):
        m.shrouds((1.2, 0, 16), [h.pt(uu, 1.0, sg) for uu in (.46, .52, .58)], 11 if lod > 1 else 0, .9)
        m.rope((1.7, sg * 5.6, 5.6), (-4, sg * 3.4, 4.2), .3, .9)
    m.rope((1.2, 0, 18), (12.8, 0, 6.4), .3, .9); m.rope((1.2, 0, 17), (-5.6, 0, 13.4), .4, .9); m.rope((-5.6, 0, 13.2), (-11, 0, 5.6), .3, .9)
    if flag: m.flag((1.2, 0, 20.4), 0, 2.6, 1.7, t, seed=seed); m.cyl((1.2, 0, 18.4), (1.2, 0, 20.6), .09, .05, '#8a5a34', 'fpole', lw=1.)
    m.anchor.update(mast=(1.2, 0, 18.6), yard=(1.5, 0, 14.75), yardL=(1.5, 5.0, 14.75), yardR=(1.5, -5.0, 14.75), nest=(1.2, 0, 17.3), stern=(-11, 0, 2))
    sheep(m, (11.3, 0, 7.0), 1.05); m.waterline = np.vstack([h.pt(np.linspace(0, 1, 30), h.q_at(np.linspace(0, 1, 30), 0.), 1), h.pt(np.linspace(1, 0, 30), h.q_at(np.linspace(1, 0, 30), 0.), -1)]); m.hull = h
    return m


def dragon_prow(m, c, s=1., col='#b5322a'):
    c = A_(c, float); L = lambda p: A_(p, float) * s + c
    P = A_([(-6, 0, -4.4), (-3.2, 0, -2.6), (-1.0, 0, -.2), (-.2, 0, 2.2), (.6, 0, 4.0), (2.0, 0, 4.9), (3.4, 0, 4.8)]); u = np.linspace(0, 1, 26); P = spl(P[:, [0, 2]], n=6); P = resample(P, .42); P = np.c_[P[:, 0], 0 * P[:, 0], P[:, 1]]
    r = np.interp(np.linspace(0, 1, len(P)), [0, .3, .6, .82, 1], [1.5, 1.2, .95, .98, .8]) * s; m.tube(L(P), r, col, 'neck', n=14)
    for k in range(3, len(P) - 4, 2): m.line(L(P[k] + A_([[0, r[k] / s * math.sin(q), -r[k] / s * math.cos(q) * .9] for q in np.linspace(-1.1, 1.1, 9)])), 1.2, None, .8)
    m.ball(L((4.1, 0, 4.9)), A_([1.7, 1.0, 1.0]) * s, col, 'dhead'); m.ball(L((5.6, 0, 4.75)), A_([1.25, .72, .55]) * s, col, 'dsn', grp='dhead'); m.ball(L((5.1, 0, 4.0)), A_([1.35, .62, .30]) * s, '#8d241e', 'djaw'); m.ball(L((6.4, 0, 5.25)), A_([.40, .48, .30]) * s, col, 'dnos', grp='dhead')
    for sg in (-1, 1):
        m.tube(L([(3.6, sg * .7, 5.6), (2.6, sg * 1.3, 6.5), (1.3, sg * 1.6, 7.1), (.2, sg * 1.5, 7.9)]), A_([.30, .25, .17, .04]) * s, '#e8d6a8', f'dh{sg}', n=8); m.ball(L((4.7, sg * .82, 5.35)), A_([.28, .16, .24]) * s, '#f6e7a0', f'de{sg}', n=(8, 10)); m.ball(L((4.82, sg * .95, 5.35)), A_([.10, .06, .14]) * s, '#17131a', f'dp{sg}', n=(6, 8), lw=0)
        for k in range(4): m.tube(L([(4.7 + .42 * k, sg * .52, 4.42), (4.75 + .42 * k, sg * .53, 4.05)]), A_([.12, .01]) * s, '#fff6dc', f'dt{sg}{k}', n=6, lw=.9)
        m.tube(L([(6.3, sg * .5, 4.8), (7.4, sg * 1.2, 5.2), (8.0, sg * 1.8, 4.6), (8.5, sg * 2.3, 4.9)]), A_([.10, .09, .07, .02]) * s, '#e8c56a', f'dw{sg}', n=6, lw=1.2)
    for k in range(7): q = P[int(len(P) * (.12 + .11 * k))]; m.tube(L([q + [-.6, 0, .2], q + [-1.6, 0, 1.0 + .1 * k]]), A_([.30, .02]) * s, '#e8c56a', f'ds{k}', n=6, lw=1.1)


def galleon(m, t, L, B, hc, sails, masts, prow=None, wind=1., lod=2, seed=0, rail0=5.2, rise=2.8, emb=None, flagemb=True, house='#e7c98c', ports=7, flagcol='#17151a', rk=0.):
    """shared 2-3 mast body: hc = (below, stripe, main, top, rail, deck) colours; masts [(x/L, h, [(z0,z1,w0,w1,belly)...])]; sails = (col, stripes)"""
    h = Hull(L, B, lambda u: rail0 + rise * np.abs((A_(u, float) - .46) / .54) ** 2.1 + 1.6 * (1 - smooth(0, .3, u)), keel=-3.2, rake=2.4, stern=.72, full=.5, srake=1.6); ns = 40 if lod > 1 else 18 if lod else 12
    dz = lambda u: rail0 - 1.1 + 1.7 * smooth(.765, .775, u) + 2.5 * (1 - smooth(.255, .265, u))
    ids = hull_parts(m, h, [(0, .36, hc[0]), (.36, .42, hc[1]), (.42, .82, hc[2]), (.82, 1., hc[3])], rail=(hc[4], .30) if lod else None, planks=7, lod=lod, deckz=dz, deckcol=hc[5], ns=ns, nq=7 if lod > 1 else 4)
    if lod > 0:
        for side in (1, -1):
            for k in range(ports):
                uu = .26 + .46 * k / max(1, ports - 1); c = h.pt(uu, .62, side); e1 = h.pt(uu + .01, .62, side) - c; e1 /= np.linalg.norm(e1); e2 = h.pt(uu, .66, side) - c; e2 /= np.linalg.norm(e2); q = A_([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * .52
                m.decal(ids[side, 2], [(c + np.outer(q[:, 0], e1) + np.outer(q[:, 1], e2), '#2a2026', 1.0)])
    m.box((-L * .47, -B * .62, rail0 + 1.4), (-L * .29, B * .62, rail0 + 4.4), house, 'house', lw=1.8 if lod else 0); m.box((-L * .485, -B * .70, rail0 + 4.4), (-L * .28, B * .70, rail0 + 4.9), hc[4], 'roof', lw=1.5 if lod else 0)
    for k, (fx, hg, sl) in enumerate(masts):
        x = fx * L; mast(m, x, rail0 - 2, hg, L * .0125, L * .006)
        if lod: m.lathe([(.8, 0), (1.35, .3), (1.4, 1.2), (1.2, 1.2)], (x, 0, hg * .80), hc[4], f'top{k}', n=12, lw=1.3)
        for j, (z0, z1, w0, w1, bl) in enumerate(sl):
            e = emb if (emb and j == 0 and (k == 0 or len(masts) == 2 or emb[0][1] != '#f6f2e6')) else None
            sq_sail(m, x, z0, z1, w0, w1, bl, t, seed + k * 5 + j, col=sails[0], stripes=sails[1], emb=e, esz=.30, wind=wind, lod=lod, rk=rk)
        if lod:
            for sg in (-1, 1): m.shrouds((x, 0, hg * .80), [h.pt(fx + .5 + d, 1., sg) for d in (-.03, 0., .03)], 12 if lod > 1 else 0, .9)
        if lod or k == 0: m.flag((x, 0, hg + 2.6), 0, L * .085, L * .055, t, col=flagcol, emb=flagemb if lod else False, seed=seed + k)
    m.hull = h; m.waterline = np.vstack([h.pt(np.linspace(0, 1, 30), h.q_at(np.linspace(0, 1, 30), 0.), 1), h.pt(np.linspace(1, 0, 30), h.q_at(np.linspace(1, 0, 30), 0.), -1)]); m.anchor.update(mast=(masts[0][0] * L, 0, masts[0][1]), prow=(L * .5, 0, rail0 + rise), stern=(-L * .5, 0, 2), deck=(0, 0, rail0 - 1.1)); return m


def red_force(t=0., wind=1., lod=2, seed=0):
    m = Model(); galleon(m, t, 46, 6.6, ('#5a1f1c', '#e3b94f', '#a8302a', '#7e2421', '#e3b94f', '#b98d58'), ('#f4ecd9', None), [(.14, 37, [(11, 23, 10, 9, 4.2), (24.5, 32, 8, 6.6, 2.6)]), (-.16, 33, [(13, 23, 8.6, 7.8, 3.4), (24, 30, 6.8, 5.6, 2.2)]), (.36, 25, [(11, 20, 6.2, 5.4, 2.4)])], wind=wind, lod=lod, seed=seed, emb=scar_emblem(), flagemb=scar_emblem())
    dragon_prow(m, (25.2, 0, 8.6), 1.0); m.anchor['rail'] = (4, 6.0, 6.4); return m


def marine(t=0., wind=1., lod=2, seed=0, guns=True):
    m = Model(); galleon(m, t, 46, 6.8, ('#33507c', '#e9c35a', '#e8edf1', '#4a6ea6', '#33507c', '#c69a62'), ('#f5f3ec', None), [(.10, 35, [(11, 22, 10, 9, 4.), (23.5, 31, 8, 6.6, 2.6)]), (-.18, 32, [(13, 22, 8.6, 7.8, 3.2), (23, 29, 6.8, 5.6, 2.2)])], wind=wind, lod=lod, seed=seed, emb=gull(), flagemb=gull('#f5f3ec'), house='#dfe6ee', flagcol='#2c5a96')
    if guns:
        for sg in (-1, 1):
            for x in (9.5, 14.): m.lathe([(0, 0), (1.5, 0), (1.5, .9), (1.0, 1.3), (0, 1.4)], (x, sg * 2.4, 6.0), '#7f93ad', f'tur{sg}{x}', n=12, lw=1.4); m.cyl((x + .6, sg * 2.4, 6.8), (x + 4.0, sg * 2.4, 7.5), .28, .24, '#5b6d88', f'gun{sg}{x}', n=8, lw=1.2)
    m.ball((24.2, 0, 8.6), (2.4, .9, 1.2), '#f5f3ec', 'gullhead', lw=1.8); m.tube([(26.2, 0, 8.5), (27.8, 0, 8.1)], [.4, .04], '#e9c35a', 'beak', n=8, lw=1.3)
    return m


def barrel(m=None, c=(0, 0, 0), r=.42, h=1.1, R=None, name='barrel'):
    m = m or Model(); v = np.linspace(0, 1, 9); pr = np.c_[r * (.80 + .20 * np.sin(math.pi * v)), (v - .5) * h]; pr = np.vstack([[0, -.5 * h], pr, [0, .5 * h]]); n = 18; a = np.linspace(0, 2 * math.pi, n + 1)[:-1]; R = np.eye(3) if R is None else R; c = A_(c, float)
    P = np.stack([pr[:, :1] * np.cos(a), pr[:, :1] * np.sin(a), np.repeat(pr[:, 1:], n, 1)], -1) @ A_(R).T + c; m.grid(P, '#b07a42', name, out=c, wrap=True, lw=1.8)
    for q in (.16, .34, .66, .84):
        rr = r * (.80 + .20 * math.sin(math.pi * q)) * 1.02; aa = np.linspace(0, 6.283, 30); m.line(np.stack([rr * np.cos(aa), rr * np.sin(aa), 0 * aa + (q - .5) * h], -1) @ A_(R).T + c, 2.2 if q in (.16, .84) else 1.5, '#4a4650')
    for k in range(9): b = k / 9 * 6.283; m.line(np.stack([pr[1:-1, 0] * math.cos(b) * 1.005, pr[1:-1, 0] * math.sin(b) * 1.005, pr[1:-1, 1]], -1) @ A_(R).T + c, .8, None, .6)
    return m


def dinghy(t=0., L=4.2, col='#a8733e', oars=True, sail=False, lod=2, barrel_=False):
    m = Model(); h = Hull(L, L * .19, lambda u: .42 + .22 * np.abs((A_(u, float) - .45) / .55) ** 2, keel=-.22, rake=.3, stern=.6, full=.45, srake=.1)
    hull_parts(m, h, [(0, .5, '#8a5a30'), (.5, 1., col)], rail=('#7b4a2a', .035), planks=3, lod=lod, deckz=lambda u: 0 * u + .10, deckcol='#8f6238', ns=24)
    for x in (-L * .24, L * .05, L * .28): b = float(h.b(x / L + .5)) * .95; m.box((x - .12, -b, .30), (x + .12, b, .36), '#c08a4e', f'th{x}', lw=1.1)
    if oars:
        for sg in (-1, 1): m.cyl((.1, sg * L * .16, .46), (-.5, sg * L * .46, -.02), .03, .03, '#c08a4e', f'oar{sg}', n=6, lw=1.)
    if barrel_: barrel(m, (-L * .08, 0, .62), .30, .78)
    if sail: mast(m, .2, .1, 3.6, .05, .03); m.sail((.25, -.9, 3.4), (.25, .9, 3.4), (.25, -1.0, 1.0), (.25, 1.0, 1.0), .5, t, name='dsail')
    m.hull = h; m.waterline = np.vstack([h.pt(np.linspace(0, 1, 30), h.q_at(np.linspace(0, 1, 30), 0.), 1), h.pt(np.linspace(1, 0, 30), h.q_at(np.linspace(1, 0, 30), 0.), -1)]); return m


def generic(seed=0, t=0., lod=0, wind=1., rk=0., big=1.):
    """a pirate ship unlike its neighbours: hull, masts, sail cut, colours, stripes, emblem all from the seed"""
    r = np.random.default_rng(seed * 101 + 7); L = r.uniform(18, 46); B = L * r.uniform(.13, .17); m = Model()
    hulls = [('#5b3622', '#e2c48a', '#8a5732', '#6b4126', '#4a2c1c', '#a97c4a'), ('#3a2a2c', '#c9a24a', '#5f3d34', '#3f2a2a', '#c9a24a', '#8d6a44'), ('#6a2420', '#e9d7a8', '#a13a2c', '#7a2a24', '#e9d7a8', '#c79b62'), ('#20344f', '#e9d7a8', '#3a5f86', '#274566', '#c9d4dc', '#c79b62'), ('#2c4a35', '#e2c48a', '#4d7a4e', '#35583c', '#e2c48a', '#c79b62'), ('#6b5a3a', '#f0e6cc', '#b89a5c', '#8a7444', '#6b5a3a', '#d2ab72'), ('#3d2b4a', '#e2c48a', '#6a4a7a', '#4d3658', '#e2c48a', '#c79b62')]
    sc = ['#f3ead6', '#f6f1e4', '#e9dcc0', '#d9c7a0', '#c74a3c', '#2d2a33', '#e2b64a', '#5f8fae', '#7fa56a', '#b48ab0', '#f0d9c6', '#d97f4a']; st = ['#c74a3c', '#2d4a7a', '#2d2a33', '#4d7a4e', '#d9a23a', '#7a3a6a']
    col = sc[int(r.integers(0, len(sc)))]; stripes = (st[int(r.integers(0, len(st)))], int(r.choice([5, 7, 9]))) if r.random() < .42 else None
    nm = int(r.choice([1, 2, 2, 3])); masts = []; k = L / 46
    for j in range(nm):
        fx = [[.05], [.14, -.16], [.16, -.06, -.30]][nm - 1][j]; hg = L * r.uniform(.66, .82) * (1 - .10 * j); w = L * r.uniform(.17, .24) * (1 - .08 * j) * big; z0 = L * .24
        masts.append((fx, hg, [(z0, hg * .68, w, w * .88, w * .62), (hg * .71, hg * .93, w * .78, w * .6, w * .4)][:int(r.choice([1, 2, 2]))]))
    e = None
    if r.random() < .6: e = [(p, c if c != '#f6f2e6' else ('#2d2a33' if rgb(col).mean() > .5 else '#f6f2e6'), 0) for p, c, _ in emblem(False)]
    galleon(m, t, L, B, hulls[int(r.integers(0, len(hulls)))], (col, stripes), masts, wind=wind, lod=lod, seed=seed, rail0=5.2 * k + .8, rise=2.8 * k, emb=e, flagemb=False, rk=rk); m.L = L; return m


# ------------------------------------------------------------------ drawing
class Drawn: pass


def draw(models, cam, light=(-.5, .6, .62), ss=2, ink='pen', lw=1.0, shadow=True, rim=None, detail=1.0, dark=0., sil='#141220', bounce=None, amb=None, roi=None, para=None, edge_col='#17131a', decals=True, noink=(), lines=True, shade_amt=1., emis=None, texture=.02, thr=0.):
    L = A_(light, float); L = L / np.linalg.norm(L); Vs, Fs, As, pid, meta, wl = [], [], [], [], [], []; nv = 0; ng = 0; line_w = []
    for m, ps in models:
        R, t = ps; sc = abs(np.linalg.det(R)) ** (1 / 3); Rn = R / sc
        for p in m.parts:
            V = p.V @ R.T + t
            z = cam.cam(V)[:, 2]
            if z.max() < .1: continue
            N = p.N @ Rn.T
            if np.linalg.det(Rn) < 0 and not p.sheet: N = N
            if p.sheet: s = ((V - cam.eye) * N).sum(-1) > 0; N = np.where(s[:, None], -N, N)
            nl = N @ L; wl.append(V); Vs.append(V); Fs.append(p.F + nv); nv += len(V); pid.append(np.full(len(p.F), len(meta))); As.append(np.c_[nl, N[:, 2], V] if shadow else np.c_[nl, N[:, 2]]); meta.append((p, ng + p.grp, m))
        ng += len(m._g)
    out = Drawn(); out.layer = np.zeros((H, W, 4), f32); out.z = np.full((H, W), 1e9, f32); out.pid = np.full((H, W), -1, np.int32); out.sh = np.zeros((H, W), f32)
    if not Vs: return out
    V = np.concatenate(Vs); F = np.concatenate(Fs); A = np.concatenate(As); pid = np.concatenate(pid)
    pr, z = cam.project(V); ok = z > .1
    if roi is None:
        if (~ok).any(): x0, y0, x1, y1 = -8, -8, W + 8, H + 8
        else: x0, y0 = np.floor(pr.min(0)).astype(int) - 14; x1, y1 = np.ceil(pr.max(0)).astype(int) + 14; x0, y0, x1, y1 = max(x0, -8), max(y0, -8), min(x1, W + 8), min(y1, H + 8)
        if x1 - x0 < 2 or y1 - y0 < 2: return out
    else: x0, y0, x1, y1 = roi
    rw, rh = x1 - x0, y1 - y0; R = Raster(A.shape[1], x0, y0, rw, rh, ss); R.add(V, F, A, pid, cam)
    I = R.i; hit = I >= 0; nl = R.a[..., 0]; nz = R.a[..., 1]
    LC = np.stack([q[0].col for q in meta] + [np.zeros(3, f32)]).astype(f32); SHm = np.stack([q[0].sh for q in meta] + [np.ones(3, f32)]).astype(f32); TH = A_([q[0].thr for q in meta] + [0], f32); EM = A_([q[0].em for q in meta] + [0], f32); GR = A_([q[1] for q in meta] + [-1])
    lit = smooth(-.03, .03, nl - TH[I] - thr).astype(f32)
    if shadow:
        u = np.cross(L, [0, 0, 1.]); u /= np.linalg.norm(u) + 1e-9; v = np.cross(u, L); pu, pv, pl = V @ u, V @ v, V @ L; N_ = 1200; ext = max(np.ptp(pu), np.ptp(pv)) * 1.04; sc = N_ / ext; u0, v0 = (pu.max() + pu.min()) / 2 - ext / 2, (pv.max() + pv.min()) / 2 - ext / 2
        class O: pass
        o = O(); o.f = 1.; o.cx = 0.; o.cy = 0.; far = pl.max() + 10
        sh_ = _ortho(V, F, np.c_[(pu - u0) * sc, (pv - v0) * sc, far - pl], N_)
        P = R.a[..., 2:5]; qx = ((P @ u.astype(f32)) - u0) * sc; qy = ((P @ v.astype(f32)) - v0) * sc; qd = far - (P @ L.astype(f32))
        dm = cv2.remap(sh_, qx.astype(f32), qy.astype(f32), cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=1e6)
        cs = cv2.GaussianBlur(((qd - dm) > max(.22, ext * .006)).astype(f32), (0, 0), ss * 1.2); lit = lit * (1 - smooth(.42, .58, cs) * hit)
    lit = np.where(hit, lit, 1)
    area = lambda a: cv2.resize(a, (rw, rh), interpolation=cv2.INTER_AREA) if ss > 1 else a
    alpha = area(hit.astype(f32)); col = area(LC[I] * hit[..., None]); shm = area(np.where(hit[..., None], SHm[I], 1.)); shd = area((1 - lit) * hit * (1 - EM[I])) * shade_amt; nzz = area(nz * hit)
    inv = np.maximum(alpha, 1e-3)
    if decals:
        k0 = 0; mi = 0
        for m, ps in models:
            loc = {}
            for j, (p, g, mm) in enumerate(meta[k0:]):
                if mm is not m: break
                loc[id(p)] = k0 + j
            cnt = len(loc)
            for pi in sorted(set(d[0] for d in m.decals)):
                j = loc.get(id(m.parts[pi]))
                if j is None: continue
                pv2, pz2 = cam.project(m.parts[pi].V @ ps[0].T + ps[1])
                if (pz2 < .15).any(): lo_, hi_ = A_([0, 0]), A_([rw, rh])
                else: lo_ = np.clip(np.floor(pv2.min(0)) - [x0 + 2, y0 + 2], 0, [rw, rh]).astype(int); hi_ = np.clip(np.ceil(pv2.max(0)) - [x0 - 3, y0 - 3], 0, [rw, rh]).astype(int)
                if (hi_ - lo_).min() < 3: continue
                lo_, hi_ = lo_.astype(int), hi_.astype(int); mk = I[lo_[1] * ss:hi_[1] * ss, lo_[0] * ss:hi_[0] * ss] == j
                if mk.sum() < 24: continue
                cel = Cel(hi_[0] - lo_[0], hi_[1] - lo_[1], x0 + lo_[0], y0 + lo_[1])
                for q, polys in m.decals:
                    if q != pi: continue
                    for P3, c, w in polys:
                        p2, zz = cam.project(P3 @ ps[0].T + ps[1])
                        if (zz < .15).any() or np.ptp(p2, 0).max() < .6: continue
                        cel.fill(p2, c)
                        if w and ink == 'pen': cel.line(p2, max(.6, w * lw * .9), edge_col, closed=True, tremor=.15)
                a = cel.rgba(); mk1 = cv2.resize(mk.astype(f32), (hi_[0] - lo_[0], hi_[1] - lo_[1]), interpolation=cv2.INTER_AREA); sl = (slice(lo_[1], hi_[1]), slice(lo_[0], hi_[0])); k = a[..., 3:4] * mk1[..., None]; col[sl] = col[sl] * (1 - k) + a[..., :3] * mk1[..., None]
            k0 += cnt
    c = col * (1 - shd[..., None] * (1 - shm))
    if bounce is not None: b = rgb(bounce) if isinstance(bounce, str) else A_(bounce, f32); c = c + (np.clip(-nzz, 0, 1) * .34 * alpha)[..., None] * (b - c) * 1.0
    c = c * (1 + .045 * np.clip(nzz, -1, 1)[..., None])
    if texture: n = fbm(rh, rw, (60, 18, 6), (1, .6, .5), 4); c = c * (1 + texture * n[..., None])
    if amb is not None: c = c * A_(amb, f32)
    if dark: c = c * (1 - dark) + rgb(sil) * alpha[..., None] * dark
    lay = np.dstack([np.clip(c, 0, 2), alpha]).astype(f32)
    wz = R.w; zz = np.where(hit, 1 / np.maximum(wz, 1e-9), 1e9).astype(f32)
    if ink == 'edge':
        gm = GR[I]; e = np.zeros(gm.shape, np.uint8); e[:, 1:] |= (gm[:, 1:] != gm[:, :-1]).astype(np.uint8); e[1:] |= (gm[1:] != gm[:-1]).astype(np.uint8)
        if lw > 1.2: e = cv2.dilate(e, np.ones((2, 2), np.uint8))
        ea = area(e.astype(f32) * hit) * min(1., .85 * lw); lay[..., :3] = lay[..., :3] * (1 - ea[..., None]) + rgb(edge_col) * (ea * alpha)[..., None]
    if rim is not None:
        rc, rp, ra = rim[:3]; d = cam.R @ L; dx, dy = d[0], d[1]; n = math.hypot(dx, dy) + 1e-6; lay = kit_rim(lay, -dx / n * rp, -dy / n * rp, rc, ra, soft=.8)
    if ink == 'pen':
        cel = Cel(rw, rh, x0, y0); gm = np.where(hit, GR[I], -1); wdict = {}
        for p, g, m in meta: wdict[g] = p.lw
        ids, cnts = np.unique(gm, return_counts=True); kk = np.hanning(9); kk /= kk.sum()
        for g, cn in zip(ids, cnts):
            if g < 0 or cn < 40 or wdict.get(g, 0) <= 0: continue
            mk = (gm == g); ys = np.where(mk.any(1))[0]; xs = np.where(mk.any(0))[0]; b0, b1, a0, a1 = ys[0], ys[-1] + 1, xs[0], xs[-1] + 1; sub = np.pad(mk[b0:b1, a0:a1].astype(np.uint8), 2)
            cs, _ = cv2.findContours(sub, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
            for q, ct in enumerate(cs):
                if len(ct) < 22 * ss / 2: continue
                ct = ct[:, 0].astype(float); n = len(ct); pw = min(9, n // 3 | 1); kq = np.hanning(pw + 2)[1:-1]; kq /= kq.sum(); ee = np.vstack([ct[-pw:], ct, ct[:pw]]); s = np.stack([np.convolve(ee[:, i], kq, 'same') for i in (0, 1)], -1)[pw:-pw]
                s = (s - 2 + [a0, b0] + .5) / ss + [x0, y0]; per = n / ss; w = wdict[g] * lw * min(1., .45 + per / 500.)
                outline(cel, s, w, edge_col, seed=g * 7 + q, press=.4, tremor=.2)
        if lines:
            for m, ps in models:
                for P3, w, c, al in m.lines:
                    Pw = P3 @ ps[0].T + ps[1]; p2, zc = cam.project(Pw)
                    if (zc < .1).all(): continue
                    if (zc < .12).any(): k = zc > .12; Pw = Pw[k]; p2 = p2[k]; zc = zc[k]
                    if len(p2) < 2: continue
                    if p2[:, 0].max() < x0 or p2[:, 0].min() > x1 or p2[:, 1].max() < y0 or p2[:, 1].min() > y1: continue
                    sg = np.hypot(*np.diff(p2, axis=0).T); ln = sg.sum()
                    if ln < 3: continue
                    n = int(np.clip(ln / 3., 4, 500)); cum = np.r_[0, np.cumsum(sg)]; u = np.linspace(0, ln, n); q2 = np.c_[np.interp(u, cum, p2[:, 0]), np.interp(u, cum, p2[:, 1])]; qz = np.interp(u, cum, zc)
                    ix = np.round((q2[:, 0] - x0) * ss).astype(int); iy = np.round((q2[:, 1] - y0) * ss).astype(int); ins = (ix >= 1) & (iy >= 1) & (ix < rw * ss - 1) & (iy < rh * ss - 1); vis = np.zeros(n, bool)
                    jx, jy = ix[ins], iy[ins]; bz = np.minimum.reduce([wz[jy, jx], wz[jy + 1, jx], wz[jy - 1, jx], wz[jy, jx + 1], wz[jy, jx - 1]]); vis[ins] = qz[ins] < 1 / np.maximum(bz, 1e-9) * 1.004 + .16
                    e = np.flatnonzero(np.diff(np.r_[0, vis.astype(int), 0])); px = cam.f / max(np.median(zc), .1); wv = max(.5, w * lw * min(1.25, .45 + px / 45.))
                    for a, b in zip(e[::2], e[1::2]):
                        if b - a >= 3: cel.line(q2[a:b], wv, c or edge_col, taper=(.1, .1), press=.3, seed=a, tremor=.15, alpha=min(1., (al or 1.) * min(1., .35 + px / 22.)))
        l = cel.rgba(); lay = lay * (1 - l[..., 3:4]) + l
    sx0, sy0 = max(0, x0), max(0, y0); sx1, sy1 = min(W, x1), min(H, y1); s = (slice(sy0 - y0, sy1 - y0), slice(sx0 - x0, sx1 - x0)); d = (slice(sy0, sy1), slice(sx0, sx1))
    out.layer[d] = lay[s]; zz1 = zz[::ss, ::ss] if ss > 1 else zz; out.z[d] = zz1[s]; out.pid[d] = I[::ss, ::ss][s]; out.sh[d] = shd[s]
    k = cv2.erode(out.z, np.ones((5, 5), np.uint8)); out.z = np.where(out.z > 1e8, np.where(out.layer[..., 3] > .02, k, 1e9), out.z)
    return out


def _ortho(V, F, P, N):
    """depth map (nearest to the light) at N x N"""
    from .sea import _raster
    zb = np.zeros((N, N)); ab = np.zeros((N, N, 1), f32); ib = np.full((N, N), -1, np.int32)
    return _oz(P.astype(np.float64), np.ascontiguousarray(F.astype(np.int64)), N).astype(f32)


from numba import njit
@njit(cache=True, fastmath=True)
def _oz(P, F, N):
    out = np.full((N, N), 1e6)
    for i in range(F.shape[0]):
        a, b, c = F[i, 0], F[i, 1], F[i, 2]
        x0, y0, z0, x1, y1, z1, x2, y2, z2 = P[a, 0], P[a, 1], P[a, 2], P[b, 0], P[b, 1], P[b, 2], P[c, 0], P[c, 1], P[c, 2]
        det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(det) < 1e-9: continue
        ix0 = max(0, int(min(x0, min(x1, x2)))); ix1 = min(N - 1, int(max(x0, max(x1, x2)))); iy0 = max(0, int(min(y0, min(y1, y2)))); iy1 = min(N - 1, int(max(y0, max(y1, y2))))
        for y in range(iy0, iy1 + 1):
            for x in range(ix0, ix1 + 1):
                px = x + .5 - x0; py = y + .5 - y0; b1 = (px * (y2 - y0) - (x2 - x0) * py) / det; b2 = ((x1 - x0) * py - px * (y1 - y0)) / det
                if b1 < 0 or b2 < 0 or b1 + b2 > 1: continue
                z = z0 + b1 * (z1 - z0) + b2 * (z2 - z0)
                if z < out[y, x]: out[y, x] = z
    return out


def collar(m, ps, cam, w=26, amt=.9, t=0.):
    """screen-space foam density round a hull's waterline for Sea.render(foam=...)"""
    P = m.waterline @ ps[0].T + ps[1]; P[:, 2] = 0; p, z = cam.project(P); o = np.zeros((H, W), f32)
    if (z < .2).any(): return o
    for i in range(len(p) - 1):
        k = max(2, int(w * min(3., cam.f / z[i] / 26.))); cv2.line(o, tuple(np.round(p[i] * 8).astype(int).tolist()), tuple(np.round(p[i + 1] * 8).astype(int).tolist()), float(amt), k, cv2.LINE_AA, 3)
    return cv2.GaussianBlur(o, (0, 0), max(1.5, w * .22))
