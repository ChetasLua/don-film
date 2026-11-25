"""SEA: a real 3-D wave field (FFT ocean, three cascades, choppy crests) rasterised in perspective from any camera
(sea level, aerial, straight down) and drawn as painted anime water.   Metres, +Z up, water plane z=0, seconds.

    from op.sea import Sea, DAY, TROPIC, SUNSET, STORM, NIGHT, DAWN, glitter, sheet, drops, reflect_cam, under
    S  = Sea(seed=3, hs=1.6, wind=(1,.2))                 # hs = significant wave height, build once in setup()
    w  = S.render(cam, t, DAY, sun=(-.6,.7,.5))           # Water: w.rgb  w.a (alpha)  w.z (camera depth)  w.X w.Y (world)
    img = w.over(sky)                                     # water over a sky frame
    img = under(img, ship_layer, ship_z, w)               # composite a cel whose lower part the water hides
    S.height(x, y, t)                                     # bob a ship;  S.tilt(x,y,t,L) -> (pitch, roll)
  render(...) extras
    rings=[(x,y,radius,width,height,foam)]   travelling shock rings        wakes=[dict(x,y,head,L,W,amt)]  foam V + trail behind (x,y), head = travel angle
    foam=HxW density 0..1 (screen space: collars, splashes; it becomes lace)  mirror=RGBA premult layer of the MIRRORED scene (render it with the same
    camera from geometry with z -> -z): it is broken up by the waves     depth=fn(X,Y)->0 shallow..1 deep (tropical shelves, reefs)
    tone/detail/foam_amt/haze/lw : amounts.   step = mesh step in px (4 sea level, 10 aerial)
  glitter(cam,t,sun,...) -> RGB light (drawn lens flecks on the sun path)    sheet(cam, P[Ns,Na,3], ...) -> RGBA ballistic wall of water, lace + ink
  drops(cam, P, V, r, ...) -> RGBA drawn drops   splash_ring(...)  tri(...) / Raster = the z-buffer rasteriser (ships use it)
"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import f32, rgb, smooth, fbm, ramp, mix
from .ink import Cel, spl, noise1
from .cam import Camera3D

_c = lambda v: rgb(v) if isinstance(v, str) else np.asarray(v, f32)

# palettes: deep trough, body, mid facet, light facet, sky fleck, back-lit crest, foam, foam shade, horizon tone, haze 1/m
DAY = dict(deep='#0b3f86', body='#1464b4', mid='#2a8ad2', light='#63b9ea', sky='#c9ecfb', crest='#2fc2c4', foam='#f7fdff', foamsh='#9ccbe6', far='#7fc0e6', haze=1 / 900., ink='#0a2c62')
TROPIC = dict(deep='#067a92', body='#12a7b4', mid='#35c6c6', light='#7fe4d8', sky='#e2fbf2', crest='#5fe6cf', foam='#fbfffb', foamsh='#a9e2dc', far='#8fdde0', haze=1 / 4000., ink='#045a72',
              shallow='#86ecd0', deepc='#0a5f9c')
SUNSET = dict(deep='#3b2a5c', body='#7a4468', mid='#c4705f', light='#f2a860', sky='#ffe3a0', crest='#e88a58', foam='#fff3dc', foamsh='#d79a8a', far='#f3b878', haze=1 / 700., ink='#2c1a44')
STORM = dict(deep='#0b1a2e', body='#163450', mid='#28536f', light='#4f8296', sky='#a3c3c8', crest='#3f8f8f', foam='#e9f3f2', foamsh='#7f9fb0', far='#3f566b', haze=1 / 350., ink='#070f1f')
NIGHT = dict(deep='#04060f', body='#080d1c', mid='#101a30', light='#22314d', sky='#51658a', crest='#152a44', foam='#cdd9ec', foamsh='#5f7191', far='#1a2439', haze=1 / 500., ink='#03040a')
DAWN = dict(deep='#8f7f74', body='#b7a48f', mid='#d3c0a5', light='#ead9bd', sky='#f6ead2', crest='#dcae8c', foam='#fbf3e2', foamsh='#c7b39c', far='#eadcc3', haze=1 / 800., ink='#6c5c52')


# =================================================================== z-buffer rasteriser
@njit(cache=True, fastmath=True)
def _tri(x0, y0, w0, x1, y1, w1, x2, y2, w2, a0, a1, a2, fid, zb, ab, ib):
    h = zb.shape[0]; w = zb.shape[1]; K = ab.shape[2]
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    if abs(det) < 1e-7: return
    xa = min(x0, min(x1, x2)); xb = max(x0, max(x1, x2)); ya = min(y0, min(y1, y2)); yb = max(y0, max(y1, y2))
    if xb < 0 or yb < 0 or xa > w or ya > h: return
    ix0 = max(0, int(xa)); ix1 = min(w - 1, int(xb)); iy0 = max(0, int(ya)); iy1 = min(h - 1, int(yb))
    inv = 1.0 / det; e = -1e-5
    d1x = (y2 - y0) * inv; d2x = -(y1 - y0) * inv
    for y in range(iy0, iy1 + 1):
        py = y + .5 - y0
        b1 = ((ix0 + .5 - x0) * (y2 - y0) - (x2 - x0) * py) * inv
        b2 = ((x1 - x0) * py - (ix0 + .5 - x0) * (y1 - y0)) * inv
        for x in range(ix0, ix1 + 1):
            b0 = 1 - b1 - b2
            if b0 >= e and b1 >= e and b2 >= e:
                wz = b0 * w0 + b1 * w1 + b2 * w2
                if wz > zb[y, x]:
                    zb[y, x] = wz; ib[y, x] = fid; q = 1.0 / wz
                    k0 = b0 * w0 * q; k1 = b1 * w1 * q; k2 = b2 * w2 * q
                    for k in range(K): ab[y, x, k] = k0 * a0[k] + k1 * a1[k] + k2 * a2[k]
            b1 += d1x; b2 += d2x


@njit(cache=True, fastmath=True)
def _raster(C, A, F, fid, f, cx, cy, near, zb, ab, ib):
    """C camera-space verts (n,3), A attributes (n,k), F faces, fid per face id; zb = 1/z (0 empty)"""
    K = A.shape[1]
    q = np.empty((4, 3)); aq = np.empty((4, K))
    for i in range(F.shape[0]):
        a, b, c = F[i, 0], F[i, 1], F[i, 2]
        za, zb_, zc = C[a, 2], C[b, 2], C[c, 2]
        n_in = (za > near) + (zb_ > near) + (zc > near)
        if n_in == 0: continue
        if n_in == 3:
            wa = 1. / za; wb = 1. / zb_; wc = 1. / zc
            _tri(C[a, 0] * wa * f + cx, C[a, 1] * wa * f + cy, wa, C[b, 0] * wb * f + cx, C[b, 1] * wb * f + cy, wb,
                 C[c, 0] * wc * f + cx, C[c, 1] * wc * f + cy, wc, A[a], A[b], A[c], fid[i], zb, ab, ib)
            continue
        idx = (a, b, c); m = 0
        for j in range(3):
            p = idx[j]; s = idx[(j + 1) % 3]; zp = C[p, 2]; zs = C[s, 2]
            if zp > near:
                for k in range(3): q[m, k] = C[p, k]
                for k in range(K): aq[m, k] = A[p, k]
                m += 1
            if (zp > near) != (zs > near):
                t = (near - zp) / (zs - zp)
                for k in range(3): q[m, k] = C[p, k] + (C[s, k] - C[p, k]) * t
                for k in range(K): aq[m, k] = A[p, k] + (A[s, k] - A[p, k]) * t
                q[m, 2] = near * 1.0001; m += 1
        for j in range(1, m - 1):
            w0 = 1. / q[0, 2]; w1 = 1. / q[j, 2]; w2 = 1. / q[j + 1, 2]
            _tri(q[0, 0] * w0 * f + cx, q[0, 1] * w0 * f + cy, w0, q[j, 0] * w1 * f + cx, q[j, 1] * w1 * f + cy, w1,
                 q[j + 1, 0] * w2 * f + cx, q[j + 1, 1] * w2 * f + cy, w2, aq[0], aq[j], aq[j + 1], fid[i], zb, ab, ib)


class Raster:
    """z-buffer over a screen ROI at supersampling ss.  r.add(V, F, A, fid, cam) ; r.w = 1/z, r.a attributes, r.i face ids (-1 empty)"""
    def __init__(self, k, ox=0, oy=0, w=W, h=H, ss=1):
        self.ox, self.oy, self.w0, self.h0, self.ss = int(ox), int(oy), int(w), int(h), ss
        self.w = np.zeros((self.h0 * ss, self.w0 * ss)); self.a = np.zeros((self.h0 * ss, self.w0 * ss, k), f32); self.i = np.full((self.h0 * ss, self.w0 * ss), -1, np.int32)

    def add(self, V, F, A, fid, cam, near=.08):
        F = np.asarray(F, np.int64); fid = np.broadcast_to(np.asarray(fid, np.int64), (len(F),)).copy()
        _raster(np.ascontiguousarray(cam.cam(V), np.float64), np.ascontiguousarray(A, np.float64), np.ascontiguousarray(F), fid,
                float(cam.f * self.ss), float((cam.cx - self.ox) * self.ss), float((cam.cy - self.oy) * self.ss), near, self.w, self.a, self.i)
        return self


def grid_faces(n, m, off=0, wrap=False):
    """two triangles per cell of an n x m vertex grid (row major)"""
    I = np.arange(n * m).reshape(n, m) + off
    if wrap: I = np.concatenate([I, I[:, :1]], 1)
    a, b, c, d = I[:-1, :-1].ravel(), I[:-1, 1:].ravel(), I[1:, :-1].ravel(), I[1:, 1:].ravel()
    return np.concatenate([np.stack([a, b, d], 1), np.stack([a, d, c], 1)])


# =================================================================== textures
def _pad(T, p=3): return np.pad(T, ((p, p), (p, p)) + ((0, 0),) * (T.ndim - 2), mode='wrap')


def _samp(Tp, X, Y, L, p=3, inter=cv2.INTER_LINEAR):
    n = Tp.shape[0] - 2 * p; k = n / L
    return cv2.remap(Tp, (np.mod(X * k, n) + p).astype(f32), (np.mod(Y * k, n) + p).astype(f32), inter, borderMode=cv2.BORDER_REPLICATE)


@njit(cache=True, fastmath=True)
def _worley(N, n, P, out):
    for y in range(N):
        for x in range(N):
            u = x * n / N; v = y * n / N; iu = int(u); iv = int(v); d1 = 9.; d2 = 9.
            for dy in range(-2, 3):
                for dx in range(-2, 3):
                    q = P[(iv + dy) % n, (iu + dx) % n]
                    px = iu + dx + q[0] - u; py = iv + dy + q[1] - v; d = math.sqrt(px * px + py * py)
                    if d < d1: d2 = d1; d1 = d
                    elif d < d2: d2 = d
            out[y, x] = d2 - d1


_lace = {}
def lace(kind=0):
    """tileable foam net (512, wraps): 1 on the strands, 0 in the holes"""
    if kind not in _lace:
        r = np.random.default_rng(71 + kind); N = 512; o = np.zeros((N, N)); n = (11, 19)[kind % 2]
        _worley(N, n, r.uniform(.08, .92, (n, n, 2)), o)
        o = 1 - np.clip(o / .62, 0, 1) ** .72
        n2 = n * 2 + 3; o2 = np.zeros((N, N)); _worley(N, n2, r.uniform(.1, .9, (n2, n2, 2)), o2); o2 = 1 - np.clip(o2 / .7, 0, 1) ** .8
        # domain warp so strands are drawn, not straight
        g = np.fft.fft2(r.standard_normal((N, N))); k = np.hypot(*np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N))
        wx = np.real(np.fft.ifft2(g * np.exp(-(k / 9.) ** 2))); wy = np.real(np.fft.ifft2(np.roll(g, 7, 0) * np.exp(-(k / 9.) ** 2)))
        wx = wx / wx.std() * 9; wy = wy / wy.std() * 9; y, x = np.mgrid[:N, :N]
        q = lambda a: cv2.remap(_pad(a.astype(f32), 40), (x + wx + 40).astype(f32), (y + wy + 40).astype(f32), cv2.INTER_LINEAR)
        _lace[kind] = _pad(np.maximum(q(o), q(o2) * .72).astype(f32))
    return _lace[kind]


_pn = None
def pnoise():
    """tileable smooth paint noise (512), zero mean unit variance"""
    global _pn
    if _pn is None:
        r = np.random.default_rng(5); N = 512; k = np.hypot(*np.meshgrid(np.fft.fftfreq(N) * N, np.fft.fftfreq(N) * N)); o = 0
        for b, a in ((5, 1.), (14, .6), (40, .35)): o = o + a * np.real(np.fft.ifft2(np.fft.fft2(r.standard_normal((N, N))) * np.exp(-(k / b) ** 2))) * N / b
        _pn = _pad(((o - o.mean()) / o.std()).astype(f32))
    return _pn


# =================================================================== wakes (screen loop, world pattern)
@njit(cache=True, fastmath=True)
def _wake(X, Y, out, box, wk, lc, L):
    n = lc.shape[0] - 6
    for i in range(wk.shape[0]):
        x, y, hd, ln, wd, am, dec, arm_ = wk[i, 0], wk[i, 1], wk[i, 2], wk[i, 3], wk[i, 4], wk[i, 5], wk[i, 6], wk[i, 7]
        c = math.cos(hd); s = math.sin(hd)
        for py in range(box[i, 1], box[i, 3]):
            for px in range(box[i, 0], box[i, 2]):
                dx = X[py, px] - x; dy = Y[py, px] - y
                a = -(dx * c + dy * s); b = abs(-dx * s + dy * c)
                if a < -wd * 1.2 or a > ln or b > wd * 2.4 + max(a, 0.) * .22: continue
                q = a / ln
                d = 0.0
                if a > 0:
                    core = wd * (.42 + .25 * min(1., a / (wd * 4))) + a * .009           # turbulent trail
                    e = 1 - b / core
                    if e > 0: d = (1 - q) ** dec * min(1., e * 1.7) * 1.15
                    arm = wd * .55 + a * .135                                             # V arms with feather scallops
                    th = wd * .16 + a * .020
                    k = 1 - abs(b - arm) / th
                    if k > 0:
                        sc = .62 + .38 * math.sin(a / (wd * .55) + b / wd * 1.6)
                        d2 = k * (1 - q) ** 1.7 * sc * 1.05 * arm_ * math.exp(-a / (wd * 14.))
                        if d2 > d: d = d2
                else:
                    r = math.sqrt((a / 1.2) ** 2 + b * b)                                 # bow moustache
                    k = 1 - abs(r - wd * .62) / (wd * .28)
                    if k > 0 and k > d: d = k * .9
                d *= am
                if d > out[py, px]: out[py, px] = d


# =================================================================== the ocean
class Water:
    def over(self, base):
        return base * (1 - self.a[..., None]) + self.rgb * self.a[..., None]


class Sea:
    def __init__(self, seed=0, hs=1.4, wind=(1., .25), length=46., chop=1.0, spread=3.0, rough=1.0, N=256, swell=0.):
        """hs significant height (m), length = dominant wavelength (m), chop = crest sharpness, spread = directional focus (higher = long-crested)"""
        self.hs, self.chop = hs, chop; self.C = []; r = np.random.default_rng(seed + 101); wd = np.asarray(wind, float); wd /= np.hypot(*wd); self.wd = wd
        kp = 2 * math.pi / length
        bands = ((length * 4.2, length * .22, 1e9, 1.), (length * .83, length * .05, length * .22, rough), (length * .17, length * .0105, length * .05, rough * .9))
        tot = 0
        for T, lo, hi, rg in bands:
            k1 = np.fft.fftfreq(N) * N * 2 * math.pi / T; kx, ky = np.meshgrid(k1, k1); k = np.hypot(kx, ky); k[0, 0] = 1e-6
            lam = 2 * math.pi / k; bp = smooth(lo * .9, lo * 1.15, lam) * (1 - smooth(hi * .9, hi * 1.15, lam)) if hi < 1e8 else smooth(lo * .9, lo * 1.15, lam)
            c = (kx * wd[0] + ky * wd[1]) / k; dr = np.where(c > 0, c ** (2 * spread), .02 * np.abs(c) ** 2)
            S = np.exp(-1.25 * (kp / k) ** 2) / k ** 4 * dr * bp * (2 * math.pi / T) ** 2; S[0, 0] = 0
            h0 = (r.standard_normal((N, N)) + 1j * r.standard_normal((N, N))) * np.sqrt(S / 2) * rg
            self.C.append(dict(T=T, kx=kx, ky=ky, k=k, h0=h0, om=np.sqrt(9.81 * k), lo=lo, N=N)); tot += 2 * (np.abs(h0) ** 2).sum()
        sc = hs / 4 / math.sqrt(tot + 1e-12)
        for c in self.C: c['h0'] = c['h0'] * sc
        self._t = None; self._f = None

    def _spec(self, c, t):
        e = np.exp(1j * c['om'] * t); hm = np.conj(np.roll(np.flip(c['h0'], (0, 1)), 1, (0, 1)))
        return c['h0'] * e + hm * np.conj(e)

    def fields(self, t):
        """per cascade: padded texture [h, sx, sy, foam] and [dx, dy], slope rms"""
        if self._t == t: return self._f
        out = []; lam = self.chop
        for ci, c in enumerate(self.C):
            n2 = c['N'] ** 2; F = lambda q: np.real(np.fft.ifft2(q)) * n2
            h = self._spec(c, t); kx, ky, k = c['kx'], c['ky'], c['k']
            hh = F(h); dx, dy = F(-1j * kx / k * h), F(-1j * ky / k * h)
            jx, jy, jc = F(kx * kx / k * h), F(ky * ky / k * h), F(kx * ky / k * h)
            sx, sy = F(1j * kx * h), F(1j * ky * h)
            sx = sx / np.maximum(.25, 1 - lam * jx); sy = sy / np.maximum(.25, 1 - lam * jy)
            fold = lam * (jx + jy) - lam * lam * (jx * jy - jc * jc)
            if ci < 2:
                for tau in (.35, .8, 1.5, 2.4):
                    q = self._spec(c, t - tau); a, b, d = F(kx * kx / k * q), F(ky * ky / k * q), F(kx * ky / k * q)
                    fold = np.maximum(fold, (lam * (a + b) - lam * lam * (a * b - d * d)) * math.exp(-tau * .42))
            out.append((_pad(np.stack([hh, sx, sy, fold], -1).astype(f32)), _pad(np.stack([-lam * dx * -1, -lam * dy * -1], -1).astype(f32)), float(np.sqrt((sx ** 2 + sy ** 2).mean() / 2)), float(fold.std())))
        self._t, self._f = t, out
        return out

    def disp(self, X, Y, t, wt=(1, 1, 1)):
        """displaced surface point of parameter (X,Y): dx, dy, h"""
        X = np.asarray(X, f32); Y = np.asarray(Y, f32); sh = X.shape; X = X.reshape(sh[0] if X.ndim else 1, -1); Y = Y.reshape(X.shape)
        dx = np.zeros(X.shape, f32); dy = dx.copy(); h = dx.copy()
        for c, (A, D, _, _), w in zip(self.C, self.fields(t), wt):
            if np.isscalar(w) and w == 0: continue
            if np.ndim(w): w = np.asarray(w, f32).reshape(X.shape)
            q = _samp(A, X, Y, c['T']); d = _samp(D, X, Y, c['T'])
            h += q[..., 0] * w; dx += d[..., 0] * w; dy += d[..., 1] * w
        return dx.reshape(sh), dy.reshape(sh), h.reshape(sh)

    def height(self, x, y, t):
        x = np.atleast_1d(np.asarray(x, f32)); y = np.atleast_1d(np.asarray(y, f32))
        dx, dy, h = self.disp(x, y, t, (1, 1, 0))
        dx, dy, h = self.disp(x - dx, y - dy, t, (1, 1, 0))
        return float(h[0]) if h.size == 1 else h

    def tilt(self, x, y, t, L=20., B=8., head=0.):
        """(pitch, roll, z) of a hull of length L, beam B at heading angle head"""
        c, s = math.cos(head), math.sin(head)
        a, b = self.height(x + c * L * .35, y + s * L * .35, t), self.height(x - c * L * .35, y - s * L * .35, t)
        p, q = self.height(x - s * B * .5, y + c * B * .5, t), self.height(x + s * B * .5, y - c * B * .5, t)
        return math.atan2(a - b, L * .7), math.atan2(p - q, B), (a + b + p + q) / 4

    # ---------------------------------------------------------------- mesh
    def _mesh(self, cam, t, step, far, rings):
        e = cam.eye; hgt = max(.15, e[2])
        ys, xs = np.mgrid[-.14:1.24:40j, -.12:1.12:60j]
        d = np.stack([(xs * cam.w - cam.cx) / cam.f, (ys * cam.h - cam.cy) / cam.f, np.ones_like(xs)], -1) @ cam.R
        az = np.arctan2(d[..., 0], d[..., 1]); dep = np.arctan2(-d[..., 2], np.hypot(d[..., 0], d[..., 1])); fa = math.atan2(cam.R[2, 0], cam.R[2, 1])
        if abs(cam.R[2, 2]) > .98: fa = math.atan2(cam.R[1, 0], cam.R[1, 1])
        nad, zn = cam.project(e - np.array([0, 0, 1.]))
        nin = zn > 0 and -.15 * cam.w < nad[0] < 1.15 * cam.w and -.15 * cam.h < nad[1] < 1.25 * cam.h
        dm = dep > 1e-4
        if not dm.any(): return None
        dmax = math.pi / 2 - 1e-3 if nin else min(math.pi / 2 - .01, dep.max() + .10); dmin = max(hgt / far, (dep[dm].min() - .06))
        if nin: a0, a1 = -math.pi, math.pi
        else:
            q = (az[dm] - fa + math.pi) % (2 * math.pi) - math.pi; a0, a1 = q.min() - .05, q.max() + .05
        dth = step / cam.f
        while True:
            rr = [hgt / math.tan(dmax)]; rmax = hgt / math.tan(dmin)
            while rr[-1] < rmax and len(rr) < 2400:
                x = rr[-1]; dd = (x * x + hgt * hgt) * dth / hgt
                rr.append(x + (min(dd, max(step * .22, .022 * x)) if x < 900 else dd if dd < .07 * x else .07 * x))
            na = int((a1 - a0) / dth) + 2
            if len(rr) * na < 420000: break
            dth *= 1.25; step *= 1.25
        rr = np.array(rr); A = np.linspace(a0, a1, na) + fa
        X = (e[0] + rr[:, None] * np.sin(A)[None]).astype(f32); Y = (e[1] + rr[:, None] * np.cos(A)[None]).astype(f32)
        wts = []
        for c in self.C:          # mesh LOD: row spacing against the band
            dr = np.gradient(rr); wts.append(np.broadcast_to(np.clip(1.6 - dr / (c['lo'] * .55), 0, 1)[:, None], X.shape))
        dx, dy, h = self.disp(X, Y, t, (wts[0], wts[1], 0))
        for (rx, ry, rad, wd, ht, _) in rings:
            rd = np.hypot(X - rx, Y - ry); h = h + ht * np.exp(-((rd - rad) / wd) ** 2).astype(f32)
        V = np.stack([X + dx, Y + dy, h], -1).reshape(-1, 3); return V, np.stack([X.ravel(), Y.ravel(), h.ravel()], -1), grid_faces(len(rr), na)

    # ---------------------------------------------------------------- picture
    def render(self, cam, t, pal=DAY, sun=(-.6, .7, .5), step=4, far=9000., rings=(), wakes=(), foam=None, mirror=None, mirror_amt=1., mirror_k=1., depth=None, foam_amt=1., tone=1., detail=1.,
               haze=1., lace_m=7., light=None, spec=1., edges=1., aa=True, crest=1.):
        o = Water(); h, w = cam.h, cam.w; R = Raster(3, 0, 0, w, h)
        M = self._mesh(cam, t, step, far, rings)
        o.rgb = np.zeros((h, w, 3), f32); o.a = np.zeros((h, w), f32); o.z = np.full((h, w), 1e9, f32); o.X = np.zeros((h, w), f32); o.Y = o.X.copy(); o.foam = o.X.copy()
        if M is None: return o
        R.add(M[0], M[2], M[1], 0, cam)
        hit = R.i >= 0
        if not hit.any(): return o
        rows = np.where(hit.any(1))[0]; y0, y1 = rows[0], rows[-1] + 1; sl = slice(y0, y1)
        m = hit[sl]; wz = R.w[sl]; z = np.where(m, 1 / np.maximum(wz, 1e-9), 1e6).astype(f32); X, Y, ht = [np.ascontiguousarray(R.a[sl][..., i]).astype(f32) for i in range(3)]
        yy, xx = np.mgrid[y0:y1, 0:w].astype(f32)
        d = np.stack([(xx - cam.cx) / cam.f, (yy - cam.cy) / cam.f, np.ones_like(xx)], -1) @ cam.R.astype(f32)
        dn = np.sqrt((d ** 2).sum(-1)); dirv = d / dn[..., None]; cosv = np.clip(-dirv[..., 2], .004, 1)
        hd = np.hypot(dirv[..., 0], dirv[..., 1]) + 1e-5; vx, vy = dirv[..., 0] / hd, dirv[..., 1] / hd
        fp = z * dn / cam.f; fpd = fp / cosv
        S = np.asarray(sun, float); S = S / np.linalg.norm(S); sh = math.hypot(S[0], S[1]) + 1e-6; sxn, syn = float(S[0] / sh), float(S[1] / sh)
        sx = np.zeros_like(z); sy = sx.copy(); fo = sx.copy(); var = sx.copy() + 1e-9; det = (1., .62 * detail, .34 * detail); low = None
        for ci, (c, (A, D, rms, fs)) in enumerate(zip(self.C, self.fields(t))):
            wt = np.clip(1.35 - fpd / (c['lo'] * 2.2), 0, 1) * np.clip(1.35 - fp / (c['lo'] * .6), 0, 1) * det[ci]
            if wt.max() <= 0: continue
            q = _samp(A, X, Y, c['T'], inter=cv2.INTER_CUBIC if ci == 0 else cv2.INTER_LINEAR)
            sx += q[..., 1] * wt; sy += q[..., 2] * wt; var += (rms * wt) ** 2
            if ci < 2: fo = np.maximum(fo, q[..., 3] / .20 * np.minimum(wt / det[ci] * 1.5, 1) * (1. if ci == 0 else .72))
            if ci == 0: low = q[..., 1:3] * wt[..., None]
        for (rx, ry, rad, wd, hgt, fa) in rings:
            ddx, ddy = X - rx, Y - ry; rd = np.hypot(ddx, ddy) + 1e-3; q = (rd - rad) / wd; g = np.exp(-q * q); k = -2 * q / wd * hgt * g / rd
            sx += ddx * k; sy += ddy * k; fo = np.maximum(fo, (g * 4.2 * fa * (1 + .25 * np.where(q > 0, 1, -q))))
        rms = np.sqrt(var)
        pn = _samp(pnoise(), X, Y, max(6., self.C[1]['T'] * 1.7)); pn2 = _samp(pnoise(), X + 31, Y - 17, self.C[0]['T'] * 1.3)
        grz = np.clip(1.15 - cosv * .9, .25, 1)
        val = ((sx * vx + sy * vy) * -1. * grz * 0 + (-(sx * vx + sy * vy)) * -grz)         # tilted away from the eye = sky
        val = (val - (sx * sxn + sy * syn) * .75) / (rms * 1.05)
        val = (val * tone + .16 * pn * np.clip(1.5 - fp * 6, 0.15, 1) + .18 * pn2) * np.clip(rms / (self.fields(t)[0][2] * .30), 0.12, 1)
        val = val.astype(f32)
        # crisp, antialiased paint edges
        gr = cv2.GaussianBlur(np.abs(cv2.Sobel(val, cv2.CV_32F, 1, 0, scale=.125)) + np.abs(cv2.Sobel(val, cv2.CV_32F, 0, 1, scale=.125)), (0, 0), 1.2) + .012
        st = lambda th, k=.8: np.clip((val - th) / (gr * k * 2) + .5, 0, 1)[..., None]
        P = {k: _c(v) for k, v in pal.items() if k != 'haze'}
        deep, body, mid, light_ = P['deep'], P['body'], P['mid'], P['light']
        if depth is not None:
            dp = np.clip(depth(X, Y), 0, 1)[..., None].astype(f32); shl, dpc = P.get('shallow', light_), P.get('deepc', deep)
            g3 = lambda c, k: (shl * (.75 + .25 * k) + (c - shl) * smooth(0., .5, dp)) * (1 - smooth(.5, 1., dp)) + (c * .25 + dpc * .75 * (.85 + .3 * k)) * smooth(.5, 1., dp)
            deep, body, mid, light_ = g3(deep, 0.), g3(body, .35), g3(mid, .7), g3(light_, 1.)
        col = deep + (body - deep) * st(-.85) + 0
        col = col + (mid - col) * st(.42); col = col + (light_ - col) * st(1.18)
        col = col * (1 + .05 * np.clip(val, -1.5, 1.5)[..., None] + .022 * pn2[..., None])
        # translucent back-lit crest
        hn = ht / max(self.hs * .5, .05); tr = smooth(.35, 1.15, hn + .25 * np.clip(val, -1, 1)) * np.clip(1.3 - fpd / 2.2, 0, 1) * crest
        back = np.clip(.45 + .55 * (vx * sxn + vy * syn), .15, 1)
        col = col + (P['crest'] - col) * (tr * back * .62)[..., None]
        # sky fleck on the steepest facets
        col = col + (P['sky'] - col) * st(1.95, .7) * .75 * np.clip(1.4 - fp * 2., .2, 1)[..., None]
        if mirror is not None:
            amp = (cam.f / np.maximum(z, 1.))
            nx = sx * float(cam.R[0, 0]) + sy * float(cam.R[0, 1]); ny = sx * vx + sy * vy
            mx = xx + nx * amp * 1.5 * mirror_k; my = yy + ny * amp * (2.2 + 1.2 * hd) * mirror_k
            q = cv2.remap(mirror.astype(f32), mx.astype(f32), my.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            fr = np.clip(.25 + .85 * (1 - cosv) ** 2.2, 0, 1)
            k = (np.clip(fr + .16 * np.clip(val, -1.4, 1.2), 0, 1) * mirror_amt)[..., None]
            col = col * (1 - q[..., 3:4] * k) + q[..., :3] * k
        # foam lace
        dens = np.clip((fo - 1.55) * .42, 0, .80) * foam_amt
        if len(wakes):
            wk = np.array([[q['x'], q['y'], q['head'], q.get('L', 60.), q.get('W', 6.), q.get('amt', 1.), q.get('dec', 1.25), q.get('arm', 1.)] for q in wakes]); bx = np.zeros((len(wk), 4), np.int64); wo = np.zeros_like(z)
            for i, (x, y, hh, L, wd) in enumerate(wk[:, :5]):
                c, s = math.cos(hh), math.sin(hh); b2 = wd * 2.4 + L * .22; pts = [(x + c * wd * 1.3 + sg * -s * wd * 2.4, y + s * wd * 1.3 + sg * c * wd * 2.4, 0) for sg in (-1, 1)] + [(x - c * L * k - s * sg * b2, y - s * L * k + c * sg * b2, 0) for sg in (-1, 1) for k in (.5, 1)]
                p, zz = cam.project(np.array(pts))
                if (zz < .3).any(): bx[i] = (0, 0, w, y1 - y0); continue
                lo, hi = p.min(0) - 4, p.max(0) + 4; bx[i] = (max(0, int(lo[0])), max(0, int(lo[1]) - y0), min(w, int(hi[0])), min(y1 - y0, int(hi[1]) - y0))
            _wake(X, Y, wo, bx, wk, lace(0), lace_m); dens = np.maximum(dens, wo)
        if foam is not None: dens = np.maximum(dens, foam[sl])
        dens = np.minimum(dens, .80 + .10 * pn)
        lm = lace_m; l0 = _samp(lace(0), X, Y, lm); l1 = _samp(lace(1), X + 3.1, Y + 7.7, lm * .37)
        kf = np.clip(1.5 - fpd / (lm * .10), 0, 1); kn = np.clip(1.4 - fp / (lm * .012), 0, 1)
        la = (l0 * (1 - .4 * kn) + l1 * .4 * kn) * kf + .45 * (1 - kf); la = la + .10 * pn * kf
        fv = (dens * 1.25 - (1 - la) * .95).astype(f32)
        gf = cv2.GaussianBlur(np.abs(cv2.Sobel(fv, cv2.CV_32F, 1, 0, scale=.125)) + np.abs(cv2.Sobel(fv, cv2.CV_32F, 0, 1, scale=.125)), (0, 0), .8) + .02
        fa = np.clip((fv - .06) / (gf * 2) + .5, 0, 1) * smooth(.02, .12, dens)
        sd = cv2.warpAffine(fa, np.float32([[1, 0, 0], [0, 1, 2.2]]), (w, y1 - y0))
        col = col + (deep * .8 - col) * (np.clip(sd - fa, 0, 1) * .35 * kf)[..., None]         # painted contact line under foam
        fsh = np.clip((val * .42 + (la - .42) * 1.5 + dens * .5 - .22 + .16 * pn) / (.10 + gf) + .5, 0, 1)[..., None]
        col = col * (1 - fa[..., None]) + (P['foamsh'] + (P['foam'] - P['foamsh']) * fsh) * fa[..., None]
        o.foam[sl] = fa
        # haze
        hz = (1 - np.exp(-np.maximum(0, z * dn - 25) * pal.get('haze', 1 / 900.) * haze))[..., None]
        col = col + (P['far'] - col) * hz
        # extend colour above crests for a clean antialiased silhouette
        a = m.astype(f32)
        if aa:
            ab = cv2.GaussianBlur(a, (0, 0), .7); k3 = cv2.GaussianBlur(col * a[..., None], (0, 0), 1.0) / np.maximum(cv2.GaussianBlur(a, (0, 0), 1.0), 1e-3)[..., None]
            col = np.where(m[..., None], col, k3); a = np.clip(ab * 1.0, 0, 1); a = np.where(m, np.maximum(a, .0), a)
            fill = cv2.erode(m.astype(np.uint8), np.ones((3, 3), np.uint8)); a = np.where(fill > 0, 1, a); a[-1] = np.where(m[-1], 1, a[-1])
        o.rgb[sl] = col; o.a[sl] = a * m.any(); o.z[sl] = np.where(m, z, 1e9); o.X[sl] = X; o.Y[sl] = Y
        # slice edges (first row of a steep view) must stay opaque
        if y0 == 0: o.a[0] = hit[0]
        return o


def under(base, layer, z, water, soft=.8, op=1.0):
    """composite a cel (premult RGBA + its depth map, 1e9 empty) over `base` that already holds the water: hidden where the water is nearer"""
    vis = (z < water.z + .02).astype(f32); vis = np.where(z > 1e8, cv2.dilate(vis, np.ones((5, 5), np.uint8)), vis)
    vis = cv2.GaussianBlur(vis, (0, 0), soft)[..., None] * op
    return base * (1 - layer[..., 3:4] * vis) + layer[..., :3] * vis


def reflect_z(V):
    V = np.array(V, float); V[..., 2] *= -1; return V


# =================================================================== drawn glitter
def glitter(cam, t, sun, n=700, seed=0, col='#fff6d8', rmin=12., rmax=1800., spread=.10, size=1.0, hz=2.0, z0=0., amt=1.0, frac=.5, asp=.16):
    """lens-shaped flecks along the sun path -> RGB light layer (add or screen it).  spread = half width of the path (rad)"""
    S = np.asarray(sun, float); az0 = math.atan2(S[0], S[1]); r = np.random.default_rng(seed + 9); cel = Cel(ss=2)
    u = r.random(n); dist = rmin * (rmax / rmin) ** u; az = az0 + r.normal(0, 1, n) * spread * (.55 + .75 * (1 - u)); ph = r.random(n); rate = hz * r.uniform(.6, 1.5, n); sz = r.uniform(.5, 1.6, n)
    for i in range(n):
        q = t * rate[i] + ph[i]; k = math.floor(q); q -= k
        rr = np.random.default_rng((seed * 7 + i * 131 + k * 17) & 0x7fffffff); ju, jv, jw = rr.random(3)
        if jw > frac + .35 * (1 - abs(az[i] - az0) / (spread * 3 + 1e-6)): continue
        d = dist[i] * (.93 + .14 * ju); a = az[i] + (jv - .5) * .02; P = np.array([cam.eye[0] + d * math.sin(a), cam.eye[1] + d * math.cos(a), z0])
        p, z = cam.project(P)
        if z < .5 or not (-60 < p[0] < W + 60 and -20 < p[1] < H + 20): continue
        env = math.sin(math.pi * q) ** .6; px = (3.2 + 34 * math.exp(-z / 90.)) * sz[i] * size * env; hy = max(.55, px * asp * (.7 + .6 * ju))
        if px < .8: continue
        m = np.linspace(-1, 1, 7); top = np.c_[p[0] + m * px, p[1] - (1 - m * m) * hy + m * jv * hy * .3]; bot = np.c_[p[0] + m[::-1] * px, p[1] + (1 - m[::-1] ** 2) * hy * .8 + m[::-1] * jv * hy * .3]
        cel.fill(np.vstack([top, bot]), '#ffffff', min(1., .55 + .5 * env))
    return cel.layer()[..., 3:4] * _c(col) * amt


# =================================================================== sheets of water
def sheet(cam, P, seed=0, pal=DAY, flow=0., holes=1., light=(-.5, -.3, .8), ink=True, lw=1.6, su=1., sv=1., frag=1., thin=.0, back=.0, alpha=1.0, z_out=False, root=0.0):
    """ballistic wall of water on a (Ns,Na,3) grid: s along the tear, a 0 root .. 1 tip.  Lace opens toward the tip, fingers fray.
    su,sv = metres of pattern per unit s / a; flow scrolls the pattern toward the tip.  -> RGBA layer"""
    P = np.asarray(P, float); ns, na = P.shape[:2]; V = P.reshape(-1, 3); s, a = np.meshgrid(np.linspace(0, 1, ns), np.linspace(0, 1, na), indexing='ij')
    ds = np.gradient(P, axis=0); da = np.gradient(P, axis=1); N = np.cross(ds, da); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9
    vd = P - cam.eye; N = np.where((N * vd).sum(-1, keepdims=True) > 0, -N, N); L = np.asarray(light, float); L /= np.linalg.norm(L); nl = (N @ L)
    R = Raster(3); R.add(V, grid_faces(ns, na), np.stack([s.ravel(), a.ravel(), nl.ravel()], -1), 0, cam)
    hit = R.i >= 0; lay = np.zeros((H, W, 4), f32)
    if not hit.any(): return (lay, np.full((H, W), 1e9, f32)) if z_out else lay
    S_, A_, NL = [R.a[..., i].astype(f32) for i in range(3)]
    U, Vv = S_ * su, A_ * sv - flow
    l0 = _samp(lace(0), U, Vv * .55, 1.); l1 = _samp(lace(1), U + .3, Vv * .6, .41); pn = _samp(pnoise(), U, Vv * .2, 2.3); pf = _samp(pnoise(), U + .17, 0 * U + seed * .13, 1.1)
    tip = .80 + .16 * pf                                                                     # finger length per s
    q = A_ / np.maximum(tip, .2)
    dens = 1.25 - np.clip(q, 0, 1.4) ** (1.6 / max(frag, .2)) * 1.25 * holes
    edge = smooth(0, .05, S_) * smooth(0, .05, 1 - S_)
    la = l0 * .62 + l1 * .38 + .10 * pn
    fv = ((dens * edge) * 1.25 - (1 - la) * .95 + root * (1 - A_)).astype(f32)
    g = cv2.GaussianBlur(np.abs(cv2.Sobel(fv, cv2.CV_32F, 1, 0, scale=.125)) + np.abs(cv2.Sobel(fv, cv2.CV_32F, 0, 1, scale=.125)), (0, 0), 1.0) + .02
    al = np.clip(fv / (g * 2) + .5, 0, 1) * hit
    al = cv2.GaussianBlur(al, (0, 0), .7)
    Pc = {k: _c(v) for k, v in pal.items() if k != 'haze'}
    body = Pc['mid'] + (Pc['crest'] - Pc['mid']) * .55
    v = NL * .9 + .22 * pn + .25 * (la - .5)
    e = lambda x, t, k=.05: np.clip((x - t) / k + .5, 0, 1)[..., None]
    col = Pc['body'] + (body - Pc['body']) * e(v, .02)
    col = col + (Pc['light'] - col) * e(v, .42)
    wht = np.maximum(smooth(.42, .72, q + .22 * (la - .5)), np.clip((.30 - fv) / .16, 0, 1) * smooth(.05, .3, q))[..., None]
    fcol = Pc['foamsh'] + (Pc['foam'] - Pc['foamsh']) * e(v + .25 * (l1 - .5), .10)
    col = col * (1 - wht) + fcol * wht
    strk = (smooth(.60, .78, _samp(lace(1), U * 2.2, Vv * .16, .8)) * .55 * smooth(0., .1, A_))[..., None]
    col = col + (Pc['foam'] - col) * strk * (1 - thin)
    aa = al * (alpha - thin * .5 * (1 - wht[..., 0]))
    lay[..., :3] = col * aa[..., None]; lay[..., 3] = aa
    if ink:
        cel = Cel(); m8 = (al > .5).astype(np.uint8); cs, _ = cv2.findContours(m8, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        kk = np.hanning(7); kk /= kk.sum(); n_ = 0
        for c in sorted(cs, key=len, reverse=True)[:150]:
            c = c[:, 0].astype(float)
            if len(c) < 46: break
            cc = np.vstack([c[-6:], c, c[:6]]); cc = np.stack([np.convolve(cc[:, i], kk, 'same') for i in (0, 1)], -1)[6:-6]
            r = np.random.default_rng(seed * 13 + n_); n_ += 1; i = int(r.integers(0, 30))
            while i < len(cc) - 12:
                ln = int(r.uniform(26, 150)); seg = cc[i:i + ln]
                if len(seg) > 10 and not ((seg[:, 0] < 3) | (seg[:, 0] > W - 4) | (seg[:, 1] < 3) | (seg[:, 1] > H - 4)).all(): cel.line(seg, lw * r.uniform(.7, 1.3), pal['ink'], taper=(.3, .4), press=.4, seed=i, alpha=.75)
                i += ln + int(r.uniform(8, 70))
        L_ = cel.layer(); lay = lay * (1 - L_[..., 3:4]) + L_
    if z_out: return lay, np.where(hit & (al > .3), 1 / np.maximum(R.w, 1e-9), 1e9).astype(f32)
    return lay


def drops(cam, P, V=None, r=.06, pal=DAY, shutter=1 / 48., lw=1.2, seed=0, zmask=None, rmin=1.0, minpx=.7, maxpx=40):
    """drawn drops at world points P (n,3) with velocities V (streak over the shutter).  r scalar/array metres -> RGBA"""
    P = np.asarray(P, float); n = len(P); r = np.broadcast_to(np.asarray(r, float), (n,)); p, z = cam.project(P); pb = p if V is None else cam.project(P - np.asarray(V) * shutter)[0]
    cel = Cel(); o = np.argsort(-z); a = np.linspace(0, 2 * math.pi, 13)[:-1]
    for i in o:
        if z[i] < .3: continue
        x, y = p[i]
        if not (-60 < x < W + 60 and -60 < y < H + 60): continue
        if zmask is not None and 0 <= x < W and 0 <= y < H and zmask[int(y), int(x)] < z[i]: continue
        R = min(maxpx, r[i] * cam.f / z[i])
        if R < minpx: continue
        d = p[i] - pb[i]; L = min(R * 5, np.hypot(*d)); th = math.atan2(d[1], d[0]); c, s = math.cos(th), math.sin(th)
        lx = np.cos(a) * (R + L * .5 * (np.cos(a) < 0)); ly = np.sin(a) * R * (1 - .35 * (np.cos(a) < 0) * min(1, L / (R * 2)))
        Q = np.c_[x + lx * c - ly * s, y + lx * s + ly * c]
        if R > 5: cel.fill(np.c_[x + lx * c * 1.0 - ly * s, y + lx * s + ly * c] + (Q - [x, y]) * (1.2 / R), pal['ink'], .8)
        cel.fill(Q, pal['foam'] if R < 4 else pal['light'])
        if R >= 4: cel.fill([x - R * .12, y - R * .16] + (Q - [x, y]) * .62, pal['sky']); cel.fill(np.array([x - R * .3, y - R * .32]) + (Q - [x, y]) * .22, '#ffffff')
    return cel.layer()


def splash_ring(cam, x, y, t, r0=2., v=9., vz=8., n=40, seed=0, g=9.8, drop=.15):
    """crown splash: sheet grid (Ns,Na,3) for sheet(), plus drop positions/velocities.  t seconds since impact"""
    s = np.linspace(0, 2 * math.pi, n); rr = np.random.default_rng(seed); k = 1 + .18 * np.interp(np.linspace(0, 12, n), np.arange(13), np.r_[rr.normal(0, 1, 12), 0])
    a = np.linspace(0, 1, 16); P = np.zeros((n, 16, 3))
    for j, q in enumerate(a):
        tt = t * q; rad = r0 + v * k * tt; P[:, j] = np.stack([x + np.cos(s) * rad, y + np.sin(s) * rad, np.maximum(vz * k * tt - g * tt * tt / 2, -.2) * (1 if t else 0)], -1)
    m = 90; th = rr.uniform(0, 2 * math.pi, m); sp = rr.uniform(.7, 1.5, m); vv = np.stack([np.cos(th) * v * sp, np.sin(th) * v * sp, vz * rr.uniform(.8, 1.6, m)], -1)
    D = np.array([x, y, 0]) + np.stack([np.cos(th), np.sin(th), 0 * th], -1) * r0 + vv * t; D[:, 2] -= g * t * t / 2
    vv = vv.copy(); vv[:, 2] -= g * t; ok = D[:, 2] > 0
    return P, D[ok], vv[ok], (drop * rr.uniform(.4, 1.4, m))[ok]
