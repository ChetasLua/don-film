"""fleet3: 3-D marine warships, masts, planks and the lion's head, rasterised (op.sea.Raster) as flat cels and re-inked by hand (region contours, tapered pen).
    G = warship(seed=1)                        local frame: x bow, y port, z up, metres, waterline z = 0.   G.at(R, t) -> moved copy
    r = draw(cam, [G1, G2, ...], light, fog=('#bfe3ee', 60, 1/420.)) -> R: r.rgba premult layer, r.z depth map (1e9 empty)
    img = sea.under(water.over(sky), r.rgba, r.z, water)
planks(cam-free): Debris(...) closed-form rigid bodies -> Geo per frame.   splitmast(...) lengthwise split mast half.   lion() figurehead."""
import numpy as np, cv2, math
from numba import njit, prange
from scipy import ndimage
from . import W, H
from .paint import rgb, f32, fbm
from .ink import Cel, spl
from .kit import outline
from .sea import Raster, grid_faces

# name: base, shade, hi, line, pat bits (1 plank rows, 2 butt joints, 4 vertical seams, 8 emblem, 16 hull bands), pitch u, pitch v, half line m, thr, extra
MATS = dict(
    hull=('#41587a', '#283a58', '#5d789c', '#182438', 1 | 2 | 16, 4.5, .46, .022, .10, ('#6d2a2c', '#f1efe8', '#4a6288')), deck=('#d3a468', '#94663f', '#e8c58c', '#6d4527', 1 | 2, 3.6, .34, .018, .05, None),
    rail=('#e9dcc0', '#a89678', '#fff4da', '#6a5a44', 0, 1, 1, 0, .10, None), wall=('#eef0ee', '#a9b4c8', '#ffffff', '#6d7890', 1, 1, .62, .018, .12, None), roof=('#3f68a8', '#26447a', '#6b92cc', '#1b2f58', 1, 1, .5, .02, .1, None),
    mast=('#b5824c', '#744a2c', '#d8ab70', '#4d2f1c', 1, 1, .5, .015, .12, None), raw=('#f0cf94', '#c0925a', '#fbe6b8', '#96673a', 1, 1, .24, .014, -.2, None), yard=('#8d623a', '#573a24', '#b58652', '#33210f', 0, 1, 1, 0, .1, None),
    sail=('#f7f3e6', '#bfc3d2', '#fffdf4', '#9aa2b8', 4, 1.9, 1, .020, .30, None), gull=('#f7f3e6', '#bfc3d2', '#fffdf4', '#9aa2b8', 4 | 8, 1.9, 1, .020, .30, ('#2c5aa8', '#2c5aa8', '#2c5aa8')),
    iron=('#4c5362', '#282c38', '#7f8aa0', '#171a22', 0, 1, 1, 0, .25, None), glass=('#2b3d58', '#1b2840', '#6f93bd', '#101828', 0, 1, 1, 0, .3, None), plank=('#c99a5e', '#86572f', '#e8c68c', '#5a3820', 1, 1, .2, .012, .0, None),
    dark=('#3a2a24', '#231916', '#5a4436', '#120c0a', 1, 1, .5, .02, .4, None), lion=('#f4b93c', '#c47c1e', '#ffe08a', '#8a4f14', 0, 1, 1, 0, .16, None), mane=('#f0872a', '#b4501c', '#ffb458', '#7a3210', 0, 1, 1, 0, .16, None),
    cream=('#fbf0d8', '#c9b694', '#ffffff', '#8a7858', 0, 1, 1, 0, .1, None), lawn=('#6fb548', '#3f7f34', '#a6dc6e', '#2c5a26', 0, 1, 1, 0, .0, None), red=('#c5303a', '#7d1a2c', '#e8605c', '#4d0d1a', 0, 1, 1, 0, .1, None),
    skback=('#565a9a', '#2b2d5e', '#8f93cc', '#191a3c', 32 | 64, 1, .62, .02, .10, None), skbelly=('#f4dfa6', '#bf9058', '#fff3cc', '#7d5730', 1 | 64, 1, 1.05, .03, .12, None), skmouth=('#b23350', '#6c1636', '#dd5f72', '#420b22', 0, 1, 1, 0, .0, None), skfin=('#d9503c', '#8e2436', '#f58a5c', '#561222', 4, .9, 1, .03, .05, None), tongue=('#e0677c', '#a83450', '#f7a0a4', '#6c1a34', 0, 1, 1, 0, .1, None), skeye=('#ffe274', '#d89a2c', '#fff8c8', '#8a5a14', 0, 1, 1, 0, .05, None), tooth=('#fbf6e2', '#c4b394', '#ffffff', '#7d6c50', 0, 1, 1, 0, .1, None), pale=('#d8c7e4', '#9d86b8', '#ffffff', '#6a5488', 0, 1, 1, 0, .1, None),
    ink=('#1a1820', '#121017', '#33303e', '#08070a', 0, 1, 1, 0, .1, None), white=('#ffffff', '#c8cede', '#ffffff', '#8890a6', 0, 1, 1, 0, .1, None), char=('#2c2422', '#1a1514', '#4a3a32', '#0a0808', 1, 1, .24, .014, .2, None))
NAMES = list(MATS); MID = {k: i for i, k in enumerate(NAMES)}
_T = [np.stack([rgb(MATS[k][j]) for k in NAMES]).astype(f32) for j in range(4)]
_PAT = np.array([[MATS[k][4], MATS[k][5], MATS[k][6], MATS[k][7], MATS[k][8]] for k in NAMES], f32)
_EX = np.stack([np.stack([rgb(q) for q in (MATS[k][9] or (MATS[k][0],) * 3)]) for k in NAMES]).astype(f32)
_noise = None


def _tex():
    """gull emblem 256 x 256, paint noise 256 x 256 (wrapping)"""
    global _noise
    if _noise is None:
        c = Cel(256, 256, ss=2); L = spl([(6, 92), (40, 62), (86, 74), (118, 118), (128, 138), (138, 118), (170, 74), (216, 62), (250, 92), (214, 88), (180, 104), (152, 142), (140, 176), (128, 190), (116, 176), (104, 142), (76, 104), (42, 88)], True, n=8); c.fill(L, '#ffffff')
        c.fill(spl([(112, 176), (104, 214), (128, 200), (152, 214), (144, 176)], True, n=6), '#ffffff'); c.fill(spl([(120, 120), (128, 96), (140, 106), (136, 128)], True, n=6), '#ffffff')
        n = fbm(256 + 64, 256 + 64, (40, 16, 6), None, 5); k = np.linspace(0, 1, 64)[:, None]; n[:64] = n[:64] * k + n[256:320] * (1 - k); n[:, :64] = n[:, :64] * k.T + n[:, 256:320] * (1 - k.T)
        _noise = (c.rgba()[..., 3].astype(f32), n[:256, :256].astype(f32))
    return _noise


@njit(parallel=True, fastmath=True, cache=True)
def _shade(ib, ab, wb, fm, TB, TS, TH, TL, PT, EX, em, nz, f, fr, fg, fb, f0, fd, warm, out):
    h, w = ib.shape
    for y in prange(h):
        for x in range(w):
            i = ib[y, x]
            if i < 0: continue
            m = fm[i]; u = ab[y, x, 0]; v = ab[y, x, 1]; nl = ab[y, x, 2]; z = 1. / wb[y, x]; px = f / z
            n = nz[int(math.floor(v * 41.)) & 255, int(math.floor(u * 9.)) & 255] * .55 + nz[int(math.floor(v * 13.)) & 255, int(math.floor(u * 4.)) & 255] * .45
            t = nl + .028 * n; pat = int(PT[m, 0]); th = PT[m, 4]; r = 0.; g = 0.; b = 0.
            e0 = min(1., max(0., (t - th) * 60. + .5)); e1 = min(1., max(0., (t - (th + .62)) * 50. + .5)) * .8
            B0 = TB[m, 0]; B1 = TB[m, 1]; B2 = TB[m, 2]; S0 = TS[m, 0]; S1 = TS[m, 1]; S2 = TS[m, 2]; ln = 0.
            if pat & 16:
                k = -1
                if v < .55: k = 0
                elif v > 2.7 and v < 3.5: k = 1
                elif v > 4.1: k = 2
                if k >= 0:
                    B0 = EX[m, k, 0]; B1 = EX[m, k, 1]; B2 = EX[m, k, 2]; S0 = B0 * .60; S1 = B1 * .64; S2 = B2 * .76
            r = S0 + (B0 - S0) * e0; g = S1 + (B1 - S1) * e0; b = S2 + (B2 - S2) * e0
            r += (TH[m, 0] - r) * e1 * (0. if pat & 16 else 1.); g += (TH[m, 1] - g) * e1 * (0. if pat & 16 else 1.); b += (TH[m, 2] - b) * e1 * (0. if pat & 16 else 1.)
            if pat & 8:
                eu = (u / 4.6 + 1.) * 128.; ev = (1. - v / 4.6) * 128.
                if eu > 1 and ev > 1 and eu < 254 and ev < 254:
                    if em[int(ev), int(eu)] > .5:
                        k = .66 + .34 * e0; r = EX[m, 0, 0] * k; g = EX[m, 0, 1] * k; b = EX[m, 0, 2] * k
            if pat & 32:
                q = v / PT[m, 2]; row = math.floor(q); cu = u / PT[m, 2] * .8 + .5 * (int(row) & 1); fx = cu - math.floor(cu) - .5; fy = q - row; d = abs(math.sqrt(fx * fx * 1.1 + fy * fy) - .62) * PT[m, 2]; hw0 = max(.022, .6 / px)
                a_ = min(1., max(0., (hw0 - d) * px + .5)) * min(1., max(0., (PT[m, 2] * px - 7.) / 12.)) * min(1., max(0., (nz[int(math.floor(v * 3.)) & 255, int(math.floor(u * 3.)) & 255] + .25) * 2.5)) * .7
                r += (TL[m, 0] - r) * a_; g += (TL[m, 1] - g) * a_; b += (TL[m, 2] - b) * a_
            if pat & 64:
                if t > th - .015 and t < th + .13:
                    q = (u * .55 + v * 1.0) * 2.6 + n * .25; d = q - math.floor(q); ww = .30 * (1. - (t - th) / .13)
                    if d < ww and 2.0 * px > 9.:
                        a_ = min(1., (2.0 * px - 9.) / 14.) * .85; r += (TL[m, 0] - r) * a_; g += (TL[m, 1] - g) * a_; b += (TL[m, 2] - b) * a_
            k = 1. + .045 * n; r *= k; g *= k; b *= k
            hw = max(PT[m, 3], .55 / px); pv = PT[m, 2]; pu = PT[m, 1]
            if pat & 1:
                q = v / pv; d = abs(q - math.floor(q + .5)) * pv; ln = min(1., max(0., (hw - d) * px + .5)) * min(1., max(0., (pv * px - 5.) / 9.))
                if pat & 2:
                    row = math.floor(q); q2 = (u + row * 1.37 * pu * .37) / pu; d = abs(q2 - math.floor(q2 + .5)) * pu; ln = max(ln, min(1., max(0., (hw - d) * px + .5)) * min(1., max(0., (pv * px - 5.) / 9.)))
            if pat & 4:
                q = u / pu; d = abs(q - math.floor(q + .5)) * pu; ln = max(ln, .6 * min(1., max(0., (hw - d) * px + .5)) * min(1., max(0., (pu * px - 5.) / 12.)))
            ln *= .72; r += (TL[m, 0] - r) * ln; g += (TL[m, 1] - g) * ln; b += (TL[m, 2] - b) * ln
            k = 1. - math.exp(-max(0., z - f0) * fd); r += (fr - r) * k; g += (fg - g) * k; b += (fb - b) * k
            out[y, x, 0] = r; out[y, x, 1] = g; out[y, x, 2] = b; out[y, x, 3] = 1.


class Geo:
    def __init__(self): self.V, self.N, self.U, self.F, self.M, self.Pt, self.two, self.ropes, self.n, self.np = [], [], [], [], [], [], [], [], 0, 0

    def part(self): self.np += 1; return self.np

    def mesh(self, V, F, N, U, mat, part=None, two=False):
        V = np.asarray(V, float).reshape(-1, 3); self.V.append(V); self.N.append(np.asarray(N, float).reshape(-1, 3)); self.U.append(np.asarray(U, float).reshape(-1, 2)); F = np.asarray(F, int); self.F.append(F + self.n); self.n += len(V)
        self.M.append(np.full(len(F), MID[mat])); self.Pt.append(np.full(len(F), self.part() if part is None else part)); self.two.append(np.full(len(V), two)); return self

    def grid(self, P, mat, part=None, uv=None, two=False, flip=False, wrap=False):
        P = np.asarray(P, float); n, m = P.shape[:2]; du = np.gradient(P, axis=0); dv = np.gradient(P, axis=1) if not wrap else np.roll(P, -1, 1) - np.roll(P, 1, 1); N = np.cross(du, dv); N /= np.linalg.norm(N, axis=-1, keepdims=True) + 1e-9; N = -N if flip else N
        if uv is None: s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P.mean(1), axis=0), axis=1))]; t = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P.mean(0), axis=0), axis=1))]; uv = np.stack(np.meshgrid(s, t, indexing='ij'), -1)
        return self.mesh(P, grid_faces(n, m, wrap=wrap), N, uv, mat, part, two)

    def quad(self, a, b, c, d, mat, part=None, uv=None, two=True):
        a, b, c, d = [np.asarray(q, float) for q in (a, b, c, d)]; n = np.cross(b - a, d - a); n /= np.linalg.norm(n) + 1e-9; lu, lv = np.linalg.norm(b - a), np.linalg.norm(d - a)
        uv = [(0, 0), (lu, 0), (lu, lv), (0, lv)] if uv is None else uv
        return self.mesh([a, b, c, d], [(0, 1, 2), (0, 2, 3)], [n] * 4, uv, mat, part, two)

    def box(self, c, half, mat, R=None, part=None, top=None):
        c = np.asarray(c, float); R = np.eye(3) if R is None else R; h = np.asarray(half, float)
        for ax in range(3):
            for s in (-1, 1):
                i, j = [(1, 2), (2, 0), (0, 1)][ax]; e = np.zeros((4, 3)); e[:, ax] = s * h[ax]; e[:, i] = np.array([-1, 1, 1, -1]) * h[i]; e[:, j] = np.array([-1, -1, 1, 1]) * h[j]
                if s < 0: e = e[::-1]
                k = (j, i) if h[i] < h[j] else (i, j); U = np.stack([e[:, k[0]], e[:, k[1]]], -1); n = np.zeros(3); n[ax] = s
                self.mesh(c + e @ R.T, [(0, 1, 2), (0, 2, 3)], [R @ n] * 4, U, top if top and ax == 2 and s > 0 else mat, part)
        return self

    def cyl(self, p0, p1, r0, r1, mat, n=12, part=None, caps=True, m=2):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.linalg.norm(d); d /= L; a = np.cross(d, [0, 0, 1.] if abs(d[2]) < .9 else [1., 0, 0]); a /= np.linalg.norm(a); b = np.cross(d, a); th = np.linspace(0, 2 * math.pi, n + 1); C = np.cos(th)[:, None] * a + np.sin(th)[:, None] * b; pt = self.part() if part is None else part
        T = np.linspace(0, 1, m); P = np.stack([p0 + d * L * t + C * (r0 + (r1 - r0) * t) for t in T]); U = np.stack(np.meshgrid(T * L, th * max(r0, r1), indexing='ij'), -1); self.mesh(P, grid_faces(m, n + 1), np.stack([C] * m), U, mat, pt)
        if caps:
            for p, r, s in ((p0, r0, -1), (p1, r1, 1)):
                if r > .05: self.mesh(np.vstack([p, p + C * r]), [(0, k, k + 1) for k in range(1, n + 1)], [d * s] * (n + 2), np.vstack([[0, 0], np.c_[np.cos(th), np.sin(th)] * r]), mat, self.part())
        return self

    def rope(self, P, r=.05, col='#3a2c22'): self.ropes.append((np.asarray(P, float), r, col)); return self

    def add(self, o):
        for k in ('V', 'N', 'U', 'M', 'two'): getattr(self, k).extend(getattr(o, k))
        self.F += [f + self.n for f in o.F]; self.Pt += [p + self.np for p in o.Pt]; self.ropes += o.ropes; self.n += o.n; self.np += o.np; return self

    def at(self, R=None, t=(0, 0, 0)):
        g = Geo(); R = np.eye(3) if R is None else np.asarray(R, float); t = np.asarray(t, float); g.V = [np.vstack(self.V) @ R.T + t] if self.V else []; g.N = [np.vstack(self.N) @ R.T] if self.V else []; g.U, g.F, g.M, g.Pt, g.two = self.U, self.F, self.M, self.Pt, self.two; g.ropes = [(P @ R.T + t, r, c) for P, r, c in self.ropes]; g.n, g.np = self.n, self.np
        return g

    def pack(self):
        if not hasattr(self, '_pk'): self._pk = (np.vstack(self.U), np.vstack(self.F), np.concatenate(self.M), np.concatenate(self.Pt), np.concatenate(self.two))
        return (np.vstack(self.V), np.vstack(self.N)) + self._pk


class Res: pass


def draw(cam, geos, light=(.6, -.3, .7), fog=('#bfe3ee', 80., 1 / 500.), ss=2, ink=True, lw=.030, lwmax=3.4, inkcol='#17131a', glow=None, minarea=40):
    """-> Res: rgba premult, z (1x, 1e9 empty).  glow = [(world point, radius m, hex, amt)] local lights added to the cels"""
    R = Raster(3, 0, 0, W, H, ss); L = np.asarray(light, float); L /= np.linalg.norm(L); fm, fp = [], []; nf = 0; npt = 0; ropes = []
    for g in geos:
        ropes += g.ropes
        if not g.V: continue
        V, N, U, F, M, Pt, two = g.pack(); c = cam.cam(V);
        if c[:, 2].max() < .1: npt += g.np; continue
        e = cam.eye - V; s = np.where(two & ((N * e).sum(1) < 0), -1., 1.); nl = (N @ L) * s; nl = np.where(two & (s < 0), np.maximum(nl, .25 * np.abs(N @ L)), nl)
        R.add(V, F, np.c_[U, nl], np.arange(nf, nf + len(F)), cam); fm.append(M); fp.append(Pt + npt); nf += len(F); npt += g.np
    o = Res(); o.rgba = np.zeros((H, W, 4), f32); o.z = np.full((H, W), 1e9, f32)
    cel = Cel(ss=2); fg = rgb(fog[0]); zt = None
    if nf:
        fm = np.concatenate(fm); fp = np.concatenate(fp); em, nz = _tex(); big = np.zeros((H * ss, W * ss, 4), f32)
        _shade(R.i, R.a, R.w, fm.astype(np.int64), *_T, _PAT, _EX, em, nz, float(cam.f * ss), float(fg[0]), float(fg[1]), float(fg[2]), float(fog[1]), float(fog[2]), 0., big)
        if glow:
            hit = R.i >= 0; ys, xs = np.nonzero(hit)
            for p, rad, col, amt in glow:
                q, z = cam.project(np.asarray(p, float)); zz = 1 / R.w[ys, xs]; s_ = cam.f * ss / zz; d2 = ((xs - q[0] * ss) / s_) ** 2 + ((ys - q[1] * ss) / s_) ** 2 + ((zz - z) * .8) ** 2; k = (amt * np.exp(-d2 / rad ** 2)).astype(f32)[:, None]; big[ys, xs, :3] = 1 - (1 - np.clip(big[ys, xs, :3], 0, 1)) * (1 - np.clip(k * rgb(col), 0, 1))
        o.rgba = cv2.resize(big, (W, H), interpolation=cv2.INTER_AREA); wm = R.w.reshape(H, ss, W, ss).max((1, 3)); o.z = np.where(wm > 0, 1 / np.maximum(wm, 1e-9), 1e9).astype(f32); zt = wm
        if ink:
            pid = np.where(R.i >= 0, fp[np.maximum(R.i, 0)], 0).astype(np.int32); sl = ndimage.find_objects(pid)
            items = []
            for k, s in enumerate(sl):
                if s is None: continue
                m = pid[s] == k + 1; a = int(m.sum())
                if a < minarea: continue
                items.append((float(R.w[s][m].mean()), k + 1, s, m, a))
            items.sort(key=lambda q: q[0])
            for wz, k, s, m, a in items:
                q = cam.f * wz; lwk = float(np.clip(lw * q + .45, .6, lwmax)) * min(1., (a / 900.) ** .25 + .15); fk = 1 - math.exp(-max(0, 1 / wz - fog[1]) * fog[2]); col = rgb(inkcol) * (1 - fk * .85) + fg * fk * .85 * .8
                cs, _ = cv2.findContours(np.pad(m.astype(np.uint8), 1), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
                for c in cs:
                    if len(c) < 16: continue
                    c = c[:, 0].astype(float) - 1; kr = np.array([1, 2, 3, 2, 1.]) / 9; cc = np.vstack([c[-2:], c, c[:2]]); c = np.stack([np.convolve(cc[:, i], kr, 'valid') for i in (0, 1)], -1)
                    c = (c + [s[1].start, s[0].start] + .5) / ss; on = (c[:, 0] < 2) | (c[:, 0] > W - 3) | (c[:, 1] < 2) | (c[:, 1] > H - 3)
                    if on.all(): continue
                    if on.any():
                        i0 = int(np.argmax(on)); c = np.roll(c, -i0, 0); on = np.roll(on, -i0); seg = []; st = None
                        for j, b in enumerate(on):
                            if not b and st is None: st = j
                            if b and st is not None: seg.append(c[st:j]); st = None
                        if st is not None: seg.append(c[st:])
                        for g_ in seg:
                            if len(g_) > 6: outline(cel, g_, lwk, col, k % 997, closed=False, press=.45, tremor=.25)
                    else: outline(cel, c, lwk, col, k % 997, press=.45, tremor=.25, breaks=int(len(c) > 260))
    for P, r, col in ropes:
        n = max(8, int(len(P) * 3)); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
        if s[-1] < 1e-3: continue
        t = np.linspace(0, s[-1], max(n, 24)); Q = np.stack([np.interp(t, s, P[:, i]) for i in range(3)], -1); p, z = cam.project(Q); ok = z > .2
        ok &= (p[:, 0] > -50) & (p[:, 0] < W + 50) & (p[:, 1] > -50) & (p[:, 1] < H + 50)
        if zt is not None:
            xi, yi = np.clip(p[:, 0], 0, W - 1).astype(int), np.clip(p[:, 1], 0, H - 1).astype(int); ok &= ~(zt[yi, xi] * z > 1.015)
        i = 0
        while i < len(ok):
            if ok[i]:
                j = i
                while j + 1 < len(ok) and ok[j + 1]: j += 1
                if j - i >= 2:
                    zz = float(z[i:j + 1].mean()); wd = max(.55, 2 * r * cam.f / zz); fk = 1 - math.exp(-max(0, zz - fog[1]) * fog[2])
                    if fk < .85: cel.line(p[i:j + 1], min(wd, 9), rgb(col) * (1 - fk) + fg * fk, taper=(.02, .02), press=.2, seed=i, alpha=min(1., .4 + wd * .5))
                i = j + 1
            else: i += 1
    I = cel.rgba(); z0 = cv2.erode(o.z, np.ones((5, 5), np.uint8)); o.z = np.where((I[..., 3] > .05) & (o.z > 1e8), z0, o.z)
    o.rgba = o.rgba * (1 - I[..., 3:4]) + I; o.ink = I
    return o


# ------------------------------------------------------------------ ship parts
def _hullP(L, B, s0=0., s1=1., n=26, m=9, side=1, zr=None):
    s = np.linspace(s0, s1, n); q = np.linspace(0, 1, m); P = np.zeros((n, m, 3)); D = np.zeros((n, 3))
    for i, t in enumerate(s):
        b = B / 2 * (1 - max(0, (t - .60) / .40) ** 2.3) * (1 - .22 * max(0, (.16 - t) / .16) ** 2); rail = 4.7 + 1.9 * (abs(t - .42) / .58) ** 2.2; keel = -2.4 + 2.6 * max(0, (t - .86) / .14) ** 2
        z = keel + (rail - keel) * q; x = -L / 2 + L * t + max(0, t - .7) * 9.0 * q ** 1.5 - max(0, .10 - t) * 10 * q; y = b * np.maximum(q, 0) ** .42 * (1 + .10 * max(0, t - .7) * q * 3) * (1 - .05 * q * (t < .7)); P[i] = np.stack([np.full(m, 1.) * x, side * np.maximum(y, .02), z], -1)
    return P


def sail(g, yc, x, z0, z1, w0, w1, belly, mat='sail', seed=0, n=12, ragged=None):
    """billowed square sail between two yards (along y), centre x; belly along +x"""
    r = np.random.default_rng(seed); u = np.linspace(-1, 1, n); v = np.linspace(0, 1, n); P = np.zeros((n, n, 3)); U = np.zeros((n, n, 2))
    for j, b in enumerate(v):
        z = z1 + (z0 - z1) * b; wd = (w1 + (w0 - w1) * b) / 2 * (1 - .10 * math.sin(math.pi * b)); bl = belly * math.sin(math.pi * min(1, b * .92 + .04)) ** .8
        for i, a in enumerate(u):
            zz = z + (.55 * a * a - .1) * (b > .5) * (b - .5) * 2; P[j, i] = (x + bl * (1 - .62 * a * a) + .12 * math.sin(a * 5 + b * 3 + seed), yc + a * wd, zz); U[j, i] = (a * wd, (b - .5) * (z1 - z0) * -1)
    g.grid(P, mat, uv=U, two=True); return P


def mast(g, x, h, base=3.6, yards=((.50, 17, .36), (.92, 12.5, .26)), r=.62, sails=True, belly=2.6, emblem=True, crow=True, seed=0, beam=9.):
    g.cyl((x, 0, base), (x, 0, h), r, r * .5, 'mast', 12); zs = []
    for k, (t, w, yr) in enumerate(yards):
        z = base + (h - base) * t; g.cyl((x + r * 1.2, -w / 2, z), (x + r * 1.2, w / 2, z), yr, yr, 'yard', 8); zs.append((z, w))
        g.rope([(x, 0, z + 3.2), (x + r, w / 2, z)], .05); g.rope([(x, 0, z + 3.2), (x + r, -w / 2, z)], .05)
    if sails:
        lo = base + 3.0; tops = [z for z, _ in zs]
        sail(g, 0, x + r * 1.4, lo, tops[0] - .4, zs[0][1] * 1.02, zs[0][1] * .94, belly, 'gull' if emblem else 'sail', seed)
        if len(zs) > 1: sail(g, 0, x + r * 1.4, tops[0] + .5, tops[1] - .4, zs[0][1] * .92, zs[1][1] * .94, belly * .8, 'sail', seed + 1)
    if crow: g.cyl((x, 0, h * .80), (x, 0, h * .80 + 1.3), 1.0, 1.45, 'rail', 12, caps=False); g.cyl((x, 0, h * .80), (x, 0, h * .80 + .1), 1.0, 1.0, 'deck', 12)
    for sg in (-1, 1):
        top = np.array([x, 0, base + (h - base) * .78])
        ft = [np.array([x - 2.4 + 1.3 * k, sg * beam * .48, 4.6]) for k in range(4)]
        for q in ft: g.rope([q, top], .045)
        for k in range(1, 13): t = k / 14.; g.rope([ft[0] * (1 - t) + top * t, ft[-1] * (1 - t) + top * t], .025)
    return g


def warship(seed=0, L=46., B=11., masts=3, lod=0, main=True, s0=0., s1=1., rig=True, jag=0.):
    """lod 0 near, 1 far.  s0..s1 cuts a hull piece (broken ship): jag = raggedness of the break"""
    r = np.random.default_rng(seed); g = Geo(); n = 26 if lod == 0 else 14; whole = s0 <= 0 and s1 >= 1; Ps = []
    for sd in (1, -1):
        P = _hullP(L, B, s0, s1, n, 10, sd)
        if jag:
            if s0 > 0: P[0, :, 0] += r.uniform(-jag, jag, 10)
            if s1 < 1: P[-1, :, 0] += r.uniform(-jag, jag, 10)
        Ps.append(P); U = np.stack([P[..., 0], P[..., 2]], -1); g.grid(P, 'hull', uv=U, flip=sd < 0, two=True)
        rl = P[:, -1]; top = rl + np.array([0, 0, .28]); out = np.array([0, sd * .22, 0]); g.grid(np.stack([rl + out, top + out, top - out, rl - out], 1), 'rail', two=True)
        g.grid(np.stack([rl - out * .8 - [0, 0, 1.15], rl - out * .8], 1), 'wall', two=True)
    q = np.linspace(0, 1, 7); A, Bb = Ps[0][:, -1] - [0, 0, 1.1], Ps[1][:, -1] - [0, 0, 1.1]; D = A[:, None] * (1 - q)[None, :, None] + Bb[:, None] * q[None, :, None]; D[..., 2] += (.25 * np.sin(math.pi * q))[None]; g.grid(D, 'deck', uv=D[..., :2], two=True)
    if s0 > 0 or s1 < 1:                                                   # broken end: dark section with ribs
        i = 0 if s0 > 0 else -1; S = np.stack([Ps[0][i], Ps[1][i]], 0); g.grid(S, 'dark', two=True)
        for k in (2, 4, 6, 8): a, b = Ps[0][i][k], Ps[1][i][k]; g.cyl(a, b * .5 + a * .5 + r.normal(0, .25, 3), .16, .12, 'raw', 6); g.cyl(b, a * .62 + b * .38 + r.normal(0, .25, 3), .16, .12, 'raw', 6)
        for k in range(16): a = Ps[k % 2][i][1 + k // 2]; d = np.array([1. if s1 < 1 else -1., r.normal(0, .3), r.normal(0, .4)]); g.cyl(a, a + d * r.uniform(.6, 2.2), .15, .03, 'raw', 5, caps=False)
    if s0 < .02:
        x0 = -L / 2 + .6; zd = 5.3
        g.box((x0 + 4.6, 0, zd + 1.3), (4.6, B * .36, 1.7), 'wall', top='deck'); g.box((x0 + 3.2, 0, zd + 4.2), (3.0, B * .27, 1.3), 'wall', top='roof'); g.box((x0 + 4.6, 0, zd + 3.12), (4.9, B * .39, .14), 'rail'); g.box((x0 + 3.2, 0, zd + 5.6), (3.3, B * .31, .16), 'roof')
        if lod == 0:
            for sg in (-1, 1):
                for k in range(4): g.box((x0 + 1.6 + 2.1 * k, sg * B * .363, zd + 1.5), (.55, .05, .55), 'glass')
            for k in range(3): g.box((x0 + 9.24, -2 + 2 * k, zd + 1.3), (.05, .5, .7), 'glass')
    if s1 > .98:
        x1 = L / 2 - 3.4; g.box((x1 - 2.0, 0, 5.9), (2.6, B * .23, .5), 'wall', top='deck'); g.cyl((L / 2 - .5, 0, 6.6), (L / 2 + 7.6, 0, 9.4), .42, .22, 'mast', 8); g.cyl((L / 2 + .6, 0, 4.9), (L / 2 + 3.2, 0, 5.5), 1.15, 1.35, 'iron', 14)
    for tx, tz in ((L * .31, 5.6), (-L * .17, 5.0)):
        if not (s0 < tx / L + .5 < s1): continue
        g.cyl((tx, 0, tz - .6), (tx, 0, tz + .9), 1.7, 1.5, 'iron', 14)
        for sg in (-.55, .55): g.cyl((tx + .8, sg, tz + .4), (tx + 4.4, sg, tz + .8), .34, .28, 'iron', 8)
    if lod == 0:
        for sd in (1, -1):
            for k in range(6):
                t = .20 + .085 * k
                if not (s0 + .02 < t < s1 - .02): continue
                i = int((t - s0) / (s1 - s0) * (n - 1)); p = Ps[0 if sd > 0 else 1][i]; y = p[6] * .5 + p[7] * .5; g.box(y + [0, sd * .02, 0], (.5, .08, .42), 'ink'); g.cyl(y - [0, sd * .4, 0], y + [0, sd * 1.5, .05], .26, .22, 'iron', 8)
    for k, (mx, mh) in enumerate(((0., 34.), (L * .27, 29.), (-L * .24, 24.))[:masts]):
        if (k == 0 and not main) or not (s0 < mx / L + .5 < s1): continue
        mast(g, mx, mh, seed=seed * 7 + k, beam=B, emblem=True, crow=k < 2, yards=((.50, 17 - 2.4 * k, .36), (.92, 12.5 - 2 * k, .26)), r=.62 - .08 * k) if rig else None
        if lod: g.ropes = g.ropes[:0]
    return g


def splitmast(h0, h1, r, side, split, bend, part_char=.0, n=26):
    """half of a mast split lengthwise (plane x = 0): local z along the mast, the half lies on y*side > 0.  split = share of the length (from the top) already open, bend = peel (m) at the tip"""
    g = Geo(); z = np.linspace(h0, h1, n); u = (z - h0) / (h1 - h0); op = np.clip((u - (1 - split)) / max(split, 1e-3), 0, 1) ** 1.6 * bend if split > 0 else 0 * u; a = np.linspace(0, math.pi, 9); rr = r * (1 - .42 * u)
    P = np.zeros((n, 9, 3)); P[..., 0] = np.cos(a)[None] * rr[:, None]; P[..., 1] = side * (np.sin(a)[None] * rr[:, None] + op[:, None]); P[..., 2] = z[:, None]; g.grid(P, 'mast', uv=np.stack(np.broadcast_arrays(z[:, None], a[None] * r), -1), flip=side < 0, two=True)
    x = np.linspace(1, -1, 7); jg = .04 * np.sin(z * 9)[:, None] * (1 - x ** 2)[None]; Q = np.zeros((n, 7, 3)); Q[..., 0] = x[None] * rr[:, None]; Q[..., 1] = side * (op[:, None] + jg); Q[..., 2] = z[:, None]; g.grid(Q, 'raw', uv=np.stack(np.broadcast_arrays(z[:, None], x[None] * r), -1), two=True)
    if part_char > 0:
        for sg in (1, -1): C = np.stack([np.stack([np.full(n, sg) * rr, side * (op + .02), z], -1), np.stack([sg * rr * (.80 - .12 * np.abs(np.sin(z * 2.3))), side * (op + .035 * side * 0 + .02), z], -1)], 1); g.grid(C, 'char', two=True)
    return g


# ------------------------------------------------------------------ debris
class Debris:
    """closed-form rigid planks / beams / barrels: origins within a volume, burst + drag + gravity + spin.  geo(t) -> Geo, hits(t) -> sea splashes [(x, y, age, size)]"""
    def __init__(self, o, n=60, seed=0, vel=(6, 22), up=(8, 22), size=(.5, 3.4), spread=(3, 3, 1), dirv=None, cone=1.0, spin=9., g=9.8, t0=0., kinds=('plank', 'plank', 'plank', 'beam', 'barrel')):
        r = np.random.default_rng(seed); self.n, self.g, self.t0 = n, g, t0; th = r.uniform(0, 2 * math.pi, n); v = r.uniform(*vel, n); self.p = np.asarray(o, float) + r.normal(0, 1, (n, 3)) * np.asarray(spread, float)
        self.v = np.stack([np.cos(th) * v, np.sin(th) * v, r.uniform(*up, n)], -1) * (1 if dirv is None else cone) + (0 if dirv is None else np.asarray(dirv, float) * r.uniform(.5, 1.2, (n, 1)))
        L = size[0] + (size[1] - size[0]) * r.random(n) ** 2.2; self.h = np.stack([L / 2, L * .07 + .05, np.full(n, .035) + L * .012], -1); self.kind = [kinds[i % len(kinds)] for i in range(n)]
        for i, k in enumerate(self.kind):
            if k == 'beam': self.h[i] = (L[i] * .7, .14, .14)
            if k == 'barrel': self.h[i] = (.45, .36, .36)
        ax = r.normal(0, 1, (n, 3)); self.ax = ax / np.linalg.norm(ax, axis=1)[:, None]; self.w = r.uniform(.3, 1., n) * spin; self.ph = r.uniform(0, 6.28, n); self.drag = .35

    def state(self, t):
        t = max(0., t - self.t0); k = (1 - math.exp(-self.drag * t)) / self.drag; return self.p + self.v * k + np.array([0, 0, -.5 * self.g * t * t]), t

    def rot(self, i, t):
        a = self.ph[i] + self.w[i] * t; u = self.ax[i]; c, s = math.cos(a), math.sin(a); K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]]); return np.eye(3) + s * K + (1 - c) * K @ K

    def geo(self, t, zmin=-.4, step=0):
        g = Geo()
        if t < self.t0: return g
        p, tt = self.state(t)
        for i in range(self.n):
            if p[i, 2] < zmin: continue
            R = self.rot(i, tt)
            if self.kind[i] == 'barrel': a = R[:, 0] * .5; g.cyl(p[i] - a, p[i] + a, .36, .36, 'plank', 8)
            else: g.box(p[i], self.h[i], 'plank' if i % 4 else 'raw', R, part=g.part())
        return g

    def hits(self, t, life=.7):
        out = []
        for i in range(self.n):
            v, z = self.v[i, 2], self.p[i, 2]; th = (v + math.sqrt(max(0, v * v + 2 * self.g * z))) / self.g + self.t0
            if th < t < th + life: q, _ = self.state(th); out.append((q[i, 0], q[i, 1], t - th, float(self.h[i, 0])))
        return out


# ------------------------------------------------------------------ the lion figurehead (x = the way it looks)
def lion(R0=2.1):
    g = Geo(); th = np.linspace(0, 2 * math.pi, 25); ph = np.linspace(.02, math.pi * .56, 11); P = np.zeros((11, 25, 3))
    for i, p in enumerate(ph): P[i] = np.stack([np.full(25, math.cos(p) * R0 * .80), np.cos(th) * math.sin(p) * R0, np.sin(th) * math.sin(p) * R0 * .96], -1)
    g.grid(P, 'lion', flip=True, two=True)
    def petal(a, r0, r1, wd, x0, x1, mat):
        u = np.linspace(0, 1, 7); w = wd * np.sin(math.pi * (.12 + .80 * u)) ** .7; c, s = math.cos(a), math.sin(a); Q = np.zeros((7, 5, 3))
        for j, q in enumerate(np.linspace(-1, 1, 5)): rr = r0 + (r1 - r0) * u; off = q * w; Q[:, j] = np.stack([x0 + (x1 - x0) * u + .22 * (1 - q * q), c * rr - s * off, s * rr + c * off], -1)
        g.grid(Q, mat, two=True)
        g.grid(Q - np.array([.42, 0, 0]), mat, two=True)
    for k in range(12): petal(k / 12 * 2 * math.pi + .13, R0 * .80, R0 * 1.62, .72, -.2, -.35, 'mane')
    for k in range(12): petal((k + .5) / 12 * 2 * math.pi + .13, R0 * .80, R0 * 1.45, .66, -.45, -.65, 'lion')
    for sg in (-1, 1):
        g.cyl((R0 * .66, sg * R0 * .36, R0 * .26), (R0 * .74, sg * R0 * .36, R0 * .26), .30, .28, 'white', 14); g.cyl((R0 * .74, sg * R0 * .34, R0 * .25), (R0 * .77, sg * R0 * .34, R0 * .25), .12, .11, 'ink', 10)
        g.cyl((-.3, sg * R0 * .62, R0 * .78), (.1, sg * R0 * .66, R0 * .86), .42, .36, 'mane', 10)
    g.cyl((R0 * .72, 0, -R0 * .02), (R0 * .96, 0, -R0 * .02), .34, .26, 'ink', 12)
    a = np.linspace(.15, math.pi - .15, 13); M_ = np.stack([np.sqrt(np.maximum(0, 1 - (np.cos(a) * .55) ** 2 - (.12 + np.sin(a) * .30) ** 2)) * R0 * .80 + .05, np.cos(a) * R0 * .55, -R0 * (.18 + np.sin(a) * .34)], -1); g.rope(M_, .06, '#4a2410')
    g.cyl((-6.5, 0, -2.0), (-.6, 0, -.2), 2.5, 1.5, 'cream', 16); g.cyl((-6.6, 0, -2.0), (-6.4, 0, -2.0), 2.7, 2.7, 'red', 16)
    return g
