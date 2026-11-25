"""dragon lane helpers: painted plates, lanterns, rain in 3-D, the skull dome, moon, cameras."""
import numpy as np, cv2, math, os
from scipy.interpolate import CubicSpline, PchipInterpolator
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp, Canvas, repaint, mottle
from op.ink import Cel, spl, resample
from op.clouds import Sky
from op.cam import Camera3D
from op.fx_core import h1, R, noise, FX, put, lin01
from op import kit
from op.comp import blur, radial, screen
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def cached(name, fn):
    p = os.path.join(ROOT, 'cache', f'dr_{name}.npy')
    if not os.path.exists(p): a = fn(); q = p + f'.{os.getpid()}.npy'; np.save(q, np.asarray(a).astype(np.float16)); os.replace(q, p)
    return np.load(p, mmap_mode='r')


def full(name, fn): return np.array(cached(name, fn), f32)


def vgrad(h, w, stops): return np.broadcast_to(ramp(np.linspace(0, 1, h), stops)[:, None], (h, w, 3)).astype(f32).copy()


def path_keys(keys):
    """[(f, vec), ...] -> smooth interpolator (clamped cubic)"""
    t = [k[0] for k in keys]; v = np.array([k[1] for k in keys], float); c = CubicSpline(t, v, bc_type='natural'); a, b = t[0], t[-1]
    return lambda f: c(np.clip(f, a, b))


# ---------------------------------------------------------------- skies
def night_plate():
    h, w = 1250, 2600
    g = vgrad(h, w, [(0, '#070b1e'), (.35, '#101a3c'), (.62, '#22305f'), (.80, '#3d4580'), (1., '#20264e')]); x, y = grid(h, w); mx, my = 1300, 330
    d = np.sqrt((x - mx) ** 2 + (y - my) ** 2); g = g + (np.exp(-(d / 420) ** 1.4) * .42)[..., None] * rgb('#7f9be0') + (np.exp(-(d / 170) ** 2) * .30)[..., None] * rgb('#dfeaff')
    S = Sky(h, w, g, light=(.0, -.82, .45), seed=5); S.strata(380, 760, 46, ('#3a4a86', '#1c2654'), 3)
    r = np.random.default_rng(4)
    far = ('#7184c4', '#42528f', '#263264', '#161f44'); mid = ('#8ea3de', '#42528e', '#222c5c', '#10183a'); near = ('#5f70b4', '#2c3a70', '#182148', '#0b1028')
    for x_, b, w_, h_, T in ((250, 900, 900, 420, far), (2300, 920, 1000, 470, far), (900, 960, 700, 260, far), (1700, 950, 760, 250, far), (-40, 1080, 900, 520, mid), (2680, 1080, 900, 560, mid), (640, 1120, 800, 300, mid), (1980, 1130, 820, 320, mid), (1300, 1180, 900, 230, mid),
                              (200, 1300, 900, 330, near), (1100, 1320, 1100, 260, near), (2100, 1310, 1100, 320, near)):
        S.bank(x_, b, w_, h_, T, '#2c3768', seed=int(r.integers(999)), tiers=3, n=6)
    for i in range(4): S.bank(r.uniform(200, 2400), r.uniform(120, 300), r.uniform(500, 900), r.uniform(60, 120), ('#5a6cae', '#33427c', '#1c2752', '#0e1534'), '#162048', seed=40 + i, levels=2, pool=.2)
    return S.paint(2, amt=.75)


def cloud_sprites(T, seed=0, n=5, light=(0, -.8, .4), S=620):
    out = []
    for i in range(n):
        s = Sky(S, S, '#000000', light=light, seed=seed + i, layered=True); r = np.random.default_rng(seed + i)
        for k in range(3): s.bank(S * (.5 + r.uniform(-.14, .14)), S * (.66 + .06 * k), S * r.uniform(.55, .8), S * r.uniform(.22, .36), T, T[2], seed=seed * 7 + i * 3 + k, levels=2, pool=0.)
        l = s.layer(); m = np.clip(l[..., 3], 0, 1); x, y = grid(S, S); e = smooth(1., .72, np.sqrt(((x - S / 2) / (S * .5)) ** 2 + ((y - S * .58) / (S * .34)) ** 2)); out.append(l * e[..., None])
    return np.stack(out)


def skydome(plate, cam, az=1.35, el=(.75, -.34), haz=0.):
    """plate wrapped on a far cylinder"""
    h, w = plate.shape[:2]; d = cam.rays(); a = np.arctan2(d[..., 0], d[..., 1]) - haz; e = np.arctan2(d[..., 2], np.hypot(d[..., 0], d[..., 1]))
    return cv2.remap(plate, ((a / az * .5 + .5) * (w - 1)).astype(f32), ((el[0] - e) / (el[0] - el[1]) * (h - 1)).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def moon(img, x, y, r, amt=1., col='#f4f7ff', veil=None):
    g = radial(x, y, r * 7, power=2.6) * .42 + radial(x, y, r * 2.6, power=2.) * .40; img = screen(img, (g * amt)[..., None] * rgb('#9fb8ee'))
    c = Cel(); a = np.linspace(0, 6.283, 90, endpoint=False); c.fill(np.c_[x + np.cos(a) * r, y + np.sin(a) * r], col)
    c.clip(np.c_[x + np.cos(a) * r, y + np.sin(a) * r])
    for i in range(9):
        q = .75 * math.sqrt(h1(3, i, 1)); t = 6.28 * h1(3, i, 2); rr = r * (.08 + .22 * h1(3, i, 3)); b = np.linspace(0, 6.283, 20, endpoint=False); c.fill(spl(np.c_[x + r * q * math.cos(t) + np.cos(b) * rr * (1 + .2 * np.sin(b * 3 + i)), y + r * q * math.sin(t) + np.sin(b) * rr * .8], True, n=3), '#cfd9f2', .75)
    c.unclip(); l = c.layer() * amt
    if veil is not None: l = l * (1 - veil[..., None])
    return img * (1 - l[..., 3:4]) + l[..., :3] * 1.08


# ---------------------------------------------------------------- rain in the camera's space
def rain3d(cam, f, vel=(0, 0, 0), n=1500, seed=0, wind=(7., 3., -46.), box=(34., 62., 30.), shutter=1 / 36., col='#cfe0ff', force=None, segs=5, alpha=.5, thick=1.0):
    """drops fixed in the world, falling; streak = path during the shutter seen from the moving camera.  force(P)->(n,3) extra velocity (bends them)"""
    r = np.random.default_rng(seed); B = np.array([box[0] * 2, box[1] * 2, box[2] * 2]); p0 = r.random((n, 3)) * B; wv = np.array(wind, float); t = f / 24.; sp = .8 + .4 * r.random(n)[:, None]
    P = p0 + wv * t * sp; P = (P - cam.eye + B / 2) % B - B / 2 + cam.eye
    rel = np.zeros((n, segs, 2)); zs = None; Q = P.copy(); dt = shutter / (segs - 1); vc = np.asarray(vel, float); ok = np.ones(n, bool)
    for k in range(segs):
        q, z = cam.project(Q + vc * (dt * (segs - 1 - k))); rel[:, k] = q; ok &= z > 1.2
        if k == 0: zs = z
        v = wv * sp
        if force is not None: v = v + force(Q)
        Q = Q + v * dt
    A = np.zeros((H, W), np.uint8)
    for li, (z0, z1, w, a) in enumerate(((26, 99, 1, .42), (9, 26, 1, .72), (1.2, 9, 2, .85))):
        m = ok & (zs >= z0) & (zs < z1) & (np.abs(rel[:, 0, 0] - W / 2) < W) & (np.abs(rel[:, 0, 1] - H / 2) < H); L = np.linalg.norm(rel[:, 0] - rel[:, -1], axis=1); m &= L < 900
        B_ = np.zeros((H, W), np.uint8)
        cv2.polylines(B_, [np.round(q * 8).astype(np.int32) for q in rel[m]], False, 255, max(1, int(round(w * thick))), cv2.LINE_AA, 3); A = np.maximum(A, (B_ * a).astype(np.uint8))
    a = A.astype(f32)[..., None] / 255. * alpha; return np.dstack([a * rgb(col), a])


# ---------------------------------------------------------------- lanterns
def lantern_sprite(S=128, seed=0, near=False):
    """paper sky lantern, lit from inside: warm body, darker crown, ribs, brushed mark, small flame"""
    c = Cel(S, S, ss=3); cx = S / 2; w0, w1 = S * .20, S * .29; y0, y1 = S * .16, S * .80; r = np.random.default_rng(seed)
    P = spl([(cx - w0 * .8, y0 + S * .02), (cx, y0 - S * .02), (cx + w0 * .8, y0 + S * .02), (cx + w1 * 1.02, y0 + S * .22), (cx + w1 * .95, y0 + S * .44), (cx + w0 * .86, y1), (cx, y1 + S * .035), (cx - w0 * .86, y1), (cx - w1 * .95, y0 + S * .44), (cx - w1 * 1.02, y0 + S * .22)], True, n=8)
    c.grad(P, (cx, y0), (cx, y1), [(0, '#b23a1c', 1), (.28, '#f0702a', 1), (.62, '#ffb44a', 1), (1., '#fff3b8', 1)])
    c.clip(P)
    for q in (-.55, 0., .55): c.line(spl([(cx + q * w0 * .6, y0), (cx + q * w1 * 1.25, y0 + S * .3), (cx + q * w0 * 1.05, y1)], n=8), S * .010, '#8a2c14', taper=(.1, .1), alpha=.6, seed=seed)
    c.fill(spl([(cx + w1 * .4, y0), (cx + w1 * 1.2, y0), (cx + w1 * 1.2, y1), (cx + w0 * .55, y1), (cx + w1 * .75, y0 + S * .3)], True, n=6), '#a8321a', .35)
    for k in range(2): c.line(spl([(cx - w1 * .45 + k * w1 * .3, y0 + S * .22 + k * S * .06), (cx - w1 * .1, y0 + S * .30 + r.uniform(-.02, .02) * S), (cx + w1 * .4, y0 + S * .34 + k * S * .07)], n=6), S * .024, '#5a1a10', taper=(.1, .5), alpha=.8, seed=seed + k)
    c.unclip(); e = np.linspace(0, 6.283, 30); c.fill(np.c_[cx + np.cos(e) * w0 * .84, y1 + np.sin(e) * S * .035], '#fffbe0'); c.line(np.c_[cx + np.cos(e) * w0 * .86, y1 + np.sin(e) * S * .036], S * .012, '#7a2a14', closed=True, taper=(0, 0))
    kit.outline(c, P, S * .018, '#4a1610', seed=seed); l = c.rgba()
    return l


class Lanterns:
    """world lanterns: rise, sway, spin; scatter by a shock.  draw() -> composited frame, far first"""
    def __init__(self, n=260, seed=0, box=((-160, 160), (60, 420), (-60, 90)), size=2.2, rise=(3., 7.)):
        r = np.random.default_rng(seed); self.P = np.stack([r.uniform(*b, n) for b in box], -1); self.v = r.uniform(*rise, n); self.ph = r.uniform(0, 6.28, n); self.sz = size * r.uniform(.75, 1.3, n); self.n = n; self.sp = [lantern_sprite(128, i) for i in range(4)]; self.k = r.integers(0, 4, n); self.lo, self.hi = box[2]

    def pos(self, t):
        P = self.P.copy(); P[:, 2] = self.lo + (P[:, 2] - self.lo + self.v * t) % (self.hi - self.lo); P[:, 0] += 1.6 * np.sin(t * 1.1 + self.ph); P[:, 1] += 1.2 * np.cos(t * .8 + self.ph * 2); return P

    def draw(self, img, cam, t, shock=None, zmin=2., zmax=1e9, amt=1., iz=None, dim=1., push=None, wrap=True):
        P = self.pos(t)
        if push is not None: P = P + push(P)
        p, z = cam.project(P); o = np.argsort(-z); gl = np.zeros((H // 4, W // 4), f32); L = np.zeros((H, W, 4), f32)
        for i in o:
            if not zmin < z[i] < zmax: continue
            s = cam.f * self.sz[i] / z[i]; x, y = p[i]
            if s < 1.5 or x < -s * 2 or x > W + s * 2 or y < -s * 2 or y > H + s * 2: continue
            fl = .86 + .14 * math.sin(t * 9 + self.ph[i] * 3)
            if iz is not None and 0 <= int(x) < W and 0 <= int(y) < H and iz[int(y), int(x)] > 1. / z[i]: continue
            if s < 5:
                cv2.circle(gl, (int(x / 4 * 16), int(y / 4 * 16)), int(max(1, s * .16) * 16), float(.8 * fl), -1, cv2.LINE_AA, 4); cv2.circle(L, (int(x * 16), int(y * 16)), int(s * .42 * 16), (1., .74, .34, 1.), -1, cv2.LINE_AA, 4); continue
            sc = s / 128. * 1.6; rot = .14 * math.sin(t * 1.3 + self.ph[i]); S = 128; c_, s_ = math.cos(rot) * sc, math.sin(rot) * sc; q = s * 1.1; x0, y0, x1, y1 = int(max(0, x - q)), int(max(0, y - q)), int(min(W, x + q)), int(min(H, y + q))
            if x1 - x0 < 2 or y1 - y0 < 2: continue
            M = np.float32([[c_, -s_, x - x0 - c_ * S / 2 + s_ * S / 2], [s_, c_, y - y0 - s_ * S / 2 - c_ * S / 2]]); sp = self.sp[self.k[i]]
            if sc < .5: k = max(.4, .35 / sc); sp = cv2.GaussianBlur(sp, (0, 0), k)
            a = cv2.warpAffine(sp, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR); d = L[y0:y1, x0:x1]; L[y0:y1, x0:x1] = d * (1 - a[..., 3:4]) + a * np.array([fl, fl, fl, 1], f32)
            gx, gy, gr = int(x / 4), int(y / 4), max(2, int(s / 4 * .55)); cv2.circle(gl, (gx, gy), gr, float(.75 * fl), -1, cv2.LINE_AA)
        out = img * (1 - L[..., 3:4] * amt) + L[..., :3] * amt * dim
        g = cv2.GaussianBlur(gl, (0, 0), 3) * .55 + cv2.GaussianBlur(gl, (0, 0), 11) * .75; return out + cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * rgb('#ff8a30') * amt * dim


# ---------------------------------------------------------------- the island: horned skull dome
def skull_dome(cel, x, y, s, pal=('#3a4272', '#252c56', '#161b3a', '#0b0e24'), line='#080a18', glow='#ffb84a', l=(-.6, -.7), lw=2.0, lights=True, seed=0):
    """s = width of the skull; (x, y) = chin line.  Horns curve up from the temples; terraces of roofs at its foot"""
    X = lambda P: np.asarray(P, float) * s + [x, y]
    for sg in (-1, 1):
        h = spl([(sg * .40, -.52), (sg * .62, -.66), (sg * .86, -.96), (sg * .90, -1.42), (sg * .70, -1.08), (sg * .56, -.86), (sg * .34, -.74)], True, corners=(3,), n=8); cel.fill(X(h), pal[1]); cel.fill(X(spl([(sg * .40, -.52), (sg * .62, -.66), (sg * .86, -.96), (sg * .90, -1.42), (sg * .80, -1.0), (sg * .60, -.74)], True, corners=(3,), n=8)), pal[2]); kit.outline(cel, X(h), lw, line, seed=seed + sg)
        for k in range(4): q = .25 + k * .16; a = np.array([sg * (.40 + .46 * q), -.58 - .62 * q * q]); cel.line(X([a + [-sg * .06, -.08], a + [sg * .05, .02]]), lw * .7, line, seed=k)
    d = spl([(-.50, -.06), (-.54, -.34), (-.48, -.64), (-.26, -.90), (0, -.97), (.26, -.90), (.48, -.64), (.54, -.34), (.50, -.06), (.36, .02), (.30, .16), (-.30, .16), (-.36, .02)], True, n=8); cel.fill(X(d), pal[0]); cel.clip(X(d))
    cel.fill(X(spl([(.06, -1.0), (.30, -.7), (.20, -.4), (.34, -.2), (.2, .2), (.7, .2), (.7, -1.0)], True, n=8)), pal[1]); cel.fill(X(spl([(.34, -.95), (.44, -.6), (.38, -.3), (.46, 0), (.7, .2), (.7, -1)], True, n=8)), pal[2])
    for sg in (-1, 1): cel.fill(X(spl([(sg * .10, -.34), (sg * .14, -.50), (sg * .30, -.54), (sg * .40, -.42), (sg * .34, -.28), (sg * .20, -.26)], True, n=8)), pal[3])
    cel.fill(X(spl([(-.06, -.12), (0, -.26), (.06, -.12), (0, -.15)], True, corners=(0, 1, 2), n=6)), pal[3])
    for i in range(7): xx = -.27 + i * .09; cel.fill(X([(xx - .038, -.04), (xx + .038, -.04), (xx + .034, .13), (xx - .034, .13)]), pal[0]); cel.line(X([(xx + .045, -.06), (xx + .042, .14)]), lw, line, seed=i)
    cel.line(X(spl([(-.34, -.05), (0, -.02), (.34, -.05)], n=8)), lw * 1.2, line); r = np.random.default_rng(seed)
    for i in range(26): a = r.uniform(-.5, .5); b = r.uniform(-.95, .1); cel.line(X(spl([(a, b), (a + r.normal(0, .03), b + .05), (a + r.normal(0, .05), b + .11)], n=4)), lw * .55, line, taper=(.2, .6), seed=i, alpha=.6)
    for c in ([(-.2, -.95), (-.12, -.8), (-.18, -.68), (-.08, -.58)], [(.28, -.86), (.22, -.72), (.30, -.62)]): cel.line(X(c), lw, line, seed=3)
    cel.unclip(); kit.outline(cel, X(d), lw * 1.3, line, seed=seed)
    if lights:
        for sg in (-1, 1): cel.fill(X(spl([(sg * .18, -.34), (sg * .24, -.44), (sg * .32, -.38), (sg * .25, -.31)], True, n=6)), glow, .9)
        for i in range(60):
            a = r.uniform(-.62, .62); b = .13 + .22 * r.random() ** 1.3; k = .010 + .010 * r.random(); cel.fill(X([(a - k, b - k * .7), (a + k, b - k * .7), (a + k, b + k * .7), (a - k, b + k * .7)]), glow, .5 + .5 * r.random())


def roofs(cel, x0, x1, y, h, seed, cols=('#1c2247', '#12173a', '#0a0d26'), glow='#ffb64c', n=26, lit=.75, lw=1.6, line='#070918', banners=True):
    """a festival town in silhouette: tiled curved roofs, pagodas, warm windows, strings of lanterns"""
    r = np.random.default_rng(seed); xs = np.sort(r.uniform(x0, x1, n))
    for i, x in enumerate(xs):
        w = h * r.uniform(.45, 1.1); t = 1 + int(r.random() < .30) * int(r.integers(1, 4)); yb = y + h * r.uniform(-.1, .25); hh = h * r.uniform(.30, .55); c = cols[0 if r.random() < .4 else 1]; lv = r.random() < lit
        for k in range(t):
            ww = w * (1 - .22 * k); yy = yb - k * hh * .95; cel.fill([(x - ww * .40, yy + hh), (x + ww * .40, yy + hh), (x + ww * .40, yy - hh * .1), (x - ww * .40, yy - hh * .1)], cols[2])
            if lv:
                m = int(max(1, ww / (h * .16)))
                for q in range(m): a = x - ww * .36 + ww * .72 * (q + .5) / m; g = ww * .30 / m; cel.fill([(a - g, yy + hh * .25), (a + g, yy + hh * .25), (a + g, yy + hh * .70), (a - g, yy + hh * .70)], glow, .55 + .45 * r.random())
            rf = spl([(x - ww * .66, yy + hh * .12), (x - ww * .34, yy - hh * .16), (x - ww * .14, yy - hh * .55), (x + ww * .14, yy - hh * .55), (x + ww * .34, yy - hh * .16), (x + ww * .66, yy + hh * .12), (x, yy + hh * .06)], True, corners=(0, 2, 3, 5), n=6); cel.fill(rf, c); cel.fill(spl([(x + ww * .02, yy - hh * .55), (x + ww * .14, yy - hh * .55), (x + ww * .34, yy - hh * .16), (x + ww * .66, yy + hh * .12), (x + ww * .1, yy + hh * .06)], True, corners=(0, 1, 3), n=6), cols[2]); kit.outline(cel, rf, lw, line, seed=i * 5 + k)
