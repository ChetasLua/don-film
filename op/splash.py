"""splash (fight lane): painted sky panorama for moving cameras, water curtains, rainbow, smears.   Water itself is op.sea."""
import numpy as np, cv2, math, os
from . import W, H
from .paint import rgb, f32, ramp, grid, fbm, smooth
from . import clouds as CL, comp
from .ink import Cel, spl
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EL1, EL0 = math.radians(66), math.radians(-5)
PW, PH = 3800, 1430


class Pano:
    """painted sky over 180 degrees of azimuth (mirrored beyond), sampled by view direction.  sky(cam, az0, dim) -> HxWx3"""
    def __init__(self, name='day', seed=4, cols=((0, '#1b5fb8'), (.42, '#3f9fe0'), (.80, '#9fd9f2'), (.95, '#dcf4f6'), (1, '#e8f8f6')), T=('#ffffff', '#e9f0fa', '#b0c9ea', '#7fa3d6'), mist='#cdeaf6', light=(.62, -.70, .36), big=((.30, 950, 900), (.74, 1200, 1060), (.53, 560, 520), (.06, 700, 620))):
        p = f'{ROOT}/cache/fight_pano_{name}.npy'
        if not os.path.exists(p):
            os.makedirs(ROOT + '/cache', exist_ok=True); y, x = np.mgrid[0:PH, 0:PW].astype(f32); hz = PH * EL1 / (EL1 - EL0); sky = ramp(np.clip(y / hz, 0, 1), cols); r = np.random.default_rng(seed); S = CL.Sky(PH, PW, sky, light=light, seed=seed)
            for k in range(9): S.bank(PW * (k + .5 + r.uniform(-.3, .3)) / 9, hz + 8, r.uniform(500, 820), r.uniform(110, 210), ('#f4f8fb', '#d8e8f4', '#b3d0ea', '#97bde0'), mist, seed=seed + 20 + k, detail=.7, levels=2)
            for k, (fx, w, h) in enumerate(big): S.bank(PW * fx, hz + 30, w, h, T, mist, seed=seed + 40 + k, profile=CL.peak(.5 + r.uniform(-.1, .1), .24, .30, c2=.22 + .5 * (k % 2), h2=.62), detail=1.1)
            for k in range(6): S.bank(PW * r.random(), hz * r.uniform(.22, .55), r.uniform(260, 520), r.uniform(60, 110), T, mist, seed=seed + 60 + k, detail=.8, levels=2, pool=0)
            S.mist(hz - 6, PW / 2, PW, 60, mist, .55, seed); im = S.paint(seed, amt=.7); np.save(p + '.tmp.npy', im.astype(np.float16)); os.replace(p + '.tmp.npy', p)
        self.im = np.load(p).astype(f32)

    def sky(self, cam, az0=0.):
        d = cam.rays(); n = np.sqrt((d * d).sum(-1)); az = np.arctan2(d[..., 0], d[..., 1]) + az0; el = np.arcsin(np.clip(d[..., 2] / n, -1, 1)); u = (az / math.pi + .5) % 2.0; u = np.where(u > 1, 2 - u, u)
        return cv2.remap(self.im, (u * (PW - 1)).astype(f32), ((EL1 - el) / (EL1 - EL0) * (PH - 1)).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def curtain(root, v0, t, na=18, g=9.8, lag=.5, jit=.12, seed=0):
    """ballistic sheet grid for sea.sheet: root (ns,3) tear line, v0 (ns,3) launch velocity, t seconds of age; lag = emission spread along the sheet (s)"""
    root, v0 = np.asarray(root, float), np.asarray(v0, float); r = np.random.default_rng(seed); ns = len(root); k = 1 + jit * np.interp(np.linspace(0, 9, ns), np.arange(10), r.normal(0, 1, 10)); P = np.zeros((ns, na, 3))
    for j, a in enumerate(np.linspace(0, 1, na)):
        tt = max(0, t - lag * (1 - a)) * a ** .25 if t > 0 else 0; tt = t * a; P[:, j] = root + v0 * (k * tt)[:, None] + np.array([0, 0, -.5 * g * tt * tt])
    return P


def rainbow(cx, cy, r, w=46., amt=.5):
    """soft additive arc -> RGB light"""
    x, y = grid(); d = (np.hypot(x - cx, (y - cy)) - r) / w; c = np.zeros((H, W, 3), f32)
    for k, h in enumerate(('#7a5cff', '#3f8cff', '#38d8c8', '#9be24a', '#ffe14a', '#ff9a3a', '#ff4a4a')): c += np.exp(-((d - (k - 3) * .42) / .30) ** 2)[..., None] * rgb(h)
    return c * amt * .55


def burst_lines(cel, cx, cy, r0, r1, n, seed, col='#ffffff', w=6., alpha=1., ry=1.):
    r = np.random.default_rng(seed)
    for k in range(n):
        a = (k + r.uniform(-.4, .4)) / n * 2 * math.pi; q0 = r0 * r.uniform(.8, 1.5); q1 = q0 + (r1 - q0) * r.uniform(.3, 1.); c, s = math.cos(a), math.sin(a); m = np.array([-s, c]) * w * r.random() ** 1.6; p0 = np.array([cx + c * q0, cy + s * q0 * ry]); p1 = np.array([cx + c * q1, cy + s * q1 * ry]); cel.fill(np.array([p0, p0 * .55 + p1 * .45 + m, p1, p0 * .55 + p1 * .45 - m]), col, alpha)
    return cel


def grow_fit(f, f0, f1, p=2.): return min(1., max(0., (f - f0) / max(1e-6, f1 - f0))) ** p
