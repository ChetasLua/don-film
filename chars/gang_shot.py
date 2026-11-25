"""Shot helpers for the roll call (GANG lane): cached painted plates, cameras, card opening, wipes."""
import os
from .gang_kit import *
from op import comp, paint, kit
from op import hands as HD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OW, OH = 2304, 1296                                       # overscan plates (x1.2); plate centre = frame centre


def cached(name, fn):
    p = os.path.join(ROOT, 'cache', name + '.npy')
    if not os.path.exists(p):
        a = np.clip(fn(), 0, 4).astype(np.float16); t = p + '.%d.npy' % os.getpid(); np.save(t, a); os.replace(t, p)
    return np.load(p, mmap_mode='r')


def view(plate, dx=0., dy=0., zoom=1., rot=0.):
    """camera over an overscan plate; dx, dy px in frame space (picture moves by -dx)"""
    return comp.cam2d(np.asarray(plate, f32), OW / 2 + dx, OH / 2 + dy, zoom, rot, border=cv2.BORDER_REFLECT_101 if plate.shape[2] == 3 else cv2.BORDER_CONSTANT)


def painted(g, seed=1, sizes=(34, 16, 8), angle=None, density=1.0, mask=None, **k):
    return paint.mottle(paint.repaint(g, sizes, seed, angle, mask=mask, density=density, **k), .016, seed + 3)


def flat(polys_cols, h=OH, w=OW, base=None, ss=1):
    """guide picture from [(poly, hex)]"""
    c = Cel(w, h, ss=ss)
    if base: c.fill(np.array([[0, 0], [w, 0], [w, h], [0, h]], float), base)
    for P, col in polys_cols: c.fill(P, col)
    return c.rgba()


def split(a): return a[..., :3] / np.maximum(a[..., 3:4], 1e-3), a[..., 3]


def plate_rgba(polys_cols, seed=1, sizes=(26, 13, 6), shade=None, edge=1.4, angle=None, **k):
    """painted cut-out plate: guide shapes -> gouache strokes inside the mask, premultiplied RGBA"""
    g = flat(polys_cols); c, a = split(g); m = (a > .5).astype(np.uint8); d = cv2.dilate(m, np.ones((3, 3), np.uint8)); base = cv2.inpaint((np.clip(c[::4, ::4], 0, 1) * 255).astype(np.uint8), (1 - m[::4, ::4]), 5, cv2.INPAINT_TELEA) if m.min() == 0 and m.max() == 1 else None
    if base is not None: bb = cv2.resize(base, (c.shape[1], c.shape[0]), interpolation=cv2.INTER_LINEAR).astype(f32) / 255.; c = np.where(a[..., None] > .6, c, bb)
    if shade is not None: c = shade(c.astype(f32), a)
    c = painted(np.clip(c, 0, 1).astype(f32), seed, sizes, angle, **k); a = paint.warp_by(a, edge, 18, seed + 5); return np.dstack([c * a[..., None], a])


def lerp2(a, b, m): return a * (1 - m[..., None]) + b * m[..., None]


def blob(cx, cy, R, seed, n=90, spike=.35, lob=7, ry=1.):
    """paint-splat polygon: lobes with thrown fingers"""
    r = np.random.default_rng(seed); a = np.linspace(0, 2 * math.pi, n * 3, endpoint=False); q = np.ones_like(a)
    for _ in range(lob * 2): c = r.uniform(0, 6.28); w = r.uniform(.04, .16); d = np.angle(np.exp(1j * (a - c))); q += spike * r.uniform(.3, 1.3) * np.exp(-(d / w) ** 2)
    q += .06 * np.sin(a * 5 + r.uniform(0, 6)) + .05 * np.sin(a * 11 + r.uniform(0, 6)); return np.c_[cx + np.cos(a) * R * q, cy + np.sin(a) * R * q * ry]


def mask_soft(P): return kit.mask_of(P, ss=1)


def focus(cel, cx, cy, r0, seed, n=120, col='#17131a', w=8, ry=.75): return kit.focus_lines(cel, cx, cy, r0, 1500, n, w, col, seed, ry=ry)


def warp_layer(l, M): return cv2.warpAffine(l, np.float32(M), (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)


def ring_poly(cx, cy, R, w, ry=1., seed=0, jag=.06):
    r = np.random.default_rng(seed); a = np.linspace(0, 2 * math.pi, 120, endpoint=False); n1 = 1 + jag * np.interp(a, np.linspace(0, 2 * math.pi, 25), np.r_[r.normal(0, 1, 24), 0]); O = np.c_[cx + np.cos(a) * R * n1, cy + np.sin(a) * R * n1 * ry]; w_ = w * (1 + .5 * np.sin(a * 3 + seed)); I = np.c_[cx + np.cos(a) * np.maximum(1, R * n1 - w_), cy + np.sin(a) * np.maximum(1, R * n1 - w_) * ry]
    return [O, I[::-1]]


def hand(p, pose, cx, cy, s, R, pal, side='R', dist=6., light=(-.5, .75, .6), w=None, style=None, res=420, bld=None, seed=3, pos=(0, 0, 0)):
    """op.hands drawn on the two planes of a Part (flats / pen)"""
    h = HD.Hand(pose, side, R, pos, bld or HD.MALE, **(style or {})); g = HD.render([h], HD.Cam(cx, cy, s, dist), light, res=res, clip=(p.f.ox, p.f.oy, p.f.ox + p.f.w, p.f.oy + p.f.h))
    pl = dict(HD.PAL['day'], **pal); HD.fill_flat(p.f, g, pl); HD.fill_shadow(p.f, g, pl); HD.fill_overlays(p.f, g, pl, w or p.k)
    c = p.l.ctx; c.set_operator(cairo.OPERATOR_DEST_OUT)
    for G in g.groups:
        if G['polys']: p.l._path(G['polys']); c.set_source_rgba(0, 0, 0, 1); c.set_fill_rule(cairo.FILL_RULE_EVEN_ODD); c.fill()
    c.set_operator(cairo.OPERATOR_OVER); HD.ink_lines(p.l, g, pl, w or p.k, seed); return g


def capsule(a, b, ra, rb, bulge=0., n=14):
    """tapered limb segment with round ends; bulge = extra half width at 40 %"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = np.hypot(*d) + 1e-6; d /= L; m = np.array([-d[1], d[0]]); t = np.linspace(0, 1, 18); w = ra + (rb - ra) * t + bulge * np.sin(np.pi * np.clip(t * 1.12, 0, 1)) ** 1.5
    an = math.atan2(d[1], d[0]); e1 = np.linspace(an + math.pi / 2, an + 1.5 * math.pi, n); e2 = np.linspace(an - math.pi / 2, an + math.pi / 2, n)
    return np.vstack([b + rb * np.c_[np.cos(e2), np.sin(e2)], (a + d * L * t[:, None] + m * w[:, None])[::-1], a + ra * np.c_[np.cos(e1), np.sin(e1)], a + d * L * t[:, None] - m * w[:, None]])


def sides(a, b, ra, rb, bulge=0.):
    """the two long contours of capsule(a, b): ink a limb without its end caps"""
    a, b = np.asarray(a, float), np.asarray(b, float); d = b - a; L = np.hypot(*d) + 1e-6; d /= L; m = np.array([-d[1], d[0]]); t = np.linspace(0, 1, 18); w = ra + (rb - ra) * t + bulge * np.sin(np.pi * np.clip(t * 1.12, 0, 1)) ** 1.5
    return a + d * L * t[:, None] + m * w[:, None], a + d * L * t[:, None] - m * w[:, None]


def union(*Ps):
    """one clean contour round several polygons (screen px)"""
    Ps = [np.asarray(p, float) for p in Ps]; al = np.vstack(Ps); lo = np.floor(al.min(0)) - 6; hi = np.ceil(al.max(0)) + 6; k = 2. if max(hi - lo) < 900 else 1.; w, h = ((hi - lo) * k).astype(int) + 2; m = np.zeros((h, w), np.uint8)
    for p in Ps: cv2.fillPoly(m, [np.round((p - lo) * k * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    c = max(kit.contours(m / 255.), key=len); return resample(c / k + lo, 3.)


def sm(img, dx, dy, n=27): return kit.smear(img, dx, dy, n)
