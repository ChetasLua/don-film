"""Lightning and shock, drawn: black lightning with red edges, storm bolts, crackle around a will, rings, the air that bends.

    segs = bolt.make(p0, p1, w=16, seed=3)                    # [(points, widths)] trunk + branches, angular, thick at the bends
    bolt.ink_black(cel, segs)      bolt.ink_storm(cel, segs)   # on a Cel      bolt.glow(segs, col, r, amt) -> additive RGB
    fx = bolt.black(f, p0, p1, w=18, seed=1, t0=10, life=6)    # FX: runs out in 2 f, redrawn every 2 f, breaks into shards       put(img, fx)
    fx = bolt.storm(f, p0, p1, ...)                            # white-violet, big light
    fx = bolt.crackle(f, cx, cy, rx, ry, n=9, seed=0, amt=1.)  # conqueror crackle on an ellipse (black/red), arcs jumping outward on twos
    fx = bolt.web(f, P, n=.., w=..)                            # arcs crawling along a polyline (an arm, a body)
    bolt.ring(cel, cx, cy, t, r0, r1, asp, rot, w)             # drawn shock ring     img = bolt.bend(img, cx, cy, r, wd, amt, asp)  # lens ripple of the air
"""
from .fx_core import *
from .comp import blur

RED, HOT, CORE = '#e3132c', '#ffd9d0', '#0a0509'


def zig(p0, p1, seed, jag=.17, n=7, sub=2):
    """angular path: few big zigzags, each side broken again"""
    r = R(seed, 5); p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.hypot(*d) + 1e-6; nrm = np.array([-d[1], d[0]]) / L
    u = np.sort(np.clip((np.arange(1, n) + r.uniform(-.36, .36, n - 1)) / n, .03, .97)); sg = np.where(np.arange(n - 1) % 2, 1, -1) * r.choice([1, 1, 1, -.4], n - 1)
    P = [p0] + [p0 + d * a + nrm * L * jag * s * r.uniform(.25, 1.) * (1 - .25 * abs(2 * a - 1)) for a, s in zip(u, sg)] + [p1]
    for _ in range(sub - 1):
        Q = [P[0]]
        for a, b in zip(P[:-1], P[1:]):
            e = b - a; l = np.hypot(*e); m = np.array([-e[1], e[0]]) / (l + 1e-6); k = r.uniform(.3, .7); Q += [a + e * k + m * l * r.normal(0, .13), b]
        P = Q
    return np.array(P)


def make(p0, p1, w=14., seed=0, jag=.17, n=7, kids=4, depth=2, sub=2, spread=.75):
    """-> [(P, Wd)]: trunk first"""
    r = R(seed, 9); P = zig(p0, p1, seed, jag, n, sub); m = len(P); t = np.linspace(0, 1, m)
    Wd = w * (.55 + .65 * r.random(m)) * np.minimum(1, .25 + t * 5) * (1 - .62 * t); Wd[-1] = 0.; out = [(P, Wd)]
    if depth > 0:
        d = np.asarray(p1, float) - p0; L = np.hypot(*d); th = math.atan2(d[1], d[0])
        for i in range(kids):
            k = int(r.integers(1, max(2, m - 2))); a = th + r.choice([-1, 1]) * r.uniform(.35, 1.) * spread; l = L * r.uniform(.14, .42) * (1 - .5 * t[k])
            out += make(P[k], P[k] + l * np.array([math.cos(a), math.sin(a)]), max(1., Wd[k] * .62), seed * 13 + i + 1, jag * 1.15, max(3, int(n * .6)), max(1, kids // 2), depth - 1, sub, spread)
    return out


def cut(segs, t=1., frag=0., seed=0):
    """first t of the length; frag 0..1 breaks it into shards"""
    out = []
    for q, (P, Wd) in enumerate(segs):
        if t < 1:
            d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; s = d[-1] * (t if q == 0 else max(0, t * 1.5 - .5))
            if s < 4: continue
            k = int(np.searchsorted(d, s)); e = [np.interp(s, d, P[:, i]) for i in (0, 1)]; P = np.vstack([P[:max(1, k)], e]); Wd = np.r_[Wd[:max(1, k)], 0.]
        if frag <= 0: out.append((P, Wd)); continue
        i = 0
        while i < len(P) - 1:
            e = i + 1 + int(2 * h1(seed, q, i)); e = min(e, len(P) - 1)
            if h1(seed, q, i, 3) > frag:
                w = Wd[i:e + 1] * (1 - .5 * frag); w = w.copy(); w[0] *= .2; w[-1] = 0; out.append((P[i:e + 1], w))
            i = e
    return out


def poly(P, Wd, grow=0.):
    """mitred outline of a polyline with per-vertex width"""
    n = len(P)
    if n < 2: return np.zeros((0, 2))
    T = np.diff(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]
    A = np.vstack([N[:1], N[:-1] + N[1:], N[-1:]]); ln = np.hypot(*A.T)[:, None] + 1e-6; A = A / ln; c = np.ones(n)
    if n > 2: c[1:-1] = 1 / np.clip((A[1:-1] * N[1:]).sum(1), .42, 1)
    hw = (Wd * .5 + grow * np.where(Wd > 0, 1, .15))[:, None] * c[:, None]; a, b = P + A * hw, P - A * hw
    return np.vstack([P[:1] - T[0] * grow * 1.5, a, P[-1:] + T[-1] * grow * 2.2, b[::-1]])


def ink_black(cel, segs, edge=4.5, red=RED, hot=HOT, core=CORE, k=1.):
    for g, c in ((edge * 1.45 * k, red), (edge * .42 * k, hot), (0., core)):
        for P, Wd in segs:
            q = poly(P, Wd * k * 1.35, g * (.45 + .55 * min(1, float(Wd.max()) / 12.)))
            if len(q): cel.fill(q, c)
    return cel


def ink_storm(cel, segs, rim='#8f86ff', core='#ffffff', k=1.):
    for g, c in ((3.6 * k, rim), (0., core)):
        for P, Wd in segs:
            q = poly(P, Wd * k * (1 if g else .85), g * min(1, float(Wd.max()) / 8.))
            if len(q): cel.fill(q, c)
    return cel


def glow(segs, col=RED, r=(10, 46), amt=.7, wts=(.6, .5), grow=3.):
    return glow_poly([q for q in (poly(P, Wd, grow) for P, Wd in segs) if len(q) > 2], col, r, wts, amt)


def _fx(cel, li):
    """glow never washes out the black core"""
    l = cel.layer(); c = l[..., 3] * (1 - smooth(.06, .30, l[..., :3].max(-1))); return FX(l, li * (1 - .93 * c)[..., None])


def _life(f, t0, life, run=2):
    a = f - t0
    if a < 0 or a >= life: return None
    return a, min(1., (a + 1) / run), np.clip((a - life + 3) / 3., 0, .85)


def black(f, p0, p1, w=16., seed=0, t0=0, life=7, step=2, kids=5, jag=.17, n=7, edge=4.5, amt=.75, run=2, red=RED):
    q = _life(f, t0, life, run)
    if q is None: return None
    a, t, fr = q; sd = seed * 31 + int(a) // step; cel = Cel(); s = cut(make(p0, p1, w, sd, jag, n, kids), t, fr, sd)
    if not s: return None
    ink_black(cel, s, edge, red); return _fx(cel, glow(s, red, (12, 60), amt * (1 - .6 * fr)))


def storm(f, p0, p1, w=13., seed=0, t0=0, life=6, kids=7, jag=.15, n=9, amt=1.0, rim='#8f86ff', run=1):
    q = _life(f, t0, life, run)
    if q is None: return None
    a, t, fr = q; s = cut(make(p0, p1, w * (1.2 if a < 1 else .8), seed, jag, n, kids, 2), t, fr, seed)
    if not s: return None
    cel = Cel(); ink_storm(cel, s, rim); return FX(cel.layer(), glow(s, rim, (14, 90), amt * (1.2 if a < 2 else .6), (.7, .9)))


def crackle(f, cx, cy, rx, ry, n=9, seed=0, amt=1., w=13., out=.55, step=2, kind='black', red=RED, lw=4.2, arc=.35):
    """arcs leaping around / off an ellipse; each slot lives 5 f and is born somewhere else"""
    cel = Cel(); allq = []
    for i in range(n):
        q = f / 5. + h1(seed, i, 1); cyc = int(math.floor(q)); tau = q - cyc; sd = seed * 77 + i * 13 + cyc * 7
        if h1(sd, 2) > .82 * min(1, amt + .2): continue
        th = 6.283 * h1(sd, 3); b = th + (h1(sd, 4) - .5) * 2 * arc * 3.1; k0 = .82 + .2 * h1(sd, 5); k1 = 1.02 + out * (.3 + h1(sd, 6))
        p0 = (cx + math.cos(th) * rx * k0, cy + math.sin(th) * ry * k0); p1 = (cx + math.cos(b) * rx * k1, cy + math.sin(b) * ry * k1)
        s = cut(make(p0, p1, w * (.5 + .9 * h1(sd, 7)) * amt, sd + int(f) // step, .22, 5, 3, 1), min(1, tau * 3.4 + .25), np.clip((tau - .55) * 2, 0, .85), sd)
        allq += s
    if not allq: return None
    (ink_black if kind == 'black' else ink_storm)(cel, allq, **(dict(edge=lw, red=red) if kind == 'black' else {}))
    return _fx(cel, glow(allq, red if kind == 'black' else '#8f86ff', (10, 44), .65 * amt))


def web(f, P, n=10, w=10., seed=0, off=60., span=.16, kind='black', step=2, red=RED, amt=1., edge=4.):
    """arcs crawling along a polyline P (off = sideways reach px)"""
    p = Path(np.asarray(P, float), 8.); cel = Cel(); A = []
    for i in range(n):
        q = f / 6. + h1(seed, i, 1); cyc = int(math.floor(q)); tau = q - cyc; sd = seed * 91 + i * 17 + cyc * 5
        if h1(sd, 1) > .8 * amt + .1: continue
        u = h1(sd, 2) * (1 - span); s0, s1 = u * p.L, (u + span * (.4 + .8 * h1(sd, 3))) * p.L; o = off * (2 * h1(sd, 4) - 1)
        s = cut(make(p.at(s0, o), p.at(min(s1, p.L - 1), -o * .7), w * (.4 + .8 * h1(sd, 5)), sd + int(f) // step, .2, 6, 3, 1), min(1, tau * 3 + .3), np.clip((tau - .6) * 2, 0, .85), sd); A += s
    if not A: return None
    (ink_black if kind == 'black' else ink_storm)(cel, A, **(dict(edge=edge, red=red) if kind == 'black' else {}))
    return _fx(cel, glow(A, red if kind == 'black' else '#8f86ff', (10, 44), .6))


def _crescents(cel, cx, cy, rx, ry, w, seed, col, gaps, rot, n, spurs=.35):
    """thick-thin arcs round an ellipse with flicked ends and a few outward spurs"""
    c, s = math.cos(rot), math.sin(rot); g = R(seed, 4); e = np.sort((np.arange(n) + g.uniform(-.3, .3, n)) / n) * 6.283
    X = lambda a, q: c2(cx + np.cos(a) * rx * q * c - np.sin(a) * ry * q * s, cy + np.cos(a) * rx * q * s + np.sin(a) * ry * q * c)
    for i in range(n):
        a0 = e[i]; a1 = (e[i + 1] if i + 1 < n else e[0] + 6.283) - gaps * (.2 + .8 * g.random()) * 6.283 / n
        if a1 - a0 < .04: continue
        a = np.linspace(a0, a1, 28); v = np.linspace(0, 1, 28); hw = w * .5 * np.sin(np.pi * v ** (.7 + .7 * g.random())) ** .8 * (.5 + .7 * g.random()); k = hw / max(rx, 1.)
        out = X(a, 1 + k); inn = X(a, 1 - k * .7); cel.fill(np.vstack([out, inn[::-1]]), col)
        for _ in range(int(spurs * 4 * g.random())):
            b = a0 + (a1 - a0) * g.random(); ln = w / max(rx, 1.) * (1.5 + 4 * g.random()); q = X(np.array([b - .012, b + .012, b + .03 * g.normal()]), np.array([1.0, 1.0, 1 + ln])); cel.fill(q, col)


def ring(cel, cx, cy, t, r0=60., r1=1200., asp=.35, rot=0., w=26., seed=0, col='#ffffff', shade='#8a86a8', n=9):
    """drawn shock ring, t 0..1: white thick-thin crescents over a shadow lip, breaking up as it grows"""
    if not 0 <= t < 1: return cel
    e = 1 - (1 - t) ** 2.6; r = r0 + (r1 - r0) * e; k = (1 - t) ** 1.2
    if shade: _crescents(cel, cx, cy + w * .3 * k, r * .975, r * asp * .965, w * k * 1.3, seed, shade, .15 + .5 * t, rot, n)
    _crescents(cel, cx, cy, r, r * asp, w * k, seed, col, .15 + .55 * t, rot, n); _crescents(cel, cx, cy, r * 1.09, r * asp * 1.09, w * k * .22 + 1, seed + 5, col, .5 + .3 * t, rot, n + 4, .2)
    return cel


def bend(img, cx, cy, r, wd=120., amt=26., asp=1., light=.10):
    """the air itself: a refracting ring of radius r"""
    h, w = img.shape[:2]; x, y = grid(h, w); dx, dy = x - cx, (y - cy) / asp; d = np.sqrt(dx * dx + dy * dy) + 1e-3; u = (d - r) / wd; g = -u * np.exp(-u * u) * 2.3
    o = cv2.remap(img, (x + dx / d * g * amt).astype(f32), (y + dy / d * g * amt).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    return o * (1 + light * np.exp(-((u + .2) / .35) ** 2))[..., None] if light else o
