"""The lord of the coast: a scarred sea king, sculpted as a deformable 3-D build and drawn as an inked cel (op.fleet3).

    G = seaking(spine, belly, R, jaw=.5, t=sec, hit=0..1, dent=(point, dir, amount, radius), ripple=(s, amp, width), eye='glare')  -> fleet3.Geo (world metres)
spine = control points, deep water -> back of the head; belly = world vector the pale side faces; R columns = (forward, left, up) of the head.
head_pt(R, o, (h, y, z)) -> world point in head space (h forward 0..9, y left, z up)."""
import numpy as np, math
from op.fleet3 import Geo, grid_faces
from scipy.interpolate import CubicSpline

HS = np.array([-2.4, -.8, .8, 2.2, 3.6, 5.0, 6.4, 7.6, 8.5, 8.95])
WD = np.array([2.75, 3.15, 3.45, 3.40, 3.05, 2.65, 2.45, 2.25, 1.55, .35])
TP = np.array([2.6, 3.1, 3.5, 3.35, 2.55, 1.95, 1.75, 1.85, 1.35, .45])
MZ = np.array([-2.6, -1.9, -.95, -.55, -.45, -.42, -.40, -.30, -.12, .05])
JH = np.array([.2, 1.2, 2.4, 3.8, 5.2, 6.4, 7.4, 8.0])
JW = np.array([2.95, 3.0, 2.9, 2.6, 2.25, 1.95, 1.45, .35])
JD = np.array([2.3, 2.5, 2.3, 1.9, 1.7, 1.6, 1.3, .5])


def head_pt(R, o, q): return np.asarray(o, float) + np.asarray(q, float) @ np.asarray(R, float).T


def _ip(x, xs, ys): return np.interp(x, xs, ys)


def seaking(spine, belly, R, jaw=.35, t=0., hit=0., dent=None, ripple=None, eye='glare', flare=0., whisk=(0, 0, 0), look=0.):
    g = Geo(); R = np.asarray(R, float); sp_ = np.asarray(spine, float); kn = np.r_[0, np.cumsum(np.linalg.norm(np.diff(sp_, axis=0), axis=1) ** .6)]; C = CubicSpline(kn, sp_)(np.linspace(0, kn[-1], 120)); d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(C, axis=0), axis=1))]; n = 64; s = np.linspace(0, d[-1], n); C = np.stack([np.interp(s, d, C[:, i]) for i in range(3)], -1); u = s / d[-1]
    T = np.gradient(C, axis=0); T /= np.linalg.norm(T, axis=1)[:, None]; B = np.asarray(belly, float); F = B - T * (T @ B)[:, None]; F /= np.linalg.norm(F, axis=1)[:, None]; S = np.cross(F, T)
    r = np.interp(u, [0, .25, .55, .8, .93, 1], [3.9, 3.8, 3.3, 2.8, 2.75, 2.9]); r = r * (1 + .018 * np.sin(u * 60))
    if ripple: sc, amp, wd = ripple; k = np.exp(-((u - sc) / wd) ** 2); r = r * (1 + amp * k * np.cos((u - sc) / wd * 5.0)); C = C + S * (amp * 2.0 * k * np.sin((u - sc) / wd * 4.2))[:, None]
    def ring(a0, a1, m, mat, k=1.):
        a = np.linspace(a0, a1, m); P = C[:, None] + (r * k)[:, None, None] * (np.cos(a)[None, :, None] * F[:, None] * 1.04 + np.sin(a)[None, :, None] * S[:, None]); uv = np.stack(np.broadcast_arrays((a * 3.2)[None, :], s[:, None]), -1); g.grid(P, mat, uv=uv, two=True); return P
    bw = .92
    ring(-bw, bw, 13, 'skbelly'); ring(bw, 2 * math.pi - bw, 23, 'skback')
    # dorsal sail
    back = C - F * r[:, None] * 1.0; z = np.abs((np.arange(n) * .30 + .2) % 2 - 1); hgt = (.5 + 2.2 * z) * np.interp(u, [0, .2, .9, 1], [.5, 1, 1, .6]); tip = C - F * (r + hgt)[:, None] + T * (.5 * z * -1.2)[:, None] + S * (.35 * np.sin(t * 5 + u * 14))[:, None]
    g.grid(np.stack([back + F * .3, back * .5 + tip * .5, tip], 1), 'skfin', two=True, uv=np.stack(np.broadcast_arrays(s[:, None], np.array([0, 1, 2.])[None]), -1))
    for sg in (-1, 1): st = C + (r * 1.005)[:, None] * (np.cos(1.35) * F * 1.04 + sg * np.sin(1.35) * S); g.rope(st[8:], .07, '#2a2c5c')
    # ---------------------------------------------------------------- head (local h forward, y left, z up)
    o = C[-1]; loc = []
    def put(P, mat, **k): loc.append((np.asarray(P, float), mat, k))
    h = np.linspace(HS[0], HS[-1], 26); w, tp, mz = _ip(h, HS, WD), _ip(h, HS, TP), _ip(h, HS, MZ); th = np.linspace(0, math.pi, 21); cs = np.sign(np.cos(th)) * np.abs(np.cos(th)) ** .80; sn = np.sin(th) ** .80
    brow = .55 * np.exp(-((h - 2.6) / .9) ** 2)
    up = np.stack([np.repeat(h[:, None], 21, 1), w[:, None] * cs[None], mz[:, None] + (tp - mz)[:, None] * sn[None] + (brow[:, None] * np.exp(-((np.abs(cs) - .62) / .22) ** 2)[None])], -1); put(up, 'skback', uv=np.stack(np.broadcast_arrays((th * 3)[None], h[:, None]), -1))
    lip = np.stack([np.stack([h, sg * w * 1.02, mz + .42], -1) for sg in (1, -1)], 0)
    for sg in (1, -1): put(np.stack([np.stack([h[4:], sg * (w[4:] * 1.015), mz[4:] + .62], -1), np.stack([h[4:], sg * (w[4:] * 1.06), mz[4:] + .15], -1), np.stack([h[4:], sg * (w[4:] * .92), mz[4:] - .12], -1)], 1), 'skbelly', uv=np.stack(np.broadcast_arrays(np.array([0, .4, .8])[None], h[4:, None]), -1))
    y_ = np.linspace(-1, 1, 9); pal = np.stack([np.repeat(h[:, None], 9, 1), (w * .95)[:, None] * y_[None], mz[:, None] + (.75 * (1 - y_ ** 2))[None]], -1); put(pal[4:], 'skmouth')
    # lower jaw
    hj = np.linspace(JH[0], JH[-1], 20); jw, jd = _ip(hj, JH, JW), _ip(hj, JH, JD); tj = np.linspace(0, math.pi, 15); c2, s2 = np.sign(np.cos(tj)) * np.abs(np.cos(tj)) ** .8, np.sin(tj) ** .85; hz = np.array([.2, -1.0]); ca, sa = math.cos(jaw), math.sin(jaw)
    def swing(P):
        P = np.asarray(P, float).copy(); x, z = P[..., 0] - hz[0], P[..., 2] - hz[1]; P[..., 0] = hz[0] + x * ca + z * sa; P[..., 2] = hz[1] - x * sa + z * ca; return P
    jl = -.55
    low = np.stack([np.repeat(hj[:, None], 15, 1), jw[:, None] * c2[None], jl - jd[:, None] * s2[None]], -1); put(swing(low[:, 3:-3]), 'skbelly', uv=np.stack(np.broadcast_arrays((tj[3:-3] * 2.5)[None], hj[:, None]), -1)); put(swing(low[:, :4]), 'skback', uv=np.stack(np.broadcast_arrays((tj[:4] * 2.5)[None], hj[:, None]), -1)); put(swing(low[:, -4:]), 'skback', uv=np.stack(np.broadcast_arrays((tj[-4:] * 2.5)[None], hj[:, None]), -1))
    flr = np.stack([np.repeat(hj[:, None], 9, 1), (jw * .95)[:, None] * y_[None], jl - (.55 * (1 - y_ ** 2))[None] - 0 * hj[:, None]], -1); put(swing(flr), 'skmouth')
    tl = np.linspace(.8, 6.6, 12); lift = .35 + .9 * np.clip(jaw - .2, 0, 1) * np.sin(np.linspace(0, 2.6, 12)); tw = 1.25 * (1 - ((tl - .8) / 6.0) ** 3 * .75); put(swing(np.stack([np.repeat(tl[:, None], 7, 1), tw[:, None] * np.linspace(-1, 1, 7)[None], (jl + lift)[:, None] + .35 * (np.cos(np.linspace(-1, 1, 7) * 1.4) - .6)[None]], -1)), 'tongue')
    # cheek skin between the jaws
    for sg in (1, -1):
        a = np.stack([h[3:11], sg * w[3:11] * .98, mz[3:11] - .02], -1); hb = np.linspace(.2, 3.0, 8); b = swing(np.stack([hb, sg * _ip(hb, JH, JW) * .98, np.full(8, jl)], -1)); q = np.linspace(0, 1, 6)[None, :, None]; m_ = a[:, None] * (1 - q) + b[:, None] * q; m_[:, :, 1] -= sg * .7 * np.sin(math.pi * q[0, :, 0])[None] * np.linspace(.2, 1, 8)[:, None]; put(m_, 'skmouth')
    rr = np.random.default_rng(4); cones = []
    for sg in (1, -1):
        for k, hh in enumerate(np.linspace(2.6, 8.4, 13)):
            ln = (.55 + .55 * rr.random()) * (2.0 if k in (9, 11) else 1.0) * (1.35 if k % 3 == 0 else 1); yy = sg * _ip(hh, HS, WD) * .93; a = np.array([hh, yy, _ip(hh, HS, MZ) + .10]); cones.append((a, a + np.array([.12, sg * .06, -ln]), .17 + .13 * ln, False))
        for k, hh in enumerate(np.linspace(2.4, 7.6, 12)):
            ln = (.5 + .5 * rr.random()) * (1.9 if k == 9 else 1.0); yy = sg * _ip(hh, JH, JW) * .92; a = np.array([hh, yy, jl - .1]); cones.append((a, a + np.array([.05, sg * .05, ln]), .16 + .12 * ln, True))
    for yy in (-.7, 0., .7): cones.append((np.array([8.72, yy, -.05]), np.array([8.85, yy, -.95]), .24, False))
    W_ = lambda P: o + np.asarray(P, float) @ R.T
    fx = []
    if dent is not None:
        p0, dv, am, rad = dent; p0, dv = np.asarray(p0, float), np.asarray(dv, float); dv = dv / np.linalg.norm(dv)
        def D(P):
            q = P - p0; al = q @ dv; per = np.linalg.norm(q - al[..., None] * dv, axis=-1); k = np.exp(-(per / rad) ** 2) * np.exp(-(np.maximum(0, al) / (rad * 1.7)) ** 2); ringk = np.exp(-((per - rad * 1.25) / (rad * .42)) ** 2) * np.exp(-(al / (rad * 1.3)) ** 2); out = (q - al[..., None] * dv) / (per[..., None] + 1e-6)
            wide = np.exp(-(np.linalg.norm(q, axis=-1) / (rad * 3.4)) ** 2)
            return P + dv * (am * k + am * .28 * wide)[..., None] - dv * (am * .30 * ringk)[..., None] + out * (am * .24 * ringk)[..., None]
    else: D = lambda P: P
    for P, mat, k in loc: g.grid(D(W_(P)), mat, two=True, **k)
    for a, b, rad_, lo in cones:
        if lo: a, b = swing(a), swing(b)
        a, b = D(W_(a)[None])[0], D(W_(b)[None])[0]; g.cyl(a, b, rad_, .02, 'tooth', 7, part=g.part(), caps=False)
    for sg in (1, -1):                                                # eyes
        c = np.array([3.05, sg * 2.72, 1.75]); er = .82 * (1 + .30 * hit); th2, ph2 = np.linspace(0, 2 * math.pi, 15), np.linspace(0, math.pi, 9); E = np.stack([np.stack([er * np.sin(p) * np.cos(th2), er * np.sin(p) * np.sin(th2), np.full(15, er * np.cos(p))], -1) for p in ph2]); ax = np.array([.62, sg * .72, .10]) + np.array([0, 0, -.05]); ax /= np.linalg.norm(ax); e1 = np.cross([0, 0, 1.], ax); e1 /= np.linalg.norm(e1); e2 = np.cross(ax, e1)
        Q = c + E[..., :1] * e1 + E[..., 1:2] * e2 + E[..., 2:] * ax; g.grid(D(W_(Q)), 'white' if hit > .4 else 'skeye', two=True)
        pr = .15 if hit > .4 else .23; pc = c + ax * (er * .985) + e1 * look * .25; dk = np.stack([pc + (np.cos(th2)[:, None] * e1 * .8 + np.sin(th2)[:, None] * e2 * 1.5) * q for q in (0.01, pr)], 0); g.grid(D(W_(dk)) + (R @ ax) * .03, 'ink', two=True)
        q = np.linspace(0, 1, 9); dr = .62 if hit < .4 else -.1; bl = np.stack([np.stack([1.4 + 3.4 * q, sg * (2.75 + .35 * np.sin(q * 3.1) - .55 * q), 2.75 - (dr) * q ** 1.2 * 1.5 + .25 * np.sin(q * 3.14)], -1) + np.array([0, sg * a_, b_]) for a_, b_ in ((0, .34), (.34, .1), (.22, -.22))], 1); g.grid(D(W_(bl)), 'skback', two=True, uv=np.stack(np.broadcast_arrays(np.array([0, .4, .8])[None], q[:, None] * 3), -1))
        nz = np.array([8.3, sg * .62, 1.15]); g.grid(D(W_(np.stack([nz + np.c_[np.zeros(15), np.cos(th2) * q, np.sin(th2) * q * .6] for q in (.02, .26)]))), 'ink', two=True)
        # fan fins
        m = 7; root = np.array([-.4, sg * 3.0, .4]); fan = []
        for j in range(m):
            a = -1.15 + 2.1 * j / (m - 1); ln = (4.6 + .9 * (j % 2)) * (1 - .25 * abs(j - 3) / 3) * (1 + .15 * flare); dr_ = np.array([-.75, sg * (.62 + .30 * flare + .10 * math.sin(t * 6 + j * .5)), math.sin(a) * 1.0]); dr_ /= np.linalg.norm(dr_); fan.append(np.stack([root + dr_ * ln * q + np.array([0, 0, .55 * (j - 3) / 3 * q]) for q in np.linspace(.0, 1, 6)]))
        fan = np.array(fan); fine = np.array([[fan[int(v)] * (1 - v % 1) + fan[min(m - 1, int(v) + 1)] * (v % 1) - (fan[int(v)] - root) * .13 * math.sin(math.pi * (v % 1)) * np.linspace(0, 1, 6)[:, None]] for v in np.linspace(0, m - 1.001, 25)])[:, 0]
        FW = D(W_(fine)); g.grid(FW, 'skfin', two=True, uv=np.stack(np.broadcast_arrays(np.linspace(0, 6, 25)[:, None] * .9, np.linspace(0, 4, 6)[None]), -1))
        for j in range(m): q = D(W_(fan[j])); g.cyl(q[0], q[-1], .17, .03, 'skback', 6, part=g.part(), caps=False)
        # barbels
        wv = np.asarray(whisk, float); p = np.array([8.0, sg * 1.5, .2]); Pw = [W_(p)]; dv_ = R @ np.array([.55, sg * .8, .1])
        for j in range(16): dv_ = dv_ + np.array([0, 0, -.14]) + wv * .05 + R @ np.array([0, sg * .10 * math.sin(t * 7 + j * .6), .12 * math.sin(t * 6 + j * .55 + sg)]); dv_ /= np.linalg.norm(dv_); Pw.append(Pw[-1] + dv_ * .55)
        Pw = D(np.array(Pw)); rw = np.linspace(.20, .04, 17)
        for j in range(0, 16, 2): g.cyl(Pw[j], Pw[j + 2], rw[j], rw[j + 2], 'skfin', 7, part=10 ** 5 if False else g.part(), caps=False)
    # scars over the left eye
    for k, (a, b) in enumerate((((.6, 2.1, 3.9), (4.6, 3.1, .3)), ((1.6, 1.7, 3.9), (5.2, 2.9, .4)))):
        q = np.linspace(0, 1, 12)[:, None]; L = np.array(a) * (1 - q) + np.array(b) * q; L[:, 1] = _ip(L[:, 0], HS, WD) * (1.02 - .30 * np.clip((L[:, 2] - 1.4) / 2.4, 0, 1)); Lw = D(W_(L)); nrm = R @ np.array([0, .8, .5]); wd2 = .20 * np.sin(np.linspace(.2, 2.95, 12)); side = R @ np.array([.5, 0, .35]); g.grid(np.stack([Lw + nrm * .10 + side * wd2[:, None], Lw + nrm * .14 - side * wd2[:, None]], 1), 'pale', two=True)
    return g
