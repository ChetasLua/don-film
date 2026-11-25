"""GEAR 5 (f1, f2, f4; small elsewhere): flame-cloud hair as a living fluid, head, rubber-hose figure, cloud scarf, bleach.
  head(cx,cy,s,yaw,pitch,roll,expr,light,f,**kw) -> layer            s = px chin to eye line
  head_parts(...) -> dict(back, face, front, H, glow)                 same, separate layers (glow = additive burn line while bleaching)
  figure(cx,cy,U,pose='dance',f=0,light='night',lag=None,**over) -> layer   (cx,cy) = ground point, U = px per head height (figure is ~5.2 U tall)
  POSES: dance rise squat leap fall reach punch      bleach(f,f0,f1) -> callable for head(bl=...)
Hair, scarf and detached licks are pure functions of f; pass lagT = (K,2) px displacement of the head k frames ago minus now, for follow-through."""
import numpy as np, cv2, math, cairo
from op import W, H as HH
from op.paint import rgb, f32, smooth, ramp, grid, fbm
from op.ink import Cel, spl, resample, ribbon, xf, paste
from op.kit import para, hatch, outline, shape, contours, shifted
from op.comp import blur, radial
from op.fx_core import h1, lin01, noise, thin, c2
from op.head3 import Head

PAL = dict(
    fire=dict(h0='#8a4c48', h1='#cf968a', h2='#f3dccf', h3='#fff8ec', hl='#5c2c26', sk='#eab07e', s1='#ba7049', s2='#763a2b', cl='#ecd2c4', c1='#b98479', c2='#8a5553', ink='#2a1211', wh='#fbeade', rim='#ffe9b0', blush='#b5503a', sash='#e8b25a', sash1='#b0703a', sole='#a8733e'),
    sun=dict(h0='#8f6fae', h1='#bd9bcd', h2='#eadbee', h3='#fffcf4', hl='#6b4468', sc2='#f0b8d2', sc3='#ffe3ee', sk='#fac69c', s1='#d08a76', s2='#8c4c5a', cl='#f8f2f6', c1='#c6b1da', c2='#9c84be', ink='#281626', wh='#fffaf4', rim='#fff0b8', blush='#e0705a', sash='#f2c45c', sash1='#c9843c', sole='#b98546'),
    night=dict(h0='#8088c0', h1='#b4bbe4', h2='#e6e9f9', h3='#ffffff', hl='#50527f', sc2='#cdb4e6', sc3='#efe0fa', sk='#f8c49a', s1='#c98878', s2='#84506a', cl='#f5f5fc', c1='#b4bbe2', c2='#8087bb', ink='#17131a', wh='#ffffff', rim='#ffe6a6', blush='#de6f62', sash='#f0c860', sash1='#bf8740', sole='#b18148'),
    day=dict(h0='#b59a96', h1='#e2c9bd', h2='#faf1ea', h3='#ffffff', hl='#8a6258', sk='#f0b98d', s1='#c9805c', s2='#8d5038', cl='#fbf7f2', c1='#d3c2bd', c2='#a8928f', ink='#17131a', wh='#ffffff', rim='#fff3c8', blush='#e27d64', sash='#f2c860', sash1='#c08a3c', sole='#b98a50'))
BLACK = [(0, '#070509'), (.5, '#120d13'), (.8, '#2c2026'), (1., '#574440')]


def ss(a, b, x):
    t = np.clip((np.asarray(x, float) - a) / (b - a), 0, 1); return t * t * (3 - 2 * t)


def back_out(t, k=1.6):
    t = np.clip(t, 0, 1) - 1; return 1 + (k + 1) * t ** 3 + k * t * t


def lay(a, b):
    return a * (1 - b[..., 3:4]) + b


def rim(layer, dx, dy, col, amt=1.0, soft=.6, where=None):
    """kit.rim with a replicated frame edge (cels that run off frame get no rim along the border)"""
    a = layer[..., 3]; h, w = a.shape; b = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = np.clip(a - b, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy()
    out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt; return out


def hotrim(L, dx, dy, w, hot='#fff3cf', warm='#ffb45a', amt=1., soft=1.):
    """near-white edge with a wide soft inner falloff; light comes from (-dx,-dy)"""
    L = rim(L, dx * w * 3.2, dy * w * 3.2, warm, .50 * amt, soft=w * 1.6 * soft); return rim(L, dx * w, dy * w, hot, .98 * amt, soft=w * .22)


def bleach(f, f0=14, f1=34):
    """per-lock bleach 0..1 for frame f: the nape starts, the crown follows, the fringe goes last"""
    def b(tier, i):
        st = dict(back=0., top=.22, front=.46)[tier] + .16 * h1(i, 71)
        return float(np.clip(((f - f0) / (f1 - f0) - st) / .40, 0, 1))
    return b


# ------------------------------------------------------------------------------------------------ head construction
G5 = dict(a=[(-1.0, .20), (-.96, .32), (-.88, .47), (-.74, .605), (-.56, .70), (-.36, .755), (-.12, .78), (.12, .785), (.35, .775), (.6, .73), (.8, .58), (.93, .35), (1.0, .05)],
          f=[(-1.0, .60), (-.92, .69), (-.8, .72), (-.64, .745), (-.46, .75), (-.28, .74), (-.06, .73), (.06, .74), (.2, .79), (.4, .79), (.6, .72), (.8, .55), (.93, .32), (1.0, .05)],
          nose=(.10, -.12, -.37, -.425, .125), cheek=.035, socket=.02, ear=(.775, .10, -.36, -.10))


class G5Head(Head):
    M = None; jawd = 0.; puff = 0.; chinw = 0.

    def P3(self, Q):
        Q = np.asarray(Q, float)
        if self.jawd or self.puff or self.chinw:
            Q = Q.copy(); y = Q[..., 1].copy(); v = np.clip((-.42 - y) / .58, 0, 1.2)
            Q[..., 1] = y - self.jawd * (ss(0, 1, v) if self.jawd < .6 else v ** 1.15)
            if self.chinw: Q[..., 0] = Q[..., 0] * (1 + self.chinw * ss(.45, 1., v) - .25 * min(1, self.jawd / 2.) * np.sin(np.clip(v, 0, 1) * math.pi))
            Q[..., 0] = Q[..., 0] * (1 + self.puff * np.exp(-((y + .52) / .24) ** 2))
        p = super().P3(Q)
        if self.M is not None:
            c = np.array([self.cx, self.cy]); p = (p - c) @ np.asarray(self.M).T + c
        return p


def root_on(H, th, phi, lift=.03):
    A = np.interp(phi, H.yy, H.A); B = np.interp(phi, H.yy, H.F if math.cos(th) >= 0 else H.B); p = H.k['p']
    sx, cz = math.sin(th), math.cos(th); r = (abs(sx) ** p + abs(cz) ** p) ** (-1 / p)
    root = np.array([A * sx * r, phi, B * cz * r]); n = np.array([sx / max(A, .1), (phi - .1) * .9, cz / max(B, .1)]); n /= np.linalg.norm(n)
    return root + n * lift, n


# ------------------------------------------------------------------------------------------------ one lock of flame-cloud
def lock(root, th, L, w, t, seed, wave=.42, k=.85, hz=.85, bend=0., curl=2.2, cu=.5, shed=False, lagT=None, n=36, r0=.10, belly=.40, tip=.85, dens=None, uf=1.1, drag=5., q=2.4, kids=0):
    """theta-integrated spine with a wave that travels root -> tip, curled end, swelling width, pinched-off tip that flies free.
    -> list of dicts(P polygon, a, b edges, c spine, w, seed, uf)"""
    u = lin01(n); ph, p2, p3 = [h1(seed, j) * 6.283 for j in (1, 2, 3)]; th = np.broadcast_to(np.asarray(th, float), (n,))
    dens = np.ones(n) if dens is None else dens
    un = .64; tau = 0.; Leff = L
    if shed:
        T = 1.0 + .7 * h1(seed, 4); tau = (t / T + h1(seed, 5)) % 1.
        if tau >= .42: Leff = L * (un + (1 - un) * float(ss(.42, 1., tau)))

    def body(L_):
        c = curl * (.86 + .26 * math.sin(6.283 * hz * .55 * t + p2))
        a = th + bend * u + wave * np.sqrt(u) * np.sin(6.283 * (k * u - hz * t) + ph) + c * np.clip((u - cu) / (1 - cu), 0, 1) ** 1.35
        ds = L_ / (n - 1) * dens
        P = c2(root[0] + np.r_[0, np.cumsum(np.cos(a[:-1]) * ds[:-1])], root[1] + np.r_[0, np.cumsum(np.sin(a[:-1]) * ds[:-1])])
        if lagT is not None and len(lagT) > 1:
            kk = np.arange(len(lagT)); g = u * drag
            P = P + np.stack([np.interp(g, kk, lagT[:, 0]), np.interp(g, kk, lagT[:, 1])], 1) * (u ** .7)[:, None]
        da = np.abs(np.gradient(a)) / np.maximum(ds, 1e-3)
        hw = w * (r0 + (1 - r0) * ss(0, belly, u)) * np.maximum(1 - u ** q, 0) ** tip * (1 + .13 * np.sin(6.283 * (1.25 * u - 1.15 * hz * t) + p3))
        hw = np.minimum(hw, .92 / np.maximum(da, 1e-4))
        ker = np.array([1, 2, 3, 2, 1.]); ker /= ker.sum(); hw = np.convolve(np.r_[hw[0], hw[0], hw, 0, 0], ker, 'valid'); hw[-1] = 0
        return P, hw

    def mk(P, hw, sd):
        Tn = np.gradient(P, axis=0); Tn /= np.hypot(Tn[:, 0], Tn[:, 1])[:, None] + 1e-9; N = c2(-Tn[:, 1], Tn[:, 0])
        a, b = P + N * hw[:, None], P - N * hw[:, None]
        return dict(P=np.vstack([a, b[::-1]]), a=a, b=b, c=P, hw=hw, N=N, w=w, seed=sd, uf=uf, u=u)
    out = []
    P, hw = body(Leff)
    if shed and tau < .42: hw = hw * (1 - .96 * float(ss(.03, .42, tau)) * np.exp(-((u - un) / .075) ** 2))
    out.append(mk(P, hw, seed))
    if kids:
        N = out[0]['N']; ang = np.unwrap(np.arctan2(np.gradient(P[:, 1]), np.gradient(P[:, 0])))
        for j in range(kids):
            ku = (.40, .58, .74, .50)[j] + .05 * (h1(seed, 30 + j) - .5); sg = 1 if (j + int(h1(seed, 33) * 2)) % 2 else -1; i = int(ku * n); sz = 1.
            br = 1 + .10 * math.sin(6.283 * (hz * .8 * t + h1(seed, 40 + j)))
            K = lock(P[i] + N[i] * hw[i] * .30 * sg, ang[i] + sg * (.50 + .25 * h1(seed, 43 + j)), w * (2.5, 2.1, 1.8, 2.)[j] * (.8 + .4 * h1(seed, 36 + j)) * br, w * (.56, .47, .40, .4)[j], t, seed * 7 + 61 + j, wave=wave * .7, k=.7, hz=hz, bend=-sg * .55, curl=sg * (1.9 + .9 * h1(seed, 46 + j)), cu=.45, lagT=None if lagT is None else np.asarray(lagT) * .35, n=22, r0=.75, belly=.15, tip=.8, uf=2. if uf > .5 else -1, shed=shed and j == 1)
            for q_ in K: q_['kid'] = True
            out += K
    if shed and .42 <= tau < .96:
        kk = (tau - .42) / .54; P, hw = body(L); m = max(6, int(n * (1 - un))); v = lin01(n)
        uu = un + (1 - un) * v; C = c2(np.interp(uu, u, P[:, 0]), np.interp(uu, u, P[:, 1])); h2 = np.interp(uu, u, hw) * ss(0, .22, v) * 1.15
        d = C[min(8, n - 1)] - C[0]; d = d / (np.hypot(*d) + 1e-6); c0 = C.mean(0); rt = (1 if curl > 0 else -1) * kk * 1.3; cs, sn = math.cos(rt), math.sin(rt); sc = (1 - kk) ** .75
        Q = (C - c0) * sc; Q = c2(Q[:, 0] * cs - Q[:, 1] * sn, Q[:, 0] * sn + Q[:, 1] * cs) + c0 + d * L * .42 * (1 - (1 - kk) ** 1.7) + np.array([0, -L * .10 * kk])
        out.append(mk(Q, h2 * sc, seed + 500))
    return out


SPEC = dict(                                                           # azimuth, root height, length, half width, curl, licks: few big masses, not many fingers
    back=[(180, .60, 2.7, .62, 1.5, 2), (-128, .86, 2.4, .54, -1.8, 2), (-150, .18, 2.1, .56, 1.9, 2), (150, .22, 1.85, .52, -1.9, 2), (-112, -.18, 1.35, .36, -2.2, 1), (112, -.18, 1.30, .36, 2.2, 1)],
    top=[(-86, .36, 1.25, .36, -2.2, 1), (86, .36, 1.05, .34, 2.2, 1), (150, .95, 1.75, .48, 2.0, 1), (62, .74, 1.50, .46, 2.3, 2), (-58, .72, 1.95, .52, 2.0, 2), (-8, .93, 2.3, .56, -1.8, 2)],
    front=[(-76, .16, .46, .13, -2.4, 0), (76, .16, .46, .13, 2.4, 0), (-60, .60, 1.05, .27, -2.2, 1), (60, .60, .95, .26, 2.3, 1), (-34, .74, 1.35, .33, 2.1, 1), (34, .745, 1.15, .31, 2.3, 1), (2, .775, 1.45, .35, 2.0, 1)])


def locks(H, f, tier, flow=(0., -1.), fs=.55, bl=1., lagT=None, amp=1., seed=0, shed=True, size=1., hz=.85):
    t = f / 24.; s = H.s; out = []
    c = H.P3(np.array([[0, .05, 0.]]))[0]; flow = np.asarray(flow, float); sc = s if H.M is None else s * math.sqrt(abs(np.linalg.det(np.asarray(H.M))))
    it = []
    for i, (td, phi, L, w, curl, kd) in enumerate(SPEC[tier]):
        r3, n3 = root_on(H, math.radians(td), phi); it.append((H.depth(r3) if tier != 'front' else i, i, td, phi, L, w, curl, r3, kd))
    if tier != 'front': it.sort(key=lambda q: q[0])
    n = 36; u = lin01(n)
    for _, i, td, phi, L, w, curl, r3, kd in it:
        p = H.P3(r3[None])[0]; rad = p - c; rl = np.hypot(*rad); rad = rad / max(rl, 1e-6) * min(1., rl / (s * .55))
        fl = fs * (1. if rad[0] * flow[0] >= 0 or abs(flow[0]) < .2 else .62); hv = (1 - fl) * rad + fl * flow; hv = hv / (np.hypot(*hv) + 1e-6); sd0 = dict(back=100, top=200, front=300)[tier] + i
        dv = .42 * noise(np.array([td * .013 + seed]), phi * 2.1, t * .55, 7)[0] * amp + (.30 * (h1(sd0, 14) - .5)); hv = np.array([hv[0] * math.cos(dv) - hv[1] * math.sin(dv), hv[0] * math.sin(dv) + hv[1] * math.cos(dv)]); bnd = .75 * float(np.clip(rad[0] * flow[1] - rad[1] * flow[0], -1, 1)) * (.5 + .8 * h1(sd0, 15))
        sd = seed * 1000 + dict(back=100, top=200, front=300)[tier] + i; b = bl(tier, i) if callable(bl) else float(bl)
        if b >= .999:
            out += lock(p, math.atan2(hv[1], hv[0]), L * sc * size, w * sc * (.8 + .2 * size), t, sd, wave=.54 * amp, k=.95, bend=bnd, curl=curl, shed=shed and h1(sd, 6) < .62, lagT=lagT, hz=hz, kids=kd, r0=.12 if tier != 'back' else .5)
        else:
            g = dict(back=.75, top=.45, front=1.25)[tier]; limp = rad * dict(back=.7, top=.75, front=.30)[tier] + np.array([(h1(sd, 8) - .5) * .75, g]); limp = limp / np.hypot(*limp)
            bb = np.clip((b - .22) / .78, 0, 1); bu = back_out(np.clip(bb * 1.55 - .55 * u, 0, 1), 1.9)[:, None]; V = (1 - bu) * limp + bu * hv; mg = np.hypot(V[:, 0], V[:, 1])
            th = np.unwrap(np.arctan2(V[:, 1], V[:, 0])); Lb = dict(back=.75, top=.62, front=.78)[tier] * (.72 + .55 * h1(sd, 9)); wb = dict(back=.34, top=.27, front=.185)[tier] * (.75 + .5 * h1(sd, 10))
            m = float(ss(0, 1, bb)); pf = 1 + .08 * math.sin(math.pi * bb); m5 = m ** .5
            out += lock(p, th, (Lb + (L * size - Lb) * m) * sc, (wb + (w - wb) * m) * sc * pf, t, sd, wave=.06 + .48 * m5 * amp, curl=curl * (.12 + .88 * m), cu=.5, lagT=lagT, dens=np.clip(mg, .32, 1.4), r0=.80 - .6 * m5, belly=.12 + .26 * m, tip=1.15 - .3 * m5, q=1.6 + .8 * m, kids=kd if b > .6 else 0, uf=float(np.clip(b * 1.75 - .12, 0, 1.1)) if b > .02 else -1, hz=hz)
    return out


LOOSE = [(-80, .20, 1.15, .085, -1), (-50, .54, .62, .062, -1), (-20, .65, .96, .078, 1), (14, .67, .50, .055, 1), (37, .60, .80, .070, 1), (74, .28, 1.28, .088, 1)]
LICK = [(-100, .18, .80, .17), (-82, .46, 1.05, .18), (-60, .78, .95, .16), (-32, .92, .85, .15), (4, .96, 1.0, .16), (31, .91, .85, .15), (61, .76, 1.0, .16), (82, .46, 1.05, .18), (100, .18, .80, .17)]


def loose(H, f, m, lagT, flow, seed=0, hz=.85, amp=1.):
    """half-bleached hair (m 0..1 of the fringe): licks that break the crown's edge, and forelocks that still fall over the brow, tips burning last"""
    wl, wf = float(ss(.0, .30, m) * ss(.95, .70, m)), float(ss(-.04, .26, m) * ss(.96, .66, m)); out = []; fo = []
    if max(wl, wf) < .15: return out, fo
    t = f / 24.; u = lin01(36); c = H.P3(np.array([[0, .05, 0.]]))[0]; sc = H.s if H.M is None else H.s * math.sqrt(abs(np.linalg.det(np.asarray(H.M)))); fl = np.asarray(flow, float); fl = fl / (np.hypot(*fl) + 1e-6)
    for i, (td, phi, L, w) in enumerate(LICK):
        p = H.P3(root_on(H, math.radians(td), phi)[0][None])[0]; rad = p - c; rad = rad / (np.hypot(*rad) + 1e-6); hv = .66 * rad + .34 * fl; sd = seed * 1000 + 700 + i
        if wl > .15: out += lock(p, math.atan2(hv[1], hv[0]) + .20 * math.sin(t * 7 + i * 1.9), L * sc * (.35 + .65 * wl), w * sc * (.25 + .75 * wl), t, sd, wave=.46 * amp, k=.95, curl=(2. if i % 2 else -2.) * (1 if td > 0 else -1), lagT=lagT, hz=hz, r0=.14)
    for i, (td, phi, L, w, g) in enumerate(LOOSE):
        p = H.P3(root_on(H, math.radians(td), phi)[0][None])[0]; sd = seed * 1000 + 800 + i; th = -math.pi / 2 + g * (.70 + 2.30 * float(ss(1.05, .62, m)) * ss(0, .30, u) + .12 * math.sin(t * 6 + i * 2.1) * u)
        if wf > .15: fo += lock(p, th, L * sc * wf, w * sc * (.2 + .8 * wf), t, sd, wave=.10 * amp, k=.8, curl=-g * .75, cu=.55, lagT=lagT, hz=hz, r0=.38, belly=.18, tip=1.1, q=1.7, uf=float(np.clip(.25 + m * 2.4 - .25 * h1(sd, 3), .03, 1.1)))
    return out, fo


def paint_hair(L, pal, s, cap=(), lw=2., light=(-.45, -.85), close=.055, dim=1., seed=0, glow=None, inner=True, wh=HH, ww=W, sil=1., cut=None, after=()):
    """painter's hair: union mass with rounded crotches, thin brown contour, per-lock airbrush.  -> full-frame premultiplied layer.
    glow: pass a list to collect burn-front polylines"""
    out = np.zeros((wh, ww, 4), f32)
    if not L and not cap: return out
    A = np.vstack([k['P'] for k in L] + [np.asarray(c) for c in cap]); pad = int(.10 * s + 10)
    x0, y0 = [int(v) for v in np.maximum(np.floor(A.min(0)) - pad, -26)]; x1 = int(min(ww + 26, A[:, 0].max() + pad)); y1 = int(min(wh + 26, A[:, 1].max() + pad))
    if x1 - x0 < 12 or y1 - y0 < 12: return out
    hw, hh = (x1 - x0) // 2 + 1, (y1 - y0) // 2 + 1; org = np.array([x0, y0], float)

    def strip(k, u1=2.):
        m = np.zeros((hh, hw), np.uint8); a, b = (k['a'] - org) * 8, (k['b'] - org) * 8
        e = len(a) if u1 > 1 else max(2, int(u1 * len(a)))
        for j in range(e - 1):
            cv2.fillConvexPoly(m, np.round(np.array([a[j], a[j + 1], b[j + 1], b[j]])).astype(np.int32), 255, cv2.LINE_8, 4)
        return m
    M = [strip(k) for k in L]; U = np.zeros((hh, hw), np.uint8)
    for m in M: U = np.maximum(U, m)
    CM = []
    for c in cap:
        m = np.zeros((hh, hw), np.uint8); cv2.fillPoly(m, [np.round((np.asarray(c) - org) * 8).astype(np.int32)], 255, cv2.LINE_8, 4); U = np.maximum(U, m); CM.append(m)
    r = int(close * s / 2)
    if r >= 1: U = cv2.morphologyEx(U, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    nc, lab, st, _ = cv2.connectedComponentsWithStats((U < 128).astype(np.uint8), connectivity=4)      # fill pinholes left by closing
    for j in range(1, nc):
        if st[j, 4] < (.11 * s) ** 2 * .8 and st[j, 0] > 0 and st[j, 1] > 0: U[lab == j] = 255
    Uf = cv2.GaussianBlur(U, (0, 0), 1.1).astype(f32) / 255.
    cs = [c * 2 + org for c in contours(cv2.resize(Uf, None, fx=1, fy=1), min_len=16)]
    l = np.asarray(light, float); l = l / np.hypot(*l)
    S = np.full((hh, hw), .60 * dim, f32); B = np.ones((hh, hw), f32); anyb = any(k['uf'] < 1.05 for k in L)
    if CM and L and L[0]['uf'] < 1.05:
        for m in CM: B[m > 0] = 1 if min(k['uf'] for k in L) > .15 else 0
    if anyb and not CM: B[:] = 1 if min(k['uf'] for k in L) > .2 else 0
    rb = max(1.2, .075 * s)
    for k, m8 in zip(L, M):                                       # airbrush, back to front
        ys, xs = np.where(m8.any(1))[0], np.where(m8.any(0))[0]
        if len(ys) == 0: continue
        pd = int(rb * 3 + 4); Y0, Y1, X0, X1 = max(0, ys[0] - pd), min(hh, ys[-1] + pd), max(0, xs[0] - pd), min(hw, xs[-1] + pd)
        m = cv2.GaussianBlur(m8[Y0:Y1, X0:X1], (0, 0), .8).astype(f32) / 255.; sub = S[Y0:Y1, X0:X1]
        b = cv2.GaussianBlur(shifted(m, -l[0] * rb * .55, -l[1] * rb * .55), (0, 0), rb)
        d = max(1., k['w'] * .30); m2 = cv2.GaussianBlur(shifted(m, l[0] * d, l[1] * d), (0, 0), max(1., d * .55))
        C = (k['c'] - org) / 2; dv = C[len(C) // 2] - C[0]; n2 = float(dv @ dv) + 1e-6
        yy, xx = np.mgrid[Y0:Y1, X0:X1].astype(f32); g = np.clip(((xx - C[0, 0]) * dv[0] + (yy - C[0, 1]) * dv[1]) / n2, 0, 1)
        kdd = k.get('kid'); own = dim * (1.08 - (.12 if kdd else .46) * (1 - g) ** 1.15 - (.34 if kdd else .44) * np.clip(1 - m2, 0, 1)); b = b * (.40 if kdd else 1.)
        fd = ss(.0, .34, g) if not k.get('kid') else ss(-.05, .45, g); b = b * (.15 + .85 * fd); mf = (m * fd).astype(f32); sub *= 1 - .58 * b.astype(f32); sub *= 1 - mf; sub += (own * mf).astype(f32)
        if anyb:
            mw = cv2.GaussianBlur(strip(k, k['uf'])[Y0:Y1, X0:X1], (0, 0), .9).astype(f32) / 255. if k['uf'] > 0 else 0 * m
            bb = B[Y0:Y1, X0:X1]; bb *= 1 - m; bb += np.minimum(mw, m)
    S = cv2.GaussianBlur(S, (0, 0), .7)
    col = ramp(np.clip(S, 0, 1), [(0, pal['h0']), (.40, pal['h1']), (.74, pal['h2']), (.98, pal['h3'])]); line = np.broadcast_to(rgb(pal['hl']), col.shape)
    if anyb:
        col = col * B[..., None] + ramp(np.clip(S, 0, 1), BLACK) * (1 - B[..., None]); line = line * B[..., None] + rgb('#050306') * (1 - B[..., None])
    cel = Cel(x1 - x0, y1 - y0, x0, y0)
    if cs: cel.fill(cs, '#ffffff', evenodd=True)
    for k in L:
        sd = k['seed']; cel.fill(k['P'], '#ffffff'); n = len(k['a']); mid = int(n * .45); lit = float(k['N'][mid] @ l) > 0
        shd, hi = (k['b'], k['a']) if lit else (k['a'], k['b'])
        if not inner: continue
        kd_ = k.get('kid')
        j0 = int(n * ((.34 if not kd_ else .40) + .16 * h1(sd, 21))); cel.line(shd[j0:], lw * .92, '#000000', taper=(.55, .12), press=.45, seed=sd, tremor=.15, tmin=.05)
        j1 = int(n * (.62 + .18 * h1(sd, 22)))
        if n - j1 > 5: cel.line(hi[j1:], lw * .62, '#000000', taper=(.5, .15), press=.45, seed=sd + 1, tremor=.15, tmin=.05)
        if h1(sd, 23) < .80 and k['w'] > 14 and not kd_:
            sg = 1 if h1(sd, 24) < .5 else -1; a0, a1 = int(n * (.14 + .16 * h1(sd, 25))), int(n * (.64 + .24 * h1(sd, 26)))
            cel.line(k['c'][a0:a1] + k['N'][a0:a1] * (k['hw'][a0:a1] * sg * (.18 + .3 * h1(sd, 27)))[:, None], lw * .55, '#000000', taper=(.45, .5), press=.4, seed=sd + 2, tremor=.1, tmin=.05)
        if h1(sd, 28) < .5 and k['w'] > 20:                        # a small filled accent sliver beside the fold
            a0 = int(n * (.28 + .2 * h1(sd, 29))); a1 = a0 + int(n * .13); Q = k['c'][a0:a1] - k['N'][a0:a1] * (k['hw'][a0:a1] * .45)[:, None]; cel.line(Q, lw * 1.5, '#5a5a5a', taper=(.5, .5), press=.1, seed=sd + 3, tremor=0, tmin=.02)
        if glow is not None and .02 < k['uf'] < 1.0:
            e = min(n - 2, max(1, int(k['uf'] * n))); a, b, c = k['a'][e], k['b'][e], k['c']; jag = [a]
            for v in (.2, .4, .6, .8): jag.append(a + (b - a) * v + (c[min(n - 1, e + (3 if int(v * 5) % 2 else 1))] - c[e]))
            glow.append((np.array(jag + [b]), k['w']))
    for i, c in enumerate(cs): outline(cel, c, lw * 1.08 * sil, '#000000', seed + i * 7, breaks=max(0, int(len(c) / 260)), press=.5, tremor=.12)
    if cut is not None:                                            # designed hairline: clear the forehead, ink the edge, forelocks fall over it
        cx = cel.ctx; cx.save(); cx.set_operator(cairo.OPERATOR_CLEAR); cel._path(cut[0]); cx.fill(); cx.restore()
        cel.fill(cut[2], '#ffffff'); outline(cel, cut[1], lw * 1.15, '#000000', seed + 77, closed=False, press=.5)
    for k in after:
        inked(cel, k['P'], '#ffffff', lw * .62, '#000000', heavy=(0, 0)); n = len(k['c']); cel.line(k['c'][int(n * .2):int(n * .8)], lw * .5, '#000000', taper=(.4, .5), seed=k['seed'])
    lay_ = cel.rgba(); a = lay_[..., 3:4]; K = np.where(a > 1e-3, lay_[..., :1] / np.maximum(a, 1e-3), 1.)
    C = cv2.resize(col, (hw * 2, hh * 2), interpolation=cv2.INTER_LINEAR)[:y1 - y0, :x1 - x0]; Ln = cv2.resize(np.ascontiguousarray(line), (hw * 2, hh * 2), interpolation=cv2.INTER_LINEAR)[:y1 - y0, :x1 - x0]
    res = np.dstack([(Ln + (C - Ln) * K) * a, a]).astype(f32); paste(out, res, x0, y0); return out


# ------------------------------------------------------------------------------------------------ face
EX = dict(
    calm=dict(eye=1., mouth='calm', brow=0., tilt=0.),
    shut=dict(eye=0., mouth='line', smile=0., brow=-.02, tilt=0.),
    wide=dict(eye=1.18, mouth='line', smile=.02, brow=.06, tilt=0., iris=.9),
    grin=dict(eye=1.04, mouth='grin', gw=.56, gc=-.37, gt=-.53, gb=-.83, brow=.05, tilt=-.03, puff=.06),
    joy=dict(eye=1.12, mouth='laugh', brow=.10, tilt=0., jaw=.14, iris=.85),
    laugh=dict(eye=-1, mouth='laugh', brow=.08, tilt=0., jaw=.20),
    roar=dict(eye=1.22, mouth='laugh', mw=.36, mc=-.42, mt=-.47, brow=.16, tilt=0., jaw=.50, iris=.55),
    arcs=dict(eye=-1, mouth='laugh', mw=.50, mc=-.40, mt=-.47, brow=.14, tilt=0., jaw=.34, tongue=1, puff=.08),
    wink=dict(eyeA=-1, eyeB=1.2, eye=1.2, mouth='laugh', mw=.46, skew=1., brow=.12, tilt=-.02, jaw=.24, iris=.8),
    cres=dict(eye=1.12, cheek=.86, mouth='laugh', mw=.50, mc=-.40, mt=-.46, brow=.13, tilt=0., jaw=.38, puff=.08),
    crazy=dict(eye=1.30, cheek=.15, iris=.55, mouth='laugh', mw=.50, mc=-.40, mt=-.46, brow=.18, tilt=-.04, jaw=.46, tongue=1, puff=.08),
    teeth=dict(eye=-1, mouth='grin', gw=.60, gc=-.35, gt=-.53, gb=-.86, brow=.10, tilt=0., puff=.08),
    shock=dict(eye=1.35, mouth='laugh', brow=.18, tilt=.03, jaw=.5, iris=.55),
    fierce=dict(eye=.86, mouth='grin', gw=.50, gc=-.50, gt=-.62, gb=-.86, brow=-.05, tilt=.09))


def spiral(c, r, turns, a0, sg, n=40):
    v = lin01(n); a = a0 + sg * v * turns * 6.283; rr = r * (1 - .86 * v); return c2(c[0] + np.cos(a) * rr, c[1] + np.sin(a) * rr)


def _face(H, pal, E, s, lw, f=0, shade_side=-1, spir=1., wb=1., brim=None, socket=.55, hatching=True):
    """one face cel -> full-frame layer.  wb = white brows 0..1, spir = how far the brow spiral is rolled, brim = front-y the shadow of the hair reaches down to"""
    ink = pal['ink']; P = H.face(.8); lo, hi = P.min(0) - s * .42, P.max(0) + s * .42
    x0, y0 = int(max(-20, lo[0])), int(max(-20, lo[1])); x1, y1 = int(min(W + 20, hi[0])), int(min(HH + 20, hi[1]))
    out = np.zeros((HH, W, 4), f32)
    if x1 - x0 < 8 or y1 - y0 < 8: return out
    cel = Cel(x1 - x0, y1 - y0, x0, y0); Q = H.P; yaw = H.yaw; big = s > 150; near = -1 if yaw >= 0 else 1

    def ear(sd):
        e = H.ear(sd)
        if e['vis'] < -.15: return
        shape(cel, e['outer'], pal['sk'], lw, ink, 11); cel.shade(e['inner'], pal['s1'], e['outer']); cel.shade(e['tragus'], pal['s2'], e['outer'])
        cel.line(e['inner'], lw * .6, ink, taper=(.2, .4), seed=3)
        if big: cel.line(e['tragus'], lw * .45, ink, taper=(.3, .3), seed=4)
    ear(-near)
    if abs(yaw) < .2: ear(near)
    cel.fill(P, pal['sk']); cel.clip(P)
    if shade_side:                                                   # ONE hard side plane
        g = shade_side; sh = spl([(g * .98, .70), (g * .60, .50), (g * .52, .26), (g * .58, .06), (g * .60, -.14), (g * .49, -.30), (g * .455, -.46), (g * .40, -.66), (g * .27, -.86), (g * .10, -1.03), (g * .98, -1.12)], n=8)
        cel.fill(Q(sh), pal['s1'])
    cel.fill(Q(spl([(-.20, -.80), (-.05, -.785), (.14, -.80), (.10, -.835), (-.06, -.83)], True, n=6)), pal['s1'])
    tip = H.k['nose'][2]; zt = float(H.z(0, tip)) + H.k['nose'][4]; nx = .5 * np.sign(yaw if abs(yaw) > .04 else 1) * .1
    cel.fill(H.P3(np.array([(nx, tip + .0, zt), (-nx * .4, tip + .06, zt - .07), (-nx * 1.3, tip - .03, zt - .09), (-nx * .2, tip - .06, float(H.z(0, tip - .06)))])), pal['s1'])
    hl = [(-.80, .05), (-.69, .32), (-.50, .53), (-.25, .635), (0, .66), (.25, .635), (.50, .53), (.69, .32), (.80, .05)]
    if brim is None:
        lowr = [(.80, -.12), (.63, .10), (.48, .29), (.25, .41), (.06, .455), (-.20, .40), (-.46, .27), (-.61, .07), (-.80, -.16)]; cel.fill(Q(np.vstack([spl(hl, n=8), spl(lowr, n=8)])), pal['s1'])
    else:
        b = brim; lowr = [(.85, b - .05), (.58, b + .06), (.42, b - .06), (.22, b + .05), (.05, b - .09), (-.16, b + .04), (-.36, b - .06), (-.60, b + .06), (-.85, b - .06)]
        cel.fill(Q(np.vstack([[(-.95, 1.)], spl(hl, n=8), [(.95, 1.)], spl(lowr, n=8)])), pal['s2'], .80)
    cel.unclip()
    outline(cel, P, lw * 1.25, ink, 5, press=.45, breaks=1)
    if hatching and big:
        for sd, (hx, hy) in ((near, (.46, -.50)), (-near, (.40, -.46))):
            for j in range(6 if sd == near else 4):
                a = Q([(sd * (hx - .02 + j * .024), hy + .085 - j * .052), (sd * (hx + .12 + j * .016), hy - .0 - j * .052)]); cel.line(a, lw * .42, ink, taper=(.25, .45), seed=j + 40, press=.2)
        for j in range(5): cel.line(Q([(-.34 + j * .055 - .12, .475 + .012 * j), (-.24 + j * .05, .41 - .015 * (j % 3))]), lw * .4, ink, taper=(.2, .6), seed=j + 60)
    # nose
    q = H.nose_profile(); off = np.array([nx * .25, 0, 0]); L = H.P3(q[2:5] + off); cel.line(spl(L, n=8), lw * .72, ink, taper=(.75, .12), seed=7)
    cel.line(spl(H.P3(np.array([(nx * .9, tip, zt), (nx * .2, tip - .045, zt - .03), (-nx * 1.1, tip - .03, zt - .085)])), n=6), lw * .8, ink, taper=(.2, .5), seed=8)
    # scar under his left eye
    cel.line(Q(spl([(.20, -.292), (.33, -.312), (.50, -.275)], n=6)), lw * .55, ink, taper=(.3, .3), seed=9)
    for xx in (.30, .41): cel.line(Q([(xx - .012, -.262), (xx + .012, -.345)]), lw * .5, ink, taper=(.3, .3), seed=10)
    # eyes
    eo = E.get('eye', 1.); gz = E.get('gaze', (0., 0.)); ir = E.get('iris', 1.); pop = E.get('ring', 0.)
    for g in (-1, 1):
        eo = E.get('eyeA' if g < 0 else 'eyeB', E.get('eye', 1.)); ck = E.get('cheek', 0.); ex, ey = g * .350, -.03; n = 48; a = np.linspace(0, 2 * math.pi, n, endpoint=False)
        if eo < 0:                                                   # squeezed shut with laughter
            k = -eo; A = spl([(ex - .20, ey - .07), (ex - .10, ey + .045 * k), (ex, ey + .075 * k), (ex + .10, ey + .045 * k), (ex + .20, ey - .07)], n=8); cel.line(Q(A), lw * 1.9, ink, taper=(.22, .22), seed=12 + g)
            for j in (0, 1): cel.line(Q([(ex + g * (.21 + .02 * j), ey - .04 + .06 * j), (ex + g * (.29 + .02 * j), ey - .02 + .085 * j)]), lw * .7, ink, taper=(.1, .5), seed=j)
            continue
        if eo < .14:
            A = spl([(ex - g * -.20, ey - .03), (ex + g * .10, ey - .085), (ex - g * .05, ey - .095), (ex - g * .20, ey - .05)], n=8); cel.line(Q(A), lw * 1.7, ink, taper=(.2, .3), seed=12 + g)
            for j in range(3): cel.line(Q([(ex + g * (.06 + .05 * j), ey - .088 + .012 * j), (ex + g * (.075 + .06 * j), ey - .125 + .016 * j)]), lw * .6, ink, taper=(.1, .5), seed=j)
            if big: cel.line(Q(spl([(ex - .16, ey + .07), (ex, ey + .085), (ex + .16, ey + .05)], n=6)), lw * .45, ink, taper=(.4, .4), seed=18 + g)
            continue
        ru, rl_ = .188 * eo ** .9, .172 * (.35 + .65 * eo); rx = .212 * (1 + .04 * (eo - 1)); rl_ = rl_ * (1 - 1.42 * ck); sy = np.where(np.sin(a) > 0, ru, rl_); sq = .90
        X = ex + rx * np.sign(np.cos(a)) * np.abs(np.cos(a)) ** sq; Y = ey + sy * np.sign(np.sin(a)) * np.abs(np.sin(a)) ** sq - (np.cos(a) * g < -.6) * .0
        Ev = np.stack([X, Y], -1); pe = Q(Ev); cel.fill(pe, pal['wh']); cel.clip(pe)
        cel.fill(Q(np.stack([X, np.maximum(Y, ey + ru * .62 - .05 * np.cos((X - ex) / rx * 1.2))], -1)), pal['c1'], .75)
        ic = (ex - g * .012 + gz[0] * .10, ey - .01 + gz[1] * .10); R = .092 * ir
        circ = lambda r_, dx=0., dy=0.: Q(np.stack([ic[0] + dx + r_ * np.cos(a), ic[1] + dy + r_ * np.sin(a) * 1.04], -1))
        cel.fill(circ(R * 1.12), ink); cel.fill(circ(R), '#d2564f'); cel.fill(circ(R * .80), '#f29a88'); cel.fill(circ(R * .62), '#8e1d22'); cel.fill(circ(R * .52), '#e4685c'); cel.fill(circ(R * .30), '#2a0608')
        if big: cel.fill(circ(R * .13, -R * .42, R * .46), '#fff6ea'); cel.fill(circ(R * .07, R * .40, -R * .42), '#ffd9c8')
        if pop > 0:                                                   # concentric rings closing in on the iris (the eyes open like a target)
            for j in range(4):
                rr = R * (1.3 + (j + 1) * .55 * pop); c = circ(rr); cel.line(np.vstack([c, c[:1]]), lw * (.9 - .12 * j) * pop, '#b8262a', taper=(0, 0), press=.3, seed=j, alpha=min(1., pop * 1.2))
        cel.unclip()
        o = np.argsort(a); ua = np.linspace(math.pi * 1.04, -math.pi * .04, 30); U = np.stack([ex + (-g) * -1 * 0 + rx * 1.02 * np.sign(np.cos(ua)) * np.abs(np.cos(ua)) ** sq, ey + ru * np.sign(np.sin(ua)) * np.abs(np.sin(ua)) ** sq], -1)
        if g < 0: U = U[::-1]
        cel.line(Q(U), lw * 1.9, ink, taper=(.16, .10), seed=14 + g, press=.2); la = np.linspace(math.pi * 1.02, math.pi * 1.98, 30); Lw = np.stack([ex + rx * np.cos(la), ey + rl_ * np.sin(la)], -1)
        cel.line(Q(Lw), lw * .95, ink, taper=(.18, .18), seed=16 + g, press=.4)
        if ck > .15:
            for j in range(2): yy = ey - rl_ - .045 - .05 * j; cel.line(Q(spl([(ex - .20 + .03 * j, yy - .05), (ex, yy + .035 * ck), (ex + .20 - .03 * j, yy - .05)], n=7)), lw * (.8 - .25 * j), ink, taper=(.3, .3), seed=50 + j)
        if big:
            cel.line(Q(spl([(ex - .18, ey + ru + .015), (ex - .02, ey + ru + .062), (ex + .17, ey + ru + .02)], n=7)), lw * .45, ink, taper=(.4, .4), seed=18 + g)
            cel.line(Q(spl([(ex - g * .10, ey - rl_ - .035), (ex + g * .06, ey - rl_ - .050), (ex + g * .20, ey - rl_ * .62 - .035)], n=7)), lw * .5, pal['s2'], taper=(.3, .5), seed=19 + g)
            cel.line(Q([(ex - g * .245, ey - .03), (ex - g * .275, ey - .055)]), lw * .6, pal['s2'], taper=(.1, .6), seed=20)
    # brows
    br, tl = E.get('brow', 0.), E.get('tilt', 0.)
    for g in (-1, 1):
        iny, outy = .290 + br - tl, .305 + br * .8 + tl * .6; th = .050 if wb > .5 else .032
        C = spl([(g * .635, outy - .04), (g * .51, outy + .045), (g * .35, (iny + outy) / 2 + .050), (g * .20, iny + .0)], n=10); m = len(C); v = np.linspace(0, 1, m)
        hw = th * (.75 + .25 * ss(0, .3, v)) * (1 - .12 * v) * (np.clip(v / .06, 0, 1) ** .5 if wb > .5 else np.sin(np.pi * np.clip(v * .9 + .08, 0, 1)) ** .5)
        T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = c2(-T[:, 1], T[:, 0]); poly = np.vstack([C + N * hw[:, None], (C - N * hw[:, None])[::-1]])
        fill = pal['h2'] if wb > .5 else '#0d090c'; li = pal['hl'] if wb > .5 else ink
        shape(cel, Q(poly), fill, lw * .62, li, 30 + g, press=.3)
        if wb > .5:
            cel.fill(Q(np.vstack([C - N * hw[:, None], (C - N * (hw * .25)[:, None])[::-1]]) if g > 0 else np.vstack([C + N * hw[:, None], (C + N * (hw * .25)[:, None])[::-1]])), pal['h1'], .9)
            if spir > .03:
                cc = np.array([g * (.20 - .025), iny + .006]); r = .072 * min(1, .4 + .6 * spir); bl = np.stack([cc[0] + r * np.cos(np.linspace(0, 6.283, 30)), cc[1] + r * np.sin(np.linspace(0, 6.283, 30))], -1)
                shape(cel, Q(bl), fill, lw * .62, li, 32 + g, press=.2); cel.line(Q(spiral(cc, r * .98, 1.55 * spir, math.pi * (1 if g > 0 else 0), g)), lw * .62, li, taper=(.05, .35), press=.2, seed=34, tremor=.05)
        if big and tl > .02: cel.line(Q(spl([(g * .085, .20), (g * .075, .28), (g * .10, .36)], n=6)), lw * .6, ink, taper=(.4, .4), seed=36 + g)
    _mouth(cel, H, pal, E, s, lw, ink, big)
    L = cel.rgba()
    if socket > 0:                                                    # airbrush round the eyes, under the chin and cheeks
        hgt, wid = L.shape[:2]; qx = 4; yy, xx = np.mgrid[0:hgt:qx, 0:wid:qx].astype(f32); m = np.zeros_like(xx)
        for g in (-1, 1):
            c = Q([(g * .34, -.06)])[0] - [x0, y0]; m += .9 * np.exp(-(((xx - c[0]) / (s * .33)) ** 2 + ((yy - c[1]) / (s * .30)) ** 2) ** 1.2)
            c = Q([(g * .48, -.42)])[0] - [x0, y0]; m += .30 * np.exp(-(((xx - c[0]) / (s * .22)) ** 2 + ((yy - c[1]) / (s * .20)) ** 2))
        c = Q([(0, -1.0)])[0] - [x0, y0]; m += .35 * np.exp(-(((xx - c[0]) / (s * .5)) ** 2 + ((yy - c[1]) / (s * .16)) ** 2))
        m = cv2.resize(m, (wid, hgt), interpolation=cv2.INTER_LINEAR)[..., None] * socket
        Lm = (L[..., :1] * .3 + L[..., 1:2] * .59 + L[..., 2:3] * .11); keep = smooth(.92, .75, L[..., 2:3] / np.maximum(L[..., 3:4], 1e-3))   # leave eye whites and teeth clean
        t = rgb(pal['blush']); L[..., :3] *= 1 - np.clip(m, 0, 1) * keep * (1 - (.45 + .55 * t)) * 1.15
    paste(out, L, x0, y0); return out


def _mouth(cel, H, pal, E, s, lw, ink, big):
    Q = H.P; kind = E.get('mouth', 'calm')
    if kind == 'calm':
        u = spl([(-.17, -.655), (-.05, -.632), (.06, -.638), (.17, -.655)], n=8); lo = spl([(.17, -.655), (.05, -.676), (-.06, -.678), (-.17, -.655)], n=8)
        cel.fill(Q(np.vstack([u, lo])), pal['wh']); cel.line(Q(u), lw * .8, ink, taper=(.2, .2), seed=70); cel.line(Q(lo), lw * .5, ink, taper=(.3, .3), seed=71); cel.line(Q(spl([(-.08, -.765), (.02, -.775), (.10, -.762)], n=5)), lw * .5, ink, taper=(.4, .4), seed=72)
    elif kind == 'line':
        k = E.get('smile', 0.); w = E.get('mw', .18); cel.line(Q(spl([(-w, -.655 + k), (-w * .4, -.668 - k * .2), (w * .4, -.668 - k * .2), (w, -.655 + k)], n=8)), lw * .95, ink, taper=(.15, .15), seed=70)
        cel.line(Q(spl([(-.07, -.765), (.02, -.773), (.09, -.763)], n=5)), lw * .45, ink, taper=(.4, .4), seed=72)
        if k > .03:
            for g in (-1, 1): cel.line(Q(spl([(g * (w - .005), -.655 + k - .03), (g * (w + .025), -.655 + k), (g * (w + .012), -.655 + k + .035)], n=5)), lw * .6, ink, taper=(.3, .4), seed=73)
    elif kind == 'grin':
        gw, gc, gt, gb = E.get('gw', .5), E.get('gc', -.46), E.get('gt', -.58), E.get('gb', -.86); n = 40; x = np.linspace(-1, 1, n)
        yt = gc + (gt - gc) * (1 - np.abs(x) ** 1.7); yb = gc + (gb - gc) * (1 - np.abs(x) ** 2.3) ** .85; X = x * gw
        top, bot = np.stack([X, yt], -1), np.stack([X, yb], -1); M = np.vstack([top, bot[::-1]]); pm = Q(M)
        cel.fill(pm, '#fff8ea'); cel.clip(pm)
        band = np.vstack([top, np.stack([X, yt - (yt - yb) * .16], -1)[::-1]]); cel.fill(Q(band), pal['c1'], .75)           # lip shadow on the teeth
        mid = np.stack([X, yt + (yb - yt) * .50], -1)
        for j, xx in enumerate(np.linspace(-.86, .86, 13)):
            i = np.interp(xx, x, np.arange(n)); a, b = int(i), min(n - 1, int(i) + 1); q = lambda A: A[a] + (A[b] - A[a]) * (i - a)
            cel.line(Q(spl([q(top), q(mid) + [xx * .012, 0], q(bot)], n=5)), lw * .42, ink, taper=(.12, .12), press=.3, seed=80 + j, alpha=.85)
        cel.line(Q(mid), lw * .62, ink, taper=(.06, .06), press=.2, seed=79); cel.unclip()
        cel.line(Q(top), lw * 1.35, ink, taper=(.10, .10), seed=77); cel.line(Q(bot), lw * 1.15, ink, taper=(.10, .10), seed=78)
        for g in (-1, 1):
            cel.line(Q(spl([(g * (gw - .01), gc - .055), (g * (gw + .035), gc - .005), (g * (gw + .035), gc + .06), (g * (gw - .01), gc + .11)], n=6)), lw * .95, ink, taper=(.25, .45), seed=81)
            if big:
                for j in range(3): cel.line(Q([(g * (gw + .05 + .03 * j), gc + .02 - .05 * j), (g * (gw + .11 + .03 * j), gc + .045 - .05 * j)]), lw * .42, ink, taper=(.2, .5), seed=84 + j)
        cel.line(Q(spl([(-.12, gb - .05), (0, gb - .062), (.12, gb - .05)], n=5)), lw * .5, ink, taper=(.4, .4), seed=72)
    elif kind == 'laugh':
        mw, yc, top, bt = E.get('mw', .43), E.get('mc', -.43), E.get('mt', -.50), E.get('mb', -.905); n = 36; x = np.linspace(-1, 1, n); X = x * mw
        yt = yc + (top - yc) * (1 - np.abs(x) ** 2.2); yb = yc + (bt - yc) * (1 - np.abs(x) ** 2.6) ** .62; sk = E.get('skew', 0.); X = X + sk * .10 * (1 - x * x); yt = yt + sk * .075 * x; yb = yb + sk * .075 * x; T, B = np.stack([X, yt], -1), np.stack([X, yb], -1); pm = Q(np.vstack([T, B[::-1]]))
        cel.fill(pm, '#4c0f1a'); cel.clip(pm)
        tg = spl([(-mw * .62, bt - .06), (-mw * .55, bt + .13), (-mw * .22, bt + .20), (0, bt + .165), (mw * .22, bt + .20), (mw * .55, bt + .13), (mw * .62, bt - .06)], n=8); cel.fill(Q(tg), '#d9605c'); cel.line(Q(tg), lw * .6, ink, taper=(.1, .1), seed=90); cel.line(Q([(0, bt + .16), (.0, bt + .05)]), lw * .45, '#8a2c34', taper=(.1, .6), seed=91)
        tb = np.vstack([T, np.stack([X, yt - .105 * (1 - np.abs(x) ** 4)], -1)[::-1]]); cel.fill(Q(tb), '#fff8ea'); cel.line(Q(np.stack([X, yt - .105 * (1 - np.abs(x) ** 4)], -1)), lw * .55, ink, taper=(.05, .05), seed=92)
        lb = np.vstack([B, np.stack([X, yb + .050 * (1 - np.abs(x) ** 4)], -1)[::-1]]); cel.fill(Q(lb[np.r_[6:n - 6, n + 6:2 * n - 6]]), '#f2e4d4'); cel.unclip()
        cel.line(Q(T), lw * 1.3, ink, taper=(.1, .1), seed=77); cel.line(Q(B), lw * 1.25, ink, taper=(.08, .08), seed=78)
        if E.get('tongue'):
            tq = spl([(-.15, bt + .10), (-.19, bt - .06), (-.09, bt - .22), (.04, bt - .26), (.17, bt - .18), (.20, bt - .02), (.17, bt + .10)], n=8); shape(cel, Q(tq), '#e0656a', lw * .9, ink, 95); cel.line(Q(spl([(.01, bt + .08), (.03, bt - .06), (.02, bt - .16)], n=5)), lw * .55, '#8a2c34', taper=(.1, .6), seed=96)
        for g in (-1, 1): cel.line(Q(spl([(g * (mw + .0), yc - .05), (g * (mw + .05), yc), (g * (mw + .035), yc + .07)], n=5)), lw * .85, ink, taper=(.3, .45), seed=81)


def hairline_cap(H, top=True):
    """closed screen polygon: skull above the hairline (goes under the front locks)"""
    sk = H.skull(); hl = spl([(-.80, .0), (-.70, .24), (-.53, .43), (-.26, .54), (0, .565), (.26, .54), (.53, .43), (.70, .24), (.80, .0)], n=12); hl = H.P(hl)
    up = H.P3(np.array([[0, 1.6, 0.]]))[0] - H.P3(np.array([[0, .2, 0.]]))[0]
    big = np.vstack([hl, hl[-1:] + (hl[-1:] - hl[:1]) * .25 + up * .2, hl[-1:] + (hl[-1:] - hl[:1]) * .3 + up * 1.4, hl[:1] - (hl[-1:] - hl[:1]) * .3 + up * 1.4, hl[:1] - (hl[-1:] - hl[:1]) * .25 + up * .2])
    lo, hi = np.minimum(sk.min(0), hl.min(0)) - 4, np.maximum(sk.max(0), hl.max(0)) + 4; k = min(1., 700. / max(hi - lo)); w, h = [int(v) + 2 for v in (hi - lo) * k]
    a = np.zeros((h, w), np.uint8); b = a.copy(); cv2.fillPoly(a, [np.round((sk - lo) * k).astype(np.int32)], 255); cv2.fillPoly(b, [np.round((big - lo) * k).astype(np.int32)], 255)
    cs = contours((a & b) / 255.)
    return [max(cs, key=len) / k + lo] if cs else []


def head_parts(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, flow=None, fs=.55, bl=1., lagT=None, amp=1., M=None, lw=None, seed=0, spir=1., wb=None, brim=None, shade_side=-1, socket=.5,
               size=1., shed=True, persp=.10, hz=.85, ldir=(-.45, -.85), **ov):
    pal = PAL[light] if isinstance(light, str) else light
    E = dict(EX[expr]) if isinstance(expr, str) else dict(expr); E.update(ov)
    H = G5Head(**G5); H.M = M; H.jawd = E.get('jaw', 0.); H.puff = E.get('puff', 0.); H.chinw = E.get('chinw', 0.); H.pose(cx, cy, s, yaw, pitch, roll, persp)
    lw = lw or float(np.clip(s * .0125, 1.15, 5.4))
    if flow is None: flow = (math.sin(roll) * .5 - .2 * math.sin(yaw), -1.)
    b0 = bl if not callable(bl) else None; glow = []
    kw = dict(flow=flow, fs=fs, bl=bl, lagT=lagT, amp=amp, seed=seed, size=size, hz=hz); hlw = lw * .62 if s > 150 else lw * .92
    bq = [b0] if b0 is not None else [bl('front', i) for i in range(len(SPEC['front']))]; bmin, bav = min(bq), float(np.mean(bq)); half = b0 is None and bmin < .999
    back = paint_hair(locks(H, f, 'back', shed=shed, **kw), pal, s, lw=hlw, light=ldir, dim=.84, seed=seed + 1, glow=glow, close=.11 if not half else .07)
    if wb is None: wb = 1. if (b0 is None or b0 > .5) else 0.
    face = _face(H, pal, E, s, lw, f, shade_side, spir, wb, brim, socket)
    cap = hairline_cap(H)
    fl_ = locks(H, f, 'front', shed=False, **kw); fr = locks(H, f, 'top', shed=shed, **kw) + fl_; cut, aft = None, []
    if half: lk, fo = loose(H, f, bav, lagT, flow, seed, hz, amp); aft = lk + [q for q in fl_ if q['uf'] < 1.05] + fo; fr = fr + lk + fo
    tg = 1. if not half else float(ss(.04, .42, bav)) * (1 + 1.05 * float(ss(1., .7, bav)))
    if bmin > .5 or (half and bav > .06):                                                      # designed hairline: an arc of flame tufts that lick onto the forehead
        hl = spl([(-.82, -.06), (-.72, .23), (-.54, .425), (-.27, .535), (0, .565), (.27, .535), (.54, .425), (.72, .23), (.82, -.06)], n=28); m = len(hl); tt = np.linspace(0, 1, m)
        tn = np.gradient(hl, axis=0); tn /= np.hypot(tn[:, 0], tn[:, 1])[:, None]; inn = c2(tn[:, 1], -tn[:, 0]); hp = hl.copy()
        for tc, wd_, dp, sl_ in ((.075, .030, .070, .6), (.145, .030, .115, .9), (.225, .030, .080, .8), (.315, .022, .045, .6), (.40, .032, .105, .9), (.485, .020, .040, .5), (.565, .026, .075, -.7), (.655, .022, .045, -.6), (.735, .034, .115, -.9), (.83, .028, .085, -.8), (.92, .030, .070, -.6)):
            u_ = (tt - tc) / (wd_ * 1.75); d_ = dp * 1.12 * tg * np.maximum(0, 1 - np.abs(u_)) ** 1.9; sw_ = sl_ * 1.15 + .35 * math.sin(f * .28 + tc * 19) * amp; hp = hp + inn * d_[:, None] - tn * (d_ * sw_)[:, None]
        low = np.array([(.70, -.35), (.3, -.5), (-.3, -.5), (-.70, -.35)]); up_ = hl - inn * (.16 * ss(0, .14, tt) * ss(1, .86, tt))[:, None]; cut = (H.P(np.vstack([hp, low])), H.P(hp), H.P(np.vstack([hp, up_[::-1]])))
    front = paint_hair(fr, pal, s, cap=cap, lw=hlw, light=ldir, seed=seed + 2, glow=glow, close=.085 if not half else .045, cut=cut, after=aft)
    return dict(back=back, face=face, front=front, H=H, glow=glow, pal=pal, lw=lw, E=E)


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, rimdir=None, rimw=None, **kw):
    """whole head as one premultiplied full-frame layer"""
    p = head_parts(cx, cy, s, yaw, pitch, roll, expr, light, f, **kw); pal = p['pal']
    L = lay(lay(p['back'], p['face']), p['front'])
    w = rimw or max(1.5, s * .02); dx, dy = rimdir or (.6, .8)
    return rim(L, dx * w, dy * w, pal['rim'], .9, soft=w * .3)


def scarf(base, nx, ax, U, f, lagT=None, seed=900, grow=1.):
    """cloud scarf: two long licking ribbons that leave the nape, swing out and curl up; base = nape px, nx/ax = screen right / up unit vectors, U px"""
    t = f / 24.; out = []
    for g in (-1, 1):
        h0 = math.atan2((nx * g - ax * .15)[1], (nx * g - ax * .15)[0])
        out += lock(base + nx * g * U * .02, h0 - g * .10, 3.0 * U * grow, .30 * U * grow, t, seed + g, wave=.62, k=1.15, hz=.65, bend=-g * 1.15, curl=-g * 1.9, cu=.60, shed=True, lagT=lagT, drag=9., r0=.5, belly=.42, tip=.7, kids=2)
        out += lock(base + nx * g * U * .50 + ax * U * .02, h0 + g * .35, 1.5 * U * grow, .21 * U * grow, t, seed + 10 + g, wave=.5, k=.9, hz=.8, bend=-g * .5, curl=g * 2.0, lagT=lagT, drag=7., shed=True, kids=1, r0=.4)
    return out


# ------------------------------------------------------------------------------------------------ rubber-hose figure
def clampl(L, m):
    L = np.asarray(L, float); r = np.hypot(L[:, 0], L[:, 1])[:, None]; return L * np.minimum(1, m / np.maximum(r, 1e-6))


def out_bow(p0, p1, mag, c):
    """bow sign that pushes the joint away from point c"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.hypot(*d) + 1e-6; nr = np.array([d[1], -d[0]]) / L; o = (p0 + p1) / 2 - c
    return abs(mag) * math.tanh(2.6 * float(nr @ o) / (np.hypot(*o) + 1e-6))


def hose(p0, p1, bow=0., w0=.15, w1=.12, n=26, bulge=0., bp=.35, c=None, wv=None, rp=None, coil=None):
    """rubber limb: p0 -> p1 with a smooth bow (fraction of length, + = left of travel).  -> (polygon, spine, normals, half widths)"""
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float); d = p1 - p0; L = np.hypot(*d) + 1e-6; nr = np.array([d[1], -d[0]]) / L
    u = lin01(n)[:, None]; m1 = p0 + d * .30 + nr * L * bow * 1.15; m2 = p0 + d * .70 + nr * L * bow * 1.15
    if c is not None: m1, m2 = np.asarray(c[0], float), np.asarray(c[1], float)
    P = (1 - u) ** 3 * p0 + 3 * (1 - u) ** 2 * u * m1 + 3 * (1 - u) * u * u * m2 + u ** 3 * p1
    if wv is not None: P = P + nr * L * wv[0] * np.sin(6.283 * 1.15 * u + wv[1]) * np.sin(np.pi * u) ** .8
    if coil is not None: en = np.sin(np.pi * u) ** .85; om = 6.283 * 2.15; P = P + (nr * np.sin(om * u + coil[1]) + d / L * .85 * np.cos(om * u + coil[1])) * L * coil[0] * en
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = c2(T[:, 1], -T[:, 0]); v = lin01(n)
    hw = w0 + (w1 - w0) * v; hw = hw * (1 + bulge * np.exp(-((v - bp) / .16) ** 2) + bulge * .8 * np.exp(-((v - .68) / .13) ** 2) - bulge * .5 * np.exp(-((v - .52) / .06) ** 2))
    if rp is not None: hw = hw * (1 + rp[1] * np.exp(-((v - rp[0]) / .14) ** 2))
    return np.vstack([P + N * hw[:, None], (P - N * hw[:, None])[::-1]]), P, N, hw


def tube(cel, P, N, hw, fill, sh, lw, ink, l, step=3):
    """a coiling hose drawn shoulder -> hand in short pieces so that its loops overlap like a drawn spiral"""
    n = len(P)
    for i in range(0, n - 1, step):
        e = min(n, i + step + 2); Q, M, w = P[i:e], N[i:e], hw[i:e, None]; A, B = Q + M * w, Q - M * w
        cel.fill(np.vstack([A, B[::-1]]), fill); sg = 1 if float((M @ np.asarray(l)).mean()) < 0 else -1; cel.fill(np.vstack([Q + M * w * sg, (Q + M * w * sg * .28)[::-1]]), sh)
        cel.line(A, lw * 1.15, ink, taper=(0, 0), press=0, tremor=0); cel.line(B, lw * 1.15, ink, taper=(0, 0), press=0, tremor=0)


def inked(cel, polys, fill, lw, ink, heavy=(.35, .5)):
    """union of shapes with one contour, heavier on the shadow side: ink stroke first (offset), fills after"""
    polys = polys if isinstance(polys, list) else [polys]; c = cel.ctx; c.save(); c.translate(heavy[0] * lw, heavy[1] * lw); c.set_line_join(1); c.set_line_cap(1)
    for p in polys: cel._path(p); cel._src(ink, 1.); c.set_line_width(lw * 2.); c.stroke_preserve(); c.fill()
    c.restore()
    for p in polys: cel.fill(p, fill)
    return cel


def limb_shade(cel, P, N, hw, col, l, k=.30):
    sg = 1 if float((N @ np.asarray(l)).mean()) < 0 else -1
    cel.fill(np.vstack([P + N * (hw * sg)[:, None], (P + N * (hw * sg * k)[:, None])[::-1]]), col)


def hand(cel, p, ang, u, kind='open', side=1, pal=None, lw=3., l=(-.5, -.8)):
    """cartoon hand with real fingers; ang = direction the fingers point, side=+1 thumb to the left of that direction, u = px per head height"""
    k = u * .235; d = lambda a, r: np.array([math.cos(a), math.sin(a)]) * r; p = np.asarray(p, float); polys = []
    palm_c = p + d(ang, k * .62); a = np.linspace(0, 6.283, 26, endpoint=False)
    palm = palm_c + np.c_[np.cos(a) * k * .66, np.sin(a) * k * .60] @ np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang), math.cos(ang)]])
    def fing(a0, L, w, base, bend=0.):
        b = palm_c + d(a0 * .75 + ang * .25 if False else a0, k * base); t = b + d(a0 + bend * .6, L * k); Pq, P, N, hw = hose(b, t, bend * .25, w * k * .92, w * k, n=12); cap = np.linspace(-math.pi / 2, math.pi / 2, 9); dv = P[-1] - P[-2]; th = math.atan2(dv[1], dv[0]); return np.vstack([Pq[:12], P[-1] + np.c_[np.cos(th - math.pi / 2 + 0 * cap + (cap + math.pi / 2)), np.sin(th - math.pi / 2 + (cap + math.pi / 2))] * w * k, Pq[12:]])
    if kind == 'open':
        fs = [(-.62, .84), (-.22, 1.0), (.18, .97), (.56, .76)]; polys = [fing(ang + side * q, L, .205 - .02 * j, .42, .10 * side) for j, (q, L) in enumerate(fs)] + [palm, fing(ang - side * 1.35, .74, .215, .30, side * .35)]
    elif kind == 'flat':
        polys = [fing(ang + side * q, L, .195, .42) for q, L in ((-.20, .95), (-.04, 1.06), (.12, 1.0), (.28, .82))] + [palm, fing(ang - side * 1.1, .6, .2, .3, side * .2)]
    elif kind == 'point':
        polys = [fing(ang - side * .05, 1.15, .2, .42), palm, fing(ang - side * 1.2, .5, .21, .3, side * .6)]
    else:                                                              # fist / grip
        polys = [palm_c + (palm - palm_c) * 1.22]
    inked(cel, polys, pal['sk'], lw, pal['ink'])
    cel.clip(polys); cel.fill(palm_c - np.asarray(l) * k * .75 + np.c_[np.cos(a), np.sin(a)] * k * .72, pal['s1']); cel.unclip()
    if kind in ('fist', 'grip'):
        for j in range(3): q = palm_c + d(ang + side * (-.45 + .45 * j), k * .80); cel.line(spl([q + d(ang, k * .03), q - d(ang, k * .22), q - d(ang, k * .42)], n=5), lw * .6, pal['ink'], taper=(.1, .5), seed=j)
        t0 = palm_c + d(ang - side * 1.2, k * .55); Pq = fing(ang - side * 1.2 + side * 1.5, .62, .2, .62, side * .3); inked(cel, Pq, pal['sk'], lw * .8, pal['ink'])
    elif kind == 'open':
        for j, q in enumerate((-.42, -.02, .37)): b = palm_c + d(ang + side * q, k * .52); cel.line([b, b - d(ang + side * q, k * .26)], lw * .5, pal['ink'], taper=(.05, .7), seed=j)


def foot(cel, p, ang, u, side, pal, lw, l):
    """sandalled foot, side-on; toe points along ang, side = which way the sole faces (+1: sole to the right of travel)"""
    k = u * .30; R = np.array([[math.cos(ang), math.sin(ang)], [-math.sin(ang) * side, math.cos(ang) * side]]); X = lambda Q: np.asarray(p) + np.asarray(Q, float) * k @ R
    sole = X(spl([(-.78, .34), (.2, .40), (1.55, .36), (1.72, .48), (1.5, .60), (.2, .62), (-.75, .60), (-.92, .48)], True, n=6)); ft = X(spl([(-.50, -.35), (-.72, .10), (-.62, .40), (.4, .44), (1.40, .42), (1.60, .22), (1.38, .02), (.80, -.02), (.50, -.30), (.35, -.45)], True, n=6))
    inked(cel, sole, pal['sole'], lw, pal['ink']); inked(cel, ft, pal['sk'], lw, pal['ink']); cel.clip(ft); cel.fill(X([(-1, .16), (.5, .22), (1.8, .28), (1.8, .8), (-1, .8)]), pal['s1']); cel.unclip()
    inked(cel, X(spl([(.72, -.06), (1.10, .0), (.98, .46), (.66, .46)], True, n=5)), '#7a4a2a', lw * .6, pal['ink'])
    for q in (1.30, 1.46): cel.line(X([(q, .10), (q + .04, .26)]), lw * .5, pal['ink'], taper=(.2, .5))


def cuff(cel, P, N, hw, j, u, pal, lw, seed, f):
    """fluffy cloud cuff at the end of a short / sleeve"""
    c = P[j]; n = N[j]; t = np.array([-n[1], n[0]]); w = hw[j]; m = 5; pts = []
    for i in range(m):
        v = -1 + 2 * i / (m - 1.); r = u * .115 * (1 + .18 * math.sin(f * .5 + i * 1.7 + seed)); a = np.linspace(0, 6.283, 14, endpoint=False); q = c + n * w * v + t * u * .03; pts.append(q + np.c_[np.cos(a), np.sin(a)] * r)
    inked(cel, pts, pal['cl'], lw, pal['ink']); cel.clip(pts); 
    for p in pts: cel.fill(p + [u * .035, u * .06], pal['c1'])
    for p in pts: cel.fill(p + [-u * .02, -u * .035], pal['cl'])
    cel.unclip()


BASE = dict(pel=(0, 2.0), che=(0, 3.05), hd=(0, 4.12), roll=0., yaw=0., pitch=0., hA=(-1.3, 2.4), hB=(1.3, 2.4), bA=0.2, bB=0.2, fA=(-.5, 0), fB=(.5, 0), kA=0.25, kB=0.25, aA=math.pi, aB=0., kA_h='open', kB_h='open', hdA=None, hdB=None,
            sq=1., belly=0., expr='joy', back=(0, 0), E=None, arms='front', scarf=1., flow=None, amp=1., hsz=1.16, wv=.05, st=0., ht=0., sink=0., spA=0., spB=0., rip=None)
POSES = dict(
    dance=dict(BASE, pel=(-.25, 1.6), che=(.05, 2.65), hd=(.15, 3.72), roll=.16, hA=(-1.1, 5.0), hB=(1.45, 1.95), bA=0.35, bB=0.35, fA=(-.55, 0), fB=(1.0, 1.4), kA=0.35, kB=0.5),
    rise=dict(BASE, pel=(0, 2.3), che=(0, 3.5), hd=(0, 4.6), hA=(-1.5, 4.9), hB=(1.5, 4.9), bA=0.2, bB=0.2),
    squat=dict(BASE, pel=(0, 1.1), che=(0, 1.95), hd=(0, 2.9), hA=(-.6, .25), hB=(.6, .25), bA=0.35, bB=0.35, fA=(-.95, 0), fB=(.95, 0), kA=0.55, kB=0.55, sq=.8, kA_h='fist', kB_h='fist'),
    leap=dict(BASE, pel=(0, 2.4), che=(0, 3.6), hd=(0, 4.75), hA=(-2.1, 5.6), hB=(2.1, 5.6), bA=0.15, bB=0.15, fA=(-1.4, .9), fB=(1.4, .9), kA=0.3, kB=0.3, sq=1.15),
    fall=dict(BASE, pel=(0, 2.2), che=(.1, 3.2), hd=(.3, 4.2), roll=.5, hA=(-1.6, 4.0), hB=(1.7, 4.3), bA=0.2, bB=0.2, fA=(-1.0, .6), fB=(.9, .9), expr='laugh'),
    reach=dict(BASE, pel=(0, 2.0), che=(.1, 3.1), hd=(.15, 4.2), hA=(-1.2, 2.2), hB=(1.6, 5.8), bA=0.2, bB=0.1),
    punch=dict(BASE, pel=(-.2, 1.7), che=(.1, 2.7), hd=(.3, 3.8), hA=(-1.4, 2.6), hB=(3.2, 3.4), bA=0.35, bB=0.0, fA=(-1.0, 0), fB=(.8, 0), kA=0.3, kB=0.35, kB_h='fist', kA_h='fist', expr='fierce'))


def figure(cx, cy, U, pose='dance', f=0, light='night', lagT=None, l=(-.45, -.85), rims=((.7, .9), (-.7, .9)), lw=None, ss_=2, head_kw=None, rimw=None, part=None, **over_):
    """rubber-hose Gear 5.  pose = name or dict (units of U, y UP from the ground point).  -> full-frame premultiplied layer"""
    p = dict(BASE); p.update(POSES[pose] if isinstance(pose, str) else pose); p.update(over_)
    pal = PAL[light] if isinstance(light, str) else light; ink = pal['ink']; lw = lw or float(np.clip(U * .020, 1.1, 4.2)); il = lw * .58
    X = lambda q: np.array([cx + q[0] * U, cy - q[1] * U]); pel, che, hd = X(p['pel']), X(p['che']), X(p['hd'])
    ax = che - pel; aL = np.hypot(*ax) + 1e-6; ax /= aL; nx = np.array([-ax[1], ax[0]])                                     # nx = screen-left... (right of up axis)
    if nx[0] < 0: nx = -nx
    sq = p['sq']; wd = U / math.sqrt(max(sq, .3))
    st, ht, sk_, rip = p['st'], p['ht'], p['sink'], p['rip']; su = .18 + .17 * sk_
    shA, shB = che + ax * U * (su + .27 * st) - nx * wd * .50, che + ax * U * (su - .27 * st) + nx * wd * .50; hpA, hpB = pel - nx * wd * .25 + ax * U * .20 * ht, pel + nx * wd * .25 - ax * U * .20 * ht
    rL = None if rip is None else (1.05 - rip * 3.0, .34); rA = None if rip is None else ((rip - .50) * 2.4, .34); rT = 0. if rip is None else rip * 2.6 - .75; rH = 0. if rip is None else math.exp(-((rip - .80) / .16) ** 2)
    t = f / 24.; L2 = np.asarray(l, float); lagT = np.zeros((2, 2)) if lagT is None else np.asarray(lagT, float)
    lagH, lagT = clampl(lagT * .34, .40 * U), clampl(lagT * .55, .75 * U)
    neck0 = che + ax * U * .22
    # --- scarf behind
    layers = []
    if p['scarf'] > 0:
        sc = scarf(neck0 + ax * U * .08, nx, ax, U * p['scarf'], f, lagT)
        spal = dict(pal); spal.update(h0=pal['c2'], h1=pal['c1'], h2=pal.get('sc2', '#e6c6e0'), h3=pal.get('sc3', '#fbeaf4')); layers.append(paint_hair(sc, spal, U * .5, lw=lw * .8, light=l, seed=9, dim=.97, close=.10))
    s = U * .5 * p['hsz']; hk = dict(roll=p['roll'], yaw=p['yaw'], pitch=p['pitch'], expr=p['expr'], light=pal, f=f, lagT=lagH, amp=p['amp'], flow=p['flow'], lw=lw * .9); hk.update(head_kw or {}); hk.update(p['E'] or {}); hk['puff'] = hk.get('puff', EX[p['expr']].get('puff', 0.) if isinstance(p['expr'], str) else 0.) + .14 * rH
    M = np.array([[1 / math.sqrt(sq), 0], [0, math.sqrt(sq)]]) if abs(sq - 1) > .01 else None
    hp = head_parts(hd[0], hd[1], s, M=M, **hk)
    layers.append(hp['back'])
    cel = Cel(ss=ss_)
    # --- shirt tails behind
    for g in (-1, 1):
        r = pel + nx * g * wd * .30 + ax * U * .15; w_ = .28 * U; a0 = math.atan2((-ax)[1], (-ax)[0]) - g * .45 * (1 if True else 0)
        k = lock(r, a0, .80 * U, w_, t, 930 + g, wave=.30, k=.7, hz=1.0, curl=-g * .5, bend=-g * .3, lagT=lagH, drag=5., r0=1., belly=.2, tip=.45, q=6)[0]
        inked(cel, k['P'], pal['c1'], lw, ink); cel.line(k['c'][6:-6], il, pal['c2'], taper=(.4, .4), seed=2)

    def leg(hip, ft, kb, ang, sd):
        fx = X(ft); d = fx - hip; Ln = np.hypot(*d); th = .155 * U * math.sqrt(np.clip(1.5 * U / max(Ln, 1), .55, 1.25))
        poly, P, N, hw = hose(hip, fx, out_bow(hip, fx, kb, pel + ax * U * .3), th * 1.15, th * .78, bulge=.16, bp=.25, rp=rL)
        foot(cel, fx + [0, -U * .10], ang, U, 1 if math.cos(ang) >= 0 else -1, pal, lw, l)
        inked(cel, poly, pal['sk'], lw, ink); limb_shade(cel, P, N, hw, pal['s1'], l)
        j = int(len(P) * .52); e = slice(0, j); w2 = np.linspace(.27, .31, j) * U * math.sqrt(np.clip(1.3 * U / max(Ln, 1), .6, 1.15)); Q, Nq = P[e], N[e]
        sh = np.vstack([Q + Nq * w2[:, None], (Q - Nq * w2[:, None])[::-1]]); inked(cel, sh, pal['cl'], lw, ink); limb_shade(cel, Q, Nq, w2, pal['c1'], l, .22)
        for q in (.35, .62): a = int(j * q); cel.line(spl([Q[a] + Nq[a] * w2[a] * .9, Q[a] + Nq[a] * w2[a] * .2 + (Q[min(j - 1, a + 3)] - Q[a]), Q[a] - Nq[a] * w2[a] * .25], n=5), il, ink, taper=(.1, .6), seed=a)
        cuff(cel, P, N, np.r_[w2 * .92, hw[j:]], j - 1, U, pal, lw, sd, f)
    order = [(p['fA'][1], hpA, p['fA'], p['kA'], p['aA'], 1), (p['fB'][1], hpB, p['fB'], p['kB'], p['aB'], 2)]
    order.sort(key=lambda q: q[0])
    for _, h_, ft, kb, ang, sd in order: leg(h_, ft, kb, ang, sd)

    def arm(sh, hn, bow, kind, side, hdir):
        hn = X(hn); d = hn - sh; Ln = np.hypot(*d); th = .135 * U * math.sqrt(np.clip(1.7 * U / max(Ln, 1), .5, 1.3))
        sp = p['spA'] if side > 0 else p['spB']; nn = 62 if sp > .08 else 26
        poly, P, N, hw = hose(sh, hn, out_bow(sh, hn, bow, che - ax * U * .2), th * 1.1 * (1 - .18 * sp), th * .80 * (1 - .15 * sp), n=nn, bulge=.15 * (1 - sp), bp=.3, wv=(p['wv'] * min(1., Ln / U / 1.6), f * .55 + side), rp=rA, coil=(.105 * sp, -f * 1.25 * side + side) if sp > .08 else None)
        dv = P[-1] - P[-4]; a = math.atan2(dv[1], dv[0]) if hdir is None else hdir
        hand(cel, P[-1], a, U * (1.05 + .10 * max(0, (Ln / U - 2) / 2)), kind, side, pal, lw, l)
        if sp > .08: tube(cel, P, N, hw, pal['sk'], pal['s1'], lw, ink, l); cel.fill(hand_cap(P[-1], th * .72), pal['sk'])
        else: inked(cel, poly[np.r_[0:len(P) - 1, len(P) + 1:2 * len(P)]], pal['sk'], lw, ink); cel.fill(hand_cap(P[-1], th * .8), pal['sk']); limb_shade(cel, P, N, hw, pal['s1'], l)
        j = int(len(P) * (.40 - .14 * sp)); w2 = np.linspace(.22, .245, j) * U * math.sqrt(np.clip(1.5 * U / max(Ln, 1), .6, 1.2)); Q, Nq = P[:j], N[:j]
        inked(cel, np.vstack([Q + Nq * w2[:, None], (Q - Nq * w2[:, None])[::-1]]), pal['cl'], lw, ink); limb_shade(cel, Q, Nq, w2, pal['c1'], l, .22)
        e = np.vstack([Q[-3:] + Nq[-3:] * (w2[-3:] * 1.10)[:, None], (Q[-3:] - Nq[-3:] * (w2[-3:] * 1.10)[:, None])[::-1]]); inked(cel, e, pal['cl'], lw * .9, ink)
        cel.line(spl([Q[2] + Nq[2] * w2[2] * .2, Q[j // 2] + Nq[j // 2] * w2[j // 2] * .35, Q[-4] + Nq[-4] * w2[-4] * .05], n=5), il, ink, taper=(.3, .5), seed=5)
    if p['arms'] == 'back': arm(shA, p['hA'], p['bA'], p['kA_h'], 1, p['hdA']); arm(shB, p['hB'], p['bB'], p['kB_h'], -1, p['hdB'])
    # --- torso: skin, X scar, abs, shirt panels
    bl = p['belly']; v = np.array([-.10, .10, .34, .62, .86, 1.10, 1.24]); hw = np.array([.30, .33 + .30 * bl, .30 + .62 * bl, .40 + .34 * bl, .50, .46, .20]) * wd; bend = (np.asarray(p['back']) * U)
    C = pel + (che - pel) * v[:, None] + np.sin(np.clip(v, 0, 1) * math.pi)[:, None] * bend; hw = hw * (1 + .20 * np.exp(-((v - rT) / .24) ** 2) * (rip is not None)); tv = ((ht + (st - ht) * np.clip(v, 0, 1)) * U * .5 * hw / wd)[:, None]; TL = spl(C - nx * hw[:, None] + ax * tv, n=8); TR = spl(C + nx * hw[:, None] - ax * tv, n=8); torso = np.vstack([TL, TR[::-1]])
    inked(cel, torso, pal['sk'], lw, ink); cel.clip(torso); cel.fill(torso + nx * wd * .52 * (1 if L2[0] < 0 else -1) + [0, U * .05], pal['s1'])
    mid = lambda vv, xx: np.array([np.interp(vv, v, C[:, 0]), np.interp(vv, v, C[:, 1])]) + nx * xx * wd - ax * xx * U * .5 * (ht + (st - ht) * min(max(vv, 0), 1))
    for a, b in (((1.02, -.30), (.52, .28)), ((1.02, .30), (.52, -.28))):                                                           # burn scar X
        P0, P1 = mid(*a), mid(*b); d = P1 - P0; n_ = np.array([-d[1], d[0]]) / (np.hypot(*d) + 1e-6) * U * .085; m = 7; r = lambda i: (1 + .35 * (h1(int(a[1] * 100), i) - .5) * 2); e1 = [P0 + d * k / m + n_ * r(k) for k in range(m + 1)]; e2 = [P0 + d * k / m - n_ * r(k + 20) for k in range(m + 1)]
        S = np.vstack([e1, e2[::-1]]); cel.fill(S, pal['blush'], .34); cel.line(np.array(e1), il * .7, pal['s2'], taper=(.2, .2), alpha=.8); cel.line(np.array(e2), il * .7, pal['s2'], taper=(.2, .2), alpha=.8)
    if bl < .25:
        cel.line([mid(.50, .0), mid(.16, .0)], il * .8, pal['s2'], taper=(.3, .5))
        for g in (-1, 1): cel.line(spl([mid(.62, .06 * g), mid(.58, .2 * g), mid(.66, .34 * g)], n=5), il * .9, ink, taper=(.4, .3)); cel.line(spl([mid(.34, .07 * g), mid(.33, .16 * g)], n=4), il * .7, pal['s2'], taper=(.3, .4))
    else:                                                                                                                    # drum belly: hoop, tacks, skin highlight
        cb = mid(.36, 0); a = np.linspace(0, 6.283, 40); rr = wd * (.30 + .46 * bl); E_ = cb + np.c_[np.cos(a) * rr, np.sin(a) * rr * .92 * min(1, aL / U * 1.1)]
        cel.line(E_, il * 1.2, ink, taper=(0, 0)); cel.line(cb + (E_ - cb) * .86, il * .8, pal['s2'], taper=(0, 0))
        for k in range(12): q = cb + (E_[int(k * 40 / 12)] - cb) * .93; cel.fill(q + np.c_[np.cos(a), np.sin(a)] * U * .022, ink)
        cel.fill(mid(.42, -.16) + np.c_[np.cos(a) * U * .13, np.sin(a) * U * .08], pal['rim'], .55)
    cel.unclip()
    for g, E1 in ((-1, TL), (1, TR)):                                                                                        # open shirt panels
        m = len(E1); fl = .05 * U * math.sin(t * 9 + g) ; inn = np.array([mid(1.16, g * .18), mid(.92, g * .27), mid(.62, g * .30), mid(.30, g * .34 + .0), mid(.04, g * (.44 + .10 * bl)) + [fl, 0]]); outr = np.array([mid(-.02, g * (.62 + .2 * bl)) + [fl * 1.4, 0], mid(.30, g * (.45 + .5 * bl)), mid(.62, g * (.50 + .3 * bl)), mid(.9, g * .57), mid(1.16, g * .46)])
        pan = np.vstack([spl(inn, n=8), spl(outr, n=8)]); inked(cel, pan, pal['cl'], lw, ink); cel.clip(pan); cel.fill(pan + nx * wd * .13 * (1 if L2[0] < 0 else -1) + [0, U * .03], pal['c1']); cel.unclip()
        col = np.vstack([spl([mid(1.22, g * .10), mid(1.0, g * .22), mid(.74, g * .30)], n=6), spl([mid(.80, g * .42), mid(1.02, g * .36), mid(1.24, g * .24)], n=6)]); inked(cel, col, pal['cl'], lw * .8, ink)
        for q in (.36, .16): bt = mid(q, g * (.40 + .18 * bl)); cel.fill(bt + np.c_[np.cos(np.linspace(0, 6.28, 12)), np.sin(np.linspace(0, 6.28, 12))] * U * .028, pal['sash']); cel.line(spl([mid(q + .14, g * .50), mid(q + .06, g * .42), mid(q + .02, g * .36)], n=5), il * .9, ink, taper=(.2, .6))
    # --- sash
    sa = np.vstack([spl([mid(.10, -.36), mid(.06, 0), mid(.10, .36)], n=8), spl([mid(-.10, .37), mid(-.15, 0), mid(-.10, -.37)], n=8)]); r = mid(-.02, .36 * 1.0)
    
    for dh, L_, sd_ in ((-1.18, .60, 951), (-.80, .48, 950)):                                                               # two short tails at the left hip, always thrown outward
        tl = lock(r, math.atan2((-ax)[1], (-ax)[0]) + dh, L_ * U, .115 * U, t, sd_, wave=.42, hz=1.2, k=.9, curl=.1, lagT=lagH * .6, drag=6., r0=.42, belly=.95, tip=.12, q=18)[0]; inked(cel, tl['P'], pal['sash'], lw, ink); limb_shade(cel, tl['c'], tl['N'], tl['hw'], pal['sash1'], l, .1)
   
    inked(cel, sa, pal['sash'], lw, ink); cel.clip(sa); cel.fill(sa + [U * .02, U * .13], pal['sash1']); cel.unclip()
    for q in (-.2, 0, .18): cel.line(spl([mid(.06, q), mid(-.02, q + .05), mid(-.12, q + .02)], n=5), il * .8, ink, taper=(.3, .5))
    # --- neck, head, arms
    nk, P, N, hw_ = hose(neck0 - ax * U * .10, hd + (hd - neck0) * -.02 + np.array([0, U * .30]), 0., .15 * U, .13 * U, n=10); inked(cel, nk, pal['sk'], lw, ink); cel.clip(nk); cel.fill(nk + [0, -U * .08], pal['s2']); cel.unclip()
    layers.append(cel.layer())
    layers += [hp['face'], hp['front']]
    cel = Cel(ss=ss_)
    if p['arms'] == 'front':
        for sh, h_, b_, k_, sd, dr in ((shA, p['hA'], p['bA'], p['kA_h'], 1, p['hdA']), (shB, p['hB'], p['bB'], p['kB_h'], -1, p['hdB'])): arm(sh, h_, b_, k_, sd, dr)
        layers.append(cel.layer())
    out = layers[0]
    for q in layers[1:]: out = lay(out, q)
    rw = rimw or max(1.3, U * .016)
    for dx, dy in rims: out = rim(out, dx * rw, dy * rw, pal['rim'], .92, soft=rw * .25)
    return out


def hand_cap(p, r):
    a = np.linspace(0, 6.283, 20, endpoint=False); return np.asarray(p) + np.c_[np.cos(a), np.sin(a)] * r
