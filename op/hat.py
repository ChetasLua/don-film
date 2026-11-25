"""THE straw hat: a true 3-D hat (dome crown + wavering brim) inked as a cel at any angle.  One module for the whole film.

    hat(cx, cy, w, yaw=0, pitch=0, roll=0, light='day', wear=.6, bounce=0, part='all', f=0, tilt=0, lean=0, wind=.4, rim=1) -> premult RGBA layer (1080,1920,4)
        cx, cy = middle of the crown (on the axis, half way up the dome), w = brim width in px.  yaw/pitch/roll exactly as op.head3 (pitch > 0 looks up and
        shows the underside, roll > 0 clockwise).  tilt/lean = the hat's own pitch/roll on the head (tilt > 0 front brim up, < 0 pulled down over the eyes).
        wear 0..1: ribbon dulls, edge frays, the stitched slash shows from .35.   bounce: > 0 brim flops DOWN, < 0 flips up (about -1..1); feed it a damped spring.
        part: 'back' (draw first), then your head, then 'front' (crown + ribbon + the near half of the brim).  'all' = whole hat.  Both parts come from one cached drawing.
        f only drives the small flutter of the brim (wind = amount).  light = name in LIGHTS or your own dict (copy one and change keys).
    seat(Head, w=3.0, tilt=0, lean=0, up=0) -> kwargs (cx, cy, w, yaw, pitch, roll, tilt, lean) that put the hat on a posed op.head3.Head:
        k = seat(H); lay = over(back) ; head ; over(hat(**k, part='front', light=..., f=f))
    settle(f, f0, amp=.5) -> bounce value of a damped 4-frame flop that starts at f0 (brim dips, rebounds, settles)
    under_brim(f, amt, stars=1., sun=(960,300)) -> layer: looking out from UNDER it.  amt 0 nothing, .5 a heavy band along the top, 1 the top third, 3 covers the frame.
"""
import numpy as np, cv2, math
from numba import njit
from . import W, H
from .paint import rgb, f32, smooth, ramp, fbm
from .ink import Cel, spl, resample, paste, noise1
from .kit import contours, outline, para

RB, HC, Y0 = .555, .47, -.225
VR = .19                                             # top of the ribbon (crown parameter)

BASE = dict(L=(-.5, .72, .48), straw='#e3b557', hi='#f5d27c', sh='#a9742c', dk='#7a4a1c', under='#8d5c26', under_dk='#5f3916', rib='#b3202a', rib_sh='#73101e', rib_hi='#dd4548',
            line='#1e1210', weave='#8a591f', weave_sh='#4c2a10', weave_hi='#fbe6a6', rim='#fff3cf', rd=(-1., -.75), rw=1.0, glow=0., para=None)


def _mk(mul=(1, 1, 1), sh=(1, 1, 1), mix_=None, **k):
    d = dict(BASE); m, s = np.array(mul, f32), np.array(sh, f32)
    for q in ('straw', 'hi', 'rib', 'rib_hi', 'weave', 'weave_hi'): d[q] = np.clip(rgb(d[q]) * m, 0, 1.2)
    for q in ('sh', 'dk', 'under', 'under_dk', 'rib_sh', 'weave_sh'): d[q] = np.clip(rgb(d[q]) * s, 0, 1.2)
    if mix_:                                         # pull everything toward an ambient colour (keeps straw from going olive in blue light)
        c, a = rgb(mix_[0]), mix_[1]
        for q in ('straw', 'hi', 'rib', 'rib_hi', 'weave', 'weave_hi', 'sh', 'dk', 'under', 'under_dk', 'rib_sh', 'weave_sh'):
            v = rgb(d[q]) if isinstance(d[q], str) else d[q]; a_ = a * (.55 if q.startswith('rib') else 1); d[q] = v * (1 - a_) + c * a_ * (.45 + .9 * float(v.mean()))
    d.update(k); return d


LIGHTS = dict(
    day=_mk(para=[(0, '#ffffff'), (1, '#e8cfc0')]),
    dusk=_mk((1.02, .90, .78), (.92, .70, .74), rim='#ffe2a0', rd=(1., -.35), L=(.62, .45, .35), glow=.55, para=[(0, '#fff6e0'), (1, '#b98a94')], line='#22100e'),
    gold=_mk((.80, .66, .54), (.70, .52, .50), rim='#ffd98a', rd=(1., -.7), L=(-.75, .5, -.1), glow=.4, para=[(0, '#ffe9c4'), (1, '#8a6a70')], line='#1c0e0c', rw=1.25),
    storm=_mk((.80, .78, .80), (.62, .62, .68), ('#6f87a8', .42), rim='#dff0ff', rd=(1., -.9), L=(.5, .8, .2), para=[(0, '#ffffff'), (1, '#7f8aa6')], line='#0d0f18'),
    lamp=_mk((1.02, .80, .55), (.74, .46, .36), rim='#ffd27a', rd=(-1., -.2), L=(-.8, .35, .5), glow=.15, para=[(0, '#fff0cc'), (1, '#8a5a50')], line='#1a0c0a'),
    fire=_mk((.98, .70, .46), (.62, .34, .28), rim='#ffc060', rd=(1., -.5), L=(.65, .5, .2), glow=.35, para=[(0, '#ffe0b0'), (1, '#7a3c38')], line='#1d0a08', rw=1.3),
    night=_mk((.66, .62, .70), (.50, .48, .58), ('#4a5a96', .45), rim='#bcd4ff', rd=(-1., -.8), L=(-.5, .75, .4), line='#0c0c16', para=[(0, '#ffffff'), (1, '#6f76a4')]),
    sea=_mk((.84, .80, .70), (.56, .54, .56), ('#2f7fae', .32), rim='#d6fbff', rd=(.2, -1.), L=(.1, .95, .2), line='#06101e', para=[(0, '#ffffff'), (1, '#3c6a98')], glow=.2),
    dark=_mk((.30, .28, .34), (.22, .20, .26), rim='#ffe9c0', rd=(1., -.7), L=(.5, .8, .2), line='#08060a', para=[(0, '#ffffff'), (1, '#8a7a90')]),
    flat=_mk(rim='#fff3cf'),
    charcoal=dict(BASE, straw='#8f8780', hi='#9b938b', sh='#5f5852', dk='#433d39', under='#4f4844', under_dk='#322d2b', rib='#a01822', rib_sh='#6e0f18', rib_hi='#b8323a', weave='#443e3a',
                  weave_sh='#221e1c', weave_hi='#b5ada4', line='#141110', rim='#d8d0c6'))


# ------------------------------------------------------------------ surfaces (hat space: x right, y up, z front; theta 0 = front, + = right)
def crown(v, th):
    a = np.clip(v, 0, 1) * math.pi / 2
    r = RB * np.maximum(np.cos(a), 0) ** .68 * (1 + .010 * np.sin(3 * th + 1) + .010 * np.sin(2 * th + .4))
    y = Y0 + HC * np.sin(a) ** .95 + .012 * np.sin(2 * th + .3) * np.sin(a)
    return np.stack([r * np.sin(th) * .985, y, r * np.cos(th) * 1.02], -1)


def ribbon_s(v, th):
    q = crown(v * VR, th); k = 1 + .022 + .006 * np.sin(5 * th + 2) * v
    q[..., 0] *= k; q[..., 2] *= k; q[..., 1] += .006 * np.sin(4 * th + .7) * v
    return q


class _Brim:
    def __init__(s, wear, bounce, f, wind): s.wear, s.b, s.ph, s.w = wear, bounce, f * .36, wind

    def edge(s, th):
        e = 1 + .018 * np.sin(4 * th + 1) + .011 * np.sin(7 * th + 2) - .012
        return e + (.004 + .012 * s.wear) * (np.sin(23 * th + 1.7) * np.sin(9 * th + .2) + .6 * np.sin(41 * th) * np.sin(13 * th + 2))

    def y(s, t, th):
        t = np.clip(t, 0, 1.2)
        k = .052 * np.sin(2 * th + .6) + .034 * np.sin(3 * th + 2.1) + .016 * np.sin(5 * th + .3)
        fl = s.w * (.020 * np.sin(2 * th + s.ph) + .012 * np.sin(3 * th - 1.3 * s.ph + 1))
        return Y0 + .012 - .115 * t ** 1.5 + t ** 1.4 * (k + fl) - s.b * .30 * t ** 1.7 * (.78 + .22 * np.cos(2 * th))

    def __call__(s, t, th):
        sh = 1 - .10 * abs(s.b) * np.clip(t, 0, 1.2) ** 2 * .8
        r = (RB + t * (s.edge(th) - RB)) * sh
        return np.stack([r * np.sin(th), s.y(t, th), r * np.cos(th)], -1)


@njit(cache=True)
def _ras(Z, LB, UA, VA, P, D, U, V, F, lp, ln):
    h, w = Z.shape
    for k in range(F.shape[0]):
        a, b, c = F[k, 0], F[k, 1], F[k, 2]
        x0, y0, x1, y1, x2, y2 = P[a, 0], P[a, 1], P[b, 0], P[b, 1], P[c, 0], P[c, 1]
        ar = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(ar) < 1e-9: continue
        lab = lp if ar > 0 else ln
        xa = max(0, int(math.floor(min(x0, x1, x2)))); xb = min(w - 1, int(math.ceil(max(x0, x1, x2))))
        ya = max(0, int(math.floor(min(y0, y1, y2)))); yb = min(h - 1, int(math.ceil(max(y0, y1, y2))))
        for y in range(ya, yb + 1):
            for x in range(xa, xb + 1):
                w1 = ((x - x0) * (y2 - y0) - (x2 - x0) * (y - y0)) / ar
                w2 = ((x1 - x0) * (y - y0) - (x - x0) * (y1 - y0)) / ar
                w0 = 1 - w1 - w2
                if w0 < -1e-4 or w1 < -1e-4 or w2 < -1e-4: continue
                d = w0 * D[a] + w1 * D[b] + w2 * D[c]
                if d > Z[y, x]:
                    Z[y, x] = d; LB[y, x] = lab; UA[y, x] = w0 * U[a] + w1 * U[b] + w2 * U[c]; VA[y, x] = w0 * V[a] + w1 * V[b] + w2 * V[c]


def _faces(nr, nc):
    i, j = np.meshgrid(np.arange(nr - 1), np.arange(nc - 1), indexing='ij'); a = (i * nc + j).ravel()
    return np.vstack([np.c_[a, a + 1, a + nc], np.c_[a + 1, a + nc + 1, a + nc]]).astype(np.int64)


def rotm(yaw, pitch, roll, tilt=0., lean=0.):
    c, s = math.cos(yaw), math.sin(yaw); Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    c, s = math.cos(pitch), math.sin(pitch); Rx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c, s = math.cos(-roll), math.sin(-roll); Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    c, s = math.cos(tilt), math.sin(tilt); Tx = np.array([[1, 0, 0], [0, c, s], [0, -s, c]])
    c, s = math.cos(-lean), math.sin(-lean); Tz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return Rz @ Rx @ Ry @ Tz @ Tx


def _sliver(P, w):
    """(n,k,2) polylines + (n,) widths -> list of tapered polygons"""
    T = np.gradient(P, axis=1); T /= np.linalg.norm(T, axis=-1, keepdims=True) + 1e-9; N = np.stack([-T[..., 1], T[..., 0]], -1)
    k = P.shape[1]; prof = np.sin(np.linspace(.12, math.pi - .05, k)) ** .7
    hw = (np.asarray(w)[:, None] * .5 * prof[None])[..., None]
    return list(np.concatenate([P + N * hw, (P - N * hw)[:, ::-1]], 1))


def _runs(m):
    m = np.r_[False, m, False]; d = np.flatnonzero(m[1:] != m[:-1]); return list(zip(d[::2], d[1::2]))


_cache = {}


def _draw(cx, cy, w, yaw, pitch, roll, light, wear, bounce, f, tilt, lean, wind, rim, lw, seed):
    LT = LIGHTS[light] if isinstance(light, str) else dict(BASE, **light)
    C = {k: (rgb(v) if isinstance(v, str) else np.asarray(v, f32)) for k, v in LT.items() if k not in ('L', 'rd', 'rw', 'glow', 'para')}
    fade = np.array([.70, .46, .40], f32) * .9
    for q in ('rib', 'rib_hi', 'rib_sh'): C[q] = C[q] * (1 - .16 * wear) + fade * C[q].mean() * 2.1 * .16 * wear
    for q in ('straw', 'hi'): C[q] = C[q] * (1 - .05 * wear)
    R = rotm(yaw, pitch, roll, tilt, lean); s = w / 2; br = _Brim(wear, bounce, f, wind); pk = .12
    def pr(Q):
        q = np.asarray(Q, float) @ R.T; k = 1 / (1 - np.clip(q[..., 2] * pk, -.5, .5))
        return np.stack([cx + q[..., 0] * k * s, cy - q[..., 1] * k * s], -1), q[..., 2]
    th = np.linspace(-math.pi, math.pi, 145); tb = np.linspace(-.02, 1, 17); vc = np.linspace(0, 1, 27); vr = np.linspace(-.1, 1, 6)
    G = [(crown, vc, 1, 5), (ribbon_s, vr, 2, 5), (br, tb, 4, 3)]
    grids = []
    for fn, u, a, b in G:
        U, T = np.meshgrid(u, th, indexing='ij'); p, d = pr(fn(U, T)); grids.append((p, d, U, T, a, b))
    allp = np.vstack([g[0].reshape(-1, 2) for g in grids]); pad = int(12 + w * .06)
    x0, y0 = np.maximum(np.floor(allp.min(0)).astype(int) - pad, [-90, -90]); x1, y1 = np.minimum(np.ceil(allp.max(0)).astype(int) + pad, [W + 90, H + 90])
    if x1 - x0 < 4 or y1 - y0 < 4: return None
    k = min(1., 760. / max(x1 - x0, y1 - y0)); hh, ww = int((y1 - y0) * k) + 2, int((x1 - x0) * k) + 2
    Z = np.full((hh, ww), -9., np.float64); LB = np.zeros((hh, ww), np.int32); UA = np.zeros((hh, ww)); VA = np.zeros((hh, ww))
    for p, d, U, T, a, b in grids:                   # grid winding: +u up / out, +theta right -> outside has one sign on screen
        F = _faces(*U.shape); _ras(Z, LB, UA, VA, (p.reshape(-1, 2) - [x0, y0]) * k, d.ravel(), U.ravel().astype(float), T.ravel().astype(float), F, b, a)
    # orientation is the same for all three grids; fix the sign once from the crown (outside must win when seen from the front)
    off = np.array([x0, y0], float)
    def cont(m, eps=20):
        m = m.astype(f32)
        if m.max() <= 0: return []
        return [resample(c / k + off, 2.0) for c in contours(cv2.GaussianBlur(m, (0, 0), .9), min_len=eps)]
    # ---- light
    L = np.array(LT['L'], float); L /= np.linalg.norm(L); l = R.T @ L
    e = 2e-3; N = np.zeros((hh, ww, 3)); PX = np.zeros((hh, ww, 3))
    for lab, fn in ((1, crown), (2, ribbon_s), (3, br), (4, br), (5, crown)):
        m = LB == lab
        if not m.any(): continue
        u, t = UA[m], VA[m]; p = fn(u, t); n = np.cross(fn(u, t + e) - p, fn(u + e, t) - p); n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
        c = p - np.array([0, Y0 + .1, 0])
        sg = np.sign((n * c).sum(1)) if lab in (1, 2) else (-np.sign((n * c).sum(1)) if lab == 5 else np.sign(n[:, 1]) * (1 if lab == 3 else -1))
        N[m] = n * sg[:, None]; PX[m] = p
    vv = R.T @ np.array([0, 0, 1.]); l2 = l + .55 * vv; l2 /= np.linalg.norm(l2); ndl = (N * l).sum(-1)
    cm = (LB == 1) | (LB == 2)                       # drawn terminator: a curved crescent that steps along the coils, not a ray-traced plane
    if cm.any():
        dr = (N * l2).sum(-1) - .38 + .16 * (PX[..., 1] - Y0) / HC; dr += .035 * np.sin(UA * 34 + 2 * np.sin(VA * 3)) + .03 * np.sin(VA * 11 + UA * 7); ndl = np.where(cm, dr + .22, ndl)
    top = LB == 3; cast = np.zeros((hh, ww), bool)
    if top.any():
        p = PX[top]; c = np.zeros(len(p), bool)
        for q in np.linspace(.04, 1.3, 14):
            pp = p + l * q; c |= ((np.hypot(pp[:, 0], pp[:, 2]) / RB) ** 2.6 + (np.abs(pp[:, 1] - Y0) / HC) ** 2.6 < 1) & (pp[:, 1] > Y0)
        cast[top] = c; cast = (cv2.GaussianBlur(cast.astype(f32), (0, 0), 5.5) > .5) & top
    shade = ((LB == 1) & (ndl < .22)) | ((LB == 2) & (ndl < .22)) | (top & (cast | (ndl < .10)))
    deep = ((LB == 1) & (ndl < -.16)) | (top & cast & (UA < .16 + .05 * np.sin(VA * 5))) | (LB == 5)
    hi = ((LB == 1) & (ndl > .90)) | (top & ~cast & (ndl > .965) & (UA > .25))
    sil = LB > 0
    cel = Cel(x1 - x0, y1 - y0, x0, y0); r = np.random.default_rng(seed)
    S = cont(sil); cr = cont((LB == 1) | (LB == 2)); rb = cont(LB == 2); un = cont((LB == 4)); ins = cont(LB == 5)
    cel.fill(S, C['straw'], evenodd=True)
    if un: cel.fill(un, C['under'], evenodd=True)
    if ins: cel.fill(ins, C['under_dk'] * .55, evenodd=True)
    if cr: cel.fill(cr, C['straw'], evenodd=True)
    q = cont(hi & ~(LB == 2));  cel.fill(q, C['hi'], .75, evenodd=True) if q else None
    q = cont(shade & (LB != 2)); cel.fill(q, C['sh'], evenodd=True) if q else None
    q = cont(deep & ((LB == 1) | (LB == 3))); cel.fill(q, C['dk'], .62, evenodd=True) if q else None
    q = cont((LB == 4) & (UA < .40 + .07 * np.sin(VA * 4 + 1))); cel.fill(q, C['under_dk'], evenodd=True) if q else None
    if rb:
        cel.fill(rb, C['rib'], evenodd=True)
        q = cont((LB == 2) & shade); cel.fill(q, C['rib_sh'], evenodd=True) if q else None
        q = cont((LB == 2) & (ndl > .70) & (UA > .30) & (UA < .62) & (np.sin(VA * 9) > -.55)); cel.fill(q, C['rib_hi'], .85, evenodd=True) if q else None
    # ---- weave
    def samp(A, P):
        i = np.clip(np.round((P - off) * k).astype(int), 0, [ww - 1, hh - 1]); return A[i[..., 1], i[..., 0]]
    def vis(P, d, tol=.022): return d >= samp(Z, P) - tol
    cell = max(9., w * .0165); lwi = lw * .34; small = w < 260
    nC, nB = (5, 4) if small else (10, 8)
    vs = np.linspace(VR + .03, .965, nC); ts = np.linspace(.07, .955, nB)
    A0, A1, T0, T1, FN, KD = [], [], [], [], [], []
    for fi, (fn, us, du) in enumerate(((crown, vs, (vs[1] - vs[0])), (br, ts, ts[1] - ts[0]))):
        for j, u in enumerate(us):
            rad = float(np.hypot(*fn(np.array(u), np.array(.3))[[0, 2]])); n = max(10, int(2 * math.pi * rad * s / (cell * (1.0 if fi == 0 else 1.15))))
            t0 = np.linspace(-math.pi, math.pi, n, endpoint=False) + r.uniform(0, 1) + r.normal(0, .12 / n, n); dt = 2 * math.pi / n
            ln_ = r.uniform(.9, 2.3, n) * dt                                       # dash along the coil
            A0.append(np.full(n, u)); A1.append(u + r.normal(0, du * .05, n)); T0.append(t0); T1.append(t0 + ln_); FN.append(np.full(n, fi)); KD.append(np.zeros(n))
            m = r.random(n) < (.22 if small else .42); q = int(m.sum()); sgn = np.where(np.arange(q) % 2 == 0, 1, -1)
            A0.append(np.full(q, u + du * .16)); A1.append(u + du * r.uniform(.55, .86, q)); T0.append(t0[m]); T1.append(t0[m] + sgn * dt * r.uniform(.25, .55, q)); FN.append(np.full(q, fi)); KD.append(np.ones(q))
            if not small:
                m = r.random(n) < .30; q = int(m.sum())
                A0.append(np.full(q, u + du * .5)); A1.append(np.full(q, u + du * .5)); T0.append(t0[m]); T1.append(t0[m] + dt * r.uniform(.6, 1.1, q)); FN.append(np.full(q, fi)); KD.append(np.full(q, 2))
    A0, A1, T0, T1, FN, KD = [np.concatenate(a) for a in (A0, A1, T0, T1, FN, KD)]
    t = np.linspace(0, 1, 6)[None]; UU = A0[:, None] * (1 - t) + A1[:, None] * t; TT = T0[:, None] * (1 - t) + T1[:, None] * t
    UU = np.where(FN[:, None] == 0, np.minimum(UU, .995), np.minimum(UU, 1.0))
    QP = np.where((FN == 0)[:, None, None], crown(UU, TT), br(UU, TT)); P, d = pr(QP)
    mid = P[:, 3]; lab = samp(LB, mid); ok = vis(mid, d[:, 3]) & (((FN == 0) & (lab == 1)) | ((FN == 1) & ((lab == 3) | (lab == 4)))) & vis(P[:, 0], d[:, 0], .04) & vis(P[:, 5], d[:, 5], .04)
    ishade = samp(shade.astype(np.uint8), mid) > 0; iun = lab == 4; rr = r.random(len(mid))
    keep = np.where(ishade | iun, .92, .72)
    wd = lwi * r.uniform(.65, 1.35, len(mid)) * np.where(KD == 1, .85, 1.0)
    for sel, col, al in (((KD < 2) & ~ishade & ~iun, 'weave', .92), ((KD < 2) & (ishade | iun), 'weave_sh', .9)):
        m = ok & sel & (rr < keep)
        if m.any(): cel.fill(_sliver(P[m], wd[m]), C[col], al)
    m = ok & (KD == 2) & ~iun & (rr < .9)
    if m.any():
        a = m & ~ishade; cel.fill(_sliver(P[a], wd[a] * 2.2), C['weave_hi'], .70) if a.any() else None
        a = m & ishade; cel.fill(_sliver(P[a], wd[a] * 2.0), C['straw'], .30) if a.any() else None
    if LT.get('leak', 0) > 0:                                                          # back light leaking between the coils of the near brim
        m = ok & (KD == 2) & ~iun & (FN == 1) & (rr < .62 * LT['leak'])
        if m.any(): cel.fill(_sliver(P[m], wd[m] * 2.6), '#fff6cf', .95)
    if iun.any() and LT['glow'] > 0:                                                  # gold through the weave
        m = ok & iun & (KD == 2) & (rr < .75)
        if m.any(): cel.fill(_sliver(P[m], wd[m] * 2.0), np.clip(C['hi'] * 1.15, 0, 1), min(1., LT['glow'] * 1.3))
    # ---- slash + stitches
    if wear > .34:
        a = np.linspace(0, 1, 14); tt = .34 + .66 * a; thh = -.62 + .10 * a + .02 * np.sin(a * 9); op = .004 + .045 * np.clip(a - .55, 0, 1) ** 1.4
        c0, d0 = pr(br(tt, thh - op)); c1, _ = pr(br(tt, thh + op)); v = vis(c0, d0, .03)
        for a_, b_ in _runs(v):
            if b_ - a_ > 3:
                cel.fill(np.vstack([c0[a_:b_], c1[a_:b_][::-1]]), C['under_dk'] * .45); cel.line(c0[a_:b_], lw * .62, C['line'], taper=(.5, .02)); cel.line(c1[a_:b_], lw * .42, C['line'], taper=(.5, .02))
        for q in np.linspace(.10, .80, 6):
            i = int(q * 13); t_, h_ = tt[i], thh[i]; ax = np.linspace(-1, 1, 6)
            p, d_ = pr(br(t_ + .035 * ax * (1 if int(q * 100) % 2 else -1) * .5, h_ + .062 * ax))
            if vis(p, d_, .03).all(): cel.line(p, lw * .85, C['line'], taper=(.1, .1), tmin=.5); cel.line(p - [0, lw * .12], lw * .34, C['weave_hi'], taper=(.2, .2))
    # ---- frayed straws on the edge
    n = int((14 if small else 22) + 34 * wear); tf = r.uniform(-math.pi, math.pi, n); lf = r.uniform(.006, .026, n) * (.7 + .7 * wear); sk = r.normal(0, .05, n)
    u0 = r.uniform(.93, .99, n); t5 = np.linspace(0, 1, 5)[None]
    P5, d5 = pr(br(u0[:, None] + (1 - u0[:, None] + lf[:, None] / (1 - RB)) * t5, tf[:, None] + sk[:, None] * t5)); v = vis(P5[:, 0], d5[:, 0], .05)
    if v.any():
        cel.fill(_sliver(P5[v], np.full(v.sum(), lw * .95)), C['line']); lt = v & (r.random(n) < .9)
        cel.fill(_sliver(P5[lt], np.full(lt.sum(), lw * .55)), C['straw'])
    # ---- ink
    ink = C['line']
    def curve(fn, u, w_, amt=1., brk=0, inner=False, tol=.03, sd=0):
        t = np.linspace(-math.pi, math.pi, 400); p, d = pr(fn(np.full(400, u), t)); v = vis(p, d, tol)
        if inner: v &= samp(dist, p) > lw * .9 * k
        if brk: v &= noise1(400, brk, np.random.default_rng(seed + sd + 3)) > -.75 + (1 - amt) * 1.2
        for a, b in _runs(v):
            if b - a > 5: cel.line(p[a:b], w_, ink, taper=(.14, .14), press=.30, seed=seed + a + sd, tremor=.2)
    dist = cv2.distanceTransform(sil.astype(np.uint8), cv2.DIST_L2, 3)
    curve(br, .945, lw * .34, .75, 16, sd=11)
    curve(ribbon_s, 1.0, lw * .62, sd=1); curve(br, .03, lw * .62, sd=2)
    if not small: curve(ribbon_s, .84, lw * .22, .6, 22, sd=5); curve(ribbon_s, .16, lw * .22, .6, 22, sd=6)
    curve(br, 1.0, lw * .9, inner=True, sd=4)
    if cr:                                          # crown over brim
        for q in cr:
            m = (samp(dist, q) > lw * 1.1 * k)
            if len(q) > 40:
                for a, b in _runs(m):
                    if b - a > 8: cel.line(q[a:b], lw * .85, ink, taper=(.12, .12), press=.3, seed=seed + a)
    for q in sorted(S, key=len)[-1:]: outline(cel, q, lw, ink, seed, press=.45, breaks=0 if small else 3)
    # ---- light passes on the ROI
    out = cel.rgba()
    if LT['para']:
        g = LT['rd']; p0 = (cx + g[0] * s - x0, cy + g[1] * s - y0); p1 = (cx - g[0] * s - x0, cy - g[1] * s - y0); out = para(out, p0, p1, LT['para'])
    if rim > 0:
        g = np.array(LT['rd'], float); g /= np.hypot(*g); wr = max(2., w * .0105) * LT['rw']; al = out[..., 3]
        def band(q, so):
            m = cv2.warpAffine(al, np.float32([[1, 0, -g[0] * q], [0, 1, -g[1] * q]]), (al.shape[1], al.shape[0])); m = np.clip(al - m, 0, 1)
            return cv2.GaussianBlur(m, (0, 0), so) * (al > .02) if so else m
        m = np.clip(band(wr, .7) + .42 * band(wr * 2.6, wr * .9), 0, 1)[..., None] * rim * .92
        out[..., :3] = out[..., :3] * (1 - m) + C['rim'] * (m * al[..., None])
    # ---- front mask: crown, ribbon and the near half of the brim
    dm = np.zeros((hh, ww), f32); near = (LB == 1) | (LB == 2); zz = (PX @ R.T)[..., 2]; bx = np.array([0, Y0, 0]) @ R.T
    near |= ((LB == 3) | (LB == 4)) & (zz > bx[2] - .06); dm[near] = 1
    # frayed straws / outline live just outside the masks: grow the near region into empty pixels
    dn = cv2.distanceTransform((~near).astype(np.uint8), cv2.DIST_L2, 3); far = (LB > 0) & ~near; df = cv2.distanceTransform((~far).astype(np.uint8), cv2.DIST_L2, 3) if far.any() else dn * 0 + 1e9; dm = np.maximum(dm, (LB == 0) & (dn < df - 1.0) & (dn < (w * .06 + lw) * k + 3))
    dm = cv2.GaussianBlur(cv2.resize(dm, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR), (0, 0), .8)
    return x0, y0, out, dm


def hat(cx, cy, w, yaw=0., pitch=0., roll=0., light='day', wear=.6, bounce=0., part='all', f=0, tilt=0., lean=0., wind=.4, rim=1., lw=None, seed=7):
    lw = lw or float(np.clip(w / 900 * 5.6, 1.5, 7.5))
    key = (round(cx, 2), round(cy, 2), round(w, 2), round(yaw, 4), round(pitch, 4), round(roll, 4), light if isinstance(light, str) else id(light), round(wear, 3), round(bounce, 4), f if wind else 0, round(tilt, 4), round(lean, 4), wind, rim, lw, seed)
    g = _cache.get(key)
    if g is None:
        if len(_cache) > 6: _cache.clear()
        g = _cache[key] = _draw(cx, cy, w, yaw, pitch, roll, light, wear, bounce, f, tilt, lean, wind, rim, lw, seed) or 0
    out = np.zeros((H, W, 4), f32)
    if g:
        x0, y0, a, dm = g; paste(out, a if part != 'front' else a * dm[..., None], x0, y0)
    return out


def seat(Hd, w=3.0, tilt=0., lean=0., up=0., fwd=0.):
    """where the hat sits on an op.head3.Head (already posed).  w = brim width in head units, up = lift in head units"""
    h = np.array([[0, .44 + up - Y0 * w / 2, -.04 + fwd]]); p = Hd.P3(h)[0]; k = 1 / (1 - float(Hd.depth(h)[0]) * Hd.persp)
    return dict(cx=float(p[0]), cy=float(p[1]), w=w * Hd.s * k, yaw=Hd.yaw, pitch=Hd.pitch, roll=Hd.roll, tilt=tilt, lean=lean)


def settle(f, f0, amp=.5, per=7., dec=4.5):
    """damped flop of the brim after a hit at f0"""
    t = f - f0
    return 0. if t < 0 else amp * math.exp(-t / dec) * math.cos(2 * math.pi * t / per)


# ------------------------------------------------------------------ looking out from under it
_UB = {}


def _under_plate():
    if _UB: return _UB
    PW, PH = 2360, 1500; r = np.random.default_rng(41); EY = 1380                  # brim edge at its lowest, plate px
    x = np.arange(PW); u = (x - PW / 2) / (PW / 2)
    edge = EY - 190 * u * u + 22 * np.sin(u * 3.1 + .9) + 8 * np.sin(u * 8.3 + 1) + noise1(PW, 60, r) * 2.0
    yy, xx = np.mgrid[0:PH, 0:PW].astype(f32); d = edge[None].astype(f32) - yy       # distance above the edge
    t = np.clip(d / 1300, 0, 1)
    col = ramp(t, [(0, '#8a5420'), (.035, '#4a2a12'), (.12, '#2b180d'), (.45, '#180d0a'), (1, '#0b0608')])
    col *= (1 + .10 * fbm(PH, PW, (60, 18, 6), seed=5))[..., None]
    cel = Cel(PW, PH, ss=1); cel.ctx.set_antialias(3)
    rings = [0]; q = 26.
    while rings[-1] < 1500: rings.append(rings[-1] + q); q *= 1.075
    holes = []
    def E(xs): return np.interp(xs, x, edge)
    for j, (a, b) in enumerate(zip(rings[:-1], rings[1:])):
        h = b - a; n = int(PW / (h * .95)); xs = np.linspace(0, PW, n) + r.normal(0, h * .06, n) + (j % 2) * h * .45; k = np.linspace(0, 1, 6)
        seg = [np.c_[x0 + k * h * q_, E(x0 + k * h * q_) - a - h * .05 * np.sin(k * 3.14)] for x0, q_ in zip(xs, r.uniform(.9, 2.2, n))]
        cel.fill(_sliver(np.array(seg), r.uniform(.06, .12, n) * h + 1), '#080406', .85)
        sg = np.where(np.arange(n) % 2 == 0, 1, -1)
        tick = [np.c_[x0 + g * h * .42 * k, E(x0) - a - h * (.16 + .7 * k)] for x0, g in zip(xs, sg)]
        m = r.random(n) < .55; cel.fill(_sliver(np.array(tick)[m], np.full(m.sum(), .07 * h + .8)), '#0a0507', .6)
        lit = [np.c_[x0 + g * h * .42 * k + h * .16, E(x0) - a - h * (.18 + .6 * k)] for x0, g in zip(xs, sg)]
        m = r.random(n) < .45; cel.fill(_sliver(np.array(lit)[m], np.full(m.sum(), .13 * h)), '#7a4a1e', max(.08, .42 - a / 900))
        band = [np.c_[x0 + k * h * 1.4, E(x0 + k * h * 1.4) - a - h * .62] for x0 in xs[r.random(n) < .5]]; cel.fill(_sliver(np.array(band), np.full(len(band), .30 * h)), '#6b3f1a', max(.06, .30 - a / 1200)) if band else None
        for x0 in xs:
            if r.random() < .15: holes.append((x0 + r.normal(0, h * .1), E(x0) - a - r.uniform(-.05, .1) * h, h * r.uniform(.035, .075) + .6, r.uniform(0, 6.28), r.uniform(.4, 1.) ** 2))
    g = cel.rgba(); col = col * (1 - g[..., 3:4]) + g[..., :3]
    a = np.clip(d + .5, 0, 1)
    fr = Cel(PW, PH, ss=1); n = 150; xs = r.uniform(0, PW, n); k = np.linspace(0, 1, 5); L = 4 + 30 * r.random(n) ** 2.2; sk = r.normal(0, 13, n)
    st = np.array([np.c_[x0 + s_ * k, E(x0) - 10 + (l_ + 10) * k] for x0, l_, s_ in zip(xs, L, sk)])
    fr.fill(_sliver(st, r.uniform(1.6, 4.0, n)), '#2a150b'); fr.fill(_sliver(st[::2] + [1.0, 0], np.full(len(st[::2]), 1.7)), '#f2b552', .95); q = fr.rgba()
    lay = np.dstack([col * a[..., None], a]).astype(f32); lay = lay * (1 - q[..., 3:4]) + q
    er = np.exp(-np.abs(d - 3) / 3.2) * (d > -1) * (.65 + .35 * np.clip(fbm(PH, PW, (34, 9), seed=8), -1, 1)); lay[..., :3] += er[..., None] * rgb('#ffc765') * .75 * lay[..., 3:4]
    tr = np.exp(-np.clip(d, 0, None) / 46) * (d > 0); lay[..., :3] += tr[..., None] * rgb('#c9701e') * .22
    _UB.update(lay=lay, edge=edge, EY=EY, PW=PW, PH=PH, holes=np.array(holes))
    return _UB


def under_brim(f, amt, stars=1., sun=(960, 300), sway=1.):
    """dark weave along the top of frame, pin-hole light stars.  Premult layer; the glows are additive (rgb > alpha)."""
    if amt <= 0: return np.zeros((H, W, 4), f32)
    u = _under_plate(); ey = -70 + 430 * amt + sway * 5 * math.sin(f * .11); ang = sway * .006 * math.sin(f * .07 + 1); z = 1.0 + .05 * max(0, amt - 1)
    c, s = math.cos(ang) * z, math.sin(ang) * z; tx = W / 2 + sway * 7 * math.sin(f * .083); M = np.float32([[c, -s, 0], [s, c, 0]])
    M[:, 2] = np.array([tx, ey]) - M[:, :2] @ np.array([u['PW'] / 2, u['EY']])
    out = cv2.warpAffine(u['lay'], M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    if ey > u['EY'] * z: out[:int(ey - u['EY'] * z) + 2] = np.r_[rgb('#0b0608'), 1]
    if stars > 0:
        hp = u['holes'][:, :2] @ M[:, :2].T + M[:, 2]; gl = np.zeros((H, W, 3), f32)
        for (x, y), (_, _, rad, ph, b) in zip(hp, u['holes']):
            if not (-20 < x < W + 20 and -20 < y < H + 20): continue
            dist = math.hypot(x - sun[0], y - sun[1]); k = stars * b * (.55 + .45 * math.sin(f * .31 + ph)) * (.35 + 1.1 * math.exp(-dist / 520)); rr = rad * 1.2 + .8
            if k < .05: continue
            R = int(min(90, 10 + rr * 11 * (.5 + k))); xi, yi = int(x), int(y); xa, xb, ya, yb = max(0, xi - R), min(W, xi + R + 1), max(0, yi - R), min(H, yi + R + 1)
            yy, xx = np.mgrid[ya:yb, xa:xb].astype(f32); dx, dy = xx - x, yy - y; d = np.sqrt(dx * dx + dy * dy)
            g = 1.6 * np.clip(1.2 - d / rr, 0, 1) + .34 * np.exp(-d / (rr * 2.1))
            if k > .42: q = rr * 5.5 * k; g = g + .8 * (np.exp(-np.abs(dx) / (.5 + rr * .16)) * np.exp(-np.abs(dy) / q) + np.exp(-np.abs(dy) / (.5 + rr * .16)) * np.exp(-np.abs(dx) / q))
            fall = np.clip(1 - d / R, 0, 1); gl[ya:yb, xa:xb] += (g * fall * k)[..., None] * rgb('#ffd98c')
        out[..., :3] += gl * np.clip(out[..., 3:4] * 1.5, 0, 1)
    return out


if __name__ == '__main__':
    import sys
    from .comp import over, save
    base = np.ones((H, W, 3), f32) * rgb('#b9c3cf')
    for i, (ya, pi, ro, li, b, ti) in enumerate([(0, 0, 0, 'day', 0, 0), (.5, .0, .1, 'dusk', 0, 0), (.28, .18, -.08, 'gold', 0, .1), (0, -.5, 0, 'storm', 0, 0), (0, .7, 0, 'sea', 0, 0), (-.8, -.2, -.2, 'fire', .6, 0), (0, 0, 0, 'lamp', 0, -.35), (1.5, .1, 0, 'night', -.5, 0), (2.6, -.1, .1, 'charcoal', 0, 0)]):
        base = over(base, hat(330 + 630 * (i % 3), 200 + 350 * (i // 3), 540, ya, pi, ro, li, wear=.2 + .1 * i, bounce=b, tilt=ti))
    save(sys.argv[1], (np.clip(base, 0, 1) * 255).astype(np.uint8) if False else base)
    b2 = np.ones((H, W, 3), f32) * rgb('#f4ddb0'); b2 = over(b2, under_brim(10, .8)); save(sys.argv[1].replace('.jpg', '_ub.jpg'), b2)
