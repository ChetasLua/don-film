"""The azure dragon: a true 3-D spline body drawn as a cel (scales in rows, belly plates, mane, claws, whiskers with lag, horned head, jaw, flame clouds).

    D = Dragon(ctrl, R=6.)                     # ctrl (k,3) world points, head first; R = body radius in metres
    o = D.render(cam, f, jaw=.6, furn=1., burn=0., wind=(..), lit=LIT['storm'], legs=True, ss=1.5)    # -> dict layer, light, iz (1/depth, 1x), pts (mouth eyes nose horn px)
    head(cx, cy, s, yaw, pitch, roll, jaw=.5, furn=0., ...) -> layer      (s = px per skull unit, the head is about 4.4 units long)
    body(ctrl, cam, f=0, **k) -> layer                flame_cloud(...) -> small RGBA sprite        sprite(base_layer...) z-tested paste
Everything is meshes rasterised here (numba z-buffer), shaded in hard tones, inked from depth / part / crease edges with broken, weighted lines."""
import numpy as np, cv2, math, os
from numba import njit, prange
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp
from op.ink import Cel, spl, resample, paste
from op.cam import Camera3D
from op.fx_core import h1, R as RNG, noise, FX, band, leaf, tongue, lin01, ink_broken, glow_poly
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INK = '#14111c'

# ---------------------------------------------------------------- materials
M_BODY, M_MANE, M_HORN, M_CLAW, M_SKIN, M_JAW, M_TONGUE, M_EYE, M_WHISK, M_LEG, M_FIN, M_FIRE, M_GUM = range(1, 14)
#        light      base       shade      deep
BASE = {M_BODY: ('#7fd0dc', '#2f86b4', '#1d4f86', '#122b58'), 0: ('#f6ead0', '#e2d2b2', '#a98d86', '#5f4d66'), M_MANE: ('#6b6fae', '#2b2d5a', '#191a3a', '#0e0e22'),
        M_HORN: ('#f7dc9a', '#d9a552', '#97622f', '#543020'), M_CLAW: ('#fff6df', '#ecdcb8', '#b08f72', '#6a4c4a'), M_SKIN: ('#86d6df', '#3089b6', '#1d5088', '#122b58'),
        M_JAW: ('#86d6df', '#3089b6', '#1d5088', '#122b58'), M_TONGUE: ('#f2838a', '#cf4a5c', '#8c2440', '#4e1028'), M_EYE: ('#fff6c8', '#ffe27a', '#e09a2e', '#8c4a1a'),
        M_WHISK: ('#fbe3a4', '#dcae5c', '#9a6a38', '#5a3a26'), M_LEG: ('#74c6d4', '#2b78a6', '#1a4779', '#10264f'), M_FIN: ('#5fb8cc', '#245f96', '#173a6c', '#0f2145'),
        M_FIRE: ('#fff2b4', '#ffc23a', '#f4701a', '#b3200e'), M_GUM: ('#c4455a', '#8c1c2c', '#560d1e', '#2a0612')}


def palette(mul=(1, 1, 1), sh=(1, 1, 1), mix=None, over=None):
    P = np.zeros((16, 4, 3), f32)
    for m, t in BASE.items():
        for i, c in enumerate((over or {}).get(m, t)):
            q = rgb(c) * np.array(mul if i < 2 else sh, f32)
            P[m, i] = q if mix is None else q * (1 - mix[1]) + rgb(mix[0]) * mix[1]
    return P


LIT = dict(
    storm=dict(pal=palette((.72, .80, .92), (.60, .62, .86)), L=(-.45, -.35, .82), rim='#e6f4ff', rd=(-1., -.75), rw=9., line='#0c0c18', sky='#26386a', amb=.0),
    day=dict(pal=palette(), L=(-.5, -.5, .7), rim='#fff6dc', rd=(-1., -.8), rw=9., line='#17131a', sky='#7aa0c8', amb=.0),
    fire=dict(pal=palette((.95, .80, .72), (.70, .46, .56)), L=(.55, -.45, .40), rim='#ffc766', rd=(1., .5), rw=11., line='#1a0a10', sky='#5a2240', amb=.0),
    dawn=dict(pal=palette((.80, .70, .74), (.62, .46, .62)), L=(.2, .6, .6), rim='#ffd48a', rd=(.3, -1.), rw=10., line='#1a1020', sky='#7a4a78', amb=.0))


# ---------------------------------------------------------------- raster
@njit(fastmath=True, cache=True, nogil=True)
def _raster(X, Y, Z, F, zb, tri, b0, b1, near):
    h, w = zb.shape
    for t in range(F.shape[0]):
        i, j, k = F[t, 0], F[t, 1], F[t, 2]
        z0, z1, z2 = Z[i], Z[j], Z[k]
        if z0 < near or z1 < near or z2 < near: continue
        x0, y0, x1, y1, x2, y2 = X[i], Y[i], X[j], Y[j], X[k], Y[k]
        xa = min(x0, x1, x2); xb = max(x0, x1, x2); ya = min(y0, y1, y2); yb = max(y0, y1, y2)
        if xb < 0 or yb < 0 or xa > w or ya > h: continue
        A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(A) < 1e-7: continue
        ia = 1. / A; i0 = 1. / z0; i1 = 1. / z1; i2 = 1. / z2
        ix0 = max(0, int(math.floor(xa))); ix1 = min(w - 1, int(math.ceil(xb))); iy0 = max(0, int(math.floor(ya))); iy1 = min(h - 1, int(math.ceil(yb)))
        d0x = (y1 - y2) * ia; d1x = (y2 - y0) * ia
        for y in range(iy0, iy1 + 1):
            py = y + .5; px = ix0 + .5
            w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * ia; w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * ia
            for x in range(ix0, ix1 + 1):
                w2 = 1. - w0 - w1
                if w0 >= -1e-5 and w1 >= -1e-5 and w2 >= -1e-5:
                    q = w0 * i0 + w1 * i1 + w2 * i2
                    if q > zb[y, x]:
                        zb[y, x] = q; tri[y, x] = t; b0[y, x] = w0 * i0 / q; b1[y, x] = w1 * i1 / q
                w0 += d0x; w1 += d1x


@njit(fastmath=True, cache=True, nogil=True)
def _raster_o(X, Y, Z, F, zb):
    h, w = zb.shape
    for t in range(F.shape[0]):
        i, j, k = F[t, 0], F[t, 1], F[t, 2]
        x0, y0, x1, y1, x2, y2 = X[i], Y[i], X[j], Y[j], X[k], Y[k]
        xa = min(x0, x1, x2); xb = max(x0, x1, x2); ya = min(y0, y1, y2); yb = max(y0, y1, y2)
        if xb < 0 or yb < 0 or xa > w or ya > h: continue
        A = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(A) < 1e-7: continue
        ia = 1. / A
        for y in range(max(0, int(ya)), min(h - 1, int(yb) + 1) + 1):
            py = y + .5
            for x in range(max(0, int(xa)), min(w - 1, int(xb) + 1) + 1):
                px = x + .5
                w0 = ((x1 - px) * (y2 - py) - (x2 - px) * (y1 - py)) * ia; w1 = ((x2 - px) * (y0 - py) - (x0 - px) * (y2 - py)) * ia; w2 = 1. - w0 - w1
                if w0 >= -1e-4 and w1 >= -1e-4 and w2 >= -1e-4:
                    q = w0 * Z[i] + w1 * Z[j] + w2 * Z[k]
                    if q > zb[y, x]: zb[y, x] = q


@njit(fastmath=True, inline='always')
def _hs(a, b, c):
    n = (a * 374761393 + b * 668265263 + c * 2147483647) & 0x7fffffff
    n = ((n ^ (n >> 13)) * 1274126177) & 0x7fffffff
    n = (n ^ (n >> 16)) & 0xffffff
    return n / 16777216.


@njit(fastmath=True)
def _n3(x, y, z):
    ix, iy, iz = int(math.floor(x)), int(math.floor(y)), int(math.floor(z)); fx, fy, fz = x - ix, y - iy, z - iz
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz)
    r = 0.
    for k in range(2):
        for j in range(2):
            for i in range(2):
                r += (fx if i else 1 - fx) * (fy if j else 1 - fy) * (fz if k else 1 - fz) * _hs(ix + i, iy + j, iz + k)
    return r * 2 - 1


@njit(fastmath=True, inline='always')
def _ss(a, b, x):
    t = (x - a) / (b - a)
    t = 0. if t < 0 else 1. if t > 1 else t
    return t * t * (3 - 2 * t)


@njit(parallel=True, fastmath=True, cache=True)
def _shade(tri, b0, b1, iz, F, VN, VA, VP, FM, FP, eye, L, PAL, FIRE, K, SM, SMM, mouth, out, pid, vis, pat, em, nrm):
    """hard tones, scales, plates, hair strands, fire.  K: 0 px per metre*z, 1 R, 2 time, 3 burn, 4 scales round, 5 wet, 6 furnace, 7 terminator, 8 hatch band, 9 pattern line px, 10 shadow on, 11 fire ink, 12 scale lod lo"""
    h, w = tri.shape
    foc, R_, tm, burn, N_, wet, furn, th1, hb, lwp, shon = K[0], K[1], K[2], K[3], K[4], K[5], K[6], K[7], K[8], K[9], K[10]
    ls = 6.2831853 / N_ * 1.02; TB = 2.36; nsm = SM.shape[0]
    for y in prange(h):
        for x in range(w):
            t = tri[y, x]
            if t < 0: pid[y, x] = -1; continue
            i, j, k = F[t, 0], F[t, 1], F[t, 2]; u0 = b0[y, x]; u1 = b1[y, x]; u2 = 1 - u0 - u1
            nx = u0 * VN[i, 0] + u1 * VN[j, 0] + u2 * VN[k, 0]; ny = u0 * VN[i, 1] + u1 * VN[j, 1] + u2 * VN[k, 1]; nz = u0 * VN[i, 2] + u1 * VN[j, 2] + u2 * VN[k, 2]
            px = u0 * VP[i, 0] + u1 * VP[j, 0] + u2 * VP[k, 0]; py = u0 * VP[i, 1] + u1 * VP[j, 1] + u2 * VP[k, 1]; pz = u0 * VP[i, 2] + u1 * VP[j, 2] + u2 * VP[k, 2]
            a0 = u0 * VA[i, 0] + u1 * VA[j, 0] + u2 * VA[k, 0]; a1 = u0 * VA[i, 1] + u1 * VA[j, 1] + u2 * VA[k, 1]; a2 = u0 * VA[i, 2] + u1 * VA[j, 2] + u2 * VA[k, 2]
            a3 = u0 * VA[i, 3] + u1 * VA[j, 3] + u2 * VA[k, 3]; a4 = u0 * VA[i, 4] + u1 * VA[j, 4] + u2 * VA[k, 4]; a5 = u0 * VA[i, 5] + u1 * VA[j, 5] + u2 * VA[k, 5]
            l = math.sqrt(nx * nx + ny * ny + nz * nz) + 1e-9; nx /= l; ny /= l; nz /= l
            vx, vy, vz = eye[0] - px, eye[1] - py, eye[2] - pz; l = math.sqrt(vx * vx + vy * vy + vz * vz) + 1e-9; vx /= l; vy /= l; vz /= l
            fc = nx * vx + ny * vy + nz * vz
            if fc < 0: nx, ny, nz, fc = -nx, -ny, -nz, -fc
            nrm[y, x, 0] = nx; nrm[y, x, 1] = ny; nrm[y, x, 2] = nz
            m = FM[t]; pid[y, x] = FP[t]; z = 1. / iz[y, x]; ppm = foc / z; un = ppm * R_
            v = nx * L[0] + ny * L[1] + nz * L[2]
            v += .10 * _n3(px / R_ * 1.7, py / R_ * 1.7, pz / R_ * 1.7)
            sh = 0.
            if shon > 0 and m != 2 and m != 12:
                sx = SMM[0, 0] * px + SMM[0, 1] * py + SMM[0, 2] * pz + SMM[0, 3]; sy = SMM[1, 0] * px + SMM[1, 1] * py + SMM[1, 2] * pz + SMM[1, 3]
                sd = SMM[2, 0] * px + SMM[2, 1] * py + SMM[2, 2] * pz + SMM[2, 3] + R_ * .09 + R_ * .10 * (1 - max(0., v))
                cnt = 0.
                for q in range(4):
                    ix = int(sx + (q % 2) * 1.0 - .5); iy = int(sy + (q // 2) * 1.0 - .5)
                    if 0 <= ix < nsm and 0 <= iy < nsm and SM[iy, ix] > sd: cnt += .25
                sh = _ss(.3, .7, cnt)
            v0 = v; ln = 0.; hc = 0.; belly = False; mm = m; lift = 0.; dk = 0.; tint = 1.; gl = 0.
            lod = _ss(5., 15., ls * un)
            if m == 1 or m == 10:                                  # scaled body / legs
                s, th, uu = a0, a1, a2
                Nn = N_ if m == 1 else N_ * .5; lss = 6.2831853 / Nn * 1.02 * (1. if m == 1 else .62)
                if m == 1 and abs(th) > TB + .05 * math.sin(s * 5.):
                    belly = True; q = (3.14159 - abs(th)) / (3.14159 - TB); lb = .46
                    b = (s - .10 * (1 - q * q)) / lb; fr = b - math.floor(b)
                    e = min(fr, 1 - fr) * lb * un; ln = _ss(lwp * 1.0 + .8, lwp * 1.0 - .8, e) * lod
                    v += (.45 - fr) * .34 * lod
                    if fr < .16: dk = .9 * lod
                    gl2 = _ss(.5, .56, q) * 0.; e2 = (abs(th) - TB) * un; ln = max(ln, _ss(lwp * 1.1 + .8, lwp * 1.1 - .8, abs(e2)) * lod)
                    # fine plate grain
                    g = abs((th * 9. + .4 * _n3(s * 3., th * 4., 1.)) % 1. - .5) * 2
                    if g < .10 and fr > .3 and _n3(s * 2.2, th * 3., 7.) > .15: ln = max(ln, .30 * _ss(14., 40., lb * un))
                else:
                    a = th / 6.2831853 * Nn + 4 * Nn; b = s / lss + 4000.
                    rb = int(math.floor(b)); r0 = rb - 1; off = .5 * (r0 & 1); c0 = int(math.floor(a - off)); x0 = (a - off - c0 - .5) * 2; y0 = b - r0
                    ye0 = 1.02 + .56 * math.sqrt(max(0., 1 - x0 * x0)) + .05 * (_hs(r0, c0, 5) - .5)
                    if y0 < ye0: row, col, xx, yy, ye, un_ = r0, c0, x0, y0, ye0, 9.
                    else:
                        row = rb; off = .5 * (rb & 1); col = int(math.floor(a - off)); xx = (a - off - col - .5) * 2; yy = b - rb; ye = 1.02 + .56 * math.sqrt(max(0., 1 - xx * xx)); un_ = y0 - ye0
                    hh = _hs(row, col, 1); pxs = lss * un
                    ep = abs(y0 - ye0) * pxs
                    thick = lwp * (.75 + .6 * _hs(r0, c0, 2)) * (1.25 - .5 * abs(x0))
                    cov = 1.
                    if v > .50 and hh < .52: cov = .16
                    if un_ < 9.:
                        sm = (1 - abs(xx)) * .5 * 6.2831853 / Nn * un
                        if yy > .5: ln = _ss(lwp * .8 + .8, lwp * .8 - .8, sm) * .8
                    ln = max(ln, _ss(thick + .8, thick - .8, ep) * cov) * lod
                    v += ((yy / ye - .5) * .20 + (hh - .5) * .20) * lod
                    if un_ < .17: dk = lod * .85
                    dl = ye - yy
                    if .07 < dl < .26 and v > .05 and hh > .25: lift = .55 * lod
                    if abs(xx) < .05 and .62 < yy < ye - .18 and pxs > 26: ln = max(ln, .55 * _ss(26., 60., pxs))
                    if pxs > 70:                                   # growth rings + chips
                        for q in range(3):
                            g = abs(dl - (.30 + .11 * q)) * pxs
                            if hh > .35 + .2 * q and abs(xx) < .72 - .16 * q: ln = max(ln, _ss(lwp * .5 + .7, lwp * .5 - .7, g) * .45 * _ss(70., 130., pxs))
                    tint = 1 + (hh - .5) * .22 * lod
                    if wet > 0 and lod > .5 and v > .18 and hh > .45:
                        gx = (xx + .30) / .30; gy = (yy - ye * .62) / .17
                        if gx * gx + gy * gy < 1. and gx + gy * .5 < .55: gl = wet * lod
                    if m == 10 and abs(th) < .75:                  # shin / toe plates
                        b = (s + .10 * math.cos(th * 2)) / .17; fr = b - math.floor(b); ln = _ss(lwp + .8, lwp - .8, min(fr, 1 - fr) * .17 * un) * lod; dk = .8 * lod if fr < .2 else 0.; lift = 0.; tint = 1.; v = v0 + (.5 - fr) * .3
                hc = s * 5.5 + th * .6
            elif m == 2 or m == 12:                                # hair / fire ribbons: a0 across -1..1, a1 along, a2 seed, a3 aspect
                v = .30 - .75 * a0 + .25 * math.sin(a1 * 5. + a2 * 7.) * (1 if a2 > .5 else -1)
                wpx = a3 * un * .5 * (1 - a1 * .9)
                for q in range(3):
                    c = (-.45, .12, .58)[q] + .10 * math.sin(a1 * 6. + a2 * 9. + q)
                    tl = .12 + .25 * _hs(int(a2 * 999), q, 3); th_ = .55 + .42 * _hs(int(a2 * 999), q, 4)
                    if tl < a1 < th_: ln = max(ln, _ss(lwp * .62 * (1 - (a1 - tl) / (th_ - tl)) + .8, lwp * .62 * (1 - (a1 - tl) / (th_ - tl)) - .8, abs(a0 - c) * wpx))
                ln *= _ss(7., 18., a3 * un)
            elif m == 3:                                           # horns: rings
                b = a0 * 5.2 + .25 * math.sin(a1 * 3.); fr = b - math.floor(b); v += (.5 - fr) * .35
                if a0 < 2.2: ln = _ss(lwp + .8, lwp - .8, min(fr, 1 - fr) / 5.2 * un) * .9
                if fr < .15: dk = .7
                hc = a0 * 14.
            elif m == 4: hc = a0 * 16.
            elif m == 5 or m == 6:                                 # head: object xyz, mouth, lip, brow
                hc = (a0 * 1.0 + a1 * 3.4 + a2 * 2.2) * 2.2
                if a3 > 0: mm = 13
                elif m == 6 and a2 < -.70 + .05 * math.sin(a1 * 9.): belly = True
                if m == 5 and a3 <= 0 and a4 > 0: tint = .80
                if a5 > 0 and a3 <= 0: tint = .74
                if m == 5 and a4 > 5.: tint = .22
                if mm == m and not belly and a1 < 2.2 and a5 < .04 and a4 < .0 and tint > .5:     # scales gather on the skull and cheeks
                    an = math.atan2(a0, a2 + .4); nn_ = 9.; a = an * nn_ + 40.; b = (-a1) / .155 + 400.
                    rb = int(math.floor(b)); r0 = rb - 1; off = .5 * (r0 & 1); c0 = int(math.floor(a - off)); x0 = (a - off - c0 - .5) * 2; y0 = b - r0; ye0 = 1.02 + .56 * math.sqrt(max(0., 1 - x0 * x0))
                    if y0 < ye0: row, col, yy, ye, u_ = r0, c0, y0, ye0, 9.
                    else: row = rb; off = .5 * (rb & 1); col = int(math.floor(a - off)); xx = (a - off - col - .5) * 2; yy = b - rb; ye = 1.02 + .56 * math.sqrt(max(0., 1 - xx * xx)); u_ = y0 - ye0
                    hh = _hs(row, col, 1); fa = _ss(2.25, .2, a1) * _ss(6., 14., .155 * un); tint *= 1 + (hh - .5) * .16 * fa; keep = 1. if _n3(a0 * 1.6, a1 * 1.6, a2 * 1.6) + (.25 if v < .3 else -.25) > -.42 else 0.
                    ln = max(ln, _ss(lwp * .85 + .8, lwp * .85 - .8, abs(y0 - ye0) * .155 * un) * fa * (keep if hh < .8 else 0.)); v += (yy / ye - .5) * .16 * fa + (hh - .5) * .12 * fa
                    if u_ < .16: dk = .6 * fa * keep
            elif m == 11: hc = a0 * 10.; b = a1 * 1.0; v += .25 * math.sin(a1 * 6.)
            # ---- tones
            if sh > 0: v = v * (1 - sh) + min(v, -.12) * sh
            T = th1; tb = _ss(T - .02, T + .02, v); tl = _ss(.70, .74, v); td = _ss(-.46, -.42, v)
            if dk > 0 and tb > 0: tb *= 1 - dk
            if hb > 0 and (m == 1 or m == 10 or m == 5 or m == 6 or m == 3) and T < v < T + hb and hc != 0.:   # tapered hatching off the terminator
                g = abs(hc - math.floor(hc) - .5) * 2; k_ = 1 - (v - T) / hb
                if g < .30 * k_ * k_ and _hs(int(math.floor(hc)), int(a1 * 2.), 11) > .35 and (m != 1 and m != 10 or lod < .5 or True): ln = max(ln, .78 * _ss(8., 22., un / 5.5))
            if hb > 0 and (m == 1 or m == 5 or m == 6 or m == 10) and -.44 < v < -.20 and hc != 0.:
                g = abs(hc * 1.13 + .3 - math.floor(hc * 1.13 + .3) - .5) * 2; k_ = 1 - (v + .44) / .24
                if g < .26 * k_ and _hs(int(math.floor(hc * 1.13 + .3)), int(a1 * 2.), 12) > .45: ln = max(ln, .6 * _ss(8., 22., un / 5.5))
            pm = 0 if belly else mm
            tn = tint
            r = g_ = b_ = 0.
            for c in range(3):
                cv = PAL[pm, 3, c] * (1 - td) + PAL[pm, 2, c] * td; cv = cv * (1 - tb) + PAL[pm, 1, c] * tb; cv = cv * (1 - tl) + PAL[pm, 0, c] * tl
                if lift > 0: cv = cv * (1 - lift * tb) + PAL[pm, 0, c] * lift * tb
                cv *= tn
                if gl > 0: cv = cv * (1 - gl) + (.86 + .06 * c) * gl
                if c == 0: r = cv
                elif c == 1: g_ = cv
                else: b_ = cv
            e_ = 0.
            if mm == 13 or m == 7:                                 # mouth, tongue: furnace light
                dp = _ss(1.5, -.55, a1) * (.80 if m == 7 else 1.); n = .16 * _n3(a0 * 4., a1 * 3. + tm * 3., a2 * 4.)
                q = furn * (dp * 1.15 + .10 + n)
                if q > .22 and furn > 0:
                    idx = 4 if q < .42 else 3 if q < .62 else 2 if q < .84 else 1 if q < 1.02 else 0
                    r, g_, b_ = FIRE[idx, 0], FIRE[idx, 1], FIRE[idx, 2]; e_ = min(1., q)
            elif furn > 0 and m >= 3:
                dx, dy, dz = mouth[0] - px, mouth[1] - py, mouth[2] - pz; d = math.sqrt(dx * dx + dy * dy + dz * dz) + 1e-6
                q = furn * max(0., (nx * dx + ny * dy + nz * dz) / d) / (1 + (d / (1.25 * R_)) ** 2 * .55)
                wq = _ss(.20, .25, q + .03 * _n3(px * 9 / R_, py * 9 / R_, pz * 9 / R_)) * .85; w2_ = _ss(.46, .50, q) * .7
                r = r * (1 - wq * .55) + (1.0 * wq * .55) * 1.0; g_ = g_ * (1 - wq * .6) + .46 * wq * .6; b_ = b_ * (1 - wq * .75) + .12 * wq * .75
                r += .25 * w2_; g_ += .22 * w2_; b_ += .10 * w2_
            # ---- fire skin
            fire = m == 12
            fv = 0.
            if burn > 0 and (m == 1 or m == 10) and not fire:
                fr_ = 1.06 - burn * 1.3; n = .06 * _n3(a0 * 1.2 - tm * 4., a1 * 2., tm) + .03 * _n3(a0 * 4 - tm * 8, a1 * 5., 3.)
                if a2 > fr_ + n: fire = True; fv = -.20 * _ss(.05, .0, a2 - fr_ - n)
            if burn > 1.0 and m >= 2 and m != 12:
                fv = -.05; fire = _n3(px / R_ * 1.5, py / R_ * 1.5, pz / R_ * 1.5 - tm * 2.) * .30 + (burn - 1.0) * 6. > .5
            if fire:
                if m == 12: val = (.92 if burn <= 1. else .70) - .95 * a1 - .32 * abs(a0) + .24 * _n3(a0 * 1.2, a1 * 4. - tm * 5., a2 * 60.)
                elif m == 1 or m == 10:
                    val = .13 + .32 * fc + .44 * _n3(a0 * .40 - tm * 2.6, a1 * 2.7 + .7 * math.sin(a0 * .55 - tm * 2.), tm * .8) + .24 * _n3(a0 * .85 - tm * 5., a1 * 6.5 + .6 * math.sin(a0 * 1.3), 9. + tm)
                    if belly: val += .12
                else: val = .07 + .28 * fc + .42 * v0 + .18 * _n3(px / R_ * 1.6, py / R_ * 1.6, pz / R_ * 1.6 - tm * 3.) - (.36 if tint < .9 else 0.) - .22 * dk - (.16 if m == 3 else 0.) + (.40 if m == 4 else 0.)
                val += fv; q0 = val
                idx = 4 if val < .22 else 3 if val < .44 else 2 if val < .66 else 1 if val < .88 else 0
                if m == 8: idx = 5 if a5 > -.04 or a3 < 0 else 0
                if mm == 13: idx = 5 if val < .9 else 4
                r, g_, b_ = FIRE[idx, 0], FIRE[idx, 1], FIRE[idx, 2]; e_ = .85 if idx < 3 else .45
                ln = ln * .80 if m == 5 or m == 6 or m == 3 else 0.
                wl = .020 + 1.2 / max(un, 30.)
                if abs(val - .44) < wl * .8 and _n3(px / R_ * 3, py / R_ * 3, pz / R_ * 3) > -.1: ln = K[11]
                tb = 1.
            if m == 8 and not fire:                                # eye: a3 = signed pupil distance, a4 iris
                if a3 < 0: r, g_, b_ = .05, .03, .06
                elif a4 < 0: r, g_, b_ = PAL[8, 1, 0], PAL[8, 1, 1], PAL[8, 1, 2]
                else: r, g_, b_ = PAL[8, 0, 0], PAL[8, 0, 1], PAL[8, 0, 2]
                e_ = .30 + .5 * furn
                if furn > .8 and a3 >= 0:
                    q = 1 if a4 >= 0 else 0
                    r, g_, b_ = FIRE[q, 0] * 1.1, FIRE[q, 1] * 1.05, FIRE[q, 2]; e_ = 1.
                if a5 > -.26 and furn <= .8: r, g_, b_ = r * .62, g_ * .50, b_ * .48; e_ *= .4
                if a5 > 0: r, g_, b_ = PAL[5, 3, 0], PAL[5, 3, 1], PAL[5, 3, 2]; e_ = 0.
                if abs(a5 - .03) < .10: r, g_, b_ = .06, .05, .10; e_ = 0.
            out[y, x, 0] = r; out[y, x, 1] = g_; out[y, x, 2] = b_; vis[y, x] = tb; pat[y, x] = ln; em[y, x] = e_


@njit(parallel=True, fastmath=True, cache=True)
def _edges(pid, iz, nrm, foc, R_, e1, e2, flag):
    """outer = against nothing or a far jump; inner = part borders, small jumps, creases.  flag[part] 1 = soft part (hair): only jumps"""
    h, w = pid.shape
    for yp in prange(h - 1):
        y = np.int64(yp)
        for x in range(w - 1):
            p = pid[y, x]
            for d in range(2):
                yy = y + d; xx = x + 1 - d; q = pid[yy, xx]
                if p < 0 and q < 0: continue
                if p < 0 or q < 0: e1[y, x] = 1; continue
                za = 1. / iz[y, x]; zb = 1. / iz[yy, xx]; dz = abs(za - zb); px = za / foc
                if p != q:
                    if dz > 1.1 * R_: e1[y, x] = 1
                    elif dz > 3. * px or flag[p] == 0 or flag[q] == 0: e2[y, x] = 1
                elif dz > 14. * px and dz > .05 * R_:
                    if dz > 1.1 * R_: e1[y, x] = 1
                    else: e2[y, x] = 1
                elif flag[p] == 0:
                    if nrm[y, x, 0] * nrm[yy, xx, 0] + nrm[y, x, 1] * nrm[yy, xx, 1] + nrm[y, x, 2] * nrm[yy, xx, 2] < .40: e2[y, x] = 1


# ---------------------------------------------------------------- curves, tubes, ribbons
def cr3(P, n):
    """centripetal-ish Catmull-Rom through 3-D points, resampled to n points of equal arc length -> (n,3), length"""
    P = np.asarray(P, float); m = len(P); ext = np.vstack([2 * P[0] - P[1], P, 2 * P[-1] - P[-2]]); out = []
    t = np.linspace(0, 1, 24, endpoint=False)[:, None]
    for i in range(m - 1):
        a, b, c, d = ext[i], ext[i + 1], ext[i + 2], ext[i + 3]
        out.append(.5 * (2 * b + (c - a) * t + (2 * a - 5 * b + 4 * c - d) * t ** 2 + (3 * b - a - 3 * c + d) * t ** 3))
    Q = np.vstack(out + [P[-1:]]); s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(Q, axis=0), axis=1))]; u = np.linspace(0, s[-1], n)
    return np.stack([np.interp(u, s, Q[:, k]) for k in range(3)], -1), s[-1]


def frames(P, up=(0, 0, 1), blend=.10):
    """tangent, dorsal, side along a polyline: parallel transport that relaxes toward `up`"""
    T = np.gradient(P, axis=0); T /= np.linalg.norm(T, axis=1)[:, None] + 1e-9; up = np.asarray(up, float); up = np.broadcast_to(up, P.shape); U = np.zeros_like(P)
    u = up[0] - T[0] * (T[0] @ up[0]); u /= np.linalg.norm(u) + 1e-9
    for i in range(len(P)):
        u = u - T[i] * (T[i] @ u); q = up[i] - T[i] * (T[i] @ up[i]); l = np.linalg.norm(q); u = u + q * blend * min(1, l * 1.5); u /= np.linalg.norm(u) + 1e-9; U[i] = u
    return T, U, np.cross(T, U)


class Mesh:
    def __init__(self): self.V, self.N, self.A, self.F, self.M, self.I, self.n, self.k, self.soft = [], [], [], [], [], [], 0, 0, []

    def add(self, V, N, A, F, mat, part=None, soft=0):
        if A.shape[1] < 6: A = np.hstack([A, np.zeros((len(A), 6 - A.shape[1]))])
        pid = self.k if part is None else part
        if part is None: self.k += 1; self.soft.append(soft)
        self.V.append(V); self.N.append(N); self.A.append(A); self.F.append(F + self.n); self.M.append(np.full(len(F), mat, np.int32)); self.I.append(np.full(len(F), pid, np.int32)); self.n += len(V); return pid

    def done(self):
        c = np.ascontiguousarray
        return c(np.concatenate(self.V)), c(np.concatenate(self.N)).astype(f32), c(np.concatenate(self.A)).astype(f32), c(np.concatenate(self.F)).astype(np.int32), np.concatenate(self.M), np.concatenate(self.I), np.array(self.soft, np.int32)


def _grid_faces(n, m, wrap=True):
    i = np.arange(n - 1)[:, None]; j = np.arange(m if wrap else m - 1)[None]; a = i * m + j; b = i * m + (j + 1) % m; c = a + m; d = b + m
    return np.stack([np.stack([a, c, b], -1).reshape(-1, 3), np.stack([b, c, d], -1).reshape(-1, 3)], 1).reshape(-1, 3)


def tube(mesh, P, r, mat, up=(0, 0, 1), m=14, flat=1.0, u0=0., u1=1., s0=0., unit=1.0, part=None, cap=True, sec=None, fr=None):
    """lofted tube.  r (n,) radii; flat = up-radius / side-radius; attributes s (in `unit`s), th (0 = dorsal), u"""
    P = np.asarray(P, float); n = len(P); T, U, S = frames(P, up) if fr is None else fr; th = np.linspace(-math.pi, math.pi, m + 1) if True else 0
    r = np.broadcast_to(np.asarray(r, float), (n,)); c, s = np.cos(th), np.sin(th); fl = np.broadcast_to(np.asarray(flat, float), (n,))
    k = np.ones(m + 1) if sec is None else sec(th)
    V = P[:, None] + (U[:, None] * (c * k)[None, :, None] * fl[:, None, None] + S[:, None] * (s * k)[None, :, None]) * r[:, None, None]
    dr = np.gradient(r) / (np.linalg.norm(np.gradient(P, axis=0), axis=1) + 1e-9)
    N = U[:, None] * c[None, :, None] + S[:, None] * s[None, :, None] * fl[:, None, None] - T[:, None] * dr[:, None, None]
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))]
    A = np.zeros((n, m + 1, 6)); A[..., 0] = (d[:, None] + s0) / unit; A[..., 1] = th; A[..., 2] = np.linspace(u0, u1, n)[:, None]
    F = _grid_faces(n, m + 1, False); V = V.reshape(-1, 3); N = N.reshape(-1, 3); A = A.reshape(-1, 6)
    if cap:
        for e, sg in ((0, -1), (n - 1, 1)):
            o = len(V); V = np.vstack([V, P[e:e + 1] + T[e] * sg * r[e] * .5]); N = np.vstack([N, T[e:e + 1] * sg]); A = np.vstack([A, A[e * (m + 1):e * (m + 1) + 1]])
            j = np.arange(m); F = np.vstack([F, np.stack([np.full(m, o), e * (m + 1) + j, e * (m + 1) + j + 1], -1)])
    return mesh.add(V, N, A, F, mat, part)


def ribbons(mesh, roots, dirs, L, wd, eye, mat=M_MANE, wind=(0, 0, 0), f=0., seed=0, n=12, sway=.12, curl=.5, up=(0, 0, 1), unit=1., hz=.9, grav=0., prof=None, each=True, lagk=1.):
    """hair / flame clumps: billboard blades with an S-curve and a travelling wave.  roots, dirs (m,3); L, wd (m,)"""
    m = len(roots); roots = np.asarray(roots, float); dirs = np.asarray(dirs, float); dirs = dirs / (np.linalg.norm(dirs, axis=1)[:, None] + 1e-9); L = np.asarray(L, float); wd = np.asarray(wd, float)
    t = np.linspace(0, 1, n)[None, :, None]; wind = np.asarray(wind, float); r = np.array([[h1(seed, i, q) for q in range(5)] for i in range(m)]); upv = np.broadcast_to(np.asarray(up, float), dirs.shape)
    side = np.cross(dirs, upv); side /= np.linalg.norm(side, axis=1)[:, None] + 1e-9; nrm = np.cross(side, dirs)
    ph = r[:, 0] * 6.283; tt = f / 24. * 6.283 * hz
    w1 = np.sin(6.283 * .75 * t[..., 0] - tt + ph[:, None]) * t[..., 0] ** 1.2; w2 = np.sin(6.283 * .95 * t[..., 0] - tt * 1.15 + ph[:, None] * 2) * t[..., 0] ** 1.3
    LL = L[:, None, None]
    C = roots[:, None] + dirs[:, None] * LL * t + wind[None, None] * LL * t ** 1.7 * lagk + nrm[:, None] * LL * ((r[:, 1] - .5) * curl)[:, None, None] * t ** 2 * 1.2 + side[:, None] * LL * ((r[:, 2] - .5) * curl)[:, None, None] * t ** 2 \
        + (nrm[:, None] * (w1 * sway)[..., None] + side[:, None] * (w2 * sway)[..., None]) * LL - np.array([0, 0, grav]) * LL * t ** 2
    T = np.gradient(C, axis=1); T /= np.linalg.norm(T, axis=2)[..., None] + 1e-9; V = np.asarray(eye, float) - C; V /= np.linalg.norm(V, axis=2)[..., None] + 1e-9
    X = np.cross(T, V); X /= np.linalg.norm(X, axis=2)[..., None] + 1e-9
    v = t[..., 0]; p = (np.minimum(1, .45 + v / .22 * .55) * (1 - v) ** .62 * (1 + .35 * np.sin(v * 3.1))) if prof is None else prof(v)
    hw = wd[:, None] * p; aX = np.array([-1., 0., 1.]); 
    Vs = C[:, :, None] + X[:, :, None] * (hw[..., None] * aX)[..., None]; Nn = X[:, :, None] * (aX * .8)[None, None, :, None] + V[:, :, None] * .6
    A = np.zeros((m, n, 3, 6)); A[..., 0] = aX; A[..., 1] = v[0][None, :, None] if v.ndim == 2 else v; A[..., 2] = r[:, 3][:, None, None]; A[..., 3] = (wd * 2 / unit)[:, None, None]
    F = _grid_faces(n, 3, False)
    if not each:
        mesh.add(Vs.reshape(-1, 3), Nn.reshape(-1, 3), A.reshape(-1, 6), (F[None] + (np.arange(m) * n * 3)[:, None, None]).reshape(-1, 3), mat, None, 1); return
    for i in range(m): mesh.add(Vs[i].reshape(-1, 3), Nn[i].reshape(-1, 3), A[i].reshape(-1, 6), F, mat, None, 1)


# ---------------------------------------------------------------- the head (skull units: x right, y snout, z up)
def _ell(p, c, r):
    q = (p - np.array(c)) / np.array(r); k0 = np.linalg.norm(q, axis=-1); k1 = np.linalg.norm(q / np.array(r), axis=-1); return k0 * (k0 - 1) / (k1 + 1e-9)


def _cap(p, a, b, ra, rb, sc=(1, 1, 1)):
    sc = np.array(sc, float); a, b = np.array(a) * sc, np.array(b) * sc; q = p * sc - a; d = b - a; h = np.clip((q @ d) / (d @ d), 0, 1); return np.linalg.norm(q - h[..., None] * d, axis=-1) - (ra + (rb - ra) * h)


def _smin(a, b, k): h = np.maximum(k - np.abs(a - b), 0); return np.minimum(a, b) - h * h / (4 * k)
def _smax(a, b, k): return -_smin(-a, -b, k)


LIPS = [((.92, -.30, -.10), (.72, .70, -.30), .17, .18), ((.72, .70, -.30), (.58, 1.36, .00), .18, .15), ((.58, 1.36, .00), (.50, 2.06, -.24), .15, .20), ((.50, 2.06, -.22), (0., 2.46, -.08), .19, .17)]


def sdf_skull(p):
    q = p.copy(); q[..., 0] = np.abs(q[..., 0]); x, y, z = q[..., 0], q[..., 1], q[..., 2]
    d = _ell(q, (0, -.60, .28), (1.0, 1.25, .84)); d = _smin(d, _cap(q, (0, .2, .30), (0, 2.05, .14), .70, .41, (1, 1, 1.06)), .30)
    d = _smin(d, _ell(q, (0, 2.20, .34), (.40, .34, .34)), .16); d = _smin(d, _ell(q, (.35, 2.22, .34), (.23, .34, .22)), .08)
    d = _smin(d, _cap(q, (.16, 1.16, .56), (.90, .44, .86), .19, .27), .10); d = _smin(d, _cap(q, (.90, .44, .86), (1.42, -.62, 1.30), .25, .04), .08)
    d = _smin(d, _cap(q, (.62, 1.0, .16), (1.12, .10, .14), .13, .21), .10); d = _smin(d, _cap(q, (1.12, .10, .14), (1.22, -.75, .30), .20, .05), .10)
    d = _smin(d, _ell(q, (.92, -.30, .02), (.42, .80, .50)), .2)
    for a, b, ra, rb in LIPS: d = _smin(d, _cap(q, a, b, ra, rb), .10)
    d = _smin(d, _cap(q, (0, -1.0, .05), (0, -2.3, -.10), .90, .98), .3)
    wr = smooth(.72, .95, y) * (1 - smooth(1.75, 1.98, y)) * smooth(.18, .42, z); d = d - .034 * np.sin(y * 17 - x * 5) * wr
    chk = smooth(.5, .8, x) * smooth(-.9, -.6, y) * (1 - smooth(.3, .6, y)) * (1 - smooth(.25, .45, z)); d = d - .022 * np.sin(y * 13 + z * 10) * chk
    fr = smooth(.55, .75, z) * smooth(-.2, .1, y) * (1 - smooth(.9, 1.05, y)) * (1 - smooth(.5, .75, x)); d = d - .03 * np.sin((y * 11 - x * 9)) * fr
    d = _smax(d, -_ell(q, (.31, 2.47, .42), (.085, .17, .055)), .03); d = _smax(d, -_ell(q, (.86, .50, .50), (.24, .36, .25)), .07)
    d = _smax(d, -_ell(q, (0, 1.05, -.46), (.50, 1.62, .40)), .05)
    return d


def sdf_jaw(p):
    q = p.copy(); q[..., 0] = np.abs(q[..., 0])
    d = _cap(q, (.84, -.60, -.44), (.58, 1.15, -.66), .27, .22); d = _smin(d, _cap(q, (.58, 1.15, -.66), (.40, 2.00, -.56), .22, .22), .1)
    d = _smin(d, _ell(q, (0, 2.06, -.60), (.46, .34, .26)), .14); d = _smin(d, _ell(q, (0, .85, -.82), (.60, 1.5, .17)), .12); d = _smin(d, _ell(q, (.84, -.62, -.26), (.30, .44, .44)), .16)
    d = _smin(d, _cap(q, (0, 2.12, -.72), (0, 2.02, -1.02), .16, .04), .1)
    return _smax(d, -_ell(q, (0, .92, -.46), (.46, 1.40, .30)), .05)


def surf_nets(fn, lo, hi, h):
    n = np.ceil((np.array(hi) - lo) / h).astype(int) + 1; ax = [lo[i] + np.arange(n[i]) * h for i in range(3)]
    G = np.stack(np.meshgrid(*ax, indexing='ij'), -1); D = np.concatenate([fn(G[i:i + 8]) for i in range(0, n[0], 8)])
    c = (n - 1); off = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)]); corner = [D[o[0]:o[0] + c[0], o[1]:o[1] + c[1], o[2]:o[2] + c[2]] for o in off]
    neg = np.stack([q < 0 for q in corner], -1); act = neg.any(-1) & ~neg.all(-1); idx = np.full(c, -1, np.int64); I = np.argwhere(act); idx[tuple(I.T)] = np.arange(len(I))
    cv_ = np.stack([q[act] for q in corner], -1); ac = np.zeros((len(I), 3)); cnt = np.zeros(len(I))
    for a in range(8):
        for b in range(a + 1, 8):
            if np.abs(off[a] - off[b]).sum() != 1: continue
            m = (cv_[:, a] < 0) != (cv_[:, b] < 0); t = cv_[:, a] / (cv_[:, a] - cv_[:, b] + 1e-12); ac += np.where(m[:, None], off[a] + (off[b] - off[a]) * t[:, None], 0); cnt += m
    V = (I + ac / cnt[:, None]) * h + np.array(lo); Fs = []
    for ax_ in range(3):
        sl0 = [slice(0, c[k]) for k in range(3)]; sl0 = [slice(1, c[k]) if k != ax_ else slice(0, c[k]) for k in range(3)]
        a = D[tuple(slice(1, c[k]) if k != ax_ else slice(0, c[k]) for k in range(3))]; b = D[tuple(slice(1, c[k]) if k != ax_ else slice(1, c[k] + 1) for k in range(3))]
        m = (a < 0) != (b < 0); J = np.argwhere(m)
        for k in range(3):
            if k != ax_: J[:, k] += 1
        u, v = [k for k in range(3) if k != ax_]; e = np.eye(3, dtype=int); g = lambda du, dv: idx[tuple((J - e[u] * du - e[v] * dv).T)]
        q = np.stack([g(0, 0), g(1, 0), g(1, 1), g(0, 1)], -1); ok = (q >= 0).all(1); q = q[ok]; Fs += [q[:, [0, 1, 2]], q[:, [0, 2, 3]]]
    N = np.stack([(fn(V + e_ * .012) - fn(V - e_ * .012)) for e_ in np.eye(3)], -1); N /= np.linalg.norm(N, axis=1)[:, None] + 1e-9
    return V, N, np.concatenate(Fs)


def _lipd(q):
    q = q.copy(); q[..., 0] = np.abs(q[..., 0]); return np.max([rb * .5 + ra * .5 + .035 - _cap(q, a, b, 0, 0) for a, b, ra, rb in LIPS], axis=0)


CURVES = [  # accent pen strokes on the skull (right side, mirrored): points, width
    ([(.02, 1.02, .9), (.28, .96, .8), (.50, .82, .6)], 2.6), ([(.02, 1.26, .8), (.30, 1.20, .7), (.52, 1.04, .45)], 2.4), ([(.02, 1.50, .75), (.30, 1.43, .65), (.50, 1.28, .4)], 2.2), ([(.04, 1.72, .7), (.28, 1.66, .6), (.46, 1.50, .4)], 2.0),
    ([(.06, .78, 1.0), (.13, 1.0, .9), (.10, 1.2, .9)], 3.0), ([(.24, .86, 1.0), (.30, 1.12, .9)], 2.4), ([(.30, .98, 1.0), (.58, .70, 1.1), (.88, .36, 1.2), (1.12, -.2, 1.3)], 2.6),
    ([(.52, 1.06, .10), (.80, .74, -.02), (1.08, .22, -.02), (1.20, -.2, .0)], 2.6), ([(1.12, -.10, -.16), (1.14, -.45, -.22), (1.02, -.85, -.30)], 2.2),
    ([(.14, 2.42, .66), (.36, 2.30, .66), (.56, 2.08, .52), (.56, 1.86, .28)], 2.8), ([(.50, 1.86, .30), (.56, 1.64, .20), (.56, 1.40, .10)], 2.2),
    ([(.98, -.24, .06), (.86, .30, -.10), (.76, .74, -.12), (.68, 1.08, .06), (.64, 1.38, .16), (.62, 1.74, .02), (.58, 2.04, -.04), (.40, 2.32, .04)], 2.6),
    ([(.10, .40, 1.2), (.34, .22, 1.2), (.60, -.10, 1.2)], 2.2), ([(.06, .16, 1.2), (.30, -.04, 1.2), (.52, -.36, 1.2)], 2.0), ([(1.0, -.45, .55), (1.06, -.8, .4), (.98, -1.15, .25)], 2.2),
    ([(.46, 2.40, .0), (.30, 2.58, .02), (.0, 2.66, .04)], 2.2), ([(.0, 2.60, .42), (.02, 2.62, .2), (.0, 2.64, .02)], 2.2)]
CURVES_J = [([(1.0, -.35, -.5), (.86, .3, -.66), (.74, .9, -.72), (.6, 1.5, -.74)], 2.2), ([(.5, 2.1, -.78), (.3, 2.3, -.82), (0, 2.38, -.84)], 2.2), ([(.9, -.6, -.62), (.82, -.1, -.78), (.66, .4, -.9)], 2.0)]
_HD = {}


def head_mesh():
    if 'ok' in _HD: return _HD
    c = os.path.join(ROOT, 'cache', 'kaido_head_7.npz')
    if os.path.exists(c): z = np.load(c); _HD.update({k: z[k] for k in z.files})
    else:
        V, N, F = surf_nets(sdf_skull, (-1.6, -2.4, -.95), (1.6, 2.95, 1.6), .040); Vj, Nj, Fj = surf_nets(sdf_jaw, (-1.5, -1.5, -1.3), (1.5, 2.7, .4), .040); d = {}
        A = np.zeros((len(V), 6)); A[:, :3] = V; q = V.copy(); q[:, 0] = abs(q[:, 0]); A[:, 3] = .045 - _ell(q, (0, 1.05, -.46), (.50, 1.62, .40)); A[:, 3] = np.where(N[:, 2] > .3, -.1, A[:, 3]); A[:, 4] = _lipd(V)
        A[:, 5] = np.maximum(.22 - _cap(q, (.16, 1.16, .56), (.90, .44, .86), 0, 0), .20 - _cap(q, (.90, .44, .86), (1.42, -.62, 1.30), 0, 0)); ns = _ell(q, (.31, 2.47, .42), (.085, .17, .055)); A[:, 4] = np.where(ns < .05, 9., A[:, 4])
        B = np.zeros((len(Vj), 6)); B[:, :3] = Vj; q = Vj.copy(); q[:, 0] = abs(q[:, 0]); B[:, 3] = .045 - _ell(q, (0, .92, -.46), (.46, 1.40, .30)); B[:, 4] = -1; B[:, 5] = -1
        d.update(V=V, N=N, F=F, A=A, Vj=Vj, Nj=Nj, Fj=Fj, B=B); np.savez(c + '.tmp.npz', **d); os.replace(c + '.tmp.npz', c); _HD.update(d)
    def snap(P, fn):
        P = cr3(np.array(P, float), 18)[0] if len(P) > 2 else np.linspace(P[0], P[1], 10)
        for _ in range(6):
            d = fn(P); g = np.stack([(fn(P + e * .01) - fn(P - e * .01)) for e in np.eye(3)], -1) / .02; P = P - g * (d / (np.sum(g * g, -1) + 1e-6))[:, None]
        return P, g / (np.linalg.norm(g, axis=-1)[:, None] + 1e-9)
    if 'cur' not in _HD: _HD['cur'] = [(*snap(P, sdf_skull), w) for P, w in CURVES]; _HD['curj'] = [(*snap(P, sdf_jaw), w) for P, w in CURVES_J]
    _HD['ok'] = 1; return _HD


JP = np.array([0, -.70, -.26])


def build_head(mesh, C, Rm, sc, eye, jaw=.4, f=0., wind=(0, 0, 0), look=(0., 0.), whisk=1., mane=True, seed=0, frown=1., fire=False, wlag=None, hair=1., breath=0., eyes=1., whz=1.):
    """put the head in the world.  C = origin, Rm 3x3 (columns = skull x,y,z axes in world), sc = metres per skull unit -> dict of world anchors"""
    H_ = head_mesh(); X = lambda p: np.asarray(p, float) @ Rm.T * sc + C; Nn = lambda n: np.asarray(n, float) @ Rm.T
    cj, sj = math.cos(-jaw), math.sin(-jaw); RJ = np.array([[1, 0, 0], [0, cj, -sj], [0, sj, cj]]); J = lambda p: (np.asarray(p, float) - JP) @ RJ.T + JP
    nw = jaw * .05; up = Rm[:, 2]
    p0 = mesh.add(X(H_['V']), Nn(H_['N']), H_['A'], H_['F'], M_SKIN); p1 = mesh.add(X(J(H_['Vj'])), Nn(H_['Nj'] @ RJ.T), H_['B'], H_['Fj'], M_JAW)
    # eyes
    th = np.linspace(0, math.pi, 13); ph = np.linspace(0, 2 * math.pi, 25); T_, P_ = np.meshgrid(th, ph, indexing='ij'); S = np.stack([np.sin(T_) * np.cos(P_), np.sin(T_) * np.sin(P_), np.cos(T_)], -1).reshape(-1, 3); Fs = _grid_faces(13, 25, False)
    for sg in (-1, 1):
        ec = np.array([sg * .75, .52, .46]); g = np.array([sg * (.80 + look[0] * sg), .62, -.02 + look[1]]); g /= np.linalg.norm(g); e1 = np.cross(g, [0, 0, 1]); e1 /= np.linalg.norm(e1); e2 = np.cross(e1, g)
        A = np.zeros((len(S), 6)); a, b, c = S @ e1, S @ e2, S @ g; A[:, 3] = np.where(c > 0, np.sqrt((a / .115) ** 2 + (b / .34) ** 2) - 1, 1.); A[:, 4] = np.where(c > 0, np.sqrt(a ** 2 + b ** 2) - .46, 1.); o_ = a * sg; A[:, 5] = np.maximum(b - (.10 + .62 * (eyes - 1) + .46 * o_ - .25 * o_ ** 2), (-.40 - .12 * o_) - b)
        mesh.add(X(ec + g * .10 * (eyes - 1) / .35 + S * .262 * eyes), Nn(S), A, Fs, M_EYE)
    # teeth: (base, dir, length, radius)
    def tooth(b, d, L, r, jw):
        t = np.linspace(0, 1, 7)[:, None]; b, d = np.array(b, float), np.array(d, float); P = b + d * L * t + np.array([0, -.30 * L, 0]) * t ** 2 * .7
        P = J(P) if jw else P; tube(mesh, X(P), r * sc * (1 - t[:, 0]) ** .75 + .004 * sc, M_CLAW, up=Rm[:, 1], m=9, cap=False)
    for sg in (-1, 1):
        for (bx, by, bz), L, r in (((.46, 2.06, -.30), .74, .14), ((.24, 2.36, -.30), .24, .075), ((.50, 1.76, -.16), .30, .085), ((.53, 1.48, -.06), .46, .11), ((.56, 1.20, -.10), .30, .085), ((.60, .92, -.22), .38, .095), ((.64, .62, -.34), .26, .08), ((.67, .34, -.32), .22, .07), ((.70, .08, -.28), .18, .06)):
            tooth((sg * bx, by, bz), (sg * .06, .08, -1), L, r, 0)
        for (bx, by, bz), L, r in (((.34, 1.98, -.50), .66, .125), ((.14, 2.14, -.48), .20, .065), ((.42, 1.64, -.52), .22, .07), ((.46, 1.34, -.54), .32, .085), ((.50, 1.04, -.56), .20, .07), ((.54, .72, -.56), .26, .075), ((.58, .40, -.54), .18, .06)):
            tooth((sg * bx, by, bz), (sg * .05, .08, 1), L, r, 1)
    # tongue
    t = np.linspace(0, 1, 16); wob = .06 * np.sin(f / 24 * 6.28 * 1.2 - t * 4); P = np.c_[np.zeros(16) + .10 * np.sin(t * 3 + f * .09) * t, -.35 + 2.15 * t, -.52 + .20 * np.sin(t * math.pi) + .12 * t ** 3 * jaw + wob * t]
    V, l = J(P), 0; n0 = mesh.n; pid = tube(mesh, X(J(P)), sc * (.30 * (1 - t ** 3) ** .6 + .02), M_TONGUE, up=Nn(np.array([[0, 0, 1.]]))[0], m=12, flat=.36); A = mesh.A[-1]; loc = (mesh.V[-1] - C) @ Rm / sc; A[:, 0] = loc[:, 0]; A[:, 1] = loc[:, 1]; A[:, 2] = loc[:, 2]
    # horns, ears, fins
    for sg in (-1, 1):
        S_ = np.array([sg, 1, 1.]); hp, _ = cr3(np.array([(.42, -.40, .80), (.72, -.82, 1.50), (1.12, -1.36, 2.05), (1.46, -2.05, 2.36), (1.56, -2.75, 2.86), (1.30, -3.25, 3.66)]) * S_, 36); u = np.linspace(0, 1, 36)
        tube(mesh, X(hp), sc * (.30 * (1 - u) ** .8 + .012) * (1 + .05 * np.sin(u * 60)), M_HORN, up=up, m=16, unit=sc)
        bp, _ = cr3(np.array([(1.18, -1.46, 2.10), (1.58, -1.40, 2.46), (1.96, -1.62, 2.92), (2.05, -2.05, 3.5)]) * S_, 22); u = np.linspace(0, 1, 22); tube(mesh, X(bp), sc * (.17 * (1 - u) ** .8 + .01), M_HORN, up=up, m=12, unit=sc)
        bp, _ = cr3(np.array([(.62, -1.0, 1.40), (.60, -.85, 1.80), (.72, -.9, 2.2)]) * S_, 14); u = np.linspace(0, 1, 14); tube(mesh, X(bp), sc * (.11 * (1 - u) ** .8 + .01), M_HORN, up=up, m=10, unit=sc)
        flap = .12 * math.sin(f / 24 * 6.28 * 1.3 + sg)
        ep, _ = cr3(np.array([(.96, -.72, .52), (1.30, -1.05, .62 + flap * .3), (1.62, -1.55, .55 + flap), (1.76, -2.0, .42 + flap * 1.6)]) * S_, 16); u = np.linspace(0, 1, 16); tube(mesh, X(ep), sc * (.25 * np.sin(np.pi * (u * .86 + .12)) ** .8 + .01), M_FIN, up=Nn(np.array([[sg * .5, .5, .4]]))[0], m=10, flat=.35, unit=sc)
        for k, (bx, by, bz, L) in enumerate(((1.08, -.62, .10, 1.35), (1.06, -.66, -.22, 1.15), (.96, -.70, -.50, .95))):
            fp, _ = cr3(np.array([(bx, by, bz), (bx + .30, by - .45 * L, bz - .05 * k), (bx + .44, by - .95 * L, bz + .06 + flap * .4), (bx + .36, by - 1.35 * L, bz + .22 + flap)]) * S_, 16); u = np.linspace(0, 1, 16)
            tube(mesh, X(fp), sc * (.20 * (1 - u) ** .7 + .01), M_FIN, up=Nn(np.array([[sg * .8, 0, .4]]))[0], m=10, flat=.45, unit=sc)
    th_ = np.array([0, -.28, -.40]) + S * np.array([.66, .50, .60]); A = np.zeros((len(S), 6)); A[:, :2] = np.c_[th_[:, 0], -.50 + 1.15 * (1 - S[:, 1])]; A[:, 2] = th_[:, 2]; A[:, 3] = 1.; mesh.add(X(th_), Nn(S), A, Fs, M_JAW)
    if breath > 0:
        g = RNG(seed, int(f) // 2, 31); n_ = int(70 * breath); a_ = g.uniform(0, 6.283, n_); q_ = np.sqrt(g.random(n_)) * .9; ro = np.c_[np.cos(a_) * q_ * .55, np.full(n_, -.15) + .6 * g.random(n_), -.46 + np.sin(a_) * q_ * .34]; di = np.c_[np.cos(a_) * .30, np.full(n_, 1.), np.sin(a_) * .22 - .42 * jaw * .6]; kk = g.random(n_)
        ribbons(mesh, X(ro), Nn(di), sc * np.where(kk < .16, 1.25, np.where(kk < .5, .85, .5)) * (.75 + .5 * g.random(n_)) * min(1., breath), sc * np.where(kk < .16, .15, np.where(kk < .5, .10, .06)), eye, M_FIRE, wind=(0, 0, .10), f=f, seed=int(f) // 2, n=12, sway=.15, curl=.5, up=up, unit=sc, hz=2.6)
    wv = np.asarray(wind, float)
    # whiskers: long, travelling wave with lag
    if whisk > 0:
        for sg in (-1, 1):
            t = np.linspace(0, 1, 60); L = 6.6 * whisk; om = f / 24 * 6.283 * .62
            q = np.stack([sg * (.50 + 1.10 * np.sin(np.minimum(1, t * 1.6) * 1.571) + .40 * t * np.sin(t * 7.2 - om)), 2.12 + .42 * np.sin(np.minimum(1, t * 4.) * 1.571) - 5.0 * t ** 1.25, .05 + .75 * whz * np.sin(np.minimum(1, t * 3.0) * 1.571) * np.exp(-t * 1.2) - .7 * t + .75 * t * np.sin(t * 8.2 - om * 1.15 + sg)], -1) * np.array([1, whisk, 1]) + np.array([0, 2.1 * (1 - whisk), 0])
            P = X(q) + wv * (t ** 1.6)[:, None] * L * .40 * sc
            tube(mesh, P, sc * (.060 * (1 - t) ** .5 + .010), M_WHISK, up=up, m=9, unit=sc)
    if hair > 0:
        g = RNG(seed, 77); ro, di, Ln, wd = [], [], [], []
        for row in range(3):
            m = (19, 23, 21)[row]
            for i in range(m):
                a = -2.35 + 4.7 * (i + .5 + .3 * g.normal()) / m; y = -.92 - .42 * row; rr = (1.0, 1.04, 1.02)[row]
                ro.append((math.sin(a) * .90 * rr, y + .1 * g.normal(), .22 + math.cos(a) * .82 * rr)); q = np.array([math.sin(a) * (.75 - .12 * row), -1.0, math.cos(a) * (.62 - .1 * row)]); di.append(q + g.normal(0, .12, 3)); k = g.random()
                big = k < .24; Ln.append((2.5 if big else 1.8 if k < .6 else 1.15) * (.9 + .3 * g.random()) * (1 + .25 * row)); wd.append((.30 if big else .20 if k < .6 else .12) * (.9 + .3 * g.random()))
        for i in range(9): ro.append((g.normal(0, .18), 1.95 - .10 * i + g.normal(0, .04), -.98 + .02 * i)); di.append((g.normal(0, .25), -.25 - .04 * i, -1)); Ln.append(1.0 + .7 * g.random()); wd.append(.10 + .09 * g.random())
        ro = np.array(ro); beard = np.arange(len(ro)) >= len(ro) - 9; ro[beard] = J(ro[beard]); dd = np.array(di); dd[beard] = dd[beard] @ RJ.T
        for sg in (-1, 1):
            for k in range(5): ro = np.vstack([ro, [(sg * (.95 - .05 * k), -.45 + .12 * k, -.72 + J(np.array([[0, .1, -.6]]))[0, 2] * .0)]]); dd = np.vstack([dd, [(sg * .5, -.9, -.6)]]); Ln.append(.9 + .5 * g.random()); wd.append(.13)
        ribbons(mesh, X(ro), Nn(dd), np.array(Ln) * sc * hair, np.array(wd) * sc, eye, M_FIRE if fire else M_MANE, wind=wv, f=f, seed=seed + 3, n=13, sway=.10 if not fire else .16, curl=.6, up=up, unit=sc, hz=.9 if not fire else 1.9, grav=0. if fire else .05)
    an = dict(mouth=X(J(np.array([[0, 1.5, -.5]]))[0] * .5 + np.array([0, 1.5, -.3]) * .5), throat=X((0, .1, -.42)), nose=X((0, 2.6, .25)), eyeL=X((-.86, .60, .52)), eyeR=X((.86, .60, .52)), top=X((0, -.3, 1.1)), horn=X((1.30, -3.25, 3.66)), chin=X(J(np.array([(0, 2.1, -.9)]))[0]), fwd=Rm[:, 1], up=up)
    an['curves'] = [(X(P * [sg, 1, 1]), Nn(N * [sg, 1, 1]), w) for P, N, w in H_['cur'] for sg in (-1, 1)] + [(X(J(P * [sg, 1, 1])), Nn((N * [sg, 1, 1]) @ RJ.T), w) for P, N, w in H_['curj'] for sg in (-1, 1)]
    return an


# ---------------------------------------------------------------- legs
def build_leg(mesh, o, T, U, S, sg, rad, eye, pose=(.5, -.2), grip=.35, f=0., unit=1.):
    """o = hip, T body forward (to the head), U dorsal, S side; rad = body radius there.  pose = (forward swing, drop)"""
    k = rad; out = S * sg; a = o + out * .70 * k - U * .35 * k
    el = a + (out * .95 - T * (.55 - pose[0]) - U * (.55 + pose[1])) * k * .95; wr = el + (out * .20 + T * (1.05 + pose[0] * .4) - U * (.62 + pose[1])) * k * .95; u = np.linspace(0, 1, 18)
    P, _ = cr3([a - out * k * .3, a, (a + el) / 2 + U * k * .06, el], 18); tube(mesh, P, k * (.46 - .16 * u + .07 * np.sin(u * 3.1)), M_LEG, up=U, m=20, unit=unit)
    P, _ = cr3([el - (wr - el) * .06, el + (wr - el) * .3 + out * k * .08, wr - (wr - el) * .25, wr], 18); tube(mesh, P, k * (.31 + .09 * np.sin(u * 2.6) - .13 * u), M_LEG, up=U, m=18, unit=unit)
    fw = (wr - el); fw /= np.linalg.norm(fw); dn = -U * .75 + fw * .4; dn /= np.linalg.norm(dn); sd = np.cross(fw, U); sd /= np.linalg.norm(sd) + 1e-9
    pc = wr + fw * .22 * k; u = np.linspace(0, 1, 8); tube(mesh, np.linspace(wr - fw * .12 * k, pc + fw * .22 * k, 8), k * (.20 + .10 * np.sin(u * 2.8)), M_LEG, up=U, m=16, flat=.75, unit=unit)
    for i, a_ in enumerate((-.80, -.28, .28, .80, 2.75)):
        d = fw * math.cos(a_) + sd * math.sin(a_); b = pc + d * .22 * k; L = k * (.62 if i < 4 else .44); g = grip
        P, _ = cr3([b, b + d * L * .42 + U * L * .16, b + d * L * (.86 - .1 * g) - U * L * (.06 + .25 * g), b + d * L * (1.05 - .45 * g) - U * L * (.42 + .45 * g)], 16); u = np.linspace(0, 1, 16)
        tube(mesh, P, k * (.115 * (1 - .28 * u) * (1 + .20 * np.sin(u * 9.4))), M_LEG, up=U, m=12, unit=unit)
        e = P[-1]; dd = P[-1] - P[-3]; dd /= np.linalg.norm(dd); Q, _ = cr3([e - dd * .05 * k, e + dd * .16 * k + d * .03 * k, e + dd * .30 * k - d * .04 * k, e + dd * .44 * k - d * .18 * k], 12); u = np.linspace(0, 1, 12)
        tube(mesh, Q, k * (.095 * (1 - u) ** .7 + .003), M_CLAW, up=U, m=10, cap=False, unit=unit)
    return dict(wrist=wr, claw=pc + fw * .5 * k)


# ---------------------------------------------------------------- flame clouds (flat drawn sprites)
def flame_cloud(size=420, t=0., seed=0, ang=math.pi, pal=('#7a1410', '#e2471a', '#ff9a2a', '#ffe08a', '#fff6d0'), ink='#3a0c10', lw=3.2, lobes=6, tail=1.0):
    """engraved curl of fire: scalloped head with spirals and licking tails streaming along `ang` -> premult RGBA (size,size)"""
    S = size; cel = Cel(S, S); c = np.array([S * .5 - math.cos(ang) * S * .12, S * .5 - math.sin(ang) * S * .12]); r0 = S * .17; d = np.array([math.cos(ang), math.sin(ang)]); n_ = np.array([-d[1], d[0]]); k = int(t * 12)
    L = []
    for i in range(lobes):
        a = ang + math.pi + (i / (lobes - 1.) - .5) * 4.0 + .15 * (h1(seed, i, 1) - .5); rr = r0 * (.62 + .34 * h1(seed, i, 2)) * (1 + .05 * math.sin(t * 5 + i)); q = c + np.array([math.cos(a), math.sin(a)]) * r0 * .86; L.append((q, rr, a))
    tl = []
    for i in range(7):
        o = (i - 3) / 3.; ln = S * tail * (.30 + .22 * h1(seed, i, 5) + .06 * math.sin(t * 9 + i * 2)) * (1 - .35 * abs(o)); b = c + n_ * o * r0 * 1.15 + d * r0 * .1; v = lin01(16); w_ = S * .035 * np.sin(v * 5.5 - t * 11 + i * 1.7) * v
        sp = b[None] + d[None] * (ln * v)[:, None] + n_[None] * (w_ + o * ln * .16 * v)[:, None]; tl.append((sp, r0 * (.46 - .10 * abs(o))))
    def blob(q, rr): a = np.linspace(0, 6.283, 40); return np.c_[q[0] + np.cos(a) * rr, q[1] + np.sin(a) * rr]
    for g, col in ((lw * 1.05, ink), (0, pal[0]), (-S * .012, pal[1]), (-S * .030, pal[2]), (-S * .052, pal[3])):
        sh = -g / S
        for sp, w in tl:
            ww = w * (1 if g >= 0 else max(.12, 1 + g * 9 / S * 2.2)); m = sp if g >= 0 else sp[:max(4, int(16 * (1 + g * 13 / S)))]
            cel.fill(band(m, lambda v: ww * tongue(v, .18, 1.3) + max(0, g), seed, 0, 40, .4), col)
        for q, rr, a in L: cel.fill(blob(q + d * max(0, -g) * .5, max(2, rr + g * 1.3)), col)
        cel.fill(blob(c + d * r0 * .25, max(2, r0 * 1.02 + g * 1.3)), col)
    for i, (q, rr, a) in enumerate(L):                              # spirals
        sg = 1 if i >= lobes / 2 else -1; u = np.linspace(0, 1, 34); th = a + sg * (1.9 - u * 6.6); rad = rr * (.94 - .80 * u)
        cel.line(np.c_[q[0] + np.cos(th) * rad, q[1] + np.sin(th) * rad], lw * .95, ink, taper=(.12, .5), seed=seed + i, press=.3)
    for i in (1, 3, 5):
        sp, w = tl[i]; cel.line(sp[1:11] + n_ * w * .12, lw * .7, ink, taper=(.2, .6), seed=i)
    return cel.rgba()


def sprite(layer, iz, spr, x, y, depth, scale=1.0, rot=0.):
    """paste a premult RGBA sprite centred at (x,y) where it is nearer than the z-buffer (iz = 1/depth at 1x)"""
    S = spr.shape[0]; c, s = math.cos(rot) * scale, math.sin(rot) * scale; M = np.float32([[c, -s, 0], [s, c, 0]]); r = abs(scale) * S * .75
    x0, y0, x1, y1 = int(max(0, x - r)), int(max(0, y - r)), int(min(W, x + r)), int(min(H, y + r))
    if x1 - x0 < 2 or y1 - y0 < 2: return layer
    M[:, 2] = [x - x0 - c * S / 2 + s * S / 2, y - y0 - s * S / 2 - c * S / 2]
    q = cv2.warpAffine(spr, M, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR if scale > .6 else cv2.INTER_AREA); q = q * (iz[y0:y1, x0:x1] < 1. / max(depth, 1e-3))[..., None]
    d = layer[y0:y1, x0:x1]; layer[y0:y1, x0:x1] = d * (1 - q[..., 3:4]) + q; return layer


# ---------------------------------------------------------------- render
FIRE5 = np.stack([rgb(c) for c in ('#fff6c4', '#ffd648', '#ff9224', '#e8400f', '#96140e', '#4a0a10')]).astype(f32)


def render(mesh, cam, f=0., R=1., lit='storm', burn=0., furn=0., mouth=None, curves=(), ss=1.5, scales=34, lw=(5.4, 2.6, 1.7), wet=0., term=.10, hatch=.26, shadow=True, sm_c=None, sm_r=None, grad=None, fog=None, light=True, rim=1., sil=None, rd=None, soften=0.):
    """mesh -> dict(layer, light, iz, mask).  fog = (rgb, start, end) by depth; grad = (p0, p1, stops) multiply gradient; sil = colour: graphic silhouette amount tuple (hex, amt)"""
    lt = LIT[lit] if isinstance(lit, str) else lit; V, N, A, F, FM, FP, soft = mesh.done(); L = np.asarray(lt['L'], float); L = L / np.linalg.norm(L)
    w, h = int(W * ss), int(H * ss); p, z = cam.project(V); X = (p[:, 0] * ss).astype(np.float64); Y = (p[:, 1] * ss).astype(np.float64); z = z.astype(np.float64)
    iz = np.zeros((h, w), f32); tri = np.full((h, w), -1, np.int32); b0 = np.zeros((h, w), f32); b1 = np.zeros((h, w), f32)
    _raster(X, Y, z, F, iz, tri, b0, b1, max(.05, R * .02))
    n = 1400 if shadow else 4; SM = np.full((n, n), -1e9, f32); SMM = np.zeros((3, 4))
    if shadow:
        c = np.asarray(V[z > 0].mean(0) if sm_c is None else sm_c, float); r = sm_r or float(np.percentile(np.linalg.norm(V - c, axis=1), 96)) * 1.05; a = np.cross(L, [0, 0, 1.]); a = a / (np.linalg.norm(a) + 1e-9) if np.linalg.norm(a) > .05 else np.array([1., 0, 0]); b = np.cross(a, L)
        SMM[0, :3] = a * n / (2 * r); SMM[1, :3] = b * n / (2 * r); SMM[2, :3] = L; SMM[0, 3] = n / 2 - SMM[0, :3] @ c; SMM[1, 3] = n / 2 - SMM[1, :3] @ c
        q = V @ SMM[:, :3].T + SMM[:, 3]; cast = np.isin(FM, (M_BODY, M_HORN, M_SKIN, M_JAW, M_LEG, M_FIN, M_CLAW)); _raster_o(q[:, 0], q[:, 1], q[:, 2], F[cast], SM)
    out = np.zeros((h, w, 3), f32); pid = np.full((h, w), -1, np.int32); vis = np.ones((h, w), f32); pat = np.zeros((h, w), f32); em = np.zeros((h, w), f32); nrm = np.zeros((h, w, 3), f32)
    K = np.array([cam.f * ss, R, f / 24., burn, scales, wet, furn, term, hatch, lw[2] * ss * .5, 1. if shadow else 0., .8, 0.], float)
    _shade(tri, b0, b1, iz, F, N, A, V.astype(f32), FM, FP, cam.eye.astype(float), L, lt['pal'].astype(f32), FIRE5, K, SM, SMM, np.asarray(mouth if mouth is not None else (0, 0, 1e6), float), out, pid, vis, pat, em, nrm)
    e1 = np.zeros((h, w), np.uint8); e2 = np.zeros((h, w), np.uint8); _edges(pid, iz, nrm, cam.f * ss, R, e1, e2, np.append(soft, 0))
    mask = (pid >= 0).astype(f32); zz = np.where(iz > 0, 1 / np.maximum(iz, 1e-9), 1e9); zr = float(np.percentile(zz[iz > 0], 12)) if (iz > 0).any() else 1.
    near = cv2.erode(zz.astype(f32), np.ones((9, 9), np.uint8)); ws = np.clip((zr / np.maximum(near, 1e-3)) ** .55, .34, 1.25)
    nz = _screen_noise(h, w); ws = ws * (1 + .34 * nz)
    lc = rgb(lt['line'])
    if fog is not None:
        k = np.clip((zz - fog[1]) / (fog[2] - fog[1]), 0, 1)[..., None] * fog[3] * (1 - em[..., None] * .7); out = out * (1 - k) + rgb(fog[0]) * k
    # rim: the neighbour toward the light is far behind
    if rim > 0:
        rx, ry = (np.array(rd if rd is not None else lt['rd'], float)); l = math.hypot(rx, ry); rw = lt['rw'] * ss; dx, dy = rx / l * rw, ry / l * rw
        izs = cv2.warpAffine(iz, np.float32([[1, 0, -dx], [0, 1, -dy]]), (w, h), flags=cv2.INTER_NEAREST); zs = np.where(izs > 0, 1 / np.maximum(izs, 1e-9), 1e9)
        hairp = np.append(soft, 0)[pid] > 0; mk = ((zs - zz) > 1.0 * R) & (pid >= 0); mk = mk.astype(f32) * np.where(hairp, .55, 1.) * (.35 + .65 * np.clip(vis + .2, 0, 1)) * (1 - em)
        mk = cv2.GaussianBlur(mk, (0, 0), 1.1 * ss) * mask; mk = smooth(.12, .7, mk) * rim
        out = out * (1 - mk[..., None]) + rgb(lt['rim']) * mk[..., None] * 1.12
    if grad is not None:
        gx, gy = grid(h, w); p0, p1 = np.array(grad[0], float) * ss, np.array(grad[1], float) * ss; d = p1 - p0; t = np.clip(((gx - p0[0]) * d[0] + (gy - p0[1]) * d[1]) / (d @ d), 0, 1); g = ramp(t, grad[2]); ek = em[..., None]; out = out * (g * (1 - ek) + ek)
    if sil is not None: out = out * (1 - sil[1]) + rgb(sil[0]) * sil[1]
    inkA = np.clip(pat, 0, 1) * np.where(vis > .5, .74, .95)
    fcol = np.where((em > .3)[..., None], rgb('#5a1008'), lc); out = out * (1 - inkA[..., None]) + fcol * inkA[..., None]
    d1 = cv2.distanceTransform(1 - e1, cv2.DIST_L2, 5); d2 = cv2.distanceTransform(1 - e2, cv2.DIST_L2, 5)
    br = np.where(vis > .5, smooth(-.75, -.35, nz), 1.)                     # broken where it is lit
    a1 = np.clip(lw[0] * ss * .5 * ws - d1 + .5, 0, 1); a2 = np.clip(lw[1] * ss * .5 * ws * (.45 + .55 * br) - d2 + .5, 0, 1) * (.25 + .75 * br)
    al = np.maximum(a1, a2 * mask); hot = cv2.dilate(em, np.ones((5, 5), np.uint8)) > .3; fc = np.where(hot[..., None], rgb('#6a120a') if burn > 0 else rgb('#3a0c10'), lc)
    if sil is not None: fc = fc * (1 - sil[1]) + rgb(sil[0]) * sil[1]
    a = np.maximum(mask, a1); rgb_ = out * (1 - al[..., None]) + fc * al[..., None]; lay = np.dstack([rgb_ * a[..., None], a])
    lay = cv2.resize(lay, (W, H), interpolation=cv2.INTER_AREA).astype(f32); iz1 = cv2.resize(iz, (W, H), interpolation=cv2.INTER_NEAREST)
    if len(curves):
        cel = Cel(); gi = 0
        for P, Nc, wd in curves:
            q, zc = cam.project(P); fz = np.einsum('ij,ij->i', Nc, cam.eye - P); xi = np.clip(q[:, 0].astype(int), 0, W - 1); yi = np.clip(q[:, 1].astype(int), 0, H - 1)
            ok = (fz > 0) & (q[:, 0] > 0) & (q[:, 0] < W) & (q[:, 1] > 0) & (q[:, 1] < H) & (iz1[yi, xi] < 1. / np.maximum(zc, .01) * 1.035); gi += 1
            if ok.sum() < 5: continue
            i0 = int(np.argmax(ok)); i1 = len(ok) - int(np.argmax(ok[::-1])); seg = q[i0:i1]; k = cam.f * R / float(np.median(zc)) / 190.
            if np.ptp(seg, axis=0).max() > 14: cel.line(seg, wd * np.clip(k, .4, 1.5) * lw[1] / 2.6, lt['line'] if burn < 1 else '#6a0f08', taper=(.18, .55), seed=gi, press=.4, alpha=.92 * (1 - (sil[1] if sil else 0)))
        l = cel.layer(); lay = lay * (1 - l[..., 3:4]) + l
    li = None
    if light and (em > .05).any():
        q = cv2.resize(out * em[..., None] * mask[..., None], (W // 4, H // 4), interpolation=cv2.INTER_AREA); li = cv2.resize((.42 * cv2.GaussianBlur(q, (0, 0), 5) + .36 * cv2.GaussianBlur(q, (0, 0), 18) + .25 * cv2.GaussianBlur(q, (0, 0), 46)).astype(f32), (W, H)) * (.8 if burn < .5 else .42)
    return dict(layer=lay, light=li, iz=iz1, em=em)


_nzc = {}


def _screen_noise(h, w):
    if (h, w) not in _nzc: _nzc[h, w] = np.clip(fbm(h, w, (70, 26), (1, .6), 41), -1.5, 1.5) / 1.5
    return _nzc[h, w]


# ---------------------------------------------------------------- the whole dragon
U_, R_U = np.array([0, .03, .09, .22, .5, .74, .9, 1.]), np.array([.74, .74, .80, .98, 1.0, .80, .44, .05])


class Dragon:
    def __init__(self, ctrl, R=6., up=(0, 0, 1), n=420, around=34, head=1.08, legs=(.16, .50), roll=0., mane_step=.55, seed=0):
        self.P, self.L = cr3(ctrl, n); self.R = R; self.n = n; self.m = around; u = np.linspace(0, 1, n); self.rad = R * np.interp(u, U_, R_U) * (1 + .03 * np.sin(u * self.L / R * .9)); self.fr = frames(self.P, up); self.hs = head * R; self.legs = legs; self.seed = seed; self.ms = mane_step

    def at(self, u):
        i = int(np.clip(u, 0, 1) * (self.n - 1)); return self.P[i], self.fr[0][i], self.fr[1][i], self.fr[2][i], self.rad[i]

    def mesh(self, cam, f=0., jaw=.3, wind=(0, 0, 0), hd=(0., 0., 0.), fire=False, look=(0, 0), with_head=True, whisk=1., grip=(.35, .35, .35, .35), pose=((.5, -.2),) * 4, flick=1., Rm=None, hC=None, u1=1., hair=1., tongues=0, breath=0., eyes=1., whz=1.):
        me = Mesh(); T, U, S = self.fr; n1 = max(8, int(self.n * u1)); s = slice(0, n1)
        tube(me, self.P[s], self.rad[s], M_BODY, m=self.m, flat=1.05, unit=self.R, u0=0, u1=u1, fr=(T[s], U[s], S[s]), sec=lambda th: 1 - .07 * smooth(2.2, 2.7, np.abs(th)) + .05 * np.exp(-(th / .28) ** 2))
        st = self.ms * self.R; N_ = int(self.L * u1 / st); i = np.clip(((np.arange(N_) + .3) * st / self.L * (self.n - 1)).astype(int), 0, n1 - 1); i = i[(i > 10)]
        g = RNG(self.seed, 5); ro, di, Ln, wd = [], [], [], []
        for a in i:
            d = np.linalg.norm(self.P[a] - cam.eye)
            if cam.f * self.R / d < 5: continue
            for q in range(3 if cam.f * self.R / d > 40 else 2):
                r = self.rad[a]; o = (q - 1) * .16 + g.normal(0, .05); ro.append(self.P[a] + U[a] * r * 1.0 + S[a] * r * o + T[a] * r * g.normal(0, .25)); di.append(U[a] * (.75 + .3 * g.random()) + T[a] * 1.0 + S[a] * (o * 1.6 + g.normal(0, .1))); k = g.random(); Ln.append(r * (1.35 if k < .25 else .95 if k < .6 else .62)); wd.append(r * (.21 if k < .25 else .15 if k < .6 else .09))
        if ro: ribbons(me, ro, di, Ln, wd, cam.eye, M_FIRE if fire else M_MANE, wind=wind, f=f, seed=self.seed + 9, n=11, sway=.13 if not fire else .2, curl=.5, unit=self.R, hz=.8 if not fire else 1.9)
        if tongues:
            g = RNG(self.seed, int(f) // 3, 8); ro, di, Ln, wd = [], [], [], []
            for q in range(tongues):
                a = int(g.integers(4, n1 - 2)); th = g.uniform(-3.14, 3.14); r = self.rad[a]; o = U[a] * math.cos(th) + S[a] * math.sin(th); ro.append(self.P[a] + o * r * .9); di.append(o * .55 + T[a] * 1.0 + g.normal(0, .15, 3)); k = g.random(); Ln.append(r * (2.2 if k < .15 else 1.4 if k < .5 else .8) * (.8 + .4 * g.random())); wd.append(r * (.36 if k < .15 else .24 if k < .5 else .13))
            ribbons(me, ro, di, Ln, wd, cam.eye, M_FIRE, wind=wind, f=f, seed=int(f) // 3, n=11, sway=.18, curl=.7, unit=self.R, hz=2.2)
        info = dict(legs=[])
        for q, uu in enumerate(np.repeat(self.legs, 2) if len(self.legs) else ()):
            if uu > u1: continue
            p, t, uv, sv, r = self.at(uu); d = np.linalg.norm(p - cam.eye)
            if cam.f * self.R / d < 9: continue
            info['legs'].append(build_leg(me, p, -t, uv, sv, 1 if q % 2 else -1, r, cam.eye, pose[q % len(pose)], grip[q % len(grip)], f, self.R))
        if with_head:
            t, uv = -T[0], U[0]; x = np.cross(t, uv); x /= np.linalg.norm(x); B = np.stack([x, t, np.cross(x, t)], 1)
            y, p, r = hd; cy, sy, cp, sp, cr, sr = math.cos(y), math.sin(y), math.cos(p), math.sin(p), math.cos(r), math.sin(r)
            Q = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]])
            Rm = B @ Q if Rm is None else Rm; C = self.P[0] + t * self.hs * .25 if hC is None else hC
            info.update(build_head(me, C, Rm, self.hs, cam.eye, jaw, f, wind, look, whisk, seed=self.seed, fire=fire, hair=hair, breath=breath, eyes=eyes, whz=whz)); info['Rm'] = Rm; info['C'] = C
        return me, info

    def render(self, cam, f=0., lit='storm', burn=0., furn=0., fire=None, clouds=True, ss=1.5, ccol=None, **k):
        rk = {q: k.pop(q) for q in ('scales', 'lw', 'wet', 'term', 'hatch', 'shadow', 'sm_c', 'sm_r', 'grad', 'fog', 'light', 'rim', 'sil', 'rd') if q in k}
        me, info = self.mesh(cam, f, fire=(burn > .9) if fire is None else fire, **k)
        o = render(me, cam, f, self.R, lit, burn, furn, info.get('throat'), info.get('curves', ()), ss=ss, **rk); o['info'] = info
        o['pts'] = {q: cam.project(info[q][None])[0][0] for q in ('mouth', 'throat', 'nose', 'eyeL', 'eyeR', 'top', 'horn', 'chin') if q in info}
        if clouds:
            for q, l in enumerate(info['legs']):
                p, z = cam.project(l['claw'][None]); z = float(z[0])
                if z < self.R * .5: continue
                s = cam.f * self.R / z
                if s < 14: continue
                S = int(np.clip(s * 3.4, 120, 900)); sp = flame_cloud(S, (int(f) // 2) / 12., seed=q, ang=math.pi * .9, lw=max(1.2, S / 170.)); sprite(o['layer'], o['iz'], sp, p[0, 0], p[0, 1], z - self.R * .1, s * 3.4 / S)
        return o


def body(ctrl, cam, f=0, **k): return Dragon(ctrl, k.pop('R', 6.)).render(cam, f, with_head=k.pop('with_head', False), **k)['layer']


def head(cx, cy, s, yaw=0., pitch=0., roll=0., jaw=.4, expr='rage', light='storm', f=0, furn=0., wind=(0, 0, 0), persp=9., ss=1.5, neck=True, full=False, burn=0., look=(0, 0), hair=1., breath=0., **k):
    """the head alone.  s = px per skull unit; yaw 0 = facing the lens, + turns the snout to screen LEFT; pitch + looks up; persp = eye distance in units (smaller = wilder)"""
    cam = Camera3D((0, -persp, 0), (0, 0, 0), s * persp, cx=cx, cy=cy)
    cy_, sy = math.cos(math.pi - yaw), math.sin(math.pi - yaw); cp, sp = math.cos(pitch), math.sin(pitch); cr, sr = math.cos(roll), math.sin(roll)
    Rm = np.array([[cr, 0, sr], [0, 1, 0], [-sr, 0, cr]]) @ np.array([[cy_, -sy, 0], [sy, cy_, 0], [0, 0, 1]]) @ np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    me = Mesh(); info = build_head(me, np.zeros(3), Rm, 1., cam.eye, jaw, f, wind, look, fire=burn > 1, hair=hair, breath=breath)
    if neck:
        P = np.array([(0, -1.2, -.05), (0, -2.6, -.25), (.2, -4.4, -.9), (.5, -6.5, -2.4), (.6, -9., -5.)]) @ Rm.T; P, _ = cr3(P, 60); u = np.linspace(0, 1, 60); tube(me, P, .92 + .35 * u, M_BODY, up=Rm[:, 2], m=34, unit=1.2, u0=0., u1=.2, sec=lambda th: 1 - .07 * smooth(2.2, 2.7, np.abs(th)))
    o = render(me, cam, f, 1., light, burn, furn, info['throat'], info['curves'], ss=ss, scales=k.pop('scales', 30), **k); o['info'] = info
    o['pts'] = {q: cam.project(info[q][None])[0][0] for q in ('mouth', 'throat', 'nose', 'eyeL', 'eyeR', 'top', 'horn', 'chin')}
    return o if full else o['layer']
