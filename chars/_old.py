"""OLD ERA drawing kit: head projection with jaw drop, big grins, eyes, hair clumps, edge hatching, light palettes."""
import numpy as np, math, cv2
from op import W, H as HH
from op.head3 import Head
from op.ink import Cel, spl, resample
from op.kit import spike, outline, rim, para, shifted
from op.paint import rgb, smooth, fbm, f32, grid, Canvas
from op.comp import over, blur

INK = '#17131a'
LIGHT = dict(
    gold=dict(skin='#bd7c50', shade='#78412f', deep='#452630', lite='#e9ae6c', line='#170f16', hair='#18151d', hair2='#0b090e', hi='#3c4459', hl='#34303d', white='#f1e6d2', white_sh='#a78d86',
              tooth='#fdf3de', tooth_sh='#cfae90', tooth_dk='#94705f', gum='#a5474c', mouth='#34101a', tongue='#b4525a', rim='#fff2c4', rim2='#ffb347', cool='#6d7da0'),
    storm=dict(skin='#a9a0ad', shade='#5c6180', deep='#343a5c', lite='#d3d3de', line='#0f1020', hair='#12131f', hair2='#080912', hi='#4b5b85', hl='#2c3148', white='#e4e8f2', white_sh='#8a90b0',
               tooth='#eef1f8', tooth_sh='#a3aac4', tooth_dk='#666d8c', gum='#7a4462', mouth='#231226', tongue='#8e4a66', rim='#f2f8ff', rim2='#9fc3ff', cool='#5a74b0'),
    lantern=dict(skin='#dc9a5f', shade='#8f4c34', deep='#4a232b', lite='#ffd08a', line='#1b0d10', hair='#a42a1c', hair2='#4f1012', hi='#f07a38', hl='#6e1714', white='#fbeccb', white_sh='#b78a66',
                 tooth='#fdf0d0', tooth_sh='#c79a6e', tooth_dk='#8a5a42', gum='#a8413f', mouth='#3a0d12', tongue='#b4504a', rim='#fff0b0', rim2='#ff9d3a', cool='#4a4f86'),
    lamp=dict(skin='#e6a872', shade='#ab6444', deep='#6a3230', lite='#ffd79c', line='#1b1014', hair='#c2301f', hair2='#6d1515', hi='#f5763c', hl='#841a18', white='#fff3dc', white_sh='#c59c7c',
              tooth='#fff6e2', tooth_sh='#d7b08a', tooth_dk='#9a6f58', gum='#b04a4a', mouth='#3d1016', tongue='#c05a58', rim='#fff0bd', rim2='#ffb04a', cool='#6a5a86'),
    day=dict(skin='#f0b98d', shade='#c9805c', deep='#8f5142', lite='#ffdcb2', line='#17131a', hair='#1b1820', hair2='#0c0b10', hi='#4b5872', hl='#383544', white='#ffffff', white_sh='#b9c0d6',
             tooth='#ffffff', tooth_sh='#d9d3d6', tooth_dk='#a79aa4', gum='#c65a5e', mouth='#4a141c', tongue='#d4666a', rim='#fffbe6', rim2='#ffd27a', cool='#8fa8d6'),
    sunset=dict(skin='#e2975f', shade='#a4573f', deep='#633036', lite='#ffcf8c', line='#1a1016', hair='#b32c1e', hair2='#5d1216', hi='#f4743a', hl='#7a1817', white='#fff0d8', white_sh='#bb917e',
                tooth='#fff4de', tooth_sh='#d2a682', tooth_dk='#966a58', gum='#a8464a', mouth='#3a1018', tongue='#b8545a', rim='#fff2c0', rim2='#ffab45', cool='#7f6f9c'))


def A(*p): return np.array(p, float)


class FX:
    """head3.Head + jaw drop + cheek puff; author everything in front coordinates."""
    def __init__(self, props, cx, cy, s, yaw=0., pitch=0., roll=0., drop=0., puff=0., anchor=None, persp=.10):
        self.H = h = Head(**props); h.pose(0, 0, s, yaw, pitch, roll, persp); self.s = s; self.k = s / 690.; self.yaw = yaw; self.drop = drop; self.puff = puff
        if anchor is not None:
            q = h.P(np.array([anchor], float))[0]; cx, cy = cx - q[0], cy - q[1]
        h.pose(cx, cy, s, yaw, pitch, roll, persp); self.cx, self.cy = cx, cy
        o = h.P(A((0, -.46)))[0]; c = h.P(A((0, -.92)))[0]; self.o = o; self.d = (c - o) / np.hypot(*(c - o)); self.r = A(self.d[1], -self.d[0])

    def J(self, p):
        p = np.asarray(p, float); q = p - self.o; v = q @ self.d / self.s; u = q @ self.r / self.s
        dv = self.drop * smooth(.04, .46, v); du = self.puff * np.exp(-((v + .02) / .24) ** 2) * np.sign(u) * smooth(.22, .5, np.abs(u))
        return p + np.outer(dv * self.s, self.d) + np.outer(du * self.s, self.r)

    def P(self, xy, dz=0., xc=.56):
        xy = np.atleast_2d(np.asarray(xy, float)); x = np.clip(xy[:, 0], -xc, xc)
        return self.H.P3(np.c_[xy, self.H.z(x, xy[:, 1]) + dz])

    def N(self, xyd):
        """nose / relief points (x, y, lift from the centre line)"""
        q = np.asarray(xyd, float); return self.H.P3(np.c_[q[:, 0], q[:, 1], self.H.z(0 * q[:, 0], q[:, 1]) + q[:, 2]])

    def S(self, xy, dz=0., closed=False, corners=(), n=10): return spl(self.P(xy, dz), closed, corners, n)

    def face(self, top=.75): return self.J(self.H.face(top))


def mir(p, sg):
    p = np.array(p, float); p[:, 0] *= sg; return p


def edge_hatch(cel, P, n, L, ang, w, col, seed=0, spread=.35, along=None):
    """a cluster of n pen strokes crossing a shadow edge P (screen polyline); L length px, ang = stroke direction"""
    P = resample(P, 2.); r = np.random.default_rng(seed); t = np.linspace(.08, .92, n) + r.normal(0, .25 / n, n)
    for i, u in enumerate(np.clip(t, 0, 1)):
        p = P[int(u * (len(P) - 1))]; a = ang + r.normal(0, spread * .25); ln = L * r.uniform(.55, 1.15) * (1 - .45 * abs(2 * u - 1)); d = A(math.cos(a), math.sin(a)); nn = A(-d[1], d[0])
        q = p + d * ln; m = (p + q) / 2 + nn * ln * r.normal(0, .05)
        cel.line(spl([p - d * ln * .18, m, q], n=8), w * r.uniform(.75, 1.2), col, taper=(.1, .75), press=.3, seed=int(r.integers(1 << 30)))


def poly_u(t, q, f): return f(t, q)


TE = [0, .150, .272, .386, .490, .584, .664, .734, .794, .845, .886]
TL = [0, .092, .186, .286, .384, .476, .560, .636, .704, .766]


def grin(cel, fx, cx, cy, uy, ly, pal, asym=0., gum=0., dz=.0, seed=0, bite=.57, lit=-1, up=1.9, lo=2.7, lip=True, lw=1.0, xo=0.):
    """One Piece grin: banana mouth, teeth in two rows, gum, dark corners.  front coordinates."""
    k = fx.k * lw; t = np.linspace(-1, 1, 81)
    def cyv(t): return cy + asym * t
    def xy(t, q):
        t = np.asarray(t, float); at = np.abs(t); c = cyv(t); yu = uy + (c - uy) * at ** up; yl = ly + (c - ly) * at ** lo
        return np.c_[cx * t + xo, yu + (yl - yu) * np.asarray(q, float)]
    P = lambda t, q: fx.P(xy(t, np.broadcast_to(q, np.shape(t))), dz)
    U, Lw = P(t, 0.), P(t, 1.); gap = uy - ly; M = np.vstack([U, Lw[::-1]])
    if gap > .028:
        rr = np.random.default_rng(seed + 77); sm_ = lambda a, b, v: smooth(a, b, np.abs(v)); cel.fill(M, pal['mouth']); cel.clip(M)
        qT = lambda v: .06 * sm_(.60, 1., v); qB = lambda v: 1.02 - .36 * sm_(.46, .98, v) ** 1.3                       # rows shorten toward the molars, dark behind them
        tt = np.linspace(-.945, .945, 61); band = np.vstack([P(tt, qT(tt) - .03), P(tt[::-1], qB(tt[::-1]))]); cel.fill(band, pal['tooth']); cel.ctx.save(); cel._path(band); cel.ctx.clip()
        for a, c in ((.52, 'tooth_sh'), (.76, 'tooth_dk')):
            for sg in (-1, 1):
                q = np.linspace(0, 1, 16); e = a + .05 * np.sin(q * 9 + sg) + .03 * np.sin(q * 21 + 2 * sg); cel.fill(np.vstack([P(sg * e, q), P(np.full(16, sg * 1.), q[::-1])]), pal[c])
        sd = np.linspace(-1, 1, 61); dep = .15 + .05 * np.sin(sd * 7.3 + 1) + .03 * np.sin(sd * 17.) + .07 * (sd * -lit > 0) * np.abs(sd)
        cel.fill(np.vstack([P(sd, -.05), P(sd[::-1], dep[::-1])]), pal['tooth_sh']); cel.fill(np.vstack([P(sd, 1.05), P(sd[::-1], .90 - .03 * np.cos(sd[::-1] * 6) - .36 * sm_(.46, .98, sd[::-1]) ** 1.3)]), pal['tooth_sh'])
        e = [-v + rr.normal(0, .007) for v in TE[::-1][:-1]] + [0.] + [v + rr.normal(0, .007) for v in TE[1:]]; bt = {}
        B_ = lambda v: bite - .075 * np.abs(v) ** 2.2
        for i, (a, b) in enumerate(zip(e[:-1], e[1:])):                                              # each tooth: own height, own face, own weight of line
            c_ = abs(a + b) / 2; off = rr.normal(0, .013) * (1 - .4 * c_); v = np.linspace(a, b, 11); m = np.sin(np.linspace(0, math.pi, 11)) ** .55; bt[i] = off
            if rr.random() < .55: s0 = a if a > 0 else b; s1 = s0 + (b - a) * .26 * (1 if a > 0 else -1); q = np.linspace(.10, float(B_(c_)) - .02, 8); cel.fill(np.vstack([P(np.full(8, s0), q), P(np.linspace(s1, s0 + (s1 - s0) * .4, 8), q)[::-1]]), pal['tooth_sh'], .55)
            cel.line(P(v, B_(v) + off + .020 * (m - .5)), (1.8 + 2.3 * rr.random()) * k * (1 - .45 * c_), pal['line'], taper=(.10, .22), press=.45, seed=seed + 5 + i, tmin=.35)
            if c_ < .5 and rr.random() < .5: cel.fill(P(np.array([b - .010, b + .010, b]), np.array([B_(b) + off + .012, B_(b) + off + .012, B_(b) + off - .05])), pal['line'])
        gq = lambda t: (.03 + gum * .16) * (1 - .5 * np.abs(t))
        if gum > .02:
            pts = []
            for a, b in zip(e[:-1], e[1:]): v = np.linspace(a, b, 7); m = np.sin(np.linspace(0, math.pi, 7)); pts.append(P(v, gq(v) * (1.25 - .8 * m) * (1 + .18 * rr.normal(0, 1))))
            G = np.vstack(pts); cel.fill(np.vstack([P(np.linspace(-1, 1, 41), -.06), G[::-1]]), pal['gum']); cel.line(G, 2.0 * k, pal['line'], taper=(.1, .1), press=.5, seed=seed + 3)
        for i, v in enumerate(e):
            if abs(v) > .90: continue
            av = abs(v); q0 = (.05 + gum * .15) * (1 - .5 * av) + (0. if rr.random() < .5 else rr.uniform(.05, .20)); q1 = float(B_(v)) - rr.uniform(0, .03); qq = np.linspace(q0, q1, 10); bow = rr.normal(.004, .004) * (1 if v >= 0 else -1) * np.sin(np.linspace(0, math.pi, 10))
            cel.line(P(v + bow, qq), (3.0 if abs(v) < .01 else 1.1 + 1.9 * rr.random()) * k * (1 - .5 * av), pal['line'], taper=(rr.uniform(.25, .7), .05), press=.4, seed=seed + 20 + i)
        for i, v in enumerate([-v for v in TL[::-1]][:-1] + TL):
            if abs(v) > .80 or gap < .12: continue
            v = v + rr.normal(0, .010); qq = np.linspace(float(B_(v)) + rr.uniform(.01, .05), float(qB(v)) - rr.uniform(.06, .22), 8); cel.line(P(v * (1 - .05 * (qq - bite)) + rr.normal(0, .003) * np.sin(qq * 9), qq), (.9 + 1.5 * rr.random()) * k * (1 - .4 * abs(v)), pal['line'], taper=(.06, rr.uniform(.4, .8)), press=.4, seed=seed + 60 + i)
        cel.ctx.restore(); cel.line(P(tt, qB(tt))[:20], 2.0 * k, pal['line'], taper=(.2, .6), seed=seed + 14); cel.line(P(tt, qB(tt))[-20:], 2.0 * k, pal['line'], taper=(.6, .2), seed=seed + 15)
        cel.unclip()
    cel.line(U, 5.0 * k, pal['line'], taper=(.10, .10), press=.22, seed=seed + 1)
    cel.line(Lw, (6.2 if gap > .03 else 5.4) * k, pal['line'], taper=(.14, .14), press=.22, seed=seed + 2)
    # corner tucks
    for sg in (-1, 1):
        c = cyv(sg); x = cx * sg + xo
        cel.line(fx.S([(x - .015 * sg, c - .045), (x + .022 * sg, c - .012), (x + .036 * sg, c + .038), (x + .022 * sg, c + .075)], dz), 4.2 * k, pal['line'], taper=(.4, .5), seed=seed + 9 + sg)
    if lip:
        w = cx * (.30 + .25 * min(1, gap / .3)); y = ly - .045 - .012 * min(1, gap / .3)
        cel.line(fx.S([(-w + xo, y + .020), (xo, y - .006), (w + xo, y + .020)], dz - .01), 3.6 * k, pal['line'], taper=(.45, .45), seed=seed + 12)
    return M


def laugh(cel, fx, cx, cy, uy, ly, pal, seed=0, up=2.4, lo=2.2, tongue=.42, lw=1.0, asym=0., low=False, gap_tooth=False, wob=0.):
    """big open mouth: upper teeth band, tongue, dark throat"""
    k = fx.k * lw; t = np.linspace(-1, 1, 81)
    def xy(t, q):
        t = np.asarray(t, float); at = np.abs(t); c = cy + asym * t; yu = uy + (c - uy) * at ** up + wob * np.sin(t * 9.0); yl = ly + (c - ly) * at ** lo + wob * np.sin(t * 7.0 + 1.3)
        return np.c_[cx * t, yu + (yl - yu) * np.asarray(q, float)]
    P = lambda t, q: fx.P(xy(t, np.broadcast_to(q, np.shape(t))), -.01)
    U, Lw = P(t, 0.), P(t, 1.); M = np.vstack([U, Lw[::-1]]); cel.fill(M, pal['mouth']); cel.clip(M)
    tt = np.linspace(-.8, .8, 41); cel.fill(np.vstack([P(tt, 1.05), P(tt[::-1], 1 - tongue * (1 - (tt[::-1] / .8) ** 2) ** .6 + .05 * np.exp(-(tt[::-1] / .1) ** 2))]), pal['tongue'])
    cel.line(P(np.zeros(9), np.linspace(1 - tongue * .72, .97, 9)), 2.6 * k, pal['gum'], taper=(.3, .3), seed=seed + 8)
    b = np.linspace(-.97, .97, 61); h = .26 * (1 - .35 * np.abs(b) ** 2); T = np.vstack([P(b, -.05), P(b[::-1], h[::-1])]); cel.fill(T, pal['tooth'])
    cel.fill(np.vstack([P(b, -.05), P(b[::-1], (h * .34)[::-1])]), pal['tooth_sh']); cel.line(P(b, h), 2.6 * k, pal['line'], taper=(.05, .05), seed=seed + 3)
    for i, v in enumerate([-v for v in TE[:0:-1]] + TE):
        if abs(v) < .86: cel.line(P(np.full(6, v), np.linspace(.03, .25 * (1 - .35 * v * v), 6)), 1.8 * k, pal['line'], taper=(.4, .1), seed=seed + 30 + i)
    if gap_tooth:
        v = np.linspace(.0, .125, 7); cel.fill(np.vstack([P(v + .13, 0. * v), P(v[::-1] + .13, 0. * v + .285)]), pal['mouth'])
    if low:
        b = np.linspace(-.72, .72, 41); cel.fill(np.vstack([P(b, 1.06), P(b[::-1], .90 + .03 * (b / .72) ** 2)]), pal['tooth']); cel.line(P(b, .90 + .03 * (b / .72) ** 2), 2.2 * k, pal['line'], seed=seed + 4)
    cel.unclip()
    cel.line(U, 4.8 * k, pal['line'], taper=(.08, .08), seed=seed + 1); cel.line(Lw, 6.2 * k, pal['line'], taper=(.12, .12), seed=seed + 2)
    for sg in (-1, 1):
        c = cy + asym * sg; x = cx * sg
        cel.line(fx.S([(x - .01 * sg, c - .05), (x + .026 * sg, c - .01), (x + .04 * sg, c + .05)], 0), 4. * k, pal['line'], taper=(.4, .5), seed=seed + 9 + sg)
    cel.line(fx.S([(-cx * .32, ly - .042), (0, ly - .066), (cx * .32, ly - .042)], -.01), 3.6 * k, pal['line'], taper=(.45, .45), seed=seed + 12)
    return M


def eye(cel, fx, sg, pal, lid=.6, sq=0., gx=0., gy=0., r=.033, x=.305, y=0., w=.16, h=.10, slant=0., lw=1., seed=0, bag=True, white=True, hl=True):
    """almond/round eye with heavy upper lid; lid 0 shut .. 1 wide; sq = lower lid pushed up"""
    k = fx.k * lw; q = lambda L: mir([(x + u * w, y + v * h + slant * u * .5) for u, v in L], sg)
    up = q([(-1, -.35), (-.62, -.1 + .62 * lid), (0, .05 + .95 * lid), (.6, -.05 + .80 * lid), (1.02, -.12 + .10 * lid)])
    lo = q([(-1, -.35), (-.5, -.78 + .55 * sq), (.05, -.95 + .70 * sq), (.6, -.72 + .55 * sq), (1.02, -.12 + .10 * lid)])
    U, Lw = spl(fx.P(up), n=8), spl(fx.P(lo), n=8); E = np.vstack([U, Lw[::-1]])
    if lid > .04:
        cel.fill(E, pal['white']); cel.clip(E)
        cel.fill(np.vstack([U - A(0, 20 * k), (U + A(0, (.34 * h * fx.s)))[::-1]]), pal['white_sh'])
        c = fx.P([(sg * x + gx, y + gy)])[0]; a = np.linspace(0, 2 * math.pi, 26)[:-1]; R = r * fx.s
        cel.fill(np.c_[c[0] + R * np.cos(a) * .95, c[1] + R * 1.08 * np.sin(a)], pal['line'])
        if hl: cel.fill(np.c_[c[0] - R * .35 + R * .30 * np.cos(a), c[1] - R * .45 + R * .30 * np.sin(a)], '#ffffff')
        cel.unclip()
    cel.line(U, 8.2 * k, pal['line'], taper=(.22, .16), press=.2, seed=seed + sg); cel.line(Lw, 2.6 * k, pal['line'], taper=(.35, .35), seed=seed + 4 + sg)
    cel.line(spl(fx.P(q([(-.75, .35 + .7 * lid), (0, .62 + .95 * lid), (.75, .40 + .72 * lid)])), n=8), 2.6 * k, pal['line'], taper=(.4, .5), seed=seed + 8 + sg)
    if bag:
        cel.line(spl(fx.P(q([(-.55, -1.15 + .35 * sq), (.15, -1.48 + .40 * sq), (.95, -.95 + .3 * sq)])), n=8), 2.6 * k, pal['line'], taper=(.5, .5), seed=seed + 12 + sg)
    return E


def arc_eye(cel, fx, sg, pal, x=.31, y=0., w=.15, h=.09, lw=1., seed=0, up=1):
    """shut laughing / crying eye: one thick arc + crow's feet"""
    k = fx.k * lw
    cel.line(fx.S(mir([(x - w, y - h * .5 * up), (x - w * .5, y + h * .25 * up), (x, y + h * .45 * up), (x + w * .6, y + h * .15 * up), (x + w, y - h * .55 * up)], sg)), 8. * k, pal['line'], taper=(.3, .22), seed=seed + sg)
    for i, (a, b) in enumerate(((.10, .10), (.16, .0), (.12, -.10))):
        cel.line(fx.S(mir([(x + w + .02, y - h * .2 + b * .35), (x + w + .02 + a * .6, y - h * .2 + b)], sg)), 2.6 * k, pal['line'], taper=(.2, .6), seed=seed + 7 + i)


def brow(cel, fx, sg, pal, pts, lw=1., seed=0, col=None):
    """pts = closed outline in right-side front coordinates (inner bottom, inner top .. tip .. bottom)"""
    Q = fx.S(mir(pts, sg), .015, True, (0, 1, 3)); cel.fill(Q, col or pal['hair']); outline(cel, Q, 2.6 * fx.k * lw, pal['line'], seed + sg)
    p = fx.P(mir(pts, sg), .015)
    for i in range(6):
        u = .12 + .13 * i; a = p[1] * (1 - u) + p[3] * u; b = p[0] * (1 - u) + p[3] * u
        cel.line(spl([b * .8 + a * .2, (a + b) / 2 + (p[3] - p[1]) * .05, a + (p[3] - p[1]) * .07], n=6), 1.8 * fx.k * lw, pal['hl'] if col is None else pal['hair2'], taper=(.3, .6), seed=seed + 5 + i)


def clump(cel, a, tip, b, pal, k, seed=0, bulge=.16, bend=0., lines=3, hi=.0, hcol=None, w=2.4, fill=None, edge=None):
    """one inked hair clump in screen px, with interior lines and a highlight blade"""
    a, tip, b = A(*a), A(*tip), A(*b); S = spike(a, tip, b, bulge, bend); n = len(S) // 2; L, Rr = S[:n + 1], S[n:][::-1]
    m = min(len(L), len(Rr)); L, Rr = L[:m], Rr[:m]; cel.fill(S, fill or pal['hair']); r = np.random.default_rng(seed + 11)
    for i in range(lines):
        u = (i + .7 + r.uniform(-.2, .2)) / (lines + .4); e = int(m * r.uniform(.6, .96)); s0 = int(m * r.uniform(0, .3))
        if e - s0 > 3: cel.line((L * (1 - u) + Rr * u)[s0:e], w * .72 * k, pal['hl'], taper=(.35, .6), press=.3, seed=seed * 7 + i)
    if hi > 0:
        s0, e = int(m * .18), int(m * (.18 + .55 * hi)); u0, u1 = .30, .62
        if e - s0 > 3:
            x0 = (L * (1 - u0) + Rr * u0)[s0:e]; x1 = (L * (1 - u1) + Rr * u1)[s0:e]; t = np.sin(np.linspace(0, math.pi, e - s0))[:, None]; mid = (x0 + x1) / 2
            cel.fill(np.vstack([mid + (x0 - mid) * t, (mid + (x1 - mid) * t)[::-1]]), hcol or pal['hi'])
    cel.line(L, w * k, edge or pal['hl'], taper=(.1, .25), press=.3, seed=seed * 5 + 1); cel.line(Rr, w * k, edge or pal['line'], taper=(.1, .25), press=.3, seed=seed * 5 + 2)
    return S


def rim(layer, dx, dy, col, amt=1.0, soft=.6, where=None):
    """kit.rim that ignores the picture border (a cel running off frame gets no false edge there)"""
    a = layer[..., 3]; h, w = a.shape; sh = cv2.warpAffine(a, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE); m = np.clip(a - sh, 0, 1)
    if soft > 0: m = cv2.GaussianBlur(m, (0, 0), soft) * (a > .02)
    if where is not None: m = m * where
    c = rgb(col) if isinstance(col, str) else np.asarray(col, f32); out = layer.copy(); out[..., :3] = out[..., :3] * (1 - m[..., None] * amt) + c * (m * a)[..., None] * amt; return out


def rimlit(layer, dx, dy, pal, k=1., amt=1., wide=26., hot=9., where=None):
    """soft orange falloff + hard hot edge; light from (-dx,-dy) (unit-ish direction)"""
    o = rim(layer, dx * wide * k, dy * wide * k, pal['rim2'], .62 * amt, 7 * k, where)
    return rim(o, dx * hot * k, dy * hot * k, pal['rim'], amt, .8, where)


def dirt(h, w, strokes, col, seed=0, op=.5):
    cv = Canvas(h, w, color=col, layer=True)
    for i, (P, wd) in enumerate(strokes):
        Q = resample(np.asarray(P, float), max(3., wd * .4)); idx = np.linspace(0, len(Q) - 1, 10).round().astype(int)
        cv.strokes(Q[idx][None], wd / 2, rgb(col)[None], op=op, dry=.72, seed=seed + i, shape='flat', bristle=.2)
    return cv.layer()


def inside(layer, mask_layer):
    return layer * mask_layer[..., 3:4]


def over_l(a, b):
    """premultiplied layer b over layer a"""
    return a * (1 - b[..., 3:4]) + b


def nose(C, fx, pal, lw=1., sz=1., side=1, seed=0, light=True):
    """plain straight nose (Oda): one side line, tip, small shade"""
    k = fx.k * lw; N = fx.N; z = sz
    C.fill(spl(N([(.045 * side, .05, .04), (.055 * side * z, -.12, .08), (.055 * side * z, -.25, .12 * z), (.025 * side, -.33, .14 * z), (.085 * side * z, -.345, .08), (.11 * side * z, -.28, .05), (.085 * side, -.1, .04)]), True), pal['shade'])
    C.line(spl(N([(.060 * side, .06, .03), (.050 * side, -.06, .07), (.055 * side * z, -.20, .10 * z), (.058 * side * z, -.29, .13 * z)])), 4.0 * k, pal['line'], taper=(.55, .25), seed=seed)
    C.line(spl(N([(-.050 * z, -.325, .10 * z), (-.02, -.36, .14 * z), (.025, -.36, .14 * z), (.066 * side * z, -.30, .12 * z)])), 4.2 * k, pal['line'], taper=(.35, .25), seed=seed + 1)
    for sg in (-1, 1): C.line(spl(N(mir([(.075 * z, -.31, .07), (.098 * z, -.345, .05), (.070 * z, -.372, .06)], sg))), 3.2 * k, pal['line'], taper=(.5, .4), seed=seed + 2 + sg)


def tear(C, P, w, pal=None, seed=0, bead=True):
    """a fat rope of tears along P (screen px), outlined, with a highlight and a bead at the end"""
    P = spl(P, n=10); n = len(P); t = np.linspace(0, 1, n); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; nn = np.c_[-T[:, 1], T[:, 0]]; r = np.random.default_rng(seed)
    hw = w * .5 * (.50 + .50 * t + .14 * np.sin(t * 8 + seed)) * np.clip(t / .30, .06, 1) ** .6
    S = np.vstack([P + nn * hw[:, None], (P - nn * hw[:, None])[::-1]]); C.fill(S, '#f2f8fd', .80); C.fill(np.vstack([P + nn * (hw * .95)[:, None], (P + nn * (hw * .45)[:, None])[::-1]]), '#a9cdea', .75); C.line(P - nn * (hw * .35)[:, None], w * .20, '#ffffff', taper=(.3, .3), seed=seed); C.line(P + nn * hw[:, None], max(1.4, w * .12), '#2a2a44', taper=(.45, .1), press=.4, seed=seed); C.line(P - nn * hw[:, None], max(1.4, w * .10), '#2a2a44', taper=(.55, .1), press=.4, seed=seed + 1)
    if bead:
        e = P[-1]; a = np.linspace(0, 2 * math.pi, 22)[:-1]; R = w * .62; Q = np.c_[e[0] + R * np.cos(a), e[1] + R * .25 + R * 1.15 * np.sin(a) - R * .5 * (np.sin(a) < -.6) * (np.abs(np.cos(a)) < .6)]; C.fill(Q, '#dceaf6'); C.fill(Q * .55 + (e + A(R * .18, R * .5)) * .45, '#9fc0de'); C.fill(np.c_[e[0] - R * .35 + R * .22 * np.cos(a), e[1] - R * .05 + R * .28 * np.sin(a)], '#ffffff'); outline(C, Q, max(1.6, w * .13), '#2a2a44', seed + 1)
