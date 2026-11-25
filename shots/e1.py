"""e1  Under.  He sinks on his back, arms and legs trailing up; the hat goes the other way."""
import numpy as np, cv2, math
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp, repaint, mix
from op.comp import over, screen, blur, lerp_keys, radial, cam2d, shaft
from op.ink import Cel, spl, resample, noise1, xf
from op.kit import outline, hatch, para
from op import hat as HT
from chars import luffy as L
from shots.d8 import cached
from shots.b3 import caustic

TAU = 2 * math.pi
DISC = (1180., 96.)


def plate():
    x, y = grid(); n = fbm(H, W, (260, 110, 46, 18), seed=5, aspect=1.6); d = np.hypot((x - DISC[0]) / 1.5, y - DISC[1])
    g = ramp(np.clip(d / 1150 + .05 * n, 0, 1), [(0, '#8fe0e6'), (.10, '#3fa6c4'), (.26, '#1c6a9c'), (.48, '#0f3d72'), (.72, '#08214a'), (1, '#040c22')])
    bl = fbm(H, W, (120, 44), seed=9, aspect=3.2); g = g * (1 + .10 * bl * np.clip(1 - d / 900, 0, 1))[..., None]
    g = repaint(np.clip(g, 0, 1.2).astype(f32), (46, 22, 11), seed=7)
    return g.astype(f32)


def limb(pts, wa, wb, n=40):
    """one limb: centre line through pts, two different side profiles (muscle bellies), not a tube"""
    P = resample(spl(np.array(pts, float), n=14), .016); t = np.linspace(0, 1, len(P)); T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; N = np.c_[-T[:, 1], T[:, 0]]; k = np.linspace(0, 1, len(wa))
    sm = lambda w: np.interp(t, np.linspace(0, 1, 60), np.convolve(np.pad(np.interp(np.linspace(0, 1, 60), k, w), 9, 'edge'), np.hanning(21)[1:-1] / np.hanning(21).sum(), 'valid'))
    return P + N * sm(wa)[:, None], P - N * sm(wb)[:, None], P


HATL = dict(HT.LIGHTS['sea'], glow=.55)


class Shot:
    id = 'e1'; n = 60; finish = dict(bloom=.36, vig=.24, soft=.18)
    EVENTS = [(0, 'under_pressure', dict(size=.6, note='from black: deep pressure, muffled')), (8, 'bubbles', dict(size=.3, dur=20, note='last breath leaves')), (5, 'hat_leaves', dict(size=.3, note='soft lift of straw from the head')), (20, 'sea_king_pass', dict(size=.6, dur=40, note='very low, huge body passes behind')), (27, 'bubble_forms', dict(size=.1, dur=9, note='swells at his lips')), (36, 'bubble', dict(size=.3, note='one bubble leaves his lips'))]

    def setup(self):
        self.bg = cached('e1_bg', plate); pass
        r = np.random.default_rng(3); self.bub = [(r.uniform(-1, 1), r.uniform(0, 60), r.uniform(2, 15) ** 1.0, r.uniform(0, 6.28), r.uniform(5, 11)) for _ in range(46)]

    def seaking(self, img, f):
        """an eel-headed king as long as the frame, behind the light"""
        g = f; k = (g - 2) / 58.; cx = 2150 - 2450 * k; u = np.linspace(0, 1, 90); X = cx + (u - .0) * 2700; Y = 420 + 150 * np.sin(u * 5.2 + g * .05) + 60 * np.sin(u * 11 - g * .07) - 120 * (1 - u) + 130 * u; C = np.c_[X, Y]; T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = np.c_[-T[:, 1], T[:, 0]]
        w = np.interp(u, [0, .015, .06, .12, .17, .30, .60, .9, 1], [8, 70, 150, 178, 150, 235, 205, 80, 10]); A, B = C - N * w[:, None], C + N * w[:, None] * np.interp(u, [0, .1, .2, 1], [.7, .95, 1, 1])[:, None]
        c = Cel(W, H, ss=1); body = np.vstack([A, B[::-1]]); base, dk, lt = '#0a2c56', '#07203f', '#2a78a6'
        # dorsal fin: rays and torn web
        for i in range(12, 84, 4):
            h = 150 * math.sin((i - 10) / 80 * 3.1) * (1 + .10 * math.sin(g * .2 + i)); p, q = A[i], A[min(89, i + 5)]; tip = p - N[i] * h + T[i] * 90; c.fill(spl([p, p + (tip - p) * .5 + T[i] * 8, tip, (tip + q) / 2 - N[i] * 20 + T[i] * 10, q], False, n=5), dk); c.line(np.array([p, (p + tip) / 2 + T[i] * 6, tip]), 3, lt, taper=(.1, .6), alpha=.5)
        c.fill(body, base); c.clip(body)
        c.fill(np.vstack([C + N * (w * .30)[:, None] + np.c_[0 * u, 14 * np.sin(u * 60)], B[::-1]]), dk)
        for i in range(8, 88, 2): c.line(spl([B[i], (C[i] + B[i]) / 2 + T[i] * 16, C[i] + N[i] * w[i] * .25], n=5), 2.2, lt, taper=(.2, .5), alpha=.28)
        for j in range(3):
            for i in range(16, 86): p = C[i] - N[i] * w[i] * (.15 + .25 * j) + T[i] * (14 * (j % 2)); c.line(spl([p - N[i] * 9, p + T[i] * 12, p + N[i] * 9], n=4), 1.8, lt, taper=(.3, .3), alpha=.22)
        # scar
        c.line(spl([C[14] - N[14] * 120, C[17] - N[17] * 30, C[16] + N[16] * 40], n=8), 7, lt, alpha=.6)
        c.unclip()
        # gill, jaw, teeth, eye
        c.line(spl([C[13] - N[13] * 90, C[14] + T[14] * 26, C[13] + N[13] * 110], n=8), 5, dk); c.line(spl([C[1], C[5] + N[5] * 46, C[9] + N[9] * 74, C[11] + N[11] * 60], n=8), 5, '#04142c')
        for i in range(2, 9): p = C[i] + N[i] * (12 + 8.5 * i); c.fill(np.array([p - T[i] * 11, p + N[i] * 34 + T[i] * 2, p + T[i] * 11]), '#3f88a8', .45)
        e = C[7] - N[7] * 52; c.fill(L._circ(e[0], e[1], 17, 24, .7), '#9fe8e6', .8); c.fill(L._circ(e[0] - 3, e[1] + 1, 5, 16, 1.9), '#04101e'); c.line(spl([e + [-40, 6], e + [0, -14], e + [44, -22]], n=6), 13, '#04142c')
        lay = c.rgba(); lay = L.rimlight(lay, np.array([-.1, -1.]), 9, '#6fd0e2', .55); lay = cv2.GaussianBlur(lay, (0, 0), 3.4)
        # water between us and it
        fog = .66; lay[..., :3] = lay[..., :3] * (1 - fog) + rgb('#0f4478') * fog * lay[..., 3:4]
        return over(img, lay, float(smooth(6, 22, f)) * .95)

    def figure(self, f):
        g = f // 2 * 2; t = g / 24.; S = 140.; sk = float(lerp_keys(f, [(0, -92), (60, 104)])); rot = float(lerp_keys(f, [(0, -.16), (60, -.34)])); O = np.array([772 + 12 * math.sin(t * 1.3), 690 + sk])
        co, si = math.cos(rot), math.sin(rot); lw = 4.2; C0 = L.LIGHT['deep']['c']
        def B(P): P = np.atleast_2d(np.asarray(P, float)); return O + S * np.c_[P[:, 0] * co - P[:, 1] * si, P[:, 0] * si + P[:, 1] * co]
        d = lambda ph, a=.08: a * math.sin(t * 2.2 + ph); w = lambda i, a=.16: a * math.sin(t * 2.5 - i * .85)
        R = L.Rec(); Lt = L.LIGHT['deep']
        def group(polys, key, shade=(), lines=(), name='g', rimv=.30, sh=None):
            """several pieces, ONE contour: ink all the edges first, then flood them together"""
            R.part(name, rim=rimv); Q = [B(p) for p in polys]
            for q_, p in enumerate(Q): R.edge(p, lw * 2.0, seed=q_ + 3 * len(R.parts))
            for p in Q: R.fill(p, key)
            for p in Q:
                for s_ in shade: R.fill(B(s_), sh or key + '_sh', clip=p)
            for l, w_ in lines: R.line(B(spl(l, n=7)), lw * w_, taper=(.12, .5))
            return Q
        def arm(sh, el, wr, hd, flip, name):
            """shoulder to fingertips: deltoid, biceps, forearm, wrist, a half curled hand"""
            sh, el, wr = [np.array(v, float) for v in (sh, el, wr)]; mid1 = (sh + el) / 2 + np.array([-.10, 0]) * flip; mid2 = (el + wr) / 2 + np.array([.09, -.02]) * flip
            A, C, P = limb([sh, mid1, el, mid2, wr, wr + (wr - mid2) * .18], [.24, .285, .265, .215, .185, .195, .215, .195, .155, .125, .115, .12], [.24, .265, .26, .225, .195, .215, .225, .185, .145, .12, .112, .12])
            c_, s_ = math.cos(hd), math.sin(hd); M = np.array([[c_, s_], [-s_, c_]]); H_ = lambda q: np.asarray(q, float) * np.array([1, flip]) @ M + wr
            palm = H_(spl([(-.06, .115), (.12, .15), (.34, .185), (.46, .13), (.48, -.02), (.40, -.15), (.18, -.16), (-.06, -.115)], True, n=6)); fing = []
            for q_, (y0, l1, l2, c1) in enumerate(((.125, .30, .26, .55), (.055, .34, .28, .72), (-.02, .33, .27, .86), (-.09, .27, .22, 1.02))):
                a0 = -.10 - .10 * q_ + d(q_ * .5, .10); p0 = np.array([.40, y0]); p1 = p0 + l1 * np.array([math.cos(a0), math.sin(a0)]); a1 = a0 - c1 - d(q_ * .7 + 1, .12); p2 = p1 + l2 * .62 * np.array([math.cos(a1), math.sin(a1)]); a2 = a1 - c1 * .55; p3 = p2 + l2 * .5 * np.array([math.cos(a2), math.sin(a2)])
                a, c, _ = limb([p0 - [.08, 0], p0, (p0 + p1) / 2, p1, p2, p3], [.066, .068, .066, .064, .058, .050, .040, .012], [.066, .068, .066, .062, .056, .048, .038, .012]); fing.append(H_(np.vstack([a, c[::-1]])))
            a, c, _ = limb([(.06, -.08), (.20, -.20 - d(2, .03)), (.36, -.27), (.50, -.25)], [.085, .078, .066, .052, .014], [.085, .078, .064, .05, .014]); th = H_(np.vstack([a, c[::-1]]))
            fl = [f_[[0, -1]].mean(0) for f_ in fing]
            pol = [np.vstack([A, C[::-1]])] + fing[::-1] + [palm, th]
            side = C if B(C)[:, 0].mean() < B(A)[:, 0].mean() else A; oth = A if side is C else C; sh_ = np.vstack([side, (side * .62 + oth * .38)[::-1]])
            ln = [([el + [.10 * flip, .08], el + [.0, .02], el + [-.02 * flip, -.10]], .5), ([mid1 + [.10 * flip, .20], mid1 + [.12 * flip, -.1], mid1 + [.05 * flip, -.34]], .42), ([mid2 + [-.10 * flip, .30], mid2 + [-.12 * flip, 0], mid2 + [-.05 * flip, -.28]], .38)]
            group(pol, 'skin', [sh_], ln, name)
            for f_ in fing: n_ = len(f_) // 2; R.line(B(f_[4:n_ - 1]), lw * .52, taper=(.3, .15)); R.line(B(f_[n_ + 1:-int(n_ * .45)]), lw * .45, taper=(.1, .6))
            R.line(B(H_(spl([(.14, -.12), (.28, -.07), (.36, .0)], n=5))), lw * .45, taper=(.2, .6)); R.fill(B(H_(spl([(.0, -.11), (.2, -.16), (.40, -.15), (.46, -.02), (.3, -.06), (.1, -.05)], True, n=5))), 'skin_sh', clip=B(palm)); R.edge(B(th), lw * .62, seed=44)
            return A, C
        sh1, sh2 = np.array([-.84, .02]), np.array([-.60, -.14])
        e2 = np.array([-.10 + d(1), -1.30]); h2 = np.array([.14 + d(1.6, .14), -2.46 + d(.4, .05)]); e1 = np.array([-1.28 + d(.6), -1.22]); h1 = np.array([-1.30 + d(1.2, .16), -2.44 + d(.2, .05)])
        dirn = lambda a, b: math.atan2(*(b - a)[::-1])
        # ---- far arm, far leg
        arm(sh2, e2, h2, dirn(e2, h2) + .62 + d(2, .18), 1, 'farm')
        hip = np.array([1.55, .30]); k2 = np.array([2.86 + d(.3), -.60]); a2 = np.array([3.86 + d(.9, .14), -1.96]); k1 = np.array([3.02 + d(0), .06]); a1 = np.array([4.34 + d(.6, .14), -.88 + d(.2, .06)])
        def leg(k, a, name, fa):
            v = a - k; A, C, P = limb([k - v * .22 + [0, .06], k, (k + a) / 2 + np.array([-v[1], v[0]]) * .035, a, a + v * .07], [.29, .275, .25, .225, .205, .175, .145, .125, .13], [.29, .285, .30, .345, .345, .28, .20, .14, .13]); c_, s_ = math.cos(fa), math.sin(fa); M = np.array([[c_, s_], [-s_, c_]]); q = lambda p: np.asarray(p, float) @ M + a
            F = q(spl([(-.10, -.14), (.14, -.17), (.36, -.22), (.62, -.40), (.84, -.60), (.95, -.60), (.92, -.48), (.70, -.16), (.42, .13), (.18, .19), (-.08, .15)], True, n=7)); low = C if B(C)[:, 1].mean() > B(A)[:, 1].mean() else A; oth = A if low is C else C
            group([np.vstack([A, C[::-1]]), F], 'skin', [np.vstack([low, (low * .66 + oth * .34)[::-1]]), q([(-.1, .04), (.3, .02), (.6, -.22), (1, -.5), (1, 0), (.3, .5), (-.1, .4)])], [([k + v * .06 + [.10, .10], k + v * .16, k + v * .20 + [-.08, -.08]], .45)], name)
            for i in range(4): R.line(B(q([(.90 - .05 * i, -.58 + .075 * i), (.80 - .05 * i, -.47 + .07 * i)])), lw * .5)
            S_ = q(spl([(.12, .21), (.44, .16), (.70, -.14), (.97, -.52), (1.04, -.50), (.78, -.04), (.50, .26), (.14, .30)], True, n=5)); R.fill(B(S_), 'cord'); R.edge(B(S_), lw * .7, seed=25); R.line(B(q(spl([(.72, -.16), (.64, -.36), (.50, -.30)], n=6))), lw * 1.5, key='cord_sh'); R.line(B(q(spl([(.40, .15), (.36, -.06), (.50, -.30)], n=6))), lw * 1.3, key='cord_sh')
        leg(k2, a2, 'fleg', math.atan2(*(a2 - k2)[::-1]) + .12 + d(1.2, .2)); self.short(R, B, hip + [.02, -.06], k2, lw, 'fshort', True)
        # ---- far half of the shirt: a sail of cloth going up
        R.part('sail', rim=.45); T = [(-.95, -.28), (-.72, -.78 + w(0, .08)), (-.30, -1.18 + w(1, .12)), (.25, -1.30 + w(2, .16)), (.78, -1.52 + w(3, .20)), (1.18, -1.96 + w(4, .26)), (1.62, -2.02 + w(5, .30)), (1.94, -1.60 + w(6, .28)), (1.74, -1.16 + w(5, .2)), (1.52, -.72 + w(4, .1)), (1.40, -.30), (.3, -.30)]
        P = B(spl(T, True, n=8)); R.fill(P, 'shirt_sh'); R.fill(B(spl([(-.6, -.35), (-.35, -.85 + w(1)), (.2, -1.0 + w(2)), (.75, -1.2 + w(3)), (1.15, -1.55 + w(4)), (1.5, -1.5 + w(5)), (1.3, -.8 + w(4)), (1.2, -.35)], True, n=7)), 'shirt_dk', clip=P)
        for q_ in (((-.55, -.4), (-.35, -.75), (-.12, -1.05 + w(1))), ((.0, -.4), (.22, -.8), (.36, -1.2 + w(2))), ((.55, -.4), (.80, -.85), (.95, -1.45 + w(3))), ((.95, -.4), (1.25, -.95), (1.42, -1.8 + w(5)))):
            c_ = spl(q_, n=8); R.line(B(c_), lw * .62, clip=P, taper=(.1, .5)); R.fill(B(np.vstack([c_, (c_ + np.c_[.10 * np.sin(np.linspace(0, 3.1, len(c_))), 0 * c_[:, 0]])[::-1]])), 'shirt', clip=P, al=.8)
        R.edge(P, lw, seed=5)
        # ---- head hangs back; neck and torso are one sagging curve
        hp = B([(-2.10, .46 + d(.5, .035))])[0]; roll = -1.66 + rot * .5 + d(.2, .05); Sh = S * 1.02
        e = dict(lu=.08, arc=-.62, mw=.135, yc=-.685, uy=-.628, ly=-.745 - .012 * math.sin(t * 1.9), pu=2.4, tt=.030, bi=.425, bm=.335, bo=.215, bw=.80, bx=-.035, fur=.95, jaw=.05, hat=0., es=1.14)
        neck = spl([(-.72, -.36), (-1.08, -.22), (-1.46, .06), (-1.78, .20), (-1.80, .62), (-1.42, .62), (-1.02, .50), (-.70, .44)], True, n=7)
        tor = spl([(-1.10, -.06), (-.86, -.38), (-.44, -.58), (-.02, -.60), (.38, -.46), (.82, -.32), (1.30, -.32), (1.70, -.24), (1.86, .10), (1.72, .42), (1.30, .50), (.80, .50), (.25, .58), (-.38, .64), (-.86, .52), (-1.08, .28)], True, n=8)
        group([neck, tor], 'skin', [spl([(-1.9, .34), (-1.3, .18), (-.8, .08), (-.35, .12), (.25, .02), (.8, .08), (1.4, .04), (2, .2), (2, .9), (-1.9, .9)], True, n=6)],
              [([(-.62, -.52), (-.44, -.34), (-.10, -.24), (.14, -.34), (.18, -.52)], .62), ([(-1.02, -.16), (-.92, .02), (-.96, .18)], .5), ([(-1.50, .14), (-1.22, .06), (-.98, -.10)], .5)] + [([(.34 + .24 * q_, -.42 + .03 * q_), (.39 + .24 * q_, -.24), (.35 + .24 * q_, -.06)], .42) for q_ in range(4)], 'torso')
        tq = B(tor); sc = B(spl([(-.50, -.62), (-.30, -.38), (-.02, -.42), (.16, -.26), (.06, -.20), (-.14, -.32), (-.34, -.24), (-.56, -.42)], True, n=4)); R.fill(sc, 'scar', clip=tq); R.edge(sc, lw * .4, key='soft', clip=tq)
        # ---- near half of the shirt: filled like a lung, hem streaming up behind the hip
        R.part('shirt', rim=.45); SP = [(-1.02, .02), (-.80, .10), (-.42, .02 + w(1, .04)), (.05, .10 + w(2, .05)), (.52, -.02 + w(3, .07)), (.98, -.18 + w(4, .12)), (1.36, -.62 + w(5, .20)), (1.80, -1.06 + w(6, .28)), (2.14, -1.02 + w(7, .30)), (2.22, -.62 + w(6, .26)), (1.98, -.10 + w(5, .14)), (1.74, .34 + w(4, .08)), (1.42, .72), (.92, .86 + w(2, .05)), (.30, .98 + w(1, .04)), (-.34, .92), (-.82, .70), (-1.06, .36)]
        P = B(spl(SP, True, n=8)); R.fill(P, 'shirt'); R.fill(B(spl([(-1.2, .40), (-.7, .46), (-.2, .56), (.4, .50), (.9, .42), (1.3, .10), (1.7, -.35 + w(6, .2)), (2.1, -.6 + w(6, .2)), (2.4, -.3), (2.0, 1.2), (-1.2, 1.3)], True, n=6)), 'shirt_sh', clip=P)
        for q_ in (((-.86, .12), (-.60, .34), (-.56, .66)), ((-.40, .06), (-.14, .40), (-.18, .84)), ((.10, .14), (.34, .44), (.30, .88)), ((.56, .04), (.82, .34), (.84, .74)), ((1.0, -.14), (1.26, .16), (1.32, .60)), ((1.40, -.56 + w(5, .2)), (1.62, -.2), (1.66, .28)), ((1.82, -.98 + w(6, .25)), (1.92, -.6), (1.86, -.12))):
            c_ = spl(q_, n=8); R.line(B(c_), lw * .62, clip=P, taper=(.08, .55)); R.fill(B(np.vstack([c_, (c_ + np.c_[.12 * np.sin(np.linspace(0, 3.1, len(c_))), 0 * c_[:, 0]])[::-1]])), 'shirt_dk', clip=P); R.fill(B(np.vstack([c_ - [.02, 0], (c_ - np.c_[.02 + .08 * np.sin(np.linspace(0, 3.1, len(c_))), 0 * c_[:, 0]])[::-1]])), 'shirt_hi', clip=P, al=.45)
        R.edge(P, lw, seed=7)
        # ---- sash: two long tails drifting up
        for j, (x0, ln, ph) in enumerate(((1.34, 1.75, 0.), (1.60, 1.35, 1.1))):
            R.part('sash%d' % j, rim=.4); c_ = spl([(x0, -.20), (x0 + .10 + w(j, .05), -.60), (x0 + .02 + w(j + 2, .10), -1.02), (x0 + .16 + w(j + 3, .15), -1.44), (x0 + .20 * j + .10 + w(j + 4, .20), -.36 - ln)], n=10); A, C, _ = limb(c_, [.07, .11, .14, .10, .15, .12, .17], [.07, .10, .11, .15, .10, .15, .09]); Q = B(np.vstack([A, [(A[-16] + C[-16]) / 2], C[::-1]])); R.fill(Q, 'sash'); R.fill(B(np.vstack([C, (A * .35 + C * .65)[::-1]])), 'cord_sh', clip=Q, al=.85); R.edge(Q, lw * .8, seed=9 + j)
            for q_ in (.25, .5, .75): i = int(q_ * len(A)); R.line(B(np.array([A[i], (A[i] + C[i]) / 2 + [.02, .05], C[max(0, i - 10)]])), lw * .4, clip=Q)
        leg(k1, a1, 'leg', math.atan2(*(a1 - k1)[::-1]) + .10 + d(.8, .2)); self.short(R, B, hip, k1, lw, 'short', False)
        R.part('belt', rim=.3); SB = B(spl([(1.20, -.36), (1.56, -.34), (1.66, .06), (1.58, .50), (1.22, .52), (1.28, .08)], True, n=6)); R.fill(SB, 'sash'); R.fill(B([(1.0, .12), (2, .12), (2, .8), (1.0, .8)]), 'cord_sh', clip=SB, al=.8); R.edge(SB, lw * .8, seed=12)
        for q_ in (-.12, .10, .32): R.line(B(spl([(1.26, q_), (1.42, q_ + .04), (1.60, q_)], n=5)), lw * .45, clip=SB)
        # ---- near arm + its sleeve, pushed up to the elbow and full of water
        arm(sh1, e1, h1, dirn(e1, h1) + .70 + d(1.7, .18), 1, 'arm')
        R.part('sleeve', rim=.4); v = e1 - sh1; n_ = np.array([-v[1], v[0]]) / np.hypot(*v); SL = [sh1 + n_ * .42 - v * .10, sh1 + n_ * .52 + v * .25, sh1 + n_ * .50 + v * .58 + w(2, .05), sh1 + n_ * .40 + v * .86 + w(3, .06), sh1 + v * .80, sh1 - n_ * .36 + v * .92 + w(4, .06), sh1 - n_ * .50 + v * .56, sh1 - n_ * .50 + v * .16, sh1 - n_ * .36 - v * .16]
        P = B(spl(SL, True, corners=(3, 5), n=8)); R.fill(P, 'shirt'); R.fill(B(spl([sh1 + n_ * .10 - v * .3, sh1 + n_ * .04 + v * .3, sh1 + n_ * .14 + v * .7, sh1 + n_ * .05 + v * 1.1, sh1 + n_ * 1.2 + v, sh1 + n_ * 1.2 - v * .3], True, n=5)) if B([n_])[0][0] < B([-n_])[0][0] else B(spl([sh1 - n_ * .10 - v * .3, sh1 - n_ * .04 + v * .3, sh1 - n_ * .14 + v * .7, sh1 - n_ * .05 + v * 1.1, sh1 - n_ * 1.2 + v, sh1 - n_ * 1.2 - v * .3], True, n=5)), 'shirt_sh', clip=P)
        for a_, b_, c__ in ((.40, .10, .30), (.16, .30, .62), (-.12, .20, .60), (-.36, .34, .70)): R.line(B(spl([sh1 + n_ * a_ + v * b_, sh1 + n_ * (a_ * .7) + v * (b_ + c__) / 2 + [.04, 0], sh1 + n_ * a_ * .9 + v * c__], n=6)), lw * .55, clip=P, taper=(.1, .55))
        R.edge(P, lw, seed=14)
        lay = R.cel(Lt, S * .62)
        HR, Hd, _, _ = L.record(hp[0], hp[1], Sh, .60, .10, roll, e, 'deep', f, hat=False, body=False, scar=False, wind=2.0, blow=.45, lw=4.2, drift=(.30 + .06 * math.sin(t * 2.1), .12))
        lay = HR.cel(Lt, S * .6, lay, skip=('neck',)); lay = L.rimlight(lay, np.array([.38, -.92]), 7.5, '#e6fff8', 1.)
        self.mouth = Hd.P(np.array([(0., -.66)]))[0]; self.roll = roll; self.Hd = Hd
        return lay

    def short(self, R, B, hip, k, lw, name, far):
        R.part(name); v = k - hip; n = np.array([-v[1], v[0]]) / np.hypot(*v); m = (hip + k) / 2; P = [hip + n * .46 - v * .12, m + n * .46, k + n * .42 - v * .06, k - n * .42 - v * .06, m - n * .48, hip - n * .48 - v * .10]
        Q = B(spl(P, True, corners=(2, 3), n=7)); C = L.LIGHT['sea']['c']; b = np.array([.10, .21, .42]) * (.7 if far else 1); R.fill(Q, tuple(b)); low = P if n[1] > 0 else P; sgn = 1 if n[1] > 0 else -1
        R.fill(B(spl([hip + n * sgn * .06, m + n * sgn * .10, k + n * sgn * .04, k + n * sgn * .8, hip + n * sgn * .8], True, n=5)), tuple(b * .55), clip=Q)
        for q in (.12, .40, .66): p = hip + v * q; R.line(B(spl([p + n * .40, p + n * .1 + v * .07, p - n * (.2 + q * .2)], n=5)), lw * .55, clip=Q)
        R.edge(Q, lw, seed=20)
        e = k - v * .06; F = []                              # white fur cuff
        for i in range(13): u = -1 + 2 * i / 12; F.append(e + n * .50 * u - v * .045 + v / np.hypot(*v) * (.02 * (i % 2)))
        for i in range(13): u = 1 - 2 * i / 12; F.append(e + n * .50 * u + v / np.hypot(*v) * (.12 + .05 * (i % 2)))
        Q = B(spl(F, True, n=3)); R.fill(Q, 'tee'); R.fill(B([e + n * sgn * .05 - v * .2, e + n * sgn * .05 + v * .2, e + n * sgn + v * .2, e + n * sgn - v * .2]), 'tee_sh', clip=Q); R.edge(Q, lw * .8, seed=22)

    def frame(self, f):
        dr = f * 1.0; img = cam2d(self.bg, 960 - 8 + f * .2, 540 + 18 - f * .5, 1.05, 0)
        # the surface: a breathing disc
        x, y = grid(); w = 9 * np.sin(x / 47 + f * .21) + 6 * np.sin(x / 23 - f * .33) + 5 * np.sin(y / 11 + f * .18); d = np.sqrt(((x - DISC[0] + w) / 330) ** 2 + ((y - DISC[1] + f * .6 + w * .4) / 92) ** 2); core = smooth(1.06, .86, d) ; ca = cv2.warpAffine(caustic(f * 1.5), np.float32([[.62, 0, DISC[0] - 560], [0, .26, DISC[1] - f * .6 - 140]]), (W, H), borderMode=cv2.BORDER_REFLECT_101); rip = .46 + .75 * ca; img = screen(img, (core * rip)[..., None] * rgb('#eafff6') * .95); img = screen(img, np.exp(-(d / 2.3) ** 1.4)[..., None] * rgb('#6fd8e6') * .42)
        for j, (a, l, wd, am) in enumerate(((-.52, 1250, 120, .20), (-.22, 1300, 90, .26), (.02, 1350, 150, .24), (.26, 1250, 80, .22), (.46, 1200, 110, .16))):
            a += .03 * math.sin(f * .07 + j * 2); p0 = np.array(DISC) + [a * 300, 0]; dv = np.array([math.sin(a), math.cos(a)]); n = np.array([dv[1], -dv[0]]); q = [p0 - n * 34, p0 + n * 34, p0 + dv * l + n * wd * 2, p0 + dv * l - n * wd * 2]; img = screen(img, shaft(H, W, np.array(q), 16, seed=j)[..., None] * rgb('#7fdcee') * am * (.8 + .2 * math.sin(f * .19 + j * 1.7)))
        img = self.seaking(img, f)
        r = np.random.default_rng(12); c = Cel(W, H, ss=1)                                 # marine snow, far
        for j in range(150): z = r.random(); px = (r.uniform(0, W) + (6 + 30 * z) * math.sin(f * .03 + j)) % W; py = (r.uniform(0, H) - f * (.5 + 2.2 * z)) % H; c.fill(L._circ(px, py, .8 + 1.8 * z * z, 8), '#a8e6ee', .25 + .4 * z)
        img = over(img, c.rgba())
        fig = self.figure(f)
        # the hat: worn until f14, then it goes up as he goes down
        h0 = HT.seat(self.Hd, w=3.0); K = lambda k: float(lerp_keys(f, k)); k = max(0, f - 4) / 55.
        hx = h0['cx'] if f < 4 else K([(4, h0['cx']), (11, h0['cx'] + 26), (20, 640), (34, 800), (48, 985), (59, 1105)]) + 10 * math.sin(f * .17) * min(1, k * 4); hy = h0['cy'] if f < 4 else K([(4, h0['cy']), (11, h0['cy'] - 250), (20, 318), (34, 262), (48, 214), (59, 176)])
        ro = self.roll * (1 - smooth(4, 19, f)) + (.34 * math.sin(f * .115 + .4) - .10) * smooth(8, 22, f); pi = .12 + .24 * smooth(4, 22, f) + .16 * math.sin(f * .14) * smooth(8, 24, f); ya = .60 + 1.2 * k
        wd = h0['w'] * (1 - .20 * k); kb = dict(light=HATL, wear=.65, f=f, wind=1.6, bounce=-.45 * math.exp(-((f - 9) / 5.) ** 2) + .16 * math.sin(f * .33), rim=1.1)
        worn = f < 4; lay_h = HT.hat(hx, hy, wd, ya, pi, ro, part='back' if worn else 'all', **kb)
        img = over(img, lay_h); img = over(img, fig)                       # once it is off it rises on the far side of him, between his open hands and the light
        if worn: img = over(img, HT.hat(hx, hy, wd, ya, pi, ro, part='front', **kb))
        # breath
        c = Cel()
        for (a, t0, rad, ph, sp) in self.bub:
            tt = t0 * .9 - 4; k = f - tt
            src = self.mouth if a < .55 else np.array([hx, hy + 20])
            if k < 0 or (a >= .55 and tt < 8) or (a < .55 and tt > 15): continue
            px = src[0] * 0 + (src[0] if k < 2 else src[0]) + 26 * a * k ** .7 + 10 * math.sin(k * .35 + ph); py = src[1] - sp * k ** 1.06; R_ = rad * (.55 + .02 * k)
            if py < 60: continue
            sq = 1 - .22 * math.sin(k * .7 + ph); P = L._circ(px, py, R_, 24, sq); c.fill(P, '#bff4f6', .22); c.line(P, max(1.2, R_ * .16), '#d8fffc', closed=True, alpha=.85, press=.6); c.fill(L._circ(px - R_ * .35, py - R_ * .35 * sq, R_ * .20, 12), '#ffffff', .95); c.line(spl([(px + R_ * .2, py + R_ * .55), (px + R_ * .5, py + R_ * .35), (px + R_ * .6, py)], n=5), max(1, R_ * .12), '#ffffff', alpha=.7)
        # the one that leaves his lips
        kb_ = f - 27
        if kb_ >= 0:
            gr = min(1., kb_ / 9.); up = max(0, kb_ - 9); px, py = self.mouth + [4 + 9 * math.sin(up * .42) * min(1, up / 4), -8 - 10 * gr - 7.2 * up ** 1.05]; R_ = 4 + 15 * gr + .16 * up; sq = 1.0 + (.30 * math.exp(-up / 2.5) * math.cos(up * 1.5) if up > 0 else .12 * gr); P = L._circ(px, py, R_, 36, sq)
            c.fill(P, '#bff4f6', .20); c.line(P, 3.2, '#dcfffb', closed=True, press=.6, alpha=.95); c.fill(L._circ(px - R_ * .36, py - R_ * .40 * sq, R_ * .20, 14), '#ffffff'); c.line(spl([(px + R_ * .15, py + R_ * .62 * sq), (px + R_ * .52, py + R_ * .40 * sq), (px + R_ * .64, py)], n=6), 2.6, '#ffffff', alpha=.85)
        img = over(img, c.layer())
        # near plane: defocused bubbles and snow
        r = np.random.default_rng(15); c = Cel(W, H, ss=1)
        for j in range(14): px = r.uniform(0, W) + 20 * math.sin(f * .05 + j); py = (r.uniform(0, H + 300) - f * (7 + 6 * r.random())) % (H + 300) - 100; R_ = r.uniform(14, 46); c.line(L._circ(px, py, R_, 30), R_ * .22, '#8fe0ee', closed=True, alpha=.30); c.fill(L._circ(px, py, R_ * .9, 30), '#5fb8d8', .10)
        img = over(img, cv2.GaussianBlur(c.rgba(), (0, 0), 7))
        img = img * float(lerp_keys(f, [(0, 0), (2, .05), (10, .84), (18, 1.), (59, 1.)]))
        return img.astype(f32)
