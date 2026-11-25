"""Chibi crew: thick clean line, big heads, flat + one shadow.  Head unit = head radius.
   d = Draw(cel, cx, cy, s, roll=0, turn=0, sq=0)   d.head('luffy', 'grin', look=(0,0))   d.torso('luffy')   hand(cel, p, ang, size, kind, skin)   arm(cel, P, w0, w1, ...)
   head(cx, cy, s, name, expr, ...) -> layer"""
import numpy as np, math
from op import W, H, kit
from op.ink import Cel, spl, resample
from op.kit import s1, svg
from op.paint import rgb, mix

LN = '#17131a'
SK = dict(luffy=('#f3c096', '#cd8a66'), zoro=('#e0a274', '#a8694e'), sanji=('#f7cfa8', '#cf9a7c'), nami=('#f9d2ae', '#d79a80'), usopp=('#c98b5c', '#8f5a42'), chopper=('#b97b48', '#7d4a3a'), robin=('#f0c19a', '#bf8674'), franky=('#eab584', '#b57a58'), brook=('#f4ecd8', '#b4aabb'), jinbe=('#7fb3cd', '#45749c'))
FACE = 'M-78 -10 C-84 30 -66 66 -34 82 C-14 92 14 92 34 82 C66 66 84 30 78 -10 C74 -84 -74 -84 -78 -10 Z'
# expression -> eyes, mouth, brow tilt (inner end: + = down/angry), extras
EX = dict(calm=('open', 'flat', 0), smile=('open', 'smile', -.2), grin=('wide', 'grin', -.3), laugh=('happy', 'laugh', -.4), shock=('white', 'gape', -1.), gasp=('wide', 'o', -.9), chew=('open', 'chew', -.2), stuffed=('happy', 'chew', -.5), mad=('angry', 'shark', 1.4), cross=('angry', 'frown', 1.1),
          heart=('heart', 'laugh', -.5), shut=('shut', 'flat', 0), sip=('shut', 'tiny', 0), drink=('shut', 'tiny', .2), sly=('half', 'smirk', .5), cry=('cry', 'wave', -1.), dead=('x', 'wave', -.3), blank=('dot', 'flat', 0), stare=('wide', 'flat', -.1), holy=('shut', 'smile', -.2), fire=('white', 'scream', -1.2), whistle=('side', 'tiny', -.3), peek=('wide', 'grin', -1.1), dull=('half', 'flat', .3), sweat=('dot', 'wave', -.6), burp=('half', 'tiny', -.4), bliss=('happy', 'smile', -.5))


class Draw:
    def __init__(s, cel, cx, cy, sc, roll=0., turn=0., sq=0., nod=0., lw=None):
        s.c, s.cx, s.cy, s.s, s.roll, s.turn, s.nod = cel, cx, cy, sc, roll, turn, nod; s.sx, s.sy = 1 - sq * .7, 1 + sq; s.lw = lw or max(2.6, sc * .062); s.cr, s.sr = math.cos(roll), math.sin(roll); s.k = 0

    def T(s, P):
        P = np.asarray(P, float); x, y = P[..., 0] * s.sx * s.s, (P[..., 1] + (s.sy - 1) * -.0) * s.sy * s.s; return np.stack([s.cx + x * s.cr - y * s.sr, s.cy + x * s.sr + y * s.cr], -1)

    def q(s, d): return (s1(d, 10) / 100.) if isinstance(d, str) else np.asarray(d, float)

    def F(s, P):
        """feature space: the face turns"""
        P = np.asarray(P, float); x = np.clip(P[..., 0], -1.2, 1.2); return np.stack([P[..., 0] * (1 - .10 * abs(s.turn)) + s.turn * .30 * np.sqrt(np.maximum(0, 1 - x * x * .8)), P[..., 1] + s.nod * .22 * np.sqrt(np.maximum(0, 1 - x * x * .5))], -1)

    def sh(s, d, fill, w=1., feat=False, seed=0, line=LN):
        P = s.q(d); P = s.F(P) if feat else P; Q = s.T(P); s.c.fill(Q, fill)
        if w > 0: s.k += 1; kit.outline(s.c, Q, s.lw * w, line, seed + s.k * 13, press=.28, tremor=.15)
        return Q

    def shade(s, d, col, within, feat=False): P = s.q(d); P = s.F(P) if feat else P; s.c.shade(s.T(P), col, within)

    def ln(s, d, w=.7, feat=False, col=LN, taper=(.18, .25), al=1.): P = s.q(d); P = s.F(P) if feat else P; s.k += 1; s.c.line(s.T(P if len(P) > 4 else spl(P, n=8)), s.lw * w, col, taper=taper, press=.3, seed=s.k * 7, tremor=.15, alpha=al)

    def el(s, x, y, rx, ry, col, w=0., feat=True, rot=0., n=30):
        t = np.linspace(0, 2 * math.pi, n, endpoint=False); px, py = np.cos(t) * rx, np.sin(t) * ry; c, sn = math.cos(rot), math.sin(rot); P = np.c_[x + px * c - py * sn, y + px * sn + py * c]
        if feat: o = s.F(np.array([[x, y]]))[0] - [x, y]; P = P * [1, 1] + o
        Q = s.T(P); s.c.fill(Q, col)
        if w > 0: s.k += 1; s.c.line(np.vstack([Q, Q[:1]]), s.lw * w, LN, taper=(0, 0), press=.2, seed=s.k, tremor=.1)
        return Q

    # ------------------------------------------------------------------ features
    def eye(s, x, y, rx, ry, kind, look=(0, 0), pup=.055, lash=0, iris=None, slant=0.):
        if kind == 'white': kind = 'wide'; pup = 0
        if kind in ('open', 'wide', 'side', 'half', 'angry'):
            k = 1.16 if kind == 'wide' else 1.; Q = s.el(x, y, rx * k, ry * k, '#fffaf2', .62, rot=slant); lx, ly = look if kind != 'side' else (.6, -.2)
            if kind != 'angry' and pup > 0:
                if iris: s.el(x + lx * rx * .45, y + ly * ry * .4 + ry * .06, rx * .52, ry * .62, iris)
                s.el(x + lx * rx * .5, y + ly * ry * .45 + ry * .05, pup, pup * 1.1, LN)
                if iris: s.el(x + lx * rx * .5 - pup * .5, y + ly * ry * .45 - ry * .16, pup * .45, pup * .5, '#ffffff')
            s.ln([(x - rx * 1.06, y - ry * .30), (x - rx * .5, y - ry * 1.02), (x + rx * .5, y - ry * 1.02), (x + rx * 1.06, y - ry * .30)], .95, True)
            if kind == 'half': s.sh([(x - rx * 1.2, y - ry * 1.2), (x + rx * 1.2, y - ry * 1.2), (x + rx * 1.1, y - ry * .05), (x - rx * 1.1, y - ry * .05)], s.skin, 0, True); s.ln([(x - rx * 1.15, y - ry * .05), (x, y - ry * .12), (x + rx * 1.15, y - ry * .05)], 1.05, True)
            if kind == 'angry': sg = 1 if x < 0 else -1; s.sh([(x - sg * rx * 1.3, y - ry * 1.3), (x + sg * rx * 1.3, y - ry * 1.3), (x + sg * rx * 1.2, y + ry * .05), (x - sg * rx * 1.2, y - ry * .75)], s.skin, 0, True); s.ln([(x - sg * rx * 1.25, y - ry * .78), (x + sg * rx * 1.25, y + ry * .08)], 1.15, True)
            if lash: sg = -1 if x < 0 else 1; [s.ln([(x + sg * rx * .95, y - ry * (.35 - .3 * k_)), (x + sg * rx * 1.35, y - ry * (.62 - .35 * k_))], .6, True) for k_ in range(2)]
        elif kind == 'happy': s.ln([(x - rx, y + ry * .35), (x - rx * .5, y - ry * .5), (x + rx * .5, y - ry * .5), (x + rx, y + ry * .35)], 1.05, True)
        elif kind == 'shut': s.ln([(x - rx, y - ry * .1), (x - rx * .4, y + ry * .34), (x + rx * .4, y + ry * .34), (x + rx, y - ry * .1)], 1.0, True)
        elif kind == 'dot': s.el(x, y, pup * 1.3, pup * 1.4, LN)
        elif kind == 'x': [s.ln([(x - rx * .8, y - a * ry * .6), (x + rx * .8, y + a * ry * .6)], 1.1, True) for a in (-1, 1)]
        elif kind == 'heart': o = s.F(np.array([[x, y]]))[0] - [x, y]; P = np.array([(0, .9), (-.95, -.1), (-.8, -.75), (-.3, -.8), (0, -.4), (.3, -.8), (.8, -.75), (.95, -.1)]) * [rx * 1.25, ry * 1.15] + [x, y] + o; Q = spl(s.T(P), True, corners=(0, 4), n=6); s.c.fill(Q, '#e8384f'); kit.outline(s.c, Q, s.lw * .7, LN, 4)
        elif kind == 'cry':
            s.ln([(x - rx, y - ry * .1), (x - rx * .4, y + ry * .34), (x + rx * .4, y + ry * .34), (x + rx, y - ry * .1)], 1.0, True); s.sh([(x - rx * .5, y + ry * .3), (x + rx * .5, y + ry * .3), (x + rx * .75, y + ry * 2.6), (x - rx * .75, y + ry * 2.6)], '#bfe6fa', .5, True)

    def mouth(s, kind, x=0., y=.46, w=.34, side=1, beard=False):
        MO, TE, TO = '#5a0f18', '#fffaf0', '#e0706c'
        if kind == 'flat': s.ln([(x - w * .6, y), (x, y + .015), (x + w * .6, y)], .85, True)
        elif kind == 'smile': s.ln([(x - w, y - .06), (x - w * .45, y + .07), (x + w * .45, y + .07), (x + w, y - .06)], .95, True)
        elif kind == 'frown': s.ln([(x - w * .8, y + .06), (x - w * .3, y - .04), (x + w * .3, y - .04), (x + w * .8, y + .06)], .95, True)
        elif kind == 'smirk': s.ln([(x - w * .7, y + .03), (x, y + .06), (x + w * .9, y - .08)], .95, True)
        elif kind == 'tiny': s.ln([(x - .06, y), (x + .06, y)], .9, True)
        elif kind == 'wave': s.ln([(x - w, y + .02), (x - w * .5, y - .06), (x, y + .03), (x + w * .5, y - .06), (x + w, y + .02)], .95, True)
        elif kind == 'o': s.el(x, y + .02, .10, .14, MO, .8)
        elif kind == 'grin':
            P = [(x - w * 1.3, y - .12), (x - w * .6, y - .04), (x, y - .02), (x + w * .6, y - .04), (x + w * 1.3, y - .12), (x + w * .95, y + .16), (x + w * .4, y + .27), (x - w * .4, y + .27), (x - w * .95, y + .16)]; s.sh(spl(P, True, corners=(0, 4), n=6), TE, .9, True); s.ln([(x - w * 1.12, y + .05), (x - w * .5, y + .10), (x + w * .5, y + .10), (x + w * 1.12, y + .05)], .5, True); [s.ln([(x + w * u, y - .03 + .0), (x + w * u * .9, y + .20)], .32, True) for u in (-.75, -.4, 0, .4, .75)]
        elif kind == 'laugh':
            P = [(x - w * 1.1, y - .12), (x - w * .5, y - .07), (x + w * .5, y - .07), (x + w * 1.1, y - .12), (x + w * .9, y + .18), (x + w * .35, y + .34), (x - w * .35, y + .34), (x - w * .9, y + .18)]; Q = s.sh(spl(P, True, corners=(0, 3), n=6), MO, .9, True); s.shade([(x - w * 1.2, y - .2), (x + w * 1.2, y - .2), (x + w, y + .02), (x - w, y + .02)], TE, Q, True); s.shade(spl([(x - w * .6, y + .36), (x - w * .4, y + .18), (x + w * .4, y + .18), (x + w * .6, y + .36)], True), TO, Q, True)
        elif kind in ('gape', 'scream'):
            h = .50 if kind == 'gape' else .62; P = [(x - w * .8, y - .10), (x, y - .16), (x + w * .8, y - .10), (x + w * .95, y + h * .55), (x + w * .45, y + h), (x - w * .45, y + h), (x - w * .95, y + h * .55)]; Q = s.sh(spl(P, True, n=6), MO, .95, True); s.shade([(x - w, y - .2), (x + w, y - .2), (x + w, y - .02), (x - w, y - .02)], TE, Q, True); s.shade(spl([(x - w * .6, y + h * 1.1), (x - w * .4, y + h * .55), (x + w * .4, y + h * .55), (x + w * .6, y + h * 1.1)], True), TO, Q, True)
        elif kind == 'shark':
            P = [(x - w * 1.25, y - .06), (x, y - .18), (x + w * 1.25, y - .06), (x + w * 1.05, y + .34), (x, y + .44), (x - w * 1.05, y + .34)]; Q = s.sh(spl(P, True, corners=(0, 2), n=6), MO, 1., True)
            for k in range(6): u = -1 + (k + .5) / 3; s.shade([(x + w * (u - .20), y - .20), (x + w * (u + .20), y - .20), (x + w * u, y + .10)], TE, Q, True); s.shade([(x + w * (u - .20), y + .50), (x + w * (u + .20), y + .50), (x + w * u, y + .16)], TE, Q, True)
        elif kind == 'chew': s.ln([(x - .10, y - .02), (x, y + .03), (x + .10, y - .02)], .9, True); s.ln([(x - .16, y - .07), (x - .10, y - .02), (x - .14, y + .04)], .7, True); s.ln([(x + .16, y - .07), (x + .10, y - .02), (x + .14, y + .04)], .7, True)

    def brow(s, x, y, w, tilt, col=LN, th=1.6): sg = 1 if x < 0 else -1; s.ln([(x - sg * w, y - tilt * .05 + .02), (x, y - .03), (x + sg * w, y + tilt * .075)], th, True, col, (.25, .15))

    def ear(s, sd, col, shd, y=.10): s.sh([(sd * .74, y - .16), (sd * .90, y - .18), (sd * .97, y - .02), (sd * .88, y + .17), (sd * .74, y + .14)], col, .85); s.ln([(sd * .80, y - .06), (sd * .89, y - .05), (sd * .86, y + .08)], .5)

    def base(s, name, d=FACE, fat=0., ears=True):
        s.skin, s.shd = SK[name]; P = s.q(d).copy()
        if fat > 0: k = np.exp(-((P[:, 1] - .36) / .42) ** 2); P[:, 0] *= 1 + fat * .75 * k; P[:, 1] += fat * .05 * k
        if ears: [s.ear(sd, s.skin, s.shd) for sd in (-1, 1) if not (sd * s.turn > .55)]
        Q = s.sh(P, s.skin, 1.); s.shade([(.36 + s.turn * .2, -1.2), (.52 + s.turn * .15, .1), (.22, .72), (-.5, .86), (-.5, 1.3), (2, 1.3), (2, -1.2)], s.shd, Q); s.faceQ = Q; return Q

    # ------------------------------------------------------------------ the crew
    def head(s, name, expr='calm', look=(0, 0), fat=0., blink=False, **k):
        e, m, b = EX[expr]; getattr(s, '_' + name)(e, m, b, look, fat, expr, **k); return s

    def _hat(s, part, rib='#b3202a'):
        ST, SS, DK = '#e6bb5c', '#b07a30', '#8a561f'
        if part == 'back': s.sh('M-158 -40 C-150 -78 -70 -92 0 -92 C70 -92 150 -78 158 -40 C150 -14 70 -20 0 -20 C-70 -20 -150 -14 -158 -40 Z', DK, 1.); return
        Q = s.sh('M-86 -56 C-96 -130 -54 -172 0 -172 C54 -172 96 -130 86 -56 C50 -40 -50 -40 -86 -56 Z', ST, 1.05); s.shade([(.18, -2), (.46, -1.2), (.40, -.3), (1.4, -.3), (1.4, -2)], SS, Q); [s.ln([(-.62 + .10 * k, -1.06 - .12 * k), (-.2, -1.0 - .16 * k), (.1, -1.02 - .15 * k)], .32, col=DK, al=.8) for k in range(4)]
        R = s.sh('M-87 -60 C-90 -78 -90 -88 -90 -96 C-50 -80 50 -80 90 -96 C90 -88 90 -78 87 -60 C50 -44 -50 -44 -87 -60 Z', rib, .8); s.shade([(.2, -1.3), (.46, -.9), (.42, -.3), (1.4, -.3), (1.4, -1.3)], '#73101e', R)
        B = s.sh('M-164 -44 C-150 -74 -104 -70 -87 -60 C-50 -44 50 -44 87 -60 C104 -70 150 -74 164 -44 C150 -14 90 -22 60 -14 C20 -4 -20 -6 -60 -16 C-100 -22 -150 -14 -164 -44 Z', ST, 1.1); s.shade([(-1.7, -.30), (-.9, -.32), (0, -.18), (.9, -.30), (1.7, -.34), (1.7, .2), (-1.7, .2)], SS, B); [s.ln([(x, -.42 + abs(x) * .02), (x + .12, -.38 + abs(x) * .02)], .34, col=DK, al=.8) for x in (-1.3, -1.05, -.8, -.5, -.2, .12, .42, .72, 1.0, 1.25)]

    def _luffy(s, e, m, b, look, fat, expr, hat=True, red=0.):
        if hat: s._hat('back')
        s.sh('M-84 10 C-104 -60 -60 -104 0 -104 C60 -104 104 -60 84 10 L92 22 L74 34 L-74 34 L-92 22 Z', LN, .6); Q = s.base('luffy', fat=fat)
        if red: s.c.fill(Q, '#e03a2a', red)
        if hat: s.shade([(-1.1, -.9), (1.1, -.9), (.92, -.20), (.45, -.08), (-.1, -.05), (-.6, -.12), (-.95, -.22)], s.shd, Q)
        if e == 'white' and fat: e = 'wide'
        for sd in (-1, 1): s.eye(sd * .35, .06 - .06 * fat, .205, .235, e, look, .055); s.brow(sd * .36, -.26, .15, b, th=1.35)
        s.ln([(.22, .37), (.35, .395), (.48, .365)], .6, True); [s.ln([(x, .33), (x + .01, .43)], .5, True) for x in (.30, .40)]
        s.ln([(.03, .24), (.0, .29), (-.05, .28)], .7, True); s.mouth(m, 0, .50, .40 if fat < .5 else .2)
        if fat > .25: [s.ln([(sd * (.52 + .22 * fat), .30), (sd * (.62 + .30 * fat), .42), (sd * (.54 + .25 * fat), .56)], .6) for sd in (-1, 1)]
        s.sh('M-80 14 L-90 -30 C-84 -96 84 -96 90 -30 L80 14 L72 -10 L62 12 L48 -26 L34 2 L20 -32 L4 -6 L-12 -34 L-28 0 L-42 -28 L-56 10 L-68 -10 Z', '#241f2b', .8); s.ln([(-.5, -.5), (-.56, -.2)], .4, col='#5b5f86'); s.ln([(.1, -.6), (.1, -.3)], .4, col='#5b5f86')
        if hat: s._hat('front')

    def _zoro(s, e, m, b, look, fat, expr, **k):
        G, D_ = '#62b552', '#35763f'; s.sh('M-80 8 C-98 -60 -84 -100 -40 -108 L40 -108 C84 -100 98 -60 80 8 Z', D_, .6); s.base('zoro'); s.eye(-.34, .06, .20, .15, 'half' if e == 'open' else e, look, .05); s.ln([(.16, .08), (.34, .13), (.54, .07)], 1.1, True); s.ln([(.36, -.34), (.34, .04), (.38, .40)], .75, True, '#7a3a34'); s.brow(-.36, -.20, .19, max(b, .6), '#2f6b36', 2.0); s.brow(.36, -.20, .19, max(b, .6), '#2f6b36', 2.0)
        s.ln([(.02, .18), (.05, .31), (-.03, .32)], .75, True); s.mouth(m, 0, .54, .28); Q = s.sh('M-82 6 C-100 -60 -84 -104 -46 -110 L-38 -124 L-24 -110 L-8 -128 L6 -111 L24 -126 L36 -109 C86 -100 100 -60 82 6 L76 -16 L66 -44 L52 -34 L40 -52 L24 -40 L8 -56 L-8 -42 L-24 -56 L-38 -40 L-52 -50 L-66 -38 L-76 -14 Z', G, 1.); s.shade([(.30, -1.4), (.52, -.5), (.6, .2), (1.4, .2), (1.4, -1.4)], D_, Q); [s.ln([(x, -.98), (x * .95, -.72)], .4, col=D_) for x in (-.6, -.4, -.2, 0, .2)]
        [s.sh([(.83 + .06 * k_, .26), (.87 + .06 * k_, .26), (.865 + .06 * k_, .48), (.835 + .06 * k_, .48)], '#f2c23c', .45) for k_ in range(3)]

    def _sanji(s, e, m, b, look, fat, expr, cig=True, **k):
        Y, D_ = '#f4d165', '#c4923c'; s.sh('M-84 30 C-108 -60 -80 -114 0 -114 C80 -114 108 -60 88 34 L70 60 L-70 50 Z', D_, .8); s.base('sanji'); s.eye(-.34, .06, .19, .18, e, look, .055, iris='#3f6fa8' if e not in ('heart',) else None); s.brow(-.36, -.20, .17, b, '#8a6020', 1.3); t = np.linspace(0, 4.9, 20); r = .075 * (1 - t / 7.5); o = np.array([-.60, -.18]); s.ln(np.c_[o[0] + np.cos(t + .4) * r, o[1] + np.sin(t + .4) * r], .9, True, '#8a6020', (.05, .3))
        s.ln([(.0, .20), (.03, .30), (-.03, .31)], .7, True); s.mouth(m, 0, .52, .28); s.ln([(-.06, .80), (.0, .86), (.06, .80)], .9, True, '#a87a30'); s.ln([(-.03, .74), (.03, .74)], .5, True, '#a87a30')
        if cig and m not in ('laugh', 'gape', 'shark'): s.sh([(.12, .50), (.38, .58), (.375, .625), (.12, .545)], '#fffaf0', .5, True); s.el(.38, .60, .03, .03, '#ff7a2a')
        Q = s.sh('M-82 22 C-106 -60 -80 -116 0 -118 C78 -118 108 -60 92 30 C90 62 76 80 58 84 C64 50 62 20 34 -22 C24 -6 6 -2 -6 -4 C-2 -20 -4 -34 -14 -44 C-26 -24 -50 -14 -68 -16 C-74 -4 -78 8 -82 22 Z', Y, 1.); s.shade('M40 -120 C80 -80 78 0 58 84 L130 90 L130 -120 Z', D_, Q); [s.ln(P, .4, col=D_) for P in ([(-.1, -1.08), (.28, -.8), (.52, -.1)], [(-.25, -1.0), (-.5, -.7), (-.62, -.3)], [(.2, -1.1), (.62, -.7), (.74, .0)])]

    def _nami(s, e, m, b, look, fat, expr, **k):
        O, D_ = '#f2873a', '#b2492c'; Q = s.sh('M-80 -10 C-110 -70 -70 -120 0 -118 C70 -120 112 -70 92 10 C110 70 98 130 120 168 C90 160 80 150 70 130 L60 60 L-60 60 C-70 110 -66 140 -96 170 C-110 120 -96 60 -90 20 Z', O, 1.); s.shade([(-.7, .3), (-.6, 1.8), (-1.3, 1.8), (-1.0, .3)], D_, Q); s.shade([(.60, .3), (.8, 1.0), (.9, 1.8), (.5, 1.8), (.5, .3)], D_, Q); s.base('nami')
        for sd in (-1, 1): s.eye(sd * .35, .10, .17, .20, e, look, .07, 1, '#8a4a2a'); s.brow(sd * .36, -.18, .13, b, '#a8482a', 1.0)
        s.ln([(.0, .30), (.03, .35)], .6, True); s.mouth(m, 0, .54, .24); Q = s.sh('M-84 22 C-112 -60 -76 -120 0 -120 C76 -120 112 -60 86 22 C80 -6 72 -24 58 -40 C40 -10 20 0 4 0 C18 -20 18 -40 10 -58 C-6 -20 -40 -6 -66 -14 C-74 -2 -80 10 -84 22 Z', O, 1.); s.shade([(.30, -1.4), (.62, -.6), (.8, .3), (1.4, .3), (1.4, -1.4)], D_, Q); [s.ln(P, .4, col=D_) for P in ([(-.05, -1.1), (.06, -.8), (.08, -.6)], [(-.3, -1.0), (-.5, -.6), (-.62, -.25)], [(.3, -1.0), (.5, -.7), (.58, -.45)])]

    def _usopp(s, e, m, b, look, fat, expr, **k):
        for x, y, r in ((-.78, -.15, .34), (.78, -.15, .34), (-.86, .22, .30), (.86, .22, .30), (-.62, -.62, .36), (.62, -.62, .36), (0, -.82, .42), (-.80, .55, .24), (.80, .55, .24), (-.3, -.85, .36), (.3, -.85, .36)): s.el(x, y, r, r, LN, .5, False)
        for x, y, r in ((-.78, -.15, .31), (.78, -.15, .31), (-.86, .22, .27), (.86, .22, .27), (-.62, -.62, .33), (.62, -.62, .33), (0, -.82, .39), (-.3, -.85, .33), (.3, -.85, .33)): s.el(x, y, r, r, '#2a2230', 0, False)
        for x, y in ((-.86, .1), (.9, .3), (-.7, -.5), (.72, -.55), (-.84, .5)): t = np.linspace(0, 4.6, 16); s.ln(np.c_[x + np.cos(t) * .13 * (1 - t / 8), y + np.sin(t) * .13 * (1 - t / 8)], .4, col='#5b4f5f')
        s.base('usopp', ears=False)
        for sd in (-1, 1): s.eye(sd * .34, .02, .18, .20, e, look, .055); s.brow(sd * .36, -.20, .13, b, th=1.5)
        sd = 1 if s.turn >= 0 else -1; s.mouth(m, 0, .56, .30)
        if m in ('flat', 'smile', 'smirk'): s.ln([(-.20, .66), (0, .70), (.20, .66)], .6, True)
        n = [(-.05 * sd, .18), (.25 * sd, .12), (.70 * sd, .14), (1.06 * sd, .20), (1.10 * sd, .27), (1.02 * sd, .32), (.60 * sd, .34), (.2 * sd, .38), (-.03 * sd, .36)]; Q = s.sh(spl(n, True, n=6), s.skin, .95, True); s.shade([(-.1 * sd, .30), (.6 * sd, .27), (1.2 * sd, .27), (1.2 * sd, .5), (-.1 * sd, .5)], s.shd, Q, True)
        Q = s.sh('M-80 -24 C-84 -92 84 -92 80 -24 C40 -44 -40 -44 -80 -24 Z', '#e9d9a8', .9); s.shade([(.3, -1.2), (.5, -.3), (1.2, -.3), (1.2, -1.2)], '#b79c74', Q)
        s.ln([(-.82, -.42), (0, -.64), (.82, -.42)], 2.0, False, '#4a3a4a', (0, 0)); [(s.el(x, -.60, .20, .17, '#7a2f3a', .9, True), s.el(x, -.60, .13, .11, '#a9d4e4', .5, True)) for x in (-.27, .27)]

    def _chopper(s, e, m, b, look, fat, expr, **k):
        B_, D_ = SK['chopper']; s.skin, s.shd = B_, D_
        for sd in (-1, 1):                                                                # antlers, ears
            Q = s.sh(spl([(sd * .66, -.62), (sd * .92, -.86), (sd * .98, -1.32), (sd * 1.10, -1.30), (sd * 1.10, -.98), (sd * 1.28, -1.16), (sd * 1.40, -1.10), (sd * 1.24, -.82), (sd * 1.00, -.62), (sd * .74, -.46)], True, corners=(2, 3, 5, 6), n=5), '#8d5c3a', .95); s.shade([(sd * .9, -.5), (sd * 1.5, -.9), (sd * 1.5, -.4)], '#5f3a30', Q)
            s.sh(spl([(sd * .74, -.12), (sd * 1.20, -.22), (sd * 1.30, -.02), (sd * 1.10, .12), (sd * .76, .12)], True, n=5), B_, .9); s.sh(spl([(sd * .86, -.06), (sd * 1.12, -.12), (sd * 1.16, 0), (sd * .88, .06)], True, n=5), '#e8a9a0', 0)
        Q0 = Q = s.sh('M-86 -10 C-94 40 -70 78 -34 84 C-14 88 14 88 34 84 C70 78 94 40 86 -10 C80 -84 -80 -84 -86 -10 Z', B_, 1.); s.shade([(.46, -1), (.60, .2), (.25, .74), (-.4, .86), (-.4, 1.2), (2, 1.2), (2, -1)], D_, Q); s.sh(spl([(-.44, .34), (0, .20), (.44, .34), (.46, .62), (.22, .80), (-.22, .80), (-.46, .62)], True, n=6), '#e7bd8c', .0, True)
        s.shade([(-1.1, -.9), (1.1, -.9), (.95, -.24), (0, -.14), (-.95, -.24)], D_, Q0)
        for sd in (-1, 1): s.eye(sd * .37, .06, .15, .22, e, look, .085); s.brow(sd * .38, -.26, .10, b, th=1.1)
        s.el(0, .34, .13, .10, '#2f6fc0', .8); s.el(-.04, .31, .04, .025, '#bfe0ff'); s.mouth(m, 0, .56, .24); H_ = s.sh('M-92 -30 C-100 -60 -70 -70 0 -72 C70 -70 100 -60 92 -30 C60 -22 -60 -22 -92 -30 Z', '#d84a68', 1.); C_ = s.sh('M-78 -56 C-92 -150 -50 -164 0 -164 C50 -164 92 -150 78 -56 C40 -66 -40 -66 -78 -56 Z', '#ee5f7c', 1.05); s.shade([(.28, -1.8), (.46, -.4), (1.2, -.4), (1.2, -1.8)], '#b23a5e', C_)
        for a in (-1, 1): c_, sn = math.cos(a * .78), math.sin(a * .78); P = np.array([[-.36, -.075], [.36, -.075], [.36, .075], [-.36, .075]]) @ np.array([[c_, sn], [-sn, c_]]) + [0, -1.06]; s.sh(P, '#ffffff', .0)
        P = [(-.31, -1.26), (-.20, -1.36), (0, -1.16), (.20, -1.36), (.31, -1.26), (.11, -1.06), (.31, -.86), (.20, -.76), (0, -.96), (-.20, -.76), (-.31, -.86), (-.11, -1.06)]; s.sh(P, '#ffffff', .7)

    def _robin(s, e, m, b, look, fat, expr, **k):
        K, D_ = '#2a2440', '#15111f'; Q = s.sh('M-84 0 C-106 -70 -66 -116 0 -116 C66 -116 106 -70 86 0 C98 60 92 120 100 160 L52 150 L46 60 L-46 60 L-52 150 L-100 160 C-92 120 -98 60 -84 0 Z', K, 1.); s.base('robin', ears=False)
        for sd in (-1, 1): s.eye(sd * .34, .08, .17, .15, e, look, .06, 1, '#3f7fd0', slant=sd * -.08); s.brow(sd * .36, -.16, .14, b * .6, th=.95)
        s.ln([(.0, .16), (.04, .32), (-.02, .33)], .6, True); s.mouth(m, 0, .54, .20); Q = s.sh('M-84 60 C-104 -60 -70 -118 0 -118 C70 -118 104 -60 84 60 C76 10 70 -20 52 -44 C30 -62 10 -68 0 -84 C-10 -68 -30 -62 -52 -44 C-70 -20 -76 10 -84 60 Z', K, 1.); [s.ln(P, .5, col='#6a63a0') for P in ([(-.2, -1.0), (-.5, -.75), (-.72, -.3)], [(.2, -1.0), (.5, -.75), (.72, -.3)])]

    def _franky(s, e, m, b, look, fat, expr, droop=0., **k):
        C_, D_ = '#58c4e6', '#2a7aae'; s.base('franky', 'M-76 -20 C-82 30 -74 62 -50 76 C-40 92 -14 98 0 98 C14 98 40 92 50 76 C74 62 82 30 76 -20 C72 -84 -72 -84 -76 -20 Z')
        for sd in (-1, 1): s.eye(sd * .34, .02, .16, .12, e, look, .05); s.brow(sd * .34, -.18, .16, max(b, .5), C_ if 0 else '#2a7aae', 1.6)
        Q = s.sh(spl([(-.13, .04), (.13, .04), (.20, .34), (0, .42), (-.20, .34)], True, n=5), '#b9c2cf', .9, True); s.shade([(.0, 0), (.4, 0), (.4, .5), (.05, .5)], '#767f96', Q, True); s.mouth(m, 0, .60, .30); [s.ln([(x, .82), (x + .02, .90)], .5, True, '#2a7aae') for x in (-.22, -.1, .04, .16, .28)]
        s.ln([(-.74, -.36), (0, -.52), (.74, -.36)], 1.6, False, '#2b2430', (0, 0)); [s.sh([(sd * .10, -.56), (sd * .60, -.60), (sd * .56, -.36), (sd * .16, -.34)], '#23202e', .8, True) for sd in (-1, 1)]
        h = 1 - droop; Q = s.sh(spl([(-.80, .10), (-.94, -.50), (-.70, -.92), (-.50, -.98 - .10 * h), (-.46, -1.2 - .46 * h), (0, -1.10 - .86 * h), (.52 + .3 * droop, -1.0 - .66 * h), (.62 + .5 * droop, -.80 - .24 * h + .5 * droop), (.70, -.90), (.94, -.50), (.80, .10), (.70, -.30), (.40, -.62), (-.40, -.62), (-.70, -.30)], True, corners=(0, 10), n=6), C_, 1.05); s.shade([(.16, -2.2), (.42, -1.0), (.6, .2), (1.6, .2), (1.6, -2.2)], D_, Q); [s.ln([(-.3 + .2 * k_, -1.0), (-.3 + .24 * k_, -1.2 - .3 * h)], .45, col=D_) for k_ in range(3)]

    def _brook(s, e, m, b, look, fat, expr, jaw=0., hat=0., **k):
        a = np.linspace(0, 2 * math.pi, 25)[:-1]; r = np.random.default_rng(3); P = np.c_[np.cos(a) * 1.36 * (1 + .03 * r.normal(0, 1, 24)), -.42 + np.sin(a) * 1.16 * (1 + .03 * r.normal(0, 1, 24))]; s.sh(spl(P, True, n=4), '#2a2430', 1.1); [s.ln(np.c_[x + np.cos(np.linspace(0, 4.4, 14)) * .10, y + np.sin(np.linspace(0, 4.4, 14)) * .10], .45, col='#5e5670') for x, y in ((-.9, -.6), (.9, -.7), (-.5, -1.2), (.5, -1.2), (-1.05, -.1), (1.05, -.1), (0, -1.32))]
        s.skin, s.shd = SK['brook']; j = jaw * .5; Q = s.sh(spl([(-.46, .52 + j), (-.40, .86 + j), (0, .98 + j), (.40, .86 + j), (.46, .52 + j)], True, n=6), s.skin, 1.) if jaw > 0 else None
        if jaw > 0: s.sh(spl([(-.42, .40), (.42, .40), (.40, .62 + j), (-.40, .62 + j)], True, n=4), '#1a1020', .7); [s.sh([(x - .07, .60 + j), (x + .07, .60 + j), (x + .06, .72 + j), (x - .06, .72 + j)], '#fffaf0', .5) for x in (-.24, -.08, .08, .24)]
        Q = s.sh('M-70 -20 C-74 10 -60 30 -50 36 L-46 62 C-40 80 40 80 46 62 L50 36 C60 30 74 10 70 -20 C66 -90 -66 -90 -70 -20 Z' if jaw <= 0 else 'M-70 -20 C-74 10 -60 30 -50 36 L-46 52 C-40 60 40 60 46 52 L50 36 C60 30 74 10 70 -20 C66 -90 -66 -90 -70 -20 Z', s.skin, 1.05); s.shade([(.30, -1), (.46, .1), (.3, .9), (1.4, .9), (1.4, -1)], s.shd, Q)
        for sd in (-1, 1): s.el(sd * .30, .02, .22 + .05 * (jaw > 0), .23 + .07 * (jaw > 0), '#1a1020', .7)
        if e in ('happy', 'shut'): [s.ln([(sd * .30 - .16, .06), (sd * .30, -.06 if e == 'happy' else .14), (sd * .30 + .16, .06)], .7, True, '#f4ecd8') for sd in (-1, 1)]
        s.sh([(-.07, .30), (0, .20), (.07, .30)], '#1a1020', .4, True)
        if jaw <= 0: s.ln([(-.40, .52), (0, .57), (.40, .52)], .6, True); [s.ln([(x, .46), (x, .64)], .45, True) for x in (-.30, -.18, -.06, .06, .18, .30)]
        yh = -1.42 - hat * .6; Q = s.sh([(-.34, yh), (-.30, yh - .66), (.30, yh - .66), (.34, yh)], '#3a3450', .95); s.shade([(.1, yh + .1), (.1, yh - .8), (.5, yh - .8), (.5, yh + .1)], '#201c30', Q); s.sh(spl([(-.62, yh + .02), (0, yh - .06), (.62, yh + .02), (0, yh + .12)], True, n=6), '#3a3450', .95); s.ln([(-.34, yh - .1), (0, yh - .06), (.34, yh - .1)], 1.6, False, '#e8b84a', (0, 0))

    def _jinbe(s, e, m, b, look, fat, expr, **k):
        s.base('jinbe', 'M-86 -16 C-92 34 -80 74 -44 88 C-20 96 20 96 44 88 C80 74 92 34 86 -16 C80 -84 -80 -84 -86 -16 Z', ears=False)
        for sd in (-1, 1):
            s.eye(sd * .34, .04, .15, .11, e, look, .055); s.brow(sd * .36, -.14, .19, max(b, .7), th=2.6); s.sh(spl([(sd * .70, -.30), (sd * 1.04, -.20), (sd * 1.16, .16), (sd * 1.02, .42), (sd * .86, .34), (sd * .96, .16), (sd * .84, .06), (sd * .72, .10)], True, n=5), LN, .5)
            s.sh([(sd * .30, .70), (sd * .42, .66), (sd * .42, .36), (sd * .34, .34)], '#f2d27a', .7, True)
        s.ln([(.52, -.20), (.60, .14), (.56, .34)], .7, True, '#2d4f7c'); s.ln([(-.08, .22), (0, .27), (.08, .22)], .8, True); s.mouth(m if m not in ('smile',) else 'smile', 0, .58, .36); s.sh(spl([(-.14, .86), (.14, .86), (.07, 1.20), (0, 1.32), (-.07, 1.20)], True, n=5), LN, .5, True)
        s.sh(spl([(-.86, -.10), (-.90, -.60), (-.5, -1.02), (0, -1.08), (.5, -1.02), (.90, -.60), (.86, -.10), (.60, -.50), (0, -.66), (-.60, -.50)], True, corners=(0, 6), n=6), LN, .7); s.sh(spl([(-.20, -1.02), (-.26, -1.40), (0, -1.60), (.26, -1.40), (.20, -1.02)], True, n=5), LN, .6); s.ln([(-.20, -1.06), (0, -1.02), (.20, -1.06)], 1.8, False, '#e0b040', (0, 0)); [s.ln([(-.1 + .1 * k_, -1.14), (-.1 + .1 * k_, -1.4)], .4, col='#5f6f9c') for k_ in range(3)]

    # ------------------------------------------------------------------ bodies (shoulders at y 1.1, run down behind the table)
    def torso(s, name, w=1., fat=0.):
        sk, sh_ = SK[name]
        def body(col, shc, wd=1.0, top=1.02): Q = s.sh(spl([(-.22, top - .12), (-.62 * wd, top + .10), (-.86 * wd, top + .55), (-.90 * wd, 3.0), (.90 * wd, 3.0), (.86 * wd, top + .55), (.62 * wd, top + .10), (.22, top - .12)], True, corners=(0, 3, 4, 7), n=6), col, 1.); s.shade([(.28, .6), (.46, 1.6), (.4, 3.2), (1.8, 3.2), (1.8, .6)], shc, Q); return Q
        neck = lambda: s.sh([(-.24, .70), (.24, .70), (.28, 1.18), (-.28, 1.18)], sh_, .8)
        if name == 'luffy':
            if fat > 0: t = np.linspace(0, 2 * math.pi, 40); Q = s.sh(np.c_[np.cos(t) * (.95 + 1.5 * fat), 1.75 + np.sin(t) * (.85 + 1.25 * fat)], sk, 1.1); s.shade(np.c_[.5 + np.cos(t) * (.95 + 1.5 * fat), 2.1 + np.sin(t) * (.85 + 1.25 * fat)], sh_, Q); [s.sh(spl([(sd * (.55 + 1.0 * fat), 1.1 - .3 * fat), (sd * (1.0 + 1.5 * fat), 1.5), (sd * (.95 + 1.45 * fat), 3.0), (sd * (.55 + 1.15 * fat), 3.0), (sd * (.45 + .95 * fat), 1.9)], True, n=6), '#c3272f', 1.) for sd in (-1, 1)]; [s.ln([(-.22, 1.35), (.22, 1.75)], .9, col='#c07a5a'), s.ln([(.22, 1.35), (-.22, 1.75)], .9, col='#c07a5a')]; return
            neck(); Q = body('#c3272f', '#7d141f'); s.sh([(-.24, .98), (.24, .98), (.30, 3), (-.30, 3)], sk, .8); s.ln([(-.18, 1.3), (.18, 1.7)], .9, col='#c07a5a'); s.ln([(.18, 1.3), (-.18, 1.7)], .9, col='#c07a5a')
        elif name == 'zoro': neck(); body('#2f5a46', '#173327', 1.1); s.sh([(-.26, 1.0), (.26, 1.0), (.20, 3), (-.20, 3)], sk, .8); s.ln([(-.16, 1.3), (.18, 2.2)], .8, col='#7a3a34')
        elif name == 'sanji': neck(); body('#23212c', '#100f18'); s.sh([(-.22, 1.0), (.22, 1.0), (0, 2.0)], '#f0c648', .7); s.sh([(-.07, 1.06), (.07, 1.06), (.11, 1.7), (0, 1.9), (-.11, 1.7)], '#1a1822', .6); s.ln([(-.24, 1.0), (0, 1.9)], .9); s.ln([(.24, 1.0), (0, 1.9)], .9)
        elif name == 'nami': neck(); Q = body('#f4f2ee', '#b9b6cc', .82); [s.shade([(-2, 1.35 + .34 * k), (2, 1.35 + .34 * k), (2, 1.50 + .34 * k), (-2, 1.50 + .34 * k)], '#3f78c8', Q) for k in range(5)]; s.ln([(-.3, 1.0), (0, 1.12), (.3, 1.0)], .8)
        elif name == 'usopp': neck(); body(sk, sh_, .92); [s.sh([(sd * .22, 1.0), (sd * .44, 1.04), (sd * .46, 3), (sd * .20, 3)], '#7a5a34', .8) for sd in (-1, 1)]; s.sh([(-.5, 1.9), (.5, 1.9), (.6, 3), (-.6, 3)], '#7a5a34', .9)
        elif name == 'chopper': Q = body('#8fd0e0', '#4f8fb0', .72, 1.0); [s.shade([(-2, 1.3 + .36 * k), (2, 1.3 + .36 * k), (2, 1.46 + .36 * k), (-2, 1.46 + .36 * k)], '#f6e6b0', Q) for k in range(4)]
        elif name == 'robin': neck(); body('#7a4fa8', '#482a6e', .82); s.sh([(-.20, 1.0), (.20, 1.0), (.10, 1.5), (-.10, 1.5)], sk, .7); s.ln([(0, 1.5), (0, 3)], .8)
        elif name == 'franky': neck(); body('#e8452f', '#8f1f3a', 1.1); s.sh([(-.26, 1.0), (.26, 1.0), (.34, 3), (-.34, 3)], sk, .9); [(s.el(sd * 1.15, 1.45, .48, .46, '#d8402f', 1., False), s.sh([(sd * 1.15 + .26 * math.cos(-1.57 + k * 1.2566) * (1 if 1 else 0), 1.45 + .26 * math.sin(-1.57 + k * 1.2566)) if j == 0 else (sd * 1.15 + .11 * math.cos(-1.57 + (k + .5) * 1.2566), 1.45 + .11 * math.sin(-1.57 + (k + .5) * 1.2566)) for k in range(5) for j in (0, 1)], '#f4f2ee', .6)) for sd in (-1, 1)]
        elif name == 'brook': s.sh([(-.09, .8), (.09, .8), (.09, 1.2), (-.09, 1.2)], sk, .7); body('#2c2340', '#15102a', .8); [s.sh(spl([(sd * .12, 1.0), (sd * .62, .92), (sd * .74, 1.24), (sd * .5, 1.5), (sd * .16, 1.3)], True, n=5), '#f08a4a', .9) for sd in (-1, 1)]; s.sh([(-.10, 1.2), (.10, 1.2), (.14, 1.8), (-.14, 1.8)], '#4f9ae0', .7)
        elif name == 'jinbe': body('#d9592c', '#7c2440', 1.35); s.sh([(-.30, .95), (.30, .95), (.36, 3), (-.36, 3)], sk, .9); t = np.linspace(0, 6.283, 20); s.c.fill(s.T(np.c_[np.cos(t) * .15, 1.6 + np.sin(t) * .15]), '#c8352e'); [s.ln([(.18 * math.cos(a), 1.6 + .18 * math.sin(a)), (.28 * math.cos(a), 1.6 + .28 * math.sin(a))], .8, col='#c8352e') for a in np.linspace(0, 6.283, 9)[:-1]]


# ----------------------------------------------------------------------------- hands and arms
_FG = dict(open=[(-.62, .62, .20), (-.22, .80, .21), (.18, .80, .21), (.58, .62, .19)], flat=[(-.30, .78, .20), (-.10, .86, .21), (.10, .84, .20), (.30, .74, .18)], point=[(-.05, .95, .21), (.36, .30, .22), (.52, .26, .22), (.66, .22, .20)], fist=[(-.42, .30, .25), (-.14, .34, .25), (.16, .34, .25), (.44, .28, .23)],
           grab=[(-.66, .50, .22), (-.24, .62, .23), (.22, .62, .23), (.66, .50, .21)], cup=[(-.34, .46, .22), (-.10, .52, .22), (.16, .50, .22), (.40, .42, .20)])


def hand(cel, p, ang, r, kind='open', skin=('#f3c096', '#cd8a66'), flip=1, lw=None, hoof=False):
    """chibi hand: a round palm, four real fingers and a thumb, pointing along ang; r = palm radius px"""
    lw = lw or max(2.2, r * .16); c, s_ = math.cos(ang), math.sin(ang); R = lambda P: np.asarray(P, float) * r @ np.array([[c, s_], [-s_, c]]) + p
    if hoof:
        for sd in (-1, 1): Q = spl(R([(-.2, sd * .05), (.7, sd * .08), (.95, sd * .36), (.6, sd * .66), (-.3, sd * .62)]), True, n=5); cel.fill(Q, '#4a3433'); kit.outline(cel, Q, lw, LN, 3)
        return
    def cap(a, L, w, base=.52):
        b = np.array([math.cos(a) * base, math.sin(a) * base]); e = np.array([math.cos(a), math.sin(a)]); t = b + e * L; n = np.array([-e[1], e[0]]); th = np.linspace(-math.pi / 2, math.pi / 2, 8); return np.vstack([[b - n * w - e * .2], t[None] + np.outer(np.cos(th), e) * w + np.outer(np.sin(th), n) * w, [b + n * w - e * .2]])
    F_ = _FG[kind]; th = np.linspace(0, 6.283, 24)[:-1]; palm = np.c_[np.cos(th) * .66, np.sin(th) * .64]; tb = dict(open=(-1.25, .52), flat=(-1.0, .5), point=(-1.2, .34), fist=(-.62, .34), grab=(-1.35, .5), cup=(-1.1, .46))[kind]
    for a, L, w in F_: Q = R(cap(a, L, w) * [1, flip]); cel.fill(Q, skin[0]); kit.outline(cel, Q, lw, LN, 2)
    cel.fill(R(palm), skin[0]); kit.outline(cel, R(palm), lw, LN, 5)
    for a, L, w in F_: cel.fill(R(cap(a, L * .55, w * .80, .30) * [1, flip]), skin[0])
    Q = R(cap(tb[0], tb[1], .21, .40) * [1, flip]); cel.fill(Q, skin[0]); kit.outline(cel, Q, lw, LN, 7); cel.fill(R(np.c_[np.cos(th) * .50, np.sin(th) * .48]), skin[0]); cel.c if 0 else None
    cel.shade(R([(-.9, .25 * flip), (.5, .40 * flip), (.5, 1.2 * flip), (-.9, 1.2 * flip)]), skin[1], R(np.c_[np.cos(th) * .58, np.sin(th) * .56]))


def arm(cel, P, w0, w1, col, shd, lw=4., seed=0, wob=0.):
    """thick spline limb with outline"""
    C = resample(spl(P, n=12), 5.); T = np.gradient(C, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; u = np.linspace(0, 1, len(C)); hw = (w0 * (1 - u) + w1 * u) / 2 * (1 + wob * np.sin(u * 30)); A, B = C + N * hw[:, None], C - N * hw[:, None]; poly = np.vstack([A, B[::-1]]); lo = np.where(N[:, 1:2] > 0, 1, -1)
    cel.fill(poly, LN); cel.line(A, lw, LN, taper=(0, 0), seed=seed, tremor=.1); cel.line(B, lw, LN, taper=(0, 0), seed=seed + 1, tremor=.1)
    for i in range(len(C) - 1): q = np.array([A[i], A[i + 1], B[i + 1], B[i]]); cel.fill(q, col); cel._path(q); cel._src(col, 1.); cel.ctx.set_line_width(1.2); cel.ctx.stroke()
    S0, S1 = C + N * lo * hw[:, None] * .25, C + N * lo * hw[:, None] * .96
    for i in range(len(C) - 1): q = np.array([S0[i], S0[i + 1], S1[i + 1], S1[i]]); cel.fill(q, shd)
    return C


def head(cx, cy, s, name='luffy', expr='calm', yaw=0., roll=0., look=(0, 0), **k):
    c = Cel(); Draw(c, cx, cy, s, roll, yaw).head(name, expr, look, **k); return c.layer()


NAMES = ['luffy', 'zoro', 'nami', 'usopp', 'sanji', 'chopper', 'robin', 'franky', 'brook', 'jinbe']
