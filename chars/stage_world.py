"""Painted boards for b4: the whole voyage as standing paintings, each with its own depth.  board(name) -> premult RGBA float, base line at the bottom centre."""
import numpy as np, math, cv2, os
from op import W, H, kit
from op.ink import Cel, spl, resample
from op.paint import rgb, f32, smooth, mix, fbm, repaint, ramp
from op.kit import s1

INK = '#131a2e'


class B:
    """a board in the making: colour cel (gets a gouache pass) + ink cel"""
    def __init__(s, w, h, seed=0): s.w, s.h, s.c, s.k, s.r, s.n = w, h, Cel(w, h), Cel(w, h), np.random.default_rng(seed), seed
    def p(s, d): return s1(d) if isinstance(d, str) else np.asarray(d, float)
    def fill(s, d, col, al=1.): s.c.fill(s.p(d), col, al); return s
    def shade(s, d, col, within): s.c.shade(s.p(d), col, s.p(within)); return s
    def shape(s, d, col, w=3., seed=None): q = s.p(d); s.c.fill(q, col); kit.outline(s.k, q, w, INK, s.n * 7 + len(q) if seed is None else seed, press=.45, breaks=2); return q
    def line(s, d, w=2., col=INK, al=1., taper=(.15, .4)): s.k.line(s.p(d), w, col, taper=taper, press=.4, seed=int(s.r.integers(1 << 20)), alpha=al); return s
    def hatch(s, d, a, gap=8, w=1.3, al=.7, length=34): kit.hatch(s.k, s.p(d), a, gap, w, INK, int(s.r.integers(1 << 20)), length, .6, alpha=al); return s
    def rock(s, poly, lit, mid, dark, side=1, w=3., strata=.0):
        """faceted rock mass: side=+1 lit on the right"""
        q = s.p(poly); s.c.fill(q, mid); x0, x1, y0, y1 = q[:, 0].min(), q[:, 0].max(), q[:, 1].min(), q[:, 1].max(); r = s.r
        s.c.clip(q)
        for i in range(int(18 + (x1 - x0) * (y1 - y0) / 9000)):
            x, y = r.uniform(x0, x1), r.uniform(y0, y1); L, Wd = r.uniform(.10, .30) * (y1 - y0), r.uniform(.03, .10) * (x1 - x0) + 4; t = (x - x0) / (x1 - x0); lt = (t if side > 0 else 1 - t) + r.normal(0, .28); col = lit if lt > .72 else dark if lt < .38 else None
            if col is None: continue
            tl = r.normal(0, .12) * L; s.c.fill(np.array([[x, y - L / 2], [x + Wd, y - L * .3], [x + Wd * .7 + tl, y + L * .4], [x + tl, y + L / 2], [x - Wd * .4 + tl * .5, y]]), col, .9)
        s.c.unclip(); s.k.clip(q)
        for i in range(int(14 + (y1 - y0) / 16)):
            x, y = r.uniform(x0, x1), r.uniform(y0, y1); L = r.uniform(.05, .2) * (y1 - y0); s.k.line(spl([(x, y), (x + r.normal(0, 7), y + L * .4), (x + r.normal(0, 12), y + L)], n=5), r.uniform(1., 2.4), INK, taper=(.1, .6), seed=i, alpha=.75)
        s.k.unclip(); kit.outline(s.k, q, w, INK, s.n + 3, press=.5, breaks=3); return q
    def leaf(s, x, y, R, cols=('#1f6a4a', '#3a9a56', '#8fce62'), n=9, ry=.6):
        for k, col in enumerate(cols):
            for i in range(n): a, q = s.r.uniform(0, 6.28), s.r.uniform(0, .8) ** .7; rr = R * (.46 - .11 * k) * s.r.uniform(.7, 1.2); t = np.linspace(0, 6.28, 12)[:-1]; s.c.fill(spl(np.c_[x + math.cos(a) * R * q + R * .10 * k + np.cos(t) * rr, y + math.sin(a) * R * q * ry - R * .12 * k + np.sin(t) * rr * .8], True, n=3), col)
        for i in range(5): a = s.r.uniform(2.8, 6.6); s.k.line(spl([(x + math.cos(a + d) * R * 1.02, y + math.sin(a + d) * R * .9 * ry - R * .1) for d in (0, .2, .4)], n=4), 1.8, INK, taper=(.3, .5), seed=i)
    def fall(s, x0, x1, y0, y1, sp=1.25):
        q = np.array([[x0, y0], [x1, y0], [x1 + (x1 - x0) * (sp - 1), y1], [x0 - (x1 - x0) * (sp - 1), y1]]); s.c.fill(q, '#bfe9f5')
        for i in range(int((x1 - x0) * .9) + 4): u = s.r.uniform(0, 1); a, b = sorted(s.r.uniform(0, 1, 2)); xa = x0 + (x1 - x0) * u; xb = xa + (u - .5) * (x1 - x0) * (sp - 1) * 2; s.c.line([(xa + (xb - xa) * a, y0 + (y1 - y0) * a), (xa + (xb - xa) * b, y0 + (y1 - y0) * b)], s.r.uniform(1., 4.), '#ffffff' if i % 3 else '#6fb8dc', taper=(.2, .5), seed=i)
        for i in range(14): x, R = s.r.uniform(x0 - 20, x1 + 20), s.r.uniform(8, 26); t = np.linspace(0, 6.28, 14); s.c.fill(np.c_[x + np.cos(t) * R, y1 - R * .2 + np.sin(t) * R * .6], '#f2fcff', .75)
        for x, g in ((x0, -1), (x1, 1)): s.k.line([(x, y0), (x + g * (x1 - x0) * (sp - 1) * .5, (y0 + y1) / 2), (x + g * (x1 - x0) * (sp - 1), y1)], 1.8, INK, taper=(.1, .5), seed=4, alpha=.6)
    def done(s, sizes=(9, 4.5), haze=None, hz=0., rimd=None, rimc='#fff0b8', rw=4., foot=None):
        L = s.c.rgba(); a = L[..., 3]; und = cv2.GaussianBlur(L, (0, 0), 14); bg = und[..., :3] / np.maximum(und[..., 3:4], 1e-3); g = L[..., :3] + (1 - a[..., None]) * np.where(und[..., 3:4] > .01, bg, .5)
        p = repaint(np.clip(g, 0, 1).astype(f32), sizes=sizes, seed=s.n + 1, thr=(0, .012), jit=(.012, .03, .03), dry=.18); out = np.dstack([p * a[..., None], a]); K = s.k.rgba(); out = out * (1 - K[..., 3:4]) + K
        if rimd is not None: out = kit.rim(out, rimd[0] * rw, rimd[1] * rw, rimc, .85, .7)
        if foot is not None: y = np.linspace(0, 1, s.h)[:, None, None]; f_ = smooth(1 - foot, 1., y); out = out * (1 - f_) + np.concatenate([np.ones((s.h, s.w, 3), f32) * rgb('#e8f8fb'), np.ones((s.h, s.w, 1), f32)], -1) * out[..., 3:4] * f_
        if haze: h = rgb(haze); out[..., :3] = out[..., :3] * (1 - hz) + h * out[..., 3:4] * hz
        return np.clip(out, 0, 1.1).astype(f32)


def cliffs():
    b = B(640, 980, 1); R = dict(lit='#e0a860', mid='#8d6a5c', dark='#473a5e')
    b.rock('M0 980 L0 60 C60 20 120 60 170 40 L210 180 L190 330 L260 420 L250 620 L330 700 L330 980 Z', **R, w=4); b.rock('M200 980 L240 560 L300 470 L420 500 L450 640 L520 720 L540 980 Z', **R, w=3.5); b.rock('M440 980 L470 800 L560 770 L610 860 L640 980 Z', **R, w=3)
    b.fall(96, 150, 110, 620); b.fall(300, 350, 520, 900); b.fall(196, 214, 350, 660); b.fall(500, 530, 800, 975)
    for x, y, q in ((40, 70, 60), (150, 60, 50), (200, 190, 34), (270, 440, 46), (400, 500, 52), (250, 560, 30), (470, 660, 40), (560, 780, 46), (610, 850, 30), (330, 720, 40), (20, 420, 40), (60, 660, 46)): b.leaf(x, y, q)
    for x, y, q in ((250, 620, 60), (560, 960, 70), (330, 940, 70), (130, 640, 60)): t = np.linspace(0, 6.28, 20); b.c.fill(np.c_[x + np.cos(t) * q, y + np.sin(t) * q * .45], '#f0fbff', .55)
    return b.done(rimd=(-1, .2))


def skull_island():
    b = B(720, 560, 2); dome = 'M60 470 C10 330 30 120 190 50 C300 5 440 10 530 70 C640 140 660 300 630 400 C620 440 590 455 560 455 L552 520 L160 520 L150 470 C120 480 80 490 60 470 Z'
    b.rock(dome, '#f1dcae', '#b79a86', '#5d4c6e', side=1, w=4)
    for d in ('M130 270 C110 190 190 150 260 180 C310 210 300 300 250 330 C190 350 140 330 130 270 Z', 'M370 190 C440 150 540 180 545 260 C545 330 470 350 410 325 C350 300 340 230 370 190 Z', 'M300 345 L330 300 L362 350 L345 392 L318 390 Z'): b.shape(d, '#22203c', 3.5); b.hatch(d, .9, 7, 1.4)
    for i in range(8): x = 176 + i * 47; b.shape(f'M{x} 436 C{x} 420 {x+40} 420 {x+42} 436 L{x+38} 512 C{x+26} 526 {x+12} 526 {x+4} 512 Z', '#efdcb2' if i > 3 else '#b9a08c', 2.6)
    b.fall(425, 470, 322, 540); b.line('M60 300 C120 250 130 200 120 120', 2.5); b.line('M560 120 C600 180 600 240 590 300', 2.2)
    for x, y, q in ((190, 50, 60), (270, 26, 70), (370, 18, 66), (470, 44, 64), (560, 100, 50), (110, 110, 46), (60, 240, 34), (640, 300, 30), (40, 500, 60), (680, 505, 60), (580, 500, 40), (120, 515, 46)): b.leaf(x, y, q)
    for x, h_, l in ((300, 110, -1), (350, 130, 1), (420, 100, 1)):
        b.line(f'M{x} 20 C{x+6*l} {20-h_*.4} {x+14*l} {20-h_*.7} {x+20*l} {22-h_}', 6, '#6a4a3a')
    b.fill('M0 560 C40 520 160 500 360 510 C560 500 690 520 720 560 Z', '#f3e3b0'); b.line('M0 556 C40 520 160 502 360 512 C560 502 690 520 720 556', 2)
    return b.done(rimd=(-1, .5), haze='#cfe9f0', hz=.06)


def _roof(b, cx, y, w, h, col='#c8402f', sh='#6f2038'):
    d = f'M{cx-w} {y} C{cx-w*.7} {y-h*.2} {cx-w*.45} {y-h*.55} {cx-w*.36} {y-h} L{cx+w*.36} {y-h} C{cx+w*.45} {y-h*.55} {cx+w*.7} {y-h*.2} {cx+w} {y} C{cx+w*.5} {y-h*.10} {cx-w*.5} {y-h*.10} {cx-w} {y} Z'; q = b.shape(d, col, 2.8); b.c.shade(b.p(f'M{cx-w} {y-h} L{cx-w*.05} {y-h} L{cx-w*.2} {y+10} L{cx-w} {y+10} Z'), sh, q)
    for i in range(-5, 6): b.line([(cx + i * w * .07, y - h * .96), (cx + i * w * .17, y - h * .16)], 1.2, al=.7)
    b.line(f'M{cx-w*.36} {y-h} L{cx+w*.36} {y-h}', 4, '#e8b84a')


def pagodas():
    b = B(640, 860, 3); R = dict(lit='#b9c07a', mid='#5f7a62', dark='#2f3f58', side=-1)
    b.rock('M60 860 L90 600 L180 520 L330 540 L360 470 L520 480 L600 600 L640 860 Z', **R, w=4)
    for cx, base, n, w in ((240, 528, 4, 84), (450, 474, 6, 96), (560, 596, 3, 60)):
        y = base
        for k in range(n): ww = w * (1 - .10 * k); hh = ww * .42; b.shape([(cx - ww * .58, y), (cx - ww * .54, y - hh), (cx + ww * .54, y - hh), (cx + ww * .58, y)], '#f3e2bd', 2.4); b.fill([(cx + ww * .1, y), (cx + ww * .1, y - hh), (cx + ww * .54, y - hh), (cx + ww * .58, y)], '#a98a8a'); [b.fill([(cx + ww * u - 4, y), (cx + ww * u - 4, y - hh * .7), (cx + ww * u + 5, y - hh * .7), (cx + ww * u + 5, y)], '#8c2f30') for u in (-.36, -.12, .12, .36)]; _roof(b, cx, y - hh + 4, ww, ww * .34); y -= hh + ww * .22
        b.line([(cx, y), (cx, y - w * .5)], 4, '#d8a83a'); [b.line([(cx - 9 + k, y - 10 - k * 7), (cx + 9 - k, y - 10 - k * 7)], 2.5, '#d8a83a') for k in range(5)]
    for x, y, q in ((140, 560, 56), (320, 560, 60), (380, 500, 40), (610, 640, 48), (90, 640, 50), (500, 560, 50), (420, 640, 60)): b.leaf(x, y, q, ('#b4577c', '#ee93b0', '#ffd3df'))
    for x, y, q in ((200, 700, 60), (520, 740, 60), (110, 800, 60), (360, 780, 50)): b.leaf(x, y, q)
    b.fall(296, 322, 560, 850); return b.done(rimd=(1, .3), haze='#d6ebee', hz=.10)


def clock():
    b = B(520, 900, 4); S_, D_ = '#f0d7a4', '#a5806c'
    for cx, w, y, h in ((120, 78, 900, 150), (410, 84, 900, 170), (330, 56, 900, 260)):
        b.shape([(cx - w, y), (cx - w, y - h), (cx + w, y - h), (cx + w, y)], S_, 3); b.fill([(cx - w, y), (cx - w, y - h), (cx - w * .3, y - h), (cx - w * .4, y)], D_); d = f'M{cx-w} {y-h} C{cx-w*1.25} {y-h-w*.7} {cx-w*.3} {y-h-w*1.25} {cx} {y-h-w*1.7} C{cx+w*.3} {y-h-w*1.25} {cx+w*1.25} {y-h-w*.7} {cx+w} {y-h} Z'; q = b.shape(d, '#4fae96', 3); b.c.shade(b.p(f'M{cx-w*1.4} {y-h} L{cx-w*1.4} {y-h-w*2} L{cx-w*.1} {y-h-w*2} L{cx-w*.35} {y-h} Z'), '#255f70', q); b.line([(cx, y - h - w * 1.7), (cx, y - h - w * 2.2)], 3, '#e8b84a')
    b.shape([(170, 900), (190, 250), (330, 250), (350, 900)], S_, 3.5); b.fill([(170, 900), (190, 250), (236, 250), (228, 900)], D_); b.shape([(166, 262), (166, 120), (354, 120), (354, 262)], '#f7e3b4', 3.5); b.fill([(166, 262), (166, 120), (214, 120), (214, 262)], D_); t = np.linspace(0, 6.283, 40); ci = lambda R: np.c_[260 + np.cos(t) * R, 190 + np.sin(t) * R]; b.shape(ci(58), '#fffdf2', 4); b.line(ci(50), 1.5)
    for k in range(12): a = k * math.pi / 6; b.line([(260 + 42 * math.cos(a), 190 + 42 * math.sin(a)), (260 + 50 * math.cos(a), 190 + 50 * math.sin(a))], 3)
    b.line([(260, 190), (260, 150)], 5, taper=(0, .2)); b.line([(260, 190), (286, 204)], 5, taper=(0, .2)); _roof(b, 260, 124, 112, 96, '#4fae96', '#255f70'); b.line([(260, 30), (260, 0)], 4, '#e8b84a')
    for y in (330, 420, 510, 600): b.shape(f'M244 {y+46} L244 {y+14} C244 {y} 276 {y} 276 {y+14} L276 {y+46} Z', '#4a3a56', 2)
    for x in (30, 470, 250): b.leaf(x, 880, 46)
    return b.done(rimd=(1, .4), haze='#e4ecd8', hz=.16)


def river():
    b = B(900, 900, 5); b.rock('M0 900 L70 700 L150 640 L230 420 L330 300 L380 120 L430 60 L470 60 L520 130 L580 310 L690 430 L740 620 L840 720 L900 900 Z', '#d88a62', '#9a5560', '#4c3058', side=1, w=4)
    q = b.p('M400 900 C420 700 390 500 430 300 C440 200 436 120 440 60 L466 60 C470 130 462 210 480 300 C520 500 480 700 520 900 Z'); b.c.fill(q, '#d8f2fb'); b.c.clip(q)
    for i in range(90): x = b.r.uniform(390, 520); y = b.r.uniform(40, 860); L = b.r.uniform(40, 150); b.c.line(spl([(x, y), (x + (455 - x) * .2, y - L * .5), (x + (455 - x) * .4, y - L)], n=5), b.r.uniform(1.5, 5), '#ffffff' if i % 3 else '#5fb0da', taper=(.2, .4), seed=i)
    b.c.unclip(); kit.outline(b.k, q, 2.6, INK, 4)
    for s_ in (-1, 1):                                             # at the summit it leaps and splits
        for k in range(5): b.c.line(spl([(453, 66), (453 + s_ * (30 + 14 * k), 10 + 6 * k), (453 + s_ * (80 + 22 * k), 60 + 22 * k)], n=8), 9 - k, '#f4fdff', taper=(.1, .6), seed=k)
    for i in range(26): x, y, R = b.r.uniform(60, 840), b.r.uniform(760, 890), b.r.uniform(20, 60); t = np.linspace(0, 6.28, 16); b.c.fill(np.c_[x + np.cos(t) * R, y + np.sin(t) * R * .4], '#eef9fc', .5)
    return b.done(haze='#dfe6ee', hz=.30, rimd=(-1, .6))


def whale():
    b = B(620, 620, 6); body = 'M40 620 C30 420 60 200 170 90 C250 20 390 20 470 90 C570 190 600 420 590 620 Z'; q = b.shape(body, '#31456f', 5); b.shade('M40 620 C30 420 60 200 170 90 C210 60 240 50 270 46 C180 200 150 420 160 620 Z', '#17203f', body)
    b.shape('M120 620 C130 520 200 470 315 470 C430 470 500 520 510 620 Z', '#cfd3dc', 4)
    for i in range(13): x = 150 + i * 27; b.line(f'M{x} 620 C{x+ (315-x)*.1} 560 {x+(315-x)*.2} 520 {x+(315-x)*.25} 484', 2.2)
    for s_ in (-1, 1): x = 315 + s_ * 212; t = np.linspace(0, 6.283, 22); b.shape(np.c_[x + np.cos(t) * 22, 400 + np.sin(t) * 17], '#f4f0e4', 3); b.fill(np.c_[x + np.cos(t) * 9, 402 + np.sin(t) * 9], '#101424')
    # the promise painted on its scarred brow: skull, crossed bones, straw hat
    t = np.linspace(0, 6.283, 30)
    for a in (.5, -.5): c_, s_ = math.cos(a), math.sin(a); b.fill(np.array([[-140, -12], [140, -12], [140, 12], [-140, 12]]) @ np.array([[c_, s_], [-s_, c_]]) + [315, 262], '#f4efe2')
    b.shape(np.c_[315 + np.cos(t) * 78, 262 + np.sin(t) * 72], '#f4efe2', 3); b.shape([(272, 312), (358, 312), (350, 350), (280, 350)], '#f4efe2', 3)
    for s_ in (-1, 1): b.fill(np.c_[315 + s_ * 30 + np.cos(t) * 20, 268 + np.sin(t) * 22], '#141a30')
    b.shape('M200 236 C260 250 370 250 430 236 C430 222 400 218 378 214 C372 160 350 138 315 138 C280 138 258 160 252 214 C230 218 200 222 200 236 Z', '#e9c053', 3); b.fill('M252 214 C270 222 360 222 378 214 L376 196 C350 206 280 206 254 196 Z', '#b3202a')
    for d in ('M110 200 L190 300', 'M100 250 L170 340', 'M460 140 L540 250', 'M230 80 L300 110', 'M480 300 L550 380', 'M410 90 L360 60'): b.line(d, 6, '#8fa2c8', taper=(.3, .4))
    b.hatch('M60 400 L150 120 L230 60 L160 300 L150 560 Z', 1.2, 10, 1.6); return b.done(rimd=(-1, .5), haze='#d8e6ee', hz=.14)


def dome():
    b = B(760, 560, 7); d = 'M110 470 C80 300 170 150 380 140 C590 150 680 300 650 470 C640 520 600 520 590 560 L170 560 C160 520 120 520 110 470 Z'; b.rock(d, '#7d7896', '#4a4668', '#262446', side=-1, w=4)
    for s_ in (-1, 1): m = lambda x: 380 + s_ * (x - 380); pts = [(m(560), 190), (m(640), 170), (m(730), 110), (m(752), 10), (m(700), 70), (m(640), 100), (m(530), 130)]; q = spl(pts, True, corners=(3,), n=8); b.shape(q, '#8f86a0', 4); b.c.shade(q + [0, 22], '#3f3a5c', q)
    for s_ in (-1, 1): m = lambda x: 380 + s_ * (x - 380); q = np.array([(m(420), 350), (m(470), 300), (m(570), 300), (m(590), 370), (m(520), 410), (m(440), 400)], float); b.shape(q, '#140f24', 3); t = np.linspace(0, 6.28, 16); b.fill(np.c_[m(500) + np.cos(t) * 22, 356 + np.sin(t) * 16], '#ff9a3a'); b.fill(np.c_[m(500) + np.cos(t) * 10, 356 + np.sin(t) * 8], '#fff0b0')
    b.shape([(350, 430), (380, 385), (410, 430), (395, 452), (365, 452)], '#140f24', 3)
    for i in range(9): x = 206 + i * 39; b.shape(f'M{x} 480 L{x+36} 480 L{x+30} 560 L{x+6} 560 Z', '#8a84a0' if i > 4 else '#5d587c', 2.4)
    b.hatch('M110 470 C80 300 170 150 380 140 L330 300 L250 520 Z', 1.0, 9, 1.5); return b.done(haze='#cfdcea', hz=.38, rimd=(1, .6))


def elephant():
    b = B(1200, 1400, 8); M, D_ = '#8f98b4', '#6a7398'
    for x, w in ((120, 190), (420, 150), (800, 160), (960, 210)): d = f'M{x} 1400 C{x+10} 1000 {x-30} 700 {x-10} 300 L{x+w+30} 300 C{x+w+20} 700 {x+w-20} 1000 {x+w+10} 1400 Z'; q = b.shape(d, M, 4); b.shade(f'M{x-60} 1400 L{x-60} 300 L{x+w*.45} 300 C{x+w*.3} 800 {x+w*.4} 1100 {x+w*.3} 1400 Z', D_, d); [b.line(f'M{x+6} {y} C{x+w*.3} {y+22} {x+w*.7} {y+20} {x+w-6} {y-4}', 2.2, al=.6) for y in range(480, 1380, 74)]
    b.shape('M0 340 C80 160 400 120 640 130 C900 110 1120 180 1200 340 C1000 420 200 420 0 340 Z', M, 4); b.fill('M0 340 C80 250 300 300 640 310 C900 300 1120 260 1200 340 C1000 420 200 420 0 340 Z', D_)
    for i in range(9): b.leaf(260 + i * 80, 140 + 10 * math.sin(i), 44, ('#5f8a86', '#7fae98', '#b9d8a8'))
    for x in (420, 640, 820): b.shape([(x - 16, 120), (x - 10, 50), (x, 20), (x + 10, 50), (x + 16, 120)], '#a8a6b8', 2.5)
    o = b.done(haze='#d2e4ee', hz=.56); y = np.linspace(0, 1, o.shape[0])[:, None, None]; return (o * (1 - .72 * smooth(.45, 1., y)) * (1 - .0 * y)).astype(f32)


def bell():
    b = B(560, 460, 9); t = np.linspace(0, 6.283, 22)
    for k, (col, off, sc) in enumerate((('#a9cde6', 10, 1.), ('#e6f5fb', 0, .95), ('#ffffff', -10, .78), ('#fff4cf', -18, .42))):
        r = np.random.default_rng(3)
        for i in range(26): x = 280 + r.normal(0, 120); R = r.uniform(34, 74) * sc * (1 - abs(x - 280) / 620); b.c.fill(np.c_[x + np.cos(t) * R, 370 - r.uniform(0, 60) + off + np.sin(t) * R * .78], col)
    b.line(spl([(330, 330), (340, 240), (318, 150), (330, 60), (322, 0)], n=8), 12, '#4f9a5a', taper=(0, .1)); [b.leaf(330 + (-1) ** k * 16, 40 + k * 56, 16, n=4) for k in range(5)]
    for x in (180, 270): b.line([(x, 340), (x, 170)], 9, '#c9a04a', taper=(0, 0))
    b.line([(166, 176), (284, 176)], 11, '#c9a04a', taper=(0, 0)); q = b.shape('M225 184 C200 190 196 240 190 280 C186 300 172 308 168 320 L282 320 C278 308 264 300 260 280 C254 240 250 190 225 184 Z', '#f6c845', 4); b.shade('M160 180 L212 180 C196 230 196 290 180 330 L150 330 Z', '#a8702c', q); b.fill('M236 200 C248 240 250 270 262 300 L252 300 C246 270 238 230 230 204 Z', '#fff3b8'); b.fill(np.c_[225 + np.cos(t) * 12, 326 + np.sin(t) * 10], '#7a4a1e'); return b.done(haze='#ffffff', hz=.06, rimd=(-.6, -.6))


def drums():
    b = B(620, 760, 10)
    for x, w, y in ((60, 110, 300), (200, 90, 170), (320, 150, 60), (500, 100, 240)):
        d = f'M{x} 760 L{x+6} {y+60} C{x+10} {y} {x+w-10} {y} {x+w-6} {y+60} L{x+w} 760 Z'; q = b.shape(d, '#a9b4d0', 3); b.shade(f'M{x-9} 760 L{x-9} {y-40} L{x+w*.42} {y-40} L{x+w*.36} 760 Z', '#6d7aa8', d); b.shape(f'M{x+2} {y+84} C{x+6} {y} {x+w-6} {y} {x+w-2} {y+84} C{x+w*.85} {y+66} {x+w*.8} {y+120} {x+w*.66} {y+78} C{x+w*.5} {y+120} {x+w*.4} {y+70} {x+w*.28} {y+110} C{x+w*.15} {y+66} {x+w*.1} {y+96} {x+2} {y+84} Z', '#fbfdff', 2.6); [b.line([(x + w * u, y + 150 + 40 * u), (x + w * u, 740)], 1.6, al=.4) for u in (.2, .4, .62, .8)]
    b.shape([(360, 66), (362, 30), (372, 24), (382, 8), (392, 24), (404, 14), (414, 30), (430, 34), (432, 66)], '#8d7f9c', 2.4)
    for x, y, q in ((330, 60, 60), (430, 50, 70), (390, 20, 66), (480, 90, 50), (290, 100, 46)): b.leaf(x, y - 20, q, ('#f0a0c0', '#ffc4d8', '#fff0f4'), 8)
    b.fill('M0 760 C120 690 500 690 620 760 Z', '#f4f9ff'); return b.done(haze='#dfe9f4', hz=.34, rimd=(1, .4))


def sunny_board():
    from op import ships
    from op.cam import Camera3D
    cam = Camera3D((58, 40, 9), (0, 0, 12.5), 1500, 1400, 1200, cx=700, cy=640); d = ships.draw([(ships.sunny(.6), ships.pose(0, 0, 0, 0))], cam, light=(-.4, -.5, .7), lw=1.1); L = d.layer if hasattr(d, 'layer') else d
    if isinstance(L, tuple): l, ox, oy = L; o = np.zeros((1200, 1400, 4), f32); over_paste(o, l, ox, oy); L = o
    L = L[:1200, :1400]; ys, xs = np.nonzero(L[..., 3] > .02); L = L[ys.min():ys.max() + 1, xs.min():xs.max() + 1]; L = kit.rim(L, 5, 2, '#fff0c0', .9, .7); return L.astype(f32)


def over_paste(o, l, ox, oy):
    from op.ink import paste
    paste(o, l, ox, oy)


def boat(kind=0):
    b = B(300, 340, 20 + kind); cols = (('#c8402f', '#7d2238'), ('#f4ead2', '#b09a94'), ('#e8b84a', '#9a642c'))[kind]
    b.line([(150, 280), (146, 20)], 7, '#5a3a2c', taper=(0, 0)); q = b.shape('M140 40 C90 90 70 170 64 240 C100 250 120 250 140 244 Z', cols[0], 3); b.c.shade(b.p('M64 260 L60 180 C90 200 120 200 146 190 L146 260 Z'), cols[1], q); q = b.shape('M154 30 C210 90 236 170 240 236 C206 250 180 250 154 244 Z', cols[0], 3); b.c.shade(b.p('M154 20 L190 70 C176 130 176 200 190 250 L150 250 Z'), cols[1], q)
    for y in (110, 160, 206): b.line([(132, y), (80 + (206 - y) * .1, y + 8)], 1.6, al=.5); b.line([(160, y), (222, y + 8)], 1.6, al=.5)
    h = b.shape('M50 272 C90 284 210 284 254 268 C248 300 226 322 200 326 L104 328 C76 322 58 300 50 272 Z', '#a8733e', 3.5); b.c.shade(b.p('M40 300 C120 306 200 304 260 290 L260 340 L40 340 Z'), '#5b3838', h); b.line([(60, 290), (244, 286)], 2)
    return b.done(rimd=(-1, .5), sizes=(6, 3))


ALL = dict(boat0=lambda: boat(0), boat1=lambda: boat(1), boat2=lambda: boat(2), cliffs=cliffs, skull=skull_island, pagodas=pagodas, clock=clock, river=river, whale=whale, dome=dome, elephant=elephant, bell=bell, drums=drums, sunny=sunny_board)


def board(name):
    fn = f'cache/b4_{name}.npy'
    if os.path.exists(fn): return np.load(fn).astype(f32)
    o = ALL[name](); np.save(fn, o.astype(np.float16)); return o
