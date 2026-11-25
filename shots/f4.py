"""f4 60: rubber-hose comedy: eyes pop f5, jaw on the floor f9, skips rope with the dragon (f15 30 45) on the bouncing skull island, hurls it f50."""
import numpy as np, cv2, math, os
from op import W, H as HH
from op.paint import rgb, f32, smooth, ramp, grid, fbm, mix, repaint
from op.ink import Cel, spl, resample
from op.kit import outline, shape, hatch, para, speed_lines, smear, zoomblur, focus_lines, flare
from op.comp import over, screen, blur, radial, ease, cam2d
from op.cam import Camera3D
from op.clouds import Sky
from op.fx_core import h1, lin01, noise, c2
from op.fx_break import smoke_burst, dust_skirt, puffs, DUST
from chars import gear5 as G
from chars.gear5 import ss, lay, back_out, rim, hotrim, inked, hose, lock, paint_hair
from shots.g5kit import Keys, Rub, PROPS, SUNP, star
from chars import kaido as KD

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
U, GYB = 127., 932.
SLAP = (15, 30, 45)
TR = {46: (26, -60), 47: (70, -34), 48: (84, 4), 49: (84, 10), 50: (170, -30)}
PAL_D = dict(lit='#e9d2cf', mid='#a08fb6', dark='#5d4f86', ink='#1c1630')
NP = dict(SUNP, wall='#c6b4d0', wall1='#8d7fae', roof='#4b4a8c', roof1='#2f2e66', rooft='#8d90cc', red='#c5414a', red1='#7d2a50', ink='#17122a', glow='#ffd98a')
cam = Camera3D((0, -15., 3.3), (0, 0, 3.1), 15. * U)
CT = Keys(dict(                                                                              # close-up cartoon take, f0-12
    sq={0: 1.0, 1: .70, 2: .64, 3: 1.40, 4: 1.28, 5: 1.12, 6: 1.18, 8: 1.06, 9: .84, 10: 1.06, 11: .98, 12: .70}, pop={0: 0, 2: 0, 3: .55, 4: 1.30, 5: 1.14, 6: .94, 7: 1.06, 8: 1.0, 9: 1.08, 10: .98, 11: .9, 12: 0},
    jaw={0: .0, 2: 0, 4: .30, 5: .42, 6: .8, 7: 1.35, 8: 2.1, 9: 2.62, 10: 2.40, 11: 2.50, 12: .2}, chin={0: 0, 7: 0, 8: -.1, 9: .85, 10: .42, 11: .56, 12: 0}, tong={0: 0, 8: 0, 9: .25, 10: .75, 11: 1.0, 12: 0}))
HOP = lambda f: max([4 * 1.68 * t * (1 - t) if 0 < t < 1 else 0. for t in [(f - b + 5.6) / 11.2 for b in SLAP]] + [0.])


def phi(f):
    a = 2 * math.pi * (f - 15) / 15.; return a + .46 * math.sin(a)


WK = Keys(dict(                                                                               # wide: body
    sq={12: 1.1, 15: .92, 18: 1.12, 19: 1.18, 21: .78, 22.5: .66, 24.5: 1.28, 27: 1.10, 30: .94, 34: 1.18, 36: .78, 37.5: .66, 39.5: 1.28, 42: 1.08, 45: 1.02, 49: .78, 50: .74, 51: 1.22, 53: 1.12, 55: .94, 57: 1.03, 59: 1.0},
    lean={12: -.1, 15: -.26, 18: -.12, 20.5: -.34, 23: -.62, 25.5: -.24, 30: -.32, 33: -.15, 35.5: -.36, 38.2: -.66, 41: -.22, 45: -.2, 47: -.50, 48.5: -.92, 50: .30, 51: .80, 53: .66, 56: .30, 59: .24}, roll={12: 0, 15: .16, 22: 0, 30: -.2, 37: 0, 45: 0, 48.5: -.35, 51: .40, 54: .20, 59: .10},
    px={12: 0, 15: 0, 20.4: -.10, 24.4: .42, 30: .12, 35.4: -.05, 39.4: .52, 45: .10, 47: -.15, 49.4: -.55, 51: .10, 55: .25, 59: .2}, fA={13: (-.5, .6), 15: (-1.0, 2.30), 18: (-.7, 1.0), 20.4: (-.5, 0), 24.4: (-.5, 0), 27: (-.95, 1.2), 30: (-1.50, 2.65), 33: (-.9, .9), 35.4: (-.5, 0), 39.4: (-.5, 0), 42: (-1.1, 1.3), 45: (-1.85, 2.75), 47.6: (-.9, .3), 48.6: (-.45, 0), 50: (-.80, 0), 52: (-1.35, .95), 54: (-1.2, .8), 57: (-.7, .02), 59: (-.7, 0)},
    fB={13: (.5, .6), 15: (1.0, 2.30), 18: (.7, 1.0), 20.4: (.5, 0), 24.4: (.5, 0), 27: (.5, .8), 30: (.75, 1.95), 33: (.45, .6), 35.4: (.5, 0), 39.4: (.5, 0), 42: (1.1, 1.3), 45: (1.85, 2.75), 47.6: (1.1, .3), 48.6: (1.15, 0), 59: (.75, 0)}))
RING_C, RING_R = np.array([0., 0., 5.35]), 2.25


def hands(f):
    """(x, z) of both hands in U, as they swing the rope, whirl it overhead, throw"""
    a = phi(f); k = (1 - math.cos(a)) / 2; hop = HOP(f); A = np.array([-1.55 - .16 * math.sin(a), 1.95 + .75 * hop + 1.35 * k]); B = np.array([1.55 + .16 * math.sin(a), 1.95 + .75 * hop + 1.35 * k])
    if f > 45:
        w = float(ease(np.clip((f - 45.5) / 2.5, 0, 1))); an = (f - 46) * 1.25; R_ = .34; A2 = np.array([-.28 + R_ * math.cos(an), 5.25 + .16 * math.sin(an)]); B2 = np.array([.28 + R_ * math.cos(an + .6), 5.30 + .16 * math.sin(an + .6)]); bu = 1.15 * math.sin(math.pi * min(1., w)); A, B = A * (1 - w) + A2 * w + np.array([-bu, 0.]), B * (1 - w) + B2 * w + np.array([bu, 0.])          # the arms go up round the face, not across it
    if f >= 49.5:
        w = float(back_out(np.clip((f - 49.5) / 2.0, 0, 1), 2.2)); A = A * (1 - w) + np.array([1.55, 4.7]) * w; B = B * (1 - w) + np.array([2.85, 5.25]) * w
    if f >= 53:
        w = float(back_out(np.clip((f - 53) / 4., 0, 1), 1.5)); A = A * (1 - w) + np.array([-.95, 2.25]) * w; B = B * (1 - w) + np.array([1.02, 4.62]) * w
    return A, B


def axes(f):
    """lasso plane: centre, right, throw direction, and the ring's second axis (far side dips behind him, levels out on the release)"""
    ex = np.array([.34, .90, .165]); ex /= np.linalg.norm(ex); e1 = np.array([1., -.15, .06]); e1 -= ex * (e1 @ ex); e1 /= np.linalg.norm(e1); tl = float(ss(50, 49, f)); ey = ex + np.array([0, 0, -.42]) * tl; return RING_C + np.array([0, 0, .3 - .42 * tl]), e1, ex, ey / np.linalg.norm(ey)


def trail(L, dx, dy, n=8, a=.85):
    """cel smear: the drawing stays crisp, its colour is dragged out behind it"""
    o = np.zeros_like(L); T = np.linspace(.1, 1, n)
    for t in T: o += cv2.warpAffine(L, np.float32([[1, 0, -dx * t], [0, 1, -dy * t]]), (W, HH)) * (1 - .6 * t)
    return L + o * (a / (1 - .6 * T).sum()) * (1 - L[..., 3:4])


def rope(f):
    """dragon control points, head first (x right, y away, z up), U units"""
    A, B = hands(f); n = 26; v = np.linspace(0, 1, n); a = phi(f) + .80 * (1 - np.sin(np.pi * v)); s = np.sin(np.pi * v); k = (1 - np.cos(phi(f))) / 2; rho = ((A[1] + B[1]) / 2 - .57) * (1 - k) * s + 3.30 * k * s
    zh = B[1] + (A[1] - B[1]) * v; x = (A[0] + B[0]) / 2 + (B[0] - A[0]) / 2 * np.cos(np.pi * v) * (1 + .42 * s)
    P = np.stack([x, np.sin(a) * rho, zh - np.cos(a) * rho], 1)
    P[:, 2] = np.maximum(P[:, 2], .56 + .02 * np.sin(v * 20))
    sw = .35 * math.sin(phi(f) - .9); head_ = np.array([[B[0] + 1.55, -.45 + sw * .6, B[1] + .48 + sw], [B[0] + .80, -.28, B[1] + .30 + sw * .4]]); tail = np.array([[A[0] - .50, -.1, A[1] + .25 - sw * .3], [A[0] - 1.15, -.1, A[1] + .10 - sw]])
    P = np.vstack([head_, P, tail])
    if f <= 45.5: return P
    # overhead lasso, then unspool along the tangent into the sky
    m = len(P); Lr = 12.6; s0 = (f - 49.5) * (1.55 if f < 49.5 else 1.) ; sh_ = s0 if f < 49.5 else 2.6 * (f - 49.5) ** 1.62 + 1.2 * (f - 49.5)
    C0, e1, ex, ey = axes(f); sv = sh_ - np.linspace(0, 1, m) * Lr * (1 + .05 * max(0, f - 49.5)); Q = np.empty((m, 3))
    for i, q in enumerate(sv):
        if q > 0: Q[i] = C0 + e1 * RING_R + ex * q + np.array([0, 0, 1.]) * .22 * math.sin(q * .9 - f * .9) * min(1, q / 4)
        else: th = q / RING_R; Q[i] = C0 + e1 * RING_R * math.cos(th) + ey * RING_R * math.sin(th)
    w = float(ease(np.clip((f - 45.2) / 2.4, 0, 1))); return P * (1 - w) + Q * w


class Shot:
    id = 'f4'; n = 60; finish = dict(bloom=.30, vig=.16)
    EVENTS = [(1, 'squeak', dict(size=.5, note='head squashes down, anticipation')), (3, 'pop', dict(size=.8, note='eyes shoot out: slide whistle up, boing on f5')), (6, 'slide_down', dict(size=.6, note='jaw falling')), (9, 'thud', dict(size=.9, note='jaw hits the floor, teeth rattle f9 f10 f11')), (11, 'zip', dict(size=.5, note='rubber arms shoot out both ways')), (12, 'snap', dict(size=.7, note='everything snaps back')),
              (15, 'rope', dict(size=1.0, note='dragon slaps the skull like a rope, island boings')), (21, 'bounce', dict(size=.5, note='lands')), (30, 'rope', dict(size=1.0)), (36, 'bounce', dict(size=.5)), (45, 'rope', dict(size=1.0, note='splits')), (46, 'whirl', dict(size=.7, note='lasso whirl rising to f50')), (50, 'throw', dict(size=1.0, note='hurl, stars and dust')), (57, 'ting', dict(size=.6, note='twinkle in the sky'))]

    def setup(self):
        self.gx, self.gy = grid(); p = os.path.join(ROOT, 'cache/f4_sky3.npy')
        if os.path.exists(p): self.sky = np.load(p).astype(f32)
        else: self.sky = self.paint_sky(); np.save(p, self.sky.astype(np.float16))
        self.tx = fbm(HH, W, (120, 40, 14, 6), None, 21); self.D = None

    def paint_sky(self):
        x, y = grid(); n = fbm(HH, W, (260, 90, 30), None, 5); d = np.hypot(x - 960, (y - 430) * 1.25)
        sky = ramp(np.clip(d / 1250 + .06 * n, 0, 1), [(0, '#8a6fd0'), (.22, '#5a4bb0'), (.5, '#343486'), (.8, '#1c2260'), (1, '#12163f')])
        m = np.hypot(x - 400, y - 250); sky = sky + np.exp(-(m / 330) ** 1.5)[..., None] * rgb('#ffe3a0') * .40
        s = Sky(HH, W, sky, (.35, -.5, .6), 2)
        for x0, b, w, h, sd, T in ((60, 560, 900, 360, 3, ('#8f86d8', '#5650a6', '#33317a', '#1d1e52')), (1860, 560, 1000, 430, 4, ('#9a86d2', '#574fa4', '#322f78', '#1c1d50')), (1250, 300, 560, 130, 8, ('#b7a2e2', '#6b5fb6', '#403c88', '#26275f')), (760, 190, 460, 100, 9, ('#b7a2e2', '#6b5fb6', '#403c88', '#26275f')),
                                   (300, 830, 1100, 380, 5, ('#a08be0', '#5d53ac', '#37347e', '#201f55')), (1640, 850, 1150, 420, 6, ('#ab90dc', '#5f54aa', '#38347c', '#211f54')), (960, 900, 1500, 260, 7, ('#c9a8e4', '#7565b8', '#473f8c', '#2a2864'))): s.bank(x0, b, w, h, T, T[2], sd, pool=.3, foot=.25)
        out = s.paint(3).astype(f32); cal = ramp(np.clip(d / 1250 + .03 * n, 0, 1), [(0, '#2c2a78'), (.22, '#2a2873'), (.5, '#302f7c'), (.8, '#1f2464'), (1, '#12163f')]); kk = (.62 * np.exp(-(d / 620) ** 2) + .24)[..., None]; out = out * (1 - kk) + cal * kk + (np.exp(-((d - 560) / 170) ** 2))[..., None] * rgb('#8f6fd8') * .10
        md = smooth(172, 169, m + 2.2 * fbm(HH, W, (40, 14), (1, .5), 6))[..., None]; moon = ramp(np.clip((x - 400) / 340 * .6 + (y - 250) / 340 * .5 + .5 + .06 * n, 0, 1), [(0, '#fffbe0'), (.5, '#ffeab0'), (1, '#f3c987')]); out = out * (1 - md) + moon * md
        return out

    # ------------------------------------------------------------------ the skull island
    def skull(self, f, cel, dy, sq, wob):
        ink = '#16122a'; bone, b1, b2 = '#b9a9cf', '#7f70a6', '#4d4280'; lw = 4.2; cx, top = 960., GYB + dy
        X = lambda P: c2(cx + (np.asarray(P, float)[:, 0] - cx) * (1 / math.sqrt(sq)), 1360 + (np.asarray(P, float)[:, 1] - 1360) * sq + dy)
        for g in (-1, 1):                                                                    # horns with lagging tips
            C = spl([(cx + g * 600, 1040), (cx + g * 800, 960), (cx + g * 905, 770), (cx + g * (860 + wob * 20 * g), 560), (cx + g * 770 + wob * 46, 430 + abs(wob) * 12)], n=14); m = len(C); v = np.linspace(0, 1, m); hw = 118 * (1 - v) ** .72 + 3; T = np.gradient(C, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None]; N = c2(-T[:, 1], T[:, 0]); P = X(np.vstack([C + N * hw[:, None], (C - N * hw[:, None])[::-1]]))
            inked(cel, P, '#a58fb4', lw, ink); cel.clip(P); sg = 1 if g > 0 else -1; cel.fill(X(np.vstack([C - N * (hw * .1 * 1)[:, None], (C - N * hw[:, None] * 1.1 * 1)[::-1]])) if g < 0 else X(np.vstack([C + N * (hw * .1)[:, None], (C + N * hw[:, None] * 1.1)[::-1]])), '#5f5190'); cel.unclip()
            for k in range(3, m - 6, 5): q = spl([C[k] + N[k] * hw[k] * .96, C[k] + T[k] * hw[k] * .30, C[k] - N[k] * hw[k] * .96], n=6); cel.line(X(q), 2.2, ink, taper=(.15, .4), seed=k)
        a = np.linspace(math.pi, 2 * math.pi, 90); dome = np.vstack([c2(cx + np.cos(a) * 860, GYB + 462 + np.sin(a) * 462), [(1920, 1400), (0, 1400)]]); D = X(dome); inked(cel, D, bone, lw * 1.2, ink); cel.clip(D)
        cel.fill(X(np.vstack([c2(cx + np.cos(a) * 1000, 1400 + np.sin(a) * 452), [(1920, 1500), (0, 1500)]])), b1)
        for g in (-1, 1):
            ew = 1 + .16 * abs(wob) ; e = spl([(cx + g * 110, 1080), (cx + g * 190, 1018 - 8 * ew), (cx + g * 330, 1004 - 12 * ew), (cx + g * 440, 1060), (cx + g * 420, 1180), (cx + g * 200, 1180)], True, n=8); inked(cel, X(e), '#1d1234', lw, ink)
            cel.fill(X(c2(cx + g * 270 + np.cos(a * 2) * 64 * ew, 1106 + np.sin(a * 2) * 58 * ew)), '#d8541e'); cel.fill(X(c2(cx + g * 270 + np.cos(a * 2) * 36, 1106 + np.sin(a * 2) * 32)), '#ffc24a')
            br = spl([(cx + g * 80, 1060), (cx + g * 190, 985 - 10 * wob), (cx + g * 340, 968 - 16 * wob), (cx + g * 480, 1040)], n=8); cel.line(X(br), 9, ink, taper=(.2, .3), seed=3); cel.line(X(br - [0, 12]), 4, '#dccbe4', taper=(.3, .4), seed=4)
        cel.unclip()
        for k in range(13):                                                                  # sutures and cracks
            x = 250 + 118 * k + 30 * h1(k, 2); y = GYB + 462 - 462 * math.sqrt(max(0, 1 - ((x - cx) / 860) ** 2)); P = [(x, y + 5)]
            for j in range(5): P.append((P[-1][0] + (h1(k, j) - .5) * 46 + (cx - x) * -.02, P[-1][1] + 14 + 18 * h1(k, j, 2)))
            cel.line(X(P), 2.4 - 1.0 * h1(k, 5), ink, taper=(.05, .7), seed=k)
        hatch(cel, X(np.array([(520, 985), (960, 930), (1420, 985), (1380, 1040), (960, 990), (560, 1040)])), .12, 13, 1.5, b2, 4, length=60, alpha=.7)

    def swirl(self, cel, c, r, f, sg=1):
        a = np.linspace(0, 6.283, 30); inked(cel, c + c2(np.cos(a), np.sin(a)) * r, '#ffffff', 2.2, '#14111c'); v = np.linspace(0, 1, 50); th = sg * (v * 15 + f * .9); cel.line(c + c2(np.cos(th), np.sin(th)) * (r * .88 * v)[:, None], 2.2, '#14111c', taper=(.1, .1), press=.1)

    # ------------------------------------------------------------------ close-up: the take
    def take(self, f):
        sq, pop, jaw, chin, tong = [CT(k, f) for k in ('sq', 'pop', 'jaw', 'chin', 'tong')]; cx, s = 660., 186.; base = 420.; vib = (4 * (-1) ** f if 4 <= f <= 11 else 0.)
        cy = base - (sq - 1) * 150 + vib * .4; img = cam2d(self.sky, 900 + 8 * f, 420, 1.45); roar = .75 + .25 * math.sin(f * 2.3)
        lg = np.clip((self.gx - 700) / 1220, 0, 1) ** 1.6; img = img * (1 - .25 * lg[..., None]) + (lg * roar)[..., None] * rgb('#ff7a2a') * .85 + (lg ** 3)[..., None] * rgb('#ffe08a') * .5
        sl = Cel(ss=1); speed_lines(sl, math.pi + .05 * math.sin(f), 60, 4.5, (260, 1200), '#ffd9a0', seed=f * 3 + 1, alpha=.40); img = over(img, sl.rgba())
        fl = Cel(); fy = 1000.; gnd = np.vstack([spl([(-40, fy + 6), (500, fy - 26), (1100, fy - 20), (1980, fy + 20)], n=8), [(1980, 1140), (-40, 1140)]]); inked(fl, gnd, '#a895c0', 5, '#16122a')
        for k in range(16): x = 60 + 120 * k; fl.line(spl([(x, fy + 18), (x + 40, fy + 40 + 20 * h1(k)), (x + 20, fy + 84)], n=5), 2.6, '#4d4280', taper=(.1, .7), seed=k)
        img = over(img, fl.rgba() * 1.0)
        M = np.array([[1 / math.sqrt(sq), 0], [0, sq ** .75]]); pal = G.PAL['night']; lag = np.array([((0. - 6 * k) * (1 if f > 1 else 0), (CT('sq', max(0, f - k)) - sq) * -150.) for k in range(8)])
        e = dict(eye=(-1. if f in (1, 2) else 1.0), mouth='laugh' if jaw > .12 else ('grin' if f == 12 else 'line'), jaw=jaw, chinw=chin, brow=.10 + .16 * min(1, pop), mw=.40 + .05 * min(1, jaw), mc=-.45, mt=-.50, mb=-.93, smile=-.03, gw=.6, gc=-.36, gt=-.55, gb=-.84, iris=.4)
        shc = Cel(); dY = (1 - sq) * 70; nb = np.array([cx - 46, 742 + dY])                  # bust: neck, shirt, shoulders; squashes with the head
        bx = lambda P: np.asarray(P, float) + [cx - 46, 742 + dY]; tor = bx(np.vstack([spl([(-70, -30), (-150, 10), (-262, 60), (-300, 160), (-290, 420)], n=8), spl([(270, 420), (280, 150), (236, 50), (130, 0), (66, -30)], n=8)])); inked(shc, tor, pal['sk'], 5, pal['ink']); shc.clip(tor); shc.fill(tor + [-150, 40], pal['s1']); shc.unclip()
        n0 = np.array([cx - 20, cy + s * .70]); nk, NP_, NN, NW = hose(n0, nb + [0, 10], .04, s * .38, s * .46, n=14); inked(shc, nk, pal['sk'], 4, pal['ink']); shc.fill(bx(spl([(-74, -8), (0, -44), (74, -8), (60, 70), (-60, 70)], True, n=6)), pal['sk']); shc.clip(nk); shc.fill(nk + [-40, -60], pal['s1']); shc.unclip()
        for a_, b_ in (((-60, -6), (-12, 60)), ((58, -6), (14, 60)), ((-130, 40), (-30, 52)), ((130, 40), (34, 52))): shc.line(bx([a_, ((a_[0] + b_[0]) / 2, (a_[1] + b_[1]) / 2 + 6), b_]), 3, pal['ink'], taper=(.3, .4))
        for P in ([(-84, -44), (-64, 70), (-74, 180), (-60, 420), (-310, 420), (-306, 150), (-262, 50), (-150, 0)], [(74, -44), (70, 70), (86, 180), (70, 420), (290, 420), (290, 150), (240, 44), (134, -10)]):
            Q = bx(P); inked(shc, Q, pal['cl'], 5, pal['ink']); shc.clip(Q); shc.fill(Q + [-110, 30], pal['c1']); shc.unclip()
        for g in (-1, 1): shc.line(bx(spl([(g * 190, 60), (g * 160, 130), (g * 176, 220)], n=5)), 3, pal['ink'], taper=(.1, .6))
        bust = shc.layer()
        P = G.head_parts(cx, cy, s, .46, -.02, .03, e, 'night', f, flow=(-1., -.28), fs=.74, lagT=lag, M=M, lw=3.6, size=1.05, seed=2, ldir=(.8, -.4), shade_side=-1, hz=1.3, socket=.3); H = P['H']
        sc = G.scarf(np.array([cx - 60, 735 + dY]), np.array([1., 0]), np.array([0, -1.]), 300, f, lag, 970); spal = dict(pal, h0=pal['c2'], h1=pal['c1'], h2=pal['sc2'], h3=pal['sc3']); sl_ = paint_hair([k for k in sc if k['c'][len(k['c']) // 2][0] < cx - 20] , spal, 190, lw=2.6, light=(.8, -.4), seed=4)
        fig = lay(lay(lay(lay(sl_, P['back']), bust), P['face']), P['front'])
        ov = Cel()                                                                           # telescoping eyes
        if pop > .04:
            for g in (-1, 1):
                c0 = H.P([(g * .350, -.03)])[0]; rr = s * .255; L = np.array([300. + 70 * (g > 0), -46. + 26 * g]) * pop; tip = c0 + L + [0, vib * 1.2]; th = lerpw = 1 + .25 * (pop - 1); poly, C, N, hw = hose(c0, tip, .06 * g * math.sin(f * 1.3), rr * .80, rr * 1.0 * th, n=22); k = len(C); sh = 1 - .24 * np.sin(np.pi * lin01(k)) ; hw = hw * sh; poly = np.vstack([C + N * hw[:, None], (C - N * hw[:, None])[::-1]]); a = np.linspace(0, 6.283, 44); ball = tip + c2(np.cos(a) * rr * 1.12 * th, np.sin(a) * rr * 1.28 * th)
                inked(ov, [poly, ball], '#ffffff', 5, pal['ink']); ov.clip([poly, ball]); ov.fill(np.vstack([C + N * (hw * .25)[:, None], (C + N * (hw * 1.2)[:, None])[::-1]]) if (N[8][1] > 0) else np.vstack([C - N * (hw * .25)[:, None], (C - N * (hw * 1.2)[:, None])[::-1]]), '#b9c0e2'); ov.fill(ball + [-rr * .55, rr * .55], '#c9cfec', .7); ov.fill(tip + [rr * .18, -rr * .1] + c2(np.cos(a) * rr * .98 * th, np.sin(a) * rr * 1.12 * th), '#ffffff'); ov.unclip()
                for q in (.28, .5, .72): i = int(q * k); ov.line(spl([C[i] + N[i] * hw[i], C[i] + (C[min(k - 1, i + 2)] - C[i]) * 1.2, C[i] - N[i] * hw[i]], n=6), 2.4, pal['ink'], taper=(.1, .1))
                for j in range(3): v0 = C[4] + N[4] * hw[4] * (.5 - .5 * j); ov.line(spl([v0, v0 + L * .20 + [0, 10 - 10 * j], v0 + L * .34 + [0, 14 * (j - 1)]], n=5), 2.0, '#d2564f', taper=(.1, .7), seed=j)
                ic = tip + [rr * .34, rr * .02]
                for r_, c_ in ((.56, pal['ink']), (.50, '#d2564f'), (.40, '#f29a88'), (.29, '#8e1d22'), (.22, '#e4685c'), (.13, '#2a0608')): ov.fill(ic + c2(np.cos(a) * rr * r_ * .9, np.sin(a) * rr * r_), c_)
                ov.fill(ic + [-5, -9] + c2(np.cos(a) * 4, np.sin(a) * 4), '#ffffff')
        if tong > .02:                                                                       # tongue unrolls like a carpet
            m0 = H.P([(.0, -.90)])[0]; p0 = np.array([m0[0], min(m0[1] - 20, 985)]); L = 620 * tong; n = 30; v = lin01(n); C = c2(p0[0] + v * L, p0[1] + 26 * np.sin(v * 1.2) + 5 * np.sin(v * 9 - f * 2)); hw = (34 + 6 * v) * (.25 + .75 * ss(0, .16, v)); up = C - c2(0 * v, hw); dn = C + c2(0 * v, hw * .62); T_ = np.vstack([up, dn[::-1]]); inked(ov, T_, '#e0656a', 4.5, pal['ink']); ov.fill(np.vstack([C, dn[::-1]]), '#b23f52'); ov.line(C - [0, 12], 2.4, '#8e2c40', taper=(.1, .3))
            if tong < .98: r = 46 * (1 - tong) + 22; v2 = np.linspace(0, 1, 40); ov_c = C[-1] + [r * .3, -r * .6]; a = np.linspace(0, 6.283, 30); inked(ov, ov_c + c2(np.cos(a), np.sin(a)) * r, '#e0656a', 4.5, pal['ink']); ov.line(ov_c + c2(np.cos(v2 * 12 + f), np.sin(v2 * 12 + f)) * (r * .9 * v2)[:, None], 2.6, pal['ink'])
        if f >= 10:                                                                          # arms shoot out; back with tail and whisker
            w = {10: .18, 11: 1.0, 12: .52}[f]; sA, sB = bx([(-240, 110)])[0], bx([(210, 100)])[0]; hA, hB = np.array([sA[0] - 700 * w, 720 + 60 * w]), np.array([sB[0] + 1150 * w, 640 - 120 * w])
            if f == 12:
                tl = lock(hA + [20, 0], math.pi - .4, 620, 66, f / 24., 33, wave=.4, curl=-1.6, r0=1., tip=.6)[0]; inked(ov, tl['P'], '#2f86b4', 5, '#14111c'); ov.clip(tl['P']); ov.fill(tl['P'] + [0, 46], '#1d4f86'); ov.unclip()
                for k in range(5, 32, 3): ov.line(spl([tl['c'][k] + tl['N'][k] * tl['hw'][k] * .8, tl['c'][k + 1], tl['c'][k] - tl['N'][k] * tl['hw'][k] * .8], n=5), 2.2, '#14111c', taper=(.2, .2))
            for sh_, h_, sd in ((sA, hA, 1), (sB, hB, -1)):
                po, C, N, hw = hose(sh_, h_, .10 * sd, 52, 40, n=24); G.hand(ov, h_, math.atan2(*(h_ - sh_)[::-1]), 300, 'fist' if f == 12 else 'open', sd, pal, 4.5, (.8, -.4)); inked(ov, po, pal['sk'], 4.5, pal['ink']); G.limb_shade(ov, C, N, hw, pal['s1'], (.8, -.4)); j = 9; w2 = np.linspace(74, 70, j); sl2 = np.vstack([C[:j] + N[:j] * w2[:, None], (C[:j] - N[:j] * w2[:, None])[::-1]]); inked(ov, sl2, pal['cl'], 4.5, pal['ink']); G.limb_shade(ov, C[:j], N[:j], w2, pal['c1'], (.8, -.4), .2)
        fig = lay(fig, ov.layer())
        if f >= 11:
            o = KD.head(1790 if f == 11 else 1640, 420, 120, -.75, .05, .1, jaw=.95, light='fire', f=f, ss=1.); img = over(img, o)
            if f == 12: w_ = Cel(); w_.line(spl([(1480, 520), (1380, 560), (1300, 590), (1218, 578)], n=8), 13, '#14111c', taper=(.1, .02)); w_.line(spl([(1480, 520), (1380, 560), (1300, 590), (1218, 578)], n=8), 8, '#f0c468', taper=(.1, .02)); fig = lay(fig, w_.layer())
        fig = hotrim(fig, -1., .25, 5.5, '#fff0c0', '#ff9c46', 1.); fig = rim(fig, 3, 4, '#9fb4ff', .55, soft=1.5); fig = para(fig, (300, 0), (1300, 0), [(0, '#8f8ad0'), (.55, '#ffffff'), (1, '#fff0da')], amt=.85)
        img = screen(img, blur(fig[..., 3], 20)[..., None] * rgb('#ffb070') * .22); img = over(img, fig)
        if f >= 9:
            dc = Cel(); dust_skirt(dc, (f - 9) * 2 + 2, 690, 1004, 360, 30, 40, 3, n=12, pal=PAL_D, life=.9, step=1); img = over(img, dc.rgba())
        sh = {9: 16, 10: -9, 11: 5}.get(f, 0)
        if sh: img = cam2d(img, 960, 540 + sh, 1.02)
        if f in (3, 4): img = smear(img, 26 if f == 3 else 10, 0, 7)
        if f == 11: img = smear(img, 34, 0, 7)
        return img

    # ------------------------------------------------------------------ wide: skipping the dragon
    def wide(self, f):
        hit = min([f - b for b in SLAP if f >= b] or [9]); rb = min([f - b for b in (21, 36) if f >= b] or [9]); dy = 44 * math.exp(-hit / 2.8) * math.cos(hit * 1.15) + 16 * math.exp(-rb / 2.4) * math.cos(rb * 1.2); sq = 1 - dy * .0011; wob = math.exp(-hit / 5.) * math.sin(hit * .95) * 1.3
        thr = f - 49.5; cz = 1 + .03 * float(ease((f - 13) / 30.)); px = 0. if f < 50 else -26 * float(ss(50, 56, f))
        img = cam2d(self.sky, 960 - px * .3, 540 + dy * .12, 1.03)
        br = math.exp(-hit / 4.); mc = Cel(ss=1)                                              # the moon watches
        mo = 1 + .5 * br; a = np.linspace(0, 6.283, 26)
        for x, y, r in ((310, 205, 30), (470, 330, 24), (352, 342, 15), (474, 168, 18), (404, 262, 12), (520, 250, 14), (290, 290, 11), (420, 130, 10)): mc.fill(c2(x + np.cos(a) * r, y + np.sin(a) * r * .85), '#e7bd84', .7)
        img = over(img, mc.rgba(), .9)
        lc = Cel(W // 2, HH // 2, ss=1)                                                       # far paper lanterns already rising
        for i in range(30):
            x = 2000 * h1(i, 31) - 40 + 24 * math.sin(f * .05 + i); y = 1000 - ((h1(i, 32) * 900 + f * (1.1 + 1.6 * h1(i, 33))) % 900); r = 2.2 + 4.2 * h1(i, 34) ** 2; lc.fill(c2(x / 2 + np.array([-1, 1, .8, -.8]) * r * .8, y / 2 + np.array([-1.2, -1.2, 1.2, 1.2]) * r), '#ffcf7a', .95); lc.fill(c2(x / 2 + np.array([-1, 1, .8, -.8]) * r * .5, y / 2 + np.array([.2, .2, 1.1, 1.1]) * r), '#fff3c4')
        lm = lc.rgba(); img = over(img, cv2.resize(cv2.GaussianBlur(lm, (0, 0), .8), (W, HH)), .9); img = img + cv2.resize(cv2.GaussianBlur(lm[..., 3], (0, 0), 9), (W, HH))[..., None] * rgb('#ff9a3c') * .55
        g_ = np.exp(-(np.hypot(self.gx - 960, (self.gy - 560) * 1.1) / 560) ** 2)[..., None]; img = img + g_ * rgb('#ffcf8a') * (.16 + .12 * br)
        c = Cel(); self.skull(f, c, dy, sq, wob); isl = c.layer(); m = (1 + .06 * self.tx)[..., None]; isl[..., :3] *= m; isl = hotrim(isl, .35, 1., 4., '#ffe9c0', '#d79ad8', .95); isl = para(isl, (960, 820), (960, 1080), [(0, '#ffffff'), (1, '#8f86c8')]); isl = para(isl, (0, 0), (1920, 0), [(0, '#8a86cc'), (.5, '#ffffff'), (1, '#8a86cc')], amt=.7); img = over(img, isl)
        gy = GYB + dy
        pc = Cel()                                                                            # debris bouncing on the dome, real arcs
        for i, (k, x, s0, hh, per) in enumerate((('barrel', -560, .70, 120, 1), ('lantern', 480, .62, 150, 1), ('hut', -380, .55, 200, 2), ('barrel', 640, .6, 110, 1), ('torii', 330, .55, 250, 2), ('pine', -690, .55, 110, 1))):
            T = 15 * per; tau = ((f - 15) / T) % 1.; kk = math.floor((f - 15) / T); h = 4 * hh * tau * (1 - tau); c_ = min(tau, 1 - tau) * T; s_ = 1 + .18 * abs(1 - 2 * tau) * float(ss(.5, 3, c_)) - .36 * math.exp(-(c_ / 1.1) ** 2); y0 = GYB + 462 - 462 * math.sqrt(max(0, 1 - (x / 860.) ** 2)) + dy; PROPS[k](pc, Rub(960 + x, y0 - h, s0, (.35 if i % 2 else -.3) * math.sin(math.pi * tau) * (1 if kk % 2 else -1), s_, .1 * math.sin(f)), NP, lw=2.2, seed=i, face=float(ss(.1, .4, tau)))
        pl = hotrim(pc.layer(), .3, 1., 2.4, '#ffe9c0', '#c99ae0', .95); img = over(img, pl)
        # ---- dragon
        D = KD.Dragon(rope(f) * np.array([1, 1, 1.]), R=.60 * (1 + .06 * math.sin(f * 2.1)), n=300, around=30, legs=(.24, .72) if f < 58 else (), mane_step=.5, seed=2, head=.74)
        jaw = .55 + .40 * math.sin(f * 1.1); o = D.render(cam, f, 'day', jaw=jaw, wind=(0, 0, 0), hd=(.35 * math.sin(f * .8), -.1, .3 * math.sin(f * .6)), clouds=False, ss=1.15); L, iz = o['layer'], o['iz']; pz = 1 / 15.
        fr = np.clip((iz - pz) / .0006 + .5, 0, 1) if iz is not None else np.zeros((HH, W), f32); fr = fr[..., None].astype(f32); bk = 46 <= f <= 49
        if bk: fr = fr * 0                                                                     # the whirl: he is always in front
        if f in TR: L = trail(L, *TR[f])
        dr = Cel(); P = o['pts']; far = 1.0 if f < 50 else max(0., 1 - (f - 50) / 4.)
        if far > 0:
            for q, e in enumerate(('eyeL', 'eyeR')):
                if e in P: self.swirl(dr, np.asarray(P[e]) + [0, -2], 20 * far, f, 1 if q else -1)
            hd = np.asarray(P.get('nose', (0, 0)), float)
            for j in range(4): an = f * .45 + j * 1.57; st = np.asarray(P.get('top', hd), float) + [math.cos(an) * 84, -40 + math.sin(an) * 22]; inked(dr, star(st[0], st[1], 15, f * .2 + j), '#ffd85a', 2., '#14111c')
            for j in range(3): q = (f * .12 + j / 3.) % 1; p_ = hd + [60 * q * (1 if j % 2 else -1) + 20, -70 * q + 120 * q * q]; th = lin01(20) * 6.283; inked(dr, p_ + c2(np.sin(th) * 7 * (1 - .5 * np.cos(th)), -np.cos(th) * 11), '#cfe8ff', 1.6, '#14111c')
        ovl = dr.layer(); shd = L * (1 - fr); frt = L * fr
        # ---- him
        hop = HOP(f); w_ = lambda k: WK(k, f); sqf = w_('sq'); A, B = hands(f); ln = w_('lean'); rl = w_('roll'); px_ = w_('px')
        pel = np.array([px_, 1.62 + hop * .98 + .55 * (sqf - 1)]); che = pel + [ln, 1.03 * (.5 + .5 * sqf)]; hd_ = che + [ln * .4 + math.sin(rl) * .25, .98]
        fA, fB = w_('fA') + [px_, 0.], w_('fB') + [px_, 0.]; ex = 'laugh' if (hit <= 4 or f in (49, 50, 51)) else 'joy'; vis = f >= 55
        pose = dict(pel=pel, che=che, hd=hd_, roll=rl * .8, back=(.45 * float(np.clip(-ln, 0, 1)), 0) if f < 50 else (0, 0), sink=float(np.clip(.95 - sqf, 0, .3)) * 2.6, st=-.45 * float(np.clip(-ln, 0, 1)) if f < 50 else .3, ht=.35 * float(np.clip(-ln, 0, 1)), rip=((f - 20.5) / 7. if 20.5 <= f < 27.5 else (f - 35.5) / 7. if 35.5 <= f < 42.5 else None), yaw=.15 * math.sin(f * .3) if f < 46 else .45 * float(ss(48, 51, f)), pitch=.05 + .12 * hop if f < 48 else .30 * float(ss(49, 52, f)), hA=A, hB=B, bA=.30, bB=.30 if f < 50 else .10 + .25 * float(ss(53, 57, f)), fA=fA, fB=fB, kA=.34 + .3 * max(0, 1.5 - pel[1]) + .40 * min(1, hop) * (f < 44), kB=.34 + .3 * max(0, 1.5 - pel[1]) + .40 * min(1, hop) * (f < 44), aA=math.pi - .8 * min(1, fA[1]), aB=.8 * min(1, fB[1]),
                    kA_h='fist' if f < 50 else 'open', kB_h='fist' if f < 50 else ('open' if f < 54 else 'flat'), hdB=2.7 if f >= 55 else None, sq=float(np.clip(.4 + .6 * sqf, .74, 1.15)), expr=ex if f < 53 else 'joy', E=dict(jaw=.10 + .10 * (math.sin(f * 2.1) * .5 + .5)), amp=1.15)
        lagT = []
        for k in range(10):
            g = max(13, f - k); h2 = HOP(g); lagT.append(((WK('px', g) + WK('lean', g) * 1.4 - px_ - ln * 1.4) * U, -(h2 - hop) * U))
        sh = Cel(ss=1); e = np.linspace(0, 6.283, 40); sh.fill(c2(960 + px + pel[0] * U + np.cos(e) * 120 / (1 + .5 * hop), gy + 4 + np.sin(e) * 17 / (1 + .5 * hop)), '#2a1d5a', .6 / (1 + .6 * hop))
        if f < 50: lo = float(ss(1.6, .3, rope(f)[12:16, 2].mean())); sh.fill(c2(960 + np.cos(e) * 330, gy + 8 + np.sin(e) * 22), '#2a1d5a', .55 * lo)
        img = over(img, cv2.GaussianBlur(sh.rgba(), (0, 0), 4))
        img = over(img, shd); img = img + (o['light'] if o.get('light') is not None else 0)
        if bk:                                                                                 # lasso arcs, and his dizzy dressing, behind him
            C0, e1, ex, ey = axes(f); q0 = (f - 49.5) * 1.55 / RING_R; ac = Cel(ss=1)
            for j in range(8): an = q0 - .25 - .75 * (j % 3) - np.linspace(1.5 + .5 * h1(j, 2), 0, 26); R_ = RING_R * (1.26 + .10 * (j % 4)); ac.line(cam.project(C0 + (e1 * np.cos(an)[:, None] + ey * np.sin(an)[:, None]) * R_ + [0, 0, .5 * (h1(j, 3) - .5)])[0], 8 - 1.2 * (j % 4), '#f2fbff' if j % 2 else '#8fd6ea', taper=(.7, .04), seed=j, alpha=.8 * float(ss(45.2, 47, f)))
            img = over(img, ac.rgba()); img = over(img, ovl)
        fig = G.figure(960 + px, gy, U, pose, f, 'night', lagT=np.array(lagT), l=(-.5, -.8), rims=((-.8, .5), (.8, .8)), rimw=2.2, lw=2.7, head_kw=dict(fs=.42, flow=(-.12 * math.sin(f * .3), -1.), ldir=(-.4, -.9), socket=.3))
        fig = para(fig, (0, gy - 560), (0, gy), [(0, '#fff8ea'), (.6, '#ffffff'), (1, '#cfc4ee')]); img = screen(img, blur(fig[..., 3], 24)[..., None] * rgb('#ffe9b0') * .34); img = over(img, fig)
        wc = Cel()                                                                            # whisker in his fist
        if f < 50 and 'nose' in P:
            h = np.array([960 + px + B[0] * U, gy - B[1] * U]); n_ = np.asarray(P['nose'], float); C = spl([n_, (n_ + h) / 2 + [10, 40 + 14 * math.sin(f)], h, h + [-16, 46]], n=10); wc.line(C, 8.5, '#14111c', taper=(.05, .4)); wc.line(C, 5., '#f0c468', taper=(.05, .4))
        img = over(img, frt); img = img if bk else over(img, ovl); img = over(img, wc.rgba())
        # ---- impact dressing
        dc = Cel()
        for b in SLAP:
            if b <= f < b + 14: dust_skirt(dc, (f - b) + 1, 960, gy + 6, 420, 34, 42, b, n=14, pal=PAL_D, life=.75, step=1)
        for b in (21, 36):
            if b <= f < b + 10: dust_skirt(dc, (f - b) + 1, 960, gy + 4, 170, 18, 22, b, n=9, pal=PAL_D, life=.5, step=1)
        for b in (20, 35, 47):                                                                 # skid: the rope's pull drags his heels, grooves and dust kick out behind them
            if b <= f < b + 8:
                for q, fx in ((0, fA[0]), (1, fB[0])):
                    x1 = 960 + px + fx * U; x0 = x1 - (f - b + 1) * 16 * (1 if b != 47 else -1)
                    for j in range(3): dc.line(spl([(x0 - 24 * j * (1 if b != 47 else -1), gy + 5 + 6 * j), ((x0 + x1) / 2, gy + 9 + 6 * j), (x1, gy + 5 + 6 * j)], n=5), 3.2 - .6 * j, '#f0e2f4' if j % 2 else '#4d4280', taper=(.5, .1), seed=j)
                puffs(dc, [(960 + px + fA[0] * U - (14 + 15 * j) * (1 if b != 47 else -1), gy - 2 - 7 * j - 3 * (f - b), max(3., 20 - 4 * j - 1.5 * (f - b)), b * 7 + j, f * .3, .0) for j in range(3)], PAL_D, lw=1.8, seed=b)
        if f >= 50:
            smoke_burst(dc, f - 50 + 1, 960 + px, gy, 74, 5, n=8, ang=(-math.pi * .98, -.02), speed=520, life=1.1, pal=PAL_D, step=1, squash=.6, lw=2.2)
            for j in range(13):
                t = (f - 50 + 1) / 24.; an = -math.pi * (.06 + .88 * j / 12.); v = 620 + 420 * h1(j, 3); x, y = 960 + px + math.cos(an) * v * t, gy - 40 + math.sin(an) * v * t * 1.1 + 900 * t * t; r = (18 + 20 * h1(j, 4)) * min(1, t * 12) * (1 - .4 * t)
                if y < gy + 40: inked(dc, star(x, y, r, f * .25 * (1 if j % 2 else -1) + j), '#ffd85a' if j % 3 else '#fff6d0', 2.6, '#16122a')
        img = over(img, dc.rgba())
        if hit < 3 and f < 50:
            k = Cel(ss=1)
            for j in range(16):
                an = -math.pi * (j + .5) / 16; r0 = 250 + 40 * hit; r1 = r0 + 90 + 60 * h1(j, f); k.line([(960 + math.cos(an) * r0 * 1.5, gy + 6 + math.sin(an) * r0 * .5), (960 + math.cos(an) * r1 * 1.5, gy + 6 + math.sin(an) * r1 * .5)], 6, '#fff2d0', taper=(.1, .8), alpha=.9)
            img = over(img, k.rgba())
        if f >= 56:                                                                           # ting
            p_ = cam.project(rope(f)[:1])[0][0]; kk = [.5, 1., .8, .45][f - 56]; st = Cel(ss=1); st.fill(star(p_[0], p_[1], 84 * kk, .0, .12, 4), '#fffbe0'); st.fill(star(p_[0], p_[1], 44 * kk, .785, .14, 4), '#fffbe0'); m = st.rgba(); img = over(img, m); img = img + cv2.GaussianBlur(m[..., 3], (0, 0), 16)[..., None] * rgb('#ffe9a0') * .8
        z = cz * (1 + .02 * math.exp(-hit / 2.)) ; img = cam2d(img, 960 + 4 * math.sin(hit * 2.4) * br, 560 - dy * .25, z)
        if f in (13,): img = zoomblur(img, 960, 560, .12, 9)
        if f in (50,): img = smear(img, 12, -7, 5)
        return img

    def frame(self, f):
        return (self.take(f) if f < 13 else self.wide(f)).astype(f32)
