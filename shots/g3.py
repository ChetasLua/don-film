"""g3  SMILE FRAME again: the numbers of a2.  He turns into it."""
import numpy as np, cv2, math
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp, repaint, mix
from op.comp import over, screen, blur, lerp_keys, radial, cam2d
from op.ink import Cel, spl
from op.kit import rim
from op.fx_air import motes
from op import fx_core as FC
from chars import luffy as L
from shots.d8 import cached

SUN = (-40., 330.)


def plate():
    BW, BH = 2200, 1240; x, y = np.meshgrid(np.arange(BW, dtype=f32), np.arange(BH, dtype=f32)); n = fbm(BH, BW, (150, 60, 24), seed=4, aspect=2.6); n2 = fbm(BH, BW, (60, 22, 9), seed=6, aspect=6); sx, sy = 100., 410.; sd = np.hypot((x - sx) / 1.5, y - sy); hz = 800
    g = ramp(np.clip(y / hz, 0, 1), [(0, '#f3e3bd'), (.35, '#f8dca6'), (.7, '#f7bf85'), (.95, '#f2a06e'), (1, '#f6c08a')]); g = g * (1 - .16 * smooth(0, 1, x / BW))[..., None] + (.16 * smooth(0, 1, x / BW))[..., None] * rgb('#c98a98')
    cl = smooth(.15, .7, n + .5 * n2 - .1) * (y < hz - 10); e = np.clip(np.roll(n, (10, 26), (0, 1)) - n + .5, 0, 1); g = mix(g, mix(rgb('#d99a86') * np.ones_like(g), rgb('#fff3cf') * np.ones_like(g), e[..., None]), (cl * .75)[..., None])
    s = ramp(np.clip((y - hz) / (BH - hz), 0, 1), [(0, '#ffe2a8'), (.08, '#e8a577'), (.4, '#a8707c'), (1, '#6a4a6c')]); gl = (n2 > .25) * np.exp(-((x - sx) / (420 + 1.6 * (y - hz))) ** 2) * (y > hz); s = s + gl[..., None] * rgb('#fff0c0') * .6
    g = np.where((y < hz)[..., None], g, s); g = screen(np.clip(g, 0, 1.2), np.exp(-(sd / 420) ** 1.25)[..., None] * rgb('#ffd98e') * .8); g = repaint(g.astype(f32), (40, 18, 9), seed=2)
    c = Cel(BW, BH, ss=1); c.fill(spl([(1500, 810), (1620, 720), (1800, 690), (2000, 740), (2240, 700), (2240, 830)], False), '#a87880'); q = c.rgba(); g = g * (1 - q[..., 3:4]) + q[..., :3]
    return blur(g, 8).astype(f32)


def crew():
    """friends behind him, soft focus: hair, hats and gold edges"""
    c = Cel(W + 240, H + 80, -120, 0, ss=1); r = np.random.default_rng(8)
    def blob(P, k): c.fill(spl(P, True, n=6), k)
    def hd(x, y, a, b, skin, sh): blob([(x - a, y - b * .2), (x - a * .7, y - b * .9), (x, y - b), (x + a * .7, y - b * .9), (x + a, y - b * .2), (x + a * .85, y + b * .5), (x, y + b), (x - a * .85, y + b * .5)], skin); blob([(x + a * .15, y - b * .9), (x + a, y - b * .3), (x + a * .9, y + b * .5), (x + a * .1, y + b), (x + a * .35, y)], sh)
    def spikes(x, y, ra, rb, n, k, a0=-3.3, a1=.2, amp=.28):
        t = np.linspace(a0, a1, 2 * n + 1); q = np.array([1 + (amp if i % 2 else 0) * r.uniform(.7, 1.2) for i in range(len(t))]); c.fill(np.vstack([np.c_[x + ra * np.cos(t) * q, y + rb * np.sin(t) * q], [(x + ra * .9, y + rb * .3), (x - ra * .9, y + rb * .3)]]), k)
    # swordsman: green crop, long coat, three hilts
    blob([(1270, 1180), (1300, 960), (1390, 850), (1470, 800), (1540, 800), (1630, 850), (1710, 960), (1740, 1180)], '#4c5a4a'); blob([(1440, 1180), (1455, 860), (1500, 830), (1550, 860), (1560, 1180)], '#b07a58'); blob([(1462, 830), (1466, 720), (1540, 720), (1545, 830)], '#9a664c'); hd(1503, 660, 74, 96, '#c08a64', '#8f5c4a'); spikes(1503, 640, 78, 84, 9, '#5d7a4c', -3.35, .20)
    for i in range(3): c.line(np.array([(1300 - 26 * i, 1120), (1250 - 30 * i, 960 + 18 * i)]), 20, ['#e8e0d0', '#7a3c34', '#3c3046'][i], taper=(.05, .05)); c.line(np.array([(1268 - 28 * i, 1030 + 10 * i), (1300 - 28 * i, 1012 + 10 * i)]), 12, '#d8b060', taper=(.1, .1))
    # musician: the great afro, a thin neck, boa
    blob([(1620, 1180), (1660, 900), (1730, 800), (1790, 770), (1860, 790), (1950, 880), (2060, 1180)], '#4a3644'); blob([(1768, 800), (1772, 660), (1812, 660), (1818, 800)], '#e2d2b8'); hd(1792, 560, 72, 104, '#eadcc2', '#b89c8a'); t = np.linspace(0, 6.283, 40, endpoint=False); c.fill(np.c_[1795 + 178 * np.cos(t) * (1 + .05 * np.sin(t * 11)), 420 + 168 * np.sin(t) * (1 + .05 * np.cos(t * 9))], '#43313e'); blob([(1700, 300), (1716, 170), (1800, 150), (1880, 176), (1892, 300)], '#3c2c3a'); blob([(1640, 318), (1800, 270), (1950, 318), (1800, 350)], '#3c2c3a')
    blob([(1690, 800), (1740, 760), (1800, 786), (1860, 756), (1910, 800), (1870, 850), (1800, 826), (1730, 852)], '#c9707c')
    # navigator: long hair in the wind
    blob([(150, 1180), (180, 1000), (270, 900), (330, 880), (400, 900), (470, 980), (500, 1180)], '#b7845f'); blob([(322, 900), (326, 790), (372, 790), (378, 900)], '#a8724f'); hd(346, 725, 62, 84, '#cf9a70', '#a06a50'); blob([(276, 730), (284, 640), (350, 604), (416, 636), (440, 730), (470, 850), (560, 960), (640, 1010), (560, 1040), (470, 1010), (420, 920), (404, 760), (350, 690), (300, 740), (288, 880), (240, 1010), (214, 1000), (262, 860)], '#c8713c'); blob([(424, 740), (462, 850), (560, 970), (470, 960), (420, 860)], '#9c5230')
    # sniper: curls, goggles, the nose
    blob([(-140, 1180), (-130, 860), (-30, 740), (70, 700), (170, 760), (250, 900), (270, 1180)], '#7d6a3e'); blob([(40, 740), (44, 620), (104, 620), (110, 740)], '#8d5a40'); hd(74, 545, 70, 92, '#a9704e', '#7c4c3c'); blob([(120, 540), (214, 548), (222, 562), (128, 574)], '#a9704e')
    for i in range(22): a = -3.5 + 3.9 * i / 21; blob([(70 + 108 * math.cos(a) + 44 * math.cos(q), 505 + 104 * math.sin(a) + 40 * math.sin(q)) for q in (0, 1.26, 2.5, 3.77, 5.0)], '#43303a')
    blob([(6, 470), (70, 440), (140, 470), (136, 506), (70, 484), (8, 506)], '#7a6a4a')
    # doctor: a hat, two antlers
    blob([(1180, 1180), (1190, 1010), (1250, 960), (1320, 960), (1370, 1010), (1380, 1180)], '#8d5a44'); blob([(1168, 968), (1280, 930), (1392, 968), (1380, 1000), (1280, 978), (1180, 1000)], '#a8444e'); blob([(1200, 960), (1206, 850), (1280, 820), (1354, 850), (1360, 960)], '#c4606a')
    for sd in (-1, 1): c.line(spl([(1280 + sd * 74, 920), (1280 + sd * 130, 870), (1280 + sd * 150, 790)], n=6), 20, '#6a4a3a', taper=(.05, .4)); c.line(spl([(1280 + sd * 128, 872), (1280 + sd * 178, 850), (1280 + sd * 196, 806)], n=6), 15, '#6a4a3a', taper=(.05, .5))
    for a in (.785, 2.356): d = np.array([math.cos(a), math.sin(a)]) * 34; c.line(np.array([(1280, 892) - d, (1280, 892) + d]), 15, '#f0e4d4', taper=(0, 0))
    q = c.rgba(); q = rim(q, 9, 5, '#ffeab4', .95, 1.6); return L.rimlight(q, np.array([-.9, -.45]), 20, '#ffd890', .35)


class Shot:
    id = 'g3'; n = 60; finish = dict(bloom=.36, vig=.12)
    EVENTS = [(0, 'wind', dict(size=.3, dur=60)), (2, 'turn', dict(size=.3, note='head and hair swing round to us')), (16, 'settle', dict(size=.2)), (20, 'grin_opens', dict(size=.5, dur=20)), (38, 'grin_full', dict(size=.7, note='same teeth as Roger in a2')), (45, 'eye_shine', dict(size=.5, note='small bright glint')), (50, 'chuckle', dict(size=.4)), (55, 'chuckle', dict(size=.3))]

    def setup(self): self.bg = cached('g3_bg', plate); self.crew = cv2.GaussianBlur(crew(), (0, 0), 7.5)

    def frame(self, f):
        g = f if f < 18 or 34 <= f <= 43 else f // 2 * 2; K = lambda k: float(lerp_keys(g, k)); S = L.SMILE
        yaw = [-.98, -.99, -1.0, -.985, -.93, -.84, -.73, -.60, -.465, -.33, -.20, -.08, .03, .12, .195, .25, .285, .30, .30, .29, .28][min(g, 20)]; pitch = K([(0, .02), (3, -.06), (10, -.10), (16, .10), (20, .12), (30, .12), (33, .07), (38, .205), (42, .18)]); roll = K([(0, .03), (5, .065), (11, .01), (16, -.095), (19, -.07), (30, -.06), (33, -.03), (38, -.098), (42, -.08)])
        s = S['s'] * float(lerp_keys(f, [(0, .90), (16, .95), (40, 1.0), (59, 1.0)]))
        ch = sum(a * math.exp(-(g - t) / 2.4) * math.cos((g - t) * 1.1) for t, a in ((50, 11), (55, 8)) if g >= t)
        Hd = L.LHead(**L.GROWN).pose(0, 0, s, yaw, pitch, roll); wp = 1. if g >= 20 else float(smooth(.62, 0., abs(yaw - .28))); p = Hd.P(np.array([(0., S['mouth_y'])]))[0]; p = p if wp == 1. else p * wp + Hd.P3(np.array([(0., S['mouth_y'], .62)]))[0] * (1 - wp)
        tx = S['mouth'][0] + K([(0, 10), (10, 28), (16, -8), (20, 0)]); ty = S['mouth'][1] + K([(0, 24), (8, 40), (16, -6), (20, 6), (33, 18), (38, -6), (42, 0)]) - ch; p0 = L.LHead(**L.GROWN).pose(0, 0, s, .28, pitch, roll).P(np.array([(0., S['mouth_y'])]))[0]
        u = smooth(4, 16, g); q = p * u + p0 * 0 * (1 - u) + (1 - u) * (p * .55 + p0 * .45); cx, cy = tx - q[0], ty - q[1]
        e = ('calm', 'smile', K([(0, .25), (12, .25), (18, .8), (20, 1.)])) if g < 22 else ('smile', 'grin', K([(20, 0), (24, .14), (28, .36), (33, .62), (36, .92), (38, 1.10), (41, .97), (43, 1.)]))
        kx = dict(mw=None); kx = {}
        if 8 <= g <= 14: kx.update(lu=[.72, .40, .10, .06, .10, .38, .74][g - 8], arc=-.35)
        kx['hl'] = K([(0, .8), (30, .9), (40, 1.5), (59, 1.7)]); kx['st'] = K([(0, 0), (20, 0), (33, -.02), (37, .03), (41, 0)]); kx['sh'] = 0.
        Lt = dict(L.LIGHT['dawn'], cover=float(lerp_keys(g, [(0, .40), (30, .34), (40, .22), (59, .18)])), rw=1.25 + .25 * smooth(36, 44, f))
        wind = 1.3 + .5 * math.sin(f * .13); bl = -.50 + .66 * (g < 36) * math.exp(-((g - 12) / (4. if g < 12 else 5.)) ** 2) + .10 * math.sin(f * .31) + .06 * math.sin(f * .53 + 1)
        R, Hd, E, _ = L.record(cx, cy, s, yaw, pitch, roll, e, Lt, f, hat=False, body=True, shirt='red', scar=True, kx=kx, wind=wind, blow=bl, drift=(.08 + .05 * math.sin(f * .29) + .24 * math.exp(-((g - 11) / 4.) ** 2) * -1 + .24 * math.exp(-((g - 18) / 2.8) ** 2), .02 * math.sin(f * .37)), body_dy=.004 * ch, bturn=(-.4) * (1 - u))
        img = cam2d(self.bg, 1100 - 30 + f * .7, 620 - f * .25, 1.0 + .0008 * f, 0)
        r = np.random.default_rng(4); c = Cel(W, H, ss=1)
        for j in range(70): x = r.uniform(-50, 1100) ** 1.0; y = 715 + r.random() ** 1.5 * 330; k = max(0, math.sin(f * (.2 + .3 * r.random()) + r.uniform(0, 6.3))) ** 3; w_ = (8 + (y - 700) * .06) * k; c.fill(np.array([(x - w_, y), (x, y - 2 - 3 * k), (x + w_, y), (x, y + 2 + 3 * k)]), '#fff6d8', .8) if k > .1 else None
        img = screen(img, cv2.GaussianBlur(c.rgba()[..., :3], (0, 0), 5) * 1.1)
        sd = np.hypot(grid()[0] - (905 + f * .25), grid()[1] - 474); disc = cv2.GaussianBlur(smooth(452, 440, sd), (0, 0), 11); img = img * (1 - disc[..., None] * .86) + ramp(np.clip((grid()[1] - 60) / 860, 0, 1), [(0, '#f6613a'), (.6, '#ee4a33'), (1, '#f58a5c')]) * disc[..., None] * .86; img = screen(img, np.exp(-np.clip(sd - 440, 0, None) / 150)[..., None] * rgb('#ffb070') * .34)
        cr = cv2.warpAffine(self.crew, np.float32([[1, 0, -94 - f * 1.0], [0, 1, 8 + 5 * math.sin(f * .1) - ch * .3]]), (W, H), borderMode=cv2.BORDER_REPLICATE); cr = cr.copy(); cr[..., :3] = cr[..., :3] * .64 + rgb('#f6cda2') * .36 * cr[..., 3:4]; img = over(img, cr, .95)
        img = screen(img, radial(SUN[0], SUN[1], 900, 760, power=1.8)[..., None] * rgb('#ffd890') * (.55 + .05 * math.sin(f * .2)))
        lay = R.cel(Lt, s); img = over(img, lay)
        # light wrapping round him from the left + the shine in the eyes
        al = lay[..., 3]; wr = cv2.GaussianBlur(cv2.resize(1 - al, (W // 4, H // 4)), (0, 0), 9); wr = cv2.resize(wr, (W, H)) * al * np.clip(1 - grid()[0] / 1250, 0, 1); img = screen(img, wr[..., None] * rgb('#ffdf9a') * .55)
        if f >= 42:
            k = math.exp(-((f - 46) / 2.6) ** 2); c = Cel()
            for sd in (-1, 1):
                pt = Hd.P(np.array([(sd * .335 - .018, -.02)]))[0]; n = 34 + 70 * k
                for (ax, ay) in ((1, 0), (0, 1)): c.fill(np.array([pt + [ax * n - ay * 2.6, ay * n - ax * 2.6], pt + [ax * 3, ay * 3], pt + [ax * 0 + ay * 3.2, ay * 0 + ax * 3.2], pt - [ax * n, ay * n], pt - [ay * 3.2, ax * 3.2]]), '#fffdf0', .95 * max(k, .0)) if k > .08 else None
                c.line(Hd.P(np.array([(sd * .335 - .10, -.175), (sd * .335, -.186), (sd * .335 + .09, -.172)])), 4.5, '#fffdf2', taper=(.3, .3), alpha=float(smooth(42, 47, f)) * .9)
            img = over(img, c.layer())
        img = screen(img, np.clip(motes(f, 60, seed=3), 0, 1) * .8)
        fl = float(lerp_keys(f, [(0, 0), (44, 0), (54, .20), (59, .52)])); img = screen(img, np.clip(radial(SUN[0], SUN[1], 1500 + 900 * fl, 1200 + 700 * fl, power=1.5) * fl * 1.6, 0, 1)[..., None] * rgb('#fff0cc')); img = img * (1 - .22 * fl) + rgb('#f6e6c4') * .22 * fl
        return img.astype(f32)
