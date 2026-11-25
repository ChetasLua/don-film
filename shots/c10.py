"""c10: the crew's dinner as a fresco, then dinner with this crew.  Ten people, ten gags, one freeze that becomes a fresco again."""
import numpy as np, cv2, math, os
from op import W, H, kit, comp, clouds
from op.paint import rgb, f32, smooth, mix, ramp, grid, fbm, repaint, Canvas, arcs
from op.ink import Cel, spl, resample, over_roi
from op.comp import over
from op.kit import s1
from chars import chibi as CB
from chars.chibi import Draw, hand, arm, LN, SK

TT = 648                                                              # far edge of the table top
SEAT = dict(franky=(168, 398, 80), robin=(345, 436, 74), usopp=(520, 446, 76), chopper=(692, 530, 62), luffy=(960, 432, 84), nami=(1196, 450, 74), sanji=(1368, 424, 76), jinbe=(1526, 398, 82), brook=(1676, 430, 62), zoro=(1800, 462, 76))
SLV = dict(franky=('#eab584', '#b57a58'), robin=('#7a4fa8', '#482a6e'), usopp=SK['usopp'], chopper=SK['chopper'], luffy=('#c3272f', '#7d141f'), nami=('#f4f2ee', '#b9b6cc'), sanji=('#23212c', '#100f18'), jinbe=('#d9592c', '#7c2440'), brook=('#2c2340', '#15102a'), zoro=('#2f5a46', '#173327'))
AW_ = 1.12
AW = dict(franky=(54, 62), robin=(30, 24), usopp=(30, 26), chopper=(34, 30), luffy=(36, 28), nami=(30, 24), sanji=(34, 30), jinbe=(58, 46), brook=(26, 24), zoro=(44, 40))
HR = dict(franky=32, robin=19, usopp=21, chopper=20, luffy=23, nami=19, sanji=21, jinbe=29, brook=18, zoro=25)
U = -math.pi / 2
# key poses: (frame, changes).  L, R = hand (x, y, angle, kind); e expression, t turn, r roll, x, y offsets, s squash
SEAT = {n: (x, y - 6, s * 1.17) for n, (x, y, s) in SEAT.items()}; HR_ = 1.12
K = dict(
    franky=[(0, dict(e='smile', t=.5, r=-.06, L=(58, 520, U - .4, 'open'), R=(282, 530, U + .4, 'open'))), (26, dict(e='shock', t=.6, y=-10)), (30, dict(e='cry', t=.2, r=.10, y=14, droop=1., L=(110, 640, .4, 'cup'), R=(236, 640, math.pi - .4, 'cup'))), (44, dict(e='mad', t=.7, r=-.12, y=-16, droop=.55, L=(60, 470, U, 'fist'), R=(300, 470, U, 'fist')))],
    robin=[(0, dict(e='holy', t=.35, r=.10, L=(338, 596, -.6, 'flat'), R=(424, 652, .2, 'flat'))), (24, dict(e='calm', t=.4)), (27, dict(e='sip', t=.15, r=-.04, L=(330, 560, -.5, 'cup'), R=(392, 586, math.pi + .5, 'cup'))), (46, dict(e='laugh', t=.5, r=.08, L=(352, 520, U + .4, 'flat')))],
    usopp=[(0, dict(e='calm', t=.45, r=-.05, L=(448, 654, .3, 'flat'), R=(598, 478, U, 'point'))), (23, dict(e='shock', t=.7, y=-8, R=(604, 560, U, 'open'))), (27, dict(e='sly', t=.8, r=.08, y=4, R=(606, 636, 0, 'cup'))), (30, dict(R=(640, 620, -.1, 'grab'))), (36, dict(e='whistle', t=-.7, r=-.10, R=(560, 640, U, 'flat'))), (43, dict(e='laugh', t=.6, r=.16, y=6, R=(590, 560, U + .3, 'point')))],
    chopper=[(0, dict(e='holy', t=.4, r=.08, L=(660, 640, U + .5, 'flat'), R=(722, 640, U - .5, 'flat'))), (22, dict(e='shock', t=.5, y=-18, L=(640, 560, U, 'open'), R=(746, 560, U, 'open'))), (26, dict(e='cry', y=0, r=-.1)), (32, dict(e='cross', ball=1., y=-24)), (36, dict(ball=1., y=0)), (52, dict(e='bliss', ball=1.))],
    luffy=[(0, dict(e='holy', t=0, r=0, L=(826, 668, math.pi - .3, 'flat'), R=(1094, 668, .3, 'flat'))), (12, dict(e='dull', look=(.0, 1.))), (15, dict(e='stare', look=(-.9, .3), t=-.35)), (17, dict(look=(.9, .3), t=.35)), (19, dict(e='grin', t=0, look=(0, 0), s=-.14, y=20)), (20, dict(e='grin', s=.12, y=-8)), (22, dict(s=0, y=0)), (34, dict(e='chew', fat=.45)), (36, dict(e='stuffed', fat=.7)), (38, dict(fat=.62, s=-.05)), (40, dict(fat=.85, body=.14, s=.04)), (42, dict(e='fire', fat=.2, red=.75, s=.14, y=-12)), (47, dict(e='stuffed', fat=.8, red=.0, body=.34, s=-.08, y=0)), (49, dict(body=.55, s=.06)), (51, dict(body=.74, s=-.04)), (53, dict(body=1., s=.05, y=6)), (55, dict(e='bliss', fat=.55, s=0))],
    nami=[(0, dict(e='calm', t=-.4, r=.14, x=10, L=(1180, 596, U, 'flat'), R=(1270, 654, .3, 'flat'))), (22, dict(e='blank', t=-.2)), (25, dict(e='stare', look=(0, 1.), r=.06, x=0)), (28, dict(e='mad', t=-.6, r=-.06, y=-12, R=(1262, 400, U - .3, 'fist'), L=(1120, 590, math.pi, 'fist'))), (32, dict(x=-60, r=-.22, R=(1078, 330, 2.6, 'fist'))), (33, dict(x=-84, R=(1000, 296, 2.3, 'fist'), s=-.08)), (37, dict(x=-10, r=.0, s=0, e='cross', R=(1262, 590, U, 'fist'))), (41, dict(e='cross', t=-.3, L=(1160, 612, .3, 'fist'), R=(1238, 612, math.pi - .3, 'fist')))],
    sanji=[(0, dict(e='calm', t=-.5, r=-.08, L=(1272, 560, math.pi + .2, 'open'), R=(1464, 548, -.2, 'open'))), (20, dict(e='smile', t=-.8, look=(-.6, 0))), (23, dict(e='heart', t=-.7, r=.12, y=-12, L=(1290, 560, math.pi + .5, 'cup'))), (29, dict(e='mad', t=-.7, r=-.16, y=-20, leg=.4)), (31, dict(leg=1., y=0, r=-.06, s=-.08)), (38, dict(leg=0., e='heart', s=0, t=-.8, r=.16, L=(1282, 590, math.pi, 'cup'))), (42, dict(e='stare', t=-.6, r=.04, look=(-.6, .3))), (45, dict(e='smile', r=0)), (50, dict(e='cross', t=-.5, r=0))],
    jinbe=[(0, dict(e='calm', t=-.35, r=.04, L=(1472, 652, math.pi, 'flat'), R=(1612, 652, 0, 'flat'))), (22, dict(L=(1486, 560, U + .3, 'cup'), R=(1600, 560, U - .3, 'cup'), y=-8)), (27, dict(e='sip', t=-.1, L=(1502, 520, U + .5, 'cup'), R=(1584, 520, U - .5, 'cup'))), (46, dict(e='blank', L=(1486, 600, U + .3, 'cup'), R=(1600, 600, U - .3, 'cup'))), (50, dict(e='sweat', t=-.4))],
    brook=[(0, dict(e='calm', t=-.5, r=.10, L=(1650, 600, math.pi + .6, 'flat'), R=(1752, 520, U, 'point'))), (25, dict(e='shock', jaw=1., hat=1., y=-22, s=.12, L=(1640, 500, U - .4, 'open'), R=(1772, 500, U + .4, 'open'))), (28, dict(s=0, y=-8, hat=.2)), (34, dict(hat=0., jaw=1., y=0, r=-.14)), (45, dict(e='bliss', jaw=.5, r=.16, L=(1650, 590, math.pi, 'flat'), R=(1764, 560, U, 'open')))],
    zoro=[(0, dict(e='drink', t=-.55, r=.16, L=(1776, 520, U + .7, 'grab'), R=(1880, 610, U, 'fist'))), (27, dict(y=-5, r=.20, L=(1778, 498, U + .7, 'grab'))), (33, dict(y=0, r=.16, L=(1776, 520, U + .7, 'grab')))])
PLT = dict(franky=(196, 'cola', 26), robin=(345, 'sandwich', 24), usopp=(510, 'fish', 23), chopper=(688, 'cotton', 22), nami=(1208, 'cake', 22), sanji=(1372, 'steak', 23), jinbe=(1540, 'bowl', 99), brook=(1690, 'curry', 25), zoro=(1822, 'onigiri', 99))
PL = np.array([(902, 596), (846, 700), (760, 640), (680, 704), (596, 640), (505, 704), (410, 640), (320, 704), (232, 648), (196, 640)], float); PR = np.array([(1018, 596), (1080, 704), (1160, 642), (1240, 706), (1320, 640), (1420, 706), (1520, 640), (1610, 704), (1700, 650), (1770, 566)], float)


def food(c, kind, x, y, s, sd=0):
    q = lambda P: np.asarray(P, float) * s + [x, y]; o = lambda P, col, w=3.2: (c.fill(spl(q(P), True, n=6), col), kit.outline(c, spl(q(P), True, n=6), w, LN, sd + len(P), press=.25))
    el = lambda cx, cy, rx, ry, col, w=3.: (c.fill(q(np.c_[cx + np.cos(_t) * rx, cy + np.sin(_t) * ry]), col), c.line(q(np.vstack([np.c_[cx + np.cos(_t) * rx, cy + np.sin(_t) * ry]])), w, LN, taper=(0, 0)) if w else None)
    if kind == 'meat': o([(-1.5, -.1), (-1.7, -.4), (-1.4, -.5), (-1.1, -.2), (1.1, -.2), (1.4, -.5), (1.7, -.4), (1.5, -.1), (1.7, .2), (1.4, .3), (1.1, .05), (-1.1, .05), (-1.4, .3), (-1.7, .2)], '#fbf3df'); o([(-.95, -.1), (-.8, -.75), (0, -.95), (.8, -.75), (.95, -.1), (.8, .55), (0, .75), (-.8, .55)], '#a8552f'); c.fill(spl(q([(-.6, -.45), (0, -.75), (.4, -.6), (0, -.35)]), True, n=5), '#e0915a')
    elif kind == 'cola': o([(-.28, .7), (-.30, -.3), (-.12, -.8), (-.12, -1.2), (.12, -1.2), (.12, -.8), (.30, -.3), (.28, .7)], '#5a1a24'); c.fill(q([(-.28, 0), (.28, 0), (.28, .3), (-.28, .3)]), '#e8384f')
    elif kind == 'sandwich': o([(-.8, .3), (0, -.7), (.8, .3)], '#f6e2b4'); c.line(q([(-.6, .18), (.6, .18)]), 4, '#6fae4e')
    elif kind == 'fish': o([(-1., 0), (-.5, -.4), (.4, -.35), (.8, 0), (1.2, -.35), (1.1, .35), (.8, .1), (.4, .35), (-.5, .35)], '#8fb4d0'); el(-.6, -.05, .08, .08, LN, 0)
    elif kind == 'cotton': [el(a, b, .5, .45, '#ffc4da', 2.4) for a, b in ((-.35, -.6), (.35, -.6), (0, -.95), (0, -.5))]; c.line(q([(0, -.1), (0, .6)]), 4, '#e8d6b0', taper=(0, 0))
    elif kind == 'cake': o([(-.8, .4), (-.8, -.2), (.1, -.6), (.8, -.2), (.8, .4)], '#fdf3e0'); c.fill(q([(-.8, .0), (.8, .0), (.8, .15), (-.8, .15)]), '#f2873a'); el(.1, -.62, .22, .20, '#f2873a')
    elif kind == 'steak': o([(-.9, .1), (-.6, -.4), (.2, -.45), (.9, -.1), (.7, .35), (-.4, .4)], '#8a4a2c'); c.line(q([(-.4, -.1), (.3, .1)]), 3, '#4a2418')
    elif kind == 'curry': o([(-.9, .1), (-.4, -.4), (.5, -.35), (.9, .1), (.5, .35), (-.5, .35)], '#d89a3a'); o([(-.8, .1), (-.5, -.35), (0, -.1), (-.2, .3)], '#fffaf0', 2.4)
    elif kind == 'onigiri': [(o([(dx - .55, .35), (dx, -.55), (dx + .55, .35)], '#fffaf2', 2.8), c.fill(q([(dx - .2, .05), (dx + .2, .05), (dx + .2, .40), (dx - .2, .40)]), '#2a3040')) for dx in (-.5, .5)]
    elif kind == 'bowl': o([(-.9, -.4), (.9, -.4), (.6, .5), (-.6, .5)], '#b8342e'); el(0, -.42, .88, .30, '#fffaf2')
    elif kind == 'tabasco': o([(-.3, .9), (-.32, -.3), (-.14, -.7), (.14, -.7), (.32, -.3), (.3, .9)], '#e02a1e'); c.fill(q([(-.16, -1.1), (.16, -1.1), (.16, -.66), (-.16, -.66)]), '#3a8a3a'); o([(0, -.15), (.2, .3), (0, .5), (-.2, .3)], '#ffd23a', 2.); c.line(q([(0, .0), (0, .4)]), 2.5, LN)


_t = np.linspace(0, 2 * math.pi, 26)


def plate(c, x, y, r=56): P = lambda k, dy=0: np.c_[x + np.cos(_t) * r * k, y + dy + np.sin(_t) * r * k * .26]; c.fill(P(1.04, 4), '#8d92b4'); c.fill(P(1), '#fffdf6'); c.line(P(1), 3.2, LN, taper=(0, 0)); c.fill(P(.66, 1), '#dfe3ee')


def star(c, x, y, R, sd=0, cols=('#fffdf0', '#ffd23a')):
    r = np.random.default_rng(sd); n = 11
    for k, col in enumerate(cols[::-1]): a = np.linspace(0, 2 * math.pi, 2 * n, endpoint=False) + .2; rr = np.where(np.arange(2 * n) % 2 == 0, R * (1 - .34 * k) * r.uniform(.8, 1.2, 2 * n), R * .46 * (1 - .34 * k)); Q = np.c_[x + np.cos(a) * rr, y + np.sin(a) * rr * .85]; c.fill(Q, col); (kit.outline(c, Q, 3.6, LN, sd) if k == 0 else None)
    for k in range(9): a = k * .7 + sd; c.line([(x + math.cos(a) * R * 1.15, y + math.sin(a) * R), (x + math.cos(a) * R * 1.6, y + math.sin(a) * R * 1.35)], 4, LN, taper=(.1, .7), seed=k)


def flame(c, x, y, L, w, f, ang):
    e = np.array([math.cos(ang), math.sin(ang)]); n = np.array([-e[1], e[0]]); r = np.random.default_rng(f // 1)
    for k, (col, s) in enumerate((('#d8341c', 1.), ('#ff8a1e', .78), ('#ffd84a', .54), ('#fffbe0', .30))):
        P = [np.array([x, y]) - n * w * .12]
        for i in range(1, 6): u = i / 6.; P.append(np.array([x, y]) + e * L * s * u - n * w * s * (.3 + .7 * math.sin(u * 2.6)) * r.uniform(.8, 1.2))
        P.append(np.array([x, y]) + e * L * s * r.uniform(.92, 1.1))
        for i in range(5, 0, -1): u = i / 6.; P.append(np.array([x, y]) + e * L * s * u + n * w * s * (.3 + .7 * math.sin(u * 2.6)) * r.uniform(.8, 1.2))
        Q = spl(P + [np.array([x, y]) + n * w * .12], True, n=5); c.fill(Q, col); (kit.outline(c, Q, 4, '#5a1408', 3) if k == 0 else None)


class Shot:
    id = 'c10'; n = 60; finish = dict(bloom=.22, vig=.14, soft=.08)
    EVENTS = [(0, 'holy_choir', dict(size=.4, note='one solemn chord, cut dead at 19')), (12, 'stomach_growl', dict(size=.5)), (15, 'blink', dict(note='eyes left')), (17, 'blink', dict(note='eyes right')), (19, 'plaster_crack', dict(size=1.)), (20, 'stretch', dict(size=1., note='two rubber arms shoot out')), (22, 'snatch', dict(x=690)), (22, 'snatch', dict(x=1208)), (23, 'snatch', dict(x=510)), (23, 'snatch', dict(x=1372)), (24, 'snatch', dict(x=345)), (25, 'snatch', dict(x=1690)), (25, 'jaw_drop', dict(x=1702, note='bones clack, hat pops')), (26, 'snatch', dict(x=196)),
              (27, 'miss', dict(x=1780, note='hand closes on air, knot squeak')), (28, 'pop', dict(x=1196, note='shark teeth')), (30, 'slap', dict(x=420)), (31, 'stomp', dict(size=1., x=1300, note='heel on the rubber arm')), (32, 'inflate', dict(x=692, note='fur ball boing')), (33, 'bonk', dict(size=1., x=985)), (33, 'slap', dict(x=470)), (34, 'gulp', dict(x=960)), (36, 'slap', dict(x=520)), (38, 'snap', dict(x=1300, note='pinned arm whips home')), (40, 'gulp', dict(x=960, note='the red bottle')), (42, 'flame', dict(size=1., x=960, dur=5)), (47, 'gulp', dict(x=960, note='body +1')), (49, 'balloon', dict(note='body +2')), (51, 'balloon', dict(note='body +3')), (53, 'balloon', dict(note='button pops')), (55, 'freeze', dict(note='all stop, the chord again')), (57, 'burp', dict(size=.2, note='tiny'))]

    def setup(s):
        s.bg = s._bg(); s.tab = s._table(); s.x, s.y = grid(); s._fres(); s.f18 = None

    # -------------------------------------------------------------- plates
    def _bg(s):
        fn = 'cache/c10_bg.npy'
        if os.path.exists(fn): return np.load(fn).astype(f32)
        x, y = grid(); sky = ramp(np.clip(y / 520, 0, 1), [(0, '#3f9fe4'), (.5, '#8fd0f0'), (1, '#f8ecc8')]); S = clouds.Sky(H, W, sky, light=(.0, -.8, .4), seed=2)
        for X, b, w, h, sd in ((640, 500, 420, 200, 1), (1280, 500, 460, 220, 2), (960, 505, 300, 120, 3)): S.bank(X, b, w, h, ('#fffdf4', '#e4f0f8', '#a9cfe8', '#7fb2d8'), '#f4f2e0', seed=sd)
        img = S.paint(seed=3); d = np.hypot(x - 960, y - 420); img = img + (np.exp(-(d / 210) ** 2) * .35)[..., None] * rgb('#ffe9a0'); img = img * (1 - smooth(176, 172, d)[..., None]) + rgb('#fff0b8') * smooth(176, 172, d)[..., None]; sea_ = smooth(496, 498, y) * (y < 600); img = img * (1 - sea_[..., None]) + sea_[..., None] * ramp(np.clip((y - 498) / 70, 0, 1), [(0, '#8fd8ea'), (.3, '#2fa0d4'), (1, '#1668b4')])
        c = Cel(); k = Cel(); r = np.random.default_rng(3)
        def wall(P, base): c.fill(P, base)
        def ln(P, w=3., col=LN, al=1.): k.line(spl(P, n=6) if len(P) > 2 else P, w, col, taper=(0, .1), seed=int(r.integers(9999)), alpha=al)
        VP = np.array([960., 430.]); toward = lambda p, u: np.asarray(p, float) * (1 - u) + VP * u
        # ceiling canvas
        c.fill([(0, 0), (W, 0), (1460, 140), (460, 140)], '#f3e6c6')
        for i in range(9): a = np.array([i * 240., 0]); b = np.array([460 + i * 125., 140]); c.line([a, b], 7, '#b98a58', taper=(0, 0)); ln([a, b], 2.4)
        for i in range(1, 5): u = i / 5.; yy = 140 * u ** .7; xx = 460 * u ** .7; c.line([(xx, yy), (W - xx, yy)], 6, '#b98a58', taper=(0, 0)); ln([(xx, yy), (W - xx, yy)], 2.2)
        for sd in (0, 1):
            m = (lambda p: p) if sd == 0 else (lambda p: [(W - a, b) for a, b in p]); base, sh_, win = (('#d8a468', '#a8703f', '#7fc8e0'), ('#c98a56', '#8f5836', '#7fc8e0'))[sd]
            c.fill(m([(0, 0), (460, 140), (460, 600), (0, 860)]), base)
            for i in range(1, 9): u = i / 9.; ln(m([(0, 860 * u), (460, 140 + 460 * u)]), 2., '#6d3f28', .8)
            for i, (xa, xb) in enumerate(((20, 130), (190, 270), (320, 380), (412, 446))):
                ya = lambda q: 140 * q / 460; hy = lambda q, v: ya(q) + v * ((600 - 140) * (q / 460) + 860 * (1 - q / 460)) ; P = m([(xa, hy(xa, .16)), (xb, hy(xb, .18)), (xb, hy(xb, .56)), (xa, hy(xa, .58))]); c.fill(P, '#5f3a2c'); Pi = m([(xa + 8, hy(xa, .19)), (xb - 6, hy(xb, .21)), (xb - 6, hy(xb, .53)), (xa + 8, hy(xa, .55))]); c.fill(Pi, win if i % 2 == 0 else '#8f2f34'); kit.outline(k, np.array(P, float), 3.4, LN, i)
                if i % 2: [ln(m([(xa + (xb - xa) * u, hy(xa, .2)), (xa + (xb - xa) * u, hy(xb, .52))]), 1.8, '#e0b84a') for u in (.3, .55, .8)]
            ln(m([(460, 140), (460, 600)]), 5); ln(m([(0, 0), (460, 140)]), 4)
        # back wall with three lights
        hole = [s1('M800 600 L800 300 C800 200 1120 200 1120 300 L1120 600 Z'), np.array([(560, 600), (560, 268), (716, 268), (716, 600)], float), np.array([(1204, 600), (1204, 268), (1360, 268), (1360, 600)], float)]
        c.fill([np.array([(460, 140), (1460, 140), (1460, 600), (460, 600)], float)] + hole, '#e2b67e', evenodd=True); [kit.outline(k, h, 4.5, LN, i) for i, h in enumerate(hole)]
        for q in (hole[1], hole[2]): x0, x1 = q[:, 0].min(), q[:, 0].max(); ln([(x0, 420), (x1, 420)], 5, '#8f5836'); ln([((x0 + x1) / 2, 268), ((x0 + x1) / 2, 600)], 5, '#8f5836')
        for X in (760, 1160): c.fill([(X - 22, 140), (X + 22, 140), (X + 22, 600), (X - 22, 600)], '#b9824e'); ln([(X - 22, 140), (X - 22, 600)], 3); ln([(X + 22, 140), (X + 22, 600)], 3)
        c.fill([(560, 560), (716, 560), (716, 600), (560, 600)], '#a8703f'); c.fill([(1204, 560), (1360, 560), (1360, 600), (1204, 600)], '#a8703f'); c.fill([(800, 560), (1120, 560), (1120, 600), (800, 600)], '#a8703f')
        # lion-sun crest over the middle light
        for i in range(16): a = i * math.pi / 8; c.fill(np.array([(960 + math.cos(a - .16) * 34, 176 + math.sin(a - .16) * 34), (960 + math.cos(a) * 54, 176 + math.sin(a) * 54), (960 + math.cos(a + .16) * 34, 176 + math.sin(a + .16) * 34)]), '#f2a83a')
        c.fill(np.c_[960 + np.cos(_t) * 34, 176 + np.sin(_t) * 32], '#ffd45a'); k.line(np.c_[960 + np.cos(_t) * 34, 176 + np.sin(_t) * 32], 3, LN, taper=(0, 0)); [k.el if 0 else k.fill(np.c_[960 + q * 11 + np.cos(_t) * 3, 172 + np.sin(_t) * 4], LN) for q in (-1, 1)]; k.line([(948, 186), (960, 192), (972, 186)], 3, LN)
        # lawn
        c.fill([(0, 600), (W, 600), (W, H), (0, H)], '#6fb24e'); c.fill([(0, 860), (460, 600), (1460, 600), (W, 860), (W, 600), (0, 600)], '#4f8f40')
        L = c.rgba(); img = img * (1 - L[..., 3:4]) + L[..., :3]; cv = Canvas(img=np.clip(img, 0, 1)); n = 5200; gx, gy = r.uniform(0, W, n), 600 + 480 * r.uniform(0, 1, n) ** .8; col = np.where(r.random((n, 1)) < .5, rgb('#8fcf5e'), rgb('#4a8a3c')); cv.strokes(arcs(gx, gy, np.full(n, -1.4) + r.normal(0, .25, n), 10 + (gy - 600) * .06, .2), 1.4 + (gy - 600) * .006, col, op=.6, seed=3)
        img = repaint(cv.img, sizes=(18, 8), seed=4, thr=(0, .012), dry=.2); light = np.exp(-((x - 960) / 620) ** 2) * np.exp(-((y - 420) / 420) ** 2); img = img * (.86 + .22 * light[..., None]); K_ = k.rgba(); img = img * (1 - K_[..., 3:4] * .9) + K_[..., :3] * .9
        np.save(fn, img.astype(np.float16)); return img.astype(f32)

    def _table(s):
        c = Cel(); r = np.random.default_rng(5); c.fill([(30, TT), (1890, TT), (1916, 706), (4, 706)], '#fffaf0'); c.fill([(4, 706), (1916, 706), (1900, 930), (20, 930)], '#efe6dc'); c.shade([(4, 706), (1916, 706), (1916, 722), (4, 722)], '#c9bccb', np.array([(4, 706), (1916, 706), (1900, 930), (20, 930)], float))
        for i in range(34):
            xx = 24 + i * 56 + r.normal(0, 10); wd = r.uniform(10, 26); c.fill(spl([(xx, 708), (xx + wd, 760), (xx + wd * .8, 930), (xx - wd * .5, 930), (xx - wd * .1, 780)], True, corners=(0, 2, 3), n=5), '#c9bccb'); c.line([(xx + wd * .4, 716), (xx + wd * .9, 800), (xx + wd * .5, 926)], 2.4, LN, taper=(.1, .4), seed=i)
        hem = spl([(20 + i * 63, 930 + 10 * math.sin(i * 1.7)) for i in range(31)], n=5); c.fill(np.vstack([[(20, 900)], hem, [(1900, 900)]]), '#efe6dc'); c.line(hem, 5, LN, taper=(0, 0)); [c.line(np.asarray(P, float), 5, LN, taper=(0, 0)) for P in ([(30, TT), (1890, TT)], [(4, 706), (1916, 706)], [(30, TT), (4, 706), (20, 930)], [(1890, TT), (1916, 706), (1900, 930)])]
        for i in range(16): xx = 70 + i * 118; c.fill(np.array([(xx, 884), (xx + 22, 872), (xx + 44, 884), (xx + 22, 896)]), '#d8503a')
        return c.layer()

    def _fres(s):
        fn = 'cache/c10_fres2.npz'
        if os.path.exists(fn): z = np.load(fn); s.plas, s.crk, s.loss, s.lab = z['p'].astype(f32), z['c'].astype(f32), z['l'].astype(f32), z['b']; s.cen = z['q']; return
        n = fbm(H, W, (300, 110, 40, 12, 4), (1, .7, .5, .4, .4), 3); s.plas = (1 + .045 * n)[..., None]; r = np.random.default_rng(4); pts = np.c_[r.uniform(0, W, 900), r.uniform(0, H, 900)]; big = np.c_[r.uniform(0, W, 64), r.uniform(0, H, 64)]
        from scipy.spatial import cKDTree
        wx, wy = 18 * fbm(H, W, (60, 20), (1, .6), 8), 18 * fbm(H, W, (60, 20), (1, .6), 9); Q = np.stack([s.x + wx, s.y + wy], -1).reshape(-1, 2); d, _ = cKDTree(pts).query(Q, 2); e = (d[:, 1] - d[:, 0]).reshape(H, W); s.crk = (smooth(2.2, .5, e) * smooth(-.4, .9, fbm(H, W, (220, 80), (1, .6), 10))).astype(f32); _, b = cKDTree(big).query(Q); s.lab = b.reshape(H, W).astype(np.uint8); s.cen = big; s.loss = smooth(1.62, 1.72, fbm(H, W, (150, 60, 22, 8), (1, .8, .6, .4), 12) + .45 * smooth(700, 1000, s.y) + .35 * smooth(260, 60, s.y) - .7 * np.exp(-((s.y - 470) / 170.) ** 2)).astype(f32)
        np.savez(fn, p=s.plas.astype(np.float16), c=s.crk.astype(np.float16), l=s.loss.astype(np.float16), b=s.lab, q=big)

    def fresco(s, img):
        L = img @ np.array([.3, .59, .11], f32); g = L[..., None] + (img - L[..., None]) * .55; g = g * rgb('#f4d8a8') * 1.10; g = .07 + g * .90; g = g * s.plas; q = cv2.GaussianBlur(g, (0, 0), 1.0); g = q * .7 + g * .3; g = g * (1 - .55 * s.crk[..., None] * np.array([.8, .9, 1.], f32)); l = s.loss[..., None]; g = g * (1 - l) + rgb('#e6d3b0') * s.plas * l; v = np.clip(np.hypot((s.x - 960) / 1200, (s.y - 520) / 760), 0, 1.2); return (g * (1.04 - .34 * v[..., None] ** 2)).astype(f32)

    # -------------------------------------------------------------- acting
    def pose(s, n, f):
        keys = K[n]; cur = dict(e='calm', t=0., r=0., x=0., y=0., s=0., fat=0., body=0., red=0., jaw=0., hat=0., droop=0., ball=0., leg=0., look=(0, 0), L=None, R=None); prev = dict(cur); k0 = 0
        for kf, d in keys:
            if kf <= f: prev = dict(cur); cur = dict(cur, **d); k0 = kf
        u = (.42, 1.14, .96, 1.)[min(3, f - k0)] if k0 > 0 else 1.; o = dict(cur)
        for q in ('t', 'r', 'x', 'y', 's', 'fat', 'body', 'jaw', 'hat', 'droop', 'ball', 'leg', 'red'): o[q] = prev[q] + (cur[q] - prev[q]) * u
        for q in ('L', 'R'):
            if cur[q] is not None and prev[q] is not None: a, b = np.array(prev[q][:3], float), np.array(cur[q][:3], float); v = a + (b - a) * u; o[q] = (*v, cur[q][3])
        if f - k0 < 2 and k0 > 0: o['s'] += (.06, -.04)[f - k0]
        if f < 55 and not (f < 19 and n != 'luffy' and n != 'zoro'): o['y'] += 2.2 * math.sin((f // 2) * .9 + len(n)); o['r'] += .012 * math.sin((f // 2) * .7 + len(n) * 2)
        return o

    def shake_at(s, f): return sum((kit.shake(f, a, sd, 2.4, t0) for t0, a, sd in ((19, 7, 1), (20, 6, 2), (31, 9, 3), (33, 11, 4), (42, 8, 5), (53, 4, 6)) if 0 <= f - t0 < 8), np.zeros(2))

    def person(s, c, n, f, part='body'):
        o = s.pose(n, f); X, Y, S = SEAT[n]; X, Y = X + o['x'], Y + o['y']; d = Draw(c, X, Y, S, o['r'], o['t'], o['s']); tb = Draw(c, SEAT[n][0] + o['x'] * .5, SEAT[n][1] + o['y'] * .4, S, o['r'] * .3, 0, 0)
        if part == 'body':
            if n == 'chopper' and o['ball'] > .05: s.ball(c, f, o); return
            if n == 'luffy' and o['body'] > .06: return
            tb.torso(n); kw = {}
            if n == 'luffy': kw = dict(fat=o['fat'], look=o['look'], red=max(0, o['red']))
            if n == 'brook': kw = dict(jaw=max(0, o['jaw']), hat=max(0, o['hat']))
            if n == 'franky': kw = dict(droop=float(np.clip(o['droop'], 0, 1)))
            if n == 'nami': kw = dict(look=o['look'])
            d.head(n, o['e'], **kw)
        else:
            if n == 'luffy' or (n == 'chopper' and o['ball'] > .05): return
            for sd, q in ((-1, 'L'), (1, 'R')):
                if o[q] is None: continue
                hx, hy, a, kd = o[q]; sh = np.array([SEAT[n][0] + o['x'] * .5 + sd * S * (.70 if n not in ('jinbe', 'franky') else 1.0), SEAT[n][1] + S * 1.38]); hp = np.array([hx, hy]) + [o['x'] * .3 if n != 'nami' else 0, o['y'] * .3]; m = (sh + hp) / 2 + np.array([sd * S * .42, S * .30]); wr = hp - np.array([math.cos(a), math.sin(a)]) * HR[n] * .7
                w0, w1 = AW[n][0] * 1.14, AW[n][1] * 1.12; col, shd = SLV[n]; arm(c, [sh, m, wr], w0, w1, col, shd, 4.4, sd + 3)
                if n in ('sanji', 'zoro', 'robin', 'jinbe', 'brook'): t = np.linspace(0, 6.283, 14); e = np.array([math.cos(a), math.sin(a)]); c.line([wr - np.array([-e[1], e[0]]) * w1 * .55, wr + np.array([-e[1], e[0]]) * w1 * .55], 8, dict(sanji='#f0c648', zoro='#173327', robin='#c9a0e8', jinbe='#8f2f34', brook='#fffaf2')[n], taper=(0, 0))
                if n == 'zoro' and q == 'L': s.bottle(c, hp, f)
                if n == 'robin' and f >= 27 and f < 46 and q == 'L': food(c, 'bowl', (hp[0] + 384) / 2, hp[1] - 4, 26)
                if n == 'jinbe' and f >= 22 and q == 'L': food(c, 'bowl', 1542, hp[1] - 2, 40)
                if n == 'usopp' and q == 'R' and 30 <= f < 34: food(c, 'tabasco', hp[0] + 16, hp[1] - 20, 30)
                hand(c, hp, a, HR[n] * (1.5 if n == 'franky' else 1.14), kd, SK[n], flip=sd, hoof=n == 'chopper')
            if n == 'sanji' and o['leg'] > .05:                                                # the heel over the table
                l = o['leg']; hip = np.array([1340, 600]); kn = np.array([1330 - 10 * l, 520 - 60 * (1 - l)]); ft = np.array([1300, 440]) * (1 - l) + np.array([1296, 668]) * l; arm(c, [hip, kn, ft], 46, 34, '#23212c', '#100f18', 5, 9); Q = spl([ft + p for p in ((-16, -14), (20, -16), (30, 10), (10, 22), (-54, 24), (-60, 8), (-30, -2))], True, n=5); c.fill(Q, '#15131c'); kit.outline(c, Q, 4.4, LN, 4); c.line(ft + np.array([(-50, 24), (12, 24)]), 4, '#8f6a4a')
                if l > .7: flame(c, ft[0] + 10, ft[1] - 6, 90, 26, f, U)

    def bottle(s, c, p, f):
        e = np.array([math.cos(-.28), math.sin(-.28)]); n = np.array([-e[1], e[0]]); q = lambda u, v: p + e * u + n * v; Q = np.array([q(-44, -20), q(20, -20), q(36, -8), q(58, -7), q(58, 7), q(36, 8), q(20, 20), q(-44, 20)]); c.fill(Q, '#3f6a4a'); kit.outline(c, Q, 4.2, LN, 3); c.fill([q(-34, -14), q(4, -14), q(4, 14), q(-34, 14)], '#f2e6c4'); c.line([q(-24, -8), q(-6, 8)], 4, '#b3202a'); c.line([q(-6, -8), q(-24, 8)], 4, '#b3202a')

    def ball(s, c, f, o):
        g = max(.2, o['ball']) * (1 + .05 * math.sin(f * 1.3) * math.exp(-max(0, f - 33) / 6.)); X, Y = SEAT['chopper'][0], SEAT['chopper'][1] + 20 + o['y']; R = 60 + 74 * g; r = np.random.default_rng(3); a = np.linspace(0, 2 * math.pi, 61)[:-1]; rr = R * np.where(np.arange(60) % 2, 1.0, 1.12) * (1 + .03 * r.normal(0, 1, 60)); Q = np.c_[X + np.cos(a) * rr, Y + np.sin(a) * rr * .96]; c.fill(Q, SK['chopper'][0]); kit.outline(c, Q, 4.6, LN, 4); c.shade(np.c_[X + R * .5 + np.cos(a) * R, Y + R * .5 + np.sin(a) * R], SK['chopper'][1], Q)
        for i in range(26): u, v = r.uniform(0, 6.28), r.uniform(.4, .95); p = np.array([X + math.cos(u) * R * v, Y + math.sin(u) * R * v]); c.line([p, p + np.array([math.cos(u), math.sin(u)]) * 14], 2.6, '#6d3f2c', taper=(.1, .6), seed=i)
        Draw(c, X, Y - 4, 58, .04 * math.sin(f * .8), 0).head('chopper', o['e'])

    def rubber(s, f):
        """left and right arm: (polyline fraction, state)"""
        a = float(np.interp(f, [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 34, 35], [0, .10, .26, .40, .56, .70, .84, .97, 1.0, 1.0, .97, .10, 0])); b = float(np.interp(f, [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 37, 38, 39, 40], [0, .10, .26, .38, .52, .66, .82, .95, 1.0, 1.0, 1.0, .55, .12, 0])); return a, b

    def arms(s, c, f, o):
        a, b = s.rubber(f); out = []
        for P, u, sd, jit in ((PL, a, -1, 1), (PR, b, 1, 2)):
            C = resample(spl(P, n=16), 6.); n = max(6, int(len(C) * u))
            if u <= .02: continue
            q = C[:n].copy(); ph = f * .9; T = np.gradient(q, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-6; N = np.c_[-T[:, 1], T[:, 0]]; env = np.sin(np.linspace(0, math.pi, n)); amp = 9 if 20 <= f <= 29 else 4; q = q + N * (np.sin(np.linspace(0, n * .12, n) - ph) * amp * env)[:, None]
            if sd == 1 and 31 <= f < 38: i = np.argmin(np.abs(q[:, 0] - 1296)); k = np.exp(-((np.arange(n) - i) / 10.) ** 2); q[:, 1] = q[:, 1] * (1 - k) + 694 * k                      # pinned flat under the heel
            w = 30 * (1 - .38 * u); arm(c, q[::3] if n > 12 else q, 38, w, '#f3c096', '#cd8a66', 5., jit); e = q[-1]; d = q[-1] - q[-6]; ang = math.atan2(d[1], d[0]); red = sd == -1 and f >= 30; out.append((e, ang, sd, u)); hand(c, e + d / (np.hypot(*d) + 1e-6) * 16, ang, 26, 'grab' if 21 <= f < 27 else 'fist' if f < 34 else 'grab', ('#f59a86', '#c8604a') if red else SK['luffy'])
        return out

    def frame(s, f):
        if f >= 55: return s.final(f)
        img = s.draw(f); sh = s.shake_at(f); z = 1 + .030 * smooth(0, 18, f) * (f < 19); z = z if f < 19 else 1.03 + .07 * smooth(19, 24, f)
        if f < 19:
            fr = s.fresco(img)
            if f == 18: pass
            img = fr
        else:
            if s.f18 is None: s.f18 = s.fresco(s.draw(18))
            img = s.chips(img, f)
        return comp.cam2d(img, 960 + sh[0], 540 - 34 * smooth(19, 24, f) + sh[1], z).astype(f32)

    def chips(s, img, f):
        t = f - 19
        if t > 8: return img
        if t == 0:                                                                          # the crack runs out from his grin
            m = np.zeros((H, W), f32); d = np.hypot(s.x - 960, s.y - 450); g = np.abs(cv2.Laplacian(s.lab.astype(np.float32), cv2.CV_32F)) > .5; m = cv2.dilate(g.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(f32); return s.f18 * (1 - m[..., None] * .85) + rgb('#2a1a14') * m[..., None] * .85
        out = img.copy(); r = np.random.default_rng(9)
        for i in range(len(s.cen)):
            cx, cy = s.cen[i]; dl = np.hypot(cx - 960, cy - 450) / 700.; v = np.array([(cx - 960) * .062 + r.normal(0, 8), -26 + (cy - 450) * .05 + r.normal(0, 9)]); w_ = r.normal(0, .16); tt = t - dl
            if tt < 0: tt = 0
            p = v * tt + np.array([0, 13.5 * tt * tt])
            if cy + p[1] > H + 200: continue
            m = (s.lab == i); ys, xs = np.nonzero(m)
            if len(xs) == 0: continue
            x0, x1, y0, y1 = max(0, xs.min() - 2), min(W, xs.max() + 3), max(0, ys.min() - 2), min(H, ys.max() + 3); a = cv2.GaussianBlur(m[y0:y1, x0:x1].astype(f32), (0, 0), .8); lay = np.dstack([s.f18[y0:y1, x0:x1] * a[..., None], a]); ed = cv2.erode(m[y0:y1, x0:x1].astype(np.uint8), np.ones((5, 5), np.uint8)).astype(f32); lay[..., :3] = lay[..., :3] * (.72 + .28 * ed[..., None]); c_, sn = math.cos(w_ * tt), math.sin(w_ * tt); M = np.float32([[c_, -sn, cx + p[0] - c_ * (cx - x0) + sn * (cy - y0)], [sn, c_, cy + p[1] - sn * (cx - x0) - c_ * (cy - y0)]]); L = cv2.warpAffine(lay, M, (W, H), flags=cv2.INTER_LINEAR); sd = kit.shifted(L[..., 3], 8, 12)[..., None] * .35; out = out * (1 - sd * (1 - L[..., 3:4])); out = out * (1 - L[..., 3:4]) + L[..., :3]
        return out

    def final(s, f):
        """freeze: the chaos is a fresco now"""
        if getattr(s, 'fz', None) is None: s.fz = s.fresco(s.draw(55))
        if f < 56: z = 1.10
        else: z = 1.03 + .02 * math.exp(-(f - 56) / 1.2)
        img = s.fz
        if f >= 57:                                                                        # one small burp, in full colour
            t = f - 57; c = Cel(ss=2); x, y = 1042 + t * 9, 470 - t * 16; R = 20 + 9 * t; P = []
            for i in range(7): a = i * .897; P.append((x + math.cos(a) * R * .55, y + math.sin(a) * R * .45, R * .62))
            for k, (g, col) in enumerate(((1.12, LN), (1., '#fffdf6'))): [c.fill(np.c_[a + np.cos(_t) * q * g, b + np.sin(_t) * q * g], col) for a, b, q in P]
            c.line(np.c_[x - R * .1 + np.cos(np.linspace(.4, 4.6, 14)) * R * .4, y + np.sin(np.linspace(.4, 4.6, 14)) * R * .34], 3.2, '#9aa6c4'); [c.line([(1020 - 4, 478 - 10 * k), (1030 + 6 * t, 470 - 10 * k - 4 * t)], 3, LN, taper=(.1, .5)) for k in range(2)]; img = over(img, c.layer())
        return comp.cam2d(img, 960, 540, z).astype(f32)

    def draw(s, f):
        img = s.bg.copy(); c = Cel(); o = s.pose('luffy', f)
        for n in ('franky', 'robin', 'usopp', 'nami', 'sanji', 'jinbe', 'brook', 'zoro', 'chopper'): s.person(c, n, f, 'body')
        s.person(c, 'luffy', f, 'body'); L = c.layer(); L = kit.para(L, (0, 250), (0, 700), [(0, '#ffffff'), (.7, '#fff4ea'), (1, '#e0c8d0')], 'mul', .9); xl = np.sign(960 - s.x); L1 = kit.rim(L, 7, 5, '#fff3c4', 1., .5); L2 = kit.rim(L, -7, 5, '#fff3c4', 1., .5); L = np.where((s.x > 960)[..., None], L1, L2); sh = cv2.GaussianBlur(kit.shifted(L[..., 3], 0, 16), (0, 0), 14)[..., None] * .30; img = img * (1 - sh) + rgb('#3a2030') * sh * .6; img = over(img, L)
        img = over(img, s.tab); t = Cel(); r = np.random.default_rng(2)
        for n, (px, kd, fj) in PLT.items():
            if n == 'zoro': plate(t, px, 676, 44)
            elif n != 'jinbe' or f < 22: plate(t, px, 678, 56)
            if f < fj and not (n == 'jinbe' and f >= 22): food(t, kd, px, 668 - (14 if kd in ('cola', 'cotton') else 4), 34 if kd != 'bowl' else 40, 3)
            elif f < fj + 2 and fj < 99: [t.line([(px + dx, 664), (px + dx * 1.6, 640 - 10 * (f - fj))], 3, LN, taper=(.1, .6)) for dx in (-34, 0, 34)]
        if f < 34:                                                                         # his own plate, in the middle
            plate(t, 960, 682, 86)
            if f < 20: food(t, 'meat', 960, 662, 48)
        for X in (604, 846, 1090, 1290, 1456, 1620): t.fill([(X - 12, 690), (X - 15, 648), (X + 15, 648), (X + 12, 690)], '#c9e6f0'); t.line([(X - 12, 690), (X - 15, 648), (X + 15, 648), (X + 12, 690), (X - 12, 690)], 3, LN, taper=(0, 0)); t.fill([(X - 13, 668), (X + 13, 668), (X + 12, 688), (X - 12, 688)], '#e8b04a' if X % 4 else '#b3404a')
        for n in ('franky', 'robin', 'usopp', 'nami', 'sanji', 'jinbe', 'brook', 'zoro', 'chopper'): s.person(t, n, f, 'hands') if n not in ('nami', 'sanji') or f < 28 else None
        if o['body'] > .06:                                                                # the sphere grows over the table
            X, Y, S = SEAT['luffy']; X, Y = X + o['x'], Y + o['y']; d = Draw(t, X, Y + 10, S, 0, 0, o['s'] * .4); d.torso('luffy', fat=float(np.clip(o['body'], 0, 1.06)))
            for sd in (-1, 1): e = np.array([X + sd * S * (1.0 + 1.55 * o['body']), Y + S * (1.6 - .1 * o['body'])]); arm(t, [e - [sd * 30, 10], e + [sd * 22, 20], e + [sd * 40, 46]], 38, 26, '#c3272f', '#7d141f', 4.4, sd + 2); hand(t, e + [sd * 48, 60], 1.2 if sd > 0 else 1.9, 24, 'open', SK['luffy'], flip=sd)
            Draw(t, X, Y - 18 * o['body'], S, o['r'], o['t'], o['s']).head('luffy', o['e'], fat=o['fat'], red=max(0., o['red']), look=o['look'])
        elif f < 20 or (f >= 35 and f < 40):
            for sd, q in ((-1, 'L'), (1, 'R')):
                if f < 20: hx, hy, a, kd = o[q]; hy += o['y'] * .4; arm(t, [(960 + sd * 60, 560), (960 + sd * 118, 620), (hx - sd * 8, hy)], 40, 28, '#c3272f', '#7d141f', 4.6, sd + 2); hand(t, (hx, hy), a, 25, kd, SK['luffy'], flip=sd)
                elif sd == -1 or f >= 40: arm(t, [(960 + sd * 60, 566), (960 + sd * 120, 640), (960 + sd * 60, 590)], 40, 28, '#c3272f', '#7d141f', 4.6, sd + 2); hand(t, (960 + sd * 52, 566), U, 25, 'cup', SK['luffy'], flip=sd)
        if 20 <= f < 40:
            E = s.arms(t, f, o)
            for e, ang, sd, u in E:                                                        # the loot rides in the hand
                got = [(n, PLT[n]) for n in (('chopper', 'usopp', 'robin', 'franky') if sd < 0 else ('nami', 'sanji', 'brook')) if f >= PLT[n][2]]
                for i, (n, (px, kd, fj)) in enumerate(got): food(t, kd, e[0] + 16 * math.cos(i * 2.1 + f) + sd * 14, e[1] - 22 - 18 * i, 26, i)
                if sd < 0 and 33 <= f < 36: food(t, 'tabasco', e[0] - 10, e[1] - 70, 26)
            if f < 26: kit.speed_lines(t, 0.02, 22, 5, (120, 420), LN, f, (120, 600, 1800, 720), .8)
        for n in ('nami', 'sanji'):
            if f >= 28: s.person(t, n, f, 'hands')
        if 36 <= f < 40: food(t, 'tabasco', 960 - 4, 508 + (f - 36) * -4, 30)
        # robin's extra hands out of the cloth: slap, slap, slap
        for i, (t0, px) in enumerate(((30, 420), (33, 470), (36, 520))):
            if t0 - 4 <= f < t0 + 8:
                u = smooth(t0 - 4, t0 - 1, f) * (1 - smooth(t0 + 4, t0 + 7, f)); sw = -.9 + 1.6 * smooth(t0 - 1, t0 + 1, f); b = np.array([px, 700.]); tip = b + np.array([math.sin(sw) * 70, -math.cos(sw) * 70]) * u
                if u > .05: arm(t, [b, (b + tip) / 2 + [-10, 0], tip], 24, 18, *SK['robin'], 4., i); hand(t, tip, math.atan2(tip[1] - b[1], tip[0] - b[0]), 19, 'flat', SK['robin'])
                for k in range(6): a = k * 1.05 + t0; q = b + np.array([math.cos(a), math.sin(a) * .5]) * (10 + 7 * (f - t0 + 4)); t.fill(np.c_[q[0] + np.cos(_t) * 8, q[1] + np.sin(_t) * 5], '#f7b6cc'); t.line(np.c_[q[0] + np.cos(_t) * 8, q[1] + np.sin(_t) * 5], 2, LN, taper=(0, 0))
                if f in (t0, t0 + 1): star(t, px + 60, 648, 46, i)
        if f in (31, 32): star(t, 1290, 690, 74, 5)
        if f in (33, 34): star(t, 985, 300, 96, 8)
        if 35 <= f < 55:                                                                   # the lump
            X, Y = SEAT['luffy'][0] + 22, SEAT['luffy'][1] + o['y'] - 150 - 18 * o['body']; g = min(1., (f - 34) / 3.) * (1 + .1 * math.sin(f)); P = np.c_[X + np.cos(_t) * 30 * g, Y - 14 * g + np.sin(_t) * 34 * g]; t.fill(P, '#f59a9a'); t.line(P, 4, LN, taper=(0, 0)); t.fill(np.c_[X - 8 + np.cos(_t) * 8 * g, Y - 28 * g + np.sin(_t) * 7 * g], '#fff0ee'); [t.line([(X + dx, Y - 56 - 4 * ((f + k) % 4)), (X + dx + 5, Y - 70 - 4 * ((f + k) % 4))], 3.4, '#8a6a7a', taper=(.3, .5)) for k, dx in enumerate((-16, 0, 16))]
        if 42 <= f < 47: flame(t, 972, 512, 470 * (1 if f > 42 else .6), 84, f, -.20 + .05 * math.sin(f * 2.)); [t.line([(960 + sd * 92, 420), (960 + sd * (150 + 12 * (f % 3)), 390 - 10 * k)], 7, '#fffdf6', taper=(.1, .6)) for sd in (-1, 1) for k in range(2)]
        if 52 <= f < 56: b = np.array([960 + 60 * (f - 52), 690 - 50 * (f - 52) + 10 * (f - 52) ** 2]); t.fill(np.c_[b[0] + np.cos(_t) * 11, b[1] + np.sin(_t) * 11], '#f0c648'); t.line(np.c_[b[0] + np.cos(_t) * 11, b[1] + np.sin(_t) * 11], 3, LN, taper=(0, 0))
        T = t.layer(); T = kit.rim(T, 4, 5, '#fff6dd', .55, .5); sh = cv2.GaussianBlur(kit.shifted(T[..., 3], 4, 12), (0, 0), 7)[..., None] * .30; img = img * (1 - sh) + rgb('#5a4060') * sh * .5; return over(img, T)
