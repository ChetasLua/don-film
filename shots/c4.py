"""c4  Sanji.  Black and gold; a lighter; a smoke heart; then the heated leg comes round and the heel lands on the lens."""
from chars.swkit import *
from chars import sanji, swbody as BD
from op import hands as HD, fx_fire as FF, fx_break as FB
import functools

DK = dict(key=-1, rims=[((-8, 3), '#ffd35a', 1.), ((6, 6), '#5a4a8a', .4)], g=(('#8a7a9a', '#3a3048')), L=(-.5, .3, .7))
LT = dict(key=-1, rims=[((9, -5), '#fff0b0', 1.), ((-7, 3), '#ffd35a', .9)], g=(('#ffd9a0', '#ffb070')), L=(-.75, -.15, .65))
RM = dict(key=-1, rims=[((7, 4), '#ffcf6a', 1.), ((-6, 3), '#ffb03a', .7)], g=(('#fff2dc', '#b98a76')), L=(-.4, .5, .7))
HP = dict(skin='#f9cba0', shade='#c8805c', line=INK, nail='#ffe4c8', nail_shade='#cf9472', light='#fff0d8', cuff='#1c1b24', cuff_shade='#0c0b12', cuff_light='#4a4860')


def heart(n=90):
    t = np.linspace(0, 2 * math.pi, n); return np.c_[16 * np.sin(t) ** 3, -(13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t))] / 16.


def hall():
    x, y = grid(); im = ramp(y / 780., [(0, '#140d0c'), (.4, '#2c1c14'), (1, '#57351d')]); im = im * (1 + .05 * fbm(H, W, (200, 60, 14), None, 3, aspect=.25)[..., None]); c = Cel()
    for k in range(13): xx = -60 + k * 170; c.line(np.array([(xx, -20), (xx, 770)]), 7, '#120a07', taper=(.01, .01)); c.line(np.array([(xx + 7, -20), (xx + 7, 770)]), 2.4, '#7a4f2a', taper=(.01, .01), alpha=.6)
    for yy in (128, 560): c.fill(np.array([(-10, yy), (W + 10, yy), (W + 10, yy + 30), (-10, yy + 30)]), '#3a2415'); c.fill(np.array([(-10, yy), (W + 10, yy), (W + 10, yy + 6), (-10, yy + 6)]), '#8a5c30'); c.line(np.array([(-10, yy + 31), (W + 10, yy + 31)]), 4, '#0c0705')
    a = np.linspace(0, 6.283, 90); cx, cy = 960, 300                         # the house sign: crossed knife and fork on a brass plate, a fish-scale rim
    for rr, col in ((252, '#0c0705'), (244, '#7a4c14'), (232, '#d9a336'), (204, '#7a4c14'), (196, '#22140d')): c.fill(np.c_[cx + np.cos(a) * rr, cy + np.sin(a) * rr], col)
    for k in range(26): t = k / 26 * 6.283; c.fill(np.c_[cx + math.cos(t) * 218 + np.cos(a) * 15, cy + math.sin(t) * 218 + np.sin(a) * 15], '#f0c65a'); c.line(np.c_[cx + math.cos(t) * 218 + np.cos(a) * 15, cy + math.sin(t) * 218 + np.sin(a) * 15], 2, '#5a3408', closed=True)
    emblem_cutlery(c, cx + 5, cy + 6, 172, '#5a3408'); emblem_cutlery(c, cx, cy, 172, '#f2d27a')
    for sx in (-1, 1):
        for j in range(5): px = cx + sx * (330 + j * 104); c.line(np.array([(px, 158), (px, 215 + 12 * (j % 2))]), 3.4, '#0c0705', taper=(.01, .01)); r = 34 - 3 * j; py = 250 + 12 * (j % 2); c.fill(np.c_[px + np.cos(a) * r, py + np.sin(a) * r], '#8a4a22'); c.fill(np.c_[px - 4 + np.cos(a) * r * .78, py - 4 + np.sin(a) * r * .78], '#c87a3a'); c.fill(np.c_[px - r * .35 + np.cos(a) * r * .2, py - r * .35 + np.sin(a) * r * .2], '#ffdca0'); c.line(np.c_[px + np.cos(a) * r, py + np.sin(a) * r], 3.4, '#120a07', closed=True)
        for j in range(7): bx = cx + sx * (360 + j * 76); B = spl([(bx - 17, 560), (bx - 17, 486), (bx - 7, 462), (bx - 7, 430), (bx + 7, 430), (bx + 7, 462), (bx + 17, 486), (bx + 17, 560)], True, (0, 3, 4, 7), n=5); cloth(c, B, ('#1c3a2a', '#3a1420', '#2a2c40')[j % 3], '#0a0d10', [B[len(B) // 2:]], '#86b08a', [np.array([(bx - 12, 545), (bx - 12, 492), (bx - 7, 488), (bx - 8, 545)])], w=2.8)
    im = over(im, c.layer()); fl = y > 770; f_ = np.clip((y - 770) / 310, 1e-3, 1); z = 1 / (f_ + .10); u = (x - 960) * z / 330.; v = z * 2.6; ck = ((np.floor(u) + np.floor(v)).astype(int) % 2 == 0); tile = np.where(ck[..., None], rgb('#e6cfa0'), rgb('#1a1216')) * (.45 + .55 * f_)[..., None]; e = np.minimum(np.abs(u - np.round(u)), np.abs(v - np.round(v))); tile = tile * (1 - .5 * (e < .025))[..., None]
    im = np.where(fl[..., None], tile, im); im[764:772] = rgb('#0c0705'); im = im + (np.exp(-((y - 800) / 40) ** 2) * .05)[..., None] * rgb('#ffcf8a')
    for lx in (250, 1670): im = glow(im, lx, 430, 520, '#ff9a3a', .65, 2.2); c = Cel(); c.fill(np.c_[lx + np.cos(a) * 30, 400 + np.sin(a) * 42], '#ffe9a8'); c.line(np.c_[lx + np.cos(a) * 30, 400 + np.sin(a) * 42], 5, '#2a1508', closed=True); c.fill(np.array([(lx - 26, 356), (lx + 26, 356), (lx + 14, 338), (lx - 14, 338)]), '#3a2010'); c.fill(np.array([(lx - 24, 440), (lx + 24, 440), (lx + 12, 456), (lx - 12, 456)]), '#3a2010'); im = over(im, c.layer())
    d = np.hypot((x - 960) / 1150, (y - 560) / 700)[..., None]; return np.clip(im * (1.08 - .5 * np.clip(d - .3, 0, 1)), 0, 1.4).astype(f32)


BASE = dict(neck=(945, 430), sL=(872, 462), sR=(1030, 450), eL=(836, 562), eR=(1064, 556), pL=(902, 648), pR=(1008, 646), hL=(932, 652), hR=(986, 650), kL=(922, 780), aL=(926, 898), tL=(872, 934), kR=(1000, 780), aR=(1010, 898), tR=(1062, 934), head=(938, 338))
def mv(J, dx=0, dy=0, **k): q = {a: ((v[0] + dx, v[1] + dy) if np.ndim(v) else v) for a, v in J.items()}; q.update(k); return q


class Shot:
    id = 'c4'; n = 30; finish = dict(bloom=.36, vig=.18)
    EVENTS = [(0, 'lighter', dict(note='wheel scratch')), (3, 'fire_whoosh', dict(size=.4, note='lighter flare')), (6, 'eye_snap', dict(who='sanji')), (7, 'exhale', dict(note='smoke heart')), (10, 'cloth', dict(note='anticipation')), (13, 'fire_whoosh', dict(size=.8, note='spin, leg ignites')), (19, 'leap', dict()), (22, 'fire_whoosh', dict(size=1.0, note='axe kick falls')), (25, 'impact', dict(size=1.0, note='heel, floor breaks, DON')), (26, 'fire_whoosh', dict(size=1.2, note='ring of fire wipes lens'))]

    def setup(self):
        os.makedirs('cache', exist_ok=True); p = 'cache/c4_plates.npy'
        if not os.path.exists(p):
            a = brush_card(['#0a0706', '#1c1309', '#4a320c', '#a8781a', '#f2c03c'], -.5, 21, dark='#060404', n=(40, 90, 200)); e = Cel(ss=1); emblem_cutlery(e, 1330, 470, 430, '#ffffff'); m = e.rgba()[..., 0] * smooth(-.6, .3, fbm(H, W, (60, 18, 5), None, 4, aspect=3.)); a = a * (1 - .5 * m[..., None]) + rgb('#f4c84a') * (.42 * m)[..., None]
            b = brush_card(['#6a1a4a', '#b23a78', '#e8609c', '#f78fb8', '#ffd0e0'], -.3, 22, dark='#a22e6c'); np.save(p + '.tmp.npy', np.array([a, b, hall()]).astype(np.float16)); os.replace(p + '.tmp.npy', p)
        q = np.load(p).astype(f32); self.card, self.next, self.hall = q[0], q[1], q[2]; self._c = {}; self.gp = FB.ground_proj(None, (0., 0.), hy=560., cx=812, cy=900.)

    # ---- card
    def bust(self, f):
        d = f if 5 <= f <= 7 else f // 2 * 2; lit = f >= 3; key = (d, lit)
        if key in self._c: return self._c[key]
        look = (-.035, -.030) if f < 6 else (0, 0); lid = .62 if f < 5 else (.30 if f == 5 else 1.10 if f == 6 else .95); pit = float(lerp_keys(f, [(0, -.16), (4, -.15), (6, .02), (8, -.02), (10, -.02)])); m = ('o', dict(w=.055 + .02 * max(0, f - 6))) if f >= 7 else ('smirk', {}) if f == 6 else ('flat', {})
        c = Cel(W + 300, H + 300, -150, -100); H_ = sanji.draw(c, 1040, 408, 322, -.30, pit, -.09 + (.02 if f >= 6 else 0), 'smirk' if f >= 6 else 'calm', key=-1, f=d, body=True, broll=-.03, look=look, lid=lid, mouth_=m, wind=.03 * math.sin(d * .7), smoke=False, ember=1., cig_dir=(-.36, -.02, .34)); tip = H_.tip
        L = c.layer(); D = light_layer(L, DK, 1040, 408, 322); B = light_layer(L, LT, 1040, 408, 322) if lit else D; D[..., :3] *= .62; self._c = {key: (D, B, tip)}; return self._c[key]

    DEV = 24.
    @functools.lru_cache(4)
    def hand(self, dev=None):
        dev = self.DEV if dev is None else dev; c = Cel(); ps = dict(HD.POSES['hammer'], wrist=(6, dev)); h = HD.Hand(ps, 'R', R=HD.euler(10, 12, -62), bld=HD.SLIM, cuff=dict(kind='uniform', y=.10)); g = HD.render([h], HD.Cam(668, 946, 246, 5), light=(.5, .6, .5), res=560, clip=(-200, 300, 1300, 1500)); L0 = Cel(); x, y = 742, 846
        P = np.array([(x - 46, y - 92), (x + 46, y - 96), (x + 44, y + 190), (x - 50, y + 190)]); cloth(L0, P, '#e2b02e', '#8a5c12', [np.array([(x + 6, y - 96), (x + 52, y - 96), (x + 52, y + 190), (x + 8, y + 190)])], '#fff0a8', [np.array([(x - 36, y - 84), (x - 22, y - 86), (x - 24, y + 180), (x - 38, y + 180)])], w=6.); cloth(L0, np.array([(x - 34, y - 142), (x + 34, y - 144), (x + 36, y - 94), (x - 36, y - 92)]), '#b9c0cc', '#5c6474', [np.array([(x + 4, y - 144), (x + 38, y - 144), (x + 38, y - 94), (x + 4, y - 94)])], w=5.)
        HD.ink(L0, g, HP, 6.6, 4); return L0.layer()

    def cardf(self, f):
        k = (0, 0, 0, 1.35, .86, 1.05, .95, 1.02, .97, 1.0)[f]; D, B, tip = self.bust(f); img = cam2d(self.card, 960 + 7 * f, 540, 1.05 + .004 * f) * (.42 + .50 * k); fx, fy = 742, 630
        img = glow(img, fx, fy, 1050, '#ff9a30', .40 * k, 2.); L = D + (B - D) * min(1., k); img = cast(img, L, 60, 10, '#000000', .6, 26); img = over(img, L * np.array([1, 1, 1, 1], f32))
        h = self.hand(); h2 = para(h, (742, 700), (600, 1080), [(0, '#ffe2b0'), (1, '#8a5a4a')]) if k else tint(h, '#3a3048', .85); h2 = rim2(h2, 5, 7, '#fff0b8', .9 * min(1, k), .8) if k else h2
        dy = float(lerp_keys(f, [(0, 90), (2, 16), (3, -6), (5, 0), (8, 0), (10, 26)])); img = over(img, shifted(h2, 0, dy))
        if k:
            c = Cel(); t = f / 24.; hh = 150 * (1.5 if f == 3 else 1.) + 8 * math.sin(f * 2.2); x, y0 = 742, 704 + dy
            for w, hgt, col in ((34, hh, '#f2650e'), (25, hh * .84, '#fb9c22'), (17, hh * .64, '#ffe92e'), (9, hh * .40, '#fffbd8')): s = math.sin(f * 1.7) * 7; c.fill(spl([(x - w, y0), (x - w * .9, y0 - hgt * .35), (x + s * .5, y0 - hgt * .75), (x + s, y0 - hgt), (x + w * .5 + s * .5, y0 - hgt * .6), (x + w, y0 - hgt * .2), (x + w * .6, y0 + 8), (x, y0 + 12)], True, (3,), n=8), col)
            L = c.layer(); img = img + blur(L[..., 3], 30)[..., None] * rgb('#ff8a20') * .55 * k; img = over(img, L)
            if f == 3: img = flare(img, 742, 620, 420, '#ffd27a', .9, .6)
            img = over(img, FF.embers(f, 26, 5, origin=(742, 620), spread=160, bokeh=False).rgba) if f > 3 else img
        if f >= 6:                                                           # the smoke turns into a heart
            e = float(ease_out((f - 5) / 4.)); c = Cel(); hp = heart(); cx, cy = tip[0] - 230 * e - 20, tip[1] - 360 * e - 40; sc = 40 + 150 * e; N = 46
            P = [(cx + hp[int(i * 89 / N)][0] * sc + 5 * math.sin(i * 2.1 + f), cy + hp[int(i * 89 / N)][1] * sc + 5 * math.cos(i * 1.7 + f), (15 + 12 * e) * (1 + .25 * math.sin(i * 1.9)), 300 + i, i * .4 + f * .1, 0.) for i in range(N)]
            FB.puffs(c, P, dict(lit='#fff6e8', mid='#e9cfb4', dark='#b58a78', ink='#3a2420'), (.4, -.8), 2.6, 4, curls=False)
            B_, *_ = limb([tip + [0, -8], tip + [-40, -60 * e], (cx + 10, cy + sc * 1.0)], [3, 12, 5], [3, 12, 5]); c2 = Cel(); cloth(c2, B_, '#f2dcc4', w=2.2, ink='#3a2420'); img = over(img, c2.layer(), .85); img = over(img, c.layer(), .95)
        if f >= 8: c = Cel(); FF.flame_stream(c, [(2100, 300 + 100 * (f - 8)), (1500, 620), (900, 560), (2100 - 1100 * (f - 7), 760)], 120 + 150 * (f - 8), f / 24., 4, smooth=.8); img = over(img, c.layer())
        return img

    # ---- kick
    def figure(self, f):
        """(layer, flames drawn behind, flames in front)"""
        d = f // 2; c = Cel(); bk, fr = Cel(), Cel(); t = f / 24.; KZ = 1.30
        def SUIT(J, d, hs=64, **k): BD.suit(c, {a: ((960 + (v[0] - 960) * KZ, 950 + (v[1] - 930) * KZ) if np.ndim(v) else v) for a, v in J.items()}, d, hs=hs * KZ, bs=KZ * .98, **k)
        Z = lambda P: [(960 + (q[0] - 960) * KZ, 950 + (q[1] - 930) * KZ) for q in P]
        if f < 13:
            q = (f - 10) / 3.; J = mv(BASE, -10 * q, 26 * ease(q), kR=(1086, 742 + 10 * q), aR=(1010 - 20 * q, 828), tR=(960 - 20 * q, 862), hR=(992, 672), neck=(938 - 10 * q, 466 + 12 * q)); SUIT(J, d, expr='glare', hs=64, yaw=-.30, pitch=-.14, roll=-.06, wind=-.06)
            return c.layer(), bk, fr, 0.
        if f < 19:                                                           # one full turn on the left foot, drawn in the round on ones
            yaw = (-.42, -1.75, -3.10, -4.40, -5.55, -6.20)[f - 13]; phi = 2.79 - yaw; hip, ank, uz = BD.spin_fig(c, yaw, phi, f, KZ, wind=(.6, 1., 1.1, 1., .8, .4)[f - 13]); L = c.layer()
            g = float(ease_out((f - 12) / 5.)); rx, ry = (318 + 40 * g) * KZ, 128 * KZ; a = np.linspace(phi - min(3.9, 1.35 * (f - 12)), phi - .10, 70); P = np.c_[hip[0] + np.cos(a) * rx / (1 - .10 * np.sin(a)), hip[1] - 12 + np.sin(a) * ry / (1 - .30 * np.sin(a))]; v_ = np.linspace(0, 1, 70)
            for cel, sel in ((bk, np.sin(a) < 0), (fr, np.sin(a) >= 0)):
                ii = np.nonzero(sel)[0]
                for sg in (np.split(ii, np.nonzero(np.diff(ii) > 1)[0] + 1) if len(ii) else []):
                    if len(sg) < 6: continue
                    m_ = float(v_[sg].mean()); w_ = (26 + 92 * m_) * (1 + .55 * max(0, math.sin(a[sg].mean()))); FF.flame_stream(cel, P[sg], w_, t * 1.6, 7 + f, smooth=.35, prof=lambda v, m=m_: .35 + .65 * np.clip(v * .8 + .3 * m, 0, 1) if True else v)
            gs = np.linspace(phi - 1.05, phi - .12, 24); R1 = np.c_[hip[0] + np.cos(gs) * rx * .97, hip[1] - 12 + np.sin(gs) * ry * .97]; R0 = np.c_[hip[0] + np.cos(gs) * rx * .34, hip[1] - 10 + np.sin(gs) * ry * .34]; sm = Cel()
            for q0, q1, col in ((.0, .40, '#16151e'), (.34, .70, '#8f2a10'), (.62, 1.0, '#f27a1c')):                      # the leg that was there a moment ago: one drawn smear
                I, O = R0 + (R1 - R0) * q0, R0 + (R1 - R0) * q1; k_ = np.linspace(0, 1, 24)[:, None] ** .7; mid = (I + O) / 2; sm.fill(np.vstack([mid + (I - mid) * k_, (mid + (O - mid) * k_)[::-1]]), col, .86)
            S_ = sm.layer(); L = over_rgba(S_, L) if uz >= -.1 else over_rgba(L, S_ * (1 - L[..., 3:4])); L = .72 * L + .28 * smear(L, -70 if f != 18 else -26, 4, 9); return L, bk, fr, g
        if f < 22:
            q = float(ease_out((f - 18) / 3.)); dy = -190 * q; J = dict(neck=(890, 560 + dy * .9), sL=(818, 606 + dy), sR=(964, 560 + dy), eL=(800, 716 + dy), eR=(1030, 650 + dy), pL=(880, 780 + dy), pR=(1000, 742 + dy), hL=(934, 788 + dy), hR=(996, 762 + dy), kL=(884, 880 + dy), aL=(962, 952 + dy * .8), tL=(940, 1004 + dy * .8), kR=(1120 - 94 * q, 690 - 90 * q + dy), aR=(1262 - 250 * q, 640 - 210 * q + dy), tR=(1310 - 278 * q, 596 - 224 * q + dy), growR=1.05); SUIT(J, d, expr='glare', hs=64, yaw=-.12, pitch=-.12, roll=-.28, wind=.15, hot='R', far='L', tails=.6)
            P = [(1240 + 90 * q, 900 + dy), (1340 - 40 * q, 720 - 100 * q + dy), (1320 - 170 * q, 660 - 230 * q + dy), (1262 - 250 * q, 620 - 220 * q + dy)]; FF.flame_stream(bk, Z(P), 100, t * 1.5, 9, smooth=.5); FF.flames(fr, np.array(Z([(1150 - 90 * q, 660 - 100 * q + dy), (1200 - 170 * q, 630 - 160 * q + dy), (1262 - 250 * q, 610 - 210 * q + dy)])), 150, t, 3, n=7, hz=2.4); return c.layer(), bk, fr, 1.
        if f in (23, 24):                                                    # the heel comes down the lens
            k = f - 23; x, y, sc, rot = ((960, 420, 190, .42), (850, 590, 430, .16))[k]; dx, dy = ((0, 0), (40, -30))[k]
            J = dict(neck=(1150, 300), sL=(1082, 346), sR=(1222, 312), eL=(1052, 452), eR=(1288, 408), pL=(1118, 530), pR=(1226, 512), hL=(1156, 534), hR=(1196, 524), kL=(1270, 640), aL=(1352, 742), tL=(1418, 742), kR=(1120, 560), aR=(1080, 560), tR=(1050, 560)); J = {a: (v[0] + dx, v[1] + dy) for a, v in J.items()}
            BD.suit(c, J, d, expr='glare', hs=62, bs=1.0, yaw=-.28, pitch=-.16, roll=-.10, wind=.35, far='L', tails=1.4); hp = A_(1166 + dx, 530 + dy); e = A_(x, y) - hp; m = A_(-e[1], e[0]) / np.hypot(*e)
            B, a, b, cc = limb([hp, hp + e * .45 - m * 16, A_(x, y) - e * .02], [46, sc * .30 + 24, sc * .46], [44, sc * .28 + 22, sc * .44]); cloth(c, B, '#24232e', '#100f17', [np.vstack([cc, b[::-1]])], '#4b4a5c', [], w=5.5, folds=[np.array([hp + e * .40 - m * (sc * .2 + 20), hp + e * .45 + m * 10]), np.array([hp + e * .48 - m * (sc * .16), hp + e * .52 + m * (sc * .2)]), np.array([hp + e * .62 + m * (sc * .26), hp + e * .58 - m * 6])], fw=3.)
            n = len(cc); i0 = int(n * .66); c.fill(np.vstack([a[i0:], b[i0:][::-1]]), '#e8461a', .8); c.fill(np.vstack([(a * .7 + b * .3)[int(n * .74):], (a * .35 + b * .65)[int(n * .74):][::-1]]), '#ffb43a', .9); BD.sole(c, x, y, sc, rot, 1., 6. + 4 * k)
            th = np.linspace(math.pi * 1.04, math.pi * 1.96, 30); R_ = np.c_[x + np.cos(th) * sc * .74, y + np.sin(th) * sc * 1.15]; FF.flames(bk, R_, sc * 1.15, t, 7, n=12, hz=2.4, spread=.5); FF.flame_stream(bk, [(1500, -120), (1340, 120), (1120 - 60 * k, 260), (x + sc * .3, y - sc * .5)], 120 + 110 * k, t * 1.8, 11, smooth=.6); FF.flames(fr, np.c_[x + np.cos(th[::-1] - math.pi) * sc * .66, y + np.sin(th[::-1] - math.pi) * sc * 1.2][6:24], sc * .42, t, 9, n=9, hz=2.6); return c.layer(), bk, fr, 1.
        if f < 25:
            q = (f - 22) / 2.; dy = -170 * (1 - q); an = A_(1010 - 110 * q, 250 + 620 * q ** .8); J = dict(neck=(930 + 60 * q, 470 + dy), sL=(850 + 70 * q, 510 + dy), sR=(1010 + 50 * q, 480 + dy), eL=(830 + 60 * q, 624 + dy), eR=(1076 + 30 * q, 580 + dy), pL=(916 + 40 * q, 690 + dy), pR=(1040 + 20 * q, 668 + dy), hL=(960 + 30 * q, 700 + dy), hR=(1010, 690 + dy), kL=(1040, 820 + dy), aL=(1120, 900 + dy), tL=(1160, 950 + dy), kR=tuple((A_(1010, 690 + dy) + an) / 2 + A_(30, -10)), aR=tuple(an), tR=tuple(an + A_(-70, -10 + 10 * q)), growR=1.2 + .8 * q); SUIT(J, d, expr='glare', hs=64, yaw=-.1, pitch=-.1, roll=-.12 + .1 * q, wind=.1, hot='R', far='L', tails=.8); L = c.layer(); L = .5 * L + .5 * smear(L, -10, 90, 11) if f < 24 else L
            P = [(1010, 60), (1180 - 40 * q, 250), (1120 - 60 * q, 520 + 120 * q), tuple(an)]; FF.flame_stream(bk, Z(P), 190, t * 1.8, 11, smooth=.6); FF.flame_stream(fr, Z([tuple(A_(1080, 360 + 200 * q)), tuple((A_(1090, 400) + an) / 2), tuple(an + A_(20, 10))]), 60, t * 1.8, 12, smooth=.4); return L, bk, fr, 1.
        q = min(1., (f - 25) / 3.); J = dict(neck=(1060, 416), sL=(984, 462), sR=(1140, 430), eL=(952, 570), eR=(1190, 540), pL=(1034, 646), pR=(1140, 636), hL=(1072, 650), hR=(1104, 648), kL=(1180, 790), aL=(1236, 912), tL=(1292, 940), kR=(960, 704), aR=(806, 838), tR=(690, 716), growR=2.2); J = {k: ((v[0], v[1] + (14 if f == 25 else -5 if f == 26 else 0)) if np.ndim(v) else v) for k, v in J.items()}
        SUIT(J, d, expr='glare', hs=64, yaw=-.22, pitch=-.12, roll=.04, wind=.3 - .1 * q, hot='R', far='L', tails=1.2); FF.flames(bk, np.array(Z([(1010, 720), (900, 800), (800, 860), (700, 800)])), 300, t, 5, n=11, hz=2.4); FF.flames(fr, np.array(Z([(900, 905), (800, 915), (720, 900)])), 60, t, 6, n=7, hz=2.4); return c.layer(), bk, fr, 1.

    def frame(self, f):
        if f < 10: return np.maximum(self.cardf(f), 0)
        t = f - 25; z = 1.22 + .006 * (f - 10) + (.10 * float(ease((f - 24) / 2.)) if f >= 24 else 0); cx, cy = 960, 590 - (70 * math.sin(min(1, max(0, f - 18) / 5.) * math.pi) if f < 25 else 0); img = blur(cam2d(self.hall, cx, cy, z), 1.8); L, bk, fr, g = self.figure(f); B, F_ = bk.layer(), fr.layer(); L = light_layer(L, RM, 960, 520, 130, boost=1. + .1 * g)
        fl = blur(np.maximum(B[..., 3], F_[..., 3]), 70); lamp = np.clip(img - .62, 0, 1); img = img * (.62 + .25 * g) + lamp * .9 + fl[..., None] * rgb('#ff7a1e') * .62 * g; refl = cv2.flip(np.clip(B + F_, 0, 1), 0); refl = shifted(refl, 0, 780 if f < 25 else 820); fm = smooth(770, 800, grid()[1]); img = img + (blur(refl[..., :3], 9) * fm[..., None] * .30)
        sh = cv2.warpAffine(L[..., 3], np.float32([[1, 0, 0], [0, .16, 786]]), (W, H)); img = img * (1 - .55 * blur(sh, 8) * fm)[..., None]
        if f >= 25:
            c = Cel(); FB.ground_crack(c, self.gp, 2.6, min(1., .35 + .25 * t), 5, glow='#ff8a2a', top='#b89a70', topdark='#2a1c18'); G = c.layer(); img = over(img, G); c = Cel(); FB.debris(c, t + 3, (812, 890), 40, 8, (900, 2100), pal=FB.ROCK); DB = c.layer()
        if f >= 25: img = over(img, DB)
        if f == 26:
            c = Cel(); FF.flame_ring(c, 800, 900, 420, 92, 260, f / 24., 5, n=44, front=False); img = over(img, c.layer())
        img = over(img, B); img = over(img, L); img = over(img, F_)
        if f == 26:
            c = Cel(); FF.flame_ring(c, 800, 915, 420, 92, 120, f / 24., 6, n=36, back=False); FB.shockwave(c, 800, 905, .28, 120, 1300, .22, 34, col='#fff3c8', shade='#ff8a2a'); img = over(img, c.layer())
        if 12 < f < 25 or f > 25: img = over(img, FF.embers(f, 60, 8, origin=(960, 640), spread=420, bokeh=True).rgba); e = FF.embers(f, 40, 9).light; img = img + (e if e is not None else 0)
        if f == 25: return impact(np.clip(img, 0, 1), 'red', .22, .62, ink='#080406', paper='#fff8e6', accent='#f08a1c')
        if f >= 27:
            r = (170, 470, 1000, 1640)[f - 26]; ry = r * (.42, .55, .72, .9)[f - 26]; cx, cy = 840, (880, 800, 700, 590)[f - 26]; a = np.linspace(0, 6.283, 120); E = np.c_[cx + np.cos(a) * r, cy + np.sin(a) * ry]; m = poly_mask(E, soft=2)[..., None]; nx = cam2d(self.next, 960, 540, 1.12 - .02 * (f - 26)); img = img * (1 - m) + nx * m; c = Cel(); [FF.flame_stream(c, np.c_[cx + np.cos(b) * r, cy + np.sin(b) * ry], (50, 110, 170, 210)[f - 26], f / 24. * 1.6 + k, 31 + k, smooth=.35, prof=lambda v: .55 + .45 * np.sin(np.pi * np.clip(v, 0, 1))) for k, b in enumerate((np.linspace(-.3, 1.75, 50), np.linspace(3.5, 1.4, 50), np.linspace(1.2, -.9, 50) - 1.0, np.linspace(2.8, 5.0, 50)))]; FF.flame_ring(c, cx, cy, r, ry, (190, 300, 420, 460)[f - 26], f / 24., 5, n=76, hot=True); R_ = c.layer(); img = img + blur(R_[..., 3], 40)[..., None] * rgb('#ff7a1e') * .4; img = over(img, R_)
        if f in (23, 24): c = Cel(); x, y, r0 = ((960, 420, 420), (850, 590, 820))[f - 23]; focus_lines(c, x, y, r0, 2200, 130, 11, '#fff0b8', 60 + f, ry=.8); focus_lines(c, x, y, r0 * 1.1, 2200, 60, 9, '#1a0806', 90 + f, ry=.8); img = over(img, c.layer())
        if 12 < f < 19: c = Cel(); speed_lines(c, .05, 34, 4, (300, 1200), '#ffd27a', 70 + f, box=(0, 380, W, 900), alpha=.55); img = over(img, c.layer())
        if f in (24, 26, 27): d = shake(f, 13); img = cam2d(img, 960 + d[0], 540 + d[1], 1.025)
        return np.maximum(img, 0)


def over_rgba(a, b):
    k = b[..., 3:4]; return a * (1 - k) + b
