"""b4: a tear lands on paper and the whole voyage blooms inside a boy's profile; we fly in through the sun path to the open sea of b5."""
import numpy as np, cv2, math, os
from scipy.interpolate import PchipInterpolator
from op import W, H, kit, comp, sea, sumi, ships, clouds, fx_core as FX
from op import hat as HAT
from op.cam import Camera3D
from op.paint import rgb, f32, smooth, mix, ramp, grid, fbm, repaint
from op.ink import Cel, spl, over_roi
from op.comp import over, screen, radial
from chars import stage_world as SW

SUN = np.array([960., 486.]); HZ = 602.; FOC = 1450.; E0, T0 = np.array([1.6, -9.5, 9.6]), np.array([.2, .6, .2]); D0 = 430.
GLASS = dict(sea.TROPIC, deep='#0a86a0', body='#1aa9b8', mid='#3cc4c6', light='#86e6da'); AZ = dict(sea.DAY, deep='#0c4f9c', body='#1f7fc8', mid='#46b0de', light='#9fe0ee', sky='#fff3c4', far='#ffe6a8', crest='#3fd0c8')
# name, screen x of the base at f0, distance (m), width in px at f0, lift of the base (px, for things in the sky)
BOARDS = [('elephant', 1300, 2600, 340, -18), ('drums', 1176, 1500, 120, 0), ('dome', 1070, 1250, 150, 0), ('bell', 1262, 900, 120, 190), ('river', 1058, 760, 170, 0), ('whale', 1036, 640, 58, 0), ('clock', 1172, 520, 84, 0), ('skull', 892, 330, 112, 0), ('pagodas', 1290, 250, 176, 0),
          ('sunny', 1092, 150, 150, 0), ('cliffs', 1400, 96, 220, 0), ('boat1', 1178, 70, 54, 0), ('boat0', 1322, 52, 70, 0), ('boat2', 1066, 47, 66, 0)]
OX = 186                                                              # the profile is set so that the sun sits where his eye would be
TR = PchipInterpolator([0, 8, 16, 24, 32, 40, 47, 53, 57, 59.9], [0, .010, .040, .13, .30, .55, .80, .955, .993, 1.0])
PI = PchipInterpolator([0, 20, 38, 45, 50, 55, 59.9], [2.45, 2.45, 1.6, -3.5, -16., -36., -42.7])


class Shot:
    id = 'b4'; n = 60; finish = dict(bloom=.34, vig=.14, soft=.14)
    EVENTS = [(0, 'tear_on_paper', dict(size=.5, note='drop lands, paper breathes')), (1, 'ink_bloom', dict(dur=16, note='wet ink spreading, colour opens')), (6, 'gull', dict(size=.3)), (14, 'push', dict(dur=38, note='wind rises, sails crack, bell far off')), (20, 'bell', dict(size=.3, note='golden bell, high and far')),
              (30, 'waterfall_pass', dict(size=.5, x=300)), (36, 'ship_pass', dict(size=.6, x=1500, note='hull and rigging rush by right')), (44, 'whale_call', dict(size=.4)), (48, 'dive', dict(size=.6, note='wind drops away, glass water')), (55, 'water', dict(size=.3, note='calm lapping into b5'))]

    def setup(s):
        s.sky = s._sky(); s.P, s.M, s.K, s.A, s.RD = s._paper(); s.B = {n: SW.board(n) for n, *_ in BOARDS}; s.S = sea.Sea(5, hs=.20, length=10, chop=.6); s.hat = s._hat(); s.x, s.y = grid()

    def _hat(s):
        fn = 'cache/b4_hat4.npy'
        if os.path.exists(fn): return np.load(fn).astype(f32)
        L = HAT.hat(1006 + OX, 172, 1150, yaw=-1.30, pitch=.055, roll=.06, light='gold', wear=.7, f=0, wind=.2); np.save(fn, L.astype(np.float16)); return L

    def _sky(s):
        fn = 'cache/b4_sky2.npy'
        if os.path.exists(fn): return np.load(fn).astype(f32)
        h, w = 1300, 2300; y, x = np.mgrid[0:h, 0:w].astype(f32); hz = 900.; cx = 1150.; sy = hz - (HZ - SUN[1]); d = np.hypot((x - cx), (y - sy) * 1.15)
        g = ramp(np.clip((hz - y) / 900, 0, 1), [(0, '#ffe9b4'), (.08, '#f6e6c2'), (.20, '#a8dcec'), (.42, '#4fb0e6'), (.75, '#1f74cf'), (1, '#154fa8')]); k = np.exp(-(d / 300) ** 1.3)[..., None]; k2 = np.exp(-(d / 300) ** 2)[..., None]; g = g * (1 - .86 * k2) + rgb('#ffb648') * .86 * k2; k3 = np.exp(-(d / 130) ** 2)[..., None]; g = g * (1 - .55 * k3) + rgb('#ffe08a') * .55 * k3
        S = clouds.Sky(h, w, g, light=(0., .35, -.15), seed=4); T1 = ('#fff6dc', '#cfe6f2', '#7fb4de', '#4a86c4'); T2 = ('#fffaf0', '#dcecf5', '#8fbce0', '#4f88c6')
        for X, bs, wd, hg, tt, sd in ((420, 900, 700, 330, T1, 3), (1900, 900, 760, 360, T1, 5), (760, 890, 380, 150, T2, 7), (1560, 890, 420, 170, T2, 8), (1150, 892, 300, 60, T2, 9), (120, 620, 520, 220, T1, 11), (2150, 560, 560, 230, T1, 12), (900, 300, 420, 120, T2, 13), (1450, 260, 460, 130, T2, 14)): S.bank(X, bs, wd, hg, tt, '#e6f0ee', seed=sd, detail=.9)
        img = S.paint(seed=2); th = np.arctan2(y - sy, x - cx); rays = np.clip(FX.noise(np.cos(th) * 5, np.sin(th) * 5, 1., 3), 0, 1).astype(f32) * np.exp(-d / 620.); img = img + rays[..., None] * rgb('#fff3c0') * .20 + (np.exp(-(d / 120) ** 2) * .16)[..., None] * rgb('#ffdf90'); img = img * (1 - smooth(66, 62, d)[..., None]) + rgb('#fffbe6') * 1.12 * smooth(66, 62, d)[..., None]; rg = (np.exp(-((d - 72) / 3.2) ** 2) * .75)[..., None]; img = img * (1 - rg) + rgb('#e88a2a') * rg
        np.save(fn, img.astype(np.float16)); return img.astype(f32)

    def _paper(s):
        fn = 'cache/b4_paper4.npz'
        if os.path.exists(fn): z = np.load(fn); return [z[k].astype(f32) for k in 'PMKAR']
        P = sumi.paper(H, W, 21, '#f3ead6'); c = Cel(); r = np.random.default_rng(6)                   # an old chart: rhumb lines meet at the sun, a rose, soundings
        for k in range(48): a = k * math.pi / 24; c.line([SUN + np.array([math.cos(a), math.sin(a)]) * 120, SUN + np.array([math.cos(a), math.sin(a)]) * 1500], 1.4 if k % 4 else 2.2, '#b99a70', taper=(0, .1), seed=k, alpha=.36)
        for R in (470, 476, 640): t = np.linspace(0, 6.283, 200); c.line(np.c_[SUN[0] + np.cos(t) * R, SUN[1] + np.sin(t) * R], 1.3, '#b08c60', taper=(0, 0), seed=R, alpha=.42)
        for k in range(120): a = k * math.pi / 60; c.line([SUN + np.array([math.cos(a), math.sin(a)]) * 640, SUN + np.array([math.cos(a), math.sin(a)]) * (654 if k % 10 else 672)], 1.2, '#a88458', taper=(0, 0), seed=k, alpha=.5)
        for X, Y, n_ in ((230, 260, 11), (330, 830, 9), (1790, 150, 8), (560, 560, 6)):                  # coast scribbles
            t = np.linspace(0, 6.283, 30)[:-1]; R = 90 * (1 + .25 * r.normal(0, 1, 29)); c.line(spl(np.c_[X + np.cos(t) * R, Y + np.sin(t) * R * .7], True, n=5), 1.8, '#8d6a48', taper=(0, 0), seed=n_, alpha=.6)
            for k in range(n_ * 3): q = r.normal(0, 70, 2); c.line([(X + q[0], Y + q[1]), (X + q[0] + 5, Y + q[1] - 8), (X + q[0] + 10, Y + q[1])], 1.2, '#8d6a48', seed=k, alpha=.5)
        P = over(P, c.layer())
        face = [(748, 300), (722, 352), (708, 400), (708, 436), (692, 470), (664, 502), (648, 520), (656, 536), (690, 546), (692, 560), (678, 576), (674, 590), (692, 600), (684, 614), (690, 632), (702, 646), (692, 674), (702, 708), (738, 738), (794, 758), (842, 776), (864, 808), (866, 852), (846, 898), (786, 936), (696, 968), (604, 1010), (560, 1060), (546, 1120)]
        back = [(1600, 1120), (1560, 1040), (1470, 968), (1340, 920), (1250, 884), (1214, 842), (1216, 790), (1260, 716), (1298, 640), (1324, 540), (1320, 440), (1290, 360), (1250, 300)]
        face = [(x_ + OX, y_) for x_, y_ in face]; back = [(x_ + OX, y_) for x_, y_ in back]
        C = np.vstack([spl(face, n=10, corners=(6,)), spl(back, n=10)]); m = kit.mask_of(C); m = sumi.inkify(m, .55, 3, 1.4, 2.6)
        pg = sumi.Page(seed=5)                                                                      # bleeding edges of the bust
        for X, Y, R, an, sd in ((600, 1030, 70, 2.6, 1), (560, 980, 46, 2.9, 2), (1540, 1010, 80, .4, 3), (1430, 950, 50, -.2, 4), (1320, 560, 44, .1, 5), (690, 970, 40, 2.4, 6)): pg.add(sumi.splash(X + OX, Y, R, sd, ang=an, spread=.9, n=90, big=10., fingers=8))
        sp = pg.ink()[0]; M = np.clip(np.maximum(m, sp * smooth(760, 1000, grid()[1])), 0, 1)
        ink = Cel(); hk = []                                                                       # black hair: fringe under the brim, sideburn, nape
        for a, tip, b, bd in (((742, 296), (668, 440), (800, 300), .05), ((784, 300), (728, 424), (838, 306), -.06), ((700, 300), (636, 396), (760, 296), .06), ((838, 304), (812, 410), (884, 306), .04), ((1010, 330), (1004, 470), (1046, 340), .03), ((1170, 350), (1204, 640), (1226, 380), .03), ((1206, 356), (1216, 806), (1280, 380), .05), ((1250, 350), (1296, 786), (1330, 390), .06), ((1286, 340), (1372, 700), (1352, 400), .08), ((1300, 340), (1410, 590), (1352, 420), .06), ((1310, 360), (1450, 486), (1356, 440), .05), ((1290, 400), (1426, 792), (1350, 470), .10), ((1180, 380), (1172, 720), (1230, 400), -.03)): hk.append(kit.spike(*[(v[0] + OX, v[1]) for v in (a, tip, b)], .16, bd))
        for q in hk: ink.fill(q, '#000000')
        ink.fill(spl([(720 + OX, 296), (1000 + OX, 330), (1200 + OX, 366), (1340 + OX, 400), (1330 + OX, 320), (1000 + OX, 270), (720 + OX, 270)], True), '#000000'); K1 = sumi.inkify(ink.layer()[..., 3], .7, 7, 1.0, 2.2)
        wl = Cel()                                                                                 # light strands in the black
        for i, q in enumerate(hk):
            n = len(q) // 2
            for u, al in ((.30, .5), (.48, .9), (.66, .4)): L = q[1:n - 1] * (1 - u) + q[::-1][1:n - 1] * u; wl.line(L[int(len(L) * .25):], 3.4 * al + .8, '#ffffff', taper=(.5, .5), seed=i, alpha=al)
        K1 = K1 * (1 - .40 * sumi.inkify(wl.layer()[..., 3], .9, 4, .8, 1.0))
        pen = Cel(); pw = lambda Q, w, **k: pen.line(spl(Q, n=10), w, '#000000', press=.5, **k)      # the few ink lines of the face
        pw(face[2:20], 6.5, taper=(.1, .25), seed=3); pw(face[19:], 8, taper=(.2, .1), seed=4); pw(back[1:8], 8, taper=(.1, .3), seed=5)
        sc = [(924, 566), (960, 575), (998, 564)]; pw(sc, 6.5, taper=(.15, .3), seed=12); [pw([(x_, y_ - 17), (x_ + 2, y_ + 16)], 5., taper=(.2, .4), seed=13) for x_, y_ in ((946, 572), (976, 572))]
        K2 = sumi.inkify(pen.layer()[..., 3], .6, 9, .8, 1.2)
        rd = sumi.Page(seed=8)                                                                     # vermilion: the open shirt as brush strokes
        for q in ([(740, 970, 40), (640, 1010, 150), (520, 1060, 190), (400, 1120, 90)], [(1260, 900, 30), (1400, 950, 130), (1540, 1030, 190), (1680, 1120, 110)], [(860, 930, 20), (800, 980, 90), (720, 1080, 130), (690, 1140, 80)], [(1200, 900, 20), (1240, 990, 100), (1290, 1090, 140), (1300, 1150, 80)]): rd.stroke([(v[0] + OX, *v[1:]) for v in q], dry=.35, deplete=.9, split=.7)
        rd.add(sumi.splash(470 + OX, 1030, 70, 21, ang=2.8, spread=.8, n=90)); rd.add(sumi.splash(1640 + OX, 1010, 70, 22, ang=.2, spread=.8, n=80)); RD = rd.ink()[0]
        x, y = grid(); T = np.full((H, W), 9., f32)                                                 # when the wet front reaches each point
        for X, Y, t0, v in ((960, 486, 0, 46), (900, 640, 4, 40), (1300, 520, 4, 40), (1086, 800, 5, 44), (1426, 900, 8, 44), (826, 1000, 9, 44)): T = np.minimum(T * 0 + np.minimum(T, 99), 99) if X == 0 else np.minimum(T if T.max() > 9 else T + 90, t0 + np.hypot(x - X, y - Y) / v)
        T = T + (1.15 * fbm(H, W, (260, 110), (1, .55), 3) + .16 * fbm(H, W, (26, 11), (1, .6), 4)) * smooth(30, 300, np.hypot(x - SUN[0], y - SUN[1])) + .35 * fbm(H, W, (90, 40), (1, .5), 6) * smooth(10, 120, np.hypot(x - SUN[0], y - SUN[1])); T = cv2.GaussianBlur(T, (0, 0), 2.2); np.savez(fn, P=P.astype(np.float16), M=M.astype(np.float16), K=np.maximum(K1, K2).astype(np.float16), A=T.astype(np.float16), R=RD.astype(np.float16))
        return P, M, np.maximum(K1, K2), T, RD

    # ---------------------------------------------------------------- camera
    def cam(s, f):
        u = float(TR(min(f, 59.9))); eye = np.array([1.6 * u, E0[1] - D0 * (1 - u), 12. * (1 - u) + 9.6 * u + 5. * math.sin(math.pi * u)]); p = math.radians(float(PI(min(f, 59.9)))); yw = math.atan2(T0[0] - E0[0], T0[1] - E0[1]) * smooth(.9, 1., u); d = np.array([math.sin(yw) * math.cos(p), math.cos(yw) * math.cos(p), math.sin(p)]); return Camera3D(eye, eye + d, FOC), u

    def world(s, f):
        t = 2 + (f - 60) / 24.; cam, u = s.cam(f); hp, _ = cam.project(np.array([cam.eye[0], cam.eye[1] + 1e6, cam.eye[2]])); hy = float(hp[1]); sx = float(hp[0])
        tilt = smooth(42, 56, f); pal = {k: (mix(rgb(AZ[k]), rgb(GLASS[k]), float(tilt)) if isinstance(AZ[k], str) else AZ[k] * (1 - tilt) + GLASS[k] * tilt) for k in AZ if k in GLASS}; pal['ink'] = mix(rgb(AZ['ink']), rgb(GLASS['ink']), float(tilt)); sd = np.array([0, 1., .10]) * (1 - tilt) + np.array([.70, .25, .66]) * tilt
        z = 1 + .06 * u; M = np.float32([[z, 0, sx - 1150 * z], [0, z, hy - 900 * z]]); img = cv2.warpAffine(s.sky, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        w = s.S.render(cam, t, pal, sun=tuple(sd), step=8 if f > 30 else 6, haze=1 - .7 * tilt); img = w.over(img)
        if tilt < 1:                                                                                 # the sun's road on the water, laid in gold dabs
            g = sea.glitter(cam, t, (0, 1., .10), 900, seed=3, spread=.05, size=1.4, rmin=8.) ; yy = np.clip((s.y - hy) / (H - hy + 1e-3), 0, 1); wd = 26 + 210 * yy; road = np.exp(-((s.x - sx) / wd) ** 2) * smooth(0, 3, s.y - hy); br = FX.noise_img(H, W, 26, t * 1.6, 7, 2, 4, aniso=3.4); img = img + ((smooth(-.05, .18, br + .25 * (1 - yy)) * road * .50 * (1 - tilt))[..., None]) * rgb('#ffd98a') + g * .5 * (1 - tilt)
        img = img + (np.exp(-np.abs(s.y - hy) / 26.) * .06 * (1 - tilt))[..., None] * rgb('#fff0c8')
        order = sorted(BOARDS, key=lambda b: -b[2])
        for nm, bx, dist, wpx, lift in order:
            T = s.B[nm]; Wm = wpx * dist / FOC; X = (bx - 960) * dist / FOC; Y = E0[1] - D0 + dist; Z = lift * dist / FOC; c = cam.cam(np.array([X, Y, Z])); zc = c[2]
            if zc < 6: continue
            q, _ = cam.project(np.array([X, Y, Z])); k = FOC * Wm / zc / T.shape[1]; al = float(smooth(6, 16, zc)) * (1 - float(smooth(46.5, 50.5, f)))
            if al < .01 or k * T.shape[1] < 4: continue
            bob = 0.
            if nm == 'sunny': bob = math.sin(t * 2.1) * 2 * k * 10
            if nm == 'whale': bob = math.sin(t * 1.4) * 3
            Mt = np.float32([[k, 0, q[0] - k * T.shape[1] / 2], [0, k, q[1] - k * T.shape[0] + bob + 2]])
            if lift <= 0: Mr = np.float32([[k, 0, q[0] - k * T.shape[1] / 2], [0, -k * .8, q[1] + k * .8 * T.shape[0] - 2]]); R = cv2.warpAffine(T, Mr, (W, H), flags=cv2.INTER_AREA if k < .5 else cv2.INTER_LINEAR); wob = cv2.remap(R, (s.x + 5 * np.sin(s.y * .22 + t * 5) * np.clip((s.y - q[1]) / 80, 0, 1)).astype(f32), s.y, cv2.INTER_LINEAR); img = over(img, cv2.GaussianBlur(wob, (0, 0), sigmaX=1.2, sigmaY=4), .42 * al * (1 - tilt)) if zc < 900 else img
            src = T if k > .5 else cv2.GaussianBlur(T, (0, 0), min(2.4, .35 / max(k, .05)))
            img = over(img, cv2.warpAffine(src, Mt, (W, H), flags=cv2.INTER_LINEAR), al)
            if lift <= 0:                                                                            # foam at the foot, haze behind the next plane
                hw = k * T.shape[1] * .52; fm = np.exp(-((s.x - q[0]) / hw) ** 6) * np.exp(-((s.y - q[1]) / max(2., hw * .035)) ** 2); br = smooth(.0, .3, FX.noise_img(H, W, 18, t, 11, 2, 4, aniso=3.)); img = img * (1 - (fm * br * .8 * al)[..., None]) + (fm * br * .8 * al)[..., None] * rgb('#f4fdff')
            if zc > 200: img = img + (np.exp(-((s.y - hy + 6) / 34.) ** 2) * .05 * (1 - tilt))[..., None] * rgb('#fff2d0')
        img = over(img, s.gulls(f, cam)); img = img + (np.exp(-(np.hypot(s.x - sx, s.y - (hy - (HZ - SUN[1]))) / 190.) ** 2) * .10 * (1 - tilt))[..., None] * rgb('#ffe9a8')
        if f >= 46:                                                                                  # the dinghy on the glass, exactly where b5 finds it
            ps = ships.pose(0, 0, .10 + float(s.S.height(0, 0, t)), 1.95, .03 * math.sin(t * 2.2), .05 * math.sin(t * 1.7 + 1))
            try: d = ships.draw([(ships.dinghy(t, barrel_=True), ps)], cam, light=(.70, .25, .66), lw=1.1); img = sea.under(img, d.layer, d.z, w)
            except Exception: pass
            try:
                from shots import b5
                if not hasattr(s, 'act'): s.act = b5.Actor('luffy', 'vest')
                R0, t0 = ps[0], ps[1]; o = t0 + R0 @ np.array([-.25, 0, .42]); l, ox, oy = s.act.pose(b5.SIT).place(o, 1.95 - math.pi / 2 + .1).draw(cam, 0, light=(.70, .25, .66), rim=dict(col='#fffbe0', w=2.4, dir=(.70, .25, .66), amt=.95), expr='calm', wind=(.3, .1, 0), w=1.7); img = over_roi(img, l, ox, oy)
            except Exception as e: s.err = repr(e)
            if f >= 53: img = screen(img, sea.glitter(cam, t, (.25, .9, .5), 260, seed=2, rmin=2., rmax=60., spread=.6, size=.5) * .55 * float(smooth(53, 58, f)))
        return img

    def gulls(s, f, cam):
        c = Cel(); r = np.random.default_rng(12); t = f / 24.
        for i in range(16):
            d = r.uniform(60, 520); X, Z = r.normal(0, .16) * d, r.uniform(.05, .22) * d + 8; P_ = np.array([X + math.sin(t * .8 + i) * 4 + t * 5 * (1 if i % 2 else -1), E0[1] - D0 + d, Z]); q, z = cam.project(P_)
            if z < 8: continue
            k = FOC / z * 1.5; fl = math.sin((f // 2 * 2) * .55 + i * 1.7) * .5 + .25; sg = 1 if i % 2 else -1; w_ = [(-1, -.1 - fl * .55), (-.55, -.32 - fl * .25), (-.12, -.02), (0, .10), (.12, -.02), (.55, -.32 - fl * .25), (1, -.1 - fl * .55), (.5, -.16 - fl * .05), (.1, .16), (0, .22), (-.1, .16), (-.5, -.16 - fl * .05)]; Q = q + np.array(w_) * k * [sg, 1]; c.fill(spl(Q, True, corners=(0, 6), n=4), '#fbfdff' if z < 160 else '#e8f2f6'); c.line(spl(Q[:7], n=5), max(1., k * .05), '#23304e', taper=(.3, .3), seed=i); c.fill([Q[0], Q[0] * .75 + Q[1] * .25, Q[11] * .4 + Q[0] * .6], '#23304e')
        return c.layer()

    def frame(s, f):
        img = s.world(f); z = float(PchipInterpolator([0, 6, 14, 22, 30, 36, 42, 46], [1.0, 1.015, 1.05, 1.22, 1.75, 2.8, 6.5, 14.])(min(f, 46)))
        if f < 46:
            tf = [0, 1.2, 2.6, 4.2, 5.8, 7.5, 9.2, 11, 13, 15, 17, 19, 22, 26][min(13, f)] + max(0, f - 13) * 3; M = s.M * smooth(tf + .10, tf - .10, s.A); ed = (np.exp(-((s.A - tf + .10) / .20) ** 2) + .50 * np.exp(-np.maximum(tf - s.A, 0) / .8) * (s.A < tf) + .30 * np.exp(-np.maximum(s.A - tf, 0) / .35) * (s.A >= tf)) * s.M
            k_in = smooth(4, 15, f); h_in = smooth(5, 13, f); P = s.P.copy()
            a_rd = smooth(tf + 1., tf - 1., s.A - 3) * s.RD; P = P * (1 - a_rd[..., None]) + mix(rgb('#b3202a'), rgb('#e0452e'), np.clip(.5 + .35 * fbmc(s), 0, 1)[..., None]) * a_rd[..., None]
            a = 1 - M; lay = np.dstack([P * a[..., None], a]); ek = np.clip(ed * .62, 0, 1)[..., None]; lay = lay * (1 - ek) + np.concatenate([np.ones((H, W, 3), f32) * rgb('#173a72'), np.ones((H, W, 1), f32)], -1) * ek
            kk = (s.K * smooth(tf - 2.2, tf - 3., s.A))[..., None]; fl = 1.; lay = lay * (1 - kk * fl) + np.concatenate([np.ones((H, W, 3), f32) * rgb('#0d0b0e'), np.ones((H, W, 1), f32)], -1) * kk * fl
            hh = s.hat; sh = cv2.GaussianBlur(kit.shifted(hh[..., 3], 6, 30), (0, 0), 14)[..., None] * .34 * smooth(tf - 1.2, tf - 2., s.A * .92)[..., None]; lay[..., :3] *= 1 - sh; lay[..., 3:4] = np.maximum(lay[..., 3:4], sh * .9 * (M[..., None])); hm = smooth(tf - 1.2, tf - 2., s.A * .92)[..., None]; lay = lay * (1 - hh[..., 3:4] * hm) + hh * hm
            if f < 4:                                                                                # the tear itself: a drop, then a crown of ink
                c = Cel(); q = SUN; t_ = np.linspace(0, 6.283, 30)
                if f == 0: c.fill(np.c_[q[0] + np.cos(t_) * 46, q[1] + np.sin(t_) * 40], '#1d4f92'); c.fill(np.c_[q[0] - 12 + np.cos(t_) * 14, q[1] - 12 + np.sin(t_) * 9], '#ffffff')
                kit.splatter(c, q[0], q[1], 70 + 50 * f, 16, 9, '#1d4f92', 3); lay = over4(lay, c.layer())
            if z > 1.0005:
                Mt = np.float32([[z, 0, SUN[0] * (1 - z)], [0, z, SUN[1] * (1 - z)]]); lay = cv2.warpAffine(lay, Mt, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            if z > 1.5: lay = kit.zoomblur(lay, SUN[0], SUN[1], min(.16, (z - 1.5) * .05), 9)
            img = over(img, lay)
        if 30 <= f < 53: img = kit.zoomblur(img, 960, SUN[1] if f < 46 else 540, .022 * math.sin((f - 30) / 23. * math.pi), 7)
        return np.maximum(img, 0).astype(f32)


_fb = {}
def fbmc(s):
    if 'a' not in _fb: _fb['a'] = fbm(H, W, (160, 50, 18), (1, .6, .4), 9)
    return _fb['a']


def over4(a, b): return a * (1 - b[..., 3:4]) + b
