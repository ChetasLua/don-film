"""c12: the door peek.  Three heads, one per four frames, and a reindeer who hides the wrong way round."""
import numpy as np, cv2, math, os
from op import W, H, kit, comp
from op.paint import rgb, f32, smooth, mix, ramp, grid, fbm, repaint, Canvas, arcs
from op.ink import Cel, spl
from op.comp import over, radial
from chars import chibi as CB
from chars.chibi import Draw, hand, arm, LN, SK

DX = 600; PX = 1400; SH = -220                                                    # door edge, post
HEADS = [('luffy', 0, (748, 296), 148, .34, 'peek'), ('zoro', 4, (704, 622), 134, -.16, 'dull'), ('sanji', 8, (744, 886), 130, .30, 'calm')]
TURN = dict(luffy=.34, zoro=-.30, sanji=.10)


def pop(t): return (0., .40, 1.16, .95, 1.02, 1.)[min(5, max(0, t))]


class Shot:
    id = 'c12'; n = 30; finish = dict(bloom=.20, vig=.12, soft=.08)
    EVENTS = [(0, 'pop', dict(who='luffy', size=.4)), (4, 'pop', dict(who='zoro', size=.4)), (8, 'pop', dict(who='sanji', size=.4)), (10, 'tiptoe', dict(x=1640, dur=8, note='four hoof taps 10 12 14 16')), (17, 'bump', dict(x=1540, note='forehead meets the post')), (19, 'blink', dict(note='three pairs of eyes slide right')), (25, 'sweatdrop', dict(note='ting, three at once'))]

    def setup(s): s.x, s.y = grid(); s.bg, s.door, s.post = s._plates()

    def _plates(s):
        fn = 'cache/c12b.npz'
        if os.path.exists(fn): z = np.load(fn); return [z[k].astype(f32) for k in 'abc']
        r = np.random.default_rng(3); c = Cel(); k = Cel()
        c.fill([(0, 0), (W, 0), (W, 860), (0, 860)], '#f2dfb6')
        for i in range(17): x = i * 118 - 20; c.fill([(x, 0), (x + 112, 0), (x + 112, 860), (x, 860)], ('#f4e2bc', '#eed7aa', '#f6e8c8')[i % 3]); k.line([(x + 114, 0), (x + 115, 860)], 2.6, '#a8804e', taper=(0, 0), seed=i, alpha=.8)
        for i in range(60): x, y = r.uniform(0, W), r.uniform(0, 840); k.line(spl([(x, y), (x + r.normal(0, 4), y + 40), (x + r.normal(0, 6), y + r.uniform(60, 160))], n=5), 1.5, '#c9a268', taper=(.3, .6), seed=i, alpha=.7)
        c.fill([(0, 640), (W, 640), (W, 866), (0, 866)], '#5fa8a0'); c.fill([(0, 628), (W, 628), (W, 652), (0, 652)], '#b9783e'); [k.line([(0, y), (W, y)], 3.4, LN, taper=(0, 0)) for y in (628, 652, 866)]
        for i in range(17): x = i * 118 - 20; k.line([(x + 114, 654), (x + 114, 864)], 2.4, '#2f6f70', taper=(0, 0), alpha=.8)
        c.fill([(0, 866), (W, 866), (W, H), (0, H)], '#b9783e')
        for i in range(13): x = -300 + i * 210; c.fill([(960 + (x - 960) * .8, 866), (960 + (x + 200 - 960) * .8, 866), (x + 200, H), (x, H)], ('#c58848', '#b4743a')[i % 2]); k.line([(960 + (x - 960) * .8, 866), (x, H)], 2.6, '#6d3f22', taper=(0, 0), alpha=.8)
        # a shelf with mugs, a ladle, a wanted-poster back, a round window with the sea
        c.fill([(1030, 300), (1360, 300), (1360, 322), (1030, 322)], '#8f5836'); k.line([(1030, 300), (1360, 300), (1360, 322), (1030, 322), (1030, 300)], 3.4, LN, taper=(0, 0))
        for i, (x, col) in enumerate(((1068, '#d8503a'), (1140, '#f0c648'), (1212, '#4f8fd0'), (1288, '#f4f2ee'))): q = spl([(x, 300), (x - 4, 236), (x + 46, 236), (x + 42, 300)], True, corners=(0, 1, 2, 3), n=3); c.fill(q, col); kit.outline(k, q, 3.2, LN, i); c.shade([(x + 26, 230), (x + 60, 230), (x + 60, 304), (x + 30, 304)], mix(rgb(col), rgb('#3a2a4a'), .35), q); k.line(spl([(x + 46, 250), (x + 64, 262), (x + 44, 284)], n=6), 4.4, LN, taper=(0, 0))
        for i, x in enumerate((1080, 1180, 1290)): k.line([(x, 322), (x, 350)], 3, LN); c.line([(x, 350), (x + 4, 440)], 10, '#b9c2cf', taper=(0, 0)); c.fill(np.c_[x + 4 + np.cos(CBt) * 24, 456 + np.sin(CBt) * 22], '#b9c2cf'); k.line(np.c_[x + 4 + np.cos(CBt) * 24, 456 + np.sin(CBt) * 22], 3, LN, taper=(0, 0)); k.line([(x - 2, 350), (x + 1, 436)], 2.2, LN); k.line([(x + 8, 350), (x + 10, 436)], 2.2, LN)
        wx, wy = 1690, 250; q = lambda R: np.c_[wx + np.cos(CBt) * R, wy + np.sin(CBt) * R]; c.fill(q(150), '#c99a4a'); c.fill(q(122), '#8fd4f0'); c.ctx.save(); c._path(q(122)); c.ctx.clip(); c.fill([(1560, 330), (1900, 330), (1900, 460), (1560, 460)], '#2f8fd0'); [c.fill(np.c_[a + np.cos(CBt) * R, b + np.sin(CBt) * R * .7], '#ffffff') for a, b, R in ((1680, 300, 44), (1726, 286, 56), (1776, 304, 46))]; c.ctx.restore(); [k.line(q(R), 4.4, LN, taper=(0, 0)) for R in (150, 122)]; [c.fill(np.c_[wx + math.cos(a) * 136 + np.cos(CBt) * 8, wy + math.sin(a) * 136 + np.sin(CBt) * 8], '#7a4a1e') for a in np.linspace(0, 6.283, 9)[:-1]]
        L = c.rgba(); img = repaint(np.clip(L[..., :3], 0, 1), sizes=(16, 7), seed=2, thr=(0, .012), dry=.2); x, y = s.x, s.y; beam = smooth(0, 120, (x - 900) * .9 - (y - 100) * .55) * smooth(760, 600, (x - 900) * .9 - (y - 100) * .55); img = img * (.90 + .16 * beam[..., None]) + (beam * .05)[..., None] * rgb('#fff0b8'); ao = np.exp(-np.clip(x - DX, 0, None) / 120.) * .42 * (x > DX - 4); img = img * (1 - ao[..., None] * np.array([.8, 1., .9], f32)); K = k.rgba(); bg = img * (1 - K[..., 3:4]) + K[..., :3]
        # the door
        c, k = Cel(), Cel(); D = np.array([(-40, -40), (DX, -40), (DX, 1120), (-40, 1120)], float); c.fill(D, '#c8402f')
        for i in range(6): x = 40 + SH + i * 140; c.fill([(x, -40), (x + 132, -40), (x + 132, 1200), (x, 1200)], ('#cf4a36', '#bd3a2c')[i % 2]); k.line([(x + 134, 0), (x + 134, 1080)], 3, '#6d1420', taper=(0, 0), seed=i, alpha=.8)
        c.fill([(DX, -40), (DX + 34, -20), (DX + 34, 1120), (DX, 1120)], '#f2b79a'); c.fill([(DX + 20, -40), (DX + 34, -20), (DX + 34, 1120), (DX + 20, 1120)], '#b06a5a')
        for yy in (90, 830): c.fill([(-40, yy), (DX, yy), (DX, yy + 90), (-40, yy + 90)], '#e0b048'); c.fill([(-40, yy + 58), (DX, yy + 58), (DX, yy + 90), (-40, yy + 90)], '#a8702c'); [k.line([(-40, yy + d), (DX, yy + d)], 4, LN, taper=(0, 0)) for d in (0, 90)]; [(c.fill(np.c_[x + np.cos(CBt) * 15, yy + 44 + np.sin(CBt) * 15], '#fff0b8'), k.line(np.c_[x + np.cos(CBt) * 15, yy + 44 + np.sin(CBt) * 15], 3, LN, taper=(0, 0))) for x in (80, 300, 520)]
        wx, wy = 250, 470; [c.fill(q_, col) for q_, col in ((np.c_[wx + np.cos(CBt) * 190, wy + np.sin(CBt) * 190], '#e0b048'), (np.c_[wx + np.cos(CBt) * 158, wy + np.sin(CBt) * 158], '#1d3f66'), (np.c_[wx - 16 + np.cos(CBt) * 120, wy - 18 + np.sin(CBt) * 116], '#2f6a9c'))]; [k.line(np.c_[wx + np.cos(CBt) * R, wy + np.sin(CBt) * R], 5, LN, taper=(0, 0)) for R in (190, 158)]; c.fill([(wx - 100, wy - 60), (wx - 30, wy - 120), (wx - 10, wy - 100), (wx - 90, wy - 30)], '#9fd4f0'); c.fill([(wx - 60, wy + 10), (wx + 40, wy - 90), (wx + 54, wy - 76), (wx - 44, wy + 26)], '#6fb0dc')
        for a in np.linspace(0, 6.283, 13)[:-1]: c.fill(np.c_[wx + math.cos(a) * 174 + np.cos(CBt) * 9, wy + math.sin(a) * 174 + np.sin(CBt) * 9], '#8a561f')
        q = spl([(478, 660), (548, 652), (560, 680), (548, 708), (478, 702)], True, n=6); c.fill(q, '#f2c850'); kit.outline(k, q, 4.4, LN, 3); kit.outline(k, np.array([(DX, -60), (DX + 34, -40), (DX + 34, 1140), (DX, 1140)], float), 5.5, LN, 4)
        L = c.rgba(); a = L[..., 3:4]; g = repaint(np.clip(L[..., :3] + (1 - a) * rgb('#c8402f'), 0, 1).astype(f32), sizes=(14, 6), seed=5, thr=(0, .012), dry=.2); g = g * (.80 + .24 * smooth(0, DX, x)[..., None]); K = k.rgba(); door = np.dstack([g * a, a]); door = door * (1 - K[..., 3:4]) + K
        c, k = Cel(), Cel(); Pp = np.array([(PX, -40), (PX + 96, -40), (PX + 96, 960), (PX, 960)], float); c.fill(Pp, '#b9783e'); c.fill([(PX + 58, -40), (PX + 96, -40), (PX + 96, 960), (PX + 58, 960)], '#7d4a2c'); c.fill([(PX - 22, 900), (PX + 118, 900), (PX + 118, 984), (PX - 22, 984)], '#8f5836'); kit.outline(k, Pp, 5.5, LN, 3); kit.outline(k, np.array([(PX - 22, 900), (PX + 118, 900), (PX + 118, 984), (PX - 22, 984)], float), 5, LN, 4)
        for i in range(14): xx, yy = r.uniform(PX + 8, PX + 90), r.uniform(0, 880); k.line(spl([(xx, yy), (xx + 4, yy + 40), (xx - 2, yy + 90)], n=5), 2, '#5f3420', taper=(.3, .6), seed=i)
        L = c.rgba(); a = L[..., 3:4]; g = repaint(np.clip(L[..., :3] + (1 - a) * rgb('#b9783e'), 0, 1).astype(f32), sizes=(10, 5), seed=6, thr=(0, .012)); K = k.rgba(); post = np.dstack([g * a, a]); post = post * (1 - K[..., 3:4]) + K; post = kit.rim(post, -6, 0, '#fff3cf', .9, .6); door = kit.rim(door, -5, 0, '#fff0d0', .7, .6)
        np.savez(fn, a=bg.astype(np.float16), b=door.astype(np.float16), c=post.astype(np.float16)); return bg.astype(f32), door.astype(f32), post.astype(f32)

    def frame(s, f):
        img = s.bg.copy(); c = Cel(); Ls = []; look = (0., 0.) if f < 19 else (min(1., (f - 18) / 2.) * .95, .05); g = Cel()
        for i, (n, t0, (hx, hy), S, rl, e) in reversed(list(enumerate(HEADS))):
            t = f - t0
            if t < 0: continue
            u = pop(t); x = DX - 230 + (hx - DX + 230) * u; y = hy + 2.5 * math.sin(f * .32 + i * 2) - (5 if f in (25, 26) else 0); sq = (.16, .10, -.10, .06, 0, 0)[min(5, t)]; r = rl * u + .02 * math.sin(f * .28 + i); ex = e
            if n == 'sanji' and 13 <= f < 19: ex = 'heart'
            if f >= 25: ex = dict(luffy='stare', zoro='dull', sanji='dull')[n]
            Draw(c, x - 10 * u, y + 6, S * .92, r + .62, 0, 0).torso(n); d = Draw(c, x, y, S, r, TURN[n] * (1 - .5 * abs(look[0])) + (.42 * look[0]), sq)
            d.head(n, ex, look=look if n != 'zoro' else (look[0], 0)); Li = c.layer(); Ls.append(Li); c = Cel()
        L = np.zeros((H, W, 4), f32)
        for Li in Ls:
            Li = kit.rim(Li, -8, 6, '#fff0c4', 1., .6); Li = kit.rim(Li, -20, 14, '#ffd98a', .30, 3.); shd = cv2.GaussianBlur(kit.shifted(Li[..., 3], -24, 30), (0, 0), 3.5)[..., None] * .46; L[..., :3] *= 1 - shd * np.array([.75, .95, .85], f32); L = L * (1 - Li[..., 3:4]) + Li
        dk = (smooth(DX + 96, DX + 88, s.x + (s.y - 540) * .05) * .42)[..., None]; L[..., :3] *= 1 - dk * np.array([.7, .95, .85], f32); L = kit.para(L, (DX, 700), (DX + 460, 300), [(0, '#8f6a80'), (.30, '#d8b8b8'), (.62, '#fff4e4'), (1, '#fff2d0')], 'mul', .95); sh = cv2.GaussianBlur(kit.shifted(L[..., 3], -46, 44), (0, 0), 7)[..., None] * .46; img = img * (1 - sh) + rgb('#5a3040') * sh * .6; img = over(img, L)
        # fingers on the edge come first, then the face
        img = over(img, s.door); h = Cel()
        for i, (n, t0, (hx, hy), S, rl, e) in enumerate(HEADS):
            t = f - t0 + 1
            if t < 0: continue
            u = pop(t); y = hy + 60 + 4 * math.sin(i) ; x = DX + 30 - 46 * (1 - u)
            for j in range(4): q = np.array([(x + 8, y + j * 29 - 14), (x - 40, y + j * 29 - 15), (x - 62, y + j * 29 - 2), (x - 42, y + j * 29 + 12), (x + 8, y + j * 29 + 13)]); Q = spl(q, True, n=6); h.fill(Q, SK[n][0]); kit.outline(h, Q, 5.2, LN, j); h.shade(q + [4, 12], SK[n][1], Q)
        hl = h.layer(); hl = kit.rim(hl, -4, 3, '#fff6dc', .8, .5); sh = cv2.GaussianBlur(kit.shifted(hl[..., 3], -6, 10), (0, 0), 5)[..., None] * .35; img = img * (1 - sh); img = over(img, hl)
        # Chopper tiptoes in and hides behind a post, from the wrong side
        c2 = Cel()
        if f >= 10:
            t = f - 10; hop = abs(math.sin(min(t, 8) * math.pi / 2)) * 26 if t < 8 else 0; x = np.interp(f, [10, 17, 18, 30], [2040, 1562, 1574, 1574]); y = 600 - hop; sq = 0 if t != 7 else -.12; tr = 1.6 * math.sin(f * 2.4) if f >= 19 else 0; x += tr; S = 148
            b = Draw(c2, x, y, S, .06, 0, 0)
            for sd in (-1, 1): st = math.sin(min(t, 8) * math.pi / 2 + (0 if sd < 0 else math.pi)) * 20 if t < 8 else 0; arm(c2, [(x + sd * 44, y + 262), (x + sd * 52, y + 320), (x + sd * 54, y + 356 - max(0, st))], 62, 54, *SK['chopper'], 5.5, sd + 2); Q = spl([(x + sd * 54 - 40, y + 356 - max(0, st)), (x + sd * 54 + 40, y + 356 - max(0, st)), (x + sd * 54 + 46, y + 396 - max(0, st)), (x + sd * 54 - 46, y + 396 - max(0, st))], True, corners=(0, 1, 2, 3), n=4); c2.fill(Q, '#4a3433'); kit.outline(c2, Q, 5.5, LN, 4)
            Q = spl([(x - 84, y + 96), (x - 104, y + 220), (x - 78, y + 290), (x + 78, y + 290), (x + 104, y + 220), (x + 84, y + 96)], True, n=6); c2.fill(Q, SK['chopper'][0]); kit.outline(c2, Q, 6, LN, 5); c2.shade([(x + 30, y + 80), (x + 60, y + 300), (x + 160, y + 300), (x + 160, y + 80)], SK['chopper'][1], Q); Q = spl([(x - 92, y + 190), (x - 100, y + 296), (x + 100, y + 296), (x + 92, y + 190)], True, corners=(0, 1, 2, 3), n=4); c2.fill(Q, '#8a3f7a'); kit.outline(c2, Q, 6, LN, 6); c2.shade([(x + 30, y + 180), (x + 50, y + 300), (x + 120, y + 300), (x + 120, y + 180)], '#552a5e', Q)
            b = Draw(c2, x, y, S, -.05 + .02 * math.sin(f * .8), -.35, sq); b.head('chopper', 'stare' if f < 25 else 'smile', look=(-.8, .0))
            arm(c2, [(x - 70, y + 150), (x - 120, y + 160), (PX + 112, y + 120)], 46, 40, *SK['chopper'], 5.5, 3); hand(c2, (PX + 92, y + 112), math.pi + .2, 36, hoof=True)
        L = c2.layer(); L = kit.rim(L, -8, 6, '#fff0c4', 1., .6); L = kit.rim(L, -18, 12, '#ffd98a', .30, 3.); L = kit.para(L, (1400, 900), (1720, 420), [(0, '#8f6a80'), (.45, '#e6cdc4'), (1, '#fff4dc')], 'mul', .95)
        if f >= 10: pk = (smooth(PX + 150, PX + 142, s.x - (s.y - 500) * .10) * smooth(PX + 60, PX + 70, s.x) * .48)[..., None]; L[..., :3] *= 1 - pk * np.array([.7, .95, .85], f32); fs = radial(x - 40, 992, 250, 34, 1.2); img = img * (1 - .55 * fs[..., None])
        sh = cv2.GaussianBlur(kit.shifted(L[..., 3], -64, 40), (0, 0), 8)[..., None] * .42; img = img * (1 - sh) + rgb('#5a3040') * sh * .6; img = over(img, L); pl = s.post.copy()
        if f >= 10:                                                                        # only a sliver of him is behind the post; the hoof stays in front
            img = over(img, pl); c3 = Cel(); hand(c3, (PX + 84, 610 + 112 - (abs(math.sin(min(f - 10, 8) * math.pi / 2)) * 26 if f < 18 else 0)), math.pi + .2, 36, hoof=True) if f >= 17 else None; img = over(img, c3.layer())
        else: img = over(img, pl)
        c4 = Cel()
        if f >= 25:
            t = f - 25; e = (.5, 1.2, .95, 1., 1.)[min(4, t)]
            for x, y, R in ((936, 150, 52), (886, 506, 48), (912, 776, 48)):
                yy = y + 3 * max(0, t - 2); P = spl([(x, yy - R * 1.6 * e), (x + R * .62 * e, yy - R * .1), (x + R * .5 * e, yy + R * .62 * e), (x, yy + R * .86 * e), (x - R * .5 * e, yy + R * .62 * e), (x - R * .62 * e, yy - R * .1)], True, corners=(0,), n=8); c4.fill(P, '#a9dcf6'); c4.shade(np.asarray(P) + [R * .4, R * .3], '#5fa8dc', P); kit.outline(c4, P, 5, LN, 3); c4.fill(spl([(x - R * .28, yy), (x - R * .12, yy + R * .1), (x - R * .16, yy + R * .42), (x - R * .32, yy + R * .3)], True, n=5), '#ffffff')
        if 10 <= f < 18: [c4.line([(1760 + 60 * k + (f % 3) * 8, 470 + 40 * k), (1830 + 60 * k, 462 + 40 * k)], 5, LN, taper=(.1, .6), seed=k) for k in range(3)]
        for k in range(26):                                                                # smoke curls from the cigarette
            if f < 10: break
            u = k / 25.; x = 806 + 18 * math.sin(u * 7 - f * .35) * u + 30 * u; y = 958 - 160 * u; r_ = 4 + 12 * u; c4.fill(np.c_[x + np.cos(CBt) * r_, y + np.sin(CBt) * r_], '#fffdf6', .10 + .18 * (1 - u))
        q = np.array([(1640, 130), (1790, 330), (1010, 1180), (430, 760)], float); sf = comp.shaft(H, W, q, 26, 3, .25)[..., None]; img = img * (.80 + .26 * sf) * np.array([1., .985, .96], f32) + sf * rgb('#ffdf9a') * .060
        img = over(img, c4.layer()); dust = Cel(ss=1); rr = np.random.default_rng(6)
        for i in range(60): x = 900 + (rr.uniform(0, 1000) + f * rr.uniform(.4, 1.6)) % 1000; y = (rr.uniform(0, 900) - f * rr.uniform(.1, .8)) % 900; dust.fill(np.c_[x + np.cos(CBt[::4]) * 2.2, y + np.sin(CBt[::4]) * 2.2], '#fff6d0', .35)
        img = over(img, dust.layer()); z = 1 + .02 * smooth(0, 29, f); q = kit.shake(f, 3, 2, 2., 17) if 17 <= f < 21 else np.zeros(2); return comp.cam2d(img, 960 + q[0], 540 + q[1], z).astype(f32)


CBt = np.linspace(0, 2 * math.pi, 40)
