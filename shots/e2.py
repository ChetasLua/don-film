"""e2: nine arms.  From under water, straight up at Snell's window.  The ring of arms is a heart: each arrival is a beat and the whole ring contracts."""
import numpy as np, cv2, math, os
from op import W, H
from op import kit, comp, fx_core as FX, hands as HD
from op.paint import rgb, f32, smooth, mix, ramp, grid, fbm, repaint, Canvas
from op.ink import Cel, spl, ribbon, over_roi
from op.comp import over, screen, blur, radial, ease, lerp_keys
from chars import stage_arms as SA
from chars.stage_arms import SPEC, ORDER, C0, GRIP, unit2

F = 232.; CACHE = 'cache/e2'
TK = {n: 15 * SPEC[n]['k'] for n in ORDER}
# ----------------------------------------------------------------- the rubber hand: one drawing every two frames
P = HD.POSES; LIMP = dict(P['relaxed'], wrist=(34, 6))
def _lk():
    op = dict(P['open']); wide = HD.lerp_pose(P['open'], P['thrust'], .6); K = {}
    A = [(120, LIMP, -20, 2.3, 10), (122, LIMP, -22, 2.3, 8), (124, dict(LIMP, wrist=(30, 4)), -24, 2.35, 5), (126, dict(LIMP, wrist=(26, 0)), -26, 2.4, 2), (128, HD.lerp_pose(LIMP, op, .12), -28, 2.4, -1),
         (130, HD.blend(LIMP, op, .30, drag=.5, order=(0, 1, 2, 3)), -29, 2.45, -3), (132, HD.blend(LIMP, op, .48, drag=.5), -30, 2.5, -3), (134, HD.blend(LIMP, op, .66, drag=.45), -31, 2.55, -2),
         (136, HD.blend(LIMP, op, .86, drag=.3), -32, 2.7, 0), (138, wide, -34, 3.0, 1), (140, HD.lerp_pose(wide, P['splay'], .35), -35, 3.25, 2), (142, HD.lerp_pose(P['splay'], LIMP, .40), -31, 2.9, 4), (143, HD.lerp_pose(P['splay'], LIMP, .55), -30, 2.75, 5),
         (144, wide, -32, 2.9, 1), (145, HD.lerp_pose(P['grip'], P['hook'], .35), -20, 2.6, -2), (147, HD.lerp_pose(P['grip'], P['fist'], .22), -20, 2.6, -1)]
    for f, p, pt, arm, rl in A: K['f%d' % f] = dict(pose=p, pitch=pt, arm=arm, roll=rl, h=(960, 560), yaw=6 + rl * .6, seed=2, s=190 if f >= 145 else 170)
    return K
SA.LK.update(_lk()); LKEYS = sorted(int(k[1:]) for k in SA.LK)
LPATH = [(119, 1300), (122, 1120), (125, 960), (128, 850), (131, 770), (134, 712), (137, 660), (140, 612), (141, 618), (143, 668), (144, 652), (145, 566), (149, 540)]


def pulse(dt):
    """lub-dub of one arrival"""
    if dt < 0: return 0.
    a = (.55 if dt < 1 else 1.) * math.exp(-max(0, dt - 1) / 3.0); b = 0.
    if dt >= 5: b = .62 * (.55 if dt < 6 else 1.) * math.exp(-max(0, dt - 6) / 3.4)
    return a + b


def energy(f):
    """0..1 how much light has come back"""
    return float(sum(1 - math.exp(-max(0, f - TK[n]) / 7.) if f >= TK[n] else 0 for n in ORDER) / 9.)


class Shot:
    id = 'e2'; n = 150; finish = dict(bloom=.34, vig=.26, soft=.14, halation='#fff0cf', grain=.011)
    EVENTS = [(0, 'underwater', dict(size=.6, note='muffled pressure bed, until 145')), (3, 'tear_on_surface', dict(size=.2, x=1010)), (9, 'tear_on_surface', dict(size=.2, x=900))]
    for _n in ORDER:
        _t = 15 * SPEC[_n]['k']; _s = dict(zoro=1., nami=.7, usopp=.75, sanji=.8, chopper=.4, robin=.6, franky=1.3, brook=.6, jinbe=1.4)[_n]
        EVENTS += [(_t, 'arm_break', dict(who=_n, size=_s, x=SPEC[_n]['h'][0], note='plunge, crown of bubbles')), (_t, 'heart', dict(size=.6 + .045 * _t / 15)), (_t + 5, 'heart2', dict(size=.4)), (_t + 3, 'bubbles', dict(size=_s, x=SPEC[_n]['h'][0]))]
    EVENTS += [(107, 'sprout', dict(who='robin', size=.3)), (109, 'sprout', dict(who='robin', size=.3)), (104, 'last_bubble', dict(size=.3, note='his breath ends')), (120, 'hand_rise', dict(size=.4)), (132, 'fingers_open', dict(size=.2)), (141, 'slip', dict(size=.4, note='strength goes, hand sinks')),
               (142, 'lunge', dict(size=.9, note='nine arms dive at once')), (145, 'grip', dict(size=1.4, note='DON: nine hands on one')), (146, 'light', dict(size=1.2, note='light floods to the cut'))]
    EVENTS = sorted(EVENTS, key=lambda e: e[0])

    def setup(s):
        os.makedirs(CACHE, exist_ok=True)
        s.x, s.y = grid(H // 2, W // 2); s.x = s.x * 2 + .5; s.y = s.y * 2 + .5
        s.X, s.Y = (s.x - C0[0]) / F, (s.y - C0[1]) / F; s.r = np.hypot(s.X, s.Y); s.th = np.arctan2(s.Y, s.X)
        s.sky, s.sil = s._sky(); s.tex = s._tex()
        for n in ORDER:
            for k in ['e%d' % i for i in range(7)] + ['i%d' % i for i in range(8)] + ['a0', 'a1', 'g0', 'g1']: SA.drawing(n, k) if not os.path.exists(f'{CACHE}/{SA.VER}_{n}_{k}.npz') else None
        for k in SA.LK: SA.drawing('luffy', k) if not os.path.exists(f'{CACHE}/{SA.VER}_luffy_{k}.npz') else None
        s.D = {}
        s.bub = s._bubbles()

    def d(s, n, k):
        if (n, k) not in s.D: s.D[n, k] = SA.drawing(n, k)
        return s.D[n, k]

    # ------------------------------------------------------------- plates
    def _tex(s):
        fn = f'{CACHE}/tex.npy'
        if os.path.exists(fn): return np.load(fn).astype(f32)
        cv = Canvas(H // 2, W // 2, color=(.5, .5, .5)); r = np.random.default_rng(4); n = 5200; x, y = r.uniform(0, W / 2, n), r.uniform(0, H / 2, n); a = np.arctan2(y - C0[1] / 2, x - C0[0] / 2) + math.pi / 2 + r.normal(0, .35, n)
        from op.paint import arcs
        cv.strokes(arcs(x, y, a, r.uniform(24, 90, n), r.normal(0, .5, n)), r.uniform(2, 8, n), np.clip(.5 + r.normal(0, .10, (n, 1)), 0, 1) * np.ones((1, 3)), op=.55, dry=.4, seed=3)
        t = cv.img[..., 0].astype(f32); np.save(fn, t.astype(np.float16)); return t

    def _sky(s):
        fn = f'{CACHE}/sky2.npz'
        if os.path.exists(fn): z = np.load(fn); return z['a'].astype(f32), z['b'].astype(f32)
        N = 1200; y, x = np.mgrid[0:N, 0:N].astype(f32); tx, ty = (x - 600) / 560, (y - 600) / 560; rho = np.hypot(tx, ty); th = np.arctan2(ty, tx)
        n = fbm(N, N, (160, 60, 24), (1, .6, .3), 4)
        img = ramp(rho + .02 * n, [(0, '#ffffff'), (.07, '#fffdf0'), (.15, '#fff0bf'), (.28, '#d9f3ee'), (.46, '#8fd6ea'), (.70, '#3fa6d6'), (.90, '#2f7fbe'), (.97, '#ffd9a0'), (1.03, '#e89a5a')])
        c = Cel(N, N); r = np.random.default_rng(11)
        for ring, cnt, sz in ((.80, 15, .13), (.60, 11, .10), (.43, 7, .065)):
            for k in range(cnt):
                a = 2 * math.pi * (k + r.uniform(-.3, .3)) / cnt; q = ring + r.normal(0, .03); e = np.array([math.cos(a), math.sin(a)]); nn = np.array([-e[1], e[0]]); lob = []
                for m in range(13): u = r.normal(0, .55); lob.append((e * (q - r.uniform(-.2, .8) * sz * (1 - .5 * abs(u))) + nn * u * sz * 1.6, sz * r.uniform(.28, .62) * (1 - .25 * abs(u))))
                def blob(sc, off, col, al):
                    for p, R in lob: t = np.linspace(0, 6.283, 22); rr = R * sc; c.fill(np.c_[600 + 560 * (p[0] + off * e[0] + np.cos(t) * rr), 600 + 560 * (p[1] + off * e[1] + np.sin(t) * rr)], col, al)
                blob(1.0, .012, '#7fb9de', 1.); blob(.94, .0, '#cfeaf5', 1.); blob(.74, -.016, '#ffffff', 1.); blob(.34, -.03, '#fff3cf', 1.)
        L = c.rgba(); L = cv2.GaussianBlur(L, (0, 0), 1.2); w = smooth(.30, .42, rho)[..., None]; img = img * (1 - L[..., 3:4] * w * .88) + L[..., :3] * w * .88
        img = repaint(img, sizes=(16, 7), seed=3, thr=(0, .012), dry=.2)
        ray = np.clip(np.sin(th * 9 + 1.2) * np.sin(th * 4 - .3), 0, 1) * smooth(.55, .05, rho); img = img + ray[..., None] * rgb('#fff2c8') * .35
        sil = np.zeros((9, N, N), f32)
        for k, nm in enumerate(ORDER): sil[k] = s._figure(nm, N)
        np.savez(fn, a=np.clip(img, 0, 1.4).astype(np.float16), b=sil.astype(np.float16)); return img.astype(f32), sil

    def _figure(s, nm, N):
        """one of the crew leaning over the rim of the window, as the refracted horizon shows them"""
        h, d = SA.dirs(SPEC[nm]); ang = math.atan2(-d[1], -d[0]); ang += dict(zoro=.32, nami=-.24, usopp=.21, sanji=-.22, chopper=.36, robin=.46, franky=-.40, brook=.22, jinbe=-.52)[nm]
        e = np.array([math.cos(ang), math.sin(ang)]); t = np.array([-e[1], e[0]]); sz = dict(chopper=.23, jinbe=.40, franky=.37, brook=.35).get(nm, .32); c = Cel(N, N)
        def Q(P_): P_ = np.asarray(P_, float); w = e[None] * (1.02 - P_[:, 1] * sz)[:, None] + t[None] * (P_[:, 0] * sz)[:, None]; return 600 + 560 * w
        def fill(P_, sm=True): c.fill(spl(Q(P_), True, n=6) if sm else Q(P_), '#ffffff')
        def circ(x, y, r_, sy=1.): a = np.linspace(0, 6.283, 24); fill(np.c_[x + np.cos(a) * r_, y + np.sin(a) * r_ * sy], False)
        bw = dict(jinbe=1.35, franky=1.1, brook=.62, robin=.75, nami=.75, chopper=.9).get(nm, 1.)
        fill([(-.46 * bw, -.1), (-.44 * bw, .30), (-.26 * bw, .52), (-.08, .58), (.08, .58), (.26 * bw, .52), (.44 * bw, .30), (.46 * bw, -.1)]); circ(0, .74, .17)
        if nm == 'zoro':
            for k in range(7): a = math.pi * (k + .5) / 7; fill([(.16 * math.cos(a - .2), .76 + .16 * math.sin(a - .2)), (.26 * math.cos(a), .76 + .26 * math.sin(a)), (.16 * math.cos(a + .2), .76 + .16 * math.sin(a + .2))], False)
            for k in range(3): fill([(.30, .10 + .07 * k), (.82, .34 + .10 * k), (.80, .39 + .10 * k), (.30, .15 + .07 * k)], False)
        if nm == 'nami': fill([(-.14, .86), (.10, .93), (.22, .78), (.27, .5), (.44, .18), (.30, .05), (.16, .3), (.08, .62)])
        if nm == 'usopp':
            fill([(.10, .78), (.52, .735), (.53, .70), (.10, .68)], False)
            for k in range(9): a = 2.2 + k * .42; circ(-.10 + .22 * math.cos(a), .80 + .20 * math.sin(a), .11)
        if nm == 'sanji': fill([(-.18, .72), (-.14, .92), (.08, .95), (.22, .82), (.20, .62), (.10, .70)]); fill([(.10, .655), (.32, .62), (.32, .60), (.10, .63)], False); fill([(.33, .63), (.36, .74), (.31, .84), (.37, .95), (.34, .84), (.39, .74)])
        if nm == 'chopper':
            fill([(-.34, .86), (.34, .86), (.36, .80), (-.36, .80)], False); fill([(-.22, .84), (-.24, 1.12), (.24, 1.12), (.22, .84)], False)
            for q in (-1, 1): fill([(q * .22, .92), (q * .42, 1.0), (q * .46, 1.2), (q * .50, 1.02), (q * .62, 1.1), (q * .52, .94), (q * .24, .86)], False); circ(q * .34, .70, .10)
        if nm == 'robin': fill([(-.17, .88), (0, .95), (.17, .88), (.21, .6), (.22, .2), (.10, .2), (-.10, .2), (-.22, .2), (-.21, .6)])
        if nm == 'franky':
            circ(.56, .40, .22); circ(-.56, .40, .22); fill([(-.14, .86), (-.10, 1.02), (.18, 1.10), (.36, 1.0), (.34, .90), (.12, .88)])
        if nm == 'brook': circ(0, .80, .33, .92); fill([(-.14, 1.06), (-.12, 1.26), (.12, 1.26), (.14, 1.06)], False); fill([(-.24, 1.08), (.24, 1.08), (.24, 1.04), (-.24, 1.04)], False)
        if nm == 'jinbe': circ(0, .78, .23); circ(-.02, 1.05, .09); circ(.26, .66, .10); circ(-.26, .66, .10)
        return c.rgba()[..., 3]

    # ------------------------------------------------------------- water
    def slopes(s, f):
        t = f / 24.; x, y = s.x, s.y; wx = x + 26 * np.sin(y / 83. + t * 1.3) + 12 * np.sin(y / 31. - t * 2.1); wy = y + 24 * np.sin(x / 97. - t * 1.1) + 10 * np.sin(x / 37. + t * 1.9); sx = np.zeros_like(x); sy = np.zeros_like(x)
        for lam, a, amp, sp in ((300, .35, .024, 1.0), (214, 2.1, .021, 1.25), (150, 4.0, .019, 1.5), (104, 1.2, .015, 1.9), (70, 5.2, .013, 2.4), (46, 3.1, .010, 3.0)):
            k = 2 * math.pi / lam; c_, s_ = math.cos(a), math.sin(a); g = np.cos(k * (wx * c_ + wy * s_) - sp * t * 2.2 + a * 3) * amp; sx += g * c_; sy += g * s_
        for n in ORDER:
            dt = f - TK[n]
            if not 0 <= dt < 34: continue
            E = s.d(n, 'i0')['E']; dx, dy = x - E[0], y - E[1]; r = np.hypot(dx, dy) + 1e-3; A = SPEC[n]['s'] / 150. * .26 * math.exp(-dt / 11.); c = 17.
            w = A * np.cos((r - c * dt) / 11.) * np.exp(-((r - c * dt) / (30 + 1.6 * dt)) ** 2) + A * .5 * np.cos((r - .5 * c * dt) / 9.) * np.exp(-((r - .5 * c * dt) / 26.) ** 2)
            sx += w * dx / r; sy += w * dy / r
        for t0, px, py in ((3, 1016, 452), (9, 896, 540), (12, 1050, 560)):                    # tears on the surface
            dt = f - t0
            if 0 <= dt < 20:
                dx, dy = x - px, y - py; r = np.hypot(dx, dy) + 1e-3; w = .12 * math.exp(-dt / 8.) * np.cos((r - 5. * dt) / 5.) * np.exp(-((r - 5. * dt) / 13.) ** 2); sx += w * dx / r; sy += w * dy / r
        return sx, sy

    def water(s, f, E, beat):
        sx, sy = s.slopes(f); X, Y = s.X, s.Y; nv = 1 / np.sqrt(1 + s.r ** 2); vx, vy, vz = X * nv, Y * nv, nv; m = 1 / np.sqrt(1 + sx ** 2 + sy ** 2); nx, ny, nz = -sx * m, -sy * m, m; cs = vx * nx + vy * ny + vz * nz; eta = 1.333; k = 1 - eta ** 2 * (1 - cs ** 2)
        inside = smooth(-.012, .014, k); q = np.sqrt(np.maximum(k, 0)); b = eta * cs - q; tx, ty = eta * vx - b * nx, eta * vy - b * ny; rt = np.hypot(tx, ty); sc = np.where(rt > .999, .999 / (rt + 1e-6), 1.); mx, my = (600 + 560 * tx * sc).astype(f32), (600 + 560 * ty * sc).astype(f32)
        day = cv2.remap(s.sky, mx, my, cv2.INTER_LINEAR)
        L = day @ np.array([.3, .59, .11], f32); cold = ramp(np.clip(L * .42 + .46 + .34 * np.exp(-(s.r / .42) ** 2), 0, 1.4), [(0, '#0a1c38'), (.45, '#123f6e'), (.66, '#2f7fb4'), (.80, '#63b4dc'), (.95, '#b4e4f6'), (1.2, '#ffffff')]); w = .06 + .94 * E; sk = cold * (1 - w) + day * w; sk = sk * (.74 + .10 * E + .12 * beat)
        for i, n in enumerate(ORDER):
            a = smooth(TK[n] - 5, TK[n] + 1, f)
            if a > 0: mk = cv2.remap(s.sil[i], mx, my, cv2.INTER_LINEAR)[..., None] * a * .90; sk = sk * (1 - mk) + mk * mix(rgb('#0a2038'), rgb('#164a6b'), E)
        # outside: the deep, mirrored, with tilted facets
        rr = s.r; rad = (sx * X + sy * Y) / (rr + .2); tan = (-sx * Y + sy * X) / (rr + .2); tx_ = (s.tex - .5) * .22
        base = ramp(np.clip((rr - 1.0) / 3.4 + .05 * tx_, 0, 1), [(0, '#14587a'), (.12, '#0e4263'), (.34, '#08294a'), (.65, '#051a35'), (1, '#030f24')]); warmb = ramp(np.clip((rr - 1.0) / 3.4, 0, 1), [(0, '#4fb3bc'), (.10, '#2a86a2'), (.27, '#115682'), (.5, '#083560'), (.8, '#051f42'), (1, '#03152f')]); dp = base * (1 - E) + warmb * E
        fa = rad + .5 * tan + tx_ * .10; fall = np.exp(-np.maximum(rr - 1.05, 0) * 1.05)[..., None]
        for th_, c0, c1, o in ((.020, '#1f6f93', '#58c4cf', .55), (.044, '#4aa3bd', '#b5efe6', .62), (.072, '#a7dbe2', '#fff7d8', .8)): mk = smooth(th_, th_ + .010, fa)[..., None] * fall * o; dp = dp * (1 - mk) + mk * (rgb(c0) * (1 - E) + rgb(c1) * E)
        mk = smooth(.022, .032, -fa)[..., None] * .42 * fall; dp = dp * (1 - mk * .5)
        img = dp * (1 - inside[..., None]) + sk * inside[..., None]
        edge = np.exp(-(k / .020) ** 2) * (1 - np.exp(-((rr - 1.134) / .6) ** 2) * 0); img = img + (edge * (.10 + .34 * E))[..., None] * mix(rgb('#8fd0e6'), rgb('#ffe9b8'), E)
        s.silm = cv2.resize(np.clip(sum(cv2.remap(s.sil[i], mx, my, cv2.INTER_LINEAR) * smooth(TK[n] - 5, TK[n] + 1, f) for i, n in enumerate(ORDER)), 0, 1) * inside, (W, H), interpolation=cv2.INTER_CUBIC)[..., None]
        return cv2.resize(img.astype(f32), (W, H), interpolation=cv2.INTER_CUBIC), (cv2.resize(sx, (W, H)), cv2.resize(sy, (W, H)))

    def rays(s, f, E, beat):
        a = np.linspace(-math.pi, math.pi, 721); out = np.zeros((H // 4, W // 4), f32); yy, xx = np.mgrid[0:H // 4, 0:W // 4].astype(f32); dx, dy = xx * 4 - C0[0], yy * 4 - C0[1]; th = np.arctan2(dy, dx); r = np.hypot(dx, dy)
        for sc, sp, g in ((3.1, .34, 1.), (7.0, .55, .7)):
            v = FX.noise(np.cos(a) * sc + 5, np.sin(a) * sc + 9, f / 24. * sp, 3); v = np.clip(v * 1.6 + .12, 0, 1) ** 1.5
            for n in ORDER:                                                              # each arm throws a shadow through the water
                dt = f - TK[n]
                if dt >= 0: h, d = SA.dirs(SPEC[n]); an = math.atan2(-d[1], -d[0]); da = (a - an + math.pi) % (2 * math.pi) - math.pi; v = v * (1 - .8 * min(1, dt / 3.) * np.exp(-(da / .085) ** 2))
            out += np.interp(th, a, v).astype(f32) * g
        out *= smooth(180, 330, r) * np.exp(-np.maximum(r - 300, 0) / 520.); out = cv2.resize(cv2.GaussianBlur(out, (0, 0), 2.0), (W, H), interpolation=cv2.INTER_LINEAR)
        return out[..., None] * mix(rgb('#6fb8d8'), rgb('#ffeec0'), E) * (.10 + .34 * E + .10 * beat)

    # ------------------------------------------------------------- bubbles
    def _bubbles(s):
        B = []
        for n in ORDER:
            sp = SPEC[n]; h, d = SA.dirs(sp); E = s.d(n, 'i0')['E']; r = np.random.default_rng(100 + sp['k']); m = {'chopper': 46, 'jinbe': 120, 'franky': 120, 'zoro': 110, 'brook': 54}.get(n, 78); q = sp['s'] / 150.; nn = np.array([-d[1], d[0]])
            for i in range(m):
                side = r.choice([-1, 1]); b = r.random()
                B.append(dict(t0=TK[n] + int(r.integers(0, 4)), T=r.uniform(14, 52), E=E + nn * side * r.uniform(.1, .42) * sp['s'] + d * r.uniform(-10, 10), va=d * r.uniform(.2, 1.25) * 230 * q + nn * side * abs(r.normal(0, 74)) * q, drift=r.uniform(.0, .30), R=(2. + 24 * b ** 4.2) * q, ph=r.uniform(0, 6.28), wob=r.uniform(2, 9), near=b > .8))
            for i in range(22 if n not in ('chopper', 'brook') else 14):                     # beads off the fingertips
                t0 = TK[n] + int(r.integers(3, 60)); tp = s.d(n, 'i0')['tip']; B.append(dict(t0=t0, T=r.uniform(18, 40), E=tp * r.uniform(.5, 1) + s.d(n, 'i0')['palm'] * 0 + (1 - 1) * tp + (s.d(n, 'i0')['palm'] - tp) * r.uniform(0, 1.), va=nn * r.normal(0, 26) + d * r.uniform(-14, 18), drift=r.uniform(.15, .5), R=r.uniform(1.2, 4.2), ph=r.uniform(0, 6.28), wob=r.uniform(2, 7), near=False))
                B[-1]['E'] = tp + (s.d(n, 'i0')['palm'] - tp) * r.uniform(0, 1.)
        r = np.random.default_rng(7)                                                           # his last breath, from below the lens
        for t0, x, R in ((0, 1000, 30), (2, 1032, 12), (5, 968, 16), (30, 930, 22), (34, 952, 9), (62, 990, 15), (104, 962, 20)): B.append(dict(t0=t0 - 6, T=62, E=np.array([x, 1150.]), va=np.array([r.normal(0, 30), -60.]), drift=1.0, R=R, ph=r.uniform(0, 6), wob=14, near=True, rise=1))
        for i in range(90): B.append(dict(t0=int(r.integers(-40, 150)), T=r.uniform(40, 90), E=np.array([r.uniform(0, W), r.uniform(0, H)]), va=r.normal(0, 30, 2), drift=r.uniform(.05, .25), R=r.uniform(.8, 2.4), ph=r.uniform(0, 6.28), wob=r.uniform(2, 8), near=False))
        return B

    def bubble_pos(s, b, f):
        t = f - b['t0']
        if t < 0 or t > b['T']: return None
        u = t / b['T']
        if b.get('rise'): e = u ** .8; p = b['E'] * (1 - e) + (C0 + (b['E'] - [960, 1150]) * .35) * e; return p + [math.sin(t * .42 + b['ph']) * b['wob'] * (1 - .6 * e), 0], b['R'] * (1 - .78 * e), 1.
        p = b['E'] + b['va'] * (1 - math.exp(-t / 4.5)); p = p + (C0 - p) * b['drift'] * u ** 1.3; p = p + np.array([math.sin(t * .5 + b['ph']), math.cos(t * .43 + b['ph'])]) * b['wob'] * min(1, t / 6.)
        return p, b['R'] * (.55 + .45 * min(1, t / 2.)) * (1 - .45 * u), min(1, (1 - u) * 4)

    def draw_bubbles(s, f, E):
        c = Cel(ss=1); x = c.ctx; hi = mix(rgb('#9fd8ea'), rgb('#fff6da'), E)
        for b in s.bub:
            o = s.bubble_pos(b, f)
            if o is None: continue
            (px, py), R, al = o
            if R < .5 or not (-40 < px < W + 40 and -40 < py < H + 40): continue
            if R < 2.6: x.arc(px, py, R, 0, 6.2832); x.set_source_rgba(*[float(v) for v in hi], .8 * al); x.fill(); continue
            sq = 1 + .10 * math.sin(f * .7 + b['ph']); x.save(); x.translate(px, py); x.scale(sq, 1 / sq)
            x.arc(0, 0, R, 0, 6.2832); x.set_source_rgba(.62, .88, .95, .16 * al); x.fill()
            x.arc(0, 0, R * .96, 0, 6.2832); x.set_line_width(max(1., R * .10)); x.set_source_rgba(*[float(v) for v in hi], .85 * al); x.stroke()
            d = unit2(C0 - [px, py]); a = math.atan2(d[1], d[0]); x.arc(0, 0, R * .74, a - .75, a + .75); x.set_line_width(max(1., R * .20)); x.set_source_rgba(1, 1, 1, .95 * al); x.stroke()
            x.arc(0, 0, R * .82, a + 2.3, a + 4.0); x.set_line_width(max(.8, R * .09)); x.set_source_rgba(.05, .16, .30, .55 * al); x.stroke(); x.arc(-d[0] * R * .35, -d[1] * R * .35, R * .10, 0, 6.2832); x.set_source_rgba(1, 1, 1, .8 * al); x.fill(); x.restore()
        return c.layer()

    def crown(s, c, n, f, E):
        """the silver crown and air sheath of one entry"""
        t = f - TK[n]
        if not 0 <= t < 13: return
        sp = SPEC[n]; h, d = SA.dirs(sp); e0 = s.d(n, 'i0')['E']; q = sp['s']; r = np.random.default_rng(500 + sp['k']); nn = np.array([-d[1], d[0]]); ra = q * (dict(jinbe=.80, franky=1.0, zoro=.62, chopper=.46, brook=.42).get(n, .46)); gr = (.75, 1.0, 1.2, 1.36, 1.48, 1.58, 1.66, 1.72, 1.78, 1.82, 1.86, 1.9, 1.92)[t]; M = 18
        W1, W2 = mix(rgb('#cfeaf2'), rgb('#fffdf2'), E), mix(rgb('#5f9dbb'), rgb('#9fd6e4'), E)
        M = 40
        for pas, col in ((0, '#103a58'), (1, W2), (2, W1)):
            for i in range(M):
                rr = np.random.default_rng(900 + i + 37 * sp['k']); lo = rr.uniform(.45, 1.5); a = 2 * math.pi * (i + rr.uniform(-.4, .4)) / M; sz = q * .088 * lo ** 1.4 * max(0., 1 - (t / 12.) ** 1.2 * rr.uniform(.75, 1.25))
                if sz < 1.2: continue
                gq = gr * rr.uniform(.80, 1.12) * .82; p = e0 + nn * math.cos(a) * ra * gq + d * (math.sin(a) * ra * gq * .5 + t * 3.5 * lo); k = np.linspace(0, 6.283, 12)[:-1]; R_ = sz * (1 + .08 * np.sin(k * 2 + i + t)); off = d * sz * .22 if pas == 2 else d * 0; R2 = R_ * (1.18, 1.0, .62)[pas]
                c.fill(spl(np.c_[p[0] + off[0] + np.cos(k) * R2, p[1] + off[1] + np.sin(k) * R2], True, n=3), col, (.6, 1, 1)[pas])
            if t < 8:                                                                        # sheath streaming back along the forearm
                for i in range(16):
                    rr = np.random.default_rng(1400 + i + 41 * sp['k']); v = rr.uniform(0, 1); side = rr.choice([-1, 1]); ln = q * rr.uniform(.18, .5) * (1 - t / 8.); sz = q * rr.uniform(.04, .10) * (1 - (t / 8.) ** 2); p = e0 + d * (v * q * (1.5 - .05 * t)) + nn * side * ra * rr.uniform(.55, 1.05) * (1 + .03 * t); T_ = np.array([[p - d * ln], [p + nn * sz], [p + d * ln * .5], [p - nn * sz]])[:, 0]
                    if pas > 0: c.fill(spl(T_ if pas == 1 else T_ * .55 + p * .45, True, n=4), col, .9)

    def meniscus(s, c, n, f, E, M):
        """the bright wet line where the arm meets the surface"""
        t = f - TK[n]; e0 = M[:, :2] @ s.d(n, 'i0')['E'] + M[:, 2]; h, d = SA.dirs(SPEC[n]); nn = np.array([-d[1], d[0]]); q = SPEC[n]['s']; ra = q * dict(jinbe=.66, franky=.84, zoro=.50, chopper=.40, brook=.36, robin=.30).get(n, .36); hi = mix(rgb('#9fd8ea'), rgb('#fff3d0'), E)
        for j, (g, w, al) in enumerate(((1.0, 8.0, .42), (1.22, 3.4, .22))):
            a = np.linspace(0, 2 * math.pi, 60); wb = 1 + .035 * np.sin(a * 3 + f * .23 + j) + .03 * np.sin(a * 5 - f * .31 + SPEC[n]['k']); P_ = e0 + nn * (np.cos(a) * ra * g * wb)[:, None] + d * (np.sin(a) * ra * g * .46 * wb)[:, None]
            for k in range(3): i0 = (k * 20 + int(f * .4) * (1 if j % 2 else -1) + SPEC[n]['k'] * 5) % 60; ii = np.arange(i0, i0 + (17 if j == 0 else 11)) % 60; c.line(P_[ii], w * (.75 + .25 * math.sin(f * .3 + k)), hi, taper=(.35, .35), seed=k + j, alpha=al)

    # ------------------------------------------------------------- cloth
    def tail(s, c, p, base, L, w, f, t, seed, col='#ece7dc', sh='#8a90b2', kick=0.):
        n = 16; u = np.linspace(0, 1, n); th = base + (.25 + .95 * u) * .62 * np.sin(f / 24. * 3.4 - u * 4.2 + seed * 2.1) + u * .5 * math.sin(f / 24. * 1.3 + seed) + kick * u ** .6 * math.exp(-t / 7.); st = L / (n - 1); pts = np.vstack([[p], p + np.cumsum(np.c_[np.cos(th), np.sin(th)] * st, 0)[:-1]])
        T_ = np.gradient(pts, axis=0); T_ /= np.hypot(*T_.T)[:, None] + 1e-6; N_ = np.c_[-T_[:, 1], T_[:, 0]]; tw = np.cos(u * 5.0 - f / 24. * 2.6 + seed * 1.7); hw = w * (.38 + .62 * np.abs(tw)) * .5 * (1 - .15 * u)
        for i in range(n - 1): q = np.array([pts[i] + N_[i] * hw[i], pts[i + 1] + N_[i + 1] * hw[i + 1], pts[i + 1] - N_[i + 1] * hw[i + 1], pts[i] - N_[i] * hw[i]]); cc = col if tw[i] > 0 else sh; c.fill(q, cc); c._path(q); c._src(cc, 1.); c.ctx.set_line_width(1.); c.ctx.stroke()
        A, B = pts + N_ * hw[:, None], pts - N_ * hw[:, None]; nt = np.array([A[-1], pts[-1] - T_[-1] * w * .35, B[-1]])
        c.line(A, 2.0, SA.LINE, taper=(.1, .1), seed=seed); c.line(B, 2.4, SA.LINE, taper=(.1, .1), seed=seed + 1); c.line(nt, 1.8, SA.LINE, taper=(.1, .1), seed=seed + 2); c.line(pts[2:-3] + N_[2:-3] * hw[2:-3, None] * .3, 1.2, '#6d7398', taper=(.3, .6), seed=seed + 3)

    # ------------------------------------------------------------- arms
    def key(s, n, f):
        t = f - TK[n]
        if t < 0: return None
        if f >= 145: return 'g0' if f < 148 else 'g1'
        if f >= 142: return 'a1'
        if f >= 139 and t > 5: return 'a0'
        if t < 3: return 'e%d' % t
        if t < 6: return 'e3'
        if t < 9: return 'e4'
        if t < 12: return 'e5'
        if t < 15: return 'e6'
        return 'i%d' % (((t - 15) // 3) % 8)

    def xform(s, n, f):
        """2x3 of the sprite at frame f: ring contraction on every beat, slow drift, the lunge"""
        sp = SPEC[n]; h, d = SA.dirs(sp); t = f - TK[n]; E = s.d(n, 'i0')['E']; pu = sum(pulse(f - TK[m]) for m in ORDER if TK[m] > TK[n]) if f < 139 else 0.; pu = min(pu, 1.1)
        rot = math.radians(.9) * math.sin(f / 24. * 1.7 + sp['k'] * 1.9) * min(1, t / 12.); sc = 1 + .022 * pu; off = d * 15 * pu + np.array([math.sin(f / 24. * 1.1 + sp['k']), math.cos(f / 24. * .9 + sp['k'] * 2)]) * 3 * min(1, max(0, t - 8) / 12.)
        if n == 'jinbe' and f < 145: off = off - d * 30.
        if 3 <= t < 24 and f < 139: off = off - d * (10 * sp['s'] / 150.) * (1 - math.exp(-(t - 3) / 5.)) + d * 10 * sp['s'] / 150. * smooth(12, 24, t) * .6
        if f >= 139 and f < 145: g = np.array(GRIP[n]['h'], float) - h; e = {139: -.05, 140: -.09, 141: -.10, 142: .10, 143: .42, 144: .84}[f]; off = off * (0 if f >= 142 else 1) + g * e; rot *= .3
        if f >= 145: off = np.array(shk(f)) * 0; rot = 0; sc = 1.; E = np.array([960., 560.])
        return mat(E, rot, sc, off)

    def frame(s, f):
        E = energy(f); beat = min(1.2, sum(pulse(f - TK[n]) for n in ORDER)); hold = f
        if f < 15: E = -0.0; dim = 1 - .30 * smooth(0, 14, f)
        else: dim = 1.
        img, (sx, sy) = s.water(f, E, beat); img *= dim
        R = s.rays(f, E, beat); img = img + R
        # halo of scattered light round the window
        rr = np.hypot(*(np.stack(grid()) - C0[:, None, None])); img = img + (np.exp(-(rr / 420.) ** 2) * (.05 + .06 * E + .05 * beat))[..., None] * mix(rgb('#2f7fa6'), rgb('#ffe2a0'), E)
        img = img + (np.exp(-(rr / 150.) ** 2) * (.05 + .30 * E + .10 * beat) + np.exp(-(rr / 300.) ** 2) * .12 * E)[..., None] * rgb('#ffe9a8')
        img = img * (1 - s.silm * .72) + s.silm * .72 * mix(rgb('#081a30'), rgb('#1c5878'), E)
        img = s.hat(img, f, E)
        # shadow blots just before an arm arrives, reflections in the underside, rings
        rc = Cel(ss=1)
        for n in ORDER:
            t = f - TK[n]; D0 = s.d(n, 'i0'); e0 = D0['E']
            if -5 <= t < 0: g = radial(e0[0], e0[1], SPEC[n]['s'] * (1.6 + t * .12), power=1.2); img = img * (1 - .5 * (1 + t / 5.) * g[..., None])
            if 0 <= t < 16:
                for j in range(3):
                    q = t - j * 3
                    if q < 0: continue
                    rad = SPEC[n]['s'] * (.35 + .11 * q + .0 * j); a0 = j * 1.7 + SPEC[n]['k']
                    for seg in range(5):
                        a = np.linspace(a0 + seg * 1.256, a0 + seg * 1.256 + .85 * (1 - t / 18.), 16); P_ = np.c_[e0[0] + np.cos(a) * rad, e0[1] + np.sin(a) * rad * .92]; rc.line(P_, 3.4 * (1 - t / 16.) + .6, mix(rgb('#a9dcea'), rgb('#fff3cf'), E), taper=(.4, .4), seed=seg + j, alpha=.75 * (1 - t / 16.))
        img = over(img, rc.layer())
        order = [n for n in ('chopper', 'robin', 'brook', 'usopp', 'sanji', 'nami', 'zoro', 'jinbe', 'franky') if f >= TK[n]] if f < 145 else ['robin', 'jinbe', 'chopper', 'brook', 'sanji', 'usopp', 'nami', 'zoro', 'franky', 'luffy']
        if 119 <= f < 145: order = order + ['luffy']
        for n in ORDER:                                                                      # each arrival lights its own water
            t = f - TK[n]
            if 0 <= t < 26 and f < 143: h_, d_ = SA.dirs(SPEC[n]); g = radial(h_[0] - d_[0] * 60, h_[1] - d_[1] * 60, SPEC[n]['s'] * 3.0, power=2.); img = img + (g * .30 * math.exp(-t / 9.) * min(1, (t + 1) / 2.))[..., None] * mix(rgb(SPEC[n]['tint']), rgb('#bfefff'), .25)
        if f >= 145: img = img * (.50 + .10 * (f - 145))                                     # the light dips behind the clasp so the hands read
        arms = np.zeros((H, W, 4), f32); cb = Cel(ss=1); cr = Cel(ss=1)
        for n in order:
            if n == 'luffy':
                lk = max(k for k in LKEYS if k <= max(120, f)); D = s.d('luffy', 'f%d' % lk); yv = float(lerp_keys(f, LPATH, lambda t: t)); M = mat(np.array([960., 560.]), math.radians(1.4) * math.sin(f * .21), 1., np.array([10 * math.sin(f * .16), yv - 560.])); ms = [M]
                if f in (144,): ms = [mat(np.array([960., 560.]), 0, 1., np.array([0, yv - 560. + o])) for o in (40, 26, 12, 0)]
            else:
                k = s.key(n, f); D = s.d(n, k); ms = [s.xform(n, f)]
                if f in (143, 144): ms = [lerpm(s.xform(n, f - 1), s.xform(n, f), v) for v in np.linspace(.25, 1., 9)]
                if f - TK[n] in (0, 1): d = unit2(C0 - np.array(SPEC[n]['h'])); ms = [ms[0] + np.array([[0, 0, -d[0] * o], [0, 0, -d[1] * o]]) for o in (30, 16, 0)]
            lay = np.zeros((H, W, 4), f32); im = D['im'].astype(f32) / 255.; tg = f - TK.get(n, 999)
            if 0 <= tg < 30 and f < 143: im = im.copy(); im[..., :3] = np.minimum(im[..., :3] * (1 + .34 * math.exp(-tg / 9.)), im[..., 3:4] * 1.08)
            for j, M in enumerate(ms):
                Q = M.copy(); Q[:, 2] += M[:, :2] @ np.array([float(D['ox']), float(D['oy'])]); lay += cv2.warpAffine(im, Q.astype(f32), (W, H), flags=cv2.INTER_LINEAR) * ((.07, .08, .09, .10, .11, .12, .13, .14, .16)[j] if len(ms) == 9 else (.16, .22, .30, .32)[j] if len(ms) == 4 else (.25, .30, .45)[j] if len(ms) == 3 else 1.)
            if f < 145 and n not in ('luffy', 'robin'):                                                     # its mirror image in the underside of the surface
                e0 = s.d(n, 'i0')['E']; d = unit2(C0 - np.array(SPEC[n]['h'])); Rf = np.eye(2) - 2 * np.outer(d, d); M = ms[-1]; Q = np.c_[Rf @ M[:, :2], Rf @ (M[:, 2] - e0) + e0]; Q[:, 2] += Q[:, :2] @ np.array([float(D['ox']), float(D['oy'])]); rf = cv2.warpAffine(im, Q.astype(f32), (W, H), flags=cv2.INTER_LINEAR); yy_, xx_ = grid()[::-1]; dd_ = (xx_ - e0[0]) * -d[0] + (yy_ - e0[1]) * -d[1]; rf = cv2.GaussianBlur(rf, (0, 0), 2.0) * (.78 * np.exp(-np.maximum(dd_, 0) / (SPEC[n]['s'] * .9)))[..., None]; arms_under = rf[..., 3:4]; img = img * (1 - arms_under) + rf[..., :3] * rgb('#5f9cc8') * .75
            # cloth tails behind this arm's own body
            if n in SPEC and f < 148:
                tc = Cel(ss=2); M = ms[-1]; kp = M[:, :2] @ D['knot'] + M[:, 2]; t = f - TK.get(n, 118); sz = float(D['s']); d = D['d']; back = math.atan2(-d[1], -d[0]); sd = SPEC[n]['k'] if n in SPEC else 12; sgn = 1 if sd % 2 else -1
                bs = back + sgn * (.35 + .75 * (1 - math.exp(-max(0, t) / 10.))); L_ = sz * (.95 if n != 'luffy' else 2.1); s.tail(tc, kp, bs, L_, sz * .17, f, t, sd, kick=-sgn * .9); s.tail(tc, kp, bs + sgn * .5, L_ * .68, sz * .15, f + 9, t, sd + 20, kick=-sgn * .6); tl = tc.layer()
                tl = kit.para(tl, kp, kp + d * sz, [(0, '#b9c8e4'), (1, '#ffffff')], 'mul', .8); tl = kit.rim(tl, -d[0] * 2.4, -d[1] * 2.4, '#fff6dc', .9, .6); lay = tl * (1 - lay[..., 3:4]) + lay if n != 'luffy' else lay * (1 - tl[..., 3:4]) + tl
            a = lay[..., 3:4]; sh = blur(lay[..., 3], 16)[..., None] * .55; arms[..., :3] = (arms[..., :3] * (1 - sh * rgb('#ffffff') * .8)) * (1 - a) + lay[..., :3]; arms[..., 3:4] = arms[..., 3:4] * (1 - a) + a
            if n in SPEC and f < 145: s.crown(cr, n, f, E); s.meniscus(cr, n, f, E, ms[-1])
        # light on the arms: cold when alone, colour comes back with each friend; caustics crawl
        aa = arms[..., 3:4]; lumA = arms[..., :3] @ np.array([.3, .59, .11], f32); coldA = lumA[..., None] * rgb('#8fb9e6') * 1.05; w = .74 + .26 * E; col = coldA * (1 - w) + arms[..., :3] * w; col = col * (.80 + .24 * E + .12 * beat)
        cau = s.caustic(f, sx, sy)[..., None]; nearc = np.exp(-(rr / 620.) ** 2)[..., None]; cs = cau * aa * (.20 + .30 * E) * (.35 + .65 * nearc); col = col + cs * mix(rgb('#8fe0f2'), rgb('#fff2c4'), E) * (.35 + .65 * np.clip(lumA[..., None] * 1.6, 0, 1))
        s.aa, s.ac = aa, arms[..., :3]; img = img * (1 - aa) + col; img = over(img, cr.layer())
        img = img + R * .16 + (np.exp(-(rr / 300.) ** 2) * (.02 + .03 * E))[..., None] * rgb('#ffe9b0')
        img = over(img, s.petals(f, E)) if TK['robin'] - 2 <= f < 145 else img
        img = over(img, s.draw_bubbles(f, E))
        # he is sinking: the ring recedes; every beat pulls; at the grip we are hauled up
        z = float(lerp_keys(f, [(0, 1.075), (118, 1.012), (140, 1.004), (143, 1.0), (145, 1.02), (149, 1.10)])) + .006 * beat; q = kit.shake(f, 5, 3, 3., 145) if f >= 145 else sum((kit.shake(f, 2.6 * SPEC[n]['s'] / 150., SPEC[n]['k'], 2.2, TK[n]) for n in ORDER if 0 <= f - TK[n] < 6), np.zeros(2)); rot = math.radians(1.1) * math.sin(f / 150. * 2.4 - .6)
        img = comp.cam2d(img, 960 + (1 - 1 / z) * 0 + q[0], 486 + 54 / z * 1.0 + q[1], z, rot, interp=cv2.INTER_CUBIC)
        if f >= 145: img = s.finale(img, f)
        return np.maximum(img, 0).astype(f32)

    def caustic(s, f, sx, sy):
        a = FX.noise_img(H, W, 92, f / 24. * .55, 5, 2, 4); b = FX.noise_img(H, W, 60, f / 24. * .7, 9, 2, 4); v = 1 - np.abs(a + .4 * b) * 4.2; u = 1 - np.abs(b - .35 * a) * 4.6; return np.clip(smooth(.45, .72, v) * .9 + smooth(.55, .8, u) * .6, 0, 1.2)

    def petals(s, f, E):
        c = Cel(ss=1); r = np.random.default_rng(61); tip = s.d('robin', 'i0')['palm']
        for i in range(34):
            t0 = TK['robin'] + (i % 3) * 2 + r.integers(0, 3); t = f - t0; v = r.normal(0, 1, 2) * [120, 80] + [0, 20]; ph = r.uniform(0, 6.28); sz = r.uniform(5, 13)
            if t < 0: continue
            p = np.array([tip[0] + r.normal(0, 40), tip[1] - 40 + r.normal(0, 60)]) + v * (1 - math.exp(-t / 9.)) + [math.sin(t * .13 + ph) * 12, t * .25]; a = ph + t * .12; sq = abs(math.cos(t * .17 + ph)) * .8 + .2; al = max(0, 1 - t / 55.); u = np.linspace(0, 6.283, 13)[:-1]; Q = np.c_[np.cos(u) * sz, np.sin(u) * sz * .55 * sq]; Q = Q @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + p
            c.fill(Q, '#f7b6cc' if i % 2 else '#ffd9e2', al); c.line(np.vstack([Q, Q[:1]]), 1.2, '#8a4a78', alpha=al * .8, taper=(0, 0))
        return c.layer()

    def hat(s, img, f, E):
        """the straw hat afloat on the light, seen from below"""
        t = f / 24.; cx, cy = 985 + 14 * math.sin(t * .8), 462 + 9 * math.cos(t * .7); a = .5 * math.sin(t * .5) + .3; k = 1 + .05 * math.sin(t * 1.7); R = 62 * k; n = 64; u = np.linspace(0, 6.2832, n, endpoint=False); r = np.random.default_rng(5); wv = 1 + .035 * np.sin(u * 3 + 1 + t) + .025 * np.sin(u * 5 + 2) + .018 * r.normal(0, 1, n)
        rot = lambda P_: np.asarray(P_) @ np.array([[math.cos(a), math.sin(a)], [-math.sin(a), math.cos(a)]]) + [cx, cy]; c = Cel(ss=2, w=260, h=260, ox=int(cx - 130), oy=int(cy - 130)); br = rot(np.c_[np.cos(u) * R * wv, np.sin(u) * R * wv * .86]); c.fill(br, mix(rgb('#3a3a52'), rgb('#b9782c'), E))
        for k_ in range(5): q = rot(np.c_[np.cos(u) * R * (.55 + .085 * k_), np.sin(u) * R * .86 * (.55 + .085 * k_)]); c.line(np.vstack([q, q[:1]]), 1.1, mix(rgb('#22243c'), rgb('#7a4418'), E), taper=(0, 0), seed=k_)
        for k_ in range(0, n, 2): v = u[k_]; c.line(rot([[math.cos(v) * R * .5, math.sin(v) * R * .43], [math.cos(v + .12) * R * wv[k_] * .98, math.sin(v + .12) * R * .84 * wv[k_]]]), .9, mix(rgb('#22243c'), rgb('#8a5220'), E), taper=(.2, .3), seed=k_)
        c.fill(rot(np.c_[np.cos(u) * R * .50, np.sin(u) * R * .43]), mix(rgb('#1a1a30'), rgb('#5d2f18'), E)); c.fill(rot(np.c_[np.cos(u) * R * .42 + 3, np.sin(u) * R * .35 - 3]), mix(rgb('#252844'), rgb('#8a5526'), E)); q = rot(np.c_[np.cos(u) * R * .50, np.sin(u) * R * .43]); c.line(np.vstack([q, q[:1]]), 4.6, mix(rgb('#3a1c34'), rgb('#a51f26'), E), taper=(0, 0)); kit.outline(c, br, 2.2, SA.LINE, 3, breaks=3)
        r2 = np.random.default_rng(8)
        for _ in range(26): v, q_ = r2.uniform(0, 6.28), r2.uniform(.1, .92); p = rot([[math.cos(v) * R * q_, math.sin(v) * R * q_ * .86]])[0]; w = r2.uniform(.7, 1.7); c.fill(p + np.array([[-w, 0], [0, -w], [w, 0], [0, w]]), '#fff6d0', .5 + .5 * E)
        L = c.rgba(); L = kit.rim(L, -3, -3, '#fff0c0', .8 * (.4 + .6 * E), .6); return over_roi(img, L, c.ox, c.oy, .96)

    def finale(s, img, f):
        x, y = grid(); r = np.hypot(x - 960, y - 540); th = np.arctan2(y - 540, x - 960); a = np.linspace(-math.pi, math.pi, 1441); g = FX.noise(np.cos(a) * 9, np.sin(a) * 9, 2. + (f - 145) * .25, 4); st = np.interp(th, a, np.clip(g * 1.7 + .25, 0, 1) ** 1.3).astype(f32)
        if f == 145:                                                                        # hit drawing: indigo, cream, one vermilion
            M = lambda a_: cv2.warpAffine(a_, np.float32([[1.02, 0, 960 - 1.02 * 960], [0, 1.02, 540 - 1.02 * 540]]), (W, H)); aa = M(s.aa[..., 0]); L = M(cv2.GaussianBlur(s.ac @ np.array([.3, .59, .11], f32), (0, 0), .8)); m = smooth(.24, .28, L)[..., None]; m2 = smooth(.50, .54, L)[..., None]; o = rgb('#150d30') * (1 - m) + rgb('#d8452a') * m; o = o * (1 - m2) + rgb('#fff3d2') * m2
            bg = np.ones((H, W, 3), f32) * rgb('#1a1140'); g = np.exp(-(r / 420.) ** 2)[..., None]; bg = bg * (1 - g) + rgb('#fff3d2') * g; c = Cel(ss=1); kit.focus_lines(c, 960, 540, 300, 1500, 170, 11, '#fff3d2', 5, ry=.8); bg = over(bg, c.layer()); ed = cv2.dilate(aa, np.ones((9, 9), np.uint8))[..., None]; bg = bg * (1 - ed) + rgb('#150d30') * ed; a3 = aa[..., None]; return bg * (1 - a3) + o * a3
        k = (f - 145) / 4.; img = img * (1 + .25 * k) + (np.exp(-(r / (120 + 420 * k)) ** 2) * (.0 + .55 * k * k) + st * smooth(120, 420, r) * np.exp(-r / (420 + 620 * k)) * (.30 + .6 * k))[..., None] * rgb('#fff0c8')
        if f >= 148: R0 = {148: 640, 149: 1500}[f]; e = smooth(R0 * 1.0, R0 * .72, r * (1 - .30 * st)); img = img * (1 - e[..., None]) + rgb('#fff8e6') * 1.08 * e[..., None]
        return img


def shk(f): return kit.shake(f, 3, 7, 3., 145)


def mat(p, rot, sc, off):
    c, s_ = math.cos(rot) * sc, math.sin(rot) * sc; A = np.array([[c, -s_], [s_, c]]); t = p - A @ p + off; return np.c_[A, t]


def lerpm(A, B, t): return A * (1 - t) + B * t
