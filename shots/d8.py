"""d8  The sound is gone.  Red beads -> the face empties -> the scream -> silence: the drawing comes apart -> one bead."""
import numpy as np, cv2, math, os
from op import W, H
from op.paint import rgb, f32, smooth, grid, fbm, ramp, repaint, mix
from op.comp import over, screen, blur, lerp_keys, ease, radial, cam2d
from op.ink import Cel, spl, resample, noise1
from op.kit import focus_lines, impact, shake, smear, zoomblur, drybrush, rim
from op import fx_fire as FF, fx_core as FC, hat as HT
from chars import luffy as L

TAU = 2 * math.pi
NB = 11                                                # beads on the string
HERO = 5


def cached(name, fn):
    p = f'cache/{name}.npy'
    if os.path.exists(p): return np.load(p, mmap_mode='r').astype(f32)
    a = fn(); np.save(p, a.astype(np.float16)); return a


def fire_plate():
    """a burning field after the battle, soft focus: smoke masses, fire walls left and right, broken ice"""
    x, y = grid(); n = fbm(H, W, (230, 96, 40, 16), seed=11, aspect=1.5); n2 = fbm(H, W, (120, 50, 20), seed=12, aspect=1.2)
    g = ramp(np.clip(y / H + .06 * n, 0, 1), [(0, '#140a12'), (.35, '#2c1017'), (.62, '#5c1f18'), (.80, '#7a2c18'), (1, '#1c0a0e')])
    fr = np.exp(-((x - 1700) / 520) ** 2 - ((y - 560) / 460) ** 2) + .55 * np.exp(-((x - 120) / 360) ** 2 - ((y - 700) / 330) ** 2)
    g = g + (fr * smooth(-.6, 1.4, n2 + fr))[..., None] * rgb('#ff7a22') * .85 + (fr ** 2.2)[..., None] * rgb('#ffd070') * .55
    sm = smooth(.25, .55, n + .35 * np.sin(x / 260 + y / 170)); g = mix(g, rgb('#1b0c13') * np.ones_like(g), (sm * .78 * smooth(-.2, .9, 1 - fr))[..., None])
    ed = smooth(.12, .3, n + .35 * np.sin(x / 260 + y / 170)) - sm; g = g + (np.clip(ed, 0, 1) * (.25 + fr))[..., None] * rgb('#d0541e') * .6
    g = repaint(np.clip(g, 0, 1.2), (44, 22, 11), seed=4)
    c = Cel(); r = np.random.default_rng(6)
    for x0, w, h in ((-60, 420, 430), (300, 260, 250), (1280, 300, 300), (1500, 520, 520), (860, 180, 150)):
        P = [(x0, 1100), (x0 + w * .06, 900 - h * .5), (x0 + w * .22, 900 - h * .62), (x0 + w * .30, 900 - h), (x0 + w * .46, 900 - h * .86), (x0 + w * .58, 900 - h * .40), (x0 + w * .80, 900 - h * .5), (x0 + w * .9, 900 - h * .2), (x0 + w, 1100)]
        c.fill(np.array(P), '#1a0b10'); c.fill(np.array(P[2:5] + [(x0 + w * .40, 900 - h * .3), (x0 + w * .3, 900 - h * .2)]), '#6a2416')
    g = over(g, rim(c.layer(), -6, -3, '#ff9a3c', .9, 1.0)); return blur(g.astype(f32), 9)


def env_map():
    """what every bead sees: the fire, the dark ground, a boy on his knees holding someone"""
    s = 512; y, x = np.mgrid[0:s, 0:s].astype(f32) / s; n = fbm(s, s, (90, 34, 12), seed=3)
    g = ramp(np.clip(y * 1.02 + .05 * n, 0, 1), [(0, '#2a0d14'), (.30, '#7a2016'), (.52, '#ff7a28'), (.60, '#ffe09a'), (.615, '#2a0a0c'), (1, '#0c0406')])
    sm = smooth(.5, .9, n) * (y < .55); g = g * (1 - .6 * sm[..., None]); c = Cel(s, s); k = '#12060a'
    for i in range(16): xx = 20 + i * 32; h = 30 + 44 * abs(math.sin(i * 2.3)); c.fill(spl([(xx - 16, 312), (xx - 10, 300 - h * .5), (xx + 3, 300 - h), (xx + 7, 300 - h * .5), (xx + 16, 312)]), '#ffb040' if i % 2 else '#ff8a2c')
    c.fill(spl([(196, 330), (206, 268), (232, 232), (262, 226), (284, 250), (296, 300), (330, 318), (332, 332)], False), k); c.fill(spl([(238, 236), (228, 214), (238, 196), (258, 194), (270, 208), (266, 230)], True), k)
    c.fill(spl([(250, 290), (292, 262), (340, 268), (372, 296), (380, 330), (250, 332)], True), k); c.fill(spl([(300, 262), (292, 246), (304, 234), (322, 238), (326, 254), (314, 266)], True), k)
    for xx in (60, 110, 430, 470): c.fill(np.array([(xx - 5, 318), (xx - 3, 286), (xx + 3, 286), (xx + 6, 318)]), k); c.fill(spl([(xx - 4, 286), (xx, 278), (xx + 5, 286)], True), k)
    q = c.rgba(); return (g * (1 - q[..., 3:4]) + q[..., :3]).astype(f32)


def bead(cx, cy, r, env, rot, hole=None, clear=0., face=None, fk=0.):
    """one lacquered bead -> (x0, y0, premult RGBA).  clear 0..1 removes the red; face = full frame picture seen in it (hero)"""
    x0, y0, x1, y1 = int(max(-40, cx - r - 4)), int(max(-40, cy - r - 4)), int(min(W + 40, cx + r + 5)), int(min(H + 40, cy + r + 5))
    if x1 - x0 < 3 or y1 - y0 < 3: return None
    cx, cy, r = float(cx), float(cy), float(r); Y, X = np.mgrid[y0:y1, x0:x1].astype(f32); nx, ny = (X - cx) / r, (Y - cy) / r; d2 = nx * nx + ny * ny; d = np.sqrt(d2); a = np.clip((1 - d) * r + .5, 0, 1)
    nz = np.sqrt(np.clip(1 - d2, 0, 1)); dif = .62 * nx - .10 * ny + .62 * nz
    col = ramp(np.clip(dif, -1, 1), [(-1, '#22030a'), (-.12, '#3d050d'), (-.10, '#6f0a14'), (.42, '#a30f1a'), (.46, '#c9161f'), (1, '#e6352f')])
    if face is None:
        c, s = math.cos(rot), math.sin(rot); k = 205 * (.62 + .55 * d2); u = 256 + (nx * c - ny * s * .0 + .25 * s) * k; v = 262 + (ny + .10 * math.sin(rot * 1.7)) * k
        q = cv2.remap(env, np.clip(u, 0, 511).astype(f32), np.clip(v, 0, 511).astype(f32), cv2.INTER_LINEAR); fr = (.26 + .66 * d2 ** 1.6)[..., None]; col = col * (1 - .55 * fr) + q * fr * .85 * (rgb('#ff8d7a') * .6 + .4) + (q.max(-1, keepdims=True) > .72) * q * .18
    else:
        k = r * (1 - fk) * .62 + r * fk; b = 1 + (1 - fk) * .9 * d2; u = 960 + nx * b * (430 * (1 - fk) + k * fk); v = 500 + ny * b * (430 * (1 - fk) + k * fk)
        u = u * (1 - fk) + fk * (960 + (X - cx) * (1 + 0 * d2)); v = v * (1 - fk) + fk * (540 + (Y - cy))
        q = cv2.remap(face.astype(f32), u.astype(f32), v.astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); t = rgb('#ff5a4c') * (1 - clear) + clear
        edge = (smooth(.55, 1., d) * (1 - clear))[..., None]; col = q * t * (1 - edge * .75) + col * edge * .75
    rimm = smooth(.80, .93, d) * np.clip(nx * .85 - ny * .35, 0, 1) * (1 - clear); col = col * (1 - rimm[..., None] * .85) + rgb('#ffc46a') * rimm[..., None] * .85
    if hole is not None:
        hx, hy, hs = hole; m = np.clip((1 - np.sqrt(((nx - hx) / (.17 * hs + .02)) ** 2 + ((ny - hy) / .17) ** 2)) * 6, 0, 1)[..., None]; col = col * (1 - m) + rgb('#1a0308') * m
    sp = np.clip((1 - np.sqrt(((nx + .36) * .82 + (ny + .42) * .57) ** 2 / .19 ** 2 + (-(nx + .36) * .57 + (ny + .42) * .82) ** 2 / .10 ** 2)) * 7, 0, 1) + .8 * np.clip((1 - np.hypot(nx + .13, ny + .60) / .05) * 4, 0, 1)
    col = col + (sp * max(0, 1 - clear * 2.2))[..., None] * rgb('#fff2e6') * .95
    ol = np.clip((d - (1 - max(1.6, r * .035) / r)) * r, 0, 1) * (1 - clear); col = col * (1 - ol[..., None] * .9) + rgb('#19040a') * ol[..., None] * .9
    return x0, y0, np.dstack([col * a[..., None], a]).astype(f32)


def lay(img, q, sig=0.):
    if q is None: return img
    x0, y0, p = q
    if sig > .5: pd = int(sig * 2.6); p = cv2.copyMakeBorder(p, pd, pd, pd, pd, cv2.BORDER_CONSTANT); x0 -= pd; y0 -= pd; p = cv2.GaussianBlur(p, (0, 0), sig)
    xa, ya, xb, yb = max(0, x0), max(0, y0), min(W, x0 + p.shape[1]), min(H, y0 + p.shape[0])
    if xb > xa and yb > ya: s = p[ya - y0:yb - y0, xa - x0:xb - x0]; img[ya:yb, xa:xb] = img[ya:yb, xa:xb] * (1 - s[..., 3:4]) + s[..., :3]
    return img


# ---- charcoal timeline: ten drawings on threes
SS = 358.
PIT = [.24, .27, .22, .25, .17, .05, -.10, -.24, -.31, -.34]; CY = [404, 398, 410, 402, 420, 452, 500, 545, 566, 574]; EMP = [0, 0, 0, .05, .16, .36, .58, .78, .88, .92]
LOO = [.03, .05, .09, .16, .27, .40, .54, .68, .82, .94]; FIL = [1, 1, 1, .92, .76, .56, .34, .12, 0, 0]; RED = [1, 1, 1, 1, .97, .90, .76, .56, .34, .12]; DET = [1, 1, 1, .95, .85, .70, .52, .34, .16, 0.]; CON = [0, 0, 0, .10, .30, .55, .80, 1, 1, .9]; LIN = [1, 1, 1, 1, 1, 1, .95, .85, .60, .30]


class Shot:
    id = 'd8'; n = 90; finish = dict(vig=.14)
    EVENTS = [(0, 'beads_tension', dict(size=.5, note='string of red beads under load, tiny creak, fire far behind')), (5, 'string_fray', dict(size=.3)), (8, 'string_snap', dict(size=.9, note='dry crack, then time slows')),
              (9, 'beads_fly', dict(dur=20, size=.6, note='slow motion glass-wood clicks, one by one, reverberant')), (14, 'bead_click', dict(size=.3)), (19, 'bead_click', dict(size=.3)), (24, 'bead_click', dict(size=.25)), (25, 'bead_rush', dict(size=.7, note='hero bead fills the lens by f29')),
              (30, 'face_cut', dict(size=.2, note='fire muffled, breath')), (33, 'light_dies', dict(size=.1, note='eye highlights go out')), (36, 'inhale', dict(size=.5)), (38, 'scream', dict(size=1., note='voice, fire roar, everything; holds to f44')),
              (45, 'silence', dict(size=1., note='HARD CUT: nothing at all from here')), (46, 'heart', dict(size=.6)), (51, 'heart', dict(size=.5)), (68, 'heart', dict(size=.4, note='late, alone')), (75, 'bead_roll', dict(size=.12, dur=9, note='optional, barely there')), (84, 'bead_stop', dict(size=.1)), (87, 'black', dict(size=0))]

    def setup(self):
        self.fire = cached('d8_fire', fire_plate); self.env = env_map(); r = np.random.default_rng(21)
        self.paper = (np.ones((H, W, 3), f32) * rgb('#c9c1b6')) * (1 + .028 * fbm(H, W, (2.2, 5, 60), seed=31))[..., None]
        self.gst = [L._smudge(k) for k in range(5)]
        t = np.linspace(-.06, 1.06, NB); self.b0 = np.c_[t * W, 620 - 900 * (t - .5) ** 2 + 250 * (t - .5) * -1 + 8 * np.sin(t * 9)]
        a = r.uniform(0, TAU, NB); self.v = np.c_[(self.b0[:, 0] - 960) * r.uniform(.25, .6, NB) + r.normal(0, 130, NB), r.uniform(-560, -60, NB)]; self.vz = np.array([.10, .62, -.30, .34, -.12, 0, .74, -.34, .22, .80, -.10])[:NB]; self.w = r.uniform(-2.4, 2.4, NB)
        self.vz[HERO] = 0; self.v[HERO] = 0; self._f30 = None; self._M = None

    # ------------------------------------------------------------- beads
    def beads(self, f):
        push = 1 + .0016 * f; img = cam2d(self.fire, 960 + 60 - f * 2.5, 540 - f, 1.12, 0); img = img * .84; img = screen(img, radial(1330, 330, 700, 560, power=1.6)[..., None] * rgb('#ff8a3a') * .62); img = screen(img, radial(260, 760, 620, 420, power=1.6)[..., None] * rgb('#ff6a22') * .40)
        # the one who is falling: a soft shoulder goes down behind the beads
        c = Cel(); k = max(0, f - 7); dy = 40 + 14 * math.sin(min(f, 7) * .2) + 2.6 * k + .50 * k * k; dx = 18 * math.sin(f * .07); rot = .20 + .016 * k
        P = spl([(520, 1400), (560, 640), (690, 440), (900, 372), (1010, 320), (1030, 270), (962, 250), (1000, 196), (940, 130), (1010, 100), (1000, 20), (1080, 30), (1120, -50), (1180, 8), (1250, -40), (1270, 40), (1350, 60), (1320, 130), (1372, 190), (1310, 230), (1296, 300), (1350, 380), (1560, 470), (1660, 700), (1720, 1400)], True); co, si = math.cos(rot), math.sin(rot); P = (P - [1120, 600]) @ np.array([[co, si], [-si, co]]) + [1330 + dx, 590 + dy]
        c.fill(P, '#1a090e'); sh = rim(rim(c.layer(), -20, 14, '#ffb060', 1., 4.0), 16, 12, '#ff8030', .8, 4.0); sh = cv2.GaussianBlur(cv2.resize(sh, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4.8); img = over(img, cv2.resize(sh, (W, H), interpolation=cv2.INTER_LINEAR))
        img = FC.put(img, FF.embers(40 + f * .45, n=46, seed=5, wind=(40, -110), near=.22, size=1.3))
        tau = 0. if f < 8 else (2.6 * (1 - math.exp(-(f - 8) / 2.0)) + .42 * (f - 8)) / 9.
        jit = (lambda i: 0.) if f >= 8 else (lambda i: (.6 + f * .42) * math.sin(f * 2.9 + i * 1.3))
        pos = []
        for i in range(NB):
            x, y = self.b0[i]; z = 0.
            if f >= 8: x += self.v[i, 0] * tau; y += self.v[i, 1] * tau + 420 * tau * tau; z = self.vz[i] * tau * 1.0
            else: y += jit(i) + (-f * .9 if i in (4, 5) else 0)
            q = 1 / max(.12, 1 - z); pos.append([960 + (x - 960) * q * push, 540 + (y - 540) * q * push, 92 * q * push, z])
        hero = None
        if f >= 8:
            u = (f - 8) / 21.; hx, hy = self.b0[HERO]; e = u ** 2.2; hero = (hx + (960 - hx) * smooth(.15, 1, u) - 26 * math.sin(u * 3.1), hy + (540 - hy) * smooth(.1, 1, u) - 120 * math.sin(u * 3.1) * (1 - u), 92 * (1 - e) + 1330 * e, u)
        # string
        cel = Cel()
        if f < 8:
            S = spl([(p[0], p[1]) for p in pos], n=8); m = len(S); mid = (pos[4][0] + pos[5][0]) / 2
            for s0 in (S[S[:, 0] <= mid - 4], S[S[:, 0] >= mid + 4]) if f >= 5 else (S,): cel.line(s0, 7, '#1c0a0c', taper=(0, 0), press=.1); cel.line(s0 - [0, 1.5], 3, '#b98a52', taper=(0, 0), press=.2)
            if f >= 4:                                  # fibres let go one by one
                r = np.random.default_rng(f); y_ = np.interp(mid, S[:, 0], S[:, 1])
                for j in range(3 + 3 * (f - 4)):
                    sd = -1 if j % 2 else 1; a = r.uniform(-.9, .9); l = r.uniform(16, 46); p0 = np.array([mid + sd * r.uniform(4, 22), y_]); cel.line(spl([p0, p0 + [sd * -l * .4, a * 8], p0 + [sd * -l * .3, -l * .7 * (1 if a > 0 else -1)]], n=6), 2.0, '#e0b878', taper=(.05, .6))
                if f < 7: cel.line(np.array([(mid - 30, y_), (mid, y_ - 1), (mid + 30, y_)]), 3.2 - .8 * (f - 4), '#c89a60', taper=(0, 0))
        else:
            k = f - 8; w = math.exp(-k / 3.5)
            for sd, idx in ((-1, range(0, 5)), (1, range(5, NB))):
                pts = [self.b0[i] for i in idx]; pts = pts if sd < 0 else pts[::-1]; pts = np.array(pts, float); n = len(pts); u = np.linspace(0, 1, n); off = np.c_[-sd * -150 * (1 - w) * u ** 2, (-90 * np.sin(u * 5 + k * .5) * (1 - w) - 30 * k * .4 * u) * u]
                S = spl(pts * 1. + off * 1.1, n=8); al = max(0, 1 - k / 26.); cel.line(S, 6, '#1c0a0c', taper=(0, .1), press=.1, alpha=al); cel.line(S - [0, 1.4], 2.6, '#b98a52', taper=(0, .1), alpha=al)
                for j in range(4): p = S[-1]; cel.line(spl([S[-3], p, p + [sd * -10 - 6 * j, (j - 1.5) * 12]], n=5), 1.8, '#e0b878', taper=(.1, .7), alpha=al)
        sl = cel.layer(); img = over(img, cv2.GaussianBlur(sl, (0, 0), 1.2 + .25 * max(0, f - 8)) if f >= 8 else sl)
        for i in sorted(range(NB), key=lambda i: pos[i][3]):
            if i == HERO and hero is not None: continue
            if i == HERO: x, y, r_, z = pos[i]; img = lay(img, bead(x, y, r_, self.env, 1.2, None, face=self.f30()), 0); continue
            x, y, r_, z = pos[i]; an = self.w[i] * tau * 2.2 + i; hd = math.cos(an * 1.3 + i)
            hole = (math.sin(an) * .8, .12 * math.cos(an * 2), abs(math.cos(an))) if hd > 0 and f >= 8 else None
            B = bead(x, y, r_, self.env, an, hole); sig = abs(z - .05) * 22 * (1 if z > 0 else .8) + (5.5 if z > .35 else 0)
            if f in (8, 9) and B is not None: B = (B[0], B[1], smear(B[2], self.v[i, 0] * .05, self.v[i, 1] * .05, 9))
            img = lay(img, B, sig)
        if f == 8:
            c = Cel(); p = self.b0[4:6].mean(0); r = np.random.default_rng(2)
            for j in range(26): a = r.uniform(0, TAU); l = r.uniform(40, 210); d = np.array([math.cos(a), math.sin(a) * .7]); c.line(np.array([p + d * 14, p + d * l * .6, p + d * l]), r.uniform(1.5, 5), '#fff0d0', taper=(.1, .8))
            img = over(img, c.layer()); img = screen(img, radial(p[0], p[1], 240, power=2)[..., None] * rgb('#ffce8a') * .8)
        if hero is not None:
            u = hero[3]; fk = smooth(.62, 1., u); img = lay(img, bead(hero[0], hero[1], hero[2], self.env, 1.2 + u, None, clear=smooth(.62, 1.0, u) * .72, face=self.f30(), fk=fk), 0)
        return img

    def f30(self):
        if self._f30 is None: self._f30 = self.face(30)
        return self._f30

    # ------------------------------------------------------------- the face
    def pose(self, f):
        g = f if f >= 37 else f // 2 * 2
        pitch = float(lerp_keys(g, [(30, .02), (34, .00), (35, .0), (37, -.20), (38, .10), (39, .30), (41, .20), (44, .24)])); cy = float(lerp_keys(g, [(30, 500), (35, 502), (37, 545), (38, 420), (39, 382), (41, 410), (44, 404)]))
        st = float(lerp_keys(g, [(30, 0), (35, 0), (37, -.07), (38, .16), (39, .08), (41, -.01), (42, .0)])); s = float(lerp_keys(g, [(30, 432), (37, 446), (38, 380), (39, 352), (41, 360), (44, 358)])); roll = float(lerp_keys(g, [(30, .03), (37, .05), (39, -.03), (44, -.015)]))
        return g, pitch, cy, st, s, roll

    def rec(self, f, light='fire'):
        g, pitch, cy, st, s, roll = self.pose(f)
        if f < 38:
            e = (('worry', 'hollow', float(lerp_keys(g, [(30, .25), (32, .55), (34, 1.0)]))) if f < 36 else ('hollow', 'lip', .0)); kx = dict(hl=float(g < 33), gy=float(lerp_keys(g, [(30, -.035), (33, -.02), (34, 0)])), st=st, tear=0., sweat=0.)
            if f >= 36: kx.update(mw=.14, uy=-.625, ly=-.69, yc=-.655, fore=.5, fur=.4, bi=.40, bo=.25)
            if f == 37: kx.update(mw=.12, ly=-.75, jaw=.04, lu=1.18)
        else: e = 'scream'; kx = dict(st=st, wob=.004, tear=1., **(dict(mw=.36, jaw=.34) if f == 38 else dict(mw=.47, jaw=.31) if f == 39 else {}))
        jx, jy = (0, 0) if f < 39 else shake(f, 2.2, 7)
        return L.record(960 + jx, cy + jy, s, .07, pitch, roll, e, light, f, hat='back', shirt='vest', scar=False, kx=dict(L.ex(e), **kx) if False else kx, wind=.9 if f < 38 else 1.8, drift=(-.05 + .03 * math.sin(f * .5), 0.) if f < 38 else (-.10 + .05 * math.sin(f * 1.1), .05), blow=float(lerp_keys(f, [(37, 0), (38, .35), (39, .70), (41, .40), (44, .46)])), body_dy=float(lerp_keys(f, [(30, 0), (34, 0), (37, -.13), (38, -.04), (39, .05), (41, 0)])))

    def face(self, f):
        z = 1 + .0035 * (f - 30) if f < 38 else 1.04; fl = 1 + .07 * math.sin(f * 1.9) + .05 * math.sin(f * 3.1)
        img = cam2d(self.fire, 960 - 20 + (f - 30) * 2, 540, 1.10 + .004 * (f - 30), 0) * fl
        c = Cel(); t = f / 24.; n = 1.0 if f < 38 else 1.5
        FF.flames(c, spl([(-60, 1090), (300, 1040), (620, 1075)], n=8), 360 * n, t=t, seed=3, ink=False); FF.flames(c, spl([(1260, 1075), (1600, 1010), (1980, 1060)], n=8), 560 * n, t=t + .3, seed=8, ink=False)
        fl_ = c.layer(); q = cv2.GaussianBlur(cv2.resize(fl_, (W // 2, H // 2), interpolation=cv2.INTER_AREA), (0, 0), 5); img = over(img, cv2.resize(q, (W, H)) * .95); img = screen(img, cv2.resize(cv2.GaussianBlur(q[..., :3], (0, 0), 34), (W, H)) * .55)
        img = FC.put(img, FF.embers(f, n=60, seed=9, wind=(-60, -190) if f < 38 else (-120, -420), near=.12))
        R, Hd, E, Lt = self.rec(f); g, pitch, cy, st, s, roll = self.pose(f)
        h = Hd.P3(np.array([(0, -1.05, -.9)]))[0]; hb = HT.hat(h[0] + s * .42, h[1] - s * .02, s * 3.25, .38, 1.12, roll - .22, light='fire', wear=.7, f=f, wind=.8)
        lay_ = R.cel(Lt, s, hb.copy())
        if f >= 39:                                    # flung tears
            c = Cel(); r = np.random.default_rng(5); k = f - 38
            for j in range(26):
                sd = -1 if j % 2 else 1; e0 = Hd.P(np.array([(sd * .40, -.20)]))[0]; a = (-.35 + 1.0 * r.random()); sp = r.uniform(30, 82); d = np.array([sd * math.cos(a), -math.sin(a) * .9 + .2]); kk = k + r.uniform(-3, 2)
                if kk <= 0: continue
                p = e0 + d * sp * kk ** .86 + [0, 2.0 * kk * kk]; R_ = r.uniform(5, 15) * (1 + .05 * kk); v = d * sp * .86 + [0, 4 * kk]; an = math.atan2(v[1], v[0]); D = np.array([(1.7, 0), (.5, .62), (-.6, .66), (-1.1, 0), (-.6, -.66), (.5, -.62)]) * R_; co, si = math.cos(an), math.sin(an); D = spl(D, True, n=5) @ np.array([[co, si], [-si, co]]) + p
                c.fill(D, '#f6efe6', .85); c.line(D, 2.0, '#4a1c1c', closed=True); c.fill(p + (D - p) * .38 + [-R_ * .2, -R_ * .3], '#ffffff')
            lay_ = over(np.zeros((H, W, 3), f32), lay_) * 0 + 0 if False else L.over4(lay_, c.layer())
        img = over(img, lay_)
        e_ = FF.embers(f + 50, n=16, seed=19, wind=(-90, -230), near=.8, size=1.5); gx_, gy_ = grid(); mk = smooth(.75, 1.25, np.sqrt(((gx_ - 960) / 420) ** 2 + ((gy_ - 480) / 460) ** 2))[..., None]; img = FC.put(img, FC.FX(None if e_.rgba is None else e_.rgba * mk, e_.light * mk), .85)
        if f >= 38:
            c = Cel(); focus_lines(c, 960, 470, 660, 1500, 130, 9, '#16050a', seed=f, ry=.70); img = over(img, c.layer() * .88); c = Cel(); focus_lines(c, 960, 470, 820, 1500, 40, 5, '#ff9a4a', seed=f + 99, ry=.70); img = screen(img, c.layer()[..., :3] * .7)
            a = 22 * math.exp(-(f - 38) / 3.2) + 3; sx, sy = shake(f, a, 3); img = cam2d(img, 960 + sx, 540 + sy, 1.035 + (.05 if f == 38 else .02 if f == 39 else 0), 0)
            if f == 38: img = zoomblur(img, 960, 470, .07, 9); img = impact(img, 'red', lo=.20, hi=.50)
        else: img = cam2d(img, 960, 540, z, 0)
        return img

    # ------------------------------------------------------------- silence
    def construct(self, Hd, s, seed, amt, np_):
        """what is under every drawing: skull ball, jaw, axes, a shoulder line"""
        r = np.random.default_rng(seed); c = Cel(W, H, ss=1); col = '#2e2724'
        def ln(P, w=2.0, n=2, col=col, j=4., ext=0.):
            for _ in range(n):
                Q = resample(np.asarray(P, float), 10); m = len(Q)
                if m < 3: continue
                Q = Q + np.c_[noise1(m, 20, r), noise1(m, 20, r)] * j + r.normal(0, j * .6, 2); i = int(r.integers(0, max(1, m // 5)))
                c.line(Q[i:m - int(r.integers(0, max(1, m // 5)))], w * r.uniform(.6, 1.3), col, taper=(.3, .3), press=.6, seed=int(r.integers(1 << 20)), alpha=.88, tremor=1.)
        t = np.linspace(0, TAU, 90); ph = r.uniform(0, 6)
        ln(Hd.P3(np.c_[.80 * np.cos(t + ph), .28 + .80 * np.sin(t + ph), 0 * t]), 2.2, 3, j=6); ln(Hd.P3(np.c_[0 * t, .28 + .80 * np.sin(t), .78 * np.cos(t)]), 1.6, 1)
        y = np.linspace(.98, -1, 40); ln(Hd.P(np.c_[0 * y, y], .02), 2.4, 2, '#a5212b', j=3)
        x = np.linspace(-.72, .72, 40)
        for yy, w in ((-.03, 2.2), (.30, 1.4), (-.40, 1.4), (-.66, 1.6)): ln(Hd.P(np.c_[x, 0 * x + yy]), w, 2, j=3.5)
        for sd in (-1, 1): ln(Hd.jaw(sd), 2.0, 2, j=4); ln(Hd.P(L._circ(sd * .335, -.04, .185, 40)), 1.8, 2, j=3)
        ln(Hd.P(np.c_[.33 * np.cos(t), -.70 + .20 * np.sin(t)], .01), 2.6, 3, j=5)
        for sd in (-1, 1): ln(np.array([np_ + [sd * s * .33, -s * .55], np_ + [sd * s * .36, s * .02], np_ + [sd * s * .75, s * .16], np_ + [sd * s * 1.35, s * .5], np_ + [sd * s * 1.45, s * 1.4]]), 2.0, 2, j=6)
        ln(np_ + np.c_[s * 1.3 * np.linspace(-1, 1, 30), s * (.28 + .10 * np.linspace(-1, 1, 30) ** 2)], 1.6, 2, j=5); ln(np.array([np_ + [0, s * .1], np_ + [6, s * .7], np_ + [0, s * 1.5]]), 2.0, 2, '#a5212b', j=4)
        # the animator's timing ticks, half rubbed out
        ln(np.array([(1652, 250), (1650, 420), (1653, 610)]), 2.0, 1, j=1.)
        for k, yy in enumerate((250, 340, 385, 410, 520, 610)): ln(np.array([(1636, yy), (1652, yy), (1670 + 10 * (k in (0, 5)), yy + 1)]), 2.0, 1, j=.8)
        a = c.rgba() * np.clip(L._grain(0) * 1.3, 0, 1)[..., None]; return a * amt

    def silent(self, f):
        k = min(9, (f - 45) // 3); seed = 300 + k * 17; s = SS; r = np.random.default_rng(seed)
        e = ('scream', 'hollow', EMP[k]); kx = dict(tear=1. - .6 * EMP[k], mw=.43 * (1 - EMP[k]) + .17 * EMP[k], jaw=.27 * (1 - .7 * EMP[k]))
        j = r.normal(0, 2.0 + 3 * LOO[k], 2) * (k < 5); pk = PIT[k]
        R, Hd, E, Lt = L.record(960 + j[0], CY[k] + j[1], s, .07, pk, -.015 + .03 * EMP[k], e, 'fire', 45 + k * 3, hat='back', shirt='vest', scar=False, kx=kx, wind=1.2 * (1 - EMP[k]), blow=.46 * (1 - EMP[k]), body_dy=.06 * EMP[k])
        np_ = Hd.P3(np.array([(0, -1.20 + E['jaw'] * .8, -.16)]))[0]
        # paper: charcoal dust rubbed across it, thinning as the tones are lost
        dust = np.clip(.42 + .20 * self.gst[k % 5] + .10 * self.gst[(k + 2) % 5], 0, 1) * (.95 * FIL[k] + .10); x, y = grid(); vg = np.clip(((x - 960) / 1100) ** 2 + ((y - 520) / 700) ** 2, 0, 1.4)
        img = self.paper * (1 - (dust * (.34 + .42 * vg))[..., None] * np.array([1, 1.01, 1.02], f32))
        # broad strokes of the side of the stick, radiating from the mouth
        c = Cel(W, H, ss=1)
        for q in range(34):
            a = r.uniform(0, TAU); d0 = r.uniform(560, 760); d = np.array([math.cos(a), math.sin(a) * .72]); p0 = np.array([960, 480]) + d * d0; l = r.uniform(260, 620); b = a + r.normal(0, .12); c.line(np.array([p0, p0 + d * l * .5 + r.normal(0, 10, 2), p0 + np.array([math.cos(b), math.sin(b) * .72]) * l]), r.uniform(5, 34), '#1d1917', taper=(.1, .5), press=.7, alpha=r.uniform(.25, .7), tremor=2.)
        bg = c.rgba() * np.clip(L._grain(0) * 1.5 - .35, 0, 1)[..., None] * (.95 * max(0, 1 - LOO[k] * 1.25)); img = over(img, bg)
        h = Hd.P3(np.array([(0, -1.05, -.9)]))[0]; hl = HT.hat(h[0] + s * .42, h[1] - s * .02, s * 3.25, .38, 1.12, -.24, light='charcoal', wear=.7, f=0, wind=0, rim=.25); m = np.clip((FIL[k] * 1.9 - .72 + .30 * self.gst[1]) * 5, 0, 1) * (.7 + .3 * L._grain(0)); img = over(img, hl * m[..., None])
        if CON[k] > 0: img = over(img, self.construct(Hd, s, seed + 3, CON[k], np_))
        img = over(img, R.charcoal(seed, FIL[k], LOO[k], 1., DET[k], redfill=RED[k], line=LIN[k], shade_hatch=float(np.interp(k, [2, 4, 7, 9], [0, 1, .8, 0])), w=1.15))
        if k >= 8: img = img * .62 + self.paper * .38 if k == 9 and f >= 74 else img
        pulse = sum(a * math.exp(-(f - t) / 2.2) for t, a in ((46, .016), (51, .013), (68, .011)) if f >= t); img = cam2d(img, 960, 540, 1.0 + pulse, 0)
        if pulse > .001: g = img.mean(-1, keepdims=True); rd = np.clip(img[..., :1] - g, 0, 1) * 2; img = img + rd * rgb('#ff2030') * pulse * 14 * .12
        return img

    def last(self, f):
        """an empty sheet, the ghost of the drawing, one bead"""
        R0 = getattr(self, '_ghost', None)
        if R0 is None:
            Hd = L.LHead(**L.GROWN).pose(960, 574, SS, .07, -.36, .01, jaw=.1); q = self.construct(Hd, SS, 461, 1., Hd.P3(np.array([(0, -1.2, -.16)]))[0]); R0 = self._ghost = cv2.GaussianBlur(smear(q, 14, 4, 5), (0, 0), 1.4)
        x, y = grid(); img = self.paper * (1 - (.10 * np.clip(.5 + .4 * self.gst[2], 0, 1))[..., None]); img = over(img, R0 * .20)
        t = f - 75; gx = 1010.; u = min(1., t / 9.); px = -90 + (gx + 100) * (1 - (1 - u) ** 2.1)
        if t > 9: px = gx + 10 - 13 * math.sin((t - 9) * .62) * math.exp(-(t - 9) / 2.4) - 0
        py = 742.; rb = 62; c = Cel(W, H, ss=1); r = np.random.default_rng(8)
        for q in range(3): X = np.linspace(-40 + 60 * q, 1960 - 90 * q, 60); c.line(np.c_[X, 742 + rb + 2 + 3 * noise1(60, 12, r) + q * 1.5], 2.2, '#4b443f', taper=(.2, .4), press=.7, seed=q, alpha=.55, tremor=1.2)
        X = np.linspace(-60, px - rb * .2, 40)
        if px > 0: c.line(np.c_[X, py + rb - 2 + 2 * noise1(40, 9, r)], 4, '#8f1a22', taper=(.5, .2), press=.8, seed=4, alpha=.40)
        sh = L._circ(px + 10, py + rb + 3, 1, 30) ; sh = (sh - [px + 10, py + rb + 3]) * [rb * 1.05, 8] + [px + 10, py + rb + 3]; c.fill(sh, '#1b1614', .55); img = over(img, c.rgba() * np.clip(L._grain(0) * 1.3, 0, 1)[..., None])
        an = px / rb; c = Cel(W, H, ss=2); C0 = L._circ(px, py, rb, 60); c.fill(C0, '#b0141e'); c.clip(C0); c.fill(L._circ(px + rb * .34, py + rb * .30, rb * 1.02, 50)[::1], '#d02028', .0); D = L._circ(px - rb * .30, py - rb * .28, rb * 1.02, 50); c.fill([L._circ(px, py, rb * 1.2, 50), D], '#5c0a12', evenodd=True)
        hx, hy = math.cos(an), math.sin(an)
        for sg in (1, -1): c.fill(L._circ(px + sg * hx * rb * .78, py + sg * hy * rb * .78, rb * .17, 20), '#2c0508')
        c.unclip(); c.fill(L._circ(px - rb * .36, py - rb * .42, rb * .17, 20, .7), '#e9dfd2')
        for q in range(3): c.line(C0 + r.normal(0, 1.2, 2), 2.6, '#1b1210', closed=True, press=.5, seed=q, alpha=.8)
        bl = c.rgba() * (np.clip(L._grain(0) * .5 + .62, 0, 1))[..., None]
        if 0 < u < .75: v = 34 * (1 - u) ** 1.1; bl = smear(bl, v, 0, 7)
        img = over(img, bl)
        dk = float(lerp_keys(f, [(75, 0), (80, .06), (84, .34), (87, .86), (88, 1.), (89, 1.)])); d = np.sqrt(((x - px) / 900) ** 2 + ((y - py) / 560) ** 2); m = np.clip(1 - dk * 1.0 - smooth(0, 1.2, d + dk * .9 - .35) * min(1, dk * 2.2), 0, 1)[..., None]
        out = img * m
        if f >= 88: out = out * 0
        return out

    def frame(self, f):
        if f < 30: return self.beads(f).astype(f32)
        if f < 45: return self.face(f).astype(f32)
        if f < 75: return self.silent(f).astype(f32)
        return self.last(f).astype(f32)
