"""d2 the hat: dust, yellow light.  A hand sets the hat on Nami's head on 12; she looks up under the brim, the face breaks; rack focus on 34 to four backs walking, a step on the beat; pinwheel."""
from chars.lkit import *
from chars import nami as N
from op import hat as HT
from op.comp import over, radial, blur, cam2d, bokeh, disc_kernel
from op.paint import Canvas, arcs, repaint, ramp, fbm
from op.fx_air import motes
from op.kit import flare, smear
import os

BX, BY, S = 690, 606, 240
HAT = dict(w=3.0, tilt=-.02)


def defocus(L, r):
    if r < .6: return L
    k = disc_kernel(r); h, w = L.shape[:2]
    if r > 6: q = cv2.resize(L, (w // 2, h // 2), interpolation=cv2.INTER_AREA); q = cv2.filter2D(q, -1, disc_kernel(r / 2), borderType=cv2.BORDER_REPLICATE); return cv2.resize(q, (w, h), interpolation=cv2.INTER_LINEAR)
    return cv2.filter2D(L, -1, k, borderType=cv2.BORDER_REPLICATE)


# ------------------------------------------------------------------ the four backs
def limb(cel, P, w, col, shade, lw, sg=1, seed=0, cap_=True):
    P = resample(spl(np.asarray(P, float), n=8), 3.); n = len(P); T = np.gradient(P, axis=0); T /= np.hypot(*T.T)[:, None] + 1e-9; N_ = np.c_[-T[:, 1], T[:, 0]]; ww = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(w)), w)[:, None] * .5; A, B = P + N_ * ww, P - N_ * ww; q = np.vstack([A, B[::-1]]); cel.fill(q, col); L = A if A[:, 0].mean() < B[:, 0].mean() else B; R = B if L is A else A; cel.fill(np.vstack([L, (L + (R - L) * .52)[::-1]]), shade); cel.line(A, lw, INK, taper=(.03, .03), seed=seed); cel.line(B, lw, INK, taper=(.03, .03), seed=seed + 1); return P, A, B


def leg(ph):
    """walk away seen from behind: ph 0 = heel contact in front.  -> (depth, lift, kneelift)"""
    ph = ph % 1.
    if ph < .6: u = ph / .6; return lerp(.5, -.5, u), .0 + .06 * max(0, (u - .8) / .2) ** 1.5, 0.
    u = (ph - .6) / .4; return lerp(-.5, .5, sm(u)), .14 * math.sin(math.pi * u) ** .9, .5 * math.sin(math.pi * min(1, u * 1.15))


# the line stands till 30, then eases into one step per 15 f (contacts 45, 60) and only ever gets smaller
def stride(f):
    if f <= 30: return -.5
    u = (f - 30) / 15.; return -.5 + .5 * (2 * u * u - u ** 3) if u < 1 else (f - 45) / 30.


def scale(f): return 1.2 * math.exp(-.11 * (stride(f) + .5))


# Luffy: his own quicker steps (contacts 25, 35, 45) till he falls in with the line's beat
LF, LP = (5., 25., 35., 45., 60., 75., 90.), (-2., -1., -.5, 0., .5, 1., 1.5)


def trail(f):
    """his path on the road: (sideways px at scale 1, depth = 1/scale).  Off frame right and near at 25, on his mark at 54"""
    u = min((f - 25) / 29., 1.); return lerp(288., -100., 1 - (1 - u) ** 1.9), lerp(1 / 3., 1 / scale(f), 1 - (1 - u) ** 1.2)


def foot(ph, sd, w=None):
    """a foot on the road: the print stays where it landed, then swings to the next one"""
    rel = (ph + (.5 if sd > 0 else 0.)) % 1.; pc = ph - rel; P = lambda q: np.array(trail(float(np.interp(q + .28, LP, LF)))); q = P(pc) if rel < .6 else lerp(P(pc), P(pc + 1), sm((rel - .6) / .4))
    if w is not None: w = np.array(w); m = np.array([45., .075 * w[1]]); q = w + m * np.tanh((q - w) / m)      # a stride no longer than a leg allows
    return q + [sd * 25, 0]


class Man:
    def __init__(self, who): self.who = who

    def draw(self, x, y, h, ph, f, lw=3., rimc='#fff0b4', amp=1., lean=0., feet=None):
        w_ = self.who; c = Cel(); u = h; bob = .010 * amp * math.cos(TAU * ph * 2) * u; sway = -.012 * amp * math.sin(TAU * (ph + .05)) * u; t = f / 24.
        def Q(P): P = np.asarray(P, float); return P * [u, -u] + [x + sway, y + bob] + P[..., 1:2] * [lean * u, 0.]
        sk, sks = ('#e3a06c', '#9c5c38') if w_ != 'zoro' else ('#d1905c', '#8d5230')
        tr = dict(luffy=('#3a63b0', '#243c78'), zoro=('#2c3d36', '#17231f'), sanji=('#26242e', '#12111a'), usopp=('#8a6434', '#573a1c'))[w_]
        for sd in (-1, 1):                                             # legs (near = trailing first)
            d, lift, kl = leg(ph + (.5 if sd > 0 else 0.)); hip = np.array([sd * .055, .50]); ank = np.array([sd * .060 - .012 * sd * min(1., lift / .05), .045 + lift + d * .040]) if feet is None else np.array([(feet[sd][0] - x) / u, (y - feet[sd][1]) / u + .045 + lift]); ank = ank + [-sway / u - lean * ank[1], bob / u]; kn = (hip + ank) / 2 + [sd * .014, .012 + .03 * kl]
            if w_ == 'luffy':
                limb(c, Q([kn + [0, .03], (kn + ank) / 2, ank]), [.072 * u, .074 * u, .046 * u], sk, sks, lw); limb(c, Q([hip, (hip + kn) / 2, kn - [0, .02]]), [.125 * u, .118 * u, .110 * u], *tr, lw, seed=sd + 3); e = Q([kn + [-.056, -.030], kn + [.056, -.030], kn + [.056, .004], kn + [-.056, .004]]); c.fill(e, '#e9e2d2'); c.fill(e[[0, 3]].mean(0) + (e - e[[0, 3]].mean(0)) * [.5, 1], '#a59a92'); stroke(c, e, lw * .8)
            else: limb(c, Q([hip, (hip + kn) / 2, kn, (kn + ank) / 2, ank + [0, -.012]]), [.128 * u, .118 * u, .098 * u, .094 * u, .086 * u if w_ != 'zoro' else .080 * u], *tr, lw, seed=sd + 3); fold(c, Q([kn + [-.035, .006], kn + [0, -.010], kn + [.03, .004]]), lw * .5); fold(c, Q([ank + [-.03, .06], ank + [.005, .045], ank + [.03, .055]]), lw * .5)
            s_ = .5 * kl; ft = Q(spl([ank + [-.036, .03], ank + [.036, .03], ank + [.040, -.012 - .02 * s_], ank + [.030, -.045 - .03 * s_], ank + [-.030, -.045 - .03 * s_], ank + [-.040, -.012 - .02 * s_]], True, n=5)); fc = dict(luffy=('#c99a5c', '#7a542c'), zoro=('#1f1a1a', '#0e0b0c'), sanji=('#16141c', '#0a090e'), usopp=('#5a3a26', '#33200f'))[w_]; c.fill(ft, fc[0]); c.fill(ft[len(ft) // 2:] , fc[1]); stroke(c, ft, lw * .9)
            if w_ == 'luffy': c.line(Q([ank + [-.03, .02], ank + [.03, .02]]), lw * 1.2, '#4a3018', taper=(0, 0))
        tp = dict(luffy=('#c22b30', '#7c161f'), zoro=('#ece6d6', '#a29a94'), sanji=('#2a2833', '#14121c'), usopp=('#8a6434', '#573a1c'))[w_]; sh_ = .012 * amp * math.sin(TAU * ph); T3 = [(-.085, .47), (-.092, .56), (-.100, .68), (-.150, .785 + sh_), (-.125, .822 + sh_), (-.045, .842), (.045, .842), (.125, .822 - sh_), (.150, .785 - sh_), (.100, .68), (.092, .56), (.085, .47)]
        # arms behind / beside
        for sd in (-1, 1):
            a = -.30 * math.sin(TAU * (ph + (.5 if sd > 0 else 0))); shd = np.array([sd * .140, .800 - sh_ * sd]); el = shd + [sd * .045, -.145 + .03 * (amp - 1) * a]; wr = el + [sd * .005, -.130 + .04 * amp * a]; ac = (sk, sks) if w_ in ('luffy', 'zoro', 'usopp') else tp
            if w_ == 'sanji': el = shd + [sd * .062, -.150]; wr = np.array([sd * .100, .515])
            if w_ == 'zoro' and sd < 0: el = shd + [-.055, -.140]; wr = np.array([-.105, .55])
            limb(c, Q([shd, el, wr]), [.074 * u, .062 * u, .048 * u], *ac, lw, seed=sd + 8)
            if w_ != 'sanji': hd_ = Q(spl([wr + [-.027, .012], wr + [.027, .012], wr + [.034, -.03], wr + [.018, -.058], wr + [-.018, -.058], wr + [-.034, -.03]], True, n=5)); c.fill(hd_, sk); c.fill(hd_ * [.5, 1] + [hd_[:, 0].min() * .5, 0], sks); stroke(c, hd_, lw * .85); [c.line(Q([wr + [k * .012, -.030], wr + [k * .011, -.055]]), lw * .5, INK) for k in (-1, 0, 1)]
            if w_ == 'zoro': limb(c, Q([shd, lerp(shd, el, .55)]), [.092 * u, .088 * u], *tp, lw, seed=sd + 5); 
            if w_ == 'zoro' and sd < 0: b = lerp(shd, el, .72); limb(c, Q([b + [-.004, .012], b + [.004, -.012]]), [.074 * u, .074 * u], '#15141a', '#08080c', lw)
        B = Q(spl(T3, True, n=6)); c.fill(B, tp[0]); c.clip(B); c.fill(Q(spl([(-.2, .9), (-.02, .86), (-.035, .74), (-.015, .62), (-.03, .4), (-.2, .4)], True, n=5)), tp[1])
        if w_ == 'luffy': c.fill(Q([(-.2, .515), (.2, .505), (.2, .4), (-.2, .4)]), '#3a63b0'); c.fill(Q([(-.2, .515), (0, .51), (0, .4), (-.2, .4)]), '#243c78')
        if w_ == 'zoro': c.fill(Q([(-.2, .625), (.2, .620), (.2, .515), (-.2, .520)]), '#4d8a44'); c.fill(Q([(-.2, .625), (-.02, .622), (-.03, .517), (-.2, .520)]), '#2c5430'); [c.line(Q([(-.2, .60 - k * .026), (.2, .596 - k * .026)]), lw * .6, '#1c3a20', taper=(0, 0)) for k in range(4)]
        if w_ == 'usopp': c.fill(Q([(-.2, .56), (.2, .555), (.2, .505), (-.2, .51)]), '#e9e0c8'); [c.line(Q([(sd * .03, .60), (sd * .085, .85)]), lw * 2.6, '#5c3f1e', taper=(0, 0)) for sd in (-1, 1)]; c.fill(Q([(-.2, .9), (.2, .9), (.2, .70), (-.2, .70)]), sk, .0)
        for q in ([(-.05, .80), (-.03, .70), (-.04, .60)], [(.06, .79), (.04, .72), (.05, .64)], [(-.08, .60), (-.02, .57), (.05, .585)]): fold(c, Q(q), lw * .6)
        c.unclip(); outline(c, B, lw * 1.05, INK, 3)
        if w_ == 'sanji': c.line(Q([(0, .80), (.004, .62), (0, .46)]), lw * .6, '#07060a'); tl = .012 * math.sin(t * 4); [outline(c, Q(spl(P, True, n=5)), lw * .8, INK, 5) for P in ([(-.10, .52), (-.015, .515), (-.02 + tl, .40), (-.11 + tl, .405)],)]
        if w_ == 'zoro':
            for k in range(3): a_ = .035 * math.sin(TAU * (ph + .1 * k)); p0, p1 = np.array([.16 + .02 * k, .60 - .025 * k]), np.array([-.10 - .05 * k, .36 - .05 * k + a_]); limb(c, Q([p0 + [.10, .095], p0]), [.026 * u, .026 * u], ('#f2f0ea', '#4a2320', '#2a1a3a')[k], ('#b5b0b4', '#2a1210', '#150c22')[k], lw * .8); limb(c, Q([p0, lerp(p0, p1, .5) + [0, -.008], p1]), [.034 * u, .034 * u, .026 * u], ('#f2f0ea', '#5c2a24', '#1b1628')[k], ('#a9a4a8', '#34150f', '#0c0a14')[k], lw * .8); g = Q([p0]); cel_ = np.linspace(0, TAU, 18); c.fill(np.c_[g[0, 0] + np.cos(cel_) * .030 * u, g[0, 1] + np.sin(cel_) * .022 * u], '#d8a93c'); stroke(c, np.c_[g[0, 0] + np.cos(cel_) * .030 * u, g[0, 1] + np.sin(cel_) * .022 * u], lw * .7)
        nk = Q([(-.030, .83), (.030, .83), (.028, .875), (-.028, .875)]); c.fill(nk, sks); stroke(c, nk, lw * .7)
        # heads from behind
        hx, hy = .004 * math.sin(TAU * ph), .922 + .004 * math.cos(TAU * ph * 2)
        for sd in (-1, 1): e = Q(spl([(hx + sd * .060, hy + .005), (hx + sd * .080, hy + .014), (hx + sd * .078, hy - .018), (hx + sd * .058, hy - .022)], True, n=5)); c.fill(e, sk if sd > 0 else sks); stroke(c, e, lw * .7)
        r = rs(len(w_), 4); wind = .010 * math.sin(t * 5.)
        hc = dict(luffy=('#1d1a22', '#0c0a10'), zoro=('#6fb45a', '#3c7a3c'), sanji=('#f0cf6a', '#b4863a'), usopp=('#1c1714', '#0a0807'))[w_]
        if w_ == 'usopp': P = spl([(hx + .105 * math.cos(a) * (1 + .10 * math.sin(a * 9)), hy + .012 + .098 * math.sin(a) * (1 + .10 * math.sin(a * 9))) for a in np.linspace(0, TAU, 60)[:-1]], True, n=2)
        else:
            K = dict(luffy=15, zoro=19, sanji=9)[w_]; P = []
            for k in range(K):
                a = math.pi * (1.12 - 1.24 * 0) ; th = -math.pi * .10 + math.pi * 1.2 * k / (K - 1.); R0 = .078; R1 = dict(luffy=.104, zoro=.098, sanji=.084)[w_] * r.uniform(.94, 1.10); P += [(hx + R0 * math.cos(th), hy + .004 + R0 * math.sin(th)), (hx + R1 * math.cos(th + .06) + wind, hy + .004 + R1 * math.sin(th + .06))]
            for k in range(7 if w_ != 'sanji' else 5): q = lerp(-.060, .060, k / (6. if w_ != 'sanji' else 4.)); P += [(hx + q - .008, hy - .040), (hx + q + wind * .5, hy - dict(luffy=.085, zoro=.066, sanji=.080)[w_] * r.uniform(.9, 1.1))]
            P = np.array(P); n1 = K * 2; P = np.vstack([P[:n1], P[n1:]])
        HP = Q(P); c.fill(HP, hc[0]); c.clip(HP); c.fill(Q(spl([(hx - .15, hy + .12), (hx - .01, hy + .10), (hx + .012, hy + .03), (hx - .004, hy - .04), (hx + .0, hy - .12), (hx - .15, hy - .12)], True, n=5)), hc[1]); [c.line(Q(spl([(hx + .006 * k, hy + .075), (hx + .022 * k, hy + .02), (hx + .020 * k, hy - .055)], n=5)), lw * .55, INK if w_ != 'luffy' else '#4a4658', taper=(.3, .5)) for k in (-2, -1, 1, 2)]; c.unclip(); outline(c, HP, lw, INK, 9)
        if w_ == 'usopp': b = Q(spl([(hx - .086, hy + .030), (hx, hy + .050), (hx + .086, hy + .030), (hx + .084, hy + .058), (hx, hy + .080), (hx - .084, hy + .058)], True, n=5)); c.fill(b, '#8a8a4a'); c.clip(b); [c.line(Q([(hx - .1 + k * .03, hy + .02), (hx - .085 + k * .03, hy + .09)]), lw, '#4f5a2c', taper=(0, 0)) for k in range(7)]; c.unclip(); stroke(c, b, lw * .8); bg = Q(spl([(.06, .60), (.16, .585), (.165, .48), (.07, .47)], True, n=5)); c.fill(bg, '#c9a15c'); c.shade(bg - [.03 * u, 0], '#7f5c2c', bg, .0); outline(c, bg, lw, INK); c.line(Q([(-.11, .81), (.0, .70), (.11, .60)]), lw * 2.4, '#4a3014', taper=(0, 0))
        L = c.layer(); L = para(L, (x, y - h), (x, y), [(0, '#fbeccc'), (.55, '#dcb384'), (1, '#946440')]); L = lit(L, (-.75, .66), rimc, lw * .5, h * .0125, 1., .5); L = lit(L, (.95, .3), rimc, lw * .5, h * .006, .7, .3)
        if w_ == 'sanji':
            c = Cel(); p0 = Q([(hx + .05, hy + .0)])[0]
            for k in range(2): P = np.array([p0 + [-v * 90 - 24 * math.sin(v * 6 + t * 4 + k) * v, -v * 120 - 10 * math.sin(v * 9 + t * 3)] for v in np.linspace(0, 1, 18)]); c.line(P, 9, '#fff3cf', taper=(.1, .5), alpha=.40)
            L = lay(blur(c.layer(), 2.), L)
        return L


class Shot:
    id = 'd2'; n = 60; finish = dict(bloom=.42, vig=.16)
    EVENTS = [(0, 'sob', dict(size=.4, note='Nami, held back, f0-10')), (3, 'hitch', dict(size=.3)), (9, 'cloth', dict(size=.3, note='an arm comes down')), (12, 'hat', dict(size=.8, note='straw hat set on her head, soft thump of straw')), (16, 'pat', dict(size=.3, note='hand presses once more')), (18, 'cloth', dict(size=.2, note='hand lifts away')), (25, 'step', dict(size=.3, note='Luffy turns and goes')), (35, 'step', dict(size=.35, note='Luffy, second step, catching the line up')), (28, 'breath', dict(size=.5, note='Nami: shaky breath in')), (31, 'sob', dict(size=.9, note='the face breaks')),
              (34, 'rack', dict(note='focus pulls')), (36, 'gust', dict(size=.4, note='wind, pinwheel whirrs up')), (45, 'step', dict(size=1.0, note='four men, one footfall, on the beat')), (47, 'jingle', dict(size=.2, note='three swords')), (59, 'step', dict(size=.8, note='next footfall lands on the cut'))]

    def setup(self):
        p = 'cache/d2_bg.npy'
        if os.path.exists(p): self.bg = np.array(np.load(p, mmap_mode='r'), f32); self.mid = np.array(np.load('cache/d2_mid.npy', mmap_mode='r'), f32)
        else: self.bg, self.mid = self.paint(); np.save(p, self.bg.astype(np.float16)); np.save('cache/d2_mid.npy', self.mid.astype(np.float16))
        pl = dict(skin='#eeb27c', shade='#a8683f'); sty = dict(nails='short'); self.hs = []; self.arm_ = None
        for po in (dict(HD.lerp_pose('rest', 'splay', .45), thumb=(66, 34, 22, 8, 22)), dict(HD.blend('rest', 'open', .35), thumb=(58, 26, 24, 6, 6)), HD.blend('rest', 'relaxed', .5)): a, _ = hand_sprite(po, 'L', HD.euler(14, 34, 128), 116, HD.MALE, pl, light=(.6, .6, .5), w=4.6, res=440, rimc='#fff0b4', rimd=(-.82, .57), rw=8, style=sty); self.hs.append(a); self.hg = _
        R_ = max(self.hg.wrist[0], key=lambda q: abs(q['y'])); PR = np.asarray(R_['P'], float)[:, :2]; S2 = a.shape[0] / 2.; dv = PR.mean(0) - S2; self.adir = dv / np.hypot(*dv); self.alen = float(np.hypot(*dv)); nn = np.array([-self.adir[1], self.adir[0]]); self.awid = (float((PR - S2).dot(nn).min()), float((PR - S2).dot(nn).max()))
        st = N.style('dusk'); pn = dict(skin=st['skin'], shade=st['shade']); self.own = [hand_sprite(po, 'R', HD.euler(16, 10, -62), 98, HD.FEMALE, pn, light=(.6, .6, .5), w=4.2, res=400, rimc='#fff0b4', rimd=(-.82, .57), rw=7, style=sty)[0] for po in (HD.POSES['hook'], HD.blend('hook', 'rest', .55))]
        self.men = {k: Man(k) for k in ('luffy', 'zoro', 'sanji', 'usopp')}

    def paint(self):
        h, w = 1240, 2240; y = np.linspace(0, 1, h)[:, None].repeat(w, 1); gx, gy = grid(h, w); hz = 600.; vx = 1560.
        im = ramp(y, [(0, '#e6c983'), (.28, '#f3e0a2'), (.49, '#fff3c8'), (.50, '#f2d894')]); im = im + np.exp(-(((gx - 1640) / 560) ** 2 + ((gy - 470) / 300) ** 2))[..., None] * rgb('#fff0c0') * .28
        c = Cel(w, h); r = rs(3)
        for k, (yy, col, amp) in enumerate(((600, '#e8cd8c', 70), (606, '#dcb878', 46), (612, '#cfa566', 30))): P = np.c_[np.linspace(-50, w + 50, 40), yy - amp * (.5 + .5 * np.sin(np.linspace(0, 9 + k * 3, 40) + k * 2)) - r.uniform(0, 10, 40)]; c.fill(np.vstack([spl(P, n=4), [(w + 50, 640), (-50, 640)]]), col)
        c.fill(np.array([(-50, 605), (w + 50, 605), (w + 50, h), (-50, h)]), '#d8b06a')
        road = np.array([(vx - 26, 606), (vx + 30, 606), (vx + 560, h), (vx - 900, h)]); c.fill(road, '#ecd09a'); c.fill(np.array([(vx - 26, 606), (vx - 14, 606), (vx - 620, h), (vx - 900, h)]), '#c79c5c')
        for k in range(150): q = r.random() ** 1.7; yy = 612 + q * 620; x0 = vx + (r.uniform(-1.3, 1.2)) * (30 + q * 820); ww = 12 + q * 160; c.fill(spl(np.array([(x0 - ww, yy), (x0, yy - 2 - q * 9), (x0 + ww, yy), (x0, yy + 2 + q * 7)]), True, n=4), mix(rgb('#f7e2ae'), rgb('#b98a4c'), r.random()), .55)
        L = c.layer(h, w); im = im * (1 - L[..., 3:4]) + L[..., :3]; im = repaint(im.astype(f32), (34, 15, 7), seed=3, jit=(.008, .02, .022), thr=(0, .012, .016), density=1.05, op=.9, dry=.2)
        # mid layer: houses, palms, grove either side of the road
        c = Cel(w, h)
        def house(x, yb, s, flip, seed):
            q = lambda P: np.array([(x + (u if not flip else -u) * s, yb - v * s) for u, v in P]); c.fill(q([(0, 0), (0, 1.1), (1.5, 1.0), (1.5, 0)]), '#b98a56'); c.fill(q([(1.5, 0), (1.5, 1.0), (2.1, .82), (2.1, .06)]), '#8a5f3a'); c.fill(q([(-.25, 1.02), (.5, 1.85), (1.75, 1.62), (2.35, .72), (1.6, .95)]), '#a5523a'); c.fill(q([(-.25, 1.02), (.5, 1.85), (.9, 1.78), (.5, 1.0)]), '#c9724c'); rr = rs(seed)
            for k in range(6): c.line(q([(lerp(-.15, 1.6, k / 5.), lerp(1.02, .95, k / 5.)), (lerp(.5, 1.75, k / 5.), lerp(1.8, 1.6, k / 5.))]), 2.4, '#6a3020', taper=(.1, .1), alpha=.8)
            for k in range(5): c.line(q([(0, .2 + k * .18), (1.5, .18 + k * .165)]), 2.0, '#7a5430', taper=(.1, .1), alpha=.7)
            c.fill(q([(.35, 0), (.35, .62), (.75, .60), (.75, 0)]), '#4a2f22'); c.fill(q([(1.0, .45), (1.0, .78), (1.32, .76), (1.32, .44)]), '#3c2a22'); c.line(q([(0, 0), (0, 1.1)]), 4, '#3a2418', taper=(0, 0)); c.line(q([(-.25, 1.02), (.5, 1.85), (1.75, 1.62), (2.35, .72)]), 4, '#3a1c14', taper=(.1, .1))
        def palm(x, yb, s, lean, seed):
            rr = rs(seed); P = spl(np.array([(x, yb), (x + lean * s * .3, yb - s * .55), (x + lean * s, yb - s * 1.1)]), n=12); c.line(P, s * .085, '#6a4a2c', taper=(0, .3)); c.line(P - [s * .014, 0], s * .030, '#c99a5c', taper=(0, .3)); top = P[-1]
            for k in range(9): a = -math.pi * (.02 + .96 * k / 8.) + rr.normal(0, .12); ln = s * rr.uniform(.5, .68); tip = top + [math.cos(a) * ln, math.sin(a) * ln * .45 + ln * .30]; m = top + [math.cos(a) * ln * .55, math.sin(a) * ln * .75 - ln * .06]; F_ = spl(np.array([top, m, tip]), n=12); nn = np.linspace(0, 1, len(F_)); ww = s * .085 * np.sin(np.pi * nn) ** .7; c.fill(np.vstack([F_ + np.c_[0 * nn, -ww * .4], (F_ + np.c_[0 * nn, ww * 1.1])[::-1]]), ('#7a7a3a', '#5c6430', '#989048')[k % 3]); [c.fill(np.array([F_[i], F_[i] + [4, ww[i] * 1.9], F_[min(i + 2, len(F_) - 1)]]), '#5c6430') for i in range(3, len(F_) - 2, 2)]
        for x, yb, s, fl, sd in ((1020, 690, 120, False, 1), (1990, 720, 150, True, 2), (2230, 790, 210, True, 3), (640, 735, 175, False, 4), (190, 800, 240, False, 5)): house(x, yb, s, fl, sd)
        for x, yb, s, ln, sd in ((1290, 650, 190, -.15, 1), (1410, 636, 120, .2, 2), (1830, 660, 220, .22, 3), (1960, 640, 150, -.2, 4), (900, 690, 320, .25, 5), (2120, 720, 380, -.25, 6), (420, 760, 420, .18, 7), (1730, 634, 110, -.1, 8)): palm(x, yb, s, ln, sd)
        for k in range(70):                                            # tangerine grove and grass
            side = -1 if r.random() < .55 else 1; q = r.random() ** 1.4; yy = 640 + q * 470; x0 = vx + side * (90 + q * 700 + r.uniform(0, 420 + q * 700)); s = 16 + q * 74; B = x0 + 0
            c.fill(spl(np.array([(x0 + math.cos(a) * s * 1.3 * (1 + .12 * math.sin(a * 5 + k)), yy - s * .7 + math.sin(a) * s * .8 * (1 + .10 * math.sin(a * 7))) for a in np.linspace(0, TAU, 22)[:-1]]), True, n=3), mix(rgb('#8f8a44'), rgb('#4f5a2c'), q)); c.fill(spl(np.array([(x0 + s * .2 + math.cos(a) * s * .9, yy - s * 1.0 + math.sin(a) * s * .5) for a in np.linspace(0, TAU, 14)[:-1]]), True, n=3), mix(rgb('#c9bd66'), rgb('#8f9444'), q), .8)
            for j in range(3): a_ = np.linspace(0, TAU, 12); fx, fy = x0 + r.normal(0, s * .6), yy - s * .7 + r.normal(0, s * .4); c.fill(np.c_[fx + np.cos(a_) * s * .13, fy + np.sin(a_) * s * .12], '#f0902c')
        L = c.layer(h, w); a = L[..., 3]; guide = L[..., :3] / np.maximum(a, 1e-3)[..., None]; pt = repaint(np.where(a[..., None] > .02, guide, rgb('#d8b06a')).astype(f32), (11, 6, 3.2), seed=6, jit=(.008, .03, .03), thr=(0, .012, .016), density=1.1, op=.85, dry=.14, mask=a); pt = pt + (rgb('#f6e0a4') - pt) * (np.clip((760 - gy) / 260., 0, 1) * .50)[..., None]; return im.astype(f32), np.dstack([pt * a[..., None], a]).astype(f32)

    def pinwheel(self, f):
        c = Cel(); P0 = np.array([168., 846.]); R = 128; spin = -.17 * f - float(keys(f, [(0, 0), (34, 0), (44, 3.2, lambda q: q * q), (60, 11.5, lambda q: q)])); c.line(np.array([(150, 1130), (162, 980), P0]), 17, INK, taper=(0, 0)); c.line(np.array([(149, 1130), (161, 980), P0]), 10, '#a9743c', taper=(0, 0)); cols = ('#d8432f', '#f4ead0', '#f0a92c', '#f4ead0')
        for k in range(8):
            a = spin + k * math.pi / 4; e = lambda r, q: P0 + r * R * np.array([math.cos(a + q), math.sin(a + q) * .94]); v = spl(np.array([P0, e(.62, -.10), e(1.0, .06), e(.96, .34), e(.60, .52), e(.30, .60)]), True, corners=(0, 2, 3), n=6); c.fill(v, cols[k % 4]); c.clip(v); c.fill(spl(np.array([P0, e(.55, .22), e(1.0, .34), e(.7, .7), e(.2, .7)]), True, n=5), '#6a3a2c', .42); c.unclip(); stroke(c, v, 4.4); c.line(np.array([P0, e(.9, .2)]), 2, INK, taper=(.1, .5))
        a = np.linspace(0, TAU, 24); c.fill(P0 + np.c_[np.cos(a), np.sin(a)] * 13, '#f7d25a'); stroke(c, P0 + np.c_[np.cos(a), np.sin(a)] * 13, 4); return lit(c.layer(), (-.82, .57), '#fff0b4', 3, 5, .9)

    def frame(self, f):
        d = f // 2 * 2; t = f / 24.; rk = float(keys(f, [(33, 0.), (40, 1.06), (44, 1.)])); pan = float(keys(f, [(0, 0.), (32, 22.), (44, 132.), (60, 168.)])); zm = 1.0 + .0007 * f + .012 * rk; br, nr = 12.5 * (1 - min(rk, 1)) ** 1.1, 13.5 * min(rk, 1.) ** 1.1
        base = cam2d(self.bg, 1090 + pan * .55, 620, 1.03 * zm).astype(f32); mid = cam2d(self.mid, 1100 + pan * .75, 620, 1.03 * zm, border=cv2.BORDER_CONSTANT); base = over(base, mid); gy0 = np.linspace(0, 1, H)[:, None, None]; base = base * (1 - .10 * (1 - gy0) - .16 * np.clip(gy0 - .55, 0, 1) * 2.); base = np.clip((base - .5) * 1.10 + .5, 0, 2)
        # haze + dust banks rolling
        x, y = grid(); dust = fbm(H // 8, (W + 1600) // 8, (46, 18, 8), None, 5); off = int(f * 2.2) // 1; ds = cv2.resize(dust[:, off // 8:off // 8 + W // 8], (W, H), interpolation=cv2.INTER_CUBIC); ds = np.clip(.5 + .42 * ds, 0, 1); hz = np.exp(-((y - 610) / 150.) ** 2); base = base + (rgb('#fff0c0') - base) * (hz * (.14 + .22 * ds))[..., None]; base = base + (rgb('#f3d690') - base) * (np.clip((y - 640) / 440., 0, 1) * ds * .18)[..., None]
        # the four
        gx0 = 1375 - pan * .8; ph = stride(f); walk = f >= 30; sc = scale(f); drawn = []; pj = lambda w: np.array([gx0 + w[0] / w[1], 592. + 250. / w[1]])
        for k, (who, ox, hh, dl) in enumerate((('usopp', 250, 432, .05), ('zoro', -290, 444, 0.), ('sanji', 92, 446, .02), ('luffy', -100, 428, 0.))):
            if who != 'luffy': X, Y = pj((ox, 1. / sc)); drawn.append((who, X, Y, self.men[who].draw(X, Y, hh * sc, ph - dl, f), 1.))
            elif f >= 25:                                              # past the lens and away: prints stay where they land
                w = trail(f); X, Y = pj(w); s = 1. / w[1]; pl = float(np.interp(f, LF, LP)); rest = 1 - min(1., (f - 25) / 29.); L = self.men[who].draw(X, Y, hh * s, pl, f, 2.9 * (s / 1.15) ** .6, amp=1. + 1.1 * rest, lean=-.05 * rest, feet={sd: pj(foot(pl, sd, w)) for sd in (-1, 1)}); dv = pj(trail(f + .5)) - [X, Y]
                drawn.append((who, X, Y, smear(L, dv[0], dv[1], 7) if abs(dv[0]) > 12 else L, s / 1.15))
        sh_ = np.zeros((H, W), f32)
        for who, X, Y, L, q_ in drawn: S_ = np.zeros((H, W), np.uint8); cv2.ellipse(S_, (int(X - 60 * q_), int(Y + 10 * q_)), (int(110 * q_), int(15 * q_)), 0, 0, 360, 255, -1, cv2.LINE_AA); sh_ = np.maximum(sh_, S_.astype(f32) / 255.)
        base = base * (1 - blur(sh_, 5)[..., None] * .42); G = np.zeros((H, W, 4), f32)
        for who, X, Y, L, q_ in drawn: G = lay(G, L)
        if walk:                                                       # dust at each footfall
            c = Cel()
            for b in (35, 45):
                a_ = f - b; r = rs(6 + b)
                if 0 <= a_ < 13:
                    for who, X, Y, L, q_ in drawn:
                        if who == 'luffy': X, Y = pj(foot(-.5, 1) if b == 35 else foot(0., -1))
                        elif b == 35: continue
                        for j in range(5): th = math.pi * (1.0 + r.random()); q = a_ / 13.; rad = (9 + 24 * q ** .6) * r.uniform(.6, 1.15) * (1.35 if b == 35 else 1.); px = X + r.uniform(-1, 1) * 34 + math.cos(th) * 70 * q ** .6; py = Y + 2 - rad * .3 - 22 * q * r.random(); e = np.linspace(0, TAU, 26)[:-1]; P = np.c_[px + np.cos(e) * rad * (1 + .10 * np.sin(e * 5 + j)), py + np.sin(e) * rad * .72 * (1 + .10 * np.sin(e * 4))]; c.fill(P, '#fbe9bb', .62 * (1 - q)); c.line(P[3:14], 2.0, '#b98a4c', taper=(.3, .4), alpha=.7 * (1 - q))
            G = lay(G, c.layer())
        base = over(base, G)
        if br > .6: base = defocus(base, br)
        base = base + (radial(1500 - pan, 420, 1150, 800, 1.5) * .13)[..., None] * rgb('#ffe6a0')
        xx = x / W; edge = np.clip(1.06 - .46 * np.clip((.62 - (x + pan) / W) / .62, 0, 1) ** 1.2 - .22 * np.clip((y / H - .62) / .38, 0, 1), .4, 1.1)[..., None]; base = base * (edge + (1 - edge) * rgb('#b5713c') * .55)
        for k, (xa, wd, am) in enumerate(((1480, 120, .16), (1180, 70, .10), (1720, 90, .12))):   # sun shafts through the dust
            u = (x - (xa - pan)) + (y + 200) * .62; base = base + (np.exp(-(u / (wd + 18 * math.sin(t * 1.3 + k))) ** 2) * np.clip(1 - y / 1000., 0, 1) * am * (.7 + .5 * ds))[..., None] * rgb('#ffe3a0')
        base = base + motes(f, 110, 4, None, '#fff3c4', (26, -8), 1.3, .55)
        # ------------------------------------------------ Nami
        def P(q): return np.array([BX - 0 + float(keys(q, [(0, 8.), (10, 4.), (12, -4., out3), (16, 2.), (30, 0.), (33, 5.), (60, 14.)])), BY + float(keys(q, [(0, 44.), (3, 34., out3), (6, 46.), (9, 37., out3), (11, 44.), (12, 66., out3), (14, 56.), (16, 64., out3), (19, 56.), (28, 6.), (30, -8.), (31, 12., out3), (33, 4.), (38, 14., out3), (42, 4.), (46, 14., out3), (50, 5.), (55, 13., out3), (60, 6.)]))])
        p = P(f) - [pan * 1.35, 0]; pitch = float(keys(f, [(0, -.30), (6, -.33), (11, -.30), (12, -.44, out3), (15, -.38), (16, -.43, out3), (19, -.38), (27, .05), (30, .12), (31, -.02, out3), (33, .03), (60, -.04)])); yaw = float(keys(f, [(0, .20), (12, .22), (20, .18), (28, .12), (33, .16), (46, .58), (60, .60)])); roll = float(keys(f, [(0, .10), (9, .07), (12, .13, out3), (20, .06), (28, -.05), (31, -.02), (40, .02), (60, .0)]))
        hitch = float(keys(f, [(0, .10), (3, .20, out3), (6, .08), (9, .20, out3), (12, .02), (28, .10), (30, .22), (31, .26, out3), (34, .10), (38, .24, out3), (42, .1), (46, .24, out3), (50, .08), (55, .2, out3), (60, .08)]))
        if f < 12: ex = dict(o=0., shut='tight', M='clench', clench=True, msx=1.0, msy=.9, bi=.11, bo=-.05, bk=.7, rope=.075, fore=.7, blush=.5, jit=.0035, mseed=d)
        elif f < 14: ex = dict(o=0., shut='tight', M='press', bi=.10, bo=-.03, bk=.5, rope=.075, blush=.5)
        elif f < 20: ex = dict(o=1.15, ir=.86, M='oh', msy=.9, bi=.10, bo=.02, bk=.15, pool=.40, wob=1., rope=.065, blush=.5, gaze=(.05, .35))
        elif f < 27: ex = dict(o=1.10, ir=1.06, M=('tremble' if (d // 2) % 2 else 'press'), bi=.12, bo=-.03, bk=.35, pool=.50, wob=1.2, rope=.08, blush=.5, gaze=(.15, .55), fore=.4)
        elif f < 30: ex = dict(o=.80, ir=.95, M=mmix('tremble', 'sob', .5 + .25 * (f - 27)), msx=1.1, clench=True, bi=.14, bo=-.06, bk=.6, pool=.75, wob=1., rope=.095, blush=.6, gaze=(.12, .4), fore=.8)
        elif f == 30: ex = dict(o=.25, shut='tight', M='clench', clench=True, bi=.14, bo=-.06, bk=.8, rope=.10, blush=.6, fore=1.)
        else: ex = dict(o=0., shut='tight', M=mmix('clench', 'sobopen', min(1., .45 + .25 * (f - 31))), msx=.94, msy=.88, teeth=(.078, .026), lip2=.040, mc=(0, -.595), tongue=.45, jaw=.14 + .03 * math.sin(f * 1.1), bi=.14, bo=-.06, bk=.8, rope=.115, rope2=1, fore=1., blush=.6, snot=.5, jit=.002, mseed=d)
        sway = np.array([-16 + 10 * math.sin(d * .35), 2]) + [0, (14 if 12 <= f < 14 else 0)]; st = N.style('dusk'); bnc = HT.settle(f, 12, .75) + HT.settle(f, 16, .28) + HT.settle(f, 31, -.30) + .05 * math.sin(f * .5); hat_on = f >= 12; shade_y = float(keys(f, [(12, -.75), (19, -.70), (27, -.12), (30, -.06), (31, -.20), (60, -.18)]))
        def post(c, F, C):
            if not hat_on: return
            y0 = shade_y; B = [(-1.1, y0 - .05), (-.6, y0 + .03), (-.2, y0 - .02), (.2, y0 + .05), (.6, y0), (1.1, y0 + .1)]; Q = np.vstack([spl(F.X(B), n=6), F.X([(1.2, 1.2), (-1.2, 1.2)])]); c.fill(Q, '#a4552f', .44); q0 = F.X([(.05, -.75)], .0)[0]; soft(c, q0[0], q0[1], S * .85, '#ffd68a', .42); r = rs(5)
            for k in range(15):                                        # light through the weave
                u, v = r.uniform(-.6, .7), r.uniform(y0 + .03, .40); a = np.linspace(0, TAU, 14); k2 = r.uniform(.5, 1.3); q = F.X(np.c_[u + np.cos(a) * .022 * k2, v + np.sin(a) * .010 * k2], .01); c.fill(q, '#ffdf9a', .75 * (.5 + .5 * math.sin(t * 2.4 + k * 1.9)))
        def body(c, F, s_):
            b = p * [1, 0] + [-6, lerp(BY, P(f)[1], .35)]; bust(c, b[0], b[1], S, roll * .35, s_, 'tee', .12, 5., -1, sh=hitch, length=3.2, f=f)
            c.fill(spl(np.array([b + [236, 352], b + [286, 346], b + [300, 400], b + [262, 452], b + [232, 420]]), True, n=6), '#7a1620', .85); [c.line(spl(np.array([b + [256 + k * 10, 440], b + [250 + k * 8, 486 + k * 12]]), n=5), 6 - k, '#7a1620', taper=(.05, .7), alpha=.85) for k in range(3)]
        D = N.draw(p[0], p[1], S, yaw, pitch, roll, None, 'dusk', d, hair='short', wind=(math.pi * .92, .18 + .12 * (f > 36)), ex=ex, body=body, posfn=lambda q: P(q), sway=sway, under=post, crown=not hat_on or True); F = D['F']
        k = HT.seat(F.h, w=HAT['w'], tilt=HAT['tilt'] + (-.05 if 12 <= f < 20 else 0) + float(keys(f, [(11, 0.), (12, -.10, out3), (14, 0)])), up=-.02); hb = hf = hand_ = None
        if f >= 4:
            if not hat_on: q = float(keys(f, [(4, 0.), (11, .90), (12, 1.)], lambda u: u)); k['cx'] += (1 - q) * 250; k['cy'] -= (1 - q) ** 1.2 * 430; k['roll'] += (1 - q) * .35; k['pitch'] = lerp(.15, k['pitch'], q)
            kw = dict(k, light='dusk', wear=.75, bounce=bnc if hat_on else -.10, f=f, wind=.5); hb = HT.hat(part='back', **kw); hf = HT.hat(part='front', **kw)
            if f < 24:
                pw = 0 if f < 17 else 1 if f < 19 else 2; lift = float(keys(f, [(12, 0.), (14, 4.), (16, -4.), (17, 4.), (20, -80.), (24, -520., lambda u: u * u)])); ox = float(keys(f, [(12, 0.), (17, 0.), (24, 260., lambda u: u * u)])); hand_ = np.zeros((H, W, 4), f32); crown = np.array([k['cx'], k['cy'] - k['w'] * .150]); rt = float(keys(f, [(4, -.10), (12, .05, out3), (15, 0.), (16, .05, out3), (17, 0.), (24, -.25)])); wx = np.array([crown[0] + 84 + ox, crown[1] - 40 + lift]); put(hand_, self.hs[pw], wx[0], wx[1], 1.16, rt)
                cr_, sr_ = math.cos(rt), math.sin(rt); RM = np.array([[cr_, -sr_], [sr_, cr_]]); dv = RM @ self.adir; nv = np.array([-dv[1], dv[0]]); L0 = self.alen * 1.16 - 34; w0, w1 = self.awid[0] * 1.16, self.awid[1] * 1.16; e0 = wx + dv * L0; e1 = wx + dv * (L0 + 620); cA = Cel(); A_ = spl([e0 + nv * w0, e0 + dv * 160 + nv * w0 * 1.12, e1 + nv * w0 * 1.5], n=8); B_ = spl([e0 + nv * w1, e0 + dv * 160 + nv * w1 * 1.10, e1 + nv * w1 * 1.5], n=8); q = np.vstack([A_, B_[::-1]]); cA.fill(q, '#eeb27c'); lo = A_ if A_[:, 1].mean() > B_[:, 1].mean() else B_; hi_ = B_ if lo is A_ else A_; cA.fill(np.vstack([lo, (lo + (hi_ - lo) * (.40 * sm(np.linspace(0, 1, len(lo)) / .42))[:, None])[::-1]]), '#a8683f'); cA.line(A_, 5, INK, taper=(0, 0)); cA.line(B_, 5, INK, taper=(0, 0)); hand_ = lay(hand_, lit(cA.layer(), (-.82, .57), '#fff0b4', 4, 8, .95))
        Ls = [hb, D['back'], D['body']]; own = np.zeros((H, W, 4), f32); b = p * [1, 0] + [-6, lerp(BY, P(f)[1], .35)]; c = Cel(); ax, ay = b + [238, 372 - hitch * 60]; A = spl(np.array([(ax - 620, ay + 330), (ax - 360, ay + 205), (ax - 110, ay + 70), (ax - 70, ay + 120), (ax - 300, ay + 300), (ax - 480, ay + 480), (ax - 700, ay + 480)]), True, n=6); c.fill(A, st['skin']); c.clip(A); c.fill(A + [-6, 62], st['shade']); c.unclip(); outline(c, A, 5, INK); own = lit(c.layer(), (-.5, .86), '#fff0b4', 4, 7, .85); put(own, self.own[1 if 14 <= f < 27 else 0], ax - 96, ay + 84, 1., float(keys(f, [(0, 0.), (12, .06, out3), (16, 0), (30, -.05), (31, .05, out3), (60, 0)])) + .012 * math.sin(f * 2.1) * (f > 28))
        layN = stack(*[l for l in (hb, D['back'], D['body'], own, D['face'], D['front'], hf) if l is not None])
        if hand_ is not None:                                       # contact shadow of the pressing hand on the straw
            sa = blur(cv2.warpAffine(hand_[..., 3], np.float32([[1, 0, -12], [0, 1, 20]]), (W, H)), 3.)[..., None] * (f < 19); layN[..., :3] = layN[..., :3] * (1 - .42 * sa * (layN[..., 3:4] > .5)); layN = lay(layN, hand_)
        layN = para(layN, (p[0] + 420, 200), (p[0] - 420, 1080), [(0, '#fffdf2'), (.45, '#fbe6c2'), (1, '#b98568')])
        # falling tears
        c = Cel()
        for i in range(26):
            t0 = 1 + i * 2.1 + rs(i, 2).random() * 2
            if t0 > 33: t0 += 1.5 * (i - 14)
            q = (f - t0)
            if 0 <= q < 12: r = rs(i, 9); sd = -1 if i % 2 else 1; e = F.X([(sd * .30, -.95)], .02)[0] + [r.normal(0, 14), 0]; y_ = e[1] + 5 * q + 2.1 * q * q; bead(c, e[0] + q * 1.2 * (1 - 2 * (i % 2)) + q * 1.5, y_, r.uniform(8, 13), math.pi / 2, .7, 2.6)
        layN = lay(layN, c.layer())
        if nr > .6: layN = defocus(layN, nr)
        base = over(base, layN)
        pw = self.pinwheel(f); pw = defocus(pw, 5.5 + 3 * rk); sx = -pan * 1.9; pw = cv2.warpAffine(pw, np.float32([[1, 0, sx * .35], [0, 1, 0]]), (W, H)); base = over(base, pw)
        base = flare(base, 1700 - pan, 300, 620, '#ffe9b0', .16)
        return np.clip(base, 0, 2.5).astype(f32)
