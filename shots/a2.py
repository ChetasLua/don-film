"""a2 SMILE FRAME: Roger.  SMILE holds the numbers g3 copies (exact from f58 on)."""
import numpy as np, math, cv2
from op import W, H, comp
from op.paint import rgb, f32, smooth, grid, fbm
from op.comp import over, screen, blur, bokeh, lerp_keys, ease, radial, cam2d
from op.ink import Cel, spl
from op.kit import para, outline, rim, flare
from op.fx_air import motes
from chars import roger as RG, scaffold as SC
from chars._old import A, rimlit, LIGHT, over_l

SMILE = dict(anchor=(0., -.70), cx=960., cy=640., s=645., yaw=.28, pitch=.18, roll=-.08, light='gold', sun=(-300., 300.))      # anchor = mouth centre in head3 front coordinates
KEYS = [(0, 'calm'), (4, 'calm'), (14, 'smile'), (29, 'smile'), (34, 'press'), (36, 'press'), (44, 'grin'), (49, 'grin'), (52, 'load'), (55, 'over'), (58, 'huge')]


def collar(f, k=1.):
    """red coat collar, two flaps in the wind (screen px)"""
    t = f // 2 * 2; pal = LIGHT['gold']; w = lambda i, a: A(a * math.sin(t * .62 - i * .9) + a * .4 * math.sin(t * 1.3 - i * 1.7), a * .7 * math.cos(t * .62 - i * .9 + .6))
    C = Cel(); red, dk, dk2, gold, gdk = '#8f1d22', '#511019', '#2c0a14', '#d8a03c', '#7e4e22'
    # far (right) flap: streams away to the right
    R = [(1105, 1120), (1140, 1010), A(1216, 905) + w(0, 5), A(1330, 838) + w(1, 9), A(1480, 800) + w(2, 14), A(1640, 800) + w(3, 20), A(1800, 762) + w(4, 26), A(1990, 776) + w(5, 30), (2050, 1160)]
    P = spl(R, True, (0, 8)); C.fill(P, dk); C.clip(P)
    C.fill(spl([(1180, 1130), A(1290, 980) + w(1, 8), A(1480, 915) + w(2, 12), A(1700, 925) + w(3.4, 18), A(1960, 880) + w(4.6, 22), (2060, 1160)], True, (0, 5)), red)
    for i in range(7):
        x = 1250 + i * 118; C.line(spl([A(x + 150 + i * 4, 838 - i * 8) + w(i * .7 + 1, 12), A(x + 92, 960) + w(i * .7, 8), A(x + 20, 1060), A(x - 30, 1140)], n=14), 46 - 14 * (i % 2), dk2, taper=(.85, .05), press=.3, seed=90 + i, alpha=.85); C.line(spl([A(x + 160 + i * 4, 846 - i * 8) + w(i * .7 + 1, 12), A(x + 112, 960) + w(i * .7, 8), A(x + 44, 1060), A(x - 2, 1140)], n=14), 3., pal['line'], taper=(.5, .1), seed=120 + i)
    C.unclip()
    e = [A(1216, 905) + w(0, 5), A(1330, 838) + w(1, 9), A(1480, 800) + w(2, 14), A(1640, 800) + w(3, 20), A(1800, 762) + w(4, 26), A(1990, 776) + w(5, 30)]
    T = spl(e, n=14); C.line(T + A(0, 13), 30, gdk, taper=(.1, .02), press=.1, seed=2); C.line(T + A(0, 9), 20, gold, taper=(.1, .02), press=.1, seed=2); C.line(T, 6, pal['line'], taper=(.2, .02), seed=3); C.line(T + A(2, 30), 3.4, pal['line'], taper=(.2, .02), seed=4)
    # near (left) flap
    L = [(520, 1130), (500, 1040), A(420, 930) + w(6, 5), A(300, 868) + w(7, 9), A(150, 850) + w(8, 13), A(20, 884) + w(9, 14), (-60, 900), (-60, 1140)]
    P = spl(L, True, (0, 6, 7)); C.fill(P, red); C.clip(P)
    C.fill(spl([(540, 1140), A(440, 1000) + w(6, 4), A(300, 968) + w(7, 8), A(170, 1010) + w(8, 8), (120, 1140)], True, (0, 4)), dk)
    for i in range(4):
        x = 60 + i * 110; C.line(spl([A(x + 30, 888) + w(i + 6.6, 8), A(x + 4, 990) + w(i + 6, 6), A(x - 30, 1130)], n=14), 40, dk2, taper=(.85, .05), press=.3, seed=140 + i, alpha=.8)
    C.unclip()
    e = [(508, 1060), A(420, 930) + w(6, 5), A(300, 868) + w(7, 9), A(150, 850) + w(8, 13), A(20, 884) + w(9, 14), (-60, 900)]
    T = spl(e, n=14); C.line(T + A(2, 14), 30, gdk, taper=(.06, .02), press=.1, seed=5); C.line(T + A(1, 9), 20, gold, taper=(.06, .02), press=.1, seed=5); C.line(T, 6, pal['line'], taper=(.1, .02), seed=6); C.line(T + A(6, 31), 3.4, pal['line'], taper=(.1, .02), seed=7)
    for i in range(16):                                                            # rope fringe of the epaulette, left corner
        x = -20 + i * 17; d = w(i * .4 + 9, 7); C.line(spl([A(x, 985), A(x + 5, 1030) + d * .5, A(x + 10, 1090) + d], n=8), 13, gdk if i % 2 else gold, taper=(.05, .2), press=.15, seed=30 + i); C.line(spl([A(x - 5, 985), A(x, 1030) + d * .5, A(x + 5, 1090) + d], n=8), 2.6, pal['line'], seed=60 + i)
    C.line(spl([(-30, 975), (120, 962), (258, 1000)], n=10), 26, gdk, taper=(.02, .1), seed=8); C.line(spl([(-30, 970), (120, 957), (254, 994)], n=10), 15, gold, taper=(.02, .1), seed=8)
    lay = C.layer(); lay = para(lay, (0, 800), (1500, 1080), [(0, '#ffdcae'), (.35, '#d9a79a'), (1, '#6f6a92')]); return rimlit(lay, .9, .7, pal, 1., 1., 22, 8)


class Shot:
    id = 'a2'; n = 60; finish = dict(bloom=.34, vig=.20, soft=.16)
    EVENTS = [(0, 'wind', dict(note='high open wind on the scaffold, cloth and hair, whole shot', len=60)), (0, 'cloth', dict(note='coat collar flapping, about 2.4 flaps/s', len=60)), (8, 'spear_rise', dict(note='left spear lifted, wood + steel whisper', len=22)), (14, 'spear_rise', dict(note='right spear lifted', len=22)),
              (36, 'steel', dict(size=.35, note='two spear shafts cross behind his head')), (34, 'breath', dict(note='breath in through the nose, lips pressed')), (45, 'grin', dict(size=.6, note='grin opens, teeth')), (48, 'glint', dict(size=.8, note='sun on the blades')), (55, 'grin', dict(size=1.0, note='enormous grin, eyes come out of the shadow into lens; low drum + inhaled silence'))]

    def setup(self):
        s = np.asarray(SC.sky(), f32); self.bg = cv2.GaussianBlur(s, (0, 0), 13); self.ml = radial(40, 330, 1500, 1100, 1.0)

    def rig(self, f):
        fp = f if 49 <= f <= 58 else f // 2 * 2
        z = float(lerp_keys(fp, [(0, .950), (50, .992), (55, 1.012), (58, 1.0)]))
        k = SMILE; yaw = float(lerp_keys(fp, [(0, .235), (29, .245), (34, .225), (55, .285), (58, .28)])); pitch = float(lerp_keys(fp, [(0, .05), (14, .065), (29, .06), (35, .005), (45, .115), (52, .085), (55, .20), (58, .18)])); roll = float(lerp_keys(fp, [(0, -.045), (29, -.05), (35, -.03), (45, -.065), (52, -.055), (55, -.092), (58, -.08)]))
        return fp, z, dict(cx=0, cy=0, s=k['s'] * z, yaw=yaw, pitch=pitch, roll=roll, anchor=k['anchor'])

    def frame(self, f):
        fp, z, r = self.rig(f); e = RG.pose_at(fp, KEYS); gust = float(lerp_keys(fp, [(0, 0), (30, .1), (42, .35), (48, .7), (55, 1.), (59, .85)]))
        k = SMILE; cy = k['cy'] + float(lerp_keys(fp, [(0, 10), (35, 16), (55, -4), (58, 0)]))
        fx_kw = dict(r); a = fx_kw.pop('anchor'); s = fx_kw.pop('s'); fx_kw.pop('cx'); fx_kw.pop('cy')
        LY = RG.draw(k['cx'], cy, s, expr=e, light='gold', f=f, anchor=a, wind=(1., 0.), gust=gust, rimdir=(1., .12), **fx_kw)
        # background: blurred iron sky, sun far left
        img = cam2d(self.bg, 1180 + f * .45, 690 - f * .35, 1.12)
        x, y = grid(); img = img * (1 - .30 * smooth(700, 1900, x)[..., None]); img = screen(img, (radial(40, 330, 1150, 900, 1.8))[..., None] * rgb('#ffcf84') * .85)
        # spears, defocused
        sp = Cel(ss=1); l = float(lerp_keys(f, [(6, 0), (30, 1.04), (36, 1.)])); q = float(lerp_keys(f, [(13, 0), (34, 1.05), (40, 1.)])); c = float(lerp_keys(f, [(30, 0), (38, 1.06), (42, 1.)]))
        for (tx, ty, an, l_) in ((1720, 64, -.505, l), (168, 96, math.pi + .495, q)):
            an = an + (.16 if tx > 900 else -.16) * (1 - c); d_ = A(math.cos(an), math.sin(an)); o = A(tx, ty) - d_ * (2300 + 980 * (1 - l_)); SC.spear(sp, o[0], o[1], an, 2300, 2.2, w=13, steel='#aeb4b4', steel_hi='#fffbe8')
        sl = para(sp.layer(), (0, 400), (1900, 700), [(0, '#ffe0aa'), (.4, '#b9b0aa'), (1, '#6a7690')]); sl = rimlit(sl, 1., .3, LIGHT['gold'], 1., .9, 16, 6); img = over(img, bokeh(sl, 8, 1.2))
        if 44 <= f <= 57:
            g = math.exp(-abs(f - 48) / (1.2 if f < 48 else 3.2)); img = flare(img, 1610, 126, 210, '#fff6da', 1.25 * g, streak=1.4, ang=-.5); img = flare(img, 262, 148, 150, '#fff0c8', .6 * g, streak=.8, ang=-.5)
        rl = 1 - .18 * smooth(54, 58, f)
        g0 = ('#ffe9c8', '#d8a58c', '#7c6f93')
        pr = lambda L, a=1.: para(L, (250, 120), (1330, 980), [(0, '#fff2dc'), (.36, '#ffdcbc'), (.62, '#e9b598'), (1, '#8f7f9c')], amt=a)
        mm = (LY['mm'] * .72)[..., None]; back, face, st, hr = pr(LY['back']), pr(LY['face']) * (1 - mm) + LY['face'] * mm, pr(LY['stache'], .6), pr(LY['hair'], .6)
        al = np.maximum(face[..., 3], hr[..., 3]); img = screen(img, np.clip(blur(al, 40) * 1.3, 0, 1)[..., None] * rgb('#ffb765') * .22 * (1 - smooth(500, 1500, x))[..., None])
        img = over(img, back); img = over(img, face); img = over(img, hr); img = over(img, st); img = over(img, collar(f))
        img = screen(img, (radial(-60, 300, 900, 1000, 2.2))[..., None] * rgb('#ffbe6a') * .50)
        img = img + motes(f, 70, seed=12, light=self.ml, col="#ffe6b4", drift=(44., -10.), size=1.5, amt=.8)
        return img
