"""Luffy: grown head + bust, child head + bust.  Recorded drawing (Rec) -> rendered as theatrical cel, charcoal, or pencil construction.

head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='calm', light='day', f=0, hat=True, body=True, ...) -> premult RGBA (1080,1920,4)
    (cx, cy) = middle of the eye line inside the head, s = px from eye line to chin (as op.head3).  expr = name | dict | (a, b, t) blend.
    EXPR: calm smile grin laugh worry glare hollow shock scream battered cry | child: lip howl kidgrin kidcry
    hat=True worn | False bare | 'back' hanging behind the shoulders on its cord.   hk=dict(tilt=, bounce=, wear=, up=...) for the hat.
    light: day dusk gold fire storm night sea flat.   shirt='red' open cardigan | 'vest'.   scar=True the X on the chest.   wind=amount of hair sway.
kid_head(...)  same arguments, child proportions, anchor tee, oversize hat.
SMILE = the numbers shared with a2 / g3.      smile_pose(Head, s) poses a head so that its mouth sits on (960, 640).
"""
import numpy as np, cv2, math, cairo
from op import W, H
from op.paint import rgb, f32, smooth, mix, fbm
from op.ink import Cel, spl, resample, paste, boil, noise1
from op.kit import outline, spike, hatch as khatch, para
from op.head3 import Head
from op.comp import over
from op import hat as HT

SMILE = dict(mouth=(960, 640), yaw=.28, pitch=.18, roll=-.08, s=645., mouth_y=-.70)          # = shots/a2.SMILE
TAU = 2 * math.pi


class LHead(Head):
    """head3.Head with a hinged jaw (units of chin drop) and drawn stretch"""
    def pose(self, *a, jaw=0., st=0., **k): self.jw, self.st = jaw, st; return super().pose(*a, **k)

    def P3(self, Q):
        Q = np.array(Q, float); ck = self.k.get('chin', 1.)
        if ck < 1: Q[..., 1] = Q[..., 1] - np.minimum(0, Q[..., 1] + .22) * (1 - ck)
        if self.jw:
            d = np.clip(-.34 - Q[..., 1], 0, .66) / .66; Q[..., 1] = Q[..., 1] - self.jw * d ** 1.15; Q[..., 2] = Q[..., 2] - self.jw * .22 * d
        if self.st: Q[..., 1] = Q[..., 1] * (1 + self.st); Q[..., 0] = Q[..., 0] * (1 - .45 * self.st)
        return super().P3(Q)


GROWN = dict(a=[(-1.0, .15), (-.97, .27), (-.91, .385), (-.81, .50), (-.67, .61), (-.5, .695), (-.3, .755), (-.1, .785), (.12, .80), (.35, .80), (.6, .75), (.8, .60), (.93, .38), (1.0, .06)],
             nose=(.09, -.14, -.325, -.385, .112), ear=(.785, .06, -.38, -.10), chin=.90)
KID = dict(a=[(-1.0, .22), (-.96, .40), (-.88, .55), (-.75, .69), (-.6, .78), (-.4, .845), (-.2, .86), (.05, .85), (.35, .83), (.6, .76), (.8, .60), (.93, .38), (1.0, .06)],
           nose=(.08, -.22, -.40, -.45, .10), ear=(.83, .02, -.40, -.10), p=2.1)

# ------------------------------------------------------------------ expressions
DEF = dict(lu=1., ll=1., ci=.5, co=.5, lc=-.5, gx=0., gy=0., pup=.052, hl=1., arc=0., bi=.285, bm=.345, bo=.30, bw=1., bx=0.,
           mw=.17, mx=0., yc=-.62, sk=0., uy=-.615, ly=-.622, pu=2., pl=2., tt=0., tb=0., tg=0., wob=0., bite=0., jaw=0., st=0.,
           fur=0., nos=0., fore=0., naso=0., sh=0., tear=0., sweat=0., blood=0., snot=0., blush=0., lip=0., strain=0., es=1.07, hat=1.)
EXPR = dict(
    calm={},
    smile=dict(mw=.33, yc=-.555, uy=-.655, ly=-.665, naso=0., bi=.30, bm=.36),
    grin=dict(mw=.565, yc=-.43, uy=-.56, ly=-.875, pu=1.7, pl=1.75, bite=1., naso=.55, bi=.33, bm=.40, bo=.34, pup=.05, blush=.5, lc=-.135),
    laugh=dict(mw=.52, yc=-.42, uy=-.50, ly=-.93, pu=1.8, pl=1.8, tt=.13, tg=.22, jaw=.06, lu=.08, arc=1., bi=.36, bm=.42, bo=.35, naso=.9, blush=.8),
    worry=dict(es=1.14, hat=.0, bw=.82, mw=.16, yc=-.665, uy=-.60, ly=-.70, pu=2, pl=2, tt=.04, bi=.40, bm=.385, bo=.28, lu=1.05, sweat=.6, pup=.048),
    glare=dict(lu=1., ci=.030, co=.150, pup=.043, gy=.0, hl=.4, bi=.105, bm=.235, bo=.345, bw=1.15, mw=.215, yc=-.675, uy=-.635, ly=-.642, fur=1., nos=.6, sh=.4),
    hollow=dict(es=1.24, hat=0., bw=.78, lu=1.14, ll=1.10, pup=.021, hl=0., bi=.375, bm=.375, bo=.31, mw=.105, yc=-.67, uy=-.635, ly=-.715, pu=2, pl=2, tt=.02),
    shock=dict(es=1.22, hat=0., bw=.8, lu=1.18, ll=1.15, pup=.026, hl=.2, bi=.42, bm=.45, bo=.38, mw=.20, yc=-.66, uy=-.56, ly=-.80, tt=.06, tg=.05, jaw=.08, sweat=.5),
    scream=dict(lu=1.16, ll=1.06, ci=.40, co=.20, es=1.26, hat=0., pup=.022, hl=0., bi=.45, bm=.395, bo=.26, bw=.72, mw=.40, yc=-.70, uy=-.475, ly=-.935, pu=3.4, pl=3.2, tt=.12, tb=.055, tg=.27, jaw=.25, fore=0., nos=0., naso=0., tear=1., fur=.8, snot=0., strain=0.),
    battered=dict(ci=.025, co=.145, pup=.041, hl=.25, bi=.10, bm=.23, bo=.345, bw=1.15, mw=.22, yc=-.685, uy=-.64, ly=-.645, fur=1., nos=.7, sh=.55, blood=1.),
    cry=dict(lu=.9, ll=.6, bi=.42, bm=.35, bo=.22, mw=.36, yc=-.72, uy=-.56, ly=-.80, pu=3, pl=3, tt=.12, tb=.05, wob=.016, tear=.9, snot=.7, fore=.6, naso=.6, blush=.4),
    lip=dict(lu=.08, arc=-.6, bi=.40, bm=.33, bo=.22, mw=.175, yc=-.66, uy=-.605, ly=-.612, wob=.016, lip=1., blush=.7, fore=.4),
    howl=dict(lu=.08, arc=-1., bi=.46, bm=.36, bo=.20, mw=.43, yc=-.70, uy=-.45, ly=-.97, pu=3.2, pl=2.8, tt=.10, tb=.06, tg=.27, wob=.016, jaw=.16, tear=1., snot=1., fore=.8, naso=.8, blush=1., strain=.4),
    kidgrin=dict(mw=.50, yc=-.47, uy=-.56, ly=-.86, pu=1.8, pl=1.8, bite=1.5, naso=.6, bi=.22, bm=.30, bo=.26, blush=1., pup=.06),
    kidcry=dict(lu=.85, ll=.8, bi=.36, bm=.28, bo=.16, mw=.30, yc=-.74, uy=-.60, ly=-.80, pu=3, pl=3, tt=.12, wob=.018, tear=1., snot=.8, blush=1.))


def ex(e):
    if isinstance(e, str): return dict(DEF, **EXPR[e])
    if isinstance(e, dict): return dict(DEF, **e)
    a, b, t = ex(e[0]), ex(e[1]), float(e[2]); return {k: a[k] * (1 - t) + b[k] * t for k in DEF}


# ------------------------------------------------------------------ colour
BASE = dict(skin='#f0b98d', skin_sh='#c9805c', skin_dk='#97513f', skin_hi='#ffe0ba', hair='#241f2b', hair_sh='#131018', hair_hi='#4b4f70', hair_ln='#615f82', white='#fbf6ee', white_sh='#bfb7c6',
            pupil='#0c090d', mouth='#4b0e17', tongue='#df6c6a', tongue_sh='#a8434b', teeth='#fffaf0', teeth_sh='#cfc5cc', shirt='#c3272f', shirt_sh='#7d141f', shirt_dk='#4e0b16', shirt_hi='#e8574e',
            scar='#c98064', scar_sh='#a0563f', blush='#e4876a', tear='#eaf6ff', tear_sh='#9cc3e4', blood='#6b1017', line='#17131a', soft='#7c4034', lip='#a85644', tee='#eaf0f3', tee_sh='#9fb4c8',
            anchor='#2d5f9c', anchor_sh='#1c3f6e', sash='#e9ba3c', cord='#d9b56a', cord_sh='#8d6a32')


def _pal(mul=(1, 1, 1), sh=(1, 1, 1), amb=None, line='#17131a', soft=None, **k):
    o = {}
    for q, v in BASE.items():
        c = rgb(v); m = np.array(sh if q.endswith(('_sh', '_dk')) or q in ('hair', 'mouth') else mul, f32)
        if q.split('_')[0] in ('white', 'teeth', 'tear'): m = 1 - (1 - m) * .55
        c = c * m
        if amb: c = c * (1 - amb[1]) + rgb(amb[0]) * amb[1] * (.35 + .9 * float(c.mean()))
        o[q] = c
    o['line'] = rgb(line)
    if soft: o['soft'] = rgb(soft)
    return dict(c=o, **k)


LIGHT = dict(
    day=_pal(rim='#fff6dd', rd=(-1, -.8), side=1, cover=0., para=[(0, '#ffffff'), (.5, '#fdf3ee'), (1, '#dcbcbc')], hat='day', rw=1.),
    dusk=_pal((1.02, .93, .82), (.95, .84, .90), rim='#ffdf9c', rd=(1, -.45), side=-1, cover=.30, para=[(0, '#fff2d6'), (.5, '#fbe0cc'), (1, '#a98496')], hat='dusk', rw=1.25, line='#1c1014'),
    gold=_pal((1.02, .91, .78), (.90, .78, .82), rim='#ffd88a', rd=(-1, -.55), side=1, cover=.45, para=[(0, '#ffeccb'), (.5, '#f2cdb6'), (1, '#8a6878')], hat='gold', rw=1.4, line='#1a0e12'),
    fire=_pal((1.05, .95, .80), (.93, .84, .88), rim='#ffd88c', rd=(1, -.40), side=-1, cover=.60, para=[(0, '#fff6e2'), (.45, '#f4d8cb'), (1, '#9a7180')], hat='fire', rw=1.5, line='#1a090c', soft='#4a1c1c'),
    storm=_pal((.80, .84, .92), (.62, .66, .80), ('#6f87a8', .25), rim='#eaf6ff', rd=(1, -.9), side=-1, cover=.25, para=[(0, '#ffffff'), (1, '#8694b4')], hat='storm', rw=1.2, line='#0e0f18'),
    night=_pal((.66, .68, .84), (.50, .52, .72), ('#4a5a96', .35), rim='#c4d8ff', rd=(-1, -.8), side=1, cover=.3, para=[(0, '#ffffff'), (1, '#6c74a4')], hat='night', rw=1.1, line='#0c0c16'),
    dawn=_pal((1.05, .985, .90), (1.0, .90, .93), rim='#fff0c4', rd=(-1, -.40), side=1, cover=.20, para=[(0, '#fffaf0'), (.5, '#fff2e6'), (1, '#d2adb4')], hat='gold', rw=1.5, line='#1a1014'),
    deep=_pal((.74, .78, .76), (.30, .42, .56), ('#1d5f96', .20), rim='#dcfcff', rd=(.38, -.92), side=1, cover=.30, para=[(0, '#ffffff'), (.5, '#9fc0d8'), (1, '#33517e')], hat='sea', rw=1.1, line='#040c1c', soft='#0c2744'),
    sea=_pal((.80, .90, .92), (.46, .60, .74), ('#2f7fae', .24), rim='#dcfcff', rd=(.15, -1), side=-1, cover=.35, para=[(0, '#ffffff'), (1, '#2f5a88')], hat='sea', rw=1.2, line='#06101e', soft='#123048'),
    flat=_pal(rim='#fff6dd', rd=(-1, -.8), side=1, cover=0., para=None, hat='flat', rw=1.))


# ------------------------------------------------------------------ recorder
class Rec:
    def __init__(s): s.parts = []

    def part(s, name, rim=1., para=1.): s.cur = dict(name=name, ops=[], rim=rim, para=para); s.parts.append(s.cur); return s

    def fill(s, P, key, clip=None, al=1., eo=False): s.cur['ops'].append(('f', P if isinstance(P, list) else [np.asarray(P, float)], key, clip, al, eo)); return s

    def line(s, P, w, key='line', taper=(.25, .25), press=.28, al=1., clip=None, seed=0, tmin=.15, kind='in'): s.cur['ops'].append(('l', [np.asarray(P, float)], key, clip, al, w, taper, press, seed, tmin, kind)); return s

    def edge(s, P, w, key='line', seed=0, clip=None, breaks=0, al=1., kind='out', press=.42): s.cur['ops'].append(('o', [np.asarray(P, float)], key, clip, al, w, seed, breaks, kind, press)); return s

    def hatch(s, P, ang, gap, w, key='line', length=None, fade=0., al=1., seed=0, clip=None): s.cur['ops'].append(('h', [np.asarray(P, float)], key, clip, al, ang, gap, w, length, fade, seed)); return s

    # ---------------- renderers
    @staticmethod
    def _box(ops, pad=26):
        Q = np.vstack([p for o in ops for p in o[1]]); lo = np.floor(Q.min(0)).astype(int) - pad; hi = np.ceil(Q.max(0)).astype(int) + pad
        x0, y0 = max(-60, lo[0]), max(-60, lo[1]); x1, y1 = min(W + 60, hi[0]), min(H + 60, hi[1])
        return (x0, y0, x1, y1) if x1 - x0 > 3 and y1 - y0 > 3 else None

    def cel(s, L, s_px, out=None, box=None, only=None, skip=()):
        """theatrical cel: flats, hard shadow, pen line, airbrush gradient, hot rim, part over part"""
        C = L['c']; out = np.zeros((H, W, 4), f32) if out is None else out
        bx = [s._box(p['ops']) for p in s.parts if p['ops'] and p['name'] not in skip]; bx = [b for b in bx if b]
        if not bx: return out
        gx = box or (min(b[0] for b in bx), min(b[1] for b in bx), max(b[2] for b in bx), max(b[3] for b in bx))
        rd = np.array(L['rd'], float); rd /= np.hypot(*rd)
        for p in s.parts:
            if not p['ops'] or p['name'] in skip or (only and p['name'] not in only): continue
            b = s._box(p['ops'])
            if not b: continue
            c = Cel(b[2] - b[0], b[3] - b[1], b[0], b[1])
            for o in p['ops']:
                col = C[o[2]] if o[2] in C else o[2]
                if o[3] is not None: c.clip(o[3])
                if o[0] == 'f': c.fill(o[1], col, o[4], evenodd=o[5])
                elif o[0] == 'l': c.line(o[1][0], o[5], col, taper=o[6], press=o[7], alpha=o[4], seed=o[8], tmin=o[9], tremor=.2)
                elif o[0] == 'o': outline(c, o[1][0], o[5], col, seed=o[6], breaks=o[7], press=o[9], tremor=.2)
                elif o[0] == 'h': khatch(c, o[1][0], o[5], o[6], o[7], col, o[10], o[8], o[9], alpha=o[4])
                if o[3] is not None: c.unclip()
            a = c.rgba()
            if L.get('para') and p['para']:
                cx, cy, R = (gx[0] + gx[2]) / 2, (gx[1] + gx[3]) / 2, min(1500, max(gx[2] - gx[0], gx[3] - gx[1])) * .46
                a = para(a, (cx + rd[0] * R - b[0], cy + rd[1] * R - b[1]), (cx - rd[0] * R - b[0], cy - rd[1] * R - b[1]), L['para'], amt=p['para'])
            if p['rim'] > 0: a = rimlight(a, rd, np.clip(s_px * .019 * L['rw'], 1.2, 15) , L['rim'], p['rim'])
            x0, y0 = max(0, b[0]), max(0, b[1]); x1, y1 = min(W, b[2]), min(H, b[3])
            if x1 <= x0 or y1 <= y0: continue
            q = a[y0 - b[1]:y1 - b[1], x0 - b[0]:x1 - b[0]]; d = out[y0:y1, x0:x1]; out[y0:y1, x0:x1] = d * (1 - q[..., 3:4]) + q
        return out

    GREY = dict(skin=.56, skin_sh=.36, skin_dk=.24, skin_hi=.66, hair=.13, hair_sh=.08, hair_hi=.26, hair_ln=.34, white=.80, white_sh=.52, pupil=.06, mouth=.10, tongue=.36, tongue_sh=.22, teeth=.82, teeth_sh=.55,
                scar=.46, scar_sh=.3, blush=.50, tear=.90, tear_sh=.6, blood=.2, soft=.2, lip=.3, tee=.7, tee_sh=.45, anchor=.3, anchor_sh=.2, sash=.5, cord=.5, cord_sh=.3)

    def charcoal(s, seed, fills=1., loose=0., tone=1., detail=1., red=1., redfill=None, line=1., base=(.96, .94, .92), w=1.25, shade_hatch=.0, skip=()):
        """charcoal drawing with one red.  fills 1 -> 0 rubs the tones away (torn by grain), loose 0..1 turns the line into searching strokes, detail drops inner lines"""
        r = np.random.default_rng(seed); Lf = np.zeros((H, W, 4), f32); Cf = Cel(W, H, ss=1); Cl = Cel(W, H, ss=1); Cr = Cel(W, H, ss=1); Cq = Cel(W, H, ss=1); redfill = fills if redfill is None else redfill; OUT = cairo.OPERATOR_DEST_OUT; OV = cairo.OPERATOR_OVER
        G = _grain(0); k = np.random.default_rng(99); order = 0; FR = ()
        for p in s.parts:
            if p['name'] in skip: continue
            if p['ops'] and p['name'] in ('face', 'neck', 'body'):     # a part hides the ink of the parts behind it
                o0 = [o for o in p['ops'] if o[0] == 'f'][0]; Cl.ctx.set_operator(OUT); Cl.fill(o0[1], '#000000', 1 - .85 * min(1, loose * 1.4)); Cl.ctx.set_operator(OV)
            for o in p['ops']:
                key = o[2]; isred = str(key).startswith('shirt') or key == 'blood'; order += 1
                if o[0] == 'f':
                    g = s.GREY.get(key, .4 if not isred else .3); col = np.array(base, f32) * g * tone
                    if isred: col = rgb({'shirt': '#a81a22', 'shirt_hi': '#c4343a', 'shirt_sh': '#741018', 'shirt_dk': '#45090f', 'blood': '#7a0f16'}[key])
                    off = r.normal(0, 1.2 + 7 * loose, 2); pts = [q + off for q in o[1]]
                    if p['name'] in FR and not isred:              # nearer shapes hide the lines behind them
                        Cl.ctx.set_operator(OUT); Cl.fill(pts, '#000000', max(.25, 1 - loose)); Cl.ctx.set_operator(OV)
                    for cc, op_ in ((Cq, OV if isred else OUT), (Cf, OV)) if not isred else ((Cq, OV),):
                        if o[3] is not None: cc.clip(o[3])
                        cc.ctx.set_operator(op_); cc.fill(pts, col, o[4] if cc is Cf or isred else min(1., o[4]), evenodd=o[5]); cc.ctx.set_operator(OV)
                        if o[3] is not None: cc.unclip()
                    # what the tone looks like as a line when the fill is gone
                    if p['name'] in ('face', 'neck', 'body', 'ear') and key in ('skin_sh', 'skin_dk') and shade_hatch > 0 and len(o[1][0]) > 5:
                        if o[3] is not None: Cl.clip(o[3])
                        khatch(Cl, o[1][0], .95, 13, 2.0, '#141110', int(r.integers(1 << 20)), 70, .25, alpha=.55 * shade_hatch)
                        if o[3] is not None: Cl.unclip()
                    continue
                if o[0] == 'h':
                    continue
                kind = o[-1] if o[0] == 'l' else o[8]
                if o[0] == 'l' and k.random() > detail and kind != 'key': continue
                P = o[1][0]
                if len(P) < 3: continue
                n = 1 + int(loose * 2.2 + r.random()); dst = Cr if isred else Cl
                for j in range(n):
                    Q = resample(P, 9.); m = len(Q)
                    if m < 3: continue
                    amp = (.6 + 9 * loose) * (1 if j else .55); Q = Q + np.c_[noise1(m, 16, r), noise1(m, 16, r)] * amp + r.normal(0, amp * .5, 2)
                    if loose > .25:                                # strokes run past their ends, as when you look for the form
                        e = (6 + 26 * loose) * r.random(2); a = Q[0] - Q[min(3, m - 1)]; b_ = Q[-1] - Q[max(-4, -m)]
                        Q = np.vstack([Q[0] + a / (np.hypot(*a) + 1e-6) * e[0], Q, Q[-1] + b_ / (np.hypot(*b_) + 1e-6) * e[1]])
                        if len(Q) > 60:
                            c0 = int(r.integers(0, 30)); Q = Q[c0:len(Q) - int(r.integers(0, 20))]
                    ww = o[5] * w * (1 - .45 * loose) * r.uniform(.7, 1.3) / (1 + .25 * j)
                    if o[3] is not None and loose < .3: dst.clip(o[3])
                    dst.line(Q, max(1.0, ww), '#141110' if not isred else '#5c0b12', taper=(.2, .2), press=.5, seed=int(r.integers(1 << 20)), alpha=.95 - .18 * loose, tremor=.8)
                    if o[3] is not None and loose < .3: dst.unclip()
        Lf = Cf.rgba()
        if fills < 1:                                              # rubbed out in the direction of a hand wiping
            m = np.clip((fills * 1.9 - .78 + .30 * _smudge(1)) * 5.0, 0, 1)[..., None]; Lf = Lf * m
        rq = Cq.rgba()
        if redfill < 1: rq = rq * np.clip((redfill * 1.9 - .78 + .30 * _smudge(3)) * 5.0, 0, 1)[..., None]
        Lf = over4(Lf, rq) * (.80 + .20 * G[..., None]); ln = Cl.rgba() * line * np.clip(G * 1.25 + .05, 0, 1)[..., None]; rd_ = Cr.rgba() * np.clip(G * 1.2 + .1, 0, 1)[..., None]
        out = over4(over4(Lf, ln), rd_)
        return out


_GR = {}


def _grain(k):
    if k not in _GR:
        a = fbm(H, W, (3.2, 1.6), seed=70 + k); b = fbm(H, W, (40, 11), seed=80 + k); _GR[k] = np.clip(.72 + .42 * a + .16 * b, 0, 1).astype(f32)
    return _GR[k]


def _smudge(k):
    if ('s', k) not in _GR:
        r = fbm(H // 2, W // 2, (150, 40, 12, 4, 2), weights=(1, .55, .34, .26, .22), seed=90 + k, aspect=4.5); M = cv2.getRotationMatrix2D((W / 4, H / 4), 24 + 7 * k, 1.35); _GR['s', k] = cv2.resize(cv2.warpAffine(r, M, (W // 2, H // 2), borderMode=cv2.BORDER_REFLECT_101), (W, H), interpolation=cv2.INTER_LINEAR)
    return _GR['s', k]


def over4(a, b): return a * (1 - b[..., 3:4]) + b


def rimlight(a, rd, wpx, col, amt=1.):
    """hot rim on a premult layer: hard core + soft inner falloff (the film's signature)"""
    al = a[..., 3]
    if al.max() <= 0: return a
    def band(q, so):
        m = cv2.warpAffine(al, np.float32([[1, 0, -rd[0] * q], [0, 1, -rd[1] * q]]), (al.shape[1], al.shape[0])); m = np.clip(al - m, 0, 1)
        return cv2.GaussianBlur(m, (0, 0), so) * (al > .02) if so > 0 else m
    m = np.clip(band(wpx, .7) * .97 + .28 * band(wpx * 2.5, wpx * .9), 0, 1)[..., None] * amt; c = rgb(col) if isinstance(col, str) else np.asarray(col, f32)
    o = a.copy(); o[..., :3] = a[..., :3] * (1 - m) + c * m * al[..., None]; return o


# ------------------------------------------------------------------ drawing
def _circ(x, y, r, n=22, sq=1.): t = np.linspace(0, TAU, n, endpoint=False); return np.c_[x + r * np.cos(t), y + r * sq * np.sin(t)]


def _taperline(rec, P, w, **k): rec.line(spl(P, n=6), w, **k)


def _eye(R, Hd, sd, E, face, lw, kid):
    ek = 1. if kid else E['es']; ex_, ey = (.375, -.15) if kid else (.338, -.045); w = .20 if kid else .192 * ek; hu = (.225 if kid else .205 * ek) * E['lu']; hl = (.17 if kid else .162 * ek) * E['ll']
    if Hd.facing(sd * ex_, ey) < -.05: return
    u = np.linspace(-1, 1, 27); x = sd * (ex_ + u * w)
    if E['lu'] < .16:
        y = ey - .02 + E['arc'] * .105 * (1 - u * u) - .02 * u
        R.line(Hd.P(np.c_[x * 1.0, y]), lw * 1.35, taper=(.18, .14), clip=face, kind='key', tmin=.3)
        for q in (.35, .75): R.line(Hd.P(np.array([(sd * (ex_ + w * q), ey + .02 + E['arc'] * .05), (sd * (ex_ + w * (q + .22)), ey + .05 + E['arc'] * .02)])), lw * .5, key='soft', clip=face)
        if abs(E['arc']) > .8:
            for q in (0, 1): _taperline(R, Hd.P(np.array([(sd * (ex_ + w * (1.05 + .12 * q)), ey - .02 - .055 * q), (sd * (ex_ + w * (1.35 + .05 * q)), ey - .01 - .085 * q)])), lw * .42, clip=face)
        return
    yu = ey + hu * (1 - np.abs(u) ** 2.3) ** .60; c = ey + E['ci'] + (E['co'] - E['ci']) * (u + 1) / 2; yu = np.minimum(yu, c); yl = ey - hl * (1 - np.abs(u) ** 2.2) ** .60; yl = np.maximum(yl, ey + E['lc']); yu = np.maximum(yu, yl + .012)
    Po = Hd.P(np.vstack([np.c_[x, yu], np.c_[x, yl][::-1]]))
    R.fill(Po, 'white', clip=face)
    sh = np.vstack([np.c_[x, yu], np.c_[x, np.maximum(yu - (.030 + .05 * min(1, E['ci'] * 0 + (E['ci'] < .4))), yl)][::-1]]); R.fill(Hd.P(sh), 'white_sh', clip=face)
    pr = E['pup'] * (1.15 if kid else 1); pc = (sd * ex_ + E['gx'], ey + E['gy'] - .005)
    R.fill(Hd.P(_circ(*pc, pr, sq=1.08)), 'pupil', clip=Po)
    if E['hl'] > .15: R.fill(Hd.P(_circ(pc[0] - pr * .32, pc[1] + pr * .40, pr * .34 * E['hl'])), 'white', clip=Po)
    if pr < .030:                                        # shock: drawn pressure lines in the white
        for a in (2.25, 2.6, 3.7, 5.6, .5): R.line(Hd.P(np.array([(pc[0] + .16 * math.cos(a), pc[1] + .16 * math.sin(a)), (pc[0] + .105 * math.cos(a + .12), pc[1] + .105 * math.sin(a + .12))])), lw * .26, key='blood', al=.8, clip=Po)
    th = (.020 + .022 * np.clip(1 - u * u, 0, 1) + .006 * (u > .4)) * (1.25 if E['ci'] < .4 else 1.)
    R.fill(Hd.P(np.vstack([np.c_[x, yu - .004], np.c_[x, yu + th][::-1]])), 'line', clip=face)
    R.line(Hd.P(np.vstack([[sd * (ex_ - w * 1.02), yu[0] - .01], np.c_[x, yu + th * .5], [sd * (ex_ + w * 1.12), yu[-1] + .004]])), lw * .7, clip=face, kind='key')
    R.line(Hd.P(np.c_[x, yl])[2:-1], lw * .52, clip=face, taper=(.3, .12), kind='key')
    if E['ci'] > .4: R.line(Hd.P(np.c_[x, yu + .062 + .01 * np.sin(u * 2)])[8:24], lw * .34, key='soft', clip=face)
    if E['strain'] > .2:
        for q in range(3): _taperline(R, Hd.P(np.array([(sd * (ex_ - w * .6 + w * .5 * q), ey - hl - .05 - .012 * q), (sd * (ex_ - w * .35 + w * .5 * q), ey - hl - .072), (sd * (ex_ + w * .02 + w * .5 * q), ey - hl - .052 + .01 * q)])), lw * .36, key='soft', clip=face)
    elif E['ci'] < .4: _taperline(R, Hd.P(np.array([(sd * (ex_ - w * .7), ey + E['lc'] - .045), (sd * ex_, ey + E['lc'] - .075), (sd * (ex_ + w * .6), ey + E['lc'] - .045)])), lw * .34, key='soft', clip=face)


def _brow(R, Hd, sd, E, clip, lw, kid):
    k = .8 if kid else 1.; xi, xm, xo = .155 + E['bx'], .36, .575
    if Hd.facing(sd * xm, .28) < -.05: return
    c = spl([(sd * xi, E['bi']), (sd * xm, E['bm']), (sd * xo, E['bo'])], n=8); t = np.linspace(0, 1, len(c)); th = .040 * k * E['bw'] * np.interp(t, [0, .12, .5, 1], [.75, 1.0, .95, .08])
    P = np.vstack([c + np.c_[0 * t, th], (c - np.c_[0 * t, th])[::-1]]); R.fill(Hd.P(P, .02), 'hair', clip=clip); R.edge(Hd.P(P, .02), lw * .42, clip=clip, kind='key')


def _mouth(R, Hd, E, face, lw, side, f=0):
    u = np.linspace(-1, 1, 41); mw, yc = E['mw'], E['yc']; sk = E['sk'] * u
    wb = E['wob'] * (np.sin(u * 7.5 + 1.2 + (f // 2) * 1.7) + .45 * np.sin(u * 13 + (f // 2) * 2.9)) * (1 - u ** 4) * .8
    yu = yc + sk + (E['uy'] - yc) * (1 - np.abs(u) ** E['pu']) + wb; yl = yc + sk + (E['ly'] - yc) * (1 - np.abs(u) ** E['pl']) + wb * .8
    x = E['mx'] + u * mw; yl = np.minimum(yl, yu); U = Hd.P(np.c_[x, yu], .012); L_ = Hd.P(np.c_[x, yl], .012)
    op = float((yu - yl).max())
    if op < .035:
        R.line(U, lw * 1.0, taper=(.16, .16), clip=face, kind='key', press=.4)
        for s in (-1, 1): _taperline(R, Hd.P(np.array([(E['mx'] + s * mw * 1.0, yc + s * E['sk'] + .022), (E['mx'] + s * mw * 1.08, yc + s * E['sk'] - .004), (E['mx'] + s * mw * 1.02, yc + s * E['sk'] - .030)])), lw * .55, clip=face)
        if E['lip'] > .1:                                    # bitten, trembling lip
            for q in (-.6, -.2, .25, .65): R.line(Hd.P(np.array([(E['mx'] + q * mw, yc + .03), (E['mx'] + q * mw * .96, yc - .045)])) + [0, 2], lw * .34, key='lip', clip=face)
            for q in range(5): R.line(Hd.P(_circ(-.10 + .05 * q, -.815 + .018 * (q % 2), .011, 8)), lw * .24, key='soft', clip=face)
        _taperline(R, Hd.P(np.array([(E['mx'] - mw * .42, yc - .115), (E['mx'], yc - .132), (E['mx'] + mw * .42, yc - .115)])), lw * .45, key='soft', clip=face)
        return
    P = np.vstack([U, L_[::-1]])
    if E['bite'] > .1:                                       # closed white grin
        R.fill(P, 'teeth', clip=face); m = (yu * .52 + yl * .48)
        hs = u * side > .25; R.fill(Hd.P(np.vstack([np.c_[x, yu], np.c_[x, yu - .045 * (1 - u ** 4)][::-1]]), .012), 'teeth_sh', clip=P)
        if hs.any(): R.fill(Hd.P(np.vstack([np.c_[x, yu][hs], np.c_[x, yl][hs][::-1]]), .012), 'teeth_sh', clip=P, al=.75)
        R.line(Hd.P(np.c_[x, m], .012)[2:-2], lw * .46, clip=P, kind='key', taper=(.1, .1))
        for q in np.linspace(-.78, .78, 7 if E['bite'] < 1.4 else 6):
            yy0, yy1, ym = np.interp(q, u, yu), np.interp(q, u, yl), np.interp(q, u, m); xx = E['mx'] + q * mw
            R.line(Hd.P(np.array([(xx, yy0), (xx * 1.0, yy0 * .35 + ym * .65)]), .012), lw * .30, key='soft', clip=P, taper=(.05, .6)); R.line(Hd.P(np.array([(xx * .98, ym), (xx * .97, ym * .5 + yy1 * .5)]), .012), lw * .24, key='soft', clip=P, taper=(.1, .7))
        if E['bite'] > 1.4:                                  # the child's missing tooth
            g = [(-.115, np.interp(-.2, u, yu) + .02), (.0, np.interp(0, u, yu) + .02), (-.004, np.interp(0, u, m) + .012), (-.108, np.interp(-.2, u, m) + .012)]; R.fill(Hd.P(np.array(g), .012), 'mouth', clip=P)
    else:
        R.fill(P, 'mouth', clip=face)
        if E['tg'] > .02:
            tgx = .72; tu = np.linspace(-1, 1, 24); ty = np.interp(tu * tgx, u, yl) + E['tg'] * (1 - np.abs(tu) ** 2.4) * (1 - .16 * np.exp(-(tu / .16) ** 2))
            T = Hd.P(np.vstack([np.c_[E['mx'] + tu * tgx * mw, ty], np.c_[E['mx'] + tu[::-1] * tgx * mw, np.interp(tu[::-1] * tgx, u, yl) - .02]]), -.04)
            R.fill(T, 'tongue', clip=P); tsh = np.vstack([np.c_[E['mx'] + tu * tgx * mw, ty], np.c_[E['mx'] + tu[::-1] * tgx * mw, ty[::-1] - E['tg'] * .35]]); R.fill(Hd.P(tsh, -.04), 'tongue_sh', clip=P)
            R.line(Hd.P(np.c_[E['mx'] + tu * tgx * mw, ty], -.04), lw * .5, clip=P); R.line(Hd.P(np.array([(E['mx'], ty[12] - .02), (E['mx'] + .01, ty[12] - E['tg'] * .7)]), -.04), lw * .34, clip=P, key='tongue_sh')
        if E['tt'] > .01:
            b = np.maximum(yu - E['tt'] * (1 - .55 * np.abs(u) ** 3), yl); Tt = Hd.P(np.vstack([np.c_[x, yu], np.c_[x, b][::-1]]), .0); R.fill(Tt, 'teeth', clip=P)
            R.fill(Hd.P(np.vstack([np.c_[x, yu], np.c_[x, yu - E['tt'] * .34][::-1]])), 'teeth_sh', clip=Tt); R.line(Hd.P(np.c_[x, b]), lw * .42, clip=P)
            if E['tt'] > .05:
                for q in (-.62, -.36, -.12, .12, .36, .62): R.line(Hd.P(np.array([(E['mx'] + q * mw, np.interp(q, u, b)), (E['mx'] + q * mw, np.interp(q, u, b) + E['tt'] * .55)])), lw * .24, key='soft', clip=P)
        if E['tb'] > .01:
            v = u * .78; yb = np.interp(v, u, yl); bt = np.minimum(yb + E['tb'] * (1 - .6 * np.abs(u) ** 3), np.interp(v, u, yu)); Tb = Hd.P(np.vstack([np.c_[E['mx'] + v * mw, yb], np.c_[E['mx'] + v * mw, bt][::-1]]), -.02); R.fill(Tb, 'teeth', clip=P); R.fill(Hd.P(np.vstack([np.c_[E['mx'] + v * mw, yb], np.c_[E['mx'] + v * mw, (yb * .55 + bt * .45)][::-1]]), -.02), 'teeth_sh', clip=P); R.line(Hd.P(np.c_[E['mx'] + v * mw, bt], -.02), lw * .42, clip=P)
            for q in (-.5, -.3, -.1, .1, .3, .5): R.line(Hd.P(np.array([(E['mx'] + q * mw, np.interp(q, v, bt)), (E['mx'] + q * mw, np.interp(q, v, bt) - E['tb'] * .6)]), -.02), lw * .24, key='soft', clip=P)
    R.edge(P, lw * .86, clip=face, kind='key', seed=3)
    for s in (-1, 1):                                         # corner creases
        _taperline(R, Hd.P(np.array([(E['mx'] + s * mw * 1.02, yc + s * E['sk'] + .060), (E['mx'] + s * mw * 1.12, yc + s * E['sk'] + .005), (E['mx'] + s * mw * 1.06, yc + s * E['sk'] - .055)])), lw * .62, clip=face)
    _taperline(R, Hd.P(np.array([(E['mx'] - mw * .36, yl.min() - .052), (E['mx'], yl.min() - .066), (E['mx'] + mw * .30, yl.min() - .054)])), lw * .50, key='soft', clip=face)


TIPS = [(-.80, -.16, .30), (-.66, .17, .36), (-.47, .37, .30), (-.29, .12, .36), (-.08, .38, .29), (.11, .20, .35), (.34, .39, .30), (.51, .15, .34), (.69, .24, .32), (.82, -.12, .28)]
SIDES = [(1.36, .34, -.02, .46), (1.40, .00, -.14, .42), (1.22, -.32, -.20, .38), (1.02, -.60, -.28, .34), (1.34, .72, -.05, .46), (.80, -.76, -.36, .32), (1.20, 1.04, -.06, .42), (1.18, .18, .10, .34)]
TEMPLE = [(.98, -.02, .22, .34), (.93, -.34, .16, .28), (1.08, .30, .14, .38)]
NAPE = [(0., -.80, -.70, .24), (.22, -.74, -.68, .24), (.44, -.70, -.60, .24), (.62, -.60, -.48, .24), (.76, -.52, -.32, .24)]
TUFTS = [(-1.06, 1.14, .0, .42), (-.62, 1.40, -.05, .50), (-.16, 1.44, -.1, .46), (.30, 1.42, -.06, .50), (.76, 1.27, -.02, .48), (1.10, .86, .0, .44), (-.36, 1.20, .25, .44), (.44, 1.18, .25, .44)]


def _lock(Hd, C3, wd, prof=(.80, 1., .96, .50, 0.), N=30, bend=0.):
    """one hair lock: 3-D centre line, belly profile -> polygon (side A then side B back)"""
    C = Hd.P3(np.asarray(C3, float)); Q = spl(C, n=9); d = np.r_[0, np.cumsum(np.hypot(*np.diff(Q, axis=0).T))]; t = np.linspace(0, 1, N); P = np.c_[np.interp(t * d[-1], d, Q[:, 0]), np.interp(t * d[-1], d, Q[:, 1])]
    T = np.gradient(P, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; n = np.c_[-T[:, 1], T[:, 0]]; w = wd * .5 * Hd.s * np.interp(t, [0, .18, .48, .80, 1], prof); P = P + n * (bend * Hd.s * np.sin(t * math.pi * 1.1) * t)[:, None]
    return np.vstack([P + n * w[:, None], (P - n * w[:, None])[::-1]])


def _fr(Hd, tx, ty, wd, worn, sway=0., dz=1., k=0):
    """centre line of a fringe lock ending at (tx, ty) on the face"""
    z = lambda x, y: float(Hd.z(np.clip(x, -.70, .70), np.clip(y, -.6, .66)))
    if worn: return [(tx * .84, .80, z(tx * .84, .6) + .10 * dz), (tx * .94, .48 + ty * .2, z(tx * .9, .46) + .13 * dz), (tx + sway, ty, z(tx, ty) + .085 * dz)]
    return [(.06 + tx * .26, 1.22, .02), (tx * .74, 1.06, z(tx * .62, .60) * .80 + .26), (tx * 1.06, .62 + ty * .2, z(tx * .94, .50) + .28 * dz), (tx + sway, ty, z(tx, ty) + .13 * dz)]


def _strand(R, S, lw, side, r, hi=True, rootdark=True, cw=1., band=(.34, .60)):
    """one hair clump: flat, root shadow, sheen sliver, strand lines, edge"""
    n = len(S) // 2; A, B = S[:n], S[::-1][:n]; m = min(len(A), len(B)); A, B = A[:m], B[:m]
    R.fill(S, 'hair')
    def lane(a, t0, t1): i0, i1 = int(t0 * (m - 1)), int(t1 * (m - 1)) + 1; return (A * (1 - a) + B * a)[i0:i1]
    if rootdark: R.fill(np.vstack([lane(0, 0, .24), lane(.5, 0, .30)[-1:], lane(1, 0, .18)[::-1]]), 'hair_sh')
    if hi:
        a0 = .14 if side > 0 else .58; t0, t1 = band[0] + .06 * r.random(), band[1] + .08 * r.random(); L0 = np.linspace(0, 1, 12); i = np.clip(((t0 + (t1 - t0) * L0) * (m - 1)).astype(int), 0, m - 1); wv = np.sin(L0 * math.pi) ** .7 * (.26 + .06 * r.random())
        p = A[i] * (1 - a0) + B[i] * a0; q = A[i] * (1 - a0 - wv)[:, None] + B[i] * (a0 + wv)[:, None]; R.fill(np.vstack([p, q[::-1]]), 'hair_hi')
        l = lane(a0 + .14, band[0] - .12, band[0] - .03); R.line(l, lw * .40, key='hair_hi', taper=(.4, .4)) if len(l) > 2 else None
    for a in (.30, .52, .74):
        l = lane(a + r.normal(0, .05), .22 + .3 * r.random(), .90)
        if len(l) > 3: R.line(l, lw * .30, key='hair_ln', taper=(.4, .5), al=.8)
    R.edge(S, lw * .55 * cw, kind='key', seed=int(r.integers(999)))


def record(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, kid=False, hat=True, body=True, shirt='red', scar=True, wind=.5, hk=None, kx=None, seed=3, lw=None, bturn=None, hair=1., kid_stitch=True, blow=0., body_dy=0., dapple=None, drift=(0., 0.)):
    """build the recorded drawing.  kx = overrides of expression keys (tear=.., gx=.. ...)"""
    E = ex(expr); E.update(kx or {}); L = LIGHT[light] if isinstance(light, str) else light; side = L['side']; cov = L['cover']
    Hd = LHead(**(KID if kid else GROWN)).pose(cx, cy, s, yaw, pitch, roll, jaw=E['jaw'], st=E['st'])
    lw = lw or float(np.clip(s / 420 * 6.0, 1.3, 8.5)); R = Rec(); r = np.random.default_rng(seed); g = f // 2 * 2
    sw = lambda k, a=.03: wind * a * (math.sin(g * .24 + k * 1.7) + .5 * math.sin(g * .41 + k * 2.9))
    face = Hd.face(.70)
    # ---------------------------------------------------------------- back hair
    R.part('backhair', rim=.9)
    sk = Hd.skull(); hc = Hd.P3(np.array([(0, .30, -.05)]))[0]; bare = not hat or hat == 'back'
    dv = sk - hc; R.fill(sk if not bare else hc + dv * np.interp(dv @ np.array([math.sin(roll), -math.cos(roll)]) / s, [-.55, -.1, .35], [1.0, 1.10, 1.20])[:, None], 'hair_sh')
    ls = []
    for q, (tx, ty, tz, wd) in enumerate(SIDES):
        if not bare: tx, wd = .86 + (tx - .86) * .55, wd * .78
        for sd in (-1, 1): ls.append(((sd * .55, 1.02, -.10) if bare else (sd * .62, .36, -.20), (sd * ((.98 if bare else .80) if ty > -.2 else (.88 if bare else .74)), ((.62 + ty) / 2 + .10) if bare else min(.30, (.45 + ty) / 2), tz * .6 - .02), (sd * (tx + (.02 if kid else 0)) + sw(q * 2 + sd, .05) + drift[0], (ty if bare else min(ty, .16 - .05 * q)) + sw(q, .02) + drift[1], tz), wd, q))
    for q, (tx, ty, tz, wd) in enumerate(NAPE):
        for sd in (-1, 1) if tx else (1,): ls.append(((sd * tx * 1.0, -.05, tz * 1.28), (sd * tx * 1.08, -.40, tz * 1.22), (sd * tx * 1.18 + sw(q * 3 + sd, .05) + drift[0] * .9, ty + .05 * (q % 2) + drift[1] + .25 * abs(drift[0]), tz * 1.28), wd * 1.25, q + 50))
    if bare:
        for q, (tx, ty, tz, wd) in enumerate(TUFTS): ls.append(((tx * .35, 1.00, -.05), (tx * .72, ty * .90 + .08, tz), (tx + sw(q + 40, .05) + drift[0], ty + drift[1], tz - .03), wd, q + 20))
    for a, b, c, wd, q in sorted(ls, key=lambda v: float(Hd.depth(np.array([v[2]]))[0])): _strand(R, _lock(Hd, [a, b, c], wd * r.uniform(.9, 1.15), bend=.03 * (1 if q % 2 else -1)), lw, side, r)
    # ---------------------------------------------------------------- neck + body
    np_ = Hd.P3(np.array([(0, (-1.10 if kid else -1.20) + E['jaw'] * .8, -.16)]))[0] + [0, body_dy * s]; bt = bturn or 0.
    def B(P):
        P = np.atleast_2d(np.asarray(P, float)); z = .5 * (1 - np.clip(P[:, 0] / 1.9, -1, 1) ** 2); x = P[:, 0] * math.cos(bt + yaw * .35) + z * math.sin(bt + yaw * .35); c_, s_ = math.cos(roll * .25), math.sin(roll * .25)
        return np_ + s * np.c_[x * c_ - P[:, 1] * s_, x * s_ + P[:, 1] * c_]
    if body: _body(R, Hd, B, E, L, s, lw, side, cov, kid, shirt, scar, r, g, wind, hat)
    else:
        R.part('neck', rim=.8)
        nk = np.vstack([B([(-.36, -.9)]), B([(-.36, .25)]), B([(.36, .25)]), B([(.36, -.9)])]); R.fill(nk, 'skin_sh')
    # ---------------------------------------------------------------- ears
    R.part('ear', rim=.9)
    for sd in (-1, 1):
        e = Hd.ear(sd)
        if e['vis'] > -.05:
            shd = sd == side or cov > .5 and sd != -side
            R.fill(e['outer'], 'skin_sh' if shd else 'skin'); R.fill(np.vstack([e['inner'], e['tragus'][::-1]]), 'skin_dk' if shd else 'skin_sh'); R.edge(e['outer'], lw * .8, seed=4); R.line(e['inner'], lw * .5); R.line(e['tragus'], lw * .42)
    # ---------------------------------------------------------------- face
    R.part('face', rim=1.)
    R.fill(face, 'skin')
    if E['blush'] > .05:
        for sd in (-1, 1):
            R.fill(Hd.P(_circ(sd * (.50 if kid else .47), -.36, .15, sq=.55), .01), 'blush', clip=face, al=.42 * E['blush'])
            for q in range(4): R.line(Hd.P(np.array([(sd * (.40 + .055 * q) + .02, -.315), (sd * (.40 + .055 * q) - .03, -.385)]), .01), lw * .28, key='soft', clip=face, al=.8)
    ys = np.array([.80, .62, .40, .22, .10, -.04, -.14, -.24, -.36, -.48, -.60, -.72, -.84, -.95, -1.02]); sh = np.array([.02, .04, .05, .0, -.07, -.02, .10, .12, .06, .02, .04, -.02, -.12, -.24, -.36])
    xb = side * (.45 - 1.02 * cov + sh); far = side * 2.2
    pb = Hd.P(np.c_[np.clip(xb, -.72, .72), np.clip(ys, -1, .8)]); fx = Hd.P3(np.c_[np.full(2, far), [-1.2, 1.2], [0., 0.]]); pb = spl(pb, n=5)
    shp = np.vstack([pb, fx[:1] + [0, s * .6], fx[1:] - [0, s * .6]]); R.fill(shp, 'skin_sh', clip=face)
    # hatching on the shadow edge (pen clusters)
    for (yy, n_, ln_) in (((-.30, 6, .13), (-.62, 5, .10)) if E['hat'] > .5 else ()) if not kid else ((-.34, 4, .09),):
        xe = float(np.interp(-yy, -ys, np.clip(xb, -.7, .7)))
        for q in range(n_): a = Hd.P(np.array([(np.clip(xe - side * .015 * (q % 2), -.72, .72), yy - q * .028), (np.clip(xe - side * (ln_ + .02 * math.sin(q * 2.1)), -.72, .72), yy + .030 - q * .028)]), .004); R.line(a, lw * .25, key='soft' if cov < .5 else 'line', taper=(.05, .7), clip=face, al=.85, seed=q)
    # under brow / socket, lip, nose shadows
    if E['sh'] > .05 or cov > .4:
        for sd in (-1, 1): R.fill(Hd.P(spl([(sd * .10, .20), (sd * .34, .25), (sd * .58, .18), (sd * .56, .04), (sd * .34, .09), (sd * .13, .02)], True), .003), 'skin_dk', clip=face, al=max(E['sh'], .22 * cov))
    nz = Hd.nose_profile(); nd = side
    R.fill(Hd.P3(np.array([(0, -.20, nz[2][2] - .04), (nd * .005, -.31, nz[3][2]), (nd * .085, -.385, nz[4][2] - .10), (nd * .12, -.43, nz[6][2] - .06), (nd * .04, -.455, nz[6][2] - .02), (-nd * .04, -.425, nz[6][2] + .01), (0, -.37, nz[4][2])])), 'skin_sh' if cov < .5 else 'skin_dk', clip=face)
    R.fill(Hd.P(spl([(-.10, min(E['ly'], E['yc']) - .06 - E['jaw'] * .0), (0, min(E['ly'], E['yc']) - .085), (.12, min(E['ly'], E['yc']) - .06), (.02, min(E['ly'], E['yc']) - .12)], True)), 'skin_sh', clip=face, al=.9)
    if cov > .4:                                             # bounce: lit nose plane + lit lip + lit cheek facet on the fire side
        ls = -side
        R.fill(Hd.P3(np.array([(ls * .01, -.02, nz[1][2] - .02), (ls * .045, -.06, nz[1][2] - .04), (ls * .075, -.30, nz[4][2] - .10), (ls * .03, -.36, nz[4][2] - .02), (0, -.33, nz[3][2]), (0, -.14, nz[2][2])])), 'skin', clip=face)
        R.fill(Hd.P3(np.array([(ls * .022, -.06, nz[1][2] - .02), (ls * .04, -.09, nz[1][2] - .04), (ls * .05, -.28, nz[4][2] - .07), (ls * .02, -.32, nz[4][2] - .01)])), 'skin_hi', clip=face)
    # hair cast shadow
    cast = []
    for q, (tx, ty, wd) in enumerate(TIPS): cast.append(_lock(Hd, _fr(Hd, tx * (1 + .10 * blow) + .03 * side + sw(q), ty - .075 + blow * .20, wd, not bare, dz=.0), wd * 1.12))
    if hat is True or hat == 1: R.fill(Hd.P(spl([(-.74, .9), (.74, .9), (.74, .18), (.45, .10 if not kid else .0), (0, .06 if not kid else -.04), (-.45, .10 if not kid else .0), (-.74, .18)], True, corners=(0, 1, 2, 6))), 'skin_sh', clip=face)
    for c in cast: R.fill(c, 'skin_dk' if (hat is True or cov > .5) else 'skin_sh', clip=face, al=.9)
    if dapple is not None:
        for (x, y, a, b) in dapple: R.fill(Hd.P(spl([(x - a, y), (x - a * .3, y + b), (x + a * .8, y + b * .7), (x + a, y - b * .1), (x + a * .2, y - b), (x - a * .7, y - b * .7)], True, n=5), .004), 'skin_hi', clip=face)
    # features
    for sd in (-1, 1): _eye(R, Hd, sd, E, face, lw, kid)
    # scar under the left eye (viewer's right)
    sx, sy = (.40, -.395) if kid else (.365, -.268 - .46 * max(0, E['es'] - 1.07))
    _taperline(R, Hd.P(np.array([(sx - .13, sy + .012), (sx, sy - .008), (sx + .135, sy + .018)]), .004), lw * .62, clip=face, kind='key')
    for q in ((-.05, .055) if not kid else (-.07, 0, .07)):
        _taperline(R, Hd.P(np.array([(sx + q - .008, sy + .052), (sx + q + .006, sy - .048)]), .004), lw * .55, clip=face, kind='key')
        if kid and kid_stitch: R.fill(Hd.P(_circ(sx + q, sy + .05, .011, 8), .004), 'line', clip=face)
    if kid: _taperline(R, Hd.P(np.array([(sx - .13, sy - .02), (sx, sy - .04), (sx + .13, sy - .012)]), .004), lw * .8, key='blush', clip=face)
    # nose line
    prof = Hd.P3(nz[3 if abs(yaw) > .18 else 4:7])
    R.line(spl(prof, n=6), lw * (.62 if abs(yaw) > .18 else .5), taper=(.5, .2), kind='key')
    _taperline(R, Hd.P3(np.array([(-.065, -.405, nz[6][2] - .02), (-.03, -.435, nz[6][2] + .03), (.03, -.435, nz[6][2] + .03), (.065, -.405, nz[6][2] - .02)])), lw * .55, clip=face, kind='key')
    # wrinkles
    if E['fur'] > .1:
        for sd in (-1, 1): _taperline(R, Hd.P(np.array([(sd * .075, .25), (sd * .055, .16), (sd * .09, .075)])), lw * .62 * E['fur'], clip=face)
        _taperline(R, Hd.P(np.array([(-.02, .19), (.005, .13), (-.012, .07)])), lw * .4 * E['fur'], clip=face)
    if E['nos'] > .1:
        for q in range(3): _taperline(R, Hd.P(np.array([(-.085 + .015 * q, .015 - .052 * q), (-.02, -.012 - .052 * q), (.02, .002 - .052 * q), (.09 - .015 * q, .02 - .052 * q)])), lw * .42 * E['nos'], clip=face)
    if E['fore'] > .1:
        for q in range(3): _taperline(R, Hd.P(np.array([(-.26 + .03 * q, .43 + .06 * q), (-.08, .50 + .06 * q), (.02, .475 + .06 * q), (.12, .505 + .06 * q), (.27 - .03 * q, .44 + .06 * q)])), lw * .46 * E['fore'], clip=face)
    if E['naso'] > .1:
        for sd in (-1, 1): _taperline(R, Hd.P(np.array([(sd * (.10 + E['mw'] * .42), -.415), (sd * (.10 + E['mw'] * .66), -.44), (sd * (.11 + E['mw'] * .86), max(E['yc'] + .02, -.58))])), lw * .55 * E['naso'], clip=face)
    _mouth(R, Hd, E, face, lw, side, f)
    if E['blood'] > .05: _battered(R, Hd, face, lw, r, side)
    for sd in (-1, 1): _brow(R, Hd, sd, E, face, lw, kid)
    R.edge(face, lw, seed=1, breaks=2)
    _taperline(R, Hd.P(np.array([(-.10, -.935), (0, -.955), (.09, -.938)])), lw * .34, key='soft', clip=face)
    if E['tear'] > .05: _tears(R, Hd, E, face, lw, kid, f)
    if E['sweat'] > .1:
        for (x, y, q) in ((-.60, .10, 1.), (.56, -.28, .8)): d = Hd.P(spl([(x, y + .09), (x + .028, y + .02), (x + .022, y - .03), (x, y - .045), (x - .022, y - .03), (x - .028, y + .02)], True, corners=(0,)), .02); R.fill(d, 'tear'); R.fill(d + [1, -2], 'tear_sh', al=.4, clip=d); R.edge(d, lw * .42)
    # ---------------------------------------------------------------- fringe + sideburns
    R.part('fringe', rim=.95)
    if bare: cap = Hd._sil([Hd._mesh(.62, 1, 1.12)]); R.fill(cap, 'hair'); R.fill(hc + (cap - hc) * 1.22, 'hair')
    if bare:
        for q, (tx, ty, tz, wd) in enumerate(TEMPLE):
            for sd in (-1, 1): _strand(R, _lock(Hd, [(sd * .42, 1.12, .12), (sd * .88, .72, tz + .10), (sd * 1.0, .34 + ty * .4, tz + .04), (sd * tx * (1 + .08 * blow) + sw(q * 5 + sd, .06) + drift[0] * .8, ty + sw(q + sd, .02) + .16 * blow * (q != 1) + drift[1], tz)], wd * r.uniform(.9, 1.1), prof=(.5, 1., 1., .55, 0.), bend=.05 * (1 if q % 2 else -1)), lw, side, r, rootdark=False, band=(.40, .62))
    od = [0, 9, 2, 4, 6, 8, 1, 7, 3, 5] if not bare else [0, 9, 2, 8, 4, 6, 1, 7, 3, 5]
    for q in od: tx, ty, wd = TIPS[q]; _strand(R, _lock(Hd, _fr(Hd, tx * (1 + .10 * blow) + drift[0] * .55, ty + drift[1] + sw(q + 3, .012) + blow * (.20 + .06 * (q % 2)), wd * (1 - .12 * blow), not bare, sway=sw(q), dz=1 + 1.2 * blow), wd * (.86 if not bare else r.uniform(.95, 1.25)), prof=(.85, 1., .86, .48, 0.) if not bare else (.55, 1., 1.0, .56, 0.), bend=.035 * (1 if q % 2 else -1)), lw, side, r, rootdark=not bare, band=(.30, .56) if not bare else (.36, .60))
    return R, Hd, E, L


def _tears(R, Hd, E, face, lw, kid, f):
    t = min(1., E['tear']); ey = -.27 if kid else -.045 - .145 * E['es']; g = f // 2; k = 1.15 if kid else 1.
    def rope(path, hw, ph, bead):
        P = spl(np.array(path, float), n=8); d = np.r_[0, np.cumsum(np.hypot(*np.diff(P, axis=0).T))]; n = 26; a = np.linspace(0, 1, n) ; q = np.c_[np.interp(a * d[-1] * t, d, P[:, 0]), np.interp(a * d[-1] * t, d, P[:, 1])]
        T = np.gradient(q, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; nn = np.c_[-T[:, 1], T[:, 0]]; w = hw * (.86 + .20 * np.sin(a * 8.5 - g * 1.15 + ph)) * np.interp(a, [0, .1, .55, .80, .91, 1], [.55, 1, .86, .70, .62 * bead, .04])
        A, B = q + nn * w[:, None], q - nn * w[:, None]; Q = Hd.P(np.vstack([A, B[::-1]]), .025); R.fill(Q, 'tear', al=.62); R.fill(Hd.P(np.vstack([q + nn * (w * .40)[:, None], A[::-1]]), .025), 'tear_sh', al=.75, clip=Q); R.edge(Q, lw * (.44 if kid else .50), kind='key', key='soft' if kid else 'line')
        R.line(Hd.P((q - nn * (w * .42)[:, None])[3:15], .03), lw * .85, key='white', kind='key'); R.line(Hd.P((q - nn * (w * .40)[:, None])[15:18], .03), lw * .5, key='white')
    for sd in (-1, 1):
        if Hd.facing(sd * .4, -.4) < -.1: continue
        x0 = .40 if kid else .37
        if False: rope([(sd * .22, ey), (sd * .235, ey - .16), (sd * (.20 + E['mw'] * .30), -.50), (sd * (E['mw'] + .06), -.62)], .036 * k, 1 + sd, 1.25)
        rope([(sd * x0, ey - .02), (sd * (x0 + .07), ey - .20), (sd * (x0 + .15), -.54), (sd * (x0 + .14), -.74), (sd * (x0 + .08), -.94)], (.058 if kid else .046) * k, 2.2 + sd, 1.2)
        ll = Hd.P(spl([(sd * (x0 - .17), ey + .015), (sd * x0, ey - .055), (sd * (x0 + .19), ey + .025), (sd * x0, ey + .02)], True, corners=(0, 2)), .02); R.fill(ll, 'tear', al=.75); R.edge(ll, lw * .40, kind='key', key='soft'); R.line(Hd.P(np.array([(sd * (x0 - .08), ey - .012), (sd * (x0 + .03), ey - .02)]), .03), lw * .5, key='white')
    if E['snot'] > .1:                                        # a rope that leans, thins, swings behind the breath and ends in a drop
        g2 = f / 2.; lean = .050 * math.sin(g2 * .78 + .6) + .018 * math.sin(g2 * 1.7); ln = .20 * E['snot'] * (1 + .16 * math.sin(g2 * .78 + 2.1)); sd = 1; a = np.linspace(0, 1, 26)
        x = sd * .050 + .030 * np.sin(a * 2.6) * (1 - .4 * a) + lean * a ** 1.8; y = -.425 - ln * a ** 1.05; hw = np.interp(a, [0, .10, .45, .72, .88, .96, 1], [.030, .024, .0125, .0115, .0225, .0190, .002]); c = np.c_[x, y]; T = np.gradient(c, axis=0); T /= np.hypot(T[:, 0], T[:, 1])[:, None] + 1e-9; nn = np.c_[-T[:, 1], T[:, 0]]; A, B = c + nn * hw[:, None], c - nn * hw[:, None]
        Q = Hd.P(spl(np.vstack([A, B[::-1]]), True, n=3), .05); R.fill(Q, 'tear', al=.72); R.fill(Hd.P(np.vstack([c + nn * (hw * .35)[:, None], A[::-1]]), .05), 'tear_sh', al=.8, clip=Q); R.edge(Q, lw * .36, key='soft', kind='key'); R.line(Hd.P((c - nn * (hw * .35)[:, None])[2:11], .055), lw * .34, key='white'); R.line(Hd.P((c - nn * (hw * .40)[:, None])[20:23], .055), lw * .62, key='white')
        R.fill(Hd.P(spl([(-.075, -.415), (-.045, -.40), (-.02, -.425), (-.045, -.455)], True, n=4), .04), 'tear', al=.6)


def _battered(R, Hd, face, lw, r, side):
    """dried smears and scratches: pen clusters and brush ink, sealed"""
    for (x, y, w_, h_, a) in ((-.42, -.46, .20, .10, .5), (.50, -.44, .13, .09, -.4), (.05, .36, .30, .05, .1), (-.58, .02, .08, .13, 1.1), (.22, -.80, .13, .07, .3), (.62, .05, .06, .10, 1.2)):
        n = 22; t = np.linspace(0, TAU, n, endpoint=False); rad = 1 + .42 * r.normal(0, 1, n) * .55; c, s = math.cos(a), math.sin(a); px, py = w_ * .5 * np.cos(t) * rad, h_ * .5 * np.sin(t) * rad
        P = Hd.P(np.clip(np.c_[x + px * c - py * s, y + px * s + py * c], -.97, .72), .004); R.fill(P, 'skin_dk', clip=face, al=.62)
        for q in range(7): k = (q - 3) / 3.; u = r.uniform(.5, .9); a0 = np.array([x + k * w_ * .5 * c, y + k * w_ * .5 * s]); d = np.array([-.5 * h_ * u, .75 * h_ * u]); R.line(Hd.P(np.array([a0 - d, a0 + d * .3, a0 + d]), .006), lw * r.uniform(.28, .5), clip=face, seed=q, al=.9)
    for L_ in ([(-.30, -.36), (-.40, -.46), (-.53, -.50)], [(.44, -.34), (.53, -.42), (.58, -.52)], [(-.10, .40), (-.03, .33), (.0, .24)]): _taperline(R, Hd.P(np.array(L_), .006), lw * .8, key='blood', clip=face)
    P = Hd.P(spl([(-.17, -.655), (-.145, -.67), (-.15, -.76), (-.165, -.83), (-.18, -.76)], True), .01); R.fill(P, 'blood', clip=face)
    P = Hd.P(spl([(.10, .56), (.15, .55), (.135, .40), (.16, .28), (.145, .20), (.115, .30), (.10, .42)], True), .006); R.fill(P, 'blood', clip=face)


def _body(R, Hd, B, E, L, s, lw, side, cov, kid, shirt, scar, r, g, wind, hat):
    lf = lambda k, a=.02: wind * a * math.sin(g * .22 + k)
    sil = [(-.33, -.62), (-.305, -.16), (-.37, .04), (-.66, .19), (-1.00, .27), (-1.22, .39), (-1.37, .68), (-1.42, 1.1), (-1.42, 2.4), (1.42, 2.4), (1.42, 1.1), (1.37, .68), (1.22, .39), (1.00, .27), (.66, .19), (.37, .04), (.305, -.16), (.33, -.62)]
    if kid: sil = [(-.30, -.62), (-.30, -.12), (-.36, .02), (-.66, .14), (-.92, .30), (-1.12, .60), (-1.24, 1.1), (-1.3, 2.4), (1.3, 2.4), (1.24, 1.1), (1.12, .60), (.92, .30), (.66, .14), (.36, .02), (.30, -.12), (.30, -.62)]
    n = len(sil); cor = (0, n // 2 - 1, n // 2, n - 1); S = B(spl(sil, True, corners=cor, n=7))
    R.part('neck', rim=.95); R.fill(S, 'skin')
    # shade: chin shadow on the neck, side plane
    u = Hd.P3(np.array([(-.60, -.62, .10), (-.42, -.86, .35), (0, -1.0, .6), (.42, -.86, .35), (.60, -.62, .10)]))
    chs = np.vstack([u[:1], B([(-.46, -.02), (-.30, .10), (-.06, .22), (.16, .10), (.46, -.04)]), u[-1:], u[1:4][::-1]]); R.fill(chs, 'skin_sh', clip=S); R.fill(np.vstack([u[:1], B([(-.42, -.20), (-.1, -.02), (.2, -.10), (.42, -.22)]), u[-1:]]), 'skin_dk', clip=S, al=.75)
    sx = side; x0 = .62 - 1.25 * cov
    R.fill(B(spl([(sx * (x0 + .20), -.3), (sx * (x0 - .05), .25), (sx * (x0 + .06), .55), (sx * (x0 - .12), .95), (sx * (x0 + .05), 1.3), (sx * (x0 - .05), 2.4), (sx * 3, 2.4), (sx * 3, -.3)], True, corners=(0, 5, 6, 7), n=6)), 'skin_sh', clip=S)
    if not kid:
        for sd in (-1, 1):                                    # clavicles, sterno-mastoid, pecs
            R.line(B(spl([(sd * .10, .20), (sd * .35, .13), (sd * .72, .20), (sd * 1.05, .31)], n=6)), lw * .66, taper=(.12, .5)); R.line(B(spl([(sd * .14, .27), (sd * .40, .22), (sd * .70, .28)], n=6)), lw * .34, key='soft')
            R.line(B(spl([(sd * .33, -.42), (sd * .24, -.08), (sd * .09, .20)], n=6)), lw * .6, taper=(.4, .2))
            R.line(B(spl([(sd * .05, 1.10), (sd * .38, 1.22), (sd * .72, 1.16), (sd * 1.02, .92)], n=6)), lw * .66, taper=(.2, .45))
            R.fill(B(spl([(sd * .05, 1.10), (sd * .38, 1.22), (sd * .72, 1.16), (sd * 1.0, .94), (sd * .72, 1.25), (sd * .36, 1.36)], True, corners=(0, 3))), 'skin_sh', clip=S)
            R.fill(B(_circ(sd * .62, 1.02, .05, sq=.7)), 'skin_dk', clip=S, al=.7)
            for q in range(6): R.line(B([(sd * (.30 + .06 * q), 1.17 + .01 * q), (sd * (.26 + .06 * q), 1.08 + .012 * q)]), lw * .24, key='soft', taper=(.05, .7))
        R.line(B(spl([(0, .22), (.015, .6), (-.01, 1.0), (0, 1.2)], n=6)), lw * .5, taper=(.3, .5)); R.line(B(spl([(-.06, .16), (0, .22), (.06, .16)], n=6)), lw * .6)
        if scar:
            for k_, (a, b) in enumerate((((-.62, .52), (.58, 1.62)), ((.60, .50), (-.60, 1.64)))):
                a, b = np.array(a), np.array(b); t = np.linspace(0, 1, 26); d = (b - a) / np.hypot(*(b - a)); nr = np.array([-d[1], d[0]]); c = a + (b - a) * t[:, None]; w0 = (.15 + .05 * r.normal(0, 1, 26)) * np.clip(np.minimum(t, 1 - t) * 6, .15, 1); w1 = (.15 + .05 * r.normal(0, 1, 26)) * np.clip(np.minimum(t, 1 - t) * 6, .15, 1)
                P = B(np.vstack([c + nr * w0[:, None], (c - nr * w1[:, None])[::-1]])); R.fill(P, 'scar', clip=S); R.fill(B(np.vstack([c + nr * w0[:, None], (c + nr * w0[:, None] * .35)[::-1]])), 'scar_sh', clip=S, al=.7); R.edge(P, lw * .34, key='soft', clip=S)
                for q in np.linspace(.06, .94, 17): p = a + (b - a) * q; h_ = .10 * r.uniform(.5, 1.1); R.line(B([p - nr * h_, p + nr * h_ * .3 + d * .03]), lw * .24, key='soft', clip=S, al=.9)
    R.edge(S, lw, seed=2)
    R.part('body', rim=1.)
    if kid:
        T = [(-.34, .0), (-.64, .12), (-.92, .28), (-1.13, .58), (-1.25, 1.1), (-1.32, 2.4), (1.32, 2.4), (1.25, 1.1), (1.13, .58), (.92, .28), (.64, .12), (.34, .0), (.22, .16), (0, .22), (-.22, .16)]
        P = B(spl(T, True, corners=(0, 5, 6, 11), n=6)); R.fill(P, 'tee'); R.fill(B(spl([(side * .55, .1), (side * .42, .7), (side * .60, 1.2), (side * .46, 2.4), (side * 2, 2.4), (side * 2, .1)], True, corners=(3, 4, 5))), 'tee_sh', clip=P)
        R.fill(B(spl([(-.36, .0), (0, .40), (.36, .0), (.22, .16), (0, .22), (-.22, .16)], True, corners=(0, 2))), 'anchor', clip=P)
        for y in (.62, .74): R.line(B(spl([(-1.3, y + .06), (0, y + .10), (1.3, y + .06)], n=8)), lw * 2.0, key='anchor', clip=P, taper=(0, 0)) if y < .6 else None
        ac = lambda q: B(np.array(q) * .40 + [0, 1.02])           # anchor
        R.fill(ac(_circ(0, -.78, .20)), 'anchor', clip=P); R.fill(ac(_circ(0, -.78, .09)), 'tee', clip=P)
        R.fill(ac([(-.08, -.62), (.08, -.62), (.08, .78), (-.08, .78)]), 'anchor', clip=P); R.fill(ac([(-.42, -.42), (.42, -.42), (.42, -.26), (-.42, -.26)]), 'anchor', clip=P)
        t = np.linspace(.15, math.pi - .15, 20); R.fill(ac(np.vstack([np.c_[.80 * np.cos(t), .20 + .72 * np.sin(t)], np.c_[.60 * np.cos(t), .14 + .54 * np.sin(t)][::-1]])), 'anchor', clip=P)
        for sd in (-1, 1): R.fill(ac([(sd * .98, .06), (sd * .86, .52), (sd * .52, .30)]), 'anchor', clip=P)
        for q, (a, b, c) in enumerate([((-.9, .35), (-.7, .6), (-.62, 1.0)), ((.88, .36), (.72, .64), (.70, 1.1)), ((-.25, 1.5), (-.1, 1.7), (-.2, 2.1)), ((.4, 1.4), (.35, 1.75), (.5, 2.2))]): R.line(B(spl([a, b, c], n=6)), lw * .45, clip=P)
        R.edge(P, lw, seed=6); R.line(B(spl([(-.36, .0), (0, .40), (.36, .0)], n=8)), lw * .5); return
    for sd in (-1, 1):
        o = lf(sd * 2 + 1)
        if shirt == 'vest': T = [(.50, .06), (.78, .18), (1.06, .30), (1.13, .48), (1.10, .80), (1.20, 1.25), (1.34, 2.4), (.92 + o, 2.4), (.86 + o, 1.9), (.80 + o, 1.2), (.66, .62), (.46, .26)]
        else: T = [(.42, -.02), (.68, .11), (1.04, .20), (1.30, .34), (1.47, .66), (1.53, 1.1), (1.60, 2.4), (.95 + o, 2.4), (.90 + o, 1.9), (.82 + o, 1.2), (.68, .62), (.44, .26)]
        P = B(spl([(sd * x, y) for x, y in T], True, corners=(0, 6, 7, 11), n=6)); R.fill(P, 'shirt')
        shs = sd == side or cov > .5
        if shs: R.fill(B(spl([(sd * x, y) for x, y in [(.62, .05), (.80, .40), (.92, .85), (.96, 1.3), (1.08, 2.4), (.5, 2.4), (.5, .05)]], True, corners=(4, 5, 6)) if sd != side else np.array([(sd * .3, -.2), (sd * 2.4, -.2), (sd * 2.4, 2.6), (sd * .3, 2.6)])), 'shirt_sh', clip=P)
        else: R.fill(B(spl([(sd * x, y) for x, y in [(.60, .34), (.82, .9), (.95, 1.5), (1.0, 2.4), (.75, 2.4), (.78, 1.3), (.60, .6)]], True)), 'shirt_sh', clip=P)
        if shirt != 'vest':                                  # the sleeve is its own plane
            ar = spl([(sd * 1.16, .86), (sd * 1.10, 1.25), (sd * 1.14, 1.8), (sd * 1.16, 2.4)], n=7); R.fill(B(np.vstack([ar, (ar + np.array([sd * .10, 0]) * np.linspace(.2, 1.6, len(ar))[:, None])[::-1]])), 'shirt_dk' if shs else 'shirt_sh', clip=P); R.line(B(ar), lw * .8, taper=(.35, 0), clip=P)
        # folds run from the shoulder point and the armpit
        for q, pts in enumerate([[(1.02, .34), (.92, .62), (.96, 1.0)], [(1.12, .95), (1.0, 1.25), (1.06, 1.7)], [(.80, .24), (.74, .5), (.80, .85)], [(1.34, .60), (1.28, .95), (1.33, 1.4)], [(1.20, .40), (1.26, .62), (1.22, .86)], [(.90, 1.3), (.98, 1.65), (.92, 2.1)]]):
            p = np.array([(sd * x, y) for x, y in pts]); c = spl(p, n=7); R.line(B(c), lw * .55, taper=(.1, .6), clip=P); w_ = np.sin(np.linspace(0, math.pi, len(c)))[:, None] * np.array([.07 * sd, 0]); R.fill(B(np.vstack([c, (c + w_)[::-1]])), 'shirt_dk' if shs else 'shirt_sh', clip=P)
            if not (shs and sd == side): R.fill(B(np.vstack([c - w_ * .25, (c - w_ * .8)[::-1]])), 'shirt_hi', clip=P, al=.8)
        R.edge(P, lw * .95, seed=8 + sd)
        cl = B(spl([(sd * x, y) for x, y in [(.40, -.10), (.56, .02), (.80, .20), (.86, .40), (.66, .50), (.50, .34), (.42, .12)]], True, corners=(0, 3, 4))); R.fill(cl, 'shirt_sh' if shs else 'shirt'); R.fill(cl + [0, s * .05], 'shirt_dk' if shs else 'shirt_sh', clip=cl, al=.6); R.edge(cl, lw * .8, seed=10)
        for q in range(3): c = B([(sd * (.80 + .02 * q), .95 + .5 * q)]); R.fill(c + _circ(0, 0, s * .042), 'sash'); R.edge(c + _circ(0, 0, s * .042), lw * .4)
    if hat == 'back':                                         # the cord of the hat
        for sd in (-1, 1):
            c = spl([(sd * .46, -.02), (sd * .30, .18), (sd * .07, .44)], n=10); R.line(B(c), lw * 2.0, key='line', taper=(0, 0), press=.05); R.line(B(c), lw * 1.25, key='cord', taper=(0, 0), press=.05)
            for q in range(2, len(c) - 1, 2): R.line(B([c[q] + [-.02, -.012], c[q] + [.02, .012]]), lw * .34, key='cord_sh')
        for q, e in enumerate(((-.10, .86), (.05, .92), (.13, .80))): c = spl([(0, .46), (e[0] * .5, .62), (e[0] + lf(q, .03), e[1])], n=8); R.line(B(c), lw * 1.5, taper=(0, .1)); R.line(B(c), lw * .9, key='cord', taper=(0, .1))
        R.fill(B(_circ(0, .45, .055)), 'cord'); R.edge(B(_circ(0, .45, .055)), lw * .5)


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, kid=False, hat=True, hk=None, hat_light=None, ret=False, **kw):
    R, Hd, E, L = record(cx, cy, s, yaw, pitch, roll, expr, light, f, kid=kid, hat=hat, **kw)
    hl = hat_light or L['hat']; K = dict(w=3.0, tilt=.04, lean=0., up=0.); KB = {}
    if kid: K.update(w=3.55, tilt=-.04, up=-.06)
    for k, v in (hk or {}).items(): (K if k in K or k == 'fwd' else KB)[k] = v
    out = np.zeros((H, W, 4), f32)
    if hat is True:
        st = HT.seat(Hd, **K); kb = dict(dict(wear=.25 if kid else .6, light=hl, f=f), **KB); out = HT.hat(**st, part='back', **kb).copy(); out = R.cel(L, s, out); out = over4(out, HT.hat(**st, part='front', **kb))
    elif hat == 'back':
        h = Hd.P3(np.array([(0, -1.05, -.9)]))[0]; kb = dict(dict(wear=.6, light=hl, f=f), **KB); out = HT.hat(h[0] + s * .42, h[1] - s * .02, s * 3.25, yaw * .4 + .35, 1.12, roll - .22, **kb).copy(); out = R.cel(L, s, out)
    else: out = R.cel(L, s, out)
    return (out, R, Hd, L) if ret else out


def kid_head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='kidgrin', light='day', f=0, **kw): return head(cx, cy, s, yaw, pitch, roll, expr, light, f, kid=True, **kw)


_HN = {}


def hand_patch(pose='pinch', light='gold', side='R', euler=(0, 20, 10), s=220, hl=(-.6, .5, .3), w=4.5, cuff=None, over=None):
    """an op/hands hand in Luffy's colours rendered once -> (premult patch 900x900, wrist px).  Move it with put_patch"""
    k = (pose, light, side, euler, s, hl, w)
    if k not in _HN:
        from op import hands as HD
        c = LIGHT[light]['c']; hx = lambda q: '#%02x%02x%02x' % tuple(int(np.clip(v, 0, 1) * 255) for v in c[q]); cel = Cel(900, 900)
        g = HD.draw(cel, pose, HD.Cam(450, 450, s, 5.0), pal=dict(skin=hx('skin'), shade=hx('skin_sh'), line=hx('line'), nail=hx('skin_hi'), nail_shade=hx('skin')), light=hl, w=w, side=side, R=HD.euler(*euler), clip=(0, 0, 900, 900), res=480)
        a = rimlight(cel.rgba(), np.array(LIGHT[light]['rd'], float) / np.hypot(*LIGHT[light]['rd']), s * .030, LIGHT[light]['rim'], 1.); _HN[k] = a
    return _HN[k]


def put_patch(a, x, y, rot=0., z=1., anchor=(450, 450)):
    c, s = math.cos(rot) * z, math.sin(rot) * z; M = np.float32([[c, -s, 0], [s, c, 0]]); M[:, 2] = np.array([x, y]) - M[:, :2] @ np.array(anchor, float)
    return cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_AREA if z < .7 else cv2.INTER_LINEAR)


def g1(f):
    """Luffy in the ring of faces (POV from the deck): worry, f20 laugh, f36-60 the hat comes off and down over the lens"""
    from op.comp import lerp_keys
    g = f // 2 * 2; k = lambda K: float(lerp_keys(g if f < 36 else f, K))
    e = ('worry', 'smile', 0.) if g < 18 else (('worry', 'laugh', -.12) if g == 18 else ('worry', 'laugh', k([(20, .7), (22, 1.1), (24, .96), (26, 1.)]))) if g < 46 else 'grin'
    hop = 0 if g < 20 else 15 * math.sin((g - 20) * .78) * math.exp(-max(0, g - 34) / 8); dip = k([(0, 0), (16, 0), (18, 12), (20, -24), (24, 0), (33, 0), (36, 10), (40, -6), (60, 0)])
    gx = 0 if g > 16 else .030 * math.sin(g * .5); cx, cy, s = 960 + 4 * math.sin(g * .2), 880 + dip - hop * .6, 190.
    R, Hd, E, L = record(cx, cy, s, .02 + .04 * math.sin(g * .13), .20 + (.06 if g >= 20 else 0) - .0012 * dip, .02 * math.sin(g * .22) + (.015 * math.sin(g * .78) if g >= 20 else 0), e, 'gold', f, hat=True if f < 38 else False, kx=dict(gx=gx, sweat=.5 if g < 18 else 0.), wind=.8, body_dy=.02 * math.sin(g * .7))
    st = HT.seat(Hd); out = np.zeros((H, W, 4), f32)
    if f < 38:
        kb = dict(wear=.6, light='gold', f=f, bounce=HT.settle(f, 20, .45) - (.12 if f >= 36 else 0)); out = HT.hat(**st, part='back', **kb).copy(); out = R.cel(L, s, out); out = over4(out, HT.hat(**st, part='front', **kb)); hp = np.array([st['cx'] - st['w'] * .30, st['cy'] + st['w'] * .10]); hz = .9
    else:
        out = R.cel(L, s, out); h = lambda K: float(lerp_keys(f, K))
        w = h([(38, st['w']), (42, st['w'] * 1.08), (46, 860), (50, 1300), (53, 1800), (56, 2700), (59, 3700)]); x = h([(38, st['cx']), (44, 900), (52, 950), (59, 960)]); y = h([(38, st['cy']), (42, st['cy'] - 70), (46, 600), (50, 470), (53, 120), (56, -620), (59, -1520)]); low = y + .47 * w; pi = h([(38, st['pitch'] + .05), (42, .55), (47, .98), (53, 1.22), (59, 1.35)]); ro = h([(38, 0), (43, -.22), (50, -.10), (59, 0)])
        hw, hy, dk = w, y, 1.
        if 50 < f < 56: i = f - 51; hw = (1570, 1950, 2480, 3150, 3900)[i]; hy = (1112, 1105, 975, 715, 600)[i] - .47 * hw; dk = (.86, .66, .46, .32, .25)[i]                # the held hat grows past the lens and goes dark; its low edge becomes the edge of the brim
        if f < 55: q = HT.hat(x, hy, hw, .05, pi, ro, light='gold', wear=.6, f=f, wind=.8, bounce=h([(38, -.3), (42, .5), (46, -.2), (52, .1), (59, 0)])).copy(); q[..., :3] *= dk; out = over4(out, q)
        hp = np.array([x - hw * .36, hy + hw * .16 * math.cos(pi)]); hz = .9 * hw / 570
    if f >= 30:
        u = float(lerp_keys(f, [(30, 0), (35, 1.08), (37, 1), (59, 1)])); a = hand_patch('pinch', 'gold', 'R', (150, 10, 18), 200)[:, ::-1]; p = np.array([300, 1250]) * (1 - u) + (hp + np.array([-34, 52]) * min(hz, 2.4)) * u
        if f < 55: q = put_patch(a, p[0], p[1], .25 - .2 * (1 - u), min(hz, 3.2), (500, 330)); out = over4(out, q if f < 51 else q * np.array([.5 + .5 * dk] * 3 + [1.], f32))
    if f >= 51:
        lo = {51: 30., 52: 330., 53: 640., 54: 870., 55: 752.}.get(f, low); q = HT.under_brim(f, (lo + 70) / 430., stars=float(lerp_keys(f, [(53, 0), (58, 1.1)])), sun=(960, 300)); fe = {51: 340., 52: 340., 53: 240.}.get(f, 0)
        if fe: v = np.clip((lo + 40 - np.arange(H, dtype=f32)) / fe, 0, 1)[:, None, None]; q = q * (v * v * (3 - 2 * v)); q[..., :3] *= .7
        out = over4(out, q)
    return out


def smile_pose(s=None, jaw=0., kid=False):
    """(cx, cy, s) so that the mouth lands on (960, 640) in the SMILE FRAME"""
    s = s or SMILE['s']; Hd = LHead(**(KID if kid else GROWN)).pose(0, 0, s, SMILE['yaw'], SMILE['pitch'], SMILE['roll']); p = Hd.P(np.array([(0., SMILE['mouth_y'])]))[0]
    return SMILE['mouth'][0] - p[0], SMILE['mouth'][1] - p[1], s


def under_hat(cx, cy, w, yaw=0., pitch=0., roll=0., kid=False, hw=None, **k):
    """inverse seat: hat fixed at (cx, cy, w) -> (head cx, cy, s)"""
    hw = hw or (3.55 if kid else 3.0); s = w / hw; x = y = 0.
    for _ in range(3):
        Hd = LHead(**(KID if kid else GROWN)).pose(x, y, s, yaw, pitch, roll); q = HT.seat(Hd, w=hw, up=-.06 if kid else 0.); x += cx - q['cx']; y += cy - q['cy']; s *= w / q['w']
    return x, y, s
