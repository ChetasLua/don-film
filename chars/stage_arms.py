"""Nine left arms (and the rubber one) for shot e2.  Every drawing is a cached sprite: premultiplied uint8 RGBA + screen anchors.
   D = drawing('zoro', 'i3')  -> dict(im, ox, oy, E entry px, knot px, d unit dir to centre, s)"""
import numpy as np, math, cv2, os, json
from op import W, H
from op import hands as HD
from op import kit
from op.ink import Cel, spl, resample, over_roi
from op.paint import rgb, f32, smooth, mix

C0 = np.array([960., 486.])
LINE = '#12122a'
CACHE = os.path.join(os.path.dirname(__file__), '..', 'cache', 'e2'); VER = 'v6'
P = HD.POSES


def _pal(skin, shade, **k):
    d = dict(HD.PAL['day'], skin=skin, shade=shade, line=LINE, nail='#f6d8c4', nail_shade='#9b7f92', tape='#ece6da', tape_shade='#8b90ae')
    d.update(k); return d


SPEC = dict(
    zoro=dict(k=2, h=(664, 736), s=172, pitch=36, yaw=164, roll=4, bld=HD.build(len=1.0, wid=1.14, boxy=.6, tip=.6, knuckle=1.14, palm=1.06, wrist=1.20, meat=1.1), pal=_pal('#c98450', '#7f4a4a', prop='#c98450', prop_shade='#7f4a4a', steel='#26332c', steel_shade='#131a1c'),
              a=P['thrust'], b=P['open'], ye=3.35, band=(.50, 1.46), tint='#7fd67a'),
    nami=dict(k=3, h=(1366, 644), s=138, pitch=32, yaw=198, roll=-4, bld=HD.FEMALE, pal=_pal('#ffdcc0', '#d09a98', nail='#ff8a5a', nail_shade='#b0506a'), a=P['reach'], b=P['open'], ye=2.1, band=(.74, 1.52), tint='#ffb15a'),
    usopp=dict(k=4, h=(524, 566), s=137, pitch=30, yaw=172, roll=-5, bld=HD.build(len=1.06, wid=.9, boxy=.45, tip=.5), pal=_pal('#a5683e', '#633a48', nail='#c99068'), a=P['splay'], b=P['open'], ye=2.1, band=(.60, 1.40), tint='#ffd75a', nerv=3.2),
    sanji=dict(k=5, h=(1404, 414), s=140, pitch=30, yaw=158, roll=3, bld=HD.SLIM, pal=_pal('#fbe0cc', '#b99aa6', prop='#1d1b26', prop_shade='#0d0c16', cuff='#1d1b26', cuff_shade='#0d0c16', cuff_light='#3f4767'), a=P['reach'], b=P['relaxed'], ye=2.1, band=(.66, 1.44), cuff=dict(kind='uniform', y=.34, flare=.8), tint='#ffe08a'),
    chopper=dict(k=6, h=(730, 252), s=124, pitch=0, yaw=0, roll=0, bld=None, pal=None, a=None, b=None, ye=1.6, band=(.5, 1.0), tint='#ff9fb8', weak=1),
    robin=dict(k=7, h=(950, 262), s=100, pitch=24, yaw=176, roll=0, bld=HD.build(len=1.08, wid=.84, boxy=.34, tip=.2, knuckle=.9, palm=.92, wrist=.86, meat=.95), pal=_pal('#eebf9a', '#a8708c', nail='#b06ad0', nail_shade='#6a3f8c'), a=P['open'], b=P['reach'], ye=1.5, band=(.34, 1.02), tint='#d79bff', weak=1),
    franky=dict(k=8, h=(1216, 760), s=186, pitch=23, yaw=186, roll=-3, bld=HD.build(len=.94, wid=1.26, boxy=.86, tip=1.0, knuckle=1.15, palm=1.16, wrist=1.35, meat=1.15), pal=_pal('#b9c4d4', '#56618a', nail='#b9c4d4', nail_shade='#56618a', prop='#d8331f', prop_shade='#7a1630', steel='#7f8aa0', steel_shade='#39415f'), a=P['open'], b=P['splay'], ye=2.9, band=(.50, 1.20), tint='#6fd8ff'),
    brook=dict(k=9, h=(548, 342), s=136, pitch=32, yaw=176, roll=5, bld=HD.SLIM, pal=None, a=P['splay'], b=P['open'], ye=2.15, band=(.46, 1.14), tint='#cfd6ff', weak=1),
    jinbe=dict(k=1, h=(1200, 262), s=224, pitch=40, yaw=182, roll=0, bld=HD.build(len=.92, wid=1.22, boxy=.6, tip=.5, knuckle=1.1, palm=1.12, wrist=1.3, meat=1.2), pal=_pal('#4f9ccb', '#22558c', nail='#f2ecd2', nail_shade='#7c86a0', prop='#d9592c', prop_shade='#7c2440', kimono='#d9592c', kimono_shade='#7c2440', kimono_light='#ff9c58'),
               a=P['open'], b=P['palm_up'], ye=2.0, band=(.44, 1.24), cuff=dict(kind='kimono', y=1.36, flare=1.0), tint='#5ab0ff'),
)
ORDER = ['jinbe', 'zoro', 'nami', 'usopp', 'sanji', 'chopper', 'robin', 'franky', 'brook']
LUFFY = dict(s=170, pal=_pal('#f0b98d', '#85607a', nail='#fbdcc6', cuff='#c9302c', cuff_shade='#5d1636', cuff_light='#ff6a4a'), bld=HD.build(len=1.0, wid=1.0, boxy=.45, tip=.5))


def unit2(v): v = np.asarray(v, float); return v / (np.hypot(*v) + 1e-9)


def dirs(spec):
    h = np.array(spec['h'], float); d = unit2(C0 - h); return h, d


def mixpose(a, b, ph, amt=.5, nerv=0., seed=0):
    """finger-phased blend of two poses"""
    q = {}
    for i, f in enumerate(('thumb',) + HD.FING):
        t = amt * (.5 + .5 * math.sin(ph - i * .55)); q[f] = tuple(np.asarray(a[f], float) * (1 - t) + np.asarray(b[f], float) * t)
    t = amt * (.5 + .5 * math.sin(ph)); q['wrist'] = tuple(np.asarray(a['wrist'], float) * (1 - t) + np.asarray(b['wrist'], float) * t); q['cup'] = a['cup'] * (1 - t) + b['cup'] * t
    return HD.tremble(q, seed, nerv) if nerv else q


def keys_for(name):
    """drawing keys -> (u travel, pose, dyaw, droll, dpitch, seed)"""
    sp = SPEC[name]; a, b = sp['a'], sp['b']; K = {}
    knife = P['slap']; wk = sp.get('weak', 0); nv = sp.get('nerv', 0)
    if a is None: a = b = knife
    wd = lambda p, deg: dict(p, wrist=(p['wrist'][0] + deg, p['wrist'][1]))
    K['e0'] = (2.15, knife, -10, 6, 0, 1); K['e1'] = (1.0, HD.lerp_pose(knife, a, .15), -8, 5, 0, 2)
    K['e2'] = (.12, HD.blend(knife, a, .55, drag=.3, curl=-6), -3, 2, 2, 3); K['e3'] = (-.30, HD.lerp_pose(HD.lerp_pose(knife, a, 1.), P['splay'], .25), 4, -3, 4, 4)
    K['e4'] = (-.22, a, 5, -3, 3, 5); K['e5'] = (-.06, HD.lerp_pose(a, b, .25), 3, -1, 1, 6); K['e6'] = (.06, HD.lerp_pose(a, b, .4), 1, 0, 0, 7)
    for i in range(8):
        ph = 2 * math.pi * i / 8; p = mixpose(a, b, ph + sp['k'], .62, nv + 1.2 * wk, 40 + i + 10 * sp['k'])
        if wk: p = wd(p, 10 + 5 * math.sin(ph))
        K['i%d' % i] = (.05 * math.sin(ph + 1.1), p, 3.2 * math.sin(ph), 1.8 * math.sin(ph + .9), 1.6 * math.sin(ph + 2.), 8)
    K['a0'] = (.26, HD.lerp_pose(a, P['splay'], .4), -3, 2, -3, 9); K['a1'] = (-.42, P['open'], 2, -2, 5, 10)
    return K


# final clasp: where each hand ends up (screen), roll offset from the line to its own side, yaw (back of hand mostly), scale
GRIP = dict(
    robin=dict(h=(946, 330), yaw=180, pitch=40, s=100, front=0), jinbe=dict(h=(1078, 398), yaw=24, pitch=20, s=158, front=0), chopper=dict(h=(826, 380), s=108, front=0), brook=dict(h=(796, 486), yaw=20, pitch=14, s=118, front=0),
    sanji=dict(h=(1126, 534), yaw=-16, pitch=12, s=122, front=1), usopp=dict(h=(800, 612), yaw=14, pitch=12, s=118, front=1), nami=dict(h=(1126, 644), yaw=-14, pitch=12, s=116, front=1),
    zoro=dict(h=(838, 760), yaw=12, pitch=14, s=150, front=1), franky=dict(h=(1096, 778), yaw=-12, pitch=14, s=164, front=1))


# ------------------------------------------------------------------------------------------ cloth on the forearm
def _tube(g, idx, ys, grow, n=64):
    sc = g.sc; ph = np.linspace(-math.pi, math.pi, n, endpoint=False); out = []
    for y in ys:
        q = HD._ring(sc, idx, np.full(n, y), ph); q[:, 0] *= grow; q[:, 2] *= grow
        Pw = q @ sc.R[idx] + sc.O[idx]; c = Pw.mean(0); nr = Pw - c; vd = np.array([0, 0, g.cam.dist]) - Pw
        out.append(dict(px=g.cam.px(Pw), near=(nr * vd).sum(1) > 0, n=nr / (np.linalg.norm(nr, axis=1)[:, None] + 1e-9), c=g.cam.px(c), w=Pw))
    return ph, out


def _arc(r):
    """near arc of a ring, ordered"""
    m = r['near']; n = len(m)
    if m.all() or not m.any(): return r['px']
    st = [i for i in range(n) if m[i] and not m[i - 1]][0]
    idx = [];  i = st
    while m[i % n] and len(idx) < n: idx.append(i % n); i += 1
    return r['px'][idx], r['n'][idx]


def _side(Rn, sgn):
    """silhouette points of the tube on one side"""
    ax = unit2(Rn[-1]['c'] - Rn[0]['c']); nr = np.array([-ax[1], ax[0]]) * sgn
    return np.array([r['px'][np.argmax((r['px'] - r['c']) @ nr)] for r in Rn])


def bandage(cel, g, idx, y0, y1, L, s, col='#ece7dc', shade='#8a90b2', ink='#14101a', grow=1.06, x=True, seed=0, w=3.2, wraps=5, lit=.10, xw=.105):
    """white wrap with a black brushed X on the forearm primitive idx, facing the lens.  returns knot px"""
    off = .10; ys = np.linspace(y0 + off, y1 + off, 9); ph, Rn = _tube(g, idx, ys, grow)
    A, B = _side(Rn, 1), _side(Rn, -1)
    e0, e1 = _arc(Rn[0]), _arc(Rn[-1]); e0 = e0[0] if isinstance(e0, tuple) else e0; e1 = e1[0] if isinstance(e1, tuple) else e1
    if np.hypot(*(e1[0] - A[-1])) > np.hypot(*(e1[-1] - A[-1])): e1 = e1[::-1]
    if np.hypot(*(e0[0] - A[0])) > np.hypot(*(e0[-1] - A[0])): e0 = e0[::-1]
    body = np.vstack([A, e1, B[::-1], e0[::-1]]); polys = [body]; cel.fill(body, col)
    L = HD.unit(L)
    # shadow: near points whose normal turns from the light
    cel.ctx.save(); cel._path(body); cel.ctx.clip()
    for a, b in zip(Rn[:-1], Rn[1:]):
        for i in range(len(ph)):
            j = (i + 1) % len(ph)
            if a['near'][i] and a['near'][j] and (a['n'][i] @ L) < lit and (a['n'][j] @ L) < lit:
                q = np.array([a['px'][i], a['px'][j], b['px'][j], b['px'][i]]); cel.fill(q, shade); cel._path(q); cel._src(shade, 1.0); cel.ctx.set_line_width(1.2); cel.ctx.stroke()
    cel.ctx.restore()
    # wraps
    vc = g.sc.R[idx] @ np.array([0, 0, 1.]); pc = math.atan2(vc[0], vc[2])
    def line(ya, yb, pa, pb, n=16, gr=grow):
        t = np.linspace(0, 1, n); q = HD._ring(g.sc, idx, off + ya + (yb - ya) * t, pc + pa + (pb - pa) * t); q[:, 0] *= gr; q[:, 2] *= gr
        return g.cam.px(q @ g.sc.R[idx] + g.sc.O[idx])
    sl = (y1 - y0) * .16
    for k in range(wraps):
        ya = y0 + (y1 - y0) * (k + .35) / wraps; cel.line(line(ya - sl, ya + sl, -1.35, 1.35), w * .42, '#6d7398', taper=(.1, .45), seed=seed + k, press=.4)
        if k % 2: cel.line(line(ya - sl * .2, ya + sl * .9, -.2, 1.2), w * .30, '#6d7398', taper=(.3, .5), seed=seed + 20 + k)
    if x:
        m = (y1 - y0) * .10
        for k, (pa, pb) in enumerate(((-.78, .78), (.78, -.78))):
            q = line(y0 + m, y1 - m, pa, pb, 22); cel.line(q, s * xw, ink, taper=(.05, .10), press=.35, seed=seed + 50 + k, tremor=1.2, tmin=.72)
            for e in (0, 1):                                                         # dry hairs at the stroke ends
                o = q[0] if e == 0 else q[-1]; dv = unit2(q[0] - q[3] if e == 0 else q[-1] - q[-4]); nn = np.array([-dv[1], dv[0]])
                for v in (-.36, .05, .4): cel.line([o + nn * s * xw * v - dv * 3, o + nn * s * xw * v + dv * s * .045 * (1.2 - abs(v))], w * .5, ink, taper=(.1, .7), seed=seed + k * 7 + e)
    for r in (Rn[0], Rn[-1]):
        a = _arc(r)
        if isinstance(a, tuple): cel.line(a[0], w * .8, LINE, taper=(.12, .12), seed=seed + 3)
    cel.line(A, w, LINE, taper=(.06, .06), seed=seed + 5); cel.line(B, w, LINE, taper=(.06, .06), seed=seed + 6)
    k = Rn[5]; j = np.argmax(((k['px'] - k['c']) @ unit2(_side(Rn, 1)[5] - k['c'])))
    return k['px'][j] * .86 + k['c'] * .14, pc


def ring_band(cel, g, idx, y0, y1, col, shade, L, w, grow=1.07, seed=0, stripes=None):
    off = .10; ys = np.linspace(y0 + off, y1 + off, 5); ph, Rn = _tube(g, idx, ys, grow); A, B = _side(Rn, 1), _side(Rn, -1); L = HD.unit(L)
    e0, e1 = _arc(Rn[0]), _arc(Rn[-1]); e0 = e0[0] if isinstance(e0, tuple) else e0; e1 = e1[0] if isinstance(e1, tuple) else e1
    if np.hypot(*(e1[0] - A[-1])) > np.hypot(*(e1[-1] - A[-1])): e1 = e1[::-1]
    if np.hypot(*(e0[0] - A[0])) > np.hypot(*(e0[-1] - A[0])): e0 = e0[::-1]
    cel.fill(np.vstack([A, e1, B[::-1], e0[::-1]]), col)
    if stripes:
        n = stripes[1]; yy = np.linspace(y0 + off, y1 + off, 2 * n + 1)
        for k in range(n):
            _, R2 = _tube(g, idx, np.linspace(yy[2 * k], yy[2 * k + 1], 3), grow * 1.003); a, b = _side(R2, 1), _side(R2, -1); e0, e1 = _arc(R2[0]), _arc(R2[-1])
            if isinstance(e0, tuple) and isinstance(e1, tuple): cel.fill(np.vstack([e0[0], e1[0][::-1]]), stripes[0])
    for a, b in zip(Rn[:-1], Rn[1:]):
        for i in range(len(ph)):
            j = (i + 1) % len(ph)
            if a['near'][i] and a['near'][j] and (a['n'][i] @ L) < .05: cel.fill(np.array([a['px'][i], a['px'][j], b['px'][j], b['px'][i]]), shade, .55)
    for r in (Rn[0], Rn[-1]):
        a = _arc(r)
        if isinstance(a, tuple): cel.line(a[0], w * .8, LINE, taper=(.1, .1), seed=seed)
    cel.line(A, w, LINE, taper=(.1, .1), seed=seed + 1); cel.line(B, w, LINE, taper=(.1, .1), seed=seed + 2)
    return Rn


# ------------------------------------------------------------------------------------------ light on a finished sprite
def light_sprite(L4, ox, oy, h, d, s, E, warm='#ff7a4a', hot='#fff6dc', skin=True, amt=1.0):
    """gradient + translucent warm edge + hot rim; light comes from direction d (toward the disc)"""
    a = L4[..., 3]
    hh = np.array([h[0] - ox, h[1] - oy]); e = np.array([E[0] - ox, E[1] - oy])
    L4 = kit.para(L4, e, hh + d * s * 1.1, [(0, '#9fc4e2'), (.25, '#cfdaee'), (.55, '#ffffff'), (1, '#fff4e2')], 'mul', .9)
    L4 = kit.para(L4, hh - d * s * .2, hh + d * s * 1.35, [(0, '#000000'), (.55, '#1a0d10'), (1, '#5a2a1e')], 'screen', .9)
    if skin: L4 = kit.rim(L4, -d[0] * s * .085, -d[1] * s * .085, warm, .42 * amt, 1.6)
    L4 = kit.rim(L4, -d[0] * s * .052, -d[1] * s * .052, '#ffd79a', .75 * amt, 1.2)
    L4 = kit.rim(L4, -d[0] * s * .030, -d[1] * s * .030, hot, 1.0 * amt, .6)
    n = np.array([-d[1], d[0]]); n = n if n[1] > 0 else -n
    L4 = kit.rim(L4, n[0] * s * .026, n[1] * s * .026, '#fff0cc', .78 * amt, .7); L4 = kit.rim(L4, -n[0] * s * .018, -n[1] * s * .018, '#8fdcf0', .55, .7)
    # faint cool bounce from the open water on the far edge
    L4 = kit.rim(L4, d[0] * s * .018, d[1] * s * .018, '#5f9fd8', .5, .8)
    return L4


def _pack(L4, ox, oy, **an):
    a = L4[..., 3]; ys, xs = np.nonzero(a > .004)
    if len(xs) == 0: L4 = L4[:8, :8]; x0 = y0 = 0
    else:
        x0, x1, y0, y1 = max(0, xs.min() - 2), xs.max() + 3, max(0, ys.min() - 2), ys.max() + 3; L4 = L4[y0:y1, x0:x1]
    return dict(im=(np.clip(L4, 0, 1) * 255 + .5).astype(np.uint8), ox=int(ox + x0), oy=int(oy + y0), **{k: np.asarray(v, float) for k, v in an.items()})


# ------------------------------------------------------------------------------------------ HD arms
def draw_hd(name, u, pose, dyaw=0, droll=0, dpitch=0, seed=0, h=None, s=None, yaw=None, pitch=None, clip_plane=True, dirv=None, res=560, E0=None):
    sp = SPEC[name]; H0, d0 = dirs(sp); hp = np.array(sp['h'] if h is None else h, float); d = unit2(C0 - H0) if dirv is None else unit2(dirv); s = s or sp['s']
    yw = (sp['yaw'] if yaw is None else yaw) + dyaw; pt = (sp['pitch'] if pitch is None else pitch) + dpitch
    roll = math.degrees(math.atan2(-d[0], -d[1])) + sp['roll'] + droll
    R = HD.euler(yw, pt, roll); ax = R @ np.array([0, 1., 0]); pos = -ax * u; cam = HD.Cam(hp[0], hp[1], s, 5.0)
    R0 = HD.euler(sp['yaw'] if yaw is None else yaw, sp['pitch'] if pitch is None else pitch, math.degrees(math.atan2(-d[0], -d[1])) + sp['roll']); ax0 = R0 @ np.array([0, 1., 0])
    ye = sp['ye']; zs = float((-ax0 * (.75 + ye))[2]); E = cam.px(-ax0 * (.75 + ye)); L = HD.unit((d[0] * .80, -d[1] * .80, .48))
    AL = 2.5
    if E0 is not None: AL = max(2.5, np.hypot(*(np.asarray(E0) - hp)) / s * 1.12)
    st = dict(nails='short', arm=AL)
    if sp.get('cuff'): st['cuff'] = sp['cuff']
    hands = []; props = []
    if name == 'robin':
        # a chain: each hand holds the wrist the next one sprouts from
        def lk(x, y, rl, z, yw_, gr=1.25): R_ = HD.euler(yw_, 16, rl); sgn = max(0., u); return HD.Hand(dict(P['grip'], wrist=(0, 0)), 'L', R=R_, pos=np.array([x, y, z]) + np.array([0, 1., -.5]) * sgn * .8, bld=sp['bld'], nails='short', arm=gr, pivot=(0, .55, -.1))
        base = [lk(-.66, 1.86, 139 + droll, -1.2, 176 + dyaw, 5.2), lk(.80, 1.16, 244 - droll, -.6, 184 - dyaw, 1.35)]
        R = HD.euler(yw, 24 + dpitch, 148 + droll * 1.4); ax = R @ np.array([0, 1., 0]); pos = -ax * u + np.array([.02, -.10, .1]); st['arm'] = 1.22
        hd = HD.Hand(pose, 'L', R=R, pos=pos, bld=sp['bld'], **st); hands = base + [hd]; hi = 2; E = cam.px(np.array([.75, 3.1, -1.2])); zs = -9.
        if u > 1.5: hands = hands[:1]; hi = 0
        elif u > .6: hands = hands[:2]; hi = 1
        hd = hands[-1]
    else:
        hd = HD.Hand(pose, 'L', R=R, pos=pos, bld=sp['bld'], **st); hands = [hd]; hi = 0
    Wf = hd.sk.fr['wrist']
    if name == 'zoro':
        q = lambda y, x=0., z=0.: hd.world(Wf.pt((x, y, z)))
        props = [HD.rod(q(2.05), q(4.6 if E0 is None else max(4.6, AL), .10, .12), .41, .50, boxy=.25, g='prop'), HD.rod(q(2.92, .03, .04), q(3.22, .04, .05), .50, .52, boxy=.15, g='steel')]
    if name == 'franky':
        q = lambda y: hd.world(Wf.pt((0, y, -.02)))
        props = [HD.rod(q(.30), q(3.6 if E0 is None else AL), .72, 1.16, boxy=.45, cap=.10, g='prop'), HD.rod(q(.16), q(.32), .50, .60, boxy=.3, g='steel')]
    if name == 'sanji' and E0 is not None: q = lambda y: hd.world(Wf.pt((0, y, 0))); props = [HD.rod(q(2.2), q(AL), .44, .50, boxy=.3, g='prop')]
    if name == 'jinbe' and E0 is not None: q = lambda y: hd.world(Wf.pt((0, y, 0))); props = [HD.rod(q(3.0), q(AL + 1), .84, 1.0, boxy=.3, g='prop')]
    w_, h_ = (1240, 1240) if E0 is None else (1700, 1240); mid = (hp + (E if E0 is None else np.asarray(E0))) / 2; ox, oy = int(mid[0] - w_ / 2), int(mid[1] - h_ / 2)
    g = HD.render(hands, cam, light=L, res=res, clip=(ox, oy, ox + w_, oy + h_), props=props, thr=.16, style=.5)
    pal = sp['pal']; lw = 2.2 + s * .016
    cel = Cel(w_, h_, ox, oy)
    if name == 'jinbe': _webs(cel, g, hd, cam, L, lw)
    HD.fill_flat(cel, g, pal)
    if name == 'jinbe' and 90 < (yw % 360) < 270:
        gid = g.gb['gid']; M = np.isin(gid % 16, (1, 2, 3, 4, 5, 6)); j_, i_ = np.nonzero(M); Pq = np.stack([cam.cx + cam.s * (g.gb['x0'] + i_ * g.gb['px']), cam.cy - cam.s * (g.gb['y0'] - j_ * g.gb['px'])], -1); wr_ = g.pts[hi]['wrist']; keep = ((Pq - wr_) @ d) > s * .06; m2 = np.zeros(M.shape, np.uint8); m2[j_[keep], i_[keep]] = 1
        for sep in (np.diff(g.gb['gid'], axis=0, prepend=0) != 0, np.diff(g.gb['gid'], axis=1, prepend=0) != 0): m2[sep] = 0
        kr = max(3, int(.042 * g.u)) | 1; er = cv2.erode(m2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kr + 1, 2 * kr + 1)))
        for q_ in HD._regions(er > 0):
            if len(q_) > .12 * g.u: cel.fill(g.b2s(HD.angular(q_, .012 * g.u, .4, True, 5)), '#bfe2ee', evenodd=True)
    HD.fill_shadow(cel, g, pal)
    if name == 'jinbe' and 90 < (yw % 360) < 270:
        for q_ in HD._regions(er > 0):
            if len(q_) > .12 * g.u: cel.fill(g.b2s(HD.angular(q_, .012 * g.u, .4, True, 5)), '#a5d3e6', .42, evenodd=True)
    deep = HD.big(HD.depth_polys(g, L, -.28), s, .03); HD.fill_polys(cel, g, deep, mix(rgb(pal['shade']), rgb('#2a2458'), .42), .85, names=('palm', 'thumb') + HD.FING)
    HD.fill_overlays(cel, g, pal, lw)
    rp = HD.big(HD.rim_polys(g, (d[0], -d[1], -.30), lo=.42, graze=.62), s, .006); HD.fill_polys(cel, g, rp, mix(rgb(pal['skin']), rgb('#ffe2b4'), .72), .95, names=('palm', 'thumb') + HD.FING)
    HD.hatch(cel, g, deep, gap=.062, w=1.35, col=mix(rgb(pal['shade']), rgb(LINE), .55), seed=seed + 3, ang=1.0, alpha=.75, names=('palm', 'thumb') + HD.FING, length=.15, fade=.75)
    HD.ink_lines(cel, g, pal, lw, seed + 11)
    if name == 'jinbe': _claws(cel, g, hd, cam, lw)
    ov = Cel(w_, h_, ox, oy)
    fa = [i for i, (hh, pr) in enumerate(g.sc.meta) if hh == hi and pr['where'] == 'body' and pr['L'] == hd.sk.arm][0]
    if name == 'franky':
        pr_i = [i for i, (hh, pr) in enumerate(g.sc.meta) if pr['g'] == 'prop'][0]
        knot, pc = bandage(ov, g, pr_i, sp['band'][0] - .30, sp['band'][1] - .30, L, s, seed=7, w=lw * .85, xw=.15, grow=1.03)
        _star(ov, g, pr_i, 1.78, pc, .52, lw)
        def rg(y, p0, p1, w_, col): tt = np.linspace(p0, p1, 26); q = HD._ring(g.sc, pr_i, np.full(26, y), pc + tt); q[:, 0] *= 1.012; q[:, 2] *= 1.012; ov.line(cam.px(q @ g.sc.R[pr_i] + g.sc.O[pr_i]), w_, col, taper=(.1, .1), seed=int(y * 10))
        def rv(y, ph): q = HD._ring(g.sc, pr_i, np.array([y]), np.array([pc + ph])) * [1.015, 1, 1.015]; p_ = cam.px(q @ g.sc.R[pr_i] + g.sc.O[pr_i])[0]; a_ = np.linspace(0, 6.283, 14); r_ = s * .030; ov.fill(np.c_[p_[0] + np.cos(a_) * r_, p_[1] + np.sin(a_) * r_], '#8a1c24'); ov.fill(np.c_[p_[0] - r_ * .2 + np.cos(a_) * r_ * .6, p_[1] - r_ * .2 + np.sin(a_) * r_ * .6], '#ffb09a')
        for y_ in (.10, 1.10, 2.46): rg(y_, -1.5, 1.5, lw * .9, LINE); rg(y_ + .05, -1.4, 1.4, lw * .45, '#ff9a7a'); [rv(y_ + .13, ph_) for ph_ in np.linspace(-1.2, 1.2, 7)]
        for ph_ in (-1.02, 1.02): tt = np.linspace(1.12, 2.46, 20); q = HD._ring(g.sc, pr_i, tt, np.full(20, pc + ph_)); q[:, 0] *= 1.012; q[:, 2] *= 1.012; ov.line(cam.px(q @ g.sc.R[pr_i] + g.sc.O[pr_i]), lw * .6, LINE, taper=(.1, .1), seed=int(ph_ * 10))
        for f_ in HD.FING:                                                                  # knuckle plates
            for a_, b_, u_ in (('mcp', 'pip', .18), ('mcp', 'pip', .84), ('pip', 'dip', .20), ('pip', 'dip', .82)):
                A_, B_ = g.pts[hi][f_, a_], g.pts[hi][f_, b_]; ax_ = unit2(B_ - A_); nn_ = np.array([-ax_[1], ax_[0]]); c_ = A_ + (B_ - A_) * u_; w__ = s * .115; cel.line(np.array([c_ - nn_ * w__, c_ + ax_ * s * .018, c_ + nn_ * w__]), lw * .6, LINE, taper=(.15, .15), seed=3)
        pm_ = g.pts[hi]['palm']; a_ = np.linspace(0, 6.283, 30); cel.line(np.c_[pm_[0] + np.cos(a_) * s * .22, pm_[1] + np.sin(a_) * s * .22], lw * .8, LINE, taper=(0, 0)); cel.line(np.c_[pm_[0] + np.cos(a_) * s * .15, pm_[1] + np.sin(a_) * s * .15], lw * .45, LINE, taper=(0, 0)); [cel.fill(np.c_[pm_[0] + math.cos(v) * s * .30 + np.cos(a_) * s * .022, pm_[1] + math.sin(v) * s * .30 + np.sin(a_) * s * .022], '#39415f') for v in np.linspace(.4, 6.68, 7)[:-1]]
    else:
        cf = [i for i, (hh, pr) in enumerate(g.sc.meta) if pr['where'] == 'ext' and pr['T'] == 2] if sp.get('cuff') else []
        if name == 'sanji':
            ring_band(ov, g, fa, .16, .37, '#f3c64a', '#94602f', L, lw * .8, 1.10, 3); knot, pc = bandage(ov, g, fa, *sp['band'], L, s, seed=8, w=lw * .85, grow=1.30)
            q = HD._ring(g.sc, fa, np.array([.37]), np.array([pc - .7])); q[:, 0] *= 1.12; q[:, 2] *= 1.12; p0 = cam.px(q @ g.sc.R[fa] + g.sc.O[fa])[0]; a_ = np.linspace(0, 6.283, 18); r_ = s * .052
            ov.fill(np.c_[p0[0] + np.cos(a_) * r_, p0[1] + np.sin(a_) * r_], '#ffe9a8'); ov.line(np.c_[p0[0] + np.cos(a_) * r_, p0[1] + np.sin(a_) * r_], lw * .5, LINE, taper=(0, 0))
        else:
            knot, pc = bandage(ov, g, fa, *sp['band'], L, s, seed=9, w=lw * .85)
        if name == 'usopp': ring_band(ov, g, fa, .10, .46, '#e9edf2', '#7d86ac', L, lw * .8, 1.09, 5, stripes=('#3f74c8', 3))
        if name == 'nami':
            ring_band(ov, g, fa, .50, .60, '#ffd45a', '#a8662c', L, lw * .7, 1.20, 4); ring_band(ov, g, fa, .08, .26, '#8a5a3a', '#4b2c3e', L, lw * .7, 1.10, 6); _logpose(ov, g, fa, pc, s, lw)
    ovl = ov.rgba()
    fm = Cel(w_, h_, ox, oy)                                                                # fingers stay over the cloth
    for G in g.groups:
        if (G['name'] in HD.FING + ('thumb',) or G['name'].startswith('nail') or (name == 'robin' and G['hand'] != hi)) and G['polys']:
            if name == 'robin' and G['hand'] != hi and G['name'] == 'palm': continue
            fm.fill(G['polys'], '#ffffff', evenodd=True)
    ovl = ovl * (1 - fm.rgba()[..., 3:4]); L4 = cel.rgba(); L4 = L4 * (1 - ovl[..., 3:4]) + ovl
    if clip_plane:
        gb = g.gb; m = ((gb['z'] < zs) & (gb['gid'] > 0)).astype(f32); k = gb['px'] * cam.s; org = (cam.cx + cam.s * gb['x0'] - ox, cam.cy - cam.s * gb['y0'] - oy)
        M = cv2.warpAffine(cv2.dilate(m, np.ones((5, 5), np.uint8)), np.float32([[k, 0, org[0]], [0, k, org[1]]]), (w_, h_), flags=cv2.INTER_LINEAR)
        # everything (cloth too) beyond the waterline: half plane behind the entry ellipse
        yy, xx = np.mgrid[0:h_, 0:w_].astype(f32); t = (xx + ox - E[0]) * d[0] + (yy + oy - E[1]) * d[1]; M = np.maximum(M, smooth(-s * .34, -s * .52, t)) if name != 'robin' else M * 0; L4 = L4 * (1 - cv2.GaussianBlur(M, (0, 0), 1.2))[..., None]
    if E0 is not None: yy, xx = np.mgrid[0:h_, 0:w_].astype(f32); e0 = np.asarray(E0); dd = unit2(hp - e0); tt = (xx + ox - e0[0]) * dd[0] + (yy + oy - e0[1]) * dd[1]; L4 = L4 * (smooth(.06, .58, tt / (np.hypot(*(hp - e0)) + 1e-6)) if name not in ('zoro', 'franky', 'jinbe', 'robin') else smooth(-s * .25, s * .05, tt))[..., None]
    if E0 is not None: L4 = kit.para(L4, (E0[0] - ox, E0[1] - oy), (hp[0] - ox, hp[1] - oy), [(0, '#1d3f66'), (.35, '#4d7ba6'), (.72, '#c9d8ea'), (1, '#ffffff')], 'mul', .95)
    L4 = light_sprite(L4, ox, oy, hp + (ax * np.array([1, -1, 1]))[:2] * -u * s * 0, d, s, E, skin=True, warm='#ff7a4a' if name != 'jinbe' else '#48d0ff')
    tip = g.pts[hi]['middle', 'tip']; return _pack(L4, ox, oy, E=E, knot=knot, d=d, s=s, tip=tip, palm=g.pts[hi]['palm'], wrist=g.pts[hi]['wrist'])


def _star(c, g, idx, y, pc, r, lw):
    sc = g.sc; P5 = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5; q = r * (1 if k % 2 == 0 else .42); P5.append((y + q * math.sin(a), q * math.cos(a) / .8))
    def pr(P_, gr=1.02):
        P_ = np.array(P_ + P_[:1]); out = []
        for a, b in zip(P_[:-1], P_[1:]):
            t = np.linspace(0, 1, 8)[:, None]; out.append(a + (b - a) * t)
        v = np.vstack(out); q = HD._ring(sc, idx, v[:, 0], pc + v[:, 1]); q[:, 0] *= gr; q[:, 2] *= gr; return g.cam.px(q @ sc.R[idx] + sc.O[idx])
    c.fill(pr([(y + (a - y) * 1.28, b * 1.28) for a, b in P5]), '#f4f2ee'); c.fill(pr(P5), '#2f63c8'); kit.outline(c, pr([(y + (a - y) * 1.28, b * 1.28) for a, b in P5]), lw * .6, LINE, 3)


def _logpose(c, g, idx, pc, s, lw):
    q = HD._ring(g.sc, idx, np.array([.27]), np.array([pc])); q[:, 0] *= 1.4; q[:, 2] *= 1.4; p = g.cam.px(q @ g.sc.R[idx] + g.sc.O[idx])[0]; r = s * .185; a = np.linspace(0, 6.2832, 30)
    ci = lambda k, dx=0., dy=0., sy=1.: np.c_[p[0] + dx * r + np.cos(a) * r * k, p[1] + dy * r + np.sin(a) * r * k * sy]
    c.fill(ci(1.14), '#c99a4a'); c.line(ci(1.14), lw * .7, LINE, taper=(0, 0)); c.fill(ci(1.0), '#bfe6f2'); c.fill(np.vstack([ci(1.0)[2:17], ci(.92, -.12, -.28)[2:17][::-1]]), '#6ea9cc')
    c.line(ci(1.0), lw * .6, LINE, taper=(0, 0)); n = np.array([.62, -.5]); n /= np.hypot(*n); m = np.array([-n[1], n[0]])
    c.fill(np.array([p + n * r * .72, p + m * r * .13, p - m * r * .13]), '#d8262e'); c.fill(np.array([p - n * r * .62, p + m * r * .13, p - m * r * .13]), '#f2efe8'); c.fill(ci(.10), '#17131a')
    c.fill(ci(.22, -.42, -.44, .8), '#ffffff'); c.fill(ci(.08, -.12, -.62), '#ffffff')


def _webs(c, g, hd, cam, L, lw):
    pt = g.pts[0]
    for f, n in zip(HD.FING[:-1], HD.FING[1:]):
        a0, a1, a2 = pt[f, 'mcp'], pt[f, 'pip'], pt[f, 'dip']; b0, b1, b2 = pt[n, 'mcp'], pt[n, 'pip'], pt[n, 'dip']
        m = (a1 + b1) / 2 * .80 + (a0 + b0) / 2 * .20; Q = np.vstack([a0, a1, a1 * .45 + a2 * .55, spl([a1 * .45 + a2 * .55, (a1 + b1) / 2 * .62 + m * .38, b1 * .45 + b2 * .55], n=8), b1, b0])
        c.fill(Q, '#8fc3d8'); e = spl([a1 * .45 + a2 * .55, (a1 + b1) / 2 * .62 + m * .38, b1 * .45 + b2 * .55], n=8); c.fill(np.vstack([e, spl([b1 * .6 + b0 * .4, m, a1 * .6 + a0 * .4], n=6)]), '#5f97ba')
        c.line(e, lw * .8, LINE, taper=(.1, .1), seed=4)
        for t in (.3, .55, .78): c.line([a0 * (1 - t) + b0 * t, e[int(t * (len(e) - 1))]], lw * .35, '#3c6c98', taper=(.2, .5), seed=int(t * 10))
    a0, a1, b0 = pt['index', 'mcp'], pt['index', 'pip'], pt['thumb', 'ip']; e = spl([a0 * .3 + a1 * .7, (a0 + b0) / 2, pt['thumb', 'mcp'] * .4 + b0 * .6], n=8); c.fill(np.vstack([a0, e, pt['thumb', 'mcp']]), '#8fc3d8'); c.line(e, lw * .8, LINE, taper=(.1, .1))


def _claws(c, g, hd, cam, lw):
    pass


# ------------------------------------------------------------------------------------------ bones
def draw_brook(u, pose, dyaw=0, droll=0, dpitch=0, seed=0, h=None, s=None, yaw=None, pitch=None, clip_plane=True, dirv=None, E0=None):
    sp = SPEC['brook']; H0, d0 = dirs(sp); hp = np.array(sp['h'] if h is None else h, float); d = d0 if dirv is None else unit2(dirv); s = s or sp['s']
    yw = (sp['yaw'] if yaw is None else yaw) + dyaw; pt = (sp['pitch'] if pitch is None else pitch) + dpitch; base = math.degrees(math.atan2(-d[0], -d[1])) + sp['roll']
    R = HD.euler(yw, pt, base + droll); ax = R @ np.array([0, 1., 0]); cam = HD.Cam(hp[0], hp[1], s, 5.0); hd = HD.Hand(pose, 'L', R=R, pos=-ax * u, bld=sp['bld'], arm=2.5)
    ax0 = HD.euler(sp['yaw'] if yaw is None else yaw, sp['pitch'] if pitch is None else pitch, base) @ np.array([0, 1., 0]); E = cam.px(-ax0 * (.75 + sp['ye']))
    w_, h_ = 1100, 1100; mid = (hp + E) / 2; ox, oy = int(mid[0] - w_ / 2), int(mid[1] - h_ / 2); lw = 4.0; cel = Cel(w_, h_, ox, oy)
    LT, MD, DK = '#f1ead6', '#a6a3ba', '#54567c'
    bones = []
    wp = lambda v: hd.world(v)
    for f in HD.FING:
        F = hd.sk.fr[f]; Ln = hd.sk.fr[f + '_L']; mb, mo = hd.sk.fr[f + '_mc']; J = [np.asarray(mb) + [0, .04, 0], F[0].o, F[1].o, F[2].o, F[2].pt((0, Ln[2], 0))]
        for i, (a, b) in enumerate(zip(J[:-1], J[1:])): bones.append((wp(a), wp(b), (.088, .080, .070, .060)[i], (.092, .074, .064, .036)[i]))
    C, T1, T2 = hd.sk.fr['thumb']; TL = hd.sk.fr['thumb_L']; J = [C.o, T1.o, T2.o, T2.pt((0, TL[2], 0))]
    for i, (a, b) in enumerate(zip(J[:-1], J[1:])): bones.append((wp(a), wp(b), (.10, .088, .072)[i], (.094, .078, .042)[i]))
    Wf = hd.sk.fr['wrist']; q = lambda x, y, z=0.: wp(Wf.pt((x, y, z)))
    bones.append((q(.105, .05), q(.15, 2.4), .125, .10)); bones.append((q(-.10, .02), q(-.14, 2.4), .105, .13))
    r = np.random.default_rng(3)
    for k in range(7):                                                                   # carpals
        x, y = [(-.18, .02), (-.06, .0), (.07, .0), (.19, .02), (-.12, .14), (.02, .13), (.15, .14)][k]; c_ = wp((x, y, 0)); bones.append((c_, c_, .085, .085))
    def bone(a, b, wa, wb, sd):
        pa, pb = cam.px(a), cam.px(b); fa, fb = [cam.dist / (cam.dist - v[2]) for v in (a, b)]; ra, rb = wa * s * fa, wb * s * fb; v = pb - pa; Ln = np.hypot(*v)
        if Ln < ra * .6:
            t = np.linspace(0, 6.283, 14)[:-1]; rr = ra * (1 + .12 * np.sin(t * 3 + sd)); Q = np.c_[pa[0] + np.cos(t) * rr, pa[1] + np.sin(t) * rr * .86]; cel.fill(spl(Q, True), LT); cel.shade(np.c_[pa[0] + np.cos(t) * rr - d[0] * ra * .5, pa[1] + np.sin(t) * rr - d[1] * ra * .5], MD, spl(Q, True)); kit.outline(cel, spl(Q, True), lw * .55, LINE, sd); return
        v /= Ln; n = np.array([-v[1], v[0]]); o = lambda t, w_: pa + v * Ln * t + n * w_
        Q = [o(-.02, 0), o(-.045, ra * .62), o(.03, ra * 1.08), o(.15, ra * .86), o(.34, ra * .56), o(.6, rb * .54), o(.82, rb * .72), o(.93, rb * 1.05), o(1.02, rb * .66), o(1.0, 0), o(1.02, -rb * .66), o(.93, -rb * 1.05), o(.82, -rb * .72), o(.6, -rb * .54), o(.34, -ra * .56), o(.15, -ra * .86), o(.03, -ra * 1.08), o(-.045, -ra * .62)]
        S = spl(Q, True, n=6); cel.fill(S, LT); sg = -1 if n @ d > 0 else 1
        cel.shade(np.vstack([[o(-.1, sg * ra * .10), o(.5, sg * min(ra, rb) * -.02), o(1.1, sg * rb * .10)], [o(1.1, sg * rb * 2), o(-.1, sg * ra * 2)]]), MD, S)
        cel.shade(np.vstack([[o(-.1, sg * ra * .72), o(.5, sg * min(ra, rb) * .36), o(1.1, sg * rb * .72)], [o(1.1, sg * rb * 2), o(-.1, sg * ra * 2)]]), DK, S)
        cel.line([o(.30, -sg * ra * .05), o(.62, -sg * rb * .05)], lw * .3, '#8f8aa6', taper=(.4, .6), seed=sd); cel.line([o(.10, ra * .3), o(.17, ra * .1)], lw * .35, LINE, taper=(.2, .6), seed=sd + 1)
        kit.outline(cel, S, lw * .78, LINE, sd, press=.45)
    for k, b in sorted(enumerate(bones), key=lambda t: (t[1][0][2] + t[1][1][2])): bone(*b, 10 + k)
    # bandage over both forearm bones, lace cuff and dark sleeve beyond
    def tube(y, hw): c_ = q(0, y); f = cam.dist / (cam.dist - c_[2]); return cam.px(c_), hw * s * f
    ys = np.linspace(*sp['band'], 7); T = [tube(y, .245 + .02 * math.sin(y * 9)) for y in ys]; av = unit2(T[-1][0] - T[0][0]); nn = np.array([-av[1], av[0]]); A = np.array([p + nn * w for p, w in T]); B = np.array([p - nn * w for p, w in T]); bd = np.vstack([A, B[::-1]])
    cel.fill(spl(bd, True, corners=(0, 6, 7, 13), n=4), '#ece7dc'); sg = -1 if nn @ d > 0 else 1; cel.shade(np.vstack([np.array([p + nn * w * .25 * sg for p, w in T]), np.array([p + nn * w * 1.5 * sg for p, w in T])[::-1]]), '#8a90b2', bd)
    for k in range(5):
        i = (k + .3) / 5 * 5.6; j = int(i); p0 = A[j] * (1 - .5) + A[min(6, j + 1)] * .5; p1 = B[min(6, j + 1)]; cel.line([p0, (p0 + p1) / 2 + av * 4, p1], lw * .4, '#6d7398', taper=(.1, .5), seed=k)
    for a_, b_ in ((A, B), (B, A)): cel.line(np.array([a_[0] * .80 + b_[0] * .20 + av * 6, (a_[3] + b_[3]) / 2, a_[-1] * .20 + b_[-1] * .80 - av * 6]), s * .135, '#14101a', taper=(.05, .1), press=.35, seed=4, tremor=1.2, tmin=.7)
    kit.outline(cel, bd, lw * .8, LINE, 5)
    T2_ = [tube(y, w) for y, w in ((1.34, .36), (1.7, .40), (2.1, .42), (2.6 if E0 is None else 5.4, .44 if E0 is None else .62))]; A2 = np.array([p + nn * w for p, w in T2_]); B2 = np.array([p - nn * w for p, w in T2_]); sl = np.vstack([A2, B2[::-1]])
    cel.fill(sl, '#2c2340'); cel.shade(np.vstack([np.array([p + nn * w * .1 * sg for p, w in T2_]), np.array([p + nn * w * 1.6 * sg for p, w in T2_])[::-1]]), '#17122a', sl); kit.outline(cel, sl, lw, LINE, 6)
    for m, (y, w, n_) in enumerate(((1.40, .50, 9), (1.24, .46, 8))):                       # two lace frills
        c_, rw = tube(y, w); P_ = []
        for k in range(n_ * 2 + 1):
            t = -1 + k / n_; P_.append(c_ + nn * rw * t - av * (s * (.05 + (.13 if k % 2 else .02)) * (1.0 - .35 * t * t)) + av * s * .10 * t * t)
        top = [c_ + nn * rw * .92 + av * s * .16, c_ + av * s * .10, c_ - nn * rw * .92 + av * s * .16]; Q = np.vstack([spl(P_, n=6), top]); cel.fill(Q, '#f6f1ee'); cel.shade(Q + nn * sg * rw * .8, '#9aa0c2', Q); kit.outline(cel, Q, lw * .7, LINE, 20 + m)
        for k in range(2, n_ * 2 - 1, 2): cel.line([P_[k], P_[k] + av * s * .14], lw * .4, '#6d7398', taper=(.1, .6), seed=k)
        for k in range(1, n_ * 2, 2):
            p = P_[k] + av * s * .035; t = np.linspace(0, 6.283, 9); cel.fill(np.c_[p[0] + np.cos(t) * 3.2, p[1] + np.sin(t) * 3.2], '#7d84aa')
    L4 = cel.rgba(); yy, xx = np.mgrid[0:h_, 0:w_].astype(f32); t = (xx + ox - E[0]) * d[0] + (yy + oy - E[1]) * d[1]
    if clip_plane: L4 = L4 * (1 - smooth(s * .05, -s * .12, t))[..., None]
    if E0 is not None: e0 = np.asarray(E0); dd = unit2(hp - e0); L4 = L4 * smooth(.06, .58, ((xx + ox - e0[0]) * dd[0] + (yy + oy - e0[1]) * dd[1]) / (np.hypot(*(hp - e0)) + 1e-6))[..., None]
    L4 = light_sprite(L4, ox, oy, hp, d, s, E, skin=False)
    return _pack(L4, ox, oy, E=E, knot=A[4], d=d, s=s, tip=cam.px(bones[11][1]), palm=cam.px(wp((0, .5, 0))), wrist=cam.px(wp((0, 0, 0))))


# ------------------------------------------------------------------------------------------ hoof
def draw_chopper(u, spread=.5, bend=0., seed=0, h=None, s=None, clip_plane=True, dirv=None, E0=None):
    sp = SPEC['chopper']; H0, d0 = dirs(sp); hp = np.array(sp['h'] if h is None else h, float); d = d0 if dirv is None else unit2(dirv); s = s or sp['s']; n = np.array([-d[1], d[0]])
    if n[1] < 0: n = -n
    E = hp - d * s * 1.55; w_, h_ = 800, 800; mid = (hp + E) / 2; ox, oy = int(mid[0] - w_ / 2), int(mid[1] - h_ / 2); cel = Cel(w_, h_, ox, oy); lw = 4.2
    k = lambda y: 1 + .16 * y                                                             # perspective along the arm
    def T(P_, b=1.):
        P_ = np.asarray(P_, float); x, y = P_[:, 0] * k(P_[:, 1]), P_[:, 1] - u * .82; x = x + bend * np.maximum(0, y + .8) ** 2 * .22; return hp + n * (x * s)[:, None] + d * (y * s)[:, None]
    FUR, FSH, HF, HFS, HFL = '#a8703f', '#57385a', '#4a3433', '#241a30', '#8f7672'
    arm = [(-.30, .30), (-.36, .02), (-.33, -.4), (-.37, -.9), (-.42, -1.5), (-.45 if E0 is None else -.58, -2.4 if E0 is None else -4.4), (.45 if E0 is None else .58, -2.4 if E0 is None else -4.4), (.43, -1.5), (.38, -.9), (.34, -.4), (.37, .02), (.30, .30)]
    q = T(arm); S = spl(q, False, n=8); S = np.vstack([S, S[:1]])
    # furry silhouette
    r = np.random.default_rng(5); Q = []
    for i in range(0, len(S) - 1, 3):
        p = S[i]; v = S[min(i + 3, len(S) - 1)] - p; ln = np.hypot(*v)
        if ln < 1: continue
        m = np.array([v[1], -v[0]]) / ln; c_ = p + v * .62; out = m if (c_ - T([(0, -.8)])[0]) @ m > 0 else -m; Q += [p, p + v * .35 + out * 3, p + v * .62 + out * (s * .055) * r.uniform(.5, 1.2) - d * s * .05, p + v * .72]
    Q = np.array(Q); cel.fill(Q, FUR)
    sg = 1 if n @ d >= 0 else -1; lit = n * np.sign(n @ unit2(C0 - hp) + 1e-6)
    cel.shade(T([(.05, .5), (.10, -.3), (.02, -1.2), (.06, -4.4), (.9, -4.4), (.9, .5)]), FSH, Q); cel.shade(T([(-.26, .5), (-.20, -.3), (-.28, -1.2), (-.25, -4.4), (-.9, -4.4), (-.9, .5)]), '#8a5a3c', Q)
    for i in range(26):
        x, y = r.uniform(-.30, .34), r.uniform(-1.6, .2); a = T([(x, y), (x * 1.04 + .01, y - .07), (x * 1.1 + .03, y - .13)]); cel.line(a, 1.5, '#3d2638' if x > 0 else '#6d4432', taper=(.1, .7), seed=i)
    kit.outline(cel, Q, lw * .82, LINE, 3, press=.4)
    # cloven hoof
    for sd in (-1, 1):
        sp_ = sd * (.04 + .13 * spread); toe = [(sd * .34, .16), (sd * .45 + sp_ * .4, .42), (sd * .43 + sp_, .74), (sd * .28 + sp_, .98), (sd * .12 + sp_ * .9, .99), (sd * .035 + sp_ * .5, .74), (sd * .03, .46), (sd * .03, .2)]
        t = spl(T(toe), True, corners=(0, 7), n=8); cel.fill(t, HF); cel.shade(T([(sd * .5, .1), (sd * .30, .3), (sd * .34 + sp_, .7), (sd * .2 + sp_, 1.05), (sd * .8, 1.1), (sd * .8, .1)]) if sd > 0 else T([(sd * .02, .1), (sd * .12, .4), (sd * .14, .8), (sd * .05, 1.1), (.2, 1.1), (.2, .1)]), HFS, t)
        cel.fill(spl(T([(sd * .30 + sp_ * .6, .46), (sd * .36 + sp_ * .9, .64), (sd * .33 + sp_, .80), (sd * .30 + sp_ * .8, .64)]), True, n=6), HFL)
        kit.outline(cel, t, lw, LINE, 8 + sd, press=.4)
    tuft = []                                                                             # fur fringe over the hoof
    for i in range(13): tuft.append((-.40 + .80 * i / 12, .20 + (.16 if i % 2 else .0) * r.uniform(.6, 1.2) + .05 * math.cos((i - 6) / 6)))
    tq = T(tuft + [(.40, .02), (-.40, .02)]); cel.fill(tq, FUR); cel.shade(T([(.04, .6), (.10, 0), (.6, 0), (.6, .6)]), FSH, tq); cel.line(T(tuft), lw * .7, LINE, taper=(.1, .1), seed=2)
    y0, y1 = -.42, -.98; bd = T([(-.40, y0), (0, y0 + .07), (.40, y0), (.425, y1), (0, y1 + .07), (-.415, y1)]); bs = spl(bd, True, corners=(0, 2, 3, 5), n=6)
    cel.fill(bs, '#ece7dc'); cel.shade(T([(.10, 0), (.12, -2), (.9, -2), (.9, 0)]), '#8a90b2', bs)
    for i in range(4): cel.line(T([(-.40, y0 - .10 - i * .13), (0, y0 - .02 - i * .13), (.40, y0 - .04 - i * .13)]), lw * .38, '#6d7398', taper=(.1, .4), seed=i)
    for a_, b_ in (((-.27, y0 - .06), (.27, y1 + .10)), ((.27, y0 - .06), (-.27, y1 + .10))): cel.line(T([a_, (0, (y0 + y1) / 2 + .03), b_]), s * .125, '#14101a', taper=(.06, .1), press=.35, seed=6, tremor=1.2, tmin=.7)
    kit.outline(cel, bs, lw * .85, LINE, 5); L4 = cel.rgba(); yy, xx = np.mgrid[0:h_, 0:w_].astype(f32); t = (xx + ox - E[0]) * d[0] + (yy + oy - E[1]) * d[1]
    if clip_plane: L4 = L4 * (1 - smooth(s * .04, -s * .10, t))[..., None]
    if E0 is not None: e0 = np.asarray(E0); dd = unit2(hp - e0); L4 = L4 * smooth(.06, .58, ((xx + ox - e0[0]) * dd[0] + (yy + oy - e0[1]) * dd[1]) / (np.hypot(*(hp - e0)) + 1e-6))[..., None]
    L4 = light_sprite(L4, ox, oy, hp, d, s, E, skin=False); return _pack(L4, ox, oy, E=E, knot=T([(.40, -.75)])[0], d=d, s=s, tip=T([(0, .98)])[0], palm=T([(0, .5)])[0], wrist=T([(0, .1)])[0])


# ------------------------------------------------------------------------------------------ the rubber hand
def draw_luffy(pose, h, pitch=-34, yaw=8, roll=0, arm=2.6, s=None, seed=0, res=640):
    """his left arm from below: back of the hand to us, forearm coming at the lens; arm = forearm length in hand units (stretch)"""
    s = s or LUFFY['s']; hp = np.array(h, float); R = HD.euler(yaw, pitch, roll); cam = HD.Cam(hp[0], hp[1], s, 4.4); hd = HD.Hand(pose, 'L', R=R, bld=LUFFY['bld'], nails='short', arm=arm)
    L = HD.unit((.10, .86, -.10)); w_, h_ = 1100, 1400; ox, oy = int(hp[0] - 550), int(hp[1] - 420); g = HD.render([hd], cam, light=L, res=res, clip=(ox, oy, ox + w_, min(oy + h_, 1200)), thr=.02, style=.5)
    pal = LUFFY['pal']; cel = Cel(w_, h_, ox, oy); lw = 5.2; HD.fill_flat(cel, g, pal); HD.fill_shadow(cel, g, pal)
    HD.fill_polys(cel, g, HD.big(HD.depth_polys(g, L, -.55), s, .03), '#4c3a66', .7, names=('palm', 'thumb') + HD.FING); HD.fill_overlays(cel, g, pal, lw)
    HD.fill_polys(cel, g, HD.big(HD.rim_polys(g, (0, .9, -.35), lo=.42, graze=.62), s, .006), '#ffe0b4', .95); HD.hatch(cel, g, gap=.05, w=1.5, col='#4c3350', seed=seed, ang=.9, alpha=.7, length=.15, fade=.75); HD.ink_lines(cel, g, pal, lw, 5)
    ov = Cel(w_, h_, ox, oy); fa = [i for i, (hh, pr) in enumerate(g.sc.meta) if pr['where'] == 'body' and pr['L'] == hd.sk.arm][0]; st = min(1.25, arm / 2.6)
    vc = g.sc.R[fa] @ np.array([0, 0, 1.]); pc = math.atan2(vc[0], vc[2])
    def sp_(y, ph): q = HD._ring(g.sc, fa, np.atleast_1d(y).astype(float), pc + np.atleast_1d(ph)); q[:, 0] *= 1.004; q[:, 2] *= 1.004; return g.cam.px(q @ g.sc.R[fa] + g.sc.O[fa])
    sgn = 1. if sp_(.8, .3)[0][0] > sp_(.8, -.3)[0][0] else -1.
    GL = {'3': [[(.12, .10), (.45, .0), (.84, .10), (.86, .34), (.46, .46)], [(.46, .46), (.92, .56), (.92, .86), (.50, 1.), (.08, .88)]], 'D': [[(.14, .0), (.14, 1.)], [(.02, .02), (.60, .06), (.98, .40), (.80, .86), (.04, .98)]], '2': [[(.06, .24), (.34, .0), (.80, .06), (.86, .36), (.46, .68), (.08, .96)], [(.08, .96), (.96, .94)]], 'Y': [[(.02, .0), (.48, .48)], [(.98, .0), (.48, .48), (.44, 1.)]]}
    def glyph(ch, x0, y0, w__, h__):
        for st_ in GL[ch]: q = spl(np.array(st_), n=8); ov.line(sp_(.10 + (y0 + q[:, 1] * h__) * st, sgn * (x0 + q[:, 0] * w__)), s * .046, '#241a3a', taper=(.06, .12), press=.3, seed=ord(ch), tremor=.6, tmin=.6)
    glyph('3', -1.0, .30, .80, .52); glyph('D', .20, .30, .80, .52); glyph('2', -1.0, .98, .80, .52); glyph('Y', .20, .98, .80, .52)
    for x0, x1 in ((-1.18, 1.18), (1.18, -1.18)): q = np.linspace(0, 1, 16); ov.line(sp_(.10 + (.24 + .64 * q) * st, sgn * (x0 + (x1 - x0) * q)), s * .034, '#241a3a', taper=(.06, .12), press=.35, seed=int(x0 * 10) + 30, tremor=.8, tmin=.6)
    knot = sp_(.10 + 1.7 * st, 1.3 * sgn)[0]; ovl = ov.rgba(); L4 = cel.rgba(); L4 = L4 * (1 - ovl[..., 3:4] * .94) + ovl * .94
    d = np.array([0., -1.]); hl = np.array([hp[0] - ox, hp[1] - oy]); L4 = kit.para(L4, hl + [0, s * 4.2], hl - [0, s * 1.2], [(0, '#4d6496'), (.45, '#a9b4d8'), (.8, '#ffffff'), (1, '#fff0da')], 'mul', .95)
    L4 = kit.rim(L4, 0, s * .075, '#ff7a4a', .5, 1.6); L4 = kit.rim(L4, 0, s * .045, '#ffd79a', .8, 1.2); L4 = kit.rim(L4, 0, s * .026, '#fff6dc', 1., .6)
    for sx in (-1, 1): L4 = kit.rim(L4, sx * s * .020, 0, '#8fd0f0', .55, .8)
    return _pack(L4, ox, oy, E=hp, knot=knot, d=d, s=s, tip=g.pts[0]['middle', 'tip'], palm=g.pts[0]['palm'], wrist=g.pts[0]['wrist'])


# ------------------------------------------------------------------------------------------ cache
def drawing(name, key):
    os.makedirs(CACHE, exist_ok=True); fn = os.path.join(CACHE, f'{VER}_{name}_{key}.npz')
    if os.path.exists(fn):
        z = np.load(fn); return {k: z[k] for k in z.files}
    D = make(name, key); tmp = fn[:-4] + '.%d.tmp.npz' % os.getpid(); np.savez(tmp, **D); os.replace(tmp, fn); return D


LK = {}                                                                                   # luffy drawings: key -> kwargs (filled by e2)


def make(name, key):
    if name == 'luffy': return draw_luffy(**LK[key])
    if key[0] == 'g':
        G = GRIP[name]; sp = SPEC[name]; E0 = np.array(G['h'], float) + (drawing(name, 'i0')['E'] - np.array(G['h'], float)) * 1.9; dv = np.array(G['h'], float) - E0; tight = key == 'g0'
        if name == 'chopper': return draw_chopper(0, spread=.1, bend=.1, h=G['h'], s=G['s'], clip_plane=False, dirv=dv, E0=E0)
        p = HD.lerp_pose(P['grip'], P['reach'], .0 if tight else .10); p = dict(p, wrist=(0, 0))
        if name == 'brook': return draw_brook(0, p, h=G['h'], s=G['s'], yaw=G['yaw'], pitch=G['pitch'], clip_plane=False, dirv=dv, E0=E0)
        if name == 'robin': return draw_hd(name, 0, HD.lerp_pose(P['reach'], P['grip'], .6), h=G['h'], s=G['s'], yaw=G['yaw'], pitch=G['pitch'], clip_plane=False, dirv=(0, 1), E0=(960, -60))
        return draw_hd(name, 0, p, h=G['h'], s=G['s'], yaw=G['yaw'], pitch=G['pitch'], clip_plane=False, dirv=dv, E0=E0, res=700)
    u, pose, dy, dr, dp, sd = keys_for(name)[key]
    if name == 'chopper':
        i = int(key[1:]); ph = 2 * math.pi * i / 8; return draw_chopper(u, spread=dict(e=(0, 0, .3, 1., .9, .6, .5), i=[.45 + .3 * math.sin(ph + j) for j in range(8)], a=(.8, 1.))[key[0]][i if key[0] != 'i' else 0], bend=dr * .05 if key[0] != 'i' else .16 * math.sin(ph))
    if name == 'brook': return draw_brook(u, pose, dy, dr, dp)
    return draw_hd(name, u, pose, dy, dr, dp, sd)
