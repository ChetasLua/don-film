"""Roronoa Zoro (fan art).  head(cx,cy,s,yaw,pitch,roll,expr,light,f,**kw) -> layer;  draw(cel, ...) -> Head for close-ups;  g1(f).
s = px from eye line to chin.  expr: calm glare bite grin laugh worry cry.  kw: young, bandana, swing (earrings, rad), look, lid, mouth, body('coat'|'tee'|None), wind, key"""
from .swkit import *
from op.head3 import Head

PAL = dict(base='#f0b27a', shade='#b4613a', deep='#652920', light='#ffd9a6', line=INK, term='#df7a3c', bounce='#d08a5c')
HAIR = dict(deep='#0c2a18', shade='#22622f', base='#55b046', light='#95de66', hot='#eaffb4', line='#17401f')
COAT = dict(base='#23523d', shade='#12302a', deep='#0b1d1c', light='#3f7a56', lin='#7a2430')


class OHead(Head):
    """head with a jaw that drops"""
    drop = 0.
    def P3(self, Q):
        Q = np.array(Q, float)
        if self.drop: Q[..., 1] = Q[..., 1] - self.drop * np.clip((-.46 - Q[..., 1]) / .54, 0, 1.25) ** 1.15 * np.clip((Q[..., 2] + .25) / .4, 0, 1)
        return Head.P3(self, Q)


def make(young=False):
    if young:
        a = [(-1.0, .13), (-.965, .24), (-.88, .39), (-.73, .525), (-.56, .615), (-.36, .67), (-.12, .70), (.12, .715), (.35, .72), (.6, .69), (.8, .55), (.93, .34), (1.0, .05)]
    else:
        a = [(-1.0, .135), (-.97, .245), (-.89, .385), (-.75, .515), (-.57, .605), (-.36, .655), (-.12, .685), (.12, .70), (.35, .70), (.6, .675), (.8, .55), (.93, .34), (1.0, .05)]
    return OHead(a=a, nose=(.11, -.06, -.40, -.47, .20), cheek=.06, socket=.10)


_H0 = {}
def _hl(th):
    a = np.abs(th); return np.interp(a, [0, .22, .42, .62, .85, 1.02, 1.16, 1.28, 1.40, 1.52, 1.62, 1.9, 2.4, 3.15], [.585, .60, .645, .655, .60, .43, .12, -.20, -.22, .16, .12, -.05, -.38, -.58])


def _flow(th, y, n, edge, r):
    m = len(th); a = np.abs(th); s = np.sign(th); d = np.zeros((m, 3)); lk = np.ones(m)
    top = smooth(.70, .92, y); front = 1 - smooth(.75, 1.15, a); back = smooth(1.9, 2.6, a); side = np.clip(1 - front - back, 0, 1)
    d += front[:, None] * (n * .62 + np.c_[s * .10 + r.normal(0, .22, m), .80 + r.normal(0, .18, m), .30 + r.normal(0, .12, m)])
    d += side[:, None] * (n * .70 + np.c_[s * .10, .22 + r.normal(0, .16, m), -.52 + r.normal(0, .16, m)])
    d += back[:, None] * (n * .85 + np.c_[r.normal(0, .12, m), -.10 + r.normal(0, .15, m), -.25 + 0 * a])
    d = d * (1 - top[:, None] * .7) + top[:, None] * .7 * (n * .9 + np.c_[r.normal(0, .2, m), .35 + 0 * a, -.10 + r.normal(0, .2, m)])
    sb = (a > 1.05) & (a < 1.5) & (y < .30); d[sb] = n[sb] * .22 + np.c_[s[sb] * .02, np.full(sb.sum(), -1.), np.full(sb.sum(), -.12)]; lk[sb] = .42
    fr = edge & (a < .95); dr = fr & (r.uniform(0, 1, m) < .42)                                # a few spikes hang over the brow
    d[dr] = n[dr] * .75 + np.c_[s[dr] * .12 + r.normal(0, .25, dr.sum()), -.45 + r.normal(0, .18, dr.sum()), np.full(dr.sum(), .35)]; lk[dr] = .74
    lk[(a > 1.0) & ~sb] *= .80; lk[back > .5] *= .85; return d, lk


def table(H_):
    if 'T' not in _H0: _H0['T'] = crop_set(H_, _hl, _flow)
    return _H0['T']


BROW = dict(glare=[(.072, -.006), (.17, .062), (.33, .158), (.52, .205), (.66, .168)], calm=[(.11, .085), (.25, .155), (.42, .200), (.61, .165)], worry=[(.11, .215), (.26, .215), (.42, .185), (.59, .105)],
            laugh=[(.12, .13), (.27, .215), (.44, .235), (.60, .155)], cry=[(.075, .015), (.20, .125), (.40, .165), (.59, .075)], grin=[(.10, .050), (.23, .145), (.41, .215), (.60, .18)], bite=[(.080, .0), (.19, .070), (.37, .152), (.58, .155), (.67, .125)])
EYE = dict(glare=([(.145, -.042), (.235, .010), (.37, .050), (.590, .058)], [(.565, .0), (.44, -.062), (.30, -.068), (.17, -.054)], (.345, -.006)),
           calm=([(.155, -.015), (.26, .065), (.40, .098), (.565, .055)], [(.55, .005), (.43, -.066), (.28, -.070), (.17, -.040)], (.345, .020)),
           worry=([(.155, -.005), (.25, .100), (.39, .142), (.56, .060)], [(.545, .010), (.43, -.085), (.28, -.092), (.17, -.045)], (.345, .022)),
           grin=([(.150, -.028), (.25, .050), (.39, .095), (.565, .070)], [(.55, .010), (.43, -.058), (.28, -.060), (.165, -.045)], (.335, .018)))
EYE['bite'] = EYE['glare']; EYE['cry'] = EYE['worry']


def box_mouth(p, o, w, ph=0., teeth=1., cx=0., top=-.555, tl=.075, grit=0.):
    """Oda's wobbling box mouth. o = open height, w = half width"""
    wb = lambda i, a=.011: a * math.sin(i * 2.3 + ph)
    up = [(cx - w, top - .075 + wb(0)), (cx - w * .86, top - .018 + wb(1)), (cx - w * .45, top + .012 + wb(2)), (cx, top - .004 + wb(3)), (cx + w * .45, top + .014 + wb(4)), (cx + w * .86, top - .020 + wb(5)), (cx + w, top - .080 + wb(6))]
    lo = [(cx + w, top - .080 + wb(6)), (cx + w * 1.0, top - .06 - o * .55 + wb(7)), (cx + w * .74, top - .015 - o + wb(8)), (cx + w * .30, top - .03 - o + wb(9, .015)), (cx, top - .012 - o + wb(10, .015)), (cx - w * .30, top - .03 - o + wb(11, .015)), (cx - w * .74, top - .015 - o + wb(12)), (cx - w * 1.0, top - .06 - o * .55 + wb(13)), (cx - w, top - .075 + wb(0))]
    tg = None if o < .13 or grit else [(cx - w * .50, top - o * .97), (cx - w * .42, top - o * .62), (cx - w * .12, top - o * .50), (cx, top - o * .55), (cx + w * .12, top - o * .50), (cx + w * .42, top - o * .62), (cx + w * .50, top - o * .97)]
    mouth_open(p, up, lo, tu=(.085 * teeth if not grit else o * .52 + .02), tl=(tl if not grit else o * .52 + .02), tongue=tg, lw=4.6, gum=.018 if o > .14 else None, seams=7 if grit else 0, tm=.94)


def mouth(p, kind, pal, k, prm=None):
    prm = prm or {}
    if kind == 'frown':
        p.poly([(-.14, -.695), (.0, -.712), (.15, -.690), (.10, -.748), (-.06, -.755)], pal['shade'])
        p.line([(-.285, -.712), (-.18, -.672), (-.02, -.655), (.14, -.662), (.25, -.700), (.295, -.745)], 4.6, taper=(.18, .22), press=.25, wfun=lambda t: .75 + .5 * np.sin(np.pi * t))
        p.line([(-.262, -.712), (-.238, -.694)], 3.2); p.line([(-.09, -.762), (.02, -.770), (.11, -.758)], 2.2, taper=(.4, .4)); p.line([(-.06, -.905), (.05, -.912)], 1.8, taper=(.4, .4))
        p.hatch((-.05, -.775), (.10, -.770), 5, (.02, -.045), 1.3)
    elif kind == 'calm':
        p.line([(-.20, -.672), (-.06, -.660), (.10, -.662), (.21, -.680)], 3.6, taper=(.2, .25)); p.line([(-.06, -.755), (.08, -.752)], 2., taper=(.4, .4)); p.poly([(-.10, -.69), (.12, -.688), (.06, -.742), (-.05, -.742)], pal['shade'])
    elif kind == 'worry':
        p.line([(-.17, -.700), (-.06, -.672), (.07, -.670), (.18, -.700)], 3.6, taper=(.2, .25)); p.line([(-.06, -.765), (.07, -.765)], 2., taper=(.4, .4))
    elif kind == 'grin':
        o = prm.get('o', .05); mouth_open(p, [(-.30, -.585), (-.16, -.625), (.0, -.640), (.18, -.622), (.335, -.565)], [(.335, -.565), (.22, -.685 - o), (.02, -.725 - o), (-.18, -.69 - o), (-.30, -.585)], tu=.11, tl=.0, lw=4.2, seams=8, tm=.7)
        p.line([(.335, -.54), (.36, -.575), (.345, -.61)], 2.4); p.line([(-.315, -.565), (-.325, -.60)], 2.2)
    elif kind == 'laugh':
        o = prm.get('o', .22); mouth_open(p, [(-.34, -.535), (-.18, -.575), (.0, -.588), (.18, -.575), (.34, -.535)], [(.34, -.535), (.27, -.66 - o * .55), (.0, -.70 - o), (-.27, -.66 - o * .55), (-.34, -.535)], tu=.085, tl=.035, tongue=[(-.17, -.71 - o), (-.15, -.64 - o * .6), (0, -.62 - o * .55), (.15, -.64 - o * .6), (.17, -.71 - o)], lw=4.2, tm=.7)
        p.line([(.345, -.50), (.375, -.545), (.355, -.59)], 2.4); p.line([(-.345, -.50), (-.375, -.545), (-.355, -.59)], 2.4)
    elif kind == 'bite':                                                     # lips stretched round the hilt: gums and clenched teeth show at both corners
        mouth_open(p, [(-.36, -.60), (-.22, -.575), (.0, -.572), (.22, -.575), (.37, -.61)], [(.37, -.61), (.31, -.735), (.0, -.765), (-.30, -.735), (-.36, -.60)], tu=.10, tl=.095, lw=4.4, gum=.02, seams=9, tm=.9)
        p.line([(.39, -.555), (.425, -.63), (.395, -.715)], 2.6); p.line([(-.385, -.545), (-.42, -.62), (-.385, -.70)], 2.4)
    elif kind == 'box': box_mouth(p, **prm)


def draw(cel, cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', key=-1, f=0, young=False, bandana=False, earrings=True, swing=0., look=(0, 0), lid=1., mouth_=None, body='coat', broll=0., wind=0., seed=1,
         pal=None, hairpal=None, L=None, eyes=True, drop=0., grav=None, tails=0., hilt_=None, sweat=False, brow=None, scar=True, noface=False, detail=True, kick=.92):
    pal = dict(PAL, **(pal or {})); hp = dict(HAIR, **(hairpal or {})); H_ = make(young); H_.drop = drop; H_.pose(cx, cy, s, yaw, pitch, roll); p = Pen(cel, H_, s, pal['line'], seed); k = -key; c = cel
    L = L or (key * .55, .62, .55); T = table(H_); near = 1 if yaw < -.02 else (-1 if yaw > .02 else 0); lw = p.w
    wd = (lambda T_: np.c_[np.sin(T_['u'] * 6.28 + f * .55) * .018 + wind * .05, np.cos(T_['u'] * 9 + f * .4) * .010, 0 * T_['u']] * (T_['ln'] / .25)[:, None]) if (f or wind) else None
    fac = (T['n'] @ H_.R.T)[:, 2]
    hem = lambda th: np.interp(np.abs(th), [0, .5, 1.0, 1.4, 1.9, 3.15], [.295, .330, .335, .245, .06, -.16])
    vis = np.ones(len(T['th']), bool) if not bandana else (T['y'] < hem(T['th']) - .02)
    if body: (body_coat if body == 'coat' else body_tee)(c, H_, cx, cy, s, pal, k, broll, yaw, seed, p, f)
    back = crop_draw(c, H_, T, (fac < .02) & vis & (fac > -.75), hp, L, lw=1.6, wind=wd, seed=seed); polys = []
    for q in union(back, 1., sig=.7, min_len=30): outline(c, q, lw(3.6), pal['line'], seed + 12, press=.4)
    c.fill(H_.skull(), hp['deep'])
    # ---- face flats
    F = H_.face(); c.fill(F, pal['base']); c.clip(F)
    side = [(.30, .84), (.335, .62), (.40, .42), (.515, .27), (.455, .15), (.60, .06), (.555, -.07), (.435, -.135), (.500, -.225), (.470, -.35), (.365, -.455), (.325, -.56), (.365, -.665), (.295, -.80), (.175, -.905), (.11, -1.03), (.6, -1.1), (1.0, -.8), (1.0, .85)]
    p.poly([((x - .020) * k, y - .004) for x, y in side], pal['term']); p.poly([(x * k, y) for x, y in side], pal['shade'])
    for sg in (1, -1):                                                       # brow shadow pools over both sockets
        q = [(.045, .075), (.13, .095), (.24, .145), (.41, .205), (.60, .185), (.585, .085), (.40, .080), (.26, .035), (.145, -.035), (.07, -.02)]
        p.poly([(x * sg, y) for x, y in q], pal['shade'])
    if expr in ('glare', 'bite', 'cry') or bandana: p.poly([(-.10, .05), (.10, .05), (.075, -.10), (.02, -.03), (-.02, -.03), (-.075, -.10)], pal['shade'])
    if bandana: p.poly([(-.75, .27), (.75, .27), (.75, .06), (.5, .085), (.28, .03), (.10, .0), (-.10, .0), (-.28, .03), (-.5, .085), (-.75, .06)], pal['shade'])
    p.poly([(-k * x, y) for x, y in [(.60, -.16), (.50, -.27), (.43, -.42), (.425, -.56), (.50, -.50), (.60, -.34)]], pal['shade'])    # hollow of the lit cheek
    p.poly([(k * x, y) for x, y in [(-.10, -.86), (.02, -.885), (.16, -.86), (.10, -.965), (-.06, -.975)]], pal['shade'])
    p.poly([(-k * x, y) for x, y in [(.10, .42), (.26, .47), (.44, .40), (.40, .30), (.22, .28)]], pal['light'], .34)
    p.poly([(-k * x, y) for x, y in [(.28, -.145), (.47, -.135), (.50, -.205), (.38, -.265), (.27, -.215)]], pal['light'], .85)
    p.poly([(-k * x, y) for x, y in [(.02, -.83), (.11, -.825), (.10, -.885), (.03, -.89)]], pal['light'], .7)
    # cast shadow of the fringe on the forehead
    if not bandana:
        e = T['edge'] & (np.abs(T['th']) < 1.1) & (fac > 0); off = A_(k * .030, -.075) * s; cr, sr = math.cos(roll), math.sin(roll); off = A_(off[0] * cr + off[1] * sr, off[0] * sr - off[1] * cr)
        tip = H_.P3(T['R'][e] + T['d'][e] * T['ln'][e][:, None]); a0 = H_.P3(T['R'][e] - T['e'][e] * T['wd'][e][:, None] * .6); b0 = H_.P3(T['R'][e] + T['e'][e] * T['wd'][e][:, None] * .6); up = H_.P3(T['R'][e] + A_(0, .16, -.02))
        for i in range(len(tip)): c.fill(np.array([up[i], a0[i], (a0[i] + b0[i]) / 2 * .35 + tip[i] * .65, b0[i]]) + off, pal['shade'])
    jn = H_.jaw(k); ji = np.r_[0, np.cumsum(np.hypot(*np.diff(jn, axis=0).T))]; inw = (H_.P([(0, -.45)])[0] - jn); inw /= np.hypot(*inw.T)[:, None]
    c.fill(np.vstack([jn, (jn + inw * s * np.array([.0, .03, .045, .05, .05, .04, .02, 0])[:, None])[::-1]]), pal['bounce'], .8)
    if kick: c.fill([F, F + A_(k * .032, .008) * s], '#fff1d4', kick, evenodd=True)
    c.unclip()
    outline(c, F, lw(5.4), pal['line'], seed, press=.42, breaks=2)
    if eyes and not noface:
        br = BROW.get(brow or expr, BROW['calm']); nose(p, pal, k, near or k, tip=.20, w=.135)
        E = EYE.get(expr, EYE['calm'])
        for sg in ((-1, 1) if young else (-1,)):
            if expr == 'laugh': p.line([(sg * .16, -.015), (sg * .27, .055), (sg * .41, .065), (sg * .56, -.02)], 6.5, taper=(.12, .16)); p.line([(sg * .58, .02), (sg * .64, .05)], 2); p.line([(sg * .22, -.085), (sg * .36, -.10)], 1.6)
            else: eye(p, sg, E[0], E[1], (E[2][0] + look[0] * sg, E[2][1] + look[1]), .050, shade='#a58a88' if expr in ('glare', 'bite') else '#c2adaa', lw=10., lid=.058, open_=lid, crease=[(.19, .055), (.34, .118), (.54, .115)] if expr not in ('glare', 'bite') else None)
            p.line([(sg * .50, -.085), (sg * .38, -.128), (sg * .26, -.125)], 1.9, taper=(.4, .5))
            if expr in ('glare', 'bite', 'cry'): p.line([(sg * .185, -.10), (sg * .27, -.165), (sg * .37, -.185)], 2.0, taper=(.2, .6))
        if not young:                                                        # the shut eye and the scar
            p.fill([(.20, -.03), (.36, .035), (.53, -.01), (.40, -.065)], pal['shade'])
            p.line([(.155, -.030), (.25, -.062), (.38, -.074), (.49, -.050), (.575, .005)], 6.6, taper=(.14, .2), wfun=lambda t: .7 + .4 * np.sin(np.pi * t))
            for x0, y0, dx, dy in ((.50, -.05, .03, -.028), (.44, -.066, .015, -.03), (.30, -.07, -.012, -.026)): p.line([(x0, y0), (x0 + dx, y0 + dy)], 2.2, taper=(.05, .6))
            p.line([(.185, .015), (.34, .032), (.53, .030)], 1.9, taper=(.3, .5)); p.line([(.25, -.115), (.38, -.135), (.50, -.10)], 1.8, taper=(.4, .5))
        for sg in (-1, 1):
            ribbon_fill(p, [(x * sg, y) for x, y in br], lambda t: .004 + .040 * math.sin(math.pi * min(1, t ** .8 * .86 + .10)) ** .9 * (1 - t ** 3), hp['deep'])
            p.line([(x * sg, y) for x, y in br], 2.2, taper=(.1, .3))
        if not young and scar:
            q = [(.405, .430), (.384, .20), (.362, -.04), (.346, -.22), (.330, -.400)]
            ribbon_fill(p, q, lambda t: .017 * math.sin(math.pi * np.clip(t, .02, .98)) ** .5, '#7c3426'); c.line(H_.P(np.array(q)), p.w(3.4), pal['line'], taper=(.3, .35), press=.25, seed=77); p.line([(.335, .150), (.395, .205)], 2.0, taper=(.2, .5)); p.line([(.320, -.215), (.375, -.185)], 1.8, taper=(.2, .5))
        if expr in ('glare', 'bite', 'cry'):
            p.line([(-.030, .055), (-.052, .12), (-.045, .185)], 2.3, taper=(.12, .6)); p.line([(.034, .050), (.056, .11), (.054, .165)], 2.1, taper=(.12, .6)); p.line([(-.075, .005), (-.04, -.02)], 2.); p.line([(.07, .005), (.04, -.02)], 1.8)
            p.line([(-.14, .27), (-.02, .255), (.10, .272)], 1.5, taper=(.5, .5), alpha=.8); p.line([(-.11, .012), (-.145, .075), (-.135, .13)], 2.2, taper=(.1, .6)); p.line([(.105, .01), (.135, .065)], 2.2, taper=(.1, .6)); p.line([(k * .555, -.20), (k * .48, -.29), (k * .44, -.36)], 2.0, taper=(.3, .5))
        kind = mouth_[0] if mouth_ else dict(glare='frown', calm='calm', bite='bite', grin='grin', laugh='laugh', worry='worry', cry='box').get(expr, 'calm')
        mouth(p, kind, pal, k, mouth_[1] if mouth_ else (dict(o=.20, w=.30) if kind == 'box' else None))
        if detail and s > 120:                                               # Oda pen clusters on the shadow edge
            ln = pal['line']
            p.hatch((k * .425, -.14), (k * .36, -.45), 7, (-k * .105, -.015), 1.6, ln, seed=3); p.hatch((k * .34, -.58), (k * .24, -.84), 6, (-k * .085, .02), 1.5, ln, seed=4)
            p.hatch((k * .40, .41), (k * .50, .28), 5, (-k * .085, -.03), 1.4, ln, seed=5); p.hatch((-k * .52, -.28), (-k * .435, -.53), 6, (k * .075, -.03), 1.5, ln, seed=6)
            p.hatch((-.20, -.105), (-.43, -.11), 5, (-.012, -.050), 1.3, ln, seed=7); p.hatch((k * .055, -.12), (k * .075, -.30), 4, (k * .045, -.012), 1.4, ln, seed=8)
            p.hatch((-.085, -.555), (.07, -.555), 4, (.0, -.030), 1.2, ln, seed=9)
        p.line([(.36 * -k, -.52), (.405 * -k, -.60), (.40 * -k, -.70)], 1.7, taper=(.3, .6), alpha=.9)
    # ---- near hair, ear, earrings
    nearmask = (fac >= .02) & vis; be = np.zeros_like(nearmask)
    if near: be = (np.sign(T['th']) == near) & (np.abs(T['th']) > 1.5) & (T['y'] < .30)
    if not bandana:
        for q in front_mass(H_, _hl): c.fill(q, hp['deep']); polys.append(q)
    polys += crop_draw(c, H_, T, nearmask & be, hp, L, lw=1.6, wind=wd, seed=seed)
    for sd in ((near,) if near and abs(yaw) > .12 else (-1, 1)):
        if not sd: continue
        e = H_.ear(sd); c.fill(e['outer'], pal['base'] if sd != k else pal['shade']); c.clip(e['outer']); c.fill(e['inner'], pal['shade'] if sd != k else pal['deep']); c.fill(e['tragus'], pal['deep']); c.unclip()
        outline(c, e['outer'], lw(4.0), pal['line'], seed + 7); p.line(e['inner'], 2.2, raw=True, taper=(.1, .3)); p.line(e['tragus'][:len(e['tragus']) * 2 // 3], 2.0, raw=True, taper=(.2, .4))
    polys += crop_draw(c, H_, T, nearmask & ~be, hp, L, lw=1.7, wind=wd, seed=seed)
    for q in union(polys, 1., sig=.7, min_len=30): outline(c, q, lw(4.0), pal['line'], seed + 11, press=.4)
    if bandana: _bandana(c, H_, hem, p, pal, k, f, tails, seed, near)
    if earrings and (near == 1 or abs(yaw) <= .02 or grav is not None): _earrings(c, H_, s, lw, swing, grav, roll if grav is None else 0.)
    if sweat: bead(c, *H_.P([(-.55, .30)])[0], s * .05, math.pi / 2)
    return H_


def _earrings(c, H_, s, lw, swing, grav, roll, lit='#fff6c0'):
    ex, t, b, z = H_.k['ear']; g = A_(*grav) if grav is not None else A_(0, 1.); k = s
    for i, (u, v) in enumerate(((.085, b + .035), (.045, b - .015), (.005, b + .015))):
        o = H_.P3(np.array([(ex + u * .95 + .01, v, z - u * 1.55 + .02)]))[0]; sw = swing * (1 + .25 * i) + .05 * (i - 1); d = A_(g[0] * math.cos(sw) - g[1] * math.sin(sw), g[0] * math.sin(sw) + g[1] * math.cos(sw)); m = A_(-d[1], d[0])
        a = np.linspace(0, 6.283, 20); rr = k * .030; c.line(o + np.c_[np.cos(a), np.sin(a)] * rr + d * rr * .6, lw(2.6), INK, closed=True); ln = k * (.315 + .02 * i); w = k * .040
        q0 = o + d * rr * 1.3; P = spl(np.array([q0 + m * w * .22, q0 + d * ln * .25 + m * w * .34, q0 + d * ln * .78 + m * w * .92, q0 + d * ln * .95 + m * w * .66, q0 + d * ln, q0 + d * ln * .95 - m * w * .66, q0 + d * ln * .78 - m * w * .92, q0 + d * ln * .25 - m * w * .34, q0 - m * w * .22]), True, (0, 8), n=6)
        c.fill(P, '#e8b22c'); c.clip(P); c.fill(P - m * w * .95, '#a4691a'); c.fill(P - m * w * 1.5, '#6a3c10'); c.unclip(); outline(c, P, lw(2.9), INK, 5 + i)
        c.line(np.array([q0 + d * ln * .30 + m * w * .10, q0 + d * ln * .60 + m * w * .34, q0 + d * ln * .86 + m * w * .42]), lw(3.2), lit, taper=(.4, .3))
        c.fill(np.array([o + d * rr * .4 + m * rr, o + d * rr * 1.8 + m * rr * .7, o + d * rr * 1.8 - m * rr * .7, o + d * rr * .4 - m * rr]), '#f7d769'); outline(c, np.array([o + d * rr * .4 + m * rr, o + d * rr * 1.8 + m * rr * .7, o + d * rr * 1.8 - m * rr * .7, o + d * rr * .4 - m * rr]), lw(1.6), INK, 2)


def _bandana(c, H_, hem, p, pal, k, f, tails, seed, near):
    P, M, th = cap(H_, hem, .075); s = H_.s; lw = p.w; C = dict(base='#1f2a28', shade='#0e1415', light='#41574e', hot='#7fa08a')
    sd = -1 if near == 1 else 1; kn = H_.P3(root3(H_, np.array([2.35 * (near or 1)]), np.array([.12]), .12)[0])[0] if True else None
    for j, (dx, dy, ln) in enumerate(((1., .16, 1.0), (.92, -.12, .80))):                     # tails stream behind
        sg = near or 1; t = np.linspace(0, 1, 12); wv = np.sin(t * 5.2 - f * .75 + j * 1.6) * (.03 + .12 * t) * s
        q = kn + np.c_[sg * dx * ln * s * t * (1.0 + tails * .3), dy * ln * s * t + wv - .06 * s * t]; B, l, r, cc = limb(q, np.array([.05, .09, .12, .13]) * s, np.array([.05, .08, .10, .11]) * s)
        cloth(c, B, C['base'], C['shade'], [np.vstack([cc, r[::-1]])], C['light'], [], w=lw(3.6), folds=[cc[len(cc) // 8:len(cc) * 3 // 4]], fw=lw(1.8), seed=j)
    c.fill(P, C['base']); c.clip(P); mid = lambda a, y, e=.08: H_.P3(root3(H_, np.array([a]), np.array([y]), e)[0])[0]
    nv = lambda a, y: (root3(H_, np.array(a), np.array(y))[1] @ H_.R.T)
    ps = [(k * .55, 1.02), (k * .62, .80), (k * .74, .60), (k * .70, .44), (k * .86, .30), (k * .90, .16)]
    sh = np.array([mid(a, y) for a, y in [(k * .62, hem(k * .62)), (k * .70, .46), (k * .50, .62), (k * .66, .80), (k * .30, .97)]] + [H_.P3(np.array([(k * 1.2, 1.2, 0.)]))[0], H_.P3(np.array([(k * 1.5, .0, 0.)]))[0]]); c.fill(sh, C['shade'])
    c.fill(np.array([mid(-k * a, y) for a, y in [(.25, .80), (.62, .74), (.74, .52), (.55, .42), (.30, .55)]]), C['light'])
    c.unclip(); outline(c, P, lw(5.0), INK, seed + 3)
    for j, (a0, y0, a1, y1) in enumerate(((1.25, .20, .20, .50), (1.35, .30, .45, .70), (1.15, .10, .05, .32), (1.5, .42, .8, .86), (1.28, .16, .60, .30))):
        sg = near or 1; t = np.linspace(0, 1, 10); L_ = np.array([mid(sg * (1.6 + (a1 - 1.6) * u if False else a0 + .35 + (a1 - a0 - .35) * u), y0 + (y1 - y0) * u + .05 * math.sin(u * 3.1)) for u in t]); c.line(L_, lw(2.4), '#05080a', taper=(.08, .7), press=.4, seed=j); c.line(L_[2:8] + [0, -lw(3)], lw(2.0), C['light'], taper=(.3, .7), seed=j)
    a = np.linspace(-1.3, 1.3, 40); hl = np.array([mid(q, float(hem(q)) + .012) for q in a]); c.line(hl, lw(2.4), C['hot'], taper=(.3, .3), press=.5)
    B = spl(np.array([kn + A_(-.10, -.08) * s, kn + A_(.02, -.12) * s, kn + A_(.11, -.04) * s, kn + A_(.09, .09) * s, kn + A_(-.04, .10) * s, kn + A_(-.11, .03) * s]), True, n=6)
    if near: cloth(c, B, C['light'], C['base'], [B + [0, .06 * s]], w=lw(3.4), seed=9)


def body_coat(c, H_, cx, cy, s, pal, k, broll, yaw, seed, p, f=0):
    """neck, trapezius, bare scarred chest, open dark green coat"""
    off = -yaw * .30; B = lambda P: xf(np.asarray(P, float) + [off, -.20], s, s, broll, cx, cy); lw = p.w; ln = pal['line']
    torso = spl(B([(-.50, .56), (-.515, .95), (-.57, 1.20), (-.80, 1.40), (-1.3, 1.56), (-2.1, 1.80), (-2.5, 2.6), (-2.5, 4.2), (2.5, 4.2), (2.5, 2.6), (2.1, 1.76), (1.3, 1.50), (.80, 1.34), (.56, 1.14), (.50, .90), (.48, .56)]), True, (6, 7, 8, 9), n=6)
    c.fill(torso, pal['base']); c.clip(torso)
    c.fill(spl(B([(k * .08, .6), (k * .02, 1.0), (k * .20, 1.30), (k * .02, 1.56), (k * .30, 1.66), (k * .55, 1.85), (k * .40, 2.2), (k * .62, 2.5), (k * .5, 3.4), (k * 2.6, 3.4), (k * 2.6, .5), (k * .5, .4)]), True, (0, 1, 3, 8, 9, 10, 11), n=6), pal['shade'])
    jaw_cast(c, H_, s, broll, k, pal)
    c.fill(spl(B([(-k * .10, 1.98), (-k * .50, 1.88), (-k * .62, 2.05), (-k * .40, 2.14)]), True, n=6), pal['light'], .6)
    c.unclip()
    for q, w in (([(-.46, .80), (-.30, 1.18), (-.10, 1.50), (-.03, 1.60)], 3.0), ([(.44, .70), (.34, 1.12), (.16, 1.46), (.08, 1.59)], 3.4), ([(-.06, 1.60), (-.30, 1.545), (-.62, 1.50), (-.95, 1.52)], 2.8), ([(.12, 1.60), (.36, 1.53), (.66, 1.47), (1.0, 1.48)], 2.8),
                 ([(.02, 1.10), (.06, 1.18), (.03, 1.28)], 2.2), ([(-.02, 1.66), (.0, 1.95), (-.03, 2.4), (.0, 3.0)], 2.4), ([(-.10, 2.75), (-.45, 2.66), (-.85, 2.40)], 2.6), ([(.12, 2.75), (.50, 2.64), (.9, 2.36)], 2.6), ([(-.36, 1.60), (-.42, 1.74)], 1.8), ([(.44, 1.58), (.50, 1.72)], 1.8)):
        c.line(spl(B(q), n=8), lw(w), ln, taper=(.12, .4), press=.35, seed=seed + int(w * 10))
    for i in range(7): c.line(B([(k * (.20 + .05 * i), 1.26 + .035 * i), (k * (.10 + .06 * i), 1.33 + .04 * i)]), lw(1.5), ln, taper=(.1, .6), seed=i)
    sc = spl(B([(.62, 1.42), (.30, 1.95), (-.12, 2.55), (-.55, 3.3)]), n=10); c.line(sc, lw(9), '#8a4434', taper=(.1, .1), press=.5); c.line(sc + [lw(3), 0], lw(2.2), ln, taper=(.1, .2))
    for j in range(5, 38, 5): d = A_(.11, .05) * s * (1 if j % 2 else .85); q = sc[min(j, len(sc) - 1)]; c.line(np.array([q - d, q + d]), lw(2.4), ln, taper=(.2, .3), seed=j)
    sw = math.sin(f * .42) * .02
    for sg in (-1, 1):                                                       # coat panels with a wide turned collar
        O = spl(B([(sg * .52, 1.08), (sg * .66, 1.06), (sg * .98, 1.28), (sg * 1.45, 1.43), (sg * 2.1, 1.62), (sg * 2.6, 2.1), (sg * 2.7, 4.2), (sg * (.42 + sw), 4.2), (sg * (.40 + sw), 3.0), (sg * .52, 2.2), (sg * .60, 1.62)]), True, (0, 1, 5, 6, 7), n=6)
        cloth(c, O, COAT['base'] if sg != k else COAT['shade'], COAT['shade'] if sg != k else COAT['deep'], [spl(B([(sg * 1.05, 1.5), (sg * 1.35, 2.0), (sg * 1.15, 2.6), (sg * 1.4, 4.2), (sg * .4, 4.2), (sg * .44, 2.6)]), True, (3, 4), n=6), spl(B([(sg * 1.9, 1.7), (sg * 2.0, 2.3), (sg * 1.75, 3.2), (sg * 2.8, 4.2), (sg * 2.8, 1.8)]), True, n=6)],
              COAT['light'] if sg != k else COAT['base'], [spl(B([(sg * 1.2, 1.38), (sg * 1.75, 1.58), (sg * 1.62, 1.74), (sg * 1.25, 1.56)]), True, n=6)], w=lw(5.0), folds=[B([(sg * 1.02, 1.45), (sg * 1.25, 1.95), (sg * 1.2, 2.7)]), B([(sg * 1.5, 1.62), (sg * 1.7, 2.2), (sg * 1.6, 3.0)]), B([(sg * .62, 1.75), (sg * .78, 2.3), (sg * .70, 2.9)])], fw=lw(2.4), seed=seed + sg)
        Lp = spl(B([(sg * .52, 1.08), (sg * .66, 1.06), (sg * .96, 1.27), (sg * 1.24, 1.66), (sg * 1.02, 1.84), (sg * 1.12, 2.02), (sg * .62, 2.75), (sg * .48, 2.2), (sg * .60, 1.60)]), True, (0, 1, 3, 4, 5, 6), n=6)
        cloth(c, Lp, COAT['light'] if sg != k else COAT['base'], COAT['base'] if sg != k else COAT['shade'], [spl(B([(sg * .66, 1.30), (sg * .86, 1.55), (sg * .80, 1.9), (sg * .70, 2.5), (sg * .4, 2.7), (sg * .5, 1.3)]), True, n=6)], w=lw(4.2), folds=[B([(sg * .74, 1.10), (sg * .90, 1.45), (sg * .92, 1.75)])], fw=lw(2.), seed=seed + sg + 5)
        for i in range(6): c.line(B([(sg * (.66 + .05 * i), 2.46 - .10 * i), (sg * (.74 + .055 * i), 2.34 - .10 * i)]), lw(1.6), '#07100f', taper=(.1, .6), seed=i)


def jaw_cast(c, H_, s, broll, k, pal, d=.30):
    """hard shadow the jaw throws on the neck"""
    jl, jr = H_.jaw(-1), H_.jaw(1); J = np.vstack([jl, jr[::-1][1:]]); n = len(J); u = np.linspace(-1, 1, n); sx, cx_ = math.sin(broll), math.cos(broll)
    dp = s * d * (.55 + .45 * (u * k * .5 + .5)) * (1 - .25 * np.abs(u)); D = J + np.c_[k * s * .07 * cx_ - dp * sx, dp * cx_ + k * s * .07 * sx]; c.fill(np.vstack([J, spl(D, n=5)[::-1]]), pal['deep'])


def body_tee(c, H_, cx, cy, s, pal, k, broll, yaw, seed, p, f=0):
    """young: white tee, round neck"""
    B = lambda P: xf(np.asarray(P, float) + [-yaw * .3, -.24], s, s, broll, cx, cy); lw = p.w; ln = pal['line']
    N = spl(B([(-.42, .72), (-.44, 1.05), (-.54, 1.28), (-1.0, 1.50), (-1.0, 2.2), (1.0, 2.2), (1.0, 1.48), (.54, 1.26), (.44, 1.0), (.42, .72)]), True, (4, 5), n=6); c.fill(N, pal['base']); c.clip(N)
    jaw_cast(c, H_, s, broll, k, dict(deep=pal['shade'])); c.fill(spl(B([(k * .12, .9), (k * .30, 1.3), (k * .10, 1.55), (k * .6, 1.9), (k * 1.1, 1.9), (k * 1.1, .8)]), True, n=5), pal['shade']); c.unclip()
    for q, w in (([(-.40, .78), (-.26, 1.18), (-.09, 1.52)], 2.8), ([(.40, .74), (.28, 1.16), (.10, 1.52)], 2.8), ([(-.10, 1.54), (-.36, 1.48), (-.66, 1.46)], 2.4), ([(.12, 1.54), (.38, 1.47), (.68, 1.44)], 2.4), ([(.0, 1.06), (.04, 1.14), (.0, 1.24)], 2.)): c.line(spl(B(q), n=8), lw(w), ln, taper=(.12, .4), press=.35, seed=seed + int(w * 10))
    T_ = spl(B([(-.66, 1.36), (-.50, 1.64), (-.20, 1.86), (.10, 1.90), (.42, 1.80), (.66, 1.34), (1.2, 1.46), (1.9, 1.66), (2.4, 2.3), (2.6, 3.0), (1.75, 3.2), (1.7, 5.6), (-1.7, 5.6), (-1.75, 3.2), (-2.6, 3.0), (-2.4, 2.3), (-1.9, 1.68), (-1.2, 1.48)]), True, (0, 5, 9, 10, 11, 12, 13, 14), n=6)
    cloth(c, T_, '#f6f1e6', '#b9b4c4', [spl(B([(k * .5, 1.9), (k * .75, 2.4), (k * .55, 3.0), (k * .8, 4.0), (k * .7, 5.6), (k * 2.7, 5.6), (k * 2.7, 1.6), (k * 1.2, 1.5)]), True, (4, 5, 6), n=6), spl(B([(-k * 1.6, 2.2), (-k * 1.35, 2.7), (-k * 1.5, 3.4), (-k * 1.9, 3.1)]), True, n=6), spl(B([(-k * .2, 1.95), (-k * .55, 2.35), (-k * .35, 2.9), (-k * .62, 3.6), (-k * .25, 3.2), (-k * .12, 2.5)]), True, (0, 3), n=6), spl(B([(k * .1, 2.0), (-k * .05, 2.6), (k * .12, 3.3), (k * .3, 3.0), (k * .22, 2.4)]), True, (2,), n=6)], w=lw(5.), folds=[B([(-1.55, 2.0), (-1.2, 2.3), (-.8, 2.4)]), B([(1.55, 2.0), (1.25, 2.4), (.9, 2.6)]), B([(-1.6, 2.7), (-1.1, 2.9), (-.5, 2.8)]), B([(.2, 1.92), (.3, 2.5), (.2, 3.2)]), B([(-.3, 1.9), (-.45, 2.4)]), B([(-.9, 1.7), (-.7, 2.1), (-.35, 2.35)]), B([(.95, 1.68), (.8, 2.05), (.5, 2.2)]), B([(-1.2, 3.2), (-.6, 3.35), (.1, 3.1)]), B([(1.3, 3.3), (.8, 3.1), (.45, 3.25)]), B([(-.1, 2.6), (-.5, 3.0), (-1.1, 3.0)])], fw=lw(2.2), seed=seed)
    rb = spl(B([(-.60, 1.40), (-.45, 1.60), (-.18, 1.78), (.10, 1.81), (.38, 1.72), (.60, 1.38)]), n=8); c.line(rb, lw(2.2), ln, taper=(.05, .05))


def head(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='day', f=0, **kw):
    x0, y0 = int(max(0, cx - 4.4 * s)), int(max(0, cy - 4.4 * s)); x1, y1 = int(min(W, cx + 4.4 * s)), int(min(H, cy + 4.4 * s))
    if x1 - x0 < 8 or y1 - y0 < 8: return np.zeros((H, W, 4), f32)
    kw = dict(kw); rims = kw.pop('rims', None); boost = kw.pop('boost', 1.); D = LIGHTS.get(light, LIGHTS['day']) if isinstance(light, str) else light; kw.setdefault('key', D['key']); kw.setdefault('broll', roll)
    c = Cel(x1 - x0, y1 - y0, x0, y0); draw(c, cx, cy, s, yaw, pitch, roll, expr, f=f, **kw); return light_layer(c.layer(), D, cx, cy, s, boost, rims)


def ring_pose(x, y, f, seed):
    """lean, bob and the laugh break at f20 -> (dx, dy, roll add, scale)"""
    d = f // 2 * 2; b = math.sin(d * .21 + seed) * 2.2
    if f < 17: return 0, b, .012 * math.sin(d * .17 + seed), 1.
    t = f - 17
    if t < 3: return 0, b - 3 * t, -.02 * t, 1 - .012 * t
    u = t - 3; k = 1 + .075 * math.exp(-u / 4.) * math.cos(u * .75); l = math.sin(u * math.pi / 2.5) * 7 * min(1, u / 4); return 0, b + l, .035 * math.sin(u * .31 + seed), k


def g1(f):
    x, y = 520, 860; ux, uy = (960 - x), (500 - y); n = math.hypot(ux, uy); ux, uy = ux / n, uy / n; roll = math.atan2(ux, -uy); dx, dy, dr, k = ring_pose(x, y, f, 1.3); lg = f >= 20
    o = .16 + .06 * math.sin((f // 2) * 1.25) if lg else 0
    D = dict(key=-1, rims=[((-ux * 10, -uy * 10), '#fff6d2', 1.)], g=(('#fffaf2', '#cfa9a0')), L=(-.5, .6, .55))
    return head(x - ux * 4 + dx - uy * dy * .4, y + dy, 158 * k, .10, -.10, roll + dr, 'laugh' if f >= 21 else ('worry' if f < 18 else 'grin'), D, f // 2, mouth_=('laugh', dict(o=o)) if f >= 21 else None, swing=.25 * math.sin(f * .35) * (2 if lg else 1), look=(.0, -.02), wind=.3)
