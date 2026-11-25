"""Roger: sweeping black moustache, black hair, the greatest grin there ever was, red captain's coat on the shoulders."""
from ._old import *

PROPS = dict(a=[(-1.0, .22), (-.97, .33), (-.90, .44), (-.77, .535), (-.60, .60), (-.38, .64), (-.12, .672), (.12, .69), (.35, .70), (.6, .68), (.8, .55), (.93, .34), (1.0, .05)],
             p=2.2, nose=(.10, -.06, -.36, -.43, .17), cheek=.045, socket=.05, ear=(.69, .10, -.38, -.12))
KEY = ('cx', 'cy', 'uy', 'ly', 'drop', 'cheek', 'sq', 'lid', 'brow', 'ml', 'sh', 'gx', 'gy', 'asym', 'gum', 'frl')
EXPR = dict(
    calm=dict(cx=.31, cy=-.560, uy=-.595, ly=-.600, drop=0., cheek=.10, sq=.15, lid=.45, brow=0., ml=0., sh=1., gx=.0, gy=-.02, asym=.012, gum=0., frl=0.),
    smile=dict(cx=.385, cy=-.520, uy=-.590, ly=-.596, drop=0., cheek=.35, sq=.35, lid=.42, brow=.01, ml=.25, sh=1., gx=.0, gy=-.01, asym=.02, gum=0., frl=.05),
    press=dict(cx=.31, cy=-.555, uy=-.588, ly=-.593, drop=-.03, cheek=.12, sq=.55, lid=.22, brow=-.035, ml=-.22, sh=1.05, gx=.0, gy=-.02, asym=.0, gum=0., frl=-.05),
    grin=dict(cx=.500, cy=-.415, uy=-.550, ly=-.735, drop=.035, cheek=.75, sq=.55, lid=.50, brow=.025, ml=.62, sh=.45, gx=.004, gy=.0, asym=.018, gum=.10, frl=.45),
    load=dict(cx=.440, cy=-.462, uy=-.553, ly=-.715, drop=.025, cheek=.68, sq=.62, lid=.40, brow=.0, ml=.50, sh=.40, gx=.004, gy=.0, asym=.016, gum=.06, frl=.40),
    huge=dict(cx=.590, cy=-.335, uy=-.530, ly=-.878, drop=.110, cheek=1.0, sq=.72, lid=.62, brow=.030, ml=1.0, sh=.0, gx=.012, gy=.004, asym=.014, gum=.26, frl=1.),
    over=dict(cx=.605, cy=-.315, uy=-.522, ly=-.905, drop=.135, cheek=1.12, sq=.60, lid=.82, brow=.060, ml=1.18, sh=-.05, gx=.012, gy=.010, asym=.010, gum=.34, frl=1.15),
    laugh=dict(cx=.43, cy=-.43, uy=-.53, ly=-.96, drop=.16, cheek=1., sq=.8, lid=0., brow=.05, ml=.9, sh=0., gx=0., gy=0., asym=.0, gum=0., frl=.6))


def vec(e): return np.array([EXPR[e][q] for q in KEY]) if isinstance(e, str) else np.asarray(e, float)


def pose_at(f, keys, e=None):
    from op.comp import lerp_keys, ease
    return lerp_keys(f, [(q, vec(v)) for q, v in keys], e or ease)


def moustache(fx, p, wob=0., grow=1.):
    """[(side polygon, centre line fn)] in screen px"""
    out = []
    for sg in (-1, 1):
        ml = p['ml']; w = wob * sg
        C = np.array([(0, -.470), (.12, -.450), (.26, -.466), (.42, -.458 + .010 * ml), (.575, -.395 + .04 * ml), (.725, -.285 + .075 * ml), (.855 + .01 * ml, -.150 + .10 * ml + w * .4), (.940 + .02 * ml, .005 + .12 * ml + w), (.955 + .03 * ml + w, .135 + .13 * ml + w)])
        C[:, 0] *= 1 + .04 * ml; C[:, 1] += .018 * max(0, ml) * np.linspace(0, 1, len(C))
        h = np.array([.052, .080, .096, .096, .084, .064, .042, .020, .0]) * grow
        c = spl(C, n=12); uu = np.linspace(0, 1, len(c)); hh = np.interp(uu, np.linspace(0, 1, 9), h); hh = np.convolve(np.r_[[hh[0]] * 4, hh, [0] * 4], np.hanning(11)[1:-1] / np.hanning(11).sum(), 'valid'); T = np.gradient(c, axis=0); T /= np.hypot(*T.T)[:, None]; nn = np.c_[-T[:, 1], T[:, 0]]; zt = .24
        def at(v, c=c, hh=hh, nn=nn, uu=uu, sg=sg):
            q = c + nn * (hh * v)[:, None]; zf = fx.H.z(np.clip(q[:, 0], 0, .46) * sg, q[:, 1]) + .08 + .05 * uu; kz = smooth(.42, 1., uu); return fx.H.P3(np.c_[q[:, 0] * sg, q[:, 1], zf * (1 - kz) + zt * kz])
        out.append((np.vstack([at(1.), at(-1.)[::-1]]), at, sg))
    return out


def mane(cel, fx, pal, k, sway, seed=0, z=-.36, scale=1., lo=-62, hi=236, wind=.85, droop=.30, cols=None, hat=False):
    """wind-swept back hair: three layers of clumps around the skull"""
    r = np.random.default_rng(seed + 900); cols = cols or (pal['hair2'], pal['hair2'], pal['hair'])
    for ly, (n, L0, w0) in enumerate(((13, .74, .30), (15, .56, .25), (17, .40, .18))):
        ths = np.radians(np.linspace(lo, hi, n) + r.normal(0, 4.5, n))
        for i in r.permutation(n):
            th = ths[i]; c, s_ = math.cos(th), math.sin(th)
            if hat and s_ > .16: continue
            pt = A(.66 * c, .10 + .80 * s_); out = A(c, s_); tg = A(-s_, c)
            d = out * .62 + A(wind, 0.) * (.75 if c > -.3 else .30) + A(0, -droop); d /= np.hypot(*d); ln = L0 * scale * r.uniform(.65, 1.25) * (.52 + .48 * (c + 1) / 2 * 1.35); w = w0 * r.uniform(.8, 1.2); zz = z + .10 * ly
            q = fx.H.P3(np.array([(*(pt - tg * w / 2 - out * .12), zz + .06), (*(pt + d * ln), zz - .10), (*(pt + tg * w / 2 - out * .12), zz + .06)]))
            clump(cel, q[0], q[1] + sway(i + ly * 5, 1.3 + .9 * (c > 0)), q[2], pal, k, seed + 300 + ly * 40 + int(i), .15, r.choice((-1, 1)) * r.uniform(.05, .16) if c < .2 else -r.uniform(.04, .16), 5 - ly, (.75 if i % 3 == 0 else .45 if i % 3 == 1 else 0) if ly else 0, fill=cols[ly], w=2.4)
    for i in range(12):
        th = math.radians(r.uniform(lo, hi) if not hat else r.uniform(-60, 8)); c, s_ = math.cos(th), math.sin(th); pt = A(.70 * c, .10 + .82 * s_); d = A(c * .5 + wind * .8, s_ * .5 - droop); d /= np.hypot(*d); q = fx.H.P3(np.array([(*pt, z), (*(pt + d * .25 + A(0, r.uniform(-.06, .06))), z), (*(pt + d * r.uniform(.4, .8)), z)]))
        cel.line(spl([q[0], q[1], q[2] + sway(i, 2.2)], n=12), 4. * k, pal['hair'], taper=(.1, .8), seed=seed + 480 + i)


def draw(cx, cy, s, yaw=0., pitch=0., roll=0., expr='calm', light='gold', f=0, anchor=None, wind=(1., 0.), hat=False, rimdir=(1., .38), rim_amt=1., persp=.10, lw=1., seed=40, hair=True, dirty=True, gust=0., paraargs=None):
    """-> dict(back, face, stache, hair) lit premultiplied layers, and fx"""
    pal = LIGHT[light]; v = vec(expr); p = dict(zip(KEY, v)); laughing = (p['ly'] < -.93)
    fx = FX(PROPS, cx, cy, s, yaw, pitch, roll, p['drop'], .035 * p['cheek'], anchor, persp); k = fx.k * lw; P = fx.P; tw = f // 2 * 2
    sway = lambda i, a: A(a * (math.sin(tw * .42 + i * 1.7) + .5 * math.sin(tw * .9 + i * 2.9)) * wind[0] + gust * a * 1.6, a * .45 * math.sin(tw * .55 + i * 2.3)) * fx.s * .030
    # ------------------------------------------------ back: hair mass, neck
    B = Cel()
    if hair:
        core = fx.H.P3(np.array([(-.72, .8 if not hat else .42, -.3), (-.8, .3, -.3), (-.76, -.3, -.25), (-.62, -.72, -.15), (.56, -.76, -.15), (.76, -.3, -.35), (.80, .3, -.4), (.7, .85 if not hat else .42, -.3), (0, 1.05 if not hat else .42, -.1)]))
        B.fill(spl(core, True), pal['hair2'])
        mane(B, fx, pal, k, sway, seed, hat=hat)
    nk = fx.H.P3(np.array([(-.46, -.35, -.10), (-.43, -.8, -.05), (-.43, -1.2, -.05), (-.58, -1.6, -.05), (-1.0, -2.2, 0), (1.0, -2.2, 0), (.62, -1.6, -.05), (.45, -1.2, -.05), (.43, -.8, -.05), (.46, -.35, -.10)])); NK = spl(nk, True, (0, 4, 5, 9))
    B.fill(NK, pal['shade']); B.clip(NK)
    B.fill(fx.H.P3(np.array([(-.7, -.5, 0), (-.36, -1.22 - p['drop'], .1), (0, -1.36 - p['drop'], .2), (.34, -1.40 - p['drop'], .1), (.46, -1.75, .1), (.40, -2.4, 0), (1.4, -2.4, 0), (1.2, -.5, 0)])), pal['deep'])
    for i, (a, b) in enumerate((((-.34, -1.1), (-.12, -1.75)), ((.30, -1.1), (.14, -1.75)), ((-.05, -1.3), (-.03, -1.6)))):
        B.line(spl(fx.H.P3(np.array([(*a, .25), ((a[0] + b[0]) / 2 - .02, (a[1] + b[1]) / 2, .3), (*b, .25)])), n=8), 3.0 * k, pal['line'], taper=(.3, .6), seed=seed + 60 + i)
    edge_hatch(B, fx.H.P3(np.array([(-.36, -1.22 - p['drop'], .1), (0, -1.40 - p['drop'], .2), (.2, -1.5, .1), (.40, -1.75, .1)])), 16, 110 * k, 1.75, 2.4 * k, pal['line'], seed + 64, .6)
    B.unclip(); outline(B, NK, 4. * k, pal['line'], seed + 33)
    # ------------------------------------------------ face
    un = fx.J(fx.H.under()); B.fill(un, pal['deep'])
    C = Cel(); F = fx.face(); ears = [fx.H.ear(sg) for sg in (-1, 1)]
    for e in ears:
        if e['vis'] > 0:
            C.fill(e['outer'], pal['skin']); C.fill(e['inner'], pal['shade']); outline(C, e['outer'], 4.2 * k, pal['line'], seed + 1); C.line(e['inner'], 2.6 * k, pal['line'], seed=seed + 2); C.line(e['tragus'], 2.4 * k, pal['line'], seed=seed + 3)
    C.fill(F, pal['skin']); C.clip(F); ck = p['cheek']; sh = pal['shade']
    # shadow planes
    edge = np.array([(.16, .9), (.30, .52), (.27, .34), (.46, .20), (.545, .02), (.50, -.12), (.43, -.20 - .02 * ck), (.50, -.30), (.585, -.46), (.56, -.62), (.47, -.80), (.33, -.95), (.20, -1.08)])
    far = np.array([(.4, -1.4), (1.3, -1.4), (1.3, .95)]); C.fill(np.vstack([fx.J(fx.S(edge)), fx.H.P3(np.array([(.2, -1.5, .5), (1.5, -1.5, .2), (1.5, 1.2, .2)]))]), sh)
    for sg in (-1, 1):                                                                         # eye sockets
        C.fill(fx.S(mir([(.055, .10), (.12, .02), (.16, -.05), (.10, -.10), (.055, -.16), (.045, 0)], sg), .02, True), sh)
        C.fill(fx.S(mir([(.14, .13), (.32, .20 + p['brow']), (.52, .12), (.49, .02), (.32, .08 + .05 * p['lid']), (.17, .02)], sg), 0, True), sh)
    C.fill(fx.J(fx.S([(-.20, p['ly'] - .035), (0, p['ly'] - .055), (.22, p['ly'] - .03), (.17, p['ly'] - .085), (0, p['ly'] - .105), (-.15, p['ly'] - .085)], 0, True)), sh)      # under lip
    C.fill(fx.J(fx.S([(-.46, -.83), (-.30, -.965), (-.05, -1.0), (.2, -1.02), (.2, -1.3), (-.6, -1.2)], 0, True)), sh)
    # lit plane on the sun side
    lit = [(-.40, .95), (-.50, .55), (-.47, .34), (-.545, .18), (-.53, -.02), (-.475, -.14), (-.505, -.28), (-.57, -.40), (-.555, -.56), (-.49, -.72), (-.40, -.86), (-.27, -.97)]
    C.fill(np.vstack([fx.J(fx.S(lit)), fx.H.P3(np.array([(-.2, -1.5, .5), (-1.5, -1.5, .2), (-1.5, 1.2, .2)]))]), pal['lite'])
    # cheek folds
    cxx, cyy = p['cx'], p['cy']
    for sg in (-1, 1):
        fold = [(.140, -.335), (.215, -.375), (.30, -.42), (cxx + .030, cyy + .06), (cxx + .075, cyy - .05), (cxx + .07, cyy - .17), (cxx + .0, cyy - .30 - .05 * ck)]
        C.fill(fx.S(mir([(cxx + .004, cyy + .10), (cxx + .052, cyy + .04), (cxx + .085, cyy - .06), (cxx + .078, cyy - .18), (cxx + .01, cyy - .31 - .05 * ck), (cxx + .030, cyy - .19), (cxx + .038, cyy - .07), (cxx + .022, cyy + .02)], sg), 0, True, (0, 4)), sh)
        if ck > .3:                                                                           # under the cheek ball
            C.fill(fx.S(mir([(.150, -.33), (.24, -.315 + .02 * ck), (.38, -.33 + .03 * ck), (.50, -.295 + .03 * ck), (.50, -.36), (.36, -.43), (.24, -.42)], sg), 0, True, (0,)), sh)
    C.unclip()
    # cast shadow of the moustache
    MS = moustache(fx, p, wob=.012 * math.sin(tw * .5) * wind[0], grow=1.)
    C.clip(F)
    for pol, at, sg in MS: C.fill(pol + A(10, 22) * k, sh)
    C.unclip()
    # cheek/fold lines
    for sg in (-1, 1):
        fold = [(.140, -.325), (.20, -.35), (.28, -.41)]; out = [(cxx + .004, cyy + .10), (cxx + .052, cyy + .04), (cxx + .085, cyy - .06), (cxx + .078, cyy - .18), (cxx + .01, cyy - .31 - .05 * ck)]
        C.line(fx.S(mir(fold, sg)), 5.0 * k, pal['line'], taper=(.25, .3), seed=seed + 70 + sg); C.line(fx.S(mir(out, sg)), 5.0 * k, pal['line'], taper=(.35, .5), seed=seed + 71 + sg)
        if ck > .35:
            for j, (o, u0, u1) in enumerate(((.040, 1, 4), (.076, 1, 3))):
                g = np.array(out)[u0:u1 + 1] + A(o, .012); C.line(fx.S(mir(g, sg), 0), (3.4 - j * .8) * k * min(1, ck), pal['line'], taper=(.4, .6), seed=seed + 74 + sg + j * 3)
            C.line(fx.S(mir([(.175, -.285), (.27, -.275 + .02 * ck), (.39, -.285 + .03 * ck), (.50, -.255 + .03 * ck), (.565, -.20)], sg)), 4.2 * k, pal['line'], taper=(.5, .5), seed=seed + 80 + sg)
            C.line(fx.S(mir([(.21, -.172), (.33, -.160), (.45, -.172), (.54, -.225)], sg)), 3.0 * k, pal['line'], taper=(.5, .5), seed=seed + 82 + sg)
    # nose
    N = fx.N
    C.fill(spl(N([(.045, .06, .05), (.05, -.12, .10), (.055, -.27, .14), (.03, -.35, .165), (.085, -.375, .10), (.155, -.37, .05), (.135, -.30, .06), (.105, -.18, .05), (.085, .0, .03)]), True), sh)
    C.fill(spl(N([(-.085, -.365, .10), (0, -.385, .155), (.085, -.365, .10), (.13, -.40, .05), (.06, -.425, .07), (0, -.43, .08), (-.06, -.425, .07), (-.125, -.40, .05)]), True), sh)
    for sg in (-1, 1):
        C.fill(spl(N(mir([(.030, -.405, .10), (.075, -.395, .085), (.095, -.410, .065), (.06, -.422, .08)], sg) * [1, 1, 1]), True), pal['line'])
        C.line(spl(N(mir([(.088, -.285, .085), (.135, -.315, .065), (.150, -.365, .050), (.128, -.412, .040), (.095, -.418, .06)], sg))), 4.4 * k, pal['line'], taper=(.5, .25), seed=seed + 90 + sg)
    C.line(spl(N([(.075, .07, .03), (.058, -.04, .085), (.062, -.18, .12), (.070, -.27, .145), (.060, -.33, .165)])), 4.2 * k, pal['line'], taper=(.5, .25), seed=seed + 95)
    C.line(spl(N([(-.055, -.335, .15), (-.035, -.375, .165), (0, -.39, .165), (.04, -.375, .165), (.060, -.33, .165)])), 4.4 * k, pal['line'], taper=(.3, .2), seed=seed + 96)
    C.fill(spl(N([(-.052, .0, .05), (-.070, -.14, .08), (-.082, -.26, .105), (-.064, -.315, .13), (-.046, -.27, .15), (-.048, -.14, .11), (-.044, -.04, .075)]), True, (0, 3)), pal['lite']); C.fill(spl(N([(-.058, -.05, .06), (-.072, -.17, .09), (-.076, -.25, .11), (-.062, -.20, .12), (-.060, -.12, .09)]), True, (0, 2)), pal['rim'])
    for i, (a, b) in enumerate((((-.05, .115), (.00, .045)), ((.055, .12), (.035, .05)), ((-.02, .15), (-.005, .10)))):
        C.line(fx.S([a, ((a[0] + b[0]) / 2 + .008, (a[1] + b[1]) / 2), b], .05), 2.8 * k, pal['line'], taper=(.3, .6), seed=seed + 97 + i)
    # eyes
    for sg in (-1, 1):
        if laughing: arc_eye(C, fx, sg, pal, y=-.01, seed=seed + 100)
        else:
            eye(C, fx, sg, pal, p['lid'], p['sq'], p['gx'] + (.006 if sg < 0 else -.006), p['gy'] - .012, seed=seed + 100, w=.172, h=.100, slant=.02, r=.030, lw=1.15)
            for i, (a, b) in enumerate((((.515, .015), (.60, .055)), ((.52, -.025), (.62, -.025)), ((.505, -.062), (.59, -.10)))):
                C.line(fx.S(mir([a, ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + .006), b], sg)), 2.8 * k, pal['line'], taper=(.2, .65), seed=seed + 110 + i)
        b = p['brow']; brow(C, fx, sg, pal, [(.085, .060 + b * .3), (.085, .150 + b * .5), (.33, .262 + b * 1.2), (.64, .160 + b * .9), (.34, .165 + b)], seed=seed + 120)
        C.line(fx.S(mir([(.10, .33 + b), (.28, .375 + b * 1.4), (.45, .34 + b)], sg)), 2.8 * k, pal['line'], taper=(.5, .6), seed=seed + 126)
    # mouth
    if laughing:
        laugh(C, fx, p['cx'], p['cy'], p['uy'], p['ly'], pal, seed + 130, low=True)
    else:
        MP = grin(C, fx, p['cx'], p['cy'], p['uy'], p['ly'], pal, p['asym'], p['gum'], seed=seed + 130)
    C.line(fx.J(fx.S([(-.12, -.925), (0, -.955), (.13, -.925)])), 3.2 * k, pal['line'], taper=(.45, .45), seed=seed + 140)
    C.line(fx.J(fx.S([(.01, -.965), (.016, -1.0)])), 3.0 * k, pal['line'], taper=(.3, .6), seed=seed + 141)
    for sg in (-1, 1): C.line(fx.J(fx.S(mir([(.225, -.885), (.205, -.935), (.215, -.975)], sg))), 2.8 * k, pal['line'], taper=(.5, .5), seed=seed + 142 + sg)
    # pen hatching on the shadow edges
    h0 = fx.P([(.2, 0)])[0]; ha = 2.55
    C.clip(F)
    edge_hatch(C, fx.S(edge[1:4]), 7, 60 * k, ha, 2.4 * k, pal['line'], seed + 150)
    edge_hatch(C, fx.S(edge[6:9]), 8, 64 * k, ha, 2.4 * k, pal['line'], seed + 151)
    edge_hatch(C, fx.J(fx.S(edge[9:12])), 8, 66 * k, ha + .15, 2.4 * k, pal['line'], seed + 152)
    edge_hatch(C, fx.S([(-.50, -.24), (-.38, -.22), (-.26, -.25)]), 6, 44 * k, 1.9, 2.0 * k, pal['line'], seed + 153)
    edge_hatch(C, fx.S([(-.58, -.30), (-.60, -.42), (-.57, -.55)]), 7, 52 * k, .35, 2.0 * k, pal['line'], seed + 154)
    edge_hatch(C, spl(N([(.06, -.10, .1), (.06, -.2, .12), (.065, -.28, .14)])), 5, 34 * k, 2.7, 2.0 * k, pal['line'], seed + 155)
    edge_hatch(C, fx.S([(.18, -.33), (.30, -.36), (.44, -.33)]), 8, 50 * k, 1.2, 2.0 * k, pal['line'], seed + 156); edge_hatch(C, fx.S([(-.44, -.34), (-.32, -.37), (-.20, -.36)]), 7, 46 * k, 1.9, 2.0 * k, pal['line'], seed + 157)
    edge_hatch(C, fx.J(fx.S([(.12, -.90), (.24, -.86), (.36, -.80)])), 7, 60 * k, 2.2, 2.2 * k, pal['line'], seed + 158); edge_hatch(C, fx.S([(.18, .30), (.28, .36), (.34, .50)]), 8, 60 * k, 2.6, 2.2 * k, pal['line'], seed + 159)
    edge_hatch(C, fx.S([(-.16, .02), (-.10, -.04), (-.075, -.14)]), 6, 40 * k, .6, 2.0 * k, pal['line'], seed + 161); edge_hatch(C, fx.J(fx.S([(-.40, -.90), (-.28, -.98), (-.14, -1.0)])), 8, 50 * k, -1.2, 2.2 * k, pal['line'], seed + 162)
    C.unclip()
    outline(C, F, 6.2 * k, pal['line'], seed + 160, tremor=.3)
    face = C.layer()
    # ------------------------------------------------ fringe, side hair (drawn first: it casts the shadow over the eyes)
    Hc = Cel()
    if hair:
        fl = p['frl'] * .13; fr = p['frl'] * .18; top = 1.0 if not hat else .62
        L = [((-.66, .80, .10), (-.70, .06, .20), (-.52, .95, .3), .16, -.10), ((-.72, .52, -.02), (-.745, -.16, .02), (-.66, .75, .1), .14, -.06),
             ((-.60, top, .40), (-.50 + fr * .6, -.10 + fl, .66), (-.22, top + .02, .6), .20, -.10), ((-.42, top, .5), (-.24 + fr, -.02 + fl * .9, .82), (-.04, top, .66), .20, -.12), ((-.22, top, .6), (-.02 + fr, -.16 + fl * 1.1, .86), (.14, top, .7), .20, -.14), ((.0, top, .62), (.24 + fr, .0 + fl, .84), (.32, top, .62), .19, -.16),
             ((.16, top, .62), (.44 + fr, -.10 + fl * 1.1, .74), (.50, top, .52), .19, -.16), ((.36, top, .52), (.70 + fr * .8, -.06 + fl, .52), (.64, top - .05, .4), .18, -.16), ((.54, .84, .3), (.94 + fr, -.12 + fl, .20), (.72, .60, .2), .18, -.16), ((.62, .70, .12), (1.02 + fr * .6, -.36 + fl * .6, .0), (.75, .34, .05), .2, -.10),
             ((-.33, top, .55), (-.38 + fr, .02 + fl, .78), (-.25, top, .58), .1, .05), ((-.12, top, .66), (-.14 + fr, .0 + fl, .86), (-.04, top, .66), .1, -.06), ((.10, top, .66), (.13 + fr, -.08 + fl, .88), (.18, top, .66), .1, -.1), ((.30, top, .6), (.36 + fr, .06 + fl, .78), (.37, top, .56), .09, -.1)]
        for i, (a, t, b, bu, be) in enumerate(L):
            q = fx.H.P3(np.array([a, t, b])); clump(Hc, q[0], q[1] + sway(i + 20, 1.6 if i > 1 else .7), q[2], pal, k, seed + 200 + i, bu, be, 6 if i < 10 else 1, (.8 if i % 2 == 0 else .45) if i < 10 else 0, w=2.6)
    hl_ = Hc.layer()
    off = (.045 + .135 * max(p['sh'], -.2)) * fx.s; m = cv2.GaussianBlur(cv2.warpAffine(hl_[..., 3], np.float32([[1, 0, off * .30], [0, 1, off]]), (W, HH), borderMode=cv2.BORDER_REPLICATE), (0, 0), .8) * np.clip(face[..., 3], 0, 1) * (.80 if light == 'gold' else .6)
    if hat: m = m * 0
    face[..., :3] = face[..., :3] * (1 - m[..., None]) + (face[..., :3] * rgb('#63403c') * .72) * m[..., None]
    if dirty:
        r = np.random.default_rng(seed + 7); st = []
        for (x, y, n, ln) in ((-.40, -.30, 5, .13), (.36, -.28, 5, .12), (-.30, .30, 4, .14), (.20, .42, 4, .16), (-.36, -.86, 4, .12), (.33, -.90, 3, .1)):
            for _ in range(n):
                a = 1.9 + r.normal(0, .35); q = A(x + r.normal(0, .07), y + r.normal(0, .05)); d = A(math.cos(a), math.sin(a)) * ln * r.uniform(.4, 1); st.append((fx.J(fx.P([q - d / 2, q, q + d / 2])), r.uniform(6, 22) * k))
        D = inside(dirt(HH, W, st, pal['deep'], seed, .30), np.where(True, face, face)); tm = Cel(); face = over_l(face, D * .55)
    # ------------------------------------------------ moustache
    Mc = Cel()
    for pol, at, sg in MS:
        Mc.fill(pol, pal['hair']); n = len(at(0.))
        for i, v_ in enumerate((-.75, -.45, -.15, .2, .5, .78)):
            q = at(v_); a0 = int(n * (.04 + .03 * (i % 3))); a1 = int(n * (.62 + .05 * ((i * 2) % 5))); Mc.line(q[a0:a1], 2.2 * k, pal['hl'], taper=(.3, .6), press=.35, seed=seed + 170 + i)
        a, b = at(.62), at(.18); i0, i1 = int(n * .18), int(n * .62); t = np.sin(np.linspace(0, math.pi, i1 - i0))[:, None]; m = (a + b)[i0:i1] / 2; Mc.fill(np.vstack([m + (a[i0:i1] - m) * t, (m + (b[i0:i1] - m) * t)[::-1]]), pal['hi'])
        for j, (u, ln, si) in enumerate(((.24, .15, -1), (.40, .21, -1), (.54, .18, -1), (.50, .14, 1))):        # loose points
            i0 = int(n * u); q0, q1, c0 = at(si * .55)[max(0, i0 - 9)], at(si * .55)[min(n - 1, i0 + 9)], at(0.)[min(n - 1, i0 + 22)]
            e = at(si * 1.0)[i0]; dirn = (at(si)[min(n - 1, i0 + 10)] - e); dirn = dirn / (np.hypot(*dirn) + 1e-6); nrm = (e - at(0.)[i0]); nrm /= np.hypot(*nrm) + 1e-6
            clump(Mc, q0, e + dirn * ln * fx.s * .75 + nrm * fx.s * .022 + sway(j, .25), q1, pal, k, seed + 180 + j, .1, 0, 1, 0, w=2.0)
        outline(Mc, pol, 3.0 * k, pal['line'], seed + 190 + sg)
    mq = Cel(ss=1)
    if not laughing and p['uy'] - p['ly'] > .03: mq.fill(MP, '#ffffff')
    LY = dict(mm=mq.layer()[..., 3], back=B.layer(), face=face, stache=Mc.layer(), hair=hl_, fx=fx, pal=pal, pose=p)
    dx, dy = rimdir
    for n_ in ('back', 'face'): LY[n_] = rim(LY[n_], -dx * 9 * k, 1.5 * k, pal['cool'], .55, 3. * k)
    for n_, a in (('back', .8), ('face', 1.), ('stache', 1.), ('hair', 1.)): LY[n_] = rimlit(LY[n_], dx, dy, pal, k, a * rim_amt, wide=24 if n_ != 'stache' else 10, hot=9 if n_ != 'stache' else 5)
    return LY


def flat(LY, order=('back', 'face', 'hair', 'stache')):
    o = np.zeros((HH, W, 4), f32)
    for n in order: o = over_l(o, LY[n])
    return o


def head(cx, cy, s, yaw=0, pitch=0, roll=0, expr='calm', light='gold', f=0, **kw):
    return flat(draw(cx, cy, s, yaw, pitch, roll, expr, light, f, **kw))
