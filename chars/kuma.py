"""Bartholomew Kuma (fear version): the bear-eared cap, small round glasses, the book, and the padded palm that fills the lens.
    paw(cx, cy, s, spread=1, roll=0, tilt=0, f=0)  -> layer (s = px per half palm)      bust(cx, cy, s, f, glint) -> layer
    pads(cx, cy, s) -> polygons of the paw print (main, 4 toes);   dent(img, cx, cy, s, amt) paw-shaped dent in the air"""
import numpy as np, math, cv2
from op import W, H
from op.paint import rgb, f32, grid, smooth
from op.ink import Cel, spl, xf
from op import kit
from op.kit import outline, hatch, shape

INK = '#0c0a10'
C = dict(skin='#8f9a5c', sh='#48573a', dk='#1d2820', hi='#d8dc94', pad='#e2766c', padS='#9c3a44', padD='#551d30', padH='#ffc0a8', coat='#141a1c', coatL='#2a3a34', cap='#e8e4c4', capS='#7f8a64', glass='#f6ffe0')
A = np.linspace(0, 2 * math.pi, 40, endpoint=False)


def pad_shapes():
    main = spl([(0, .02), (.34, .06), (.56, .26), (.62, .50), (.48, .74), (.24, .70), (0, .62), (-.24, .70), (-.48, .74), (-.62, .50), (-.56, .26), (-.34, .06)], True, n=6)
    toes = [np.c_[x + np.cos(A) * rx, y + np.sin(A) * ry] for x, y, rx, ry in ((-.66, -.20, .19, .24), (-.23, -.42, .21, .27), (.23, -.42, .21, .27), (.66, -.20, .19, .24))]
    return [main] + toes


def pads(cx, cy, s, roll=0.): return [xf(p, s, s, roll, cx, cy) for p in pad_shapes()]


def paw(cx, cy, s, spread=1., roll=0., f=0, lw=None, thrust=0.):
    """palm to lens, fingers fanned and foreshortened; thrust 0..1 tips the fingers back"""
    c = Cel(ss=2); w = lw or max(2.2, min(13., s * .024)); T = lambda P: xf(np.asarray(P, float), s, s, roll, cx, cy); k = 1 - .30 * thrust
    def finger(bx, by, ang, ln, r0, r1, seed, side=1):
        a = ang * spread; d = np.array([math.sin(a), -math.cos(a)]); n = np.array([-d[1], d[0]]); ln = ln * k; b = np.array([bx, by]); L = [b - n * r0, b + d * ln * .30 - n * r0 * 1.03, b + d * ln * .55 - n * r0 * .95, b + d * ln * .85 - n * r1 * 1.04, b + d * ln * 1.06 - n * r1 * .5]; R = [b + d * ln * 1.10, b + d * ln * 1.06 + n * r1 * .5, b + d * ln * .85 + n * r1 * 1.04, b + d * ln * .55 + n * r0 * .95, b + d * ln * .30 + n * r0 * 1.03, b + n * r0]; Q = T(spl(L + R, n=6)); c.fill(Q, C['skin']); c.clip(Q); sh = T(spl([b - n * r0 * 1.3, b + d * ln * .3 - n * r0 * .20, b + d * ln * .52 - n * r0 * .42, b + d * ln * .80 - n * r1 * .15, b + d * ln * 1.3 - n * r1 * .6, b + d * ln * 1.3 - n * r1 * 2, b - n * r0 * 2.4], n=5)); c.fill(sh, C['sh']); hatch(c, T([b - n * r0 * .7, b + d * ln - n * r0 * .6, b + d * ln + n * r0 * .1, b + n * r0 * .1]), a + roll + .5, max(5, s * .028), max(1.2, w * .28), INK, seed, length=s * .13, fade=.6)
        for q in (.36, .70):
            p0, p1 = T(b + d * ln * q - n * r0 * .8)[None], T(b + d * ln * q + n * r0 * .8)[None]; c.line(np.vstack([p0, (p0 + p1) / 2 + (T(d * .07)[None] - T((0, 0))[None]), p1]), w * .62, INK, taper=(.3, .3), seed=seed); c.line(np.vstack([p0, (p0 + p1) / 2 + (T(d * .12)[None] - T((0, 0))[None]), p1]), w * .34, INK, taper=(.4, .4), seed=seed + 1)
        c.unclip(); outline(c, Q, w, INK, seed, press=.5)
    finger(.95, .55, 1.02, 1.05, .30, .25, 5)
    for i, (bx, by, an, ln, r0) in enumerate(((-.80, -.40, -.30, 1.02, .25), (-.28, -.60, -.09, 1.22, .27), (.27, -.60, .09, 1.20, .27), (.78, -.40, .28, 1.0, .25))): finger(bx, by, an, ln, r0, r0 * .84, 11 + i)
    palm = T(spl([(-1.05, -.40), (-.80, -.62), (-.30, -.80), (.28, -.80), (.78, -.62), (1.06, -.30), (1.22, .30), (1.06, .82), (.74, 1.24), (.30, 1.42), (-.34, 1.42), (-.78, 1.18), (-1.04, .72), (-1.14, .15)], True, n=7)); c.fill(palm, C['skin']); c.clip(palm)
    c.fill(T(spl([(-1.3, -.6), (-.86, -.44), (-.78, .1), (-.86, .52), (-.50, .98), (-.1, 1.12), (.4, 1.16), (.2, 1.6), (-1.3, 1.6)], True, n=6)), C['sh']); c.fill(T(spl([(-1.3, .2), (-1.0, .3), (-.8, .84), (-.3, 1.3), (-.2, 1.6), (-1.4, 1.6)], True, n=6)), C['dk']); hatch(c, T([(-.9, -.4), (-.6, -.4), (-.55, .6), (-.2, 1.2), (-.8, 1.0)]), 1.0 + roll, max(5, s * .026), max(1.2, w * .28), INK, 7, length=s * .16, fade=.4); c.unclip(); outline(c, palm, w * 1.1, INK, 3, press=.5)
    for j, q in enumerate(pad_shapes()):                              # pads: domes with a hard shadow, a gloss, a lip line
        Q = T(q); m = Q.mean(0); c.fill(m + (Q - m) * 1.05 + np.array([-.02, .05]) * s, C['dk']); c.fill(Q, C['padS']); c.clip(Q); c.fill(m + (Q - m) * .86 + np.array([.07, -.07]) * s, C['pad']); c.fill(m + (Q - m) * .95 + np.array([-.30, .34]) * s * (1 if j == 0 else .45), C['padD'], .0 if j else .55)
        g = m + (Q - m) * np.array([.42, .22]) + np.array([.12, -.26 if j == 0 else -.10]) * s * (1 if j == 0 else .6); c.fill(g, C['padH']); hatch(c, m + (Q - m) * .9 + np.array([-.12, .10]) * s, .9 + roll, max(5, s * .024), max(1.2, w * .26), INK, 20 + j, length=s * .10, fade=.5, alpha=.8); c.unclip(); outline(c, Q, w * .95, INK, 30 + j, press=.55)
    return c


def bust(cx, cy, s, f=0, glint=0.):
    """colossal body in shadow, head small; s = px of head half width"""
    c = Cel(ss=2); w = max(2.4, s * .06); T = lambda P: np.asarray(P, float) * s + [cx, cy]
    body = T(spl([(-5.4, 9), (-5.2, 4.2), (-4.4, 2.6), (-2.4, 1.7), (-1.0, 1.2), (1.0, 1.2), (2.4, 1.7), (4.4, 2.6), (5.2, 4.2), (5.4, 9)], n=8)); shape(c, body, C['coat'], w, INK, 1); c.clip(body); c.fill(T(spl([(1.0, 1.2), (2.6, 1.9), (4.2, 2.9), (4.7, 5), (4.8, 9), (5.6, 9), (5.6, 3), (3.0, 1.2)], True, n=6)), C['coatL'])
    for k in range(8):                                                # paw prints on the coat
        x, y = -4.2 + 1.2 * k, 3.6 + 1.5 * (k % 3); [c.fill(xf(p, .42 * s, .42 * s, .2 * k, cx + x * s, cy + y * s), '#24332d') for p in pad_shapes()]
    c.unclip(); coll = T(spl([(-1.7, 1.9), (-1.35, .6), (-.6, 1.0), (0, 1.5), (.6, 1.0), (1.35, .6), (1.7, 1.9), (0, 3.4)], True, n=6)); shape(c, coll, '#1c2622', w, INK, 2)
    for sg in (-1, 1):
        e = T(np.c_[sg * .92 + np.cos(A) * .46, -1.18 + np.sin(A) * .46]); shape(c, e, C['capS'], w, INK, 4); c.fill(T(np.c_[sg * .92 + np.cos(A) * .24, -1.16 + np.sin(A) * .26]), C['dk'])
        c.fill(T(spl([(sg * .85, -.4), (sg * 1.12, -.1), (sg * 1.18, .5), (sg * .95, .9), (sg * .8, .5)], True, n=6)), INK)
    face = T(spl([(-.95, -.6), (-1.0, .1), (-.86, .72), (-.5, 1.18), (0, 1.30), (.5, 1.18), (.86, .72), (1.0, .1), (.95, -.6)], n=7)); shape(c, face, C['sh'], w, INK, 5); c.clip(face); c.fill(T(spl([(.30, -.7), (.42, -.1), (.34, .3), (.46, .7), (.2, 1.4), (1.3, 1.4), (1.3, -.7)], True, n=6)), C['skin']); c.fill(T([(-1.2, -.8), (1.2, -.8), (1.2, -.18), (0, -.05), (-1.2, -.18)]), C['dk'])
    c.line(T([(-.42, .80), (0, .76), (.42, .80)]), w * .9, INK); c.line(T([(-.06, .18), (.10, .46), (-.06, .50)]), w * .6, INK); [c.line(T([(x, .92), (x + .02, 1.08)]), w * .5, INK) for x in (-.3, -.1, .12, .3)]; c.unclip()
    cap = T(spl([(-1.06, -.42), (-1.10, -.92), (-.72, -1.34), (0, -1.50), (.72, -1.34), (1.10, -.92), (1.06, -.42), (0, -.28)], True, n=7)); c.fill(cap, C['capS']); c.clip(cap); c.fill(T(spl([(.1, -1.6), (.35, -1.0), (.3, -.2), (1.4, -.2), (1.4, -1.6)], True, n=6)), C['cap'])
    for x, y in ((-.55, -.85), (-.1, -1.15), (.45, -.8), (.75, -1.1), (-.8, -.55)): c.fill(T(np.c_[x + np.cos(A) * .11, y + np.sin(A) * .10]), C['dk'])
    c.unclip(); outline(c, cap, w, INK, 8)
    for sg in (-1, 1):
        g = T(np.c_[sg * .40 + np.cos(A) * .27, .04 + np.sin(A) * .27]); c.fill(g, C['glass']); c.clip(g); c.fill(T([(sg * .40 - .4, .34), (sg * .40 + .4, -.10), (sg * .40 + .4, .5), (sg * .40 - .4, .5)]), '#9fb46a'); c.unclip(); outline(c, g, w * 1.1, INK, 9)
    c.line(T([(-.14, .02), (0, -.03), (.14, .02)]), w, INK)
    bk = T(spl([(3.0, 4.6), (4.6, 4.2), (4.9, 6.4), (3.3, 6.8)], True, corners=(0, 1, 2, 3), n=3)); shape(c, bk, '#5a3a2c', w, INK, 12); c.line(T([(3.25, 4.8), (3.5, 6.6)]), w * .8, '#e8dcae')
    L = c.layer(); L = kit.rim(L, -max(3, s * .06), max(2, s * .03), '#dfe88a', .9, soft=1.2)
    if glint > 0:
        x, y = grid(); 
        for sg in (-1, 1): px, py = cx + sg * .40 * s, cy + .04 * s; g = np.exp(-np.abs(y - py) / (2 + 2 * glint)) * np.exp(-np.abs(x - px) / (s * 1.6 * glint + 4)) + np.exp(-((x - px) ** 2 + (y - py) ** 2) / (s * .30) ** 2); L[..., :3] = np.maximum(L[..., :3], 0) + (np.clip(g, 0, 1) * glint)[..., None] * rgb('#f6ffe0'); L[..., 3] = np.maximum(L[..., 3], np.clip(g * glint, 0, 1))
    return L


def dent(img, cx, cy, s, amt=1., edge=1.):
    """the air itself is pressed in the shape of a paw: refraction + pressed rim light"""
    h = np.zeros((H // 2, W // 2), f32)
    for p in pads(cx / 2, cy / 2, s / 2): cv2.fillPoly(h, [np.round(p * 16).astype(np.int32)], 1., cv2.LINE_AA, 4)
    hb = cv2.GaussianBlur(h, (0, 0), max(1.5, s * .035)); hb2 = cv2.GaussianBlur(h, (0, 0), max(3, s * .14)); hh = cv2.resize(hb * .6 + hb2 * .5, (W, H)); gx = cv2.Sobel(hh, cv2.CV_32F, 1, 0, ksize=5) / 48; gy = cv2.Sobel(hh, cv2.CV_32F, 0, 1, ksize=5) / 48; x, y = grid(); k = amt * s * s * .055
    o = cv2.remap(img.astype(f32), (x + gx * k).astype(f32), (y + gy * k).astype(f32), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101); d = cv2.resize(h, (W, H)); e = np.clip(cv2.GaussianBlur(d, (0, 0), 1.5) - cv2.GaussianBlur(kit.shifted(d, s * .03, s * .035), (0, 0), 2.5), 0, 1); e2 = np.clip(cv2.GaussianBlur(kit.shifted(d, s * .03, s * .035), (0, 0), 2) - d, 0, 1)
    o = o * (1 - .18 * amt * d[..., None]); o = o + (e * edge * .85)[..., None] * rgb('#f2ffd8'); return o * (1 - (e2 * .6 * amt)[..., None])
