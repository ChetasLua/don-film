"""f2 / f4 props: rubber houses, torii, pine, barrel, lantern (local units, base at the origin, y up negative, 100 = one storey), keyed splines, sun disc, dust stars."""
import numpy as np, cv2, math
from scipy.interpolate import Akima1DInterpolator
from op import W, H as HH
from op.paint import rgb, f32, smooth, ramp, grid, fbm, mix
from op.ink import Cel, spl
from op.kit import outline, shape
from op.fx_core import h1, lin01, noise, c2
from chars.gear5 import inked, ss, lay, back_out, rim, hotrim

SUNP = dict(wall='#e3c2bf', wall1='#b58bab', roof='#5a5596', roof1='#3c3a74', rooft='#8f86c4', wood='#8c5a6a', red='#d5473a', red1='#972f4a', pine='#4c7a72', pine1='#34566a', straw='#dfb678', straw1='#a9797c', stone='#b3a0b8', stone1='#77688f', ink='#24152c', glow='#ffd77a')


class Keys:
    """{channel: {frame: value}} -> smooth (Akima) sampler, tuples allowed"""
    def __init__(self, tab):
        self.s = {}
        for k, d in tab.items():
            f = sorted(d); v = np.array([np.atleast_1d(d[q]) for q in f], float); f = [f[0] - 6, f[0] - 3] + f + [f[-1] + 4, f[-1] + 8]; v = np.vstack([v[:1], v[:1], v, v[-1:], v[-1:]]); self.s[k] = (Akima1DInterpolator(f, v), f[0], f[-1])

    def __call__(self, k, f):
        s, a, b = self.s[k]; v = s(min(max(f, a), b)); return float(v[0]) if len(v) == 1 else np.array(v)


class Rub:
    """rubber transform: squash sy, bend (tips lag sideways), rotation about (0, -piv)"""
    def __init__(self, x, y, s, rot=0., sq=1., bend=0., piv=60.): self.x, self.y, self.s, self.sq, self.bend, self.piv = x, y, s, sq, bend, piv; self.c, self.n = math.cos(rot), math.sin(rot)
    def __call__(self, P):
        P = np.asarray(P, float); h = -P[:, 1] / 100.; X = P[:, 0] / math.sqrt(self.sq) * (1 + .10 * (1 - self.sq) * np.sin(np.clip(h, 0, 2) * 1.5)) + self.bend * 26 * h * np.abs(h); Y = P[:, 1] * self.sq + self.piv
        return c2(self.x + (X * self.c - Y * self.n) * self.s, self.y - self.piv * self.s + (X * self.n + Y * self.c) * self.s)


def house(cel, T, p=SUNP, lw=2.4, seed=0, face=0., kind=0):
    """thatched / tiled town house with a face: windows are eyes, the door is a mouth (face 0..1 = how joyful)"""
    ink = p['ink']; w = 62; inked(cel, T([(-w, 0), (-w - 3, -50), (-w, -96), (w, -96), (w + 3, -50), (w, 0)]), p['wall'], lw, ink); cel.fill(T([(10, 0), (16, -96), (w, -96), (w, 0)]), p['wall1'])
    for x in (-w + 6, w - 6, 0): cel.fill(T([(x - 4, 0), (x - 4, -96), (x + 4, -96), (x + 4, 0)]), p['wood'])
    for x in (-33, 33):                                                                      # lit paper windows with lattice
        inked(cel, T([(x - 15, -46), (x - 15, -80), (x + 15, -80), (x + 15, -46)]), p['glow'], lw * .8, ink)
        for q in (-5, 5): cel.line(T([(x + q, -80), (x + q, -46)]), lw * .6, p['wood'], taper=(0, 0))
        cel.line(T([(x - 15, -63), (x + 15, -63)]), lw * .6, p['wood'], taper=(0, 0))
    inked(cel, T([(-15, 0), (-15, -62), (15, -62), (15, 0)]), '#4a2138', lw * .8, ink); cel.line(T([(0, -62), (0, 0)]), lw * .6, p['wood'], taper=(0, 0))
    for q in range(3): inked(cel, T([(-15 + 10 * q, -62), (-15 + 10 * q, -42 - 2 * (q % 2)), (-6 + 10 * q, -42 - 2 * ((q + 1) % 2)), (-5 + 10 * q, -62)]), p['red'], lw * .6, ink)
    if kind == 0:
        roof = np.vstack([spl([(-100, -82), (-74, -112), (-40, -150), (0, -172)], n=6), spl([(0, -172), (40, -150), (74, -112), (100, -82)], n=6), spl([(100, -82), (50, -92), (0, -94), (-50, -92), (-100, -82)], n=6)])
        inked(cel, T(roof), p['roof'], lw, ink); cel.fill(T(np.vstack([spl([(0, -172), (40, -150), (74, -112), (100, -82)], n=6), spl([(100, -82), (60, -92), (22, -100), (6, -130)], n=6)])), p['roof1'])
        for k in range(9): x = -80 + 20 * k; cel.line(T(spl([(x, -88 - 2 * abs(k - 4) * .0), (x * .55, -126 + abs(x) * .06), (x * .12, -160)], n=6)), lw * .55, ink, taper=(.1, .4), seed=k)
        inked(cel, T(spl([(-16, -168), (0, -184), (16, -168), (0, -162)], True, n=5)), p['roof1'], lw * .8, ink)
        for g in (-1, 1): inked(cel, T(spl([(g * 96, -84), (g * 110, -96), (g * 112, -112), (g * 100, -98)], True, n=5)), p['rooft'], lw * .7, ink)
    else:
        roof = np.vstack([spl([(-92, -78), (-70, -130), (-26, -176), (0, -184), (26, -176), (70, -130), (92, -78)], n=8), spl([(92, -78), (44, -92), (0, -96), (-44, -92), (-92, -78)], n=6)]); inked(cel, T(roof), p['straw'], lw, ink); cel.fill(T(np.vstack([spl([(20, -178), (60, -140), (92, -78)], n=6), spl([(92, -78), (54, -90), (30, -120)], n=6)])), p['straw1'])
        for k in range(16): x = -84 + 11 * k; cel.line(T([(x * .9, -92 + 3 * math.cos(k * 2.)), (x * .72, -118 - 10 * h1(k, seed))]), lw * .5, ink, taper=(.2, .6), seed=k)
        for g in (-1, 0, 1): inked(cel, T(spl([(g * 16 - 6, -188), (g * 16, -176), (g * 16 + 6, -188)], n=4)), p['wood'], lw * .6, ink)


def pagoda(cel, T, p=SUNP, lw=2.4, seed=0, face=0.):
    ink = p['ink']
    for i, (y, w, r) in enumerate(((0, 46, 84), (-92, 38, 70), (-176, 30, 56))):
        inked(cel, T([(-w, y), (-w, y - 58), (w, y - 58), (w, y)]), p['wall'], lw, ink); cel.fill(T([(8, y), (8, y - 58), (w, y - 58), (w, y)]), p['wall1'])
        for x in (-w + 8, 0, w - 8): cel.fill(T([(x - 3, y), (x - 3, y - 56), (x + 3, y - 56), (x + 3, y)]), p['red'])
        if i < 2: inked(cel, T([(-w * .55, y - 14), (-w * .55, y - 40), (w * .55, y - 40), (w * .55, y - 14)]), p['glow'], lw * .7, ink); [cel.line(T([(w * q, y - 40), (w * q, y - 14)]), lw * .6, p['wood'], taper=(0, 0)) for q in (-.28, 0, .28)]
        roof = np.vstack([spl([(-r, y - 50), (-r * .62, y - 66), (-w * .6, y - 98)], n=6), spl([(w * .6, y - 98), (r * .62, y - 66), (r, y - 50)], n=6), spl([(r, y - 50), (0, y - 60), (-r, y - 50)], n=6)]); inked(cel, T(roof), p['roof'], lw, ink); cel.fill(T(np.vstack([spl([(w * .6, y - 98), (r * .62, y - 66), (r, y - 50)], n=6), [(6, y - 60), (4, y - 96)]])), p['roof1'])
        for g in (-1, 1): inked(cel, T(spl([(g * (r - 4), y - 52), (g * (r + 10), y - 62), (g * (r + 8), y - 76), (g * (r - 4), y - 60)], True, n=4)), p['rooft'], lw * .6, ink)
    inked(cel, T([(-3, -270), (-2, -330), (2, -330), (3, -270)]), p['glow'], lw * .6, ink)
    for k in range(3): inked(cel, T(spl([(-9 + k, -284 - 14 * k), (0, -290 - 14 * k), (9 - k, -284 - 14 * k), (0, -279 - 14 * k)], True, n=4)), p['glow'], lw * .5, ink)


def torii(cel, T, p=SUNP, lw=2.6, seed=0, face=0.):
    ink = p['ink']
    for g in (-1, 1): inked(cel, T([(g * 58 - 9, 0), (g * 48 - 8, -150), (g * 48 + 8, -150), (g * 58 + 9, 0)]), p['red'] if g < 0 else p['red1'], lw, ink); inked(cel, T([(g * 58 - 12, 0), (g * 57 - 11, -20), (g * 57 + 11, -20), (g * 58 + 12, 0)]), p['roof1'], lw * .8, ink)
    inked(cel, T([(-74, -112), (-74, -126), (74, -126), (74, -112)]), p['red'], lw, ink); inked(cel, T([(-8, -126), (-8, -146), (8, -146), (8, -126)]), p['roof'], lw * .7, ink)
    inked(cel, T(np.vstack([spl([(-104, -172), (-50, -158), (0, -155), (50, -158), (104, -172)], n=6), spl([(98, -148), (50, -143), (0, -141), (-50, -143), (-98, -148)], n=6)])), p['red'], lw, ink)
    inked(cel, T(np.vstack([spl([(-110, -182), (-50, -167), (0, -164), (50, -167), (110, -182)], n=6), spl([(104, -170), (50, -157), (0, -154), (-50, -157), (-104, -170)], n=6)])), p['roof1'], lw * .9, ink)


def pine(cel, T, p=SUNP, lw=2.2, seed=0, face=0.):
    ink = p['ink']; inked(cel, T(np.vstack([spl([(-10, 0), (-6, -40), (-22, -80), (-4, -130)], n=6), spl([(8, -130), (-6, -84), (12, -40), (14, 0)], n=6)])), p['wood'], lw, ink)
    for x, y, r in ((-44, -86, 34), (40, -112, 38), (-6, -150, 44)):
        a = np.linspace(0, 6.283, 60); q = r * (1 + .13 * np.abs(np.sin(a * 4.5 + seed))); P = c2(x + np.cos(a) * q * 1.35, y + np.sin(a) * q * .55); inked(cel, T(P), p['pine'], lw, ink); cel.clip(T(P)); cel.fill(T(P + [9, 13]), p['pine1']); cel.unclip()
        for k in range(5): cel.line(T([(x - r + k * r * .5, y + 2), (x - r * .9 + k * r * .5, y - 9)]), lw * .5, ink, taper=(.2, .6), seed=k)


def barrel(cel, T, p=SUNP, lw=2.2, seed=0, face=0.):
    ink = p['ink']; P = np.vstack([spl([(-30, 0), (-38, -32), (-31, -64)], n=6), spl([(31, -64), (38, -32), (30, 0)], n=6)]); inked(cel, T(P), p['straw'], lw, ink); cel.clip(T(P)); cel.fill(T([(8, 0), (12, -70), (50, -70), (50, 0)]), p['straw1']); cel.unclip()
    a = np.linspace(0, 6.283, 26); inked(cel, T(c2(np.cos(a) * 31, -64 + np.sin(a) * 9)), p['wall'], lw * .8, ink)
    for y in (-16, -48): cel.line(T(spl([(-36, y), (0, y + 8), (36, y)], n=6)), lw * 1.3, p['wood'], taper=(.02, .02))
    inked(cel, T(c2(np.cos(a) * 15, -32 + np.sin(a) * 15)), p['wall'], lw * .6, ink); cel.fill(T(c2(np.cos(a) * 9, -32 + np.sin(a) * 9)), p['red'])


def lantern(cel, T, p=SUNP, lw=2.2, seed=0, face=0.):
    ink = p['ink']
    for P, c in (([(-28, 0), (-24, -16), (24, -16), (28, 0)], p['stone']), ([(-11, -14), (-9, -62), (9, -62), (11, -14)], p['stone']), ([(-26, -60), (-22, -72), (22, -72), (26, -60)], p['stone']), ([(-18, -70), (-18, -104), (18, -104), (18, -70)], p['stone']), ([(-42, -98), (-14, -126), (14, -126), (42, -98), (0, -106)], p['stone1'])): inked(cel, T(P), c, lw, ink)
    a = np.linspace(0, 6.283, 22); inked(cel, T(c2(np.cos(a) * 9, -87 + np.sin(a) * 10)), p['glow'], lw * .7, ink); inked(cel, T(c2(np.cos(a) * 7, -131 + np.sin(a) * 8)), p['stone'], lw * .7, ink)


PROPS = dict(house=house, hut=lambda c, T, p=SUNP, lw=2.4, seed=0, face=0.: house(c, T, p, lw, seed, face, 1), pagoda=pagoda, torii=torii, pine=pine, barrel=barrel, lantern=lantern)


def star(cx, cy, r, rot=0., k=.42, n=5):
    a = np.arange(n * 2) / (n * 2) * 6.283 + rot; q = np.where(np.arange(n * 2) % 2 == 0, r, r * k); return c2(cx + np.cos(a) * q, cy + np.sin(a) * q)
