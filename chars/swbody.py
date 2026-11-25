"""SWORDS lane: full figures for c1 (Zoro's crouch), c4 (Sanji, hands in pockets), small fists, boots, shoes."""
from .swkit import *
from . import zoro, sanji

def fist(c, x, y, r, ang, pal, w=3., flip=1, seed=0):
    """closed hand round a hilt, knuckles to the viewer; ang = direction of the hilt (thumb side)"""
    R_ = lambda P: xf(np.asarray(P, float) * [1, flip] * r, 1, 1, ang, x, y)
    B = spl(R_([(-.95, -.55), (-.35, -.95), (.45, -.92), (.98, -.50), (1.0, .30), (.70, .85), (-.55, .90), (-1.0, .45)]), True, n=6); cloth(c, B, pal['base'], pal['shade'], [spl(R_([(-1.2, .15), (-.3, .30), (.5, .22), (1.2, .0), (1.2, 1.2), (-1.2, 1.2)]), True, n=5)], pal['light'], [spl(R_([(-.5, -.75), (.4, -.75), (.5, -.45), (-.5, -.45)]), True, n=5)], w=w, seed=seed)
    for k in range(3): u = -.45 + .46 * k; c.line(spl(R_([(u, -.86 + .05 * abs(k - 1)), (u + .05, -.2), (u - .02, .62)]), n=6), w * .7, pal['line'], taper=(.1, .35), seed=seed + k)
    T = spl(R_([(.98, -.45), (.75, -.15), (.15, .0), (-.10, .20), (.05, .42), (.65, .34), (1.05, .15)]), True, n=6); cloth(c, T, pal['base'], pal['shade'], [spl(R_([(.9, .05), (.2, .18), (0, .42), (.8, .4)]), True, n=5)], w=w * .9, seed=seed + 5)


def spin_fig(c, yaw, phi, f=0, K=1.3, seed=3, lw=5., wind=1.):
    """Sanji turning on one foot: yaw = where he faces (0 lens, + viewer's right), phi = where the kicking leg points on the ground circle (0 right, pi/2 at the lens).
    returns (hip px, foot px, depth)"""
    W2 = lambda P: A_(960 + (P[0] - 960) * K, 950 + (P[1] - 930) * K); cy_, sy_ = math.cos(yaw), math.sin(yaw); hw = math.hypot(78 * cy_, 44 * sy_); C_ = dict(b='#24232e', d='#100f17', l='#4b4a5c'); S = sanji.SUIT
    ux, uz = math.cos(phi), math.sin(phi); lean = -20 * ux; per = 1 / (1 - .42 * uz); hip0 = A_(960 + 30 * ux, 664 + 10 * uz); L = 318.; kn = hip0 + A_(ux * L * .46, uz * 62 - 26) * per ** .5; an = hip0 + A_(ux * L * per ** .8, uz * 128 * per ** .7 - 10)
    hip, knee, ank = W2(hip0), W2(kn), W2(an); w0, w1 = 44 * K, (26 + 0) * K * per ** 1.25
    def kick():
        B, a, b, cc = limb([hip, knee, ank], [w0, w0 * .95, (w0 + w1) * .5, w1, w1], [w0, w0 * .9, (w0 + w1) * .48, w1 * .95, w1]); lo = a if a.mean(0)[1] > b.mean(0)[1] else b; n = len(cc); i0 = int(n * .55); cloth(c, B, C_['b'], C_['d'], [np.vstack([cc, lo[::-1]])], C_['l'], [], w=lw, folds=[np.array([knee + [-18 * K, -24 * K], knee + [14 * K, 16 * K]])], fw=lw * .5)
        c.fill(np.vstack([a[i0:], b[i0:][::-1]]), '#f0641a', .92); c.fill(np.vstack([(a * .75 + b * .25)[i0 + n // 9:], (a * .3 + b * .7)[i0 + n // 9:][::-1]]), '#ffc23a'); c.fill(np.vstack([(a * .6 + b * .4)[i0 + n // 5:], (a * .45 + b * .55)[i0 + n // 5:][::-1]]), '#fff6cc'); outline(c, B, lw, INK, seed)
        if uz > .55: sole(c, ank[0] + 16 * ux * K, ank[1] - 6 * K, 66 * K * per, .35 * ux, 1., lw)
        else: shoe(c, ank, ank + A_(ux * 56 * per + 10, -46 * per + 10 * uz) * K, 24 * K * per, lw, seed, True)
    def leg():
        B, a, b, cc = limb([W2((952, 662)), W2((944, 790)), W2((950, 900))], np.array([44, 42, 34, 32, 28, 27]) * K, np.array([42, 40, 34, 32, 28, 27]) * K); shoe(c, W2((950, 906)), W2((950 + 62 * sy_ + (8 if cy_ > 0 else -8), 934 - 8 * (cy_ < 0))), 23 * K, lw, seed); cloth(c, B, C_['b'], C_['d'], [np.vstack([cc, b[::-1]])], C_['l'], [], w=lw, folds=[np.array([W2((920, 788)), W2((966, 800))])], fw=lw * .5)
    def arm(sg, depth):
        ex = sg * (hw + 34) if abs(cy_) > .45 else -sy_ * 46 + sg * 8; sh = W2((960 + lean + sg * hw * .92, 466)); e = W2((960 + lean * .6 + ex, 566)); p_ = W2((960 + sg * hw * .62, 650)); B, a, b, cc = limb([sh, (sh + e) / 2, e, (e + p_) / 2, p_], np.array([31, 29, 27, 26, 23, 21]) * K, np.array([30, 28, 26, 25, 23, 21]) * K); cloth(c, B, C_['b'] if depth > 0 else C_['d'], C_['d'], [np.vstack([cc, (a if a.mean(0)[1] > b.mean(0)[1] else b)[::-1]])], C_['l'], [], w=lw, folds=[np.array([e + [-20 * K, -6 * K], e + [12 * K, 8 * K]])], fw=lw * .5)
    if uz < 0: kick()
    arm(-1 if sy_ > 0 else 1, -1); leg(); tw = 34 * wind
    T = spl(np.array([W2((960 + lean + x, y)) for x, y in ((-18, 428), (-hw * .95, 446), (-hw * 1.05, 486), (-hw * .72, 600), (-hw * .80, 668), (-hw * 1.0 + tw, 742), (tw * .7, 716), (hw * 1.0 + tw, 748), (hw * .80, 668), (hw * .72, 600), (hw * 1.05, 486), (hw * .95, 446), (18, 428))]), True, (0, 5, 6, 7, 12), n=6)
    cloth(c, T, C_['b'], C_['d'], [spl(np.array([W2((960 + lean + x, y)) for x, y in ((hw * .15, 430), (hw * .4, 540), (hw * .1, 640), (hw * .9 + tw, 760), (hw * 1.3, 600), (hw * 1.2, 430))]), True, (0, 3), n=5)], C_['l'], [spl(np.array([W2((960 + lean + x, y)) for x, y in ((-hw * .85, 452), (-hw * .3, 446), (-hw * .35, 470), (-hw * .8, 484))]), True, n=5)], w=lw * 1.1, folds=[np.array([W2((960 + lean - hw * .7, 612)), W2((960 + lean - 6, 630))]), np.array([W2((960 + lean + hw * .7, 604)), W2((960 + lean + 10, 634))])], fw=lw * .5)
    vx = 960 + lean + sy_ * hw * .72; q = max(.22, abs(cy_))
    if cy_ > -.25:
        V = [W2((vx - 22 * q, 428)), W2((vx + 2, 540)), W2((vx + 22 * q, 428))]; cloth(c, np.array(V), S['shirt'], S['shirtd'], [np.array([W2((vx, 428)), V[1], V[2]])], w=lw * .7); c.fill(np.array([W2((vx - 6 * q, 438)), W2((vx + 6 * q, 438)), W2((vx + 6 * q, 500)), W2((vx, 530))]), '#14131b')
        for sg in (-1, 1): cloth(c, np.array([W2((vx + sg * 22 * q, 420)), W2((vx + sg * 50 * q, 470)), W2((vx + sg * 34 * q, 482)), W2((vx + sg * 44 * q, 496)), W2((vx + 2, 552))]), C_['l'] if sg < 0 else C_['b'], w=lw * .7)
    else:
        c.line(np.array([W2((960 + lean - sy_ * hw * .3, 470)), W2((960 + lean - sy_ * hw * .34, 600)), W2((960 + lean + tw * .7, 716))]), lw * .7, '#05050a', taper=(.1, .1)); cloth(c, np.array([W2((960 + lean - 26, 420)), W2((960 + lean + 26, 420)), W2((960 + lean + 30, 440)), W2((960 + lean - 30, 440))]), S['shirt'], w=lw * .7)
    arm(1 if sy_ > 0 else -1, 1) if abs(cy_) < .45 else (arm(1, 1), arm(-1, 1))
    hx, hy = W2((960 + lean * 1.5, 352)); N, *_ = limb([W2((960 + lean, 440)), (hx, hy + 30 * K)], [17 * K, 17 * K], [17 * K, 17 * K]); cloth(c, N, sanji.PAL['shade'], w=lw * .7)
    if cy_ > -.12: sanji.draw(c, hx, hy, 64 * K, float(np.clip(math.atan2(sy_, cy_), -1.35, 1.35)), -.08, -.10 * ux, 'glare', key=-1, f=f, body=False, wind=-.7 * wind, smoke=False, seed=5, cig=abs(yaw) < .8)
    else: sanji.back_head(c, hx, hy, 64 * K, yaw, -.10 * ux, f, -.8 * wind)
    if uz >= 0: kick()
    return hip, ank, uz


def boot(c, P, base, shade, light, w, sole='#17100c'):
    P = np.asarray(P, float); B = spl(P, True, (0, 1, len(P) - 1, len(P) - 2), n=6); cloth(c, B, base, shade, [spl(np.vstack([P[len(P) // 2:], P[:1]]), True, n=5) * .5 + B.mean(0) * .5 + [0, 8]], light, [], w=w)


def zoro_crouch(c, f, dx=0., dy=0., glint=None, k=1., face=True, wave=1., seed=2):
    """low guard after the cut: three blades out, coat thrown to the right (1920x1080 layout)"""
    S = lambda P: np.asarray(P, float) + [dx, dy]; C_ = zoro.COAT; pal = zoro.PAL; d = f // 2; wv = lambda i, a=14.: a * wave * math.sin(d * .9 + i * 1.3)
    T = spl(S([(905, 650), (1085, 645), (1190, 700), (1330, 706 + wv(0, 8)), (1490, 648 + wv(1)), (1660, 560 + wv(2, 22)), (1610, 640 + wv(2, 18)), (1720, 634 + wv(3, 24)), (1590, 722 + wv(3)), (1640, 790 + wv(4, 22)), (1450, 812 + wv(4)), (1290, 862 + wv(5, 10)), (1120, 905), (1030, 860), (920, 800)]), True, (5, 6, 7, 8, 9), n=7)
    cloth(c, T, C_['base'], C_['shade'], [spl(S([(1060, 700), (1250, 770), (1430, 730), (1600, 700 + wv(3)), (1640, 800 + wv(4)), (1300, 900), (1000, 880)]), True, (4,), n=6)], C_['light'], [spl(S([(1230, 700), (1400, 690 + wv(0, 8)), (1620, 590 + wv(2, 18)), (1430, 668 + wv(1)), (1300, 690)]), True, (2,), n=6)], w=5.5, folds=[S([(1080, 690), (1260, 745), (1480, 700 + wv(1)), (1600, 640 + wv(2))]), S([(1080, 760), (1280, 800), (1500, 770 + wv(4))]), S([(1180, 850), (1320, 830), (1440, 800 + wv(4))])], fw=2.6)
    lin = spl(S([(1640, 790 + wv(4, 22)), (1520, 800 + wv(4)), (1560, 760 + wv(4))]), True, (0,), n=5); c.fill(lin, C_['lin'])
    # legs
    L, a, b, cc = limb(S([(1035, 835), (1150, 845), (1225, 862), (1320, 905), (1408, 945)]), [78, 72, 62, 54, 44, 38], [66, 66, 60, 50, 42, 36]); cloth(c, L, '#2a3330', '#151b1a', [np.vstack([cc, (a if a.mean(0)[1] > b.mean(0)[1] else b)[::-1]])], w=5., folds=[S([(1180, 800), (1215, 850), (1205, 905)]), S([(1240, 822), (1262, 868)]), S([(1090, 800), (1130, 850)])], fw=2.4)
    cloth(c, spl(S([(1372, 905), (1436, 912), (1470, 950), (1520, 972), (1524, 992), (1370, 992), (1352, 950)]), True, (0, 1, 4, 5), n=6), '#3a2c22', '#1c140f', [S([(1352, 950), (1420, 962), (1524, 985), (1524, 1000), (1350, 1000)])], '#7a5c40', [S([(1400, 916), (1430, 916), (1456, 946), (1420, 940)])], w=5.)
    L, a, b, cc = limb(S([(958, 830), (860, 792), (765, 772), (735, 850), (722, 945)]), [70, 82, 78, 62, 52, 44], [78, 86, 66, 56, 48, 42]); cloth(c, L, '#2a3330', '#151b1a', [np.vstack([cc, b[::-1]]) if b.mean(0)[0] > a.mean(0)[0] else np.vstack([cc, a[::-1]])], '#51615a', [spl(S([(860, 735), (780, 712), (716, 740), (780, 748)]), True, n=6)], w=5.5, folds=[S([(800, 700), (772, 760), (790, 822)]), S([(838, 728), (820, 790)]), S([(700, 810), (750, 826), (782, 812)]), S([(900, 760), (880, 820)])], fw=2.6)
    cloth(c, spl(S([(676, 905), (772, 905), (786, 958), (780, 992), (612, 994), (590, 972), (650, 948)]), True, (0, 1, 3, 4), n=6), '#3a2c22', '#1c140f', [S([(590, 972), (690, 962), (786, 952), (786, 1000), (590, 1000)])], '#7a5c40', [S([(690, 912), (740, 912), (736, 940), (690, 944)])], w=5.5)
    # scabbards at the right hip (screen left), torso
    for j, kd in enumerate(('enma', 'kitetsu', 'wado')): scabbard(c, S([(1010 + 8 * j, 838 + 10 * j)])[0], S([(1330 + 30 * j, 770 + 46 * j)])[0], 25, kd, 3.6)
    Tz = spl(S([(820, 640), (900, 615), (990, 615), (1090, 640), (1110, 720), (1062, 840), (912, 845), (850, 730)]), True, n=7); cloth(c, Tz, pal['base'], pal['shade'], [spl(S([(960, 610), (985, 680), (955, 730), (1000, 770), (985, 850), (1120, 850), (1130, 640)]), True, (0, 1, 2, 3, 4, 5, 6), n=5)], pal['light'], [spl(S([(860, 668), (925, 660), (930, 700), (880, 712)]), True, n=6)], w=4., folds=[S([(950, 668), (944, 740), (955, 800)]), S([(862, 728), (905, 742), (944, 728)]), S([(958, 728), (1010, 742), (1066, 722)]), S([(900, 640), (940, 655)]), S([(1030, 640), (985, 655)]), S([(925, 770), (945, 776)]), S([(962, 776), (990, 770)])], fw=2.6)
    sc = spl(S([(1062, 660), (1000, 715), (930, 775), (880, 838)]), n=8); c.line(sc, 8, '#8a4434', taper=(.1, .1), press=.4); c.line(sc + [2, 1], 2.4, INK, taper=(.1, .1))
    for j in range(3, 24, 4): q = sc[j]; c.line(np.array([q - [9, 7], q + [9, 7]]), 2.4, INK, taper=(.3, .3))
    cloth(c, spl(S([(898, 790), (1066, 780), (1082, 850), (1050, 880), (930, 882), (890, 842)]), True, n=6), '#8e1f2a', '#4e0d18', [S([(980, 780), (1090, 780), (1090, 890), (960, 890)])], '#c9404a', [S([(905, 800), (960, 796), (958, 812), (905, 816)])], w=4.2, folds=[S([(900, 815), (980, 806), (1070, 808)]), S([(900, 842), (980, 836), (1074, 836)])], fw=2.2)
    for j, q in enumerate(([(1050, 860), (1120, 900 + wv(6)), (1150, 985 + wv(7)), (1100, 960), (1030, 880)], [(1020, 870), (1060, 930 + wv(7)), (1070, 1005 + wv(8)), (1030, 985), (1000, 880)])): cloth(c, spl(S(q), True, (2,), n=6), '#8e1f2a', '#4e0d18', [S(q)[[1, 2, 3]]], w=4.)
    # coat fronts
    for q, sh in (([(905, 628), (820, 622), (772, 664), (800, 745), (868, 838), (915, 842), (892, 780), (884, 700)], 0), ([(985, 622), (1080, 618), (1135, 655), (1128, 742), (1092, 842), (1052, 836), (1060, 770), (1040, 690)], 1)):
        cloth(c, spl(S(q), True, (0, 4, 5), n=6), C_['shade'] if sh else C_['base'], C_['deep'] if sh else C_['shade'], [S([(q[6][0], q[6][1]), (q[4][0], q[4][1] - 60), (q[4][0], q[4][1]), (q[5][0], q[5][1])])], C_['light'], [] if sh else [S([(840, 634), (900, 640), (880, 680), (830, 664)])], w=5., folds=[S([(q[2][0], q[2][1] + 6), ((q[2][0] + q[6][0]) / 2, 730), (q[4][0], q[4][1] - 10)])], fw=2.4)
    # arms: rolled coat sleeves, bare forearms
    def arm(sh, el, wr, sg, gd, tp, kd, ed, w0, w1, gl):
        sh, el, wr = S([sh])[0], S([el])[0], S([wr])[0]; gd, tp = S([gd])[0], S([tp])[0]; u = (gd - tp) / np.hypot(*(gd - tp)); pm = gd + u * 78
        katana(c, gd, tp, w0, w1, kd, .035, ed, gl, 3.4); hilt(c, gd, pm, 27, kd, 3., asp=.36, n=6, guard=1.0)
        F, a, b, cc = limb([el, el * .6 + wr * .4, wr], [36, 43, 38, 28, 22], [34, 36, 30, 24, 21]); lo = a if a.mean(0)[1] > b.mean(0)[1] else b; cloth(c, F, pal['base'], pal['shade'], [np.vstack([cc * .7 + lo * .3, lo[::-1]])], pal['light'], [], w=5., folds=[cc[len(cc) // 5:len(cc) * 3 // 5] - (lo - cc)[len(cc) // 5:len(cc) * 3 // 5] * .35], fw=2.2)
        U, a, b, cc = limb([sh, (sh + el) / 2 + [0, -6], el], [56, 60, 54, 50], [54, 56, 52, 48]); lo = a if a.mean(0)[1] > b.mean(0)[1] else b; e = el - sh; e /= np.hypot(*e); m = A_(-e[1], e[0]); cloth(c, U, C_['base'], C_['shade'], [np.vstack([cc, lo[::-1]])], C_['light'], [], w=5.2, folds=[np.array([el - e * 60 + m * 40, el - e * 40, el - e * 52 - m * 36]), np.array([el - e * 92 + m * 30, el - e * 78 - m * 10])], fw=2.6)
        K = spl(np.array([el - e * 20 + m * 54, el + e * 14 + m * 50, el + e * 16 - m * 48, el - e * 18 - m * 52]), True, n=6); cloth(c, K, C_['light'], C_['base'], [np.array([el - e * 20, el + e * 16, el + e * 16 + m * 50 * (1 if (m @ A_(0, 1)) > 0 else -1), el - e * 20 + m * 54 * (1 if (m @ A_(0, 1)) > 0 else -1)])], w=4.6)
        fist(c, wr[0] + (gd - wr)[0] * .25 + u[0] * 14, wr[1] + (gd - wr)[1] * .25 + u[1] * 14, 35, math.atan2(-u[1], -u[0]), pal, 4., 1 if sg > 0 else -1)
    arm((1082, 672), (1232, 618), (1340, 520), 1, (1368, 486), (1668, 118), 'enma', 1, 31, 25, glint)
    arm((830, 676), (672, 706), (552, 786), -1, (520, 806), (66, 972), 'kitetsu', -1, 31, 44, glint)
    if face:
        H_ = zoro.draw(c, *S([(930, 520)])[0], 104, -.16, -.10, -.07, 'bite', key=-1, f=f, body=None, bandana=True, tails=.4, swing=.25 * math.sin(d * .8), seed=4); mouth_sword(c, H_, 26, 34, glint, 3.4)


def mouth_sword(c, H_, w0, w1, glint=None, lw=4., L=4.4, dz=.12, up=.05, refl=0.):
    """Wado Ichimonji across clenched teeth: pommel on his right, blade to his left"""
    z = float(H_.z(0., -.66)) + dz; P = H_.P3(np.array([(-.86, -.672, z - .02), (.60, -.664, z), (L, -.62 + up * 6, z + .10 + .35 * (H_.yaw < 0))])); pm, g, t = P; s = H_.s
    katana(c, g, t, w0, w1, 'wado', .022, -1, glint, lw, refl=refl); hilt(c, g, pm, w0 * .98, 'wado', lw, asp=.34, n=6)
    p = Pen(c, H_, s, INK, 9)
    for x in np.linspace(-.215, .215, 6):                                    # canines and incisors bite over the wrap
        hw = .036; h_ = .050 if abs(x) < .16 else .068
        q = [(x - hw, -.585, dz * .9), (x + hw, -.585, dz * .9), (x + hw * .8, -.585 - h_, dz * 1.35), (x, -.592 - h_ * 1.1, dz * 1.4), (x - hw * .8, -.585 - h_, dz * 1.35)]; p.poly(q, '#fdf9ef'); p.line(q + q[:1], 1.8, n=3)
        q = [(x - hw, -.752, dz * .9), (x + hw, -.752, dz * .9), (x + hw * .8, -.752 + h_ * .8, dz * 1.35), (x - hw * .8, -.752 + h_ * .8, dz * 1.35)]; p.poly(q, '#ece4d8'); p.line(q + q[:1], 1.6, n=3)
    p.line([(-.36, -.60, .02), (-.22, -.570, dz * .6), (.0, -.566, dz * .8), (.22, -.570, dz * .6), (.37, -.61, .02)], 4.8, taper=(.12, .12)); p.line([(-.33, -.735, .02), (.0, -.775, dz * .6), (.33, -.735, .02)], 3.6, taper=(.15, .15))


def shoe(c, an, toe, w, lw=4., seed=0, hot=False):
    """black dress shoe: ankle point -> toe point"""
    an, toe = A_(*an), A_(*toe); d = toe - an; L = np.hypot(*d); u = d / L; m = A_(-u[1], u[0]); m = m if m[1] > 0 else -m
    P = spl(np.array([an - u * w * .8 - m * w * .55, an + u * w * .4 - m * w * .75, an + u * L * .45 - m * w * .30, toe - u * w * .4 - m * w * .12, toe + m * w * .25, toe - u * w * .2 + m * w * .62, an + u * L * .3 + m * w * .60, an + u * w * .1 + m * w * .62, an - u * w * .2 + m * w * 1.0, an - u * w * .95 + m * w * .95, an - u * w * 1.0 + m * w * .1]), True, (8, 9), n=6)
    cloth(c, P, '#ff9a34' if hot else '#1a1920', '#c7341a' if hot else '#0b0a0f', [np.vstack([P[len(P) // 2:]])], '#fff0a0' if hot else '#6a6978', [spl(np.array([an + u * L * .35 - m * w * .25, toe - u * w * .6 - m * w * .02, toe - u * w * .5 + m * w * .12, an + u * L * .4]), True, n=5)], w=lw, seed=seed)


def sole(c, x, y, s, rot, hot=1., lw=6.):
    """the bottom of a dress shoe coming at the lens, heated"""
    P = lambda q: xf(np.asarray(q, float) * s, 1, 1, rot, x, y); B = spl(P([(-.36, -1.02), (0, -1.16), (.36, -1.02), (.52, -.58), (.42, -.05), (.33, .30), (.40, .58), (.36, .98), (0, 1.10), (-.36, .98), (-.40, .58), (-.33, .30), (-.42, -.05), (-.52, -.58)]), True, n=8)
    W_ = spl(P([(-.44, -1.02), (.02, -1.26), (.46, -1.04), (.64, -.55), (.52, .0), (.42, .30), (.50, .60), (.46, 1.05), (0, 1.22), (-.46, 1.05), (-.50, .60), (-.42, .30), (-.52, .0), (-.64, -.55)]), True, n=8); cloth(c, W_, '#5a1408', '#2a0806', [P([(.1, -1.3), (.8, -1.2), (.8, 1.4), (.2, 1.3), (.3, 0)])], w=lw * 1.2)
    c.fill(B, '#e8461a'); c.clip(B); c.fill(spl(P([(-.30, -.90), (0, -1.02), (.30, -.88), (.40, -.55), (.30, -.10), (.0, .02), (-.30, -.10), (-.40, -.55)]), True, n=8), '#ff9a2a'); c.fill(spl(P([(-.20, -.76), (.04, -.84), (.22, -.70), (.26, -.48), (.12, -.22), (-.14, -.24), (-.26, -.5)]), True, n=8), '#ffe27a'); c.fill(spl(P([(-.10, -.64), (.06, -.68), (.14, -.54), (.04, -.36), (-.12, -.42)]), True, n=8), '#fffbe0')
    H2 = spl(P([(-.36, .52), (0, .46), (.36, .52), (.32, .94), (0, 1.02), (-.32, .94)]), True, (0, 2), n=6); c.fill(H2, '#ffb43a'); c.fill(spl(P([(-.22, .62), (0, .58), (.22, .62), (.18, .84), (0, .90), (-.18, .84)]), True, n=6), '#fff3b0')
    for k in range(6): c.line(spl(P([(-.44, -.92 + .16 * k), (0, -.98 + .16 * k), (.44, -.92 + .16 * k)]), n=6), lw * .5, '#8a1f0c', taper=(.2, .2), press=.4, seed=k)
    c.unclip(); outline(c, H2, lw * .8, '#4a1008', 2); outline(c, B, lw, INK, 3)
    q = np.linspace(2.7, 4.2, 20); c.line(P(np.c_[np.cos(q) * .58, np.sin(q) * 1.1 - .05]), lw * 1.5, '#fff6cf', taper=(.2, .4))


def suit(c, J, f=0, seed=3, expr='glare', hs=78, yaw=0., pitch=0., roll=0., lw=4., hot=None, wind=0., look=(0, 0), head=True, far='L', pk=True, tails=0., bs=None):
    """Sanji in the black double-breasted suit, hands in his pockets.  J: joints in px (neck, sL, sR, eL, eR, pL, pR, hL, hR, kL, kR, aL, aR, tL, tR); L = screen left"""
    S = sanji.SUIT; K = dict(b='#24232e', d='#100f17', l='#4b4a5c'); q = {k: A_(*v) for k, v in J.items() if np.ndim(v)}; s = bs or hs / 78.
    def leg(sd):
        gw = float(J.get('grow' + sd, 1.)); h, k, a, t = q['h' + sd], q['k' + sd], q['a' + sd], q['t' + sd]; P, A, B, cc = limb([h, (h + k) / 2 + (q['o' + sd] if 'o' + sd in q else 0), k, (k + a) / 2, a], np.array([44, 46, 40, 34, 33, 28, 27]) * s * np.linspace(1, gw, 7), np.array([42, 42, 36, 33, 32, 28, 27]) * s * np.linspace(1, gw, 7)); dk = A if A.mean(0)[1] > B.mean(0)[1] else B
        hotleg = hot == sd; e = a - k; e = e / (np.hypot(*e) + 1e-6); m = A_(-e[1], e[0])
        shoe(c, a + e * 8 * s * gw, t, 24 * s * gw, lw * (1 + .3 * (gw - 1)), seed, hotleg)
        cloth(c, P, K['b'], K['d'], [np.vstack([cc, dk[::-1]])], K['l'], [], w=lw * 1.15, folds=[np.array([k - m * 30 * s, k + e * 4 * s, k + m * 22 * s]), np.array([k - m * 26 * s - e * 22 * s, k + m * 8 * s - e * 10 * s]), np.array([k + e * 30 * s - m * 20 * s, k + e * 44 * s + m * 14 * s]), np.array([h * .7 + k * .3 + m * 26 * s, h * .4 + k * .6])], fw=lw * .55, seed=seed)
        if hotleg:
            n = len(cc); i0 = int(n * .52); G = [A[i0:], B[i0:][::-1]]; c.fill(np.vstack(G), '#f0641a', .85); c.fill(np.vstack([cc[i0 + 8:] * .6 + A[i0 + 8:] * .4, (cc[i0 + 8:] * .5 + B[i0 + 8:] * .5)[::-1]]), '#ffd24a', .9); c.fill(np.vstack([cc[int(n * .78):] * .8 + A[int(n * .78):] * .2, (cc[int(n * .78):] * .7 + B[int(n * .78):] * .3)[::-1]]), '#fffbe0'); outline(c, P, lw, INK, seed)
    def armp(sd):
        sh, e, p = q['s' + sd], q['e' + sd], q['p' + sd]; P, A, B, cc = limb([sh, (sh + e) / 2, e, (e + p) / 2, p], np.array([33, 30, 27, 27, 25, 21, 21]) * s, np.array([31, 28, 26, 26, 24, 21, 21]) * s); dk = A if A.mean(0)[1] > B.mean(0)[1] else B; d = e - sh; d = d / (np.hypot(*d) + 1e-6); m = A_(-d[1], d[0])
        cloth(c, P, K['b'], K['d'], [np.vstack([cc, dk[::-1]])], K['l'], [], w=lw * 1.1, folds=[np.array([e - m * 26 * s, e + m * 12 * s]), np.array([e - m * 22 * s - d * 18 * s, e + m * 4 * s - d * 12 * s])], fw=lw * .55, seed=seed)
    nr = 'R' if far == 'L' else 'L'
    if not (J.get('flip')): leg(far); armp(far)
    sL, sR, nk, hL, hR = q['sL'], q['sR'], q['neck'], q['hL'], q['hR']; hp = (hL + hR) / 2; ax = hp - nk; ax = ax / np.hypot(*ax); mx = A_(-ax[1], ax[0]); wd = np.hypot(*(sR - sL)) * .5; mx = mx if mx @ (sR - sL) > 0 else -mx
    tl = [hp + mx * (52 + 8 * tails) * s + ax * (78 + 10 * math.sin(f * .9)) * s, hp - mx * (52 + 18 * tails) * s + ax * (82 - 8 * math.sin(f * .9 + 1)) * s]
    T = spl(np.array([nk - mx * 18 * s, sL + (sL - nk) * .10, sL + ax * 52 * s + mx * 6 * s, nk * .45 + hp * .55 - mx * wd * .66, hp - mx * 46 * s, tl[1], hp + ax * 60 * s - mx * 6 * s, tl[0], hp + mx * 46 * s, nk * .45 + hp * .55 + mx * wd * .66, sR + ax * 52 * s - mx * 6 * s, sR + (sR - nk) * .10, nk + mx * 18 * s]), True, (0, 5, 6, 7, 12), n=6)
    sgk = 1
    cloth(c, T, K['b'], K['d'], [spl(np.array([nk + mx * 20 * s, nk + ax * 90 * s + mx * 34 * s, nk + ax * 150 * s + mx * 10 * s, tl[0], sR + ax * 200 * s + mx * 60 * s, sR + mx * 60 * s]), True, (0, 3, 4, 5), n=5)], K['l'], [spl(np.array([sL + ax * 4 * s + mx * 8 * s, sL + mx * 40 * s - ax * 8 * s, sL + mx * 36 * s + ax * 14 * s, sL + ax * 22 * s + mx * 14 * s]), True, n=5)], w=lw * 1.25, folds=[np.array([hp + mx * 44 * s - ax * 10 * s, hp + mx * 8 * s + ax * 6 * s]), np.array([hp - mx * 44 * s - ax * 4 * s, hp - mx * 10 * s + ax * 10 * s]), np.array([nk + ax * 120 * s - mx * 50 * s, nk + ax * 140 * s - mx * 18 * s])], fw=lw * .55, seed=seed)
    V = np.array([nk - mx * 20 * s, nk + ax * 96 * s + mx * 4 * s, nk + mx * 20 * s]); cloth(c, V, S['shirt'], S['shirtd'], [np.array([nk, nk + ax * 96 * s, nk + mx * 22 * s])], w=lw * .8)
    c.fill(np.array([nk - mx * 7 * s + ax * 8 * s, nk + mx * 7 * s + ax * 8 * s, nk + mx * 8 * s + ax * 70 * s, nk + ax * 92 * s, nk - mx * 5 * s + ax * 70 * s]), '#14131b')
    for sg in (-1, 1): cloth(c, np.array([nk + mx * sg * 22 * s - ax * 8 * s, nk + mx * sg * 50 * s + ax * 34 * s, nk + mx * sg * 34 * s + ax * 46 * s, nk + mx * sg * 42 * s + ax * 58 * s, nk + ax * 106 * s + mx * 4 * s, nk + mx * sg * 18 * s + ax * 14 * s]), K['l'] if sg < 0 else K['b'], w=lw * .8); cloth(c, np.array([nk + mx * sg * 6 * s, nk + mx * sg * 26 * s - ax * 14 * s, nk + mx * sg * 34 * s + ax * 10 * s, nk + mx * sg * 14 * s + ax * 24 * s]), S['shirt'], w=lw * .7)
    for j in range(2):
        for sg in (-1, 1): b = nk + ax * (122 + 34 * j) * s + mx * sg * 20 * s; a = np.linspace(0, 6.283, 14); c.fill(b + np.c_[np.cos(a), np.sin(a)] * 5.4 * s, S['btn']); c.line(b + np.c_[np.cos(a), np.sin(a)] * 5.4 * s, lw * .5, INK, closed=True)
    leg(nr) if not J.get('flip') else (leg('L'), leg('R'))
    armp(nr) if not J.get('flip') else (armp('L'), armp('R'))
    if head:
        hx = (nk - ax * hs * 1.34 if 'head' not in q else q['head']) + A_(*J.get('ho', (0, 0))); N, *_ = limb([nk + ax * 6 * s, hx + ax * 30 * s], [17 * s, 17 * s], [17 * s, 17 * s]); cloth(c, N, sanji.PAL['shade'], w=lw * .8); sanji.draw(c, hx[0], hx[1], hs, yaw, pitch, roll, expr, key=-1, f=f, body=False, wind=wind, look=look, smoke=False, seed=5)
