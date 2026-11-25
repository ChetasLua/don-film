// voice_style.mjs : how the LEAD performs each line.  eff = effort at start / peak / end (0 whisper-close .. 1.2 everything he has).  ACT indexes the notes after slurs are merged.
export const LEAD = { Lt: 17.0, synth: { fp: 5, fpd: .08, hfc: 5600, hfn: 6, live: 1.0 } };
const B = { sharp: 10, rd: .38, breath: .50, tiltK: .86, vib: 26, vibRate: 5.3, vibMin: .6, accent: .8, wn: 40, zeta: .62, scoop0: -85, rel: .22, onset: .045, cons: 1.0, breathIn: 1.1, brAmp: .024, brLen: .30, sib: .92, fricDb: -5, jit: 1.1, arc: 1.8, tau: .0265, rasp: .03, lilt: .024, open: .04 };
const C = { sharp: 18, rd: .12, breath: .30, vib: 34, vibRate: 5.7, vibMin: .55, accent: 1.2, wn: 47, zeta: .58, scoop0: -120, rel: .13, onset: .022, cons: 1.0, breathIn: 1.1, brAmp: .026, brLen: .26, sib: .92, jit: 1.15, rasp: .07, belt: .8, open: .22, tau: .025, lilt: .018, fall: 35, tiltK: 1.1 };
const D = { rd: .68, breath: .70, tiltK: .86, vib: 20, vibRate: 4.9, vibMin: .6, accent: .6, wn: 35, zeta: .72, scoop0: -110, rel: .36, onset: .06, cons: 1.0, breathIn: 1.5, brAmp: .030, brLen: .34, sib: .90, fricDb: -5, jit: 1.5, arc: 1.8, tau: .0285, rasp: .04, tremor: .7, fall: 70, lilt: 0 };
const F = { sharp: 16, rd: .00, breath: .26, vib: 46, vibRate: 6.0, vibMin: .55, accent: 1.5, wn: 50, zeta: .56, scoop0: -150, rel: .12, onset: .018, cons: 1.06, breathIn: 1.2, brAmp: .028, brLen: .22, sib: 1.0, jit: 1.2, rasp: .13, belt: 1, open: .32, tau: .025, lilt: .022, fall: 50, tiltK: 1.2, vot: 1.2, head: .22 };
const G = { sharp: 14, rd: .42, breath: .50, tiltK: .86, vib: 30, vibRate: 5.2, vibMin: .6, accent: .6, wn: 39, zeta: .66, scoop0: -75, rel: .28, onset: .05, cons: 1.0, breathIn: 1, brAmp: .020, brLen: .30, sib: .92, fricDb: -5, jit: 1.1, arc: 1.7, tau: .0275, lilt: .014 };
export const STYLE = {
  b1: { ...B, eff: [.34, .42, .32] }, b2: { ...B, eff: [.36, .44, .30] }, b3: { ...B, eff: [.38, .48, .40] }, b4: { ...B, eff: [.44, .62, .52], peak: .6, accent: 1.1, rasp: .05 }, b5: { ...B, eff: [.58, .74, .80], peak: .4, rel: .10, vib: 40, belt: .5, accent: 1.1, rasp: .06, open: .16 },
  c10: { ...C, eff: [.62, .70, .78], vib: 0, fall: 160, rasp: .10, scoop0: -60, nasal: .22 }, c13: { ...C, eff: [.82, .95, .95] }, c15: { ...C, eff: [.88, .98, .98], vib: 40 },
  d1: { ...D, eff: [.42, .56, .28], peak: .35 }, d2: { ...D, eff: [.34, .48, .30], rd: .72 }, d3: { ...D, eff: [.30, .38, .24], rd: .72 }, d4: { ...D, eff: [.62, .92, 1.02], rd: .22, breath: .6, rasp: .12, belt: .7, open: .25, accent: 1, wn: 42, fall: 0, peak: .5, tremor: .9 }, d5: { ...D, eff: [.28, .32, .20], rd: .66, breath: .85 }, d6: { ...D, eff: [.4, .4, .3], breath: 1, breathIn: .5, sib: 1, cons: 1.2, brAmp: .02 }, d7: { ...D, eff: [.30, .36, .18], rd: .64, tremor: .9 }, d8: { ...D, eff: [.34, .40, .36], rd: .62, peak: .6, tremor: 1.0 },
  f2: { ...F, eff: [.86, .98, .98] }, f3: { ...F, eff: [.94, 1.02, 1.0] }, f4: { ...F, eff: [.98, 1.04, 1.0] }, f5: { ...F, eff: [.96, 1.08, 1.12], peak: .6, vib: 54 },
  g1: { ...G, eff: [.36, .46, .38] }, g2: { ...G, eff: [.52, .64, .58], rasp: .03, vib: 36 }, g3: { ...G, eff: [.26, .32, .18], rd: .48, breath: .6, vib: 0, wn: 58, zeta: .8, drift: 1.8, fall: 120, scoop0: -40, lilt: 0 },
};
export const ACT = {
  b1: { 0: { scoop: -160, scoopT: .07 }, 2: { dt: .012 }, 3: { dt: .02, acc: .5 }, 6: { dt: .012 }, 7: { acc: .9 }, 8: { dt: .018 }, 9: { fall: 50 } },
  b2: { 0: { acc: .9, scoop: -70 }, 2: { dt: .012, acc: .5 }, 3: { dt: .012 }, 4: { dt: .01 }, 8: { vib: 22, airTail: .55, fall: 60 } },
  b3: { 0: { dev: .25 }, 2: { dev: .45 }, 3: { acc: 1, flip: [[.02, .06, 160, .4]] }, 5: { dt: .02 }, 7: { acc: .9 }, 10: { airTail: .5, fall: 80, dd: -.02 } },
  b4: { 3: { acc: .8 }, 6: { acc: 1.9, scoop: -150, eff: .16, rasp: .12 }, 7: { acc: .3 }, 10: { vib: 24, fall: 40 } },
  b5: { 0: { scoop: -130, acc: 1.2 }, 2: { acc: .9 }, 3: { acc: 1 }, 6: { acc: 1.3, scoop: -110, scoopT: .08, cresc: .26, vib: 44, vibOn: .42, rasp: .10, rel: .06 } },
  c10: { 0: { scoop: -60 }, 3: { acc: 1.4, bend: [[.06, 90, .03]] }, 6: { acc: 1.8, pitch: 'A4>D4', glideAt: .30, rasp: .2, d: 3.2, fall: 260, fallT: .3 } },
  c13: { 3: { acc: 1.2, scoop: -140, vib: 0 }, 5: { acc: 1.6, scoop: -200, scoopT: .08, rasp: .12 }, 8: { rel: .1, fall: 60 } },
  c15: { 0: { acc: 1.4, scoop: -200, scoopT: .07 }, 4: { acc: .9 }, 6: { acc: 1 }, 7: { acc: 1.7, scoop: -150, vib: 48, vibOn: .34, cresc: .05, rasp: .14, rel: .14, fall: 90 } },
  d1: { 0: { scoop: -140, dt: .02 }, 1: { dt: .015 }, 2: { flip: [[.015, .075, 330, .8]], acc: .9 }, 3: { airTail: .8, vib: 0, fall: 80 }, 4: { scoop: -200, scoopT: .10, acc: .6, airTail: .9, fall: 120 }, 5: { dt: .015 }, 6: { dt: .02 }, 7: { dt: .02 }, 8: { airTail: .9, vib: 12, fall: 110, flip: [[.16, .22, 170, .6]] } },
  d2: { 0: { dt: .03, acc: .4 }, 1: { dev: .65 }, 3: { acc: .9, flip: [[.05, .11, 210, .7]] }, 4: { dt: .02 }, 5: { dt: .03 }, 6: { dt: .02 }, 7: { acc: .7, scoop: -130, scoopT: .09, vib: 18, vibOn: .42, airTail: .85, fall: 90, rel: .3 } },
  d3: { 0: { dt: .02, scoop: -90 }, 4: { vib: 0, airTail: .4 }, 5: { dev: .3, dt: .03 }, 7: { dt: .02 }, 9: { airTail: .9, fall: 130, dt: .02 } },
  d4: { 0: { scoop: -220, scoopT: .10, acc: .8, cresc: .18, vib: 0 }, 1: { acc: .6, rasp: .14 }, 2: { acc: 1.5, flip: [[.03, .10, 380, .9], [.24, .28, 240, .6]], rasp: .22, raspT: .22, bend: [[.0, -60, .03]] }, 3: { flip: [[.05, .13, -330, .8]], rasp: .16 }, 4: { acc: 1, nobreath: 1 }, 5: { acc: .8, rasp: .16 }, 6: { acc: 1, rasp: .18 }, 8: { acc: 1.8, scoop: -260, scoopT: .07, vib: 34, vibOn: .2, rasp: .24, raspT: .3, flip: [[.27, .33, 300, 1]], dd: .06, fall: 260, fallT: .2, rel: .14 } },
  d5: { 0: { dev: .6 }, 2: { airTail: .9, fall: 90, dd: -.02 }, 4: { dt: .03, vib: 0 }, 5: { dev: .6 }, 7: { airTail: 1, fall: 140 } },
  d7: { 0: { dt: .03 }, 3: { acc: .6, flip: [[.10, .15, 190, .7]] }, 7: { airTail: 1, vib: 14, vibOn: .4, fall: 150, fallT: .3 } },
  f2: { 0: { scoop: -240, scoopT: .06, acc: 1.2 }, 2: { acc: .8 }, 3: { acc: 1.6 }, 6: { acc: 1.2 }, 9: { acc: 1.4, scoop: -90 } },
  f3: { 0: { acc: 1.5 }, 3: { acc: 1.3 }, 6: { acc: 1.7, scoop: -180, rasp: .2 }, 8: { acc: 1.4, vib: 44, fall: 90 } },
  f4: { 0: { acc: 1.8, scoop: -200 }, 3: { acc: 1.3 }, 6: { acc: 1.5 }, 8: { acc: 1.3, scoop: -120, vib: 44, fall: 60 } },
  f5: { 0: { acc: 1.5, scoop: -170, scoopT: .07 }, 2: { acc: 1.2, vib: 40 }, 3: { acc: 1.3 }, 6: { acc: 1.9, scoop: -320, scoopT: .10, rasp: .24, raspT: .35, vib: 30, cresc: .04 }, 7: { acc: .8, vib: 64, vibOn: .30, rasp: .14, rel: .10 } },
  g1: { 0: { scoop: -90, dt: .02 }, 3: { acc: .8, dt: .02 }, 4: { dt: .015 }, 6: { acc: .9 }, 8: { dt: .015 }, 10: { vib: 20, vibMin: .4, airTail: .4, fall: 40 } },
  g2: { 0: { dev: .15 }, 3: { acc: 1.2, scoop: -100 }, 6: { acc: .9 }, 9: { acc: .8 }, 12: { acc: 1.2, scoop: -80 }, 13: { vib: 34, vibOn: .34, rel: .34, airTail: .3 } },
  g3: { 0: { dt: .05, scoop: -60 }, 1: { dt: .03 }, 2: { dt: .03 }, 3: { dt: .07 }, 4: { dt: .05 }, 5: { dt: .03 }, 6: { dt: .05 }, 7: { dt: .09, dd: -.05 }, 8: { dt: .08 }, 9: { dt: .08, airTail: 1, fall: 160, fallT: .25 } },
};
// target level of each lead line (gated RMS dBFS before the chain): the dynamic plan A pp .. F fff
export const LEVEL = { b1: -25, b2: -25, b3: -24, b4: -22.5, b5: -20.5, c10: -20.5, c13: -15, c15: -14, d1: -23, d2: -23.5, d3: -25, d4: -18.5, d5: -26, d6: -27, d7: -26, d8: -23.5, f2: -14, f3: -13.5, f4: -13.5, f5: -12.5, g1: -20.5, g2: -17.5, g3: -23.5 };
export const SCREAM = -9;
