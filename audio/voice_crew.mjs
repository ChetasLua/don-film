// voice_crew.mjs : nine throats.  Lt tract length (cm), fp/fpd own pole-zero fingerprint, pan, lag (s, constant lateness), cents, wander; st = performing habits (see voice_perf); oct for calls and group lines.
export const CREW = {
  zoro:    { Lt: 17.9, pan: -.34, lag: .016, cents: -6, wander: 6, oct: -12, trim: 0, synth: { fp: 11, fpd: 0.20, hfc: 4700 }, st: { rd: .26, breath: .22, vib: 12, vibRate: 4.7, jit: 1.35, rasp: .09, scoop0: -100, drift: 1.1, hvBase: 62, wn: 36, fall: 50, piv: 190, accent: 1.1, col: { a: { o: .18 }, i: { e: .08 }, u: { o: .08 } } } },
  nami:    { Lt: 14.4, pan: .30, lag: .005, cents: 4, wander: 5, oct: 0, trim: -1, synth: { fp: 23, fpd: 0.17, hfc: 6400 }, st: { rd: .40, breath: .30, vib: 20, vibRate: 5.9, jit: .9, hvBase: 71, piv: 330, pivR: 300, wn: 50, scoop0: -60, col: { a: { e: .18 }, o: { a: .08 } } } },
  usopp:   { Lt: 16.9, pan: -.58, lag: .021, cents: 9, wander: 9, oct: 0, trim: -.5, synth: { fp: 37, fpd: 0.25, hfc: 5400 }, st: { rd: .34, breath: .28, vib: 34, vibRate: 6.5, vibMin: .4, jit: 1.5, drift: 1.8, nasal: .26, rasp: .05, hvBase: 66, wn: 52, zeta: .52, scoop0: -120, col: { a: { e: .28 }, o: { a: .14 }, u: { i: .10 } } } },
  sanji:   { Lt: 17.3, pan: .50, lag: .012, cents: -3, wander: 5, oct: 0, trim: 0, synth: { fp: 51, fpd: 0.14, hfc: 5200 }, st: { rd: .60, breath: .5, vib: 34, vibRate: 5.3, jit: .85, wn: 33, scoop0: -120, rel: .18, hvBase: 64, col: { e: { i: .10 }, o: { u: .10 } } } },
  chopper: { Lt: 12.5, pan: .14, lag: .010, cents: 11, wander: 11, oct: 0, trim: -3, synth: { fp: 67, fpd: 0.17, hfc: 7400 }, st: { rd: .56, breath: .42, vib: 0, jit: 1.2, drift: 1.9, hvBase: 77, piv: 420, pivR: 380, wn: 56, scoop0: -45, accent: .8, col: { a: { e: .08 }, u: { i: .16 }, o: { u: .10 } } } },
  robin:   { Lt: 15.4, pan: -.14, lag: .024, cents: -7, wander: 5, oct: 0, trim: -.5, synth: { fp: 83, fpd: 0.17, hfc: 5200 }, st: { rd: .86, breath: .85, vib: 20, vibRate: 4.9, vibMin: .4, jit: .9, hvBase: 68, piv: 290, pivR: 280, wn: 34, scoop0: -80, col: { a: { o: .34 }, e: { a: .06 } } } },
  franky:  { Lt: 18.5, pan: .72, lag: .014, cents: 6, wander: 8, oct: -12, trim: 0, synth: { fp: 97, fpd: 0.25, hfc: 5000 }, st: { rd: -.08, breath: .2, vib: 32, vibRate: 5.5, jit: 1.6, rasp: .20, belt: 1, scoop0: -220, hvBase: 63, piv: 190, col: { a: { e: .18 } }, tiltK: 1.3 } },
  brook:   { Lt: 15.9, pan: -.78, lag: .008, cents: 2, wander: 6, oct: 0, trim: -2, tract: { r: 260, c1: 2.4 }, synth: { fp: 109, fpd: 0.36, hfc: 6200 }, st: { rd: .92, breath: 1.0, vib: 60, vibRate: 6.4, vibMin: .30, jit: .8, hvBase: 61, wn: 40, scoop0: -70, col: { a: { o: .42 }, e: { o: .12 }, i: { u: .12 } } } },
  jinbe:   { Lt: 19.9, pan: .02, lag: .028, cents: -9, wander: 6, oct: -12, trim: 1.5, synth: { fp: 127, fpd: 0.20, hfc: 4200 }, st: { rd: .36, breath: .25, vib: 12, vibRate: 4.4, jit: 1.4, rasp: .08, hvBase: 56, piv: 150, pivR: 160, wn: 30, scoop0: -90, col: { a: { o: .40 } } } },
};
export const NAMES = Object.keys(CREW);
// the nine calls: style on top of the singer's habits, per-note acting
export const CALL = {
  zoro: { eff: [.74, .82, .70], act: { 0: { acc: 1.2 }, 3: { acc: .8 }, 4: { fall: 120, fallT: .12 } } },
  nami: { eff: [.66, .76, .70], act: { 0: { acc: 1.1 }, 3: { acc: 1, scoop: -60 }, 4: { fall: -30 } } },
  usopp: { eff: [.74, .88, .80], act: { 0: { acc: 1 }, 3: { acc: 1.2, flip: [[.06, .12, 420, .8]] }, 4: { bend: [[.08, 60, .04]] } } },
  sanji: { eff: [.60, .70, .66], act: { 0: { acc: .9 }, 4: { acc: 1.3, scoop: -60, fall: 60 } } },
  chopper: { eff: [.66, .78, .74], act: { 0: { acc: 1 }, 1: { acc: .6 }, 3: { acc: 1 }, 4: { fall: -60 } } },
  robin: { eff: [.48, .56, .46], act: { 0: { acc: .7 }, 3: { acc: .7 }, 4: { airTail: .3 } } },
  franky: { eff: [.96, 1.06, 1.12], act: { 0: { acc: 1.4 }, 4: { acc: 1.8, scoop: -420, scoopT: .07, rasp: .34, raspT: .25 } } },
  brook: { eff: [.56, .64, .62], act: { 0: { acc: .8 }, 4: { acc: 1, vib: 60, vibOn: .05, vibIn: .06 } } },
  jinbe: { eff: [.76, .86, .80], act: { 0: { acc: 1.2 }, 4: { acc: 1.2, scoop: -120, fall: 60 } } },
};
// shouted DON: start > end pitch, weight
export const SHOUT = { lead: ['B4', 'E4', 1], zoro: ['E4', 'G3', 1], nami: ['D5', 'G4', .75], usopp: ['B4', 'D4', .9], sanji: ['G#4', 'C4', .95], chopper: ['F#5', 'B4', .6], robin: ['G4', 'C4', .75], franky: ['F4', 'A3', 1.2], brook: ['D5', 'G4', .6], jinbe: ['A3', 'C3', 1.35] };
export const SOLO = { lead: ['A4', 'C#4'], zoro: ['D4', 'F3'], sanji: ['G4', 'B3'] };
// group lines: octave / harmony part of each voice per line (number = semitone shift, '3-' third below the tune, '3+' third above, '3-8' third below an octave down)
export const PARTS = {
  c13: { zoro: -12, nami: 0, usopp: 0, sanji: 0, chopper: 0, robin: -12, franky: -12, brook: '3-', jinbe: -24 },
  c15: { zoro: -12, nami: 0, usopp: 0, sanji: '3-', chopper: 0, robin: '3-8', franky: -12, brook: 0, jinbe: -24 },
  f2: { zoro: -12, nami: 0, usopp: 0, sanji: 0, chopper: 0, robin: 0, franky: -12, brook: 0, jinbe: -12 },
  f3: { zoro: -12, nami: 0, usopp: 0, sanji: '3-', chopper: 0, robin: '3-', franky: -12, brook: 0, jinbe: -12 },
  f4: { zoro: -12, nami: 0, usopp: 0, sanji: '3-', chopper: 0, robin: '3-', franky: '3-8', brook: '3+', jinbe: -12 },
  f5: { zoro: -12, nami: 0, usopp: '3-', sanji: '3-', chopper: 0, robin: '3-', franky: -12, brook: 0, jinbe: -24 },
  g1: { nami: 0, robin: '3-', sanji: '3-', zoro: -12, brook: 0 },
  g2: { zoro: -12, nami: 0, usopp: 0, sanji: '3-', chopper: 0, robin: '3-', franky: -12, brook: '3+', jinbe: -24 },
};
export const PARTS2 = { // second takes: more bodies in the room
  f2: { zoro: -12, nami: 0, usopp: -12, sanji: 0, robin: 0, franky: -12 }, f3: { zoro: -12, nami: 0, usopp: -12, sanji: 0, robin: 0, franky: '3-8' }, f4: { zoro: -12, nami: 0, usopp: -12, sanji: 0, robin: 0, franky: -12 }, f5: { zoro: -12, nami: '3-', usopp: -12, sanji: 0, robin: 0, franky: '3-8' }, g2: { nami: 0, usopp: -12, sanji: 0, robin: 0 }, c15: { zoro: -12, nami: 0, usopp: -12, sanji: 0, robin: 0, franky: -12 }, c13: { nami: 0, usopp: -12, sanji: 0, franky: -12 },
};
const g = (eff, x = {}) => ({ eff, accent: 1.2, cons: .96, sib: .62, fricDb: -9, rel: .14, brAmp: .016, brLen: .22, lilt: .012, ...x });
export const GROUP = { c13: g([.82, .95, .92]), c15: g([.86, .98, .96]), f2: g([.86, .98, .96], { accent: 1.5, lilt: .018 }), f3: g([.92, 1.0, .98], { accent: 1.5, lilt: .018 }), f4: g([.94, 1.02, 1.0], { accent: 1.5, lilt: .018 }), f5: g([.94, 1.06, 1.08], { accent: 1.4, lilt: .016 }), g1: g([.30, .38, .32], { accent: .5, breathIn: .4, rel: .3 }), g2: g([.50, .62, .56], { accent: .7, rel: .34 }) };
// bloom of e2 (second takes) and the wide E major "a" of f6
export const BLOOM = [['zoro', 'F#3', 'F#3'], ['nami', 'D4', 'D#4'], ['usopp', 'A3', 'A3'], ['sanji', 'D4', 'D#4'], ['jinbe', 'D3', 'D#3'], ['nami', 'F#4', 'F#4'], ['sanji', 'F#4', 'F#4'], ['chopper', 'D5', 'D#5'], ['usopp', 'F#4', 'F#4']];
export const WIDE = [['jinbe', 'E2'], ['franky', 'B2'], ['zoro', 'E3'], ['robin', 'A3', 'G#3', 2213], ['usopp', 'B3', 'B3'], ['sanji', 'F#4', 'E4', 2219], ['nami', 'A4', 'G#4', 2213], ['brook', 'C#5', 'B4', 2225], ['chopper', 'E5'], ['zoro', 'G#3'], ['nami', 'E4'], ['usopp', 'F#4', 'G#4', 2207], ['sanji', 'C#4', 'B3', 2225], ['robin', 'E4'], ['franky', 'E3']];
