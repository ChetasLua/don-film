// voice.mjs : every human sound in DON!   node audio/voice.mjs [task,task,...]   (no list = render all; a list re-renders only those and mixes the rest from cache/voice)
// stems: out/voice_lead.wav voice_crew.wav voice_fx.wav (dry, 48 kHz float stereo, 2544 frames, sample 0 = film frame 0), out/voice_mix.wav = all three with rooms (premix for listening)
import fs from 'node:fs';
import { Worker, isMainThread, parentPort } from 'node:worker_threads';
import { SR, writeWav, db, todb, rng, midi, clamp, smooth, lerp, slowNoise, Biquad } from './voice_dsp.mjs';
import { synth, CR, SPF, VI } from './voice_core.mjs';
import { Tract } from './voice_tract.mjs';
import { buildLine, E8 } from './voice_perf.mjs';
import { LEAD, STYLE, ACT, LEVEL, SCREAM } from './voice_style.mjs';
import { CREW, NAMES, CALL, SHOUT, PARTS, PARTS2, GROUP, BLOOM, WIDE, SOLO } from './voice_crew.mjs';
import * as fx from './voice_fx.mjs';
const SCORE = JSON.parse(fs.readFileSync(new URL('./score.json', import.meta.url)));
export const FPS = 24, SPFR = SR / FPS, TOTAL = 2544 * SPFR, CUT0 = 1605 * SPFR, CUT1 = 1725 * SPFR;
const LN = Object.fromEntries(SCORE.lines.map(l => [l.id, l])), hash = s => { let h = 2166136261; for (const c of s) { h ^= c.charCodeAt(0); h = Math.imul(h, 16777619) >>> 0; } h ^= h >>> 15; return (h >>> 0) % 999983; };
export function merged(line, act = {}) {                       // slurred same-pitch notes become one; acting attached
  const out = []; for (const n of line.notes) { if (n.slur && out.length && out[out.length - 1].pitch === n.pitch) out[out.length - 1].d += n.d; else out.push({ ...n }); }
  return out.map((n, i) => { const x = { ...(act[i] || {}) }, o = { ...n, x }; if (x.pitch) o.pitch = x.pitch; if (x.d) o.d = x.d; return o; });
}
function nextT(line, sameVoice) { const t1 = line.t1 * E8; let m; for (const q of SCORE.lines) { if (q === line || !sameVoice(q)) continue; const t = q.notes[0].t * E8; if (t >= line.notes[line.notes.length - 1].t * E8 && t < t1 + .5 && (m === undefined || t < m)) m = t; } return m; }
const isLead = q => /lead|all/.test(q.voice);
// one take: notes -> performed, trimmed mono segment at an absolute sample position
const SYN = { hfj: .14, hfs: .24, white: 3.4, flutter: 1.8, aspHp: 1100, aspLp: 6000 }, PERF = { aspFl: .10, vOn: 1.6 };   // smooth source: four blind listenings ranked less high-frequency randomness as more human
export function take(notes, st0, o = {}) {
  if (o.synth && o.synth !== LEAD.synth && (o.synth.fp ?? 5) !== 5) o = { ...o, crew: true };
  const st = { ...PERF, ...st0, rd: (st0.rd ?? 0) + .22, breath: (st0.breath ?? .5) * (o.brK ?? (o.crew ? .72 : 1)) }, b = buildLine(notes, st, o); if (o.post) o.post(b.s, b);
  const r = synth(b.s, { tract: o.tract, seed: (o.seed ?? 1) * 1009 + 17, ...SYN, ...(o.synth || {}) }); let a = 0, z = r.y.length - 1; while (a < z && Math.abs(r.y[a]) < 1e-6) a++; while (z > a && Math.abs(r.y[z]) < 1e-6) z--;
  return { start: Math.round(b.t0 * SR) - r.pad + a, y: r.y.slice(a, z + 1) };
}
export const rmsOf = y => { const W = 960; let mx = 0; const e = []; for (let i = 0; i + W <= y.length; i += W) { let s = 0; for (let k = 0; k < W; k++) s += y[i + k] ** 2; e.push(s / W); mx = Math.max(mx, s / W); } let s = 0, n = 0; for (const v of e) if (v > mx * .006) { s += v; n++; } return Math.sqrt(s / Math.max(1, n)) + 1e-9; };
const level = (seg, dbfs) => { const g = db(dbfs) / rmsOf(seg.y); for (let i = 0; i < seg.y.length; i++) seg.y[i] *= g; return seg; };
const f2t = f => f / 5;                                           // frames -> eighths

// ---------- harmony ----------
const PC = { C: 0, D: 2, E: 4, F: 5, G: 7, A: 9, B: 11 }, chordPcs = c => { const m = /^([A-G])([#b]?)(m?)(7?)/.exec(c), r = (PC[m[1]] + (m[2] === '#' ? 1 : m[2] === 'b' ? 11 : 0)) % 12; return [0, m[3] ? 3 : 4, 7, ...(m[4] ? [10] : [])].map(x => (r + x) % 12); };
function harmony(line, notes, dir) {
  const scale = new Set(line.bar >= 62 ? [4, 6, 8, 9, 11, 1, 3] : [2, 4, 6, 7, 9, 11, 1]), S = [];
  notes.forEach((n, k) => { const m = midi(n.pitch.split('>')[0]), ch = chordPcs((SCORE.chords[Math.floor(n.t / 6)] || ['D', 'D'])[n.t % 6 < 3 ? 0 : 1]), row = []; for (let iv = 3; iv <= 9; iv++) for (const d of (dir > 0 ? [1, -1] : [-1])) { const h = m + d * iv, pc = ((h % 12) + 12) % 12; if (h > 76) continue; let c = (iv <= 4 ? 0 : iv === 5 ? .8 : iv >= 8 ? 1.1 : 2.2) + (d !== dir ? 2.5 : 0); if (!scale.has(pc)) c += 8; if (!ch.includes(pc)) c += (n.d >= 2 || n.t % 3 === 0) ? 1.8 : .4; let best = 0, from = -1; if (k) { best = 1e9; S[k - 1].forEach((p, j) => { const q = Math.abs(p.h - h), v = p.cost + .25 * q + (q > 5 ? 1.2 : 0); if (v < best) { best = v; from = j; } }); } row.push({ h, cost: c + best, from }); } S.push(row); });
  let j = S[S.length - 1].reduce((a, p, i, r) => p.cost < r[a].cost ? i : a, 0); const out = []; for (let k = notes.length - 1; k >= 0; k--) { out[k] = S[k][j].h; j = S[k][j].from; }
  return notes.map((n, k) => ({ ...n, h: out[k] - midi(n.pitch.split('>')[0]) }));
}

// ---------- LEAD ----------
const leadTract = () => new Tract(LEAD.Lt);
function leadLine(id, o = {}) {
  const l = LN[id]; let notes = o.notes || merged(l, ACT[id]); const st = STYLE[id], adj = SCORE.lines.some(q => q !== l && isLead(q) && Math.abs(q.t1 - l.notes[0].t) < .5);
  const s = take(notes, st, { tract: o.tract || leadTract(), seed: (o.seed ?? 1) + hash(id), synth: { ...LEAD.synth, ...(o.synth || {}) }, nextT: nextT(l, isLead), room: adj ? .34 : .75, hum: /^[bdg]/.test(id) ? .018 : .013, ...(o.o || {}) });
  return level({ ...s, bus: 'lead', id, act: id[0] }, LEVEL[id]);
}
const mouthful = s => { for (let i = 0; i < s.n; i++) { const t = i / CR, chew = .5 + .5 * Math.sin(2 * Math.PI * 3.1 * t); for (let v = 0; v < 5; v++) s.w[v][i] *= .70; s.w[2][i] += .18 + .08 * chew; s.w[4][i] += .12 - .08 * chew; s.tilt[i] *= .62; } const m = new Float32Array(s.n).fill(.45); s.gest.push({ loc: 42.5, hw: 2.2, amin: .5, m }); };
function leadDon() { const [a, b] = SOLO.lead; return take([{ mora: 'do', pitch: a + '>' + b, t: f2t(1005), d: 1.5, x: { acc: 2.6, glideAt: .20, scoop: -200, scoopT: .03, rasp: .28, raspT: .5, vib: 0 } }, { mora: 'N', pitch: b, t: f2t(1005) + 1.5, d: .8, x: { acc: 0, fall: 260 } }], { ...STYLE.f4, ...YELL, rd: -.40 }, { tract: leadTract(), seed: 911, synth: LEAD.synth, hum: 0 }); }
// ---------- CREW ----------
const NB = { a: ['o', 'e'], i: ['e', 'u'], u: ['o', 'i'], e: ['i', 'a'], o: ['u', 'a'] };
function colours(k, tk = 0) { const R = rng(hash('col' + k + tk)), c = { ...((CREW[k] || {}).st?.col || {}) }; for (const v in NB) if (!c[v] || tk) c[v] = { [NB[v][R.u() < .5 ? 0 : 1]]: .05 + .16 * R.u() }; return c; }
const DR = {}, driftOf = (k, tk) => DR[k + tk] || (DR[k + tk] = slowNoise(hash('drift' + k + tk)));        // each singer's own clock

const TK2 = {}, tractOf = k => new Tract(CREW[k].Lt, CREW[k].tract || {});
function partNotes(line, part, notes) { if (typeof part === 'number') return notes.map(n => ({ ...n, h: part })); const h = harmony(line, notes, part.startsWith('3+') ? 1 : -1); return part.endsWith('-8') ? h.map(n => ({ ...n, h: n.h - 12 })) : h; }
function groupLine(k, id, part, tk, tract) {
  if (tk) tract = TK2[k] || (TK2[k] = new Tract(CREW[k].Lt * (1.05 + .02 * (hash(k) % 3))));
  const c = CREW[k], l = LN[id], R = rng(hash('part' + k + id + tk)), g = GROUP[id]; let notes = merged(l, {}).map((n, i, a) => { const L = ACT[id]?.[i] || {}; const own = driftOf(k, tk), tt = n.t * E8, loose = id[0] === 'g' ? .6 : 1; return { ...n, x: { dt: loose * (.013 * own(tt, .55) + .009 * own(tt + 400, 1.9) + .006 * R.g()), acc: (L.acc ?? undefined), scoop: L.scoop !== undefined ? L.scoop * (.4 + .6 * R.u()) : undefined, vib: n.d >= 2 ? ((c.st.vib ?? 0) * .9 + 5) * (.6 + .8 * R.u()) : 0, vibRate: (c.st.vibRate ?? 5.4) * (.9 + .2 * R.u()), rel: i === a.length - 1 ? .14 + .12 * R.u() : undefined } }; });
  notes = partNotes(l, part, notes); const lo = tk ? .4 : 1;
  const s = take(notes, { ...c.st, ...g, inton: 24, drift: 1.9 * (c.st.drift ?? 1), vibMin: .34, col: colours(k, tk), rasp: (c.st.rasp ?? 0) * .6, rd: (c.st.rd ?? 0) + .06 + .15 * (1 - lo), breathIn: tk ? 0 : g.breathIn ?? .8, belt: c.st.belt ?? .4, open: .12, tau: .022 }, { tract, seed: hash('grp' + k + id + tk), synth: { ...c.synth, fp: c.synth.fp + tk * 23, fpd: .28, disp: .2, balance: .5 }, lag: c.lag + (tk ? .020 * R.u() : 0) + .004 * R.g(), hum: id[0] === 'g' ? .010 : .011, detune: c.cents + (tk ? 10 * R.g() : 0), wander: c.wander + 6, rub: id[0] === 'g' ? .6 : 1.1, nextT: nextT(l, q => q.voice === 'all') });
  return level({ ...s, bus: id === 'g1' ? 'soft' : 'grp', id, who: k, tk, room: k + tk, pan: clamp(c.pan * (tk ? -.8 : 1), -.9, .9) }, (id[0] === 'g' ? -31.5 : -31.2) + c.trim + (tk ? -3 : 0) + 1.4 * R.g());
}
const YELL = { eff: [1.12, 1.2, .85], breathIn: 0, rel: .08, wn: 60, cons: .95, belt: .6, open: .45, vib: 0, tiltK: 1.35, breath: .8, jit: 2.2, drift: 3, accDb: 4.5, weak: 0, inton: 30, rub: 0, dBurst: 1.0, evolve: .3, col: { o: { a: .40 } } };
function donShout(k, f, i, tract, tk = 0) { const [a, b, w] = SHOUT[k], R = rng(hash('sh' + k + i + '_' + tk)), c = CREW[k] || { st: STYLE.f4, synth: LEAD.synth, lag: .006, pan: .06, trim: 0 }, sh = Math.round(R.g() * 2.4) - 1 + (tk ? (k === 'nami' || k === 'chopper' || k === 'robin' ? -2 : -4) : 0), d0 = .72 + .34 * R.u(); if (tk) tract = TK2[k] || (TK2[k] = new Tract((CREW[k]?.Lt ?? 17) * (1.05 + .02 * (hash(k) % 3))));
  const s = take([{ mora: 'do', pitch: a + '>' + b, t: f2t(f), d: d0, x: { acc: 2.2, glideAt: .16 + .16 * R.u(), scoop: -180 - 260 * R.u(), scoopT: .035, rasp: .06 + .10 * R.u() + (c.st.rasp ?? 0), raspT: .25, vib: 0 } }, { mora: 'N', pitch: b, t: f2t(f) + d0, d: .52 + .36 * R.u(), x: { acc: 0, fall: 260 * R.u() } }], { ...c.st, ...YELL, rd: (c.st.rd ?? 0) * .4 - .40, col: { ...(c.st.col || {}), ...YELL.col } }, { tract, seed: hash('don' + k + i + '_' + tk), shift: sh, synth: { ...c.synth, fp: c.synth.fp + 19 * tk, fpd: .30, disp: .25 }, lag: c.lag * .4 + .014 * R.g() + (tk ? .010 : 0), hum: .006, detune: c.cents ?? 0 });
  return level({ ...s, bus: 'don', id: 'don' + i, who: k, room: k + tk, pan: clamp(c.pan * (tk ? -.7 : 1), -.9, .9) }, -23.5 + 20 * Math.log10(w) + 1.5 * R.g() - (tk ? 2.5 : 0)); }
function crewTask(k) {
  const c = CREW[k], tract = tractOf(k), out = [], idx = NAMES.indexOf(k), l = SCORE.lines.find(q => q.voice === k), cs = CALL[k];
  // call
  out.push(level({ ...take(merged(l, cs.act).map(n => ({ ...n, h: c.oct })), { accent: 1, cons: 1.14, sib: 1.1, fricDb: -3, rel: .07, breathIn: .9, brAmp: .02, brLen: .2, lilt: .012, open: .12, reduce: 0, weak: .8, onsetDip: 18, sharp: 14, ...c.st, rd: (c.st.rd ?? 0) - .12, eff: cs.eff }, { tract, seed: hash('call' + k), synth: c.synth, lag: c.lag * .5, hum: .009, detune: c.cents * .6, wander: c.wander * .6, nextT: (l.notes[4].t + 1.12) * E8, room: .3 }), bus: 'call', id: l.id, who: k, room: k + 0, pan: clamp(c.pan * .45, -.35, .35) }, -19.5 + (k === 'chopper' ? -1 : k === 'jinbe' ? 1 : 0)));
  // DON! after every call but his own
  for (let i = 0; i < 9; i++) if (i !== idx) { out.push(donShout(k, 565 + 30 * i, i, tract)); out.push(donShout(k, 565 + 30 * i, i, tract, 1)); }
  if (k === 'zoro' || k === 'sanji') { const f = k === 'zoro' ? 990 : 1020, [a, b] = SOLO[k], lo = k === 'zoro'; out.push(level({ ...take([{ mora: 'do', pitch: a + '>' + b, t: f2t(f), d: 1.5, x: { acc: 2.6, glideAt: .16, scoop: -170, scoopT: .03, rasp: lo ? .34 : .20, raspT: .5, vib: 0 } }, { mora: 'N', pitch: b, t: f2t(f) + 1.5, d: .8, x: { acc: 0, fall: 260 } }], { ...c.st, ...YELL, rd: lo ? -.40 : -.30 }, { tract, seed: hash('solo' + k), synth: c.synth, hum: 0 }), bus: 'solo', id: 'solo', who: k, pan: lo ? -.25 : .25 }, -15.5)); }
  for (const id in PARTS) if (PARTS[id][k] !== undefined) out.push(groupLine(k, id, PARTS[id][k], 0, tract));
  for (const id in PARTS2) if (PARTS2[id][k] !== undefined) out.push(groupLine(k, id, PARTS2[id][k], 1, tract));
  // e2 entry: accent, dip under the next, swell to the DON
  const aNote = (pitch, to, turn, f, end, x) => [{ mora: 'a', pitch: to && to !== pitch ? pitch + '>' + to : pitch, t: f2t(f), d: (end - f) / 5, x: { glideAt: turn ? (turn - f - 1) / (end - f) : .5, vib: (c.st.vib ?? 16) * 1.2 + 8, vibOn: Math.max(.5, (1790 - f) / 24 + .5 * (idx % 3) / 3), vibIn: 1.2, scoop: -110, scoopT: .07, rel: .07, ...x } }];
  const env = (s, pts) => { for (let i = 0; i < s.y.length; i++) { const f = (s.start + i) / SPFR; let g = pts[0][1]; for (let q = 1; q < pts.length; q++) if (f >= pts[q - 1][0]) { const u = clamp((f - pts[q - 1][0]) / (pts[q][0] - pts[q - 1][0]), 0, 1); g = lerp(pts[q - 1][1], pts[q][1], smooth(u)); } s.y[i] *= db(g); } return s; };
  const ei = SCORE.swell.findIndex(q => q.who === k), E = SCORE.swell[ei], sw = { ...c.st, accent: 1, breathIn: 1.6, brAmp: .034, brLen: .34, onset: .03, eff: [.46, .66, 1.02], peak: .55, swell: .4, open: .1, inton: 18, drift: (c.st.drift ?? 1) * 1.5, rasp: (c.st.rasp ?? 0) * .4, tremor: .3 };
  { const last = E.end - E.f < 40, reb = { jinbe: 1803, zoro: 1811, nami: 1819, usopp: 1827, sanji: 1834 }[k], nx = { acc: 1.4, cresc: last ? .2 : .30, ...({ zoro: { fry: .09, scoop: -160 }, nami: { scoop: -130 }, usopp: { scoop: -190, scoopT: .12, flip: [[.5, .58, 160, .6]] }, sanji: { scoop: -240, scoopT: .13 }, chopper: { scoop: -60, bend: [[.35, -35, .08], [.7, 30, .1]] }, robin: { scoop: -120, scoopT: .11 }, franky: { scoop: -300, scoopT: .10, rasp: .25, raspT: .3 }, brook: { scoop: -90, vib: 60, vibOn: .25 }, jinbe: { fry: .10, scoop: -120 } })[k] };
    const line1 = reb ? [...aNote(E.pitch, null, 0, E.f, reb - 9, { ...nx, cresc: .12, vibOn: 1.0, vibIn: .8, rel: .16 }), ...aNote(E.pitch, E.to, E.turnFrame, reb, E.end, { acc: 1.2, cresc: .26, scoop: -140, scoopT: .09, vibOn: .35, vibIn: .5, brLen: .24 })] : aNote(E.pitch, E.to, E.turnFrame, E.f, E.end, nx);
    const s = level({ ...take(line1, { ...sw, inton: 28, wn: 26, eff: last ? (k === 'brook' ? [.62, .7, .74] : [.82, .9, .95]) : sw.eff, col: colours(k) }, { tract, brK: .55, seed: hash('e2' + k), synth: c.synth, lag: .0, hum: .003, detune: c.cents * .4, wander: 4, post: s => { const i0 = 1.0 * CR, i1 = i0 + (E.end - E.f) / 24 * CR; for (let i = 0; i < s.n; i++) { const u = clamp((i - i0) / (i1 - i0), 0, 1), q = last ? 1 : smooth(u); s.eff[i] = clamp(s.eff[i] + lerp(-.30, .50, q), 0, 1.98); s.tilt[i] *= lerp(.55, 1.3, q); s.asp[i] *= lerp(1.7, 1, q); const k2 = .30 * (1 - q); s.w[4][i] += k2 * s.w[0][i] * .7; s.w[3][i] += k2 * s.w[0][i] * .3; s.w[0][i] *= 1 - k2; } } }), bus: 'swell', id: 'e2', who: k, room: k + 0, pan: clamp(Math.cos(Math.PI * (ei + .5) / 9) * -.85 * (ei % 2 ? -1 : 1) * (k === 'jinbe' ? .1 : 1), -.9, .9) }, -23 + (k === 'jinbe' ? 4 : k === 'brook' ? 0 : k === 'franky' ? 2 : 0)); out.push(env(s, [[E.f, 6], [E.f + 11, 5], [E.f + 24, -2], [Math.max(E.f + 26, 1800), -1.5], [Math.max(E.f + 30, 1853), 1]])); }
  BLOOM.forEach(([w, p, q], i) => { if (w !== k) return; const s = level({ ...take(aNote(p, q, q !== p ? 1845 : 0, 1806, 1855, { acc: .2, cresc: .3, scoop: -40 }), { ...sw, slowAtk: .9, eff: [.5, .7, .9], breathIn: 0, rd: (c.st.rd ?? 0) + .15 }, { tract, seed: hash('bloom' + k + i), synth: { ...c.synth, fp: c.synth.fp + 2 }, hum: .02, detune: c.cents + 6, wander: 8 }), bus: 'swell', id: 'bloom', who: k, room: k + 'b' + i, pan: clamp(.75 * Math.sin(i * 2.4), -.9, .9) }, -26); out.push(env(s, [[1806, -9], [1840, -2], [1853, 3]])); });
  WIDE.forEach(([w, p, q, tf], i) => { if (w !== k) return; const end = 2238 + (i * 7) % 14; const s = level({ ...take([{ mora: 'a', pitch: q && q !== p ? p + '>' + q : p, t: f2t(2194) + .10 * ((i * 5) % 7), d: (end - 2194) / 5, x: { glideAt: tf ? (tf - 2196) / (end - 2194) : .5, acc: 1, vib: (c.st.vib ?? 16) * 1.1, vibOn: .5 + .12 * (i % 5), scoop: -110, scoopT: .08, rel: .5, airTail: .45 } }], { ...c.st, eff: [.84, .78, .44], peak: .3, breathIn: 1, accent: 1, slowAtk: .10, onset: .05, open: .2, rasp: 0, drift: 1.8, inton: 20 }, { tract, seed: hash('wide' + k + i), synth: c.synth, hum: .02, detune: c.cents, wander: 9 }), bus: 'wide', id: 'f6', who: k, room: k + 'w' + i, pan: clamp(.8 * Math.sin(1 + i * 2.1), -.9, .9) }, -26 + (k === 'jinbe' ? 2 : 0)); out.push(s); });
  return out;
}
// ---------- small human sounds ----------
// free control: fn(t, u) gives {f0, ee, rd, asp, brth, rasp, velum, eff, w:{a..}, jit}
function raw(start, dur, tract, fn, seed, syn = {}) {
  const n = Math.ceil((dur + .25) * CR), F = v => new Float32Array(n).fill(v), s = { n, f0: F(200), ee: F(0), rd: F(1.6), asp: F(0), aspfl: F(.3), rasp: F(0), air: F(0), jit: F(1.4), tilt: F(4500), mvf: F(8500), bwo: F(160), bwc: F(40), eff: F(.6), velum: F(0), w: [F(0), F(0), F(0), F(0), F(0)], gest: [], fric: [], brth: F(0) };
  for (let i = 0; i < n; i++) { const t = i / CR, u = t / dur, q = u <= 1 ? fn(t, u) : fn(dur, 1), off = u <= 1 ? 1 : 0; s.f0[i] = q.f0 ?? 200; s.ee[i] = (q.ee ?? 0) * off; s.rd[i] = q.rd ?? 1.6; s.asp[i] = (q.asp ?? 0) * off; s.aspfl[i] = q.aspfl ?? .6; s.rasp[i] = q.rasp ?? 0; s.eff[i] = q.eff ?? .6; s.velum[i] = q.velum ?? 0; s.jit[i] = q.jit ?? 1.4; s.brth[i] = (q.brth ?? 0) * off; s.tilt[i] = q.tilt ?? 4500; const w = q.w || { a: 1 }; for (const v in w) s.w[VI[v]][i] = w[v]; }
  const r = synth(s, { tract, seed, hfj: .3, hfs: .5, fp: 5, fpd: .15, ...syn }); return { start: Math.round(start * SR) - r.pad, y: r.y };
}
const bell = (u, a, p = 1) => Math.pow(Math.max(0, Math.sin(Math.PI * Math.pow(clamp(u, 0, 1), a))), p);
function fxSmall() {
  const out = [], R = rng(4711), lt = leadTract(), kid = new Tract(11.3), nz = slowNoise(55), T = f => f / FPS;
  // child: caught breaths (three quick pulls), the wail, caught breaths, a smaller wail
  const pulls = (f, n, a) => { const gap = .118, dur = gap * n + .06, ks = Array.from({ length: n }, () => .7 + .5 * R.u()); out.push({ ...level(raw(T(f), dur, kid, (t, u) => { const k = Math.min(n - 1, Math.floor(t / gap)), v = (t - k * gap) / gap, e = bell(clamp(v * 1.1, 0, 1), .75, 2.2) * ks[k]; return { brth: .30 * e * (1 + .5 * nz(t, 55)), asp: .05 * e, aspfl: 1, f0: 300 + 160 * v, ee: .05 * e * smooth((v - .35) / .2) * (1 - smooth((v - .7) / .2)), rd: 2.4, jit: 3, rasp: .2, w: { a: .25 + .45 * v, e: .25, u: .5 - .45 * v }, eff: .25 + .7 * v, velum: .35 * (1 - v) }; }, 11 + f, { aspHp: 350, aspLp: 3600 }), -43 + 6 * Math.log2(a + .2)), bus: 'sob', g: 1 }); };
  const wail = (f, spec, a, sd) => out.push({ ...level(take(spec.map(([m, p, t, d, x]) => ({ mora: m, pitch: p, t: f2t(f) + t, d, x: { vib: 0, ...x } })), { rd: .30, breath: .9, jit: 2.2, drift: 3, tremor: 3.5, rasp: .10, accDb: 2, weak: 0, reduce: 0, inton: 30, breathIn: 0, eff: [.46, .80, .40], peak: .35, rel: .2, scoop0: -150, hvBase: 79, piv: 420, pivR: 380, wn: 26, rub: 0, nasal: .12 }, { tract: kid, seed: sd, synth: { fp: 61, fpd: .2, hfc: 7000 }, hum: 0 }), a), bus: 'sob', g: 1 });
  const ex = { acc: 1, scoop: -160, scoopT: .04, fall: 220, fallT: .12, rel: .06 };
  pulls(336, 2, 1.0); pulls(343, 1, .8); pulls(350, 3, 1.5);
  wail(360, [['u', 'G4', 0, .6, { acc: .4 }], ['wa', 'B4>G#4', .6, 1.9, { acc: 1.2, glideAt: .4, scoop: -120, rasp: .2, raspT: .4, fall: 160, flip: [[.10, .16, 240, .6]] }]], -28.5, 515);
  pulls(373.5, 2, 1.9); wail(379, [['a', 'A4>F#4', 0, .95, { ...ex }]], -30, 518); pulls(384, 1, 1.5); wail(387, [['a', 'G4>E4', 0, .8, { ...ex, airTail: .5 }]], -32, 517); pulls(391.5, 2, 1.3);
  // the laugh (f1) and its echoes
  const laugh = (f, spec, st, a, tr, seed, o = {}) => { let t = f2t(f); const notes = []; spec.forEach(([m, p, gap, d], i) => { notes.push({ mora: m, pitch: p, t: t + (i ? .10 * R.g() : 0), d: d * (.85 + .3 * R.u()), x: { acc: i === 0 ? 1.2 : .5 + .3 * R.u(), scoop: 60 + 60 * R.u(), scoopT: .03, fall: 90, rel: .05, vib: 0, nobreath: i > 0 ? 1 : 0 } }); t += gap; }); const s = take(notes, { ...st, rel: .05 }, { tract: tr, seed, synth: o.synth || LEAD.synth, hum: .006, lag: 0, detune: 0 }); out.push({ ...level(s, a), bus: o.bus || 'laugh', pan: o.pan ?? 0, g: 1 }); };
  const jo = { ...STYLE.f2, eff: [.78, .92, .70], sib: .85, breath: .8, rd: .45, breathIn: 0, belt: 0, drift: 2, cons: .8, rasp: .06, lilt: 0, jit: 1.6, accent: .6, nasal: .05, col: { i: { e: .12 } } };
  laugh(1910, [['shi', 'A4', 1.55, .85], ['shi', 'G4', 1.45, .75], ['hi', 'G4', 1.25, .66], ['ha', 'F#4', 1.25, .66], ['ha', 'E4', 1.35, .66], ['ha', 'C#4', 1.6, .8]], { ...jo, eff: [.80, .82, .46], cons: 1.0, sib: .7, fricDb: -9, accDb: 2 }, -21, lt, 71);
  laugh(2203, [['a', 'A4', 1.1, .80], ['ha', 'C#5', 1.0, .62], ['ha', 'B4', .98, .58], ['ha', 'A4', 1.0, .58], ['ha', 'G#4', 1.06, .58], ['ha', 'F#4', 1.16, .60], ['ha', 'E4', 1.3, .62], ['ha', 'C#4', 1.3, .7]], { ...jo, eff: [.92, .96, .36], open: .45, breath: .9 }, -23, lt, 79);
  // sharp breath in at the grip; deep breath before the laugh; crew laughter on the deck
  const inhale = (f0, f1, a, sharp, w, seed) => out.push({ ...raw(T(f0), (f1 - f0) / FPS, lt, (t, u) => ({ brth: a * (sharp ? Math.pow(u, 1.2) * (1 - smooth((u - .93) / .07)) : bell(u, 1.5, .9)) * (1 + .18 * nz(t + 3, 30)), asp: .05 * a * Math.pow(u, 1.5) * (1 - smooth((u - .92) / .08)), f0: 180, w, eff: .4 + .8 * u, aspfl: 1 }), seed), bus: 'breath', g: 1 });
  inhale(1846, 1855.5, .9, true, { a: .6, e: .4 }, 83); inhale(1886, 1907, .40, false, { o: .3, a: .5, u: .2 }, 89);
  const ch = [['chopper', 2270, [['ha', 'D5', 1, .55], ['ha', 'C5', .95, .55], ['ha', 'B4', 1.0, .55], ['ha', 'A4', 1, .5]]], ['usopp', 2272, [['a', 'E4', 1.1, .7], ['ha', 'F#4', 1.0, .55], ['ha', 'D4', 1.0, .55], ['ha', 'B3', 1, .5]]], ['nami', 2276, [['ha', 'B4', .9, .5], ['ha', 'G#4', .9, .5], ['ha', 'F#4', .95, .5]]], ['franky', 2274, [['a', 'C#4', 1, .7], ['ha', 'B3', .95, .6], ['ha', 'A3', .95, .6], ['ha', 'F#3', 1, .6]]], ['brook', 2278, [['ho', 'G#4', .9, .55], ['ho', 'G#4', .9, .55], ['ho', 'F#4', 1, .55]]], ['zoro', 2281, [['ha', 'A3', 1.0, .55], ['ha', 'F#3', 1, .5]]], ['jinbe', 2279, [['ha', 'E3', 1.1, .65], ['ha', 'D3', 1.1, .65], ['ha', 'B2', 1.1, .6]]], ['robin', 2283, [['hu', 'G4', .9, .5], ['hu', 'F4', .9, .5]]]];
  for (const [k, f, sp] of ch) { const c = CREW[k]; laugh(f, sp, { ...c.st, eff: [.5, .56, .34], breath: .9, rd: (c.st.rd ?? 0) + .25, breathIn: 0, cons: .85, vib: 0, rasp: .04, jit: 1.7, accent: .5, belt: 0 }, -36, tractOf(k), hash(k) + 7, { synth: c.synth, bus: 'chuckle', pan: c.pan }); }
  return out;
}
function fxRoar() {
  const R = rng(99), takes = [], SH = [['u', 'o'], ['wa', 'a'], ['o', 'o'], ['yo', 'o'], ['ya', 'a'], ['he', 'e'], ['u', 'a'], ['i', 'e']], nm = m => ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B'][m % 12] + (Math.floor(m / 12) - 1);
  for (let i = 0; i < 52; i++) { const fem = i % 6 === 5, Lt = fem ? 13.8 + 1.4 * R.u() : 16.2 + 3.4 * R.u(), base = Math.round((fem ? 66 : 58 - (Lt - 17.5) * 2.2) + 4.5 * R.g()), up = base - 2 - Math.round(6 * R.u()), [m0, m1] = SH[i % 8], L = .25 + .75 * R.u(); takes.push(level(take([{ mora: m0, pitch: nm(base), t: 5, d: .4 + .4 * R.u(), x: { acc: 1, scoop: -200 - 250 * R.u(), scoopT: .08, vib: 0 } }, { mora: m1, pitch: nm(base) + '>' + nm(up), t: 5.6, d: L / E8, x: { slur: 1, acc: .8, glideAt: .1 + .3 * R.u(), vib: 0, rasp: .1 + .35 * R.u(), raspT: .8, fall: 200 + 400 * R.u(), fallT: .25, rel: .25, bend: [[.2 + .3 * R.u(), 120 * R.g(), .09]] } }], { ...YELL, col: {}, rd: -.2 + .4 * R.u(), rasp: .25, inton: 60, rel: .25 }, { tract: new Tract(Lt), seed: 1200 + i, synth: { fp: 200 + i * 7, fpd: .35, hfc: 5200 } }), -24)); }
  const N = Math.round(4.4 * SR), L = new Float32Array(N), Rr = new Float32Array(N), start = Math.round(131 / FPS * SR);
  for (let k = 0; k < 240; k++) { const s = takes[k % takes.length], near = k < 46, rate = .92 + .18 * R.u(), far = near ? .05 + .25 * R.u() : .35 + .65 * R.u(), dl = Math.round((near ? .03 + 1.5 * Math.pow(R.u(), 1.4) : .02 + .34 * Math.abs(R.g()) + (k % 4 === 0 ? .7 * R.u() : 0)) * SR), g = (near ? .75 : .34 * (1.2 - far)) * (.5 + .5 * R.u()), pan = R.u(), gl = Math.cos(pan * Math.PI / 2) * g, gr = Math.sin(pan * Math.PI / 2) * g, a = Math.exp(-2 * Math.PI * (2200 + 6500 * (1 - far)) / SR), it = Math.round((pan - .5) * .0012 * SR); let lp = 0; for (let i = 0; i < s.y.length / rate - 2; i++) { const x = i * rate, j = x | 0, u = x - j; lp = (s.y[j] * (1 - u) + s.y[j + 1] * u) * (1 - a) + lp * a; const o = i + dl, oL = o + Math.max(0, it), oR = o + Math.max(0, -it); if (oL < N) L[oL] += lp * gl; if (oR < N) Rr[oR] += lp * gr; } }
  const e = 1 / Math.max(1e-9, rmsOf(L)) * db(-16); for (let i = 0; i < N; i++) { const f = 131 + i / SPFR, g = e * (1 - .85 * smooth((f - 152) / 40)) * smooth((f - 130.5) / 3); L[i] *= g; Rr[i] *= g; } return [{ start, y: L, y2: Rr, bus: 'roar', g: 1 }];
}
function fxCrowd() {                                              // wordless open fifth, hundreds far away: D3 A3 D4 A4
  const R = rng(5150), N = Math.round(3.6 * SR), L = new Float32Array(N), Rr = new Float32Array(N), start = Math.round(146 / FPS * SR), th = [['D3', 19.2], ['D3', 18.3], ['A3', 17.6], ['A3', 18.8], ['D4', 17.0], ['D4', 15.0], ['A4', 14.6], ['A4', 13.8], ['A3', 18.0], ['D4', 16.4], ['D3', 19.8], ['A4', 14.2], ['D4', 17.3], ['A3', 16.8], ['D4', 14.4], ['A3', 18.4], ['D3', 17.9], ['A4', 13.2], ['D4', 18.2], ['A3', 17.0], ['D3', 18.8], ['A4', 15.1], ['D4', 15.8], ['A3', 19.0]];
  th.forEach(([p, Lt], i) => { const s = level(take([{ mora: 'a', pitch: p, t: f2t(155) + 1.2 * R.u(), d: 10.4, x: { acc: 0, vib: 14, vibOn: .7, scoop: -50, rel: .5, cresc: .12 } }], { rd: .45, breath: .7, jit: 1.5, drift: 2.6, inton: 25, accent: 0, slowAtk: .35 + .6 * R.u(), breathIn: 0, eff: [.40, .62, .46], rel: .6, col: { a: { o: .15 * R.u() } }, hvBase: Lt < 15.5 ? 70 : 65 }, { tract: new Tract(Lt), seed: 1700 + i, hum: .03, detune: 10 * R.g(), wander: 9, synth: { fp: 300 + i * 5, fpd: .35, hfc: 5600 } }), -25); for (let c = 0; c < 4; c++) { const rate = Math.pow(2, 12 * R.g() / 1200), dl = Math.round(.10 * R.u() * SR), pan = R.u(), g = .5 + .5 * R.u(), gl = Math.cos(pan * Math.PI / 2) * g, gr = Math.sin(pan * Math.PI / 2) * g, o0 = s.start - start + dl; for (let k = 0; k < s.y.length / rate - 2; k++) { const x = k * rate, j = x | 0, u = x - j, v = s.y[j] * (1 - u) + s.y[j + 1] * u, o = o0 + k; if (o >= 0 && o < N) { L[o] += v * gl; Rr[o] += v * gr; } } } });
  const e = db(-25) / Math.max(1e-9, rmsOf(L)); for (let i = 0; i < N; i++) { L[i] *= e; Rr[i] *= e; } return [{ start, y: L, y2: Rr, bus: 'crowdA', g: 1 }];
}
// ---------- tasks ----------
const D8 = () => [{ mora: 'o', pitch: 'B3', t: 312, d: 2, x: { scoop: -120, scoopT: .09, acc: .5, vib: 0 } }, { mora: 'to', pitch: 'A3', t: 314, d: 1, x: { dt: .02 } }, { mora: 'ga', pitch: 'F#3', t: 315, d: 3, x: { dt: .02, vib: 0, airTail: .4, fall: 30 } }, { mora: 'ki', pitch: 'G3', t: 318, d: .5, x: { acc: .9, gain: 3, rasp: .15 } }, { mora: 'e', pitch: 'A3>E4', t: 318.5, d: .55, x: { acc: .8, gain: 7, eff: .30, rd: -.2, glideAt: .1, flip: [[.03, .07, 300, .2]], rasp: .3, raspT: .5 } }, { mora: 'a', pitch: 'A#4>A3', t: 319.05, d: 2.3, x: { slur: 1, acc: 0, gain: 12, eff: .80, rd: -.45, glideAt: .34, rasp: .35, raspT: .12, vib: 0, rel: .3, airTail: .6, flip: [[.155, .21, 430, 1.0], [.27, .30, 260, 1.0]], bend: [[.05, -120, .03], [.10, 70, .02]] } }];
export const TASKS = {
  lead_b: () => { const tr = leadTract(); return [...['b1', 'b2', 'b3', 'b4', 'b5'].map(id => leadLine(id, { tract: tr })), leadLine('c10', { tract: tr, o: { post: mouthful } })]; },
  lead_c: () => { const tr = leadTract(); return [...['c13', 'c15'].map(id => leadLine(id, { tract: tr })), level({ ...leadDon(), bus: 'lead', id: 'solo', act: 'c' }, -14.5), ...SCORE.shouts.filter(s => s.who === 'all').map((s, i) => donShout('lead', s.f, i, tr))]; },
  lead_d: () => { const tr = leadTract(), wt = new Tract(16.2, { r: 900, c1: 9 }); return [...['d1', 'd2', 'd3', 'd4', 'd5', 'd7'].map(id => leadLine(id, { tract: tr })), (() => { const q = take(D8(), { ...STYLE.d8, rel: .3 }, { tract: tr, seed: 606 + hash('d8'), synth: LEAD.synth, hum: .006, room: .34 }), cutI = 1586 * SPFR - q.start, g = db(LEVEL.d8) / rmsOf(q.y.subarray(0, cutI)); let m = 0; const i2 = 1595.5 * SPFR - q.start, i3 = 1599 * SPFR - q.start; for (let i = i2; i < i3 && i < q.y.length; i++) m += q.y[i] ** 2; const gs = db(SCREAM) / Math.sqrt(m / (i3 - i2) + 1e-12); for (let i = 0; i < q.y.length; i++) { const f = (q.start + i) / SPFR, u = smooth((f - 1590) / 5.5); q.y[i] *= Math.exp(Math.log(g) * (1 - u) + Math.log(gs) * u) * db(-12 * smooth((f - 1599.6) / 5.4)); } return { ...q, bus: 'lead', id: 'd8', act: 'd' }; })(), leadLine('d6', { tract: wt, o: { whisper: true, hum: .004 }, notes: merged(LN.d6, { 0: { acc: .3, vib: 0 }, 2: { acc: .3, dt: .03 }, 3: { vib: 0, rel: .4, dd: -.18, dt: .03 } }), synth: { aspHp: 500, aspLp: 6000 } })]; },
  lead_f: () => { const tr = leadTract(); return ['f2', 'f3', 'f4', 'f5'].map(id => leadLine(id, { tract: tr, notes: id === 'f2' ? merged(LN.f2, ACT.f2).slice(6) : undefined })); },
  lead_g: () => { const tr = leadTract(); return ['g1', 'g2', 'g3'].map(id => leadLine(id, { tract: tr })); },
  fx_small: fxSmall, fx_roar: fxRoar, fx_crowd: fxCrowd,
};
for (const k of NAMES) TASKS[k] = () => crewTask(k);

// ---------- mix ----------
const CDIR = new URL('../cache/voice/', import.meta.url).pathname;
const save = (k, segs) => { fs.mkdirSync(CDIR, { recursive: true }); const meta = [], bufs = []; let o = 0; for (const s of segs) { const { y, y2, ...m } = s; meta.push({ ...m, off: o, len: y.length, st: !!y2 }); bufs.push(Buffer.from(y.buffer, y.byteOffset, y.byteLength)); o += y.length; if (y2) { bufs.push(Buffer.from(y2.buffer, y2.byteOffset, y2.byteLength)); o += y2.length; } } fs.writeFileSync(CDIR + k + '.f32', Buffer.concat(bufs)); fs.writeFileSync(CDIR + k + '.json', JSON.stringify(meta)); };
const load = k => { const meta = JSON.parse(fs.readFileSync(CDIR + k + '.json')), b = fs.readFileSync(CDIR + k + '.f32'), f = new Float32Array(b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength)); return meta.map(m => ({ ...m, y: f.subarray(m.off, m.off + m.len), y2: m.st ? f.subarray(m.off + m.len, m.off + 2 * m.len) : null })); };
const Z = () => new Float32Array(TOTAL), ST = () => [Z(), Z()];
function reflections(id) { const R = rng(hash('room' + id)); return Array.from({ length: 7 }, (_, i) => ({ d: Math.round((.0035 + .030 * Math.pow((i + R.u()) / 7, 1.15)) * SR), g: db(-8.5 - 9 * i / 7 - 4 * R.u()) * (R.u() < .5 ? -1 : 1), p: R.u(), a: Math.exp(-2 * Math.PI * (1500 + 2600 * R.u()) / SR) })); }
function put(buf, s, pan = 0, g = 1) { const p = (pan + 1) / 2, gl = Math.cos(p * Math.PI / 2) * g * 1.2, gr = Math.sin(p * Math.PI / 2) * g * 1.2, d = Math.round(Math.abs(pan) * .00055 * SR), dl = pan > 0 ? d : 0, dr = pan < 0 ? d : 0; for (let i = 0; i < s.y.length; i++) { const j = s.start + i; if (j < 0) continue; if (j + dl < TOTAL) buf[0][j + dl] += s.y[i] * gl; if (j + dr < TOTAL) buf[1][j + dr] += (s.y2 ? s.y2[i] : s.y[i]) * gr; }
  if (s.room) for (const r of reflections(s.room)) { let lp = 0; const l = Math.cos(r.p * Math.PI / 2) * r.g * g * 1.2, rr = Math.sin(r.p * Math.PI / 2) * r.g * g * 1.2; for (let i = 0; i < s.y.length; i++) { lp = s.y[i] * (1 - r.a) + lp * r.a; const j = s.start + i + r.d; if (j >= 0 && j < TOTAL) { buf[0][j] += lp * l; buf[1][j] += lp * rr; } } } }
const cut = (st) => { for (const c of st) { const fd = 72; for (let i = 0; i < fd; i++) c[CUT0 - fd + i] *= 1 - (i + 1) / fd; c.fill(0, CUT0, CUT1 - 12 * 480); for (let i = 0; i < 2400; i++) c[CUT1 - 5760 + i] *= smooth(i / 2400); } return st; };   // silence 1605 -> just before the first swell breath
const eqS = (st, bands) => st.map(c => fx.eq(c, bands));
function compS(st, o) { const m = Z(); for (let i = 0; i < TOTAL; i++) m[i] = .7 * Math.max(Math.abs(st[0][i]), Math.abs(st[1][i])); return st.map(c => fx.comp(c, { ...o, sc: m })); }
const sect = (t, tab) => { let i = 0; while (i < tab.length - 1 && tab[i + 1][0] <= t) i++; const a = tab[i], p = tab[i - 1]; if (!p) return a[1]; const u = clamp((t - a[0]) / .15, 0, 1); return p[1] + (a[1] - p[1]) * smooth(u); };
const rideF = (x, tab) => { const y = Z(); let g = 0; for (let i = 0; i < TOTAL; i++) { if ((i & 63) === 0) g = sect(i / SPFR, tab); y[i] = x[i] * g; } return y; };
export function mix(R, rooms = true) {
  const segs = Object.values(R).flat(), bus = n => segs.filter(s => s.bus === n);
  // LEAD
  const lm = Z(); for (const s of bus('lead')) for (let i = 0; i < s.y.length; i++) { const j = s.start + i; if (j >= 0 && j < TOTAL) lm[j] += s.y[i]; }
  let x = fx.eq(lm, [['hp', 75, .7, 0], ['ls', 260, .7, 2.6], ['pk', 1050, .8, -1.6], ['pk', 2950, .75, -2.0], ['hs', 8000, .7, 2]]);
  x = fx.comp(x, { thr: -20, ratio: 2.2, atk: .010, rel: .14, makeup: 1.6 }); x = fx.comp(x, { thr: -10, ratio: 2.4, atk: .004, rel: .10, makeup: 0 }); x = fx.warm(x, 1.35, .24); x = fx.deess(x, 6200, -33, 3);
  const lead = [Float32Array.from(x), Float32Array.from(x)];
  // CREW
  const call = ST(), grp = ST(), swell = ST(), wide = ST(), soft = ST(), don = ST();
  for (const s of bus('call')) put(call, s, s.pan); for (const s of bus('solo')) put(don, s, s.pan); for (const s of bus('don')) put(don, s, s.pan * .9); for (const s of bus('grp')) put(grp, s, s.pan); for (const s of bus('soft')) put(soft, s, s.pan); for (const s of bus('swell')) put(swell, s, s.pan); for (const s of bus('wide')) put(wide, s, s.pan);
  const pCall = compS(eqS(call, [['hp', 80, .7, 0], ['pk', 2900, .6, 3.5], ['hs', 8000, .7, 2]]), { thr: -20, ratio: 2.4, atk: .008, rel: .10, makeup: 2 });
  const pGrp = compS(eqS(grp, [['hp', 70, .7, 0], ['pk', 350, .8, -1.5], ['pk', 2800, .7, 2.5]]), { thr: -20, ratio: 2, atk: .010, rel: .14, makeup: 4.5 });
  const pDon = compS(eqS(don, [['hp', 70, .7, 0], ['ls', 300, .7, 4], ['pk', 2900, .6, 5], ['hs', 6000, .7, 3]]), { thr: -18, ratio: 3.5, atk: .001, rel: .09, makeup: .5 });
  const pSwC = st => compS(st, { thr: -20, ratio: 3, atk: .02, rel: .2, makeup: 1 });
  const pSoft = eqS(soft, [['hp', 110, .7, 0], ['hs', 5500, .7, -3]]), pWide = eqS(wide, [['hp', 45, .7, 0], ['hs', 5000, .7, -2]]);
  // under water at first: the mouth of the sea opens with each hand
  for (const c of swell) for (let i = CUT1 - 20000; i < 1880 * SPFR; i++) c[i] *= db(-8.5 * (1 - smooth((i / SPFR - 1728) / 116)));   // pp to ff across the nine
  const pSw = swell.map(c => { const y = Z(); let a = 0, b = 0; for (let i = CUT1 - 20000; i < 1890 * SPFR; i++) { const f = i / SPFR, u = smooth((f - 1725) / 70), fc = 4200 * Math.pow(4, u), k = 1 - Math.exp(-2 * Math.PI * fc / SR); a += (c[i] - a) * k; b += (a - b) * k; y[i] = b * (1 - u) + (.35 * b + .65 * c[i]) * u; } return y; });
  const pSw2 = pSwC(pSw); const crew = ST(); for (let c = 0; c < 2; c++) for (let i = 0; i < TOTAL; i++) crew[c][i] = pCall[c][i] + pGrp[c][i] + pDon[c][i] + pSoft[c][i] + pSw2[c][i] + pWide[c][i];
  // FX
  const mk = (n, g, pan) => { const b = ST(); for (const s of bus(n)) put(b, s, s.y2 ? 0 : s.pan ?? pan ?? 0, g * (s.g ?? 1)); return b; };
  const roar = mk('roar', 1), crowd = mk('crowdA', 1), sob = mk('sob', 1, .08), laugh = mk('laugh', 1, 0), breath = mk('breath', 1, 0), chk = mk('chuckle', 1);
  const norm = (st, target) => { const r = rmsOf(st[0]); const g = db(target) / r; for (const c of st) for (let i = 0; i < TOTAL; i++) c[i] *= g; return st; };
  norm(breath, -27);
  const eRoar = eqS(roar, [['hp', 160, .7, 0], ['lp', 7000, .6, 0]]), eCr = eqS(crowd, [['hp', 90, .7, 0], ['hs', 4000, .7, 0]]), eSob = eqS(sob, [['hp', 160, .7, 0], ['hs', 4500, .7, -6]]), eL = compS(eqS(laugh, [['hp', 110, .7, 0], ['hs', 8000, .7, 1]]), { thr: -22, ratio: 2.2, makeup: 1.5 }), eB = eqS(breath, [['hp', 200, .7, 0], ['lp', 5200, .7, 0]]), eC = eqS(chk, [['hp', 120, .7, 0]]);
  const fxs = ST(); for (let c = 0; c < 2; c++) for (let i = 0; i < TOTAL; i++) fxs[c][i] = eRoar[c][i] + eCr[c][i] + eSob[c][i] + eL[c][i] + eB[c][i] + eC[c][i];
  for (const s of [lead, crew, fxs]) cut(s);
  // premix: rooms on sends
  const mono = st => { const m = Z(); for (let i = 0; i < TOTAL; i++) m[i] = .5 * (st[0][i] + st[1][i]); return m; }, pm = ST(), addS = (d, s, g) => { for (let c = 0; c < 2; c++) for (let i = 0; i < TOTAL; i++) d[c][i] += s[c][i] * g; };
  for (const s of [lead, crew, fxs]) addS(pm, s, 1);
  if (rooms) {
    const lead1 = mono(lead), wetL = rideF(lead1, [[0, .12], [450, .18], [540, .10], [930, .16], [1140, .26], [1560, .34], [1605, 0], [1900, .16], [2160, .24], [2250, .22], [2400, .18]]), crewM = mono(crew), wetC = rideF(crewM, [[0, .12], [930, .22], [1720, .55], [1860, .2], [2185, .5], [2260, .26]]), fxM = mono(fxs), wetF = rideF(fxM, [[0, .4], [215, .12], [1840, .12], [2195, .4], [2260, .15]]);
    addS(pm, fx.plate(wetL, { decay: .66, pre: .028, damp: 3200, hicut: 5200, lowcut: 220 }), .9); addS(pm, fx.plate(wetC, { decay: .72, pre: .02, damp: 3000, hicut: 5200, lowcut: 160, size: 1.15 }), .9); addS(pm, fx.plate(wetF, { decay: .74, pre: .03, damp: 2600, hicut: 4200, lowcut: 200, size: 1.2 }), .8);
    addS(pm, fx.room(lead1, 5, { n: 30, lp: 3400, len: .034 }), .06); addS(pm, fx.room(crewM, 17, { n: 36, lp: 3200, len: .05 }), .22); cut(pm); for (const c of pm) { c.fill(0, CUT0, CUT1 - 5760); }
  }
  return { lead, crew, fxs, pm };
}
if (!isMainThread) { parentPort.on('message', k => { if (k === null) process.exit(0); const segs = TASKS[k](); save(k, segs); parentPort.postMessage({ k, n: segs.length }); }); }
else if (process.argv[1] && import.meta.url.endsWith(process.argv[1].split('/').pop())) {
  const t0 = Date.now(), all = Object.keys(TASKS), want = process.argv[2] && process.argv[2] !== 'all' ? process.argv[2] === 'mix' ? [] : process.argv[2].split(',') : all, jobs = 3; let next = 0, done = 0;
  const order = want.slice().sort((a, b) => (NAMES.includes(b) ? 1 : 0) - (NAMES.includes(a) ? 1 : 0));
  if (order.length) await new Promise((res, rej) => { const nW = Math.min(jobs, order.length); for (let w = 0; w < nW; w++) { const W = new Worker(new URL(import.meta.url)); W.on('error', rej); W.on('message', m => { done++; if (next < order.length) W.postMessage(order[next++]); else W.postMessage(null); if (done === order.length) res(); }); W.postMessage(order[next++]); } });
  console.log('sung', order.length, 'tasks in', ((Date.now() - t0) / 1000).toFixed(1), 's');
  const R = {}; for (const k of all) { try { R[k] = load(k); } catch { console.log('missing', k); } }
  const { lead, crew, fxs, pm } = mix(R), O = new URL('../out/', import.meta.url).pathname; fs.mkdirSync(O, { recursive: true });
  let pk = 0; for (const c of pm) for (const v of c) pk = Math.max(pk, Math.abs(v)); const g = Math.min(1, db(-.5) / pk) ; const gs = Math.min(1, db(2.0) / pk); for (const st of [lead, crew, fxs, pm]) for (const c of st) for (let i = 0; i < TOTAL; i++) c[i] *= gs; fx.limit(pm, db(-1));
  const W = (n, st) => { let p = 0, e = 0; for (const c of st) for (const v of c) { p = Math.max(p, Math.abs(v)); e += v * v; if (!Number.isFinite(v)) throw Error('NaN in ' + n); } if (p > .999) fx.limit(st, db(-.3)); writeWav(O + n + '.wav.part', st); fs.renameSync(O + n + '.wav.part', O + n + '.wav'); console.log(n, 'peak', todb(p).toFixed(1), 'dB  rms', (10 * Math.log10(e / TOTAL / 2 + 1e-12)).toFixed(1), 'samples', st[0].length); };
  W('voice_lead', lead); W('voice_crew', crew); W('voice_fx', fxs); W('voice_mix', pm); console.log('premix gain', todb(gs).toFixed(1), 'total', ((Date.now() - t0) / 1000).toFixed(1), 's');
}
