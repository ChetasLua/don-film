// The effects library of DON!  Original synthesis, no samples.   render(name, { seed, size, dur, v, chord, ... }) -> St { l, r, n, sync }
// reference level: -18 LUFS in the loudest 100 ms, peak <= -1.5 dBFS; `sync` = seconds from the first sample to the moment that sits on the frame.
//   node audio/sfx_lib.mjs list | one NAME out.wav '{"seed":2}' | demo
export { SFX, render, CH, def } from './sfx_core.mjs';
import { SFX, render } from './sfx_core.mjs';
import './sfx_1.mjs'; import './sfx_2.mjs'; import './sfx_3.mjs'; import './sfx_4.mjs';
import { SR, St, writeWav } from './sfx_dsp.mjs';
import path from 'node:path'; import { fileURLToPath } from 'node:url';
const here = path.dirname(fileURLToPath(import.meta.url));
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) { const [cmd, a, b, c] = process.argv.slice(2);
  if (cmd === 'list') for (const m of Object.values(SFX)) console.log(m.name.padEnd(16), m.d);
  else if (cmd === 'one') { const s = render(a, c ? JSON.parse(c) : {}); writeWav(b, s.l, s.r, 24); console.log(a, (s.n / SR).toFixed(2) + 's'); }
  else if (cmd === 'demo') { const all = []; let t = .3; for (const n of Object.keys(SFX)) { const s = render(n, {}); all.push([t, s]); console.log(t.toFixed(2).padStart(7), n); t += s.n / SR + .35; } const st = new St(t, Float32Array); for (const [at, s] of all) st.addSt(s, { at }); writeWav(path.join(here, '..', 'out', 'sfx_demo.wav'), st.l, st.r, 16); } }
