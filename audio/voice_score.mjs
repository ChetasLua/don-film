// voice_score.mjs : the sung score of DON!, from docs/SCORE.md.  node audio/voice_score.mjs -> audio/score.json
// time: t, d in EIGHTHS from film start (1 eighth = 5 frames = .2083333 s = 10000 samples at 48 kHz); f = frame of the vowel onset = 5 t; s, dur in seconds. Every note starts on a frame multiple of 5.
import fs from 'node:fs';
const E = 5 / 24, L = [];
function line(id, bar, voice, ja, en, txt, extra = {}) {
  let t = bar * 6; const notes = []; for (const tok of txt.trim().split(/\s+/)) { let [m, p, d] = tok.split(':'); if (m === '-') { t += +p; continue; } d = +(d ?? 1); const slur = m.endsWith('~'); if (slur) m = m.slice(0, -1); const mora = m === 'n' ? 'N' : m === 't' ? 'Q' : m; notes.push({ t, d, f: t * 5, s: +(t * E).toFixed(5), dur: +(d * E).toFixed(5), pitch: p, mora, ...(slur ? { slur: true } : {}) }); t += d; }
  L.push({ id, bar, voice, ja, en, t0: notes[0].t, t1: t, ...extra, notes });
}
line('b1', 7, 'lead', '笑う男の帽子が', 'The hat of a man who laughed', 'wa:A3:1 ra:D4:1 u:E4:1 o:F#4:1 to:E4:1 ko:D4:1 no:E4:1 bo:F#4:1 o~:F#4:1 shi:A4:1 ga:F#4:2');
line('b2', 9, 'lead', '泣くちびの頭へ', 'came down on a crying kid', 'na:G4:2 ku:F#4:1 chi:E4:1 bi:D4:1 no:E4:1 a:F#4:1 ta:E4:1 ma:D4:1 e:B3:3');
line('b3', 11, 'lead', '深くて前が見えない', "too big, too deep: I can't see ahead", 'fu:A3:1 ka:B3:1 ku:D4:1 te:E4:2 ma:E4:1 e:F#4:1 ga:E4:1 mi:G4:1 e:F#4:1 na:E4:1 i:D4:1');
line('b4', 13, 'lead', '波が胸でドンと鳴る', 'but the waves go DON inside my chest', 'na:B3:1 mi:D4:1 ga:E4:1 mu:F#4:1 ne:G4:1 de:A4:1 do:B4:1 n:B4:1 to:A4:1 na:F#4:1 ru:E4:2');
line('b5', 15, 'lead', '海は広いぞ', 'The sea is wide!', 'u:A4:1 mi:A4:1 wa:B4:1 hi:A4:1 ro:F#4:1 i:E4:1 zo:D4:6 -:6');
const call = (id, bar, who, ja, en, txt) => line(id, bar, who, ja, en, txt, { call: true });
call('c1', 18, 'zoro', '迷うなよ', "Don't get lost", 'ma:D4 yo:D4 u:E4 na:F#4 yo:D4'); call('c2', 19, 'nami', 'みかん色', 'Tangerine-bright', 'mi:E4 ka:E4 n:F#4 i:G4 ro:E4'); call('c3', 20, 'usopp', '嘘も夢', 'Even lies are dreams', 'u:F#4 so:F#4 mo:G4 yu:A4 me:F#4');
call('c4', 21, 'sanji', '腹減った', 'Hungry? Eat', 'ha:G4 ra:G4 he:A4 t:B4 ta:G4'); call('c5', 22, 'chopper', '桜咲け', 'Bloom, cherry', 'sa:A4 ku:B4 ra:A4 sa:F#4 ke:E4'); call('c6', 23, 'robin', '手は花に', 'Hands into flowers', 'te:F#4 wa:F#4 ha:E4 na:D4 ni:F#4');
call('c7', 24, 'franky', '胸を張れ', 'Chest out!', 'mu:G4 ne:G4 wo:A4 ha:B4 re:D5'); call('c8', 25, 'brook', '骨も歌', 'Bones sing too', 'ho:B4 ne:B4 mo:A4 u:G4 ta:B4'); call('c9', 26, 'jinbe', '舵を取れ', 'Take the helm', 'ka:A4 ji:A4 wo:A4 to:B4 re:C#5');
line('c10', 27, 'lead', '肉は俺のだ', 'The meat is MINE', 'ni:D4:1 ku:D4:1 wa:E4:1 o:F#4:1 re:F#4:1 no:E4:1 da:A4:6');
line('c13', 31, 'all', '風よ吹け 帆を張れ', 'Blow, wind! Spread, sail!', 'ka:F#4:1 ze:G4:1 yo:A4:1 fu:B4:2 ke:A4:1 ho:D5:2 wo:B4:1 ha:A4:1 re:F#4:2');
line('c15', 35, 'all', 'どこまでも行こうぜ', 'On, as far as it goes!', 'do:D5:1 ko:C#5:1 ma:B4:1 de:A4:1 mo:F#4:2 i:G4:1 ko:A4:1 u~:A4:1 ze:B4:3 -:6');
line('d1', 38, 'lead', '負けない もう 負けない', "I won't lose. Not again", 'ma:B3:1 ke:D4:1 na:F#4:1 i:E4:3 mo:D4:2 u~:D4:1 -:3 ma:F#4:1 ke:E4:1 na:D4:1 i:B3:3');
line('d2', 41, 'lead', '助けてと言えた', 'At last I said: help me', 'ta:D4:1 su:D4:1 ke:E4:1 te:F#4:2 to:E4:1 i:D4:1 e:E4:1 ta:F#4:4');
line('d4', 43, 'lead', '生きたいと叫んだ', 'I want to live, I cried', 'i:F#4:2 ki:G4:1 ta:A4:2 i:A4:1 to:F#4:1 sa:G4:1 ke:A4:1 n:A4:1 da:B4:2');
line('d3', 45, 'lead', '雪の中 船が泣く', 'In the snow a ship is crying', 'yu:B3:1 ki:D4:1 no:E4:1 na:F#4:1 ka:E4:2 fu:D4:1 ne:E4:1 ga:D4:1 na:B3:1 ku:A3:2');
line('d6', 47, 'lead', '何にも', 'Nothing happened', 'na:B3:1 n:B3:1 ni:C#4:1 mo:B3:3', { whisper: true });
line('d5', 48, 'lead', 'ひとり またひとり', 'One. And then another', 'hi:F#4:1 to:F#4:1 ri:F#4:1 -:3 ma:E4:1 ta:D4:2 hi:E4:1 to:D4:1 ri:C#4:1');
line('d7', 50, 'lead', '紙が燃えてゆく', 'The paper is burning down', 'ka:D4:1 mi:D4:1 ga:E4:1 mo:F#4:2 e:E4:1 te:D4:1 yu:C#4:1 ku:B3:4');
line('d8', 52, 'lead', '音が消えた', 'The sound is gone', 'o:B3:2 to:A3:1 ga:F#3:3 ki:G3:2 e:F#3:1 ta:F#3:3', { cutFrame: 1605, note: 'the last mora "ta" is written but never heard: all voices are cut to digital silence at frame 1605; the performed "ki-e" tears upward, breaks, and falls away into breath as the cut comes' });
line('f2', 64, 'all', '笑え 笑え 胸の太鼓', 'Laugh, laugh, drum of my chest', 'wa:B3:1 ra:E4:1 e:F#4:1 wa:G#4:1 ra:F#4:1 e:E4:1 mu:F#4:1 ne:G#4:1 no:B4:1 ta:C#5:1 i:B4:1 ko:G#4:1');
line('f3', 66, 'all', '夜を叩き起こせ', 'Beat the night awake', 'yo:A4:1 ru:G#4:1 wo:F#4:1 ta:G#4:1 ta:A4:1 ki:B4:1 o:C#5:2 ko:B4:1 se:G#4:3');
line('f4', 68, 'all', '跳んで跳ねて回せ', 'Jump, bounce, swing it', 'to:C#5:1 n:C#5:1 de:B4:1 ha:A4:1 ne:G#4:1 te:F#4:1 ma:E4:1 wa:F#4:1 se:G#4:4');
line('f5', 70, 'all', '朝を連れて来い', 'Bring the morning here', 'a:B4:2 sa:C#5:1 wo:B4:3 tsu:G#4:1 re:A4:1 te:B4:1 ko:E5:3 i:E5:6');
line('g1', 75, 'lead+soft crew', '目を開ければ みんないる', "I open my eyes: everyone's here", 'me:E4:1 wo:F#4:1 a:G#4:1 ke:B4:1 re:A4:1 ba:G#4:1 mi:F#4:1 n:F#4:1 na:G#4:1 i:A4:1 ru:F#4:2');
line('g2', 77, 'all', 'ひとりじゃないから 朝が来る', "I'm not alone, so morning comes", 'hi:G#4:1 to:A4:1 ri:B4:1 ja:C#5:2 na:B4:1 i:A4:1 ka:G#4:1 ra:F#4:1 a:G#4:1 sa:A4:1 ga:B4:1 ku:C#5:2 ru:B4:4');
line('g3', 80, 'lead', '笑う男の帽子が', 'The hat of a man who laughed...', 'wa:B3:1 ra:E4:1 u:F#4:1 o:G#4:1 to:F#4:1 ko:E4:1 no:F#4:1 bo:G#4:1 o~:G#4:1 shi:B4:1 ga:G#4:2', { spoken: true });
const shouts = []; ['c1', 'c2', 'c3', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9'].forEach((c, i) => shouts.push({ id: c + '_don', f: (18 + i) * 30 + 25, who: 'all', mora: ['do', 'N'], en: 'DON!' }));
shouts.push({ id: 'c14_zoro', f: 990, who: 'zoro', mora: ['do', 'N'], en: 'DON!' }, { id: 'c14_luffy', f: 1005, who: 'lead', mora: ['do', 'N'], en: 'DON!' }, { id: 'c14_sanji', f: 1020, who: 'sanji', mora: ['do', 'N'], en: 'DON!' });
const SW = [['jinbe', 'D2', 'B1', 1838], ['zoro', 'D3', 'D#3', 1845], ['nami', 'A3', 'A3', 0], ['usopp', 'F#3', 'F#3', 0], ['sanji', 'D4', 'D#4', 1845], ['chopper', 'A4', 'A4', 0], ['robin', 'E3', 'F#3', 1833], ['franky', 'A2', 'B2', 1842], ['brook', 'B4', 'B4', 0]];
const swell = SW.map(([who, pitch, to, turn], k) => ({ k: k + 1, who, f: 1725 + 15 * k, end: 1855, mora: 'a', pitch, to, turnFrame: turn || null }));
// chords the group harmonies lean on, two per bar (eighths 1-3, 4-6); C 18-26 is SCORE.md's row, the rest is the default implied by the tune
const chords = { 18: ['D', 'D'], 19: ['A', 'A'], 20: ['D', 'D'], 21: ['G', 'G'], 22: ['D', 'D'], 23: ['Bm', 'Bm'], 24: ['G', 'G'], 25: ['Em', 'Em'], 26: ['A', 'A'], 31: ['D', 'G'], 32: ['G', 'D'], 35: ['D', 'Bm'], 36: ['Em', 'G'], 64: ['E', 'E'], 65: ['B', 'E'], 66: ['A', 'E'], 67: ['A', 'E'], 68: ['A', 'B'], 69: ['C#m', 'E'], 70: ['E', 'E'], 71: ['C#m', 'A'], 72: ['E', 'E'], 73: ['E', 'E'], 74: ['E', 'E'], 75: ['E', 'E'], 76: ['F#m', 'B'], 77: ['E', 'A'], 78: ['F#m', 'B'], 79: ['A', 'E'] };
const fx = [{ id: 'roar', f: 131, end: 185, en: 'crowd roar' }, { id: 'crowd_a', f: 155, end: 212, en: 'wordless open fifth D-A' }, { id: 'sob', f: 336, end: 398, en: 'child sob hitches, howl at 360' }, { id: 'swell', f: 1725, end: 1855, en: 'nine entries' }, { id: 'gasp', f: 1846, end: 1855, en: 'one sharp breath in' }, { id: 'breath', f: 1886, end: 1907, en: 'deep breath' }, { id: 'laugh', f: 1910, end: 1952, en: 'the laugh, over the first warae (crew only); the lead joins f2 at mune, frame 1950' }, { id: 'fall_laugh', f: 2203, end: 2246, en: 'laughing, falling' }, { id: 'wide_a', f: 2193, end: 2250, en: 'wide E major a' }, { id: 'chuckles', f: 2270, end: 2298, en: 'the crew breaks into laughter' }];
const out = { title: 'DON!', fps: 24, frames: 2544, seconds: 106, meter: '6/8', bpm_dotted: 96, eighth_frames: 5, eighth_s: E, bar_frames: 30, bars: 84, time_unit: 't and d are eighths from film start; f is the frame of the vowel onset (consonants come before it); s and dur are seconds. N = moraic nasal, Q = geminate, slur = continues the previous vowel', sections: [['A', 0, 'the man who laughed'], ['B', 7, 'the hat'], ['C', 18, 'nakama'], ['D', 38, 'loss'], ['E', 55, 'under'], ['F', 62, 'the drum'], ['G', 75, 'dawn']], chords, silence_frames: [1605, 1725], lines: L, shouts, swell, fx };
const enc = JSON.stringify(out, null, 1).replace(/\{\n\s+"t": ([\s\S]*?)\n\s+\}/g, (m) => m.replace(/\n\s+/g, ' '));
fs.writeFileSync(new URL('./score.json', import.meta.url), enc);
for (const l of L) { const ok = l.notes.every(n => Number.isInteger(n.t)), n = l.notes[l.notes.length - 1]; if (!ok) console.log('OFF GRID', l.id); console.log(l.id.padEnd(4), l.voice.padEnd(14), 'bars', l.bar, '-', ((l.t1) / 6).toFixed(2), 'frames', l.notes[0].f, '-', l.t1 * 5, l.notes.map(q => q.mora + (q.slur ? '~' : '')).join(' ')); }
