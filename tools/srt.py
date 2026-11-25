"""tools/srt.py -> dist/don.en.srt  English lyric lines from audio/score.json, never overlapping"""
import json, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')); D = json.load(open('audio/score.json'))
L = sorted([l for l in D['lines'] if l.get('notes') and l.get('en')], key=lambda l: l['notes'][0]['s'])
ts = lambda s: '%02d:%02d:%02d,%03d' % (s // 3600, s % 3600 // 60, int(s % 60), int(round((s - int(s)) * 1000)) % 1000)
o = []
for i, l in enumerate(L):
    a = max(0, l['notes'][0]['s'] - .04); b = max(n['s'] + n['dur'] for n in l['notes']) + .25
    if i + 1 < len(L): b = min(b, L[i + 1]['notes'][0]['s'] - .06)
    o.append('%d\n%s --> %s\n%s\n' % (i + 1, ts(a), ts(max(b, a + .4)), l['en']))
open('dist/don.en.srt', 'w').write('\n'.join(o)); print(len(o), 'lines')
