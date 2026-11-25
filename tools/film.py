"""python tools/film.py [--jobs 4] [--only a1,d4] [--assemble] [--crf 13]
Renders every shot to out/<id>.mkv (near-lossless; --only forces a list), then --assemble joins them with out/soundtrack.wav
and the English lyric track into dist/don.mp4 (1920x1080, 24 fps, H.264, AAC 320k, soft subtitles)."""
import sys, os, argparse, subprocess, time
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, ROOT); os.chdir(ROOT)
import film as FM
FF = '.mm/env/bin/ffmpeg'; PY = '.mm/env/bin/python'


def check():
    bad = []
    for s, a, n in FM.SHOTS:
        p = f'out/{s}.mkv'
        if not os.path.exists(p): bad.append((s, 'missing')); continue
        r = subprocess.run(['.mm/env/bin/ffprobe', '-v', 'error', '-count_packets', '-select_streams', 'v', '-show_entries', 'stream=nb_read_packets', '-of', 'csv=p=0', p], capture_output=True, text=True).stdout.strip().strip(',')
        if r != str(n): bad.append((s, r, n))
    return bad


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--jobs', type=int, default=4); ap.add_argument('--only'); ap.add_argument('--assemble', action='store_true'); ap.add_argument('--crf', default='13'); ap.add_argument('--out', default='dist/don.mp4')
    a = ap.parse_args()
    for d in ('cache', 'out', 'dist', 'qa'): os.makedirs(d, exist_ok=True)
    if not a.assemble or a.only:
        for s in (a.only.split(',') if a.only else [s for s, _, _ in FM.SHOTS]):
            if not a.only and os.path.exists(f'out/{s}.mkv'): continue
            t = time.time(); r = subprocess.run([PY, 'tools/render.py', s, '--jobs', str(a.jobs)], capture_output=True, text=True)
            print(s, 'ok' if r.returncode == 0 else 'FAIL ' + r.stderr[-600:], '%.0fs' % (time.time() - t), flush=True)
    if a.assemble:
        bad = check(); print('check', bad or 'all shots complete')
        if bad: sys.exit(1)
        subprocess.run([PY, 'tools/srt.py'], check=True)
        open('out/list.txt', 'w').write(''.join(f"file '{os.path.abspath('out/' + s + '.mkv')}'\n" for s, _, _ in FM.SHOTS))
        sys.exit(subprocess.run([FF, '-y', '-v', 'error', '-f', 'concat', '-safe', '0', '-r', '24', '-i', 'out/list.txt', '-i', 'out/soundtrack.wav', '-i', 'dist/don.en.srt', '-map', '0:v', '-map', '1:a', '-map', '2:s',
               '-vf', 'scale=in_range=tv:out_range=tv:in_color_matrix=bt601:out_color_matrix=bt709:flags=lanczos+accurate_rnd+full_chroma_int,format=yuv420p,setsar=1',
               '-c:v', 'libx264', '-preset', 'slow', '-crf', a.crf, '-profile:v', 'high', '-level', '4.2', '-g', '48', '-x264-params', 'aq-mode=3:deblock=-1,-1:psy-rd=1.0,0.15',
               '-color_range', 'tv', '-colorspace', 'bt709', '-color_primaries', 'bt709', '-color_trc', 'bt709', '-c:a', 'aac', '-b:a', '320k', '-c:s', 'mov_text', '-metadata:s:s:0', 'language=eng', '-disposition:s:0', '0', '-movflags', '+faststart', a.out]).returncode)
