# ドン! (DON!)

A 106-second *One Piece* fan film, drawn, animated, scored and sung entirely in code.
1920×1080, 24 fps, 2,544 frames, 44 shots, 6/8 at 96.

ドン is the sound of a drum and of a heart. The film is one heart: it starts under a scaffold, gathers a crew, stops, and is started again by nine hands. A straw hat carries it from head to head.

No image models, no image files, no samples, no text-to-speech, no fonts on screen: the picture is Python (numpy, opencv, scipy, pycairo, numba, Pillow) and the sound is zero-dependency JavaScript for Node.

Unofficial, non-commercial fan work. *One Piece* and its characters are by Eiichiro Oda; this project is not made or endorsed by the author or the rights holders. The song, its words and all drawings here are original code.

## Build

pycairo has no macOS wheel, so the environment comes from conda-forge ([micromamba](https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html)). Run from the repository root. Sound needs Node 24 or newer.

```bash
micromamba create -y -p .mm/env -c conda-forge python=3.12 numpy opencv pycairo scipy pillow numba ffmpeg
```

Picture: one shot per module in `shots/`; plates are cached on the first run, so it is slow the first time.

```bash
./.mm/env/bin/python tools/film.py --jobs 4
```

Voices, band, then effects and mix:

```bash
node audio/voice.mjs && node audio/band.mjs && node audio/mix.mjs
```

Join picture and sound into `dist/don.mp4`:

```bash
./.mm/env/bin/python tools/film.py --assemble
```

One frame, with a 1:1 crop:

```bash
./.mm/env/bin/python tools/still.py g1 22 --crop 900,400
```

## What is where

| Path | What it does |
|---|---|
| `film.py` | the timeline: every shot, its start and length |
| `op/hat.py` | the straw hat: a 3-D weave inked as a cel at any angle, and the view from under its brim |
| `op/sea.py` `ships.py` `seakit.py` `fleet3.py` | wave fields, wakes, spray; ships as 3-D hulls, sails and rigging |
| `op/don.py` `sumi.py` `bolt.py` | brush lettering, dry-brush ink, lightning and shock rings |
| `op/head3.py` `hands.py` | a face drawn once in front view, correct at any angle; hands as anatomy |
| `op/kit.py` `ink.py` `paint.py` `clouds.py` `comp.py` | cel ink, gouache, painted clouds, light and finishing |
| `chars/` | the crew and everyone else, one module each |
| `shots/` | the 44 shots, one module each |
| `audio/` | the song: score, a glottal-pulse voice with a 44-tube vocal tract, nine crew singers, a synthesized band, effects and the mix |
| `tools/` | stills, shot renders, film assembly, subtitles, review sheets |
