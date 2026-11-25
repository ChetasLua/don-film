#!/bin/zsh
# tools/strip.sh <shot> a b [step=1]  consecutive frames a..b of out/<shot>.mkv, 4 per row, 1600 px (judging motion, no queue)
cd "${0:A:h}/.."; FF=.mm/env/bin/ffmpeg; s=$1; a=$2; b=$3; st=${4:-1}; c=$(( (b - a) / st + 1 )); r=$(( (c + 3) / 4 ))
$FF -v error -y -i out/$s.mkv -vf "select='between(n,$a,$b)*not(mod(n-$a,$st))',scale=400:-1,tile=4x$r" -frames:v 1 -q:v 4 qa/strip_$s.jpg && echo qa/strip_$s.jpg
