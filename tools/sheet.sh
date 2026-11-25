#!/bin/zsh
# tools/sheet.sh <shot> [n=6] [out]  evenly spaced frames of out/<shot>.mkv on one 1500 px sheet (ffmpeg only, no queue)
cd "${0:A:h}/.."; FF=.mm/env/bin/ffmpeg; FP=.mm/env/bin/ffprobe; s=$1; k=${2:-6}; o=${3:-qa/sheet_$s.jpg}
n=$($FP -v error -count_packets -select_streams v -show_entries stream=nb_read_packets -of csv=p=0 out/$s.mkv | tr -d ',')
st=$(( (n + k - 1) / k )); r=$(( (k + 2) / 3 ))
$FF -v error -y -i out/$s.mkv -vf "select='not(mod(n,$st))',scale=500:-1,tile=3x$r" -frames:v 1 -q:v 4 $o && echo $o
