#!/usr/bin/zsh
#ls | cat -n | while read n f; do mv "$f" "${n}_.wav"; done

for file in *_.wav
do
	sox $file ${file[0,-5]}8k.wav rate 8k
done

#find . -type f ! -name '*8k.wav' -delete