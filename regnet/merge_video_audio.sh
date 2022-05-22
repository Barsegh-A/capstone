video_path=$1
audio_path=$2
output_path=$3

for v in {2657..2784}
do
	ffmpeg -i ${video_path}/video_0${v}.mp4 -i ${audio_path}/video_0${v}.wav -map 0:v -map 1:a -c:v copy -shortest ${output_path}/video_0${v}_hifigan.mp4
done