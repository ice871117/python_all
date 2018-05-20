from ffmpy import FFmpeg
import sys

inputFile = "/Volumes/SanDisk/downloaded_programs/11440971_为什么性骚扰叫「咸猪手」?_字媒体_历史人文_1"
outputFile = "/Volumes/SanDisk/11440971_为什么性骚扰叫「咸猪手」?_字媒体_历史人文.wav"
ffmpeg = FFmpeg( inputs={inputFile: None},outputs={outputFile: "-f wav -y"})
ffmpeg.run()
