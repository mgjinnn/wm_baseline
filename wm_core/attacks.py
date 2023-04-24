import os
import random


def attack0(video_name, video_path, video_attacked_folder):
    ouput_path = f'{video_attacked_folder}/{video_name}.mp4'
    st = random.randint(1, 15)
    cmd = f'ffmpeg -i {video_path} -ss {st} -t 15 -vf noise=alls=50:allf=u+t -s 640x360 -an {ouput_path} -v warning'
    os.system(cmd)

def attack1(video_name, video_path, video_attacked_folder):
    ouput_path = f'{video_attacked_folder}/{video_name}.mp4'
    st = random.randint(1, 15)
    cmd = f'ffmpeg -i {video_path} -ss {st} -t 15 -vf crop=600:400 -c:v libx264 -crf 30 -r 20 -an -y {ouput_path} -v warning'
    os.system(cmd)

def attack2(video_name, video_path, video_attacked_folder):
    ouput_path = f'{video_attacked_folder}/{video_name}.mp4'
    st = random.randint(1, 15)
    cmd = f'ffmpeg -i {video_path} -i logo.png -i logo.png -i logo.png -filter_complex "overlay=x=20:y=200, overlay=x=20:y=350, overlay=x=420:y=350" -ss {st} -t 15 -c:v libx264 -crf 38 -r 20 -an {ouput_path} -v warning'
    os.system(cmd)

def attack3(video_name, video_path, video_attacked_folder):
    ouput_path = f'{video_attacked_folder}/{video_name}.mp4'
    cmd = f'ffmpeg -i bg.mp4 -i {video_path} -filter_complex "[1:v]scale=iw/2:ih/2[v1];[0:v][v1]overlay=20:20" -c:v "libx264" -crf 30 -an -y {ouput_path} -v warning'
    os.system(cmd)
