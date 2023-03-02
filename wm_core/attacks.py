import os

def change_resolution():
    cmd = 'ffmpeg -i /data/wm_baseline/output/output.mp4 -vf scale=1080:-1 -y /data/wm_baseline/output/output_1080.mp4'
    os.system(cmd)
