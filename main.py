import glob
import numpy as np
import os
import shutil
import time

from wm_core.attacks import attack0, attack1, attack2, attack3
from wm_core.embed import Encoder
from wm_core.extract import Decoder


def process_encode():
    video_fps = 25
    video_length = 30
    encoder = Encoder()
    wms = {}
    videos = glob.glob(data_path+'/*')
    total_time_cost = 0
    for video_path in videos:
        ts = time.time()
        video_name = video_path.split('/')[-1].split('.')[0]
        embedded_video_path = os.path.join(embedded_folder, f'{video_name}.mp4')

        # You can generate the watermark with other ways, but you cannot use a fixed watermark.
        wm = np.random.randint(0, 2, 256)
        wms[video_name] = wm

        ts = time.time()

        # Replace your embedding method, 
        # but the function name and return remain the same as baseline.
        encoder.embed_video(wm, video_path, embedded_video_path, threads=12)

        total_time_cost += (time.time()-ts)

    avg_emb_fps = (len(videos) * video_fps * video_length) / total_time_cost
    wms['avg_emb_fps'] = round(avg_emb_fps, 4)
    np.save(output_wms, wms)
    print('embedded done')


def process_attack():
    # Do not change attack function
    embedded_videos = glob.glob(embedded_folder+'/*')
    attacks = [attack0, attack1, attack2, attack3]
    for idx, video_path in enumerate(embedded_videos):
        video_name = video_path.split('/')[-1].split('.')[0]
        print('video_name', video_name)
        print('video_path', video_path)
        video_attacked_folder = os.path.join(attacked_folder, video_name)
        if not os.path.exists(video_attacked_folder):
            os.makedirs(video_attacked_folder)
        
        attacks[idx % len(attacks)](video_name, video_path, video_attacked_folder)


if __name__ == "__main__":
    data_path = 'test_data'
    output_root = 'output'
    output_wms = 'output/wms.npy'
    embedded_folder = f'{output_root}/embedded'

    if not os.path.exists(embedded_folder):
        os.makedirs(embedded_folder)
    else:
        shutil.rmtree(embedded_folder)
        os.makedirs(embedded_folder)
    
    attacked_folder = f'{output_root}/attacked'
    if not os.path.exists(attacked_folder):
        os.makedirs(attacked_folder)
    else:
        shutil.rmtree(attacked_folder)
        os.makedirs(attacked_folder)

    process_encode()
    process_attack()

    # np.load('', allow_pickle=True).item()
