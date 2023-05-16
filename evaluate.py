import glob
import numpy as np
import os
import time

from wm_core.embed import Encoder
from wm_core.extract import Decoder

from ffmpeg_quality_metrics import FfmpegQualityMetrics


def cal_psnr(video_path, output_path):
    ffqm = FfmpegQualityMetrics(video_path, output_path)
    metrics = ffqm.calc(["psnr"])
    return ffqm.get_global_stats()["psnr"]["average"]

def process_evaluation(raw_videos):
    psnr_list = []
    embedded_videos = []
    attacked_videos = []

    # 1. check embedded videos and attacked videos:
    for raw_video_path in raw_videos:
        raw_video_name = raw_video_path.split('/')[-1].split('.')[0]
        curr_embedded_video_path = raw_video_path.replace(raw_videos_path, embedded_videos_path)
        curr_attacked_video_path = f'{os.path.join(attacked_videos_path, raw_video_name)}/{raw_video_name}.mp4'
        
        if not os.path.exists(curr_embedded_video_path):
            print('All test videos must be watermarked.')
            return [0], [0], [0], 0
        if not os.path.exists(curr_attacked_video_path):
            print('All test videos must be attacked.')
            return [0], [0], [0], 0 
        
        psnr_list.append(cal_psnr(raw_video_path, curr_embedded_video_path))
        embedded_videos.append(curr_embedded_video_path)
        attacked_videos.append(curr_attacked_video_path)

    # 2. detect embedded videos:
    emb_bar_list, emb_avg_extract_fps = decoder.detect_videos(embedded_videos, wms_dict)

    # 3. detect attacked videos:
    att_bar_list, att_avg_extract_fps = decoder.detect_videos(attacked_videos, wms_dict)

    return psnr_list, emb_bar_list, att_bar_list, att_avg_extract_fps


if __name__ == "__main__":
    result_file = 'result.txt'
    if os.path.exists(result_file):
        os.remove(result_file)

    decoder = Decoder()

    # load watermarks
    wms_path = 'output/wms.npy'
    wms_dict = np.load(wms_path, allow_pickle=True).item()

    raw_videos_path = 'test_data'
    embedded_videos_path = 'output/embedded'
    attacked_videos_path = 'output/attacked'

    raw_videos = glob.glob(raw_videos_path + '/*')
    psnr_list, emb_bar_list, att_bar_list, att_avg_extract_fps = process_evaluation(raw_videos)
    
    fp = open(result_file, 'a')
    fp.write('name,psnr,none attack,attacked\n')
    for raw_video, psnr, ebars, abars in zip(raw_videos, psnr_list, emb_bar_list, att_bar_list):
        raw_video_name = raw_video.split('/')[-1].split('.')[0]
        fp.write(raw_video_name + "," + str(psnr) + ",")
        fp.write(str(ebars) + "," + str(abars))
        fp.write("\n")
    
    fp.write("\n")
    avg_psnr = round(sum(psnr_list) / len(psnr_list), 4)
    attacked_avg_bar = round(sum(att_bar_list) / len(att_bar_list), 4)
    fp.write("overall," + str(avg_psnr) + "," + str(attacked_avg_bar))
    fp.write("," + str(wms_dict['avg_emb_fps']) + "," + str(att_avg_extract_fps) + "\n")
    fp.close()
