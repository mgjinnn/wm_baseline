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
    bar_list = []
    extract_fps_list = []
    extract_wm_time_cost = 0

    for raw_video_path in raw_videos:
        raw_video_name = raw_video_path.split('/')[-1].split('.')[0]
        embedded_video_path = raw_video_path.replace(raw_videos_path, embedded_videos_path)

        if not os.path.exists(embedded_video_path):
            psnr_list.append(0)
            bar_list.append([0, 0])
            extract_wm_time_cost += 120
            continue
        else:
            psnr_list.append(cal_psnr(raw_video_path, embedded_video_path))
            # Replace your extraction method, 
            # but the function name and return remain the same as baseline.
            bar_list.append([decoder.detect_video(embedded_video_path, wms_dict[raw_video_name])[1]])
            
        curr_attacked_video_path = f'{os.path.join(attacked_videos_path, raw_video_name)}/{raw_video_name}.mp4'
        if not os.path.exists(curr_attacked_video_path):
            bar_list[-1].append(0)
        else:
            ts = time.time()
            # Replace your extraction method, 
            # but the function name and return remain the same as baseline.
            # Can not use generated wms directly in extraction.
            attacked_frames, vid_bar = decoder.detect_video(curr_attacked_video_path, wms_dict[raw_video_name])
            bar_list[-1].append(vid_bar)
            extract_fps_list.append(attacked_frames / (time.time()-ts))

    avg_extract_fps = round(sum(extract_fps_list) / len(extract_fps_list), 4)
    return psnr_list, bar_list, avg_extract_fps


if __name__ == "__main__":
    video_length = 30
    video_fps = 25
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
    psnr_list, bar_list, avg_extract_fps = process_evaluation(raw_videos)
    
    attacked_bar_list = [x[1] for x in bar_list]


    fp = open(result_file, 'a')
    fp.write('name,psnr,none attack,attacked\n')
    for raw_video, psnr, bars in zip(raw_videos, psnr_list, bar_list):
        raw_video_name = raw_video.split('/')[-1].split('.')[0]
        fp.write(raw_video_name + "," + str(psnr) + ",")
        fp.write(str(bars[0]) + "," + str(bars[1]))
        fp.write("\n")
    
    fp.write("\n")
    avg_psnr = round(sum(psnr_list) / len(psnr_list), 4)
    attacked_avg_bar = round(sum(attacked_bar_list) / len(attacked_bar_list), 4)
    fp.write("overall," + str(avg_psnr) + "," + str(attacked_avg_bar))
    fp.write("," + str(wms_dict['avg_emb_fps']) + "," + str(avg_extract_fps) + "\n")
    fp.close()
