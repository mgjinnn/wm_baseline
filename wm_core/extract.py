import cv2
import ffmpeg
import numpy as np
import time

from cv2 import dct, idct
from numpy.linalg import svd
from pywt import dwt2, idwt2
from tqdm import tqdm


class Decoder:

    def __init__(self):
        pass

    def decode(self, img_YUV):
        img_shape = img_YUV.shape[:2]
        ca_shape = [(i + 1) // 2 for i in img_shape]

        wm_size = 256

        block_shape = np.array([4, 4])
        d1 = 72

        ca, hvd, = [np.array([])] * 1, [np.array([])] * 1
        ca_block = [np.array([])] * 1
        ca_part = [np.array([])] * 1 

        ca_block_shape = (ca_shape[0] // block_shape[0], ca_shape[1] // block_shape[1],
                               block_shape[0], block_shape[1])
        strides = 4 * np.array([ca_shape[1] * block_shape[0], block_shape[1], ca_shape[1], 1])

        ca[0], hvd[0] = dwt2(img_YUV[:, :, 0], 'haar')
        ca_block[0] = np.lib.stride_tricks.as_strided(
            ca[0].astype(np.float32), ca_block_shape, strides)

        block_num = ca_block_shape[0] * ca_block_shape[1]
        block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

        wm_block_bit = np.zeros(shape=(1, block_num)) 

        wms = []
        for i in range(block_num):
            u, s, v = svd(dct(ca_block[0][block_index[i]]))
            wm = (s[0] % d1 > d1 / 2) * 1
            wms.append(wm)

        wm_block_bit[0, :] = wms

        wm_avg = np.zeros(shape=wm_size)
        for i in range(wm_size):
            wm_avg[i] = wm_block_bit[:, i::wm_size].mean()

        return one_dim_kmeans(wm_avg)

    def detect_video(self, wmed_video_path, ori_wm):
        ori_wm = np.array(ori_wm)
        wmed_cap = cv2.VideoCapture(wmed_video_path)
        frames_len = int(wmed_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        bar_list = []
        pbar = tqdm(total=int(frames_len/25))

        while wmed_cap.isOpened():
            ret, wmed_frame = wmed_cap.read()
            if ret:
                if count % 25 != 0:
                    count += 1
                else:
                    wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_BGR2YUV)
                    pbar.update(1)
                    wm = self.decode(wmed_frame)
                    bar = bar_cal(wm, ori_wm)
                    bar_list.append(bar)
                    count += 1
            else:
                break

        avg_bar = sum(bar_list) / len(bar_list)
        return frames_len, round(avg_bar, 4)

    def detect_videos(self, videos, wms_dict):
        bar_list = []
        extract_fps_list = []
        
        for video_path in videos:
            raw_video_name = video_path.split('/')[-1].split('.')[0]
            ts = time.time()

            # Replace your extraction method, 
            # but the function name and return remain the same as baseline.
            attacked_frames, vid_bar = self.detect_video(video_path, wms_dict[raw_video_name])
            bar_list.append(vid_bar)
            extract_fps_list.append(attacked_frames / (time.time()-ts))
        
        avg_extract_fps = round(sum(extract_fps_list) / len(extract_fps_list), 4)

        return bar_list, avg_extract_fps


def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01

def bar_cal(input_wm, output_wm):
    a = round(np.sum(np.equal(output_wm, input_wm)), 3)
    return a/256

