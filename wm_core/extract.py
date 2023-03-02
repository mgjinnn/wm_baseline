import cv2
import numpy as np

from cv2 import dct, idct
from numpy.linalg import svd
from pywt import dwt2, idwt2
from tqdm import tqdm


class Decoder:

    def __init__(self):
        pass

    def decode(self, img_YUV):
        # img_YUV = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_shape = img_YUV.shape[:2]
        ca_shape = [(i + 1) // 2 for i in img_shape]

        wm_size = 256

        block_shape = np.array([4, 4])
        d1 = 36

        # init data
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

        # print('### wm_block_bit', wm_block_bit, wm_block_bit.shape)

        wm_avg = np.zeros(shape=wm_size)
        for i in range(wm_size):
            wm_avg[i] = wm_block_bit[:, i::wm_size].mean()

        # print('### wm_avg', wm_avg, wm_avg.shape)

        return one_dim_kmeans(wm_avg)

    # def detect_video(self, keys, frag_length, wmed_video_path, ori_frame_size=(1080, 1920), mode="fast"):
    def detect_video(self, wmed_video_path, ori_wm, ori_frame_size=(720, 1280)):
        ori_wm = np.array(ori_wm)
        wmed_cap = cv2.VideoCapture(wmed_video_path)
        length = int(wmed_cap.get(cv2.CAP_PROP_FRAME_COUNT))

        count = 0
        pbar = tqdm(total=length)
        while wmed_cap.isOpened():
            ret, wmed_frame = wmed_cap.read()
            if ret:
                # wmed_frame = cv2.resize(wmed_frame.astype(np.float32), (ori_frame_size[1], ori_frame_size[0]))
                wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_BGR2YUV)
                pbar.update(1)
                wm = self.decode(wmed_frame)
                # print('###', wm)
                bar = bar_cal(wm, ori_wm)
                print('bar', bar)
                count += 1
                # exit()
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
            else:
                break

        return ''

def one_dim_kmeans(inputs):
    threshold = 0
    e_tol = 10 ** (-6)
    center = [inputs.min(), inputs.max()]  # 1. 初始化中心点
    for i in range(300):
        threshold = (center[0] + center[1]) / 2
        is_class01 = inputs > threshold  # 2. 检查所有点与这k个点之间的距离，每个点归类到最近的中心
        center = [inputs[~is_class01].mean(), inputs[is_class01].mean()]  # 3. 重新找中心点
        if np.abs((center[0] + center[1]) / 2 - threshold) < e_tol:  # 4. 停止条件
            threshold = (center[0] + center[1]) / 2
            break

    is_class01 = inputs > threshold
    return is_class01

def bar_cal(input_wm, output_wm):
    a = np.sum(np.equal(output_wm, input_wm))
    return 100*a/256