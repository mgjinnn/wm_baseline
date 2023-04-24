import copy
import cv2
import heapq
import multiprocessing
import numpy as np
import time

from cv2 import dct, idct
from numpy.linalg import svd
from pywt import dwt2, idwt2
from tqdm import tqdm


class Encoder:

    def __init__(self):
        pass

    def embed_video(self, wm, video_path, output_path, threads=None):
        ts = time.time()
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)

        pool = multiprocessing.Pool(threads)

        count = 0
        futures = []
        hp = []
        heapq.heapify(hp)
        out_counter = [0]
        rbar = tqdm(total=length, position=0, desc="Reading")
        wbar = tqdm(total=length, position=1, desc="Writing")
        callback = lambda x: Encoder.callback(x, out, hp, out_counter, wbar)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rbar.update(1)
                future = pool.apply_async(Encoder.encode, args=(frame, wm, count), callback=callback)
                futures.append(future)
                count += 1
            else:
                break
        for future in futures:
            future.wait()
            
        cap.release()
        out.release()

        pool.close()
        pool.join()

    def encode(img, wm, count):
        if (count % 5) != 0:
            return count, img

        img_YUV = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_shape = img.shape[:2]
        ca_shape = [(i + 1) // 2 for i in img_shape]

        wm_bit = np.array(wm)
        wm_size = wm_bit.size

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
        part_shape = ca_block_shape[:2] * block_shape
        block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

        embed_ca = copy.deepcopy(ca)
        embed_YUV = [np.array([])] * 1

        tmp = []
        for i in range(block_num):
            wm_1 = wm_bit[i % wm_size]

            u, s, v = svd(dct(ca_block[0][block_index[i]]))
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * wm_1) * d1

            tmp.append(idct(np.dot(u, np.dot(np.diag(s), v))))

        for i in range(block_num):
            ca_block[0][block_index[i]] = tmp[i]

        ca_part[0] = np.concatenate(np.concatenate(ca_block[0], 1), 1)
        embed_ca[0][:part_shape[0], :part_shape[1]] = ca_part[0]
        embed_YUV[0] = idwt2((embed_ca[0], hvd[0]), "haar")

        img_YUV[:, :, 0] = np.squeeze(np.stack(embed_YUV, axis=2))
        embed_img = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)
        embed_img = np.around(embed_img).astype(np.uint8)

        return count, embed_img


    def callback(x, out, hp, out_counter, wbar):
        # Synchronization
        if x[0] != out_counter[0]:
            heapq.heappush(hp, x)
            return
        else:
            out.write(x[1])
            wbar.update(1)
            out_counter[0] += 1
            while len(hp) != 0 and hp[0][0] == out_counter[0]:
                c, frame = heapq.heappop(hp)
                out.write(frame)
                wbar.update(1)
                out_counter[0] += 1
