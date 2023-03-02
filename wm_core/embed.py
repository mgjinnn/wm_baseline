import copy
import cv2
import heapq
import multiprocessing
import numpy as np

from cv2 import dct, idct
from numpy.linalg import svd
from pywt import dwt2, idwt2
from tqdm import tqdm


class Encoder:

    def __init__(self):
        pass

    def encode(self, img, wm):
        # img is in YUV
        # img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_YUV = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_shape = img.shape[:2]
        ca_shape = [(i + 1) // 2 for i in img_shape]

        wm_bit = np.array(wm)
        wm_size = wm_bit.size

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
        part_shape = ca_block_shape[:2] * block_shape
        block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

        embed_ca = copy.deepcopy(ca)
        embed_YUV = [np.array([])] * 1

        tmp = []
        for i in range(block_num):
            # dct->svd->打水印->逆svd->逆dct
            wm_1 = wm_bit[i % wm_size]

            u, s, v = svd(dct(ca_block[0][block_index[i]]))
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * wm_1) * d1

            tmp.append(idct(np.dot(u, np.dot(np.diag(s), v))))

        for i in range(block_num):
            ca_block[0][block_index[i]] = tmp[i]

        # 4维分块变回2维
        ca_part[0] = np.concatenate(np.concatenate(ca_block[0], 1), 1)
        # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
        embed_ca[0][:part_shape[0], :part_shape[1]] = ca_part[0]
        # 逆变换回去
        embed_YUV[0] = idwt2((embed_ca[0], hvd[0]), "haar")

        # 合并3通道
        img_YUV[:, :, 0] = np.squeeze(np.stack(embed_YUV, axis=2))
        embed_img = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

        return embed_img


    def embed_video(self, wm, video_path, output_path):
        # Embed watermark into a video
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print('### basic info ###')
        print('### width, height ###', width, height)
        print('### frame_size ###', frame_size)
        print('### fourcc ###', fourcc)
        print('### fps ###', fps)
        print('### length ###', length)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)

        count = 0
        pbar = tqdm(total=length)
        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                count += 1
                frame = cv2.cvtColor(frame.astype(np.float32), cv2.COLOR_BGR2YUV)
                pbar.update(1)
                wmed_frame = self.encode(frame, wm)
                wmed_frame = cv2.cvtColor(wmed_frame, cv2.COLOR_YUV2BGR)
                wmed_frame = np.clip(wmed_frame, a_min=0, a_max=255)
                wmed_frame = np.around(wmed_frame).astype(np.uint8)
                out.write(wmed_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        cap.release()
        out.release()

    def embed_video_async(self, wm, video_path, output_path, threads=None):
        cap = cv2.VideoCapture(video_path)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        frame_size = (int(width), int(height))
        fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        fps =  cap.get(cv2.CAP_PROP_FPS)
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        out = cv2.VideoWriter(output_path, int(fourcc), fps, frame_size)

        pool = multiprocessing.Pool(threads)

        count = 0
        futures = []
        hp = []
        heapq.heapify(hp)
        out_counter = [0]
        rbar = tqdm(total=length, position=0, desc="Reading")
        wbar = tqdm(total=length, position=1, desc="Writing")
        callback = lambda x: DtcwtImgEncoder.callback(x, out, hp, out_counter, wbar)

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                rbar.update(1)
                future = pool.apply_async(DtcwtImgEncoder.encode_async, args=(frame, wm, self.alpha, self.step, count), callback=callback)
                futures.append(future)
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        for future in futures:
            future.wait()
        cap.release()
        out.release()

    def encode_async(img, wm, alpha, step, count):
        # img is in YUV
        # img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_YUV = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2YUV)
        img_shape = img.shape[:2]
        ca_shape = [(i + 1) // 2 for i in img_shape]

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
        part_shape = ca_block_shape[:2] * block_shape
        block_index = [(i, j) for i in range(ca_block_shape[0]) for j in range(ca_block_shape[1])]

        embed_ca = copy.deepcopy(ca)
        embed_YUV = [np.array([])] * 1

        tmp = []
        for i in range(block_num):
            # dct->svd->打水印->逆svd->逆dct
            block, shuffler, i = arg
            wm_1 = wm_bit[i % wm_size]

            u, s, v = svd(dct(ca_block[channel][block_index[i]]))
            s[0] = (s[0] // d1 + 1 / 4 + 1 / 2 * wm_1) * d1

            tmp.appen(idct(np.dot(u, np.dot(np.diag(s), v))))

        for i in range(block_num):
            ca_block[0][block_index[i]] = tmp[i]

        # 4维分块变回2维
        ca_part[0] = np.concatenate(np.concatenate(ca_block[0], 1), 1)
        # 4维分块时右边和下边不能整除的长条保留，其余是主体部分，换成 embed 之后的频域的数据
        embed_ca[0][:part_shape[0], :part_shape[1]] = ca_part[0]
        # 逆变换回去
        embed_YUV[0] = idwt2((embed_ca[0], hvd[0]), "haar")

        # 合并3通道
        img_YUV[:, :, 0] = np.squeeze(np.stack(embed_YUV, axis=2))
        embed_img = cv2.cvtColor(img_YUV, cv2.COLOR_YUV2BGR)
        embed_img = np.clip(embed_img, a_min=0, a_max=255)

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
