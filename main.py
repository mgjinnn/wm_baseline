import numpy as np

from wm_core.embed import Encoder
from wm_core.extract import Decoder

if __name__ == "__main__":
    encoder = Encoder()
    
    # wm = [1, 0, 1, 0] * 64
    wm = np.random.randint(0, 2, 256)
    # print(wm)

    video_path = "test_data/100379.mp4"
    # output_path = "output/output.mp4"
    # output_path = "output/output_1080.mp4"
    output_path = "test_data/100379.mp4"

    # encoder.embed_video(wm, video_path, output_path)
    # run.embed_video_async(wm, video_path, output_path, threads=8)

    decoder = Decoder()
    decoder.detect_video(output_path, wm)