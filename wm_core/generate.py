import numpy as np

def generate_watermarks(videos):
    # You can generate the watermark with other ways, 
    # but you cannot use a fixed watermark.
    
    wms = {}
    for video_path in videos:
        wm = np.random.randint(0, 2, 256)
        video_name = video_path.split('/')[-1].split('.')[0]
        wms[video_name] = wm

    return wms