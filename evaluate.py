from wm_core.embed import Encoder
from wm_core.extract import Decoder


def cal_psnr(video_path, output_path):
    from ffmpeg_quality_metrics import FfmpegQualityMetrics
    ffqm = FfmpegQualityMetrics(video_path, output_path)

    metrics = ffqm.calc(["psnr"])
    
    # average the psnr values over all frames
    # print(sum([frame["psnr_avg"] for frame in metrics["psnr"]]) / len(metrics["psnr"]))

    # or just get the global stats
    print(ffqm.get_global_stats()["psnr"]["average"])


if __name__ == "__main__":
    encoder = Encoder()
    
    wm = [1, 0, 1, 0] * 64
    video_path = "test_data/100379.mp4"
    output_path = "output/output.mp4"

    cal_psnr(video_path, output_path)

    # decoder = Decoder()
    # decoder.detect_video(output_path, wm)

