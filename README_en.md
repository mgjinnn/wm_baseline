# Baseline

[简体中文](README.md) | English

Baseline Usage:
```bash
bash run.sh
```

baseline：<br/>
1) On the test dataset A, the BAR is about 0.99 without attack and about 0.51 after attack, showing weak anti-attack ability.<br/>
2) If embedding the watermark in each frame, the speed is very slow, and if embedding the watermark in every n frames (n > 1), the anti-attack ability is weak.<br/>

## Challenge Specifications<br/>
1) The project development directory needs to be in the /workspace/wm_baseline/ directory, the startup script always uses run.sh, and the main.py and evaluate.py in the baseline need to be used when submitting the Docker image.<br/>
2) Do not directly load and return the watermark in wms.npy in the watermark extraction stage, and extract the embedded watermark by developing a watermark extraction algorithm. If the watermark in wms.npy is returned directly in the extraction stage, the result will be invalid and the qualification will be disqualified.<br/>
3) Please arrange the log printout content reasonably and output the key information.<br/>
4) The challenge website provides a way to submit Docker images based on the image url address, and you should package the local code into an image submission. You can push it to the Aliyun Docker Hub or Dockerhub and submit the Docker image address on the submission page of the challenge platform. The challenge platform pulls the image to run, and after the computation is finished, you can check the evaluation results on the results page.<br/>
5) Push to Aliyun Docker Hub or Dockerhub (it is recommended to use Aliyun Docker Hub, and the name of the Docker image should not be associated with challenge-related words as much as possible, so as not to be retrieved and leaked). <br/>
6) When running the image, any network in the container is unavailable, please install the dependent software and packages in the image.<br/>
7) In order to allocate resources reasonably, the running time of a single submission cannot exceed 1 hour, and the program will automatically stop after exceeding it, and the running results will be failed.<br/>
8) Make sure that the cp command is available in the image.<br/>
<br/>


Computation Resources<br/>
CPU: 12vCPU <br/>
内存: 24 GiB <br/>
GPU: Nvidia RTX 3090 Driver Version: 470.82.01. The usage of GPU memory is limited to 16 GB<br/>
<br/>