# Baseline

简体中文 | [English](README_en.md)

用法：
```bash
bash run.sh
```

baseline：<br/>
1) 在A榜测试数据集上，无攻击情况下BAR约为0.99, 攻击后约为0.51, 抗攻击能力弱。<br/>
2) 采用全帧植入速度慢，隔帧植入抗攻击能力弱。<br/>


## 参赛规范<br/>
1) 工程开发目录需要在/workspace/wm_baseline/目录下, 启动脚本固定使用run.sh, 提交镜像中需使用baseline中main.py和evaluate.py。<br/>
2）请勿在水印提取阶段直接读取以及返回wms.npy中水印，应通过开发水印提取算法提取植入视频中的水印。如果在方案中直接读取返回wms.npy中水印，则成绩无效，且取消参赛资格。
3) 提交方案中请合理安排日志打印输出内容，输出关键信息即可。<br/>
4) 平台提供了基于镜像地址提交镜像的方式，将本地代码打包成镜像提交，推送至阿里云容器镜像仓库或者Dockerhub后，在比赛平台提交页面中输入镜像地址。由比赛平台拉取镜像运行，运行结束即可在成绩页面查询评测结果。<br/>
5) 推送至阿里云容器镜像仓库或者Dockerhub(建议使用阿里云仓库，镜像仓库名称尽量不关联上比赛相关的词语，以免被检索从而泄漏)。 <br/>
6) 运行镜像时，容器内任何网络不可用，请将依赖的软件、包在镜像中装好。 <br/>
7) 为了合理分配资源单次提交运行时间不能超过1个小时，超出后程序自动停止，结果将不被接受。<br/>
8) 确保镜像中cp指令可用。<br/>
<br/>


资源配置：<br/>
CPU: 12vCPU <br/>
内存: 24 GiB <br/>
GPU: Nvidia RTX 3090  24G *1  Driver Version: 470.82.01 <br/>
<br/>