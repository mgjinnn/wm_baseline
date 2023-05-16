# Baseline

简体中文 | [English](README_en.md)

用法：
```bash
bash run.sh
```

baseline：<br/>
1) 在A榜测试数据集上，无攻击情况下BAR约为0.99, 攻击后约为0.51, 抗攻击能力弱。<br/>
2) 采用全帧植入速度慢，隔帧植入抗攻击能力弱。<br/>
3) 基于Linux系统开发，最好在Linux环境下测试。<br/>

docker: <br/>
```bash
docker pull registry.cn-hangzhou.aliyuncs.com/mg_test/wm_baseline:v1
docker run -it -d --cpus=12 -m 24g -v /src:/tar registry.cn-hangzhou.aliyuncs.com/mg_test/wm_baseline:v1
docker exec -it [CONTAINER ID] bash
```

## baseline更新 <br/>
20230516: 调整main.py, evaluate.py，wm_core/embed.py, wm_core/extract.py. 方便参赛者需要并行处理视频以及更改水印生成的方式。

## 参赛规范 <br/>
1) 工程开发目录需要在/workspace/wm_baseline/目录下, 启动脚本固定使用run.sh, 提交镜像中需使用baseline中main.py和evaluate.py. evaluate.py和attacks.py需保持和baseline一致. <br/>
2) 请勿在水印提取阶段直接返回wms.npy中水印，应通过开发水印提取算法提取植入视频中的水印。如果在方案中直接读取返回wms.npy中水印，则成绩无效，且取消参赛资格.<br/>
3) 提交方案中请合理安排日志打印输出内容.<br/>
4) 平台提供了基于镜像地址提交镜像的方式, 将本地镜像推送至阿里云容器镜像仓库或者Dockerhub后, 设置为公开镜像, 在比赛平台提交页面中输入镜像地址. 由比赛平台拉取镜像运行, 运行结束即可在成绩页面查询评测结果. 推送至阿里云容器镜像仓库或者Dockerhub时, 镜像仓库名称尽量不关联上比赛相关的词语, 以免被检索从而泄漏.<br/>
5) 运行镜像时，容器内任何网络不可用，请将依赖的软件、包在镜像中装好. <br/>
6) 为了合理分配资源单次提交运行时间不能超过1个小时，超出后程序自动停止，结果将不被接受.<br/>
7) 确保镜像中cp指令可用.<br/>
8) Docker镜像大小请尽量勿超过8G, 上传镜像中请勿包含数据集, 以及程序生成的视频数据.<br/>
9) PSNR作为客观指标用于筛选和过滤较差的视频质量。最终排名会引入主观评估，增加水印后若导致明显视频质量恶化则会取消排名或者向后调整排名.<br/>
<br/>


## 数据下载链接
[百度云链接:](https://pan.baidu.com/s/1QuqDI6fk9jOQNGoIVH8Ozw) https://pan.baidu.com/s/1QuqDI6fk9jOQNGoIVH8Ozw  <br/>
提取码: mgtv <br/>
[Google Drive:](https://drive.google.com/file/d/1LzdC3T9JDn6WJuPaXzfa45j9anBIhlbJ/view?usp=sharing) https://drive.google.com/file/d/1LzdC3T9JDn6WJuPaXzfa45j9anBIhlbJ/view?usp=sharing  <br/>


## 资源配置：<br/>
CPU: 12vCPU <br/>
内存: 24 GiB <br/>
GPU: Nvidia RTX 3090  24G *1  Driver Version: 470.82.01 <br/>
<br/>


## 阿里云镜像仓库使用方法:<br/>
1) 注册阿里云账户: https://cr.console.aliyun.com/cn-hangzhou/instances. <br/>
2) 在工作台搜索[容器镜像服务], 进入后选择[个人实例]. <br/>
3) 创建镜像仓库、命名空间, 设置仓库名称，选择公开或私有仓库(此处选择公开),  选择本地仓库. <br/>
4) 本地登录阿里云Docker Registry示例: <br/>
```bash
$ docker login --username=[阿里云id] registry.cn-hangzhou.aliyuncs.com
$ docker tag [ImageId] registry.cn-hangzhou.aliyuncs.com/xx1/xx2:[镜像版本号]
$ docker push registry.cn-hangzhou.aliyuncs.com/xx1/xx2:[镜像版本号]
请根据实际镜像信息替换示例中的[阿里云id], [ImageId]和[镜像版本号]参数.
```
5) 在比赛提交页面提交: registry.cn-hangzhou.aliyuncs.com/xx1/xx2:[镜像版本号].
<br/>


## Reference <br/>
This baseline is mainly inspired by [guofei9987/blind_watermark](https://github.com/guofei9987/blind_watermark).
<br/>