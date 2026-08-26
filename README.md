# <p align=center>`DFormer for RGBD Semantic Segmentation`</p>

## 本仓库项目入口

本仓库同时包含上游 DFormer/DFormerv2 论文代码和本项目的 MUSeg 扩展。两套入口用途不同：

- **MUSeg 当前状态与恢复点：** `doc/main/MUSeg-current-status.md`
- **MUSeg 数据准备：** `doc/dataset.md`，默认目录为仓库上一级的 `../dataset/`
- **MUSeg 实验口径与处置状态：** `doc/main/MUSeg-open-decisions.md`
- **MUSeg 阶段计划与历史执行记录：** `doc/plans/`，不承担实时状态
- **正式报告与证据索引：** `doc/reports/` 和 `doc/reports/report-index.json`
- **上游论文复现：** 继续阅读下面的原版 DFormer/DFormerv2 说明

### MUSeg 对话状态维护

`doc/main/MUSeg-current-status.md` 是当前事实、正在进行事项、边界和恢复点的唯一实时入口。每个涉及 MUSeg 的对话都应先读取该文件；训练、验收、指标、checkpoint、official test、云实例、证据位置、阻塞项、恢复步骤、提交或发布状态发生变化时，必须在该对话最终答复之前同步更新。对话结束后代理无法继续写文件，因此“自动更新”统一定义为最终答复前完成，而不是答复后异步补写。

没有持久状态变化的解释或只读检查不更新时间，避免制造虚假进度。详细过程继续写入日期化报告，真正开放的研究选择及处置状态写入 `doc/main/MUSeg-open-decisions.md`。项目级强制规则见 `.cursor/rules/museg-current-status.mdc`。

根目录 `train.sh`、`eval.sh` 和 `infer.sh` 是上游 NYUv2/SUNRGBD 多卡示例，**不是 MUSeg 的审计训练入口**。MUSeg 必须使用冻结 protocol、preflight 和独立输出目录，不能直接修改根脚本绕过 split、Git、checkpoint 和 official-test 门禁。

---

非常荣幸我们收到3D视觉工坊的邀请，我们在6月19日晚上19:00开展了关于DFormerv2的论文直播，有兴趣的同学可以观看[直播回放](https://www.bilibili.com/video/BV1hGNozuEe4?t=4.2)，有问题欢迎在这个项目下提issue交流讨论，直播用到的PPT可以在这里下载[BaiduNetDisk](https://pan.baidu.com/s/1HjmiVBYZSnBGcPDJgfCeoA?pwd=ti6p)。



This repository contains the official implementation of the following papers:

> DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation<br/>
> [Bowen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Xuying Zhang](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Zhongyu Li](https://scholar.google.com/citations?user=g6WHXrgAAAAJ&hl=zh-CN),
> [Li Liu](https://scholar.google.com/citations?hl=zh-CN&user=9cMQrVsAAAAJ),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN) <br/>
> ICLR 2024. 
>[Paper Link](https://arxiv.org/abs/2309.09668) |
>[Homepage](https://yinbow.github.io/Projects/DFormer/index.html) |
>[公众号解读(集智书童)](https://mp.weixin.qq.com/s/lLFejycBr8o7JNoirRDmjQ) |
>[DFormer-SOD](https://github.com/VCIP-RGBD/DFormer-SOD) |
>[Jittor-Version(国产框架)](https://github.com/VCIP-RGBD/DFormer-Jittor) |


> DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)<br/>
> CVPR 2025. 
> [Paper Link](https://arxiv.org/abs/2504.04701) |
> [中文版](https://mftp.mmcheng.net/Papers/25CVPR_RGBDSeg-CN.pdf) |
> [直播回放](https://www.bilibili.com/video/BV1hGNozuEe4?t=4.2) |
> [PPT](https://pan.baidu.com/s/1HjmiVBYZSnBGcPDJgfCeoA?pwd=ti6p) |
> [Geometry prior demo](https://huggingface.co/spaces/bbynku/DFormerv2) |
> [Jittor-Version(国产框架)](https://github.com/VCIP-RGBD/DFormer-Jittor) |

> OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation<br/>
> [Bo-Wen Yin](https://scholar.google.com/citations?user=xr_FRrEAAAAJ&hl=zh-CN&oi=sra),
> [Jiao-Long Cao](https://github.com/caojiaolong),
> [Xuying Zhang](https://scholar.google.com/citations?user=76_hOG0AAAAJ&hl=zh-CN),
> [Yuming Chen](https://scholar.google.com/citations?user=EweNbRAAAAAJ&hl=zh-CN),
> [Ming-Ming Cheng](https://scholar.google.com/citations?hl=zh-CN&user=huWpVyEAAAAJ),
> [Qibin Hou*](https://scholar.google.com/citations?user=fF8OFV8AAAAJ&hl=zh-CN)<br/>
> Neurips 2025. 
> [Paper Link](https://arxiv.org/abs/2509.15096) |
> [Code (busy with work recently, and we will release it soon)] |



:robot:[RGB-D ImageNet and Pretrain(You can train your own encoders)](https://github.com/VCIP-RGBD/RGBD-Pretrain)

:anchor:[Application to new datasets(添加新数据集)](https://github.com/VCIP-RGBD/DFormer/tree/main/figs/application_new_dataset)


We provide the geometry prior generation manner in DFormerv2, and you can further develope it and enhance the depth-related reasearch.
We provide the RGBD pretraining code in [RGBD-Pretrain](https://github.com/VCIP-RGBD/RGBD-Pretrain).
You can pretrain more powerful RGBD encoders and contribute to the RGBD research.

We invite all to contribute in making it more acessible and useful. If you have any questions about our work, feel free to contact us via e-mail (bowenyin@mail.nankai.edu.cn, caojiaolong@mail.nankai.edu.cn). If you are using our code and evaluation toolbox for your research, please cite this paper ([BibTeX](https://scholar.googleusercontent.com/scholar.bib?q=info:GdonbkKZMYsJ:scholar.google.com/&output=citation&scisdr=ClEqKQU5EL_6hIbkmOc:AFWwaeYAAAAAZQvigOeM_E2bhS0d1niD6tYkedk&scisig=AFWwaeYAAAAAZQvigF3P1qyHXOMhOEt-zalsD8w&scisf=4&ct=citation&cd=-1&hl=zh-CN)).



<p align="center">
    <img src="figs/DFormer.png" width="600"  width="1200"/> <br />
    <em> 
    Figure 1: Comparisons between the existing methods and our DFormer (RGB-D Pre-training).
    </em>
</p>

<p align="center">
    <img src="figs/manner.jpg" width="300"  width="1200"/> <br />
    <em> 
    Figure 2: Comparisons among the main RGBD segmentation pipelines and our approach. (a) Use dual encoders to encode RGB and depth respectively and design fusion modules to fusion them, like CMX and GeminiFUsion; (b) Adopt an unified RGBD encoder to extract and
    fuse RGBD features, like DFormer; (c) DFormerv2 use depth to form
    a geometry prior of the scene and then enhance the visual features.
    </em>
</p>

<p align="center">
    <img src="figs/geo_attention.png" width="600"  width="1200"/> <br />
    <em> 
    Figure 2: The geometry attention map in our DFormerv2 and the effect of other attention mechanisms. Our geometry attention is endowed with the 3D geometry perception ability and can focus on the related regions of the whole scene. 
    A simple visualization demo is provided at 
    https://huggingface.co/spaces/bbynku/DFormerv2.
    </em>
    
</p>


## 1. 🌟  NEWS 

- [2025/09/19] Our OmniSegmentor has been accepted by Neurips 2025, the code are coming soon.
- [2025/04/08] The code of DFormerv2 is available.
- [2025/03/09] Our DFormerv2 has been accpeted by CVPR 2025.
- [2025/02/19] The jittor implementation of DFormer is avaiable at [Jittor-Version](https://github.com/VCIP-RGBD/DFormer-Jittor).
- [2024/10/12] Based on our DFormer, Wu's method UBCRCL has won the RUNNER-up at [Endoscopic Vision Challenge SegSTRONG-C Subchallenge](https://segstrongc.cs.jhu.edu/) of MICCAI 24. Congratulation!
- [2024/04/21] We have upgraded and optimized the framework, greatly reducing training time, i.e., training duration for DFormer-L is reduced to ~12h from over 1day.
- [2024/01/16] Our DFormer has been accpeted by The International Conference on Learning Representations (ICLR 2024).

## 2. 🚀 Get Start

**0. Install**

```bash
conda create -n dformer python=3.10 -y
conda activate dformer

# CUDA 11.8
conda install pytorch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 pytorch-cuda=11.8 -c pytorch -c nvidia

pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

pip install tqdm opencv-python scipy tensorboardX tabulate easydict ftfy regex
pip install timm==1.0.28 mmengine==0.10.7 matplotlib PyYAML thop

# MUSeg 审计运行还需要在线实验记录依赖
pip install -r requirements-monitoring.txt
```


**1. Download Datasets and Checkpoints.**



- **Datasets:** 

上游下载包的示例结构使用 `datasets/`，但仓库现有内置配置与 MUSeg 扩展默认使用仓库上一级的 `../dataset/`。运行前应以所选配置中的 `C.root_dir` 为准，不要无说明地混用两个目录。MUSeg 的固定约定和转换命令见 `doc/dataset.md`。

| Datasets | [GoogleDrive](https://drive.google.com/drive/folders/1RIa9t7Wi4krq0YcgjR3EWBxWWJedrYUl?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EqActCWQb_pJoHpxvPh4xRgBMApqGAvUjid-XK3wcl08Ug?e=VcIVob) | [BaiduNetdisk](https://pan.baidu.com/s/1-CEL88wM5DYOFHOVjzRRhA?pwd=ij7q) | 
|:---: |:---:|:---:|:---:|

Compred to the original datasets, we map the depth (.npy) to .png via 'plt.imsave(save_path, np.load(depth), cmap='Greys_r')', reorganize the file path to a clear format, and add the split files (.txt).



- **Checkpoints:** 

ImageNet-1K Pre-trained and NYUDepth or SUNRGBD trained DFormer-T/S/B/T and DFormerv2-S/B/L can be downloaded at:
<!-- 
| Pre-trained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL) | [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | 
|:---: |:---:|:---:|:---:|




NYUDepth v2 trained DFormers T/S/B/L can be downloaded at 

| NYUDepth v2 | [GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf) | [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | 
|:---: |:---:|:---:|:---:|


*SUNRGBD 

| SUNRGBD | [GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing) | [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che) | [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | 
|:---: |:---:|:---:|:---:| -->


| Weights | DFormer | DFormerv2 |
|-------|-------| -  |
| Pretrained | [GoogleDrive](https://drive.google.com/drive/folders/1YuW7qUtnguUFkhC-sfqGySrerjK0rZJX?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EhTTF_ZofnFIkz2WSDFAiiIBEIubZUpIwDQYwm9Hvxwu8Q?e=x8XumL), [BaiduNetdisk](https://pan.baidu.com/s/1JlexzFqMcZOXPNiNkE1zRA?pwd=gct6) | [BaiduNetdisk](https://pan.baidu.com/s/1alSvGtGpoW5TRyLxOt1Txw?pwd=i3pn), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/pretrained) |
|NYUDepthv2 |[GoogleDrive](https://drive.google.com/drive/folders/1P5HwnAvifEI6xiTAx6id24FUCt_i7GH8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/ErAmlYuhS6FCqGQZNGZy0_EBYgJsK3pFTsi2q9g14MEE_A?e=VoKUAf), [BaiduNetdisk](https://pan.baidu.com/s/1AkvlsAvJPv21bz2sXlrADQ?pwd=6vuu) | [BaiduNetdisk](https://pan.baidu.com/s/1hi_XPCv1JDRBjwk8XN7e-A?pwd=3vym), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/NYU) |
|SUNRGBD|[GoogleDrive](https://drive.google.com/drive/folders/1b005OUO8QXzh0sJM4iykns_UdlbMNZb8?usp=sharing), [OneDrive](https://mailnankaieducn-my.sharepoint.com/:f:/g/personal/bowenyin_mail_nankai_edu_cn/EiNdyUV486BFvb7H2yJWSCMBElOj-m6EppIy4dSXNX-yNw?e=fu2Che), [BaiduNetdisk](https://pan.baidu.com/s/1D6UMiBv6fApV5lafo9J04w?pwd=7ewv) | [BaiduNetdisk](https://pan.baidu.com/s/1NUOgzYmrXmwU7XA8RTRYPg?pwd=ytr7), [HuggingFace](https://huggingface.co/bbynku/DFormerv2/tree/main/DFormerv2/SUNRGBD) |


 <br />


<details>
<summary>Orgnize the checkpoints and dataset folder in the following structure:</summary>
<pre><code>

```shell
<checkpoints>
|-- <pretrained>
    |-- <DFormer_Large.pth.tar>
    |-- <DFormer_Base.pth.tar>
    |-- <DFormer_Small.pth.tar>
    |-- <DFormer_Tiny.pth.tar>
    |-- <DFormerv2_Large_pretrained.pth>
    |-- <DFormerv2_Base_pretrained.pth>
    |-- <DFormerv2_Small_pretrained.pth>
|-- <trained>
    |-- <NYUDepthv2>
        |-- ...
    |-- <SUNRGBD>
        |-- ...
<datasets>
|-- <DatasetName1>
    |-- <RGB>
        |-- <name1>.<ImageFormat>
        |-- <name2>.<ImageFormat>
        ...
    |-- <Depth>
        |-- <name1>.<DepthFormat>
        |-- <name2>.<DepthFormat>
    |-- train.txt
    |-- test.txt
|-- <DatasetName2>
|-- ...
```

</code></pre>
</details>




 <br /> 




**2. Train（上游示例）.**

`train.sh` 当前固定 NYUv2、2 张 GPU 和 `local_configs.NYUDepthv2.DFormerv2_S`。先按实际设备、数据和配置审查脚本；单卡 MUSeg 不使用此入口。
```
bash train.sh
```

After training, the checkpoints will be saved in the path `checkpoints/XXX', where the XXX is depends on the training config.


**3. Eval（上游示例）.**

`eval.sh` 当前固定 NYUv2、8 张 GPU 和一个具体 checkpoint。运行前必须确认设备数、配置和 checkpoint 路径。
```
bash eval.sh
```

**4. Visualize（上游示例）.**

`infer.sh` 当前固定 NYUv2、2 张 GPU 和 checkpoint；它不是通用或 MUSeg 推理入口。

```
bash infer.sh
```

**5. FLOPs & Parameters.**

```
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python utils/benchmark.py --config local_configs.NYUDepthv2.DFormer_Large
```

**6. Latency.**

```
PYTHONPATH="$(pwd):${PYTHONPATH:-}" python utils/latency.py --config local_configs.NYUDepthv2.DFormer_Large
```

ps: The latency highly depends on the devices. It is recommended to compare the latency on the same devices. 


## 🚩 Performance

<p align="center">
    <img src="figs/Semseg.jpg" width="600"  width="1200"/> <br />
    <em> 
    Table 1: Comparisons between the existing methods and our DFormer.
    </em>
</p>

<p align="center">
    <img src="figs/dformerv2_table.jpg" width="600"  width="1200"/> <br />
    <em> 
    Table 2: Comparisons between the existing methods and our DFormerv2.
    </em>
</p>

## 🕙 ToDo
- [ ] Tutorial on applying the DFormer encoder to the frameworks of other tasks
- ~~[-] Release the code of RGB-D pre-training.~~
- ~~[-] Tutorial on applying to a new dataset.~~
- ~~[-] Release the DFormer code for RGB-D salient obejct detection.~~

> We invite all to contribute in making it more acessible and useful. If you have any questions or suggestions about our work, feel free to contact me via e-mail (bowenyin@mail.nankai.edu.cn) or raise an issue. 


## Cursor Canvas 报告

Canvas 采用“仓库源文件 + Cursor 受管副本”的方式保存：

- Git 管理的当前源文件放在 `doc/canvases/`；只读历史版本放在 `doc/canvases/old/`；
- 默认发布只扫描 `doc/canvases/` 顶层当前版本，不递归发布 `old/`；
- 文件名必须使用 `MAJOR.MINOR.PATCH-<name>.canvas.tsx`，例如 `0.0.2-weekly-progress.canvas.tsx`；
- 页面标题或显著元数据必须显示与文件名一致的版本号；
- 已发布版本只读保留。修改 Canvas 时创建新版本，不覆盖或删除旧文件；
- `doc/reports/report-index.json` 记录已用版本和下一个可用版本。

发布方式：

- 双击 `tools/publish-canvas.cmd`，只发布 `doc/canvases` 顶层的当前 Canvas；
- 在项目根目录执行 `powershell -ExecutionPolicy Bypass -File tools/publish-canvas.ps1`，效果相同；
- 只发布单个当前文件时执行 `powershell -ExecutionPolicy Bypass -File tools/publish-canvas.ps1 -Source doc/canvases/0.0.9-markdown-consistency-audit.canvas.tsx`；
- 需要恢复某个历史预览时，显式把 `-Source` 指向 `doc/canvases/old/<version>-<name>.canvas.tsx`；归档不参与默认发布；
- 使用 `-WhatIf` 可以预览目标路径而不复制文件；
- 只有迁移历史文件时才使用 `-AllowUnversioned`。

发布脚本会校验版本前缀和版本唯一性。目标中已有同名同内容文件时跳过；同名但内容不同时拒绝覆盖，并要求提升版本号。脚本不会清理或删除已有 Canvas。

脚本会根据当前项目路径自动计算 Cursor 受管目录，例如本项目对应 `C:\Users\<用户名>\.cursor\projects\d-0Project-DFormer\canvases`。受管副本仅用于 Cursor 预览；`doc/canvases/` 的当前源和 `doc/canvases/old/` 的只读归档共同构成长期保存与审阅依据。

项目级汇报 Skill 位于 `.cursor/skills/research-progress-report/`。正式报告默认保存到 `doc/reports/`；需要组会展示或可视化布局时，再根据 Markdown 事实正文生成下一个版本的 Canvas。

## Reference
You may want to cite:
```
@inproceedings{yin2024dformer,
  title={DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation},
  author={Yin, Bowen and Zhang, Xuying and Li, Zhong-Yu and Liu, Li and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={ICLR},
  year={2024}
}

@inproceedings{yin2025dformerv2,
  title={DFormerv2: Geometry Self-Attention for RGBD Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Cheng, Ming-Ming and Hou, Qibin},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={19345--19355},
  year={2025}
}

@article{yin2025omnisegmentor,
  title={OmniSegmentor: A Flexible Multi-Modal Learning Framework for Semantic Segmentation},
  author={Yin, Bo-Wen and Cao, Jiao-Long and Zhang, Xuying and Chen, Yuming and Cheng, Ming-Ming and Hou, Qibin},
  journal={arXiv preprint arXiv:2509.15096},
  year={2025}
}
```


### Acknowledgment

Our implementation is mainly based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.24.1), [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [CMNext](https://github.com/jamycheung/DELIVER). Thanks for their authors.



### License

Code in this repo is for non-commercial use only.






