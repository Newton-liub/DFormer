# 运行环境

> 2026-10-07 依裁决建立本地独立环境并完成最小验证。旧 `df2` 环境**冻结**，不再修改，也不在其中补装依赖。

## 本地（本机 RTX 5060）

- conda 环境：`dformer`（路径 `D:\2Env\anaconda\envs\dformer`），Python 3.10.20。
- torch 2.7.0+cu128、torchvision 0.22.0+cu128、cuDNN 90701；CUDA 可用，设备 `NVIDIA GeForce RTX 5060 Laptop GPU`，compute capability (12, 0)。
- scipy 1.15.3、mmcv 1.7.2、mmengine 0.10.7、timm 1.0.30、opencv-python 5.0.0.93、numpy 2.2.6、tensorboardX 2.6.5、tabulate 0.10.0、easydict 1.13、tqdm 4.70.1、matplotlib 3.10.9。
- 为什么不照搬作者的 torch 2.1.2+cu118：本机是 Blackwell（sm_120），需要 CUDA 12.8 及以上的 wheel，因此本地按“先满足当前 GPU”取舍。mmcv 用 1.7.2（PyPI lite 版），因为 mmcv 2.1.0 没有 cu128/torch2.7 的预编译包，本地从源码编译不现实。
- `ftfy` 未安装：全仓库没有任何地方 import 它，属于 requirements 里的冗余项，未优先处理。

## 最小验证（2026-10-07 实际执行）

- 导入 `mmcv.cnn.ConvModule` 与 `models.decoders.ham_head.LightHamHead` 成功（ham 解码器路径依赖 `mmcv.cnn`）。
- 导入 `local_configs.NYUDepthv2.DFormerv2_S` 成功：backbone `DFormerv2_S`、decoder `ham`、40 类。
- 实际建模型并前向一次：`EncoderDecoder` + 本机 `D:\0Project\pretrained\DFormerv2_Small_pretrained.pth`，26.7M 参数，前向 0.78 秒，输出 `(1, 40, 480, 640)` float32、全部 finite，峰值显存 366 MiB。
- 未运行训练、未跑完整测试、未做多卡或长耗时验证。

## 需要先处理的两点

1. **权重放置方式未定**：`DFormerv2_S` 配置期望 `checkpoints/pretrained/DFormerv2_Small_pretrained.pth`（相对仓库根），而权重目前在本机 `D:\0Project\pretrained\`。本次验证是在脚本里临时覆盖路径完成的，**没有**修改任何配置，也**没有**复制权重。正式运行前需要确定：复制到期望路径、建 junction，还是在研究配置里覆盖路径。
2. **预训练键不完全匹配**：加载官方 DFormerv2_Small 权重时，`load_state_dict(strict=False)` 报告 `unexpected keys`（`proj.*`、`norm.*`、`head.*`、`aux_head.*`）与 `missing keys`（`extra_norms.0/1/2`）。这是上游行为——`DFormerv2.init_weights` 先初始化整个 backbone，再用 `strict=False` 覆盖匹配键，因此 `extra_norms.*` 保持新初始化（见 `models/encoders/DFormerv2.py:571-610`）。把 DFormerv2-S 作为第一轮 backbone 时，这是既定的上游设定，不是本轮改动引入的。

## 云端

- 4090 正式训练优先按作者推荐软件栈建环境（torch 2.1.2+cu118、mmcv 2.1.0），与本机环境分开；流程见 [云上流程](cloud.md)。
