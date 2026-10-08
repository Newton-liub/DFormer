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

## 权重放置与预训练加载（2026-10-07 已定案）

1. **权重放在作者默认路径**：`D:\0Project\pretrained\DFormerv2_Small_pretrained.pth` 已复制到 `D:\0Project\DFormer\checkpoints\pretrained\DFormerv2_Small_pretrained.pth`（两侧 110203103 bytes、sha256 一致）。不建 junction、不在研究配置里写本机绝对路径；`checkpoints/` 由 `.gitignore` 忽略，云端沿用同一相对路径。原 `D:\0Project\pretrained\` 继续作为外部权重库。
2. **`extra_norms.*` 按标准 LayerNorm 参数初始化，不需要补权重**：官方 checkpoint 不包含 `extra_norms.*`；作者代码创建这三个 segmentation-side LayerNorm 后按标准初始化（weight = 1、bias = 0），这里指仿射参数的标准初值；LayerNorm 本身仍做归一化，不能称为对输入的恒等映射。`strict=False` 加载时的 `unexpected keys`（`proj.*`、`norm.*`、`head.*`、`aux_head.*`）属于当前 segmentation backbone 不使用的预训练头参数，被忽略。2026-10-07 实测确认：三个 `extra_norms` 的 weight 全为 1、bias 全为 0（numel 128 / 256 / 512），且用配置自身路径即可成功建模型。**不补权重、不人工映射 key、不修改 loader。**
3. 第一轮实验可如实表述为“使用作者官方 DFormerv2-Small pretrained checkpoint，并严格遵循官方代码的加载方式”。公平性只要求所有基于 DFormerv2-Small 的方法与 baseline 走同一套加载逻辑；这组 missing / unexpected keys 在首次 baseline 验收时记录一次即可，不需要每轮重审。

## 云端（现有 4090 实例，无卡准备）

- 2026-10-08 复用已有 conda 环境 `py310`：`/usr/local/miniconda3/envs/py310`，Python 3.10.16、torch 2.1.2+cu118、mmcv 2.1.0、scipy 1.15.3、timm 1.0.28；SwanLab 升至 0.10.1，与本地一致。两端均 `pip check` 无 broken requirements。
- 云端沿用作者推荐 CUDA 11.8 栈，本机保留支持 RTX 5060 的 CUDA 12.8 栈；模型 / config / pretrained 共用，不要求 GPU wheel 字节一致。不重复创建另一份环境。
- 云端无卡实测：作者 SUNRGBD DFormerv2-S 配置构建 37 类模型（26,966,591 参数），成功加载相对路径的同一份官方权重；`torch.cuda.is_available=False` 与 GPU=0 模式一致。SwanLab 使用官方持久凭据登录成功；`screen -dmS dformer-setup-check /bin/true` 正常返回，该短会话不运行训练。
- 非交互 SSH 默认可能找不到 `python`，请使用上述绝对路径，或 `source /usr/local/miniconda3/bin/activate py310` 后执行。SwanLab 凭据使用官方 `~/.swanlab/.netrc`，不进入工程。
- `screen` 4.09.00 已安装，可直接使用 `screen -S <name>` / `screen -r <name>`；不要为尚未授权的训练创建会话。本轮实例仅 GPU=0 准备，无训练 / 评价。

## 本轮最小检查（2026-10-08）

- 本机实际 Python 为 **3.10.22**（以上 2026-10-07 的 3.10.20 是历史记录）；torch 2.7.0+cu128 / CUDA 12.8 可用，执行 2 元素 CUDA 运算成功。
- 新研究配置 `local_configs.research.DFormerv2_S_SUNRGBD` 成功构建 37 类模型（26,966,591 参数），按作者 loader 加载已有 pretrained；774 个匹配的编码器权重张量逐项相等。
- 权重：110,203,103 bytes，sha256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。云端已有外部库中的同名文件哈希相同，无需下载。
- 本地 SwanLab 0.10.1 持久登录在另一 Python 进程复用成功，SDK `disabled` 模式 init/log/finish 成功；定点核对 tracking config、step 与 mIoU 字段成功，没有创建在线验收实验。
- 仅做必要检查，未跑训练、数据评价或完整测试；独立配置与启动条件见 [新研究配置](research-setup.md)。
