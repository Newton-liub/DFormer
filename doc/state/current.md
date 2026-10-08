# DFormer 项目实时状态

> 事实时间：2026-10-08。本文件是当前事实、授权边界与恢复点的唯一入口。上一阶段较长状态已摘要归档至 [工程准备前状态](../archive/2026-10-08-research-setup/project-state-before-setup.md)。详细环境与配置见 [运行环境](../guides/environment.md) 和 [新研究配置](../guides/research-setup.md)。

## 当前阶段

新研究轮次的本地与云端工程准备已完成；**本轮未设计新模块、未下载新数据集、未启动训练或评价**。第一阶段 backbone 固定为 **DFormerv2-S**，保留作者最新 DFormer 仓库及 DFormer++ 代码 / 配置，后者为第二 backbone / generalization 候选，不回退或重建旧 DFormerv2 仓库。

## 已确认事实

### 仓库与模型

- 作者基线仍为 `e3273009b759b578945483828ff315d560be94c9`；研究分支 `research/dformerpp-clean-start`，本轮开始前 HEAD `fe6cea2e7efdaf8f87299bed41baa86f11e316b4`。origin 为 `https://github.com/Newton-liub/DFormer.git`，upstream push 继续禁用。
- 新研究配置：`local_configs.research.DFormerv2_S_SUNRGBD`。复制作者 SUNRGBD DFormerv2-S 配置后独立修改路径和日志参数；backbone=`DFormerv2_S`、ham decoder、37类；seed12345 / AdamW / lr8e-5 / batch16 / 300epochs / warmup10 / augmentation 均继承作者设置。
- 官方权重：`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`，110,203,103 bytes，sha256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。本地与云端已有文件哈希相同，未重新下载。
- 本机用新 config 建模成功（26,966,591参数），774个匹配的 backbone 权重张量逐项相等。云端用同一作者 SUNRGBD DFormerv2-S 模型在CPU建模加载成功。官方 loader 的 missing 项仅为 `extra_norms` 标准 LayerNorm 参数初值，unexpected 项为未使用的预训练头，不补 key / 不改 loader。
- 原 MUSeg / MMFR checkpoints 保留为历史参考，不作为新论文主 baseline。新数据集正式实验时，从相同官方 pretrained 重新训练 baseline，baseline 与候选共用 seed、schedule、augmentation、evaluator；本轮不重训。

### SwanLab 与本地环境

- 本机环境 `D:\2Env\anaconda\envs\dformer`：实际 Python3.10.22、torch2.7.0+cu128 / CUDA12.8、mmcv1.7.2、scipy1.15.3、timm1.0.30；RTX5060 Laptop CUDA 可用，2元素CUDA运算成功。旧 `df2` 环境继续冻结。
- 两端 SwanLab 均为0.10.1，使用官方持久登录，另一进程无 key 参数复用登录成功；SDK disabled 模式生命周期正常，没有新建在线验收实验。凭据仅在各用户目录 `.swanlab/.netrc`；没有进入训练代码、config 或 Git。
- 用户 `临时/key.txt` 已读取用于登录、未输出、未删除，且加入 `.gitignore`。用户可自行删除。`.serena/` 与默认本地 SwanLab 产物也被忽略。
- `utils/train.py` 原来只有 TensorBoard，本轮只增加可选日志调用，实现在 `research/tracking.py`；global rank0 记录 epoch平均loss、LR、验证mIoU、epoch / update、实验名称、核心config / CLI。未启用 `swanlab_enabled` 的旧配置不加载SDK。训练入口 `--help` 在本地正常，不启动训练。
- 两端 `pip check` 无 broken requirements；代码仅作定点语法 / 接入检查，没有新建测试文件、没有运行完整测试套件。

### 云端工程与资产

- 实例仍为 `cpod-1vbh7faqcauq`（cn-bj2-03，50GB系统盘）；本轮准备窗口实际无卡A档：2CPU / 4GiB / GPU=0，未启动GPU。无卡不是免费状态，平台返回 InstancePrice0.13；准备窗口设有12:35（UTC+8）计划关机保险。
- 当前工程：`/root/rivermind-data/DFormer`，已从现用 fork 的研究分支 clone。同步方式为本地最终提交后通过 Git bundle fetch；**不推送远端**。bundle 传递 Git 对象，不用旧源码快照覆盖工程；收尾必须核对两端 HEAD。
- 旧云端工程完整移动到 `/root/rivermind-data/DFormer-archive-20261008`，原HEAD `7d6246a7db77f77e26e20d84074dff6e191e9fa5`，其 cloud / outputs / experiments / checkpoints 等均保留。独立旧结果 `/root/rivermind-data/cloud`、现有 MUSeg `/root/rivermind-data/dataset` 与 pretrained 库原样保留；未判断已同步的旧结果也未删除。
- 清理仅针对可重新生成的 pip 下载缓存：清理前3025.1MB，清理后HTTP缓存0bytes。旧转移压缩包与有价值checkpoints保留，不冒险删除。
- 复用 `/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、scipy1.15.3、timm1.0.28；兼容相同DFormer模型与config，不重复建环境。本地GPU wheel与云端作者推荐栈允许不同。
- 官方权重通过工程相对路径软链接至已有 `/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`，无需上传或下载。SwanLab官方凭据目录权限700、文件600，已通过登录验证。
- screen4.09.00已可用，短命令启动正常；无训练会话。工程准备结束后按 [云端流程](../guides/cloud.md) 主动停机，下一次使用须无卡重新启动并核对状态；本轮不保留常驻服务。

### 数据集与外部资源

- 下载指南：`D:\0Project\DFormer\临时\DATASET_DOWNLOAD_GUIDE.md`。含SUNRGBD作者整理版单文件入口 / Windows与Linux命令 / 路径 / split / depth协议；DeLiVER官方 front-view `DELIVER.tar.gz` / RGB-depth-HHA区别 / 天气及failure筛选 / 官方评价代码；NYUv2第二阶段可选；MUSeg最终真实矿井验证复用现有数据。
- SUNRGBD 与 DeLiVER 本轮均未下载。作者SUNRGBD配置eval_source=test.txt，正式实验前必须固定开发验证策略及official-test使用边界，本轮不授权test评价。DeLiVER尚未写DFormer适配器，留作正式benchmark接入任务。
- 外部索引仍在 `D:\0Project\origin\_index\`，schema4 / revision7，37篇canonical、32篇有人工override、LIB000001已补auto.abstract，backups为revision6 / 5 / 4；不在仓库复制索引。本轮未改论文库或索引。旧论文工具报告和此前未提交的非任务文档不混入本次工程提交。
- 本地旧项目仍完整封存 `D:\0Project\DFormer-archive-20261007\`，不恢复MMFR / LER / natural_missing代码，不删除历史权重。

## 授权边界

- 本轮用户明确授权：本机SDK / 登录 / 最小接入 / 必要验证、本地一次Git提交、现有云实例无卡启动 / 工程清理 / 环境与screen配置 / 权重准备、只写数据集下载说明。
- 本轮未授权：推送远端、GPU启动、训练 / 评价、下载新数据集、创建或销毁云资源、完整测试套件或全仓扫描。SwanLab key不得进入Git；用户自行删除key.txt。
- 研究方法、最终验证 / test协议、新模块、正式baseline与候选训练需另行确认。旧训练授权与旧baseline资格不沿用。

## 下一步与恢复点

1. 到此停止工程准备。需要第一阶段实验时先单独授权SUNRGBD下载，按指南放置作者整理版，并确定开发验证策略。
2. 固定实验合同后再授权同pretrained的新baseline与候选；当前环境未做正式训练验收，不宣称训练已通过。
3. 本机研究环境与研究配置已可用；云端持久工程 / 环境 / 资产均在上述路径。云实例下一次从无卡启动恢复，GPU训练需重新授权。
4. 当前工程提交身份用仓库 `git rev-parse HEAD` 获取；本轮最终同步通过Git bundle，origin分支不更新。必要的临时bundle传输文件收尾删除。

## 阻塞与待确定项

- 工程基础检查无已发现阻塞。后续训练前置条件仍包括SUNRGBD数据落盘、验证 / test协议固定及GPU授权；当前不具备这些条件，因此不开始训练。
- DeLiVER仅记录下载与官方协议，DFormer适配尚未实现；完整六视角公开下载入口未核实，不将front-view公开包冒称完整六视角数据。
