# 2026-10-07 待裁决问题：主干与预训练权重、运行环境、外部资料索引

> **角色：** 一次性提出的待裁决问题清单，交接给上级或更高级模型判断。实时事实见 [当前状态](../state/current.md)。
> **状态：** 五项问题已于 2026-10-07 获裁决并执行完毕，见下方“裁决结果”；原有问题、证据与选项保留在本报告中供追溯。

## 裁决结果（2026-10-07，已执行）

1. **主干与权重**：不以 DFormer++ 权重阻塞科研设计，方向审计照常进行。第一轮方法验证用 **DFormerv2-Small**（本机已有权重）作为可运行 backbone；DFormer++ 保持代码基线与后续目标 backbone，其升级放到“方向被证明值得继续”之后的正式实验阶段。不下载新权重、不自行做完整 RGB-D 预训练、不用随机初始化结果与成熟预训练模型直接比较。→ 本轮未下载任何权重。
2. **环境**：旧 `df2` 冻结；新建独立 conda 环境 `dformer`，先满足本机 RTX 5060（CUDA 12.8），再最小验证；明确补 `scipy`，`ftfy` 因全仓库无 import 而不优先处理；云端 4090 正式训练另按作者推荐栈建环境。→ 已建 `dformer` 并通过最小 import / forward 验证，详见 [运行环境](../guides/environment.md)。
3. **索引落点**：外部资产目录是索引唯一真源，仓库只保留很短的入口文档。→ 索引与维护工具已迁至 `D:\0Project\origin\_index\`（7 个文件逐一 sha256 比对与源一致，工具 `--dry-run` 实跑通过），仓库入口为 [外部资源入口](../guides/external-resources.md)。
4. **索引范围**：37 篇为 canonical 正式库；`论文待处理`（9 项）标 Pending、`论文_duplicates_review`（1 项）标 Duplicate Review，索引入口知道其存在，但方向审计不得当作已核验论文使用。→ 已写入 `_index\pending-and-duplicates.md` 与 `paper-index.md`、`code-index.md` 的抬头。
5. **库外 `origin\DFormer`**：暂不更新、不删除，先标 deprecated 并禁止研究任务默认使用；当前唯一有效工作基线是 `D:\0Project\DFormer\`，待确认其无独有内容后再删除。→ 已在 `code-index.md` 抬头标注。

## 1. 背景：当前已核验状态

- 旧项目已完整封存于 `D:\0Project\DFormer-archive-20261007\`（HEAD `8c274c59`、分支 `perf/mmfr-a2-v3-pipeline-opt1`、13 个修改、3 个未跟踪条目，逐项复核一致）。
- 新仓库 `D:\0Project\DFormer\`，分支 `research/dformerpp-clean-start`，基线为作者最新 DFormer++ 代码 `e3273009b759b578945483828ff315d560be94c9`；`origin` 指向个人 fork，`upstream` 指向作者仓库且已禁用 push；已推送提交 `4425d37`。
- 未迁移任何旧研究代码（MMFR、LER、旧 MUSeg 配置、`natural_missing` 均留在归档）；未运行任何训练或评价；未操作云资源。
- Cursor 侧已建立两个 Rule 与四个 Skill（`project-state`、`research-idea`、`subagent-dispatch`、`compshare-cloud`），均已实际执行验证过。
- 本机数据集只有 MUSeg 两套产物：原始 `MUSeg`（约 1.44 GB）与转换版 `MUSeg_DFormer`（3171 样本，train 1595 + test 1576，RGB / Label / Depth / Depth16 各 3171 文件）；NYU Depth v2 与 SUN RGBD 本地不存在。
- 云侧：唯一实例 `cpod-1vbh7faqcauq`（`mmfr-a2-4090-probe`）为 `Stopped`、GPU 0；用户已决定暂不释放。

## 2. 问题一：主干与预训练权重路线

### 已核验事实

- DFormer++ 配置写死了预训练路径：

```5:5:local_configs/NYUDepthv2/DFormerPP_S.py
C.pretrained_model = "checkpoints/pretrained/DFormerPP_Small.pth.tar"
```

- 训练模式会按该路径加载，文件缺失即在建模型阶段失败：

```221:226:models/builder.py
        if self.criterion:
            self.init_weights(cfg, pretrained=cfg.pretrained_model)

    def init_weights(self, cfg, pretrained=None):
        if pretrained:
            logger.info("Loading pretrained model: {}".format(pretrained))
            self.backbone.init_weights(pretrained=pretrained)
```

- 作者 README 的权重表（`README.md:175–179`）把 DFormer++ 的 Pretrained、NYUDepthv2、SUNRGBD 三行全部标为 “Coming soon”。
- 作者 HuggingFace 账号 `bbynku` 只有一个模型仓：`bbynku/DFormerv2`，其中 `DFormerv2/pretrained/` 提供 `DFormerv2_Small_pretrained.pth`（110 MB）、`DFormerv2_Base_pretrained.pth`（211 MB）、`DFormerv2_Large_pretrained.pth`（376 MB），共约 697 MB，可直接下载。
- DFormer 系列（非 ++）预训练与 NYU / SUN 训练权重在 GoogleDrive / OneDrive / 百度网盘。
- 本机只有 `D:\0Project\pretrained\DFormerv2_Small_pretrained.pth`（105.1 MB，与 HF 同一文件）。在五处候选根目录（深度 ≤3）共 36 个权重文件里没有任何 DFormer++ 权重。

### 结论

DFormer++ 预训练编码器**官方尚未发布，当前无法从官方渠道下载**；能否使用 DFormer++ 主干因此不是下载问题，而是路线选择问题。DFormerv2 权重可下载，Small 已在本机。

### 待裁决选项

1. 新方向以 **DFormerv2 主干**起步（Small 已就绪），DFormer++ 等作者发布后再评估。
2. 先补齐 **DFormerv2 Base / Large**（HuggingFace 直链，约 587 MB），使主干选择范围更宽。
3. 坚持 **DFormer++ 但从零训练编码器**，放弃 ImageNet 预训练；需要重新设计训练预算，并说明该差异对比较公平性的影响。
4. 暂不下载任何权重，等论文方向确定后再定。

需要一并明确：是否允许下载数百 MB 级外部权重、是否接受“无 ImageNet 预训练”这一改变、以及是否允许等待作者发布。

## 3. 问题二：运行环境

### 已核验事实（`df2` 环境，`D:\2Env\anaconda\envs\df2`）

- Python 3.10.20；torch 2.7.0+cu128；torchvision 0.22.0+cu128；CUDA 可用、1 张卡。
- mmcv 1.7.2；mmengine 0.10.7；timm 1.0.28；numpy 2.2.6；opencv-python 4.13.0；matplotlib、tabulate、tensorboardX、tqdm、regex、easydict 均已安装。
- **缺 `scipy`**：作者安装清单要求它，且仓库内 `utils/loss_opr.py`、`utils/visualize.py`、`mmseg/models/backbones/beit.py` 均 import 它。
- `ftfy` 同样未安装，但全仓库没有任何地方 import，属于冗余要求。
- 与作者建议不一致：作者要求 torch 2.1.2+cu118 与 mmcv 2.1.0；本机为 torch 2.7.0+cu128 与 mmcv 1.7.2。
- 尚未用该环境 import 项目代码、做过 forward 或运行任何流程，兼容性**未验证**；本轮未安装或升级任何依赖。

### 待裁决选项

1. 为 DFormer++ 基线**新建独立环境**，按作者要求安装（不动 `df2`），代价是重新下载安装。
2. 在 `df2` 里**最小补装 `scipy`**，其余保持不动，先做兼容性验证，代价是仍需面对 mmcv / torch 版本差异。
3. 暂不动环境，等方向确定后再处理。

## 4. 问题三：外部资料索引的落点

### 已核验事实

- 论文库 `D:\0Project\origin\论文` 共 39 个目录：37 篇完整 paper bundle（含 Tables / Figure / Formula 结构，其中 MaskMentor、OmniSegmentor 缺 Formula），另两个为非论文目录（`DFormer-doc-paper`、`MMFR-附件`）。
- 归档 `MMFR/03_reference/` 已存在完整索引与审计材料：`PAPER_LIBRARY_INDEX.json`（schema_version 1、37 篇、`next_lib_number` 38、`bootstrap_complete` true，与当前 37 篇一一对应、无遗漏）、`PAPER_LIBRARY_INDEX.md`、`paper-index.md`、`code-index.md`、`literature_fulltext_audit.md`、`external_reference_provenance_2026-09-21.md`、`MMFR_reference_index_v4_1_2026-09-20.md`、`MMFR_reference_registry_v4_1_2026-09-20.json`。
- 人工编号 PR / RE / AI 已登记在索引中，其中 RE042 / RE053 / RE188 / RE447 仍标记为 unverified。
- 维护工具是归档的 `human/paper_library.py`（1039 行，支持 `--bootstrap`、`--dry-run`、`--root`、`--legacy-root`）。
- 源码库 `D:\0Project\origin` 有 12 个 git clone（DFormer、LFDA、LightDepth、MMSS、nconv、nconv-nyu、NR-MVSNet、OPM-MVS、RGBD-MMCD、RMMSS、S2MA、SMAC）；归档 `code-index.md` 已逐一记录，本次核对每个仓库的 HEAD 与 remote，**全部一致**。
- 新仓库目前没有任何指向这些资产的索引入口。

### 目标

让论文库与源码库成为“打开即可用于规划新方向、设计新方案”的参考入口，同时不在仓库内制造第二套需要同步维护的数据库。

### 待裁决选项

1. 把归档里的现有索引与维护工具**迁入新仓库**（例如 `doc/reference/` 与 `tools/paper_library/`），保留人工编号；优点是随仓库版本化、可直接被检索，代价是工具与索引进入仓库需长期维护。
2. 新仓库只放**精简 markdown 索引**（论文清单 + 仓库清单），完整机器索引继续留在归档；优点是轻，代价是两处信息可能漂移。
3. 索引**紧跟库外资产**（例如放在 `origin` 下），仓库只留一个指针；优点是不重复维护，代价是不随仓库版本化、换机器后需重建。

## 5. 问题四：索引范围

- `论文待处理` 有 9 个待处理条目；`论文_duplicates_review` 有 1 个 RMMSS 重复 bundle（目录名带 `__7a72f547` 后缀）。两者当前都不在任何索引中，且已知“内容相同”的判断并未逐字节核验。

待裁决选项：纳入索引并标注“待处理 / 重复”状态；只索引正式库内的 37 篇与 12 个仓库；或单独维护一份待处理清单而不混入主索引。

## 6. 问题五：库外 `origin/DFormer` clone

- 状态：停在 `814799b`，落后于作者最新 `e3273009`，且与我们的工作副本内容重复，引用时容易混淆。
- 待裁决选项：保留并在索引中标注其落后状态；更新到作者最新提交；或删除以避免与工作副本混淆。

## 7. 裁决前后的改动边界

裁决前（记录问题时）：未下载任何预训练权重，未安装或升级依赖，未修改论文库、外部 clone、数据集或项目配置，未提交、未推送。

裁决后（执行时，仅限已授权范围）：

- 未下载任何权重；未做预训练。
- 新建独立 conda 环境 `dformer` 并安装依赖；旧 `df2` 未改动。
- 把 7 个仍有效的索引与维护工具文件复制到 `D:\0Project\origin\_index\`（逐一 sha256 比对与源一致），并更新其中的索引路径、状态抬头与失效链接；未移动、未删除、未改写 `论文\`、`论文待处理\`、`论文_duplicates_review\` 与任何 clone。
- 仓库侧新增 `doc/guides/external-resources.md`、`doc/guides/environment.md`，更新项目指南、状态文件与本报告；未修改作者代码或配置。

## 8. 恢复点

五项裁决已执行完毕。后续从 `doc/state/current.md` 的“已裁决”与“下一步”继续：确定论文方向与第一轮研究问题 → 评审最小迁移清单 → 需要时才动数据与云端。任何权重下载、云资源操作、大规模重跑与推送仍需单独授权；`DFormerv2-Small` 的权重放置方式与 `extra_norms.*` 随机初始化两点先处理。
