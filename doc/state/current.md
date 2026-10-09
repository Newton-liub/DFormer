# DFormer 项目实时状态

> 事实时间：2026-10-09。本文件是当前事实、授权边界与恢复点的唯一入口。上一阶段较长状态已摘要归档至 [工程准备前状态](../archive/2026-10-08-research-setup/project-state-before-setup.md)。详细环境与配置见 [运行环境](../guides/environment.md) 和 [新研究配置](../guides/research-setup.md)。

## 当前阶段

工程准备与论文库工作线（书目抽取、批量入库、11 篇正文补充、44 篇质量审核、长期自动化的 Skill / Rule / 工具实施与隔离验收、两次真实试用）**全部完成并已验收**；论文库现为 schema4 / revision14、45 篇、`next_lib_number` 46。**未设计新模块、未下载新数据集、未启动训练或评价**。第一阶段 backbone 固定为 **DFormerv2-S**，保留作者最新 DFormer 仓库及 DFormer++ 代码 / 配置（后者为第二 backbone / generalization 候选），不回退或重建旧 DFormerv2 仓库。论文库工作线已收尾，下一步需要用户单独授权数据下载与第一阶段实验。

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
- 用户 `临时/key.txt` 已读取用于登录、未输出、未删除，且加入 `.gitignore`。用户可自行删除。`.serena/`、`.playwright-mcp/` 与默认本地 SwanLab 产物也被忽略。
- `utils/train.py` 原来只有 TensorBoard，本轮只增加可选日志调用，实现在 `research/tracking.py`；global rank0 记录 epoch平均loss、LR、验证mIoU、epoch / update、实验名称、核心config / CLI。未启用 `swanlab_enabled` 的旧配置不加载SDK。训练入口 `--help` 在本地正常，不启动训练。
- 两端 `pip check` 无 broken requirements；代码仅作定点语法 / 接入检查，没有新建测试文件、没有运行完整测试套件。

### 云端工程与资产

- 实例仍为 `cpod-1vbh7faqcauq`（cn-bj2-03，50GB系统盘）；准备窗口实际为无卡A档：2CPU / 4GiB / GPU=0，未启动GPU。无卡不是免费状态，平台返回 InstancePrice0.13；准备窗口设有12:35（UTC+8）计划关机保险。
- 当前工程：`/root/rivermind-data/DFormer`，已从现用 fork 的研究分支 clone。同步方式为本地最终提交后通过 Git bundle fetch；**不推送远端**。bundle 传递 Git 对象，不用旧源码快照覆盖工程；收尾必须核对两端 HEAD。
- 旧云端工程完整移动到 `/root/rivermind-data/DFormer-archive-20261008`，原HEAD `7d6246a7db77f77e26e20d84074dff6e191e9fa5`，其 cloud / outputs / experiments / checkpoints 等均保留。独立旧结果 `/root/rivermind-data/cloud`、现有 MUSeg `/root/rivermind-data/dataset` 与 pretrained 库原样保留。
- 复用 `/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、scipy1.15.3、timm1.0.28；兼容相同DFormer模型与config，不重复建环境。官方权重通过工程相对路径软链接至已有 pretrained 文件，无需上传或下载。
- screen4.09.00已可用，短命令启动正常；无训练会话，不保留常驻服务。下一次使用按 [云端流程](../guides/cloud.md) 从无卡重新启动并核对状态，GPU训练需重新授权。

### 数据集与外部资源

- 下载指南：`D:\0Project\DFormer\临时\DATASET_DOWNLOAD_GUIDE.md`。含SUNRGBD作者整理版单文件入口 / Windows与Linux命令 / 路径 / split / depth协议；DeLiVER官方 front-view `DELIVER.tar.gz` / RGB-depth-HHA区别 / 天气及failure筛选 / 官方评价代码；NYUv2第二阶段可选；MUSeg最终真实矿井验证复用现有数据。
- SUNRGBD 与 DeLiVER 均未下载。作者SUNRGBD配置eval_source=test.txt，正式实验前必须固定开发验证策略及official-test使用边界，未授权test评价。DeLiVER尚未写DFormer适配器，留作正式benchmark接入任务。
- 论文库、索引与工具的真源在仓库外：`D:\0Project\origin\论文\`（45 篇 canonical）、`D:\0Project\origin\_index\`（唯一索引 JSON + 工具 + 备份 + 导出）。**该目录不属于任何 Git 仓库**，只有 store 自身的 JSON 备份。本仓库不复制第二份索引。
- 本地旧项目仍完整封存 `D:\0Project\DFormer-archive-20261007\`，不恢复MMFR / LER / natural_missing代码，不删除历史权重。

### 论文库（2026-10-09 完成）

- 现状：schema4 / **revision14** / **45 篇** / `next_lib_number` **46**，更新时间 2026-10-09 09:21:07（UTC+8），索引指纹 `94af0e46f786483f56a50b2f62a61b167644318a60c23c25fcb34d67cac483de`；管理视图 `PAPER_LIBRARY_INDEX.md` 为 `d7b253a8…`；`backups\` 为 revision13 / 12 / 11。
- 内容：45 篇阅读状态全部仍为 `abstract_only`；非空 `human.supplement` **12 篇**；`alternate_bundles` 1 篇（LIB000002 在 `MMFR-附件\` 内的另一抽取）；`duplicate_review` 1 条（LIB000023，已移入 `论文_duplicates_review`）。论文库根目录 47 个目录 = 45 篇 canonical + 2 个容器目录。
- AI 阅读导出（`_index\exports\`）已同步至 revision14、格式1.0.2：全部 45 / 核心 0 / 已有补充 12；派生文件，不影响权威 JSON。
- 长期自动化实现：`.cursor/skills/paper-intake/SKILL.md`、`.cursor/skills/paper-supplement/SKILL.md`、`.cursor/rules/paper-library.mdc`、外部 `_index\tools\paper_workflow.py`（四子命令薄封装，复用现有扫描器与 store；无新增依赖、测试框架、数据库或常驻服务）。两个 Skill 与隔离验收均已通过。
- 两次真实试用：**入库** GeminiFusion → **LIB000045**（revision12→13；全库 dry-run 未发现任何既有记录变化，此前未实测的逐记录稳定性由此关闭）；**补充** `LIB000028`（MoSA）首次写入 `human.supplement` 10,459 字符（revision13→14，仅该篇 `supplement` 字段变化，阅读状态保持 `abstract_only`）。两次均只动批准范围内的记录。
- 历史阶段（详细材料在报告）：书目抽取与批量入库、新 7 篇年份 / 出版物 override、11 篇正文证据补充、44 篇质量审核（一次保存 37 个 override 字段）。见 [自动化最终收尾报告](../reports/2026-10-09-paper-library-automation-final.md)、[实施与隔离验收报告](../reports/2026-10-09-paper-library-automation-implementation.md)、[批量入库与审核](../reports/2026-10-08-paper-library-batch-intake.md)、[年份与出版物补充](../reports/2026-10-08-new-paper-bibliography-fill.md)、[质量审核](../reports/2026-10-08-paper-library-quality-audit.md)。
- 正文补充中已核实、仍有效的关键区分（详见各篇 supplement）：DFormerv2 正文为 softmax 后乘 prior，而当前官方代码为 softmax 前加 depth/spatial log bias 并用 bilinear 对齐 depth；DFormer++ 确有 depth 错位试验，但截至 2026-10-08 官方 README 权重列仍为 Coming soon；ECoLaF 已有 pixel-wise 预测层折扣、GeoDistill 已有 pixel-wise geometry gating 且 train/test 忽略 sensor depth。任何「局部可靠性 / 冲突融合」的新颖性仍未裁定。MoSA（LIB000028）补充记录了统计量式逐位置可靠性估计与编码器输出凸组合融合作为对照。

## 授权边界

- 论文库工作线的各阶段授权（入库预检与迁移、书目 override、指定 11 篇 supplement、44 篇质量审核、自动化实施与隔离验收、两次真实试用）均已按边界执行完毕，不属遗留事项。
- 仍未授权：推送远端、GPU启动、训练 / 评价、下载新数据集、创建或销毁云资源、完整测试套件或全仓扫描、清理来源目录、批量补充论文、自动联网补书目、修改工具代码、Git 提交或推送（收尾提交需上级审核后再执行）。
- SwanLab key 不得进入 Git；训练代码、config 与索引均不得含凭据。
- 研究方法、最终验证 / test协议、新模块、正式 baseline 与候选训练需另行确认。旧训练授权与旧 baseline 资格不沿用。

## 下一步与恢复点

1. 需要第一阶段实验时，先单独授权 SUNRGBD 下载，按指南放置作者整理版，并固定开发验证策略与 official-test 使用边界。
2. 论文库工作线已收尾：收尾提交范围见 [自动化最终收尾报告](../reports/2026-10-09-paper-library-automation-final.md)；已按上级授权在 `research/dformerpp-clean-start` 完成**一次本地提交**（`66cb9f3`，12 个文件，**未推送**，origin 仍停在 `fe6cea2`）。本文件在提交后有一次**单行状态更正，有意保留未提交**，等下一次正常 Git 提交时一并纳入；不要 amend `66cb9f3`、不要为它单独创建提交、不要推送。
3. 论文库自动化建设已由上级审核通过并正式结束，后续不再开发或测试论文库工具；工作重心转回 DFormer / MMFR 研究主线。
4. 固定实验合同后再授权同 pretrained 的新 baseline 与候选；当前环境未做正式训练验收，不宣称训练已通过。
5. 本机研究环境与研究配置已可用；云端持久工程 / 环境 / 资产均在上述路径，云实例下一次从无卡启动恢复。
6. 提交身份用仓库 `git rev-parse HEAD` 获取；同步通过 Git bundle，origin 分支不更新。

## 阻塞与待确定项

- 工程基础检查无已发现阻塞。后续训练前置条件仍包括 SUNRGBD 数据落盘、验证 / test 协议固定及 GPU 授权；当前不具备这些条件，因此不开始训练。
- 论文库待处理（均不阻塞使用）：LIB000045（GeminiFusion）正文不含 DOI / arXiv / venue，年份与出版物仍缺；LIB000028（MoSA）笔记内记录的提取不一致（Table 4 标签错位、Table 1 错行、式(20) 括号、Fig.5 均值 0.78 与图内 0.80 等）未消除，Eq.4 的维度疑点按上级要求只作记录；UMIS-Mine（旧 LIB000032）的待审副本仍留原处，历史来源当前不可读，处理前需先提供现位置并单独裁决资产差异；Rule `paper-library` 的 Rules 标签页未目视确认。
- DeLiVER仅记录下载与官方协议，DFormer适配尚未实现；完整六视角公开下载入口未核实，不将front-view公开包冒称完整六视角数据。
- 独立维护事项（上级决定暂不处理）：外部 `D:\0Project\origin\_index\` 未纳入任何版本控制，`paper_workflow.py` 等工具与索引只有 store 自身的 JSON 备份保护。
