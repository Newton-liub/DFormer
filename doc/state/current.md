# DFormer 项目实时状态

> 事实时间：2026-10-09。本文件是当前事实、授权边界与恢复点的唯一入口。上一阶段较长状态已摘要归档至 [工程准备前状态](../archive/2026-10-08-research-setup/project-state-before-setup.md)。详细环境与配置见 [运行环境](../guides/environment.md) 和 [新研究配置](../guides/research-setup.md)。

## 当前阶段

工程准备与此前论文库自动化工作已验收；**最后一轮11篇论文定点补充已获用户批准并完成一次正式保存**。11篇笔记追加、9篇局部阅读改为`needs_full_text`、15篇核心集及GeminiFusion缺失year/venue均已落盘；索引为**schema4/revision18、49篇**，管理视图和全部/核心/已有补充三范围AI导出已重建。本轮未设计模块、未改工程或工具代码、未启动训练或评价。论文搜索、PDF下载和重复阅读已停止，不重试DFormer++附录。第一阶段backbone仍为DFormerv2-S，DFormer++保留第二backbone候选。保存验收见[定点补充审核与保存结果](../reports/2026-10-09-paper-final-targeted-supplement-review.md)；本任务已结束，不自动进入实验。

## 已确认事实

### 仓库与模型

- 作者基线仍为 `e3273009b759b578945483828ff315d560be94c9`；研究分支 `research/dformerpp-clean-start`，2026-10-09本轮直接读取HEAD为 `79f8e81acd74b9f4ba65be9beaf6e3ee32736eac`（旧阶段`fe6cea2`仅为历史快照）。origin 为 `https://github.com/Newton-liub/DFormer.git`，upstream push 继续禁用；本轮没有commit/push。
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
- **2026-10-09交接只读核查更新：** 本地`D:\0Project\dataset\SUNRGBD`已存在，RGB/Depth/labels各10,335文件、train.txt5,285行；`DELIVER`的img/depth/hha/semantic各按train/val/test清点为3,983/2,005/1,897。SUN训练样本原生Depth为uint16、当前loader按8位灰度读；DeLiVER原Depth训练样本为uint8，官方`depth`模态实际读取HHA，编码/单位仍需确认。云端新数据是否落盘未核实。作者SUNRGBD配置eval_source=test.txt，正式实验前必须固定开发验证与official-test边界，未授权test评价；DeLiVER尚无DFormer适配器。完整工程、历史结果及证据边界见[新研究交接报告](../../临时/RESEARCH_HANDOFF.md)，本轮未下载、转换、训练或操作云资源。
- 论文库、索引与工具的真源在仓库外：`D:\0Project\origin\论文\`（49 篇 canonical）、`D:\0Project\origin\_index\`（唯一索引 JSON + 工具 + 备份 + 导出）。**该目录不属于任何 Git 仓库**，只有 store 自身的 JSON 备份。本仓库不复制第二份索引。
- 本地旧项目仍完整封存 `D:\0Project\DFormer-archive-20261007\`，不恢复MMFR / LER / natural_missing代码，不删除历史权重。

### 论文库（2026-10-09 完成）

- 现状：schema4 / **revision18** / **49 篇** / `next_lib_number` **50**，更新时间2026-10-09 14:41:53（UTC+8），索引文件SHA-256 `c7c0419e6bb3f1ddb4c554ae07d1ad9bf659ad191cc1a2809a152e201b96280e`；`backups\`为revision17 / 16 / 15，bak1与批准前F0逐字节一致。管理视图已按revision18重建。
- 本轮批准保存：2026-10-09 14:34用户明确“批准保存”，按联合清单核验111个不同文件及精确字段白名单后，经既有store一次提交（revision17→18）；实际仅16篇记录的11个supplement、9个reading_status、15个is_core及GeminiFusion的2个缺失书目override变化。原笔记前缀/来源、原始书目/摘要、人工编号/标签、auto与其它字段不变；未重复入库。保存回执：`_index\supplement-review\2026-10-09-final\SAVE_RESULT.json`。
- 此前书目补齐：按用户提供并明确批准的四篇 PDF 首页汇总截图，仅新增7个缺失的 `human.bibliographic_overrides.year/venue` 字段：LIB000046（2024 / Displays）、LIB000047（已有2026保持，仅补 Artificial Intelligence in Agriculture）、LIB000048（2026 / Sensors）、LIB000049（2026 / Journal of Imaging）。来源标为人工确认；该阶段未重新阅读PDF或联网独立核验，不改作者、DOI、正文、其他记录或阅读状态。
- 内容：49篇中12篇为`full_text_completed`（LIB000002 / 023 / 026 / 028 / 032 / 033 / 034 / 037 / 038 / 039 / 040 / 044，原批准状态不变），9篇为`needs_full_text`（LIB000001 / 014 / 024 / 042 / 045 / 046 / 047 / 048 / 049），28篇为`abstract_only`。旧全文状态语义为「正文已读并有证据笔记」，不表示附录/全部资产核完；本轮局部阅读没有新增全文完成。非空`human.supplement`21篇，核心15篇（清单见报告）；`alternate_bundles`1篇（LIB000002），`duplicate_review`1条（LIB000023）。论文库根目录仍51个目录=49篇canonical+2个容器目录。
- 本次入库（2026-10-09，按审核计划批准）：AGWNet → **LIB000046**，GeoSphere-DETR → **LIB000047**，RGB-D Mirror Segmentation → **LIB000048**，SPGNet → **LIB000049**。工具复核来源与索引后仅新增4条，旧记录未变；按清单校验复制资产，来源目录保留。
- AI导出已从实际保存的revision18用既有store导出实现重建（与可视化工具共用）：`_index\exports\PAPER_LIBRARY_FOR_AI.md`全部49篇、`PAPER_LIBRARY_FOR_AI_CORE.md`核心15篇、`PAPER_LIBRARY_FOR_AI_SUPPLEMENTED.md`已有补充21篇，schema4/格式1.0.2，核心/已有补充版header时间为2026-10-09T14:43:23+08:00；本轮定点核对全部版的实际header时间为2026-10-09T14:51:59+08:00，仍为revision18，11篇批准草稿经标题层级归一后均完整包含于对应条目。**全部版和已有补充版包含本轮11篇新笔记，核心版按15篇范围输出**；管理视图同为revision18。导出内容与既有renderer一致，未再次保存索引或轮转备份。
- 长期自动化实现：`.cursor/skills/paper-intake/SKILL.md`、`.cursor/skills/paper-supplement/SKILL.md`、`.cursor/rules/paper-library.mdc`、外部 `_index\tools\paper_workflow.py`（四子命令薄封装，复用现有扫描器与 store；无新增依赖、测试框架、数据库或常驻服务）。两个 Skill 与隔离验收均已通过。
- 两次真实试用：**入库** GeminiFusion → **LIB000045**（revision12→13；全库 dry-run 未发现任何既有记录变化，此前未实测的逐记录稳定性由此关闭）；**补充** `LIB000028`（MoSA）首次写入 `human.supplement` 10,459 字符（revision13→14，仅该篇 `supplement` 字段变化，阅读状态保持 `abstract_only`）。两次均只动批准范围内的记录。
- 历史阶段（详细材料在报告）：书目抽取与批量入库、新 7 篇年份 / 出版物 override、11 篇正文证据补充、44 篇质量审核（一次保存 37 个 override 字段）。见 [自动化最终收尾报告](../reports/2026-10-09-paper-library-automation-final.md)、[实施与隔离验收报告](../reports/2026-10-09-paper-library-automation-implementation.md)、[批量入库与审核](../reports/2026-10-08-paper-library-batch-intake.md)、[年份与出版物补充](../reports/2026-10-08-new-paper-bibliography-fill.md)、[质量审核](../reports/2026-10-08-paper-library-quality-audit.md)。
- 正文补充中已核实、仍有效的关键区分（详见各篇 supplement）：DFormerv2 正文为 softmax 后乘 prior，而当前官方代码为 softmax 前加 depth/spatial log bias 并用 bilinear 对齐 depth；DFormer++ 确有 depth 错位试验，但截至 2026-10-08 官方 README 权重列仍为 Coming soon；ECoLaF 已有 pixel-wise 预测层折扣、GeoDistill 已有 pixel-wise geometry gating 且 train/test 忽略 sensor depth。任何「局部可靠性 / 冲突融合」的新颖性仍未裁定。MoSA（LIB000028）补充记录了统计量式逐位置可靠性估计与编码器输出凸组合融合作为对照。

## 授权边界

- 2026-10-09 14:32用户确认论文补充已足够，停止新的论文搜索、PDF下载和重复阅读，不扩展论文范围，不重试DFormer++ supplementary。后续以`D:\0Project\origin\论文\`已有canonical材料为准，已入库论文不重新下载、转换或导入；仅在已有材料无法核实直接影响研究方案的关键问题时才考虑外部补充，本次不执行。现有11篇草稿、单篇计划及联合清单原样保留，未重新生成计划；14:34的明确保存批准已按下条执行。

- 本轮仅指定11篇定点取证与审核，不做实验/模块/工程改动；四篇LIB46–49已在库，不重复入库。14:34用户明确批准联合字段清单，已按共同F0一次store保存并验收。审核根`D:\0Project\origin\_index\supplement-review\2026-10-09-final\`，原`INDEX_FIELD_REVIEW.json`和11个单篇计划作为审批历史保留，执行结果另存`SAVE_RESULT.json`。**原计划均绑定revision17/F0，现已失效，不重复apply，不用新指纹绕过批准。**

- 论文库工作线的各阶段授权（入库预检与迁移、书目 override、指定 11 篇 supplement、44 篇质量审核、自动化实施与隔离验收、两次真实试用）均已按边界执行完毕，不属遗留事项。
- 2026-10-09 阅读状态更新为用户明确指示（**非工具自动判定**）：经一次 store 保存（revision14→15）把 12 篇有正文笔记的论文由 `abstract_only` 改为 `full_text_completed`；仅这 12 个 `human.reading_status` 字段变化，其余记录与根字段逐项不变，bak1 等于保存前索引。该状态按「正文已读并有笔记」定义，附录 / 补充材料未读与未核实项仍记录在各篇 supplement 内。
- 本轮11篇指定取证与审核之外仍未授权：GPU启动、训练 / 评价、下载新数据集、创建或销毁云资源、完整测试套件或全仓扫描、清理来源目录、扩展其它论文补充、自动批量联网补书目、修改工具代码、Git 提交或推送。本轮批准仅覆盖已执行的联合保存及派生视图/导出重建，后续索引写入需另行批准。
- SwanLab key 不得进入 Git；训练代码、config 与索引均不得含凭据。
- 研究方法、最终验证 / test协议、新模块、正式 baseline 与候选训练需另行确认。旧训练授权与旧 baseline 资格不沿用。

## 下一步与恢复点

1. 本轮论文补充与批准保存已结束，索引revision18、三范围导出及保存回执均已验收；保留原审核材料，不重复apply或重新生成计划，不继续搜索/下载/阅读。详见[定点补充审核与保存结果](../reports/2026-10-09-paper-final-targeted-supplement-review.md)。等待用户下一步指示；第一阶段实验仍需另行固定SUNRGBD开发验证/test边界、输入数值协议，核实云端数据并获得训练授权。
2. 此前论文库自动化收尾提交`66cb9f3`及当时的单行状态更正仅为历史记录，详见[自动化最终收尾报告](../reports/2026-10-09-paper-library-automation-final.md)。当前实际HEAD以上述`79f8e81`为准；本轮状态和审核报告尚未提交，不amend旧提交、不另行commit/push。
3. 论文库自动化建设已由上级审核通过并正式结束，后续不再开发或测试论文库工具；工作重心转回 DFormer / MMFR 研究主线。
4. 固定实验合同后再授权同 pretrained 的新 baseline 与候选；当前环境未做正式训练验收，不宣称训练已通过。
5. 本机研究环境与研究配置已可用；云端持久工程 / 环境 / 资产均在上述路径，云实例下一次从无卡启动恢复。
6. 提交身份用仓库 `git rev-parse HEAD` 获取；同步通过 Git bundle，origin 分支不更新。

## 阻塞与待确定项

- 工程基础检查无已发现阻塞；本地SUNRGBD/DeLiVER已落盘。后续训练前置条件仍包括输入数值协议与开发验证/test协议固定、云端数据/同HEAD同步核实及GPU授权，完整新baseline训练尚未验收，因此不开始训练。
- 本轮真正未决：DFormer++官方supplementary访问418、未读；MUSeg标签原ID0=background、1–15前景，模型0→255/1–15→0–14，以及uint16→uint8固定全局量化和dev-v1采集组隔离已有归档报告、转换代码与当前dataset_meta依据，本轮已核对；原深度物理单位仍未知。SUNRGBD三个训练样本已CPU核实8位读取等于uint16高字节，现有ValPre归一化后的输入范围已记录；文件来源、物理编码及DeLiVER单通道Depth缩放/无效值仍未确认。官方pretrained的29个Geo权重中存在负spatial/depth系数，不能把全部bias解释为距离惩罚；旧Oracle-A的精确动作、checkpoint与配对负面结果已定点补入[交接报告末节](../../临时/RESEARCH_HANDOFF.md#八最后定点补充2026-10-09)，本轮不重跑验证；UMFNet Table3/主结果数据集数值冲突、SGMA Eq3维度疑点、GeoSphere Fig9 clean-FPPI差异已记录。GeminiFusion正式ICML2024书目已取证且缺失year/venue已按批准保存。其它历史待处理项保持：MoSA提取不一致、UMIS-Mine待审副本与Rules标签页未目视确认，均不自动扩展处理。
- DeLiVER本地公开包目录已核实，DFormer适配尚未实现；完整六视角公开下载入口未核实，不将front-view公开包冒称完整六视角数据。
- 独立维护事项（上级决定暂不处理）：外部 `D:\0Project\origin\_index\` 未纳入任何版本控制，`paper_workflow.py` 等工具与索引只有 store 自身的 JSON 备份保护。
