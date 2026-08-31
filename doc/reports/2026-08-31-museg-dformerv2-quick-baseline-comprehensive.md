# MUSeg DFormerv2-S Quick-B0 总体工作汇报

- 汇报周期：2026-08-17 至 2026-08-31
- 报告对象：课题组组会
- 报告范围：`doc/plans/MUSeg-DFormerv2快速Baseline/` 所定义的四项任务及其必要历史依据
- 上游基础：作者仓库 `VCIP-RGBD/DFormer` 当前公开 `main` 提交 `814799bb1f39eb380f72fdea1cd591f2cc27b6aa`（2025-11-11）
- 训练身份：提交 `3975f7d66c78e9bed6b9053071bb274199d550e9`，protocol SHA-256 `6822e4cdfd9c6985c323123fc4d24a9f06ed269fada55203b1707fc5ab612bbd`
- 当前状态：single-seed RGB development Quick-B0 已完成，最终 checkpoint 为 epoch 420

## 一、结论先行

本阶段已经完成从作者当前公开 DFormerv2 代码与官方 DFormerv2-S 预训练权重出发，到 MUSeg 15 类 RGB-D 语义分割任务的完整适配、训练、评估和证据收口。最终模型在冻结的 318 张 `val-dev` 上取得：

- mIoU：`58.79%`；
- mAcc：`69.91%`；
- mF1：`72.73%`；
- 最终 checkpoint：epoch 420 的 `selector-epoch-420.pth`；
- checkpoint SHA-256：`f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。

这意味着项目已经有了一个身份清楚、可以复核、可以供后续模块公平比较的共同起点。训练期单尺度验证最高的是 epoch 480，但五尺度翻转主评估最高的是 epoch 420，因此本轮预先设置的“低成本筛选候选，再严格统一重排”确实避免了选错最终模型。

当前结果是一个随机种子的开发基线，不是三随机种子统计结论，也不是 MUSeg official test（官方测试集）结果。official test 继续保持 `sealed_unread`，即开发阶段封存未读。报告中的历史 BGR、固定尺寸 resize 和 sliding 等数字只用于说明输入与评估口径的敏感性，不能把跨协议差值解释为 RGB、裁剪窗口或某个模块单独带来的性能提升。

## 二、关键概念与判断边界

- **DFormerv2-S**：作者公开的 Small 规模 RGB-D 语义分割模型，同时使用彩色图和深度图。本项目保留其 RGB-D 编码与 HAM 解码结构，并面向 MUSeg 做数据、训练和评估适配。
- **Baseline / B0**：后续方法比较时保持不变的基础模型。这里的 B0 是 single-seed development B0，只用于内部开发和初步消融。
- **Checkpoint**：训练到某个 epoch 时保存的模型参数快照。最后一个 checkpoint 不一定是评估最好的 checkpoint。
- **Protocol**：训练开始前冻结的数据、输入、随机种子、预算、模型选择和评估规则。它用于防止看到结果后再改变口径。
- **Selector**：训练期间每 10 epoch 运行一次的低成本单尺度验证，只负责留下候选，不负责决定最终模型。
- **主 evaluator**：最终评估程序。它对每张图使用五个尺度及其水平翻转，共 10 个 view，并把预测恢复到原始标签尺寸后计分。
- **mIoU**：15 个类别各自的交并比再取平均，是本报告的主分割指标；mAcc 是各类准确率的平均，mF1 是各类 F1 的平均。
- **原始 Label 网格计分**：模型输出最终恢复到每张标签图原来的高和宽后再比较。训练时裁剪到 `480×640`，不代表验证指标也在 `480×640` 上计算。

## 三、作者最新公开版本的复现基础

### 3.1 上游代码身份

2026-08-31 通过只读 `git ls-remote` 直接核验，作者仓库 `https://github.com/VCIP-RGBD/DFormer.git` 的公开 `main` 当前指向：

`814799bb1f39eb380f72fdea1cd591f2cc27b6aa`

本地 `upstream/main` 指向同一提交，且该提交是本项目当前分支的祖先。这说明 MUSeg 扩展建立在作者当前公开主线之上，而不是从无法追溯的代码副本开始。

这里的“复现”指复用并核验作者公开模型实现、预训练资产和训练/测试方法，再把它们适配到 MUSeg。由于数据集、类别体系和开发协议不同，本项目不声称已经在 NYUv2 或 SUNRGBD 上重新跑出论文表格，也不把当前结果称为 MUSeg 论文中未公开测试代码的严格数值复现。

### 3.2 官方预训练权重身份

本项目使用 `DFormerv2_Small_pretrained.pth`：

- 大小：`110,203,103` bytes；
- SHA-256：`19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`；
- 上游来源：官方 Hugging Face 资产 `bbynku/DFormerv2`；
- 上游输入语义：PIL RGB 与 RGB 顺序 ImageNet mean/std。

本项目权重的大小和 SHA-256 与官方资产完全一致。这个核验回答了“初始化权重来自哪里、训练时看到的颜色通道是什么”两个基础问题。

## 四、面向 MUSeg 的适配工作

### 4.1 数据隔离

MUSeg 官方 train 共 1,595 张，开发阶段冻结为：

- `train-dev`：1,277 张，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；
- `val-dev`：318 张，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；
- official test：1,576 张，开发期不用于 split、模型选择、超参数或阈值选择。

划分采用 location-group 约束，同一位置组不跨训练和验证，降低相邻场景泄漏带来的虚高风险。

### 4.2 RGB 输入契约

历史 MUSeg loader 通过 OpenCV 读出 BGR 数组，而官方 DFormerv2-S 预训练使用 RGB。新 B0 明确执行：

1. OpenCV 读取；
2. BGR 转 RGB；
3. 按 RGB 顺序应用 ImageNet mean `[0.485, 0.456, 0.406]` 和 std `[0.229, 0.224, 0.225]`。

选择 RGB 的理由是让 MUSeg 微调输入与预训练权重已经学习的通道含义一致。这是变量最少的工程选择，不是“RGB 在 MUSeg 上统计显著优于 BGR”的实验结论；要回答后一个问题，必须对 RGB 和 BGR 从同一 pretrained 做成对重训。

### 4.3 Depth 输入

Depth 继续使用现有全局量化 8-bit 图，单通道复制为 3 通道，每通道 mean `0.48`、std `0.28`。当前 B0 没有加入 depth-validity mask/gating，因此后续 B2 不能写成已经存在于本基线。

### 4.4 训练增强与裁剪窗口

训练采用作者公开方法中的随机尺度思想：

- 随机尺度：`0.5、0.75、1.0、1.25、1.5、1.75`；
- 水平镜像概率：`0.5`；
- 随机裁剪：高 `480`、宽 `640`；
- RGB/Depth 缩放：双线性；
- Label 缩放：最近邻；
- 尺寸不足时：normalization 后 RGB/Depth 补 0，Label 补 255。

`480×640` 是训练窗口，不是统一的验证输出尺寸。RGB、Depth 和 Label 始终同步变换，避免不同模态发生空间错位。

### 4.5 优化与运行预算

- 优化器：AdamW；
- base learning rate：`6e-5`；
- weight decay：`0.01`；
- poly power：`0.9`；
- warmup：10 epoch；
- 训练总长：500 epoch；
- batch size：10；
- seed：`772961337`；
- AMP：开启；
- GPU：单张 RTX 4090。

理论 loop attempt 为 `500 × 128 = 64,000`。结构化遥测记录 attempted/completed/skipped 为 `64,000 / 63,971 / 29`，满足 `attempted = completed + skipped`。29 次跳过由运行时直接记录，不再像历史遥测那样只能在训练后用差值猜测。

## 五、主评估如何工作

主评估模式为 `msflip-whole-original-grid-v1`：

1. 从原始 RGB/Depth 尺寸分别缩放到 `0.5、0.75、1.0、1.25、1.5`；
2. 每个尺度运行原图和水平翻转，共 10 个 view；
3. normalization 后只在右侧和底部补齐到 32 的倍数；
4. 每个 view 的 logits 先取消翻转、去除 padding，再恢复到原始 Label 网格；
5. 10 份 pre-softmax logits 使用 FP32 平均，最后取 argmax；
6. FP32 前向，TF32 disabled，batch size 1。

五尺度翻转按输入像素量约为单尺度整图的 `11.25` 倍，因此训练期间没有每 10 epoch 都运行主评估，而是使用两级选择：

- 每 10 epoch 用单尺度 original-full selector 保留 top 3；
- 额外持续保存 `latest.pth`；
- 训练结束按 checkpoint SHA-256 去重；
- 对最多 4 个候选统一运行主 evaluator；
- 最终按主 mIoU 选择 B0。

最大样本尺度 1.5 的本地技术检查已先行通过：原图 `932×1082`，缩放后 `1398×1623`，补齐后 `1408×1632`；原图/翻转两个 view 未 OOM。该检查只证明资源可行，不产生模型指标。

## 六、训练、评估和证据收口过程

### 6.1 本地实现与定点检查

完成了独立 Quick-B0 config、protocol v3、RGB 输入契约、top-3 候选管理、恢复一致性、FP32 五尺度翻转 evaluator、official-test 拒绝和生命周期控制器。任务范围内聚焦 CPU 检查为 66 passed，最大样本 GPU 技术检查通过。

这些检查证明实现满足冻结契约，不等同于训练结果。

### 6.2 无卡自动关机门禁

历史运行暴露过“训练结束但实例未及时停止”的计费风险。正式训练前，项目使用 `GPU=0`、2 vCPU、4 GiB 内存的实例模式模拟：

`workload 完成 → 生成报告 → 取回证据 → 核验 SHA-256 → 控制面 stop → 等待 Stopped`

无卡门禁通过，且另设控制面 schedule 作为断联兜底。这样即使训练验收失败，计费停止也不会依赖“结果是否通过”。

### 6.3 正式训练

第一次 Screen 启动没有导出仓库 `PYTHONPATH`，在导入 `utils.metrics_new` 前退出，未加载模型、未进入训练、未产生指标。保留失败现场后，只修正启动环境并显式导出 `PYTHONPATH`，随后在同一冻结协议下重新启动。

正式训练从 `2026-08-30T14:38:48Z` 运行至 `2026-08-31T04:16:27.983518Z`，约 13 小时 37 分 40 秒；500/500 epoch 完成，`train.exit_code=0`，未自动增加 epoch 或 seed。

### 6.4 证据取回与停止实例

训练完成后，先下载 active seed、materialized protocol、preflight、Screen 日志、首次失败现场和 4 个候选，再在本地重算大小、SHA-256、run ID、seed、Git commit、split 和 optimizer 身份。核验通过后请求 CompShare 控制面 stop，并于 `2026-08-31T04:33:31.827620Z` 复查实例为 `Stopped`。schedule 未触发，也没有重复发送 stop。

### 6.5 本地主评估

4 个候选在本地 RTX 5060 Laptop、冻结训练提交的 detached 干净 worktree 和 `df2` 环境中串行评估。第一次命令同样因未导出 `PYTHONPATH` 在模型导入前退出，未产生指标；只修正环境后从第一个候选完整重跑。

4 项 evaluator 内部计时合计 `5,642.625` 秒，即 `94.044` 分钟，全部完成且未 OOM，明显低于 8 小时异常停手上限。

## 七、主评估结果

### 7.1 四个候选的最终排序

1. **epoch 420**：selector mIoU `56.39%`；主 mIoU `58.79%`、mAcc `69.91%`、mF1 `72.73%`；耗时 `1,402.672` 秒。
2. **epoch 440**：selector mIoU `56.10%`；主 mIoU `58.73%`、mAcc `69.54%`、mF1 `72.67%`；耗时 `1,405.922` 秒。
3. **epoch 480**：selector mIoU `56.87%`；主 mIoU `58.43%`、mAcc `69.34%`、mF1 `72.44%`；耗时 `1,427.140` 秒。
4. **epoch 500**：latest，没有独立 selector mIoU；主 mIoU `57.68%`、mAcc `68.84%`、mF1 `71.81%`；耗时 `1,406.891` 秒。

selector 把 epoch 480 排在第一，但主 evaluator 把它排在第三。epoch 420 与 epoch 440 的主 mIoU 只差 `0.06` 个百分点，但按预登记规则，epoch 420 仍是明确胜者，无需触发 mAcc 或更早 epoch 的同分裁决。

### 7.2 epoch 420 的 15 类结果

以下顺序为 IoU / accuracy / F1，单位均为百分比：

- person：`58.54 / 72.41 / 73.85`；
- cable：`61.41 / 78.73 / 76.10`；
- tube：`67.71 / 74.66 / 80.74`；
- indicator：`73.26 / 80.20 / 84.57`；
- metal fixture：`53.65 / 74.11 / 69.83`；
- container：`24.67 / 26.83 / 39.58`；
- tools & materials：`69.91 / 90.41 / 82.29`；
- door：`86.01 / 86.63 / 92.48`；
- electrical equipment：`59.08 / 82.16 / 74.28`；
- electronic equipment：`56.31 / 60.95 / 72.05`；
- mining equipment：`41.92 / 48.97 / 59.07`；
- anchoring equipment：`65.46 / 79.70 / 79.13`；
- support equipment：`40.23 / 43.57 / 57.38`；
- rescue equipment：`43.56 / 55.51 / 60.69`；
- rail area：`80.10 / 93.84 / 88.95`。

15 类指标全部为有限值，没有 NaN 或空类归约。door 和 rail area 表现较强；container 的 IoU 最低，为 `24.67%`，mining/support/rescue equipment 也低于总体水平。后续模块报告必须同时看总体和逐类结果，避免总体 mIoU 掩盖薄弱类别。

## 八、各项对比与正确解释

### 8.1 训练期 selector 与主 evaluator

三项具有 selector 分数的候选中：

- epoch 480：selector 第一，主评估第三；
- epoch 420：selector 第二，主评估第一；
- epoch 440：selector 第三，主评估第二。

这说明单尺度筛选和五尺度翻转最终评估相关但不等价。后续 v2 已把候选扩大为 top 8 + latest，最多 9 个去重 checkpoint，以降低漏掉主评估优胜点的概率；它不能保证完全消除排序差异，也不会回写本轮 v1 结论。

### 8.2 历史推理几何对比

历史 legacy BGR epoch-460 best checkpoint 在同一 318 张 `val-dev`、原始 Label 网格上得到：

- original-full：mIoU `52.98%`；
- resize-480x640：mIoU `56.31%`；
- sliding-480x640：mIoU `51.89%`。

三种 geometry（推理几何，即输入缩放、整图或滑窗方式及输出恢复规则）的跨度为 `4.42` 个百分点。固定 resize 最高，但它改变长宽比；该数字只能说明模型对推理几何敏感，不能因为分数最高就追认为更正确的主协议。

历史 epoch-500 final 在 resize/sliding 下分别为 `56.73% / 52.08%`。这进一步说明训练期 original-full best 不保证在其他 evaluator 下继续排名第一。

### 8.3 历史颜色诊断

固定历史 best checkpoint 的 original-full 三臂结果为：

- legacy BGR：mIoU `52.98%`；
- RGB + RGB mean/std：mIoU `33.85%`；
- BGR + 反向 mean/std：mIoU `49.53%`。

历史 checkpoint 是在 BGR 输入上微调得到的，推理时直接更换通道会破坏它已经适应的输入分布。三臂诊断证明“颜色契约必须被显式记录”，但不能回答“重新训练后的 RGB 和 BGR 哪个更好”。本轮 RGB B0 从官方 pretrained 独立训练，因此不能把它与历史 BGR 数字做单变量性能归因。

### 8.4 历史 seed1 与当前 Quick-B0 的协议差异

历史 seed1 和当前 B0 共享 DFormerv2-S、同一开发划分量级和 500 epoch，但至少存在以下协议差异：

- 历史为 legacy BGR，当前为与 pretrained 对齐的 RGB；
- 历史训练尺度固定 `[1.0]`，当前采用六个随机尺度；
- 历史在线验证和后评估口径不统一，当前最终模型由预登记五尺度翻转主 evaluator 决定；
- 历史遥测未直接分开 attempted/completed/skipped，当前已分开记录；
- 当前增加候选清单、哈希裁决和云生命周期门禁。

因此，历史 original-full `52.98%` 与当前主 mIoU `58.79%` 相差 `5.81` 个百分点，但这个差值是多项协议共同变化后的结果，不能写成 RGB 增益、随机裁剪增益或主 evaluator 增益。

## 九、主要问题、原因与解决结果

### 9.1 RGB 与 BGR 含义不一致

- 问题：OpenCV 默认 BGR，而官方 pretrained 使用 RGB。
- 原因：颜色数组顺序没有在早期协议中显式绑定。
- 处理：追溯官方 Hugging Face 权重及预训练数据代码，冻结 BGR→RGB 和 RGB ImageNet normalization。
- 结果：当前 B0 的输入语义与初始化权重一致；仍不宣称 RGB 性能胜过成对重训 BGR。

### 9.2 把训练裁剪误解成验证尺寸

- 问题：`480×640` 曾被概括成统一模型输入，容易把训练 crop、固定 resize 和原图验证混为一谈。
- 原因：训练变换和指标网格没有分层说明。
- 处理：训练明确随机尺度后裁剪；主评估明确原始尺寸缩放、padding、unflip、去 padding 和恢复原始 Label 网格。
- 结果：所有主指标都绑定明确 geometry，不再用一个尺寸概括全部评估。

### 9.3 A1：空有效像素导致非有限 loss

MUSeg 背景标签映射为 ignore 255。全背景图或某个 crop/rank 只有背景时，有效像素集合可能为空；旧逻辑对空张量调用 `mean()` 会产生非有限值。张量级测试确认：0 个有效像素时旧归约非 finite，1/5/100 个有效像素时正常。

### 9.4 B1：`safe_masked_mean` 稳定性修复

B1 在非空集合上保持普通均值，在空集合上返回 `values.sum() * 0.0`。这样 forward 得到有限 0，并保持计算图连接，backward 得到有限零梯度。B1 已进入当前 B0，是必要的数值稳定性修复；由于没有关闭 B1 的匹配训练对照，不能把 mIoU 或稳定完成归因于 B1 的性能收益。

### 9.5 `PYTHONPATH` 启动环境

云端训练和本地主评估首次命令都曾因没有导出仓库 `PYTHONPATH` 而在模型导入前退出。两个现场都未进入有效计算、未产生指标。处理方式都是只修正启动环境，保持代码、数据、协议、checkpoint 和 evaluator 参数不变后完整重跑。

### 9.6 自动关机和费用风险

历史经验表明，实例操作系统内关机或训练验收结束都不能单独证明平台停止计费。当前链路把“研究验收”和“停止实例”分开：无论训练成功、失败还是中止，都先取回必要证据，再调用控制面 stop 并等待 `Stopped`；schedule 只作为断联兜底。

### 9.7 Checkpoint 选择

历史 final 不是验证峰值，当前 selector 第一也不是主评估第一。解决方案是保存 top-k + latest，并用最终 evaluator 重排。当前 v1 已验证这项设计；后续 v2 增加到 top 8 + latest，是风险控制改进，不改变当前最终 epoch 420。

## 十、时间、资源和费用

### 10.1 已核验时间与资源

- 正式 Quick-B0 训练：约 `13 小时 37 分 40 秒`；
- 正式启动到实例确认 `Stopped`：约 `13 小时 54 分 44 秒`；
- 本地 4 候选主评估：`94.044 分钟`；
- 云端训练 GPU：RTX 4090；
- 本地主评估 GPU：RTX 5060 Laptop；
- 主评估每个候选峰值 allocated/reserved：`5,276,340,224 / 7,201,619,968` bytes；
- 4 个主评估均未 OOM。

### 10.2 训练前预算

按历史参考价 `1.98 元/小时`：

- 24–28 小时计划预算：`47.52–55.44 元`；
- 32 小时兜底上限：`63.36 元`。

实际训练明显快于计划预算，且主评估转到本地完成，避免继续占用云端 RTX 4090。

### 10.3 可计算的参考估算

在假设整个时段均按 `1.98 元/小时` 计费、没有折扣或最低计费粒度差异的前提下：

- 仅按正式训练时段估算：约 `26.98 元`；
- 按正式启动到平台确认 `Stopped` 的时段估算：约 `27.55 元`。

这两个数字是根据时间和历史参考费率计算的估算值，不是控制台账单。实例在正式启动前已用于 schedule、preflight 和启动检查，实际订单可能还包含更早的运行时段；云盘与镜像在计算实例停止后也可能继续计费。

### 10.4 当前无法确认的费用

现有证据没有最终控制台订单、实际计费类型、折扣、云盘单价或镜像费用，因此本报告不写“实际总花费”。若组会需要财务口径，应补充 CompShare 最终订单截图或账单导出，再把估算替换为核实值。

## 十一、A1、B1、A2 与下一步 B2

A1/B1 与 A2/B2 是两条不同问题链：

- A1/B1 处理标签有效像素为空时的 loss 数值稳定性；
- A2/B2 研究 Depth 无效或被额外破坏时，几何先验是否需要显式 validity 处理。

### 11.1 当前状态

- A1：空集合风险已证实；
- B1：`safe_masked_mean` 已实现、验证并进入 B0；
- A2：历史 epoch-10 checkpoint 上的 16 张样本 pilot 只观察到 q=0.3 下降 `0.1107` 个百分点、q=0.5 下降 `0.3337` 个百分点，属于中间态，不能替代当前 RGB B0 上的正式 A2；
- B2：depth-validity mask/gating 尚未实现，也未进入当前 B0。

### 11.2 B2 的建议执行顺序

1. 在当前 epoch 420 RGB B0 上重新冻结正式 A2 协议，只使用 `val-dev`；
2. 固定 channel order、normalization、checkpoint、mask 类型、corruption 强度、mask seed 和 evaluator；
3. 比较 q=0、0.1、0.3、0.5，至少包含 block/random mask 与 zero/median 对照；
4. 只有 `A2-pass + 用户明确授权` 才冻结 B2 数学和 shape 规格；
5. B2 先做 all-valid 等价性、mixed/all-invalid 金标准、strict checkpoint 和 forward/backward；
6. 再做不更新权重的 `B2-zero-train`；通过后另获授权，才做 B0/B2 对称的短训或完整独立训练。

B2 的核心目标是：两个 depth patch 都有效时保留原 depth decay；任一无效时移除该 pair 的 depth decay，但保留 positional decay。人工 corruption 可以作为工程开发门槛；如果自然无效深度证据不足，结论只能限定在人工条件，不能外推为现实部署鲁棒性。

本报告不授权正式 A2、B2 实现、GPU qualification、新训练、云资源或 official test。

## 十二、由本项目开启的“方向1”论文路线

`liu-test-exp/方案1/研究方案设计专用提示词.md` 中的方向1是 **DFormerv2-S 冻结输出上的后验校准**。后验校准不修改 RGB-D 融合骨干，而是在冻结 checkpoint 输出 logits 后，研究模型给出的概率是否可信、错误是否能被有效排序、风险阈值能否迁移。

### 12.1 为什么可以从当前 B0 开始

当前 epoch 420 B0 已绑定 checkpoint、RGB 输入、split、FP32 主 evaluator 和逐类指标，足以作为工具开发和初步方法探索的固定模型。旧提示词中“必须先完成三 seed baseline 才能启动校准开发”的前置条件已经过时；当前策略允许 single-seed B0 支持探索和初步消融。

但 single seed 仍不能估计随机方差。如果校准收益很小、接近训练波动，或者准备形成重要论文结论，应再对 B0 和方法增加成对重复或额外 seed。

### 12.2 建议的下一计划

1. **独立 calibration 集**：先冻结与 train-dev、val-dev 和 official test 职责分离的数据协议。当前 `val-dev` 已参与 checkpoint 选择，不能直接改名为独立 calibration 集。
2. **冻结 logits**：从同一 checkpoint、同一输入清单和 FP32 推理导出 15 类逐像素 logits，记录配置、哈希和 ignore mask。
3. **低自由度基线**：比较未校准 Softmax 与全局温度缩放，再根据类别覆盖决定是否加入类别条件温度、向量缩放和保序回归。
4. **分开回答三个问题**：用 ECE、NLL、Brier Score 判断概率值；用风险—覆盖曲线和 AURC 判断错误排序；用冻结阈值在独立数据上的风险与覆盖判断阈值迁移。
5. **再扩展退化条件**：正常条件成立后，再考虑低照、模糊、Depth 空洞、遮挡、模态缺失、错位和复合退化。

全局温度缩放如果保持 argmax，就应核验预测标签和 mIoU 不变。ECE 改善但 AURC 不变并不矛盾：前者说明概率数值更接近真实正确率，后者说明错误相对排序没有改善。

独立 calibration 集、数据职责隔离、同一份 float32 logits 和 official test 封存仍然是有效硬边界。A2/B2 是独立的 Depth 机制分支，不是后验校准方向的强制前置条件。

## 十三、当前结果能证明什么、不能证明什么

### 13.1 可以证明

- 作者当前公开 DFormerv2 主线和官方 DFormerv2-S pretrained 身份已核验；
- MUSeg RGB 输入、训练增强、数据划分、checkpoint 选择和主 evaluator 已显式冻结；
- single-seed Quick-B0 已完成 500 epoch 和 4 候选主评估；
- epoch 420 是冻结主 evaluator 下的最终 B0；
- 指标、checkpoint、split、protocol、提交和 official-test 状态可以通过哈希追溯；
- 训练与云生命周期已完成并安全停止；
- 颜色和推理几何会显著影响结果口径；
- B1 解决空有效像素的数值稳定性问题。

### 13.2 不能证明

- 不能证明 official-test 性能或真实部署性能；
- 不能提供三 seed 均值、方差或统计显著性；
- 不能把当前与历史基线的差值归因于 RGB、随机尺度、裁剪窗口或 B1 中任何单一因素；
- 不能证明 RGB 重训一定优于 BGR 重训；
- 不能把历史固定 resize 的最高数字追认为主协议；
- 不能证明 B2 有效，因为 B2 尚未实现；
- 不能把参考费率估算写成实际控制台花费；
- 不能把方向1的研究设计写成已经执行的论文结果。

## 十四、复现与交接信息

最终 B0：

- 文件：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/active-seed/checkpoint/selector-epoch-420.pth`；
- 大小：`321,011,270` bytes；
- SHA-256：`f246a3afc50334c81302b7bfebdadf7cf37d00326bf1c3aa54f6a151754e3a1c`。

主证据：

- 完整胜者评估：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/posteval/epoch-420-msflip-whole-original-grid-v1.json`；
- 结构化裁决：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/posteval/main-evaluation-adjudication.json`；
- 裁决 SHA-256：`20336eabfdefcf8661ed33c781a0104784950cde07d59e4d1c343c8579c27b05`；
- 本地身份复核：`cloud/DFormer-quick-b0-evidence/museg-dformerv2-s-rgb-quick-b0-v1/local-verification.json`；
- 详细主评估报告：`doc/reports/2026-08-31-museg-quick-b0-main-evaluation.md`；
- 历史训练经验：`doc/reports/2026-08-31-museg-stage05-seed1-training-lessons.md`；
- 当前实时入口：`doc/main/MUSeg-current-status.md`。

后续模块要与本 B0 做严格配对比较，必须从同一 pretrained 独立训练，并保持 `train-dev`/`val-dev`、seed、数据顺序、500 epoch、优化器、增强、checkpoint 规则和主 evaluator 不变。不能从 epoch 420 接着训练一个模块，再把它称为公平消融。

## 十五、组会建议讲述顺序

1. 先讲清楚目标：不是追一个孤立高分，而是建立可信、可复用的共同起点；
2. 再讲作者公开版本、官方权重和 RGB 输入为何需要核验；
3. 用训练裁剪与原图计分解释为什么评估口径必须提前冻结；
4. 展示 4 个候选结果，突出 epoch 480 与 epoch 420 排名反转；
5. 展示 15 类 IoU，说明总体分数下仍有 container 等薄弱类别；
6. 用历史 geometry 和颜色诊断说明“数字必须绑定协议”；
7. 说明 A1/B1、启动环境和自动关机等工程问题如何解决；
8. 最后用 B2 与方向1说明下一阶段，但明确尚未授权和尚未执行。

## 十六、验证说明

本报告只整理已经存在的代码、实验和审计证据，没有重新运行训练、GPU、长耗时评估、云端任务、完整测试套件或 official test。以上未运行项目不能写成通过；本次交付只进行内容交叉复核、索引 JSON 检查、Canvas 类型检查和文档差异检查。
