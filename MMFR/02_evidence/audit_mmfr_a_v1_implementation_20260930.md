# A-v1 实现差异审核（2026-09-30）

独立 identity 为 `MMFR-A-v1-action-utility-v1`，基准 HEAD `a7c5cc17c5b495f2210efa3604b05794b5fa16c4`。本文件只说明本轮代码边界；完整验收见 审核报告（`report_mmfr_a_v1_implementation_gateb_20260930.md`），实际 checks/hash 见 Gate-B JSON（`mmfr_a_v1_gateb.json`）。

## 本轮实际代码差异

- 新增 `models/mmfr_av1.py`：固定 256→32→256、DWConv3×3、zero-init up；四 observed stats、260→16→1 标量 gate；off 直接返回原 features 对象。
- 修改 `models/builder.py`：新增可选 `av1`，只插入 encoder stage2 输出；A-v1 base no-grad，HAM 保留到 active branch 的梯度；新 `train` 覆盖只对 A-v1 生效，固定所有 C0 eval；A-v1 forward 只接受推理输入，训练标签走独立 phase loss。
- 新增 `utils/mmfr_av1_training.py`：新 phase seed 的合法 corruption 重新构建、只开放当前 branch、复用 optimizer grouping；同 input/RNG endpoints、detached margin labels、ambiguous 安全 BCE、CE + clean KL。没有 evaluator、正式训练循环或缓存。
- 新增 `local_configs/MUSeg/DFormerv2_S_MMFR_AV1.py`：deep-copy C0，新身份/绑定 source；旧 E1 disabled，正式 margin/lambda_clean 未选，评价 sources/selector/schedule 关闭。
- 新增 `tools/mmfr/av1_gateb.py`：synthetic mini-batch + 真实 source/backbone/HAM，八次 forward、两次更新；保存 checks、输入/RNG/NMF proof 和实现 SHA-256。

## 主代理直接复核的高风险边界

1. off 在 validation/proposal/gate 计算前直接 return；真实 C0 bitwise equal，proposal/gate 零调用。
2. 只第2级 feature 被替换，原 internal geometry/backbone/HAM 算法没有改动；其余 feature 原对象保持。
3. Base 全参数/buffers 不变、eval 固定；两阶段 optimizer 精确覆盖 current branch，frozen branch 不在 optimizer 且无 gradient。
4. off/full target 在 no-grad 中形成，utility/mask/target detached；all-ambiguous classification zero 不阻断 learned CE。
5. 三行为反复 restore 同一 pre-forward state，实际 NMF initial bases 相同；同 observed tensor/label 被复用，不在分支内 corruption resample。
6. `gate_value`/`depth_stats` 仅 observed feature/depth/support；clean metadata 只用于 loss mask，不进 gate。
7. source C0 SHA-256 前后不变；无新训练 checkpoint、正式训练、验证集读取、official test 或云端执行。

## 差异身份与限制

Gate-B JSON SHA-256 为 `bf87cc2fc46dd413131f7a6778d8df82388cf7fffa08852126d1f62178be01bf`；其中 `implementation_sha256` 绑定五个代码文件的本次版本。主代理直接查看模块、训练函数、builder diff 与完整 JSON，再作 PASS 验收。

原工作区已有 13 tracked 修改、4 untracked 根条目，其中基础配置/旧报告/论文库工具等不是本轮修改。无 commit/push；旧 E1 protocol/evaluator/corruption 和历史报告正文不改。本次 design 报告仅更新当前进展标注与两篇近邻/统一创新边界。

资格边界仍是 A-v1 protocol（`../01_research/mmfr_a_v1_action_utility_protocol.md`）：小输入两次 smoke update 不证明 full-resolution batch10、训练稳定性或性能；停止在 Gate-B，等待正式授权。
