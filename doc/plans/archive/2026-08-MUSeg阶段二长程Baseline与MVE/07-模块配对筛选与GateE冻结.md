# 07：模块配对筛选与 Gate E 冻结

> **后继候选计划：** `doc/plans/deferred/2026-09-MUSeg-unexecuted/MUSeg-A2-B2深度有效性/00-总方向规划.md`（该后继计划现已延期，未执行）。

> **文档角色：** 活动阶段计划，不承担实时状态或运行授权。
> **形成或核验时点：** 2026-08-27。
> **实时入口：** `doc/main/MUSeg-current-status.md`；研究选择见 `doc/main/MUSeg-open-decisions.md`。
> **后继关系：** 接收通过 Protocol Gate 的通用模块，以及通过新 Stage-06 全部证据门禁的 B2；Gate E 后进入新 Stage-08。

## 1. 统一 screening 原则

每个候选都在独立 screening protocol 内与实际运行的 `module-screening-B0` 成对比较：

- 相同 pretrained、seed、train-dev/val-dev、样本顺序、总 epoch、batch、optimizer、LR schedule、增强、checkpoint 规则、channel order、normalization 和 evaluator；
- B0 与 variant 在同一最终代码基线上只改变预登记模块开关；
- 改变总 epoch、子集、初始化、数据顺序或 evaluator 即产生新 protocol；
- 不能用现有 `development-reference-B0` 的 500-epoch `52.84` 对比短协议候选；
- B2 不能绕过 Stage-06 的 A2、规格、金标准、zero-train 和 short gate。

模块改变显存或数据流后，先完成单元/几何测试、真实 forward/backward 和最小 GPU qualification；旧 batch 只作参考。

## 2. 运行矩阵

最小矩阵为一个预注册 seed 的 B0/variant 配对。按 seed 组织两臂，保持环境和数据顺序一致；运行顺序预登记，不能看 B0 结果后调整 variant。

只在以下情形增加可选第二 seed：

- paired Δ 接近 protocol 的噪声容差；
- per-class 方向冲突；
- 曲线或 checkpoint 选择对结果敏感。

第二 seed 必须同时补 B0 与 variant，不替代正式三 seed，也不能挑最好 seed。

## 3. 指标和聚合

每个 seed 报告：

- val-dev mIoU/mAcc/mF1 与 `variant - B0` paired Δ；
- per-class Δ、曲线、best/final 身份；
- 参数量、训练/推理吞吐、峰值显存、AMP 与失败/恢复史；
- split/pretrained/checkpoint SHA、commit、input contract 和 metric geometry。

筛选的聚合单位是 seed 内成对差。单 seed 只用于淘汰/候选排序；两 seed 时报告两个 paired Δ 和方向一致性，不发布正式 mean±std。门槛使用未四舍五入值，浮点复算绝对容差 `1e-6`。

## 4. 候选结果

每个候选在 protocol 中预先冻结最小收益、干净性能容差、per-class/资源安全条件，并输出：

- `screening-pass`：收益和安全条件均满足；
- `screening-stop`：收益不足、方向冲突、成本不可接受或证据链失败；
- `screening-inconclusive`：接近阈值或缺少预注册证据，可按第 2 节补一次 paired second seed。

B2 使用对应名称 `B2-screening-pass/stop/inconclusive`，并额外要求 Stage-06 证据链完整。不得通过事后改门槛、加 seed 或读取 official test 追求 pass。

## 5. 无候选分支

Stage-07 不强制至少一个模块进入正式阶段。所有候选 stop，或没有成熟候选时，形成可解释的 `B0-only` 结论：列出已筛选候选、失败原因、证据边界和后续研究建议，然后仍可把 formal B0 提交 Gate E。

因此新 Stage-08 的最小合法矩阵只有 formal B0；只有 Gate E 保留 variant 时才形成模块改进对照。

## 6. Gate E 冻结内容

向用户提交：

- Protocol Gate 的颜色/几何/evaluator 身份；
- 每个 B0/variant 的 paired screening 证据和资源成本；
- B2 若存在，其 A2→规格→金标准→zero-train→short→screening 链；
- 推荐 `B0-only` 或最终模块/消融组合；
- formal B0 与每个最终 variant 的完整 official-train × 三 seed 运行矩阵；
- 统一 pretrained、seed allowlist、预算、batch、LR、增强、checkpoint、evaluator、失败恢复和 qualification 规则；
- official test 继续封存以及 Gate F 条件。

用户确认后产生不可变 Gate E 记录。之后任何架构、组合、训练或 evaluator 变化都必须退回 development、升级 protocol 并重新 Gate E；旧 screening checkpoint 不进入正式统计。

## 7. 完成标准

- 所有候选与同协议 B0 实际成对运行，或形成有证据的“当前无候选”结论；
- Gate E 明确输出 `B0-only` 或 `B0 + 最终组合`；
- 正式三 seed 矩阵、checkpoint/evaluator 和资源边界已由用户冻结；
- official test 在筛选和选择过程中保持未读；
- 训练、预测、checkpoint、SwanLab 大产物和凭据不进入 Git。
