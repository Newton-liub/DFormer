# MUSeg Stage-05 seed 1 后评估与 Protocol Gate 中间收口

- 记录时间：2026-08-28 01:50 UTC
- 任务范围：已授权的 seed 1 五项 val-dev 后评估、结果身份核验和 Protocol Gate 中间证据收口
- 当前结论：五项后评估均完成且身份一致；Protocol Gate 仍未通过，validation geometry 选择、pretrained provenance 语义和 paired calibration 仍待闭合。

## 1. 执行边界

- 数据仅使用本地 `val-dev`，318 个唯一样本；与 `official-test.txt` 无交集。
- 五项评估均使用 batch 1、legacy BGR、`legacy-imagenet-positionwise-bgr` normalization 和 post-evaluation v2 evaluator。
- 所有结果的 metric grid 都是原始 Label grid，Label 未 resize；official test 未参与。
- 未启动训练、development seeds 2/3、A2/B2、模块筛选、正式三种子或 official test。

## 2. 结果

| checkpoint | geometry | status | mIoU | mAcc | mF1 | metric grid |
| --- | --- | --- | ---: | ---: | ---: | --- |
| best-val-miou.pth | original-full | completed | 52.98 | 65.67 | 67.96 | original-label-grid, 932×1082 |
| best-val-miou.pth | resize-480x640 | completed | 56.31 | 68.76 | 70.62 | original-label-grid, 932×1082 |
| best-val-miou.pth | sliding-480x640 | completed | 51.89 | 65.45 | 66.28 | original-label-grid, 932×1082 |
| epoch-500.pth | resize-480x640 | completed | 56.73 | 68.88 | 70.99 | original-label-grid, 932×1082 |
| epoch-500.pth | sliding-480x640 | completed | 52.08 | 65.60 | 66.21 | original-label-grid, 932×1082 |

结果文件位于仓库外证据目录：

- `cloud/DFormer-stage05-evidence/posteval/best-original-full-v2.json`
- `cloud/DFormer-stage05-evidence/posteval/best-resize-480x640-v2.json`
- `cloud/DFormer-stage05-evidence/posteval/best-sliding-480x640-v2.json`
- `cloud/DFormer-stage05-evidence/posteval/epoch-500-resize-480x640-v2.json`
- `cloud/DFormer-stage05-evidence/posteval/epoch-500-sliding-480x640-v2.json`

## 3. 固定 checkpoint 三臂颜色诊断

三臂均使用 best checkpoint、318 个 val-dev 样本、original-full 和原始 Label grid：

- legacy BGR + 原数组位置 mean/std：mIoU `52.98`、mAcc `65.67`、mF1 `67.96`；
- RGB + RGB mean/std：mIoU `33.85`、mAcc `46.33`、mF1 `46.80`；
- BGR + 反向 mean/std：mIoU `49.53`、mAcc `60.96`、mF1 `64.42`。

该结果证明现有 checkpoint 对输入通道与统计顺序强敏感，但不能公平比较从头训练的 BGR/RGB 谱系，不能替代 paired calibration。

## 4. 身份核验

- best checkpoint SHA-256：`b62ca049e6a647aca109c70e80823cec8e36ae1cc1df27e3bcf2b1d215b160bf`。
- epoch-500 checkpoint SHA-256：`0b88ab022db5188fd3439ea4e3af2098fe81e7c85757d1d91db33e831df2ff79`。
- val-dev split SHA-256：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 本地 val-dev bundle：318 个样本、RGB/Depth/Label 共 954 个文件；`official_test_included=false`。
- 原始归档 SHA-256：`4f6b079b707266ee358d2522fc6e4e034a5380d09ba8c65696df7aaa3e383c66`。
- 五份 JSON 均为 `status=completed`、`sample_count=318`、上述 split SHA，且 `official_test_included=false`。

## 5. 风险与待闭合项

1. `original-full` 成功完成，未触发本地 8 GB 显存限制。
2. `resize-480x640` 的数值最高，但仍只作为诊断；它存在输入长宽比改变，不能仅凭 mIoU 冻结为未来 geometry。
3. `original-full` 保持原始像素支持且避免长宽比扭曲；`sliding-480x640` 保持原始输出网格但与本次 full-frame 结果不同。未来 geometry 仍按预注册优先级和独立 calibration 决定，不改写历史曲线或 epoch-460 best 身份。
4. pretrained 文件身份已核验为 110,203,103 bytes、SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`；仓库证据仍未把该 SHA 绑定到上游发布资产及其训练通道语义，因此 provenance 仍为“待核验”。
5. 固定 checkpoint 的三臂颜色敏感性诊断以及从同一 pretrained、seed、预算和 evaluator 成对重训的 paired calibration 尚未执行；它们不能由本次单 checkpoint 五项结果替代。

## 6. 预检与实现补充

- `tests/` 全量 CPU 测试：118 passed。
- 后评估/切分/契约聚焦测试：41 passed。
- `python -m compileall -q tools utils local_configs tests`：通过。
- 当前 PyTorch 2.7 环境需要对可信旧 checkpoint 显式使用 `torch.load(..., weights_only=False)`；该兼容修复位于 `tools/evaluate_museg_checkpoint.py`，原有 `criterion=None` 和 `strict=True` 保持不变。
- 默认无参数 `pytest` 会额外收集历史 cloud 测试和 `utils/engine/dist_test.py`，因历史缺失路径/导入路径失败；本次不重复处理该非 Protocol Gate 阻塞项。
