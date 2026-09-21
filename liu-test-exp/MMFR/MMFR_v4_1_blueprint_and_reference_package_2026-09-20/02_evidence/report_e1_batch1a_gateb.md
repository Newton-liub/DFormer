# MMFR E1 Batch 1A C0/F-lite Gate-B Report

> **Check identity：** `MMFR-E1-Batch1A-C0-F-GateB`  
> **状态：** `PASS`  
> **日期：** 2026-09-21  
> **范围：** 最小实现资格；不是正式训练或效果评价。  
> **official test：** `false`

## 1. 最终裁决

- **C0：PASS**
- **F-lite：PASS**
- `failed_checks=[]`
- 当前只证明 C0/F-lite 实现能够遵守冻结协议并启动学习路径，不提供 mIoU、鲁棒性或方法收益结论。

Canonical evidence：

- 路径：`outputs/mmfr-e1-batch1a-gateb/e1-batch1a-gateb.json`；
- SHA-256：`5d5f0526e95ed81b6da8fbcd3cc395911f2266132a9148826b5ad261042e4e89`；
- 命令：`python tools/mmfr/e1_batch1a_gateb.py --latency-warmup 5 --latency-repeats 20`；
- duration：`26.896823199997016 s`；
- device：`NVIDIA GeForce RTX 5060 Laptop GPU`；
- batch size：1。

## 2. Source identity

### 2.1 Checkpoint

- 路径：`experiments/MMFR_A2_v3/checkpoints/selector-epoch-420.pth`；
- size：`321103318` bytes；
- expected/actual SHA-256：`2b72eb7f28e5c4b4f52bdc43e2ac5c7298169325a412bfbcbc7a022c29822597`；
- schema：`dformer-training-checkpoint-v2`；
- model keys：`812`；
- missing/unexpected keys：`[] / []`；
- completed/next epoch：`420 / 421`；
- global optimizer step：`53735`；
- embedded source commit：`d82d83482722776f5dc5059c80c34975e828e402`。

Qualification working commit：`e8abbe2d0b87ea5b2e29069037fcf4bc8d7e57eb`。

### 2.2 Source hashes

- A2 common v3 config：`3f309effc1a8e005844d8885bfd31cdb05c816e686632aa0d34f0f23bcffa595`；
- A2 DepthCorrupt v3 config：`b8ba2dd0c5ac4a3c445779bc44ce1d4e9a283fe309a8e190c8c84b5da2c02bef`；
- `models/encoders/DFormerv2.py`：`029ce7c5659c9e537165cdbfa5da94a2b003ced07b51e6ee3ee63b9ab10f2695`；
- `utils/dataloader/mmfr_training_v3.py`：`fff423f7d5208db315b7e521da462775ba6f52dc6d68e17cad07b97e6c54d717`；
- `utils/dataloader/multimodal_failure_v3.py`：`0a729bcb1b60a71120445894d71459895d2fd61a6bc30dfbccb3ac94108b110f`；
- `train-dev`：`a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`；
- `val-dev`：`1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。

### 2.3 Probe batch

- sample：`RGB/06-01-01-0035-230920140169-12-99.jpg`；
- corruption：`gaussian_noise + blur`；
- clean：`false`；
- curriculum progress：`0.8400131252050813`，位于冻结范围 `[0.84, 0.88]`；
- official test：未包含。

## 3. Implementation identity

Gate-B 记录的实现哈希：

- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_Common.py`：`19222c23eddcc0cb7f1ca31c8f646f868e0945cfa688d1ddd45f8ca9caf31985`；
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_C0.py`：`52baa9858f932b8e85b087ccf44e6b27d87c0b68ed51c9f273558c29aadf1911`；
- `local_configs/MUSeg/DFormerv2_S_MMFR_E1_Batch1A_FLite.py`：`b0b6d3a893d164a5ed149e0e5c178cb58edc5b8946e58b354eb5ab3e3e790845`；
- `models/feature_adapter.py`：`bb89f3b89f685aa9cc0152ed376fd1282bc5966ede4b336cfe7244216c751414`；
- `models/builder.py`：`d0a95db1b1a4010a685551a7bbd08c3827e046b112e9d1660ef0f7692cd6a89e`；
- `utils/init_func.py`：`f4b2771678b8a3efdbec832a0a776147a106579ba9d78164b851861ceac14e2f`；
- `utils/train.py`：`547e34643dd7b5318e990b5d6965b887fcdb86580aca7c3f8e3568b8f0e9d9ef`；
- `utils/training_checkpoint.py`：`4332bb1a8a8cf2b537034fd98699ef2df6d013dddee2abd9742269bb99809b61`；
- `tools/mmfr/e1_batch1a_gateb.py`：`8636fcb5e95a282b6abbcb121c659a3e2fe69501156ea50d90c2cc2ab47c797d`。

## 4. C0 identity

在任何训练 step 之前：

- 812 个 state key 与 source 完全一致；
- missing/extra/unequal keys 均为空；
- logits exact equal，最大绝对/相对差均为 0；
- prediction exact equal，最大绝对/相对差均为 0；
- total loss exact equal，最大绝对/相对差均为 0。

共同 forward 的有限 loss components：

- segmentation：`0.10337051749229431`；
- reliability：`0.021056054159998894`；
- total：`0.10547612607479095`。

因此 optimizer coverage 修复没有在执行 step 前改变 C0 forward。

## 5. Optimizer membership

### 5.1 共同规则

组顺序精确为：

1. `base_decay`；
2. `base_no_decay`；
3. `new_decay`；
4. `new_no_decay`。

所有 `requires_grad=True` 参数 membership 均精确为 1；invalid membership 为空。

### 5.2 C0

- trainable tensors/elements：`720 / 26677579`；
- `base_decay`：305 tensors，26542362 elements，LR `1e-5`，WD `0.01`，name digest `8ec9e931bdd6041df0e0659f40d1dabc005f2a5c17006c4191a0977d5adf6e16`；
- `base_no_decay`：415 tensors，135217 elements，LR `1e-5`，WD `0`，name digest `c053b988603a0e450dced06f40285cadd1e6daf416e55dcbc0c699e9be19c601`；
- `new_decay`：0 tensors，0 elements，LR `3e-5`，WD `0.01`；
- `new_no_decay`：0 tensors，0 elements，LR `3e-5`，WD `0`。

### 5.3 F-lite

- trainable tensors/elements：`732 / 26850731`；
- base groups 与 C0 完全相同；
- `new_decay`：6 tensors，172032 elements，LR `3e-5`，WD `0.01`，name digest `e3a29a2089dfbe10b1e4d916d5e49fe5619e2c596ccd61834c4efd420654bbc1`；
- `new_no_decay`：6 tensors，1120 elements，LR `3e-5`，WD `0`，name digest `414a0c3f486ec83f13271561cb2aef2db71b8bd77536089f305be21992493b36`。

6 个 adapter Conv weights 全部属于 `new_decay`，6 个 biases 全部属于 `new_no_decay`。

### 5.4 Geo 与 reliability

- 精确 29 个 `backbone.layers.*.blocks.*.Geo.weight` 全部进入 `base_decay`；
- 14 个审计到的 SyncBN 参数全部进入 `base_no_decay`；
- reliability head 的 3 个 weight 全部进入 `base_decay`；
- reliability head 的 3 个 bias 全部进入 `base_no_decay`。

## 6. F initial no-op

F-lite 新增 trainable parameters：`173152`，与配置一致。

初始状态：

- stage 1 residual exact zero，feature exact equal；
- stage 2 residual exact zero，feature exact equal；
- stage 3 residual exact zero，feature exact equal；
- 四级 decoder input 全部 exact equal；
- final logits exact equal；
- prediction exact equal；
- loss exact equal；
- 所有最大绝对/相对差均为 0。

F-lite 共享 812 个 source state key 也全部 exact equal；只允许新增 12 个 adapter state key。

## 7. Gradient 与 update

### 7.1 C0

- step loss：`0.7711184620857239`，finite；
- GradScaler：`1024 → 1024`；
- optimizer step：applied；
- shared probe parameter `backbone.patch_embed.proj.0.weight` gradient norm：`21.206388473510742`；
- 6/6 reliability parameters gradient finite 且非零；
- 29/29 Geo gradient tensor 均存在且 finite；
- 19/29 Geo 在该 batch 上 gradient nonzero；
- 29/29 Geo 在 AdamW step 后参数均发生合法变化。

Geo 最大变化量范围包含纯 weight-decay 引起的小变化与有数据梯度的约 `1e-5` 变化；因此零数据梯度的个别 Geo 仍可因 AdamW decay 合法更新。

### 7.2 F step 1

- step loss：`0.7711184620857239`，finite；
- optimizer step：applied；
- 6/6 up weight/bias gradient finite 且 nonzero；
- up gradient norms：
  - stage 1 bias/weight：`5.007231266063172e-06 / 0.052176494151353836`；
  - stage 2 bias/weight：`1.2589030120579991e-05 / 0.2756393253803253`；
  - stage 3 bias/weight：`2.3309899916057475e-05 / 0.661462128162384`；
- 6/6 up parameters changed；最大绝对变化约 `2.97e-05` 至 `3.00e-05`；
- 6 个 down gradient 全部为 0，符合 zero-init up 的冻结预期。

### 7.3 F step 2

- step loss：`0.3015517592430115`，finite；
- optimizer step：applied；
- 6/6 down weight/bias gradient finite 且 nonzero；
- down gradient norms：
  - stage 1 bias/weight：`4.298979547456838e-06 / 8.35607061162591e-05`；
  - stage 2 bias/weight：`1.6597648937022313e-05 / 0.0005126711330376565`；
  - stage 3 bias/weight：`1.601040094101336e-05 / 0.0006951669929549098`；
- 6/6 down parameters changed；最大绝对变化约 `2.22e-05` 至 `2.23e-05`。

F 的 29/29 Geo gradient tensor 均存在且 finite，19/29 nonzero，29/29 参数发生合法更新；6/6 reliability parameters gradient finite 且 nonzero。

## 8. 参数与成本

### 8.1 参数量

- C0 total/trainable：`26677579 / 26677579`；
- F-lite total/trainable：`26850731 / 26850731`；
- F-lite 新增：`173152`。

### 8.2 Peak CUDA memory

- C0 allocated：`2262162432` bytes；
- C0 reserved：`2445279232` bytes；
- F-lite allocated：`2486257664` bytes；
- F-lite reserved：`2604662784` bytes；
- allocated delta：`224095232` bytes = `213.7138671875 MiB`；
- reserved delta：`159383552` bytes = `152 MiB`。

冻结 gate 为 allocated delta `<= 512 MiB`，因此 PASS。

### 8.3 Latency

每项 warmup 5、repeats 20、batch size 1：

- C0 forward-loss median：`165.0675048828125 ms`；
- F-lite forward-loss median：`165.9855499267578 ms`；
- delta：`+0.9180450439453125 ms / +0.5561633978759639%`；
- C0 inference median：`131.17436981201172 ms`；
- F-lite inference median：`132.20088958740234 ms`；
- delta：`+1.026519775390625 ms / +0.7825612403259408%`。

冻结 inference gate 为 `<= +5%`，因此 PASS。

## 9. 实际验证与边界

实际执行：

1. 定点 Python 编译；
2. 强化版单次 Gate-B；
3. source/config/split/checkpoint identity；
4. C0/F optimizer membership；
5. C0 exact identity；
6. F exact no-op；
7. C0 一步、F 两步 AMP optimizer startup；
8. Geo/reliability/shared gradient 与 update；
9. batch-1 CUDA memory 与 latency probe。

按测试预算未运行：

- 完整测试套件或全仓扫描；
- 20 epoch / 2560 update 正式训练；
- Quick-Val 或 318 样本评价；
- Main-Val；
- 云端任务；
- official test。

## 10. 停止点

当前停止于 `Batch 1A C0/F Gate-B PASS`。下一步只能在上级审计通过且用户单独授权后，启动冻结的 20-epoch / 2560-update 正式训练。