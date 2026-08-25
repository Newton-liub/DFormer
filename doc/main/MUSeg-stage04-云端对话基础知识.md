# MUSeg Stage-04 云端对话基础知识

> 用途：提供给项目空间中的云端执行对话，保存跨对话长期有效、可减少重复扫描与重复核验的信息。
>
> 当前阶段的精确恢复点、历史失败和门禁状态仍以 `doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md` 的最新“当前云端交接状态”章节为准。
>
> 本文件不得保存 API Key、密码、登录 cookie、完整环境变量或其他凭据。

## 1. 新对话最小阅读集

正常继续阶段 04 时，只需先读：

1. 本文件；
2. `doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md` 的当前交接章节；
3. 实际将要运行的工具脚本及其 `--help`/参数解析。

只有发生门禁、协议或跨阶段争议时，才补读：

- `doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md`；
- 对应的 01–03 计划或测试。

无需默认重读 01–03 全文、研究报告、Canvas、全部测试、历史终端日志或整个仓库。

## 2. 研究语义和禁止事项

- 当前 `phase=qualification`，目标是证明工程链路可用，不是 baseline，不产生论文性能结论。
- qualification 训练只消费冻结的 `train-dev.txt` 和 `val-dev.txt`。
- B1 真实模型回归只消费冻结的官方 `train.txt`，用于选择一张普通图和一张全背景图。
- 官方 test 始终为 `sealed_unread`；不得读取、抽样、评估，也不得用于 batch、epoch、checkpoint 或超参数选择。
- batch 4/8/12/16 probe 是独立 GPU 门禁。没有用户明确授权时不得启动。
- probe 完成后必须停在人工门禁 C，等用户确认 batch；不得自动进入 3-epoch qualification。
- 用户已在门禁 C 选择 batch 10；最终 protocol 必须显式绑定 batch 10，并在该 protocol 身份下重跑 B1/full preflight 后才能启动短训。
- qualification、连续/恢复演练通过后必须停在人工门禁 D；门禁 D 前不得进入阶段 05。
- 运行证据、日志、协议物化产物、checkpoint、数据和权重保存在云端 qualification 根目录，不提交 Git。

## 3. 冻结数据身份

- `train-dev.txt`：1277 samples / 762 groups，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`。
- `val-dev.txt`：318 samples / 196 groups，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`。
- 官方 `train.txt`：1595 samples / 958 groups，SHA-256 `6ff78af2621e32bf0320aea606674a81c5bae21889ad3a3ff0109a9d1d398123`。
- 冻结 `official-test.txt`：SHA-256 `12d9834215fcbfe696ad88321539c224850ff6fb66a01f48a02b1df478f48a4b`。该文件按 CRLF 原始字节冻结，Git 已通过 `-text` 属性保持 Windows/Linux 字节一致。
- 冻结 manifest：SHA-256 `42233412f432e387cfcffc763724461e2dbc111969a595c714ac12add7bf7b01`。
- 冻结 audit report：SHA-256 `53ac30aba0230919b994202f37b3571a7b416f9129f27eabf003415721e38055`。
- 本地与云端数据根中的 `train.txt`、`test.txt` 已按冻结清单纠正并完整校验；旧文件以 `*.before-frozen-fix-*` 保留。不要再次“修复”或重建划分。

## 4. 云端固定信息

- 实例：`cpod-1tyvjsiu6ahe`。
- 当前 SSH：`ssh -p 24569 root@cpod-1tyvjsiu6ahe.podtcp.compshare.cn`。实例重启后端口可能变化，以用户控制台为准。
- 官方仓库：`/root/rivermind-data/DFormer`。
- 数据集：`/root/rivermind-data/dataset/MUSeg_DFormer`。
- qualification 根：`/root/cloud-ssd/museg-stage04-qualification`。
- Python：`/usr/local/miniconda3/envs/py310/bin/python`，Python 3.10.16。
- 已核验环境：NVIDIA GeForce RTX 4090，24,564 MiB；驱动 610.57.04；PyTorch 2.1.2+cu118；CUDA 11.8；cuDNN 8.7.0。
- SwanLab：0.9.7。持久凭据保存在 Git 仓库外的 `/root/.config/dformer/swanlab.env`，文件权限为 `600`；`/root/.bashrc` 自动加载该文件。非交互命令若未继承交互式 Bash 环境，应先执行 `source /root/.config/dformer/swanlab.env`。基础知识和 Git 中只记录路径与加载方式，不记录 Key 内容。
- 预训练权重实际协议路径：`/root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth`，大小 110,203,103 bytes，SHA-256 `19116988fc86dc9f3e879282237941e11b9b1b5c480edb51e92807311dbc11a6`。

## 5. Git、协议和证据身份规则

- 任何 GPU qualification 开始前都要检查完整 `git rev-parse HEAD` 和 `git status --short`；必须是干净工作区。
- 协议路径：`/root/cloud-ssd/museg-stage04-qualification/protocols/museg-qualification-v1.json`。
- 当前已通过的 B1 与完整 preflight 证据绑定代码提交 `4f84ee33c93c3c8be83cb2ad029879c26a5346e9`，协议 SHA-256 `cb741c26897ac32f5a76204180932b03a80ff5ef9fd29ddd218a5ddfb64387e7`。
- 文档提交也会改变 Git commit。同步本交接文档后，不得把旧协议冒充为新提交的协议；继续 exact-commit qualification 前，必须归档旧协议、为新干净提交重新物化协议，并在 batch probe 前重跑 B1 和完整 preflight。旧报告保留为已通过历史与回归对照，不得改写其中的身份。
- 不要修改已有运行证据来“更新 commit”；旧证据按原提交和原 SHA 保留。
- 恢复父身份使用逻辑 `parent_run_id`；恢复子运行必须使用唯一新 `run_id`、不同输出目录，并校验 resume checkpoint SHA-256。SwanLab 后端 ID/URL 仅作为额外远端证据。
- checkpoint 默认比较要求完整 protocol 一致；恢复等价比较可显式忽略唯一合法差异 `protocol.run_id`，其余 protocol 字段、epoch/global step、LR、best、模型、optimizer、AMP scaler 和 RNG 哈希仍必须一致。

## 6. 已通过、无需无条件重复的核验

以下结论已有结构化证据。除非代码、协议、数据、权重或环境身份变化，否则新对话先读取报告，不要重复全仓库扫描：

- frozen split、四模态样本、预训练权重、输出根、Git 和 phase role 的静态/完整 preflight 检查。
- RTX 4090、驱动、PyTorch、CUDA、cuDNN 环境核验。
- B1 四种真实模型场景：主头 mixed、主头全背景、主+辅助 mixed、主+辅助全背景。
- mixed loss 有限；全背景 loss 精确为零、梯度精确为零；所有预期梯度存在且有限；official test 未包含。
- SwanLab 0.9.7 非交互式 online smoke 已成功。
- 本地测试、compile、Shell 语法和 diff whitespace 在各修复提交后均通过。历史文档含不同提交下的 100/101/102 等计数；最新总数未作为最终交接事实保留，不要把某个历史计数冒充当前计数。只有改代码或需要新提交验收时才重跑并记录新结果。

已通过报告：

- B1：`/root/cloud-ssd/museg-stage04-qualification/b1-real-model.json`，SHA-256 `e6dafd8656a783b9569df773381de43ba652e970b7f2b4ee0e971adc455510e5`，`pass=true`。
- 完整 preflight：`/root/cloud-ssd/museg-stage04-qualification/preflight-full.json`，SHA-256 `c127a792d85b2f315a53b5494b80b68293a435887abeb551de6c973a570e7b4d`，`pass=true`、0 errors、0 warnings。

## 7. 已知失败尝试及已完成修复

这些失败已有根因，不要重新从 CUDA、数据损坏或模型随机性开始排查：

1. Linux 检出的 frozen `official-test.txt` 曾因 LF/CRLF 规范化导致哈希不符。提交 `8c9f15822a56be4dd4a55136c3c457facf34f7d2` 已固定原始 CRLF 字节。
2. 首次 B1 以绝对路径执行时报 `ModuleNotFoundError: models`。提交 `22e217303367ff51e2ece8319822e61f4a668967` 已把 repo root 加入模块搜索路径并增加回归测试。
3. 第二次 B1 的辅助头使用 `x[0][aux_index]`，错误索引 batch 维并报 `IndexError`。提交 `4f84ee33c93c3c8be83cb2ad029879c26a5346e9` 已改为按特征级选择 `x[aux_index]`，最终四种 B1 case 通过。
4. 预训练加载会输出既有的 non-strict key mismatch warning。它没有导致上述 B1 失败，B1 JSON 本身没有 warning 字段；后续报告应单独注明，不得静默忽略，也不要把它误判为 CUDA 故障。
5. SwanLab online 门禁要求同一进程环境中存在 `SWANLAB_API_KEY`。当前云端已按用户授权将凭据持久化到 Git 仓库外的 `/root/.config/dformer/swanlab.env`，并由 `/root/.bashrc` 自动加载；非交互执行入口应显式 `source` 该文件。不得把 Key 内容写入文档、日志或 Git。
6. 曾误尝试不存在的 `tools/preflight_museg.py`；正确入口是 `tools/preflight_train.py --protocol-manifest ... --report ...`。旧失败没有覆盖结构化报告。

## 8. 当前后续边界

- 已完成：代码/静态验收、云端环境、真实 B1、完整 preflight、SwanLab online smoke。
- 尚未执行：batch 4/8/12/16 probe、门禁 C、3-epoch qualification、连续/恢复演练、门禁 D。
- 下一项 GPU 工作是 `tools/probe_museg_4090.sh` 的四档各 60 optimizer steps、10-step warmup 结构化 probe。
- **当前没有 batch probe 授权。** 新对话必须先向用户明确说明 workload、输出根和停止条件，并取得单独确认。
- probe 遇到 OOM 时停止更大 batch；遇到非 OOM 错误时停止全部；只有完整 60 步、显存安全且证据一致的档位可参与推荐。
- probe 完成后只报告四档吞吐、step time、显存余量、异常、证据路径/SHA 和推荐 batch，然后停在门禁 C。

## 9. 节省 token 的执行与报告规则

- 先读取 JSON 报告的摘要字段和 SHA，不粘贴整份 JSON、完整日志或所有 telemetry。
- 只报告本轮新增证据；已通过事项引用本文件和报告路径即可。
- 不重复解释 01–03 的实现历史，除非实际门禁检查失败。
- 定位问题先读失败 run 的 `training_result.json`、`probe-result.json`、末尾日志和 Git/protocol 身份，再决定是否扩展搜索。
- 未改代码时不跑全量 pytest/compile，不创建文档提交，不改变 protocol commit。
- 发现代码缺陷时立即停止 GPU 工作；保留失败证据，回到代码修复、测试、独立提交、云端同步和协议重新物化流程。
- 任何结论都明确区分：代码提交、协议 SHA、运行报告 SHA、SwanLab 远端证据。不要用实验 URL 替代本地结构化证据。