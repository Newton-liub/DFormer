# MMFR A-v1 条件性本地 val 转移交付收据

- 日期：2026-10-01；生成于用户要求的无GPU资料收口阶段。
- 目的：交付上级审核报告与后续**另获授权**时的本地val准备资料。
- 结果：转移ZIP已生成、逐成员字节/SHA和CRC核验通过；本次模型执行0，不加载checkpoint，不运行训练或评价。

## 1. 可转交材料

- 正式报告：[`report_mmfr_a_v1_formal_quickval_upper_review_20261001.md`](report_mmfr_a_v1_formal_quickval_upper_review_20261001.md)。训练完成，但唯一Quick-Val按冻结规则为`stop`，learned三hard−matched off为−0.0018730026999946858pp，+0.50pp门槛FAIL。
- 操作交接：[`handoff_mmfr_a_v1_local_val_conditional_20261001.md`](handoff_mmfr_a_v1_local_val_conditional_20261001.md)。A-v1现有评价入口只支持四条件单视图；Main-Val适配未实现，新val未授权。
- ZIP绝对路径（仓库外）：`/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip`。
- 配套校验文件：`/root/rivermind-data/cloud/MMFR_AV1_local_val_conditional_20261001.zip.sha256`。
- 整包大小：**429993998 bytes**（429.994 MB，约410.074 MiB）。
- 整包SHA-256：**`1975bde09fcde5924dc3c947f6762e59db51b1e0a2ae1ed25ca4410a07fadcd7`**。
- 归档成员：**996个普通文件**，无重复或不安全路径；其中973个已提交源码/配套文件、19个review/运行证据文件、1个既有复现JSON基线、1个README和2个checkpoint。

**大白话：** 可以把报告交给上级，并下载保存此包。包完整、权重身份正确只证明资料能迁移，不能表示本地评价已具备执行条件或获得授权。

## 2. 包内布局与精确身份

| 目录/文件 | 实际内容与核验 |
| --- | --- |
| `README.txt` | 阅读顺序、范围、源码基线、关键文件SHA与执行限制 |
| `review/` | 本轮报告、条件性交接、原A-v1 protocol，按打包时字节保存 |
| `source/` | `4f84b469c4b04de657eaf2c455bd44e991b7f869`提交的限定源码快照：models/utils/tools/local_configs/mmseg的Python文件，README/LICENSE/监控依赖清单，一个v3 protocol模板与train-dev/val-dev清单 |
| `checkpoints/A-v1/update-2560.pth` | 107210042 bytes；SHA-256 `87ec54d10192d3aedc1d6edb864b7b60b94ce66220a748a12f1ca50ca605d336` |
| `checkpoints/C0/update-2560.pth` | 321150608 bytes；SHA-256 `ca618b23d18eabb201a0d11d18da383ac99576d0feae5864e3233bda527d9a1a` |
| `evidence/amended-run/` | identity、批准修订兼容性、training-result、两阶段result及原评价初始化stopped证据 |
| `evidence/quickval/` | summary及四条件metrics，包含保存的逐样本评分记录；4×318、结论stop |
| `evidence/original-run/` | 原运行preflight-result、stopped与Proposal停止计数 |
| `evidence/diagnostic/`、`evidence/precision-controls/` | 原单次重放失败定位和三组精度现场对照的小型证据 |
| `evidence/reproducibility_current.completed-result-baseline.json` | 直接取源码基线提交的既有完成结果复现JSON；是导出前快照，不含本收据或后来文档状态 |

冻结val-dev：318条，SHA-256 `1d0719d8f64f016d48995c25ab66d4004d76b7155d9efeef7cbb7454c0dd0e83`；train-dev：1277条，SHA-256 `a6b15b63f6d5193e3928ea24ada25be403a48e68d1c1f9372cdbbc3fe5cd8470`。源码及split从Git显式allowlist取字节，未打开official-test split或dataset。

两份checkpoint原文件边读边复制并核对SHA，未重新序列化、裁剪optimizer或改写metadata。`source/`不是Git仓库；运行入口需要真实Git身份时，应使用GitHub上对应提交的完整checkout，不能用`git init`伪造。旧文件中授权字段按历史上下文解释。

## 3. 已运行的最小检查

1. 报告所引直接training-result、Quick-Val summary与既有最终身份/核验记录复核；筛选结果不追改。
2. 报告索引、review profile、既有复现JSON以UTF-8/BOM兼容读取并成功解析；打包前MMFR四个canonical目录顶层26份Markdown路径链接通过，最终新增收据后27份MMFR文档及2份实时入口共29份链接检查通过（仅本地路径存在性，不冒充全部fragment或跨平台渲染验证）。报告登记、profile所有attachment源、Canvas版本保留与ZIP收据SHA/大小一致性亦通过。
3. ZIP使用显式源码和证据allowlist；所有996个成员逐字节SHA与输入对应，完整读到末尾核对CRC，成员集合/唯一性/安全路径通过。
4. 两份checkpoint及两个dev split匹配冻结SHA；summary为四条件、每条件318样本且stop，final身份匹配。
5. Git提交前文档差异与空白检查通过；源材料提交 **`afbeec29a22e9f46ac958bb9ec1bf727f9fcd8ed`** 已普通push至 `origin/perf/mmfr-a2-v3-pipeline-opt1`。2026-10-01 15:44:29UTC直接核验本地/远端同SHA且工作区clean，未强推。本条是该已完成源材料交付的回执，后续包含本回执的文档提交不追逐自身SHA。

本次未运行项目完整测试、模型导入、GPU、训练、重复Quick-Val、Main-Val、test或本地目标机器验证，因为交付范围只有文档与打包。跨硬件逐位等价、完整依赖锁定、本地数据可用性和Main-Val adapter仍未验证。

## 4. 排除项、资源与恢复点

- 包不含dataset图片/标签、official-test清单/数据/cache、原论文或外部clone、大型console/步骤日志、父1280/transition/诊断checkpoint、Git历史或旧生成审核包。它不是全训练恢复归档；原始证据仍留在原位置。
- checkpoint、ZIP和逐样本metrics均不进入Git/MMFR，Git只交付报告、交接、收据和已有控制/实时入口更新。
- `MMFR/99_review_packet_current/`仍是旧生成快照；缺少pwsh，本次未安装或手工编辑生成产物。现有profile已指向新报告；将来具备PowerShell环境后，先核验canonical链接再用原`MMFR/98_tools/rebuild_review_packet.ps1`重建。当前转交本份报告即可完成汇报，不能把旧包声称为最新。
- 云实例最后直接核验于09:30:44 UTC仍开启、实验进程退出/GPU空闲；本次只按用户要求做无GPU资料工作，未重新核验资源实时状态、未关闭/销毁实例。
- 下一恢复点：先上级阅读报告；只有明确批准新的val范围及必要适配/资格预算后，才接本地执行任务。当前继续保持实验停止与official test `sealed_unread`。
