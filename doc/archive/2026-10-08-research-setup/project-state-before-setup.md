# 新研究配置前的项目状态摘要归档

归档时间：2026-10-08。以下保留上一阶段仍有追溯价值的事实；最新事实以 `doc/state/current.md` 为准。详细材料仍见原 `doc/reports/` 和仓库外索引，不复制论文资产或恢复旧研究代码。

## 仓库重建与既有资产

- 旧项目完整封存于 `D:\0Project\DFormer-archive-20261007\`，归档前 HEAD `8c274c59775427544ab209957c26974f4d40d083`，分支 `perf/mmfr-a2-v3-pipeline-opt1`；1463 个跟踪文件、13 个已修改文件、3 个未跟踪条目，归档后复核一致。完整 `.git`、历史结果和 checkpoints 留存。
- 现用工作仓库保持作者最新基线 `e3273009b759b578945483828ff315d560be94c9`，研究分支 `research/dformerpp-clean-start`，本轮开始前 HEAD `fe6cea2e7efdaf8f87299bed41baa86f11e316b4`。origin 为个人 fork，upstream push 禁用，未合并旧 MMFR 分支。
- 重建阶段的提交：`64bc2bd`（文档 / Skills）、`f26612d` / `4425d37`（状态 / 验证）、`c85cc11`（五项裁决）、`7d1942f`（权重放置与措辞修正）；后续提交见 Git。先前授权仅对已确认提交的显式推送有效，不等于无限推送授权。
- 本机新建 `dformer` 环境，旧 `df2` 冻结；RTX 5060 需要 cu128，因此本地使用 torch 2.7.0+cu128、mmcv lite 1.7.2，云端 4090 优先作者 torch 2.1.2+cu118 / mmcv 2.1.0。
- 2026-10-07 曾以 NYUv2 DFormerv2-Small config + 官方 pretrained 做一次最小前向，输出 `(1,40,480,640)`、finite、峰值 366 MiB，未训练 / 评价。第一阶段可运行 backbone 当时已裁决为 DFormerv2-Small；DFormer++ 官方 pretrained 未发布，保留后续升级候选。
- `checkpoints/pretrained/DFormerv2_Small_pretrained.pth` 由外部权重库复制，110,203,103 bytes、sha256 一致；官方 loader 的 `extra_norms.*` 为标准 LayerNorm 初始化（weight=1，bias=0），不会逐轮重新审计 missing / unexpected keys。标准 LayerNorm 参数初始化不能表述为对输入的数学恒等映射。
- 本机仅有 MUSeg 原始数据和 `MUSeg_DFormer`：后者 RGB / Label / Depth / Depth16 各 3171 文件，train=1595、test=1576。NYUv2 和 SUNRGBD 尚未存在；不恢复旧 MMFR / LER / natural_missing 实现及其旧训练授权。

## 外部论文索引与工具

- 索引唯一真源 `D:\0Project\origin\_index\`；论文 canonical 库 `origin\论文\` 共 37 篇，Pending 9 项，Duplicate Review 1 项，后两类不得当已核验论文使用。
- 外部 clone 共 12 个，`origin\DFormer`（`814799b`）已标 deprecated，当前工作仓库才是唯一有效工程。旧 PR / RE / AI 人工编号依据在归档 `MMFR/03_reference/`。
- 迁入的 7 个索引 / 工具文件逐一 sha256 比对一致。Streamlit UI 使用已有普通 Python 环境，不向 GPU 环境安装 UI 依赖。
- UI V1：共享 store 提供锁、指纹、原子保存、bak1–bak3；扫描不清空人工值，不回灌历史编号，拒绝未知更高 schema。V1.1 的目录自动重关联、逻辑 / 物理合并、AI 建议等仍未实现。
- 书目抽取 V1.0.1：补作者 / 年份 / 出版物 / 摘要及来源，schema 2→3；最初作者31/37、摘要34/37、年份11/37、出版物3/37、DOI7/37。完整事实见 `doc/reports/2026-10-07-bibliographic-extraction-v1.0.1.md`。
- 外部候选 V1.0.2：36/37 篇有官方证据，37 条候选固化在 `_index/external_bibliography_candidates.json`（format v2）：accept25 / review10 / keep_current1 / not_found1。`LIB000024` 无可靠来源；`LIB000015` 正式年份2026、CVPR2026；`LIB000025/30` 不据 arXiv 首发年份写出版年。人工 override 优先 auto，schema升4；候选确认先入草稿，再安全落盘。
- V1.0.2.1：批量只补空、安全 high-confidence、非 arXiv-only 的 accept 候选，原候选集合为23篇 / 59字段，不覆盖已有值。
- 最后一轮收尾：三个固定 AI 阅读导出 `PAPER_LIBRARY_FOR_AI.md` / `_CORE.md` / `_SUPPLEMENTED.md`，只读取已保存最终有效值，输出完整作者 / 摘要 / 人工补充，不改 JSON / revision / 备份；零篇也覆盖写明0。草稿或磁盘指纹冲突时拒绝导出。导出目录偏好只写 `%LOCALAPPDATA%\PaperLibrary\ui-settings.json`。
- 备份收口至 `_index/backups/`，受控幂等迁移遇同槽位不同内容即停止；真实三份备份按原槽位迁入且字节一致。隔离副本曾完成 store54 / UI32 / CLI共享备份9 项验收，未生成真实 exports，等待用户首次导出。
- 2026-10-08 UI V1.0.3：MUSeg 正文改动经内存整篇 diff 后正式重扫，仅 LIB000001 的 abstract / 来源 / 哈希字段变化；revision6→7，human / LIB / alternate / duplicate / schema 未变。修正顶部统计裁切、增加缺失书目信息筛选与人工摘要入口，隔离副本36项验收。
- 工程准备开始前真实索引为 schema4、revision7，32篇含人工 overrides；LIB000001 有 auto.abstract。backups：bak1=revision6、bak2=5、bak3=4；真实 exports 尚未生成。本轮工程准备不改索引或论文资产。

## 云与边界（工程准备前）

- 2026-10-07 `compshare doctor` 成功，SDK 0.11.114、凭证来自本机 default profile；不读取 / 输出平台私钥。
- 唯一实例 `cpod-1vbh7faqcauq`，名称 `mmfr-a2-4090-probe`，cn-bj2-03，Postpay，系统盘50 GB，无数据盘。工程准备前 Stopped / GPU=0、未设置计划关机；用户决定暂不释放。
- 上一阶段未授权 GPU、训练、长评价、official test；本轮授权仅工程准备、本地一次提交、现有云实例无卡准备，仍不授权训练、下载新数据集或推送。
