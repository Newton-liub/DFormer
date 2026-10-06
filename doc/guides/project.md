# 项目目录与文档约定

> 角色：新 DFormer 研究仓库的目录入口与约定说明。实时事实见 `doc/state/current.md`，不要在这里复制状态。

## 目录结构

保持作者仓库原样，只在确有内容时新增目录：

```text
DFormer/
├─ README.md / LICENSE / figs/        作者原有说明与图，保持原样
├─ train.sh / eval.sh / infer.sh       作者示例启动脚本，不是我们的研究入口
├─ models/ utils/ mmseg/ local_configs/ 作者代码
├─ doc/                                我们的正式文档
│  ├─ state/current.md                 唯一实时状态
│  ├─ ideas/ideas.md                   轻量 idea 记录
│  ├─ plans/                            实施计划（有内容才创建）
│  ├─ reports/                          实验结果、阶段报告、交接（有内容才创建）
│  ├─ guides/project.md                本文件
│  ├─ guides/cloud.md                   云上重复流程
│  └─ archive/                          阶段结束后的状态与文档快照（有内容才创建）
├─ research/                            我们的研究模块与薄训练/评价入口（有内容才创建）
├─ local_configs/research/              我们的独立研究配置（有内容才创建）
└─ outputs/                             本地与云端取回产物的统一入口，默认不进 Git
```

## 约定

- 作者目录尽量不动；需要改上游代码时优先新增薄层，并记录原因。旧项目研究目录（MMFR、LER、protocols、experiments、human、tests）不恢复。
- 我们的研究代码按方向放在 `research/`，配置放在 `local_configs/research/`，沿用作者的 `--config <module>` 加载方式，不修改作者的公共配置对象。
- 所有正式项目文档进入 `doc/`，不散落在根目录；实验结论进 `doc/reports/`，`doc/state/current.md` 只保留摘要和指针。
- 产物统一使用 `outputs/<experiment>/<run-id>/`，按需使用 `checkpoints/`、`logs/`、`swanlab/`、`predictions/`、`eval/`、`tmp/`。取回的代码或配置先当待审材料，复核后归位到 `research/` 或 `local_configs/research/`。
- 数据集、预训练权重、checkpoint、预测、大日志不进 Git；作者的默认 `checkpoints/` 与 `datasets/` 已由作者的 `.gitignore` 忽略。
- 论文库和外部代码保持在仓库外：论文全文 `D:\0Project\origin\论文\`，外部 clone `D:\0Project\origin\`。旧索引位于归档 `D:\0Project\DFormer-archive-20261007\MMFR\03_reference\`；确需维护时定点迁移，不新建第二套索引或编号体系。

## 外部与历史入口

- 旧项目完整归档：`D:\0Project\DFormer-archive-20261007\`（含旧 Git 历史、旧实时状态 `doc/main/`、旧证据 `MMFR/`）。需要旧实现时按需取用，默认不迁回。
- 作者 upstream：`upstream/main`；个人 fork：`origin/main`。当前研究分支从 upstream 最新提交建立。

## 分工原则

作者的模型、骨干与数据管线承担通用能力；我们的研究作为薄层叠加。判断标准是“下一轮研究最低限度需要什么”，不是“旧项目哪些东西全部搬回来”。
