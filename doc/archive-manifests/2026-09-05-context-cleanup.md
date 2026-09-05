# 2026-09-05 上下文清理归档清单

> **文档角色：** 历史材料归档清单。
> **形成时点：** 2026-09-05。
> **实时入口：** `doc/main/MUSeg-current-status.md`。
> **后继关系：** 被归档材料不承担当前状态、研究授权或恢复点职责；相关 MUSeg 事实以实时状态和日期化报告为准。

本清单对应仓库外保留的 ZIP：`archive-export/2026-09-05-DFormer-context-cleanup.zip`。ZIP 不纳入 Git，用户可在确认后将其移动到其他位置留档。归档材料保留原仓库相对路径，并在 ZIP 内附带 `ARCHIVE-MANIFEST.json`。

## 归档文件

| 原路径 | 作用与归档理由 | 字节数 | SHA-256 |
|---|---|---:|---|
| `liu-test-exp/方案1/最短验证路径.md` | 已被正式方向1计划替代的早期最短路径草案。 | 21,658 | `fdc5a78e49868c480ec5557796b5b729aac6df11a50daa684c866506e930ee1f` |
| `liu-test-exp/方案1/段落主题1_论文编号.txt` | 方向1早期论文编号原始材料；当前计划副本已随计划整理，原始版本保留归档。 | 68,251 | `5d725dd1896704d86b8d6a33e560ed93e47450071fdf36f4cd01995212d7a5b5` |
| `doc/临时/2026-08-28-MUSeg-Protocol-Gate方案分析与资源决策.md` | 已被后续 Quick-B0 收口事实吸收的阶段性协议/资源分析。 | 20,994 | `a20627e9f3a05ce6ae502ede3fe7b374e339fbb57bacd84a90ee2889e9813f0a` |
| `doc/临时/museg-b0-external-controller.md` | 已完成 Quick-B0 云端控制器交接说明。 | 6,751 | `a5cc66443b36fb1823315b9303bdfa5dfa689184ddaf6b3a23b1ecf2a9a77976` |
| `doc/临时/museg-b0-external-controller-receipt.md` | 已完成云端 schedule、证据取回和实例停止的历史回执。 | 6,310 | `4e195dc9ece2ad1340530f1b2161146663e5a26e24c3fd438dfa18aebc24d881` |
| `liu-test-exp/对抗.md` | 与 MUSeg 无关的舆论对抗逻辑草案，避免进入研究项目默认上下文。 | 13,180 | `8bc7caa15f05d8c39f64b8e98581514e22e33f1774a87e0c19a01420afbfd375` |
| `liu-test-exp/广义战略信息传播与群体认知偏移的数学模型.md` | 与 MUSeg 无关且篇幅较大的传播/群体认知数学草案，避免上下文污染。 | 28,866 | `0c990569d9263a1b420623a6db7ff38fb3f05b9084a3cef468f9d5a9a3887fe0` |
| `liu-test-exp/地下环境RGB-D可信度建模与风险感知.md` | 面向未来的地下环境 RGB-D 长期研究构想，范围超出当前 MUSeg 基线，保留供日后参考。 | 32,036 | `00d8be5f456b7d049355f2744acdaa740b54f13df7c4bca70d3f7012f9a9a8de` |

## 归档核验

- ZIP SHA-256：`393fde476b37dbece118dd0d3c3518d005c326db2da344047b44b94dcd4cbcf1`。
- ZIP 大小：78,988 bytes。
- 已在删除源文件前解压核对 8 个条目的字节数、归档清单 SHA-256 和源文件 SHA-256，结果全部一致。
- 恢复方式：将 ZIP 解压到临时位置，再按 `ARCHIVE-MANIFEST.json` 核对路径、大小和 SHA-256；恢复到仓库前需重新确认当前目录职责和引用。

## 另行删除

- `liu-test-exp/对抗 copy.md`：已确认是短小、不完整的重复草案，直接删除，不进入 ZIP。
- `.DS_Store`：已确认是无运行价值的 macOS Finder 元数据，直接删除并加入 `.gitignore`。
