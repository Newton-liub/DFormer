# MUSeg 云端数据盘归档、审计与人工清理

> 适用范围：后续 MUSeg 训练开始前或已完成训练的证据取回后。
> 当前边界：正在运行的训练期间不执行本指南，不移动当前输出，也不清理共享数据。
> 安全原则：工具只生成审计报告，不删除任何文件；实际删除必须由用户确认精确路径。

## 1. 为什么增加候选前先治理数据盘

后续训练预登记保留单尺度 validation 的 top 8 checkpoint，并持续覆盖 `latest.pth`。训练结束后按 SHA-256 去重，最多有 9 个候选需要取回本地运行五尺度翻转主 evaluator。

增加候选可以降低低成本单尺度 selector 与最终主 evaluator 排序不一致造成的漏选风险，但不会保证两种验证排序完全一致。按本机最大样本实测的保守外推，9 个候选串行主评估约 6.3 小时，仍低于 8 小时硬上限。

## 2. 永久保护项

每次审计都必须显式保护以下实际路径：

- 当前 Git 仓库；
- 当前训练或待恢复运行的完整输出目录；
- 当前使用的 `MUSeg_DFormer` 数据集；
- official test 封存文件及 split authority；
- 当前预训练权重；
- OpenList 配置、凭据和正在执行复制任务的源文件；
- 数据盘根目录本身。

保护路径与候选路径只要存在父子重叠，审计就会失败。不要为了让审计通过而缩小保护范围。

## 3. 先取回并核验归档

在云端生成归档和 SHA-256 sidecar，例如：

```bash
tar -cf /root/rivermind-data/archives/RUN_ID.tar /exact/completed/run/path
sha256sum /root/rivermind-data/archives/RUN_ID.tar \
  > /root/rivermind-data/archives/RUN_ID.tar.sha256
```

把归档和 sidecar 取回本地后，在 Windows PowerShell 重新计算：

```powershell
Get-FileHash -Algorithm SHA256 D:\exact\archive\RUN_ID.tar
Get-Content D:\exact\archive\RUN_ID.tar.sha256
```

只有文件大小和 SHA-256 都一致，才把“本地取回完成”记为已核验。压缩命令、归档路径、字节数、SHA-256 和核验时间应写入该运行的证据清单。

## 4. 可选的 OpenList 个人云盘副本

OpenList 的本机存储 `/server` 对应云服务器实际目录，夸克存储挂载为 `/quark`。已有下载指南见 [`openlist-quark-download.md`](openlist-quark-download.md)。夸克驱动是未主动维护的逆向接口，反向上传前必须先用无敏感信息的小文件验证。

小文件验证通过后，在 OpenList 首页执行：

1. 进入 `/server/archives`，勾选归档和对应 `.sha256` sidecar；
2. 选择“复制到”，目标设为个人云盘中的明确目录，例如 `/quark/project/museg-archives/RUN_ID/`；
3. 在“任务”页面等待两个文件都显示成功；
4. 重新打开目标目录，核对文件名和显示大小；
5. 保留本地已核验副本，不把 OpenList 的“成功”状态单独当作内容完整证明。

大文件复制期间不要关闭实例、禁用存储或删除 `/server` 源文件。Cookie、密码和访问链接不得写入仓库、审计报告或聊天记录。

## 5. 运行只读审计

审计必须列出每一个候选和保护路径，不接受通配符。先取得一个同类 checkpoint 的实际字节数：

```bash
stat -c %s /exact/current/run/checkpoint/latest.pth
```

然后运行：

```bash
python tools/audit_museg_cloud_storage.py \
  --storage-root /root/rivermind-data \
  --candidate /root/rivermind-data/old-test-project \
  --candidate /root/rivermind-data/dataset/obsolete-duplicate \
  --protect /root/rivermind-data/DFormer \
  --protect /root/rivermind-data/dataset/MUSeg_DFormer \
  --protect /root/rivermind-data/outputs/CURRENT_RUN \
  --protect /root/rivermind-data/pretrained/DFormerv2_Small_pretrained.pth \
  --checkpoint-size-bytes ACTUAL_CHECKPOINT_BYTES \
  --checkpoint-candidates 9 \
  --report /root/rivermind-data/DFormer/cloud/storage-audits/before-next-run.json
```

报告包含候选文件数、目录数、占用、最近修改时间、当前剩余空间、预计释放空间和 9 个 checkpoint 的纯文件预算。预算不包含数据集、日志、归档和临时文件，不能把 `projected_free_covers_checkpoint_storage_only=true` 解释为整个训练空间一定充足。

审计会拒绝：

- 数据盘根目录；
- 数据盘外路径；
- 候选与保护项之间的父子重叠；
- 重复候选；
- 逃出数据盘的符号链接；
- 不存在或无法解析的路径。

## 6. 人工确认和删除

删除前把以下信息交给用户确认：

- 每个候选的规范化绝对路径；
- 删除原因；
- 文件数与占用；
- 本地归档路径、字节数和已核验 SHA-256；
- 可选 OpenList 目标目录及复制状态；
- 删除后预计剩余空间；
- 所有保护路径。

用户必须确认具体路径，不能只确认“清理旧文件”。获得确认后，再逐项复核并删除：

```bash
realpath -- /exact/approved/path
du -sh -- /exact/approved/path
rm -rf -- /exact/approved/path
```

每条删除命令只包含一个已批准的绝对路径。禁止使用 `*`、按名称模糊匹配、删除数据盘根目录或把多个未经单独确认的目录放进同一命令。删除后重新运行只读审计或 `df -B1`，记录实际剩余空间。

## 7. 每次训练的顺序

1. 当前训练结束并形成技术终态；
2. 打包必要证据并写 SHA-256 sidecar；
3. 取回本地并重新核验哈希；
4. 可选复制到 OpenList 个人云盘并核对目标文件；
5. 运行只读数据盘审计；
6. 用户确认精确删除清单；
7. 人工逐项删除并复核剩余空间；
8. 从干净 Git commit 物化后续 protocol，运行正式 preflight；
9. 另行授权后才启动下一次训练。
