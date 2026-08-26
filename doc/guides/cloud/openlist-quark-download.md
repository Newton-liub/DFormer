# OpenList 夸克网盘云端数据取回指南

> 核验日期：2026-08-27
> 适用版本：OpenList `v4.2.5`，核验时为最新稳定版；使用其他版本前必须重新对照官方界面和文档。
> 驱动边界：普通夸克驱动是基于历史产品的逆向接口，OpenList 项目组不会主动维护，也不接受相关修复请求；官方当前说明夸克传输只能使用本地代理。它只适合作为一次性、可复核的数据取回链路，不作为长期稳定存储或持续集成依赖。

本文用于在 CompShare 容器实例中部署 OpenList，挂载夸克网盘，并通过 OpenList 网页把文件复制到云服务器本地目录。文件复制任务在云服务器内执行，不经过本地电脑；任务进度可在 OpenList 的“任务”页面查看。官方版本见 [OpenList releases](https://github.com/OpenListTeam/OpenList/releases)，夸克驱动限制见 [OpenList 夸克驱动说明](https://doc.oplist.org/guide/drivers/quark)。

## 一、方案结构

```text
夸克网盘
    ↓ OpenList 夸克驱动
OpenList 虚拟路径：/quark
    ↓ OpenList 复制任务
OpenList 本机存储：/server
    ↓
云服务器真实目录：/root/rivermind-data
```

本方案适合将数据集下载到云服务器后进行训练。不要直接在本地浏览器点击普通“下载”，否则文件会下载到本地电脑，而不是云服务器。

## 二、已验证实例配置（2026-08-26 快照）

- 实例类型：CompShare 容器实例
- OpenList 版本：`v4.2.5`
- OpenList 安装目录：`/opt/openlist`
- OpenList 数据目录：`/opt/openlist/data`
- OpenList 日志目录：`/opt/openlist/logs`
- OpenList 监听端口：`5244`
- 云服务器本地数据目录：`/root/rivermind-data`
- OpenList 管理员密码文件：`/root/openlist-admin-password`
- 自启动脚本：`/start.d/openlist.sh`
- 公网访问地址：由 CompShare 的 `5244` 软件入口提供

管理员密码文件只保存在云服务器上，权限应为 `600`。不要将密码、夸克 Cookie、Access Token 或 Refresh Token 写入本文、聊天记录、截图或 Git 仓库。

## 三、部署 OpenList

### 3.1 实例要求

本次快照使用容器实例，并映射 HTTP `5244` 端口。该实例入口不是 systemd，因此 OpenList 使用 `/start.d` 脚本自启动；新镜像必须先核对实际自启动机制。

建议至少准备：

- Ubuntu 容器镜像；
- x86_64 架构；
- 2 核 CPU、4GB 内存可以运行 OpenList；
- 启动盘空间应大于待下载文件和解压数据总量。

### 3.2 创建目录

在云服务器终端执行：

```bash
mkdir -p /opt/openlist/data
mkdir -p /opt/openlist/logs
mkdir -p /root/rivermind-data
```

### 3.3 放置 OpenList 程序

使用 OpenList 官方发布的 Linux x86_64 二进制文件，放置为：

```text
/opt/openlist/openlist
```

并授权执行：

```bash
chmod 755 /opt/openlist/openlist
```

本文不保存下载链接、管理员密码或其他实例专属凭据。使用时应从 OpenList 官方发布页取得与当前版本匹配的程序。

### 3.4 初始化数据目录和管理员密码

如果 OpenList 尚未初始化，先按 3.5 节用 `--data /opt/openlist/data` 启动前台 server 进程。首次启动后，从 `/opt/openlist/logs/openlist.log` 查找 OpenList 生成的初始管理员密码，用它登录后台并立即修改为自己的强密码。

确认新密码已经实际登录成功后，再把同一密码保存到只允许 root 读取的服务器文件：

```bash
umask 077
read -r -s -p "输入已经验证可登录的 OpenList 管理员密码：" OPENLIST_PASSWORD
printf '\n'
printf '%s\n' "$OPENLIST_PASSWORD" > /root/openlist-admin-password
unset OPENLIST_PASSWORD
chmod 600 /root/openlist-admin-password
```

不要把真实密码替换进本文，也不要把未实际设置到 OpenList 的占位文字保存成“密码文件”。如果选择使用 `openlist --data /opt/openlist/data admin set` 修改密码，应在服务停止时操作，并注意命令参数可能短暂出现在进程列表中；优先使用管理后台。

确认权限：

```bash
ls -l /root/openlist-admin-password
```

应显示类似：

```text
-rw-------
```

### 3.5 手动启动 OpenList

```bash
nohup /opt/openlist/openlist \
  --data /opt/openlist/data \
  --log-std \
  server \
  >>/opt/openlist/logs/openlist.log 2>&1 &
```

确认进程：

```bash
ps -ef | grep '[o]penlist'
```

确认本机网页响应：

```bash
curl --fail --silent --show-error http://127.0.0.1:5244/ >/dev/null
```

### 3.6 配置 CompShare 端口

在 CompShare 实例的端口配置中确认 HTTP 端口包含：

```text
5244
```

平台通常会为该端口生成类似下面的公网入口：

```text
https://5244-<实例ID>.pod.compshare.cn/
```

实际地址以 CompShare 实例页面显示的“5244”软件入口为准。

### 3.7 配置自启动

CompShare 的容器启动脚本放在 `/start.d`，并且必须有执行权限。创建脚本：

```bash
cat > /start.d/openlist.sh <<'EOF'
#!/bin/sh
set -eu

OPENLIST_BIN=/opt/openlist/openlist
OPENLIST_DATA=/opt/openlist/data
OPENLIST_LOG_DIR=/opt/openlist/logs
OPENLIST_PID_FILE=/opt/openlist/openlist.pid

mkdir -p "$OPENLIST_DATA" "$OPENLIST_LOG_DIR" /root/rivermind-data

if pgrep -f "^${OPENLIST_BIN} --data ${OPENLIST_DATA} --log-std server" >/dev/null 2>&1; then
    exit 0
fi

nohup "$OPENLIST_BIN" \
    --data "$OPENLIST_DATA" \
    --log-std \
    server \
    >>"$OPENLIST_LOG_DIR/openlist.log" 2>&1 &
printf '%s\n' "$!" >"$OPENLIST_PID_FILE"
EOF

chmod 755 /start.d/openlist.sh
```

确认脚本可执行：

```bash
ls -l /start.d/openlist.sh
```

模拟服务恢复，验证脚本可以启动 OpenList：

```bash
pkill -x openlist
/start.d/openlist.sh
```

等待几秒后检查：

```bash
ps -ef | grep '[o]penlist'
curl --fail --silent --show-error http://127.0.0.1:5244/ >/dev/null
```

再次执行脚本不会重复启动进程：

```bash
/start.d/openlist.sh
```

## 四、登录 OpenList 管理后台

打开 CompShare 实例页面提供的 `5244` 公网地址，例如：

```text
https://5244-<实例ID>.pod.compshare.cn/
```

使用 OpenList 管理员账号登录。用户名通常为：

```text
admin
```

首次启动时先从日志读取系统生成的初始密码；修改密码并验证登录成功后，才能把最终密码写入 `/root/openlist-admin-password`。如果需要在服务器终端读取已验证密码，使用：

```bash
less /root/openlist-admin-password
```

查看完成后按 `q` 退出。不要将密码复制到聊天记录或截图中。

## 五、挂载夸克网盘

进入 OpenList 管理后台：

```text
存储 → 添加
```

本次已验证实例选择普通 Cookie 驱动：

```text
夸克
```

该驱动随夸克接口变化可能随时失效；先用无敏感信息的小文件验证目录列出和跨存储复制，再处理数据归档。不要选择：

- 夸克网盘 Open；
- 夸克 TV；
- 只读驱动。

### 5.1 推荐字段

| 字段 | 填写值 | 说明 |
|---|---|---|
| 驱动 | `夸克` | 使用普通 Cookie 驱动 |
| 挂载路径 | `/quark` | OpenList 中的虚拟入口，不能重复 |
| 序号 | `0` | 保持默认 |
| 备注 | `夸克网盘` | 可留空 |
| 缓存过期时间 | `30` | 保持默认 |
| 网页代理 | 按 `v4.2.5` 当前界面和小文件测试确认 | 不把浏览器 302 当作跨存储复制成功证据 |
| 默认下载方式 | `本地代理`（界面可选时） | 官方当前说明夸克传输只能使用本地代理 |
| 下载代理网址 | 留空 | 没有额外代理时留空 |
| 禁用代理签名 | 关闭 | 保留签名校验 |
| 禁用索引 | 关闭 | 允许正常列出目录 |
| 启用签名 | 关闭 | 当前不需要额外签名 |
| 根文件夹 ID | `0` | 挂载整个夸克网盘 |
| 排序依据 | `无` | 保持默认 |
| 排序方式 | `升序` | 保持默认 |
| 使用转码地址 | 关闭 | 数据集下载不需要视频转码 |
| 仅列出视频文件 | 关闭 | 否则压缩包等文件不可见 |

### 5.2 获取 Cookie

在本地 Windows 浏览器中获取 Cookie，但不要把 Cookie 发到聊天中：

1. 打开夸克网盘网页版并登录；
2. 按 `F12` 打开开发者工具；
3. 进入“网络”或 `Network`；
4. 刷新夸克网盘页面；
5. 选择发往夸克网盘 API 的请求；
6. 在“请求标头”中找到 `Cookie`；
7. 复制 `Cookie:` 后面的完整值，不要复制 `Cookie:` 本身；
8. 将完整 Cookie 粘贴到 OpenList 的 `Cookie` 字段。

Cookie 相当于登录凭据，只应直接填写到自己的 OpenList 后台。不要使用第三方网页代填或公开 Cookie。

点击保存后，存储状态应显示：

```text
工作中
```

然后进入 OpenList 首页，确认可以打开：

```text
quark
```

## 六、添加云服务器本机存储

为了让 OpenList 在云服务器内部复制文件，需要把服务器目录添加为第二个存储。

进入：

```text
存储 → 添加
```

选择：

```text
本机存储
```

填写：

| 字段 | 填写值 |
|---|---|
| 驱动 | `本机存储` |
| 挂载路径 | `/server` |
| 根文件夹 | `/root/rivermind-data` |
| 序号 | `0` |
| 备注 | `云服务器本地目录` |

其他字段保持默认后保存。

最终路径关系：

```text
OpenList 虚拟路径：/server
云服务器真实路径：/root/rivermind-data
```

`/server` 只是 OpenList 中的虚拟入口，不是额外创建的服务器目录。服务器实际保存位置仍是 `/root/rivermind-data`。

## 七、通过网页可视化复制文件到云服务器

这是本项目在已验证 `v4.2.5` 实例上的一次性取回方式。文件不会下载到本地 Windows 电脑，而是在云服务器内部执行复制任务；驱动显示“成功”仍需由文件大小、哈希或后续结构化数据审计独立验收。

### 7.1 选择源文件

打开 OpenList 首页，依次进入：

```text
quark → project → cmx
```

选择要下载的文件，例如：

```text
MUSeg.7z
```

实际文件名区分大小写。本实例中真实名称是 `MUSeg.7z`，不是 `MUseg.7z`。

### 7.2 复制到服务器本地存储

1. 勾选目标文件；
2. 点击顶部的“复制”或“复制到”；
3. 目标位置选择 `/server`；
4. 确认执行。

复制任务对应的路径关系为：

```text
/quark/project/cmx/MUSeg.7z
    ↓
/server/MUSeg.7z
    ↓
/root/rivermind-data/MUSeg.7z
```

不要点击普通“下载”按钮。普通“下载”会将文件传输到当前本地浏览器的下载目录，而不是服务器。

### 7.3 查看可视化进度

点击 OpenList 左侧：

```text
任务
```

可以查看复制任务的：

- 执行状态；
- 已完成大小；
- 总大小；
- 进度；
- 传输速度；
- 成功或失败信息。

任务状态通常包括：

```text
执行中
成功
失败
```

大文件复制期间不要删除源文件、禁用 `/quark` 存储或关闭实例。

## 八、下载完成后的检查

可视化任务显示成功后，在云服务器终端检查目标文件：

```bash
ls -lh /root/rivermind-data/MUSeg.7z
```

对于本次取回的同一来源 `MUSeg.7z`，已验证的完整大小是：

```text
2049552128 字节
```

该大小只约束本次同源归档，不是所有 MUSeg 发布包的通用身份。若来源提供 SHA-256，应优先比对哈希；至少还要运行压缩包完整性测试和 `tools/prepare_museg.py` 的结构化输出验证。

如果只是确认文件已生成，也可以执行：

```bash
test -f /root/rivermind-data/MUSeg.7z \
  && echo "服务器文件已生成" \
  || echo "服务器文件不存在"
```

数据集需要解压时，在服务器终端执行：

```bash
apt-get update
apt-get install -y p7zip-full
```

测试压缩包：

```bash
7z t /root/rivermind-data/MUSeg.7z
```

正常结果应包含：

```text
Everything is Ok
```

解压到 MUSeg 转换脚本的默认原始数据目录：

```bash
mkdir -p /root/rivermind-data/dataset/MUSeg
7z x \
  /root/rivermind-data/MUSeg.7z \
  -o/root/rivermind-data/dataset/MUSeg
```

随后在 `/root/rivermind-data/DFormer` 运行 `python tools/prepare_museg.py`，默认输出到 `/root/rivermind-data/dataset/MUSeg_DFormer`。如果保留其他解压目录，必须显式传入 `--source-root` 和 `--output-root`。

确认解压结果：

```bash
ls -lah /root/rivermind-data/dataset/MUSeg
du -sh /root/rivermind-data/dataset/MUSeg
```

确认数据完整后，如需释放压缩包占用的空间，再删除服务器副本：

```bash
rm /root/rivermind-data/MUSeg.7z
```

该命令只删除云服务器上的副本，不会删除夸克网盘中的原文件。

## 九、常见问题

### 首页看不到 `quark`

进入“存储”页面确认：

- `/quark` 状态为“工作中”；
- 挂载路径没有重复；
- Cookie 是完整值且仍然有效；
- “仅列出视频文件”已关闭。

保存后刷新 OpenList 首页。

### 保存夸克存储时报 `401` 或 Cookie 无效

重新登录夸克网页版并重新获取完整 Cookie。Cookie 失效后，只需在 OpenList 后台更新 Cookie，不要把 Cookie 发给第三方服务。

### 复制任务返回 `403`

先确认 `/quark` 存储仍处于“工作中”，然后更新 Cookie 并重新保存夸克存储。必要时将“网页代理”开启后再测试小文件。

### 复制任务返回 `404`

使用 OpenList 网页重新进入目标目录，确认目录层级和文件名大小写。`MUSeg.7z` 与 `MUseg.7z` 不同，不能混用。

### 任务失败或没有进度

检查：

1. `/quark` 和 `/server` 两个存储是否都显示“工作中”；
2. `/root/rivermind-data` 是否有足够磁盘空间；
3. 源文件是否仍然存在；
4. OpenList 进程是否运行；
5. 是否误点了“下载”而不是“复制”。

在服务器终端检查：

```bash
ps -ef | grep '[o]penlist'
df -h /root/rivermind-data
```

### 重启后 OpenList 没有恢复

检查自启动脚本：

```bash
ls -l /start.d/openlist.sh
```

应具有执行权限，例如：

```text
-rwxr-xr-x
```

手动恢复：

```bash
/start.d/openlist.sh
```

确认服务：

```bash
curl --fail --silent --show-error http://127.0.0.1:5244/ >/dev/null \
  && echo "OpenList 正常" \
  || echo "OpenList 未正常响应"
```

## 十、安全与持久化说明

- 管理员密码文件位于 `/root/openlist-admin-password`，权限保持 `600`；
- OpenList 配置和数据库位于 `/opt/openlist/data`；
- 2026-08-26 已验证实例的配置保存在 50GB 启动盘中；
- 普通关机和重启通常会保留配置；
- 删除实例、重装实例或更换启动盘可能导致配置和本地数据丢失；
- 该实例没有独立数据盘，重要数据必须另行备份；
- 夸克 Cookie 只在自己的 OpenList 中使用，不要提交到仓库；
- 训练前应先将数据复制到 `/root/rivermind-data`，不要让训练程序直接依赖网盘挂载；
- 不要将本指南中的示例密码写入真实配置。