# CompShare 容器 File Browser 兼容配置与故障恢复指南

> 核验日期：2026-08-27
> 生命周期边界：File Browser 上游已宣布于 2026-09-01 归档，最后一个计划版本已经发布，之后不再提供发布、缺陷修复或安全修复。本项目将它视为遗留兼容组件，而不是新部署的无条件推荐方案。

本文只用于 CompShare（优云智算）厂商预装 File Browser 的兼容配置和故障恢复，目标是让平台“文件管理”入口在受控条件下工作。新实例应先检查平台是否提供受维护的替代服务；继续使用本指南时，必须记录实际二进制版本，并确认平台入口同时提供 TLS 和独立访问控制。

官方平台说明：[文件管理功能](https://www.compshare.cn/docs/operation/gpu/filebrowser)；上游生命周期与安全说明：[filebrowser/filebrowser](https://github.com/filebrowser/filebrowser)。

> 安全边界：不要把 `8889` 直接暴露到无额外认证的公网，保持 command runner 禁用。上游建议使用非特权账号和最小目录挂载；本文的 root 示例只用于兼容 CompShare 厂商预装路径与既有启动方式，不代表通用安全最佳实践。

## 一、适用范围与工作方式

- 仅适用于 CompShare **容器实例**；虚机实例不适用。
- 核验时 CompShare 平台 File Browser 入口使用 HTTP 端口 `8889`；新实例仍须以平台页面为准。
- 平台实例详情页会显示登录用户名和初始密码；点击“文件管理”时，平台将凭据自动填入登录页面。
- FileBrowser 与 OpenList 是两套独立服务，可以同时运行：

```text
FileBrowser：8889
OpenList：5244
```

推荐路径关系：

```text
FileBrowser 程序：/model/other/filebrowser/filebrowser
FileBrowser 数据库：/root/filebrowser.db
FileBrowser 日志：/opt/filebrowser/logs/filebrowser.log
FileBrowser 管理根目录：/root/rivermind-data
FileBrowser 自启动脚本：/start.d/filebrowser.sh
```

默认只开放 `/root/rivermind-data`。不要默认开放整个 `/root`，否则网页可以访问 `.ssh`、Shell 历史、服务密码文件和其他敏感配置。

## 二、新容器的兼容初始化流程

### 2.1 创建实例并确认平台条件

1. 创建 CompShare 容器实例；
2. 等待实例状态变为“运行中”；
3. 在实例详情页确认存在“文件管理”入口；
4. 点击入口旁的提示，确认平台显示：

```text
登录用户名：admin
初始密码：平台生成的密码
```

只在自己的服务器终端和平台页面中使用初始密码。不要将密码、带 token 的登录 URL、Cookie 或数据库提交到 Git 仓库。

### 2.2 优先测试平台原生流程

先直接点击实例详情页的“文件管理”。正常情况应当自动进入文件列表，不需要手动输入用户名和密码。

如果能够正常进入，仍建议完成本文的“自启动配置”和“重启验证”。

根据页面结果选择后续操作：

```text
正常进入文件列表      → 配置自启动并验证
502 / 服务未启动      → 执行完整初始化
用户名或密码错误      → 执行密码同步
```

### 2.3 检查厂商预装程序

在容器终端执行：

```bash
FB_BIN=/model/other/filebrowser/filebrowser

test -x "$FB_BIN" \
  && "$FB_BIN" version | tee /root/rivermind-data/filebrowser-version.txt \
  || echo "未找到厂商预装的 FileBrowser"
```

如果程序不存在，不要从不明来源下载同名二进制文件。优先更换为带平台软件环境的镜像，或联系 CompShare 客服确认当前镜像是否支持文件管理。

### 2.4 创建目录

```bash
install -d -m 750 /opt/filebrowser/logs
install -d -m 750 /root/rivermind-data
install -d -m 755 /start.d
```

### 2.5 初始化数据库并同步平台密码

先从实例详情页复制“初始密码”，然后在容器终端执行以下完整命令。密码通过隐藏输入读取，不会直接写入 Shell 命令历史。

```bash
FB_BIN=/model/other/filebrowser/filebrowser
FB_DB=/root/filebrowser.db
FB_ROOT=/root/rivermind-data

read -r -s -p "粘贴平台显示的初始密码：" FB_PASSWORD
printf '\n'

pkill -x filebrowser 2>/dev/null || true

if [ ! -f "$FB_DB" ]; then
  "$FB_BIN" config init \
    --database "$FB_DB" \
    --address 0.0.0.0 \
    --port 8889 \
    --root "$FB_ROOT" \
    --locale zh-cn \
    --branding.name "文件管理"
fi

if "$FB_BIN" users ls --database "$FB_DB" | grep -Eq '^[[:space:]]*[0-9]+[[:space:]]+admin[[:space:]]'; then
  "$FB_BIN" users update admin \
    --password "$FB_PASSWORD" \
    --locale zh-cn \
    --database "$FB_DB"
else
  "$FB_BIN" users add admin "$FB_PASSWORD" \
    --perm.admin \
    --locale zh-cn \
    --database "$FB_DB"
fi

unset FB_PASSWORD
chmod 600 "$FB_DB"
```

注意：修改用户前必须停止 FileBrowser，否则 Bolt 数据库被运行进程占用时，管理命令可能返回：

```text
Error: timeout
```

## 三、配置容器自启动

### 3.1 创建启动脚本

创建 `/start.d/filebrowser.sh`：

```bash
cat > /start.d/filebrowser.sh <<'EOF'
#!/bin/sh
set -eu

FB_BIN=/model/other/filebrowser/filebrowser
FB_DB=/root/filebrowser.db
FB_ROOT=/root/rivermind-data
FB_LOG_DIR=/opt/filebrowser/logs
FB_LOG_FILE=${FB_LOG_DIR}/filebrowser.log
FB_PID_FILE=/opt/filebrowser/filebrowser.pid

install -d -m 700 "$(dirname "$FB_DB")"
install -d -m 750 "$FB_ROOT" "$FB_LOG_DIR"
chmod 700 "$(dirname "$FB_DB")"

if [ ! -x "$FB_BIN" ]; then
    printf '%s\n' "FileBrowser binary not found: $FB_BIN" >>"$FB_LOG_FILE"
    exit 1
fi

if [ ! -f "$FB_DB" ]; then
    printf '%s\n' "FileBrowser database not initialized: $FB_DB" >>"$FB_LOG_FILE"
    exit 1
fi

if pgrep -f "^${FB_BIN} .*--port 8889" >/dev/null 2>&1; then
    exit 0
fi

nohup "$FB_BIN" \
    --address 0.0.0.0 \
    --port 8889 \
    --root "$FB_ROOT" \
    --database "$FB_DB" \
    --log stdout \
    >>"$FB_LOG_FILE" 2>&1 &

printf '%s\n' "$!" >"$FB_PID_FILE"
EOF

chmod 755 /start.d/filebrowser.sh
```

脚本中不保存平台初始密码。密码经过哈希后保存在 FileBrowser 数据库中，自启动只负责使用数据库启动服务。

### 3.2 首次启动

```bash
/start.d/filebrowser.sh
```

确认进程：

```bash
ps -ef | grep '[f]ilebrowser'
```

预期参数包含：

```text
--address 0.0.0.0
--port 8889
--root /root/rivermind-data
--database /root/filebrowser.db
```

确认容器本机 HTTP 响应：

```bash
curl --silent --show-error \
  --output /dev/null \
  --write-out '%{http_code}\n' \
  http://127.0.0.1:8889/
```

返回 `200`、`302` 或其他有效 HTTP 状态表示服务已响应；`000` 或连接拒绝表示服务没有正常监听。

查看日志：

```bash
less /opt/filebrowser/logs/filebrowser.log
```

查看完成后按 `q` 退出。

### 3.3 验证公网入口

1. 关闭之前打开的 FileBrowser 页面；
2. 回到 CompShare 实例详情页；
3. 重新点击“文件管理”；
4. 必要时按 `Ctrl+F5` 强制刷新；
5. 确认自动进入文件列表；
6. 确认能看到 `/root/rivermind-data` 内的内容。

不要长期复用手工复制的带 token URL。优先始终从实例详情页重新点击“文件管理”。

## 四、完整功能验证

### 4.1 创建隔离测试目录

```bash
mkdir -p /root/rivermind-data/filebrowser-smoke-test
```

### 4.2 在网页中测试

进入：

```text
filebrowser-smoke-test
```

依次验证：

1. 新建文件 `smoke-test.txt`；
2. 刷新后仍能看到文件；
3. 上传一个无敏感信息的小文件；
4. 下载该文件；
5. 删除该文件。

### 4.3 清理

```bash
rm -rf /root/rivermind-data/filebrowser-smoke-test
```

不要在真实训练数据目录中测试删除功能。

## 五、重启验证

配置完成后，应至少执行一次普通关机、启动或重启验证。

重启后检查：

```bash
ps -ef | grep '[f]ilebrowser'
```

```bash
curl --silent --show-error \
  --output /dev/null \
  --write-out '%{http_code}\n' \
  http://127.0.0.1:8889/
```

然后从实例详情页再次点击“文件管理”。

如果重启后没有进程，检查：

```bash
ls -l /start.d/filebrowser.sh
```

应具有执行权限，例如：

```text
-rwxr-xr-x
```

手动恢复：

```bash
/start.d/filebrowser.sh
```

如果当前镜像不执行 `/start.d`，应联系平台确认镜像的自启动机制，不要同时叠加多个不明确的守护方案。

## 六、常见问题

### 6.1 页面显示“服务未启动”或返回 502

含义：平台代理能够识别 `8889` 入口，但容器内没有可访问的 FileBrowser 服务。

检查：

```bash
ps -ef | grep '[f]ilebrowser'
```

```bash
curl --silent --show-error \
  --output /dev/null \
  --write-out '%{http_code}\n' \
  http://127.0.0.1:8889/
```

恢复：

```bash
/start.d/filebrowser.sh
```

如果日志提示二进制或数据库不存在，返回本文第二部分重新初始化。

### 6.2 页面显示“用户名或密码错误”

含义：FileBrowser 已经运行，但数据库中的 `admin` 密码与平台当前显示的初始密码不一致。

先从实例详情页重新复制当前初始密码，再执行：

```bash
FB_BIN=/model/other/filebrowser/filebrowser
FB_DB=/root/filebrowser.db

pkill -x filebrowser 2>/dev/null || true
cp -p "$FB_DB" "${FB_DB}.before-password-reset"

read -r -s -p "粘贴平台当前显示的初始密码：" FB_PASSWORD
printf '\n'

"$FB_BIN" users update admin \
  --password "$FB_PASSWORD" \
  --database "$FB_DB"

unset FB_PASSWORD
/start.d/filebrowser.sh
```

关闭错误页面，再从实例详情页重新点击“文件管理”。

如果修复失败，可以恢复备份：

```bash
pkill -x filebrowser 2>/dev/null || true
cp -p \
  /root/filebrowser.db.before-password-reset \
  /root/filebrowser.db
/start.d/filebrowser.sh
```

### 6.3 用户管理命令返回 `Error: timeout`

FileBrowser 进程正在占用数据库。先停止服务：

```bash
pkill -x filebrowser
```

确认进程消失后再执行 `users ls`、`users add` 或 `users update`。

### 6.4 平台初始密码发生变化

不要修改启动脚本保存新密码。按 6.2 节重新同步数据库中的 `admin` 密码即可。

### 6.5 网页能打开但无法创建或上传文件

检查管理根目录权限和磁盘空间：

```bash
ls -ld /root/rivermind-data
df -h /root/rivermind-data
```

本兼容流程沿用厂商 root 启动方式，因此通常具有写权限；只要平台支持非特权运行，就应改用最小权限账号和目录挂载。如果网页写入请求返回 `502`，检查进程是否在请求后退出，并读取日志：

```bash
ps -ef | grep '[f]ilebrowser'
less /opt/filebrowser/logs/filebrowser.log
```

## 七、管理整个 `/root` 的可选配置

只有确实需要管理代码、配置和其他根目录文件时，才将管理根目录改为：

```text
/root
```

需要同时修改：

1. 数据库初始化或配置中的 `--root`；
2. `/start.d/filebrowser.sh` 中的 `FB_ROOT`。

启动参数改为：

```bash
--root /root
```

风险包括网页可见或可下载：

```text
/root/.ssh
/root/.bash_history
/root/openlist-admin-password
/root/filebrowser.db
其他服务凭据与配置
```

更安全的方案是继续使用 `/root/rivermind-data`，通过 SSH 或终端管理程序配置。

## 八、持久化与安全

- FileBrowser 数据库沿用厂商当前启动约定，保存在 `/root/filebrowser.db`，权限保持 `600`；
- 日志保存在 `/opt/filebrowser/logs`；
- 管理目录默认是 `/root/rivermind-data`；
- 自启动脚本不包含明文密码；
- 普通关机和重启通常保留启动盘内容；
- 删除实例、重装实例或更换启动盘可能丢失数据库和本地文件；
- 不要提交 `filebrowser.db`、数据库备份、初始密码或登录 URL；
- 不要在聊天、截图和 Issue 中暴露带 token 的公网地址；
- 数据库备份同样包含认证数据，应限制权限并避免放在 FileBrowser 可浏览目录中；
- 重要训练数据应保存到独立数据盘或另行备份。

## 九、新容器配置检查清单

配置完成后逐项确认：

- [ ] 实例类型是容器实例；
- [ ] 平台存在“文件管理”入口；
- [ ] `/model/other/filebrowser/filebrowser` 存在且可执行；
- [ ] 平台 HTTP `8889` 入口可用；
- [ ] 数据库位于 `/root/filebrowser.db`；
- [ ] `admin` 密码与平台当前初始密码一致；
- [ ] `/start.d/filebrowser.sh` 存在且可执行；
- [ ] FileBrowser 监听 `0.0.0.0:8889`；
- [ ] 本机 HTTP 检查返回有效状态码；
- [ ] 从平台入口可以自动登录；
- [ ] 可以浏览、创建、上传、下载和删除测试文件；
- [ ] 重启容器后服务能自动恢复；
- [ ] 文档、Git 和聊天中没有保存真实密码或 token。