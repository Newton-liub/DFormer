# ODG 实现与无卡准备记录（2026-10-10）

## 当前结论

已实现 ODG（局部观测分布几何先验）及 `original / mean / odg` 三模式，完成限定 CPU 数值与梯度验收，独立训练/评价入口已准备。**没有新分割指标或方法增益结论；GPU 未获授权。** 本记录遵循[上级指令](../../临时/DFormer_ODG_Cursor_NoGPU_Instructions_20261010.md)，方法和后续命令见[ODG 入口](../guides/odg.md)。

## 实现与 Git

- 基础提交 `08d59409e5251379c3305d2058a05d972fe1eb2d`；独立分支 `research/odg-sunrgbd`，用户远端 `https://github.com/Newton-liub/DFormer.git`。只提交本任务文件，不 force push，不修改 upstream、旧结果或论文库。
- 核心实现 `71aa4e4645f44279dc47d729f058f0beedf0926a`；训练/评价 `2b2d754fba9980470e88d27d8c7bad3d7ca70d29`；手动启动脚本引号修正 `4f3e40529399aff91d7af9fc9eb2450a1d752498`。以上均已推送。收尾文档提交 ID 以本轮最终 Git 回读为准，不在文档内自指其尚未生成的 hash。
- 作者入口 `utils/train.py` 未改；研究代码在 `research/`，配置在 `local_configs/research/`。作者模型只改必要传递接口、几何切换、普通 BN 转换及 HAM 基向量跟随输入设备。
- ODG 无新增可学习参数，保留每 block 原权重及符号。输入16分箱→面积聚合→核归一化 log bias；前三级轴向、末级full，stage关系只在同一次forward共享。`mean`为同支持域均值的软分箱核消融，不增加第三组训练。
- 研究训练从相同官方 encoder pretrained 起步，新分割头、optimizer/scaler，全部层正常更新。LR在更新前写入；300epoch与30/100暂停分离；恢复仅同实验epoch边界。人工孔洞同时改depth和mask，不能让original实际收到clean而候选收到删除。

## 实际 CPU 检查

- 三模式均构建并加载官方预训练；各26,966,591参数。missing为三组segmentation侧`extra_norms`的标准LayerNorm仿射初值；unexpected为当前骨干不用的预训练分类/辅助头，没有补key或改loader。
- 常量深度、对角自关系、全空支持域、部分空域、轴向/full尺寸与有限性通过；FP32核/log不受CPU autocast降精度。空支持域保留空间项逐项相等，负`w_d`未被约束。
- 区间中心点质量与原深度差最大误差 $5.96\times10^{-8}$；选定off-grid小例子误差约0.02582，明确是近似，不声称任意深度严格等价。query/group分块与未分块关系差约 $1.81\times10^{-7}$。
- 屏蔽CUDA后，ODG batch2、64×64完整模型CPU一次前向/反向：loss4.1066256，714个可训练参数张量均有有限梯度；patch embedding、Q、Geo权重及decoder梯度均非零。
- SUN开发清单4757/528，重算298次optimizer尝试/epoch，补11张到4768；总89400次、warmup2980次。全train为331次/epoch。30/100暂停均不改变总日程。首步LR2.68456e-8、中点4.28701e-5、末步0，均在更新前使用；尚未实际训练。
- Python定点语法、入口帮助、CPU训练拒绝、恢复合同字段及人工孔洞共同输入检查通过。孔洞生成器按矩形图像支持域精确取整到25%/50%，padding保持；不规则支持域会拒绝，等待另行协议，不静默改变难度。
- 云端`bash -n research/run_odg_gpu.sh`及CPU日程通过；真实GPU AMP、容量、optimizer state设备迁移、吞吐/时长仍未实测。没有完整旧测试、全量模型推理、长CPU训练、权重重哈希或正式test评价。

## 数据与来源

- 本地SUN：`D:\0Project\dataset\SUNRGBD`，RGB/Depth/labels各10335，官方train5285/test5050；四清单全部路径配对0缺失，train-dev和dev无交集且并集等于官方train。原清单不改，图像级seed12345划分，不声称场景隔离。
- 五组dev样本已生成RGB/深度/标签/16区间示例：`outputs/odg-preparation/data-preview/`。主代理实际打开`preview_train_2651.png`，图像、深度及标签结构可对应；没有把它作为方法收益图。
- 现成`D:\0Project\dataset\SUNRGBD.zip`为2,452,204,576 bytes，单顶层SUNRGBD；抽查3个Depth成员及两张清单与落盘文件大小/CRC一致。来源一致性仅闭环到这个本地压缩包，**没有直接绑定作者下载源的记录**；包大小/布局匹配此前作者入口记录只是旁证。
- 深度保留作者灰度8位读取与0.48/0.28归一化。天然0未认作物理缺失，米制单位未知；不擅自改16位输入或发明映射。正式GPU运行前应由用户确认下载来源或替换为明确作者来源的整理包。
- 云端SUN已落盘于`/root/rivermind-data/dataset/SUNRGBD`：三模态各10335、train5285/test5050；四清单路径配对0缺失，5个train及5个dev样本CPU预处理可读、支持域与padding正常。解压按zip原结构进行，不覆盖已有SUN目录；空闲约20.00GiB。NYUv2本地/云端暂无数据，已准备作者40类入口；不阻塞SUN，也没有为了第二数据集发起长下载。可信整理版入口见[已有下载指南](../../临时/DATASET_DOWNLOAD_GUIDE.md#3-nyu-depth-v2第二阶段可选不立即下载)，目标`/root/rivermind-data/dataset/NYUDepthv2/{RGB,Depth,Label,train.txt,test.txt}`。DeLiVER不适配。

## 云端、监控与费用

- 当前会话没有发现CompShare MCP工具，复用已配置官方CLI0.3.6及既有凭据，明确记录此替代。现有实例`cpod-1vbh7faqcauq`、cn-bj2-03、Postpay、50GB系统盘；启动前Stopped，无其他screen/训练任务。
- 显式无卡A启动后，平台回读`WithoutGpuSpec=A`、CPU2、Memory4096、GPU0、Running，报价字段`InstancePrice=0.13`；不是仅靠CUDA不可用判断无卡计费。保险已回读为2026-10-10 02:44:43（UTC+8）；主准备窗口已主动停止并实际回读`Stopped / GPU0`，不是只退出任务。停机事实写入后的收尾流程为：仅通过一个短无卡A窗口同步最终Git记录，再次主动关机；最终结果以本轮结束时的平台回读为准，不预先写成已执行。
- 云端GitHub fetch因TLS握手中止；没有无限重试，改用Git bundle fetch和fast-forward，保持提交关系。实现阶段两端均到`4f3e405`，收尾还会同步文档并回读最终HEAD与tracked-clean。
- 复用`/usr/local/miniconda3/envs/py310`：Python3.10.16、torch2.1.2+cu118、mmcv2.1.0、timm1.0.28、SwanLab0.10.1。官方预训练已存在110,203,103 bytes，未重复上传或下载。4GB机器未进行大模型训练模拟。
- 本地在线init/log/finish实际上传完成16 records：[CPU准备run](https://swanlab.cn/@Newton_liub/dformer-research/runs/bwmgfa13)。首次Windows GBK控制台错误后只重试一次，改用`-X utf8`成功；未直接读取`key.txt`，SDK复用登录存储，key未打印或纳入Git。
- 云端已有登录存储存在，disabled生命周期和offline init/log/finish实际完成；离线run为`outputs/odg-preparation/monitor/swanlab/run-20261009_174038-8ugt3txd`（云端UTC命名）。云端online本轮未新测，训练日志失败仍保留CSV，不假称云端在线验收通过。
- SUN短上传测量：8MiB用5.513秒，包含连接开销；对2.452GB包保守估算26.86分钟，小于3小时，因此才开始上传。实际传输1089.322秒（18分09秒）、退出码0；云端包大小2452204576bytes，解压后计数/路径/少量可读性验收完成；没有全库重哈希或下载第二份包。
- 平台对**指定示例**4090×1、16CPU、64GiB、Postpay规格返回`Instance=1.95`，这是规格报价，不是当前无卡实例的有卡实测账单。新SUN时间未知，只用 $C=P_{\mathrm{GPU}}T_{\mathrm{GPU}}+P_{\mathrm{CPU}}T_{\mathrm{CPU}}+C_{\mathrm{storage}}$；不拿旧MUSeg时长估算。

## 验证范围偏差与恢复点

- 最初外部路径Glob/Grep调用返回了工作区文件名/匹配，而非指定外部范围；发现后没有重试扩大搜索，改用明确路径读取和限定数据入口。这些返回不作为已读取外部资产的证据。
- 首次CPU模型检查触发作者HAM中硬编码的`.cuda()`基向量分配，在CPU/CUDA设备一致性检查处终止，没有完成GPU模型前向/反向。已改为输入设备，并以`CUDA_VISIBLE_DEVICES=-1`重检成功；意外分配与修复如实保留。
- 数据整理过程曾额外执行全清单的RGB/Depth头部尺寸扫描（4清单合计15620条，包含重复图像），超出约5组抽样预算。数据只读，无全库内容哈希；已停止扩展，主代理移除`--dims`和全量尺寸函数，后续只保留路径配对及少量解码，不把超范围检查隐藏成抽样。
- 脚本首次云端语法检查发现参数提示中的单引号不配对；已在本地正常提交修复、通过Git同步后重检，不留未记录的云端补丁。
- 恢复点：实现与SUN无卡数据准备已验收，主窗口已回读Stopped；收尾只同步停机文档与清理本轮Git bundle/8MiB传输块，再回读同HEAD、tracked-clean和Stopped，不留自动训练任务。下次先确认数据包来源、用户GPU预算和平台关机保险，再用指南中两条手动命令做原始/ODG最小检查及ODG首轮30epoch；baseline和resume命令也已备齐。原始与ODG的micro-batch/累积组合必须相同，2×8仍只是候选，未验收显存。

本轮未进行GPU测试/推理/训练；等待用户授权有卡运行。
