# 新研究轮次工程配置

本轮仅做环境准备。当前作者最新仓库保持不回退，第一阶段 backbone 固定为 **DFormerv2-S**；DFormer++ 代码和配置保留，作为后续第二 backbone / generalization 候选，其官方权重仍未发布。

## 配置与入口

- 独立配置：`local_configs.research.DFormerv2_S_SUNRGBD`，复制作者 SUNRGBD DFormerv2-S 配置，不修改共享配置对象。
- 官方初始化：`checkpoints/pretrained/DFormerv2_Small_pretrained.pth`；只使用编码器 pretrained，不使用旧 MUSeg/MMFR 训练权重。
- 训练入口保持 `utils/train.py`；仅新增可选 SwanLab 调用，日志实现放 `research/tracking.py`。其他配置未设置 `swanlab_enabled` 时完全不启用 SDK。
- 默认 seed=12345、AdamW、lr=8e-5、batch_size=16、300 epochs、10 epochs warmup、作者 train-scale augmentation，全部继承作者设置。未来 baseline 和候选方法必须使用相同 pretrained、seed、schedule、augmentation 与 evaluator。
- 默认数据位置：`datasets/SUNRGBD`；可在启动前设置 `DFORMER_DATASET_ROOT`，其值是包含 `SUNRGBD` 的父目录。
- 默认实验名：`dformerv2-s-sunrgbd-baseline`；可用 `DFORMER_EXPERIMENT_NAME` 改名。
- 输出：`outputs/<experiment-name>/<timestamp>/`，含训练 checkpoint、日志、TensorBoard、SwanLab 子目录。

以下仅为**未来**正式实验的启动形式，本轮未执行，且数据与验证协议确认前不得执行：

```bash
# 在仓库根目录执行；SUNRGBD + DFormerv2 必须启用作者 padding。
PYTHONPATH=. python utils/train.py \
  --config local_configs.research.DFormerv2_S_SUNRGBD \
  --gpus 1 --no-syncbn --pad_SUNRGBD
```

作者配置 `eval_source=test.txt`。正式研究前必须固定开发验证策略，避免根据 official test 反复选择方法；本轮仅准备配置，不授权使用 test，不人为编造 val split。基线本轮未重训。

## SwanLab

使用 SwanLab 官方 `swanlab.login(api_key=..., save=True)` 持久登录方式；API key 只由用户提供的本地文件读入登录进程，不写训练代码、配置、文档或 Git。本地 `临时/key.txt` 保留并加入 `.gitignore`，由用户自行删除。训练进程读取官方登录存储，无需设置 API key。

配置默认 `swanlab_enabled=True`、`swanlab_project=dformer-research`、`swanlab_mode=online`。只在分布式 global rank 0 初始化并记录：epoch 平均训练 loss、learning rate、验证 mIoU（仅发生评价的 epoch）、epoch、累计计划 update、实验名称与核心 config / CLI 参数。update 使用作者迭代计数，AMP 跳过参数更新时不代表成功 optimizer step 数。learning rate 是作者循环在最后一批后写入优化器的 schedule 值。保留 TensorBoard；不新增复杂可视化。

官方参考：
- https://docs.swanlab.cn/api/py-login.html
- https://docs.swanlab.cn/api/py-init.html

## 边界

旧 checkpoints 保留为历史参考，不作为新论文主 baseline。未下载 SUNRGBD、DeLiVER、NYUv2；未设计新模块；未启动训练或评价；只做模型构建、权重加载、环境和登录的定点检查，不运行完整测试套件。
