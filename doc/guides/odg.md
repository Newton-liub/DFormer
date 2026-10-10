# ODG 研究入口与固定协议

ODG（局部观测分布几何先验）是本项目待验证设计，没有已取得的分割增益。依据为用户执行的[上级指令](../plans/2026-10-10-odg-no-gpu-instructions.md)。本轮仅 CPU 实现验收与无卡准备；实时结果见[项目状态](../state/current.md)。

## 方法与输入

- `research/geometry.py` 在同步增强后的输入深度上构造 16 区间软直方图，再按实际 stage 网格面积平均、支持量归一化。区间中心等距覆盖 $[-0.48/0.28,\,0.52/0.28]$，对应作者 $d=(D_8/255-0.48)/0.28$，不解释为米制距离。
- 支持域只排除 padding、非有限值和显式人工删除。当前 SUN 包没有已核实的天然无效值约定，灰度 0 保留为观测；归一化 tensor 的 0 也不用于推断缺失。
- 共享核关系为：

$$
K_h[a,b]=\exp(\ell_h|c_a-c_b|),\qquad
s_{ij}^{h}=\frac{p_i^{\mathsf T}K_hp_j}{\sqrt{(p_i^{\mathsf T}K_hp_i)(p_j^{\mathsf T}K_hp_j)}}.
$$

$$
b^d_{ij,h}=w_d\log\operatorname{clamp}(s_{ij}^{h},10^{-6},1).
$$

- 核、归一化及 log 在 FP32 中计算。轴向/full attention 相加前匹配实际 logits dtype；`original` 保留作者原有数值路径。空观测涉及的 depth bias 直接为 0，空间项保留。每个原 block 的 $w_d$（包括负值）、空间权重、Q/K/V、RoPE、FFN、HAM 均保留，没有新增可学习参数。
- 前三 stage 为轴向关系，最后为 full；同一次 forward 内共享同 stage 的未乘 $w_d$ 的关系。按 group/query 分块小核乘法，无高分辨率全二维关系或位置对×区间对张量，不跨 batch 缓存。
- `geometry_mode=original` 为作者双线性 stage 深度差；`mean` 在相同支持域先求均值，再使用同样软分箱核；`odg` 保留分布。区间中心上的点质量可退化为原深度差；任意 off-grid 深度只是近似，不能声称严格等价。

## 数据与训练合同

- SUN 主配置：`local_configs.research.ODG_SUNRGBD`。DFormerv2-S / HAM 宽度 1024 / 37 类 / 480×480。
- 从官方 train 5285 行按 `random.Random(12345)` 固定抽 528 图作 dev，其余 4757 图训练；保留原行顺序。清单在 `research/splits/sunrgbd_seed12345/`。这是图像级划分，不声称场景隔离；原 `train.txt` / `test.txt` 不改。
- 默认周期评价每 10 epoch 只读 dev，单尺度、无 flip。显式 `--fulltrain` 用全 train 并关闭周期评价；官方 test 仅经独立评价入口显式选取，不用于反复挑 checkpoint。
- AdamW：lr8e-5、wd0.01、betas0.9/0.999；有效 batch16、300 epoch、warmup10、poly0.9。继承作者尺度/翻转，不新增损坏训练。单 GPU 使用普通 BN、`norm_eval=False`，全部可训练层正常更新；不 compile。
- `--micro-batch × --accum-steps = 16`。开发集每 epoch 298 次 optimizer 尝试，补齐 11 张样本到 4768；完整日程 89400 次，warmup2980次。全 train 则每 epoch331次。AMP 实际成功与跳过更新分别计数。
- `--stop-after-epoch 30/100` 只是暂停，不压缩 300 epoch 日程。LR 在 optimizer 更新之前设置，记录实际 param-group LR。只有 encoder pretrained 初始化；新分割头、新 optimizer/scaler。不用旧 MUSeg 权重。
- `last.pth`、`best-dev.pth`、30/100阶段点包含模型/optimizer/scaler/RNG/计数与训练合同；resume 仅限同实验 epoch 边界，不能从 smoke 的半 epoch 快照续正式实验。
- 产物为 resolved 配置、参数、loss/LR CSV、验证总/逐类结果及少量预测，位于 `outputs/<experiment>/<run-id>/`，不进 Git。
- 五尺度+flip 只由 `research/evaluate_odg.py --msf` 显式执行，尺度0.5/0.75/1/1.25/1.5，共同累加每视图 softmax 概率。人工连通孔洞25%/50%（矩形主体加可选末行，按真实图像支持域精确取整计数；非矩形支持域先拒绝并要求单独协议）同时改变输入深度和显式支持 mask，所有模型同输入/seed，不能称天然故障。
- NYU 第二入口 `local_configs.research.ODG_NYUv2` 保留作者40类配置；数据与开发划分未就绪不阻塞 SUN。DeLiVER 本轮不适配。

## CPU 准备命令

在仓库根目录执行；Windows 使用 `D:\2Env\anaconda\envs\dformer\python.exe -X utf8`，并将 `CUDA_VISIBLE_DEVICES=-1`。本地数据父目录是 `D:\0Project\dataset`。

```bash
python -m research.data --data-root /path/to/dataset --check-pairs
python -m research.prepare_odg --data-root /path/to/dataset
python -m research.train_odg --config local_configs.research.ODG_SUNRGBD \
  --micro-batch 2 --accum-steps 8 --stop-after-epoch 30 --print-schedule
```

路径配对只检查存在性；真实解码/分布示例限少量样本。未授权完整测试或全量推理；训练入口在 CPU 上拒绝运行。

## 后续 GPU 命令：仅准备，未执行

前置：用户另行授权 GPU；核实数据包来源与云端就绪；有卡启动后确认实际规格、设置并回读平台计划关机；确认 `MICRO_BATCH/ACCUM_STEPS` 并让两模型一致。下面的2×8只是有效 batch16的候选组合，**尚无显存/吞吐验收**。脚本不会启动实例或创建自动训练任务。

```bash
export DFORMER_DATASET_ROOT=/root/rivermind-data/dataset
export MICRO_BATCH=2 ACCUM_STEPS=8
# 仅在真实授权与平台关机保险回读成功之后设置：
export ODG_GPU_AUTHORIZED=1 ODG_PLATFORM_STOP_CONFIRMED=1

# 命令1：原始与ODG各200次真实optimizer尝试，检查数值、更新和吞吐。
bash research/run_odg_gpu.sh smoke
# 命令2：首轮ODG到30epoch，日程仍为300epoch。
bash research/run_odg_gpu.sh odg

# 匹配baseline，随后同实验从epoch边界继续到100。
bash research/run_odg_gpu.sh baseline
GEOMETRY_MODE=odg bash research/run_odg_gpu.sh resume \
  outputs/sun-dev-odg-seed12345/<run-id>/last.pth
```

真实训练时间需授权后由正常前100–200步估算；费用用当前规格价格×实际有卡时长计算，并单列存储/无卡准备。历史 MUSeg 用时不作为 SUN 估时依据。
