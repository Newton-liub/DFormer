import {
  BarChart,
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Code,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const projectRoot = "d:\\0Project\\DFormer";
const reportPath = `${projectRoot}\\doc\\reports\\2026-08-31-museg-dformerv2-quick-baseline-comprehensive.md`;
const statusPath = `${projectRoot}\\doc\\main\\MUSeg-current-status.md`;
const mainEvaluationPath = `${projectRoot}\\doc\\reports\\2026-08-31-museg-quick-b0-main-evaluation.md`;
const lessonsPath = `${projectRoot}\\doc\\reports\\2026-08-31-museg-stage05-seed1-training-lessons.md`;
const futurePath = `${projectRoot}\\liu-test-exp\\方案1\\研究方案设计专用提示词.md`;
const adjudicationPath = `${projectRoot}\\cloud\\DFormer-quick-b0-evidence\\museg-dformerv2-s-rgb-quick-b0-v1\\posteval\\main-evaluation-adjudication.json`;

const candidates = ["epoch 420", "epoch 440", "epoch 480", "epoch 500"];
const selectorCandidates = ["epoch 420", "epoch 440", "epoch 480"];

const perClassCategories = [
  "container",
  "support equipment",
  "mining equipment",
  "rescue equipment",
  "metal fixture",
  "electronic equipment",
  "person",
  "electrical equipment",
  "cable",
  "anchoring equipment",
  "tube",
  "tools & materials",
  "indicator",
  "rail area",
  "door",
];

const perClassIoU = [
  24.67, 40.23, 41.92, 43.56, 53.65, 56.31, 58.54, 59.08, 61.41, 65.46,
  67.71, 69.91, 73.26, 80.1, 86.01,
];

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();
  const openFile = (path: string) => dispatch({ type: "openFile", path });

  return (
    <Stack gap={28} style={{ padding: 24, maxWidth: 1280, margin: "0 auto" }}>
      <Stack gap={12}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.11 · 组会总体汇报</Pill>
          <Text size="small" tone="tertiary">
            工作周期：2026-08-17 至 2026-08-31
          </Text>
        </Row>
        <H1>MUSeg × DFormerv2-S：从作者公开版本到可复核 Quick-B0</H1>
        <Text tone="secondary">
          本阶段已经完成作者当前公开 DFormerv2 主线、官方预训练权重与 MUSeg 15 类 RGB-D 任务之间的适配，完成一次
          500 epoch RGB 训练、4 个候选的五尺度翻转主评估和云资源收口。最终 B0 是 epoch 420。
        </Text>
        <Callout tone="success" title="当前可以直接用于后续同口径模块比较">
          最终 checkpoint 的主评估为 mIoU <Code>58.79%</Code>、mAcc <Code>69.91%</Code>、mF1 <Code>72.73%</Code>。
          结果绑定 318 张 val-dev、冻结 protocol、训练提交与 checkpoint SHA-256；official test 继续封存未读。
        </Callout>
        <Callout tone="warning" title="结论边界">
          这是 single-seed development B0，不是三 seed 统计或 official-test 结果。历史 BGR、resize、sliding 和当前 RGB
          五尺度评估属于不同协议，跨协议差值不能归因于某一个变量。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => openFile(reportPath)}>打开完整总报告</Button>
          <Button variant="secondary" onClick={() => openFile(mainEvaluationPath)}>打开主评估报告</Button>
          <Button variant="secondary" onClick={() => openFile(statusPath)}>打开唯一当前状态</Button>
          <Button variant="secondary" onClick={() => openFile(adjudicationPath)}>打开结构化裁决</Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="58.79%" label="主 mIoU · epoch 420" tone="success" />
        <Stat value="69.91%" label="主 mAcc" tone="info" />
        <Stat value="72.73%" label="主 mF1" tone="info" />
        <Stat value="318" label="冻结 val-dev 样本" tone="neutral" />
      </Grid>

      <Divider />

      <Stack gap={14}>
        <H2>1. 复现与适配链路</H2>
        <Text tone="secondary">
          本项目复用作者公开模型与方法，并对 MUSeg 的数据、输入、训练、checkpoint 选择、评估几何和云生命周期做显式适配。
          “复现”不表示在 NYUv2/SUNRGBD 上重跑论文表格，也不表示复现 MUSeg 作者未公开的测试代码。
        </Text>
        <Grid columns="1fr 1fr 1fr" gap={16} align="start">
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="success">已核验</Pill>}>作者公开基础</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Text weight="semibold">DFormerv2 当前公开 main</Text>
                <Code>814799bb…b6aa</Code>
                <Text size="small" tone="secondary">2025-11-11；2026-08-31 通过只读 ls-remote 复核。本项目分支包含该提交。</Text>
                <Divider />
                <Text weight="semibold">官方 DFormerv2-S pretrained</Text>
                <Text size="small" tone="secondary">110,203,103 bytes</Text>
                <Code>19116988…11a6</Code>
                <Text size="small" tone="tertiary">官方 Hugging Face 资产与本地权重 SHA-256 完全一致。</Text>
              </Stack>
            </CardBody>
          </Card>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="info">MUSeg 适配</Pill>}>输入与训练</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Text><Code>BGR→RGB</Code>，使用 RGB 顺序 ImageNet mean/std</Text>
                <Text>随机尺度 <Code>0.5–1.75</Code>，同步变换 RGB/Depth/Label</Text>
                <Text>训练裁剪为高 <Code>480</Code>、宽 <Code>640</Code></Text>
                <Text>AdamW，500 epoch，batch 10，seed <Code>772961337</Code></Text>
                <Text size="small" tone="tertiary">480×640 是训练窗口，不是统一验证尺寸。</Text>
              </Stack>
            </CardBody>
          </Card>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="success">已完成</Pill>}>冻结评估与证据</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Text>五尺度 × 原图/翻转，共 10 个 view</Text>
                <Text>FP32 logits 平均，恢复原始 Label 网格计分</Text>
                <Text>top 3 selector + latest，训练后统一重排</Text>
                <Text>checkpoint、split、protocol、commit 全部绑定哈希</Text>
                <Text size="small" tone="tertiary">official_test_included=false。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Grid columns="1.1fr 0.9fr" gap={18} align="start">
        <Stack gap={10}>
          <H3>数据职责</H3>
          <BarChart
            categories={["train-dev", "val-dev", "official test"]}
            series={[{ name: "样本数", data: [1277, 318, 1576], tone: "info" }]}
            height={250}
            valueSuffix=" 张"
            showValues
          />
          <Text size="small" tone="tertiary">横轴：数据职责 · 纵轴：样本数（张）</Text>
          <Text size="small" tone="quaternary">来源：冻结 MUSeg split；official test 在开发期不参与模型或阈值选择。</Text>
        </Stack>
        <Callout tone="info" title="为什么强调原始 Label 网格">
          训练 crop 只控制训练张量大小。主评估先按五个尺度前向，再把每份 logits 恢复到每张标签原来的高宽后融合和计分，
          因此指标不会被简单描述成“在 480×640 上测得”。
        </Callout>
      </Grid>

      <Divider />

      <Stack gap={14}>
        <H2>2. 最终模型由主 evaluator 决定</H2>
        <Grid columns="1fr 1fr" gap={18} align="start">
          <Stack gap={9}>
            <H3>训练期 selector 与主 mIoU 排名变化</H3>
            <LineChart
              categories={selectorCandidates}
              series={[
                { name: "训练期 selector mIoU", data: [56.39, 56.1, 56.87], tone: "neutral" },
                { name: "五尺度翻转主 mIoU", data: [58.79, 58.73, 58.43], tone: "success" },
              ]}
              height={285}
              yMin={55.5}
              yMax={59.2}
              valueSuffix="%"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：具有独立 selector 分数的候选 epoch · 纵轴：val-dev mIoU（%）</Text>
            <Text size="small" tone="quaternary">来源：checkpoint-candidates.json 与 2026-08-31 主评估；epoch 500 为 latest，没有独立 selector mIoU，故不放入本图。</Text>
          </Stack>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="warning">排名反转</Pill>}>为什么不能只保留单尺度第一名</CardHeader>
            <CardBody>
              <Stack gap={10}>
                <Row justify="space-between"><Text tone="secondary">selector 第一</Text><Text weight="semibold">epoch 480 · 56.87%</Text></Row>
                <Row justify="space-between"><Text tone="secondary">主评估第一</Text><Text weight="semibold">epoch 420 · 58.79%</Text></Row>
                <Divider />
                <Text tone="secondary">
                  epoch 480 在主评估中降到第三；epoch 420 从 selector 第二升到主评估第一。这直接验证了“低成本筛选候选，
                  再由冻结主 evaluator 统一重排”的必要性。
                </Text>
                <Callout tone="info" title="后续 v2">
                  后续独立协议保留 top 8 + latest，最多 9 个去重候选。它降低漏选风险，但不保证排序差异完全消失，也不回写本轮 v1。
                </Callout>
              </Stack>
            </CardBody>
          </Card>
        </Grid>

        <Stack gap={9}>
          <H3>4 个候选的三项主指标</H3>
          <BarChart
            categories={candidates}
            series={[
              { name: "mIoU", data: [58.79, 58.73, 58.43, 57.68], tone: "success" },
              { name: "mAcc", data: [69.91, 69.54, 69.34, 68.84], tone: "info" },
              { name: "mF1", data: [72.73, 72.67, 72.44, 71.81], tone: "neutral" },
            ]}
            height={320}
            yMin={55}
            yMax={75}
            valueSuffix="%"
            showValues
          />
          <Text size="small" tone="tertiary">横轴：预登记 checkpoint · 纵轴：318 张 val-dev 的指标（%） · 图例：mIoU、mAcc、mF1</Text>
          <Text size="small" tone="quaternary">来源：msflip-whole-original-grid-v1 四份 completed JSON；RGB、FP32、TF32 disabled、原始 Label 网格。</Text>
        </Stack>
      </Stack>

      <Callout tone="success" title="最终身份">
        epoch 420 的 <Code>selector-epoch-420.pth</Code>，大小 <Code>321,011,270 bytes</Code>，SHA-256
        <Code>f246a3af…e3a1c</Code>。epoch 420 比 epoch 440 的主 mIoU 高 0.06 个百分点，未触发同分规则。
      </Callout>

      <Divider />

      <Stack gap={14}>
        <H2>3. 15 类结果：总体分数之外还要看薄弱类别</H2>
        <BarChart
          categories={perClassCategories}
          series={[{ name: "epoch 420 类别 IoU", data: perClassIoU, tone: "info" }]}
          height={560}
          horizontal
          yMin={0}
          yMax={100}
          valueSuffix="%"
          showValues
          referenceLines={[{ value: 58.79, label: "总体 mIoU 58.79%", tone: "success" }]}
        />
        <Text size="small" tone="tertiary">纵轴：MUSeg 15 类，按 IoU 从低到高排序 · 横轴：类别 IoU（%）</Text>
        <Text size="small" tone="quaternary">来源：epoch-420-msflip-whole-original-grid-v1.json，318 张 val-dev；绿色参考线为 15 类平均 mIoU。</Text>
        <Grid columns="1fr 1fr" gap={16} align="start">
          <Callout tone="warning" title="当前最薄弱类别">
            container IoU 为 <Code>24.67%</Code>；support、mining、rescue equipment 分别为 <Code>40.23%</Code>、
            <Code>41.92%</Code>、<Code>43.56%</Code>。后续方法不能只报总体 mIoU。
          </Callout>
          <Callout tone="success" title="当前较强类别">
            door IoU 为 <Code>86.01%</Code>，rail area 为 <Code>80.10%</Code>，indicator 为 <Code>73.26%</Code>。
            15 类 IoU/accuracy/F1 全部有限，没有 NaN 或空类归约。
          </Callout>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>4. 历史对比只用于说明“协议会改变数字”</H2>
        <Grid columns="1fr 1fr" gap={18} align="start">
          <Stack gap={9}>
            <H3>固定历史 checkpoint 的推理几何诊断</H3>
            <BarChart
              categories={["original-full", "resize-480x640", "sliding-480x640"]}
              series={[{ name: "历史 best mIoU", data: [52.98, 56.31, 51.89], tone: "info" }]}
              height={285}
              yMin={50}
              yMax={58}
              valueSuffix="%"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：历史 evaluator geometry · 纵轴：原始 Label 网格 mIoU（%）</Text>
            <Text size="small" tone="quaternary">来源：legacy BGR epoch-460 best 的三份 post-evaluation v2 JSON，318 张 val-dev。</Text>
            <Callout tone="warning" title="不能追认最高数字为主协议">
              三种 geometry 跨度为 4.42 个百分点。resize 改变长宽比；它分数最高不等于研究口径天然更正确。
            </Callout>
          </Stack>

          <Stack gap={9}>
            <H3>固定历史 checkpoint 的颜色契约诊断</H3>
            <BarChart
              categories={["legacy BGR", "RGB + RGB mean/std", "BGR + 反向 mean/std"]}
              series={[{ name: "历史 checkpoint mIoU", data: [52.98, 33.85, 49.53], tone: "warning" }]}
              height={285}
              yMin={30}
              yMax={55}
              valueSuffix="%"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：推理输入契约 · 纵轴：original-full mIoU（%）</Text>
            <Text size="small" tone="quaternary">来源：同一 legacy BGR checkpoint、同一 split 的三臂诊断；不是 RGB/BGR 成对重训。</Text>
            <Callout tone="warning" title="能证明敏感，不能证明重训胜负">
              旧模型已经适应 BGR，直接换 RGB 会破坏输入分布。本图只说明颜色契约必须冻结，不能用来断言重新训练后的 RGB 一定优于 BGR。
            </Callout>
          </Stack>
        </Grid>

        <Grid columns="1fr 1fr" gap={16} align="start">
          <Card>
            <CardHeader>历史 Stage-05 seed1</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text>legacy BGR</Text>
                <Text>固定训练尺度 <Code>[1.0]</Code></Text>
                <Text>训练期单尺度 original-full 选择</Text>
                <Text>旧遥测未直接区分 attempted/completed/skipped</Text>
                <Text size="small" tone="tertiary">用于经验复盘，不是当前正式 B0。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader>当前 Quick-B0</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text>与 pretrained 对齐的 RGB</Text>
                <Text>六个随机训练尺度 <Code>0.5–1.75</Code></Text>
                <Text>top 3 + latest，再用五尺度翻转重排</Text>
                <Text>attempted/completed/skipped 为 <Code>64000/63971/29</Code></Text>
                <Text size="small" tone="tertiary">多项协议同时变化，不能做单变量归因。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>5. 问题如何被发现并解决</H2>
        <Grid columns="1fr 1fr" gap={16} align="start">
          <Stack gap={14}>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">输入</Pill><H3>RGB / BGR 语义不一致</H3></Row>
              <Text tone="secondary">追溯官方权重和预训练代码，冻结 OpenCV BGR→RGB 与 RGB ImageNet normalization。结论是输入一致，不是颜色性能胜负。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">几何</Pill><H3>训练 crop 被误解为验证尺寸</H3></Row>
              <Text tone="secondary">分开记录训练裁剪、模型前向尺寸和指标网格；主评估统一恢复原始 Label 网格。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">数值</Pill><H3>A1 空有效像素 loss</H3></Row>
              <Text tone="secondary">旧逻辑对空集合求 mean 会产生非有限值；B1 safe_masked_mean 在空集合返回连接计算图的有限零值。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">启动</Pill><H3>PYTHONPATH 未导出</H3></Row>
              <Text tone="secondary">训练和本地评估首次命令都在模型导入前退出。保留现场，只修正环境后按同一协议完整重跑。</Text>
            </Stack>
          </Stack>

          <Stack gap={14}>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">计费</Pill><H3>结束不等于平台 Stopped</H3></Row>
              <Text tone="secondary">新增无卡 lifecycle-test、证据取回、控制面 stop、状态等待与 schedule 兜底。研究验收不再控制是否停止计费。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">遥测</Pill><H3>历史 update 语义不清</H3></Row>
              <Text tone="secondary">当前直接记录 attempted、completed、skipped，并验证等式，避免训练后从理论值反推 AMP skip。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="warning">选择</Pill><H3>final 或 selector 第一未必最终最好</H3></Row>
              <Text tone="secondary">保留 top-k 与 latest，再由冻结主 evaluator 统一排序。本轮 epoch 420/480 排名变化提供了直接证据。</Text>
            </Stack>
            <Stack gap={5} style={{ padding: "4px 2px" }}>
              <Row align="center" gap={8}><Pill size="sm" tone="success">结果</Pill><H3>完整证据已取回</H3></Row>
              <Text tone="secondary">checkpoint、protocol、日志、失败现场和裁决均已下载并重算 SHA-256；实例最终确认为 Stopped。</Text>
            </Stack>
          </Stack>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>6. 时间、资源与费用</H2>
        <Grid columns="1fr 1fr" gap={18} align="start">
          <Stack gap={9}>
            <H3>实际耗时</H3>
            <BarChart
              categories={["云端正式训练", "本地 4 候选主评估"]}
              series={[{ name: "阶段耗时（训练=墙钟；评估=evaluator 内部）", data: [13.628, 1.567], tone: "info" }]}
              height={250}
              valueSuffix=" 小时"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：执行阶段 · 纵轴：耗时（小时）</Text>
            <Text size="small" tone="quaternary">来源：正式启动/训练终态时间与四份 evaluator 内部计时；评估在本地 RTX 5060 Laptop 完成。</Text>
          </Stack>

          <Stack gap={9}>
            <H3>参考费用估算与预算</H3>
            <BarChart
              categories={["训练时段估算", "启动至 Stopped 估算", "计划下限", "计划上限", "32h 兜底"]}
              series={[{ name: "按 1.98 元/小时", data: [26.98, 27.55, 47.52, 55.44, 63.36], tone: "warning" }]}
              height={275}
              valueSuffix=" 元"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：估算或预算口径 · 纵轴：人民币（元）</Text>
            <Text size="small" tone="quaternary">来源：历史参考价 1.98 元/小时与已核验时间。前两项是计算估算，后三项是训练前预算，不是控制台账单。</Text>
          </Stack>
        </Grid>
        <Callout tone="warning" title="实际总花费尚未核实">
          当前没有最终订单、折扣、云盘单价或镜像费用证据。实例在正式训练前还执行了 schedule、preflight 和启动检查；
          Stopped 后计算资源停止计费，但云盘与镜像可能继续计费。组会中应称“正式启动至停止时段的参考估算约 27.55 元”，不能称“实际账单 27.55 元”。
        </Callout>
        <Grid columns={4} gap={14}>
          <Stat value="13h 37m" label="正式训练" tone="neutral" />
          <Stat value="94.044m" label="本地主评估" tone="neutral" />
          <Stat value="RTX 4090" label="云端训练设备" tone="info" />
          <Stat value="0 OOM" label="4 个主评估" tone="success" />
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>7. A1/B1 与 A2/B2 是两条不同问题链</H2>
        <Grid columns="1fr 1fr" gap={18} align="start">
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="success">已进入 B0</Pill>}>A1 → B1：损失数值稳定性</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <H3>A1 已证实</H3>
                <Text tone="secondary">全背景 crop/rank 可能没有有效标签像素，旧 masked mean 对空集合求均值会产生非有限 loss。</Text>
                <Divider />
                <H3>B1 已完成</H3>
                <Text tone="secondary"><Code>safe_masked_mean</Code> 在空集合返回连接计算图的有限零值，非空集合保持原均值。</Text>
                <Callout tone="info" title="它不是性能模块">
                  没有关闭 B1 的成对训练对照，不能把稳定完成或 mIoU 提升归因于 B1。
                </Callout>
              </Stack>
            </CardBody>
          </Card>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" tone="warning">尚未授权</Pill>}>A2 → B2：Depth validity</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <H3>A2 需在当前 B0 上重做</H3>
                <Text tone="secondary">历史 epoch-10 checkpoint 上的 16 张样本 pilot，其 q=0.3、q=0.5 下降仅 0.1107、0.3337 个百分点，不能替代当前 RGB B0 正式筛查。</Text>
                <Divider />
                <H3>B2 尚未实现</H3>
                <Text tone="secondary">目标是在任一 depth patch 无效时移除 pair 的 depth decay，同时保留 positional decay；all-valid 应等价 B0。</Text>
                <Callout tone="warning" title="进入条件">
                  只有正式 <Code>A2-pass + 用户明确授权</Code>，才开始 B2 数学、shape、金标准、zero-train 和后续独立训练。
                </Callout>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Stack gap={12}>
        <H3>B2 的下一步顺序</H3>
        <Grid columns="1fr 1fr 1fr 1fr" gap={12} align="start">
          <Stack gap={5}><Pill size="sm" active>1</Pill><Text weight="semibold">当前 B0 正式 A2</Text><Text size="small" tone="secondary">val-dev、冻结 corruption 和 evaluator。</Text></Stack>
          <Stack gap={5}><Pill size="sm" active>2</Pill><Text weight="semibold">冻结 B2 规格</Text><Text size="small" tone="secondary">validity 来源、shape、广播和 all-valid 等价。</Text></Stack>
          <Stack gap={5}><Pill size="sm" active>3</Pill><Text weight="semibold">B2-zero-train</Text><Text size="small" tone="secondary">同一 checkpoint，不更新权重，先查即时作用。</Text></Stack>
          <Stack gap={5}><Pill size="sm" active>4</Pill><Text weight="semibold">对称独立训练</Text><Text size="small" tone="secondary">再次授权后，同 pretrained/seed/预算比较 B0 与 B2。</Text></Stack>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>8. 下一篇方向：冻结 DFormerv2 输出的后验校准</H2>
        <Text tone="secondary">
          “方向1”研究模型置信度是否可信，而不是继续修改融合骨干。当前 single-seed B0 可以支持工具开发和初步探索；
          重要论文结论仍需要成对重复或额外 seed。
        </Text>
        <Grid columns="1fr 1fr 1fr" gap={16} align="start">
          <Stack gap={7} style={{ padding: "4px 2px" }}>
            <Pill size="sm" tone="warning">前置协议</Pill>
            <H3>独立 calibration 集</H3>
            <Text tone="secondary">与 train-dev、val-dev、official test 职责隔离。当前 val-dev 已用于 checkpoint 选择，不能直接改名。</Text>
          </Stack>
          <Stack gap={7} style={{ padding: "4px 2px" }}>
            <Pill size="sm" tone="info">共同输入</Pill>
            <H3>同一份 float32 logits</H3>
            <Text tone="secondary">绑定 checkpoint、输入清单、RGB 契约、ignore mask、配置和哈希，所有方法读取同一份输出。</Text>
          </Stack>
          <Stack gap={7} style={{ padding: "4px 2px" }}>
            <Pill size="sm" tone="success">三个问题</Pill>
            <H3>概率、排序、阈值迁移</H3>
            <Text tone="secondary">ECE/NLL/Brier 看概率值；风险—覆盖与 AURC 看错误排序；独立数据上的风险与覆盖看阈值迁移。</Text>
          </Stack>
        </Grid>
        <Callout tone="info" title="旧方案需要更新的地方">
          旧提示词把“三 seed baseline 尚未完成”写成校准开发的硬前置，这一状态已经过时。single-seed B0 足以启动探索；
          独立 calibration 集、数据隔离、同一份 float32 logits 和 official test 封存仍是有效硬边界。
        </Callout>
        <Callout tone="warning" title="A2/B2 与后验校准并非同一条前置链">
          A2/B2 研究 Depth 无效时的模型机制；后验校准研究冻结模型的概率可信度。两条路线可以共享 B0 和退化分析，但 B2 不是方向1启动的强制前置条件。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="secondary" onClick={() => openFile(futurePath)}>打开方向1设计提示词</Button>
          <Button variant="secondary" onClick={() => openFile(lessonsPath)}>打开历史训练经验</Button>
        </Row>
      </Stack>

      <Divider />

      <Stack gap={12}>
        <H2>9. 组会讲述建议</H2>
        <Grid columns="1fr 1fr" gap={14} align="start">
          <Stack gap={8}>
            <Text weight="semibold">1. 先讲共同起点</Text>
            <Text tone="secondary">目标不是追一个孤立高分，而是建立能公平复用、身份可追溯的 B0。</Text>
            <Text weight="semibold">2. 再讲输入与几何</Text>
            <Text tone="secondary">RGB 为什么要和 pretrained 对齐；480×640 为什么只是训练窗口。</Text>
            <Text weight="semibold">3. 展示候选排名反转</Text>
            <Text tone="secondary">用 epoch 480 与 420 说明最终 evaluator 的必要性。</Text>
          </Stack>
          <Stack gap={8}>
            <Text weight="semibold">4. 展示逐类薄弱点</Text>
            <Text tone="secondary">container 等类别说明总体 mIoU 不是全部。</Text>
            <Text weight="semibold">5. 讲工程问题的解决</Text>
            <Text tone="secondary">A1/B1、PYTHONPATH、checkpoint 管理和自动关机共同保证结果可信。</Text>
            <Text weight="semibold">6. 最后讲下一步</Text>
            <Text tone="secondary">B2 与方向1都是候选计划，尚未授权、尚未执行。</Text>
          </Stack>
        </Grid>
      </Stack>

      <Callout tone="success" title="本阶段已收口">
        Quick-B0 四项任务均完成：本地实现与定点检查、无卡自动关机门禁、一次 B0 训练、主评估与证据收口。
        下一阶段需另立协议和授权，不自动启动训练、云资源或 official test。
      </Callout>

      <Row gap={8} wrap>
        <Button variant="primary" onClick={() => openFile(reportPath)}>阅读 421 行完整报告</Button>
        <Button variant="secondary" onClick={() => openFile(statusPath)}>核对当前事实与恢复点</Button>
        <Button variant="secondary" onClick={() => openFile(adjudicationPath)}>核对主评估裁决 JSON</Button>
      </Row>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        事实源：作者仓库 main、官方 pretrained 资产、冻结 protocol、train-dev/val-dev 清单、Quick-B0 训练证据、
        4 份主评估 JSON、历史 Stage-05 后评估与当前 MUSeg 唯一状态入口。Canvas 只重排已核实事实，不修改实验结论。
      </Text>
    </Stack>
  );
}

export default App;
