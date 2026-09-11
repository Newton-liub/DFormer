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
  PieChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const projectRoot = "d:\\0Project\\DFormer";
const reportPath = `${projectRoot}\\doc\\reports\\2026-09-10-museg-dformerv2-to-mve-group-meeting.md`;
const statusPath = `${projectRoot}\\doc\\main\\MUSeg-current-status.md`;
const decisionsPath = `${projectRoot}\\doc\\main\\MUSeg-open-decisions.md`;
const baselinePath = `${projectRoot}\\doc\\reports\\2026-08-31-museg-dformerv2-quick-baseline-comprehensive.md`;
const dvcPlanPath = `${projectRoot}\\doc\\plans\\2026-09-MUSeg-几何可信RGBD双路径MVE\\03-共享协议与DVC-A1问题验证.md`;
const dvgPlanPath = `${projectRoot}\\doc\\plans\\2026-09-MUSeg-几何可信RGBD双路径MVE\\04-DVG-B1条件式Oracle门控.md`;
const literaturePath = `${projectRoot}\\liu-test-exp\\方案1\\DVG-B1-A1-B1摘要筛选与全文优先级.md`;

const candidateEpochs = ["epoch 420", "epoch 440", "epoch 480", "epoch 500"];
const conditionNames = ["clean", "boundary-q25", "boundary-q50", "boundary-q75", "nonboundary-q50"];

function PhaseStep({
  index,
  title,
  status,
  summary,
  detail,
  tone = "info",
}: {
  index: string;
  title: string;
  status: string;
  summary: string;
  detail: string;
  tone?: "success" | "warning" | "info";
}) {
  const theme = useHostTheme();
  const toneColor = tone === "success" ? theme.palette.green : tone === "warning" ? theme.palette.yellow : theme.accent.primary;

  return (
    <Grid columns="52px 1fr" gap={12} align="start">
      <div
        style={{
          width: 38,
          height: 38,
          borderRadius: 20,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: theme.fill.secondary,
          border: `1px solid ${theme.stroke.secondary}`,
          color: toneColor,
          fontWeight: 700,
          fontSize: 12,
        }}
      >
        {index}
      </div>
      <Stack gap={5} style={{ paddingBottom: 14, borderBottom: `1px solid ${theme.stroke.tertiary}` }}>
        <Row align="center" gap={8} wrap>
          <Text weight="semibold">{title}</Text>
          <Pill size="sm" active={tone === "success"}>{status}</Pill>
        </Row>
        <Text tone="secondary">{summary}</Text>
        <Text size="small" tone="tertiary">{detail}</Text>
      </Stack>
    </Grid>
  );
}

function EffectInterval({
  title,
  point,
  low,
  high,
  note,
}: {
  title: string;
  point: number;
  low: number;
  high: number;
  note: string;
}) {
  const theme = useHostTheme();
  const min = -0.35;
  const max = 0.35;
  const position = (value: number) => `${((value - min) / (max - min)) * 100}%`;

  return (
    <Stack gap={7}>
      <Row align="center" justify="space-between" gap={12} wrap>
        <Text weight="semibold">{title}</Text>
        <Code>{point >= 0 ? "+" : ""}{point.toFixed(4)} pp</Code>
      </Row>
      <div style={{ position: "relative", height: 50, margin: "2px 4px 0" }}>
        <div
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            top: 20,
            height: 1,
            background: theme.stroke.secondary,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: position(0),
            top: 6,
            bottom: 12,
            width: 1,
            background: theme.text.tertiary,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: position(low),
            width: `calc(${position(high)} - ${position(low)})`,
            top: 18,
            height: 5,
            borderRadius: 3,
            background: theme.fill.primary,
            border: `1px solid ${theme.accent.primary}`,
          }}
        />
        <div
          style={{
            position: "absolute",
            left: position(point),
            top: 13,
            width: 14,
            height: 14,
            borderRadius: 8,
            transform: "translateX(-7px)",
            background: theme.accent.control,
            border: `2px solid ${theme.bg.editor}`,
          }}
        />
        <Text size="small" tone="quaternary" style={{ position: "absolute", left: 0, bottom: 0 }}>−0.35</Text>
        <Text size="small" tone="quaternary" style={{ position: "absolute", left: "50%", bottom: 0, transform: "translateX(-50%)" }}>0</Text>
        <Text size="small" tone="quaternary" style={{ position: "absolute", right: 0, bottom: 0 }}>+0.35</Text>
      </div>
      <Text size="small" tone="tertiary">
        95% 区间 [{low >= 0 ? "+" : ""}{low.toFixed(4)}, {high >= 0 ? "+" : ""}{high.toFixed(4)}] pp。{note}
      </Text>
    </Stack>
  );
}

function GateShape({ label, shape, formula }: { label: string; shape: string; formula: string }) {
  const theme = useHostTheme();
  return (
    <Stack gap={5} style={{ padding: 10, background: theme.fill.tertiary, borderRadius: 6 }}>
      <Text weight="semibold">{label}</Text>
      <Code>{shape}</Code>
      <Text size="small" tone="secondary">{formula}</Text>
    </Stack>
  );
}

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();
  const openFile = (path: string) => dispatch({ type: "openFile", path });

  return (
    <Stack gap={30} style={{ padding: 24, maxWidth: 1320, margin: "0 auto" }}>
      <Stack gap={12}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.13 · 组会长报告</Pill>
          <Text size="small" tone="tertiary">汇报周期：2026-08-17 至 2026-09-10</Text>
        </Row>
        <H1>MUSeg × DFormerv2：从数据适配、可信基线到几何可信 RGB-D 双路径 MVE</H1>
        <Text tone="secondary" style={{ maxWidth: 1040 }}>
          本轮工作的主线是先把数据、训练和评价链做可信，再用最小可行实验（MVE）判断“深度几何可靠性”是否值得继续。
          当前已经形成稳定 RGB Quick-B0，完成 DVC-A1 的正式开发评价并得到明确负结果；正在推进 DVG-B1 Oracle GSA 门控的公式冻结。
        </Text>
        <Callout tone="success" title="结论先行">
          基线已经闭合，最终 checkpoint 为 epoch 420，主评价 mIoU <Code>58.79%</Code>、mAcc <Code>69.91%</Code>、mF1 <Code>72.73%</Code>。
          DVC-A1 没有观察到预注册的边界特异敏感性，裁决为 <Code>not-supported</Code>。当前重点不是继续堆复杂模块，而是先验证一个理想化的最小门控动作是否本身有价值。
        </Callout>
        <Callout tone="warning" title="汇报边界">
          所有模型结果均属于 single-seed development evidence；official test 保持 <Code>sealed_unread</Code>。Oracle 使用真实 corruption mask，只表示方案上限，不表示真实系统已经能自动识别坏深度。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => openFile(reportPath)}>打开完整 Markdown 报告</Button>
          <Button variant="secondary" onClick={() => openFile(statusPath)}>打开唯一当前状态</Button>
          <Button variant="secondary" onClick={() => openFile(dvcPlanPath)}>打开 DVC-A1 证据</Button>
          <Button variant="secondary" onClick={() => openFile(dvgPlanPath)}>打开 DVG-B1 计划</Button>
        </Row>
      </Stack>

      <Grid columns={5} gap={14}>
        <Stat value="3,171" label="确定性重建样本组" tone="info" />
        <Stat value="31.98 亿" label="逐像素深度核验" tone="info" />
        <Stat value="2 × 500" label="两次长程训练 epoch" />
        <Stat value="34,520+" label="主评价与 MVE view 前向" tone="success" />
        <Stat value="73" label="唯一 WOS 题录筛选" />
      </Grid>

      <Divider />

      <Stack gap={16}>
        <H2>1. 三周工作主线：从“数字能跑”推进到“问题可证伪”</H2>
        <Grid columns="1fr 1fr" gap={22} align="start">
          <Stack gap={12}>
            <PhaseStep
              index="01"
              title="MUSeg 数据与语义适配"
              status="已完成"
              tone="success"
              summary="重建 RGB、Depth16、Depth 和 Label，分离 background、ignore 与 invalid depth。"
              detail="3,171 组样本；3,197,712,504 个像素的深度映射匹配率 100%。"
            />
            <PhaseStep
              index="02"
              title="训练稳定性与协议收缩"
              status="已完成"
              tone="success"
              summary="修复空有效像素 loss，识别 RGB/BGR 和 evaluator geometry 对结果的强影响。"
              detail="早期路线只保留可迁移经验，不与当前 RGB Quick-B0 混合归因。"
            />
            <PhaseStep
              index="03"
              title="RGB Quick-B0 可信基线"
              status="已完成"
              tone="success"
              summary="独立完成 500 epoch 训练和 4 个候选五尺度翻转主评价。"
              detail="epoch 420 由冻结主 evaluator 选出；official test 未参与。"
            />
          </Stack>
          <Stack gap={12}>
            <PhaseStep
              index="04"
              title="DVC-A1 问题验证"
              status="已裁决"
              tone="success"
              summary="通过 v1/v2 两次合法阻塞修正操作与评价定义，v3 完成正式开发评价。"
              detail="218 张图 × 5 条件 × 10 view；联合裁决 not-supported。"
            />
            <PhaseStep
              index="05"
              title="DVG-B1 Oracle 方案验证"
              status="设计中"
              tone="warning"
              summary="唯一 gate 位置、四级 GSA 结构、参数链和 no-op 验收已核清。"
              detail="当前只缺 A/B 公式依据与 C 裁决门槛；尚未创建 protocol 或修改模型。"
            />
            <PhaseStep
              index="06"
              title="后续方向分流"
              status="候选"
              tone="info"
              summary="整体深度有效性、后验校准、低照粉尘、标定误差等保持独立候选。"
              detail="只有 Oracle 先显示实际价值，才考虑质量预测网络和完整双路径恢复。"
            />
          </Stack>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>2. 数据适配：工作量最大的基础环节</H2>
        <Grid columns="1.1fr 0.9fr" gap={22} align="start">
          <Stack gap={12}>
            <H3>从原始数据到可审计训练输入</H3>
            <Table
              headers={["对象", "已完成处理", "关键结果"]}
              rows={[
                ["样本身份", "RGB、Depth16、Depth、Label 主文件名集合核对", "3,171 组一致"],
                ["官方划分", "train/test 样本与位置组交集核验", "1,595 / 1,576，交集均为 0"],
                ["深度量化", "固定全局公式 round(Depth16 × 255 / 13932)", "31.98 亿像素，100% 匹配"],
                ["标签语义", "raw 0=background；1–15=前景类；训练 0→255", "background 与 true ignore 分离"],
                ["可重复入口", "临时目录生成、完整验证、原子替换、元数据哈希", "避免半成品和人工漂移"],
              ]}
              columnAlign={["left", "left", "left"]}
              rowTone={["success", "success", "success", "info", "info"]}
              striped
            />
            <Callout tone="info" title="这一步的实际意义">
              它不直接证明模型性能，但保证后续 corruption、validity、Boundary IoU 和门控实验都基于同一套深度与标签语义。
              Depth16 被保留为权威原始域，避免在已经量化的 8-bit 深度上二次推断。
            </Callout>
          </Stack>

          <Stack gap={10}>
            <H3>原始 Depth 无效像素占比</H3>
            <PieChart
              data={[
                { label: "有效深度像素", value: 69.2649, tone: "info" },
                { label: "无效深度 0", value: 30.7351, tone: "warning" },
              ]}
              size={250}
              donut
            />
            <Text size="small" tone="tertiary">图例：原始 Depth16 像素状态 · 单位：像素占比（%）</Text>
            <Text size="small" tone="quaternary">来源：MUSeg 全量数据语义核验；无效值定义为原始 Depth=0。</Text>
            <Callout tone="warning" title="研究动机，但不是问题结论">
              约 30.74% 的深度像素无效，使“模型是否应该无条件相信深度几何”成为合理问题；占比本身不能证明 GSA 已经受到伤害。
            </Callout>
          </Stack>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>3. 早期试错：只保留对当前路线有价值的三点</H2>
        <Table
          headers={["早期问题或路线", "观察 / 处理", "对当前路线的影响"]}
          rows={[
            [
              "空有效像素 loss",
              "11 张全背景图可能产生全 ignore crop；旧 masked mean 会出现非有限值",
              "实现 safe_masked_mean 并保留在稳定基线；这是数值稳定性修复，不宣称性能增益",
            ],
            [
              "16 张 Depth block-mask pilot",
              "历史 epoch-10 上 q=0.3 / q=0.5 仅下降 0.1107 / 0.3337 个 mIoU 百分点",
              "只够淘汰直接上复杂 validity gating 的路线，不能替代当前 B0 正式验证",
            ],
            [
              "BGR 与 evaluator geometry",
              "同一历史 checkpoint 在三种 geometry 下 mIoU 跨度 4.42 pp；直接换 RGB 也大幅变动",
              "证明协议和输入契约必须先冻结，不能按临时最高分回选路线",
            ],
          ]}
          rowTone={["success", "warning", "warning"]}
          striped
        />
        <Text size="small" tone="quaternary">来源：历史 MVE、后评价诊断和当前组会报告第 5 节。不同协议数字不做单变量因果归因。</Text>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>4. 稳定 RGB Quick-B0：最终模型由最终 evaluator 决定</H2>
        <Grid columns="1fr 1fr" gap={22} align="start">
          <Stack gap={10}>
            <H3>4 个候选 checkpoint 的主评价</H3>
            <BarChart
              categories={candidateEpochs}
              series={[
                { name: "mIoU", data: [58.79, 58.73, 58.43, 57.68], tone: "success" },
                { name: "mAcc", data: [69.91, 69.54, 69.34, 68.84], tone: "info" },
                { name: "mF1", data: [72.73, 72.67, 72.44, 71.81] },
              ]}
              height={330}
              yMin={55}
              yMax={75}
              valueSuffix="%"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：候选 epoch · 纵轴：318 张 val-dev 指标（%） · 图例：mIoU、mAcc、mF1</Text>
            <Text size="small" tone="quaternary">来源：msflip-whole-original-grid-v1；5 个尺度 × 原图/翻转，FP32 pre-softmax logits 平均。</Text>
          </Stack>

          <Stack gap={12}>
            <Card size="lg">
              <CardHeader trailing={<Pill size="sm" active>最终基线</Pill>}>epoch 420</CardHeader>
              <CardBody>
                <Grid columns={3} gap={12}>
                  <Stat value="58.79%" label="mIoU" tone="success" />
                  <Stat value="69.91%" label="mAcc" tone="info" />
                  <Stat value="72.73%" label="mF1" tone="info" />
                </Grid>
                <Divider style={{ margin: "14px 0" }} />
                <Text size="small" tone="secondary">checkpoint：<Code>selector-epoch-420.pth</Code></Text>
                <Text size="small" tone="secondary">SHA-256：<Code>f246a3af…e3a1c</Code></Text>
                <Text size="small" tone="secondary">身份：single-seed RGB development Quick-B0</Text>
              </CardBody>
            </Card>
            <Callout tone="warning" title="候选排名发生反转">
              训练期 selector 第一名是 epoch 480，但它在主评价中只排第三；epoch 420 从 selector 第二升到主评价第一。
              这直接说明最终 checkpoint 不能只按低成本单尺度 validation 决定。
            </Callout>
            <Callout tone="info" title="评价合同">
              4 个候选 × 318 张 × 10 view，共 <Code>12,720</Code> 个 view 级前向；本地主评价内部计时 <Code>94.044</Code> 分钟，4 个候选全部完成且未 OOM。
            </Callout>
          </Stack>
        </Grid>

        <Grid columns="1fr 1fr" gap={20} align="start">
          <Stack gap={9}>
            <H3>训练期 selector 与主 mIoU</H3>
            <LineChart
              categories={["epoch 420", "epoch 440", "epoch 480"]}
              series={[
                { name: "训练期 selector mIoU", data: [56.39, 56.10, 56.87] },
                { name: "五尺度翻转主 mIoU", data: [58.79, 58.73, 58.43], tone: "success" },
              ]}
              height={270}
              yMin={55.5}
              yMax={59.2}
              valueSuffix="%"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：具有独立 selector 分数的候选 · 纵轴：val-dev mIoU（%）</Text>
            <Text size="small" tone="quaternary">来源：训练候选清单与冻结主评价。epoch 500 为 latest，无独立 selector 分数，未放入本图。</Text>
          </Stack>
          <Stack gap={9}>
            <H3>训练工作量</H3>
            <BarChart
              categories={["legacy BGR", "RGB Quick-B0"]}
              series={[{ name: "单 seed 长程训练墙钟", data: [12.117, 13.633], tone: "info" }]}
              height={250}
              valueSuffix=" 小时"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：两次独立 500 epoch 训练 · 纵轴：墙钟时间（小时）</Text>
            <Text size="small" tone="quaternary">来源：历史训练与 RGB Quick-B0 运行记录；合计约 25 小时 45 分。协议不同，不比较性能胜负。</Text>
          </Stack>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>5. 为什么转向几何可信 RGB-D 双路径 MVE</H2>
        <Text tone="secondary">
          DFormerv2 把深度差异转成 geometry prior（几何先验）注入 Geometry Self-Attention（GSA，几何自注意力），并非单独编码 Depth 后再融合。
          因此首轮 MVE 被拆成两个逻辑独立的问题，避免把“问题有没有找准”和“某个动作有没有用”混成一次实验。
        </Text>
        <Grid columns="1fr 72px 1fr" gap={16} align="stretch">
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">问题验证</Pill>}>A 路径 · DVC-A1</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <H3>深度边界失效是否特别伤害语义边界？</H3>
                <Text tone="secondary">在原始 Depth16 域构造可复现 corruption，比较边界与同面积非边界干预。</Text>
                <Callout tone="info" title="输出性质">回答预设问题定位是否成立。</Callout>
              </Stack>
            </CardBody>
          </Card>
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", color: theme.text.tertiary, fontSize: 20 }}>≠</div>
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">方案验证</Pill>}>B 路径 · DVG-B1</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <H3>已知坏区时，少信这些位置的深度几何是否有用？</H3>
                <Text tone="secondary">Oracle 提供真实坏区，只门控 GSA 的 depth geometry contribution。</Text>
                <Callout tone="info" title="输出性质">回答最小动作的理想上限是否值得继续。</Callout>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
        <Callout tone="warning" title="逻辑边界">
          A 不支持时，B 仍可独立验证方案上限；B 若支持，也不能反向把 A 改写成支持。B 若同样不支持，则应关闭当前“深度边界 + GSA Oracle 门控”首选路线。
        </Callout>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>6. DVC-A1：两次合法阻塞后，才得到可解释的负结果</H2>
        <Grid columns="1fr 1fr 1fr" gap={16} align="start">
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">protocol-blocked</Pill>}>v1 · 覆盖门禁</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Stat value="58 / 196" label="无法构造非空 q75 的位置组" tone="warning" />
                <Text tone="secondary">占 <Code>29.5918%</Code>，高于预注册上限 <Code>5%</Code>。</Text>
                <Divider />
                <Text size="small" tone="tertiary">在完整模型评价前停止。含义是操作定义覆盖不足，不是模型不敏感。</Text>
              </Stack>
            </CardBody>
          </Card>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">protocol-blocked</Pill>}>v2 · 统计定义</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Stat value="10,900" label="完整 view 级模型前向" tone="info" />
                <Text tone="secondary">218 张 × 5 条件 × 10 view；用时 <Code>5445.565 s</Code>。</Text>
                <Divider />
                <Text size="small" tone="tertiary">五条件推理完成，但 Boundary IoU 只有 137/138 个有效配对组，因此停止。</Text>
              </Stack>
            </CardBody>
          </Card>

          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" active>not-supported</Pill>}>v3 · 正式裁决</CardHeader>
            <CardBody>
              <Stack gap={9}>
                <Stat value="138 / 138" label="有效配对位置组" tone="success" />
                <Text tone="secondary">只分离 background 与 true ignore；用时 <Code>5507.055 s</Code>。</Text>
                <Divider />
                <Text size="small" tone="tertiary">完成 10,000 次位置组 bootstrap；预注册联合支持条件未满足。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
        <Callout tone="info" title="为什么两次阻塞不等于浪费">
          v1 阻止了在覆盖不足的操作定义上消耗完整评价；v2 暴露 background/ignore 标签域混用；v3 只修正评价定义而不改阈值、样本、条件、checkpoint 或裁决门槛，最终结果才具备科学解释。
        </Callout>
      </Stack>

      <Stack gap={16}>
        <H3>v3 五个条件的描述性总体指标</H3>
        <BarChart
          categories={conditionNames}
          series={[
            { name: "mIoU", data: [54.64, 54.66, 54.74, 54.64, 54.43], tone: "success" },
            { name: "mAcc", data: [64.23, 64.27, 64.33, 64.32, 64.02], tone: "info" },
            { name: "mF1", data: [67.59, 67.60, 67.67, 67.57, 67.39] },
          ]}
          height={330}
          yMin={53.5}
          yMax={68.5}
          valueSuffix="%"
          showValues
        />
        <Text size="small" tone="tertiary">横轴：冻结 corruption 条件 · 纵轴：218 张开发样本总体指标（%） · 图例：mIoU、mAcc、mF1</Text>
        <Text size="small" tone="quaternary">来源：DVC-A1 v3 completed summary；这些是描述性总体指标，正式裁决使用位置组配对 Boundary IoU effect。</Text>
      </Stack>

      <Card size="lg">
        <CardHeader trailing={<Pill size="sm" active>联合裁决：not-supported</Pill>}>预注册 Boundary IoU 效应与 95% 区间</CardHeader>
        <CardBody>
          <Stack gap={18}>
            <EffectInterval
              title="dose_effect = boundary-q75 − clean"
              point={0.0731348717}
              low={-0.0620441424}
              high={0.2226503089}
              note="区间跨 0，未观察到预注册剂量损害。"
            />
            <Divider />
            <EffectInterval
              title="specificity_effect = boundary-q50 − nonboundary-q50"
              point={0.0323919541}
              low={-0.3002343847}
              high={0.2899672010}
              note="区间跨 0，未观察到边界干预相对同面积非边界干预的特异损害。"
            />
            <Text size="small" tone="tertiary">横轴：Boundary IoU 效应（百分点，pp）；圆点为配对位置组效应，线段为 10,000 次 group bootstrap 的 95% percentile interval。</Text>
            <Text size="small" tone="quaternary">来源：DVC-A1 v3 主分析，138/138 个有效配对位置组；official_test_included=false。</Text>
          </Stack>
        </CardBody>
      </Card>

      <Grid columns="1fr 1fr" gap={18} align="start">
        <Callout tone="success" title="当前证据可以支持">
          在固定的可构造 val-dev 位置组中，人工把深度边界像素置零，没有观察到预注册的边界特异敏感性；因此不再把这个窄问题当作首要瓶颈。
        </Callout>
        <Callout tone="warning" title="当前证据不能扩展为">
          不能说自然无效深度、低照、粉尘或传感器故障没有问题；不能说 Depth 对 DFormerv2 没作用；不能外推到 official test、部署、跨数据集或多 seed 统计。
        </Callout>
      </Grid>

      <Divider />

      <Stack gap={16}>
        <H2>7. 当前重点：DVG-B1 Oracle GSA 门控</H2>
        <Text tone="secondary">
          DVG-B1 不修改 Depth 输入、Q/K/V、decoder 或最终 logits 结构，只在 spatial contribution 与 depth geometry contribution 加和前，
          对后者施加由 Oracle reliability 得到的 pairwise gate。
        </Text>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">唯一允许的改动位置</Pill>}>GeoPriorGen.forward</CardHeader>
          <CardBody>
            <Grid columns="1fr 44px 1.25fr 44px 1fr" gap={10} align="center">
              <Stack gap={5} style={{ padding: 12, background: theme.fill.tertiary, borderRadius: 6 }}>
                <Text weight="semibold">spatial contribution</Text>
                <Text size="small" tone="secondary">位置几何项保持原样</Text>
              </Stack>
              <Text style={{ textAlign: "center" }}>+</Text>
              <Stack gap={5} style={{ padding: 12, background: theme.fill.secondary, borderRadius: 6, border: `1px solid ${theme.stroke.secondary}` }}>
                <Text weight="semibold">gate × depth geometry contribution</Text>
                <Text size="small" tone="secondary">只门控 <Code>self.weight[1] * mask_d*</Code></Text>
              </Stack>
              <Text style={{ textAlign: "center" }}>→</Text>
              <Stack gap={5} style={{ padding: 12, background: theme.fill.tertiary, borderRadius: 6 }}>
                <Text weight="semibold">geometry prior</Text>
                <Text size="small" tone="secondary">Attention 仍只消费合成结果</Text>
              </Stack>
            </Grid>
            <Divider style={{ margin: "16px 0" }} />
            <Text size="small" tone="tertiary">参数链：EncoderDecoder.forward/encode_decode → dformerv2.forward → BasicLayer.forward → RGBD_Block.forward → GeoPriorGen.forward。</Text>
          </CardBody>
        </Card>

        <Grid columns="1fr 1fr 1fr" gap={16} align="start">
          <GateShape label="Stage 0–2 · H gate" shape="[B,1,W,H,H]" formula="冻结同列 token pair 的 depth contribution gate。" />
          <GateShape label="Stage 0–2 · W gate" shape="[B,1,H,W,W]" formula="冻结同行 token pair 的 depth contribution gate。" />
          <GateShape label="Stage 3 · Full gate" shape="[B,1,L,L]" formula="冻结完整 token pair 的 depth contribution gate。" />
        </Grid>

        <Grid columns="1fr 1fr" gap={18} align="start">
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm" active>已闭合</Pill>}>项目内实现事实</CardHeader>
            <CardBody>
              <Stack gap={8}>
                <Text>前三个 stage 使用 H/W 分解 GSA；第四个 stage 使用 Full GSA。</Text>
                <Text>当前代码与作者保留副本 SHA-256 完全一致。</Text>
                <Text>论文写 average pooling、作者代码用 bilinear interpolation；当前以 checkpoint 对应代码为准。</Text>
                <Text>原 Depth16 corruption、五条件、10-view evaluator、原始网格和 JSON 框架可复用。</Text>
                <Text><Code>None</Code>、clean、q=0、全可信必须走原始旁路并满足 <Code>torch.equal</Code>。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card size="lg">
            <CardHeader trailing={<Pill size="sm">尚未授权实现</Pill>}>当前 A / B / C 门禁</CardHeader>
            <CardBody>
              <Stack gap={10}>
                <Text><Text as="span" weight="semibold">A · 像素 mask → 四级 token reliability：</Text>冻结 resize、聚合、部分受损 patch 和 bilinear 对齐语义。</Text>
                <Divider />
                <Text><Text as="span" weight="semibold">B · token reliability → pairwise gate：</Text>冻结 query/key 两端组合、对称性、hard/continuous 和 Full/H/W 公式。</Text>
                <Divider />
                <Text><Text as="span" weight="semibold">C · 正式科学裁决：</Text>冻结最小实际效应量、clean 不劣容忍度和指标集合。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
        <Callout tone="warning" title="当前执行边界">
          A、B、C 任一未关闭前，不创建 <Code>DVG-B1-oracle-gsa-v1</Code> protocol，不修改模型，不运行 preflight、GPU、训练、云资源或 official test。
        </Callout>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>8. 文献筛选：从 74 条记录收缩到 73 个唯一题录和 7 个优先候选</H2>
        <Grid columns="0.8fr 1.2fr" gap={22} align="start">
          <Stack gap={10}>
            <H3>筛选工作量</H3>
            <BarChart
              categories={["A-1 记录", "B-1 记录", "唯一 WOS ID", "P0 优先候选"]}
              series={[{ name: "题录数量", data: [44, 30, 73, 7], tone: "info" }]}
              height={280}
              valueSuffix=" 条"
              showValues
            />
            <Text size="small" tone="tertiary">横轴：摘要筛选阶段 · 纵轴：题录数量（条）</Text>
            <Text size="small" tone="quaternary">来源：用户提供 A-1/B-1 WOS 记录；两文件仅共享 BurnDC 一条。</Text>
          </Stack>
          <Stack gap={10}>
            <H3>当前最需要获取的全文与官方代码</H3>
            <Table
              headers={["开放项", "P0 候选", "需要提取的直接证据"]}
              rows={[
                ["A", "Confidence Propagation through CNNs", "normalized confidence 定义与跨层传播公式"],
                ["A", "Bcap-net", "hierarchical multi-scale weighted pooling 与 confidence propagation"],
                ["A", "LightDepth", "sparse-depth resize、validity 同步和插值实现"],
                ["B", "Non-local affinity propagation", "pixel reliability 如何进入 neighbor affinity / propagation weight"],
                ["B", "Multi-Affinity Matrix", "confidence 是否直接进入 pairwise affinity"],
                ["B", "NR-MVSNet", "reliable attention 对 cost-volume score 的具体修改"],
                ["B", "LFDA", "pairwise attention score 的方向性和对称性"],
              ]}
              rowTone={["info", "info", "info", "warning", "warning", "warning", "warning"]}
              striped
            />
          </Stack>
        </Grid>
        <Callout tone="info" title="摘要筛选已经完成什么">
          已把精读范围缩小，并找到“pixel depth reliability 可以调制 neighbor affinity/propagation weight”的直接方向性线索；但摘要没有给出两端组合、对称性、hard/continuous、全 1 恒等或 Full/H/W 映射，因此不能据此冻结最终公式。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="secondary" onClick={() => openFile(literaturePath)}>打开摘要筛选与全文优先级</Button>
          <Button variant="secondary" onClick={() => openFile(decisionsPath)}>打开研究选择与边界</Button>
        </Row>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>9. 当前 MVE 之外的其他方向</H2>
        <Table
          headers={["候选方向", "要回答的问题", "当前状态", "重新启用条件"]}
          rows={[
            ["整体深度有效性 A2/B2", "自然或人工整体深度失效是否需要显式 validity", "延期、未执行、未授权", "独立 protocol 与当前 B0 上的正式问题验证"],
            ["后验校准与风险—覆盖", "冻结 logits 的概率可信度、错误排序和阈值迁移", "延期、未执行、未授权", "独立 calibration 数据职责与共同 logits 身份"],
            ["RGB 低照与粉尘退化", "彩色观测退化及其与深度可靠性的联合作用", "方向候选", "冻结 corruption、数据职责和评价协议"],
            ["深度补全与标定误差", "填补、尺度、RGB-D 错位或标定偏差是否主导误差", "首轮删除", "一次只定义一个问题，避免变量过多"],
            ["可学习质量预测与联合恢复", "自动估计质量并驱动完整双路径恢复", "远期候选", "Oracle 门控先显示实际价值"],
          ]}
          rowTone={["neutral", "neutral", "info", "warning", "warning"]}
          striped
        />
        <Text size="small" tone="quaternary">来源：当前状态与组会报告第 11 节。表中候选均不构成执行授权。</Text>
      </Stack>

      <Divider />

      <Stack gap={16}>
        <H2>10. 结果、风险与下一步</H2>
        <Grid columns="1fr 1fr 1fr" gap={16} align="start">
          <Stack gap={8} style={{ padding: "4px 2px" }}>
            <Pill size="sm" active>已完成并验证</Pill>
            <H3>共同基线和 A 路径已收口</H3>
            <Text tone="secondary">数据重建、RGB Quick-B0、DVC-A1 v1/v2 阻塞证据、v3 正式裁决和 DVG-B1 实现定位均已闭合。</Text>
          </Stack>
          <Stack gap={8} style={{ padding: "4px 2px" }}>
            <Pill size="sm">正在进行</Pill>
            <H3>获取全文并冻结 A/B/C</H3>
            <Text tone="secondary">从 P0 全文、补充材料和官方代码提取可复现公式；C 需要直接参考或明确项目预注册选择。</Text>
          </Stack>
          <Stack gap={8} style={{ padding: "4px 2px" }}>
            <Pill size="sm">主要风险</Pill>
            <H3>Oracle 收益可能很小</H3>
            <Text tone="secondary">A 已 not-supported，B 必须先冻结最小效应量与 clean 不劣界，避免结果后解释或把理想上限误写成真实系统收益。</Text>
          </Stack>
        </Grid>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">恢复点明确</Pill>}>DVG-B1 最小执行顺序</CardHeader>
          <CardBody>
            <Grid columns={4} gap={14} align="start">
              <Stack gap={6}><Pill size="sm" active>1</Pill><Text weight="semibold">关闭 A / B / C</Text><Text size="small" tone="secondary">全文与代码提取公式；冻结裁决阈值。</Text></Stack>
              <Stack gap={6}><Pill size="sm" active>2</Pill><Text weight="semibold">申请 protocol / 代码授权</Text><Text size="small" tone="secondary">三组门禁关闭后再物化独立身份。</Text></Stack>
              <Stack gap={6}><Pill size="sm" active>3</Pill><Text weight="semibold">最小定点验证</Text><Text size="small" tone="secondary">shape、finite、原始旁路 torch.equal、1–2 样本 preflight。</Text></Stack>
              <Stack gap={6}><Pill size="sm" active>4</Pill><Text weight="semibold">配对开发评价</Text><Text size="small" tone="secondary">单独授权后运行；仍不读取 official test。</Text></Stack>
            </Grid>
          </CardBody>
        </Card>

        <Grid columns="1fr 1fr" gap={18} align="start">
          <Callout tone="success" title="若 Oracle 支持">
            只说明“少信已知坏区的深度几何”有理想上限价值；下一步才值得研究真实质量信号或可学习 reliability，并需要新的独立 protocol。
          </Callout>
          <Callout tone="warning" title="若 Oracle 不支持">
            关闭当前首选路线，从整体深度失效、RGB 低照/粉尘、后验校准或标定误差中选择一个新问题；不并行堆叠多个变量。
          </Callout>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>11. 建议的组会讲述顺序</H2>
        <Grid columns="1fr 1fr" gap={20} align="start">
          <Stack gap={10}>
            <Text weight="semibold">1. 用一句话交代目标</Text>
            <Text tone="secondary">先建立可信共同基线，再用最小实验判断几何可靠性方向值不值得继续。</Text>
            <Text weight="semibold">2. 用大数字体现基础工作量</Text>
            <Text tone="secondary">3,171 组数据、31.98 亿像素、30.74% 无效深度。</Text>
            <Text weight="semibold">3. 用排名反转讲基线成果</Text>
            <Text tone="secondary">selector 第一的 epoch 480 在主评价只排第三；最终 epoch 420 为 58.79 / 69.91 / 72.73。</Text>
            <Text weight="semibold">4. 三句话交代早期试错</Text>
            <Text tone="secondary">safe loss、16 张 pilot、BGR/geometry 协议敏感性；不展开无关过程。</Text>
          </Stack>
          <Stack gap={10}>
            <Text weight="semibold">5. 重点讲 DVC-A1 v1 → v2 → v3</Text>
            <Text tone="secondary">两次阻塞如何逐步修正覆盖与评价定义，最终得到有效的 not-supported 裁决。</Text>
            <Text weight="semibold">6. 展示两个 effect 接近 0 且区间跨 0</Text>
            <Text tone="secondary">明确负结果只否定当前窄问题，不外推自然故障和部署。</Text>
            <Text weight="semibold">7. 解释为什么继续 DVG-B1</Text>
            <Text tone="secondary">A 问问题是否找准，B 问动作是否有效；展示唯一 gate 位置和 A/B/C 门禁。</Text>
            <Text weight="semibold">8. 用条件式下一步收口</Text>
            <Text tone="secondary">Oracle 有效才做质量预测；Oracle 无效就换问题，不堆复杂度。</Text>
          </Stack>
        </Grid>
      </Stack>

      <Callout tone="success" title="本轮组会可以传达的核心成果">
        项目已经从 DFormerv2 与 MUSeg 的基础适配，推进到拥有可审计共同基线、能合法停止的预注册门禁、一次完整负结果和一个已定位到具体张量形状的下一步方案。
        当前工作重点已经从“继续尝试模块”收缩为“先把唯一公式与裁决标准冻结，再决定是否值得运行”。
      </Callout>

      <Row gap={8} wrap>
        <Button variant="primary" onClick={() => openFile(reportPath)}>阅读完整报告</Button>
        <Button variant="secondary" onClick={() => openFile(statusPath)}>核对当前事实</Button>
        <Button variant="secondary" onClick={() => openFile(dvgPlanPath)}>继续 DVG-B1</Button>
        <Button variant="secondary" onClick={() => openFile(literaturePath)}>查看全文优先级</Button>
      </Row>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        事实来源：MUSeg 唯一当前状态、实验口径记录、2026-09-10 组会 Markdown、Quick-B0 正式报告、DVC-A1 v1/v2/v3 冻结证据、DVG-B1 计划与 73 条 WOS 摘要筛选。
        本 Canvas 只重排已核验事实，不引入新实验结论；未重新运行训练、GPU 评价、完整测试套件、云资源或 official test。
      </Text>
    </Stack>
  );
}

export default App;
