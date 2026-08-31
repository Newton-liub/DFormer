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
const reportPath = `${projectRoot}\\doc\\reports\\2026-08-31-museg-stage05-seed1-training-lessons.md`;
const statusPath = `${projectRoot}\\doc\\main\\MUSeg-current-status.md`;
const postevalPath = `${projectRoot}\\doc\\reports\\2026-08-28-museg-stage05-posteval-protocol-gate.md`;
const trainLogPath = `${projectRoot}\\cloud\\DFormer-stage05-evidence\\train.log`;

const lossPoints = [
  [1, 3.046055], [10, 1.67724], [20, 1.175709], [30, 0.846153], [40, 0.634635],
  [50, 0.541382], [60, 0.454421], [70, 0.39765], [80, 0.326397], [90, 0.300266],
  [100, 0.287613], [110, 0.241513], [120, 0.23441], [130, 0.201561], [140, 0.182828],
  [150, 0.171375], [160, 0.189676], [170, 0.163349], [180, 0.134477], [190, 0.136804],
  [200, 0.133996], [210, 0.124781], [220, 0.118571], [230, 0.115908], [240, 0.115628],
  [250, 0.122639], [260, 0.115343], [270, 0.098989], [280, 0.097289], [290, 0.088746],
  [300, 0.091678], [310, 0.086341], [320, 0.096425], [330, 0.076141], [340, 0.079593],
  [350, 0.075637], [360, 0.072609], [370, 0.074832], [380, 0.06985], [390, 0.068684],
  [400, 0.066131], [410, 0.066961], [420, 0.064509], [430, 0.06313], [440, 0.064687],
  [450, 0.064702], [460, 0.061468], [470, 0.060879], [480, 0.060139], [490, 0.057919],
  [500, 0.0579],
] as const;

const validationPoints = [
  [10, 17.42], [20, 28.43], [30, 29.0], [40, 35.06], [50, 35.9],
  [60, 41.25], [70, 38.39], [80, 43.04], [90, 41.89], [100, 42.8],
  [110, 46.13], [120, 42.6], [130, 43.5], [140, 44.31], [150, 45.43],
  [160, 45.12], [170, 47.04], [180, 47.87], [190, 47.87], [200, 47.84],
  [210, 46.72], [220, 45.76], [230, 47.1], [240, 46.26], [250, 47.13],
  [260, 47.04], [270, 48.88], [280, 47.23], [290, 49.13], [300, 51.05],
  [310, 51.98], [320, 49.59], [330, 51.92], [340, 51.28], [350, 51.4],
  [360, 51.69], [370, 51.65], [380, 50.85], [390, 51.47], [400, 51.32],
  [410, 51.09], [420, 50.53], [430, 49.63], [440, 52.41], [450, 52.14],
  [460, 52.84], [470, 51.69], [480, 51.79], [490, 51.93], [500, 52.07],
] as const;

const learningRatePoints = [
  [1, 2.9766e-5], [2, 5.9766e-5], [10, 5.892e-5], [100, 4.9084e-5],
  [200, 3.7888e-5], [300, 2.6304e-5], [400, 1.4096e-5], [500, 2.8352e-9],
] as const;

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();
  const openFile = (path: string) => dispatch({ type: "openFile", path });

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1220, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.10 · 历史训练经验复盘</Pill>
          <Text size="small" tone="tertiary">训练：2026-08-25 至 2026-08-26 · 报告：2026-08-31</Text>
        </Row>
        <H1>Stage-05 seed 1：训练跑稳，但最后一轮不是验证峰值</H1>
        <Text tone="secondary">
          历史 legacy BGR 单 seed 完成了 500 epoch；loss 总体下降但有局部回升，val-dev mIoU 在后期进入高位波动。
          最直接的经验是保留候选 checkpoint，并把训练筛选与最终 evaluator 分开。
        </Text>
        <Callout tone="warning" title="与当前 RGB quick B0 严格隔离">
          本页只复盘历史 seed <Code>772961337</Code>，数据仅为冻结 val-dev，official test 保持封存未读。它不比较、不修改也不重定向当前训练；当前状态仍以 <Code>doc/main/MUSeg-current-status.md</Code> 为准。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => openFile(reportPath)}>打开完整报告</Button>
          <Button variant="secondary" onClick={() => openFile(statusPath)}>打开当前状态</Button>
          <Button variant="secondary" onClick={() => openFile(trainLogPath)}>打开历史训练日志</Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="500 / 500" label="完成 epoch" tone="success" />
        <Stat value="52.84" label="峰值 val mIoU · epoch 460" tone="info" />
        <Stat value="52.07" label="epoch 500 val mIoU" tone="neutral" />
        <Stat value="12h 07m" label="训练总时长" tone="neutral" />
      </Grid>

      <Grid columns="1.2fr 0.8fr" gap={18} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>50 / 50 点</Pill>}>Validation mIoU 原始曲线</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <LineChart
                categories={validationPoints.map(([epoch]) => String(epoch))}
                series={[{ name: "validation/miou", data: validationPoints.map(([, value]) => value), tone: "info" }]}
                height={290}
                yMin={15}
                yMax={55}
                valueSuffix=" mIoU"
                referenceLines={[{ value: 52.84, label: "峰值 52.84", tone: "success" }]}
              />
              <Text size="small" tone="tertiary">横轴：epoch（每 10 epoch） · 纵轴：val-dev mIoU（百分点）</Text>
              <Text size="small" tone="quaternary">来源：SwanLab run 4qbda9xh 与 train.log，50 点逐点一致；未平滑。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={14}>
          <Callout tone="info" title="后期是平台和波动，不是已证明的过拟合">
            epoch 410–500 的 mIoU 均值为 <Code>51.612</Code>，标准差约 <Code>0.902</Code>，范围 <Code>49.63–52.84</Code>。单 seed 曲线不足以给出因果判断。
          </Callout>
          <Card>
            <CardHeader>三个阶段</CardHeader>
            <CardBody>
              <Stack gap={10}>
                <Row justify="space-between"><Text tone="secondary">epoch 10–160</Text><Text weight="semibold">+27.70</Text></Row>
                <Row justify="space-between"><Text tone="secondary">epoch 170–330</Text><Text weight="semibold">+4.88</Text></Row>
                <Row justify="space-between"><Text tone="secondary">epoch 340–500</Text><Text weight="semibold">+0.79</Text></Row>
                <Divider />
                <Row justify="space-between"><Text tone="secondary">最大回撤</Text><Text>−3.53</Text></Row>
                <Text size="small" tone="tertiary">epoch 110 的 46.13 到 epoch 120 的 42.60。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Stack>
      </Grid>

      <Grid columns="1fr 1fr" gap={18} align="start">
        <Stack gap={8}>
          <H2>Loss 继续下降</H2>
          <LineChart
            categories={lossPoints.map(([epoch]) => String(epoch))}
            series={[{ name: "train/epoch_loss", data: lossPoints.map(([, value]) => value), tone: "success" }]}
            height={270}
            valueSuffix=" loss"
          />
          <Text size="small" tone="tertiary">横轴：epoch · 纵轴：epoch mean loss</Text>
          <Text size="small" tone="quaternary">来源：SwanLab 500 个原始点；画面展示 epoch 1 与每 10 epoch 抽样，无平滑。3.046055 → 0.057900，下降 98.10%。</Text>
        </Stack>

        <Stack gap={8}>
          <H2>Learning rate 按登记日程衰减</H2>
          <LineChart
            categories={learningRatePoints.map(([epoch]) => String(epoch))}
            series={[{ name: "train/learning_rate", data: learningRatePoints.map(([, value]) => value), tone: "warning" }]}
            height={270}
          />
          <Text size="small" tone="tertiary">横轴：选定 epoch · 纵轴：learning rate</Text>
          <Text size="small" tone="quaternary">来源：train.log 与 SwanLab；显示 warmup 后 poly 0.9 衰减的关键节点。</Text>
        </Stack>
      </Grid>

      <Callout tone="warning" title="Loss 与 mIoU 回答不同问题">
        epoch 400–500 的 loss 仍从 <Code>0.066131</Code> 降到 <Code>0.057900</Code>，mIoU 却在高位波动。训练目标继续下降，不等于 validation 或最终 evaluator 仍单调改善。
      </Callout>

      <Divider />

      <Grid columns="1.15fr 0.85fr" gap={18} align="start">
        <Stack gap={8}>
          <H2>五项后评估：checkpoint 与推理几何</H2>
          <BarChart
            categories={["epoch-460 best / original-full", "epoch-460 best / resize-480x640", "epoch-460 best / sliding-480x640", "epoch-500 final / resize-480x640", "epoch-500 final / sliding-480x640"]}
            series={[{ name: "val-dev mIoU", data: [52.98, 56.31, 51.89, 56.73, 52.08], tone: "info" }]}
            height={290}
            yMin={50}
            yMax={58}
            valueSuffix=" mIoU"
            showValues
          />
          <Text size="small" tone="tertiary">横轴：checkpoint 与 geometry · 纵轴：原始 Label 网格 mIoU（百分点）</Text>
          <Text size="small" tone="quaternary">来源：五份 post-evaluation v2 JSON，318 个 val-dev 样本；不同 geometry 不能混成同一协议。</Text>
        </Stack>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>关键教训</Pill>}>Geometry 会重排类别表现</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text>固定 best checkpoint 时，三种 geometry 的总体 mIoU 跨度为 <Code>4.42</Code>。</Text>
              <Divider />
              <H3>Resize 相对 original-full</H3>
              <Text tone="secondary"><Code>mining equipment +12.00</Code>，但 <Code>support equipment −6.11</Code>。</Text>
              <H3>Sliding 相对 original-full</H3>
              <Text tone="secondary"><Code>rescue equipment +8.42</Code>，但 <Code>support equipment −17.86</Code>。</Text>
              <Callout tone="warning" title="不能追认最高数字为历史主协议">
                Resize 改变输入长宽比。后评估证明敏感性，不证明它在研究上天然更正确。
              </Callout>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card>
          <CardHeader>运行资源摘要</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Row justify="space-between"><Text tone="secondary">吞吐中位数</Text><Text>17.01 images/s</Text></Row>
              <Row justify="space-between"><Text tone="secondary">step time 中位数</Text><Text>0.588 s</Text></Row>
              <Row justify="space-between"><Text tone="secondary">max allocated</Text><Text>18,956 MiB</Text></Row>
              <Row justify="space-between"><Text tone="secondary">max reserved</Text><Text>20,632 MiB</Text></Row>
              <Row justify="space-between"><Text tone="secondary">GPU 温度最大值</Text><Text>64°C</Text></Row>
              <Text size="small" tone="tertiary">SwanLab 稀疏训练与系统遥测；GPU utilization 包含 validation、保存和等待时间。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm" tone="warning">历史缺口</Pill>}>Optimizer update 语义</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Row justify="space-between"><Text tone="secondary">理论 iteration 网格</Text><Text>64,000</Text></Row>
              <Row justify="space-between"><Text tone="secondary">记录的有效 update</Text><Text>63,973</Text></Row>
              <Row justify="space-between"><Text tone="secondary">差额</Text><Text>27</Text></Row>
              <Divider />
              <Text tone="secondary">旧遥测不能逐次证明差额原因，因此不能无条件写成 27 次 AMP skip。后续应直接记录 attempted、completed、skipped。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={12}>
        <H2>只带走四条经验</H2>
        <Grid columns="1fr 1fr" gap={14} align="start">
          <Stack gap={6} style={{ padding: "4px 2px" }}>
            <H3>1. 保存 top-k 与 final</H3>
            <Text tone="secondary">最终 epoch 不是训练期峰值；低成本 validation 负责筛选，冻结主 evaluator 负责最终排序。</Text>
          </Stack>
          <Stack gap={6} style={{ padding: "4px 2px" }}>
            <H3>2. 分层看 loss 与 metric</H3>
            <Text tone="secondary">Loss、在线 validation 和主 evaluator 用途不同，任何一层都不能替代另一层。</Text>
          </Stack>
          <Stack gap={6} style={{ padding: "4px 2px" }}>
            <H3>3. 直接记录 update 计数</H3>
            <Text tone="secondary">不要在训练结束后从理论预算反推 AMP 或其他跳过原因。</Text>
          </Stack>
          <Stack gap={6} style={{ padding: "4px 2px" }}>
            <H3>4. 指标绑定完整身份</H3>
            <Text tone="secondary">Checkpoint、split、输入契约、geometry、计分网格和 per-class 结果必须一起报告。</Text>
          </Stack>
        </Grid>
      </Stack>

      <Callout tone="success" title="原始 SwanLab 曲线已找回">
        归档内 <Code>run-4qbda9xh.swanlab</Code> 含完整 500 点 epoch loss 和 50 点 validation 曲线，SHA-256 为 <Code>d12241bc…08cd</Code>。本次不需要用户重新下载，也没有调用云端 API。
      </Callout>

      <Row gap={8} wrap>
        <Button variant="secondary" onClick={() => openFile(reportPath)}>查看全部证据与限制</Button>
        <Button variant="secondary" onClick={() => openFile(postevalPath)}>查看历史后评估报告</Button>
      </Row>

      <Text size="small" tone="quaternary" style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}>
        事实源：SwanLab run 4qbda9xh、train.log、training_result.json、run_config.json、acceptance-v2.json 与五份 post-evaluation v2 JSON。Canvas 只重排历史证据，不修改当前 RGB quick B0。
      </Text>
    </Stack>
  );
}

export default App;
