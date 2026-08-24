import {
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
  LineChart,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const analysisReport = "doc/reports/2026-08-21-museg-mve-a1-b1-a2-git-alignment.md";
const dispositionReport = "doc/reports/2026-08-21-museg-mve-cleanup-and-disposition.md";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.5 · 阶段汇报</Pill>
          <Text size="small" tone="tertiary">2026-08-19 至 2026-08-21</Text>
        </Row>
        <H1>MUSeg / DFormerv2 MVE 结果与后续处置</H1>
        <Text tone="secondary">
          A1/B1 数值稳定性风险已经验证并修复；A2 只观察到轻微退化，未达到进入 B2 validity mask/gating 的门槛。当前优先事项是完成正式 MUSeg baseline。
        </Text>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => dispatch({ type: "openFile", path: dispositionReport })}>
            打开最新处置报告
          </Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: analysisReport })}>
            打开 A1/B1/A2 分析
          </Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="8/8" label="A1/B1 本地测试通过" tone="success" />
        <Stat value="20.49" label="epoch-10 全量验证 mIoU" tone="warning" />
        <Stat value="16 × 3" label="A2 样本 × 扰动条件" tone="info" />
        <Stat value="3" label="代码副本已对齐" tone="success" />
      </Grid>

      <Callout tone="success" title="阶段判断">
        保留 B1 安全归约修复；暂不实现 B2。epoch-10 checkpoint 和 16 图 A2 只用于最小可行实验筛查，不能作为正式性能结论。
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={16} align="start">
        <Stack gap={10}>
          <H2>A2：额外 Depth=0 扰动下的前景 mIoU</H2>
          <Text size="small" tone="tertiary">Y 轴：前景 mIoU（%）</Text>
          <LineChart
            categories={["q=0", "q=0.3", "q=0.5"]}
            series={[{
              name: "前景 mIoU",
              data: [33.8986, 33.7879, 33.5649],
              tone: "info",
            }]}
            height={250}
            valueSuffix="%"
            beginAtZero={false}
            yMin={33.4}
            yMax={34.0}
            showValues
          />
          <Text size="small" tone="tertiary">X 轴：新增有效深度置零比例 q</Text>
          <Text size="small" tone="quaternary">
            来源：epoch-10 DFormerv2-S checkpoint；官方 test 固定 16 张；seed 20260819；block mask；2026-08-21 独立评估。数值由报告中的 0–1 mIoU 转为百分比。
          </Text>
        </Stack>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>A2-N-G</Pill>}>
            效应量解释
          </CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Text><Code>q=0.3</Code>：下降 0.1107 个百分点。</Text>
              <Text><Code>q=0.5</Code>：下降 0.3337 个百分点。</Text>
              <Divider />
              <Text tone="secondary">
                下降随 q 单调增加，但幅度很小；当前证据不支持“Depth=0 是主要独立瓶颈”。
              </Text>
              <Callout tone="warning" title="证据边界">
                样本仅 16 张，checkpoint 仅训练到 epoch 10，且尚无 random mask、填充负对照和多 seed 复核。
              </Callout>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>问题、处理与验证状态</H2>
        <Table
          headers={["事项", "实际处理", "结果", "状态"]}
          rows={[
            ["A1：全 ignore 空集合", "复现旧 masked mean 的非有限结果", "风险边界已确认", "已完成并验证"],
            ["B1：安全损失归约", "空集合返回 values.sum() × 0；接入主头和辅助头", "8 项测试全部通过", "已完成并验证"],
            ["A2：Depth block mask", "q=0 / 0.3 / 0.5；16 张固定样本", "仅轻微退化，未过门槛", "已完成筛查"],
            ["B2：validity gating", "当前未实现", "缺少进入条件", "仅计划实施"],
          ]}
          rowTone={["warning", "success", "info", "neutral"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr 1fr" gap={14} align="start">
        <Card>
          <CardHeader trailing={<Pill size="sm" active>P0</Pill>}>正式 baseline</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>补做“全背景图 + 普通图”真实模型混合 batch 回归。</Text>
              <Text>使用提交 <Code>27437c1</Code> 完成正式训练、checkpoint 和完整 test 评估。</Text>
              <Text tone="secondary">epoch-10 screening 与正式结果分开报告。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">P1</Pill>}>条件性扩大 A2</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>仅在研究目标仍要求确认弱趋势时执行。</Text>
              <Text>扩大到 64 张或完整 test，增加 random mask、填充负对照和额外 seed。</Text>
              <Text tone="secondary">报告配对差值与置信区间。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">P2</Pill>}>门槛触发 B2</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>只有扩大 A2 达到预注册效应量门槛，才实现 validity mask/gating。</Text>
              <Text>先做固定 checkpoint 的零训练 A/B，再决定是否进行 5 epoch 微调。</Text>
              <Text tone="secondary">当前状态：尚未触发。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>代码与运行处置</H2>
        <Table
          headers={["对象", "当前状态", "说明"]}
          rows={[
            ["本地 / GitHub / 云端两份仓库", "已对齐", "统一到 27437c1；无提交分叉"],
            ["MVE 与 A2 产物", "已保留", "训练、预测、评估与证据包位于云端数据盘及本地证据目录"],
            ["云实例", "已停止", "最后核验 GPU=0；不启动新的训练或推理"],
            ["临时文档与 liu-test-exp", "保留", "未删除仍具复现、审计或草稿价值的内容"],
          ]}
          rowTone={["success", "success", "neutral", "info"]}
          striped
        />
      </Stack>

      <Callout tone="info" title="下一次执行入口">
        获取有 GPU 的实例后，先确认 CUDA、显卡与环境，再完成真实混合 batch 回归和正式 MUSeg baseline；不要回到旧提交 <Code>5c13409</Code>。
      </Callout>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        Canvas 0.0.5 · 事实源：2026-08-21 A1/B1/A2 Git 对齐分析与 MVE 清理处置报告 · 最新代码基线 27437c1ddf5ae6c8f5da05b7ae94fc6b29fc80af。
      </Text>
    </Stack>
  );
}

export default App;