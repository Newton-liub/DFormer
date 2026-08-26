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
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const projectRoot = "d:\\0Project\\DFormer";
const auditPath = `${projectRoot}\\doc\\audits\\2026-08-26-markdown-consistency-audit.md`;
const currentStatusPath = `${projectRoot}\\doc\\main\\MUSeg-current-status.md`;
const openDecisionsPath = `${projectRoot}\\doc\\main\\MUSeg-open-decisions.md`;
const handoffPath = `${projectRoot}\\doc\\reports\\2026-08-26-museg-stage05-seed1-running-handoff.md`;
const protocolPath = `${projectRoot}\\protocols\\museg-development-long500-v2.template.json`;

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  const openFile = (path: string) => dispatch({ type: "openFile", path });

  return (
    <Stack gap={24} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.9 · Markdown 一致性审计</Pill>
          <Text size="small" tone="tertiary">审计时点：2026-08-26 03:33 UTC</Text>
        </Row>
        <H1>MUSeg 文档事实链已统一</H1>
        <Text tone="secondary">
          当前状态、历史快照、运行协议和未决研究口径已经分离。以下内容冻结于审计时点，不代表训练完成后的当前状态。
        </Text>
        <Callout tone="warning" title="历史运行中快照">
          本页冻结于 2026-08-26 03:33 UTC，当时 Stage-05 seed 1 仍在运行。训练完成后的实时事实请以 <Code>doc/main/MUSeg-current-status.md</Code> 为准；本页不改写为最终结果。
        </Callout>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => openFile(currentStatusPath)}>打开当前状态</Button>
          <Button variant="secondary" onClick={() => openFile(auditPath)}>打开完整审计</Button>
          <Button variant="secondary" onClick={() => openFile(handoffPath)}>打开运行中交接</Button>
          <Button variant="secondary" onClick={() => openFile(openDecisionsPath)}>打开未决问题</Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="1" label="唯一当前入口" tone="success" />
        <Stat value="12" label="已处置冲突类别" tone="info" />
        <Stat value="5" label="待讨论研究口径" tone="warning" />
        <Stat value="sealed" label="official test" tone="success" />
      </Grid>

      <Callout tone="success" title="当前权威结论">
        Stage-04 Gate D 已完成。Stage-05 seed 1 绑定 Git 提交 <Code>56a7ed7…</Code> 与 protocol <Code>museg-development-long500-v2</Code> 正在运行；native preflight 和专项审计均通过。运行中指标不等于最终 baseline。
      </Callout>

      <Grid columns="1.15fr 0.85fr" gap={18} align="start">
        <Stack gap={12}>
          <Row align="center" justify="space-between" wrap>
            <H2>当前权威链路</H2>
            <Pill size="sm" active>从状态到证据</Pill>
          </Row>
          <Stack gap={0} style={{ borderTop: `1px solid ${theme.stroke.tertiary}` }}>
            <Row gap={12} align="start" style={{ padding: "14px 0", borderBottom: `1px solid ${theme.stroke.tertiary}` }}>
              <Pill size="sm" active>1</Pill>
              <Stack gap={4}>
                <Text weight="semibold">当前状态与恢复点</Text>
                <Text tone="secondary"><Code>doc/main/MUSeg-current-status.md</Code> 是唯一实时入口。</Text>
              </Stack>
            </Row>
            <Row gap={12} align="start" style={{ padding: "14px 0", borderBottom: `1px solid ${theme.stroke.tertiary}` }}>
              <Pill size="sm">2</Pill>
              <Stack gap={4}>
                <Text weight="semibold">运行中正式证据</Text>
                <Text tone="secondary">Stage-05 seed 1 running handoff 冻结 03:33 UTC 的只读核验。</Text>
              </Stack>
            </Row>
            <Row gap={12} align="start" style={{ padding: "14px 0", borderBottom: `1px solid ${theme.stroke.tertiary}` }}>
              <Pill size="sm">3</Pill>
              <Stack gap={4}>
                <Text weight="semibold">可移植协议事实源</Text>
                <Text tone="secondary"><Code>museg-development-long500-v2.template.json</Code> 保留 run commit 与 materialization 占位符；原始 SHA 和云端证据路径由 handoff 与审计保留。</Text>
              </Stack>
            </Row>
            <Row gap={12} align="start" style={{ padding: "14px 0", borderBottom: `1px solid ${theme.stroke.tertiary}` }}>
              <Pill size="sm">4</Pill>
              <Stack gap={4}>
                <Text weight="semibold">冻结数据边界与未决决策</Text>
                <Text tone="secondary">dev-v1 split 约束样本角色；open decisions 独立登记尚未冻结的研究口径。</Text>
              </Stack>
            </Row>
          </Stack>
        </Stack>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>运行中</Pill>}>Stage-05 seed 1 快照</CardHeader>
          <CardBody>
            <Stack gap={10}>
              <Row justify="space-between" wrap><Text tone="secondary">已完成 epoch</Text><Text weight="semibold">445 / 500</Text></Row>
              <Row justify="space-between" wrap><Text tone="secondary">当前 best val-dev mIoU</Text><Text weight="semibold">52.41</Text></Row>
              <Row justify="space-between" wrap><Text tone="secondary">best 观测点</Text><Text>epoch 440</Text></Row>
              <Row justify="space-between" wrap><Text tone="secondary">Git 工作区</Text><Text>clean</Text></Row>
              <Row justify="space-between" wrap><Text tone="secondary">preflight</Text><Text>0 errors · 0 warnings</Text></Row>
              <Divider />
              <Text size="small" tone="tertiary">52.41 仅是只读核验时的运行中观察值。训练结束、验收器通过和三个 seeds 汇总前，不构成最终 baseline。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Divider />

      <Stack gap={14}>
        <H2>冲突处置结果</H2>
        <Grid columns="1fr 1fr" gap={14} align="start">
          <Card>
            <CardHeader trailing={<Pill size="sm" active>4 项</Pill>}>状态与实验边界</CardHeader>
            <CardBody><Stack gap={8}>
              <Text><Code>当前入口</Code> 取代仍写 Gate C/D 未完成的旧阶段说明。</Text>
              <Text><Code>report-index</Code> 的最新正式报告推进到运行中交接。</Text>
              <Text>旧 20-epoch 与 test-64 A2 路径标为禁用历史入口。</Text>
              <Text>正式 A2/B2 开发筛查只使用 <Code>val-dev</Code>。</Text>
            </Stack></CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm">4 项</Pill>}>协议与统计语义</CardHeader>
            <CardBody><Stack gap={8}>
              <Text>500-epoch 运行事实以 materialized protocol 为准。</Text>
              <Text>resume 子 run 允许新 <Code>run_id</Code>，父 run 身份仍严格核验。</Text>
              <Text>qualification 与 development 均可保存 best validation checkpoint。</Text>
              <Text>1916 个官方位置与本地 1915 groups 分别保留来源。</Text>
            </Stack></CardBody>
          </Card>
        </Grid>
        <Grid columns="1fr 1fr" gap={14} align="start">
          <Stack gap={8} style={{ padding: "4px 2px" }}>
            <H3>工程说明修复</H3>
            <Text tone="secondary">README 已区分 MUSeg 审计工作流与上游论文复现；数据目录、依赖、benchmark、GPU 脚本和新数据集模板已按实际入口修正。</Text>
            <Text tone="secondary">OpenList 启动参数、进程匹配、密码闭环与解压路径已修正；FileBrowser 增加版本、权限、代理和生命周期边界。</Text>
          </Stack>
          <Stack gap={8} style={{ padding: "4px 2px" }}>
            <H3>历史材料保留</H3>
            <Text tone="secondary">Stage-01 至 Stage-04、旧 MVE 与 prelaunch 报告保留当时事实，并增加“历史快照/已被替代”标识，不用当前结论重写过去。</Text>
            <Text tone="secondary">Canvas 0.0.1–0.0.7 归档到 <Code>doc/canvases/old/</Code>；索引保留为 reference。</Text>
          </Stack>
        </Grid>
      </Stack>

      <Callout tone="warning" title="五项决定保持开放，当前训练不变更">
        <Stack gap={6}>
          <Text>1. validation 使用原分辨率整图、固定 480×640 resize，还是 480×640 sliding-window；</Text>
          <Text>2. MUSeg 保持当前 BGR，还是后续切换 RGB；</Text>
          <Text>3. 自然无效深度分层是否为进入 B2 的硬门槛；</Text>
          <Text>4. qualification 的 376 successful updates 与 384 planned iterations 如何解释；</Text>
          <Text>5. 是否重命名历史字段 <Code>run_kind=qualification</Code>。</Text>
        </Stack>
      </Callout>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card>
          <CardHeader trailing={<Pill size="sm" active>保持封存</Pill>}>Official test 边界</CardHeader>
          <CardBody><Stack gap={8}>
            <Text>development phase 不建立 test loader，不读取、抽样或评估 official test。</Text>
            <Text>test 路径、数量和 SHA 只用于身份约束，不代表已读取样本。</Text>
            <Text>最终模型和协议冻结后，才可经独立门禁一次性解封。</Text>
          </Stack></CardBody>
        </Card>
        <Card>
          <CardHeader trailing={<Pill size="sm">下一门禁</Pill>}>Seed 1 完成后</CardHeader>
          <CardBody><Stack gap={8}>
            <Text>核验退出码、<Code>training_result.json</Code> 与验收器结果。</Text>
            <Text>核验 50 个 validation 点、12 个 checkpoint 和 best/latest 身份。</Text>
            <Text>核验 SwanLab 记录与 <Code>official_test_included=false</Code>，再进入 Gate E；seed 2/3 不自动启动。</Text>
          </Stack></CardBody>
        </Card>
      </Grid>

      <Row gap={8} wrap>
        <Button variant="secondary" onClick={() => openFile(protocolPath)}>查看 v2 protocol 模板</Button>
        <Button variant="secondary" onClick={() => openFile(auditPath)}>复核全部 12 类冲突</Button>
      </Row>

      <Text size="small" tone="quaternary" style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}>
        事实源：2026-08-26 Markdown 一致性审计、MUSeg 当前状态、Stage-05 seed 1 运行中交接、MUSeg 待讨论实验口径。Canvas 只重排已核实事实，不增加实验结论。
      </Text>
    </Stack>
  );
}

export default App;
