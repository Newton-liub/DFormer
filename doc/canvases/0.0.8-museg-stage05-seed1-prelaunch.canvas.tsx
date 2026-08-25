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
  Pill,
  Row,
  Stack,
  Stat,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const reportPath = "/root/cloud-ssd/museg-stage05-development/reports/2026-08-25-seed1-prelaunch.md";
const protocolPath = "/root/cloud-ssd/museg-stage05-development/protocols/museg-development-long500-v1.json";
const screenScript = "/root/cloud-ssd/museg-stage05-development/scripts/start_seed1_screen.sh";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.8 · Stage-05 seed 1 启动前</Pill>
          <Text size="small" tone="tertiary">状态：等待人工授权</Text>
        </Row>
        <H1>MUSeg Stage-05 development</H1>
        <Text tone="secondary">
          protocol 已冻结，完整 preflight 通过；本轮没有启动训练，也没有读取 official test。
        </Text>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => dispatch({ type: "openFile", path: reportPath })}>打开启动前报告</Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: protocolPath })}>打开 protocol</Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: screenScript })}>打开 screen 脚本</Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="PASS" label="完整 preflight" tone="success" />
        <Stat value="500" label="epochs" tone="info" />
        <Stat value="22.37 h" label="线性时间下限" tone="warning" />
        <Stat value="3.59 GiB" label="12 个 checkpoint" tone="info" />
      </Grid>

      <Callout tone="success" title="Gate D 结论">
        batch 10 连续 qualification 退出码为 0，完成 376 个 optimizer steps；resume rehearsal 的 epoch 1 与 epoch 3 均逐字段一致，且证据记录 official test 未包含。该结论证明训练与恢复链路具备工程资格，不代表模型性能结论。
      </Callout>

      <Stack gap={10}>
        <H2>冻结 protocol</H2>
        <Grid columns="1fr 1fr 1fr" gap={14} align="start">
          <Card>
            <CardHeader trailing={<Pill size="sm" active>优化</Pill>}>训练与学习率</CardHeader>
            <CardBody><Stack gap={8}>
              <Text><Code>DFormerv2-S</Code> · 输入 640×480 · batch 10</Text>
              <Text>AdamW · base LR <Code>6e-5</Code> · weight decay 0.01</Text>
              <Text>WarmUpPolyLR · warmup 2 epochs / 256 steps · power 0.9</Text>
              <Text tone="secondary">总 schedule 绑定 64,000 steps，避免把中途 checkpoint 当作独立短 schedule。</Text>
            </Stack></CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" active>观察</Pill>}>验证与保存</CardHeader>
            <CardBody><Stack gap={8}>
              <Text>epoch 10 起，每 10 epochs 验证，共 50 个点</Text>
              <Text>每 50 epochs 保存 periodic checkpoint</Text>
              <Text><Code>latest.pth</Code> 用于恢复；<Code>best-val-miou.pth</Code> 只按 val 前景 mIoU 严格改善</Text>
              <Text tone="secondary">相等指标保留最早已评估 epoch；自动 early stop 关闭。</Text>
            </Stack></CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" active>重复</Pill>}>预注册 seeds</CardHeader>
            <CardBody><Stack gap={8}>
              <Text>seed 1：<Code>772961337</Code></Text>
              <Text>seed 2：<Code>1101528019</Code></Text>
              <Text>seed 3：<Code>1126246545</Code></Text>
              <Text tone="secondary">本轮只允许启动 seed 1；seed 2/3 等待后续审查与授权。</Text>
            </Stack></CardBody>
          </Card>
        </Grid>
      </Stack>

      <Divider />

      <Grid columns="1.1fr 0.9fr" gap={16} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>成本估计</Pill>}>时间与磁盘</CardHeader>
          <CardBody><Stack gap={9}>
            <Text>Stage-04 3 epoch 端到端实测 <Code>483.13936644996284 s</Code>。</Text>
            <Text>线性下限：<Code>500 / 3 × 483.13936644996284 s = 22.37 h</Code>。</Text>
            <Text>建议为验证、I/O、SwanLab 和波动预留 <Code>24–28 h</Code>；15% 余量约 25.72 h。</Text>
            <Text>12 个 checkpoint 实测合计 <Code>3,851,769,357 bytes</Code>，约 3.59 GiB；含日志等按 4.5 GB 规划。</Text>
            <Text tone="secondary">当前可用空间约 35.97 GiB，满足专项审计要求的至少 15 GiB 余量。</Text>
          </Stack></CardBody>
        </Card>
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">计费边界</Pill>}>关机不等于已确认停止计费</CardHeader>
          <CardBody><Stack gap={9}>
            <Text>平台处于 <Code>Stopped</Code> 后，官方文档说明 CPU、GPU 和内存算力计费停止。</Text>
            <Text>云盘与镜像仍可能计费。</Text>
            <Text>当前无法从本机 CLI 读取实例订单和平台状态；guest OS 的 <Code>shutdown -h</Code> 是否必然映射为平台 Stopped 需要人工确认。</Text>
            <Callout tone="warning" title="操作要求">训练成功、结构化证据通过、sync 完成后才安排 5 分钟延时关机；失败路径写现场并禁止自动关机。关机后请在控制台确认 Stopped。</Callout>
          </Stack></CardBody>
        </Card>
      </Grid>

      <Callout tone="info" title="人工启动命令">
        <Code>/root/cloud-ssd/museg-stage05-development/scripts/start_seed1_screen.sh</Code>
        <Text tone="secondary">启动后可用 <Code>screen -r museg-stage05-seed1</Code> 查看；本轮没有执行该命令。</Text>
      </Callout>

      <Text size="small" tone="quaternary" style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}>
        事实源：Stage-04 Gate D 结构化 JSON、Stage-05 protocol、Stage-05 preflight audit、2026-08-25 启动前报告。official test 保持 sealed_unread。
      </Text>
    </Stack>
  );
}

export default App;