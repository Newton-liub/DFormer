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
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const reportPath = "doc/reports/2026-08-19-museg-minimum-validation-paths.md";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1160, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.4 · MVE 决策图</Pill>
          <Text size="small" tone="tertiary">设计日期 2026-08-19</Text>
        </Row>
        <H1>全背景样本与 Depth=0 最短验证路径</H1>
        <Text tone="secondary">
          先用受控注入证明问题，再用无参数或零参数级修复做最小 A/B；不直接启动完整训练。
          [RE001][RE003][RE009]
        </Text>
        <Button
          variant="primary"
          onClick={() => dispatch({ type: "openFile", path: reportPath })}
        >
          打开完整 MVE 与参考文献
        </Button>
      </Stack>

      <Grid columns={4} gap={16}>
        <Stat value="0 GPU" label="全背景风险验证" tone="success" />
        <Stat value="64" label="Depth MVE 图像" tone="info" />
        <Stat value="512" label="完整 Depth sweep 前向" tone="info" />
        <Stat value="5 epochs" label="仅过筛后最小微调" tone="warning" />
      </Grid>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>路径 A</Pill>}>
            问题—假设验证
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>把根本原因变为可控变量，不改模型权重。[RE009]</Text>
              <Text><Code>A1</Code>：有效像素从 1 pixel 降为 0 pixel。[RE001][RE002]</Text>
              <Text><Code>A2</Code>：新增 Depth=0 比例 q=0/0.1/0.3/0.5。[RE003][RE004]</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>路径 B</Pill>}>
            方案—假设验证
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>除目标修复外，checkpoint、split、seed 与评估保持一致。[RE009]</Text>
              <Text><Code>B1</Code>：空 valid mask 返回连图零 loss。[RE001]</Text>
              <Text><Code>B2</Code>：无效 depth pair 仅保留 positional decay。[RE003][RE004]</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>MVE 一：全背景样本</H2>
        <Table
          headers={["环节", "最小操作", "G：继续", "N-G：停止/转向"]}
          rows={[
            ["A1 问题验证", "100% / 1% / 1 pixel / 0% 有效标签", "仅 0% 产生非 finite loss/grad", "0% 仍 finite：核对代码版本"],
            ["B1 方案验证", "旧归约 vs safe masked loss", "正常梯度差 <1e-6；空集返回有限 0", "正常 batch 行为改变：修实现"],
          ]}
          rowTone={["warning", "success"]}
          striped
        />
      </Stack>

      <Callout tone="info" title="最小结论">
        不删除官方 11 张全背景图；safe loss 必须覆盖随机 crop 和每个分布式 rank 的本地全 ignore。
        [RE001][RE006]
      </Callout>

      <Divider />

      <Stack gap={10}>
        <H2>MVE 二：Depth=0</H2>
        <Table
          headers={["环节", "最小操作", "G：继续", "N-G：停止/转向"]}
          rows={[
            ["A2 问题验证", "64 图；block/random；q=0/0.1/0.3/0.5", "q=0.3 block 下降 ≥2.0 pp；≥75% 图变差", "q=0.5 下降 <0.5 pp：非优先瓶颈"],
            ["B2 零训练筛选", "同 checkpoint：B0 vs pair-valid gating", "q=0.3 恢复 ≥1.0 pp；干净下降 ≤0.5 pp", "无方向性收益：不微调"],
            ["B2 最小微调", "固定 20% grouped train；同 seed；各 5 epoch", "自然高缺失组 +1.0 pp 或 q=0.3 +2.0 pp", "均 <0.5 pp 或全 test -0.5 pp：转填补"],
          ]}
          rowTone={["warning", "info", "success"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Callout tone="warning" title="必须保留的负对照">
          同样的缺失掩码分别填 0 和有效深度中位数；只有 0 更差，才支持“无效 0 被解释为真实极近深度”的机制。
          [RE003][RE004][RE009]
        </Callout>
        <Callout tone="neutral" title="不能过度宣称">
          如果 mask 只改善人工 block、不改善自然高缺失组，只能宣称鲁棒性可行，不能宣称解决了 MUSeg 自然瓶颈。
          [RE005][RE009]
        </Callout>
      </Grid>

      <Stack gap={10}>
        <H2>行动决策</H2>
        <Table
          headers={["路径结果", "下一步"]}
          rows={[
            ["A1-G / B1-G", "立即合入 safe loss；作为稳定性修复，不包装成创新"],
            ["A2-G / B2-G", "进入完整 train、3 seeds 与 validity mask 消融"],
            ["A2-G / B2-N-G", "只测试一种简单填补，不扩展复杂 mask 网络"],
            ["A2-N-G / B2-N-G", "终止 Depth=0 主线，资源转向正式训练复现"],
            ["A2-N-G / B2-G", "视为可能正则化；补自然分层和第二 seed"],
          ]}
          rowTone={["success", "success", "warning", "danger", "warning"]}
        />
      </Stack>

      <Callout tone="success" title="精益顺序">
        CPU 单元测试 → 现有 checkpoint 推理 sweep → 只有出现正向信号才做两个 5-epoch 小子集微调。
        [RE001][RE003][RE009]
      </Callout>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        Canvas 0.0.4 · 参考文献 RE001–RE009 见 Markdown 正文 ·
        事实边界：当前代码、MUSeg 全量审计、DFormerv2/MUSeg 论文与 PyTorch 官方文档。
      </Text>
    </Stack>
  );
}

export default App;