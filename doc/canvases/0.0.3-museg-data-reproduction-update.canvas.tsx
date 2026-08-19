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

const reportPath = "doc/reports/2026-08-19-museg-data-reproduction-update.md";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1160, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.3 · 数据复现更新</Pill>
          <Text size="small" tone="tertiary">汇报周期 2026-08-17 至 2026-08-19</Text>
        </Row>
        <H1>MUSeg 数据复现与标签语义</H1>
        <Text tone="secondary">
          转换过程已代码化并通过 3171 组全量验证；标签 0 与 1–15 的语义已经确认。
        </Text>
        <Row gap={8} wrap>
          <Button
            variant="primary"
            onClick={() => dispatch({ type: "openFile", path: reportPath })}
          >
            打开正式报告
          </Button>
          <Pill size="sm">转换脚本已落地</Pill>
          <Pill size="sm">标签映射已确认</Pill>
          <Pill size="sm">训练配置待补</Pill>
        </Row>
      </Stack>

      <Callout tone="success" title="当前结论">
        数据转换复现闭环已经完成：本地从官方六矿区原始目录重建 MUSeg_DFormer，
        四模态、位深、尺寸、官方划分、标签 ID 和全部深度像素映射均已验证。
        当前尚不能宣称训练闭环完成。
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value="3,171" label="全量转换样本" tone="success" />
        <Stat value="1,595 / 1,576" label="官方 train / test" tone="success" />
        <Stat value="13,932" label="固定原始深度最大值" tone="success" />
        <Stat value="0–15" label="已确认原始标签 ID" tone="info" />
      </Grid>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>原始数据语义</Pill>}>
            Label ID
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>0</Code> 是 background。</Text>
              <Text><Code>1–15</Code> 是 15 个前景类别。</Text>
              <Text tone="secondary">
                类别顺序由官方 Label_ID.pdf 确认，不再是暂定映射。
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>训练张量语义</Pill>}>
            DFormer 映射
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>gt_transform=True</Code> 执行 uint8 减 1。</Text>
              <Text><Code>0 → 255</Code>，损失中作为 ignore。</Text>
              <Text><Code>1–15 → 0–14</Code>，作为 15 个训练类别。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>15 个前景类别顺序</H2>
        <Table
          headers={["原始 ID", "类别", "训练 ID", "原始 ID", "类别", "训练 ID"]}
          rows={[
            ["1", "person", "0", "9", "electrical equipment", "8"],
            ["2", "cable", "1", "10", "electronic equipment", "9"],
            ["3", "tube", "2", "11", "mining equipment", "10"],
            ["4", "indicator", "3", "12", "anchoring equipment", "11"],
            ["5", "metal fixture", "4", "13", "support equipment", "12"],
            ["6", "container", "5", "14", "rescue equipment", "13"],
            ["7", "tools & materials", "6", "15", "rail area", "14"],
            ["8", "door", "7", "0", "background", "255 / ignore"],
          ]}
          striped
        />
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>本阶段状态变化</H2>
        <Table
          headers={["事项", "旧状态", "当前状态", "证据"]}
          rows={[
            ["16→8 转换", "只有处理结果", "脚本已实现并验证", "tools/prepare_museg.py"],
            ["输出审计", "建议新增独立脚本", "转换内置全量验证", "verify_output()"],
            ["标签映射", "暂定，名称顺序待核验", "0 与 1–15 已确认", "官方 Label_ID.pdf"],
            ["云端重建", "无统一流程", "base 环境指南已完成", "doc/云/MUSeg云端重建指南.md"],
            ["MUSeg 训练", "未完成", "仍未完成", "缺专属 config"],
          ]}
          rowTone={["success", "success", "success", "success", "warning"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Callout tone="warning" title="全背景样本风险">
          11 张标签图全部为原始 ID 0，应称为全背景图。经过 gt_transform 后才成为全 ignore；
          train 5 张、test 6 张。空有效像素 batch 仍可能导致当前 loss 产生 NaN。
        </Callout>
        <Callout tone="warning" title="Depth=0 是另一类问题">
          深度 0 表示无效/缺失深度，和 Label 0 的 background 完全不同。
          当前 DFormerv2 无 validity mask，仍会把深度 0 当普通数值参与 geometry prior。
        </Callout>
      </Grid>

      <Stack gap={10}>
        <H2>下一步执行顺序</H2>
        <Table
          headers={["顺序", "任务", "完成判据"]}
          rows={[
            ["1", "新增 MUSeg 数据与 DFormerv2-S 配置", "路径、15 类、ignore、尺寸可加载"],
            ["2", "修复全背景 batch 空均值", "单样本和分布式本地 batch 不产生 NaN"],
            ["3", "从官方 train 按采集组划 validation", "组无交集，test 保持封存"],
            ["4", "建立 Depth=0 保留方案 B0", "配置、日志、split hash 可追踪"],
            ["5", "验证填补和 validity mask", "与 B0 做同配置消融"],
          ]}
          rowTone={["info", "warning", "info", "info", "info"]}
        />
      </Stack>

      <Callout tone="neutral" title="一句话总结">
        标签语义和数据转换已经从“待确认、不可复现”推进到“已确认、可重建并全量验证”；
        下一阶段应集中完成训练配置、全背景 loss 安全性和无效深度消融。
      </Callout>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        Canvas 版本 0.0.3 · Markdown 事实源：doc/reports/2026-08-19-museg-data-reproduction-update.md ·
        生成日期 2026-08-19
      </Text>
    </Stack>
  );
}

export default App;