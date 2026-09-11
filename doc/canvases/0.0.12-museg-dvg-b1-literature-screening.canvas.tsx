import {
  Button,
  Callout,
  Card,
  CardBody,
  CardHeader,
  Divider,
  Grid,
  H1,
  H2,
  H3,
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
  useState,
} from "cursor/canvas";

const aCandidates = [
  ["P0", "Confidence Propagation through CNNs", "WOS:000567471300008", "跨层 confidence recurrence、部分有效支持域、全 1 行为"],
  ["P0", "Bcap-net", "WOS:001778286700014", "多尺度 weighted pooling 的权重、归一化与空 patch"],
  ["P0", "LightDepth", "WOS:001307644900001", "sparse depth resize 插值模式与 validity 同步"],
  ["P0", "Hierarchical reconstruction and dynamic weight sampling", "WOS:001565304100002", "多级 confidence map 的来源和尺度变换"],
  ["P0", "Octagram Propagation Matching", "WOS:001492121500010", "coarse-to-fine confidence 传播公式"],
  ["P1", "Integrated RGBD Perception for Forklifts", "WOS:001786002300042", "mask-aware patch-median 与 missing-value fallback"],
  ["P1", "Guided Depth Inpainting in ToF", "WOS:001420327300001", "invalid mask 与有效邻域传播"],
];

const bCandidates = [
  ["P0", "Non-local affinity adaptive propagation", "WOS:001026313400003", "pixel reliability 与 neighbor affinity 的组合公式"],
  ["P0", "Deep Sparse Depth Completion Using Multi-Affinity Matrix", "WOS:001042003300001", "confidence 是否直接进入 pairwise affinity"],
  ["P0", "Gaussian Splatting Confidence Supervision", "WOS:001826397300001", "SPN confidence weight 在 propagation 方程中的位置"],
  ["P0", "NR-MVSNet", "WOS:000988473800002", "reliable attention 是否修改 cost-volume score"],
  ["P0", "LFDA", "WOS:001219298900001", "center/side view attention score 的方向性与对称性"],
  ["P0", "Selective Mutual Attention and Contrast", "WOS:000880661400035", "depth cue reweight 是否作用于 non-local affinity"],
  ["P1", "DGQ-YOLO", "WOS:001858702300020", "feature gate 与 pairwise score gate 的边界"],
];

export default function LiteratureScreeningCanvas() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  const [tab, setTab] = useState<"A" | "B">("A");
  const candidates = tab === "A" ? aCandidates : bCandidates;

  return (
    <Stack
      gap={20}
      style={{
        maxWidth: 1180,
        margin: "0 auto",
        padding: 24,
        color: theme.text.primary,
        background: theme.bg.editor,
      }}
    >
      <Row justify="space-between" align="start" wrap gap={16}>
        <Stack gap={6} style={{ maxWidth: 760 }}>
          <Text size="small" tone="tertiary">MUSeg · DVG-B1 · 文献摘要筛选 · Canvas 0.0.12</Text>
          <H1>A-1/B-1 检索结果能否关闭门禁</H1>
          <Text tone="secondary">
            73 条唯一 WOS 记录提供了相关术语和全文候选，但摘要尚不能冻结像素到 token reliability 或 token 到 pairwise gate 的唯一公式。
          </Text>
        </Stack>
        <Row gap={8} wrap>
          <Button
            variant="primary"
            onClick={() => dispatch({ type: "openFile", path: "liu-test-exp/方案1/DVG-B1-A1-B1摘要筛选与全文优先级.md" })}
          >
            打开详细报告
          </Button>
          <Button
            variant="secondary"
            onClick={() => dispatch({ type: "openFile", path: "liu-test-exp/附件/A-1.txt" })}
          >
            打开 A-1
          </Button>
          <Button
            variant="secondary"
            onClick={() => dispatch({ type: "openFile", path: "liu-test-exp/附件/B-1.txt" })}
          >
            打开 B-1
          </Button>
        </Row>
      </Row>

      <Callout tone="warning" title="当前裁决">
        A、B 均保持 reference-blocked。摘要可以确定下一批全文优先级，不能直接选择 nearest、average、valid fraction、乘积、最小值或 hard/continuous gate。
      </Callout>

      <Grid columns="1.7fr 1fr" gap={20} align="stretch">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>未关闭</Pill>}>门禁覆盖情况</CardHeader>
          <CardBody>
            <Stack gap={12}>
              <Row gap={24} wrap>
                <Stat value="44" label="A-1 记录" />
                <Stat value="30" label="B-1 记录" />
                <Stat value="73" label="唯一 WOS ID" />
                <Stat value="0" label="摘要可完整关闭的门禁" tone="warning" />
              </Row>
              <Divider />
              <Grid columns={2} gap={14}>
                <Stack gap={5}>
                  <Text weight="semibold">A 已获得的线索</Text>
                  <Text size="small" tone="secondary">weighted pooling、confidence propagation、valid-only mean/median、sparse depth resize。</Text>
                </Stack>
                <Stack gap={5}>
                  <Text weight="semibold">A 仍缺</Text>
                  <Text size="small" tone="secondary">四级确定性聚合、部分有效 patch、bilinear 对齐、全可信恒等。</Text>
                </Stack>
                <Stack gap={5}>
                  <Text weight="semibold">B 已获得的线索</Text>
                  <Text size="small" tone="secondary">pixel reliability 可与 neighbor affinity 结合并调整传播权重。</Text>
                </Stack>
                <Stack gap={5}>
                  <Text weight="semibold">B 仍缺</Text>
                  <Text size="small" tone="secondary">两端组合公式、对称性、hard/continuous、全 1 恒等和 Full/H/W 映射。</Text>
                </Stack>
              </Grid>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={12}>
          <H2>最重要发现</H2>
          <Text>
            <Text weight="semibold">最接近 A：</Text> Confidence Propagation through CNNs 与 Bcap-net。
          </Text>
          <Text>
            <Text weight="semibold">最接近 B：</Text> Non-local affinity adaptive propagation；它明确把 pixel depth reliability 与 normalized neighbor affinity 结合。
          </Text>
          <Text>
            <Text weight="semibold">检索文件正常：</Text> 两文件仅共享 BurnDC 一条记录，不是同一批结果复制。
          </Text>
        </Stack>
      </Grid>

      <Divider />

      <Stack gap={12}>
        <Row justify="space-between" align="center" wrap gap={12}>
          <H2>全文优先候选</H2>
          <Row gap={8}>
            <Pill active={tab === "A"} onClick={() => setTab("A")}>开放项 A</Pill>
            <Pill active={tab === "B"} onClick={() => setTab("B")}>开放项 B</Pill>
          </Row>
        </Row>
        <Table
          headers={["优先级", "文献", "WOS ID", "全文必须确认"]}
          rows={candidates}
          rowTone={candidates.map((row) => row[0] === "P0" ? "warning" : "info")}
          striped
          stickyHeader
          columnAlign={["center", "left", "left", "left"]}
        />
      </Stack>

      <Grid columns={2} gap={20}>
        <Stack gap={10}>
          <H3>A 的全文提取字段</H3>
          <Text size="small" tone="secondary">输入 validity/confidence 定义；resize/downsample 算子；部分有效与空窗口；归一化分母；像素中心对齐；全 1 行为；官方代码位置。</Text>
        </Stack>
        <Stack gap={10}>
          <H3>B 的全文提取字段</H3>
          <Text size="small" tone="secondary">中心/邻居或 query/key reliability；pairwise 组合式；归一化顺序；方向性与对称性；hard/continuous；全可信恒等；作用于 score、bias、affinity 还是 feature。</Text>
        </Stack>
      </Grid>

      <Callout tone="info" title="准确恢复点">
        先获取 P0 全文和官方代码；只有得到唯一、可复现并可映射到 DFormerv2 四级 shape 的规则后，才回填原检索文档并考虑创建 protocol。
      </Callout>

      <Text size="small" tone="tertiary">
        来源：A-1.txt、B-1.txt 的题录与摘要 · 筛选日期：2026-09-10 · 未读取全文、未运行模型或实验
      </Text>
    </Stack>
  );
}
