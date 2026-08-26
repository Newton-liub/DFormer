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
  Pill,
  Row,
  Stack,
  Stat,
  Table,
  Text,
  useCanvasAction,
  useHostTheme,
} from "cursor/canvas";

const reportPath = "doc/reports/2026-08-19-museg-dformer-data-processing-review.md";

const labelPercentages = [
  50.9459, 0.1599, 5.6111, 4.0558, 2.1136, 2.2368, 1.2068, 6.9704,
  1.8583, 4.1974, 0.7995, 1.2193, 0.5225, 0.4876, 5.4265, 12.1885,
];

function App() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();

  return (
    <Stack gap={20} style={{ padding: 24, maxWidth: 1180, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.2 · 数据处理评审</Pill>
          <Text size="small" tone="tertiary">审计日期 2026-08-17</Text>
        </Row>
        <H1>MUSeg 接入 DFormer / DFormerv2</H1>
        <Text tone="secondary">
          基于 DFormerv2 原论文、MUSeg 数据集论文、当前代码链路与 3171 组数据的全量审计。
        </Text>
        <Row gap={8} wrap>
          <Button
            variant="primary"
            onClick={() => dispatch({ type: "openFile", path: reportPath })}
          >
            打开完整复现报告
          </Button>
          <Pill size="sm">原始 Depth16 已保留</Pill>
          <Pill size="sm">官方 group-disjoint split</Pill>
        </Row>
      </Stack>

      <Callout tone="success" title="总体判断：方向正确，但工程闭环尚未完成">
        固定全数据集线性量化 16-bit Depth 到 8-bit，符合当前 DFormer loader 与归一化接口；
        现有约 31.98 亿像素 100% 满足统一公式。必须继续解决无效深度、全 ignore 标签、
        转换脚本缺失和 MUSeg 配置缺失。
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value="3,171" label="完整 RGB-D-Label 样本" tone="success" />
        <Stat value="100%" label="Depth 量化公式匹配" tone="success" />
        <Stat value="30.74%" label="原始 Depth=0 无效像素" tone="warning" />
        <Stat value="11" label="全零标签图像" tone="danger" />
      </Grid>

      <Grid columns="1fr 1fr" gap={16}>
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>论文事实</Pill>}>
            MUSeg 原始数据
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>Depth16/*.png</Code> 是 16-bit，像素保存实际距离信息。</Text>
              <Text>值 0 由论文流程视为无效/缺失深度，不是最近距离。</Text>
              <Text>论文图中的 8-bit 深度仅用于可视化，不是发布格式。</Text>
              <Text>官方划分按拍摄位置组隔离：train 1595 / test 1576。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>代码事实</Pill>}>
            当前 DFormer 接口
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Depth 由 <Code>IMREAD_GRAYSCALE</Code> 按 8-bit 灰度链路读取。</Text>
              <Text>数值先除以 255，再以 mean 0.48 / std 0.28 标准化。</Text>
              <Text>DFormerv2 只取一个 Depth 通道，并直接计算 patch 深度差。</Text>
              <Text>因此直接切换到 16-bit 文件不会保留正确数值语义。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>证据链与判定</H2>
        <Table
          headers={["问题", "论文/数据证据", "当前实现", "判定"]}
          rows={[
            ["是否应保留 16-bit", "MUSeg 发布格式为 16-bit", "Depth16/ 已完整保留", "正确"],
            ["是否可生成 8-bit 输入", "DFormer 论文未限制位深", "Loader 与 /255 归一化要求 8-bit 接口", "兼容方案合理"],
            ["映射是否可靠", "深度差尺度必须跨图一致", "全局 max=13932；全像素公式匹配", "正确"],
            ["能否逐图 min-max", "会破坏跨样本统一尺度", "当前不是逐图归一化", "禁止"],
            ["能否直接 astype(uint8)", "会模 256 回绕", "低 8 位匹配率仅 0.40%", "已排除"],
            ["0 是否可当最近距离", "MUSeg 将 0 作为无效", "当前 geometry prior 无 mask", "必须验证/修正"],
            ["HHA 能否直接替换 Depth", "DFormerv2 论文使用 raw-depth prior", "模型只取 HHA 第一通道", "不正确"],
          ]}
          rowTone={["success", "success", "success", "danger", "danger", "warning", "danger"]}
          striped
        />
      </Stack>

      <Grid columns="minmax(0, 1.35fr) minmax(280px, 0.65fr)" gap={20} align="start">
        <Stack gap={8}>
          <H2>标签像素分布</H2>
          <H3>原始 Label ID 0–15 的全数据像素占比</H3>
          <BarChart
            categories={Array.from({ length: 16 }, (_, index) => String(index))}
            series={[{ name: "像素占比", data: labelPercentages }]}
            valueSuffix="%"
            height={310}
            showValues={false}
          />
          <Text size="small" tone="tertiary">
            横轴：原始 Label ID；纵轴：全数据像素占比（%）。来源：3171 张 Label 全量统计，2026-08-17。
            ID 0 占 50.95%，与论文“约一半像素为已标注类别”的描述一致。
          </Text>
        </Stack>

        <Stack gap={12}>
          <Callout tone="warning" title="标签训练风险">
            <Code>gt_transform=True</Code> 会把 0 映射为 255 ignore。11 张全零标签中 train 5 张、test 6 张；
            若本地 batch 全部 ignore，当前 <Code>.mean()</Code> 会产生 NaN。
          </Callout>
          <Callout tone="info" title="类别映射边界">
            数值上可暂用 0→255、1..15→0..14；类别名称顺序仍需用 MUSeg 官方 ID 映射核验，
            不能仅按论文列举顺序推断。
          </Callout>
        </Stack>
      </Grid>

      <Divider />

      <Stack gap={10}>
        <H2>问题、改动、原理与代码落点</H2>
        <Table
          headers={["优先级", "问题", "应做改动", "文件", "原理"]}
          rows={[
            ["P0", "转换过程未版本化", "新增确定性转换脚本和 metadata", "tools/prepare_museg.py", "固定 max_raw=13932；禁止逐图缩放"],
            ["P0", "无法自动验收数据", "新增只读审计脚本", "tools/audit_museg.py", "文件、位深、公式、split 全部断言"],
            ["P0", "没有 MUSeg 配置", "新增数据与模型配置", "local_configs/_base_/datasets/MUSeg.py", "显式声明 15 类、路径、尺寸与 ignore"],
            ["P0", "全 ignore batch 产生 NaN", "安全 masked mean", "models/builder.py", "空 valid mask 返回连图零损失"],
            ["P1", "无效 Depth=0 伪造几何", "填补或引入 validity mask", "models/encoders/DFormerv2.py", "无效 pair 不参与 depth decay"],
            ["P1", "RGB/BGR 隐式特判", "配置显式传入 rgb_mode", "utils/dataloader/RGBXDataset.py", "与 checkpoint 预处理一致"],
            ["P1", "验证尺寸不明确", "配置化 resize 或 sliding", "utils/dataloader/dataloader.py", "与 640×480 论文基线可比"],
            ["P1", "入口仍使用 NYUv2", "切到 MUSeg config", "train.sh / eval.sh / infer.sh", "避免错误类别数与路径"],
          ]}
          rowTone={["danger", "danger", "danger", "danger", "warning", "warning", "warning", "warning"]}
          striped
        />
      </Stack>

      <Grid columns="0.8fr 1.2fr" gap={20} align="start">
        <Card collapsible defaultOpen size="lg">
          <CardHeader trailing={<Pill size="sm">100% 已验证</Pill>}>
            当前量化公式
          </CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>D8 = round(D16 × 255 / 13932)</Code></Text>
              <Text size="small" tone="secondary">量化步长约 54.64 个原始深度单位。</Text>
              <Text size="small" tone="secondary">Depth=255 仅 388 像素，没有明显饱和。</Text>
              <Text size="small" tone="secondary">原始单位未在论文中给出，不应擅自解释为 mm。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Stack gap={10}>
          <H2>新项目复现流水线</H2>
          <Table
            headers={["阶段", "动作", "验收产物"]}
            rows={[
              ["1. 固定输入", "锁定官方数据版本、仓库 commit、标签映射", "source manifest"],
              ["2. 整理", "统一主文件名；保留 Depth16；复制 ID Label", "3171 组四模态"],
              ["3. 转换", "按固定公式生成 8-bit Depth", "dataset_meta.json"],
              ["4. 划分", "保留官方 group-disjoint train/test", "split hash + 0 组交集"],
              ["5. 审计", "全量检查位深、尺寸、公式、标签和 ignore", "audit report 全 PASS"],
              ["6. 训练", "MUSeg config + 明示 rgb_mode/eval/depth policy", "可追踪实验日志"],
            ]}
            rowTone={["info", "info", "info", "success", "success", "success"]}
          />
        </Stack>
      </Grid>

      <Stack gap={10}>
        <H2>建议消融顺序</H2>
        <Table
          headers={["实验", "Depth 表示", "无效值策略", "要回答的问题"]}
          rows={[
            ["B0", "当前 8-bit 全局线性", "保留 0", "建立最小兼容基线"],
            ["B1", "8-bit 全局线性", "深度填补", "无效 0 是否显著伤害模型"],
            ["B2", "8-bit 全局线性", "validity mask", "屏蔽伪几何边界是否有效"],
            ["B3", "16-bit/float 全局归一化", "validity mask", "8-bit 量化是否损失有效几何"],
            ["B4", "无 Depth", "不适用", "Depth 的净贡献"],
            ["B5", "真正的 HHA 分支", "按 HHA 实现", "与 MUSeg HHA 基线对照"],
          ]}
          rowTone={["info", "info", "info", "info", "neutral", "neutral"]}
          striped
        />
      </Stack>

      <Callout tone="neutral" title="组会一句话结论">
        “保留 16-bit 原始深度 + 固定全局线性量化为 8-bit 模型输入”的方向正确；
        当前仍需把转换和审计代码化，并优先解决 30.74% 无效深度对几何先验的污染。
      </Callout>

      <Text size="small" tone="quaternary" style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}>
        Canvas 版本 0.0.2 · 证据来源：DFormerv2 原论文 §3.1/§3.3/§4.1；MUSeg 数据集论文 Methods、Data Records、Technical Validation；
        当前 DFormer loader、归一化、geometry prior 与全量数据审计。完整路径与复现代码见 Markdown 报告。
      </Text>
    </Stack>
  );
}

export default App;