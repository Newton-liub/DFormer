import {
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
  useHostTheme,
} from "cursor/canvas";

const comparisonRows = [
  ["单次提示词", "低", "低", "弱", "临时试验或一次性汇报", "不作为主方案"],
  ["项目规则", "中", "高", "中", "约束所有对话中的通用行为", "范围过宽"],
  ["项目 Skill", "高", "高", "强", "按需收集证据并生成报告", "主方案"],
  ["自动化脚本", "高", "中", "强", "抽取固定格式日志和 Git 信息", "按痛点添加"],
];

function ArchitectureStep({
  index,
  title,
  detail,
}: {
  index: string;
  title: string;
  detail: string;
}) {
  const theme = useHostTheme();
  return (
    <div
      style={{
        background: theme.fill.tertiary,
        border: `1px solid ${theme.stroke.tertiary}`,
        borderRadius: 8,
        padding: 14,
        minHeight: 112,
      }}
    >
      <Row gap={8} align="center">
        <Pill size="sm" active>{index}</Pill>
        <Text weight="semibold">{title}</Text>
      </Row>
      <Text tone="secondary" size="small" style={{ marginTop: 10 }}>{detail}</Text>
    </div>
  );
}

export default function ReportWorkflowDecision() {
  const theme = useHostTheme();

  return (
    <Stack
      gap={24}
      style={{
        maxWidth: 1120,
        margin: "0 auto",
        padding: 28,
        background: theme.bg.editor,
        color: theme.text.primary,
      }}
    >
      <Stack gap={8}>
        <Row justify="space-between" align="center" wrap>
          <H1>DFormer 后续汇报工作流选型</H1>
          <Pill active>v0.0.1 · 分析结论</Pill>
        </Row>
        <Text tone="secondary">
          目标不是只生成一篇好看的报告，而是建立可重复、可追溯、能区分事实与计划的科研进展汇报流程。
        </Text>
      </Stack>

      <Card size="lg">
        <CardHeader trailing={<Pill size="sm" active>推荐</Pill>}>
          项目 Skill 为核心，显式提示词为入口，Markdown 与 Canvas 分工输出
        </CardHeader>
        <CardBody>
          <Grid columns="1.6fr 1fr" gap={20} align="center">
            <Stack gap={10}>
              <Text weight="semibold">不在“Skill 或提示词”之间二选一。</Text>
              <Text tone="secondary">
                Skill 固化证据边界、状态分类和执行步骤；短提示词声明本次范围与受众；Markdown 保存可审阅正文；Canvas 只承担可视化汇报视图；轻量检查点记录上次正式报告边界。
              </Text>
            </Stack>
            <Grid columns={2} gap={12}>
              <Stat value="1" label="核心 Skill" tone="info" />
              <Stat value="2" label="输出层" />
              <Stat value="1" label="边界检查点" />
              <Stat value="按需" label="辅助脚本" />
            </Grid>
          </Grid>
        </CardBody>
      </Card>

      <Stack gap={12}>
        <H2>为什么不是单一形式</H2>
        <Table
          headers={["形式", "复用性", "项目一致性", "证据约束", "适用场景", "本项目判断"]}
          rows={comparisonRows}
          rowTone={["warning", "neutral", "success", "info"]}
          striped
        />
      </Stack>

      <Divider />

      <Stack gap={14}>
        <H2>建议的职责分层</H2>
        <Grid columns={4} gap={12}>
          <ArchitectureStep
            index="01"
            title="调用入口"
            detail="用户用短提示词指定时间范围、汇报对象、重点和输出形式。"
          />
          <ArchitectureStep
            index="02"
            title="项目 Skill"
            detail="统一读取对话、Git、实验日志、配置、数据处理文档和论文证据。"
          />
          <ArchitectureStep
            index="03"
            title="事实正文"
            detail="Markdown 是可版本控制、可复核、可引用的规范报告。"
          />
          <ArchitectureStep
            index="04"
            title="汇报视图"
            detail="Canvas 将真实信息重排为进展、指标、风险和下一步视图。"
          />
        </Grid>
        <Text tone="tertiary" size="small">
          边界检查点独立于报告正文：记录报告截止时间或基准提交，避免依赖模糊的“上一次报告”。
        </Text>
      </Stack>

      <Grid columns="1.2fr 1fr" gap={20} align="start">
        <Stack gap={12}>
          <H2>现有 <Code>1.md</Code> 的定位</H2>
          <Text>
            它已经具备较好的 Skill 核心：范围确认、证据优先、四类状态区分、按需章节和复现要求。
          </Text>
          <Text tone="secondary">
            后续应迁移并收紧为项目 Skill，同时补充 DFormer 证据映射、报告边界持久化以及 Markdown 与 Canvas 的双输出契约。
          </Text>
        </Stack>

        <Card>
          <CardHeader>DFormer 项目已有基础</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text size="small">• 仓库包含训练、评估、数据处理、论文与配置等证据源。</Text>
              <Text size="small">• <Code>doc/</Code> 已承担项目文档。</Text>
              <Text size="small">• <Code>tools/publish-canvas.ps1</Code> 已支持发布 Canvas。</Text>
              <Text size="small">• 版本化文件名与报告索引可保护历史 Canvas。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={12}>
        <H2>推荐落地形态</H2>
        <Grid columns={3} gap={12}>
          <ArchitectureStep
            index="A"
            title="项目级 Skill"
            detail="SKILL.md 保持精简，项目证据映射和输出格式放到一级引用文件。"
          />
          <ArchitectureStep
            index="B"
            title="双输出契约"
            detail="默认生成 Markdown；需要组会展示或视觉布局时再生成 Canvas。"
          />
          <ArchitectureStep
            index="C"
            title="版本化索引"
            detail="记录正式报告边界与 Canvas 版本，禁止静默覆盖和删除旧版本。"
          />
        </Grid>
      </Stack>

      <Card variant="borderless">
        <CardHeader>保护策略</CardHeader>
        <CardBody>
          <Text tone="secondary">
            Canvas 文件名使用语义版本前缀；每次修订创建新版本；发布脚本校验命名，并在目标内容不同时拒绝覆盖。旧版本保留用于回溯。
          </Text>
        </CardBody>
      </Card>

      <Text tone="quaternary" size="small">
        Canvas 版本 0.0.1 · 依据：<Code>1.md</Code>、DFormer 仓库结构、Canvas 发布流程及 Cursor Skill/Canvas 规范 · 2026-08-17
      </Text>
    </Stack>
  );
}