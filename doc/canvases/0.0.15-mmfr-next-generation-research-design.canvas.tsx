import { Stack, Row, Grid, H1, H2, H3, Text, Table, Divider, useHostTheme, useCanvasAction, Button } from "cursor/canvas";

const report = "D:/0Project/DFormer/doc/reports/2026-09-30-mmfr-next-generation-research-design.md";
const candidates = [
  ["A · 主推荐", "固定补偿动作的任务效用", "单点空间残差 + 样本级连续控制", "同输入、同权重 off/full 风险差监督", "约 21,185 参数（草算）", "MoSA / DCF 相邻；proposal 可能无收益"],
  ["B · 主创新淘汰", "几何输入在编码前受损", "输入 Depth 的连续有界校准", "分割监督 + clean 一致性", "0.1–0.5M 预算范围（未计数）", "GeomPrompt / DCF 高度重合；梯度穿过骨干"],
  ["C · 性能备选", "缺失输入的语义表征不足", "完整输入 → 受损输入自教", "CE + detached teacher KL", "约 16,992 参数（草算）", "成熟蒸馏路线；仍无拒绝干预能力"],
];
const papers = [
  ["LIB000014 / PR089", "SGMA", "类别语义原型、鲁棒性注意力、反向可靠性采样", "attention 不等于动作收益"],
  ["LIB000023 / PR090", "RobustSeg", "teacher-student、Anymodal Dropout、hybrid prototypes", "独立模态分支不可直接搬入"],
  ["LIB000024 / PR029", "UMFNet", "Gaussian latent uncertainty → confidence fusion", "不是经过监督的错位概率"],
  ["LIB000026 / AI023", "GeomPrompt", "冻结分割器、任务驱动输入几何提示/恢复", "输入恢复已有直接先例"],
  ["LIB000027 / AI017", "Calibrated RGB-D / DCF", "任务 IoU 质量标签、raw/estimated Depth 混合", "task-aware reliability 并非首次"],
  ["LIB000028 / MoSA", "MoSA", "空间 adapter 调制 + 位置级可靠性融合", "gate + adapter 已有；监督定义存在矛盾"],
  ["LIB000029 / AI019", "MaskMentor", "多模态遮蔽、共享权重自教、pixel/token 重建", "不是轻量 logit KD 的忠实复现"],
  ["LIB000030 / AI024", "Condition Dropout", "冻结主路径、完整 encoder copy、zero-conv", "冻结与零初始化不构成新贡献"],
  ["LIB000008 / 无人工编号", "SMAC", "RGB 估计 Depth feature、残差派生选择", "样本级选择已有；残差不是质量真值"],
];

export default function MMFRResearchDesign() {
  const theme = useHostTheme();
  const dispatch = useCanvasAction();
  return (
    <Stack gap={22} style={{ background: theme.bg.editor, color: theme.text.primary, padding: 28, maxWidth: 1260, margin: "0 auto" }}>
      <Stack gap={8}>
        <Text tone="secondary" size="small">2026-09-30 · 展示版本 0.0.15 · 研究设计待审核 · 未实现、未训练、未评价</Text>
        <H1>下一代 MMFR：先判断补偿动作是否有用</H1>
        <Text>MMFR 是多形式模态失效与可靠性研究。F-lite 总是修正，R-OE-lite 只在整幅 Depth 全空时恢复。下一步优先检验：固定一个补偿动作后，能否判断哪些输入值得使用它。</Text>
        <Text style={{ color: theme.accent.primary }} weight="semibold">推荐 A：任务效用控制的单点残差。推荐依据是接口合适、贡献可拆分、容易否证；不是预设可靠性融合必胜。</Text>
        <Row gap={10} wrap>
          <Button onClick={() => dispatch({ type: "openFile", path: report })}>打开完整研究报告</Button>
          <Text tone="secondary" size="small">唯一事实正文为 Markdown；本展示不新增结论或实验授权。</Text>
        </Row>
      </Stack>
      <Divider />
      <H2>既有结果揭示了什么</H2>
      <Grid columns="repeat(auto-fit, minmax(280px, 1fr))" gap={24}>
        <Stack gap={8}>
          <H3>F-lite：收益未跨口径复现</H3>
          <Text>四条件单视图 Quick-Val 的 hard 平均增益约 +0.96 pp；十条件十视图 Main-Val 的主指标 M6 却为 −0.3966 pp，clean −0.67 pp。</Text>
          <Text tone="secondary">无显式质量/效用控制，可能产生正负干预混合。但单 seed、训练随机顺序差异与无预注册 Main-Val 门槛，使其不足以证明因果性退化。</Text>
        </Stack>
        <Stack gap={8}>
          <H3>R-OE-lite v2：覆盖窄、净贡献未分离</H3>
          <Text>2560 次更新确实完成，substitute 确实训练。EM 318/318 触发，clean / SD / Mis 均 0/318 触发。</Text>
          <Text tone="secondary">四项差为 0.00 / +0.01 / +0.01 / +0.01 pp，inconclusive。strict bypass 条件也随 base 漂移变化，因此 +0.01 不能记为恢复器收益。</Text>
        </Stack>
      </Grid>
      <H3>结果来源与单位</H3>
      <Table headers={["既有指标", "C0", "候选", "候选 − C0（百分点 pp）", "评价口径"]} rows={[
        ["F-lite · M6", "55.2883", "54.8917", "−0.3966", "十条件十视图；六单故障宏平均 mIoU"],
        ["F-lite · clean mIoU", "56.69", "56.02", "−0.67", "十条件十视图 Main-Val"],
        ["R-OE v2 · clean", "53.46", "53.46", "0.00", "四条件单视图 Quick-Val"],
        ["R-OE v2 · entire_missing", "48.76", "48.77", "+0.01", "四条件单视图 Quick-Val"],
        ["R-OE v2 · spatial_dropout", "51.40", "51.41", "+0.01", "四条件单视图 Quick-Val"],
        ["R-OE v2 · misalignment", "52.15", "52.16", "+0.01", "四条件单视图 Quick-Val"],
      ]} />
      <Text tone="secondary" size="small">来源：2026-09-22 Batch 1A Main-Val 正式报告、2026-09-24 R-OE v2 正式报告；均 318 val-dev。表内 mIoU 为百分比。两套口径的绝对值不可互比；本次没有重跑评价。</Text>
      <Divider />
      <H2>三个候选与防撞车取舍</H2>
      <Table headers={["处置", "问题", "机制", "监督", "成本", "主要风险"]} rows={candidates} />
      <Text tone="secondary">DFormerv2 没有独立 Depth encoder features，通用 RGB/Depth 双支路加权融合不是现成接口。旧 Oracle-A 已否决有效性驱动的几何抑制，本轮不以连续 gate 换名复活。</Text>
      <Stack gap={10} style={{ background: theme.fill.tertiary, padding: 18 }}>
        <H3>A 的核心流程</H3>
        <Text>冻结 C0 权重与 BN 状态 → 取 stage index 2 的 256 通道 feature → 生成有界空间残差 → 用样本级 gate 控制幅度 → 复用 HAM。</Text>
        <Text>先训练 proposal，再冻结 proposal。用同输入、同权重、同 HAM 随机状态的 off/full 任务损失差训练 gate；推理只做一次网络前向。</Text>
        <Text tone="secondary">第一版不输出局部可靠性图。非局部解码后逐像素损失差不是局部残差的因果贡献；样本级控制有真实能力上限。零初始化只保证初始不干预，训练后 clean 保真仍须验收。</Text>
      </Stack>
      <H2>九篇论文给出的机制边界</H2>
      <Table headers={["索引编号", "论文简称", "相关机制", "不能据此声称"]} rows={papers} />
      <Text tone="secondary" size="small">正文与必要结构图定点复核，不按外部性能选模块，未引用未经工作簿核对的论文分数。仅九篇内机制审计，不是全领域首创裁决。</Text>
      <H3>有限新增是什么</H3>
      <Text>A 尝试监督“这一具体残差动作相对不用它的任务增益”，区别于预测模态置信度或融合 raw/estimated Depth。冻结、低秩、零初始化与换 backbone 都是已有思想。MoSA 结构近，DCF 监督动机近，创新风险仍为中高。</Text>
      <Divider />
      <H2>最小第一阶段：一个新模型，三种同权重行为</H2>
      <Grid columns="repeat(auto-fit, minmax(280px, 1fr))" gap={24}>
        <Stack gap={8}>
          <H3>拟议运行范围</H3>
          <Text>绑定已核验 C0 fixed-final，不重训 C0。新训练建议最多 2560 成功更新：proposal 1920 + gate 640，阶段长度和损失超参数待批准。</Text>
          <Text>同 checkpoint 比较 off / full / learned，复用四条件、单视图、318 val-dev。先确认 off 匹配 C0，再解释动作收益。</Text>
        </Stack>
        <Stack gap={8}>
          <H3>拟议继续线与停止线</H3>
          <Text>learned hard 平均 ≥ +0.50 pp、clean ≥ −0.20 pp，并且 hard 优于 full，才建议继续；这是待批准资源分配线，不是显著性结论。</Text>
          <Text>proposal 无可用收益就停止；full 有用而 learned 不优，gate 未获支持；微差或混合方向仍 inconclusive，不自动追加训练。</Text>
        </Stack>
      </Grid>
      <Text tone="secondary">同权重 off/full：固定全部模型参数，只改变补偿动作。动作效用：它是否降低当前输入的任务损失，而不是 Depth 是否非零。HAM 随机回放：三种行为必须使用对应输入相同随机状态。</Text>
      <H3>审核后才可能发生的下一步</H3>
      <Text>先裁决核心假设、source/freeze 行为、两阶段目标与预算，再另行授权最小实现和资格核验；通过后才考虑一条训练与四条件筛选。现有 evaluator、冻结协议与旧审核包均保持不变。</Text>
      <Text tone="secondary">4090 是首选验证设备；5060 Laptop 完整训练可行性未确认。参数/显存影响未实测。没有项目测试、GPU 运行、新训练/评价、official test 访问、提交或推送。设计完成后停止，等待上级审核。</Text>
      <Divider />
      <Text tone="secondary" size="small">事实源：doc/reports/2026-09-30-mmfr-next-generation-research-design.md · 基准提交 a7c5cc17 · 展示 0.0.15</Text>
    </Stack>
  );
}
