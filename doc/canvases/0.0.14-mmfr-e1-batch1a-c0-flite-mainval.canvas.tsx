import {
  BarChart,
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
} from "cursor/canvas";

const CANVAS_VERSION = "0.0.14";
const reportPath = "doc/reports/2026-09-22-mmfr-e1-batch1a-c0-flite-mainval.md";
const c0Root =
  "cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/C0/mainval-10cond-v1/";
const fliteRoot =
  "cloud/MMFR_E1_Batch1A_local_transfer_20260922/MMFR_E1_Batch1A_local_transfer_20260922/FLite/mainval-10cond-v1/";

const shortNames = [
  "clean",
  "SD@0.75",
  "GN@0.75",
  "blur@0.75",
  "quant@0.75",
  "mis@0.75",
  "EM@1.0",
  "SD+GN@0.5",
  "blur+mis@0.5",
  "quant+mis@0.5",
];

const miouC0 = [56.69, 53.92, 56.34, 56.65, 56.62, 55.57, 52.63, 54.94, 56.14, 56.0];
const miouF = [56.02, 54.32, 55.85, 55.95, 55.99, 55.24, 52.0, 55.36, 55.33, 55.37];
const maccC0 = [69.53, 68.37, 69.33, 69.5, 69.47, 69.03, 67.15, 68.83, 69.14, 69.32];
const maccF = [69.09, 68.52, 69.14, 69.04, 69.06, 68.6, 66.35, 69.08, 68.63, 68.63];
const mf1C0 = [71.02, 68.83, 70.77, 70.97, 70.95, 70.18, 67.75, 69.72, 70.56, 70.54];
const mf1F = [70.36, 69.14, 70.35, 70.3, 70.33, 69.75, 67.04, 70.02, 69.75, 69.81];

const delta = miouF.map((v, i) => Number((v - miouC0[i]).toFixed(2)));

function fmt(v: number): string {
  return v.toFixed(2);
}

function DeltaCell({ value }: { value: number }) {
  const label = `${value > 0 ? "+" : ""}${value.toFixed(2)}`;
  return (
    <Text
      size="small"
      tone={value > 0 ? "primary" : "tertiary"}
      weight={value > 0 ? "semibold" : "normal"}
    >
      {label}
    </Text>
  );
}

export default function MmfrE1Batch1aMainval() {
  const positive = delta.filter((d) => d > 0).length;

  const conditionRows = shortNames.map((name, i) => [
    name,
    fmt(miouC0[i]),
    fmt(miouF[i]),
    <DeltaCell key={`delta-${name}`} value={delta[i]} />,
    fmt(maccC0[i]),
    fmt(maccF[i]),
    fmt(mf1C0[i]),
    fmt(mf1F[i]),
  ]);

  return (
    <Stack gap={20}>
      <Stack gap={8}>
        <Row gap={8} align="center" wrap>
          <Pill size="sm">{`v${CANVAS_VERSION}`}</Pill>
          <Pill size="sm">冻结评测器 msflip-whole-original-grid-v1</Pill>
          <Pill size="sm">318 条 val-dev · 10 条件 · 每图 10 view</Pill>
          <Pill size="sm">official test sealed_unread</Pill>
        </Row>
        <H1>MMFR E1 Batch 1A：C0 与 F-lite 十条件 Main-Val</H1>
        <Text tone="secondary">
          汇报周期 2026-09-22 至 2026-09-23 · 本地 RTX 5060 Laptop GPU · 两侧各 10/10
          条件完成 · 身份与配对性断言 0 项失败
        </Text>
      </Stack>

      <Callout tone="warning" title="大白话结论">
        十个条件全部跑完，两份结果都完整、可配对。但在十视图这个更严格的口径下，F-lite
        没有复现之前单视图筛查的领先：主指标（六类单故障的宏平均 mIoU，记作 M6）C0 为
        55.29、F-lite 为 54.89，低 0.40 个百分点；clean 从 56.69 降到 56.02，低 0.67
        个百分点。十个条件里只有 2 个对 F-lite 有利。这不等于 F-lite
        变差了，因为只有一个 seed、一个 checkpoint，也没做置信区间，而且本轮事先没有给
        Main-Val 定过任何数值门槛，所以这些数字只是描述性的，怎么处置需要上级判断。
      </Callout>

      <Grid columns={4} gap={16}>
        <Stat value="55.2883" label="C0 主指标 M6（mIoU %）" />
        <Stat value="54.8917" label="F-lite 主指标 M6（mIoU %）" tone="warning" />
        <Stat value="-0.3966" label="Δ M6 = F-lite − C0（pp）" tone="danger" />
        <Stat value="-0.67" label="Δ clean mIoU（pp）" tone="danger" />
      </Grid>

      <Stack gap={8}>
        <H2>每个条件上 F-lite 相对 C0 的 mIoU 差</H2>
        <Text tone="secondary" size="small">
          单位：百分点（pp），正值表示 F-lite 更好；零线用于区分方向。来源：两侧
          update-2560/&lt;condition&gt;/metrics.json 的 metrics_percent.miou。
        </Text>
        <BarChart
          categories={shortNames}
          series={[{ name: "Δ mIoU（F-lite − C0）", data: delta, tone: "info" }]}
          height={220}
          valueSuffix=" pp"
          showValues
          beginAtZero={false}
          referenceLines={[{ value: 0, label: "无差异" }]}
        />
        <Text size="small" tone="tertiary">
          10 个条件中 {positive} 个为正：spatial_dropout@0.75（+0.40）与
          spatial_dropout@0.5+gaussian_noise@0.5（+0.42）；最大负差为
          blur@0.5+misalignment@0.5（-0.81）。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>完整十条件表</H2>
        <Table
          headers={[
            "condition",
            "C0 mIoU",
            "F-lite mIoU",
            "Δ mIoU",
            "C0 mAcc",
            "F-lite mAcc",
            "C0 mF1",
            "F-lite mF1",
          ]}
          rows={conditionRows}
          columnAlign={[
            "left",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
            "right",
          ]}
          striped
        />
        <Text size="small" tone="tertiary">
          缩写：SD = spatial_dropout，GN = gaussian_noise，mis = misalignment，quant =
          quantization，EM = entire_missing；@ 后为 severity，+ 表示同时叠加两种故障。指标
          单位均为 %，Δ 为百分点。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>汇总量</H2>
        <Table
          headers={["汇总量", "C0", "F-lite", "Δ (pp)"]}
          rows={[
            ["M6：六个单故障宏平均 mIoU（主指标）", "55.2883", "54.8917", "-0.3966"],
            ["clean mIoU", "56.69", "56.02", "-0.67"],
            ["三个混合条件宏平均 mIoU", "55.6933", "55.3533", "-0.34"],
            ["九个受损条件宏平均 mIoU", "55.4233", "55.0456", "-0.3777"],
            ["十个条件宏平均 mIoU", "55.55", "55.143", "-0.407"],
            ["M6 的 mAcc", "68.8083", "68.4517", "-0.3566"],
            ["M6 的 mF1", "69.9083", "69.485", "-0.4233"],
            ["最坏单条件 mIoU（entire_missing@1.0）", "52.63", "52.00", "-0.63"],
          ]}
          columnAlign={["left", "right", "right", "right"]}
          striped
        />
        <Text size="small" tone="tertiary">
          主指标名称为 unweighted-macro-mean-single-condition-mIoU，定义为六个单故障条件各自
          mIoU 的未加权宏平均，取自两侧 summary.json 的 protocol_primary_score 字段。
        </Text>
      </Stack>

      <Stack gap={8}>
        <H2>判断边界（直接影响本次结论）</H2>
        <Table
          headers={["边界", "含义", "对结论的影响"]}
          rows={[
            [
              "无 Main-Val 预注册门槛",
              "冻结的 promote/stop 门槛只属于 Quick-Val，且已于 2026-09-22 使用",
              "本轮结果是描述性证据，不构成保留或放弃 F-lite 的判定",
            ],
            [
              "单 seed / 单 checkpoint / 单次运行",
              "无 bootstrap、无置信区间、无运行间噪声估计",
              "无法区分真实差异与运行波动，不声称因果",
            ],
            [
              "口径翻转",
              "Quick-Val（单视图）4 个共同条件中 3 个符号翻转：clean +0.69 → -0.67、EM@1.0 +1.41 → -0.63、mis@0.75 +0.48 → -0.33",
              "单视图筛查增益未在十视图口径复现；两套口径数字不可直接比较",
            ],
            [
              "训练期 shuffle 顺序差异",
              "两 run 的样本-槽位排列不同（C0 clean 6359/25600，F-lite 6369/25600），已接受为 screening-level",
              "影响 Δ_F 的解读粒度，但不改变冻结训练合同",
            ],
            [
              "逐类变化不均衡",
              "clean 的 mIoU 差几乎全部由 support equipment（-10.63 pp，约占 -0.71 pp）等少数类驱动",
              "逐类结论仅为观察，未做检验，也未预注册为门禁",
            ],
          ]}
          columnAlign={["left", "left", "left"]}
        />
      </Stack>

      <Stack gap={8}>
        <H2>下一步（需上级裁决）</H2>
        <Stack gap={6}>
          <Text>
            1. 裁决本轮十视图结果的处置：作为描述性证据保留并维持 Quick-Val 的
            promote，还是触发新的预注册确认实验。裁决前不启动新的研究分支。
          </Text>
          <Text>
            2. 如需判断 -0.3966 pp 是否为真实差异，另立预注册实验：重复运行估计运行间波动，
            并对逐样本配对差异使用冻结的 location-group 单位做 bootstrap；门槛须在看到新结果前
            写入 protocol，并需单独授权。
          </Text>
          <Text>
            3. 核查口径差来源：用同一 checkpoint 在单视图与十视图两个口径下复算 4
            个共同条件，确认单视图增益是否只存在于单视图口径。
          </Text>
          <Text>
            4. 授权前保持停止：不实现 R-OE、不运行 T、不启动 Batch 1B/2、不重选
            checkpoint、不读取 official test（保持 sealed_unread）。
          </Text>
        </Stack>
      </Stack>

      <Card>
        <CardHeader trailing={<Pill size="sm">两侧各 10/10</Pill>}>
          证据与复现入口
        </CardHeader>
        <CardBody>
          <Stack gap={8}>
            <H3>断言检查结果</H3>
            <Text size="small">
              两侧 runner、protocol 原文、条件定义、样本集合、条件顺序、视图分批、evaluation
              seed、torch 版本与 TF32 开关逐项一致；每个条件 metrics.json 的 identity_sha256
              等于 manifest 记录的 expected 值；同条件 corrupted Depth 聚合 SHA-256
              两侧相同（10/10），标签像素支持同为 155,829,149。断言失败 0 项。
            </Text>
            <Divider />
            <H3>成本</H3>
            <Text size="small">
              逐条件耗时之和：C0 16,323.43 s（约 4.53 h）、F-lite 15,883.17 s（约 4.41
              h）；单元平均耗时 5.00 s 对 4.95 s；峰值显存（allocated / reserved）4700.6 /
              6578.0 MiB 对 4701.3 / 6580.0 MiB；精度 FP32、TF32 off。
            </Text>
            <Divider />
            <H3>路径</H3>
            <Text size="small" tone="secondary">
              C0：{c0Root}
              <br />
              F-lite：{fliteRoot}
              <br />
              每侧包含 run_manifest.json、update-2560/summary.json 与 10 份
              update-2560/&lt;condition&gt;/metrics.json。目录位于仓库外，不进入 Git。
            </Text>
            <Divider />
            <H3>未执行的检查</H3>
            <Text size="small" tone="secondary">
              未做 bootstrap、配对显著性检验、多 seed 复现与逐类检验；未训练、未重跑
              Quick-Val、未重选 checkpoint；Main-Val 的 evaluator stdout 与退出码未落盘，
              完成性证据是产物完整性加身份一致性。
            </Text>
            <Divider />
            <Text size="small" tone="tertiary">
              Markdown 事实正文：{reportPath} · 生成日期 2026-09-23 · Canvas 版本 v
              {CANVAS_VERSION}
            </Text>
          </Stack>
        </CardBody>
      </Card>
    </Stack>
  );
}
