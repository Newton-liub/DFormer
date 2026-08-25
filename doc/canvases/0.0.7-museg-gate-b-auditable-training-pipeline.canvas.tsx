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

const reportPath = "doc/reports/2026-08-25-museg-stage-2-gate-b-auditable-training-pipeline.md";
const indexPath = "doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md";
const stage04Path = "doc/临时/待执行/MUSeg阶段二长程Baseline与MVE/04-静态验收与4090Qualification.md";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1220, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.7 · Gate B 阶段汇报</Pill>
          <Text size="small" tone="tertiary">汇报日期 2026-08-25</Text>
        </Row>
        <H1>MUSeg：从“能训练”到“结果可证明”</H1>
        <Text tone="secondary">
          Stage 01–03 已建立数据冻结、完整训练恢复、三种子编排和实验身份审计闭环。
          Gate B 已签署；4090 qualification 和模型训练尚未启动。
        </Text>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => dispatch({ type: "openFile", path: reportPath })}>
            打开详细阶段报告
          </Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: indexPath })}>
            打开门禁总索引
          </Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: stage04Path })}>
            查看下一阶段
          </Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="102/102" label="冻结数据审计" tone="success" />
        <Stat value="90" label="完整无卡测试通过" tone="success" />
        <Stat value="3" label="训练 phase 闭合" tone="info" />
        <Stat value="4" label="独立提交边界" tone="info" />
      </Grid>

      <Callout tone="success" title="组会结论">
        现在每个实验结果都必须能回答：用了哪份数据、哪次提交、哪个 seed、什么训练规则、从哪个 checkpoint 恢复、子进程是否真实成功。
        任一身份不一致，流程会在正式训练或汇总前拒绝结果。
      </Callout>

      <Stack gap={10}>
        <H2>为什么先修链路，而不是直接跑 4090</H2>
        <Grid columns="1fr 1fr 1fr" gap={14} align="start">
          <Card>
            <CardHeader trailing={<Pill size="sm">原风险 1</Pill>}>最终 test 参与开发</CardHeader>
            <CardBody>
              <Stack gap={8}>
                <Text>旧配置可能在训练期间查看官方 test。</Text>
                <Text tone="secondary">后果：epoch、checkpoint 或方案可能带有最终考卷的选择偏差。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm">原风险 2</Pill>}>中断后“看似恢复”</CardHeader>
            <CardBody>
              <Stack gap={8}>
                <Text>只加载模型权重，不能恢复优化器、随机数和学习率现场。</Text>
                <Text tone="secondary">后果：程序继续运行，但训练轨迹已经改变。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm">原风险 3</Pill>}>成功和身份不可信</CardHeader>
            <CardBody>
              <Stack gap={8}>
                <Text>退出码 0、一个 JSON 或一个 checkpoint 都可能来自错误 protocol。</Text>
                <Text tone="secondary">后果：三 seed 汇总可能混入缺失、错绑或历史结果。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>四步形成完整证据链</H2>
        <Grid columns="1fr 1fr 1fr 1fr" gap={12} align="start">
          <Card>
            <CardHeader trailing={<Pill size="sm" active>Stage 01</Pill>}>冻结数据</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text><Code>1277</Code> train-dev</Text>
                <Text><Code>318</Code> val-dev</Text>
                <Text><Code>1576</Code> official test 封存</Text>
                <Text tone="secondary">同一拍摄位置整体划分，防止相似场景泄漏。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" active>Stage 02</Pill>}>保存训练现场</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text>model + optimizer + AMP</Text>
                <Text>epoch + step + LR + best</Text>
                <Text>CPU/CUDA 随机数状态</Text>
                <Text tone="secondary">恢复时逐项核对协议，不兼容立即停止。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" active>Stage 03</Pill>}>控制三次实验</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text>preflight 前置检查</Text>
                <Text>单 seed 独立证据目录</Text>
                <Text>三 seed 串行、失败即停</Text>
                <Text tone="secondary">退出码 0 仍需结果和 checkpoint 双重校验。</Text>
              </Stack>
            </CardBody>
          </Card>
          <Card>
            <CardHeader trailing={<Pill size="sm" active>Gate B 修复</Pill>}>绑定权威来源</CardHeader>
            <CardBody>
              <Stack gap={7}>
                <Text>protocol v2 + frozen authority</Text>
                <Text>候选 manifest 可重建证明</Text>
                <Text>qualification 模板 + 物化</Text>
                <Text tone="secondary">协议不能再自行声明另一套 split 身份。</Text>
              </Stack>
            </CardBody>
          </Card>
        </Grid>
      </Stack>

      <Stack gap={9}>
        <H2>数据职责：练习题、模拟卷、最终考卷</H2>
        <Table
          headers={["阶段", "训练数据", "开发期验证", "official test", "大白话"]}
          rows={[
            ["qualification", "train-dev", "val-dev", "封存不读", "检查机器和流程能否可靠工作"],
            ["development", "train-dev", "val-dev", "封存不读", "用模拟卷冻结训练规则"],
            ["official", "完整 official train", "无", "仍封存", "按已冻结规则完成正式训练"],
          ]}
          rowTone={["info", "success", "warning"]}
          striped
        />
      </Stack>

      <Grid columns="1.08fr 0.92fr" gap={16} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>关键难点</Pill>}>“字段齐全”仍不等于“证据可信”</CardHeader>
          <CardBody>
            <Stack gap={9}>
              <Text>Stage 03 初版 protocol 已记录 split 路径、数量、group 和 SHA-256。</Text>
              <Text>正式复核发现：这些字段仍由 protocol 自己填写，形成的是内部自洽，而不是对 Gate A 冻结结果的外部约束。</Text>
              <Callout tone="info" title="修复逻辑">
                数据权威方向改为：冻结 manifest → protocol → preflight → run → orchestrator → summary。
                后一层只能引用和验证前一层，不能重新定义前一层。
              </Callout>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm">大白话</Pill>}>教务处试卷，而非考生自证</CardHeader>
          <CardBody>
            <Stack gap={9}>
              <Text>旧方式：答卷上写“我用的是指定试卷”。</Text>
              <Text>新方式：系统直接读取教务处封存试卷的编号与指纹，再核对答卷。</Text>
              <Text tone="secondary">即使伪造 JSON 内部每个字段都彼此一致，也无法绕过冻结 authority。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={9}>
        <H2>一次训练如何被证明</H2>
        <Table
          headers={["证据", "记录什么", "拒绝什么"]}
          rows={[
            ["protocol manifest", "数据、模型、commit、seed、训练规则", "自描述 split、未知字段、脏工作区"],
            ["command / environment", "实际 argv、包、驱动、GPU、输出目录", "口头记忆或事后补写配置"],
            ["run manifest", "进程退出、authority、resume 父子关系", "启动失败却无记录"],
            ["training result", "phase、seed、结果、checkpoint 身份", "退出码 0 但没有有效产物"],
            ["orchestrator / summary", "三个 seed 顺序与完整性", "只汇总成功 seed 或混入历史结果"],
          ]}
          rowTone={["info", "neutral", "warning", "success", "success"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card>
          <CardHeader trailing={<Pill size="sm" active>可复核结果</Pill>}>无卡验收</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>102/102</Code> 冻结数据审计通过。</Text>
              <Text><Code>90 passed</Code> 完整 CPU 测试通过。</Text>
              <Text><Code>3 passed</Code> qualification 物化专项通过。</Text>
              <Text>Python compile、JSON、Shell 语法和 Git diff check 通过。</Text>
              <Text tone="secondary">全程未调用 CUDA、4090 probe、真实训练或在线 SwanLab。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>提交链</Pill>}>分层交付边界</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text><Code>764ac4e</Code>：数据冻结与审计</Text>
              <Text><Code>d9ef428</Code>：checkpoint 与恢复</Text>
              <Text><Code>e9e3c1c</Code>：preflight 与三种子编排</Text>
              <Text><Code>4365edd</Code>：authority 绑定与 Gate B 修复</Text>
              <Text tone="secondary">每一层可以独立复核、定位回归和回溯决策。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="当前证据边界">
        Gate B 表示“代码链路和无卡契约可以进入 GPU qualification”，不表示模型性能已经验证。
        RTX 4090、真实恢复、显存探测、在线 SwanLab、长程三 seed 和 official test 仍未执行。
      </Callout>

      <Stack gap={10}>
        <H2>下一步：Stage 04 仍受有卡门禁控制</H2>
        <Table
          headers={["顺序", "动作", "通过标准", "控制点"]}
          rows={[
            ["1", "云端物化真实 qualification manifest", "干净 HEAD、数据和权重身份匹配", "用户开启有卡模式"],
            ["2", "静态 preflight + 真实混合 batch 回归", "0 error；loss/gradient 有限", "异常立即停止"],
            ["3", "batch 4/8/12/16 探测", "吞吐稳定且显存余量达标", "OOM 后停止更大 batch"],
            ["4", "汇报探测结果", "用户确认正式 batch", "Gate C"],
            ["5", "2–3 epoch qualification + resume", "数据、日志、best、checkpoint、恢复闭环", "不得写成 baseline"],
          ]}
          rowTone={["info", "info", "warning", "warning", "success"]}
          striped
        />
      </Stack>

      <Callout tone="success" title="一句话带走">
        本阶段把“一个可能跑完的训练脚本”改造成“每个结果都有数据来源、训练现场、重复实验和失败证据的受控流程”；
        当前可以申请有卡工程资格验证，但不能提前宣称模型性能结论。
      </Callout>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        Canvas 0.0.7 · 事实源：Stage 01–03 提交链、Gate B authority 修复、102/102 冻结审计、90 项完整无卡测试及 2026-08-25 正式阶段报告 · HEAD 4365edd7071f5b68e5e3306e9726ae782637fcdd。
      </Text>
    </Stack>
  );
}

export default App;