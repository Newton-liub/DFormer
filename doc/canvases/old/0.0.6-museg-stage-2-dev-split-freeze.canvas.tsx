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

const reportPath = "doc/reports/2026-08-25-museg-stage-2-dev-split-freeze.md";
const protocolPath = "doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/01-开发划分协议与生成工具.md";
const indexPath = "doc/plans/archive/2026-08-MUSeg阶段二长程Baseline与MVE/00-总索引与执行门禁.md";

function App() {
  const dispatch = useCanvasAction();
  const theme = useHostTheme();

  return (
    <Stack gap={22} style={{ padding: 24, maxWidth: 1200, margin: "0 auto" }}>
      <Stack gap={10}>
        <Row align="center" justify="space-between" wrap>
          <Pill active>v0.0.6 · Stage 01 项目汇报</Pill>
          <Text size="small" tone="tertiary">汇报日期 2026-08-25</Text>
        </Row>
        <H1>MUSeg 阶段二：开发划分已冻结</H1>
        <Text tone="secondary">
          我们已经把“训练用什么、开发验证用什么、最终考试用什么”固定下来，并通过独立审计。
          Stage 02 可以开始改造训练链路，但 GPU 和模型训练尚未启动。
        </Text>
        <Row gap={8} wrap>
          <Button variant="primary" onClick={() => dispatch({ type: "openFile", path: reportPath })}>
            打开完整汇报
          </Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: protocolPath })}>
            打开冻结协议
          </Button>
          <Button variant="secondary" onClick={() => dispatch({ type: "openFile", path: indexPath })}>
            打开阶段总索引
          </Button>
        </Row>
      </Stack>

      <Grid columns={4} gap={14}>
        <Stat value="1277" label="train-dev 图片" tone="info" />
        <Stat value="318" label="val-dev 图片" tone="success" />
        <Stat value="102/102" label="冻结审计通过" tone="success" />
        <Stat value="46" label="完整测试通过" tone="success" />
      </Grid>

      <Callout tone="success" title="一句话结论">
        Stage 01 已完整完成并通过 Gate A。大白话来说：练习题、模拟卷和最终考卷已经分开并签字封存，后面不能为了成绩好看临时换题。
      </Callout>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>为什么做</Pill>}>
            防止开发过程偷看最终答案
          </CardHeader>
          <CardBody>
            <Stack gap={9}>
              <Text>过去训练配置可能在开发过程中查看 official test。</Text>
              <Text>现在只用 <Code>val-dev</Code> 选择 epoch、checkpoint 和方案。</Text>
              <Text tone="secondary">
                official test 继续封存，等正式协议和三个 seed 都准备好后再一次性验收。
              </Text>
            </Stack>
          </CardBody>
        </Card>

        <Card size="lg">
          <CardHeader trailing={<Pill size="sm" active>怎么保证</Pill>}>
            位置组整体划分
          </CardHeader>
          <CardBody>
            <Stack gap={9}>
              <Text>同一拍摄位置的图片必须全部在同一侧，不能拆分。</Text>
              <Text tone="secondary">
                大白话：同一地点的连拍照片算“一家人”，避免模型在验证时认出训练中见过的地点。
              </Text>
              <Text>train、val、test 的样本和位置组均两两零交叉。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>最终数据职责</H2>
        <Table
          headers={["数据侧", "图片 / 位置组", "干什么", "大白话"]}
          rows={[
            ["train-dev", "1277 / 762", "开发训练和调试", "平时练习题"],
            ["val-dev", "318 / 196", "选 epoch、checkpoint 和方案", "模拟考试卷"],
            ["official test", "1576 / 957", "最终一次性验收", "封存的最终考卷"],
          ]}
          rowTone={["info", "success", "warning"]}
          striped
        />
      </Stack>

      <Divider />

      <Stack gap={10}>
        <H2>关键概念：技术说法与大白话</H2>
        <Table
          headers={["概念", "大白话解释", "实际作用"]}
          rows={[
            ["固定 seed", "固定的洗牌钥匙", "每次重做都得到同一名单"],
            ["确定性算法", "相同输入必然得到相同输出", "不能多生成几版后挑最好看的"],
            ["manifest", "划分的出生证明和说明书", "记录规则、统计、版本和文件指纹"],
            ["SHA-256", "文件的防伪指纹", "任何一个字节变化都能被发现"],
            ["独立 audit", "另一位检查员从头重新点数", "不直接相信生成器自己报的结果"],
            ["freeze", "名单签字封存", "禁止后续根据实验结果重新挑数据"],
            ["Gate A", "负责人最终签字", "把机器检查通过与项目批准分开"],
          ]}
          rowTone={["neutral", "info", "neutral", "neutral", "success", "warning", "success"]}
          striped
        />
      </Stack>

      <Grid columns="1fr 1fr 1fr" gap={14} align="start">
        <Card>
          <CardHeader trailing={<Pill size="sm">步骤 1</Pill>}>固定协议</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>固定输入哈希、seed、分组规则和六级目标。</Text>
              <Text tone="secondary">先写规则，再看结果，避免结果导向地改规则。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">步骤 2</Pill>}>生成与复核</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>算法接受 59 次严格改进后停止。</Text>
              <Text tone="secondary">没有人工挪组，没有换 seed，没有多版本择优。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm" active>步骤 3</Pill>}>批准与冻结</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>Sol 技术复核后，用户批准 Gate A。</Text>
              <Text tone="secondary">冻结发布只改变批准状态，不改变 membership。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Stack gap={10}>
        <H2>三个 warning：为什么不是失败</H2>
        <Table
          headers={["warning", "原因", "大白话", "判断"]}
          rows={[
            ["类别 6 像素比例", "像素集中在少数完整位置组", "大箱子不能拆，放哪边都会跳一下", "接受"],
            ["全背景图片比例", "总共只有 5 张，val 有 3 张", "样本太少，整数划分不可能很平滑", "接受"],
            ["单图类别数直方图", "高类别数组合本来就稀少", "少一张或多一张，百分比都会变化很大", "接受"],
          ]}
          rowTone={["warning", "warning", "warning"]}
          striped
        />
        <Callout tone="info" title="批准依据">
          warning 是黄灯，不是红线失败。所有 15 类都进入 val-dev，六个矿区双边覆盖，val 只比目标少 1 张，且所有硬约束与 102 项独立审计全部通过。
        </Callout>
      </Stack>

      <Grid columns="1fr 1fr" gap={16} align="start">
        <Card>
          <CardHeader trailing={<Pill size="sm" active>已完成</Pill>}>Stage 01 交付</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>划分生成器、独立审计器和协议测试。</Text>
              <Text>五个冻结文件发布到 <Code>data/splits/MUSeg/dev-v1</Code>。</Text>
              <Text>Gate A、文档同步、Git 可见性和最终验证完成。</Text>
            </Stack>
          </CardBody>
        </Card>

        <Card>
          <CardHeader trailing={<Pill size="sm">下一步</Pill>}>Stage 02 训练链路</CardHeader>
          <CardBody>
            <Stack gap={8}>
              <Text>让训练只读 train-dev，验证只读 val-dev。</Text>
              <Text>补齐 best/latest checkpoint、每轮验证和可靠 resume。</Text>
              <Text tone="secondary">先修管道，再申请 GPU；本次没有启动训练。</Text>
            </Stack>
          </CardBody>
        </Card>
      </Grid>

      <Callout tone="warning" title="当前边界">
        Stage 02、GPU qualification、长程训练、正式三 seed baseline 和 official test 解封都尚未执行。
        当前工作区尚未创建 Stage 01 Git commit。
      </Callout>

      <Stack gap={8}>
        <H2>冻结证据</H2>
        <Text><Code>manifest.json</Code>：<Code>42233412f432e387cfcffc763724461e2dbc111969a595c714ac12add7bf7b01</Code></Text>
        <Text><Code>audit-report.json</Code>：<Code>53ac30aba0230919b994202f37b3571a7b416f9129f27eabf003415721e38055</Code></Text>
        <Text tone="secondary">完整五文件哈希、测试命令和 warning 解释见正式汇报。</Text>
      </Stack>

      <Text
        size="small"
        tone="quaternary"
        style={{ borderTop: `1px solid ${theme.stroke.tertiary}`, paddingTop: 12 }}
      >
        Canvas 0.0.6 · 事实源：Stage 01 冻结 manifest、102/102 独立审计、46 项完整测试及 2026-08-25 项目汇报 · 当前 HEAD 9fae925a95aa8d593ef6d00882266532cbd73aed，工作区变更尚未提交。
      </Text>
    </Stack>
  );
}

export default App;