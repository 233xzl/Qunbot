# 统一配置层

2026-11-17。配置入口为 `load_settings` / `load_bundle`，文件、CLI、回放和后续 Web 表单共用现有语义校验。`Settings` 是有效配置，内含不可变映射；`ConfigBundle` 保留配置源、内容摘要和解析结果。没有另建配置数据库。

## 配置来源与生效

优先级：内置默认 → TOML / 提示词资源 → 显式 CLI 覆盖。`config effective` 显示有效值和来源；未传 CLI 参数不覆盖文件。凭据按引用选取来源，不隐式用环境变量替代文件。

CLI 发布的版本在下次受控启动时采用。管理台支持在线应用，已入队的请求保留原有模型与提示词快照。暂停、撤权和发送前检查沿用现有实时控制。

三种版本含义：

- `schema_version`：文件格式版本，目前支持无版本的旧格式与版本 1。
- 发布修订号：版本目录 ID，用于并发发布和回滚。
- `config_revision`：有效配置与提示词内容的 SHA-256，记录在 run 诊断中；CLI 覆盖会改变该摘要。

## 字段与消费者

`group-agent config schema` 从现有 dataclass 和接受的字段集合生成字段清单，包含类型、默认值、生效方式和敏感标记。范围与跨字段约束由同一校验器执行。

| 配置域 | 内容与默认 | 实际消费者 |
| --- | --- | --- |
| `identity` | bot/owner、群与用户黑名单 | Admission / Intake / 发布前复查 |
| `agent` | 工具循环、必要判断、思考策略、同模型重试、`web_enabled=true` | `build_live_config`、LiveAgent、实时回放 |
| `connections` | 协议、地址、认证引用、Vertex 连接参数 | 合并到 ModelSpec，再交协议适配器 |
| `models` / `model_roles` | 型号、报价、超时、参数；answer/decision/fallback | 生成、回退、熔断、用量计价 |
| `web` | search/fetch 和备用；独立搜索、读取价格 | WebBackends / GeminiSearchClient |
| `prompts` | persona、evidence、reply_contract、decision、tool_decision、search 的文件路径 | `prompt_options`、模型调用、搜索客户端 |
| `budgets` | 总期限 120 秒、上下文上限 80 条、每日普通用户上限 ¥3 | Settings.limits / RunFlow / UsageLedger |
| `runtime` | 全局/用户并发 4/2、队列 32/4、3 搜 5 读、工具超时 25 秒、结束预留 30 秒、80%/100% 闸门；每群受理频率 `group_admit_max` / `group_admit_window_minutes`（默认不限、10 分钟） | 调度、发送限速、工具预算、每日预算判断、受理频率 |
| `archive` | 群聊文字归档，默认关闭；保留 180 天，每次提问最多搜索 3 次、读取 5 次 | 消息入库、`archive.HistoryTools`、清理计划 |
| `session` | 群会话模式，默认关闭；闲置 20 分钟、上一轮 8,000 token 或满 120 分钟后重开一段；每轮最多带入 40 条新群消息 | LiveAgent 回答请求、`session.SessionManager` |
| `collection` / `retention` | 采集、媒体下载；24 小时普通消息、7 天关联消息、2 GB 媒体 | MediaCollector / 清理计划与执行 |
| `serve` / `shadow` / `knowledge` | 数据路径、发送、自同步、失败回复、表情与主人控制 | 运行入口、各业务存储 |
| `billing` | 汇率、报价、免费查询周期、账号总账路径 | AccountBook / GoogleFreeQuota |
| `credentials` | file / env / managed 引用 | CredentialStore，仅在需要连接或显式管理时读取 |

预算达到 100% 后仍允许主人调用，故称普通用户预算上限。累计消费、预留、免费额度已用数、暂停与熔断状态属于运行状态，不参与配置发布或回滚。赠金与新增硬预算策略仍见 tasklist 的 B3/B4。

**本群回答频率。** `runtime.group_admit_max` 设置滚动窗口内最多受理的提问数，填 0 表示不限。`group_admit_window_minutes` 设置窗口分钟数。群个性化可在 `group_profiles.<群号>.runtime` 中逐项覆盖。计数按本群已受理的提问计算，重启后从消息库延续。

以 10 次/10 分钟为例，第 10 条照常受理。从第 11 条起，每条新提问均收到“当前回答次数已达上限，预计 HH:mm:ss 恢复。”提示。被限流的提问和重复投递不占名额。主人的普通提问使用同一额度，控制命令仍可用。主人在本群发送 `#状态` 可查看上限、已用次数和更新时间。窗口满时同时显示最早恢复时刻，北京时间跨日会带日期。

启用处理表情时，已受理的提问使用配置的处理表情，限流提问使用 QQ “闭嘴”（编号 8）；若处理表情已设为 8，则限流用“困”（编号 25）。

**群会话。** 开启后，同一群的连续提问接在同一段多轮对话里：每次提问追加一轮，内容是上一轮之后的新群消息和本次提问；模型上一轮的回复原样保留。群友标签和 `#m<n>` 编号在一段内固定。回答发出（或在仅记录模式下记录）后才接入会话；失败、取消、未发送和备用模型的回答不接入。同时进行的提问从同一轮继续，只有最先发出的接入。撤回段内消息、屏蔽段内发言人、更换回答模型、人设或工具时开新段。会话保存在内存中，重启后重新开始。群个性化可在 `group_profiles.<群号>.session` 中覆盖 `enabled`、`idle_minutes` 等字段。

会话让回答口径更一致，也让供应商的隐式前缀缓存可以命中；但请求比无状态请求长，总费用通常会增加。以 Gemini 4 Hydrogen 为例，续接时上一轮请求除最后约 2k token 外可以命中缓存，缓存约 2–3 分钟后失效。

**群聊归档。** `archive.enabled` 开启后，群里每条聊天消息的文字另存一份（图片、表情、卡片、转发只记占位，单条最多 500 字），保留 `retention_days` 天（默认 180），与 `retention` 的消息保留规则互不影响。屏蔽用户、屏蔽群和关闭普通采集的群不归档；撤回的消息从归档删除。关闭归档后停止记录和检索，已有内容按保留期在清理时删除；要立即清除指定成员或日期之后的内容，使用 `qunbot archive clear --instance <实例> --group <群号> --user <QQ号>` 或 `--since YYYY-MM-DD`。开启后，回答模型可以调用两个工具：`grep_group_history` 按 1–5 个关键词（每个至少 2 字，全部或任一匹配）、发言人和日期搜索，新的在前，返回命中总数和最早可查日期；`read_group_history` 读取某条结果前后各最多 10 条。两者只查本群、只查当前提问之前，按查询时的屏蔽名单过滤，发言人用材料中的群友代称，账号数字被遮盖，机器人以前的回答标为“不是证据”。每次提问最多搜索 `searches` 次、读取 `reads` 次，单次结果约 1.5k token。群个性化可在 `group_profiles.<群号>.archive` 中覆盖 `enabled` 和 `retention_days`。

已有的 SnowLuma 历史可一次导入：`qunbot archive import --instance <实例>`（加 `--dry-run` 只统计）。导入经 SQLite 备份接口复制 SnowLuma 的 `messages.db`，在副本上读取，按群号和消息号去重，跳过未开启归档的群、屏蔽成员、超过保留期的消息和消息库中已记录为撤回的消息，运行中的服务不受影响。SnowLuma 不记录群消息撤回，Qunbot 开始记录之前被撤回的消息无法识别，导入后可用 `archive clear` 删除。

**连续对话接续。** `session.compaction` 默认为 `off`：一段达到 `max_tokens` 或 `max_age_minutes` 后，下一次提问从零开始。设为 `recent` 后，只要群里的对话仍在进行（距上一轮不超过 `idle_minutes`），达到上限时不再从零开始：新的一段先带上最近 `carry_turns` 轮（默认 2）的提问、已发出的回答、被引用的消息和仍可读取的图片，群友标签和编号保持不变，然后照常往后接。这部分由程序整理，不额外调用模型。距上一轮不超过 `hot_seconds`（默认 180 秒）时缓存仍有效，会话可以越过 `max_tokens` 继续增长，直到 `hard_max_tokens`（默认 12000）才整理。闲置超过 `idle_minutes`、撤回或屏蔽段内消息、配置变化时，仍从零开始。

上下文方案 C 的规则集合、框架权限、工具 schema、模型工具循环的内部轮次上限仍由代码定义。`runtime` 不开放尚未接入在线行为的 `moderation_threshold`、`retention_days`；原文保留使用 `retention`。工具次数可设为 0，其余数量/期限为正数；工具执行同时受总期限和模型轮次上限约束。

## 文件布局与兼容

版本 1 的 `instance_root` 相对设置文件解析，其他路径相对实例根解析。例如仓库内 `config/settings.example.toml` 使用 `instance_root=".."`。更换工作目录不改变已经指定的配置文件所指向的数据。

旧格式保持相对设置文件解析；旧的 secrets / SnowLuma 缺省位置按原目录的同级部署布局解析。新格式使用实例内 `private/credentials.env`，SnowLuma 自同步需显式配置 `serve.snowluma_db`。现有本机配置仍可加载。

迁移会固定原有数据与凭据路径，源文件保持原样；不会搬动数据库或读取密钥。旧 `endpoint`、`alternatives` 是未消费的示例字段，迁移时去除。搜索的旧配置模式保留原行为，新模式使用 search/fetch 绑定；两套键混用会报错，避免改变旧补充搜索语义。

```powershell
uv run qunbot config validate --settings config/settings.example.toml
uv run qunbot config effective --settings config/settings.example.toml
uv run qunbot config schema

# 仅预览；加 --output 才写新文件，目标存在时拒绝覆盖
uv run qunbot config migrate --settings config/settings.toml
uv run qunbot config migrate --settings config/settings.toml --output data/config-import/settings.toml
```

## 人设与模型连接

```toml
[prompts]
persona = "config/prompts.example/persona.md"
# evidence = "config/prompts/evidence.md"
# search = "config/prompts/search.md"

[connections.official]
protocol = "openai-compatible"
base_url = "https://example.invalid"
api_key_env = "ANSWER_KEY"

[models.answer]
connection = "official"
model = "example-model"
price_input = 1.0
price_output = 2.0

[model_roles]
answer = "answer"
decision = "answer"
fallback = "answer"
```

连接和模型同时声明同一字段时须一致。多个模型可引用同一连接，报价仍由各模型维护。原有 `[models.<alias>]` 内联连接写法继续支持。

未配置提示词的部分使用内置默认。自定义 persona 而未指定 reply_contract 时，末尾契约只要求聚焦本次问题、按人设格式回答，不再额外要求一两句。内置权限规则仍参与组装。search 模板只支持 `{query}`，其他资源按普通文本加载。

```powershell
uv run qunbot config prompts --settings config/settings.toml
```

该预览与模型调用共用组装函数。发布时提示词会复制进版本目录，后续编辑草稿文件不会改变已发布版本。

## 凭据

原 `secrets_file` 与 `*_env` 字段继续工作。需要明确来源或通过面板管理时添加引用：

```toml
[credentials.ANSWER_KEY]
source = "managed"
# 默认写入实例的 private/credentials.env

[credentials.GEMINI_API_KEY]
source = "env"
key = "DEPLOYMENT_GEMINI_KEY"

[credentials.SNOWLUMA_WS_TOKEN]
source = "file"
path = "../secrets/group-agent.env"
```

`CredentialStore.update(name, action, value)` 仅写 managed 引用，action 为 keep/replace/clear；file/env 引用由外部维护。私有文件在写入内容前设置 Windows ACL 或 Unix 0600 权限，原子替换。删除 managed 值不会回退读取同名环境变量。

`config credential --settings ...` 仅输出配置状态；指定 `--name ANSWER_KEY --operation replace` 时从标准输入读取一行，密钥不放在命令参数中。可用占位文件为 `config/credentials.example.txt`。

## 草稿、发布和采用

TOML 文件就是可编辑草稿。程序调用可使用 `ConfigService.save_draft(data, expected)` 校验、保存；`expected` 为之前草稿的内容摘要，避免覆盖别人的编辑。配置发布需提供之前看到的发布修订号；首次发布不传。

```powershell
uv run qunbot config diff --settings config/settings.toml --store data/config
uv run qunbot config publish --settings config/settings.toml --store data/config
uv run qunbot config status --store data/config
# 后续发布：--expect 上一步返回的 revision
uv run qunbot config publish --settings config/settings.toml --store data/config --expect <当前修订号>
uv run qunbot config rollback --store data/config --revision <历史修订号> --expect <当前修订号>
```

服务将完整配置和提示词写入新版本，校验后原子切换 `active.json`；操作记录保存在版本的 `revision.json`。发布前检查当前版本是否被外部改写。未切换指针的中断产物不影响有效版本。

采用版本目录启动时使用 `--settings data/config`。运行入口完成组装后记录实际采用版本与 CLI 覆盖，退出或启动失败时记录状态。`config status` 的 `last_runtime` 是带时间的最近记录，不承担在线心跳检测；`pending` 表示目标发布版本是否尚未成功采用。日志与 run 诊断均包含有效配置摘要。

回滚创建新发布记录，重新检查显式凭据引用；不回滚凭据文件和业务账本。普通有效配置输出与差异会隐藏账号、私有端点及存储路径；版本存储本身包含完整非密钥配置，应放在本地忽略目录。原始凭据文件不复制进版本。


## Web 热更新

管理台使用同一 ConfigService 保存草稿和版本，经 RuntimeController 应用后确认实际采用。人格、模型、搜索、请求限制作用于新 run；预算、权限、调度和关闭发送/联网更新现有运行对象。网关、目录等维护项在停止引擎后重新启动采用。字段分类由 `config schema` 输出。

CLI `config publish` 只发布配置；正在运行的 Web 引擎需通过面板应用，或在下次受控启动时读取。具体入口、密钥登录与生效规则见 [本地管理台](ADMIN-UI.md)。
