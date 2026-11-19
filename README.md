# Qunbot

Qunbot 是一个 QQ 群聊机器人后端。SnowLuma 通过 OneBot 提供消息连接，Qunbot 负责群聊上下文、模型调用、搜索、显式教学和回复发送。每个群可以继承全局配置，也可以单独指定人设、模型和每日预算。

## 启动本地管理台

准备 Python 3.12 或更新版本及 uv，在仓库根目录执行以下命令。首次初始化会显示管理密钥，请保存以便登录。

```sh
uv sync --locked
uv run qunbot admin init --instance data/demo
uv run qunbot admin serve --instance data/demo --demo
```

Windows 安装完成后，可用仓库内的 `.\qunbot.cmd` 代替 `uv run qunbot`。启动器会使用本仓库虚拟环境中的命令。

打开 `http://127.0.0.1:8765`，登录后点击“启动 Agent”，在总览发起模拟请求。演示使用本地假模型，运行数据保存在 `data/demo`。以后使用同一实例只需执行 `admin serve`。退出管理服务可按 Ctrl+C。

## 接入 QQ

使用独立实例，例如 `uv run qunbot admin init --instance data/my-agent`，随后执行 `uv run qunbot admin serve --instance data/my-agent`。在管理台配置机器人及主人身份、SnowLuma 本地 WebSocket 地址和令牌、模型连接及其凭据。设置预算和允许发送的范围后启动引擎。分群设置在“群个性化”页面配置。完整步骤见[管理台手册](docs/ADMIN-UI.md)和[配置说明](docs/CONFIGURATION.md)。

无人值守前台运行可使用 `uv run qunbot --serve --settings data/my-agent/config-history`，由宿主的服务管理器处理进程重启。管理台模式和前台模式一次只运行一个引擎。`group-agent` 保留为兼容命令，Python 导入名为 `group_agent`。

## 离线验证

```sh
uv run qunbot config validate --settings config/settings.example.toml
uv run pytest -q -p no:cacheprovider
uv run python scripts/replay/run_cases.py tests/replay/cases --check
```

[40 条合成回放](docs/SYNTHETIC-CASES.md)覆盖触发、上下文、权限、预算和发送行为。单元与集成测试使用虚构数据。公平调度、发送结果不确定时的核对和无头浏览器读取是现有的三个预期失败项。

## 许可

Qunbot 采用 [Apache-2.0](LICENSE)。第三方组件及设计参考见[致谢](ACKNOWLEDGEMENTS.md)。
