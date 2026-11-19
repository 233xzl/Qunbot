# 致谢与设计参考

Qunbot 的群聊上下文、教学知识库、运行预算和发送策略是在实际群聊需求及回放验证中逐步实现的。以下项目为本项目提供运行组件、协议实现参考和设计启发。

## 运行组件

- [Pydantic AI](https://github.com/pydantic/pydantic-ai)：模型调用、结构化结果和工具循环。
- [Pillow](https://github.com/python-pillow/Pillow)：图片处理。
- [websockets](https://github.com/python-websockets/websockets)：WebSocket 通信。
- FastAPI、Starlette 与 Uvicorn 提供本地管理 API 和 Web 服务。tomli-w 用于写入版本化配置。
- Google Gen AI、OpenAI 和 Anthropic SDK：供应商协议接入。

具体依赖与版本见 `pyproject.toml` 和 `uv.lock`。第三方组件各自保留其许可证；本项目许可证不替代第三方组件的许可条款。

## 已采用的设计参考与兼容来源

- [AstrBot](https://github.com/AstrBotDevs/AstrBot)：早期部署与迁移对照框架。上下文策略 A 对照的是本项目作者在本地 AstrBot 部署中修改过的筛选规则；教学知识库和人设延续了作者编写的本地插件与配置。
- [NoneBot OneBot adapter](https://github.com/nonebot/adapter-onebot/tree/2f5930d4f6c7fb3a82cf3201639e8f0b0a6c124e)：阅读其 echo 匹配、连接和消息预处理实现，作为 OneBot 网关设计参考。
- [Pi](https://github.com/badlogic/pi-mono/tree/35019ae989fc6e45eb18d0c757ef81dd18cb30e3) 与 [DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/tree/39163dd30f703a79079e2d2446545cff3c453738)：模型、协议适配与运行层的职责划分参考。Qunbot 使用 Python 与 Pydantic AI 实现运行链路。
- SnowLuma：本项目的 QQ 通信层，通过 OneBot 接口连接；相关迁移与同步工具适配其本地消息存储。

## 组件研究

以下项目在早期组件选型及后续记忆方案研究中提供参考；当前实现范围以项目说明为准。

- [MaiBot](https://github.com/Mai-with-u/MaiBot/tree/c6429547af711555e3d4d1b8d04b413572cb616c)：群聊历史筛选、记忆生命周期与纠错。
- [Mem0](https://github.com/mem0ai/mem0/tree/ed1f10fc87080f0b67f99be370817504888462ab)：事实提取、去重与变更历史。
- [Graphiti](https://github.com/getzep/graphiti/tree/fb60e59d4a208c99e7d0186ee6335d2a257f17f2)：事实的时间与来源表达。
- [LangGraph](https://github.com/langchain-ai/langgraph/tree/ec72ddf3c28fed766f1041e12e5c3f499a490754)：任务状态与持久化恢复。
- [Haystack](https://github.com/deepset-ai/haystack/tree/9dc5efe716056dd7c1199ded09342be8c1cdcb12)：检索结果融合。

## 数据兼容

知识库迁移工具支持旧本地插件的数据结构，该部署曾导入 [CHIME](https://github.com/yuboxie/chime) 基础知识。项目源码仓库不附带真实知识库、群聊记录、教学图片或该数据集内容。
