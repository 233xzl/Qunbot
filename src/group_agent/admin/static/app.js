"use strict";
const $ = (s) => document.querySelector(s),
  esc = (v) =>
    String(v ?? "").replace(
      /[&<>"']/g,
      (c) =>
        ({
          "&": "&amp;",
          "<": "&lt;",
          ">": "&gt;",
          '"': "&quot;",
          "'": "&#39;",
        })[c],
    );
const pages = [
  ["监控", "overview", "◷", "总览"],
  ["", "requests", "☷", "请求与日志"],
  ["", "usage", "▥", "用量与账单"],
  ["配置", "persona", "✧", "人格与提示词"],
  ["", "profiles", "⚙", "群个性化"],
  ["", "models", "⬡", "模型与供应商"],
  ["", "search", "⌕", "搜索与工具"],
  ["", "policy", "⚑", "群聊与运行策略"],
  ["", "budget", "◉", "预算策略"],
  ["管理", "storage", "▤", "存储与知识"],
  ["", "system", "⚙", "系统设置"],
];
const descriptions = {
  overview: "了解群聊 Agent 的运行状态、用量与近期活动。",
  requests: "查看每次请求的状态、耗时与调用记录。",
  usage: "最近 30 天 · 日期按 GMT+8 显示 · 费用为调用估算。",
  persona: "塑造回答风格。应用后，新请求使用新的人格与提示词。",
  profiles: "按群覆盖提示词、模型用途和每日限额。",
  models: "管理模型连接、调用参数与回答角色。",
  search: "设置搜索和网页读取，以及每次请求的工具预算。",
  policy: "设置群聊访问范围、并发与回复策略。",
  budget: "调整每日消费上限、降级阈值与计费参数。",
  storage: "查看数据库占用，预览保留策略和创建数据库快照。",
  system: "管理连接凭据、配置版本与实例设置。",
};
const labels = {
  breaker_failures: "熔断连续失败次数",
  breaker_cooldown_s: "熔断冷却时间（秒）",
  bot_id: "机器人账号",
  owner_id: "主人账号",
  blocked_users: "屏蔽用户",
  blocked_groups: "屏蔽群聊",
  daily_cost_cny: "每日普通用户预算（元）",
  deadline_seconds: "请求总时限（秒）",
  context_messages: "上下文消息上限",
  global_active: "全局运行并发",
  user_active: "每用户运行并发",
  global_queue: "全局等待队列",
  user_queue: "每用户等待队列",
  degrade_ratio: "降级阈值比例",
  cap_ratio: "普通用户停止阈值比例",
  web_enabled: "允许联网",
  answer_tool_loop: "启用回答工具循环",
  decision_mode: "判断模式",
  answer_thinking: "回答思考级别",
  answer_same_model_retry: "同模型重试",
  live_send: "发送群聊回复",
  owner_control: "群内主人控制指令",
  fallback_reply: "失败时固定回复",
  processing_reaction: "处理中表情",
  self_sync: "同步机器人消息",
  db_path: "数据库路径",
  model: "模型 ID",
  protocol: "协议",
  api_key_env: "密钥引用名称",
  base_url: "服务地址",
  base_url_env: "服务地址变量",
  project_env: "云项目变量",
  timeout_seconds: "调用超时（秒）",
  price_input: "输入单价 / 百万 token",
  price_output: "输出单价 / 百万 token",
  price_cached_input: "缓存输入单价",
  price_currency: "计价币种",
  thinking_level: "思考级别",
  thinking_budget_tokens: "思考 token 预算",
  max_output_tokens: "最大输出 token",
  channel: "计费渠道",
  connection: "供应商连接",
  answer: "回答模型",
  decision: "判断模型",
  fallback: "备用模型",
  search: "搜索主模型",
  fetch: "网页读取主模型",
  search_fallback: "搜索备用模型",
  fetch_fallback: "网页读取备用模型",
  price_per_call: "每次搜索/读取价格（元）",
  web_searches: "每次请求搜索上限",
  web_reads: "每次请求读取上限",
  tool_timeout_seconds: "工具超时（秒）",
  finish_reserve_seconds: "回答预留时间（秒）",
  persona: "人格",
  evidence: "取证规则",
  reply_contract: "输出格式",
  tool_decision: "工具判断",
  usd_cny: "美元兑人民币",
  google_search_usd_per_1000: "Google 搜索美元单价 / 千次",
  google_free_queries_per_month: "每月免费查询额度",
  billing_cycle_day: "账单周期起始日",
  recent_hours: "普通聊天保留小时",
  related_days: "关联请求保留天数",
  cleanup_interval_minutes: "清理间隔（分钟）",
  media_max_mb: "媒体容量上限（MB）",
  disabled_groups: "关闭普通消息采集的群",
  source: "凭据来源",
  key: "凭据变量名",
  path: "凭据文件路径",
  location: "区域",
  vertex: "使用 Vertex",
  api_version: "API 版本",
};
const modeNames = {
  immediate: "立即生效",
  new_run: "新请求生效",
  next_cleanup: "下一轮清理",
  maintenance: "需停止并重新启动",
  disable_now_enable_maintenance: "关闭立即生效，启用需重新启动",
};
const stateNames = {
  running: "运行中",
  paused: "已暂停",
  stopped: "已停止",
  failed: "启动失败",
  starting: "启动中",
  draining: "停止中",
  succeeded: "已完成",
  queued: "排队中",
  cancelled: "已取消",
  rejected: "已拒绝",
  expired: "已超时",
};
let cfg = null,
  status = {},
  page = "overview",
  dirty = false,
  offset = 0,
  polling = false,
  setupStep = -1;
const setupPages = [
  "policy",
  "models",
  "system",
  "search",
  "persona",
  "budget",
];
async function api(path, body) {
  const r = await fetch("/api/" + path, {
    method: body === undefined ? "GET" : "POST",
    headers:
      body === undefined
        ? {}
        : { "Content-Type": "application/json", "X-Qunbot-Request": "1" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const data = await r.json();
  if (!r.ok) {
    if (r.status === 401 && path !== "login") showLogin();
    throw Error(data.error || "请求失败");
  }
  return data;
}
function toast(message) {
  $("#toast").textContent = message;
  $("#toast").hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => ($("#toast").hidden = true), 4500);
}
function changed() {
  dirty = true;
  $("#dirty").hidden = false;
}
function showLogin() {
  $("#shell").hidden = true;
  $("#login").hidden = false;
  cfg = null;
}
function money(v) {
  return Number(v || 0).toFixed(2);
}
function num(v) {
  return Number(v || 0).toLocaleString("zh-CN");
}
function time(v) {
  return v
    ? new Date(v * 1000).toLocaleString("zh-CN", { hour12: false })
    : "—";
}
function panel(title, body, extra = "") {
  return `<section class="panel"><div class="panel-head"><h2>${esc(title)}</h2>${extra}</div>${body}</section>`;
}
function badge(state) {
  return `<span class="badge ${state === "running" || state === "succeeded" ? "green" : state === "failed" ? "orange" : ""}">${esc(stateNames[state] || state)}</span>`;
}
function empty(text) {
  return `<div class="empty">${esc(text)}</div>`;
}
function cards(items) {
  return `<div class="cards">${items.map(([label, value, unit, foot]) => `<div class="card"><div class="card-label">${label}</div><div class="metric">${value}<small>${unit || ""}</small></div><div class="card-foot">${foot || ""}</div></div>`).join("")}</div>`;
}
function modal(title, html, confirm) {
  $("#dialog-content").innerHTML = `<h2>${esc(title)}</h2>${html}`;
  $("#dialog-confirm").hidden = !confirm;
  $("#dialog-confirm").onclick = async () => {
    try {
      await confirm();
      $("#dialog").close();
    } catch (e) {
      toast(e.message);
    }
  };
  $("#dialog").showModal();
}
$("#dialog-close").onclick = () => $("#dialog").close();
$("#login-form").onsubmit = async (e) => {
  e.preventDefault();
  $("#login-error").textContent = "";
  try {
    await api("login", { key: $("#key").value });
    $("#key").value = "";
    await enter();
  } catch (e) {
    $("#login-error").textContent = e.message;
  }
};
$("#logout").onclick = async () => {
  await api("logout", {});
  showLogin();
};
$("#menu").onclick = () => $("#sidebar").classList.toggle("open");
$("#nav").innerHTML = pages
  .map(
    ([group, id, icon, label]) =>
      `${group ? `<div class="nav-group">${group}</div>` : ""}<button class="nav-item" data-page="${id}"><span class="nav-icon">${icon}</span>${label}</button>`,
  )
  .join("");
$("#nav").onclick = (e) => {
  const b = e.target.closest("[data-page]");
  if (b) navigate(b.dataset.page);
};
async function enter() {
  status = await api("status");
  cfg = await api("config");
  $("#login").hidden = true;
  $("#shell").hidden = false;
  dirty = false;
  $("#dirty").hidden = true;
  await navigate(page);
}
async function navigate(id) {
  page = id;
  window.scrollTo(0, 0);
  $("#sidebar").classList.remove("open");
  document
    .querySelectorAll("[data-page]")
    .forEach((x) => x.classList.toggle("active", x.dataset.page === id));
  $("#title").textContent = pages.find((p) => p[1] === id)[3];
  $("#breadcrumb").textContent = "工作空间 / " + $("#title").textContent;
  $("#subtitle").textContent = descriptions[id];
  $("#editor-actions").hidden = ![
    "persona",
    "profiles",
    "models",
    "search",
    "policy",
    "budget",
    "system",
  ].includes(id);
  try {
    await render();
  } catch (e) {
    toast(e.message);
  }
}
function updateBadge() {
  $("#engine-badge").outerHTML = badge(status.state).replace(
    'class="badge',
    'id="engine-badge" class="badge',
  );
  $("#connection").textContent = status.demo
    ? "本地实例 · 模拟模式"
    : "本地实例";
}
async function control(action) {
  const b = $(`[data-action="${action}"]`);
  if (b) b.disabled = true;
  try {
    status = await api("runtime/" + action, {});
    toast(status.error || "运行状态已更新");
    await refresh();
  } catch (e) {
    toast(e.message);
  } finally {
    if (b) b.disabled = false;
  }
}
function bindControls() {
  document
    .querySelectorAll("[data-action]")
    .forEach((b) => (b.onclick = () => control(b.dataset.action)));
}
function controls() {
  return `<div class="toolbar"><button class="primary" data-action="start" ${["running", "paused", "starting"].includes(status.state) ? "disabled" : ""}>启动 Agent</button><button class="secondary" data-action="${status.state === "paused" ? "resume" : "pause"}" ${!["running", "paused"].includes(status.state) ? "disabled" : ""}>${status.state === "paused" ? "恢复回复" : "暂停回复"}</button><button class="secondary" data-action="stop" ${["stopped", "failed"].includes(status.state) ? "disabled" : ""}>停止</button></div>`;
}
function chart(days) {
  const max = Math.max(...days.map((x) => x.cost), 0.01);
  return `<div class="chart">${days.length ? days.map((d, i) => `<div class="bar" style="height:${Math.max(1, (d.cost / max) * 100)}%" title="${esc(d.day)} · ¥${money(d.cost)}">${days.length < 10 || i % Math.ceil(days.length / 7) === 0 ? `<span>${esc(d.day.slice(5))}</span>` : ""}</div>`).join("") : '<div class="chart-empty">还没有消费记录，运行后将在这里展示趋势</div>'}</div>`;
}
let observedGroup = null;
function scopeQuery() {
  return observedGroup === null ? "" : "&group_id=" + encodeURIComponent(observedGroup);
}
async function groupFilter(includeUnknown) {
  const ids = await api("groups");
  const choices = [[null, "全部群"], ...ids.map((id) => [id, id])];
  if (includeUnknown) choices.push(["", "未归属"]);
  return `<label>群聊 <select id="observed-group">${choices.map(([id, label]) => `<option value="${esc(id ?? "*")}" ${observedGroup === id ? "selected" : ""}>${esc(label)}</option>`).join("")}</select></label>`;
}
function bindFilter() {
  $("#observed-group").onchange = (e) => {
    observedGroup = e.target.value === "*" ? null : e.target.value;
    offset = 0;
    render();
  };
}
function requestTable(rows) {
  return rows.length
    ? `<div class="table-wrap"><table><thead><tr><th>请求</th><th>群号</th><th>时间</th><th>状态</th><th>耗时</th><th>路线</th></tr></thead><tbody>${rows.map((r) => `<tr><td><button class="link request-detail" data-id="${esc(r.run_id)}">${esc(r.run_id.slice(0, 10))}</button></td><td>${esc(r.group_id || "—")}</td><td>${time(r.created_at)}</td><td>${badge(r.state)}</td><td>${r.ended_at ? Math.max(0, r.ended_at - r.created_at).toFixed(1) + " 秒" : "—"}</td><td>${esc(r.route || "—")}</td></tr>`).join("")}</tbody></table></div>`
    : empty("暂无请求记录");
}
function bindRequests() {
  document.querySelectorAll(".request-detail").forEach(
    (b) =>
      (b.onclick = async () => {
        try {
          const d = await api("requests/" + b.dataset.id);
          let detail = {};
          try {
            detail = JSON.parse(d.detail || "{}");
          } catch {}
          const profile = detail.group_overrides?.length ? detail.group_overrides.join("、") : "继承全局";
          const scope = { global: "全局预算", group: "本群预算" }[detail.budget_scope];
          modal(
            "请求详情",
            `<p class="muted">${esc(d.run_id)}</p><div class="notice">${esc(stateNames[d.state] || d.state)} · ${time(d.created_at)}<br>配置：${esc(detail.config_revision || "未记录")}<br>群配置：${detail.group_overrides ? esc(profile) : "—"}${scope ? `<br>预算触发来源：${scope}` : ""}</div><pre>${esc(JSON.stringify({ ...d, detail }, null, 2))}</pre>`,
          );
        } catch (e) {
          toast(e.message);
        }
      }),
  );
}
async function render() {
  updateBadge();
  const content = $("#content");
  if (page === "overview") {
    const [u, rows] = await Promise.all([api("usage"), api("requests")]);
    content.innerHTML =
      (!status.demo && cfg.data.identity?.bot_id === "BOT"
        ? panel(
            "首次配置",
            `<p class="muted">按步骤设置账号、模型、连接凭据、人设和预算，完成后再启动群聊引擎。</p><button id="setup-start" class="primary" style="margin-top:16px">开始配置 →</button>`,
          )
        : "") +
      panel(
        "运行状态",
        `<div class="status-line"><span><i class="dot ${status.gateway_connected ? "green" : ""}"></i>${status.demo ? "模拟模式 · 无外部连接" : status.gateway_connected ? "SnowLuma 已连接" : "SnowLuma 未连接"}</span><span class="muted">运行 ${status.active || 0} · 排队 ${status.queued || 0}</span></div>${status.error ? `<div class="notice">${esc(status.error)}</div>` : ""}${status.pending ? '<div class="notice">有待应用配置；维护类修改需停止引擎后重新启动。</div>' : ""}`,
        controls(),
      ) +
      cards([
        [
          "今日消费",
          "¥" + money(status.usage?.daily_spent_cny),
          "CNY",
          "普通用户预算 ¥" +
            money(
              status.usage?.daily_cap_cny ?? cfg.data.budgets?.daily_cost_cny,
            ),
        ],
        ["请求次数", num(u.requests), "次", "最近 30 天"],
        ["Token 用量", num(u.tokens), "", "最近 30 天 · 输入 + 输出"],
      ]) +
      panel(
        "消费趋势",
        chart(u.days),
        '<span class="muted">近 30 天 · CNY</span>',
      ) +
      panel(
        "近期请求",
        requestTable(rows.slice(0, 5)),
        `<button id="all-requests" class="link">查看全部 →</button>`,
      ) +
      (status.demo
        ? panel(
            "模拟试聊",
            `<p class="muted">通过正常请求流程运行，使用本地假模型，不发送 QQ 消息。</p><form id="demo-form" class="inline-form"><input id="demo-group" aria-label="模拟群号" placeholder="模拟群号" value="DEMO"><input id="demo-text" placeholder="输入一条模拟群聊提问" value="你好，介绍一下自己"><button class="primary">发送模拟请求</button></form>`,
          )
        : "");
    bindControls();
    bindRequests();
    if ($("#setup-start"))
      $("#setup-start").onclick = () => {
        setupStep = 0;
        navigate(setupPages[0]);
      };
    $("#all-requests").onclick = () => navigate("requests");
    if ($("#demo-form"))
      $("#demo-form").onsubmit = async (e) => {
        e.preventDefault();
        try {
          await api("demo/request", { text: $("#demo-text").value, group_id: $("#demo-group").value });
          toast("模拟请求已提交");
        } catch (e) {
          toast(e.message);
        }
      };
    return;
  }
  if (page === "usage") {
    const u = await api("usage?" + scopeQuery());
    content.innerHTML =
      await groupFilter(true) + cards([
        ["消费金额", "¥" + money(u.cost), "CNY"],
        ["请求次数", num(u.requests), "次"],
        ["Token 用量", num(u.tokens)],
      ]) +
      panel("消费趋势", chart(u.days)) +
      panel(
        "模型用量",
        u.providers.length
          ? `<div class="table-wrap"><table><thead><tr><th>供应商 / 模型</th><th>调用</th><th>Token</th><th>消费</th></tr></thead><tbody>${u.providers.map((p) => `<tr><td>${esc(p.provider)}<br><small class="muted">${esc(p.model)}</small></td><td>${num(p.calls)}</td><td>${num(p.tokens)}</td><td>¥${money(p.cost)}</td></tr>`).join("")}</tbody></table></div>`
          : empty("暂无用量记录"),
      );
    bindFilter();
    return;
  }
  if (page === "requests") {
    if (observedGroup === "") observedGroup = null;
    const rows = await api("requests?offset=" + offset + scopeQuery());
    content.innerHTML = await groupFilter(false) + panel(
      "请求列表",
      requestTable(rows),
      `<div class="toolbar"><button id="prev" class="secondary" ${offset === 0 ? "disabled" : ""}>上一页</button><button id="next" class="secondary" ${rows.length < 50 ? "disabled" : ""}>下一页</button></div>`,
    );
    $("#prev").onclick = () => {
      offset = Math.max(0, offset - 50);
      render();
    };
    $("#next").onclick = () => {
      offset += 50;
      render();
    };
    bindRequests();
    bindFilter();
    return;
  }
  if (page === "storage") {
    const rows = await api("storage");
    content.innerHTML =
      cards(
        rows.map((x) => [
          {
            messages: "消息与请求",
            knowledge: "长期知识",
            billing: "账号账本",
          }[x.name],
          (x.bytes / 1048576).toFixed(2),
          "MB",
        ]),
      ) +
      panel(
        "存储位置",
        rows.map(
          (x) =>
            `<div class="field"><label>${esc(x.name)}</label><span class="muted">${esc(x.path || "未配置")}</span></div>`,
        ),
      ) +
      panel(
        "维护",
        `<p class="muted">清理预览不删除数据；数据库快照需先停止引擎。</p><div class="toolbar" style="margin-top:16px"><button id="preview-cleanup" class="secondary">预览清理</button><button id="backup" class="secondary">创建数据库快照</button></div>`,
      );
    $("#preview-cleanup").onclick = async () => {
      try {
        modal(
          "清理预览",
          `<pre>${esc(JSON.stringify(await api("storage/preview", {}), null, 2))}</pre>`,
        );
      } catch (e) {
        toast(e.message);
      }
    };
    $("#backup").onclick = async () => {
      try {
        const r = await api("storage/backup", {});
        modal(
          "数据库快照已保存",
          `<p>${esc(r.path)}</p><p class="muted">包含消息、知识与账本数据库；媒体及配置仍在实例目录中。</p>`,
        );
      } catch (e) {
        toast(e.message);
      }
    };
    return;
  }
  renderEditor();
}
function field(path, value, definition = {}) {
  definition = Object.keys(definition).length
    ? definition
    : cfg.schema[path[0] + ".*"]?.[path.at(-1)] || {};
  const key = path.at(-1),
    type = definition.type || typeof value,
    id = "f-" + path.join("-");
  const isArray = Array.isArray(value) || /frozenset|tuple/.test(type),
    isBool = typeof value === "boolean" || type === "bool",
    isNum = typeof value === "number" || /^(int|float)/.test(type);
  const data = esc(JSON.stringify(path));
  let input;
  if (isBool)
    input = `<select data-path="${data}" data-kind="bool" id="${id}"><option value="true" ${value ? "selected" : ""}>开启</option><option value="false" ${!value ? "selected" : ""}>关闭</option></select>`;
  else
    input = `<input data-path="${data}" data-kind="${isArray ? "array" : isNum ? "number" : "text"}" id="${id}" ${isNum ? 'type="number" step="any"' : ""} value="${esc(isArray ? (value || []).join(", ") : value)}" placeholder="${isArray ? "多个值用逗号分隔" : "未设置"}">`;
  return `<div class="field"><label for="${id}">${esc(labels[key] || key)}</label>${input}<small>${esc(modeNames[definition.apply] || "")}${Object.hasOwn(cfg.overrides, path.join(".")) ? " · 被启动参数覆盖" : ""}</small></div>`;
}
function section(group, title, keys) {
  const values = cfg.data[group] || {},
    defs = cfg.schema[group] || {};
  const names = keys || [
    ...new Set([...Object.keys(values), ...Object.keys(defs)]),
  ];
  const draw = (xs) =>
    `<div class="grid2">${xs.map((k) => field([group, k], values[k] ?? defs[k]?.default, defs[k])).join("")}</div>`;
  return panel(
    title,
    draw(names.slice(0, 8)) +
      (names.length > 8
        ? `<details><summary>更多设置（${names.length - 8}）</summary>${draw(names.slice(8))}</details>`
        : ""),
  );
}
function objects(group, title) {
  let html = "";
  for (const [name, values] of Object.entries(cfg.data[group] || {})) {
    if (typeof values !== "object" || values === null) continue;
    html += panel(
      name,
      `<div class="grid2">${Object.entries(values)
        .map(([k, v]) => field([group, name, k], v))
        .join(
          "",
        )}</div><button type="button" class="link add-field" data-group="${group}" data-name="${esc(name)}">＋ 添加参数</button>`,
      `${group === "credentials" ? `<span class="badge ${cfg.credentials[name]?.configured ? "green" : ""}">${cfg.credentials[name]?.configured ? "已配置" : "未配置"}</span>` : ""}<button class="delete" data-delete="${esc(name)}" data-group="${group}">移除</button>`,
    );
  }
  return `<div class="panel-head"><h2>${title}</h2><button class="secondary add-object" data-group="${group}">＋ 添加</button></div>${html || empty("尚未添加，点击上方按钮创建")}`;
}
function profileEditor() {
  const profiles = cfg.data.group_profiles || {};
  const roles = cfg.data.model_roles || {};
  const aliases = Object.entries(cfg.data.models || {}).filter(([, m]) => m && typeof m === "object" && m.protocol !== "systemone").map(([id]) => id);
  let html = '<form id="add-profile" class="inline-form"><input id="profile-id" placeholder="群号" required><button class="secondary">添加群</button></form>';
  for (const [gid, profile] of Object.entries(profiles)) {
    const enabled = profile.enabled === true;
    const scoped = (section, key) => `data-profile="${esc(gid)}" data-section="${section}" data-key="${key}"`;
    const fields = [];
    for (const key of ["answer", "decision", "fallback"]) {
      const value = profile.model_roles?.[key];
      const inherited = roles[key] || (key === "fallback" ? "deepseek" : key === "decision" ? roles.answer || "gemini" : "gemini");
      fields.push(`<label class="field">${esc(labels[key] || key)}<select ${scoped("model_roles", key)}><option value="">继承全局 · ${esc(inherited)}</option>${aliases.map(id => `<option ${value === id ? "selected" : ""} value="${esc(id)}">本群 · ${esc(id)}</option>`).join("")}</select></label>`);
    }
    for (const key of Object.keys(cfg.schema.prompts)) {
      const value = profile.prompts?.[key] || "";
      const inherited = cfg.prompts[key] || cfg.prompt_defaults[key] || "";
      fields.push(`<label class="field">${esc(labels[key] || key)} · ${value ? "本群覆盖" : "继承全局"}<textarea ${scoped("prompts", key)} placeholder="${esc(inherited)}">${esc(value)}</textarea></label>`);
    }
    fields.push(`<label class="field">本群每日限额（元）<input type="number" step="any" min="0" ${scoped("", "daily_cost_cny")} value="${esc(profile.daily_cost_cny)}" placeholder="仅使用全局预算"></label>`);
    html += panel(`群 ${esc(gid)}`, `<label><input type="checkbox" ${scoped("", "enabled")} ${enabled ? "checked" : ""}> 启用个性化</label>${enabled ? '<p class="muted">清空字段即可恢复继承。</p>' + fields.join("") : ""}`, `<button class="delete" data-group="group_profiles" data-delete="${esc(gid)}">移除</button>`);
  }
  return html;
}
function bindProfiles() {
  $("#add-profile").onsubmit = (e) => {
    e.preventDefault();
    const id = $("#profile-id").value.trim();
    if (!/^[A-Za-z0-9_-]{1,64}$/.test(id) || id in (cfg.data.group_profiles || {})) return toast("请输入未添加的群号");
    (cfg.data.group_profiles ??= {})[id] = {enabled: false};
    changed();
    renderEditor();
  };
  document.querySelectorAll("[data-profile]").forEach(el => el.onchange = () => {
    const {profile: gid, section, key} = el.dataset;
    const profile = cfg.data.group_profiles[gid];
    const target = section ? (profile[section] ??= {}) : profile;
    if (el.type === "checkbox") target[key] = el.checked;
    else if (!el.value.trim()) delete target[key];
    else if (el.type === "number") target[key] = Number(el.value);
    else target[key] = el.value;
    changed();
    if (key === "enabled") renderEditor();
    else if (section === "prompts") el.parentNode.firstChild.textContent = `${labels[key] || key} · ${target[key] ? "本群覆盖" : "继承全局"}`;
  });
}
function renderEditor() {
  let html = "";
  if (page === "profiles") html = profileEditor();
  if (page === "persona") {
    html = Object.entries({ persona: cfg.default_persona, ...cfg.prompts })
      .map(([name, text]) =>
        panel(
          labels[name] || name,
          `<textarea class="prompt" data-prompt="${esc(name)}" aria-label="${esc(labels[name] || name)}">${esc(text)}</textarea>`,
          `<span class="badge">新请求生效</span>`,
        ),
      )
      .join("");
    html += '<button id="add-prompt" class="secondary">＋ 添加提示词</button>';
  }
  if (page === "models")
    html =
      section("model_roles", "模型用途", ["answer", "decision", "fallback"]) +
      objects("connections", "供应商连接") +
      objects("models", "模型配置") +
      section("models", "熔断策略", ["breaker_failures", "breaker_cooldown_s"]);
  if (page === "search")
    html =
      section("agent", "联网开关", ["web_enabled"]) +
      section("web", "搜索与读取") +
      section("runtime", "工具预算", [
        "web_searches",
        "web_reads",
        "tool_timeout_seconds",
        "finish_reserve_seconds",
      ]);
  if (page === "policy")
    html =
      section("identity", "身份与访问") +
      section("budgets", "请求上下文", [
        "deadline_seconds",
        "context_messages",
      ]) +
      section("runtime", "运行策略") +
      section("agent", "回答策略") +
      section("serve", "QQ 与发送策略") +
      section("collection", "消息与媒体采集");
  if (page === "budget")
    html =
      '<div class="notice">每日上限针对普通用户。达到上限后主人仍可调用；修改预算保留已有消费与预留。</div>' +
      section("budgets", "每日预算", ["daily_cost_cny"]) +
      section("runtime", "降级与停止阈值", ["degrade_ratio", "cap_ratio"]) +
      section("billing", "账单与计价");
  if (page === "system")
    html =
      panel(
        "配置版本",
        `<p class="revision">已发布：${esc(cfg.published_revision || "—")}</p><p class="revision">运行采用：${esc(status.published_revision || "—")}</p><div class="toolbar" style="margin-top:14px"><button id="versions" class="secondary">版本历史与回滚</button><button id="reload-credentials" class="secondary">重新加载凭据</button></div>`,
      ) +
      objects("credentials", "凭据引用") +
      panel(
        "更新管理型凭据",
        `<p class="muted">先保存并应用凭据引用，再写入值。外部文件凭据在原文件中修改。</p><form id="credential-form"><div class="grid2" style="margin-top:18px"><div class="field"><label>引用名称</label><input id="credential-name" list="credential-names" required placeholder="例如 MODEL_API_KEY"><datalist id="credential-names">${Object.keys(
          cfg.credentials,
        )
          .map((n) => `<option value="${esc(n)}"></option>`)
          .join(
            "",
          )}</datalist></div><div class="field"><label>新值</label><input id="credential-value" type="password" required autocomplete="new-password"></div></div><button class="secondary">更新凭据并刷新客户端</button></form>`,
      ) +
      section("retention", "数据保留") +
      section("knowledge", "知识库");
  if (setupStep >= 0)
    html =
      panel(
        "首次配置 · " + (setupStep + 1) + " / " + setupPages.length,
        `<p class="muted">${esc(descriptions[page])} · 修改暂存在表单中，最后统一保存。</p><div class="toolbar" style="margin-top:12px"><button id="setup-back" class="secondary" ${setupStep === 0 ? "disabled" : ""}>上一步</button><button id="setup-next" class="primary">${setupStep === setupPages.length - 1 ? "保存并完成" : "下一步"}</button><button id="setup-exit" class="secondary">退出引导</button></div>`,
      ) + html;
  $("#content").innerHTML = html;
  if (page === "profiles") bindProfiles();
  if ($("#setup-next")) {
    $("#setup-next").onclick = () => {
      if (setupStep < setupPages.length - 1) {
        setupStep++;
        navigate(setupPages[setupStep]);
      } else {
        setupStep = -1;
        save(true);
      }
    };
    $("#setup-back").onclick = () => {
      setupStep--;
      navigate(setupPages[setupStep]);
    };
    $("#setup-exit").onclick = () => {
      setupStep = -1;
      renderEditor();
    };
  }
  document.querySelectorAll("[data-path]").forEach(
    (el) =>
      (el.onchange = () => {
        try {
          const path = JSON.parse(el.dataset.path);
          let v = el.value;
          if (v === "") {
            let target = cfg.data;
            for (const key of path.slice(0, -1)) target = target[key] ??= {};
            delete target[path.at(-1)];
          } else {
            if (el.dataset.kind === "bool") v = v === "true";
            if (el.dataset.kind === "number") {
              v = Number(v);
              if (!Number.isFinite(v)) throw Error("请输入有效数字");
            }
            if (el.dataset.kind === "array")
              v = v
                .split(/[,，]/)
                .map((x) => x.trim())
                .filter(Boolean);
            let target = cfg.data;
            for (const key of path.slice(0, -1)) target = target[key] ??= {};
            target[path.at(-1)] = v;
          }
          changed();
        } catch (e) {
          toast(e.message);
        }
      }),
  );
  document.querySelectorAll("[data-prompt]").forEach(
    (el) =>
      (el.oninput = () => {
        cfg.prompts[el.dataset.prompt] = el.value;
        changed();
      }),
  );
  document.querySelectorAll("[data-delete]").forEach(
    (el) =>
      (el.onclick = () => {
        delete cfg.data[el.dataset.group][el.dataset.delete];
        changed();
        renderEditor();
      }),
  );
  document.querySelectorAll(".add-object").forEach(
    (el) =>
      (el.onclick = () => {
        const group = el.dataset.group;
        modal(
          "添加" +
            { models: "模型", connections: "连接", credentials: "凭据引用" }[
              group
            ],
          '<div class="field" style="margin-top:20px"><label>名称（字母、数字、下划线）</label><input id="new-name"></div>',
          () => {
            const name = $("#new-name").value.trim();
            if (!/^[a-zA-Z][a-zA-Z0-9_]*$/.test(name))
              throw Error("请输入有效名称");
            if (cfg.data[group]?.[name]) throw Error("名称已存在");
            cfg.data[group] ??= {};
            cfg.data[group][name] =
              group === "credentials"
                ? { source: "managed", key: name, path: cfg.data.secrets_file }
                : group === "connections"
                  ? {
                      protocol: "openai-compatible",
                      base_url: "https://api.example.com/v1",
                      api_key_env: "MODEL_API_KEY",
                    }
                  : {
                      protocol: "openai-compatible",
                      model: "model-id",
                      api_key_env: "MODEL_API_KEY",
                      base_url: "https://api.example.com/v1",
                      timeout_seconds: 30,
                      price_input: 0,
                      price_output: 0,
                    };
            changed();
            renderEditor();
          },
        );
      }),
  );
  document.querySelectorAll(".add-field").forEach(
    (el) =>
      (el.onclick = () => {
        const group = el.dataset.group,
          defs = cfg.schema[group + ".*"] || {},
          existing = cfg.data[group][el.dataset.name];
        const names = Object.keys(defs).filter(
          (k) => !Object.hasOwn(existing, k),
        );
        modal(
          "添加参数",
          `<select id="new-field" style="margin-top:20px">${names.map((n) => `<option value="${esc(n)}">${esc(labels[n] || n)} · ${esc(n)}</option>`).join("")}</select>`,
          () => {
            const key = $("#new-field").value;
            if (!key) return;
            const type = defs[key]?.type || "";
            existing[key] = type.includes("bool")
              ? false
              : /int|float/.test(type)
                ? 0
                : "";
            changed();
            renderEditor();
          },
        );
      }),
  );
  if ($("#add-prompt"))
    $("#add-prompt").onclick = () => {
      const names = [
        "evidence",
        "reply_contract",
        "decision",
        "tool_decision",
        "search",
      ].filter((x) => !Object.hasOwn(cfg.prompts, x));
      modal(
        "添加提示词",
        `<select id="prompt-name" style="margin-top:20px">${names.map((x) => `<option value="${x}">${labels[x] || x}</option>`).join("")}</select>`,
        () => {
          const n = $("#prompt-name").value;
          if (n) {
            cfg.prompts[n] = "";
            changed();
            renderEditor();
          }
        },
      );
    };
  if ($("#reload-credentials"))
    $("#reload-credentials").onclick = async () => {
      try {
        resultToast(await api("credentials/reload", {}));
      } catch (e) {
        toast(e.message);
      }
    };
  if ($("#credential-form"))
    $("#credential-form").onsubmit = async (e) => {
      e.preventDefault();
      try {
        const r = await api("credentials", {
          name: $("#credential-name").value,
          action: "replace",
          value: $("#credential-value").value,
        });
        $("#credential-value").value = "";
        resultToast(r);
        cfg.credentials = (await api("config")).credentials;
        renderEditor();
      } catch (e) {
        toast(e.message);
      }
    };
  if ($("#versions"))
    $("#versions").onclick = async () => {
      try {
        const versions = await api("config/versions");
        modal(
          "配置版本",
          `<div class="table-wrap"><table>${versions.map((v) => `<tr><td class="revision">${esc(v.revision)}</td><td><button class="link rollback" data-revision="${esc(v.revision)}">回滚到此版本</button></td></tr>`).join("")}</table></div>`,
        );
        document.querySelectorAll(".rollback").forEach(
          (b) =>
            (b.onclick = async () => {
              try {
                resultToast(
                  await api("config/rollback", {
                    revision: b.dataset.revision,
                    expected: cfg.published_revision,
                  }),
                );
                cfg = await api("config");
                $("#dialog").close();
                renderEditor();
              } catch (e) {
                toast(e.message);
              }
            }),
        );
      } catch (e) {
        toast(e.message);
      }
    };
}
function resultToast(r) {
  toast(
    {
      applied: "配置已应用，新请求将采用新值",
      saved: "配置已保存，下次启动采用",
      pending_maintenance: "已发布，需停止引擎后重新启动",
      pending_credentials: "凭据已保存，客户端刷新失败，请检查后重新加载",
    }[r.state] || "操作完成",
  );
}
async function save(apply) {
  $("#save").disabled = $("#apply").disabled = true;
  try {
    const r = await api("config/draft", {
      data: cfg.data,
      prompts: cfg.prompts,
      expected: cfg.draft_revision,
    });
    cfg.draft_revision = r.draft_revision;
    dirty = false;
    $("#dirty").hidden = true;
    $("#save-status").textContent = "草稿已保存";
    if (!apply) {
      toast("草稿已保存");
      return;
    }
    const rows = r.plan.changes;
    modal(
      "应用配置",
      `${rows.length ? `<div class="table-wrap"><table><thead><tr><th>变更字段</th><th>原值 → 新值</th><th>生效时机</th></tr></thead><tbody>${rows.map((x) => `<tr><td>${esc(x.field)}</td><td>${esc(JSON.stringify(r.diff[x.field]?.before ?? null)?.slice(0, 300))}<br>→ ${esc(JSON.stringify(r.diff[x.field]?.after ?? null)?.slice(0, 300))}</td><td>${esc(modeNames[x.apply])}</td></tr>`).join("")}</tbody></table></div>` : empty("配置内容与当前运行版本一致")}${r.plan.maintenance ? '<div class="notice">包含维护类修改。将发布完整配置，停止引擎后重新启动时采用。</div>' : ""}`,
      async () => {
        const result = await api("config/apply", {
          expected: cfg.published_revision,
          draft_revision: cfg.draft_revision,
        });
        resultToast(result);
        cfg = await api("config");
        status = await api("status");
        $("#save-status").textContent =
          result.state === "applied" ? "运行引擎已采用" : "配置已发布";
        updateBadge();
      },
    );
  } catch (e) {
    toast(e.message);
  } finally {
    $("#save").disabled = $("#apply").disabled = false;
  }
}
$("#save").onclick = () => save(false);
$("#apply").onclick = () => save(true);
async function refresh() {
  status = await api("status");
  updateBadge();
  if (
    ["overview", "usage", "requests"].includes(page) &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)
  )
    await render();
}
setInterval(async () => {
  if ($("#shell").hidden || polling || $("#dialog").open) return;
  polling = true;
  try {
    await refresh();
  } catch (e) {
    toast(e.message);
  } finally {
    polling = false;
  }
}, 5000);
enter().catch(() => showLogin());
