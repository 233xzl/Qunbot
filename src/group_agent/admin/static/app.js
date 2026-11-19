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
  persona: "塑造回答风格。应用后，新请求使用新的人格与提示词。",
  profiles: "按群覆盖提示词、模型用途和每日限额。",
  models: "管理模型连接、调用参数与回答角色。",
  search: "设置搜索和网页读取，以及每次请求的工具预算。",
  policy: "设置群聊访问范围、并发与回复策略。",
  budget: "调整每日消费上限、降级阈值与计费参数。",
  storage: "查看数据库占用，预览保留策略和创建数据库快照。",
  system: "管理连接凭据、配置版本与实例设置。",
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
    throw Error(settingsText.error(data.error) || "请求失败");
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
  pendingProviderKeys.clear();
  $("#dialog-content").replaceChildren();
  if ($("#dialog").open) $("#dialog").close();
  $("#content").replaceChildren();
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
function badge(state, label = stateNames[state] || state) {
  return `<span class="badge ${state === "running" || state === "succeeded" ? "green" : state === "failed" ? "orange" : ""}">${esc(label)}</span>`;
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
$("#dialog").addEventListener("close", () => {
  $("#dialog-content")
    .querySelectorAll("input[type=password]")
    .forEach((el) => (el.value = ""));
});
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
  await settingsText.load();
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
const periods = {
  today: "今天",
  yesterday: "昨天",
  "24h": "近 24 小时",
  "7d": "近 7 天",
  "14d": "近 14 天",
  "30d": "近 30 天",
};
const dimensions = { total: "合计", model: "按模型", group: "按群聊" };
const deliveries = {
  sent: "已发送",
  recorded_not_sent: "已记录 · 未发送",
  pending: "待发送",
  sending: "发送中",
  uncertain: "发送结果未知",
  failed: "发送失败",
};
let trendPeriod = "7d",
  dimension = "model",
  archivePeriod = "",
  archiveState = "",
  observedGroup = "",
  renderVersion = 0;
const archiveStates = {
  "": "全部状态",
  queued: "排队中",
  running: "运行中",
  succeeded: "已完成",
  recorded: "已记录",
  rejected: "已拒绝",
  failed: "失败",
  expired: "已超时",
  cancelled: "已取消",
};
function selectFilter(id, label, choices, selected) {
  return `<label class="monitor-filter">${label}<select id="${id}">${Object.entries(
    choices,
  )
    .map(
      ([v, text]) =>
        `<option value="${esc(v)}" ${selected === v ? "selected" : ""}>${esc(text)}</option>`,
    )
    .join("")}</select></label>`;
}
function costText(value) {
  if (value == null) return "费用未知";
  if (!value) return "¥0";
  return (
    "¥" +
    Number(value).toLocaleString(
      "zh-CN",
      value < 0.01
        ? { maximumSignificantDigits: 3 }
        : { minimumFractionDigits: 2, maximumFractionDigits: 4 },
    )
  );
}
function requestCost(r) {
  return (
    costText(r.cost) +
    (["running", "queued"].includes(r.state) ? " · 截至当前" : "")
  );
}
const palette = [
  "#3975df",
  "#0c967e",
  "#df9842",
  "#8a64d2",
  "#df6e86",
  "#20a4bd",
  "#698153",
];
function trend(u) {
  if (!u.series.length) return empty("所选时段暂无调用记录");
  const max = Math.max(...u.days.map((d) => d.cost)) || 1;
  const bars = u.days
    .map((d, i) => {
      const segments = u.series
        .map((s, j) =>
          s.values[i] > 0
            ? `<i style="height:${(100 * s.values[i]) / max}%;background:${palette[j % palette.length]}" title="${esc(d.day)} · ${esc(s.label)} · ${costText(s.values[i])}"></i>`
            : "",
        )
        .join("");
      const tick =
        i % Math.ceil(u.days.length / 6) === 0 ? d.day.slice(-5) : "";
      return `<div class="bucket" title="${esc(d.day)} · ${costText(d.cost)}">${segments}<small>${esc(tick)}</small></div>`;
    })
    .join("");
  return `<div class="trend-layout"><div class="trend-plot"><small class="muted">${u.cost ? costText(max) : "¥0 · 零费用调用"}</small><div class="buckets">${bars}</div></div><div class="table-wrap series-table"><table><thead><tr><th>${{ model: "模型 / 用途", group: "群聊", total: "范围" }[dimension]}</th><th>调用</th><th>Token</th><th>消费</th></tr></thead><tbody>${u.series.map((s, i) => `<tr><td><i class="swatch" style="background:${palette[i % palette.length]}"></i>${esc(s.label)}</td><td>${num(s.calls)}</td><td>${num(s.tokens)}</td><td>${costText(s.cost)}</td></tr>`).join("")}</tbody></table></div></div>`;
}
const segmenter = new Intl.Segmenter("zh-CN", { granularity: "grapheme" });
function preview(text) {
  const chars = Array.from(segmenter.segment(text), (x) => x.segment);
  return chars.slice(0, 5).join("") + (chars.length > 5 ? "…" : "");
}
function requestTable(rows) {
  return rows.length
    ? `<div class="table-wrap"><table><thead><tr><th>请求消息</th><th>群聊</th><th>时间</th><th>状态</th><th>耗时</th><th>当次费用</th></tr></thead><tbody>${rows.map((r) => `<tr><td><button class="link request-detail" data-id="${esc(r.run_id)}">${esc(preview(r.preview))}</button></td><td>${esc(r.group_name)}</td><td>${time(r.created_at)}</td><td>${badge(r.state, archiveStates[r.state])}</td><td>${r.ended_at ? Math.max(0, r.ended_at - r.created_at).toFixed(1) + " 秒" : "—"}</td><td>${requestCost(r)}</td></tr>`).join("")}</tbody></table></div>`
    : empty("暂无符合条件的请求");
}
function bindRequests() {
  document.querySelectorAll(".request-detail").forEach(
    (b) =>
      (b.onclick = async () => {
        try {
          const d = await api("requests/" + encodeURIComponent(b.dataset.id));
          const reply =
            d.reply_text ||
            {
              rejected: "已拒绝 · 无回复",
              running: "正在生成回复",
              queued: "排队中",
            }[d.state] ||
            "无回复记录";
          let diagnostic = "";
          if (page === "requests") {
            let detail = {};
            try {
              detail = JSON.parse(d.detail) || {};
            } catch {}
            diagnostic = `<details class="diagnostic"><summary>运行诊断</summary><p>配置版本 ${esc(detail.config_revision || "未记录")} · 路线 ${esc(d.route || "—")} · 预算来源 ${esc({ group: "本群", global: "全局" }[detail.budget_scope] || "—")}</p><p>群配置 ${esc(detail.group_overrides?.join("、") || "继承全局")} · 错误 ${esc(d.error_kind || "无")}</p>${d.usage.length ? d.usage.map((x) => `<p>${esc(x.model)} · ${esc(x.task || x.service || "调用")} · ${costText(x.cost_cny)}</p>`).join("") : ""}</details>`;
          }
          modal(
            "请求详情",
            `<p class="muted">${esc(d.group_name)} · ${time(d.created_at)} · ${esc(archiveStates[d.state])} · ${requestCost(d)}</p><article class="bubble"><strong>${esc(d.member_name || "未知成员")}</strong><p>${esc(d.message)}</p></article><article class="bubble answer"><strong>Bot${d.reply_kind === "busy_notice" ? " · 忙碌提示" : ""}</strong><p>${esc(reply)}</p><small>${esc(deliveries[d.delivery] || "")}</small></article>${diagnostic}`,
          );
        } catch (e) {
          toast(e.message);
        }
      }),
  );
}
function rerender() {
  render().catch((e) => toast(e.message));
}
async function render() {
  const version = ++renderVersion;
  updateBadge();
  const content = $("#content");
  if (page === "overview") {
    const [u, rows] = await Promise.all([
      api(`usage?period=${trendPeriod}&dimension=${dimension}`),
      api("requests?limit=5"),
    ]);
    if (version !== renderVersion) return;
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
      `<div class="monitor-toolbar">${selectFilter("trend-period", "时间范围", periods, trendPeriod)}<span class="muted">北京时间 · ${["today", "yesterday", "24h"].includes(trendPeriod) ? "按小时" : "按日"}</span></div>` +
      `<div class="monitor-cards">` +
      cards([
        ["请求次数", num(u.requests), "次", periods[trendPeriod]],
        ["Token 用量", num(u.tokens), "", "输入 + 输出"],
        [
          "区间消费",
          costText(u.cost),
          "",
          "今日 " +
            costText(u.today_cost) +
            " / 普通用户预算 ¥" +
            money(
              status.usage?.daily_cap_cny ?? cfg.data.budgets?.daily_cost_cny,
            ),
        ],
        [
          "平均耗时",
          u.avg_seconds == null ? "—" : Number(u.avg_seconds).toFixed(1),
          "秒",
          "已结束请求",
        ],
      ]) +
      "</div>" +
      panel(
        "消费趋势",
        trend(u),
        selectFilter("trend-dimension", "展示", dimensions, dimension),
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
    $("#trend-period").onchange = (e) => {
      trendPeriod = e.target.value;
      rerender();
    };
    $("#trend-dimension").onchange = (e) => {
      dimension = e.target.value;
      rerender();
    };
    bindControls();
    bindRequests();
    if ($("#setup-start"))
      $("#setup-start").onclick = () => {
        setupStep = 0;
        navigate(setupPages[0]);
      };
    $("#all-requests").onclick = () => {
      offset = 0;
      archivePeriod = archiveState = observedGroup = "";
      navigate("requests");
    };
    if ($("#demo-form"))
      $("#demo-form").onsubmit = async (e) => {
        e.preventDefault();
        try {
          await api("demo/request", {
            text: $("#demo-text").value,
            group_id: $("#demo-group").value,
          });
          toast("模拟请求已提交");
        } catch (e) {
          toast(e.message);
        }
      };
    return;
  }
  if (page === "requests") {
    const query = new URLSearchParams({ offset, limit: 51 });
    if (archivePeriod) query.set("period", archivePeriod);
    if (archiveState) query.set("state", archiveState);
    if (observedGroup) query.set("group_id", observedGroup);
    const [rows, groups] = await Promise.all([
      api("requests?" + query),
      api("groups?named=true"),
    ]);
    if (version !== renderVersion) return;
    const groupOptions = Object.fromEntries(groups.map((g) => [g.id, g.name]));
    const filters =
      selectFilter(
        "archive-period",
        "时间",
        { "": "全部时间", ...periods },
        archivePeriod,
      ) +
      selectFilter(
        "archive-group",
        "群聊",
        { "": "全部群聊", ...groupOptions },
        observedGroup,
      ) +
      selectFilter("archive-state", "状态", archiveStates, archiveState);
    content.innerHTML =
      `<div class="monitor-toolbar filters">${filters}</div>` +
      panel(
        "请求记录",
        requestTable(rows.slice(0, 50)),
        `<div class="toolbar"><button id="prev" class="secondary" ${offset ? "" : "disabled"}>上一页</button><span class="page-count">第 ${1 + offset / 50} 页</span><button id="next" class="secondary" ${rows.length > 50 ? "" : "disabled"}>下一页</button></div>`,
      );
    function resetPage() {
      offset = 0;
      rerender();
    }
    $("#archive-group").onchange = (e) => {
      observedGroup = e.target.value;
      resetPage();
    };
    $("#archive-period").onchange = (e) => {
      archivePeriod = e.target.value;
      resetPage();
    };
    $("#archive-state").onchange = (e) => {
      archiveState = e.target.value;
      resetPage();
    };
    $("#prev").onclick = () => {
      offset = Math.max(0, offset - 50);
      rerender();
    };
    $("#next").onclick = () => {
      offset += 50;
      rerender();
    };
    bindRequests();
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
  if (page === "profiles") {
    const rows = await api("received-groups");
    if (version !== renderVersion) return;
    groupChoices = rows;
  }
  renderEditor();
}
function field(path, value, definition = {}) {
  if (path.join(".") === "web.api_key_env")
    return keyField("web", "search", value);
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
    input = `<select data-path="${data}" data-kind="bool" id="${id}"><option value="true" ${value ? "selected" : ""}>${esc(settingsText.t("common.on"))}</option><option value="false" ${!value ? "selected" : ""}>${esc(settingsText.t("common.off"))}</option></select>`;
  else if (settingsText.enums[key]) {
    const options = [
      ...new Set(["", ...settingsText.enums[key], ...(value ? [value] : [])]),
    ];
    input = `<select data-path="${data}" id="${id}">${options.map((v) => `<option value="${esc(v)}" ${v === (value ?? "") ? "selected" : ""}>${esc(v ? settingsText.value(path, v) : settingsText.t("common.empty"))}</option>`).join("")}</select>`;
  } else
    input = `<input data-path="${data}" data-kind="${isArray ? "array" : isNum ? "number" : "text"}" id="${id}" ${isNum ? 'type="number" step="any"' : ""} value="${esc(isArray ? (value || []).join(", ") : value)}" placeholder="${isArray ? "多个值用逗号分隔" : "未设置"}">`;
  return `<div class="field"><label for="${id}">${esc(settingsText.label(path))}</label>${input}<small>${esc([settingsText.t("help." + settingsText.key(path)), settingsText.t("apply." + definition.apply)].filter(Boolean).join(" · "))}${Object.hasOwn(cfg.overrides, path.join(".")) ? " · " + esc(settingsText.t("common.override")) : ""}</small></div>`;
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
let groupChoices = [],
  selectedGroup = "";
function groupLabel(row) {
  const name = row.name || "群名待同步";
  if (!row.name || groupChoices.filter((g) => g.name === row.name).length > 1) {
    let size = 4;
    while (
      size < row.group_id.length &&
      groupChoices.some(
        (g) =>
          g.group_id !== row.group_id &&
          g.name === row.name &&
          g.group_id.slice(-size) === row.group_id.slice(-size),
      )
    )
      size++;
    return `${name} · ${row.group_id.slice(-size)}`;
  }
  return name;
}
function profileEditor() {
  const profiles = cfg.data.group_profiles || {};
  const roles = cfg.data.model_roles || {};
  const aliases = Object.entries(cfg.data.models || {})
    .filter(([, m]) => m && typeof m === "object" && m.protocol !== "systemone")
    .map(([id]) => id);
  if (!groupChoices.length)
    return empty("机器人收到群消息后，可在这里选择群聊。");
  const row =
    groupChoices.find((g) => g.group_id === selectedGroup) || groupChoices[0];
  selectedGroup = row.group_id;
  const gid = selectedGroup,
    profile = profiles[gid] || {};
  const options = groupChoices
    .map(
      (g) =>
        `<option value="${esc(g.group_id)}" ${g.group_id === gid ? "selected" : ""}>${esc(groupLabel(g))}</option>`,
    )
    .join("");
  let html = `<label class="monitor-filter">选择群聊<select id="profile-id">${options}</select></label>`;
  const running = status.state === "running";
  const confirmed = row.membership === "present";
  const state = !confirmed
    ? "群成员资格待确认"
    : !running
      ? `全局${stateNames[status.state] || "已停止"}`
      : row.enabled
        ? "已开启"
        : "已停止";
  html += panel(
    settingsText.t("group.reply"),
    `<span>${esc(state)}</span>`,
    `<button id="group-switch" class="secondary" ${(!running && !(status.state === "paused" && row.enabled)) || (!row.enabled && !confirmed) ? "disabled" : ""}>${row.enabled ? "停止本群回复" : "启动本群回复"}</button>`,
  );
  if (!confirmed) return html;
  {
    const enabled = profile.enabled === true;
    const scoped = (section, key) =>
      `data-profile="${esc(gid)}" data-section="${section}" data-key="${key}"`;
    const fields = [];
    for (const key of ["answer", "decision", "fallback"]) {
      const value = profile.model_roles?.[key];
      const inherited =
        roles[key] ||
        (key === "fallback"
          ? "deepseek"
          : key === "decision"
            ? roles.answer || "gemini"
            : "gemini");
      fields.push(
        `<label class="field">${esc(settingsText.label(["model_roles", key]))}<select ${scoped("model_roles", key)}><option value="">继承全局 · ${esc(inherited)}</option>${aliases.map((id) => `<option ${value === id ? "selected" : ""} value="${esc(id)}">本群 · ${esc(id)}</option>`).join("")}</select></label>`,
      );
    }
    for (const key of Object.keys(cfg.schema.prompts)) {
      const value = profile.prompts?.[key] || "";
      const inherited = cfg.prompts[key] || cfg.prompt_defaults[key] || "";
      fields.push(
        `<label class="field">${esc(settingsText.label(["prompts", key]))} · ${value ? "本群覆盖" : "继承全局"}<textarea ${scoped("prompts", key)} placeholder="${esc(inherited)}">${esc(value)}</textarea></label>`,
      );
    }
    fields.push(
      `<label class="field">${esc(settingsText.label(["group_profiles", gid, "daily_cost_cny"]))}<input type="number" step="any" min="0" ${scoped("", "daily_cost_cny")} value="${esc(profile.daily_cost_cny)}" placeholder="仅使用全局预算"></label>`,
    );
    html += panel(
      groupLabel(row),
      `<label><input type="checkbox" ${scoped("", "enabled")} ${enabled ? "checked" : ""}> ${esc(settingsText.t("group_profiles.*.enabled"))}</label>${enabled ? '<p class="muted">清空字段即可恢复继承。</p>' + fields.join("") : ""}`,
      profiles[gid]
        ? `<button class="delete" data-group="group_profiles" data-delete="${esc(gid)}">清除群覆盖设置</button>`
        : "",
    );
  }
  return html;
}
function bindProfiles() {
  if (!$("#profile-id")) return;
  $("#profile-id").onchange = (e) => {
    selectedGroup = e.target.value;
    renderEditor();
  };
  $("#group-switch").onclick = async (e) => {
    const row = groupChoices.find((g) => g.group_id === selectedGroup);
    e.currentTarget.disabled = true;
    try {
      await api(`received-groups/${encodeURIComponent(row.group_id)}/reply`, {
        enabled: !row.enabled,
      });
      toast(row.enabled ? "已停止本群回复" : "已启动本群回复");
    } catch (err) {
      toast(err.message);
    }
    if (page === "profiles") await navigate("profiles");
  };
  document.querySelectorAll("[data-profile]").forEach(
    (el) =>
      (el.onchange = () => {
        const { profile: gid, section, key } = el.dataset;
        const profile = ((cfg.data.group_profiles ??= {})[gid] ??= {});
        const target = section ? (profile[section] ??= {}) : profile;
        if (el.type === "checkbox") target[key] = el.checked;
        else if (!el.value.trim()) delete target[key];
        else if (el.type === "number") target[key] = Number(el.value);
        else target[key] = el.value;
        changed();
        if (key === "enabled") renderEditor();
        else if (section === "prompts")
          el.parentNode.firstChild.textContent = `${settingsText.label(["prompts", key])} · ${target[key] ? "本群覆盖" : "继承全局"}`;
      }),
  );
}
function renderEditor() {
  let html = "";
  if (page === "profiles") html = profileEditor();
  if (page === "persona") {
    html = Object.entries({ persona: cfg.default_persona, ...cfg.prompts })
      .map(([name, text]) =>
        panel(
          settingsText.label(["prompts", name]),
          `<textarea class="prompt" data-prompt="${esc(name)}" aria-label="${esc(settingsText.label(["prompts", name]))}">${esc(text)}</textarea>`,
          `<span class="badge">新请求生效</span>`,
        ),
      )
      .join("");
    html += '<button id="add-prompt" class="secondary">＋ 添加提示词</button>';
  }
  if (page === "models") html = providerPage();
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
      `<details><summary>高级凭据引用设置</summary>${objects("credentials", "凭据引用")}</details>` +
      panel(
        "更新管理型凭据",
        `<p class="muted">选择要更新的凭据并填写新值。</p><form id="credential-form"><div class="grid2" style="margin-top:18px"><div class="field"><label>凭据用途</label><select id="credential-name" required>${managedChoices()}</select></div><div class="field"><label>新值</label><input id="credential-value" type="password" required autocomplete="new-password"></div></div><button class="secondary">更新凭据并刷新客户端</button></form>`,
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
  if (page === "models") bindProviders();
  document.querySelectorAll("[data-provider-key]").forEach(
    (el) =>
      (el.oninput = () => {
        stageKey(...JSON.parse(el.dataset.providerKey), el.value);
        changed();
      }),
  );
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
            cfg.data[group][name] = {
              source: "managed",
              key: name,
              path: cfg.data.secrets_file,
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
          `<select id="new-field" style="margin-top:20px">${names.map((n) => `<option value="${esc(n)}">${esc(settingsText.label([group, "*", n]))}</option>`).join("")}</select>`,
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
        `<select id="prompt-name" style="margin-top:20px">${names.map((x) => `<option value="${x}">${esc(settingsText.label(["prompts", x]))}</option>`).join("")}</select>`,
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
function prettyChange(path) {
  const [group, name] = path.split(".");
  const owner =
    group === "group_profiles"
      ? groupChoices.find((r) => r.group_id === name)?.name || "本群"
      : ["models", "connections", "credentials", "credential_refs"].includes(
            group,
          )
        ? name
        : "";
  return [owner, settingsText.label(path)].filter(Boolean).join(" · ");
}
async function save(apply) {
  $("#save").disabled = $("#apply").disabled = true;
  try {
    const r = await api("config/draft", {
      provider_keys: Array.from(pendingProviderKeys.values()),
      data: cfg.data,
      prompts: cfg.prompts,
      expected: cfg.draft_revision,
    });
    pendingProviderKeys.clear();
    cfg = await api("config");
    renderEditor();
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
      `${rows.length ? `<div class="table-wrap"><table><thead><tr><th>变更字段</th><th>原值 → 新值</th><th>生效时机</th></tr></thead><tbody>${rows.map((x) => `<tr><td>${esc(prettyChange(x.field))}</td><td>${esc(settingsText.value(x.field, r.diff[x.field]?.before).slice(0, 300))}<br>→ ${esc(settingsText.value(x.field, r.diff[x.field]?.after).slice(0, 300))}</td><td>${esc(settingsText.t("apply." + x.apply))}</td></tr>`).join("")}</tbody></table></div>` : empty("配置内容与当前运行版本一致")}${r.plan.maintenance ? '<div class="notice">包含维护类修改。将发布完整配置，停止引擎后重新启动时采用。</div>' : ""}`,
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
  const previous = status.state;
  status = await api("status");
  updateBadge();
  if (
    page === "profiles" &&
    !["INPUT", "TEXTAREA", "SELECT"].includes(document.activeElement.tagName)
  ) {
    const rows = await api("received-groups");
    if (
      page === "profiles" &&
      (previous !== status.state ||
        JSON.stringify(rows) !== JSON.stringify(groupChoices))
    ) {
      groupChoices = rows;
      renderEditor();
    }
  }
  if (
    ["overview", "requests"].includes(page) &&
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
