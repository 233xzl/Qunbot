"use strict";
const providerProtocols = [
  "openai-responses",
  "openai-compatible",
  "anthropic",
  "google-genai",
];
const providerPresets = {
  OpenAI: ["openai-responses", "https://api.openai.com/v1"],
  DeepSeek: ["openai-compatible", "https://api.deepseek.com"],
  Anthropic: ["anthropic", "https://api.anthropic.com"],
  Gemini: ["google-genai", "https://generativelanguage.googleapis.com"],
  Vertex: ["google-genai", ""],
};
const pendingProviderKeys = new Map();
const providerText = (key) => settingsText.t("provider." + key);
function keyScope(group, name) {
  return JSON.stringify([group, name]);
}
function keyField(group, name, ref) {
  const scope = keyScope(group, name);
  const message = pendingProviderKeys.has(scope)
    ? providerText("pending")
    : cfg.credentials[ref]?.configured
      ? providerText("keep")
      : providerText("enter");
  return `<label class="field">API 密钥<input type="password" data-provider-key="${esc(scope)}" autocomplete="new-password" placeholder="${esc(message)}"></label>`;
}
function stageKey(group, name, value) {
  const scope = keyScope(group, name);
  if (value) pendingProviderKeys.set(scope, { group, name, value });
  else pendingProviderKeys.delete(scope);
}
function modelRoles() {
  const roles = (cfg.data.model_roles ||= {}),
    names = Object.keys(cfg.data.models || {}).filter(
      (n) => typeof cfg.data.models[n] === "object",
    );
  return panel(
    "模型用途",
    `<div class="grid2">${["answer", "decision", "fallback"]
      .map((r) => {
        const v = roles[r] || "";
        return `<label class="field">${esc(settingsText.label(["model_roles", r]))}<select data-path="${esc(JSON.stringify(["model_roles", r]))}"><option value="">使用默认设置</option>${names.map((n) => `<option value="${esc(n)}" ${n === v ? "selected" : ""}>${esc(n)}</option>`).join("")}</select></label>`;
      })
      .join("")}</div>`,
  );
}
function providerPage() {
  const connections = Object.entries(cfg.data.connections || {}),
    models = Object.entries(cfg.data.models || {}).filter(
      ([, v]) => v && typeof v === "object",
    );
  const row = (group, n, o) => {
    const parent =
        group === "models" && o.connection
          ? cfg.data.connections?.[o.connection]
          : null,
      protocol = parent?.protocol || o.protocol;
    const meta =
      group === "connections"
        ? `${models.filter(([, m]) => m.connection === n).length} 个模型`
        : o.model;
    return `<div class="provider-row"><div><strong>${esc(n)}</strong><small>${esc(settingsText.value(["models", "*", "protocol"], protocol))} · ${esc(meta || "")} · ${esc(pendingProviderKeys.has(keyScope(parent ? "connections" : group, parent ? o.connection : n)) ? providerText("pending") : cfg.credentials[parent?.api_key_env || o.api_key_env]?.configured ? providerText("configured") : providerText("missing"))}</small></div><div class="toolbar">${group === "connections" ? `<button class="secondary" data-create-model="${esc(n)}">添加模型</button>` : ""}<button class="secondary" data-edit-owner="${group}" data-name="${esc(n)}">编辑</button><button class="link" data-remove-owner="${group}" data-name="${esc(n)}">移除</button></div></div>`;
  };
  return (
    (cfg.retired_models?.length ? `<p class="notice">已停用旧版判断模型 ${cfg.retired_models.map(esc).join("、")}。下次发布时将清理配置。</p>` : "") +
    panel(
      "供应商连接",
      connections.map(([n, o]) => row("connections", n, o)).join("") ||
        empty("尚未添加供应商连接"),
      '<button id="new-provider" class="secondary">添加供应商</button>',
    ) +
    panel(
      "模型",
      models.map(([n, o]) => row("models", n, o)).join("") ||
        empty("添加供应商后即可添加模型"),
    ) +
    modelRoles() +
    section("models", "熔断策略", ["breaker_failures", "breaker_cooldown_s"])
  );
}
function refsTo(group, n) {
  const data = cfg.data,
    refs = [];
  if (group === "connections")
    return Object.entries(data.models || {})
      .filter(([, m]) => m.connection === n)
      .map(([k]) => k);
  const roles = (obj, prefix) =>
    Object.entries(obj || {}).forEach(([role, id]) => {
      if (n === id)
        refs.push(prefix + settingsText.label(["model_roles", role]));
    });
  roles(data.model_roles, "全局 · ");
  for (const [gid, g] of Object.entries(data.group_profiles || {}))
    roles(g.model_roles, `群 ${gid} · `);
  for (const k of [
    "gemini_credentials_model",
    "search",
    "fetch",
    "search_fallback",
    "fetch_fallback",
  ])
    if (
      data.web?.[k] === n &&
      !(k !== "gemini_credentials_model" && n === "deepseek")
    )
      refs.push(settingsText.label(["web", k]));
  return refs;
}
function providerDialog(group, existing, connection = "") {
  const isProvider = group === "connections",
    original = cfg.data[group]?.[existing] || {},
    draft = { ...original };
  const inherited = !isProvider && (original.connection || connection),
    standalone = !inherited;
  let protocol = original.protocol || "openai-responses";
  const select = (id, items, chosen) =>
    `<select id="${id}">${items.map(([v, t]) => `<option value="${esc(v)}" ${v === chosen ? "selected" : ""}>${esc(t)}</option>`).join("")}</select>`;
  const input = (id, value = "", type = "text", hint = "") =>
    `<input id="${id}" type="${type}" value="${esc(value ?? "")}" placeholder="${esc(hint)}" ${type === "number" ? 'step="any" min="0"' : ""}>`;
  const label = (text, html) =>
    `<label class="field">${esc(text)}${html}</label>`;
  const modelPath = (k) => settingsText.label(["models", "*", k]);
  const advanced = Object.keys(cfg.schema["models.*"] || {}).filter(
    (k) =>
      ![
        "connection",
        "protocol",
        "model",
        "base_url",
        "api_key_env",
        "base_url_env",
        "vertex",
        "api_version",
        "project_env",
        "location",
        "price_input",
        "price_output",
      ].includes(k),
  );
  const advKeys = isProvider
    ? ["channel"]
    : advanced.filter(
        (k) => !inherited || !Object.hasOwn(cfg.schema["connections.*"], k),
      );
  const detailField = (k) => {
    const def = cfg.schema["models.*"][k],
      choices = settingsText.enums[k];
    return label(
      modelPath(k),
      choices
        ? select(
            "opt-" + k,
            [
              ["", "使用默认设置"],
              ...choices.map((v) => [
                v,
                settingsText.value(["models", "*", k], v),
              ]),
            ],
            draft[k],
          )
        : input(
            "opt-" + k,
            draft[k],
            /int|float/.test(def.type) ? "number" : "text",
          ),
    );
  };
  const vertex = original.vertex === true;
  const html =
    (isProvider && !existing
      ? label(
          "供应商预设",
          select(
            "p-preset",
            [
              ["custom", "自定义 API"],
              ...Object.keys(providerPresets).map((k) => [k, k]),
            ],
            "custom",
          ),
        )
      : "") +
    label(
      isProvider ? "供应商名称" : "模型别名",
      existing ? `<strong>${esc(existing)}</strong>` : input("p-name"),
    ) +
    (standalone
      ? label(
          "协议",
          select(
            "p-protocol",
            providerProtocols.map((v) => [
              v,
              settingsText.value(["models", "*", "protocol"], v),
            ]),
            protocol,
          ),
        ) +
        `<div id="p-base-box">${label("服务地址", input("p-url", draft.base_url, "url", draft.base_url_env ? "留空沿用当前地址来源" : "https://…"))}</div>` +
        `<div id="p-google">${label("Google Cloud Vertex", `<input type="checkbox" id="p-vertex" ${vertex ? "checked" : ""}>`)}${label("API 版本", input("p-version", draft.api_version))}<div id="p-cloud">${label("区域", input("p-location", draft.location || "global"))}${label("项目编号的环境变量", input("p-project", draft.project_env || "GOOGLE_CLOUD_PROJECT"))}</div></div>` +
        label(
          "API 密钥",
          `<input id="p-secret" type="password" autocomplete="new-password" placeholder="${esc(pendingProviderKeys.has(keyScope(group, existing)) ? providerText("pending") : cfg.credentials[draft.api_key_env]?.configured ? providerText("keep") : providerText("enter"))}">`,
        )
      : label("所属供应商", `<strong>${esc(inherited)}</strong>`)) +
    (!isProvider
      ? label("模型 ID", input("p-model", draft.model)) +
        label(
          modelPath("price_input"),
          input("p-in", draft.price_input ?? 0, "number"),
        ) +
        label(
          modelPath("price_output"),
          input("p-out", draft.price_output ?? 0, "number"),
        )
      : "") +
    `<details><summary>更多参数</summary><div class="grid2">${advKeys.map(detailField).join("")}</div></details>`;
  modal(
    existing
      ? "编辑" + (isProvider ? "供应商" : "模型")
      : "添加" + (isProvider ? "供应商" : "模型"),
    html,
    () => {
      const n = existing || $("#p-name").value.trim();
      if (
        !existing &&
        (!/^[A-Za-z][A-Za-z0-9_-]{0,63}$/.test(n) ||
          n in (cfg.data[group] || {}))
      )
        throw Error("请输入未使用的名称，以字母开头");
      if (inherited) draft.connection = inherited;
      if (standalone) {
        draft.protocol = $("#p-protocol").value;
        const google = draft.protocol === "google-genai";
        const url = $("#p-url").value.trim();
        if (google && $("#p-vertex").checked) {
          draft.vertex = true;
          draft.location = $("#p-location").value.trim() || "global";
          draft.project_env =
            $("#p-project").value.trim() || "GOOGLE_CLOUD_PROJECT";
          delete draft.base_url;
          delete draft.base_url_env;
        } else {
          delete draft.vertex;
          delete draft.location;
          delete draft.project_env;
          if (url) {
            draft.base_url = url;
            delete draft.base_url_env;
          } else if (!draft.base_url_env) throw Error("请填写服务地址");
          else delete draft.base_url;
        }
        if (google && $("#p-version").value.trim())
          draft.api_version = $("#p-version").value.trim();
        else delete draft.api_version;
        if (
          !draft.api_key_env &&
          !$("#p-secret").value &&
          !pendingProviderKeys.has(keyScope(group, n))
        )
          throw Error("请填写 API 密钥");
      }
      if (!isProvider) {
        draft.model = $("#p-model").value.trim();
        if (!draft.model) throw Error("请填写模型 ID");
        draft.price_input = Number($("#p-in").value);
        draft.price_output = Number($("#p-out").value);
      }
      for (const k of advKeys) {
        const el = $("#opt-" + k);
        if (!el.value) delete draft[k];
        else draft[k] = el.type === "number" ? Number(el.value) : el.value;
      }
      if (standalone && $("#p-secret").value) {
        stageKey(group, n, $("#p-secret").value);
        $("#p-secret").value = "";
      }
      (cfg.data[group] ??= {})[n] = draft;
      changed();
      renderEditor();
    },
  );
  if (standalone) {
    const sync = () => {
      const google = $("#p-protocol").value === "google-genai";
      $("#p-google").hidden = !google;
      $("#p-cloud").hidden = !$("#p-vertex").checked;
      $("#p-base-box").hidden = google && $("#p-vertex").checked;
    };
    $("#p-vertex").onchange = sync;
    $("#p-protocol").onchange = sync;
    sync();
    if ($("#p-preset"))
      $("#p-preset").onchange = (e) => {
        const k = e.target.value;
        const [p, url] = providerPresets[k] || ["openai-responses", ""];
        $("#p-protocol").value = p;
        $("#p-url").value = url;
        $("#p-vertex").checked = k === "Vertex";
        $("#p-version").value =
          k === "Vertex" ? "v1" : p === "google-genai" ? "v1beta" : "";
        if (!$("#p-name").value)
          $("#p-name").value = k === "custom" ? "" : k.toLowerCase();
        sync();
      };
  }
}
function bindProviders() {
  $("#new-provider").onclick = () => providerDialog("connections");
  document
    .querySelectorAll("[data-edit-owner]")
    .forEach(
      (el) =>
        (el.onclick = () =>
          providerDialog(el.dataset.editOwner, el.dataset.name)),
    );
  document
    .querySelectorAll("[data-create-model]")
    .forEach(
      (el) =>
        (el.onclick = () =>
          providerDialog("models", null, el.dataset.createModel)),
    );
  document.querySelectorAll("[data-remove-owner]").forEach(
    (el) =>
      (el.onclick = () => {
        const { removeOwner: g, name: n } = el.dataset,
          refs = refsTo(g, n);
        if (refs.length) return toast("请先换绑：" + refs.join("、"));
        delete cfg.data[g][n];
        pendingProviderKeys.delete(keyScope(g, n));
        changed();
        renderEditor();
      }),
  );
}

function managedChoices() {
  const labels = new Map(
    Object.entries({
      SNOWLUMA_WS_URL: "QQ 网关地址",
      SNOWLUMA_WS_TOKEN: "QQ 网关令牌",
    }),
  );
  for (const group of ["connections", "models"])
    for (const [name, obj] of Object.entries(cfg.data[group] || {})) {
      for (const [field, label] of [
        ["api_key_env", "API 密钥"],
        ["base_url_env", "服务地址"],
        ["project_env", "云项目编号"],
      ])
        if (obj[field]) labels.set(obj[field], `${name} · ${label}`);
    }
  if (cfg.data.web?.api_key_env)
    labels.set(cfg.data.web.api_key_env, "搜索服务 · API 密钥");
  return Object.entries(cfg.credentials)
    .filter(([, c]) => c.source === "managed")
    .map(
      ([n, c]) =>
        `<option value="${esc(n)}">${esc(labels.get(n) || n)} · ${c.configured ? "已配置" : "待填写"}</option>`,
    )
    .join("");
}
