"use strict";
// Each language pack uses the same field paths. Dynamic object names stay user data.
const settingsText = (() => {
  let chinese = {},
    selected = {};
  const enums = {
    decision_mode: ["lightweight", "model"],
    answer_thinking: [
      "auto",
      "complex_only",
      "default",
      "MINIMAL",
      "LOW",
      "MEDIUM",
      "HIGH",
    ],
    thinking_level: ["MINIMAL", "LOW", "MEDIUM", "HIGH"],
    gemini_thinking_level: ["MINIMAL", "LOW", "MEDIUM", "HIGH"],
    reasoning_effort: [
      "none",
      "minimal",
      "low",
      "medium",
      "high",
      "xhigh",
      "max",
    ],
    source: ["managed", "env", "file"],
    protocol: ["systemone", "google-genai", "openai-compatible", "anthropic"],
    price_currency: ["CNY", "USD"],
    search_provider: ["gemini", "deepseek"],
  };
  function key(path) {
    const parts = Array.isArray(path) ? [...path] : path.split(".");
    const flat = {
      serve_db: "serve.db_path",
      knowledge_db: "knowledge.db_path",
      shadow_db: "shadow.db_path",
      model_breaker_failures: "models.breaker_failures",
      model_breaker_cooldown_s: "models.breaker_cooldown_s",
    };
    if (parts.length === 1 && flat[parts[0]]) return flat[parts[0]];
    if (parts[0] === "credential_refs") parts[0] = "credentials";
    if (parts[0] === "group_profiles" && parts.length >= 4)
      return key(parts.slice(2));
    if (
      ["models", "connections", "credentials", "group_profiles"].includes(
        parts[0],
      ) &&
      parts.length >= 3
    )
      parts[1] = "*";
    return parts.join(".");
  }
  const t = (id, fallback = "") => selected[id] ?? chinese[id] ?? fallback;
  function label(path) {
    const id = key(path);
    if (t(id)) return t(id);
    // Runtime diffs use a flat Settings object for some static fields.
    const match = Object.keys(chinese).find(
      (k) => !k.startsWith("help.") && k.endsWith("." + id),
    );
    return t(match, t("common.other"));
  }
  async function load(
    language = document.documentElement.dataset.locale || navigator.language,
  ) {
    const base = await fetch("/assets/zh-CN.json");
    if (!base.ok) throw Error("设置语言文件加载失败");
    chinese = await base.json();
    selected = {};
    if (
      language.toLowerCase() !== "zh-cn" &&
      /^[a-z]{2,3}(?:-[a-z0-9]+)*$/i.test(language)
    ) {
      const translation = await fetch(
        "/assets/" + encodeURIComponent(language) + ".json",
      );
      if (translation.ok) selected = await translation.json();
    }
  }
  function value(path, v) {
    if (typeof v === "boolean") return t(v ? "common.on" : "common.off");
    if (v == null) return t("common.unset");
    const field = key(path).split(".").pop();
    if (enums[field]?.includes(v)) return t("option." + v, v);
    return typeof v === "string" ? v : JSON.stringify(v);
  }
  function error(message) {
    const m = /^([\w.*-]+):\s*(.*)$/.exec(message || "");
    if (!m || !t(key(m[1]))) return message;
    const reason = m[2].toLowerCase();
    const code = reason.includes("unknown connection")
      ? "connection"
      : reason.includes("required")
        ? "required"
        : reason.includes("positive")
          ? "positive"
          : reason.includes("nonnegative")
            ? "nonnegative"
            : reason.includes("one of") || reason.includes("expected model or")
              ? "choice"
              : null;
    return (
      label(m[1]) +
      " · " +
      (t("help." + key(m[1])) ||
        (code && t("error." + code)) ||
        t("common.invalid") + "（" + m[2] + "）")
    );
  }
  return { load, t, key, label, value, enums, error };
})();
