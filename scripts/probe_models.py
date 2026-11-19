"""Live capability probes for Gemini (Google GenAI) and DeepSeek (Anthropic) via the relay.

Usage: python scripts/probe_models.py [--model gemini deepseek] [--via pai sdk]
                                      [--case text image ...] [--budget-cny 0.3]

Each case is run once through Pydantic AI ("pai") and once through the official SDK ("sdk").
struct_tool is each path's default forced-call shape; struct_tool_swap sends the other path's
shape (pai: named tool pinned with tool_choice=ToolOrOutput([]); sdk: generic unpinned tool).
Keys and the relay URL are read from the secrets file named in the settings and are never
printed; exception text is redacted before it is shown or written. Records go to the
git-ignored data/probes/ directory. Findings: docs/MODEL-ACCESS.md.
"""

import argparse
import asyncio
import base64
import json
import re
import struct
import sys
import time
import warnings
import zlib
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from group_agent.config import ModelSpec, load_model_credentials, load_settings  # noqa: E402

warnings.filterwarnings("ignore", category=SyntaxWarning)

CASES = (
    "text",
    "image",
    "tool",
    "struct_tool",
    "struct_tool_swap",
    "struct_native",
    "struct_prompt",
    "thinking",
    "stream",
)
# Probe cap, deliberately below production max_output_tokens to bound cost. Gemini thinking
# tokens count against it, so long-thinking runs can end with MAX_TOKENS.
MAX_TOKENS = 2048


# Extraction target. Deliberately no docstring: Pydantic AI appends a model docstring to the
# output-tool description, and a technical docstring there made Gemini (thinking HIGH) think
# ~2,000 tokens and return empty/absent function calls (see docs/MODEL-ACCESS.md).
# extra="forbid" emits additionalProperties=false for strict schemas.
class Person(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    age: int
    hobbies: list[str]


EXTRACT_PROMPT = "从这句话提取人物信息：小林今年 12 岁，喜欢足球和围棋。"
RECORD_DESCRIPTION = "记录提取到的人物信息"
PROMPTED_SUFFIX = (
    "\n只输出一个 JSON 对象，不要 Markdown 代码块，格式符合以下 JSON Schema：\n"
    + json.dumps(Person.model_json_schema(), ensure_ascii=False)
)
TEXT_PROMPT = "用一句话回答：一年有几个季节？"
THINK_PROMPT = "一个数加上它的一半等于 36，这个数是多少？只给结果。"
IMAGE_PROMPT = "这张图片的背景是什么颜色？上面写着什么数字？用一句话回答。"
TOOL_PROMPT = "杭州现在天气怎么样？请先调用工具查询，再用一句话回答。"
WEATHER_TOOL = {
    "name": "get_weather",
    "description": "查询城市当前天气",
    "schema": {
        "type": "object",
        "properties": {"city": {"type": "string", "description": "城市名"}},
        "required": ["city"],
    },
}


def weather_result(city: str) -> dict:
    return {"city": city, "condition": "小雨", "temp_c": 17}


def check_person(raw) -> tuple[bool | str, str]:
    """Validate with Pydantic, then compare against the expected extraction."""
    try:
        if isinstance(raw, str):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip())
            person = Person.model_validate_json(text)
        elif isinstance(raw, Person):
            person = raw
        else:
            person = Person.model_validate(raw)
    except ValidationError as error:
        return False, f"validation failed: {error.error_count()} errors"
    good = "小林" in person.name and person.age == 12 and set(person.hobbies) >= {"足球", "围棋"}
    return (True if good else "partial"), person.model_dump_json()


def check_image(text: str) -> tuple[bool | str, str]:
    has_number = "42" in text
    has_color = "红" in text or "red" in text.lower()
    return (
        True if has_number and has_color else "partial" if has_number or has_color else False
    ), text


def check_tool(args: dict | None, final: str) -> tuple[bool | str, str]:
    called = bool(args) and "杭州" in str(args.get("city", ""))
    used = "小雨" in final or "17" in final
    note = f"args={json.dumps(args, ensure_ascii=False)} final={final}"
    return (True if called and used else "partial" if called else False), note


def red_png_with_42(size=64, scale=4) -> bytes:
    """A red square with a white "42" from a 5x7 bitmap font; stdlib only (no Pillow)."""
    glyphs = {
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    }
    width = (5 * 2 + 1) * scale
    x0, y0 = (size - width) // 2, (size - 7 * scale) // 2
    white = set()
    for index, char in enumerate("42"):
        for row, bits in enumerate(glyphs[char]):
            for col, bit in enumerate(bits):
                if bit == "1":
                    for dy in range(scale):
                        for dx in range(scale):
                            white.add((x0 + (index * 6 + col) * scale + dx, y0 + row * scale + dy))
    raw = b"".join(
        b"\x00"
        + b"".join(b"\xff\xff\xff" if (x, y) in white else b"\xdc\x14\x14" for x in range(size))
        for y in range(size)
    )

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )


@dataclass
class Usage:
    """Normalised usage: input includes cached input; output is billed output incl. thinking."""

    input: int = 0
    cached: int = 0
    output: int = 0
    thinking: int | None = None  # separately reported thinking tokens, None if not reported
    requests: int = 0
    raw: list = field(default_factory=list)

    def add(self, other: "Usage") -> None:
        self.input += other.input
        self.cached += other.cached
        self.output += other.output
        if other.thinking is not None:
            self.thinking = (self.thinking or 0) + other.thinking
        self.requests += other.requests
        self.raw += other.raw


def gemini_sdk_usage(response) -> Usage:
    m = response.usage_metadata
    if m is None:
        return Usage(requests=1, raw=[None])
    thoughts = m.thoughts_token_count or 0
    return Usage(
        input=m.prompt_token_count or 0,
        cached=m.cached_content_token_count or 0,
        output=(m.candidates_token_count or 0) + thoughts,
        thinking=thoughts,
        requests=1,
        raw=[m.model_dump(exclude_none=True, mode="json")],
    )


def anthropic_sdk_usage(response) -> Usage:
    u = response.usage
    cached = u.cache_read_input_tokens or 0
    raw = u.model_dump(exclude_none=True, mode="json")
    # Relay extension: it translates to an OpenAI-style backend and reports hidden reasoning
    # under billing_usage.openai_usage.completion_tokens_details.reasoning_tokens.
    billing = (raw.get("billing_usage") or {}).get("openai_usage") or {}
    reasoning = (billing.get("completion_tokens_details") or {}).get("reasoning_tokens")
    return Usage(
        input=u.input_tokens + cached + (u.cache_creation_input_tokens or 0),
        cached=cached,
        output=u.output_tokens,
        thinking=reasoning,
        requests=1,
        raw=[raw],
    )


class ProbeFailure(Exception):
    """A failed probe that still consumed tokens; carries usage for cost accounting."""

    def __init__(self, note: str, usage: Usage):
        super().__init__(note)
        self.usage = usage


def pai_usage(messages, protocol: str) -> Usage:
    """Sum per-response usage.

    Measured against the SDK: for Google, Pydantic AI output_tokens already equals
    candidates + thoughts, so it is the billed output; details["thoughts_tokens"] is a subset.
    """
    total = Usage()
    for message in messages:
        u = getattr(message, "usage", None)
        if u is None or getattr(message, "kind", "") != "response":
            continue
        details = dict(u.details or {})
        thoughts = details.get("thoughts_tokens", 0) if protocol == "google-genai" else None
        total.add(
            Usage(
                input=u.input_tokens,
                cached=u.cache_read_tokens,
                output=u.output_tokens,
                thinking=thoughts,
                requests=1,
                raw=[
                    {
                        "input_tokens": u.input_tokens,
                        "output_tokens": u.output_tokens,
                        "cache_read_tokens": u.cache_read_tokens,
                        "details": details,
                    }
                ],
            )
        )
    return total


def pai_trace(messages) -> str:
    """Compact per-request trace of message parts, e.g. R1[thinking,tool-call:final_result]/stop."""
    out, n = [], 0
    for message in messages:
        kinds = []
        for part in getattr(message, "parts", []):
            kind = getattr(part, "part_kind", type(part).__name__)
            name = getattr(part, "tool_name", None)
            kinds.append(f"{kind}:{name}" if name else kind)
        if message.kind == "response":
            n += 1
            reason = getattr(message, "finish_reason", None) or ""
            raw = (getattr(message, "provider_details", None) or {}).get("finish_reason")
            if raw and str(raw).lower() != reason:
                reason += f"({raw})"
            out.append(f"R{n}[{','.join(kinds)}]{'/' + reason if reason else ''}")
        elif any(k.startswith("retry") for k in kinds):
            why = [
                str(p.content)[:160]
                for p in message.parts
                if getattr(p, "part_kind", "") == "retry-prompt"
            ]
            out.append(f"Q[{','.join(kinds)}: {' | '.join(why)}]")
    return " ".join(out)


class Redactor:
    def __init__(self, secrets: list[str]):
        self.secrets = [s for s in secrets if s]

    def __call__(self, text: str) -> str:
        for secret in self.secrets:
            text = text.replace(secret, "<redacted>")
        text = re.sub(r"https?://[^\s'\"]+", "<url>", text)
        return re.sub(r"\s+", " ", text)[:400]


class Probe:
    def __init__(self, spec: ModelSpec, base_url: str, api_key: str, png: bytes):
        self.spec, self.base_url, self.api_key, self.png = spec, base_url, api_key, png
        self.google = spec.protocol == "google-genai"

    # ---------- Pydantic AI ----------

    def pai_model(self, native_profile=False):
        if self.google:
            from pydantic_ai.models.google import GoogleModel
            from pydantic_ai.providers.google import GoogleProvider

            # GoogleProvider(base_url=...) alone gets 404: google-genai>=3 defaults to /v1,
            # the relay serves only /v1beta. Pass a client with an explicit api_version.
            provider = GoogleProvider(client=self.genai_client())
            return GoogleModel(self.spec.model, provider=provider)
        from pydantic_ai.models.anthropic import AnthropicModel
        from pydantic_ai.profiles.anthropic import anthropic_model_profile
        from pydantic_ai.providers.anthropic import AnthropicProvider

        provider = AnthropicProvider(api_key=self.api_key, base_url=self.base_url)
        profile = None
        if native_profile:
            # The default profile for unknown model names disables native JSON-schema output.
            profile = replace(
                anthropic_model_profile(self.spec.model), supports_json_schema_output=True
            )
        return AnthropicModel(self.spec.model, provider=provider, profile=profile)

    def pai_settings(self, thinking: str | int | None = "default"):
        base = {"max_tokens": MAX_TOKENS, "timeout": self.spec.timeout_seconds}
        if self.google:
            from pydantic_ai.models.google import GoogleModelSettings

            level = self.spec.thinking_level if thinking == "default" else thinking
            if level:
                base["google_thinking_config"] = {"thinking_level": level}
            return GoogleModelSettings(**base)
        from pydantic_ai.models.anthropic import AnthropicModelSettings

        budget = self.spec.thinking_budget_tokens if thinking == "default" else thinking
        if budget:
            base["anthropic_thinking"] = {"type": "enabled", "budget_tokens": budget}
        return AnthropicModelSettings(**base)

    async def pai(self, case: str):
        from pydantic_ai import Agent, NativeOutput, capture_run_messages

        notes = []
        if case == "struct_native" and not self.google:
            # Record the default-profile behaviour first, then force the native schema path.
            try:
                agent = Agent(self.pai_model(), output_type=NativeOutput(Person))
                await agent.run(EXTRACT_PROMPT, model_settings=self.pai_settings())
                notes.append("default profile: accepted")
            except Exception as error:  # noqa: BLE001 - record the library behaviour
                notes.append(f"default profile: {type(error).__name__}")
        with capture_run_messages() as messages:
            try:
                return await self._pai(case, notes)
            except Exception as error:
                usage = pai_usage(messages, self.spec.protocol)
                note = f"{type(error).__name__}: {error} | {pai_trace(messages)}"
                raise ProbeFailure(note, usage) from error

    async def _pai(self, case: str, notes: list[str]):
        from pydantic_ai import Agent, BinaryContent, NativeOutput, PromptedOutput, ToolOutput
        from pydantic_ai.messages import ThinkingPart, ToolCallPart
        from pydantic_ai.settings import ToolOrOutput

        protocol = self.spec.protocol
        if case in ("text", "image"):
            agent = Agent(self.pai_model(), model_settings=self.pai_settings())
            prompt = (
                TEXT_PROMPT
                if case == "text"
                else [IMAGE_PROMPT, BinaryContent(data=self.png, media_type="image/png")]
            )
            result = await agent.run(prompt)
            ok, note = (True, result.output) if case == "text" else check_image(result.output)
            return ok, note, pai_usage(result.all_messages(), protocol)
        if case == "tool":
            agent = Agent(self.pai_model(), model_settings=self.pai_settings())

            @agent.tool_plain
            def get_weather(city: str) -> dict:
                """查询城市当前天气"""
                return weather_result(city)

            result = await agent.run(TOOL_PROMPT)
            messages = result.all_messages()
            calls = [
                p
                for m in messages
                for p in getattr(m, "parts", [])
                if isinstance(p, ToolCallPart) and p.tool_name == "get_weather"
            ]
            # The last call is the one that produced the tool result the answer used.
            args = calls[-1].args_as_dict() if calls else None
            ok, note = check_tool(args, result.output)
            if ok is True and len(calls) > 1:
                ok = "partial"  # a malformed first call needed a framework retry
            return ok, f"{pai_trace(messages)} {note}", pai_usage(messages, protocol)
        if case.startswith("struct_"):
            output = {
                "struct_tool": lambda: ToolOutput(Person),
                # SDK request shape: named, described tool pinned via allowed_function_names
                # (Google) / {"type": "tool"} (Anthropic); see the tool_choice setting below.
                "struct_tool_swap": lambda: ToolOutput(
                    Person, name="record_person", description=RECORD_DESCRIPTION
                ),
                "struct_native": lambda: NativeOutput(Person),
                "struct_prompt": lambda: PromptedOutput(Person),
            }[case]()
            settings = self.pai_settings()
            if case == "struct_tool_swap":
                # Restricting to output tools makes Pydantic AI name them explicitly.
                settings["tool_choice"] = ToolOrOutput(function_tools=[])
            agent = Agent(
                self.pai_model(native_profile=case == "struct_native"),
                output_type=output,
                model_settings=settings,
            )
            result = await agent.run(EXTRACT_PROMPT)
            messages = result.all_messages()
            ok, note = check_person(result.output)
            if ok is True and sum(m.kind == "response" for m in messages) > 1:
                ok = "partial"  # validated only after a framework retry
            return ok, "; ".join(notes + [pai_trace(messages), note]), pai_usage(messages, protocol)
        if case == "thinking":
            total = Usage()
            levels = ["MINIMAL", "LOW", "HIGH"] if self.google else [0, 1024]
            for level in levels:
                agent = Agent(self.pai_model(), model_settings=self.pai_settings(level))
                result = await agent.run(THINK_PROMPT)
                usage = pai_usage(result.all_messages(), protocol)
                parts = [p for m in result.all_messages() for p in getattr(m, "parts", [])]
                thinking_chars = sum(len(p.content) for p in parts if isinstance(p, ThinkingPart))
                notes.append(
                    f"{level}: out={usage.output} thinking={usage.thinking} "
                    f"thinking_part_chars={thinking_chars}"
                )
                total.add(usage)
            return True, "; ".join(notes), total
        if case == "stream":
            agent = Agent(self.pai_model(), model_settings=self.pai_settings())
            chunks, text = 0, ""
            async with agent.run_stream(TEXT_PROMPT) as stream:
                async for text in stream.stream_text():
                    chunks += 1
                usage = stream.usage()
            normalised = Usage(
                input=usage.input_tokens,
                cached=usage.cache_read_tokens,
                output=usage.output_tokens,
                thinking=usage.details.get("thoughts_tokens") if self.google else None,
                requests=usage.requests,
                raw=[{"output_tokens": usage.output_tokens, "details": dict(usage.details)}],
            )
            return bool(text), f"chunks={chunks} text={text}", normalised
        raise ValueError(case)

    # ---------- official SDKs ----------

    async def sdk(self, case: str):
        return await (self.sdk_google(case) if self.google else self.sdk_anthropic(case))

    def genai_client(self):
        from google import genai
        from google.genai import types

        return genai.Client(
            api_key=self.api_key,
            http_options=types.HttpOptions(
                base_url=self.base_url,
                api_version=self.spec.api_version,
                timeout=int(self.spec.timeout_seconds * 1000),
            ),
        )

    async def sdk_google(self, case: str):
        from google.genai import types

        client = self.genai_client().aio

        def config(level="default", **extra):
            level = self.spec.thinking_level if level == "default" else level
            thinking = types.ThinkingConfig(thinking_level=level) if level else None
            return types.GenerateContentConfig(
                max_output_tokens=MAX_TOKENS, thinking_config=thinking, **extra
            )

        async def call(contents, **kwargs):
            return await client.models.generate_content(
                model=self.spec.model, contents=contents, config=config(**kwargs)
            )

        tool = types.Tool(
            function_declarations=[
                types.FunctionDeclaration(
                    name=WEATHER_TOOL["name"],
                    description=WEATHER_TOOL["description"],
                    parameters_json_schema=WEATHER_TOOL["schema"],
                )
            ]
        )
        if case == "text":
            r = await call(TEXT_PROMPT)
            return bool(r.text), r.text, gemini_sdk_usage(r)
        if case == "image":
            r = await call(
                [types.Part.from_bytes(data=self.png, mime_type="image/png"), IMAGE_PROMPT]
            )
            ok, note = check_image(r.text or "")
            return ok, note, gemini_sdk_usage(r)
        if case == "tool":
            user = types.Content(role="user", parts=[types.Part.from_text(text=TOOL_PROMPT)])
            r1 = await call([user], tools=[tool])
            usage = gemini_sdk_usage(r1)
            calls = r1.function_calls or []
            if not calls:
                reason = r1.candidates[0].finish_reason if r1.candidates else None
                return False, f"no function call: finish={reason} {r1.text}", usage
            args = dict(calls[0].args or {})
            reply = types.Content(
                role="user",
                parts=[
                    types.Part.from_function_response(
                        name="get_weather", response=weather_result(args.get("city", ""))
                    )
                ],
            )
            # Send the model turn back unchanged so thought signatures are preserved.
            r2 = await call([user, r1.candidates[0].content, reply], tools=[tool])
            usage.add(gemini_sdk_usage(r2))
            ok, note = check_tool(args, r2.text or "")
            return ok, note, usage
        if case in ("struct_tool", "struct_tool_swap"):
            # struct_tool_swap mirrors Pydantic AI's default ToolOutput request: generic name,
            # no description, mode ANY without allowed_function_names.
            alt = case == "struct_tool_swap"
            declaration = types.FunctionDeclaration(
                name="final_result" if alt else "record_person",
                description="" if alt else RECORD_DESCRIPTION,
                parameters_json_schema=Person.model_json_schema(),
            )
            r = await call(
                EXTRACT_PROMPT,
                tools=[types.Tool(function_declarations=[declaration])],
                tool_config=types.ToolConfig(
                    function_calling_config=types.FunctionCallingConfig(
                        mode="ANY", allowed_function_names=None if alt else ["record_person"]
                    )
                ),
            )
            calls = r.function_calls or []
            if not calls:
                reason = r.candidates[0].finish_reason if r.candidates else None
                return False, f"no function call: finish={reason} {r.text}", gemini_sdk_usage(r)
            ok, note = check_person(dict(calls[0].args or {}))
            return ok, note, gemini_sdk_usage(r)
        if case == "struct_native":
            r = await call(
                EXTRACT_PROMPT,
                response_mime_type="application/json",
                response_json_schema=Person.model_json_schema(),
            )
            ok, note = check_person(r.text or "")
            return ok, note, gemini_sdk_usage(r)
        if case == "struct_prompt":
            r = await call(EXTRACT_PROMPT + PROMPTED_SUFFIX)
            ok, note = check_person(r.text or "")
            return ok, note, gemini_sdk_usage(r)
        if case == "thinking":
            total, notes = Usage(), []
            for level in ("MINIMAL", "LOW", "HIGH"):
                r = await call(THINK_PROMPT, level=level)
                usage = gemini_sdk_usage(r)
                notes.append(f"{level}: out={usage.output} thinking={usage.thinking}")
                total.add(usage)
            return True, "; ".join(notes), total
        if case == "stream":
            chunks, text, last = 0, "", None
            stream = await client.models.generate_content_stream(
                model=self.spec.model, contents=TEXT_PROMPT, config=config()
            )
            async for chunk in stream:
                chunks += 1
                text += chunk.text or ""
                last = chunk if chunk.usage_metadata else last
            usage = gemini_sdk_usage(last) if last else Usage(requests=1)
            return bool(text), f"chunks={chunks} text={text}", usage
        raise ValueError(case)

    async def sdk_anthropic(self, case: str):
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(
            api_key=self.api_key, base_url=self.base_url, timeout=self.spec.timeout_seconds
        )

        async def call(messages, thinking: int | None = None, **kwargs):
            budget = self.spec.thinking_budget_tokens if thinking is None else thinking
            if budget:
                kwargs["thinking"] = {"type": "enabled", "budget_tokens": budget}
            return await client.messages.create(
                model=self.spec.model, max_tokens=MAX_TOKENS, messages=messages, **kwargs
            )

        def text_of(response) -> str:
            return "".join(b.text for b in response.content if b.type == "text")

        tool = {
            "name": WEATHER_TOOL["name"],
            "description": WEATHER_TOOL["description"],
            "input_schema": WEATHER_TOOL["schema"],
        }
        if case == "text":
            r = await call([{"role": "user", "content": TEXT_PROMPT}])
            return bool(text_of(r)), text_of(r), anthropic_sdk_usage(r)
        if case == "image":
            image = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": base64.b64encode(self.png).decode(),
                },
            }
            r = await call(
                [{"role": "user", "content": [image, {"type": "text", "text": IMAGE_PROMPT}]}]
            )
            ok, note = check_image(text_of(r))
            return ok, note, anthropic_sdk_usage(r)
        if case == "tool":
            messages = [{"role": "user", "content": TOOL_PROMPT}]
            r1 = await call(messages, tools=[tool])
            usage = anthropic_sdk_usage(r1)
            uses = [b for b in r1.content if b.type == "tool_use"]
            if not uses:
                return False, f"no tool_use: stop={r1.stop_reason} {text_of(r1)}", usage
            args = dict(uses[0].input)
            messages += [
                {"role": "assistant", "content": [b.model_dump() for b in r1.content]},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": uses[0].id,
                            "content": json.dumps(
                                weather_result(args.get("city", "")), ensure_ascii=False
                            ),
                        }
                    ],
                },
            ]
            r2 = await call(messages, tools=[tool])
            usage.add(anthropic_sdk_usage(r2))
            ok, note = check_tool(args, text_of(r2))
            return ok, note, usage
        if case in ("struct_tool", "struct_tool_swap"):
            alt = case == "struct_tool_swap"  # generic name, tool_choice "any"
            name = "final_result" if alt else "record_person"
            schema_tool = {
                "name": name,
                "description": "" if alt else RECORD_DESCRIPTION,
                "input_schema": Person.model_json_schema(),
            }
            r = await call(
                [{"role": "user", "content": EXTRACT_PROMPT}],
                tools=[schema_tool],
                tool_choice={"type": "any"} if alt else {"type": "tool", "name": name},
            )
            uses = [b for b in r.content if b.type == "tool_use"]
            if not uses:
                return False, f"no tool_use: stop={r.stop_reason}", anthropic_sdk_usage(r)
            ok, note = check_person(dict(uses[0].input))
            return ok, note, anthropic_sdk_usage(r)
        if case == "struct_native":
            # Anthropic structured outputs: output_config.format with a JSON schema.
            r = await call(
                [{"role": "user", "content": EXTRACT_PROMPT}],
                extra_body={
                    "output_config": {
                        "format": {"type": "json_schema", "schema": Person.model_json_schema()}
                    }
                },
            )
            ok, note = check_person(text_of(r))
            return ok, note, anthropic_sdk_usage(r)
        if case == "struct_prompt":
            r = await call([{"role": "user", "content": EXTRACT_PROMPT + PROMPTED_SUFFIX}])
            ok, note = check_person(text_of(r))
            return ok, note, anthropic_sdk_usage(r)
        if case == "thinking":
            total, notes = Usage(), []
            variants = {
                "off": {"thinking": 0},
                "budget1024": {"thinking": 1024},
                "disabled": {"thinking": 0, "extra_body": {"thinking": {"type": "disabled"}}},
                "effort_low": {"thinking": 0, "extra_body": {"reasoning_effort": "low"}},
                "effort_high": {"thinking": 0, "extra_body": {"reasoning_effort": "high"}},
            }
            for label, kwargs in variants.items():
                r = await call([{"role": "user", "content": THINK_PROMPT}], **kwargs)
                usage = anthropic_sdk_usage(r)
                chars = sum(len(b.thinking) for b in r.content if b.type == "thinking")
                notes.append(
                    f"{label}: out={usage.output} reasoning={usage.thinking} "
                    f"thinking_block_chars={chars} text={text_of(r)[:20]}"
                )
                total.add(usage)
            return True, "; ".join(notes), total
        if case == "stream":
            chunks, text = 0, ""
            async with client.messages.stream(
                model=self.spec.model,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": TEXT_PROMPT}],
            ) as stream:
                async for delta in stream.text_stream:
                    chunks += 1
                    text += delta
                final = await stream.get_final_message()
            return bool(text), f"chunks={chunks} text={text}", anthropic_sdk_usage(final)
        raise ValueError(case)


def cost_cny(spec: ModelSpec, usage: Usage) -> float:
    cached_price = spec.price_cached_input if spec.price_cached_input is not None else 0
    fresh = max(usage.input - usage.cached, 0)
    return (
        fresh * spec.price_input + usage.cached * cached_price + usage.output * spec.price_output
    ) / 1_000_000


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--settings", type=Path, default=ROOT / "config/settings.example.toml")
    parser.add_argument("--model", nargs="+", default=["gemini", "deepseek"])
    parser.add_argument("--via", nargs="+", choices=["pai", "sdk"], default=["pai", "sdk"])
    parser.add_argument("--case", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--repeat", type=int, default=1, help="run each case N times")
    parser.add_argument("--budget-cny", type=float, default=0.3)
    args = parser.parse_args()

    settings = load_settings(args.settings)
    png = red_png_with_42()
    out_dir = ROOT / "data/probes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"model-probe-{datetime.now():%Y%m%d-%H%M%S}.jsonl"
    spent = 0.0
    with out_path.open("w", encoding="utf-8") as out:
        for name in args.model:
            spec = settings.models[name]
            creds = load_model_credentials(settings.credentials, spec)
            redact = Redactor([creds.api_key, creds.base_url])
            probe = Probe(spec, creds.base_url, creds.api_key, png)
            for via in args.via:
                consecutive_errors = 0
                for case in [c for c in args.case for _ in range(args.repeat)]:
                    if spent >= args.budget_cny:
                        print(f"budget reached ({spent:.4f} CNY); stopping")
                        return 2
                    if consecutive_errors >= 3:
                        print(f"{name}/{via}: 3 consecutive errors; skipping remaining cases")
                        break
                    start = time.perf_counter()
                    record = {"model": name, "via": via, "case": case}
                    try:
                        runner = probe.pai if via == "pai" else probe.sdk
                        ok, note, usage = await runner(case)
                        consecutive_errors = 0
                    except ProbeFailure as failure:
                        # With responses recorded the model answered and the framework gave up;
                        # without any, it was a transport/HTTP error.
                        ok, usage, note = False, failure.usage, str(failure)
                        consecutive_errors = 0 if usage.requests else consecutive_errors + 1
                    except Exception as error:  # noqa: BLE001 - probes record every failure
                        ok, usage = False, Usage()
                        note = f"{type(error).__name__}: {redact(str(error))}"
                        consecutive_errors += 1
                    record.update(
                        ok=ok,
                        latency_s=round(time.perf_counter() - start, 2),
                        input=usage.input,
                        cached=usage.cached,
                        output=usage.output,
                        thinking=usage.thinking,
                        requests=usage.requests,
                        cost_cny=round(cost_cny(spec, usage), 6),
                        note=redact(note) if isinstance(note, str) else note,
                        raw_usage=usage.raw,
                    )
                    spent += record["cost_cny"]
                    out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out.flush()
                    print(
                        f"{name:8} {via:3} {case:13} ok={str(ok):7} {record['latency_s']:6.2f}s "
                        f"in={usage.input} cached={usage.cached} out={usage.output} "
                        f"think={usage.thinking} req={usage.requests} "
                        f"cost={record['cost_cny']:.6f} | {record['note'][:160]}"
                    )
    print(f"total_cost_cny={spent:.6f} records={out_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
