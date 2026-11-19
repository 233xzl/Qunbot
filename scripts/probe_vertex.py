"""Bounded Vertex API-key smoke probe; synthetic input only, no bot connection.

Run from the project directory with .venv/Scripts/python.exe scripts/probe_vertex.py.
Use --capabilities after IAM is fixed to also probe image, function and search support.
"""

import argparse
import io
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from google import genai
from google.genai import types
from PIL import Image

from group_agent.config import ModelSpec, load_model_credentials


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--secrets", type=Path, required=True)
    parser.add_argument("--models", nargs="+", default=["gemini-3.8-flash", "gemini-4-hydrogen"])
    parser.add_argument("--capabilities", action="store_true")
    parser.add_argument(
        "--project", help="Explicit project ID or number for a standard resource path"
    )
    parser.add_argument("--location", default="global")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    spec = ModelSpec(
        "vertex_probe",
        "google-genai",
        args.models[0],
        "",
        "GOOGLE_CLOUD_API_KEY",
        25,
        0,
        0,
        base_url="https://aiplatform.googleapis.com",
    )
    creds = load_model_credentials(args.secrets, spec)
    records = []
    with genai.Client(
        vertexai=True,
        api_key=creds.api_key,
        http_options=types.HttpOptions(
            api_version="v1",
            timeout=25000,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    ) as client:
        for model in args.models:
            resource = (
                (
                    f"projects/{args.project}/locations/{args.location}/publishers/google/"
                    f"models/{model}"
                )
                if args.project
                else model
            )
            cases = [("text", "Reply with exactly OK.", {})]
            if args.capabilities:
                png = io.BytesIO()
                Image.new("RGB", (64, 64), "red").save(png, format="PNG")
                cases.extend(
                    [
                        (
                            "image",
                            [
                                "Name the solid color in one English word.",
                                types.Part.from_bytes(data=png.getvalue(), mime_type="image/png"),
                            ],
                            {},
                        ),
                        (
                            "function",
                            "Call echo with value vertex_probe. Then reply only with its result.",
                            {
                                "tools": [
                                    types.Tool(
                                        function_declarations=[
                                            types.FunctionDeclaration(
                                                name="echo",
                                                description="Echo the value.",
                                                parameters={
                                                    "type": "OBJECT",
                                                    "properties": {"value": {"type": "STRING"}},
                                                    "required": ["value"],
                                                },
                                            )
                                        ]
                                    )
                                ],
                                "tool_config": types.ToolConfig(
                                    function_calling_config=types.FunctionCallingConfig(
                                        mode="ANY", allowed_function_names=["echo"]
                                    )
                                ),
                            },
                        ),
                        (
                            "search",
                            "Find the official Python documentation for asyncio.TaskGroup. "
                            "Give its URL and a one-sentence description, using Google Search.",
                            {
                                "tools": [types.Tool(google_search=types.GoogleSearch())],
                            },
                        ),
                    ]
                )
            for case, contents, extra in cases:
                start = time.monotonic()
                record = {
                    "model": model,
                    "case": case,
                    "location": args.location if args.project else "provider_default",
                }
                try:
                    result = client.models.generate_content(
                        model=resource,
                        contents=contents,
                        config=types.GenerateContentConfig(
                            max_output_tokens=512,
                            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                            **extra,
                        ),
                    )
                    text = "".join(
                        p.text or ""
                        for c in result.candidates or []
                        for p in (c.content.parts if c.content else []) or []
                        if not p.thought
                    )
                    calls = [{"name": f.name, "args": f.args} for f in result.function_calls or []]
                    sources = sum(
                        len(c.grounding_metadata.grounding_chunks or [])
                        for c in result.candidates or []
                        if c.grounding_metadata
                    )
                    queries = sum(
                        len(c.grounding_metadata.web_search_queries or [])
                        for c in result.candidates or []
                        if c.grounding_metadata
                    )
                    passed = {
                        "text": text.strip() == "OK",
                        "image": "red" in text.lower(),
                        "function": any(
                            f["name"] == "echo" and f["args"] == {"value": "vertex_probe"}
                            for f in calls
                        ),
                        "search": sources > 0 and queries > 0,
                    }[case]
                    record.update(
                        http_ok=True,
                        passed=passed,
                        text=text,
                        calls=calls,
                        sources=sources,
                        queries=queries,
                        usage=result.usage_metadata.model_dump(mode="json")
                        if result.usage_metadata
                        else None,
                    )
                    if case == "function" and passed:
                        followup = client.models.generate_content(
                            model=resource,
                            contents=[
                                types.Content(role="user", parts=[types.Part(text=contents)]),
                                result.candidates[0].content,
                                types.Content(
                                    role="user",
                                    parts=[
                                        types.Part.from_function_response(
                                            name="echo",
                                            response={"result": "vertex_probe"},
                                        )
                                    ],
                                ),
                            ],
                            config=types.GenerateContentConfig(
                                max_output_tokens=512,
                                thinking_config=types.ThinkingConfig(thinking_level="LOW"),
                            ),
                        )
                        record["roundtrip_text"] = followup.text
                        record["roundtrip_usage"] = (
                            followup.usage_metadata.model_dump(mode="json")
                            if followup.usage_metadata
                            else None
                        )
                        record["passed"] = "vertex_probe" in (followup.text or "")
                except Exception as exc:
                    # Do not persist exception text: it can contain URLs, headers or credentials.
                    record.update(
                        http_ok=False,
                        passed=False,
                        error_type=type(exc).__name__,
                        status=getattr(exc, "code", None),
                        missing_predict_permission="aiplatform.endpoints.predict" in str(exc),
                    )
                record["seconds"] = round(time.monotonic() - start, 3)
                records.append(record)
                print(json.dumps(record, ensure_ascii=True))
                if not record["http_ok"] and record.get("status") in (401, 403, 404):
                    break
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or Path(f"data/probes/vertex-20261114/probe-{stamp}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"at_utc": stamp, "records": records}, indent=2), encoding="utf-8")
    return 0 if all(r["passed"] for r in records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
