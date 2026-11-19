"""Small protocol adapters for Pydantic AI; credentials never leave the factory."""

from dataclasses import dataclass
from typing import Any, Callable

from .config import ModelCredentials, ModelSpec


def google_client(
    spec: ModelSpec,
    creds: ModelCredentials,
    timeout_s: float | None = None,
    *,
    relay_api_version: str | None = None,
):
    """The one google-genai client factory for generation and independent web requests.

    Vertex mode: vertexai=True with the API key, `spec.api_version` (v1) and a single
    transport attempt; requests must name the full resource path (`google_model_name`),
    because the key's default region returned 404 for Hydrogen. Relay mode: the custom
    base URL with `spec.api_version`, else `relay_api_version` (None = SDK default)."""
    from google import genai
    from google.genai import types

    timeout = max(1000, int((spec.timeout_seconds if timeout_s is None else timeout_s) * 1000))
    if spec.vertex:
        if not creds.project:
            raise ValueError(f"models.{spec.name}: Vertex credentials need a project")
        return genai.Client(
            vertexai=True,
            api_key=creds.api_key,
            http_options=types.HttpOptions(
                api_version=spec.api_version or "v1",
                timeout=timeout,
                retry_options=types.HttpRetryOptions(attempts=1),
            ),
        )
    return genai.Client(
        api_key=creds.api_key,
        http_options=types.HttpOptions(
            base_url=creds.base_url,
            api_version=spec.api_version or relay_api_version,
            timeout=timeout,
        ),
    )


def google_model_name(spec: ModelSpec, creds: ModelCredentials, model: str | None = None) -> str:
    """Request model name: the full Vertex resource path, or the short name on a relay.
    Only requests use the path; records, prices and capabilities keep the short name."""
    short = model or spec.model
    if not spec.vertex:
        return short
    return (
        f"projects/{creds.project}/locations/{spec.location or 'global'}"
        f"/publishers/google/models/{short}"
    )


def _google(spec, creds):
    from pydantic_ai.models.google import GoogleModel

    client = google_client(spec, creds)
    if spec.vertex:
        from pydantic_ai.providers.google_cloud import GoogleCloudProvider

        provider = GoogleCloudProvider(client=client)
    else:
        from pydantic_ai.providers.google import GoogleProvider

        provider = GoogleProvider(client=client)
    return GoogleModel(google_model_name(spec, creds), provider=provider)


def _anthropic(spec, creds):
    from pydantic_ai.models.anthropic import AnthropicModel
    from pydantic_ai.providers.anthropic import AnthropicProvider

    return AnthropicModel(
        spec.model,
        provider=AnthropicProvider(
            api_key=creds.api_key,
            base_url=creds.base_url,
        ),
    )


def _openai(spec, creds):
    from openai import AsyncOpenAI
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.openai import OpenAIProvider

    return OpenAIChatModel(
        spec.model,
        provider=OpenAIProvider(
            openai_client=AsyncOpenAI(
                api_key=creds.api_key,
                base_url=creds.base_url,
                max_retries=0,
            )
        ),
    )


@dataclass(frozen=True)
class ProtocolAdapter:
    factory: Callable
    native_output: bool = False
    google_thinking: bool = False

    def settings(self, spec: ModelSpec, level: str | None, pin_output: bool) -> dict[str, Any]:
        settings: dict[str, Any] = {"timeout": spec.timeout_seconds}
        if spec.max_output_tokens:
            settings["max_tokens"] = spec.max_output_tokens
        if spec.reasoning_effort is not None:
            settings["openai_reasoning_effort"] = spec.reasoning_effort
        if self.google_thinking:
            if level:
                settings["google_thinking_config"] = {"thinking_level": level}
            if pin_output:
                from pydantic_ai.settings import ToolOrOutput

                settings["tool_choice"] = ToolOrOutput(function_tools=[])
        return settings


PROTOCOL_ADAPTERS = {
    "google-genai": ProtocolAdapter(_google, native_output=True, google_thinking=True),
    # The configured Anthropic relay ignores native schema and thinking settings.
    "anthropic": ProtocolAdapter(_anthropic),
    "openai-compatible": ProtocolAdapter(_openai),
}


def protocol_adapter(protocol: str) -> ProtocolAdapter:
    try:
        return PROTOCOL_ADAPTERS[protocol]
    except KeyError:
        raise ValueError(f"protocol {protocol!r} is not a Pydantic AI model") from None


def build_pai_model(spec: ModelSpec, creds: ModelCredentials):
    return protocol_adapter(spec.protocol).factory(spec, creds)
