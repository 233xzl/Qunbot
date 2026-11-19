from group_agent.config import ModelSpec, load_model_credentials
from group_agent.provider_adapters import build_pai_model


def test_official_endpoint_uses_shared_key_without_url_secret(tmp_path):
    secrets = tmp_path / "secrets.env"
    secrets.write_text("DEEPSEEK_SEARCH_API_KEY=test-key\n")
    spec = ModelSpec(
        "official",
        "openai-compatible",
        "model-probe",
        "",
        "DEEPSEEK_SEARCH_API_KEY",
        10,
        0,
        0,
        base_url="https://api.deepseek.com",
    )
    creds = load_model_credentials(secrets, spec)
    assert creds.base_url == "https://api.deepseek.com"
    assert creds.api_key == "test-key"
    model = build_pai_model(spec, creds)
    assert model.client.max_retries == 0
