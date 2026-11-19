from group_agent.models import _error_causes, _redacted_error, infrastructure_failure, CallRecord


def test_wrapped_transport_status_is_preserved_without_messages_or_addresses():
    class HTTPError(Exception):
        status_code = 503

    inner = HTTPError("https://secret.invalid key=secret")
    outer = RuntimeError("response body with private material")
    outer.__cause__ = inner
    record = CallRecord("answer", "backup", "text", status="error", error=_redacted_error(outer))
    assert record.error == "RuntimeError:503"
    assert infrastructure_failure(record)
    assert _error_causes(outer) == ["RuntimeError:503", "HTTPError:503"]
    inner.__cause__ = outer
    assert len(_error_causes(outer)) == 2
