import importlib.util
from pathlib import Path


spec = importlib.util.spec_from_file_location("repo_safety", Path("scripts/check_repo_safety.py"))
safety = importlib.util.module_from_spec(spec)
spec.loader.exec_module(safety)


def test_rejects_private_data_and_actual_key(tmp_path):
    (tmp_path / "ordinary.py").write_text("actual-test-key-value", encoding="utf-8")
    issues = safety.check_files(
        tmp_path, ["tests/replay/raw/case.json", "ordinary.py"], [b"actual-test-key-value"]
    )
    assert issues == [
        ("tests/replay/raw/case.json", "local-only-file"),
        ("ordinary.py", "possible-secret-value"),
    ]
    assert "actual-test-key-value" not in repr(issues)


def test_git_list_supports_windows_and_modern_shell_encodings(tmp_path):
    expected = ["README.md", "docs/虚构.md"]
    for encoding in ("utf-8", "utf-8-sig", "utf-16"):
        path = tmp_path / (encoding + ".bin")
        path.write_bytes(("\0".join(expected) + "\0\r\n").encode(encoding))
        assert safety.read_git_names(path) == expected


def test_allows_safe_example_and_rejects_escape(tmp_path):
    (tmp_path / "example.toml").write_text('api_key_env = "JEV_API_KEY"', encoding="utf-8")
    assert safety.check_files(tmp_path, ["example.toml"], [b"some-secret-value"]) == []
    assert safety.check_files(tmp_path, ["../elsewhere.txt"], [b"some-secret-value"]) == [
        ("../elsewhere.txt", "outside-root-or-symlink")
    ]
