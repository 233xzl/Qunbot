"""Check a NUL-delimited Git file list without printing any secret values.

Create the list with git ls-files --cached -z, then pass it as --file-list.
Run again after staging changes. This is a guardrail, not a proof of anonymity.
"""

import argparse
from pathlib import Path
import re


FORBIDDEN_PARTS = {".venv", "data", "logs", "private", "secrets", "__pycache__"}
FORBIDDEN_SUFFIXES = {
    ".env",
    ".db",
    ".sqlite",
    ".sqlite3",
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".zip",
}
TOKEN_PATTERN = re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")


def secret_values(path: Path) -> list[bytes]:
    values = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if "=" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if any(word in key.upper() for word in ("KEY", "TOKEN", "PASSWORD")) and len(value) >= 8:
            values.append(value.encode())
    return values


def check_files(root: Path, names: list[str], secrets: list[bytes]) -> list[tuple[str, str]]:
    issues = []
    root = root.resolve()
    for name in names:
        path = root / name
        relative = Path(name)
        if path.is_symlink() or not path.resolve().is_relative_to(root):
            issues.append((name, "outside-root-or-symlink"))
            continue
        parts = set(relative.parts)
        if (
            parts & FORBIDDEN_PARTS
            or any(part.startswith("pytest-of-") for part in parts)
            or relative.name.endswith(
                (".db-wal", ".db-shm", ".sqlite-wal", ".sqlite-shm", ".sqlite3-wal", ".sqlite3-shm")
            )
            or relative.suffix.lower() in FORBIDDEN_SUFFIXES
            or relative.name.startswith(".env")
            or name.replace("\\", "/").startswith(("tests/replay/raw/", "tests/replay/private/"))
            or name.replace("\\", "/") == "config/settings.toml"
        ):
            issues.append((name, "local-only-file"))
            continue
        try:
            content = path.read_bytes()
        except OSError:
            issues.append((name, "unreadable-file"))
            continue
        if TOKEN_PATTERN.search(content) or any(secret in content for secret in secrets):
            issues.append((name, "possible-secret-value"))
    return issues


def read_git_names(path: Path) -> list[str]:
    raw = path.read_bytes()
    # Windows PowerShell redirects native stdout as BOM-prefixed UTF-16LE;
    # modern pwsh may preserve Git's UTF-8 bytes instead.
    encoding = "utf-16" if raw.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    text = raw.decode(encoding).rstrip("\r\n")
    return [name for name in text.split("\0") if name]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-list", required=True, type=Path)
    parser.add_argument("--secrets-file", type=Path, required=True)
    args = parser.parse_args()
    try:
        values = secret_values(args.secrets_file)
        names = read_git_names(args.file_list)
    except (OSError, UnicodeError):
        parser.error("cannot read file list or local secrets file")
    if not names or not values:
        parser.error("file list and secret-value set must both be nonempty")
    issues = check_files(Path.cwd(), names, values)
    for name, reason in issues:
        print(f"{reason}: {name}")
    print(f"checked_files={len(names)} issues={len(issues)} (secret values are never printed)")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
