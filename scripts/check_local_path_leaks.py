#!/usr/bin/env python3
"""Scan repository text/docs for leaked local absolute paths.

Usage:
    python scripts/check_local_path_leaks.py            # scan and report
    python scripts/check_local_path_leaks.py --fix      # rewrite leaking absolute paths to repo-relative paths
"""

import argparse
import os
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EXTS = {
    ".md", ".txt", ".rst", ".adoc", ".html", ".json",
    ".yaml", ".yml", ".csv",
}
SKIP_DIRS = {
    ".git", "target", "dist", "_internal", ".cargo",
    "node_modules", "__pycache__", ".venv",
}
BINARY_EXTS = {
    ".exe", ".dll", ".pdb", ".wasm", ".zip", ".png", ".jpg",
    ".jpeg", ".gif", ".ico", ".woff", ".ttf", ".o", ".obj",
    ".a", ".lib", ".so", ".dylib", ".pdf", ".docx", ".xlsx",
}

# A windows drive-letter absolute path, e.g. C:\foo, C:\\foo (escaped), or D:/foo.
# The path body is kept loose here; is_sensitive_path() decides which hits are real leaks.
PLAIN_DRIVE_RE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]:(?:\\{1,2}|/)[^\s`\"');]+)"
)
# A file:// URI pointing at a windows drive, e.g. file:///d:/foo
URI_DRIVE_RE = re.compile(
    r"file:///([A-Za-z]:[\\/][^\s\)\]\(]+)"
)

# A simple redacted marker used when an absolute path is outside the repo and --fix is used.
REDACTED = "<LOCAL_PATH>"


def is_sensitive_path(raw_path: str) -> bool:
    """Return True for real local user/profile/project paths rather than fake examples."""
    lowered = raw_path.lower()
    # Explicit generic examples that are commonly used in documentation/tests.
    if re.search(r"(?i)users[\\/]test|path[\\/]no[\\/]escape|/tmp/", lowered):
        return False
    # Paths containing angle brackets are placeholder masks, not real leaks.
    if "<" in raw_path or ">" in raw_path:
        return False
    return bool(
        re.search(r"(?i)users[\\/]", lowered)
        or re.search(r"(?i)desktop[\\/]", lowered)
        or re.search(r"(?i)ai开发新语言|ai寮€鍙戞柊璇|鍙茶拏", lowered)
        or re.search(r"(?i)\.cargo[\\/]|\.rustup[\\/]", lowered)
        or re.search(r"(?i)史蒂夫|steve", lowered)
    )


def should_skip_file(path: Path) -> bool:
    if path.suffix.lower() in BINARY_EXTS:
        return True
    parts = set(path.parts)
    if SKIP_DIRS & parts:
        return True
    return False


def normalize_uri_path(uri_path: str) -> str:
    """Convert 'file:///d:/...' path component to 'd:/...'."""
    p = uri_path
    if p.startswith("/") and len(p) > 2 and p[2] == ":":
        p = p[1:]  # /d:/... -> d:/...
    return p


def source_path_for_match(match: re.Match, kind: str) -> str:
    if kind == "uri":
        return normalize_uri_path(match.group(1))
    return match.group(1)


def to_repo_relative(abs_path: str, base_dir: Path, trailing: bool = False) -> str:
    """Convert an absolute path under ROOT to a repo-relative path."""
    try:
        abs_path = str(Path(abs_path).resolve())
    except OSError:
        abs_path = os.path.abspath(abs_path)
    root_str = str(ROOT.resolve())
    root_prefix = root_str.rstrip("\\/") + os.sep
    if abs_path.lower().startswith(root_prefix.lower()):
        rel = os.path.relpath(abs_path, base_dir).replace("\\", "/")
        if trailing:
            rel += "/"
        return rel
    # For user-profile paths, use a portable env-var placeholder instead of redacting
    # the whole path. This keeps commands/snippets usable while hiding the username.
    user_match = re.match(r"(?i)^[A-Za-z]:[\\/]Users[\\/][^\\/]+(?:[\\/](.*))?$", abs_path)
    if user_match:
        suffix = user_match.group(1) or ""
        suffix = suffix.replace("/", "\\")
        if suffix:
            return "%USERPROFILE%\\" + suffix
        return "%USERPROFILE%"
    return REDACTED


def redact_line(line: str, base_dir: Path) -> str:
    """Replace leaked absolute paths in one text line with relative paths/redaction."""
    def repl_plain(m: re.Match) -> str:
        raw = m.group(1)
        norm = raw.replace("\\\\", "\\")
        if not is_sensitive_path(norm):
            return raw
        trailing = norm.endswith("\\") or norm.endswith("/")
        core = norm.rstrip("\\/")
        rel = to_repo_relative(core, base_dir, trailing)
        return rel

    def repl_uri(m: re.Match) -> str:
        raw = normalize_uri_path(m.group(1))
        if not is_sensitive_path(raw):
            return m.group(0)
        trailing = raw.endswith("/") or raw.endswith("\\")
        core = raw.rstrip("\\/")
        rel = to_repo_relative(core, base_dir, trailing)
        return rel

    line = URI_DRIVE_RE.sub(repl_uri, line)
    line = PLAIN_DRIVE_RE.sub(repl_plain, line)
    return line


def iter_files(exts):
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not d.startswith(".git")
        ]
        for fn in filenames:
            path = Path(dirpath) / fn
            if path.suffix.lower() in DEFAULT_EXTS or path.suffix.lower() in exts:
                if should_skip_file(path):
                    continue
                yield path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fix", action="store_true",
        help="rewrite matched absolute paths to relative paths when under ROOT",
    )
    parser.add_argument(
        "--extend", default="", metavar="EXTS",
        help="comma-separated extra extensions to scan, e.g. '--extend .py,.sh'",
    )
    args = parser.parse_args()

    exts = set(DEFAULT_EXTS)
    for item in args.extend.split(","):
        item = item.strip()
        if item:
            if not item.startswith("."):
                item = "." + item
            exts.add(item)

    total_files = 0
    total_hits = 0
    fixed_files = 0
    for path in iter_files(exts):
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                lines = f.readlines()
        except (UnicodeDecodeError, OSError):
            continue

        changed = False
        file_hits = 0
        for i, line in enumerate(lines, start=1):
            uri_hits = [
                m for m in URI_DRIVE_RE.finditer(line)
                if is_sensitive_path(normalize_uri_path(m.group(1)))
            ]
            plain_hits = [
                m for m in PLAIN_DRIVE_RE.finditer(line)
                if is_sensitive_path(m.group(1).replace("\\\\", "\\"))
            ]
            hits = len(uri_hits) + len(plain_hits)
            if not hits:
                continue
            total_hits += hits
            file_hits += hits
            if args.fix:
                lines[i - 1] = redact_line(line, path.parent)
                changed = True
            else:
                rel = path.relative_to(ROOT).as_posix()
                for m in uri_hits:
                    print(f"{rel}:{i}: file:///{normalize_uri_path(m.group(1))}")
                for m in plain_hits:
                    print(f"{rel}:{i}: {m.group(1).replace('\\\\', '\\')}")

        if changed:
            with open(path, "w", encoding="utf-8", newline="") as f:
                f.writelines(lines)
            fixed_files += 1

        total_files += 1
        if file_hits and not args.fix:
            print(f"  --> {path.relative_to(ROOT).as_posix()} ({file_hits} hit(s))", file=sys.stderr)

    if args.fix:
        print(
            f"Scanned {total_files} files, fixed {fixed_files} files, "
            f"{total_hits} absolute path occurrence(s) processed."
        )
        return 0
    else:
        print(
            f"Scanned {total_files} files, found {total_hits} local absolute path occurrence(s)."
        )
        return 1 if total_hits else 0


if __name__ == "__main__":
    sys.exit(main())
