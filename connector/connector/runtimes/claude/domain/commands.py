"""Claude Code slash-command catalog.

Claude Code resolves slash commands from the input stream, so the connector only
has to advertise what exists and forward the raw ``/command args`` text as a turn.
The CLI reports its own catalog (built-ins, skills, plugin and project commands)
in the SDK initialization payload; it is cached per working directory because
project commands live in ``<cwd>/.claude/commands``.
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from connector.runtime_protocol import RuntimeCommand

CATALOG_TTL_SECONDS = 600.0
MAX_DESCRIPTION_LENGTH = 200

# Used only until the CLI advertises its own catalog for a working directory. The
# CLI list is authoritative and also covers skills and plugin commands.
FALLBACK_BUILTIN_COMMANDS: tuple[Mapping[str, Any], ...] = (
    {
        "name": "compact",
        "description": "Free up context by summarizing the conversation so far",
        "argumentHint": "<optional custom summarization instructions>",
        "builtin": True,
    },
    {
        "name": "clear",
        "description": (
            "Start a new session with empty context; the previous session stays "
            "on disk"
        ),
        "argumentHint": "[name]",
        "aliases": ["reset", "new"],
        "builtin": True,
    },
    {
        "name": "context",
        "description": "Show current context usage",
        "argumentHint": "",
        "builtin": True,
    },
    {
        "name": "usage",
        "description": "Show session cost and plan usage",
        "argumentHint": "",
        "aliases": ["cost", "stats"],
        "builtin": True,
    },
    {
        "name": "model",
        "description": "Set the AI model for Claude Code",
        "argumentHint": "<model>",
        "builtin": True,
    },
    {
        "name": "init",
        "description": "Initialize a new CLAUDE.md file with codebase documentation",
        "argumentHint": "",
        "builtin": True,
    },
)


def commands_from_init(info: Any) -> tuple[Mapping[str, Any], ...]:
    """Extract the command entries of an SDK initialization payload."""

    if not isinstance(info, Mapping):
        return ()
    raw = info.get("commands")
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return ()
    return tuple(entry for entry in raw if isinstance(entry, Mapping))


def command_roots(
    project_cwd: str | Path | None,
    home: Path | None = None,
) -> tuple[tuple[Path, str], ...]:
    """Command directories to scan, most specific first."""

    roots: list[tuple[Path, str]] = []
    if project_cwd:
        roots.append((Path(project_cwd).expanduser() / ".claude" / "commands", "project"))
    user_home = home if home is not None else Path.home()
    roots.append((user_home / ".claude" / "commands", "user"))

    unique: list[tuple[Path, str]] = []
    seen: set[str] = set()
    for root, source in roots:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append((root, source))
    return tuple(unique)


def commands_from_filesystem(
    project_cwd: str | Path | None,
    home: Path | None = None,
) -> tuple[Mapping[str, Any], ...]:
    """Read custom slash commands from ``.claude/commands`` markdown files."""

    entries: list[Mapping[str, Any]] = []
    for root, source in command_roots(project_cwd, home):
        if not root.is_dir():
            continue
        try:
            paths = sorted(root.rglob("*.md"))
        except OSError:
            continue
        for path in paths:
            name = _command_name(path, root)
            if not name:
                continue
            entries.append(
                {
                    "name": name,
                    "description": _command_description(path),
                    "argumentHint": "",
                    "aliases": [],
                    "builtin": False,
                    "source": source,
                    "path": str(path),
                }
            )
    return tuple(entries)


def merge_command_entries(
    entries: Iterable[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Deduplicate command entries by name, keeping the first occurrence."""

    merged: dict[str, Mapping[str, Any]] = {}
    for entry in entries:
        name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        merged.setdefault(name.strip().lower(), entry)
    return tuple(merged.values())


def normalize_command(raw: Mapping[str, Any]) -> RuntimeCommand | None:
    name = raw.get("name")
    if not isinstance(name, str) or not name.strip():
        return None
    builtin = raw.get("builtin") is True
    hint = raw.get("argumentHint")
    argument_hint = hint.strip() if isinstance(hint, str) else ""
    aliases = tuple(
        dict.fromkeys(
            alias.strip()
            for alias in _string_sequence(raw.get("aliases"))
            if alias.strip()
        )
    )
    description = raw.get("description")
    if isinstance(description, str):
        description = " ".join(description.split())[:MAX_DESCRIPTION_LENGTH] or None
    else:
        description = None
    metadata: dict[str, Any] = {}
    if argument_hint:
        metadata["argumentHint"] = argument_hint
    if builtin:
        metadata["builtin"] = True
    source = raw.get("source")
    if isinstance(source, str) and source:
        metadata["source"] = source
    return RuntimeCommand(
        id=name.strip(),
        title=name.strip(),
        description=description,
        aliases=aliases,
        category="builtin" if builtin else "custom",
        scope="session",
        enabled=True,
        disabled_reason=None,
        # Custom commands may use $ARGUMENTS even without an advertised hint.
        accepts_args=bool(argument_hint) or not builtin,
        args_schema=None,
        metadata=metadata,
    )


def filter_commands(
    commands: tuple[RuntimeCommand, ...],
    *,
    query: str | None = None,
    limit: int = 100,
) -> tuple[RuntimeCommand, ...]:
    filtered = commands
    normalized = query.strip().lower() if isinstance(query, str) else ""
    if normalized:
        filtered = tuple(
            command for command in filtered if _matches_query(command, normalized)
        )
    if isinstance(limit, int) and limit > 0:
        filtered = filtered[:limit]
    return filtered


@dataclass
class ClaudeCommandCatalog:
    """Caches the CLI-advertised catalog and merges it with on-disk commands."""

    ttl_seconds: float = CATALOG_TTL_SECONDS
    home: Path | None = None
    _advertised: dict[str, tuple[float, tuple[Mapping[str, Any], ...]]] = field(
        default_factory=dict,
        init=False,
    )

    def record_server_info(self, cwd: str | None, info: Any) -> None:
        entries = commands_from_init(info)
        if not entries:
            return
        self._advertised[_catalog_key(cwd)] = (time.monotonic(), entries)

    def advertised_entries(self, cwd: str | None) -> tuple[Mapping[str, Any], ...]:
        key = _catalog_key(cwd)
        record = self._advertised.get(key)
        if record is None:
            return ()
        recorded_at, entries = record
        if self.ttl_seconds > 0 and time.monotonic() - recorded_at > self.ttl_seconds:
            self._advertised.pop(key, None)
            return ()
        return entries

    def entries(self, cwd: str | None) -> tuple[Mapping[str, Any], ...]:
        advertised = self.advertised_entries(cwd)
        base = advertised if advertised else FALLBACK_BUILTIN_COMMANDS
        return merge_command_entries(
            (*base, *commands_from_filesystem(cwd, self.home))
        )

    def catalog(
        self,
        *,
        cwd: str | None = None,
        query: str | None = None,
        limit: int = 100,
    ) -> tuple[RuntimeCommand, ...]:
        commands = tuple(
            command
            for command in (normalize_command(entry) for entry in self.entries(cwd))
            if command is not None
        )
        return filter_commands(commands, query=query, limit=limit)

    def find(self, *, cwd: str | None, name: str) -> RuntimeCommand | None:
        wanted = name.strip().lstrip("/").lower()
        if not wanted:
            return None
        commands = self.catalog(cwd=cwd, limit=0)
        for command in commands:
            if command.id.lower() == wanted:
                return command
        for command in commands:
            if wanted in {alias.lower() for alias in command.aliases}:
                return command
        return None


def _catalog_key(cwd: str | None) -> str:
    if not cwd:
        return ""
    try:
        return str(Path(cwd).expanduser().resolve())
    except OSError:
        return str(cwd)


def _matches_query(command: RuntimeCommand, query: str) -> bool:
    return any(
        query in candidate
        for candidate in (
            command.id.lower(),
            command.title.lower(),
            *(alias.lower() for alias in command.aliases),
        )
    )


def _command_name(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return ""
    stem = relative.with_suffix("")
    return ":".join(part for part in stem.parts if part)


def _command_description(path: Path) -> str | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    frontmatter = _frontmatter_value(text, "description")
    if frontmatter:
        return frontmatter
    for line in text.splitlines():
        candidate = line.strip()
        if not candidate or candidate.startswith(("#", "---")):
            continue
        return " ".join(candidate.split())[:MAX_DESCRIPTION_LENGTH]
    return None


def _frontmatter_value(text: str, key: str) -> str | None:
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return None
    for line in lines[1:]:
        stripped = line.strip()
        if stripped == "---":
            return None
        if not stripped.lower().startswith(f"{key}:"):
            continue
        value = stripped[len(key) + 1 :].strip().strip("\"'")
        return " ".join(value.split())[:MAX_DESCRIPTION_LENGTH] or None
    return None


def _string_sequence(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(item for item in value if isinstance(item, str))
    if isinstance(value, str):
        return (value,)
    return ()