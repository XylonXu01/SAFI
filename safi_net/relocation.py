"""Resolve historical workspace paths without rewriting provenance records."""
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_root():
    return PROJECT_ROOT


def resolve_path(value):
    path = Path(value).expanduser()
    marker = PROJECT_ROOT / 'artifacts/workspace_migration/manifest.json'
    if not marker.exists():
        return path
    record = json.loads(marker.read_text())
    if path.is_absolute():
        relative = None
        for old in (Path(record['old_root']), Path(record['new_root'])):
            try:
                relative = path.relative_to(old).as_posix()
                break
            except ValueError:
                continue
        if relative is None:
            return path
    else:
        # Relative paths from old manifests are mapped only when explicitly
        # passed to this resolver, never by globally changing process cwd.
        relative = path.as_posix()
    for source, target in sorted(record['moves'], key=lambda row: -len(row[0])):
        if relative == source or relative.startswith(source + '/'):
            return PROJECT_ROOT / (target + relative[len(source):])
    return PROJECT_ROOT / relative if path.is_absolute() else path


def resolved_values(value):
    """A runtime view of absolute paths; do not hash this instead of raw records."""
    if isinstance(value, dict):
        return {key: resolved_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolved_values(item) for item in value]
    if isinstance(value, str) and value.startswith('/'):
        return str(resolve_path(value))
    return value
