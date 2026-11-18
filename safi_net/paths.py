from dataclasses import dataclass
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class ResourcePaths:
    data_root: Path
    pretrained_root: Path
    run_dir: Path


def resolve_paths(data_root, pretrained_root=None, run_dir=None):
    from .relocation import resolve_path
    if data_root is None:
        raise ValueError('data_root is required')
    return ResourcePaths(resolve_path(data_root).resolve(),
                         resolve_path(pretrained_root if pretrained_root is not None else PACKAGE_ROOT / 'pretrained').resolve(),
                         resolve_path(run_dir if run_dir is not None else Path.cwd() / 'runs').resolve())


def relative_path(root, value):
    root, relative = Path(root).resolve(), Path(value)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Manifest paths must be relative and cannot traverse parents')
    result = (root / relative).resolve()
    if not result.is_relative_to(root):
        raise ValueError('Manifest path escapes root')
    return result
