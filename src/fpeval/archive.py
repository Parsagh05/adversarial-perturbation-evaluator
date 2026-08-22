"""Portable archives for complete evaluator output folders."""

from __future__ import annotations

from pathlib import Path
import zipfile


def archive_directory(
    directory: Path, *, exclude_top_level: tuple[str, ...] = ()
) -> Path:
    """Atomically create ``<directory>.zip``, retaining the root folder name."""
    directory = directory.resolve()
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    destination = directory.with_suffix(".zip")
    temporary = destination.with_suffix(".zip.tmp")
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(
        temporary, "w", compression=zipfile.ZIP_DEFLATED, allowZip64=True
    ) as package:
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            relative = path.relative_to(directory)
            if relative.parts and relative.parts[0] in exclude_top_level:
                continue
            package.write(path, Path(directory.name) / relative)
    temporary.replace(destination)
    return destination
