from pathlib import Path
import zipfile

from fpeval.archive import archive_directory


def test_archive_directory_keeps_root_and_excludes_cache(tmp_path: Path):
    output = tmp_path / "anomalyclip"
    (output / "numerical").mkdir(parents=True)
    (output / "numerical" / "summary.csv").write_text("metric\n1\n")
    (output / "extracted_attacks").mkdir()
    (output / "extracted_attacks" / "delta.pt").write_bytes(b"cache")

    archive = archive_directory(output, exclude_top_level=("extracted_attacks",))

    with zipfile.ZipFile(archive) as package:
        assert package.namelist() == ["anomalyclip/numerical/summary.csv"]
