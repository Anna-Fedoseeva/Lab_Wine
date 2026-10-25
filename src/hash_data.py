import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "wine.csv"
MANIFEST_FILE = ROOT / "reports" / "LAB1" / "hash_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Не найден файл данных: {DATA_FILE}")

    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": {
            str(DATA_FILE.relative_to(ROOT)).replace("\\", "/"): {
                "sha256": sha256_file(DATA_FILE),
                "size_bytes": DATA_FILE.stat().st_size,
            }
        },
    }

    MANIFEST_FILE.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"SHA-256: {manifest['files']['data/raw/wine.csv']['sha256']}")
    print(f"Манифест сохранён: {MANIFEST_FILE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
