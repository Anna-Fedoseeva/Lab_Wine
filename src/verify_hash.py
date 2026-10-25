import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_FILE = ROOT / "reports" / "LAB1" / "hash_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    failed = False

    for relative_path, details in manifest["files"].items():
        path = ROOT / relative_path
        if not path.exists():
            print(f"MISSING: {relative_path}")
            failed = True
            continue

        actual = sha256_file(path)
        expected = details["sha256"]

        if actual == expected:
            print(f"OK: {relative_path}")
        else:
            print(f"MISMATCH: {relative_path}")
            failed = True

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
