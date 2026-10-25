from pathlib import Path

from sklearn.datasets import load_wine

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
RAW_FILE = RAW_DIR / "wine.csv"


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_wine(as_frame=True)
    df = dataset.frame.copy()
    df.to_csv(RAW_FILE, index=False)

    print("Источник: sklearn.datasets.load_wine")
    print(f"Файл: {RAW_FILE.relative_to(ROOT)}")
    print(f"Размер: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Классы {sorted(df['target'].unique().tolist())}")


if __name__ == "__main__":
    main()
