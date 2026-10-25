from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "wine.csv"
REPORT_DIR = ROOT / "reports" / "LAB1"
FIGURE_DIR = REPORT_DIR / "figures"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_FILE)
    features = df.drop(columns=["target"])
    target = df["target"]

    # Общая статистика признаков
    stats = features.describe().T
    stats["missing"] = features.isna().sum()
    stats["n_unique"] = features.nunique()
    stats["target_correlation"] = df.corr(numeric_only=True)["target"].drop("target")
    stats.to_csv(REPORT_DIR / "eda_stats.csv", index_label="feature")

    # Распределение классов
    class_counts = target.value_counts().sort_index()
    plt.figure(figsize=(7, 4))
    sns.barplot(x=class_counts.index.astype(str), y=class_counts.values)
    plt.xlabel("Класс вина")
    plt.ylabel("Количество образцов")
    plt.title("Распределение целевых классов")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # Масштабы признаков: среднее и стандартное отклонение
    scale_stats = features.agg(["mean", "std"]).T.sort_values("std")
    plt.figure(figsize=(10, 7))
    sns.barplot(x=scale_stats["std"], y=scale_stats.index)
    plt.xlabel("Стандартное отклонение")
    plt.ylabel("Признак")
    plt.title("Разброс значений признаков Wine")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_scales.png", dpi=150)
    plt.close()

    # Корреляционная матрица
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(numeric_only=True), cmap="coolwarm", center=0)
    plt.title("Корреляции признаков и целевого класса")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    class_share = target.value_counts(normalize=True).sort_index() * 100
    scale_ratio = features.std().max() / features.std().replace(0, float("nan")).min()

    report = f"""# Лабораторная работа 1 — EDA

## Датасет
- Источник: `sklearn.datasets.load_wine`
- Число строк: {df.shape[0]}
- Число столбцов вместе с target: {df.shape[1]}
- Число признаков: {features.shape[1]}
- Пропущенных значений: {int(df.isna().sum().sum())}

## Доли классов
{class_share.to_string(float_format=lambda x: f"{x:.2f}%")}

## Выводы
1. Датасет содержит {df.shape[0]} наблюдений и {features.shape[1]} числовых признаков.
2. Число пропущенных значений в таблице: {int(df.isna().sum().sum())}.
3. Стандартные отклонения признаков различаются: отношение максимального к минимальному ненулевому стандартному отклонению составляет {scale_ratio:.2f}. Это подтверждает необходимость исследовать масштабирование признаков перед применением методов, чувствительных к масштабу, например kNN и SVM.
4. Распределение классов показано на графике `figures/class_distribution.png`.
5. Разброс признаков показан на графике `figures/feature_scales.png`; корреляции — на `figures/correlation_heatmap.png`.
"""
    (REPORT_DIR / "lab1_report.md").write_text(report, encoding="utf-8")

    print(f"Данные: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Пропущено значений: {int(df.isna().sum().sum())}")
    print(f"Отчёт: {REPORT_DIR / 'lab1_report.md'}")
    print(f"Статистика: {REPORT_DIR / 'eda_stats.csv'}")
    print(f"Графики: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
