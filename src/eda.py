from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "raw" / "wine.csv"
REPORT_DIR = ROOT / "reports" / "Lab1"
FIGURE_DIR = REPORT_DIR / "figures"


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Не найден файл датасета: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    if "target" not in df.columns:
        raise ValueError("В датасете отсутствует столбец target.")

    features = df.drop(columns="target")
    target = df["target"]
    numeric_features = features.select_dtypes(include="number")

    if numeric_features.empty:
        raise ValueError("В датасете нет числовых признаков для анализа.")

    # 1. Общая статистика признаков
    stats = numeric_features.describe().T
    stats["missing"] = numeric_features.isna().sum()
    stats["n_unique"] = numeric_features.nunique()
    stats["target_correlation"] = numeric_features.corrwith(target)
    stats.to_csv(REPORT_DIR / "eda_stats.csv", index_label="feature")

    missing_total = int(df.isna().sum().sum())
    duplicate_rows = int(df.duplicated().sum())
    class_counts = target.value_counts().sort_index()
    class_share = target.value_counts(normalize=True).sort_index() * 100

    # 2. Распределение классов
    plt.figure(figsize=(7, 4))
    ax = sns.barplot(
        x=class_counts.index.astype(str),
        y=class_counts.values,
    )
    ax.bar_label(ax.containers[0], fmt="%.0f")
    plt.xlabel("Класс вина")
    plt.ylabel("Количество образцов")
    plt.title("Распределение классов Wine")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "class_distribution.png", dpi=150)
    plt.close()

    # 3. Масштабы признаков
    scale_stats = numeric_features.agg(["mean", "std"]).T
    scale_stats = scale_stats.sort_values("std")
    scale_stats.to_csv(REPORT_DIR / "feature_scale_stats.csv", index_label="feature")

    plt.figure(figsize=(10, 7))
    ax = sns.barplot(
        x=scale_stats["std"],
        y=scale_stats.index,
    )
    ax.bar_label(ax.containers[0], fmt="%.2f", padding=3, fontsize=8)
    plt.xlabel("Стандартное отклонение")
    plt.ylabel("Признак")
    plt.title("Масштабы признаков Wine")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "feature_scales.png", dpi=150)
    plt.close()

    # 4. Корреляции
    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 9))
    sns.heatmap(
        corr,
        cmap="coolwarm",
        center=0,
        vmin=-1,
        vmax=1,
    )
    plt.title("Корреляционная матрица")
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()

    # Сильнейшие корреляции между разными признаками
    feature_corr = numeric_features.corr()
    upper_mask = pd.DataFrame(
        True,
        index=feature_corr.index,
        columns=feature_corr.columns,
    )
    for i, first in enumerate(feature_corr.columns):
        for second in feature_corr.columns[: i + 1]:
            upper_mask.loc[first, second] = False

    pairs = feature_corr.where(upper_mask).stack().rename("correlation").reset_index()
    pairs.columns = ["feature_1", "feature_2", "correlation"]
    pairs["abs_correlation"] = pairs["correlation"].abs()
    pairs = pairs.sort_values("abs_correlation", ascending=False)
    pairs.head(10).to_csv(
        REPORT_DIR / "top_correlations.csv",
        index=False,
    )

    # 5. Числа для выводов
    std = numeric_features.std()
    nonzero_std = std[std > 0]
    scale_ratio = (
        nonzero_std.max() / nonzero_std.min() if not nonzero_std.empty else float("nan")
    )

    target_corr = numeric_features.corrwith(target).dropna()
    strongest_target_feature = target_corr.abs().idxmax()
    strongest_target_corr = target_corr[strongest_target_feature]

    strongest_pair = pairs.iloc[0] if not pairs.empty else None

    class_table = pd.DataFrame(
        {
            "Количество": class_counts,
            "Доля, %": class_share.round(2),
        }
    )
    class_table.index.name = "Класс"
    class_table_md = (
        class_table.to_markdown()
        if False
        else (
            "| Класс | Количество | Доля, % |\n"
            "|---:|---:|---:|\n"
            + "\n".join(
                f"| {cls} | {int(class_counts.loc[cls])} | {class_share.loc[cls]:.2f} |"
                for cls in class_counts.index
            )
        )
    )

    missing_by_column = df.isna().sum()
    columns_with_missing = missing_by_column[missing_by_column > 0]
    if columns_with_missing.empty:
        missing_details = "Пропусков нет."
    else:
        missing_details = "\n".join(
            f"- `{col}`: {int(count)}" for col, count in columns_with_missing.items()
        )

    if strongest_pair is not None:
        pair_text = (
            f"`{strongest_pair['feature_1']}` и "
            f"`{strongest_pair['feature_2']}` "
            f"(\u03c1 = {strongest_pair['correlation']:.3f})"
        )
    else:
        pair_text = "Недостаточно данных для расчёта."

    report = f"""# Лабораторная работа № 1 — EDA

**Вариант:** 02 — Wine
**Дата анализа:** {date.today().isoformat()}
**Источник:** `sklearn.datasets.load_wine`
**Файл:** `data/raw/wine.csv`

## 1. Паспорт данных

| Показатель | Значение |
|---|---:|
| Наблюдения | {len(df)} |
| Всего столбцов | {df.shape[1]} |
| Признаки без `target` | {features.shape[1]} |
| Числовые признаки | {numeric_features.shape[1]} |
| Целевых классов | {target.nunique()} |
| Пропущенные значения | {missing_total} |
| Полные дубликаты строк | {duplicate_rows} |

## 2. Качество данных

**Пропуски по столбцам:** {missing_details}

Полные дубликаты учитываются по всем столбцам, включая `target`.
Выбросы автоматически не удалялись: их наличие само по себе не означает ошибку данных.

## 3. Распределение классов

{class_table_md}

![Распределение классов](figures/class_distribution.png)

## 4. Масштабы признаков

Отношение максимального стандартного отклонения к минимальному ненулевому: **{scale_ratio:.2f}**.

Признак с наибольшим стандартным отклонением: **{std.idxmax()}** ({std.max():.3f}).

Признак с наименьшим ненулевым стандартным отклонением: **{nonzero_std.idxmin() if not nonzero_std.empty else "нет"}** ({nonzero_std.min():.3f}).

![Масштабы признаков](figures/feature_scales.png)

## 5. Связи между признаками

Наибольшая по модулю корреляция между парами признаков: {pair_text}.

Наибольшая по модулю корреляция числового признака с `target`:
**{strongest_target_feature}** (коэффициент корреляции Пирсона = {strongest_target_corr:.3f}).

Корреляция не доказывает причинно-следственную связь. Числовой код `target` также не следует трактовать как непрерывную величину.

![Корреляционная матрица](figures/correlation_heatmap.png)

## 6. Выводы

1. В наборе данных **{len(df)} наблюдений** и **{numeric_features.shape[1]} числовых признаков**.
2. Обнаружено **{missing_total} пропущенных значений**; число полных дубликатов — **{duplicate_rows}**.
3. Отношение стандартных отклонений равно **{scale_ratio:.2f}**, поэтому перед использованием kNN и SVM следует проверить стандартизацию признаков.
4. Наибольшая корреляция с числовым кодом целевого класса у признака **{strongest_target_feature}**: **{strongest_target_corr:.3f}**; это описательная связь, а не доказательство причинности.
5. Наиболее сильно связанные между собой признаки: {pair_text}. Полный список сохранён в `top_correlations.csv`.

## 7. Артефакты и воспроизводимость

- `eda_stats.csv` — описательная статистика и корреляция с `target`.
- `feature_scale_stats.csv` — средние и стандартные отклонения признаков.
- `top_correlations.csv` — десять наиболее сильных корреляций между признаками.
- `figures/class_distribution.png` — распределение классов.
- `figures/feature_scales.png` — масштабы признаков.
- `figures/correlation_heatmap.png` — корреляционная матрица.

Для повторного построения отчёта запустите `python src/eda.py` из корня проекта.
"""

    (REPORT_DIR / "lab1_report.md").write_text(report, encoding="utf-8")

    print(f"Данные: {df.shape[0]} строк, {df.shape[1]} столбцов")
    print(f"Пропущено значений: {missing_total}")
    print(f"Отчёт: {REPORT_DIR / 'lab1_report.md'}")
    print(f"Статистика: {REPORT_DIR / 'eda_stats.csv'}")
    print(f"Графики: {FIGURE_DIR}")


if __name__ == "__main__":
    main()
