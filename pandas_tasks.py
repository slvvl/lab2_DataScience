"""Задачи первой части лабораторной: Pandas и Titanic."""
from __future__ import annotations

import pandas as pd

from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary


def analyze_titanic(data: TitanicInput) -> TitanicSummary:
    """Выполните загрузку и анализ датасета Titanic.

    Нужно: посчитать пропуски, число пассажиров старше 30 лет, средний возраст
    и долю выживших по классам, а также пять наибольших тарифов по убыванию.
    """
    # 1. Загружаем CSV
    df = pd.read_csv(data.csv_path)
    
    # 2. Количество пропусков в каждом столбце (в виде обычного словаря)
    missing_by_column = df.isna().sum().to_dict()
    
    # 3. Число пассажиров старше 30 лет
    adults_over_30_count = int((df['Age'] > 30).sum())
    
    # 4. Средний возраст для каждого Pclass
    mean_age_by_pclass = df.groupby('Pclass')['Age'].mean().to_dict()
    mean_age_by_pclass = {int(k): float(v) for k, v in mean_age_by_pclass.items()}
    
    # 5. Доля выживших для каждого Pclass
    survival_rate_by_pclass = df.groupby('Pclass')['Survived'].mean().to_dict()
    survival_rate_by_pclass = {int(k): float(v) for k, v in survival_rate_by_pclass.items()}
    
    # 6. Пять наибольших значений Fare в порядке убывания
    highest_fares = df['Fare'].nlargest(5).tolist()
    
    return TitanicSummary(
        row_count=len(df),
        missing_by_column=missing_by_column,
        adults_over_30_count=adults_over_30_count,
        mean_age_by_pclass=mean_age_by_pclass,
        survival_rate_by_pclass=survival_rate_by_pclass,
        highest_fares=highest_fares
    )