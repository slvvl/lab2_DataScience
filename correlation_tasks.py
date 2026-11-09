"""Задачи второй части лабораторной: корреляционный анализ."""
from __future__ import annotations

import pandas as pd

from grader_contracts.correlation_tasks import BrainCorrelationSummary, BrainDataInput


def analyze_brain_correlations(data: BrainDataInput) -> BrainCorrelationSummary:
    """Проанализируйте brainsize.txt.

    Разделите наблюдения по полу и для каждой группы вычислите корреляции
    признаков FSIQ, VIQ, PIQ, Weight, Height с MRI_Count методом Пирсона.
    В strongest_mri_feature верните название признака с наибольшим модулем
    корреляции с MRI_Count среди объединённых результатов двух групп.
    """
    # 1. Загрузка данных (разделитель tab, NA как пропуски)
    df = pd.read_csv(data.csv_path, sep='\t', na_values='NA')
    
    # 2. Разделение на группы
    men = df[df['Gender'] == 'Male']
    women = df[df['Gender'] == 'Female']
    
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    
    # 3. Вычисление корреляций
    women_corr = women[features + ['MRI_Count']].corr()['MRI_Count'].drop('MRI_Count').to_dict()
    men_corr = men[features + ['MRI_Count']].corr()['MRI_Count'].drop('MRI_Count').to_dict()
    
    # 4. Поиск признака с максимальным модулем корреляции
    best_feature = None
    max_abs_corr = -1.0
    
    for feature in features:
        current_max = max(abs(women_corr[feature]), abs(men_corr[feature]))
        if current_max > max_abs_corr:
            max_abs_corr = current_max
            best_feature = feature
            
    return BrainCorrelationSummary(
        men_count=len(men),
        women_count=len(women),
        women_mri_correlation=women_corr,
        men_mri_correlation=men_corr,
        strongest_mri_feature=best_feature
    )
