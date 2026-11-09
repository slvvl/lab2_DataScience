import pytest
import os
from grader_contracts.pandas_tasks import TitanicInput, TitanicSummary
from grader_contracts.correlation_tasks import BrainDataInput, BrainCorrelationSummary

from pandas_tasks import analyze_titanic
from correlation_tasks import analyze_brain_correlations

# Пути к файлам данных 
TITANIC_PATH = "titanic.csv"
BRAIN_PATH = "brainsize.txt"

def test_analyze_titanic():
    data = TitanicInput(csv_path=TITANIC_PATH)
    summary = analyze_titanic(data)
    
    assert isinstance(summary, TitanicSummary)
    
    # Проверка граничных случаев и типов
    assert summary.row_count > 0
    assert isinstance(summary.missing_by_column, dict)
    assert "Age" in summary.missing_by_column
    assert "Cabin" in summary.missing_by_column
    
    assert summary.adults_over_30_count > 0
    assert len(summary.mean_age_by_pclass) == 3
    assert list(summary.mean_age_by_pclass.keys()) == [1, 2, 3]
    
    assert len(summary.survival_rate_by_pclass) == 3
    assert list(summary.survival_rate_by_pclass.keys()) == [1, 2, 3]
    
    # Проверка, что тарифы отсортированы по убыванию
    assert len(summary.highest_fares) == 5
    assert summary.highest_fares == sorted(summary.highest_fares, reverse=True)

def test_analyze_brain_correlations():
    data = BrainDataInput(csv_path=BRAIN_PATH)
    summary = analyze_brain_correlations(data)
    
    assert isinstance(summary, BrainCorrelationSummary)
    
    # Проверка количества мужчин и женщин
    assert summary.men_count > 0
    assert summary.women_count > 0
    
    # Проверка корреляций
    features = ['FSIQ', 'VIQ', 'PIQ', 'Weight', 'Height']
    for f in features:
        assert f in summary.women_mri_correlation
        assert f in summary.men_mri_correlation
        
    # Проверка strongest_mri_feature
    assert summary.strongest_mri_feature in features
    assert summary.strongest_mri_feature != 'MRI_Count'