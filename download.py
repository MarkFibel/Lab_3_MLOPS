import pandas as pd
import numpy as np


def clear_data(path2df):
    # Загружаем датасет из CSV-файла
    data = pd.read_csv(path2df)
    
    # Удаляем ненужные колонки 'PassengerId' и 'Name', так как они не несут полезной информации для анализа
    data = data.drop(columns=['PassengerId', 'Name'])
    
    # Кодируем категориальные (текстовые) переменные в числовые значения
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = pd.factorize(data[col])[0]
        
    # Заполняем пропущенные значения в колонке 'Age' случайными числами в диапазоне (среднее - стандартное отклонение, среднее + стандартное отклонение)
    mean = data['Age'].mean()
    std = data['Age'].std()
    data['Age'] = data['Age'].apply(lambda x: np.random.randint(mean - std, mean + std) if pd.isna(x) else x)
    
    # Сохраняем очищенные данные в новый CSV-файл без индексов
    data.to_csv("data/processed_data.csv", index=False)
    
    return True  # Возвращаем True, чтобы обозначить успешное выполнение функции


clear_data("data/train.csv")