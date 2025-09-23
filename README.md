# Avito Search Reranker

Решение тестового задания для стажировки Авито: построение модели **реранкинга объявлений в поиске**.  
Цель — обучить модель, которая внутри каждого запроса (`query_id`) упорядочивает объявления (`item_id`) по релевантности.

---

## 📂 Структура проекта

```

├── data/                # Данные
│   ├── raw/             # Сырые данные (.parquet из задания)
│   ├── transformed/     # Промежуточные/обработанные данные
│   └── solution.csv     # Финальный сабмит
│
├── models/              # Сохранённые модели
│   └── lgbm_ranker.txt  # Финальная LightGBM Ranker
│
├── notebooks/           # Основные ноутбуки
│   ├── eda.ipynb        # Рразведочный анализ
│   ├── features.ipynb   # Генерация признаков
│   └── train_lgbm.ipynb # Обучение LightGBM Ranker + валидация
│
├── scripts/
│   └── cluster.py       # Запуск локального Dask-кластера
│
├── pyproject.toml       # Зависимости проекта
└── README.md            # Документация

```

---

## ⚙️ Как запускать

1. **Подготовка окружения**

    ```bash
    uv sync
    ```

2. **Запуск локального Dask-кластера**
   Перед открытием ноутбуков необходимо запустить кластер:

    ```bash
    uv run scripts/cluster.py
    ```

    > Кластер нужен для работы с большими `.parquet` (train \~3.5GB).
    > Внутри ноутбуков используется клиент `Client()` для подключения.

3. **Порядок работы**

    - `notebooks/eda.ipynb`
      Первичный анализ данных, баланс таргета, распределения.
    - `notebooks/features.ipynb`
      Генерация базовых фичей (цена, CTR, длины текстов, match-признаки).
      Результат сохраняется в `data/transformed/`.
    - `notebooks/train_lgbm.ipynb`
      Обучение LightGBM Ranker (objective=lambdarank),
      валидация через `GroupKFold(query_id)`, метрика `NDCG@10`.
      Сохраняет модель (`models/lgbm_ranker.txt`) и сабмит (`data/solution.csv`).
