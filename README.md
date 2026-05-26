# Career Path Prediction

Предсказание карьерных траекторий IT-специалистов по резюме (российский рынок).

## Задачи

| Задача | Target | История |
|--------|--------|---------|
| **Task A** | `title_norm` (текущая должность) | job_3 → job_2 → job_1 |
| **Task B** | `job_1_position_norm` (следующая должность) | job_3 → job_2 |

## Результаты (test)

| Модель | Task A | Task B |
|--------|--------|--------|
| Majority baseline | 26.3% MRR | 24.0% MRR |
| Inertia baseline | 51.5% MRR | 49.8% MRR |
| Bigram baseline | 55.5% MRR | 57.9% MRR |
| Linear (titles) | 65.9% MRR | 62.2% MRR |
| Linear (desc) | 70.7% MRR | 65.3% MRR |
| **MLP KFold (Additive)** | **75.5% MRR** | **68.8% MRR** |
## Архитектура

1. **Энкодер**: `intfloat/multilingual-e5-large`, fine-tuned (LAST стратегия)
   - R1: MNRL, 3 эпохи
   - R2: hard negatives, 1 эпоха
   - Class-aware batching (без дублей label в батче)

2. **MLP**: Additive / Gated fusion
   - 3 слоя (256 → 256 → 128), BatchNorm, Dropout
   - Label smoothing = 0.1
   - Temperature = log(1/√VOCAB), clamp [0.01, 1.0]


## Признаки

| Блок | Размерность | Описание |
|------|-------------|----------|
| Titles (X_seq) | 1024 | Только нормализованные тайтлы из истории |
| Desc (X_doc) | 1024 | Тайтлы + описания (500 символов) |
| Skills | 1024 | Эмбеддинги поля `skills` |
| About | 1024 | Эмбеддинги поля `about` (500 символов) |
| Career | 5 | Стаж, суммарная длительность, tenure |
| Location | 3 | Москва / Питер / другое (one-hot) |
| Edu | 1 | Уровень образования (нормализованный) |

## Ключевые выводы

1. **Описания должностей** дают +4.7pp к MRR (Task A) и +2.3pp (Task B) — *p < 0.001*
2. **Skills** даёт +2.4pp поверх описаний (Task A) — *p < 0.001*
3. **Структурные признаки** дают скромный вклад (+0.4pp, p < 0.01)

## Зависимости

- Python 3.8+
- PyTorch 2.4+
- sentence-transformers 2.7+
- scikit-learn, pandas, numpy, lightgbm

## Запуск

Открыть ноутбук в Jupyter, выполнить последовательно все ячейки. 

## Датасет

`resumes_merged_filtered.csv` — резюме IT-специалистов (111k строк).  
Доступ к данным — по запросу.
