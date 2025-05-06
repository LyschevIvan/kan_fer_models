# KAN-FER: Модели распознавания эмоций на основе Kolmogorov-Arnold Networks

Этот проект содержит реализацию моделей для распознавания эмоций на изображениях лиц с использованием архитектур Kolmogorov-Arnold Networks (KAN) и их вариаций.

## Описание

Проект предлагает набор предобученных моделей для распознавания эмоций на основе архитектур KAN и KAL (Kolmogorov-Arnold Legendre Network). Модели обучены на популярных наборах данных эмоций:
- FER2013
- RAF-DB

## Установка

### Вариант 1: Установка как библиотеки

Вы можете установить этот проект как Python-пакет:

```bash
# Установка напрямую из репозитория (рекомендуется)
pip install git+https://github.com/LyschevIvan/kan_fer_models.git

# ИЛИ установка локально
git clone https://github.com/LyschevIvan/kan_fer_models.git
cd kan_fer_models
pip install -e .
```

После установки вы можете импортировать и использовать модели так:

```python
from kan_fer import KANFER2013, KANRAFDB, KALFER2013, KALRAFDB
```

### Вариант 2: Использование как отдельного проекта

```bash
# Установка зависимостей
pip install torch torchvision matplotlib numpy pillow

# Клонирование репозитория
git clone https://github.com/LyschevIvan/kan_fer_models.git
cd kan_fer_models

# Установка efficient-kan зависимости
python -m pip install git+https://github.com/Blealtan/efficient-kan.git

# Установка torchkan
git clone https://github.com/1ssb/torchkan.git

# Установка проекта локально
pip install -e .
```

## Структура проекта

- `kan_fer/` - пакет библиотеки
  - `models.py` - содержит реализации моделей
  - `pretrained/` - предобученные модели
- `src/` - исходный код (для прямого использования)
- `test.py` - скрипт для тестирования моделей на изображении

## Использование

Проект позволяет распознавать эмоции на изображениях лиц с помощью различных моделей:

```python
from kan_fer import KANFER2013, KANRAFDB, KALFER2013, KALRAFDB
from PIL import Image

# Загрузка изображения
img = Image.open("путь_к_изображению.jpg")

# Инициализация модели
model = KANFER2013()

# Получение предсказаний
predictions = model.predict(img)
```

### Тестовый скрипт

Файл `test.py` предоставляет функциональность для тестирования всех реализованных моделей на одном изображении:

1. Загружает изображение из файла `test_image.jpg`
2. Запускает предсказание с использованием всех четырех моделей (KAN-FER2013, KAN-RAF-DB, KAL-FER2013, KAL-RAF-DB)
3. Визуализирует результаты в виде гистограмм вероятностей эмоций
4. Отображает исходное изображение с указанием наиболее вероятной эмоции от каждой модели
5. Сохраняет результат в файл `result.png`

Для запуска тестового скрипта:

```bash
python test.py
```

## Особенности моделей

- **KAN-FER2013**: Модель на основе Kolmogorov-Arnold Network, обученная на датасете FER2013
- **KAN-RAF-DB**: Модель на основе Kolmogorov-Arnold Network, обученная на датасете RAF-DB
- **KAL-FER2013**: Модель на основе Kolmogorov-Arnold Legendre Network, обученная на датасете FER2013
- **KAL-RAF-DB**: Модель на основе Kolmogorov-Arnold Legendre Network, обученная на датасете RAF-DB

## Пример результата

При запуске `test.py` создается визуализация, которая показывает:
- Исходное изображение
- Гистограммы вероятностей для всех моделей
- Итоговые предсказания на исходном изображении

<img src="result.png">

## Ссылки на используемые библиотеки

- [TorchKAN](https://github.com/1ssb/torchkan) - Реализация KAN на PyTorch
- [Efficient-KAN](https://github.com/Blealtan/efficient-kan) - Эффективная реализация KAN
