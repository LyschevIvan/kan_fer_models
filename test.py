import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# Импортируем из пакета kan_fer вместо src.models

from kan_fer import KANFER2013, KANRAFDB, KALFER2013, KALRAFDB

def test_with_model(model, img, ax, title):
    """Тестирование модели и вывод результатов"""
    # Предсказание эмоций
    predictions = model.predict(img)
    sorted_pred = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
    
    # Вывод в консоль
    print(f"\nМодель {title}:")
    for emotion, prob in sorted_pred:
        print(f"{emotion}: {prob:.4f}")
    
    # Визуализация
    emotions = [e for e, _ in sorted_pred]
    probs = [p for _, p in sorted_pred]
    colors = plt.cm.viridis(np.linspace(0, 1, len(emotions)))
    
    ax.barh(emotions, probs, color=colors)
    ax.set_xlim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("Вероятность")
    
    return sorted_pred[0]  # Возвращаем наиболее вероятную эмоцию

def main():
    # Загрузка изображения
    img_path = "test_image.jpg"
    img = Image.open(img_path)
    
    # Создание моделей
    models = {
        "KAN-FER2013": KANFER2013(),
        "KAN-RAF-DB": KANRAFDB(),
        "KAL-FER2013": KALFER2013(),
        "KAL-RAF-DB": KALRAFDB()
    }
    
    # Создание графика
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    # Показываем изображение
    axes[0].imshow(img)
    axes[0].set_title("Исходное изображение")
    axes[0].axis("off")
    
    # Тестирование каждой модели
    top_emotions = {}
    for i, (name, model) in enumerate(models.items(), 1):
        emotion, prob = test_with_model(model, img, axes[i], name)
        top_emotions[name] = f"{emotion} ({prob:.2f})"
    
    # Добавляем на изображение топовые эмоции от каждой модели
    y_pos = 0.05
    for name, emotion_info in top_emotions.items():
        axes[0].text(0.05, y_pos, f"{name}: {emotion_info}", 
                     transform=axes[0].transAxes, color='white', 
                     bbox=dict(facecolor='black', alpha=0.7))
        y_pos += 0.07
    
    plt.tight_layout()
    plt.savefig("result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main() 