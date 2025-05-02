import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Параметры
IMG_SIZE = (150, 150)
BATCH_SIZE = 32
EPOCHS = 250
TRAIN_DIR = r"C:\\Project\\train"
VALID_DIR = r"C:\\Project\\valid"
TEST_DIR = r"C:\\Project\\test"
CLASS_MAP_JSON = r"C:\\Project\\cat_to_name.json"
MODEL_PATH = "best_flower_model.h5"

# Загрузка отображения классов
with open(CLASS_MAP_JSON, "r") as f:
    class_names = json.load(f)

# Аугментация данных
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# Генераторы
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
    VALID_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical')

# Модель на базе MobileNetV2
base_model = MobileNetV2(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights='imagenet')
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint(MODEL_PATH, save_best_only=True)
]

# Обучение
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=callbacks
)

# Загрузка лучшей модели (на всякий случай)
model = load_model(MODEL_PATH)

# Функция предсказания
def predict_flower(image_path):
    img = load_img(image_path, target_size=IMG_SIZE)
    img_array = img_to_array(img)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_id = np.argmax(prediction)
    flower_name = class_names.get(str(class_id), "Unknown")
    return flower_name

# Предсказания для всех файлов в TEST_DIR
results = []
print("\nРезультаты на тестовых изображениях:")
for filename in os.listdir(TEST_DIR):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(TEST_DIR, filename)
        flower = predict_flower(image_path)
        result = f"{filename} -> {flower}"
        results.append(result)
        print(result)

with open("test_predictions.txt", "w", encoding="utf-8") as f:
    for line in results:
        f.write(line + "\n")

# Визуализация истории обучения
plt.figure(figsize=(14, 5))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Точность на обучении')
plt.plot(history.history['val_accuracy'], label='Точность на валидации')
plt.title('Точность по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Точность')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Потери на обучении')
plt.plot(history.history['val_loss'], label='Потери на валидации')
plt.title('Потери по эпохам')
plt.xlabel('Эпоха')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig("training_history.png")  # Сохраняем график в файл
plt.show()

