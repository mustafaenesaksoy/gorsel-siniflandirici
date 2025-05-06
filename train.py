import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Veri klasörleri
FRUITS_DIR = 'data/fruits/Training'
ANIMALS_DIR = 'data/animals-10'

# Veri boyutu
IMG_SIZE = (100, 100)
BATCH_SIZE = 32

# ImageDataGenerator ile veri artırma
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2  # %80 eğitim, %20 doğrulama
)

# Eğitim ve doğrulama verisini yükle
train_generator = datagen.flow_from_directory(
    directory='data/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = datagen.flow_from_directory(
    directory='data/',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Sınıf sayısını al
num_classes = len(train_generator.class_indices)
print("Sınıflar:", train_generator.class_indices)

# Model mimarisi (basit CNN)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(*IMG_SIZE, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Modeli eğit
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# Modeli kaydet
model.save('model/my_model.h5')

# Eğitim grafiği
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()
plt.title('Model Doğruluk Grafiği')
plt.savefig('model/accuracy.png')
plt.show()
